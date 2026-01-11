"""SendGrid email notifier plugin."""

from __future__ import annotations

import asyncio
import html
import logging
import os
from collections import defaultdict

import aiohttp

from homesec.interfaces import Notifier
from homesec.models.alert import Alert
from homesec.models.config import SendGridEmailConfig
from homesec.models.vlm import SequenceAnalysis

logger = logging.getLogger(__name__)


class SendGridEmailNotifier(Notifier):
    """SendGrid email notifier for HomeSec alerts."""

    def __init__(self, config: SendGridEmailConfig) -> None:
        self._api_key_env = config.api_key_env
        self._from_email = config.from_email
        self._from_name = config.from_name
        self._to_emails = list(config.to_emails)
        self._cc_emails = list(config.cc_emails)
        self._bcc_emails = list(config.bcc_emails)
        self._subject_template = config.subject_template
        self._text_template = config.text_template
        self._html_template = config.html_template
        self._timeout_s = float(config.request_timeout_s)
        self._api_base = config.api_base.rstrip("/")

        self._api_key = os.getenv(self._api_key_env)
        if not self._api_key:
            logger.warning("SendGrid API key not found in env: %s", self._api_key_env)

        self._session: aiohttp.ClientSession | None = None
        self._shutdown_called = False

    async def send(self, alert: Alert) -> None:
        """Send alert notification via SendGrid."""
        if self._shutdown_called:
            raise RuntimeError("Notifier has been shut down")
        if not self._api_key:
            raise RuntimeError("SendGrid API key missing from environment")

        subject = self._render_subject(alert)
        text_body = self._render_text(alert) if self._text_template else ""
        html_body = self._render_html(alert) if self._html_template else ""
        if not text_body and not html_body:
            raise RuntimeError("SendGrid email requires text or html content")

        payload = self._build_payload(subject, text_body, html_body)
        headers = {"Authorization": f"Bearer {self._api_key}"}
        url = f"{self._api_base}/mail/send"
        session = await self._get_session()

        async with session.post(url, json=payload, headers=headers) as response:
            if response.status >= 400:
                details = await response.text()
                raise RuntimeError(f"SendGrid email send failed ({response.status}): {details}")

        logger.info(
            "Sent SendGrid email alert: to=%s clip_id=%s",
            ",".join(self._to_emails),
            alert.clip_id,
        )

    async def ping(self) -> bool:
        """Health check - verify SendGrid credentials and connectivity."""
        if self._shutdown_called or not self._api_key:
            return False

        url = f"{self._api_base}/user/profile"
        headers = {"Authorization": f"Bearer {self._api_key}"}
        session = await self._get_session()
        try:
            async with session.get(url, headers=headers) as response:
                if response.status >= 400:
                    return False
                await response.read()
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            logger.warning("SendGrid ping failed: %s", e)
            return False
        return True

    async def shutdown(self, timeout: float | None = None) -> None:
        """Cleanup resources - close HTTP session."""
        _ = timeout
        if self._shutdown_called:
            return
        self._shutdown_called = True

        if self._session and not self._session.closed:
            await self._session.close()

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self._timeout_s)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    def _build_payload(self, subject: str, text_body: str, html_body: str) -> dict[str, object]:
        personalization: dict[str, object] = {
            "to": [{"email": email} for email in self._to_emails],
            "subject": subject,
        }
        if self._cc_emails:
            personalization["cc"] = [{"email": email} for email in self._cc_emails]
        if self._bcc_emails:
            personalization["bcc"] = [{"email": email} for email in self._bcc_emails]

        sender: dict[str, str] = {"email": self._from_email}
        if self._from_name:
            sender["name"] = self._from_name

        content: list[dict[str, str]] = []
        if text_body:
            content.append({"type": "text/plain", "value": text_body})
        if html_body:
            content.append({"type": "text/html", "value": html_body})

        return {
            "personalizations": [personalization],
            "from": sender,
            "content": content,
        }

    def _build_context(self, alert: Alert) -> defaultdict[str, str]:
        view_url = alert.view_url or alert.storage_uri or "n/a"
        analysis_html = self._render_analysis_html(alert.analysis)
        return defaultdict(
            str,
            {
                "camera_name": alert.camera_name,
                "clip_id": alert.clip_id,
                "risk_level": alert.risk_level or "unknown",
                "activity_type": alert.activity_type or "unknown",
                "notify_reason": alert.notify_reason,
                "summary": alert.summary or "",
                "view_url": view_url,
                "storage_uri": alert.storage_uri or "",
                "ts": alert.ts.isoformat(),
                "upload_failed": str(alert.upload_failed),
                "analysis_html": analysis_html,
            },
        )

    def _render_subject(self, alert: Alert) -> str:
        return self._subject_template.format_map(self._build_context(alert)).strip()

    def _render_text(self, alert: Alert) -> str:
        return self._text_template.format_map(self._build_context(alert)).strip()

    def _render_html(self, alert: Alert) -> str:
        return self._html_template.format_map(self._build_context(alert)).strip()

    def _render_analysis_html(self, analysis: SequenceAnalysis | None) -> str:
        if analysis is None:
            return ""
        return self._render_value_html(analysis.model_dump())

    def _render_value_html(self, value: object) -> str:
        match value:
            case None:
                return "<em>n/a</em>"
            case dict() as mapping:
                return self._render_dict_html(mapping)
            case list() as items:
                return self._render_list_html(items)
            case _:
                return html.escape(str(value))

    def _render_dict_html(self, mapping: dict[str, object]) -> str:
        items = []
        for key, value in mapping.items():
            rendered_value = self._render_value_html(value)
            items.append(f"<li><strong>{html.escape(str(key))}:</strong> {rendered_value}</li>")
        return "<ul>" + "".join(items) + "</ul>"

    def _render_list_html(self, items: list[object]) -> str:
        if not items:
            return "<ul><li>none</li></ul>"

        rendered_items = []
        for idx, item in enumerate(items, start=1):
            rendered_value = self._render_value_html(item)
            label = f"Item {idx}"
            if isinstance(item, dict):
                rendered_items.append(f"<li>{html.escape(label)}{rendered_value}</li>")
            else:
                rendered_items.append(f"<li>{rendered_value}</li>")
        return "<ul>" + "".join(rendered_items) + "</ul>"


# Plugin registration
from typing import cast

from pydantic import BaseModel

from homesec.interfaces import Notifier
from homesec.plugins.notifiers import NotifierPlugin, notifier_plugin


@notifier_plugin(name="sendgrid_email")
def sendgrid_email_plugin() -> NotifierPlugin:
    """SendGrid email notifier plugin factory.

    Returns:
        NotifierPlugin for SendGrid email notifications
    """
    from homesec.models.config import SendGridEmailConfig

    def factory(cfg: BaseModel) -> Notifier:
        # Config is already validated by app.py, just cast
        return SendGridEmailNotifier(cast(SendGridEmailConfig, cfg))

    return NotifierPlugin(
        name="sendgrid_email",
        config_model=SendGridEmailConfig,
        factory=factory,
    )
