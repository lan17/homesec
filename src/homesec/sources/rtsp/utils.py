from __future__ import annotations

import logging
import shlex

logger = logging.getLogger(__name__)


def _redact_rtsp_url(url: str) -> str:
    if "://" not in url:
        return url
    scheme, rest = url.split("://", 1)
    if "@" not in rest:
        return url
    _creds, host = rest.split("@", 1)
    return f"{scheme}://***:***@{host}"


def _format_cmd(cmd: list[str]) -> str:
    try:
        return shlex.join([str(x) for x in cmd])
    except Exception as exc:
        logger.warning("Failed to format command with shlex.join: %s", exc, exc_info=True)
        return " ".join([str(x) for x in cmd])


def _is_timeout_option_error(stderr_text: str) -> bool:
    text = stderr_text.lower()
    return ("rw_timeout" in text and ("not found" in text or "unrecognized option" in text)) or (
        "stimeout" in text and ("not found" in text or "unrecognized option" in text)
    )


def _next_backoff(backoff_s: float, cap_s: float, *, factor: float = 1.6) -> float:
    return min(backoff_s * factor, cap_s)
