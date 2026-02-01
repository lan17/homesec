"""FTP clip source for camera uploads."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from threading import Thread
from typing import Any

from pydantic import BaseModel, Field, field_validator

from homesec.models.clip import Clip
from homesec.sources.base import ThreadedClipSource

logger = logging.getLogger(__name__)


class FtpSourceConfig(BaseModel):
    """FTP source configuration."""

    model_config = {"extra": "forbid"}

    camera_name: str | None = Field(
        default=None,
        description="Optional human-friendly camera name.",
    )
    host: str = Field(
        default="0.0.0.0",
        description="FTP bind address.",
    )
    port: int = Field(
        default=2121,
        ge=0,
        le=65535,
        description="FTP listen port (0 lets the OS choose an ephemeral port).",
    )
    root_dir: str = Field(
        default="./ftp_incoming",
        description="FTP root directory for uploads.",
    )
    ftp_subdir: str | None = Field(
        default=None,
        description="Optional subdirectory under root_dir.",
    )
    anonymous: bool = Field(
        default=True,
        description="Allow anonymous FTP uploads.",
    )
    username_env: str | None = Field(
        default=None,
        description="Environment variable containing FTP username.",
    )
    password_env: str | None = Field(
        default=None,
        description="Environment variable containing FTP password.",
    )
    perms: str = Field(
        default="elw",
        description="pyftpdlib permissions string.",
    )
    passive_ports: str | None = Field(
        default=None,
        description="Passive ports range (e.g., '60000-60100' or '60000,60010').",
    )
    masquerade_address: str | None = Field(
        default=None,
        description="Optional masquerade address for passive mode.",
    )
    heartbeat_s: float = Field(
        default=30.0,
        ge=0.0,
        description="Seconds between FTP health checks.",
    )
    allowed_extensions: list[str] = Field(
        default_factory=lambda: [".mp4"],
        description="Allowed file extensions for uploaded clips.",
    )
    delete_non_matching: bool = Field(
        default=True,
        description="Delete files with disallowed extensions.",
    )
    delete_incomplete: bool = Field(
        default=True,
        description="Delete incomplete uploads when enabled.",
    )
    default_duration_s: float = Field(
        default=10.0,
        ge=0.0,
        description="Fallback clip duration when timestamps are missing.",
    )
    log_level: str = Field(
        default="INFO",
        description="FTP server log level.",
    )

    @field_validator("allowed_extensions")
    @classmethod
    def _normalize_extensions(cls, value: list[str]) -> list[str]:
        cleaned: list[str] = []
        for item in value:
            ext = str(item).strip().lower()
            if not ext:
                continue
            if not ext.startswith("."):
                ext = f".{ext}"
            cleaned.append(ext)
        return cleaned


def _parse_passive_ports(spec: str | None) -> list[int] | None:
    if not spec:
        return None
    spec = str(spec).strip()
    if not spec:
        return None

    if "-" in spec:
        start_s, end_s = spec.split("-", 1)
        start = int(start_s)
        end = int(end_s)
        if end < start:
            raise ValueError(f"Invalid passive_ports range: {spec!r}")
        return list(range(start, end + 1))

    ports: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if part:
            ports.append(int(part))
    return ports or None


class FtpSource(ThreadedClipSource):
    """FTP server clip source."""

    def __init__(self, config: FtpSourceConfig, camera_name: str) -> None:
        super().__init__()
        self._config = config
        self.camera_name = camera_name
        self.root_dir = Path(config.root_dir).expanduser().resolve()
        if config.ftp_subdir:
            self.root_dir = self.root_dir / config.ftp_subdir
        self._allowed_extensions = set(config.allowed_extensions)
        self._username = self._resolve_env(config.username_env)
        self._password = self._resolve_env(config.password_env)

        if not config.anonymous and (not self._username or not self._password):
            raise RuntimeError(
                "FTP auth requires username/password env vars when anonymous is False"
            )

        self._server: Any | None = None
        self._heartbeat_thread: Thread | None = None

    def is_healthy(self) -> bool:
        """Check if source is healthy."""
        return self._thread_is_healthy()

    def _heartbeat_loop(self) -> None:
        while not self._stop_event.wait(self._config.heartbeat_s):
            self._touch_heartbeat()

    def _run(self) -> None:
        if self._server is None:
            return
        self._touch_heartbeat()
        try:
            self._server.serve_forever()
        except Exception:
            logger.exception("FTP server stopped unexpectedly")

    def _create_server(self) -> Any:
        try:
            from pyftpdlib.authorizers import DummyAuthorizer  # type: ignore[import-untyped]
            from pyftpdlib.handlers import FTPHandler  # type: ignore[import-untyped]
            from pyftpdlib.servers import FTPServer  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError(
                "Missing dependency: pyftpdlib. Install with: uv pip install pyftpdlib"
            ) from exc

        authorizer = DummyAuthorizer()
        if self._config.anonymous:
            authorizer.add_anonymous(str(self.root_dir), perm=self._config.perms)
        else:
            assert self._username is not None
            assert self._password is not None
            authorizer.add_user(
                self._username, self._password, str(self.root_dir), perm=self._config.perms
            )

        source = self

        class UploadHandler(FTPHandler):  # type: ignore[misc]
            def on_connect(self) -> None:
                source._touch_heartbeat()

            def on_login(self, username: str) -> None:
                source._touch_heartbeat()

            def on_file_received(self, file: str) -> None:
                source._handle_file_received(Path(file))

            def on_incomplete_file_received(self, file: str) -> None:
                source._handle_incomplete_file(Path(file))

        UploadHandler.authorizer = authorizer

        parsed_passive_ports = _parse_passive_ports(self._config.passive_ports)
        if parsed_passive_ports is not None:
            UploadHandler.passive_ports = parsed_passive_ports
        if self._config.masquerade_address:
            UploadHandler.masquerade_address = self._config.masquerade_address

        return FTPServer((self._config.host, int(self._config.port)), UploadHandler)

    def _is_extension_allowed(self, file_path: Path) -> bool:
        if not self._allowed_extensions:
            return True
        return file_path.suffix.lower() in self._allowed_extensions

    def _handle_file_received(self, file_path: Path) -> None:
        self._touch_heartbeat()
        logger.info("Received upload: %s", file_path)
        if not self._is_extension_allowed(file_path):
            logger.info("Rejecting upload with unsupported extension: %s", file_path)
            if self._config.delete_non_matching:
                try:
                    file_path.unlink(missing_ok=True)
                except Exception:
                    logger.exception("Failed to delete non-matching upload: %s", file_path)
            return

        clip = self._create_clip(file_path)
        self._emit_clip(clip)

    def _handle_incomplete_file(self, file_path: Path) -> None:
        self._touch_heartbeat()
        logger.warning("Incomplete upload (deleting): %s", file_path)
        if not self._config.delete_incomplete:
            return
        try:
            file_path.unlink(missing_ok=True)
        except Exception:
            logger.exception("Failed to delete incomplete upload: %s", file_path)

    def _create_clip(self, file_path: Path) -> Clip:
        stat = file_path.stat()
        mtime = datetime.fromtimestamp(stat.st_mtime)
        duration_s = float(self._config.default_duration_s)
        return Clip(
            clip_id=file_path.stem,
            camera_name=self.camera_name,
            local_path=file_path,
            start_ts=mtime - timedelta(seconds=duration_s),
            end_ts=mtime,
            duration_s=duration_s,
            source_backend="ftp",
        )

    def _resolve_env(self, name: str | None) -> str | None:
        if not name:
            return None
        value = os.getenv(name)
        if value:
            return value
        return None

    def _on_start(self) -> None:
        logger.setLevel(getattr(logging, str(self._config.log_level).upper(), logging.INFO))
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self._server = self._create_server()

        if self._config.heartbeat_s > 0:
            self._heartbeat_thread = Thread(target=self._heartbeat_loop, daemon=True)
            self._heartbeat_thread.start()

        logger.info(
            "FTP source started: %s:%s (root=%s)",
            self._config.host,
            self._config.port,
            self.root_dir,
        )

    def _on_stop(self) -> None:
        logger.info("Stopping FtpSource...")
        if self._server is not None:
            try:
                self._server.close_all()
            except Exception:
                logger.exception("Failed to close FTP server")

        if self._heartbeat_thread and self._heartbeat_thread.is_alive():
            self._heartbeat_thread.join(timeout=5.0)

        self._heartbeat_thread = None
        self._server = None

    def _on_stopped(self) -> None:
        logger.info("FtpSource stopped")
