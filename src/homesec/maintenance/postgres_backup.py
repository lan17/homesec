"""Periodic Postgres backup manager."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import re
import shutil
import signal
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

from pydantic import BaseModel, Field

from homesec.config import resolve_env_var
from homesec.interfaces import StorageBackend
from homesec.models.config import Config, PostgresBackupConfig
from homesec.storage_paths import build_backup_path

logger = logging.getLogger(__name__)
MANIFEST_SCHEMA_VERSION = 1
MANIFEST_NAME = "manifest.json"
PG_DUMP = "pg_dump"
_REDACTED = "[redacted]"
_PASSWORD_ASSIGNMENT = re.compile(r"(?i)(password=)[^ \n\r\t]+")
_URL_PASSWORD = re.compile(
    r"(?P<prefix>://[^:/?#\s]+:)(?P<password>[^/?#\s]*)(?=@[^/?#\s]+(?:[/?#\s]|$))"
)


@dataclass(frozen=True, slots=True)
class PostgresBackupStatus:
    """Current postgres backup subsystem status."""

    enabled: bool
    running: bool
    available: bool
    unavailable_reason: str | None
    last_attempted_at: datetime | None
    last_success_at: datetime | None
    last_error: str | None
    last_local_path: str | None
    last_uploaded_uri: str | None
    next_run_at: datetime | None
    pending_remote_delete_count: int


@dataclass(frozen=True, slots=True)
class PostgresBackupRunRequest:
    """Manual backup trigger result."""

    accepted: bool
    message: str


class _ManifestRecord(BaseModel):
    """Single backup manifest record."""

    name: str
    started_at: datetime
    completed_at: datetime | None = None
    status: str = "running"
    local_path: str
    upload_enabled: bool = True
    uploaded: bool = False
    destination_path: str | None = None
    storage_uri: str | None = None
    view_url: str | None = None
    last_error: str | None = None
    retained: bool = True
    retention_state: str = "retained"
    pruned_at: datetime | None = None
    pending_remote_delete: bool = False
    first_failed_at: datetime | None = None
    last_attempted_at: datetime | None = None
    attempt_count: int = 0


class _BackupManifest(BaseModel):
    """Local backup manifest persisted beside backup artifacts."""

    schema_version: int = MANIFEST_SCHEMA_VERSION
    status: str = "idle"
    last_attempted_at: datetime | None = None
    last_success_at: datetime | None = None
    last_error: str | None = None
    next_run_at: datetime | None = None
    records: list[_ManifestRecord] = Field(default_factory=list)


def redact_backup_text(value: str | None) -> str | None:
    """Redact DSN passwords and password assignments from status/log text."""
    if value is None:
        return None
    redacted = _URL_PASSWORD.sub(r"\g<prefix>" + _REDACTED, value)
    redacted = _PASSWORD_ASSIGNMENT.sub(r"\1" + _REDACTED, redacted)
    return redacted


def resolve_postgres_backup_dsn(config: Config) -> str | None:
    """Resolve the state-store DSN using the app runtime precedence."""
    if config.state_store.dsn_env:
        return resolve_env_var(config.state_store.dsn_env)
    return config.state_store.dsn


def build_pg_dump_env(dsn: str, base_env: Mapping[str, str] | None = None) -> dict[str, str]:
    """Build credential-bearing libpq environment variables from a Postgres DSN."""
    env = {
        key: value
        for key, value in (os.environ if base_env is None else base_env).items()
        if not key.upper().startswith("PG")
    }
    parsed = urlsplit(_normalize_libpq_scheme(dsn))
    if parsed.scheme not in {"postgresql", "postgres"}:
        raise ValueError("state_store DSN must use postgres/postgresql scheme")

    if parsed.hostname:
        env["PGHOST"] = parsed.hostname
    if parsed.port is not None:
        env["PGPORT"] = str(parsed.port)
    database = parsed.path.lstrip("/")
    if database:
        env["PGDATABASE"] = unquote(database)
    if parsed.username:
        env["PGUSER"] = unquote(parsed.username)
    if parsed.password is not None:
        env["PGPASSWORD"] = unquote(parsed.password)

    query = parse_qs(parsed.query, keep_blank_values=True)
    _copy_query_env(query, env, "host", "PGHOST")
    _copy_query_env(query, env, "port", "PGPORT")
    _copy_query_env(query, env, "dbname", "PGDATABASE")
    _copy_query_env(query, env, "user", "PGUSER")
    _copy_query_env(query, env, "password", "PGPASSWORD")
    _copy_query_env(query, env, "sslmode", "PGSSLMODE")
    _copy_ssl_query_env(query, env)
    _copy_query_env(query, env, "sslcert", "PGSSLCERT")
    _copy_query_env(query, env, "sslkey", "PGSSLKEY")
    _copy_query_env(query, env, "sslrootcert", "PGSSLROOTCERT")
    _copy_query_env(query, env, "sslcrl", "PGSSLCRL")
    _copy_query_env(query, env, "sslcrldir", "PGSSLCRLDIR")
    _copy_query_env(query, env, "channel_binding", "PGCHANNELBINDING")
    return env


def build_pg_dump_command(compression_level: int) -> list[str]:
    """Return the credential-free pg_dump command."""
    return [
        PG_DUMP,
        "--format=custom",
        f"--compress={compression_level}",
    ]


def _normalize_libpq_scheme(dsn: str) -> str:
    if dsn.startswith("postgresql+asyncpg://"):
        return dsn.replace("postgresql+asyncpg://", "postgresql://", 1)
    if dsn.startswith("postgres+asyncpg://"):
        return dsn.replace("postgres+asyncpg://", "postgres://", 1)
    return dsn


def _copy_query_env(
    query: Mapping[str, Sequence[str]],
    env: dict[str, str],
    query_key: str,
    env_key: str,
) -> None:
    values = query.get(query_key)
    if values:
        env[env_key] = values[-1]


def _copy_ssl_query_env(query: Mapping[str, Sequence[str]], env: dict[str, str]) -> None:
    values = query.get("ssl")
    if not values or "PGSSLMODE" in env:
        return
    value = values[-1].lower()
    if value in {"true", "1", "yes", "on"}:
        env["PGSSLMODE"] = "require"
    elif value in {"false", "0", "no", "off"}:
        env["PGSSLMODE"] = "disable"
    elif value in {"disable", "allow", "prefer", "require", "verify-ca", "verify-full"}:
        env["PGSSLMODE"] = value


def _now_utc() -> datetime:
    return datetime.now(UTC)


class PostgresBackupManager:
    """Manages periodic and manual Postgres backups."""

    def __init__(
        self,
        *,
        config: Config,
        storage: StorageBackend,
        clock: Callable[[], datetime] = _now_utc,
        monotonic: Callable[[], float] = time.monotonic,
        jitter_s: Callable[[], float] | None = None,
    ) -> None:
        self._config = config
        self._backup_config: PostgresBackupConfig = config.maintenance.postgres_backup
        self._storage = storage
        self._clock = clock
        self._monotonic = monotonic
        self._jitter_s = jitter_s or (lambda: random.uniform(5.0, 30.0))
        self._manifest = _BackupManifest()
        self._lock = asyncio.Lock()
        self._stop_event = asyncio.Event()
        self._loop_task: asyncio.Task[None] | None = None
        self._run_task: asyncio.Task[None] | None = None
        self._retry_task: asyncio.Task[None] | None = None
        self._run_in_flight = False
        self._available = False
        self._unavailable_reason: str | None = None
        self._started = False

    @property
    def enabled(self) -> bool:
        return self._backup_config.enabled

    async def start(self) -> None:
        """Start background scheduling if backups are enabled."""
        if self._started:
            return
        self._started = True
        try:
            await self._start()
        except Exception as exc:  # pragma: no cover - defensive safety net
            self._available = False
            self._unavailable_reason = redact_backup_text(str(exc)) or "backup startup failed"
            self._manifest.status = "unavailable"
            self._manifest.last_error = self._unavailable_reason
            logger.warning(
                "Postgres backup subsystem unavailable; continuing application startup: %s",
                self._unavailable_reason,
            )

    async def _start(self) -> None:
        """Start backup scheduling once; caller handles defensive errors."""
        await self._initialize_local_state()

        if not self.enabled:
            self._available = False
            self._unavailable_reason = "Postgres backups are disabled"
            return

        self._retry_task = asyncio.create_task(
            self._retry_pending_storage_work_safely(),
            name="postgres-backup-storage-retry",
        )
        pg_dump_path = await asyncio.to_thread(shutil.which, PG_DUMP)
        if pg_dump_path is None:
            self._available = False
            self._unavailable_reason = "pg_dump not found in PATH"
            await self._save_manifest()
            logger.warning("Postgres backup subsystem unavailable: %s", self._unavailable_reason)
            return

        self._available = True
        self._unavailable_reason = None
        self._schedule_startup_run()
        self._loop_task = asyncio.create_task(self._schedule_loop(), name="postgres-backup-loop")

    async def shutdown(self, timeout: float | None = None) -> None:
        """Stop the scheduler and wait for an active backup to finish."""
        self._stop_event.set()
        tasks = [
            task for task in (self._loop_task, self._run_task, self._retry_task) if task is not None
        ]
        if not tasks:
            return
        try:
            if timeout is None:
                await asyncio.gather(*tasks, return_exceptions=True)
            else:
                await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout)
        except TimeoutError:
            logger.warning(
                "Timed out waiting for postgres backup manager shutdown; cancelling tasks"
            )
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    def status(self) -> PostgresBackupStatus:
        """Return current backup status for API/UI consumers."""
        last_success = self._last_success_record()
        return PostgresBackupStatus(
            enabled=self.enabled,
            running=self._backup_running(),
            available=self._available,
            unavailable_reason=self._unavailable_reason,
            last_attempted_at=self._manifest.last_attempted_at,
            last_success_at=self._manifest.last_success_at,
            last_error=self._manifest.last_error,
            last_local_path=last_success.local_path if last_success else None,
            last_uploaded_uri=last_success.storage_uri if last_success else None,
            next_run_at=self._manifest.next_run_at if self.enabled and self._available else None,
            pending_remote_delete_count=sum(
                1 for record in self._manifest.records if record.pending_remote_delete
            ),
        )

    def request_backup_now(self) -> PostgresBackupRunRequest:
        """Accept a manual backup request if no backup is already running."""
        if not self.enabled:
            return PostgresBackupRunRequest(False, "Postgres backups are disabled")
        if not self._available:
            reason = self._unavailable_reason or "Postgres backup subsystem unavailable"
            return PostgresBackupRunRequest(False, reason)
        if not self._try_mark_backup_running():
            return PostgresBackupRunRequest(False, "Postgres backup already running")
        self._run_task = asyncio.create_task(
            self._run_marked_backup(reason="manual"),
            name="postgres-backup-manual",
        )
        return PostgresBackupRunRequest(True, "Postgres backup accepted")

    async def run_backup_once(self, *, reason: str = "manual") -> None:
        """Run one backup synchronously; intended for tests and controlled callers."""
        await self._run_backup(reason=reason)

    async def _initialize_local_state(self) -> None:
        local_dir = self._local_dir()
        await asyncio.to_thread(local_dir.mkdir, parents=True, exist_ok=True, mode=0o700)
        await asyncio.to_thread(os.chmod, local_dir, 0o700)
        self._manifest = await self._load_manifest()
        self._scan_local_backups()
        self._manifest.next_run_at = self._compute_next_run_at()
        await self._save_manifest()

    async def _load_manifest(self) -> _BackupManifest:
        path = self._manifest_path()
        if not path.exists():
            return _BackupManifest()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            manifest = _BackupManifest.model_validate(data)
        except Exception as exc:
            logger.warning("Postgres backup manifest could not be loaded; rebuilding: %s", exc)
            return _BackupManifest(
                status="manifest_recovered",
                last_error="Manifest could not be loaded; rebuilt from local artifacts",
            )
        if manifest.schema_version != MANIFEST_SCHEMA_VERSION:
            manifest.schema_version = MANIFEST_SCHEMA_VERSION
        return manifest

    def _scan_local_backups(self) -> None:
        known = {record.name: record for record in self._manifest.records}
        for artifact in sorted(self._local_dir().glob("homesec-postgres-*.dump")):
            existing = known.get(artifact.name)
            if existing is not None:
                if existing.status == "running":
                    stat = artifact.stat()
                    completed_at = datetime.fromtimestamp(stat.st_mtime, tz=UTC)
                    existing.completed_at = completed_at
                    existing.status = "success"
                    existing.local_path = str(artifact)
                    if existing.upload_enabled and not existing.uploaded:
                        existing.retention_state = "pending_upload"
                        existing.last_error = (
                            "Recovered local backup after interrupted run; upload will be retried"
                        )
                continue
            stat = artifact.stat()
            completed_at = datetime.fromtimestamp(stat.st_mtime, tz=UTC)
            upload_enabled = self.enabled and self._backup_config.upload.enabled
            self._manifest.records.append(
                _ManifestRecord(
                    name=artifact.name,
                    started_at=completed_at,
                    completed_at=completed_at,
                    status="success",
                    local_path=str(artifact),
                    upload_enabled=upload_enabled,
                    uploaded=False,
                    retention_state="pending_upload" if upload_enabled else "retained",
                    last_error=(
                        "Recovered local backup from manifest scan; upload will be retried"
                        if upload_enabled
                        else None
                    ),
                )
            )
        for record in self._manifest.records:
            if record.status != "running":
                continue
            record.completed_at = record.completed_at or self._clock()
            record.status = "failed"
            record.retained = False
            record.retention_state = "interrupted"
            record.last_error = "Backup interrupted before local artifact was finalized"
        self._refresh_manifest_summary()

    async def _schedule_loop(self) -> None:
        while not self._stop_event.is_set():
            next_run_at = self._manifest.next_run_at
            if next_run_at is None:
                next_run_at = self._clock() + timedelta(
                    seconds=self._backup_config.interval_seconds
                )
                self._manifest.next_run_at = next_run_at
                await self._save_manifest()

            delay_s = max(0.0, (next_run_at - self._clock()).total_seconds())
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=delay_s)
                break
            except TimeoutError:
                refreshed_next_run_at = self._manifest.next_run_at
                if refreshed_next_run_at is not None and refreshed_next_run_at > self._clock():
                    continue
                if not await self._run_backup(reason="scheduled"):
                    await self._wait_after_skipped_schedule()

    def _schedule_startup_run(self) -> None:
        now = self._clock()
        last_success = self._manifest.last_success_at
        if last_success is None:
            self._manifest.next_run_at = now + timedelta(seconds=self._jitter_s())
            return
        if last_success + timedelta(seconds=self._backup_config.interval_seconds) <= now:
            self._manifest.next_run_at = now + timedelta(seconds=self._jitter_s())

    async def _run_backup(self, *, reason: str) -> bool:
        if not self._try_mark_backup_running():
            return False
        await self._run_marked_backup(reason=reason)
        return True

    async def _wait_after_skipped_schedule(self) -> None:
        try:
            await asyncio.wait_for(self._stop_event.wait(), timeout=1.0)
        except TimeoutError:
            pass

    async def _run_marked_backup(self, *, reason: str) -> None:
        _ = reason
        try:
            async with self._lock:
                await self._run_backup_locked()
        finally:
            self._run_in_flight = False

    def _backup_running(self) -> bool:
        return self._run_in_flight or self._lock.locked()

    def _try_mark_backup_running(self) -> bool:
        if self._run_in_flight:
            return False
        self._run_in_flight = True
        return True

    async def _run_backup_locked(self) -> None:
        started_at = self._clock()
        self._manifest.status = "running"
        self._manifest.last_attempted_at = started_at
        self._manifest.last_error = None

        artifact_name = self._build_artifact_name(started_at)
        local_path = self._local_dir() / artifact_name
        temp_path = self._local_dir() / f".{artifact_name}.tmp"
        record = _ManifestRecord(
            name=artifact_name,
            started_at=started_at,
            local_path=str(local_path),
            upload_enabled=self._backup_config.upload.enabled,
        )
        self._manifest.records.append(record)
        await self._save_manifest()

        try:
            dsn = resolve_postgres_backup_dsn(self._config)
            if not dsn:
                raise RuntimeError("Postgres DSN is not configured")

            await self._dump_database(dsn=dsn, temp_path=temp_path)
            await asyncio.to_thread(os.replace, temp_path, local_path)
            await asyncio.to_thread(os.chmod, local_path, 0o600)
            record.completed_at = self._clock()
            record.status = "success"
            self._manifest.last_success_at = record.completed_at

            if self._backup_config.upload.enabled:
                await self._upload_record(record, local_path)
                if record.last_error is not None:
                    self._manifest.last_error = record.last_error

            upload_complete = not self._backup_config.upload.enabled or (
                record.uploaded and record.last_error is None
            )
            if upload_complete:
                await self._retry_pending_uploads()
                await self._retry_pending_remote_deletes()
                await self._prune_retention()
            else:
                await self._prune_pending_upload_retention()
            self._manifest.status = "idle"
            self._manifest.next_run_at = record.completed_at + timedelta(
                seconds=self._backup_config.interval_seconds
            )
        except asyncio.CancelledError:
            await asyncio.to_thread(temp_path.unlink, missing_ok=True)
            error = "Postgres backup cancelled during shutdown"
            completed_at = self._clock()
            if record.status == "running":
                record.completed_at = completed_at
                record.status = "failed"
                record.last_error = error
                self._manifest.status = "failed"
                self._manifest.last_error = error
                retry_delay = min(15 * 60.0, self._backup_config.interval_seconds)
                self._manifest.next_run_at = completed_at + timedelta(seconds=retry_delay)
            else:
                if self._backup_config.upload.enabled and not record.uploaded:
                    record.last_error = "Upload cancelled during shutdown; upload will be retried"
                    record.retention_state = "pending_upload_unknown"
                    self._manifest.last_error = record.last_error
                self._manifest.status = "idle"
                if record.completed_at is not None:
                    self._manifest.next_run_at = record.completed_at + timedelta(
                        seconds=self._backup_config.interval_seconds
                    )
            raise
        except Exception as exc:
            await asyncio.to_thread(temp_path.unlink, missing_ok=True)
            error = redact_backup_text(str(exc)) or "Postgres backup failed"
            record.completed_at = self._clock()
            record.status = "failed"
            record.last_error = error
            self._manifest.status = "failed"
            self._manifest.last_error = error
            retry_delay = min(15 * 60.0, self._backup_config.interval_seconds)
            self._manifest.next_run_at = self._clock() + timedelta(seconds=retry_delay)
            logger.warning("Postgres backup failed: %s", error)
        finally:
            self._refresh_manifest_summary()
            await self._save_manifest()

    async def _dump_database(self, *, dsn: str, temp_path: Path) -> None:
        env = build_pg_dump_env(dsn)
        cmd = build_pg_dump_command(self._backup_config.compression_level)
        started = self._monotonic()
        with open(temp_path, "wb", opener=_private_file_opener) as stdout:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=stdout,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
            try:
                _, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=self._backup_config.timeout_s,
                )
            except asyncio.CancelledError:
                await self._terminate_timed_out_dump(proc)
                raise
            except TimeoutError as exc:
                await self._terminate_timed_out_dump(proc)
                raise TimeoutError(
                    f"pg_dump timed out after {self._backup_config.timeout_s:g}s"
                ) from exc

        duration_ms = int((self._monotonic() - started) * 1000)
        stderr = redact_backup_text(
            stderr_bytes.decode("utf-8", errors="replace") if stderr_bytes else ""
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"pg_dump failed with exit code {proc.returncode}: {stderr or 'no stderr'}"
            )
        logger.info("Postgres backup dump completed in %sms", duration_ms)

    async def _terminate_timed_out_dump(self, proc: asyncio.subprocess.Process) -> None:
        if proc.returncode is not None:
            return
        with suppress(ProcessLookupError):
            proc.send_signal(signal.SIGTERM)
        try:
            await asyncio.wait_for(proc.wait(), timeout=10.0)
            return
        except TimeoutError:
            pass
        if proc.returncode is None:
            with suppress(ProcessLookupError):
                proc.kill()
            await proc.wait()

    async def _upload_record(self, record: _ManifestRecord, local_path: Path) -> None:
        destination_path = build_backup_path(local_path.name, self._config.storage.paths)
        record.destination_path = destination_path
        record.last_attempted_at = self._clock()
        record.attempt_count += 1
        try:
            result = await self._storage.put_file(local_path, destination_path)
        except asyncio.CancelledError:
            record.first_failed_at = record.first_failed_at or record.last_attempted_at
            record.last_error = "Upload cancelled during shutdown; upload will be retried"
            record.retention_state = "pending_upload_unknown"
            raise
        except Exception as exc:
            record.first_failed_at = record.first_failed_at or record.last_attempted_at
            record.last_error = redact_backup_text(f"Upload failed: {exc}")
            if record.retention_state != "pending_upload_unknown":
                record.retention_state = "pending_upload"
            return
        record.uploaded = True
        record.storage_uri = result.storage_uri
        record.view_url = result.view_url
        record.last_error = None
        if record.retained:
            record.retention_state = "retained"

    async def _retry_pending_uploads(self) -> None:
        changed = False
        for record in self._manifest.records:
            if (
                record.status != "success"
                or not record.upload_enabled
                or record.uploaded
                or not record.retained
            ):
                continue
            local_path = Path(record.local_path)
            if not local_path.exists():
                record.retention_state = "upload_retry_missing_local"
                record.last_error = "Cannot retry upload because local backup artifact is missing"
                changed = True
                continue
            await self._upload_record(record, local_path)
            changed = True
        if changed:
            await self._save_manifest()

    async def _prune_retention(self) -> None:
        retained = [
            record
            for record in self._manifest.records
            if (
                record.status == "success"
                and record.retained
                and (not record.upload_enabled or record.uploaded)
            )
        ]
        retained.sort(key=lambda record: record.completed_at or record.started_at)
        excess = max(0, len(retained) - self._backup_config.keep_count)
        for record in retained[:excess]:
            await self._prune_record(record)

    async def _prune_record(self, record: _ManifestRecord) -> None:
        await asyncio.to_thread(Path(record.local_path).unlink, missing_ok=True)
        record.retained = False
        record.pruned_at = self._clock()
        record.retention_state = "pruned"
        if not record.storage_uri:
            return
        now = self._clock()
        record.pending_remote_delete = True
        record.retention_state = "pending_remote_delete"
        record.last_attempted_at = now
        record.attempt_count += 1
        record.last_error = "Remote delete in progress; will retry if interrupted"
        await self._save_manifest()
        try:
            await self._storage.delete(record.storage_uri)
        except asyncio.CancelledError:
            record.first_failed_at = record.first_failed_at or now
            record.last_error = "Remote delete cancelled during shutdown; will retry"
            raise
        except Exception as exc:
            record.first_failed_at = record.first_failed_at or now
            record.last_error = redact_backup_text(f"Remote delete failed: {exc}")
            return
        record.pending_remote_delete = False
        record.retention_state = "pruned"
        record.last_error = None

    async def _retry_pending_storage_work_safely(self) -> None:
        try:
            async with self._lock:
                await self._retry_pending_uploads()
                await self._retry_pending_remote_deletes()
                await self._prune_retention()
                await self._prune_pending_upload_retention()
                await self._save_manifest()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("Postgres backup storage retry failed", exc_info=True)

    async def _retry_pending_remote_deletes(self) -> None:
        changed = False
        for record in self._manifest.records:
            if not record.pending_remote_delete or not record.storage_uri:
                continue
            now = self._clock()
            record.last_attempted_at = now
            record.attempt_count += 1
            try:
                await self._storage.delete(record.storage_uri)
            except Exception as exc:
                record.first_failed_at = record.first_failed_at or now
                record.last_error = redact_backup_text(f"Remote delete failed: {exc}")
                changed = True
                continue
            record.pending_remote_delete = False
            record.retention_state = "pruned"
            record.last_error = None
            changed = True
        if changed:
            await self._save_manifest()

    async def _prune_pending_upload_retention(self) -> None:
        pending = [
            record
            for record in self._manifest.records
            if (
                record.status == "success"
                and record.upload_enabled
                and not record.uploaded
                and record.retained
                and record.retention_state != "pending_upload_unknown"
            )
        ]
        pending.sort(key=lambda record: record.completed_at or record.started_at)
        excess = max(0, len(pending) - self._backup_config.keep_count)
        for record in pending[:excess]:
            await asyncio.to_thread(Path(record.local_path).unlink, missing_ok=True)
            record.retained = False
            record.pruned_at = self._clock()
            record.retention_state = "pruned_pending_upload"
            record.pending_remote_delete = False
            record.last_error = (
                "Local artifact pruned after repeated upload failures; "
                "remote upload will not be retried"
            )

    def _refresh_manifest_summary(self) -> None:
        successful = self._last_success_record()
        if successful is not None:
            self._manifest.last_success_at = successful.completed_at
        latest = self._last_attempt_record()
        if self._manifest.status != "running" and latest is not None:
            self._manifest.last_error = latest.last_error

    def _last_success_record(self) -> _ManifestRecord | None:
        successful = [
            record
            for record in self._manifest.records
            if record.status == "success" and record.completed_at is not None
        ]
        if not successful:
            return None
        return max(successful, key=lambda record: record.completed_at or record.started_at)

    def _last_attempt_record(self) -> _ManifestRecord | None:
        if not self._manifest.records:
            return None
        return max(
            self._manifest.records,
            key=lambda record: record.completed_at or record.started_at,
        )

    def _compute_next_run_at(self) -> datetime | None:
        if not self.enabled:
            return None
        last_success = self._manifest.last_success_at
        if last_success is None:
            return self._clock() + timedelta(seconds=self._backup_config.interval_seconds)
        return last_success + timedelta(seconds=self._backup_config.interval_seconds)

    def _build_artifact_name(self, started_at: datetime) -> str:
        stem = f"homesec-postgres-{started_at.strftime('%Y%m%d-%H%M%S-%f')}"
        known = {record.name for record in self._manifest.records}
        candidate = f"{stem}.dump"
        suffix = 1
        while candidate in known or (self._local_dir() / candidate).exists():
            candidate = f"{stem}-{suffix}.dump"
            suffix += 1
        return candidate

    async def _save_manifest(self) -> None:
        path = self._manifest_path()
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        data = self._manifest.model_dump(mode="json")
        tmp = path.with_name(f".{path.name}.tmp")
        text = json.dumps(data, indent=2, sort_keys=True) + "\n"
        with open(tmp, "w", encoding="utf-8", opener=_private_file_opener) as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
        os.chmod(path, 0o600)

    def _local_dir(self) -> Path:
        return self._backup_config.local_dir.expanduser().resolve()

    def _manifest_path(self) -> Path:
        return self._local_dir() / MANIFEST_NAME


def _private_file_opener(path: str, flags: int) -> int:
    return os.open(path, flags, 0o600)


def run_pg_dump_available() -> bool:
    """Return whether pg_dump is available in PATH."""
    return shutil.which(PG_DUMP) is not None


def run_pg_dump_version() -> subprocess.CompletedProcess[str]:
    """Run pg_dump --version for setup/preflight checks."""
    return subprocess.run(
        [PG_DUMP, "--version"],
        check=False,
        capture_output=True,
        text=True,
        timeout=5.0,
    )
