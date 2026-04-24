"""Tests for Postgres backup maintenance."""

from __future__ import annotations

import asyncio
import json
import signal
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from homesec.maintenance.postgres_backup import (
    PostgresBackupManager,
    build_pg_dump_command,
    build_pg_dump_env,
    redact_backup_text,
)
from homesec.models.config import Config
from homesec.models.storage import StorageUploadResult
from tests.homesec.mocks.storage import MockStorage


class _Clock:
    def __init__(self) -> None:
        self.value = datetime(2026, 4, 23, 16, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        current = self.value
        self.value = self.value + timedelta(seconds=1)
        return current


class _FixedClock:
    def __call__(self) -> datetime:
        return datetime(2026, 4, 23, 16, 0, 0, 123456, tzinfo=UTC)


class _RecordingStorage(MockStorage):
    def __init__(self) -> None:
        super().__init__()
        self.deleted: list[str] = []
        self.fail_delete = False

    async def delete(self, storage_uri: str) -> None:
        self.deleted.append(storage_uri)
        if self.fail_delete:
            raise RuntimeError("delete failed password=secret")
        await super().delete(storage_uri)


class _FailingUploadStorage(_RecordingStorage):
    def __init__(self) -> None:
        super().__init__()
        self.fail_put = False

    async def put_file(self, local_path: Path, dest_path: str) -> StorageUploadResult:
        if self.fail_put:
            raise RuntimeError("upload failed")
        return await super().put_file(local_path, dest_path)


class _CancellingUploadStorage(_RecordingStorage):
    def __init__(self) -> None:
        super().__init__()
        self.cancel_next_upload = True

    async def put_file(self, local_path: Path, dest_path: str) -> StorageUploadResult:
        if self.cancel_next_upload:
            self.cancel_next_upload = False
            storage_uri = f"mock://{dest_path}"
            self.files[storage_uri] = local_path.read_bytes()
            raise asyncio.CancelledError
        return await super().put_file(local_path, dest_path)


class _CancelThenFailUploadStorage(_RecordingStorage):
    def __init__(self) -> None:
        super().__init__()
        self.cancel_next_upload = True

    async def put_file(self, local_path: Path, dest_path: str) -> StorageUploadResult:
        if self.cancel_next_upload:
            self.cancel_next_upload = False
            storage_uri = f"mock://{dest_path}"
            self.files[storage_uri] = local_path.read_bytes()
            raise asyncio.CancelledError
        raise RuntimeError("upload outage")


class _CancellingDeleteStorage(_RecordingStorage):
    async def delete(self, storage_uri: str) -> None:
        self.deleted.append(storage_uri)
        raise asyncio.CancelledError


class _BlockingDeleteStorage(_RecordingStorage):
    def __init__(self) -> None:
        super().__init__()
        self.delete_started = asyncio.Event()
        self.release_delete = asyncio.Event()

    async def delete(self, storage_uri: str) -> None:
        self.delete_started.set()
        await self.release_delete.wait()
        await super().delete(storage_uri)


class _WritingBackupManager(PostgresBackupManager):
    async def _dump_database(self, *, dsn: str, temp_path: Path) -> None:
        _ = dsn
        temp_path.write_bytes(b"custom pg_dump bytes")


class _WaitingBackupManager(PostgresBackupManager):
    started: asyncio.Event
    release: asyncio.Event

    async def _dump_database(self, *, dsn: str, temp_path: Path) -> None:
        _ = dsn
        self.started.set()
        await self.release.wait()
        temp_path.write_bytes(b"custom pg_dump bytes")


def _config(tmp_path: Path, *, keep_count: int = 5, upload_enabled: bool = True) -> Config:
    return Config.model_validate(
        {
            "version": 1,
            "cameras": [
                {
                    "name": "front",
                    "source": {
                        "backend": "local_folder",
                        "config": {"watch_dir": "recordings"},
                    },
                }
            ],
            "storage": {
                "backend": "local",
                "config": {"root": str(tmp_path / "storage")},
                "paths": {"backups_dir": "ops/backups"},
            },
            "state_store": {"dsn": "postgresql+asyncpg://user:secret@localhost:5432/homesec"},
            "filter": {"backend": "yolo", "config": {}},
            "vlm": {
                "backend": "openai",
                "config": {"api_key_env": "OPENAI_API_KEY", "model": "gpt-4o"},
            },
            "alert_policy": {"backend": "default", "config": {"min_risk_level": "low"}},
            "maintenance": {
                "postgres_backup": {
                    "enabled": True,
                    "interval": "15m",
                    "keep_count": keep_count,
                    "local_dir": str(tmp_path / "backups"),
                    "timeout_s": 1,
                    "upload": {"enabled": upload_enabled},
                }
            },
        }
    )


@pytest.mark.parametrize("interval", ["14m", "899s"])
def test_postgres_backup_config_rejects_interval_below_minimum(
    tmp_path: Path, interval: str
) -> None:
    """Config validation should enforce the documented minimum backup interval."""
    # Given: Backup config with an interval below 15 minutes
    data = _config(tmp_path).model_dump(mode="json")
    data["maintenance"]["postgres_backup"]["interval"] = interval

    # When/Then: Pydantic validation rejects the config
    with pytest.raises(ValueError, match="at least 15m"):
        Config.model_validate(data)


def test_pg_dump_command_uses_env_for_credentials() -> None:
    """pg_dump invocation should keep credentials out of command args."""
    # Given: A SQLAlchemy async Postgres DSN with credentials
    dsn = "postgresql+asyncpg://user:p%40ss@db.example:5433/homesec?sslmode=require"

    # When: Building libpq env and command arguments
    env = build_pg_dump_env(dsn, base_env={})
    cmd = build_pg_dump_command(6)

    # Then: Credentials are only present in environment and redaction handles text safely
    assert cmd == ["pg_dump", "--format=custom", "--compress=6"]
    assert "p%40ss" not in " ".join(cmd)
    assert env["PGHOST"] == "db.example"
    assert env["PGPORT"] == "5433"
    assert env["PGDATABASE"] == "homesec"
    assert env["PGUSER"] == "user"
    assert env["PGPASSWORD"] == "p@ss"
    assert env["PGSSLMODE"] == "require"
    assert redact_backup_text("postgresql://user:p@ss@db/homesec password=secret") == (
        "postgresql://user:[redacted]@db/homesec password=[redacted]"
    )


def test_pg_dump_env_ignores_ambient_libpq_variables() -> None:
    """pg_dump should only use target values derived from the configured DSN."""
    # Given: Ambient libpq variables point at a different database
    base_env = {
        "PATH": "/usr/bin",
        "PGHOST": "wrong-host",
        "PGPORT": "1111",
        "PGDATABASE": "wrong-db",
        "PGUSER": "wrong-user",
        "PGPASSWORD": "wrong-password",
        "PGSERVICE": "wrong-service",
    }
    dsn = "postgresql://user:secret@db.example/homesec?sslmode=require"

    # When: Building pg_dump environment variables
    env = build_pg_dump_env(dsn, base_env=base_env)

    # Then: Non-libpq environment is preserved and ambient PG* values are scrubbed
    assert env["PATH"] == "/usr/bin"
    assert env["PGHOST"] == "db.example"
    assert "PGPORT" not in env
    assert env["PGDATABASE"] == "homesec"
    assert env["PGUSER"] == "user"
    assert env["PGPASSWORD"] == "secret"
    assert env["PGSSLMODE"] == "require"
    assert "PGSERVICE" not in env


def test_pg_dump_env_maps_asyncpg_tls_query_params() -> None:
    """pg_dump should preserve TLS intent from asyncpg-style state-store DSNs."""
    # Given: A state-store DSN using asyncpg SSL query parameters and IPv6 host syntax
    dsn = (
        "postgresql+asyncpg://user:secret@[2001:db8::10]/homesec"
        "?ssl=require&sslrootcert=/certs/root.pem&sslcert=/certs/client.pem"
        "&sslkey=/certs/client.key&channel_binding=require"
    )

    # When: Building pg_dump environment variables
    env = build_pg_dump_env(dsn, base_env={})

    # Then: libpq receives equivalent TLS settings for pg_dump
    assert env["PGHOST"] == "2001:db8::10"
    assert env["PGSSLMODE"] == "require"
    assert env["PGSSLROOTCERT"] == "/certs/root.pem"
    assert env["PGSSLCERT"] == "/certs/client.pem"
    assert env["PGSSLKEY"] == "/certs/client.key"
    assert env["PGCHANNELBINDING"] == "require"


@pytest.mark.asyncio
async def test_backup_timeout_removes_temp_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Timed out pg_dump should terminate the process and clean temporary output."""
    # Given: A pg_dump process that never completes before timeout
    config = _config(tmp_path)
    config.maintenance.postgres_backup.timeout_s = 0.01
    storage = MockStorage()
    signals: list[int] = []

    class _HangingProcess:
        returncode: int | None = None

        async def communicate(self) -> tuple[bytes, bytes]:
            await asyncio.sleep(3600)
            return b"", b""

        def send_signal(self, sig: signal.Signals) -> None:
            signals.append(int(sig))
            self.returncode = -int(sig)

        async def wait(self) -> int:
            return self.returncode or 0

        def kill(self) -> None:
            signals.append(int(signal.SIGKILL))
            self.returncode = -int(signal.SIGKILL)

    async def _create_subprocess_exec(*args: Any, **kwargs: Any) -> _HangingProcess:
        _ = args
        stdout = kwargs["stdout"]
        stdout.write(b"partial")
        return _HangingProcess()

    monkeypatch.setattr("asyncio.create_subprocess_exec", _create_subprocess_exec)
    manager = PostgresBackupManager(
        config=config, storage=storage, clock=_Clock(), jitter_s=lambda: 60
    )
    await manager._initialize_local_state()

    # When: A backup run times out
    await manager.run_backup_once(reason="test")

    # Then: Temporary output is removed and failure status is recorded without secrets
    assert signals == [int(signal.SIGTERM)]
    assert not list((tmp_path / "backups").glob(".*.tmp"))
    assert not list((tmp_path / "backups").glob("homesec-postgres-*.dump"))
    status = manager.status()
    assert status.last_error is not None
    assert "timed out" in status.last_error


@pytest.mark.asyncio
async def test_backup_cancellation_terminates_pg_dump_and_removes_temp_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cancelled pg_dump should terminate the process and not leave temp output."""
    # Given: A pg_dump process that is cancelled during application shutdown
    config = _config(tmp_path)
    storage = MockStorage()
    started = asyncio.Event()
    signals: list[int] = []

    class _HangingProcess:
        returncode: int | None = None

        async def communicate(self) -> tuple[bytes, bytes]:
            started.set()
            await asyncio.sleep(3600)
            return b"", b""

        def send_signal(self, sig: signal.Signals) -> None:
            signals.append(int(sig))
            self.returncode = -int(sig)

        async def wait(self) -> int:
            return self.returncode or 0

        def kill(self) -> None:
            signals.append(int(signal.SIGKILL))
            self.returncode = -int(signal.SIGKILL)

    async def _create_subprocess_exec(*args: Any, **kwargs: Any) -> _HangingProcess:
        _ = args
        stdout = kwargs["stdout"]
        stdout.write(b"partial")
        return _HangingProcess()

    monkeypatch.setattr("asyncio.create_subprocess_exec", _create_subprocess_exec)
    manager = PostgresBackupManager(
        config=config, storage=storage, clock=_Clock(), jitter_s=lambda: 60
    )
    await manager._initialize_local_state()

    # When: The active backup task is cancelled
    task = asyncio.create_task(manager.run_backup_once(reason="test"))
    await asyncio.wait_for(started.wait(), timeout=1.0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # Then: pg_dump is terminated and temporary output is removed
    assert signals == [int(signal.SIGTERM)]
    assert not list((tmp_path / "backups").glob(".*.tmp"))
    status = manager.status()
    assert status.last_error == "Postgres backup cancelled during shutdown"


@pytest.mark.asyncio
async def test_manifest_recovery_scans_local_backups(tmp_path: Path) -> None:
    """Invalid manifest should be rebuilt from local backup artifacts."""
    # Given: A corrupt manifest and an existing local dump file
    local_dir = tmp_path / "backups"
    local_dir.mkdir()
    artifact = local_dir / "homesec-postgres-20260423-160000.dump"
    artifact.write_bytes(b"dump")
    (local_dir / "manifest.json").write_text("{bad json", encoding="utf-8")
    config = _config(tmp_path, upload_enabled=False)
    config.maintenance.postgres_backup.enabled = False
    storage = MockStorage()
    manager = PostgresBackupManager(config=config, storage=storage, clock=_Clock())

    # When: The manager starts
    await manager.start()

    # Then: Status reflects the recovered successful local artifact
    status = manager.status()
    assert status.last_success_at is not None
    assert status.last_local_path == str(artifact.resolve())


@pytest.mark.asyncio
async def test_manifest_recovery_marks_local_backups_pending_upload_when_enabled(
    tmp_path: Path,
) -> None:
    """Recovered local artifacts should inherit current upload intent."""
    # Given: A corrupt manifest and an existing local dump while uploads are enabled
    local_dir = tmp_path / "backups"
    local_dir.mkdir()
    artifact = local_dir / "homesec-postgres-20260423-160000.dump"
    artifact.write_bytes(b"dump")
    (local_dir / "manifest.json").write_text("{bad json", encoding="utf-8")
    config = _config(tmp_path, upload_enabled=True)
    storage = MockStorage()
    manager = PostgresBackupManager(config=config, storage=storage, clock=_Clock())

    # When: The manager initializes local state
    await manager._initialize_local_state()

    # Then: The recovered record is retryable instead of permanently local-only
    recovered = json.loads((local_dir / "manifest.json").read_text(encoding="utf-8"))
    record = recovered["records"][0]
    assert record["status"] == "success"
    assert record["upload_enabled"] is True
    assert record["uploaded"] is False
    assert record["retention_state"] == "pending_upload"


@pytest.mark.asyncio
async def test_manifest_recovery_reconciles_running_record_with_local_artifact(
    tmp_path: Path,
) -> None:
    """Startup recovery should manage artifacts from interrupted backup runs."""
    # Given: A manifest record left running after the dump file was finalized
    local_dir = tmp_path / "backups"
    local_dir.mkdir()
    artifact = local_dir / "homesec-postgres-20260423-160000.dump"
    artifact.write_bytes(b"dump")
    manifest = {
        "schema_version": 1,
        "records": [
            {
                "name": artifact.name,
                "started_at": "2026-04-23T16:00:00Z",
                "status": "running",
                "local_path": str(artifact),
                "upload_enabled": True,
                "uploaded": False,
            }
        ],
    }
    (local_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    config = _config(tmp_path, upload_enabled=True)
    config.maintenance.postgres_backup.enabled = False
    manager = PostgresBackupManager(config=config, storage=MockStorage(), clock=_Clock())

    # When: The manager initializes local state
    await manager.start()

    # Then: The stale running record becomes a retained backup with retryable upload state
    recovered = json.loads((local_dir / "manifest.json").read_text(encoding="utf-8"))
    record = recovered["records"][0]
    assert record["status"] == "success"
    assert record["retention_state"] == "pending_upload"
    assert recovered["last_success_at"] is not None


@pytest.mark.asyncio
async def test_successful_backups_upload_and_prune_oldest(tmp_path: Path) -> None:
    """Retention should prune oldest local and uploaded backup after successful backup."""
    # Given: Backup retention keeps only the newest successful record
    config = _config(tmp_path, keep_count=1)
    storage = _RecordingStorage()
    manager = _WritingBackupManager(
        config=config, storage=storage, clock=_Clock(), jitter_s=lambda: 60
    )
    await manager._initialize_local_state()

    # When: Two successful backups run
    await manager.run_backup_once(reason="test")
    await manager.run_backup_once(reason="test")

    # Then: The oldest local artifact and known uploaded object are pruned
    manifest = json.loads((tmp_path / "backups" / "manifest.json").read_text(encoding="utf-8"))
    records = manifest["records"]
    assert len(records) == 2
    pruned = [record for record in records if not record["retained"]]
    retained = [record for record in records if record["retained"]]
    assert len(pruned) == 1
    assert len(retained) == 1
    assert not Path(pruned[0]["local_path"]).exists()
    assert pruned[0]["storage_uri"] in storage.deleted
    assert all(path.startswith("mock://ops/backups/") for path in storage.files)


@pytest.mark.asyncio
async def test_upload_failure_skips_retention_that_would_delete_previous_uploaded_backup(
    tmp_path: Path,
) -> None:
    """Remote retention should not delete the prior uploaded backup after upload failure."""
    # Given: Retention keeps one backup and an existing upload is present
    config = _config(tmp_path, keep_count=1)
    storage = _FailingUploadStorage()
    manager = _WritingBackupManager(
        config=config, storage=storage, clock=_Clock(), jitter_s=lambda: 60
    )
    await manager._initialize_local_state()
    await manager.run_backup_once(reason="test")
    storage.fail_put = True

    # When: A later local dump succeeds but upload fails
    await manager.run_backup_once(reason="test")

    # Then: The previously uploaded backup is still retained locally and remotely
    manifest = json.loads((tmp_path / "backups" / "manifest.json").read_text(encoding="utf-8"))
    records = manifest["records"]
    assert len(records) == 2
    assert all(record["retained"] for record in records)
    assert storage.deleted == []
    assert records[-1]["uploaded"] is False
    assert "Upload failed" in records[-1]["last_error"]


@pytest.mark.asyncio
async def test_upload_outage_prunes_old_pending_local_backups(tmp_path: Path) -> None:
    """Upload outages should not retain unbounded local dump artifacts."""
    # Given: One uploaded backup and repeated upload failures with keep_count=2
    config = _config(tmp_path, keep_count=2)
    storage = _FailingUploadStorage()
    manager = _WritingBackupManager(
        config=config, storage=storage, clock=_Clock(), jitter_s=lambda: 60
    )
    await manager._initialize_local_state()
    await manager.run_backup_once(reason="test")
    storage.fail_put = True

    # When: More failed uploads occur than pending-upload retention allows
    await manager.run_backup_once(reason="test")
    await manager.run_backup_once(reason="test")
    await manager.run_backup_once(reason="test")
    await manager.run_backup_once(reason="test")

    # Then: The last uploaded backup is preserved and pending local dumps are bounded
    manifest = json.loads((tmp_path / "backups" / "manifest.json").read_text(encoding="utf-8"))
    records = manifest["records"]
    retained = [record for record in records if record["retained"]]
    pending = [
        record
        for record in records
        if record["retention_state"] == "pending_upload" and record["retained"]
    ]
    pruned_pending = [
        record for record in records if record["retention_state"] == "pruned_pending_upload"
    ]
    assert len(pending) == 2
    assert len(retained) == 3
    assert records[0]["uploaded"] is True
    assert records[0]["retained"] is True
    assert storage.deleted == []
    assert all(Path(record["local_path"]).exists() for record in retained)
    assert all(not Path(record["local_path"]).exists() for record in pruned_pending)


@pytest.mark.asyncio
async def test_cancelled_upload_record_is_not_pruned_during_later_upload_outage(
    tmp_path: Path,
) -> None:
    """Unknown remote upload state should keep its local retry handle."""
    # Given: The first upload is cancelled after a remote object may have been created
    config = _config(tmp_path, keep_count=1)
    storage = _CancelThenFailUploadStorage()
    manager = _WritingBackupManager(
        config=config, storage=storage, clock=_Clock(), jitter_s=lambda: 60
    )
    await manager._initialize_local_state()
    with pytest.raises(asyncio.CancelledError):
        await manager.run_backup_once(reason="test")

    # When: Later backup uploads fail enough times to trigger pending-upload pruning
    await manager.run_backup_once(reason="test")
    await manager.run_backup_once(reason="test")
    await manager.run_backup_once(reason="test")

    # Then: The cancelled-upload record is retained so it can be retried or reconciled
    manifest = json.loads((tmp_path / "backups" / "manifest.json").read_text(encoding="utf-8"))
    unknown = manifest["records"][0]
    pruned_pending = [
        record
        for record in manifest["records"]
        if record["retention_state"] == "pruned_pending_upload"
    ]
    assert unknown["retained"] is True
    assert unknown["uploaded"] is False
    assert unknown["storage_uri"] is None
    assert unknown["retention_state"] == "pending_upload_unknown"
    assert Path(unknown["local_path"]).exists()
    assert pruned_pending


@pytest.mark.asyncio
async def test_cancelled_upload_is_retried_before_retention_prunes_record(
    tmp_path: Path,
) -> None:
    """Retention should not orphan uploads whose cancellation left remote state unknown."""
    # Given: The first upload is cancelled after the remote object may have been created
    config = _config(tmp_path, keep_count=1)
    storage = _CancellingUploadStorage()
    manager = _WritingBackupManager(
        config=config, storage=storage, clock=_Clock(), jitter_s=lambda: 60
    )
    await manager._initialize_local_state()
    with pytest.raises(asyncio.CancelledError):
        await manager.run_backup_once(reason="test")

    # When: A later successful backup runs retention
    await manager.run_backup_once(reason="test")

    # Then: The interrupted upload is retried, recorded, and remotely deleted when pruned
    manifest = json.loads((tmp_path / "backups" / "manifest.json").read_text(encoding="utf-8"))
    old_record = manifest["records"][0]
    assert old_record["uploaded"] is True
    assert old_record["storage_uri"] in storage.deleted
    assert old_record["retained"] is False
    assert old_record["pending_remote_delete"] is False


@pytest.mark.asyncio
async def test_successful_backup_clears_previous_upload_failure_status(
    tmp_path: Path,
) -> None:
    """Status should report the latest attempt instead of stale historical failures."""
    # Given: The first backup succeeds locally but fails to upload
    config = _config(tmp_path)
    storage = _FailingUploadStorage()
    manager = _WritingBackupManager(
        config=config, storage=storage, clock=_Clock(), jitter_s=lambda: 60
    )
    await manager._initialize_local_state()
    storage.fail_put = True
    await manager.run_backup_once(reason="test")
    assert manager.status().last_error is not None

    # When: A later backup fully succeeds
    storage.fail_put = False
    await manager.run_backup_once(reason="test")

    # Then: The stale upload failure is cleared from current subsystem status
    assert manager.status().last_error is None


@pytest.mark.asyncio
async def test_successful_backups_do_not_overwrite_same_timestamp_artifacts(
    tmp_path: Path,
) -> None:
    """Backups should keep unique artifacts even if the clock repeats a timestamp."""
    # Given: A backup manager with a fixed clock timestamp
    config = _config(tmp_path, upload_enabled=False)
    storage = MockStorage()
    manager = _WritingBackupManager(config=config, storage=storage, clock=_FixedClock())
    await manager._initialize_local_state()

    # When: Two successful backups run with the same timestamp
    await manager.run_backup_once(reason="test")
    await manager.run_backup_once(reason="test")

    # Then: Both dump files are retained under distinct names
    artifacts = sorted((tmp_path / "backups").glob("homesec-postgres-*.dump"))
    assert len(artifacts) == 2
    assert artifacts[0].name != artifacts[1].name


@pytest.mark.asyncio
async def test_startup_remote_delete_retry_does_not_block_manager_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pending remote delete retry should run in the background during startup."""
    # Given: A manifest with a pending remote delete whose storage call is blocked
    config = _config(tmp_path)
    local_dir = tmp_path / "backups"
    local_dir.mkdir()
    manifest = {
        "schema_version": 1,
        "records": [
            {
                "name": "homesec-postgres-20260423-160000.dump",
                "started_at": "2026-04-23T16:00:00Z",
                "completed_at": "2026-04-23T16:00:01Z",
                "status": "success",
                "local_path": str(local_dir / "homesec-postgres-20260423-160000.dump"),
                "retained": False,
                "retention_state": "pending_remote_delete",
                "pending_remote_delete": True,
                "storage_uri": "mock://ops/backups/homesec-postgres-20260423-160000.dump",
            }
        ],
    }
    (local_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    storage = _BlockingDeleteStorage()
    manager = PostgresBackupManager(
        config=config, storage=storage, clock=_Clock(), jitter_s=lambda: 60
    )
    monkeypatch.setattr(
        "homesec.maintenance.postgres_backup.shutil.which", lambda _: "/usr/bin/pg_dump"
    )

    # When: The manager starts
    await asyncio.wait_for(manager.start(), timeout=1.0)

    # Then: Startup returns while the remote delete retry remains in flight
    await asyncio.wait_for(storage.delete_started.wait(), timeout=1.0)
    assert manager.enabled is True
    storage.release_delete.set()
    await manager.shutdown(timeout=1.0)


@pytest.mark.asyncio
async def test_startup_storage_retry_serializes_with_manual_backup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Startup storage retry should not mutate the manifest concurrently with a backup."""
    # Given: A startup remote-delete retry is blocked while a manual backup is requested
    config = _config(tmp_path)
    local_dir = tmp_path / "backups"
    local_dir.mkdir()
    manifest = {
        "schema_version": 1,
        "records": [
            {
                "name": "homesec-postgres-20260423-160000.dump",
                "started_at": "2026-04-23T16:00:00Z",
                "completed_at": "2026-04-23T16:00:01Z",
                "status": "success",
                "local_path": str(local_dir / "homesec-postgres-20260423-160000.dump"),
                "retained": False,
                "retention_state": "pending_remote_delete",
                "pending_remote_delete": True,
                "storage_uri": "mock://ops/backups/homesec-postgres-20260423-160000.dump",
            }
        ],
    }
    (local_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    storage = _BlockingDeleteStorage()
    manager = _WaitingBackupManager(
        config=config, storage=storage, clock=_Clock(), jitter_s=lambda: 60
    )
    manager.started = asyncio.Event()
    manager.release = asyncio.Event()
    monkeypatch.setattr(
        "homesec.maintenance.postgres_backup.shutil.which", lambda _: "/usr/bin/pg_dump"
    )
    await asyncio.wait_for(manager.start(), timeout=1.0)
    await asyncio.wait_for(storage.delete_started.wait(), timeout=1.0)

    # When: A manual backup is accepted while the retry still holds the lock
    request = manager.request_backup_now()

    # Then: The backup does not start until the storage retry is released
    assert request.accepted is True
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(manager.started.wait(), timeout=0.05)
    storage.release_delete.set()
    await asyncio.wait_for(manager.started.wait(), timeout=1.0)
    manager.release.set()
    assert manager._run_task is not None
    await manager._run_task
    await manager.shutdown(timeout=1.0)


@pytest.mark.asyncio
async def test_scheduled_backup_backs_off_when_manual_backup_is_active(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Scheduled backup loop should not spin when another backup is already running."""
    # Given: A scheduler whose next run is due while another backup is marked active
    config = _config(tmp_path)
    storage = MockStorage()
    manager = PostgresBackupManager(
        config=config, storage=storage, clock=_FixedClock(), jitter_s=lambda: 60
    )
    await manager._initialize_local_state()
    manager._manifest.next_run_at = datetime(2026, 4, 23, 15, 0, tzinfo=UTC)
    calls = 0

    async def _skipped_run(*, reason: str) -> bool:
        nonlocal calls
        assert reason == "scheduled"
        calls += 1
        return False

    monkeypatch.setattr(manager, "_run_backup", _skipped_run)

    # When: The scheduler observes the skipped run and is then stopped
    task = asyncio.create_task(manager._schedule_loop())
    await asyncio.sleep(0.05)
    manager._stop_event.set()
    await asyncio.wait_for(task, timeout=1.0)

    # Then: The loop waited after the skipped run instead of repeatedly retrying
    assert calls == 1


@pytest.mark.asyncio
async def test_scheduler_rechecks_next_run_after_sleep_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Scheduler should honor next_run_at changes made while it is sleeping."""
    # Given: A scheduler sleeping on an old deadline that is about to be superseded
    config = _config(tmp_path)
    storage = MockStorage()
    base_time = datetime(2026, 4, 23, 16, 0, 0, 123456, tzinfo=UTC)
    manager = PostgresBackupManager(
        config=config, storage=storage, clock=_FixedClock(), jitter_s=lambda: 60
    )
    await manager._initialize_local_state()
    manager._manifest.next_run_at = base_time + timedelta(seconds=0.02)
    calls = 0

    async def _unexpected_run(*, reason: str) -> bool:
        nonlocal calls
        assert reason == "scheduled"
        calls += 1
        return True

    monkeypatch.setattr(manager, "_run_backup", _unexpected_run)

    # When: A manual backup moves next_run_at later while the scheduler is asleep
    task = asyncio.create_task(manager._schedule_loop())
    await asyncio.sleep(0.005)
    manager._manifest.next_run_at = base_time + timedelta(hours=24)
    await asyncio.sleep(0.05)
    manager._stop_event.set()
    await asyncio.wait_for(task, timeout=1.0)

    # Then: The scheduler rechecks the new deadline and does not fire early
    assert calls == 0


@pytest.mark.asyncio
async def test_manual_backup_requests_are_single_flight_before_task_starts(
    tmp_path: Path,
) -> None:
    """Manual backup requests should reject immediately once a run is accepted."""
    # Given: An available backup manager whose dump is held open
    config = _config(tmp_path)
    storage = MockStorage()
    manager = _WaitingBackupManager(
        config=config,
        storage=storage,
        clock=_Clock(),
        jitter_s=lambda: 60,
    )
    manager.started = asyncio.Event()
    manager.release = asyncio.Event()
    await manager._initialize_local_state()
    manager._available = True

    # When: Two manual backup requests arrive before the first task has run
    first = manager.request_backup_now()
    second = manager.request_backup_now()

    # Then: Only the first request is accepted and the held task can finish cleanly
    assert first.accepted is True
    assert second.accepted is False
    assert second.message == "Postgres backup already running"
    await asyncio.wait_for(manager.started.wait(), timeout=1.0)
    manager.release.set()
    assert manager._run_task is not None
    await manager._run_task


@pytest.mark.asyncio
async def test_pending_remote_delete_retries_on_successful_backup(tmp_path: Path) -> None:
    """Remote delete tombstones should retry on later successful retention passes."""
    # Given: Remote delete fails during the first retention prune
    config = _config(tmp_path, keep_count=1)
    storage = _RecordingStorage()
    manager = _WritingBackupManager(
        config=config, storage=storage, clock=_Clock(), jitter_s=lambda: 60
    )
    await manager._initialize_local_state()
    await manager.run_backup_once(reason="test")
    storage.fail_delete = True
    await manager.run_backup_once(reason="test")
    storage.fail_delete = False

    # When: A later successful backup runs
    await manager.run_backup_once(reason="test")

    # Then: The pending remote delete is retried until it succeeds
    manifest = json.loads((tmp_path / "backups" / "manifest.json").read_text(encoding="utf-8"))
    assert sum(1 for record in manifest["records"] if record["pending_remote_delete"]) == 0
    assert any(record["attempt_count"] >= 2 for record in manifest["records"])


@pytest.mark.asyncio
async def test_cancelled_remote_delete_preserves_retry_tombstone(tmp_path: Path) -> None:
    """Cancelled remote deletes should remain retryable in the local manifest."""
    # Given: Retention will prune an uploaded backup but remote delete is cancelled
    config = _config(tmp_path, keep_count=1)
    storage = _CancellingDeleteStorage()
    manager = _WritingBackupManager(
        config=config, storage=storage, clock=_Clock(), jitter_s=lambda: 60
    )
    await manager._initialize_local_state()
    await manager.run_backup_once(reason="test")

    # When: The next backup is cancelled while deleting the old remote object
    with pytest.raises(asyncio.CancelledError):
        await manager.run_backup_once(reason="test")

    # Then: The pruned record keeps a pending remote-delete tombstone
    manifest = json.loads((tmp_path / "backups" / "manifest.json").read_text(encoding="utf-8"))
    pruned = manifest["records"][0]
    assert pruned["retained"] is False
    assert pruned["pending_remote_delete"] is True
    assert pruned["retention_state"] == "pending_remote_delete"
    assert pruned["storage_uri"] in storage.deleted
