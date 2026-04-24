"""Maintenance control-plane endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends, status
from pydantic import BaseModel

from homesec.api.dependencies import get_homesec_app
from homesec.api.errors import APIError, APIErrorCode

if TYPE_CHECKING:
    from homesec.app import Application

router = APIRouter(tags=["maintenance"])


class PostgresBackupStatusResponse(BaseModel):
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


class PostgresBackupRunResponse(BaseModel):
    accepted: bool
    message: str


@router.get(
    "/api/v1/maintenance/postgres-backups/status",
    response_model=PostgresBackupStatusResponse,
)
async def get_postgres_backup_status(
    app: Application = Depends(get_homesec_app),
) -> PostgresBackupStatusResponse:
    """Return current Postgres backup subsystem status."""
    backup_status = app.postgres_backup_manager.status()
    return PostgresBackupStatusResponse(
        enabled=backup_status.enabled,
        running=backup_status.running,
        available=backup_status.available,
        unavailable_reason=backup_status.unavailable_reason,
        last_attempted_at=backup_status.last_attempted_at,
        last_success_at=backup_status.last_success_at,
        last_error=backup_status.last_error,
        last_local_path=backup_status.last_local_path,
        last_uploaded_uri=backup_status.last_uploaded_uri,
        next_run_at=backup_status.next_run_at,
        pending_remote_delete_count=backup_status.pending_remote_delete_count,
    )


@router.post(
    "/api/v1/maintenance/postgres-backups/run",
    response_model=PostgresBackupRunResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def run_postgres_backup_now(
    app: Application = Depends(get_homesec_app),
) -> PostgresBackupRunResponse:
    """Trigger a manual Postgres backup."""
    manager = app.postgres_backup_manager
    backup_status = manager.status()
    if not backup_status.enabled:
        raise APIError(
            "Postgres backups are disabled",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            error_code=APIErrorCode.BACKUP_DISABLED,
        )
    if not backup_status.available:
        raise APIError(
            backup_status.unavailable_reason or "Postgres backup subsystem unavailable",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            error_code=APIErrorCode.BACKUP_UNAVAILABLE,
        )

    request = manager.request_backup_now()
    if not request.accepted:
        raise APIError(
            request.message,
            status_code=status.HTTP_409_CONFLICT,
            error_code=APIErrorCode.BACKUP_IN_PROGRESS,
        )

    return PostgresBackupRunResponse(accepted=True, message=request.message)
