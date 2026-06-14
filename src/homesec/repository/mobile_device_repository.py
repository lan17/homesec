"""Repository for iOS mobile device registrations."""

from __future__ import annotations

import hashlib
import secrets
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any, cast

from sqlalchemy import Table, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncEngine

from homesec.models.mobile import (
    MobileDeviceCapabilities,
    MobileDeviceRecord,
    MobileDeviceRegistration,
    MobileDeviceUpdate,
)
from homesec.state.postgres import MobileDevice


def hash_apns_token(apns_token: str) -> str:
    """Return the stable lookup hash for an APNs token."""
    normalized = _normalize_apns_token(apns_token)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


class MobileDeviceRepository:
    """Persistence boundary for mobile APNs device registrations."""

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def register_device(
        self,
        registration: MobileDeviceRegistration,
        *,
        now: datetime | None = None,
    ) -> MobileDeviceRecord:
        """Create or update a device by APNs token hash."""
        recorded_at = _utc_now() if now is None else now
        token = _normalize_apns_token(registration.apns_token)
        token_hash = hash_apns_token(token)

        table = cast(Table, MobileDevice.__table__)
        insert_stmt = pg_insert(table).values(
            id=_new_device_id(),
            platform=registration.platform,
            apns_token_hash=token_hash,
            # No encryption utility exists yet. Keep the raw token confined to
            # this internal column until the APNs sender ticket adds key management.
            apns_token_encrypted=token,
            apns_environment=registration.apns_environment,
            bundle_id=registration.bundle_id,
            device_name=registration.device_name,
            app_version=registration.app_version,
            capabilities=registration.capabilities.model_dump(mode="json"),
            enabled=True,
            created_at=recorded_at,
            updated_at=recorded_at,
            last_seen_at=recorded_at,
        )
        upsert_stmt = insert_stmt.on_conflict_do_update(
            index_elements=[table.c.apns_token_hash],
            set_={
                "platform": insert_stmt.excluded.platform,
                "apns_token_encrypted": insert_stmt.excluded.apns_token_encrypted,
                "apns_environment": insert_stmt.excluded.apns_environment,
                "bundle_id": insert_stmt.excluded.bundle_id,
                "device_name": insert_stmt.excluded.device_name,
                "app_version": insert_stmt.excluded.app_version,
                "capabilities": insert_stmt.excluded.capabilities,
                "updated_at": recorded_at,
                "last_seen_at": recorded_at,
            },
        )
        returning_stmt = upsert_stmt.returning(*_device_record_columns())

        async with self._engine.begin() as conn:
            row = cast(Mapping[str, Any], (await conn.execute(returning_stmt)).mappings().one())

        return _device_record_from_mapping(row)

    async def get_device(self, device_id: str) -> MobileDeviceRecord | None:
        """Return one mobile device record without raw APNs token material."""
        stmt = select(*_device_record_columns()).where(MobileDevice.id == device_id)
        async with self._engine.connect() as conn:
            row = cast(
                Mapping[str, Any] | None, (await conn.execute(stmt)).mappings().one_or_none()
            )
        if row is None:
            return None
        return _device_record_from_mapping(row)

    async def list_devices(self, *, include_disabled: bool = False) -> list[MobileDeviceRecord]:
        """List mobile device records without raw APNs token material."""
        stmt = select(*_device_record_columns())
        if not include_disabled:
            stmt = stmt.where(MobileDevice.enabled.is_(True))
        stmt = stmt.order_by(MobileDevice.updated_at.desc(), MobileDevice.id.asc())

        async with self._engine.connect() as conn:
            rows = (await conn.execute(stmt)).mappings().all()

        return [_device_record_from_mapping(cast(Mapping[str, Any], row)) for row in rows]

    async def update_device(
        self,
        device_id: str,
        patch: MobileDeviceUpdate,
        *,
        now: datetime | None = None,
    ) -> MobileDeviceRecord | None:
        """Update mutable device metadata and enabled state."""
        changes = patch.model_dump(exclude_unset=True, mode="json")
        if changes.get("enabled") is None:
            changes.pop("enabled", None)
        if changes.get("capabilities") is None:
            changes.pop("capabilities", None)
        if not changes:
            return await self.get_device(device_id)

        changes["updated_at"] = _utc_now() if now is None else now
        stmt = (
            update(MobileDevice)
            .where(MobileDevice.id == device_id)
            .values(**changes)
            .returning(*_device_record_columns())
        )
        async with self._engine.begin() as conn:
            row = cast(
                Mapping[str, Any] | None, (await conn.execute(stmt)).mappings().one_or_none()
            )
        if row is None:
            return None
        return _device_record_from_mapping(row)

    async def disable_device(
        self,
        device_id: str,
        *,
        now: datetime | None = None,
    ) -> MobileDeviceRecord | None:
        """Disable a mobile device without deleting its registration history."""
        return await self.update_device(
            device_id,
            MobileDeviceUpdate(enabled=False),
            now=now,
        )


def _device_record_columns() -> tuple[Any, ...]:
    return (
        MobileDevice.id,
        MobileDevice.platform,
        MobileDevice.apns_token_hash,
        MobileDevice.apns_environment,
        MobileDevice.bundle_id,
        MobileDevice.device_name,
        MobileDevice.app_version,
        MobileDevice.capabilities,
        MobileDevice.enabled,
        MobileDevice.created_at,
        MobileDevice.updated_at,
        MobileDevice.last_seen_at,
        MobileDevice.last_push_at,
        MobileDevice.last_push_error,
    )


def _device_record_from_mapping(row: Mapping[str, Any]) -> MobileDeviceRecord:
    token_hash = str(row["apns_token_hash"])
    return MobileDeviceRecord(
        id=str(row["id"]),
        platform=row["platform"],
        apns_environment=row["apns_environment"],
        bundle_id=str(row["bundle_id"]),
        device_name=row["device_name"],
        app_version=row["app_version"],
        capabilities=MobileDeviceCapabilities.model_validate(row["capabilities"] or {}),
        enabled=bool(row["enabled"]),
        token_fingerprint=token_hash[:12],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        last_seen_at=row["last_seen_at"],
        last_push_at=row["last_push_at"],
        last_push_error=row["last_push_error"],
    )


def _normalize_apns_token(apns_token: str) -> str:
    return apns_token.strip()


def _new_device_id() -> str:
    return f"dev_{secrets.token_urlsafe(16)}"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)
