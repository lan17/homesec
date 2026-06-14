"""Tests for mobile device registration repository."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError
from sqlalchemy import select

from homesec.models.mobile import (
    MobileDeviceCapabilities,
    MobileDeviceRegistration,
    MobileDeviceUpdate,
)
from homesec.repository.mobile_device_repository import MobileDeviceRepository, hash_apns_token
from homesec.state.postgres import MobileDevice, PostgresStateStore


def _registration(
    *,
    apns_environment: str = "sandbox",
    apns_token: str = "raw-apns-token-123",
    bundle_id: str = "com.levneiman.homesec",
    device_name: str = "Lev's iPhone",
    app_version: str = "1.0.0",
) -> MobileDeviceRegistration:
    return MobileDeviceRegistration(
        apns_token=apns_token,
        apns_environment=apns_environment,
        bundle_id=bundle_id,
        device_name=device_name,
        app_version=app_version,
    )


def test_mobile_device_registration_rejects_blank_required_values() -> None:
    # Given: A registration payload with blank APNs material
    payload = {
        "apns_token": "   ",
        "apns_environment": "sandbox",
        "bundle_id": "com.levneiman.homesec",
    }

    # When: Validating the payload
    # Then: Validation rejects it before repository hashing
    with pytest.raises(ValidationError):
        MobileDeviceRegistration.model_validate(payload)


@pytest.mark.asyncio
async def test_register_device_creates_redacted_list_record(
    postgres_dsn: str,
    clean_test_db: None,
) -> None:
    # Given: A mobile device repository backed by Postgres
    state_store = PostgresStateStore(postgres_dsn)
    await state_store.initialize()
    repository = MobileDeviceRepository(state_store.engine)
    registration = _registration()

    # When: Registering an iOS device
    record = await repository.register_device(registration)
    records = await repository.list_devices()

    # Then: The repository returns public device metadata without raw APNs material
    assert record.id.startswith("dev_")
    assert record.platform == "ios"
    assert record.enabled is True
    assert record.apns_environment == "sandbox"
    assert record.bundle_id == "com.levneiman.homesec"
    assert record.capabilities.deep_links is True
    assert record.capabilities.rich_notifications is False
    assert records == [record]
    encoded = json.dumps(record.model_dump(mode="json"), sort_keys=True)
    assert registration.apns_token not in encoded
    assert "apns_token" not in encoded

    # And: The internal table stores a stable hash for dedupe
    async with state_store.engine.connect() as conn:
        row = (
            await conn.execute(
                select(MobileDevice.apns_token_hash).where(MobileDevice.id == record.id)
            )
        ).one()
    assert row.apns_token_hash == hash_apns_token(registration.apns_token)

    await state_store.shutdown()


@pytest.mark.asyncio
async def test_register_device_dedupes_by_token_hash(
    postgres_dsn: str,
    clean_test_db: None,
) -> None:
    # Given: An existing mobile device registration
    state_store = PostgresStateStore(postgres_dsn)
    await state_store.initialize()
    repository = MobileDeviceRepository(state_store.engine)
    first = await repository.register_device(
        _registration(),
        now=datetime(2026, 6, 14, 8, 0, tzinfo=timezone.utc),
    )

    # When: The same APNs token registers again with updated metadata
    second = await repository.register_device(
        _registration(device_name="Kitchen iPad", app_version="1.1.0"),
        now=datetime(2026, 6, 14, 8, 5, tzinfo=timezone.utc),
    )
    records = await repository.list_devices()

    # Then: The existing device row is updated instead of duplicated
    assert second.id == first.id
    assert second.device_name == "Kitchen iPad"
    assert second.app_version == "1.1.0"
    assert second.capabilities.deep_links is True
    assert second.last_seen_at == datetime(2026, 6, 14, 8, 5, tzinfo=timezone.utc)
    assert records == [second]

    await state_store.shutdown()


@pytest.mark.asyncio
async def test_disable_device_hides_record_without_deleting_it(
    postgres_dsn: str,
    clean_test_db: None,
) -> None:
    # Given: A registered mobile device
    state_store = PostgresStateStore(postgres_dsn)
    await state_store.initialize()
    repository = MobileDeviceRepository(state_store.engine)
    registered = await repository.register_device(_registration())

    # When: Disabling the device
    disabled = await repository.disable_device(registered.id)
    visible_records = await repository.list_devices()
    all_records = await repository.list_devices(include_disabled=True)

    # Then: Default listing hides it while retaining disabled history
    assert disabled is not None
    assert disabled.enabled is False
    assert visible_records == []
    assert all_records == [disabled]

    await state_store.shutdown()


@pytest.mark.asyncio
async def test_list_enabled_apns_targets_filters_disabled_environment_and_bundle(
    postgres_dsn: str,
    clean_test_db: None,
) -> None:
    # Given: Mobile devices across enabled state, APNs environment, and bundle id
    state_store = PostgresStateStore(postgres_dsn)
    await state_store.initialize()
    repository = MobileDeviceRepository(state_store.engine)
    enabled_match = await repository.register_device(
        _registration(apns_token="enabled-sandbox-token")
    )
    disabled_match = await repository.register_device(
        _registration(apns_token="disabled-sandbox-token")
    )
    await repository.disable_device(disabled_match.id)
    await repository.register_device(
        _registration(apns_environment="production", apns_token="production-token")
    )
    await repository.register_device(
        _registration(apns_token="other-bundle-token", bundle_id="com.example.other")
    )

    # When: Listing sandbox APNs push targets for the HomeSec bundle
    targets = await repository.list_enabled_apns_targets(
        environment="sandbox",
        bundle_id="com.levneiman.homesec",
    )

    # Then: Only the enabled matching iOS target is returned with token material
    assert len(targets) == 1
    assert targets[0].id == enabled_match.id
    assert targets[0].apns_token == "enabled-sandbox-token"
    assert targets[0].apns_environment == "sandbox"
    assert targets[0].bundle_id == "com.levneiman.homesec"

    await state_store.shutdown()


@pytest.mark.asyncio
async def test_record_push_result_updates_last_push_status(
    postgres_dsn: str,
    clean_test_db: None,
) -> None:
    # Given: A registered mobile device
    state_store = PostgresStateStore(postgres_dsn)
    await state_store.initialize()
    repository = MobileDeviceRepository(state_store.engine)
    registered = await repository.register_device(_registration())
    failed_at = datetime(2026, 6, 14, 8, 10, tzinfo=timezone.utc)

    # When: Recording a failed APNs delivery
    failed = await repository.record_push_result(
        registered.id,
        error=" BadDeviceToken ",
        now=failed_at,
    )

    # Then: The latest push attempt and normalized error are persisted
    assert failed is not None
    assert failed.last_push_at == failed_at
    assert failed.last_push_error == "BadDeviceToken"

    # When: Recording a later successful APNs delivery
    succeeded_at = datetime(2026, 6, 14, 8, 15, tzinfo=timezone.utc)
    succeeded = await repository.record_push_result(
        registered.id,
        error=None,
        now=succeeded_at,
    )

    # Then: The latest push time is refreshed and the previous error is cleared
    assert succeeded is not None
    assert succeeded.last_push_at == succeeded_at
    assert succeeded.last_push_error is None

    await state_store.shutdown()


@pytest.mark.asyncio
async def test_reregistering_disabled_device_preserves_disabled_state(
    postgres_dsn: str,
    clean_test_db: None,
) -> None:
    # Given: A disabled mobile device registration
    state_store = PostgresStateStore(postgres_dsn)
    await state_store.initialize()
    repository = MobileDeviceRepository(state_store.engine)
    registered = await repository.register_device(_registration())
    await repository.disable_device(registered.id)

    # When: The app registers the same APNs token again
    reregistered = await repository.register_device(
        _registration(device_name="Renamed iPhone"),
        now=datetime.now(timezone.utc) + timedelta(minutes=5),
    )

    # Then: Startup registration updates metadata without silently re-enabling push
    assert reregistered.id == registered.id
    assert reregistered.device_name == "Renamed iPhone"
    assert reregistered.enabled is False

    await state_store.shutdown()


@pytest.mark.asyncio
async def test_update_device_can_reenable_disabled_device(
    postgres_dsn: str,
    clean_test_db: None,
) -> None:
    # Given: A disabled mobile device
    state_store = PostgresStateStore(postgres_dsn)
    await state_store.initialize()
    repository = MobileDeviceRepository(state_store.engine)
    registered = await repository.register_device(_registration())
    await repository.disable_device(registered.id)

    # When: Updating the device enabled state explicitly
    updated = await repository.update_device(
        registered.id,
        MobileDeviceUpdate(
            enabled=True,
            device_name="Front Door iPhone",
            capabilities=MobileDeviceCapabilities(rich_notifications=True),
        ),
    )

    # Then: The device returns to default listings with updated metadata
    assert updated is not None
    assert updated.enabled is True
    assert updated.device_name == "Front Door iPhone"
    assert updated.capabilities.rich_notifications is True
    assert await repository.list_devices() == [updated]

    await state_store.shutdown()


@pytest.mark.asyncio
async def test_update_device_ignores_null_enabled_patch(
    postgres_dsn: str,
    clean_test_db: None,
) -> None:
    # Given: A registered mobile device
    state_store = PostgresStateStore(postgres_dsn)
    await state_store.initialize()
    repository = MobileDeviceRepository(state_store.engine)
    registered = await repository.register_device(_registration())

    # When: A partial update carries enabled=None
    updated = await repository.update_device(
        registered.id,
        MobileDeviceUpdate(enabled=None, app_version="1.2.0"),
    )

    # Then: The nullable patch value is ignored rather than writing NULL
    assert updated is not None
    assert updated.enabled is True
    assert updated.app_version == "1.2.0"

    await state_store.shutdown()
