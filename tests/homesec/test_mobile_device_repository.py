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
async def test_out_of_order_push_result_preserves_newer_delivery_and_metadata(
    postgres_dsn: str,
    clean_test_db: None,
) -> None:
    # Given: A newer accepted push and a subsequent device metadata update
    state_store = PostgresStateStore(postgres_dsn)
    await state_store.initialize()
    try:
        repository = MobileDeviceRepository(state_store.engine)
        started_at = datetime(2026, 6, 14, 8, 10, tzinfo=timezone.utc)
        registered = await repository.register_device(_registration(), now=started_at)
        accepted_at = started_at + timedelta(seconds=2)
        await repository.record_push_result(registered.id, error=None, now=accepted_at)
        metadata_at = started_at + timedelta(seconds=3)
        await repository.update_device(
            registered.id, MobileDeviceUpdate(device_name="Updated iPhone"), now=metadata_at
        )

        # When: An older send completes later with a failure
        result = await repository.record_push_result(
            registered.id, error="HTTP 503: ServiceUnavailable", now=started_at
        )

        # Then: Delivery status and metadata chronology remain at their newer values
        assert result is not None
        assert result.last_push_at == accepted_at
        assert result.last_push_error is None
        assert result.updated_at == metadata_at
        assert result.device_name == "Updated iPhone"
        assert await repository.get_device(registered.id) == result
    finally:
        await state_store.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("newer_change", ["registration", "reenable", "metadata", "accepted_push"])
async def test_stale_permanent_rejection_preserves_newer_device_state(
    postgres_dsn: str,
    clean_test_db: None,
    newer_change: str,
) -> None:
    # Given: A push starts before a newer registration, operator update, or accepted push
    state_store = PostgresStateStore(postgres_dsn)
    await state_store.initialize()
    try:
        repository = MobileDeviceRepository(state_store.engine)
        registered_at = datetime(2026, 6, 14, 8, 10, tzinfo=timezone.utc)
        registered = await repository.register_device(_registration(), now=registered_at)
        sent_at = registered_at + timedelta(seconds=1)
        target = (
            await repository.list_enabled_apns_targets(
                environment="sandbox", bundle_id="com.levneiman.homesec"
            )
        )[0]
        changed_at = sent_at + timedelta(seconds=1)
        if newer_change == "registration":
            await repository.register_device(
                _registration(apns_environment="production"), now=changed_at
            )
        elif newer_change == "accepted_push":
            await repository.record_push_result(registered.id, error=None, now=changed_at)
        else:
            patch = (
                MobileDeviceUpdate(enabled=True)
                if newer_change == "reenable"
                else MobileDeviceUpdate(device_name="Updated iPhone")
            )
            await repository.update_device(registered.id, patch, now=changed_at)

        # When: The earlier push finishes with a permanent token rejection
        result = await repository.record_push_result(
            registered.id,
            error="HTTP 410: Unregistered",
            now=sent_at,
            disable=True,
            expected_revision=target.revision,
        )

        # Then: The newer device state survives and its chronology never moves backward
        assert result is not None
        assert result.enabled is True
        assert result.updated_at == changed_at
        if newer_change == "registration":
            assert result.apns_environment == "production"
        if newer_change == "accepted_push":
            assert result.last_push_error is None
            assert result.last_push_at == changed_at
        else:
            assert result.last_push_error == "HTTP 410: Unregistered"
            assert result.last_push_at == sent_at
        assert await repository.get_device(registered.id) == result
    finally:
        await state_store.shutdown()


@pytest.mark.asyncio
async def test_current_permanent_rejection_records_failure_and_disables_device(
    postgres_dsn: str,
    clean_test_db: None,
) -> None:
    # Given: A registered device that remains unchanged during an APNs attempt
    state_store = PostgresStateStore(postgres_dsn)
    await state_store.initialize()
    try:
        repository = MobileDeviceRepository(state_store.engine)
        registered_at = datetime(2026, 6, 14, 8, 10, tzinfo=timezone.utc)
        registered = await repository.register_device(_registration(), now=registered_at)
        sent_at = registered_at + timedelta(seconds=1)
        target = (
            await repository.list_enabled_apns_targets(
                environment="sandbox", bundle_id="com.levneiman.homesec"
            )
        )[0]

        # When: Recording a permanent rejection from the current attempt
        result = await repository.record_push_result(
            registered.id,
            error="HTTP 410: Unregistered",
            now=sent_at,
            disable=True,
            expected_revision=target.revision,
        )

        # Then: The outcome and automatic disable are committed together
        assert result is not None
        assert result.enabled is False
        assert result.last_push_error == "HTTP 410: Unregistered"
        assert result.last_push_at == sent_at
        assert result.updated_at == sent_at
        assert await repository.list_devices() == []
    finally:
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


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["registration", "reenable", "metadata"])
@pytest.mark.parametrize("timestamp_offset", [-1, 0])
async def test_permanent_rejection_preserves_changes_with_non_newer_timestamps(
    postgres_dsn: str,
    clean_test_db: None,
    change: str,
    timestamp_offset: int,
) -> None:
    # Given: APNs selected a row before a later committed write with an old/equal timestamp
    state_store = PostgresStateStore(postgres_dsn)
    await state_store.initialize()
    try:
        repository = MobileDeviceRepository(state_store.engine)
        registered_at = datetime(2026, 6, 14, 8, 10, tzinfo=timezone.utc)
        registered = await repository.register_device(_registration(), now=registered_at)
        target = (
            await repository.list_enabled_apns_targets(
                environment="sandbox", bundle_id="com.levneiman.homesec"
            )
        )[0]
        changed_at = registered_at + timedelta(seconds=timestamp_offset)
        if change == "registration":
            await repository.register_device(
                _registration(apns_environment="production"), now=changed_at
            )
        else:
            patch = (
                MobileDeviceUpdate(enabled=True)
                if change == "reenable"
                else MobileDeviceUpdate(device_name="Updated iPhone")
            )
            await repository.update_device(registered.id, patch, now=changed_at)

        # When: The old APNs target is rejected after the write commits
        sent_at = registered_at + timedelta(seconds=1)
        result = await repository.record_push_result(
            registered.id,
            error="HTTP 400: BadDeviceToken",
            now=sent_at,
            disable=True,
            expected_revision=target.revision,
        )

        # Then: Row identity, rather than wall-clock ordering, protects the refreshed device
        assert result is not None
        assert result.enabled is True
        assert result.last_push_at == sent_at
        assert result.last_push_error == "HTTP 400: BadDeviceToken"
        if change == "registration":
            assert result.apns_environment == "production"
        if change == "metadata":
            assert result.device_name == "Updated iPhone"
        assert await repository.get_device(registered.id) == result
    finally:
        await state_store.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["registration", "metadata"])
async def test_queued_device_write_preserves_newer_chronology(
    postgres_dsn: str,
    clean_test_db: None,
    change: str,
) -> None:
    # Given: A newer registration followed by push bookkeeping
    state_store = PostgresStateStore(postgres_dsn)
    await state_store.initialize()
    try:
        repository = MobileDeviceRepository(state_store.engine)
        started_at = datetime(2026, 6, 14, 8, 10, tzinfo=timezone.utc)
        seen_at = started_at + timedelta(seconds=2)
        registered = await repository.register_device(_registration(), now=seen_at)
        push_at = seen_at + timedelta(seconds=1)
        await repository.record_push_result(registered.id, error=None, now=push_at)

        # When: A previously queued write commits with its older start timestamp
        if change == "registration":
            result = await repository.register_device(
                _registration(device_name="Updated iPhone"), now=started_at
            )
        else:
            result = await repository.update_device(
                registered.id, MobileDeviceUpdate(device_name="Updated iPhone"), now=started_at
            )

        # Then: Metadata changes while last-seen/update chronology stays monotonic
        assert result is not None
        assert result.device_name == "Updated iPhone"
        assert result.updated_at == push_at
        assert result.last_seen_at == seen_at
        assert result.last_push_at == push_at
    finally:
        await state_store.shutdown()
