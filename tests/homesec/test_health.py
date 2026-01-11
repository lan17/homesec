"""Tests for health check endpoint."""

from __future__ import annotations

import time

import pytest

from homesec.health import HealthServer
from tests.homesec.mocks import MockNotifier, MockStateStore, MockStorage
from homesec.sources import LocalFolderSource, LocalFolderSourceConfig


class TestHealthServer:
    """Test HealthServer implementation."""

    @pytest.mark.asyncio
    async def test_all_healthy(self, tmp_path) -> None:
        """Test health status when all components are healthy."""
        server = HealthServer()
        
        # Given healthy components
        storage = MockStorage()
        state_store = MockStateStore()
        notifier = MockNotifier()
        source = LocalFolderSource(
            LocalFolderSourceConfig(watch_dir=str(tmp_path)), camera_name="test"
        )
        
        server.set_components(
            storage=storage,
            state_store=state_store,
            notifier=notifier,
            sources=[source],
        )
        
        # When computing health
        health = await server.compute_health()
        
        # Then health is healthy with no warnings
        assert health["status"] == "healthy"
        assert health["checks"]["db"] is True
        assert health["checks"]["storage"] is True
        assert health["checks"]["mqtt"] is True
        assert health["checks"]["sources"] is True
        assert health["warnings"] == []
        assert len(health["sources"]) == 1
        assert health["sources"][0]["name"] == "test"
        assert health["sources"][0]["healthy"] is True

    @pytest.mark.asyncio
    async def test_degraded_when_db_down(self, tmp_path) -> None:
        """Test degraded status when DB is down."""
        server = HealthServer()
        
        # Given a failing state store
        storage = MockStorage()
        state_store = MockStateStore(simulate_failure=True)
        notifier = MockNotifier()
        source = LocalFolderSource(
            LocalFolderSourceConfig(watch_dir=str(tmp_path)), camera_name="test"
        )
        
        server.set_components(
            storage=storage,
            state_store=state_store,
            notifier=notifier,
            sources=[source],
        )
        
        # When computing health
        health = await server.compute_health()
        
        # Then status is degraded
        assert health["status"] == "degraded"
        assert health["checks"]["db"] is False
        assert health["checks"]["storage"] is True

    @pytest.mark.asyncio
    async def test_degraded_when_mqtt_down(self, tmp_path) -> None:
        """Test degraded status when MQTT is down (non-critical)."""
        server = HealthServer()
        
        # Given a failing notifier and mqtt_is_critical=False
        storage = MockStorage()
        state_store = MockStateStore()
        notifier = MockNotifier(simulate_failure=True)
        source = LocalFolderSource(
            LocalFolderSourceConfig(watch_dir=str(tmp_path)), camera_name="test"
        )
        
        server.set_components(
            storage=storage,
            state_store=state_store,
            notifier=notifier,
            sources=[source],
            mqtt_is_critical=False,  # Non-critical
        )
        
        # When computing health
        health = await server.compute_health()
        
        # Then status is degraded
        assert health["status"] == "degraded"
        assert health["checks"]["mqtt"] is False

    @pytest.mark.asyncio
    async def test_unhealthy_when_storage_down(self, tmp_path) -> None:
        """Test unhealthy status when storage is down."""
        server = HealthServer()
        
        # Given failing storage
        storage = MockStorage(simulate_failure=True)
        state_store = MockStateStore()
        notifier = MockNotifier()
        source = LocalFolderSource(
            LocalFolderSourceConfig(watch_dir=str(tmp_path)), camera_name="test"
        )
        
        server.set_components(
            storage=storage,
            state_store=state_store,
            notifier=notifier,
            sources=[source],
        )
        
        # When computing health
        health = await server.compute_health()
        
        # Then status is unhealthy
        assert health["status"] == "unhealthy"
        assert health["checks"]["storage"] is False

    @pytest.mark.asyncio
    async def test_unhealthy_when_sources_down(self, tmp_path) -> None:
        """Test unhealthy status when sources are down."""
        server = HealthServer()
        
        # Given a missing source watch directory
        storage = MockStorage()
        state_store = MockStateStore()
        notifier = MockNotifier()
        source = LocalFolderSource(
            LocalFolderSourceConfig(watch_dir=str(tmp_path / "nonexistent")),
            camera_name="test",
        )
        
        # When the watch dir is removed
        (tmp_path / "nonexistent").rmdir()
        
        server.set_components(
            storage=storage,
            state_store=state_store,
            notifier=notifier,
            sources=[source],
        )
        
        # Then health is unhealthy due to sources
        health = await server.compute_health()
        
        assert health["status"] == "unhealthy"
        assert health["checks"]["sources"] is False
        assert len(health["sources"]) == 1
        assert health["sources"][0]["name"] == "test"
        assert health["sources"][0]["healthy"] is False

    @pytest.mark.asyncio
    async def test_unhealthy_when_mqtt_critical_and_down(self, tmp_path) -> None:
        """Test unhealthy status when MQTT is critical and down."""
        server = HealthServer()
        
        # Given a failing notifier with mqtt_is_critical=True
        storage = MockStorage()
        state_store = MockStateStore()
        notifier = MockNotifier(simulate_failure=True)
        source = LocalFolderSource(
            LocalFolderSourceConfig(watch_dir=str(tmp_path)), camera_name="test"
        )
        
        server.set_components(
            storage=storage,
            state_store=state_store,
            notifier=notifier,
            sources=[source],
            mqtt_is_critical=True,  # Critical!
        )
        
        # When computing health
        health = await server.compute_health()
        
        # Then status is unhealthy
        assert health["status"] == "unhealthy"
        assert health["checks"]["mqtt"] is False

    @pytest.mark.asyncio
    async def test_heartbeat_warnings(self, tmp_path) -> None:
        """Test warnings for stale heartbeats."""
        server = HealthServer()
        
        # Given a source with a stale heartbeat
        source = LocalFolderSource(
            LocalFolderSourceConfig(watch_dir=str(tmp_path)), camera_name="front_door"
        )
        
        # When heartbeat is set far in the past
        source._last_heartbeat = time.monotonic() - 180  # 3 minutes ago
        
        server.set_components(sources=[source])
        
        # Then a warning is emitted
        health = await server.compute_health()
        
        assert len(health["warnings"]) == 1
        assert "source_front_door_heartbeat_stale" in health["warnings"]
        assert len(health["sources"]) == 1
        assert health["sources"][0]["name"] == "front_door"
        assert health["sources"][0]["last_heartbeat_age_s"] >= 180

    @pytest.mark.asyncio
    async def test_no_warnings_for_fresh_heartbeat(self, tmp_path) -> None:
        """Test no warnings when heartbeat is fresh."""
        server = HealthServer()
        
        # Given a source with a fresh heartbeat
        source = LocalFolderSource(
            LocalFolderSourceConfig(watch_dir=str(tmp_path)), camera_name="front_door"
        )
        
        server.set_components(sources=[source])
        
        # When computing health
        health = await server.compute_health()
        
        # Then no warnings are present
        assert health["warnings"] == []

    @pytest.mark.asyncio
    async def test_health_with_no_components(self) -> None:
        """Test health status with no components configured."""
        # Given: No components configured
        server = HealthServer()

        # When: Computing health
        health = await server.compute_health()

        # Then: Health is healthy
        assert health["status"] == "healthy"
        assert health["checks"]["db"] is True
        assert health["checks"]["storage"] is True
        assert health["checks"]["mqtt"] is True
        assert health["sources"] == []
        assert health["checks"]["sources"] is True
