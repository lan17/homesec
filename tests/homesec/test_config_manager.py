"""Tests for ConfigManager."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
import yaml

from homesec.config.manager import ConfigManager


def _write_config(path: Path, cameras: list[dict[str, object]]) -> ConfigManager:
    payload = {
        "version": 1,
        "cameras": cameras,
        "storage": {"backend": "dropbox", "config": {"root": "/homecam"}},
        "state_store": {"dsn": "postgresql://user:pass@localhost/db"},
        "notifiers": [{"backend": "mqtt", "config": {"host": "localhost"}}],
        "filter": {"backend": "yolo", "config": {}},
        "vlm": {
            "backend": "openai",
            "config": {"api_key_env": "OPENAI_API_KEY", "model": "gpt-4o"},
        },
        "alert_policy": {"backend": "default", "config": {}},
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    return ConfigManager(path)


@pytest.mark.asyncio
async def test_config_manager_add_update_remove_camera(tmp_path: Path) -> None:
    """ConfigManager should add, update, and remove cameras."""
    # Given a config with no cameras
    config_path = tmp_path / "config.yaml"
    manager = _write_config(config_path, cameras=[])

    # When adding a camera
    await manager.add_camera(
        name="front",
        enabled=True,
        source_backend="local_folder",
        source_config={"watch_dir": "/tmp"},
    )

    # Then the camera is persisted and backup exists
    config = manager.get_config()
    assert any(camera.name == "front" for camera in config.cameras)
    assert (config_path.with_suffix(".yaml.bak")).exists()

    # When updating the camera
    await manager.update_camera(
        camera_name="front",
        enabled=False,
        source_config={"watch_dir": "/new"},
    )

    # Then the camera is updated
    config = manager.get_config()
    updated = next(camera for camera in config.cameras if camera.name == "front")
    assert updated.enabled is False
    assert updated.source.config["watch_dir"] == "/new"

    # When removing the camera
    await manager.remove_camera(camera_name="front")

    # Then the camera is removed
    config = manager.get_config()
    assert not config.cameras


@pytest.mark.asyncio
async def test_config_manager_add_duplicate_raises(tmp_path: Path) -> None:
    """ConfigManager should reject duplicate camera names."""
    # Given a config with one camera
    config_path = tmp_path / "config.yaml"
    manager = _write_config(
        config_path,
        cameras=[
            {
                "name": "front",
                "enabled": True,
                "source": {"backend": "local_folder", "config": {"watch_dir": "/tmp"}},
            }
        ],
    )

    # When adding a duplicate camera
    with pytest.raises(ValueError):
        await manager.add_camera(
            name="front",
            enabled=True,
            source_backend="local_folder",
            source_config={"watch_dir": "/tmp"},
        )

    # Then it raises a validation error


@pytest.mark.asyncio
async def test_config_manager_update_missing_raises(tmp_path: Path) -> None:
    """ConfigManager should error when updating a missing camera."""
    # Given a config with no cameras
    config_path = tmp_path / "config.yaml"
    manager = _write_config(config_path, cameras=[])

    # When updating a missing camera
    with pytest.raises(ValueError):
        await manager.update_camera(camera_name="missing", enabled=False, source_config=None)

    # Then it raises a validation error


@pytest.mark.asyncio
async def test_config_manager_remove_missing_raises(tmp_path: Path) -> None:
    """ConfigManager should error when removing a missing camera."""
    # Given a config with no cameras
    config_path = tmp_path / "config.yaml"
    manager = _write_config(config_path, cameras=[])

    # When removing a missing camera
    with pytest.raises(ValueError):
        await manager.remove_camera(camera_name="missing")

    # Then it raises a validation error


@pytest.mark.asyncio
async def test_config_manager_invalid_update_raises(tmp_path: Path) -> None:
    """ConfigManager should reject invalid camera config updates."""
    # Given a config with one camera
    config_path = tmp_path / "config.yaml"
    manager = _write_config(
        config_path,
        cameras=[
            {
                "name": "front",
                "enabled": True,
                "source": {"backend": "local_folder", "config": {"watch_dir": "/tmp"}},
            }
        ],
    )

    # When updating with invalid source config
    with pytest.raises(ValueError):
        await manager.update_camera(
            camera_name="front",
            enabled=None,
            source_config={"poll_interval": -1.0},
        )

    # Then it raises a validation error


@pytest.mark.asyncio
async def test_config_manager_add_camera_concurrent_updates_preserve_all_changes(
    tmp_path: Path,
) -> None:
    """ConfigManager should serialize concurrent camera adds to avoid lost updates."""
    # Given a config with no cameras
    config_path = tmp_path / "config.yaml"
    manager = _write_config(config_path, cameras=[])

    # When adding different cameras concurrently
    await asyncio.gather(
        manager.add_camera(
            name="front",
            enabled=True,
            source_backend="local_folder",
            source_config={"watch_dir": "/tmp/front"},
        ),
        manager.add_camera(
            name="back",
            enabled=True,
            source_backend="local_folder",
            source_config={"watch_dir": "/tmp/back"},
        ),
    )

    # Then both camera additions are preserved
    config = manager.get_config()
    names = {camera.name for camera in config.cameras}
    assert names == {"front", "back"}
