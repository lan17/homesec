"""Behavioral coverage for versioned, save-only configuration edits."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import yaml
from fastapi.testclient import TestClient
from pydantic import ValidationError

from homesec.api.server import create_contract_app
from homesec.app import Application
from homesec.config.errors import (
    ConfigApplyInProgressError,
    ConfigBackendChangeUnsupportedError,
    ConfigPatchInvalidError,
    ConfigSaveError,
    ConfigVersionConflictError,
)
from homesec.config.loader import config_signature
from homesec.config.manager import ConfigManager, ConfigPatch
from homesec.models.config import Config, FastAPIServerConfig
from homesec.runtime.manager import RuntimeManager
from homesec.runtime.models import RuntimeReloadRequest, RuntimeState, RuntimeStatusSnapshot


def _manager(tmp_path: Path) -> ConfigManager:
    payload = {
        "cameras": [
            {
                "name": "front",
                "source": {
                    "backend": "rtsp",
                    "config": {"rtsp_url": "rtsp://saved-user:saved-pass@camera.local/live"},
                },
            }
        ],
        "storage": {
            "backend": "local",
            "config": {"root": "/tmp/storage", "legacy_secret": "saved-storage-secret"},
            "paths": {"clips_dir": "original-clips", "backups_dir": "original-backups"},
        },
        "state_store": {"dsn": "postgresql://saved-user:saved-pass@localhost/homesec"},
        "notifiers": [
            {
                "backend": "mqtt",
                "config": {
                    "host": "first.local",
                    "auth": {"username_env": "MQTT_USER", "password_env": "MQTT_PASSWORD"},
                    "legacy_secret": "saved-notifier-secret",
                },
            },
            {"backend": "mqtt", "enabled": False, "config": {"host": "second.local"}},
        ],
        "filter": {"backend": "yolo", "config": {"classes": ["person"], "max_workers": 7}},
        "vlm": {
            "backend": "openai",
            "run_mode": "never",
            "config": {"api_key_env": "OPENAI_API_KEY", "model": "gpt-4o"},
            "preprocessing": {"max_frames": 14, "max_size": 768, "quality": 72},
        },
        "alert_policy": {
            "backend": "default",
            "config": {
                "min_risk_level": "high",
                "overrides": {"front": {"notify_on_motion": True}},
            },
        },
        "retention": {"max_local_size_bytes": 1234},
        "maintenance": {"postgres_backup": {"enabled": True, "interval": "12h"}},
        "concurrency": {"max_clips_in_flight": 9},
        "retry": {"max_attempts": 8},
        "preview": {"enabled": True},
        "talk": {"enabled": False},
        "server": {"port": 8123},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return ConfigManager(path)


def _patch(manager: ConfigManager, **sections: object) -> ConfigPatch:
    return ConfigPatch.model_validate(
        {"expected_config_version": config_signature(manager.get_config()), **sections}
    )


@pytest.mark.asyncio
async def test_settings_patch_preserves_unrelated_sections_and_credentials(tmp_path: Path) -> None:
    # Given: A complete saved document with secrets, advanced settings, and repeated notifiers
    manager = _manager(tmp_path)
    before = manager.get_config().model_dump(mode="json")
    expected = deepcopy(before)
    expected["storage"]["config"]["root"] = "/tmp/new-storage"
    expected["storage"]["paths"]["clips_dir"] = "new-clips"
    expected["filter"]["config"]["min_confidence"] = 0.7
    expected["vlm"]["preprocessing"]["max_frames"] = 6
    expected["alert_policy"]["config"]["min_risk_level"] = "medium"
    expected["notifiers"][1]["enabled"] = True
    expected["notifiers"][1]["config"]["host"] = "updated.local"

    # When: Saving partial changes across supported sections
    saved = await manager.patch_config(
        _patch(
            manager,
            storage={"config": {"root": "/tmp/new-storage"}, "paths": {"clips_dir": "new-clips"}},
            filter={"config": {"min_confidence": 0.7}},
            vlm={"preprocessing": {"max_frames": 6}},
            alert_policy={"config": {"min_risk_level": "medium"}},
            notifiers=[{"index": 1, "enabled": True, "config": {"host": "updated.local"}}],
        )
    )

    # Then: Only requested fields change, and both same-backend notifier instances survive
    assert saved.model_dump(mode="json") == expected
    assert manager.get_config().model_dump(mode="json") == expected
    assert yaml.safe_load(Path(str(manager.config_path) + ".bak").read_text())["talk"] == {
        "enabled": False
    }


@pytest.mark.asyncio
async def test_settings_patch_null_clears_nested_optional_keys(tmp_path: Path) -> None:
    # Given: An existing notifier with two environment references
    manager = _manager(tmp_path)

    # When: Clearing one nested key and omitting the other
    saved = await manager.patch_config(
        _patch(manager, notifiers=[{"index": 0, "config": {"auth": {"username_env": None}}}])
    )

    # Then: The cleared field disappears while omitted credentials remain
    assert saved.notifiers[0].config["auth"] == {"password_env": "MQTT_PASSWORD"}
    assert saved.notifiers[0].config["legacy_secret"] == "saved-notifier-secret"


@pytest.mark.asyncio
@pytest.mark.parametrize("section", ["storage", "filter", "vlm", "alert_policy"])
async def test_settings_patch_rejects_backend_switch_without_writing(
    tmp_path: Path, section: str
) -> None:
    # Given: A valid saved config and its original bytes
    manager = _manager(tmp_path)
    before = manager.config_path.read_bytes()

    # When: Attempting to switch a backend through settings
    with pytest.raises(ConfigBackendChangeUnsupportedError):
        await manager.patch_config(_patch(manager, **{section: {"backend": "different"}}))

    # Then: The file is untouched and no backup write occurred
    assert manager.config_path.read_bytes() == before
    assert not Path(str(manager.config_path) + ".bak").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "sections",
    [
        {"filter": {"config": {"min_confidence": 4}}},
        {"storage": {"config": {"legacy_secret": "***redacted***"}}},
        {"notifiers": [{"index": 2, "enabled": True}]},
        {"notifiers": [{"index": 0, "enabled": True}, {"index": 0, "enabled": False}]},
    ],
)
async def test_settings_patch_rejects_invalid_edits_without_writing(
    tmp_path: Path, sections: dict[str, object]
) -> None:
    # Given: A valid saved configuration
    manager = _manager(tmp_path)
    before = manager.config_path.read_bytes()

    # When: A patch has invalid plugin data, masked secrets, or invalid notifier indexes
    with pytest.raises(ConfigPatchInvalidError):
        await manager.patch_config(_patch(manager, **sections))

    # Then: Validation failure preserves the original document
    assert manager.config_path.read_bytes() == before
    assert not Path(str(manager.config_path) + ".bak").exists()


@pytest.mark.asyncio
async def test_settings_patch_rejects_stale_version_after_external_edit(tmp_path: Path) -> None:
    # Given: A settings draft followed by an external reorder of repeated notifier instances
    manager = _manager(tmp_path)
    patch = _patch(manager, notifiers=[{"index": 0, "enabled": False}])
    payload = yaml.safe_load(manager.config_path.read_text())
    payload["notifiers"].reverse()
    manager.config_path.write_text(yaml.safe_dump(payload))
    externally_edited = manager.config_path.read_bytes()

    # When: Saving the stale indexed edit
    with pytest.raises(ConfigVersionConflictError):
        await manager.patch_config(patch)

    # Then: The external edit is preserved instead of patching the wrong notifier
    assert manager.config_path.read_bytes() == externally_edited


@pytest.mark.asyncio
async def test_config_snapshot_serializes_apply_acceptance_and_mutation(tmp_path: Path) -> None:
    # Given: A versioned snapshot held while apply acceptance is being scheduled
    manager = _manager(tmp_path)
    patch = _patch(manager, filter={"config": {"min_confidence": 0.8}})
    mutation: asyncio.Task[Config]
    async with manager.config_snapshot(patch.expected_config_version) as snapshot:
        # When: Another request attempts to save a patch
        mutation = asyncio.create_task(manager.patch_config(patch))
        await asyncio.sleep(0)

        # Then: It cannot write until acceptance releases the saved snapshot
        assert not mutation.done()
        assert manager.get_config() == snapshot
    assert (await mutation).filter.config["min_confidence"] == 0.8


@pytest.mark.asyncio
async def test_restart_acceptance_freezes_a_queued_settings_save(tmp_path: Path) -> None:
    # Given: A settings save queued behind the snapshot being accepted for process restart
    manager = _manager(tmp_path)
    before = manager.config_path.read_bytes()
    patch = _patch(manager, filter={"config": {"min_confidence": 0.8}})
    async with manager.config_snapshot(patch.expected_config_version):
        mutation = asyncio.create_task(manager.patch_config(patch))
        await asyncio.sleep(0)
        assert not mutation.done()

        # When: The accepted restart freezes writes before releasing the snapshot
        manager.freeze_mutations()

    # Then: The queued save reports restart in progress without creating a backup or changing YAML
    with pytest.raises(ConfigApplyInProgressError):
        await mutation
    assert manager.config_path.read_bytes() == before
    assert not Path(str(manager.config_path) + ".bak").exists()


@pytest.mark.asyncio
async def test_restart_acceptance_freezes_a_queued_camera_save(tmp_path: Path) -> None:
    # Given: A camera edit queued behind a saved snapshot held for restart acceptance
    manager = _manager(tmp_path)
    before = manager.config_path.read_bytes()
    version = config_signature(manager.get_config())
    async with manager.config_snapshot(version):
        mutation = asyncio.create_task(
            manager.update_camera(
                camera_name="front", enabled=False, source_backend=None, source_config=None
            )
        )
        await asyncio.sleep(0)
        assert not mutation.done()

        # When: Restart acceptance freezes writes before the queued mutation obtains its lock
        manager.freeze_mutations()

    # Then: The camera stays enabled on disk and no backup is written
    with pytest.raises(ConfigApplyInProgressError):
        await mutation
    assert manager.config_path.read_bytes() == before
    assert not Path(str(manager.config_path) + ".bak").exists()


@pytest.mark.asyncio
async def test_settings_patch_read_only_save_returns_safe_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given: A filesystem boundary that refuses atomic replacement
    manager = _manager(tmp_path)
    before = manager.config_path.read_bytes()

    def refuse_replace(source: Path, destination: Path) -> None:
        raise PermissionError("sensitive-filesystem-detail")

    monkeypatch.setattr("homesec.config.manager.os.replace", refuse_replace)

    # When: Saving an otherwise valid patch
    with pytest.raises(ConfigSaveError, match="directory is writable") as exc:
        await manager.patch_config(_patch(manager, filter={"config": {"min_confidence": 0.8}}))

    # Then: The error preserves its cause without exposing raw details or replacing the config
    assert isinstance(exc.value.__cause__, PermissionError)
    assert "sensitive-filesystem-detail" not in str(exc.value)
    assert manager.config_path.read_bytes() == before


@pytest.mark.parametrize(
    "sections", [{"storage": None}, {"vlm": {"run_mode": None}}, {"notifiers": None}]
)
def test_settings_patch_rejects_null_sections(sections: dict[str, object]) -> None:
    # Given: A request attempting to clear a whole section rather than a nested optional key
    payload = {"expected_config_version": "current", **sections}

    # When: Parsing the typed API input
    with pytest.raises(ValidationError):
        ConfigPatch.model_validate(payload)

    # Then: Whole-section nulls cannot be accepted as accidental defaults resets


class _SettingsApp:
    def __init__(self, manager: ConfigManager) -> None:
        self.config_manager = manager
        self.server_config = FastAPIServerConfig(auth_enabled=False)
        self.bootstrap_mode = False
        self.active_version = config_signature(manager.get_config())
        self.apply_calls: list[str] = []
        self.restart_requested = False

    def get_config_application_status(self, config: Config) -> SimpleNamespace:
        saved_version = config_signature(config)
        return SimpleNamespace(
            saved_config_version=saved_version,
            active_config_version=self.active_version,
            apply_required="none" if saved_version == self.active_version else "reload",
        )

    async def request_runtime_reload(self) -> None:
        self.apply_calls.append("reload")

    def request_restart(self) -> None:
        self.apply_calls.append("restart")


def _client(manager: ConfigManager) -> tuple[TestClient, _SettingsApp]:
    settings_app = _SettingsApp(manager)
    api = create_contract_app()
    api.state.homesec = settings_app
    return TestClient(api), settings_app


def test_config_api_reports_saved_and_active_versions_and_saves_only(tmp_path: Path) -> None:
    # Given: An active install whose saved config matches its active version
    manager = _manager(tmp_path)
    client, app = _client(manager)
    initial = client.get("/api/v1/config").json()
    assert initial["apply_required"] == "none"
    assert initial["saved_config_version"] == initial["active_config_version"]

    # When: Saving a versioned settings patch
    response = client.patch(
        "/api/v1/config",
        json={
            "expected_config_version": initial["saved_config_version"],
            "filter": {"config": {"min_confidence": 0.8}},
        },
    )

    # Then: Saved metadata changes, active metadata does not, secrets stay redacted, and no apply runs
    assert response.status_code == 200
    payload = response.json()
    assert payload["saved_config_version"] != initial["saved_config_version"]
    assert payload["active_config_version"] == initial["active_config_version"]
    assert payload["apply_required"] == "reload"
    assert payload["config"]["storage"]["config"]["legacy_secret"] == "***redacted***"
    assert payload["config"]["notifiers"][0]["config"]["auth"]["password_env"] == "MQTT_PASSWORD"
    assert app.apply_calls == []


def test_config_api_stale_patch_returns_conflict(tmp_path: Path) -> None:
    # Given: A request with an obsolete saved version
    manager = _manager(tmp_path)
    client, _ = _client(manager)
    before = manager.config_path.read_bytes()

    # When: Saving the stale patch
    response = client.patch(
        "/api/v1/config",
        json={"expected_config_version": "obsolete", "filter": {"config": {"min_confidence": 0.8}}},
    )

    # Then: A stable conflict response preserves the document
    assert response.status_code == 409
    assert response.json()["error_code"] == "CONFIG_VERSION_CONFLICT"
    assert manager.config_path.read_bytes() == before


@pytest.mark.parametrize("endpoint", ["config", "camera_create", "camera_update", "camera_delete"])
def test_restart_freeze_returns_conflict_for_all_configuration_writes(
    tmp_path: Path, endpoint: str
) -> None:
    # Given: A manager frozen by accepted restart, without relying on the route's earlier guard
    manager = _manager(tmp_path)
    payload = yaml.safe_load(manager.config_path.read_text())
    payload["alert_policy"]["config"]["overrides"] = {}
    manager.config_path.write_text(yaml.safe_dump(payload))
    client, app = _client(manager)
    before = manager.config_path.read_bytes()
    manager.freeze_mutations()

    # When: Saving global settings or adding, updating, or removing a camera
    match endpoint:
        case "config":
            response = client.patch(
                "/api/v1/config",
                json={
                    "expected_config_version": config_signature(manager.get_config()),
                    "filter": {"config": {"min_confidence": 0.8}},
                },
            )
        case "camera_create":
            response = client.post(
                "/api/v1/cameras?apply_changes=true",
                json={
                    "name": "back",
                    "source_backend": "local_folder",
                    "source_config": {"watch_dir": "/tmp/back"},
                },
            )
        case "camera_update":
            response = client.patch(
                "/api/v1/cameras/front?apply_changes=true", json={"enabled": False}
            )
        case _:
            response = client.delete("/api/v1/cameras/front?apply_changes=true")

    # Then: Every route reports restart in progress without persisting or requesting a reload
    assert response.status_code == 409
    assert response.json()["error_code"] == "CONFIG_APPLY_IN_PROGRESS"
    assert manager.config_path.read_bytes() == before
    assert not Path(str(manager.config_path) + ".bak").exists()
    assert app.apply_calls == []


class _ActiveSettingsRuntime:
    active_runtime = None

    def __init__(self, config: Config) -> None:
        self.active_version = config_signature(config)
        self.reload_requests: list[Config] = []

    def get_status(self) -> RuntimeStatusSnapshot:
        return RuntimeStatusSnapshot(
            state=RuntimeState.IDLE,
            generation=1,
            reload_in_progress=False,
            active_config_version=self.active_version,
            last_reload_at=None,
            last_reload_error=None,
        )

    def request_reload(self, config: Config) -> RuntimeReloadRequest:
        self.reload_requests.append(config)
        raise AssertionError("Pending storage changes must not reach worker reload")


@pytest.mark.parametrize("mutation", ["create", "update", "delete"])
def test_camera_save_acknowledged_when_pending_storage_requires_process_restart(
    tmp_path: Path, mutation: str
) -> None:
    # Given: A real Application with an active runtime snapshot and no database services started
    manager = _manager(tmp_path)
    payload = yaml.safe_load(manager.config_path.read_text())
    payload["alert_policy"]["config"]["overrides"] = {}
    manager.config_path.write_text(yaml.safe_dump(payload))
    active_config = manager.get_config()
    runtime = _ActiveSettingsRuntime(active_config)
    app = Application(manager.config_path)
    app._config = active_config
    app._parent_config = active_config.model_copy(deep=True)
    app._runtime_manager = cast(RuntimeManager, runtime)
    api = create_contract_app()
    api.state.homesec = app
    client = TestClient(api)
    storage_save = client.patch(
        "/api/v1/config",
        json={
            "expected_config_version": config_signature(active_config),
            "storage": {"config": {"root": "/tmp/pending-storage"}},
        },
    )
    assert storage_save.status_code == 200
    assert storage_save.json()["apply_required"] == "restart"

    # When: A camera save requests immediate application of the whole saved document
    match mutation:
        case "create":
            response = client.post(
                "/api/v1/cameras?apply_changes=true",
                json={
                    "name": "back",
                    "source_backend": "local_folder",
                    "source_config": {"watch_dir": "/tmp/back"},
                },
            )
        case "update":
            response = client.patch(
                "/api/v1/cameras/front?apply_changes=true", json={"enabled": False}
            )
        case _:
            response = client.delete("/api/v1/cameras/front?apply_changes=true")

    # Then: The successful write is acknowledged, while the pending restart remains explicit
    assert response.status_code == (201 if mutation == "create" else 200)
    saved = app.config_manager.get_config()
    result = response.json()
    assert saved.storage.config["root"] == "/tmp/pending-storage"
    assert result["restart_required"] is True
    assert result["runtime_reload"] is None
    assert result["apply_error"]["error_code"] == "CONFIG_RESTART_REQUIRED"
    assert "process restart" in result["apply_error"]["detail"]
    assert runtime.reload_requests == []
    application_status = app.get_config_application_status(saved)
    assert application_status.apply_required == "restart"
    assert application_status.active_config_version == config_signature(active_config)
    match mutation:
        case "create":
            assert [camera.name for camera in saved.cameras] == ["front", "back"]
            assert result["camera"]["name"] == "back"
        case "update":
            assert saved.cameras[0].enabled is False
            assert result["camera"]["enabled"] is False
        case _:
            assert saved.cameras == []
            assert result["camera"] is None


@pytest.mark.parametrize(
    "section",
    [
        {"storage": {"paths": {"clips_dir": ["private-input"]}}},
        {"storage": {"config": {"root": ["private-input"]}}},
    ],
)
def test_config_api_validation_errors_do_not_echo_submitted_inputs(
    tmp_path: Path, section: dict[str, object]
) -> None:
    # Given: Invalid config inputs that contain a private value
    manager = _manager(tmp_path)
    client, _ = _client(manager)

    # When: Request or plugin configuration validation fails
    response = client.patch(
        "/api/v1/config",
        json={"expected_config_version": config_signature(manager.get_config()), **section},
    )

    # Then: Neither validation path reflects the input into the error response
    assert response.status_code == 422
    assert "private-input" not in response.text


def test_config_api_read_only_save_has_actionable_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given: A config directory that refuses writes
    manager = _manager(tmp_path)
    client, _ = _client(manager)
    before = manager.config_path.read_bytes()

    def refuse_replace(source: Path, destination: Path) -> None:
        raise PermissionError("private-path")

    monkeypatch.setattr("homesec.config.manager.os.replace", refuse_replace)

    # When: Saving settings through the API
    response = client.patch(
        "/api/v1/config",
        json={
            "expected_config_version": config_signature(manager.get_config()),
            "filter": {"config": {"min_confidence": 0.8}},
        },
    )

    # Then: The API returns a stable, safe error while retaining saved state
    assert response.status_code == 503
    assert response.json()["error_code"] == "CONFIG_SAVE_FAILED"
    assert "writable" in response.json()["detail"]
    assert "private-path" not in response.text
    assert manager.config_path.read_bytes() == before
