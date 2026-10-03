"""Behavioral coverage for write-only managed credentials and safe persistence."""

from __future__ import annotations

import json
import os
import stat
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import pytest
import yaml
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field

import homesec.plugins.registry as plugin_registry
from homesec.api.server import create_contract_app
from homesec.config.credentials import (
    credential_references,
    credential_status,
    load_managed_credentials,
    managed_credentials_path,
)
from homesec.config.errors import (
    ConfigApplyInProgressError,
    ConfigPatchInvalidError,
    ConfigSaveError,
    ConfigVersionConflictError,
    CredentialStoreError,
)
from homesec.config.loader import config_signature, load_config
from homesec.config.manager import ConfigManager, ConfigPatch
from homesec.models.config import Config, FastAPIServerConfig
from homesec.plugins.registry import PluginRegistry, PluginType

_VALUE = "test-only-private-token-$()\n'\""


def _manager(tmp_path: Path) -> ConfigManager:
    payload = {
        "cameras": [],
        "storage": {"backend": "dropbox", "config": {"root": "/homecam"}},
        "state_store": {"dsn": "postgresql://unused:unused@localhost/unused"},
        "filter": {"backend": "yolo", "config": {}},
        "alert_policy": {"backend": "default", "config": {}},
        "vlm": {
            "backend": "openai",
            "run_mode": "never",
            "config": {"api_key_env": "OPENAI_API_KEY", "model": "gpt-4o"},
        },
        "notifiers": [
            {"backend": "mqtt", "config": {"host": "mqtt.local"}},
            {"backend": "mqtt", "enabled": False, "config": {"host": "other.local"}},
            {
                "backend": "sendgrid_email",
                "config": {"from_email": "from@example.test", "to_emails": ["to@example.test"]},
            },
        ],
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return ConfigManager(path)


def _patch(manager: ConfigManager, **edits: object) -> ConfigPatch:
    return ConfigPatch.model_validate(
        {"expected_config_version": config_signature(manager.get_config()), **edits}
    )


class _CredentialApp:
    def __init__(self, manager: ConfigManager, *, auth_enabled: bool) -> None:
        self.config_manager = manager
        self.server_config = FastAPIServerConfig(
            auth_enabled=auth_enabled,
            api_key_env="HOMESEC_TEST_API_KEY" if auth_enabled else None,
        )
        self.bootstrap_mode = False
        self.restart_requested = False
        self.active_version = config_signature(manager.get_config())

    def get_config_application_status(self, config: Config) -> SimpleNamespace:
        version = config_signature(config)
        return SimpleNamespace(
            saved_config_version=version,
            active_config_version=self.active_version,
            apply_required="none" if version == self.active_version else "restart",
        )


def _client(manager: ConfigManager, *, auth_enabled: bool = True) -> TestClient:
    api = create_contract_app()
    api.state.homesec = _CredentialApp(manager, auth_enabled=auth_enabled)
    return TestClient(api)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path",
    [
        "storage.config.token_env",
        "storage.config.app_key_env",
        "storage.config.app_secret_env",
        "storage.config.refresh_token_env",
        "vlm.config.api_key_env",
        "notifiers.0.config.auth.username_env",
        "notifiers.0.config.auth.password_env",
        "notifiers.1.config.auth.password_env",
        "notifiers.2.config.api_key_env",
    ],
)
async def test_credential_save_uses_private_file_and_new_yaml_reference(
    tmp_path: Path, path: str
) -> None:
    # Given: An existing configured backend and its original YAML revision
    manager = _manager(tmp_path)
    original = manager.config_path.read_bytes()
    before = config_signature(manager.get_config())

    # When: Saving a credential for a plugin-owned field, including absent optional MQTT auth
    saved = await manager.patch_config(_patch(manager, credentials={path: _VALUE}))
    reference = credential_references(saved)[path]
    assert reference is not None
    private_path = managed_credentials_path(manager.config_path)

    # Then: YAML holds a fresh reference, only the private data file has the value, and env is untouched
    assert reference.startswith("HOMESEC_SECRET_")
    assert config_signature(saved) != before
    assert _VALUE not in manager.config_path.read_text()
    assert Path(str(manager.config_path) + ".bak").read_bytes() == original
    assert load_managed_credentials(manager.config_path, saved) == {reference: _VALUE}
    assert reference not in os.environ
    assert credential_status(manager.config_path, saved)[path].configured is True
    assert credential_status(manager.config_path, saved)[path].source == "managed"
    if os.name == "posix":
        assert stat.S_IMODE(private_path.stat().st_mode) == 0o600
        assert stat.S_IMODE(private_path.parent.stat().st_mode) == 0o700


@pytest.mark.asyncio
async def test_replacing_and_clearing_credentials_preserves_old_snapshots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given: A saved credential that also belongs to an older active snapshot
    manager = _manager(tmp_path)
    path = "vlm.config.api_key_env"
    first = await manager.patch_config(_patch(manager, credentials={path: _VALUE}))
    first_ref = credential_references(first)[path]

    # When: Replacing with the same value, then clearing explicitly despite a hostile host fallback
    second = await manager.patch_config(_patch(manager, credentials={path: _VALUE}))
    cleared = await manager.patch_config(_patch(manager, credentials={path: None}))
    cleared_ref = credential_references(cleared)[path]
    assert cleared_ref is not None
    monkeypatch.setenv(cleared_ref, "must-not-be-used")

    # Then: Every operation changes YAML revision, clear never falls back, and old snapshots still resolve
    assert config_signature(first) != config_signature(second) != config_signature(cleared)
    assert load_managed_credentials(manager.config_path, first) == {first_ref: _VALUE}
    assert load_managed_credentials(manager.config_path, second)
    assert load_managed_credentials(manager.config_path, cleared) == {}
    assert credential_status(manager.config_path, cleared)[path].configured is False
    assert credential_status(manager.config_path, cleared)[path].source == "managed"
    backup = load_config(Path(str(manager.config_path) + ".bak"))
    assert load_managed_credentials(manager.config_path, backup) == load_managed_credentials(
        manager.config_path, second
    )


@pytest.mark.asyncio
async def test_environment_reference_edit_restores_external_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given: A managed API key and an independently configured host environment key
    manager = _manager(tmp_path)
    path = "vlm.config.api_key_env"
    await manager.patch_config(_patch(manager, credentials={path: _VALUE}))
    monkeypatch.setenv("EXTERNAL_OPENAI_KEY", "host-key")

    # When: Choosing the external environment reference through the existing advanced config patch
    saved = await manager.patch_config(
        _patch(manager, vlm={"config": {"api_key_env": "EXTERNAL_OPENAI_KEY"}})
    )

    # Then: The host key remains untouched and status truthfully reflects external mode
    assert load_managed_credentials(manager.config_path, saved) == {}
    assert credential_status(manager.config_path, saved)[path].source == "environment"
    assert credential_status(manager.config_path, saved)[path].configured is True
    assert os.environ["EXTERNAL_OPENAI_KEY"] == "host-key"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "edits",
    [
        {"credentials": {"server.api_key_env": _VALUE}},
        {"credentials": {"storage.config.root": _VALUE}},
        {"credentials": {"notifiers.7.config.api_key_env": _VALUE}},
        {"credentials": {"notifiers.-1.config.api_key_env": _VALUE}},
        {"credentials": {"vlm.config.api_key_env": ""}},
        {"credentials": {"vlm.config.api_key_env": "***redacted***"}},
        {
            "credentials": {"vlm.config.api_key_env": _VALUE},
            "filter": {"config": {"min_confidence": 9}},
        },
        {"vlm": {"config": {"api_key_env": "HOMESEC_SECRET_" + "a" * 32}}},
    ],
)
async def test_unsupported_or_invalid_credentials_never_write(
    tmp_path: Path, edits: dict[str, object]
) -> None:
    # Given: Original YAML without a managed credential file
    manager = _manager(tmp_path)
    before = manager.config_path.read_bytes()

    # When: Submitting an unsupported path, invalid candidate, empty value, or reserved-reference injection
    with pytest.raises(ConfigPatchInvalidError) as exc:
        await manager.patch_config(_patch(manager, **edits))

    # Then: Neither persistence target changes and the safe error contains no submitted secret
    assert _VALUE not in str(exc.value)
    assert manager.config_path.read_bytes() == before
    assert not managed_credentials_path(manager.config_path).exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("rejection", ["stale", "frozen"])
async def test_rejected_revision_or_restart_never_writes_credentials(
    tmp_path: Path, rejection: str
) -> None:
    # Given: A valid credential draft that can no longer be safely accepted
    manager = _manager(tmp_path)
    patch = _patch(manager, credentials={"vlm.config.api_key_env": _VALUE})
    if rejection == "stale":
        await manager.patch_config(_patch(manager, vlm={"config": {"model": "new-model"}}))
        expected_error = ConfigVersionConflictError
    else:
        manager.freeze_mutations()
        expected_error = ConfigApplyInProgressError
    before = manager.config_path.read_bytes()

    # When: Saving the rejected draft
    with pytest.raises(expected_error):
        await manager.patch_config(patch)

    # Then: The private file is not created and current YAML remains intact
    assert manager.config_path.read_bytes() == before
    assert not managed_credentials_path(manager.config_path).exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("failed_file", ["credentials", "config"])
async def test_atomic_save_failure_keeps_previous_credential_usable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failed_file: str
) -> None:
    # Given: A usable saved credential and a filesystem boundary that rejects one commit
    manager = _manager(tmp_path)
    first = await manager.patch_config(
        _patch(manager, credentials={"vlm.config.api_key_env": _VALUE})
    )
    before = manager.config_path.read_bytes()
    target = (
        managed_credentials_path(manager.config_path)
        if failed_file == "credentials"
        else manager.config_path
    )
    original_replace = os.replace

    def refuse_target(source: Path, destination: Path) -> None:
        if destination == target:
            raise PermissionError("filesystem refuses replacement")
        original_replace(source, destination)

    monkeypatch.setattr("homesec.config.credentials.os.replace", refuse_target)

    # When: Rotating the credential and the private file or YAML commit fails
    expected_error = CredentialStoreError if failed_file == "credentials" else ConfigSaveError
    with pytest.raises(expected_error):
        await manager.patch_config(
            _patch(manager, credentials={"vlm.config.api_key_env": "new-token"})
        )

    # Then: No false save success occurs, YAML stays old, and its credential still resolves
    assert manager.config_path.read_bytes() == before
    assert load_managed_credentials(manager.config_path, first) == {
        credential_references(first)["vlm.config.api_key_env"]: _VALUE
    }
    assert not list(managed_credentials_path(manager.config_path).parent.glob("credentials-*.tmp"))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "unsafe_target",
    [
        "directory_symlink",
        "dangling_directory_symlink",
        "file_symlink",
        "directory_mode",
        "file_mode",
    ],
)
async def test_private_credentials_reject_unsafe_paths(tmp_path: Path, unsafe_target: str) -> None:
    # Given: A saved secret whose private path was replaced or made permissive
    if os.name != "posix" and unsafe_target.endswith("mode"):
        pytest.skip("POSIX permission contract")
    manager = _manager(tmp_path)
    saved = await manager.patch_config(
        _patch(manager, credentials={"vlm.config.api_key_env": _VALUE})
    )
    private_path = managed_credentials_path(manager.config_path)
    if unsafe_target in {"directory_symlink", "dangling_directory_symlink"}:
        moved = tmp_path / "moved-private"
        private_path.parent.rename(moved)
        private_path.parent.symlink_to(moved, target_is_directory=True)
        if unsafe_target == "dangling_directory_symlink":
            (moved / "credentials.json").unlink()
            moved.rmdir()
    elif unsafe_target == "file_symlink":
        moved = private_path.with_name("moved.json")
        private_path.rename(moved)
        private_path.symlink_to(moved)
    elif unsafe_target == "directory_mode":
        private_path.parent.chmod(0o755)
    else:
        private_path.chmod(0o644)

    # When: Reading saved status or installing values for a fresh runtime
    with pytest.raises(CredentialStoreError) as exc:
        load_managed_credentials(manager.config_path, saved)

    # Then: It refuses the unsafe source instead of following it or falling back to host env
    assert _VALUE not in str(exc.value)
    assert "permissions" in str(exc.value)


def test_credential_api_requires_enabled_auth_and_valid_bearer_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given: The same configured install exposed with and without API authentication
    manager = _manager(tmp_path)
    monkeypatch.setenv("HOMESEC_TEST_API_KEY", "test-api-key")
    patch = _patch(manager, credentials={"vlm.config.api_key_env": _VALUE})
    request = {
        "expected_config_version": patch.expected_config_version,
        "credentials": {"vlm.config.api_key_env": _VALUE},
    }
    open_client = _client(manager, auth_enabled=False)
    authenticated_client = _client(manager)

    # When: Attempting writes with auth disabled, no token, wrong token, and a valid token
    disabled = open_client.patch("/api/v1/config", json=request)
    missing = authenticated_client.patch("/api/v1/config", json=request)
    wrong = authenticated_client.patch(
        "/api/v1/config", json=request, headers={"Authorization": "Bearer wrong"}
    )
    accepted = authenticated_client.patch(
        "/api/v1/config", json=request, headers={"Authorization": "Bearer test-api-key"}
    )

    # Then: Only the authenticated write succeeds, and values are absent from every response
    assert disabled.status_code == 403
    assert disabled.json()["error_code"] == "CREDENTIALS_AUTH_REQUIRED"
    assert missing.status_code == wrong.status_code == 401
    assert accepted.status_code == 200
    for response in (disabled, missing, wrong, accepted):
        assert _VALUE not in response.text
    payload = accepted.json()
    assert payload["credentials_editable"] is True
    assert payload["credentials"]["vlm.config.api_key_env"] == {
        "configured": True,
        "source": "managed",
    }


def test_config_response_status_includes_default_and_optional_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given: A read-only credential UI with a populated external API key and absent MQTT auth
    manager = _manager(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", _VALUE)
    monkeypatch.delenv("DROPBOX_TOKEN", raising=False)

    # When: Reading configuration while API authentication is disabled
    response = _client(manager, auth_enabled=False).get("/api/v1/config")

    # Then: Presence metadata includes plugin defaults but no credential values
    assert response.status_code == 200
    payload = response.json()
    assert payload["credentials_editable"] is False
    assert payload["credentials"]["vlm.config.api_key_env"] == {
        "configured": True,
        "source": "environment",
    }
    assert payload["credentials"]["storage.config.token_env"] == {
        "configured": False,
        "source": "environment",
    }
    assert payload["credentials"]["notifiers.0.config.auth.password_env"] == {
        "configured": False,
        "source": "environment",
    }
    assert _VALUE not in response.text


@pytest.mark.asyncio
async def test_unreadable_private_file_returns_safe_api_error(tmp_path: Path) -> None:
    # Given: A saved managed credential with an invalid private document
    manager = _manager(tmp_path)
    await manager.patch_config(_patch(manager, credentials={"vlm.config.api_key_env": _VALUE}))
    private_path = managed_credentials_path(manager.config_path)
    private_path.write_text('{"values": {"HOMESEC_SECRET_test": {"secret": "private-file-input"}}}')

    # When: Reading configuration metadata
    response = _client(manager, auth_enabled=False).get("/api/v1/config")

    # Then: The API reports a stable file error without reflecting private input
    assert response.status_code == 503
    assert response.json()["error_code"] == "CONFIG_CREDENTIALS_UNAVAILABLE"
    assert "private-file-input" not in response.text
    with pytest.raises(CredentialStoreError) as exc:
        load_managed_credentials(manager.config_path, manager.get_config())
    assert "private-file-input" not in "".join(traceback.format_exception(exc.value))


@pytest.mark.asyncio
@pytest.mark.parametrize("value", ["private-nul\0value", "private-surrogate-\ud800"])
async def test_environment_incompatible_credentials_never_persist(
    tmp_path: Path, value: str
) -> None:
    # Given: A credential draft that cannot be installed in a process environment
    manager = _manager(tmp_path)
    before = manager.config_path.read_bytes()

    # When: Saving through the same manager boundary used by authenticated API writes
    with pytest.raises(ConfigPatchInvalidError) as error:
        await manager.patch_config(_patch(manager, credentials={"vlm.config.api_key_env": value}))

    # Then: No YAML, backup, or private file is written and the error does not reflect the value
    assert manager.config_path.read_bytes() == before
    assert not Path(str(manager.config_path) + ".bak").exists()
    assert not managed_credentials_path(manager.config_path).exists()
    assert value not in "".join(traceback.format_exception(error.value))


@pytest.mark.asyncio
@pytest.mark.parametrize("value", ["private-nul\0value", "private-surrogate-\ud800"])
async def test_malformed_private_values_fail_safely_before_startup(
    tmp_path: Path, value: str
) -> None:
    # Given: A valid saved reference whose private document was corrupted locally
    manager = _manager(tmp_path)
    saved = await manager.patch_config(
        _patch(manager, credentials={"vlm.config.api_key_env": _VALUE})
    )
    reference = credential_references(saved)["vlm.config.api_key_env"]
    managed_credentials_path(manager.config_path).write_text(
        json.dumps({"version": 1, "values": {reference: value}}), encoding="utf-8"
    )

    # When: Loading the startup credential snapshot or reading API status
    with pytest.raises(CredentialStoreError) as error:
        load_managed_credentials(manager.config_path, saved)
    response = _client(manager, auth_enabled=False).get("/api/v1/config")

    # Then: Both paths report the safe credential-store error without exposing private content
    assert response.status_code == 503
    assert response.json()["error_code"] == "CONFIG_CREDENTIALS_UNAVAILABLE"
    assert value not in "".join(traceback.format_exception(error.value))
    assert "private-nul" not in response.text
    assert "private-surrogate" not in response.text


class _FirstAuth(BaseModel):
    kind: Literal["first"]
    first_env: str = Field(default="FIRST_KEY", json_schema_extra={"homesec_credential": True})


class _SecondAuth(BaseModel):
    kind: Literal["second"]
    second_env: str = Field(default="SECOND_KEY", json_schema_extra={"homesec_credential": True})


class _UnionPluginConfig(BaseModel):
    auth: _FirstAuth | _SecondAuth | None = None


class _UnionPlugin:
    config_cls = _UnionPluginConfig

    @classmethod
    def create(cls, config: _UnionPluginConfig) -> object:
        raise AssertionError("Credential discovery must not construct a provider")


@pytest.mark.parametrize("kind", ["first", "second", None])
def test_external_plugin_credentials_follow_actual_union_branch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str | None
) -> None:
    # Given: An externally registered plugin with mutually exclusive nested credential schemas
    registry = PluginRegistry[_UnionPluginConfig, object](PluginType.STORAGE)
    registry.register("external-union", _UnionPlugin)
    config = _manager(tmp_path).get_config()
    monkeypatch.setitem(plugin_registry._REGISTRIES, PluginType.STORAGE, registry)
    config.storage.backend = "external-union"
    config.storage.config = {"auth": {"kind": kind}} if kind else {}

    # When: Discovering credential slots through the registry's actual validated model
    references = credential_references(config)
    storage_references = {
        path: ref for path, ref in references.items() if path.startswith("storage.")
    }

    # Then: Only the selected branch is traversed, and absent ambiguous auth invents no branch
    assert storage_references == (
        {f"storage.config.auth.{kind}_env": f"{kind.upper()}_KEY"} if kind else {}
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("reference", ["private-ref\0bad", "private-ref\ud800"])
@pytest.mark.parametrize("section", ["storage", "vlm", "mqtt"])
async def test_incompatible_changed_environment_references_never_write(
    tmp_path: Path, reference: str, section: str
) -> None:
    # Given: A valid config and an edited plugin credential reference incompatible with getenv
    manager = _manager(tmp_path)
    original = manager.config_path.read_bytes()
    edits: dict[str, object]
    if section == "storage":
        edits = {"storage": {"config": {"refresh_token_env": reference}}}
    elif section == "vlm":
        edits = {"vlm": {"config": {"api_key_env": reference}}}
    else:
        edits = {"notifiers": [{"index": 0, "config": {"auth": {"password_env": reference}}}]}

    # When: Saving a direct config reference rather than a write-only credential
    with pytest.raises(ConfigPatchInvalidError) as error:
        await manager.patch_config(_patch(manager, **edits))

    # Then: The error is safe and all persistence targets remain unchanged
    assert manager.config_path.read_bytes() == original
    assert not Path(str(manager.config_path) + ".bak").exists()
    assert not managed_credentials_path(manager.config_path).exists()
    assert "private-ref" not in "".join(traceback.format_exception(error.value))


@pytest.mark.parametrize("reference", ["private-ref\0bad", "private-ref\ud800"])
def test_incompatible_legacy_environment_references_have_safe_read_errors(
    tmp_path: Path, reference: str
) -> None:
    # Given: Malformed preexisting YAML contains a plugin-owned environment reference
    manager = _manager(tmp_path)
    payload = yaml.safe_load(manager.config_path.read_text())
    payload["vlm"]["config"]["api_key_env"] = reference
    manager.config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    config = manager.get_config()

    # When: Reading availability for the malformed reference
    response = _client(manager, auth_enabled=False).get("/api/v1/config")
    with pytest.raises(CredentialStoreError) as error:
        credential_status(manager.config_path, config)

    # Then: Status returns a typed, value-free failure without making managed-only startup fail
    assert response.status_code == 503
    assert response.json()["error_code"] == "CONFIG_CREDENTIALS_UNAVAILABLE"
    assert "private-ref" not in response.text
    assert "private-ref" not in "".join(traceback.format_exception(error.value))
    assert error.value.__cause__ is None
    assert load_managed_credentials(manager.config_path, config) == {}


@pytest.mark.asyncio
async def test_unrelated_edits_preserve_legacy_hyphenated_environment_references(
    tmp_path: Path,
) -> None:
    # Given: A valid deployment uses a host env name outside the guided UI identifier syntax
    manager = _manager(tmp_path)
    payload = yaml.safe_load(manager.config_path.read_text())
    payload["storage"]["config"]["token_env"] = "DEPLOYMENT-TOKEN"
    manager.config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    # When: Saving a storage root change without editing that credential reference
    saved = await manager.patch_config(_patch(manager, storage={"config": {"root": "/new-root"}}))

    # Then: The external reference is preserved and status can be read safely
    assert credential_references(saved)["storage.config.token_env"] == "DEPLOYMENT-TOKEN"
    assert (
        credential_status(manager.config_path, saved)["storage.config.token_env"].source
        == "environment"
    )
