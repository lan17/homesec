"""Tests for configuration loading and validation."""

import os
import tempfile
from pathlib import Path

import pytest
from pydantic import BaseModel

from homesec.config import (
    ConfigError,
    load_config,
    load_config_from_dict,
    resolve_env_var,
    validate_camera_references,
    validate_config,
    validate_plugin_names,
)
from homesec.config.loader import ConfigErrorCode
from homesec.models.config import Config
from homesec.plugins.registry import PluginType, plugin


class CustomFilterConfig(BaseModel):
    model_config = {"extra": "allow"}


@plugin(plugin_type=PluginType.FILTER, name="custom_filter")
class CustomFilterPlugin:
    config_cls = CustomFilterConfig

    @classmethod
    def create(cls, config: CustomFilterConfig) -> object:
        return object()


class CustomVLMConfig(BaseModel):
    model_config = {"extra": "allow"}


@plugin(plugin_type=PluginType.ANALYZER, name="custom_vlm")
class CustomVLMPlugin:
    config_cls = CustomVLMConfig

    @classmethod
    def create(cls, config: CustomVLMConfig) -> object:
        return object()


def minimal_config() -> dict[str, object]:
    """Return minimal valid config dict."""
    return {
        "version": 1,
        "cameras": [
            {
                "name": "front_door",
                "source": {
                    "backend": "local_folder",
                    "config": {
                        "watch_dir": "recordings",
                        "poll_interval": 1.0,
                    },
                },
            }
        ],
        "storage": {
            "backend": "dropbox",
            "config": {
                "root": "/homecam",
            },
        },
        "state_store": {
            "dsn": "postgresql://user:pass@localhost/db",
        },
        "notifiers": [
            {
                "backend": "mqtt",
                "config": {
                    "host": "localhost",
                    "port": 1883,
                },
            }
        ],
        "filter": {
            "backend": "yolo",
            "config": {},
        },
        "vlm": {
            "backend": "openai",
            "config": {
                "api_key_env": "OPENAI_API_KEY",
                "model": "gpt-4o",
            },
        },
        "alert_policy": {
            "backend": "default",
            "enabled": True,
            "config": {
                "min_risk_level": "medium",
            },
        },
    }


def test_load_config_from_dict_success() -> None:
    """Test loading valid config from dict."""
    # Given a minimal valid config dict
    # When loading the config
    config = load_config_from_dict(minimal_config())

    # Then it loads successfully
    assert config.filter.backend == "yolo"
    assert isinstance(config.filter.config, dict)
    assert config.vlm.backend == "openai"
    assert config.alert_policy.backend == "default"
    assert config.alert_policy.config["min_risk_level"] == "medium"


def test_load_config_from_dict_missing_required_field() -> None:
    """Test that missing required field raises ConfigError."""
    # Given a config missing required sections
    invalid = {"filter": {"backend": "yolo"}}  # Missing vlm and alert_policy

    # When loading the config
    with pytest.raises(ConfigError) as exc_info:
        load_config_from_dict(invalid)  # type: ignore[arg-type]

    # Then a validation error is raised
    assert "validation failed" in str(exc_info.value).lower()
    assert exc_info.value.code is ConfigErrorCode.VALIDATION_FAILED


def test_load_config_from_dict_invalid_risk_level() -> None:
    """Test that invalid enum value raises ConfigError."""
    # Given a config with an invalid risk level
    data = minimal_config()
    data["alert_policy"] = {
        "backend": "default",
        "config": {"min_risk_level": "extreme"},
    }  # Invalid value

    # When loading the config
    with pytest.raises(ConfigError) as exc_info:
        load_config_from_dict(data)  # type: ignore[arg-type]

    # Then a validation error references the field
    assert exc_info.value.code is ConfigErrorCode.PLUGIN_CONFIG_INVALID
    assert "min_risk_level" in str(exc_info.value)


def test_load_config_from_yaml_file() -> None:
    """Test loading config from YAML file."""
    # Given a valid YAML config file
    yaml_content = """
cameras:
  - name: front_door
    source:
      backend: local_folder
      config:
        watch_dir: recordings
        poll_interval: 1.0

storage:
  backend: dropbox
  config:
    root: /homecam

state_store:
  dsn: postgresql://user:pass@localhost/db

notifiers:
  - backend: mqtt
    config:
      host: localhost

filter:
  backend: yolo
  config: {}

vlm:
  backend: openai
  config:
    api_key_env: OPENAI_API_KEY
    model: gpt-4o

alert_policy:
  backend: default
  config:
    min_risk_level: low
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        f.flush()
        path = Path(f.name)

    try:
        # When loading the config
        config = load_config(path)
        # Then fields are parsed and validated
        assert config.cameras[0].name == "front_door"
        assert config.storage.backend == "dropbox"
        assert config.filter.backend == "yolo"
        assert isinstance(config.filter.config, dict)
        assert config.vlm.backend == "openai"
        assert config.alert_policy.config["min_risk_level"] == "low"
    finally:
        path.unlink()


def test_load_config_warns_when_permissions_are_too_open(caplog: pytest.LogCaptureFixture) -> None:
    """Loading config should warn when file permissions expose secrets."""
    if os.name != "posix":
        pytest.skip("Permission warnings are POSIX-specific")

    # Given a valid config file with permissive mode
    yaml_content = """
cameras:
  - name: front_door
    source:
      backend: local_folder
      config:
        watch_dir: recordings
        poll_interval: 1.0

storage:
  backend: dropbox
  config:
    root: /homecam

state_store:
  dsn: postgresql://user:pass@localhost/db

notifiers:
  - backend: mqtt
    config:
      host: localhost

filter:
  backend: yolo
  config: {}

vlm:
  backend: openai
  config:
    api_key_env: OPENAI_API_KEY
    model: gpt-4o

alert_policy:
  backend: default
  config:
    min_risk_level: low
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        f.flush()
        path = Path(f.name)
    os.chmod(path, 0o644)

    try:
        # When loading the config
        with caplog.at_level("WARNING"):
            config = load_config(path)

        # Then loading succeeds and a permissions warning is emitted
        assert config.cameras[0].name == "front_door"
        assert any("permissions are too permissive" in rec.message for rec in caplog.records)
    finally:
        path.unlink()


def test_load_config_does_not_warn_when_permissions_are_restrictive(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Loading config should not warn when file permissions are already restrictive."""
    if os.name != "posix":
        pytest.skip("Permission warnings are POSIX-specific")

    # Given a valid config file with restrictive mode
    yaml_content = """
cameras:
  - name: front_door
    source:
      backend: local_folder
      config:
        watch_dir: recordings
        poll_interval: 1.0

storage:
  backend: dropbox
  config:
    root: /homecam

state_store:
  dsn: postgresql://user:pass@localhost/db

notifiers:
  - backend: mqtt
    config:
      host: localhost

filter:
  backend: yolo
  config: {}

vlm:
  backend: openai
  config:
    api_key_env: OPENAI_API_KEY
    model: gpt-4o

alert_policy:
  backend: default
  config:
    min_risk_level: low
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        f.flush()
        path = Path(f.name)
    os.chmod(path, 0o600)

    try:
        # When loading the config
        with caplog.at_level("WARNING"):
            config = load_config(path)

        # Then loading succeeds without a permissions warning
        assert config.cameras[0].name == "front_door"
        assert all("permissions are too permissive" not in rec.message for rec in caplog.records)
    finally:
        path.unlink()


def test_load_config_file_not_found() -> None:
    """Test that missing file raises ConfigError."""
    # Given a nonexistent path
    with pytest.raises(ConfigError) as exc_info:
        # When loading the config
        load_config(Path("/nonexistent/config.yaml"))

    # Then a not found error is raised
    assert "not found" in str(exc_info.value).lower()
    assert exc_info.value.code is ConfigErrorCode.FILE_NOT_FOUND


def test_load_config_invalid_yaml() -> None:
    """Test that invalid YAML raises ConfigError."""
    # Given a malformed YAML file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("invalid: yaml: content: [")
        f.flush()
        path = Path(f.name)

    try:
        # When loading the config
        with pytest.raises(ConfigError) as exc_info:
            load_config(path)
        # Then an invalid YAML error is raised
        assert "invalid yaml" in str(exc_info.value).lower()
        assert exc_info.value.code is ConfigErrorCode.YAML_INVALID
    finally:
        path.unlink()


def test_load_config_empty_file() -> None:
    """Test that empty file raises ConfigError."""
    # Given an empty YAML file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("")
        f.flush()
        path = Path(f.name)

    try:
        # When loading the config
        with pytest.raises(ConfigError) as exc_info:
            load_config(path)
        # Then an empty file error is raised
        assert "empty" in str(exc_info.value).lower()
        assert exc_info.value.code is ConfigErrorCode.EMPTY_FILE
    finally:
        path.unlink()


def test_load_config_rejects_legacy_health_block() -> None:
    """Config loading should fail fast when legacy health block is provided."""
    # Given a config payload still using retired health server settings
    data = minimal_config()
    data["health"] = {
        "host": "0.0.0.0",
        "port": 8080,
    }

    # When loading the config
    with pytest.raises(ConfigError) as exc_info:
        load_config_from_dict(data)  # type: ignore[arg-type]

    # Then validation fails and surfaces the unknown key
    assert exc_info.value.code is ConfigErrorCode.VALIDATION_FAILED
    assert "health" in str(exc_info.value)


def test_per_camera_override_merge() -> None:
    """Test that per-camera overrides are preserved in config."""
    # Given a config with per-camera alert overrides
    data = minimal_config()
    data["alert_policy"]["config"]["overrides"] = {
        "front_door": {
            "min_risk_level": "low",
            "notify_on_activity_types": ["delivery"],
        }
    }

    # When loading the config
    config = load_config_from_dict(data)  # type: ignore[arg-type]

    # Then overrides remain available in the alert policy config
    overrides = config.alert_policy.config["overrides"]
    assert overrides["front_door"]["min_risk_level"] == "low"
    assert overrides["front_door"]["notify_on_activity_types"] == ["delivery"]


def test_resolve_env_var_success() -> None:
    """Test resolving existing environment variable."""
    # Given an environment variable set
    os.environ["TEST_VAR_12345"] = "test_value"
    try:
        # When resolving it
        value = resolve_env_var("TEST_VAR_12345")
        # Then the value is returned
        assert value == "test_value"
    finally:
        del os.environ["TEST_VAR_12345"]


def test_resolve_env_var_missing_required() -> None:
    """Test that missing required env var raises ConfigError."""
    # Given a missing env var
    with pytest.raises(ConfigError) as exc_info:
        # When resolving as required
        resolve_env_var("DEFINITELY_NOT_SET_VAR_XYZ")

    # Then a ConfigError is raised
    assert "not set" in str(exc_info.value).lower()
    assert exc_info.value.code is ConfigErrorCode.ENV_VAR_MISSING


def test_resolve_env_var_missing_optional() -> None:
    """Test that missing optional env var returns None."""
    # Given a missing env var
    # When resolving as optional
    result = resolve_env_var("DEFINITELY_NOT_SET_VAR_XYZ", required=False)
    # Then None is returned
    assert result is None


def test_validate_camera_references_valid() -> None:
    """Test validation passes when camera names are valid."""
    # Given per-camera overrides for known cameras
    data = minimal_config()
    data["alert_policy"]["config"]["overrides"] = {"front_door": {"min_risk_level": "low"}}

    # When loading the config
    config = load_config_from_dict(data)  # type: ignore[arg-type]

    # Then validation passes
    validate_camera_references(config, ["front_door", "back_door"])


def test_validate_camera_references_invalid() -> None:
    """Test validation fails when camera names are invalid."""
    # Given per-camera overrides for unknown cameras
    data = minimal_config()
    data["alert_policy"]["config"]["overrides"] = {"unknown_camera": {"min_risk_level": "low"}}

    # When loading the config
    with pytest.raises(ConfigError) as exc_info:
        load_config_from_dict(data)  # type: ignore[arg-type]

    # Then the camera reference validation code is surfaced
    assert exc_info.value.code is ConfigErrorCode.CAMERA_REFERENCES_INVALID
    assert "unknown_camera" in str(exc_info.value)


def test_validate_config_rejects_unknown_camera_override() -> None:
    """Validate_config should reject overrides for unknown cameras."""
    # Given: A config with an override referencing an unknown camera
    data = minimal_config()
    data["alert_policy"]["config"]["overrides"] = {"unknown_camera": {"min_risk_level": "low"}}
    config = Config.model_validate(data)

    from homesec.plugins import discover_all_plugins

    discover_all_plugins()

    # When: validating the config
    with pytest.raises(ConfigError) as exc_info:
        validate_config(config)

    # Then: unknown camera is reported
    assert exc_info.value.code is ConfigErrorCode.CAMERA_REFERENCES_INVALID
    assert "unknown_camera" in str(exc_info.value)


def test_validate_config_rejects_unknown_plugin_backend() -> None:
    """validate_config should surface unknown plugin backends."""
    # Given: A config that references a missing filter backend
    data = minimal_config()
    data["filter"]["backend"] = "missing_filter"
    config = Config.model_validate(data)

    from homesec.plugins import discover_all_plugins

    discover_all_plugins()

    # When: validating the config
    with pytest.raises(ConfigError) as exc_info:
        validate_config(config)

    # Then: unknown backend is reported
    assert exc_info.value.code is ConfigErrorCode.PLUGIN_CONFIG_INVALID
    assert "missing_filter" in str(exc_info.value)


def test_validate_plugin_names_valid() -> None:
    """Test validation passes when plugin names are valid."""
    # Given a valid config
    config = load_config_from_dict(minimal_config())

    # When validating plugin names
    validate_plugin_names(
        config,
        valid_filters=["yolo"],
        valid_vlms=["openai"],
        valid_storage=["dropbox"],
        valid_notifiers=["mqtt", "sendgrid_email"],
        valid_alert_policies=["default"],
    )
    # Then validation succeeds without errors
    assert config.filter.backend == "yolo"


def test_validate_plugin_names_invalid() -> None:
    """Test validation fails when plugin names are invalid."""
    # Given a config with unsupported notifier
    config = load_config_from_dict(minimal_config())

    # When validating plugin names
    with pytest.raises(ConfigError) as exc_info:
        validate_plugin_names(
            config,
            valid_filters=["yolo"],
            valid_vlms=["openai"],
            valid_storage=["dropbox"],
            valid_notifiers=["sendgrid_email"],
            valid_alert_policies=["default"],
        )

    # Then the missing backend is reported
    assert exc_info.value.code is ConfigErrorCode.PLUGIN_NAMES_INVALID
    assert "mqtt" in str(exc_info.value)


def test_validate_plugin_names_case_insensitive() -> None:
    """Test validation allows plugin names with different casing."""
    # Given a config with uppercase plugin names
    data = minimal_config()
    data["filter"]["backend"] = "YOLO"
    data["vlm"]["backend"] = "OPENAI"
    data["storage"]["backend"] = "DROPBOX"
    data["notifiers"][0]["backend"] = "MQTT"
    data["alert_policy"]["backend"] = "DEFAULT"
    data["cameras"][0]["source"]["backend"] = "LOCAL_FOLDER"
    config = load_config_from_dict(data)  # type: ignore[arg-type]

    # When validating plugin names
    validate_plugin_names(
        config,
        valid_filters=["yolo"],
        valid_vlms=["openai"],
        valid_storage=["dropbox"],
        valid_notifiers=["mqtt", "sendgrid_email"],
        valid_alert_policies=["default"],
        valid_sources=["local_folder"],
    )

    # Then validation succeeds regardless of casing
    assert config.filter.backend == "yolo"
    assert config.vlm.backend == "openai"
    assert config.storage.backend == "dropbox"
    assert config.notifiers[0].backend == "mqtt"
    assert config.alert_policy.backend == "default"
    assert config.cameras[0].source.backend == "local_folder"


def test_load_example_config() -> None:
    """Test that the example config file loads successfully."""
    # Given an example config file on disk
    example_path = Path(__file__).parent.parent.parent / "config" / "example.yaml"
    if not example_path.exists():
        pytest.skip("Example config not found")

    # When loading the config
    config = load_config(example_path)
    # Then expected fields are present
    assert config.preview.backend == "hls"
    assert config.filter.backend == "yolo"
    assert config.vlm.backend == "openai"


def test_third_party_filter_config_preserved_through_config_load() -> None:
    """Test that third-party filter plugin configs (dicts) survive Config load.

    This tests the fix for the bug where third-party configs were silently dropped
    because FilterConfig.config accepted BaseModel which created empty objects.
    """
    # Given a config with a third-party filter plugin (not yolo)
    data = minimal_config()
    data["filter"] = {
        "backend": "custom_filter",
        "config": {
            "custom_field_1": "value1",
            "custom_field_2": 42,
            "custom_nested": {"nested_field": "nested_value"},
        },
    }

    # When loading the config
    config = load_config_from_dict(data)  # type: ignore[arg-type]

    # Then the filter config should be preserved as a dict (not empty BaseModel)
    assert isinstance(config.filter.config, dict)
    assert config.filter.config["custom_field_1"] == "value1"
    assert config.filter.config["custom_field_2"] == 42
    assert config.filter.config["custom_nested"]["nested_field"] == "nested_value"


def test_third_party_vlm_config_preserved_through_config_load() -> None:
    """Test that third-party VLM plugin configs (dicts) survive Config load.

    This tests the fix for the bug where third-party configs were silently dropped
    because VLMConfig.config accepted BaseModel which created empty objects.
    """
    # Given a config with a third-party VLM plugin (not openai)
    data = minimal_config()
    data["vlm"] = {
        "backend": "custom_vlm",
        "config": {
            "custom_api_key": "secret123",
            "custom_model": "custom-model-v1",
            "custom_setting": True,
        },
    }

    # When loading the config
    config = load_config_from_dict(data)  # type: ignore[arg-type]

    # Then the config should be preserved as a dict (not empty BaseModel)
    assert isinstance(config.vlm.config, dict)
    assert config.vlm.config["custom_api_key"] == "secret123"
    assert config.vlm.config["custom_model"] == "custom-model-v1"
    assert config.vlm.config["custom_setting"] is True


def test_server_ui_env_defaults_apply_when_fields_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Server UI dist path should default from env when config omits it."""
    # Given an env-configured UI dist path and config without explicit server block
    monkeypatch.setenv("HOMESEC_SERVER_UI_DIST_DIR", "/app/ui/dist")
    data = minimal_config()

    # When loading config
    config = load_config_from_dict(data)

    # Then server UI dist path comes from env default
    assert config.server.ui_dist_dir == "/app/ui/dist"


def test_server_ui_config_values_override_env_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit ui_dist_dir config should win over env defaults."""
    # Given env defaults and explicit ui_dist_dir config value
    monkeypatch.setenv("HOMESEC_SERVER_UI_DIST_DIR", "/app/ui/dist")
    data = minimal_config()
    data["server"] = {
        "ui_dist_dir": "./custom-ui-dist",
    }

    # When loading config
    config = load_config_from_dict(data)

    # Then explicit config value is preserved
    assert config.server.ui_dist_dir == "./custom-ui-dist"


def test_server_config_rejects_legacy_serve_ui_field() -> None:
    """Legacy `server.serve_ui` should fail validation after toggle removal."""
    # Given a config payload with removed legacy server.serve_ui key
    data = minimal_config()
    data["server"] = {
        "serve_ui": True,
        "ui_dist_dir": "./ui/dist",
    }

    # When/Then: Loading config fails with schema validation error
    with pytest.raises(ConfigError) as exc_info:
        _ = load_config_from_dict(data)

    assert exc_info.value.code == ConfigErrorCode.VALIDATION_FAILED
    assert "serve_ui" in str(exc_info.value)


def test_preview_defaults_apply_when_preview_block_is_absent() -> None:
    """Preview config should default to the accepted v1 HLS contract surface."""
    # Given a valid config payload without a preview block
    data = minimal_config()

    # When loading the config
    config = load_config_from_dict(data)

    # Then preview defaults match the v1 contract
    assert config.preview.enabled is False
    assert config.preview.backend == "hls"
    assert config.preview.token_ttl_s == 60
    assert config.preview.idle_timeout_s == 30.0
    assert config.preview.recording_policy == "stop_on_recording"
    assert config.preview.config.segment_duration_ms == 1000
    assert config.preview.config.live_window_segments == 4
    assert config.preview.config.storage_dir == Path("/tmp/homesec-preview")
    assert config.preview.config.audio_enabled is True
    assert config.preview.config.audio_codec == "auto"
    assert config.preview.config.video_codec == "auto"


def test_preview_hls_config_parses_explicit_values() -> None:
    """Preview config should parse the accepted HLS override surface."""
    # Given a config payload with explicit preview overrides
    data = minimal_config()
    data["preview"] = {
        "enabled": True,
        "backend": "HLS",
        "token_ttl_s": 120,
        "idle_timeout_s": 45.0,
        "recording_policy": "allow_during_recording",
        "config": {
            "segment_duration_ms": 1500,
            "live_window_segments": 6,
            "storage_dir": "/run/homesec-preview",
            "audio_enabled": False,
            "audio_codec": "aac",
            "video_codec": "h264",
        },
    }

    # When loading the config
    config = load_config_from_dict(data)

    # Then preview config is normalized and typed correctly
    assert config.preview.enabled is True
    assert config.preview.backend == "hls"
    assert config.preview.token_ttl_s == 120
    assert config.preview.idle_timeout_s == 45.0
    assert config.preview.recording_policy == "allow_during_recording"
    assert config.preview.config.segment_duration_ms == 1500
    assert config.preview.config.live_window_segments == 6
    assert config.preview.config.storage_dir == Path("/run/homesec-preview")
    assert config.preview.config.audio_enabled is False
    assert config.preview.config.audio_codec == "aac"
    assert config.preview.config.video_codec == "h264"


def test_preview_rejects_camera_names_that_alias_same_storage_path() -> None:
    """Preview config should reject camera names that collide after slug normalization."""
    # Given: Preview enabled with camera names that collapse to the same storage slug
    data = minimal_config()
    data["preview"] = {"enabled": True}
    data["cameras"] = [
        {
            "name": "front door",
            "source": {
                "backend": "local_folder",
                "config": {
                    "watch_dir": "/tmp/front-door",
                    "poll_interval": 1.0,
                },
            },
        },
        {
            "name": "front_door",
            "source": {
                "backend": "local_folder",
                "config": {
                    "watch_dir": "/tmp/front_door",
                    "poll_interval": 1.0,
                },
            },
        },
    ]

    # When: Loading the config
    with pytest.raises(ConfigError) as exc_info:
        load_config_from_dict(data)

    # Then: Validation rejects the aliasing preview storage paths
    assert exc_info.value.code is ConfigErrorCode.CAMERA_REFERENCES_INVALID
    assert "front door" in str(exc_info.value)
    assert "front_door" in str(exc_info.value)


def test_preview_rejects_unsupported_backend() -> None:
    """Preview config should reject deferred backends in the v1 contract."""
    # Given a config payload using the deferred MediaMTX backend and its native field
    data = minimal_config()
    data["preview"] = {
        "enabled": True,
        "backend": "mediamtx",
        "config": {
            "publish_rtsp_base_url": "rtsp://mediamtx:8554",
        },
    }

    # When loading the config
    with pytest.raises(ConfigError) as exc_info:
        load_config_from_dict(data)

    # Then validation rejects the backend before HLS-only config validation noise
    assert exc_info.value.code is ConfigErrorCode.VALIDATION_FAILED
    assert "preview.backend" in str(exc_info.value) or "preview -> backend" in str(exc_info.value)
    assert "hls" in str(exc_info.value)
    assert "publish_rtsp_base_url" not in str(exc_info.value)


def test_preview_rejects_v2_and_legacy_extra_fields() -> None:
    """Preview config should reject fields outside the accepted v1 contract."""
    # Given a config payload with deferred and legacy preview keys
    data = minimal_config()
    data["preview"] = {
        "enabled": True,
        "backend": "hls",
        "mode": "inline",
        "config": {
            "segment_duration_ms": 1000,
            "live_window_segments": 4,
            "storage_dir": "/tmp/homesec-preview",
            "audio_enabled": True,
            "audio_codec": "auto",
            "video_codec": "auto",
            "publish_rtsp_base_url": "rtsp://mediamtx:8554",
        },
    }

    # When loading the config
    with pytest.raises(ConfigError) as exc_info:
        load_config_from_dict(data)

    # Then validation surfaces the unknown v1-forbidden keys
    assert exc_info.value.code is ConfigErrorCode.VALIDATION_FAILED
    assert "mode" in str(exc_info.value)
    assert "publish_rtsp_base_url" in str(exc_info.value)
