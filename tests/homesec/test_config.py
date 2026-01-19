"""Tests for configuration loading and validation."""

import os
import tempfile
from pathlib import Path

import pytest

from homesec.config import (
    ConfigError,
    load_config,
    load_config_from_dict,
    resolve_env_var,
    validate_camera_references,
    validate_plugin_names,
)
from homesec.models.enums import RiskLevel
from homesec.models.filter import YoloFilterSettings


def minimal_config() -> dict[str, object]:
    """Return minimal valid config dict."""
    return {
        "version": 1,
        "cameras": [
            {
                "name": "front_door",
                "source": {
                    "type": "local_folder",
                    "config": {
                        "watch_dir": "recordings",
                        "poll_interval": 1.0,
                    },
                },
            }
        ],
        "storage": {
            "backend": "dropbox",
            "dropbox": {
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
            "plugin": "yolo",
            "max_workers": 1,
            "config": {},
        },
        "vlm": {
            "backend": "openai",
            "max_workers": 1,
            "llm": {
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
    assert config.filter.plugin == "yolo"
    assert config.filter.config.model_path == "yolo11n.pt"
    assert config.vlm.backend == "openai"
    assert config.alert_policy.backend == "default"
    assert config.get_default_alert_policy("front_door").min_risk_level == RiskLevel.MEDIUM


def test_load_config_from_dict_missing_required_field() -> None:
    """Test that missing required field raises ConfigError."""
    # Given a config missing required sections
    invalid = {"filter": {"plugin": "yolo"}}  # Missing vlm and alert_policy

    # When loading the config
    with pytest.raises(ConfigError) as exc_info:
        load_config_from_dict(invalid)  # type: ignore[arg-type]

    # Then a validation error is raised
    assert "validation failed" in str(exc_info.value).lower()


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
    assert "min_risk_level" in str(exc_info.value)


def test_load_config_from_yaml_file() -> None:
    """Test loading config from YAML file."""
    # Given a valid YAML config file
    yaml_content = """
cameras:
  - name: front_door
    source:
      type: local_folder
      config:
        watch_dir: recordings
        poll_interval: 1.0

storage:
  backend: dropbox
  dropbox:
    root: /homecam

state_store:
  dsn: postgresql://user:pass@localhost/db

notifiers:
  - backend: mqtt
    config:
      host: localhost

filter:
  plugin: yolo
  max_workers: 4
  config: {}

vlm:
  backend: openai
  max_workers: 2
  llm:
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
        assert config.filter.plugin == "yolo"
        assert isinstance(config.filter.config, YoloFilterSettings)
        assert config.filter.config.model_path == "yolo11n.pt"
        assert config.vlm.backend == "openai"
        assert config.get_default_alert_policy("front_door").min_risk_level == RiskLevel.LOW
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
    finally:
        path.unlink()


def test_per_camera_override_merge() -> None:
    """Test that per-camera overrides merge correctly."""
    # Given a config with per-camera alert overrides
    data = minimal_config()
    data["per_camera_alert"] = {
        "front_door": {
            "min_risk_level": "low",
            "notify_on_activity_types": ["delivery"],
        }
    }

    # When loading the config
    config = load_config_from_dict(data)  # type: ignore[arg-type]

    # Then default camera uses base config
    default_policy = config.get_default_alert_policy("unknown_camera")
    assert default_policy.min_risk_level == RiskLevel.MEDIUM
    assert default_policy.notify_on_activity_types == []

    # Then front_door uses merged config
    front_door_policy = config.get_default_alert_policy("front_door")
    assert front_door_policy.min_risk_level == RiskLevel.LOW
    assert front_door_policy.notify_on_activity_types == ["delivery"]


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
    data["per_camera_alert"] = {"front_door": {"min_risk_level": "low"}}

    # When loading the config
    config = load_config_from_dict(data)  # type: ignore[arg-type]

    # Then validation passes
    validate_camera_references(config, ["front_door", "back_door"])


def test_validate_camera_references_invalid() -> None:
    """Test validation fails when camera names are invalid."""
    # Given per-camera overrides for unknown cameras
    data = minimal_config()
    data["per_camera_alert"] = {"unknown_camera": {"min_risk_level": "low"}}

    # When loading the config
    config = load_config_from_dict(data)  # type: ignore[arg-type]

    # Then validation fails with unknown camera
    with pytest.raises(ConfigError) as exc_info:
        validate_camera_references(config, ["front_door"])

    assert "unknown_camera" in str(exc_info.value)


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
    assert config.filter.plugin == "yolo"


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
    assert "mqtt" in str(exc_info.value)


def test_validate_plugin_names_case_insensitive() -> None:
    """Test validation allows plugin names with different casing."""
    # Given a config with uppercase plugin names
    data = minimal_config()
    data["filter"]["plugin"] = "YOLO"
    data["vlm"]["backend"] = "OPENAI"
    data["storage"]["backend"] = "DROPBOX"
    data["notifiers"][0]["backend"] = "MQTT"
    data["alert_policy"]["backend"] = "DEFAULT"
    data["cameras"][0]["source"]["type"] = "LOCAL_FOLDER"
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
    assert config.filter.plugin == "yolo"
    assert config.vlm.backend == "openai"
    assert config.storage.backend == "dropbox"
    assert config.notifiers[0].backend == "mqtt"
    assert config.alert_policy.backend == "default"
    assert config.cameras[0].source.type == "local_folder"


def test_load_example_config() -> None:
    """Test that the example config file loads successfully."""
    # Given an example config file on disk
    example_path = Path(__file__).parent.parent.parent / "config" / "example.yaml"
    if not example_path.exists():
        pytest.skip("Example config not found")

    # When loading the config
    config = load_config(example_path)
    # Then expected fields are present
    assert config.filter.plugin == "yolo"
    assert config.vlm.backend == "openai"


def test_third_party_filter_config_preserved_through_config_load() -> None:
    """Test that third-party filter plugin configs (dicts) survive Config load.

    This tests the fix for the bug where third-party configs were silently dropped
    because FilterConfig.config accepted BaseModel which created empty objects.
    """
    # Given a config with a third-party filter plugin (not yolo)
    data = minimal_config()
    data["filter"] = {
        "plugin": "custom_filter",
        "max_workers": 2,
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
    because VLMConfig.llm accepted BaseModel which created empty objects.
    """
    # Given a config with a third-party VLM plugin (not openai)
    data = minimal_config()
    data["vlm"] = {
        "backend": "custom_vlm",
        "max_workers": 2,
        "llm": {
            "custom_api_key": "secret123",
            "custom_model": "custom-model-v1",
            "custom_setting": True,
        },
    }

    # When loading the config
    config = load_config_from_dict(data)  # type: ignore[arg-type]

    # Then the llm config should be preserved as a dict (not empty BaseModel)
    assert isinstance(config.vlm.llm, dict)
    assert config.vlm.llm["custom_api_key"] == "secret123"
    assert config.vlm.llm["custom_model"] == "custom-model-v1"
    assert config.vlm.llm["custom_setting"] is True
