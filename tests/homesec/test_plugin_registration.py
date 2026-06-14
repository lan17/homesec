"""Tests for plugin registration and discovery mechanisms."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, Field

from homesec.plugins.registry import (
    PluginType,
    get_plugin_names,
    load_plugin,
    plugin,
    validate_plugin,
)


class DummyConfig(BaseModel):
    foo: str = "bar"


class DummyPlugin:
    config_cls = DummyConfig

    def __init__(self, config: DummyConfig) -> None:
        self.config = config

    @classmethod
    def create(cls, config: DummyConfig) -> DummyPlugin:
        return cls(config)


@pytest.fixture
def clean_registry() -> None:
    """Ensure registry is clean for tests."""
    from homesec.plugins.registry import _REGISTRIES

    # Save state
    saved = {k: v._plugins.copy() for k, v in _REGISTRIES.items()}

    # Clear
    for reg in _REGISTRIES.values():
        reg._plugins.clear()

    yield

    # Restore
    for k, v in _REGISTRIES.items():
        v._plugins = saved[k]


def test_register_and_load_plugin(clean_registry: None) -> None:
    """Test basic registration and loading."""

    @plugin(plugin_type=PluginType.SOURCE, name="test_source")
    class TestSource(DummyPlugin):
        pass

    assert "test_source" in get_plugin_names(PluginType.SOURCE)

    instance = load_plugin(PluginType.SOURCE, "test_source", {"foo": "baz"})
    assert isinstance(instance, TestSource)
    assert instance.config.foo == "baz"


def test_validation_error(clean_registry: None) -> None:
    """Test config validation error."""

    @plugin(plugin_type=PluginType.SOURCE, name="test_source")
    class TestSource(DummyPlugin):
        pass

    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        load_plugin(PluginType.SOURCE, "test_source", {"foo": 123})  # Expecting string if strict


def test_duplicate_registration_error(clean_registry: None) -> None:
    """Test registering same name twice raises error."""

    @plugin(plugin_type=PluginType.SOURCE, name="dup")
    class P1(DummyPlugin):
        pass

    with pytest.raises(ValueError, match="already registered"):

        @plugin(plugin_type=PluginType.SOURCE, name="dup")
        class P2(DummyPlugin):
            pass


def test_unknown_plugin_error() -> None:
    """Test loading unknown plugin raises ValueError."""
    with pytest.raises(ValueError):
        load_plugin(PluginType.SOURCE, "missing_plugin", {})


def test_runtime_context_filters_unknown_keys_and_supports_aliases(
    clean_registry: None,
) -> None:
    class RuntimeConfig(BaseModel):
        model_config = {"extra": "forbid"}

        foo: str
        runtime_value: str = Field(default="", alias="__runtime_value__")

    class RuntimePlugin:
        config_cls = RuntimeConfig

        def __init__(self, config: RuntimeConfig) -> None:
            self.config = config

        @classmethod
        def create(cls, config: RuntimeConfig) -> RuntimePlugin:
            return cls(config)

    # Given: A strict plugin config with an alias-only runtime context field
    plugin(plugin_type=PluginType.SOURCE, name="runtime_source")(RuntimePlugin)

    # When: Loading and validating with both supported and unrelated runtime context
    loaded = load_plugin(
        PluginType.SOURCE,
        "runtime_source",
        {"foo": "configured"},
        __runtime_value__="injected",
        unrelated_dependency=object(),
    )
    validated = validate_plugin(
        PluginType.SOURCE,
        "runtime_source",
        {"foo": "configured"},
        __runtime_value__="injected",
        unrelated_dependency=object(),
    )

    # Then: Supported alias context is injected and unknown context is ignored
    assert isinstance(loaded, RuntimePlugin)
    assert loaded.config.runtime_value == "injected"
    assert isinstance(validated, RuntimeConfig)
    assert validated.runtime_value == "injected"
