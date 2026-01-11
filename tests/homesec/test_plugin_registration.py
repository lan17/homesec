"""Tests for plugin registration and discovery mechanisms."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from homesec.plugins import discover_all_plugins
from homesec.plugins.alert_policies import (
    ALERT_POLICY_REGISTRY,
    AlertPolicyPlugin,
    register_alert_policy,
)
from homesec.plugins.analyzers import (
    VLM_REGISTRY,
    VLMPlugin,
    register_vlm,
)
from homesec.plugins.filters import (
    FILTER_REGISTRY,
    FilterPlugin,
    register_filter,
)
from homesec.plugins.notifiers import (
    NOTIFIER_REGISTRY,
    NotifierPlugin,
    register_notifier,
)
from homesec.plugins.storage import (
    STORAGE_REGISTRY,
    StoragePlugin,
    register_storage,
)


class TestUnifiedDiscovery:
    """Test unified plugin discovery."""

    def test_discover_all_plugins_loads_builtins(self) -> None:
        # Given: Registries may already have plugins from previous imports
        # (Decorators execute at module import time, so modules may already be loaded)

        # When: Running unified discovery (which imports all plugin modules)
        # Note: If modules are already imported, decorators won't re-execute,
        # but that's fine - plugins should already be registered
        discover_all_plugins()

        # Then: All built-in plugins should be loaded
        # (Either freshly loaded or already present from earlier imports)
        assert "yolo" in FILTER_REGISTRY
        assert "openai" in VLM_REGISTRY
        assert "dropbox" in STORAGE_REGISTRY
        assert "local" in STORAGE_REGISTRY
        assert "mqtt" in NOTIFIER_REGISTRY
        assert "sendgrid_email" in NOTIFIER_REGISTRY
        assert "default" in ALERT_POLICY_REGISTRY
        assert "noop" in ALERT_POLICY_REGISTRY


class TestExternalPluginDiscovery:
    """Test external plugin discovery via entry points."""

    @patch("homesec.plugins.iter_entry_points")
    def test_external_plugin_entry_points_queried(self, mock_iter_eps: MagicMock) -> None:
        """Discovery queries the homesec.plugins entry point group."""
        # Given: No external plugins
        mock_iter_eps.return_value = []

        # When: Running discovery
        discover_all_plugins()

        # Then: Should query the homesec.plugins entry point group
        mock_iter_eps.assert_called_once_with("homesec.plugins")

    @patch("homesec.plugins.iter_entry_points")
    def test_external_plugin_import_error_logged_and_continues(
        self, mock_iter_eps: MagicMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Import errors from external plugins are logged but don't crash discovery."""
        # Given: A mock entry point with a non-existent module
        mock_entry_point = MagicMock()
        mock_entry_point.name = "broken_plugin"
        mock_entry_point.module = "this_module_does_not_exist_12345"
        mock_iter_eps.return_value = [mock_entry_point]

        # When: Running discovery (which will try to import the non-existent module)
        discover_all_plugins()

        # Then: Error should be logged but discovery completes
        assert "Failed to load external plugin broken_plugin" in caplog.text
        assert "this_module_does_not_exist_12345" in caplog.text

    def test_external_plugin_collision_prevents_override(self) -> None:
        """External plugins cannot override built-in plugin names."""
        # Given: Built-in "yolo" filter is already registered
        assert "yolo" in FILTER_REGISTRY
        original_plugin = FILTER_REGISTRY["yolo"]

        # When: Trying to register a plugin with the same name
        conflicting_plugin = FilterPlugin(
            name="yolo",
            config_model=BaseModel,
            factory=lambda cfg: MagicMock(),  # type: ignore[arg-type, return-value]
        )

        # Then: Should raise ValueError
        with pytest.raises(ValueError, match="already registered"):
            register_filter(conflicting_plugin)

        # And: Original plugin should still be registered
        assert FILTER_REGISTRY["yolo"] == original_plugin


class TestPluginRegistration:
    """Test manual plugin registration functions."""

    def test_register_filter_adds_to_registry(self) -> None:
        # Given: A custom filter plugin
        plugin = FilterPlugin(
            name="custom_filter",
            config_model=BaseModel,
            factory=lambda cfg: MagicMock(),  # type: ignore[arg-type, return-value]
        )
        FILTER_REGISTRY.clear()

        # When: Registering the plugin
        register_filter(plugin)

        # Then: Plugin should be in registry
        assert "custom_filter" in FILTER_REGISTRY
        assert FILTER_REGISTRY["custom_filter"] == plugin

    def test_register_vlm_adds_to_registry(self) -> None:
        # Given: A custom VLM plugin
        plugin = VLMPlugin(
            name="custom_vlm",
            config_model=BaseModel,
            factory=lambda cfg: MagicMock(),  # type: ignore[arg-type, return-value]
        )
        VLM_REGISTRY.clear()

        # When: Registering the plugin
        register_vlm(plugin)

        # Then: Plugin should be in registry
        assert "custom_vlm" in VLM_REGISTRY
        assert VLM_REGISTRY["custom_vlm"] == plugin

    def test_register_storage_adds_to_registry(self) -> None:
        # Given: A custom storage plugin
        plugin = StoragePlugin(
            name="custom_storage",
            config_model=BaseModel,
            factory=lambda cfg: MagicMock(),  # type: ignore[arg-type, return-value]
        )
        STORAGE_REGISTRY.clear()

        # When: Registering the plugin
        register_storage(plugin)

        # Then: Plugin should be in registry
        assert "custom_storage" in STORAGE_REGISTRY
        assert STORAGE_REGISTRY["custom_storage"] == plugin

    def test_register_notifier_adds_to_registry(self) -> None:
        # Given: A custom notifier plugin
        plugin = NotifierPlugin(
            name="custom_notifier",
            config_model=BaseModel,
            factory=lambda cfg: MagicMock(),  # type: ignore[arg-type, return-value]
        )
        NOTIFIER_REGISTRY.clear()

        # When: Registering the plugin
        register_notifier(plugin)

        # Then: Plugin should be in registry
        assert "custom_notifier" in NOTIFIER_REGISTRY
        assert NOTIFIER_REGISTRY["custom_notifier"] == plugin

    def test_register_alert_policy_adds_to_registry(self) -> None:
        # Given: A custom alert policy plugin
        plugin = AlertPolicyPlugin(
            name="custom_policy",
            config_model=BaseModel,
            factory=lambda cfg, per_camera, triggers: MagicMock(),  # type: ignore[arg-type, return-value]
        )
        ALERT_POLICY_REGISTRY.clear()

        # When: Registering the plugin
        register_alert_policy(plugin)

        # Then: Plugin should be in registry
        assert "custom_policy" in ALERT_POLICY_REGISTRY
        assert ALERT_POLICY_REGISTRY["custom_policy"] == plugin

    def test_register_filter_collision_raises_error(self) -> None:
        # Given: A plugin already registered
        plugin1 = FilterPlugin(
            name="test",
            config_model=BaseModel,
            factory=lambda cfg: MagicMock(),  # type: ignore[arg-type, return-value]
        )
        plugin2 = FilterPlugin(
            name="test",
            config_model=BaseModel,
            factory=lambda cfg: MagicMock(),  # type: ignore[arg-type, return-value]
        )
        FILTER_REGISTRY.clear()
        register_filter(plugin1)

        # When/Then: Registering a plugin with the same name should raise ValueError
        with pytest.raises(ValueError, match="already registered"):
            register_filter(plugin2)

    def test_register_vlm_collision_raises_error(self) -> None:
        # Given: A plugin already registered
        plugin1 = VLMPlugin(
            name="test",
            config_model=BaseModel,
            factory=lambda cfg: MagicMock(),  # type: ignore[arg-type, return-value]
        )
        plugin2 = VLMPlugin(
            name="test",
            config_model=BaseModel,
            factory=lambda cfg: MagicMock(),  # type: ignore[arg-type, return-value]
        )
        VLM_REGISTRY.clear()
        register_vlm(plugin1)

        # When/Then: Registering a plugin with the same name should raise ValueError
        with pytest.raises(ValueError, match="already registered"):
            register_vlm(plugin2)

    def test_register_storage_collision_raises_error(self) -> None:
        # Given: A plugin already registered
        plugin1 = StoragePlugin(
            name="test",
            config_model=BaseModel,
            factory=lambda cfg: MagicMock(),  # type: ignore[arg-type, return-value]
        )
        plugin2 = StoragePlugin(
            name="test",
            config_model=BaseModel,
            factory=lambda cfg: MagicMock(),  # type: ignore[arg-type, return-value]
        )
        STORAGE_REGISTRY.clear()
        register_storage(plugin1)

        # When/Then: Registering a plugin with the same name should raise ValueError
        with pytest.raises(ValueError, match="already registered"):
            register_storage(plugin2)

    def test_register_notifier_collision_raises_error(self) -> None:
        # Given: A plugin already registered
        plugin1 = NotifierPlugin(
            name="test",
            config_model=BaseModel,
            factory=lambda cfg: MagicMock(),  # type: ignore[arg-type, return-value]
        )
        plugin2 = NotifierPlugin(
            name="test",
            config_model=BaseModel,
            factory=lambda cfg: MagicMock(),  # type: ignore[arg-type, return-value]
        )
        NOTIFIER_REGISTRY.clear()
        register_notifier(plugin1)

        # When/Then: Registering a plugin with the same name should raise ValueError
        with pytest.raises(ValueError, match="already registered"):
            register_notifier(plugin2)

    def test_register_alert_policy_collision_raises_error(self) -> None:
        # Given: A plugin already registered
        plugin1 = AlertPolicyPlugin(
            name="test",
            config_model=BaseModel,
            factory=lambda cfg, per_camera, triggers: MagicMock(),  # type: ignore[arg-type, return-value]
        )
        plugin2 = AlertPolicyPlugin(
            name="test",
            config_model=BaseModel,
            factory=lambda cfg, per_camera, triggers: MagicMock(),  # type: ignore[arg-type, return-value]
        )
        ALERT_POLICY_REGISTRY.clear()
        register_alert_policy(plugin1)

        # When/Then: Registering a plugin with the same name should raise ValueError
        with pytest.raises(ValueError, match="already registered"):
            register_alert_policy(plugin2)


class TestThirdPartyPluginConfigValidation:
    """Test that third-party plugin configs are validated in loaders."""

    def test_filter_loader_validates_third_party_config(self) -> None:
        """Test that load_filter_plugin() validates dict configs using plugin.config_model."""
        from pydantic import BaseModel, Field, ValidationError

        from homesec.models.filter import FilterConfig
        from homesec.plugins.filters import (
            FILTER_REGISTRY,
            load_filter_plugin,
            register_filter,
        )

        # Given: A third-party filter plugin with custom config model
        class CustomFilterConfig(BaseModel):
            custom_field: str
            custom_number: int = Field(ge=0)

        mock_filter = MagicMock()

        def custom_factory(cfg: FilterConfig) -> object:
            # Validate that config is the validated Pydantic object
            assert isinstance(cfg.config, CustomFilterConfig)
            assert cfg.config.custom_field == "test"
            return mock_filter

        plugin = FilterPlugin(
            name="custom_filter", config_model=CustomFilterConfig, factory=custom_factory
        )

        # Register the plugin
        FILTER_REGISTRY.clear()
        register_filter(plugin)

        try:
            # When: Loading a filter config with valid dict
            filter_cfg = FilterConfig(
                plugin="custom_filter",
                max_workers=2,
                config={"custom_field": "test", "custom_number": 42},
            )

            # Then: load_filter_plugin should validate and pass validated object to factory
            result = load_filter_plugin(filter_cfg)
            assert result == mock_filter

            # And: Validation should reject invalid configs
            invalid_cfg = FilterConfig(
                plugin="custom_filter",
                max_workers=2,
                config={"custom_field": "test", "custom_number": -1},  # Violates ge=0
            )

            with pytest.raises(ValidationError):
                load_filter_plugin(invalid_cfg)
        finally:
            FILTER_REGISTRY.clear()

    def test_vlm_loader_validates_third_party_config(self) -> None:
        """Test that load_vlm_plugin() validates dict configs using plugin.config_model."""
        from pydantic import BaseModel, Field, ValidationError

        from homesec.models.vlm import VLMConfig
        from homesec.plugins.analyzers import (
            VLM_REGISTRY,
            load_vlm_plugin,
            register_vlm,
        )

        # Given: A third-party VLM plugin with custom config model
        class CustomVLMConfig(BaseModel):
            api_key: str = Field(min_length=1)
            model_name: str

        mock_vlm = MagicMock()

        def custom_factory(cfg: VLMConfig) -> object:
            # Validate that llm is the validated Pydantic object
            assert isinstance(cfg.llm, CustomVLMConfig)
            assert cfg.llm.api_key == "secret123"
            return mock_vlm

        plugin = VLMPlugin(name="custom_vlm", config_model=CustomVLMConfig, factory=custom_factory)

        # Register the plugin
        VLM_REGISTRY.clear()
        register_vlm(plugin)

        try:
            # When: Loading a VLM config with valid dict
            vlm_cfg = VLMConfig(
                backend="custom_vlm",
                max_workers=2,
                llm={"api_key": "secret123", "model_name": "custom-v1"},
            )

            # Then: load_vlm_plugin should validate and pass validated object to factory
            result = load_vlm_plugin(vlm_cfg)
            assert result == mock_vlm

            # And: Validation should reject invalid configs
            invalid_cfg = VLMConfig(
                backend="custom_vlm",
                max_workers=2,
                llm={"api_key": "", "model_name": "custom-v1"},  # Violates min_length=1
            )

            with pytest.raises(ValidationError):
                load_vlm_plugin(invalid_cfg)
        finally:
            VLM_REGISTRY.clear()
