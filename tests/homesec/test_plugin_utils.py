"""Tests for plugin utility functions."""

from __future__ import annotations

from importlib import metadata
from typing import Any
from unittest.mock import MagicMock

import pytest

from homesec.plugins.utils import iter_entry_points, load_plugin_from_entry_point


class DummyPlugin:
    """Dummy plugin class for testing."""

    def __init__(self, name: str = "test") -> None:
        self.name = name


class TestIterEntryPoints:
    """Tests for iter_entry_points function."""

    def test_returns_empty_for_unknown_group(self) -> None:
        """Returns empty iterable for non-existent group."""
        # Given: A group name that doesn't exist

        # When: Iterating entry points for the group
        result = list(iter_entry_points("nonexistent.group.xyz.12345"))

        # Then: Empty list is returned
        assert result == []

    def test_returns_real_entry_points(self) -> None:
        """Returns actual entry points for known groups."""
        # Given: A real entry point group (console_scripts is always populated)

        # When: Iterating entry points
        result = list(iter_entry_points("console_scripts"))

        # Then: We get entry points (homesec CLI should be there)
        entry_point_names = [ep.name for ep in result]
        assert "homesec" in entry_point_names


class TestLoadPluginFromEntryPoint:
    """Tests for load_plugin_from_entry_point function."""

    def _make_entry_point(self, name: str, load_returns: Any) -> MagicMock:
        """Create a mock entry point that returns the given value on load()."""
        mock_ep = MagicMock(spec=metadata.EntryPoint)
        mock_ep.name = name
        mock_ep.load.return_value = load_returns
        return mock_ep

    def test_loads_direct_plugin_instance(self) -> None:
        """Loads plugin when entry point returns instance directly."""
        # Given: An entry point that returns a DummyPlugin instance
        plugin = DummyPlugin("direct")
        mock_ep = self._make_entry_point("test_plugin", plugin)

        # When: Loading the plugin
        result = load_plugin_from_entry_point(mock_ep, DummyPlugin, "Test")

        # Then: The same plugin instance is returned
        assert result is plugin
        assert result.name == "direct"

    def test_loads_plugin_from_factory(self) -> None:
        """Loads plugin when entry point returns a factory callable."""
        # Given: An entry point that returns a factory function
        def factory() -> DummyPlugin:
            return DummyPlugin("from_factory")

        mock_ep = self._make_entry_point("test_plugin", factory)

        # When: Loading the plugin
        result = load_plugin_from_entry_point(mock_ep, DummyPlugin, "Test")

        # Then: The plugin instance from the factory is returned
        assert isinstance(result, DummyPlugin)
        assert result.name == "from_factory"

    def test_raises_for_wrong_type(self) -> None:
        """Raises TypeError when entry point returns wrong type."""
        # Given: An entry point that returns a non-plugin object
        mock_ep = self._make_entry_point("bad_plugin", "not a plugin")

        # When/Then: Loading raises TypeError with helpful message
        with pytest.raises(TypeError) as exc_info:
            load_plugin_from_entry_point(mock_ep, DummyPlugin, "Test")

        error_msg = str(exc_info.value)
        assert "bad_plugin" in error_msg  # Entry point name for debugging
        assert "Test" in error_msg  # Plugin type name
        assert "DummyPlugin" in error_msg  # Expected type

    def test_raises_when_factory_returns_wrong_type(self) -> None:
        """Raises TypeError when factory returns wrong type."""
        # Given: An entry point that returns a factory with wrong return type
        def bad_factory() -> str:
            return "not a plugin"

        mock_ep = self._make_entry_point("bad_factory", bad_factory)

        # When/Then: Loading raises TypeError
        with pytest.raises(TypeError, match="bad_factory"):
            load_plugin_from_entry_point(mock_ep, DummyPlugin, "Test")

    def test_accepts_subclass_instances(self) -> None:
        """Accepts subclass instances of the expected type."""

        # Given: A subclass of DummyPlugin
        class DummyPluginSubclass(DummyPlugin):
            pass

        plugin = DummyPluginSubclass("subclass")
        mock_ep = self._make_entry_point("subclass_plugin", plugin)

        # When: Loading the plugin
        result = load_plugin_from_entry_point(mock_ep, DummyPlugin, "Test")

        # Then: The subclass instance is accepted
        assert result is plugin
        assert isinstance(result, DummyPlugin)

    def test_factory_can_return_subclass(self) -> None:
        """Factory can return subclass of expected type."""

        # Given: A factory that returns a subclass
        class DummyPluginSubclass(DummyPlugin):
            pass

        def factory() -> DummyPluginSubclass:
            return DummyPluginSubclass("factory_subclass")

        mock_ep = self._make_entry_point("factory_plugin", factory)

        # When: Loading the plugin
        result = load_plugin_from_entry_point(mock_ep, DummyPlugin, "Test")

        # Then: The subclass instance is accepted
        assert isinstance(result, DummyPlugin)
        assert result.name == "factory_subclass"
