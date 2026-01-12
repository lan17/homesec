"""Tests for plugin utility functions."""

from __future__ import annotations

from importlib import metadata
from unittest.mock import MagicMock, patch

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
        result = list(iter_entry_points("nonexistent.group.xyz"))

        # Then: Empty list is returned
        assert result == []

    def test_uses_select_api_when_available(self) -> None:
        """Uses the select() API on Python 3.10+."""
        # Given: A mocked entry_points() with select method
        mock_eps = MagicMock()
        mock_eps.select.return_value = ["ep1", "ep2"]

        with patch.object(metadata, "entry_points", return_value=mock_eps):
            # When: Calling iter_entry_points
            result = list(iter_entry_points("test.group"))

            # Then: select() was called with the group
            mock_eps.select.assert_called_once_with(group="test.group")
            assert result == ["ep1", "ep2"]

    def test_falls_back_to_dict_api(self) -> None:
        """Falls back to dict-style API when select() is not available."""
        # Given: A mocked entry_points() without select method (Python 3.9 style)
        mock_eps: dict[str, list[str]] = {
            "test.group": ["ep1", "ep2"],
            "other.group": ["ep3"],
        }
        # Remove select attribute by using a plain dict
        with patch.object(metadata, "entry_points", return_value=mock_eps):
            # When: Calling iter_entry_points
            result = list(iter_entry_points("test.group"))

            # Then: Dict lookup was used
            assert result == ["ep1", "ep2"]

    def test_dict_api_returns_empty_for_missing_group(self) -> None:
        """Dict-style API returns empty for missing group."""
        # Given: A mocked entry_points() without the requested group
        mock_eps: dict[str, list[str]] = {
            "other.group": ["ep1"],
        }

        with patch.object(metadata, "entry_points", return_value=mock_eps):
            # When: Calling iter_entry_points for missing group
            result = list(iter_entry_points("test.group"))

            # Then: Empty list is returned
            assert result == []


class TestLoadPluginFromEntryPoint:
    """Tests for load_plugin_from_entry_point function."""

    def test_loads_direct_plugin_instance(self) -> None:
        """Loads plugin when entry point returns instance directly."""
        # Given: An entry point that returns a DummyPlugin instance
        mock_ep = MagicMock(spec=metadata.EntryPoint)
        mock_ep.name = "test_plugin"
        mock_ep.load.return_value = DummyPlugin("direct")

        # When: Loading the plugin
        result = load_plugin_from_entry_point(mock_ep, DummyPlugin, "Test")

        # Then: The plugin instance is returned
        assert isinstance(result, DummyPlugin)
        assert result.name == "direct"

    def test_loads_plugin_from_factory(self) -> None:
        """Loads plugin when entry point returns a factory callable."""
        # Given: An entry point that returns a factory function
        mock_ep = MagicMock(spec=metadata.EntryPoint)
        mock_ep.name = "test_plugin"

        def factory() -> DummyPlugin:
            return DummyPlugin("from_factory")

        mock_ep.load.return_value = factory

        # When: Loading the plugin
        result = load_plugin_from_entry_point(mock_ep, DummyPlugin, "Test")

        # Then: The plugin instance from the factory is returned
        assert isinstance(result, DummyPlugin)
        assert result.name == "from_factory"

    def test_raises_for_invalid_return_type(self) -> None:
        """Raises TypeError when entry point returns wrong type."""
        # Given: An entry point that returns a non-plugin object
        mock_ep = MagicMock(spec=metadata.EntryPoint)
        mock_ep.name = "bad_plugin"
        mock_ep.load.return_value = "not a plugin"

        # When/Then: Loading raises TypeError
        with pytest.raises(TypeError, match="Invalid Test plugin entry point"):
            load_plugin_from_entry_point(mock_ep, DummyPlugin, "Test")

    def test_raises_when_factory_returns_wrong_type(self) -> None:
        """Raises TypeError when factory returns wrong type."""
        # Given: An entry point that returns a factory with wrong return type
        mock_ep = MagicMock(spec=metadata.EntryPoint)
        mock_ep.name = "bad_factory_plugin"

        def bad_factory() -> str:
            return "not a plugin"

        mock_ep.load.return_value = bad_factory

        # When/Then: Loading raises TypeError
        with pytest.raises(TypeError, match="Invalid Test plugin entry point"):
            load_plugin_from_entry_point(mock_ep, DummyPlugin, "Test")

    def test_handles_subclass_instances(self) -> None:
        """Accepts subclass instances of the expected type."""

        # Given: A subclass of DummyPlugin
        class DummyPluginSubclass(DummyPlugin):
            pass

        mock_ep = MagicMock(spec=metadata.EntryPoint)
        mock_ep.name = "subclass_plugin"
        mock_ep.load.return_value = DummyPluginSubclass("subclass")

        # When: Loading the plugin
        result = load_plugin_from_entry_point(mock_ep, DummyPlugin, "Test")

        # Then: The subclass instance is accepted
        assert isinstance(result, DummyPlugin)
        assert result.name == "subclass"

    def test_factory_can_return_subclass(self) -> None:
        """Factory can return subclass of expected type."""

        # Given: A factory that returns a subclass
        class DummyPluginSubclass(DummyPlugin):
            pass

        mock_ep = MagicMock(spec=metadata.EntryPoint)
        mock_ep.name = "subclass_factory_plugin"

        def factory() -> DummyPluginSubclass:
            return DummyPluginSubclass("factory_subclass")

        mock_ep.load.return_value = factory

        # When: Loading the plugin
        result = load_plugin_from_entry_point(mock_ep, DummyPlugin, "Test")

        # Then: The subclass instance is accepted
        assert isinstance(result, DummyPlugin)
        assert result.name == "factory_subclass"

    def test_error_message_includes_plugin_name(self) -> None:
        """Error message includes entry point name for debugging."""
        # Given: An entry point with a specific name
        mock_ep = MagicMock(spec=metadata.EntryPoint)
        mock_ep.name = "my_broken_plugin"
        mock_ep.load.return_value = 12345  # Wrong type

        # When/Then: Error message includes the entry point name
        with pytest.raises(TypeError) as exc_info:
            load_plugin_from_entry_point(mock_ep, DummyPlugin, "Filter")

        assert "my_broken_plugin" in str(exc_info.value)
        assert "Filter" in str(exc_info.value)
