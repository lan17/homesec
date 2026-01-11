"""Shared utilities for plugin discovery and loading."""

from __future__ import annotations

from collections.abc import Iterable
from importlib import metadata
from typing import TypeVar, cast

PluginT = TypeVar("PluginT")


def iter_entry_points(group: str) -> Iterable[metadata.EntryPoint]:
    """Iterate entry points (handles Python 3.10+ and earlier).

    Args:
        group: Entry point group name (e.g., "homesec.filters")

    Returns:
        Iterable of entry points for the group
    """
    entry_points = metadata.entry_points()
    if hasattr(entry_points, "select"):
        # Python 3.10+ API
        return entry_points.select(group=group)
    # Python 3.9 API
    return cast(dict[str, list[metadata.EntryPoint]], entry_points).get(group, [])


def load_plugin_from_entry_point(
    point: metadata.EntryPoint,
    expected_type: type[PluginT],
    plugin_type_name: str,
) -> PluginT:
    """Load plugin from entry point.

    Handles both direct plugin instances and factory callables.

    Args:
        point: Entry point to load
        expected_type: Expected plugin type (e.g., FilterPlugin)
        plugin_type_name: Human-readable name for error messages (e.g., "Filter")

    Returns:
        Loaded plugin instance

    Raises:
        TypeError: If entry point doesn't return expected type
    """
    loaded = point.load()

    # Direct plugin instance
    if isinstance(loaded, expected_type):
        return loaded

    # Plugin factory callable
    if callable(loaded):
        result = loaded()
        if isinstance(result, expected_type):
            return result

    raise TypeError(
        f"Invalid {plugin_type_name} plugin entry point: {point.name}. "
        f"Expected {expected_type.__name__} or callable returning it."
    )
