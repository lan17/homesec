"""Unified plugin discovery for all plugin types."""

import importlib
import logging
import pkgutil

from homesec.plugins.utils import iter_entry_points

logger = logging.getLogger(__name__)


def discover_all_plugins() -> None:
    """Discover and register all plugins (built-in and external).

    Built-in plugins are discovered by importing all modules in plugin
    type packages. External plugins are discovered via entry points.

    All plugins use decorators for registration, so importing modules
    triggers registration automatically.
    """
    # 1. Discover built-in plugins by importing all modules
    plugin_types = ["filters", "analyzers", "storage", "notifiers", "alert_policies", "sources"]

    for plugin_type in plugin_types:
        package = importlib.import_module(f"homesec.plugins.{plugin_type}")
        for _, module_name, _ in pkgutil.iter_modules(package.__path__):
            if module_name.startswith("_"):
                continue  # Skip private modules
            importlib.import_module(f"homesec.plugins.{plugin_type}.{module_name}")

    # 2. Discover external plugins via entry points
    for point in iter_entry_points("homesec.plugins"):
        importlib.import_module(point.module)


__all__ = ["discover_all_plugins"]
