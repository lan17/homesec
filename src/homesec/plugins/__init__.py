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
    plugin_types = ["filters", "analyzers", "storage", "notifiers", "alert_policies"]

    for plugin_type in plugin_types:
        try:
            package = importlib.import_module(f"homesec.plugins.{plugin_type}")
            for _, module_name, _ in pkgutil.iter_modules(package.__path__):
                if module_name.startswith("_"):
                    continue  # Skip private modules
                try:
                    importlib.import_module(f"homesec.plugins.{plugin_type}.{module_name}")
                except Exception as exc:
                    logger.error(
                        "Failed to import built-in plugin module %s.%s: %s",
                        plugin_type,
                        module_name,
                        exc,
                        exc_info=True,
                    )
        except Exception as exc:
            logger.error(
                "Failed to discover built-in plugins for %s: %s",
                plugin_type,
                exc,
                exc_info=True,
            )

    # 2. Discover external plugins via entry points
    for point in iter_entry_points("homesec.plugins"):
        try:
            importlib.import_module(point.module)
        except Exception as exc:
            logger.error(
                "Failed to load external plugin %s from %s: %s",
                point.name,
                point.module,
                exc,
                exc_info=True,
            )


__all__ = ["discover_all_plugins"]
