"""CLI entrypoint for HomeSec application."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import fire  # type: ignore[import-untyped]

from homesec.app import Application
from homesec.config import ConfigError, load_config
from homesec.config.validation import validate_camera_references, validate_plugin_names
from homesec.logging_setup import configure_logging
from homesec.maintenance.cleanup_clips import CleanupOptions, run_cleanup
from homesec.plugins.alert_policies import ALERT_POLICY_REGISTRY
from homesec.plugins.analyzers import VLM_REGISTRY
from homesec.plugins.filters import FILTER_REGISTRY
from homesec.plugins.notifiers import NOTIFIER_REGISTRY
from homesec.plugins.storage import STORAGE_REGISTRY


def setup_logging(level: str = "INFO") -> None:
    """Configure logging for CLI."""
    configure_logging(log_level=level)


class HomeSec:
    """HomeSec CLI - Home Security Camera Pipeline."""

    def run(self, config: str, log_level: str = "INFO") -> None:
        """Run the HomeSec pipeline.

        Args:
            config: Path to YAML config file
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        """
        setup_logging(log_level)

        config_path = Path(config)

        app = Application(config_path)

        try:
            asyncio.run(app.run())
        except ConfigError as e:
            print(f"✗ Config invalid: {e}", file=sys.stderr)
            sys.exit(1)
        except KeyboardInterrupt:
            pass  # Handled by signal handlers

    def validate(self, config: str) -> None:
        """Validate config file without running.

        Args:
            config: Path to YAML config file
        """
        config_path = Path(config)

        try:
            cfg = load_config(config_path)

            # Discover all plugins
            from homesec.plugins import discover_all_plugins

            discover_all_plugins()

            # Additional validation checks
            validate_camera_references(cfg)
            validate_plugin_names(
                cfg,
                sorted(FILTER_REGISTRY.keys()),
                sorted(VLM_REGISTRY.keys()),
                valid_storage=sorted(STORAGE_REGISTRY.keys()),
                valid_notifiers=sorted(NOTIFIER_REGISTRY.keys()),
                valid_alert_policies=sorted(ALERT_POLICY_REGISTRY.keys()),
            )

            print(f"✓ Config valid: {config_path}")
            camera_names = [camera.name for camera in cfg.cameras]
            print(f"  Cameras: {camera_names}")
            notifier_backends = [
                f"{notifier.backend} (enabled={notifier.enabled})" for notifier in cfg.notifiers
            ]
            print(f"  Storage backend: {cfg.storage.backend}")
            print(f"  Notifiers: {notifier_backends}")
            print(f"  Filter plugin: {cfg.filter.plugin}")
            print(f"  VLM backend: {cfg.vlm.backend}")
            print(f"  VLM trigger classes: {cfg.vlm.trigger_classes}")
            print(f"  Alert policy backend: {cfg.alert_policy.backend}")
            print(f"  Alert policy enabled: {cfg.alert_policy.enabled}")
        except ConfigError as e:
            print(f"✗ Config invalid: {e}", file=sys.stderr)
            sys.exit(1)

    def cleanup(
        self,
        config: str,
        older_than_days: int | None = None,
        camera_name: str | None = None,
        batch_size: int = 100,
        workers: int = 2,
        dry_run: bool = True,
        recheck_model_path: str | None = None,
        recheck_min_confidence: float | None = None,
        recheck_sample_fps: int | None = None,
        recheck_min_box_h_ratio: float | None = None,
        recheck_min_hits: int | None = None,
        log_level: str = "INFO",
    ) -> None:
        """Re-analyze and optionally delete clips that appear empty.

        Args:
            config: Path to YAML config file
            older_than_days: Only consider clips older than this many days
            camera_name: Optional camera name filter
            batch_size: Postgres paging size
            workers: Concurrency for re-analysis/deletion
            dry_run: If True, log actions but do not delete or mutate state
            recheck_model_path: Override YOLO model path for recheck (default: yolo11x.pt)
            recheck_min_confidence: Override YOLO confidence for recheck
            recheck_sample_fps: Override frame sampling step for recheck
            recheck_min_box_h_ratio: Override minimum box height ratio
            recheck_min_hits: Override minimum hits setting
            log_level: Logging level
        """
        setup_logging(log_level)

        opts = CleanupOptions(
            config_path=Path(config),
            older_than_days=older_than_days,
            camera_name=camera_name,
            batch_size=batch_size,
            workers=workers,
            dry_run=dry_run,
            recheck_model_path=recheck_model_path,
            recheck_min_confidence=recheck_min_confidence,
            recheck_sample_fps=recheck_sample_fps,
            recheck_min_box_h_ratio=recheck_min_box_h_ratio,
            recheck_min_hits=recheck_min_hits,
        )

        try:
            asyncio.run(run_cleanup(opts))
        except ConfigError as e:
            print(f"✗ Config invalid: {e}", file=sys.stderr)
            sys.exit(1)
        except KeyboardInterrupt:
            pass


def main() -> None:
    """Main CLI entrypoint."""
    # Strip --help/-h when it's the only arg so Fire shows its commands list
    if len(sys.argv) == 2 and sys.argv[1] in ("--help", "-h"):
        sys.argv.pop()
    fire.Fire(HomeSec)


if __name__ == "__main__":
    main()
