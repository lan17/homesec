"""Tests for CLI module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from homesec.cli import HomeSec, main, setup_logging
from homesec.config import ConfigError


def _minimal_config() -> dict[str, object]:
    """Return minimal valid config dict for testing."""
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


class TestSetupLogging:
    """Tests for setup_logging function."""

    def test_configures_logging_with_default_level(self) -> None:
        """Configures logging with INFO level by default."""
        # Given: Default log level

        # When: Calling setup_logging
        with patch("homesec.cli.configure_logging") as mock_configure:
            setup_logging()

        # Then: configure_logging is called with INFO
        mock_configure.assert_called_once_with(log_level="INFO")

    def test_configures_logging_with_custom_level(self) -> None:
        """Configures logging with custom level."""
        # Given: Custom log level

        # When: Calling setup_logging with DEBUG
        with patch("homesec.cli.configure_logging") as mock_configure:
            setup_logging("DEBUG")

        # Then: configure_logging is called with DEBUG
        mock_configure.assert_called_once_with(log_level="DEBUG")


class TestHomeSecValidate:
    """Tests for validate command."""

    def test_validate_valid_config(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Validates and prints config details for valid config."""
        # Given: A valid config file
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.dump(_minimal_config()))

        # Mock discover_all_plugins (imported inside function) and validation functions
        with (
            patch("homesec.plugins.discover_all_plugins"),
            patch("homesec.cli.validate_camera_references"),
            patch("homesec.cli.validate_plugin_names"),
        ):
            # When: Validating the config
            cli = HomeSec()
            cli.validate(str(config_path))

        # Then: Success message is printed
        captured = capsys.readouterr()
        assert "✓ Config valid" in captured.out
        assert "Cameras:" in captured.out
        assert "front_door" in captured.out

    def test_validate_invalid_config_exits(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Exits with error for invalid config."""
        # Given: An invalid config file (missing required fields)
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.dump({"version": 1}))

        # When/Then: Validating raises SystemExit
        cli = HomeSec()
        with pytest.raises(SystemExit) as exc_info:
            cli.validate(str(config_path))

        # Then: Exit code is 1 and error is printed
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "✗ Config invalid" in captured.err

    def test_validate_nonexistent_config_exits(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Exits with error for nonexistent config file."""
        # Given: A path to a nonexistent file
        config_path = tmp_path / "nonexistent.yaml"

        # When/Then: Validating raises SystemExit
        cli = HomeSec()
        with pytest.raises(SystemExit) as exc_info:
            cli.validate(str(config_path))

        # Then: Exit code is 1
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "✗ Config invalid" in captured.err


class TestHomeSecRun:
    """Tests for run command."""

    def test_run_config_error_exits(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Exits with error when Application raises ConfigError."""
        # Given: A config file and Application that raises ConfigError
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.dump(_minimal_config()))

        mock_app = MagicMock()
        mock_app.run = AsyncMock(side_effect=ConfigError("Test error"))

        with (
            patch("homesec.cli.setup_logging"),
            patch("homesec.cli.Application", return_value=mock_app),
        ):
            # When/Then: Running raises SystemExit
            cli = HomeSec()
            with pytest.raises(SystemExit) as exc_info:
                cli.run(str(config_path))

        # Then: Exit code is 1 and error is printed
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "✗ Config invalid" in captured.err
        assert "Test error" in captured.err

    def test_run_keyboard_interrupt_handled(self, tmp_path: Path) -> None:
        """Handles KeyboardInterrupt gracefully."""
        # Given: A config file and Application that raises KeyboardInterrupt
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.dump(_minimal_config()))

        mock_app = MagicMock()
        mock_app.run = AsyncMock(side_effect=KeyboardInterrupt())

        with (
            patch("homesec.cli.setup_logging"),
            patch("homesec.cli.Application", return_value=mock_app),
        ):
            # When: Running with interrupt
            cli = HomeSec()
            cli.run(str(config_path))  # Should not raise

        # Then: No exception raised (handled gracefully)

    def test_run_uses_custom_log_level(self, tmp_path: Path) -> None:
        """Uses custom log level when specified."""
        # Given: A config file
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.dump(_minimal_config()))

        mock_app = MagicMock()
        mock_app.run = AsyncMock(side_effect=KeyboardInterrupt())

        with (
            patch("homesec.cli.setup_logging") as mock_setup,
            patch("homesec.cli.Application", return_value=mock_app),
        ):
            # When: Running with DEBUG level
            cli = HomeSec()
            cli.run(str(config_path), log_level="DEBUG")

        # Then: setup_logging called with DEBUG
        mock_setup.assert_called_once_with("DEBUG")


class TestHomeSecCleanup:
    """Tests for cleanup command."""

    def test_cleanup_config_error_exits(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Exits with error when run_cleanup raises ConfigError."""
        # Given: A config file
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.dump(_minimal_config()))

        with (
            patch("homesec.cli.setup_logging"),
            patch("homesec.cli.run_cleanup", new_callable=AsyncMock) as mock_cleanup,
        ):
            mock_cleanup.side_effect = ConfigError("Cleanup config error")

            # When/Then: Cleanup raises SystemExit
            cli = HomeSec()
            with pytest.raises(SystemExit) as exc_info:
                cli.cleanup(str(config_path))

        # Then: Exit code is 1 and error is printed
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "✗ Config invalid" in captured.err

    def test_cleanup_keyboard_interrupt_handled(self, tmp_path: Path) -> None:
        """Handles KeyboardInterrupt gracefully."""
        # Given: A config file
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.dump(_minimal_config()))

        with (
            patch("homesec.cli.setup_logging"),
            patch("homesec.cli.run_cleanup", new_callable=AsyncMock) as mock_cleanup,
        ):
            mock_cleanup.side_effect = KeyboardInterrupt()

            # When: Running cleanup with interrupt
            cli = HomeSec()
            cli.cleanup(str(config_path))  # Should not raise

        # Then: No exception raised (handled gracefully)

    def test_cleanup_passes_options(self, tmp_path: Path) -> None:
        """Passes all options to CleanupOptions."""
        # Given: A config file
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.dump(_minimal_config()))

        with (
            patch("homesec.cli.setup_logging"),
            patch("homesec.cli.CleanupOptions") as mock_opts,
            patch("homesec.cli.run_cleanup", new_callable=AsyncMock),
        ):
            # When: Running cleanup with all options
            cli = HomeSec()
            cli.cleanup(
                str(config_path),
                older_than_days=30,
                camera_name="front_door",
                batch_size=50,
                workers=4,
                dry_run=False,
                recheck_model_path="yolo11x.pt",
                recheck_min_confidence=0.5,
                recheck_sample_fps=2,
                recheck_min_box_h_ratio=0.1,
                recheck_min_hits=3,
            )

        # Then: CleanupOptions created with all parameters
        mock_opts.assert_called_once()
        call_kwargs = mock_opts.call_args.kwargs
        assert call_kwargs["older_than_days"] == 30
        assert call_kwargs["camera_name"] == "front_door"
        assert call_kwargs["batch_size"] == 50
        assert call_kwargs["workers"] == 4
        assert call_kwargs["dry_run"] is False
        assert call_kwargs["recheck_model_path"] == "yolo11x.pt"
        assert call_kwargs["recheck_min_confidence"] == 0.5
        assert call_kwargs["recheck_sample_fps"] == 2
        assert call_kwargs["recheck_min_box_h_ratio"] == 0.1
        assert call_kwargs["recheck_min_hits"] == 3

    def test_cleanup_uses_custom_log_level(self, tmp_path: Path) -> None:
        """Uses custom log level when specified."""
        # Given: A config file
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.dump(_minimal_config()))

        with (
            patch("homesec.cli.setup_logging") as mock_setup,
            patch("homesec.cli.run_cleanup", new_callable=AsyncMock),
        ):
            # When: Running cleanup with DEBUG level
            cli = HomeSec()
            cli.cleanup(str(config_path), log_level="DEBUG")

        # Then: setup_logging called with DEBUG
        mock_setup.assert_called_once_with("DEBUG")


class TestMain:
    """Tests for main entrypoint."""

    def test_main_calls_fire_with_command(self) -> None:
        """Main entrypoint calls fire.Fire when command is provided."""
        # Given: Mocked fire module and sys.argv with a command
        with (
            patch("homesec.cli.fire.Fire") as mock_fire,
            patch("homesec.cli.sys.argv", ["homesec", "validate", "--config", "x.yaml"]),
        ):
            # When: Calling main
            main()

        # Then: fire.Fire called with HomeSec
        mock_fire.assert_called_once_with(HomeSec)

    def test_main_calls_fire_with_no_args(self) -> None:
        """Main calls fire.Fire when called with no arguments (Fire shows commands)."""
        # Given: sys.argv with only the program name
        with (
            patch("homesec.cli.fire.Fire") as mock_fire,
            patch("homesec.cli.sys.argv", ["homesec"]),
        ):
            # When: Calling main
            main()

        # Then: fire.Fire is called (Fire will show commands list)
        mock_fire.assert_called_once_with(HomeSec)

    def test_main_strips_help_flag_before_fire(self) -> None:
        """Main strips --help flag so Fire shows commands list."""
        # Given: sys.argv with --help
        argv = ["homesec", "--help"]
        with (
            patch("homesec.cli.fire.Fire") as mock_fire,
            patch("homesec.cli.sys.argv", argv),
        ):
            # When: Calling main
            main()

        # Then: fire.Fire is called and --help was stripped from argv
        mock_fire.assert_called_once_with(HomeSec)
        assert argv == ["homesec"]

    def test_main_strips_h_flag_before_fire(self) -> None:
        """Main strips -h flag so Fire shows commands list."""
        # Given: sys.argv with -h
        argv = ["homesec", "-h"]
        with (
            patch("homesec.cli.fire.Fire") as mock_fire,
            patch("homesec.cli.sys.argv", argv),
        ):
            # When: Calling main
            main()

        # Then: fire.Fire is called and -h was stripped from argv
        mock_fire.assert_called_once_with(HomeSec)
        assert argv == ["homesec"]

    def test_main_preserves_command_help_flag(self) -> None:
        """Main preserves --help when used with a command."""
        # Given: sys.argv with command and --help
        argv = ["homesec", "run", "--help"]
        with (
            patch("homesec.cli.fire.Fire") as mock_fire,
            patch("homesec.cli.sys.argv", argv),
        ):
            # When: Calling main
            main()

        # Then: fire.Fire is called and argv is unchanged
        mock_fire.assert_called_once_with(HomeSec)
        assert argv == ["homesec", "run", "--help"]
