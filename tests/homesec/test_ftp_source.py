"""Tests for FTP source implementation."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from homesec.models.clip import Clip
from homesec.models.source import FtpSourceConfig
from homesec.sources.ftp import FtpSource, _parse_passive_ports


class TestParsePassivePorts:
    """Tests for _parse_passive_ports function."""

    def test_parse_range_format(self) -> None:
        """Parses range format like '6000-6100'."""
        # Given: A range format string

        # When: Parsing the range
        result = _parse_passive_ports("6000-6010")

        # Then: Returns list of ports in range
        assert result == list(range(6000, 6011))
        assert len(result) == 11
        assert result[0] == 6000
        assert result[-1] == 6010

    def test_parse_comma_format(self) -> None:
        """Parses comma-separated format like '6000,6001,6002'."""
        # Given: A comma-separated format string

        # When: Parsing the ports
        result = _parse_passive_ports("6000,6001,6002")

        # Then: Returns list of specified ports
        assert result == [6000, 6001, 6002]

    def test_parse_comma_with_whitespace(self) -> None:
        """Handles whitespace in comma-separated format."""
        # Given: Comma-separated with spaces

        # When: Parsing the ports
        result = _parse_passive_ports("6000, 6001,  6002")

        # Then: Whitespace is ignored
        assert result == [6000, 6001, 6002]

    def test_parse_invalid_range_raises(self) -> None:
        """Raises ValueError when end < start."""
        # Given: A range where end < start

        # When/Then: Parsing raises ValueError
        with pytest.raises(ValueError, match="Invalid passive_ports range"):
            _parse_passive_ports("6100-6000")

    def test_parse_none_returns_none(self) -> None:
        """Returns None for None input."""
        # Given: None input

        # When: Parsing None
        result = _parse_passive_ports(None)

        # Then: Returns None
        assert result is None

    def test_parse_empty_returns_none(self) -> None:
        """Returns None for empty string."""
        # Given: Empty string input

        # When: Parsing empty string
        result = _parse_passive_ports("")

        # Then: Returns None
        assert result is None

    def test_parse_whitespace_only_returns_none(self) -> None:
        """Returns None for whitespace-only string."""
        # Given: Whitespace-only string

        # When: Parsing
        result = _parse_passive_ports("   ")

        # Then: Returns None
        assert result is None

    def test_parse_single_port(self) -> None:
        """Parses single port number."""
        # Given: Single port string

        # When: Parsing
        result = _parse_passive_ports("6000")

        # Then: Returns list with single port
        assert result == [6000]


class TestFtpSourceClipCreation:
    """Tests for clip creation."""

    def test_clip_id_from_filename(self, tmp_path: Path) -> None:
        """Clip ID is derived from filename stem."""
        # Given: An FtpSource with a file
        config = FtpSourceConfig(root_dir=str(tmp_path))
        source = FtpSource(config, camera_name="test_cam")

        clip_file = tmp_path / "recording_2024_01_15.mp4"
        clip_file.write_bytes(b"video content")

        # When: Creating clip from file
        clip = source._create_clip(clip_file)

        # Then: Clip ID is the filename stem
        assert clip.clip_id == "recording_2024_01_15"

    def test_clip_camera_name_from_source(self, tmp_path: Path) -> None:
        """Clip camera_name comes from source configuration."""
        # Given: An FtpSource with specific camera name
        config = FtpSourceConfig(root_dir=str(tmp_path))
        source = FtpSource(config, camera_name="back_yard")

        clip_file = tmp_path / "clip.mp4"
        clip_file.write_bytes(b"video")

        # When: Creating clip
        clip = source._create_clip(clip_file)

        # Then: Camera name matches source
        assert clip.camera_name == "back_yard"

    def test_clip_source_type_is_ftp(self, tmp_path: Path) -> None:
        """Clip source_type is 'ftp'."""
        # Given: An FtpSource
        config = FtpSourceConfig(root_dir=str(tmp_path))
        source = FtpSource(config, camera_name="cam")

        clip_file = tmp_path / "clip.mp4"
        clip_file.write_bytes(b"video")

        # When: Creating clip
        clip = source._create_clip(clip_file)

        # Then: Source type is ftp
        assert clip.source_type == "ftp"

    def test_clip_timestamps_from_mtime(self, tmp_path: Path) -> None:
        """Clip timestamps are calculated from file mtime."""
        # Given: An FtpSource with custom duration
        config = FtpSourceConfig(root_dir=str(tmp_path), default_duration_s=30.0)
        source = FtpSource(config, camera_name="cam")

        clip_file = tmp_path / "clip.mp4"
        clip_file.write_bytes(b"video")
        mtime = datetime.fromtimestamp(clip_file.stat().st_mtime)

        # When: Creating clip
        clip = source._create_clip(clip_file)

        # Then: end_ts is mtime, start_ts is mtime - duration
        assert clip.end_ts == mtime
        assert clip.start_ts == mtime - timedelta(seconds=30.0)
        assert clip.duration_s == 30.0


class TestFtpSourceFileHandling:
    """Tests for file handling behavior."""

    def test_allowed_extension_emits_clip(self, tmp_path: Path) -> None:
        """Allowed extensions emit clips."""
        # Given: An FtpSource with .mp4 allowed
        config = FtpSourceConfig(
            root_dir=str(tmp_path),
            allowed_extensions=[".mp4"],
        )
        source = FtpSource(config, camera_name="cam")
        emitted: list[Clip] = []
        source.register_callback(lambda clip: emitted.append(clip))

        clip_file = tmp_path / "video.mp4"
        clip_file.write_bytes(b"video")

        # When: Handling file received
        source._handle_file_received(clip_file)

        # Then: Clip is emitted
        assert len(emitted) == 1
        assert emitted[0].clip_id == "video"

    def test_disallowed_extension_kept_when_not_configured(self, tmp_path: Path) -> None:
        """Disallowed extensions are kept when delete_non_matching=False."""
        # Given: An FtpSource that doesn't delete non-matching
        config = FtpSourceConfig(
            root_dir=str(tmp_path),
            allowed_extensions=[".mp4"],
            delete_non_matching=False,
        )
        source = FtpSource(config, camera_name="cam")
        emitted: list[Clip] = []
        source.register_callback(lambda clip: emitted.append(clip))

        bad_file = tmp_path / "doc.txt"
        bad_file.write_text("text")

        # When: Handling non-matching file
        source._handle_file_received(bad_file)

        # Then: No clip emitted but file is kept
        assert emitted == []
        assert bad_file.exists()

    def test_incomplete_upload_kept_when_not_configured(self, tmp_path: Path) -> None:
        """Incomplete uploads are kept when delete_incomplete=False."""
        # Given: An FtpSource that doesn't delete incomplete
        config = FtpSourceConfig(
            root_dir=str(tmp_path),
            delete_incomplete=False,
        )
        source = FtpSource(config, camera_name="cam")

        incomplete = tmp_path / "partial.mp4"
        incomplete.write_bytes(b"partial")

        # When: Handling incomplete file
        source._handle_incomplete_file(incomplete)

        # Then: File is kept
        assert incomplete.exists()

    def test_all_extensions_allowed_when_empty(self, tmp_path: Path) -> None:
        """All extensions are allowed when allowed_extensions is empty."""
        # Given: An FtpSource with no extension filter
        config = FtpSourceConfig(
            root_dir=str(tmp_path),
            allowed_extensions=[],  # Empty = allow all
        )
        source = FtpSource(config, camera_name="cam")
        emitted: list[Clip] = []
        source.register_callback(lambda clip: emitted.append(clip))

        # When: Handling various file types
        for ext in [".mp4", ".avi", ".txt", ".jpg"]:
            file = tmp_path / f"file{ext}"
            file.write_bytes(b"data")
            source._handle_file_received(file)

        # Then: All clips are emitted
        assert len(emitted) == 4


class TestFtpSourceConfiguration:
    """Tests for FtpSource configuration."""

    def test_raises_without_credentials_when_not_anonymous(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Raises RuntimeError when non-anonymous but no credentials."""
        # Given: Config with anonymous=False but no env vars
        monkeypatch.delenv("FTP_USER", raising=False)
        monkeypatch.delenv("FTP_PASS", raising=False)
        config = FtpSourceConfig(
            root_dir=str(tmp_path),
            anonymous=False,
            username_env="FTP_USER",
            password_env="FTP_PASS",
        )

        # When/Then: Creating source raises RuntimeError
        with pytest.raises(RuntimeError, match="username/password"):
            FtpSource(config, camera_name="cam")

    def test_anonymous_mode_no_credentials_required(self, tmp_path: Path) -> None:
        """Anonymous mode doesn't require credentials."""
        # Given: Config with anonymous=True
        config = FtpSourceConfig(
            root_dir=str(tmp_path),
            anonymous=True,
        )

        # When: Creating source
        source = FtpSource(config, camera_name="cam")

        # Then: No error (credentials not required)
        assert source is not None

    def test_ftp_subdir_appended_to_root(self, tmp_path: Path) -> None:
        """ftp_subdir is appended to root_dir."""
        # Given: Config with ftp_subdir
        config = FtpSourceConfig(
            root_dir=str(tmp_path),
            ftp_subdir="camera_uploads",
        )

        # When: Creating source
        source = FtpSource(config, camera_name="cam")

        # Then: root_dir includes subdir
        assert source.root_dir == tmp_path / "camera_uploads"


class TestFtpSourceResolveEnv:
    """Tests for environment variable resolution."""

    def test_resolve_env_returns_value(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Returns env var value when set."""
        # Given: An env var is set
        monkeypatch.setenv("TEST_VAR", "test_value")
        config = FtpSourceConfig(root_dir=str(tmp_path))
        source = FtpSource(config, camera_name="cam")

        # When: Resolving the env var
        result = source._resolve_env("TEST_VAR")

        # Then: Returns the value
        assert result == "test_value"

    def test_resolve_env_returns_none_when_not_set(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Returns None when env var not set."""
        # Given: An env var is not set
        monkeypatch.delenv("MISSING_VAR", raising=False)
        config = FtpSourceConfig(root_dir=str(tmp_path))
        source = FtpSource(config, camera_name="cam")

        # When: Resolving the env var
        result = source._resolve_env("MISSING_VAR")

        # Then: Returns None
        assert result is None

    def test_resolve_env_returns_none_for_none_name(self, tmp_path: Path) -> None:
        """Returns None when name is None."""
        # Given: None env var name
        config = FtpSourceConfig(root_dir=str(tmp_path))
        source = FtpSource(config, camera_name="cam")

        # When: Resolving None
        result = source._resolve_env(None)

        # Then: Returns None
        assert result is None


class TestFtpSourceExtensionCheck:
    """Tests for extension checking logic."""

    def test_is_extension_allowed_case_insensitive(self, tmp_path: Path) -> None:
        """Extension check is case-insensitive."""
        # Given: An FtpSource with lowercase extension filter
        config = FtpSourceConfig(
            root_dir=str(tmp_path),
            allowed_extensions=[".mp4"],
        )
        source = FtpSource(config, camera_name="cam")

        # When: Checking various cases
        uppercase = source._is_extension_allowed(Path("video.MP4"))
        mixed = source._is_extension_allowed(Path("video.Mp4"))
        lowercase = source._is_extension_allowed(Path("video.mp4"))

        # Then: All cases are allowed
        assert uppercase is True
        assert mixed is True
        assert lowercase is True

    def test_is_extension_allowed_rejects_wrong_extension(self, tmp_path: Path) -> None:
        """Rejects extensions not in allowed list."""
        # Given: An FtpSource with specific extensions
        config = FtpSourceConfig(
            root_dir=str(tmp_path),
            allowed_extensions=[".mp4", ".avi"],
        )
        source = FtpSource(config, camera_name="cam")

        # When: Checking wrong extension
        result = source._is_extension_allowed(Path("video.mkv"))

        # Then: Not allowed
        assert result is False
