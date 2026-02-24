"""Behavioral tests for clip media route helper functions."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from homesec.api.routes.media import (
    _build_media_filename,
    _cleanup_media_temp_dir,
    _guess_media_type,
    _infer_media_suffix,
)


def test_cleanup_media_temp_dir_ignores_missing_directory(tmp_path: Path) -> None:
    """Cleanup helper should silently ignore missing temp dirs."""
    # Given: A temp directory path that no longer exists
    missing_dir = tmp_path / "missing-media-dir"

    # When: Cleanup is requested for the missing directory
    _cleanup_media_temp_dir(missing_dir)

    # Then: The helper completes without raising
    assert missing_dir.exists() is False


def test_cleanup_media_temp_dir_logs_warning_on_unexpected_error(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    """Cleanup helper should warn on unexpected filesystem errors."""
    # Given: rmtree raises an unexpected error while cleaning a temp directory
    temp_dir = tmp_path / "media-dir"
    temp_dir.mkdir()

    def _raise_permission_error(_: Path) -> None:
        raise PermissionError("permission denied")

    monkeypatch.setattr("homesec.api.routes.media.shutil.rmtree", _raise_permission_error)

    # When: Cleanup is executed
    with caplog.at_level(logging.WARNING):
        _cleanup_media_temp_dir(temp_dir)

    # Then: A warning is logged with cleanup failure context
    assert any("Failed to remove temp media dir" in rec.message for rec in caplog.records)


def test_infer_media_suffix_parses_uri_and_defaults_mp4() -> None:
    """Suffix inference should strip query/fragment and default to mp4 when absent."""
    # Given: Storage URIs with and without explicit media suffixes
    explicit_suffix_uri = "dropbox:/clips/front/abc.webm?raw=1#fragment"
    no_suffix_uri = "dropbox:/clips/front/abc123"

    # When: Inferring file suffix from each URI
    explicit_suffix = _infer_media_suffix(explicit_suffix_uri)
    fallback_suffix = _infer_media_suffix(no_suffix_uri)

    # Then: Explicit suffix is preserved and missing suffix defaults to .mp4
    assert explicit_suffix == ".webm"
    assert fallback_suffix == ".mp4"


def test_build_media_filename_sanitizes_clip_id_and_normalizes_suffix() -> None:
    """Filename builder should sanitize clip IDs and normalize invalid suffixes."""
    # Given: A clip ID containing unsafe filename characters
    clip_id = "garage/clip:2026-02-15 10:00"

    # When: Building filenames with valid and invalid suffix forms
    with_dot = _build_media_filename(clip_id, ".mp4")
    without_dot = _build_media_filename(clip_id, "mp4")

    # Then: Clip ID is sanitized and invalid suffix format falls back to .mp4
    assert with_dot == "garage_clip_2026-02-15_10_00.mp4"
    assert without_dot == "garage_clip_2026-02-15_10_00.mp4"


def test_guess_media_type_falls_back_to_octet_stream_for_unknown_extension() -> None:
    """Media type guess should return octet-stream when MIME type is unknown."""
    # Given: A path with an unknown extension
    unknown_path = Path("clip.unknownext")

    # When: Guessing media type
    media_type = _guess_media_type(unknown_path)

    # Then: Fallback media type is used
    assert media_type == "application/octet-stream"
