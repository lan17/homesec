"""Tests for retention wiring helpers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest

from homesec.models.config import RetentionConfig
from homesec.repository import ClipRepository
from homesec.retention.wiring import (
    build_local_retention_pruner,
    discover_local_clip_dirs,
    resolve_max_local_size_bytes,
)
from tests.homesec.mocks import MockEventStore, MockStateStore


class _StubSource:
    def __init__(self, **attrs: Any) -> None:
        for key, value in attrs.items():
            setattr(self, key, value)


def test_discover_local_clip_dirs_collects_known_attributes_dedupes_and_sorts(
    tmp_path: Path,
) -> None:
    # Given: Sources exposing watch/root/output paths with duplicates and mixed path value types
    dir_a = tmp_path / "clips-a"
    dir_b = tmp_path / "clips-b"
    dir_c = tmp_path / "clips-c"
    sources = [
        _StubSource(watch_dir=str(dir_b)),
        _StubSource(root_dir=dir_a),
        _StubSource(output_dir=dir_a),
        _StubSource(watch_dir=123, output_dir=str(dir_c)),
    ]

    # When: Discovering local clip directories for retention scanning
    discovered = discover_local_clip_dirs(sources=sources)  # type: ignore[arg-type]

    # Then: Paths are deduped, normalized, and sorted deterministically
    assert discovered == [dir_a, dir_b, dir_c]


def test_discover_local_clip_dirs_logs_warning_for_sources_without_supported_paths(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Given: Sources without any valid retention path attributes
    sources = [_StubSource(), _StubSource(watch_dir=123)]
    caplog.set_level(logging.WARNING)

    # When: Discovering local clip directories
    discovered = discover_local_clip_dirs(sources=sources)  # type: ignore[arg-type]

    # Then: Discovery returns no directories and logs one warning per source
    assert discovered == []
    warning_messages = [
        record.message
        for record in caplog.records
        if "Retention source dir discovery skipped" in record.message
    ]
    assert len(warning_messages) == 2


@pytest.mark.asyncio
async def test_build_local_retention_pruner_warns_when_no_dirs_and_uses_configured_cap(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Given: A configured retention cap with no discoverable source directories
    retention = RetentionConfig(max_local_size_bytes=321)
    repository = ClipRepository(MockStateStore(), MockEventStore())
    caplog.set_level(logging.WARNING)

    # When: Building pruner from runtime wiring and running a prune pass
    resolved_cap = resolve_max_local_size_bytes(retention)
    pruner = build_local_retention_pruner(
        repository=repository,
        retention=retention,
        sources=[],
    )
    summary = await pruner.prune_once(reason="test")

    # Then: Builder logs a no-op warning and prune summary honors configured cap
    assert resolved_cap == 321
    assert summary.max_local_size_bytes == 321
    assert summary.discovered_local_files == 0
    assert "Retention pruner has no local clip directories" in caplog.text
