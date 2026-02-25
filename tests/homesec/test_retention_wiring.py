"""Tests for retention wiring helpers."""

from __future__ import annotations

import logging

import pytest

from homesec.models.config import RetentionConfig
from homesec.repository import ClipRepository
from homesec.retention.wiring import build_local_retention_pruner
from tests.homesec.mocks import MockEventStore, MockStateStore


@pytest.mark.asyncio
async def test_build_local_retention_pruner_starts_empty_and_uses_configured_cap(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Given: A configured retention cap and retention wiring dependencies
    retention = RetentionConfig(max_local_size_bytes=321)
    repository = ClipRepository(MockStateStore(), MockEventStore())
    caplog.set_level(logging.INFO)

    # When: Building a retention pruner and running a prune pass before any clip arrival
    pruner = build_local_retention_pruner(
        repository=repository,
        retention=retention,
    )
    summary = await pruner.prune_once(reason="test")

    # Then: Pruner starts with no local roots and still uses configured byte cap
    assert summary.max_local_size_bytes == 321
    assert summary.discovered_local_files == 0
    assert "policy=learn_from_clip_local_path" in caplog.text
