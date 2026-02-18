"""Tests for runtime worker entrypoint behavior."""

from __future__ import annotations

from argparse import Namespace

import homesec.runtime.worker as worker_module


def test_worker_main_uses_shared_logging_configuration(monkeypatch) -> None:
    """Worker main should configure shared logging before running service."""
    # Given: Parsed args and patched runtime runner hooks
    parsed_args = Namespace(generation=7)
    calls: dict[str, object] = {}

    def _fake_parse_args(argv: list[str]) -> Namespace:
        calls["argv"] = list(argv)
        return parsed_args

    def _fake_configure_logging(*, log_level: str, camera_name: str | None = None) -> None:
        calls["log_level"] = log_level
        calls["camera_name"] = camera_name

    def _fake_asyncio_run(coro: object) -> None:
        calls["ran"] = True
        # Avoid un-awaited coroutine warnings in tests.
        coro.close()

    monkeypatch.setattr(worker_module, "_parse_args", _fake_parse_args)
    monkeypatch.setattr(worker_module, "configure_logging", _fake_configure_logging)
    monkeypatch.setattr(worker_module.asyncio, "run", _fake_asyncio_run)

    # When: Running worker main entrypoint
    worker_module.main()

    # Then: Worker uses the same logging pipeline with worker-scoped camera name
    assert calls["log_level"] == "INFO"
    assert calls["camera_name"] == "runtime-worker-g7"
    assert calls["ran"] is True
