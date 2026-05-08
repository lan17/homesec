"""Tests for Tapo local backend registration and detector behavior."""

from __future__ import annotations

import asyncio
import socket

import pytest

from homesec.models.config import CameraTalkConfig, TalkConfig
from homesec.talk.backends import CameraTalkFingerprint, TalkBackendContext
from homesec.talk.registry import build_default_talk_backend_registry
from homesec.talk.tapo.backend import (
    TAPO_LOCAL_BACKEND,
    detect_tapo_local,
    tapo_local_talk_backend_registration,
)


def _context(
    camera_talk: CameraTalkConfig,
    *,
    fingerprint: CameraTalkFingerprint | None = None,
) -> TalkBackendContext:
    return TalkBackendContext(
        camera_name="office",
        source_backend="rtsp",
        runtime_talk=TalkConfig(),
        camera_talk=camera_talk,
        fingerprint=fingerprint or CameraTalkFingerprint(),
    )


def test_tapo_registration_is_non_standard_backend() -> None:
    """Tapo backend registration should be a proprietary detector-gated backend."""
    # Given: The built-in Tapo backend registration
    # When: Inspecting registration metadata
    registration = tapo_local_talk_backend_registration()

    # Then: It uses the public backend ID and stays outside standards-first probing
    assert registration.name == TAPO_LOCAL_BACKEND
    assert registration.standards_based is False
    assert registration.priority == 50
    assert registration.detector is detect_tapo_local


def test_default_registry_contains_onvif_before_tapo() -> None:
    """Default registry should keep ONVIF standards probing before Tapo fallback."""
    # Given: The built-in HomeSec talk backend registry
    # When: Reading registered backend names
    registry = build_default_talk_backend_registry()

    # Then: ONVIF remains first and Tapo is available as the proprietary fallback
    assert registry.names() == ("onvif_rtsp_backchannel", "tapo_local")
    assert registry.get("onvif_rtsp_backchannel") is not None
    tapo = registry.get("tapo_local")
    assert tapo is not None
    assert tapo.standards_based is False


def test_detector_marks_explicit_tapo_backend_safe_to_probe() -> None:
    """Explicit tapo_local selection should be safe to probe."""
    # Given: A camera explicitly configured for tapo_local
    context = _context(CameraTalkConfig(backend="tapo_local"))

    # When: Running the Tapo detector
    detection = detect_tapo_local(context)

    # Then: The detector returns an explicit safe candidate
    assert detection.backend == "tapo_local"
    assert detection.confidence == "explicit"
    assert detection.safe_to_probe is True
    assert detection.requires_credentials is True


def test_detector_marks_tapo_config_block_safe_to_probe_in_auto_mode() -> None:
    """A tapo_local config block should make auto fallback safe after standards fail."""
    # Given: A camera in auto mode with Tapo backend-specific config
    context = _context(
        CameraTalkConfig(
            backend="auto",
            backends={"tapo_local": {"host": "192.168.1.33"}},
        )
    )

    # When: Running the Tapo detector
    detection = detect_tapo_local(context)

    # Then: Tapo is a safe explicit candidate for selector fallback
    assert detection.backend == "tapo_local"
    assert detection.confidence == "explicit"
    assert detection.safe_to_probe is True
    assert detection.reason == "Tapo local backend config present"


def test_detector_marks_tp_link_tapo_fingerprint_safe_to_probe() -> None:
    """A TP-Link/Tapo fingerprint should make Tapo auto probing safe."""
    # Given: A camera fingerprint matching a Tapo C120
    context = _context(
        CameraTalkConfig(backend="auto"),
        fingerprint=CameraTalkFingerprint(manufacturer="TP-Link", model="Tapo C120"),
    )

    # When: Running the Tapo detector
    detection = detect_tapo_local(context)

    # Then: The detector returns a high-confidence safe candidate
    assert detection.backend == "tapo_local"
    assert detection.confidence == "high"
    assert detection.safe_to_probe is True
    assert detection.reason == "TP-Link/Tapo camera fingerprint"


def test_detector_rejects_auto_without_config_or_fingerprint() -> None:
    """Auto mode should not probe Tapo without config or a safe fingerprint."""
    # Given: A generic camera with no Tapo config or fingerprint
    context = _context(CameraTalkConfig(backend="auto"))

    # When: Running the Tapo detector
    detection = detect_tapo_local(context)

    # Then: Tapo is not a safe candidate
    assert detection.backend == "tapo_local"
    assert detection.confidence == "not_applicable"
    assert detection.safe_to_probe is False
    assert detection.reason == "No Tapo local config or fingerprint"


def test_detector_does_not_perform_network_io(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tapo detection should stay synchronous and local-metadata-only."""

    async def _forbidden_async_open(*_: object, **__: object) -> object:
        raise AssertionError("detector must not open network connections")

    def _forbidden_socket_connect(*_: object, **__: object) -> object:
        raise AssertionError("detector must not open network connections")

    monkeypatch.setattr(asyncio, "open_connection", _forbidden_async_open)
    monkeypatch.setattr(socket, "create_connection", _forbidden_socket_connect)
    context = _context(
        CameraTalkConfig(
            backend="auto",
            backends={"tapo_local": {"host": "192.168.1.33"}},
        )
    )

    # Given: Network connection helpers patched to fail if called
    # When: Running the detector
    detection = detect_tapo_local(context)

    # Then: Detection succeeds without touching network APIs
    assert detection.safe_to_probe is True
