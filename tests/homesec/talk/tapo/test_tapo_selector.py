"""Selector tests for the built-in Tapo local backend."""

from __future__ import annotations

import hashlib

import pytest
from pydantic import BaseModel

from homesec.models.config import CameraTalkConfig, TalkConfig
from homesec.models.talk import (
    TalkCapabilityProbeResult,
    TalkCapabilityState,
    TalkRefusalReason,
)
from homesec.talk.backends import TalkBackendContext, TalkBackendRegistration, TalkBackendRegistry
from homesec.talk.selector import TalkBackendSelector
from homesec.talk.tapo.backend import build_tapo_local_backend, tapo_local_talk_backend_registration
from homesec.talk.tapo.config import TapoLocalTalkConfig
from homesec.talk.tapo.session import TAPO_LOCAL_CODEC

from .fake_server import FakeTapoServer

_VALID_SHA256 = "A" * 64


class _StandardConfig(BaseModel):
    """Fake standards backend config for Tapo selector tests."""


class _StandardSession:
    session_id = "tk_onvif"
    camera_name = "office"
    selected_codec = "PCMU/8000"

    async def write_pcm_frame(self, frame: bytes) -> None:
        _ = frame

    async def close(self) -> None:
        return None


class _StandardBackend:
    name = "onvif_rtsp_backchannel"

    def __init__(
        self,
        *,
        calls: list[str],
        capability: TalkCapabilityState,
    ) -> None:
        self._calls = calls
        self._capability = capability

    @property
    def supported_codecs(self) -> list[str]:
        return ["PCMU/8000"]

    async def probe(self) -> TalkCapabilityProbeResult:
        self._calls.append("onvif:probe")
        if self._capability == TalkCapabilityState.SUPPORTED:
            return TalkCapabilityProbeResult(
                capability=TalkCapabilityState.SUPPORTED,
                offered_codecs=["PCMU/8000"],
                selected_codec="PCMU/8000",
            )
        return TalkCapabilityProbeResult(
            capability=TalkCapabilityState.UNSUPPORTED,
            refusal_reason=TalkRefusalReason.UNSUPPORTED_CAMERA,
            message="ONVIF backchannel unsupported",
        )

    async def open_session(self, request: object) -> _StandardSession:
        self._calls.append("onvif:open")
        _ = request
        return _StandardSession()


def _registry(
    *,
    calls: list[str],
    standard_capability: TalkCapabilityState = TalkCapabilityState.UNSUPPORTED,
) -> TalkBackendRegistry:
    registry = TalkBackendRegistry()
    registry.register(
        TalkBackendRegistration(
            name="onvif_rtsp_backchannel",
            config_model=_StandardConfig,
            factory=lambda config, context: _StandardBackend(
                calls=calls,
                capability=standard_capability,
            ),
            priority=10,
            standards_based=True,
        )
    )
    registry.register(tapo_local_talk_backend_registration())
    return registry


def _context(
    camera_talk: CameraTalkConfig,
    *,
    env: dict[str, str] | None = None,
) -> TalkBackendContext:
    values = env or {"OFFICE_TAPO_SHA256": _VALID_SHA256}
    return TalkBackendContext(
        camera_name="office",
        source_backend="rtsp",
        runtime_talk=TalkConfig(),
        camera_talk=camera_talk,
        source_uri="rtsp://admin:secret@192.168.1.33:554/stream1",
        resolved_source_uri="rtsp://admin:secret@192.168.1.33:554/stream1",
        source_connect_timeout_s=1.0,
        source_io_timeout_s=1.0,
        resolve_env=lambda name: values.get(name),
    )


@pytest.mark.asyncio
async def test_explicit_tapo_backend_does_not_fallback_to_onvif() -> None:
    """Explicit tapo_local selection should not probe standards backends."""
    # Given: A camera explicitly configured for the built-in Tapo local backend
    server = FakeTapoServer(hash_kind="sha256", credential_hash=_VALID_SHA256)
    await server.start()
    calls: list[str] = []
    try:
        selector = TalkBackendSelector(
            registry=_registry(calls=calls),
            context=_context(
                CameraTalkConfig(
                    backend="tapo_local",
                    backends={
                        "tapo_local": {
                            "host": server.host,
                            "port": server.port,
                            "password_sha256_env": "OFFICE_TAPO_SHA256",
                        }
                    },
                )
            ),
        )

        # When: Probing through explicit backend selection
        probe = await selector.probe()

        # Then: Tapo is selected, ONVIF is not probed, and local probing succeeds
        assert selector.backend == "tapo_local"
        assert selector.supported_codecs == [TAPO_LOCAL_CODEC]
        assert probe.capability == TalkCapabilityState.SUPPORTED
        assert probe.selected_codec == TAPO_LOCAL_CODEC
        assert calls == []
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_explicit_tapo_backend_can_default_to_rtsp_credentials() -> None:
    """Explicit Tapo selection should work with only backend selection configured."""
    # Given: A fake Tapo endpoint expecting the SHA256 hash of the RTSP password
    server = FakeTapoServer(
        hash_kind="sha256",
        credential_hash=hashlib.sha256(b"secret").hexdigest().upper(),
    )
    await server.start()
    calls: list[str] = []
    try:
        selector = TalkBackendSelector(
            registry=_registry(calls=calls),
            context=_context(
                CameraTalkConfig(
                    backend="tapo_local",
                    backends={"tapo_local": {"host": server.host, "port": server.port}},
                ),
                env={},
            ),
        )

        # When: Probing through explicit backend selection with no Tapo credential env
        probe = await selector.probe()

        # Then: Tapo derives username/password material from the RTSP source URL
        assert selector.backend == "tapo_local"
        assert probe.capability == TalkCapabilityState.SUPPORTED
        assert probe.selected_codec == TAPO_LOCAL_CODEC
        assert calls == []
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_explicit_tapo_missing_env_reports_config_error_without_onvif_probe() -> None:
    """Explicit Tapo config errors should not fallback to ONVIF probing."""
    # Given: A camera explicitly configured for Tapo with an unset credential env var
    calls: list[str] = []
    selector = TalkBackendSelector(
        registry=_registry(calls=calls),
        context=TalkBackendContext(
            camera_name="office",
            source_backend="rtsp",
            runtime_talk=TalkConfig(),
            camera_talk=CameraTalkConfig(
                backend="tapo_local",
                backends={
                    "tapo_local": {
                        "host": "192.168.1.33",
                        "password_sha256_env": "MISSING_TAPO_SHA256",
                    }
                },
            ),
            resolve_env=lambda name: None,
        ),
    )

    # When: Probing through explicit backend selection
    probe = await selector.probe()

    # Then: A stable config error is returned and standards backends remain untouched
    assert probe.capability == TalkCapabilityState.CONFIG_ERROR
    assert probe.refusal_reason == TalkRefusalReason.TALK_CONFIG_ERROR
    assert (
        probe.message == "Required Tapo local environment variable is not set: MISSING_TAPO_SHA256"
    )
    assert selector.backend == "tapo_local"
    assert calls == []


@pytest.mark.asyncio
async def test_explicit_tapo_missing_password_named_env_preserves_env_name() -> None:
    """Tapo config errors should preserve safe env names that include PASSWORD."""
    # Given: A camera explicitly configured with a common password-named env var
    calls: list[str] = []
    selector = TalkBackendSelector(
        registry=_registry(calls=calls),
        context=TalkBackendContext(
            camera_name="office",
            source_backend="rtsp",
            runtime_talk=TalkConfig(),
            camera_talk=CameraTalkConfig(
                backend="tapo_local",
                backends={
                    "tapo_local": {
                        "host": "192.168.1.33",
                        "password_sha256_env": "TAPO_PASSWORD_SHA256",
                    }
                },
            ),
            resolve_env=lambda name: None,
        ),
    )

    # When: Probing through explicit backend selection
    probe = await selector.probe()

    # Then: The missing env var name remains visible and ONVIF remains untouched
    assert probe.capability == TalkCapabilityState.CONFIG_ERROR
    assert probe.refusal_reason == TalkRefusalReason.TALK_CONFIG_ERROR
    assert probe.message == (
        "Required Tapo local environment variable is not set: TAPO_PASSWORD_SHA256"
    )
    assert selector.backend_reason == probe.message
    assert calls == []


def test_tapo_backend_repr_does_not_expose_hash_or_rtsp_credentials() -> None:
    """Tapo backend repr should not expose credential hashes or source URIs."""
    # Given: A Tapo backend built with a source URI containing RTSP credentials
    backend = build_tapo_local_backend(
        TapoLocalTalkConfig(
            host="192.168.1.33",
            password_sha256_env="OFFICE_TAPO_SHA256",
        ),
        _context(
            CameraTalkConfig(
                backend="tapo_local",
                backends={"tapo_local": {"password_sha256_env": "OFFICE_TAPO_SHA256"}},
            )
        ),
    )

    # When: Formatting the backend for debugging
    text = repr(backend)

    # Then: The repr keeps secret-bearing fields out of accidental logs
    assert _VALID_SHA256 not in text
    assert "admin:secret" not in text
    assert "rtsp://admin:secret@192.168.1.33" not in text


@pytest.mark.asyncio
async def test_auto_tapo_config_is_considered_after_onvif_unsupported() -> None:
    """Auto mode should consider Tapo only after standards probing fails."""
    # Given: ONVIF is unsupported and Tapo config makes proprietary probing safe
    server = FakeTapoServer(hash_kind="sha256", credential_hash=_VALID_SHA256)
    await server.start()
    calls: list[str] = []
    try:
        selector = TalkBackendSelector(
            registry=_registry(calls=calls),
            context=_context(
                CameraTalkConfig(
                    backend="auto",
                    backends={
                        "tapo_local": {
                            "host": server.host,
                            "port": server.port,
                            "password_sha256_env": "OFFICE_TAPO_SHA256",
                        }
                    },
                )
            ),
        )

        # When: Probing auto selection
        probe = await selector.probe()

        # Then: ONVIF is tried first and Tapo becomes the selected safe fallback
        assert probe.capability == TalkCapabilityState.SUPPORTED
        assert probe.selected_codec == TAPO_LOCAL_CODEC
        assert selector.backend == "tapo_local"
        assert (
            selector.backend_reason == "Selected talk backend 'tapo_local' by safe camera detector"
        )
        assert calls == ["onvif:probe"]
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_auto_ignores_tapo_without_config_or_fingerprint() -> None:
    """Auto mode should not probe Tapo without config or fingerprint evidence."""
    # Given: ONVIF is unsupported and no Tapo config/fingerprint is present
    calls: list[str] = []
    selector = TalkBackendSelector(
        registry=_registry(calls=calls),
        context=_context(CameraTalkConfig(backend="auto")),
    )

    # When: Probing auto selection
    probe = await selector.probe()

    # Then: The ONVIF unsupported result remains visible and Tapo is ignored
    assert probe.capability == TalkCapabilityState.UNSUPPORTED
    assert probe.refusal_reason == TalkRefusalReason.UNSUPPORTED_CAMERA
    assert selector.backend == "onvif_rtsp_backchannel"
    assert calls == ["onvif:probe"]


@pytest.mark.asyncio
async def test_auto_keeps_supported_onvif_ahead_of_tapo_config() -> None:
    """Auto mode should keep standards-first behavior even when Tapo config exists."""
    # Given: ONVIF is supported and Tapo config is also present
    calls: list[str] = []
    selector = TalkBackendSelector(
        registry=_registry(
            calls=calls,
            standard_capability=TalkCapabilityState.SUPPORTED,
        ),
        context=_context(
            CameraTalkConfig(
                backend="auto",
                backends={
                    "tapo_local": {
                        "host": "192.168.1.33",
                        "password_sha256_env": "OFFICE_TAPO_SHA256",
                    }
                },
            )
        ),
    )

    # When: Probing auto selection
    probe = await selector.probe()

    # Then: ONVIF wins before the Tapo detector path is reached
    assert probe.capability == TalkCapabilityState.SUPPORTED
    assert selector.backend == "onvif_rtsp_backchannel"
    assert calls == ["onvif:probe"]
