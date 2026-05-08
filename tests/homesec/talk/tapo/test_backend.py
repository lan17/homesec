"""Tests for Tapo local talk backend integration."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest
from pydantic import BaseModel

from homesec.models.config import CameraTalkConfig, TalkConfig
from homesec.models.talk import (
    TalkCapabilityProbeResult,
    TalkCapabilityState,
    TalkInputFormat,
    TalkRefusalReason,
    TalkSessionOpenRequest,
    TalkSessionPrepareRequest,
)
from homesec.sources.rtsp.talk.manager import TalkManager
from homesec.talk.backends import TalkBackendContext, TalkBackendRegistration, TalkBackendRegistry
from homesec.talk.registry import build_default_talk_backend_registry
from homesec.talk.selector import TalkBackendSelector
from homesec.talk.tapo.backend import TAPO_LOCAL_BACKEND, tapo_local_talk_backend_registration

from .fake_server import FakeTapoServer

_SHA256 = "A" * 64
_MD5 = "B" * 32


class _StandardConfig(BaseModel):
    model_config = {"extra": "forbid"}


@dataclass(slots=True)
class _StandardBackend:
    calls: list[str]
    supported: bool

    name: str = "onvif_rtsp_backchannel"

    @property
    def supported_codecs(self) -> list[str]:
        return ["PCMU/8000"]

    async def probe(self) -> TalkCapabilityProbeResult:
        self.calls.append("onvif:probe")
        if self.supported:
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

    async def open_session(self, request: TalkSessionOpenRequest) -> _StandardSession:
        self.calls.append(f"onvif:open:{request.session_id}")
        return _StandardSession(session_id=request.session_id)


@dataclass(slots=True)
class _StandardSession:
    session_id: str
    camera_name: str = "office"
    selected_codec: str = "PCMU/8000"

    async def write_pcm_frame(self, frame: bytes) -> None:
        _ = frame

    async def close(self) -> None:
        return None


def _context(
    camera_talk: CameraTalkConfig,
    *,
    env: dict[str, str] | None = None,
) -> TalkBackendContext:
    values = env or {}
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


def _tapo_talk_config(server: FakeTapoServer) -> CameraTalkConfig:
    return CameraTalkConfig(
        backend=TAPO_LOCAL_BACKEND,
        backends={
            TAPO_LOCAL_BACKEND: {
                "host": server.host,
                "port": server.port,
                "password_sha256_env": "OFFICE_TAPO_SHA256",
            }
        },
    )


def _tapo_manager(selector: TalkBackendSelector) -> TalkManager:
    return TalkManager(
        camera_name="office",
        enabled=True,
        policy_enabled=True,
        supported_codecs=selector.supported_codecs,
        supported_codecs_factory=lambda: selector.selected_supported_codecs,
        open_session_factory=selector.open_session,
        capability_probe_factory=selector.probe,
        prepare_capability_probe_factory=selector.probe_for_session_open,
        prepare_probe_cleanup=selector.clear_prepared_probe,
        max_session_s=60.0,
        idle_timeout_s=60.0,
    )


def _registry_with_standard(*, calls: list[str], supported: bool) -> TalkBackendRegistry:
    registry = TalkBackendRegistry()
    registry.register(
        TalkBackendRegistration(
            name="onvif_rtsp_backchannel",
            config_model=_StandardConfig,
            factory=lambda config, context: _StandardBackend(calls=calls, supported=supported),
            priority=10,
            standards_based=True,
        )
    )
    registry.register(tapo_local_talk_backend_registration())
    return registry


@pytest.mark.asyncio
async def test_explicit_tapo_backend_streams_pcm_through_talk_manager() -> None:
    """Explicit Tapo backend should run through the existing TalkManager lifecycle."""
    # Given: A fake Tapo endpoint and explicit tapo_local backend config
    server = FakeTapoServer(hash_kind="sha256", credential_hash=_SHA256)
    await server.start()
    try:
        selector = TalkBackendSelector(
            registry=build_default_talk_backend_registry(),
            context=_context(_tapo_talk_config(server), env={"OFFICE_TAPO_SHA256": _SHA256}),
        )
        manager = _tapo_manager(selector)
        frame = b"\x00" * TalkInputFormat().expected_bytes_per_frame

        # When: Preparing, opening, writing, and stopping a talk session
        prepared = await manager.prepare_session(TalkSessionPrepareRequest(session_id="tk_tapo"))
        session = await manager.open_session(TalkSessionOpenRequest(session_id="tk_tapo"))
        await manager.write_pcm_frame("tk_tapo", frame)
        await _wait_for_audio_parts(server, count=2)
        stopped = await manager.stop_session("tk_tapo")

        # Then: Browser PCM reaches the fake Tapo stream as MPEG-TS audio parts
        assert prepared.accepted is True
        assert session.selected_codec == "PCMA/8000"
        assert manager.status().selected_codec == "PCMA/8000"
        assert stopped is True
        assert len(server.requests) == 2
        assert len(server.audio_parts) == 2
        assert server.audio_parts[0].header("content-type") == "audio/mp2t"
        assert server.audio_parts[0].body.startswith(b"\x47")
        assert len(server.audio_parts[0].body) == 188 * 2
        assert server.audio_parts[1].body.startswith(b"\x47")
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_abandoned_tapo_prepare_closes_prepared_client() -> None:
    """Stopping a reserved Tapo talk session should clear the prepared client immediately."""
    # Given: A prepared Tapo session that never opens a HomeSec talk stream
    server = FakeTapoServer(hash_kind="sha256", credential_hash=_SHA256)
    await server.start()
    try:
        selector = TalkBackendSelector(
            registry=build_default_talk_backend_registry(),
            context=_context(_tapo_talk_config(server), env={"OFFICE_TAPO_SHA256": _SHA256}),
        )
        manager = _tapo_manager(selector)

        # When: The reservation is stopped before WebSocket stream open
        prepared = await manager.prepare_session(
            TalkSessionPrepareRequest(session_id="tk_abandoned")
        )
        stopped = await manager.stop_session("tk_abandoned")
        await _wait_for_closed_connection(server)

        # Then: The prepared probe connection is closed without sending MPEG-TS audio
        assert prepared.accepted is True
        assert stopped is True
        assert server.audio_parts == []
        assert server.closed_connections >= 1
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_tapo_backend_maps_auth_failure_to_talk_auth_failed() -> None:
    """Rejected Tapo credential hashes should surface the stable auth refusal reason."""
    # Given: A fake Tapo endpoint and a wrong SHA256 hash in config env
    server = FakeTapoServer(hash_kind="sha256", credential_hash=_SHA256)
    await server.start()
    try:
        wrong_hash = "C" * 64
        selector = TalkBackendSelector(
            registry=build_default_talk_backend_registry(),
            context=_context(_tapo_talk_config(server), env={"OFFICE_TAPO_SHA256": wrong_hash}),
        )
        manager = _tapo_manager(selector)

        # When: Preparing the session probes the fake endpoint
        prepared = await manager.prepare_session(TalkSessionPrepareRequest(session_id="tk_auth"))

        # Then: The refusal is auth-specific and never exposes hash material
        assert prepared.accepted is False
        assert prepared.refusal_reason == TalkRefusalReason.TALK_AUTH_FAILED
        assert prepared.message == "Tapo local authentication failed"
        assert wrong_hash not in (manager.status().last_error or "")
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_tapo_backend_maps_missing_credential_to_talk_config_error() -> None:
    """Missing Tapo credential env vars should fail as talk_config_error."""
    # Given: Explicit Tapo backend config with an unset credential env var
    talk = CameraTalkConfig(
        backend=TAPO_LOCAL_BACKEND,
        backends={
            TAPO_LOCAL_BACKEND: {
                "host": "192.168.1.33",
                "port": 8800,
                "password_sha256_env": "OFFICE_TAPO_SHA256",
            }
        },
    )
    selector = TalkBackendSelector(
        registry=build_default_talk_backend_registry(),
        context=_context(talk, env={}),
    )

    # When: Probing the explicit Tapo backend
    probe = await selector.probe()

    # Then: Selection fails before network open with a safe env-var diagnostic
    assert probe.capability == TalkCapabilityState.CONFIG_ERROR
    assert probe.refusal_reason == TalkRefusalReason.TALK_CONFIG_ERROR
    assert (
        probe.message == "Required Tapo local environment variable is not set: OFFICE_TAPO_SHA256"
    )
    assert selector.backend == TAPO_LOCAL_BACKEND


@pytest.mark.asyncio
async def test_tapo_backend_maps_protocol_failure_to_camera_backchannel_failed() -> None:
    """Malformed Tapo setup responses should surface as camera backchannel failures."""
    # Given: A fake Tapo endpoint that returns malformed setup JSON
    server = FakeTapoServer(
        hash_kind="sha256",
        credential_hash=_SHA256,
        malformed_setup_json=True,
    )
    await server.start()
    try:
        selector = TalkBackendSelector(
            registry=build_default_talk_backend_registry(),
            context=_context(_tapo_talk_config(server), env={"OFFICE_TAPO_SHA256": _SHA256}),
        )
        manager = _tapo_manager(selector)

        # When: Preparing the session probes setup and receives malformed protocol data
        prepared = await manager.prepare_session(
            TalkSessionPrepareRequest(session_id="tk_protocol")
        )

        # Then: The failure maps to the stable camera backchannel refusal reason
        assert prepared.accepted is False
        assert prepared.refusal_reason == TalkRefusalReason.CAMERA_BACKCHANNEL_FAILED
        assert prepared.message == "Tapo local talk protocol failed"
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_auto_selects_tapo_after_onvif_is_unsupported() -> None:
    """Auto mode should pick Tapo after standards probing fails and Tapo config exists."""
    # Given: ONVIF is unsupported and Tapo config makes local probing safe
    server = FakeTapoServer(hash_kind="sha256", credential_hash=_SHA256)
    await server.start()
    try:
        calls: list[str] = []
        selector = TalkBackendSelector(
            registry=_registry_with_standard(calls=calls, supported=False),
            context=_context(
                CameraTalkConfig(
                    backend="auto",
                    backends=_tapo_talk_config(server).backends,
                ),
                env={"OFFICE_TAPO_SHA256": _SHA256},
            ),
        )

        # When: Probing auto talk capability
        probe = await selector.probe()

        # Then: The standards backend is tried first, then tapo_local is selected
        assert probe.capability == TalkCapabilityState.SUPPORTED
        assert selector.backend == TAPO_LOCAL_BACKEND
        assert selector.selected_supported_codecs == ["PCMA/8000"]
        assert calls == ["onvif:probe"]
        assert len(server.requests) == 2
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_auto_keeps_supported_onvif_ahead_of_tapo_config() -> None:
    """Auto mode should keep standards-first behavior when ONVIF is supported."""
    # Given: A supported standards backend and a configured Tapo fallback
    calls: list[str] = []
    selector = TalkBackendSelector(
        registry=_registry_with_standard(calls=calls, supported=True),
        context=_context(
            CameraTalkConfig(
                backend="auto",
                backends={
                    TAPO_LOCAL_BACKEND: {
                        "host": "127.0.0.1",
                        "port": 1,
                        "password_sha256_env": "OFFICE_TAPO_SHA256",
                    }
                },
            ),
            env={"OFFICE_TAPO_SHA256": _SHA256},
        ),
    )

    # When: Probing auto talk capability
    probe = await selector.probe()

    # Then: ONVIF wins and Tapo is not probed
    assert probe.capability == TalkCapabilityState.SUPPORTED
    assert selector.backend == "onvif_rtsp_backchannel"
    assert selector.selected_supported_codecs == ["PCMU/8000"]
    assert calls == ["onvif:probe"]


def test_tapo_credential_falls_back_to_available_md5_env() -> None:
    """Tapo credential selection should try another configured hash env if preferred is unset."""
    # Given: SHA256 is preferred but unset while an MD5 hash env is configured
    server = FakeTapoServer(hash_kind="md5", credential_hash=_MD5)
    talk = CameraTalkConfig(
        backend=TAPO_LOCAL_BACKEND,
        backends={
            TAPO_LOCAL_BACKEND: {
                "host": server.host,
                "port": 8800,
                "password_sha256_env": "OFFICE_TAPO_SHA256",
                "password_md5_env": "OFFICE_TAPO_MD5",
            }
        },
    )

    # When: Building the explicit backend for static config validation
    selector = TalkBackendSelector(
        registry=build_default_talk_backend_registry(),
        context=_context(talk, env={"OFFICE_TAPO_MD5": _MD5}),
    )

    # Then: The backend is selectable instead of failing on the missing preferred env
    assert selector.supported_codecs == ["PCMA/8000"]


async def _wait_for_audio_parts(server: FakeTapoServer, *, count: int) -> None:
    for _ in range(50):
        if len(server.audio_parts) >= count:
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"fake Tapo server did not receive {count} audio parts")


async def _wait_for_closed_connection(server: FakeTapoServer) -> None:
    for _ in range(50):
        if server.closed_connections:
            return
        await asyncio.sleep(0.01)
    raise AssertionError("fake Tapo server did not observe a closed connection")
