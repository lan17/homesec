"""Built-in Tapo local talk backend registration and detector."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel

from homesec.models.talk import (
    TalkCapabilityProbeResult,
    TalkCapabilityState,
    TalkRefusalReason,
    TalkSessionOpenRequest,
)
from homesec.talk.backends import (
    TalkBackendAdapter,
    TalkBackendContext,
    TalkBackendDetection,
    TalkBackendOpenError,
    TalkBackendRegistration,
    TalkBackendSession,
)
from homesec.talk.tapo.config import (
    TapoCredential,
    TapoLocalTalkConfig,
    resolve_tapo_credential,
    resolve_tapo_host,
)

TAPO_LOCAL_BACKEND = "tapo_local"
TAPO_LOCAL_CODEC = "PCMA/8000"
_TAPO_PROTOCOL_NOT_IMPLEMENTED = "Tapo local protocol client is not implemented yet"


@dataclass(slots=True)
class TapoLocalTalkBackend:
    """Phase 10 Tapo backend skeleton.

    The network protocol client and audio session are implemented in later Phase
    10 tickets. This skeleton lets config, registry, and selector behavior land
    without changing API/runtime/UI contracts.
    """

    config: TapoLocalTalkConfig
    context: TalkBackendContext
    host: str
    credential: TapoCredential
    name: str = TAPO_LOCAL_BACKEND

    @property
    def supported_codecs(self) -> list[str]:
        """Return camera-side codecs planned for the Tapo local backend."""
        return [TAPO_LOCAL_CODEC]

    async def probe(self) -> TalkCapabilityProbeResult:
        """Report that protocol probing is not implemented until the next ticket."""
        return _protocol_not_implemented_result()

    async def open_session(self, request: TalkSessionOpenRequest) -> TalkBackendSession:
        """Refuse session open until the Tapo protocol client exists."""
        _ = request
        raise TalkBackendOpenError(
            _TAPO_PROTOCOL_NOT_IMPLEMENTED,
            reason=TalkRefusalReason.RUNTIME_UNAVAILABLE,
        )


def detect_tapo_local(context: TalkBackendContext) -> TalkBackendDetection:
    """Return safe, local-only Tapo applicability metadata for auto selection."""
    if context.camera_talk.backend == TAPO_LOCAL_BACKEND:
        return TalkBackendDetection(
            backend=TAPO_LOCAL_BACKEND,
            confidence="explicit",
            reason="Explicit Tapo local backend configured",
            safe_to_probe=True,
            requires_credentials=True,
        )

    if TAPO_LOCAL_BACKEND in context.camera_talk.backends:
        return TalkBackendDetection(
            backend=TAPO_LOCAL_BACKEND,
            confidence="explicit",
            reason="Tapo local backend config present",
            safe_to_probe=True,
            requires_credentials=True,
        )

    manufacturer = (context.fingerprint.manufacturer or "").lower()
    model = (context.fingerprint.model or "").lower()
    if "tp-link" in manufacturer or "tapo" in model:
        return TalkBackendDetection(
            backend=TAPO_LOCAL_BACKEND,
            confidence="high",
            reason="TP-Link/Tapo camera fingerprint",
            safe_to_probe=True,
            requires_credentials=True,
        )

    return TalkBackendDetection(
        backend=TAPO_LOCAL_BACKEND,
        confidence="not_applicable",
        reason="No Tapo local config or fingerprint",
        safe_to_probe=False,
        requires_credentials=True,
    )


def tapo_local_talk_backend_registration() -> TalkBackendRegistration:
    """Return the built-in Tapo local talk backend registration."""
    return TalkBackendRegistration(
        name=TAPO_LOCAL_BACKEND,
        config_model=TapoLocalTalkConfig,
        factory=build_tapo_local_backend,
        detector=detect_tapo_local,
        priority=50,
        standards_based=False,
    )


def build_tapo_local_backend(
    config: BaseModel,
    context: TalkBackendContext,
) -> TalkBackendAdapter:
    """Build a Tapo local backend from validated backend config."""
    if not isinstance(config, TapoLocalTalkConfig):
        raise TypeError(f"Expected TapoLocalTalkConfig, got {type(config).__name__}")
    host = resolve_tapo_host(config, context)
    credential = resolve_tapo_credential(config, context)
    return TapoLocalTalkBackend(
        config=config,
        context=context,
        host=host,
        credential=credential,
    )


def _protocol_not_implemented_result() -> TalkCapabilityProbeResult:
    return TalkCapabilityProbeResult(
        capability=TalkCapabilityState.ERROR,
        refusal_reason=TalkRefusalReason.RUNTIME_UNAVAILABLE,
        message=_TAPO_PROTOCOL_NOT_IMPLEMENTED,
        offered_codecs=[TAPO_LOCAL_CODEC],
        selected_codec=TAPO_LOCAL_CODEC,
    )
