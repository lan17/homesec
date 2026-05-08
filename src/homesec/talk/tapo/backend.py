"""Built-in Tapo local talk backend registration and adapter."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from pydantic import BaseModel

from homesec.models.talk import (
    TalkCapabilityProbeResult,
    TalkCapabilityState,
    TalkRefusalReason,
    TalkSessionOpenRequest,
)
from homesec.talk.backends import (
    TalkBackendAdapter,
    TalkBackendConfigError,
    TalkBackendContext,
    TalkBackendDetection,
    TalkBackendOpenError,
    TalkBackendRegistration,
    TalkBackendSession,
)
from homesec.talk.tapo.client import (
    TapoAuthError,
    TapoClientError,
    TapoLocalClient,
    TapoUnsupportedEndpointError,
    open_tapo_local_client,
)
from homesec.talk.tapo.config import (
    TapoLocalTalkConfig,
    resolve_tapo_credential,
    resolve_tapo_host,
)
from homesec.talk.tapo.session import TAPO_LOCAL_CODEC, TapoLocalTalkSession

TAPO_LOCAL_BACKEND = "tapo_local"
_TAPO_UNSUPPORTED_ENDPOINT = "Tapo local endpoint not detected"
_TAPO_PROTOCOL_FAILED = "Tapo local talk protocol failed"


@dataclass(slots=True)
class TapoLocalTalkBackend:
    """Talk backend adapter for the Tapo local speaker stream endpoint."""

    config: TapoLocalTalkConfig
    context: TalkBackendContext = field(repr=False)
    host: str
    name: str = TAPO_LOCAL_BACKEND
    _prepared_client: TapoLocalClient | None = field(default=None, init=False, repr=False)

    @property
    def supported_codecs(self) -> list[str]:
        """Return camera-side codecs emitted by the Tapo local backend."""
        return [TAPO_LOCAL_CODEC]

    async def probe(self) -> TalkCapabilityProbeResult:
        """Probe the Tapo local talk endpoint without retaining the connection."""
        return await self._probe(keep_for_open=False)

    async def probe_for_session_open(self) -> TalkCapabilityProbeResult:
        """Probe Tapo capability while retaining setup state for immediate open."""
        return await self._probe(keep_for_open=True)

    async def clear_prepared_probe(self) -> None:
        """Clear a prepared Tapo stream client if the reservation is abandoned."""
        client = self._prepared_client
        self._prepared_client = None
        if client is not None:
            await client.close()

    async def open_session(self, request: TalkSessionOpenRequest) -> TalkBackendSession:
        """Open a Tapo local talk session."""
        client = self._prepared_client
        self._prepared_client = None
        try:
            if client is None:
                client = await self._connect_and_setup_talk()
            return await TapoLocalTalkSession.create(
                session_id=request.session_id,
                camera_name=self.context.camera_name,
                client=client,
                input_sample_rate=request.input.sample_rate,
            )
        except asyncio.CancelledError:
            await _close_client_if_present(client)
            raise
        except TalkBackendConfigError as exc:
            await _close_client_if_present(client)
            raise TalkBackendOpenError(
                exc.public_message,
                reason=TalkRefusalReason.TALK_CONFIG_ERROR,
            ) from exc
        except TapoAuthError as exc:
            await _close_client_if_present(client)
            raise TalkBackendOpenError(
                "Tapo local authentication failed",
                reason=TalkRefusalReason.TALK_AUTH_FAILED,
            ) from exc
        except TapoUnsupportedEndpointError as exc:
            await _close_client_if_present(client)
            raise TalkBackendOpenError(
                _TAPO_UNSUPPORTED_ENDPOINT,
                reason=TalkRefusalReason.UNSUPPORTED_CAMERA,
            ) from exc
        except (TapoClientError, asyncio.TimeoutError, TimeoutError, EOFError, OSError) as exc:
            await _close_client_if_present(client)
            raise TalkBackendOpenError(
                _TAPO_PROTOCOL_FAILED,
                reason=TalkRefusalReason.CAMERA_BACKCHANNEL_FAILED,
            ) from exc

    async def _probe(self, *, keep_for_open: bool) -> TalkCapabilityProbeResult:
        await self.clear_prepared_probe()
        try:
            client = await self._connect_and_setup_talk()
        except TalkBackendConfigError as exc:
            return _config_error_result(exc.public_message)
        except TapoAuthError:
            return _auth_error_result()
        except TapoUnsupportedEndpointError:
            return _unsupported_endpoint_result()
        except (TapoClientError, asyncio.TimeoutError, TimeoutError, EOFError, OSError):
            return _protocol_error_result()

        if keep_for_open:
            self._prepared_client = client
        else:
            await client.close()
        return _supported_result()

    async def _connect_and_setup_talk(self) -> TapoLocalClient:
        return await open_tapo_local_client(self.config, self.context)


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
    resolve_tapo_credential(config, context)
    return TapoLocalTalkBackend(
        config=config,
        context=context,
        host=host,
    )


def _supported_result() -> TalkCapabilityProbeResult:
    return TalkCapabilityProbeResult(
        capability=TalkCapabilityState.SUPPORTED,
        offered_codecs=[TAPO_LOCAL_CODEC],
        selected_codec=TAPO_LOCAL_CODEC,
    )


def _config_error_result(message: str) -> TalkCapabilityProbeResult:
    return TalkCapabilityProbeResult(
        capability=TalkCapabilityState.CONFIG_ERROR,
        refusal_reason=TalkRefusalReason.TALK_CONFIG_ERROR,
        message=message,
        offered_codecs=[TAPO_LOCAL_CODEC],
        selected_codec=TAPO_LOCAL_CODEC,
    )


def _auth_error_result() -> TalkCapabilityProbeResult:
    return TalkCapabilityProbeResult(
        capability=TalkCapabilityState.ERROR,
        refusal_reason=TalkRefusalReason.TALK_AUTH_FAILED,
        message="Tapo local authentication failed",
        offered_codecs=[TAPO_LOCAL_CODEC],
        selected_codec=TAPO_LOCAL_CODEC,
    )


def _unsupported_endpoint_result() -> TalkCapabilityProbeResult:
    return TalkCapabilityProbeResult(
        capability=TalkCapabilityState.UNSUPPORTED,
        refusal_reason=TalkRefusalReason.UNSUPPORTED_CAMERA,
        message=_TAPO_UNSUPPORTED_ENDPOINT,
        offered_codecs=[TAPO_LOCAL_CODEC],
        selected_codec=TAPO_LOCAL_CODEC,
    )


def _protocol_error_result() -> TalkCapabilityProbeResult:
    return TalkCapabilityProbeResult(
        capability=TalkCapabilityState.ERROR,
        refusal_reason=TalkRefusalReason.CAMERA_BACKCHANNEL_FAILED,
        message=_TAPO_PROTOCOL_FAILED,
        offered_codecs=[TAPO_LOCAL_CODEC],
        selected_codec=TAPO_LOCAL_CODEC,
    )


async def _close_client_if_present(client: TapoLocalClient | None) -> None:
    if client is not None:
        await client.close()
