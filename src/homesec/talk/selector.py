"""Talk backend selection and adapter dispatch."""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass

from pydantic import ValidationError

from homesec.models.talk import (
    TalkCapabilityProbeResult,
    TalkCapabilityState,
    TalkRefusalReason,
    TalkSessionOpenRequest,
)
from homesec.talk.backend_ids import normalize_talk_backend_id, validate_talk_backend_id
from homesec.talk.backends import (
    TalkBackendAdapter,
    TalkBackendConfidence,
    TalkBackendContext,
    TalkBackendOpenError,
    TalkBackendRegistration,
    TalkBackendRegistry,
    TalkBackendSession,
    backend_config_for,
    model_validate_backend_config,
)


@dataclass(frozen=True, slots=True)
class _SelectedBackend:
    adapter: TalkBackendAdapter
    backend: str
    backend_reason: str

    @property
    def supported_codecs(self) -> list[str]:
        return self.adapter.supported_codecs


@dataclass(frozen=True, slots=True)
class _SelectionError:
    result: TalkCapabilityProbeResult
    backend: str | None
    backend_reason: str | None

    @property
    def supported_codecs(self) -> list[str]:
        return []


_AUTO_DETECTION_CONFIDENCE_RANK: dict[TalkBackendConfidence, int] = {
    "explicit": 0,
    "high": 1,
    "medium": 2,
    "low": 3,
    "not_applicable": 4,
}

logger = logging.getLogger(__name__)
_SAFE_ENV_VAR_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")
_MISSING_ENV_PREFIXES = (
    "RTSP URL environment variable is not set: ",
    "RTSP credential environment variable is not set: ",
)


class TalkBackendSelector:
    """Select and dispatch to the configured camera talk backend."""

    def __init__(
        self,
        *,
        registry: TalkBackendRegistry,
        context: TalkBackendContext,
    ) -> None:
        self._registry = registry
        self._context = context
        self._selection: _SelectedBackend | _SelectionError | None = None

    @property
    def supported_codecs(self) -> list[str]:
        """Return supported codecs for the selected backend, if one can be built."""
        if self._context.camera_talk.backend == "auto" and self._selection is None:
            return _unique_codecs(
                codec
                for registration in self._auto_candidate_registrations()
                for codec in self._build_registered_backend(
                    registration.name,
                    reason=self._auto_candidate_reason(registration),
                ).supported_codecs
            )
        return self._select().supported_codecs

    @property
    def backend(self) -> str | None:
        """Return the selected or explicitly requested backend name when known."""
        if self._context.camera_talk.backend == "auto" and self._selection is None:
            return None
        return self._select().backend

    @property
    def backend_reason(self) -> str | None:
        """Return a safe human-readable backend selection diagnostic."""
        if self._context.camera_talk.backend == "auto" and self._selection is None:
            return None
        return self._select().backend_reason

    async def probe(self) -> TalkCapabilityProbeResult:
        """Probe the selected backend or return a selection/config error."""
        if self._context.camera_talk.backend == "auto":
            return await self._probe_auto()
        selection = self._select()
        if isinstance(selection, _SelectionError):
            return selection.result
        return await selection.adapter.probe()

    async def open_session(self, request: TalkSessionOpenRequest) -> TalkBackendSession:
        """Open a talk session through the selected backend."""
        if self._context.camera_talk.backend == "auto" and not isinstance(
            self._selection,
            _SelectedBackend,
        ):
            probe = await self._probe_auto()
            if probe.capability != TalkCapabilityState.SUPPORTED:
                raise TalkBackendOpenError(
                    probe.message or "Talk backend is not available",
                    reason=probe.refusal_reason or TalkRefusalReason.RUNTIME_UNAVAILABLE,
                )
        selection = self._select()
        if isinstance(selection, _SelectionError):
            raise TalkBackendOpenError(
                selection.result.message or "Talk backend is not available",
                reason=selection.result.refusal_reason or TalkRefusalReason.RUNTIME_UNAVAILABLE,
            )
        return await selection.adapter.open_session(request)

    def _select(self) -> _SelectedBackend | _SelectionError:
        if self._selection is not None:
            return self._selection

        camera_talk = self._context.camera_talk
        if camera_talk.backend == "auto":
            return _SelectionError(
                _runtime_selection_error_result("Talk backend auto selection has not been probed"),
                backend=None,
                backend_reason=None,
            )

        registration = self._registry.get(camera_talk.backend)
        if registration is None:
            message = f"Talk backend '{camera_talk.backend}' is not registered in this runtime"
            self._selection = _SelectionError(
                _config_error_result(message),
                backend=camera_talk.backend,
                backend_reason=f"Talk backend '{camera_talk.backend}' is not registered",
            )
            return self._selection

        self._selection = self._build_registered_backend(
            registration.name,
            reason=f"Explicit backend {registration.name} configured",
        )
        return self._selection

    async def _probe_auto(self) -> TalkCapabilityProbeResult:
        if isinstance(self._selection, _SelectedBackend):
            return await self._selection.adapter.probe()
        candidates = self._auto_candidate_registrations()
        if not candidates:
            self._selection = _SelectionError(
                _runtime_selection_error_result("No standards-based talk backends are registered"),
                backend=None,
                backend_reason="No standards-based talk backends are registered",
            )
            return self._selection.result

        first_result: TalkCapabilityProbeResult | None = None
        first_backend: str | None = None
        first_reason: str | None = None
        for registration in candidates:
            selection = self._build_registered_backend(
                registration.name,
                reason=self._auto_candidate_reason(registration),
            )
            if isinstance(selection, _SelectionError):
                self._selection = selection
                return selection.result

            result = await selection.adapter.probe()
            if result.capability == TalkCapabilityState.SUPPORTED:
                self._selection = selection
                return result
            if first_result is None:
                first_result = result
                first_backend = selection.backend
                first_reason = selection.backend_reason

        if first_result is None:
            return _SelectionError(
                _runtime_selection_error_result("No standards-based talk backends are registered"),
                backend=None,
                backend_reason="No standards-based talk backends are registered",
            ).result
        self._selection = _SelectionError(
            first_result,
            backend=first_backend,
            backend_reason=first_reason,
        )
        return first_result

    def _auto_candidate_registrations(self) -> list[TalkBackendRegistration]:
        standards = [
            registration
            for registration in self._registry.standards_first()
            if registration.standards_based
        ]
        seen = {registration.name for registration in standards}
        detected = [
            registration
            for registration in self._detected_auto_registrations()
            if registration.name not in seen
        ]
        return standards + detected

    def _detected_auto_registrations(self) -> list[TalkBackendRegistration]:
        detected: list[tuple[int, bool, int, str, TalkBackendRegistration]] = []
        for registration in self._registry.all():
            if registration.detector is None:
                continue
            detection = registration.detector(self._context)
            detected_backend = normalize_talk_backend_id(detection.backend)
            try:
                validate_talk_backend_id(detected_backend)
            except ValueError:
                continue
            if (
                detected_backend != registration.name
                or detection.confidence == "not_applicable"
                or not detection.safe_to_probe
            ):
                continue
            detected.append(
                (
                    _AUTO_DETECTION_CONFIDENCE_RANK[detection.confidence],
                    not registration.standards_based,
                    registration.priority,
                    registration.name,
                    registration,
                )
            )
        if not detected:
            return []
        detected.sort(key=lambda item: item[:4])
        return [item[4] for item in detected]

    def _auto_candidate_reason(self, registration: TalkBackendRegistration) -> str:
        if registration.standards_based:
            return (
                f"Selected {_backend_display_name(registration.name)} "
                "by standards-first auto probing"
            )
        return f"Selected talk backend '{registration.name}' by safe camera detector"

    def _build_registered_backend(
        self,
        backend_name: str,
        *,
        reason: str,
    ) -> _SelectedBackend | _SelectionError:
        registration = self._registry.get(backend_name)
        if registration is None:
            message = f"Talk backend '{backend_name}' is not registered in this runtime"
            return _SelectionError(
                _config_error_result(message),
                backend=backend_name,
                backend_reason=f"Talk backend '{backend_name}' is not registered",
            )
        raw_config = backend_config_for(self._context.camera_talk, registration.name)
        try:
            config = model_validate_backend_config(registration, raw_config, context=self._context)
            return _SelectedBackend(
                registration.factory(config, self._context),
                backend=registration.name,
                backend_reason=reason,
            )
        except (ValueError, ValidationError) as exc:
            public_message = _public_config_error_message(registration.name, exc)
            logger.debug(
                "Talk backend config validation failed",
                extra={
                    "camera_name": self._context.camera_name,
                    "talk_backend": registration.name,
                    "error_type": type(exc).__name__,
                },
            )
            return _SelectionError(
                _config_error_result(public_message),
                backend=registration.name,
                backend_reason=public_message,
            )


def _runtime_selection_error_result(message: str) -> TalkCapabilityProbeResult:
    return TalkCapabilityProbeResult(
        capability=TalkCapabilityState.ERROR,
        refusal_reason=TalkRefusalReason.RUNTIME_UNAVAILABLE,
        message=message,
    )


def _config_error_result(message: str) -> TalkCapabilityProbeResult:
    return TalkCapabilityProbeResult(
        capability=TalkCapabilityState.CONFIG_ERROR,
        refusal_reason=TalkRefusalReason.TALK_CONFIG_ERROR,
        message=message,
    )


def _public_config_error_message(backend_name: str, exc: Exception) -> str:
    base = f"Talk backend '{backend_name}' config is invalid"
    if not isinstance(exc, ValueError):
        return base
    detail = str(exc).strip()
    if not detail:
        return base
    for prefix in _MISSING_ENV_PREFIXES:
        if detail.startswith(prefix):
            env_name = detail.removeprefix(prefix).strip()
            if _SAFE_ENV_VAR_NAME_PATTERN.fullmatch(env_name):
                return f"{base}: {prefix}{env_name}"
            return base
    return base


def _backend_display_name(backend_name: str) -> str:
    if backend_name == "onvif_rtsp_backchannel":
        return "ONVIF RTSP backchannel"
    return f"talk backend '{backend_name}'"


def _unique_codecs(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if not isinstance(value, str) or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result
