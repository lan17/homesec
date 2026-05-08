"""Talk backend selection and adapter dispatch."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass, field

from pydantic import ValidationError

from homesec.models.talk import (
    TalkCapabilityProbeResult,
    TalkCapabilityState,
    TalkRefusalReason,
    TalkSessionOpenRequest,
)
from homesec.talk.backend_ids import (
    normalize_talk_backend_id,
    sanitize_talk_backend_reason,
    validate_talk_backend_id,
)
from homesec.talk.backends import (
    TalkBackendAdapter,
    TalkBackendConfidence,
    TalkBackendConfigError,
    TalkBackendContext,
    TalkBackendOpenError,
    TalkBackendRegistration,
    TalkBackendRegistry,
    TalkBackendSession,
    TalkBackendSessionProbeAdapter,
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
    codecs: list[str] = field(default_factory=list)

    @property
    def supported_codecs(self) -> list[str]:
        return list(self.codecs)


@dataclass(frozen=True, slots=True)
class _AutoDetectionResult:
    registrations: list[TalkBackendRegistration]
    failures: list[_SelectionError]


_AUTO_DETECTION_CONFIDENCE_RANK: dict[TalkBackendConfidence, int] = {
    "explicit": 0,
    "high": 1,
    "medium": 2,
    "low": 3,
    "not_applicable": 4,
}

logger = logging.getLogger(__name__)


class TalkBackendSelector:
    """Select and dispatch to the configured camera talk backend.

    Auto selection is sticky for the lifetime of the selector once a backend is
    selected. Rebuilding the source creates a new selector and re-runs discovery.
    """

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
        if self._context.camera_talk.backend == "auto" and not isinstance(
            self._selection, _SelectedBackend
        ):
            return _unique_codecs(
                codec
                for registration in self._standards_registrations()
                for codec in self._build_registered_backend(
                    registration.name,
                    reason=self._auto_candidate_reason(registration),
                ).supported_codecs
            )
        return self._select().supported_codecs

    @property
    def selected_supported_codecs(self) -> list[str] | None:
        """Return selected backend codec hints, or None before auto selection has landed."""
        if self._selection is None:
            return None
        if isinstance(self._selection, _SelectedBackend):
            return self._selection.supported_codecs
        return self._selection.supported_codecs

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
        return await self._probe(for_session_open=False)

    async def probe_for_session_open(self) -> TalkCapabilityProbeResult:
        """Probe the selected backend while preserving state for immediate open."""
        return await self._probe(for_session_open=True)

    async def _probe(self, *, for_session_open: bool) -> TalkCapabilityProbeResult:
        if self._context.camera_talk.backend == "auto":
            return await self._probe_auto(for_session_open=for_session_open)
        selection = self._select()
        if isinstance(selection, _SelectionError):
            return selection.result
        return await _probe_backend_adapter(selection.adapter, for_session_open=for_session_open)

    async def open_session(self, request: TalkSessionOpenRequest) -> TalkBackendSession:
        """Open a talk session through the selected backend."""
        if self._context.camera_talk.backend == "auto" and not isinstance(
            self._selection,
            _SelectedBackend,
        ):
            probe = await self._probe_auto(for_session_open=True)
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

    async def _probe_auto(self, *, for_session_open: bool) -> TalkCapabilityProbeResult:
        if isinstance(self._selection, _SelectedBackend):
            return await _probe_backend_adapter(
                self._selection.adapter,
                for_session_open=for_session_open,
            )
        standards = self._standards_registrations()
        if not standards:
            self._selection = _SelectionError(
                _runtime_selection_error_result("No standards-based talk backends are registered"),
                backend=None,
                backend_reason="No standards-based talk backends are registered",
            )
            return self._selection.result

        best_failure: _SelectionError | None = None
        seen = set[str]()
        for registration in standards:
            seen.add(registration.name)
            selection = self._build_registered_backend(
                registration.name,
                reason=self._auto_candidate_reason(registration),
            )
            if isinstance(selection, _SelectionError):
                best_failure = _better_auto_failure(best_failure, selection)
                continue

            result = await _probe_backend_adapter(
                selection.adapter,
                for_session_open=for_session_open,
            )
            if result.capability == TalkCapabilityState.SUPPORTED:
                self._selection = selection
                return result
            best_failure = _better_auto_failure(
                best_failure,
                _SelectionError(
                    result,
                    backend=selection.backend,
                    backend_reason=selection.backend_reason,
                    codecs=selection.supported_codecs,
                ),
            )

        detected = self._detected_auto_registrations()
        for failure in detected.failures:
            best_failure = _better_auto_failure(best_failure, failure)

        for registration in detected.registrations:
            if registration.name in seen:
                continue
            selection = self._build_registered_backend(
                registration.name,
                reason=self._auto_candidate_reason(registration),
            )
            if isinstance(selection, _SelectionError):
                best_failure = _better_auto_failure(best_failure, selection)
                continue

            result = await _probe_backend_adapter(
                selection.adapter,
                for_session_open=for_session_open,
            )
            if result.capability == TalkCapabilityState.SUPPORTED:
                self._selection = selection
                return result
            best_failure = _better_auto_failure(
                best_failure,
                _SelectionError(
                    result,
                    backend=selection.backend,
                    backend_reason=selection.backend_reason,
                    codecs=selection.supported_codecs,
                ),
            )

        if best_failure is None:
            return _SelectionError(
                _runtime_selection_error_result("No standards-based talk backends are registered"),
                backend=None,
                backend_reason="No standards-based talk backends are registered",
            ).result
        self._selection = best_failure
        return best_failure.result

    def _standards_registrations(self) -> list[TalkBackendRegistration]:
        return [
            registration
            for registration in self._registry.standards_first()
            if registration.standards_based
        ]

    def _detected_auto_registrations(self) -> _AutoDetectionResult:
        detected: list[tuple[int, bool, int, str, TalkBackendRegistration]] = []
        failures: list[_SelectionError] = []
        for registration in self._registry.all():
            if registration.detector is None:
                continue
            try:
                detection = registration.detector(self._context)
            except Exception:
                logger.debug(
                    "Talk backend detector failed",
                    extra={
                        "camera_name": self._context.camera_name,
                        "talk_backend": registration.name,
                    },
                    exc_info=True,
                )
                message = f"Talk backend '{registration.name}' detector failed"
                failures.append(
                    _SelectionError(
                        _runtime_selection_error_result(message),
                        backend=registration.name,
                        backend_reason=message,
                    )
                )
                continue
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
            return _AutoDetectionResult(registrations=[], failures=failures)
        detected.sort(key=lambda item: item[:4])
        return _AutoDetectionResult(
            registrations=[item[4] for item in detected],
            failures=failures,
        )

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
        except TalkBackendConfigError as exc:
            public_message = _public_structured_config_error_message(registration.name, exc)
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
        except (ValueError, ValidationError) as exc:
            public_message = _public_config_error_message(registration.name)
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


async def _probe_backend_adapter(
    adapter: TalkBackendAdapter,
    *,
    for_session_open: bool,
) -> TalkCapabilityProbeResult:
    if for_session_open and isinstance(adapter, TalkBackendSessionProbeAdapter):
        return await adapter.probe_for_session_open()
    return await adapter.probe()


def _config_error_result(message: str) -> TalkCapabilityProbeResult:
    return TalkCapabilityProbeResult(
        capability=TalkCapabilityState.CONFIG_ERROR,
        refusal_reason=TalkRefusalReason.TALK_CONFIG_ERROR,
        message=message,
    )


def _public_structured_config_error_message(
    backend_name: str,
    exc: TalkBackendConfigError,
) -> str:
    base = f"Talk backend '{backend_name}' config is invalid"
    public_message = sanitize_talk_backend_reason(exc.public_message)
    return public_message if public_message is not None else base


def _public_config_error_message(backend_name: str) -> str:
    return f"Talk backend '{backend_name}' config is invalid"


def _backend_display_name(backend_name: str) -> str:
    if backend_name == "onvif_rtsp_backchannel":
        return "ONVIF RTSP backchannel"
    return f"talk backend '{backend_name}'"


def _auto_failure_rank(result: TalkCapabilityProbeResult) -> int:
    match result.capability:
        case TalkCapabilityState.CONFIG_ERROR:
            return 0
        case TalkCapabilityState.ERROR:
            return 1 if result.refusal_reason == TalkRefusalReason.TALK_AUTH_FAILED else 2
        case TalkCapabilityState.UNSUPPORTED_CODEC:
            return 3
        case TalkCapabilityState.UNSUPPORTED:
            return 4
        case _:
            return 5


def _better_auto_failure(
    current: _SelectionError | None,
    candidate: _SelectionError,
) -> _SelectionError:
    if current is None:
        return candidate
    if _auto_failure_rank(candidate.result) < _auto_failure_rank(current.result):
        return candidate
    return current


def _unique_codecs(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if not isinstance(value, str) or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result
