"""Talk backend selection and adapter dispatch."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import ValidationError

from homesec.models.talk import (
    TalkCapabilityProbeResult,
    TalkCapabilityState,
    TalkRefusalReason,
    TalkSessionOpenRequest,
)
from homesec.talk.backends import (
    TalkBackendAdapter,
    TalkBackendContext,
    TalkBackendOpenError,
    TalkBackendRegistry,
    TalkBackendSession,
    backend_config_for,
    model_validate_backend_config,
)


@dataclass(frozen=True, slots=True)
class _SelectedBackend:
    adapter: TalkBackendAdapter

    @property
    def supported_codecs(self) -> list[str]:
        return self.adapter.supported_codecs


@dataclass(frozen=True, slots=True)
class _SelectionError:
    result: TalkCapabilityProbeResult

    @property
    def supported_codecs(self) -> list[str]:
        return []


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
        return self._select().supported_codecs

    async def probe(self) -> TalkCapabilityProbeResult:
        """Probe the selected backend or return a selection/config error."""
        selection = self._select()
        if isinstance(selection, _SelectionError):
            return selection.result
        return await selection.adapter.probe()

    async def open_session(self, request: TalkSessionOpenRequest) -> TalkBackendSession:
        """Open a talk session through the selected backend."""
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
            self._selection = self._select_auto()
            return self._selection

        registration = self._registry.get(camera_talk.backend)
        if registration is None:
            self._selection = _SelectionError(
                _selection_error_result(
                    f"Talk backend '{camera_talk.backend}' is not registered in this runtime"
                )
            )
            return self._selection

        self._selection = self._build_registered_backend(registration.name)
        return self._selection

    def _select_auto(self) -> _SelectedBackend | _SelectionError:
        registrations = self._registry.standards_first()
        if not registrations:
            return _SelectionError(_selection_error_result("No talk backends are registered"))
        return self._build_registered_backend(registrations[0].name)

    def _build_registered_backend(self, backend_name: str) -> _SelectedBackend | _SelectionError:
        registration = self._registry.get(backend_name)
        if registration is None:
            return _SelectionError(
                _selection_error_result(
                    f"Talk backend '{backend_name}' is not registered in this runtime"
                )
            )
        raw_config = backend_config_for(self._context.camera_talk, registration.name)
        try:
            config = model_validate_backend_config(registration, raw_config, context=self._context)
            return _SelectedBackend(registration.factory(config, self._context))
        except (ValueError, ValidationError) as exc:
            return _SelectionError(_selection_error_result(str(exc)))


def _selection_error_result(message: str) -> TalkCapabilityProbeResult:
    return TalkCapabilityProbeResult(
        capability=TalkCapabilityState.ERROR,
        refusal_reason=TalkRefusalReason.RUNTIME_UNAVAILABLE,
        message=message,
    )
