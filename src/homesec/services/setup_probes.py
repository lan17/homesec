"""Setup-only probe registry for onboarding test-connection flows."""

from __future__ import annotations

import importlib
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from functools import partial
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from homesec.models.config import PreviewConfig
from homesec.models.setup import PreflightCheckResponse, TestConnectionResponse

SetupProbeTarget = Literal["camera", "storage", "notifier", "analyzer"]
SetupProbeFn = Callable[..., Awaitable[TestConnectionResponse]]
_DEFAULT_SETUP_PROBE_TIMEOUT_S = 5.0


class SetupProbeContext(BaseModel):
    """Existing shared configuration supplied privately to source-owned probes."""

    model_config = ConfigDict(extra="forbid")

    preview: PreviewConfig = Field(default_factory=PreviewConfig)


SetupReadinessFn = Callable[..., Awaitable[PreflightCheckResponse]]


@dataclass(frozen=True)
class _RegisteredSetupProbe:
    target: SetupProbeTarget
    backend: str
    probe: SetupProbeFn
    timeout_s: float | None
    contextual: bool
    readiness: SetupReadinessFn | None


class SetupProbeRegistry:
    """Registry for setup-only probe handlers keyed by target/backend."""

    def __init__(self) -> None:
        self._probes: dict[tuple[SetupProbeTarget, str], _RegisteredSetupProbe] = {}

    def register(
        self,
        target: SetupProbeTarget,
        backend: str,
        *,
        timeout_s: float | None = _DEFAULT_SETUP_PROBE_TIMEOUT_S,
        contextual: bool = False,
        readiness: SetupReadinessFn | None = None,
    ) -> Callable[[SetupProbeFn], SetupProbeFn]:
        """Register a setup probe for a target/backend pair."""

        normalized_backend = backend.strip().lower()
        key = (target, normalized_backend)
        if timeout_s is not None and timeout_s <= 0:
            raise ValueError("Setup probe timeout must be positive when provided.")

        def decorator(probe: SetupProbeFn) -> SetupProbeFn:
            if key in self._probes:
                raise ValueError(
                    f"Setup probe for {target} backend {normalized_backend!r} is already registered."
                )
            self._probes[key] = _RegisteredSetupProbe(
                target=target,
                backend=normalized_backend,
                probe=probe,
                timeout_s=timeout_s,
                contextual=contextual,
                readiness=readiness,
            )
            return probe

        return decorator

    def get(
        self,
        target: SetupProbeTarget,
        backend: str,
        *,
        context: SetupProbeContext | None = None,
    ) -> SetupProbeFn | None:
        """Return a registered probe for the target/backend pair, if present."""
        entry = self._probes.get((target, backend.strip().lower()))
        if entry is None:
            return None
        if entry.contextual and context is not None:
            return partial(entry.probe, context=context)
        return entry.probe

    def get_readiness(self, target: SetupProbeTarget, backend: str) -> SetupReadinessFn | None:
        """Return camera-free readiness registered beside its source probe."""
        entry = self._probes.get((target, backend.strip().lower()))
        return entry.readiness if entry is not None else None

    def uses_context(self, target: SetupProbeTarget, backend: str) -> bool:
        entry = self._probes.get((target, backend.strip().lower()))
        return entry.contextual if entry is not None else False

    def get_backends(self, target: SetupProbeTarget) -> list[str]:
        """Return known special-case backends for a target."""
        return sorted({entry.backend for entry in self._probes.values() if entry.target == target})

    def get_timeout(self, target: SetupProbeTarget, backend: str) -> float | None:
        """Return the registered timeout budget for the target/backend pair, if present."""
        entry = self._probes.get((target, backend.strip().lower()))
        if entry is None:
            return None
        return entry.timeout_s


_SETUP_PROBE_REGISTRY = SetupProbeRegistry()
_BUILTIN_SETUP_PROBE_MODULES = (
    "homesec.sources.rtsp.setup_probe",
    "homesec.sources.ftp_setup_probe",
    "homesec.sources.local_folder_setup_probe",
    "homesec.onvif.setup_probe",
    "homesec.plugins.storage.local_setup_probe",
)
_builtin_setup_probes_loaded = False


def load_builtin_setup_probes() -> None:
    """Import built-in setup-probe modules once so decorators can register handlers."""
    global _builtin_setup_probes_loaded
    if _builtin_setup_probes_loaded:
        return
    for module_name in _BUILTIN_SETUP_PROBE_MODULES:
        importlib.import_module(module_name)
    _builtin_setup_probes_loaded = True


def setup_probe(
    target: SetupProbeTarget,
    backend: str,
    *,
    timeout_s: float | None = _DEFAULT_SETUP_PROBE_TIMEOUT_S,
    contextual: bool = False,
    readiness: SetupReadinessFn | None = None,
) -> Callable[[SetupProbeFn], SetupProbeFn]:
    """Decorator for registering a setup-only probe handler."""
    return _SETUP_PROBE_REGISTRY.register(
        target, backend, timeout_s=timeout_s, contextual=contextual, readiness=readiness
    )


def get_setup_probe(
    target: SetupProbeTarget,
    backend: str,
    *,
    context: SetupProbeContext | None = None,
) -> SetupProbeFn | None:
    """Look up a setup-only probe handler."""
    load_builtin_setup_probes()
    return _SETUP_PROBE_REGISTRY.get(target, backend, context=context)


def get_setup_readiness(target: SetupProbeTarget, backend: str) -> SetupReadinessFn | None:
    """Look up a camera-free dependency check in the existing probe registry."""
    load_builtin_setup_probes()
    return _SETUP_PROBE_REGISTRY.get_readiness(target, backend)


def setup_probe_uses_context(target: SetupProbeTarget, backend: str) -> bool:
    """Only opted-in callbacks require shared configuration to be resolved."""
    load_builtin_setup_probes()
    return _SETUP_PROBE_REGISTRY.uses_context(target, backend)


def get_setup_probe_timeout(target: SetupProbeTarget, backend: str) -> float | None:
    """Look up the registered timeout budget for a setup-only probe handler."""
    load_builtin_setup_probes()
    return _SETUP_PROBE_REGISTRY.get_timeout(target, backend)


def get_setup_probe_backends(target: SetupProbeTarget) -> list[str]:
    """List special-case backends registered for a setup target."""
    load_builtin_setup_probes()
    return _SETUP_PROBE_REGISTRY.get_backends(target)
