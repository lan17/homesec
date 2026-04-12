"""Setup-only probe registry for onboarding test-connection flows."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Literal

from homesec.models.setup import TestConnectionResponse

SetupProbeTarget = Literal["camera", "storage", "notifier", "analyzer"]
SetupProbeFn = Callable[..., Awaitable[TestConnectionResponse]]


@dataclass(frozen=True)
class _RegisteredSetupProbe:
    target: SetupProbeTarget
    backend: str
    probe: SetupProbeFn


class SetupProbeRegistry:
    """Registry for setup-only probe handlers keyed by target/backend."""

    def __init__(self) -> None:
        self._probes: dict[tuple[SetupProbeTarget, str], _RegisteredSetupProbe] = {}

    def register(
        self,
        target: SetupProbeTarget,
        backend: str,
    ) -> Callable[[SetupProbeFn], SetupProbeFn]:
        """Register a setup probe for a target/backend pair."""

        normalized_backend = backend.strip().lower()
        key = (target, normalized_backend)

        def decorator(probe: SetupProbeFn) -> SetupProbeFn:
            if key in self._probes:
                raise ValueError(
                    f"Setup probe for {target} backend {normalized_backend!r} is already registered."
                )
            self._probes[key] = _RegisteredSetupProbe(
                target=target,
                backend=normalized_backend,
                probe=probe,
            )
            return probe

        return decorator

    def get(self, target: SetupProbeTarget, backend: str) -> SetupProbeFn | None:
        """Return a registered probe for the target/backend pair, if present."""
        entry = self._probes.get((target, backend.strip().lower()))
        if entry is None:
            return None
        return entry.probe

    def get_backends(self, target: SetupProbeTarget) -> list[str]:
        """Return known special-case backends for a target."""
        return sorted(
            {
                entry.backend
                for entry in self._probes.values()
                if entry.target == target
            }
        )


_SETUP_PROBE_REGISTRY = SetupProbeRegistry()


def setup_probe(target: SetupProbeTarget, backend: str) -> Callable[[SetupProbeFn], SetupProbeFn]:
    """Decorator for registering a setup-only probe handler."""
    return _SETUP_PROBE_REGISTRY.register(target, backend)


def get_setup_probe(target: SetupProbeTarget, backend: str) -> SetupProbeFn | None:
    """Look up a setup-only probe handler."""
    return _SETUP_PROBE_REGISTRY.get(target, backend)


def get_setup_probe_backends(target: SetupProbeTarget) -> list[str]:
    """List special-case backends registered for a setup target."""
    return _SETUP_PROBE_REGISTRY.get_backends(target)
