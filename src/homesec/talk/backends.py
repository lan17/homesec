"""Talk backend adapter contracts and registry primitives."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel

from homesec.models.config import CameraTalkConfig, TalkConfig
from homesec.models.talk import (
    TalkCapabilityProbeResult,
    TalkRefusalReason,
    TalkSessionOpenRequest,
)
from homesec.talk.backend_ids import normalize_talk_backend_id, validate_talk_backend_id

TalkBackendConfidence = Literal["explicit", "high", "medium", "low", "not_applicable"]


@runtime_checkable
class TalkBackendAdapter(Protocol):
    """Protocol implemented by camera speaker backend adapters."""

    name: str

    @property
    def supported_codecs(self) -> list[str]:
        """Camera-side codecs this backend can emit."""
        ...

    async def probe(self) -> TalkCapabilityProbeResult:
        """Probe whether this backend can provide talk for the camera."""
        ...

    async def open_session(self, request: TalkSessionOpenRequest) -> TalkBackendSession:
        """Open a backend-specific talk session for a reserved HomeSec session."""
        ...


@runtime_checkable
class TalkBackendSession(Protocol):
    """Camera talk session returned by a backend adapter."""

    session_id: str
    camera_name: str

    @property
    def selected_codec(self) -> str:
        """Codec selected by the camera adapter for this session."""
        ...

    async def write_pcm_frame(self, frame: bytes) -> None:
        """Send one PCM frame to the camera speaker."""
        ...

    async def close(self) -> None:
        """Close the camera talk session."""
        ...


@dataclass(frozen=True, slots=True)
class CameraTalkFingerprint:
    """Sparse camera identity hints available to talk backend detectors."""

    manufacturer: str | None = None
    model: str | None = None
    firmware: str | None = None
    profile_names: tuple[str, ...] = ()
    rtsp_hosts: tuple[str, ...] = ()
    open_ports: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class TalkBackendDetection:
    """Detector result used by auto backend selection."""

    backend: str
    confidence: TalkBackendConfidence
    reason: str
    safe_to_probe: bool = False
    requires_credentials: bool = False


@dataclass(frozen=True, slots=True)
class TalkBackendContext:
    """Source-local context available while building and selecting talk backends."""

    camera_name: str
    source_backend: str
    runtime_talk: TalkConfig
    camera_talk: CameraTalkConfig
    source_host: str | None = None
    source_uri: str | None = None
    source_uri_env: str | None = None
    resolved_source_uri: str | None = None
    source_connect_timeout_s: float = 5.0
    source_io_timeout_s: float = 5.0
    fingerprint: CameraTalkFingerprint = field(default_factory=CameraTalkFingerprint)
    resolve_env: Callable[[str], str | None] | None = None
    redact: Callable[[str], str] | None = None

    def env_value(self, name: str) -> str | None:
        """Resolve an environment value through the injected resolver when present."""
        if self.resolve_env is None:
            return None
        return self.resolve_env(name)

    def redacted(self, value: str) -> str:
        """Redact a diagnostic value through the injected redactor when present."""
        if self.redact is None:
            return value
        return self.redact(value)


class TalkBackendOpenError(RuntimeError):
    """Raised when a selected talk backend refuses or fails session open."""

    def __init__(self, message: str, *, reason: TalkRefusalReason) -> None:
        super().__init__(message)
        self.reason = reason


class TalkBackendConfigError(ValueError):
    """Raised when a backend has a safe public config error to report."""

    def __init__(self, public_message: str, *, cause: Exception | None = None) -> None:
        super().__init__(public_message)
        self.public_message = public_message
        self.__cause__ = cause


@dataclass(frozen=True, slots=True)
class TalkBackendRegistration:
    """Factory and detector metadata for one talk backend."""

    name: str
    config_model: type[BaseModel]
    factory: Callable[[BaseModel, TalkBackendContext], TalkBackendAdapter]
    config_factory: (
        Callable[
            [Mapping[str, object] | BaseModel, TalkBackendContext],
            BaseModel,
        ]
        | None
    ) = None
    detector: Callable[[TalkBackendContext], TalkBackendDetection] | None = None
    priority: int = 100
    standards_based: bool = False

    def __post_init__(self) -> None:
        normalized = validate_talk_backend_id(normalize_talk_backend_id(self.name))
        if normalized != self.name:
            object.__setattr__(self, "name", normalized)


class TalkBackendRegistry:
    """In-memory registry for built-in talk backend registrations."""

    def __init__(self) -> None:
        self._registrations: dict[str, TalkBackendRegistration] = {}

    def register(self, registration: TalkBackendRegistration) -> None:
        """Register one backend, failing fast on duplicate names."""
        name = validate_talk_backend_id(normalize_talk_backend_id(registration.name))
        if name in self._registrations:
            raise ValueError(f"Talk backend '{name}' already registered")
        self._registrations[name] = registration

    def get(self, name: str) -> TalkBackendRegistration | None:
        """Return a registration by backend name."""
        normalized = normalize_talk_backend_id(name)
        try:
            validate_talk_backend_id(normalized)
        except ValueError:
            return None
        return self._registrations.get(normalized)

    def all(self) -> list[TalkBackendRegistration]:
        """Return all registrations in deterministic selection order."""
        return sorted(
            self._registrations.values(),
            key=lambda registration: (
                not registration.standards_based,
                registration.priority,
                registration.name,
            ),
        )

    def standards_first(self) -> list[TalkBackendRegistration]:
        """Return standards-based backends before proprietary candidates."""
        return self.all()

    def names(self) -> tuple[str, ...]:
        """Return registered backend names in deterministic order."""
        return tuple(registration.name for registration in self.all())


def backend_config_for(
    camera_talk: CameraTalkConfig,
    backend_name: str,
) -> dict[str, object]:
    """Return backend config using the Phase 9 compatibility alias rules."""
    return dict(camera_talk.config_for_backend(backend_name))


def model_validate_backend_config(
    registration: TalkBackendRegistration,
    raw_config: Mapping[str, object] | BaseModel,
    *,
    context: TalkBackendContext | None = None,
) -> BaseModel:
    """Validate backend-specific config against the registered config model."""
    if registration.config_factory is not None:
        if context is None:
            raise ValueError("Talk backend context is required for config validation")
        return registration.config_factory(raw_config, context)
    if isinstance(raw_config, registration.config_model):
        return raw_config
    if isinstance(raw_config, BaseModel):
        raw_config = raw_config.model_dump(mode="json")
    return registration.config_model.model_validate(dict(raw_config))
