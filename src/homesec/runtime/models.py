"""Runtime control-plane models."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING

from homesec.notifiers.multiplex import NotifierEntry
from homesec.pipeline import ClipPipeline

if TYPE_CHECKING:
    from homesec.interfaces import AlertPolicy, ClipSource, Notifier, ObjectFilter, VLMAnalyzer
    from homesec.models.config import Config


class RuntimeState(StrEnum):
    """Lifecycle state for the active runtime."""

    IDLE = "idle"
    RELOADING = "reloading"
    FAILED = "failed"


@dataclass(slots=True)
class RuntimeBundle:
    """Restartable runtime unit managed by RuntimeManager."""

    generation: int
    config: Config
    config_signature: str
    notifier: Notifier
    notifier_entries: list[NotifierEntry]
    filter_plugin: ObjectFilter
    vlm_plugin: VLMAnalyzer
    alert_policy: AlertPolicy
    pipeline: ClipPipeline
    sources: list[ClipSource]
    sources_by_camera: dict[str, ClipSource]


@dataclass(slots=True)
class RuntimeStatusSnapshot:
    """Snapshot of runtime-manager status for API consumption."""

    state: RuntimeState
    generation: int
    reload_in_progress: bool
    active_config_version: str | None
    last_reload_at: datetime | None
    last_reload_error: str | None


@dataclass(frozen=True, slots=True)
class RuntimeReloadRequest:
    """Result of requesting a runtime reload."""

    accepted: bool
    message: str
    target_generation: int


@dataclass(frozen=True, slots=True)
class RuntimeReloadResult:
    """Final result of an attempted runtime reload."""

    success: bool
    generation: int
    error: str | None = None


def config_signature(config: Config) -> str:
    """Return a short, stable signature for a config payload."""
    payload = config.model_dump(mode="json")
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:12]
