"""Configuration endpoints."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Literal

from fastapi import APIRouter, Depends, status
from pydantic import BaseModel, Field

from homesec.api.dependencies import get_homesec_app
from homesec.api.errors import APIError, APIErrorCode
from homesec.api.redaction import is_sensitive_key, redact_config, redact_url_credentials
from homesec.config.errors import (
    ConfigApplyInProgressError,
    ConfigBackendChangeUnsupportedError,
    ConfigMutationError,
    ConfigPatchInvalidError,
    ConfigSaveError,
    ConfigVersionConflictError,
)
from homesec.config.loader import ConfigError
from homesec.config.manager import ConfigPatch
from homesec.models.config import Config
from homesec.runtime.errors import RuntimeReloadConfigError

if TYPE_CHECKING:
    from homesec.app import Application

router = APIRouter(tags=["config"])


class ConfigResponse(BaseModel):
    """Returns the full config (secrets shown as env var names, not values)."""

    config: dict[str, object]
    saved_config_version: str
    active_config_version: str | None
    apply_required: Literal["none", "reload", "restart"]


class ConfigApplyRequestPayload(BaseModel):
    model_config = {"extra": "forbid"}

    expected_config_version: str = Field(min_length=1)


class ConfigApplyResponse(BaseModel):
    accepted: bool
    message: str
    action: Literal["none", "reload", "restart"]
    target_config_version: str
    target_generation: int | None


# Backwards-compatible aliases used by existing tests and route-local callers.
_redact_url_credentials = redact_url_credentials
_is_sensitive_key = is_sensitive_key
_redact_config = redact_config


@router.get("/api/v1/config", response_model=ConfigResponse)
async def get_config(app: Application = Depends(get_homesec_app)) -> ConfigResponse:
    """Return the saved configuration and the application action it requires."""
    try:
        config = await asyncio.to_thread(app.config_manager.get_config)
    except ConfigError as exc:
        raise _config_load_error(exc) from exc
    return _config_response(app, config)


def _config_response(app: Application, config: Config) -> ConfigResponse:
    payload = config.model_dump(mode="json")
    redacted = _redact_config(payload)
    application_status = app.get_config_application_status(config)
    return ConfigResponse(
        config=redacted if isinstance(redacted, dict) else {},
        saved_config_version=application_status.saved_config_version,
        active_config_version=application_status.active_config_version,
        apply_required=application_status.apply_required,
    )


def _config_load_error(exc: ConfigError) -> APIError:
    return APIError(
        "Saved configuration could not be loaded; check the configuration file",
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        error_code=exc.code.value,
    )


def _config_mutation_error(exc: ConfigMutationError) -> APIError:
    match exc:
        case ConfigApplyInProgressError():
            status_code = status.HTTP_409_CONFLICT
            error_code = APIErrorCode.CONFIG_APPLY_IN_PROGRESS
        case ConfigVersionConflictError():
            status_code = status.HTTP_409_CONFLICT
            error_code = APIErrorCode.CONFIG_VERSION_CONFLICT
        case ConfigBackendChangeUnsupportedError():
            status_code = status.HTTP_400_BAD_REQUEST
            error_code = APIErrorCode.CONFIG_BACKEND_CHANGE_UNSUPPORTED
        case ConfigSaveError():
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            error_code = APIErrorCode.CONFIG_SAVE_FAILED
        case ConfigPatchInvalidError():
            status_code = status.HTTP_422_UNPROCESSABLE_CONTENT
            error_code = APIErrorCode.CONFIG_PATCH_INVALID
        case _:
            status_code = status.HTTP_422_UNPROCESSABLE_CONTENT
            error_code = APIErrorCode.CONFIG_PATCH_INVALID
    return APIError(str(exc), status_code=status_code, error_code=error_code)


@router.patch("/api/v1/config", response_model=ConfigResponse)
async def patch_config(
    payload: ConfigPatch,
    app: Application = Depends(get_homesec_app),
) -> ConfigResponse:
    """Save supported settings edits; applying them is a separate operation."""
    if app.restart_requested:
        raise APIError(
            "HomeSec is restarting; retry saving after it restarts",
            status_code=status.HTTP_409_CONFLICT,
            error_code=APIErrorCode.CONFIG_APPLY_IN_PROGRESS,
        )
    try:
        config = await app.config_manager.patch_config(payload)
    except ConfigMutationError as exc:
        raise _config_mutation_error(exc) from exc
    except ConfigError as exc:
        raise _config_load_error(exc) from exc
    return _config_response(app, config)


@router.post("/api/v1/config/apply", response_model=ConfigApplyResponse, status_code=202)
async def apply_config(
    payload: ConfigApplyRequestPayload,
    app: Application = Depends(get_homesec_app),
) -> ConfigApplyResponse:
    """Accept application of the saved revision the operator reviewed."""
    try:
        request = await app.request_config_apply(payload.expected_config_version)
    except ConfigMutationError as exc:
        raise _config_mutation_error(exc) from exc
    except ConfigError as exc:
        raise _config_load_error(exc) from exc
    except RuntimeReloadConfigError as exc:
        raise APIError(str(exc), status_code=exc.status_code, error_code=exc.error_code) from exc
    return ConfigApplyResponse(
        accepted=request.accepted,
        message=request.message,
        action=request.action,
        target_config_version=request.target_config_version,
        target_generation=request.target_generation,
    )
