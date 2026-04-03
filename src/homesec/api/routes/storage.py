"""Storage configuration endpoints."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, cast

from fastapi import APIRouter, Depends, status
from pydantic import BaseModel

from homesec.api.dependencies import get_homesec_app
from homesec.api.errors import APIError, APIErrorCode
from homesec.api.redaction import is_sensitive_key, redact_config
from homesec.config.errors import StorageConfigInvalidError, StorageMutationError
from homesec.models.config import StorageConfig
from homesec.plugins import discover_all_plugins
from homesec.plugins.registry import PluginType, get_plugin_config_model, get_plugin_names
from homesec.runtime.errors import RuntimeReloadConfigError

if TYPE_CHECKING:
    from homesec.app import Application

router = APIRouter(tags=["storage"])


class StorageResponse(BaseModel):
    backend: str
    config: dict[str, object]
    paths: dict[str, object]


class StorageUpdate(BaseModel):
    backend: str | None = None
    config: dict[str, object] | None = None


class StorageFieldMetadata(BaseModel):
    name: str
    type: str
    required: bool
    description: str | None = None
    default: object | None = None
    secret: bool = False


class StorageBackendMetadata(BaseModel):
    backend: str
    label: str
    description: str
    config_schema: dict[str, object]
    fields: list[StorageFieldMetadata]
    secret_fields: list[str]


class RuntimeReloadResponse(BaseModel):
    accepted: bool
    message: str
    target_generation: int


class StorageChangeResponse(BaseModel):
    restart_required: bool = True
    storage: StorageResponse | None = None
    runtime_reload: RuntimeReloadResponse | None = None


def _storage_response(storage: StorageConfig) -> StorageResponse:
    redacted_storage = redact_config(storage.config)
    if not isinstance(redacted_storage, dict):
        redacted_storage = {}
    paths_payload = storage.paths.model_dump(mode="json")
    return StorageResponse(
        backend=storage.backend,
        config=cast(dict[str, object], redacted_storage),
        paths=cast(dict[str, object], paths_payload),
    )


def _map_storage_config_error(exc: StorageMutationError) -> APIError:
    if isinstance(exc, StorageConfigInvalidError):
        return APIError(
            str(exc),
            status_code=status.HTTP_400_BAD_REQUEST,
            error_code=APIErrorCode.STORAGE_CONFIG_INVALID,
        )
    return APIError(
        str(exc),
        status_code=status.HTTP_400_BAD_REQUEST,
        error_code=APIErrorCode.STORAGE_CONFIG_INVALID,
    )


async def _reload_runtime_if_requested(
    *,
    apply_changes: bool,
    app: Application,
) -> RuntimeReloadResponse | None:
    if not apply_changes:
        return None

    try:
        request = await app.request_runtime_reload()
    except RuntimeReloadConfigError as exc:
        raise APIError(
            str(exc),
            status_code=exc.status_code,
            error_code=exc.error_code,
        ) from exc

    if not request.accepted:
        raise APIError(
            request.message,
            status_code=status.HTTP_409_CONFLICT,
            error_code=APIErrorCode.RELOAD_IN_PROGRESS,
            extra={"target_generation": request.target_generation},
        )

    return RuntimeReloadResponse(
        accepted=True,
        message="Runtime reload accepted",
        target_generation=request.target_generation,
    )


def _field_type(property_schema: dict[str, object]) -> str:
    schema_type = property_schema.get("type")
    if isinstance(schema_type, str):
        return schema_type
    if isinstance(schema_type, list):
        rendered = [item for item in schema_type if isinstance(item, str)]
        if rendered:
            return "|".join(rendered)
    return "object"


def _backend_description(backend: str, schema: dict[str, object]) -> str:
    raw_description = schema.get("description")
    if isinstance(raw_description, str) and raw_description.strip():
        return raw_description.strip()
    return f"{backend.replace('_', ' ').title()} storage backend."


def _backend_metadata(backend: str) -> StorageBackendMetadata:
    config_model = get_plugin_config_model(PluginType.STORAGE, backend)
    schema = config_model.model_json_schema()
    properties_raw = schema.get("properties", {})
    properties = properties_raw if isinstance(properties_raw, dict) else {}
    required_raw = schema.get("required", [])
    required = {item for item in required_raw if isinstance(item, str)}

    fields: list[StorageFieldMetadata] = []
    for field_name, field_schema in properties.items():
        if not isinstance(field_name, str):
            continue
        parsed_schema = field_schema if isinstance(field_schema, dict) else {}
        fields.append(
            StorageFieldMetadata(
                name=field_name,
                type=_field_type(parsed_schema),
                required=field_name in required,
                description=cast(str | None, parsed_schema.get("description")),
                default=parsed_schema.get("default"),
                secret=is_sensitive_key(field_name),
            )
        )

    fields.sort(key=lambda field: field.name)
    return StorageBackendMetadata(
        backend=backend,
        label=backend.replace("_", " ").title(),
        description=_backend_description(backend, cast(dict[str, object], schema)),
        config_schema=cast(dict[str, object], schema),
        fields=fields,
        secret_fields=[field.name for field in fields if field.secret],
    )


@router.get("/api/v1/storage", response_model=StorageResponse)
async def get_storage(app: Application = Depends(get_homesec_app)) -> StorageResponse:
    """Get active storage backend config with redacted secret values."""
    config = await asyncio.to_thread(app.config_manager.get_config)
    return _storage_response(config.storage)


@router.patch("/api/v1/storage", response_model=StorageChangeResponse)
async def patch_storage(
    payload: StorageUpdate,
    apply_changes: bool = False,
    app: Application = Depends(get_homesec_app),
) -> StorageChangeResponse:
    """Partially update storage config and optionally trigger runtime reload."""
    try:
        result = await app.config_manager.update_storage(
            storage_backend=payload.backend,
            storage_config=payload.config,
        )
    except StorageMutationError as exc:
        raise _map_storage_config_error(exc) from exc

    config = await asyncio.to_thread(app.config_manager.get_config)
    runtime_reload = await _reload_runtime_if_requested(apply_changes=apply_changes, app=app)
    return StorageChangeResponse(
        restart_required=False if runtime_reload is not None else result.restart_required,
        storage=_storage_response(config.storage),
        runtime_reload=runtime_reload,
    )


@router.get("/api/v1/storage/backends", response_model=list[StorageBackendMetadata])
async def list_storage_backends() -> list[StorageBackendMetadata]:
    """List available storage backends and their config schema metadata."""
    discover_all_plugins()
    return [_backend_metadata(backend) for backend in get_plugin_names(PluginType.STORAGE)]
