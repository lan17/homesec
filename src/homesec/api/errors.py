"""Canonical API error envelope and exception mapping."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from enum import StrEnum
from typing import Any

from fastapi import FastAPI, HTTPException, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.requests import Request

logger = logging.getLogger(__name__)


class APIErrorCode(StrEnum):
    """Stable API error codes for non-2xx responses."""

    APP_NOT_INITIALIZED = "APP_NOT_INITIALIZED"
    API_KEY_NOT_CONFIGURED = "API_KEY_NOT_CONFIGURED"
    UNAUTHORIZED = "UNAUTHORIZED"
    DB_UNAVAILABLE = "DB_UNAVAILABLE"
    CAMERA_NOT_FOUND = "CAMERA_NOT_FOUND"
    CAMERA_ALREADY_EXISTS = "CAMERA_ALREADY_EXISTS"
    CAMERA_CONFIG_INVALID = "CAMERA_CONFIG_INVALID"
    CLIP_NOT_FOUND = "CLIP_NOT_FOUND"
    CLIP_MEDIA_UNAVAILABLE = "CLIP_MEDIA_UNAVAILABLE"
    CLIP_MEDIA_FETCH_FAILED = "CLIP_MEDIA_FETCH_FAILED"
    CLIP_STORAGE_DELETE_FAILED = "CLIP_STORAGE_DELETE_FAILED"
    CLIPS_CURSOR_INVALID = "CLIPS_CURSOR_INVALID"
    CLIPS_TIME_RANGE_INVALID = "CLIPS_TIME_RANGE_INVALID"
    CLIPS_TIMESTAMP_TZ_REQUIRED = "CLIPS_TIMESTAMP_TZ_REQUIRED"
    RELOAD_IN_PROGRESS = "RELOAD_IN_PROGRESS"
    REQUEST_VALIDATION_FAILED = "REQUEST_VALIDATION_FAILED"
    BAD_REQUEST = "BAD_REQUEST"
    NOT_FOUND = "NOT_FOUND"
    CONFLICT = "CONFLICT"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
    INTERNAL_SERVER_ERROR = "INTERNAL_SERVER_ERROR"
    HTTP_ERROR = "HTTP_ERROR"


_STATUS_TO_DEFAULT_CODE: dict[int, APIErrorCode] = {
    status.HTTP_400_BAD_REQUEST: APIErrorCode.BAD_REQUEST,
    status.HTTP_401_UNAUTHORIZED: APIErrorCode.UNAUTHORIZED,
    status.HTTP_404_NOT_FOUND: APIErrorCode.NOT_FOUND,
    status.HTTP_409_CONFLICT: APIErrorCode.CONFLICT,
    status.HTTP_503_SERVICE_UNAVAILABLE: APIErrorCode.SERVICE_UNAVAILABLE,
}


class APIErrorResponse(BaseModel):
    """Canonical error envelope returned by API routes."""

    detail: str
    error_code: str


class APIError(RuntimeError):
    """Typed API exception mapped to the canonical error envelope."""

    def __init__(
        self,
        detail: str,
        *,
        status_code: int,
        error_code: str | APIErrorCode,
        extra: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(detail)
        self.status_code = status_code
        if isinstance(error_code, APIErrorCode):
            self.error_code = error_code.value
        else:
            self.error_code = error_code
        self.extra = extra
        self.headers = headers


def _default_error_code_for_status(status_code: int) -> str:
    return _STATUS_TO_DEFAULT_CODE.get(status_code, APIErrorCode.HTTP_ERROR).value


def _error_payload(
    detail: str,
    error_code: str,
    *,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = APIErrorResponse(
        detail=detail,
        error_code=error_code,
    ).model_dump(mode="json")
    if extra:
        payload.update(extra)
    return payload


def _http_error_response(
    *,
    status_code: int,
    detail: object,
    headers: Mapping[str, str] | None,
) -> JSONResponse:
    extra: dict[str, Any] | None = None
    error_code = _default_error_code_for_status(status_code)

    if isinstance(detail, dict):
        detail_obj = dict(detail)
        raw_detail = detail_obj.pop("detail", "Request failed")
        raw_error_code = detail_obj.pop("error_code", error_code)
        detail_message = str(raw_detail)
        error_code = str(raw_error_code)
        extra = detail_obj or None
    else:
        detail_message = str(detail) if detail is not None else "Request failed"

    return JSONResponse(
        status_code=status_code,
        content=_error_payload(
            detail=detail_message,
            error_code=error_code,
            extra=extra,
        ),
        headers=dict(headers) if headers is not None else None,
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Register canonical API error handlers."""

    @app.exception_handler(APIError)
    async def _api_error_handler(request: Request, exc: APIError) -> JSONResponse:
        _ = request
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_payload(
                detail=str(exc),
                error_code=exc.error_code,
                extra=exc.extra,
            ),
            headers=exc.headers,
        )

    @app.exception_handler(RequestValidationError)
    async def _request_validation_error_handler(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        _ = request
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content=_error_payload(
                detail="Request validation failed",
                error_code=APIErrorCode.REQUEST_VALIDATION_FAILED.value,
                extra={"validation_errors": exc.errors()},
            ),
        )

    @app.exception_handler(HTTPException)
    async def _http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        _ = request
        return _http_error_response(
            status_code=exc.status_code,
            detail=exc.detail,
            headers=exc.headers,
        )

    @app.exception_handler(StarletteHTTPException)
    async def _starlette_http_exception_handler(
        request: Request,
        exc: StarletteHTTPException,
    ) -> JSONResponse:
        _ = request
        return _http_error_response(
            status_code=exc.status_code,
            detail=exc.detail,
            headers=exc.headers,
        )

    @app.exception_handler(Exception)
    async def _unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled API exception for path=%s", request.url.path, exc_info=exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=_error_payload(
                detail="Internal server error",
                error_code=APIErrorCode.INTERNAL_SERVER_ERROR.value,
            ),
        )
