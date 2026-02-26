"""Setup onboarding endpoints."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends, status

from homesec.api.dependencies import get_homesec_app, require_bootstrap_mode
from homesec.api.errors import APIError, APIErrorCode
from homesec.config.loader import ConfigError
from homesec.models.setup import (
    FinalizeRequest,
    FinalizeResponse,
    PreflightResponse,
    SetupStatusResponse,
    TestConnectionRequest,
    TestConnectionResponse,
)
from homesec.services.setup import (
    SetupFinalizeValidationError,
    SetupTestConnectionRequestError,
    finalize_setup,
    get_setup_status,
    run_preflight,
    test_connection,
)

if TYPE_CHECKING:
    from homesec.app import Application

router = APIRouter(tags=["setup"])


@router.get("/api/v1/setup/status", response_model=SetupStatusResponse)
async def get_setup_status_endpoint(
    app: Application = Depends(get_homesec_app),
) -> SetupStatusResponse:
    """Return setup completion status for onboarding UX."""
    return await get_setup_status(app)


@router.post("/api/v1/setup/preflight", response_model=PreflightResponse)
async def run_setup_preflight_endpoint(
    app: Application = Depends(get_homesec_app),
) -> PreflightResponse:
    """Run setup preflight checks for onboarding UX."""
    return await run_preflight(app)


@router.post("/api/v1/setup/test-connection", response_model=TestConnectionResponse)
async def test_setup_connection_endpoint(
    payload: TestConnectionRequest,
    app: Application = Depends(get_homesec_app),
) -> TestConnectionResponse:
    """Run a non-persistent connection test for setup-managed integrations."""
    try:
        return await test_connection(payload, app)
    except SetupTestConnectionRequestError as exc:
        extra: dict[str, object] | None = None
        if exc.available_backends is not None:
            extra = {"available_backends": exc.available_backends}
        raise APIError(
            str(exc),
            status_code=status.HTTP_400_BAD_REQUEST,
            error_code=APIErrorCode.BAD_REQUEST,
            extra=extra,
        ) from exc


@router.post("/api/v1/setup/finalize", response_model=FinalizeResponse)
async def finalize_setup_endpoint(
    payload: FinalizeRequest,
    app: Application = Depends(get_homesec_app),
    _: None = Depends(require_bootstrap_mode),
) -> FinalizeResponse:
    """Persist finalized setup config and request graceful restart."""
    try:
        return await finalize_setup(payload, app)
    except SetupFinalizeValidationError as exc:
        raise APIError(
            str(exc),
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            error_code=APIErrorCode.SETUP_FINALIZE_INVALID,
            extra={"errors": exc.errors},
        ) from exc
    except ConfigError as exc:
        raise APIError(
            str(exc),
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            error_code=exc.code.value,
        ) from exc
