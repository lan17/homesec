"""Setup onboarding endpoints."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends

from homesec.api.dependencies import get_homesec_app
from homesec.models.setup import PreflightResponse, SetupStatusResponse
from homesec.services.setup import get_setup_status, run_preflight

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
