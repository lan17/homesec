"""System control endpoints."""

from __future__ import annotations

import os
import signal

from fastapi import APIRouter

router = APIRouter(tags=["system"])


@router.post("/api/v1/system/restart")
async def restart_system() -> dict[str, str]:
    """Request a graceful restart."""
    os.kill(os.getpid(), signal.SIGTERM)
    return {"message": "Shutdown initiated"}
