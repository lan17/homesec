"""Typed preview signaling contracts shared by API, runtime and sources."""

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


class PreviewOffer(BaseModel):
    model_config = {"extra": "forbid"}

    type: Literal["offer"] = "offer"
    sdp: str = Field(min_length=1, max_length=48_000)


class PreviewAnswer(BaseModel):
    session_id: str
    type: Literal["answer"] = "answer"
    sdp: str = Field(min_length=1, max_length=48_000)


class PreviewSessionAction(BaseModel):
    accepted: bool


class PreviewSessionRefusalReason(StrEnum):
    RECORDING_PRIORITY = "recording_priority"
    SESSION_BUDGET_EXHAUSTED = "session_budget_exhausted"
    PREVIEW_TEMPORARILY_UNAVAILABLE = "preview_temporarily_unavailable"
    SESSION_LIMIT = "session_limit"
    SESSION_NOT_FOUND = "session_not_found"
    INVALID_OFFER = "invalid_offer"
    UNSUPPORTED_TRANSPORT = "unsupported_transport"


class PreviewSessionRefusal(BaseModel):
    reason: PreviewSessionRefusalReason
    message: str


class PreviewIceServer(BaseModel):
    """Browser-safe ICE server description; never persisted as a session record."""

    urls: list[str]
    username: str | None = None
    credential: str | None = None
