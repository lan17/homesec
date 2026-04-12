"""Runtime control-plane specific errors."""

from __future__ import annotations


def sanitize_runtime_error(exc: Exception, *, max_length: int = 512) -> str:
    """Return a bounded runtime error message for status surfaces."""
    value = str(exc).strip()
    if not value:
        value = type(exc).__name__
    return value[:max_length]


class RuntimeReloadConfigError(RuntimeError):
    """Raised when reload request config cannot be used."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int,
        error_code: str,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code
