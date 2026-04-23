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


class RuntimePreviewError(RuntimeError):
    """Base runtime preview error with a stable machine-readable code."""

    def __init__(self, message: str, *, error_code: str) -> None:
        super().__init__(message)
        self.error_code = error_code


class PreviewCameraNotFoundError(RuntimePreviewError):
    """Raised when preview control targets an unknown camera."""

    def __init__(self, camera_name: str) -> None:
        super().__init__(
            f"Camera '{camera_name}' not found in the active runtime",
            error_code="PREVIEW_CAMERA_NOT_FOUND",
        )


class PreviewRuntimeUnavailableError(RuntimePreviewError):
    """Raised when the runtime cannot accept preview control commands."""

    def __init__(self, message: str) -> None:
        super().__init__(
            message,
            error_code="PREVIEW_RUNTIME_UNAVAILABLE",
        )
