"""Runtime control-plane specific errors."""

from __future__ import annotations


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
