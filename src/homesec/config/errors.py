"""Typed config-manager mutation errors for camera CRUD flows."""

from __future__ import annotations


class CameraMutationError(RuntimeError):
    """Base error for camera configuration mutations."""

    def __init__(self, message: str, *, cause: Exception | None = None) -> None:
        super().__init__(message)
        if cause is not None:
            self.__cause__ = cause


class CameraAlreadyExistsError(CameraMutationError):
    """Raised when attempting to add an already-existing camera."""


class CameraNotFoundError(CameraMutationError):
    """Raised when a camera lookup fails."""


class CameraConfigInvalidError(CameraMutationError):
    """Raised when a camera config mutation fails validation."""


class CameraConfigRedactedPlaceholderError(CameraConfigInvalidError):
    """Raised when a source_config mutation attempts to persist redacted placeholders."""


class StorageMutationError(RuntimeError):
    """Base error for storage configuration mutations."""

    def __init__(self, message: str, *, cause: Exception | None = None) -> None:
        super().__init__(message)
        if cause is not None:
            self.__cause__ = cause


class StorageConfigInvalidError(StorageMutationError):
    """Raised when a storage config mutation fails validation."""


class StorageConfigRedactedPlaceholderError(StorageConfigInvalidError):
    """Raised when a storage config mutation attempts to persist redacted placeholders."""
