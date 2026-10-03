"""Typed errors for configuration mutations."""

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


class ConfigMutationError(RuntimeError):
    """Base error for saved configuration changes."""

    def __init__(self, message: str, *, cause: Exception | None = None) -> None:
        super().__init__(message)
        if cause is not None:
            self.__cause__ = cause


class ConfigVersionConflictError(ConfigMutationError):
    """The saved configuration changed since the editor read it."""


class ConfigBackendChangeUnsupportedError(ConfigMutationError):
    """An edit attempts to replace the currently configured backend."""


class ConfigPatchInvalidError(ConfigMutationError):
    """The patch cannot produce a valid configuration."""


class ConfigSaveError(ConfigMutationError):
    """The validated configuration could not be persisted."""


class ConfigApplyInProgressError(ConfigMutationError):
    """A process restart has been accepted and configuration writes are frozen."""


class CredentialStoreError(ConfigMutationError):
    """The private managed credential file could not be safely read or written."""
