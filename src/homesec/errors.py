"""Error hierarchy for HomeSec pipeline stages."""

from __future__ import annotations


class PipelineError(Exception):
    """Base exception for all pipeline errors.
    
    Compatible with error-as-value pattern: instances can be returned as values
    instead of raised. Preserves stack traces via exception chaining.
    """

    def __init__(
        self, message: str, stage: str, clip_id: str, cause: Exception | None = None
    ) -> None:
        super().__init__(message)
        self.stage = stage
        self.clip_id = clip_id
        self.cause = cause
        self.__cause__ = cause  # Python's exception chaining


class UploadError(PipelineError):
    """Storage upload failed."""

    def __init__(
        self, clip_id: str, storage_uri: str | None, cause: Exception
    ) -> None:
        super().__init__(
            f"Upload failed for {clip_id}", stage="upload", clip_id=clip_id, cause=cause
        )
        self.storage_uri = storage_uri


class FilterError(PipelineError):
    """Object detection filter failed."""

    def __init__(self, clip_id: str, plugin_name: str, cause: Exception) -> None:
        super().__init__(
            f"Filter failed for {clip_id} (plugin: {plugin_name})",
            stage="filter",
            clip_id=clip_id,
            cause=cause,
        )
        self.plugin_name = plugin_name


class VLMError(PipelineError):
    """VLM analysis failed."""

    def __init__(self, clip_id: str, plugin_name: str, cause: Exception) -> None:
        super().__init__(
            f"VLM analysis failed for {clip_id} (plugin: {plugin_name})",
            stage="vlm",
            clip_id=clip_id,
            cause=cause,
        )
        self.plugin_name = plugin_name


class NotifyError(PipelineError):
    """Notification delivery failed."""

    def __init__(self, clip_id: str, notifier_name: str, cause: Exception) -> None:
        super().__init__(
            f"Notify failed for {clip_id} (notifier: {notifier_name})",
            stage="notify",
            clip_id=clip_id,
            cause=cause,
        )
        self.notifier_name = notifier_name
