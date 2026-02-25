"""Local retention pruning for uploaded clips."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Final

from homesec.models.enums import ClipStatus
from homesec.repository import ClipRepository

logger = logging.getLogger(__name__)

_DEFAULT_SUFFIXES: Final[tuple[str, ...]] = (".mp4",)


@dataclass(frozen=True)
class RetentionPruneSummary:
    """Summary for a single prune pass."""

    reason: str
    max_local_size_bytes: int
    discovered_local_files: int
    measured_local_files: int
    unmeasured_local_files: int
    measured_local_bytes: int
    non_eligible_local_bytes: int
    measurement_incomplete: bool
    blocked_over_limit: bool
    eligible_candidates: int
    eligible_bytes_before: int
    eligible_bytes_after: int
    reclaimed_bytes: int
    deleted_files: int
    skipped_not_done: int
    skipped_no_state: int
    skipped_not_uploaded: int
    skipped_path_mismatch: int
    skipped_stat_error: int
    skipped_missing_race: int
    delete_errors: int

    @classmethod
    def empty(
        cls,
        *,
        reason: str,
        max_local_size_bytes: int = 0,
    ) -> RetentionPruneSummary:
        return cls(
            reason=reason,
            max_local_size_bytes=max_local_size_bytes,
            discovered_local_files=0,
            measured_local_files=0,
            unmeasured_local_files=0,
            measured_local_bytes=0,
            non_eligible_local_bytes=0,
            measurement_incomplete=False,
            blocked_over_limit=False,
            eligible_candidates=0,
            eligible_bytes_before=0,
            eligible_bytes_after=0,
            reclaimed_bytes=0,
            deleted_files=0,
            skipped_not_done=0,
            skipped_no_state=0,
            skipped_not_uploaded=0,
            skipped_path_mismatch=0,
            skipped_stat_error=0,
            skipped_missing_race=0,
            delete_errors=0,
        )


@dataclass(frozen=True)
class _Candidate:
    clip_id: str
    local_path: Path
    created_at: datetime
    size_bytes: int


@dataclass
class _PruneCounters:
    skipped_not_done: int = 0
    skipped_no_state: int = 0
    skipped_not_uploaded: int = 0
    skipped_path_mismatch: int = 0
    skipped_stat_error: int = 0
    skipped_missing_race: int = 0
    delete_errors: int = 0


@dataclass(frozen=True)
class _ScanResult:
    local_files: list[Path]
    local_file_sizes: dict[Path, int]
    skipped_stat_error: int


@dataclass(frozen=True)
class _DeleteResult:
    remaining_bytes: int
    deleted_files: int
    skipped_missing_race: int
    delete_errors: int


class LocalRetentionPruner:
    """Delete oldest uploaded local clips until a byte cap is satisfied.

    This pruner is intentionally local-only:
    - It never deletes from remote storage.
    - It never mutates DB state or events.
    """

    def __init__(
        self,
        *,
        repository: ClipRepository,
        local_clip_dirs: list[Path],
        max_local_size_bytes: int,
        clip_suffixes: tuple[str, ...] = _DEFAULT_SUFFIXES,
    ) -> None:
        if max_local_size_bytes < 0:
            raise ValueError("max_local_size_bytes must be >= 0")

        normalized_suffixes = tuple(_normalize_suffix(suffix) for suffix in clip_suffixes)
        if not normalized_suffixes:
            raise ValueError("clip_suffixes must not be empty")

        self._repository = repository
        self._max_local_size_bytes = max_local_size_bytes
        self._local_clip_dirs = {_normalize_path(path) for path in local_clip_dirs}
        self._clip_suffixes = normalized_suffixes

    async def prune_once(
        self,
        *,
        reason: str,
        clip_local_path: Path | None = None,
    ) -> RetentionPruneSummary:
        """Run one local retention prune pass."""
        counters = _PruneCounters()

        if clip_local_path is not None:
            self._register_local_dir_from_clip(clip_local_path)

        scan_result = await asyncio.to_thread(
            self._scan_local_files,
            sorted(self._local_clip_dirs),
        )
        local_files = scan_result.local_files
        local_file_sizes = scan_result.local_file_sizes
        counters.skipped_stat_error = scan_result.skipped_stat_error
        measured_local_bytes = sum(local_file_sizes.values())
        unmeasured_local_files = len(local_files) - len(local_file_sizes)
        candidates = await self._build_candidates(
            local_files=local_files,
            local_file_sizes=local_file_sizes,
            counters=counters,
        )
        eligible_before = sum(candidate.size_bytes for candidate in candidates)

        if eligible_before <= self._max_local_size_bytes:
            summary = self._build_summary(
                reason=reason,
                discovered_local_files=len(local_files),
                measured_local_files=len(local_file_sizes),
                unmeasured_local_files=unmeasured_local_files,
                measured_local_bytes=measured_local_bytes,
                eligible_candidates=len(candidates),
                eligible_bytes_before=eligible_before,
                eligible_bytes_after=eligible_before,
                deleted_files=0,
                counters=counters,
            )
            logger.info("Retention prune summary: %s", summary)
            return summary

        delete_result = await asyncio.to_thread(
            self._delete_candidates_until_cap,
            sorted(candidates, key=lambda item: (item.created_at, item.clip_id)),
            eligible_before,
        )
        remaining_bytes = delete_result.remaining_bytes
        deleted_files = delete_result.deleted_files
        counters.skipped_missing_race = delete_result.skipped_missing_race
        counters.delete_errors = delete_result.delete_errors

        summary = self._build_summary(
            reason=reason,
            discovered_local_files=len(local_files),
            measured_local_files=len(local_file_sizes),
            unmeasured_local_files=unmeasured_local_files,
            measured_local_bytes=measured_local_bytes,
            eligible_candidates=len(candidates),
            eligible_bytes_before=eligible_before,
            eligible_bytes_after=remaining_bytes,
            deleted_files=deleted_files,
            counters=counters,
        )
        logger.info("Retention prune summary: %s", summary)
        return summary

    async def _build_candidates(
        self,
        *,
        local_files: list[Path],
        local_file_sizes: dict[Path, int],
        counters: _PruneCounters,
    ) -> list[_Candidate]:
        candidates: list[_Candidate] = []
        lookup_clip_ids = sorted({local_path.stem for local_path in local_files})

        state_by_clip_id = await self._repository.get_clip_states_with_created_at(lookup_clip_ids)

        for local_path in local_files:
            clip_id = local_path.stem
            state_with_created = state_by_clip_id.get(clip_id)
            if state_with_created is None:
                counters.skipped_no_state += 1
                continue

            state, created_at = state_with_created
            if state.storage_uri is None:
                counters.skipped_not_uploaded += 1
                continue

            if state.status != ClipStatus.DONE:
                counters.skipped_not_done += 1
                continue

            if not _paths_match(state.local_path, local_path):
                counters.skipped_path_mismatch += 1
                continue

            size_bytes = local_file_sizes.get(local_path)
            if size_bytes is None:
                # Stat failures are counted during the filesystem scan phase.
                continue

            candidates.append(
                _Candidate(
                    clip_id=clip_id,
                    local_path=local_path,
                    created_at=created_at,
                    size_bytes=size_bytes,
                )
            )

        return candidates

    def _scan_local_files(self, local_clip_dirs: list[Path]) -> _ScanResult:
        local_files = self._discover_local_files(local_clip_dirs)
        local_file_sizes: dict[Path, int] = {}
        skipped_stat_error = 0
        for local_path in local_files:
            try:
                local_file_sizes[local_path] = local_path.stat().st_size
            except OSError as exc:
                skipped_stat_error += 1
                logger.warning(
                    "Retention stat failed: path=%s error=%s",
                    local_path,
                    exc,
                    exc_info=exc,
                )
        return _ScanResult(
            local_files=local_files,
            local_file_sizes=local_file_sizes,
            skipped_stat_error=skipped_stat_error,
        )

    def _delete_candidates_until_cap(
        self,
        candidates: list[_Candidate],
        eligible_before: int,
    ) -> _DeleteResult:
        remaining_bytes = eligible_before
        deleted_files = 0
        skipped_missing_race = 0
        delete_errors = 0

        for candidate in candidates:
            if remaining_bytes <= self._max_local_size_bytes:
                break
            try:
                candidate.local_path.unlink()
                deleted_files += 1
                remaining_bytes = max(0, remaining_bytes - candidate.size_bytes)
            except FileNotFoundError:
                skipped_missing_race += 1
                remaining_bytes = max(0, remaining_bytes - candidate.size_bytes)
                logger.info(
                    "Retention skip: file disappeared before delete: clip=%s path=%s",
                    candidate.clip_id,
                    candidate.local_path,
                )
            except OSError as exc:
                delete_errors += 1
                logger.warning(
                    "Retention delete failed: clip=%s path=%s error=%s",
                    candidate.clip_id,
                    candidate.local_path,
                    exc,
                    exc_info=exc,
                )
        return _DeleteResult(
            remaining_bytes=remaining_bytes,
            deleted_files=deleted_files,
            skipped_missing_race=skipped_missing_race,
            delete_errors=delete_errors,
        )

    def _build_summary(
        self,
        *,
        reason: str,
        discovered_local_files: int,
        measured_local_files: int,
        unmeasured_local_files: int,
        measured_local_bytes: int,
        eligible_candidates: int,
        eligible_bytes_before: int,
        eligible_bytes_after: int,
        deleted_files: int,
        counters: _PruneCounters,
    ) -> RetentionPruneSummary:
        non_eligible_local_bytes = max(0, measured_local_bytes - eligible_bytes_before)
        measured_local_bytes_after = non_eligible_local_bytes + eligible_bytes_after
        measurement_incomplete = unmeasured_local_files > 0
        return RetentionPruneSummary(
            reason=reason,
            max_local_size_bytes=self._max_local_size_bytes,
            discovered_local_files=discovered_local_files,
            measured_local_files=measured_local_files,
            unmeasured_local_files=unmeasured_local_files,
            measured_local_bytes=measured_local_bytes,
            non_eligible_local_bytes=non_eligible_local_bytes,
            measurement_incomplete=measurement_incomplete,
            # Conservative: if any local files were unmeasured, we cannot safely claim
            # that disk usage is below the retention cap.
            blocked_over_limit=measurement_incomplete
            or measured_local_bytes_after > self._max_local_size_bytes,
            eligible_candidates=eligible_candidates,
            eligible_bytes_before=eligible_bytes_before,
            eligible_bytes_after=eligible_bytes_after,
            reclaimed_bytes=max(0, eligible_bytes_before - eligible_bytes_after),
            deleted_files=deleted_files,
            skipped_not_done=counters.skipped_not_done,
            skipped_no_state=counters.skipped_no_state,
            skipped_not_uploaded=counters.skipped_not_uploaded,
            skipped_path_mismatch=counters.skipped_path_mismatch,
            skipped_stat_error=counters.skipped_stat_error,
            skipped_missing_race=counters.skipped_missing_race,
            delete_errors=counters.delete_errors,
        )

    def _register_local_dir_from_clip(self, clip_local_path: Path) -> None:
        local_dir = _normalize_path(Path(clip_local_path).parent)
        if local_dir in self._local_clip_dirs:
            return
        self._local_clip_dirs.add(local_dir)
        logger.info("Retention registered local clip directory from clip arrival: %s", local_dir)

    def _discover_local_files(self, local_clip_dirs: list[Path]) -> list[Path]:
        discovered: list[Path] = []
        seen: set[Path] = set()
        patterns = tuple(f"*{suffix}" for suffix in self._clip_suffixes)

        for root in local_clip_dirs:
            if not root.exists():
                continue
            if root.is_file():
                resolved = _normalize_path(root)
                if resolved.suffix.lower() in self._clip_suffixes and resolved not in seen:
                    seen.add(resolved)
                    discovered.append(resolved)
                continue
            if not root.is_dir():
                continue

            for pattern in patterns:
                for candidate in root.rglob(pattern):
                    try:
                        is_file = candidate.is_file()
                    except OSError:
                        # Be conservative on path-type probe errors: keep candidate in scan flow
                        # so downstream stat() can record skipped_stat_error and mark the summary
                        # as measurement_incomplete.
                        is_file = True
                    if not is_file:
                        continue
                    resolved = _normalize_path(candidate)
                    if resolved in seen:
                        continue
                    seen.add(resolved)
                    discovered.append(resolved)

        discovered.sort()
        return discovered


def _normalize_suffix(raw: str) -> str:
    suffix = raw.strip().lower()
    if not suffix:
        raise ValueError("clip suffix must be non-empty")
    if not suffix.startswith("."):
        suffix = f".{suffix}"
    return suffix


def _normalize_path(raw: Path) -> Path:
    candidate = Path(raw).expanduser()
    try:
        return candidate.resolve()
    except OSError:
        return candidate


def _paths_match(state_local_path: str, discovered_path: Path) -> bool:
    """Match state local_path to discovered file path after normalization."""
    try:
        state_path = Path(state_local_path).expanduser().resolve()
    except OSError:
        return False
    return state_path == discovered_path
