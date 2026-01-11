"""Cleanup workflow for removing clips that appear empty after re-analysis.

This module is intended to be run via the HomeSec CLI (`homesec cleanup`).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field

from homesec.config import load_config, resolve_env_var
from homesec.interfaces import ObjectFilter, StorageBackend
from homesec.models.clip import ClipStateData
from homesec.models.filter import FilterConfig, YoloFilterSettings
from homesec.plugins import discover_all_plugins
from homesec.plugins.filters import load_filter_plugin
from homesec.plugins.storage import create_storage
from homesec.repository.clip_repository import ClipRepository
from homesec.state.postgres import PostgresStateStore

logger = logging.getLogger("homesec.cleanup_clips")

_DEFAULT_RECHECK_MODEL = "yolo11x.pt"


class CleanupOptions(BaseModel):
    """Options for the cleanup workflow (CLI-facing)."""

    config_path: Path

    older_than_days: int | None = None
    camera_name: str | None = None

    batch_size: int = Field(default=100, ge=1)
    workers: int = Field(default=2, ge=1)
    dry_run: bool = True

    recheck_model_path: str | None = None
    recheck_min_confidence: float | None = None
    recheck_sample_fps: int | None = None
    recheck_min_box_h_ratio: float | None = None
    recheck_min_hits: int | None = None


@dataclass(frozen=True)
class _Counts:
    scanned_rows: int = 0
    candidates: int = 0
    reanalyzed: int = 0
    deleted: int = 0
    false_negatives: int = 0
    download_errors: int = 0
    analyze_errors: int = 0
    delete_errors: int = 0
    state_errors: int = 0

    def __add__(self, other: "_Counts") -> "_Counts":
        return _Counts(
            scanned_rows=self.scanned_rows + other.scanned_rows,
            candidates=self.candidates + other.candidates,
            reanalyzed=self.reanalyzed + other.reanalyzed,
            deleted=self.deleted + other.deleted,
            false_negatives=self.false_negatives + other.false_negatives,
            download_errors=self.download_errors + other.download_errors,
            analyze_errors=self.analyze_errors + other.analyze_errors,
            delete_errors=self.delete_errors + other.delete_errors,
            state_errors=self.state_errors + other.state_errors,
        )


def _safe_filename(value: str) -> str:
    return value.replace("/", "_").replace(" ", "_")


def _log_json(level: int, message: str, payload: dict[str, object]) -> None:
    if "message" not in payload:
        payload = {"message": message, **payload}
    logger.log(level, json.dumps(payload, sort_keys=True))


def _base_payload(
    *,
    run_id: str,
    event: str,
    clip_id: str | None = None,
    camera_name: str | None = None,
    created_at: datetime | None = None,
    dry_run: bool | None = None,
    status_before: str | None = None,
    status_after: str | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {"event": event, "run_id": run_id}
    if clip_id is not None:
        payload["clip_id"] = clip_id
    if camera_name is not None:
        payload["camera_name"] = camera_name
    if created_at is not None:
        payload["created_at"] = created_at.isoformat()
    if dry_run is not None:
        payload["dry_run"] = bool(dry_run)
    if status_before is not None:
        payload["status_before"] = status_before
    if status_after is not None:
        payload["status_after"] = status_after
    return payload


def _recheck_settings(config: FilterConfig) -> dict[str, object]:
    match config.config:
        case YoloFilterSettings() as settings:
            return {
                "model_path": str(settings.model_path),
                "min_confidence": float(settings.min_confidence),
                "sample_fps": int(settings.sample_fps),
                "min_box_h_ratio": float(settings.min_box_h_ratio),
                "min_hits": int(settings.min_hits),
                "classes": list(settings.classes),
            }
        case _:
            return {}


def _build_recheck_filter_config(base: FilterConfig, opts: CleanupOptions) -> FilterConfig:
    match base.config:
        case YoloFilterSettings() as yolo:
            settings = yolo.model_copy(deep=True)
        case _:
            raise ValueError(f"Unsupported filter config type: {type(base.config).__name__}")

    settings.model_path = opts.recheck_model_path or _DEFAULT_RECHECK_MODEL
    if opts.recheck_min_confidence is not None:
        settings.min_confidence = opts.recheck_min_confidence
    if opts.recheck_sample_fps is not None:
        settings.sample_fps = opts.recheck_sample_fps
    if opts.recheck_min_box_h_ratio is not None:
        settings.min_box_h_ratio = opts.recheck_min_box_h_ratio
    if opts.recheck_min_hits is not None:
        settings.min_hits = opts.recheck_min_hits

    merged = base.model_copy(deep=True)
    merged.max_workers = int(opts.workers)
    merged.config = settings
    return merged


async def run_cleanup(opts: CleanupOptions) -> None:
    """Run the cleanup workflow."""

    run_id = str(uuid.uuid4())

    cfg = load_config(opts.config_path)

    # Discover all plugins (built-in and external)
    discover_all_plugins()

    state_cfg = cfg.state_store
    dsn = state_cfg.dsn
    if state_cfg.dsn_env:
        dsn = resolve_env_var(state_cfg.dsn_env)
    if not dsn:
        raise RuntimeError("Postgres DSN is required for cleanup")

    storage = create_storage(cfg.storage)
    state_store = PostgresStateStore(dsn)
    ok = await state_store.initialize()
    if not ok:
        raise RuntimeError("Failed to initialize Postgres state store")

    event_store = state_store.create_event_store()
    repo = ClipRepository(state_store, event_store, retry=cfg.retry)

    recheck_cfg = _build_recheck_filter_config(cfg.filter, opts)
    filter_plugin = load_filter_plugin(recheck_cfg)

    sem = asyncio.Semaphore(int(opts.workers))

    cache_dir = Path.cwd() / "video_cache" / "cleanup" / run_id
    cache_dir.mkdir(parents=True, exist_ok=True)

    totals = _Counts()

    try:
        cursor: tuple[datetime, str] | None = None
        while True:
            rows = await repo.list_candidate_clips_for_cleanup(
                older_than_days=opts.older_than_days,
                camera_name=opts.camera_name,
                batch_size=int(opts.batch_size),
                cursor=cursor,
            )
            if not rows:
                break
            last_clip_id, _last_state, last_created_at = rows[-1]
            cursor = (last_created_at, last_clip_id)

            totals = totals + _Counts(scanned_rows=len(rows))

            candidates: list[tuple[str, ClipStateData, datetime]] = [
                (clip_id, state, created_at)
                for clip_id, state, created_at in rows
                if state.filter_result is not None
                and not state.filter_result.detected_classes
            ]
            totals = totals + _Counts(candidates=len(candidates))

            tasks = [
                asyncio.create_task(
                    _process_candidate(
                        clip_id=clip_id,
                        state=state,
                        created_at=created_at,
                        recheck_cfg=recheck_cfg,
                        filter_plugin=filter_plugin,
                        storage=storage,
                        repo=repo,
                        cache_dir=cache_dir,
                        sem=sem,
                        dry_run=bool(opts.dry_run),
                        run_id=run_id,
                    )
                )
                for clip_id, state, created_at in candidates
            ]
            if tasks:
                results = await asyncio.gather(*tasks)
                for c in results:
                    totals = totals + c

        summary_payload = _base_payload(
            run_id=run_id,
            event="cleanup.summary",
            dry_run=bool(opts.dry_run),
        )
        summary_payload.update(
            {
                "scanned_rows": totals.scanned_rows,
                "candidates": totals.candidates,
                "reanalyzed": totals.reanalyzed,
                "deleted": totals.deleted,
                "false_negatives": totals.false_negatives,
                "download_errors": totals.download_errors,
                "analyze_errors": totals.analyze_errors,
                "delete_errors": totals.delete_errors,
                "state_errors": totals.state_errors,
                "filters": {
                    "older_than_days": opts.older_than_days,
                    "camera_name": opts.camera_name,
                },
                "recheck_settings": _recheck_settings(recheck_cfg),
                "batch_size": int(opts.batch_size),
                "workers": int(opts.workers),
            }
        )
        _log_json(logging.INFO, "Cleanup summary", summary_payload)
    finally:
        try:
            await filter_plugin.shutdown()
        finally:
            await storage.shutdown()
            await state_store.shutdown()


async def _process_candidate(
    *,
    clip_id: str,
    state: ClipStateData,
    created_at: datetime,
    recheck_cfg: FilterConfig,
    filter_plugin: ObjectFilter,
    storage: StorageBackend,
    repo: ClipRepository,
    cache_dir: Path,
    sem: asyncio.Semaphore,
    dry_run: bool,
    run_id: str,
) -> _Counts:
    async with sem:
        status_before = state.status
        prior_filter = state.filter_result

        local_path = Path(state.local_path)
        local_path_str = str(local_path)
        storage_uri = state.storage_uri
        video_path = local_path
        downloaded_path: Path | None = None
        download_ms: int | None = None
        analyze_ms: int | None = None

        try:
            if not video_path.exists():
                if state.storage_uri is None:
                    payload = _base_payload(
                        run_id=run_id,
                        event="cleanup.error",
                        clip_id=clip_id,
                        camera_name=state.camera_name,
                        created_at=created_at,
                        dry_run=dry_run,
                        status_before=status_before,
                    )
                    payload.update(
                        {
                            "error_code": "missing_local_and_storage_uri",
                            "local_path": local_path_str,
                            "storage_uri": storage_uri,
                        }
                    )
                    _log_json(
                        logging.WARNING,
                        "Cleanup error: missing local file and storage URI",
                        payload,
                    )
                    return _Counts(download_errors=1)

                suffix = local_path.suffix or ".mp4"
                downloaded_path = cache_dir / f"{_safe_filename(clip_id)}{suffix}"
                download_start = time.monotonic()
                await storage.get(state.storage_uri, downloaded_path)
                download_ms = int((time.monotonic() - download_start) * 1000)
                video_path = downloaded_path
        except Exception as exc:
            payload = _base_payload(
                run_id=run_id,
                event="cleanup.error",
                clip_id=clip_id,
                camera_name=state.camera_name,
                created_at=created_at,
                dry_run=dry_run,
                status_before=status_before,
            )
            payload.update(
                {
                    "error_code": "download_failed",
                    "error_detail": str(exc),
                    "local_path": local_path_str,
                    "storage_uri": storage_uri,
                }
            )
            if download_ms is not None:
                payload["download_ms"] = download_ms
            _log_json(logging.WARNING, "Cleanup error: download failed", payload)
            return _Counts(download_errors=1)

        try:
            analyze_start = time.monotonic()
            result = await filter_plugin.detect(video_path)
            analyze_ms = int((time.monotonic() - analyze_start) * 1000)
        except Exception as exc:
            if analyze_ms is None:
                analyze_ms = int((time.monotonic() - analyze_start) * 1000)
            payload = _base_payload(
                run_id=run_id,
                event="cleanup.error",
                clip_id=clip_id,
                camera_name=state.camera_name,
                created_at=created_at,
                dry_run=dry_run,
                status_before=status_before,
            )
            payload.update(
                {
                    "error_code": "reanalyze_failed",
                    "error_detail": str(exc),
                    "local_path": local_path_str,
                    "storage_uri": storage_uri,
                }
            )
            if download_ms is not None:
                payload["download_ms"] = download_ms
            payload["reanalyze_ms"] = analyze_ms
            _log_json(logging.WARNING, "Cleanup error: reanalysis failed", payload)
            return _Counts(reanalyzed=1, analyze_errors=1)
        finally:
            if downloaded_path is not None:
                try:
                    downloaded_path.unlink(missing_ok=True)
                except Exception:
                    pass

        recheck_result = result
        recheck = {
            "detected_classes": list(recheck_result.detected_classes),
            "confidence": float(recheck_result.confidence),
            "model": str(recheck_result.model),
            "sampled_frames": int(recheck_result.sampled_frames),
            "settings": {
                "model_path": str(getattr(recheck_cfg.config, "model_path", "")),
                "min_confidence": float(getattr(recheck_cfg.config, "min_confidence", 0.0)),
                "sample_fps": int(getattr(recheck_cfg.config, "sample_fps", 0)),
                "min_box_h_ratio": float(getattr(recheck_cfg.config, "min_box_h_ratio", 0.0)),
                "min_hits": int(getattr(recheck_cfg.config, "min_hits", 0)),
            },
        }

        if recheck_result.detected_classes:
            payload = _base_payload(
                run_id=run_id,
                event="cleanup.skipped_with_detection",
                clip_id=clip_id,
                camera_name=state.camera_name,
                created_at=created_at,
                dry_run=dry_run,
                status_before=status_before,
                status_after=status_before,
            )
            payload.update(
                {
                    "prior_filter": prior_filter.model_dump(mode="json") if prior_filter else None,
                    "recheck_filter": recheck,
                }
            )
            if download_ms is not None:
                payload["download_ms"] = download_ms
            if analyze_ms is not None:
                payload["reanalyze_ms"] = analyze_ms
            _log_json(logging.INFO, "Cleanup skipped: detection found", payload)
            if dry_run:
                return _Counts(reanalyzed=1, false_negatives=1)
            try:
                await repo.record_clip_rechecked(
                    clip_id,
                    result=recheck_result,
                    prior_filter=prior_filter,
                    reason="cleanup_cli",
                    run_id=run_id,
                )
            except Exception as exc:
                payload = _base_payload(
                    run_id=run_id,
                    event="cleanup.error",
                    clip_id=clip_id,
                    camera_name=state.camera_name,
                    created_at=created_at,
                    dry_run=dry_run,
                    status_before=status_before,
                )
                payload.update(
                    {
                        "error_code": "state_update_failed",
                        "error_detail": str(exc),
                        "local_path": local_path_str,
                        "storage_uri": storage_uri,
                    }
                )
                if download_ms is not None:
                    payload["download_ms"] = download_ms
                if analyze_ms is not None:
                    payload["reanalyze_ms"] = analyze_ms
                _log_json(logging.WARNING, "Cleanup error: state update failed", payload)
                return _Counts(reanalyzed=1, false_negatives=1, state_errors=1)
            return _Counts(reanalyzed=1, false_negatives=1)

        # Still empty after recheck.
        delete_local_attempted = local_path.exists()
        delete_storage_attempted = state.storage_uri is not None

        if dry_run:
            payload = _base_payload(
                run_id=run_id,
                event="cleanup.deleted",
                clip_id=clip_id,
                camera_name=state.camera_name,
                created_at=created_at,
                dry_run=dry_run,
                status_before=status_before,
                status_after=status_before,
            )
            payload.update(
                {
                    "local_path": local_path_str,
                    "storage_uri": storage_uri,
                    "prior_filter": prior_filter.model_dump(mode="json") if prior_filter else None,
                    "recheck_filter": recheck,
                    "delete": {
                        "local": {"attempted": delete_local_attempted, "ok": None, "error": None},
                        "storage": {
                            "attempted": delete_storage_attempted,
                            "ok": None,
                            "error": None,
                        },
                    },
                }
            )
            if download_ms is not None:
                payload["download_ms"] = download_ms
            if analyze_ms is not None:
                payload["reanalyze_ms"] = analyze_ms
            _log_json(logging.WARNING, "Cleanup dry-run: would delete empty clip", payload)
            return _Counts(reanalyzed=1)

        delete_ms: int | None = None
        delete_start = time.monotonic()

        delete_local_ok = True
        delete_local_err: str | None = None
        if delete_local_attempted:
            try:
                local_path.unlink(missing_ok=True)
            except Exception as exc:
                delete_local_ok = False
                delete_local_err = str(exc)

        delete_storage_ok = True
        delete_storage_err: str | None = None
        if state.storage_uri is not None:
            try:
                await storage.delete(state.storage_uri)
            except Exception as exc:
                delete_storage_ok = False
                delete_storage_err = str(exc)

        delete_ms = int((time.monotonic() - delete_start) * 1000)
        deleted_local = not local_path.exists()
        deleted_storage = True if state.storage_uri is None else delete_storage_ok

        if not delete_local_ok or not delete_storage_ok:
            payload = _base_payload(
                run_id=run_id,
                event="cleanup.error",
                clip_id=clip_id,
                camera_name=state.camera_name,
                created_at=created_at,
                dry_run=dry_run,
                status_before=status_before,
            )
            payload.update(
                {
                    "error_code": "delete_failed",
                    "local_path": local_path_str,
                    "storage_uri": storage_uri,
                    "delete": {
                        "local": {
                            "attempted": delete_local_attempted,
                            "ok": delete_local_ok,
                            "error": delete_local_err,
                        },
                        "storage": {
                            "attempted": delete_storage_attempted,
                            "ok": delete_storage_ok,
                            "error": delete_storage_err,
                        },
                    },
                }
            )
            if download_ms is not None:
                payload["download_ms"] = download_ms
            if analyze_ms is not None:
                payload["reanalyze_ms"] = analyze_ms
            payload["delete_ms"] = delete_ms
            _log_json(logging.WARNING, "Cleanup error: delete failed", payload)
            return _Counts(reanalyzed=1, delete_errors=1)

        try:
            await repo.record_clip_deleted(
                clip_id,
                reason="cleanup_cli",
                run_id=run_id,
                deleted_local=deleted_local,
                deleted_storage=deleted_storage,
            )
        except Exception as exc:
            payload = _base_payload(
                run_id=run_id,
                event="cleanup.error",
                clip_id=clip_id,
                camera_name=state.camera_name,
                created_at=created_at,
                dry_run=dry_run,
                status_before=status_before,
            )
            payload.update(
                {
                    "error_code": "state_update_failed",
                    "error_detail": str(exc),
                    "local_path": local_path_str,
                    "storage_uri": storage_uri,
                }
            )
            if download_ms is not None:
                payload["download_ms"] = download_ms
            if analyze_ms is not None:
                payload["reanalyze_ms"] = analyze_ms
            payload["delete_ms"] = delete_ms
            _log_json(logging.WARNING, "Cleanup error: state update failed", payload)
            return _Counts(reanalyzed=1, state_errors=1)

        payload = _base_payload(
            run_id=run_id,
            event="cleanup.deleted",
            clip_id=clip_id,
            camera_name=state.camera_name,
            created_at=created_at,
            dry_run=dry_run,
            status_before=status_before,
            status_after="deleted",
        )
        payload.update(
            {
                "local_path": local_path_str,
                "storage_uri": storage_uri,
                "prior_filter": prior_filter.model_dump(mode="json") if prior_filter else None,
                "recheck_filter": recheck,
                "delete": {
                    "local": {
                        "attempted": delete_local_attempted,
                        "ok": delete_local_ok,
                        "error": delete_local_err,
                    },
                    "storage": {
                        "attempted": delete_storage_attempted,
                        "ok": delete_storage_ok,
                        "error": delete_storage_err,
                    },
                },
            }
        )
        if download_ms is not None:
            payload["download_ms"] = download_ms
        if analyze_ms is not None:
            payload["reanalyze_ms"] = analyze_ms
        payload["delete_ms"] = delete_ms
        _log_json(logging.WARNING, "Cleanup deleted empty clip", payload)

        return _Counts(reanalyzed=1, deleted=1)
