# Preview Segment Path

> 46 nodes · cohesion 0.08

## Key Concepts

- **preview.py** (23 connections) — `src/homesec/api/routes/preview.py`
- **.__init__()** (12 connections) — `src/homesec/sources/rtsp/live_publisher.py`
- **preview_camera_dir()** (11 connections) — `src/homesec/preview_paths.py`
- **ensure_preview_active()** (9 connections) — `src/homesec/api/routes/preview.py`
- **get_preview_playlist()** (8 connections) — `src/homesec/api/routes/preview.py`
- **get_preview_segment()** (8 connections) — `src/homesec/api/routes/preview.py`
- **preview_paths.py** (8 connections) — `src/homesec/preview_paths.py`
- **preview_segment_path()** (6 connections) — `src/homesec/preview_paths.py`
- **_raise_camera_not_found()** (6 connections) — `src/homesec/api/routes/preview.py`
- **preview_playlist_path()** (5 connections) — `src/homesec/preview_paths.py`
- **sanitize_preview_camera_name()** (5 connections) — `src/homesec/preview_paths.py`
- **_ensure_preview_media_active()** (5 connections) — `src/homesec/api/routes/preview.py`
- **force_stop_preview()** (5 connections) — `src/homesec/api/routes/preview.py`
- **get_preview_status()** (5 connections) — `src/homesec/api/routes/preview.py`
- **_raise_runtime_unavailable()** (5 connections) — `src/homesec/api/routes/preview.py`
- **is_preview_segment_name()** (4 connections) — `src/homesec/preview_paths.py`
- **preview_ffmpeg_log_path()** (4 connections) — `src/homesec/preview_paths.py`
- **preview_segment_filename_pattern()** (4 connections) — `src/homesec/preview_paths.py`
- **_ensure_known_camera()** (4 connections) — `src/homesec/api/routes/preview.py`
- **_ensure_preview_playback_enabled()** (4 connections) — `src/homesec/api/routes/preview.py`
- **_schedule_preview_viewer_activity_best_effort()** (4 connections) — `src/homesec/api/routes/preview.py`
- **test_preview_segment_path_rejects_invalid_segment_names()** (3 connections) — `tests/homesec/test_api_preview_routes.py`
- **_preview_storage_dir()** (3 connections) — `src/homesec/api/routes/preview.py`
- **_status_response()** (3 connections) — `src/homesec/api/routes/preview.py`
- **_note_preview_viewer_activity_best_effort()** (2 connections) — `src/homesec/api/routes/preview.py`
- *... and 21 more nodes in this community*

## Relationships

- No strong cross-community connections detected

## Source Files

- `src/homesec/api/routes/preview.py`
- `src/homesec/preview_paths.py`
- `src/homesec/sources/rtsp/live_publisher.py`
- `tests/homesec/test_api_preview_routes.py`

## Audit Trail

- EXTRACTED: 153 (84%)
- INFERRED: 29 (16%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*