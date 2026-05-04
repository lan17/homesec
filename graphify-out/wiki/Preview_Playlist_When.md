# Preview Playlist When

> 109 nodes · cohesion 0.04

## Key Concepts

- **CameraPreviewStatus** (66 connections) — `src/homesec/runtime/models.py`
- **_StubPreviewApp** (50 connections) — `tests/homesec/test_api_preview_routes.py`
- **test_api_preview_routes.py** (44 connections) — `tests/homesec/test_api_preview_routes.py`
- **PreviewConfig** (38 connections) — `src/homesec/models/config.py`
- **_client()** (29 connections) — `tests/homesec/test_api_preview_routes.py`
- **HLSPreviewConfig** (24 connections) — `src/homesec/models/config.py`
- **_MatrixStubApp** (19 connections) — `tests/homesec/test_api_bootstrap_matrix.py`
- **_write_preview_files()** (15 connections) — `tests/homesec/test_api_preview_routes.py`
- **_MatrixCase** (12 connections) — `tests/homesec/test_api_bootstrap_matrix.py`
- **test_preview_tokenized_playlist_url_is_playable_end_to_end()** (9 connections) — `tests/homesec/test_api_preview_routes.py`
- **test_post_preview_returns_tokenized_playlist_when_auth_enabled()** (8 connections) — `tests/homesec/test_api_preview_routes.py`
- **test_preview_playback_rejects_stale_files_when_runtime_reports_inactive_preview()** (8 connections) — `tests/homesec/test_api_preview_routes.py`
- **test_preview_playlist_rejects_invalid_token()** (8 connections) — `tests/homesec/test_api_preview_routes.py`
- **test_preview_playlist_returns_404_when_runtime_reports_missing_camera()** (8 connections) — `tests/homesec/test_api_preview_routes.py`
- **test_preview_playlist_returns_503_when_runtime_status_lookup_is_unavailable()** (8 connections) — `tests/homesec/test_api_preview_routes.py`
- **test_preview_segment_returns_conflict_when_segment_is_missing()** (8 connections) — `tests/homesec/test_api_preview_routes.py`
- **test_api_bootstrap_matrix.py** (8 connections) — `tests/homesec/test_api_bootstrap_matrix.py`
- **_build_client()** (7 connections) — `tests/homesec/test_api_bootstrap_matrix.py`
- **test_preview_playlist_rejects_stale_files_when_preview_disabled()** (7 connections) — `tests/homesec/test_api_preview_routes.py`
- **test_preview_playlist_rejects_stale_files_when_runtime_is_unavailable()** (7 connections) — `tests/homesec/test_api_preview_routes.py`
- **test_preview_playlist_rejects_unknown_camera_before_touching_disk()** (7 connections) — `tests/homesec/test_api_preview_routes.py`
- **test_preview_playlist_returns_conflict_when_playlist_is_empty()** (7 connections) — `tests/homesec/test_api_preview_routes.py`
- **test_preview_playlist_returns_conflict_when_playlist_read_fails()** (7 connections) — `tests/homesec/test_api_preview_routes.py`
- **.__init__()** (6 connections) — `tests/homesec/test_api_preview_routes.py`
- **test_post_preview_refusals_map_to_stable_conflicts()** (6 connections) — `tests/homesec/test_api_preview_routes.py`
- *... and 84 more nodes in this community*

## Relationships

- No strong cross-community connections detected

## Source Files

- `src/homesec/models/config.py`
- `src/homesec/runtime/models.py`
- `tests/homesec/test_api_bootstrap_matrix.py`
- `tests/homesec/test_api_preview_routes.py`
- `tests/homesec/test_app.py`

## Audit Trail

- EXTRACTED: 372 (62%)
- INFERRED: 225 (38%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*