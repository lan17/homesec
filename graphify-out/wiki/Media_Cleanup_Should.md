# Media Cleanup Should

> 20 nodes · cohesion 0.14

## Key Concepts

- **get_clip_media()** (6 connections) — `src/homesec/api/routes/media.py`
- **media.py** (6 connections) — `src/homesec/api/routes/media.py`
- **test_api_media_route_helpers.py** (6 connections) — `tests/homesec/test_api_media_route_helpers.py`
- **_cleanup_media_temp_dir()** (4 connections) — `src/homesec/api/routes/media.py`
- **test_build_media_filename_sanitizes_clip_id_and_normalizes_suffix()** (3 connections) — `tests/homesec/test_api_media_route_helpers.py`
- **test_cleanup_media_temp_dir_ignores_missing_directory()** (3 connections) — `tests/homesec/test_api_media_route_helpers.py`
- **test_cleanup_media_temp_dir_logs_warning_on_unexpected_error()** (3 connections) — `tests/homesec/test_api_media_route_helpers.py`
- **test_guess_media_type_falls_back_to_octet_stream_for_unknown_extension()** (3 connections) — `tests/homesec/test_api_media_route_helpers.py`
- **test_infer_media_suffix_parses_uri_and_defaults_mp4()** (3 connections) — `tests/homesec/test_api_media_route_helpers.py`
- **_build_media_filename()** (3 connections) — `src/homesec/api/routes/media.py`
- **_guess_media_type()** (3 connections) — `src/homesec/api/routes/media.py`
- **_infer_media_suffix()** (3 connections) — `src/homesec/api/routes/media.py`
- **Behavioral tests for clip media route helper functions.** (1 connections) — `tests/homesec/test_api_media_route_helpers.py`
- **Cleanup helper should silently ignore missing temp dirs.** (1 connections) — `tests/homesec/test_api_media_route_helpers.py`
- **Cleanup helper should warn on unexpected filesystem errors.** (1 connections) — `tests/homesec/test_api_media_route_helpers.py`
- **Suffix inference should strip query/fragment and default to mp4 when absent.** (1 connections) — `tests/homesec/test_api_media_route_helpers.py`
- **Filename builder should sanitize clip IDs and normalize invalid suffixes.** (1 connections) — `tests/homesec/test_api_media_route_helpers.py`
- **Media type guess should return octet-stream when MIME type is unknown.** (1 connections) — `tests/homesec/test_api_media_route_helpers.py`
- **Clip media playback endpoint.** (1 connections) — `src/homesec/api/routes/media.py`
- **Stream clip media through HomeSec for in-app playback.** (1 connections) — `src/homesec/api/routes/media.py`

## Relationships

- No strong cross-community connections detected

## Source Files

- `src/homesec/api/routes/media.py`
- `tests/homesec/test_api_media_route_helpers.py`

## Audit Trail

- EXTRACTED: 44 (81%)
- INFERRED: 10 (19%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*