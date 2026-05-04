# Token Media Issue

> 47 nodes · cohesion 0.06

## Key Concepts

- **issue_clip_media_token** (6 connections) — `src/homesec/api/media_tokens.py`
- **validate_clip_media_token** (6 connections) — `src/homesec/api/media_tokens.py`
- **src/homesec/api/routes/preview.py** (6 connections) — `src/homesec/api/routes/preview.py`
- **issue_camera_preview_token** (6 connections) — `src/homesec/api/preview_tokens.py`
- **validate_camera_preview_token** (6 connections) — `src/homesec/api/preview_tokens.py`
- **_clip_response** (5 connections) — `src/homesec/api/routes/clips.py`
- **list_clips** (5 connections) — `src/homesec/api/routes/clips.py`
- **_decode_payload** (4 connections) — `src/homesec/api/media_tokens.py`
- **_decode_payload** (4 connections) — `src/homesec/api/preview_tokens.py`
- **APIError** (3 connections) — `src/homesec/api/routes/clips.py`
- **src/homesec/api/routes/clips.py** (3 connections) — `src/homesec/api/routes/clips.py`
- **create_clip_media_token** (3 connections) — `src/homesec/api/routes/clips.py`
- **decode_clip_cursor** (3 connections) — `src/homesec/api/routes/clips.py`
- **encode_clip_cursor** (3 connections) — `src/homesec/api/routes/clips.py`
- **issue_clip_media_token** (3 connections) — `src/homesec/api/routes/clips.py`
- **MediaTokenError** (3 connections) — `src/homesec/api/media_tokens.py`
- **_sign** (3 connections) — `src/homesec/api/media_tokens.py`
- **PreviewTokenError** (3 connections) — `src/homesec/api/preview_tokens.py`
- **_sign** (3 connections) — `src/homesec/api/preview_tokens.py`
- **ClipListResponse** (2 connections) — `src/homesec/api/routes/clips.py`
- **ClipResponse** (2 connections) — `src/homesec/api/routes/clips.py`
- **get_clip** (2 connections) — `src/homesec/api/routes/clips.py`
- **_status_value** (2 connections) — `src/homesec/api/routes/clips.py`
- **_base64url_encode** (2 connections) — `src/homesec/api/media_tokens.py`
- **MediaTokenErrorCode** (2 connections) — `src/homesec/api/media_tokens.py`
- *... and 22 more nodes in this community*

## Relationships

- No strong cross-community connections detected

## Source Files

- `src/homesec/api/media_tokens.py`
- `src/homesec/api/preview_tokens.py`
- `src/homesec/api/routes/clips.py`
- `src/homesec/api/routes/preview.py`

## Audit Trail

- EXTRACTED: 110 (90%)
- INFERRED: 10 (8%)
- AMBIGUOUS: 2 (2%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*