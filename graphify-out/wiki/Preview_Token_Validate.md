# Preview Token Validate

> 29 nodes · cohesion 0.12

## Key Concepts

- **validate_camera_preview_token()** (14 connections) — `src/homesec/api/preview_tokens.py`
- **preview_tokens.py** (13 connections) — `src/homesec/api/preview_tokens.py`
- **issue_camera_preview_token()** (11 connections) — `src/homesec/api/preview_tokens.py`
- **PreviewTokenError** (7 connections) — `src/homesec/api/preview_tokens.py`
- **test_api_preview_tokens.py** (6 connections) — `tests/homesec/test_api_preview_tokens.py`
- **_decode_payload()** (5 connections) — `src/homesec/api/preview_tokens.py`
- **_sign()** (4 connections) — `src/homesec/api/preview_tokens.py`
- **test_issue_and_validate_camera_preview_token_round_trip()** (4 connections) — `tests/homesec/test_api_preview_tokens.py`
- **test_issue_camera_preview_token_reports_actual_validation_expiry_boundary()** (4 connections) — `tests/homesec/test_api_preview_tokens.py`
- **test_validate_camera_preview_token_rejects_camera_mismatch()** (4 connections) — `tests/homesec/test_api_preview_tokens.py`
- **test_validate_camera_preview_token_rejects_expired_token()** (4 connections) — `tests/homesec/test_api_preview_tokens.py`
- **_base64url_encode()** (3 connections) — `src/homesec/api/preview_tokens.py`
- **PreviewTokenPayload** (3 connections) — `src/homesec/api/preview_tokens.py`
- **_signing_input()** (3 connections) — `src/homesec/api/preview_tokens.py`
- **test_validate_camera_preview_token_rejects_scope_mismatch()** (3 connections) — `tests/homesec/test_api_preview_tokens.py`
- **_base64url_decode()** (2 connections) — `src/homesec/api/preview_tokens.py`
- **_derive_signing_key()** (2 connections) — `src/homesec/api/preview_tokens.py`
- **PreviewTokenErrorCode** (2 connections) — `src/homesec/api/preview_tokens.py`
- **.__init__()** (1 connections) — `src/homesec/api/preview_tokens.py`
- **Preview token creation and verification helpers.** (1 connections) — `src/homesec/api/preview_tokens.py`
- **Raised when preview token validation fails.** (1 connections) — `src/homesec/api/preview_tokens.py`
- **Issue a signed short-lived token for camera preview playback.** (1 connections) — `src/homesec/api/preview_tokens.py`
- **Validate a camera preview token and return decoded payload.** (1 connections) — `src/homesec/api/preview_tokens.py`
- **Unit tests for preview token helpers.** (1 connections) — `tests/homesec/test_api_preview_tokens.py`
- **Preview token expiry metadata should match the server-side validation cutoff.** (1 connections) — `tests/homesec/test_api_preview_tokens.py`
- *... and 4 more nodes in this community*

## Relationships

- No strong cross-community connections detected

## Source Files

- `src/homesec/api/preview_tokens.py`
- `tests/homesec/test_api_preview_tokens.py`

## Audit Trail

- EXTRACTED: 83 (79%)
- INFERRED: 22 (21%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*