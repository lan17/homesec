# Token Media Validate

> 26 nodes · cohesion 0.12

## Key Concepts

- **validate_clip_media_token()** (17 connections) — `src/homesec/api/media_tokens.py`
- **issue_clip_media_token()** (12 connections) — `src/homesec/api/media_tokens.py`
- **test_api_media_tokens.py** (10 connections) — `tests/homesec/test_api_media_tokens.py`
- **test_issue_and_validate_clip_media_token_round_trip()** (4 connections) — `tests/homesec/test_api_media_tokens.py`
- **test_validate_clip_media_token_rejects_expired_token()** (4 connections) — `tests/homesec/test_api_media_tokens.py`
- **test_validate_clip_media_token_rejects_invalid_version()** (4 connections) — `tests/homesec/test_api_media_tokens.py`
- **test_validate_clip_media_token_rejects_mismatched_clip_id()** (4 connections) — `tests/homesec/test_api_media_tokens.py`
- **test_validate_clip_media_token_rejects_tampered_signature()** (4 connections) — `tests/homesec/test_api_media_tokens.py`
- **_base64url_encode()** (3 connections) — `src/homesec/api/media_tokens.py`
- **_signing_input()** (3 connections) — `src/homesec/api/media_tokens.py`
- **test_validate_clip_media_token_rejects_non_json_payload()** (3 connections) — `tests/homesec/test_api_media_tokens.py`
- **test_validate_clip_media_token_rejects_non_object_payload()** (3 connections) — `tests/homesec/test_api_media_tokens.py`
- **test_validate_clip_media_token_rejects_scope_mismatch()** (3 connections) — `tests/homesec/test_api_media_tokens.py`
- **test_validate_clip_media_token_rejects_wrong_payload_types()** (3 connections) — `tests/homesec/test_api_media_tokens.py`
- **Issue a signed short-lived token for clip media playback.** (1 connections) — `src/homesec/api/media_tokens.py`
- **Validate clip media token and return decoded payload.** (1 connections) — `src/homesec/api/media_tokens.py`
- **Unit tests for media token helpers.** (1 connections) — `tests/homesec/test_api_media_tokens.py`
- **Token should be rejected when version prefix is not supported.** (1 connections) — `tests/homesec/test_api_media_tokens.py`
- **Token should be rejected when scope does not match clip-media scope.** (1 connections) — `tests/homesec/test_api_media_tokens.py`
- **Token should be rejected when payload is not valid JSON.** (1 connections) — `tests/homesec/test_api_media_tokens.py`
- **Issued token should validate for the same clip.** (1 connections) — `tests/homesec/test_api_media_tokens.py`
- **Token should be rejected when decoded payload is not an object.** (1 connections) — `tests/homesec/test_api_media_tokens.py`
- **Token should be rejected when payload field types are invalid.** (1 connections) — `tests/homesec/test_api_media_tokens.py`
- **Token should be rejected when requested clip does not match.** (1 connections) — `tests/homesec/test_api_media_tokens.py`
- **Token should be rejected once it has expired.** (1 connections) — `tests/homesec/test_api_media_tokens.py`
- *... and 1 more nodes in this community*

## Relationships

- No strong cross-community connections detected

## Source Files

- `src/homesec/api/media_tokens.py`
- `tests/homesec/test_api_media_tokens.py`

## Audit Trail

- EXTRACTED: 59 (66%)
- INFERRED: 30 (34%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*