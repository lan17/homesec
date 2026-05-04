# When Extension Anonymous

> 48 nodes · cohesion 0.07

## Key Concepts

- **FtpSource** (36 connections) — `src/homesec/sources/ftp.py`
- **FtpSourceConfig** (26 connections) — `src/homesec/sources/ftp.py`
- **TestFtpSourceFileHandling** (11 connections) — `tests/homesec/test_ftp_source.py`
- **TestFtpSourceConfiguration** (9 connections) — `tests/homesec/test_ftp_source.py`
- **.test_anonymous_mode_no_credentials_required()** (4 connections) — `tests/homesec/test_ftp_source.py`
- **.test_ftp_subdir_appended_to_root()** (4 connections) — `tests/homesec/test_ftp_source.py`
- **.test_non_anonymous_with_credentials()** (4 connections) — `tests/homesec/test_ftp_source.py`
- **.test_raises_without_credentials_when_not_anonymous()** (4 connections) — `tests/homesec/test_ftp_source.py`
- **.test_all_extensions_allowed_when_empty()** (4 connections) — `tests/homesec/test_ftp_source.py`
- **.test_allowed_extension_emits_clip()** (4 connections) — `tests/homesec/test_ftp_source.py`
- **.test_disallowed_extension_kept_when_not_configured()** (4 connections) — `tests/homesec/test_ftp_source.py`
- **.test_emitted_clip_has_expected_fields()** (4 connections) — `tests/homesec/test_ftp_source.py`
- **.test_extension_check_is_case_insensitive()** (4 connections) — `tests/homesec/test_ftp_source.py`
- **.test_incomplete_upload_kept_when_not_configured()** (4 connections) — `tests/homesec/test_ftp_source.py`
- **test_ftp_source.py** (4 connections) — `tests/homesec/test_ftp_source.py`
- **test_accepts_allowed_extension()** (3 connections) — `tests/homesec/test_clip_sources.py`
- **test_incomplete_upload_deletes_when_enabled()** (3 connections) — `tests/homesec/test_clip_sources.py`
- **test_rejects_non_matching_extension()** (3 connections) — `tests/homesec/test_clip_sources.py`
- **test_ftp_upload_emits_clip()** (3 connections) — `tests/homesec/test_integration.py`
- **._create_clip()** (3 connections) — `src/homesec/sources/ftp.py`
- **._create_server()** (3 connections) — `src/homesec/sources/ftp.py`
- **._handle_file_received()** (3 connections) — `src/homesec/sources/ftp.py`
- **.__init__()** (3 connections) — `src/homesec/sources/ftp.py`
- **._is_extension_allowed()** (2 connections) — `src/homesec/sources/ftp.py`
- **.is_healthy()** (2 connections) — `src/homesec/sources/ftp.py`
- *... and 23 more nodes in this community*

## Relationships

- No strong cross-community connections detected

## Source Files

- `src/homesec/sources/ftp.py`
- `tests/homesec/test_clip_sources.py`
- `tests/homesec/test_ftp_source.py`
- `tests/homesec/test_integration.py`

## Audit Trail

- EXTRACTED: 100 (56%)
- INFERRED: 79 (44%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*