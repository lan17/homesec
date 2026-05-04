# Storage Uri Local

> 45 nodes · cohesion 0.07

## Key Concepts

- **LocalStorage** (28 connections) — `src/homesec/plugins/storage/local.py`
- **test_local_storage.py** (22 connections) — `tests/homesec/test_local_storage.py`
- **_make_storage()** (20 connections) — `tests/homesec/test_local_storage.py`
- **._ensure_open()** (5 connections) — `src/homesec/plugins/storage/local.py`
- **TestLocalStorageEdgeCases** (4 connections) — `tests/homesec/test_local_storage.py`
- **TestLocalStorageHappyPath** (4 connections) — `tests/homesec/test_local_storage.py`
- **TestLocalStoragePathValidation** (4 connections) — `tests/homesec/test_local_storage.py`
- **TestLocalStorageShutdown** (4 connections) — `tests/homesec/test_local_storage.py`
- **local.py** (4 connections) — `src/homesec/plugins/storage/local.py`
- **.exists()** (4 connections) — `src/homesec/plugins/storage/local.py`
- **._parse_storage_uri()** (4 connections) — `src/homesec/plugins/storage/local.py`
- **.put_file()** (4 connections) — `src/homesec/plugins/storage/local.py`
- **.delete()** (3 connections) — `src/homesec/plugins/storage/local.py`
- **.get()** (3 connections) — `src/homesec/plugins/storage/local.py`
- **test_delete_is_idempotent()** (2 connections) — `tests/homesec/test_local_storage.py`
- **test_exists_returns_false_for_invalid_uri()** (2 connections) — `tests/homesec/test_local_storage.py`
- **test_get_missing_file_raises()** (2 connections) — `tests/homesec/test_local_storage.py`
- **test_get_view_url_returns_file_uri()** (2 connections) — `tests/homesec/test_local_storage.py`
- **test_get_view_url_returns_none_for_non_local_uri()** (2 connections) — `tests/homesec/test_local_storage.py`
- **test_handles_unicode_filenames()** (2 connections) — `tests/homesec/test_local_storage.py`
- **test_operations_fail_after_shutdown()** (2 connections) — `tests/homesec/test_local_storage.py`
- **test_parse_storage_uri_rejects_invalid_prefix()** (2 connections) — `tests/homesec/test_local_storage.py`
- **test_ping_returns_true_when_root_exists()** (2 connections) — `tests/homesec/test_local_storage.py`
- **test_ping_works_after_shutdown()** (2 connections) — `tests/homesec/test_local_storage.py`
- **test_put_creates_parent_directories()** (2 connections) — `tests/homesec/test_local_storage.py`
- *... and 20 more nodes in this community*

## Relationships

- No strong cross-community connections detected

## Source Files

- `src/homesec/plugins/storage/local.py`
- `tests/homesec/test_local_storage.py`

## Audit Trail

- EXTRACTED: 138 (85%)
- INFERRED: 25 (15%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*