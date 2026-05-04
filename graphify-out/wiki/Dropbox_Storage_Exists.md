# Dropbox Storage Exists

> 42 nodes · cohesion 0.13

## Key Concepts

- **DropboxStorageConfig** (83 connections) — `src/homesec/plugins/storage/dropbox.py`
- **test_dropbox_storage.py** (41 connections) — `tests/homesec/test_dropbox_storage.py`
- **_FakeDropboxClient** (34 connections) — `tests/homesec/test_dropbox_storage.py`
- **_fake_dropbox_module()** (24 connections) — `tests/homesec/test_dropbox_storage.py`
- **test_dropbox_storage_chunked_upload()** (6 connections) — `tests/homesec/test_dropbox_storage.py`
- **test_dropbox_storage_delete_is_idempotent()** (6 connections) — `tests/homesec/test_dropbox_storage.py`
- **test_dropbox_storage_delete_removes_existing()** (6 connections) — `tests/homesec/test_dropbox_storage.py`
- **test_dropbox_storage_exists_handles_missing()** (6 connections) — `tests/homesec/test_dropbox_storage.py`
- **test_dropbox_storage_get_view_url()** (6 connections) — `tests/homesec/test_dropbox_storage.py`
- **test_dropbox_storage_uses_refresh_token_auth()** (6 connections) — `tests/homesec/test_dropbox_storage.py`
- **test_dropbox_storage_uses_token_auth()** (6 connections) — `tests/homesec/test_dropbox_storage.py`
- **test_chunked_upload_with_append()** (5 connections) — `tests/homesec/test_dropbox_storage.py`
- **test_delete_rejects_invalid_uri()** (5 connections) — `tests/homesec/test_dropbox_storage.py`
- **test_exists_returns_false_for_non_dropbox_uri()** (5 connections) — `tests/homesec/test_dropbox_storage.py`
- **test_exists_returns_true_for_existing()** (5 connections) — `tests/homesec/test_dropbox_storage.py`
- **test_get_creates_parent_directories()** (5 connections) — `tests/homesec/test_dropbox_storage.py`
- **test_get_downloads_file()** (5 connections) — `tests/homesec/test_dropbox_storage.py`
- **test_get_rejects_invalid_uri()** (5 connections) — `tests/homesec/test_dropbox_storage.py`
- **test_operations_fail_after_shutdown()** (5 connections) — `tests/homesec/test_dropbox_storage.py`
- **test_ping_returns_true_when_connected()** (5 connections) — `tests/homesec/test_dropbox_storage.py`
- **test_raises_when_no_credentials()** (5 connections) — `tests/homesec/test_dropbox_storage.py`
- **test_raises_when_partial_refresh_credentials()** (5 connections) — `tests/homesec/test_dropbox_storage.py`
- **test_rejects_backslash_paths()** (5 connections) — `tests/homesec/test_dropbox_storage.py`
- **test_rejects_empty_dest_path()** (5 connections) — `tests/homesec/test_dropbox_storage.py`
- **test_rejects_path_traversal()** (5 connections) — `tests/homesec/test_dropbox_storage.py`
- *... and 17 more nodes in this community*

## Relationships

- No strong cross-community connections detected

## Source Files

- `src/homesec/plugins/storage/dropbox.py`
- `tests/homesec/test_dropbox_storage.py`

## Audit Trail

- EXTRACTED: 190 (59%)
- INFERRED: 130 (41%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*