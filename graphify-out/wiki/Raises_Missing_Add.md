# Raises Missing Add

> 29 nodes · cohesion 0.10

## Key Concepts

- **test_config_manager.py** (15 connections) — `tests/homesec/test_config_manager.py`
- **_write_config()** (14 connections) — `tests/homesec/test_config_manager.py`
- **test_config_manager_first_save_creates_restrictive_mode_file()** (4 connections) — `tests/homesec/test_config_manager.py`
- **test_config_manager_replace_config_replaces_full_document()** (4 connections) — `tests/homesec/test_config_manager.py`
- **test_config_manager_add_camera_concurrent_updates_preserve_all_changes()** (3 connections) — `tests/homesec/test_config_manager.py`
- **test_config_manager_add_duplicate_raises()** (3 connections) — `tests/homesec/test_config_manager.py`
- **test_config_manager_add_update_remove_camera()** (3 connections) — `tests/homesec/test_config_manager.py`
- **test_config_manager_enforces_restrictive_file_modes_on_save()** (3 connections) — `tests/homesec/test_config_manager.py`
- **test_config_manager_invalid_update_raises()** (3 connections) — `tests/homesec/test_config_manager.py`
- **test_config_manager_remove_missing_raises()** (3 connections) — `tests/homesec/test_config_manager.py`
- **test_config_manager_update_camera_allows_null_clear_for_optional_source_fields()** (3 connections) — `tests/homesec/test_config_manager.py`
- **test_config_manager_update_camera_merges_source_config_and_preserves_existing_keys()** (3 connections) — `tests/homesec/test_config_manager.py`
- **test_config_manager_update_camera_rejects_redacted_placeholder_values()** (3 connections) — `tests/homesec/test_config_manager.py`
- **test_config_manager_update_camera_supports_backend_switch_with_new_config()** (3 connections) — `tests/homesec/test_config_manager.py`
- **test_config_manager_update_missing_raises()** (3 connections) — `tests/homesec/test_config_manager.py`
- **Tests for ConfigManager.** (1 connections) — `tests/homesec/test_config_manager.py`
- **ConfigManager should error when updating a missing camera.** (1 connections) — `tests/homesec/test_config_manager.py`
- **ConfigManager should error when removing a missing camera.** (1 connections) — `tests/homesec/test_config_manager.py`
- **ConfigManager should reject invalid camera config updates.** (1 connections) — `tests/homesec/test_config_manager.py`
- **ConfigManager should serialize concurrent camera adds to avoid lost updates.** (1 connections) — `tests/homesec/test_config_manager.py`
- **Camera updates should merge source_config patches instead of replacing full conf** (1 connections) — `tests/homesec/test_config_manager.py`
- **Camera update should reject redacted placeholders in source_config patches.** (1 connections) — `tests/homesec/test_config_manager.py`
- **Camera update should allow switching source backend when config patch is valid.** (1 connections) — `tests/homesec/test_config_manager.py`
- **Null values in source_config patch should clear optional fields.** (1 connections) — `tests/homesec/test_config_manager.py`
- **Config writes should enforce 0600 mode for config and backup files.** (1 connections) — `tests/homesec/test_config_manager.py`
- *... and 4 more nodes in this community*

## Relationships

- No strong cross-community connections detected

## Source Files

- `tests/homesec/test_config_manager.py`

## Audit Trail

- EXTRACTED: 80 (95%)
- INFERRED: 4 (5%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*