# Local Validate Folder

> 14 nodes · cohesion 0.14

## Key Concepts

- **validate_plugin()** (9 connections) — `src/homesec/plugins/registry.py`
- **test_local_folder_camera_connection()** (4 connections) — `src/homesec/sources/local_folder_setup_probe.py`
- **test_local_storage_connection()** (4 connections) — `src/homesec/plugins/storage/local_setup_probe.py`
- **.validate()** (3 connections) — `src/homesec/plugins/registry.py`
- **local_setup_probe.py** (3 connections) — `src/homesec/plugins/storage/local_setup_probe.py`
- **local_folder_setup_probe.py** (2 connections) — `src/homesec/sources/local_folder_setup_probe.py`
- **nearest_existing_parent()** (2 connections) — `src/homesec/plugins/storage/local_setup_probe.py`
- **Validate plugin configuration without instantiating it.** (1 connections) — `src/homesec/plugins/registry.py`
- **Validate configuration for a plugin without instantiating it.** (1 connections) — `src/homesec/plugins/registry.py`
- **Setup-only local-folder source connectivity probe.** (1 connections) — `src/homesec/sources/local_folder_setup_probe.py`
- **Validate local-folder config and ensure the watch directory is accessible.** (1 connections) — `src/homesec/sources/local_folder_setup_probe.py`
- **Setup-only local storage connectivity probe.** (1 connections) — `src/homesec/plugins/storage/local_setup_probe.py`
- **Return the closest existing ancestor for a path, if any.** (1 connections) — `src/homesec/plugins/storage/local_setup_probe.py`
- **Validate local storage config and confirm the target path is usable.** (1 connections) — `src/homesec/plugins/storage/local_setup_probe.py`

## Relationships

- No strong cross-community connections detected

## Source Files

- `src/homesec/plugins/registry.py`
- `src/homesec/plugins/storage/local_setup_probe.py`
- `src/homesec/sources/local_folder_setup_probe.py`

## Audit Trail

- EXTRACTED: 24 (71%)
- INFERRED: 10 (29%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*