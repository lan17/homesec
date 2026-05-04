# Add Cache Run

> 6 nodes · cohesion 0.33

## Key Concepts

- **._create_clip()** (4 connections) — `src/homesec/sources/local_folder.py`
- **._run()** (4 connections) — `src/homesec/sources/local_folder.py`
- **._add_to_cache()** (3 connections) — `src/homesec/sources/local_folder.py`
- **Background task that polls for new files.** (1 connections) — `src/homesec/sources/local_folder.py`
- **Add file to in-memory cache with FIFO eviction.** (1 connections) — `src/homesec/sources/local_folder.py`
- **Create Clip object from file path.** (1 connections) — `src/homesec/sources/local_folder.py`

## Relationships

- No strong cross-community connections detected

## Source Files

- `src/homesec/sources/local_folder.py`

## Audit Trail

- EXTRACTED: 13 (93%)
- INFERRED: 1 (7%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*