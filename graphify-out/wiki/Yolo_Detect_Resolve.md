# Yolo Detect Resolve

> 26 nodes · cohesion 0.10

## Key Concepts

- **YOLOFilter** (13 connections) — `src/homesec/plugins/filters/yolo.py`
- **yolo.py** (10 connections) — `src/homesec/plugins/filters/yolo.py`
- **_detect_worker()** (4 connections) — `src/homesec/plugins/filters/yolo.py`
- **_resolve_weights_path()** (4 connections) — `src/homesec/plugins/filters/yolo.py`
- **.detect()** (4 connections) — `src/homesec/plugins/filters/yolo.py`
- **.__init__()** (4 connections) — `src/homesec/plugins/filters/yolo.py`
- **_check_file_safe()** (3 connections) — `src/homesec/plugins/filters/yolo.py`
- **_ensure_yolo_dependencies()** (3 connections) — `src/homesec/plugins/filters/yolo.py`
- **_get_model()** (3 connections) — `src/homesec/plugins/filters/yolo.py`
- **_resolve_requested_path()** (2 connections) — `src/homesec/plugins/filters/yolo.py`
- **._apply_overrides()** (2 connections) — `src/homesec/plugins/filters/yolo.py`
- **._class_ids_for()** (2 connections) — `src/homesec/plugins/filters/yolo.py`
- **.ping()** (2 connections) — `src/homesec/plugins/filters/yolo.py`
- **.shutdown()** (2 connections) — `src/homesec/plugins/filters/yolo.py`
- **ObjectFilter** (2 connections)
- **create()** (1 connections) — `src/homesec/plugins/filters/yolo.py`
- **YOLO object detection filter plugin.** (1 connections) — `src/homesec/plugins/filters/yolo.py`
- **Get or create a cached YOLO model instance.      Thread-safe: uses a lock to pre** (1 connections) — `src/homesec/plugins/filters/yolo.py`
- **YOLO-based object detection filter.      Uses ProcessPoolExecutor internally for** (1 connections) — `src/homesec/plugins/filters/yolo.py`
- **Initialize YOLO filter with validated settings.          Args:             setti** (1 connections) — `src/homesec/plugins/filters/yolo.py`
- **Detect objects in video clip.          Runs inference in ProcessPoolExecutor to** (1 connections) — `src/homesec/plugins/filters/yolo.py`
- **Cleanup resources - shutdown executor.** (1 connections) — `src/homesec/plugins/filters/yolo.py`
- **Health check - verify executor is alive and model path exists.** (1 connections) — `src/homesec/plugins/filters/yolo.py`
- **Worker function for video analysis (must be at module level for pickling).** (1 connections) — `src/homesec/plugins/filters/yolo.py`
- **Fail fast with a clear error if YOLO dependencies are missing.** (1 connections) — `src/homesec/plugins/filters/yolo.py`
- *... and 1 more nodes in this community*

## Relationships

- No strong cross-community connections detected

## Source Files

- `src/homesec/plugins/filters/yolo.py`

## Audit Trail

- EXTRACTED: 66 (93%)
- INFERRED: 5 (7%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*