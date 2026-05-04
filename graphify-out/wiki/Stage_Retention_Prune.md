# Stage Retention Prune

> 33 nodes · cohesion 0.09

## Key Concepts

- **ClipPipeline** (72 connections) — `src/homesec/pipeline/core.py`
- **._process_clip()** (9 connections) — `src/homesec/pipeline/core.py`
- **._run_stage_with_retries()** (6 connections) — `src/homesec/pipeline/core.py`
- **._upload_stage()** (6 connections) — `src/homesec/pipeline/core.py`
- **._create_task()** (5 connections) — `src/homesec/pipeline/core.py`
- **._filter_stage()** (5 connections) — `src/homesec/pipeline/core.py`
- **._notify_stage()** (5 connections) — `src/homesec/pipeline/core.py`
- **._vlm_stage()** (5 connections) — `src/homesec/pipeline/core.py`
- **._notify_with_entry()** (4 connections) — `src/homesec/pipeline/core.py`
- **.request_retention_prune()** (4 connections) — `src/homesec/pipeline/core.py`
- **._vlm_skip_reason()** (4 connections) — `src/homesec/pipeline/core.py`
- **._apply_upload_result()** (3 connections) — `src/homesec/pipeline/core.py`
- **.on_new_clip()** (3 connections) — `src/homesec/pipeline/core.py`
- **._log_task_exception()** (2 connections) — `src/homesec/pipeline/core.py`
- **._run_retention_prune()** (2 connections) — `src/homesec/pipeline/core.py`
- **.set_event_loop()** (2 connections) — `src/homesec/pipeline/core.py`
- **.shutdown()** (2 connections) — `src/homesec/pipeline/core.py`
- **._on_retention_prune_done()** (1 connections) — `src/homesec/pipeline/core.py`
- **Set event loop for thread-safe callback handling.          Must be called before** (1 connections) — `src/homesec/pipeline/core.py`
- **Request one retention prune pass with single-flight drop policy.** (1 connections) — `src/homesec/pipeline/core.py`
- **Create and track a processing task in the given loop.** (1 connections) — `src/homesec/pipeline/core.py`
- **Log unexpected task exceptions.** (1 connections) — `src/homesec/pipeline/core.py`
- **Callback for ClipSource when new clip is ready.          Thread-safe: can be cal** (1 connections) — `src/homesec/pipeline/core.py`
- **Process a single clip through all stages.          Flow:         1. Parallel: up** (1 connections) — `src/homesec/pipeline/core.py`
- **Run stage with retry logic and event emission.** (1 connections) — `src/homesec/pipeline/core.py`
- *... and 8 more nodes in this community*

## Relationships

- No strong cross-community connections detected

## Source Files

- `src/homesec/pipeline/core.py`

## Audit Trail

- EXTRACTED: 96 (62%)
- INFERRED: 59 (38%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*