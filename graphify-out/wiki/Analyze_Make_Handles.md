# Analyze Make Handles

> 56 nodes · cohesion 0.08

## Key Concepts

- **OpenAIVLM** (31 connections) — `src/homesec/plugins/analyzers/openai.py`
- **test_openai_vlm.py** (19 connections) — `tests/homesec/test_openai_vlm.py`
- **_make_config()** (13 connections) — `tests/homesec/test_openai_vlm.py`
- **_make_vlm()** (12 connections) — `tests/homesec/test_openai_vlm.py`
- **.analyze()** (11 connections) — `src/homesec/plugins/analyzers/openai.py`
- **_make_filter_result()** (11 connections) — `tests/homesec/test_openai_vlm.py`
- **_create_test_video()** (9 connections) — `tests/homesec/test_openai_vlm.py`
- **test_analyze_respects_preprocessing_limits()** (9 connections) — `tests/homesec/test_openai_vlm.py`
- **test_analyze_extracts_frames_and_calls_api()** (8 connections) — `tests/homesec/test_openai_vlm.py`
- **test_shutdown_is_idempotent()** (8 connections) — `tests/homesec/test_openai_vlm.py`
- **_make_async_cm()** (7 connections) — `tests/homesec/test_openai_vlm.py`
- **_patch_session()** (7 connections) — `tests/homesec/test_openai_vlm.py`
- **test_analyze_handles_api_error()** (7 connections) — `tests/homesec/test_openai_vlm.py`
- **test_analyze_handles_malformed_json_response()** (7 connections) — `tests/homesec/test_openai_vlm.py`
- **test_analyze_handles_schema_mismatch()** (7 connections) — `tests/homesec/test_openai_vlm.py`
- **openai.py** (6 connections) — `src/homesec/plugins/analyzers/openai.py`
- **._extract_frames()** (5 connections) — `src/homesec/plugins/analyzers/openai.py`
- **_analysis_response()** (5 connections) — `tests/homesec/test_openai_vlm.py`
- **test_analyze_fails_after_shutdown()** (5 connections) — `tests/homesec/test_openai_vlm.py`
- **.__init__()** (4 connections) — `src/homesec/plugins/analyzers/openai.py`
- **test_analyze_raises_on_empty_video()** (4 connections) — `tests/homesec/test_openai_vlm.py`
- **_create_json_schema_format()** (3 connections) — `src/homesec/plugins/analyzers/openai.py`
- **_ensure_openai_dependencies()** (3 connections) — `src/homesec/plugins/analyzers/openai.py`
- **._build_payload()** (3 connections) — `src/homesec/plugins/analyzers/openai.py`
- **._call_api()** (3 connections) — `src/homesec/plugins/analyzers/openai.py`
- *... and 31 more nodes in this community*

## Relationships

- No strong cross-community connections detected

## Source Files

- `src/homesec/plugins/analyzers/openai.py`
- `tests/homesec/test_openai_vlm.py`

## Audit Trail

- EXTRACTED: 236 (93%)
- INFERRED: 18 (7%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*