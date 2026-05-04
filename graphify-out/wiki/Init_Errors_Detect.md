# Init Errors Detect

> 18 nodes · cohesion 0.15

## Key Concepts

- **FilterError** (9 connections) — `src/homesec/errors.py`
- **UploadError** (9 connections) — `src/homesec/errors.py`
- **PipelineError** (8 connections) — `src/homesec/errors.py`
- **VLMError** (8 connections) — `src/homesec/errors.py`
- **errors.py** (6 connections) — `src/homesec/errors.py`
- **.__init__()** (5 connections) — `src/homesec/errors.py`
- **.detect()** (3 connections) — `tests/homesec/mocks/filter.py`
- **.analyze()** (3 connections) — `tests/homesec/mocks/vlm.py`
- **.__init__()** (2 connections) — `src/homesec/errors.py`
- **.__init__()** (2 connections) — `src/homesec/errors.py`
- **.__init__()** (2 connections) — `src/homesec/errors.py`
- **.__init__()** (2 connections) — `src/homesec/errors.py`
- **Error hierarchy for HomeSec pipeline stages.** (1 connections) — `src/homesec/errors.py`
- **Storage upload failed.** (1 connections) — `src/homesec/errors.py`
- **Object detection filter failed.** (1 connections) — `src/homesec/errors.py`
- **Base exception for all pipeline errors.      Compatible with error-as-value patt** (1 connections) — `src/homesec/errors.py`
- **Detect objects in video clip (mock implementation).** (1 connections) — `tests/homesec/mocks/filter.py`
- **Analyze clip and produce structured assessment (mock implementation).** (1 connections) — `tests/homesec/mocks/vlm.py`

## Relationships

- No strong cross-community connections detected

## Source Files

- `src/homesec/errors.py`
- `tests/homesec/mocks/filter.py`
- `tests/homesec/mocks/vlm.py`

## Audit Trail

- EXTRACTED: 48 (74%)
- INFERRED: 17 (26%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*