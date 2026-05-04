# Filterresult Basemodel Fastapiserverconfig

> 236 nodes · cohesion 0.03

## Key Concepts

- **FilterResult** (129 connections) — `src/homesec/models/filter.py`
- **BaseModel** (108 connections)
- **FastAPIServerConfig** (93 connections) — `src/homesec/models/config.py`
- **VLMConfig** (93 connections) — `src/homesec/models/vlm.py`
- **FilterConfig** (76 connections) — `src/homesec/models/filter.py`
- **AnalysisResult** (75 connections) — `src/homesec/models/vlm.py`
- **CameraConfig** (70 connections) — `src/homesec/models/config.py`
- **AlertPolicyConfig** (64 connections) — `src/homesec/models/config.py`
- **StateStoreConfig** (63 connections) — `src/homesec/models/config.py`
- **NotifierConfig** (62 connections) — `src/homesec/models/config.py`
- **StorageConfig** (61 connections) — `src/homesec/models/config.py`
- **CameraSourceConfig** (60 connections) — `src/homesec/models/config.py`
- **OpenAIConfig** (52 connections) — `src/homesec/plugins/analyzers/openai.py`
- **YoloFilterConfig** (47 connections) — `src/homesec/plugins/filters/yolo.py`
- **_StubVLM** (31 connections) — `tests/homesec/test_runtime_manager.py`
- **RiskLevel** (31 connections) — `src/homesec/models/enums.py`
- **_StubNotifier** (29 connections) — `tests/homesec/test_runtime_assembly.py`
- **_DisconnectingWriter** (29 connections) — `tests/homesec/test_runtime_worker.py`
- **SetupTestConnectionRequestError** (28 connections) — `src/homesec/services/setup.py`
- **_StubSetupApp** (27 connections) — `tests/homesec/test_api_setup_routes.py`
- **_StubFilter** (27 connections) — `tests/homesec/test_runtime_assembly.py`
- **SetupFinalizeValidationError** (27 connections) — `src/homesec/services/setup.py`
- **_PingablePlugin** (26 connections) — `src/homesec/services/setup.py`
- **config.py** (26 connections) — `src/homesec/models/config.py`
- **PipelineMocks** (25 connections) — `tests/homesec/test_pipeline.py`
- *... and 211 more nodes in this community*

## Relationships

- No strong cross-community connections detected

## Source Files

- `src/homesec/errors.py`
- `src/homesec/maintenance/postgres_backup.py`
- `src/homesec/models/config.py`
- `src/homesec/models/enums.py`
- `src/homesec/models/filter.py`
- `src/homesec/models/setup.py`
- `src/homesec/models/vlm.py`
- `src/homesec/plugins/alert_policies/default.py`
- `src/homesec/plugins/analyzers/openai.py`
- `src/homesec/plugins/filters/yolo.py`
- `src/homesec/services/setup.py`
- `tests/homesec/test_api_routes.py`
- `tests/homesec/test_api_setup_routes.py`
- `tests/homesec/test_openai_vlm.py`
- `tests/homesec/test_pipeline.py`
- `tests/homesec/test_pipeline_events.py`
- `tests/homesec/test_runtime_assembly.py`
- `tests/homesec/test_runtime_manager.py`
- `tests/homesec/test_runtime_worker.py`

## Audit Trail

- EXTRACTED: 752 (30%)
- INFERRED: 1786 (70%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*