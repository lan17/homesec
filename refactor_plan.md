# HomeSec Abstraction Boundary Refactor Plan

Date: 2026-01-31  
Scope: enforce “backend + config” across plugins, remove core knowledge of concrete backends, and keep VLM gating in core without alert-policy coupling.

## Goals
- Make core config and orchestration fully backend-agnostic.
- Standardize on a single schema: `backend` + `config` (and only truly generic fields at top level).
- Remove built-in plugin config models from core; plugins own their config models.
- Keep VLM gating as core logic, driven by core VLM policy (not alert policy config).
- Make alert policy remain pluggable without core knowing about the default policy’s config shape.

## Decisions (confirmed)
- Breaking changes are OK. No shims or backward-compat.
- Standardize selectors to `backend` everywhere (`filter.backend`, `camera.source.backend`, etc.).
- Move `per_camera_alert` into `alert_policy.config.overrides`.
- Remove `filter.max_workers` / `vlm.max_workers` from core config; rely on `concurrency.*`.
- VLM execution policy lives in core: `vlm.run_mode = trigger_only | always | never`.
- Keep `alert_policy.enabled` as a first-class switch (no requirement to use `backend: noop`).
- Plugin config models live in the same module as the plugin implementation.
- Rename `Clip.source_type` → `Clip.source_backend` for naming consistency.
- Cleanup CLI remains YOLO-specific for now; only adjust for schema renames.

## New Config Shape (illustrative)
```yaml
storage:
  backend: dropbox
  config:
    root: "/homecam"
    token_env: DROPBOX_TOKEN
  paths:
    clips_dir: "clips"
    backups_dir: "backups"
    artifacts_dir: "artifacts"

filter:
  backend: yolo
  config:
    classes: ["person", "car"]
    min_confidence: 0.6

vlm:
  backend: openai
  trigger_classes: ["person"]
  run_mode: "trigger_only"   # trigger_only | always | never
  config:
    api_key_env: OPENAI_API_KEY
    model: "gpt-4o"
  preprocessing:
    max_frames: 10
    max_size: 1024
    quality: 85

alert_policy:
  backend: default
  enabled: true
  config:
    min_risk_level: "medium"
    notify_on_motion: false
    overrides:
      front_door:
        min_risk_level: "low"

notifiers:
  - backend: mqtt
    enabled: true
    config:
      host: "localhost"

cameras:
  - name: front_door
    source:
      backend: rtsp
      config: { ... }
```

## Refactor Steps

### Phase 1 — Core Config Model Cleanup
1. **Rename selector fields to `backend`**:
   - `filter.plugin` → `filter.backend`
   - `camera.source.type` → `camera.source.backend`
   - keep `storage.backend`, `vlm.backend`, `alert_policy.backend`, `notifiers[].backend`
2. **Replace backend-specific top-level keys with `config`:**
   - `storage.dropbox`/`storage.local` → `storage.config`
   - `vlm.llm` → `vlm.config`
3. **Move per-camera overrides into alert policy config:**
   - remove `Config.per_camera_alert`
   - add `DefaultAlertPolicySettings.overrides` as a real field (no exclude) inside config.
4. **Remove built-in plugin config validation from core:**
   - delete `Config._validate_builtin_plugin_configs()`
   - delete `AlertPolicyConfig._validate_alert_policy()`
   - delete `StorageConfig._validate_builtin_backends()`
   - remove built-in source config parsing in `CameraSourceConfig`
5. **Remove core-owned plugin config models:**
   - remove `DropboxStorageConfig`, `LocalStorageConfig`, `MQTTConfig`, `SendGridEmailConfig`,
     `RTSPSourceConfig`, `LocalFolderSourceConfig`, `FtpSourceConfig`, `OpenAILLMConfig`,
     `YoloFilterSettings` from `src/homesec/models/config.py` and `src/homesec/models/__init__.py`.

### Phase 2 — Plugin Config Ownership
6. **Move plugin config models into plugin modules:**
   - `plugins/storage/dropbox.py`: define `DropboxStorageConfig`
   - `plugins/storage/local.py`: define `LocalStorageConfig`
   - `plugins/notifiers/mqtt.py`: define `MQTTConfig`
   - `plugins/notifiers/sendgrid_email.py`: define `SendGridEmailConfig`
   - `plugins/filters/yolo.py`: define `YoloFilterConfig` (rename from `YoloFilterSettings`)
   - `plugins/analyzers/openai.py`: define `OpenAIConfig` (rename from `OpenAILLMConfig`)
   - `plugins/sources/rtsp.py`: define `RTSPSourceConfig`
   - `plugins/sources/local_folder.py`: define `LocalFolderSourceConfig`
   - `plugins/sources/ftp.py`: define `FtpSourceConfig`
7. **Update plugin `config_cls` references** to point at new local config classes.
8. **Remove any plugin imports of core config models** and replace with local config models.

### Phase 3 — Registry Validation Without Instantiation
9. **Add registry validation helpers** (no instantiation):
   - `PluginRegistry.validate(name, config_dict, **ctx)` → returns validated config model
   - `validate_plugin_configs(config: Config)` in `homesec/config/validation.py`
     that calls registry.validate for:
       - storage backend
       - filter backend
       - vlm backend
       - alert_policy backend
       - notifiers list
       - camera sources
10. **Application + CLI flow (fail fast, after discovery):**
    - `discover_all_plugins()`
    - `validate_plugin_names(...)`
    - `validate_plugin_configs(config)` (new)
    - then instantiate plugins.
    - `load_config()` stays structural; plugin validation happens after discovery.

### Phase 4 — Core Logic Adjustments
11. **VLM gating controlled by core config:**
    - Add `VLMRunMode` enum (or str literal) in core models.
    - Update `ClipPipeline._should_run_vlm()` to use `vlm.run_mode` only.
    - Remove any dependency on `alert_policy.backend` inside pipeline.
12. **Alert policy overrides:**
    - Update `DefaultAlertPolicy` to read `overrides` from its config model.
    - Remove `Config.get_default_alert_policy()` and any use of `per_camera_alert`.
13. **Notifier multiplexing:**
    - Move `NotifierEntry`/`MultiplexNotifier` to a core-neutral module
      (`homesec/notifiers/multiplex.py`) and update imports in app/pipeline.

### Phase 5 — Update Docs & Examples
14. **README + config/example.yaml**: convert to new schema (`backend + config` and `source.backend`).
15. **DESIGN.md**: add a “Config Shape” subsection under Architecture Constraints.
16. **PLUGIN_DEVELOPMENT.md**: update examples to show config_cls in plugin modules and new schema.

### Phase 6 — Tests & Cleanup
17. **Update tests for new config schema** (especially any fixtures loading YAML).
18. **Remove obsolete exports** from `homesec.models.__all__`.
19. **Typecheck + tests**: run `make typecheck` and `make test`.

## Expected Touch Points (non-exhaustive)
- Core config: `src/homesec/models/config.py`, `src/homesec/models/__init__.py`
- Plugin registry: `src/homesec/plugins/registry.py`
- App wiring: `src/homesec/app.py`
- Pipeline: `src/homesec/pipeline/core.py`
- Alert policy: `src/homesec/plugins/alert_policies/default.py`
- Storage loader: `src/homesec/plugins/storage/__init__.py`
- Clip model: `src/homesec/models/clip.py` (rename `source_type` → `source_backend`)
- Example config: `config/example.yaml`
- Docs: `README.md`, `DESIGN.md`, `PLUGIN_DEVELOPMENT.md`

## Risks / Notes
- This is a breaking config change. All existing YAMLs must be updated manually.
- Plugin-specific config models moving out of core may require import updates in tests.
- Registry validation should be added before instantiation to keep “fail fast” behavior.
- `VLMPreprocessConfig` stays core since it’s generic to all VLM analyzers.

## Open Questions (if needed later)
- Should we formalize a `ConfigPlugin` protocol for tooling or CLI validation?
- Do we want a first-class “multiplex” notifier plugin rather than core assembly?
