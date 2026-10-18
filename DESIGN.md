# HomeSec Architecture and Design

This document explains implementation structure and design priorities. The maintained
[behavioral specs](openspec/specs/) describe public behavior; the
[reference and known limitations](docs/reference.md) distinguish observed behavior
from intended guarantees. Configuration and interface definitions below link to
canonical sources rather than duplicating their schemas.

## BIG NOTES (Read First)

1. **Postgres is the state store.** Workflow snapshots use the table `clip_states` with `clip_id TEXT PRIMARY KEY` plus a `data JSONB` column (schema evolves independently; include `schema_version` inside `data`). Lifecycle history is stored separately in `clip_events`. Persistence is best-effort; recording/upload are priorities during outages, not an end-to-end durability guarantee.
2. **Use Pydantic everywhere.** All boundaries (YAML config, DB `data` JSONB, VLM outputs, MQTT payloads) should be validated/serialized via Pydantic models so interfaces stay clean and types stay honest.
3. **Clips originate on the local filesystem.** Sources hand finalized local clips to the pipeline. Local files protect bytes from an immediate storage outage, but there is no durable whole-clip retry queue or automatic stage-resuming replay. See the [processing and retention limits](docs/reference.md#storage-retention).
4. **Single async flow per clip.** In-process handoff with bounded concurrency (global + per-stage `asyncio.Semaphore`). Callback-based ClipSource interface for extensibility.
5. **Pluggable object detection and VLM analysis.** Specific models (YOLO, GPT-4, etc.) are abstracted behind async plugin interfaces. Pipeline is model-agnostic; plugins manage their own resources (GPU, process pools, API clients).
6. **Errors as values.** Stage methods return `Result | StageError` instead of raising exceptions. This enables partial failures (e.g., upload fails but filter succeeds → still send alert). Stack traces are preserved in error objects. **Strict type checking required** (mypy --strict or pyright) to prevent runtime errors from missing `isinstance()` checks.
7. **Type-safe enums for domain values.** Use `StrEnum`/`IntEnum` from `models/enums.py` for type safety, IDE support, and maintainability. See [Type Safety & Enums](#type-safety--enums) below.

## Architecture Constraints (Separation of Concerns & Abstraction Boundaries)

**Separation of concerns**: each component has a single responsibility. The core pipeline orchestrates *what* happens; plugins and backends implement *how* it happens.  
**Abstraction boundaries**: the core depends only on stable contracts (interfaces, registries, repository APIs, and Pydantic models). Concrete implementations may depend on core abstractions, but the core must never depend on concrete implementations.

### Constraints (Non‑Negotiable)

1. **Dependency direction**: core modules (`pipeline`, `app`, `repository`, `models`) must not import concrete plugin/backends. Use interfaces and the existing typed registry loaders (for example, `load_filter()` and `load_source_plugin()`).
2. **Config boundary**: core sees only `backend` + opaque config payload. Plugin loaders validate/instantiate using plugin‑specific Pydantic config models. Core must not reference backend‑specific fields.
3. **Persistence boundary**: all state/event writes go through `ClipRepository`; never touch `StateStore`/`EventStore` directly.
4. **Plugin boundary**: plugins may import core interfaces/models; the core may not import plugin modules.
5. **Boundary validation**: external inputs (config, DB JSONB, VLM outputs, MQTT payloads) must be validated at the boundary with Pydantic before entering core logic.

Violations are architecture bugs and should be treated as such during reviews.

## Type Safety & Enums

Domain values that appear in multiple places use centralized enums in `models/enums.py`:

| Enum | Type | Purpose | Example |
|------|------|---------|---------|
| `EventType` | `StrEnum` | Clip lifecycle event types | `EventType.CLIP_RECORDED` |
| `ClipStatus` | `StrEnum` | Clip processing status | `ClipStatus.UPLOADED` |
| `RiskLevel` | `IntEnum` | VLM risk assessment (ordered) | `RiskLevel.HIGH > RiskLevel.LOW` |

### Benefits

- **Type safety**: Catch typos at compile time with mypy/pyright
- **IDE support**: Autocomplete and refactoring
- **Single source of truth**: No duplicate string literals
- **Natural comparison**: `RiskLevel` uses `IntEnum` for `>=` comparisons

### Usage Examples

```python
from homesec.models.enums import EventType, ClipStatus, RiskLevel

# Event types in event models
class ClipRecordedEvent(ClipEvent):
    event_type: Literal[EventType.CLIP_RECORDED] = EventType.CLIP_RECORDED

# Status comparisons
if state.status == ClipStatus.UPLOADED:
    ...

# Risk level comparison (IntEnum)
if analysis.risk_level >= RiskLevel.MEDIUM:
    send_alert()

# Parse from string (for config)
level = RiskLevel.from_string("high")  # Returns RiskLevel.HIGH
```

### Serialization

- `StrEnum` values serialize to their string value automatically
- Fields annotated with `RiskLevelField` use custom Pydantic serialization to
  retain risk names (`"medium"` rather than `1`).

## Plugin Architecture & Philosophy

For detailed implementation guides and how-tos, see **[PLUGIN_DEVELOPMENT.md](PLUGIN_DEVELOPMENT.md)**.

HomeSec employs a **Class-Based Plugin Architecture (V2)** designed for strict type safety, runtime validation, and clear separation of concerns. This section details the design patterns and philosophies driving the system.

### 1. Core Philosophy: "Configuration is the Contract"

In V1, plugins were factories accepting raw dictionaries and loose context objects. In V2, the **Configuration Model** (Pydantic) defines the entire contract for a plugin.

-   **Type Safety**: Every plugin defines a `config_cls` (Pydantic model). The registry validates raw JSON/YAML against this model *before* the plugin is instantiated.
-   **Backend/Config Boundary**: Core config models expose only `backend` + opaque `config` payloads. Plugin‑specific fields live in the plugin’s config model (defined alongside the implementation).
-   **Runtime Context (when needed)**: For plugin types that require runtime data (e.g., `camera_name` for sources, default alert policy `trigger_classes`), the loader injects those fields into the config dict *before* validation.
    -   *Benefit*: The plugin implementation doesn’t need a separate `context` argument; required runtime fields are validated alongside static config.
-   **Fail Fast**: Invalid configs cause a `ValidationError` at loading time, preventing partial start-ups.

### 2. The Unified Registry Pattern

Each plugin type has a registry backed by the same generic `PluginRegistry[ConfigT, PluginInterfaceT]` implementation in [registry.py](src/homesec/plugins/registry.py).

-   **Generics for Safety**: The registry is generic over the configuration type and the plugin interface.
    ```python
    from homesec.plugins.sources import load_source_plugin

    # The type-specific loader returns ClipSource; the generic load_plugin returns Any.
    source = load_source_plugin("my_source", config_dict, camera_name="front_door")
    ```
-   **Declarative Registration**: Plugins register themselves using the `@plugin` decorator, which captures metadata (name, type) and creates the association without manual mapping code.

### 3. Factory Pattern vs. Class Creation

We transitioned from functional factories (`make_source()`) to class-based factories (`Class.create()`).

-   **Encapsulation**: The class handles its own construction logic in `create()`, keeping `__init__` clean or free for dependency injection.
-   **State Management**: Plugins often hold state (database connections, ML model handles). Classes naturally encapsulate this state and provide lifecycle methods (`shutdown()`) to clean it up.

### 4. Decoupling & Local State

A key design validation was separating `LocalFolderSource` from the global `StateStore`.

-   **Philosophy**: Components should own their local truth.
-   **Implementation**: `LocalFolderSource` uses a local manifest
    (`.homesec_state.json`) to track emitted IDs independently of Postgres.
    Emission does not establish successful processing or eventual replay.

### 5. Error Handling Philosophy

-   **Contain expected failures**: Pipeline stage calls wrap exhausted plugin failures as error values. Unexpected exceptions can still terminate a clip task; its completion callback logs them.
-   **Errors as Values**: Where possible, return error objects/states rather than raising exceptions, allowing the pipeline to degrade gracefully (e.g., skip analysis but still upload).

## Design Goals

Build a reliable, pluggable pipeline to:
1. Detect motion from home cameras.
2. Record short video clips and store them in a pluggable backend (bundled: Dropbox and local filesystem).
3. Filter clips for configured target classes using a pluggable object detection plugin (bundled: YOLO, but swappable).
4. For clips that match configured trigger classes, run analysis via a pluggable VLM plugin (default: OpenAI-compatible API, but swappable) to summarize, classify risk, and set an alert level.
5. Notify via a pluggable/multiplexable notifier (MQTT → Home Assistant, email, SMS, etc.) when either:
   - `risk_level` meets or exceeds a configured threshold, OR
   - per-camera custom alert conditions match (even if `risk_level` is low) via `notify_on_activity_types`.

**Note:** Specific model implementations (YOLO weights, VLM providers) are outside the scope of this pipeline design—they are abstracted behind plugin interfaces.

### Non-Goals (Out of Scope for MVP)
- Extra VLM classifier passes (use `activity_type` from base VLM prompt instead)
- Post-upload Dropbox folder moves (destinations use the configured clips prefix,
  camera name, and filename)
- Public share links (use web URLs requiring Dropbox login)
- Multi-worker distributed processing (single process with async concurrency)

### Configuration Scope
- Filter and VLM defaults are global. Default alert-policy overrides live in its plugin config. `FilterOverrides` supports per-call filter settings, including maintenance rechecks; it is not a per-camera config block.

### Latency Target (Design Estimate)

These are illustrative design estimates, not measured latency guarantees.
- **ASAP after motion ends**: Target 16-28s for upload + filter + VLM → notification
- **Breakdown** (parallel upload + filter):
  - Upload: 2-5s (parallel with filter, not on critical path if filter+VLM > upload time)
  - Filter (YOLO): 1-3s (via process pool, non-blocking)
  - VLM: 15-25s (longest pole)
  - Total critical path: max(upload, filter + VLM) ≈ 16-28s
- Further optimization (e.g., model quantization, caching) is nice-to-have but not required for MVP

### Reliability via Partial Failures
- **Error-as-value pattern** enables pipeline to continue when non-critical stages fail
- Example: If Dropbox is down, upload fails but filter + VLM + notify still run → user gets alert with `view_url=null`
- This supports the recording/upload priority. It does not guarantee eventual upload or notification after retry exhaustion; see [current limitations](docs/reference.md#storage-retention).

## Current Building Blocks (Repo)

- Application/control plane: [app.py](src/homesec/app.py), the
  [API server](src/homesec/api/server.py), and
  [runtime manager](src/homesec/runtime/manager.py).
- Runtime assembly and shutdown: [assembly.py](src/homesec/runtime/assembly.py),
  [subprocess controller](src/homesec/runtime/subprocess_controller.py), and
  [runtime worker](src/homesec/runtime/worker.py).
- Clip processing: [pipeline/core.py](src/homesec/pipeline/core.py).
- Sources: [RTSP](src/homesec/sources/rtsp/core.py),
  [FTP](src/homesec/sources/ftp.py), and
  [local folder](src/homesec/sources/local_folder.py).
- Reference plugins: [YOLO](src/homesec/plugins/filters/yolo.py),
  [OpenAI-compatible analyzer](src/homesec/plugins/analyzers/openai.py),
  [storage](src/homesec/plugins/storage/), and
  [notifiers](src/homesec/plugins/notifiers/).
- Persistence: [ClipRepository](src/homesec/repository/clip_repository.py),
  [Postgres stores](src/homesec/state/postgres.py), and [Alembic](alembic/).
- Telemetry: [logging setup](src/homesec/logging_setup.py) and
  [database log handler](src/homesec/telemetry/db_log_handler.py).
- Live media: source/runtime boundaries supervise preview and talk. Native helper
  build and deployment details live in the [WebRTC guide](docs/webrtc-preview.md)
  and [talk guide](docs/push-to-talk.md).

The root-level legacy scripts (`src/motion_recorder.py`,
`src/ftp_dropbox_server.py`, `src/human_filter/`, `src/evals/`) are not references
for new development.

## Pipeline Architecture

**Conceptual flow**

`ClipSource` → (parallel: `StorageBackend` + `ObjectFilter`) → `VLMAnalyzer` → `AlertPolicy` → `Notifier(s)`

### Components

1. **Clip Source (pluggable ingest + recorder)**
   - Produces finalized local `Clip` values and invokes the registered callback.
     The pipeline owns upload; sources do not upload through `StorageBackend`.
   - Current sources are RTSP motion recording, FTP ingest, and local-folder
     watching. Source-specific readiness and heartbeat checks describe receipt
     capability, independently of whether motion is occurring.
   - Sources use filename stems as clip IDs. IDs must be distinct across cameras
     and previous records; ingestion does not enforce that precondition. A reused
     ID can overwrite prior state. See [identity limits](docs/reference.md#capture-recording).

2. **Storage Backend (pluggable)**
   - Current built-ins are Dropbox and local filesystem storage. `put_file()`
     returns `StorageUploadResult` with storage URI and optional view URL.
   - The pipeline constructs destinations through
     [storage_paths.py](src/homesec/storage_paths.py); configurable path prefixes
     live under `storage.paths`. Classification does not move uploaded clips.

3. **State Store (Postgres) and Local Files**
   - `clip_states` holds a typed JSON snapshot per clip ID; `clip_events` holds
     lifecycle history. Use `ClipRepository` for coordinated access.
   - Repository state writes and event appends are independent operations with
     bounded retries. They are not a shared transaction. Events may be missing
     after state advances, and concurrent whole-snapshot writes can lose fields
     or status even when each write succeeds.
   - Existing local files and source manifests are not a universal durable work
     queue. No generic startup stage-resume or missing-file redownload path is
     implemented for the processing pipeline.
   - Missing DSN configuration can prevent startup; a configured but unavailable
     Postgres server is handled best-effort. Read [consistency and recovery
     limitations](docs/reference.md#analysis-alerts) before relying on state/history.

4. **Object Detection Filter (Pluggable)**
   - Uses pluggable object detection to decide "needs VLM".
   - Plugin interface: `ObjectFilter.detect(video_path, overrides=None) -> FilterResult`
   - Reference implementation: YOLO with configurable classes (default: person), frame sampling, early exit on detection.
   - Output: `FilterResult` (detected_classes, confidence, sampled_frames, model).
   - VLM trigger classes are configured globally (default: person only) based on `filter_result.detected_classes`.

5. **VLM Analyzer (Pluggable)**
   - Consumes clips (depending on `vlm.run_mode`) and produces structured analysis.
   - Plugin interface: `VLMAnalyzer.analyze(video_path, filter_result, config) -> AnalysisResult`
   - Reference implementation: OpenAI-compatible API (GPT-4o or similar).
   - Trigger rule:
     - `run_mode: always` → run for every clip (after filter succeeds)
     - `run_mode: trigger_only` → run if `filter_result.detected_classes` intersects `vlm.trigger_classes`
     - `run_mode: never` → skip VLM
   - Output: `AnalysisResult` (risk level, activity type, summary, optional structured `analysis`, and token counts). Timelines and `requires_review` live inside `SequenceAnalysis`.
   - Results are recorded in workflow state/history; the pipeline does not export an `analysis.json` sidecar.
   - Prompting is configured globally (same for all cameras); per-camera alert policy decides which `activity_type` values trigger notifications.

6. **Alert Policy (per camera)**
   - Decides whether to notify based on VLM output plus per-camera overrides.
   - Bundled default-policy logic:
     - Notify when risk is at least `min_risk_level`, or a configured activity matches; the default threshold is `medium`.
     - Additionally: allow per-camera `notify_on_activity_types` list that can trigger notifications even when `risk_level` is low.
     - Optional per-camera `notify_on_motion` selects notifications without
       requiring analysis. It does not bypass pipeline filter failure.
     - Without analysis, matching filter trigger classes can still select a
       notification, including when the VLM was skipped or failed. Otherwise
       `notify_on_motion=false` yields no notification.
     - Notifications wait for the upload outcome so a `view_url` can be included.
   - Configuration:
     - `config`: `min_risk_level`, `notify_on_activity_types`, and
       `notify_on_motion`, plus `overrides` keyed by camera name.
   - Example use cases:
     - With the bundled analyzer, `normal_delivery` can select a low-risk
       delivery notification. Use activity values produced by the selected
       analyzer; an activity-list match bypasses the risk threshold.

7. **Notifier (pluggable + multiplexable)**
   - Behavior: send selected alerts to all enabled configured notification sinks.
   - Primary integration target: MQTT via Mosquitto → Home Assistant.
   - Bundled MQTT defaults and payload:
     - Topic: `homecam/alerts/{camera_name}` (per-camera topics)
     - QoS: 1, retained: false
     - Payload (JSON): `clip_id`, `camera_name`, `storage_uri`, `view_url`, `risk_level`, `activity_type`, `notify_reason`, `summary`, `ts`, `dedupe_key`, `upload_failed`, `vlm_failed`, and optional `analysis`
   - View URL: compute a Dropbox web URL for the clip (requires login; not a share link).
   - **Partial failure handling**: A policy-selected notification can still be
     attempted after upload failure with `view_url=null` and `upload_failed=true`.
     User can check local recordings.
   - `dedupe_key` equals `clip_id`; this labels notifications but does not provide
     cross-run or recipient-side duplicate suppression.

### Review UX

The React UI provides setup, camera configuration, event browsing, playback, and
live media controls. MQTT/Home Assistant notifications and authenticated Dropbox
view URLs are additional review surfaces. See the [UI guide](ui/README.md) and
[events/playback contract](openspec/specs/events-playback/spec.md).

### Core Interfaces (Python)

[interfaces.py](src/homesec/interfaces.py) defines the abstract plugin interfaces;
[errors.py](src/homesec/errors.py) defines pipeline error values. Use those files
for exact method signatures and [PLUGIN_DEVELOPMENT.md](PLUGIN_DEVELOPMENT.md)
for extension examples.

- `ClipSource.start()` and managed-component `shutdown(timeout=None)` are async.
- Storage uses `put_file`, `get_view_url`, `get`, `exists`, `delete`, `ping`,
  and `shutdown`; there is no `put_json()` or `close()` interface method.
- `AlertPolicy` is synchronous and has no shutdown/ping requirement.
- Plugins raise operational errors; pipeline stage adapters apply bounded retries
  and convert exhausted failures to stage error values that retain their cause.

## Configuration

Prefer a single YAML file for non-secret configuration (cameras, sources, policies, MQTT). Keep secrets in `.env` and reference them by env var name from YAML.

### Configuration Hierarchy & Overrides

Per-camera alert overrides live inside `alert_policy.config.overrides` and are merged by
the alert policy implementation (not the core config). The core remains backend-agnostic.

**Example (default alert policy):**
```yaml
alert_policy:
  backend: default
  enabled: true
  config:
    min_risk_level: medium
    notify_on_motion: false
    overrides:
      front_door:
        min_risk_level: low
        notify_on_activity_types: [normal_delivery]
```

### Config Structure

Use [config/example.yaml](config/example.yaml), the
[core Pydantic models](src/homesec/models/config.py), and the selected plugin's
`config_cls` for current fields/defaults. The top-level configuration includes
`cameras`, `storage`, `state_store`, `notifiers`, `filter`, `vlm`, `alert_policy`,
`retention`, `maintenance`, `concurrency`, `retry`, `server`, `preview`, and `talk`.
MQTT settings belong inside a `notifiers` entry with `backend: mqtt`, not a
standalone `mqtt` section. `retry` has `max_attempts` and `backoff_s`; there is no
`on_exhausted` setting.

Secrets should be supplied through environment-variable references. Follow the [configuration reference](docs/reference.md#setup-configuration) for
saved versus active config, secret handling, and Save/Apply behavior. Plugin
validation does not imply construction or successful provider calls; setup
validation-only has narrower checks than applying a configuration.

## Data Model (Clip-Centric)

- [Clip](src/homesec/models/clip.py) describes a finalized local file, camera,
  source, and timing. Sources use its filename stem as `clip_id`.
- `ClipStateData` is a current snapshot with status, local/storage pointers,
  filter result, analysis result, and alert decision. It does not contain a
  per-stage attempt log; lifecycle event models hold attempt and duration data.
- [Events](src/homesec/models/events.py) describe processing history, separately
  from the state snapshot. History is best-effort and may be incomplete.
- [FilterResult](src/homesec/models/filter.py),
  [AnalysisResult and SequenceAnalysis](src/homesec/models/vlm.py), and
  [Alert/AlertDecision](src/homesec/models/alert.py) are the canonical result and
  notification payload definitions.
- [Enums](src/homesec/models/enums.py) define statuses, lifecycle event types,
  VLM run modes, and ordered risk levels. Fields annotated with `RiskLevelField`
  validate risk strings and serialize risk names.

## Production Requirements

### Product Priorities (Use-Case Driven)

1. **P0 — Never miss new clips:** keep recording and spooling locally even if Dropbox or Postgres is down.
2. **P0 — Upload ASAP:** upload to Dropbox best-effort as soon as possible; do not block recording.
3. **P1 — Analyze + notify:** run filter/VLM and send MQTT notifications best-effort; duplicates are acceptable during outages.
4. **P2 — State consistency:** prefer best-effort correctness over strict exactly-once behavior; state may be incomplete when dependencies are unavailable.

### Reliability & Correctness
- **Best-effort idempotency:** avoid duplicate uploads/alerts when possible; duplicates are acceptable during outages.
- **Bounded stage retries:** live processing retries configured stage failures. There is no end-to-end at-least-once delivery guarantee or automatic whole-clip replay.
- **Concurrency:** semaphores bound active clip processing and stage work. Clip callbacks can still accumulate waiting tasks; the global limit is not a bounded admission queue.
- **Graceful DB degradation:** with valid startup configuration, recording/uploading and processing remain best-effort during a database outage. State/history may be missing; there is no local persistence fallback for workflow metadata.
- **Offline tolerance:** sources finalize clips locally and live upload attempts retry. Retry exhaustion leaves local bytes without guaranteeing a later upload; retention and operator recovery must account for protected/orphan files.
- **Retention:** local cleanup is independent of notify and only deletes clips that are both uploaded and finalized (`storage_uri` present and `status == DONE`).
  - Attempt oldest-first pruning of eligible local clips against
    `retention.max_local_size_bytes`. Protected files and incomplete state can
    leave total usage over the cap; it is not a hard disk bound.
  - Trigger prune on clip arrival (`reason=clip_arrived`) with single-flight `drop` while a prune pass is active.
  - Retention path is local-only: no storage-backend delete calls and no retention-driven DB cleanup/mutation writes.

### Performance
- Motion detection should run in real time per camera; recording should not block on upload.
- Object detection should run faster than real time (sampling) and short-circuit on first match.
- **Plugin implementations**:
  - CPU/GPU-bound plugins (e.g., YOLO) should use `ProcessPoolExecutor` internally to avoid blocking the event loop.
  - I/O-bound plugins (e.g., OpenAI API) should use async HTTP clients.
  - Plugins manage their own resources (process pools, GPU allocation, API rate limits).
- VLM is the expensive step; in `run_mode: trigger_only`, only run it when `filter_result.detected_classes` intersects `vlm.trigger_classes`.
- Prefer local processing when available to minimize Dropbox transfers, while still uploading clips ASAP.

### Dropbox URLs (No Share Links)
- The Dropbox storage backend should derive `view_url` from its configured `web_url_prefix` and the uploaded path (requires Dropbox login; not public).
  - Example: `view_url = "{web_url_prefix}{dropbox_path}"` where `dropbox_path` is the path inside the Dropbox root.
- The current backend builds this URL from its destination URI; it does not use the SDK response's `path_display` or make a separate URL lookup.
- URLs are stable as long as files are not moved; MVP assumes no post-upload moves.

### Runtime and Clip Lifecycle

**Control flow:**
- `ClipPipeline` runs async work per clip within its runtime. The API/control plane manages active runtimes through a controller; the managed runtime can run in a subprocess.
- ClipSource implementations (RTSP recorder, FTP server) register callbacks via `register_callback(fn)`
- On new clip: callback invokes `pipeline.on_new_clip(clip)` to schedule tracked processing work
- Each clip flows through:
  - **Parallel**: upload + filter (start simultaneously to reduce latency)
  - **Sequential**: VLM (conditional, after filter completes) → alert → notify (after upload completes for `view_url`)
- Bounded concurrency per stage via `asyncio.Semaphore` (configurable limits) for I/O-bound stages
- **Plugins manage their own concurrency**: CPU/GPU-bound plugins (YOLO) manage `ProcessPoolExecutor` internally; I/O-bound plugins (OpenAI API) use async HTTP clients
- Pipeline is model-agnostic; it just awaits plugin async methods

**Startup and recovery:**
- Runtime assembly loads plugins, creates the pipeline, registers source callbacks,
  and starts sources. Startup cleanup tears down partially built bundles.
- `LocalFolderSource` scans stable files and persists emitted IDs in its local
  manifest. It marks emission, not processing completion; a later failure does
  not automatically cause replay on restart. FTP and RTSP have their own source
  lifecycle. There is no generic "scan all recordings and resume last stage"
  recovery loop.
- Saving configuration and activating it are separate operations. Apply/reload
  uses runtime generations and can fail independently of a successful save;
  see [setup/configuration](openspec/specs/setup-configuration/spec.md).

**Shutdown ownership:**
- The runtime stops sources, waits for tracked pipeline/retention work, and then
  shuts down filter, analyzer, and notifier resources with bounded waits.
- `ClipPipeline.shutdown()` waits for its own work; it does not own plugin or
  source shutdown. The application closes the control plane, runtime manager,
  maintenance manager, and its own persistence/storage stack. A subprocess
  runtime worker separately builds and closes its own persistence/storage stack.
- Shutdown is not a durable-drain guarantee. Review source/device and cancellation
  limitations in the [reference](docs/reference.md).

**Error handling and retry:**
- Upload, filter, analysis, and each selected notifier use configured bounded
  retries with exponential delays `backoff_s * 2 ** (attempt - 1)`.
- Attempt events are written through the repository. Exhausted filter failure
  ends analysis/notification for that clip after collecting the upload outcome;
  upload/VLM/notifier failures can leave partial results.
- No automatic `homecam/system/errors` notification is emitted on exhaustion.
  Errors are logged and recorded best-effort. Preparation/orchestration errors
  outside retry containment can still abort a clip task.
- Repository retries are separate and apply to retryable database exceptions;
  they do not make state/event operations atomic or repair competing snapshots.

### Observability
- Regular Python logs use structured extras such as `camera_name`, `recording_id`, and stable `event_type`; see [logging_setup.py](src/homesec/logging_setup.py) and the [telemetry handler](src/homesec/telemetry/db_log_handler.py). Telemetry logging is separate from repository lifecycle history.
- Useful operational measures include:
  - clip count/day, upload failures, filter latency, VLM latency/cost, alert count.
- `DB_DSN` enables database telemetry logging. Workflow persistence resolves `state_store.dsn_env` (or `dsn`) independently; using the same database is a configuration choice. Both paths are best-effort during database failures.

### Health Check Endpoints
- FastAPI is the only HTTP listener.
- Public probe endpoints:
  - `GET /health`: unauthenticated ops probe endpoint.
  - `GET /api/v1/health`: versioned health summary endpoint.
- Authenticated endpoint:
  - `GET /api/v1/diagnostics`: versioned detailed diagnostics endpoint (API key required when auth is enabled).
- `/health` and `/api/v1/health` derive overall status from pipeline availability and repository ping, and report the source health count separately. Unhealthy cameras or storage do not alone degrade this summary. `/api/v1/diagnostics` additionally pings storage and considers enabled-camera health. A healthy probe does not establish recording, upload, or notification success.
- Non-health API endpoints may require API key auth when enabled.

**Probe integration example:**
```yaml
binary_sensor:
  - platform: rest
    name: "Camera Pipeline Health"
    resource: "http://pipeline-host:8081/health"
    value_template: "{{ value_json.status == 'healthy' }}"
    scan_interval: 60
```

### Testing Strategy
- **Unit tests**: Individual stages (filter, VLM, alert policy) with mocked dependencies
- **Integration tests**: Component/pipeline tests use boundary fakes; Postgres store/API tests use isolated schemas on a real test database. Current CI does not run a real MQTT broker for the full suite.
- **Manual/provider/device validation**: Real camera streams and real storage/MQTT are separate from fake-boundary tests and require an appropriate test environment.
- **Config validation**: Exercise model/plugin/reference validation and saved versus active state. Structural validity or setup probe success does not prove runtime/provider conformance.

### Type Safety (Critical for Error-as-Value Pattern)
- **Strict type checking required**: Use `mypy --strict` or `pyright` to enforce type safety
- **Why critical**: Error-as-value pattern (`Result | StageError`) requires explicit type narrowing (match or isinstance)
- **Without strict checking**: Runtime errors from accessing `.detected_classes` on an error object
- **Preferred: Python 3.10+ match statements** (cleaner than isinstance):
  ```python
  match await self._filter_stage(clip):
      case FilterError() as err:
          logger.error(err.cause)
          return
      case FilterResult() as result:
          filter_result = result  # Type narrowing: result is FilterResult
  ```
- **Alternative: isinstance checks** (if match not preferred):
  ```python
  filter_res = await self._filter_stage(clip)
  if isinstance(filter_res, FilterError):
      logger.error(filter_res.cause)
      return
  filter_result: FilterResult = filter_res  # Type narrowing
  ```
- **CI/CD integration**: `make typecheck` checks the `homesec` package and native build helper with strict mypy. See [CI.md](CI.md) for the complete gate.

### Security & Privacy
- Never commit secrets (`.env`, tokens, RTSP creds).
- Encrypt storage where possible; limit who can access raw clips.
- Redact sensitive data from logs (RTSP URLs, tokens).
- Treat VLM output as advisory. `SequenceAnalysis.requires_review` is analysis data; the pipeline does not implement a separate human-review queue.

## Deployment and Further Work

[README.md](README.md) covers installation and deployment;
[CI.md](CI.md) describes the current validation graph;
[TESTING.md](TESTING.md) describes safe test setup and behavioral testing.
The control plane manages runtime selection and reload, while source/runtime
boundaries own camera media lifecycles. See the source modules above for exact
process ownership rather than assuming every component runs in the API process.

The old MVP implementation checklist has been superseded by the implementation
and maintained behavioral specs. Current consistency, recovery, deletion,
retention, media, and provider limitations are tracked in
[docs/reference.md](docs/reference.md) and its linked follow-up work. External
plugin discovery is already supported through the `homesec.plugins` entry-point
group; it is not a proposed future feature.
