# Home Camera Pipeline — Architecture & Production Requirements

## BIG NOTES (Read First)

1. **Postgres is the state store.** Use a single workflow/state table `clip_states` with `clip_id TEXT PRIMARY KEY` plus a `data JSONB` column (schema evolves independently; include `schema_version` inside `data`). The pipeline continues if Postgres is down; state is best-effort.
2. **Use Pydantic everywhere.** All boundaries (YAML config, DB `data` JSONB, VLM outputs, MQTT payloads) should be validated/serialized via Pydantic models so interfaces stay clean and types stay honest.
3. **Local filesystem is the canonical queue.** Clip bytes live locally first (e.g., `recordings/` / `ftp_incoming/`) so DB outages or Dropbox outages don't lose data; Dropbox is the configured storage surface and classification is logical in `clip_states`.
4. **Single async flow per clip.** In-process handoff with bounded concurrency (global + per-stage `asyncio.Semaphore`). Callback-based ClipSource interface for extensibility.
5. **Pluggable object detection and VLM analysis.** Specific models (YOLO, GPT-4, etc.) are abstracted behind async plugin interfaces. Pipeline is model-agnostic; plugins manage their own resources (GPU, process pools, API clients).
6. **Errors as values.** Stage methods return `Result | StageError` instead of raising exceptions. This enables partial failures (e.g., upload fails but filter succeeds → still send alert). Stack traces are preserved in error objects. **Strict type checking required** (mypy --strict or pyright) to prevent runtime errors from missing `isinstance()` checks.
7. **Type-safe enums for domain values.** Use `StrEnum`/`IntEnum` from `models/enums.py` for type safety, IDE support, and maintainability. See [Type Safety & Enums](#type-safety--enums) below.

## Architecture Constraints (Separation of Concerns & Abstraction Boundaries)

**Separation of concerns**: each component has a single responsibility. The core pipeline orchestrates *what* happens; plugins and backends implement *how* it happens.  
**Abstraction boundaries**: the core depends only on stable contracts (interfaces, registries, repository APIs, and Pydantic models). Concrete implementations may depend on core abstractions, but the core must never depend on concrete implementations.

### Constraints (Non‑Negotiable)

1. **Dependency direction**: core modules (`pipeline`, `app`, `repository`, `models`) must not import concrete plugin/backends. Use interfaces + registry loaders only (e.g., `load_*_plugin()`).
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
- `RiskLevel` (IntEnum) uses custom Pydantic serialization to maintain string representation in configs (`"medium"` not `1`)

## Plugin Architecture & Philosophy

For detailed implementation guides and how-tos, see **[PLUGIN_DEVELOPMENT.md](PLUGIN_DEVELOPMENT.md)**.

HomeSec employs a **Class-Based Plugin Architecture (V2)** designed for strict type safety, runtime validation, and clear separation of concerns. This section details the design patterns and philosophies driving the system.

### 1. Core Philosophy: "Configuration is the Contract"

In V1, plugins were factories accepting raw dictionaries and loose context objects. In V2, the **Configuration Model** (Pydantic) defines the entire contract for a plugin.

-   **Type Safety**: Every plugin defines a `config_cls` (Pydantic model). The registry validates raw JSON/YAML against this model *before* the plugin is instantiated.
-   **Backend/Config Boundary**: Core config models expose only `backend` + opaque `config` payloads. Plugin‑specific fields live in the plugin’s config model (defined alongside the implementation).
-   **Runtime Context (when needed)**: For plugin types that require runtime data (e.g., `camera_name`/timezone for sources, default alert policy `trigger_classes`), the loader injects those fields into the config dict *before* validation.
    -   *Benefit*: The plugin implementation doesn’t need a separate `context` argument; required runtime fields are validated alongside static config.
-   **Fail Fast**: Invalid configs cause a `ValidationError` at loading time, preventing partial start-ups.

### 2. The Unified Registry Pattern

Instead of maintaining separate registries for each plugin type (`SOURCE_REGISTRY`, `FILTER_REGISTRY`), we use a single, generic `PluginRegistry[ConfigT, PluginInterfaceT]`.

-   **Generics for Safety**: The registry is generic over the configuration type and the plugin interface.
    ```python
    # strict typing ensures that load_plugin(PluginType.SOURCE, ...) returns a ClipSource
    source = registry.load_plugin(PluginType.SOURCE, "my_source", config_dict)
    ```
-   **Declarative Registration**: Plugins register themselves using the `@plugin` decorator, which captures metadata (name, type) and creates the association without manual mapping code.

### 3. Factory Pattern vs. Class Creation

We transitioned from functional factories (`make_source()`) to class-based factories (`Class.create()`).

-   **Encapsulation**: The class handles its own construction logic in `create()`, keeping `__init__` clean or free for dependency injection.
-   **State Management**: Plugins often hold state (database connections, ML model handles). Classes naturally encapsulate this state and provide lifecycle methods (`shutdown()`) to clean it up.

### 4. Decoupling & Local State

A key design validation was separating `LocalFolderSource` from the global `StateStore`.

-   **Philosophy**: Components should own their local truth.
-   **Implementation**: `LocalFolderSource` uses a "Local State Manifest" (`.homesec_state.json`) to track processed files. This means the source can function even if the central database is down, fulfilling the "P0: Never miss new clips" requirement.

### 5. Error Handling Philosophy

-   **Boundaries must never crash**: Top-level loops wrap plugin calls in broad exception handlers (specifically catching `PipelineError`).
-   **Errors as Values**: Where possible, return error objects/states rather than raising exceptions, allowing the pipeline to degrade gracefully (e.g., skip analysis but still upload).

## Goals

Build a reliable, pluggable pipeline to:
1. Detect motion from home cameras.
2. Record short video clips and store them in a pluggable backend (Dropbox today, others later).
3. Filter clips for configured target classes using a pluggable object detection plugin (default: YOLOv8, but swappable).
4. For clips that match configured trigger classes, run analysis via a pluggable VLM plugin (default: OpenAI-compatible API, but swappable) to summarize, classify risk, and set an alert level.
5. Notify via a pluggable/multiplexable notifier (MQTT → Home Assistant, email, SMS, etc.) when either:
   - `risk_level` exceeds a configured threshold, OR
   - per-camera custom alert conditions match (even if `risk_level` is low) via `notify_on_activity_types`.

**Note:** Specific model implementations (YOLO weights, VLM providers) are outside the scope of this pipeline design—they are abstracted behind plugin interfaces.

### Non-Goals (Out of Scope for MVP)
- Extra VLM classifier passes (use `activity_type` from base VLM prompt instead)
- Post-upload Dropbox folder moves (upload directly to `{camera_name}/{clip_id}` path)
- Public share links (use web URLs requiring Dropbox login)
- Multi-worker distributed processing (single process with async concurrency)

### Open Questions
- None for MVP. Per-camera overrides are limited to alert policy; filter and VLM configs remain global.

### Latency Target
- **ASAP after motion ends**: Target 16-28s for upload + filter + VLM → notification
- **Breakdown** (parallel upload + filter):
  - Upload: 2-5s (parallel with filter, not on critical path if filter+VLM > upload time)
  - Filter (YOLO): 1-3s (via process pool, non-blocking)
  - VLM: 15-25s (longest pole)
  - Total critical path: max(upload, filter) + VLM ≈ 16-28s
- Further optimization (e.g., model quantization, caching) is nice-to-have but not required for MVP

### Reliability via Partial Failures
- **Error-as-value pattern** enables pipeline to continue when non-critical stages fail
- Example: If Dropbox is down, upload fails but filter + VLM + notify still run → user gets alert with `view_url=null`
- This ensures P0 requirement: "Never miss new clips" even during infrastructure outages

## Current Building Blocks (Repo)

- Motion + recording: `src/homesec/sources/rtsp/core.py` (OpenCV motion detection + ffmpeg recording).
  - RTSP helpers live in `src/homesec/sources/rtsp/`:
    - `frame_pipeline.py` (ffmpeg frame reader + stall/reconnect support)
    - `recorder.py` (ffmpeg recording process)
    - `motion.py` (motion detection logic)
    - `hardware.py` (hwaccel detection)
    - `clock.py` (clock abstraction for tests)
    - `utils.py` (shared helpers)
- Object detection plugin (reference): `src/homesec/plugins/filters/yolo.py` (YOLOv8 sampling to detect people/animals).
- VLM plugin (reference): `src/homesec/plugins/analyzers/openai.py` (structured output with `risk_level`, `activity_type`, timeline).
- Postgres (workflow state when available): Alembic migrations (`alembic/`) + telemetry logging (`src/db_log_handler.py`) + workflow state table `clip_states` (best-effort; DB outages may drop state updates).

## Proposed Architecture (Event-Driven Pipeline)

**Conceptual flow**

`ClipSource` → (parallel: `StorageBackend` + `ObjectFilter`) → `VLMAnalyzer` → `AlertPolicy` → `Notifier(s)`

### Components

1. **Clip Source (pluggable ingest + recorder)**
   - Responsibility: produce *finalized* video clips and write them to the configured `StorageBackend` (Dropbox today).
   - Current implementations in this repo:
     - RTSP motion recorder: `src/motion_recorder.py` (OpenCV motion detection + ffmpeg clip recording; async upload already supported).
     - FTP ingest: `src/ftp_dropbox_server.py` (cameras upload clips via FTP; then stored/uploaded).
   - Future: additional clip sources (custom camera integrations, ONVIF events, Home Assistant triggers, etc.).
   - Clip boundaries / debounce / retrigger: reuse existing `src/motion_recorder.py` defaults and logic (treat as the reference implementation for now).
   - Dropbox layout (MVP):
     - Upload path template: `{camera_name}/{clip_id}` (per-camera folders; `clip_id` includes timestamp).
     - No post-upload moves; classification is logical only (stored in `clip_states` via `alert_decision` and analysis results).
   - Naming / identity (clip_id):
     - Use the filename as `clip_id`.
     - Filenames must include: `{camera_name}` prefix and a `{timestamp_seconds}` (second-resolution) component.
     - If multiple clips can occur in the same second, add a deterministic suffix (e.g., `_2`, `_3`) while preserving the camera+timestamp parts.
   - **Health check** (`is_healthy()`):
     - RTSP: frame pipeline running + frames received recently (within `frame_timeout_s * 3`)
     - FTP: server thread alive + accepting connections
     - Returns `False` if source has failed and needs restart
   - **Heartbeat** (`last_heartbeat()`):
     - RTSP: timestamp of last frame received (updated continuously via background task)
     - FTP: timestamp of last successful connection check
     - Updated every ~60s regardless of motion/clips
     - Used for observability warnings, not health status

2. **Storage Backend (pluggable)**
   - Stores raw clips and derived artifacts (thumbnails, JSON analysis).
   - Implementations: `DropboxStorage`, `LocalFsStorage`, `S3Storage` (future), etc.
   - Key requirement: stable addressing (a `storage_uri`) and idempotent writes (overwrite or content-addressing).

3. **State Store (Postgres) + Local Queue (Filesystem)**
   - Canonical queue: local filesystem (clips persist even if Dropbox/DB is down).
   - Workflow/state: Postgres table `clip_states` (one row per `clip_id`) drives idempotency, retries, and worker coordination.
   - Minimal table sketch:
     - `clip_states(clip_id TEXT PRIMARY KEY, data JSONB NOT NULL, created_at TIMESTAMPTZ, updated_at TIMESTAMPTZ)`
     - Recommended indexes (as needed): `(data->>'status')`, `(data->>'camera_name')`
     - `data` should include a `schema_version` field and typed sub-objects (filter result, VLM result, alert decision, etc.).
    - Store `view_url` (Dropbox web URL) inside `data` once available; MQTT payloads should read it from here.
   - DB outage behavior (graceful degradation):
     - Clip sources continue writing clips to the local queue.
     - Workers continue best-effort:
       - If Postgres is unavailable, skip/disable coordination and continue processing from the local queue (duplicates are acceptable).
       - If Postgres writes fail, it is acceptable to drop state updates and rely on ERROR logs for visibility (no reconciliation required for MVP).
     - No local lock fallback in DB-down mode; duplicates are acceptable to keep the system simple.
   - If `local_path` is missing but `storage_uri` exists, workers should re-download to a temp path for processing.

4. **Object Detection Filter (Pluggable)**
   - Uses pluggable object detection to decide "needs VLM".
   - Plugin interface: `ObjectFilter.detect(video_path, overrides=None) -> FilterResult`
   - Reference implementation: YOLOv8 with configurable classes (default: person), frame sampling, early exit on detection.
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
   - Output: `AnalysisResult` (risk_level, activity_type, summary, entities_timeline, requires_review, etc.).
   - Stores `analysis.json` in the same storage backend, next to the clip.
   - Prompting is configured globally (same for all cameras); per-camera alert policy decides which `activity_type` values trigger notifications.

6. **Alert Policy (per camera)**
   - Decides whether to notify based on VLM output plus per-camera overrides.
   - MVP logic:
     - Default: notify if `risk_level > low` (or a configurable global threshold).
     - Additionally: allow per-camera `notify_on_activity_types` list that can trigger notifications even when `risk_level` is low.
    - Optional per-camera `notify_on_motion` to alert on any clip (VLM may still be skipped).
    - If VLM is skipped (run_mode=never or no trigger classes detected), default to `notify=false` unless `notify_on_motion` is enabled.
    - Notifications wait for upload so a `view_url` can be included.
  - Configuration:
    - `config`: `min_risk_level` + `notify_on_activity_types` list, plus `overrides` keyed by `camera_name`.
   - Example use cases:
     - Front door: notify on `activity_type == "person_at_door"` or `"delivery"` even at low risk
     - Backyard: notify on `activity_type == "animal_running"` at medium+ risk

7. **Notifier (pluggable + multiplexable)**
   - MVP behavior: send to all configured notification sinks.
   - Primary integration target: MQTT via Mosquitto → Home Assistant.
   - Future: routing rules, per-sink throttling, and richer dedupe/rate limiting.
   - Suggested MQTT contract (MVP):
     - Topic: `homecam/alerts/{camera_name}` (per-camera topics)
     - QoS: 1, retained: false
     - Payload (JSON): `clip_id`, `camera_name`, `storage_uri`, `view_url`, `risk_level`, `activity_type`, `notify_reason`, `summary`, `ts`, `dedupe_key`, `upload_failed`
   - View URL: compute a Dropbox web URL for the clip (requires login; not a share link).
   - **Partial failure handling**: If upload fails but filter detects a person, still send notification with `view_url=null` and `upload_failed=true`. User can check local recordings.
   - Deduping: per-clip only (use `clip_id` as `dedupe_key`); some duplicates are acceptable.

### Review UX (MVP)

- Primary review surface: Home Assistant notification history (each alert includes a clickable `view_url`).
- Manual fallback: browse Dropbox per-camera folders; `clip_id` embeds timestamps for sorting.
- No dedicated UI required for MVP.

### Core Interfaces (Python)

Keep these small and stable to make backends swappable:

```python
from pydantic import BaseModel, Field
from typing import Protocol, Callable, Optional, Any
from pathlib import Path
from datetime import datetime

# === Error Hierarchy ===

class PipelineError(Exception):
    """Base exception for all pipeline errors."""
    def __init__(self, message: str, stage: str, clip_id: str, cause: Exception | None = None):
        super().__init__(message)
        self.stage = stage
        self.clip_id = clip_id
        self.cause = cause  # Original exception
        self.__cause__ = cause  # Python's exception chaining

class UploadError(PipelineError):
    """Storage upload failed."""
    def __init__(self, clip_id: str, storage_uri: str | None, cause: Exception):
        super().__init__(f"Upload failed for {clip_id}", stage="upload", clip_id=clip_id, cause=cause)
        self.storage_uri = storage_uri

class FilterError(PipelineError):
    """Object detection filter failed."""
    def __init__(self, clip_id: str, plugin_name: str, cause: Exception):
        super().__init__(f"Filter failed for {clip_id} (plugin: {plugin_name})", stage="filter", clip_id=clip_id, cause=cause)
        self.plugin_name = plugin_name

class VLMError(PipelineError):
    """VLM analysis failed."""
    def __init__(self, clip_id: str, plugin_name: str, cause: Exception):
        super().__init__(f"VLM analysis failed for {clip_id} (plugin: {plugin_name})", stage="vlm", clip_id=clip_id, cause=cause)
        self.plugin_name = plugin_name

class NotifyError(PipelineError):
    """Notification delivery failed."""
    def __init__(self, clip_id: str, notifier_name: str, cause: Exception):
        super().__init__(f"Notify failed for {clip_id} (notifier: {notifier_name})", stage="notify", clip_id=clip_id, cause=cause)
        self.notifier_name = notifier_name

# === Core Data Models ===

class Clip(BaseModel):
    """Represents a finalized video clip ready for processing."""
    clip_id: str
    camera_name: str
    local_path: Path
    start_ts: datetime
    end_ts: datetime
    duration_s: float
    source_backend: str  # "rtsp", "ftp", etc.

class ClipSource(Protocol):
    """Produces finalized clips and notifies pipeline via callback."""
    def register_callback(self, callback: Callable[[Clip], None]) -> None:
        """Register callback to be invoked when a new clip is finalized."""
    
    def start(self) -> None:
        """Start producing clips (long-running, blocks or runs in background)."""
    
    def stop(self) -> None:
        """Graceful shutdown."""
    
    def is_healthy(self) -> bool:
        """
        Check if source is actively able to receive clips.
        
        Implementation should check:
        - Process/thread is alive
        - Receiving data recently (e.g., RTSP frames within timeout)
        - NOT dependent on motion/clip activity
        
        Examples:
        - RTSP: frame pipeline running + receiving frames recently
        - FTP: server thread alive + accepting connections
        
        Returns False if source has failed and needs restart.
        """
    
    def last_heartbeat(self) -> float:
        """
        Return timestamp (monotonic) of last successful operation.
        
        Examples:
        - RTSP: timestamp of last frame received
        - FTP: timestamp of last connection check
        
        Updated continuously (every ~60s), independent of motion/clips.
        Used for observability, not health status.
        """

class StorageBackend(Protocol):
    async def put(self, local_path: Path, dest_key: str) -> str:  # Returns storage_uri
    async def get(self, storage_uri: str, local_path: Path) -> None:
    async def exists(self, storage_uri: str) -> bool:
    async def put_json(self, obj: dict, dest_key: str) -> str:  # Returns storage_uri
    async def ping(self) -> bool:  # Health check
    async def close(self) -> None:  # Cleanup resources

class StateStore(Protocol):
    async def upsert(self, clip_id: str, data: ClipStateData) -> None:
    async def get(self, clip_id: str) -> Optional[ClipStateData]:
    async def ping(self) -> bool:  # Health check

class Notifier(Protocol):
    async def send(self, alert: Alert) -> None:
    async def ping(self) -> bool:  # Health check
    async def close(self) -> None:  # Cleanup resources

class AlertPolicy(Protocol):
    def should_notify(
        self, camera_name: str, 
        filter_result: Optional[FilterResult],
        analysis: Optional[AnalysisResult]
    ) -> tuple[bool, str]:  # (notify, reason)

# === Plugin Interfaces ===

class ObjectFilter(Shutdownable, Protocol):
    """Plugin interface for object detection in video clips."""
    
    async def detect(
        self, video_path: Path, overrides: FilterOverrides | None = None
    ) -> FilterResult:
        """
        Detect objects in video clip.
        
        Implementation notes:
        - MUST be async (use asyncio.to_thread or run_in_executor for blocking code)
        - CPU/GPU-bound plugins should manage their own ProcessPoolExecutor internally
        - I/O-bound plugins can use async HTTP clients directly
        - If managing a worker pool, use concurrency settings from the plugin's config model
        - Should support early exit on first detection for efficiency
        - overrides apply per-call (model path cannot be overridden)
        
        Returns:
            FilterResult with detected_classes, confidence, sampled_frames, model name
        """
        ...
    
    async def shutdown(self) -> None:
        """
        Cleanup resources (process pools, GPU memory, file handles).
        """
        ...

class VLMAnalyzer(Shutdownable, Protocol):
    """Plugin interface for VLM-based clip analysis."""
    
    async def analyze(
        self, video_path: Path, filter_result: FilterResult, config: VLMConfig
    ) -> AnalysisResult:
        """
        Analyze clip and produce structured assessment.
        
        Implementation notes:
        - MUST be async (use asyncio.to_thread or run_in_executor for blocking code)
        - Local models: manage ProcessPoolExecutor internally
        - API-based: use async HTTP clients (aiohttp, httpx)
        - If managing a worker pool, use concurrency settings from the plugin's config model
        - Should use filter_result to focus analysis (e.g., detected person at timestamp X)
        
        Returns:
            AnalysisResult with risk_level, activity_type, summary, etc.
        """
        ...
    
    async def shutdown(self) -> None:
        """
        Cleanup resources (HTTP sessions, process pools, model memory).
        """
        ...
```

**Reference Implementations (MVP):**
- `YOLOv8Filter`: Default object detection using YOLOv8n model
  - Uses `ProcessPoolExecutor` internally for CPU/GPU-bound inference
  - `shutdown()` implementation: `self._executor.shutdown(wait=True)`
- `OpenAIVLM`: Default VLM using OpenAI-compatible API (GPT-4o, etc.)
  - Uses `aiohttp.ClientSession` for async HTTP calls
  - `shutdown()` implementation: `await self._session.close()`
- `MockFilter` / `MockVLM`: For testing (instant responses, no actual inference)
  - `shutdown()` implementation: no-op (no resources to clean up)

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
        notify_on_activity_types: [delivery]
```

### Config Structure

- `cameras[]`: `{name, source: {backend, config}}` (source-specific config is validated by each source implementation)
  - Default MQTT topic if not specified: `homecam/alerts/{name}`
- `storage`: `{backend, config, paths}` (backend config validated by storage plugin)
- `retention`: local disk limits and cleanup strategy
- `mqtt`: broker host/port/credentials env var names
- `filter`: plugin selection + defaults (global for all cameras)
  - `backend`: name of object detection plugin (e.g., "yolo", "mock")
  - `config`: plugin-specific config (classes to detect, model name, etc.)
- `vlm`: plugin selection + defaults (global for all cameras)
  - `backend`: name of VLM plugin (e.g., "openai", "anthropic", "mock")
  - `run_mode`: `trigger_only | always | never`
  - `trigger_classes`: object classes that gate VLM in `trigger_only` mode
  - `config`: plugin-specific config (model, prompts, activity_types, etc.)
  - `preprocessing`: frame extraction config
- `alert_policy`: backend selection + config (defaults + per-camera overrides live in `config`)
- `concurrency`: per-stage parallel processing limits (upload, filter, vlm) + global limit
- `retry`: max attempts, backoff delay, behavior on exhaustion
- `health`: HTTP health check endpoint config

YAML config v1 sketch (env var names only; secrets live in `.env` or a secret manager):
```yaml
version: 1

storage:
  backend: dropbox
  config:
    root: "/homecam"
    path_template: "{camera_name}/{filename}"
    token_env: "DROPBOX_TOKEN"
    web_url_prefix: "https://www.dropbox.com/home"
  paths:
    clips_dir: "clips"
    backups_dir: "backups"
    artifacts_dir: "artifacts"

retention:
  max_local_size: "10GB"

mqtt:
  host: "localhost"
  port: 1883
  auth:
    username_env: "MQTT_USERNAME"
    password_env: "MQTT_PASSWORD"
  topic_template: "homecam/alerts/{camera_name}"
  qos: 1
  retain: false

filter:
  backend: "yolo"  # Options: "yolo", "mock" (add more as needed)
  config:
    model: "yolov8n"
    classes: ["person", "animal"]
    min_confidence: 0.5
    sample_fps: 2
    max_workers: 4
  # per-camera overrides for filter are not supported in MVP

vlm:
  backend: "openai"  # Options: "openai", "anthropic", "mock" (add more as needed)
  run_mode: "trigger_only"
  trigger_classes: ["person"]
  config:
    model: "gpt-4o"
    api_key_env: "OPENAI_API_KEY"
    base_prompt: "Summarize activity and risk; be concise and structured."
    activity_types: ["delivery", "doorbell", "person_at_door", "unknown"]
  preprocessing:
    max_frames: 10
    max_size: 1024
    quality: 85
  # per-camera overrides for VLM are not supported in MVP

alert_policy:
  backend: "default"
  enabled: true
  config:
    min_risk_level: "medium"
    notify_on_activity_types: []
    notify_on_motion: false
    overrides:
      front_door:
        min_risk_level: "low"
        notify_on_activity_types: ["person_at_door", "delivery"]
        notify_on_motion: false
      backyard:
        min_risk_level: "medium"
        notify_on_activity_types: ["animal_running"]
        notify_on_motion: false

concurrency:
  max_clips_in_flight: 10  # Global limit (prevents OOM when many cameras trigger)
  upload_workers: 3        # Per-stage limits (within global limit)
  filter_workers: 4
  vlm_workers: 2

retry:
  max_attempts: 3
  backoff_s: 5.0
  on_exhausted: "alert_and_continue"  # Options: "alert_and_continue", "fail"

health:
  enabled: true
  port: 8080
  endpoint: "/health"

cameras:
  - name: "front_door"
    source:
      backend: "rtsp"
      config:
        rtsp_url_env: "FRONT_DOOR_RTSP_URL"
  - name: "driveway"
    source:
      backend: "ftp"
      config:
        ftp_subdir: "driveway"
```

## Data Model (Clip-Centric)

Define a single “clip record” (DB row or JSON) as the unit of work:

- Identity: `clip_id` is the clip filename (must include `{camera_name}` + `{timestamp_seconds}`).
- Pointers:
  - `storage_uri` for the video clip in the storage backend (e.g., Dropbox path).
  - Optional artifact URIs for exported JSON stored alongside the clip (useful for debugging/portability): `filter_result_uri`, `analysis_uri`.
  - Canonical structured results should also be stored in `clip_states.data` (Pydantic-validated); exported JSON artifacts are a convenience, not the source of truth.
- Timing: `start_ts`, `end_ts`, `duration_s`.
- Decisions: `risk_level`, `alert_sent_at`, `retry_count`.
- Provenance: models used + versions (`yolo_model`, `vlm_model`, prompts/config hashes).

### Workflow State (`clip_states`)

Use a single workflow/state table with a typed `clip_id` column plus a schemaless `data jsonb` payload (validated by Pydantic).

**Table schema:**
```sql
CREATE TABLE clip_states (
    clip_id TEXT PRIMARY KEY,
    data JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX idx_clip_states_status ON clip_states ((data->>'status'));
CREATE INDEX idx_clip_states_camera ON clip_states ((data->>'camera_name'));
```

**Pydantic schema for `data` (v1):**

```python
from pydantic import BaseModel
from typing import Literal, Optional
from datetime import datetime

class StageLog(BaseModel):
    """Detailed journal for each processing stage."""
    status: Literal["pending", "running", "ok", "error", "skipped"]
    attempts: int = 0
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    last_error: Optional[str] = None

class FilterResult(BaseModel):
    detected_classes: list[str]
    confidence: float
    model: str
    sampled_frames: int

class AnalysisResult(BaseModel):
    risk_level: Literal["low", "medium", "high"]
    activity_type: str
    summary: str
    # ... other VLM output fields

class AlertDecision(BaseModel):
    notify: bool
    notify_reason: str  # e.g., "risk_level=high" or "activity_type=delivery (per-camera)"

class Alert(BaseModel):
    """MQTT notification payload."""
    clip_id: str
    camera_name: str
    storage_uri: str | None
    view_url: str | None
    risk_level: Literal["low", "medium", "high"] | None  # None if VLM skipped
    activity_type: str | None
    notify_reason: str
    summary: str | None
    ts: datetime
    dedupe_key: str  # Same as clip_id for MVP
    upload_failed: bool  # True if storage_uri is None due to upload failure

class FilterConfig(BaseModel):
    """Base filter configuration (plugin-agnostic)."""
    backend: str  # e.g., "yolo", "mock"
    config: dict[str, Any]  # Plugin-specific config (validated by plugin at load time)

class VLMConfig(BaseModel):
    """Base VLM configuration (plugin-agnostic)."""
    backend: str  # e.g., "openai", "anthropic", "mock"
    trigger_classes: list[str] = Field(default_factory=lambda: ["person"])
    run_mode: Literal["trigger_only", "always", "never"] = "trigger_only"
    config: dict[str, Any]  # Plugin-specific config (validated by plugin at load time)
    preprocessing: VLMPreprocessConfig = Field(default_factory=VLMPreprocessConfig)

# Note: Plugin-specific config validation:
# - Each plugin receives the config dict and validates/transforms it during initialization
# - Plugins should raise clear errors for missing/invalid config at load time (fail fast)
# - Base configs (FilterConfig, VLMConfig) only validate plugin-agnostic fields

class ClipStateData(BaseModel):
    """Complete state for a single clip (stored in clip_states.data JSONB)."""
    schema_version: int = 1
    camera_name: str
    
    # Lifecycle: single status for queries, plus detailed per-stage journal
    status: Literal["queued_local", "uploaded", "filtered", "analyzed", "done", "error"]
    stages: dict[str, StageLog]  # Keys: "upload", "filter", "vlm", "notify"
    
    # Pointers
    local_path: str
    storage_uri: Optional[str] = None
    view_url: Optional[str] = None
    
    # Processing results
    filter_result: Optional[FilterResult] = None
    analysis_result: Optional[AnalysisResult] = None
    alert_decision: Optional[AlertDecision] = None
```

**Status model:**
- `status` is a best-effort summary for queries/UX, derived from detailed `stages.*` journal.
- Status values: `queued_local`, `uploaded`, `filtered`, `analyzed`, `done`, `error`.
- Derivation logic (applied after each stage update):
  ```python
  def derive_status(data: ClipStateData) -> str:
      if any(s.status == "error" for s in data.stages.values()):
          return "error"
      if data.stages.get("notify", StageLog(status="pending")).status == "ok":
          return "done"
      if data.stages.get("vlm", StageLog(status="pending")).status == "ok":
          return "analyzed"
      if data.stages.get("filter", StageLog(status="pending")).status == "ok":
          return "filtered"
      if data.stages.get("upload", StageLog(status="pending")).status == "ok":
          return "uploaded"
      return "queued_local"
  ```
- Per-stage journal (`stages.*`) provides detailed debugging history: attempts, timestamps, errors.

**Example `data` JSON (illustrative):**
```json
{
  "schema_version": 1,
  "camera_name": "front_door",
  "status": "analyzed",
  "stages": {
    "upload": {
      "status": "ok",
      "attempts": 1,
      "started_at": "2024-01-01T12:00:00Z",
      "finished_at": "2024-01-01T12:00:01Z",
      "last_error": null
    },
    "filter": {
      "status": "ok",
      "attempts": 1,
      "started_at": "2024-01-01T12:00:01Z",
      "finished_at": "2024-01-01T12:00:04Z",
      "last_error": null
    },
    "vlm": {
      "status": "ok",
      "attempts": 1,
      "started_at": "2024-01-01T12:00:04Z",
      "finished_at": "2024-01-01T12:00:10Z",
      "last_error": null
    },
    "notify": {
      "status": "pending",
      "attempts": 0,
      "started_at": null,
      "finished_at": null,
      "last_error": null
    }
  },
  "local_path": "recordings/front_door_1734307501.mp4",
  "storage_uri": "dropbox:/front_door/front_door_1734307501.mp4",
  "view_url": "https://www.dropbox.com/home/homecam/front_door/front_door_1734307501.mp4",
  "filter_result": {
    "detected_classes": ["person"],
    "confidence": 0.72,
    "model": "yolov8n",
    "sampled_frames": 30
  },
  "analysis_result": {
    "risk_level": "low",
    "activity_type": "delivery",
    "summary": "Delivery person left package at door."
  },
  "alert_decision": {
    "notify": true,
    "notify_reason": "activity_type=delivery (per-camera)"
  }
}
```

## Production Requirements

### Product Priorities (Use-Case Driven)

1. **P0 — Never miss new clips:** keep recording and spooling locally even if Dropbox or Postgres is down.
2. **P0 — Upload ASAP:** upload to Dropbox best-effort as soon as possible; do not block recording.
3. **P1 — Analyze + notify:** run filter/VLM and send MQTT notifications best-effort; duplicates are acceptable during outages.
4. **P2 — State consistency:** prefer best-effort correctness over strict exactly-once behavior; state may be incomplete when dependencies are unavailable.

### Reliability & Correctness
- **Best-effort idempotency:** avoid duplicate uploads/alerts when possible; duplicates are acceptable during outages.
- **At-least-once processing:** workers retry on failure; queue/state tracks progress.
- **Backpressure:** bound concurrency for downloads/decoding/VLM calls; drop or defer gracefully.
- **Graceful DB degradation:** if Postgres is down/unreachable, continue recording/uploading and process best-effort from the local queue with no local lock fallback; accept duplicates and tolerate missing state.
- **Offline tolerance:** if internet/storage is down, persist clips locally and upload later (async uploader is allowed/encouraged).
- **Retention:** never delete a local clip unless upload is confirmed (`stages.upload.status=ok`).
  - Enforce `retention.max_local_size` by deleting oldest uploaded clips first.
  - If `alert_decision.notify == true`, delete after `stages.notify.status=ok` unless storage pressure requires earlier cleanup of already-uploaded clips.
  - Otherwise delete after filter/VLM completes (success or failure); retries can re-download from storage.

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
- Prefer using the `path_display` returned by the Dropbox upload response to build `view_url` (no extra API call).
- URLs are stable as long as files are not moved; MVP assumes no post-upload moves.

### Pipeline Architecture (Single Process, Async Flow)

**Control flow:**
- Single process orchestrator (`ClipPipeline`) with async processing per clip
- ClipSource implementations (RTSP recorder, FTP server) register callbacks via `register_callback(fn)`
- On new clip: callback invokes `_on_new_clip(clip)` → `asyncio.create_task(_process_clip(clip))`
- Each clip flows through:
  - **Parallel**: upload + filter (start simultaneously to reduce latency)
  - **Sequential**: VLM (conditional, after filter completes) → alert → notify (after upload completes for `view_url`)
- Bounded concurrency per stage via `asyncio.Semaphore` (configurable limits) for I/O-bound stages
- **Plugins manage their own concurrency**: CPU/GPU-bound plugins (YOLO) manage `ProcessPoolExecutor` internally; I/O-bound plugins (OpenAI API) use async HTTP clients
- Pipeline is model-agnostic; it just awaits plugin async methods

**Startup/recovery:**
1. On startup, scan local filesystem for clips not in `clip_states` or with incomplete status
2. Prioritize newest clips first (sort by modification time descending)
3. Enqueue via `_on_new_clip()` for processing with small delays between clips
4. Resume from last known stage (idempotent operations skip if already complete)

*Implementation note:*
```python
async def recover_incomplete_clips(self):
    """Scan local filesystem and resume processing incomplete clips."""
    all_clips = []
    for clip_file in Path(self.config.output_dir).glob("*.mp4"):
        state = await self.state_store.get(clip_file.name)
        if not state or state.status != "done":
            all_clips.append((clip_file.stat().st_mtime, self._clip_from_file(clip_file)))
    
    # Sort by modification time (newest first)
    all_clips.sort(reverse=True)
    
    # Process newest first, but process all eventually
    for _, clip in all_clips:
        asyncio.create_task(self._process_clip_bounded(clip))
        await asyncio.sleep(0.1)  # Small delay to avoid startup spike
```

**Implementation sketch (error-as-value pattern for partial failures):**

*Note: The following is PSEUDOCODE for illustration only. It demonstrates the design pattern but omits retry logic, comprehensive error handling, logging details, state management helper methods, and other production concerns. Actual implementation will include proper type annotations, error handling, logging, and resource cleanup as specified in AGENTS.md and throughout this document.*

```python
# Plugin loading (registry pattern)
def load_filter(config: FilterConfig) -> ObjectFilter:
    return load_plugin(PluginType.FILTER, config.backend, config.config)

def load_analyzer(config: VLMConfig) -> VLMAnalyzer:
    return load_plugin(PluginType.ANALYZER, config.backend, config.config)

# Future: Support setuptools entry points for third-party plugins (see Open Questions)

class ClipPipeline:
    def __init__(self, config: Config):
        # ... other init
        self.filter_plugin = load_filter(config.filter)
        self.vlm_plugin = load_analyzer(config.vlm)
        
        # Global concurrency limit (prevents OOM)
        self.global_semaphore = asyncio.Semaphore(config.concurrency.max_clips_in_flight)
        
        # Per-stage limits (within global limit)
        self.upload_semaphore = asyncio.Semaphore(config.concurrency.upload_workers)
        self.filter_semaphore = asyncio.Semaphore(config.concurrency.filter_workers)
        self.vlm_semaphore = asyncio.Semaphore(config.concurrency.vlm_workers)
    
    def _on_new_clip(self, clip: Clip) -> None:
        """Callback from ClipSource when new clip is ready."""
        # Don't block the callback - create task
        asyncio.create_task(self._process_clip_bounded(clip))
    
    async def _process_clip_bounded(self, clip: Clip) -> None:
        """Wrapper that enforces global concurrency limit."""
        async with self.global_semaphore:
            await self._process_clip(clip)
    
    async def _upload_stage(self, clip: Clip) -> StorageUploadResult | UploadError:
        """Upload clip. Returns StorageUploadResult on success, UploadError on failure."""
        try:
            return await self.storage.put_file(clip.local_path, f"{clip.camera_name}/{clip.clip_id}")
        except Exception as e:
            return UploadError(clip.clip_id, storage_uri=None, cause=e)
    
    async def _filter_stage(self, clip: Clip) -> FilterResult | FilterError:
        """Run filter. Returns FilterResult on success, FilterError on failure."""
        try:
            result = await self.filter_plugin.detect(clip.local_path)
            return result
        except Exception as e:
            return FilterError(clip.clip_id, plugin_name=self.config.filter.backend, cause=e)
    
    async def _process_clip(self, clip: Clip) -> None:
        # Stage 1 & 2: Upload and Filter in parallel
        upload_task = asyncio.create_task(self._upload_stage(clip))
        filter_task = asyncio.create_task(self._filter_stage(clip))
        
        upload_res = await upload_task
        filter_res = await filter_task
        
        # Handle filter result (critical - cannot proceed without it)
        match filter_res:
            case FilterError() as err:
                logger.error("Filter failed: %s", err.cause, exc_info=err.cause,
                            extra={"clip_id": clip.clip_id, "plugin": err.plugin_name})
                self._update_state_stage(clip.clip_id, "filter", status="error", last_error=str(err))
                return
            case FilterResult() as result:
                filter_result = result
                self._update_state_stage(clip.clip_id, "filter", status="ok")
        
        # Handle upload result (non-critical - can proceed without it)
        match upload_res:
            case UploadError() as err:
                logger.error("Upload failed, proceeding without URL: %s", err.cause,
                            exc_info=err.cause, extra={"clip_id": clip.clip_id})
                self._update_state_stage(clip.clip_id, "upload", status="error", last_error=str(err))
                storage_uri = None
                view_url = None
            case StorageUploadResult() as result:
                storage_uri = result.storage_uri
                view_url = result.view_url
                self._update_state_stage(clip.clip_id, "upload", status="ok")
        
        # Stage 3: VLM (conditional)
        analysis_result = None
        if self._should_run_vlm(filter_result):
            vlm_res = await self._vlm_stage(clip, filter_result)
            match vlm_res:
                case VLMError() as err:
                    logger.error("VLM failed: %s", err.cause, exc_info=err.cause,
                                extra={"clip_id": clip.clip_id, "plugin": err.plugin_name})
                    self._update_state_stage(clip.clip_id, "vlm", status="error", last_error=str(err))
                    # Continue to alert (might have notify_on_motion enabled)
                case AnalysisResult() as result:
                    analysis_result = result
                    self._update_state_stage(clip.clip_id, "vlm", status="ok")
        
        # Stage 4: Alert decision
        alert_decision = self._alert_stage(clip.camera_name, filter_result, analysis_result)
        
        # Stage 5: Notify (conditional, works even without view_url)
        if alert_decision.notify:
            notify_res = await self._notify_stage(clip, alert_decision, storage_uri, view_url)
            match notify_res:
                case NotifyError() as err:
                    logger.error("Notify failed: %s", err.cause, exc_info=err.cause,
                                extra={"clip_id": clip.clip_id})
                    self._update_state_stage(clip.clip_id, "notify", status="error", last_error=str(err))
                case None:
                    self._update_state_stage(clip.clip_id, "notify", status="ok")
        
        self._update_state(clip.clip_id, status="done")
        self._cleanup_local_file(clip.local_path, alert_decision.notify)
    
    def _should_run_vlm(self, filter_result: FilterResult) -> bool:
        """Check if VLM should run based on run_mode + detected classes."""
        match self.config.vlm.run_mode:
            case "never":
                return False
            case "always":
                return True
            case "trigger_only":
                detected = set(filter_result.detected_classes)
                trigger = set(self.config.vlm.trigger_classes)
                return bool(detected & trigger)
    
    # Note: The following helper methods are omitted from pseudocode for brevity:
    # - _vlm_stage(clip, filter_result) -> AnalysisResult | VLMError
    # - _alert_stage(camera_name, filter_result, analysis_result) -> AlertDecision
    # - _notify_stage(clip, alert_decision, storage_uri, view_url) -> None | NotifyError
    # - _update_state_stage(clip_id, stage, status, last_error=None) -> None
    # - _update_state(clip_id, status) -> None
    # - _cleanup_local_file(local_path, notify_sent) -> None
    # - _clip_from_file(clip_file) -> Clip
    # Implementation follows same error-as-value pattern as _upload_stage and _filter_stage

    async def shutdown(self, timeout: float = 30.0) -> None:
        """Graceful shutdown of all components with timeout."""
        logger.info("Shutting down pipeline...")
        
        # Stop accepting new clips
        for source in self.sources:
            source.stop()
        
        # Give in-flight clips time to finish
        logger.info("Waiting for in-flight clips to complete...")
        await asyncio.sleep(2)
        
        # Cancel remaining tasks
        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        if tasks:
            logger.warning("Cancelling %d remaining tasks", len(tasks))
            for task in tasks:
                task.cancel()
            
            # Wait for cancellations with timeout
            try:
                await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                logger.error("Some tasks did not cancel within %ds timeout", timeout)
        
        # Shutdown plugins and resources
        await self.filter_plugin.shutdown()
        await self.vlm_plugin.shutdown()
        await self.storage.close()
        await self.notifier.close()
        
        logger.info("Pipeline shutdown complete")
```

**Key points:**
- Stage methods return `Result | StageError` instead of raising exceptions
- **Pattern matching** (Python 3.10+) used for clean error handling: `match result: case Error() as err: ...`
- Upload and filter run in parallel via `asyncio.create_task()`
- Upload failure is non-critical: pipeline continues, sends alert with `view_url=null`
- Filter failure is critical: pipeline aborts (cannot make alert decision without knowing what's in video)
- Stack traces preserved in error objects via `.cause` attribute
- Plugins manage their own multiprocessing/async internally
- **Lifecycle management**: All plugins and components must implement `shutdown()` to prevent resource leaks
- **Global concurrency limit**: `max_clips_in_flight` prevents OOM when many cameras trigger simultaneously

**Main application entry point:**
```python
async def main():
    config = load_config("config.yaml")
    pipeline = ClipPipeline(config)
    
    try:
        await pipeline.run()  # Long-running
    finally:
        await pipeline.shutdown()  # Cleanup on exit/signal
```

**Error handling & retry:**
- **Errors as values**: Stage methods return `Result | StageError` (not raise) to enable partial failures
- **Stack traces preserved**: Error objects contain `.cause` with original exception and full traceback
- Each stage tracks attempts in `clip_states.data.stages.<stage>.attempts`
- On failure: increment attempts, record error in `last_error`, wait `retry.backoff_s`, retry
- After `retry.max_attempts`: mark stage `status=error`, log ERROR, send system alert to `homecam/system/errors` topic
- **Critical vs non-critical failures**:
  - Filter failure: abort clip processing (cannot make alert decision)
  - Upload/VLM/Notify failure: continue processing, log error, send alert with partial data (e.g., `view_url=null`)
- Continue to next clip (non-blocking)

### Observability
- Structured logs with `camera_name`, `clip_id`, and `event_type` (existing patterns in `src/db_log_handler.py`).
- Metrics to track:
  - clip count/day, upload failures, filter latency, VLM latency/cost, alert count.
- Postgres enabled via `DB_DSN` provides telemetry + workflow state; if DB writes fail, emit ERROR logs and continue best-effort.

### Health Check Endpoint
- Simple HTTP endpoint on `:8080/health` (configurable)
- **Component-based health checks**: Verify each component can do its job, not whether it's doing work
- Returns JSON with:
  - `status`: "healthy" | "degraded" | "unhealthy"
  - `checks`: dict of component health results
    - `db`: bool (Postgres ping)
    - `storage`: bool (Dropbox/storage ping)
    - `mqtt`: bool (MQTT broker ping)
    - `sources`: bool (all ClipSource.is_healthy() checks)
    - `plugins`: bool (optional: plugin health checks)
  - `clips_in_flight`: count of clips currently being processed
  - `last_clip_ts`: timestamp of most recent clip (informational only, not used for health status)
  - `warnings`: list of non-critical issues (e.g., "no_clips_24h")

**Health status logic:**
- `healthy`: All critical components operational
- `degraded`: Non-critical component failure (DB or MQTT down; pipeline can still process clips)
- `unhealthy`: Critical component failure (clip sources, storage, or plugins down; pipeline cannot function)

**Critical checks (unhealthy if fail):**
- `sources`: At least one ClipSource is healthy (can receive clips)
- `storage`: Storage backend is reachable (can upload clips)
- `plugins`: Object detection and VLM plugins are available (optional check)

**Non-critical checks (degraded if fail):**
- `db`: Postgres unavailable (state tracking disabled, but processing continues)
- `mqtt`: MQTT broker unreachable (notifications disabled, but processing continues)
  - Configurable via `health.mqtt_is_critical`: if `true`, MQTT failure is treated as critical (unhealthy)
  - Failed notifications are retried at `alert_policy.mqtt_retry_interval_s` intervals

**Activity monitoring:**
- `last_clip_ts` is informational only; no automatic alerting on "no recent clips"
- Users can create custom Home Assistant automations for suspicious inactivity (e.g., no motion in 7 days)

**Heartbeat monitoring:**
- Each ClipSource updates `last_heartbeat()` continuously (~60s) independent of motion/clips
- Health check uses `is_healthy()` for status (checks if source is functional)
- Heartbeat age appears in `warnings` if stale (>2 minutes), not in health status
- Purpose: distinguish between "no motion" (normal) vs "source dead" (actionable)

**Implementation example:**
```python
class HealthServer:
    async def health(self, request):
        checks = {
            "db": await self.pipeline.state_store.ping(),
            "storage": await self.pipeline.storage.ping(),
            "mqtt": await self.pipeline.notifier.ping(),
            "sources": all(source.is_healthy() for source in self.pipeline.sources),
        }
        
        # Determine status
        status = "healthy"
        if not checks["sources"] or not checks["storage"]:
            status = "unhealthy"  # Critical failure
        elif not checks["mqtt"] and self.config.health.mqtt_is_critical:
            status = "unhealthy"  # MQTT failure treated as critical (configurable)
        elif not checks["db"] or not checks["mqtt"]:
            status = "degraded"  # Non-critical failure
        
        # Add warnings (informational)
        warnings = []
        last_clip_ts = self.pipeline.get_last_clip_timestamp()
        if last_clip_ts and (time.time() - last_clip_ts) > 86400:
            warnings.append("no_clips_24h")  # No motion in 24h (informational only)
        
        # Check heartbeats (separate from clip activity)
        for source in self.pipeline.sources:
            heartbeat_age = time.monotonic() - source.last_heartbeat()
            if heartbeat_age > 120:  # 2 minutes without heartbeat
                warnings.append(f"source_{source.camera_name}_heartbeat_stale")
        
        return web.json_response({
            "status": status,
            "checks": checks,
            "clips_in_flight": self.pipeline.get_in_flight_count(),
            "last_clip_ts": last_clip_ts,
            "warnings": warnings,
        })
```

**Home Assistant integration:**
```yaml
binary_sensor:
  - platform: rest
    name: "Camera Pipeline Health"
    resource: "http://pipeline-host:8080/health"
    value_template: "{{ value_json.status == 'healthy' }}"
    scan_interval: 60
    json_attributes:
      - checks
      - last_clip_ts
      - warnings
```

### Testing Strategy
- **Unit tests**: Individual stages (filter, VLM, alert policy) with mocked dependencies
- **Integration tests**: Full pipeline with pytest, real Postgres + MQTT (via docker-compose), mocked VLM/storage
- **End-to-end tests**: Real camera streams (or recorded samples), real Dropbox/MQTT
- **Config validation**: Fail fast on startup if YAML is malformed or required env vars are missing

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
- **CI/CD integration**: Run `mypy src/ --strict` in CI to prevent merging unsafe code

### Security & Privacy
- Never commit secrets (`.env`, tokens, RTSP creds).
- Encrypt storage where possible; limit who can access raw clips.
- Redact sensitive data from logs (RTSP URLs, tokens).
- Treat VLM output as advisory; keep a “requires_review” path for ambiguous cases.

## Deployment Model (Practical Home Setup)

**Recommended MVP deployment**
- Single process orchestrator (`ClipPipeline`) runs as a long-lived process (systemd or Docker)
- Clip sources (RTSP recorder, FTP server) run as part of the same process or separate services that callback to pipeline
- Filter and VLM plugins run in-process (plugins manage their own ProcessPoolExecutor or async resources)
- All components on the same machine to avoid Dropbox re-downloads (local file access for filter/VLM)
- Optional services via Docker Compose:
  - `postgres` (recommended: telemetry + workflow state; recording/upload should still work if DB is down)
  - `mosquitto` (MQTT → Home Assistant)

## Open Questions / Decisions

- MQTT payload finalization: confirm all fields needed by Home Assistant automations (current: `clip_id`, `camera_name`, `storage_uri`, `view_url`, `risk_level`, `activity_type`, `notify_reason`, `summary`, `ts`, `dedupe_key`).
- Evaluation loop: use `src/evals/` to pick the best VLM config for cost/quality before production defaults.
- Schema migration: define process for handling `schema_version` bumps in `clip_states.data` (e.g., lazy migration on read, or explicit migration script).
- Plugin extensibility: Current MVP uses simple registry pattern for built-in plugins. Future enhancement: support setuptools entry points for third-party plugins (e.g., `homecam.filters`, `homecam.analyzers`) to enable external packages to register without modifying core code.

## MVP Roadmap (Implementation Order)

1. **Define core Pydantic models**: `Clip`, `ClipStateData`, `StageLog`, `FilterResult`, `AnalysisResult`, `AlertDecision`, config models
2. **Define error hierarchy**: `PipelineError`, `UploadError`, `FilterError`, `VLMError`, `NotifyError`
3. **Define plugin interfaces**: `ObjectFilter` and `VLMAnalyzer` Protocol definitions with `detect/analyze()` and `shutdown()` methods
4. **Implement reference plugins**:
   - `YOLOv8Filter`: Wrap existing `src/human_filter/human_filter.py` with plugin interface, add `ProcessPoolExecutor` + `shutdown()`
   - `OpenAIVLM`: Wrap existing `src/vlm.py` with plugin interface, use `aiohttp.ClientSession` + `shutdown()`
   - `MockFilter` and `MockVLM`: For testing (instant responses, no-op `shutdown()`)
5. **Implement `ClipSource` interface and adapters**: Callback-based interface with `is_healthy()` and `last_heartbeat()`, adapter for existing `MotionRecorder`, FTP source
6. **Add Postgres `clip_states` table**: Alembic migration, `StateStore` implementation with Pydantic serialization
7. **Implement `ClipPipeline` orchestrator**: Single async flow per clip with error-as-value pattern, plugin loading, parallel upload+filter, startup recovery, `shutdown()` method
8. **Implement `StorageBackend`**: Dropbox implementation with `put`, `get`, `exists`, `put_json`, `ping`, `close()`; compute `view_url` from upload response
9. **Wire stages with retry logic**: Upload → Filter → VLM (conditional) → Alert → Notify (conditional), with error handling as values
10. **Implement `AlertPolicy`**: Evaluate `min_risk_level` + `notify_on_activity_types` per camera
11. **Implement `Notifier`**: MQTT integration (Mosquitto), send alerts to `homecam/alerts/{camera_name}` with full payload including `upload_failed` flag
12. **Add health check endpoint**: Simple HTTP server on `:8080/health` with component checks + heartbeat warnings
13. **Integration tests**: End-to-end tests with pytest, real Postgres/MQTT, mock plugins, validate full pipeline including shutdown
14. **Config validation**: YAML schema validation on startup, fail fast on errors
