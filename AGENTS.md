# HomeSec Development Guidelines

**Last reviewed:** 2026-01-18
**Purpose:** Critical patterns to prevent runtime bugs when extending HomeSec. For architecture overview, see `DESIGN.md`.

---

## Absolute Rules

- **Strict type checking required**: Run `make typecheck` before committing. Error-as-value pattern requires explicit type narrowing via match/isinstance.
- **Program to interfaces**: Use factory/registry helpers (e.g., `load_filter_plugin()`). Avoid direct instantiation of plugins.
- **Repository pattern**: Use `ClipRepository` for all state/event writes. Never touch `StateStore`/`EventStore` directly.
- **Preserve stack traces**: Custom errors must set `self.__cause__ = cause` to preserve original exception.
- **Tests must use Given/When/Then comments**: Every test must include these comments and follow behavioral testing principles (see `TESTING.md`).
- **Import from canonical source**: Import types from their defining module, not re-exports. For example, import `RiskLevel` from `models.enums`, not from `models.vlm`. Avoid creating re-exports in `__all__`.
- **Postgres for state**: Use `clip_states` table with `clip_id` (primary key) + `data` (jsonb) for evolvable schema.
- **Pydantic everywhere**: Validate config, DB payloads, VLM outputs, and MQTT payloads with Pydantic models.
- **Clarify before complexity**: Ask user for clarification when simpler design may exist. Don't proceed with complex workarounds.
- **Product priorities**: Recording + uploading (P0) must work even if Postgres is down. Analysis/notifications are best-effort (P1).

---

## Core Pattern Templates

### 1. Error-as-Value + Type Narrowing

**Rule:** For partial failures, return `Result | ErrorType` instead of raising exceptions. Always narrow types with match or isinstance.

**✅ Template: Error-as-Value with Type Narrowing**

```python
# Define custom error that preserves stack traces
class FilterError(PipelineError):
    def __init__(self, clip_id: str, plugin_name: str, cause: Exception):
        super().__init__(f"Filter failed for {clip_id}", stage="filter", clip_id=clip_id)
        self.plugin_name = plugin_name
        self.__cause__ = cause  # ← Preserves full stack trace

# Stage returns error as value, not raise
async def _filter_stage(self, clip: Clip) -> FilterResult | FilterError:
    try:
        async with self._sem_filter:
            return await self._filter.detect(clip.local_path)
    except Exception as e:
        return FilterError(clip.clip_id, self._config.filter.plugin, cause=e)

# Caller uses match for type narrowing
filter_result = await self._filter_stage(clip)

match filter_result:
    case FilterError() as err:
        # Type narrowed to FilterError
        logger.error("Filter failed: %s", err.cause, exc_info=err.cause)
        await self._repository.record_filter_failed(...)
        return  # Abort on critical failure

    case FilterResult() as result:
        # Type narrowed to FilterResult
        logger.info("Detected: %s", result.detected_classes)
        await self._repository.record_filter_completed(...)

# Partial failure example: upload fails but filter succeeds
match await self._upload_stage(clip):
    case UploadError() as err:
        logger.warning("Upload failed (continuing): %s", err.cause)
        upload_failed = True
    case UploadOutcome() as outcome:
        storage_uri = outcome.storage_uri
        upload_failed = False

# Continue processing even though upload failed
match await self._filter_stage(clip):
    case FilterError():
        return  # Critical failure - abort
    case FilterResult():
        # Can still notify with upload_failed=True
        pass
```

**❌ Anti-Pattern: Raising Exceptions**

```python
# BAD: Raises on failure, can't handle partial failures
async def _filter_stage(self, clip: Clip) -> FilterResult:
    return await self._filter.detect(clip.local_path)  # Raises!

# Caller can't distinguish upload vs filter failures
try:
    await self._upload_stage(clip)
    await self._filter_stage(clip)
except Exception:
    # Both failed? Just one? Can't tell!
    return
```

**❌ Anti-Pattern: Missing Type Narrowing**

```python
# BAD: No type narrowing - runtime crash
result = await self._filter_stage(clip)  # Type: FilterResult | FilterError
logger.info("Detected: %s", result.detected_classes)  # ← CRASH if FilterError!

# GOOD: Use isinstance if not using match
if isinstance(result, FilterError):
    logger.error("Failed: %s", result.cause)
    return
# Type narrowed to FilterResult here
logger.info("Detected: %s", result.detected_classes)  # ✅ Safe
```

**When to use exceptions vs error-as-value:**

- **Exceptions**: Programmer errors (ValueError, TypeError), unrecoverable failures (out of disk), abort-entire-operation
- **Error-as-value**: Partial failures (upload fails but continue), expected failures (network timeout), fine-grained error handling

---

### 2. Async Pattern Selection

**Rule:** Choose async pattern based on whether work is CPU-bound, I/O-bound, or blocking sync.

**Quick Reference:**

| Work Type | Pattern | Example |
|-----------|---------|---------|
| CPU/GPU-bound | `ProcessPoolExecutor` | YOLO inference, video processing |
| I/O-bound | `aiohttp` (async library) | OpenAI API calls, webhooks |
| Blocking sync | `run_in_executor(None, fn)` | Dropbox SDK, file I/O |

**Template: CPU-Bound (ProcessPoolExecutor)**

```python
class YOLOv8Filter:
    def __init__(self, config: FilterConfig):
        self._executor = ProcessPoolExecutor(max_workers=config.max_workers)

    async def detect(self, video_path: Path) -> FilterResult:
        loop = asyncio.get_running_loop()
        # Worker must be module-level function (picklable)
        return await loop.run_in_executor(
            self._executor,
            _detect_worker,
            str(video_path),  # Args must be picklable
        )

    async def shutdown(self, timeout: float | None = None):
        self._executor.shutdown(wait=True, cancel_futures=False)

# Module-level worker for pickle
def _detect_worker(video_path: str) -> FilterResult:
    import torch
    model = YOLO("model.pt").to("cuda" if torch.cuda.is_available() else "cpu")
    # ... inference ...
    return FilterResult(detected_classes=["person"], confidence=0.9)
```

**Template: I/O-Bound (aiohttp)**

```python
class OpenAIVLM:
    def __init__(self, config: VLMConfig):
        self._session: aiohttp.ClientSession | None = None

    async def analyze(self, video_path: Path) -> AnalysisResult:
        if self._session is None:
            timeout = aiohttp.ClientTimeout(total=30.0)
            self._session = aiohttp.ClientSession(timeout=timeout)

        async with self._session.post(url, json=payload, headers=headers) as resp:
            data = await resp.json()

        return AnalysisResult.model_validate(data)

    async def shutdown(self, timeout: float | None = None):
        if self._session:
            await self._session.close()
```

**Template: Blocking Sync (run_in_executor)**

```python
class DropboxStorage:
    async def put_file(self, local_path: Path, dest: str) -> StorageUploadResult:
        def _upload():
            with local_path.open("rb") as f:
                return self._dbx.files_upload(f.read(), dest)  # Blocking SDK call

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, _upload)  # None = default ThreadPoolExecutor

        return StorageUploadResult(storage_uri=f"dropbox:{result.path_display}")
```

---

### 3. Repository Pattern

**Rule:** Use `ClipRepository` for all state/event persistence. Never touch `StateStore`/`EventStore` directly.

**✅ Template: Use Repository**

```python
# In pipeline - use repository methods
await self._repository.record_filter_completed(
    clip_id=clip.clip_id,
    result=filter_result,
    duration_ms=int(duration * 1000),
)

# Repository coordinates state + event atomically:
# 1. Updates state.filter_result = result
# 2. Appends FilterCompletedEvent(...)
# 3. Handles retries with exponential backoff
```

**❌ Anti-Pattern: Direct State/Event Manipulation**

```python
# BAD: Manually updating state and events
state = await self._state_store.get(clip_id)
state.filter_result = result
await self._state_store.upsert(clip_id, state)

event = FilterCompletedEvent(...)
await self._event_store.append(event)  # What if this fails? State already updated!
```

**Why:** Repository ensures state and events stay consistent, provides retry logic, and prevents forgetting to emit events.

**Reference:** See `src/homesec/repository/clip_repository.py` for all available methods (`record_upload_completed`, `record_vlm_started`, etc.).

---

### 4. Plugin Registration

**Rule:** Use decorator-based registration with typed factories. All plugins follow the same pattern: `@<type>_plugin` decorator + `load_<type>_plugin()` loader.

**✅ Template: Decorator-Based Registration**

```python
# In src/homesec/plugins/notifiers/__init__.py
from dataclasses import dataclass
from pydantic import BaseModel

@dataclass(frozen=True)
class NotifierPlugin:
    name: str
    config_model: type[BaseModel]
    factory: Callable[[BaseModel], Notifier]

NOTIFIER_REGISTRY: dict[str, NotifierPlugin] = {}

def register_notifier(plugin: NotifierPlugin) -> None:
    """Register with collision detection."""
    if plugin.name in NOTIFIER_REGISTRY:
        raise ValueError(f"Plugin '{plugin.name}' already registered")
    NOTIFIER_REGISTRY[plugin.name] = plugin

def notifier_plugin(name: str) -> Callable[[T], T]:
    """Decorator to register a notifier plugin."""
    def decorator(factory_fn: T) -> T:
        plugin = factory_fn()
        register_notifier(plugin)
        return factory_fn
    return decorator

def load_notifier_plugin(backend: str, config: dict | BaseModel) -> Notifier:
    """Load and instantiate a notifier plugin with config validation."""
    plugin = NOTIFIER_REGISTRY[backend.lower()]
    validated = plugin.config_model.model_validate(config) if isinstance(config, dict) else config
    return plugin.factory(validated)
```

**✅ Template: Plugin Implementation with Type Safety**

```python
# In src/homesec/plugins/notifiers/mqtt.py
from pydantic import BaseModel
from homesec.plugins.notifiers import NotifierPlugin, notifier_plugin

class MQTTNotifierConfig(BaseModel):
    broker_url: str
    topic_prefix: str = "homesec"

@notifier_plugin(name="mqtt")
def mqtt_notifier_plugin() -> NotifierPlugin:
    def factory(cfg: BaseModel) -> Notifier:
        # Type-safe: validate config type at runtime
        if not isinstance(cfg, MQTTNotifierConfig):
            raise TypeError(f"Expected MQTTNotifierConfig, got {type(cfg).__name__}")
        return MQTTNotifier(cfg)

    return NotifierPlugin(
        name="mqtt",
        config_model=MQTTNotifierConfig,
        factory=factory,
    )
```

**Plugin Types and Context Objects:**

| Plugin Type | Decorator | Context | Factory Signature |
|-------------|-----------|---------|-------------------|
| Filters | `@filter_plugin` | `FilterContext` | `(BaseModel, FilterContext) -> ObjectFilter` |
| Analyzers | `@vlm_plugin` | `VLMContext` | `(BaseModel, VLMContext) -> VLMAnalyzer` |
| Sources | `@source_plugin` | `SourceContext` | `(BaseModel, SourceContext) -> ClipSource` |
| Alert Policies | `@alert_policy_plugin` | `AlertPolicyContext` | `(BaseModel, AlertPolicyContext) -> AlertPolicy` |
| Notifiers | `@notifier_plugin` | None | `(BaseModel) -> Notifier` |
| Storage | `@storage_plugin` | None | `(BaseModel) -> StorageBackend` |

**Why some plugins have Context and others don't:**
- **With Context** (Filters, Analyzers, Sources, AlertPolicies): These plugins need runtime information beyond their config—camera names, executor pools, file paths, or pipeline state. Context objects provide this without polluting config models.
- **Without Context** (Notifiers, Storage): These are stateless services that only need their own config. They don't depend on which camera or pipeline invokes them.

**Note:** Always use `isinstance()` checks in factories instead of `cast()` to ensure type safety at runtime.

**Reference:** See `PLUGIN_DEVELOPMENT.md` for complete guide. Example implementations: `src/homesec/plugins/filters/yolo.py`, `src/homesec/plugins/notifiers/mqtt.py`.

---

### 5. Testing Requirements

**Rule:** All tests must use Given/When/Then comments. Follow principles in `TESTING.md` for writing high-quality tests.

**Key Principles (see `TESTING.md` for full details):**
- **Test observable behavior, not internal state** - Never assert on `obj._private_attr`
- **Mock at the boundary** - Mock HTTP/network, not internal methods
- **Use contract testing** - Capture and verify API request structure
- **Use real implementations where cheap** - Filesystem via `tmp_path`, pure functions
- **Track API calls, not state flags** - Use `api_calls: list[str]` not `was_called: bool`

**✅ Template: Test with Given/When/Then**

```python
async def test_filter_stage_success():
    # Given: A clip and a filter configured to detect person
    clip = Clip(clip_id="test-001", camera_name="front_door", ...)
    mock_filter = MockFilter(result=FilterResult(detected_classes=["person"], confidence=0.9))
    pipeline = ClipPipeline(filter_plugin=mock_filter, ...)

    # When: Processing clip through filter stage
    result = await pipeline._filter_stage(clip)

    # Then: Should return FilterResult with detected person
    assert isinstance(result, FilterResult)
    assert "person" in result.detected_classes
    assert result.confidence == 0.9

async def test_filter_stage_failure():
    # Given: A filter that simulates failure
    mock_filter = MockFilter(simulate_failure=True)
    pipeline = ClipPipeline(filter_plugin=mock_filter, ...)

    # When: Processing clip through filter stage
    result = await pipeline._filter_stage(clip)

    # Then: Should return FilterError (error-as-value pattern)
    assert isinstance(result, FilterError)
    assert result.clip_id == clip.clip_id
```

**Available Mocks:** `MockFilter`, `MockVLM`, `MockStorage`, `MockNotifier`, `MockStateStore`, `MockEventStore`

**Reference:** See `TESTING.md` for comprehensive testing guidelines and anti-patterns to avoid.

---

## Project Context

### Key Directories

**Source code:**
- `src/homesec/interfaces.py` - Protocol definitions for all plugin types
- `src/homesec/pipeline/core.py` - ClipPipeline orchestrator (main business logic)
- `src/homesec/app.py` - Application class (component wiring and lifecycle)
- `src/homesec/models/` - Pydantic models (Config, ClipStateData, events, etc.)
- `src/homesec/plugins/` - Plugin implementations (filters, VLMs, notifiers, storage, alert_policies)
- `src/homesec/repository/` - ClipRepository for coordinated state + event writes
- `src/homesec/state/` - StateStore and EventStore implementations
- `src/homesec/sources/` - Clip source implementations (RTSP, FTP, LocalFolder)

**Tests:**
- `tests/homesec/mocks/` - Mock implementations of all interfaces
- `tests/homesec/test_pipeline.py` - Comprehensive pipeline tests
- `tests/homesec/test_integration.py` - End-to-end tests

**Legacy (ignore for new development):**
- `src/motion_recorder.py`, `src/ftp_dropbox_server.py`, `src/human_filter/`, `src/evals/` - Deprecated

### Build & Test Commands

```bash
uv sync                 # Install dependencies
make typecheck          # Run mypy --strict (mandatory before commit)
make test               # Run pytest
make check              # Run both typecheck + test
make db-up              # Start Postgres for development
make db-migrate         # Run Alembic migrations
```

### Pattern Usage Examples

**Error-as-value:** `src/homesec/pipeline/core.py::_filter_stage()`, `_upload_stage()`, `_vlm_stage()`
**Async patterns:** `src/homesec/plugins/filters/yolo.py`, `src/homesec/plugins/analyzers/openai.py`, `src/homesec/plugins/storage/dropbox.py`
**Repository:** `src/homesec/pipeline/core.py` (search for `self._repository.record_`)
**Plugin registration:** `src/homesec/plugins/notifiers/__init__.py`, `src/homesec/plugins/alert_policies/__init__.py`
**Testing:** `tests/homesec/test_pipeline.py`

---

## Style & Security

### Modern Python (3.10+)

- **Match statements** for pattern matching (preferred over if/elif with isinstance)
- **Union types** with `|` instead of `Union` (e.g., `str | None` not `Optional[str]`)
- **Type narrowing** via match or isinstance for union types
- **Type hints everywhere** - `make typecheck` enforces this

### Naming Conventions

- `snake_case` for functions/variables/modules
- `PascalCase` for classes
- `UPPER_SNAKE_CASE` for constants
- 4-space indentation

### Import Organization

- **All imports at top of file** - Never use local/inline imports except when absolutely necessary to avoid circular imports
- **Standard ordering**: stdlib → third-party → homesec (ruff handles this with `ruff check --fix`)
- **No re-imports at end of file** - Plugin registration decorators should use imports from the top
- **Circular import avoidance**: Use `TYPE_CHECKING` blocks for type-only imports when needed

### Security

```python
# ✅ GOOD: Reference env var name in config
storage:
  backend: dropbox
  dropbox:
    token_env: "DROPBOX_TOKEN"  # Config stores env var NAME

# ❌ BAD: Never commit secrets
token: "sl.ABC123..."  # Don't do this!
```

- Never commit: `.env`, tokens, RTSP credentials, database DSNs
- Validate external inputs with Pydantic before processing
- Never log secrets - redact sensitive data in error messages

### Commit Style

- Short, task-focused messages (follow repo history)
- Optional prefixes: `docs:`, `fix:`, `test:`
- Include issue refs: `(#42)`
- Run `make check` before committing

---

## Quick Reference Links

**Architecture & Design:**
- See `DESIGN.md` for full architecture overview and design decisions
- See `TESTING.md` for comprehensive testing guidelines and principles
- See `src/homesec/interfaces.py` for complete interface definitions

**Plugin Development:**
- Notifier example: `src/homesec/plugins/notifiers/mqtt.py`
- Filter example: `src/homesec/plugins/filters/yolo.py`
- VLM example: `src/homesec/plugins/analyzers/openai.py`
- Storage example: `src/homesec/plugins/storage/dropbox.py`
- Alert policy example: `src/homesec/plugins/alert_policies/default.py`

**Testing Examples:**
- Pipeline tests: `tests/homesec/test_pipeline.py`
- Mock usage: `tests/homesec/conftest.py`
- Integration tests: `tests/homesec/test_integration.py`
