# HomeSec Development Guidelines

**Last reviewed:** 2026-04-25
**Purpose:** Critical patterns to prevent runtime bugs when extending HomeSec. For architecture overview, see `DESIGN.md`.

---

## Absolute Rules

- **Strict type checking required**: Run `make typecheck` before committing. Error-as-value pattern requires explicit type narrowing via match/isinstance.
- **Program to interfaces**: Use factory/registry helpers (e.g., `load_filter()`). Avoid direct instantiation of plugins.
- **Architecture constraints**: See `DESIGN.md` → “Architecture Constraints”. Boundary violations (core ↔ concrete plugins, backend-specific config in core) are bugs.
- **Repository pattern**: Use `ClipRepository` for all state/event writes. Never touch `StateStore`/`EventStore` directly.
- **Preserve stack traces**: Custom errors must set `self.__cause__ = cause` to preserve original exception.
- **Tests must use Given/When/Then comments**: Every test must include these comments and follow behavioral testing principles (see `TESTING.md`).
- **Run `make check` before pushing code or opening/updating a PR**: Targeted checks are not enough for publish flow unless the user explicitly asks for a narrower validation scope.
- **PR titles must satisfy repo CI**: Use semantic PR titles such as `fix: ...`, `refactor: ...`, or `docs: ...`. Do not use `[codex] ...` in this repository because CI rejects non-semantic PR titles.
- **Respect `uv.lock`**: Validation commands run with `--locked`. If `uv` reports the lockfile is stale, run `uv lock`, inspect the diff, and commit the lockfile sync intentionally. Do not revert `uv.lock` churn without checking whether `pyproject.toml` or dependency metadata changed.
- **Import from canonical source**: Import types from their defining module, not re-exports. For example, import `RiskLevel` from `models.enums`, not from `models.vlm`. Avoid creating re-exports in `__all__`.
- **Postgres for state**: Use `clip_states` table with `clip_id` (primary key) + `data` (jsonb) for evolvable schema.
- **Pydantic everywhere**: Validate config, DB payloads, VLM outputs, and MQTT payloads with Pydantic models.
- **Clarify before complexity**: Ask user for clarification when simpler design may exist. Don't proceed with complex workarounds.
- **Product priorities**: Recording + uploading (P0) must work even if Postgres is down. Analysis/notifications are best-effort (P1).

---

## Media Feature Changes

**Rule:** Keep media transport and camera protocol details behind runtime/source
boundaries. This applies to preview, push-to-talk, camera media sessions, RTSP,
RTP, SDP, codecs, browser microphone/audio capture, and media auth tokens.

- API routes must not import camera protocol implementation modules directly. They
  should work through application/runtime interfaces and typed API models.
- UI code must not know camera protocol details. Browser code should handle
  browser-safe media concerns such as microphone capture, PCM framing, playback,
  backpressure, and display state.
- Runtime/controller code owns active runtime selection, command routing, and IPC
  stream bridging. Source implementations own camera-specific media lifecycle.
- Credentials, API keys, media tokens, and raw audio/video frames must not be
  logged or persisted. Logs may include redacted URLs, env var names, stable
  error codes, and low-cardinality operational context.
- Session lifecycles must be bounded and must clean up on explicit stop,
  disconnect, failure, cancellation, and timeout. Concurrency and camera session
  budget behavior must be explicit in code and covered by behavioral tests or
  operator docs.
- Cross-boundary payloads must use typed Pydantic/API models rather than raw
  dictionaries. Preserve stable refusal/error reasons for UI and automation.
- Protocol, codec, or transport changes need behavioral coverage at the protocol
  boundary, preferably with fake-camera/server fixtures. Update operator docs or
  troubleshooting notes when deployment behavior changes.
- Push-to-talk UI changes must preserve exact PCM frame sizing, backpressure
  handling, microphone-denial behavior, and preview audio mute coordination while
  talk is active.

---

## Logging Guidance

**Rule:** Use regular Python logging for operational logs, and structured extras for queryable telemetry. Logging is best-effort observability; use `ClipRepository` for workflow state and durable events.

**Structured logging conventions:**
- Use normal log levels (`debug`, `info`, `warning`, `error`) for severity.
- Put queryable context in `extra={...}` instead of packing metadata into the message string.
- For telemetry events, include a stable low-cardinality `event_type` such as `vlm_usage`, `recording_started`, or `upload_failed`.
- Treat `event_type` as the canonical event marker. DB filtering and payload shaping classify any truthy `event_type` as `kind="event"`.
- `kind="event"` is optional when `event_type` is present. Never set `kind="log"` on a record that has `event_type`.
- Do not manually add standard `LogRecord` fields such as `message`, `asctime`, `levelname`, `pathname`, or `lineno` in `extra`.
- Prefer the existing logging context injection for `camera_name` and `recording_id`; only pass them explicitly when the caller is outside that context and has accurate values.
- Never log secrets, tokens, credentials, or full DSNs. Log secret source names such as env var names when needed.

**Template: Telemetry Event Log**

```python
logger.info(
    "VLM usage recorded",
    extra={
        "event_type": "vlm_usage",
        "camera_name": camera_name,
        "recording_id": clip_id,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "total_tokens": usage.total_tokens,
    },
)
```

**Template: Operational Log**

```python
logger.warning(
    "Upload failed; continuing without storage URI",
    extra={"recording_id": clip_id, "storage_backend": backend},
    exc_info=exc,
)
```

---

## Core Pattern Templates

### 1. Error-as-Value + Type Narrowing

**Rule:** For partial failures, return `Result | ErrorType` instead of raising exceptions. Always narrow types with match or isinstance.

**✅ Template: Error-as-Value with Type Narrowing**

```python
# Define custom error that preserves stack traces
class FilterError(PipelineError):
    def __init__(self, clip_id: str, plugin_name: str, cause: Exception):
        super().__init__(
            f"Filter failed for {clip_id}", stage="filter", clip_id=clip_id, cause=cause
        )
        self.plugin_name = plugin_name
        # PipelineError sets cause and __cause__, preserving the original exception.

# Stage returns error as value, not raise
async def _filter_stage(self, clip: Clip) -> FilterResult | FilterError:
    try:
        async with self._sem_filter:
            return await self._filter.detect(clip.local_path)
    except Exception as e:
        return FilterError(clip.clip_id, self._config.filter.backend, cause=e)

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

# Repository coordinates separate best-effort writes:
# 1. Loads and upserts the clip snapshot
# 2. Appends FilterCompletedEvent(...) independently
# 3. Retries retryable failures with exponential backoff; logs exhaustion
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

**Why:** Repository centralizes transitions, retry handling, and lifecycle-event calls. Snapshot upserts and event appends are separate best-effort operations, not an atomic transaction. Events can be missing after state advances, and concurrent whole-snapshot updates can overwrite one another. See [current persistence limitations](docs/reference.md#analysis-alerts); using the repository does not establish durable replay or exactly-once processing.

**Reference:** See `src/homesec/repository/clip_repository.py` for all available methods (`record_upload_completed`, `record_vlm_started`, etc.).

---

### 4. Plugin Registration

**Rule:** Use the existing unified registry in `src/homesec/plugins/registry.py`.
Plugins register a class with `@plugin(plugin_type=PluginType.<TYPE>, name="...")`,
a Pydantic `config_cls`, and a typed `create()` classmethod. Do not introduce a
second registry or copy the former per-category decorator/factory API.

Use the existing category loaders instead of directly constructing plugins:

| Plugin type | Loader | Registry type |
| --- | --- | --- |
| Filters | `load_filter(config)` | `PluginType.FILTER` |
| Analyzers | `load_analyzer(config)` | `PluginType.ANALYZER` |
| Sources | `load_source_plugin(backend, config, camera_name, **runtime_context)` | `PluginType.SOURCE` |
| Alert policies | `load_alert_policy(config, trigger_classes)` | `PluginType.ALERT_POLICY` |
| Notifiers | `load_notifier_plugin(backend, config)` | `PluginType.NOTIFIER` |
| Storage | `load_storage_plugin(config)` | `PluginType.STORAGE` |

The registry merges supplied runtime context into configuration before Pydantic
validation and calls the class factory with the validated model. Factory
signatures and required abstract methods must match the canonical interfaces;
there are no separate `FilterContext`/`SourceContext` factory arguments in this
registry.

Built-ins are discovered through module imports; external packages use the
`homesec.plugins` entry-point group. See [PLUGIN_DEVELOPMENT.md](PLUGIN_DEVELOPMENT.md)
and the existing filter/notifier implementations for complete examples.

---

### 5. Testing Requirements

**Rule:** All tests must use Given/When/Then comments. Follow principles in `TESTING.md` for writing high-quality tests.

**Key Principles (see `TESTING.md` for full details):**
- **Test observable behavior, not internal state** - Never assert on `obj._private_attr`
- **Mock at the boundary** - Mock HTTP/network, not internal methods
- **Use contract testing** - Capture and verify API request structure
- **Use real implementations where cheap** - Filesystem via `tmp_path`, pure functions
- **Track API calls, not state flags** - Use `api_calls: list[str]` not `was_called: bool`

Use the current [testing examples](TESTING.md) and
[pipeline tests](tests/homesec/test_pipeline.py) for complete payloads, fixture
construction, and behavioral boundaries. Stage helpers above illustrate error
handling; they do not replace tests through the appropriate observable entry
point. Avoid copying abbreviated constructors or inventing mock arguments that
the current test helpers do not support.

**Available Mocks:** `MockFilter`, `MockVLM`, `MockStorage`, `MockNotifier`, `MockStateStore`, `MockEventStore`

**Reference:** See `TESTING.md` for comprehensive testing guidelines and anti-patterns to avoid.

---

## Project Context

### Key Directories

**Source code:**
- `src/homesec/interfaces.py` - Abstract interfaces and protocols for pipeline components
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
make check              # Run lock, Rust, Python, and UI checks
make db                 # Start Postgres for development
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

```yaml
# ✅ GOOD: Reference env var name in config
storage:
  backend: dropbox
  config:
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
