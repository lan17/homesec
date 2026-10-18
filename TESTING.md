# HomeSec Testing Guidelines

**Last reviewed:** 2026-10-18
**Purpose:** Principles for writing high-quality, maintainable tests that verify behavior without creating false confidence.

---

## Commands and Test Environment

Run from the repository root after installing the locked dependencies. Current
commands come from [Makefile](Makefile), [ui/Makefile](ui/Makefile), and
[CI.md](CI.md):

| Command | Scope |
|---------|-------|
| `make typecheck lint lock-check` | Strict Python typing, Ruff lint/format, lock consistency |
| `make test` | `tests/homesec/` with terminal coverage |
| `make coverage` | `tests/homesec/` with HTML/XML coverage |
| `make rust-check` | Rust formatting, Clippy, and native tests through the dependency wrapper |
| `make ui-check` | Generated API consistency, ESLint, Vitest, TypeScript, and production build |
| `make check` | Lock, Rust, Python, and UI gates |

The Python suite includes real Postgres fixtures. `TEST_DB_DSN` selects their
database, defaulting to the local development Postgres service. See
[postgres_test_support.py](tests/homesec/postgres_test_support.py) and
[conftest.py](tests/homesec/conftest.py). A per-run schema is normally created and
dropped, and fixtures recreate tables/write rows. `HOMESEC_TEST_DB_SCHEMA` can
select a schema explicitly; schema isolation does not make tests read-only.
Use an appropriate disposable database and obtain the required approval before
database-writing validation. `make db` starts the development Postgres service;
there is no `make db-up` target.

CI support tests live outside the application target; run
`uv run --locked python .github/scripts/test_ci.py` when changing that support
code. Some Python tests also require FFmpeg/ffprobe or the built
`homesec-webrtc` helper. Native tests use
fixtures/fake peers; provider and physical-camera conformance require separate
validation. Inspect test fixtures before treating a selected suite as offline.
CI installs those prerequisites and runs migrations on its disposable Postgres.

Repository policy requires `make check` before publication unless Leonidas
explicitly approves a narrower scope. Structural OpenSpec validation and a clean
generated API diff do not establish runtime conformance; link tests to relevant
scenarios and state any untested provider/device behavior.

The examples below illustrate testing patterns; some omit setup/imports for
brevity. Linked repository tests supply complete current payloads and fixtures.

---

## TL;DR - The 5 Rules

1. **Test behavior, not internals** - Assert on return values and exceptions, never on `obj._private_attr`
2. **Mock at boundaries** - Mock HTTP/network calls, not internal methods like `_extract_frames()`
3. **Use Given/When/Then** - Every test needs these comments; they enforce behavioral thinking
4. **Prefer real over fake** - Use `tmp_path` for files, real entry points, real pure functions
5. **Capture API contracts** - When mocking, verify request structure, not just "was called"

---

## Core Principles

### 1. Test Observable Behavior, Not Internal State

**Rule:** Verify what the code *does*, not how it *does it*. Tests should pass if behavior is correct, even if implementation changes.
If the only public surface is a library callback or hook, treat that callback as the boundary and assert on observable effects.

**Bad - Testing internal state:**
```python
async def test_shutdown_is_idempotent(self):
    notifier = MQTTNotifier(config)
    await notifier.shutdown()

    # BAD: Checking private attribute
    assert notifier._shutdown_called is True
    assert notifier._connected is False
```

**Good - Testing observable behavior:**
```python
async def test_shutdown_is_idempotent(self):
    # Given: A notifier using a fake MQTT boundary.
    notifier = MQTTNotifier(config)

    # When: Shutting down twice.
    await notifier.shutdown()
    await notifier.shutdown()  # Should not raise

    # Then: Shutdown is observable through public operations.
    result = await notifier.ping()
    assert result is False  # Shutdown state observable via ping

    # GOOD: Verify operations fail as expected
    with pytest.raises(RuntimeError, match="shut down"):
        await notifier.send(alert)
```

---

### 2. Mock at the Boundary, Not Internal Methods

**Rule:** Mock external dependencies (HTTP, databases) at their boundary, not internal methods. This tests more of your actual code.
Filesystem is a special case: prefer real `tmp_path` usage when cheap and deterministic.

**Bad - Mocking internal methods:**
```python
async def test_analyze_video(self):
    vlm = OpenAIVLM(config)

    # BAD: Mocking internal implementation details
    vlm._extract_frames = MagicMock(return_value=[...])
    vlm._resize_image = MagicMock(return_value=...)
    vlm._call_api = AsyncMock(return_value={"response": "..."})

    result = await vlm.analyze(video_path)
```

**Good - Mocking at HTTP boundary:**

[test_openai_vlm.py](tests/homesec/test_openai_vlm.py) creates a real tiny video
and mocks `homesec.plugins.analyzers.openai.aiohttp.ClientSession`. The real
analyzer extracts frames and builds the request; the fake captures URL, headers,
and JSON. It returns a complete `SequenceAnalysis` payload in the provider's
response envelope, so parsing and required-field validation remain exercised.

The current call is `await analyzer.analyze(video_path, filter_result, vlm_config)`;
construction takes the backend's `OpenAIConfig`, while the call takes the core
`VLMConfig`. Assert enum values such as `RiskLevel.LOW`, rather than assuming risk
levels are plain strings. An incomplete `{"risk_level": "low"}` response does not
satisfy the current provider schema.

---

### 3. Use Contract Testing for API Calls

**Rule:** When mocking external services, verify that your code calls them correctly by capturing and asserting on the request structure.

**Bad - Just checking call count:**
```python
async def test_sends_email(self):
    notifier = SendGridEmailNotifier(config)
    await notifier.send(alert)

    # BAD: Only verifies something was called
    mock_session.post.assert_called_once()
```

**Good - Verifying request contract:**

The bundled [SendGrid email notifier](src/homesec/plugins/notifiers/sendgrid_email.py)
uses `SendGridEmailNotifier` and `SendGridEmailConfig`. Its complete
[HTTP contract tests](tests/homesec/test_sendgrid_notifier.py) replace the
`aiohttp.ClientSession` boundary, capture URL/headers/JSON, and assert the
recipient, sender, subject, content, and optional structured-analysis rendering.
The real notifier still builds the request.

Use those tests as the current fixture/payload example. Assert the request data
that matters to the provider contract rather than only counting `post()` calls.
For the MQTT boundary, see
[test_mqtt_notifier.py](tests/homesec/test_mqtt_notifier.py).

---

### 4. Use Real Implementations Where Cheap

**Rule:** Prefer real implementations over mocks when they're fast, deterministic, and don't require external resources.

**Use real:**
- Filesystem operations via `tmp_path` fixture
- In-memory data structures
- Pure functions (validators, formatters, parsers)
- Entry points behavior via stubbed `entry_points()` (deterministic; no packaging dependency)

**Mock:**
- Network calls (HTTP, MQTT, etc.)
- Databases for component/unit tests; use the existing isolated Postgres fixtures
  for database contracts rather than assuming the suite uses testcontainers
- GPU/ML inference
- Time-sensitive operations

**Example - Using real filesystem:**
```python
async def test_put_get_delete_roundtrip(self, tmp_path: Path):
    # Given: Input bytes and real local storage.
    # GOOD: Real filesystem via tmp_path
    storage = LocalStorage(LocalStorageConfig(root=str(tmp_path / "storage")))

    source_file = tmp_path / "source.mp4"
    source_file.write_bytes(b"video content")

    try:
        # When: Uploading, downloading, and deleting through the public interface.
        result = await storage.put_file(source_file, "clips/test.mp4")
        download_path = tmp_path / "downloaded.mp4"
        await storage.get(result.storage_uri, download_path)

        # Then: Storage preserves bytes and deletion is idempotent.
        assert download_path.read_bytes() == b"video content"
        assert await storage.exists(result.storage_uri)
        await storage.delete(result.storage_uri)
        await storage.delete(result.storage_uri)
        assert not await storage.exists(result.storage_uri)
    finally:
        await storage.shutdown()
```

**Example - Using stubbed entry points (deterministic):**
```python
def test_returns_stubbed_entry_points(self, monkeypatch):
    # Given: Deterministic entry-point metadata.
    # GOOD: Deterministic entry points without relying on installation metadata
    fake_eps = [
        metadata.EntryPoint(name="homesec", value="homesec.cli:main", group="console_scripts")
    ]

    class _FakeEntryPoints(list):
        def select(self, group: str):
            return [ep for ep in self if ep.group == group]

    monkeypatch.setattr(
        "homesec.plugins.utils.metadata.entry_points",
        lambda: _FakeEntryPoints(fake_eps),
    )

    # When: Selecting entry points by group.
    result = list(iter_entry_points("console_scripts"))

    # Then: Only the matching registered command is returned.
    assert [ep.name for ep in result] == ["homesec"]
```

---

### 5. Track API Calls, Not State Flags

**Rule:** When faking external clients, track what methods were called rather than setting internal state flags.

**Bad - State flags in fake:**
```python
class _FakeDropboxClient:
    def __init__(self):
        self.session_started = False
        self.session_finished = False
        self.session_appends = 0

    def files_upload_session_start(self, chunk):
        self.session_started = True  # Internal state flag
        return _FakeSession("session_1")

# Test checks internal state
assert client.session_started is True
assert client.session_appends >= 1
```

**Good - Track API calls:**
```python
class _FakeDropboxClient:
    def __init__(self):
        self.api_calls: list[str] = []  # Track what was called

    def files_upload_session_start(self, chunk):
        self.api_calls.append("files_upload_session_start")
        return _FakeSession("session_1")

    def files_upload_session_append_v2(self, chunk, cursor):
        self.api_calls.append("files_upload_session_append_v2")
        cursor.offset += len(chunk)

# Test verifies contract
assert "files_upload_session_start" in client.api_calls
assert "files_upload_session_append_v2" in client.api_calls
assert "files_upload_session_finish" in client.api_calls
```

---

### 6. Handle Global State Properly

**Rule:** Use fixtures to save and restore global state between tests. Never let one test pollute another.

**Bad - Tests pollute each other:**
```python
def test_sets_camera_name(self):
    set_camera_name("front_door")
    # Global state now modified for all subsequent tests!
```

**Good - Fixture restores state:**
```python
@pytest.fixture(autouse=True)
def reset_logging_state():
    """Reset global logging state before each test."""
    import homesec.logging_setup as module

    original_camera = module._CURRENT_CAMERA_NAME
    original_recording = module._CURRENT_RECORDING_ID

    yield

    # Restore after test completes
    module._CURRENT_CAMERA_NAME = original_camera
    module._CURRENT_RECORDING_ID = original_recording

def test_sets_camera_name(self):
    set_camera_name("front_door")
    # State automatically restored after test
```

---

### 7. Test Names Must Match Reality

**Rule:** If the test name says it tests something specific, the test must actually test that thing.

**Bad - Misleading test name:**
```python
async def test_handles_unicode_filenames(self):
    # Claims to test unicode but uses ASCII!
    result = await storage.put_file(source, "clips/test_file.mp4")
    assert await storage.exists(result.storage_uri)
```

**Good - Test matches name:**
```python
async def test_handles_unicode_filenames(self):
    # Actually tests unicode (Cyrillic and Japanese)
    result = await storage.put_file(source, "clips/камера_передняя.mp4")
    assert await storage.exists(result.storage_uri)

    result2 = await storage.put_file(source, "clips/玄関カメラ.mp4")
    assert await storage.exists(result2.storage_uri)
```

---

## Given/When/Then Structure

**Rule:** All tests must use Given/When/Then comments. This structure enforces behavioral thinking and makes tests self-documenting.

### Why Given/When/Then?

The pattern forces you to think about tests as **behavior specifications**:
- **Given** = Preconditions (setup state, create dependencies)
- **When** = Action (the single thing being tested)
- **Then** = Postconditions (observable outcomes)

This naturally leads to behavioral testing because:
1. **Given** focuses on *what state exists*, not *how to create internal state*
2. **When** tests *one action*, not implementation steps
3. **Then** verifies *observable outcomes*, not internal changes

### Template

```python
async def test_filter_detects_person(self, person_video: Path):
    # Given: A fixture containing visible person footage, not blank synthetic frames.
    # config is YoloFilterConfig; this test needs local weights and inference deps.
    filter_plugin = YOLOFilter(config)

    # When: Running detection
    try:
        result = await filter_plugin.detect(person_video)

        # Then: Person should be detected with high confidence.
        assert "person" in result.detected_classes
        assert result.confidence > 0.5
    finally:
        await filter_plugin.shutdown()
```

### Writing Good "Given" Sections

Focus on **what** state exists, not **how** to manipulate internals:

**Bad:**
```python
# Given: A notifier with _connected set to False
notifier = MQTTNotifier(config)
notifier._connected = False  # Manipulating internal state!
```

**Good:**
```python
# Given: A notifier that failed to connect
fake_client = _FakeClientNoConnect()
monkeypatch.setattr("...mqtt.Client", lambda: fake_client)
notifier = MQTTNotifier(config)  # Naturally not connected
```

### Writing Good "When" Sections

Test **one action**. If you need multiple actions, you're testing a workflow (which may be valid for integration tests).

**Bad:**
```python
# When: Uploading and then checking
await storage.put_file(source, "test.mp4")
exists = await storage.exists(uri)
await storage.delete(uri)
```

**Good:**
```python
# When: Uploading a file
result = await storage.put_file(source, "test.mp4")
```

Or for lifecycle tests, make the workflow explicit:
```python
async def test_put_get_delete_roundtrip(self):
    """Full lifecycle: put -> exists -> get -> delete."""
    # Given: A storage instance and source file
    ...
    # When/Then: Each step of the lifecycle
    result = await storage.put_file(source, "test.mp4")
    assert await storage.exists(result.storage_uri)
    ...
```

### Writing Good "Then" Sections

Assert on **observable behavior**, not internal state:

**Bad:**
```python
# Then: Notifier should be shut down
assert notifier._shutdown_called is True
assert notifier._client is None
```

**Good:**
```python
# Then: Notifier should reject new operations
with pytest.raises(RuntimeError, match="shut down"):
    await notifier.send(alert)

# Then: Ping should return False
assert await notifier.ping() is False
```

### Connecting Given/When/Then to Behavioral Testing

| Section | Behavioral Focus | Anti-Pattern |
|---------|------------------|--------------|
| Given | Set up via public APIs or realistic fakes | Directly setting `obj._private = value` |
| When | Call public methods | Call internal helpers or set flags |
| Then | Assert return values, exceptions, side effects | Assert on `obj._internal_state` |

The key insight: **if you can't express your test in Given/When/Then without touching internals, you're testing implementation, not behavior.**

---

## Quick Reference: What to Test

| Test Type | What to Verify | What to Mock |
|-----------|----------------|--------------|
| Unit tests | Return values, exceptions raised, side effects | External services |
| Integration | Component interactions, data flow | Nothing (or just external APIs) |
| Contract | Request/response structure | Network layer only |

| Observable Behavior | Internal Implementation |
|---------------------|------------------------|
| Return values | Private attributes (`_foo`) |
| Raised exceptions | State flags |
| Side effects (files created, APIs called) | Method call order |
| Public method behavior | Internal helper methods |

---

## Anti-Patterns to Avoid

1. **Testing implementation details** - If refactoring breaks tests but behavior is unchanged, tests are too coupled
2. **Mocking everything** - If you mock 5+ things per test, you're not testing real behavior
3. **State flag assertions** - `assert obj._internal_flag is True` tests nothing useful
4. **Misleading test names** - Test name must match what's actually being tested
5. **Missing cleanup** - Global state must be restored between tests
6. **Elaborate fakes that mirror implementation** - Fakes should be simple recorders, not reimplementations

---

## See Also

- `AGENTS.md` - Development guidelines and patterns
- `DESIGN.md` - Architecture overview
- `tests/homesec/conftest.py` - Shared fixtures and mocks
- `tests/homesec/mocks/` - Mock implementations
