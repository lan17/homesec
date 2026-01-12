# HomeSec Testing Guidelines

**Last reviewed:** 2025-01-11
**Purpose:** Principles for writing high-quality, maintainable tests that verify behavior without creating false confidence.

---

## Core Principles

### 1. Test Observable Behavior, Not Internal State

**Rule:** Verify what the code *does*, not how it *does it*. Tests should pass if behavior is correct, even if implementation changes.

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
    notifier = MQTTNotifier(config)
    await notifier.shutdown()
    await notifier.shutdown()  # Should not raise

    # GOOD: Verify behavior via public interface
    result = await notifier.ping()
    assert result is False  # Shutdown state observable via ping

    # GOOD: Verify operations fail as expected
    with pytest.raises(RuntimeError, match="shut down"):
        await notifier.send(alert)
```

---

### 2. Mock at the Boundary, Not Internal Methods

**Rule:** Mock external dependencies (HTTP, databases, filesystems) at their boundary, not internal methods. This tests more of your actual code.

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
```python
async def test_analyze_video(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # Create real test video
    video_path = tmp_path / "test.mp4"
    _create_test_video(video_path, frames=5)

    # Mock only at HTTP boundary
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value={
        "choices": [{"message": {"content": '{"risk_level": "low"}'}}]
    })

    async_cm = AsyncMock()
    async_cm.__aenter__ = AsyncMock(return_value=mock_response)
    async_cm.__aexit__ = AsyncMock(return_value=None)

    mock_session = MagicMock()
    mock_session.post = MagicMock(return_value=async_cm)
    mock_session.close = AsyncMock()

    monkeypatch.setattr("aiohttp.ClientSession", lambda **kw: mock_session)

    # Now real frame extraction happens, only HTTP is mocked
    vlm = OpenAIVLM(config)
    result = await vlm.analyze(video_path)

    assert result.risk_level == "low"
```

---

### 3. Use Contract Testing for API Calls

**Rule:** When mocking external services, verify that your code calls them correctly by capturing and asserting on the request structure.

**Bad - Just checking call count:**
```python
async def test_sends_email(self):
    notifier = SendGridNotifier(config)
    await notifier.send(alert)

    # BAD: Only verifies something was called
    mock_session.post.assert_called_once()
```

**Good - Verifying request contract:**
```python
async def test_sends_email_with_correct_structure(self):
    captured_request: dict[str, Any] = {}

    def capture_post(url: str, json: dict, headers: dict) -> AsyncMock:
        captured_request["url"] = url
        captured_request["json"] = json
        captured_request["headers"] = headers
        return mock_response

    mock_session.post = capture_post

    notifier = SendGridNotifier(config)
    await notifier.send(alert)

    # GOOD: Verify the actual API contract
    assert captured_request["url"] == "https://api.sendgrid.com/v3/mail/send"
    assert "Authorization" in captured_request["headers"]
    assert captured_request["json"]["personalizations"][0]["to"][0]["email"] == "user@example.com"
    assert "subject" in captured_request["json"]
```

---

### 4. Use Real Implementations Where Cheap

**Rule:** Prefer real implementations over mocks when they're fast, deterministic, and don't require external resources.

**Use real:**
- Filesystem operations via `tmp_path` fixture
- In-memory data structures
- Pure functions (validators, formatters, parsers)
- Entry points discovery (test against actual registered plugins)

**Mock:**
- Network calls (HTTP, MQTT, etc.)
- Databases (unless using testcontainers)
- GPU/ML inference
- Time-sensitive operations

**Example - Using real filesystem:**
```python
async def test_put_get_delete_roundtrip(self, tmp_path: Path):
    # GOOD: Real filesystem via tmp_path
    storage = LocalStorage(LocalStorageConfig(root=str(tmp_path / "storage")))

    source_file = tmp_path / "source.mp4"
    source_file.write_bytes(b"video content")

    # Real operations on real filesystem
    result = await storage.put_file(source_file, "clips/test.mp4")
    assert await storage.exists(result.storage_uri)

    download_path = tmp_path / "downloaded.mp4"
    await storage.get(result.storage_uri, download_path)
    assert download_path.read_bytes() == b"video content"
```

**Example - Using real entry points:**
```python
def test_returns_real_entry_points(self):
    # GOOD: Test against actual registered entry points
    result = list(iter_entry_points("console_scripts"))
    entry_point_names = [ep.name for ep in result]
    assert "homesec" in entry_point_names
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
def test_handles_unicode_filenames(self):
    # Claims to test unicode but uses ASCII!
    result = await storage.put_file(source, "clips/test_file.mp4")
    assert await storage.exists(result.storage_uri)
```

**Good - Test matches name:**
```python
def test_handles_unicode_filenames(self):
    # Actually tests unicode (Cyrillic and Japanese)
    result = await storage.put_file(source, "clips/\u043a\u0430\u043c\u0435\u0440\u0430.mp4")
    assert await storage.exists(result.storage_uri)

    result2 = await storage.put_file(source, "clips/\u7384\u95a2\u30ab\u30e1\u30e9.mp4")
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
async def test_filter_detects_person(self):
    # Given: A video with a person visible
    video_path = tmp_path / "person.mp4"
    _create_test_video(video_path, frames=10)
    filter_plugin = YOLOFilter(config)

    # When: Running detection
    result = await filter_plugin.detect(video_path)

    # Then: Person should be detected with high confidence
    assert "person" in result.detected_classes
    assert result.confidence > 0.5
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
