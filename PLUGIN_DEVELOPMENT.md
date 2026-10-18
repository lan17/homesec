# HomeSec Plugin Development Guide

This guide explains how to develop plugins for HomeSec, a pluggable async video pipeline for home security cameras.
Exact interfaces live in [interfaces.py](src/homesec/interfaces.py); registration
and construction live in [registry.py](src/homesec/plugins/registry.py). The class
sketches below show extension points and require the indicated backend logic.

## Table of Contents

1. [Plugin Architecture Overview](#plugin-architecture-overview)
2. [Plugin Types](#plugin-types)
3. [Creating a Filter Plugin](#creating-a-filter-plugin)
4. [Creating a VLM Analyzer Plugin](#creating-a-vlm-analyzer-plugin)
5. [Creating a Storage Backend Plugin](#creating-a-storage-backend-plugin)
6. [Creating a Notifier Plugin](#creating-a-notifier-plugin)
7. [Creating an Alert Policy Plugin](#creating-an-alert-policy-plugin)
8. [Creating a Clip Source Plugin](#creating-a-clip-source-plugin)
9. [Configuration Models](#configuration-models)
10. [Lifecycle Management](#lifecycle-management)
11. [External Plugin Registration](#external-plugin-registration)
12. [Testing Plugins](#testing-plugins)

---

## Plugin Architecture Overview

HomeSec uses a unified **Class-Based Plugin Architecture**. All plugins follow a consistent pattern:

1.  **Interface**: Inherit from a specific interface (e.g., `ObjectFilter`, `Notifier`).
2.  **Configuration**: Define a Pydantic model for configuration.
3.  **Registration**: Use `@plugin(plugin_type=PluginType.FILTER, name="custom")`
    (with the appropriate plugin type) to register the class.
4.  **Creation**: Implement a `create(cls, config)` class method.

### Key Principles

-   **Context-Free Creation**: The `create` method receives a validated
    configuration object. Runtime fields (like source `camera_name` or the
    default policy's `trigger_classes`) are supplied by the calling loader.
    Secret references may still need resolution by the backend.
-   **Strict Typing**: All configs are Pydantic models.
-   **Backend/Config Boundary**: Core config uses `backend` + opaque `config` payloads; plugin-specific fields live in the plugin’s `config_cls`.
-   **Async Interface**: Async interface methods must avoid blocking the event
    loop; offload blocking SDK/file calls and CPU work where needed. Factories
    and alert-policy decisions are synchronous.
-   **Lifecycle Management**: Implement every abstract method on the chosen
    interface. The six plugin types have different lifecycle requirements.

---

## Plugin Types

| Plugin Type | Interface | Registry Imports | Purpose |
| :--- | :--- | :--- | :--- |
| Filter | `ObjectFilter` | `PluginType.FILTER` | Object detection in video |
| VLM Analyzer | `VLMAnalyzer` | `PluginType.ANALYZER` | Vision-language analysis |
| Storage | `StorageBackend` | `PluginType.STORAGE` | File storage (bundled: Dropbox, local) |
| Notifier | `Notifier` | `PluginType.NOTIFIER` | Alert delivery (bundled: MQTT, SendGrid email) |
| Alert Policy | `AlertPolicy` | `PluginType.ALERT_POLICY` | Notification decision logic |
| Clip Source | `ClipSource` | `PluginType.SOURCE` | Video clip production |

---

## Common Plugin Structure

All plugins, regardless of their type, share a common structure that ensures type safety and consistent lifecycle management.

### 1. The @plugin Decorator

Every plugin class must be decorated with `@plugin`. This handles registration automatically.

```python
@plugin(plugin_type=PluginType.FILTER, name="my_plugin")
class MyPlugin(ObjectFilter):
    ...
```

### 2. Configuration Model

Every plugin must define a Pydantic model for its configuration. This model serves as the contract for the plugin.

- **Strict Validation**: Utilize Pydantic's validation features (types, `Field` constraints) to ensure invalid configs are caught early.
- **Injected Fields**: Declare runtime fields supplied by the calling loader.
  `load_source_plugin()` supplies `camera_name` and any explicitly passed runtime
  context; the alert-policy loader supplies trigger classes only for the bundled
  default policy. The generic registry merges caller-provided context before
  validation; it does not discover dependencies from your field declarations.

Tip: Define the plugin-specific config model in the same module as the plugin implementation (no re-exports).

```python
class MyConfig(BaseModel):
    model_path: str
    camera_name: str  # Required when the source loader supplies camera context
```

### 3. The `create()` Factory

Plugins are instantiated via a `create` class method, not `__init__` directly (though `create` usually calls `__init__`).

- **Signature**: `create(cls, config: YourConfigModel) -> Interface`
- **Purpose**: Allows for any pre-processing or setup before object creation.
- **Context-Free**: The `config` object passed to `create` is fully populated. You should not need to access global state.

### 4. Async Interface

HomeSec is an async-first pipeline.

- **I/O Bound**: Use `aiohttp`, `asyncio` file operations, or database drivers.
- **CPU Bound**: Use `asyncio.get_running_loop().run_in_executor` with a `ProcessPoolExecutor` to avoid blocking the main event loop.

### 5. Lifecycle Methods

- **shutdown(timeout=None)**: Required for sources, filters, analyzers, storage,
  and notifiers. Release owned resources and stop background work. Runtime
  teardown also calls it during reload or partial-startup cleanup.
- **ping()**: Required by those same five interfaces. Return `True` when the
  component's defined health check succeeds; a ping need not exercise delivery,
  upload, or inference.
- **Alert policies**: Implement synchronous `should_notify()`. `AlertPolicy`
  provides `make_decision()` and requires no shutdown or ping method.

---

## Creating a Filter Plugin

Filter plugins detect objects in video clips.

### Step 1: Define Config and Implementation

```python
# my_filter/custom_filter.py
from pathlib import Path

from pydantic import BaseModel, Field

from homesec.interfaces import ObjectFilter
from homesec.models.filter import FilterOverrides, FilterResult
from homesec.plugins.registry import PluginType, plugin

# 1. Define Configuration
class CustomFilterSettings(BaseModel):
    """Configuration for CustomFilter plugin."""
    model_config = {"extra": "forbid"}

    model_path: str
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)

# 2. Implement and Register
@plugin(plugin_type=PluginType.FILTER, name="custom")
class CustomFilter(ObjectFilter):
    """Custom object detection filter."""

    config_cls = CustomFilterSettings

    @classmethod
    def create(cls, config: CustomFilterSettings) -> ObjectFilter:
        return cls(config)

    def __init__(self, config: CustomFilterSettings) -> None:
        self._config = config
        # Load your backend here; keep expensive work out of the event loop.

    async def detect(
        self,
        video_path: Path,
        overrides: FilterOverrides | None = None,
    ) -> FilterResult:
        # Run your backend and return all required FilterResult fields.
        raise NotImplementedError("Implement backend detection")

    async def ping(self) -> bool:
        # Check that your backend is ready to accept work.
        raise NotImplementedError("Implement backend health check")

    async def shutdown(self, timeout: float | None = None) -> None:
        # Release any model/executor resources allocated by your backend.
        pass
```

### Step 2: Use in Configuration

```yaml
filter:
  backend: custom
  config:
    model_path: /path/to/model.pt
    confidence: 0.6
```

---

## Creating a VLM Analyzer Plugin

VLM analyzers provide vision-language analysis.

### Implementation

```python
# my_vlm/custom_vlm.py
from pathlib import Path

from pydantic import BaseModel

from homesec.interfaces import VLMAnalyzer
from homesec.models.filter import FilterResult
from homesec.models.vlm import AnalysisResult, VLMConfig
from homesec.plugins.registry import PluginType, plugin

class CustomVLMSettings(BaseModel):
    model_name: str
    api_key_env: str | None = None
    base_url: str | None = None

@plugin(plugin_type=PluginType.ANALYZER, name="custom")
class CustomVLM(VLMAnalyzer):
    config_cls = CustomVLMSettings

    @classmethod
    def create(cls, config: CustomVLMSettings) -> VLMAnalyzer:
        return cls(config)

    def __init__(self, config: CustomVLMSettings) -> None:
        self._config = config

    async def analyze(
        self,
        video_path: Path,
        filter_result: FilterResult,
        config: VLMConfig,
    ) -> AnalysisResult:
        # Return a validated AnalysisResult from your provider/backend.
        raise NotImplementedError("Implement backend analysis")

    async def ping(self) -> bool:
        raise NotImplementedError("Implement backend health check")

    async def shutdown(self, timeout: float | None = None) -> None:
        # Close provider sessions and owned workers.
        pass
```

Note: `vlm.trigger_classes` and `vlm.run_mode` live in the core VLM config, not the analyzer plugin config. Keep analyzer configs backend-specific only.

---

## Creating a Storage Backend Plugin

Storage plugins handle file persistence.

```python
@plugin(plugin_type=PluginType.STORAGE, name="s3")
class S3Storage(StorageBackend):
    config_cls = S3Config

    @classmethod
    def create(cls, config: S3Config) -> StorageBackend:
        return cls(config)
    # Implement every StorageBackend abstract method, including get() (not get_file()).
```

---

## Creating a Notifier Plugin

Notifiers send alerts.

```python
@plugin(plugin_type=PluginType.NOTIFIER, name="slack")
class SlackNotifier(Notifier):
    config_cls = SlackConfig

    @classmethod
    def create(cls, config: SlackConfig) -> Notifier:
        return cls(config)

    async def send(self, alert: Alert) -> None:
        # ...
        pass

    # Implement ping() and shutdown(timeout=None) as well.
```

These S3/Slack sketches illustrate extension points; those backends are not
bundled. Import their interface/payload types and define their Pydantic settings
in the implementation module. Storage `delete()` must tolerate a missing object.

---

## Creating an Alert Policy Plugin

Implement the synchronous `AlertPolicy.should_notify(camera_name, filter_result,
analysis) -> tuple[bool, str]` interface and register the class using
`PluginType.ALERT_POLICY`. Accept absent filter/analysis results where the
interface permits them. Keep backend-specific thresholds and camera overrides in
your settings model. The bundled
[default policy](src/homesec/plugins/alert_policies/default.py) is a complete
example; its missing-analysis fallback can notify when filter trigger classes
match, including when analysis was skipped. No lifecycle methods are required.

---

## Configuration Models

`config_cls` defines backend-specific fields. Core models retain `backend` and an
opaque `config` payload so third-party settings survive parsing. The registry
validates with `config_cls.model_validate()` before calling `create()`; use
`validate_plugin()` when validation alone is needed. Validation does not prove
provider reachability, construct the plugin, or perform inference/upload.

---

## Creating a Clip Source Plugin

Sources produce finalized local clips and emit them through a registered
callback; the pipeline owns storage upload. `camera_name` is injected by
`load_source_plugin()`. Implement the complete `ClipSource` interface, including
async start/shutdown/ping and synchronous callback/health/heartbeat methods.

```python
class MySourceConfig(BaseModel):
    host: str
    camera_name: str  # Supplied by load_source_plugin()

@plugin(plugin_type=PluginType.SOURCE, name="mysource")
class MySource(ClipSource):
    config_cls = MySourceConfig

    @classmethod
    def create(cls, config: MySourceConfig) -> ClipSource:
        return cls(config.host, config.camera_name)

    # Supply __init__ and every abstract ClipSource method.
```

---

## Lifecycle Management

The runtime assembler owns source/filter/analyzer/notifier teardown. The host
that constructs a storage/persistence stack closes it: the application owns its
control-plane stack, while a subprocess runtime worker builds and closes its own
stack. The pipeline waits for tracked clip work; it does not close plugins itself. Follow existing bounded
cleanup patterns in [runtime assembly](src/homesec/runtime/assembly.py).
Never assume a source `stop()` or a storage/notifier `close()` interface exists.

---

## External Plugin Registration

1.  **Entry Point**: Add to `pyproject.toml`:
    ```toml
    [project.entry-points."homesec.plugins"]
    my_plugin = "my_plugin_package"
    ```
2.  **Package**: Ensure `my_plugin_package/__init__.py` imports the plugin modules so decorators run.
3.  **Discovery**: Application startup calls `discover_all_plugins()`, which imports
    built-in modules and each `homesec.plugins` entry point's module. It imports
    the module rather than calling an entry-point factory. Direct loader use in
    tests must first import the decorated module or invoke discovery explicitly.

---

## Testing Plugins

Test observable plugin behavior through the public interface. Use type-specific
loaders where available; a generic `load_plugin()` call returns `Any`. This
filesystem-only smoke test exercises the existing storage loader without
provider credentials or a database:

```python
from pathlib import Path

from homesec.models.config import StorageConfig
from homesec.plugins.storage import load_storage_plugin
from homesec.plugins.storage import local  # noqa: F401 - runs decorator registration

async def test_local_storage_roundtrip(tmp_path: Path) -> None:
    # Given: A real local backend and input bytes.
    source = tmp_path / "input.mp4"
    source.write_bytes(b"test-clip")
    storage = load_storage_plugin(
        StorageConfig(backend="local", config={"root": str(tmp_path / "stored")})
    )
    try:
        # When: Uploading through the storage interface and reading it back.
        uploaded = await storage.put_file(source, "clips/test.mp4")
        downloaded = tmp_path / "downloaded.mp4"
        await storage.get(uploaded.storage_uri, downloaded)

        # Then: The public operations preserve bytes.
        assert downloaded.read_bytes() == b"test-clip"
    finally:
        await storage.shutdown()
```
For custom plugins, test invalid settings, meaningful return payloads, failure
propagation, and owned-resource cleanup at their boundaries. See
[TESTING.md](TESTING.md). Keep imports light and logs structured; avoid logging
secret values or raw media.

---

*Last reviewed: 2026-10-18*
