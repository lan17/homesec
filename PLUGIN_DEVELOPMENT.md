# HomeSec Plugin Development Guide

This guide explains how to develop plugins for HomeSec, a pluggable async video pipeline for home security cameras.

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
3.  **Registration**: Use the `@plugin` decorator to register the class.
4.  **Creation**: Implement a `create(cls, config)` class method.

### Key Principles

-   **Context-Free Creation**: The `create` method receives *only* a fully resolved configuration object. Runtime dependencies (like `camera_name` for sources or default alert-policy `trigger_classes`) are injected into the config model before validation when needed.
-   **Strict Typing**: All configs are Pydantic models.
-   **Backend/Config Boundary**: Core config uses `backend` + opaque `config` payloads; plugin-specific fields live in the plugin’s `config_cls`.
-   **Async Interface**: All I/O operations must be async.
-   **Lifecycle Management**: Implement `shutdown()` and `ping()` (where applicable).

---

## Plugin Types

| Plugin Type | Interface | Registry Imports | Purpose |
| :--- | :--- | :--- | :--- |
| Filter | `ObjectFilter` | `PluginType.FILTER` | Object detection in video |
| VLM Analyzer | `VLMAnalyzer` | `PluginType.ANALYZER` | Vision-language analysis |
| Storage | `StorageBackend` | `PluginType.STORAGE` | File storage (Dropbox, local) |
| Notifier | `Notifier` | `PluginType.NOTIFIER` | Alert delivery (MQTT, email) |
| Alert Policy | `AlertPolicy` | `PluginType.ALERT_POLICY` | Notification decision logic |
| Clip Source | `ClipSource` | `PluginType.SOURCE` | Video clip production |

---

## Common Plugin Structure

All plugins, regardless of their type, share a common structure that ensures type safety and consistent lifecycle management.

### 1. The @plugin Decorator

Every plugin class must be decorated with `@plugin`. This handles registration automatically.

```python
@plugin(type=PluginType.FILTER, name="my_plugin")
class MyPlugin(ObjectFilter):
    ...
```

### 2. Configuration Model

Every plugin must define a Pydantic model for its configuration. This model serves as the contract for the plugin.

- **Strict Validation**: Utilize Pydantic's validation features (types, `Field` constraints) to ensure invalid configs are caught early.
- **Injected Fields**: If your plugin type needs runtime context (sources, alert policy), declare those fields in your config model. The registry will inject them automatically.

Tip: Define the plugin-specific config model in the same module as the plugin implementation (no re-exports).

```python
class MyConfig(BaseModel):
    model_path: str
    camera_name: str | None = None  # Injected field
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

- **shutdown()**: Mandatory. Release all resources (thread pools, GPU memory, network sessions). This is called during application shutdown.
- **ping()**: Optional (if supported by interface). Return `True` if the plugin is healthy.

---

## Creating a Filter Plugin

Filter plugins detect objects in video clips.

### Step 1: Define Config and Implementation

```python
# my_filter/custom_filter.py
from pathlib import Path
from pydantic import BaseModel, Field

from homesec.interfaces import ObjectFilter
from homesec.models.filter import FilterResult, FilterOverrides
from homesec.plugins.registry import PluginType, plugin

# 1. Define Configuration
class CustomFilterSettings(BaseModel):
    """Configuration for CustomFilter plugin."""
    model_config = {"extra": "forbid"}

    model_path: str
    confidence: float = 0.5

# 2. Implement and Register
@plugin(type=PluginType.FILTER, name="custom")
class CustomFilter(ObjectFilter):
    """Custom object detection filter."""

    config_cls = CustomFilterSettings

    @classmethod
    def create(cls, config: CustomFilterSettings) -> ObjectFilter:
        return cls(config)

    def __init__(self, config: CustomFilterSettings) -> None:
        self._config = config
        self._model = self._load_model(config.model_path)

    async def detect(
        self,
        video_path: Path,
        overrides: FilterOverrides | None = None,
    ) -> FilterResult:
        # Implementation...
        return FilterResult(...)

    async def shutdown(self, timeout: float | None = None) -> None:
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
from homesec.interfaces import VLMAnalyzer
from homesec.models.vlm import AnalysisResult, VLMConfig
from homesec.plugins.registry import PluginType, plugin
from pydantic import BaseModel

class CustomVLMSettings(BaseModel):
    model_name: str
    api_key_env: str | None = None
    base_url: str | None = None

@plugin(type=PluginType.ANALYZER, name="custom")
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
        # ... logic ...
        pass
```

Note: `vlm.trigger_classes` and `vlm.run_mode` live in the core VLM config, not the analyzer plugin config. Keep analyzer configs backend-specific only.

---

## Creating a Storage Backend Plugin

Storage plugins handle file persistence.

```python
@plugin(type=PluginType.STORAGE, name="s3")
class S3Storage(StorageBackend):
    config_cls = S3Config

    @classmethod
    def create(cls, config: S3Config) -> StorageBackend:
        return cls(config)
    # ... implementation of put_file, get_file, etc.
```

---

## Creating a Notifier Plugin

Notifiers send alerts.

```python
@plugin(type=PluginType.NOTIFIER, name="slack")
class SlackNotifier(Notifier):
    config_cls = SlackConfig

    @classmethod
    def create(cls, config: SlackConfig) -> Notifier:
        return cls(config)

    async def send(self, alert: Alert) -> None:
        # ...
        pass
```

---

## Creating a Clip Source Plugin

Sources like cameras. Note: `camera_name` is injected into the config.

```python
class MySourceConfig(BaseModel):
    host: str
    camera_name: str | None = None  # Injected at runtime

@plugin(type=PluginType.SOURCE, name="mysource")
class MySource(ClipSource):
    config_cls = MySourceConfig

    @classmethod
    def create(cls, config: MySourceConfig) -> ClipSource:
        if not config.camera_name:
             raise ValueError("Camera name missing")
        return cls(config.host, config.camera_name)
```

---

## Lifecycle Management

-   `ping()`: For health checks (Notifier, Storage). Returns `bool`.
-   `shutdown(timeout)`: Cleanup resources.

---

## External Plugin Registration

1.  **Entry Point**: Add to `pyproject.toml`:
    ```toml
    [project.entry-points."homesec.plugins"]
    my_plugin = "my_plugin_package"
    ```
2.  **Package**: Ensure `my_plugin_package/__init__.py` imports the plugin modules so decorators run.

---

## Testing Plugins

Use `homesec.plugins.registry.load_plugin` for integration tests.

```python
from homesec.plugins.registry import PluginType, load_plugin

def test_my_plugin():
    plugin = load_plugin(
        PluginType.FILTER,
        "custom",
        {"model_path": "foo"},
    )
    assert plugin is not None
```
3. **Log appropriately**: Use structured logging with `logging.getLogger(__name__)`
4. **Validate early**: Fail fast on invalid configuration
5. **Import hygiene**: Keep module imports light; for optional/heavy deps, either import lazily or guard module-level imports with clear runtime errors
6. **Document thoroughly**: Include docstrings and type hints
7. **Test comprehensively**: Unit tests, integration tests, and edge cases
8. **Manage resources**: Always implement proper `shutdown()`

---

*Document version: 1.2*
*Last updated: 2026-02-01*
