# HomeSec

[![PyPI](https://img.shields.io/pypi/v/homesec)](https://pypi.org/project/homesec/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python: 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![Typing: Typed](https://img.shields.io/badge/typing-typed-2b825b)](https://peps.python.org/pep-0561/)
[![codecov](https://codecov.io/gh/lan17/HomeSec/branch/main/graph/badge.svg)](https://codecov.io/gh/lan17/HomeSec)

HomeSec is a self-hosted, extensible video pipeline for home security cameras. Connect cameras directly via RTSP with motion detection, receive clips over FTP, or watch a folder—then filter with AI and get smart notifications. Your footage stays private and off third-party clouds.

Under the hood, it's a pluggable async pipeline: ingest clips from multiple sources, run object detection (YOLO), optionally call a vision-language model ([VLM](https://en.wikipedia.org/wiki/Vision%E2%80%93language_model)) for structured analysis, and send alerts via [MQTT](https://en.wikipedia.org/wiki/MQTT) or email. Every component—sources, filters, storage, analyzers, notifiers—is a plugin you can swap or extend.

## Table of Contents

- [Highlights](#highlights)
- [Pipeline at a glance](#pipeline-at-a-glance)
- [Quickstart](#quickstart)
  - [Install](#1-install) | [Configure](#2-configure) | [Run](#3-run) | [With Docker](#with-docker)
- [Configuration](#configuration)
- [CLI](#cli)
- [Plugins](#plugins)
  - [Built-in plugins](#built-in-plugins)
  - [Plugin interfaces](#plugin-interfaces)
  - [Writing a custom plugin](#writing-a-custom-plugin)
- [Observability](#observability)
- [Development](#development)
- [Contributing](#contributing)
- [License](#license)

## Highlights

- Multiple pluggable video clip sources: [RTSP](https://en.wikipedia.org/wiki/Real-Time_Streaming_Protocol) motion detection, [FTP](https://en.wikipedia.org/wiki/File_Transfer_Protocol) uploads, or a watched folder
- Parallel upload + filter ([YOLO11](https://en.wikipedia.org/wiki/You_Only_Look_Once)) with frame sampling and early exit
- OpenAI-compatible VLM analysis with structured output
- Policy-driven alerts with per-camera overrides
- Fan-out notifiers (MQTT for Home Assistant, SendGrid email)
- Postgres-backed state + events with graceful degradation
- Health endpoint plus optional Postgres telemetry logging

## Pipeline at a glance

```
ClipSource -> (Upload + Filter) -> VLM (optional) -> Alert Policy -> Notifier(s)
```

- Upload and filter run in parallel; VLM runs only when trigger classes are detected.
- Upload failures do not block alerts; filter failures stop processing.
- State is stored in Postgres (`clip_states` + `clip_events`) when available.

## Quickstart

### 1. Install

```bash
pip install homesec
```

Requires Python 3.10+ and ffmpeg.

### 2. Configure

Create a `config.yaml` file (see [Configuration](#configuration) for all options):

```yaml
version: 1

cameras:
  - name: front_door
    source:
      type: rtsp
      config:
        rtsp_url_env: FRONT_DOOR_RTSP_URL
        output_dir: "./recordings"

storage:
  backend: local
  local:
    root: "./storage"

state_store:
  dsn_env: DB_DSN

notifiers:
  - backend: mqtt
    config:
      host: "localhost"
      port: 1883

filter:
  plugin: yolo

vlm:
  backend: openai
  llm:
    api_key_env: OPENAI_API_KEY
    model: gpt-4o

alert_policy:
  backend: default
  enabled: true
  config:
    min_risk_level: medium
```

Set environment variables in your shell or a `.env` file:

```bash
export FRONT_DOOR_RTSP_URL="rtsp://user:pass@camera-ip:554/stream"
export DB_DSN="postgresql://user:pass@localhost/homesec"
export OPENAI_API_KEY="sk-..."
```

### 3. Run

```bash
homesec run --config config.yaml
```

### With Docker

For Docker deployment, clone the repo:

```bash
git clone https://github.com/lan17/homesec.git
cd homesec

cp config/example.yaml config/config.yaml
cp .env.example .env
# Edit both files with your settings

make up      # Start HomeSec + Postgres
make down    # Stop
```

## Configuration

Configs are YAML and validated with Pydantic. See [Quickstart](#2-configure) for a minimal example, or `config/example.yaml` for all options.

### Full example (Dropbox + per-camera alerts)

```yaml
version: 1

cameras:
  - name: front_door
    source:
      type: rtsp
      config:
        rtsp_url_env: FRONT_DOOR_RTSP_URL
        output_dir: "./recordings"

storage:
  backend: dropbox
  dropbox:
    root: "/homecam"
    token_env: DROPBOX_TOKEN
    app_key_env: DROPBOX_APP_KEY
    app_secret_env: DROPBOX_APP_SECRET
    refresh_token_env: DROPBOX_REFRESH_TOKEN

state_store:
  dsn_env: DB_DSN

notifiers:
  - backend: mqtt
    config:
      host: "localhost"
      port: 1883

filter:
  plugin: yolo
  config:
    classes: ["person"]
    min_confidence: 0.5

vlm:
  backend: openai
  llm:
    api_key_env: OPENAI_API_KEY
    model: gpt-4o

alert_policy:
  backend: default
  enabled: true
  config:
    min_risk_level: medium

per_camera_alert:
  front_door:
    min_risk_level: low
    notify_on_activity_types: ["person_at_door", "delivery"]
```

### Tips

- Secrets never go in YAML. Use env var names (`*_env`) and set values in your shell or `.env`.
- At least one notifier must be enabled (`mqtt` or `sendgrid_email`).
- Built-in YOLO classes: `person`, `car`, `truck`, `motorcycle`, `bicycle`, `dog`, `cat`, `bird`, `backpack`, `handbag`, `suitcase`.
- Set `alert_policy.enabled: false` to disable notifications.

## CLI

After installation, the `homesec` command is available:

```bash
homesec --help
```

### Commands

**Run the pipeline:**
```bash
homesec run --config config/config.yaml
```

**Validate config:**
```bash
homesec validate --config config/config.yaml
```

**Cleanup old clips** (reanalyze and optionally delete empty clips):
```bash
homesec cleanup --config config/config.yaml --older_than_days 7 --dry_run=False
```

Use `homesec <command> --help` for detailed options on each command.

## Plugins

HomeSec uses a plugin architecture—every component is discovered at runtime via entry points.

### Built-in plugins

| Type | Plugins |
|------|---------|
| Sources | [`rtsp`](src/homesec/sources/rtsp.py), [`ftp`](src/homesec/sources/ftp.py), [`local_folder`](src/homesec/sources/local_folder.py) |
| Filters | [`yolo`](src/homesec/plugins/filters/yolo.py) |
| Storage | [`dropbox`](src/homesec/plugins/storage/dropbox.py), [`local`](src/homesec/plugins/storage/local.py) |
| VLM analyzers | [`openai`](src/homesec/plugins/analyzers/openai.py) |
| Notifiers | [`mqtt`](src/homesec/plugins/notifiers/mqtt.py), [`sendgrid_email`](src/homesec/plugins/notifiers/sendgrid_email.py) |
| Alert policies | [`default`](src/homesec/plugins/alert_policies/default.py), [`noop`](src/homesec/plugins/alert_policies/noop.py) |

### Plugin interfaces

All interfaces are defined in [`src/homesec/interfaces.py`](src/homesec/interfaces.py).

| Type | Interface | Decorator |
|------|-----------|-----------|
| Sources | `ClipSource` | `@source_plugin` |
| Filters | `ObjectFilter` | `@filter_plugin` |
| Storage | `StorageBackend` | `@storage_plugin` |
| VLM analyzers | `VLMAnalyzer` | `@vlm_plugin` |
| Notifiers | `Notifier` | `@notifier_plugin` |
| Alert policies | `AlertPolicy` | `@alert_policy_plugin` |

### Writing a custom plugin

Each plugin provides a name, a Pydantic config model, and a factory:

```python
# my_package/filters/custom.py
from pydantic import BaseModel
from homesec.interfaces import ObjectFilter
from homesec.plugins.filters import FilterPlugin, filter_plugin

class CustomConfig(BaseModel):
    threshold: float = 0.5

class CustomFilter(ObjectFilter):
    ...

@filter_plugin(name="custom")
def register() -> FilterPlugin:
    return FilterPlugin(
        name="custom",
        config_model=CustomConfig,
        factory=lambda cfg: CustomFilter(cfg),
    )
```

Register via entry points in `pyproject.toml`:

```toml
[project.entry-points."homesec.plugins"]
my_filters = "my_package.filters.custom"
```

## Observability

- Health endpoint: `GET /health` (configurable in `health.host`/`health.port`)
- Optional telemetry logs to Postgres when `DB_DSN` is set:
  - Start local DB: `make db`
  - Run migrations: `make db-migrate`

## Development

### Setup

1. Clone the repository
2. Install [uv](https://docs.astral.sh/uv/) for dependency management
3. `uv sync` to install dependencies
4. `make db` to start Postgres locally

### Commands

- Run tests: `make test`
- Run type checking (strict): `make typecheck`
- Run both: `make check`
- Run the pipeline: `make run`

### Notes

- Tests must include Given/When/Then comments
- Architecture notes: `DESIGN.md`

## Contributing

Contributions are welcome! Here's how to get started:

1. **Fork and clone** the repository
2. **Create a branch** for your feature or fix: `git checkout -b my-feature`
3. **Install dependencies**: `uv sync`
4. **Make your changes** and ensure tests pass: `make check`
5. **Submit a pull request** with a clear description of your changes

### Guidelines

- All code must pass CI checks: `make check`
- Tests should include Given/When/Then comments explaining the test scenario
- New plugins should follow the existing patterns in `src/homesec/plugins/`
- Keep PRs focused on a single change for easier review

### Reporting Issues

Found a bug or have a feature request? Please [open an issue](../../issues) with:
- A clear description of the problem or suggestion
- Steps to reproduce (for bugs)
- Your environment (OS, Python version, HomeSec version)

## License

Apache 2.0. See `LICENSE`.
