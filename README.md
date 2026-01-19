# HomeSec

[![PyPI](https://img.shields.io/pypi/v/homesec)](https://pypi.org/project/homesec/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python: 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![Typing: Typed](https://img.shields.io/badge/typing-typed-2b825b)](https://peps.python.org/pep-0561/)
[![codecov](https://codecov.io/gh/lan17/HomeSec/branch/main/graph/badge.svg)](https://codecov.io/gh/lan17/HomeSec)

HomeSec is a self-hosted, extensible video pipeline for home security cameras. Connect cameras directly via RTSP with motion detection, receive clips over FTP, or watch a folder—then filter with AI and get smart notifications. Your footage stays private and off third-party clouds.

## Design Principles

- **Local-Only Data Processing**: Video footage remains on the local network by default. Cloud storage is opt-in via storage backends.
- **Modular Architecture**: All major components (sources, filters, analyzers, notifiers) are decoupled plugins defined by strict interfaces.
- **Resilience**: The pipeline handles intermittent stream failures and network instability without crashing or stalling.

## Pipeline at a glance



```mermaid
graph TD
    S[Clip Source] -->|New Job| C(Clip File)
    C --> U[Upload]
    C --> F[Filter]
    U -->|Store| DB[Storage]
    F -->|Detect| AI{Object?}
    AI -->|Yes| V[VLM Analysis]
    AI -->|No| D[Discard]
    V -->|Context| P[Alert Policy]
    P -->|Decide| N[Notifiers]
    
    style S fill:#d4e6f1,stroke:#3498db
    style P fill:#f9e79f,stroke:#f1c40f
    style N fill:#e8daef,stroke:#8e44ad
```

- **Parallel Processing**: Upload and filter run in parallel.
- **Resilience**: Upload failures do not block alerts; filter failures stop expensive VLM calls.
- **State**: Metadata is stored in Postgres (`clip_states` + `clip_events`) for full observability.


## Table of Contents

- [Highlights](#highlights)
- [Pipeline at a glance](#pipeline-at-a-glance)
- [Quickstart](#quickstart)
  - [Install](#1-install)
  - [Configure](#2-configure)
  - [Run](#3-run)
  - [With Docker](#with-docker-easier-setup)
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
- Parallel upload + filter ([YOLO](https://en.wikipedia.org/wiki/You_Only_Look_Once)) with frame sampling and early exit
- OpenAI-compatible VLM analysis with structured output
- Policy-driven alerts with per-camera overrides
- Fan-out notifiers (MQTT for Home Assistant, SendGrid email)
- Postgres-backed state + events with graceful degradation
- Health endpoint plus optional Postgres telemetry logging



## Quickstart

### 30-Second Start (Docker)
The fastest way to see it in action. Includes a pre-configured Postgres and a dummy local source.

```bash
git clone https://github.com/lan17/homesec.git
cd homesec
make up
```
*Modify `config/config.yaml` to add your real cameras, then restart.*

### Manual Setup
For standard production usage without Docker Compose:

1. **Prerequisites**:
   - Python 3.10+
   - ffmpeg
   - PostgreSQL (running and accessible)

2. **Install**
   ```bash
   pip install homesec
   ```

3. **Configure**
   ```bash
   # Download example config
   curl -O https://raw.githubusercontent.com/lan17/homesec/main/config/example.yaml
   mv example.yaml config.yaml
   
   # Setup environment (DB_DSN is required)
   export DB_DSN="postgresql://user:pass@localhost/homesec"
   ```

4. **Run**
   ```bash
   homesec run --config config.yaml
   ```

### Developer Setup
If you are contributing or running from source:

1. **Install dependencies**
   ```bash
   uv sync
   ```

2. **Start Infrastructure**
   ```bash
   make db  # Starts just Postgres in Docker
   ```

3. **Run**
   ```bash
   uv run python -m homesec.cli run --config config/config.yaml
   ```


## Configuration

Configuration is YAML-based and strictly validated. Secrets (API keys, passwords) should always be loaded from environment variables (`_env` suffix).

### Configuration Examples

#### 1. The "Power User" (Robust RTSP)
Best for real-world setups with flaky cameras.

```yaml
cameras:
  - name: driveway
    source:
      type: rtsp
      config:
        rtsp_url_env: CAM_URL
        output_dir: "./recordings"
        # Critical for camera compatibility:
        ffmpeg_flags: ["-rtsp_transport", "tcp", "-vsync", "0"]
        reconnect_backoff_s: 5

filter:
  plugin: yolo
  config:
    classes: ["person", "car"]
    min_confidence: 0.6
```

#### 2. The "Cloud Storage" (Dropbox)
Uploads to Cloud but keeps analysis local.

```yaml
storage:
  backend: dropbox
  dropbox:
    token_env: DROPBOX_TOKEN
    root: "/SecurityCam"

notifiers:
    - backend: sendgrid_email
      config:
        api_key_env: EMAIL_KEY
        to_emails: ["me@example.com"]
```

See [`config/example.yaml`](config/example.yaml) for a complete reference of all options.

### Tips

- **Secrets**: Never put secrets in YAML. Use env vars (`*_env`) and set them in your shell or `.env`.
- **Notifiers**: At least one notifier (mqtt/email) must be enabled unless `alert_policy.enabled` is false.
- **YOLO Classes**: Built-in classes include `person`, `car`, `truck`, `motorcycle`, `bicycle`, `dog`, `cat`, `bird`, `backpack`, `handbag`, `suitcase`.

After installation, the `homesec` command is available:

```bash
homesec --help
```

### Commands

**Run the pipeline:**
```bash
homesec run --config config.yaml
```

**Validate config:**
```bash
homesec validate --config config.yaml
```

**Cleanup old clips** (reanalyze and optionally delete empty clips):
```bash
homesec cleanup --config config.yaml --older_than_days 7 --dry_run=False
```

Use `homesec <command> --help` for detailed options on each command.

## Plugins

## Extensible by design

HomeSec is intentionally modular. Each major capability is an interface (`ClipSource`, `StorageBackend`, `ObjectFilter`, `VLMAnalyzer`, `AlertPolicy`, `Notifier`) defined in `src/homesec/interfaces.py`. This allows you to swap out components (e.g., replace YOLO with a different detector) without changing the core pipeline.

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

Extending HomeSec is designed to be easy. You can write custom sources, filters, storage backends, and more.

👉 **See [PLUGIN_DEVELOPMENT.md](PLUGIN_DEVELOPMENT.md) for a complete guide.**

## Observability

- Health endpoint: `GET /health` (configurable via `health.host`/`health.port` in config)
- Telemetry logs to Postgres when `DB_DSN` is set

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
