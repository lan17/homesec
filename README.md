# HomeSec

**Local-first AI security for your home.**

[![PyPI](https://img.shields.io/pypi/v/homesec)](https://pypi.org/project/homesec/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python: 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![Typing: Typed](https://img.shields.io/badge/typing-typed-2b825b)](https://peps.python.org/pep-0561/)
[![codecov](https://codecov.io/gh/lan17/HomeSec/branch/main/graph/badge.svg)](https://codecov.io/gh/lan17/HomeSec)

HomeSec turns your IP cameras into an intelligent security system. It detects people, vehicles, and objects with on-device AI, then uses vision language models to understand what's actually happening — not just "motion detected" but "person with package at front door."

Get smart alerts for what matters. Your footage stays on your network.

- **No cloud required** — runs entirely on your hardware
- **No subscriptions** — own your security, forever
- **Home Assistant ready** — MQTT integration out of the box
- **Fully extensible** — swap any component with Python plugins

## Table of Contents

- [Why HomeSec?](#why-homesec)
- [How It Works](#how-it-works)
- [Features](#features)
- [Quickstart](#quickstart)
  - [30-Second Start (Docker)](#30-second-start-docker)
  - [Manual Setup](#manual-setup)
  - [Developer Setup](#developer-setup)
- [Configuration](#configuration)
  - [Configuration Examples](#configuration-examples)
  - [Commands](#commands)
- [Plugins](#plugins)
  - [Built-in plugins](#built-in-plugins)
  - [Plugin interfaces](#plugin-interfaces)
  - [Writing a custom plugin](#writing-a-custom-plugin)
- [Observability](#observability)
- [Development](#development)
- [Contributing](#contributing)
- [License](#license)

## Why HomeSec?

| | Cloud NVRs | Basic Local NVR | HomeSec |
|---|:---:|:---:|:---:|
| No subscription fees | :x: | :white_check_mark: | :white_check_mark: |
| Footage stays private | :x: | :white_check_mark: | :white_check_mark: |
| AI object detection | :white_check_mark: | Some | :white_check_mark: |
| Scene understanding (VLM) | Limited | :x: | :white_check_mark: |
| Home Assistant integration | Varies | Varies | :white_check_mark: |
| Extensible with plugins | :x: | :x: | :white_check_mark: |

**What makes HomeSec different:** Two-stage AI analysis. Most systems stop at "person detected." HomeSec goes further — using vision language models to understand context: Is this a delivery? A family member? Someone unfamiliar lingering? You get alerts that actually mean something.

## How It Works

```mermaid
graph TD
    subgraph Wrapper [" "]
        direction LR
        style Wrapper fill:none,stroke:none

        S[Camera Feed]

        subgraph Processing [" "]
            direction TB
            C(New Clip) --> U([Backup to Storage])
            C --> F([Detect Objects: YOLO])
            F -->|Objects found| AI{Matches trigger?}
            AI -->|Yes| V([Analyze Scene: VLM])
            AI -->|No| D([Skip])
            V -->|Assessment| P{Alert worthy?}
            P -->|No| D
            P -->|Yes| N[Send Alert]
        end

        S -->|Motion detected| Processing

        PG[(Database)]
        Processing -.->|Events| PG
    end
```

1. **Capture** — Camera detects motion, creates clip
2. **Detect** — YOLO identifies objects (person, car, dog, etc.)
3. **Analyze** — VLM understands the scene in context
4. **Alert** — Smart notification via MQTT or email

### Design Principles

- **Local-Only by Default**: Video footage stays on your network. Cloud storage (Dropbox) and cloud AI (OpenAI) are opt-in.
- **Modular Architecture**: All components (sources, filters, analyzers, notifiers) are plugins. Swap out any piece with a few lines of Python.
- **Resilience**: Handles camera disconnects and network issues gracefully. Clips are backed up before processing.

## Features

- **Smart alerts, not noise** — AI filters out false positives so you only hear about what matters
- **Understands context** — VLM analysis knows the difference between a delivery and a stranger lingering
- **Works with your cameras** — RTSP streams, FTP uploads, or any video source via plugins
- **Home Assistant native** — MQTT notifications integrate seamlessly
- **Privacy by design** — cloud storage and cloud AI are opt-in, never required
- **Built to extend** — write custom sources, filters, storage backends, and notifiers in Python
- **Observable** — health endpoints plus Postgres-backed event logging

## Quickstart

### Docker
Use the included [docker-compose.yml](docker-compose.yml) (HomeSec + Postgres, pulls `leva/homesec:latest`).

Configure your own config.yaml and .env files as described in Manual Setup.

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
   # Download example config & env
   curl -O https://raw.githubusercontent.com/lan17/homesec/main/config/example.yaml
   mv example.yaml config.yaml

   curl -O https://raw.githubusercontent.com/lan17/homesec/main/.env.example
   mv .env.example .env

   # Setup environment (DB_DSN is required)
   # Edit .env to set your secrets!
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
   uv run python -m homesec.cli run --config config.yaml
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
      backend: rtsp
      config:
        rtsp_url_env: DRIVEWAY_RTSP_URL
        output_dir: "./recordings"
        stream:
          # Critical for camera compatibility:
          ffmpeg_flags: ["-rtsp_transport", "tcp", "-vsync", "0"]
        reconnect:
          backoff_s: 5

filter:
  backend: yolo
  config:
    classes: ["person", "car"]
    min_confidence: 0.6
```

In your `.env`:
```bash
DRIVEWAY_RTSP_URL="rtsp://user:pass@192.168.1.100:554/stream"
```

#### 2. The "Cloud Storage" (Dropbox)
Uploads to Cloud but keeps analysis local.

```yaml
storage:
  backend: dropbox
  config:
    token_env: DROPBOX_TOKEN
    root: "/SecurityCam"

notifiers:
    - backend: sendgrid_email
      config:
        api_key_env: SENDGRID_API_KEY
        to_emails: ["me@example.com"]
```

In your `.env`:
```bash
DROPBOX_TOKEN="sl.Al..."
SENDGRID_API_KEY="SG.xyz..."
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

**Run HomeSec:**
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

### Extensible by design

HomeSec is built to be modular. Each major capability is an interface (`ClipSource`, `StorageBackend`, `ObjectFilter`, `VLMAnalyzer`, `AlertPolicy`, `Notifier`) defined in `src/homesec/interfaces.py`. Swap out components — like replacing YOLO with a different detector — without touching the core.

Plugins are discovered at runtime via entry points.

### Built-in plugins

| Type | Plugins |
|------|---------|
| Sources | [`rtsp`](src/homesec/sources/rtsp/core.py), [`ftp`](src/homesec/sources/ftp.py), [`local_folder`](src/homesec/sources/local_folder.py) |
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

Extending HomeSec is straightforward. You can write custom sources, filters, storage backends, and more.

See [PLUGIN_DEVELOPMENT.md](PLUGIN_DEVELOPMENT.md) for a complete guide.

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
- Run HomeSec: `make run`

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
