# HomeSec

[![PyPI](https://img.shields.io/pypi/v/homesec)](https://pypi.org/project/homesec/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python: 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![Typing: Typed](https://img.shields.io/badge/typing-typed-2b825b)](https://peps.python.org/pep-0561/)
[![codecov](https://codecov.io/gh/lan17/HomeSec/branch/main/graph/badge.svg)](https://codecov.io/gh/lan17/HomeSec)

HomeSec is a self-hosted, extensible video pipeline for home security cameras. You can connect cameras directly via RTSP, receive clips over FTP, or watch a local folder. The pipeline uploads clips, detects objects, optionally runs VLM analysis, and sends policy-driven notifications. Configuration determines which local or cloud services receive footage and alerts.

The [behavior reference](docs/reference.md) links the maintained public specifications, operator guides, and known limitations.

## Design Principles

- **Operator-controlled destinations**: Choose local or Dropbox storage and configure the analyzer endpoint and notification backends. The supplied example selects Dropbox and OpenAI; review those choices before running it.
- **Modular Architecture**: All major components (sources, filters, analyzers, notifiers) are decoupled plugins defined by strict interfaces. If you want to use a different AI model or storage backend, you can swap it out with a few lines of Python.
- **Resilience**: Recording and uploads can continue when Postgres is unavailable. Reconnect and stage retries have explicit limits; see the [known limitations](docs/reference.md) before relying on recovery behavior.

## Pipeline at a glance



```mermaid
graph TD
    %% Layout Wrapper for horizontal alignment
    subgraph Wrapper [" "]
        direction LR
        style Wrapper fill:none,stroke:none
        
        S[Clip Source]
        
        subgraph Pipeline [Media Processing Pipeline]
            direction TB
            C(Clip File) --> U([Upload to Storage])
            C --> F([Detect objects: YOLO])
            F -->|Filter succeeds| AI{VLM run mode and trigger classes}
            AI -->|Analyze| V([VLM Analysis])
            AI -->|Skip| P{Alert Policy}
            V --> P
            U --> P
            P -->|Notify| N[Notifiers]
        end
        
        S -->|New Clip File| Pipeline
        
        PG[(Postgres)]
        Pipeline -.->|State & Events| PG
    end
```

- **Parallel Processing**: Upload and filter run in parallel.
- **Resilience**: Upload failures do not block alerts; filter failures stop expensive VLM calls.
- **State**: Postgres stores current clip snapshots and best-effort lifecycle history (`clip_states` + `clip_events`). These writes are separate; a stored snapshot does not prove complete event history or recovery after a process restart.


## Table of Contents

- [Highlights](#highlights)
- [Pipeline at a glance](#pipeline-at-a-glance)
- [Quickstart](#quickstart)
  - [Docker](#docker)
  - [Manual Setup](#manual-setup)
  - [Developer Setup](#developer-setup)
- [Configuration](#configuration)
  - [Commands](#commands)
- [Plugins](#plugins)
  - [Built-in plugins](#built-in-plugins)
  - [Plugin interfaces](#plugin-interfaces)
  - [Writing a custom plugin](#writing-a-custom-plugin)
- [Observability](#observability)
- [Development](#development)
- [Documentation](docs/README.md)
- [Contributing](#contributing)
- [License](#license)

## Highlights

- Multiple pluggable video clip sources: [RTSP](https://en.wikipedia.org/wiki/Real-Time_Streaming_Protocol) motion detection, [FTP](https://en.wikipedia.org/wiki/File_Transfer_Protocol) uploads, or a watched folder
- Parallel upload + filter ([YOLO](https://en.wikipedia.org/wiki/You_Only_Look_Once)) with frame sampling and early exit
- OpenAI-compatible VLM analysis with structured output
- Policy-driven alerts with per-camera overrides
- Fan-out notifiers (MQTT for Home Assistant, SendGrid email)
- Optional Home Assistant MQTT discovery with per-camera alert entities and a push-notification [blueprint](docs/home-assistant.md)
- Postgres-backed state + events with graceful degradation
- Health endpoint plus optional Postgres telemetry logging
- Push-to-talk for RTSP cameras with ONVIF audio backchannel support and a local
  TP-Link Tapo talkback backend path



## Quickstart

### Docker
From a repository checkout, use the included [docker-compose.yml](docker-compose.yml) for HomeSec and Postgres:

```bash
cp config/example.yaml config/config.yaml
cp .env.example .env
# Edit both files for your cameras, storage, analyzer, and credentials.
# Comment out the local-development DB_DSN line in .env for Compose.
docker compose up -d
```

Compose mounts `config/config.yaml` at `/config/config.yaml` and `.env` at `/config/.env`; its application image is `leva/homesec:latest`. It publishes HTTP on port 8081 and WebRTC UDP on 8189–8199. For preview reachability, helper requirements, and persistent media paths, follow the [deployment guide](docs/preview-deployment.md).

Compose supplies a database DSN using its `postgres` service hostname. The entrypoint sources the mounted `.env` afterward, so the example's active `DB_DSN` pointing at `localhost` would override that value inside the container. Comment out that line for Compose, or deliberately set a container-reachable database DSN.

### Manual Setup
For standard production usage without Docker Compose:

1. **Prerequisites**:
   - Python 3.10+
   - ffmpeg and ffprobe
   - PostgreSQL (running and accessible, with HomeSec's migrations applied)
   - For enabled native media features, the separately built helper described in [native setup](docs/webrtc-preview.md#local-development-and-python-installations)

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
   export DB_DSN="postgresql+asyncpg://user:pass@localhost:5432/homesec"
   ```

4. **Prepare the schema and validate**
   Run the Alembic migrations from a matching release's repository checkout with the same exported `DB_DSN`:
   ```bash
   python -m alembic -c alembic.ini upgrade head
   homesec validate --config config.yaml
   ```
   The CLI run command does not apply migrations. Configuration validation does not prove camera, provider, or database connectivity.

5. **Run**
   ```bash
   homesec run --config config.yaml
   ```

### Developer Setup
If you are contributing or running from source:

1. **Install tools and build dependencies**
   Follow the [developer prerequisites](docs/webrtc-preview.md#local-development-and-python-installations), then run:
   ```bash
   make dev-setup
   ```
   This installs locked Python/UI dependencies and builds the native helper and UI. It does not start Postgres or HomeSec.

2. **Configure and start infrastructure**
   Copy and edit `config/example.yaml` as `config/config.yaml` and `.env.example` as `.env`. Keep the local-development `DB_DSN` for this host-run setup, then:
   ```bash
   make db  # Starts just Postgres in Docker
   ```

3. **Run**
   ```bash
   make run
   ```
   `make run` applies database migrations before starting; migration failure stops this command. The Docker entrypoint instead warns and continues after a failed migration, so check container logs and schema readiness during upgrades.


## Configuration

Configuration is YAML-based and validated against core and plugin models. Reference secrets through environment-variable names (`*_env`) rather than putting credentials in YAML. The examples below are fragments to merge into a complete configuration.

### Configuration Examples

#### 1. RTSP camera with reconnects
Configure connection timeouts and an unlimited retry budget for a camera that can become temporarily unavailable.

```yaml
cameras:
  - name: driveway
    source:
      backend: rtsp
      config:
        rtsp_url_env: DRIVEWAY_RTSP_URL
        output_dir: "./recordings"
        stream:
          connect_timeout_s: 2.0
          io_timeout_s: 2.0
        reconnect:
          backoff_s: 5
          max_attempts: 0  # Retry indefinitely

filter:
  backend: yolo
  config:
    classes: ["person", "dog"]
    min_confidence: 0.6
```

In your `.env`:
```bash
DRIVEWAY_RTSP_URL="rtsp://user:pass@192.168.1.100:554/stream"
```

#### 2. The "Cloud Storage" (Dropbox)
Selects Dropbox storage and an email notifier. Analyzer behavior is configured separately under `vlm`.

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
        from_email: "alerts@example.com"
        to_emails: ["me@example.com"]
```

In your `.env`:
```bash
DROPBOX_TOKEN="sl.Al..."
SENDGRID_API_KEY="SG.xyz..."
```

See [`config/example.yaml`](config/example.yaml) for a starting configuration. The [core models](src/homesec/models/config.py) and each plugin's configuration model define supported settings and defaults.

### Tips

- **Secrets**: Never put secrets in YAML. Use env vars (`*_env`) and set them in your shell or `.env`.
- **Notifiers**: Notifiers are optional. With no enabled notifiers, alert decisions are still evaluated and recorded, but no external notifications are sent.
- **YOLO classes**: The current filter allowlist is `person`, `bird`, `cat`, `dog`, `horse`, `sheep`, `cow`, `elephant`, `bear`, `zebra`, and `giraffe`. Other names are ignored when building target class IDs; an all-unsupported list fails detection. A name appearing in the model's COCO labels does not make it supported by this filter.
- **Live preview**: Rust WebRTC is the default backend and shares one preview input per camera to deliver H.264/Opus directly to viewers. Preview remains disabled until enabled in config; set `preview.config.advertised_ip` to the HomeSec server's client-reachable LAN/VPN IP before enabling it. See [`docs/webrtc-preview.md`](docs/webrtc-preview.md) for native helper installation and UDP reachability. Existing HLS configurations remain supported; select `preview.backend: hls` and keep its scratch storage on tmpfs separate from recordings and durable storage. Preview defaults to `recording_policy: stop_on_recording`; `allow_during_recording` is best-effort and may consume an extra RTSP session. See [`docs/preview-deployment.md`](docs/preview-deployment.md).
- **Push-to-talk**: Browser microphone to camera speaker support is auto-probed for RTSP cameras by default. HomeSec tries standards-based backends first; the bundled ONVIF backend supports `PCMU/8000` and `PCMA/8000`, and the bundled `tapo_local` backend implements a local TP-Link Tapo talkback path without TP-Link cloud calls at runtime. Auto mode keeps ONVIF ahead of Tapo, then can identify Tapo from persisted ONVIF identity metadata or passive local endpoint discovery for manual RTSP cameras. Use `cameras[].talk.backend: onvif_rtsp_backchannel` or `cameras[].talk.backend: tapo_local` for explicit selection. Tapo local talk defaults to username `admin`, can derive host and password material from the RTSP URL, and can use explicit backend env vars when credentials differ. Opt out cameras with `cameras[].talk.mode: disabled` when needed. Physical Tapo C120 compatibility still needs per-firmware validation. See [`docs/push-to-talk.md`](docs/push-to-talk.md).

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
homesec cleanup --config config.yaml --older_than_days 7
```

Cleanup defaults to a dry run. Inspect its output before explicitly applying with `--dry_run=False`. It rechecks eligible clips using the YOLO-shaped cleanup configuration, deletes storage objects before marking records deleted, and can also remove a separate local recording. It is distinct from automatic retention and API deletion; review the [cleanup limitations](docs/reference.md#storage-retention).

Use `homesec <command> --help` for detailed options on each command.

## Plugins

### Extensible by design

We designed HomeSec to be modular. Each major capability is an interface (`ClipSource`, `StorageBackend`, `ObjectFilter`, `VLMAnalyzer`, `AlertPolicy`, `Notifier`) defined in `src/homesec/interfaces.py`. This means you can swap out components (like replacing YOLO with a different detector) without changing the core pipeline.
  
Built-in plugins are registered by decorators when their modules are imported. External packages can participate through the `homesec.plugins` entry-point group.

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

| Type | Interface | `PluginType` for `@plugin` |
|------|-----------|-----------|
| Sources | `ClipSource` | `SOURCE` |
| Filters | `ObjectFilter` | `FILTER` |
| Storage | `StorageBackend` | `STORAGE` |
| VLM analyzers | `VLMAnalyzer` | `ANALYZER` |
| Notifiers | `Notifier` | `NOTIFIER` |
| Alert policies | `AlertPolicy` | `ALERT_POLICY` |

### Writing a custom plugin

Extending HomeSec is designed to be easy. You can write custom sources, filters, storage backends, and more.

👉 **See [PLUGIN_DEVELOPMENT.md](PLUGIN_DEVELOPMENT.md) for a complete guide.**

## Observability

- Public health: `GET /health` and `/api/v1/health`, served on `server.host`/`server.port`. A running pipeline with unavailable Postgres reports `degraded` with HTTP 200; stopped/stale runtime health reports HTTP 503. Bootstrap mode reports `setup_required` with HTTP 200.
- Telemetry: `DB_DSN` enables the separate Postgres log handler. It classifies a truthy `event_type` or `kind="event"` as event telemetry; ordinary logs use `DB_LOG_LEVEL`, with warnings/errors also admitted. The bounded background queue and failed batch writes can lose records. Telemetry is best-effort and does not replace `ClipRepository` state/history.
- Log hygiene: the handler serializes messages, extras, and exception text; it is not a universal secret scrubber. Keep credentials, tokens, raw media, and full DSNs out of log calls and troubleshooting captures. See [logging guidance](AGENTS.md#logging-guidance).

## Development

### Setup

1. Clone the repository
2. Install [uv](https://docs.astral.sh/uv/) for dependency management
3. `make dev-setup` to install locked dependencies and build the UI/native helper
4. Copy and edit `config/config.yaml` and `.env` as described in [Developer Setup](#developer-setup)
5. `make db` to start Postgres locally

### Commands

- Run tests: `make test`
- Run type checking (strict): `make typecheck`
- Run the full Rust/Python/UI gate: `make check`
- Check Python lint/types and lockfile without running application tests: `make lint typecheck lock-check`
- Run the pipeline: `make run`

### Notes

- Tests must include Given/When/Then comments
- Architecture and guide navigation: [documentation index](docs/README.md)
- Test setup and database isolation: [TESTING.md](TESTING.md); CI lanes: [CI.md](CI.md)

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

Found a bug or have a feature request? Please [open an issue](https://github.com/lan17/homesec/issues) with:
- A clear description of the problem or suggestion
- Steps to reproduce (for bugs)
- Your environment (OS, Python version, HomeSec version)

## License

Apache 2.0. See `LICENSE`.
