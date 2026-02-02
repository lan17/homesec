# Phase 3: Home Assistant Add-on

**Goal**: Provide one-click installation for Home Assistant OS/Supervised users.

**Estimated Effort**: 3-4 days

**Dependencies**: Phase 1 (REST API), Phase 2 (HA Notifier)

---

## Overview

This phase creates a Home Assistant add-on that:
- Bundles PostgreSQL for zero-config database
- Uses s6-overlay for service management
- Provides ingress access to the API
- Auto-configures the HA notifier via SUPERVISOR_TOKEN

---

## 3.1 Repository Structure

**Location**: `homeassistant/addon/` in the main homesec monorepo.

Users add the add-on via: `https://github.com/lan17/homesec`

```
homeassistant/addon/
├── README.md
└── homesec/
    ├── config.yaml           # Add-on manifest
    ├── Dockerfile            # Container build
    ├── build.yaml            # Build configuration
    ├── DOCS.md               # Documentation
    ├── CHANGELOG.md          # Version history
    ├── icon.png              # Add-on icon (512x512)
    ├── logo.png              # Add-on logo (256x256)
    ├── rootfs/               # s6-overlay services
    │   └── etc/
    │       ├── s6-overlay/
    │       │   └── s6-rc.d/
    │       │       ├── postgres-init/
    │       │       ├── postgres/
    │       │       ├── homesec/
    │       │       └── user/
    │       └── nginx/
    │           └── includes/
    │               └── ingress.conf
    └── translations/
        └── en.yaml           # UI strings
```

**Note**: `repository.json` must be at the repo root (not in `homeassistant/addon/`) for Home Assistant to discover it.

---

## 3.2 Add-on Manifest

**File**: `homeassistant/addon/homesec/config.yaml`

### Interface

```yaml
name: HomeSec
version: "1.2.2"
slug: homesec
description: Self-hosted AI video security pipeline
url: https://github.com/lan17/homesec
arch:
  - amd64
  - aarch64
init: false                    # We use s6-overlay
homeassistant_api: true        # Access HA API
hassio_api: true               # Access Supervisor API
host_network: false
ingress: true
ingress_port: 8080
ingress_stream: true
panel_icon: mdi:cctv
panel_title: HomeSec

ports:
  8080/tcp: null               # API (exposed via ingress)

map:
  - addon_config:rw            # /config - HomeSec managed config
  - media:rw                   # /media - Media storage
  - share:rw                   # /share - Shared data

schema:
  config_path: str?
  log_level: list(debug|info|warning|error)?
  database_url: str?           # External DB (optional)
  storage_type: list(local|dropbox)?
  storage_path: str?
  dropbox_token: password?     # Mapped to DROPBOX_TOKEN env var
  vlm_enabled: bool?
  openai_api_key: password?    # Mapped to OPENAI_API_KEY env var
  openai_model: str?

options:
  config_path: /config/homesec/config.yaml
  log_level: info
  database_url: ""
  storage_type: local
  storage_path: /media/homesec/clips
  vlm_enabled: false

# Secret handling: Add-on options with `password` type are mapped to
# environment variables by the run script. Config YAML stores env var
# names (e.g., `token_env: DROPBOX_TOKEN`), not actual secret values.
# This follows HomeSec's existing pattern for credential management.
#
# Bootstrap-only: These options are used to generate the initial config.yaml.
# After first run, use the REST API or HA integration to modify configuration.
# Changing options here won't affect an existing config file.

startup: services
stage: stable
advanced: true
privileged: []
apparmor: true

# Watchdog for auto-restart (returns 200 if pipeline running, even if DB degraded)
watchdog: http://[HOST]:[PORT:8080]/api/v1/health
```

### Constraints

- Must support both amd64 and aarch64
- `homeassistant_api: true` injects SUPERVISOR_TOKEN
- Ingress provides secure access without port exposure
- Watchdog ensures auto-restart on failure

---

## 3.3 Repository Manifest

**File**: `repository.json` (at repo root)

```json
{
  "name": "HomeSec",
  "url": "https://github.com/lan17/homesec",
  "maintainer": "lan17",
  "addons": [
    {
      "name": "HomeSec",
      "slug": "homesec",
      "description": "Self-hosted AI video security pipeline",
      "url": "https://github.com/lan17/homesec",
      "path": "homeassistant/addon/homesec"
    }
  ]
}
```

---

## 3.4 Dockerfile

**File**: `homeassistant/addon/homesec/Dockerfile`

### Interface

```dockerfile
ARG BUILD_FROM=ghcr.io/hassio-addons/base:15.0.8
FROM ${BUILD_FROM}

# Install: python3, ffmpeg, postgresql16, opencv, curl
# Install: homesec from PyPI
# Copy: rootfs (s6-overlay services)
# Set: HEALTHCHECK
```

### Constraints

- Use `ghcr.io/hassio-addons/base` for s6-overlay support
- PostgreSQL 16 for bundled database
- ffmpeg for video processing
- opencv for YOLO filter (if using)
- Install homesec from PyPI (specific version)

---

## 3.5 s6-overlay Services

### Service Dependency Order

```
base → postgres-init (oneshot) → postgres (longrun) → homesec (longrun)
```

### postgres-init (oneshot)

**File**: `rootfs/etc/s6-overlay/s6-rc.d/postgres-init/up`

Initializes PostgreSQL if not already initialized:
- Creates `/data/postgres/data` directory
- Runs `initdb`
- Creates `homesec` database
- Sets postgres password

### postgres (longrun)

**File**: `rootfs/etc/s6-overlay/s6-rc.d/postgres/run`

Runs PostgreSQL in foreground:
```bash
exec su postgres -c "postgres -D /data/postgres/data"
```

### homesec (longrun)

**File**: `rootfs/etc/s6-overlay/s6-rc.d/homesec/run`

Waits for PostgreSQL, generates config if missing, runs HomeSec:
- Reads options from `/data/options.json` via Bashio
- Maps secret options (dropbox_token, openai_api_key) to environment variables
- Waits for `pg_isready`
- **Bootstrap only**: Generates initial config if not exists (options used only on first run)
- Runs `python3 -m homesec.cli run --config /config/homesec/config.yaml`

### Constraints

- Use Bashio for reading options and logging
- HomeSec must wait for PostgreSQL before starting
- Generate default config with home_assistant notifier pre-configured
- Use bundled PostgreSQL unless `database_url` option is set
- **Options are bootstrap-only**: After initial config generation, all changes go through
  the API. Changing add-on options won't affect an existing config file. This avoids
  split-brain between UI options and API-managed config.

---

## 3.6 Ingress Configuration

**File**: `rootfs/etc/nginx/includes/ingress.conf`

```nginx
location / {
    proxy_pass http://127.0.0.1:8080;
    proxy_http_version 1.1;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
    proxy_connect_timeout 60s;
    proxy_send_timeout 60s;
    proxy_read_timeout 60s;
}
```

---

## 3.7 Default Generated Config

When HomeSec starts for the first time, it generates:

**File**: `/config/homesec/config.yaml`

```yaml
version: 1

cameras: []

storage:
  backend: local  # or dropbox based on options
  config:
    path: /media/homesec/clips

state_store:
  dsn_env: DATABASE_URL

notifiers:
  - backend: home_assistant
    config: {}  # Uses SUPERVISOR_TOKEN automatically

server:
  enabled: true
  host: 0.0.0.0
  port: 8080
```

---

## File Changes Summary

| File | Change |
|------|--------|
| `repository.json` | New file at repo root |
| `homeassistant/addon/README.md` | Add-on documentation |
| `homeassistant/addon/homesec/config.yaml` | Add-on manifest |
| `homeassistant/addon/homesec/Dockerfile` | Container build |
| `homeassistant/addon/homesec/build.yaml` | Build config |
| `homeassistant/addon/homesec/DOCS.md` | User documentation |
| `homeassistant/addon/homesec/CHANGELOG.md` | Version history |
| `homeassistant/addon/homesec/icon.png` | 512x512 icon |
| `homeassistant/addon/homesec/logo.png` | 256x256 logo |
| `homeassistant/addon/homesec/rootfs/...` | s6-overlay services |
| `homeassistant/addon/homesec/translations/en.yaml` | UI strings |

---

## Test Expectations

**Note**: Add-on testing requires running Home Assistant. Automated testing is limited.

### Manual Test Cases

**Installation**
- Given HA OS or Supervised, when add repo URL and install, then add-on appears in Supervisor
- Given add-on installed, when start, then PostgreSQL and HomeSec start successfully

**PostgreSQL**
- Given fresh install, when add-on starts first time, then postgres-init creates database
- Given existing install, when add-on restarts, then postgres-init skips (already initialized)

**Configuration**
- Given fresh install, when add-on starts, then default config generated at /config/homesec/config.yaml
- Given database_url option set, when add-on starts, then uses external database instead of bundled

**Ingress**
- Given add-on running, when open from sidebar, then API accessible via ingress
- Given add-on running, when call /api/v1/health via ingress, then returns healthy

**Events**
- Given add-on running, when clip triggers alert, then homesec_alert event appears in HA

---

## Verification

```bash
# Build add-on locally (requires Docker)
cd homeassistant/addon/homesec
docker build -t homesec-addon .

# Test s6-overlay scripts syntax
shellcheck rootfs/etc/s6-overlay/s6-rc.d/*/run
shellcheck rootfs/etc/s6-overlay/s6-rc.d/*/up

# Install in HA (manual)
# 1. Add repository: https://github.com/lan17/homesec
# 2. Install HomeSec add-on
# 3. Check logs in Supervisor
# 4. Open ingress panel
```

---

## Definition of Done

- [ ] Add-on installs successfully from monorepo URL
- [ ] Bundled PostgreSQL starts and initializes correctly
- [ ] HomeSec waits for PostgreSQL before starting
- [ ] SUPERVISOR_TOKEN enables zero-config HA Events API
- [ ] Ingress provides access to API
- [ ] Configuration options (log_level, storage_type, etc.) work
- [ ] Watchdog restarts on failure
- [ ] Logs accessible in HA Supervisor
- [ ] Works on both amd64 and aarch64
- [ ] Default config includes home_assistant notifier
