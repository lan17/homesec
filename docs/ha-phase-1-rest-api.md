# Phase 1: REST API for Configuration

**Goal**: Enable remote configuration and monitoring of HomeSec via HTTP API.

**Estimated Effort**: 5-7 days

**Dependencies**: Phase 0 (Prerequisites)

---

## Overview

This phase adds a FastAPI-based REST API to HomeSec for:
- Camera CRUD operations
- Clip listing and management
- Event history
- System stats and health
- Configuration management with optimistic concurrency

---

## 1.1 API Framework Setup

### Files

- `src/homesec/api/__init__.py`
- `src/homesec/api/server.py`
- `src/homesec/api/dependencies.py`

### Interfaces

**APIServer** (`server.py`)
```python
def create_app(app_instance: Application) -> FastAPI:
    """Create the FastAPI application.

    - Stores Application reference in app.state.homesec
    - Configures CORS from server_config.cors_origins
    - Registers all route modules
    """
    ...

class APIServer:
    """Manages the API server lifecycle."""

    def __init__(self, app: FastAPI, host: str, port: int): ...
    async def start(self) -> None: ...
    async def stop(self) -> None: ...
```

**Dependencies** (`dependencies.py`)
```python
async def get_homesec_app(request: Request) -> Application:
    """Get the HomeSec Application instance from request state.

    Raises HTTPException 503 if not initialized.
    """
    ...

async def verify_api_key(request: Request, app=Depends(get_homesec_app)) -> None:
    """Verify API key if authentication is enabled.

    - Checks app.config.server.auth_enabled
    - Expects Authorization: Bearer <token>
    - Raises HTTPException 401 on failure
    """
    ...
```

### Constraints

- All endpoints must be `async def`
- No blocking operations (use `asyncio.to_thread` for file I/O)
- CORS origins configurable via `FastAPIServerConfig.cors_origins`
- Port configurable, replaces old `HealthConfig`

---

## 1.2 Config Management

### Files

- `src/homesec/config/manager.py`
- `src/homesec/config/loader.py`

### Interfaces

**ConfigManager** (`manager.py`)
```python
class ConfigUpdateResult(BaseModel):
    """Result of a config update operation."""
    restart_required: bool = True

class ConfigManager:
    """Manages configuration persistence (last-write-wins semantics)."""

    def __init__(self, base_paths: list[Path], override_path: Path): ...

    def get_config(self) -> Config:
        """Get the current merged configuration."""
        ...

    async def add_camera(
        self,
        name: str,
        enabled: bool,
        source_backend: str,
        source_config: dict,
    ) -> ConfigUpdateResult:
        """Add a new camera to the override config.

        Raises:
            ValueError: If camera name already exists
        """
        ...

    async def update_camera(
        self,
        camera_name: str,
        enabled: bool | None,
        source_config: dict | None,
    ) -> ConfigUpdateResult:
        """Update an existing camera in the override config.

        Raises:
            ValueError: If camera doesn't exist
        """
        ...

    async def remove_camera(
        self,
        camera_name: str,
    ) -> ConfigUpdateResult:
        """Remove a camera from the override config.

        Raises:
            ValueError: If camera doesn't exist
        """
        ...
```

**ConfigLoader** (`loader.py`)
```python
def load_configs(paths: list[Path]) -> Config:
    """Load and merge multiple YAML config files.

    Merge semantics:
    - Files loaded left to right, rightmost wins
    - Dicts: deep merge (recursive)
    - Lists: merge (union, preserving order, no duplicates)
    """
    ...

def deep_merge(base: dict, override: dict) -> dict:
    """Deep merge two dicts.

    - Nested dicts are merged recursively
    - Lists are merged (union)
    - Scalars: override wins
    """
    ...
```

### Constraints

- Override file written atomically (write temp, fsync, rename)
- All file I/O via `asyncio.to_thread`
- Base config is read-only; all mutations go to override file
- Override file is machine-owned (no comment preservation needed)

---

## 1.2.1 ClipRepository Extensions

**File**: `src/homesec/repository/clip_repository.py`

The existing `ClipRepository` needs these additional methods for the API:

### Interface

```python
class ClipRepository:
    """Coordinates state + event writes with best-effort retries."""

    # ... existing methods ...

    # NEW: Read methods for API
    async def get_clip(self, clip_id: str) -> ClipStateData | None:
        """Get clip state by ID."""
        ...

    async def list_clips(
        self,
        *,
        camera: str | None = None,
        status: ClipStatus | None = None,
        alerted: bool | None = None,
        risk_level: str | None = None,
        activity_type: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        offset: int = 0,
        limit: int = 50,
    ) -> tuple[list[ClipStateData], int]:
        """List clips with filtering and pagination.

        Filters:
        - alerted: If True, only clips that triggered notifications
        - risk_level: Filter by analysis risk level (low/medium/high/critical)
        - activity_type: Filter by detected activity type

        Returns (clips, total_count).
        """
        ...

    async def list_events(
        self,
        *,
        clip_id: str | None = None,
        event_type: str | None = None,
        camera: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int = 100,
    ) -> tuple[list[ClipLifecycleEvent], int]:
        """List events with filtering.

        Returns (events, total_count).
        """
        ...

    async def delete_clip(self, clip_id: str) -> None:
        """Mark clip as deleted and delete from storage.

        Uses existing record_clip_deleted() internally.
        Also deletes from storage backend.
        """
        ...

    async def count_clips_since(self, since: datetime) -> int:
        """Count clips created since the given timestamp."""
        ...

    async def count_alerts_since(self, since: datetime) -> int:
        """Count alert events (notification_sent) since the given timestamp."""
        ...

    async def ping(self) -> bool:
        """Health check - verify database is reachable.

        Delegates to StateStore.ping().
        """
        return await self._state.ping()
```

### Constraints

- Must use async SQLAlchemy
- Counts should be efficient (use SQL COUNT, not fetch all)
- `count_alerts_since` counts events where `event_type='notification_sent'`
- `list_clips` and `list_events` return tuple of (items, total_count) for pagination
- `delete_clip` should delete from both local and cloud storage

---

## 1.3 API Routes

### Files

- `src/homesec/api/routes/__init__.py`
- `src/homesec/api/routes/cameras.py`
- `src/homesec/api/routes/clips.py`
- `src/homesec/api/routes/events.py`
- `src/homesec/api/routes/stats.py`
- `src/homesec/api/routes/health.py`
- `src/homesec/api/routes/config.py`
- `src/homesec/api/routes/system.py`

### Route Summary

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/health` | Health check |
| GET | `/api/v1/config` | Config summary and version |
| GET | `/api/v1/cameras` | List cameras |
| GET | `/api/v1/cameras/{name}` | Get camera |
| POST | `/api/v1/cameras` | Create camera |
| PUT | `/api/v1/cameras/{name}` | Update camera |
| DELETE | `/api/v1/cameras/{name}` | Delete camera |
| GET | `/api/v1/cameras/{name}/status` | Camera status |
| POST | `/api/v1/cameras/{name}/test` | Test camera connection |
| GET | `/api/v1/clips` | List clips (paginated, filterable) |
| GET | `/api/v1/clips/{id}` | Get clip |
| DELETE | `/api/v1/clips/{id}` | Delete clip |
| POST | `/api/v1/clips/{id}/reprocess` | Reprocess clip |
| GET | `/api/v1/events` | List events (filterable) |
| GET | `/api/v1/stats` | System statistics |
| POST | `/api/v1/system/restart` | Request graceful restart |
| GET | `/api/v1/system/health/detailed` | Detailed health with error codes |

### Request/Response Models

**Camera Models**
```python
class CameraCreate(BaseModel):
    name: str
    enabled: bool = True
    source_backend: str  # rtsp, ftp, local_folder
    source_config: dict

class CameraUpdate(BaseModel):
    enabled: bool | None = None
    source_config: dict | None = None

class CameraResponse(BaseModel):
    name: str
    enabled: bool
    source_backend: str
    healthy: bool
    last_heartbeat: float | None
    source_config: dict  # Secrets are env var names, not values

class ConfigChangeResponse(BaseModel):
    restart_required: bool = True
    camera: CameraResponse | None = None
```

### Constraints

- All config-mutating endpoints return `restart_required: True`
- Last-write-wins semantics (no optimistic concurrency in v1)
- Return 503 Service Unavailable when Postgres is down
- Pagination: `page` (1-indexed), `page_size` (default 50, max 100)

---

## 1.4 Server Configuration

**File**: `src/homesec/models/config.py`

### Interface

```python
class FastAPIServerConfig(BaseModel):
    """Configuration for the FastAPI server (replaces HealthConfig)."""

    enabled: bool = True
    host: str = "0.0.0.0"
    port: int = 8080
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])
    auth_enabled: bool = False
    api_key_env: str | None = None  # Env var name, not the key itself

    def get_api_key(self) -> str | None:
        """Resolve API key from environment variable."""
        ...

# Update Config class:
# - Replace `health: HealthConfig` with `server: FastAPIServerConfig`
```

---

## 1.5 CLI Updates

**File**: `src/homesec/cli.py`

### Interface

```python
# Support multiple --config flags
# Example: homesec run --config base.yaml --config overrides.yaml

@click.option(
    "--config",
    "config_paths",
    multiple=True,
    type=click.Path(exists=True),
    help="Config file(s). Can be specified multiple times. Later files override earlier.",
)
```

### Constraints

- Order matters: files loaded left to right
- Default override path: `config/ha-overrides.yaml` (if exists)
- Override path can be explicitly passed as last `--config`

---

## File Changes Summary

| File | Change |
|------|--------|
| `src/homesec/api/__init__.py` | New package |
| `src/homesec/api/server.py` | FastAPI app factory and server |
| `src/homesec/api/dependencies.py` | Request dependencies |
| `src/homesec/api/routes/*.py` | All route modules |
| `src/homesec/config/manager.py` | Config persistence |
| `src/homesec/config/loader.py` | Multi-file config loading |
| `src/homesec/models/config.py` | Add `FastAPIServerConfig` |
| `src/homesec/repository/clip_repository.py` | Add read/list/count methods |
| `src/homesec/cli.py` | Support multiple `--config` flags |
| `src/homesec/app.py` | Integrate API server |
| `pyproject.toml` | Add fastapi, uvicorn dependencies |

---

## Test Expectations

### Fixtures Needed

- `test_client` - FastAPI TestClient with mocked Application
- `mock_config_store` - ConfigManager that operates in-memory
- `mock_repository` - ClipRepository with canned data
- `sample_camera_config` - Valid CameraConfig for RTSP
- `sample_clip` - Clip with all fields populated

### Test Cases

**Camera CRUD**
- Given no cameras, when POST /cameras with valid data, then 201 and camera created
- Given camera "front", when GET /cameras/front, then returns camera data
- Given camera "front", when DELETE /cameras/front, then 200 and camera removed

**Health**
- Given Postgres is up, when GET /health, then status="healthy"
- Given Postgres is down, when GET /health, then 503 and status="unhealthy"

**Clips**
- Given 100 clips, when GET /clips?page=2&page_size=10, then returns clips 11-20

**ClipRepository**
- Given 5 clips created today and 10 yesterday, when `count_clips_since(today_start)`, then returns 5
- Given 0 alerts, when `count_alerts_since(any_date)`, then returns 0
- Given clips with mixed cameras, when `list_clips(camera="front")`, then returns only "front" clips
- Given 10 clips (5 alerted, 5 not), when `list_clips(alerted=True)`, then returns only 5 alerted clips
- Given clips with mixed risk levels, when `list_clips(risk_level="high")`, then returns only high-risk clips
- Given clip exists, when `delete_clip(clip_id)`, then clip marked deleted and storage files removed
- Given StateStore is up, when `ping()`, then returns True

---

## Verification

```bash
# Run API tests
pytest tests/unit/api/ -v

# Start server manually
homesec run --config config/example.yaml

# Test endpoints
curl http://localhost:8080/api/v1/health
curl http://localhost:8080/api/v1/cameras
curl http://localhost:8080/api/v1/config

# Check OpenAPI docs
open http://localhost:8080/docs
```

---

## Definition of Done

- [ ] FastAPI server starts and serves requests
- [ ] All CRUD operations for cameras work
- [ ] Config changes are validated and persisted to override file
- [ ] Config changes return `restart_required: true`
- [ ] `/api/v1/system/restart` triggers graceful shutdown
- [ ] Clip listing with pagination and filtering works
- [ ] Event history API works
- [ ] Stats endpoint returns correct counts
- [ ] OpenAPI documentation is accurate at `/docs`
- [ ] CORS works for configured origins
- [ ] API authentication (when enabled) works
- [ ] Returns 503 when Postgres is unavailable
- [ ] Multiple `--config` flags work in CLI
- [ ] All tests pass
