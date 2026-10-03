# Configuration editing

HomeSec keeps installation settings in one YAML document. The settings UI and
camera API use `ConfigManager` to validate and atomically persist that document.
Postgres holds clip history and telemetry; configuration editing does not require
a database connection.

## Editor scope

The storage editor supports the configured Local or Dropbox backend. It edits
provider settings and storage paths while preserving unspecified options. It
does not switch providers, migrate clips, or manage credential values. Changing
Dropbox credential references can change the accessible account; existing clips
still require access to the account and paths that contain them.

Notifications edits each configured MQTT/SendGrid instance separately, including
disabled instances and repeated instances of a backend. The existing default
alert-policy threshold can be edited without replacing camera overrides.

Detection edits supported YOLO classes/confidence and OpenAI-compatible VLM
settings. Disabling AI uses `vlm.run_mode: never`. Existing advanced settings are
preserved. Unsupported provider configurations remain read-only. Adding/removing
providers and a generic schema renderer are outside this editor version.

Checks validate settings and plugin readiness; some providers also probe
connectivity. The AI check verifies local readiness without contacting the API.
Checks do not perform a sample upload, send a notification, or run paid AI inference.

## Save and Apply

Save validates the merged document before writing it. Omitted fields retain their
saved values, nested dictionaries merge, and explicit null inside a plugin config
clears a key. The UI submits only changed fields. Masked credential placeholders
are rejected; unchanged credential fields must be omitted. Top-level sections
cannot be cleared with null.

Each editor retains the revision it loaded. A conflicting change elsewhere
returns `409 CONFIG_VERSION_CONFLICT`; the unsaved draft remains available until
the operator explicitly discards it and loads the latest saved values.

Apply is a separate action. Request acceptance does not mean activation completed.
The UI monitors the server until the reviewed revision is active or a bounded
timeout/error occurs. Saved values remain saved after an activation failure, and
their pending state survives a browser refresh.

Worker-owned settings use the existing runtime reload and rollback mechanism.
Storage, database connection, backup maintenance, or server settings require a
full process restart. Ordinary worker reload refuses a saved document with such
pending changes, including reloads requested through the camera API. This prevents
the worker moving to different storage than playback and backups.

Camera edits that save successfully return their saved result even if a requested
reload is refused. The response includes `restart_required: true` and a typed
`apply_error`; the UI refreshes the saved camera list and explains that activation
is still pending. A validation or persistence failure still returns an HTTP error.

## Deployment

The configuration directory must be writable. Atomic saves create a sibling
temporary file and backup, then replace the YAML file. Mount the directory, not
an individual file, for example:

```yaml
volumes:
  - ./config:/config
  - ./.env:/config/.env:ro
```

Saved YAML and backup files use mode `0600` on POSIX systems. A permission or
read-only filesystem failure returns `503 CONFIG_SAVE_FAILED` with an operator
message; the UI does not claim success.

A process restart gracefully shuts HomeSec down with exit code 42. Docker Compose
uses `restart: unless-stopped`; another supervisor must likewise restart HomeSec
on this exit. A manual CLI launch requires starting the command again. Applying
settings interrupts recording while the worker/process is replaced. The UI
waits up to 45 seconds for activation; slower starts can be checked with Refresh.

Changing environment variable names in YAML is supported. Changing credential
values in the launch environment requires a process/container restart so the
new values are inherited. The settings UI does not write `.env` files.

## API

All configuration routes require the existing API authentication and normal mode.

- `GET /api/v1/config` returns redacted saved `config`, `saved_config_version`,
  `active_config_version`, and `apply_required` (`none`, `reload`, or `restart`).
- `PATCH /api/v1/config` requires `expected_config_version`. The allowlist is
  `storage`, `filter`, `vlm`, `alert_policy`, and `notifiers`. It saves only and
  returns the same response shape as GET. Provider changes are rejected.
- `notifiers` is a list of patches with the existing entry's `index` and optional
  `enabled`/`config`; untouched entries and order are preserved. Duplicate or
  out-of-range indexes are rejected.
- `POST /api/v1/config/apply` requires `expected_config_version` and returns
  asynchronous acceptance, `action`, `target_config_version`, and an optional
  `target_generation`. It does not persist another document.

Save and apply acceptance share the configuration manager's mutation lock. Once
a process restart is accepted, subsequent global or camera writes are rejected
with `409 CONFIG_APPLY_IN_PROGRESS` until the new process starts. Existing camera
CRUD remains available with its existing routes and patch behavior.
