# Configuration editing

HomeSec keeps installation settings in one YAML document. The settings UI and
camera API use `ConfigManager` to validate and atomically persist that document.
Postgres holds clip history and telemetry; configuration editing does not require
a database connection.

## Editor scope

The storage editor supports the configured Local or Dropbox backend. It edits
provider settings and storage paths while preserving unspecified options. It
does not switch providers or migrate clips. Changing
Dropbox credential references can change the accessible account; existing clips
still require access to the account and paths that contain them.

Notifications edits each configured MQTT/SendGrid instance separately, including
disabled instances and repeated instances of a backend. The existing default
alert-policy threshold can be edited without replacing camera overrides.

Detection edits supported YOLO classes/confidence and OpenAI-compatible VLM
settings. Disabling AI uses `vlm.run_mode: never`. Existing advanced settings are
preserved. Unsupported provider configurations remain read-only. Adding/removing
providers and a generic schema renderer are outside this editor version.

## Credentials

Authenticated installations can enter Dropbox credentials, an AI API key, MQTT
credentials, and a SendGrid API key directly in settings. Password inputs are
write-only: the UI shows Configured or Not configured, leaving a blank input
unchanged. Replace enters a new value; Clear explicitly removes it. Advanced
keeps the existing environment-variable reference workflow available. When API
authentication is disabled, direct credential entry is unavailable; enable the
existing server authentication settings first.

YAML stores generated environment-variable references, never UI-entered values.
The values live in `.homesec/credentials.json` beside the YAML, with a private
directory (`0700`) and file (`0600`) on POSIX systems. This file is plaintext and
must be included in protected installation backups. HomeSec refuses unsafe
permissions, ownership, or symlink paths. The file and directory are ignored by
Git. HomeSec does not modify the launch environment's `.env` file.

Every replacement or clear gets a new private reference, so it changes the
configuration revision without publishing a secret-derived fingerprint. Save
writes the private file before committing YAML; a failed YAML save preserves
the previous references and running credentials. Previous private values are
retained for the active process and YAML backup, including after Clear. Clear
stops using a value after Apply; it does not erase historical values from the
private file. Restore YAML and the private file together when recovering an
installation.

Credential changes require a full process restart. Until Apply completes, the
old credentials remain active. At startup, HomeSec installs only its generated
private references into the process environment before constructing providers;
the worker inherits the same snapshot. External environment-variable names and
values are preserved. Connection checks are unavailable for unsaved secret
drafts or saved credentials awaiting activation.

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
Credentials, storage, database connection, backup maintenance, or server settings require a
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

Saved YAML, backup, and private credential files use mode `0600` on POSIX systems. A permission or
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

All configuration routes use the existing API authentication and require normal
mode. Writing credential values additionally requires authentication to be
enabled and a valid API key.

- `GET /api/v1/config` returns redacted saved `config`, `saved_config_version`,
  `active_config_version`, and `apply_required` (`none`, `reload`, or `restart`).
  `credentials` reports only `configured` and `source` for each supported path;
  `credentials_editable` reports whether direct entry is available.
- `PATCH /api/v1/config` requires `expected_config_version`. The allowlist is
  `storage`, `filter`, `vlm`, `alert_policy`, and `notifiers`. It saves only and
  returns the same response shape as GET. Provider changes are rejected.
  Optional `credentials` maps supported paths such as `vlm.config.api_key_env`
  or `notifiers.0.config.auth.password_env` to a new string value or null to
  clear. Omitted paths preserve their values. Arbitrary environment-variable
  writes and manually supplied private references are rejected.
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
