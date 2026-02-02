# HomeSec Add-on

HomeSec is a self-hosted AI video security pipeline. This add-on runs HomeSec
inside Home Assistant with optional bundled PostgreSQL and ingress access.

## Configuration

Options are **bootstrap-only**. On first start the add-on generates
`/config/homesec/config.yaml` from these options. After that, configure HomeSec
via the REST API or the Home Assistant integration.

### Options

- `config_path`: Path to HomeSec config file.
- `log_level`: Log level for HomeSec (`debug`, `info`, `warning`, `error`).
- `database_url`: Optional external Postgres DSN. If empty, bundled Postgres is used.
- `db_password`: Password for bundled Postgres (leave blank to auto-generate).
- `storage_type`: `local` or `dropbox`.
- `storage_path`: Root path for local storage (used when `storage_type=local`).
- `dropbox_token`: Dropbox token (used when `storage_type=dropbox`).
- `vlm_enabled`: Enable VLM analysis (OpenAI). When false, VLM is disabled.
- `openai_api_key`: OpenAI API key (required when `vlm_enabled=true`).
- `openai_model`: OpenAI model name (default: `gpt-4o`).

## Notes

- The add-on uses `SUPERVISOR_TOKEN` automatically for HA event notifications.
- Ingress provides secure access to the REST API.
- The bundled database is not exposed outside the add-on container.
- Auto-generated Postgres passwords are stored at `/data/postgres/db_password`.
- Changing `db_password` after initialization does not update the existing DB user.
