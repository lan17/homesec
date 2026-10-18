# HomeSec UI

React + TypeScript SPA for HomeSec self-serve control plane.

## Toolchain

- Package manager: `pnpm` (version pinned in `packageManager`).
- Node: `20.19+` or `22.12+`; CI uses `22.12.0`.
- Build/runtime: `Vite + React + TypeScript`.
- Router/data: `react-router-dom` + `@tanstack/react-query`.
- Lint: `eslint`.
- Unit tests: `vitest`.

## Commands

From the repository root, use the canonical Makefile proxy targets:

```bash
make ui-install
make ui-run-local
make ui-api-generate
make ui-api-check
make ui-check
```

Inside `ui/`, the equivalent targets omit the `ui-` prefix. Package scripts also
run there (`pnpm api:generate`, `pnpm api:check`, `pnpm check`). `pnpm check`
includes generated API consistency, lint, unit tests, TypeScript, and the
production build; `build` already invokes TypeScript checking.

`make ui-run-local` binds Vite to `0.0.0.0` by default so other devices on your LAN can access it.
Override host/port if needed:

```bash
VITE_DEV_HOST=0.0.0.0 VITE_DEV_PORT=5173 make ui-run-local
```

## API Client Generation

The UI uses generated OpenAPI artifacts in `src/api/generated/`:

- `openapi.json` (exported from FastAPI contract)
- `schema.ts` (generated via `openapi-typescript`)
- `types.ts` + `client.ts` (generated adapter layer consumed by the app)

Generation flow:

1. Export the backend contract schema using `uv run python -m homesec.api.openapi_export` (invoked by the generation script from the repository root).
2. Generate `schema.ts` from OpenAPI using `openapi-typescript`.
3. Generate `types.ts` + `client.ts` adapter artifacts consumed by the app.

From the repository root:

```bash
make ui-api-generate
UV_LOCKED=1 make ui-api-check
```

`api:check` fails if committed generated artifacts are stale. The exporter needs
the Python dependency environment but no running backend or database. Set
`UV_LOCKED=1` to keep its nested `uv run` locked, as CI does.

Generated files describe declared API models. Custom error envelopes can differ
from OpenAPI's default validation-error schema; runtime parsing and canonical
`APIError` handling belong in the handwritten `src/api/client.ts`. See the
[error-contract limitations](../docs/reference.md#integration-contracts).
Never edit `src/api/generated/` directly.

## Environment

- `VITE_API_BASE_URL` (optional): absolute base URL for API requests.
  - Default is same-origin (`/api/...`).
- `VITE_API_PROXY_TARGET` (optional): local dev proxy target used by Vite.
  - Default: `http://127.0.0.1:8081`.

## Structure

```text
src/
  app/         # global providers + shell
  routes/      # top-level route wiring
  features/    # feature pages
  components/  # reusable UI primitives
  api/         # typed client seam + hooks
  styles/      # tokenized theme + global styles
```
