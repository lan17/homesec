# HomeSec UI

React + TypeScript SPA for HomeSec self-serve control plane.

## Toolchain

- Package manager: `pnpm` (pinned in `packageManager`).
- Build/runtime: `Vite + React + TypeScript`.
- Router/data: `react-router-dom` + `@tanstack/react-query`.
- Lint: `eslint`.
- Unit tests: `vitest`.

## Commands

```bash
pnpm install
pnpm dev
pnpm api:generate
pnpm api:check
pnpm check
```

`pnpm check` runs lint, unit tests, typecheck, and production build.

Make target shortcuts are available too:

```bash
make run-local
make check
```

`make run-local` binds Vite to `0.0.0.0` by default so other devices on your LAN can access it.
Override host/port if needed:

```bash
VITE_DEV_HOST=0.0.0.0 VITE_DEV_PORT=5173 make run-local
```

## API Client Generation

The UI uses generated OpenAPI artifacts in `src/api/generated/`:

- `openapi.json` (exported from FastAPI contract)
- `schema.ts` (generated via `openapi-typescript`)
- `types.ts` + `client.ts` (generated adapter layer consumed by the app)

Generation flow:

1. Export OpenAPI schema from backend using `uv run python -m homesec.api.openapi_export`.
2. Generate `schema.ts` from OpenAPI using `openapi-typescript`.
3. Generate `types.ts` + `client.ts` adapter artifacts consumed by the app.

Use:

```bash
pnpm api:generate
pnpm api:check
```

`api:check` fails if committed generated artifacts are stale.

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
