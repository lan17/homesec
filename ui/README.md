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
pnpm check
```

`pnpm check` runs lint, unit tests, typecheck, and production build.

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
