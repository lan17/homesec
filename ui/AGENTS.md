# UI Agent Guide

Scope: applies to everything under `/Users/levneiman/code/homesec3/ui`.

## Goals

- Keep UI delivery fast, typed, and deterministic.
- Prefer simple feature slices over clever abstractions.
- Preserve clean boundaries: generated API types -> API client -> hooks -> feature pages/components.

## Stack + Commands

- Package manager/runtime: `pnpm` + `make`.
- Prefer root Makefile proxy targets (canonical):
  - `make ui-run-local`
  - `make ui-api-generate`
  - `make ui-api-check`
  - `make ui-lint`
  - `make ui-test`
  - `make ui-build`
  - `make ui-check` (required before commit)
- Fallback only (when debugging): `pnpm --dir ui <script>`.
- Node: use `20.19+` or `22.12+` (Vite warns on older versions).

## Project Structure

- `src/api/generated/*`: generated OpenAPI artifacts. Never hand-edit.
- `src/api/client.ts`: runtime parsing, canonical `APIError`, request/query serialization.
- `src/api/hooks/*`: react-query wrappers only (thin, typed).
- `src/features/<feature>/*`: route-level logic, filter state, presentation helpers.
- `src/components/ui/*`: reusable presentational components.
- `src/styles/*`: global tokens/layout/theme styles.

## API Contract Rules

- If backend API changes:
  1. update `scripts/api_codegen.mjs` when needed,
  2. run `make ui-api-generate`,
  3. update `src/api/client.ts` parser/adapter,
  4. update tests in `src/api/client.test.ts`.
- Treat API payloads as untrusted: parse/validate shape in client parser functions.
- Preserve canonical backend error behavior (`detail`, `error_code`) in UI messaging.

## State + Query Patterns

- URL is source of truth for list/filter pages.
- Keep conversion pipeline explicit:
  - `URLSearchParams -> typed query`
  - `typed query -> form state`
  - `form state -> typed query`
  - `typed query -> URLSearchParams`
- For keyset pagination, reset cursor history when filter signature changes.
- Keep query keys stable and minimal (e.g., `['clips', query]`).

## Component Patterns

- Page components orchestrate data/hooks + URL state.
- Reusable UI components stay dumb/presentational.
- Extract pure helpers for dense logic (formatting, cursor behavior, query transforms).
- Avoid effect-driven state sync when derived values can be computed from props/URL.

## Styling + Responsive

- Mobile is required (`>=320px`).
- No horizontal page scrolling at mobile widths.
- For tabular data, provide mobile card/list fallback.
- Keep spacing/colors/typography driven by existing tokens and theme variables.

## Testing Expectations

- Include behavioral tests for:
  - query/filter normalization,
  - cursor transitions and filter persistence semantics,
  - API parser and query serialization behavior.
- Use Given/When/Then comments in tests.
- Prefer deterministic unit tests for pure logic; avoid flaky integration tests.

## Do / Don’t

- Do keep generated code and handwritten code clearly separated.
- Do update both UI + backend tests when introducing new query parameters.
- Do keep route params and query param names aligned with backend contract.
- Don’t silently bypass type errors with `any`.
- Don’t add direct fetch calls inside feature pages; go through `api/client.ts`.
- Don’t hand-edit `src/api/generated/*`.

## Pre-Commit Checklist

1. `make ui-check`
2. If backend API changed: `make ui-api-generate` and commit generated artifacts.
3. Ensure filter URLs are shareable/reload-safe.
4. Ensure mobile layout remains usable.
