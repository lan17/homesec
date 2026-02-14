# Speakeasy Workflow

`pnpm api:generate` treats Speakeasy as the primary generator when both of the following are true:

1. `speakeasy` CLI is available in `PATH` (or `SPEAKEASY_BIN` points to it)
2. `ui/.speakeasy/workflow.yaml` exists

If either is missing, generation automatically falls back to `openapi-typescript`.

The codegen script exports a deterministic schema and provides the path to Speakeasy through:

- `HOMESEC_OPENAPI_SCHEMA_PATH`

Your `workflow.yaml` should reference that environment variable for schema input.
