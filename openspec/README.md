# HomeSec specifications and OpenSpec workflow

Maintained public behavior lives in [capability specs](specs/). The
[behavior reference](../docs/reference.md) connects those contracts to operator
guides, exact API shapes, and known limitations. Proposed changes and review
evidence live under `changes/`; merging a Git PR does not itself sync delta specs.

Developer tooling: OpenSpec **1.14.1**, Node.js **20.19+**. Use the existing
`spec-driven` schema and `core` workflow profile with the Codex integration.

Select a supported Node runtime before installing the pinned CLI. The local
setup uses Node.js 22.22.0.

```sh
OPENSPEC_TELEMETRY=0 npm install --global @fission-ai/openspec@1.14.1
OPENSPEC_TELEMETRY=0 openspec --version
```

Set `OPENSPEC_TELEMETRY=0` when running the CLI to disable telemetry. CLI
installation and any local launchers are machine-local, outside this repository.

The project configuration lives in `config.yaml`. Codex's six generated core
skills live in `../.agents/skills/openspec-*`; keep project-specific guidance in
`config.yaml` and the existing `AGENTS.md` files rather than editing generated
skills. Configuration rules are agent guidance, not automated correctness gates.

Start a discussion with `openspec-explore`, or draft a change with
`openspec-propose`, selected from Codex's Skills picker. CLI/IDE invocations use
`$openspec-explore` and `$openspec-propose`. Implementation uses
`openspec-apply-change` after the proposal is reviewed.

## Maintaining behavior and references

Read the affected maintained specs before proposing a feature or behavior change.
Use an OpenSpec change with deltas for the existing capability names, meaningful
boundary verification, and affected reference/guide updates in the same PR.
Behavior-preserving internal refactors do not need an invented spec delta.

After review and verification, use `openspec-sync-specs` to update maintained
specs, or `openspec-archive-change` to sync and archive completed work. Preserve
known limitations in the reference page and keep links independent of an active
change directory. Specs own behavioral semantics; the existing OpenAPI exporter
owns exact API shapes, and operator guides own procedures. No additional document
generator or copied schema inventory is required.

## Regenerating tooling

When regenerating this project's integrations, explicitly select the same
profile and target:

```sh
OPENSPEC_TELEMETRY=0 openspec init --tools codex --profile core --no-animation
```

Check any generated diff before committing. The existing repository validation
and approval requirements still apply.
