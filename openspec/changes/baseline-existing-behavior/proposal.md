# Proposal

## Why

HomeSec began this bootstrap with an OpenSpec workflow but no maintained capability specifications. Its public behavior is spread across implementation, tests, configuration models, and documents, making it difficult to review behavior changes or maintain reliable reference documentation.

## What Changes

- Establish a first-pass, whole-product baseline of already implemented public behavior, with concrete scenarios for important normal paths, failures, lifecycle transitions, security, and compatibility.
- Establish eight maintained capability specs under `openspec/specs/` from the reviewed capability deltas. These are new entries in the specification inventory, not new runtime features.
- Record implementation and existing-test evidence, explicit coverage limits, and unresolved code/documentation disagreements in the design and review tasks.
- Connect maintained specs to `docs/reference.md`, preserving reviewed known limitations and operator-guide links. Reference documentation draws behavioral semantics from the maintained specs and precise API shapes from existing schema generation.
- Keep pending features and corrective behavior changes separate from this baseline. This proposal does not change application behavior or introduce a documentation generator.

## Capabilities

### New Capabilities

- `setup-configuration`: first-run setup, camera configuration, validation, saved configuration, and runtime activation.
- `capture-recording`: source ingestion, motion-triggered recording, reconnection, and recording lifecycle.
- `storage-retention`: clip upload outcomes, retries, local retention eligibility, and clip deletion.
- `analysis-alerts`: filter and analysis triggers, alert decisions, notifier outcomes, and partial failures.
- `events-playback`: event browsing, filtering, pagination, detail views, media access, and playback.
- `preview-talk`: live preview and push-to-talk availability, session lifecycles, concurrency, and browser interaction.
- `integration-contracts`: API authentication/errors, browser authentication and route compatibility, notification payload semantics, and supported plugin extension contracts.
- `operations-maintenance`: CLI operation, health, diagnostics, shutdown, and database maintenance/backup responsibilities.

### Modified Capabilities

None. The pre-bootstrap maintained specification inventory was empty; these deltas introduce the eight new capabilities.

## Impact

The planning artifacts cover the UI, API, CLI, configuration, camera sources, processing pipeline, integrations, and operator documentation. Application code, dependencies, runtime defaults, external services, and API schemas are unchanged by the proposed baseline.

The baseline targets the fetched `origin/main` after the OpenSpec setup merge on 2026-10-10. It excludes pending work such as activation-only event-card media loading and the broader configuration editor. The reviewed baseline keeps unclear product promises and incomplete evidence explicit in the reference notes; corrective runtime changes remain separate. The active change retains review evidence and the remaining deeper-verification/guide follow-ups.
