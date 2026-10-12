# Proposal

## Why

The reviewed whole-product baseline is maintained in `openspec/specs/`, but eleven deeper-verification and guide-reconciliation tasks remain. Track them independently so the completed bootstrap can be archived before later behavior changes make its original ADDED deltas stale.

## What Changes

- Carry forward the eleven unfinished tasks from the [archived bootstrap](../archive/2026-10-10-baseline-existing-behavior/tasks.md), preserving their IDs and wording.
- Assess the specifically identified test gaps at existing boundaries, reuse existing coverage, and add tests only where justified.
- Reconcile the affected operator/developer guides with the maintained contracts and update [reference evidence limits](../../../docs/reference.md) as verification improves.
- Keep unresolved behavior decisions and corrective runtime changes in separately approved changes.

## Capabilities

### New Capabilities

None. This follow-up adds verification and reference work, not public behavior.

### Modified Capabilities

None. `.openspec.yaml` declares `skip_specs: true`; no delta specs are needed for the current scope. Any proposed change to specified behavior requires a separate reviewed change with deltas.

## Impact

The planned work affects existing tests and operator/developer documentation across the eight maintained capabilities. The [archived design](../archive/2026-10-10-baseline-existing-behavior/design.md) provides the original source/test evidence and review gaps; maintained specs and `docs/reference.md` remain the current behavior and limitation references. This proposal does not implement the transferred work or change application code, API schemas, dependencies, or runtime defaults.
