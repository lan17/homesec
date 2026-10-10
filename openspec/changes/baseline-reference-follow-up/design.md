# Design

## Context

See [proposal.md](proposal.md) for motivation and scope. The [archived baseline design](../archive/2026-10-10-baseline-existing-behavior/design.md#review-gaps-and-documentation-disagreements) records the source/test evidence and gaps behind the eleven transferred tasks. The maintained specs and [behavior reference](../../../docs/reference.md) describe the current contracts and verification limits; archived notes describe the reviewed bootstrap snapshot.

The work spans existing setup, sources, retention/cleanup, analysis, playback, native media, plugin, health, and backup boundaries. Its task descriptions identify the specific guides and missing checks already assessed during baseline research. No new verification framework or schema generator is needed.

## Goals / Non-Goals

**Goals:**

- Keep each unfinished task independently trackable under its original ID and wording.
- Improve evidence and guides using existing test boundaries and documented commands.
- Keep current limitations accurate as checks are added or existing coverage is confirmed.

**Non-Goals:**

- Reapply the bootstrap's ADDED requirements or duplicate its limitation inventory.
- Change public contracts, runtime policies, schemas, or dependencies through documentation cleanup.
- Claim application/device conformance from source inspection or structural validation.

## Decisions

Use the existing `spec-driven` workflow with `skip_specs: true`, a proposal, this design, and the transferred task checklist. These are documentation/verification tasks with no specified-behavior changes. Copying the original deltas would create stale ADDED operations; inventing replacement requirements would change the scope. A separate schema is unnecessary.

Keep `docs/reference.md` as the live limitation/evidence reference. Link to archived evidence for provenance instead of maintaining another gap table here. Read current implementation and tests before carrying out each identified check; the archive is a snapshot. Any disputed intended behavior or architecture choice requires Leonidas's confirmation and a separate reviewed change with applicable deltas.

Reuse tests that already establish observable behavior. Add focused coverage only where an identified gap remains, follow `TESTING.md`, and mock existing external boundaries. Update the affected guide and reference evidence note alongside verified results.

## Risks / Trade-offs

- **Older evidence can become stale.** Compare the relevant maintained contract, source, and tests before implementing each task; retain explicit limits where execution is incomplete.
- **Guide cleanup could accidentally promise revised behavior.** Preserve reviewed semantics and separate requested runtime changes from this follow-up.
- **Validation can write to a database.** Follow AGENTS.md and obtain immediate approval before database-writing checks, or an explicit narrower scope for the future implementation. PR118's docs-only publication checks do not certify completion of these tasks.

## Migration Plan

The task transfer and bootstrap archive are bookkeeping changes in PR118. Later implementation proceeds by capability group, with the repository's required validation and truthful evidence updates before publication. No deployment or data migration is part of this plan.
