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

## Subsequent PR118 review corrections

After the original task transfer, review identified event-list failure masking (task 6.3), bounded redaction (2.3), backend comparison before normalization (2.4), global clip-ID collisions (3.3), and concurrent whole-snapshot overwrites including deletion races (5.4). The original eleven tasks and archived bootstrap remain historical records; the new tasks are additional unchecked verification work.

The maintained specifications and current reference notes are corrected to describe the implementation, not to adopt these defects as intended policies. The redaction and same-backend patch guarantees are narrowed; in-process stage outcomes are distinguished from persisted snapshot preservation; deletion ordering is distinguished from terminal-state safety; and clip identity and notification deduplication scope are explicit. No runtime, schema, or default changes are included.

Source-derived interleavings are not an executed concurrency test. In particular, the state fake's shared mutable objects do not reproduce independent Postgres snapshots. Task 5.4 requires controlled read/write ordering and observable persisted results; task 3.3 covers sequential ID reuse separately from concurrency. Retain the current evidence limits until these checks actually run. Do not re-sync archived ADDED deltas over the corrected maintained specifications.

A later review identified startup upload-retry cancellation losing an unsaved unknown-outcome marker (9.4), cancellation before the talk start message bypassing reservation cleanup (7.3), and missing motion-disabled preconditions for activity/fallback decision reasons (5.5). Corrected requirements distinguish in-memory retry metadata from recovered manifest state and pre-start waiting from camera-session opening. The activity scenario, failed-analysis requirement, and misleading-reason reference note consistently give motion notification precedence.

The cancellation and activity counterexamples were reproduced with retrieved method excerpts, fake transport/storage, and temporary manifests. These checks are not the repository application suite or proof of full startup/shutdown behavior. The added tasks stay unchecked until repository-level coverage is verified. Direct re-review of these corrections is not an independent fresh-context agent review.

The independent Codex review at `ad171a8` also found that clip destinations sanitize camera and basename segments (with possible collisions), storage plugins may return no view URL despite successful upload, and media-token issuance is gated by normal mode and database availability before clip lookup. The maintained storage/playback contracts now distinguish these boundaries. Follow-up tasks 4.4 and 6.4 remain unchecked until behavior is verified at actual storage/pipeline and mounted-route boundaries. No runtime fixes or collision-rejection guarantees are introduced.

## Direct review of the current-behavior baseline

The direct reviews at `9927085` identified API error-serialization and credential-comparison limits, authorization alternatives, Unicode media-header and staging-cleanup failures, upload preparation outside retries, backup housekeeping reclassification, cleanup's YOLO-shaped applicability and zero-frame deletion consequence, and selection-time retention races. The maintained specs/reference now describe those boundaries. Normal patch algebra, disabled-component validation, camera-reference deletion, probe meanings, time origins, attempt budgets, and structured-analysis interpretation are explicit as well. A subsequent correction review also narrowed FTP unlink and retention-deletion success guarantees.

The initial 77 requirements and 222 scenarios remain historical promotion counts, not acceptance targets. The [scenario evidence register](../../../docs/behavior-evidence.md) accounts for the corrected inventory and distinguishes retained source review from isolated checks and unverified production-boundary coverage. New characterization scenarios describe limitations without making accidental failures desirable permanent policies. The older bootstrap and all pre-existing unchecked task lines are retained.

New tasks 2.5, 3.4, 4.5, 4.6, 5.6, 5.7, 6.5, 8.3, 9.5, and 10.1 require verification at the stated real boundaries. Prior and rerun isolated harnesses use excerpts, fake external dependencies, and temporary files; they do not complete these repository-level tasks or prove full application/native/device conformance. Runtime fixes remain separate.

## Approved correction review on 2026-10-18

The maintained setup contract now distinguishes limited validation-only checks from full candidate validation on apply. The generated-contract claim is scoped to declared models because custom validation-error envelopes can differ from default OpenAPI schemas. Actual-service and mounted-contract-route characterizations reproduced both limits with fake application boundaries and no runtime/database operations; they are not committed regression tests. New tasks 2.6 and 8.4 retain that verification work. The original 31 unchecked task lines remain unchanged.

Long requirement summaries were shortened for strict OpenSpec validation, retaining their guards in the relevant scenarios or explicitly linked current reference notes. Requirement and existing scenario names remain stable; concrete scenarios make the two newly verified limitations and previously described patch branches explicit. Native viewer-cap evidence now cites the existing admission/timeout integration test without claiming a new local native test run. The archived bootstrap and application behavior remain unchanged.

## Risks / Trade-offs

- **Older evidence can become stale.** Compare the relevant maintained contract, source, and tests before implementing each task; retain explicit limits where execution is incomplete.
- **Guide cleanup could accidentally promise revised behavior.** Preserve reviewed semantics and separate requested runtime changes from this follow-up.
- **Validation can write to a database.** Follow AGENTS.md and obtain immediate approval before database-writing checks, or an explicit narrower scope for the future implementation. PR118's docs-only publication checks do not certify completion of these tasks.

## Migration Plan

The task transfer and bootstrap archive are bookkeeping changes in PR118. Later implementation proceeds by capability group, with the repository's required validation and truthful evidence updates before publication. No deployment or data migration is part of this plan.
