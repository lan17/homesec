# Tasks

The eleven original unfinished tasks were transferred from the [archived bootstrap](../archive/2026-10-10-baseline-existing-behavior/tasks.md). Original task IDs and wording are retained for traceability. Tasks 2.3, 2.4, 3.3, 5.4, and 6.3 were added during subsequent PR118 review; they are not part of the archived transfer. Read [design.md](design.md), the [maintained specs](../../specs/), and [current reference limits](../../../docs/reference.md) before implementation. The archive records historical evidence; update current reference notes as checks are completed.

## 2. Review setup and configuration

- [ ] 2.2 Reconcile setup and camera-editing guidance in README/DESIGN with bootstrap-only finalization and the saved-file rollback limit; verify the updated guidance does not describe a general configuration editor or transactional file/runtime rollback. Record remaining coverage limits beside the evidence.

- [ ] 2.3 Characterize configuration/camera response redaction through the real redaction path: recognized sensitive strings and URL userinfo are masked, `_env` names remain, and query/path/fragment credentials in URL-named fields can remain. Include URLs with and without userinfo plus unrecognized-key/non-string controls. Update the reference evidence after verification; do not silently implement broader redaction in this documentation follow-up.
- [ ] 2.4 Characterize camera patches through the route and real configuration manager with a temporary config file. Compare omitted backend, exact saved normalized identifier, case-only identifier changes, and a genuine backend switch. Assert persisted settings, including loss of a custom `watch_dir` for a valid case-only `LOCAL_FOLDER` patch and no save when discarded required fields make validation fail. Keep a normalization-order fix separately reviewed.

## 3. Review capture and recording

- [ ] 3.2 Assess manifest-restart and unfinished-file coverage and add a focused source-boundary test only where the existing suite leaves a critical gap; verify the selected check and update README/DESIGN recording guidance to avoid durable replay or universal playable-file promises and obsolete startup-flag examples.

- [ ] 3.3 Characterize two cameras ingesting the same filename stem and a repeated FTP upload after deletion through the actual source/repository boundaries. Use independent persisted snapshots, not shared mutable state references; assert one global identity, replacement camera/path/results/status, preserved database `created_at`, shared lifecycle identity, and reused notification `dedupe_key` when notification is selected. Include distinct-ID and local repeat-suppression controls. Verify retention/playback consequences using the surviving snapshot and update current reference evidence. Namespace or collision-rejection changes require a separate reviewed runtime change.

## 4. Review storage, retention, and deletion

- [ ] 4.2 Review API deletion and cleanup separately using their existing route and cleanup tests; verify mark-before-storage failure, original-local-file handling, non-atomic media deletion, deleted-record addressability/conditional media failures, and best-effort cleanup persistence are correctly documented. Confirm any requested change to Delete semantics separately; no new all-copies deletion policy is established by this baseline. Add focused boundary coverage for default dry-run preservation of original media and workflow writes if existing tests do not establish it.
- [ ] 4.3 Reconcile storage/retention/cleanup reference guidance with the reviewed specs; verify examples distinguish retention, API deletion, and cleanup and promise only the destination syntax checks actually implemented, including the filesystem-confinement limit.

## 5. Review analysis and alerts

- [ ] 5.2 Review detector setting/class discrepancies and the intentional-`never` fallback reason with Leonidas; verify the baseline either remains explicitly bounded or reflects a confirmed current contract. Correct inaccurate README/DESIGN examples; put any requested behavior fix in a separate change.
- [ ] 5.3 Assess critical gaps for disabled-policy suppression, threshold equality, and event-append failure containment; verify each is supported by an existing test, a justified new boundary test, or an explicit retained evidence limit in the design. Update analysis/alert reference guidance in the same group.

- [ ] 5.4 Add deterministic barrier-controlled characterization of competing snapshot writes through the real repository and database boundary. Exercise both upload/filter write orders, a stale upload write after exhausted filter failure, and a stale processing write after API deletion, with every database operation succeeding. Assert persisted metadata/status plus resulting list/media/retention behavior separately from in-process notification outcomes. Use independent serialized snapshots or an approved isolated Postgres database; the existing shared-object state fake is not evidence of isolation. Include sequential controls and verify that collecting the upload result does not repair lost stored fields. Update analysis/storage/events reference evidence without claiming terminal deletion or atomic updates; runtime fixes remain separately scoped.

## 6. Review events and playback

- [ ] 6.2 Assess mounted playback-token refresh coverage and add a timer-boundary UI test if needed; verify refresh behavior with the focused UI check and update playback reference guidance without importing activation-only loading or requiring current eager acquisition timing.

- [ ] 6.3 Add focused characterization coverage for event-list failure masking: given successful authentication and database availability checks, a later connection/query failure currently returns HTTP 200 with an empty page. Exercise the real Postgres store, repository retry handling, and route response with failures at the database boundary, rather than mocking the repository to return an empty page. Cover store-contained failures bypassing retries, exceptions reaching the repository exhausting retries, and invalid-row hydration producing incomplete pages with pagination based on fetched rows. Include a genuine empty-result control and update the current reference evidence note after verification. This records the current limitation, not a policy to preserve; any runtime correction requires a separate reviewed change.

## 7. Review preview and talk

- [ ] 7.2 Assess focused native viewer-cap coverage and add a boundary test if the existing suite does not establish it; verify the selected native check and reconcile `docs/webrtc-preview.md`, `docs/push-to-talk.md`, and relevant deployment guidance. Preserve explicit hardware/codec/browser QA limits.

## 8. Review integration contracts

- [ ] 8.2 Assess production per-entry mixed-success notifier fan-out, installed-entry-point discovery, and validate-before-factory/no-construction coverage; add only justified plugin-boundary tests and verify the selected checks. Reconcile `PLUGIN_DEVELOPMENT.md` and obsolete AGENTS examples with the existing unified registry and correct decorator arguments.

## 9. Review operations and maintenance

- [ ] 9.2 Assess missing-executable startup degradation, private dump permissions, and critical backup recovery coverage; add focused process/filesystem tests where justified and verify them using fake dump boundaries. Reconcile `docs/postgres-backups.md` with durable local-directory/manifest mounting, upload outcomes, retention limits, and operator restore responsibilities without claiming a real restore was tested.
- [ ] 9.3 Update operator reference guidance for CLI validation/cleanup and health/statistics interpretation; verify documented options against current CLI/config models and existing tests. Keep applied cleanup semantics owned by `storage-retention`.

## Validation and publication

For each implemented group, run appropriate checks and update its guide/evidence notes together. Before committing or publishing, run `make typecheck` and the required `make check`, plus relevant UI/native checks, following AGENTS.md. Database-writing checks require immediate approval; an explicitly approved narrower validation scope is the alternative. The earlier PR118 docs-only checks did not execute these tasks. Any new tests must follow TESTING.md, including Given/When/Then comments and observable-boundary assertions.

## Workflow follow-up

- Archive this documentation/verification change after its tracked tasks and project review requirements are satisfied.
- Verify maintained specs and durable reference links remain intact. This change declares `skip_specs: true` and has no delta specs.
