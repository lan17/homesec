# Tasks

The eleven original unfinished tasks were transferred from the [archived bootstrap](../archive/2026-10-10-baseline-existing-behavior/tasks.md). Original task IDs and wording are retained for traceability. Tasks 2.3, 2.4, 3.3, 4.4, 5.4, 5.5, 6.3, 6.4, 7.3, and 9.4 were added during subsequent PR118 review; they are not part of the archived transfer. Read [design.md](design.md), the [maintained specs](../../specs/), [scenario evidence](../../../docs/behavior-evidence.md), and [current reference limits](../../../docs/reference.md) before implementation. Additional direct-review tasks below are also new verification work, not part of the archived transfer. The archive records historical evidence; update current reference notes as checks are completed.

## 2. Review setup and configuration

- [ ] 2.2 Reconcile setup and camera-editing guidance in README/DESIGN with bootstrap-only finalization and the saved-file rollback limit; verify the updated guidance does not describe a general configuration editor or transactional file/runtime rollback. Record remaining coverage limits beside the evidence.

- [ ] 2.3 Characterize configuration/camera response redaction through the real redaction path: recognized sensitive strings and URL userinfo are masked, `_env` names remain, and query/path/fragment credentials in URL-named fields can remain. Include URLs with and without userinfo plus unrecognized-key/non-string controls. Update the reference evidence after verification; do not silently implement broader redaction in this documentation follow-up.
- [ ] 2.4 Characterize camera patches through the route and real configuration manager with a temporary config file. Compare omitted backend, exact saved normalized identifier, case-only identifier changes, and a genuine backend switch. Assert persisted settings, including loss of a custom `watch_dir` for a valid case-only `LOCAL_FOLDER` patch and no save when discarded required fields make validation fail. Keep a normalization-order fix separately reviewed.

- [ ] 2.5 Verify source-patch object/null/list semantics and default reapplication through persisted configuration; camera removal with policy references; disabled source/notifier and never-mode analyzer validation; and registered versus generic setup probes. Use real manager/loader boundaries with fake external dependencies and update scenario-specific evidence. Do not claim every successful probe exercised actual processing.

- [ ] 2.6 Add repository-level characterization of validation-only versus applied setup finalization through the actual service and mounted route with fake application boundaries and a temporary config path. Cover unknown backend acceptance by validation-only, full apply rejection before persistence/activation, camera/DSN prerequisite rejection, and a valid control. Record the exact boundary and result; full-validation-on-dry-run behavior changes require separate review.

## 3. Review capture and recording

- [ ] 3.2 Assess manifest-restart and unfinished-file coverage and add a focused source-boundary test only where the existing suite leaves a critical gap; verify the selected check and update README/DESIGN recording guidance to avoid durable replay or universal playable-file promises and obsolete startup-flag examples.

- [ ] 3.3 Characterize two cameras ingesting the same filename stem and a repeated FTP upload after deletion through the actual source/repository boundaries. Use independent persisted snapshots, not shared mutable state references; assert one global identity, replacement camera/path/results/status, preserved database `created_at`, shared lifecycle identity, and reused notification `dedupe_key` when notification is selected. Include distinct-ID and local repeat-suppression controls. Verify retention/playback consequences using the surviving snapshot and update current reference evidence. Namespace or collision-rejection changes require a separate reviewed runtime change.

- [ ] 3.4 Characterize local-folder and FTP mtime/duration estimates with a controlled clock/filesystem, and FTP unlink-failure handling for disallowed and incomplete files. Distinguish source estimates, database creation time, lifecycle time, and notification time. Keep original files and external services outside the disposable test boundary.

## 4. Review storage, retention, and deletion

- [ ] 4.2 Review API deletion and cleanup separately using their existing route and cleanup tests; verify mark-before-storage failure, original-local-file handling, non-atomic media deletion, deleted-record addressability/conditional media failures, and best-effort cleanup persistence are correctly documented. Confirm any requested change to Delete semantics separately; no new all-copies deletion policy is established by this baseline. Add focused boundary coverage for default dry-run preservation of original media and workflow writes if existing tests do not establish it.
- [ ] 4.3 Reconcile storage/retention/cleanup reference guidance with the reviewed specs; verify examples distinguish retention, API deletion, and cleanup and promise only the destination syntax checks actually implemented, including the filesystem-confinement limit.

- [ ] 4.4 Characterize `build_clip_path` camera/filename sanitization and collisions between distinct valid filenames, then exercise a successful storage upload and retry with a fake plugin returning a storage URI but `view_url=None`. Assert the pipeline preserves optional view URLs and the backend destination results; do not infer uniqueness or collision rejection. Use temporary paths and fake storage, with no provider writes, and update the reference evidence.

- [ ] 4.5 Verify cleanup's YOLO-shaped builder and zero/partial-decoding outcomes through real detector/cleanup boundaries with fake decoding and temporary media. Assert applied deletion versus default dry-run preservation, separately from detector exceptions. Include non-YOLO configuration rejection and successful readable/no-detection controls. The source-derived destructive consequence is not an executed end-to-end test.
- [ ] 4.6 Use a deterministic selection/delete barrier and independent state to replace a selected retention pathname before unlink; inspect deletion and captured-size accounting. Include deletion-error, exhausted-candidate, unchanged-file, and physically separate storage-copy controls. Do not infer atomic file identity or state revalidation from the earlier isolated deletion-step check.

## 5. Review analysis and alerts

- [ ] 5.2 Review detector setting/class discrepancies and the intentional-`never` fallback reason with Leonidas; verify the baseline either remains explicitly bounded or reflects a confirmed current contract. Correct inaccurate README/DESIGN examples; put any requested behavior fix in a separate change.
- [ ] 5.3 Assess critical gaps for disabled-policy suppression, threshold equality, and event-append failure containment; verify each is supported by an existing test, a justified new boundary test, or an explicit retained evidence limit in the design. Update analysis/alert reference guidance in the same group.

- [ ] 5.4 Add deterministic barrier-controlled characterization of competing snapshot writes through the real repository and database boundary. Exercise both upload/filter write orders, a stale upload write after exhausted filter failure, and a stale processing write after API deletion, with every database operation succeeding. Assert persisted metadata/status plus resulting list/media/retention behavior separately from in-process notification outcomes. Use independent serialized snapshots or an approved isolated Postgres database; the existing shared-object state fake is not evidence of isolation. Include sequential controls and verify that collecting the upload result does not repair lost stored fields. Update analysis/storage/events reference evidence without claiming terminal deletion or atomic updates; runtime fixes remain separately scoped.

- [ ] 5.5 Verify default-policy decision precedence with the real policy: a below-threshold matching activity produces the activity reason only when motion notification is disabled; enabled motion notification takes precedence even when activity matches or a failed analysis leaves a trigger detection. Include effective per-camera overrides and a disabled-policy control through its loader. Reuse existing tests where they establish these combinations and add only missing boundary checks. Record repository-test evidence separately from the isolated source-excerpt harness; no runtime policy change is part of this task.

- [ ] 5.6 Verify destination normalization for ordinary, dot, and parent camera names through the actual upload stage and pipeline join. Assert no retry/storage attempt for preparation failure and no later alert/completion through that path. Separately characterize a pending backend attempt and notifier fan-out to distinguish attempt limits from elapsed-time deadlines; fake time/external I/O, no real provider calls.
- [ ] 5.7 Verify the real structured-analysis parser accepts schema-valid but inconsistent timestamp/frame-count fields while rejecting actual schema violations and zero extracted frames. Do not promote schema validation into a visual-accuracy or temporal-consistency guarantee.

## 6. Review events and playback

- [ ] 6.2 Assess mounted playback-token refresh coverage and add a timer-boundary UI test if needed; verify refresh behavior with the focused UI check and update playback reference guidance without importing activation-only loading or requiring current eager acquisition timing.

- [ ] 6.3 Add focused characterization coverage for event-list failure masking: given successful authentication and database availability checks, a later connection/query failure currently returns HTTP 200 with an empty page. Exercise the real Postgres store, repository retry handling, and route response with failures at the database boundary, rather than mocking the repository to return an empty page. Cover store-contained failures bypassing retries, exceptions reaching the repository exhausting retries, and invalid-row hydration producing incomplete pages with pagination based on fetched rows. Include a genuine empty-result control and update the current reference evidence note after verification. This records the current limitation, not a policy to preserve; any runtime correction requires a separate reviewed change.

- [ ] 6.4 Characterize the mounted media-token and media routes with normal/bootstrap mode, authentication on/off, database guards available/unavailable, successful/failed clip lookup, and present/missing storage URI. Check setup/database 503 responses, lookup failure masking, missing-media 409, and authorized direct/tokenized URLs only after prerequisites. Use fake database/storage boundaries without live writes and update reference evidence.

- [ ] 6.5 With the repository-locked environment and actual mounted media routes, characterize ordinary and non-Latin clip identities, successful and failed retrieval, range requests, cancellation after partial staging, and response-construction failure. Assert storage-call order and staging-directory lifecycle separately from status codes. Include all required admission gates; do not treat isolated ASGI checks as full-route conformance.

## 7. Review preview and talk

- [ ] 7.2 Assess focused native viewer-cap coverage and add a boundary test if the existing suite does not establish it; verify the selected native check and reconcile `docs/webrtc-preview.md`, `docs/push-to-talk.md`, and relevant deployment guidance. Preserve explicit hardware/codec/browser QA limits.

- [ ] 7.3 Characterize cancellation of an accepted, authorized talk WebSocket while it awaits the first start message, using the real route and source-manager boundary with fake transport. Inspect the reservation after cancellation and its release by the existing duration timer or an explicit stop. Keep normal disconnect, start timeout, cancellation during camera-session opening, and active-stream cleanup as controls. Distinguish a requested best-effort stop from confirmed release and do not infer continued microphone transmission before streaming begins. Update current reference evidence; fixing the handler cleanup scope is a separate runtime change.

## 8. Review integration contracts

- [ ] 8.2 Assess production per-entry mixed-success notifier fan-out, installed-entry-point discovery, and validate-before-factory/no-construction coverage; add only justified plugin-boundary tests and verify the selected checks. Reconcile `PLUGIN_DEVELOPMENT.md` and obsolete AGENTS examples with the existing unified registry and correct decorator arguments.

- [ ] 8.3 Characterize API gate order and media/preview/talk authorization alternatives through mounted HTTP/WebSocket routes. Include valid-key plus invalid-token, disabled-auth tokens, token-only wrong-scope/expiry, non-ASCII bearer comparison, JSON-unsafe model-validator errors, and ordinary 401/422 controls. Inspect talk acceptance followed by authorization close separately from HTTP envelopes. Update scenario-specific evidence; malformed-input runtime fixes remain separately reviewed.

- [ ] 8.4 Compare freshly generated OpenAPI error schemas with actual mounted-route JSON-safe validation errors, including the default array-valued `detail` versus the canonical textual envelope. Add a committed regression characterization with fake external boundaries and update scenario evidence; changing generated error schemas or handler behavior requires a separately reviewed implementation change.

## 9. Review operations and maintenance

- [ ] 9.2 Assess missing-executable startup degradation, private dump permissions, and critical backup recovery coverage; add focused process/filesystem tests where justified and verify them using fake dump boundaries. Reconcile `docs/postgres-backups.md` with durable local-directory/manifest mounting, upload outcomes, retention limits, and operator restore responsibilities without claiming a real restore was tested.
- [ ] 9.3 Update operator reference guidance for CLI validation/cleanup and health/statistics interpretation; verify documented options against current CLI/config models and existing tests. Keep applied cleanup semantics owned by `storage-retention`.

- [ ] 9.4 Characterize cancellation of startup pending-upload retry through the real backup manager, with fake dump/storage boundaries and a temporary manifest directory. Cancel after a remote object may have been created, compare in-memory and saved markers, reconstruct the manager from disk, and exercise another failed retry plus pending-upload pruning. Assert loss of an unsaved unknown-outcome marker and pruning eligibility; include a persisted-marker control that retains the local artifact and the new-backup cancellation path as a separate control. No real database or provider writes are needed. Update reference evidence without claiming an isolated source-excerpt harness proves full startup/shutdown conformance; marker-persistence fixes require a separate reviewed runtime change.

- [ ] 9.5 Exercise completed new dump plus failed upload followed by older-artifact housekeeping failure through the real backup manager and fake process/storage boundaries. Assert final artifact, new-record classification, saved manifest, and later retry eligibility separately; include a successful-housekeeping control and no real database/provider writes.

## Validation and publication

For each implemented group, run appropriate checks and update its guide/evidence notes together. Before committing or publishing, run `make typecheck` and the required `make check`, plus relevant UI/native checks, following AGENTS.md. Database-writing checks require immediate approval; an explicitly approved narrower validation scope is the alternative. The earlier PR118 docs-only checks did not execute these tasks. Any new tests must follow TESTING.md, including Given/When/Then comments and observable-boundary assertions.

## Evidence completion

- [ ] 10.1 Map each maintained scenario to a specific existing or newly verified boundary test, or retain its explicit source-only/isolated-check limit in `docs/behavior-evidence.md`. Record tested commit, test name, execution command and boundary; distinguish device QA. A green suite, a structural check, or a representative requirement-level link does not complete this task.

## Workflow follow-up

- Archive this documentation/verification change after its tracked tasks and project review requirements are satisfied.
- Verify maintained specs and durable reference links remain intact. At archive time repair depth-sensitive links in this follow-up and the historical bootstrap transfer references; do not overwrite historical spec text. This change declares `skip_specs: true` and has no delta specs.
