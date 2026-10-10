# Tasks

This checklist is for the next, explicitly requested apply phase. The planning artifacts are already written; baseline review, promotion, and reference updates are still pending. Use [design.md](design.md) for evidence and bounded gaps. Keep corrective runtime changes in separate approved changes. Any added tests must follow `TESTING.md`, use Given/When/Then comments, and verify behavior at existing boundaries.

## 1. Confirm the baseline snapshot and review scope

- [ ] 1.1 Refresh the target branch and compare relevant source changes since the researched `origin/main` snapshot; verify the draft still describes the target implementation and record any necessary evidence updates in the design.
- [ ] 1.2 Review the eight-capability inventory and the design's exclusions with Leonidas; verify the reviewed scope covers current major public surfaces and distinguishes pending features from existing behavior.

## 2. Review setup and configuration

- [ ] 2.1 Review `setup-configuration` scenarios against the setup, ONVIF, camera mutation, redaction, and runtime-manager evidence; verify first-run validation, saved-versus-active state, reload acceptance/completion, save-before-reload rejection and create-retry outcomes, and candidate rollback each have a supported contract. Assess a focused persisted-mutation/retry test or retain the explicit source-only limit.
- [ ] 2.2 Reconcile setup and camera-editing guidance in README/DESIGN with bootstrap-only finalization and the saved-file rollback limit; verify the updated guidance does not describe a general configuration editor or transactional file/runtime rollback. Record remaining coverage limits beside the evidence.

## 3. Review capture and recording

- [ ] 3.1 Review `capture-recording` against local-folder, FTP, RTSP, and native publication tests; verify ready-file conditions, repeat suppression, platform-dependent filename case matching, finite-exhaustion loss of camera monitoring until reload/restart, rotation, and failed-finalization scenarios match source behavior.
- [ ] 3.2 Assess manifest-restart and unfinished-file coverage and add a focused source-boundary test only where the existing suite leaves a critical gap; verify the selected check and update README/DESIGN recording guidance to avoid durable replay or universal playable-file promises and obsolete startup-flag examples.

## 4. Review storage, retention, and deletion

- [ ] 4.1 Review `storage-retention` and the eligible-byte versus total-usage disagreement with Leonidas; verify the reviewed contract preserves eligibility, oldest-first pruning, protected files, and truthful over-limit results without adopting an unconfirmed hard disk-budget policy. Review how deleted originals, filter-error clips, and discovered stateless MP4s can remain ineligible indefinitely; keep recovery/deletion policy changes separate and retain explicit limits where coverage is missing.
- [ ] 4.2 Review API deletion and cleanup separately using their existing route and cleanup tests; verify mark-before-storage failure, original-local-file handling, non-atomic media deletion, deleted-record addressability/conditional media failures, and best-effort cleanup persistence are correctly documented. Confirm any requested change to Delete semantics separately; no new all-copies deletion policy is established by this baseline. Add focused boundary coverage for default dry-run preservation of original media and workflow writes if existing tests do not establish it.
- [ ] 4.3 Reconcile storage/retention/cleanup reference guidance with the reviewed specs; verify examples distinguish retention, API deletion, and cleanup and promise only the destination syntax checks actually implemented, including the filesystem-confinement limit.

## 5. Review analysis and alerts

- [ ] 5.1 Review `analysis-alerts` against pipeline, analyzer, filter, and policy evidence; verify stage overlap, run modes, retry limits, failed-analysis fallback, notification eligibility, and completed-status meaning remain distinct.
- [ ] 5.2 Review detector setting/class discrepancies and the intentional-`never` fallback reason with Leonidas; verify the baseline either remains explicitly bounded or reflects a confirmed current contract. Correct inaccurate README/DESIGN examples; put any requested behavior fix in a separate change.
- [ ] 5.3 Assess critical gaps for disabled-policy suppression, threshold equality, and event-append failure containment; verify each is supported by an existing test, a justified new boundary test, or an explicit retained evidence limit in the design. Update analysis/alert reference guidance in the same group.

## 6. Review events and playback

- [ ] 6.1 Review `events-playback` against API/store and UI tests; verify query filters/cursors, URL context, detail navigation, scoped authorization, default deleted-record exclusion and explicit status filtering, unavailable media, conditional deleted-media retrieval errors, and bounded refresh all describe current behavior. Bound identity guarantees to successful lookup and retain the query-failure-to-404 gap.
- [ ] 6.2 Assess mounted playback-token refresh coverage and add a timer-boundary UI test if needed; verify refresh behavior with the focused UI check and update playback reference guidance without importing activation-only loading or requiring current eager acquisition timing.

## 7. Review preview and talk

- [ ] 7.1 Review `preview-talk` against API, source, native, and UI evidence; verify recording priority, HLS compatibility, token/lease expiry, peer cleanup, exclusive talk reservation, separate reservation/active duration timers, the WebSocket first-message deadline, PCM framing, release cancellation, microphone denial, and backpressure.
- [ ] 7.2 Assess focused native viewer-cap coverage and add a boundary test if the existing suite does not establish it; verify the selected native check and reconcile `docs/webrtc-preview.md`, `docs/push-to-talk.md`, and relevant deployment guidance. Preserve explicit hardware/codec/browser QA limits.

## 8. Review integration contracts

- [ ] 8.1 Review `integration-contracts` against authentication, error-envelope, OpenAPI, browser key-storage/redirect, alert payload, MQTT, and SendGrid evidence; verify exact generated shapes remain authoritative and no delivery deduplication, timestamp timezone, or general HTML-escaping guarantee is invented.
- [ ] 8.2 Assess production per-entry mixed-success notifier fan-out, installed-entry-point discovery, and validate-before-factory/no-construction coverage; add only justified plugin-boundary tests and verify the selected checks. Reconcile `PLUGIN_DEVELOPMENT.md` and obsolete AGENTS examples with the existing unified registry and correct decorator arguments.
- [ ] 8.3 Verify the existing OpenAPI export and UI generated-contract check with the repository's existing commands; record the result and update integration reference links rather than introduce another hand-maintained schema inventory.

## 9. Review operations and maintenance

- [ ] 9.1 Review `operations-maintenance` against CLI, health, diagnostics, stats, shutdown, and backup evidence; verify summary versus component-health meaning, UTC-day clip-level counts, count-failure limits, and bounded worker/backup shutdown behavior. Exclude the unreachable mocked CLI restart branch from public contracts; dead-code removal is a separate change.
- [ ] 9.2 Assess missing-executable startup degradation, private dump permissions, and critical backup recovery coverage; add focused process/filesystem tests where justified and verify them using fake dump boundaries. Reconcile `docs/postgres-backups.md` with durable local-directory/manifest mounting, upload outcomes, retention limits, and operator restore responsibilities without claiming a real restore was tested.
- [ ] 9.3 Update operator reference guidance for CLI validation/cleanup and health/statistics interpretation; verify documented options against current CLI/config models and existing tests. Keep applied cleanup semantics owned by `storage-retention`.

## 10. Promote reviewed specs and connect reference documentation

- [ ] 10.1 Confirm every design disagreement is resolved for this baseline or explicitly bounded/deferred; verify no unsupported requirement is promoted and all eight delta specs pass `openspec validate baseline-existing-behavior --strict --no-interactive`.
- [ ] 10.2 Sync the reviewed deltas to maintained specs through the existing OpenSpec sync workflow; verify `openspec/specs/` contains exactly the eight planned capabilities with the reviewed requirements and no pending-feature behavior. The CLI does not copy the deltas' `Current limitations` sections automatically; preserve retained limitations in the planned `docs/reference.md` and add links from affected maintained specs before completing promotion.
- [ ] 10.3 Add a small `docs/reference.md` capability index and README link connecting maintained specs, the generated API contract, and existing operator guides; verify all local links resolve and each capability has a clear entry point. Include retained current limitations and their reviewed evidence limits, and verify affected specs link to this durable reference rather than the active change directory.
- [ ] 10.4 Extend the existing README/OpenSpec configuration guidance so public behavior changes update spec deltas, meaningful behavioral coverage, and affected references together; verify the instructions are injected through the current workflow and do not require deltas for behavior-preserving refactors. Do not edit generated skill files to encode project policy.

## 11. Complete integration validation and prepare publication

- [ ] 11.1 Validate the complete change and maintained spec inventory with strict OpenSpec validation; verify capability names, requirement/scenario structure, evidence references, and documentation links agree and record executed checks separately from inspected test evidence.
- [ ] 11.2 Run `make typecheck` and the required `make check` before committing/publishing, plus UI/native checks appropriate to any added tests. Obtain immediate approval before database-writing validation, or an explicit narrower validation scope for this change. Verify the final diff contains only intended specs, documentation, and justified tests; prepare a semantic PR title and describe remaining limitations.

## Workflow follow-up

- Archive this change through the existing OpenSpec archive workflow after tracked tasks and project review requirements are satisfied.
- Verify the archived result preserves all eight maintained specs, retained current limitations, and reference links; none may depend on the old active-change path.
