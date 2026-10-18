# Tasks

The reviewed bootstrap is complete and archived. All eight capabilities are maintained in `openspec/specs/` and connected through [the behavior reference](../../../../docs/reference.md). Completed review tasks reflect source and representative-test inspection, not execution of application/device tests. Eleven unfinished deeper-verification and guide tasks were transferred to [baseline-reference-follow-up](../../baseline-reference-follow-up/tasks.md); they were not marked complete. Corrective runtime changes remain separate approved work.

## 1. Confirm the baseline snapshot and review scope

- [x] 1.1 Refresh the target branch and compare relevant source changes since the researched `origin/main` snapshot; verify the draft still describes the target implementation and record any necessary evidence updates in the design.
- [x] 1.2 Review the eight-capability inventory and the design's exclusions with Leonidas; verify the reviewed scope covers current major public surfaces and distinguishes pending features from existing behavior.

## 2. Review setup and configuration

- [x] 2.1 Review `setup-configuration` scenarios against the setup, ONVIF, camera mutation, redaction, and runtime-manager evidence; verify first-run validation, saved-versus-active state, reload acceptance/completion, save-before-reload rejection and create-retry outcomes, and candidate rollback each have a supported contract. Assess a focused persisted-mutation/retry test or retain the explicit source-only limit.

## 3. Review capture and recording

- [x] 3.1 Review `capture-recording` against local-folder, FTP, RTSP, and native publication tests; verify ready-file conditions, repeat suppression, platform-dependent filename case matching, finite-exhaustion loss of camera monitoring until reload/restart, rotation, and failed-finalization scenarios match source behavior.

## 4. Review storage, retention, and deletion

- [x] 4.1 Review `storage-retention` and the eligible-byte versus total-usage disagreement with Leonidas; verify the reviewed contract preserves eligibility, oldest-first pruning, protected files, and truthful over-limit results without adopting an unconfirmed hard disk-budget policy. Review how deleted originals, filter-error clips, and discovered stateless MP4s can remain ineligible indefinitely; keep recovery/deletion policy changes separate and retain explicit limits where coverage is missing.

## 5. Review analysis and alerts

- [x] 5.1 Review `analysis-alerts` against pipeline, analyzer, filter, and policy evidence; verify stage overlap, run modes, retry limits, failed-analysis fallback, notification eligibility, and completed-status meaning remain distinct.

## 6. Review events and playback

- [x] 6.1 Review `events-playback` against API/store and UI tests; verify query filters/cursors, URL context, detail navigation, scoped authorization, default deleted-record exclusion and explicit status filtering, unavailable media, conditional deleted-media retrieval errors, and bounded refresh all describe current behavior. Bound identity guarantees to successful lookup and retain the query-failure-to-404 gap.

## 7. Review preview and talk

- [x] 7.1 Review `preview-talk` against API, source, native, and UI evidence; verify recording priority, HLS compatibility, token/lease expiry, peer cleanup, exclusive talk reservation, separate reservation/active duration timers, the WebSocket first-message deadline, PCM framing, release cancellation, microphone denial, and backpressure.

## 8. Review integration contracts

- [x] 8.1 Review `integration-contracts` against authentication, error-envelope, OpenAPI, browser key-storage/redirect, alert payload, MQTT, and SendGrid evidence; verify exact generated shapes remain authoritative and no delivery deduplication, timestamp timezone, or general HTML-escaping guarantee is invented.
- [x] 8.3 Verify the existing OpenAPI export and UI generated-contract check with the repository's existing commands; record the result and update integration reference links rather than introduce another hand-maintained schema inventory.

## 9. Review operations and maintenance

- [x] 9.1 Review `operations-maintenance` against CLI, health, diagnostics, stats, shutdown, and backup evidence; verify summary versus component-health meaning, UTC-day clip-level counts, count-failure limits, and bounded worker/backup shutdown behavior. Exclude the unreachable mocked CLI restart branch from public contracts; dead-code removal is a separate change.

## 10. Promote reviewed specs and connect reference documentation

- [x] 10.1 Confirm every design disagreement is resolved for this baseline or explicitly bounded/deferred; verify no unsupported requirement is promoted and all eight delta specs pass `openspec validate baseline-existing-behavior --strict --no-interactive`.
- [x] 10.2 Sync the reviewed deltas to maintained specs through the existing OpenSpec sync workflow; verify `openspec/specs/` contains exactly the eight planned capabilities with the reviewed requirements and no pending-feature behavior. The CLI does not copy the deltas' `Current limitations` sections automatically; preserve retained limitations in the planned `docs/reference.md` and add links from affected maintained specs before completing promotion.
- [x] 10.3 Add a small `docs/reference.md` capability index and README link connecting maintained specs, the generated API contract, and existing operator guides; verify all local links resolve and each capability has a clear entry point. Include retained current limitations and their reviewed evidence limits, and verify affected specs link to this durable reference rather than the active change directory.
- [x] 10.4 Extend the existing README/OpenSpec configuration guidance so public behavior changes update spec deltas, meaningful behavioral coverage, and affected references together; verify the instructions are injected through the current workflow and do not require deltas for behavior-preserving refactors. Do not edit generated skill files to encode project policy.

## 11. Complete integration validation and prepare publication

- [x] 11.1 Validate the complete change and maintained spec inventory with strict OpenSpec validation; verify capability names, requirement/scenario structure, evidence references, and documentation links agree and record executed checks separately from inspected test evidence.
- [x] 11.2 Run `make typecheck` and the required `make check` before committing/publishing, plus UI/native checks appropriate to any added tests. Obtain immediate approval before database-writing validation, or an explicit narrower validation scope for this change. Verify the final diff contains only intended specs, documentation, and justified tests; prepare a semantic PR title and describe remaining limitations.

## Validation scope

Leonidas approved docs-only validation for PR118. Promotion checks preserve the reviewed Purpose and all 77 requirements/222 scenarios, validate the active change and eight maintained specs strictly, resolve evidence/reference links, verify workflow guidance injection, and run `make typecheck lint lock-check`. The existing OpenAPI exporter and UI generated-contract consistency check also passed with `UV_LOCKED=1 pnpm --dir ui api:check`. Existing tests remain inspected evidence; full `make check` and application/device tests were not executed locally. No database writes were performed.

## Transferred follow-up work

The following original task IDs remain unfinished in [baseline-reference-follow-up](../../baseline-reference-follow-up/tasks.md): `2.2`, `3.2`, `4.2`, `4.3`, `5.2`, `5.3`, `6.2`, `7.2`, `8.2`, `9.2`, `9.3`. Their checkbox descriptions were transferred verbatim. The bootstrap retains its seventeen completed tasks; this transfer does not claim the follow-up work was performed.

## Archive status

Leonidas approved archiving the promoted baseline and transferring its remaining work. The eight deltas were already synced; archive skips spec writes to preserve maintained files byte-for-byte. Archived delta limitations and design evidence are historical snapshots. Use `docs/reference.md` for current limitations and the follow-up change for remaining work.
