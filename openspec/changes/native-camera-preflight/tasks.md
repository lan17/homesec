# Tasks

Implementation tracking: mark a task complete only after its specified behavior and evidence pass. Read the [design](design.md) and both delta specs before applying. Groups 1–5 prepare independently verified native primitives, adapters, and compatibility protections; group 6 switches default caller wiring only after the required compatibility and lifetime coverage passes. Earlier adapter tests inject native implementations without changing production defaults; no public migration flag is added. Intermediate commits do not establish ticket completion.

## 1. Native inspection boundary and reproducible build

- [x] 1.1 Add the exclusive inspection worker mode and matching private serde/Pydantic request, readiness, reply, and error models through the existing helper transport; verify actual executable/helper-boundary tests reject incompatible contracts, oversized/malformed replies, and conflicting modes without exposing secret-bearing input.
- [x] 1.2 Extract reusable libav input/deadline/quiet-log handling without moving policy or weakening existing camera-worker codec gates; verify existing native input tests still pass and a metadata-only fake camera works without the recorder's first-IDR requirement.
- [x] 1.3 Extend the pinned static build for software validation components, including HEVC parsing/decoding, AAC encoding, and required copy/conversion audio inputs; verify `test_bundled_native_build.py`, a real linked capability exchange, cache invalidation, checksum/static-link isolation, private temporary downloads, and parallel builds on Linux/macOS.
- [x] 1.4 Document the inspection command/IPC ownership and expanded build dependencies in the existing native/shared-media guide; verify examples match the implemented private contract and distinguish build features from runtime eligibility.

## 2. Codec-independent metadata inspection

- [x] 2.1 Implement bounded native metadata results preserving first video/audio, dimensions, available numeric/raw rational FPS, and structured failures; verify real fake-RTSP tests for H.264, HEVC/non-recorder media, no video, unavailable dimensions, invalid/absent FPS, authentication failure, and stalled discovery.
- [x] 2.2 Implement the native discovery adapter behind the existing Python probe models, ready for group 6 default wiring; extend `tests/homesec/test_rtsp_discovery.py` at the helper boundary to verify candidate deduplication/order, partial results, and low/high stream selection with real policy code and no CLI tools.
- [x] 2.3 Prepare detect-stream recovery and HLS metadata/audio planning to consume that same probe model/adapter, using native injection for tests until group 6 cutover; verify affected recovery/source and `tests/homesec/rtsp/test_live_publisher.py` cases preserve retry, codec planning, and failure behavior with absent/old host tools. Remove the duplicate recovery parser at cutover.
- [x] 2.4 Update capture/preview reference evidence and probing troubleshooting notes for the migrated metadata paths; verify every new metadata scenario links to the corresponding boundary test and remaining execution prerequisites are still explicit.

## 3. Recording-profile and timestamp validation

- [ ] 3.1 Implement a bounded MP4 video-copy trial using semantic profile fields, reusing/extending native mux primitives without broadening shared-runtime eligibility; verify actual helper trials for H.264/HEVC, AAC/AC3/ALAC/EAC3/MP2/MP3 copy, audio presence, no audio, and successfully finalized nonempty media output.
- [x] 3.2 Add pinned-library AAC conversion validation for supported fallback inputs including G.711 and Opus; verify copy → AAC → silent ordering, genuine conversion failures, and missing-component errors that do not silently select silent output, using protocol/helper fixtures plus `test_rtsp_preflight.py` policy coverage.
- [ ] 3.3 Populate bounded timestamp stability signals and preserve existing retry thresholds/weighting/profile IDs; verify baseline, improved wallclock, failed retry, and no-improvement selection outcomes against deterministic packet-timing fixtures rather than fabricated zero diagnostics or CLI stderr-string assertions.
- [x] 3.4 Exercise trial write/finalization failure, empty/header-only output, interruption, and cleanup in actual helper tests; verify no trial reaches the clip callback/repository and no writer remains when temporary output is removed.
- [x] 3.5 Document supported trial profiles, timestamp evidence, custom-flag limitations, and the distinction from compatible runtime execution; verify recording-negotiation scenarios have concrete tests and unsupported combinations remain explicit delivery gaps.

## 4. Truthful native codec and hardware capabilities

- [x] 4.1 Expose linked build identity and compiled components without opening a camera; verify positive and missing-feature helper capability fixtures and unchanged behavior when arbitrary host FFmpeg executables appear first on `PATH`.
- [x] 4.2 Add applicable Linux VAAPI, CUDA/NVENC, and QSV pinned-build/device inspection support while preserving a GPU-free software build; verify build-recipe cases and native unavailable-device/refusal cases without inferring capability from host CLI or `nvidia-smi` output.
- [x] 4.3 Add applicable macOS VideoToolbox build/device inspection support; verify macOS build and actual native capability results while preserving software operation on unsupported hosts.
- [x] 4.4 Implement the native adapter for `hardware.py` enumeration/decode trials, ready for group 6 default wiring and preserving applicable priority and explicit software/runtime fallback; verify `tests/homesec/rtsp/test_hardware.py` and helper tests require actual frames through the requested backend, distinguish requested encoder availability, and treat timeout as failure/inconclusive rather than success.
- [x] 4.5 Add the compatible motion executor's one-time software retry when native-validated hardware is rejected or fails before usable frames; verify source/frame-pipeline boundary tests for native-positive/legacy-negative input, both-paths-fail retry/exhaustion, remembered hardware failure, no leaked input processes, and no hardware/software retry cycle. Preserve existing HLS fallback.
- [x] 4.6 Record the supported target/backend evidence matrix and device prerequisites in operator/reference docs; verify no row is marked hardware-validated by software CI alone, no native result certifies a separate CLI executor, and missing support on a supported target blocks the compatibility gate.

## 5. Overlap evidence and bounded lifetime ownership

- [x] 5.1 Implement native independent role-overlap checks retaining equal-URL opens and existing role-specific media/completion evidence; verify actual fake-camera session counts, successful overlap, late failure, empty input, and explicit session-limit responses, preserving the existing single-stream fallback and preview downgrade reason.
- [x] 5.2 Preserve failed/unclassified versus proven session-limit results and recording priority; extend `test_rtsp_preflight.py` and preview/source boundary cases to verify generic auth/network/decode/timeouts do not become session-budget denial and inspection never stops an existing recorder.
- [x] 5.3 Add the inspector owner's overall deadline/cancellation/cleanup path, deriving startup's total budget from its finite plan without reinterpreting its per-operation cap as a group deadline; reserve cleanup time and reuse helper stop escalation with unchanged defaults for current media clients. Verify multi-step startup longer than one operation cap succeeds within its derived budget, while single-step stalls, exhausted setup totals, resolver/read stalls, helper crash, blocked/malformed output, cancellation, stop, and shutdown terminate/reap owned work within the applicable budget.
- [x] 5.4 Add lifecycle/privacy boundary checks for normal and failed groups; verify temporary artifacts are removed only after writers stop, session teardown is attempted within budget, cleanup failures are surfaced, and secret-bearing URLs/tokens/raw media never appear in argv, logs, persisted diagnostics, or errors.
- [x] 5.5 Update inspection lifecycle/session documentation and scenario evidence; verify the documented limits distinguish bounded local cleanup from remote camera lease expiry and defer shared-aware admission to ticket 192.

## 6. Startup and setup cutover

- [x] 6.1 Wire startup negotiation, overlap, discovery, recovery/HLS metadata, and hardware callers to the verified native operations; remove the duplicate recovery parser and their CLI execution/timeout-option discovery paths. Verify existing `test_rtsp_preflight.py` selection/profile/refusal cases, source background retry/finite-exhaustion cases, and all migrated consumers with absent/old media CLIs.
- [x] 6.2 Extend the existing setup-probe registration with a camera-free readiness callable and typed shared preview-config context for RTSP callbacks, using the same source-owned helper selection as runtime; route the existing `ffmpeg` setup check through it. Verify setup aggregation/key/response compatibility, native identity/readiness and missing/incompatible/insufficient helper results, no RTSP opens, bootstrap defaults, custom helper with WebRTC enabled/disabled, HLS, active-versus-saved config precedence, and no direct generic-service dependency on RTSP implementation. Preserve unrelated callbacks and fail an invalid configured override rather than silently using another helper.
- [x] 6.3 Propagate the setup request deadline and cancellation to the actual inspector owner instead of abandoning `to_thread` work; extend `tests/homesec/test_setup_test_connection_service.py` and setup API/helper tests to verify success/failure detail fields, 10/11-second outer budgets including cleanup, no later camera opens, and no save/activation/production clip handoff.
- [x] 6.4 Update affected capture/setup reference notes, scenario evidence, and operator troubleshooting alongside cutover; verify the docs distinguish inspection readiness from remaining recorder/WebRTC/HLS CLI execution and preserve existing custom-flag and physical-QA limitations.

## 7. Integrated acceptance and publication

- [x] 7.1 Run startup, setup Test Connection, detect-stream recovery, HLS metadata planning, recording trials, overlap, and hardware capability inspection with both media tools unavailable and with incompatible tools first on `PATH`; verify a scoped process-launch audit/traps show no `ffmpeg`/`ffprobe` launched by inspection, including the Rust helper, while fixture/oracle processes remain outside that scope.
- [x] 7.2 Run a local IPyCam integration on this Mac with isolated ports and temporary clip storage, exercising startup, motion/recording, preview coexistence, recovery, and shutdown; verify clips are finalized/nonempty and inspect process/session cleanup. Record actual codecs and limits; do not treat this as complete audio/hardware parity or use Lenovo for this test.
- [ ] 7.3 Complete the design's codec/audio/platform compatibility matrix with protocol/library tests and actual device checks for applicable hardware targets; verify each supported combination has evidence or an explicit unresolved blocker, with no silent support removal and no claim that untested rows passed.
- [x] 7.4 Run `openspec validate native-camera-preflight --strict`, required `make typecheck` before commit, and `make check` before push/PR updates under `AGENTS.md`; verify results, use only an explicitly approved disposable database for writing tests, and disclose any user-approved validation exception or environment failure accurately.
- [x] 7.5 Review the integrated implementation against every changed scenario and the no-CLI inspection inventory; verify ticket 194's evidence and PR links are current, ticket 192 can consume the semantic operations, and tickets 195/196 still track remaining execution and packaging work without prematurely marking the epic complete.

### Acceptance status on 2026-10-18

Default caller wiring is implemented; the PR remains a draft while the following
exact task criteria are incomplete. The completed Linux build tasks cover the
GPU-free arm64 build and unavailable-device evidence, not physical GPU or x86_64
qualification. See the [scenario register](../../../docs/behavior-evidence.md#native-camera-inspection)
and [compatibility matrix](../../../docs/shared-rust-media.md#inspection-compatibility-evidence).

- **3.1:** The full audio allowlist has linked-library fixtures. Actual RTSP helper AAC/AC3/MP2/MP3/EAC3 copy passed; ALAC still lacks an actual RTSP/RPC fixture.
- **3.3:** Actual packet-clock-reset trials establish baseline/improved wallclock evidence. Failed/no-improvement outcomes have typed-peer policy coverage; their real packet-fixture cases remain pending.
- **7.3:** Physical Linux VAAPI/CUDA/NVENC/QSV, x86_64 and remaining codec/device qualification are pending; no untested row is passed.

The final stable `make check` passed: 186 Rust tests, 1,705 Python tests and 325 UI
tests, plus typing, lint/format, Clippy, generated API consistency, locked
dependencies and production UI build. Strict OpenSpec validation passed. Database
writes were confined to the explicitly approved disposable Postgres 16 tmpfs
container on loopback port 15439, which was stopped and removed after validation.
The final Linux arm64 helper suite passed 16 tests and its hardware suite passed
four tests; static linkage and absent/incompatible-CLI capability checks passed.

The existing [draft PR](https://github.com/lan17/homesec/pull/124) and
[ticket 194](https://app.notion.com/p/3fdd8336c59f813eac7effebd396fb79)
now describe this implementation, passing validation and the three remaining
acceptance tasks. The original planning commit is retained. Tickets 192, 195
and 196 were rechecked and retain admission, remaining media execution and final
packaging scope respectively; the epic is not marked complete. No merge or
production rollout was performed.

## Workflow follow-up

- After implementation and evidence review, explicitly sync/archive this change through the OpenSpec workflow, retaining reference limitations and repairing any moved links. Merging the planning PR alone does not promote these proposed contracts or authorize implementation.
- Keep runtime execution/compatibility replacement in ticket 195 and final no-CLI application packaging/acceptance in ticket 196. Production rollout and any database-writing validation remain subject to the applicable approval rules.
