# Tasks

This is the implementation plan for ticket 190. Every checkbox is intentionally unfinished in the planning PR. Refer to [design.md](design.md) for ownership and acceptance gates. Tasks within each group are ordered; groups 4 and 5 can proceed independently after the common reader and lifecycle work they consume is available.

## 1. Freeze compatibility and performance baselines

- [ ] 1.1 Add redistributable local-clip fixtures and a Python-baseline manifest covering H.264/B-frames, MPEG-4, supported HEVC/containers, VFR, count estimates/unknown duration, nonzero start/missing timestamps, multiple streams, truncation and rotation; verify recorded sample content, labels, oriented dimensions and read-failure behavior through the existing OpenCV consumers.
- [ ] 1.2 Extend the current YOLO/model-boundary and VLM/HTTP-boundary tests to lock stride, distinct-class `min_hits`, threshold equality, early exit, timestamp formatting, no-upscale dimension rounding and preprocessing edge cases; verify existing zero/partial results and no-frame/no-provider behavior without requiring network/model downloads.
- [ ] 1.3 Add a repeatable test/benchmark-only baseline runner for cold/warm preparation, full worker round trips and fixed-model operation; deliver baseline batches with wall median/p95, CPU, peak RSS, thread counts, image errors/output bytes and reviewed fixture-specific image/confidence tolerances established from repeatability/pixel sensitivity before candidate comparison. Document invocation and prohibit production use of the old decoder through this helper.

## 2. Build and package the private native component

- [ ] 2.1 Reconcile current main/PR 124's shared build recipe, then add a preparation profile using the same verified FFmpeg archive/checksum, safe extraction, private temporary cache/locks and parallel jobs; verify profile-specific cache invalidation/required artifacts and that the camera helper's existing capabilities still build.
- [ ] 2.2 Add the private preparation crate and exact-pinned binding/encoder dependencies with a locked dependency graph; verify a native decode/JPEG smoke fixture and compiled capability inventory cover the baseline corpus without linking WebRTC, camera protocols, unused external encoders or an independent system FFmpeg.
- [ ] 2.3 Replace the Python-only package build with Maturin's mixed layout and a minimal pre-Cargo PEP 517 bootstrap delegate; verify `uv sync --locked`, wheel, editable and unpacked-sdist builds include native sources/lockfiles, existing Python package data/entry points and extension typing. Intentionally update and check `uv.lock`.
- [ ] 2.4 Update developer prerequisites/Makefile and build documentation; verify documented setup commands work from a clean supported checkout and missing compilers/tools cause actionable errors without falling back to system libraries.
- [ ] 2.5 Update Docker and release/CI build stages to produce installed native artifacts for the declared CPython 3.10–3.14 and Linux/macOS targets; verify installed import/decode smoke, truthful platform tags/manylinux audit, import order with `cv2`/`torch`, dynamic dependency/configuration identity and component-specific source/license notices. Do not publish a wheel or image as part of local verification.

## 3. Implement the typed local-file preparation boundary

- [ ] 3.1 Implement the private reader/metadata adapter and owned typed outputs with checked dimensions/strides, explicit close and GIL release around native work; verify regular-file-only input, disabled network/external references, buffer lifetime after reader close and failed-open cleanup with real local fixtures.
- [ ] 3.2 Implement compatible count/FPS estimation, stream selection, indexed seeking, timestamps and autorotation; verify exact sample identities/labels/oriented dimensions against group 1 for CFR/VFR/B-frame/metadata-edge fixtures, without counting/decoding the whole clip eagerly.
- [ ] 3.3 Implement one-sample BGR output and native resizing/JPEG output preserving each consumer's settings, dimension rounding and accepted quality behavior; verify the frozen structural/image gates, byte ownership, no-upscale behavior and bounded per-sample memory. Keep image-implementation choices private to this boundary.
- [ ] 3.4 Document the adapter's ownership, format coverage and known compatibility limits alongside native media documentation; verify no public HTTP/config shape or unrelated camera behavior changes and every unsupported parity gap blocks cutover rather than choosing another decoder.

## 4. Preserve Python YOLO through the native reader

- [ ] 4.1 Replace YOLO's `cv2.VideoCapture` use with the typed native reader inside its existing Python model worker; verify original-size writable BGR inputs, oriented box-height thresholds, model-cache/device behavior, result order/confidence/model/sample fields and per-camera overrides at the model boundary.
- [ ] 4.2 Preserve incremental stride sampling, distinct-class early exit and zero/partial read results; verify exact requested sample content/count and that corrupt/unreadable samples after early exit are never requested. Verify cleanup/recheck consumers of the filter follow the same reader path.
- [ ] 4.3 Update detector reference/evidence notes and run fixed-weight/device parity measurements with existing Python inference; verify classes, trigger decisions, sample count and early exit match, confidence stays within the frozen reviewed tolerance, and no inference backend or deletion-safety claim has changed.

## 5. Isolate eligible VLM preparation behind the analyzer plugin

- [ ] 5.1 Add the narrow optional worker-capacity interface and analyzer-loader/runtime wiring described in the design; verify existing registry discovery, strict config validation, third-party analyzers without the capability, disabled analysis and unchanged `analyze()` inputs all work. Document the opt-in lifecycle contract in `PLUGIN_DEVELOPMENT.md`.
- [ ] 5.2 Add the lazy spawned preparation pool and module-level worker, replacing `asyncio.to_thread`/OpenCV/Pillow preparation with native sampled JPEG output; verify all VLM sampling/label/size/quality baselines at the captured provider HTTP boundary, with binary process output and Python-only request construction/response validation.
- [ ] 5.3 Verify no preparation or provider work occurs for filter failure, `never` mode or a missing trigger; reuse pipeline behavioral tests to confirm upload overlap, existing retries, failure fallback and plugin result contracts remain unchanged. Verify no-frame failures precede provider requests.
- [ ] 5.4 Update VLM reference/evidence notes and native-install troubleshooting to describe actual coverage and parity; verify source/installed-package tests cannot silently use a Python-wheel decoder or CLI preparation path even when those tools are available.

## 6. Complete cooperative worker lifecycle and containment

- [ ] 6.1 Add bounded pre-submission admission, retained concurrent futures and per-job pipe cancellation for the two plugin-owned pools; verify real spawned-worker tests cover waiting/queued/running cancellation, one-job isolation, post-inference cancellation, no accepted cancelled output and capacity held until actual settlement.
- [ ] 6.2 Add serialized broken-pool retirement/replacement without private executor internals or hidden retries; verify abrupt worker death fails affected attempts, no successful empty result is fabricated, old generations retire before replacement, and a later Python-controlled attempt succeeds without restarting recording.
- [ ] 6.3 Implement async plugin shutdown draining with existing component wait budgets and retained cleanup ownership while its runtime is alive; verify event-loop responsiveness, admission refusal, repeated-job descriptor/process stability and honest reporting of unfinished model/native calls. Separately test normal runtime process-group stop without accepting interrupted results; document that group signals can bypass cooperative cleanup and abrupt parent death does not guarantee descendant cleanup. Preserve SIGTERM behavior needed by broken-pool retirement.

## 7. Verify end-to-end acceptance and prepare publication

- [ ] 7.1 Run paired baseline/native batches for preparation, process round trips and full pipeline/model work at identical settings; deliver the design's CPU/latency/RSS/output/thread evidence and explain variance. Correct repeatable regressions or obtain explicit evidence-based acceptance before cutover; claim speedup only when distinguishable from noise.
- [ ] 7.2 Run local IPyCam recording with a temporary recording/upload folder while analysis jobs run, cancel and crash; verify clip completion, video/audio playability, upload results and recording continuity against baseline. Use the local testing runbook, isolated ports and no Lenovo/production changes; obtain immediate approval before any required disposable Postgres writes.
- [ ] 7.3 Run strict OpenSpec validation, native unit/boundary tests, installed-package matrix smoke and AGENTS.md-required `make typecheck` / full `make check`; obtain explicit approval before database-writing checks or a narrower publication scope. Verify no unintended files, dependency changes or secret/media artifacts are staged.
- [ ] 7.4 Update the scenario evidence register and reference links with actual verification levels, run an independent implementation review, and publish follow-up commits only after required checks; verify ticket/PR status describes implemented evidence rather than treating planning completeness as runtime proof.

## Scenario-to-verification map

| Delta scenarios | Tasks providing meaningful evidence |
| --- | --- |
| Installed media tools do not change analysis preparation; preparation failure is contained | 2.2, 2.5, 5.4, 6.2, 7.2 |
| YOLO sampling/early exit; VLM sampling/settings; decoder metadata/read failures | 1.1–1.2, 3.2–3.3, 4.1–4.3, 5.2 |
| Skipped VLM performs no preparation | 5.1, 5.3 |
| Slow/cancelled work does not grow a hidden queue | 6.1, 6.3, 7.1 |
| Cancellation releases resources; broken pool recovery; cooperative plugin shutdown; whole-runtime termination | 6.1–6.3 |
| Clean installed preparation; actionable missing prerequisites | 2.3–2.5, 7.3 |

## Workflow follow-up

- This planning PR stops before implementation. Apply requires a new explicit implementation request.
- Archive/promote the delta only after implementation, review and evidence are complete; retain known limits in the maintained spec/reference.
- Keep ticket 191's durable handoff, ticket 194/PR 124's probing and tickets 195/196's other execution/CLI retirement separate. Do not merge a default branch without explicit immediate approval.
