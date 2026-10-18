# Proposal

## Why

Completed clips still reach YOLO and the OpenAI-compatible analyzer through Python OpenCV decoding, independently of HomeSec's pinned Rust media build. Moving general preparation into a packaged Rust library closes that versioning gap and isolates VLM preparation without changing the Python analysis policy or assuming a performance win.

## What Changes

- Prepare local-clip frames through a private PyO3 extension inside Python analysis workers: decode and select full-size BGR frames for existing Python YOLO; decode, sample, resize and JPEG-encode images for eligible VLM analysis.
- Preserve consumer-specific sampling, timestamps, dimensions, settings, YOLO early exit and result semantics. Equivalent images are acceptable with explicit parity gates; JPEG bytes need not match Pillow.
- Keep YOLO's process pool and add a lazy VLM preparation pool bounded by existing runtime concurrency. Cancel queued work, cooperate between preparation steps, recover broken pools for Python-controlled retries, and allow ongoing model inference to finish.
- Package one HomeSec distribution with Maturin, a pre-Cargo native bootstrap and the existing verified FFmpeg source pin. Installations must not select system media libraries or retain an OpenCV-wheel decoder bypass.
- Verify correctness, worker failure/recovery, packaging and paired baseline performance before enabling the new path; retain provider calls, inference, retries, alerts and repository ownership in Python.

Leonidas confirmed cooperative cancellation/pool recovery, equivalent-image parity and one-package Maturin packaging on 2026-10-18. This PR contains planning artifacts only.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `analysis-alerts`: add isolated preparation, consumer-compatible inputs, bounded cancellation/recovery and pinned installation contracts. Existing stage ordering, result limitations and retry policy remain applicable.

## Impact

- Analysis adapters and tests: `plugins/filters/yolo.py`, `plugins/analyzers/openai.py`, analyzer loading/runtime assembly and existing pipeline boundary tests.
- Native packaging: a private preparation crate, shared private bootstrap in `native/webrtc/build_native.py`, Python build metadata/lock, Makefile, Docker, CI/release wheel and source builds. Reuse the FFmpeg pin while keeping the preparation artifact's dependencies explicit.
- Documentation/evidence: analysis reference, native build/development instructions and scenario evidence updated with implementation, not promoted by this planning PR.
- No new HTTP API, user configuration setting, inference backend or service. No camera-session changes, durable handoff implementation, CLI retirement for unrelated media paths, production testing or database changes.
- Independent of [camera preflight PR 124](https://github.com/lan17/homesec/pull/124) and ticket 191; implementation must reconcile their shared bootstrap/build-file edits. Tracking: [ticket 190](https://app.notion.com/p/3fdd8336c59f81aeb3a0d280767dd3a3), [migration epic](https://app.notion.com/p/3efd8336c59f8195b6b5c50c71c1d5e7).
