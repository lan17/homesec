# Continuous integration

`.github/workflows/ci.yml` keeps the original `CI` workflow and final `ci`
check, but schedules independent checks on separate Ubuntu 24.04 runners.
No product behavior, test selection, test concurrency, or native build recipe
changes are part of this refactor.

## Job graph

```text
rust-format -------------------------------+
python-static [lint, typecheck] ------------+
ui [api:check, lint, test, build] -----------+--> ci
native-deps ---+--> rust-tests -------------+
              +--> native-build -----------+
                        +--> python-tests -+
```

Matrix jobs use `fail-fast: false` so one failure does not cancel sibling
checks. The aggregate `ci` job runs with `always()` and requires every direct
dependency to succeed; failures, cancellations, and skips are not accepted.
Every non-gate job is a direct dependency of `ci`, including the producers.
Existing default-branch push/PR triggers and cancellation of superseded runs
are retained. Branch-protection settings are not changed.

Rust formatting no longer installs native libraries. Clippy and Rust tests
share one runner to reuse their debug build outputs. The release helper builds
on a separate runner and retains the static-media-library linkage check.
The complete Python suite waits only for that release build, not Rust tests.
It still receives the current run's helper on PATH, FFmpeg/ffprobe, Postgres,
migrations, and the existing coverage command. The helper is a required,
commit-named artifact, not a binary inferred from a cache hit; executable
permission is restored after download. Artifact replacement permits rerunning
the producer within the same workflow run.

UI checks use the checked-in generated contract independently. `api:check`
installs the locked Python environment to run the existing contract exporter;
it does not need a live backend or database. `build` already runs TypeScript
checking, so there is no duplicate typecheck lane. `UV_LOCKED=1` also covers
the exporter script's nested `uv run` command.

## Caching

The `setup-native` composite action restores only the exact native-library
cache for the runner image, architecture, UID, compiler/CMake versions, pinned
Cargo/toolchain inputs, and native recipe. The `native-deps` job always runs
the existing wrapper with `cargo --version`: the wrapper validates/prepares
x264, Opus, FFmpeg, and OpenCV before invoking that command. Completed libraries
are saved immediately, rather than waiting for unrelated checks to succeed.

Native consumers restore that cache and still run the wrapper, including its
fingerprint, ownership, completeness, and static-library checks. Cache eviction,
service unavailability, or a runner-image mismatch can cause a rebuild; caches
are accelerators, not correctness prerequisites. Normally a cold matching
cache is populated by the producer before either consumer starts. No library
artifact, new build-only mode, or native locking change is introduced.

Cargo check/release outputs have separate keys and source-versioned entries,
with same-environment restore prefixes. Python retains its venv cache plus uv
package caching, and each use synchronizes the locked environment. UI lanes
retain the pnpm store cache and frozen-lockfile installation.

## Verification and follow-up

Run `uv run --locked python .github/scripts/test_ci.py` to check graph and
command preservation, cache-save ordering, artifact handoff, conditional Python
setup, and shell syntax. The tests execute the exact gate shell command for
all-success and individual failure/cancellation/skip cases. CI also lints,
formats, and type-checks this support script.

These tests do not emulate GitHub's scheduler or execute application tests.
The actual CI run must validate tool installation, both cold/warm cache paths,
artifact transfer, native linkage, and the complete Rust/Python/UI suites.
Compare cold and warm job durations before claiming a measured speedup:
extra runners introduce installation, cache-transfer, and scheduling overhead.

Further Python test sharding requires an inventory of native-helper use and
shared database/file/socket fixtures. Native-library parallelism would require
reviewing the global build lock and CPU budget. Neither optimization, path-based
skipping, dependency upgrades, nor test retries are included here.

This is an internal CI refactor, so its design and verification live here and
in the PR rather than introducing product requirements under `openspec/specs`.
