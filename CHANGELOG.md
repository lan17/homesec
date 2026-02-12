# CHANGELOG

<!-- version list -->

## v1.3.0 (2026-02-12)

### Continuous Integration

- Allow manual docker publish ([#22](https://github.com/lan17/homesec/pull/22),
  [`d6a31fe`](https://github.com/lan17/homesec/commit/d6a31fe8ed2c4f3fce56eed473b57953a43af742))

### Features

- **rtsp**: Startup preflight, profile locking, and reconnect hardening
  ([#23](https://github.com/lan17/homesec/pull/23),
  [`e2f094e`](https://github.com/lan17/homesec/commit/e2f094e9efc2a534a5d1ee55368cb7e4d545f1b2))


## v1.2.3 (2026-02-04)

### Bug Fixes

- Fix dockerfile ([#21](https://github.com/lan17/homesec/pull/21),
  [`29e8ce4`](https://github.com/lan17/homesec/commit/29e8ce43a667d288b668417b7735c0c51620fca1))

### Continuous Integration

- Add docker publish workflow ([#20](https://github.com/lan17/homesec/pull/20),
  [`0ec59a0`](https://github.com/lan17/homesec/commit/0ec59a0509cf7a7a98900963d3d265bfdd2cb65c))

### Refactoring

- Plugin config boundary cleanup ([#18](https://github.com/lan17/homesec/pull/18),
  [`6746fa2`](https://github.com/lan17/homesec/commit/6746fa2f4285657df01d0dc1d25c7a1d1580407b))


## v1.2.2 (2026-01-27)

### Bug Fixes

- Align recording sensitivity constraints ([#16](https://github.com/lan17/homesec/pull/16),
  [`8ac42d2`](https://github.com/lan17/homesec/commit/8ac42d2e828cf2c5e9f7c0cc5da32b677ddbbd54))

- Defer rtsp detect fallback while recording ([#16](https://github.com/lan17/homesec/pull/16),
  [`8ac42d2`](https://github.com/lan17/homesec/commit/8ac42d2e828cf2c5e9f7c0cc5da32b677ddbbd54))

- Harden rtsp recording retries ([#16](https://github.com/lan17/homesec/pull/16),
  [`8ac42d2`](https://github.com/lan17/homesec/commit/8ac42d2e828cf2c5e9f7c0cc5da32b677ddbbd54))

- Honor exact rtsp reconnect attempts ([#16](https://github.com/lan17/homesec/pull/16),
  [`8ac42d2`](https://github.com/lan17/homesec/commit/8ac42d2e828cf2c5e9f7c0cc5da32b677ddbbd54))

- Improve rtsp reconnect and fallback ([#16](https://github.com/lan17/homesec/pull/16),
  [`8ac42d2`](https://github.com/lan17/homesec/commit/8ac42d2e828cf2c5e9f7c0cc5da32b677ddbbd54))

- Make recording motion threshold more sensitive ([#16](https://github.com/lan17/homesec/pull/16),
  [`8ac42d2`](https://github.com/lan17/homesec/commit/8ac42d2e828cf2c5e9f7c0cc5da32b677ddbbd54))

### Chores

- Remove rtsp improvements plan from repo ([#16](https://github.com/lan17/homesec/pull/16),
  [`8ac42d2`](https://github.com/lan17/homesec/commit/8ac42d2e828cf2c5e9f7c0cc5da32b677ddbbd54))

### Documentation

- Add homesec db logs skill ([#16](https://github.com/lan17/homesec/pull/16),
  [`8ac42d2`](https://github.com/lan17/homesec/commit/8ac42d2e828cf2c5e9f7c0cc5da32b677ddbbd54))

### Refactoring

- Centralize rtsp recording state ([#16](https://github.com/lan17/homesec/pull/16),
  [`8ac42d2`](https://github.com/lan17/homesec/commit/8ac42d2e828cf2c5e9f7c0cc5da32b677ddbbd54))

- Export rtsp public api ([#16](https://github.com/lan17/homesec/pull/16),
  [`8ac42d2`](https://github.com/lan17/homesec/commit/8ac42d2e828cf2c5e9f7c0cc5da32b677ddbbd54))

- Extract rtsp motion detector ([#16](https://github.com/lan17/homesec/pull/16),
  [`8ac42d2`](https://github.com/lan17/homesec/commit/8ac42d2e828cf2c5e9f7c0cc5da32b677ddbbd54))

- Make rtsp pipeline lifecycle explicit ([#16](https://github.com/lan17/homesec/pull/16),
  [`8ac42d2`](https://github.com/lan17/homesec/commit/8ac42d2e828cf2c5e9f7c0cc5da32b677ddbbd54))

- Reorganize source configs ([#16](https://github.com/lan17/homesec/pull/16),
  [`8ac42d2`](https://github.com/lan17/homesec/commit/8ac42d2e828cf2c5e9f7c0cc5da32b677ddbbd54))

- Restrict rtsp package exports ([#16](https://github.com/lan17/homesec/pull/16),
  [`8ac42d2`](https://github.com/lan17/homesec/commit/8ac42d2e828cf2c5e9f7c0cc5da32b677ddbbd54))

- Simplify rtsp run loop ([#16](https://github.com/lan17/homesec/pull/16),
  [`8ac42d2`](https://github.com/lan17/homesec/commit/8ac42d2e828cf2c5e9f7c0cc5da32b677ddbbd54))

- Simplify rtsp run loop and probes ([#16](https://github.com/lan17/homesec/pull/16),
  [`8ac42d2`](https://github.com/lan17/homesec/commit/8ac42d2e828cf2c5e9f7c0cc5da32b677ddbbd54))

- Split rtsp into package modules ([#16](https://github.com/lan17/homesec/pull/16),
  [`8ac42d2`](https://github.com/lan17/homesec/commit/8ac42d2e828cf2c5e9f7c0cc5da32b677ddbbd54))

- Tighten rtsp detect fallback and backoff ([#16](https://github.com/lan17/homesec/pull/16),
  [`8ac42d2`](https://github.com/lan17/homesec/commit/8ac42d2e828cf2c5e9f7c0cc5da32b677ddbbd54))

### Testing

- Add rtsp reconnect deferral coverage ([#16](https://github.com/lan17/homesec/pull/16),
  [`8ac42d2`](https://github.com/lan17/homesec/commit/8ac42d2e828cf2c5e9f7c0cc5da32b677ddbbd54))

- Add rtsp stall and detect recovery coverage ([#16](https://github.com/lan17/homesec/pull/16),
  [`8ac42d2`](https://github.com/lan17/homesec/commit/8ac42d2e828cf2c5e9f7c0cc5da32b677ddbbd54))

- Cover detect switch fallback ([#16](https://github.com/lan17/homesec/pull/16),
  [`8ac42d2`](https://github.com/lan17/homesec/commit/8ac42d2e828cf2c5e9f7c0cc5da32b677ddbbd54))

- Cover ffmpeg timeout fallback ([#16](https://github.com/lan17/homesec/pull/16),
  [`8ac42d2`](https://github.com/lan17/homesec/commit/8ac42d2e828cf2c5e9f7c0cc5da32b677ddbbd54))

- Cover ffprobe error handling ([#16](https://github.com/lan17/homesec/pull/16),
  [`8ac42d2`](https://github.com/lan17/homesec/commit/8ac42d2e828cf2c5e9f7c0cc5da32b677ddbbd54))

- Cover ffprobe timeout fallback ([#16](https://github.com/lan17/homesec/pull/16),
  [`8ac42d2`](https://github.com/lan17/homesec/commit/8ac42d2e828cf2c5e9f7c0cc5da32b677ddbbd54))

- Cover recording start backoff ([#16](https://github.com/lan17/homesec/pull/16),
  [`8ac42d2`](https://github.com/lan17/homesec/commit/8ac42d2e828cf2c5e9f7c0cc5da32b677ddbbd54))

- Cover rtsp frame pipeline edge cases ([#16](https://github.com/lan17/homesec/pull/16),
  [`8ac42d2`](https://github.com/lan17/homesec/commit/8ac42d2e828cf2c5e9f7c0cc5da32b677ddbbd54))

- Cover rtsp recording timers ([#16](https://github.com/lan17/homesec/pull/16),
  [`8ac42d2`](https://github.com/lan17/homesec/commit/8ac42d2e828cf2c5e9f7c0cc5da32b677ddbbd54))

- Expand rtsp config coverage ([#16](https://github.com/lan17/homesec/pull/16),
  [`8ac42d2`](https://github.com/lan17/homesec/commit/8ac42d2e828cf2c5e9f7c0cc5da32b677ddbbd54))

- Expand rtsp coverage and regroup tests ([#16](https://github.com/lan17/homesec/pull/16),
  [`8ac42d2`](https://github.com/lan17/homesec/commit/8ac42d2e828cf2c5e9f7c0cc5da32b677ddbbd54))

- Harden rtsp config paths ([#16](https://github.com/lan17/homesec/pull/16),
  [`8ac42d2`](https://github.com/lan17/homesec/commit/8ac42d2e828cf2c5e9f7c0cc5da32b677ddbbd54))

- Harden rtsp hardware detection ([#16](https://github.com/lan17/homesec/pull/16),
  [`8ac42d2`](https://github.com/lan17/homesec/commit/8ac42d2e828cf2c5e9f7c0cc5da32b677ddbbd54))


## v1.2.1 (2026-01-19)

### Bug Fixes

- Correct ToC anchor for 'With Docker' section ([#9](https://github.com/lan17/homesec/pull/9),
  [`77b5f65`](https://github.com/lan17/homesec/commit/77b5f65461ce4016186cfbf2b3961779fd5a9fef))

- Improve CLI help output to show available commands ([#9](https://github.com/lan17/homesec/pull/9),
  [`77b5f65`](https://github.com/lan17/homesec/commit/77b5f65461ce4016186cfbf2b3961779fd5a9fef))

### Chores

- Sync uv.lock version ([#9](https://github.com/lan17/homesec/pull/9),
  [`77b5f65`](https://github.com/lan17/homesec/commit/77b5f65461ce4016186cfbf2b3961779fd5a9fef))

- Update uv.lock
  ([`f417abc`](https://github.com/lan17/homesec/commit/f417abc93632e52dbb8bb109ba9bedf354713252))

### Documentation

- Add .env setup to manual quickstart to address feedback
  ([#9](https://github.com/lan17/homesec/pull/9),
  [`77b5f65`](https://github.com/lan17/homesec/commit/77b5f65461ce4016186cfbf2b3961779fd5a9fef))

- Add installation section and update CLI examples ([#9](https://github.com/lan17/homesec/pull/9),
  [`77b5f65`](https://github.com/lan17/homesec/commit/77b5f65461ce4016186cfbf2b3961779fd5a9fef))

- Add Postgres state tracking to mermaid diagram ([#15](https://github.com/lan17/homesec/pull/15),
  [`908b0d8`](https://github.com/lan17/homesec/commit/908b0d807dfe91eae5a0b29601c598d305c676d6))

- Address copilot feedback (ToC links, hierarchy, config cleanup)
  ([#15](https://github.com/lan17/homesec/pull/15),
  [`908b0d8`](https://github.com/lan17/homesec/commit/908b0d807dfe91eae5a0b29601c598d305c676d6))

- Address PR feedback on Intro and Design Principles ([#9](https://github.com/lan17/homesec/pull/9),
  [`77b5f65`](https://github.com/lan17/homesec/commit/77b5f65461ce4016186cfbf2b3961779fd5a9fef))

- Clean up mermaid diagram with simplified Postgres tracking
  ([#15](https://github.com/lan17/homesec/pull/15),
  [`908b0d8`](https://github.com/lan17/homesec/commit/908b0d807dfe91eae5a0b29601c598d305c676d6))

- Consolidate plugin sections with interfaces table ([#9](https://github.com/lan17/homesec/pull/9),
  [`77b5f65`](https://github.com/lan17/homesec/commit/77b5f65461ce4016186cfbf2b3961779fd5a9fef))

- Fix ToC links to point to valid headers ([#15](https://github.com/lan17/homesec/pull/15),
  [`908b0d8`](https://github.com/lan17/homesec/commit/908b0d807dfe91eae5a0b29601c598d305c676d6))

- Fix yolo docstring in tests ([#9](https://github.com/lan17/homesec/pull/9),
  [`77b5f65`](https://github.com/lan17/homesec/commit/77b5f65461ce4016186cfbf2b3961779fd5a9fef))

- Huge readme refactor and config update ([#9](https://github.com/lan17/homesec/pull/9),
  [`77b5f65`](https://github.com/lan17/homesec/commit/77b5f65461ce4016186cfbf2b3961779fd5a9fef))

- Humanize tone, simpler phrasing, no em-dashes ([#9](https://github.com/lan17/homesec/pull/9),
  [`77b5f65`](https://github.com/lan17/homesec/commit/77b5f65461ce4016186cfbf2b3961779fd5a9fef))

- Improve mermaid diagram to show clip file intermediate step
  ([#9](https://github.com/lan17/homesec/pull/9),
  [`77b5f65`](https://github.com/lan17/homesec/commit/77b5f65461ce4016186cfbf2b3961779fd5a9fef))

- Improve README intro with clearer value prop and PyPI badge
  ([#9](https://github.com/lan17/homesec/pull/9),
  [`77b5f65`](https://github.com/lan17/homesec/commit/77b5f65461ce4016186cfbf2b3961779fd5a9fef))

- Move Clip Source inside subgraph to fix title overlap
  ([#15](https://github.com/lan17/homesec/pull/15),
  [`908b0d8`](https://github.com/lan17/homesec/commit/908b0d807dfe91eae5a0b29601c598d305c676d6))

- Overhaul README & Update Config for Accuracy/DX ([#9](https://github.com/lan17/homesec/pull/9),
  [`77b5f65`](https://github.com/lan17/homesec/commit/77b5f65461ce4016186cfbf2b3961779fd5a9fef))

- Polish README - consistent paths, clearer tips ([#9](https://github.com/lan17/homesec/pull/9),
  [`77b5f65`](https://github.com/lan17/homesec/commit/77b5f65461ce4016186cfbf2b3961779fd5a9fef))

- Position Postgres to the right of pipeline ([#15](https://github.com/lan17/homesec/pull/15),
  [`908b0d8`](https://github.com/lan17/homesec/commit/908b0d807dfe91eae5a0b29601c598d305c676d6))

- Refine mermaid diagram (graph LR, remove styles) ([#15](https://github.com/lan17/homesec/pull/15),
  [`908b0d8`](https://github.com/lan17/homesec/commit/908b0d807dfe91eae5a0b29601c598d305c676d6))

- Refine mermaid layout with nested wrapper subgraph
  ([#15](https://github.com/lan17/homesec/pull/15),
  [`908b0d8`](https://github.com/lan17/homesec/commit/908b0d807dfe91eae5a0b29601c598d305c676d6))

- Refine README tone to be more technical ([#9](https://github.com/lan17/homesec/pull/9),
  [`77b5f65`](https://github.com/lan17/homesec/commit/77b5f65461ce4016186cfbf2b3961779fd5a9fef))

- Remove dup header, fix ToC, update yolo docstring ([#9](https://github.com/lan17/homesec/pull/9),
  [`77b5f65`](https://github.com/lan17/homesec/commit/77b5f65461ce4016186cfbf2b3961779fd5a9fef))

- Rename RTSP var to DRIVEWAY_RTSP_URL to match camera name
  ([#9](https://github.com/lan17/homesec/pull/9),
  [`77b5f65`](https://github.com/lan17/homesec/commit/77b5f65461ce4016186cfbf2b3961779fd5a9fef))

- Reorganize README table of contents ([#9](https://github.com/lan17/homesec/pull/9),
  [`77b5f65`](https://github.com/lan17/homesec/commit/77b5f65461ce4016186cfbf2b3961779fd5a9fef))

- Restructure Configuration with minimal and full examples
  ([#9](https://github.com/lan17/homesec/pull/9),
  [`77b5f65`](https://github.com/lan17/homesec/commit/77b5f65461ce4016186cfbf2b3961779fd5a9fef))

- Restructure Quickstart - shared config, Docker/non-Docker run options
  ([#9](https://github.com/lan17/homesec/pull/9),
  [`77b5f65`](https://github.com/lan17/homesec/commit/77b5f65461ce4016186cfbf2b3961779fd5a9fef))

- Revert mermaid diagram to vertical (graph TD) ([#15](https://github.com/lan17/homesec/pull/15),
  [`908b0d8`](https://github.com/lan17/homesec/commit/908b0d807dfe91eae5a0b29601c598d305c676d6))

- Revert to mermaid diagram but make it vertical ([#9](https://github.com/lan17/homesec/pull/9),
  [`77b5f65`](https://github.com/lan17/homesec/commit/77b5f65461ce4016186cfbf2b3961779fd5a9fef))

- Simplify Running without Docker, expand Development section
  ([#9](https://github.com/lan17/homesec/pull/9),
  [`77b5f65`](https://github.com/lan17/homesec/commit/77b5f65461ce4016186cfbf2b3961779fd5a9fef))

- Streamline Quickstart for pip-only users ([#9](https://github.com/lan17/homesec/pull/9),
  [`77b5f65`](https://github.com/lan17/homesec/commit/77b5f65461ce4016186cfbf2b3961779fd5a9fef))

- Sync readme config examples with .env.example ([#9](https://github.com/lan17/homesec/pull/9),
  [`77b5f65`](https://github.com/lan17/homesec/commit/77b5f65461ce4016186cfbf2b3961779fd5a9fef))

- Update Highlights - YOLO11 and remove redundant bullet
  ([#9](https://github.com/lan17/homesec/pull/9),
  [`77b5f65`](https://github.com/lan17/homesec/commit/77b5f65461ce4016186cfbf2b3961779fd5a9fef))

- Wrap mermaid pipeline in subgraph for better visual grouping
  ([#15](https://github.com/lan17/homesec/pull/15),
  [`908b0d8`](https://github.com/lan17/homesec/commit/908b0d807dfe91eae5a0b29601c598d305c676d6))

### Refactoring

- Rename YOLOv8Filter to YOLOFilter and overhaul docs based on feedback
  ([#9](https://github.com/lan17/homesec/pull/9),
  [`77b5f65`](https://github.com/lan17/homesec/commit/77b5f65461ce4016186cfbf2b3961779fd5a9fef))

- Use Fire's native help output for CLI commands ([#9](https://github.com/lan17/homesec/pull/9),
  [`77b5f65`](https://github.com/lan17/homesec/commit/77b5f65461ce4016186cfbf2b3961779fd5a9fef))


## v1.2.0 (2026-01-19)

### Bug Fixes

- **rtsp**: Remove incompatible -rw_timeout flag and unused variables
  ([`40339a2`](https://github.com/lan17/homesec/commit/40339a27ff3f45117ef918c6485da8dbcc0be724))

- **rtsp**: Use -vsync 0 instead of -fps_mode for older ffmpeg compat
  ([`22fc7e3`](https://github.com/lan17/homesec/commit/22fc7e3fab57aefa869d4246b036d5d0b47494dd))

### Chores

- Sync uv.lock with project version 1.1.2 ([#14](https://github.com/lan17/homesec/pull/14),
  [`64f50d8`](https://github.com/lan17/homesec/commit/64f50d86d22f0a52a8298c9ba38146cc17a1f0be))

### Features

- **rtsp**: Add configurable ffmpeg_flags and robust defaults
  ([`e460637`](https://github.com/lan17/homesec/commit/e460637bbdeee7f3947ddf2778a69c980edcf24b))


## v1.1.2 (2026-01-19)

### Bug Fixes

- Use PAT token for release workflow to bypass branch protection
  ([#13](https://github.com/lan17/homesec/pull/13),
  [`511ac6d`](https://github.com/lan17/homesec/commit/511ac6d8899365d324ede41aefdbe8fab910f1cb))

### Refactoring

- Complete plugin architecture standardization ([#10](https://github.com/lan17/homesec/pull/10),
  [`4e5b85f`](https://github.com/lan17/homesec/commit/4e5b85fd9a32d4f71c4365222a735c2e01eba583))


## v1.1.1 (2026-01-16)

### Bug Fixes

- Improve CLI help output to show available commands ([#8](https://github.com/lan17/homesec/pull/8),
  [`4c8342f`](https://github.com/lan17/homesec/commit/4c8342f71274f2110231f112b77d68c8bb119c17))

### Chores

- Sync uv.lock version ([#8](https://github.com/lan17/homesec/pull/8),
  [`4c8342f`](https://github.com/lan17/homesec/commit/4c8342f71274f2110231f112b77d68c8bb119c17))

### Refactoring

- Use Fire's native help output for CLI commands ([#8](https://github.com/lan17/homesec/pull/8),
  [`4c8342f`](https://github.com/lan17/homesec/commit/4c8342f71274f2110231f112b77d68c8bb119c17))


## v1.1.0 (2026-01-12)

### Bug Fixes

- Add Codecov token to CI workflow ([#5](https://github.com/lan17/homesec/pull/5),
  [`6657cf2`](https://github.com/lan17/homesec/commit/6657cf2e94fe7bf8d0eaed39cfa98242f9766188))

- Correct Codecov badge URL case (HomeSec) ([#6](https://github.com/lan17/homesec/pull/6),
  [`74019fe`](https://github.com/lan17/homesec/commit/74019fe4b313d588a72e96ff86a7b647e69c8150))

### Features

- Add code coverage and improve documentation ([#4](https://github.com/lan17/homesec/pull/4),
  [`b1d9490`](https://github.com/lan17/homesec/commit/b1d949057db675ed20f2623f06cef1fd58c4dcc3))

### Testing

- Improve code coverage from 64% to 70% ([#7](https://github.com/lan17/homesec/pull/7),
  [`7b93468`](https://github.com/lan17/homesec/commit/7b934685e8389bb24c8cc8878cb43b5570f5371e))


## v1.0.2 (2026-01-11)

### Bug Fixes

- CI and release workflow improvements ([#3](https://github.com/lan17/homesec/pull/3),
  [`ab61e47`](https://github.com/lan17/homesec/commit/ab61e47c7c16c79e8472807c07e4974e31a13c29))


## v1.0.1 (2026-01-11)

### Bug Fixes

- Improve release workflow dry run and prevent major version jumps
  ([#2](https://github.com/lan17/HomeSec/pull/2),
  [`ba02512`](https://github.com/lan17/HomeSec/commit/ba025129de5caa984552807e4e0f376666db485e))


## v1.0.0 (2026-01-11)

- Initial Release
