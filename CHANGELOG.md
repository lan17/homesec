# CHANGELOG

<!-- version list -->

## v1.9.0 (2026-04-24)

### Bug Fixes

- Classify RTSP concurrent preview support ([#63](https://github.com/lan17/homesec/pull/63),
  [`eaa585f`](https://github.com/lan17/homesec/commit/eaa585fd3dea782d2b2786fbd4052ea2428a8914))

- Count alerts from alert decision events ([#45](https://github.com/lan17/homesec/pull/45),
  [`51f216a`](https://github.com/lan17/homesec/commit/51f216a2d48d4b106a14ac6a9ddd4f9a257bd9e5))

- Make runtime resilient for dev/CI environments ([#43](https://github.com/lan17/homesec/pull/43),
  [`c89ea3d`](https://github.com/lan17/homesec/commit/c89ea3d77d148b80297bdb47f21cbe9f57ceea50))

- Update agents.md and uv.lock ([#47](https://github.com/lan17/homesec/pull/47),
  [`04521ba`](https://github.com/lan17/homesec/commit/04521ba9113b1064822539b7951b43e90eb6772c))

### Continuous Integration

- Publish compose artifact alongside docker image ([#39](https://github.com/lan17/homesec/pull/39),
  [`dd4a946`](https://github.com/lan17/homesec/commit/dd4a94669e8ba317ecd312fa2b67c7fc494cd3c4))

### Features

- Add configurable Postgres backups
  ([`6222c9e`](https://github.com/lan17/homesec/commit/6222c9ebe0b3ae1d6f776a0119cf6ebfcdaa4290))

- Add fake ONVIF + RTSP camera for local development
  ([#41](https://github.com/lan17/homesec/pull/41),
  [`f2f32a7`](https://github.com/lan17/homesec/commit/f2f32a77e6f3400a77da8c09797e8182dc995da2))

- Integrate RTSP live preview stack ([#62](https://github.com/lan17/homesec/pull/62),
  [`8717d2b`](https://github.com/lan17/homesec/commit/8717d2b407702e8e62882c8dc4ff846af8c70c09))

- Make notifiers optional in Config model ([#42](https://github.com/lan17/homesec/pull/42),
  [`d0a725b`](https://github.com/lan17/homesec/commit/d0a725be360fd9ec4f124227199896ef0560d26c))

### Refactoring

- Add setup probe registry for setup flow ([#46](https://github.com/lan17/homesec/pull/46),
  [`8ddc25a`](https://github.com/lan17/homesec/commit/8ddc25a81ed0fcc284091b501f83056bf94be972))

- Centralize clip status transitions ([#49](https://github.com/lan17/homesec/pull/49),
  [`20c0484`](https://github.com/lan17/homesec/commit/20c0484149715decfcb79b29376f0ea97fe8d329))

- Move setup probes beside backends ([#51](https://github.com/lan17/homesec/pull/51),
  [`9cbd920`](https://github.com/lan17/homesec/commit/9cbd920bc1dca142ad4127fb5a6775d83c80fcf3))

- Share runtime persistence bootstrap wiring ([#48](https://github.com/lan17/homesec/pull/48),
  [`298a5c5`](https://github.com/lan17/homesec/commit/298a5c54072856500bddcea9776746a7628bab3d))

### Testing

- Isolate parallel postgres test runs ([#52](https://github.com/lan17/homesec/pull/52),
  [`61c1568`](https://github.com/lan17/homesec/commit/61c15688c5d7d6d2897be18e846b562906f9c9b0))


## v1.8.0 (2026-02-27)

### Features

- Always serve UI dist and fail fast when missing ([#38](https://github.com/lan17/homesec/pull/38),
  [`c1ae42c`](https://github.com/lan17/homesec/commit/c1ae42c1ad7e790a77fb94eaa0813cd367615f26))


## v1.7.0 (2026-02-27)

### Bug Fixes

- Clean setup probe timeout/error handling ([#33](https://github.com/lan17/homesec/pull/33),
  [`81631d6`](https://github.com/lan17/homesec/commit/81631d600135897710f1ce8a79082aa1f373ac79))

- **setup**: Finalize in-process without killing FastAPI
  ([`969b99f`](https://github.com/lan17/homesec/commit/969b99fda50699b3b6d26076af41b8529a880f4c))

### Chores

- Fixup setup.py
  ([`9b29b13`](https://github.com/lan17/homesec/commit/9b29b13387313b9fa105409d50d47ea6382599e7))

### Features

- Bootstrap mode + setup status/preflight endpoints
  ([#29](https://github.com/lan17/homesec/pull/29),
  [`9b7160f`](https://github.com/lan17/homesec/commit/9b7160fde0d0585225a48abde008a069acd25c0f))

- Setup wizard shell + welcome preflight + first-run redirect (#49, #50, #57)
  ([#34](https://github.com/lan17/homesec/pull/34),
  [`4ce496e`](https://github.com/lan17/homesec/commit/4ce496e8da1b38e85ac7c2aa3afd6d895f28d830))

- **setup**: Add generic setup test-connection endpoint
  ([#32](https://github.com/lan17/homesec/pull/32),
  [`4a428d6`](https://github.com/lan17/homesec/commit/4a428d69c860436d196ab1b36b21d95effaa1b34))

- **ui**: Complete reusable CameraAddFlow architecture (#51)
  ([#35](https://github.com/lan17/homesec/pull/35),
  [`a398383`](https://github.com/lan17/homesec/commit/a398383ebc633b34e3633f8f12f170ec54bdc43c))

- **ui**: Complete setup wizard flows through review + launch (#53/#54/#55/#56)
  ([#37](https://github.com/lan17/homesec/pull/37),
  [`1386d19`](https://github.com/lan17/homesec/commit/1386d19b98d42b83a5b9c4de40051a17511d87e1))

- **ui**: Integrate CameraAddFlow into setup camera step
  ([#36](https://github.com/lan17/homesec/pull/36),
  [`765b978`](https://github.com/lan17/homesec/commit/765b978c5886366ab40b27dad5ad59ddcab7d368))


## v1.6.0 (2026-02-25)

### Features

- Implement oldest-first local clip retention pruning
  ([#30](https://github.com/lan17/homesec/pull/30),
  [`54b4a47`](https://github.com/lan17/homesec/commit/54b4a47c468689a7d238ebb8714332458d79bfa7))

- Onvif support ([#28](https://github.com/lan17/homesec/pull/28),
  [`44e6b38`](https://github.com/lan17/homesec/commit/44e6b38ac47d1a3845c0074c12b7d4cc075b3b27))

- Trigger local retention pruning after upload (#71)
  ([#31](https://github.com/lan17/homesec/pull/31),
  [`9f88a3e`](https://github.com/lan17/homesec/commit/9f88a3ee16cc63a4a0cbecba9b53a655a85e4e18))


## v1.5.0 (2026-02-20)

### Features

- Camera apply-changes flow + secret-safe source_config patch UX
  ([#27](https://github.com/lan17/homesec/pull/27),
  [`93323ca`](https://github.com/lan17/homesec/commit/93323cab0b13ab8efbca01a067305c23e0a71ba4))

### Testing

- Add API bootstrap dependency matrix ([#25](https://github.com/lan17/homesec/pull/25),
  [`06d659c`](https://github.com/lan17/homesec/commit/06d659cf8fd463498a735635cddc8ccd27bfd7ec))

- **ui**: Add behavioral coverage for cameras management flows
  ([#26](https://github.com/lan17/homesec/pull/26),
  [`9279134`](https://github.com/lan17/homesec/commit/9279134e40bafc10363825f877c0476bc906fca5))


## v1.4.0 (2026-02-19)

### Features

- FastAPI server, supervised runtime, and self-serve web UI
  ([#24](https://github.com/lan17/homesec/pull/24),
  [`b850b09`](https://github.com/lan17/homesec/commit/b850b09170129ba83b4ee0d75e907078104ece92))


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
