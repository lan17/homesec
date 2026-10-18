# HomeSec documentation

Start with the [project README](../README.md) for installation and configuration.
The [behavior reference](reference.md) connects the maintained
[capability specs](../openspec/specs/) to known limitations and
[scenario evidence](behavior-evidence.md). Guides describe how to operate or
extend the existing implementation; use the specs and reference for precise
behavior and verification limits.

## Operator guides

| Guide | Use it for |
| --- | --- |
| [Preview deployment](preview-deployment.md) | HTTP/WebRTC reachability, HLS scratch storage, and recording/preview resource policy |
| [WebRTC and native media](webrtc-preview.md) | Native helper installation, build prerequisites, codecs, motion/recording paths, and troubleshooting |
| [Push-to-talk](push-to-talk.md) | Browser microphone requirements, camera backends, configuration, and session troubleshooting |
| [Postgres backups](postgres-backups.md) | Dump prerequisites, durable backup directories, retention/status interpretation, and restore procedures |

## Developer guides

| Guide | Use it for |
| --- | --- |
| [Architecture](../DESIGN.md) | Component ownership, plugin boundaries, persistence, and runtime lifecycle |
| [Shared Rust media](shared-rust-media.md) | Shared camera-input ownership, native/compatibility paths, and historical rollout evidence |
| [Plugin development](../PLUGIN_DEVELOPMENT.md) | Registry integration and complete plugin examples |
| [Testing](../TESTING.md) | Behavioral tests, boundary fakes, and database isolation |
| [CI](../CI.md) | Check dependencies, native artifacts/caches, and CI verification limits |
| [UI development](../ui/README.md) | Frontend setup, generated API models, and local UI checks |
| [Development rules](../AGENTS.md) | Repository constraints and required validation |

## History and maintenance

[CHANGELOG.md](../CHANGELOG.md) records released changes. OpenSpec's
[archived changes](../openspec/changes/archive/) preserve historical proposals
and review evidence; they are not the current behavior contract. Generated wiki
pages and local planning scratch files can lag the code and should not replace
the maintained guides above.

When changing public behavior, follow the [OpenSpec workflow](../openspec/README.md)
and update the affected guide alongside the specification and implementation.
For documentation corrections, check claims against current code/configuration
and preserve the distinction between source inspection, executed tests, and
device/provider validation.
