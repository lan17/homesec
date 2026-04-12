# Services Development Notes

## Setup Probes

Setup probes validate backend connectivity during onboarding. Each probe lives beside its backend:

| Backend | Probe module |
|---------|-------------|
| RTSP | `sources/rtsp/setup_probe.py` |
| FTP | `sources/ftp_setup_probe.py` |
| Local folder | `sources/local_folder_setup_probe.py` |
| ONVIF | `onvif/setup_probe.py` |
| Local storage | `plugins/storage/local_setup_probe.py` |

**Registry:** `services/setup_probes.py` — contains `SetupProbeRegistry`, the `@setup_probe` decorator, and `_BUILTIN_SETUP_PROBE_MODULES` manifest.

**Adding a new probe:** Create a module beside the backend, decorate with `@setup_probe(target, backend)`, add the module path to `_BUILTIN_SETUP_PROBE_MODULES` in `setup_probes.py`.

**Shared helpers:** `services/setup_probe_support.py` has `build_test_connection_response`, `format_validation_error`, and `SETUP_TEST_CAMERA_NAME`. Domain-specific helpers (timeout constants, config models, URL resolvers) live in the probe module that uses them.

**Testing:** Patch the probe module directly (e.g., `rtsp_setup_probe.validate_plugin`), not `setup_service`. See `tests/homesec/test_setup_test_connection_service.py`.
