# Onvif Cli Port

> 33 nodes · cohesion 0.06

## Key Concepts

- **test_onvif_cli.py** (10 connections) — `tests/homesec/test_onvif_cli.py`
- **_parse_host_port()** (6 connections) — `src/homesec/onvif/cli.py`
- **cli.py** (6 connections) — `src/homesec/onvif/cli.py`
- **.streams()** (5 connections) — `src/homesec/onvif/cli.py`
- **test_onvif_cli_discover_prints_results()** (4 connections) — `tests/homesec/test_onvif_cli.py`
- **_exit_with_error()** (4 connections) — `src/homesec/onvif/cli.py`
- **.discover()** (4 connections) — `src/homesec/onvif/cli.py`
- **test_onvif_cli_discover_prints_debug_tips_when_no_cameras_found()** (3 connections) — `tests/homesec/test_onvif_cli.py`
- **test_onvif_cli_info_bounds_close_timeout()** (3 connections) — `tests/homesec/test_onvif_cli.py`
- **test_onvif_cli_info_preserves_success_when_close_fails()** (3 connections) — `tests/homesec/test_onvif_cli.py`
- **test_onvif_cli_info_prints_device_and_profiles()** (3 connections) — `tests/homesec/test_onvif_cli.py`
- **test_onvif_cli_streams_exits_with_error_when_client_fails()** (3 connections) — `tests/homesec/test_onvif_cli.py`
- **test_onvif_cli_streams_preserves_primary_error_when_close_fails()** (3 connections) — `tests/homesec/test_onvif_cli.py`
- **test_parse_host_port_supports_bare_ipv6_literal()** (3 connections) — `tests/homesec/test_onvif_cli.py`
- **test_parse_host_port_supports_bracketed_ipv6_with_port()** (3 connections) — `tests/homesec/test_onvif_cli.py`
- **_build_default_onvif_service()** (3 connections) — `src/homesec/onvif/cli.py`
- **main()** (2 connections) — `src/homesec/onvif/cli.py`
- **.__init__()** (2 connections) — `src/homesec/onvif/cli.py`
- **Tests for ONVIF standalone CLI command handlers.** (1 connections) — `tests/homesec/test_onvif_cli.py`
- **OnvifCLI.info should preserve successful output when close fails.** (1 connections) — `tests/homesec/test_onvif_cli.py`
- **OnvifCLI.streams should exit non-zero and print error on client failure.** (1 connections) — `tests/homesec/test_onvif_cli.py`
- **OnvifCLI.streams should preserve stream failure when close also fails.** (1 connections) — `tests/homesec/test_onvif_cli.py`
- **OnvifCLI.discover should print discovered devices.** (1 connections) — `tests/homesec/test_onvif_cli.py`
- **OnvifCLI.info should not block on close longer than configured timeout.** (1 connections) — `tests/homesec/test_onvif_cli.py`
- **_parse_host_port should not misinterpret bare IPv6 literals as host:port.** (1 connections) — `tests/homesec/test_onvif_cli.py`
- *... and 8 more nodes in this community*

## Relationships

- No strong cross-community connections detected

## Source Files

- `src/homesec/onvif/cli.py`
- `tests/homesec/test_onvif_cli.py`

## Audit Trail

- EXTRACTED: 70 (82%)
- INFERRED: 15 (18%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*