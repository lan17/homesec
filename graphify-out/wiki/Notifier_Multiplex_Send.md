# Notifier Multiplex Send

> 79 nodes · cohesion 0.06

## Key Concepts

- **SendGridEmailNotifier** (42 connections) — `src/homesec/plugins/notifiers/sendgrid_email.py`
- **test_sendgrid_notifier.py** (24 connections) — `tests/homesec/test_sendgrid_notifier.py`
- **SequenceAnalysis** (19 connections) — `src/homesec/models/vlm.py`
- **_make_config()** (18 connections) — `tests/homesec/test_sendgrid_notifier.py`
- **_StubNotifier** (15 connections) — `tests/homesec/test_notifiers.py`
- **_make_alert()** (14 connections) — `tests/homesec/test_sendgrid_notifier.py`
- **SendGridEmailConfig** (14 connections) — `src/homesec/plugins/notifiers/sendgrid_email.py`
- **_mock_http_response()** (13 connections) — `tests/homesec/test_sendgrid_notifier.py`
- **_patch_session()** (12 connections) — `tests/homesec/test_sendgrid_notifier.py`
- **test_notifiers.py** (8 connections) — `tests/homesec/test_notifiers.py`
- **test_multiplex_notifier_fans_out_success()** (7 connections) — `tests/homesec/test_notifiers.py`
- **test_multiplex_notifier_raises_on_failure()** (7 connections) — `tests/homesec/test_notifiers.py`
- **test_send_escapes_analysis_html()** (7 connections) — `tests/homesec/test_sendgrid_notifier.py`
- **.send()** (7 connections) — `src/homesec/plugins/notifiers/sendgrid_email.py`
- **test_multiplex_notifier_ping_aggregates()** (6 connections) — `tests/homesec/test_notifiers.py`
- **test_multiplex_notifier_shutdown_is_resilient()** (6 connections) — `tests/homesec/test_notifiers.py`
- **test_sendgrid_templates_render()** (6 connections) — `tests/homesec/test_notifiers.py`
- **test_send_constructs_correct_api_request()** (6 connections) — `tests/homesec/test_sendgrid_notifier.py`
- **test_send_handles_none_analysis()** (6 connections) — `tests/homesec/test_sendgrid_notifier.py`
- **test_send_html_only_content()** (6 connections) — `tests/homesec/test_sendgrid_notifier.py`
- **test_send_includes_cc_and_bcc()** (6 connections) — `tests/homesec/test_sendgrid_notifier.py`
- **test_send_includes_sender_name()** (6 connections) — `tests/homesec/test_sendgrid_notifier.py`
- **test_send_raises_on_api_error()** (6 connections) — `tests/homesec/test_sendgrid_notifier.py`
- **test_send_text_only_content()** (6 connections) — `tests/homesec/test_sendgrid_notifier.py`
- **test_shutdown_is_idempotent()** (6 connections) — `tests/homesec/test_sendgrid_notifier.py`
- *... and 54 more nodes in this community*

## Relationships

- No strong cross-community connections detected

## Source Files

- `src/homesec/models/vlm.py`
- `src/homesec/plugins/notifiers/sendgrid_email.py`
- `tests/homesec/test_notifiers.py`
- `tests/homesec/test_sendgrid_notifier.py`

## Audit Trail

- EXTRACTED: 298 (74%)
- INFERRED: 106 (26%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*