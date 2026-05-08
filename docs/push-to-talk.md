# Push-to-Talk Deployment Notes

HomeSec push-to-talk is a half-duplex browser microphone to camera speaker path.
The bundled backend supports RTSP cameras that expose an ONVIF RTSP audio
backchannel. The protocol path stays inside HomeSec: browser PCM frames are
authenticated by the API when API auth is enabled, bridged to the runtime worker,
encoded to G.711 (`PCMU/8000` or `PCMA/8000`), packetized as RTP, and sent to the
camera over RTSP-over-TCP interleaved RTP. When API auth is disabled, talk
endpoints follow the same trusted-LAN access model as the rest of the control
API.

Talk policy defaults to enabled/auto. HomeSec probes RTSP cameras for ONVIF
backchannel support at runtime and only enables the UI when the camera advertises
a supported backchannel. Operators can opt out globally with `talk.enabled: false`
or per camera with `cameras[].talk.mode: disabled`.

## Supported MVP Contract

| Area | MVP support |
| --- | --- |
| UI mode | Push and hold, half-duplex talk |
| Browser input | Mono PCM S16LE frames over WebSocket |
| Default input format | `pcm_s16le`, `16000 Hz`, `1` channel, `20 ms` frames |
| Camera codec | `PCMU/8000` and `PCMA/8000` |
| Camera backend | Backend adapter abstraction; bundled backend is `onvif_rtsp_backchannel` |
| Transport | RTSP over TCP with interleaved RTP |
| Concurrency | One reserved or active talk session per camera |
| Session limits | Bounded by `talk.max_session_s` and `talk.idle_timeout_s` |
| Persistence | Microphone audio is not recorded or intentionally persisted |

References:

- ONVIF Streaming Specification, audio backchannel setup: <https://www.onvif.org/specs/stream/ONVIF-Streaming-Spec.pdf>
- RTSP 1.0: <https://www.rfc-editor.org/rfc/rfc2326>
- RTP: <https://www.rfc-editor.org/rfc/rfc3550>
- G.711 PCMU/PCMA payload naming: <https://www.rfc-editor.org/rfc/rfc3551>

## Configuration

The global feature is enabled by default. Configure it explicitly when you need
different token/session limits or want to disable talk everywhere:

```yaml
talk:
  enabled: true
  token_ttl_s: 30
  max_session_s: 60
  idle_timeout_s: 2.0
  input:
    codec: pcm_s16le
    sample_rate: 16000
    channels: 1
    frame_ms: 20
```

RTSP cameras default to `mode: auto`, which probes for ONVIF talk support. Use an
explicit camera block when you need a dedicated talk URI/profile, credentials, or
codec preference order:

```yaml
cameras:
  - name: front_door
    source:
      backend: rtsp
      config:
        rtsp_url_env: FRONT_DOOR_RTSP_URL
        output_dir: ./recordings
    talk:
      mode: auto
      backend: onvif_rtsp_backchannel
      config:
        rtsp_url_env: FRONT_DOOR_TALK_RTSP_URL
        # Optional if credentials are not embedded in the RTSP URL:
        username_env: FRONT_DOOR_RTSP_USERNAME
        password_env: FRONT_DOOR_RTSP_PASSWORD
        preferred_codecs:
          - PCMU/8000
          - PCMA/8000
        transport: rtsp_tcp_interleaved
        connect_timeout_s: 5.0
        io_timeout_s: 5.0
```

If the camera should use the default source RTSP URL and default codec order, omit
the camera `talk` block entirely:

```yaml
cameras:
  - name: front_door
    source:
      backend: rtsp
      config:
        rtsp_url_env: FRONT_DOOR_RTSP_URL
    # talk omitted: defaults to mode=auto, backend=auto
```

The top-level `cameras[].talk.config` field remains a compatibility alias for
the ONVIF RTSP backchannel backend. New backend-specific options can also be
placed under `cameras[].talk.backends.<backend_name>`:

```yaml
cameras:
  - name: front_door
    talk:
      mode: auto
      backend: auto
      backends:
        onvif_rtsp_backchannel:
          rtsp_url_env: FRONT_DOOR_TALK_RTSP_URL
          preferred_codecs:
            - PCMU/8000
            - PCMA/8000
```

Use environment variables for RTSP URLs and credentials. Do not commit camera
passwords in YAML.

Talk can use the same RTSP URL as the camera source when the camera advertises
backchannel support on that URI. Some cameras expose talk on a different profile
or URI; configure `cameras[].talk.config.rtsp_url_env` separately when needed.
If an explicit talk URL env var is configured, it must be set; HomeSec does not
fall back to the main stream URL for a missing talk override.
If explicit talk credential env vars are configured, they must also be set;
HomeSec reports a talk config error rather than breaking the RTSP source when
those env vars are missing.

Opt out cameras that should never expose speaker access:

```yaml
cameras:
  - name: side_yard
    talk:
      mode: disabled
```

## Backend Selection

Talk policy and backend selection are separate:

```plain text
cameras[].talk.mode: disabled
  disables talk for that camera; no backend probes run

cameras[].talk.mode: auto
  allows runtime capability discovery

cameras[].talk.backend: auto
  probes standards-based backends first
  ONVIF wins when it is supported
  proprietary candidates are considered only after standards-based probing fails
  proprietary candidates must be explicitly configured or safely fingerprinted

cameras[].talk.backend: <name>
  selects that backend only
  no fallback runs
  unknown registered-safe names produce talk_config_error until a backend is registered
```

The bundled registry currently includes `onvif_rtsp_backchannel`. It is marked as
standards-based, so auto mode tries it before any future proprietary backend.
Future proprietary detectors must return `safe_to_probe: true` before HomeSec
will probe them automatically. A safe detector should be based on local camera
identity, explicit backend config, or another low-risk local signal. It must not
send credentials to an unknown local service as part of detection.

## Backend-Specific Configuration

Camera talk config has two backend configuration forms:

```yaml
# Compatibility alias for ONVIF when backend is auto or onvif_rtsp_backchannel.
talk:
  mode: auto
  backend: auto
  config:
    preferred_codecs:
      - PCMU/8000
      - PCMA/8000
```

```yaml
# Preferred form for backend-specific config.
talk:
  mode: auto
  backend: auto
  backends:
    onvif_rtsp_backchannel:
      preferred_codecs:
        - PCMU/8000
        - PCMA/8000
```

For explicit backend selection, HomeSec only builds the named backend:

```yaml
cameras:
  - name: front_door
    talk:
      mode: auto
      backend: onvif_rtsp_backchannel
      backends:
        onvif_rtsp_backchannel:
          preferred_codecs:
            - PCMU/8000
            - PCMA/8000
```

Explicit future backend names are accepted by config parsing so operators can
stage configuration, but backend names must be safe identifiers: lowercase
letters, numbers, and underscores, starting with a letter, with at most 64
characters. Unregistered safe backend names report a talk backend
selection/config error until a matching backend adapter is implemented and
registered.

## Future Proprietary Backends

Phase 9 provides the backend adapter abstraction, not Tapo support. The example
below documents the intended future shape for Phase 10 only. It will not work
until a `tapo_local` backend is implemented and registered.

```yaml
# Future Phase 10 example only. Not supported until tapo_local lands.
cameras:
  - name: office
    talk:
      mode: auto
      backend: auto
      backends:
        tapo_local:
          host: 192.168.1.33
          port: 8800
          username_env: OFFICE_TAPO_USERNAME
          password_sha256_env: OFFICE_TAPO_SHA256
```

Future proprietary backends should implement the same contracts as the bundled
ONVIF backend:

- register a `TalkBackendRegistration` with a strict Pydantic config model;
- provide a safe detector when auto selection is appropriate;
- implement `probe()` for capability, codec, and auth diagnostics;
- implement `open_session()` and return a `TalkBackendSession`;
- write browser PCM frames through `TalkSession.write_pcm_frame()`;
- expose selected backend and selected codec diagnostics without leaking secrets.

Phase 10 Tapo work should add only the Tapo-local backend pieces: config model,
safe detector, local auth/client code, local talk setup, packet writer/session
implementation, fake endpoint tests, and Tapo-specific docs. It should not need
to modify the API WebSocket protocol, runtime IPC, TalkManager lifecycle, or UI
PCM capture path.

## Status Diagnostics

`GET /api/v1/talk/cameras/{camera_name}` returns the current talk policy,
capability, selected backend, codec negotiation, and session state.

Backend diagnostics are intentionally minimal:

- `backend`: selected talk backend name once selection has completed, or the
  explicit backend name when a configured backend is invalid or unavailable.
- `backend_reason`: safe human-readable text explaining backend selection or why
  no backend was selected.

Backend names are constrained to safe identifiers before they can appear in API
or WebSocket diagnostics. Do not put URLs, credentials, tokens, or raw protocol
payloads in `cameras[].talk.backend` or `cameras[].talk.backends` keys.
`backend_reason` must not include RTSP URLs, camera credentials, password
hashes, API keys, stream tokens, raw auth headers, or raw SDP.
Backend config validation failures expose a stable public `last_error` rather
than raw validator text, because validator text can include rejected input.
Missing explicit talk URL or credential environment variables are reported as
`capability=config_error` / `reason=talk_config_error`; the message may include
the missing environment variable name, but never its value.

## Security and Privacy Notes

- Treat talk as live microphone access. Browser permission prompts are expected.
- Keep `server.auth_enabled: true` for any deployment reachable outside a trusted
  local network. The REST session endpoint issues short-lived, camera-scoped talk
  tokens for WebSocket use when API auth is enabled.
- Do not log microphone frames, decoded PCM, RTP payloads, or RTSP credentials.
  Existing protocol logging must keep credentials redacted.
- Backend credentials should use environment variables. Missing env vars may be
  named in public config errors, but values must never be logged or returned.
- Proprietary backend diagnostics must not expose camera credentials, RTSP URLs
  with credentials, tokens, password hashes, auth headers, or raw protocol
  payloads.
- Proprietary backend probes must stay local and must not make vendor cloud calls
  unless a future feature explicitly opts into that behavior. Phase 9 and the
  planned Tapo-local path are local-only.
- Talk sessions are transient. The MVP does not intentionally store microphone
  audio in clips, preview storage, logs, or databases.
- Prefer HTTPS or a trusted reverse proxy for browser access so API keys and talk
  tokens are not exposed on the network.
- Do not rely on global defaults as proof that a camera is safe to use. HomeSec
  discovers protocol support, but operators should still opt out cameras that do
  not need speaker access with `cameras[].talk.mode: disabled`.
- Use the shortest practical `max_session_s`; the default `60` seconds prevents a
  stuck client from keeping a speaker session open indefinitely.
- Preview audio is muted while browser talk is active to reduce feedback. Browser
  echo cancellation can help, but the MVP is not full-duplex intercom.

## Operational Checklist

Before relying on talk for a camera:

1. Confirm the camera has a speaker and ONVIF/RTSP audio backchannel support.
2. Confirm the camera advertises `PCMU/8000` or `PCMA/8000` in the backchannel SDP.
3. Confirm RTSP credentials work for the talk URI.
4. Keep the camera and HomeSec on a low-latency local network where possible.
5. Start with one camera and one browser client.
6. Test while recording and preview are active; some cameras have limited RTSP
   session budgets.
7. Watch HomeSec logs for `unsupported_codec`, `unsupported_camera`,
   `talk_config_error`, `talk_auth_failed`, `runtime_unavailable`,
   `session_budget_exhausted`, `camera_backchannel_failed`, or repeated idle
   timeouts.
8. Record the result in the compatibility matrix below.

Before implementing a new proprietary backend, confirm Phase 9 already provides:

- backend registration;
- backend-specific config validation;
- safe detector hook;
- `probe()` hook;
- `open_session()` hook;
- `TalkSession.write_pcm_frame()` hook;
- config and auth error semantics;
- selected backend diagnostics;
- fake proprietary backend readiness tests.

During operation:

- Monitor camera CPU/load if talk causes video stream instability.
- Keep `max_session_s` and `idle_timeout_s` bounded.
- If users report choppy talk, prefer improving network quality before increasing
  buffering; push-to-talk needs low latency more than lossless delivery.
- If a camera becomes unstable after talk testing, disable per-camera talk first
  and restart the camera only if its own RTSP service is wedged.

## Real-Camera Compatibility Matrix

Use this table to track tested devices. Do not mark a model compatible until a
real camera plays browser microphone audio through HomeSec.

| Vendor / model | Firmware | Talk URI/profile | Advertised codec | Auth | Result | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| Fake RTSP backchannel test server | N/A | Test fixture | `PCMU/8000`, `PCMA/8000` | N/A | Pass | Validates HomeSec protocol path without hardware. |
| _Add tested camera_ |  |  |  |  | Untested |  |

Suggested result values:

- `Pass`: browser push-to-talk audio plays through the camera speaker.
- `Partial`: session opens but audio quality, latency, or recording interaction is
  not acceptable yet.
- `Fail`: camera rejects setup, lacks a supported G.711 backchannel codec, or cannot play audio.
- `Untested`: no real-camera verification yet.

## Troubleshooting

### Talk control is hidden or disabled

- Confirm global `talk.enabled` is not `false`.
- Confirm `cameras[].talk.mode` is not `disabled` for the selected camera.
- Confirm the camera source is RTSP; non-RTSP sources are not talk-capable in the MVP.
- Check `GET /api/v1/talk/cameras/{camera_name}` for `state`, `capability`,
  `backend`, `backend_reason`, `offered_codecs`, `selected_codec`, and
  `last_error`.

### Browser microphone permission is denied

- Re-enable microphone permission in the browser site settings.
- Use HTTPS or localhost; browsers may restrict microphone capture on insecure
  origins.
- Close other applications that have exclusive microphone access.

### Session is refused with `talk_disabled`

- Set global `talk.enabled: true` or omit it, and set per-camera
  `cameras[].talk.mode: auto` or omit the camera talk block.
- Reload/restart HomeSec after config changes if your deployment does not hot-reload
  camera config.

### Session is refused with `source_not_talk_capable`

- The camera source backend does not expose the talk capability. The MVP supports
  RTSP sources with `onvif_rtsp_backchannel` talk config.

### Session is refused with `talk_config_error`

- If `cameras[].talk.backend` names a future or custom backend, confirm that
  backend is installed and registered in this HomeSec runtime.
- If `cameras[].talk.config.rtsp_url_env` or credential env vars are configured,
  confirm those environment variables are set for the HomeSec process.
- Check `last_error` / `backend_reason` for the missing backend or environment
  variable name.

### Session is refused with `runtime_unavailable`

- Confirm the runtime process is healthy and recently heartbeating.
- Retry after a runtime reload if the process was restarted while the browser
  had a reserved talk session.

### Session is refused with `talk_auth_failed`

- HomeSec API authentication already succeeded; this means the selected camera
  backend rejected HomeSec's configured camera credentials.
- Verify the talk backend's RTSP or vendor credentials and confirm the camera
  account has speaker/backchannel permissions.

### Session is refused with `unsupported_camera` or `camera_backchannel_failed`

- Verify the talk RTSP URI/profile supports ONVIF audio backchannel.
- Try the camera's primary ONVIF media profile and any vendor-documented talk URI.
- Confirm the camera allows RTSP-over-TCP interleaved RTP.
- Check whether another client has already consumed the camera's backchannel or
  media-session budget.

### Session is refused with `unsupported_codec`

- The MVP supports `PCMU/8000` and `PCMA/8000`. If the camera advertises only
  `G726`, AAC, or a vendor-specific codec, leave talk disabled for that camera
  until that codec or backend is implemented and tested.

### Session is refused with `session_already_active`

- Another browser client has already reserved or opened talk for that camera.
- Release the talk button in the other client or wait for `max_session_s` /
  `idle_timeout_s` cleanup.

### Session is refused with `session_budget_exhausted`

- The camera likely has too few concurrent RTSP sessions for recording, preview,
  and talk together.
- Stop live preview and retry.
- Avoid changing recording behavior automatically; if the camera cannot support
  all sessions, decide explicitly which feature should yield.

### WebSocket closes with invalid audio frame errors

- Confirm the browser sends the same input format returned by the session create
  response.
- Expected default frame size is `640` bytes (`16000 Hz * 20 ms * 1 channel * 2`).
- Browser or UI changes that alter sample rate, channel count, or frame duration
  must update both the REST/WebSocket contract and tests.

### Talk starts but audio is delayed or choppy

- Keep browser and HomeSec close to the camera network path.
- Prefer wired or strong Wi-Fi for the camera.
- Check browser developer tools for WebSocket backpressure; the UI stops talk when
  queued audio exceeds roughly 500 ms.
- Avoid raising client-side buffering without validating perceived latency.
