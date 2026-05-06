# Push-to-Talk Deployment Notes

HomeSec push-to-talk is a half-duplex browser microphone to camera speaker path for
RTSP cameras that expose an ONVIF RTSP audio backchannel. The MVP stays inside
HomeSec: browser PCM frames are authenticated by the API, bridged to the runtime
worker, encoded to G.711 µ-law (`PCMU/8000`), packetized as RTP, and sent to the
camera over RTSP-over-TCP interleaved RTP.

This feature is opt-in at both the global and per-camera levels. Enable it only
for cameras you have tested locally.

## Supported MVP Contract

| Area | MVP support |
| --- | --- |
| UI mode | Push and hold, half-duplex talk |
| Browser input | Mono PCM S16LE frames over WebSocket |
| Default input format | `pcm_s16le`, `16000 Hz`, `1` channel, `20 ms` frames |
| Camera codec | `PCMU/8000` only |
| Camera backend | `onvif_rtsp_backchannel` |
| Transport | RTSP over TCP with interleaved RTP |
| Concurrency | One reserved or active talk session per camera |
| Session limits | Bounded by `talk.max_session_s` and `talk.idle_timeout_s` |
| Persistence | Microphone audio is not recorded or intentionally persisted |

References:

- ONVIF Streaming Specification, audio backchannel setup: <https://www.onvif.org/specs/stream/ONVIF-Streaming-Spec.pdf>
- RTSP 1.0: <https://www.rfc-editor.org/rfc/rfc2326>
- RTP: <https://www.rfc-editor.org/rfc/rfc3550>
- G.711 PCMU payload naming: <https://www.rfc-editor.org/rfc/rfc3551>

## Configuration

Enable the global feature first:

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

Then enable talk on each compatible camera:

```yaml
cameras:
  - name: front_door
    source:
      backend: rtsp
      config:
        rtsp_url_env: FRONT_DOOR_RTSP_URL
        output_dir: ./recordings
    talk:
      enabled: true
      backend: onvif_rtsp_backchannel
      config:
        rtsp_url_env: FRONT_DOOR_TALK_RTSP_URL
        # Optional if credentials are not embedded in the RTSP URL:
        username_env: FRONT_DOOR_RTSP_USERNAME
        password_env: FRONT_DOOR_RTSP_PASSWORD
        preferred_codecs:
          - PCMU/8000
        transport: rtsp_tcp_interleaved
        connect_timeout_s: 5.0
        io_timeout_s: 5.0
```

Use environment variables for RTSP URLs and credentials. Do not commit camera
passwords in YAML.

Talk can use the same RTSP URL as the camera source when the camera advertises
backchannel support on that URI. Some cameras expose talk on a different profile
or URI; configure `cameras[].talk.config.rtsp_url_env` separately when needed.

## Security and Privacy Notes

- Treat talk as live microphone access. Browser permission prompts are expected.
- Keep `server.auth_enabled: true` for any deployment reachable outside a trusted
  local network. The REST session endpoint issues short-lived, camera-scoped talk
  tokens for WebSocket use when API auth is enabled.
- Do not log microphone frames, decoded PCM, RTP payloads, or RTSP credentials.
  Existing protocol logging must keep credentials redacted.
- Talk sessions are transient. The MVP does not intentionally store microphone
  audio in clips, preview storage, logs, or databases.
- Prefer HTTPS or a trusted reverse proxy for browser access so API keys and talk
  tokens are not exposed on the network.
- Enable talk per camera, not globally for every camera. Leave `cameras[].talk.enabled`
  false for cameras that do not need speaker access.
- Use the shortest practical `max_session_s`; the default `60` seconds prevents a
  stuck client from keeping a speaker session open indefinitely.
- Preview audio is muted while browser talk is active to reduce feedback. Browser
  echo cancellation can help, but the MVP is not full-duplex intercom.

## Operational Checklist

Before enabling talk for a camera:

1. Confirm the camera has a speaker and ONVIF/RTSP audio backchannel support.
2. Confirm the camera advertises `PCMU/8000` in the backchannel SDP.
3. Confirm RTSP credentials work for the talk URI.
4. Keep the camera and HomeSec on a low-latency local network where possible.
5. Start with one camera and one browser client.
6. Test while recording and preview are active; some cameras have limited RTSP
   session budgets.
7. Watch HomeSec logs for `unsupported_codec`, `unsupported_camera`,
   `session_budget_exhausted`, `camera_backchannel_failed`, or repeated idle
   timeouts.
8. Record the result in the compatibility matrix below.

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
| Fake RTSP backchannel test server | N/A | Test fixture | `PCMU/8000` | N/A | Pass | Validates HomeSec protocol path without hardware. |
| _Add tested camera_ |  |  |  |  | Untested |  |

Suggested result values:

- `Pass`: browser push-to-talk audio plays through the camera speaker.
- `Partial`: session opens but audio quality, latency, or recording interaction is
  not acceptable yet.
- `Fail`: camera rejects setup, lacks `PCMU/8000`, or cannot play audio.
- `Untested`: no real-camera verification yet.

## Troubleshooting

### Talk control is hidden or disabled

- Confirm global `talk.enabled: true`.
- Confirm `cameras[].talk.enabled: true` for the selected camera.
- Confirm the camera source is RTSP; non-RTSP sources are not talk-capable in the MVP.
- Check `GET /api/v1/talk/cameras/{camera_name}` for `state`, `supported_codecs`,
  and `last_error`.

### Browser microphone permission is denied

- Re-enable microphone permission in the browser site settings.
- Use HTTPS or localhost; browsers may restrict microphone capture on insecure
  origins.
- Close other applications that have exclusive microphone access.

### Session is refused with `talk_disabled`

- Enable both the global `talk.enabled` and per-camera `cameras[].talk.enabled`.
- Reload/restart HomeSec after config changes if your deployment does not hot-reload
  camera config.

### Session is refused with `source_not_talk_capable`

- The camera source backend does not expose the talk capability. The MVP supports
  RTSP sources with `onvif_rtsp_backchannel` talk config.

### Session is refused with `unsupported_camera` or `camera_backchannel_failed`

- Verify the talk RTSP URI/profile supports ONVIF audio backchannel.
- Try the camera's primary ONVIF media profile and any vendor-documented talk URI.
- Confirm the camera allows RTSP-over-TCP interleaved RTP.
- Check whether another client has already consumed the camera's backchannel or
  media-session budget.

### Session is refused with `unsupported_codec`

- The MVP supports `PCMU/8000` only. If the camera advertises only `PCMA`, `G726`,
  AAC, or a vendor-specific codec, leave talk disabled for that camera until that
  codec is implemented and tested.

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

## Review Checklist

Use this checklist for PRs that touch push-to-talk:

- [ ] API routes do not import ONVIF, RTSP, SDP, RTP, or G.711 implementation modules.
- [ ] UI code sends browser PCM only and does not know camera protocol details.
- [ ] Runtime/source code still enforces one reserved or active session per camera.
- [ ] Talk sessions remain bounded by `max_session_s` and cleaned up on WebSocket
      close, cancellation, open failure, and stop.
- [ ] Audio frames, credentials, API keys, and talk tokens are not logged or persisted.
- [ ] New cross-boundary payloads use typed Pydantic/API models rather than raw dicts.
- [ ] Codec or transport changes include fake-server coverage and at least one
      compatibility-matrix update.
- [ ] UI changes keep preview muted while talk is active and handle microphone denial.
- [ ] Browser audio changes preserve exact PCM frame sizes and backpressure behavior.
- [ ] Docs and `config/example.yaml` stay in sync with any contract changes.
