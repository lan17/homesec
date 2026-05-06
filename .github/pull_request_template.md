## Summary

-

## Validation

-

## Media feature checklist

If this PR touches preview, push-to-talk, camera media sessions, RTSP, RTP, SDP,
codecs, browser microphone/audio capture, or media auth tokens:

- [ ] API routes do not import camera protocol implementation modules directly.
- [ ] UI code does not know camera protocol details.
- [ ] Media credentials, API keys, tokens, and raw audio/video frames are not logged.
- [ ] Session lifecycles are bounded and clean up on stop, disconnect, failure, and cancellation.
- [ ] Concurrency/session-budget behavior is explicit and covered by tests or docs.
- [ ] Operator docs or troubleshooting notes were updated when behavior changed.

For push-to-talk-specific changes, also check `docs/push-to-talk.md#review-checklist`.
