# Motion characterization corpus

These synthetic fixtures capture the current grayscale `MotionDetector` behavior
in `src/homesec/sources/rtsp/motion.py`. They can be replayed by Python and a future
Rust implementation without decoding video or opening a camera. No real camera
frames are included.

`corpus.json` contains detector settings and ordered operations. Each case starts
with a fresh detector. A `frame` operation supplies an optional percentage
threshold override and records exact blurred pixels, changed pixel count,
changed percentage, and the motion decision. A `reset` operation clears the
baseline and metrics before the next frame. JSON `null` for `threshold` means use
the configured `min_changed_pct`.

`frames.gray` is a concatenation of tightly packed, row-major unsigned 8-bit
grayscale arrays, with no header or row padding. Both `input_offset` and
`blurred_offset` are byte offsets into this file; read exactly `width * height`
bytes. Identical arrays share offsets. The binary format and JSON manifest have
no Python-specific serialization.

The corpus covers:

- First-frame initialization, strict pixel thresholds, inclusive percentage
  thresholds, recording-style sensitivity overrides, zero and negative overrides,
  reset, and comparison to the immediately preceding frame.
- Pixel thresholds at and above 255, zero pixel threshold, and percentages above 100.
- Blur kernels 0, 1, 3, 5, 7, 9, 11, and 31; impulses, edges, checkerboards, and
  deterministic arithmetic noise; thin/tiny images and kernels larger than an image.
- Custom kernels 255 and 1001 on a tiny image, plus the production 320x240 shape
  with default kernel 5, pixel threshold 45, and changed percentage 1.0.

The source normalizes positive even kernels to the next odd size before creating
the detector. These fixtures exercise the resulting odd kernels directly.
Kernel 0 and 1 skip blur. Larger kernels use OpenCV `GaussianBlur` with sigma 0
and the default `BORDER_REFLECT_101` border. Exact bytes include OpenCV's coefficient
quantization and rounding; using the same Gaussian formula alone may differ.
The reference OpenCV version is recorded in the manifest.

From the repository root, regenerate or check reproducibility with locked dependencies:

```sh
PYTHONPATH=src uv run --locked python native/webrtc/tests/fixtures/motion/generate.py
PYTHONPATH=src uv run --locked python native/webrtc/tests/fixtures/motion/generate.py --check
uv run --locked pytest tests/homesec/rtsp/test_motion.py
```

Replay tests read the committed expectations; they never regenerate them. Review
any changed expectations when updating the detector or OpenCV rather than
automatically accepting regenerated output. `--check` also checks reference
version metadata. The generator uses deterministic arithmetic patterns, not an
RNG, wall-clock time, or platform-dependent image files.

This is bounded characterization, not exhaustive parity for every accepted
configuration. In particular, current configuration has no maximum blur kernel.
The corpus does not choose a new limit or cover BGR conversion, decoding, resizing,
color-range conversion, stream lifecycle, or native runtime integration.
