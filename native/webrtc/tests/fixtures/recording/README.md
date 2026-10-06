# Synthetic compressed recording fixture

`h264-aac-bframes.mp4` is a two-second, 160x120 synthetic test pattern plus a
440 Hz tone. It contains no camera footage. It exercises H.264 packet copying,
distinct presentation/decode timestamps, negative initial decode timestamps,
AAC codec configuration, audio/video synchronization, and MP4 finalization.

Generated using FFmpeg 8.1.2:

```sh
ffmpeg -hide_banner -loglevel error \
  -f lavfi -i testsrc2=size=160x120:rate=12 \
  -f lavfi -i sine=frequency=440:sample_rate=48000 \
  -t 2 -c:v libx264 -threads:v 1 -preset veryfast -pix_fmt yuv420p \
  -g 12 -bf 2 -c:a aac -threads:a 1 -ac 2 -b:a 32k \
  -movflags +faststart h264-aac-bframes.mp4
```

The test corpus is committed, rather than recreated during test runs. The
installed FFmpeg CLI independently verifies decoded video and audio; the
production muxer uses the pinned, statically linked libraries.

The `H264_AAC` fake-camera mode replays this fixture's raw AAC access units over
MPEG4-GENERIC RTP with 16-bit AU headers, alongside the baseline H.264 corpus.
Its two TCP-interleaved tracks share an RTCP NTP epoch, with audio's initial RTP
presentation time offset by 250 ms. The source test checks packet clocks and
payloads, exact MP4 track offsets, and independently decoded audio/video samples.
