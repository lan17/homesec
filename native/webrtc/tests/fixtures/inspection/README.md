# Synthetic inspection compatibility fixtures

These contain only the existing synthetic test pattern and sine tone from
`../recording/h264-aac-bframes.mp4`; no private camera media is included. Each is
two seconds at 160x120. They are committed so native tests do not launch media
CLIs to create their own input.

The native library tests execute MP4 copy for AAC, AC3, ALAC, EAC3, MP2 and MP3;
AAC conversion for G.711 A-law/u-law and Opus; H.264, HEVC, MPEG-4 Part 2 and MJPEG
metadata/decode/copy validation; silent trials; actual write errors; and timestamp
signals. The executable tests use localhost RTSP/TCP fake cameras, including
HEVC, no-IDR H.264, stalled input, explicit 453 and 401 refusals, independent
same-URL roles and a resetting packet clock. Hardware and physical-camera QA are
separate evidence.

Regenerate with a fixture/oracle FFmpeg installation, outside inspection:

```sh
# Repeat for ac3, alac, eac3, mp2 and mp3 (stereo), or pcm_alaw / pcm_mulaw (mono).
ffmpeg -i ../recording/h264-aac-bframes.mp4 -map 0:v:0 -map 0:a:0 \
  -c:v copy -c:a ac3 -threads:a 1 -ac 2 h264-ac3.mov
ffmpeg -i ../recording/h264-aac-bframes.mp4 -c:v copy \
  -c:a libopus -threads:a 1 h264-libopus.mp4
ffmpeg -i ../recording/h264-aac-bframes.mp4 -c:v libx265 -threads:v 1 \
  -x265-params log-level=error:pools=none:frame-threads=1 -c:a copy hevc-aac.mp4
ffmpeg -i hevc-aac.mp4 -an -c:v copy -f hevc hevc.h265
# Repeat with -c:v mjpeg.
ffmpeg -i ../recording/h264-aac-bframes.mp4 -an -c:v mpeg4 \
  -threads:v 1 -q:v 5 mpeg4.mov
```

`h264-eac3.ts` adds standard MPEG-TS container framing to the existing synthetic
H.264/E-AC-3 MOV, without re-encoding. The fake RTSP camera carries complete TS
packets in standard RTP payload 33 (MP2T). This exercises E-AC-3 through an actual
supported container transport, without incorrectly labelling it AC-3 RTP.
Generated outside the scoped inspector/helper execution with:

```sh
ffmpeg -nostdin -hide_banner -loglevel error \
  -i h264-eac3.mov -map 0:v:0 -map 0:a:0 -c copy -f mpegts h264-eac3.ts
```

The G.711 RTP fixtures publish independent mono A-law/u-law sample bytes at the
standard 8 kHz PCMA/PCMU RTP clock. Opus uses its 48 kHz RTP clock. Their helper
cases exercise actual AAC conversion; per-codec copy helper cases cover AAC,
AC-3, MP2, MP3, and E-AC-3 via MP2T. ALAC retains direct pinned-library copy
coverage; no unverified ALAC RTP framing is invented.
