`baseline-160x120.h264` is synthetic H.264 camera media (ten 160x120 frames at
10 fps, constrained baseline, level 3.1, no B-frames), generated with:

```sh
ffmpeg -hide_banner -loglevel error -f lavfi -i testsrc2=size=160x120:rate=10 \
  -frames:v 10 -c:v libx264 -threads 1 -preset ultrafast -tune zerolatency \
  -profile:v baseline -level:v 3.1 -pix_fmt yuv420p -g 10 -bf 0 \
  -x264-params aud=1:repeat-headers=1 -f h264 baseline-160x120.h264
```

The fake RTSP server removes SPS/PPS from RTP and advertises them in SDP. It
packetizes the remaining access units with payload type 101, including FU-A
fragments, over TCP interleaving. This verifies native RTSP parameter handling
and the helper's encrypted WebRTC output against real encoded video.

`baseline-multislice-160x120.h264` uses the same command with `-frames:v 3`,
`-g 3`, and `-x264-params aud=1:repeat-headers=1:slices=2`. Each picture has
two slices. The partial-IDR fixture omits the first IDR slice only on startup,
then loops the intact GOP to verify recovery at the next complete keyframe.
