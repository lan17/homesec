//! Exercise source ownership through actual RTSP, control pipes, MP4 and SRTP.

#[path = "support/rtsp_camera.rs"]
mod rtsp_camera;

use rtsp_camera::RtspCamera;
use serde_json::{Value, json};
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::net::UdpSocket;
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{self, Receiver};
use std::thread;
use std::time::{Duration, Instant};
use str0m::change::SdpAnswer;
use str0m::format::Codec;
use str0m::media::{Direction, MediaKind};
use str0m::net::{Protocol, Receive};
use str0m::rtp::rtcp::SenderInfo;
use str0m::{Candidate, Event, Input, Output, Rtc};

const CONTROL_WAIT: Duration = Duration::from_secs(8);

struct Helper {
    child: Child,
    stdin: Option<ChildStdin>,
    replies: Receiver<Value>,
    pending: HashMap<String, Value>,
    ready: Value,
    next_request: usize,
}

impl Helper {
    fn start(preview: bool) -> Self {
        let mut command = Command::new(env!("CARGO_BIN_EXE_homesec-webrtc"));
        command.arg("--shared");
        if preview {
            let reservation = UdpSocket::bind("127.0.0.1:0").unwrap();
            let port = reservation.local_addr().unwrap().port();
            drop(reservation);
            command.args([
                "--advertised-ip",
                "127.0.0.1",
                "--udp-port-start",
                &port.to_string(),
                "--udp-port-end",
                &port.saturating_add(15).to_string(),
                "--max-viewers",
                "2",
            ]);
        }
        let mut child = command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .unwrap();
        let stdin = child.stdin.take().unwrap();
        let output = child.stdout.take().unwrap();
        let (sender, replies) = mpsc::channel();
        thread::spawn(move || {
            for line in BufReader::new(output).lines() {
                let line = line.unwrap();
                let reply = serde_json::from_str(&line).expect("invalid helper JSON");
                if sender.send(reply).is_err() {
                    break;
                }
            }
        });
        let ready: Value = replies.recv_timeout(CONTROL_WAIT).unwrap();
        assert_eq!(ready["event"], "ready", "{ready}");
        Self {
            child,
            stdin: Some(stdin),
            replies,
            pending: HashMap::new(),
            ready,
            next_request: 0,
        }
    }

    fn send(&mut self, mut request: Value) -> String {
        self.next_request += 1;
        let id = self.next_request.to_string();
        request["request_id"] = json!(id);
        let stdin = self.stdin.as_mut().unwrap();
        serde_json::to_writer(&mut *stdin, &request).unwrap();
        stdin.write_all(b"\n").unwrap();
        stdin.flush().unwrap();
        id
    }

    fn receive(&mut self, id: &str, wait: Duration) -> Value {
        if let Some(reply) = self.pending.remove(id) {
            return reply;
        }
        let deadline = Instant::now() + wait;
        loop {
            let reply = self
                .replies
                .recv_timeout(deadline.saturating_duration_since(Instant::now()))
                .expect("helper control request timed out");
            let Some(reply_id) = reply["request_id"].as_str().filter(|id| !id.is_empty()) else {
                assert_eq!(
                    reply["event"], "frame",
                    "unexpected unsolicited event: {reply}"
                );
                continue;
            };
            if reply_id == id {
                return reply;
            }
            self.pending.insert(reply_id.to_owned(), reply);
        }
    }

    fn request(&mut self, request: Value) -> Value {
        let id = self.send(request);
        self.receive(&id, CONTROL_WAIT)
    }

    fn success(&mut self, request: Value) -> Value {
        let reply = self.request(request);
        assert_eq!(reply["ok"], true, "{reply}");
        reply
    }

    fn start_motion(&mut self, camera: &RtspCamera, connect: f64) {
        self.success(json!({
            "command": "start_motion", "rtsp_url": camera.url,
            "motion_config": { "pixel_threshold": 25, "min_changed_pct": 0.1,
                "blur_kernel": 5, "recording_sensitivity_factor": 1.5 },
            "frame_queue_size": 8, "connect_timeout_s": connect, "io_timeout_s": 5.0,
        }));
    }

    fn start_recording(&mut self, camera: &RtspCamera, id: &str, path: &Path) -> Value {
        self.request(json!({ "command": "start_recording", "recording_id": id,
            "rtsp_url": camera.url, "output_path": path, "audio_mode": "none",
            "connect_timeout_s": 5.0, "io_timeout_s": 5.0 }))
    }

    fn recording_active(&mut self, id: &str) {
        let status = self.success(json!({"command": "recording_status", "recording_id": id}));
        assert_eq!(status["recording_active"], true, "{status}");
    }

    fn stop_recording(&mut self, id: &str) {
        let reply = self.success(json!({"command": "stop_recording", "recording_id": id}));
        assert_eq!(reply["recording_active"], false, "{reply}");
        assert_eq!(reply["recording_finalized"], true, "{reply}");
    }

    fn wait_exit(&mut self) {
        wait_until(Duration::from_secs(3), || {
            self.child.try_wait().unwrap().is_some()
        });
        assert!(self.child.wait().unwrap().success());
    }
}

impl Drop for Helper {
    fn drop(&mut self) {
        self.stdin.take();
        let deadline = Instant::now() + Duration::from_secs(2);
        while self.child.try_wait().ok().flatten().is_none() {
            if Instant::now() >= deadline {
                let _ = self.child.kill();
                let _ = self.child.wait();
                break;
            }
            thread::sleep(Duration::from_millis(5));
        }
    }
}

struct Viewer {
    rtc: Rtc,
    socket: UdpSocket,
    connected: bool,
    frames: usize,
    keyframes: usize,
    sample: Vec<u8>,
    audio_samples: Vec<Vec<u8>>,
    sender_reports: [Option<SenderInfo>; 2],
}

impl Viewer {
    fn attach(helper: &mut Helper, id: &str) -> Self {
        let mut rtc = Rtc::builder()
            .clear_codecs()
            .enable_h264(true)
            .enable_opus(true, false)
            .build(Instant::now());
        let socket = UdpSocket::bind("127.0.0.1:0").unwrap();
        socket.set_nonblocking(true).unwrap();
        rtc.add_local_candidate(Candidate::host(socket.local_addr().unwrap(), "udp").unwrap())
            .unwrap();
        let mut change = rtc.sdp_api();
        change.add_media(MediaKind::Video, Direction::RecvOnly, None, None, None);
        change.add_media(MediaKind::Audio, Direction::RecvOnly, None, None, None);
        let (offer, pending) = change.apply().unwrap();
        let reply = helper.success(json!({"command": "offer", "session_id": id,
            "sdp": offer.to_sdp_string(), "lease_seconds": 30.0}));
        let answer = SdpAnswer::from_sdp_string(reply["sdp"].as_str().unwrap()).unwrap();
        rtc.sdp_api().accept_answer(pending, answer).unwrap();
        Self {
            rtc,
            socket,
            connected: false,
            frames: 0,
            keyframes: 0,
            sample: Vec::new(),
            audio_samples: Vec::new(),
            sender_reports: [None, None],
        }
    }

    fn progress(&mut self) {
        self.rtc
            .handle_input(Input::Timeout(Instant::now()))
            .unwrap();
        let mut packet = [0; 65_536];
        loop {
            match self.socket.recv_from(&mut packet) {
                Ok((length, source)) => {
                    let input = Receive::new(
                        Protocol::Udp,
                        source,
                        self.socket.local_addr().unwrap(),
                        &packet[..length],
                    )
                    .unwrap();
                    self.rtc
                        .handle_input(Input::Receive(Instant::now(), input))
                        .unwrap();
                }
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => break,
                Err(error) => panic!("viewer UDP failed: {error}"),
            }
        }
        loop {
            match self.rtc.poll_output().unwrap() {
                Output::Transmit(packet) => {
                    self.socket
                        .send_to(&packet.contents, packet.destination)
                        .unwrap();
                }
                Output::Event(Event::Connected) => self.connected = true,
                Output::Event(Event::MediaData(data)) => {
                    let video = data.params.spec().codec == Codec::H264;
                    assert!(video || data.params.spec().codec == Codec::Opus);
                    // str0m can emit a startup report before this track has sent
                    // media. Its placeholder RTP zero is not a clock mapping.
                    if let Some(info) = data
                        .last_sender_info
                        .filter(|info| info.sender_packet_count > 0)
                    {
                        self.sender_reports[usize::from(!video)] = Some(info);
                    }
                    if video {
                        self.frames += 1;
                        self.keyframes += usize::from(data.is_keyframe());
                        if self.frames <= 3 {
                            self.sample.extend_from_slice(&data.data);
                        }
                    } else if self.audio_samples.len() < 8 {
                        self.audio_samples.push(data.data.to_vec());
                    }
                }
                Output::Timeout(_) => break,
                Output::Event(_) => {}
            }
        }
    }

    fn ready(&self) -> bool {
        self.connected && self.frames >= 3 && self.keyframes >= 1
    }

    fn assert_decodes(&self) {
        let mut sample = tempfile::NamedTempFile::new().unwrap();
        sample.write_all(&self.sample).unwrap();
        assert_decode(sample.path(), true);
    }
}

fn assert_opus_decodes(viewer: &Viewer) {
    use ffmpeg_next::{codec, ffi, frame};
    let mut parameters = codec::Parameters::new();
    // SAFETY: Own this fresh parameter object; set the negotiated browser codec.
    unsafe {
        let raw = &mut *parameters.as_mut_ptr();
        raw.codec_type = ffi::AVMediaType::AVMEDIA_TYPE_AUDIO;
        raw.codec_id = ffi::AVCodecID::AV_CODEC_ID_OPUS;
        raw.sample_rate = 48_000;
        ffi::av_channel_layout_default(&mut raw.ch_layout, 2);
    }
    let mut decoder = codec::Context::from_parameters(parameters)
        .unwrap()
        .decoder()
        .audio()
        .unwrap();
    let mut count = 0;
    for bytes in &viewer.audio_samples {
        decoder
            .send_packet(&ffmpeg_next::Packet::copy(bytes))
            .unwrap();
        let mut audio = frame::Audio::empty();
        while decoder.receive_frame(&mut audio).is_ok() {
            assert_eq!(audio.rate(), 48_000);
            assert_eq!(audio.channels(), 2);
            assert!(audio.samples() > 0);
            count += 1;
        }
    }
    assert!(count >= 3, "received Opus must actually decode");
}

fn progress_until(viewers: &mut [&mut Viewer], complete: impl Fn(&[&mut Viewer]) -> bool) {
    let deadline = Instant::now() + Duration::from_secs(8);
    loop {
        for viewer in viewers.iter_mut() {
            viewer.progress();
        }
        if complete(viewers) {
            return;
        }
        assert!(Instant::now() < deadline, "shared preview media timed out");
        thread::sleep(Duration::from_millis(2));
    }
}

fn wait_until(wait: Duration, mut ready: impl FnMut() -> bool) {
    let deadline = Instant::now() + wait;
    while !ready() {
        assert!(
            Instant::now() < deadline,
            "shared lifecycle operation timed out"
        );
        thread::sleep(Duration::from_millis(5));
    }
}

fn partial(path: &Path) -> PathBuf {
    PathBuf::from(format!("{}.partial", path.display()))
}

fn assert_decode(path: &Path, elementary: bool) {
    let mut command = Command::new("ffmpeg");
    command.args([
        "-hide_banner",
        "-loglevel",
        "error",
        "-xerror",
        "-filter_threads",
        "1",
        "-threads",
        "1",
    ]);
    if elementary {
        command.args(["-f", "h264"]);
    }
    let decoded = command
        .arg("-i")
        .arg(path)
        .args([
            "-map", "0:v:0", "-threads", "1", "-pix_fmt", "rgb24", "-vsync", "0", "-f", "rawvideo",
            "pipe:1",
        ])
        .output()
        .unwrap();
    assert!(
        decoded.status.success(),
        "synthetic output did not decode: {}",
        String::from_utf8_lossy(&decoded.stderr)
    );
    assert!(!decoded.stdout.is_empty());
    assert_eq!(decoded.stdout.len() % (160 * 120 * 3), 0);
}

#[test]
fn motion_recording_rotation_and_two_preview_viewers_share_one_camera_play() {
    // Given: One real RTSP camera feeds the shared helper's motion consumer.
    let camera = RtspCamera::start("H264");
    let directory = tempfile::tempdir().unwrap();
    let old = directory.path().join("old.mp4");
    let next = directory.path().join("next.mp4");
    let mut helper = Helper::start(true);
    helper.start_motion(&camera, 5.0);
    let observed =
        helper.success(json!({"command": "read_motion", "threshold": 0.1, "wait_timeout_s": 5.0}));
    assert_eq!(observed["frame_available"], true);
    assert_eq!(observed["observation"]["changed_pixels"], 0);
    let started = helper.start_recording(&camera, "old", &old);
    assert_eq!(started["ok"], true, "{started}");
    assert!(partial(&old).exists());
    assert!(!old.exists());

    // When: Two encrypted viewers join, then recording rotates with overlap.
    helper.success(json!({"command": "start", "rtsp_url": camera.url}));
    let mut first = Viewer::attach(&mut helper, "first");
    let mut second = Viewer::attach(&mut helper, "second");
    progress_until(&mut [&mut first, &mut second], |viewers| {
        viewers.iter().all(|v| v.ready())
    });
    first.assert_decodes();
    second.assert_decodes();
    let started = helper.start_recording(&camera, "next", &next);
    assert_eq!(started["ok"], true, "{started}");
    let refused = helper.start_recording(&camera, "third", &directory.path().join("third.mp4"));
    assert_eq!(refused["error_code"], "recording_budget_exceeded");
    assert!(!partial(&directory.path().join("third.mp4")).exists());

    // Then: Motion, preview, and both clip owners use one selected camera input.
    assert_eq!(camera.count("PLAY"), 1);
    helper.recording_active("old");
    helper.recording_active("next");

    // When: Motion stops independently, then preview releases both viewers.
    helper.success(json!({"command": "stop_motion"}));
    let frames = second.frames;
    progress_until(&mut [&mut first, &mut second], |v| {
        v[1].frames >= frames + 12
    });
    helper.success(json!({"command": "stop_preview"}));

    // Then: Both recording owners remain healthy and no camera teardown occurs.
    assert_eq!(
        helper.success(json!({"command": "status"}))["viewer_count"],
        0
    );
    helper.recording_active("old");
    helper.recording_active("next");
    assert_eq!(camera.count("DISCONNECTED"), 0);
    helper.stop_recording("old");
    assert_decode(&old, false);
    assert!(!partial(&old).exists());
    helper.recording_active("next");

    // When: The last recording consumer releases the selected stream.
    helper.stop_recording("next");

    // Then: Both clips are complete and the camera closes exactly once.
    assert_decode(&next, false);
    assert!(!partial(&next).exists());
    wait_until(Duration::from_secs(3), || camera.count("DISCONNECTED") == 1);
    assert_eq!(camera.count("PLAY"), 1);
    assert_eq!(camera.count("TEARDOWN"), 1);
    let stopped = helper.success(json!({"command": "stop_recording", "recording_id": "next"}));
    assert_eq!(stopped["recording_finalized"], true);
    helper.success(json!({"command": "stop"}));
    helper.wait_exit();
}

#[test]
fn pending_motion_read_does_not_delay_recording_stop_or_motion_cancellation() {
    // Given: A healthy recording and a second motion input stalled in DESCRIBE.
    let camera = RtspCamera::start("H264");
    let stalled = RtspCamera::start("STALL");
    let directory = tempfile::tempdir().unwrap();
    let clip = directory.path().join("clip.mp4");
    let mut helper = Helper::start(false);
    let started = helper.start_recording(&camera, "clip", &clip);
    assert_eq!(started["ok"], true, "{started}");
    helper.start_motion(&stalled, 30.0);
    let read =
        helper.send(json!({"command": "read_motion", "threshold": 0.1, "wait_timeout_s": 20.0}));
    thread::sleep(Duration::from_millis(1200));

    // When: Recording finalizes and motion stops while that read remains pending.
    let started = Instant::now();
    let stop = helper.send(json!({"command": "stop_recording", "recording_id": "clip"}));
    let reply = helper.receive(&stop, Duration::from_secs(2));
    assert_eq!(reply["ok"], true, "{reply}");
    assert_eq!(reply["recording_finalized"], true, "{reply}");
    helper.success(json!({"command": "stop_motion"}));

    // Then: Neither operation waits for the read deadline and cancellation replies.
    assert!(started.elapsed() < Duration::from_secs(3));
    assert_eq!(
        helper.receive(&read, Duration::from_secs(1))["error_code"],
        "motion_stopped"
    );
    assert_decode(&clip, false);
    wait_until(Duration::from_secs(3), || {
        stalled.count("DISCONNECTED") == 1
    });
    helper.success(json!({"command": "stop"}));
    helper.wait_exit();
}

#[test]
fn recording_start_acknowledges_an_idr_before_rotation_can_stop_the_old_clip() {
    // Given: A native recorder owns the selected input, whose GOP is one second.
    let camera = RtspCamera::start("H264");
    let directory = tempfile::tempdir().unwrap();
    let old = directory.path().join("old.mp4");
    let next = directory.path().join("next.mp4");
    let mut helper = Helper::start(false);
    assert_eq!(helper.start_recording(&camera, "old", &old)["ok"], true);
    thread::sleep(Duration::from_millis(250));

    // When: Rotation joins midway through the GOP, then immediately stops both clips.
    assert_eq!(helper.start_recording(&camera, "next", &next)["ok"], true);
    helper.stop_recording("old");
    helper.stop_recording("next");

    // Then: Startup success already guarantees a decodable IDR in the new clip.
    assert_decode(&old, false);
    assert_decode(&next, false);
    let packets = |path: &Path| {
        let mut input = ffmpeg_next::format::input(path).unwrap();
        input
            .packets()
            .map(|(_, packet)| (packet.is_key(), packet.data().unwrap().to_vec()))
            .collect::<Vec<_>>()
    };
    let old_packets = packets(&old);
    let new_packets = packets(&next);
    assert!(new_packets[0].0);
    assert!(old_packets.iter().any(|packet| packet == &new_packets[0]));
    assert_eq!(camera.count("PLAY"), 1);
    helper.success(json!({"command": "stop"}));
    helper.wait_exit();
}

#[test]
fn forced_helper_abort_leaves_only_an_unpublished_partial_clip() {
    // Given: An active recording writes only its private temporary filename.
    let camera = RtspCamera::start("H264");
    let directory = tempfile::tempdir().unwrap();
    let clip = directory.path().join("aborted.mp4");
    let mut helper = Helper::start(false);
    let started = helper.start_recording(&camera, "abort", &clip);
    assert_eq!(started["ok"], true, "{started}");
    assert!(partial(&clip).exists());

    // When: The owned helper process dies without a finalization handshake.
    helper.child.kill().unwrap();
    helper.child.wait().unwrap();

    // Then: A replay scanner cannot ingest a truncated file under the final name.
    assert!(!clip.exists());
    assert!(partial(&clip).exists());
    wait_until(Duration::from_secs(3), || camera.count("DISCONNECTED") == 1);
}

#[test]
fn failed_clip_publication_preserves_an_existing_file_and_the_other_recording() {
    // Given: Two recording owners share a camera while a final path is claimed.
    let camera = RtspCamera::start("H264");
    let directory = tempfile::tempdir().unwrap();
    let occupied = directory.path().join("occupied.mp4");
    let good = directory.path().join("good.mp4");
    let mut helper = Helper::start(false);
    for (id, path) in [("occupied", &occupied), ("good", &good)] {
        let reply = helper.start_recording(&camera, id, path);
        assert_eq!(reply["ok"], true, "{reply}");
    }
    thread::sleep(Duration::from_millis(1200));
    std::fs::write(&occupied, b"already owned by another writer").unwrap();

    // When: Finalization cannot atomically publish over that existing file.
    let reply = helper.request(json!({"command": "stop_recording", "recording_id": "occupied"}));

    // Then: Failure remains visible and the unrelated clip/source stay healthy.
    assert_eq!(reply["ok"], false, "{reply}");
    assert_eq!(reply["error_code"], "recording_publish_failed");
    assert_eq!(reply["recording_active"], false, "{reply}");
    assert_eq!(reply["recording_finalized"], false);
    assert_eq!(
        std::fs::read(&occupied).unwrap(),
        b"already owned by another writer"
    );
    assert!(partial(&occupied).exists());
    helper.recording_active("good");
    helper.stop_recording("good");
    assert_decode(&good, false);
    assert_eq!(camera.count("PLAY"), 1);
    wait_until(Duration::from_secs(3), || camera.count("DISCONNECTED") == 1);
    helper.success(json!({"command": "stop"}));
    helper.wait_exit();
}

#[test]
fn configured_preview_start_deadline_preserves_an_independent_recording() {
    // Given: A healthy clip owner and a second camera that stalls during DESCRIBE.
    let camera = RtspCamera::start("H264");
    let stalled = RtspCamera::start("STALL");
    let directory = tempfile::tempdir().unwrap();
    let clip = directory.path().join("surviving.mp4");
    let mut helper = Helper::start(true);
    let recording = helper.start_recording(&camera, "surviving", &clip);
    assert_eq!(recording["ok"], true, "{recording}");

    // When: Preview uses its configured 200 ms startup budget.
    let started = Instant::now();
    let request = helper.send(json!({"command": "start", "rtsp_url": stalled.url,
        "connect_timeout_s": 0.2, "io_timeout_s": 5.0}));
    let reply = helper.receive(&request, Duration::from_secs(2));

    // Then: That local timeout closes only the preview input within its budget.
    assert_eq!(reply["ok"], false, "{reply}");
    assert!(
        matches!(
            reply["error_code"].as_str(),
            Some("rtsp_timeout" | "source_timeout")
        ),
        "{reply}"
    );
    assert!(started.elapsed() < Duration::from_secs(1));
    assert_eq!(stalled.count("DESCRIBE"), 1);
    wait_until(Duration::from_secs(1), || {
        stalled.count("DISCONNECTED") == 1
    });
    helper.recording_active("surviving");
    assert_eq!(camera.count("DISCONNECTED"), 0);
    thread::sleep(Duration::from_millis(1200));
    helper.stop_recording("surviving");
    assert_decode(&clip, false);
    helper.success(json!({"command": "stop"}));
    helper.wait_exit();
}

#[test]
fn changing_codec_or_backwards_camera_clock_fails_only_preview() {
    // Given: A shared helper with healthy recording on an independent selected input.
    for mode in ["LEVEL_CHANGE", "CLOCK_BACKWARDS"] {
        let camera = RtspCamera::start(mode);
        let recording_camera = RtspCamera::start("H264");
        let directory = tempfile::tempdir().unwrap();
        let clip = directory.path().join("surviving.mp4");
        let mut helper = Helper::start(true);
        let reply = helper.start_recording(&recording_camera, "clip", &clip);
        assert_eq!(reply["ok"], true, "{reply}");
        helper.success(json!({"command": "start", "rtsp_url": camera.url}));
        wait_until(Duration::from_secs(2), || {
            helper.success(json!({"command": "status"}))["state"] == "ready"
        });

        // When: The camera changes its H.264 level or regresses its frame clock.
        wait_until(Duration::from_secs(5), || {
            helper.success(json!({"command": "status"}))["state"] == "error"
        });

        // Then: Preview fails closed while recording control and finalization remain healthy.
        assert_eq!(
            helper.success(json!({"command": "status"}))["media_active"],
            false,
            "{mode}"
        );
        helper.recording_active("clip");
        helper.stop_recording("clip");
        assert_decode(&clip, false);
        helper.success(json!({"command": "stop"}));
        helper.wait_exit();
    }
}

#[test]
fn shared_recording_copies_aac_from_the_same_motion_camera_input() {
    // Given: One synthetic H.264/AAC camera already serves native motion.
    let camera = RtspCamera::start("H264_AAC");
    let directory = tempfile::tempdir().unwrap();
    let clip = directory.path().join("video-audio.mp4");
    let mut helper = Helper::start(false);
    helper.start_motion(&camera, 5.0);
    helper.success(json!({"command": "read_motion", "threshold": 0.1, "wait_timeout_s": 5.0}));

    // When: Compressed recording joins, then motion detaches independently.
    helper.success(json!({"command": "start_recording", "recording_id": "clip",
        "rtsp_url": camera.url, "output_path": clip, "audio_mode": "copy",
        "connect_timeout_s": 5.0, "io_timeout_s": 5.0 }));
    thread::sleep(Duration::from_millis(1500));
    helper.success(json!({"command": "stop_motion"}));
    helper.recording_active("clip");
    thread::sleep(Duration::from_millis(300));
    helper.stop_recording("clip");

    // Then: Both tracks decode, AAC access units retain their original bytes, and only one PLAY occurred.
    assert_decode(&clip, false);
    let mut input = ffmpeg_next::format::input(&clip).unwrap();
    let audio = input
        .streams()
        .find(|stream| stream.parameters().medium() == ffmpeg_next::media::Type::Audio)
        .unwrap();
    assert_eq!(audio.parameters().id(), ffmpeg_next::codec::Id::AAC);
    let audio_index = audio.index();
    let source = rtsp_camera::aac_access_units();
    let packets: Vec<_> = input
        .packets()
        .filter(|(stream, _)| stream.index() == audio_index)
        .map(|(_, packet)| packet.data().unwrap().to_vec())
        .collect();
    assert!(!packets.is_empty());
    assert!(packets.iter().all(|packet| source.contains(packet)));
    let audio = Command::new("ffmpeg")
        .args(["-v", "error", "-xerror", "-i"])
        .arg(&clip)
        .args(["-map", "0:a:0", "-f", "s16le", "pipe:1"])
        .output()
        .unwrap();
    assert!(
        audio.status.success(),
        "{}",
        String::from_utf8_lossy(&audio.stderr)
    );
    assert!(!audio.stdout.is_empty());
    assert_eq!(camera.count("PLAY"), 1);
    assert_eq!(camera.count("TEARDOWN"), 1);
    helper.success(json!({"command": "stop"}));
    helper.wait_exit();
}

#[test]
fn damaged_idr_or_reset_camera_clock_cannot_publish_a_successful_clip() {
    // Given: A selected input records valid media before the camera damages a keyframe or resets its clock.
    for mode in ["DAMAGED_IDR", "CLOCK_RESET"] {
        let camera = RtspCamera::start(mode);
        let directory = tempfile::tempdir().unwrap();
        let clip = directory.path().join("rejected.mp4");
        let mut helper = Helper::start(false);
        assert_eq!(helper.start_recording(&camera, "clip", &clip)["ok"], true);

        // When: The damaged stream reaches the recorder after its initial IDR.
        wait_until(Duration::from_secs(5), || {
            let status =
                helper.request(json!({"command": "recording_status", "recording_id": "clip"}));
            status["ok"] == false && status["recording_active"] == false
        });
        let stopped = helper.request(json!({"command": "stop_recording", "recording_id": "clip"}));

        // Then: Failure is explicit, the writer is closed, and only its unpublished partial remains.
        assert_eq!(stopped["ok"], false, "{mode}: {stopped}");
        assert_eq!(stopped["recording_active"], false, "{mode}: {stopped}");
        assert_eq!(stopped["recording_finalized"], false, "{mode}: {stopped}");
        assert!(!clip.exists(), "{mode}");
        assert!(partial(&clip).exists(), "{mode}");
        helper.success(json!({"command": "stop"}));
        helper.wait_exit();
    }
}

#[test]
fn parent_eof_finalizes_owned_recording_and_releases_every_source_and_socket() {
    // Given: Shared motion and recording own one camera and a temporary clip.
    let camera = RtspCamera::start("H264");
    let directory = tempfile::tempdir().unwrap();
    let clip = directory.path().join("eof.mp4");
    let mut helper = Helper::start(false);
    helper.start_motion(&camera, 5.0);
    helper.success(json!({"command": "discard_frame", "wait_timeout_s": 5.0}));
    let started = helper.start_recording(&camera, "eof", &clip);
    assert_eq!(started["ok"], true, "{started}");
    thread::sleep(Duration::from_millis(1200));

    // When: The Python owner's actual control pipe closes.
    helper.stdin.take();
    helper.wait_exit();

    // Then: The complete owned clip publishes and all camera/socket leases close.
    assert!(clip.exists());
    assert!(!partial(&clip).exists());
    assert_decode(&clip, false);
    wait_until(Duration::from_secs(3), || camera.count("DISCONNECTED") == 1);
    assert_eq!(camera.count("PLAY"), 1);
    assert_eq!(camera.count("TEARDOWN"), 1);
    for key in ["media_port", "video_port", "audio_port"] {
        let port = helper.ready[key].as_u64().unwrap();
        assert!(UdpSocket::bind(format!("127.0.0.1:{port}")).is_ok());
    }
}

#[test]
fn native_preview_audio_and_transcoding_share_input_with_recording_and_two_viewers() {
    for video_codec in ["h264", "copy"] {
        // Given: One H.264/AAC camera serves native motion and copied recording.
        let camera = RtspCamera::start("H264_AAC");
        let directory = tempfile::tempdir().unwrap();
        let first_clip = directory.path().join("first.mp4");
        let next_clip = directory.path().join("next.mp4");
        let mut helper = Helper::start(true);
        assert_eq!(helper.ready["native_preview"], true);
        helper.start_motion(&camera, 5.0);
        helper.success(json!({"command": "read_motion", "threshold": 0.1, "wait_timeout_s": 5.0}));
        helper.success(json!({"command":"start_recording", "recording_id":"first", "rtsp_url":camera.url,
            "output_path":first_clip, "audio_mode":"copy", "connect_timeout_s":5.0,"io_timeout_s":5.0}));

        // When: Configured native preview converts audio, with shared two-viewer output.
        helper.success(json!({"command":"start", "rtsp_url":camera.url,
            "preview_settings":{"video_codec":video_codec,"audio_enabled":true},
            "connect_timeout_s":5.0,"io_timeout_s":5.0}));
        let mut first = Viewer::attach(&mut helper, "first-viewer");
        let mut second = Viewer::attach(&mut helper, "second-viewer");
        progress_until(&mut [&mut first, &mut second], |viewers| {
            viewers.iter().all(|v| {
                v.ready()
                    && v.audio_samples.len() >= 3
                    && v.sender_reports.iter().all(Option::is_some)
            })
        });

        // Then: Video and Opus decode, source media clocks align, and no extra PLAY occurs.
        for viewer in [&first, &second] {
            viewer.assert_decodes();
            assert_opus_decodes(viewer);
            let [video_report, audio_report] = viewer.sender_reports.map(Option::unwrap);
            let origin = |report: SenderInfo| {
                report
                    .ntp_time
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs_f64()
                    - report.rtp_time.as_seconds()
            };
            let clock_delta = origin(video_report) - origin(audio_report);
            assert!(
                clock_delta.abs() < 0.010,
                "source A/V timeline drifted by {clock_delta}s in {video_codec}; \
                 video SR={video_report:?}; audio SR={audio_report:?}"
            );
        }
        assert_eq!(camera.count("PLAY"), 1);
        helper.success(json!({"command":"start_recording", "recording_id":"next", "rtsp_url":camera.url,
            "output_path":next_clip, "audio_mode":"copy", "connect_timeout_s":5.0,"io_timeout_s":5.0}));
        helper.stop_recording("first");
        helper.recording_active("next");

        // When: Preview and motion stop, then preview restarts on the still-active input.
        helper.success(json!({"command":"stop_preview"}));
        helper.success(json!({"command":"stop_motion"}));
        helper.recording_active("next");
        helper.success(json!({"command":"start", "rtsp_url":camera.url,
            "preview_settings":{"video_codec":video_codec,"audio_enabled":true},
            "connect_timeout_s":5.0,"io_timeout_s":5.0}));
        let mut late = Viewer::attach(&mut helper, "late-viewer");
        progress_until(&mut [&mut late], |viewers| {
            viewers[0].ready() && viewers[0].audio_samples.len() >= 3
        });
        late.assert_decodes();
        assert_opus_decodes(&late);
        helper.success(json!({"command":"stop_preview"}));
        helper.stop_recording("next");

        // Then: Both finalized AAC clips decode and retain original compressed audio.
        for clip in [&first_clip, &next_clip] {
            assert_decode(clip, false);
            let mut input = ffmpeg_next::format::input(clip).unwrap();
            let index = input
                .streams()
                .find(|s| s.parameters().medium() == ffmpeg_next::media::Type::Audio)
                .unwrap()
                .index();
            let source = rtsp_camera::aac_access_units();
            let packets: Vec<_> = input
                .packets()
                .filter(|(s, _)| s.index() == index)
                .map(|(_, p)| p.data().unwrap().to_vec())
                .collect();
            assert!(!packets.is_empty());
            assert!(packets.iter().all(|packet| source.contains(packet)));
        }
        assert_eq!(camera.count("PLAY"), 1);
        helper.success(json!({"command":"stop"}));
        helper.wait_exit();
        assert_eq!(camera.count("TEARDOWN"), 1);
    }
}

#[test]
fn native_audio_enabled_preview_without_camera_audio_still_delivers_video() {
    // Given: The default audio-enabled preview configuration and a video-only camera.
    let camera = RtspCamera::start("H264");
    let mut helper = Helper::start(true);
    // When: The native path discovers no audio stream.
    helper.success(json!({"command":"start", "rtsp_url":camera.url,
        "preview_settings":{"video_codec":"h264","audio_enabled":true},
        "connect_timeout_s":5.0,"io_timeout_s":5.0}));
    let mut viewer = Viewer::attach(&mut helper, "video-only-source");
    progress_until(&mut [&mut viewer], |viewers| viewers[0].ready());
    // Then: Video decodes and no invented/silent audio packets appear.
    viewer.assert_decodes();
    assert!(viewer.audio_samples.is_empty());
    helper.success(json!({"command":"stop_preview"}));
    helper.success(json!({"command":"stop"}));
    helper.wait_exit();
    assert_eq!(camera.count("PLAY"), 1);
    assert_eq!(camera.count("TEARDOWN"), 1);
}
