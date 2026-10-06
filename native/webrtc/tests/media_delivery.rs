//! Exercise the executable boundary with real RTSP, FFmpeg RTP and encrypted WebRTC.

#[path = "support/rtsp_camera.rs"]
mod rtsp_camera;
use rtsp_camera::{RtspCamera, annex_b_nals};

use std::io::{BufRead, BufReader, Write};
use std::net::UdpSocket;
use std::path::Path;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{self, Receiver};
use std::thread;
use std::time::{Duration, Instant};

use serde_json::{Value, json};
use str0m::change::SdpAnswer;
use str0m::format::Codec;
use str0m::media::{Direction, MediaKind, Pt};
use str0m::net::{Protocol, Receive};
use str0m::{Candidate, Event, Input, Output, Rtc};

struct Helper {
    child: Child,
    stdin: Option<ChildStdin>,
    replies: Receiver<Value>,
    next_request: usize,
}

impl Helper {
    fn start() -> (Self, Value) {
        Self::start_with(2, 10.0, None)
    }

    fn start_with(
        max_viewers: usize,
        negotiation_timeout_s: f64,
        ffmpeg_fixture: Option<&Path>,
    ) -> (Self, Value) {
        let reservation = UdpSocket::bind("127.0.0.1:0").unwrap();
        let port = reservation.local_addr().unwrap().port();
        let port_end = port.saturating_add(15).to_string();
        let port = port.to_string();
        drop(reservation);
        let mut command = Command::new(env!("CARGO_BIN_EXE_homesec-webrtc"));
        command
            .args([
                "--advertised-ip",
                "127.0.0.1",
                "--udp-port-start",
                &port,
                "--udp-port-end",
                &port_end,
                "--max-viewers",
                &max_viewers.to_string(),
                "--negotiation-timeout-s",
                &negotiation_timeout_s.to_string(),
                "--max-session-duration-s",
                "60",
            ])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit());
        if let Some(directory) = ffmpeg_fixture {
            let mut paths = vec![directory.to_owned()];
            paths.extend(std::env::split_paths(&std::env::var_os("PATH").unwrap()));
            command.env("PATH", std::env::join_paths(paths).unwrap());
            command.env("HOMESEC_FFMPEG_PID_FILE", directory.join("ffmpeg.pid"));
        }
        let mut child = command.spawn().unwrap();
        let stdin = child.stdin.take().unwrap();
        let stdout = child.stdout.take().unwrap();
        let (tx, replies) = mpsc::channel();
        thread::spawn(move || {
            for line in BufReader::new(stdout).lines() {
                let line = line.unwrap();
                let value = serde_json::from_str(&line).expect("helper emitted invalid JSON");
                if tx.send(value).is_err() {
                    break;
                }
            }
        });
        let helper = Self {
            child,
            stdin: Some(stdin),
            replies,
            next_request: 0,
        };
        let ready = helper
            .replies
            .recv_timeout(Duration::from_secs(10))
            .expect("helper did not become ready");
        assert_eq!(ready["event"], "ready", "{ready}");
        (helper, ready)
    }

    fn request(&mut self, mut request: Value) -> Value {
        self.next_request += 1;
        let id = self.next_request.to_string();
        request["request_id"] = json!(id);
        let stdin = self.stdin.as_mut().unwrap();
        serde_json::to_writer(&mut *stdin, &request).unwrap();
        stdin.write_all(b"\n").unwrap();
        stdin.flush().unwrap();
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            let reply = self
                .replies
                .recv_timeout(remaining)
                .expect("helper request timed out");
            if reply["request_id"] == id {
                return reply;
            }
        }
    }

    fn successful_request(&mut self, request: Value) -> Value {
        let reply = self.request(request);
        assert_eq!(reply["ok"], true, "{reply}");
        reply
    }
}

impl Drop for Helper {
    fn drop(&mut self) {
        // EOF is the real parent-loss boundary and should release FFmpeg too.
        self.stdin.take();
        let deadline = Instant::now() + Duration::from_secs(2);
        loop {
            if self.child.try_wait().ok().flatten().is_some() {
                return;
            }
            if Instant::now() >= deadline {
                let _ = self.child.kill();
                let _ = self.child.wait();
                return;
            }
            thread::sleep(Duration::from_millis(5));
        }
    }
}

struct Viewer {
    rtc: Rtc,
    socket: UdpSocket,
    connected: bool,
    video_frames: usize,
    keyframes: usize,
    audio_packets: usize,
    video_sample: Vec<u8>,
    video_payload_types: Vec<Pt>,
}

impl Viewer {
    fn attach(helper: &mut Helper, session_id: &str) -> Self {
        let rtc = Rtc::builder()
            .clear_codecs()
            .enable_h264(true)
            .enable_opus(true, false)
            .build(Instant::now());
        Self::attach_rtc(helper, session_id, rtc)
    }

    fn attach_rtc(helper: &mut Helper, session_id: &str, mut rtc: Rtc) -> Self {
        let socket = UdpSocket::bind("127.0.0.1:0").unwrap();
        socket.set_nonblocking(true).unwrap();
        rtc.add_local_candidate(Candidate::host(socket.local_addr().unwrap(), "udp").unwrap())
            .unwrap();
        let mut change = rtc.sdp_api();
        change.add_media(MediaKind::Video, Direction::RecvOnly, None, None, None);
        change.add_media(MediaKind::Audio, Direction::RecvOnly, None, None, None);
        let (offer, pending) = change.apply().unwrap();
        let reply = helper.successful_request(json!({
            "command": "offer",
            "session_id": session_id,
            "sdp": offer.to_sdp_string(),
            "lease_seconds": 30.0,
        }));
        let answer = SdpAnswer::from_sdp_string(reply["sdp"].as_str().unwrap()).unwrap();
        rtc.sdp_api().accept_answer(pending, answer).unwrap();
        Self {
            rtc,
            socket,
            connected: false,
            video_frames: 0,
            keyframes: 0,
            audio_packets: 0,
            video_sample: Vec::new(),
            video_payload_types: Vec::new(),
        }
    }

    fn progress(&mut self) {
        let now = Instant::now();
        self.rtc.handle_input(Input::Timeout(now)).unwrap();
        let mut packet = [0_u8; 65_536];
        loop {
            match self.socket.recv_from(&mut packet) {
                Ok((length, source)) => {
                    let receive = Receive::new(
                        Protocol::Udp,
                        source,
                        self.socket.local_addr().unwrap(),
                        &packet[..length],
                    )
                    .unwrap();
                    self.rtc
                        .handle_input(Input::Receive(Instant::now(), receive))
                        .unwrap();
                }
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => break,
                Err(error) => panic!("viewer UDP receive failed: {error}"),
            }
        }
        loop {
            match self.rtc.poll_output().unwrap() {
                Output::Transmit(transmit) => {
                    self.socket
                        .send_to(&transmit.contents, transmit.destination)
                        .unwrap();
                }
                Output::Event(Event::Connected) => self.connected = true,
                Output::Event(Event::MediaData(data)) => {
                    assert!(!data.data.is_empty());
                    match data.params.spec().codec {
                        Codec::H264 => {
                            self.video_frames += 1;
                            self.video_payload_types.push(data.params.pt());
                            self.keyframes += usize::from(data.is_keyframe());
                            if self.video_frames <= 3 {
                                self.video_sample.extend_from_slice(&data.data);
                            }
                        }
                        Codec::Opus => self.audio_packets += 1,
                        codec => panic!("unexpected negotiated codec: {codec}"),
                    }
                }
                Output::Timeout(_) => break,
                Output::Event(_) => {}
            }
        }
    }

    fn has_media(&self) -> bool {
        self.connected && self.video_frames >= 3 && self.keyframes >= 1 && self.audio_packets >= 5
    }

    fn assert_video_decodes(&self) {
        let mut sample = tempfile::NamedTempFile::new().unwrap();
        sample.write_all(&self.video_sample).unwrap();
        // Bound FFmpeg thread pools when media tests run in parallel.
        let output = Command::new("ffmpeg")
            .args([
                "-hide_banner",
                "-loglevel",
                "error",
                "-xerror",
                "-filter_threads",
                "1",
                "-threads",
                "1",
                "-f",
                "h264",
                "-i",
            ])
            .arg(sample.path())
            .args([
                "-threads",
                "1",
                "-frames:v",
                "1",
                "-pix_fmt",
                "rgb24",
                "-f",
                "rawvideo",
                "pipe:1",
            ])
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "received H.264 did not decode: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert_eq!(output.stdout.len(), 160 * 120 * 3);
    }
}

fn ffmpeg_args(ready: &Value) -> Vec<String> {
    let video_port = ready["video_port"].as_u64().unwrap();
    let audio_port = ready["audio_port"].as_u64().unwrap();
    [
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-re",
        "-f",
        "lavfi",
        "-i",
        "testsrc2=size=160x120:rate=10",
        "-re",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:sample_rate=48000",
        "-map",
        "0:v:0",
        "-c:v",
        "libx264",
        "-threads",
        "1",
        "-preset",
        "ultrafast",
        "-tune",
        "zerolatency",
        "-profile:v",
        "baseline",
        "-pix_fmt",
        "yuv420p",
        "-g",
        "10",
        "-bf",
        "0",
        "-payload_type",
        "96",
        "-f",
        "rtp",
        &format!("rtp://127.0.0.1:{video_port}?pkt_size=1200"),
        "-map",
        "1:a:0",
        "-c:a",
        "libopus",
        "-ar",
        "48000",
        "-ac",
        "2",
        "-payload_type",
        "97",
        "-f",
        "rtp",
        &format!("rtp://127.0.0.1:{audio_port}?pkt_size=1200"),
    ]
    .into_iter()
    .map(str::to_owned)
    .collect()
}

fn progress_until(viewers: &mut [&mut Viewer], complete: impl Fn(&[&mut Viewer]) -> bool) {
    let deadline = Instant::now() + Duration::from_secs(15);
    loop {
        for viewer in viewers.iter_mut() {
            viewer.progress();
        }
        if complete(viewers) {
            return;
        }
        assert!(
            Instant::now() < deadline,
            "WebRTC media timed out: {:?}",
            viewers
                .iter()
                .map(|v| (v.connected, v.video_frames, v.keyframes, v.audio_packets))
                .collect::<Vec<_>>()
        );
        thread::sleep(Duration::from_millis(2));
    }
}

fn wait_until(mut complete: impl FnMut() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(3);
    while !complete() {
        assert!(Instant::now() < deadline, "lifecycle operation timed out");
        thread::sleep(Duration::from_millis(5));
    }
}

fn unconnected_offer(packetization_mode: bool) -> String {
    let mut config = Rtc::builder().clear_codecs();
    config
        .codec_config()
        .add_h264(96.into(), None, packetization_mode, 0x42e01f);
    let mut rtc = config.build(Instant::now());
    rtc.add_local_candidate(Candidate::host("127.0.0.1:9".parse().unwrap(), "udp").unwrap())
        .unwrap();
    let mut change = rtc.sdp_api();
    change.add_media(MediaKind::Video, Direction::RecvOnly, None, None, None);
    change.apply().unwrap().0.to_sdp_string()
}

#[test]
fn encrypted_media_reaches_two_viewers_and_survives_one_closing() {
    // Given: The real helper receives H.264 and Opus RTP from a single FFmpeg input.
    assert!(
        Command::new("ffmpeg")
            .arg("-version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .expect("install ffmpeg to run the WebRTC integration test")
            .success()
    );
    let (mut helper, ready) = Helper::start();
    let media_port = ready["media_port"].as_u64().unwrap();
    assert!(media_port > 0);
    helper.successful_request(json!({"command": "start", "ffmpeg_args": ffmpeg_args(&ready)}));

    // When: A receive-only peer starts playing, then a second joins over real UDP.
    let mut first = Viewer::attach(&mut helper, "first");
    progress_until(&mut [&mut first], |viewers| viewers[0].has_media());
    let mut second = Viewer::attach(&mut helper, "second");
    progress_until(&mut [&mut first, &mut second], |viewers| {
        viewers.iter().all(|viewer| viewer.has_media())
    });

    // Then: DTLS/SRTP delivery depayloads video keyframes and Opus for both peers.
    assert!(first.has_media());
    assert!(second.has_media());
    first.assert_video_decodes();
    second.assert_video_decodes();
    let status = helper.successful_request(json!({"command": "status"}));
    assert_eq!(status["viewer_count"], 2);
    assert_eq!(status["active_session_count"], 2);
    assert_eq!(status["media_active"], true);

    // When: One viewer closes, and the surviving viewer renews its authorization.
    helper.successful_request(json!({"command": "close", "session_id": "first"}));
    helper.successful_request(json!({"command": "close", "session_id": "first"}));
    helper.successful_request(json!({
        "command": "renew", "session_id": "second", "lease_seconds": 30.0
    }));
    let previous_video = second.video_frames;
    let previous_audio = second.audio_packets;
    progress_until(&mut [&mut second], |viewers| {
        viewers[0].video_frames >= previous_video + 5
            && viewers[0].audio_packets >= previous_audio + 10
    });

    // Then: The camera input and surviving peer remain live; stop releases them.
    let status = helper.successful_request(json!({"command": "status"}));
    assert_eq!(status["viewer_count"], 1);
    assert_eq!(status["active_session_count"], 1);
    assert_eq!(status["media_active"], true);
    helper.successful_request(json!({"command": "close", "session_id": "second"}));
    let status = helper.successful_request(json!({"command": "status"}));
    assert_eq!(status["viewer_count"], 0);
    assert_eq!(status["active_session_count"], 0);
    helper.successful_request(json!({"command": "stop"}));
}

#[test]
fn malformed_offers_and_negotiation_timeout_release_admission_capacity() {
    // Given: One viewer slot and a short negotiation deadline on a live input.
    let (mut helper, ready) = Helper::start_with(1, 0.2, None);
    helper.successful_request(json!({"command": "start", "ffmpeg_args": ffmpeg_args(&ready)}));

    // When: Malformed or oversized SDP and invalid leases are submitted.
    for (sdp, lease_seconds) in [
        ("not SDP".to_owned(), 30.0),
        ("x".repeat(128 * 1024 + 1), 30.0),
        (unconnected_offer(true), 0.0),
    ] {
        let reply = helper.request(json!({
            "command": "offer", "session_id": "invalid", "sdp": sdp,
            "lease_seconds": lease_seconds,
        }));
        assert_eq!(reply["ok"], false);
        assert_eq!(reply["error_code"], "invalid_offer");
    }

    // Then: Invalid offers consume no slot; an unconnected valid peer does.
    helper.successful_request(json!({
        "command": "offer", "session_id": "unconnected", "sdp": unconnected_offer(true),
        "lease_seconds": 30.0,
    }));
    let status = helper.successful_request(json!({"command": "status"}));
    assert_eq!(status["viewer_count"], 0);
    assert_eq!(status["active_session_count"], 1);
    let replacement_offer = unconnected_offer(true);
    let request = json!({
        "command": "offer", "session_id": "replacement", "sdp": replacement_offer,
        "lease_seconds": 30.0,
    });
    let reply = helper.request(request.clone());
    assert_eq!(reply["ok"], false);
    assert_eq!(reply["error_code"], "session_limit");

    // When: The first peer never sends ICE/DTLS and its negotiation times out.
    wait_until(|| {
        let status = helper.successful_request(json!({"command": "status"}));
        assert_eq!(status["viewer_count"], 0);
        status["active_session_count"] == 0
    });
    wait_until(|| {
        let reply = helper.request(request.clone());
        if reply["ok"] == true {
            return true;
        }
        assert_eq!(reply["error_code"], "session_limit");
        false
    });

    // Then: Another peer can use the released slot, and the old lease cannot renew.
    let reply = helper.request(json!({
        "command": "renew", "session_id": "unconnected", "lease_seconds": 30.0
    }));
    assert_eq!(reply["error_code"], "session_not_found");
    helper.successful_request(json!({"command": "stop"}));
}

#[test]
fn unsupported_h264_packetization_is_refused_without_affecting_valid_viewers() {
    // Given: A live helper can only send H.264 using STAP-A/FU-A packetization.
    let (mut helper, ready) = Helper::start();
    helper.successful_request(json!({"command": "start", "ffmpeg_args": ffmpeg_args(&ready)}));

    // When: A receive-only browser offers only H.264 packetization mode 0.
    let reply = helper.request(json!({
        "command": "offer", "session_id": "mode-zero", "sdp": unconnected_offer(false),
        "lease_seconds": 30.0,
    }));

    // Then: Its unsupported offer consumes no slot, while a valid peer still receives media.
    assert_eq!(reply["ok"], false);
    assert_eq!(reply["error_code"], "invalid_offer");
    let status = helper.successful_request(json!({"command": "status"}));
    assert_eq!(status["active_session_count"], 0);
    let mut viewer = Viewer::attach(&mut helper, "mode-one");
    progress_until(&mut [&mut viewer], |viewers| viewers[0].has_media());
    assert!(viewer.has_media());
    viewer.assert_video_decodes();
    let status = helper.successful_request(json!({"command": "status"}));
    assert_eq!(status["viewer_count"], 1);
    assert_eq!(status["active_session_count"], 1);
    helper.successful_request(json!({"command": "stop"}));
}

#[test]
fn authorization_expiry_closes_a_connected_peer_without_stopping_the_input() {
    // Given: A connected peer receiving encrypted video and audio.
    let (mut helper, ready) = Helper::start();
    helper.successful_request(json!({"command": "start", "ffmpeg_args": ffmpeg_args(&ready)}));
    let mut viewer = Viewer::attach(&mut helper, "expiring");
    progress_until(&mut [&mut viewer], |viewers| viewers[0].has_media());
    assert_eq!(
        helper.successful_request(json!({"command": "status"}))["viewer_count"],
        1
    );

    // When: Its renewed authorization lease expires while the socket stays open.
    helper.successful_request(json!({
        "command": "renew", "session_id": "expiring", "lease_seconds": 0.15
    }));
    wait_until(|| {
        viewer.progress();
        helper.successful_request(json!({"command": "status"}))["viewer_count"] == 0
    });

    // Then: The expired session cannot renew; source-owned input lifetime remains separate.
    let reply = helper.request(json!({
        "command": "renew", "session_id": "expiring", "lease_seconds": 30.0
    }));
    assert_eq!(reply["ok"], false);
    assert_eq!(reply["error_code"], "session_not_found");
    assert_eq!(
        helper.successful_request(json!({"command": "status"}))["media_active"],
        true
    );
    helper.successful_request(json!({"command": "stop"}));
}

#[test]
fn duplicate_sending_tracks_are_refused_without_consuming_a_viewer_slot() {
    // Given: One viewer slot on an input supporting one H.264 video and one Opus audio track.
    let (mut helper, ready) = Helper::start_with(1, 10.0, None);
    helper.successful_request(json!({"command": "start", "ffmpeg_args": ffmpeg_args(&ready)}));

    // When: Offers request duplicate sending video or audio tracks in the answer.
    for (video_tracks, audio_tracks) in [(2, 1), (1, 2)] {
        let mut rtc = Rtc::builder()
            .clear_codecs()
            .enable_h264(true)
            .enable_opus(true, false)
            .build(Instant::now());
        rtc.add_local_candidate(Candidate::host("127.0.0.1:9".parse().unwrap(), "udp").unwrap())
            .unwrap();
        let mut change = rtc.sdp_api();
        for (kind, count) in [
            (MediaKind::Video, video_tracks),
            (MediaKind::Audio, audio_tracks),
        ] {
            for _ in 0..count {
                change.add_media(kind, Direction::RecvOnly, None, None, None);
            }
        }
        let offer = change.apply().unwrap().0.to_sdp_string();
        let reply = helper.request(json!({
            "command": "offer", "session_id": "duplicate", "sdp": offer,
            "lease_seconds": 30.0,
        }));

        // Then: Unsupported track multiplicity has a stable refusal and consumes no slot.
        assert_eq!(reply["ok"], false);
        assert_eq!(reply["error_code"], "invalid_offer");
        let status = helper.successful_request(json!({"command": "status"}));
        assert_eq!(status["active_session_count"], 0);
    }
    let mut viewer = Viewer::attach(&mut helper, "valid");
    progress_until(&mut [&mut viewer], |viewers| viewers[0].has_media());
    assert!(viewer.has_media());
    let status = helper.successful_request(json!({"command": "status"}));
    assert_eq!(status["viewer_count"], 1);
    assert_eq!(status["active_session_count"], 1);
    helper.successful_request(json!({"command": "stop"}));
}

#[test]
fn failed_ffmpeg_input_refuses_new_sessions() {
    // Given: FFmpeg starts but exits before producing any video.
    let (mut helper, _) = Helper::start();
    helper.successful_request(json!({
        "command": "start", "ffmpeg_args": ["-homesec-invalid-option"]
    }));

    // When: The worker observes the process failure.
    wait_until(|| helper.successful_request(json!({"command": "status"}))["state"] == "error");

    // Then: The failed input is inactive and no viewer can negotiate a dead stream.
    let status = helper.successful_request(json!({"command": "status"}));
    assert_eq!(status["media_active"], false);
    assert_eq!(status["viewer_count"], 0);
    let reply = helper.request(json!({
        "command": "offer", "session_id": "refused", "sdp": unconnected_offer(true),
        "lease_seconds": 30.0,
    }));
    assert_eq!(reply["ok"], false);
    assert_eq!(reply["error_code"], "preview_temporarily_unavailable");
    helper.successful_request(json!({"command": "stop"}));
}

#[cfg(unix)]
#[test]
fn parent_eof_reaps_the_ffmpeg_process_and_releases_the_media_socket() {
    use std::os::unix::fs::PermissionsExt;

    // Given: A live FFmpeg fixture records its PID at the subprocess boundary.
    let fixture = tempfile::tempdir().unwrap();
    let executable = fixture.path().join("ffmpeg");
    std::fs::write(
        &executable,
        "#!/bin/sh\nprintf '%s' \"$$\" > \"$HOMESEC_FFMPEG_PID_FILE\"\nexec sleep 60\n",
    )
    .unwrap();
    std::fs::set_permissions(&executable, std::fs::Permissions::from_mode(0o700)).unwrap();
    let (mut helper, ready) = Helper::start_with(1, 10.0, Some(fixture.path()));
    helper.successful_request(json!({"command": "start", "ffmpeg_args": ["fixture"]}));
    let pid_file = fixture.path().join("ffmpeg.pid");
    wait_until(|| pid_file.exists() && !std::fs::read_to_string(&pid_file).unwrap().is_empty());
    let pid = std::fs::read_to_string(pid_file).unwrap();

    // When: Its parent closes control stdin without sending a stop command.
    helper.stdin.take();
    wait_until(|| helper.child.try_wait().unwrap().is_some());

    // Then: The helper exits successfully, reaps FFmpeg, and releases its UDP port.
    assert!(helper.child.wait().unwrap().success());
    assert!(
        !Command::new("kill")
            .args(["-0", &pid])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .unwrap()
            .success()
    );
    let media_port = ready["media_port"].as_u64().unwrap();
    assert!(UdpSocket::bind(format!("127.0.0.1:{media_port}")).is_ok());
}

#[cfg(unix)]
fn ffmpeg_trap() -> tempfile::TempDir {
    use std::os::unix::fs::PermissionsExt;
    let fixture = tempfile::tempdir().unwrap();
    let executable = fixture.path().join("ffmpeg");
    std::fs::write(
        &executable,
        "#!/bin/sh\nprintf invoked > \"$HOMESEC_FFMPEG_PID_FILE\"\nexit 64\n",
    )
    .unwrap();
    std::fs::set_permissions(&executable, std::fs::Permissions::from_mode(0o700)).unwrap();
    fixture
}

#[cfg(unix)]
#[test]
fn native_rtsp_shares_one_play_and_sdp_parameters_between_two_viewers() {
    // Given: A TCP camera advertises PT101 and supplies SPS/PPS only in SDP.
    let camera = RtspCamera::start("H264");
    let trap = ffmpeg_trap();
    let (mut helper, _) = Helper::start_with(2, 10.0, Some(trap.path()));
    helper.successful_request(json!({"command":"start", "rtsp_url":camera.url}));
    wait_until(|| helper.successful_request(json!({"command":"status"}))["state"] == "ready");

    // When: Two real encrypted WebRTC viewers join the same native input.
    let mut first = Viewer::attach(&mut helper, "native-first");
    progress_until(&mut [&mut first], |viewers| {
        viewers[0].connected && viewers[0].video_frames >= 3 && viewers[0].keyframes >= 1
    });
    let mut second = Viewer::attach(&mut helper, "native-second");
    progress_until(&mut [&mut first, &mut second], |viewers| {
        viewers
            .iter()
            .all(|viewer| viewer.connected && viewer.video_frames >= 3 && viewer.keyframes >= 1)
    });

    // Then: Both late-join streams initialize a decoder without a second camera PLAY.
    first.assert_video_decodes();
    second.assert_video_decodes();
    for viewer in [&first, &second] {
        let nals = annex_b_nals(&viewer.video_sample);
        assert!(nals.iter().any(|nal| nal[0] & 31 == 7));
        assert!(nals.iter().any(|nal| nal[0] & 31 == 8));
        assert_eq!(viewer.audio_packets, 0);
    }
    assert_eq!(camera.count("DESCRIBE"), 1);
    assert_eq!(camera.count("SETUP"), 1);
    assert_eq!(camera.count("PLAY"), 1);
    assert!(
        !trap.path().join("ffmpeg.pid").exists(),
        "native ingestion invoked FFmpeg"
    );
    let status = helper.successful_request(json!({"command":"status"}));
    assert_eq!(status["viewer_count"], 2);

    // When: One peer detaches while the other keeps receiving from that camera session.
    helper.successful_request(json!({"command":"close", "session_id":"native-first"}));
    let previous = second.video_frames;
    progress_until(&mut [&mut second], |viewers| {
        viewers[0].video_frames >= previous + 3
    });

    // Then: Input remains shared until explicit worker stop tears it down.
    assert_eq!(camera.count("PLAY"), 1);
    helper.successful_request(json!({"command":"stop"}));
    wait_until(|| camera.count("DISCONNECTED") == 1);
    assert_eq!(camera.count("TEARDOWN"), 1);
}

#[cfg(unix)]
#[test]
fn native_rtsp_parent_eof_tears_down_input_and_releases_media_socket() {
    // Given: The native helper owns one playing TCP camera session.
    let camera = RtspCamera::start("H264");
    let trap = ffmpeg_trap();
    let (mut helper, ready) = Helper::start_with(1, 10.0, Some(trap.path()));
    helper.successful_request(json!({"command":"start", "rtsp_url":camera.url}));
    wait_until(|| camera.count("PLAY") == 1);

    // When: Its parent closes control stdin without a stop command.
    helper.stdin.take();
    wait_until(|| helper.child.try_wait().unwrap().is_some());

    // Then: The helper releases RTSP and its UDP socket without starting FFmpeg.
    assert!(helper.child.wait().unwrap().success());
    wait_until(|| camera.count("DISCONNECTED") == 1);
    assert_eq!(camera.count("TEARDOWN"), 1);
    assert!(!trap.path().join("ffmpeg.pid").exists());
    let port = ready["media_port"].as_u64().unwrap();
    assert!(UdpSocket::bind(format!("127.0.0.1:{port}")).is_ok());
}

#[test]
fn native_rtsp_unsupported_codec_is_unavailable_without_playing_camera() {
    // Given: The camera describes a video codec outside the H.264 preview contract.
    let camera = RtspCamera::start("H265");
    let (mut helper, _) = Helper::start();

    // When: Native ingestion inspects that camera's SDP.
    let reply = helper.request(json!({"command":"start", "rtsp_url":camera.url}));
    if reply["ok"] == true {
        wait_until(|| helper.successful_request(json!({"command":"status"}))["state"] == "error");
    } else {
        assert_eq!(reply["error_code"], "unsupported_codec");
    }

    // Then: No viewer can attach and no unsupported media session reaches PLAY.
    let reply = helper.request(json!({"command":"offer", "session_id":"unsupported", "sdp":unconnected_offer(true), "lease_seconds":30.0}));
    assert_eq!(reply["ok"], false);
    assert_eq!(reply["error_code"], "preview_temporarily_unavailable");
    assert_eq!(camera.count("PLAY"), 0);
    helper.successful_request(json!({"command":"stop"}));
}

#[test]
fn native_rtsp_receiver_profile_and_level_constraints_refuse_unsupported_offers() {
    // Given: The camera's actual H.264 stream is constrained baseline at level 3.1.
    let camera = RtspCamera::start("H264");
    let (mut helper, _) = Helper::start();
    helper.successful_request(json!({"command":"start", "rtsp_url":camera.url}));
    wait_until(|| helper.successful_request(json!({"command":"status"}))["state"] == "ready");

    // When: Receivers accept only baseline level 1.0 or the incompatible high profile.
    for profile_level_id in [0x42e00a, 0x64001f] {
        let mut config = Rtc::builder().clear_codecs();
        config
            .codec_config()
            .add_h264(96.into(), None, true, profile_level_id);
        let mut rtc = config.build(Instant::now());
        rtc.add_local_candidate(Candidate::host("127.0.0.1:9".parse().unwrap(), "udp").unwrap())
            .unwrap();
        let mut change = rtc.sdp_api();
        change.add_media(MediaKind::Video, Direction::RecvOnly, None, None, None);
        let offer = change.apply().unwrap().0.to_sdp_string();
        let reply = helper.request(json!({"command":"offer", "session_id":"unsupported-receiver", "sdp":offer, "lease_seconds":30.0}));

        // Then: Negotiation refuses the incompatible actual stream and consumes no slot.
        assert_eq!(reply["ok"], false);
        assert_eq!(reply["error_code"], "invalid_offer");
        assert_eq!(
            helper.successful_request(json!({"command":"status"}))["active_session_count"],
            0
        );
    }

    // When: A receiver offers a compatible profile and level after those refusals.
    let mut viewer = Viewer::attach(&mut helper, "compatible-receiver");
    progress_until(&mut [&mut viewer], |viewers| {
        viewers[0].connected && viewers[0].video_frames >= 3 && viewers[0].keyframes >= 1
    });

    // Then: The shared native source still delivers decodable encrypted video.
    viewer.assert_video_decodes();
    assert_eq!(camera.count("PLAY"), 1);
    helper.successful_request(json!({"command":"stop"}));
}

#[test]
fn native_rtsp_pending_describe_does_not_block_stop_or_leave_camera_connected() {
    // Given: The fake camera accepts TCP but leaves its DESCRIBE reply pending.
    let camera = RtspCamera::start("STALL");
    let (mut helper, _) = Helper::start();
    let stdin = helper.stdin.as_mut().unwrap();
    serde_json::to_writer(
        &mut *stdin,
        &json!({"request_id":"pending", "command":"start", "rtsp_url":camera.url}),
    )
    .unwrap();
    stdin.write_all(b"\n").unwrap();
    stdin.flush().unwrap();
    wait_until(|| camera.count("DESCRIBE") == 1);

    // When: The parent stops the helper while camera metadata is still unavailable.
    let stopped_at = Instant::now();
    helper.successful_request(json!({"command":"stop"}));
    wait_until(|| helper.child.try_wait().unwrap().is_some());

    // Then: Cancellation is bounded and releases the connection before setup or PLAY.
    assert!(stopped_at.elapsed() < Duration::from_secs(2));
    assert!(helper.child.wait().unwrap().success());
    wait_until(|| camera.count("DISCONNECTED") == 1);
    assert_eq!(camera.count("SETUP"), 0);
    assert_eq!(camera.count("PLAY"), 0);
}

#[test]
fn native_rtsp_stalled_read_expires_input_while_control_remains_responsive() {
    // Given: Camera negotiation succeeds but its playing TCP stream produces no RTP.
    let camera = RtspCamera::start("STALL_PLAY");
    let (mut helper, _) = Helper::start();
    helper.successful_request(json!({"command":"start", "rtsp_url":camera.url}));
    assert_eq!(camera.count("PLAY"), 1);
    let pending = helper.successful_request(json!({"command":"status"}));
    assert_eq!(pending["state"], "starting");
    assert_eq!(pending["media_active"], false);

    // When: The read remains pending beyond the native source's bounded I/O timeout.
    let deadline = Instant::now() + Duration::from_secs(12);
    loop {
        let requested_at = Instant::now();
        let status = helper.successful_request(json!({"command":"status"}));
        assert!(
            requested_at.elapsed() < Duration::from_secs(2),
            "stalled camera blocked control"
        );
        if status["state"] == "error" {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "stalled native source did not expire"
        );
        thread::sleep(Duration::from_millis(50));
    }

    // Then: The camera connection closes and new peers cannot attach to a dead input.
    wait_until(|| camera.count("DISCONNECTED") == 1);
    let status = helper.successful_request(json!({"command":"status"}));
    assert_eq!(status["media_active"], false);
    assert_eq!(status["active_session_count"], 0);
    let reply = helper.request(json!({"command":"offer", "session_id":"stalled-input", "sdp":unconnected_offer(true), "lease_seconds":30.0}));
    assert_eq!(reply["error_code"], "preview_temporarily_unavailable");
    assert_eq!(camera.count("TEARDOWN"), 1);
    helper.successful_request(json!({"command":"stop"}));
}

#[test]
fn native_rtsp_selects_compatible_receiver_level_over_a_closer_unsupported_level() {
    // Given: A real level-3.1 source and one peer offering baseline levels 3.0 and 4.2.
    let camera = RtspCamera::start("H264");
    let (mut helper, _) = Helper::start();
    helper.successful_request(json!({"command":"start", "rtsp_url":camera.url}));
    wait_until(|| helper.successful_request(json!({"command":"status"}))["state"] == "ready");
    let mut config = Rtc::builder().clear_codecs().enable_opus(true, false);
    config
        .codec_config()
        .add_h264(96.into(), None, true, 0x42e01e);
    config
        .codec_config()
        .add_h264(98.into(), None, true, 0x42e02a);

    // When: The peer joins with a supported receive envelope after the lower-level PT.
    let mut viewer = Viewer::attach_rtc(&mut helper, "mixed-levels", config.build(Instant::now()));
    progress_until(&mut [&mut viewer], |viewers| {
        viewers[0].connected && viewers[0].video_frames >= 3 && viewers[0].keyframes >= 1
    });

    // Then: Encrypted video uses PT98 and decodes at the source's actual dimensions.
    assert!(
        viewer
            .video_payload_types
            .iter()
            .all(|pt| *pt == Pt::from(98))
    );
    viewer.assert_video_decodes();
    assert_eq!(camera.count("PLAY"), 1);
    helper.successful_request(json!({"command":"stop"}));
}
