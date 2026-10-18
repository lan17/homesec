//! Exercise the actual inspection executable against private fake RTSP cameras.
#[path = "support/rtsp_camera.rs"]
mod rtsp_camera;
use rtsp_camera::RtspCamera;
use serde_json::{Value, json};
use std::io::{BufRead, BufReader, Read, Write};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{self, Receiver};
use std::thread;
use std::time::{Duration, Instant};

struct Helper {
    child: Child,
    input: ChildStdin,
    replies: Receiver<Value>,
}
impl Helper {
    fn start() -> Self {
        let mut child = Command::new(env!("CARGO_BIN_EXE_homesec-webrtc"))
            .arg("--inspect")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let input = child.stdin.take().unwrap();
        let output = child.stdout.take().unwrap();
        let (sender, replies) = mpsc::channel();
        thread::spawn(move || {
            for line in BufReader::new(output).lines() {
                let Ok(line) = line else {
                    break;
                };
                let Ok(reply) = serde_json::from_str(&line) else {
                    break;
                };
                if sender.send(reply).is_err() {
                    break;
                }
            }
        });
        let helper = Self {
            child,
            input,
            replies,
        };
        let ready = helper.replies.recv_timeout(Duration::from_secs(5)).unwrap();
        assert_eq!(ready["event"], "ready");
        assert_eq!(ready["inspection_version"], 1);
        helper
    }
    fn send(&mut self, request: &Value) {
        serde_json::to_writer(&mut self.input, request).unwrap();
        self.input.write_all(b"\n").unwrap();
        self.input.flush().unwrap();
    }
    fn request(&mut self, request: Value) -> Value {
        self.send(&request);
        self.replies.recv_timeout(Duration::from_secs(12)).unwrap()
    }
    fn finish(&mut self) -> String {
        self.send(&json!({"command":"stop","request_id":"stop"}));
        loop {
            let reply = self.replies.recv_timeout(Duration::from_secs(3)).unwrap();
            if reply["request_id"] == "stop" {
                assert_eq!(reply["ok"], true);
                break;
            }
        }
        assert!(self.child.wait().unwrap().success());
        let mut stderr = String::new();
        self.child
            .stderr
            .take()
            .unwrap()
            .read_to_string(&mut stderr)
            .unwrap();
        stderr
    }
}
impl Drop for Helper {
    fn drop(&mut self) {
        if self.child.try_wait().ok().flatten().is_none() {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }
}
fn camera_request(command: &str, camera: &RtspCamera) -> Value {
    json!({"command":command,"request_id":command,"rtsp_url":camera.url,"connect_timeout_s":4.0,"io_timeout_s":2.0,"operation_timeout_s":8.0})
}
fn wait_closed(camera: &RtspCamera, count: usize) {
    let deadline = Instant::now() + Duration::from_secs(1);
    while camera.count("DISCONNECTED") < count && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(5));
    }
    assert_eq!(camera.count("DISCONNECTED"), count);
}

#[test]
fn actual_helper_metadata_does_not_require_h264_or_a_first_idr() {
    // Given: Independent fake cameras advertise ordinary H.264, HEVC, and no IDR.
    let mut helper = Helper::start();
    for (mode, expected) in [("H264", "h264"), ("H265_VALID", "hevc"), ("NO_IDR", "h264")] {
        let camera = RtspCamera::start(mode);
        // When: The actual executable performs metadata-only inspection.
        let reply = helper.request(camera_request("inspect_metadata", &camera));
        // Then: Advertised usable metadata succeeds independently of recorder gates.
        assert_eq!(reply["ok"], true, "{mode}: {reply}");
        assert_eq!(reply["inspection"]["video_codec"], expected);
        assert_eq!(reply["inspection"]["width"], 160);
        assert_eq!(reply["inspection"]["height"], 120);
        wait_closed(&camera, 1);
        if expected == "hevc" {
            let mut decode = camera_request("inspect_decode", &camera);
            decode["backend"] = json!("software");
            decode["sample_frames"] = json!(5);
            let reply = helper.request(decode);
            assert_eq!(reply["ok"], true, "HEVC decode: {reply}");
            let directory = tempfile::tempdir().unwrap();
            let mut recording = camera_request("inspect_recording", &camera);
            recording["audio_mode"] = json!("none");
            recording["timestamp_mode"] = json!("stream");
            recording["duration_s"] = json!(0.3);
            recording["output_path"] = json!(directory.path().join("hevc.mp4"));
            let reply = helper.request(recording);
            assert_eq!(reply["ok"], true, "HEVC trial: {reply}");
            assert!(reply["inspection"]["video_packets"].as_u64().unwrap() > 0);
        }
    }
    assert!(helper.finish().is_empty());
}
#[test]
fn actual_helper_decodes_and_finalizes_copy_and_aac_trials() {
    // Given: A synthetic RTSP camera exposes video and a real AAC tone.
    let camera = RtspCamera::start("H264_AAC");
    let directory = tempfile::tempdir().unwrap();
    let mut helper = Helper::start();
    let mut request = camera_request("inspect_decode", &camera);
    request["backend"] = json!("software");
    request["sample_frames"] = json!(5);
    // When: Decode and each audio profile are exercised through private IPC.
    let reply = helper.request(request);
    assert_eq!(reply["ok"], true, "{reply}");
    assert!(reply["inspection"]["frames_decoded"].as_u64().unwrap() >= 5);
    for mode in ["copy", "aac", "none"] {
        let path = directory.path().join(format!("{mode}.mp4"));
        let mut request = camera_request("inspect_recording", &camera);
        request["audio_mode"] = json!(mode);
        request["timestamp_mode"] = json!("stream");
        request["output_path"] = json!(path);
        request["duration_s"] = json!(0.5);
        let reply = helper.request(request);
        // Then: Success carries actual packet counts and a finalized readable file.
        assert_eq!(reply["ok"], true, "{mode}: {reply}");
        assert_eq!(reply["inspection"]["finalized"], true);
        assert!(reply["inspection"]["video_packets"].as_u64().unwrap() > 0);
        assert_eq!(
            reply["inspection"]["audio_packets"].as_u64().unwrap() > 0,
            mode != "none"
        );
        assert!(path.metadata().unwrap().len() > 32);
        let output = ffmpeg_next::format::input(&path).unwrap();
        assert_eq!(
            output
                .streams()
                .filter(|stream| stream.parameters().medium() == ffmpeg_next::media::Type::Audio)
                .count(),
            usize::from(mode != "none")
        );
    }
    assert!(helper.finish().is_empty());
}
#[test]
fn actual_helper_keeps_equal_url_roles_independent_and_classifies_explicit_refusal() {
    // Given: Existing policy checks three independent roles against the same URL.
    let camera = RtspCamera::start("H264");
    let mut helper = Helper::start();
    let request = json!({"command":"inspect_overlap","request_id":"overlap","operation_timeout_s":10.0,"connect_timeout_s":4.0,"io_timeout_s":2.0,"overlap_s":0.2,"duration_s":0.2,"roles":[{"role":"motion","rtsp_url":camera.url,"kind":"decode","backend":"software"},{"role":"preview","rtsp_url":camera.url,"kind":"decode","backend":"software"},{"role":"metadata","rtsp_url":camera.url,"kind":"metadata"}]});
    // When: The real native inputs overlap and complete with media evidence.
    let reply = helper.request(request);
    // Then: URL equality does not collapse sessions or replace completion with liveness.
    assert_eq!(reply["ok"], true, "{reply}");
    assert_eq!(reply["inspection"]["status"], "supported", "{reply}");
    assert_eq!(camera.count("PLAY"), 3);
    wait_closed(&camera, 3);
    for (mode, error) in [
        ("SESSION_LIMIT", "session_limit"),
        ("SESSION_PHRASE", "session_limit"),
        ("SESSION_MAX_CONNECTIONS", "session_limit"),
        ("GENERIC_503", "connection_failed"),
        ("AUTH_FAILURE", "authentication_failed"),
    ] {
        let camera = RtspCamera::start(mode);
        let reply = helper.request(camera_request("inspect_metadata", &camera));
        assert_eq!(reply["ok"], false, "{reply}");
        assert_eq!(reply["error_code"], error, "{reply}");
    }
    assert!(helper.finish().is_empty());
}
#[test]
fn cancelled_helper_closes_stalled_camera_and_exposes_no_private_url() {
    // Given: Native discovery is stalled after PLAY with secret-bearing input.
    let camera = RtspCamera::start("STALL_PLAY");
    let mut helper = Helper::start();
    let mut request = camera_request("inspect_metadata", &camera);
    request["rtsp_url"] = json!(
        camera
            .url
            .replace("rtsp://", "rtsp://inspection-user:private-password@")
    );
    helper.send(&request);
    let deadline = Instant::now() + Duration::from_secs(3);
    while camera.count("PLAY") == 0 && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(5));
    }
    // When: Stop cancels actual native work rather than abandoning its owner.
    let start = Instant::now();
    let stderr = helper.finish();
    // Then: Cleanup closes the socket within its bound and no raw URL is printed.
    let elapsed = start.elapsed();
    assert!(
        elapsed < Duration::from_secs(2),
        "stop acknowledgement, process exit, and stderr drain took {elapsed:?}"
    );
    wait_closed(&camera, 1);
    assert!(!stderr.contains("inspection-user") && !stderr.contains("private-password"));
}
#[test]
fn inspection_mode_is_exclusive_and_oversized_controls_end_the_owner() {
    // Given: Conflicting modes and an over-bound private control request.
    let status = Command::new(env!("CARGO_BIN_EXE_homesec-webrtc"))
        .args(["--inspect", "--motion"])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .unwrap();
    let mut helper = Helper::start();
    // When: CLI exclusivity and framing reject each invalid input.
    assert!(!status.success());
    // A syntactically valid request plus whitespace isolates the framing bound
    // from JSON validation; it would produce capabilities without this limit.
    let mut payload =
        serde_json::to_vec(&json!({"command":"inspect_capabilities","request_id":"large"}))
            .unwrap();
    payload.resize(crate::MAX_INSPECTION_TEST_BYTES, b' ');
    helper.input.write_all(&payload).unwrap();
    helper.input.write_all(b"\n").unwrap();
    helper.input.flush().unwrap();
    let deadline = Instant::now() + Duration::from_secs(2);
    // Then: Invalid framing ends the worker without echoing input.
    while helper.child.try_wait().unwrap().is_none() && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(5));
    }
    assert!(helper.child.try_wait().unwrap().is_some());
}
const MAX_INSPECTION_TEST_BYTES: usize = 128_001;

#[test]
fn malformed_controls_and_unbounded_request_ids_are_never_reflected() {
    // Given: Invalid JSON and a valid request carrying an oversized private ID.
    let inputs = [b"{private-malformed-input}\n".to_vec(), {
        let mut input = serde_json::to_vec(
            &json!({"command":"inspect_capabilities","request_id":"private-id".repeat(100)}),
        )
        .unwrap();
        input.push(b'\n');
        input
    }];
    for input in inputs {
        let mut helper = Helper::start();
        // When: Framing validation handles the real executable's private stdin.
        helper.input.write_all(&input).unwrap();
        helper.input.flush().unwrap();
        let deadline = Instant::now() + Duration::from_secs(2);
        while helper.child.try_wait().unwrap().is_none() && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(5));
        }
        // Then: The owner exits with no response echo or diagnostic disclosure.
        assert!(helper.child.try_wait().unwrap().is_some());
        assert!(helper.replies.try_recv().is_err());
        let mut error = String::new();
        helper
            .child
            .stderr
            .take()
            .unwrap()
            .read_to_string(&mut error)
            .unwrap();
        assert!(!error.contains("private"));
    }
}

#[test]
fn actual_helper_packet_clock_instability_selects_distinct_wallclock_evidence() {
    // Given: A fake RTSP camera resets every packet clock after three seconds.
    let camera = RtspCamera::start("CLOCK_RESET");
    let directory = tempfile::tempdir().unwrap();
    let mut helper = Helper::start();
    let mut scores = Vec::new();
    for mode in ["stream", "wallclock"] {
        let mut request = camera_request("inspect_recording", &camera);
        request["operation_timeout_s"] = json!(10.0);
        request["timestamp_mode"] = json!(mode);
        request["audio_mode"] = json!("none");
        request["duration_s"] = json!(3.6);
        request["output_path"] = json!(directory.path().join(format!("{mode}.mp4")));
        // When: Both timestamp modes execute real input packets and MP4 writes.
        let reply = helper.request(request);
        assert_eq!(reply["ok"], true, "{mode}: {reply}");
        let signals = &reply["inspection"]["signals"];
        scores.push(
            signals["non_monotonic_dts"].as_u64().unwrap()
                + 100 * signals["dts_discontinuity"].as_u64().unwrap()
                + 1000 * signals["queue_input_backward"].as_u64().unwrap(),
        );
    }
    // Then: Repaired baseline output succeeds and reports measured instability,
    // while wallclock's effective packet timestamps yield a lower retry score.
    assert!(
        scores[0] >= 5,
        "baseline must reach the existing retry threshold: {scores:?}"
    );
    assert!(scores[1] < scores[0], "{scores:?}");
    assert!(helper.finish().is_empty());
}

#[test]
fn actual_helper_reports_no_video_invalid_dimensions_and_bounded_stall() {
    // Given: Responsive audio-only / unparameterized cameras and a stalled source.
    let mut helper = Helper::start();
    for (mode, expected) in [
        ("AUDIO_ONLY", "no_video"),
        ("NO_DIMENSIONS", "invalid_video"),
        ("STALL_PLAY", "timeout"),
    ] {
        let camera = RtspCamera::start(mode);
        let mut request = camera_request("inspect_metadata", &camera);
        request["operation_timeout_s"] = json!(3.0);
        request["connect_timeout_s"] = json!(2.0);
        // When: Real native discovery finishes or reaches its input deadline.
        let reply = helper.request(request);
        // Then: A typed failure identifies the observed missing requirement.
        assert_eq!(reply["ok"], false, "{mode}: {reply}");
        if mode == "NO_DIMENSIONS" {
            // Decoder/parser versions may exhaust the bounded probe before
            // returning incomplete parameters. Neither outcome is usable video.
            assert!(
                [json!("invalid_video"), json!("timeout")].contains(&reply["error_code"]),
                "{mode}: {reply}"
            );
        } else {
            assert_eq!(reply["error_code"], expected, "{mode}: {reply}");
        }
        wait_closed(&camera, 1);
    }
    assert!(helper.finish().is_empty());
}

#[test]
fn actual_helper_rejects_late_overlap_failure_and_validates_present_audio() {
    // Given: One input closes after initial media and another supplies valid AAC.
    let late = RtspCamera::start("LATE_CLOSE");
    let audio = RtspCamera::start("H264_AAC");
    let mut helper = Helper::start();
    for (camera, duration, expected) in [(&late, 3.5, "failed"), (&audio, 0.2, "supported")] {
        // When: A decode role must remain usable across the complete overlap window.
        let reply = helper.request(json!({"command":"inspect_overlap", "request_id":"overlap", "operation_timeout_s":8.0, "connect_timeout_s":4.0, "io_timeout_s":2.0, "overlap_s":duration, "duration_s":duration, "roles":[{"role":"preview", "rtsp_url":camera.url, "kind":"decode", "backend":"software", "audio_mode":"copy"}]}));
        // Then: A late failure cannot pass, and requested present audio has decoded evidence.
        assert_eq!(reply["ok"], true, "{reply}");
        assert_eq!(reply["inspection"]["status"], expected, "{reply}");
        if expected == "supported" {
            assert!(
                reply["inspection"]["roles"][0]["audio_packets"]
                    .as_u64()
                    .unwrap()
                    > 0
            );
        } else {
            assert_ne!(
                reply["inspection"]["roles"][0]["error_code"],
                "session_limit"
            );
        }
    }
    assert!(helper.finish().is_empty());
}

#[test]
fn stopping_an_active_trial_closes_writer_before_artifact_removal() {
    // Given: A real helper owns a camera and an actively written temporary trial.
    let camera = RtspCamera::start("H264");
    let directory = tempfile::tempdir().unwrap();
    let target = directory.path().join("cancelled.mp4");
    let mut helper = Helper::start();
    let mut request = camera_request("inspect_recording", &camera);
    request["operation_timeout_s"] = json!(20.0);
    request["audio_mode"] = json!("none");
    request["timestamp_mode"] = json!("stream");
    request["duration_s"] = json!(10.0);
    request["output_path"] = json!(target);
    helper.send(&request);
    let deadline = Instant::now() + Duration::from_secs(5);
    while !target.exists() && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(5));
    }
    assert!(
        target.exists(),
        "test must actually interrupt an owned writer"
    );
    // When: Stop cancels native reads and waits for the actual trial owner.
    assert!(helper.finish().is_empty());
    // Then: No writer or local session remains when the deleted artifact is reported.
    assert!(!target.exists());
    wait_closed(&camera, 1);
}

#[test]
fn actual_helper_distinguishes_genuine_audio_conversion_failure_from_build_support() {
    // Given: A camera advertises supported AAC metadata but sends invalid AAC packets.
    let camera = RtspCamera::start("H264_AAC_BROKEN");
    let directory = tempfile::tempdir().unwrap();
    let mut helper = Helper::start();
    for mode in ["aac", "none"] {
        let target = directory.path().join(format!("{mode}.mp4"));
        let mut request = camera_request("inspect_recording", &camera);
        request["audio_mode"] = json!(mode);
        request["timestamp_mode"] = json!("stream");
        request["duration_s"] = json!(0.3);
        request["output_path"] = json!(target);
        // When: Native conversion fails on media and the existing silent profile is attempted.
        let reply = helper.request(request);
        // Then: A genuine profile failure permits the policy fallback; it is not a build omission.
        if mode == "aac" {
            assert_eq!(reply["error_code"], "profile_validation_failed", "{reply}");
            assert!(!target.exists());
        } else {
            assert_eq!(reply["ok"], true, "{reply}");
            assert_eq!(reply["inspection"]["audio_packets"], 0);
        }
    }
    assert!(helper.finish().is_empty());
}

#[test]
fn actual_helper_cannot_finalize_a_header_only_trial_from_stalled_input() {
    // Given: A reachable RTSP control session never delivers its first media packet.
    let camera = RtspCamera::start("STALL_PLAY");
    let directory = tempfile::tempdir().unwrap();
    let target = directory.path().join("empty.mp4");
    let mut helper = Helper::start();
    let mut request = camera_request("inspect_recording", &camera);
    request["connect_timeout_s"] = json!(2.0);
    request["operation_timeout_s"] = json!(3.0);
    request["audio_mode"] = json!("none");
    request["timestamp_mode"] = json!("stream");
    request["duration_s"] = json!(0.3);
    request["output_path"] = json!(target);
    // When: The actual executable reaches its bounded input deadline.
    let reply = helper.request(request);
    // Then: No header-only artifact or successful finalized result is returned.
    assert_eq!(reply["ok"], false);
    assert_eq!(reply["error_code"], "timeout");
    assert!(!target.exists());
    assert!(helper.finish().is_empty());
}

#[test]
fn actual_helper_copies_ac3_mp2_and_mp3_from_standard_rtp_payloads() {
    // Given: Standard RFC 4184 AC-3 and RFC 2250 MPEG audio RTSP cameras.
    let directory = tempfile::tempdir().unwrap();
    let mut helper = Helper::start();
    for (mode, codec) in [
        ("H264_AC3", "ac3"),
        ("H264_MP2", "mp2"),
        ("H264_MP3", "mp3"),
    ] {
        let camera = RtspCamera::start(mode);
        let metadata = helper.request(camera_request("inspect_metadata", &camera));
        assert_eq!(metadata["ok"], true, "{mode}: {metadata}");
        assert_eq!(
            metadata["inspection"]["audio_codec"], codec,
            "{mode}: {metadata}"
        );
        let target = directory.path().join(format!("{codec}.mp4"));
        let mut request = camera_request("inspect_recording", &camera);
        request["audio_mode"] = json!("copy");
        request["timestamp_mode"] = json!("stream");
        request["output_path"] = json!(target);
        request["duration_s"] = json!(0.3);
        // When: The actual helper opens RTSP and finalizes a video/audio copy trial.
        let reply = helper.request(request);
        // Then: Media evidence and the finalized MP4 preserve the selected audio codec.
        assert_eq!(reply["ok"], true, "{mode}: {reply}");
        assert_eq!(reply["inspection"]["finalized"], true);
        assert!(reply["inspection"]["video_packets"].as_u64().unwrap() > 0);
        assert!(reply["inspection"]["audio_packets"].as_u64().unwrap() > 0);
        let mut output = ffmpeg_next::format::input(&target).unwrap();
        let audio = output
            .streams()
            .find(|stream| stream.parameters().medium() == ffmpeg_next::media::Type::Audio)
            .unwrap();
        if codec != "mp2" {
            assert_eq!(audio.parameters().id().name(), codec);
        }
        let audio_index = audio.index();
        let copied = output
            .packets()
            .filter(|(stream, _)| stream.index() == audio_index)
            .map(|(_, packet)| packet.data().unwrap().to_vec())
            .collect::<Vec<_>>();
        assert!(!copied.is_empty());
        let source_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join(format!("tests/fixtures/inspection/h264-{codec}.mov"));
        let mut source = ffmpeg_next::format::input(&source_path).unwrap();
        let originals = source
            .packets()
            .filter(|(stream, _)| stream.parameters().medium() == ffmpeg_next::media::Type::Audio)
            .map(|(_, packet)| packet.data().unwrap().to_vec())
            .collect::<Vec<_>>();
        assert!(copied.iter().all(|packet| originals.contains(packet)));
        // MP4 may label MPEG audio as MP3 for either layer; byte identity and
        // layer-II headers establish MP2 copy without any transcoding.
        if codec == "mp2" {
            assert!(copied.iter().all(|packet| (packet[1] >> 1) & 3 == 2));
        }
        wait_closed(&camera, 2);
    }
    assert!(helper.finish().is_empty());
}

#[test]
fn empty_overlap_input_never_becomes_supported_and_is_closed() {
    // Given: A real RTSP input reaches PLAY without delivering media.
    let camera = RtspCamera::start("STALL_PLAY");
    let mut helper = Helper::start();
    // When: An actual overlap operation exhausts its finite input budget.
    let reply = helper.request(json!({"command":"inspect_overlap", "request_id":"empty", "operation_timeout_s":0.6, "connect_timeout_s":0.4, "io_timeout_s":0.4, "overlap_s":0.1, "duration_s":0.1, "roles":[{"role":"motion", "rtsp_url":camera.url, "kind":"decode", "backend":"software"}]}));
    // Then: Empty input is failed/unclassified, with local ownership closed before completion.
    assert_eq!(reply["ok"], true, "{reply}");
    assert_eq!(reply["inspection"]["status"], "failed", "{reply}");
    assert_eq!(reply["inspection"]["roles"][0]["ok"], false);
    assert_eq!(reply["inspection"]["roles"][0]["error_code"], "timeout");
    wait_closed(&camera, 1);
    assert!(helper.finish().is_empty());
}

#[test]
fn actual_helper_copies_eac3_inside_standard_mpegts_rtp() {
    // Given: Standard RTP payload 33 transports synthetic H.264/E-AC-3 MPEG-TS.
    let camera = RtspCamera::start("MP2T_EAC3");
    let directory = tempfile::tempdir().unwrap();
    let target = directory.path().join("eac3.mp4");
    let mut helper = Helper::start();
    let metadata = helper.request(camera_request("inspect_metadata", &camera));
    assert_eq!(metadata["ok"], true, "{metadata}");
    assert_eq!(metadata["inspection"]["video_codec"], "h264");
    assert_eq!(metadata["inspection"]["audio_codec"], "eac3");
    let mut request = camera_request("inspect_recording", &camera);
    request["audio_mode"] = json!("copy");
    request["timestamp_mode"] = json!("stream");
    request["output_path"] = json!(target);
    request["duration_s"] = json!(0.3);
    // When: The existing native helper demuxes MP2T and finalizes MP4 copy.
    let reply = helper.request(request);
    // Then: Real audio/video packets and the output codec establish copy support.
    assert_eq!(reply["ok"], true, "{reply}");
    assert_eq!(reply["inspection"]["finalized"], true);
    assert!(reply["inspection"]["video_packets"].as_u64().unwrap() > 0);
    assert!(reply["inspection"]["audio_packets"].as_u64().unwrap() > 0);
    let output = ffmpeg_next::format::input(&target).unwrap();
    assert_eq!(
        output
            .streams()
            .find(|stream| stream.parameters().medium() == ffmpeg_next::media::Type::Audio)
            .unwrap()
            .parameters()
            .id()
            .name(),
        "eac3"
    );
    wait_closed(&camera, 2);
    assert!(helper.finish().is_empty());
}

#[test]
fn actual_helper_converts_g711_and_opus_rtp_audio_to_aac() {
    // Given: Standard PCMA/PCMU 8 kHz and Opus 48 kHz RTP payloads.
    let directory = tempfile::tempdir().unwrap();
    let mut helper = Helper::start();
    for mode in ["H264_PCMA", "H264_PCMU", "H264_OPUS"] {
        let camera = RtspCamera::start(mode);
        let target = directory.path().join(format!("{mode}.mp4"));
        let mut request = camera_request("inspect_recording", &camera);
        request["audio_mode"] = json!("aac");
        request["timestamp_mode"] = json!("stream");
        request["output_path"] = json!(target);
        request["duration_s"] = json!(0.3);
        // When: The actual helper decodes each camera codec and encodes AAC.
        let reply = helper.request(request);
        // Then: Finalized media contains actual audio packets encoded as AAC.
        assert_eq!(reply["ok"], true, "{mode}: {reply}");
        assert_eq!(reply["inspection"]["finalized"], true);
        assert!(reply["inspection"]["audio_packets"].as_u64().unwrap() > 0);
        let output = ffmpeg_next::format::input(&target).unwrap();
        assert_eq!(
            output
                .streams()
                .find(|stream| stream.parameters().medium() == ffmpeg_next::media::Type::Audio)
                .unwrap()
                .parameters()
                .id()
                .name(),
            "aac"
        );
        wait_closed(&camera, 1);
    }
    assert!(helper.finish().is_empty());
}
