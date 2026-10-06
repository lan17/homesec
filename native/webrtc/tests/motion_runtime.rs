//! Exercise always-on motion through the executable and native RTSP boundary.

#[path = "support/ffmpeg_reference.rs"]
mod ffmpeg_reference;
#[allow(dead_code)]
#[path = "../src/motion.rs"]
mod motion;
#[path = "support/rtsp_camera.rs"]
mod rtsp_camera;

use motion::{MotionConfig, MotionDetector};
use rtsp_camera::RtspCamera;
use serde_json::{Value, json};
use std::collections::VecDeque;
use std::io::{BufRead, BufReader, Read, Write};
use std::process::{Child, ChildStdin, Command, ExitStatus, Stdio};
use std::sync::mpsc::{self, Receiver};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

const WAIT: Duration = Duration::from_secs(10);

struct Helper {
    child: Child,
    stdin: Option<ChildStdin>,
    replies: Receiver<Value>,
    pending: VecDeque<Value>,
    history: Vec<Value>,
    next_request: usize,
    stdout: Option<JoinHandle<()>>,
    stderr: Option<JoinHandle<Vec<u8>>>,
}

impl Helper {
    fn start() -> Self {
        let mut child = Command::new(env!("CARGO_BIN_EXE_homesec-webrtc"))
            .arg("--motion")
            // Native motion must work without an executable FFmpeg on PATH.
            .env("PATH", "")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let stdin = child.stdin.take().unwrap();
        let output = child.stdout.take().unwrap();
        let mut error = child.stderr.take().unwrap();
        let (sender, replies) = mpsc::channel();
        let stdout = thread::spawn(move || {
            for line in BufReader::new(output).lines() {
                let Ok(line) = line else { break };
                let value =
                    serde_json::from_str(&line).expect("motion helper emitted invalid JSON");
                if sender.send(value).is_err() {
                    break;
                }
            }
        });
        let stderr = thread::spawn(move || {
            let mut bytes = Vec::new();
            error.read_to_end(&mut bytes).unwrap();
            bytes
        });
        let mut helper = Self {
            child,
            stdin: Some(stdin),
            replies,
            pending: VecDeque::new(),
            history: Vec::new(),
            next_request: 0,
            stdout: Some(stdout),
            stderr: Some(stderr),
        };
        let ready = helper.receive(WAIT);
        assert_eq!(ready["event"], "ready", "{ready}");
        assert_eq!(ready["ok"], true, "{ready}");
        assert!(ready.get("video_port").is_none());
        assert!(ready.get("audio_port").is_none());
        helper
    }

    fn receive(&mut self, timeout: Duration) -> Value {
        let reply = self
            .replies
            .recv_timeout(timeout)
            .expect("motion helper timed out");
        self.history.push(reply.clone());
        reply
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

    fn reply(&mut self, id: &str) -> Value {
        if let Some(index) = self
            .pending
            .iter()
            .position(|reply| reply["request_id"] == id)
        {
            return self.pending.remove(index).unwrap();
        }
        let deadline = Instant::now() + WAIT;
        loop {
            let reply = self.receive(deadline.saturating_duration_since(Instant::now()));
            if reply["request_id"] == id {
                return reply;
            }
            if reply.get("request_id").is_some() {
                self.pending.push_back(reply);
            }
        }
    }

    fn request(&mut self, request: Value) -> Value {
        let id = self.send(request);
        self.reply(&id)
    }

    fn success(&mut self, request: Value) -> Value {
        let reply = self.request(request);
        assert_eq!(reply["ok"], true, "{reply}");
        reply
    }

    fn start_camera(&mut self, url: &str) {
        self.success(json!({
            "command": "start",
            "rtsp_url": url,
            "motion_config": {
                "pixel_threshold": 18,
                "min_changed_pct": 1.0,
                "blur_kernel": 5,
                "recording_sensitivity_factor": 2.0,
            },
            "frame_queue_size": 64,
            "connect_timeout_s": 30.0,
            "io_timeout_s": 30.0,
        }));
    }

    fn observation(&mut self) -> Value {
        let reply = self.success(json!({
            "command": "read_motion",
            "threshold": 0.5,
            "wait_timeout_s": 5.0,
        }));
        assert_eq!(reply["frame_available"], true, "{reply}");
        assert!(reply["observation"].is_object(), "{reply}");
        reply["observation"].clone()
    }

    fn finish(&mut self) -> Vec<u8> {
        self.stdin.take();
        let status = wait_child(&mut self.child, WAIT);
        assert!(status.success(), "motion helper failed: {status}");
        self.stdout.take().unwrap().join().unwrap();
        self.stderr.take().unwrap().join().unwrap()
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
        if let Some(reader) = self.stdout.take() {
            let _ = reader.join();
        }
        if let Some(reader) = self.stderr.take() {
            let _ = reader.join();
        }
    }
}

fn wait_child(child: &mut Child, timeout: Duration) -> ExitStatus {
    let deadline = Instant::now() + timeout;
    loop {
        if let Some(status) = child.try_wait().unwrap() {
            return status;
        }
        if Instant::now() >= deadline {
            child.kill().unwrap();
            child.wait().unwrap();
            panic!("motion helper did not exit within {timeout:?}");
        }
        thread::sleep(Duration::from_millis(5));
    }
}

fn wait_until(complete: impl Fn() -> bool) {
    let deadline = Instant::now() + WAIT;
    while !complete() {
        assert!(Instant::now() < deadline, "camera condition timed out");
        thread::sleep(Duration::from_millis(5));
    }
}

fn prepared_fixture() -> Vec<u8> {
    let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/baseline-160x120.h264");
    ffmpeg_reference::prepare_gray(&fixture, 10, 10)
}

#[test]
fn native_motion_runs_without_preview_or_ffmpeg_executable_and_matches_prepared_observations() {
    // Given: CLI-prepared synthetic input and the existing parity-tested motion settings.
    let prepared = prepared_fixture();
    let mut detector = MotionDetector::new(MotionConfig {
        pixel_threshold: 18,
        min_changed_pct: 1.0,
        blur_kernel: 5,
        recording_sensitivity_factor: 2.0,
    })
    .unwrap();
    let camera = RtspCamera::start("H264");
    let mut helper = Helper::start();
    helper.start_camera(&camera.url);

    // When: Always-on motion consumes camera frames without any preview session or FFmpeg executable.
    for frame in prepared.as_chunks::<{ 320 * 240 }>().0.iter().take(6) {
        let expected = detector.detect(frame, 320, 240, Some(0.5)).unwrap();
        let actual = helper.observation();
        // Then: Frame order, preparation, configuration and observations match the established algorithm.
        assert_eq!(actual["changed_pixels"], expected.changed_pixels);
        assert_eq!(
            actual["changed_pct"].as_f64().unwrap(),
            expected.changed_pct
        );
        assert_eq!(actual["motion"], expected.motion);
    }
    assert_eq!(camera.count("DESCRIBE"), 1);
    assert_eq!(camera.count("PLAY"), 1);
    assert!(helper.history.iter().any(|reply| reply["event"] == "frame"));
    helper.success(json!({"command": "stop"}));
    assert!(helper.finish().is_empty());
    wait_until(|| camera.count("DISCONNECTED") == 1);
    assert_eq!(camera.count("TEARDOWN"), 1);
}

#[test]
fn parent_eof_releases_the_active_motion_camera() {
    // Given: The executable has an active camera and has delivered a motion observation.
    let camera = RtspCamera::start("H264");
    let mut helper = Helper::start();
    helper.start_camera(&camera.url);
    helper.observation();

    // When: The parent closes stdin instead of sending an explicit stop command.
    let stderr = helper.finish();

    // Then: The process exits cleanly and tears down the actual RTSP connection.
    assert!(stderr.is_empty());
    wait_until(|| camera.count("DISCONNECTED") == 1);
    assert_eq!(camera.count("TEARDOWN"), 1);
}

#[test]
fn unsupported_codec_returns_only_a_sanitized_failure() {
    // Given: A camera advertises unsupported H.265 and the URL contains private credentials.
    let camera = RtspCamera::start("H265");
    let url = camera
        .url
        .replacen("rtsp://", "rtsp://private-user:secret-password@", 1);
    let mut helper = Helper::start();
    helper.start_camera(&url);

    // When: The parent requests its first motion observation.
    let failure = helper.request(json!({
        "command": "read_motion", "threshold": 0.5, "wait_timeout_s": 5.0,
    }));
    helper.success(json!({"command": "stop"}));
    let stderr = helper.finish();

    // Then: A stable refusal reaches the fallback boundary without credentials, URL or codec diagnostics.
    assert_eq!(failure["ok"], false, "{failure}");
    assert_eq!(failure["error_code"], "unsupported_codec", "{failure}");
    assert!(failure.get("observation").is_none());
    assert!(stderr.is_empty());
    let history = serde_json::to_string(&helper.history).unwrap();
    for private in ["private-user", "secret-password", "rtsp://", &camera.url] {
        assert!(!history.contains(private));
    }
    assert_eq!(camera.count("PLAY"), 0);
    wait_until(|| camera.count("DISCONNECTED") == 1);
}

#[test]
fn stop_cancels_a_pending_read_during_stalled_startup_or_media() {
    // Given: Startup or first media is stalled, with timeouts much longer than shutdown's deadline.
    for (mode, reached) in [("STALL", "DESCRIBE"), ("STALL_PLAY", "PLAY")] {
        let camera = RtspCamera::start(mode);
        let mut helper = Helper::start();
        helper.start_camera(&camera.url);
        wait_until(|| camera.count(reached) == 1);
        let read = helper.send(json!({
            "command": "read_motion", "threshold": 0.5, "wait_timeout_s": 60.0,
        }));
        helper.success(json!({"command": "status"}));

        // When: The parent stops while the observation is waiting on the camera.
        let started = Instant::now();
        let stop = helper.send(json!({"command": "stop"}));
        let cancelled = helper.reply(&read);
        let stopped = helper.reply(&stop);
        let stderr = helper.finish();

        // Then: The pending read is refused and all camera work ends without waiting for network timeouts.
        assert_eq!(cancelled["ok"], false, "{cancelled}");
        assert_eq!(cancelled["error_code"], "motion_stopped", "{cancelled}");
        assert_eq!(stopped["ok"], true, "{stopped}");
        assert!(
            started.elapsed() < Duration::from_secs(3),
            "{mode}: slow shutdown"
        );
        assert!(stderr.is_empty());
        wait_until(|| camera.count("DISCONNECTED") == 1);
        if mode == "STALL_PLAY" {
            assert_eq!(camera.count("TEARDOWN"), 1);
        }
    }
}
