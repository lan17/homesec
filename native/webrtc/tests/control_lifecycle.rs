//! Slow-parent control output must not keep an actual camera session alive.

#[path = "support/rtsp_camera.rs"]
mod rtsp_camera;

use rtsp_camera::RtspCamera;
use serde_json::{Value, json};
use std::io::{BufRead, BufReader, Read, Write};
use std::os::fd::OwnedFd;
use std::os::unix::net::UnixStream;
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

const WAIT: Duration = Duration::from_secs(10);

struct Helper {
    child: Child,
    input: Option<UnixStream>,
    read_requests: Option<Sender<()>>,
    replies: Receiver<Value>,
    reader: Option<JoinHandle<()>>,
}

impl Helper {
    fn start() -> Self {
        let (input, child_input) = UnixStream::pair().unwrap();
        input.set_nonblocking(true).unwrap();
        let mut child = Command::new(env!("CARGO_BIN_EXE_homesec-webrtc"))
            .arg("--motion")
            .stdin(Stdio::from(OwnedFd::from(child_input)))
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let output = child.stdout.take().unwrap();
        let (requests, read_requests) = mpsc::channel();
        let (sender, replies) = mpsc::channel();
        // Read only on request. Once the parent abandons output, this thread
        // keeps the real pipe open without draining it in the background.
        let reader = thread::spawn(move || {
            let mut output = BufReader::new(output);
            while read_requests.recv().is_ok() {
                let mut line = String::new();
                if output.read_line(&mut line).unwrap_or(0) == 0 {
                    break;
                }
                let Ok(value) = serde_json::from_str(&line) else {
                    break;
                };
                if sender.send(value).is_err() {
                    break;
                }
            }
        });
        let mut helper = Self {
            child,
            input: Some(input),
            read_requests: Some(requests),
            replies,
            reader: Some(reader),
        };
        assert_eq!(helper.receive(WAIT)["event"], "ready");
        helper
    }

    fn receive(&mut self, timeout: Duration) -> Value {
        self.read_requests.as_ref().unwrap().send(()).unwrap();
        self.replies.recv_timeout(timeout).unwrap()
    }

    fn request(&mut self, request: Value) -> Value {
        let id = request["request_id"].as_str().unwrap().to_owned();
        let mut payload = serde_json::to_vec(&request).unwrap();
        payload.push(b'\n');
        let deadline = Instant::now() + WAIT;
        let input = self.input.as_mut().unwrap();
        let mut remaining = payload.as_slice();
        while !remaining.is_empty() {
            assert!(Instant::now() < deadline, "helper input timed out");
            match input.write(remaining) {
                Ok(0) => panic!("helper input closed"),
                Ok(count) => remaining = &remaining[count..],
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                    thread::sleep(Duration::from_millis(5));
                }
                Err(error) => panic!("helper input failed: {error}"),
            }
        }
        loop {
            let reply = self.receive(deadline.saturating_duration_since(Instant::now()));
            if reply["request_id"] == id {
                return reply;
            }
        }
    }

    fn fill_output_and_close_input(&mut self) -> usize {
        let payload = b"{\"command\":\"status\",\"request_id\":\"flood\"}\n".repeat(10_000);
        let deadline = Instant::now() + Duration::from_secs(2);
        let mut written = 0;
        while written < payload.len() && Instant::now() < deadline {
            if self.child.try_wait().unwrap().is_some() {
                break;
            }
            match self.input.as_mut().unwrap().write(&payload[written..]) {
                Ok(0) => break,
                Ok(count) => written += count,
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                    thread::sleep(Duration::from_millis(5));
                }
                Err(error) if error.kind() == std::io::ErrorKind::BrokenPipe => break,
                Err(error) => panic!("helper input failed: {error}"),
            }
        }
        self.input.take();
        written
    }

    fn wait_for_exit(&mut self) -> ExitStatus {
        let deadline = Instant::now() + Duration::from_secs(2);
        loop {
            if let Some(status) = self.child.try_wait().unwrap() {
                return status;
            }
            assert!(
                Instant::now() < deadline,
                "blocked output prevented shutdown"
            );
            thread::sleep(Duration::from_millis(5));
        }
    }

    fn stderr(&mut self) -> Vec<u8> {
        let mut bytes = Vec::new();
        self.child
            .stderr
            .take()
            .unwrap()
            .read_to_end(&mut bytes)
            .unwrap();
        bytes
    }
}

impl Drop for Helper {
    fn drop(&mut self) {
        self.input.take();
        if self.child.try_wait().ok().flatten().is_none() {
            let _ = self.child.kill();
        }
        let _ = self.child.wait();
        self.read_requests.take();
        if let Some(reader) = self.reader.take() {
            let _ = reader.join();
        }
    }
}

#[test]
fn blocked_parent_output_releases_motion_camera_without_draining_replies() {
    // Given: An actual camera session has delivered motion while its URL carries private credentials.
    let camera = RtspCamera::start("H264");
    let url = camera
        .url
        .replacen("rtsp://", "rtsp://private-user:secret-password@", 1);
    let mut helper = Helper::start();
    let start = helper.request(json!({
        "command": "start", "request_id": "start", "rtsp_url": url,
        "motion_config": {
            "pixel_threshold": 18, "min_changed_pct": 1.0, "blur_kernel": 5,
            "recording_sensitivity_factor": 2.0,
        },
        "frame_queue_size": 20, "connect_timeout_s": 30.0, "io_timeout_s": 30.0,
    }));
    assert_eq!(start["ok"], true);
    let observation = helper.request(json!({
        "command": "read_motion", "request_id": "observation",
        "threshold": 0.5, "wait_timeout_s": 5.0,
    }));
    assert_eq!(observation["ok"], true);
    assert!(observation["observation"].is_object());
    assert_eq!(camera.count("PLAY"), 1);

    // When: The parent stops draining replies, fills the output pipe, then closes its control input.
    let written = helper.fill_output_and_close_input();
    let status = helper.wait_for_exit();
    let stderr = helper.stderr();

    // Then: A bounded transport failure exits the worker and releases the camera without exposing private data.
    assert!(written > 0, "status flood was not written");
    assert!(!status.success());
    assert_eq!(stderr, b"homesec-webrtc: worker_failed\n");
    let deadline = Instant::now() + Duration::from_secs(2);
    while camera.count("DISCONNECTED") == 0 {
        assert!(
            Instant::now() < deadline,
            "camera session survived helper exit"
        );
        thread::sleep(Duration::from_millis(5));
    }
    assert_eq!(camera.count("TEARDOWN"), 1);
    assert_eq!(camera.count("DISCONNECTED"), 1);
}
