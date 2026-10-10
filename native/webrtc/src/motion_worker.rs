//! Always-on camera motion input. Pixels stay here; Python consumes typed observations.
use crate::decode::{GrayDecoder, GrayFrame};
use crate::motion::{MotionConfig, MotionDetector};
use crate::protocol::{Control, controls, output};
use crate::rtp::Event;
use crate::rtsp::RtspSource;
use mio::{Events, Poll, Token, Waker};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Settings {
    pixel_threshold: u64,
    min_changed_pct: f64,
    blur_kernel: usize,
    recording_sensitivity_factor: f64,
}

#[derive(Deserialize)]
struct Request {
    request_id: String,
    #[serde(flatten)]
    operation: Operation,
}

#[derive(Deserialize)]
#[serde(tag = "command", rename_all = "snake_case")]
enum Operation {
    Start {
        rtsp_url: String,
        motion_config: Settings,
        frame_queue_size: usize,
        connect_timeout_s: f64,
        io_timeout_s: f64,
    },
    ReadMotion {
        threshold: f64,
        wait_timeout_s: f64,
    },
    DiscardFrame {
        wait_timeout_s: f64,
    },
    Status,
    Stop,
}

#[derive(Serialize)]
struct Observation {
    changed_pixels: usize,
    changed_pct: f64,
    motion: bool,
}

#[derive(Default, Serialize)]
struct Reply {
    #[serde(skip_serializing_if = "Option::is_none")]
    request_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    event: Option<&'static str>,
    ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    error_code: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    observation: Option<Observation>,
    frame_available: bool,
}

impl Reply {
    fn success(request_id: String) -> Self {
        Self {
            request_id: Some(request_id),
            ok: true,
            ..Self::default()
        }
    }
    fn error(request_id: String, code: &'static str) -> Self {
        Self {
            request_id: Some(request_id),
            error_code: Some(code),
            ..Self::default()
        }
    }
}

#[derive(Default)]
struct Frames {
    ready: VecDeque<GrayFrame>,
    received: u64,
    failure: Option<&'static str>,
}

impl Frames {
    fn push(&mut self, frame: GrayFrame, capacity: usize) {
        // Preserve Python's drop-oldest BEFORE detection, including the baseline.
        if self.ready.len() == capacity {
            self.ready.pop_front();
        }
        self.ready.push_back(frame);
        self.received = self.received.saturating_add(1);
    }
    fn take(&mut self) -> std::result::Result<Option<GrayFrame>, &'static str> {
        if let Some(code) = self.failure {
            return Err(code);
        }
        Ok(self.ready.pop_front())
    }
}

struct Worker {
    frames: Arc<Mutex<Frames>>,
    detector: MotionDetector,
    stop: Arc<AtomicBool>,
    finished: mpsc::Receiver<()>,
    thread: Option<JoinHandle<()>>,
    notified: u64,
}

impl Worker {
    fn start(
        url: String,
        settings: Settings,
        capacity: usize,
        connect: f64,
        io: f64,
        waker: Arc<Waker>,
    ) -> std::result::Result<Self, &'static str> {
        if capacity == 0
            || [connect, io]
                .iter()
                .any(|v| !v.is_finite() || *v <= 0.0 || *v > 120.0)
        {
            return Err("invalid_motion_config");
        }
        let config = MotionConfig {
            pixel_threshold: settings.pixel_threshold,
            min_changed_pct: settings.min_changed_pct,
            blur_kernel: settings.blur_kernel,
            recording_sensitivity_factor: settings.recording_sensitivity_factor,
        };
        if !config.min_changed_pct.is_finite()
            || !config.recording_sensitivity_factor.is_finite()
            || !config.recording_threshold().is_finite()
        {
            return Err("invalid_motion_config");
        }
        let detector = MotionDetector::new(config).map_err(|_| "invalid_motion_config")?;
        let frames = Arc::new(Mutex::new(Frames::default()));
        let stop = Arc::new(AtomicBool::new(false));
        let (sender, finished) = mpsc::sync_channel(1);
        let thread_frames = Arc::clone(&frames);
        let thread_stop = Arc::clone(&stop);
        let worker = thread::Builder::new()
            .name("motion-decode".into())
            .spawn(move || {
                let result = prepare(
                    url,
                    capacity,
                    Duration::from_secs_f64(connect),
                    Duration::from_secs_f64(io),
                    &thread_frames,
                    &thread_stop,
                    &waker,
                );
                if let Err(code) = result
                    && let Ok(mut frames) = thread_frames.lock()
                {
                    frames.failure = Some(code);
                }
                let _ = waker.wake();
                let _ = sender.send(());
            })
            .map_err(|_| "motion_worker_failed")?;
        Ok(Self {
            frames,
            detector,
            stop,
            finished,
            thread: Some(worker),
            notified: 0,
        })
    }

    fn notify(&mut self) -> Result<()> {
        let count = self
            .frames
            .lock()
            .map_err(|_| "motion_worker_failed")?
            .received;
        while self.notified < count {
            output(&Reply {
                event: Some("frame"),
                ..Reply::default()
            })?;
            self.notified += 1;
        }
        Ok(())
    }

    fn consume(
        &mut self,
        threshold: Option<f64>,
    ) -> std::result::Result<Option<Observation>, &'static str> {
        let frame = self
            .frames
            .lock()
            .map_err(|_| "motion_worker_failed")?
            .take()?;
        let Some(frame) = frame else {
            return Ok(None);
        };
        let observation = if let Some(threshold) = threshold {
            self.detector
                .detect(&frame.data, frame.width, frame.height, Some(threshold))
                .map_err(|_| "invalid_motion_frame")?
        } else {
            // Readiness discards must not establish or advance a detector baseline.
            self.detector.observation()
        };
        Ok(Some(Observation {
            changed_pixels: observation.changed_pixels,
            changed_pct: observation.changed_pct,
            motion: observation.motion,
        }))
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        self.detector.reset();
        if self
            .finished
            .recv_timeout(Duration::from_millis(500))
            .is_ok()
            && let Some(worker) = self.thread.take()
        {
            let _ = worker.join();
        }
    }
}

fn prepare(
    url: String,
    capacity: usize,
    connect: Duration,
    io: Duration,
    frames: &Mutex<Frames>,
    stop: &AtomicBool,
    main_waker: &Waker,
) -> std::result::Result<(), &'static str> {
    let mut poll = Poll::new().map_err(|_| "motion_worker_failed")?;
    let waker =
        Arc::new(Waker::new(poll.registry(), Token(0)).map_err(|_| "motion_worker_failed")?);
    let source = RtspSource::start(url, connect, io, waker)?;
    let mut decoder = GrayDecoder::new()?;
    let mut events = Events::with_capacity(4);
    while !stop.load(Ordering::Acquire) {
        // Bound each batch so an always-readable camera cannot starve cancellation.
        for _ in 0..32 {
            if stop.load(Ordering::Acquire) {
                return Ok(());
            }
            match source.try_recv()? {
                Some(Event::Frame(frame)) => {
                    let mut failed = false;
                    decoder.push(&frame, |frame| {
                        if let Ok(mut frames) = frames.lock() {
                            frames.push(frame, capacity);
                        } else {
                            failed = true;
                        }
                        let _ = main_waker.wake();
                    })?;
                    if failed {
                        return Err("motion_worker_failed");
                    }
                }
                Some(Event::Info { .. } | Event::Audio { .. }) => {}
                None => break,
            }
        }
        poll.poll(&mut events, Some(Duration::from_millis(20)))
            .map_err(|_| "motion_worker_failed")?;
    }
    Ok(())
}

struct Pending {
    request_id: String,
    deadline: Instant,
    threshold: Option<f64>,
}

pub fn run() -> Result<()> {
    let mut poll = Poll::new()?;
    let waker = Arc::new(Waker::new(poll.registry(), Token(0))?);
    let controls = controls(Arc::clone(&waker), |bytes| {
        let request: Request = serde_json::from_slice(bytes).ok()?;
        (request.request_id.len() <= 128).then_some(request)
    });
    let mut events = Events::with_capacity(4);
    let mut worker: Option<Worker> = None;
    let mut pending: Option<Pending> = None;
    output(&Reply {
        event: Some("ready"),
        ok: true,
        ..Reply::default()
    })?;
    loop {
        for _ in 0..16 {
            let control = match controls.try_recv() {
                Ok(control) => control,
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => return Ok(()),
            };
            let Control::Request(request) = control else {
                return Ok(());
            };
            let id = request.request_id;
            match request.operation {
                Operation::Start {
                    rtsp_url,
                    motion_config,
                    frame_queue_size,
                    connect_timeout_s,
                    io_timeout_s,
                } => {
                    if worker.is_some() {
                        output(&Reply::error(id, "motion_already_started"))?;
                    } else {
                        match Worker::start(
                            rtsp_url,
                            motion_config,
                            frame_queue_size,
                            connect_timeout_s,
                            io_timeout_s,
                            Arc::clone(&waker),
                        ) {
                            Ok(value) => {
                                worker = Some(value);
                                output(&Reply::success(id))?;
                            }
                            Err(code) => output(&Reply::error(id, code))?,
                        }
                    }
                }
                Operation::ReadMotion {
                    threshold,
                    wait_timeout_s,
                    ..
                } => {
                    queue_read(
                        &mut pending,
                        worker.is_some(),
                        id,
                        wait_timeout_s,
                        Some(threshold),
                    )?;
                }
                Operation::DiscardFrame { wait_timeout_s, .. } => {
                    queue_read(&mut pending, worker.is_some(), id, wait_timeout_s, None)?;
                }
                Operation::Status => output(&Reply::success(id))?,
                Operation::Stop => {
                    if let Some(read) = pending.take() {
                        output(&Reply::error(read.request_id, "motion_stopped"))?;
                    }
                    output(&Reply::success(id))?;
                    return Ok(());
                }
            }
        }
        if let Some(worker) = worker.as_mut() {
            worker.notify()?;
            if let Some(read) = pending.as_ref() {
                let response = match worker.consume(read.threshold) {
                    Ok(Some(observation)) => {
                        let mut reply = Reply::success(read.request_id.clone());
                        reply.frame_available = true;
                        if read.threshold.is_some() {
                            reply.observation = Some(observation);
                        }
                        Some(reply)
                    }
                    Ok(None) if Instant::now() >= read.deadline => {
                        Some(Reply::success(read.request_id.clone()))
                    }
                    Ok(None) => None,
                    Err(code) => Some(Reply::error(read.request_id.clone(), code)),
                };
                if let Some(response) = response {
                    output(&response)?;
                    pending = None;
                }
            }
        }
        let timeout = pending
            .as_ref()
            .map(|p| p.deadline.saturating_duration_since(Instant::now()))
            .unwrap_or(Duration::from_millis(100))
            .min(Duration::from_millis(100));
        poll.poll(&mut events, Some(timeout))?;
    }
}

fn queue_read(
    pending: &mut Option<Pending>,
    running: bool,
    request_id: String,
    timeout: f64,
    threshold: Option<f64>,
) -> Result<()> {
    if !timeout.is_finite()
        || !(0.0..=120.0).contains(&timeout)
        || threshold.is_some_and(|v| !v.is_finite() || v < 0.0)
    {
        output(&Reply::error(request_id, "invalid_motion_request"))?;
    } else if !running {
        output(&Reply::error(request_id, "motion_not_started"))?;
    } else if pending.is_some() {
        output(&Reply::error(request_id, "motion_read_busy"))?;
    } else {
        *pending = Some(Pending {
            request_id,
            deadline: Instant::now() + Duration::from_secs_f64(timeout),
            threshold,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn worker() -> Worker {
        let (_, finished) = mpsc::sync_channel(1);
        Worker {
            frames: Arc::new(Mutex::new(Frames::default())),
            detector: MotionDetector::new(MotionConfig {
                pixel_threshold: 10,
                min_changed_pct: 1.0,
                blur_kernel: 0,
                recording_sensitivity_factor: 2.0,
            })
            .unwrap(),
            stop: Arc::new(AtomicBool::new(false)),
            finished,
            thread: None,
            notified: 0,
        }
    }

    fn frame(value: u8) -> GrayFrame {
        GrayFrame {
            data: vec![value; 4],
            width: 2,
            height: 2,
        }
    }

    #[test]
    fn overflow_drops_frames_before_the_previous_frame_comparison() {
        // Given: A black baseline and a single-slot queue consumed by Python.
        let mut worker = worker();
        worker.frames.lock().unwrap().push(frame(0), 1);
        assert!(!worker.consume(Some(1.0)).unwrap().unwrap().motion);
        // When: A white frame is displaced by black before the next observation request.
        worker.frames.lock().unwrap().push(frame(255), 1);
        worker.frames.lock().unwrap().push(frame(0), 1);
        let result = worker.consume(Some(1.0)).unwrap().unwrap();
        // Then: Only consumed black frames are compared; recording is not triggered.
        assert_eq!(result.changed_pixels, 0);
        assert_eq!(result.changed_pct, 0.0);
        assert!(!result.motion);
    }

    #[test]
    fn readiness_discards_and_unconsumed_frames_do_not_establish_a_baseline() {
        // Given: A black readiness frame during source reconnect.
        let mut worker = worker();
        worker.frames.lock().unwrap().push(frame(0), 1);
        assert!(worker.consume(None).unwrap().is_some());
        // When: The next consumed frame is white with zero motion threshold.
        worker.frames.lock().unwrap().push(frame(255), 1);
        let baseline = worker.consume(Some(0.0)).unwrap().unwrap();
        worker.frames.lock().unwrap().push(frame(255), 1);
        let following = worker.consume(Some(0.0)).unwrap().unwrap();
        // Then: Readiness did not seed detection; only the second white frame can be motion.
        assert!(!baseline.motion);
        assert_eq!(baseline.changed_pixels, 0);
        assert!(following.motion);
    }

    #[test]
    fn source_failure_is_not_hidden_by_prepared_frames() {
        // Given: Prepared input followed by a terminal decoder/network failure.
        let mut worker = worker();
        let mut frames = worker.frames.lock().unwrap();
        frames.push(frame(255), 1);
        frames.failure = Some("rtsp_timeout");
        drop(frames);
        // When: Python asks for its next observation.
        let result = worker.consume(Some(1.0));
        // Then: It receives the stable failure instead of stale media.
        assert!(matches!(result, Err("rtsp_timeout")));
    }
}
