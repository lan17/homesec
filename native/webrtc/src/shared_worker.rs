//! Source-owned compressed packet fanout. Decode, disk I/O and WebRTC each
//! consume independent bounded queues; none runs on the camera input thread.
use crate::camera_input::{CameraInput, Event as CameraEvent, Metadata};
use crate::decode::{GrayDecoder, GrayFrame};
use crate::motion::{MotionConfig, MotionDetector};
use crate::protocol::{Operation, Reply, Request};
use crate::recording::{EncodedPacket, Mp4Recorder, StreamConfig, Track};
use crate::rtp::{EncodedFrame, Event};
use ffmpeg::Rescale;
use ffmpeg_next as ffmpeg;
use mio::Waker;
use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

const PACKETS: usize = 16;
const MAX_STREAMS: usize = 2;
const RETIREMENT_TIMEOUT: Duration = Duration::from_millis(500);
const MAX_RECORDINGS: usize = 2; // Old and new clip overlap at policy-controlled rotation.
type Failure = Arc<Mutex<Option<&'static str>>>;
type Result<T> = std::result::Result<T, &'static str>;

struct Consumer {
    sender: mpsc::SyncSender<Packet>,
    failure: Failure,
    audio: bool,
}
struct Packet {
    track: Track,
    packet: ffmpeg::Packet,
}
#[derive(Default)]
struct Hub {
    metadata: Option<Metadata>,
    consumers: HashMap<String, Consumer>,
}
impl Hub {
    fn deliver(&mut self, event: CameraEvent) {
        match event {
            CameraEvent::Ready(metadata) => self.metadata = Some(metadata),
            CameraEvent::Packet { track, packet } => {
                // av_packet_ref clones share immutable compressed storage. A
                // full consumer fails independently; camera reads never wait.
                for consumer in self.consumers.values() {
                    if track == Track::Audio && !consumer.audio {
                        continue;
                    }
                    if consumer
                        .sender
                        .try_send(Packet {
                            track,
                            packet: packet.clone(),
                        })
                        .is_err()
                    {
                        fail(&consumer.failure, "media_queue_overflow");
                    }
                }
            }
        }
    }
}
struct Input {
    source: CameraInput,
    hub: Arc<Mutex<Hub>>,
}
#[derive(Default)]
struct Frames {
    ready: VecDeque<GrayFrame>,
    received: u64,
}
struct Motion {
    motion_id: Option<String>,
    url: String,
    frames: Arc<Mutex<Frames>>,
    detection: mpsc::SyncSender<DetectionRequest>,
    observations: mpsc::Receiver<DetectionResult>,
    next_read: u64,
    failure: Failure,
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
    notified: u64,
}
struct PendingMotion {
    id: String,
    token: u64,
    deadline: Instant,
    threshold: Option<f64>,
    submitted: bool,
}
struct DetectionRequest {
    id: String,
    token: u64,
    frame: GrayFrame,
    threshold: f64,
}
struct DetectionResult {
    token: u64,
    reply: Reply,
}
#[derive(Default)]
struct RecordingState {
    started: bool,
    finished: bool,
    finalized: bool,
}
struct Recording {
    url: String,
    state: Arc<Mutex<RecordingState>>,
    failure: Failure,
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
    start_request: Option<String>,
    stop_request: Option<String>,
    deadline: Instant,
}
struct Preview {
    url: String,
    receiver: mpsc::Receiver<Event>,
    failure: Failure,
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

pub struct SharedRuntime {
    inputs: HashMap<String, Input>,
    motion: Option<Motion>,
    read: Option<PendingMotion>,
    pending_motion_start: Option<(Request, Instant)>,
    recordings: HashMap<String, Recording>,
    completed: VecDeque<(String, bool)>,
    preview: Option<Preview>,
    retired_motion: Option<JoinHandle<()>>,
    retired_preview: Option<JoinHandle<()>>,
    replies: Vec<Reply>,
    waker: Arc<Waker>,
}
impl SharedRuntime {
    pub fn new(waker: Arc<Waker>) -> Self {
        Self {
            inputs: HashMap::new(),
            motion: None,
            read: None,
            pending_motion_start: None,
            recordings: HashMap::new(),
            completed: VecDeque::new(),
            preview: None,
            retired_motion: None,
            retired_preview: None,
            replies: Vec::new(),
            waker,
        }
    }
    fn subscribe(
        &mut self,
        url: &str,
        id: &str,
        connect: Duration,
        io: Duration,
        audio: bool,
    ) -> Result<(mpsc::Receiver<Packet>, Arc<Mutex<Hub>>, Failure)> {
        if !self.inputs.contains_key(url) {
            if self.inputs.len() >= MAX_STREAMS {
                return Err("camera_session_budget_exceeded");
            }
            let hub = Arc::new(Mutex::new(Hub::default()));
            let sink = Arc::clone(&hub);
            let waker = Arc::clone(&self.waker);
            let source = CameraInput::start(url.to_owned(), false, connect, io, move |event| {
                sink.lock()
                    .map_err(|_| "camera_input_failed")?
                    .deliver(event);
                let _ = waker.wake();
                Ok(())
            })?;
            self.inputs.insert(url.to_owned(), Input { source, hub });
        }
        let input = self.inputs.get(url).ok_or("camera_input_failed")?;
        if let Some(code) = input.source.failure() {
            return Err(code);
        }
        let (sender, receiver) = mpsc::sync_channel(PACKETS);
        let failure = Arc::new(Mutex::new(None));
        input
            .hub
            .lock()
            .map_err(|_| "camera_input_failed")?
            .consumers
            .insert(
                id.to_owned(),
                Consumer {
                    sender,
                    failure: Arc::clone(&failure),
                    audio,
                },
            );
        Ok((receiver, Arc::clone(&input.hub), failure))
    }
    fn unsubscribe(&mut self, url: &str, id: &str) {
        if let Some(input) = self.inputs.get(url) {
            if let Ok(mut hub) = input.hub.lock() {
                hub.consumers.remove(id);
            }
            let empty = input.hub.lock().is_ok_and(|hub| hub.consumers.is_empty());
            if empty {
                self.inputs.remove(url);
            }
        }
    }
    pub fn handle(&mut self, request: Request) -> Option<Reply> {
        if operation_motion_id(&request.operation).is_some_and(|id| id.is_empty() || id.len() > 128)
        {
            return Some(Reply::error(request.request_id, "invalid_motion_id"));
        }
        if matches!(request.operation, Operation::StartMotion { .. }) {
            reap_worker(&mut self.retired_motion);
            if self.pending_motion_start.is_some() {
                return Some(Reply::error(request.request_id, "motion_already_started"));
            }
            if self.motion.is_none() && self.retired_motion.is_some() {
                // A normal stop needs one receive-loop turn to retire. Queue
                // one bounded successor instead of pinning compatibility on
                // this transient overlap; unrelated control never waits here.
                self.pending_motion_start = Some((request, Instant::now() + RETIREMENT_TIMEOUT));
                return None;
            }
        }
        let id = request.request_id;
        let mut reply = Reply::success(id.clone());
        let result = match request.operation {
            Operation::StartMotion {
                motion_id,
                rtsp_url,
                motion_config,
                frame_queue_size,
                connect_timeout_s,
                io_timeout_s,
            } => (|| {
                reap_worker(&mut self.retired_motion);
                if self.motion.is_some() || self.retired_motion.is_some() {
                    return Err("motion_already_started");
                }
                if !(1..=1024).contains(&frame_queue_size) {
                    return Err("invalid_motion_config");
                }
                let connect = timeout(connect_timeout_s)?;
                let io = timeout(io_timeout_s)?;
                let config = MotionConfig {
                    pixel_threshold: motion_config.pixel_threshold,
                    min_changed_pct: motion_config.min_changed_pct,
                    blur_kernel: motion_config.blur_kernel,
                    recording_sensitivity_factor: motion_config.recording_sensitivity_factor,
                };
                if !config.min_changed_pct.is_finite()
                    || !config.recording_sensitivity_factor.is_finite()
                    || !config.recording_threshold().is_finite()
                {
                    return Err("invalid_motion_config");
                }
                let mut detector =
                    MotionDetector::new(config).map_err(|_| "invalid_motion_config")?;
                let (packets, hub, failure) =
                    self.subscribe(&rtsp_url, "motion", connect, io, false)?;
                let frames = Arc::new(Mutex::new(Frames::default()));
                let (detection, requests) = mpsc::sync_channel(1);
                let (results, observations) = mpsc::sync_channel(1);
                let stop = Arc::new(AtomicBool::new(false));
                let thread_frames = Arc::clone(&frames);
                let thread_stop = Arc::clone(&stop);
                let thread_failure = Arc::clone(&failure);
                let waker = Arc::clone(&self.waker);
                let thread = thread::Builder::new()
                    .name("shared-motion".into())
                    .spawn(move || {
                        let outcome = (|| {
                            let metadata =
                                wait_metadata(&hub, &thread_stop, &thread_failure, connect)?;
                            let mut decoder =
                                GrayDecoder::from_parameters(metadata.video.parameters)?;
                            let mut wait_keyframe = true;
                            while !thread_stop.load(Ordering::Acquire) {
                                // Gaussian blur and motion comparisons can be
                                // expensive for accepted operator settings. Run
                                // them only here, without a frame-queue lock or
                                // any control/WebRTC event-loop work on this thread.
                                if let Ok(request) = requests.try_recv() {
                                    let result = detect_selected_frame(&mut detector, request);
                                    if thread_stop.load(Ordering::Acquire) {
                                        break;
                                    }
                                    results
                                        .try_send(result)
                                        .map_err(|_| "motion_worker_failed")?;
                                    let _ = waker.wake();
                                }
                                if let Some(code) = failure_code(&thread_failure) {
                                    return Err(code);
                                }
                                let Some(packet) = receive(&packets, &thread_stop)? else {
                                    continue;
                                };
                                if packet.track != Track::Video {
                                    continue;
                                }
                                if wait_keyframe && !packet.packet.is_key() {
                                    continue;
                                }
                                wait_keyframe = false;
                                decoder.push_packet(
                                    &packet.packet,
                                    metadata.video.time_base,
                                    |frame| {
                                        if publish_prepared_frame(
                                            &thread_frames,
                                            frame_queue_size,
                                            frame,
                                        )
                                        .is_err()
                                        {
                                            fail(&thread_failure, "motion_worker_failed");
                                        }
                                        let _ = waker.wake();
                                    },
                                )?;
                            }
                            Ok(())
                        })();
                        if let Err(code) = outcome {
                            fail(&thread_failure, code);
                        }
                        let _ = waker.wake();
                    })
                    .map_err(|_| {
                        self.unsubscribe(&rtsp_url, "motion");
                        "motion_worker_failed"
                    })?;
                self.motion = Some(Motion {
                    motion_id,
                    url: rtsp_url,
                    frames,
                    detection,
                    observations,
                    next_read: 0,
                    failure,
                    stop,
                    thread: Some(thread),
                    notified: 0,
                });
                Ok(())
            })(),
            Operation::ReadMotion {
                motion_id,
                threshold,
                wait_timeout_s,
            } => {
                return self.queue_read(id, motion_id.as_deref(), wait_timeout_s, Some(threshold));
            }
            Operation::DiscardFrame {
                motion_id,
                wait_timeout_s,
            } => {
                return self.queue_read(id, motion_id.as_deref(), wait_timeout_s, None);
            }
            Operation::StopMotion { motion_id } => {
                let owner = self
                    .motion
                    .as_ref()
                    .map(|motion| motion.motion_id.as_deref())
                    .or_else(|| {
                        self.pending_motion_start
                            .as_ref()
                            .map(|(request, _)| operation_motion_id(&request.operation))
                    });
                if motion_id
                    .as_deref()
                    .is_some_and(|id| owner.is_some_and(|owner| owner != Some(id)))
                {
                    Err("motion_generation_mismatch")
                } else {
                    self.stop_motion();
                    Ok(())
                }
            }
            Operation::StartRecording {
                recording_id,
                rtsp_url,
                output_path,
                audio_mode,
                connect_timeout_s,
                io_timeout_s,
            } => {
                let result = self.start_recording(
                    id.clone(),
                    &recording_id,
                    rtsp_url,
                    output_path,
                    audio_mode,
                    connect_timeout_s,
                    io_timeout_s,
                );
                if result.is_ok() {
                    return None;
                }
                result
            }
            Operation::StopRecording { recording_id } => {
                if let Some(recording) = self.recordings.get_mut(&recording_id) {
                    if recording.stop_request.is_some() {
                        Err("recording_stop_pending")
                    } else {
                        recording.stop_request = Some(id);
                        recording.stop.store(true, Ordering::Release);
                        let url = recording.url.clone();
                        self.unsubscribe(&url, &recording_id);
                        return None;
                    }
                } else {
                    let completed = self.completed.iter().find(|(id, _)| id == &recording_id);
                    reply.recording_id = Some(recording_id);
                    reply.recording_active = Some(false);
                    reply.recording_finalized =
                        Some(completed.is_some_and(|(_, finalized)| *finalized));
                    // An unknown ID owns no writer. Acknowledging teardown is
                    // safe and makes a lost startup/stop reply recoverable.
                    Ok(())
                }
            }
            Operation::RecordingStatus { recording_id } => {
                reply.recording_id = Some(recording_id.clone());
                if let Some(recording) = self.recordings.get(&recording_id) {
                    let state = recording.state.lock().ok();
                    // A failed/pending writer still owns its output until its
                    // worker finishes. Failure alone never proves it closed.
                    reply.recording_active = state.as_ref().map(|state| !state.finished);
                    reply.recording_finalized = Some(state.as_ref().is_some_and(|s| s.finalized));
                    if let Some(code) = failure_code(&recording.failure) {
                        Err(code)
                    } else {
                        Ok(())
                    }
                } else {
                    let completed = self.completed.iter().find(|(id, _)| id == &recording_id);
                    reply.recording_active = Some(false);
                    reply.recording_finalized =
                        Some(completed.is_some_and(|(_, finalized)| *finalized));
                    Ok(())
                }
            }
            _ => return Some(Reply::error(id, "invalid_shared_command")),
        };
        if let Err(code) = result {
            reply.ok = false;
            reply.error_code = Some(code);
        }
        Some(reply)
    }
    fn queue_read(
        &mut self,
        id: String,
        motion_id: Option<&str>,
        seconds: f64,
        threshold: Option<f64>,
    ) -> Option<Reply> {
        let code = if !seconds.is_finite()
            || !(0.0..=120.0).contains(&seconds)
            || threshold.is_some_and(|v| !v.is_finite() || v < 0.0)
        {
            Some("invalid_motion_request")
        } else if self.motion.is_none() {
            Some("motion_not_started")
        } else if motion_id.is_some_and(|id| {
            self.motion
                .as_ref()
                .is_some_and(|motion| motion.motion_id.as_deref() != Some(id))
        }) {
            Some("motion_generation_mismatch")
        } else if self.read.is_some() {
            Some("motion_read_busy")
        } else {
            None
        };
        if let Some(code) = code {
            return Some(Reply::error(id, code));
        }
        let motion = self.motion.as_mut().unwrap();
        let Some(token) = motion.next_read.checked_add(1) else {
            return Some(Reply::error(id, "motion_worker_failed"));
        };
        motion.next_read = token;
        self.read = Some(PendingMotion {
            id,
            token,
            deadline: Instant::now() + Duration::from_secs_f64(seconds),
            threshold,
            submitted: false,
        });
        None
    }
    fn poll_motion(&mut self) {
        let Some(motion) = self.motion.as_mut() else {
            return;
        };
        if let Ok(frames) = motion.frames.lock()
            && frames.received != motion.notified
        {
            // One heartbeat covers any prepared-frame burst, without raw media.
            motion.notified = frames.received;
            self.replies.push(Reply {
                event: Some("frame"),
                ..Reply::default()
            });
        }
        let result = motion.observations.try_recv().ok();
        let Some(mut read) = self.read.take() else {
            return;
        };
        if let Some(code) = failure_code(&motion.failure) {
            let mut reply = Reply::error(read.id, code);
            reply.frame_available = Some(false);
            self.replies.push(reply);
            return;
        }
        if let Some(result) = result
            && result.token == read.token
        {
            self.replies.push(result.reply);
            return;
        }
        if !read.submitted {
            let frame = motion
                .frames
                .lock()
                .ok()
                .and_then(|mut frames| frames.ready.pop_front());
            if let Some(frame) = frame {
                if let Some(threshold) = read.threshold {
                    let request = DetectionRequest {
                        id: read.id.clone(),
                        token: read.token,
                        frame,
                        threshold,
                    };
                    if motion.detection.try_send(request).is_err() {
                        self.replies
                            .push(Reply::error(read.id, "motion_worker_failed"));
                        return;
                    }
                    read.submitted = true;
                } else {
                    let mut reply = Reply::success(read.id);
                    reply.frame_available = Some(true);
                    self.replies.push(reply);
                    return;
                }
            } else if Instant::now() >= read.deadline {
                let mut reply = Reply::success(read.id);
                reply.frame_available = Some(false);
                self.replies.push(reply);
                return;
            }
        }
        // wait_timeout is the existing prepared-frame wait, not a new OpenCV
        // deadline. Stop and source failure still cancel an in-flight reply.
        self.read = Some(read);
    }
    fn stop_motion(&mut self) {
        if let Some((request, _)) = self.pending_motion_start.take() {
            self.replies
                .push(Reply::error(request.request_id, "motion_stopped"));
        }
        if let Some(read) = self.read.take() {
            self.replies.push(Reply::error(read.id, "motion_stopped"));
        }
        if let Some(mut motion) = self.motion.take() {
            motion.stop.store(true, Ordering::Release);
            self.unsubscribe(&motion.url, "motion");
            self.retired_motion = motion.thread.take();
            reap_worker(&mut self.retired_motion);
        }
    }
    #[allow(clippy::too_many_arguments)]
    fn start_recording(
        &mut self,
        request: String,
        id: &str,
        url: String,
        output: String,
        audio: String,
        connect_s: f64,
        io_s: f64,
    ) -> Result<()> {
        if id.is_empty()
            || id.len() > 128
            || id == "motion"
            || id == "preview"
            || self.recordings.contains_key(id)
            || self.completed.iter().any(|(completed, _)| completed == id)
        {
            return Err("invalid_recording_id");
        }
        if self.recordings.len() >= MAX_RECORDINGS {
            return Err("recording_budget_exceeded");
        }
        if !matches!(audio.as_str(), "copy" | "none") {
            return Err("unsupported_recording_codec");
        }
        let connect = timeout(connect_s)?;
        let io = timeout(io_s)?;
        let path = PathBuf::from(output);
        if !path.is_absolute()
            || path.extension().is_none_or(|e| e != "mp4")
            || path.as_os_str().len() > 4096
        {
            return Err("recording_path_invalid");
        }
        let (packets, hub, failure) = self.subscribe(&url, id, connect, io, audio == "copy")?;
        let state = Arc::new(Mutex::new(RecordingState::default()));
        let stop = Arc::new(AtomicBool::new(false));
        let thread_state = Arc::clone(&state);
        let thread_failure = Arc::clone(&failure);
        let thread_stop = Arc::clone(&stop);
        let waker = Arc::clone(&self.waker);
        let worker = thread::Builder::new()
            .name("shared-recording".into())
            .spawn(move || {
                let outcome = record(
                    packets,
                    hub,
                    &path,
                    audio == "copy",
                    connect,
                    io,
                    &thread_state,
                    &thread_failure,
                    &thread_stop,
                    &waker,
                );
                if let Err(code) = outcome {
                    fail(&thread_failure, code);
                }
                if let Ok(mut state) = thread_state.lock() {
                    state.finished = true;
                    state.finalized = outcome.is_ok();
                }
                let _ = waker.wake();
            })
            .map_err(|_| {
                self.unsubscribe(&url, id);
                "recording_worker_failed"
            })?;
        self.recordings.insert(
            id.to_owned(),
            Recording {
                url,
                state,
                failure,
                stop,
                thread: Some(worker),
                start_request: Some(request),
                stop_request: None,
                deadline: Instant::now() + connect + io,
            },
        );
        Ok(())
    }
    pub fn poll(&mut self) -> Vec<Reply> {
        reap_worker(&mut self.retired_motion);
        reap_worker(&mut self.retired_preview);
        if let Some((request, deadline)) = self.pending_motion_start.take() {
            if self.retired_motion.is_none() {
                if let Some(reply) = self.handle(request) {
                    self.replies.push(reply);
                }
            } else if Instant::now() >= deadline {
                self.replies
                    .push(Reply::error(request.request_id, "motion_restart_timeout"));
            } else {
                self.pending_motion_start = Some((request, deadline));
            }
        }
        for input in self.inputs.values() {
            if let Some(code) = input.source.failure()
                && let Ok(hub) = input.hub.lock()
            {
                for consumer in hub.consumers.values() {
                    fail(&consumer.failure, code);
                }
            }
        }
        self.poll_motion();
        let mut remove = Vec::new();
        let mut detach = Vec::new();
        for (id, recording) in &mut self.recordings {
            let Ok(state) = recording.state.lock() else {
                continue;
            };
            let failure = failure_code(&recording.failure);
            if recording.start_request.is_some()
                && (state.started || failure.is_some() || Instant::now() >= recording.deadline)
            {
                let request = recording.start_request.take().unwrap();
                let mut reply = match failure
                    .or_else(|| (!state.started).then_some("recording_start_timeout"))
                {
                    Some(code) => {
                        recording.stop.store(true, Ordering::Release);
                        fail(&recording.failure, code);
                        Reply::error(request, code)
                    }
                    None => Reply::success(request),
                };
                reply.recording_id = Some(id.clone());
                reply.recording_active = Some(!state.finished);
                self.replies.push(reply);
                if !state.started || failure.is_some() {
                    detach.push((id.clone(), recording.url.clone()));
                }
            }
            if state.finished && recording.stop_request.is_some() {
                let request = recording.stop_request.take().unwrap();
                let mut reply = if let Some(code) = failure {
                    Reply::error(request, code)
                } else {
                    Reply::success(request)
                };
                reply.recording_id = Some(id.clone());
                reply.recording_active = Some(false);
                reply.recording_finalized = Some(state.finalized);
                self.replies.push(reply);
                remove.push(id.clone());
            } else if state.finished
                && recording.stop.load(Ordering::Acquire)
                && recording.start_request.is_none()
            {
                remove.push(id.clone());
            }
        }
        for (id, url) in detach {
            self.unsubscribe(&url, &id);
        }
        for id in remove {
            if let Some(mut recording) = self.recordings.remove(&id) {
                let finalized = recording.state.lock().is_ok_and(|state| state.finalized);
                if self.completed.len() == 32 {
                    self.completed.pop_front();
                }
                self.completed.push_back((id.clone(), finalized));
                self.unsubscribe(&recording.url, &id);
                if let Some(thread) = recording.thread.take()
                    && thread.is_finished()
                {
                    let _ = thread.join();
                }
            }
        }
        std::mem::take(&mut self.replies)
    }
    pub fn start_preview(&mut self, url: String, connect: Duration, io: Duration) -> Result<()> {
        reap_worker(&mut self.retired_preview);
        if self.preview.is_some() || self.retired_preview.is_some() {
            return Err("preview_already_started");
        }
        let (packets, hub, failure) = self.subscribe(&url, "preview", connect, io, false)?;
        let (sender, receiver) = mpsc::sync_channel(8);
        let stop = Arc::new(AtomicBool::new(false));
        let thread_stop = Arc::clone(&stop);
        let thread_failure = Arc::clone(&failure);
        let waker = Arc::clone(&self.waker);
        let thread = thread::Builder::new()
            .name("shared-preview".into())
            .spawn(move || {
                let outcome = (|| {
                    let metadata = wait_metadata(&hub, &thread_stop, &thread_failure, connect)?;
                    let mut framing = VideoFraming::new(&metadata.video.parameters)?;
                    sender
                        .try_send(Event::Info {
                            profile_level_id: framing.profile,
                        })
                        .map_err(|_| "media_queue_overflow")?;
                    let _ = waker.wake();
                    let mut wait_keyframe = true;
                    let mut previous_timestamp = None;
                    while !thread_stop.load(Ordering::Acquire) {
                        if let Some(code) = failure_code(&thread_failure) {
                            return Err(code);
                        }
                        let Some(packet) = receive(&packets, &thread_stop)? else {
                            continue;
                        };
                        if packet.track != Track::Video {
                            continue;
                        }
                        let (data, parameters_changed) = framing.annex_b(
                            packet.packet.data().ok_or("media_frame_invalid")?,
                            packet.packet.is_key(),
                        )?;
                        // Passthrough preserves the existing no-B-frame browser contract.
                        if !crate::rtsp::validate_access_unit(&data)? {
                            continue;
                        }
                        let pts = packet.packet.pts().ok_or("media_timestamp_invalid")?;
                        let timestamp =
                            pts.rescale(metadata.video.time_base, ffmpeg::Rational(1, 90_000));
                        if previous_timestamp.is_some_and(|previous| timestamp <= previous) {
                            return Err("unsupported_frame_reordering");
                        }
                        previous_timestamp = Some(timestamp);
                        wait_keyframe |= parameters_changed;
                        if wait_keyframe && !packet.packet.is_key() {
                            continue;
                        }
                        wait_keyframe = false;
                        sender
                            .try_send(Event::Frame(EncodedFrame {
                                timestamp: timestamp as u32,
                                keyframe: packet.packet.is_key(),
                                data: data.into(),
                            }))
                            .map_err(|_| "media_queue_overflow")?;
                        let _ = waker.wake();
                    }
                    Ok(())
                })();
                if let Err(code) = outcome {
                    fail(&thread_failure, code);
                }
                let _ = waker.wake();
            })
            .map_err(|_| {
                self.unsubscribe(&url, "preview");
                "preview_worker_failed"
            })?;
        self.preview = Some(Preview {
            url,
            receiver,
            failure,
            stop,
            thread: Some(thread),
        });
        Ok(())
    }
    pub fn try_recv_preview(&mut self) -> Result<Option<Event>> {
        let preview = self.preview.as_ref().ok_or("preview_not_started")?;
        if let Some(code) = failure_code(&preview.failure) {
            return Err(code);
        }
        match preview.receiver.try_recv() {
            Ok(event) => Ok(Some(event)),
            Err(mpsc::TryRecvError::Empty) => Ok(None),
            Err(mpsc::TryRecvError::Disconnected) => Err("preview_worker_failed"),
        }
    }
    pub fn stop_preview(&mut self) {
        if let Some(mut preview) = self.preview.take() {
            preview.stop.store(true, Ordering::Release);
            self.unsubscribe(&preview.url, "preview");
            self.retired_preview = preview.thread.take();
            reap_worker(&mut self.retired_preview);
        }
    }
}
impl Drop for SharedRuntime {
    fn drop(&mut self) {
        self.stop_preview();
        self.stop_motion();
        // EOF/normal shutdown can complete owned clips. A forced process death
        // leaves only .partial files, which the replay scanner cannot ingest.
        for recording in self.recordings.values() {
            recording.stop.store(true, Ordering::Release);
        }
        self.inputs.clear();
        let deadline = Instant::now() + RETIREMENT_TIMEOUT;
        for worker in [&mut self.retired_motion, &mut self.retired_preview] {
            while worker.as_ref().is_some_and(|thread| !thread.is_finished())
                && Instant::now() < deadline
            {
                thread::sleep(Duration::from_millis(5));
            }
            reap_worker(worker);
        }
        for recording in self.recordings.values_mut() {
            if let Some(thread) = recording.thread.take() {
                while !thread.is_finished() && Instant::now() < deadline {
                    thread::sleep(Duration::from_millis(5));
                }
                if thread.is_finished() {
                    let _ = thread.join();
                }
            }
        }
    }
}
fn operation_motion_id(operation: &Operation) -> Option<&str> {
    match operation {
        Operation::StartMotion { motion_id, .. }
        | Operation::ReadMotion { motion_id, .. }
        | Operation::DiscardFrame { motion_id, .. }
        | Operation::StopMotion { motion_id } => motion_id.as_deref(),
        _ => None,
    }
}
fn publish_prepared_frame(frames: &Mutex<Frames>, capacity: usize, frame: GrayFrame) -> Result<()> {
    let mut frames = frames.lock().map_err(|_| "motion_worker_failed")?;
    if frames.ready.len() == capacity {
        frames.ready.pop_front();
    }
    frames.ready.push_back(frame);
    frames.received = frames.received.saturating_add(1);
    Ok(())
}
fn detect_selected_frame(
    detector: &mut MotionDetector,
    request: DetectionRequest,
) -> DetectionResult {
    let mut reply = Reply::success(request.id);
    reply.frame_available = Some(true);
    match detector.detect(
        &request.frame.data,
        request.frame.width,
        request.frame.height,
        Some(request.threshold),
    ) {
        Ok(observation) => reply.observation = Some(observation.into()),
        Err(_) => {
            reply.ok = false;
            reply.error_code = Some("motion_frame_invalid");
        }
    }
    DetectionResult {
        token: request.token,
        reply,
    }
}
// Keep at most one retiring generation of each consumer. Its private frame
// state cannot reach a successor, and a stuck decode cannot block control I/O.
fn reap_worker(worker: &mut Option<JoinHandle<()>>) {
    if worker.as_ref().is_some_and(JoinHandle::is_finished)
        && let Some(thread) = worker.take()
    {
        let _ = thread.join();
    }
}
pub(crate) fn timeout(seconds: f64) -> Result<Duration> {
    if !seconds.is_finite() || seconds <= 0.0 || seconds > 120.0 {
        return Err("invalid_rtsp_timeout");
    }
    let duration = Duration::from_secs_f64(seconds);
    if duration.is_zero() {
        return Err("invalid_rtsp_timeout");
    }
    Ok(duration)
}
fn fail(failure: &Failure, code: &'static str) {
    if let Ok(mut failure) = failure.lock() {
        failure.get_or_insert(code);
    }
}
fn failure_code(failure: &Failure) -> Option<&'static str> {
    failure
        .lock()
        .map_or(Some("media_worker_failed"), |value| *value)
}
fn wait_metadata(
    hub: &Mutex<Hub>,
    stop: &AtomicBool,
    failure: &Failure,
    timeout: Duration,
) -> Result<Metadata> {
    let deadline = Instant::now() + timeout;
    loop {
        if stop.load(Ordering::Acquire) {
            return Err("media_stopped");
        }
        if let Some(code) = failure_code(failure) {
            return Err(code);
        }
        if let Some(metadata) = hub
            .lock()
            .map_err(|_| "camera_input_failed")?
            .metadata
            .as_ref()
        {
            return Ok(metadata.clone());
        }
        if Instant::now() >= deadline {
            return Err("rtsp_timeout");
        }
        thread::sleep(Duration::from_millis(5));
    }
}
fn receive(packets: &mpsc::Receiver<Packet>, stop: &AtomicBool) -> Result<Option<Packet>> {
    match packets.recv_timeout(Duration::from_millis(20)) {
        Ok(packet) => Ok(Some(packet)),
        Err(mpsc::RecvTimeoutError::Timeout) => Ok(None),
        Err(mpsc::RecvTimeoutError::Disconnected) if stop.load(Ordering::Acquire) => Ok(None),
        Err(_) => Err("camera_input_failed"),
    }
}
#[allow(clippy::too_many_arguments)]
fn record(
    packets: mpsc::Receiver<Packet>,
    hub: Arc<Mutex<Hub>>,
    path: &Path,
    audio: bool,
    connect: Duration,
    io: Duration,
    state: &Mutex<RecordingState>,
    failure: &Failure,
    stop: &AtomicBool,
    waker: &Waker,
) -> Result<()> {
    if path.symlink_metadata().is_ok() {
        return Err("recording_path_invalid");
    }
    let metadata = wait_metadata(&hub, stop, failure, connect + io)?;
    let video = StreamConfig::copy(metadata.video.parameters.clone(), metadata.video.time_base)?;
    let audio_config = if audio {
        metadata
            .audio
            .as_ref()
            .map(|stream| StreamConfig::copy(stream.parameters.clone(), stream.time_base))
            .transpose()?
    } else {
        None
    };
    let partial = PathBuf::from(format!("{}.partial", path.display()));
    let mut writer = Mp4Recorder::create(&partial, video, audio_config)?;
    let deadline = Instant::now() + io;
    let mut epoch = None;
    let mut audio_started = false;
    let mut count = 0_u64;
    loop {
        if let Some(code) = failure_code(failure) {
            return Err(code);
        }
        let packet = if stop.load(Ordering::Acquire) {
            match packets.try_recv() {
                Ok(packet) => packet,
                Err(_) => break,
            }
        } else {
            let Some(packet) = receive(&packets, stop)? else {
                if epoch.is_none() && Instant::now() >= deadline {
                    return Err("recording_keyframe_timeout");
                }
                continue;
            };
            packet
        };
        if packet.track == Track::Audio && !audio {
            continue;
        }
        let stream = match packet.track {
            Track::Video => &metadata.video,
            Track::Audio => metadata
                .audio
                .as_ref()
                .ok_or("recording_parameters_invalid")?,
        };
        let pts = packet.packet.pts().ok_or("recording_timestamp_invalid")?;
        let dts = packet.packet.dts().ok_or("recording_timestamp_invalid")?;
        if epoch.is_none() {
            if packet.track != Track::Video || !packet.packet.is_key() {
                continue;
            }
            // One epoch across both track clocks preserves the demuxer's A/V offset.
            epoch = Some((dts, stream.time_base));
        }
        let (epoch_ticks, epoch_time_base) = epoch.unwrap();
        let offset = epoch_ticks.rescale(epoch_time_base, stream.time_base);
        if dts < offset {
            if packet.track == Track::Audio && !audio_started {
                continue; // Initial audio cannot precede the clip's IDR epoch.
            }
            return Err("recording_timestamp_invalid");
        }
        writer.push(EncodedPacket {
            track: packet.track,
            data: packet.packet.data().ok_or("recording_packet_invalid")?,
            pts: pts
                .checked_sub(offset)
                .ok_or("recording_timestamp_invalid")?,
            dts: dts
                .checked_sub(offset)
                .ok_or("recording_timestamp_invalid")?,
            duration: packet.packet.duration(),
            keyframe: packet.packet.is_key(),
        })?;
        audio_started |= packet.track == Track::Audio;
        if count == 0 {
            // Python stops the previous clip after this acknowledgement.
            // Header allocation is not enough: a validated IDR must already
            // be recorded, so rotation cannot introduce a wait-for-keyframe gap.
            state.lock().map_err(|_| "recording_worker_failed")?.started = true;
            let _ = waker.wake();
        }
        count += 1;
        if count >= 1_000_000 {
            return Err("recording_packet_budget_exceeded");
        }
    }
    writer.finish()?;
    if let Some(code) = failure_code(failure) {
        return Err(code);
    }
    // Atomic publication without replacing a concurrently created final file.
    // The temporary and final path share a directory/filesystem by construction.
    std::fs::hard_link(&partial, path).map_err(|_| "recording_publish_failed")?;
    let _ = std::fs::remove_file(&partial);
    Ok(())
}

struct VideoFraming {
    prefix: Vec<u8>,
    sps: Vec<u8>,
    pps: Vec<u8>,
    length_size: Option<usize>,
    profile: u32,
}
impl VideoFraming {
    fn new(parameters: &ffmpeg::codec::Parameters) -> Result<Self> {
        // SAFETY: Parameters owns bounded extradata for this borrow. The demux
        // boundary and recording validator check its allocation and codec.
        let raw = unsafe { &*parameters.as_ptr() };
        if raw.extradata.is_null() || !(1..=65536).contains(&raw.extradata_size) {
            return Err("unsupported_codec");
        }
        let extra =
            unsafe { std::slice::from_raw_parts(raw.extradata, raw.extradata_size as usize) };
        let (nals, length_size) = if extra.first() == Some(&1) {
            if extra.len() < 7 {
                return Err("unsupported_codec");
            }
            let length = usize::from((extra[4] & 3) + 1);
            let mut pos = 6;
            let mut nals = Vec::new();
            for _ in 0..(extra[5] & 31) {
                nals.push(avcc_parameter(extra, &mut pos)?);
            }
            let pps = *extra.get(pos).ok_or("unsupported_codec")?;
            pos += 1;
            for _ in 0..pps {
                nals.push(avcc_parameter(extra, &mut pos)?);
            }
            (nals, Some(length))
        } else {
            (annex_nals(extra), None)
        };
        let sps = nals
            .iter()
            .find(|nal| nal.first().is_some_and(|b| b & 31 == 7))
            .ok_or("unsupported_codec")?;
        let pps = nals
            .iter()
            .find(|nal| nal.first().is_some_and(|b| b & 31 == 8))
            .ok_or("unsupported_codec")?;
        let parameters = retina::codec::h264::parameters_from_sps_and_pps(
            sps,
            pps,
            retina::codec::h26x::Framing::AnnexB,
        )
        .map_err(|_| "unsupported_codec")?;
        let profile = crate::rtsp::profile_level_id(&parameters)?;
        let mut prefix = Vec::new();
        for nal in [sps, pps] {
            prefix.extend_from_slice(&[0, 0, 0, 1]);
            prefix.extend_from_slice(nal);
        }
        Ok(Self {
            prefix,
            sps: sps.to_vec(),
            pps: pps.to_vec(),
            length_size,
            profile,
        })
    }
    fn annex_b(&mut self, bytes: &[u8], keyframe: bool) -> Result<(Vec<u8>, bool)> {
        let mut output = Vec::new();
        if let Some(length) = self.length_size {
            let mut pos = 0;
            while pos < bytes.len() {
                let header = bytes.get(pos..pos + length).ok_or("media_frame_invalid")?;
                let size = header
                    .iter()
                    .fold(0usize, |size, byte| (size << 8) | usize::from(*byte));
                pos += length;
                if size == 0 || size > 2 * 1024 * 1024 {
                    return Err("media_frame_invalid");
                }
                let nal = bytes.get(pos..pos + size).ok_or("media_frame_invalid")?;
                output.extend_from_slice(&[0, 0, 0, 1]);
                output.extend_from_slice(nal);
                pos += size;
            }
        } else {
            output.extend_from_slice(bytes);
        }
        if output.len() > 2 * 1024 * 1024 {
            return Err("media_frame_invalid");
        }
        // Match the standalone source's eligibility checks before forwarding
        // any in-band parameters under the previously negotiated profile.
        let mut sps = self.sps.as_slice();
        let mut pps = self.pps.as_slice();
        for nal in annex_nals(&output) {
            match nal.first().map(|byte| byte & 31) {
                Some(7) => sps = nal,
                Some(8) => pps = nal,
                _ => {}
            }
        }
        let changed = sps != self.sps || pps != self.pps;
        if changed {
            let parameters = retina::codec::h264::parameters_from_sps_and_pps(
                sps,
                pps,
                retina::codec::h26x::Framing::AnnexB,
            )
            .map_err(|_| "unsupported_codec")?;
            if crate::rtsp::profile_level_id(&parameters)? != self.profile {
                return Err("codec_parameters_changed");
            }
            self.sps = sps.to_vec();
            self.pps = pps.to_vec();
            self.prefix.clear();
            for nal in [&self.sps, &self.pps] {
                self.prefix.extend_from_slice(&[0, 0, 0, 1]);
                self.prefix.extend_from_slice(nal);
            }
        }
        if keyframe {
            if output.len() + self.prefix.len() > 2 * 1024 * 1024 {
                return Err("media_frame_invalid");
            }
            output.splice(..0, self.prefix.iter().copied());
        }
        Ok((output, changed))
    }
}
fn avcc_parameter<'a>(extra: &'a [u8], pos: &mut usize) -> Result<&'a [u8]> {
    let bytes: [u8; 2] = extra
        .get(*pos..*pos + 2)
        .ok_or("unsupported_codec")?
        .try_into()
        .unwrap();
    *pos += 2;
    let length = usize::from(u16::from_be_bytes(bytes));
    if length == 0 || length > 4096 {
        return Err("unsupported_codec");
    }
    let nal = extra.get(*pos..*pos + length).ok_or("unsupported_codec")?;
    *pos += length;
    Ok(nal)
}
fn annex_nals(bytes: &[u8]) -> Vec<&[u8]> {
    let mut positions = Vec::new();
    let mut pos = 0;
    while pos + 3 <= bytes.len() {
        let length = if bytes.get(pos..pos + 4) == Some(&[0, 0, 0, 1]) {
            4
        } else if bytes.get(pos..pos + 3) == Some(&[0, 0, 1]) {
            3
        } else {
            pos += 1;
            continue;
        };
        positions.push((pos, pos + length));
        pos += length;
    }
    positions
        .iter()
        .enumerate()
        .map(|(i, (_, start))| {
            let end = positions.get(i + 1).map_or(bytes.len(), |(pos, _)| *pos);
            let mut nal = &bytes[*start..end];
            while nal.last() == Some(&0) {
                nal = &nal[..nal.len() - 1];
            }
            nal
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn attach(hub: &mut Hub, id: &str, audio: bool) -> (mpsc::Receiver<Packet>, Failure) {
        let (sender, receiver) = mpsc::sync_channel(1);
        let failure = Arc::new(Mutex::new(None));
        hub.consumers.insert(
            id.into(),
            Consumer {
                sender,
                failure: Arc::clone(&failure),
                audio,
            },
        );
        (receiver, failure)
    }

    #[test]
    fn undrained_preview_and_motion_cannot_interrupt_recording_delivery() {
        // Given: One camera dispatcher with idle preview/motion and an active recorder.
        let mut hub = Hub::default();
        let (_preview, preview_failure) = attach(&mut hub, "preview", false);
        let (_motion, motion_failure) = attach(&mut hub, "motion", false);
        let (recording, recording_failure) = attach(&mut hub, "clip", true);
        // When: The camera continues while only recording consumes its queue.
        for index in 0..10_u8 {
            hub.deliver(CameraEvent::Packet {
                track: Track::Video,
                packet: ffmpeg::Packet::copy(&[index]),
            });
            assert_eq!(
                recording.try_recv().unwrap().packet.data(),
                Some([index].as_slice())
            );
        }
        // Then: Compressed recording packets remain intact; each slow consumer fails independently.
        assert_eq!(failure_code(&preview_failure), Some("media_queue_overflow"));
        assert_eq!(failure_code(&motion_failure), Some("media_queue_overflow"));
        assert_eq!(failure_code(&recording_failure), None);
    }

    #[test]
    fn audio_packets_only_use_the_selected_recording_budget() {
        // Given: A video-only motion/preview consumer and a recorder needing audio.
        let mut hub = Hub::default();
        let (video, video_failure) = attach(&mut hub, "motion", false);
        let (recording, recording_failure) = attach(&mut hub, "clip", true);
        // When: An audio burst precedes the next video packet.
        for _ in 0..10 {
            hub.deliver(CameraEvent::Packet {
                track: Track::Audio,
                packet: ffmpeg::Packet::copy(&[1]),
            });
            assert_eq!(recording.try_recv().unwrap().track, Track::Audio);
        }
        hub.deliver(CameraEvent::Packet {
            track: Track::Video,
            packet: ffmpeg::Packet::copy(&[2]),
        });
        // Then: Irrelevant audio cannot fill the video consumer or stop recording.
        assert_eq!(video.try_recv().unwrap().track, Track::Video);
        assert_eq!(failure_code(&video_failure), None);
        assert_eq!(failure_code(&recording_failure), None);
    }

    fn runtime() -> (mio::Poll, SharedRuntime) {
        let poll = mio::Poll::new().unwrap();
        let waker = Arc::new(Waker::new(poll.registry(), mio::Token(0)).unwrap());
        let runtime = SharedRuntime::new(waker);
        (poll, runtime)
    }

    fn test_motion(
        runtime: &mut SharedRuntime,
        motion_id: Option<&str>,
        block: Option<(mpsc::SyncSender<()>, mpsc::Receiver<()>)>,
    ) -> (Arc<Mutex<Frames>>, mpsc::Receiver<()>) {
        let frames = Arc::new(Mutex::new(Frames::default()));
        let (detection, requests) = mpsc::sync_channel(1);
        let (results, observations) = mpsc::sync_channel(1);
        let (finished, completion) = mpsc::sync_channel(1);
        let stop = Arc::new(AtomicBool::new(false));
        let worker_stop = Arc::clone(&stop);
        let waker = Arc::clone(&runtime.waker);
        let thread = thread::spawn(move || {
            let mut block = block;
            let mut detector = MotionDetector::new(MotionConfig {
                pixel_threshold: 20,
                min_changed_pct: 10.0,
                blur_kernel: 1,
                recording_sensitivity_factor: 2.0,
            })
            .unwrap();
            while !worker_stop.load(Ordering::Acquire) {
                let Ok(request) = requests.recv_timeout(Duration::from_millis(20)) else {
                    continue;
                };
                // This is the controlled CPU boundary: the real detector and
                // production reply conversion run after the test releases it.
                if let Some((entered, release)) = block.take() {
                    entered.send(()).unwrap();
                    release.recv_timeout(Duration::from_secs(3)).unwrap();
                }
                let result = detect_selected_frame(&mut detector, request);
                if !worker_stop.load(Ordering::Acquire) {
                    results.try_send(result).unwrap();
                    let _ = waker.wake();
                }
            }
            let _ = finished.send(());
        });
        runtime.motion = Some(Motion {
            motion_id: motion_id.map(str::to_owned),
            url: "test-motion".into(),
            frames: Arc::clone(&frames),
            detection,
            observations,
            next_read: 0,
            failure: Arc::new(Mutex::new(None)),
            stop,
            thread: Some(thread),
            notified: 0,
        });
        (frames, completion)
    }

    fn gray(value: u8) -> GrayFrame {
        GrayFrame {
            data: vec![value; 64],
            width: 8,
            height: 8,
        }
    }

    fn read_motion(runtime: &mut SharedRuntime, id: &str, threshold: f64) -> Reply {
        assert!(
            runtime
                .handle(Request {
                    request_id: id.into(),
                    operation: Operation::ReadMotion {
                        motion_id: None,
                        threshold,
                        wait_timeout_s: 0.0,
                    }
                })
                .is_none()
        );
        let deadline = Instant::now() + Duration::from_secs(2);
        loop {
            if let Some(reply) = runtime
                .poll()
                .into_iter()
                .find(|reply| reply.request_id == id)
            {
                assert!(reply.ok, "motion read failed: {:?}", reply.error_code);
                return reply;
            }
            assert!(
                Instant::now() < deadline,
                "selected-frame detection must finish"
            );
            thread::sleep(Duration::from_millis(2));
        }
    }

    #[test]
    fn blocked_detection_keeps_control_responsive_and_stop_cancels_its_reply() {
        // Given: The selected-frame CPU boundary is blocked after one read is submitted.
        let (_poll, mut runtime) = runtime();
        let (entered, executing) = mpsc::sync_channel(1);
        let (release, blocked) = mpsc::sync_channel(1);
        let (frames, finished) = test_motion(&mut runtime, None, Some((entered, blocked)));
        publish_prepared_frame(&frames, 2, gray(0)).unwrap();
        assert!(
            runtime
                .handle(Request {
                    request_id: "reused-id".into(),
                    operation: Operation::ReadMotion {
                        motion_id: None,
                        threshold: 10.0,
                        wait_timeout_s: 0.0,
                    }
                })
                .is_none()
        );
        runtime.poll();
        executing.recv_timeout(Duration::from_secs(1)).unwrap();
        // When: Unrelated control and stop arrive while the worker cannot finish detection.
        let start = Instant::now();
        let status = runtime
            .handle(Request {
                request_id: "status".into(),
                operation: Operation::RecordingStatus {
                    recording_id: "unrelated".into(),
                },
            })
            .unwrap();
        let stopped = runtime
            .handle(Request {
                request_id: "stop".into(),
                operation: Operation::StopMotion { motion_id: None },
            })
            .unwrap();
        let replies = runtime.poll();
        // Then: Control returns promptly, the pending read is canceled, and late work has no reply.
        assert!(start.elapsed() < Duration::from_millis(200));
        assert!(status.ok);
        assert_eq!(status.recording_active, Some(false));
        assert!(stopped.ok);
        let canceled = replies
            .into_iter()
            .find(|reply| reply.request_id == "reused-id")
            .unwrap();
        assert_eq!(canceled.error_code, Some("motion_stopped"));
        release.send(()).unwrap();
        finished.recv_timeout(Duration::from_secs(2)).unwrap();
        assert!(
            runtime
                .poll()
                .iter()
                .all(|reply| reply.request_id != "reused-id")
        );

        // A successor owns fresh channels and baseline even if the public ID is reused.
        let (frames, _) = test_motion(&mut runtime, None, None);
        publish_prepared_frame(&frames, 2, gray(0)).unwrap();
        read_motion(&mut runtime, "baseline", 10.0);
        publish_prepared_frame(&frames, 2, gray(255)).unwrap();
        let reply = read_motion(&mut runtime, "reused-id", 10.0);
        let observation = reply.observation.unwrap();
        assert_eq!(observation.changed_pixels, 64);
        assert!(observation.motion);
    }

    #[test]
    fn selected_frames_preserve_drop_oldest_discard_baseline_and_per_read_thresholds() {
        // Given: Prepared frames have a two-frame queue and the real detector starts from darkness.
        let (_poll, mut runtime) = runtime();
        let (frames, _) = test_motion(&mut runtime, None, None);
        publish_prepared_frame(&frames, 2, gray(0)).unwrap();
        read_motion(&mut runtime, "baseline", 10.0);
        // When: A dropped bright frame precedes retained darkness and a changed frame.
        for value in [255, 0, 40] {
            publish_prepared_frame(&frames, 2, gray(value)).unwrap();
        }
        let unchanged = read_motion(&mut runtime, "unchanged", 10.0)
            .observation
            .unwrap();
        let high_threshold = read_motion(&mut runtime, "high-threshold", 101.0)
            .observation
            .unwrap();
        publish_prepared_frame(&frames, 2, gray(0)).unwrap();
        assert!(
            runtime
                .handle(Request {
                    request_id: "discard".into(),
                    operation: Operation::DiscardFrame {
                        motion_id: None,
                        wait_timeout_s: 0.0
                    }
                })
                .is_none()
        );
        assert!(
            runtime
                .poll()
                .iter()
                .any(|reply| reply.request_id == "discard" && reply.frame_available == Some(true))
        );
        publish_prepared_frame(&frames, 2, gray(40)).unwrap();
        let after_discard = read_motion(&mut runtime, "after-discard", 50.0)
            .observation
            .unwrap();
        publish_prepared_frame(&frames, 2, gray(0)).unwrap();
        let low_threshold = read_motion(&mut runtime, "low-threshold", 50.0)
            .observation
            .unwrap();
        // Then: Only consumed detection frames advance baseline, and each read uses its own threshold.
        assert_eq!(unchanged.changed_pixels, 0);
        assert_eq!(high_threshold.changed_pct, 100.0);
        assert!(!high_threshold.motion);
        assert_eq!(after_discard.changed_pixels, 0);
        assert_eq!(low_threshold.changed_pct, 100.0);
        assert!(low_threshold.motion);
    }

    #[test]
    fn failed_or_timed_out_start_retains_recording_ownership_until_the_writer_exits() {
        // Given: A recording worker holds its actual output file while startup failure unwinds.
        for (id, failure_code) in [
            ("failed", Some("recording_write_failed")),
            ("timed-out", None),
        ] {
            let (_poll, mut runtime) = runtime();
            let directory = tempfile::tempdir().unwrap();
            let path = directory.path().join("owned.partial");
            let state = Arc::new(Mutex::new(RecordingState::default()));
            let worker_state = Arc::clone(&state);
            let (opened, ready) = mpsc::sync_channel(1);
            let (release, blocked) = mpsc::sync_channel(1);
            let (finished, completion) = mpsc::sync_channel(1);
            let thread = thread::spawn(move || {
                let file = std::fs::File::create(path).unwrap();
                opened.send(()).unwrap();
                blocked.recv_timeout(Duration::from_secs(3)).unwrap();
                drop(file);
                worker_state.lock().unwrap().finished = true;
                finished.send(()).unwrap();
            });
            ready.recv_timeout(Duration::from_secs(1)).unwrap();
            runtime.recordings.insert(
                id.into(),
                Recording {
                    url: "test-recording".into(),
                    state,
                    failure: Arc::new(Mutex::new(failure_code)),
                    stop: Arc::new(AtomicBool::new(false)),
                    thread: Some(thread),
                    start_request: Some(format!("start-{id}")),
                    stop_request: None,
                    deadline: Instant::now(),
                },
            );
            // When: The native wire reports startup failure before the blocked writer has closed.
            let reply = runtime
                .poll()
                .into_iter()
                .find(|reply| reply.request_id == format!("start-{id}"))
                .unwrap();
            let status = runtime
                .handle(Request {
                    request_id: format!("status-{id}"),
                    operation: Operation::RecordingStatus {
                        recording_id: id.into(),
                    },
                })
                .unwrap();
            // Then: Failure never claims that output ownership has ended; only worker completion does.
            assert!(!reply.ok);
            assert_eq!(reply.recording_active, Some(true));
            assert_eq!(status.recording_active, Some(true));
            release.send(()).unwrap();
            completion.recv_timeout(Duration::from_secs(1)).unwrap();
            runtime.poll();
            let status = runtime
                .handle(Request {
                    request_id: format!("closed-{id}"),
                    operation: Operation::RecordingStatus {
                        recording_id: id.into(),
                    },
                })
                .unwrap();
            assert_eq!(status.recording_active, Some(false));
            assert_eq!(status.recording_finalized, Some(false));
        }
    }

    fn start_motion_request(id: &str, motion_id: &str, url: &str) -> Request {
        Request {
            request_id: id.into(),
            operation: Operation::StartMotion {
                motion_id: Some(motion_id.into()),
                rtsp_url: url.into(),
                motion_config: crate::protocol::NativeOperationSettings {
                    pixel_threshold: 20,
                    min_changed_pct: 10.0,
                    blur_kernel: 1,
                    recording_sensitivity_factor: 2.0,
                },
                frame_queue_size: 2,
                connect_timeout_s: 2.0,
                io_timeout_s: 2.0,
            },
        }
    }

    #[test]
    fn stale_generation_read_discard_and_stop_cannot_consume_a_successor_frame() {
        // Given: The successor's detector has consumed darkness and retains a bright next frame.
        let (_poll, mut runtime) = runtime();
        let (frames, _) = test_motion(&mut runtime, Some("successor"), None);
        publish_prepared_frame(&frames, 2, gray(0)).unwrap();
        read_motion(&mut runtime, "baseline", 10.0);
        publish_prepared_frame(&frames, 2, gray(255)).unwrap();
        // When: Delayed read/discard/stop commands arrive with the previous generation's ID.
        for operation in [
            Operation::ReadMotion {
                motion_id: Some("previous".into()),
                threshold: 10.0,
                wait_timeout_s: 0.0,
            },
            Operation::DiscardFrame {
                motion_id: Some("previous".into()),
                wait_timeout_s: 0.0,
            },
            Operation::StopMotion {
                motion_id: Some("previous".into()),
            },
        ] {
            let reply = runtime
                .handle(Request {
                    request_id: "stale".into(),
                    operation,
                })
                .unwrap();
            assert_eq!(reply.error_code, Some("motion_generation_mismatch"));
        }
        assert!(
            runtime
                .handle(Request {
                    request_id: "current".into(),
                    operation: Operation::ReadMotion {
                        motion_id: Some("successor".into()),
                        threshold: 10.0,
                        wait_timeout_s: 0.0,
                    }
                })
                .is_none()
        );
        let deadline = Instant::now() + Duration::from_secs(2);
        let current = loop {
            if let Some(reply) = runtime
                .poll()
                .into_iter()
                .find(|reply| reply.request_id == "current")
            {
                break reply;
            }
            assert!(Instant::now() < deadline);
            thread::sleep(Duration::from_millis(2));
        };
        // Then: The retained frame still compares against the successor's original baseline.
        assert!(current.ok);
        let observation = current.observation.unwrap();
        assert_eq!(observation.changed_pixels, 64);
        assert!(observation.motion);
    }

    fn active_test_recording(runtime: &mut SharedRuntime) {
        runtime.recordings.insert(
            "active".into(),
            Recording {
                url: "test-recording".into(),
                state: Arc::new(Mutex::new(RecordingState {
                    started: true,
                    ..RecordingState::default()
                })),
                failure: Arc::new(Mutex::new(None)),
                stop: Arc::new(AtomicBool::new(false)),
                thread: None,
                start_request: None,
                stop_request: None,
                deadline: Instant::now(),
            },
        );
    }

    #[test]
    fn pending_motion_restart_waits_for_retirement_without_blocking_recording_status() {
        // Given: A stopped worker needs one final receive turn while recording stays active.
        let camera = crate::rtsp_camera::RtspCamera::start("H264");
        let (_poll, mut runtime) = runtime();
        active_test_recording(&mut runtime);
        let (release, retired) = mpsc::sync_channel(1);
        runtime.retired_motion = Some(thread::spawn(move || retired.recv().unwrap()));
        // When: The immediate successor queues its start instead of interpreting retirement as refusal.
        assert!(
            runtime
                .handle(start_motion_request("restart", "next", &camera.url))
                .is_none()
        );
        let extra = runtime
            .handle(start_motion_request("extra", "extra", &camera.url))
            .unwrap();
        assert_eq!(extra.error_code, Some("motion_already_started"));
        let start = Instant::now();
        let status = runtime
            .handle(Request {
                request_id: "status".into(),
                operation: Operation::RecordingStatus {
                    recording_id: "active".into(),
                },
            })
            .unwrap();
        assert!(
            runtime
                .poll()
                .iter()
                .all(|reply| reply.request_id != "restart")
        );
        assert!(start.elapsed() < Duration::from_millis(200));
        release.send(()).unwrap();
        let deadline = Instant::now() + Duration::from_secs(1);
        let restarted = loop {
            if let Some(reply) = runtime
                .poll()
                .into_iter()
                .find(|reply| reply.request_id == "restart")
            {
                break reply;
            }
            assert!(Instant::now() < deadline);
            thread::sleep(Duration::from_millis(2));
        };
        // Then: Startup resumes after reaping, and unrelated recording ownership never changes.
        assert!(status.ok);
        assert_eq!(status.recording_active, Some(true));
        assert!(restarted.ok, "restart failed: {:?}", restarted.error_code);
    }

    #[test]
    fn pending_motion_start_has_generation_scoped_cancel_and_a_bounded_retirement_wait() {
        // Given: A stopped CPU worker remains stuck beyond the retirement budget.
        let (_poll, mut runtime) = runtime();
        let (release, retired) = mpsc::sync_channel(1);
        runtime.retired_motion = Some(thread::spawn(move || retired.recv().unwrap()));
        assert!(
            runtime
                .handle(start_motion_request(
                    "queued",
                    "next",
                    "rtsp://127.0.0.1:9/camera"
                ))
                .is_none()
        );
        // When: An old stop arrives, followed by the pending generation's explicit stop.
        let stale = runtime
            .handle(Request {
                request_id: "old-stop".into(),
                operation: Operation::StopMotion {
                    motion_id: Some("old".into()),
                },
            })
            .unwrap();
        let stopped = runtime
            .handle(Request {
                request_id: "matching-stop".into(),
                operation: Operation::StopMotion {
                    motion_id: Some("next".into()),
                },
            })
            .unwrap();
        let canceled = runtime
            .poll()
            .into_iter()
            .find(|reply| reply.request_id == "queued")
            .unwrap();
        assert!(
            runtime
                .handle(start_motion_request(
                    "bounded",
                    "later",
                    "rtsp://127.0.0.1:9/camera"
                ))
                .is_none()
        );
        let deadline = Instant::now() + Duration::from_secs(1);
        let timed_out = loop {
            if let Some(reply) = runtime
                .poll()
                .into_iter()
                .find(|reply| reply.request_id == "bounded")
            {
                break reply;
            }
            assert!(Instant::now() < deadline);
            thread::sleep(Duration::from_millis(2));
        };
        // Then: Stale cancellation cannot affect startup; a genuine stuck worker has finite refusal.
        assert_eq!(stale.error_code, Some("motion_generation_mismatch"));
        assert!(stopped.ok);
        assert_eq!(canceled.error_code, Some("motion_stopped"));
        assert_eq!(timed_out.error_code, Some("motion_restart_timeout"));
        release.send(()).unwrap();
    }

    #[test]
    fn overlong_motion_generation_is_refused_before_any_worker_starts() {
        // Given: The private generation identifier exceeds its bounded control field.
        let (_poll, mut runtime) = runtime();
        let request =
            start_motion_request("oversized", &"x".repeat(129), "rtsp://127.0.0.1:9/camera");
        // When: Handling startup before camera or CPU worker allocation.
        let reply = runtime.handle(request).unwrap();
        // Then: The native boundary refuses the identifier with a stable reason.
        assert_eq!(reply.error_code, Some("invalid_motion_id"));
    }
}
