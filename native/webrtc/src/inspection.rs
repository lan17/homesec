//! Bounded, private inspection operations. Policy remains in the Python source.
//! Each operation owns its inputs and trial writers; no library diagnostics or
//! media leave this process. The parent supervisor bounds stuck OS/native work.
use crate::camera_input::InspectionInput;
use crate::inspection_hardware;
use crate::protocol::{self, Control};
use ffmpeg::{Rational, codec, filter, format, frame, media};
use ffmpeg_next as ffmpeg;
use serde::{Deserialize, Serialize};
use std::fs::OpenOptions;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, mpsc};
use std::thread;
use std::time::{Duration, Instant};

type Result<T> = std::result::Result<T, &'static str>;
const MAX_PACKETS: usize = 16_384;
const MAX_STREAMS: usize = 16;
const INSPECTION_VERSION: u32 = 1;

#[derive(Deserialize)]
struct Request {
    request_id: String,
    #[serde(flatten)]
    operation: Operation,
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct InputRequest {
    rtsp_url: String,
    operation_timeout_s: f64,
    connect_timeout_s: f64,
    io_timeout_s: f64,
}

#[derive(Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum AudioMode {
    Copy,
    Aac,
    None,
}
#[derive(Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum TimestampMode {
    Stream,
    Wallclock,
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct Role {
    role: String,
    rtsp_url: String,
    kind: RoleKind,
    #[serde(default = "software")]
    backend: String,
    device: Option<String>,
    #[serde(default = "silent")]
    audio_mode: AudioMode,
    #[serde(default = "stream_timestamps")]
    timestamp_mode: TimestampMode,
    output_path: Option<String>,
}
#[derive(Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
enum RoleKind {
    Metadata,
    Decode,
    Recording,
}
fn software() -> String {
    "software".into()
}
fn silent() -> AudioMode {
    AudioMode::None
}
fn stream_timestamps() -> TimestampMode {
    TimestampMode::Stream
}

#[derive(Deserialize)]
#[serde(tag = "command", rename_all = "snake_case")]
enum Operation {
    InspectCapabilities {
        backend: Option<String>,
        device: Option<String>,
    },
    InspectMetadata {
        #[serde(flatten)]
        input: InputRequest,
    },
    InspectDecode {
        #[serde(flatten)]
        input: InputRequest,
        backend: String,
        device: Option<String>,
        sample_frames: usize,
    },
    InspectRecording {
        #[serde(flatten)]
        input: InputRequest,
        audio_mode: AudioMode,
        timestamp_mode: TimestampMode,
        output_path: String,
        duration_s: f64,
    },
    InspectOverlap {
        operation_timeout_s: f64,
        connect_timeout_s: f64,
        io_timeout_s: f64,
        roles: Vec<Role>,
        overlap_s: f64,
        duration_s: f64,
    },
    Stop,
}

#[derive(Default, Serialize)]
struct Reply {
    request_id: String,
    ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    event: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    inspection_version: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    inspection: Option<Evidence>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error_code: Option<&'static str>,
}
#[derive(Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub(crate) enum Evidence {
    Capabilities {
        #[serde(flatten)]
        capabilities: inspection_hardware::Capabilities,
    },
    Metadata(Metadata),
    Decode {
        frames_decoded: usize,
        backend: String,
    },
    Recording {
        finalized: bool,
        video_packets: usize,
        audio_packets: usize,
        output_bytes: u64,
        signals: Signals,
    },
    Overlap {
        status: &'static str,
        roles: Vec<RoleEvidence>,
    },
}
#[derive(Serialize)]
pub(crate) struct Metadata {
    video_codec: String,
    width: u32,
    height: u32,
    fps: Option<f64>,
    fps_raw: Option<String>,
    audio_codec: Option<String>,
    audio_sample_rate: Option<u32>,
    audio_channels: Option<u32>,
}
#[derive(Default, Serialize)]
pub(crate) struct Signals {
    non_monotonic_dts: usize,
    queue_input_backward: usize,
    dts_discontinuity: usize,
    sei_truncated: usize,
}
#[derive(Default)]
struct TimingClock {
    first_dts: Option<i64>,
    previous_dts: Option<i64>,
    previous_pts: Option<i64>,
    samples: usize,
}
#[derive(Serialize)]
pub(crate) struct RoleEvidence {
    role: String,
    ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    error_code: Option<&'static str>,
    frames_decoded: usize,
    video_packets: usize,
    audio_packets: usize,
}

fn seconds(value: f64) -> Result<Duration> {
    if !value.is_finite() || value <= 0.0 || value > 120.0 {
        return Err("invalid_inspection_request");
    }
    Ok(Duration::from_secs_f64(value))
}
impl InputRequest {
    fn open(&self, cancel: Arc<AtomicBool>, wallclock_timestamps: bool) -> Result<InspectionInput> {
        let operation = seconds(self.operation_timeout_s)?;
        InspectionInput::open(
            &self.rtsp_url,
            seconds(self.connect_timeout_s)?,
            seconds(self.io_timeout_s)?,
            Instant::now() + operation,
            cancel,
            wallclock_timestamps,
        )
    }
}
fn input_context(input: &InspectionInput) -> Result<&format::context::Input> {
    let input = input.input.as_ref().ok_or("connection_failed")?;
    if input.nb_streams() as usize > MAX_STREAMS {
        return Err("invalid_video");
    }
    Ok(input)
}
fn video_stream(input: &InspectionInput) -> Result<format::stream::Stream<'_>> {
    input_context(input)?
        .streams()
        .find(|stream| stream.parameters().medium() == media::Type::Video)
        .ok_or("no_video")
}
fn metadata(input: &InspectionInput) -> Result<Metadata> {
    metadata_context(input_context(input)?)
}
fn metadata_context(input: &format::context::Input) -> Result<Metadata> {
    let video = input
        .streams()
        .find(|stream| stream.parameters().medium() == media::Type::Video)
        .ok_or("no_video")?;
    let parameters = video.parameters();
    // SAFETY: parameters owns a live AVCodecParameters; this read cannot escape.
    let raw = unsafe { &*parameters.as_ptr() };
    if raw.width <= 0 || raw.height <= 0 || raw.width > 16_384 || raw.height > 16_384 {
        return Err("invalid_video");
    }
    let rate = [video.avg_frame_rate(), video.rate()]
        .into_iter()
        .find(|rate| rate.numerator() > 0 && rate.denominator() > 0);
    let audio = input
        .streams()
        .find(|stream| stream.parameters().medium() == media::Type::Audio);
    let (audio_codec, audio_sample_rate, audio_channels) = if let Some(audio) = audio {
        let parameters = audio.parameters();
        // SAFETY: same bounded, owned parameter borrow as video above.
        let raw = unsafe { &*parameters.as_ptr() };
        (
            Some(parameters.id().name().into()),
            u32::try_from(raw.sample_rate)
                .ok()
                .filter(|value| *value > 0),
            u32::try_from(raw.ch_layout.nb_channels)
                .ok()
                .filter(|value| *value > 0),
        )
    } else {
        (None, None, None)
    };
    Ok(Metadata {
        video_codec: parameters.id().name().into(),
        width: raw.width as u32,
        height: raw.height as u32,
        fps: rate.map(|rate| f64::from(rate.numerator()) / f64::from(rate.denominator())),
        fps_raw: rate.map(|rate| format!("{}/{}", rate.numerator(), rate.denominator())),
        audio_codec,
        audio_sample_rate,
        audio_channels,
    })
}
fn normalized(error: &'static str) -> &'static str {
    match error {
        "rtsp_timeout" => "timeout",
        "rtsp_connection_failed" => "connection_failed",
        "local_hardware_unavailable" => "local_capability_unavailable",
        "inspection_decode_failed" | "inspection_mux_failed" | "inspection_audio_failed" => {
            "profile_validation_failed"
        }
        "rtsp_source_failed" | "rtsp_packet_invalid" | "rtsp_packet_corrupt" => {
            "profile_validation_failed"
        }
        other => other,
    }
}

/// The reader remains responsive while native work runs. Stop/EOF cancels the
/// actual operation, and a stuck native thread makes the entire helper exit.
pub fn run() -> std::result::Result<(), Box<dyn std::error::Error>> {
    ffmpeg::util::log::set_level(ffmpeg::util::log::Level::Quiet);
    ffmpeg::init().map_err(|_| "local_capability_unavailable")?;
    let mut poll = mio::Poll::new()?;
    let waker = Arc::new(mio::Waker::new(poll.registry(), mio::Token(0))?);
    let requests = protocol::controls(waker, |bytes| {
        if bytes.len() > 128_000 {
            return None;
        }
        let request = serde_json::from_slice::<Request>(bytes).ok()?;
        if request.request_id.len() > 256 {
            return None;
        }
        Some(request)
    });
    protocol::output(&Reply {
        ok: true,
        event: Some("ready"),
        inspection_version: Some(INSPECTION_VERSION),
        ..Reply::default()
    })?;
    let mut events = mio::Events::with_capacity(1);
    let mut active: Option<(
        Arc<AtomicBool>,
        mpsc::Receiver<Reply>,
        thread::JoinHandle<()>,
    )> = None;
    loop {
        while let Ok(control) = requests.try_recv() {
            match control {
                Control::Request(Request {
                    request_id,
                    operation: Operation::Stop,
                }) => {
                    if let Some((cancel, _, worker)) = active.take() {
                        cancel.store(true, Ordering::Release);
                        let deadline = Instant::now() + Duration::from_millis(400);
                        while !worker.is_finished() && Instant::now() < deadline {
                            thread::sleep(Duration::from_millis(5));
                        }
                        if !worker.is_finished() {
                            return Err("inspection_cleanup_failed".into());
                        }
                        let _ = worker.join();
                    }
                    protocol::output(&Reply {
                        request_id,
                        ok: true,
                        ..Reply::default()
                    })?;
                    return Ok(());
                }
                Control::End => {
                    if let Some((cancel, _, worker)) = active.take() {
                        cancel.store(true, Ordering::Release);
                        let deadline = Instant::now() + Duration::from_millis(400);
                        while !worker.is_finished() && Instant::now() < deadline {
                            thread::sleep(Duration::from_millis(5));
                        }
                        if !worker.is_finished() {
                            return Err("inspection_cleanup_failed".into());
                        }
                        let _ = worker.join();
                    }
                    return Ok(());
                }
                Control::Request(request) => {
                    if active.is_some() {
                        protocol::output(&Reply {
                            request_id: request.request_id,
                            error_code: Some("inspection_busy"),
                            ..Reply::default()
                        })?;
                        continue;
                    }
                    let cancel = Arc::new(AtomicBool::new(false));
                    let operation_cancel = Arc::clone(&cancel);
                    let (send, receive) = mpsc::sync_channel(1);
                    let worker = thread::spawn(move || {
                        let outcome =
                            execute(request.operation, operation_cancel).map_err(normalized);
                        let mut reply = Reply {
                            request_id: request.request_id,
                            ..Reply::default()
                        };
                        match outcome {
                            Ok(evidence) => {
                                reply.ok = true;
                                reply.inspection = Some(evidence);
                            }
                            Err(error) => reply.error_code = Some(error),
                        }
                        let _ = send.send(reply);
                    });
                    active = Some((cancel, receive, worker));
                }
            }
        }
        if let Some((_, receiver, _)) = &active {
            match receiver.try_recv() {
                Ok(reply) => {
                    let (_, _, worker) = active.take().ok_or("inspection_worker_failed")?;
                    worker.join().map_err(|_| "inspection_worker_failed")?;
                    if reply.error_code == Some("inspection_cleanup_failed") {
                        return Err("inspection_cleanup_failed".into());
                    }
                    protocol::output(&reply)?;
                }
                Err(mpsc::TryRecvError::Disconnected) => {
                    return Err("inspection_worker_failed".into());
                }
                Err(mpsc::TryRecvError::Empty) => {}
            }
        }
        poll.poll(&mut events, Some(Duration::from_millis(5)))?;
    }
}
fn execute(operation: Operation, cancel: Arc<AtomicBool>) -> Result<Evidence> {
    match operation {
        Operation::InspectCapabilities { backend, device } => {
            bounded_backend(backend.as_deref(), device.as_deref())?;
            Ok(Evidence::Capabilities {
                capabilities: inspection_hardware::capabilities(
                    backend.as_deref(),
                    device.as_deref(),
                ),
            })
        }
        Operation::InspectMetadata { input } => {
            metadata(&input.open(cancel, false)?).map(Evidence::Metadata)
        }
        Operation::InspectDecode {
            input,
            backend,
            device,
            sample_frames,
        } => {
            if !(1..=32).contains(&sample_frames) {
                return Err("invalid_inspection_request");
            }
            bounded_backend(Some(&backend), device.as_deref())?;
            let mut input = input.open(cancel, false)?;
            let frames_decoded =
                decode(&mut input, &backend, device.as_deref(), sample_frames, None)?;
            Ok(Evidence::Decode {
                frames_decoded,
                backend,
            })
        }
        Operation::InspectRecording {
            input,
            audio_mode,
            timestamp_mode,
            output_path,
            duration_s,
        } => {
            let duration = seconds(duration_s)?;
            let input = input.open(cancel, timestamp_mode == TimestampMode::Wallclock)?;
            trial(input, &output_path, audio_mode, duration, None)
        }
        Operation::InspectOverlap {
            operation_timeout_s,
            connect_timeout_s,
            io_timeout_s,
            roles,
            overlap_s,
            duration_s,
        } => overlap(
            operation_timeout_s,
            connect_timeout_s,
            io_timeout_s,
            roles,
            seconds(overlap_s)?,
            seconds(duration_s)?,
            cancel,
        ),
        Operation::Stop => Err("invalid_inspection_request"),
    }
}
fn drained(error: ffmpeg::Error) -> bool {
    error == ffmpeg::Error::Eof
        || error
            == ffmpeg::Error::Other {
                errno: ffmpeg::error::EAGAIN,
            }
}
fn bounded_backend(backend: Option<&str>, device: Option<&str>) -> Result<()> {
    if backend.is_some_and(|value| value.len() > 64 || value.chars().any(char::is_control))
        || device.is_some_and(|value| value.len() > 4096 || value.chars().any(char::is_control))
    {
        return Err("invalid_inspection_request");
    }
    Ok(())
}
fn decode(
    input: &mut InspectionInput,
    backend: &str,
    device: Option<&str>,
    sample_frames: usize,
    until: Option<Instant>,
) -> Result<usize> {
    decode_media(input, backend, device, sample_frames, until, false).map(|(frames, _)| frames)
}

fn decode_media(
    input: &mut InspectionInput,
    backend: &str,
    device: Option<&str>,
    sample_frames: usize,
    until: Option<Instant>,
    include_audio: bool,
) -> Result<(usize, usize)> {
    bounded_backend(Some(backend), device)?;
    let video = video_stream(input)?;
    let index = video.index();
    let mut decoder =
        inspection_hardware::open_video_decoder(&video.parameters(), backend, device)?;
    decoder.set_packet_time_base(video.time_base());
    let audio = if include_audio {
        input_context(input)?
            .streams()
            .find(|stream| stream.parameters().medium() == media::Type::Audio)
    } else {
        None
    };
    let audio_index = audio.as_ref().map(|stream| stream.index());
    let mut audio_decoder = audio
        .map(|stream| {
            if codec::decoder::find(stream.parameters().id()).is_none() {
                return Err("local_capability_unavailable");
            }
            codec::Context::from_parameters(stream.parameters())
                .map_err(|_| "profile_validation_failed")?
                .decoder()
                .audio()
                .map_err(|_| "profile_validation_failed")
        })
        .transpose()?;
    let mut frames = 0;
    let mut audio_frames = 0;
    let mut audio_packets = 0;
    for _ in 0..MAX_PACKETS {
        let Some(packet) = input.read()? else {
            break;
        };
        if packet.stream() == index {
            decoder
                .send_packet(&packet)
                .map_err(|_| "profile_validation_failed")?;
            receive_video(&mut decoder, backend, &mut frames)?;
        } else if Some(packet.stream()) == audio_index
            && let Some(decoder) = &mut audio_decoder
        {
            decoder
                .send_packet(&packet)
                .map_err(|_| "profile_validation_failed")?;
            receive_audio(decoder, &mut audio_frames)?;
            audio_packets += 1;
        }
        if frames >= sample_frames
            && (audio_decoder.is_none() || audio_frames > 0)
            && until.is_none_or(|until| Instant::now() >= until)
        {
            return Ok((frames, audio_packets));
        }
    }
    decoder
        .send_eof()
        .map_err(|_| "profile_validation_failed")?;
    receive_video(&mut decoder, backend, &mut frames)?;
    if frames < sample_frames
        || (audio_decoder.is_some() && audio_frames == 0)
        || until.is_some_and(|until| Instant::now() < until)
    {
        return Err("profile_validation_failed");
    }
    Ok((frames, audio_packets))
}

fn receive_audio(decoder: &mut codec::decoder::Audio, frames: &mut usize) -> Result<()> {
    for _ in 0..32 {
        let mut frame = frame::Audio::empty();
        match decoder.receive_frame(&mut frame) {
            Ok(()) if frame.samples() > 0 => *frames += 1,
            Ok(()) => return Err("profile_validation_failed"),
            Err(error) if drained(error) => return Ok(()),
            Err(_) => return Err("profile_validation_failed"),
        }
    }
    Err("inspection_sample_overflow")
}
fn receive_video(
    decoder: &mut codec::decoder::Video,
    backend: &str,
    frames: &mut usize,
) -> Result<()> {
    for _ in 0..32 {
        let mut frame = frame::Video::empty();
        match decoder.receive_frame(&mut frame) {
            Ok(())
                if frame.width() > 0
                    && frame.height() > 0
                    && inspection_hardware::decoded_frame_backend(&frame, backend) =>
            {
                *frames += 1
            }
            Ok(()) => return Err("profile_validation_failed"),
            Err(error) if drained(error) => return Ok(()),
            Err(_) => return Err("profile_validation_failed"),
        }
    }
    Err("inspection_sample_overflow")
}

struct AudioConverter {
    decoder: codec::decoder::Audio,
    encoder: codec::encoder::Audio,
    graph: filter::Graph,
    encoder_time_base: Rational,
    output_time_base: Rational,
    packets: usize,
}
impl AudioConverter {
    fn new(
        parameters: codec::Parameters,
        time_base: Rational,
        output: &mut format::context::Output,
    ) -> Result<Self> {
        if codec::decoder::find(parameters.id()).is_none() {
            return Err("local_capability_unavailable");
        }
        let mut decoder = codec::Context::from_parameters(parameters)
            .map_err(|_| "profile_validation_failed")?
            .decoder()
            .audio()
            .map_err(|_| "profile_validation_failed")?;
        decoder.set_packet_time_base(time_base);
        let codec = codec::encoder::find(codec::Id::AAC)
            .ok_or("local_capability_unavailable")?
            .audio()
            .map_err(|_| "local_capability_unavailable")?;
        let rate = decoder.rate();
        if rate == 0 || rate > 192_000 || decoder.channels() == 0 || decoder.channels() > 8 {
            return Err("profile_validation_failed");
        }
        let mut layout = decoder.channel_layout();
        if layout.is_empty() {
            layout = ffmpeg::ChannelLayout::default(i32::from(decoder.channels()));
            decoder.set_channel_layout(layout);
        }
        let mut encoder = codec::Context::new_with_codec(*codec)
            .encoder()
            .audio()
            .map_err(|_| "local_capability_unavailable")?;
        encoder.set_flags(codec::flag::Flags::GLOBAL_HEADER);
        encoder.set_rate(rate as i32);
        encoder.set_channel_layout(layout);
        encoder.set_format(
            codec
                .formats()
                .and_then(|mut formats| formats.next())
                .ok_or("local_capability_unavailable")?,
        );
        encoder.set_bit_rate(128_000);
        encoder.set_time_base((1, rate as i32));
        let encoder = encoder
            .open_as(codec)
            .map_err(|_| "local_capability_unavailable")?;
        let mut stream = output
            .add_stream(codec)
            .map_err(|_| "profile_validation_failed")?;
        stream.set_parameters(&encoder);
        stream.set_time_base((1, rate as i32));
        let mut graph = filter::Graph::new();
        let arguments = format!(
            "time_base={}/{}:sample_rate={}:sample_fmt={}:channel_layout=0x{:x}",
            time_base.numerator(),
            time_base.denominator(),
            rate,
            decoder.format().name(),
            layout.bits()
        );
        graph
            .add(
                &filter::find("abuffer").ok_or("local_capability_unavailable")?,
                "in",
                &arguments,
            )
            .map_err(|_| "profile_validation_failed")?;
        graph
            .add(
                &filter::find("abuffersink").ok_or("local_capability_unavailable")?,
                "out",
                "",
            )
            .map_err(|_| "profile_validation_failed")?;
        // Explicit aformat avoids obsolete buffersink AVOption setters whose
        // channel_layouts representation changed in FFmpeg 8.
        let filter = format!(
            "anull,aformat=sample_fmts={}:sample_rates={}:channel_layouts=0x{:x}",
            encoder.format().name(),
            rate,
            layout.bits()
        );
        graph
            .output("in", 0)
            .map_err(|_| "profile_validation_failed")?
            .input("out", 0)
            .map_err(|_| "profile_validation_failed")?
            .parse(&filter)
            .map_err(|_| "local_capability_unavailable")?;
        graph
            .validate()
            .map_err(|_| "local_capability_unavailable")?;
        graph
            .get("out")
            .ok_or("local_capability_unavailable")?
            .sink()
            .set_frame_size(encoder.frame_size());
        Ok(Self {
            decoder,
            encoder,
            graph,
            encoder_time_base: Rational(1, rate as i32),
            output_time_base: Rational(1, rate as i32),
            packets: 0,
        })
    }
    fn packet(
        &mut self,
        packet: &ffmpeg::Packet,
        output: &mut format::context::Output,
    ) -> Result<()> {
        self.decoder
            .send_packet(packet)
            .map_err(|_| "profile_validation_failed")?;
        self.decoded(output)
    }
    fn decoded(&mut self, output: &mut format::context::Output) -> Result<()> {
        for _ in 0..64 {
            let mut decoded = frame::Audio::empty();
            match self.decoder.receive_frame(&mut decoded) {
                Ok(()) => {
                    let timestamp = decoded.timestamp().or(decoded.pts());
                    decoded.set_pts(timestamp);
                    self.graph
                        .get("in")
                        .ok_or("profile_validation_failed")?
                        .source()
                        .add(&decoded)
                        .map_err(|_| "profile_validation_failed")?;
                    self.filtered(output)?;
                }
                Err(error) if drained(error) => return Ok(()),
                Err(_) => return Err("profile_validation_failed"),
            }
        }
        Err("inspection_sample_overflow")
    }
    fn filtered(&mut self, output: &mut format::context::Output) -> Result<()> {
        for _ in 0..64 {
            let mut frame = frame::Audio::empty();
            match self
                .graph
                .get("out")
                .ok_or("profile_validation_failed")?
                .sink()
                .frame(&mut frame)
            {
                Ok(()) => {
                    self.encoder
                        .send_frame(&frame)
                        .map_err(|_| "profile_validation_failed")?;
                    self.encoded(output)?;
                }
                Err(error) if drained(error) => return Ok(()),
                Err(_) => return Err("profile_validation_failed"),
            }
        }
        Err("inspection_sample_overflow")
    }
    fn encoded(&mut self, output: &mut format::context::Output) -> Result<()> {
        for _ in 0..64 {
            let mut packet = ffmpeg::Packet::empty();
            match self.encoder.receive_packet(&mut packet) {
                Ok(()) => {
                    packet.set_stream(1);
                    packet.rescale_ts(self.encoder_time_base, self.output_time_base);
                    packet
                        .write_interleaved(output)
                        .map_err(|_| "profile_validation_failed")?;
                    self.packets += 1;
                }
                Err(error) if drained(error) => return Ok(()),
                Err(_) => return Err("profile_validation_failed"),
            }
        }
        Err("inspection_sample_overflow")
    }
    fn finish(&mut self, output: &mut format::context::Output) -> Result<()> {
        self.decoder
            .send_eof()
            .map_err(|_| "profile_validation_failed")?;
        self.decoded(output)?;
        self.graph
            .get("in")
            .ok_or("profile_validation_failed")?
            .source()
            .flush()
            .map_err(|_| "profile_validation_failed")?;
        self.filtered(output)?;
        self.encoder
            .send_eof()
            .map_err(|_| "profile_validation_failed")?;
        self.encoded(output)
    }
}

fn copy_stream(
    output: &mut format::context::Output,
    parameters: codec::Parameters,
    time_base: Rational,
) -> Result<()> {
    let mut stream = output
        .add_stream(codec::encoder::find(codec::Id::None))
        .map_err(|_| "local_capability_unavailable")?;
    stream.set_parameters(parameters);
    stream.set_time_base(time_base);
    // SAFETY: the newly allocated output stream owns this parameter object.
    unsafe {
        (*stream.parameters().as_mut_ptr()).codec_tag = 0;
    }
    Ok(())
}
fn ticks(microseconds: i64, time_base: Rational) -> Result<i64> {
    if time_base.numerator() <= 0 || time_base.denominator() <= 0 {
        return Err("profile_validation_failed");
    }
    i64::try_from(
        i128::from(microseconds) * i128::from(time_base.denominator())
            / (1_000_000 * i128::from(time_base.numerator())),
    )
    .map_err(|_| "profile_validation_failed")
}
fn microseconds(ticks: i64, time_base: Rational) -> Result<i64> {
    if time_base.numerator() <= 0 || time_base.denominator() <= 0 {
        return Err("profile_validation_failed");
    }
    i64::try_from(
        i128::from(ticks) * i128::from(time_base.numerator()) * 1_000_000
            / i128::from(time_base.denominator()),
    )
    .map_err(|_| "profile_validation_failed")
}
fn trial(
    input: InspectionInput,
    output_path: &str,
    audio_mode: AudioMode,
    duration: Duration,
    until: Option<Instant>,
) -> Result<Evidence> {
    let path = Path::new(output_path);
    if output_path.len() > 4096 || !path.is_absolute() {
        return Err("invalid_inspection_request");
    }
    // The output is exclusively created and the writer is dropped before any
    // failure cleanup. A symlink or pre-existing trial is never overwritten.
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|_| "profile_validation_failed")?;
    let result = trial_writer(input, file, audio_mode, duration, until);
    if result.is_err() {
        let _ = std::fs::remove_file(path);
    }
    result
}
fn trial_writer(
    input: InspectionInput,
    file: std::fs::File,
    audio_mode: AudioMode,
    duration: Duration,
    until: Option<Instant>,
) -> Result<Evidence> {
    let sync_file = file.try_clone().map_err(|_| "profile_validation_failed")?;
    let (video_packets, audio_packets, signals) =
        trial_mux(input, file, audio_mode, duration, until)?;

    sync_file
        .sync_all()
        .map_err(|_| "profile_validation_failed")?;
    let output_bytes = sync_file
        .metadata()
        .map_err(|_| "profile_validation_failed")?
        .len();
    if output_bytes <= 32 {
        return Err("profile_validation_failed");
    }
    Ok(Evidence::Recording {
        finalized: true,
        video_packets,
        audio_packets,
        output_bytes,
        signals,
    })
}
// libav does not consistently retain seek failures in AVIOContext.error. Keep
// the actual I/O outcome so a failed MP4 trailer cannot pass as finalized.
struct TrialWriter<T> {
    inner: T,
    failed: Arc<AtomicBool>,
}
impl<T: std::io::Write> std::io::Write for TrialWriter<T> {
    fn write(&mut self, data: &[u8]) -> std::io::Result<usize> {
        let result = self.inner.write(data);
        if result.is_err() || matches!(result, Ok(0)) && !data.is_empty() {
            self.failed.store(true, Ordering::Release);
        }
        result
    }
    fn flush(&mut self) -> std::io::Result<()> {
        let result = self.inner.flush();
        if result.is_err() {
            self.failed.store(true, Ordering::Release);
        }
        result
    }
}
impl<T: std::io::Seek> std::io::Seek for TrialWriter<T> {
    fn seek(&mut self, position: std::io::SeekFrom) -> std::io::Result<u64> {
        let result = self.inner.seek(position);
        if result.is_err() {
            self.failed.store(true, Ordering::Release);
        }
        result
    }
}

fn trial_mux(
    mut input: InspectionInput,
    writer: impl std::io::Write + std::io::Seek + Send + 'static,
    audio_mode: AudioMode,
    duration: Duration,
    until: Option<Instant>,
) -> Result<(usize, usize, Signals)> {
    let failed = Arc::new(AtomicBool::new(false));
    let writer = TrialWriter {
        inner: writer,
        failed: Arc::clone(&failed),
    };
    let io = format::context::StreamIo::from_write_seek_with_capacity(writer, 32 * 1024)
        .map_err(|_| "profile_validation_failed")?;
    let video = video_stream(&input)?;
    let video_index = video.index();
    let video_time_base = video.time_base();
    let video_codec = video.parameters().id();
    let info = metadata(&input)?;
    if info.width == 0 || info.height == 0 {
        return Err("invalid_video");
    }
    // Generalized trial eligibility intentionally does not broaden the native
    // shared recorder's separate H.264/AAC-LC compatibility gate.
    let audio = input_context(&input)?
        .streams()
        .find(|stream| stream.parameters().medium() == media::Type::Audio);
    let audio_index = audio.as_ref().map(|stream| stream.index());
    let audio_time_base = audio.as_ref().map(|stream| stream.time_base());
    if audio_mode != AudioMode::None && audio.is_none() {
        return Err("profile_validation_failed");
    }
    if audio_mode == AudioMode::Copy
        && !audio.as_ref().is_some_and(|stream| {
            matches!(
                stream.parameters().id(),
                codec::Id::AAC
                    | codec::Id::AC3
                    | codec::Id::ALAC
                    | codec::Id::EAC3
                    | codec::Id::MP2
                    | codec::Id::MP3
            )
        })
    {
        return Err("profile_validation_failed");
    }
    let mut output = format::output_to_stream(io, None, Some("mp4"))
        .map_err(|_| "local_capability_unavailable")?;
    copy_stream(&mut output, video.parameters().clone(), video_time_base)?;
    let mut converter = if audio_mode == AudioMode::Aac {
        let audio = audio.as_ref().ok_or("profile_validation_failed")?;
        Some(AudioConverter::new(
            audio.parameters().clone(),
            audio.time_base(),
            &mut output,
        )?)
    } else {
        if audio_mode == AudioMode::Copy {
            let audio = audio.as_ref().ok_or("profile_validation_failed")?;
            copy_stream(&mut output, audio.parameters().clone(), audio.time_base())?;
        }
        None
    };
    output
        .write_header()
        .map_err(|_| "profile_validation_failed")?;
    let video_output_time_base = output
        .stream(0)
        .ok_or("profile_validation_failed")?
        .time_base();
    let audio_output_time_base = output.stream(1).map(|stream| stream.time_base());
    if let Some(converter) = &mut converter {
        converter.output_time_base = audio_output_time_base.ok_or("profile_validation_failed")?;
    }
    let mut media_started = None;
    let mut epoch = None;
    let mut video_packets = 0;
    let mut audio_packets = 0;
    let mut signals = Signals::default();
    let mut clocks = [TimingClock::default(), TimingClock::default()];
    let mut previous_output_dts = [None, None];
    for _ in 0..MAX_PACKETS {
        if until.is_some_and(|end| Instant::now() >= end)
            || (until.is_none()
                && media_started.is_some_and(|start: Instant| start.elapsed() >= duration))
        {
            break;
        }
        let Some(mut packet) = input.read()? else {
            break;
        };
        let (track, source_time_base, output_time_base) = if packet.stream() == video_index {
            (0, video_time_base, video_output_time_base)
        } else if Some(packet.stream()) == audio_index && audio_mode != AudioMode::None {
            (
                1,
                audio_time_base.ok_or("profile_validation_failed")?,
                audio_output_time_base.ok_or("profile_validation_failed")?,
            )
        } else {
            continue;
        };
        if epoch.is_none() {
            if track != 0 || !packet.is_key() || packet.dts().is_none() || packet.pts().is_none() {
                continue;
            }
            epoch = Some(microseconds(
                packet.dts().ok_or("profile_validation_failed")?,
                source_time_base,
            )?);
            media_started = Some(Instant::now());
        }
        let original_dts = packet.dts().ok_or("profile_validation_failed")?;
        let original_pts = packet.pts().ok_or("profile_validation_failed")?;
        if track == 0 && video_codec == codec::Id::H264 {
            signals.sei_truncated += truncated_sei(packet.data().unwrap_or_default());
        }
        let epoch_ticks = ticks(epoch.ok_or("profile_validation_failed")?, source_time_base)?;
        // Wallclock mode is applied by libavformat before discovery buffering,
        // exactly like the existing executor's use_wallclock_as_timestamps.
        // Rebase the resulting effective clock rather than timing dequeue work.
        let (mut pts, mut dts) = (
            original_pts
                .checked_sub(epoch_ticks)
                .ok_or("profile_validation_failed")?,
            original_dts
                .checked_sub(epoch_ticks)
                .ok_or("profile_validation_failed")?,
        );
        observe_timing(&mut signals, &mut clocks[track], dts, pts, track);
        // The compatible CLI repairs backward DTS before muxing and reports a
        // warning. Keep that successful-baseline behavior while returning the
        // measured disorder, so policy can compare an explicit wallclock retry.
        if let Some(previous) = previous_output_dts[track]
            && dts <= previous
        {
            dts = previous.checked_add(1).ok_or("profile_validation_failed")?;
            pts = pts.max(dts);
        }
        previous_output_dts[track] = Some(dts);
        packet.set_pts(Some(pts));
        packet.set_dts(Some(dts));
        if track == 1
            && let Some(converter) = &mut converter
        {
            converter.packet(&packet, &mut output)?;
        } else {
            packet.set_stream(track);
            packet.rescale_ts(source_time_base, output_time_base);
            packet
                .write_interleaved(&mut output)
                .map_err(|_| "profile_validation_failed")?;
            if track == 0 {
                video_packets += 1;
            } else {
                audio_packets += 1;
            }
        }
    }
    if let Some(converter) = &mut converter {
        converter.finish(&mut output)?;
        audio_packets = converter.packets;
    }
    if video_packets == 0
        || (audio_mode != AudioMode::None && audio_packets == 0)
        || until.is_some_and(|until| Instant::now() < until)
    {
        return Err("profile_validation_failed");
    }
    output
        .write_trailer()
        .map_err(|_| "profile_validation_failed")?;
    crate::recording::flush_output(&mut output).map_err(|_| "profile_validation_failed")?;
    if failed.load(Ordering::Acquire) {
        return Err("profile_validation_failed");
    }
    drop(output);
    Ok((video_packets, audio_packets, signals))
}

fn observe_timing(
    signals: &mut Signals,
    clock: &mut TimingClock,
    dts: i64,
    pts: i64,
    track: usize,
) {
    if let Some(last) = clock.previous_dts {
        if dts <= last {
            signals.non_monotonic_dts += 1;
            clock.first_dts = None;
            clock.samples = 0;
        }
        if track == 1 && clock.previous_pts.is_some_and(|previous| pts < previous) {
            signals.queue_input_backward += 1;
        }
        // Match pinned libavformat/demux.c's DTS-discontinuity warning:
        // a forward gap exceeds 1000 times the observed average packet
        // duration. Both backward clocks and discontinuities reset its window.
        if let Some(first) = clock.first_dts
            && clock.samples > 1
            && (i128::from(dts) - i128::from(last)) / 1000
                > (i128::from(last) - i128::from(first)) / (clock.samples - 1) as i128
        {
            signals.dts_discontinuity += 1;
            clock.first_dts = None;
            clock.samples = 0;
        }
    }
    clock.first_dts.get_or_insert(dts);
    clock.previous_dts = Some(dts);
    clock.previous_pts = Some(pts);
    clock.samples += 1;
}
/// Count malformed SEI payload lengths without exposing media or log text.
fn truncated_sei(bytes: &[u8]) -> usize {
    let mut nals = Vec::new();
    if bytes.starts_with(&[0, 0, 1]) || bytes.starts_with(&[0, 0, 0, 1]) {
        let mut starts = Vec::new();
        let mut index = 0;
        while index + 3 <= bytes.len() {
            let length = if bytes[index..].starts_with(&[0, 0, 0, 1]) {
                4
            } else if bytes[index..].starts_with(&[0, 0, 1]) {
                3
            } else {
                index += 1;
                continue;
            };
            starts.push((index, index + length));
            index += length;
        }
        for (index, &(_, start)) in starts.iter().enumerate() {
            let end = starts.get(index + 1).map_or(bytes.len(), |&(end, _)| end);
            nals.push(&bytes[start..end]);
        }
    } else {
        // Container-demuxed AVC packets use bounded length prefixes. Discover
        // the framing only when every declared NAL exactly covers this packet.
        for prefix in (1..=4).rev() {
            let mut candidate = Vec::new();
            let mut index = 0;
            while index + prefix <= bytes.len() {
                let length = bytes[index..index + prefix]
                    .iter()
                    .fold(0_usize, |length, byte| (length << 8) | usize::from(*byte));
                index += prefix;
                if length == 0 || length > bytes.len() - index {
                    break;
                }
                candidate.push(&bytes[index..index + length]);
                index += length;
            }
            if index == bytes.len() && !candidate.is_empty() {
                nals = candidate;
                break;
            }
        }
    }
    let mut count = 0;
    for nal in nals
        .into_iter()
        .filter(|nal| nal.first().is_some_and(|byte| byte & 31 == 6))
    {
        let mut payload = Vec::with_capacity(nal.len());
        let mut zeroes = 0;
        for &byte in &nal[1..] {
            if byte == 3 && zeroes >= 2 {
                zeroes = 0;
                continue;
            }
            payload.push(byte);
            zeroes = if byte == 0 { zeroes + 1 } else { 0 };
        }
        let mut index = 0;
        while index + 1 < payload.len() && payload[index] != 0x80 {
            let mut kind = 0;
            while index < payload.len() && payload[index] == 255 {
                kind += 255;
                index += 1;
            }
            if index >= payload.len() {
                break;
            }
            kind += usize::from(payload[index]);
            index += 1;
            let mut size = 0;
            while index < payload.len() && payload[index] == 255 {
                size += 255;
                index += 1;
            }
            if index >= payload.len() {
                count += usize::from(kind == 764);
                break;
            }
            size += usize::from(payload[index]);
            index += 1;
            if size > payload.len() - index {
                count += usize::from(kind == 764);
                break;
            }
            index += size;
        }
    }
    count
}

fn overlap(
    operation_timeout_s: f64,
    connect_timeout_s: f64,
    io_timeout_s: f64,
    roles: Vec<Role>,
    overlap_duration: Duration,
    duration: Duration,
    cancel: Arc<AtomicBool>,
) -> Result<Evidence> {
    if roles.is_empty()
        || roles.len() > 3
        || roles
            .iter()
            .any(|role| role.role.is_empty() || role.role.len() > 64)
    {
        return Err("invalid_inspection_request");
    }
    let deadline = Instant::now() + seconds(operation_timeout_s)?;
    seconds(connect_timeout_s)?;
    seconds(io_timeout_s)?;
    let gate = Arc::new((
        std::sync::Mutex::new(None::<Result<Instant>>),
        std::sync::Condvar::new(),
    ));
    let (ready_sender, ready) = mpsc::channel();
    let (result_sender, results) = mpsc::channel();
    let count = roles.len();
    let mut workers = Vec::with_capacity(count);
    for (index, role) in roles.into_iter().enumerate() {
        let gate = Arc::clone(&gate);
        let cancel = Arc::clone(&cancel);
        let ready = ready_sender.clone();
        let completed = result_sender.clone();
        workers.push(thread::spawn(move || {
            let request = InputRequest {
                rtsp_url: role.rtsp_url.clone(),
                operation_timeout_s,
                connect_timeout_s,
                io_timeout_s,
            };
            let mut input = match request.open(
                Arc::clone(&cancel),
                matches!(role.kind, RoleKind::Recording)
                    && role.timestamp_mode == TimestampMode::Wallclock,
            ) {
                Ok(input) => input,
                Err(error) => {
                    let _ = ready.send(Err(error));
                    let _ = completed.send((index, failed_role(role.role, error)));
                    return;
                }
            };
            if let Err(error) = metadata(&input) {
                let _ = ready.send(Err(error));
                let _ = completed.send((index, failed_role(role.role, error)));
                return;
            }
            let _ = ready.send(Ok(()));
            let (lock, condition) = &*gate;
            let outcome = (|| {
                let mut state = lock.lock().map_err(|_| "inspection_worker_failed")?;
                while state.is_none() {
                    if cancel.load(Ordering::Acquire) || Instant::now() >= deadline {
                        return Err("timeout");
                    }
                    state = condition
                        .wait_timeout(state, Duration::from_millis(10))
                        .map_err(|_| "inspection_worker_failed")?
                        .0;
                }
                let until = state.ok_or("inspection_worker_failed")??;
                drop(state);
                match role.kind {
                    RoleKind::Metadata => {
                        let video = video_stream(&input)?.index();
                        let mut packets = 0;
                        for _ in 0..MAX_PACKETS {
                            let Some(packet) = input.read()? else {
                                return Err("profile_validation_failed");
                            };
                            if packet.stream() == video {
                                packets += 1;
                            }
                            if packets > 0 && Instant::now() >= until {
                                return Ok((0, packets, 0));
                            }
                        }
                        Err("inspection_sample_overflow")
                    }
                    RoleKind::Decode => decode_media(
                        &mut input,
                        &role.backend,
                        role.device.as_deref(),
                        5,
                        Some(until),
                        role.audio_mode != AudioMode::None,
                    )
                    .map(|(frames, audio)| (frames, 0, audio)),
                    RoleKind::Recording => match trial(
                        input,
                        role.output_path
                            .as_deref()
                            .ok_or("invalid_inspection_request")?,
                        role.audio_mode,
                        duration,
                        Some(until),
                    )? {
                        Evidence::Recording {
                            video_packets,
                            audio_packets,
                            ..
                        } => Ok((0, video_packets, audio_packets)),
                        _ => Err("inspection_worker_failed"),
                    },
                }
            })();
            let evidence = match outcome {
                Ok((frames_decoded, video_packets, audio_packets)) => RoleEvidence {
                    role: role.role,
                    ok: true,
                    error_code: None,
                    frames_decoded,
                    video_packets,
                    audio_packets,
                },
                Err(error) => failed_role(role.role, error),
            };
            let _ = completed.send((index, evidence));
        }));
    }
    drop(ready_sender);
    drop(result_sender);
    let mut opening_error = None;
    for _ in 0..count {
        match ready.recv_timeout(deadline.saturating_duration_since(Instant::now())) {
            Ok(Ok(())) => {}
            Ok(Err(error)) => {
                opening_error = Some(error);
                break;
            }
            Err(_) => {
                opening_error = Some("timeout");
                break;
            }
        }
    }
    {
        let (lock, condition) = &*gate;
        *lock.lock().map_err(|_| "inspection_worker_failed")? = Some(match opening_error {
            Some(error) => Err(error),
            None => Ok(Instant::now() + overlap_duration.max(duration)),
        });
        condition.notify_all();
    }
    let mut evidence = Vec::with_capacity(count);
    for _ in 0..count {
        match results.recv_timeout(deadline.saturating_duration_since(Instant::now())) {
            Ok(result) => evidence.push(result),
            Err(_) => {
                cancel.store(true, Ordering::Release);
                let cleanup_deadline = Instant::now() + Duration::from_millis(400);
                while workers.iter().any(|worker| !worker.is_finished())
                    && Instant::now() < cleanup_deadline
                {
                    thread::sleep(Duration::from_millis(5));
                }
                if workers.iter().any(|worker| !worker.is_finished()) {
                    // A fatal outcome exits the whole helper; no subsequent
                    // operation or successful cleanup reply can race writers.
                    return Err("inspection_cleanup_failed");
                }
                for worker in workers {
                    let _ = worker.join();
                }
                return Err("timeout");
            }
        }
    }
    for worker in workers {
        worker.join().map_err(|_| "inspection_worker_failed")?;
    }
    evidence.sort_by_key(|(index, _)| *index);
    let roles = evidence
        .into_iter()
        .map(|(_, evidence)| evidence)
        .collect::<Vec<_>>();
    let status = if roles.iter().all(|role| role.ok) {
        "supported"
    } else if roles
        .iter()
        .any(|role| role.error_code == Some("session_limit"))
    {
        "session_limit"
    } else {
        "failed"
    };
    Ok(Evidence::Overlap { status, roles })
}
fn failed_role(role: String, error: &'static str) -> RoleEvidence {
    RoleEvidence {
        role,
        ok: false,
        error_code: Some(normalized(error)),
        frames_decoded: 0,
        video_packets: 0,
        audio_packets: 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> String {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/recording/h264-aac-bframes.mp4")
            .to_string_lossy()
            .into_owned()
    }
    fn fixture_input(path: &str) -> InspectionInput {
        InspectionInput::open_inner(
            path,
            Duration::from_secs(2),
            Duration::from_secs(2),
            Instant::now() + Duration::from_secs(5),
            Arc::new(AtomicBool::new(false)),
            false,
        )
        .unwrap()
    }

    #[test]
    fn metadata_does_not_inherit_recorder_or_first_idr_requirements() {
        // Given: A demuxed camera representation with independent metadata.
        let input = fixture_input(&fixture());
        // When: Metadata is inspected before any packet or decoder is requested.
        let metadata = metadata(&input).unwrap();
        // Then: The first video/audio parameters are present without starting recording.
        assert_eq!(metadata.video_codec, "h264");
        assert_eq!(metadata.audio_codec.as_deref(), Some("aac"));
        assert_eq!((metadata.width, metadata.height), (160, 120));
        assert!(metadata.fps.is_some_and(|fps| fps > 0.0));
    }

    #[test]
    fn incomplete_dimensions_fail_and_unavailable_frame_rates_remain_absent() {
        // Given: Libav has described a video stream without dimensions or valid FPS.
        let mut input = unsafe {
            let context = ffmpeg::ffi::avformat_alloc_context();
            assert!(!context.is_null());
            let stream = ffmpeg::ffi::avformat_new_stream(context, std::ptr::null());
            assert!(!stream.is_null());
            (*(*stream).codecpar).codec_type = ffmpeg::ffi::AVMediaType::AVMEDIA_TYPE_VIDEO;
            (*(*stream).codecpar).codec_id = ffmpeg::ffi::AVCodecID::AV_CODEC_ID_H264;
            (*(*stream).codecpar).height = 120;
            format::context::Input::wrap(context)
        };
        // When: The native metadata boundary validates the incomplete description.
        assert!(matches!(metadata_context(&input), Err("invalid_video")));
        // Then: Once dimensions exist, absent/invalid rates are still not invented.
        unsafe {
            let stream = *(*input.as_mut_ptr()).streams;
            (*(*stream).codecpar).width = 160;
            (*stream).avg_frame_rate = ffmpeg::ffi::AVRational { num: 0, den: 0 };
            (*stream).r_frame_rate = ffmpeg::ffi::AVRational { num: 30, den: 0 };
        }
        let metadata = metadata_context(&input).unwrap();
        assert_eq!((metadata.width, metadata.height), (160, 120));
        assert!(metadata.fps.is_none() && metadata.fps_raw.is_none());
    }
    #[test]
    fn software_decode_requires_actual_complete_frames() {
        // Given: Actual compressed H.264 with B-frames and a five-frame sample.
        let mut input = fixture_input(&fixture());
        // When: The native codec implementation validates software decoding.
        let frames = decode(&mut input, "software", None, 5, None).unwrap();
        // Then: Success reports decoded frames, not metadata or process liveness.
        assert!(frames >= 5);
    }
    #[test]
    fn copy_and_aac_trials_finalize_real_video_and_audio() {
        // Given: Existing H.264/AAC media and exclusively owned trial paths.
        let directory = tempfile::tempdir().unwrap();
        for mode in [AudioMode::Copy, AudioMode::Aac, AudioMode::None] {
            let path = directory.path().join(match mode {
                AudioMode::Copy => "copy.mp4",
                AudioMode::Aac => "aac.mp4",
                AudioMode::None => "none.mp4",
            });
            // When: The actual pinned library muxes and finalizes the selected profile.
            let evidence = trial(
                fixture_input(&fixture()),
                path.to_str().unwrap(),
                mode,
                Duration::from_secs(1),
                None,
            )
            .unwrap();
            // Then: Both packet evidence and a readable finalized MP4 establish success.
            let Evidence::Recording {
                finalized,
                video_packets,
                audio_packets,
                output_bytes,
                ..
            } = evidence
            else {
                panic!("recording evidence");
            };
            assert!(finalized && video_packets > 0 && output_bytes > 32);
            assert_eq!(audio_packets > 0, mode != AudioMode::None);
            let output = format::input(&path).unwrap();
            assert_eq!(
                output
                    .streams()
                    .filter(|stream| stream.parameters().medium() == media::Type::Audio)
                    .count(),
                usize::from(mode != AudioMode::None)
            );
        }
    }
    #[test]
    fn cancelled_input_cannot_open_a_camera_or_leave_a_trial() {
        // Given: Cancellation has already won ownership before an operation starts.
        let cancel = Arc::new(AtomicBool::new(true));
        // When: An actual native open is requested with a secret-bearing URL.
        let input = InspectionInput::open(
            "rtsp://secret:password@127.0.0.1:1/camera",
            Duration::from_secs(1),
            Duration::from_secs(1),
            Instant::now() + Duration::from_secs(2),
            cancel,
            false,
        );
        // Then: The stable timeout is returned before connection work can begin.
        assert!(matches!(input, Err("rtsp_timeout")));
    }
    #[test]
    fn timestamp_signals_measure_packet_disorder_and_discontinuity() {
        // Given: A stream clock with a regression and a discontinuity beyond 1000 mean durations.
        let mut signals = Signals::default();
        let mut previous = TimingClock::default();
        // When: Actual packet timestamp values reach the native measurement boundary.
        for timestamp in [100, 90, 100, 200_100] {
            observe_timing(&mut signals, &mut previous, timestamp, timestamp, 1);
        }
        // Then: Python receives the existing independently weighted stability signals.
        assert_eq!(signals.non_monotonic_dts, 1);
        assert_eq!(signals.queue_input_backward, 1);
        assert_eq!(signals.dts_discontinuity, 1);
    }

    #[test]
    fn discontinuity_matches_pinned_average_duration_boundary_and_resets() {
        // Given: One-second packet intervals followed by boundary-sized forward gaps.
        for (next, expected) in [(33_000, 0), (1_002_000, 0), (1_003_000, 1)] {
            let mut clock = TimingClock::default();
            let mut signals = Signals::default();
            // When: Native diagnostics see the same clocks as pinned libavformat.
            for timestamp in [0, 1_000, 2_000, next] {
                observe_timing(&mut signals, &mut clock, timestamp, timestamp, 0);
            }
            // Then: A 31-second gap and exactly 1000 mean durations do not warn.
            assert_eq!(signals.dts_discontinuity, expected);
            if expected == 1 {
                for timestamp in [1_004_000, 1_005_000, 2_006_000] {
                    observe_timing(&mut signals, &mut clock, timestamp, timestamp, 0);
                }
                assert_eq!(
                    signals.dts_discontinuity, 2,
                    "the warning resets its averaging window"
                );
            }
        }
    }

    #[test]
    fn library_trials_preserve_copy_audio_and_convert_g711_and_opus() {
        // Given: Small synthetic inputs spanning the existing MP4 audio allowlist.
        let directory = tempfile::tempdir().unwrap();
        for codec in [
            "ac3",
            "alac",
            "eac3",
            "mp2",
            "mp3",
            "pcm_alaw",
            "pcm_mulaw",
            "libopus",
        ] {
            let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(format!(
                "tests/fixtures/inspection/h264-{codec}.{}",
                if codec == "libopus" { "mp4" } else { "mov" }
            ));
            let audio_mode = if matches!(codec, "pcm_alaw" | "pcm_mulaw" | "libopus") {
                AudioMode::Aac
            } else {
                AudioMode::Copy
            };
            let target = directory.path().join(format!("{codec}.mp4"));
            // When: The real library executes each copy or AAC conversion profile.
            let outcome = trial(
                fixture_input(path.to_str().unwrap()),
                target.to_str().unwrap(),
                audio_mode,
                Duration::from_secs(1),
                None,
            );
            // Then: Each finalized result includes actual video and audio media.
            let Evidence::Recording {
                video_packets,
                audio_packets,
                finalized,
                ..
            } = outcome.unwrap_or_else(|error| panic!("{codec}: {error}"))
            else {
                panic!("recording evidence");
            };
            assert!(
                finalized && video_packets > 0 && audio_packets > 0,
                "{codec}"
            );
            assert!(
                format::input(&target)
                    .unwrap()
                    .streams()
                    .any(|stream| stream.parameters().medium() == media::Type::Audio)
            );
        }
    }

    #[test]
    fn hevc_metadata_decode_and_video_copy_are_independent_of_shared_recorder() {
        // Given: A real synthetic HEVC input, unsupported by the shared recorder.
        let path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/inspection/hevc-aac.mp4");
        let directory = tempfile::tempdir().unwrap();
        let target = directory.path().join("hevc.mp4");
        // When: Inspection discovers, decodes, and copy-trials the same input.
        assert_eq!(
            metadata(&fixture_input(path.to_str().unwrap()))
                .unwrap()
                .video_codec,
            "hevc"
        );
        assert!(
            decode(
                &mut fixture_input(path.to_str().unwrap()),
                "software",
                None,
                5,
                None
            )
            .unwrap()
                >= 5
        );
        let evidence = trial(
            fixture_input(path.to_str().unwrap()),
            target.to_str().unwrap(),
            AudioMode::Copy,
            Duration::from_secs(1),
            None,
        )
        .unwrap();
        // Then: General inspection succeeds without widening strict runtime eligibility.
        assert!(matches!(
            evidence,
            Evidence::Recording {
                finalized: true,
                video_packets: 1..,
                audio_packets: 1..,
                ..
            }
        ));
    }

    #[test]
    fn other_camera_video_is_discovered_and_decoded_without_h264_eligibility() {
        // Given: Synthetic MPEG-4 Part 2 and MJPEG inputs, outside shared eligibility.
        let directory = tempfile::tempdir().unwrap();
        for codec in ["mpeg4", "mjpeg"] {
            let path = Path::new(env!("CARGO_MANIFEST_DIR"))
                .join(format!("tests/fixtures/inspection/{codec}.mov"));
            // When: General inspection discovers, decodes and attempts an MP4 copy trial.
            assert_eq!(
                metadata(&fixture_input(path.to_str().unwrap()))
                    .unwrap()
                    .video_codec,
                codec
            );
            assert!(
                decode(
                    &mut fixture_input(path.to_str().unwrap()),
                    "software",
                    None,
                    5,
                    None
                )
                .unwrap()
                    >= 5
            );
            let target = directory.path().join(format!("{codec}.mp4"));
            let outcome = trial(
                fixture_input(path.to_str().unwrap()),
                target.to_str().unwrap(),
                AudioMode::None,
                Duration::from_secs(1),
                None,
            );
            // Then: Both valid MP4 copy profiles retain their current support.
            assert!(outcome.is_ok(), "{codec}");
        }
    }

    #[test]
    fn existing_trial_targets_and_real_write_failures_cannot_report_success() {
        // Given: An existing target and a separate file descriptor opened read-only.
        let directory = tempfile::tempdir().unwrap();
        let target = directory.path().join("existing.mp4");
        std::fs::write(&target, b"existing content").unwrap();
        let readonly = std::fs::File::open(&target).unwrap();
        // When: Trial creation and actual mux writes meet those filesystem boundaries.
        let existing = trial(
            fixture_input(&fixture()),
            target.to_str().unwrap(),
            AudioMode::None,
            Duration::from_secs(1),
            None,
        );
        let failed_write = trial_writer(
            fixture_input(&fixture()),
            readonly,
            AudioMode::None,
            Duration::from_secs(1),
            None,
        );
        // Then: Neither path reports finalized media or overwrites the existing target.
        assert!(existing.is_err());
        assert!(failed_write.is_err());
        assert_eq!(std::fs::read(target).unwrap(), b"existing content");
    }

    #[test]
    fn actual_mux_trailer_io_failure_cannot_return_finalized_evidence() {
        struct RefuseTrailerSeek {
            bytes: std::io::Cursor<Vec<u8>>,
            seeks: Arc<std::sync::Mutex<Vec<std::io::SeekFrom>>>,
        }
        impl std::io::Write for RefuseTrailerSeek {
            fn write(&mut self, data: &[u8]) -> std::io::Result<usize> {
                self.bytes.write(data)
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        impl std::io::Seek for RefuseTrailerSeek {
            fn seek(&mut self, position: std::io::SeekFrom) -> std::io::Result<u64> {
                if self.bytes.get_ref().len() > 32 {
                    self.seeks.lock().unwrap().push(position);
                    return Err(std::io::Error::other("trial trailer seek refused"));
                }
                self.bytes.seek(position)
            }
        }
        // Given: An actual mux output whose I/O fails when MP4 rewrites its trailer.
        let seeks = Arc::new(std::sync::Mutex::new(Vec::new()));
        let writer = RefuseTrailerSeek {
            bytes: std::io::Cursor::new(Vec::new()),
            seeks: Arc::clone(&seeks),
        };
        // When: Real compressed packets reach the actual MP4 finalization boundary.
        let outcome = trial_mux(
            fixture_input(&fixture()),
            writer,
            AudioMode::Copy,
            Duration::from_secs(1),
            None,
        );
        // Then: The failed trailer seek is observed and no finalized evidence escapes.
        assert!(!seeks.lock().unwrap().is_empty());
        assert!(matches!(outcome, Err("profile_validation_failed")));
    }

    #[test]
    fn sei_diagnostics_parse_annex_b_and_length_prefixed_payloads() {
        // Given: A truncated SEI type 764 payload in both supported packet framings.
        let nal = [6, 255, 255, 254, 30, 1, 2, 0x80];
        let mut annex_b = vec![0, 0, 0, 1];
        annex_b.extend_from_slice(&nal);
        let mut avcc = (nal.len() as u32).to_be_bytes().to_vec();
        avcc.extend_from_slice(&nal);
        // When: The bounded packet parser measures the legacy warning category.
        let signals = [
            truncated_sei(&annex_b),
            truncated_sei(&avcc),
            truncated_sei(&[0, 0, 1, 6, 5, 2, 1, 2, 0x80]),
        ];
        // Then: Both malformed payloads count once, and an ordinary payload does not.
        assert_eq!(signals, [1, 1, 0]);
    }
}
