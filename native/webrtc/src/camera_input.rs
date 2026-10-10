//! One cancellable compressed camera input for independent native consumers.
//!
//! libavformat owns RTSP negotiation, authentication, RTP depacketization and
//! codec timestamp discovery. The dispatcher callback must only perform bounded
//! nonblocking publication: decoding, muxing and socket writes belong to its
//! consumers, never this network thread. Errors contain no camera diagnostics.

use crate::recording::{StreamConfig, Track, initial_video_parameters};
use ffmpeg::{Rational, codec, format, media};
use ffmpeg_next as ffmpeg;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};
use url::Url;

const MAX_PACKET_BYTES: usize = 2 * 1024 * 1024;
const DROP_TIMEOUT: Duration = Duration::from_millis(300);
const CLOSE_TIMEOUT: Duration = Duration::from_millis(200);
type Result<T> = std::result::Result<T, &'static str>;

#[derive(Clone)]
pub struct TrackInfo {
    pub parameters: codec::Parameters,
    pub time_base: Rational,
}

#[derive(Clone)]
pub struct Metadata {
    pub video: TrackInfo,
    pub audio: Option<TrackInfo>,
}

pub enum Event {
    Ready(Metadata),
    Packet {
        track: Track,
        packet: ffmpeg::Packet,
    },
}

struct Interruption {
    cancel: AtomicBool,
    closing: AtomicBool,
    deadline: Mutex<Instant>,
}

impl Interruption {
    fn interrupted(&self) -> bool {
        (self.cancel.load(Ordering::Acquire) && !self.closing.load(Ordering::Acquire))
            || self
                .deadline
                .lock()
                .map_or(true, |deadline| Instant::now() >= *deadline)
    }

    fn refresh(&self, timeout: Duration) -> Result<()> {
        *self.deadline.lock().map_err(|_| "rtsp_source_failed")? = Instant::now() + timeout;
        Ok(())
    }
}

pub struct CameraInput {
    interruption: Arc<Interruption>,
    failure: Arc<Mutex<Option<&'static str>>>,
    finished: mpsc::Receiver<()>,
    worker: Option<JoinHandle<()>>,
}

impl CameraInput {
    pub fn start(
        url: String,
        audio_enabled: bool,
        connect_timeout: Duration,
        io_timeout: Duration,
        publish: impl FnMut(Event) -> Result<()> + Send + 'static,
    ) -> Result<Self> {
        let parsed = Url::parse(&url).map_err(|_| "invalid_rtsp_url")?;
        if url.len() > 16 * 1024
            || parsed.scheme() != "rtsp"
            || parsed.host_str().is_none()
            || parsed.fragment().is_some()
        {
            return Err("invalid_rtsp_url");
        }
        Self::spawn(url, audio_enabled, connect_timeout, io_timeout, publish)
    }

    fn spawn(
        url: String,
        audio_enabled: bool,
        connect_timeout: Duration,
        io_timeout: Duration,
        mut publish: impl FnMut(Event) -> Result<()> + Send + 'static,
    ) -> Result<Self> {
        if [connect_timeout, io_timeout]
            .iter()
            .any(|timeout| timeout.is_zero() || *timeout > Duration::from_secs(120))
        {
            return Err("invalid_rtsp_timeout");
        }
        let interruption = Arc::new(Interruption {
            cancel: AtomicBool::new(false),
            closing: AtomicBool::new(false),
            deadline: Mutex::new(Instant::now() + connect_timeout),
        });
        let failure = Arc::new(Mutex::new(None));
        let worker_interruption = Arc::clone(&interruption);
        let worker_failure = Arc::clone(&failure);
        let (finished_tx, finished) = mpsc::sync_channel(1);
        let worker = thread::Builder::new()
            .name("camera-input".into())
            .spawn(move || {
                let result = ingest(
                    &url,
                    audio_enabled,
                    io_timeout,
                    &worker_interruption,
                    &mut publish,
                );
                if let Err(error) = result
                    && !worker_interruption.cancel.load(Ordering::Acquire)
                    && let Ok(mut slot) = worker_failure.lock()
                {
                    *slot = Some(error);
                }
                let _ = finished_tx.send(());
            })
            .map_err(|_| "rtsp_source_failed")?;
        Ok(Self {
            interruption,
            failure,
            finished,
            worker: Some(worker),
        })
    }

    /// Terminal state is independent of the dispatcher's bounded packet queues.
    pub fn failure(&self) -> Option<&'static str> {
        self.failure
            .lock()
            .map_or(Some("rtsp_source_failed"), |failure| *failure)
    }
}

impl Drop for CameraInput {
    fn drop(&mut self) {
        self.interruption.cancel.store(true, Ordering::Release);
        if self.finished.recv_timeout(DROP_TIMEOUT).is_ok()
            && let Some(worker) = self.worker.take()
        {
            let _ = worker.join();
        }
        // An OS resolver can outlive libavformat's interrupt callback. Never
        // block the owner indefinitely; its process supervisor remains the
        // outer deadline for an uninterruptible platform resolver.
    }
}

fn ingest(
    url: &str,
    audio_enabled: bool,
    io_timeout: Duration,
    interruption: &Arc<Interruption>,
    publish: &mut impl FnMut(Event) -> Result<()>,
) -> Result<()> {
    ffmpeg::util::log::set_level(ffmpeg::util::log::Level::Quiet);
    ffmpeg::init().map_err(|_| "rtsp_source_failed")?;
    let mut options = ffmpeg::Dictionary::new();
    options.set("rtsp_transport", "tcp");
    options.set("probesize", "262144");
    options.set("analyzeduration", "1000000");
    options.set("max_probe_packets", "64");
    options.set("reorder_queue_size", "0");
    options.set("rw_timeout", &io_timeout.as_micros().to_string());
    options.set("max_streams", "16");
    if url.starts_with("rtsp:") {
        options.set("format_whitelist", "rtsp");
        options.set("protocol_whitelist", "tcp,udp,rtp");
    }
    let interrupt = Arc::clone(interruption);
    let mut input =
        format::input_with_interrupt_and_dictionary(&url, move || interrupt.interrupted(), options)
            .map_err(|_| {
                if interruption.interrupted() {
                    "rtsp_timeout"
                } else {
                    "rtsp_connection_failed"
                }
            })?;
    let result = consume_input(&mut input, audio_enabled, io_timeout, interruption, publish);
    // Cancellation must still allow a bounded TEARDOWN on the established
    // connection. The original cancel flag remains set for terminal reporting;
    // no new input is opened, and the interrupt deadline bounds close I/O.
    if interruption.refresh(CLOSE_TIMEOUT).is_ok() {
        interruption.closing.store(true, Ordering::Release);
    }
    drop(input);
    result
}

fn consume_input(
    input: &mut format::context::Input,
    audio_enabled: bool,
    io_timeout: Duration,
    interruption: &Arc<Interruption>,
    publish: &mut impl FnMut(Event) -> Result<()>,
) -> Result<()> {
    let video = input
        .streams()
        .find(|stream| stream.parameters().medium() == media::Type::Video)
        .ok_or("unsupported_codec")?;
    let video_index = video.index();
    let video = TrackInfo {
        parameters: video.parameters().clone(),
        time_base: video.time_base(),
    };
    // This checks codec, dimensions and actual SPS/PPS before any consumer can
    // allocate a decoder or report native recording eligibility.
    StreamConfig::copy(video.parameters.clone(), video.time_base)?;
    let audio = input
        .streams()
        .find(|stream| stream.parameters().medium() == media::Type::Audio);
    let audio_index = audio.as_ref().map(|stream| stream.index());
    let audio = audio.map(|stream| TrackInfo {
        parameters: stream.parameters().clone(),
        time_base: stream.time_base(),
    });
    if audio_enabled {
        let audio = audio.as_ref().ok_or("unsupported_recording_codec")?;
        StreamConfig::copy(audio.parameters.clone(), audio.time_base)?;
    }
    // Discover audio even for a motion-only owner. A later recording consumer
    // can join the same selected stream without opening another RTSP PLAY.
    let mut metadata = Some(Metadata { video, audio });
    let startup_deadline = Instant::now() + io_timeout;
    interruption.refresh(io_timeout)?;
    let mut video_clock_started = false;
    loop {
        if interruption.interrupted() {
            return Err("rtsp_timeout");
        }
        if metadata.is_some() && Instant::now() >= startup_deadline {
            return Err("rtsp_keyframe_timeout");
        }
        let mut packet = ffmpeg::Packet::empty();
        packet.read(input).map_err(|_| {
            if interruption.interrupted() {
                "rtsp_timeout"
            } else {
                "rtsp_source_failed"
            }
        })?;
        let track = if packet.stream() == video_index {
            Track::Video
        } else if Some(packet.stream()) == audio_index {
            Track::Audio
        } else {
            continue;
        };
        if packet.size() == 0 || packet.size() > MAX_PACKET_BYTES {
            return Err("rtsp_packet_invalid");
        }
        if track == Track::Video {
            // RTP startup can produce an incomplete first access unit without
            // any camera clock. Discard that prefix; never invent DTS for
            // streams with reordered pictures. A later clock loss is terminal.
            if packet.pts().is_none() || packet.dts().is_none() {
                if !video_clock_started {
                    continue;
                }
                return Err("rtsp_timestamp_invalid");
            }
            video_clock_started = true;
            interruption.refresh(io_timeout)?;
        }
        // Preserve the untimestamped startup-prefix discard above, but never
        // fan out demuxer-recognized corruption to recording or other consumers.
        validate_packet_integrity(&packet)?;
        if metadata.is_some() {
            if track != Track::Video || !packet.is_key() {
                continue;
            }
            let mut initial = metadata.take().ok_or("rtsp_source_failed")?;
            initial.video.parameters = initial_video_parameters(
                initial.video.parameters,
                initial.video.time_base,
                &packet,
            )?;
            publish(Event::Ready(initial))?;
        }
        publish(Event::Packet { track, packet })?;
    }
}

fn validate_packet_integrity(packet: &ffmpeg::Packet) -> Result<()> {
    if packet.flags().contains(codec::packet::Flags::CORRUPT) {
        return Err("rtsp_packet_corrupt");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rtsp_camera::RtspCamera;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::path::Path;

    #[test]
    fn demuxer_corruption_flag_refuses_packet_fanout() {
        // Given: The demuxer labels a compressed packet corrupt, regardless of its key flag.
        let mut packet = ffmpeg::Packet::copy(&[0x65, 0x80]);
        packet.set_flags(codec::packet::Flags::KEY | codec::packet::Flags::CORRUPT);
        // When: Applying the source-boundary integrity check before publication.
        let outcome = validate_packet_integrity(&packet);
        // Then: Every consumer receives a stable source failure rather than corrupt media.
        assert_eq!(outcome, Err("rtsp_packet_corrupt"));
    }

    fn wait_for_failure(input: &CameraInput) -> &'static str {
        let deadline = Instant::now() + Duration::from_secs(4);
        loop {
            if let Some(error) = input.failure() {
                return error;
            }
            assert!(
                Instant::now() < deadline,
                "input failure must meet its deadline"
            );
            thread::sleep(Duration::from_millis(5));
        }
    }

    fn assert_session_closed(camera: &RtspCamera) {
        // libavformat sends TEARDOWN without awaiting its response. Observe
        // completion at the remote boundary instead of racing the server's
        // request reader immediately after the local input has been dropped.
        let deadline = Instant::now() + Duration::from_millis(500);
        while camera.count("DISCONNECTED") == 0 && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(camera.count("DISCONNECTED"), 1);
        assert_eq!(camera.count("TEARDOWN"), 1);
    }

    #[test]
    fn audio_metadata_and_distinct_packet_timestamps_survive_a_motion_first_input() {
        // Given: One selected source contains H.264 B-frames and AAC, initially used for motion.
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/recording/h264-aac-bframes.mp4");
        let events = Arc::new(Mutex::new(Vec::new()));
        let capture = Arc::clone(&events);
        let input = CameraInput::spawn(
            path.to_string_lossy().into_owned(),
            false,
            Duration::from_secs(3),
            Duration::from_secs(1),
            move |event| {
                capture
                    .lock()
                    .map_err(|_| "test_capture_failed")?
                    .push(event);
                Ok(())
            },
        )
        .unwrap();
        // When: The input reaches EOF without opening another source for recording audio.
        assert_eq!(wait_for_failure(&input), "rtsp_source_failed");
        let events = events.lock().unwrap();
        let Some(Event::Ready(metadata)) = events.first() else {
            panic!("metadata precedes packets");
        };
        // Then: Audio eligibility and original video decode/presentation timing remain available.
        assert_eq!(metadata.video.parameters.id(), codec::Id::H264);
        let audio = metadata.audio.as_ref().unwrap();
        assert_eq!(audio.parameters.id(), codec::Id::AAC);
        StreamConfig::copy(audio.parameters.clone(), audio.time_base).unwrap();
        let video = events
            .iter()
            .filter_map(|event| match event {
                Event::Packet {
                    track: Track::Video,
                    packet,
                } => Some(packet),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(video.len(), 24);
        assert!(video.iter().any(|packet| packet.pts() != packet.dts()));
        assert!(events.iter().any(|event| matches!(
            event,
            Event::Packet {
                track: Track::Audio,
                ..
            }
        )));
        let directory = tempfile::tempdir().unwrap();
        let mut recorder = crate::recording::Mp4Recorder::create(
            &directory.path().join("copied.mp4"),
            StreamConfig::copy(metadata.video.parameters.clone(), metadata.video.time_base)
                .unwrap(),
            Some(StreamConfig::copy(audio.parameters.clone(), audio.time_base).unwrap()),
        )
        .unwrap();
        for event in events.iter() {
            if let Event::Packet { track, packet } = event {
                recorder
                    .push(crate::recording::EncodedPacket {
                        track: *track,
                        data: packet.data().unwrap(),
                        pts: packet.pts().unwrap(),
                        dts: packet.dts().unwrap(),
                        duration: packet.duration(),
                        keyframe: packet.is_key(),
                    })
                    .unwrap();
            }
        }
        recorder.finish().unwrap();
    }

    #[test]
    fn one_camera_input_publishes_complete_video_through_one_play_session() {
        // Given: A fake RTSP camera sends actual H.264 over TCP with SDP-only parameter sets.
        let camera = RtspCamera::start("H264");
        let (sender, events) = mpsc::sync_channel(16);
        let input = CameraInput::start(
            camera.url.clone(),
            false,
            Duration::from_secs(3),
            Duration::from_secs(1),
            move |event| sender.try_send(event).map_err(|_| "test_consumer_full"),
        )
        .unwrap();
        // When: Native libavformat prepares stream metadata and multiple complete packets.
        let metadata = events.recv_timeout(Duration::from_secs(4)).unwrap();
        let Event::Ready(metadata) = metadata else {
            panic!("metadata precedes packets");
        };
        assert_eq!(metadata.video.parameters.id(), codec::Id::H264);
        let mut video = 0;
        while video < 3 {
            if let Event::Packet {
                track: Track::Video,
                packet,
            } = events.recv_timeout(Duration::from_secs(2)).unwrap()
            {
                assert!(packet.size() > 0);
                assert!(packet.size() <= MAX_PACKET_BYTES);
                assert!(packet.pts().is_some());
                assert!(packet.dts().is_some());
                video += 1;
            }
        }
        // Then: Consumers receive packets from one input and cancellation closes its camera session.
        assert_eq!(camera.count("PLAY"), 1);
        let start = Instant::now();
        drop(input);
        assert!(start.elapsed() < Duration::from_millis(700));
        assert_session_closed(&camera);
    }

    #[test]
    fn stale_sdp_parameters_are_replaced_before_source_metadata_is_published() {
        // Given: A real RTSP boundary advertises valid but stale PPS encoder settings.
        let camera = RtspCamera::start("STALE_PPS");
        let (sender, events) = mpsc::sync_channel(32);
        let input = CameraInput::start(
            camera.url.clone(),
            false,
            Duration::from_secs(4),
            Duration::from_secs(2),
            move |event| sender.try_send(event).map_err(|_| "test_consumer_full"),
        )
        .unwrap();
        // When: The input waits for the actual timestamped keyframe and its in-band parameters.
        let Event::Ready(metadata) = events.recv_timeout(Duration::from_secs(5)).unwrap() else {
            panic!("metadata precedes packets");
        };
        let Event::Packet {
            track: Track::Video,
            packet,
        } = events.recv_timeout(Duration::from_secs(2)).unwrap()
        else {
            panic!("complete keyframe follows metadata");
        };
        let directory = tempfile::tempdir().unwrap();
        let mut recorder = crate::recording::Mp4Recorder::create(
            &directory.path().join("clip.mp4"),
            StreamConfig::copy(metadata.video.parameters, metadata.video.time_base).unwrap(),
            None,
        )
        .unwrap();
        let outcome = recorder.push(crate::recording::EncodedPacket {
            track: Track::Video,
            data: packet.data().unwrap(),
            pts: packet.pts().unwrap(),
            dts: packet.dts().unwrap(),
            duration: packet.duration(),
            keyframe: packet.is_key(),
        });
        // Then: The actual initial PPS is the immutable mux header; no stale-parameter refusal occurs.
        assert_eq!(outcome, Ok(crate::recording::WriteOutcome::Written));
        recorder.finish().unwrap();
        drop(input);
        assert_eq!(camera.count("PLAY"), 1);
    }

    fn decoded(path: &Path, audio: bool) -> Vec<u8> {
        let mut command = std::process::Command::new("ffmpeg");
        command.args([
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-xerror",
            "-threads",
            "1",
            "-i",
        ]);
        command.arg(path);
        if audio {
            command.args(["-map", "0:a:0", "-c:a", "pcm_s16le", "-f", "s16le"]);
        } else {
            command.args([
                "-map",
                "0:v:0",
                "-fps_mode",
                "passthrough",
                "-pix_fmt",
                "rgb24",
                "-f",
                "rawvideo",
            ]);
        }
        let output = command.arg("pipe:1").output().unwrap();
        assert!(
            output.status.success(),
            "synthetic decode failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        output.stdout
    }

    #[test]
    fn rtsp_aac_access_units_preserve_the_shared_clock_and_decode_with_copied_video() {
        // Given: One RTSP/TCP source sends H.264 and AAC-LC with common RTCP sender reports.
        // AAC's RTP presentation clock deliberately starts 250ms after video's clock.
        let camera = RtspCamera::start("H264_AAC");
        let (sender, events) = mpsc::sync_channel(128);
        let input = CameraInput::start(
            camera.url.clone(),
            true,
            Duration::from_secs(5),
            Duration::from_secs(2),
            move |event| sender.try_send(event).map_err(|_| "test_consumer_full"),
        )
        .unwrap();
        let Event::Ready(metadata) = events.recv_timeout(Duration::from_secs(6)).unwrap() else {
            panic!("metadata");
        };
        let audio = metadata.audio.as_ref().unwrap();
        assert_eq!(metadata.video.time_base, Rational(1, 90_000));
        assert_eq!(audio.time_base, Rational(1, 48_000));
        assert_eq!(audio.parameters.id(), codec::Id::AAC);
        let mut packets = Vec::new();
        let mut video_count = 0;
        // When: The demuxer delivers complete compressed packets and the native muxer copies them.
        while video_count < 15 {
            let Event::Packet { track, packet } =
                events.recv_timeout(Duration::from_secs(3)).unwrap()
            else {
                panic!("packet");
            };
            if track == Track::Video {
                video_count += 1;
            }
            packets.push((track, packet));
        }
        assert_eq!(input.failure(), None);
        drop(input);
        let (_, first) = packets.first().unwrap();
        assert!(first.is_key());
        assert_eq!(first.pts(), Some(90_000));
        assert_eq!(first.dts(), Some(90_000));
        let epoch = first.dts().unwrap();
        let audio_epoch = epoch * 48_000 / 90_000;
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("copied.mp4");
        let mut recorder = crate::recording::Mp4Recorder::create(
            &path,
            StreamConfig::copy(metadata.video.parameters, metadata.video.time_base).unwrap(),
            Some(StreamConfig::copy(audio.parameters.clone(), audio.time_base).unwrap()),
        )
        .unwrap();
        let fixture_audio = crate::rtsp_camera::aac_access_units();
        let mut expected_audio = Vec::new();
        let mut copied_audio = 0;
        for (track, packet) in &packets {
            let offset = if *track == Track::Video {
                epoch
            } else {
                audio_epoch
            };
            let pts = packet.pts().unwrap();
            let dts = packet.dts().unwrap();
            assert_eq!(pts, dts);
            if *track == Track::Audio {
                assert!(pts >= 12_000);
                assert_eq!((pts - 12_000) % 1024, 0);
                assert_eq!(packet.duration(), 1024);
                let bytes = packet.data().unwrap();
                assert!(fixture_audio.iter().any(|original| original == bytes));
                let length = bytes.len() + 7;
                expected_audio.extend_from_slice(&[
                    0xff,
                    0xf1,
                    0x4c,
                    0x80 | ((length >> 11) as u8 & 3),
                    (length >> 3) as u8,
                    ((length as u8 & 7) << 5) | 0x1f,
                    0xfc,
                ]);
                expected_audio.extend_from_slice(bytes);
                copied_audio += 1;
            } else {
                assert_eq!(packet.duration(), 9000);
            }
            recorder
                .push(crate::recording::EncodedPacket {
                    track: *track,
                    data: packet.data().unwrap(),
                    pts: pts - offset,
                    dts: dts - offset,
                    duration: packet.duration(),
                    keyframe: packet.is_key(),
                })
                .unwrap();
        }
        assert!(copied_audio >= 30);
        recorder.finish().unwrap();
        let expected_path = directory.path().join("audio.aac");
        std::fs::write(&expected_path, expected_audio).unwrap();
        // Then: Payloads, independent track timebases and the shared epoch offset survive MP4.
        let mut copied = format::input(&path).unwrap();
        let copied_packets = copied
            .packets()
            .map(|(stream, packet)| {
                let track = if stream.parameters().medium() == media::Type::Video {
                    Track::Video
                } else {
                    Track::Audio
                };
                (track, packet)
            })
            .collect::<Vec<_>>();
        assert_eq!(copied_packets.len(), packets.len());
        for track in [Track::Video, Track::Audio] {
            let originals = packets
                .iter()
                .filter(|(kind, _)| *kind == track)
                .map(|(_, packet)| packet)
                .collect::<Vec<_>>();
            let copied = copied_packets
                .iter()
                .filter(|(kind, _)| *kind == track)
                .map(|(_, packet)| packet)
                .collect::<Vec<_>>();
            assert_eq!(originals.len(), copied.len());
            for (original, packet) in originals.iter().zip(copied) {
                let offset = if track == Track::Video {
                    epoch
                } else {
                    audio_epoch
                };
                assert_eq!(packet.pts(), Some(original.pts().unwrap() - offset));
                assert_eq!(packet.dts(), Some(original.dts().unwrap() - offset));
                assert_eq!(packet.duration(), original.duration());
                if track == Track::Audio {
                    assert!(packet.data() == original.data(), "AAC access unit changed");
                }
            }
        }
        let actual_video = decoded(&path, false);
        assert_eq!(actual_video.len(), 15 * 160 * 120 * 3);
        let expected_video = decoded(
            &Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/baseline-160x120.h264"),
            false,
        );
        let expected_video = expected_video
            .iter()
            .copied()
            .cycle()
            .take(actual_video.len())
            .collect::<Vec<_>>();
        assert!(
            actual_video == expected_video,
            "decoded video frames changed"
        );
        let actual = decoded(&path, true);
        let expected = decoded(&expected_path, true);
        assert_eq!(actual.len(), expected.len());
        assert!(actual == expected, "decoded AAC samples changed");
        assert_eq!(camera.count("PLAY"), 1);
        assert_eq!(camera.count("SETUP"), 2);
        assert_session_closed(&camera);
    }

    #[test]
    fn cancellation_interrupts_stalled_rtsp_negotiation() {
        // Given: A camera accepts the connection and stalls its DESCRIBE response.
        let camera = RtspCamera::start("STALL");
        let input = CameraInput::start(
            camera.url.clone(),
            false,
            Duration::from_secs(10),
            Duration::from_secs(10),
            |_| Ok(()),
        )
        .unwrap();
        let deadline = Instant::now() + Duration::from_secs(2);
        while camera.count("DESCRIBE") == 0 {
            assert!(Instant::now() < deadline);
            thread::sleep(Duration::from_millis(5));
        }
        // When: The last consumer closes during connection setup.
        let start = Instant::now();
        drop(input);
        // Then: Parent shutdown is bounded rather than waiting for the server timeout.
        assert!(start.elapsed() < Duration::from_millis(700));
    }

    fn rejects_response(response: Vec<u8>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("rtsp://{}/camera", listener.local_addr().unwrap());
        let server = thread::spawn(move || {
            let (mut socket, _) = listener.accept().unwrap();
            socket
                .set_read_timeout(Some(Duration::from_secs(3)))
                .unwrap();
            socket
                .set_write_timeout(Some(Duration::from_secs(3)))
                .unwrap();
            let mut request = [0; 4096];
            assert!(socket.read(&mut request).unwrap() > 0);
            let _ = socket.write_all(&response);
            let _ = socket.read(&mut request);
        });
        let input = CameraInput::start(
            url,
            false,
            Duration::from_secs(3),
            Duration::from_secs(3),
            |_| Ok(()),
        )
        .unwrap();
        let start = Instant::now();
        assert_eq!(wait_for_failure(&input), "rtsp_connection_failed");
        assert!(start.elapsed() < Duration::from_secs(2));
        drop(input);
        server.join().unwrap();
    }

    #[test]
    fn oversized_announced_rtsp_body_is_refused_before_receiving_it() {
        // Given: A server announces an oversized body but supplies no body bytes.
        let response = b"RTSP/1.0 200 OK\r\nCSeq: 1\r\nContent-Length: 268435456\r\n\r\n".to_vec();
        // When: The pinned library parses the header before allocating the response body.
        rejects_response(response);
        // Then: The stable failure arrives before the deadline without waiting for body data.
    }

    #[test]
    fn unterminated_rtsp_header_is_refused_at_the_aggregate_byte_budget() {
        // Given: A response contains an unterminated header beyond the fixed control budget.
        let mut response = b"RTSP/1.0 200 OK\r\nX-Long: ".to_vec();
        response.resize(300 * 1024, b'x');
        // When: The pinned parser reads it incrementally with a three-second I/O deadline.
        rejects_response(response);
        // Then: Byte accounting refuses promptly even though a header terminator never arrives.
    }
}
