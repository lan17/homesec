//! Native RTSP/TCP ingestion with bounded media buffers for H.264 copy preview.
//!
//! Retina owns RTSP authentication and session maintenance. The existing bounded
//! H.264 assembler owns depacketization so malformed fragments cannot grow an
//! incomplete access unit without limit.
//! Its current-thread runtime lives off the WebRTC loop. No URLs, credentials,
//! library errors or frame bytes cross the diagnostic boundary.
//!
//! Media access units and delivery queues have byte/count limits. Retina's RTSP
//! control-response parser currently exposes no byte cap through SessionOptions:
//! those reads have deadlines, but an oversized response can allocate as its
//! bytes arrive. Resolve that upstream before sharing this process with recording.

use crate::rtp::{H264Assembler, Packet};
use futures_util::StreamExt;
use h264_reader::nal::{Nal, RefNal};
use h264_reader::rbsp::BitRead;
use mio::Waker;
use percent_encoding::percent_decode_str;
use retina::client::{
    Credentials, PacketItem, PlayOptions, Session, SessionGroup, SessionOptions, SetupOptions,
    Transport,
};
use retina::codec::{ParametersRef, VideoParameters, VideoParametersCodec};
use std::sync::mpsc::{self, Receiver, SyncSender, TryRecvError, TrySendError};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;
use tokio::sync::oneshot;
use url::Url;

const QUEUE_CAPACITY: usize = 8;
const MAX_FRAME_BYTES: usize = 2 * 1024 * 1024;
const TEARDOWN_TIMEOUT: Duration = Duration::from_millis(200);
const DROP_TIMEOUT: Duration = Duration::from_millis(300);
type Result<T> = std::result::Result<T, &'static str>;

pub struct EncodedFrame {
    pub timestamp: u32,
    pub keyframe: bool,
    /// Annex B access unit, including current SPS/PPS before each keyframe.
    pub data: Arc<[u8]>,
}

pub enum Event {
    Info { profile_level_id: u32 },
    Frame(EncodedFrame),
}

/// An overflowing consumer fails closed instead of receiving broken interframes.
/// Terminal failures have a separate slot so a full media queue cannot hide them.
pub struct RtspSource {
    events: Receiver<Event>,
    failure: Arc<Mutex<Option<&'static str>>>,
    cancel: Option<oneshot::Sender<()>>,
    finished: Receiver<()>,
    worker: Option<JoinHandle<()>>,
}

impl RtspSource {
    pub fn start(
        url: String,
        connect_timeout: Duration,
        io_timeout: Duration,
        waker: Arc<Waker>,
    ) -> Result<Self> {
        let (url, credentials) = camera_url(&url)?;
        if [connect_timeout, io_timeout]
            .iter()
            .any(|timeout| timeout.is_zero() || *timeout > Duration::from_secs(120))
        {
            return Err("invalid_rtsp_timeout");
        }
        let (sender, events) = mpsc::sync_channel(QUEUE_CAPACITY);
        let (cancel, mut cancellation) = oneshot::channel();
        let (finished_tx, finished) = mpsc::sync_channel(1);
        let failure = Arc::new(Mutex::new(None));
        let worker_failure = Arc::clone(&failure);
        let worker = thread::Builder::new()
            .name("rtsp-ingest".into())
            .spawn(move || {
                let runtime = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build();
                let outcome = match runtime {
                    Ok(runtime) => {
                        let group = Arc::new(SessionGroup::default());
                        let outcome = runtime.block_on(async {
                            let outcome = tokio::select! {
                                biased;
                                _ = &mut cancellation => Ok(()),
                                result = ingest(url, credentials, connect_timeout, io_timeout, Arc::clone(&group), &sender, &waker) => result,
                            };
                            // Dropping the ingest future drops its session and initiates
                            // TEARDOWN. Allow a bounded attempt before closing the runtime.
                            let _ = tokio::time::timeout(TEARDOWN_TIMEOUT, group.await_teardown()).await;
                            outcome
                        });
                        // In particular, do not wait indefinitely for a resolver thread.
                        runtime.shutdown_timeout(Duration::from_millis(10));
                        outcome
                    }
                    Err(_) => Err("rtsp_source_failed"),
                };
                if let Err(code) = outcome
                    && let Ok(mut failure) = worker_failure.lock()
                {
                    *failure = Some(code);
                }
                drop(sender);
                let _ = waker.wake();
                let _ = finished_tx.send(());
            })
            .map_err(|_| "rtsp_source_failed")?;
        Ok(Self {
            events,
            failure,
            cancel: Some(cancel),
            finished,
            worker: Some(worker),
        })
    }

    pub fn try_recv(&self) -> Result<Option<Event>> {
        if let Some(code) = *self.failure.lock().map_err(|_| "rtsp_source_failed")? {
            return Err(code);
        }
        match self.events.try_recv() {
            Ok(event) => Ok(Some(event)),
            Err(TryRecvError::Empty) => Ok(None),
            Err(TryRecvError::Disconnected) => Err("rtsp_source_failed"),
        }
    }
}

impl Drop for RtspSource {
    fn drop(&mut self) {
        if let Some(cancel) = self.cancel.take() {
            let _ = cancel.send(());
        }
        // Network teardown must never indefinitely block media/control shutdown.
        if self.finished.recv_timeout(DROP_TIMEOUT).is_ok()
            && let Some(worker) = self.worker.take()
        {
            let _ = worker.join();
        }
    }
}

fn camera_url(raw: &str) -> Result<(Url, Option<Credentials>)> {
    if raw.len() > 16 * 1024 {
        return Err("invalid_rtsp_url");
    }
    let mut url = Url::parse(raw).map_err(|_| "invalid_rtsp_url")?;
    if url.scheme() != "rtsp" || url.host_str().is_none() || url.fragment().is_some() {
        return Err("invalid_rtsp_url");
    }
    let credentials = if !url.username().is_empty() || url.password().is_some() {
        let username = percent_decode_str(url.username())
            .decode_utf8()
            .map_err(|_| "invalid_rtsp_url")?
            .into_owned();
        let password = percent_decode_str(url.password().unwrap_or_default())
            .decode_utf8()
            .map_err(|_| "invalid_rtsp_url")?
            .into_owned();
        url.set_username("").map_err(|_| "invalid_rtsp_url")?;
        url.set_password(None).map_err(|_| "invalid_rtsp_url")?;
        Some(Credentials { username, password })
    } else {
        None
    };
    Ok((url, credentials))
}

fn publish(sender: &SyncSender<Event>, waker: &Waker, event: Event) -> Result<()> {
    sender.try_send(event).map_err(|error| match error {
        TrySendError::Full(_) => "media_queue_overflow",
        TrySendError::Disconnected(_) => "rtsp_source_failed",
    })?;
    waker.wake().map_err(|_| "rtsp_source_failed")
}

async fn ingest(
    url: Url,
    credentials: Option<Credentials>,
    connect_timeout: Duration,
    io_timeout: Duration,
    group: Arc<SessionGroup>,
    sender: &SyncSender<Event>,
    waker: &Waker,
) -> Result<()> {
    let (mut stream, stream_id) = tokio::time::timeout(connect_timeout, async {
        let options = SessionOptions::default()
            .creds(credentials)
            .session_group(group);
        let mut session = Session::describe(url, options)
            .await
            .map_err(|_| "rtsp_connection_failed")?;
        // Preserve first-video semantics of the previous FFmpeg -map 0:v:0 path.
        let stream_id = session
            .streams()
            .iter()
            .position(|stream| stream.media() == "video")
            .ok_or("unsupported_codec")?;
        let selected = &session.streams()[stream_id];
        if selected.encoding_name() != "h264" || selected.clock_rate_hz() != 90_000 {
            return Err("unsupported_codec");
        }
        if let Some(ParametersRef::Video(parameters)) = selected.parameters() {
            profile_level_id(parameters)?;
        }
        session
            .setup(
                stream_id,
                SetupOptions::default().transport(Transport::Tcp(Default::default())),
            )
            .await
            .map_err(|_| "rtsp_connection_failed")?;
        let stream = session
            .play(PlayOptions::default())
            .await
            .map_err(|_| "rtsp_connection_failed")?;
        Ok((stream, stream_id))
    })
    .await
    .map_err(|_| "rtsp_timeout")??;

    let mut assembler = H264Assembler::default();
    let mut profile = None;
    let mut parameter_sets: Option<(Vec<u8>, Vec<u8>)> = None;
    if let Some(ParametersRef::Video(parameters)) = stream.streams()[stream_id].parameters() {
        let VideoParametersCodec::H264 { sps, pps } = parameters.codec_params() else {
            return Err("unsupported_codec");
        };
        if !assembler.seed_parameter_sets(sps, pps) {
            return Err("unsupported_codec");
        }
        parameter_sets = Some((sps.to_vec(), pps.to_vec()));
        let profile_level_id = profile_level_id(parameters)?;
        publish(sender, waker, Event::Info { profile_level_id })?;
        profile = Some(profile_level_id);
    }
    let mut recovering = true;
    let mut initial_timestamp = None;
    let mut previous_timestamp = None;
    let mut packet_batch = 0_u8;
    // A stream of RTCP or incomplete fragments must not reset the media deadline.
    let mut frame_deadline = tokio::time::Instant::now() + io_timeout;
    loop {
        if tokio::time::Instant::now() >= frame_deadline {
            return Err("rtsp_timeout");
        }
        let item = tokio::time::timeout_at(frame_deadline, stream.next())
            .await
            .map_err(|_| "rtsp_timeout")?
            .ok_or("rtsp_stream_ended")?
            .map_err(|_| "rtsp_stream_failed")?;
        packet_batch = packet_batch.wrapping_add(1);
        if packet_batch == 0 {
            // An always-readable camera cannot starve cancellation on the source runtime.
            tokio::task::yield_now().await;
        }
        let PacketItem::Rtp(packet) = item else {
            continue;
        };
        if packet.stream_id() != stream_id {
            continue;
        }
        let timestamp = packet.timestamp().timestamp();
        let first_timestamp = *initial_timestamp.get_or_insert(timestamp);
        // Retina validates the negotiated payload type. Normalize only this
        // private handoff's payload type for the shared bounded assembler.
        let Some(frame) = assembler.push(Packet {
            timestamp: packet.header().timestamp(),
            sequence: packet.sequence_number(),
            marker: packet.mark(),
            payload_type: 96,
            payload: packet.payload(),
        }) else {
            continue;
        };
        // PLAY can begin midway through a picture. RTP-Info may be absent or
        // inaccurate, so only a later timestamp establishes a known boundary.
        // Still assemble the initial packets to retain in-band SPS/PPS and
        // sequence continuity; recovery waits for the next complete IDR.
        if timestamp == first_timestamp {
            continue;
        }
        let Some((sps, pps)) = assembler.parameter_sets() else {
            continue;
        };
        if parameter_sets
            .as_ref()
            .is_none_or(|(old_sps, old_pps)| old_sps != sps || old_pps != pps)
        {
            let parameters = retina::codec::h264::parameters_from_sps_and_pps(
                sps,
                pps,
                retina::codec::h26x::Framing::AnnexB,
            )
            .map_err(|_| "unsupported_codec")?;
            let current_profile = profile_level_id(&parameters)?;
            match profile {
                Some(previous) if previous != current_profile => {
                    return Err("codec_parameters_changed");
                }
                None => {
                    publish(
                        sender,
                        waker,
                        Event::Info {
                            profile_level_id: current_profile,
                        },
                    )?;
                    profile = Some(current_profile);
                }
                _ => {}
            }
            parameter_sets = Some((sps.to_vec(), pps.to_vec()));
            recovering = true;
        }
        if !validate_access_unit(&frame.data)? {
            // Some cameras mark an SPS/PPS-only packet separately from the IDR
            // at the same timestamp. It updates parameters, not the frame clock.
            continue;
        }
        if previous_timestamp.is_some_and(|previous| timestamp <= previous) {
            return Err("unsupported_frame_reordering");
        }
        previous_timestamp = Some(timestamp);
        let keyframe = frame.keyframe;
        if recovering && !keyframe {
            continue;
        }
        recovering = false;
        publish(
            sender,
            waker,
            Event::Frame(EncodedFrame {
                timestamp: timestamp as u32,
                keyframe,
                data: frame.data.into(),
            }),
        )?;
        frame_deadline = tokio::time::Instant::now() + io_timeout;
    }
}

fn profile_level_id(parameters: &VideoParameters) -> Result<u32> {
    let profile = parameters
        .rfc6381_codec()
        .strip_prefix("avc1.")
        .filter(|hex| hex.len() == 6)
        .and_then(|hex| u32::from_str_radix(hex, 16).ok())
        .ok_or("unsupported_codec")?;
    validate_level(
        profile,
        parameters.coded_pixel_dimensions(),
        parameters.frame_rate(),
    )?;
    Ok(profile)
}

fn validate_level(
    profile: u32,
    dimensions: (u32, u32),
    frame_rate: Option<(u32, u32)>,
) -> Result<()> {
    if !matches!(profile >> 16, 66 | 77 | 100) {
        return Err("unsupported_codec");
    }
    // H.264 Table A-1 MaxFS/MaxMBPS, matching FFmpeg's h264_levels.c.
    // Use the declared level, including level 1b's constraint_set3 distinction.
    let (max_frame, max_rate): (u64, u64) = match profile & 0xff {
        10 => (99, 1485),
        11 if profile >> 16 != 100 && profile & 0x1000 != 0 => (99, 1485),
        11 => (396, 3000),
        12 => (396, 6000),
        13 | 20 => (396, 11880),
        21 => (792, 19800),
        22 => (1620, 20250),
        30 => (1620, 40500),
        31 => (3600, 108000),
        _ => return Err("unsupported_codec"),
    };
    let (width, height) = dimensions;
    let width_mb = u64::from(width.div_ceil(16));
    let height_mb = u64::from(height.div_ceil(16));
    let macroblocks = width_mb * height_mb;
    if macroblocks == 0
        || macroblocks > max_frame
        || width_mb * width_mb > 8 * max_frame
        || height_mb * height_mb > 8 * max_frame
    {
        return Err("unsupported_codec");
    }
    if let Some((duration, scale)) = frame_rate
        && (duration == 0 || macroblocks * u64::from(scale) > max_rate * u64::from(duration))
    {
        return Err("unsupported_codec");
    }
    Ok(())
}

fn validate_access_unit(data: &[u8]) -> Result<bool> {
    if data.len() > MAX_FRAME_BYTES || !data.starts_with(&[0, 0, 0, 1]) {
        return Err("invalid_video_frame");
    }
    let mut remaining = &data[4..];
    let mut has_slice = false;
    while !remaining.is_empty() {
        let end = remaining
            .windows(4)
            .position(|window| window == [0, 0, 0, 1])
            .unwrap_or(remaining.len());
        let nal = &remaining[..end];
        if nal.is_empty() {
            return Err("invalid_video_frame");
        }
        if matches!(nal[0] & 31, 1 | 5) {
            has_slice = true;
            let unit = RefNal::new(nal, &[], true);
            let mut bits = unit.rbsp_bits();
            bits.read_ue("first_mb_in_slice")
                .map_err(|_| "invalid_video_frame")?;
            let slice_type = bits
                .read_ue("slice_type")
                .map_err(|_| "invalid_video_frame")?;
            if slice_type > 9 {
                return Err("invalid_video_frame");
            }
            if slice_type % 5 == 1 {
                return Err("unsupported_frame_reordering");
            }
        }
        if end == remaining.len() {
            break;
        }
        remaining = &remaining[end + 4..];
    }
    Ok(has_slice)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn joining_mid_idr_waits_for_a_complete_keyframe() {
        // Given: SDP initializes decoding, but RTP starts at the second IDR slice.
        // seq=1 in RTP-Info is ignored by Retina for compatibility with real cameras.
        let camera = crate::rtsp_camera::RtspCamera::start("PARTIAL_IDR");
        let poll = mio::Poll::new().unwrap();
        let waker = Arc::new(Waker::new(poll.registry(), mio::Token(0)).unwrap());
        let source = RtspSource::start(
            camera.url.clone(),
            Duration::from_secs(5),
            Duration::from_secs(10),
            waker,
        )
        .unwrap();

        // When: The partial IDR, dependent pictures, and a later intact IDR arrive.
        let deadline = std::time::Instant::now() + Duration::from_secs(3);
        let frame = loop {
            if let Some(Event::Frame(frame)) = source.try_recv().unwrap() {
                break frame;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "no intact IDR received"
            );
            std::thread::sleep(Duration::from_millis(5));
        };

        // Then: Delivery begins at the next GOP, preserving both real IDR slices.
        assert_eq!(frame.timestamp, 27000);
        assert!(frame.keyframe);
        let expected = crate::rtsp_camera::annex_b_nals(include_bytes!(
            "../tests/fixtures/baseline-multislice-160x120.h264"
        ));
        let actual = crate::rtsp_camera::annex_b_nals(&frame.data);
        let expected_slices: Vec<_> = expected.iter().filter(|nal| nal[0] & 31 == 5).collect();
        let actual_slices: Vec<_> = actual.iter().filter(|nal| nal[0] & 31 == 5).collect();
        assert_eq!(expected_slices.len(), 2);
        assert_eq!(actual_slices, expected_slices);
    }

    #[test]
    fn undrained_media_queue_fails_closed_and_reports_overflow_before_queued_frames() {
        // Given: A real RTSP camera and a source whose media receiver is never drained.
        let camera = crate::rtsp_camera::RtspCamera::start("H264");
        let poll = mio::Poll::new().unwrap();
        let waker = Arc::new(Waker::new(poll.registry(), mio::Token(0)).unwrap());
        let source = RtspSource::start(
            camera.url.clone(),
            Duration::from_secs(5),
            Duration::from_secs(10),
            waker,
        )
        .unwrap();

        // When: The camera keeps streaming until the bounded queue fills and teardown completes.
        let deadline = std::time::Instant::now() + Duration::from_secs(3);
        while camera.count("DISCONNECTED") == 0 {
            assert!(
                std::time::Instant::now() < deadline,
                "media queue overflow did not close RTSP"
            );
            std::thread::sleep(Duration::from_millis(5));
        }

        // Then: Terminal overflow becomes visible after teardown and remains the stable outcome.
        // The server can observe TCP close just before the worker publishes its failure.
        let deadline = std::time::Instant::now() + Duration::from_secs(1);
        let failure = loop {
            if let Err(code) = source.try_recv() {
                break code;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "overflow failure was not published"
            );
            std::thread::sleep(Duration::from_millis(5));
        };
        assert_eq!(failure, "media_queue_overflow");
        assert!(matches!(source.try_recv(), Err("media_queue_overflow")));
        assert_eq!(camera.count("PLAY"), 1);
        assert_eq!(camera.count("TEARDOWN"), 1);
    }

    #[test]
    fn camera_credentials_are_decoded_and_removed_from_request_url() {
        // Given: URL userinfo contains escaped reserved characters.
        let raw = "rtsp://camera%40user:p%3Aass%2Bword@127.0.0.1/live";
        // When: preparing the separate protocol URL and authentication values.
        let (url, credentials) = camera_url(raw).unwrap();
        let credentials = credentials.unwrap();
        // Then: requests use a credential-free URL and original decoded credentials.
        assert_eq!(url.as_str(), "rtsp://127.0.0.1/live");
        assert_eq!(credentials.username, "camera@user");
        assert_eq!(credentials.password, "p:ass+word");
    }

    #[test]
    fn rejects_unsupported_url_without_exposing_input() {
        // Given: a URL with an unsupported transport and credentials.
        let raw = "https://private:secret@camera/live";
        // When: preparing an RTSP source.
        let result = camera_url(raw);
        // Then: only the stable public failure code is returned.
        assert!(matches!(result, Err("invalid_rtsp_url")));
    }

    #[test]
    fn b_slices_are_rejected_even_with_monotonic_timestamps() {
        // Given: minimal NAL headers followed by first_mb=0 and slice_type codes.
        let p_slice = [0, 0, 0, 1, 0x41, 0b1100_0000];
        let b_slice = [0, 0, 0, 1, 0x41, 0b1010_0000];
        let b_slice_all = [0, 0, 0, 1, 0x41, 0b1001_1100];
        // When: inspecting complete access units before WebRTC delivery.
        let p_result = validate_access_unit(&p_slice);
        let b_results = [
            validate_access_unit(&b_slice),
            validate_access_unit(&b_slice_all),
        ];
        // Then: P slices pass and both B slice encodings fail closed.
        assert_eq!(p_result, Ok(true));
        assert_eq!(b_results, [Err("unsupported_frame_reordering"); 2]);
    }

    #[test]
    fn truncated_or_oversized_access_units_are_rejected() {
        // Given: a slice with no header bits and an oversized access unit.
        let truncated = [0, 0, 0, 1, 0x41];
        let mut oversized = vec![0; MAX_FRAME_BYTES + 1];
        oversized[..5].copy_from_slice(&[0, 0, 0, 1, 0x41]);
        // When: validating the source output.
        let results = [
            validate_access_unit(&truncated),
            validate_access_unit(&oversized),
        ];
        // Then: invalid media is never forwarded.
        assert_eq!(results, [Err("invalid_video_frame"); 2]);
    }

    #[test]
    fn level_limits_match_the_source_level_including_level_1b() {
        // Given: valid 720p30 media, a dishonest level-1 SPS and a level-1b stream.
        let hd = ((1280, 720), Some((1, 30)));
        // When: checking the declared level before SDP advertisement.
        let valid_hd = validate_level(0x42e01f, hd.0, hd.1);
        let false_level = validate_level(0x42e00a, hd.0, hd.1);
        let level_1b = validate_level(0x42f00b, (352, 288), Some((1, 5)));
        let level_11 = validate_level(0x42e00b, (352, 288), Some((1, 5)));
        // Then: a lower level never inherits level 3.1's larger budget.
        assert!(valid_hd.is_ok());
        assert_eq!(false_level, Err("unsupported_codec"));
        assert_eq!(level_1b, Err("unsupported_codec"));
        assert!(level_11.is_ok());
        assert_eq!(
            validate_level(0x42e01f, hd.0, Some((1, 60))),
            Err("unsupported_codec")
        );
    }

    #[test]
    fn parameter_only_units_do_not_advance_the_video_clock() {
        // Given: a camera marks decoder initialization separately from its IDR.
        let parameters = [0, 0, 0, 1, 0x67, 0x42, 0, 31, 0, 0, 0, 1, 0x68, 1];
        // When: checking whether this access unit contains a picture.
        let result = validate_access_unit(&parameters);
        // Then: it updates initialization but does not count as a video frame.
        assert_eq!(result, Ok(false));
    }

    #[test]
    fn dropping_a_pending_describe_closes_the_camera_connection() {
        use std::io::Read;
        use std::net::TcpListener;
        use std::time::Instant;

        // Given: a reachable camera accepts DESCRIBE but never answers.
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let (accepted_tx, accepted_rx) = mpsc::sync_channel(1);
        let (closed_tx, closed_rx) = mpsc::sync_channel(1);
        let camera = thread::spawn(move || {
            let (mut connection, _) = listener.accept().unwrap();
            connection
                .set_read_timeout(Some(Duration::from_secs(2)))
                .unwrap();
            let mut request = [0; 4096];
            assert!(connection.read(&mut request).unwrap() > 0);
            accepted_tx.send(()).unwrap();
            let eof = connection.read(&mut request).unwrap();
            closed_tx.send(eof).unwrap();
        });
        let poll = mio::Poll::new().unwrap();
        let waker = Arc::new(Waker::new(poll.registry(), mio::Token(1)).unwrap());
        let source = RtspSource::start(
            format!("rtsp://{address}/camera"),
            Duration::from_secs(5),
            Duration::from_secs(10),
            waker,
        )
        .unwrap();
        accepted_rx.recv_timeout(Duration::from_secs(2)).unwrap();
        // When: the owner cancels during the pending RTSP handshake.
        let before = Instant::now();
        drop(source);
        // Then: cancellation is bounded and closes the real socket promptly.
        assert!(before.elapsed() < Duration::from_secs(1));
        assert_eq!(closed_rx.recv_timeout(Duration::from_secs(1)).unwrap(), 0);
        camera.join().unwrap();
    }
}
