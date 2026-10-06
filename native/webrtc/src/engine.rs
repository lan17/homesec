use crate::Options;
use crate::protocol::{
    Control, MAX_CONTROL_BYTES, MAX_SDP_BYTES, Operation, Reply, Request, controls, output,
};
use crate::rtp::{Event as SourceEvent, TimestampClock};
use crate::rtp_receiver::RtpReceiver;
use crate::rtsp::RtspSource;
use mio::net::UdpSocket;
use mio::{Events, Interest, Poll, Token, Waker};
use std::collections::HashMap;
use std::io;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use str0m::change::{SdpAnswer, SdpOffer};
use str0m::format::{Codec, CodecConfig};
use str0m::media::{Frequency, MediaKind, MediaTime, Mid, Pt};
use str0m::net::{Protocol, Receive};
use str0m::{Candidate, Event, IceConnectionState, Input, Output, Rtc, RtcConfig};

const MEDIA: Token = Token(0);
const CONTROL: Token = Token(1);
const MEDIA_TIMEOUT: Duration = Duration::from_secs(10);
const MAX_BATCH: usize = 4096;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

struct Peer {
    rtc: Rtc,
    video: Option<(Mid, Pt)>,
    audio: Option<(Mid, Pt)>,
    connected: bool,
    wait_keyframe: bool,
    failed: bool,
    timeout: Instant,
    negotiation_deadline: Instant,
    lease_deadline: Instant,
    hard_deadline: Instant,
}

impl Peer {
    fn pump(&mut self, socket: &UdpSocket, now: Instant) -> Result<()> {
        if self.expired(now) {
            self.failed = true;
            return Ok(());
        }
        for _ in 0..MAX_BATCH {
            match self.rtc.poll_output()? {
                Output::Timeout(deadline) => {
                    self.timeout = deadline;
                    return Ok(());
                }
                Output::Transmit(packet) => {
                    if let Err(error) = socket.send_to(&packet.contents, packet.destination) {
                        if error.kind() != io::ErrorKind::WouldBlock {
                            return Err(error.into());
                        }
                        // A full kernel socket cannot grow a user-space queue.
                        self.wait_keyframe = true;
                    }
                }
                Output::Event(Event::Connected) => {
                    self.connected = true;
                    self.wait_keyframe = true;
                }
                Output::Event(Event::IceConnectionStateChange(
                    IceConnectionState::Disconnected,
                )) => {
                    self.failed = true;
                }
                Output::Event(Event::MediaAdded(media)) if media.direction.is_sending() => {
                    if let Some(writer) = self.rtc.writer(media.mid) {
                        let codec = match media.kind {
                            MediaKind::Video => Codec::H264,
                            MediaKind::Audio => Codec::Opus,
                        };
                        if let Some(params) = writer.payload_params().find(|p| {
                            p.spec().codec == codec
                                && (codec != Codec::H264
                                    || p.spec().format.packetization_mode == Some(1))
                        }) {
                            match media.kind {
                                MediaKind::Video => self.video = Some((media.mid, params.pt())),
                                MediaKind::Audio => self.audio = Some((media.mid, params.pt())),
                            }
                        }
                    }
                }
                Output::Event(Event::KeyframeRequest(_)) => self.wait_keyframe = true,
                Output::Event(Event::Closed) => self.failed = true,
                _ => {}
            }
        }
        // A pathological offer or slow peer cannot monopolize the camera worker.
        self.failed = true;
        self.timeout = now;
        Ok(())
    }

    fn expired(&self, now: Instant) -> bool {
        self.failed
            || now >= self.lease_deadline
            || now >= self.hard_deadline
            || (!self.connected && now >= self.negotiation_deadline)
    }

    fn write(&mut self, video: bool, keyframe: bool, time: u64, data: Arc<[u8]>, now: Instant) {
        if !self.connected || self.expired(now) || (video && self.wait_keyframe && !keyframe) {
            return;
        }
        let Some((mid, pt)) = (if video { self.video } else { self.audio }) else {
            return;
        };
        let frequency = if video {
            Frequency::NINETY_KHZ
        } else {
            Frequency::FORTY_EIGHT_KHZ
        };
        if let Some(writer) = self.rtc.writer(mid) {
            if writer
                .write(pt, now, MediaTime::new(time, frequency), data)
                .is_err()
            {
                self.failed = true;
            } else if video && keyframe {
                self.wait_keyframe = false;
            }
        }
    }
}

/// Drop always kills/reaps FFmpeg, including JSON/UDP failures and stdin EOF.
struct MediaChild(Child);
impl Drop for MediaChild {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

enum MediaInput {
    Ffmpeg {
        child: MediaChild,
        source: RtpReceiver,
    },
    Rtsp(RtspSource),
}

// Passthrough cannot lower the camera's encoder level to match a receiver.
// str0m matches profiles but permits level differences, so check the selected
// payload against the original offer before promising to send camera bytes.
fn h264_level(profile: u32) -> u32 {
    // Level 1b lies between 1.0 and 1.1 (RFC 6184, section 8.1).
    if profile & 0xff == 11 && profile & 0x1000 != 0 && matches!(profile >> 16, 66 | 77 | 88) {
        105
    } else {
        (profile & 0xff) * 10
    }
}

fn source_receiver_profile(offer: &SdpOffer, source: u32) -> Option<u32> {
    let mut codecs = CodecConfig::empty();
    codecs.add_h264(127.into(), None, true, source);
    // str0m prefers the closest level even when it is too low for passthrough.
    // Pick a compatible receive envelope first, then let its exact match win.
    offer
        .media_lines
        .iter()
        .filter(|line| line.direction().is_receiving())
        .flat_map(|line| line.rtp_params())
        .filter(|params| codecs.match_params(*params).is_some())
        .map(|params| params.spec().format.profile_level_id.unwrap_or(0x42e01f))
        .filter(|profile| h264_level(*profile) >= h264_level(source))
        .min_by_key(|profile| h264_level(*profile))
}

fn source_level_supported(offer: &SdpOffer, answer: &SdpAnswer, source: u32) -> bool {
    answer
        .media_lines
        .iter()
        .filter(|line| line.direction().is_sending())
        .all(|line| {
            line.rtp_params()
                .iter()
                .filter(|param| param.spec().codec == Codec::H264)
                .all(|param| {
                    offer
                        .media_lines
                        .iter()
                        .find(|offered| offered.mid() == line.mid())
                        .is_some_and(|offered| {
                            offered
                                .rtp_params()
                                .iter()
                                .find(|candidate| candidate.pt() == param.pt())
                                .is_some_and(|candidate| {
                                    h264_level(source)
                                        <= h264_level(
                                            candidate
                                                .spec()
                                                .format
                                                .profile_level_id
                                                .unwrap_or(0x42e01f),
                                        )
                                })
                        })
                })
        })
}

fn lease(value: f64, expires_at: Option<f64>, now: Instant, limit: Duration) -> Option<Instant> {
    if !value.is_finite() || value <= 0.0 {
        return None;
    }
    let mut remaining = value.min(limit.as_secs_f64());
    if let Some(expires_at) = expires_at {
        let epoch = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .ok()?
            .as_secs_f64();
        if !expires_at.is_finite() || expires_at <= epoch {
            return None;
        }
        remaining = remaining.min(expires_at - epoch);
    }
    now.checked_add(Duration::from_secs_f64(remaining))
}

pub fn run(options: Options) -> Result<()> {
    if options.udp_port_start == 0
        || options.udp_port_end < options.udp_port_start
        || !(1..=32).contains(&options.max_viewers)
        || !options.negotiation_timeout_s.is_finite()
        || options.negotiation_timeout_s <= 0.0
        || options.negotiation_timeout_s > 120.0
        || !options.max_session_duration_s.is_finite()
        || options.max_session_duration_s <= 0.0
        || options.max_session_duration_s > 86400.0
    {
        return Err("invalid_options".into());
    }
    let negotiation_timeout = Duration::from_secs_f64(options.negotiation_timeout_s);
    let session_limit = Duration::from_secs_f64(options.max_session_duration_s);
    let advertised_ip = options.advertised_ip.ok_or("invalid_options")?;
    let bind_ip = match advertised_ip {
        IpAddr::V4(_) => IpAddr::V4(Ipv4Addr::UNSPECIFIED),
        IpAddr::V6(_) => IpAddr::V6(Ipv6Addr::UNSPECIFIED),
    };
    let mut media = (options.udp_port_start..=options.udp_port_end)
        .find_map(|port| UdpSocket::bind(SocketAddr::new(bind_ip, port)).ok())
        .ok_or("media_port_unavailable")?;
    let candidate_addr = SocketAddr::new(advertised_ip, media.local_addr()?.port());
    let video = std::net::UdpSocket::bind("127.0.0.1:0")?;
    let audio = std::net::UdpSocket::bind("127.0.0.1:0")?;
    video.set_nonblocking(true)?;
    audio.set_nonblocking(true)?;
    let mut poll = Poll::new()?;
    poll.registry()
        .register(&mut media, MEDIA, Interest::READABLE)?;
    let waker = Arc::new(Waker::new(poll.registry(), CONTROL)?);
    let controls = controls(Arc::clone(&waker), |bytes| {
        let request: Request = serde_json::from_slice(bytes).ok()?;
        (request.request_id.len() <= 128).then_some(request)
    });
    output(
        &serde_json::json!({"event":"ready", "video_port":video.local_addr()?.port(),
        "audio_port":audio.local_addr()?.port(), "media_port":candidate_addr.port()}),
    )?;
    let mut peers: HashMap<String, Peer> = HashMap::new();
    let mut input: Option<MediaInput> = None;
    let mut pending_start: Option<String> = None;
    let mut source_profile: Option<u32> = None;
    let mut started: Option<Instant> = None;
    let mut last_video: Option<Instant> = None;
    let mut media_failed = false;
    let mut video_clock = TimestampClock::default();
    let mut audio_clock = TimestampClock::default();
    let mut events = Events::with_capacity(16);
    // Mio readiness is edge-triggered. A bounded receive batch must keep its
    // readiness until recv_from reaches WouldBlock, even without a new edge.
    let mut readable = false;
    let mut bytes = [0u8; 65536];
    loop {
        let now = Instant::now();
        let exited = match &mut input {
            Some(MediaInput::Ffmpeg { child, .. }) => child.0.try_wait()?.is_some(),
            _ => false,
        };
        if input.is_some()
            && (exited
                || last_video
                    .or(started)
                    .is_some_and(|last| now.duration_since(last) >= MEDIA_TIMEOUT))
        {
            input = None;
            media_failed = true;
            peers.clear();
            if let Some(request_id) = pending_start.take() {
                output(&Reply::error(request_id, "source_timeout"))?;
            }
        }
        // Source reads run outside this loop. A bounded batch preserves control
        // and peer deadlines even if the camera produces frames continuously.
        for _ in 0..16 {
            let event = match &mut input {
                Some(MediaInput::Rtsp(source)) => source.try_recv(),
                Some(MediaInput::Ffmpeg { source, .. }) => source.try_recv(),
                None => break,
            };
            match event {
                Ok(Some(SourceEvent::Info { profile_level_id })) => {
                    source_profile = Some(profile_level_id);
                    if let Some(request_id) = pending_start.take() {
                        output(&Reply::success(request_id))?;
                    }
                }
                Ok(Some(SourceEvent::Frame(frame))) => {
                    last_video = Some(now);
                    if let Some(time) = video_clock.extend(frame.timestamp) {
                        for peer in peers.values_mut() {
                            peer.write(true, frame.keyframe, time, Arc::clone(&frame.data), now);
                            if peer.pump(&media, now).is_err() {
                                peer.failed = true;
                            }
                        }
                    }
                }
                Ok(Some(SourceEvent::Audio { timestamp, data })) => {
                    if let Some(time) = audio_clock.extend(timestamp) {
                        for peer in peers.values_mut() {
                            peer.write(false, false, time, Arc::clone(&data), now);
                            if peer.pump(&media, now).is_err() {
                                peer.failed = true;
                            }
                        }
                    }
                }
                Ok(None) => break,
                Err(code) => {
                    input = None;
                    media_failed = true;
                    peers.clear();
                    if let Some(request_id) = pending_start.take() {
                        output(&Reply::error(request_id, code))?;
                    }
                    break;
                }
            }
        }
        peers.retain(|_, peer| !peer.expired(now));
        for peer in peers.values_mut() {
            if now >= peer.timeout && peer.rtc.handle_input(Input::Timeout(now)).is_err() {
                peer.failed = true;
            }
            if peer.pump(&media, now).is_err() {
                peer.failed = true;
            }
        }
        for _ in 0..16 {
            let Ok(control) = controls.try_recv() else {
                break;
            };
            let now = Instant::now();
            let Control::Request(request) = control else {
                return Ok(());
            };
            let mut reply = Reply::success(request.request_id.clone());
            let mut stop = false;
            match request.operation {
                Operation::Start {
                    ffmpeg_args,
                    rtsp_url,
                } => {
                    if started.is_some() {
                        reply = Reply::error(request.request_id, "preview_temporarily_unavailable");
                    } else {
                        match (ffmpeg_args, rtsp_url) {
                            (Some(ffmpeg_args), None)
                                if !ffmpeg_args.is_empty()
                                    && ffmpeg_args.iter().map(String::len).sum::<usize>()
                                        <= MAX_CONTROL_BYTES / 2 =>
                            {
                                // Register reception before FFmpeg can emit its
                                // first large access unit into the loopback socket.
                                match RtpReceiver::start(&video, &audio, Arc::clone(&waker)) {
                                    Ok(source) => {
                                        match Command::new("ffmpeg")
                                            .args(ffmpeg_args)
                                            .stdin(Stdio::null())
                                            .stdout(Stdio::null())
                                            .stderr(Stdio::null())
                                            .spawn()
                                        {
                                            Ok(process) => {
                                                input = Some(MediaInput::Ffmpeg {
                                                    child: MediaChild(process),
                                                    source,
                                                });
                                                started = Some(now);
                                            }
                                            Err(_) => {
                                                reply = Reply::error(
                                                    request.request_id,
                                                    "preview_temporarily_unavailable",
                                                )
                                            }
                                        }
                                    }
                                    Err(code) => reply = Reply::error(request.request_id, code),
                                }
                            }
                            (None, Some(url)) if !url.is_empty() && url.len() <= 16_384 => {
                                match RtspSource::start(
                                    url,
                                    Duration::from_secs(5),
                                    MEDIA_TIMEOUT,
                                    Arc::clone(&waker),
                                ) {
                                    Ok(source) => {
                                        input = Some(MediaInput::Rtsp(source));
                                        started = Some(now);
                                        pending_start = Some(request.request_id);
                                        continue;
                                    }
                                    Err(code) => reply = Reply::error(request.request_id, code),
                                }
                            }
                            _ => {
                                reply = Reply::error(
                                    request.request_id,
                                    "preview_temporarily_unavailable",
                                )
                            }
                        }
                    }
                }
                Operation::Offer {
                    session_id,
                    sdp,
                    lease_seconds,
                    lease_expires_at,
                } => {
                    let deadline = lease(lease_seconds, lease_expires_at, now, session_limit);
                    if input.is_none() || media_failed || pending_start.is_some() {
                        reply = Reply::error(request.request_id, "preview_temporarily_unavailable");
                    } else if peers.len() >= options.max_viewers {
                        reply = Reply::error(request.request_id, "session_limit");
                    } else if peers.contains_key(&session_id)
                        || session_id.is_empty()
                        || session_id.len() > 128
                        || sdp.len() > MAX_SDP_BYTES
                        || deadline.is_none()
                    {
                        reply = Reply::error(request.request_id, "invalid_offer");
                    } else {
                        let offer = SdpOffer::from_sdp_string(&sdp).ok();
                        let mut config = RtcConfig::new()
                            .clear_codecs()
                            // Browsers also offer an audio m-line for video-only
                            // sources. Keep it negotiable; no samples are sent.
                            .enable_opus(true, false)
                            .set_ice_lite(true)
                            .set_send_buffer_video(2048)
                            .set_send_buffer_audio(64);
                        // str0m packetizes H.264 with STAP-A/FU-A, which requires mode 1.
                        let profiles = if let Some(source) = source_profile {
                            offer
                                .as_ref()
                                .and_then(|offer| source_receiver_profile(offer, source))
                                .map(|profile| vec![(127_u8, 121_u8, profile)])
                                .unwrap_or_default()
                        } else {
                            vec![
                                (127_u8, 121_u8, 0x42001f),
                                (108, 109, 0x42e01f),
                                (123, 119, 0x4d001f),
                                (114, 115, 0x64001f),
                            ]
                        };
                        for (pt, resend, profile) in profiles {
                            config.codec_config().add_h264(
                                pt.into(),
                                Some(resend.into()),
                                true,
                                profile,
                            );
                        }
                        let mut rtc = config.build(now);
                        let candidate = Candidate::host(candidate_addr, "udp")?;
                        rtc.add_local_candidate(candidate)
                            .ok_or("invalid_candidate")?;
                        match offer.and_then(|offer| {
                            let answer = rtc
                                .sdp_api()
                                .accept_offer(SdpOffer::from((*offer).clone()))
                                .ok()?;
                            source_profile
                                .is_none_or(|profile| {
                                    source_level_supported(&offer, &answer, profile)
                                })
                                .then_some(answer)
                        }) {
                            Some(answer) => {
                                let mut peer = Peer {
                                    rtc,
                                    video: None,
                                    audio: None,
                                    connected: false,
                                    wait_keyframe: true,
                                    failed: false,
                                    timeout: now,
                                    negotiation_deadline: now + negotiation_timeout,
                                    lease_deadline: deadline.unwrap_or(now),
                                    hard_deadline: now + session_limit,
                                };
                                // MediaAdded is delayed until DTLS completes. Validate the
                                // negotiated answer now; bind writers from that event later.
                                let video_count = answer
                                    .media_lines
                                    .iter()
                                    .filter(|line| {
                                        line.direction().is_sending()
                                            && line.rtp_params().iter().any(|p| {
                                                p.spec().codec == Codec::H264
                                                    && p.spec().format.packetization_mode == Some(1)
                                            })
                                    })
                                    .count();
                                let audio_count = answer
                                    .media_lines
                                    .iter()
                                    .filter(|line| {
                                        line.direction().is_sending()
                                            && line
                                                .rtp_params()
                                                .iter()
                                                .any(|p| p.spec().codec == Codec::Opus)
                                    })
                                    .count();
                                if video_count == 1
                                    && audio_count <= 1
                                    && peer.pump(&media, now).is_ok()
                                    && !peer.failed
                                {
                                    reply.sdp = Some(answer.to_sdp_string());
                                    peers.insert(session_id, peer);
                                } else {
                                    reply = Reply::error(request.request_id, "invalid_offer");
                                }
                            }
                            None => reply = Reply::error(request.request_id, "invalid_offer"),
                        }
                    }
                }
                Operation::Renew {
                    session_id,
                    lease_seconds,
                    lease_expires_at,
                } => {
                    if let Some(peer) = peers.get_mut(&session_id).filter(|p| !p.expired(now)) {
                        if let Some(deadline) =
                            lease(lease_seconds, lease_expires_at, now, session_limit)
                        {
                            peer.lease_deadline = deadline.min(peer.hard_deadline);
                        } else {
                            reply = Reply::error(request.request_id, "invalid_offer");
                        }
                    } else {
                        reply = Reply::error(request.request_id, "session_not_found");
                    }
                }
                Operation::Close { session_id } => {
                    peers.remove(&session_id);
                }
                Operation::Status => {
                    reply.state = Some(if media_failed {
                        "error"
                    } else if last_video.is_some() {
                        "ready"
                    } else {
                        "starting"
                    });
                    reply.viewer_count = Some(
                        peers
                            .values()
                            .filter(|p| p.connected && !p.expired(now))
                            .count(),
                    );
                    reply.active_session_count =
                        Some(peers.values().filter(|p| !p.expired(now)).count());
                    reply.media_active = Some(input.is_some() && last_video.is_some());
                }
                Operation::Stop => stop = true,
            }
            output(&reply)?;
            if stop {
                return Ok(());
            }
        }
        let wake = peers
            .values()
            .map(|p| {
                p.timeout
                    .min(p.lease_deadline)
                    .min(p.hard_deadline)
                    .min(if p.connected {
                        p.hard_deadline
                    } else {
                        p.negotiation_deadline
                    })
            })
            .min()
            .unwrap_or(now + Duration::from_millis(100));
        poll.poll(
            &mut events,
            Some(if readable {
                Duration::ZERO
            } else {
                wake.saturating_duration_since(Instant::now())
                    .min(Duration::from_millis(100))
            }),
        )?;
        for event in &events {
            if event.token() == MEDIA {
                readable = true;
            }
        }
        if !readable {
            continue;
        }
        for _ in 0..256 {
            let (len, source) = match media.recv_from(&mut bytes) {
                Ok(packet) => packet,
                Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                    readable = false;
                    break;
                }
                Err(error) => return Err(error.into()),
            };
            let now = Instant::now();
            let Ok(contents) = bytes[..len].try_into() else {
                continue;
            };
            let input = Input::Receive(
                now,
                Receive {
                    proto: Protocol::Udp,
                    source,
                    destination: candidate_addr,
                    contents,
                },
            );
            if let Some(peer) = peers
                .values_mut()
                .find(|p| !p.expired(now) && p.rtc.accepts(&input))
                && (peer.rtc.handle_input(input).is_err() || peer.pump(&media, now).is_err())
            {
                peer.failed = true;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn level_1b_flag_only_applies_to_its_defined_profiles() {
        // Given: Level 1.1 with constraint_set3 in baseline, main, extended and high profiles.
        let profiles = [0x42100b, 0x4d100b, 0x58100b, 0x64100b];
        // When: Comparing each profile's advertised receiver level.
        let levels = profiles.map(h264_level);
        // Then: High stays at level 1.1; the other profiles encode level 1b.
        assert_eq!(levels, [105, 105, 105, 110]);
    }

    #[test]
    fn queued_authorization_cannot_outlive_absolute_expiry() {
        // Given: A relative lease longer than the authorization's remaining time.
        let now = Instant::now();
        let epoch = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();
        // When: The control command is processed with its absolute expiry.
        let deadline = lease(60.0, Some(epoch + 0.2), now, Duration::from_secs(10)).unwrap();
        // Then: Neither queue time nor the relative bound extends authorization.
        assert!(deadline.duration_since(now) <= Duration::from_millis(201));
        assert!(lease(60.0, Some(epoch - 1.0), now, Duration::from_secs(10)).is_none());
    }

    #[test]
    fn leases_are_finite_positive_and_capped() {
        // Given: A ten-second maximum lifetime.
        let now = Instant::now();
        let maximum = Duration::from_secs(10);
        // When: A longer lease and invalid values are supplied.
        let deadline = lease(60.0, None, now, maximum).unwrap();
        // Then: The maximum is enforced and invalid leases are refused.
        assert_eq!(deadline.duration_since(now), maximum);
        for value in [0.0, -1.0, f64::INFINITY, f64::NAN] {
            assert!(lease(value, None, now, maximum).is_none());
        }
    }
}
