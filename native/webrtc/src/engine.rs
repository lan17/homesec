use crate::Options;
use crate::protocol::{MAX_CONTROL_BYTES, MAX_SDP_BYTES, Operation, Reply, Request};
use crate::rtp::{H264Assembler, Packet, TimestampClock};
use mio::net::UdpSocket;
use mio::{Events, Interest, Poll, Token, Waker};
use serde::Serialize;
use std::collections::HashMap;
use std::io::{self, BufRead, Read, Write};
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use std::sync::mpsc::{self, Receiver};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use str0m::change::SdpOffer;
use str0m::format::Codec;
use str0m::media::{Frequency, MediaKind, MediaTime, Mid, Pt};
use str0m::net::{Protocol, Receive};
use str0m::{Candidate, Event, IceConnectionState, Input, Output, Rtc, RtcConfig};

const MEDIA: Token = Token(0);
const VIDEO: Token = Token(1);
const AUDIO: Token = Token(2);
const CONTROL: Token = Token(3);
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

enum Control {
    Request(Request),
    End,
}

fn controls(poll: &Poll) -> Result<Receiver<Control>> {
    let (sender, receiver) = mpsc::sync_channel(16);
    let waker = Arc::new(Waker::new(poll.registry(), CONTROL)?);
    std::thread::spawn(move || {
        let mut input = io::stdin().lock();
        loop {
            let mut bytes = Vec::new();
            // take() bounds allocation even if the sender forgets a newline.
            let read = (&mut input)
                .take((MAX_CONTROL_BYTES + 1) as u64)
                .read_until(b'\n', &mut bytes);
            let Ok(size) = read else { break };
            if size == 0 || size > MAX_CONTROL_BYTES || bytes.last() != Some(&b'\n') {
                break;
            }
            let Ok(request) = serde_json::from_slice::<Request>(&bytes) else {
                break;
            };
            if request.request_id.len() > 128 {
                break;
            }
            if sender.send(Control::Request(request)).is_err() {
                return;
            }
            let _ = waker.wake();
        }
        let _ = sender.send(Control::End);
        let _ = waker.wake();
    });
    Ok(receiver)
}

fn output(value: &impl Serialize) -> Result<()> {
    let mut stdout = io::stdout().lock();
    serde_json::to_writer(&mut stdout, value)?;
    stdout.write_all(b"\n")?;
    stdout.flush()?;
    Ok(())
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
    let bind_ip = match options.advertised_ip {
        IpAddr::V4(_) => IpAddr::V4(Ipv4Addr::UNSPECIFIED),
        IpAddr::V6(_) => IpAddr::V6(Ipv6Addr::UNSPECIFIED),
    };
    let mut media = (options.udp_port_start..=options.udp_port_end)
        .find_map(|port| UdpSocket::bind(SocketAddr::new(bind_ip, port)).ok())
        .ok_or("media_port_unavailable")?;
    let candidate_addr = SocketAddr::new(options.advertised_ip, media.local_addr()?.port());
    let mut video = UdpSocket::bind("127.0.0.1:0".parse()?)?;
    let mut audio = UdpSocket::bind("127.0.0.1:0".parse()?)?;
    let mut poll = Poll::new()?;
    for (socket, token) in [
        (&mut media, MEDIA),
        (&mut video, VIDEO),
        (&mut audio, AUDIO),
    ] {
        poll.registry()
            .register(socket, token, Interest::READABLE)?;
    }
    let controls = controls(&poll)?;
    output(
        &serde_json::json!({"event":"ready", "video_port":video.local_addr()?.port(),
        "audio_port":audio.local_addr()?.port(), "media_port":candidate_addr.port()}),
    )?;
    let mut peers: HashMap<String, Peer> = HashMap::new();
    let mut child: Option<MediaChild> = None;
    let mut started: Option<Instant> = None;
    let mut last_video: Option<Instant> = None;
    let mut media_failed = false;
    let mut assembler = H264Assembler::default();
    let mut video_clock = TimestampClock::default();
    let mut audio_clock = TimestampClock::default();
    let mut events = Events::with_capacity(16);
    // Mio readiness is edge-triggered. A bounded receive batch must keep its
    // readiness until recv_from reaches WouldBlock, even without a new edge.
    let mut readable = [false; 3];
    let mut bytes = [0u8; 65536];
    loop {
        let now = Instant::now();
        if let Some(process) = &mut child {
            if process.0.try_wait()?.is_some()
                || last_video
                    .or(started)
                    .is_some_and(|last| now.duration_since(last) >= MEDIA_TIMEOUT)
            {
                child = None;
                media_failed = true;
                peers.clear();
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
                Operation::Start { ffmpeg_args } => {
                    if started.is_some()
                        || ffmpeg_args.is_empty()
                        || ffmpeg_args.iter().map(String::len).sum::<usize>()
                            > MAX_CONTROL_BYTES / 2
                    {
                        reply = Reply::error(request.request_id, "preview_temporarily_unavailable");
                    } else {
                        match Command::new("ffmpeg")
                            .args(ffmpeg_args)
                            .stdin(Stdio::null())
                            .stdout(Stdio::null())
                            .stderr(Stdio::null())
                            .spawn()
                        {
                            Ok(process) => {
                                child = Some(MediaChild(process));
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
                }
                Operation::Offer {
                    session_id,
                    sdp,
                    lease_seconds,
                    lease_expires_at,
                } => {
                    let deadline = lease(lease_seconds, lease_expires_at, now, session_limit);
                    if child.is_none() || media_failed {
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
                        let mut config = RtcConfig::new()
                            .clear_codecs()
                            .enable_opus(true, false)
                            .set_ice_lite(true)
                            .set_send_buffer_video(2048)
                            .set_send_buffer_audio(64);
                        // str0m packetizes H.264 with STAP-A/FU-A, which requires mode 1.
                        for (pt, resend, profile) in [
                            (127_u8, 121_u8, 0x42001f),
                            (108, 109, 0x42e01f),
                            (123, 119, 0x4d001f),
                            (114, 115, 0x64001f),
                        ] {
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
                        match SdpOffer::from_sdp_string(&sdp)
                            .ok()
                            .and_then(|offer| rtc.sdp_api().accept_offer(offer).ok())
                        {
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
                                let has_video = answer.media_lines.iter().any(|line| {
                                    line.direction().is_sending()
                                        && line.rtp_params().iter().any(|p| {
                                            p.spec().codec == Codec::H264
                                                && p.spec().format.packetization_mode == Some(1)
                                        })
                                });
                                if peer.pump(&media, now).is_ok() && has_video && !peer.failed {
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
                    reply.media_active = Some(child.is_some() && last_video.is_some());
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
            Some(if readable.iter().any(|ready| *ready) {
                Duration::ZERO
            } else {
                wake.saturating_duration_since(Instant::now())
                    .min(Duration::from_millis(100))
            }),
        )?;
        for event in &events {
            if let Some(ready) = readable.get_mut(event.token().0) {
                *ready = true;
            }
        }
        for (index, ready) in readable.iter_mut().enumerate() {
            if !*ready {
                continue;
            }
            let token = Token(index);
            let socket = match token {
                MEDIA => &media,
                VIDEO => &video,
                AUDIO => &audio,
                _ => continue,
            };
            for _ in 0..256 {
                let (len, source) = match socket.recv_from(&mut bytes) {
                    Ok(packet) => packet,
                    Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                        *ready = false;
                        break;
                    }
                    Err(error) => return Err(error.into()),
                };
                let now = Instant::now();
                if token == MEDIA {
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
                    {
                        if peer.rtc.handle_input(input).is_err() || peer.pump(&media, now).is_err()
                        {
                            peer.failed = true;
                        }
                    }
                } else if source.ip().is_loopback() && child.is_some() {
                    let Some(packet) = Packet::parse(&bytes[..len]) else {
                        continue;
                    };
                    if token == VIDEO {
                        if let Some(frame) = assembler.push(packet) {
                            last_video = Some(now);
                            if let Some(time) = video_clock.extend(frame.timestamp) {
                                let data: Arc<[u8]> = frame.data.into();
                                for peer in peers.values_mut() {
                                    peer.write(true, frame.keyframe, time, Arc::clone(&data), now);
                                    if peer.pump(&media, now).is_err() {
                                        peer.failed = true;
                                    }
                                }
                            }
                        }
                    } else if packet.payload_type == 97 && packet.payload.len() <= 4000 {
                        if let Some(time) = audio_clock.extend(packet.timestamp) {
                            let data: Arc<[u8]> = packet.payload.into();
                            for peer in peers.values_mut() {
                                peer.write(false, false, time, Arc::clone(&data), now);
                                if peer.pump(&media, now).is_err() {
                                    peer.failed = true;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
