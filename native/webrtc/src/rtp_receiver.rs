//! Drain FFmpeg's private loopback RTP independently of WebRTC encryption/fanout.
//! Only complete bounded access units cross to the media loop. A full consumer
//! queue stops the source rather than forwarding broken compressed interframes.

use crate::rtp::{EncodedFrame, Event, H264Assembler, Packet};
use mio::net::UdpSocket;
use mio::{Events, Interest, Poll, Token, Waker};
use std::io;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, SyncSender, TryRecvError, TrySendError};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

const VIDEO: Token = Token(0);
const AUDIO: Token = Token(1);
const CANCEL: Token = Token(2);
const VIDEO_CAPACITY: usize = 8;
const AUDIO_CAPACITY: usize = 64;
const RECEIVE_BATCH: usize = 4096;
type Result<T> = std::result::Result<T, &'static str>;

pub struct RtpReceiver {
    video: Receiver<Event>,
    audio: Receiver<Event>,
    failure: Arc<Mutex<Option<&'static str>>>,
    cancel: Arc<AtomicBool>,
    cancel_waker: Waker,
    worker: Option<JoinHandle<()>>,
    next_audio: bool,
}

impl RtpReceiver {
    pub fn start(
        video: &std::net::UdpSocket,
        audio: &std::net::UdpSocket,
        media_waker: Arc<Waker>,
    ) -> Result<Self> {
        let mut video = UdpSocket::from_std(video.try_clone().map_err(|_| "rtp_source_failed")?);
        let mut audio = UdpSocket::from_std(audio.try_clone().map_err(|_| "rtp_source_failed")?);
        let mut poll = Poll::new().map_err(|_| "rtp_source_failed")?;
        for (socket, token) in [(&mut video, VIDEO), (&mut audio, AUDIO)] {
            poll.registry()
                .register(socket, token, Interest::READABLE)
                .map_err(|_| "rtp_source_failed")?;
        }
        let cancel_waker = Waker::new(poll.registry(), CANCEL).map_err(|_| "rtp_source_failed")?;
        let (video_tx, video_rx) = mpsc::sync_channel(VIDEO_CAPACITY);
        let (audio_tx, audio_rx) = mpsc::sync_channel(AUDIO_CAPACITY);
        let failure = Arc::new(Mutex::new(None));
        let worker_failure = Arc::clone(&failure);
        let cancel = Arc::new(AtomicBool::new(false));
        let worker_cancel = Arc::clone(&cancel);
        let worker = thread::Builder::new()
            .name("preview-rtp".into())
            .spawn(move || {
                let sockets = [video, audio];
                let senders = [video_tx, audio_tx];
                let outcome = receive(&mut poll, &sockets, &senders, &worker_cancel, &media_waker);
                // Keep terminal failure outside both full media queues.
                if let Err(code) = outcome
                    && let Ok(mut failure) = worker_failure.lock()
                {
                    *failure = Some(code);
                }
                drop(senders);
                let _ = media_waker.wake();
            })
            .map_err(|_| "rtp_source_failed")?;
        Ok(Self {
            video: video_rx,
            audio: audio_rx,
            failure,
            cancel,
            cancel_waker,
            worker: Some(worker),
            next_audio: false,
        })
    }

    pub fn try_recv(&mut self) -> Result<Option<Event>> {
        if let Some(code) = *self.failure.lock().map_err(|_| "rtp_source_failed")? {
            return Err(code);
        }
        // Alternate priority so neither continuously replenished queue can
        // monopolize the bounded delivery batch.
        let receivers = if self.next_audio {
            [&self.audio, &self.video]
        } else {
            [&self.video, &self.audio]
        };
        self.next_audio = !self.next_audio;
        for receiver in receivers {
            match receiver.try_recv() {
                Ok(event) => return Ok(Some(event)),
                Err(TryRecvError::Empty) => {}
                Err(TryRecvError::Disconnected) => return Err("rtp_source_failed"),
            }
        }
        Ok(None)
    }
}

impl Drop for RtpReceiver {
    fn drop(&mut self) {
        self.cancel.store(true, Ordering::Release);
        let _ = self.cancel_waker.wake();
        // Reads are nonblocking, cancellation is checked for every datagram,
        // and waking the private poll avoids waiting on input or a timeout.
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

fn publish(sender: &SyncSender<Event>, waker: &Waker, event: Event) -> Result<()> {
    sender.try_send(event).map_err(|error| match error {
        TrySendError::Full(_) => "media_queue_overflow",
        TrySendError::Disconnected(_) => "rtp_source_failed",
    })?;
    waker.wake().map_err(|_| "rtp_source_failed")
}

fn receive(
    poll: &mut Poll,
    sockets: &[UdpSocket; 2],
    senders: &[SyncSender<Event>; 2],
    cancel: &AtomicBool,
    media_waker: &Waker,
) -> Result<()> {
    let mut events = Events::with_capacity(3);
    let mut readable = [false; 2];
    let mut bytes = [0; 65_536];
    let mut assembler = H264Assembler::default();
    while !cancel.load(Ordering::Acquire) {
        // Mio is edge-triggered. Keep readiness across bounded batches until
        // WouldBlock so a large access unit needs no subsequent arrival edge.
        poll.poll(
            &mut events,
            Some(if readable.iter().any(|ready| *ready) {
                Duration::ZERO
            } else {
                Duration::from_millis(100)
            }),
        )
        .map_err(|_| "rtp_source_failed")?;
        for event in &events {
            if let Some(ready) = readable.get_mut(event.token().0) {
                *ready = true;
            }
        }
        for (index, ready) in readable.iter_mut().enumerate() {
            if !*ready {
                continue;
            }
            let socket = &sockets[index];
            for _ in 0..RECEIVE_BATCH {
                if cancel.load(Ordering::Acquire) {
                    return Ok(());
                }
                let (length, source) = match socket.recv_from(&mut bytes) {
                    Ok(packet) => packet,
                    Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                        *ready = false;
                        break;
                    }
                    Err(_) => return Err("rtp_source_failed"),
                };
                if !source.ip().is_loopback() {
                    continue;
                }
                let Some(packet) = Packet::parse(&bytes[..length]) else {
                    continue;
                };
                if index == VIDEO.0 {
                    if let Some(frame) = assembler.push(packet) {
                        publish(
                            &senders[VIDEO.0],
                            media_waker,
                            Event::Frame(EncodedFrame {
                                timestamp: frame.timestamp,
                                keyframe: frame.keyframe,
                                data: frame.data.into(),
                            }),
                        )?;
                    }
                } else if packet.payload_type == 97 && packet.payload.len() <= 4000 {
                    publish(
                        &senders[AUDIO.0],
                        media_waker,
                        Event::Audio {
                            timestamp: packet.timestamp,
                            data: packet.payload.into(),
                        },
                    )?;
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::burst_rtp;
    use std::net::UdpSocket;
    use std::time::Instant;

    fn socket() -> UdpSocket {
        let socket = UdpSocket::bind("127.0.0.1:0").unwrap();
        socket.set_nonblocking(true).unwrap();
        socket2::SockRef::from(&socket)
            .set_recv_buffer_size(32 * 1024)
            .unwrap();
        socket
    }

    fn sender(destination: std::net::SocketAddr, frames: u32) -> JoinHandle<()> {
        thread::spawn(move || {
            let socket = UdpSocket::bind("127.0.0.1:0").unwrap();
            let nals = burst_rtp::keyframe();
            let mut sequence = 1;
            for frame in 0..frames {
                burst_rtp::send(&socket, destination, &nals, &mut sequence, frame * 9000);
            }
        })
    }

    #[test]
    fn burst_keyframes_survive_a_busy_consumer_without_growing_its_udp_buffer() {
        // Given: Valid 197 KiB access units and a private 32 KiB receive buffer.
        // The inline baseline cannot drain while its consumer is doing fanout.
        let baseline = socket();
        sender(baseline.local_addr().unwrap(), 1).join().unwrap();
        let mut assembler = H264Assembler::default();
        let mut packet = [0; 65_536];
        let mut baseline_frames = 0;
        while let Ok((length, _)) = baseline.recv_from(&mut packet) {
            if let Some(packet) = Packet::parse(&packet[..length]) {
                baseline_frames += usize::from(assembler.push(packet).is_some());
            }
        }
        assert_eq!(baseline_frames, 0, "baseline must reproduce local RTP loss");
        let video = socket();
        let audio = socket();
        let poll = Poll::new().unwrap();
        let waker = Arc::new(Waker::new(poll.registry(), Token(0)).unwrap());
        let mut source = RtpReceiver::start(&video, &audio, waker).unwrap();

        // When: The same packet bursts arrive while this consumer does no reads.
        sender(video.local_addr().unwrap(), 4).join().unwrap();

        // Then: Every whole keyframe survives, including the last marker with no
        // subsequent datagram to trigger another readiness edge.
        let deadline = Instant::now() + Duration::from_secs(1);
        let mut frames = Vec::new();
        while frames.len() < 4 {
            if let Some(Event::Frame(frame)) = source.try_recv().unwrap() {
                assert!(frame.keyframe);
                assert!(frame.data.len() > 192 * 1024);
                assert!(frame.data.ends_with(&[0x41, 0x41, 0x80]));
                frames.push(frame.timestamp);
            }
            assert!(Instant::now() < deadline, "whole burst frames were lost");
            thread::sleep(Duration::from_millis(1));
        }
        assert_eq!(frames, [0, 9000, 18000, 27000]);
    }

    #[test]
    fn full_video_queue_fails_before_returning_queued_frames_and_cancels_promptly() {
        // Given: A complete-frame receiver whose consumer remains idle.
        let video = socket();
        let audio = socket();
        let poll = Poll::new().unwrap();
        let waker = Arc::new(Waker::new(poll.registry(), Token(0)).unwrap());
        let mut source = RtpReceiver::start(&video, &audio, waker).unwrap();

        // When: More complete frames arrive than its bounded channel can retain.
        sender(video.local_addr().unwrap(), 10).join().unwrap();

        // Then: Terminal overflow remains visible ahead of old compressed media.
        assert!(matches!(source.try_recv(), Err("media_queue_overflow")));
        assert!(matches!(source.try_recv(), Err("media_queue_overflow")));
        let stopped = Instant::now();
        drop(source);
        assert!(stopped.elapsed() < Duration::from_secs(1));
    }

    #[test]
    fn dropping_an_idle_receiver_releases_both_private_socket_clones() {
        // Given: An idle source is waiting for input on its private poll.
        let video = socket();
        let audio = socket();
        let ports = [video.local_addr().unwrap(), audio.local_addr().unwrap()];
        let poll = Poll::new().unwrap();
        let waker = Arc::new(Waker::new(poll.registry(), Token(0)).unwrap());
        let source = RtpReceiver::start(&video, &audio, waker).unwrap();
        drop(video);
        drop(audio);

        // When: Its media owner stops or loses the control connection.
        let stopped = Instant::now();
        drop(source);

        // Then: No input is required for shutdown and both ports can be rebound.
        assert!(stopped.elapsed() < Duration::from_secs(1));
        for port in ports {
            assert!(UdpSocket::bind(port).is_ok());
        }
    }
}
