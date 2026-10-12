//! Force the final DTLS flight and initial H.264 parameters into one UDP batch.

use super::*;
use std::net::SocketAddr;
use str0m::config::DtlsVersion;
use str0m::media::{Frequency, MediaTime};

fn drain_sender(
    rtc: &mut Rtc,
    socket: &UdpSocket,
    connected: &mut bool,
    queued: &mut Vec<(SocketAddr, Vec<u8>)>,
) {
    loop {
        match rtc.poll_output().unwrap() {
            Output::Event(Event::Connected) => *connected = true,
            Output::Transmit(packet) => {
                if *connected {
                    // The passive DTLS 1.2 peer reports Connected before it
                    // transmits its final flight. Keep that flight queued so
                    // the viewer cannot install its SRTP keys yet.
                    queued.push((packet.destination, packet.contents.to_vec()));
                } else {
                    socket
                        .send_to(&packet.contents, packet.destination)
                        .unwrap();
                }
            }
            Output::Timeout(_) => break,
            Output::Event(_) => {}
        }
    }
}

#[test]
fn first_keyframe_decodes_when_queued_with_the_final_dtls_flight() {
    // Given: A real receive-only Viewer and a passive DTLS 1.2 sender, with
    // the final handshake flight deliberately held until media is ready.
    let socket = UdpSocket::bind("127.0.0.1:0").unwrap();
    socket.set_nonblocking(true).unwrap();
    let sender_socket = UdpSocket::bind("127.0.0.1:0").unwrap();
    sender_socket.set_nonblocking(true).unwrap();
    let mut rtc = Rtc::builder()
        .clear_codecs()
        .enable_h264(true)
        .set_dtls_version(DtlsVersion::Dtls12)
        .build(Instant::now());
    let mut sender = Rtc::builder()
        .clear_codecs()
        .enable_h264(true)
        .set_dtls_version(DtlsVersion::Dtls12)
        .build(Instant::now());
    rtc.add_local_candidate(Candidate::host(socket.local_addr().unwrap(), "udp").unwrap())
        .unwrap();
    sender
        .add_local_candidate(Candidate::host(sender_socket.local_addr().unwrap(), "udp").unwrap())
        .unwrap();
    let mut change = rtc.sdp_api();
    let video_mid = change.add_media(MediaKind::Video, Direction::RecvOnly, None, None, None);
    let (offer, pending) = change.apply().unwrap();
    let answer = sender.sdp_api().accept_offer(offer).unwrap();
    rtc.sdp_api().accept_answer(pending, answer).unwrap();
    let mut viewer = Viewer {
        rtc,
        socket,
        connected: false,
        video_frames: 0,
        keyframes: 0,
        audio_packets: 0,
        video_sample: VideoSample::default(),
        video_payload_types: Vec::new(),
    };
    let mut connected = false;
    let mut queued = Vec::new();
    drain_sender(&mut sender, &sender_socket, &mut connected, &mut queued);
    let deadline = Instant::now() + Duration::from_secs(5);
    let mut packet = [0; 65_536];
    'handshake: loop {
        viewer.progress();
        sender.handle_input(Input::Timeout(Instant::now())).unwrap();
        drain_sender(&mut sender, &sender_socket, &mut connected, &mut queued);
        if connected {
            break;
        }
        loop {
            match sender_socket.recv_from(&mut packet) {
                Ok((length, source)) => {
                    let receive = Receive::new(
                        Protocol::Udp,
                        source,
                        sender_socket.local_addr().unwrap(),
                        &packet[..length],
                    )
                    .unwrap();
                    sender
                        .handle_input(Input::Receive(Instant::now(), receive))
                        .unwrap();
                    drain_sender(&mut sender, &sender_socket, &mut connected, &mut queued);
                    if connected {
                        break 'handshake;
                    }
                }
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => break,
                Err(error) => panic!("sender UDP receive failed: {error}"),
            }
        }
        assert!(Instant::now() < deadline, "DTLS handshake did not complete");
        thread::yield_now();
    }
    assert!(sender.is_connected());
    assert!(
        !viewer.connected,
        "the final DTLS flight must remain queued"
    );

    let nals = annex_b_nals(include_bytes!("../fixtures/baseline-160x120.h264"));
    // The fixture uses access-unit delimiters. Preserve the real first GOP:
    // one IDR with its SPS/PPS followed by two dependent frames, so a decoder
    // cannot recover a missing initial parameter set from another keyframe.
    let frames: Vec<Arc<[u8]>> = nals
        .split(|nal| nal[0] & 31 == 9)
        .filter(|nals| !nals.is_empty())
        .take(3)
        .map(|nals| {
            let mut frame = Vec::new();
            for nal in nals {
                frame.extend_from_slice(&[0, 0, 0, 1]);
                frame.extend_from_slice(nal);
            }
            frame.into()
        })
        .collect();
    assert_eq!(frames.len(), 3);
    let pt = sender
        .writer(video_mid)
        .unwrap()
        .payload_params()
        .find(|params| {
            params.spec().codec == Codec::H264 && params.spec().format.packetization_mode == Some(1)
        })
        .unwrap()
        .pt();
    for (index, frame) in frames.into_iter().enumerate() {
        sender
            .writer(video_mid)
            .unwrap()
            .write(
                pt,
                Instant::now(),
                MediaTime::new(index as u64 * 9000, Frequency::NINETY_KHZ),
                frame,
            )
            .unwrap();
        drain_sender(&mut sender, &sender_socket, &mut connected, &mut queued);
        sender.handle_input(Input::Timeout(Instant::now())).unwrap();
        drain_sender(&mut sender, &sender_socket, &mut connected, &mut queued);
    }
    let final_dtls = queued
        .iter()
        .position(|(_, bytes)| (20..64).contains(&bytes[0]))
        .expect("the sender must queue its final DTLS flight");
    let first_srtp = queued
        .iter()
        .position(|(_, bytes)| (128..192).contains(&bytes[0]) && !(192..224).contains(&bytes[1]))
        .expect("the sender must queue encrypted media");
    assert!(final_dtls < first_srtp);

    // When: The final DTLS flight and the first RTP packet (the SPS/PPS STAP-A)
    // are already waiting in the viewer's socket before one receive-loop call.
    // The remaining packets arrive in the next call, just as a keyframe burst
    // can span two socket drains in the live integration test.
    let remaining = queued.split_off(first_srtp + 1);
    for (destination, bytes) in queued {
        sender_socket.send_to(&bytes, destination).unwrap();
    }
    viewer.progress();
    for (destination, bytes) in remaining {
        sender_socket.send_to(&bytes, destination).unwrap();
    }
    viewer.progress();

    // Then: Processing the handshake installs SRTP keys before the next
    // datagram, preserving the first keyframe's SPS/PPS and decodable sample.
    assert!(viewer.connected);
    assert_eq!(viewer.video_frames, 3, "startup SRTP packets were dropped");
    assert_eq!(viewer.keyframes, 1);
    assert!(viewer.has_video());
    viewer.assert_video_decodes();
}
