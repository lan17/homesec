//! Valid large H.264 access units fragmented into camera-sized loopback datagrams.

use std::net::{SocketAddr, UdpSocket};
use std::thread;
use std::time::Duration;

pub(crate) fn keyframe() -> Vec<Vec<u8>> {
    let mut nals =
        super::rtsp_camera::annex_b_nals(include_bytes!("../fixtures/baseline-160x120.h264"));
    let idr = nals.iter().position(|nal| nal[0] & 31 == 5).unwrap();
    nals.truncate(idr + 1);
    // A valid user_data_unregistered SEI is carried through encrypted WebRTC,
    // while a decoder ignores its application-defined payload. Unlike filler
    // NALs (which str0m intentionally strips), it exercises large-frame fanout
    // without an expensive encoder or a checked-in large binary fixture.
    let payload_size = 192 * 1024;
    let mut metadata = vec![0x06, 5];
    let mut remaining = payload_size;
    while remaining >= 255 {
        metadata.push(255);
        remaining -= 255;
    }
    metadata.push(remaining as u8);
    metadata.extend_from_slice(&[0x55; 16]); // 16-byte application UUID.
    metadata.resize(metadata.len() + payload_size - 16, 0x41);
    metadata.push(0x80); // rbsp_trailing_bits.
    nals.push(metadata);
    nals
}

pub(crate) fn send(
    socket: &UdpSocket,
    destination: SocketAddr,
    nals: &[Vec<u8>],
    sequence: &mut u16,
    timestamp: u32,
) {
    for (index, nal) in nals.iter().enumerate() {
        let last_nal = index + 1 == nals.len();
        if nal.len() <= 1188 {
            packet(socket, destination, sequence, timestamp, last_nal, nal);
        } else {
            let chunks: Vec<_> = nal[1..].chunks(1186).collect();
            for (fragment, bytes) in chunks.iter().enumerate() {
                let last = fragment + 1 == chunks.len();
                let mut payload = Vec::with_capacity(bytes.len() + 2);
                payload.push((nal[0] & 0xe0) | 28);
                payload.push(
                    (nal[0] & 31)
                        | if fragment == 0 { 0x80 } else { 0 }
                        | if last { 0x40 } else { 0 },
                );
                payload.extend_from_slice(bytes);
                packet(
                    socket,
                    destination,
                    sequence,
                    timestamp,
                    last_nal && last,
                    &payload,
                );
            }
        }
    }
}

fn packet(
    socket: &UdpSocket,
    destination: SocketAddr,
    sequence: &mut u16,
    timestamp: u32,
    marker: bool,
    payload: &[u8],
) {
    let mut packet = vec![0x80, 96 | if marker { 0x80 } else { 0 }];
    packet.extend_from_slice(&sequence.to_be_bytes());
    packet.extend_from_slice(&timestamp.to_be_bytes());
    packet.extend_from_slice(&1_u32.to_be_bytes());
    packet.extend_from_slice(payload);
    assert!(packet.len() <= 1200);
    socket.send_to(&packet, destination).unwrap();
    *sequence = sequence.wrapping_add(1);
    // A short packet burst keeps the producer below loopback receiver capacity,
    // while remaining much larger than the private 32 KiB regression socket.
    thread::sleep(Duration::from_micros(100));
}
