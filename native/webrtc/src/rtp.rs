//! The private FFmpeg handoff: RTP/H.264 access units and Opus packets.
//! WebRTC packetization, encryption and retransmission belong to str0m.

const MAX_FRAME_BYTES: usize = 2 * 1024 * 1024;
const MAX_PARAMETER_BYTES: usize = 4096;

pub struct Packet<'a> {
    pub timestamp: u32,
    pub sequence: u16,
    pub marker: bool,
    pub payload_type: u8,
    pub payload: &'a [u8],
}

impl<'a> Packet<'a> {
    pub fn parse(data: &'a [u8]) -> Option<Self> {
        if data.len() < 12 || data[0] >> 6 != 2 {
            return None;
        }
        let mut start = 12 + usize::from(data[0] & 15) * 4;
        if data[0] & 0x10 != 0 {
            let header = data.get(start..start + 4)?;
            start += 4 + usize::from(u16::from_be_bytes([header[2], header[3]])) * 4;
        }
        let mut end = data.len();
        if data[0] & 0x20 != 0 {
            let padding = usize::from(*data.last()?);
            if padding == 0 || padding > end.saturating_sub(start) {
                return None;
            }
            end -= padding;
        }
        let payload = data.get(start..end)?;
        if payload.is_empty() {
            return None;
        }
        Some(Self {
            timestamp: u32::from_be_bytes(data[4..8].try_into().ok()?),
            sequence: u16::from_be_bytes(data[2..4].try_into().ok()?),
            marker: data[1] & 0x80 != 0,
            payload_type: data[1] & 0x7f,
            payload,
        })
    }
}

pub struct Frame {
    pub timestamp: u32,
    pub keyframe: bool,
    pub data: Vec<u8>,
}

/// Never deliver a partial access unit or dependent frames after input loss.
#[derive(Default)]
pub struct H264Assembler {
    timestamp: Option<u32>,
    expected_sequence: Option<u16>,
    data: Vec<u8>,
    fragment_start: Option<(usize, u8)>,
    damaged: bool,
    keyframe: bool,
    recovering: bool,
    sps: Option<Vec<u8>>,
    pps: Option<Vec<u8>>,
}

impl H264Assembler {
    pub fn push(&mut self, packet: Packet<'_>) -> Option<Frame> {
        if packet.payload_type != 96 {
            return None;
        }
        let new_frame = self.timestamp != Some(packet.timestamp);
        if new_frame {
            if !self.data.is_empty() || self.fragment_start.is_some() {
                self.recovering = true;
            }
            self.timestamp = Some(packet.timestamp);
            self.data.clear();
            self.fragment_start = None;
            self.damaged = false;
            self.keyframe = false;
        }
        if self.expected_sequence.is_some_and(|v| v != packet.sequence) {
            self.recovering = true;
            if !new_frame {
                self.damaged = true;
            }
        }
        self.expected_sequence = Some(packet.sequence.wrapping_add(1));
        if !self.damaged && !self.append_payload(packet.payload) {
            self.damaged = true;
            self.recovering = true;
            self.data.clear();
        }
        if !packet.marker {
            return None;
        }
        if self.damaged || self.fragment_start.is_some() || self.data.is_empty() {
            self.recovering = true;
            self.data.clear();
            return None;
        }
        let keyframe = self.keyframe && self.sps.is_some() && self.pps.is_some();
        if self.recovering && !keyframe {
            self.data.clear();
            return None;
        }
        if keyframe {
            self.recovering = false;
        }
        let data = std::mem::take(&mut self.data);
        let mut output = Vec::with_capacity(data.len() + 2 * MAX_PARAMETER_BYTES + 8);
        if keyframe {
            for parameter in [&self.sps, &self.pps].into_iter().flatten() {
                output.extend_from_slice(&[0, 0, 0, 1]);
                output.extend_from_slice(parameter);
            }
        }
        output.extend_from_slice(&data);
        Some(Frame {
            timestamp: packet.timestamp,
            keyframe,
            data: output,
        })
    }

    fn append_payload(&mut self, payload: &[u8]) -> bool {
        if payload[0] & 0x80 != 0 {
            return false;
        }
        match payload[0] & 31 {
            1..=23 => {
                if self.fragment_start.is_some() {
                    return false;
                }
                self.append_nal(payload)
            }
            24 => {
                if self.fragment_start.is_some() {
                    return false;
                }
                let mut remaining = &payload[1..];
                while !remaining.is_empty() {
                    if remaining.len() < 2 {
                        return false;
                    }
                    let len = usize::from(u16::from_be_bytes([remaining[0], remaining[1]]));
                    if len == 0 || remaining.len() < len + 2 {
                        return false;
                    }
                    if !self.append_nal(&remaining[2..len + 2]) {
                        return false;
                    }
                    remaining = &remaining[len + 2..];
                }
                true
            }
            28 => {
                if payload.len() < 3 || payload[1] & 0x20 != 0 {
                    return false;
                }
                let start = payload[1] & 0x80 != 0;
                let end = payload[1] & 0x40 != 0;
                let nal_type = payload[1] & 31;
                if !(1..=23).contains(&nal_type) || (start && end) {
                    return false;
                }
                if start {
                    if self.fragment_start.is_some() {
                        return false;
                    }
                    self.fragment_start = Some((self.data.len(), (payload[0] & 0xe0) | nal_type));
                    self.keyframe |= nal_type == 5;
                    self.data
                        .extend_from_slice(&[0, 0, 0, 1, (payload[0] & 0xe0) | nal_type]);
                } else if !self
                    .fragment_start
                    .is_some_and(|(_, header)| header == (payload[0] & 0xe0) | nal_type)
                {
                    return false;
                }
                if self.data.len() + payload.len() - 2 > MAX_FRAME_BYTES {
                    return false;
                }
                self.data.extend_from_slice(&payload[2..]);
                if end {
                    let (offset, _) = self.fragment_start.take().unwrap_or_default();
                    let nal = &self.data[offset + 4..];
                    match nal_type {
                        7 | 8 if nal.len() > MAX_PARAMETER_BYTES => return false,
                        7 => self.sps = Some(nal.to_vec()),
                        8 => self.pps = Some(nal.to_vec()),
                        _ => {}
                    }
                }
                true
            }
            _ => false,
        }
    }

    fn append_nal(&mut self, nal: &[u8]) -> bool {
        let kind = nal[0] & 31;
        if nal[0] & 0x80 != 0
            || !(1..=23).contains(&kind)
            || self.data.len() + nal.len() + 4 > MAX_FRAME_BYTES
        {
            return false;
        }
        match kind {
            7 | 8 if nal.len() > MAX_PARAMETER_BYTES => return false,
            7 => self.sps = Some(nal.to_vec()),
            8 => self.pps = Some(nal.to_vec()),
            5 => self.keyframe = true,
            _ => {}
        }
        self.data.extend_from_slice(&[0, 0, 0, 1]);
        self.data.extend_from_slice(nal);
        true
    }
}

#[derive(Default)]
pub struct TimestampClock {
    previous: Option<u32>,
    extended: u64,
}

impl TimestampClock {
    pub fn extend(&mut self, value: u32) -> Option<u64> {
        if let Some(previous) = self.previous {
            let difference = value.wrapping_sub(previous);
            if difference > i32::MAX as u32 {
                return None;
            }
            self.extended += u64::from(difference);
        } else {
            self.extended = u64::from(value);
        }
        self.previous = Some(value);
        Some(self.extended)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn packet(seq: u16, timestamp: u32, marker: bool, payload: &[u8]) -> Packet<'_> {
        Packet {
            sequence: seq,
            timestamp,
            marker,
            payload_type: 96,
            payload,
        }
    }

    fn parameters(assembler: &mut H264Assembler, seq: u16) {
        assembler.push(packet(seq, 0, false, &[0x67, 0x42, 0, 31]));
        assembler.push(packet(seq + 1, 0, true, &[0x68, 1]));
    }

    #[test]
    fn mismatched_fragment_headers_require_keyframe_recovery() {
        // Given: An IDR start followed by a fragment claiming another NAL type.
        let mut assembler = H264Assembler::default();
        parameters(&mut assembler, 1);
        assembler.push(packet(3, 9000, false, &[0x7c, 0x85, 1]));
        // When: The malformed continuation and a dependent frame arrive.
        let malformed = assembler.push(packet(4, 9000, true, &[0x7c, 0x41, 2]));
        let dependent = assembler.push(packet(5, 18000, true, &[0x61, 1]));
        // Then: Neither is forwarded; a complete IDR restores decodability.
        assert!(malformed.is_none());
        assert!(dependent.is_none());
        assert!(
            assembler
                .push(packet(6, 27000, true, &[0x65, 1]))
                .unwrap()
                .keyframe
        );
    }

    #[test]
    fn keyframe_includes_cached_parameters_for_late_join() {
        // Given: Parameter sets and a fragmented IDR access unit.
        let mut assembler = H264Assembler::default();
        parameters(&mut assembler, 1);
        assert!(
            assembler
                .push(packet(3, 9000, false, &[0x7c, 0x85, 1, 2]))
                .is_none()
        );
        // When: The final contiguous fragment arrives.
        let frame = assembler
            .push(packet(4, 9000, true, &[0x7c, 0x45, 3]))
            .unwrap();
        // Then: A complete Annex B keyframe includes decoder initialization.
        assert!(frame.keyframe);
        assert_eq!(frame.timestamp, 9000);
        assert!(frame.data.ends_with(&[0, 0, 0, 1, 0x65, 1, 2, 3]));
        assert!(frame.data.starts_with(&[0, 0, 0, 1, 0x67]));
    }

    #[test]
    fn packet_loss_drops_dependents_until_complete_keyframe() {
        // Given: A missing fragment in an IDR frame.
        let mut assembler = H264Assembler::default();
        parameters(&mut assembler, 1);
        assembler.push(packet(3, 9000, false, &[0x7c, 0x85, 1]));
        // When: The end fragment arrives with a sequence gap, then a P-frame.
        assert!(
            assembler
                .push(packet(5, 9000, true, &[0x7c, 0x45, 3]))
                .is_none()
        );
        assert!(assembler.push(packet(6, 12000, true, &[0x61, 7])).is_none());
        // Then: Only the next complete IDR resumes delivery.
        assert!(
            assembler
                .push(packet(7, 15000, true, &[0x65, 8]))
                .unwrap()
                .keyframe
        );
    }

    #[test]
    fn oversized_and_malformed_packets_do_not_produce_frames() {
        // Given: Invalid STAP lengths and an oversized access unit.
        let mut assembler = H264Assembler::default();
        // When: They are delivered through the private RTP input.
        assert!(
            assembler
                .push(packet(1, 0, true, &[24, 255, 255, 0]))
                .is_none()
        );
        let mut oversized = vec![1; MAX_FRAME_BYTES + 1];
        oversized[0] = 0x65;
        // Then: Neither payload is forwarded or accumulated unboundedly.
        assert!(assembler.push(packet(2, 9000, true, &oversized)).is_none());
    }

    #[test]
    fn parser_rejects_truncated_extension_and_invalid_padding() {
        // Given: RTP headers with incomplete extension or excessive padding.
        let mut header = vec![0x90, 96, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0];
        // When/Then: Parsing rejects each truncated boundary safely.
        assert!(Packet::parse(&header).is_none());
        header[0] = 0xa0;
        header.push(255);
        assert!(Packet::parse(&header).is_none());
    }

    #[test]
    fn timestamp_rollover_is_monotonic_and_old_packet_is_rejected() {
        // Given: A stream close to the RTP clock rollover.
        let mut clock = TimestampClock::default();
        let before = clock.extend(u32::MAX - 10).unwrap();
        // When: The clock wraps and an old packet arrives.
        let after = clock.extend(9).unwrap();
        // Then: Media time advances without accepting the stale packet.
        assert_eq!(after - before, 20);
        assert_eq!(clock.extend(u32::MAX - 9), None);
    }
}
