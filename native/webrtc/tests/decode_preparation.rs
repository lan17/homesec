//! Compare native preparation with the installed FFmpeg CLI, using synthetic media only.

#[path = "../src/decode.rs"]
mod decode;
#[path = "support/ffmpeg_reference.rs"]
mod ffmpeg_reference;

mod rtp {
    use std::sync::Arc;

    pub struct EncodedFrame {
        pub timestamp: u32,
        pub data: Arc<[u8]>,
    }
}

use decode::{GrayDecoder, GrayFrame};
use std::path::{Path, PathBuf};
use std::sync::Arc;

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/decode")
        .join(name)
}

fn access_units(bytes: &[u8]) -> Vec<&[u8]> {
    let offsets: Vec<_> = bytes
        .windows(5)
        .enumerate()
        .filter_map(|(offset, nal)| (nal == [0, 0, 0, 1, 9]).then_some(offset))
        .collect();
    assert_eq!(offsets.first(), Some(&0), "fixture must begin with an AUD");
    offsets
        .iter()
        .enumerate()
        .map(|(index, start)| {
            &bytes[*start..offsets.get(index + 1).copied().unwrap_or(bytes.len())]
        })
        .collect()
}

fn append(frame: GrayFrame, output: &mut Vec<u8>) {
    assert_eq!((frame.width, frame.height), (320, 240));
    assert_eq!(frame.data.len(), 320 * 240);
    output.extend_from_slice(&frame.data);
}

#[test]
fn native_preparation_matches_cli_cadence_and_color_conversion() {
    // Given: Frozen H.264 patterns covering downsampling, duplication, and two color matrices/ranges.
    let cases = [
        ("15fps-bt709-tv.h264", 15),
        ("7fps-bt709-pc.h264", 7),
        ("15fps-smpte170m-tv.h264", 15),
        ("7fps-smpte170m-pc.h264", 7),
    ];
    for (name, fps) in cases {
        let path = fixture(name);
        let bytes = std::fs::read(&path).unwrap();
        let access_units = access_units(&bytes);
        assert_eq!(access_units.len(), 2 * fps as usize);
        let reference = ffmpeg_reference::prepare_gray(&path, fps, 20);
        assert_eq!(reference.len(), 20 * 320 * 240);
        for initial_timestamp in [0_u32, u32::MAX - 45_000] {
            let mut decoder = GrayDecoder::new().unwrap();
            let mut actual = Vec::new();
            // When: Complete access units use their original RTP cadence, including rollover.
            for (index, data) in access_units.iter().enumerate() {
                let timestamp = initial_timestamp.wrapping_add((index as u32 * 90_000) / fps);
                decoder
                    .push(
                        &rtp::EncodedFrame {
                            timestamp,
                            data: Arc::from(*data),
                        },
                        |frame| append(frame, &mut actual),
                    )
                    .unwrap();
            }
            decoder
                .finish(180_000, |frame| append(frame, &mut actual))
                .unwrap();
            // Then: Every sampled gray byte and frame count matches the same-version CLI.
            assert_eq!(actual.len(), reference.len(), "{name}: frame count");
            let different = actual
                .iter()
                .zip(&reference)
                .filter(|(a, b)| a != b)
                .count();
            assert_eq!(different, 0, "{name}: differing gray bytes");
        }
    }
}

#[test]
fn shared_demux_parameters_preserve_the_existing_prepared_pixel_contract() {
    // Given: The same frozen cadence/color corpus and codec metadata from libavformat.
    for (name, fps) in [
        ("15fps-bt709-tv.h264", 15),
        ("7fps-bt709-pc.h264", 7),
        ("15fps-smpte170m-tv.h264", 15),
        ("7fps-smpte170m-pc.h264", 7),
    ] {
        let path = fixture(name);
        let input = ffmpeg_next::format::input(&path).unwrap();
        let parameters = input.stream(0).unwrap().parameters();
        let mut decoder = GrayDecoder::from_parameters(parameters).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        let reference = ffmpeg_reference::prepare_gray(&path, fps, 20);
        let mut actual = Vec::new();
        // When: The shared decoder consumes the original compressed access units,
        // retaining source timestamps rather than depending on a zero camera epoch.
        for (index, data) in access_units(&bytes).iter().enumerate() {
            let timestamp = 37_000_000 + index as i64 * 90_000 / i64::from(fps);
            let mut packet = ffmpeg_next::Packet::copy(data);
            packet.set_pts(Some(timestamp));
            packet.set_dts(Some(timestamp));
            decoder
                .push_packet(&packet, ffmpeg_next::Rational(1, 90_000), |frame| {
                    append(frame, &mut actual)
                })
                .unwrap();
        }
        decoder
            .finish(180_000, |frame| append(frame, &mut actual))
            .unwrap();
        // Then: Count and every prepared grayscale byte match the existing CLI oracle.
        assert_eq!(actual, reference, "{name}: shared preparation differs");
    }
}

#[test]
fn timestamp_regression_refuses_without_emitting_media() {
    // Given: One accepted access unit establishes the RTP timestamp baseline.
    let bytes = std::fs::read(fixture("15fps-bt709-tv.h264")).unwrap();
    let units = access_units(&bytes);
    for next_timestamp in [10_000, 9_000] {
        let mut decoder = GrayDecoder::new().unwrap();
        decoder
            .push(
                &rtp::EncodedFrame {
                    timestamp: 10_000,
                    data: Arc::from(units[0]),
                },
                |_| {},
            )
            .unwrap();
        let mut emitted = 0;
        // When: The next access unit repeats or regresses the timestamp.
        let result = decoder.push(
            &rtp::EncodedFrame {
                timestamp: next_timestamp,
                data: Arc::from(units[1]),
            },
            |_| emitted += 1,
        );
        // Then: A stable refusal allows the runtime to fall back without using inconsistent motion samples.
        assert_eq!(result, Err("motion_timestamp_invalid"));
        assert_eq!(emitted, 0);
    }
}

#[test]
fn changing_decoded_parameters_refuses_with_stable_reason() {
    // Given: A decoder established with one source color range.
    let first = std::fs::read(fixture("15fps-bt709-tv.h264")).unwrap();
    let changed = std::fs::read(fixture("7fps-bt709-pc.h264")).unwrap();
    let mut decoder = GrayDecoder::new().unwrap();
    decoder
        .push(
            &rtp::EncodedFrame {
                timestamp: 0,
                data: Arc::from(access_units(&first)[0]),
            },
            |_| {},
        )
        .unwrap();
    // When: New parameter sets change the decoded frame preparation contract.
    let result = decoder.push(
        &rtp::EncodedFrame {
            timestamp: 9_000,
            data: Arc::from(access_units(&changed)[0]),
        },
        |_| {},
    );
    // Then: The old graph is not reused for incompatible frames.
    assert_eq!(result, Err("motion_parameters_changed"));
}

#[test]
fn timestamp_gap_cannot_generate_unbounded_duplicate_frames() {
    // Given: A valid frame starts a ten-fps preparation graph.
    let bytes = std::fs::read(fixture("15fps-bt709-tv.h264")).unwrap();
    let units = access_units(&bytes);
    let mut decoder = GrayDecoder::new().unwrap();
    decoder
        .push(
            &rtp::EncodedFrame {
                timestamp: 0,
                data: Arc::from(units[0]),
            },
            |_| {},
        )
        .unwrap();
    let mut emitted = 0;
    // When: A source discontinuity would duplicate many seconds of stale video.
    let result = decoder.push(
        &rtp::EncodedFrame {
            timestamp: 900_000,
            data: Arc::from(units[1]),
        },
        |_| emitted += 1,
    );
    // Then: Work and emitted buffers stop at the documented per-push limit.
    assert_eq!(result, Err("motion_prepare_overflow"));
    assert_eq!(emitted, 32);
}

#[test]
fn empty_access_unit_is_refused_before_decoder_allocation() {
    // Given: A native decoder with no established source frame.
    let mut decoder = GrayDecoder::new().unwrap();
    // When: An empty encoded access unit is submitted.
    let result = decoder.push(
        &rtp::EncodedFrame {
            timestamp: 0,
            data: Arc::from([]),
        },
        |_| panic!("empty frame emitted media"),
    );
    // Then: Input validation returns a stable reason before media processing.
    assert_eq!(result, Err("motion_frame_invalid"));
}
