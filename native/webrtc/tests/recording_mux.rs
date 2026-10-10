//! Real offline compressed media exercises the native muxer without a camera.

#[path = "../src/recording.rs"]
mod recording;

use ffmpeg_next as ffmpeg;
use h264_reader::nal::{Nal, RefNal};
use h264_reader::rbsp::BitRead;
use recording::{EncodedPacket, Mp4Recorder, StreamConfig, Track, WriteOutcome};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::Command;

struct Packet {
    track: Track,
    data: Vec<u8>,
    pts: i64,
    dts: i64,
    duration: i64,
    keyframe: bool,
}

impl Packet {
    fn borrowed(&self) -> EncodedPacket<'_> {
        EncodedPacket {
            track: self.track,
            data: &self.data,
            pts: self.pts,
            dts: self.dts,
            duration: self.duration,
            keyframe: self.keyframe,
        }
    }
}

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/recording/h264-aac-bframes.mp4")
}

fn media(path: &Path) -> (StreamConfig, StreamConfig, Vec<Packet>) {
    ffmpeg::init().unwrap();
    ffmpeg::util::log::set_level(ffmpeg::util::log::Level::Quiet);
    let mut input = ffmpeg::format::input(path).unwrap();
    let video = input.stream(0).unwrap();
    let video = StreamConfig::copy(video.parameters(), video.time_base()).unwrap();
    let audio = input.stream(1).unwrap();
    let audio = StreamConfig::copy(audio.parameters(), audio.time_base()).unwrap();
    let packets = input
        .packets()
        .map(|(stream, packet)| Packet {
            track: if stream.index() == 0 {
                Track::Video
            } else {
                Track::Audio
            },
            data: packet.data().unwrap().to_vec(),
            pts: packet.pts().unwrap(),
            dts: packet.dts().unwrap(),
            duration: packet.duration(),
            keyframe: packet.is_key(),
        })
        .collect();
    (video, audio, packets)
}

fn decode(path: &Path, audio: bool) -> Vec<u8> {
    let mut command = Command::new("ffmpeg");
    command.args([
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-xerror",
        "-threads",
        "1",
        "-filter_threads",
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
            "-threads",
            "1",
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
        "synthetic clip decode failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    output.stdout
}

fn assert_decode_matches(actual: &Path, expected: &Path, audio: bool) {
    let actual = decode(actual, audio);
    let expected = decode(expected, audio);
    assert_eq!(actual.len(), expected.len(), "decoded sample count");
    assert!(actual == expected, "decoded samples differ");
}

#[test]
fn copied_audio_video_and_b_frame_timing_match_the_input() {
    // Given: Synthetic H.264 B-frames and AAC share one source epoch, including negative DTS.
    let (video, audio, packets) = media(&fixture());
    assert!(packets.iter().any(|packet| packet.pts != packet.dts));
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("clip.mp4");
    let mut recorder = Mp4Recorder::create(&path, video, Some(audio)).unwrap();
    // When: Native libavformat copies compressed packets and explicitly finalizes the clip.
    for packet in &packets {
        assert_eq!(recorder.push(packet.borrowed()), Ok(WriteOutcome::Written));
    }
    recorder.finish().unwrap();
    let (_, _, copied) = media(&path);
    // Then: Encoded payloads, audio/video timestamps, and decoded samples remain unchanged.
    assert_eq!(copied.len(), packets.len());
    for (original, copied) in packets.iter().zip(&copied) {
        assert_eq!(original.track, copied.track);
        assert_eq!(original.data, copied.data);
        assert_eq!(original.pts, copied.pts);
        assert_eq!(original.dts, copied.dts);
        assert_eq!(original.duration, copied.duration);
    }
    assert_decode_matches(&path, &fixture(), false);
    assert_decode_matches(&path, &fixture(), true);
}

#[test]
fn video_only_clips_finalize_independently_for_rotation() {
    // Given: Recording policy chooses video-only and owns two independent output paths.
    let directory = tempfile::tempdir().unwrap();
    let expected = decode(&fixture(), false);
    for name in ["first.mp4", "second.mp4"] {
        let (video, _, packets) = media(&fixture());
        let path = directory.path().join(name);
        let mut recorder = Mp4Recorder::create(&path, video, None).unwrap();
        // When: Each recording receives a keyframe-led sequence and its own finalization.
        for packet in packets.iter().filter(|packet| packet.track == Track::Video) {
            recorder.push(packet.borrowed()).unwrap();
        }
        recorder.finish().unwrap();
        // Then: Every completed clip is playable and contains only the selected video track.
        let input = ffmpeg::format::input(&path).unwrap();
        assert_eq!(input.nb_streams(), 1);
        let actual = decode(&path, false);
        assert_eq!(actual.len(), expected.len());
        assert!(actual == expected, "decoded samples differ");
    }
}

#[test]
fn audio_and_interframes_wait_until_the_first_video_keyframe() {
    // Given: Audio and a dependent video frame arrive before the recorder's first keyframe.
    let (video, audio, packets) = media(&fixture());
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("clip.mp4");
    let mut recorder = Mp4Recorder::create(&path, video, Some(audio)).unwrap();
    let audio = packets
        .iter()
        .find(|packet| packet.track == Track::Audio)
        .unwrap();
    let interframe = packets
        .iter()
        .find(|packet| packet.track == Track::Video && !packet.keyframe)
        .unwrap();
    // When: They are followed by the complete keyframe-led source sequence.
    assert_eq!(
        recorder.push(audio.borrowed()),
        Ok(WriteOutcome::WaitingForKeyframe)
    );
    assert_eq!(
        recorder.push(interframe.borrowed()),
        Ok(WriteOutcome::WaitingForKeyframe)
    );
    for packet in &packets {
        recorder.push(packet.borrowed()).unwrap();
    }
    recorder.finish().unwrap();
    // Then: No dependent prefix or audio-only clip is emitted; the completed clip decodes normally.
    assert_decode_matches(&path, &fixture(), false);
    assert_decode_matches(&path, &fixture(), true);
}

#[test]
fn regressing_decode_timestamps_fail_the_clip_with_a_stable_reason() {
    // Given: A recording has accepted its initial video keyframe.
    let (video, _, packets) = media(&fixture());
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("clip.mp4");
    let mut recorder = Mp4Recorder::create(&path, video, None).unwrap();
    let first = packets
        .iter()
        .find(|packet| packet.track == Track::Video)
        .unwrap();
    recorder.push(first.borrowed()).unwrap();
    // When: The source repeats a decode timestamp and then attempts to keep recording.
    assert_eq!(
        recorder.push(first.borrowed()),
        Err("recording_timestamp_invalid")
    );
    assert_eq!(
        recorder.push(first.borrowed()),
        Err("recording_timestamp_invalid")
    );
    // Then: Finalization cannot report the failed clip as complete.
    assert_eq!(recorder.finish(), Err("recording_timestamp_invalid"));
}

#[test]
fn existing_clip_is_never_truncated() {
    // Given: A clip already exists at the owner-selected destination.
    let (video, _, _) = media(&fixture());
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("clip.mp4");
    std::fs::write(&path, b"already recorded").unwrap();
    // When: A new recording attempts to use the same path.
    let result = Mp4Recorder::create(&path, video, None);
    // Then: Opening fails explicitly and the existing clip remains intact.
    assert!(matches!(result, Err("recording_open_failed")));
    assert_eq!(std::fs::read(path).unwrap(), b"already recorded");
}

#[test]
fn dropping_without_finalization_does_not_produce_a_completed_clip() {
    // Given: A conventional MP4 recording has begun writing compressed media.
    let (video, _, packets) = media(&fixture());
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("clip.mp4");
    let mut recorder = Mp4Recorder::create(&path, video, None).unwrap();
    recorder.push(packets[0].borrowed()).unwrap();
    // When: The owner cancels or dies without running finalization.
    drop(recorder);
    // Then: This partial file cannot be treated as a playable completed recording.
    assert!(ffmpeg::format::input(&path).is_err());
}

#[test]
fn ending_before_a_keyframe_reports_no_completed_media() {
    // Given: A source has not yet supplied an independently decodable video start.
    let (video, _, _) = media(&fixture());
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("clip.mp4");
    let recorder = Mp4Recorder::create(&path, video, None).unwrap();
    // When: Recording policy stops it before the first keyframe.
    let result = recorder.finish();
    // Then: An empty/header-only file is refused as a completed clip.
    assert_eq!(result, Err("recording_no_keyframe"));
}

#[test]
fn unsupported_audio_requires_compatibility_recording() {
    // Given: A valid audio stream uses a codec outside native copy eligibility.
    ffmpeg::init().unwrap();
    let input = ffmpeg::format::input(&fixture()).unwrap();
    let audio = input.stream(1).unwrap();
    let mut parameters = audio.parameters();
    parameters.set_id(ffmpeg::codec::Id::PCM_MULAW);
    // When: The owner checks whether it can select native recording.
    let result = StreamConfig::copy(parameters, audio.time_base());
    // Then: A typed stable refusal retains audio through the compatibility path.
    assert!(matches!(result, Err("unsupported_recording_codec")));
}

#[test]
fn truncated_codec_configuration_is_refused_before_opening_a_clip() {
    // Given: A discovered H.264/AAC stream contains truncated out-of-band metadata.
    ffmpeg::init().unwrap();
    let input = ffmpeg::format::input(&fixture()).unwrap();
    for index in [0, 1] {
        let stream = input.stream(index).unwrap();
        let mut parameters = stream.parameters().clone();
        // SAFETY: this only shortens an owned fixture metadata allocation.
        unsafe { (*parameters.as_mut_ptr()).extradata_size = 1 };
        // When: Native recording eligibility examines the codec configuration.
        let result = StreamConfig::copy(parameters, stream.time_base());
        // Then: A successful MP4 trailer cannot conceal an empty/unusable codec configuration.
        assert!(matches!(result, Err("recording_parameters_invalid")));
    }
}

#[test]
fn unknown_video_packet_duration_cannot_report_a_complete_clip() {
    // Given: The source has timestamps but no duration for its final video sample.
    let (video, audio, packets) = media(&fixture());
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("clip.mp4");
    let mut recorder = Mp4Recorder::create(&path, video, Some(audio)).unwrap();
    let first = packets
        .iter()
        .find(|packet| packet.track == Track::Video)
        .unwrap();
    // When: The recording receives an unknown sample duration instead of guessing its end time.
    let mut packet = first.borrowed();
    packet.duration = 0;
    let outcome = recorder.push(packet);
    // Then: Explicit refusal prevents handing off an MP4 that hides its final decoded frame.
    assert_eq!(outcome, Err("recording_packet_invalid"));
    assert_eq!(recorder.finish(), Err("recording_packet_invalid"));
}

#[test]
fn annex_b_parameter_sets_and_packets_are_copied_into_playable_mp4() {
    // Given: Native RTSP supplies Annex B parameter sets and complete access units.
    ffmpeg::init().unwrap();
    let fixture =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/decode/15fps-bt709-tv.h264");
    let mut input = ffmpeg::format::input(&fixture).unwrap();
    let stream = input.stream(0).unwrap();
    let video = StreamConfig::copy(stream.parameters(), (1, 90_000).into()).unwrap();
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("clip.mp4");
    let mut recorder = Mp4Recorder::create(&path, video, None).unwrap();
    // When: The native muxer receives the selected source's Annex B frames at their real cadence.
    for (index, (_, packet)) in input.packets().enumerate() {
        let timestamp = index as i64 * 6000;
        recorder
            .push(EncodedPacket {
                track: Track::Video,
                data: packet.data().unwrap(),
                pts: timestamp,
                dts: timestamp,
                duration: 6000,
                keyframe: packet.is_key(),
            })
            .unwrap();
    }
    recorder.finish().unwrap();
    // Then: Conversion to MP4's length-prefixed samples preserves all decoded video bytes.
    assert_decode_matches(&path, &fixture, false);
}

#[test]
fn in_band_parameter_changes_refuse_instead_of_muxing_an_old_header() {
    // Given: A recording's H.264 parameters are fixed by its initial codec configuration.
    let (video, _, packets) = media(&fixture());
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("clip.mp4");
    let mut recorder = Mp4Recorder::create(&path, video, None).unwrap();
    let first = packets
        .iter()
        .find(|packet| packet.track == Track::Video)
        .unwrap();
    // When: A subsequent source packet advertises a new sequence parameter set.
    let mut changed = vec![0, 0, 0, 2, 0x67, 0x80];
    changed.extend_from_slice(&first.data);
    let result = recorder.push(EncodedPacket {
        data: &changed,
        ..first.borrowed()
    });
    // Then: The owner gets a stable failure and cannot hand off incompatible media as complete.
    assert_eq!(result, Err("recording_parameters_changed"));
    assert_eq!(recorder.finish(), Err("recording_parameters_changed"));
}

fn annex_nals(data: &[u8]) -> Vec<Vec<u8>> {
    let mut nals = Vec::new();
    let mut reader = h264_reader::annexb::AnnexBReader::accumulate(|nal: RefNal<'_>| {
        if !nal.is_complete() {
            return h264_reader::push::NalInterest::Buffer;
        }
        let mut bytes = Vec::new();
        nal.reader().read_to_end(&mut bytes).unwrap();
        nals.push(bytes);
        h264_reader::push::NalInterest::Ignore
    });
    reader.push(data);
    reader.reset();
    drop(reader);
    nals
}

#[test]
fn multislice_idr_missing_its_primary_slice_cannot_finalize_a_clip() {
    // Given: A real synthetic IDR uses several slices and the RTP stream drops only its primary one.
    ffmpeg::init().unwrap();
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/baseline-multislice-160x120.h264");
    let mut input = ffmpeg::format::input(&path).unwrap();
    let stream = input.stream(0).unwrap();
    let video = StreamConfig::copy(stream.parameters(), (1, 90_000).into()).unwrap();
    let (_, packet) = input.packets().next().unwrap();
    assert!(packet.is_key());
    let nals = annex_nals(packet.data().unwrap());
    let mut missing_primary = Vec::new();
    let mut removed = 0;
    let mut retained_idr = 0;
    for bytes in nals {
        if bytes[0] & 31 == 5 {
            let nal = RefNal::new(&bytes, &[], true);
            let first_mb = nal.rbsp_bits().read_ue("first_mb_in_slice").unwrap();
            if first_mb == 0 {
                removed += 1;
                continue;
            }
            retained_idr += 1;
        }
        missing_primary.extend_from_slice(&[0, 0, 0, 1]);
        missing_primary.extend_from_slice(&bytes);
    }
    assert_eq!(removed, 1);
    assert!(retained_idr > 0);
    let directory = tempfile::tempdir().unwrap();
    let mut recorder =
        Mp4Recorder::create(&directory.path().join("clip.mp4"), video, None).unwrap();
    // When: The demuxer still advertises KEY but the access unit lacks macroblock zero.
    let outcome = recorder.push(EncodedPacket {
        track: Track::Video,
        data: &missing_primary,
        pts: 0,
        dts: 0,
        duration: 9000,
        keyframe: true,
    });
    // Then: This incomplete IDR is a sticky failure, so publication cannot claim a complete clip.
    assert_eq!(outcome, Err("recording_packet_invalid"));
    assert_eq!(recorder.finish(), Err("recording_packet_invalid"));
}

#[test]
fn malformed_idr_first_macroblock_fields_fail_without_finalized_media() {
    // Given: Truncated and overflowing unsigned Exp-Golomb first-macroblock fields.
    for bytes in [&[0x65][..], &[0x65, 0][..], &[0x65, 0, 0, 0, 0, 0x80][..]] {
        let (video, _, _) = media(&fixture());
        let directory = tempfile::tempdir().unwrap();
        let mut recorder =
            Mp4Recorder::create(&directory.path().join("clip.mp4"), video, None).unwrap();
        let mut data = (bytes.len() as u32).to_be_bytes().to_vec();
        data.extend_from_slice(bytes);
        // When: The bounded slice-header reader attempts the first field of an advertised keyframe.
        let outcome = recorder.push(EncodedPacket {
            track: Track::Video,
            data: &data,
            pts: 0,
            dts: 0,
            duration: 1024,
            keyframe: true,
        });
        // Then: Every malformed header refuses completion; no decoder or timing guess is involved.
        assert_eq!(outcome, Err("recording_packet_invalid"));
        assert_eq!(recorder.finish(), Err("recording_packet_invalid"));
    }
}
