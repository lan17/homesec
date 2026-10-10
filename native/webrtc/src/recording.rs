//! Compressed H.264/AAC recording through the bundled libavformat MP4 muxer.
//!
//! The source owns codec discovery, RTP clock mapping, reconnects, and rotation.
//! All packet timestamps supplied here refer to one clip epoch. In particular,
//! callers must preserve distinct presentation/decode timestamps for B-frames.
//! Codec parameters are immutable for a clip; source parameter changes require
//! finalizing it before opening another. No decoding or transcoding occurs here.

use ffmpeg::{Rational, codec, format, media};
use ffmpeg_next as ffmpeg;
use h264_reader::nal::{Nal, RefNal, UnitType, pps::PicParameterSet, sps::SeqParameterSet};
use h264_reader::rbsp::BitRead;
use std::fs::{File, OpenOptions};
use std::io::{Seek, Write};
use std::path::Path;

const MAX_PACKET_BYTES: usize = 2 * 1024 * 1024;
const MAX_EXTRA_BYTES: i32 = 64 * 1024;
const IO_BUFFER_BYTES: usize = 32 * 1024;
type Result<T> = std::result::Result<T, &'static str>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Track {
    Video,
    Audio,
}

/// Metadata and packet framing must come from the same source. H.264 AVCC
/// metadata describes length-prefixed packets; Annex B metadata describes
/// Annex B packets. The muxer converts the latter without decoding the video.
pub struct StreamConfig {
    parameters: codec::Parameters,
    time_base: Rational,
    track: Track,
    h264: Option<H264Config>,
}

impl StreamConfig {
    pub fn copy(parameters: codec::Parameters, time_base: Rational) -> Result<Self> {
        if time_base.numerator() <= 0 || time_base.denominator() <= 0 {
            return Err("recording_parameters_invalid");
        }
        // SAFETY: Parameters owns its allocated AVCodecParameters. This borrow
        // cannot outlive it and no pointer is retained or mutated here.
        let raw = unsafe { &*parameters.as_ptr() };
        if raw.extradata_size <= 0
            || raw.extradata_size > MAX_EXTRA_BYTES
            || raw.extradata.is_null()
        {
            return Err("recording_parameters_invalid");
        }
        let track = match (parameters.medium(), parameters.id()) {
            (media::Type::Video, codec::Id::H264)
                if raw.width > 0 && raw.width <= 8192 && raw.height > 0 && raw.height <= 8192 =>
            {
                Track::Video
            }
            (media::Type::Audio, codec::Id::AAC)
                if raw.sample_rate > 0
                    && raw.sample_rate <= 192_000
                    && raw.ch_layout.nb_channels > 0
                    && raw.ch_layout.nb_channels <= 8 =>
            {
                Track::Audio
            }
            _ => return Err("unsupported_recording_codec"),
        };
        // SAFETY: the positive bounded size and non-null pointer were checked
        // above. Parameters owns this allocation for the entire borrowed slice.
        let extra =
            unsafe { std::slice::from_raw_parts(raw.extradata, raw.extradata_size as usize) };
        let h264 = match track {
            Track::Video => Some(validate_h264(extra, raw.width as u32, raw.height as u32)?),
            Track::Audio => {
                validate_aac(extra, raw.sample_rate, raw.ch_layout.nb_channels)?;
                None
            }
        };
        Ok(Self {
            parameters,
            time_base,
            track,
            h264,
        })
    }
}

/// RTSP SDP parameter sets can be stale even when the negotiated codec is
/// correct. Before publishing source metadata, prefer the complete first IDR's
/// in-band Annex B SPS/PPS. This changes only initial discovery; the recorder
/// still refuses every parameter change after its immutable header is written.
#[cfg_attr(test, allow(dead_code))]
pub fn initial_video_parameters(
    mut parameters: codec::Parameters,
    time_base: Rational,
    packet: &ffmpeg::Packet,
) -> Result<codec::Parameters> {
    let config = StreamConfig::copy(parameters.clone(), time_base)?;
    let h264 = config.h264.ok_or("recording_parameters_invalid")?;
    let data = packet.data().ok_or("recording_packet_invalid")?;
    if data.is_empty() || data.len() > MAX_PACKET_BYTES || !packet.is_key() {
        return Err("recording_packet_invalid");
    }
    // Container-demuxed AVCC packets retain their existing framing/header.
    // The startup correction applies to RTSP's Annex B access units only.
    if h264.length_bytes.is_some() {
        h264.validate_packet(data, true)?;
        return Ok(parameters);
    }
    let nals = annex_b_nals(data)?;
    validate_idr_slices(&nals, true)?;
    let sets = nals
        .iter()
        .filter(|nal| matches!(nal[0] & 31, 7 | 8))
        .collect::<Vec<_>>();
    if sets.is_empty() {
        return Ok(parameters);
    }
    if !sets.iter().any(|nal| nal[0] & 31 == 7) || !sets.iter().any(|nal| nal[0] & 31 == 8) {
        return Err("recording_parameters_invalid");
    }
    let length = sets.iter().try_fold(0_usize, |length, nal| {
        length
            .checked_add(4 + nal.len())
            .ok_or("recording_parameters_invalid")
    })?;
    if length > MAX_EXTRA_BYTES as usize {
        return Err("recording_parameters_invalid");
    }
    let mut extra = Vec::with_capacity(length);
    for nal in sets {
        extra.extend_from_slice(&[0, 0, 0, 1]);
        extra.extend_from_slice(nal);
    }
    // SAFETY: Parameters owns its AVCodecParameters; FFmpeg owns both the old
    // allocation and the zero-padded replacement. No borrowed pointer escapes.
    unsafe {
        let raw = &mut *parameters.as_mut_ptr();
        let allocation =
            ffmpeg::ffi::av_mallocz(length + ffmpeg::ffi::AV_INPUT_BUFFER_PADDING_SIZE as usize)
                .cast::<u8>();
        if allocation.is_null() {
            return Err("recording_parameters_invalid");
        }
        std::ptr::copy_nonoverlapping(extra.as_ptr(), allocation, length);
        ffmpeg::ffi::av_free(raw.extradata.cast());
        raw.extradata = allocation;
        raw.extradata_size = length as i32;
    }
    StreamConfig::copy(parameters.clone(), time_base)?;
    Ok(parameters)
}

pub struct EncodedPacket<'a> {
    pub track: Track,
    pub data: &'a [u8],
    pub pts: i64,
    pub dts: i64,
    pub duration: i64,
    pub keyframe: bool,
}

#[derive(Debug, PartialEq, Eq)]
pub enum WriteOutcome {
    WaitingForKeyframe,
    Written,
}

struct Stream {
    source_time_base: Rational,
    output_time_base: Rational,
    previous_dts: Option<i64>,
    h264: Option<H264Config>,
}

pub struct Mp4Recorder {
    output: format::context::Output,
    streams: Vec<Stream>,
    file: Option<File>,
    started: bool,
    failure: Option<&'static str>,
}

impl Mp4Recorder {
    pub fn create(path: &Path, video: StreamConfig, audio: Option<StreamConfig>) -> Result<Self> {
        validate_tracks(&video, audio.as_ref())?;
        // Existing recordings must never be truncated. FFmpeg writes through
        // this already-open file rather than reopening an operator path.
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .map_err(|_| "recording_open_failed")?;
        let sync_file = file.try_clone().map_err(|_| "recording_open_failed")?;
        let mut recorder = Self::from_writer(file, video, audio)?;
        recorder.file = Some(sync_file);
        Ok(recorder)
    }

    fn from_writer(
        writer: impl Write + Seek + Send + 'static,
        video: StreamConfig,
        audio: Option<StreamConfig>,
    ) -> Result<Self> {
        let movie_timescale = validate_tracks(&video, audio.as_ref())?;
        ffmpeg::util::log::set_level(ffmpeg::util::log::Level::Quiet);
        ffmpeg::init().map_err(|_| "recording_muxer_unavailable")?;
        let io = format::context::StreamIo::from_write_seek_with_capacity(writer, IO_BUFFER_BYTES)
            .map_err(|_| "recording_open_failed")?;
        let mut output = format::output_to_stream(io, None, Some("mp4"))
            .map_err(|_| "recording_muxer_unavailable")?;
        let configs = std::iter::once(video).chain(audio);
        let mut streams = Vec::with_capacity(2);
        for config in configs {
            let mut stream = output
                .add_stream(codec::encoder::find(codec::Id::None))
                .map_err(|_| "recording_header_failed")?;
            stream.set_parameters(config.parameters);
            stream.set_time_base(config.time_base);
            // SAFETY: this is the newly created output stream's owned mutable
            // parameter object. Codec tags are container-specific; let MP4
            // choose its own rather than retaining the input container's tag.
            unsafe { (*stream.parameters().as_mut_ptr()).codec_tag = 0 };
            streams.push(Stream {
                source_time_base: config.time_base,
                output_time_base: config.time_base,
                previous_dts: None,
                h264: config.h264,
            });
        }
        // MP4 edit-list offsets use the movie clock, whose default 1000Hz
        // rounds AAC offsets by whole milliseconds. A common exact denominator
        // preserves both track clocks, including nonzero camera A/V offsets.
        let mut options = ffmpeg::Dictionary::new();
        options.set("movie_timescale", &movie_timescale.to_string());
        output
            .write_header_with(options)
            .map_err(|_| "recording_header_failed")?;
        // The muxer may replace each requested time base while writing headers.
        for (index, stream) in streams.iter_mut().enumerate() {
            stream.output_time_base = output
                .stream(index)
                .ok_or("recording_header_failed")?
                .time_base();
        }
        Ok(Self {
            output,
            streams,
            file: None,
            started: false,
            failure: None,
        })
    }

    pub fn push(&mut self, input: EncodedPacket<'_>) -> Result<WriteOutcome> {
        if let Some(error) = self.failure {
            return Err(error);
        }
        let result = self.write_packet(input);
        if let Err(error) = result {
            self.failure = Some(error);
        }
        result
    }

    fn write_packet(&mut self, input: EncodedPacket<'_>) -> Result<WriteOutcome> {
        let index = match input.track {
            Track::Video => 0,
            Track::Audio => 1,
        };
        let stream = self
            .streams
            .get_mut(index)
            .ok_or("recording_packet_invalid")?;
        if input.data.is_empty()
            || input.data.len() > MAX_PACKET_BYTES
            || input.duration <= 0
            || input.pts == i64::MIN
            || input.dts == i64::MIN
            || input.pts < input.dts
        {
            return Err("recording_packet_invalid");
        }
        if let Some(h264) = &stream.h264 {
            h264.validate_packet(input.data, input.keyframe)?;
        }
        if !self.started {
            if input.track != Track::Video || !input.keyframe {
                return Ok(WriteOutcome::WaitingForKeyframe);
            }
            self.started = true;
        }
        let mut packet = ffmpeg::Packet::copy(input.data);
        packet.set_pts(Some(input.pts));
        packet.set_dts(Some(input.dts));
        packet.set_duration(input.duration);
        packet.rescale_ts(stream.source_time_base, stream.output_time_base);
        let pts = packet.pts().ok_or("recording_timestamp_invalid")?;
        let dts = packet.dts().ok_or("recording_timestamp_invalid")?;
        if pts < dts
            || packet.duration() < 0
            || (input.duration > 0 && packet.duration() == 0)
            || stream.previous_dts.is_some_and(|previous| dts <= previous)
        {
            return Err("recording_timestamp_invalid");
        }
        packet.set_position(-1);
        packet.set_stream(index);
        if input.keyframe {
            packet.set_flags(codec::packet::Flags::KEY);
        }
        // MP4 accepts packets in the source's decode order. av_write_frame
        // avoids libavformat's additional interleaving queue: consumers cannot
        // accumulate waiting packets when the other track stalls.
        packet
            .write(&mut self.output)
            .map_err(|_| "recording_write_failed")?;
        stream.previous_dts = Some(dts);
        self.flush()?;
        Ok(WriteOutcome::Written)
    }

    fn flush(&mut self) -> Result<()> {
        // SAFETY: output owns a live custom AVIOContext for its whole lifetime.
        // Explicit flushing exposes errors otherwise lost by destructor cleanup.
        unsafe {
            let io = (*self.output.as_mut_ptr()).pb;
            ffmpeg::ffi::avio_flush(io);
            if (*io).error < 0 {
                return Err("recording_write_failed");
            }
        }
        Ok(())
    }

    /// Consume the writer to make finalization explicit and impossible twice.
    /// An owner must only deliver a clip after this returns success. Drop closes
    /// resources but cannot report errors or promise a finalized/playable file.
    pub fn finish(mut self) -> Result<()> {
        let trailer = self
            .output
            .write_trailer()
            .map_err(|_| "recording_finalize_failed");
        let flush = self.flush();
        let sync = self
            .file
            .as_ref()
            .map_or(Ok(()), |file| file.sync_all())
            .map_err(|_| "recording_finalize_failed");
        if let Some(error) = self.failure {
            return Err(error);
        }
        trailer?;
        flush?;
        sync?;
        if !self.started {
            return Err("recording_no_keyframe");
        }
        Ok(())
    }
}

fn validate_tracks(video: &StreamConfig, audio: Option<&StreamConfig>) -> Result<i32> {
    if video.track != Track::Video || audio.is_some_and(|audio| audio.track != Track::Audio) {
        return Err("recording_parameters_invalid");
    }
    let video_clock = i64::from(video.time_base.denominator());
    let Some(audio) = audio else {
        return Ok(video.time_base.denominator());
    };
    let audio_clock = i64::from(audio.time_base.denominator());
    let (mut left, mut right) = (video_clock, audio_clock);
    while right != 0 {
        (left, right) = (right, left % right);
    }
    let common = (video_clock / left)
        .checked_mul(audio_clock)
        .ok_or("recording_parameters_invalid")?;
    i32::try_from(common).map_err(|_| "recording_parameters_invalid")
}

struct H264Config {
    length_bytes: Option<usize>,
    parameter_sets: Vec<Vec<u8>>,
}

impl H264Config {
    fn validate_packet(&self, data: &[u8], keyframe: bool) -> Result<()> {
        let nals = match self.length_bytes {
            Some(length_bytes) => length_prefixed_nals(data, length_bytes)?,
            None => annex_b_nals(data)?,
        };
        validate_idr_slices(&nals, keyframe)?;
        for nal in nals {
            match nal[0] & 31 {
                7 | 8 if !self.parameter_sets.iter().any(|original| original == nal) => {
                    return Err("recording_parameters_changed");
                }
                _ => {}
            }
        }
        Ok(())
    }
}

fn validate_idr_slices(nals: &[&[u8]], keyframe: bool) -> Result<()> {
    let mut idr = false;
    let mut primary_slice = false;
    for bytes in nals {
        if bytes[0] & 31 != 5 {
            continue;
        }
        idr = true;
        let nal = RefNal::new(bytes, &[], true);
        nal.header().map_err(|_| "recording_packet_invalid")?;
        let first_mb = nal
            .rbsp_bits()
            .read_ue("first_mb_in_slice")
            .map_err(|_| "recording_packet_invalid")?;
        // Search every slice: valid arbitrary slice order can place the
        // primary slice later. An RTP marker/key flag alone cannot prove it
        // arrived, even when packet sequence numbers remain contiguous.
        primary_slice |= first_mb == 0;
    }
    if (keyframe || idr) && !primary_slice {
        return Err("recording_packet_invalid");
    }
    Ok(())
}

fn length_prefixed_nals(mut data: &[u8], length_bytes: usize) -> Result<Vec<&[u8]>> {
    let mut nals = Vec::new();
    while !data.is_empty() {
        let size = data.get(..length_bytes).ok_or("recording_packet_invalid")?;
        let length = size
            .iter()
            .fold(0_usize, |length, byte| (length << 8) | usize::from(*byte));
        data = &data[length_bytes..];
        let nal = data.get(..length).ok_or("recording_packet_invalid")?;
        if nal.is_empty() || nals.len() >= 1024 {
            return Err("recording_packet_invalid");
        }
        nals.push(nal);
        data = &data[length..];
    }
    Ok(nals)
}

fn validate_h264(extra: &[u8], width: u32, height: u32) -> Result<H264Config> {
    let mut length_bytes = None;
    let nals = if extra.first() == Some(&1) {
        // Reuse the existing AVC parser for its fixed-field/length validation.
        h264_reader::avcc::AvcDecoderConfigurationRecord::try_from(extra)
            .map_err(|_| "recording_parameters_invalid")?;
        length_bytes = Some(usize::from(extra[4] & 3) + 1);
        let mut offset = 6;
        let mut nals = Vec::new();
        for group in 0..2 {
            let count = if group == 1 {
                let count = *extra.get(offset).ok_or("recording_parameters_invalid")?;
                offset += 1;
                count
            } else {
                extra[5] & 31
            };
            if count == 0 {
                return Err("recording_parameters_invalid");
            }
            for _ in 0..count {
                let bytes = extra
                    .get(offset..offset + 2)
                    .ok_or("recording_parameters_invalid")?;
                let length = u16::from_be_bytes([bytes[0], bytes[1]]) as usize;
                offset += 2;
                let nal = extra
                    .get(offset..offset + length)
                    .ok_or("recording_parameters_invalid")?;
                if nal.is_empty() {
                    return Err("recording_parameters_invalid");
                }
                nals.push(nal);
                offset += length;
            }
        }
        nals
    } else {
        annex_b_nals(extra)?
    };
    let mut context = h264_reader::Context::new();
    for bytes in &nals {
        let nal = RefNal::new(bytes, &[], true);
        if nal
            .header()
            .map_err(|_| "recording_parameters_invalid")?
            .nal_unit_type()
            == UnitType::SeqParameterSet
        {
            let sps = SeqParameterSet::from_bits(nal.rbsp_bits())
                .map_err(|_| "recording_parameters_invalid")?;
            if sps
                .pixel_dimensions()
                .map_err(|_| "recording_parameters_invalid")?
                != (width, height)
            {
                return Err("recording_parameters_invalid");
            }
            context.put_seq_param_set(sps);
        }
    }
    for bytes in &nals {
        let nal = RefNal::new(bytes, &[], true);
        match nal
            .header()
            .map_err(|_| "recording_parameters_invalid")?
            .nal_unit_type()
        {
            UnitType::SeqParameterSet => {}
            UnitType::PicParameterSet => {
                let pps = PicParameterSet::from_bits(&context, nal.rbsp_bits())
                    .map_err(|_| "recording_parameters_invalid")?;
                context.put_pic_param_set(pps);
            }
            _ => return Err("recording_parameters_invalid"),
        }
    }
    if context.sps().next().is_none() || context.pps().next().is_none() {
        return Err("recording_parameters_invalid");
    }
    Ok(H264Config {
        length_bytes,
        parameter_sets: nals.into_iter().map(<[u8]>::to_vec).collect(),
    })
}

fn annex_b_nals(mut data: &[u8]) -> Result<Vec<&[u8]>> {
    let mut nals = Vec::new();
    while !data.is_empty() {
        let prefix = if data.starts_with(&[0, 0, 0, 1]) {
            4
        } else if data.starts_with(&[0, 0, 1]) {
            3
        } else {
            return Err("recording_parameters_invalid");
        };
        data = &data[prefix..];
        let end = data
            .windows(3)
            .position(|bytes| bytes == [0, 0, 1])
            .unwrap_or(data.len());
        let mut nal = &data[..end];
        while nal.last() == Some(&0) {
            nal = &nal[..nal.len() - 1];
        }
        if nal.is_empty() || nals.len() >= 1024 {
            return Err("recording_parameters_invalid");
        }
        nals.push(nal);
        data = &data[end..];
    }
    Ok(nals)
}

fn validate_aac(extra: &[u8], sample_rate: i32, channels: i32) -> Result<()> {
    if extra.len() < 2 {
        return Err("recording_parameters_invalid");
    }
    // Native eligibility currently covers ordinary AAC-LC raw access units.
    // Extended/object-specific configurations retain compatibility recording.
    let object_type = extra[0] >> 3;
    let frequency = ((extra[0] & 7) << 1) | (extra[1] >> 7);
    let channel_config = (extra[1] >> 3) & 15;
    let rates = [
        96_000, 88_200, 64_000, 48_000, 44_100, 32_000, 24_000, 22_050, 16_000, 12_000, 11_025,
        8_000, 7_350,
    ];
    let counts = [0, 1, 2, 3, 4, 5, 6, 8];
    if object_type != 2 || extra[1] & 7 != 0 {
        return Err("unsupported_recording_codec");
    }
    if rates.get(frequency as usize) != Some(&sample_rate)
        || counts.get(channel_config as usize) != Some(&channels)
    {
        return Err("recording_parameters_invalid");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{self, Cursor, SeekFrom};

    struct FailedDisk(Cursor<Vec<u8>>);

    impl Write for FailedDisk {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            if self.0.position().saturating_add(bytes.len() as u64) > 128 {
                return Err(io::Error::from(io::ErrorKind::StorageFull));
            }
            self.0.write(bytes)
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    impl Seek for FailedDisk {
        fn seek(&mut self, position: SeekFrom) -> io::Result<u64> {
            self.0.seek(position)
        }
    }

    #[test]
    fn primary_idr_slice_can_follow_other_idr_slices() {
        // Given: Complete slice-header prefixes with first macroblocks 1 then 0.
        let slices = [&[0x65, 0x40][..], &[0x65, 0x80][..]];
        // When: Validating the IDR packet's primary slice across arbitrary slice order.
        let outcome = validate_idr_slices(&slices, true);
        // Then: The primary slice need not be the first NAL in the access unit.
        assert_eq!(outcome, Ok(()));
    }

    #[test]
    fn buffered_io_failure_cannot_report_a_completed_clip() {
        // Given: A disk fails at the output I/O boundary after buffered header creation.
        ffmpeg::init().unwrap();
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/recording/h264-aac-bframes.mp4");
        let mut input = format::input(&path).unwrap();
        let stream = input.stream(0).unwrap();
        let config = StreamConfig::copy(stream.parameters(), stream.time_base()).unwrap();
        let mut recorder =
            Mp4Recorder::from_writer(FailedDisk(Cursor::new(Vec::new())), config, None).unwrap();
        let (_, first) = input.packets().next().unwrap();
        // When: Writing the first keyframe explicitly flushes the native I/O buffer.
        let result = recorder.push(EncodedPacket {
            track: Track::Video,
            data: first.data().unwrap(),
            pts: first.pts().unwrap(),
            dts: first.dts().unwrap(),
            duration: first.duration(),
            keyframe: true,
        });
        // Then: The stable storage failure survives finalization instead of silently losing data.
        assert_eq!(result, Err("recording_write_failed"));
        assert_eq!(recorder.finish(), Err("recording_write_failed"));
    }
}
