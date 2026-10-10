//! Worker-local preview codecs. Source packet clocks stay intact across codecs;
//! transport, packet queues and consumer ownership belong to the shared runtime.
//!
//! Encoder choices are private adapters below. Changing an encoder does not
//! change the source, control protocol, or recording/motion consumers.

use ffmpeg::{ChannelLayout, Error, Rational, codec, color, filter, format, frame, picture};
use ffmpeg_next as ffmpeg;
use std::sync::Arc;

const VIDEO_CLOCK: Rational = Rational(1, 90_000);
const AUDIO_CLOCK: Rational = Rational(1, 48_000);
const MAX_PACKET_BYTES: usize = 2 * 1024 * 1024;
const MAX_AUDIO_PACKET_BYTES: usize = 4000;
const MAX_DECODED_FRAMES: usize = 16;
const MAX_OUTPUT_PACKETS: usize = 64;
const MAX_PIXELS: i64 = 8192 * 8192;
const MAX_AUDIO_SAMPLES: i64 = 8192;
const OPUS_SAMPLES: usize = 960;
type Result<T> = std::result::Result<T, &'static str>;

pub struct EncodedVideo {
    /// Presentation time in 90 kHz ticks, including the original source offset.
    pub timestamp: i64,
    pub keyframe: bool,
    /// Annex B, with SPS/PPS repeated before each IDR.
    pub data: Arc<[u8]>,
}

pub struct EncodedAudio {
    /// Presentation time in 48 kHz ticks, including encoder lookahead.
    pub timestamp: i64,
    pub data: Arc<[u8]>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct VideoShape {
    width: u32,
    height: u32,
    format: format::Pixel,
    aspect: Rational,
    color_space: color::Space,
    color_range: color::Range,
    color_primaries: color::Primaries,
    color_transfer: color::TransferCharacteristic,
}

struct VideoEncoding {
    shape: VideoShape,
    graph: filter::Graph,
    encoder: X264Encoder,
}

pub struct VideoTranscoder {
    decoder: codec::decoder::Video,
    source_clock: Rational,
    encoding: Option<VideoEncoding>,
    previous_dts: Option<i64>,
    previous_frame: Option<i64>,
}

impl VideoTranscoder {
    pub fn new(parameters: codec::Parameters, time_base: Rational) -> Result<Self> {
        valid_clock(time_base)?;
        if parameters.id() != codec::Id::H264 {
            return Err("unsupported_preview_codec");
        }
        initialize()?;
        let mut context = codec::Context::from_parameters(parameters)
            .map_err(|_| "preview_decoder_unavailable")?
            .decoder();
        context.set_packet_time_base(VIDEO_CLOCK);
        context.set_threading(codec::threading::Config {
            kind: codec::threading::Type::Slice,
            count: 1,
        });
        // SAFETY: This owned decoder context is not open or shared yet. Bound
        // allocations inside libavcodec before decoding untrusted dimensions.
        unsafe { (*context.as_mut_ptr()).max_pixels = MAX_PIXELS };
        let decoder = context.video().map_err(|_| "preview_decoder_unavailable")?;
        Ok(Self {
            decoder,
            source_clock: time_base,
            encoding: None,
            previous_dts: None,
            previous_frame: None,
        })
    }

    pub fn push(&mut self, input: &ffmpeg::Packet) -> Result<Vec<EncodedVideo>> {
        let packet = packet_on_clock(input, self.source_clock, VIDEO_CLOCK)?;
        let dts = packet.dts().ok_or("preview_timestamp_invalid")?;
        if self.previous_dts.is_some_and(|previous| dts <= previous) {
            return Err("preview_timestamp_invalid");
        }
        self.previous_dts = Some(dts);
        self.decoder
            .send_packet(&packet)
            .map_err(|_| "preview_decode_failed")?;
        let mut output = Vec::new();
        self.receive(&mut output)?;
        Ok(output)
    }

    fn receive(&mut self, output: &mut Vec<EncodedVideo>) -> Result<()> {
        let mut decoded = frame::Video::empty();
        for _ in 0..MAX_DECODED_FRAMES {
            match self.decoder.receive_frame(&mut decoded) {
                Ok(()) => {
                    let pts = decoded
                        .timestamp()
                        .or(decoded.pts())
                        .ok_or("preview_timestamp_invalid")?;
                    if self.previous_frame.is_some_and(|previous| pts <= previous) {
                        return Err("preview_timestamp_invalid");
                    }
                    self.previous_frame = Some(pts);
                    decoded.set_pts(Some(pts));
                    self.process(&decoded, output)?;
                }
                Err(error) if drained(error) => return Ok(()),
                Err(_) => return Err("preview_decode_failed"),
            }
        }
        Err("preview_decode_overflow")
    }

    fn process(&mut self, decoded: &frame::Video, output: &mut Vec<EncodedVideo>) -> Result<()> {
        let shape = VideoShape {
            width: decoded.width(),
            height: decoded.height(),
            format: decoded.format(),
            aspect: ratio_or(decoded.aspect_ratio(), Rational(1, 1)),
            color_space: decoded.color_space(),
            color_range: decoded.color_range(),
            color_primaries: decoded.color_primaries(),
            color_transfer: decoded.color_transfer_characteristic(),
        };
        if shape.width == 0
            || shape.height == 0
            || shape.width > 8192
            || shape.height > 8192
            || !shape.width.is_multiple_of(2)
            || !shape.height.is_multiple_of(2)
            || shape.format == format::Pixel::None
        {
            return Err("preview_shape_invalid");
        }
        match &self.encoding {
            Some(encoding) if encoding.shape != shape => {
                return Err("preview_parameters_changed");
            }
            None => {
                let frame_rate = self
                    .decoder
                    .frame_rate()
                    .filter(|rate| rate.numerator() > 0 && rate.denominator() > 0);
                self.encoding = Some(VideoEncoding::new(shape, frame_rate)?);
            }
            _ => {}
        }
        let encoding = self.encoding.as_mut().ok_or("preview_encode_failed")?;
        encoding
            .graph
            .get("in")
            .ok_or("preview_prepare_failed")?
            .source()
            .add(decoded)
            .map_err(|_| "preview_prepare_failed")?;
        let mut prepared = frame::Video::empty();
        for _ in 0..MAX_DECODED_FRAMES {
            match encoding
                .graph
                .get("out")
                .ok_or("preview_prepare_failed")?
                .sink()
                .frame(&mut prepared)
            {
                Ok(()) => encoding.encoder.push(&mut prepared, output)?,
                Err(error) if drained(error) => return Ok(()),
                Err(_) => return Err("preview_prepare_failed"),
            }
        }
        Err("preview_prepare_overflow")
    }

    #[cfg(test)]
    fn finish(&mut self) -> Result<Vec<EncodedVideo>> {
        let mut output = Vec::new();
        self.decoder
            .send_eof()
            .map_err(|_| "preview_decode_failed")?;
        self.receive(&mut output)?;
        if let Some(encoding) = &mut self.encoding {
            encoding
                .encoder
                .encoder
                .send_eof()
                .map_err(|_| "preview_encode_failed")?;
            encoding.encoder.drain(&mut output)?;
        }
        Ok(output)
    }
}

impl VideoEncoding {
    fn new(shape: VideoShape, rate: Option<Rational>) -> Result<Self> {
        let mut graph = filter::Graph::new();
        // SAFETY: This private graph is not configured or shared yet.
        unsafe { (*graph.as_mut_ptr()).nb_threads = 1 };
        let mut arguments = format!(
            "video_size={}x{}:pix_fmt={}:time_base=1/90000:pixel_aspect={}",
            shape.width,
            shape.height,
            ffmpeg::ffi::AVPixelFormat::from(shape.format) as i32,
            shape.aspect,
        );
        if shape.color_space != color::Space::Unspecified {
            arguments.push_str(&format!(
                ":colorspace={}",
                ffmpeg::ffi::AVColorSpace::from(shape.color_space) as i32
            ));
        }
        if shape.color_range != color::Range::Unspecified {
            arguments.push_str(&format!(
                ":range={}",
                ffmpeg::ffi::AVColorRange::from(shape.color_range) as i32
            ));
        }
        add_filter(&mut graph, "buffer", "in", &arguments)?;
        add_filter(&mut graph, "buffersink", "out", "")?;
        graph
            .output("in", 0)
            .and_then(|parser| parser.input("out", 0))
            .and_then(|parser| parser.parse("format=pix_fmts=yuv420p"))
            .map_err(|_| "preview_prepare_failed")?;
        graph.validate().map_err(|_| "preview_prepare_failed")?;
        let encoder = X264Encoder::new(shape, rate)?;
        Ok(Self {
            shape,
            graph,
            encoder,
        })
    }
}

/// Keep all x264-specific settings and IDR policy in this private adapter.
struct X264Encoder {
    encoder: codec::encoder::video::Encoder,
    next_idr: Option<i64>,
    previous_output: Option<i64>,
}

impl X264Encoder {
    fn new(shape: VideoShape, rate: Option<Rational>) -> Result<Self> {
        let implementation =
            ffmpeg::encoder::find_by_name("libx264").ok_or("preview_encoder_unavailable")?;
        let mut encoder = codec::Context::new_with_codec(implementation)
            .encoder()
            .video()
            .map_err(|_| "preview_encoder_unavailable")?;
        encoder.set_width(shape.width);
        encoder.set_height(shape.height);
        encoder.set_format(format::Pixel::YUV420P);
        encoder.set_aspect_ratio(shape.aspect);
        encoder.set_colorspace(shape.color_space);
        encoder.set_color_range(shape.color_range);
        encoder.set_color_primaries(shape.color_primaries);
        encoder.set_color_transfer_characteristic(shape.color_transfer);
        encoder.set_time_base(VIDEO_CLOCK);
        // Some camera SPS omit timing information. A nominal rate keeps x264
        // from treating the 90kHz timestamp clock as the frame rate; actual
        // frame spacing and IDR decisions still use the source presentation PTS.
        encoder.set_frame_rate(Some(rate.unwrap_or(Rational(25, 1))));
        encoder.set_max_b_frames(0);
        encoder.set_threading(codec::threading::Config::count(1));
        let mut options = ffmpeg::Dictionary::new();
        options.set("preset", "veryfast");
        options.set("tune", "zerolatency");
        options.set("profile", "baseline");
        options.set("sc_threshold", "0");
        options.set("forced-idr", "1");
        options.set("x264-params", "repeat-headers=1:annexb=1");
        let encoder = encoder
            .open_as_with(implementation, options)
            .map_err(|_| "preview_encoder_unavailable")?;
        Ok(Self {
            encoder,
            next_idr: None,
            previous_output: None,
        })
    }

    fn push(&mut self, prepared: &mut frame::Video, output: &mut Vec<EncodedVideo>) -> Result<()> {
        let pts = prepared.pts().ok_or("preview_timestamp_invalid")?;
        if self.next_idr.is_none_or(|deadline| pts >= deadline) {
            prepared.set_kind(picture::Type::I);
            self.next_idr = Some(pts.checked_add(90_000).ok_or("preview_timestamp_invalid")?);
        } else {
            prepared.set_kind(picture::Type::None);
        }
        self.encoder
            .send_frame(prepared)
            .map_err(|_| "preview_encode_failed")?;
        self.drain(output)
    }

    fn drain(&mut self, output: &mut Vec<EncodedVideo>) -> Result<()> {
        let mut packet = ffmpeg::Packet::empty();
        loop {
            match self.encoder.receive_packet(&mut packet) {
                Ok(()) => {
                    if output.len() >= MAX_OUTPUT_PACKETS || packet.size() > MAX_PACKET_BYTES {
                        return Err("preview_encode_overflow");
                    }
                    let timestamp = packet.pts().ok_or("preview_timestamp_invalid")?;
                    if self
                        .previous_output
                        .is_some_and(|previous| timestamp <= previous)
                    {
                        return Err("preview_timestamp_invalid");
                    }
                    self.previous_output = Some(timestamp);
                    let data = packet
                        .data()
                        .filter(|data| !data.is_empty())
                        .ok_or("preview_encode_failed")?;
                    output.push(EncodedVideo {
                        timestamp,
                        keyframe: packet.is_key(),
                        data: Arc::from(data),
                    });
                }
                Err(error) if drained(error) => return Ok(()),
                Err(_) => return Err("preview_encode_failed"),
            }
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct AudioShape {
    rate: u32,
    format: format::Sample,
    layout: ChannelLayout,
}

struct AudioPreparation {
    shape: AudioShape,
    graph: filter::Graph,
}

pub struct AudioTranscoder {
    decoder: codec::decoder::Audio,
    source_clock: Rational,
    preparation: Option<AudioPreparation>,
    encoder: OpusEncoder,
    previous_frame: Option<i64>,
}

impl AudioTranscoder {
    pub fn new(parameters: codec::Parameters, time_base: Rational) -> Result<Self> {
        valid_clock(time_base)?;
        if !matches!(
            parameters.id(),
            codec::Id::AAC | codec::Id::PCM_ALAW | codec::Id::PCM_MULAW | codec::Id::OPUS
        ) {
            return Err("unsupported_preview_audio_codec");
        }
        // SAFETY: Read scalar metadata from the owned parameters before opening
        // a codec. Channel count and rate limit decoder/filter allocation.
        let raw = unsafe { &*parameters.as_ptr() };
        if !(8000..=192000).contains(&raw.sample_rate)
            || !(1..=8).contains(&raw.ch_layout.nb_channels)
        {
            return Err("preview_audio_shape_invalid");
        }
        initialize()?;
        let mut context = codec::Context::from_parameters(parameters)
            .map_err(|_| "preview_audio_decoder_unavailable")?
            .decoder();
        context.set_packet_time_base(AUDIO_CLOCK);
        context.set_threading(codec::threading::Config::count(1));
        // SAFETY: Own the unopened context; libavcodec checks this before
        // allocating a decoded frame from an untrusted compressed packet.
        unsafe { (*context.as_mut_ptr()).max_samples = MAX_AUDIO_SAMPLES };
        let decoder = context
            .audio()
            .map_err(|_| "preview_audio_decoder_unavailable")?;
        let encoder = OpusEncoder::new()?;
        Ok(Self {
            decoder,
            source_clock: time_base,
            preparation: None,
            encoder,
            previous_frame: None,
        })
    }

    pub fn push(&mut self, input: &ffmpeg::Packet) -> Result<Vec<EncodedAudio>> {
        let packet = packet_on_clock(input, self.source_clock, AUDIO_CLOCK)?;
        self.decoder
            .send_packet(&packet)
            .map_err(|_| "preview_audio_decode_failed")?;
        let mut output = Vec::new();
        let mut decoded = frame::Audio::empty();
        for _ in 0..MAX_DECODED_FRAMES {
            match self.decoder.receive_frame(&mut decoded) {
                Ok(()) => self.process(&mut decoded, &mut output)?,
                Err(error) if drained(error) => return Ok(output),
                Err(_) => return Err("preview_audio_decode_failed"),
            }
        }
        Err("preview_audio_decode_overflow")
    }

    fn process(
        &mut self,
        decoded: &mut frame::Audio,
        output: &mut Vec<EncodedAudio>,
    ) -> Result<()> {
        let timestamp = decoded
            .timestamp()
            .or(decoded.pts())
            .ok_or("preview_timestamp_invalid")?;
        if self
            .previous_frame
            .is_some_and(|previous| timestamp <= previous)
        {
            return Err("preview_timestamp_invalid");
        }
        self.previous_frame = Some(timestamp);
        decoded.set_pts(Some(timestamp));
        if decoded.samples() == 0
            || decoded.samples() > MAX_AUDIO_SAMPLES as usize
            || !(8000..=192000).contains(&decoded.rate())
            || !(1..=8).contains(&decoded.channels())
            || decoded.format() == format::Sample::None
        {
            return Err("preview_audio_shape_invalid");
        }
        if decoded.channel_layout().is_empty() {
            decoded.set_channel_layout(ChannelLayout::default(i32::from(decoded.channels())));
        }
        let shape = AudioShape {
            rate: decoded.rate(),
            format: decoded.format(),
            layout: decoded.channel_layout(),
        };
        match &self.preparation {
            Some(preparation) if preparation.shape != shape => {
                return Err("preview_parameters_changed");
            }
            None => self.preparation = Some(AudioPreparation::new(shape)?),
            _ => {}
        }
        let preparation = self
            .preparation
            .as_mut()
            .ok_or("preview_audio_prepare_failed")?;
        preparation
            .graph
            .get("in")
            .ok_or("preview_audio_prepare_failed")?
            .source()
            .add(decoded)
            .map_err(|_| "preview_audio_prepare_failed")?;
        let mut prepared = frame::Audio::empty();
        for _ in 0..MAX_OUTPUT_PACKETS {
            match preparation
                .graph
                .get("out")
                .ok_or("preview_audio_prepare_failed")?
                .sink()
                .frame(&mut prepared)
            {
                Ok(()) => {
                    if output.len() >= MAX_OUTPUT_PACKETS {
                        return Err("preview_audio_encode_overflow");
                    }
                    self.encoder.push(&prepared, output)?;
                }
                Err(error) if drained(error) => return Ok(()),
                Err(_) => return Err("preview_audio_prepare_failed"),
            }
        }
        Err("preview_audio_prepare_overflow")
    }
}

impl AudioPreparation {
    fn new(shape: AudioShape) -> Result<Self> {
        let mut graph = filter::Graph::new();
        // SAFETY: This private graph is not configured or shared yet.
        unsafe { (*graph.as_mut_ptr()).nb_threads = 1 };
        let arguments = format!(
            "time_base=1/48000:sample_rate={}:sample_fmt={}:channel_layout=0x{:x}",
            shape.rate,
            shape.format.name(),
            shape.layout.bits()
        );
        add_filter(&mut graph, "abuffer", "in", &arguments)?;
        add_filter(&mut graph, "abuffersink", "out", "")?;
        graph
            .output("in", 0)
            .and_then(|parser| parser.input("out", 0))
            .and_then(|parser| {
                parser.parse("aresample=48000,aformat=sample_fmts=flt:channel_layouts=stereo")
            })
            .map_err(|_| "preview_audio_prepare_failed")?;
        graph
            .validate()
            .map_err(|_| "preview_audio_prepare_failed")?;
        graph
            .get("out")
            .ok_or("preview_audio_prepare_failed")?
            .sink()
            .set_frame_size(OPUS_SAMPLES as u32);
        Ok(Self { shape, graph })
    }
}

/// Opus-specific settings remain private, like the H.264 adapter above.
struct OpusEncoder {
    encoder: codec::encoder::audio::Encoder,
    previous_output: Option<i64>,
}

impl OpusEncoder {
    fn new() -> Result<Self> {
        let implementation =
            ffmpeg::encoder::find_by_name("libopus").ok_or("preview_audio_encoder_unavailable")?;
        let mut encoder = codec::Context::new_with_codec(implementation)
            .encoder()
            .audio()
            .map_err(|_| "preview_audio_encoder_unavailable")?;
        encoder.set_rate(48_000);
        encoder.set_channel_layout(ChannelLayout::STEREO);
        encoder.set_format(format::Sample::F32(format::sample::Type::Packed));
        encoder.set_bit_rate(64_000);
        encoder.set_time_base(AUDIO_CLOCK);
        let mut options = ffmpeg::Dictionary::new();
        options.set("application", "lowdelay");
        options.set("frame_duration", "20");
        let encoder = encoder
            .open_as_with(implementation, options)
            .map_err(|_| "preview_audio_encoder_unavailable")?;
        if encoder.frame_size() != OPUS_SAMPLES as u32 {
            return Err("preview_audio_encoder_unavailable");
        }
        Ok(Self {
            encoder,
            previous_output: None,
        })
    }

    fn push(&mut self, frame: &frame::Audio, output: &mut Vec<EncodedAudio>) -> Result<()> {
        if frame.samples() != OPUS_SAMPLES || frame.pts().is_none() {
            return Err("preview_audio_frame_invalid");
        }
        self.encoder
            .send_frame(frame)
            .map_err(|_| "preview_audio_encode_failed")?;
        let mut packet = ffmpeg::Packet::empty();
        loop {
            match self.encoder.receive_packet(&mut packet) {
                Ok(()) => {
                    if output.len() >= MAX_OUTPUT_PACKETS || packet.size() > MAX_AUDIO_PACKET_BYTES
                    {
                        return Err("preview_audio_encode_overflow");
                    }
                    let timestamp = packet.pts().ok_or("preview_timestamp_invalid")?;
                    if self
                        .previous_output
                        .is_some_and(|previous| timestamp <= previous)
                    {
                        return Err("preview_timestamp_invalid");
                    }
                    self.previous_output = Some(timestamp);
                    let data = packet
                        .data()
                        .filter(|data| !data.is_empty())
                        .ok_or("preview_audio_encode_failed")?;
                    output.push(EncodedAudio {
                        timestamp,
                        data: Arc::from(data),
                    });
                }
                Err(error) if drained(error) => return Ok(()),
                Err(_) => return Err("preview_audio_encode_failed"),
            }
        }
    }
}

fn initialize() -> Result<()> {
    ffmpeg::util::log::set_level(ffmpeg::util::log::Level::Quiet);
    ffmpeg::init().map_err(|_| "preview_decoder_unavailable")
}

fn valid_clock(clock: Rational) -> Result<()> {
    if clock.numerator() <= 0 || clock.denominator() <= 0 {
        return Err("preview_timestamp_invalid");
    }
    Ok(())
}

fn packet_on_clock(
    input: &ffmpeg::Packet,
    source: Rational,
    destination: Rational,
) -> Result<ffmpeg::Packet> {
    if input.size() == 0 || input.size() > MAX_PACKET_BYTES {
        return Err("preview_packet_invalid");
    }
    if input.pts().is_none() || input.dts().is_none() {
        return Err("preview_timestamp_invalid");
    }
    let mut packet = input.clone();
    packet.rescale_ts(source, destination);
    Ok(packet)
}

fn ratio_or(value: Rational, fallback: Rational) -> Rational {
    if value.numerator() > 0 && value.denominator() > 0 {
        value
    } else {
        fallback
    }
}

fn add_filter(
    graph: &mut filter::Graph,
    name: &str,
    instance: &str,
    arguments: &str,
) -> Result<()> {
    graph
        .add(
            &filter::find(name).ok_or("preview_prepare_unavailable")?,
            instance,
            arguments,
        )
        .map_err(|_| "preview_prepare_failed")?;
    Ok(())
}

fn drained(error: Error) -> bool {
    matches!(
        error,
        Error::Eof
            | Error::Other {
                errno: ffmpeg::error::EAGAIN
            }
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use ffmpeg::Rescale;
    use std::path::PathBuf;

    fn fixture() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/recording/h264-aac-bframes.mp4")
    }

    fn packets(kind: ffmpeg::media::Type) -> (codec::Parameters, Rational, Vec<ffmpeg::Packet>) {
        let mut input = ffmpeg::format::input(&fixture()).unwrap();
        let stream = input
            .streams()
            .find(|stream| stream.parameters().medium() == kind)
            .unwrap();
        let index = stream.index();
        let parameters = stream.parameters();
        let clock = stream.time_base();
        let packets = input
            .packets()
            .filter(|(stream, _)| stream.index() == index)
            .map(|(_, packet)| packet)
            .collect();
        (parameters, clock, packets)
    }

    fn offset(packet: &mut ffmpeg::Packet, clock: Rational, seconds: i64) {
        let ticks = seconds.rescale(Rational(1, 1), clock);
        packet.set_pts(packet.pts().map(|pts| pts + ticks));
        packet.set_dts(packet.dts().map(|dts| dts + ticks));
    }

    fn assert_video_decodes(encoded: &[EncodedVideo]) {
        let implementation = ffmpeg::decoder::find(codec::Id::H264).unwrap();
        let mut decoder = codec::Context::new_with_codec(implementation)
            .decoder()
            .video()
            .unwrap();
        let mut count = 0;
        for encoded in encoded {
            let mut packet = ffmpeg::Packet::copy(&encoded.data);
            packet.set_pts(Some(encoded.timestamp));
            packet.set_dts(Some(encoded.timestamp));
            decoder.send_packet(&packet).unwrap();
            let mut decoded = frame::Video::empty();
            while decoder.receive_frame(&mut decoded).is_ok() {
                assert_eq!((decoded.width(), decoded.height()), (160, 120));
                count += 1;
            }
        }
        decoder.send_eof().unwrap();
        let mut decoded = frame::Video::empty();
        while decoder.receive_frame(&mut decoded).is_ok() {
            count += 1;
        }
        assert_eq!(count, encoded.len());
    }

    fn assert_audio_decodes(parameters: codec::Parameters, encoded: &[EncodedAudio]) {
        let mut decoder = codec::Context::from_parameters(parameters)
            .unwrap()
            .decoder()
            .audio()
            .unwrap();
        let mut samples = 0;
        for encoded in encoded {
            let mut packet = ffmpeg::Packet::copy(&encoded.data);
            packet.set_pts(Some(encoded.timestamp));
            packet.set_dts(Some(encoded.timestamp));
            decoder.send_packet(&packet).unwrap();
            let mut decoded = frame::Audio::empty();
            while decoder.receive_frame(&mut decoded).is_ok() {
                assert_eq!(decoded.channels(), 2);
                assert_eq!(decoded.rate(), 48_000);
                assert!(decoded.samples() <= OPUS_SAMPLES);
                samples += decoded.samples();
            }
        }
        assert!(samples > OPUS_SAMPLES * 3);
    }

    fn g711_parameters(id: codec::Id) -> codec::Parameters {
        let mut parameters = codec::Parameters::new();
        // SAFETY: The fresh owned parameters have no extradata or allocated
        // layout. These scalar fields describe the synthetic protocol fixture.
        unsafe {
            let raw = &mut *parameters.as_mut_ptr();
            raw.codec_type = ffmpeg::ffi::AVMediaType::AVMEDIA_TYPE_AUDIO;
            raw.codec_id = id.into();
            raw.sample_rate = 8000;
            raw.ch_layout = ChannelLayout::MONO.into();
        }
        parameters
    }

    fn g711_packet(timestamp: i64) -> ffmpeg::Packet {
        let bytes: Vec<u8> = (0..160)
            .map(|index| if index % 16 < 8 { 0x80 } else { 0x00 })
            .collect();
        let mut packet = ffmpeg::Packet::copy(&bytes);
        packet.set_pts(Some(timestamp));
        packet.set_dts(Some(timestamp));
        packet.set_duration(160);
        packet
    }

    #[test]
    fn reordered_h264_becomes_decodable_baseline_with_source_pts_and_periodic_idr() {
        // Given: A real two-second B-frame stream starts five seconds into the camera clock.
        let (parameters, clock, mut input) = packets(ffmpeg::media::Type::Video);
        let mut transcoder = VideoTranscoder::new(parameters, clock).unwrap();
        let mut output = Vec::new();

        // When: Native preview decodes and re-encodes every source packet.
        for packet in &mut input {
            offset(packet, clock, 5);
            output.extend(transcoder.push(packet).unwrap());
        }
        output.extend(transcoder.finish().unwrap());

        // Then: All pictures survive on their original clock, without B slices;
        // each one-second IDR carries baseline SPS/PPS for independent late join.
        assert_eq!(output.len(), 24);
        assert_eq!(output[0].timestamp, 450_000);
        assert!(
            output
                .windows(2)
                .all(|pair| pair[1].timestamp - pair[0].timestamp == 7500)
        );
        let keys: Vec<_> = output.iter().filter(|frame| frame.keyframe).collect();
        assert_eq!(
            keys.iter().map(|frame| frame.timestamp).collect::<Vec<_>>(),
            [450_000, 540_000]
        );
        for key in keys {
            let nals = annex_nals(&key.data);
            let sps = nals
                .iter()
                .find(|nal| nal.first().is_some_and(|byte| byte & 31 == 7))
                .unwrap();
            assert_eq!(sps[1], 66);
            assert!(
                nals.iter()
                    .any(|nal| nal.first().is_some_and(|byte| byte & 31 == 8))
            );
            assert!(
                nals.iter()
                    .any(|nal| nal.first().is_some_and(|byte| byte & 31 == 5))
            );
        }
        assert!(output.iter().all(|frame| {
            crate::rtsp::validate_access_unit(&normalize_annex_b(&frame.data)).unwrap()
        }));
        assert_video_decodes(&output);
    }

    #[test]
    fn aac_becomes_stereo_opus_without_resetting_offset_or_encoder_lookahead() {
        // Given: AAC packets and video share a camera clock five seconds from zero.
        let (parameters, clock, mut input) = packets(ffmpeg::media::Type::Audio);
        let mut transcoder = AudioTranscoder::new(parameters, clock).unwrap();
        let mut output = Vec::new();
        let parameters = codec::Parameters::from(&transcoder.encoder.encoder);
        // SAFETY: Borrow this private opened encoder for its scalar timing field.
        let padding = unsafe { (*transcoder.encoder.encoder.as_ptr()).initial_padding };

        // When: Native preview uses decoded sample clocks and actual Opus packet PTS.
        for packet in &mut input {
            offset(packet, clock, 5);
            output.extend(transcoder.push(packet).unwrap());
        }

        // Then: Each packet is 20ms, including the initial codec lookahead;
        // no independent audio normalization erases the source/video offset.
        assert!(output.len() >= 95);
        assert!(padding > 0);
        assert_eq!(output[0].timestamp, 240_000 - i64::from(padding));
        assert!(
            output
                .windows(2)
                .all(|pair| pair[1].timestamp - pair[0].timestamp == 960)
        );
        assert_audio_decodes(parameters, &output);
    }

    #[test]
    fn both_g711_codecs_resample_mono_8khz_to_decodable_stereo_48khz_opus() {
        // Given: Camera PCM A-law and mu-law packet clocks start at five seconds.
        for id in [codec::Id::PCM_ALAW, codec::Id::PCM_MULAW] {
            let mut transcoder =
                AudioTranscoder::new(g711_parameters(id), Rational(1, 8000)).unwrap();
            let parameters = codec::Parameters::from(&transcoder.encoder.encoder);
            let mut output = Vec::new();

            // When: One second of twenty-millisecond camera packets is transcoded.
            for index in 0..50 {
                output.extend(transcoder.push(&g711_packet(40_000 + index * 160)).unwrap());
            }

            // Then: Rate/channel conversion stays bounded and retains the source clock.
            assert!(output.len() >= 48, "{} packets", output.len());
            assert!((239_040..240_000).contains(&output[0].timestamp));
            assert!(
                output
                    .windows(2)
                    .all(|pair| pair[1].timestamp - pair[0].timestamp == 960)
            );
            assert_audio_decodes(parameters, &output);
        }
    }

    #[test]
    fn existing_opus_camera_packets_can_feed_the_same_preview_audio_adapter() {
        // Given: One native encoder supplies actual Opus camera packets with codec metadata.
        let mut source =
            AudioTranscoder::new(g711_parameters(codec::Id::PCM_MULAW), Rational(1, 8000)).unwrap();
        let parameters = codec::Parameters::from(&source.encoder.encoder);
        let mut source_packets = Vec::new();
        for index in 0..50 {
            source_packets.extend(source.push(&g711_packet(40_000 + index * 160)).unwrap());
        }
        let mut preview = AudioTranscoder::new(parameters, AUDIO_CLOCK).unwrap();
        let parameters = codec::Parameters::from(&preview.encoder.encoder);
        let mut output = Vec::new();

        // When: Opus is decoded and converted through the same 48kHz stereo pipeline.
        for source in source_packets {
            let mut packet = ffmpeg::Packet::copy(&source.data);
            packet.set_pts(Some(source.timestamp));
            packet.set_dts(Some(source.timestamp));
            output.extend(preview.push(&packet).unwrap());
        }

        // Then: The source offset remains and its output is actual decodable Opus.
        assert!(output.len() >= 46);
        assert!((238_080..240_000).contains(&output[0].timestamp));
        assert_audio_decodes(parameters, &output);
    }

    #[test]
    fn missing_clock_oversized_packets_and_camera_clock_reset_are_refused() {
        // Given: A valid mono camera has started a native preview audio decoder.
        let mut preview =
            AudioTranscoder::new(g711_parameters(codec::Id::PCM_MULAW), Rational(1, 8000)).unwrap();
        let first = g711_packet(40_000);
        preview.push(&first).unwrap();

        // When: A camera repeats the old clock, omits PTS, or exceeds the packet bound.
        let reset = preview.push(&first).err();
        let mut missing = g711_packet(40_160);
        missing.set_pts(None);
        let missing = preview.push(&missing).err();
        let mut oversized = ffmpeg::Packet::copy(&vec![0; MAX_PACKET_BYTES + 1]);
        oversized.set_pts(Some(40_160));
        oversized.set_dts(Some(40_160));
        let oversized = preview.push(&oversized).err();

        // Then: Stable errors stop this consumer without guessing timestamps.
        assert_eq!(reset, Some("preview_timestamp_invalid"));
        assert_eq!(missing, Some("preview_timestamp_invalid"));
        assert_eq!(oversized, Some("preview_packet_invalid"));
    }

    #[test]
    fn midstream_audio_shape_changes_fail_before_reconfiguring_or_queuing_samples() {
        // Given: A preview has a configured eight-kilohertz mono PCM source.
        let mut preview =
            AudioTranscoder::new(g711_parameters(codec::Id::PCM_MULAW), Rational(1, 8000)).unwrap();
        preview.push(&g711_packet(40_000)).unwrap();
        let mut changed = frame::Audio::new(
            format::Sample::I16(format::sample::Type::Packed),
            160,
            ChannelLayout::MONO,
        );
        changed.set_rate(16_000);
        changed.set_pts(Some(240_960));

        // When: Decoding produces a frame with a different source sample rate.
        let result = preview.process(&mut changed, &mut Vec::new());

        // Then: The existing resampler lifetime cannot silently reinterpret it.
        assert_eq!(result, Err("preview_parameters_changed"));
    }

    fn annex_nals(bytes: &[u8]) -> Vec<&[u8]> {
        let mut starts = Vec::new();
        let mut index = 0;
        while index + 3 <= bytes.len() {
            let prefix = if bytes[index..].starts_with(&[0, 0, 0, 1]) {
                4
            } else if bytes[index..].starts_with(&[0, 0, 1]) {
                3
            } else {
                index += 1;
                continue;
            };
            starts.push((index, index + prefix));
            index += prefix;
        }
        starts
            .iter()
            .enumerate()
            .map(|(i, &(_, start))| {
                &bytes[start..starts.get(i + 1).map_or(bytes.len(), |next| next.0)]
            })
            .collect()
    }

    fn normalize_annex_b(bytes: &[u8]) -> Vec<u8> {
        let mut output = Vec::new();
        for nal in annex_nals(bytes) {
            output.extend_from_slice(&[0, 0, 0, 1]);
            output.extend_from_slice(nal);
        }
        output
    }
}
