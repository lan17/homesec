//! Decode native H.264 access units into the existing motion input format.
//!
//! FFmpeg's filter graph preserves timestamp sampling, bicubic scaling, color
//! conversion and gray8 rounding. No codec error text or media bytes are logged.

use crate::rtp::EncodedFrame;
use ffmpeg::{Error, Rational, codec, color, filter, format::Pixel, frame::Video};
use ffmpeg_next as ffmpeg;

const WIDTH: usize = 320;
const HEIGHT: usize = 240;
const MAX_PACKET_BYTES: usize = 2 * 1024 * 1024;
const MAX_OUTPUTS_PER_PUSH: usize = 32;
const MAX_DECODED_FRAMES_PER_PUSH: usize = 16;

pub struct GrayFrame {
    pub data: Vec<u8>,
    pub width: usize,
    pub height: usize,
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct Parameters {
    width: u32,
    height: u32,
    pixel_format: Pixel,
    aspect_ratio: Rational,
    frame_rate: Rational,
    color_space: color::Space,
    color_range: color::Range,
}

struct Preparation {
    graph: filter::Graph,
    parameters: Parameters,
}

pub struct GrayDecoder {
    decoder: codec::decoder::Video,
    preparation: Option<Preparation>,
    previous_timestamp: Option<u32>,
    elapsed_ticks: i64,
}

impl GrayDecoder {
    pub fn new() -> Result<Self, &'static str> {
        // FFmpeg diagnostics can contain camera URLs and codec payload details.
        // This worker reports only the stable errors returned below.
        ffmpeg::util::log::set_level(ffmpeg::util::log::Level::Quiet);
        ffmpeg::init().map_err(|_| "motion_decoder_unavailable")?;
        let codec = codec::decoder::find(codec::Id::H264).ok_or("motion_decoder_unavailable")?;
        let mut decoder = codec::Context::new_with_codec(codec).decoder();
        decoder.set_packet_time_base((1, 90_000));
        decoder.set_threading(codec::threading::Config::count(1));
        let decoder = decoder
            .open_as(codec)
            .and_then(|opened| opened.video())
            .map_err(|_| "motion_decoder_unavailable")?;
        Ok(Self {
            decoder,
            preparation: None,
            previous_timestamp: None,
            elapsed_ticks: 0,
        })
    }

    pub fn push(
        &mut self,
        frame: &EncodedFrame,
        mut emit: impl FnMut(GrayFrame),
    ) -> Result<(), &'static str> {
        if frame.data.is_empty() || frame.data.len() > MAX_PACKET_BYTES {
            return Err("motion_frame_invalid");
        }
        if let Some(previous) = self.previous_timestamp {
            let delta = frame.timestamp.wrapping_sub(previous);
            if delta == 0 || delta >= (1 << 31) {
                return Err("motion_timestamp_invalid");
            }
            self.elapsed_ticks = self
                .elapsed_ticks
                .checked_add(i64::from(delta))
                .ok_or("motion_timestamp_invalid")?;
        }
        self.previous_timestamp = Some(frame.timestamp);
        // Packet::copy supplies the padding required by FFmpeg's bit readers.
        let mut packet = ffmpeg::Packet::copy(&frame.data);
        packet.set_pts(Some(self.elapsed_ticks));
        packet.set_dts(Some(self.elapsed_ticks));
        self.decoder
            .send_packet(&packet)
            .map_err(|_| "motion_decode_failed")?;
        self.receive(&mut emit, &mut 0)
    }

    fn receive(
        &mut self,
        emit: &mut impl FnMut(GrayFrame),
        outputs: &mut usize,
    ) -> Result<(), &'static str> {
        let mut decoded = Video::empty();
        for _ in 0..MAX_DECODED_FRAMES_PER_PUSH {
            match self.decoder.receive_frame(&mut decoded) {
                Ok(()) => {
                    let timestamp = decoded.timestamp().or(decoded.pts());
                    if timestamp.is_none() {
                        return Err("motion_timestamp_invalid");
                    }
                    decoded.set_pts(timestamp);
                    let parameters = Parameters {
                        width: decoded.width(),
                        height: decoded.height(),
                        pixel_format: decoded.format(),
                        aspect_ratio: nonzero_ratio(decoded.aspect_ratio(), Rational(1, 1)),
                        frame_rate: nonzero_ratio(
                            self.decoder.frame_rate().unwrap_or(Rational(0, 1)),
                            Rational(0, 1),
                        ),
                        color_space: decoded.color_space(),
                        color_range: decoded.color_range(),
                    };
                    match &self.preparation {
                        Some(preparation) if preparation.parameters != parameters => {
                            return Err("motion_parameters_changed");
                        }
                        None => self.preparation = Some(Preparation::new(parameters)?),
                        _ => {}
                    }
                    let preparation = self.preparation.as_mut().ok_or("motion_prepare_failed")?;
                    preparation
                        .graph
                        .get("in")
                        .ok_or("motion_prepare_failed")?
                        .source()
                        .add(&decoded)
                        .map_err(|_| "motion_prepare_failed")?;
                    preparation.drain(emit, outputs)?;
                }
                Err(error) if drained(error) => return Ok(()),
                Err(_) => return Err("motion_decode_failed"),
            }
        }
        Err("motion_decode_overflow")
    }

    #[cfg(test)]
    // The executable test target compiles this fixture-only EOF helper too.
    #[allow(dead_code)]
    pub(crate) fn finish(
        &mut self,
        end_ticks: i64,
        mut emit: impl FnMut(GrayFrame),
    ) -> Result<(), &'static str> {
        let mut outputs = 0;
        self.decoder
            .send_eof()
            .map_err(|_| "motion_decode_failed")?;
        self.receive(&mut emit, &mut outputs)?;
        if let Some(preparation) = &mut self.preparation {
            preparation
                .graph
                .get("in")
                .ok_or("motion_prepare_failed")?
                .source()
                .close(end_ticks)
                .map_err(|_| "motion_prepare_failed")?;
            preparation.drain(&mut emit, &mut outputs)?;
        }
        Ok(())
    }
}

impl Preparation {
    fn new(parameters: Parameters) -> Result<Self, &'static str> {
        if parameters.width == 0 || parameters.height == 0 || parameters.pixel_format == Pixel::None
        {
            return Err("motion_frame_invalid");
        }
        let mut graph = filter::Graph::new();
        let mut args = format!(
            "video_size={}x{}:pix_fmt={}:time_base=1/90000:pixel_aspect={}/{}:frame_rate={}/{}",
            parameters.width,
            parameters.height,
            ffmpeg::ffi::AVPixelFormat::from(parameters.pixel_format) as i32,
            parameters.aspect_ratio.numerator(),
            parameters.aspect_ratio.denominator(),
            parameters.frame_rate.numerator(),
            parameters.frame_rate.denominator(),
        );
        // FFmpeg 7 added buffer-source color options. Earlier versions consume
        // the corresponding decoded AVFrame metadata directly.
        if filter::version() >> 16 >= 10 {
            args.push_str(&format!(
                ":colorspace={}:range={}",
                ffmpeg::ffi::AVColorSpace::from(parameters.color_space) as i32,
                ffmpeg::ffi::AVColorRange::from(parameters.color_range) as i32,
            ));
        }
        graph
            .add(
                &filter::find("buffer").ok_or("motion_prepare_unavailable")?,
                "in",
                &args,
            )
            .map_err(|_| "motion_prepare_failed")?;
        graph
            .add(
                &filter::find("buffersink").ok_or("motion_prepare_unavailable")?,
                "out",
                "",
            )
            .map_err(|_| "motion_prepare_failed")?;
        // The explicit format filter also works with FFmpeg versions that
        // refuse changing buffersink pixel options after initialization.
        graph
            .output("in", 0)
            .and_then(|parser| parser.input("out", 0))
            .and_then(|parser| parser.parse("fps=10,scale=320:240,format=pix_fmts=gray"))
            .map_err(|_| "motion_prepare_failed")?;
        graph.validate().map_err(|_| "motion_prepare_failed")?;
        Ok(Self { graph, parameters })
    }

    fn drain(
        &mut self,
        emit: &mut impl FnMut(GrayFrame),
        outputs: &mut usize,
    ) -> Result<(), &'static str> {
        let mut filtered = Video::empty();
        loop {
            match self
                .graph
                .get("out")
                .ok_or("motion_prepare_failed")?
                .sink()
                .frame(&mut filtered)
            {
                Ok(()) => {
                    if *outputs >= MAX_OUTPUTS_PER_PUSH {
                        return Err("motion_prepare_overflow");
                    }
                    if filtered.format() != Pixel::GRAY8
                        || filtered.width() as usize != WIDTH
                        || filtered.height() as usize != HEIGHT
                    {
                        return Err("motion_frame_invalid");
                    }
                    let stride = filtered.stride(0);
                    if stride < WIDTH {
                        return Err("motion_frame_invalid");
                    }
                    let mut data = Vec::with_capacity(WIDTH * HEIGHT);
                    for row in filtered.data(0).chunks_exact(stride).take(HEIGHT) {
                        data.extend_from_slice(&row[..WIDTH]);
                    }
                    if data.len() != WIDTH * HEIGHT {
                        return Err("motion_frame_invalid");
                    }
                    *outputs += 1;
                    emit(GrayFrame {
                        data,
                        width: WIDTH,
                        height: HEIGHT,
                    });
                }
                Err(error) if drained(error) => return Ok(()),
                Err(_) => return Err("motion_prepare_failed"),
            }
        }
    }
}

fn nonzero_ratio(value: Rational, fallback: Rational) -> Rational {
    if value.numerator() == 0 || value.denominator() == 0 {
        fallback
    } else {
        value
    }
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
