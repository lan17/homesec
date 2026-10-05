//! Grayscale motion parity with the existing Python/OpenCV detector.
//!
//! The parity suite compiles this private module before decoder/runtime rollout.
//! Config values come from the existing RTSP motion settings; there are no Rust
//! defaults or separate operator settings. Input preparation remains upstream.

#[derive(Clone, Copy, Debug)]
pub(crate) struct MotionConfig {
    pub pixel_threshold: u64,
    pub min_changed_pct: f64,
    pub blur_kernel: usize,
    pub recording_sensitivity_factor: f64,
}

impl MotionConfig {
    pub(crate) fn recording_threshold(self) -> f64 {
        // Python's max(0.0, value) also maps NaN (inf / inf) to zero.
        (self.min_changed_pct / self.recording_sensitivity_factor).max(0.0)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct MotionObservation {
    pub changed_pixels: usize,
    pub changed_pct: f64,
    pub motion: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MotionError {
    InvalidConfig,
    InvalidFrame,
    FrameDimensionsChanged,
}

struct GrayFrame {
    pixels: Vec<u8>,
    width: usize,
    height: usize,
}

pub(crate) struct MotionDetector {
    config: MotionConfig,
    kernel: GaussianKernel,
    previous: Option<GrayFrame>,
    observation: MotionObservation,
}

impl MotionDetector {
    pub(crate) fn new(config: MotionConfig) -> Result<Self, MotionError> {
        if !(config.min_changed_pct >= 0.0 && config.recording_sensitivity_factor >= 1.0) {
            return Err(MotionError::InvalidConfig);
        }
        Ok(Self {
            config,
            kernel: GaussianKernel::new(config.blur_kernel)?,
            previous: None,
            observation: MotionObservation::default(),
        })
    }

    pub(crate) fn reset(&mut self) {
        self.previous = None;
        self.observation = MotionObservation::default();
    }

    pub(crate) fn observation(&self) -> MotionObservation {
        self.observation
    }

    pub(crate) fn detect(
        &mut self,
        frame: &[u8],
        width: usize,
        height: usize,
        threshold: Option<f64>,
    ) -> Result<MotionObservation, MotionError> {
        validate_frame(frame, width, height)?;
        if let Some(previous) = &self.previous
            && (previous.width != width || previous.height != height)
        {
            return Err(MotionError::FrameDimensionsChanged);
        }
        let pixels = self.kernel.blur(frame, width, height);
        let observation = match &self.previous {
            None => MotionObservation::default(),
            Some(previous) => {
                let changed_pixels = pixels
                    .iter()
                    .zip(&previous.pixels)
                    .filter(|(current, prior)| {
                        u64::from(current.abs_diff(**prior)) > self.config.pixel_threshold
                    })
                    .count();
                let changed_pct = changed_pixels as f64 / pixels.len() as f64 * 100.0;
                let threshold = threshold.unwrap_or(self.config.min_changed_pct);
                // Preserve Python comparisons, including NaN and +/- infinity.
                let threshold = if threshold < 0.0 { 0.0 } else { threshold };
                MotionObservation {
                    changed_pixels,
                    changed_pct,
                    motion: changed_pct >= threshold,
                }
            }
        };
        self.previous = Some(GrayFrame {
            pixels,
            width,
            height,
        });
        self.observation = observation;
        Ok(observation)
    }
}

#[cfg(test)]
#[allow(dead_code)] // The frozen corpus compiles this module in a separate test target.
pub(crate) fn blur_gray(
    frame: &[u8],
    width: usize,
    height: usize,
    blur_kernel: usize,
) -> Result<Vec<u8>, MotionError> {
    validate_frame(frame, width, height)?;
    Ok(GaussianKernel::new(blur_kernel)?.blur(frame, width, height))
}

fn validate_frame(frame: &[u8], width: usize, height: usize) -> Result<(), MotionError> {
    if width == 0
        || height == 0
        || width > isize::MAX as usize / 2
        || height > isize::MAX as usize / 2
        || width.checked_mul(height) != Some(frame.len())
    {
        return Err(MotionError::InvalidFrame);
    }
    Ok(())
}

struct GaussianKernel {
    // Q8 weights sum to 256, so there are at most 256 nonzero taps even for
    // large accepted kernels. Omit zero weights without changing convolution.
    taps: Vec<(isize, u32)>,
}

impl GaussianKernel {
    fn new(size: usize) -> Result<Self, MotionError> {
        // Match RTSPSource's even-to-odd normalization and OpenCV's i32 Size.
        let size = if size <= 1 {
            1
        } else if size.is_multiple_of(2) {
            size.checked_add(1).ok_or(MotionError::InvalidConfig)?
        } else {
            size
        };
        if size > i32::MAX as usize {
            return Err(MotionError::InvalidConfig);
        }
        let exact: &[u32] = match size {
            1 => &[256],
            3 => &[64, 128, 64],
            5 => &[16, 64, 96, 64, 16],
            7 => &[8, 28, 56, 72, 56, 28, 8],
            9 => &[4, 13, 30, 51, 60, 51, 30, 13, 4],
            _ => &[],
        };
        let half = (size / 2) as isize;
        if !exact.is_empty() {
            return Ok(Self {
                taps: exact
                    .iter()
                    .enumerate()
                    .map(|(i, weight)| (i as isize - half, *weight))
                    .collect(),
            });
        }

        // OpenCV 4.12 getGaussianKernelBitExact / getGaussianKernelFixedPoint_ED:
        // https://github.com/opencv/opencv/blob/4.12.0/modules/imgproc/src/smooth.dispatch.cpp
        // Normalize symmetrically, diffuse quantization error along the left
        // half, mirror it, and choose the center to retain an exact Q8 sum.
        let sigma = (size as f64).mul_add(0.15, 0.35);
        let exponent_scale = -0.125 / (sigma * sigma);
        let weight_at = |offset: isize| {
            let x = (2 * offset) as f64;
            (x * x * exponent_scale).exp()
        };
        let mut left_sum = 0.0;
        for offset in -half..0 {
            left_sum += weight_at(offset);
        }
        let scale = 1.0 / (left_sum * 2.0 + 1.0);
        let mut error = 0.0;
        let mut integer_sum = 0;
        let mut taps = Vec::new();
        for offset in -half..0 {
            let adjusted = weight_at(offset) * scale * 256.0 + error;
            let rounded = adjusted.round_ties_even();
            error = adjusted - rounded;
            let weight = rounded as u32;
            integer_sum += weight;
            if weight != 0 {
                taps.push((offset, weight));
                taps.push((-offset, weight));
            }
        }
        if integer_sum > 128 {
            return Err(MotionError::InvalidConfig);
        }
        let center = 256 - integer_sum * 2;
        if center != 0 {
            taps.push((0, center));
        }
        Ok(Self { taps })
    }

    fn blur(&self, frame: &[u8], width: usize, height: usize) -> Vec<u8> {
        if self.taps == [(0, 256)] {
            return frame.to_vec();
        }
        // Keep the horizontal result in Q8. Round only after the vertical Q8
        // pass, as OpenCV's uint8 bit-exact separable Gaussian filter does.
        let mut horizontal = vec![0_u16; frame.len()];
        for y in 0..height {
            for x in 0..width {
                let value = if width == 1 {
                    u32::from(frame[y]) * 256
                } else {
                    self.taps
                        .iter()
                        .map(|(offset, weight)| {
                            let column = reflect_101(x as isize + offset, width);
                            u32::from(frame[y * width + column]) * weight
                        })
                        .sum()
                };
                horizontal[y * width + x] = value as u16;
            }
        }
        let mut output = vec![0_u8; frame.len()];
        for y in 0..height {
            for x in 0..width {
                let value = if height == 1 {
                    u32::from(horizontal[x]) * 256
                } else {
                    self.taps
                        .iter()
                        .map(|(offset, weight)| {
                            let row = reflect_101(y as isize + offset, height);
                            u32::from(horizontal[row * width + x]) * weight
                        })
                        .sum()
                };
                output[y * width + x] = ((value + (1 << 15)) >> 16) as u8;
            }
        }
        output
    }
}

fn reflect_101(position: isize, length: usize) -> usize {
    let period = 2 * (length as isize - 1);
    let folded = position.rem_euclid(period) as usize;
    if folded < length {
        folded
    } else {
        period as usize - folded
    }
}
