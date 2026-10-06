//! Grayscale motion parity with the existing Python/OpenCV detector.
//!
//! Config values come from the existing RTSP motion settings; there are no Rust
//! defaults or separate operator settings. Input preparation remains upstream.

use opencv::{
    core::{self, Mat, Size},
    imgproc,
    prelude::*,
};

// The frozen Python oracle fixes the native implementation as well as the
// bindings version. Refuse headers from another OpenCV build at compile time.
const _: () = assert!(
    core::CV_VERSION_MAJOR == 4 && core::CV_VERSION_MINOR == 12 && core::CV_VERSION_REVISION == 0
);

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

pub(crate) struct MotionDetector {
    config: MotionConfig,
    kernel_size: i32,
    previous: Mat,
    current: Mat,
    diff: Mat,
    mask: Mat,
    dimensions: Option<(usize, usize)>,
    observation: MotionObservation,
}

impl MotionDetector {
    pub(crate) fn new(config: MotionConfig) -> Result<Self, MotionError> {
        if !(config.min_changed_pct >= 0.0 && config.recording_sensitivity_factor >= 1.0) {
            return Err(MotionError::InvalidConfig);
        }
        Ok(Self {
            config,
            kernel_size: kernel_size(config.blur_kernel)?,
            previous: Mat::default(),
            current: Mat::default(),
            diff: Mat::default(),
            mask: Mat::default(),
            dimensions: None,
            observation: MotionObservation::default(),
        })
    }

    pub(crate) fn reset(&mut self) {
        self.dimensions = None;
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
        validate_frame_size(frame.len(), width, height)?;
        if let Some(dimensions) = self.dimensions
            && dimensions != (width, height)
        {
            return Err(MotionError::FrameDimensionsChanged);
        }
        // This borrowed Mat cannot outlive the input slice. The detector only
        // retains owned output buffers, reused by OpenCV for subsequent frames.
        let input = opencv_result(Mat::new_rows_cols_with_data(
            height as i32,
            width as i32,
            frame,
        ))?;
        if self.kernel_size == 1 {
            opencv_result(input.copy_to(&mut self.current))?;
        } else {
            opencv_result(imgproc::gaussian_blur(
                &input,
                &mut self.current,
                Size::new(self.kernel_size, self.kernel_size),
                0.0,
                0.0,
                core::BORDER_REFLECT_101,
                core::AlgorithmHint::ALGO_HINT_ACCURATE,
            ))?;
        }
        let observation = if self.dimensions.is_none() {
            MotionObservation::default()
        } else {
            let changed_pixels = if self.config.pixel_threshold >= 255 {
                0
            } else {
                opencv_result(core::absdiff(&self.current, &self.previous, &mut self.diff))?;
                opencv_result(imgproc::threshold(
                    &self.diff,
                    &mut self.mask,
                    self.config.pixel_threshold as f64,
                    255.0,
                    imgproc::THRESH_BINARY,
                ))?;
                opencv_result(core::count_non_zero(&self.mask))? as usize
            };
            let changed_pct = changed_pixels as f64 / frame.len() as f64 * 100.0;
            let threshold = threshold.unwrap_or(self.config.min_changed_pct);
            // Preserve Python comparisons, including NaN and +/- infinity.
            let threshold = if threshold < 0.0 { 0.0 } else { threshold };
            MotionObservation {
                changed_pixels,
                changed_pct,
                motion: changed_pct >= threshold,
            }
        };
        // Commit the baseline only after all processing succeeds. Rejected
        // frames leave both the last observation and previous frame intact.
        std::mem::swap(&mut self.current, &mut self.previous);
        self.dimensions = Some((width, height));
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
    validate_frame_size(frame.len(), width, height)?;
    let kernel = kernel_size(blur_kernel)?;
    if kernel == 1 {
        return Ok(frame.to_vec());
    }
    let input = opencv_result(Mat::new_rows_cols_with_data(
        height as i32,
        width as i32,
        frame,
    ))?;
    let mut output = Mat::default();
    opencv_result(imgproc::gaussian_blur(
        &input,
        &mut output,
        Size::new(kernel, kernel),
        0.0,
        0.0,
        core::BORDER_REFLECT_101,
        core::AlgorithmHint::ALGO_HINT_ACCURATE,
    ))?;
    Ok(opencv_result(output.data_bytes())?.to_vec())
}

fn kernel_size(size: usize) -> Result<i32, MotionError> {
    // Match RTSPSource's even-to-odd normalization and OpenCV's i32 Size.
    let normalized = if size <= 1 {
        1
    } else if size.is_multiple_of(2) {
        size.checked_add(1).ok_or(MotionError::InvalidConfig)?
    } else {
        size
    };
    i32::try_from(normalized).map_err(|_| MotionError::InvalidConfig)
}

fn validate_frame_size(size: usize, width: usize, height: usize) -> Result<(), MotionError> {
    // Mat dimensions and countNonZero's result use signed 32-bit integers.
    // Bound their product too, so a fully changed frame cannot overflow.
    if width == 0
        || height == 0
        || width > i32::MAX as usize
        || height > i32::MAX as usize
        || size > i32::MAX as usize
        || width.checked_mul(height) != Some(size)
    {
        return Err(MotionError::InvalidFrame);
    }
    Ok(())
}

fn opencv_result<T>(result: opencv::Result<T>) -> Result<T, MotionError> {
    // Preserve the runtime's existing stable invalid_motion_frame refusal;
    // native exception text must not cross the protocol/logging boundary.
    result.map_err(|_| MotionError::InvalidFrame)
}

#[cfg(test)]
mod tests {
    use super::{MotionError, validate_frame_size};

    #[test]
    fn frames_exceeding_opencv_dimension_or_count_limits_are_refused() {
        // Given: Logical gray-frame sizes without allocating multi-GB buffers.
        let limit = i32::MAX as usize;
        let shapes = [
            (limit + 1, limit + 1, 1),
            (limit + 1, 1, limit + 1),
            (46_341 * 46_341, 46_341, 46_341),
        ];

        for (size, width, height) in shapes {
            // When: Validating a matching shape that exceeds an OpenCV i32 limit.
            let result = validate_frame_size(size, width, height);

            // Then: It is refused before any native allocation or pixel counting.
            assert_eq!(result, Err(MotionError::InvalidFrame));
        }
        assert_eq!(validate_frame_size(limit, limit, 1), Ok(()));
        assert_eq!(validate_frame_size(limit, 1, limit), Ok(()));
    }
}
