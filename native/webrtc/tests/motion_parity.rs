//! Replay frozen Python/OpenCV observations without loading a camera or regenerating fixtures.

#[path = "../src/motion.rs"]
mod motion;

use motion::{MotionConfig, MotionDetector, MotionObservation, blur_gray};
use serde::Deserialize;

const CORPUS: &str = include_str!("fixtures/motion/corpus.json");
const FRAMES: &[u8] = include_bytes!("fixtures/motion/frames.gray");

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct MotionCorpus {
    schema_version: u32,
    reference_opencv: String,
    pixel_format: String,
    data_file: String,
    cases: Vec<MotionCase>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct MotionCase {
    name: String,
    width: usize,
    height: usize,
    pixel_threshold: u64,
    min_changed_pct: f64,
    blur_kernel: usize,
    steps: Vec<MotionStep>,
}

#[derive(Deserialize)]
#[serde(tag = "operation", rename_all = "snake_case", deny_unknown_fields)]
enum MotionStep {
    Frame {
        input_offset: usize,
        blurred_offset: usize,
        threshold: Option<f64>,
        changed_pixels: usize,
        changed_pct: f64,
        motion: bool,
    },
    Reset,
}

fn config() -> MotionConfig {
    MotionConfig {
        pixel_threshold: 45,
        min_changed_pct: 1.0,
        blur_kernel: 0,
        recording_sensitivity_factor: 2.0,
    }
}

fn assert_observation(
    actual: MotionObservation,
    changed_pixels: usize,
    changed_pct: f64,
    motion: bool,
    context: &str,
) {
    assert_eq!(
        actual.changed_pixels, changed_pixels,
        "{context}: changed pixels"
    );
    assert!(
        (actual.changed_pct - changed_pct).abs() <= 1e-12,
        "{context}: changed percentage: got {}, expected {changed_pct}",
        actual.changed_pct,
    );
    assert_eq!(actual.motion, motion, "{context}: motion decision");
}

#[test]
fn every_frozen_python_case_matches_exact_blur_and_motion_observations() {
    // Given: A language-neutral corpus frozen from the production Python/OpenCV detector.
    let corpus: MotionCorpus = serde_json::from_str(CORPUS).expect("valid motion corpus");
    assert_eq!(corpus.schema_version, 1);
    assert!(!corpus.reference_opencv.is_empty());
    assert_eq!(corpus.pixel_format, "gray8");
    assert_eq!(corpus.data_file, "frames.gray");
    assert!(!corpus.cases.is_empty());

    for case in corpus.cases {
        let mut detector = MotionDetector::new(MotionConfig {
            pixel_threshold: case.pixel_threshold,
            min_changed_pct: case.min_changed_pct,
            blur_kernel: case.blur_kernel,
            ..config()
        })
        .unwrap_or_else(|error| panic!("{}: invalid detector config: {error:?}", case.name));
        let frame_size = case
            .width
            .checked_mul(case.height)
            .expect("fixture shape fits");
        assert!(
            frame_size > 0,
            "{}: fixture dimensions are nonempty",
            case.name
        );
        assert!(
            !case.steps.is_empty(),
            "{}: fixture has operations",
            case.name
        );

        for (index, step) in case.steps.into_iter().enumerate() {
            let context = format!("{} step {index}", case.name);
            match step {
                MotionStep::Reset => {
                    // When: A stream transition resets the detector between frames.
                    detector.reset();

                    // Then: Metrics clear immediately, and the following frame is a new baseline.
                    assert_observation(detector.observation(), 0, 0.0, false, &context);
                }
                MotionStep::Frame {
                    input_offset,
                    blurred_offset,
                    threshold,
                    changed_pixels,
                    changed_pct,
                    motion,
                } => {
                    let input = FRAMES
                        .get(input_offset..input_offset + frame_size)
                        .unwrap_or_else(|| panic!("{context}: input outside fixture data"));
                    let expected_blur = FRAMES
                        .get(blurred_offset..blurred_offset + frame_size)
                        .unwrap_or_else(|| panic!("{context}: blur outside fixture data"));

                    // When: Blurring and observing the next gray frame with its threshold override.
                    let actual_blur = blur_gray(input, case.width, case.height, case.blur_kernel)
                        .unwrap_or_else(|error| panic!("{context}: blur failed: {error:?}"));
                    let actual = detector
                        .detect(input, case.width, case.height, threshold)
                        .unwrap_or_else(|error| panic!("{context}: detection failed: {error:?}"));

                    // Then: Every blurred byte and public observation matches the frozen oracle.
                    assert_eq!(
                        actual_blur.len(),
                        expected_blur.len(),
                        "{context}: blur length"
                    );
                    assert!(
                        actual_blur == expected_blur,
                        "{context}: blurred bytes differ; first mismatch {:?}",
                        actual_blur
                            .iter()
                            .zip(expected_blur)
                            .enumerate()
                            .find(|(_, (actual, expected))| actual != expected),
                    );
                    assert_observation(actual, changed_pixels, changed_pct, motion, &context);
                    assert_observation(
                        detector.observation(),
                        changed_pixels,
                        changed_pct,
                        motion,
                        &context,
                    );
                }
            }
        }
    }
}

#[test]
fn positive_even_kernels_match_the_sources_next_odd_kernel_normalization() {
    // Given: Identical synthetic frames and detectors using an even kernel or its next odd value.
    let width = 11;
    let height = 7;
    let frames: Vec<Vec<u8>> = (0..3)
        .map(|phase| {
            (0..width * height)
                .map(|pixel| ((pixel * 73 + phase * 151 + pixel * phase * 19) % 256) as u8)
                .collect()
        })
        .collect();

    for even_kernel in [2, 4, 6, 8, 10, 30, 254, 1000] {
        let mut even = MotionDetector::new(MotionConfig {
            blur_kernel: even_kernel,
            ..config()
        })
        .unwrap();
        let mut odd = MotionDetector::new(MotionConfig {
            blur_kernel: even_kernel + 1,
            ..config()
        })
        .unwrap();

        for frame in &frames {
            // When: Feeding the same frame sequence to both configurations.
            let actual = even.detect(frame, width, height, None).unwrap();
            let expected = odd.detect(frame, width, height, None).unwrap();

            // Then: Normalizing before detection produces identical motion observations.
            assert_observation(
                actual,
                expected.changed_pixels,
                expected.changed_pct,
                expected.motion,
                &format!("kernel {even_kernel}"),
            );
        }
    }
}

#[test]
fn pixel_thresholds_larger_than_a_byte_do_not_wrap() {
    // Given: Configurations exceeding all possible unsigned 8-bit intensity differences.
    for pixel_threshold in [255, 256, u64::MAX] {
        let mut detector = MotionDetector::new(MotionConfig {
            pixel_threshold,
            ..config()
        })
        .unwrap();
        detector.detect(&[0, 0, 0, 0], 2, 2, None).unwrap();

        // When: Every pixel changes by the maximum possible intensity difference.
        let actual = detector.detect(&[255, 255, 255, 255], 2, 2, None).unwrap();

        // Then: The strict comparison counts no pixels and does not trigger motion.
        assert_observation(actual, 0, 0.0, false, "large pixel threshold");
    }
}

#[test]
fn invalid_frame_buffers_do_not_replace_the_last_valid_baseline() {
    // Given: A detector that has observed a valid baseline and one changed pixel.
    let mut detector = MotionDetector::new(config()).unwrap();
    detector.detect(&[0, 0, 0, 0], 2, 2, None).unwrap();
    detector.detect(&[46, 0, 0, 0], 2, 2, None).unwrap();

    for (frame, width, height) in [
        (&[][..], 0, 2),
        (&[][..], 2, 0),
        (&[0, 0, 0][..], 2, 2),
        (&[0, 0, 0, 0, 0][..], 2, 2),
        (&[][..], usize::MAX, 2),
    ] {
        // When: A malformed frame has an empty, overflowing, short, or oversized shape.
        let result = detector.detect(frame, width, height, None);

        // Then: It is refused without clearing or advancing the previous valid observation.
        assert!(
            result.is_err(),
            "invalid shape {width}x{height} must be refused"
        );
        assert_observation(detector.observation(), 1, 25.0, true, "invalid frame");
    }

    let next = detector.detect(&[46, 0, 0, 0], 2, 2, None).unwrap();
    assert_observation(next, 0, 0.0, false, "baseline survives invalid frame");
}

#[test]
fn dimension_changes_require_reset_and_reset_permits_a_new_shape() {
    // Given: A valid detector baseline established on a 2x2 gray frame.
    let mut detector = MotionDetector::new(config()).unwrap();
    detector.detect(&[0, 0, 0, 0], 2, 2, None).unwrap();

    // When: The stream supplies a different shape, including one with the same byte count.
    for (frame, width, height) in [(&[46, 0, 0, 0][..], 4, 1), (&[46, 0][..], 2, 1)] {
        let result = detector.detect(frame, width, height, None);

        // Then: Comparing incompatible shapes is refused until the source explicitly resets.
        assert!(
            result.is_err(),
            "shape change {width}x{height} must be refused"
        );
        assert_observation(detector.observation(), 0, 0.0, false, "shape mismatch");
    }

    detector.reset();
    let first = detector.detect(&[46, 0], 2, 1, Some(0.0)).unwrap();
    assert_observation(first, 0, 0.0, false, "first frame after shape reset");
    let second = detector.detect(&[0, 0], 2, 1, None).unwrap();
    assert_observation(second, 1, 50.0, true, "new shape baseline");
}

#[test]
fn configuration_constraints_match_the_existing_motion_settings() {
    // Given: Configurations below the existing minimum percentage or sensitivity factor.
    for invalid in [
        MotionConfig {
            min_changed_pct: -1.0,
            ..config()
        },
        MotionConfig {
            min_changed_pct: f64::NAN,
            ..config()
        },
        MotionConfig {
            min_changed_pct: f64::NEG_INFINITY,
            ..config()
        },
        MotionConfig {
            recording_sensitivity_factor: 0.999,
            ..config()
        },
        MotionConfig {
            recording_sensitivity_factor: 0.0,
            ..config()
        },
        MotionConfig {
            recording_sensitivity_factor: f64::NAN,
            ..config()
        },
    ] {
        // When: Building a detector without Python's validated configuration boundary.
        let result = MotionDetector::new(invalid);

        // Then: Values rejected by the Python configuration are also rejected here.
        assert!(result.is_err());
    }

    // Existing configuration deliberately permits percentage and pixel thresholds above 100/255.
    assert!(
        MotionDetector::new(MotionConfig {
            pixel_threshold: 256,
            min_changed_pct: 100.1,
            recording_sensitivity_factor: 1.0,
            ..config()
        })
        .is_ok()
    );
}

#[test]
fn recording_sensitivity_uses_the_existing_percentage_division() {
    // Given: Existing validated settings, including accepted positive infinity values.
    for (min_changed_pct, factor, expected) in [
        (1.0, 1.0, 1.0),
        (1.0, 2.0, 0.5),
        (30.0, 2.0, 15.0),
        (1.0, 3.0, 1.0 / 3.0),
        (0.0, 2.0, 0.0),
        (100.1, 1.0, 100.1),
        (1.0, f64::INFINITY, 0.0),
        (f64::INFINITY, 2.0, f64::INFINITY),
        (f64::INFINITY, f64::INFINITY, 0.0),
    ] {
        let settings = MotionConfig {
            min_changed_pct,
            recording_sensitivity_factor: factor,
            ..config()
        };

        // When: Computing the same override used by Python while a recording is active.
        let actual = settings.recording_threshold();
        let detector = MotionDetector::new(settings);

        // Then: Division and zero-clamping match the existing source, including infinity/NaN math.
        assert!(
            detector.is_ok(),
            "accepted settings {min_changed_pct}/{factor}"
        );
        assert_eq!(
            actual, expected,
            "recording threshold {min_changed_pct}/{factor}"
        );
    }
}

#[test]
fn recording_threshold_changes_only_the_percentage_decision() {
    // Given: The same four pixels, a 30% idle threshold, and a factor-two recording override.
    let settings = MotionConfig {
        min_changed_pct: 30.0,
        ..config()
    };
    let threshold = settings.recording_threshold();
    let mut detector = MotionDetector::new(settings).unwrap();
    detector.detect(&[0, 0, 0, 0], 2, 2, None).unwrap();

    // When: One pixel changes at idle and then changes back during recording.
    let idle = detector.detect(&[46, 0, 0, 0], 2, 2, None).unwrap();
    let recording = detector
        .detect(&[0, 0, 0, 0], 2, 2, Some(threshold))
        .unwrap();

    // Then: The changed count/percentage stay identical while the recording override detects motion.
    assert_observation(idle, 1, 25.0, false, "idle threshold");
    assert_observation(recording, 1, 25.0, true, "recording threshold");
}

#[test]
fn threshold_overrides_preserve_python_nan_and_infinity_comparisons() {
    // Given: Threshold overrides accepted by the Python detector, including non-finite floats.
    for (threshold, motion) in [
        (f64::NAN, false),
        (f64::INFINITY, false),
        (f64::NEG_INFINITY, true),
        (-1.0, true),
    ] {
        let mut detector = MotionDetector::new(config()).unwrap();

        // When: Establishing the baseline, changing one pixel, then repeating that changed frame.
        let first = detector
            .detect(&[0, 0, 0, 0], 2, 2, Some(threshold))
            .unwrap();
        let changed = detector
            .detect(&[46, 0, 0, 0], 2, 2, Some(threshold))
            .unwrap();
        let repeated = detector.detect(&[46, 0, 0, 0], 2, 2, None).unwrap();

        // Then: Initialization never detects motion, and an override still advances the baseline.
        assert_observation(first, 0, 0.0, false, "non-finite first-frame override");
        assert_observation(changed, 1, 25.0, motion, "non-finite override");
        assert_observation(repeated, 0, 0.0, false, "baseline after override");
    }
}
