//! Bounded, same-installation CLI oracle for synthetic motion preparation.

use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

pub(crate) fn prepare_gray(path: &Path, fps: u32, frames: usize) -> Vec<u8> {
    // Raw H.264 has no container timestamps. Set the demuxer cadence rather
    // than overriding decoded timestamps with -r, which breaks EOF durations
    // on FFmpeg 5.1. Cap output at the known synthetic fixture duration.
    let expected_bytes = frames * 320 * 240;
    let mut output = tempfile::tempfile().expect("temporary reference output");
    let executable =
        std::env::var_os("HOMESEC_FFMPEG_REFERENCE").unwrap_or_else(|| "ffmpeg".into());
    let mut child = Command::new(executable)
        .args([
            "-hide_banner",
            "-nostdin",
            "-loglevel",
            "error",
            "-fflags",
            "+genpts+igndts",
            "-framerate",
        ])
        .arg(fps.to_string())
        .arg("-i")
        .arg(path)
        .args([
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray",
            "-vf",
            "fps=10,scale=320:240",
            "-an",
            "-frames:v",
        ])
        .arg(frames.to_string())
        .arg("-")
        .stdin(Stdio::null())
        .stdout(Stdio::from(output.try_clone().unwrap()))
        .stderr(Stdio::null())
        .spawn()
        .expect("FFmpeg CLI is required for preparation parity tests");
    let deadline = Instant::now() + Duration::from_secs(5);
    let status = loop {
        if let Some(status) = child.try_wait().unwrap() {
            break status;
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            panic!("FFmpeg reference preparation timed out");
        }
        thread::sleep(Duration::from_millis(10));
    };
    assert!(status.success(), "FFmpeg reference preparation failed");
    output.seek(SeekFrom::Start(0)).unwrap();
    let mut bytes = Vec::with_capacity(expected_bytes);
    output
        .take(expected_bytes as u64 + 1)
        .read_to_end(&mut bytes)
        .unwrap();
    assert_eq!(bytes.len(), expected_bytes, "reference gray frame count");
    bytes
}
