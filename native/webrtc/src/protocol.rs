use serde::{Deserialize, Serialize};

pub const MAX_CONTROL_BYTES: usize = 256 * 1024;
pub const MAX_SDP_BYTES: usize = 128 * 1024;

#[derive(Deserialize)]
pub struct Request {
    pub request_id: String,
    #[serde(flatten)]
    pub operation: Operation,
}

#[derive(Deserialize)]
#[serde(tag = "command", rename_all = "snake_case")]
pub enum Operation {
    Start {
        ffmpeg_args: Option<Vec<String>>,
        rtsp_url: Option<String>,
    },
    Offer {
        session_id: String,
        sdp: String,
        lease_seconds: f64,
        lease_expires_at: Option<f64>,
    },
    Renew {
        session_id: String,
        lease_seconds: f64,
        lease_expires_at: Option<f64>,
    },
    Close {
        session_id: String,
    },
    Status,
    Stop,
}

#[derive(Default, Serialize)]
pub struct Reply {
    pub request_id: String,
    pub ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_code: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sdp: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub state: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub viewer_count: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub active_session_count: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub media_active: Option<bool>,
}

impl Reply {
    pub fn success(request_id: String) -> Self {
        Self {
            request_id,
            ok: true,
            ..Self::default()
        }
    }
    pub fn error(request_id: String, error_code: &'static str) -> Self {
        Self {
            request_id,
            error_code: Some(error_code),
            ..Self::default()
        }
    }
}

pub enum Control<T> {
    Request(T),
    End,
}

/// Shared bounded stdin transport. A malformed or unfinished message ends the worker.
pub fn controls<T: Send + 'static>(
    waker: std::sync::Arc<mio::Waker>,
    decode: fn(&[u8]) -> Option<T>,
) -> std::sync::mpsc::Receiver<Control<T>> {
    use std::io::{self, BufRead, Read};
    let (sender, receiver) = std::sync::mpsc::sync_channel(16);
    std::thread::spawn(move || {
        let mut input = io::stdin().lock();
        loop {
            let mut bytes = Vec::new();
            let read = (&mut input)
                .take((MAX_CONTROL_BYTES + 1) as u64)
                .read_until(b'\n', &mut bytes);
            let Ok(size) = read else { break };
            if size == 0 || size > MAX_CONTROL_BYTES || bytes.last() != Some(&b'\n') {
                break;
            }
            let Some(request) = decode(&bytes) else { break };
            if sender.send(Control::Request(request)).is_err() {
                return;
            }
            let _ = waker.wake();
        }
        let _ = sender.send(Control::End);
        let _ = waker.wake();
    });
    receiver
}

struct Output {
    bytes: Vec<u8>,
    completed: std::sync::mpsc::SyncSender<bool>,
}

/// An abandoned parent cannot block the camera lifecycle on a full stdout pipe.
/// A single writer preserves reply order; both queueing and completion are bounded.
pub fn output(value: &impl Serialize) -> Result<(), Box<dyn std::error::Error>> {
    use std::io::{self, Write};
    use std::sync::{OnceLock, mpsc};
    use std::time::Duration;
    static WRITER: OnceLock<mpsc::SyncSender<Output>> = OnceLock::new();
    let writer = WRITER.get_or_init(|| {
        let (sender, receiver) = mpsc::sync_channel::<Output>(1);
        std::thread::spawn(move || {
            let mut stdout = io::stdout().lock();
            while let Ok(message) = receiver.recv() {
                let written = stdout
                    .write_all(&message.bytes)
                    .and_then(|_| stdout.flush())
                    .is_ok();
                let _ = message.completed.try_send(written);
                if !written {
                    return;
                }
            }
        });
        sender
    });
    let mut bytes = serde_json::to_vec(value)?;
    // SDP escaping can expand otherwise bounded signaling strings.
    if bytes.len() > MAX_CONTROL_BYTES * 4 {
        return Err("control_output_overflow".into());
    }
    bytes.push(b'\n');
    let (completed, acknowledgement) = mpsc::sync_channel(1);
    writer
        .try_send(Output { bytes, completed })
        .map_err(|_| "control_output_failed")?;
    match acknowledgement.recv_timeout(Duration::from_millis(500)) {
        Ok(true) => Ok(()),
        _ => Err("control_output_failed".into()),
    }
}
