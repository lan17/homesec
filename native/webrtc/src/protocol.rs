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
        ffmpeg_args: Vec<String>,
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
