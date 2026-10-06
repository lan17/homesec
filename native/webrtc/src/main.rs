//! Camera-local media worker with bounded media buffers. str0m owns the WebRTC protocols.
#[cfg(test)]
#[path = "../tests/support/burst_rtp.rs"]
mod burst_rtp;
mod decode;
mod engine;
mod motion;
mod motion_worker;
mod protocol;
mod rtp;
mod rtp_receiver;
mod rtsp;
#[cfg(test)]
#[path = "../tests/support/rtsp_camera.rs"]
mod rtsp_camera;

use clap::Parser;
use std::net::IpAddr;

#[derive(Parser)]
struct Options {
    #[arg(long, required_unless_present = "motion")]
    advertised_ip: Option<IpAddr>,
    #[arg(long)]
    motion: bool,
    #[arg(long, default_value_t = 8189)]
    udp_port_start: u16,
    #[arg(long, default_value_t = 8199)]
    udp_port_end: u16,
    #[arg(long, default_value_t = 4)]
    max_viewers: usize,
    #[arg(long, default_value_t = 15.0)]
    negotiation_timeout_s: f64,
    #[arg(long, default_value_t = 3600.0)]
    max_session_duration_s: f64,
}

fn main() {
    // Do not install tracing: SDP, candidates and camera command lines are private.
    str0m::crypto::from_feature_flags().install_process_default();
    let options = Options::parse();
    let result = if options.motion {
        motion_worker::run()
    } else {
        engine::run(options)
    };
    if result.is_err() {
        eprintln!("homesec-webrtc: worker_failed");
        std::process::exit(1);
    }
}
