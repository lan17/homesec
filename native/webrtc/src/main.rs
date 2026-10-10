//! Camera-local media worker with bounded media buffers. str0m owns the WebRTC protocols.
#[cfg(test)]
#[path = "../tests/support/burst_rtp.rs"]
mod burst_rtp;
mod camera_input;
mod decode;
mod engine;
mod motion;
mod motion_worker;
mod protocol;
mod recording;
mod rtp;
mod rtp_receiver;
mod rtsp;
#[cfg(test)]
#[path = "../tests/support/rtsp_camera.rs"]
mod rtsp_camera;
mod shared_worker;

use clap::Parser;
use std::net::IpAddr;

#[derive(Parser)]
struct Options {
    #[arg(long, required_unless_present_any = ["motion", "shared"])]
    advertised_ip: Option<IpAddr>,
    #[arg(long)]
    motion: bool,
    #[arg(long, conflicts_with = "motion")]
    shared: bool,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_motion_and_recording_can_start_without_a_preview_address() {
        // Given: A camera source needs motion/recording but preview is disabled.
        let arguments = ["homesec-webrtc", "--shared"];
        // When: The actual executable command-line contract is parsed.
        let options = Options::try_parse_from(arguments).unwrap();
        // Then: Shared mode starts without requiring a media address.
        assert!(options.shared);
        assert!(options.advertised_ip.is_none());
    }

    #[test]
    fn legacy_preview_still_requires_an_address_and_worker_modes_are_exclusive() {
        // Given: Preview-only startup and conflicting worker-mode arguments.
        let preview = ["homesec-webrtc"];
        let conflicting = ["homesec-webrtc", "--motion", "--shared"];
        // When: The existing CLI and the new shared worker option are parsed.
        let preview_result = Options::try_parse_from(preview);
        let conflicting_result = Options::try_parse_from(conflicting);
        // Then: Preview needs its address and cannot silently pick a worker mode.
        assert!(preview_result.is_err());
        assert!(conflicting_result.is_err());
    }
}
