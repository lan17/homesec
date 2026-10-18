use ffmpeg_next::{Rational, Rescale};
use std::io::{BufRead, BufReader, Read, Write};
use std::thread;
use std::time::Duration;

/// An RTSP/TCP camera at the actual source boundary, without a media subprocess.
pub(crate) struct RtspCamera {
    pub(crate) url: String,
    calls: std::sync::Arc<std::sync::Mutex<Vec<String>>>,
    stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
    connections: std::sync::Arc<std::sync::Mutex<Vec<std::net::TcpStream>>>,
    worker: Option<thread::JoinHandle<()>>,
}

impl RtspCamera {
    pub(crate) fn start(codec: &'static str) -> Self {
        use std::sync::{
            Arc, Mutex,
            atomic::{AtomicBool, Ordering},
        };
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let url = format!("rtsp://{}/camera", listener.local_addr().unwrap());
        let calls = Arc::new(Mutex::new(Vec::new()));
        let stop = Arc::new(AtomicBool::new(false));
        let connections = Arc::new(Mutex::new(Vec::new()));
        let worker = {
            let calls = Arc::clone(&calls);
            let stop = Arc::clone(&stop);
            let connections = Arc::clone(&connections);
            let url = url.clone();
            thread::spawn(move || {
                let mut workers = Vec::new();
                while !stop.load(Ordering::Acquire) {
                    match listener.accept() {
                        Ok((socket, _)) => {
                            socket.set_nonblocking(false).unwrap();
                            connections
                                .lock()
                                .unwrap()
                                .push(socket.try_clone().unwrap());
                            let calls = Arc::clone(&calls);
                            let url = url.clone();
                            workers.push(thread::spawn(move || {
                                serve_rtsp_camera(socket, &url, codec, calls)
                            }));
                        }
                        Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                            thread::sleep(Duration::from_millis(5));
                        }
                        Err(error) => panic!("RTSP camera accept failed: {error}"),
                    }
                }
                for worker in workers {
                    worker.join().unwrap();
                }
            })
        };
        Self {
            url,
            calls,
            stop,
            connections,
            worker: Some(worker),
        }
    }

    pub(crate) fn count(&self, method: &str) -> usize {
        self.calls
            .lock()
            .unwrap()
            .iter()
            .filter(|call| *call == method)
            .count()
    }
}

impl Drop for RtspCamera {
    fn drop(&mut self) {
        self.stop.store(true, std::sync::atomic::Ordering::Release);
        for socket in self.connections.lock().unwrap().iter() {
            let _ = socket.shutdown(std::net::Shutdown::Both);
        }
        self.worker.take().unwrap().join().unwrap();
    }
}

pub(crate) fn annex_b_nals(data: &[u8]) -> Vec<Vec<u8>> {
    let mut starts = Vec::new();
    let mut index = 0;
    while index + 3 <= data.len() {
        let prefix = if data[index..].starts_with(&[0, 0, 0, 1]) {
            4
        } else if data[index..].starts_with(&[0, 0, 1]) {
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
        .map(|(index, &(_, start))| {
            let end = starts.get(index + 1).map_or(data.len(), |&(end, _)| end);
            data[start..end].to_vec()
        })
        .filter(|nal| !nal.is_empty())
        .collect()
}

fn base64_parameter(data: &[u8]) -> String {
    const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut result = String::new();
    for bytes in data.chunks(3) {
        let value = (u32::from(bytes[0]) << 16)
            | (u32::from(*bytes.get(1).unwrap_or(&0)) << 8)
            | u32::from(*bytes.get(2).unwrap_or(&0));
        for (index, shift) in [18, 12, 6, 0].into_iter().enumerate() {
            result.push(if index > bytes.len() {
                '='
            } else {
                char::from(ALPHABET[((value >> shift) & 63) as usize])
            });
        }
    }
    result
}

fn serve_rtsp_camera(
    socket: std::net::TcpStream,
    url: &str,
    codec: &str,
    calls: std::sync::Arc<std::sync::Mutex<Vec<String>>>,
) {
    use std::sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    };
    let partial_idr = codec == "PARTIAL_IDR";
    let multislice = partial_idr || codec == "DAMAGED_IDR";
    let hevc = codec == "H265_VALID";
    let transport_stream = codec == "MP2T_EAC3";
    let nals = annex_b_nals(if hevc {
        include_bytes!("../fixtures/inspection/hevc.h265")
    } else if multislice {
        include_bytes!("../fixtures/baseline-multislice-160x120.h264")
    } else {
        include_bytes!("../fixtures/baseline-160x120.h264")
    });
    let sps = nals
        .iter()
        .find(|nal| {
            if hevc {
                (nal[0] >> 1) & 63 == 33
            } else {
                nal[0] & 31 == 7
            }
        })
        .unwrap();
    let pps = nals
        .iter()
        .find(|nal| {
            if hevc {
                (nal[0] >> 1) & 63 == 34
            } else {
                nal[0] & 31 == 8
            }
        })
        .unwrap();
    let stale_pps = codec == "STALE_PPS";
    let changing_preview = matches!(codec, "LEVEL_CHANGE" | "CLOCK_BACKWARDS");
    // Keep both parameter IDs and syntax valid, but advertise pic_init_qp_minus26
    // as -2 instead of the actual encoder's -3. Real cameras can similarly
    // advertise stale encoder settings before sending their in-band parameters.
    let advertised_pps: &[u8] = if stale_pps {
        assert_eq!(pps.as_slice(), &[0x68, 0xce, 0x0f, 0xc8]);
        &[0x68, 0xce, 0x0b, 0xc8]
    } else {
        pps
    };
    let encoding = if codec == "H265" { "H265" } else { "H264" };
    let audio_only = codec == "AUDIO_ONLY";
    let broken_audio = codec == "H264_AAC_BROKEN";
    let copied_audio = match codec {
        "H264_AC3" => Some("ac3"),
        "H264_MP2" => Some("mp2"),
        "H264_MP3" => Some("mp3"),
        "H264_PCMA" => Some("pcm_alaw"),
        "H264_PCMU" => Some("pcm_mulaw"),
        "H264_OPUS" => Some("libopus"),
        _ => None,
    };
    let audio_clock = match copied_audio {
        Some("mp2" | "mp3") => 90_000,
        Some("pcm_alaw" | "pcm_mulaw") => 8_000,
        _ => 48_000,
    };
    let audio = codec == "H264_AAC" || audio_only || broken_audio || copied_audio.is_some();
    let mut sdp = format!(
        "v=0\r\no=- 1 1 IN IP4 127.0.0.1\r\ns=synthetic camera\r\nt=0 0\r\na=control:*\r\nm=video 0 RTP/AVP 101\r\na=rtpmap:101 {encoding}/90000\r\na=fmtp:101 packetization-mode=1;profile-level-id={:02x}{:02x}{:02x};sprop-parameter-sets={},{}\r\na=control:trackID=0\r\n",
        sps[1],
        sps[2],
        sps[3],
        base64_parameter(sps),
        base64_parameter(advertised_pps)
    );
    if hevc {
        let vps = nals.iter().find(|nal| (nal[0] >> 1) & 63 == 32).unwrap();
        sdp = format!(
            "v=0\r\no=- 1 1 IN IP4 127.0.0.1\r\ns=synthetic camera\r\nt=0 0\r\na=control:*\r\nm=video 0 RTP/AVP 101\r\na=rtpmap:101 H265/90000\r\na=fmtp:101 sprop-vps={};sprop-sps={};sprop-pps={}\r\na=control:trackID=0\r\n",
            base64_parameter(vps),
            base64_parameter(sps),
            base64_parameter(pps)
        );
    }
    if transport_stream {
        sdp = "v=0\r\no=- 1 1 IN IP4 127.0.0.1\r\ns=synthetic camera\r\nt=0 0\r\na=control:*\r\nm=video 0 RTP/AVP 33\r\na=rtpmap:33 MP2T/90000\r\na=control:trackID=0\r\n".into();
    }
    if audio_only {
        sdp = "v=0\r\no=- 1 1 IN IP4 127.0.0.1\r\ns=synthetic camera\r\nt=0 0\r\na=control:*\r\n"
            .into();
    }
    if codec == "NO_DIMENSIONS" {
        sdp = sdp
            .split_inclusive("\r\n")
            .filter(|line| !line.starts_with("a=fmtp:"))
            .collect();
    }
    if let Some(audio) = copied_audio {
        sdp.push_str(match audio {
            "ac3" => "m=audio 0 RTP/AVP 102\r\na=rtpmap:102 ac3/48000/2\r\na=control:trackID=1\r\n",
            "pcm_alaw" => {
                "m=audio 0 RTP/AVP 8\r\na=rtpmap:8 PCMA/8000/1\r\na=control:trackID=1\r\n"
            }
            "pcm_mulaw" => {
                "m=audio 0 RTP/AVP 0\r\na=rtpmap:0 PCMU/8000/1\r\na=control:trackID=1\r\n"
            }
            "libopus" => {
                "m=audio 0 RTP/AVP 102\r\na=rtpmap:102 opus/48000/2\r\na=control:trackID=1\r\n"
            }
            _ => "m=audio 0 RTP/AVP 14\r\na=rtpmap:14 MPA/90000\r\na=control:trackID=1\r\n",
        });
    } else if audio {
        sdp.push_str("m=audio 0 RTP/AVP 102\r\na=rtpmap:102 MPEG4-GENERIC/48000/2\r\na=fmtp:102 streamtype=5;profile-level-id=1;mode=AAC-hbr;config=1190;SizeLength=13;IndexLength=3;IndexDeltaLength=3\r\na=control:trackID=1\r\n");
    }
    let writer = Arc::new(Mutex::new(socket.try_clone().unwrap()));
    let streaming_stop = Arc::new(AtomicBool::new(false));
    let mut streaming = None;
    let mut reader = BufReader::new(socket);
    let mut video_channel = 0;
    let mut audio_channel = 2;
    loop {
        if reader
            .fill_buf()
            .ok()
            .and_then(|bytes| bytes.first())
            .copied()
            == Some(b'$')
        {
            let mut header = [0; 4];
            if reader.read_exact(&mut header).is_err() {
                break;
            }
            let mut rtcp = vec![0; usize::from(u16::from_be_bytes([header[2], header[3]]))];
            if reader.read_exact(&mut rtcp).is_err() {
                break;
            }
            continue;
        }
        let mut request = String::new();
        if reader.read_line(&mut request).unwrap_or(0) == 0 {
            break;
        }
        let Some(method) = request.split_whitespace().next().map(str::to_owned) else {
            break;
        };
        let mut cseq = String::new();
        let mut channel = 0_u8;
        loop {
            let mut header = String::new();
            if reader.read_line(&mut header).unwrap_or(0) == 0 {
                break;
            }
            if header == "\r\n" {
                break;
            }
            if let Some((name, value)) = header.split_once(':') {
                if name.eq_ignore_ascii_case("cseq") {
                    cseq = value.trim().to_owned();
                }
                if name.eq_ignore_ascii_case("transport")
                    && let Some(channels) = value.split("interleaved=").nth(1)
                {
                    channel = channels.split('-').next().unwrap().parse().unwrap();
                }
            }
        }
        if method == "SETUP" {
            if request
                .split_whitespace()
                .nth(1)
                .unwrap()
                .ends_with("trackID=1")
            {
                audio_channel = channel;
            } else {
                video_channel = channel;
            }
        }
        calls.lock().unwrap().push(method.clone());
        if (matches!(
            codec,
            "SESSION_LIMIT" | "SESSION_PHRASE" | "SESSION_MAX_CONNECTIONS" | "GENERIC_503"
        ) && method == "SETUP")
            || (codec == "AUTH_FAILURE" && method == "DESCRIBE")
        {
            let status = match codec {
                "SESSION_LIMIT" => "453 Not Enough Bandwidth",
                "SESSION_PHRASE" => "503 Too Many Sessions",
                "SESSION_MAX_CONNECTIONS" => "503 MaX NuMbEr Of CoNnEcTiOnS ReAcHeD",
                "GENERIC_503" => "503 Service Unavailable",
                _ => "401 Unauthorized",
            };
            let response =
                format!("RTSP/1.0 {status}\r\nCSeq: {cseq}\r\nContent-Length: 0\r\n\r\n");
            let _ = writer.lock().unwrap().write_all(response.as_bytes());
            continue;
        }
        if codec == "STALL" && method == "DESCRIBE" {
            continue;
        }
        let mut headers = format!("CSeq: {cseq}\r\n");
        let mut body = "";
        match method.as_str() {
            "OPTIONS" => headers.push_str("Public: OPTIONS, DESCRIBE, SETUP, PLAY, TEARDOWN, GET_PARAMETER\r\n"),
            "DESCRIBE" => {
                headers.push_str(&format!("Content-Type: application/sdp\r\nContent-Base: {url}/\r\n"));
                body = &sdp;
            }
            "SETUP" => headers.push_str(&format!("Session: fixture;timeout=60\r\nTransport: RTP/AVP/TCP;unicast;interleaved={channel}-{}\r\n", channel + 1)),
            "PLAY" => {
                headers.push_str(&format!("Session: fixture\r\nRange: npt=0.000-\r\nRTP-Info: url={url}/trackID=0;seq=1;rtptime=0"));
                if audio { headers.push_str(&format!(",url={url}/trackID=1;seq=1;rtptime={}", audio_clock / 4)); }
                headers.push_str("\r\n");
            },
            "TEARDOWN" | "GET_PARAMETER" => headers.push_str("Session: fixture\r\n"),
            _ => panic!("Unexpected RTSP method: {method}"),
        }
        let response = format!(
            "RTSP/1.0 200 OK\r\n{headers}Content-Length: {}\r\n\r\n{body}",
            body.len()
        );
        if writer
            .lock()
            .unwrap()
            .write_all(response.as_bytes())
            .is_err()
        {
            break;
        }
        if method == "TEARDOWN" {
            break;
        }
        if method == "PLAY" && codec != "STALL_PLAY" && streaming.is_none() {
            let writer = Arc::clone(&writer);
            let stop = Arc::clone(&streaming_stop);
            let nals = nals.clone();
            let codec = codec.to_owned();
            streaming = Some(thread::spawn(move || {
                if transport_stream {
                    stream_mpegts(writer, stop, video_channel);
                    return;
                }
                let audio_worker = if audio {
                    // Both sender reports share one NTP epoch, while the first
                    // AAC sample deliberately begins 250ms after video time zero.
                    if !sender_report(&writer, video_channel + 1, 0x12345678)
                        || !sender_report(&writer, audio_channel + 1, 0x87654321)
                    {
                        return;
                    }
                    let audio_writer = Arc::clone(&writer);
                    let audio_stop = Arc::clone(&stop);
                    Some(thread::spawn(move || {
                        if let Some(codec) = copied_audio {
                            stream_copy_audio(audio_writer, audio_stop, audio_channel, codec);
                        } else {
                            stream_aac(audio_writer, audio_stop, audio_channel, broken_audio);
                        }
                    }))
                } else {
                    None
                };
                if audio_only {
                    while !stop.load(Ordering::Acquire) {
                        thread::sleep(Duration::from_millis(5));
                    }
                } else if hevc {
                    stream_h265(writer, Arc::clone(&stop), video_channel, nals);
                } else {
                    stream_h264(
                        writer,
                        Arc::clone(&stop),
                        video_channel,
                        nals,
                        partial_idr,
                        stale_pps || changing_preview,
                        &codec,
                    );
                }
                stop.store(true, Ordering::Release);
                if let Some(audio_worker) = audio_worker {
                    audio_worker.join().unwrap();
                }
            }));
        }
    }
    streaming_stop.store(true, Ordering::Release);
    let _ = reader.get_ref().shutdown(std::net::Shutdown::Both);
    if let Some(streaming) = streaming {
        streaming.join().unwrap();
    }
    calls.lock().unwrap().push("DISCONNECTED".to_owned());
}

fn stream_h264(
    writer: std::sync::Arc<std::sync::Mutex<std::net::TcpStream>>,
    stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
    channel: u8,
    nals: Vec<Vec<u8>>,
    partial_idr: bool,
    inband_parameters: bool,
    mode: &str,
) {
    let mut frames: Vec<Vec<Vec<u8>>> = Vec::new();
    for nal in nals {
        match nal[0] & 31 {
            9 => frames.push(Vec::new()),
            7 | 8 if !inband_parameters => {} // Exercise parameters supplied only in SDP.
            _ => frames.last_mut().unwrap().push(nal),
        }
    }
    let mut sequence = if partial_idr { 2_u16 } else { 1_u16 };
    let mut timestamp = 0_u32;
    for (frame_index, frame) in frames.iter().cycle().enumerate() {
        if stop.load(std::sync::atomic::Ordering::Acquire) {
            return;
        }
        if mode == "LATE_CLOSE" && frame_index == 30 {
            let _ = writer.lock().unwrap().shutdown(std::net::Shutdown::Both);
            return;
        }
        if mode == "CLOCK_BACKWARDS" && frame_index == 30 {
            timestamp -= 18_000;
        }
        if mode == "CLOCK_RESET" && frame_index >= 30 {
            timestamp = 0;
        }
        let mut payloads = Vec::new();
        let first_slice = frame.iter().position(|nal| nal[0] & 31 == 5);
        for (index, original) in frame.iter().enumerate() {
            let mut nal = original.clone();
            if mode == "NO_IDR" && nal[0] & 31 == 5 {
                continue;
            }
            if mode == "LEVEL_CHANGE" && frame_index >= 30 && nal[0] & 31 == 7 {
                nal[3] = 42;
            }
            if partial_idr && timestamp == 0 && Some(index) == first_slice {
                continue;
            }
            if mode == "DAMAGED_IDR" && frame_index >= 30 && Some(index) == first_slice {
                continue;
            }
            if nal.len() <= 1000 {
                payloads.push(nal);
            } else {
                let chunks: Vec<_> = nal[1..].chunks(998).collect();
                for (index, chunk) in chunks.iter().enumerate() {
                    let mut payload = vec![
                        (nal[0] & 0xe0) | 28,
                        (nal[0] & 31)
                            | if index == 0 { 0x80 } else { 0 }
                            | if index + 1 == chunks.len() { 0x40 } else { 0 },
                    ];
                    payload.extend_from_slice(chunk);
                    payloads.push(payload);
                }
            }
        }
        for (index, payload) in payloads.iter().enumerate() {
            let mut packet = vec![
                0x80,
                101 | if index + 1 == payloads.len() { 0x80 } else { 0 },
            ];
            packet.extend_from_slice(&sequence.to_be_bytes());
            packet.extend_from_slice(&timestamp.to_be_bytes());
            packet.extend_from_slice(&0x12345678_u32.to_be_bytes());
            packet.extend_from_slice(payload);
            let mut interleaved = vec![b'$', channel];
            interleaved.extend_from_slice(&(packet.len() as u16).to_be_bytes());
            interleaved.extend_from_slice(&packet);
            if writer.lock().unwrap().write_all(&interleaved).is_err() {
                return;
            }
            sequence = sequence.wrapping_add(1);
        }
        timestamp = timestamp.wrapping_add(9000);
        thread::sleep(Duration::from_millis(100));
    }
}

/// Actual AAC-LC compressed access units from the committed synthetic tone.
pub(crate) fn aac_access_units() -> Vec<Vec<u8>> {
    ffmpeg_next::init().unwrap();
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/recording/h264-aac-bframes.mp4");
    let mut input = ffmpeg_next::format::input(&path).unwrap();
    input
        .packets()
        .filter(|(stream, _)| stream.parameters().medium() == ffmpeg_next::media::Type::Audio)
        .map(|(_, packet)| packet.data().unwrap().to_vec())
        .collect()
}

fn sender_report(
    writer: &std::sync::Arc<std::sync::Mutex<std::net::TcpStream>>,
    channel: u8,
    ssrc: u32,
) -> bool {
    let mut report = vec![0x80, 200, 0, 6];
    report.extend_from_slice(&ssrc.to_be_bytes());
    report.extend_from_slice(&(3_900_000_000_u64 << 32).to_be_bytes());
    report.extend_from_slice(&[0; 12]); // RTP timestamp=0, packet/octet counts=0.
    write_interleaved(writer, channel, &report)
}

fn write_interleaved(
    writer: &std::sync::Arc<std::sync::Mutex<std::net::TcpStream>>,
    channel: u8,
    packet: &[u8],
) -> bool {
    let mut interleaved = vec![b'$', channel];
    interleaved.extend_from_slice(&(packet.len() as u16).to_be_bytes());
    interleaved.extend_from_slice(packet);
    writer.lock().unwrap().write_all(&interleaved).is_ok()
}

fn stream_aac(
    writer: std::sync::Arc<std::sync::Mutex<std::net::TcpStream>>,
    stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
    channel: u8,
    broken: bool,
) {
    let packets = if broken {
        vec![vec![0xff; 8]]
    } else {
        aac_access_units()
    };
    let mut sequence = 1_u16;
    let mut timestamp = 12000_u32;
    for access_unit in packets.iter().cycle() {
        if stop.load(std::sync::atomic::Ordering::Acquire) {
            return;
        }
        assert!(access_unit.len() < 8192);
        let mut packet = vec![0x80, 0x80 | 102];
        packet.extend_from_slice(&sequence.to_be_bytes());
        packet.extend_from_slice(&timestamp.to_be_bytes());
        packet.extend_from_slice(&0x87654321_u32.to_be_bytes());
        packet.extend_from_slice(&16_u16.to_be_bytes()); // AU-headers-length in bits.
        packet.extend_from_slice(&((access_unit.len() as u16) << 3).to_be_bytes());
        packet.extend_from_slice(access_unit);
        if !write_interleaved(&writer, channel, &packet) {
            return;
        }
        sequence = sequence.wrapping_add(1);
        timestamp = timestamp.wrapping_add(1024);
        thread::sleep(Duration::from_micros(21_333));
    }
}

fn stream_h265(
    writer: std::sync::Arc<std::sync::Mutex<std::net::TcpStream>>,
    stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
    channel: u8,
    nals: Vec<Vec<u8>>,
) {
    let mut frames = Vec::<Vec<Vec<u8>>>::new();
    for nal in nals {
        let kind = (nal[0] >> 1) & 63;
        if kind > 31 {
            continue;
        }
        if nal.get(2).is_some_and(|byte| byte & 0x80 != 0) || frames.is_empty() {
            frames.push(Vec::new());
        }
        frames.last_mut().unwrap().push(nal);
    }
    let mut sequence = 1_u16;
    let mut timestamp = 0_u32;
    for frame in frames.iter().cycle() {
        if stop.load(std::sync::atomic::Ordering::Acquire) {
            return;
        }
        let mut payloads = Vec::new();
        for nal in frame {
            if nal.len() <= 1000 {
                payloads.push(nal.clone());
            } else {
                let chunks = nal[2..].chunks(997).collect::<Vec<_>>();
                for (index, chunk) in chunks.iter().enumerate() {
                    let mut payload = vec![
                        (nal[0] & 0x81) | (49 << 1),
                        nal[1],
                        ((nal[0] >> 1) & 63)
                            | if index == 0 { 0x80 } else { 0 }
                            | if index + 1 == chunks.len() { 0x40 } else { 0 },
                    ];
                    payload.extend_from_slice(chunk);
                    payloads.push(payload);
                }
            }
        }
        for (index, payload) in payloads.iter().enumerate() {
            let mut packet = vec![
                0x80,
                101 | if index + 1 == payloads.len() { 0x80 } else { 0 },
            ];
            packet.extend_from_slice(&sequence.to_be_bytes());
            packet.extend_from_slice(&timestamp.to_be_bytes());
            packet.extend_from_slice(&0x12345678_u32.to_be_bytes());
            packet.extend_from_slice(payload);
            if !write_interleaved(&writer, channel, &packet) {
                return;
            }
            sequence = sequence.wrapping_add(1);
        }
        timestamp = timestamp.wrapping_add(7500);
        thread::sleep(Duration::from_millis(83));
    }
}

/// Complete AC-3 frames (RFC 4184) and MPEG audio frames (RFC 2250).
/// The committed fixture supplies compressed AUs; no executable makes media.
fn stream_copy_audio(
    writer: std::sync::Arc<std::sync::Mutex<std::net::TcpStream>>,
    stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
    channel: u8,
    codec: &str,
) {
    let clock = match codec {
        "mp2" | "mp3" => 90_000,
        "pcm_alaw" | "pcm_mulaw" => 8_000,
        _ => 48_000,
    };
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(format!(
        "tests/fixtures/inspection/h264-{codec}.{}",
        if codec == "libopus" { "mp4" } else { "mov" }
    ));
    let mut input = ffmpeg_next::format::input(&path).unwrap();
    let packets = input
        .packets()
        .filter_map(|(stream, packet)| {
            (stream.parameters().medium() == ffmpeg_next::media::Type::Audio).then(|| {
                // G.711 bytes are independent mono samples with no embedded
                // rate; publish them at the standard RTP 8 kHz clock.
                let ticks = if matches!(codec, "pcm_alaw" | "pcm_mulaw") {
                    packet.size() as i64
                } else {
                    packet
                        .duration()
                        .rescale(stream.time_base(), Rational(1, clock))
                };
                (
                    packet.data().unwrap().to_vec(),
                    u32::try_from(ticks).unwrap(),
                )
            })
        })
        .collect::<Vec<_>>();
    assert!(!packets.is_empty());
    let mut sequence = 1_u16;
    let mut timestamp = clock as u32 / 4;
    for (access_unit, ticks) in packets.iter().cycle() {
        if stop.load(std::sync::atomic::Ordering::Acquire) {
            return;
        }
        let payload = match codec {
            "mp2" | "mp3" => 14,
            "pcm_alaw" => 8,
            "pcm_mulaw" => 0,
            _ => 102,
        };
        let mut packet = vec![0x80, 0x80 | payload];
        packet.extend_from_slice(&sequence.to_be_bytes());
        packet.extend_from_slice(&timestamp.to_be_bytes());
        packet.extend_from_slice(&0x87654321_u32.to_be_bytes());
        if codec == "ac3" {
            packet.extend_from_slice(&[0, 1]);
        }
        // FT=0, one complete frame.
        else if matches!(codec, "mp2" | "mp3") {
            packet.extend_from_slice(&[0; 4]);
        } // MBZ and zero fragment offset.
        packet.extend_from_slice(access_unit);
        if !write_interleaved(&writer, channel, &packet) {
            return;
        }
        sequence = sequence.wrapping_add(1);
        timestamp = timestamp.wrapping_add(*ticks);
        thread::sleep(Duration::from_micros(
            u64::from(*ticks) * 1_000_000 / clock as u64,
        ));
    }
}

/// Standard RTP payload 33 carries complete 188-byte MPEG-TS packets.
fn stream_mpegts(
    writer: std::sync::Arc<std::sync::Mutex<std::net::TcpStream>>,
    stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
    channel: u8,
) {
    let bytes = include_bytes!("../fixtures/inspection/h264-eac3.ts");
    assert_eq!(bytes.len() % 188, 0);
    let chunks = bytes.chunks(7 * 188).collect::<Vec<_>>();
    let interval = 2_000_000 / chunks.len() as u64;
    let mut sequence = 1_u16;
    let mut timestamp = 0_u32;
    for payload in chunks.iter().cycle() {
        if stop.load(std::sync::atomic::Ordering::Acquire) {
            return;
        }
        let mut packet = vec![0x80, 33];
        packet.extend_from_slice(&sequence.to_be_bytes());
        packet.extend_from_slice(&timestamp.to_be_bytes());
        packet.extend_from_slice(&0x12345678_u32.to_be_bytes());
        packet.extend_from_slice(payload);
        if !write_interleaved(&writer, channel, &packet) {
            return;
        }
        sequence = sequence.wrapping_add(1);
        timestamp = timestamp.wrapping_add((interval * 90_000 / 1_000_000) as u32);
        thread::sleep(Duration::from_micros(interval));
    }
}
