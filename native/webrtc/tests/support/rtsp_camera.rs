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
    let nals = annex_b_nals(if multislice {
        include_bytes!("../fixtures/baseline-multislice-160x120.h264")
    } else {
        include_bytes!("../fixtures/baseline-160x120.h264")
    });
    let sps = nals.iter().find(|nal| nal[0] & 31 == 7).unwrap();
    let pps = nals.iter().find(|nal| nal[0] & 31 == 8).unwrap();
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
    let audio = codec == "H264_AAC";
    let mut sdp = format!(
        "v=0\r\no=- 1 1 IN IP4 127.0.0.1\r\ns=synthetic camera\r\nt=0 0\r\na=control:*\r\nm=video 0 RTP/AVP 101\r\na=rtpmap:101 {encoding}/90000\r\na=fmtp:101 packetization-mode=1;profile-level-id={:02x}{:02x}{:02x};sprop-parameter-sets={},{}\r\na=control:trackID=0\r\n",
        sps[1],
        sps[2],
        sps[3],
        base64_parameter(sps),
        base64_parameter(advertised_pps)
    );
    if audio {
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
                if audio { headers.push_str(&format!(",url={url}/trackID=1;seq=1;rtptime=12000")); }
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
                        stream_aac(audio_writer, audio_stop, audio_channel)
                    }))
                } else {
                    None
                };
                stream_h264(
                    writer,
                    Arc::clone(&stop),
                    video_channel,
                    nals,
                    partial_idr,
                    stale_pps || changing_preview,
                    &codec,
                );
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
) {
    let packets = aac_access_units();
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
