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
    let nals = annex_b_nals(include_bytes!("../fixtures/baseline-160x120.h264"));
    let sps = nals.iter().find(|nal| nal[0] & 31 == 7).unwrap();
    let pps = nals.iter().find(|nal| nal[0] & 31 == 8).unwrap();
    let encoding = if codec == "H265" { "H265" } else { "H264" };
    let sdp = format!(
        "v=0\r\no=- 1 1 IN IP4 127.0.0.1\r\ns=synthetic camera\r\nt=0 0\r\na=control:*\r\nm=video 0 RTP/AVP 101\r\na=rtpmap:101 {encoding}/90000\r\na=fmtp:101 packetization-mode=1;profile-level-id={:02x}{:02x}{:02x};sprop-parameter-sets={},{}\r\na=control:trackID=0\r\n",
        sps[1],
        sps[2],
        sps[3],
        base64_parameter(sps),
        base64_parameter(pps)
    );
    let writer = Arc::new(Mutex::new(socket.try_clone().unwrap()));
    let streaming_stop = Arc::new(AtomicBool::new(false));
    let mut streaming = None;
    let mut reader = BufReader::new(socket);
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
            "PLAY" => headers.push_str(&format!("Session: fixture\r\nRange: npt=0.000-\r\nRTP-Info: url={url}/trackID=0;seq=1;rtptime=0\r\n")),
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
            streaming = Some(thread::spawn(move || {
                stream_h264(writer, stop, channel, nals)
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
) {
    let mut frames: Vec<Vec<Vec<u8>>> = Vec::new();
    for nal in nals {
        match nal[0] & 31 {
            9 => frames.push(Vec::new()),
            7 | 8 => {} // Deliberately exercise parameter sets supplied only in SDP.
            _ => frames.last_mut().unwrap().push(nal),
        }
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
