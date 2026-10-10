//! Boundary tests for the vendored Retina control-message allocation guard.

use retina::client::{Session, SessionOptions};
use retina::inputs::{Input, Split};
use retina::rtsp::msg::Message;
use retina::rtsp::parse::{FeedError, Parser};
use std::io::{Read, Write};
use std::net::TcpListener;
use std::thread;
use std::time::Duration;
use url::Url;

const CONTROL_LIMIT: usize = 256 * 1024;

fn assert_too_large(error: FeedError) {
    let FeedError::Invalid(error) = error else {
        panic!("expected oversized control-message refusal");
    };
    assert!(error.context.contains(&"message-too-large"));
}

#[test]
fn unterminated_single_header_is_rejected_at_the_byte_limit() {
    // Given: a partial header whose bytes exactly exhaust the total budget.
    let head = b"RTSP/1.0 200 OK\r\n";
    let mut parser = Parser::builder().max_message_size(64).build();
    let mut input = head.as_slice();
    assert!(matches!(
        parser.feed(&mut input),
        Err(FeedError::Incomplete(_))
    ));
    assert!(input.is_empty());
    let mut line = b"X-Long: ".to_vec();
    line.resize(64 - head.len(), b'x');

    // When: the line still lacks its terminator at the limit.
    let error = parser.feed(&mut line.as_slice()).unwrap_err();

    // Then: the parser refuses immediately without waiting for more bytes.
    assert_too_large(error);
}

#[test]
fn complete_oversized_header_and_status_lines_are_rejected() {
    // Given: complete lines supplied in one read, beyond the allowed prefix.
    let mut status = b"RTSP/1.0 200 ".to_vec();
    status.extend([b'x'; 128]);
    status.extend_from_slice(b"\r\n\r\n");
    let mut header = b"RTSP/1.0 200 OK\r\nX-Long: ".to_vec();
    header.extend([b'x'; 128]);
    header.extend_from_slice(b"\r\n\r\n");

    // When: either line is parsed with a smaller control-message budget.
    for data in [status, header] {
        let error = Parser::builder()
            .max_message_size(64)
            .build()
            .feed(&mut data.as_slice())
            .unwrap_err();

        // Then: a complete terminator cannot bypass the pre-allocation bound.
        assert_too_large(error);
    }
}

#[test]
fn separately_received_valid_headers_share_one_total_budget() {
    // Given: valid short headers whose old network bytes are already consumed.
    let mut parser = Parser::builder().max_message_size(64).build();
    let mut input = &b"RTSP/1.0 200 OK\r\n"[..];
    assert!(matches!(
        parser.feed(&mut input),
        Err(FeedError::Incomplete(_))
    ));
    let line = &b"X-Repeat: yes\r\n"[..];
    for _ in 0..3 {
        let mut input = line;
        assert!(matches!(
            parser.feed(&mut input),
            Err(FeedError::Incomplete(_))
        ));
        assert!(input.is_empty());
    }

    // When: another individually valid header exceeds the remaining budget.
    let error = parser.feed(&mut &line[..]).unwrap_err();

    // Then: aggregate retained headers cannot grow without a bound.
    assert_too_large(error);
}

#[test]
fn oversized_announced_body_is_rejected_before_body_bytes_arrive() {
    // Given: a valid response head announcing a body beyond the total limit.
    let data = &b"RTSP/1.0 200 OK\r\nContent-Length: 64\r\n\r\n"[..];
    let mut parser = Parser::builder().max_message_size(64).build();

    // When: only the head has arrived.
    let error = parser.feed(&mut &data[..]).unwrap_err();

    // Then: the parser refuses instead of reserving or waiting for that body.
    assert_too_large(error);
}

#[test]
fn slowly_received_body_succeeds_at_the_exact_total_limit() {
    // Given: a response whose head and body exactly fit the budget.
    let head = &b"RTSP/1.0 200 OK\r\nContent-Length: 8\r\n\r\n"[..];
    let body = b"12345678";
    let mut parser = Parser::builder()
        .max_message_size(head.len() + body.len())
        .build();
    let mut input = head;
    assert!(matches!(
        parser.feed(&mut input),
        Err(FeedError::Incomplete(_))
    ));
    assert!(input.is_empty());

    // When: the body arrives one byte at a time, remaining buffered until complete.
    for received in 1..=body.len() {
        let mut input = &body[..received];
        let result = parser.feed(&mut input);

        // Then: incomplete bodies stay bounded, and the complete body is intact.
        if received < body.len() {
            let Err(FeedError::Incomplete(error)) = result else {
                panic!("expected incomplete body");
            };
            assert_eq!(error.needed.unwrap().get(), body.len() - received);
            assert_eq!(input.len(), received);
        } else {
            let (message, actual) = result.unwrap().unwrap();
            assert!(matches!(message, Message::Response(_)));
            assert_eq!(actual, body);
            assert!(input.is_empty());
        }
    }
}

#[test]
fn pipelined_responses_and_maximum_interleaved_packet_have_separate_limits() {
    // Given: one input read larger than the control limit, with three messages.
    let response = b"RTSP/1.0 200 OK\r\nCSeq: 1\r\n\r\n";
    let payload = vec![0x5a; u16::MAX as usize];
    let mut data = response.to_vec();
    data.extend_from_slice(b"$\x00\xff\xff");
    data.extend_from_slice(&payload);
    data.extend_from_slice(response);
    let mut input = data.as_slice();
    let mut parser = Parser::builder().max_message_size(response.len()).build();

    // When: the parser handles each complete message without losing boundaries.
    let first = parser.feed(&mut input).unwrap().unwrap();
    let interleaved = parser.feed(&mut input).unwrap().unwrap();
    let last = parser.feed(&mut input).unwrap().unwrap();

    // Then: the control cap applies per response and preserves the full media packet.
    assert!(matches!(first.0, Message::Response(_)));
    assert!(first.1.is_empty());
    assert!(matches!(interleaved.0, Message::Data(_)));
    assert_eq!(interleaved.1, payload.as_slice());
    assert!(matches!(last.0, Message::Response(_)));
    assert!(last.1.is_empty());
    assert!(input.is_empty());
}

#[test]
fn control_message_limit_works_across_ring_buffer_splits() {
    // Given: an exactly fitting response at every possible ring-buffer split.
    let data = b"RTSP/1.0 200 OK\r\nContent-Length: 4\r\n\r\ndata";
    for split in 0..=data.len() {
        let mut input = Split::new(&data[..split], &data[split..]);
        let mut parser = Parser::builder().max_message_size(data.len()).build();

        // When: the head/body crosses either half of the ring.
        let (message, body) = parser.feed(&mut input).unwrap().unwrap();

        // Then: bounded views preserve the parser's split-input support.
        assert!(matches!(message, Message::Response(_)));
        assert_eq!(body.to_owned(), b"data");
        assert!(input.is_empty());
    }
}

fn describe_response_chunks(chunks: Vec<Vec<u8>>) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let server = thread::spawn(move || {
        let (mut socket, _) = listener.accept().unwrap();
        socket
            .set_read_timeout(Some(Duration::from_secs(2)))
            .unwrap();
        socket
            .set_write_timeout(Some(Duration::from_secs(2)))
            .unwrap();
        let mut request = [0; 4096];
        assert!(socket.read(&mut request).unwrap() > 0);
        for chunk in chunks {
            if socket.write_all(&chunk).is_err() {
                break; // The bounded client closes after its terminal refusal.
            }
        }
    });
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let error = runtime.block_on(async {
        let url = Url::parse(&format!("rtsp://{address}/test")).unwrap();
        tokio::time::timeout(
            Duration::from_secs(3),
            Session::describe(url, SessionOptions::default()),
        )
        .await
        .expect("response guard must reject before the I/O deadline")
        .err()
        .expect("oversized response must fail")
        .to_string()
    });
    server.join().unwrap();
    error
}

#[test]
fn client_connection_bounds_unterminated_header_socket_reads() {
    // Given: an actual RTSP socket sends a header larger than the fixed client cap.
    let mut header = b"RTSP/1.0 200 OK\r\nX-Long: ".to_vec();
    header.resize(CONTROL_LIMIT + 8192, b'x');

    // When: the session's production connection reads fragmented TCP data.
    let error = describe_response_chunks(header.chunks(4096).map(Vec::from).collect());

    // Then: it rejects an incomplete header with the parser's bounded failure.
    assert!(error.contains("message-too-large"), "{error}");
}

#[test]
fn client_connection_rejects_announced_oversized_body_without_receiving_it() {
    // Given: an actual RTSP response announces more bytes than the fixed cap.
    let head = format!("RTSP/1.0 200 OK\r\nCSeq: 1\r\nContent-Length: {CONTROL_LIMIT}\r\n\r\n");

    // When: the server sends only that head.
    let error = describe_response_chunks(vec![head.into_bytes()]);

    // Then: no body arrival or timeout is required to trigger the refusal.
    assert!(error.contains("message-too-large"), "{error}");
}
