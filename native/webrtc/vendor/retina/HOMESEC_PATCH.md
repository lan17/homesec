# HomeSec RTSP control-message bound

This directory contains the exact published Retina 0.4.20 source, except for
the two narrowly patched files below. HomeSec retains upstream authentication,
session maintenance, depacketization, and error handling.

- Upstream: https://github.com/scottlamb/retina
- Published archive: https://static.crates.io/crates/retina/retina-0.4.20.crate
- Archive SHA-256: `0e0eb740f743e678e071628ff6bf84ed5ed03df879997cc3d43b5afef64aff93`
- Upstream revision from the archive's `.cargo_vcs_info.json`:
  `dead7c664569eb0adce5e7c42591f0b12e060988`
- License: MIT OR Apache-2.0. License texts are included from that revision;
  the published archive omitted them. Original copyright/SPDX notices remain.

Local changes:

1. `src/rtsp/parse.rs`: reuse `ParserBuilder::max_message_size`, but apply its
   head-plus-body bound incrementally. Parse each status/request/header line
   through a view limited to its remaining byte budget, before allocating owned
   strings. Previously parsed headers remain charged. Reject an oversized
   announced body before receiving it. Add a crate-private remaining-read budget.
2. `src/tokio.rs`: set the fixed internal limit to **256 KiB per RTSP control
   message**. Bound reservation and vectored socket-read slices before each
   read. Reclaim consumed idle CRLF separators. No new HomeSec configuration or
   upstream public client API is added.

The limit covers requests and responses, including SDP bodies and keepalive
responses. Each control message gets a fresh budget; bytes belonging to later
pipelined messages do not count against it. Interleaved RTP/RTCP payloads retain
their protocol-defined 16-bit length bound and do not consume a control-message
budget. Header object/container overhead is bounded by the accepted wire bytes,
but is additional to that wire-byte count. The ring buffer rounds capacity to a
power of two; existing media marks may retain earlier media independently of
this control-message bound.

Regression coverage lives in `../../tests/rtsp_control_limit.rs`, and runs with
HomeSec's normal native test suite. To update this vendor copy, compare the two
patched files against the verified archive, port or remove the patch, and retain
those tests. This cap can be removed when an upstream release provides equivalent
incremental parsing and pre-read limits through the client connection.
