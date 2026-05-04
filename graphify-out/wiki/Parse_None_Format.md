# Parse None Format

> 22 nodes · cohesion 0.12

## Key Concepts

- **TestParsePassivePorts** (13 connections) — `tests/homesec/test_ftp_source.py`
- **_parse_passive_ports()** (11 connections) — `src/homesec/sources/ftp.py`
- **ftp.py** (5 connections) — `src/homesec/sources/ftp.py`
- **.test_parse_range_format()** (4 connections) — `tests/homesec/test_ftp_source.py`
- **.test_parse_comma_format()** (3 connections) — `tests/homesec/test_ftp_source.py`
- **.test_parse_comma_with_whitespace()** (3 connections) — `tests/homesec/test_ftp_source.py`
- **.test_parse_empty_returns_none()** (3 connections) — `tests/homesec/test_ftp_source.py`
- **.test_parse_invalid_range_raises()** (3 connections) — `tests/homesec/test_ftp_source.py`
- **.test_parse_none_returns_none()** (3 connections) — `tests/homesec/test_ftp_source.py`
- **.test_parse_single_port()** (3 connections) — `tests/homesec/test_ftp_source.py`
- **.test_parse_whitespace_only_returns_none()** (3 connections) — `tests/homesec/test_ftp_source.py`
- **Tests for _parse_passive_ports function.** (1 connections) — `tests/homesec/test_ftp_source.py`
- **Parses range format like '6000-6100'.** (1 connections) — `tests/homesec/test_ftp_source.py`
- **Parses comma-separated format like '6000,6001,6002'.** (1 connections) — `tests/homesec/test_ftp_source.py`
- **Handles whitespace in comma-separated format.** (1 connections) — `tests/homesec/test_ftp_source.py`
- **Raises ValueError when end < start.** (1 connections) — `tests/homesec/test_ftp_source.py`
- **Returns None for None input.** (1 connections) — `tests/homesec/test_ftp_source.py`
- **Returns None for empty string.** (1 connections) — `tests/homesec/test_ftp_source.py`
- **Returns None for whitespace-only string.** (1 connections) — `tests/homesec/test_ftp_source.py`
- **Parses single port number.** (1 connections) — `tests/homesec/test_ftp_source.py`
- **_normalize_extensions()** (1 connections) — `src/homesec/sources/ftp.py`
- **FTP clip source for camera uploads.** (1 connections) — `src/homesec/sources/ftp.py`

## Relationships

- No strong cross-community connections detected

## Source Files

- `src/homesec/sources/ftp.py`
- `tests/homesec/test_ftp_source.py`

## Audit Trail

- EXTRACTED: 44 (68%)
- INFERRED: 21 (32%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*