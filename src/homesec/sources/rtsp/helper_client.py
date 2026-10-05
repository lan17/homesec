"""Bounded typed control transport and process supervision for media helpers."""

from __future__ import annotations

import os
import selectors
import signal
import subprocess
import time
import uuid
from collections.abc import Callable
from queue import Empty, Queue
from threading import Lock, Thread
from typing import Generic, TypeVar

from pydantic import BaseModel

_MAX_MESSAGE_BYTES = 128_000


class HelperMessage(BaseModel):
    request_id: str | None = None
    event: str | None = None
    ok: bool = False
    error_code: str | None = None


class HelperError(RuntimeError):
    """Bounded transport failure; messages deliberately contain no input data."""


RequestT = TypeVar("RequestT", bound=BaseModel)
ReplyT = TypeVar("ReplyT", bound=HelperMessage)


class HelperClient(Generic[RequestT, ReplyT]):
    def __init__(
        self,
        args: list[str],
        *,
        request_model: type[RequestT],
        reply_model: type[ReplyT],
        on_message: Callable[[ReplyT], None] | None = None,
        context: str = "Media",
    ) -> None:
        self._request_model = request_model
        self._reply_model = reply_model
        self._on_message = on_message
        self._context = context
        self.process = subprocess.Popen(
            args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            bufsize=0,
        )
        if self.process.stdin is not None:
            os.set_blocking(self.process.stdin.fileno(), False)
        self._pending: dict[str, Queue[ReplyT | None]] = {}
        self._pending_lock = Lock()
        self._write_lock = Lock()
        self._ready: Queue[ReplyT | None] = Queue(maxsize=1)
        self._reader = Thread(target=self._read, name="webrtc-helper-replies", daemon=True)
        self._reader.start()

    def wait_ready(self, timeout_s: float) -> ReplyT:
        try:
            result = self._ready.get(timeout=timeout_s)
        except Empty as exc:
            raise HelperError(f"{self._context} helper startup timed out") from exc
        if result is None or result.event != "ready":
            raise HelperError(f"{self._context} helper did not become ready")
        return result

    def request(self, command: str, *, timeout_s: float = 2.0, **fields: object) -> ReplyT:
        deadline = time.monotonic() + timeout_s
        request_id = str(uuid.uuid4())
        waiter: Queue[ReplyT | None] = Queue(maxsize=1)
        with self._pending_lock:
            if self.process.poll() is not None:
                raise HelperError(f"{self._context} helper exited")
            if len(self._pending) >= 64:
                raise HelperError(f"{self._context} helper command limit reached")
            self._pending[request_id] = waiter
        try:
            try:
                request = self._request_model.model_validate(
                    {"command": command, "request_id": request_id, **fields}
                )
            except ValueError as exc:
                raise HelperError(f"Invalid {self._context.lower()} helper command") from exc
            payload = request.model_dump_json(exclude_none=True).encode() + b"\n"
            if len(payload) > _MAX_MESSAGE_BYTES:
                raise HelperError(f"{self._context} helper request exceeds limit")
            self._write(payload, deadline)
            try:
                result = waiter.get(timeout=max(0.0, deadline - time.monotonic()))
            except Empty as exc:
                raise HelperError(f"{self._context} helper command timed out") from exc
            if result is None:
                raise HelperError(f"{self._context} helper disconnected")
            return result
        finally:
            with self._pending_lock:
                self._pending.pop(request_id, None)

    def _write(self, payload: bytes, deadline: float) -> None:
        if not self._write_lock.acquire(timeout=max(0.0, deadline - time.monotonic())):
            raise HelperError(f"{self._context} helper command timed out")
        try:
            stdin = self.process.stdin
            if stdin is None:
                raise HelperError(f"{self._context} helper input unavailable")
            with selectors.DefaultSelector() as selector:
                descriptor = stdin.fileno()
                selector.register(descriptor, selectors.EVENT_WRITE)
                remaining = memoryview(payload)
                while remaining:
                    timeout = deadline - time.monotonic()
                    if timeout <= 0:
                        raise HelperError(f"{self._context} helper command timed out")
                    try:
                        written = os.write(descriptor, remaining)
                    except BlockingIOError:
                        if not selector.select(timeout):
                            raise HelperError(f"{self._context} helper command timed out") from None
                        continue
                    except InterruptedError:
                        continue
                    if written == 0:
                        raise HelperError(f"{self._context} helper disconnected")
                    remaining = remaining[written:]
        except (OSError, ValueError) as exc:
            raise HelperError(f"{self._context} helper disconnected") from exc
        finally:
            self._write_lock.release()

    def _read(self) -> None:
        stdout = self.process.stdout
        try:
            if stdout is None:
                return
            while True:
                raw = stdout.readline(_MAX_MESSAGE_BYTES + 1)
                if not raw:
                    return
                if len(raw) > _MAX_MESSAGE_BYTES or not raw.endswith(b"\n"):
                    return
                message = self._reply_model.model_validate_json(raw)
                if self._on_message is not None:
                    self._on_message(message)
                if message.event == "ready":
                    if self._ready.empty():
                        self._ready.put_nowait(message)
                elif message.request_id is not None:
                    with self._pending_lock:
                        waiter = self._pending.get(message.request_id)
                        if waiter is not None and waiter.empty():
                            waiter.put_nowait(message)
        except (OSError, ValueError):
            # Protocol failures can contain SDP or transport details: never log them.
            return
        finally:
            with self._pending_lock:
                for waiter in self._pending.values():
                    if waiter.empty():
                        waiter.put_nowait(None)
            if self._ready.empty():
                self._ready.put_nowait(None)

    def stop(self) -> None:
        """Allow camera teardown, then bound helper and child process cleanup."""
        if self.process.poll() is None:
            try:
                self.request("stop", timeout_s=0.5)
            except HelperError:
                pass
            # The reply precedes Rust destruction, including RTSP TEARDOWN.
            try:
                self.process.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                pass
        if self.process.poll() is None:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                self.process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(self.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self.process.wait(timeout=2.0)
        # A helper could have exited while leaving its FFmpeg child in the group.
        try:
            os.killpg(self.process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        for stream in (self.process.stdin, self.process.stdout):
            if stream is not None:
                stream.close()
        self._reader.join(timeout=1.0)
