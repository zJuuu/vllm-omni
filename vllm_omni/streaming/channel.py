"""ZMQ-based streaming channel for TTS."""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import threading
from typing import TYPE_CHECKING

import msgspec

if TYPE_CHECKING:
    from vllm_omni.engine import StreamingAudioChunk

logger = logging.getLogger(__name__)

_zmq = None
_zmq_asyncio = None


def _get_zmq():
    global _zmq
    if _zmq is None:
        import zmq
        _zmq = zmq
    return _zmq


def _get_zmq_asyncio():
    global _zmq_asyncio
    if _zmq_asyncio is None:
        import zmq.asyncio
        _zmq_asyncio = zmq.asyncio
    return _zmq_asyncio


def get_streaming_ipc_path() -> str:
    return os.path.join(tempfile.gettempdir(), "vllm_omni_streaming.ipc")


def get_cancel_ipc_path() -> str:
    return os.path.join(tempfile.gettempdir(), "vllm_omni_cancel.ipc")


class StreamingChunkEmitter:
    """PUSH socket emitter for the engine process."""

    def __init__(self):
        self._socket = None
        self._cancel_socket = None
        self._context = None
        self._encoder = msgspec.msgpack.Encoder()
        self._lock = threading.Lock()
        self._cancelled_requests: set[str] = set()

    def connect(self) -> bool:
        with self._lock:
            if self._socket is not None:
                return True

            try:
                zmq = _get_zmq()
                ipc_path = get_streaming_ipc_path()
                cancel_path = get_cancel_ipc_path()

                self._context = zmq.Context()

                self._socket = self._context.socket(zmq.PUSH)
                self._socket.setsockopt(zmq.SNDHWM, 100)
                self._socket.setsockopt(zmq.LINGER, 0)
                self._socket.connect(f"ipc://{ipc_path}")

                self._cancel_socket = self._context.socket(zmq.SUB)
                self._cancel_socket.setsockopt(zmq.RCVHWM, 100)
                self._cancel_socket.setsockopt(zmq.LINGER, 0)
                self._cancel_socket.setsockopt_string(zmq.SUBSCRIBE, "")
                self._cancel_socket.connect(f"ipc://{cancel_path}")

                logger.info("StreamingChunkEmitter connected to ipc://%s (cancel: %s)", ipc_path, cancel_path)
                return True
            except Exception as e:
                logger.warning("Failed to connect streaming emitter: %s", e)
                self._socket = None
                self._cancel_socket = None
                self._context = None
                return False

    def is_cancelled(self, request_id: str) -> bool:
        if self._cancel_socket is None:
            return request_id in self._cancelled_requests

        try:
            zmq = _get_zmq()
            while self._cancel_socket.poll(timeout=0):
                cancelled_id = self._cancel_socket.recv_string(flags=zmq.NOBLOCK)
                self._cancelled_requests.add(cancelled_id)
        except Exception:
            pass

        return request_id in self._cancelled_requests

    def clear_cancelled(self, request_id: str) -> None:
        self._cancelled_requests.discard(request_id)

    def emit(self, request_id: str, chunk: "StreamingAudioChunk") -> bool:
        if self._socket is None:
            if not self.connect():
                return False

        try:
            zmq = _get_zmq()
            data = self._encoder.encode(chunk)
            self._socket.send_multipart(
                [request_id.encode(), data],
                flags=zmq.NOBLOCK,
            )
            return True
        except Exception as e:
            logger.debug("Failed to emit chunk: %s", e)
            return False

    def close(self):
        with self._lock:
            if self._socket:
                self._socket.close()
                self._socket = None
            if self._cancel_socket:
                self._cancel_socket.close()
                self._cancel_socket = None
            if self._context:
                self._context.term()
                self._context = None
            self._cancelled_requests.clear()


class StreamingChunkListener:
    """PULL socket listener for the API server process."""

    def __init__(self):
        self._socket = None
        self._cancel_socket = None
        self._context = None
        self._running = False
        self._task = None
        self._queues: dict[str, asyncio.Queue] = {}
        self._lock = asyncio.Lock()

    async def start(self):
        if self._running:
            return

        try:
            from vllm_omni.engine import StreamingAudioChunk

            zmq_async = _get_zmq_asyncio()
            zmq = _get_zmq()
            ipc_path = get_streaming_ipc_path()
            cancel_path = get_cancel_ipc_path()

            self._context = zmq_async.Context()

            self._socket = self._context.socket(zmq.PULL)
            self._socket.setsockopt(zmq.RCVHWM, 100)
            self._socket.bind(f"ipc://{ipc_path}")

            self._cancel_socket = self._context.socket(zmq.PUB)
            self._cancel_socket.setsockopt(zmq.SNDHWM, 100)
            self._cancel_socket.setsockopt(zmq.LINGER, 0)
            self._cancel_socket.bind(f"ipc://{cancel_path}")

            self._running = True

            self._decoder = msgspec.msgpack.Decoder(type=StreamingAudioChunk)

            logger.info("StreamingChunkListener bound to ipc://%s (cancel: %s)", ipc_path, cancel_path)

            while self._running:
                try:
                    if await self._socket.poll(timeout=100):
                        frames = await self._socket.recv_multipart()
                        request_id = frames[0].decode()
                        chunk = self._decoder.decode(frames[1])

                        async with self._lock:
                            q = self._queues.get(request_id)
                            if q:
                                try:
                                    q.put_nowait(chunk)
                                except asyncio.QueueFull:
                                    pass
                except Exception as e:
                    if self._running:
                        logger.debug("Listener error: %s", e)
                    await asyncio.sleep(0.01)

        except Exception as e:
            logger.warning("StreamingChunkListener failed to start: %s", e)
        finally:
            self._running = False

    def stop(self):
        self._running = False
        if self._socket:
            self._socket.close()
            self._socket = None
        if self._cancel_socket:
            self._cancel_socket.close()
            self._cancel_socket = None
        if self._context:
            self._context.term()
            self._context = None

    async def subscribe(self, request_id: str) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=50)
        async with self._lock:
            self._queues[request_id] = q
        return q

    async def unsubscribe(self, request_id: str):
        async with self._lock:
            self._queues.pop(request_id, None)

    def cancel_request(self, request_id: str) -> bool:
        if self._cancel_socket is None:
            return False

        try:
            zmq = _get_zmq()
            self._cancel_socket.send_string(request_id, flags=zmq.NOBLOCK)
            return True
        except Exception:
            return False


_listener: StreamingChunkListener | None = None
_listener_lock = threading.Lock()


def get_listener() -> StreamingChunkListener:
    global _listener
    with _listener_lock:
        if _listener is None:
            _listener = StreamingChunkListener()
        return _listener


async def ensure_listener_started() -> StreamingChunkListener:
    listener = get_listener()
    if not listener._running and listener._task is None:
        listener._task = asyncio.create_task(listener.start())
        await asyncio.sleep(0.1)
    return listener
