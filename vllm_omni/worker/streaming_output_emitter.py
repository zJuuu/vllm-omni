"""Streaming output emitter for TTS."""

from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Callable

from vllm.v1.engine import FinishReason

from vllm_omni.engine import OmniEngineCoreOutput, StreamingCodecChunk

logger = logging.getLogger(__name__)


class StreamingOutputEmitter:
    """Daemon thread that polls streaming queue and emits outputs."""

    def __init__(
        self,
        streaming_queue: queue.Queue,
        output_callback: Callable[[str, OmniEngineCoreOutput], None],
        poll_interval: float = 0.001,
    ):
        self._streaming_queue = streaming_queue
        self._output_callback = output_callback
        self._poll_interval = poll_interval
        self._running = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            self._running = True
            self._thread = threading.Thread(
                target=self._run,
                daemon=True,
                name="StreamingOutputEmitter",
            )
            self._thread.start()

    def stop(self) -> None:
        with self._lock:
            self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def is_running(self) -> bool:
        return self._running

    def _run(self) -> None:
        while self._running:
            try:
                request_id, chunk = self._streaming_queue.get(
                    timeout=self._poll_interval
                )
                self._emit_chunk(request_id, chunk)
            except queue.Empty:
                continue
            except Exception as e:
                logger.exception("Error in StreamingOutputEmitter: %s", e)
                continue

    def _emit_chunk(self, request_id: str, chunk: StreamingCodecChunk) -> None:
        try:
            finish_reason = FinishReason.STOP if chunk.is_final else None

            output = OmniEngineCoreOutput(
                request_id=request_id,
                new_token_ids=[],
                finish_reason=finish_reason,
                streaming_codec_chunk=chunk,
            )
            self._output_callback(request_id, output)
        except Exception as e:
            logger.exception("Failed to emit streaming chunk for %s: %s", request_id, e)
