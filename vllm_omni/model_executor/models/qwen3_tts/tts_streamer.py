from __future__ import annotations

import asyncio
from dataclasses import dataclass

import torch


@dataclass
class TTSCodecChunk:
    codec_ids: torch.Tensor
    is_final: bool = False
    chunk_index: int = 0


class TTSCodecCallback:
    """Buffers codec_ids and yields them in chunks via async queue."""

    def __init__(self, chunk_size: int = 24, min_first_chunk: int = 12):
        self.chunk_size = chunk_size
        self.min_first_chunk = min_first_chunk
        self._buffer: list[torch.Tensor] = []
        self._queue: asyncio.Queue[TTSCodecChunk | None] = asyncio.Queue()
        self._chunk_index = 0
        self._first_chunk_emitted = False
        self._finalized = False
        self._loop: asyncio.AbstractEventLoop | None = None

    def set_event_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def on_codec_generated(self, codec_ids: torch.Tensor) -> None:
        if self._finalized:
            return

        self._buffer.append(codec_ids.detach().clone().cpu())

        threshold = self.min_first_chunk if not self._first_chunk_emitted else self.chunk_size
        if len(self._buffer) >= threshold:
            self._emit_chunk(is_final=False)

    def _emit_chunk(self, is_final: bool = False) -> None:
        if not self._buffer and not is_final:
            return

        if self._buffer:
            codec_ids = torch.stack([c.squeeze(0) for c in self._buffer], dim=0)
            self._buffer.clear()
        else:
            codec_ids = torch.empty(0)

        chunk = TTSCodecChunk(
            codec_ids=codec_ids,
            is_final=is_final,
            chunk_index=self._chunk_index,
        )
        self._chunk_index += 1
        self._first_chunk_emitted = True

        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, chunk)
        else:
            self._queue.put_nowait(chunk)

    def finalize(self) -> None:
        if self._finalized:
            return
        self._finalized = True
        self._emit_chunk(is_final=True)

        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, None)
        else:
            self._queue.put_nowait(None)

    def signal_error(self, error: Exception) -> None:
        self._finalized = True
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, None)
        else:
            self._queue.put_nowait(None)

    async def __aiter__(self):
        while True:
            chunk = await self._queue.get()
            if chunk is None:
                break
            yield chunk

    def __iter__(self):
        while True:
            try:
                chunk = self._queue.get_nowait()
                if chunk is None:
                    break
                yield chunk
            except asyncio.QueueEmpty:
                break
