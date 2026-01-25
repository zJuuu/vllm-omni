from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .tts_streamer import TTSCodecChunk


@dataclass
class AudioChunk:
    audio_data: np.ndarray
    sample_rate: int
    chunk_index: int
    is_final: bool = False


class StreamingAudioDecoder:
    """Decodes codec chunks using overlap-discard for smooth transitions."""

    def __init__(
        self,
        decoder,
        left_context_frames: int = 25,
        sample_rate: int = 24000,
    ):
        self.decoder = decoder
        self.left_context_frames = left_context_frames
        self.sample_rate = sample_rate
        self._prev_codes: torch.Tensor | None = None
        self._total_upsample: int | None = None

    def _get_total_upsample(self) -> int:
        if self._total_upsample is None:
            if hasattr(self.decoder, "total_upsample"):
                self._total_upsample = int(self.decoder.total_upsample)
            elif hasattr(self.decoder, "model") and hasattr(self.decoder.model, "decoder"):
                inner_decoder = self.decoder.model.decoder
                if hasattr(inner_decoder, "total_upsample"):
                    self._total_upsample = int(inner_decoder.total_upsample)
                else:
                    self._total_upsample = 2000
            else:
                self._total_upsample = 2000
        return self._total_upsample

    @torch.inference_mode()
    def decode_chunk(self, codec_chunk: TTSCodecChunk) -> AudioChunk:
        if codec_chunk.codec_ids.numel() == 0:
            return AudioChunk(
                audio_data=np.array([], dtype=np.float32),
                sample_rate=self.sample_rate,
                chunk_index=codec_chunk.chunk_index,
                is_final=codec_chunk.is_final,
            )

        codec_ids = codec_chunk.codec_ids
        device = next(self.decoder.parameters()).device

        if codec_ids.device != device:
            codec_ids = codec_ids.to(device)

        context_size = 0
        if self._prev_codes is not None:
            context_size = min(self.left_context_frames, self._prev_codes.shape[0])

        if context_size > 0 and self._prev_codes is not None:
            context = self._prev_codes[-context_size:].to(device)
            codes_with_context = torch.cat([context, codec_ids], dim=0)
        else:
            codes_with_context = codec_ids

        codes_batch = codes_with_context.unsqueeze(0).transpose(1, 2)
        wav = self.decoder(codes_batch)

        if wav.dim() == 3:
            wav = wav.squeeze(0).squeeze(0)
        elif wav.dim() == 2:
            wav = wav.squeeze(0)

        total_upsample = self._get_total_upsample()
        samples_to_trim = context_size * total_upsample

        if samples_to_trim > 0 and samples_to_trim < wav.shape[0]:
            wav = wav[samples_to_trim:]

        self._prev_codes = codec_ids.cpu()
        audio = wav.float().cpu().numpy()

        return AudioChunk(
            audio_data=audio,
            sample_rate=self.sample_rate,
            chunk_index=codec_chunk.chunk_index,
            is_final=codec_chunk.is_final,
        )

    def reset(self) -> None:
        self._prev_codes = None


class StreamingAudioDecoderWrapper:
    """Wrapper that works with Qwen3TTSTokenizer."""

    def __init__(
        self,
        speech_tokenizer,
        left_context_frames: int = 25,
    ):
        self.speech_tokenizer = speech_tokenizer
        self.left_context_frames = left_context_frames
        self.sample_rate = speech_tokenizer.get_output_sample_rate()

        model_type = speech_tokenizer.get_model_type()
        if model_type != "qwen3_tts_tokenizer_12hz":
            raise ValueError(f"Streaming decode only supports 12Hz tokenizer, got {model_type}")

        self._streaming_decoder = StreamingAudioDecoder(
            decoder=speech_tokenizer.model.decoder,
            left_context_frames=left_context_frames,
            sample_rate=self.sample_rate,
        )

    def decode_chunk(self, codec_chunk: TTSCodecChunk) -> AudioChunk:
        return self._streaming_decoder.decode_chunk(codec_chunk)

    def reset(self) -> None:
        self._streaming_decoder.reset()

    def get_sample_rate(self) -> int:
        return self.sample_rate
