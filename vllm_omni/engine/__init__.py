"""
Engine components for vLLM-Omni.
"""

from typing import Any

import msgspec
import torch
from vllm.v1.engine import (
    EngineCoreOutput,
    EngineCoreOutputs,
    EngineCoreRequest,
)


class PromptEmbedsPayload(msgspec.Struct):
    """Serialized prompt embeddings payload for direct transfer.

    data: raw bytes of the tensor in row-major order
    shape: [seq_len, hidden_size]
    dtype: torch dtype name (e.g., "float16", "float32")
    """

    data: bytes
    shape: list[int]
    dtype: str


class AdditionalInformationEntry(msgspec.Struct):
    """One entry of additional_information.

    Two supported forms are encoded:
      - tensor: data/shape/dtype
      - list: a Python list (msgspec-serializable)
    Exactly one of (tensor_data, list_data) should be non-None.
    """

    # Tensor form
    tensor_data: bytes | None = None
    tensor_shape: list[int] | None = None
    tensor_dtype: str | None = None

    # List form
    list_data: list[Any] | None = None


class AdditionalInformationPayload(msgspec.Struct):
    """Serialized dictionary payload for additional_information.

    Keys are strings; values are encoded as AdditionalInformationEntry.
    """

    entries: dict[str, AdditionalInformationEntry]


class OmniEngineCoreRequest(EngineCoreRequest):
    """Engine core request for omni models with embeddings support.

    Extends the base EngineCoreRequest with support for prompt embeddings
    and additional information payloads, enabling direct transfer of
    pre-computed embeddings between pipeline stages.

    Attributes:
        prompt_embeds: Optional serialized prompt embeddings payload for
            direct transfer between stages
        additional_information: Optional serialized additional information
            dictionary containing tensors or lists to pass along with the request
    """

    # Optional prompt embeddings (direct-transfer version)
    prompt_embeds: PromptEmbedsPayload | None = None
    # Optional additional information dictionary (serialized)
    additional_information: AdditionalInformationPayload | None = None


class StreamingCodecChunk(msgspec.Struct):
    """Streaming codec chunk for TTS streaming."""

    codec_ids: list[int]  # Stored as list for msgspec serialization
    codec_shape: list[int]  # Original tensor shape
    chunk_index: int
    is_final: bool = False

    @classmethod
    def from_tensor(
        cls,
        codec_tensor: torch.Tensor,
        chunk_index: int,
        is_final: bool = False,
    ) -> "StreamingCodecChunk":
        """Create a StreamingCodecChunk from a tensor."""
        return cls(
            codec_ids=codec_tensor.flatten().tolist(),
            codec_shape=list(codec_tensor.shape),
            chunk_index=chunk_index,
            is_final=is_final,
        )

    def to_tensor(self, device: str = "cpu") -> torch.Tensor:
        """Convert the codec_ids back to a tensor."""
        tensor = torch.tensor(self.codec_ids, dtype=torch.long, device=device)
        return tensor.reshape(self.codec_shape) if self.codec_shape else tensor


class StreamingAudioChunk(msgspec.Struct):
    """Streaming audio chunk with pre-decoded PCM bytes for TTS streaming."""

    audio_bytes: bytes  # Raw PCM audio bytes
    sample_rate: int
    chunk_index: int
    is_final: bool = False


class OmniEngineCoreOutput(EngineCoreOutput):
    pooling_output: dict[str, torch.Tensor] | None = None
    # Streaming TTS support: intermediate codec chunks
    streaming_codec_chunk: StreamingCodecChunk | None = None


class OmniEngineCoreOutputs(EngineCoreOutputs):
    outputs: list[OmniEngineCoreOutput] = []
