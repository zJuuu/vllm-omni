from typing import Literal

import numpy as np
from pydantic import BaseModel, Field, field_validator


class OpenAICreateSpeechRequest(BaseModel):
    input: str
    model: str | None = None
    voice: str
    instructions: str | None = None
    response_format: Literal["wav", "pcm", "flac", "mp3", "aac", "opus"]
    speed: float | None = Field(
        default=1.0,
        ge=0.25,
        le=4.0,
    )
    stream_format: Literal["sse", "audio"] | None = "audio"
    language: str | None = "auto"
    stream: bool = Field(
        default=False,
        description="If true, audio will be streamed as chunks using Transfer-Encoding: chunked. "
        "Only PCM format is supported for streaming. The audio is 16-bit signed little-endian PCM "
        "at 24kHz sample rate.",
    )

    @field_validator("stream_format")
    @classmethod
    def validate_stream_format(cls, v: str) -> str:
        if v == "sse":
            raise ValueError("'sse' is not a supported stream_format yet. Please use 'audio'.")
        return v

    @field_validator("stream")
    @classmethod
    def validate_stream(cls, v: bool, info) -> bool:
        if v:
            response_format = info.data.get("response_format", "wav")
            if response_format != "pcm":
                raise ValueError(f"Streaming requires response_format='pcm', got '{response_format}'")
        return v


class CreateAudio(BaseModel):
    audio_tensor: np.ndarray
    sample_rate: int = 24000
    response_format: str = "wav"
    speed: float = 1.0
    stream_format: Literal["sse", "audio"] | None = "audio"
    base64_encode: bool = True

    class Config:
        arbitrary_types_allowed = True


class AudioResponse(BaseModel):
    audio_data: bytes | str
    media_type: str
