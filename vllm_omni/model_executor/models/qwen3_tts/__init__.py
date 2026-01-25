__all__ = [
    "Qwen3TTSForConditionalGeneration",
    "Qwen3TTSPreTrainedModel",
    "Qwen3TTSTalkerForConditionalGeneration",
    "Qwen3TTSTalkerModel",
    "TTSCodecCallback",
    "TTSCodecChunk",
    "StreamingAudioDecoder",
    "StreamingAudioDecoderWrapper",
    "AudioChunk",
]


def __getattr__(name):
    if name in (
        "Qwen3TTSForConditionalGeneration",
        "Qwen3TTSPreTrainedModel",
        "Qwen3TTSTalkerForConditionalGeneration",
        "Qwen3TTSTalkerModel",
    ):
        from .modeling_qwen3_tts import (
            Qwen3TTSForConditionalGeneration,
            Qwen3TTSPreTrainedModel,
            Qwen3TTSTalkerForConditionalGeneration,
            Qwen3TTSTalkerModel,
        )

        return {
            "Qwen3TTSForConditionalGeneration": Qwen3TTSForConditionalGeneration,
            "Qwen3TTSPreTrainedModel": Qwen3TTSPreTrainedModel,
            "Qwen3TTSTalkerForConditionalGeneration": Qwen3TTSTalkerForConditionalGeneration,
            "Qwen3TTSTalkerModel": Qwen3TTSTalkerModel,
        }[name]

    if name in ("TTSCodecCallback", "TTSCodecChunk"):
        from .tts_streamer import TTSCodecCallback, TTSCodecChunk

        return {"TTSCodecCallback": TTSCodecCallback, "TTSCodecChunk": TTSCodecChunk}[name]

    if name in ("StreamingAudioDecoder", "StreamingAudioDecoderWrapper", "AudioChunk"):
        from .streaming_decoder import AudioChunk, StreamingAudioDecoder, StreamingAudioDecoderWrapper

        return {
            "StreamingAudioDecoder": StreamingAudioDecoder,
            "StreamingAudioDecoderWrapper": StreamingAudioDecoderWrapper,
            "AudioChunk": AudioChunk,
        }[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
