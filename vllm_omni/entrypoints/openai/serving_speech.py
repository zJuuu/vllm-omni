import asyncio
from collections.abc import AsyncGenerator

from fastapi import Request
from fastapi.responses import Response, StreamingResponse
from vllm.entrypoints.openai.serving_engine import OpenAIServing
from vllm.logger import init_logger
from vllm.utils import random_uuid

from vllm_omni.entrypoints.openai.audio_utils_mixin import AudioMixin
from vllm_omni.entrypoints.openai.protocol.audio import AudioResponse, CreateAudio, OpenAICreateSpeechRequest
from vllm_omni.outputs import OmniRequestOutput

logger = init_logger(__name__)

DEFAULT_SAMPLE_RATE = 24000
OPENAI_VOICES = {"alloy", "ash", "ballad", "coral", "echo", "fable", "onyx", "nova", "sage", "shimmer", "verse"}
DEFAULT_SPEAKER = "Chelsie"


def _get_speakers_from_config(hf_config) -> set[str]:
    talker_config = getattr(hf_config, "talker_config", None) if hf_config else None
    spk_id = getattr(talker_config, "spk_id", None) if talker_config else None
    if isinstance(spk_id, dict):
        return {name.lower() for name in spk_id.keys()}
    return set()


def _get_languages_from_config(hf_config) -> set[str]:
    talker_config = getattr(hf_config, "talker_config", None) if hf_config else None
    codec_language_id = getattr(talker_config, "codec_language_id", None) if talker_config else None
    if isinstance(codec_language_id, dict):
        return {name.lower() for name in codec_language_id.keys()}
    return set()


def _get_tts_model_type(hf_config) -> str | None:
    return getattr(hf_config, "tts_model_type", None) if hf_config else None


class OmniOpenAIServingSpeech(OpenAIServing, AudioMixin):

    async def create_speech(
        self,
        request: OpenAICreateSpeechRequest,
        raw_request: Request | None = None,
    ):
        """
        Create Speech API similar to OpenAI's API.

        See https://platform.openai.com/docs/api-reference/audio/createSpeech
        for the API specification. This API mimics the OpenAI Create Speech API.
        """
        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            logger.error("Error with model %s", error_check_ret)
            return error_check_ret

        if self.engine_client.errored:
            raise self.engine_client.dead_error

        request_id = f"speech-{random_uuid()}"

        try:
            hf_config = getattr(self.model_config, "hf_config", None) if self.model_config else None
            valid_speakers = _get_speakers_from_config(hf_config)
            valid_languages = _get_languages_from_config(hf_config)
            tts_model_type = _get_tts_model_type(hf_config)

            voice_lower = request.voice.lower()
            if voice_lower in valid_speakers:
                speaker = request.voice
            elif voice_lower in OPENAI_VOICES:
                speaker = DEFAULT_SPEAKER
            elif not valid_speakers:
                speaker = request.voice
            else:
                valid_list = ", ".join(sorted(valid_speakers))
                return self.create_error_response(
                    f"Invalid voice '{request.voice}'. "
                    f"Valid voices: {valid_list}. OpenAI-compatible voices (mapped to default): "
                    f"{', '.join(sorted(OPENAI_VOICES))}"
                )

            language = (request.language or "auto").lower()
            if language != "auto" and valid_languages and language not in valid_languages:
                return self.create_error_response(
                    f"Invalid language '{request.language}'. Valid languages: auto, {', '.join(sorted(valid_languages))}"
                )

            if tts_model_type == "voice_design":
                task_type = "VoiceDesign"
            elif request.instructions:
                return self.create_error_response(
                    f"Instructions are not supported by this model (tts_model_type={tts_model_type}). "
                    "Use a VoiceDesign model to use instructions for voice style control."
                )
            else:
                task_type = "CustomVoice"

            if request.stream:
                return await self._create_speech_streaming(
                    request=request,
                    raw_request=raw_request,
                    speaker=speaker,
                    language=language,
                    task_type=task_type,
                    request_id=request_id,
                )

            additional_information = {
                "text": [request.input],
                "task_type": [task_type],
                "speaker": [speaker],
                "language": [language.capitalize()],
                "instruct": [request.instructions or ""],
            }

            prompt = {
                "prompt": request.input,
                "additional_information": additional_information,
            }

            sampling_params_list = self.engine_client.default_sampling_params_list

            generator = self.engine_client.generate(
                prompt=prompt, request_id=request_id, sampling_params_list=sampling_params_list
            )

            final_output: OmniRequestOutput | None = None
            try:
                async for res in generator:
                    if raw_request is not None and await raw_request.is_disconnected():
                        await self.engine_client.abort(request_id)
                        return self.create_error_response("Client disconnected")
                    final_output = res
            except asyncio.CancelledError:
                await self.engine_client.abort(request_id)
                raise

            if final_output is None:
                return self.create_error_response("No output generated from the model.")

            if final_output.final_output_type != "audio":
                return self.create_error_response(f"Unexpected final output type: {final_output.final_output_type}")

            audio_tensor = final_output.request_output.multimodal_output["audio"].float().detach().cpu().numpy()
            sr_tensor = final_output.request_output.multimodal_output.get("sr")
            sample_rate = int(sr_tensor.item()) if sr_tensor is not None else DEFAULT_SAMPLE_RATE

            audio_obj = CreateAudio(
                audio_tensor=audio_tensor,
                sample_rate=sample_rate,
                response_format=request.response_format,
                speed=request.speed,
                stream_format=request.stream_format,
                base64_encode=False,
            )

            audio_response: AudioResponse = self.create_audio(audio_obj)
            return Response(content=audio_response.audio_data, media_type=audio_response.media_type)

        except asyncio.CancelledError:
            await self.engine_client.abort(request_id)
            return self.create_error_response("Client disconnected")
        except ValueError as e:
            # TODO: Use a vllm-specific Validation Error
            return self.create_error_response(str(e))
        except Exception as e:
            return self.create_error_response(f"{e} {e.__cause__}")

    async def _create_speech_streaming(
        self,
        request: OpenAICreateSpeechRequest,
        raw_request: Request | None,
        speaker: str,
        language: str,
        task_type: str,
        request_id: str,
    ) -> StreamingResponse:
        """Create streaming speech response via ZMQ."""

        async def audio_generator() -> AsyncGenerator[bytes, None]:
            zmq_queue = None
            generation_task = None
            client_disconnected = False

            additional_information = {
                "text": [request.input],
                "task_type": [task_type],
                "speaker": [speaker],
                "language": [language.capitalize()],
                "instruct": [request.instructions or ""],
                "streaming_tts": [True],
                "global_request_id": [request_id],
            }

            prompt = {
                "prompt": request.input,
                "additional_information": additional_information,
            }

            sampling_params_list = self.engine_client.default_sampling_params_list

            try:
                logger.info("Starting streaming TTS for request %s", request_id)

                try:
                    from vllm_omni.streaming.channel import ensure_listener_started
                    listener = await ensure_listener_started()
                    zmq_queue = await listener.subscribe(request_id)
                except Exception as e:
                    logger.warning("Failed to start ZMQ listener: %s", e)
                    zmq_queue = None

                async def run_generation():
                    async for _ in self.engine_client.generate(
                        prompt=prompt,
                        request_id=request_id,
                        sampling_params_list=sampling_params_list,
                    ):
                        pass

                generation_task = asyncio.create_task(run_generation())

                if zmq_queue is not None:
                    while True:
                        if raw_request is not None and await raw_request.is_disconnected():
                            logger.info("Client disconnected for request %s", request_id)
                            client_disconnected = True
                            break

                        try:
                            chunk = await asyncio.wait_for(zmq_queue.get(), timeout=0.5)

                            if chunk.audio_bytes:
                                yield chunk.audio_bytes

                            if chunk.is_final:
                                break

                        except asyncio.TimeoutError:
                            if generation_task.done():
                                while not zmq_queue.empty():
                                    chunk = zmq_queue.get_nowait()
                                    if chunk.audio_bytes:
                                        yield chunk.audio_bytes
                                break
                            continue
                else:
                    logger.info("Using fallback mode for %s", request_id)
                    await generation_task
                    generation_task = None

                logger.info("Streaming TTS completed for %s", request_id)

            except asyncio.CancelledError:
                logger.info("Streaming cancelled for request %s", request_id)
                client_disconnected = True
            except Exception as e:
                logger.error("Error in streaming TTS for %s: %s", request_id, e, exc_info=True)
            finally:
                if generation_task and not generation_task.done():
                    generation_task.cancel()

                if client_disconnected:
                    try:
                        from vllm_omni.streaming.channel import get_listener
                        get_listener().cancel_request(request_id)
                    except Exception:
                        pass

                    try:
                        await self.engine_client.abort(request_id)
                    except Exception:
                        pass

                if zmq_queue is not None:
                    try:
                        from vllm_omni.streaming.channel import get_listener
                        await get_listener().unsubscribe(request_id)
                    except Exception:
                        pass

        return StreamingResponse(
            audio_generator(),
            media_type="audio/pcm",
            headers={
                "Transfer-Encoding": "chunked",
                "X-Audio-Sample-Rate": str(DEFAULT_SAMPLE_RATE),
                "X-Audio-Channels": "1",
                "X-Audio-Bit-Depth": "16",
                "X-Audio-Format": "pcm_s16le",
            },
        )


