"""Code2Wav GPU Model Runner for vLLM-Omni.

Handles direct conversion from codec codes to audio waveforms for Qwen3 Omni MoE Code2Wav.
This is a non-autoregressive model that doesn't require sampling or logits computation.
"""

from __future__ import annotations

import gc
import logging
import queue
import threading
from collections.abc import Callable
from copy import copy

import numpy as np
import torch
from vllm.config import CUDAGraphMode
from vllm.distributed.ec_transfer import get_ec_transfer, has_ec_transfer
from vllm.distributed.kv_transfer import get_kv_transfer_group, has_kv_transfer_group
from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
    RoutedExpertsCapturer,
)
from vllm.model_executor.models.interfaces import supports_mm_encoder_only
from vllm.utils.math_utils import cdiv
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.outputs import AsyncModelRunnerOutput, make_empty_encoder_model_runner_output
from vllm.v1.spec_decode.eagle import EagleProposer
from vllm.v1.utils import record_function_or_nullcontext
from vllm.v1.worker.gpu_model_runner import (
    EMPTY_MODEL_RUNNER_OUTPUT,
    AsyncGPUModelRunnerOutput,
    IntermediateTensors,
    PerLayerAttnMetadata,
    get_pp_group,
    set_forward_context,
)
from vllm.v1.worker.ubatch_utils import maybe_create_ubatch_slices
from vllm.v1.worker.utils import sanity_check_mm_encoder_outputs

from vllm_omni.engine import OmniEngineCoreOutput, StreamingAudioChunk, StreamingCodecChunk
from vllm_omni.outputs import OmniModelRunnerOutput
from vllm_omni.worker.gpu_ar_model_runner import ExecuteModelState
from vllm_omni.worker.gpu_model_runner import OmniGPUModelRunner
from vllm_omni.worker.streaming_output_emitter import StreamingOutputEmitter

logger = logging.getLogger(__name__)


class GenerationCancelled(Exception):
    pass


class StreamingTTSCallback:
    """Callback for streaming TTS codec generation."""

    _shared_emitter = None
    _emitter_lock = threading.Lock()

    @classmethod
    def _get_emitter(cls):
        if cls._shared_emitter is None:
            with cls._emitter_lock:
                if cls._shared_emitter is None:
                    try:
                        from vllm_omni.streaming import StreamingChunkEmitter
                        cls._shared_emitter = StreamingChunkEmitter()
                        cls._shared_emitter.connect()
                        logger.info("StreamingTTSCallback: ZMQ emitter initialized")
                    except Exception as e:
                        logger.warning("Failed to init ZMQ emitter: %s", e)
        return cls._shared_emitter

    def __init__(
        self,
        request_id: str,
        output_queue: queue.Queue,
        chunk_threshold: int = 24,
        streaming_decoder=None,
    ):
        self.request_id = request_id
        self.output_queue = output_queue
        self.chunk_threshold = chunk_threshold
        self.streaming_decoder = streaming_decoder
        self._accumulated_codec_ids: list[torch.Tensor] = []
        self._chunk_index: int = 0
        self._accumulated_frames: int = 0
        self._lock = threading.Lock()
        self._sample_rate = 24000
        self._cancelled = False

        if streaming_decoder is not None:
            try:
                self._sample_rate = streaming_decoder.get_sample_rate()
            except Exception:
                pass

    def cancel(self) -> None:
        self._cancelled = True

    def on_codec_generated(self, codec_ids: torch.Tensor) -> None:
        if self._cancelled:
            raise GenerationCancelled(f"Request {self.request_id} was cancelled")

        emitter = self._get_emitter()
        if emitter is not None and emitter.is_cancelled(self.request_id):
            self._cancelled = True
            raise GenerationCancelled(f"Request {self.request_id} was cancelled")

        with self._lock:
            if codec_ids is None:
                return

            codec_chunk = codec_ids.detach().cpu()
            self._accumulated_codec_ids.append(codec_chunk)
            self._accumulated_frames += 1

            if self._accumulated_frames >= self.chunk_threshold:
                self._emit_chunk(is_final=False)

    def _emit_chunk(self, is_final: bool = False) -> None:
        if not self._accumulated_codec_ids:
            if is_final:
                audio_chunk = StreamingAudioChunk(
                    audio_bytes=b"",
                    sample_rate=self._sample_rate,
                    chunk_index=self._chunk_index,
                    is_final=True,
                )
                self._send_audio_chunk(audio_chunk)
            return

        combined = torch.stack([c.squeeze(0) for c in self._accumulated_codec_ids], dim=0)
        audio_chunk = self._decode_to_audio(combined, is_final)
        self._send_audio_chunk(audio_chunk)

        self._accumulated_codec_ids = []
        self._accumulated_frames = 0
        self._chunk_index += 1

    def _decode_to_audio(self, codec_ids: torch.Tensor, is_final: bool) -> StreamingAudioChunk:
        if self.streaming_decoder is None:
            return StreamingAudioChunk(
                audio_bytes=b"",
                sample_rate=self._sample_rate,
                chunk_index=self._chunk_index,
                is_final=is_final,
            )

        try:
            from vllm_omni.model_executor.models.qwen3_tts.tts_streamer import TTSCodecChunk

            tts_chunk = TTSCodecChunk(
                codec_ids=codec_ids,
                chunk_index=self._chunk_index,
                is_final=is_final,
            )

            audio_result = self.streaming_decoder.decode_chunk(tts_chunk)

            audio_data = audio_result.audio_data
            if audio_data.size > 0:
                audio_data = np.clip(audio_data, -1.0, 1.0)
                audio_int16 = (audio_data * 32767).astype(np.int16)
                audio_bytes = audio_int16.tobytes()
            else:
                audio_bytes = b""

            return StreamingAudioChunk(
                audio_bytes=audio_bytes,
                sample_rate=audio_result.sample_rate,
                chunk_index=self._chunk_index,
                is_final=is_final,
            )

        except Exception as e:
            logger.error("Failed to decode codec chunk: %s", e, exc_info=True)
            return StreamingAudioChunk(
                audio_bytes=b"",
                sample_rate=self._sample_rate,
                chunk_index=self._chunk_index,
                is_final=is_final,
            )

    def _send_audio_chunk(self, audio_chunk: StreamingAudioChunk) -> None:
        emitter = self._get_emitter()
        if emitter is not None:
            if emitter.emit(self.request_id, audio_chunk):
                return

        self.output_queue.put((self.request_id, audio_chunk))

    def finalize(self) -> None:
        with self._lock:
            self._emit_chunk(is_final=True)

            emitter = self._get_emitter()
            if emitter is not None:
                emitter.clear_cancelled(self.request_id)


class GPUGenerationModelRunner(OmniGPUModelRunner):
    """Generation model runner for vLLM-Omni (non-autoregressive).

    - Reuses GPUModelRunner preparation, multimodal handling, and TP/PP/DP glue.
    - Does not compute logits or perform token sampling.
    - Executes generation process and returns tensors via `pooler_output`.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._streaming_callbacks: dict[str, StreamingTTSCallback] = {}
        self._streaming_output_queue: queue.Queue = queue.Queue()
        self._streaming_emitter: StreamingOutputEmitter | None = None
        self._streaming_output_callback: Callable[[str, OmniEngineCoreOutput], None] | None = None
        self._streaming_decoder = None
        self._streaming_decoder_initialized = False

    def _get_or_create_streaming_decoder(self):
        if self._streaming_decoder_initialized:
            return self._streaming_decoder

        self._streaming_decoder_initialized = True

        try:
            from vllm_omni.model_executor.models.qwen3_tts.streaming_decoder import (
                StreamingAudioDecoderWrapper,
            )

            speech_tokenizer = self._get_speech_tokenizer()
            if speech_tokenizer is not None:
                self._streaming_decoder = StreamingAudioDecoderWrapper(
                    speech_tokenizer=speech_tokenizer,
                    left_context_frames=25,
                )
                logger.info("Streaming decoder initialized")
        except Exception as e:
            logger.error("Failed to initialize streaming decoder: %s", e)

        return self._streaming_decoder

    def _get_speech_tokenizer(self):
        try:
            if hasattr(self.model, "model"):
                qwen3_tts_model = self.model.model
                if hasattr(qwen3_tts_model, "model"):
                    qwen3_tts_for_gen = qwen3_tts_model.model
                    if hasattr(qwen3_tts_for_gen, "speech_tokenizer"):
                        return qwen3_tts_for_gen.speech_tokenizer
        except Exception:
            pass
        return None

    def setup_streaming_callback(
        self,
        request_id: str,
        chunk_threshold: int = 24,
    ) -> StreamingTTSCallback:
        streaming_decoder = self._get_or_create_streaming_decoder()

        if streaming_decoder is not None:
            streaming_decoder.reset()

        callback = StreamingTTSCallback(
            request_id=request_id,
            output_queue=self._streaming_output_queue,
            chunk_threshold=chunk_threshold,
            streaming_decoder=streaming_decoder,
        )
        self._streaming_callbacks[request_id] = callback
        return callback

    def get_streaming_callback(self, request_id: str) -> StreamingTTSCallback | None:
        return self._streaming_callbacks.get(request_id)

    def remove_streaming_callback(self, request_id: str) -> None:
        self._streaming_callbacks.pop(request_id, None)

    def cancel_streaming_request(self, request_id: str) -> bool:
        callback = self._streaming_callbacks.get(request_id)
        if callback is not None:
            callback.cancel()
            return True
        return False

    def get_streaming_chunks(self) -> list[tuple[str, StreamingCodecChunk]]:
        chunks = []
        while True:
            try:
                chunk = self._streaming_output_queue.get_nowait()
                chunks.append(chunk)
            except queue.Empty:
                break
        return chunks

    def set_streaming_output_callback(
        self,
        callback: Callable[[str, OmniEngineCoreOutput], None],
    ) -> None:
        self._streaming_output_callback = callback
        if self._streaming_emitter is None:
            self._streaming_emitter = StreamingOutputEmitter(
                streaming_queue=self._streaming_output_queue,
                output_callback=callback,
            )
            self._streaming_emitter.start()

    def stop_streaming_emitter(self) -> None:
        if self._streaming_emitter is not None:
            self._streaming_emitter.stop()
            self._streaming_emitter = None

    def _update_states(self, scheduler_output: SchedulerOutput) -> None:
        for req_id in scheduler_output.finished_req_ids:
            if req_id in self._streaming_callbacks:
                callback = self._streaming_callbacks[req_id]
                try:
                    callback.finalize()
                except Exception:
                    pass
                self.remove_streaming_callback(req_id)

        super()._update_states(scheduler_output)

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
        intermediate_tensors: IntermediateTensors | None = None,
    ) -> OmniModelRunnerOutput | IntermediateTensors:
        if self.execute_model_state is not None:
            raise RuntimeError("State error: sample_tokens() must be called after execute_model() returns None.")

        if self.vllm_config.model_config.enable_return_routed_experts:
            capturer = RoutedExpertsCapturer.get_instance()
            if capturer is not None:
                capturer.clear_buffer()  # noqa
            else:
                logger.error("RoutedExpertsCapturer not initialized.")

        if scheduler_output.preempted_req_ids and has_kv_transfer_group():
            get_kv_transfer_group().handle_preemptions(scheduler_output.preempted_req_ids)

        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        with (
            record_function_or_nullcontext("gpu_model_runner: preprocess"),
            self.synchronize_input_prep(),
        ):
            self._update_states(scheduler_output)
            if not scheduler_output.total_num_scheduled_tokens:
                return EMPTY_MODEL_RUNNER_OUTPUT

            if has_ec_transfer() and get_ec_transfer().is_producer:
                with self.maybe_get_ec_connector_output(
                    scheduler_output,
                    encoder_cache=self.encoder_cache,
                ) as ec_connector_output:
                    self._execute_mm_encoder(scheduler_output)
                    return make_empty_encoder_model_runner_output(scheduler_output)

            if not num_scheduled_tokens:
                if (
                    self.parallel_config.distributed_executor_backend == "external_launcher"
                    and self.parallel_config.data_parallel_size > 1
                ):
                    # this is a corner case when both external launcher
                    # and DP are enabled, num_scheduled_tokens could be
                    # 0, and has_unfinished_requests in the outer loop
                    # returns True. before returning early here we call
                    # dummy run to ensure coordinate_batch_across_dp
                    # is called into to avoid out of sync issues.
                    self._dummy_run(1)
                if not has_kv_transfer_group():
                    # Return empty ModelRunnerOutput if no work to do.
                    return EMPTY_MODEL_RUNNER_OUTPUT

                return self.kv_connector_no_forward(scheduler_output, self.vllm_config)

            if self.cache_config.kv_sharing_fast_prefill:
                assert not self.num_prompt_logprobs, (
                    "--kv-sharing-fast-prefill produces incorrect "
                    "logprobs for prompt tokens, tokens, please disable "
                    "it when the requests need prompt logprobs"
                )
            num_reqs = self.input_batch.num_reqs
            req_ids = self.input_batch.req_ids
            tokens = [scheduler_output.num_scheduled_tokens[i] for i in req_ids]
            num_scheduled_tokens_np = np.array(tokens, dtype=np.int32)
            max_num_scheduled_tokens = int(num_scheduled_tokens_np.max())
            num_tokens_unpadded = scheduler_output.total_num_scheduled_tokens

            logits_indices, spec_decode_metadata = self._prepare_inputs(
                scheduler_output,
                num_scheduled_tokens_np,
            )

            cascade_attn_prefix_lens = None
            # Disable cascade attention when using microbatching (DBO)
            if self.cascade_attn_enabled and not self.parallel_config.use_ubatching:
                # Pre-compute cascade attention prefix lengths
                cascade_attn_prefix_lens = self._compute_cascade_attn_prefix_lens(
                    num_scheduled_tokens_np,
                    self.input_batch.num_computed_tokens_cpu[:num_reqs],
                    scheduler_output.num_common_prefix_blocks,
                )

            (
                cudagraph_mode,
                batch_desc,
                should_ubatch,
                num_tokens_across_dp,
                cudagraph_stats,
            ) = self._determine_batch_execution_and_padding(
                num_tokens=num_tokens_unpadded,
                num_reqs=num_reqs,
                num_scheduled_tokens_np=num_scheduled_tokens_np,
                max_num_scheduled_tokens=max_num_scheduled_tokens,
                use_cascade_attn=cascade_attn_prefix_lens is not None,
                num_encoder_reqs=len(scheduler_output.scheduled_encoder_inputs),
            )

            logger.debug(
                "Running batch with cudagraph_mode: %s, batch_descriptor: %s, "
                "should_ubatch: %s, num_tokens_across_dp: %s",
                cudagraph_mode,
                batch_desc,
                should_ubatch,
                num_tokens_across_dp,
            )

            num_tokens_padded = batch_desc.num_tokens
            num_reqs_padded = batch_desc.num_reqs if batch_desc.num_reqs is not None else num_reqs
            ubatch_slices, ubatch_slices_padded = maybe_create_ubatch_slices(
                should_ubatch,
                num_scheduled_tokens_np,
                num_tokens_padded,
                num_reqs_padded,
                self.parallel_config.num_ubatches,
            )

            logger.debug(
                "ubatch_slices: %s, ubatch_slices_padded: %s",
                ubatch_slices,
                ubatch_slices_padded,
            )

            pad_attn = cudagraph_mode == CUDAGraphMode.FULL

            use_spec_decode = len(scheduler_output.scheduled_spec_decode_tokens) > 0
            ubatch_slices_attn = ubatch_slices_padded if pad_attn else ubatch_slices

            attn_metadata, spec_decode_common_attn_metadata = self._build_attention_metadata(
                num_tokens=num_tokens_unpadded,
                num_tokens_padded=num_tokens_padded if pad_attn else None,
                num_reqs=num_reqs,
                num_reqs_padded=num_reqs_padded if pad_attn else None,
                max_query_len=max_num_scheduled_tokens,
                ubatch_slices=ubatch_slices_attn,
                logits_indices=logits_indices,
                use_spec_decode=use_spec_decode,
                num_scheduled_tokens=scheduler_output.num_scheduled_tokens,
                cascade_attn_prefix_lens=cascade_attn_prefix_lens,
            )

            (
                input_ids,
                inputs_embeds,
                positions,
                intermediate_tensors,
                model_kwargs,
                ec_connector_output,
            ) = self._preprocess(
                scheduler_output,
                num_tokens_padded,
                intermediate_tensors,
            )

        # Set cudagraph mode to none if calc_kv_scales is true.
        # KV scales calculation involves dynamic operations that are incompatible
        # with CUDA graph capture.
        if self.calculate_kv_scales:
            cudagraph_mode = CUDAGraphMode.NONE
            # Mark KV scales as calculated after the first forward pass
            self.calculate_kv_scales = False

        # Run the model.
        # Use persistent buffers for CUDA graphs.
        with (
            set_forward_context(
                attn_metadata,
                self.vllm_config,
                num_tokens=num_tokens_padded,
                num_tokens_across_dp=num_tokens_across_dp,
                cudagraph_runtime_mode=cudagraph_mode,
                batch_descriptor=batch_desc,
                ubatch_slices=ubatch_slices_padded,
            ),
            record_function_or_nullcontext("Forward"),
            self.maybe_get_kv_connector_output(scheduler_output) as kv_connector_output,
        ):
            outputs = self._run_generation_model(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                model_kwargs=model_kwargs,
                logits_indices=logits_indices,
            )

        if outputs is None:
            self.execute_model_state = ExecuteModelState(
                scheduler_output,
                None,
                spec_decode_metadata,
                spec_decode_common_attn_metadata,
                None,
                None,
                None,
                ec_connector_output,
                cudagraph_stats,
                {},
            )
            self.kv_connector_output = kv_connector_output
            return None

        _, multimodal_outputs = self.extract_multimodal_outputs(outputs)
        self.execute_model_state = ExecuteModelState(
            scheduler_output,
            None,
            spec_decode_metadata,
            spec_decode_common_attn_metadata,
            None,
            None,
            None,
            ec_connector_output,
            cudagraph_stats,
            multimodal_outputs,
        )
        self.kv_connector_output = kv_connector_output
        return None

    @torch.inference_mode()
    def sample_tokens(
        self,
        grammar_output: GrammarOutput | None = None,
    ) -> OmniModelRunnerOutput | AsyncModelRunnerOutput | IntermediateTensors:
        # NOTE: Even though the model is non-autoregressive, we still need
        # this function to match the interface of the engine core.
        # In this case, this function
        kv_connector_output = self.kv_connector_output
        self.kv_connector_output = None

        if self.execute_model_state is None:
            # Nothing to do (PP non-final rank case), output isn't used.
            if not kv_connector_output:
                return None  # type: ignore[return-value]

            # In case of PP with kv transfer, we need to pass through the
            # kv_connector_output
            if kv_connector_output.is_empty():
                return EMPTY_MODEL_RUNNER_OUTPUT

            output = copy(EMPTY_MODEL_RUNNER_OUTPUT)
            output.kv_connector_output = kv_connector_output
            return output

        # Unpack ephemeral state.
        (
            scheduler_output,
            logits,
            spec_decode_metadata,
            spec_decode_common_attn_metadata,
            hidden_states,
            sample_hidden_states,
            aux_hidden_states,
            ec_connector_output,
            cudagraph_stats,
            multimodal_outputs,
        ) = self.execute_model_state
        self.execute_model_state = None

        self._finalize_streaming_callbacks()

        pooler_output: list[object] = []
        if isinstance(multimodal_outputs, torch.Tensor):
            assert multimodal_outputs.shape[0] == 1, (
                "model should return a single tensor, to return multiple tensors, use a dict"
            )
            assert multimodal_outputs.shape[0] == self.input_batch.num_reqs
            for i in range(self.input_batch.num_reqs):
                pooler_output.append({"model_outputs": multimodal_outputs[i].detach().to("cpu").contiguous()})
        elif isinstance(multimodal_outputs, list):
            assert len(multimodal_outputs) == 1, (
                "model should return a single list, to return multiple lists, use a dict"
            )
            for out in multimodal_outputs:
                pooler_output.append(
                    {"model_outputs": out.detach().to("cpu").contiguous() if out is not None else None}
                )
        elif isinstance(multimodal_outputs, dict):
            mm_payload = {}
            for key, out in multimodal_outputs.items():
                if out is not None and isinstance(out, torch.Tensor):
                    mm_payload[key] = out.detach().to("cpu").contiguous()
            pooler_output.append(mm_payload)
        else:
            raise RuntimeError("Unsupported diffusion output type")

        output = OmniModelRunnerOutput(
            req_ids=self.input_batch.req_ids,
            req_id_to_index=self.input_batch.req_id_to_index,
            sampled_token_ids=[],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=pooler_output,
            kv_connector_output=kv_connector_output,
            num_nans_in_logits={},
            cudagraph_stats=cudagraph_stats,
            ec_connector_output=ec_connector_output if self.supports_mm_inputs else None,
        )

        if not self.use_async_scheduling:
            return output

        return AsyncGPUModelRunnerOutput(
            model_runner_output=output,
            sampled_token_ids=torch.tensor([], device=self.device),
            invalid_req_indices=[],
            async_output_copy_stream=self.async_output_copy_stream,
            vocab_size=self.input_batch.vocab_size,
            logprobs_tensors=None,
        )

    def _finalize_streaming_callbacks(self) -> list[tuple[str, StreamingCodecChunk]]:
        callbacks_to_finalize = list(self._streaming_callbacks.keys())
        for req_id in callbacks_to_finalize:
            callback = self._streaming_callbacks.get(req_id)
            if callback is not None:
                callback.finalize()
                self.remove_streaming_callback(req_id)

        if self._streaming_emitter is not None and self._streaming_emitter.is_running():
            return []

        return self.get_streaming_chunks()

    def _run_generation_model(
        self,
        *,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor | None,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None,
        model_kwargs: dict,
        logits_indices: torch.Tensor,
    ) -> torch.Tensor | list[torch.Tensor]:
        """Run generation from codec codes to waveforms.

        Args:
            scheduler_output: Contains codec codes in input_ids or additional info
            intermediate_tensors: PP intermediate tensors if applicable

        Returns:
            Audio waveforms: [batch, 1, waveform_len] or list of tensors
        """
        streaming_callback = self._get_streaming_callback_for_batch()
        self._register_engine_streaming_callback(streaming_callback)

        kwargs = dict(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **model_kwargs,
            sampling_metadata=self.input_batch.sampling_metadata,
            logits_index=logits_indices,
            sampler=self.sampler,
        )

        try:
            if hasattr(self.model, "forward"):
                return self._model_forward(**kwargs)

            raise RuntimeError(
                "The loaded model does not expose diffusion interfaces 'sample', "
                "'forward', or 'diffuse'. Please implement one of them or adapt the runner."
            )
        except GenerationCancelled:
            return None
        finally:
            self._unregister_engine_streaming_callback()

    def _register_engine_streaming_callback(self, callback: StreamingTTSCallback | None) -> None:
        if callback is None:
            return

        talker = self._get_talker_model()
        if talker is not None:
            if not hasattr(talker, "_streaming_callbacks"):
                talker._streaming_callbacks = {}
            talker._streaming_callbacks["__engine__"] = callback

    def _unregister_engine_streaming_callback(self) -> None:
        talker = self._get_talker_model()
        if talker is not None and hasattr(talker, "_streaming_callbacks"):
            talker._streaming_callbacks.pop("__engine__", None)

    def _get_talker_model(self):
        try:
            if hasattr(self.model, "model"):
                qwen3_tts_model = self.model.model
                if hasattr(qwen3_tts_model, "model"):
                    qwen3_tts_for_gen = qwen3_tts_model.model
                    if hasattr(qwen3_tts_for_gen, "talker"):
                        return qwen3_tts_for_gen.talker
        except Exception:
            pass
        return None

    def _get_streaming_callback_for_batch(self) -> StreamingTTSCallback | None:
        for req_id in self.input_batch.req_ids:
            req_state = self.requests.get(req_id)
            if req_state is None:
                continue

            additional_info = getattr(req_state, "additional_information_cpu", None)

            if additional_info is not None and isinstance(additional_info, dict):
                streaming_tts = additional_info.get("streaming_tts")
                if streaming_tts is not None:
                    if isinstance(streaming_tts, list):
                        streaming_tts = streaming_tts[0] if streaming_tts else False
                    if streaming_tts:
                        global_req_id = additional_info.get("global_request_id")
                        if isinstance(global_req_id, list):
                            global_req_id = global_req_id[0] if global_req_id else None
                        emit_req_id = global_req_id or req_id

                        callback = self._streaming_callbacks.get(emit_req_id)
                        if callback is None:
                            callback = self.setup_streaming_callback(emit_req_id)
                        return callback

        return None

    @torch.inference_mode()
    def _dummy_sampler_run(self, hidden_states: torch.Tensor) -> None:
        logger.warning("Dummy sampler run is not implemented for generation model")
        return None

    @torch.inference_mode()
    def _dummy_run(
        self,
        num_tokens: int,
        cudagraph_runtime_mode: CUDAGraphMode | None = None,
        force_attention: bool = False,
        uniform_decode: bool = False,
        allow_microbatching: bool = True,
        skip_eplb: bool = False,
        is_profile: bool = False,
        create_mixed_batch: bool = False,
        remove_lora: bool = True,
        activate_lora: bool = False,
        is_graph_capturing: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run a dummy forward pass to warm up/profile run or capture the
        CUDA graph for the model.

        Args:
            num_tokens: Number of tokens to run the dummy forward pass.
            cudagraph_runtime_mode: used to control the behavior.
                - if not set will determine the cudagraph mode based on using
                    the self.cudagraph_dispatcher.
                - CUDAGraphMode.NONE: No cudagraph, for warm up and profile run
                - CUDAGraphMode.PIECEWISE: Piecewise cudagraph.
                - CUDAGraphMode.FULL: Full cudagraph, attention metadata is
                    needed.
            force_attention: If True, always create attention metadata. Used to
                warm up attention backend when mode is NONE.
            uniform_decode: If True, the batch is a uniform decode batch.
            skip_eplb: If True, skip EPLB state update.
            is_profile: If True, this is a profile run.
            create_mixed_batch: If True, create a mixed batch with both decode
                (1 token) and prefill (multiple tokens) requests.
            remove_lora: If False, dummy LoRAs are not destroyed after the run
            activate_lora: If False, dummy_run is performed without LoRAs.
        """
        if supports_mm_encoder_only(self.model):
            # The current dummy run only covers LM execution, so we can skip it.
            # mm encoder dummy run may need to add in the future.
            return torch.tensor([]), torch.tensor([])

        assert cudagraph_runtime_mode is None or cudagraph_runtime_mode.valid_runtime_modes()

        # If cudagraph_mode.decode_mode() == FULL and
        # cudagraph_mode.separate_routine(). This means that we are using
        # different graphs and/or modes for mixed prefill-decode batches vs.
        # uniform decode batches. A uniform decode batch means that all
        # requests have identical query length, except a potential virtual
        # request (shorter) in the batch account for padding.
        # Uniform decode batch could either be common pure decode, where
        # max_query_len == 1, or speculative decode, where
        # max_query_len == 1 + num_spec_decode_tokens.

        # When setting max_query_len = 1, we switch to and capture the optimized
        # routine of FA2 for pure decode, i.e., Flashdecode + an optimization
        # for GQA/MQA.
        max_query_len = self.uniform_decode_query_len if uniform_decode else num_tokens

        # Set num_scheduled_tokens based on num_tokens and max_num_seqs
        # for dummy run with LoRA so that the num_reqs collectively
        # has num_tokens in total.
        assert num_tokens <= self.scheduler_config.max_num_batched_tokens
        max_num_reqs = self.scheduler_config.max_num_seqs
        if create_mixed_batch:
            assert not uniform_decode
            # Create mixed batch:
            # first half decode tokens, second half one prefill
            num_decode_tokens = min(max_num_reqs - 1, num_tokens // 2)
            num_prefill_tokens = num_tokens - num_decode_tokens
            num_reqs = num_decode_tokens + 1

            # Create decode requests (1 token each) followed by prefill request
            num_scheduled_tokens_list = [1] * num_decode_tokens + [num_prefill_tokens]
            # Note: Overriding max_query_len to be the prefill tokens
            max_query_len = num_prefill_tokens
        elif uniform_decode:
            assert not create_mixed_batch
            num_reqs = min(max_num_reqs, cdiv(num_tokens, max_query_len))
            num_scheduled_tokens_list = [max_query_len] * num_reqs
            if num_tokens % max_query_len != 0:
                num_scheduled_tokens_list[-1] = num_tokens % max_query_len
        else:
            num_reqs = min(num_tokens, max_num_reqs)
            min_tokens_per_req = num_tokens // num_reqs
            num_scheduled_tokens_list = [min_tokens_per_req] * num_reqs
            num_scheduled_tokens_list[-1] += num_tokens % num_reqs

        assert sum(num_scheduled_tokens_list) == num_tokens
        assert len(num_scheduled_tokens_list) == num_reqs
        num_scheduled_tokens = np.array(num_scheduled_tokens_list, dtype=np.int32)
        num_tokens_unpadded = int(num_scheduled_tokens.sum())

        num_sampled_tokens = np.ones(num_reqs, dtype=np.int32)

        _cudagraph_mode, batch_desc, should_ubatch, num_tokens_across_dp, _ = (
            self._determine_batch_execution_and_padding(
                num_tokens=num_tokens_unpadded,
                num_reqs=num_reqs,
                num_scheduled_tokens_np=num_scheduled_tokens,
                max_num_scheduled_tokens=max_query_len,
                use_cascade_attn=False,
                allow_microbatching=allow_microbatching,
                force_eager=is_profile or (cudagraph_runtime_mode == CUDAGraphMode.NONE),
                # `force_uniform_decode` is used for cudagraph capture; because for
                # capturing mixed prefill-decode batches, we sometimes use
                # num_tokens == num_reqs which looks like a uniform decode batch to the
                # dispatcher; but we actually want to capture a piecewise cudagraph
                force_uniform_decode=uniform_decode,
                # `force_has_lora` is used for cudagraph capture; because LoRA is
                # activated later in the context manager, but we need to know the
                # LoRA state when determining the batch descriptor for capture
                force_has_lora=activate_lora,
            )
        )

        if cudagraph_runtime_mode is None:
            cudagraph_runtime_mode = _cudagraph_mode
        else:
            assert cudagraph_runtime_mode == _cudagraph_mode, (
                f"Cudagraph runtime mode mismatch in dummy_run. "
                f"Expected {_cudagraph_mode}, but got {cudagraph_runtime_mode}."
            )

        num_tokens_padded = batch_desc.num_tokens
        num_reqs_padded = batch_desc.num_reqs if batch_desc.num_reqs is not None else num_reqs
        ubatch_slices, ubatch_slices_padded = maybe_create_ubatch_slices(
            should_ubatch,
            num_scheduled_tokens,
            num_tokens_padded,
            num_reqs_padded,
            self.vllm_config.parallel_config.num_ubatches,
        )
        logger.debug(
            "ubatch_slices: %s, ubatch_slices_padded: %s",
            ubatch_slices,
            ubatch_slices_padded,
        )

        attn_metadata: PerLayerAttnMetadata | None = None

        # If force_attention is True, we always capture attention. Otherwise,
        # it only happens for cudagraph_runtime_mode=FULL.
        if force_attention or cudagraph_runtime_mode == CUDAGraphMode.FULL:
            if create_mixed_batch:
                # In the mixed batch mode (used for FI warmup), we use
                # shorter sequence lengths to run faster.
                # TODO(luka) better system for describing dummy batches
                seq_lens = [1] * num_decode_tokens + [num_prefill_tokens + 1]
            else:
                seq_lens = max_query_len  # type: ignore[assignment]
            self.seq_lens.np[:num_reqs] = seq_lens
            self.seq_lens.np[num_reqs:] = 0
            self.seq_lens.copy_to_gpu()

            cum_num_tokens, _ = self._get_cumsum_and_arange(num_scheduled_tokens)
            self.query_start_loc.np[1 : num_reqs + 1] = cum_num_tokens
            self.query_start_loc.copy_to_gpu()

            pad_attn = cudagraph_runtime_mode == CUDAGraphMode.FULL
            attn_metadata, _ = self._build_attention_metadata(
                num_tokens=num_tokens_unpadded,
                num_reqs=num_reqs_padded,
                max_query_len=max_query_len,
                ubatch_slices=ubatch_slices_padded if pad_attn else ubatch_slices,
                for_cudagraph_capture=is_graph_capturing,
            )

        with self.maybe_dummy_run_with_lora(
            self.lora_config,
            num_scheduled_tokens,
            num_sampled_tokens,
            activate_lora,
            remove_lora,
        ):
            # Make sure padding doesn't exceed max_num_tokens
            assert num_tokens_padded <= self.max_num_tokens
            model_kwargs = self._init_model_kwargs()
            if self.supports_mm_inputs and not self.model_config.is_encoder_decoder:
                input_ids, inputs_embeds = self._prepare_mm_inputs(num_tokens_padded)

                model_kwargs = {
                    **model_kwargs,
                    **self._dummy_mm_kwargs(num_reqs),
                }
            elif self.enable_prompt_embeds:
                input_ids = None
                inputs_embeds = self.inputs_embeds.gpu[:num_tokens_padded]
                model_kwargs = self._init_model_kwargs()
            else:
                input_ids = self.input_ids.gpu[:num_tokens_padded]
                inputs_embeds = None

            if self.uses_mrope:
                positions = self.mrope_positions.gpu[:, :num_tokens_padded]
            elif self.uses_xdrope_dim > 0:
                positions = self.xdrope_positions.gpu[:, :num_tokens_padded]
            else:
                positions = self.positions.gpu[:num_tokens_padded]

            if get_pp_group().is_first_rank:
                intermediate_tensors = None
            else:
                if self.intermediate_tensors is None:
                    self.intermediate_tensors = self.model.make_empty_intermediate_tensors(
                        batch_size=self.max_num_tokens,
                        dtype=self.model_config.dtype,
                        device=self.device,
                    )

                intermediate_tensors = self.sync_and_slice_intermediate_tensors(num_tokens_padded, None, False)

            if ubatch_slices_padded is not None:
                # Adjust values to reflect a single ubatch.
                # TODO(sage,lucas): this is cruft that should be addressed in
                #  the padding refactor.
                num_tokens_padded = ubatch_slices_padded[0].num_tokens
                if num_tokens_across_dp is not None:
                    num_tokens_across_dp[:] = num_tokens_padded

            with (
                self.maybe_randomize_inputs(input_ids, inputs_embeds),
                set_forward_context(
                    attn_metadata,
                    self.vllm_config,
                    num_tokens=num_tokens_padded,
                    num_tokens_across_dp=num_tokens_across_dp,
                    cudagraph_runtime_mode=cudagraph_runtime_mode,
                    batch_descriptor=batch_desc,
                    ubatch_slices=ubatch_slices_padded,
                ),
            ):
                outputs = self.model(
                    input_ids=input_ids,
                    positions=positions,
                    intermediate_tensors=intermediate_tensors,
                    inputs_embeds=inputs_embeds,
                    **model_kwargs,
                )

            if self.use_aux_hidden_state_outputs:
                hidden_states, _ = outputs
            else:
                hidden_states = outputs
            hidden_states, multimodal_outputs = self.extract_multimodal_outputs(hidden_states)
            if self.speculative_config and self.speculative_config.use_eagle():
                assert isinstance(self.drafter, EagleProposer)
                # Eagle currently only supports PIECEWISE cudagraphs.
                # Therefore only use cudagraphs if the main model uses PIECEWISE
                # NOTE(lucas): this is a hack, need to clean up.
                use_cudagraphs = (
                    (is_graph_capturing and cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE)
                    or (not is_graph_capturing and cudagraph_runtime_mode != CUDAGraphMode.NONE)
                ) and not self.speculative_config.enforce_eager

                # Note(gnovack) - We need to disable cudagraphs for one of the two
                # lora cases when cudagraph_specialize_lora is enabled. This is a
                # short term mitigation for issue mentioned in
                # https://github.com/vllm-project/vllm/issues/28334
                if self.compilation_config.cudagraph_specialize_lora and activate_lora:
                    use_cudagraphs = False

                self.drafter.dummy_run(
                    num_tokens,
                    use_cudagraphs=use_cudagraphs,
                    is_graph_capturing=is_graph_capturing,
                )

        # We register layerwise NVTX hooks here after the first dynamo tracing is
        # done to avoid nvtx operations in hook functions being traced by
        # torch dynamo and causing graph breaks.
        # Note that for DYNAMO_ONCE and VLLM_COMPILE mode,
        # compiled model's dynamo tracing is only done once and the compiled model's
        # __call__ function is replaced by calling the compiled function.
        # So it's safe to register hooks here. Hooks will be registered to
        # both compiled and uncompiled models but they will never
        # be called on the compiled model execution path.
        self._register_layerwise_nvtx_hooks()

        # This is necessary to avoid blocking DP.
        # For dummy runs, we typically skip EPLB since we don't have any real
        # requests to process.
        # However, in DP settings, there may be cases when some DP ranks do
        # not have any requests to process, so they're executing dummy batches.
        # In such cases, we still have to trigger EPLB to make sure
        # ranks execute the rearrangement in synchronization.
        if not skip_eplb:
            self.eplb_step(is_dummy=True, is_profile=is_profile)

        return hidden_states, None

    def profile_run(self) -> None:
        # Profile with multimodal encoder & encoder cache.
        if self.supports_mm_inputs:
            mm_config = self.model_config.multimodal_config
            if mm_config is not None and mm_config.skip_mm_profiling:
                logger.info("Skipping memory profiling for multimodal encoder and encoder cache.")
            else:
                mm_budget = self.mm_budget
                assert mm_budget is not None

                if (encoder_budget := mm_budget.get_encoder_budget()) > 0:
                    # NOTE: Currently model is profiled with a single non-text
                    # modality with the max possible input tokens even when
                    # it supports multiple.
                    dummy_modality = mm_budget.get_modality_with_max_tokens()
                    max_mm_items_per_batch = mm_budget.max_items_per_batch_by_modality[dummy_modality]

                    logger.info(
                        "Encoder cache will be initialized with a budget of "
                        "%s tokens, and profiled with %s %s items of the "
                        "maximum feature size.",
                        encoder_budget,
                        max_mm_items_per_batch,
                        dummy_modality,
                    )

                    # Create dummy batch of multimodal inputs.
                    batched_dummy_mm_inputs = self._get_mm_dummy_batch(
                        dummy_modality,
                        max_mm_items_per_batch,
                    )

                    # Run multimodal encoder.
                    dummy_encoder_outputs = self.model.embed_multimodal(**batched_dummy_mm_inputs)

                    sanity_check_mm_encoder_outputs(
                        dummy_encoder_outputs,
                        expected_num_items=max_mm_items_per_batch,
                    )

                    # NOTE: This happens when encoder cache needs to store
                    # the embeddings that encoder outputs are scattered onto.
                    # In this case we create dummy embeddings of size
                    # (max_tokens_for_modality, hidden_size) and scatter
                    # encoder output into it.
                    encoder_output_shape = dummy_encoder_outputs[0].shape
                    max_mm_tokens_per_item = mm_budget.max_tokens_by_modality[dummy_modality]
                    if encoder_output_shape[0] < max_mm_tokens_per_item:
                        encoder_hidden_size = encoder_output_shape[-1]
                        expanded_outputs = []
                        for output in dummy_encoder_outputs:
                            expanded = output.new_zeros((max_mm_tokens_per_item, encoder_hidden_size))
                            num_tokens = output.shape[0]
                            expanded[:num_tokens].copy_(output)
                            expanded_outputs.append(expanded)

                        dummy_encoder_outputs = expanded_outputs

                    # Cache the dummy encoder outputs.
                    self.encoder_cache["tmp"] = dict(enumerate(dummy_encoder_outputs))

        # Add `is_profile` here to pre-allocate communication buffers
        hidden_states, _ = self._dummy_run(self.max_num_tokens, is_profile=True)
        output = None
        self._sync_device()
        del hidden_states
        self.encoder_cache.clear()
        gc.collect()
