import os
import asyncio
import re
from typing import List, AsyncGenerator, Optional, Dict
from dataclasses import dataclass
from datetime import datetime
from fastapi import WebSocket
from ..get_env_var import get_env_var
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoConfig
from collections import defaultdict
import json

# Load environment variables
LLM_MODEL = get_env_var('LLM_MODEL')
IS_PRODUCTION = get_env_var('IS_PRODUCTION')

class LLMService:
    # Initialize the vLLM engine.
    llm = LLM(
        # Model parameters (https://docs.vllm.ai/en/latest/api/offline_inference/llm.html)
        model=self.model,  # Default model name or path
            # tokenizer=None,  # Defaults to the model's tokenizer
            # tokenizer_mode="auto",  # Automatically selects the tokenizer mode
        trust_remote_code=not IS_PRODUCTION,  # Does not trust remote code execution by default
            # download_dir=None,  # Uses the default download directory
            # load_format="auto",  # Automatically detects the model's weight format
            # dtype="auto",  # Determines the appropriate data type automatically
        # quantization=None,  # No quantization method applied by default
            # enforce_eager=False,  # Allows the use of CUDA graphs for execution
            # max_seq_len_to_capture=8192,  # Maximum sequence length for CUDA graph capture
            # seed=None,  # No specific random seed applied
        gpu_memory_utilization=0.8,  # Utilizes up to % of GPU memory
            # swap_space=4,  # Allocates 4 GiB of CPU memory per GPU for swap space
            # cpu_offload_gb=0,  # No CPU offloading of model weights by default
            # block_size=None,  # Uses the default block size
            # disable_custom_all_reduce=False,  # Enables custom all-reduce operations
            # disable_async_output_proc=False,  # Allows asynchronous output processing
            # hf_overrides=None,  # No additional HuggingFace configuration overrides applied
            # compilation_config=0,  # Default torch.compile optimization level

        # Additional engine arguments (https://docs.vllm.ai/en/latest/serving/engine_args.html#engine-args)
        task="generate",  # Automatically selects the task based on the model
            # hf_config_path=None,  # Defaults to the model's configuration path
            # skip_tokenizer_init=False,  # Initializes the tokenizer by default
            # revision=None,  # Uses the default model version
            # code_revision=None,  # Uses the default code revision
            # tokenizer_revision=None,  # Uses the default tokenizer revision
            # allowed_local_media_path=None,  # No local media paths allowed by default
            # config_format="auto",  # Automatically detects the model's config format
            # kv_cache_dtype="auto",  # Uses the model's data type for KV cache
        # max_model_len=512,  # Automatically derived from the model config - default: None
            # guided_decoding_backend="xgrammar",  # Default guided decoding backend
            # logits_processor_pattern=None,  # No logits processor pattern specified
            # model_impl="auto",  # Automatically selects the model implementation
            # distributed_executor_backend=None,  # Defaults based on parallel sizes and GPU availability
            # pipeline_parallel_size=1,  # Single pipeline stage
        # tensor_parallel_size=1,  # Split LLM Layer computations horizontally across N GPUs
            # data_parallel_size=1,  # Single data parallel replica
            # enable_expert_parallel=False,  # Expert parallelism disabled by default
            # max_parallel_loading_workers=None,  # No specific limit on parallel loading workers
            # ray_workers_use_nsight=False,  # Nsight profiling for Ray workers disabled
            # enable_prefix_caching=False,  # Prefix caching disabled by default
            # prefix_caching_hash_algo="builtin",  # Default hash algorithm for prefix caching
            # disable_sliding_window=False,  # Sliding window enabled by default
            # use_v2_block_manager=False,  # Deprecated; no effect on behavior
            # num_lookahead_slots=0,  # No lookahead slots by default
        # max_num_batched_tokens=256,  # No specific limit on batched tokens - default: None
            # max_num_partial_prefills=1,  # Single partial prefill allowed
            # max_long_partial_prefills=1,  # Single long partial prefill allowed
            # long_prefill_token_threshold=0,  # No threshold for long prefill tokens
        # max_num_seqs=4,  # No specific limit on sequences per iteration - default: None
            # max_logprobs=20,  # Maximum number of log probabilities to return
            # disable_log_stats=False,  # Logging statistics enabled by default
            # rope_scaling=None,  # No RoPE scaling configuration specified
            # rope_theta=None,  # No RoPE theta specified
            # tokenizer_pool_size=0,  # Synchronous tokenization by default
            # tokenizer_pool_type="ray",  # Default tokenizer pool type
            # tokenizer_pool_extra_config=None,  # No extra config for tokenizer pool
            # limit_mm_per_prompt=None,  # Defaults to 1 for each multimodal input
            # mm_processor_kwargs=None,  # No overrides for multimodal processing
            # disable_mm_preprocessor_cache=False,  # Multimodal preprocessor cache enabled
            # enable_lora=False,  # LoRA adapters handling disabled by default
            # enable_lora_bias=False,  # LoRA bias disabled by default
            # max_loras=1,  # Maximum of 1 LoRA in a single batch
            # max_lora_rank=16,  # Maximum LoRA rank
            # lora_extra_vocab_size=256,  # Extra vocabulary size for LoRA adapters
            # lora_dtype="auto",  # LoRA data type defaults to base model dtype
            # long_lora_scaling_factors=None,  # No specific scaling factors for Long LoRA
            # max_cpu_loras=None,  # No specific limit on CPU-stored LoRAs
            # fully_sharded_loras=False,  # Partial sharding for LoRA computation
            # enable_prompt_adapter=False,  # PromptAdapters handling disabled by default
            # max_prompt_adapters=1,  # Maximum of 1 PromptAdapter in a batch
            # max_prompt_adapter_token=0,  # No tokens allocated for PromptAdapters
            # device="auto",  # Automatically selects the device for execution
            # num_scheduler_steps=1,  # Single forward step per scheduler call
            # use_tqdm_on_load=True,  # Progress bar enabled when loading model weights
            # multi_step_stream_outputs=True,  # Streams outputs at the end of all steps
            # scheduler_delay_factor=0.0,  # No delay applied before scheduling next prompt
            # enable_chunked_prefill=False,  # Chunked prefill disabled by default
            # speculative_config=None  # No configuration for speculative decoding
    )
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
    llm_config = AutoConfig.from_pretrained(LLM_MODEL)
    max_tokens = getattr(llm_config, "max_position_embeddings", tokenizer.model_max_length)

    def __init__(self, session_id):
        self.session_id = session_id


    async def query(
        self,
        prompt: str,
        context: Optional[str] = None,
        history: Optional[str] = None,
        req_id: str = None,
        sampling_params: Optional[SamplingParams] = None,
        allow_chunking: bool = True,
        process_chunks_concurrently: bool = True,
    ) -> AsyncGenerator[str, None]:
        if sampling_params is None:
            sampling_params = SamplingParams(temperature=0.3, top_p=0.6, max_tokens=256)

        chunks = self._get_chunks(prompt, context, history, allow_chunking)

        # If only one chunk, always stream each new token
        if len(chunks) == 1:
            async for output in self.llm.generate(
                prompt_token_ids=chunks[0],
                sampling_params=sampling_params,
                request_id=req_id,
                stream=True
            ):
                yield output.outputs[0].text
            return

        # Multiple chunks: concurrent full or sequential streaming
        if process_chunks_concurrently:
            async def _generate_full(token_ids) -> str:
                output = await asyncio.to_thread(
                    self.llm.generate,
                    token_ids,
                    sampling_params,
                    request_id=req_id,
                    stream=False
                )
                return output.outputs[0].text
            
            tasks = [
                asyncio.create_task(_generate_full(chunk, sampling_params))
                for chunk in chunks
            ]
            for task in tasks:
                text = await task
                yield text
        else:
            for chunk in chunks:
                async for output in self.llm.generate(
                    prompt_token_ids=chunk,
                    sampling_params=sampling_params,
                    request_id=req_id,
                    stream=True
                ):
                    yield output.outputs[0].text


    def cancel(self, req_id: str):
        """
        Cancel an ongoing request by its request ID.
        """
        try:
            self.llm.abort(req_id)
        except Exception as e:
            # Log or handle specific errors as needed
            raise RuntimeError(f"Failed to cancel request {req_id}: {str(e)}")
        

    def _get_chunks(self, prompt, context, history, allow_chunking):
        # Tokenize prompt and optional context
        prompt_ids = self.tokenizer.encode('\n\n' + prompt + '\n\n', add_special_tokens=False)
        context_ids = self.tokenizer.encode(context, add_special_tokens=False) if context else []
        history_ids = self.tokenizer.encode(history, add_special_tokens=False) if history else []

        if self.max_tokens < len(prompt_ids):
            raise ValueError('Prompt is too long for LLM context frame')

        # Enforce prompt size <= 1/4 frame
        quarter_frame = self.max_tokens // 4
        if self.max_tokens < len(prompt_ids) + len(context_ids): # chunking occurrs
            if quarter_frame < len(prompt_ids):
                raise ValueError('Prompt is too long for LLM context frame')
            
            if not allow_chunking:
                raise ValueError('Prompt and context are too long and chunking was disallowed')
            
        chunks = []
        available_per_chunk = self.max_tokens - len(prompt_ids)

        # Take the last chunk from the chat history and append the prompt
        if history_ids:
            chunk_ids = history_ids[-available_per_chunk:] + prompt_ids
            if len(chunk_ids) + len(context_ids) < self.max_tokens:
                chunks.append(chunk_ids + context_ids)
                return chunks # Everything fits in one chunk
            chunks.append(chunk_ids)
        
        # Build chunks: always include at least the prompt
        if context_ids:
            # Available space per chunk after the prompt
            for start in range(0, len(context_ids), available_per_chunk):
                end = min(start + available_per_chunk, len(context_ids))
                chunk_ids = prompt_ids + context_ids[start:end]
                chunks.append(chunk_ids)

        if not chunks:
            chunks.append([prompt_ids])

        return chunks