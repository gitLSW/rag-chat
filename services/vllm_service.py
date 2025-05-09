import uuid
import asyncio
from typing import List, Deque, Dict, AsyncGenerator, Optional
from ..get_env_var import get_env_var
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoConfig
from collections import defaultdict, deque

# Load environment variables
LLM_MODEL = get_env_var('LLM_MODEL')
IS_PRODUCTION = get_env_var('IS_PRODUCTION')
DEFAULT_SAMPLING_PARAMS = SamplingParams(temperature=0.3, top_p=0.6, max_tokens=256)

# Initialize the vLLM engine.
llm = LLM(
    # Model parameters (https://docs.vllm.ai/en/latest/api/offline_inference/llm.html)
    model=LLM_MODEL,  # Default model name or path
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

@dataclass
class RequestState:
    sampling_params: SamplingParams
    chunk_state: Dict[str, Deque] = defaultdict(lambda: deque(maxlen=max_tokens)) # [chunk_req_id: chunk_token_ids]


class LLMService:
    def __init__(self):
        self._ongoing_requests: Dict[str, RequestState] = {}

    async def query(
        self,
        prompt: str,
        context: Optional[str] = None,
        history: Optional[str] = None,
        req_id = uuid.uuid4(),
        sampling_params: Optional[SamplingParams] = None,
        allow_chunking: bool = True
    ) -> AsyncGenerator[str, None]:
        if sampling_params is None:
            sampling_params = DEFAULT_SAMPLING_PARAMS

        chunks = self._get_chunks(prompt, context, history, allow_chunking)
        
        self._ongoing_requests[req_id] = RequestState(sampling_params)

        # Concurrent generation and sequential yielding
        queues: List[asyncio.Queue] = [asyncio.Queue() for _ in chunks]

        # Start all generators
        tasks = [asyncio.create_task(self._stream_chunk(req_id, i, chunk, queues, sampling_params)) for i, chunk in enumerate(chunks)]

        # Stream outputs sequentially in chunk order
        for chunk_queue in queues:
            while True:
                text_token = await chunk_queue.get()
                if text_token is None:
                    break
                yield text_token # Stream all chunk queues out the same stream

        # Ensure all tasks are completed
        await asyncio.gather(*tasks)
        del self._ongoing_requests[req_id]


    async def _stream_chunk(self, req_id, chunk_index, chunk, queues, sampling_params):
        chunk_req_id = f'{req_id}-chunk{chunk_index}'
        try:
            async for output in llm.generate(
                prompt_token_ids=chunk,
                sampling_params=sampling_params,
                request_id=chunk_req_id,
                stream=True
            ):
                self._ongoing_requests[req_id].chunk_states[chunk_req_id].append(output.outputs[0].token_ids)
                await queues[chunk_index].put(output.outputs[0].text)
        finally:
            await queues[chunk_index].put(None)  # Signal that the stream is done


    def cancel(self, req_id):
        """
        Cancel an ongoing request by its request ID.
        """
        try:
            llm.abort(req_id)
        except Exception as e:
            # Log or handle specific errors as needed
            raise RuntimeError(f"Failed to cancel request {req_id}: {str(e)}")
        

    async def resume(self, req_id):
        req_state = self._ongoing_requests.get(req_id)
        if not req_state:
            raise ValueError('No request found for ' + req_id)
        
        sampling_params = req_state.sampling_params
        chunk_states = req_state.chunk_states

        # Concurrent generation and sequential yielding
        queues: List[asyncio.Queue] = [asyncio.Queue() for _ in chunk_states.keys()]

        # Start all generators
        tasks = [asyncio.create_task(self._stream_chunk(req_id, i, chunk, queues, sampling_params)) for i, chunk in enumerate(chunk_states.values())]

        # Stream outputs sequentially in chunk order
        for chunk_queue in queues:
            while True:
                text_token = await chunk_queue.get()
                if text_token is None:
                    break
                yield text_token # Stream all chunk queues out the same stream

        # Ensure all tasks are completed
        await asyncio.gather(*tasks)
        del self._ongoing_requests[req_id]
        

    def _get_chunks(self, prompt, context, history, allow_chunking):
        # Tokenize prompt and optional context
        prompt_ids = tokenizer.encode('\n\n' + prompt + '\n\n', add_special_tokens=False)
        context_ids = tokenizer.encode(context, add_special_tokens=False) if context else []
        history_ids = tokenizer.encode(history, add_special_tokens=False) if history else []

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