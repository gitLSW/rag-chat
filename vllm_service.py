from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

LLM_MODEL = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'

class LLMService:
    """
    A Python class that uses vLLM to query a Llama model in a parallel batch of prompts.
    The class initializes the vLLM LLM engine using the specified model name and other
    optional engine keyword arguments. The `query` method accepts a list of prompt strings,
    sends them together to the engine, and returns a list of generated responses.

    Attributes:
        llm (LLM): The vLLM engine instance.
        sampling_config (Dict[str, Any]): A dictionary of sampling parameters.
    """
    def __init__(self):
        """
        Initializes the vLLM LLM instance.
        """
        # TODO: SET FOR PROD
        # Initialize the vLLM engine.
        self.llm = LLM(
            # Model parameters (https://docs.vllm.ai/en/latest/api/offline_inference/llm.html)
            model=LLM_MODEL,  # Default model name or path
                # tokenizer=None,  # Defaults to the model's tokenizer
                # tokenizer_mode="auto",  # Automatically selects the tokenizer mode
            trust_remote_code=True,  # Does not trust remote code execution by default
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
            tensor_parallel_size=1,  # Split LLM Layer computations horizontally across N GPUs
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
        self.tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)


    def query(self, prompts):
        """
        Generates responses for a batch of prompts.

        This method accepts a list of prompt strings and passes them as a batch to
        the vLLM engine's generate() method. The engine processes all incoming prompts
        from all threads concurrently using its internal intelligent batching mechanism.

        Args:
            prompts (List[str]): A list of prompt strings to process in parallel.

        Returns:
            List[str]: A list of generated response strings corresponding to each prompt.
        """
        # Define decoding/generation parameters
        sampling_params = SamplingParams(
            temperature=0.3,
            top_p=0.6,
            max_tokens=32,  # Controls how much *new* text to generate per prompt
            stop=["\n\n", "\n", "Q:", "###"]
        )

        print(prompts)

        # Generate outputs for all prompts in the batch.
        outputs = self.llm.generate(prompts, sampling_params)

        # Extract the generated text from the output objects.
        # Each output object typically contains the prompt and one or more outputs.
        responses = [output.outputs[0].text for output in outputs]
        return responses



# # List of sample prompts.
# batch_prompts = [
#     "Who was Caesar?",
#     "Who was ALexander the Great?",
#     "Which lands did Charlemagne own?",
#     "Who defeated King Sigismund ?",
# ]

# # Initialize our batch query object.
# llm_service = LLMService()

# # Query the model with the batch of prompts.
# results = llm_service.query(batch_prompts)

# for prompt, output in zip(batch_prompts, results):
#     print(f"Prompt: {prompt!r}\nGenerated text: {output!r}\n")