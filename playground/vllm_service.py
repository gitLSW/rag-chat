# from vllm import LLM, SamplingParams

# LLM_MODEL = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-32B'

# class LLMService:
#     """
#     A Python class that uses vLLM to query a Llama model in a parallel batch of prompts.
#     The class initializes the vLLM LLM engine using the specified model name and other
#     optional engine keyword arguments. The `query` method accepts a list of prompt strings,
#     sends them together to the engine, and returns a list of generated responses.

#     Attributes:
#         llm (LLM): The vLLM engine instance.
#         sampling_config (Dict[str, Any]): A dictionary of sampling parameters.
#     """
#     def __init__(self):
#         """
#         Initializes the LLamaBatchQuery instance.
#         """

#         # Initialize the vLLM engine.
#         # See documentation: https://docs.vllm.ai/en/latest/getting_started/quickstart.html
#         self.llm = LLM(model=LLM_MODEL,
#                        task="generate",
#                        gpu_memory_utilization=0.8,
#                        max_num_batched_tokens=512,
#                        max_num_seqs=64,
#                     #    cpu_offload_gb=10,
#                        tensor_parallel_size=1,  # change this value based on your hardware
#                        dtype="auto",
#                        trust_remote_code=True)

#     def query(self, prompts):
#         """
#         Generates responses for a batch of prompts.

#         This method accepts a list of prompt strings, bundles them into a single batch,
#         and passes them to the vLLM engine's generate() method. The engine processes
#         the prompts concurrently using its internal intelligent batching mechanism.

#         Args:
#             prompts (List[str]): A list of prompt strings to process in parallel.

#         Returns:
#             List[str]: A list of generated response strings corresponding to each prompt.
#         """
#         # Create a SamplingParams instance using the provided config.
#         sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=50)
#         # Generate outputs for all prompts in the batch.
#         outputs = self.llm.generate(prompts, sampling_params)

#         # Extract the generated text from the output objects.
#         # Each output object typically contains the prompt and one or more outputs.
#         responses = [output.outputs[0].text for output in outputs]
#         return responses



# # List of sample prompts.
# batch_prompts = [
#     "Who was Caesar?",
#     "Who was ALexander the Great?",
#     "Which lands did Charlemagne own?",
#     "Who defeated King Sigismund ?",
# ]

# # Initialize our batch query object.
# llama_query = LLMService()

# # Query the model with the batch of prompts.
# results = llama_query.query(batch_prompts)

# for prompt, output in zip(batch_prompts, results):
#     print(f"Prompt: {prompt!r}\nGenerated text: {output!r}\n")