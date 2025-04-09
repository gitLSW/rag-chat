# import asyncio
# from ollama import AsyncClient

# LLM_MODEL = 'deepseek-r1:32b'  # The specific LLM model used to answer questions

# class LLMBatchProcessor:
#     def __init__(self, model=LLM_MODEL, url="http://localhost:11434", batch_timeout=0.1, max_batch_size=4):
#         """
#         Args:
#             model (str): Model name to use.
#             base_url (str): Ollama service URL.
#             batch_timeout (float): Time (in seconds) to wait for additional prompts before processing.
#             max_batch_size (int): Maximum number of prompts to batch.
#         """
#         self.model = model
#         self.client = AsyncClient(host=url)
#         self.batch_timeout = batch_timeout
#         self.max_batch_size = max_batch_size
#         self._batch = []  # Queue of (prompt, Future) tuples
#         self._batch_lock = asyncio.Lock()

#     async def _drain_batch(self):
#         """
#         Drains the current batch and processes it.
#         Returns:
#             List[str]: Responses (in the same order as the prompts).
#         """
#         # Get the current batch and reset the queue
#         async with self._batch_lock:
#             if not self._batch:
#                 return []
#             current_batch = self._batch.copy()
#             self._batch = []
#         # Process all prompts in current_batch concurrently
#         tasks = [
#             self.client.chat(
#                 model=self.model,
#                 messages=[{"role": "user", "content": prompt}],
#                 stream=False
#             )
#             for prompt, _ in current_batch
#         ]
#         results = await asyncio.gather(*tasks)
#         responses = [res.message.content for res in results]
#         # Set the result of each corresponding future
#         for (_, fut), response in zip(current_batch, responses):
#             if not fut.done():
#                 fut.set_result(response)
#         return responses

#     async def process(self, prompt: str) -> str:
#         """
#         Adds a prompt to the batch and returns its response.
#         The batch is processed when either the max batch size is reached or after the timeout.
#         """
#         loop = asyncio.get_running_loop()
#         future = loop.create_future()
#         async with self._batch_lock:
#             self._batch.append((prompt, future))
#             current_size = len(self._batch)

#         if current_size >= self.max_batch_size:
#             # Drain the batch immediately if we've reached the maximum size.
#             await self._drain_batch()
#         else:
#             # Wait for a short timeout to allow for more prompts.
#             await asyncio.sleep(self.batch_timeout)
#             # Try draining the batch if it hasn't been processed yet.
#             async with self._batch_lock:
#                 if self._batch:
#                     await self._drain_batch()
#         # Return this prompt's response (its future should now have been resolved)
#         return await future

# # Example usage:
# # async def test():
# #     processor = LLMBatchProcessor(model="your-model")
# #     responses = await asyncio.gather(
# #         processor.process("Hello"),
# #         processor.process("How's the weather?"),
# #         processor.process("Tell me a joke."),
# #         processor.process("What's the news?")
# #     )
# #     print(responses)
# #
# # asyncio.run(test())