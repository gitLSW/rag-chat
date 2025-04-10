# from text_generation import AsyncClient

# class LLMService:
#     def __init__(self, url='http://localhost:8080'):
#         """
#         Initializes the LLM service with the specified inference URL.

#         :param url: The URL of the hosted LLM, e.g., 'http://localhost:8080'
#         """
#         self.client = AsyncClient(base_url=url)

#     async def query_llm(self, question):
#         """
#         Sends a question to the LLM and retrieves the generated response.

#         :param question: The input prompt/question string.
#         :return: The generated response from the LLM.
#         """
#         response = await self.client.generate(question)
#         return response.generated_text



# import asyncio

# llm_service = LLMService()

# questions = [
#     "Who was Caesar?",
#     "Who was ALexander the Great?",
#     "Which lands did Charlemagne own?",
#     "Who defeated King Sigismund ?",
# ]

# async def main():
#     tasks = [llm_service.query_llm(question) for question in questions]

#     # Run all queries concurrently and wait for their results
#     responses = await asyncio.gather(*tasks)

#     for question, response in zip(questions, responses):
#         print(f"Question: {question}")
#         print(f"Answer: {response}\n")

# asyncio.run(main())