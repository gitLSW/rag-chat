import asyncio
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
LLM_URL = os.getenv('LLM_URL')
LLM_MODEL = os.getenv('LLM_MODEL')

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
    def __init__(self, vllm_url=LLM_URL):
        """
        Initializes the vLLM LLM instance.
        """
        self.model = LLM_MODEL
        self.client = AsyncOpenAI(  # Use async client
            base_url=vllm_url.rstrip('/') + "/v1",
            api_key="token-abc123"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model)


    async def query(self, prompt, sampling_params=None, stream=False):
        if sampling_params is None:
            sampling_params = SamplingParams(
                temperature=0.3,
                top_p=0.6,
                max_tokens=2048,
                # stop=["\n\n", "\n", "Q:", "###"]
            )
            
        res = await self.client.completions.create(
            model=self.model,
            prompt=prompt,
            temperature=sampling_params.temperature,
            top_p=sampling_params.top_p,
            max_tokens=sampling_params.max_tokens,
            stop=sampling_params.stop,
            stream=stream  # Enable streaming
        )
        
        # Return the completed text if streaming is disabled
        if not stream:
            return res.choices[0].text

        # Yield text chunks incrementally
        async for chunk in res:
            yield chunk.choices[0].text
        




from doc_extractor import DocExtractor
doc_extractor = DocExtractor()

# List of sample prompts.
# batch_prompts = [
#     "Who was Caesar?",
#     "Who was Alexander the Great?",
#     "Which lands did Charlemagne own?",
#     "Who defeated King Sigismund ?",
# ]

# Initialize our batch query object.
llm_service = LLMService()

async def test1(path):
    text = doc_extractor.extract_text(path)
    prompt = f"""Welche Leistungen wurden in der folgenden Rechnung abgerechnet: {text}"""
    results = await llm_service.query(prompt)
    print(f"Generated text: {results}\n")


asyncio.run(test1('/home/lsw/Desktop/invoice.pdf'))

# async def test2(path, json_schema):
#     print('JSON SCHEMA:', json_schema)
#     text = doc_extractor.extract_text(path)
#     result = await llm_service.extract_json(text, json_schema)
#     print(result)

# json_schema = json.dumps({
# "invoice_id": "str",
# "invoice_date": "DD.MM.YYYY",
# "customer_name": "str",
# "customer_address": "str",
# "vendor_name": "str",
# "vendor_address": "str",
# "vendor_email": "str",
# "vendor_phone_num": "str",
# "total_cost": "float",
# "vat": "float"
# })

# asyncio.run(test2('/home/lsw/Desktop/invoice.pdf', json_schema))