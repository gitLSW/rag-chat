import re
import json
import asyncio
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from openai import AsyncOpenAI

LLM_URL = 'http://0.0.0.0:6096'
# LLM_MODEL = 'MasterControlAIML/DeepSeek-R1-Qwen2.5-1.5b-SFT-R1-JSON-Unstructured-To-Structured' # 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'
LLM_MODEL = "RedHatAI/DeepSeek-R1-Distill-Qwen-7B-quantized.w8a8" # 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B'

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


    async def query(self, prompt, sampling_params=None):
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
            stop=sampling_params.stop
        )
        return res.choices[0].text
    

    async def extract_json(self, text, json_schema):
        """
        Extracts a filled JSON object from LLM output based on a schema and checks if all fields are filled.

        Args:
            text (str): Input text to extract data from.
            json_schema (dict): The target JSON schema structure.

        Returns:
            tuple:
                - dict: The parsed JSON object.
                - bool: Whether all schema fields were filled.
        """
        sampling_params = SamplingParams(
            temperature=0.1,
            top_p=0.4,
            max_tokens=4096
            # stop=["\n\n", "\n", "Q:", "###"]
        )

        # prompt = f"""
        #     ### Role:
        #     You are an expert data extractor specializing in mapping hierarchical text data into a given JSON Schema.

        #     ### DATA INPUT:
        #     - **Text:** ```{text}```
        #     - **Blank JSON Schema:** ```{json.dumps(json_schema)}```

        #     ### TASK REQUIREMENT:
        #     1. Analyze the given text and map all relevant information strictly into the provided JSON Schema.
        #     2. Provide your output in **two mandatory sections**:
        #     - **`<answer>`:** The filled JSON object  
        #     - **`<think>`:** Reasoning for the mapping decisions  

        #     ### OUTPUT STRUCTURE:

        #     `<think> /* Explanation of mapping logic */ </think>`
        #     `<answer> /* Completed JSON Object */ </answer>`

        #     ### STRICT RULES FOR GENERATING OUTPUT:
        #     1. **Both Tags Required:**  
        #     - Always provide both the `<think>` and the `<answer>` sections.  
        #     - If reasoning is minimal, state: "Direct mapping from text to schema."
        #     2. **JSON Schema Mapping:**  
        #     - Strictly map the text data to the given JSON Schema without modification or omissions.
        #     3. **Hierarchy Preservation:**  
        #     - Maintain proper parent-child relationships and follow the schema's hierarchical structure.
        #     4. **Correct Mapping of Attributes:**  
        #     - Map key attributes, including `id`, `idc`, `idx`, `level_type`, and `component_type`.
        #     5. **JSON Format Compliance:**  
        #     - In your answer follow the JSON Format strictly ! This means NO comments and NO '...'
        #     - STRICTLY FOLLOW THE PROMPT AND TEXT !!! DO NOT IMAGINE ANYTHING !!!
        #     - Escape quotes (`\n`), replace newlines with `\\n`, avoid trailing commas, and use double quotes exclusively.
        #     6. **Step-by-Step Reasoning:**  
        #     - Explain your reasoning within the `<think>` tag.

        #     ### IMPORTANT:
        #     If either the `<think>` or `<answer>` tags is missing, the response will be considered incomplete."""
        
        
        prompt = f"""This is a JSON Schema which you need to fill:

            {json.dumps(json_schema)}

            ### TASK REQUIREMENT
            You are a json extractor. You are tasked with extracting the relevant information needed to fill the JSON schema from the text below.
            
            {text}

            ### STRICT RULES FOR GENERATING OUTPUT:
            **ALWAYS PROVIDE YOUR FINAL ANSWER**:
            - Always provide the filled JSON
            **JSON Schema Mapping:**:
            - Carefully check to find ALL relevant information in the text like ids, names, addresses, etc.
            - Sometimes ids my be called numbers
            - Strictly map the data you found to the given JSON Schema without modification or omissions.
            **Hierarchy Preservation:**:
            - Maintain proper parent-child relationships and follow the schema's hierarchical structure.
            **Correct Mapping of Attributes:**:
            - Map all the relevant information you find to its appropriate keys
            **JSON Format Compliance:**:
            - In your answer follow the JSON Format strictly !
            - If your answer doesn't conform to the JSON Format or is incompatible with the provided JSON schema, the output will be disgarded !
            
            Write your reasoning below here inside think tags and once you are done thinking, provide your answer in the described format !"""

        res = await self.query(prompt, sampling_params)

        try:
            answer_json = re.search(r"```json\s*(.*?)\s*```", res, re.DOTALL).group(1)
            parsed_json = json.loads(answer_json)
        except (IndexError, json.JSONDecodeError) as e:
            raise ValueError(f"Failed to extract or parse JSON from model response: {e}")

        # Helper function to check if all schema keys are present and filled
        def all_fields_filled(schema_part, json_part):
            if isinstance(schema_part, dict):
                for key, value in schema_part.items():
                    if key not in json_part or json_part[key] in [None, "", []]:
                        return False
                    if isinstance(value, dict) and isinstance(json_part[key], dict):
                        if not all_fields_filled(value, json_part[key]):
                            return False
            return True

        is_complete = all_fields_filled(json_schema, parsed_json)

        return parsed_json, is_complete




# List of sample prompts.
# batch_prompts = [
#     "Who was Caesar?",
#     "Who was Alexander the Great?",
#     "Which lands did Charlemagne own?",
#     "Who defeated King Sigismund ?",
# ]

# # Initialize our batch query object.
# llm_service = LLMService()

# async def test1(prompt):
#     results = await llm_service.query(batch_prompts[0])
#     print(f"Prompt: {prompt}\nGenerated text: {results}\n")

# # Query the model with the batch of prompts.
# # for prompt in batch_prompts:
# #     asyncio.run(test1(prompt))

# from doc_extractor import DocExtractor
# doc_extractor = DocExtractor()
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