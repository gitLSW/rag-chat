#!/bin/bash
# start_vllm.sh

# Model name or path (HuggingFace model or local folder)
MODEL="RedHatAI/DeepSeek-R1-Distill-Qwen-7B-quantized.w8a8" # "RedHatAI/DeepSeek-R1-Distill-Qwen-32B-quantized.w8a8"
# MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
# MODEL="MasterControlAIML/DeepSeek-R1-Qwen2.5-1.5b-SFT-R1-JSON-Unstructured-To-Structured"

# Start vLLM using OpenAI-compatible API
vllm serve \
  "$MODEL" \
  --task generate \
  --trust-remote-code \
  --gpu-memory-utilization 0.95 \
  --tensor-parallel-size 1 \
  --dtype auto \
  --api-key token-abc123 \
  --port 6096


# python3 -m vllm.entrypoints.openai.api_server \
#     --model $MODEL \
#     --task generate \
#     --trust-remote-code \
#     --gpu-memory-utilization 0.8 \
#     --tensor-parallel-size 1 \
#     --port 6096