#!/bin/bash
# start_vllm.sh

# Model name or path (HuggingFace model or local folder)
MODEL="MasterControlAIML/DeepSeek-R1-Qwen2.5-1.5b-SFT-R1-JSON-Unstructured-To-Structured"

# Start vLLM using OpenAI-compatible API
vllm serve \
  "$MODEL" \
  --task generate \
  --trust-remote-code \
  --gpu-memory-utilization 0.8 \
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