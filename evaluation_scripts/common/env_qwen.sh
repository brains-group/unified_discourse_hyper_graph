# Local Qwen for the insurance-contracts benchmark (CONTRACTS_EVAL_PLAN.md). Source this INSTEAD of env.sh:
#   source evaluation_scripts/common/env_qwen.sh
# It sets the LLM endpoint, then sources env.sh (whose ${VAR:-default} keep these values). Override by exporting first.
#   QWEN_BASE_URL  default http://localhost:8000/v1 (vLLM serving Qwen/Qwen3-30B-A3B-Instruct-2507-FP8)
export LLM_BASE_URL="${QWEN_BASE_URL:-http://localhost:8000/v1}"
export LLM_MODEL="${QWEN_MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507-FP8}"
# vLLM needs no key; every runner reads DEEPSEEK_API_KEY / OPENAI_API_KEY / LLM_API_KEY, so give them a dummy value
export DEEPSEEK_API_KEY="EMPTY" LLM_API_KEY="EMPTY"
# Qwen3-Instruct-2507 has no thinking mode; don't send DeepSeek's `thinking` field
export LLM_DISABLE_THINKING=0
# no temperature sent (vLLM then applies the model's generation_config: 0.7 / top-p 0.8 / top-k 20)
export LLM_TEMPERATURE="${LLM_TEMPERATURE-}"
# the unh_eval env imports every system except MS GraphRAG / HyGRAG (own envs, set in their run_contracts.sh)
export PYTHON="${PYTHON:-$HOME/miniconda3/envs/unh_eval/bin/python}"
# models load from the local Hugging Face cache (no Hub requests, no 429s)
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
# per-call client timeout where a runner makes it configurable (HyperGraphRAG): long extractions on a busy local server
export LLM_TIMEOUT="${LLM_TIMEOUT:-1800}"
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
# one local server: default per-system concurrency is set in each run_contracts.sh; CONTRACT_JOBS = contracts
# built / retrieved in parallel per system
export CONTRACT_JOBS="${CONTRACT_JOBS:-5}"
export CONTRACT_SCORE_WORKERS="${CONTRACT_SCORE_WORKERS:-32}"
