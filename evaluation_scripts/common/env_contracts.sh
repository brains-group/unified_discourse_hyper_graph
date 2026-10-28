# Model settings for the insurance-contracts runners (CONTRACTS_EVAL_PLAN.md). Every <system>/run_contracts.sh and
# launch_contracts.sh sources this instead of env.sh:   source evaluation_scripts/common/env_contracts.sh
#   CONTRACTS_LLM  deepseek (default, 2026-10-11: the same DeepSeek setup as HotpotQA / MuSiQue / QuALITY / G-Medical,
#                  via env.sh) | qwen (local Qwen3-30B on :8000, common/env_qwen.sh, the original 2026-10-09 setup)
if [ "${CONTRACTS_LLM:-deepseek}" = "qwen" ]; then
  source "$(dirname "${BASH_SOURCE[0]}")/env_qwen.sh"
else
  _repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
  if [ -z "${DEEPSEEK_API_KEY:-}" ] && [ -f "$_repo/.env" ]; then
    DEEPSEEK_API_KEY="$(grep -E '^\s*DEEPSEEK_API_KEY\s*=' "$_repo/.env" | head -1 | cut -d= -f2- | tr -d "\"' ")"
    export DEEPSEEK_API_KEY
  fi
  export LLM_API_KEY="${LLM_API_KEY:-$DEEPSEEK_API_KEY}"
  # the unh_eval env imports every system except MS GraphRAG / HyGRAG (own envs, set in their run_contracts.sh)
  export PYTHON="${PYTHON:-$HOME/miniconda3/envs/unh_eval/bin/python}"
  # models load from the local Hugging Face cache (no Hub requests, no 429s)
  export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
  # env.sh: deepseek-flash at api.deepseek.com, thinking disabled, no temperature sent, voyage-4-nano on :8001
  source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
  # per-system LLM concurrency defaults are in each run_contracts.sh (64-ish per contract); CONTRACT_JOBS contracts
  # run at once per system, so the peak over all six baselines stays far below DeepSeek's 2,500 concurrent requests
  export CONTRACT_JOBS="${CONTRACT_JOBS:-5}"
  export CONTRACT_SCORE_WORKERS="${CONTRACT_SCORE_WORKERS:-32}"
fi
