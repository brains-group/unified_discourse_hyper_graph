#!/usr/bin/env bash
# HippoRAG 2 on the insurance contracts (CONTRACTS_EVAL_PLAN.md): one index per contract over the 1200/100-token chunks
# HyGRAG's chunker cuts from it (HippoRAG 2 has no chunker), top-5 passages retrieved for every query (QA questions,
# then facts verbatim), then the shared answer step + judges (common/contracts_score.py). DeepSeek by default (env_contracts.sh; CONTRACTS_LLM=qwen for local Qwen).
# The contract inputs dir has the QuALITY inputs layout, so main.py's inputs-dir loader (--dataset quality) reads it.
# Usage: bash evaluation_scripts/hipporag2/run_contracts.sh <short|long|smoke>     (CONTRACTS="a b" for some only)
#   HIPPORAG_OPENIE_WORKERS  OpenIE workers per contract (default 8, upstream default)
set -euo pipefail
SET="${1:?usage: run_contracts.sh <short|long|smoke>}"
source "$(dirname "$0")/../common/env_contracts.sh"
source "$EVAL_ROOT/common/contracts_lib.sh"
cd "$EVAL_ROOT/hipporag2"
export OPENAI_API_KEY="$DEEPSEEK_API_KEY"
run_one() {
  "$PYTHON" main.py --dataset quality --quality_inputs "$CONTRACTS_DATA/$SET/$1" --save_dir "outputs/contracts_$SET/$1" \
    --llm_base_url "$LLM_BASE_URL" --llm_name "$LLM_MODEL" \
    --embedding_provider vllm --embedding_name "$EMBEDDING_MODEL" --embedding_base_url "$EMBEDDING_BASE_URL/embeddings" \
    --openie_max_workers "${HIPPORAG_OPENIE_WORKERS:-8}" --force_index_from_scratch "${HIPPORAG_FORCE_INDEX:-false}"
}
for_each_contract hipporag2 "$SET" run_one
score_contracts hipporag2 "$SET"
