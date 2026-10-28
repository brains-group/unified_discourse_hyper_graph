#!/usr/bin/env bash
# HyperGraphRAG (upstream, unmodified) on the insurance contracts (CONTRACTS_EVAL_PLAN.md): rag.insert(the whole
# contract as one document; HyperGraphRAG chunks it 1200/100 itself) -> hybrid-mode context for every query (QA
# questions, then facts verbatim) -> shared answer step + judges (common/contracts_score.py). DeepSeek by default (env_contracts.sh; CONTRACTS_LLM=qwen for local Qwen).
# The contract inputs dir has the QuALITY inputs layout, so main.py's inputs-dir loader (--dataset quality) reads it.
# Usage: bash evaluation_scripts/hypergraphrag/run_contracts.sh <short|long|smoke>     (CONTRACTS="a b" for some only)
set -euo pipefail
SET="${1:?usage: run_contracts.sh <short|long|smoke>}"
source "$(dirname "$0")/../common/env_contracts.sh"
source "$EVAL_ROOT/common/contracts_lib.sh"
cd "$EVAL_ROOT/hypergraphrag"
run_one() {  # upstream max_async values (index 32/32, retrieval 16/16, 32 queries in flight)
  "$PYTHON" main.py --dataset quality --quality-inputs "$CONTRACTS_DATA/$SET/$1" \
    --out-dir "$EVAL_ROOT/hypergraphrag/outputs/contracts_$SET/$1" --step all \
    --index-llm-max-async 32 --index-embedding-max-async 32 \
    --retrieve-llm-max-async 16 --retrieve-embedding-max-async 16 --retrieve-concurrency 32
}
for_each_contract hypergraphrag "$SET" run_one
score_contracts hypergraphrag "$SET"
