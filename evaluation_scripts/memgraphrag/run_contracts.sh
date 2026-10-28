#!/usr/bin/env bash
# MemGraphRAG on the insurance contracts (CONTRACTS_EVAL_PLAN.md): index.py over the contract .txt (it cuts its own
# 256/32-token windows), retrieval_dataset_test.py for every query (QA questions, then facts verbatim; its own QA answer
# is kept for reference), then the shared answer step + judges over the same top-5 windows (common/contracts_score.py).
# All index / retrieval flags are the ones run.sh / run_quality.sh use. DeepSeek by default (env_contracts.sh; CONTRACTS_LLM=qwen for local Qwen).
# Usage: bash evaluation_scripts/memgraphrag/run_contracts.sh <short|long|smoke>     (CONTRACTS="a b" for some only)
set -euo pipefail
SET="${1:?usage: run_contracts.sh <short|long|smoke>}"
source "$(dirname "$0")/../common/env_contracts.sh"
source "$EVAL_ROOT/common/contracts_lib.sh"
cd "$EVAL_ROOT/memgraphrag"
export OPENAI_API_KEY="$DEEPSEEK_API_KEY"
EMB="VLLM/$EMBEDDING_MODEL"          # routes to src/embedding_model/VLLM.py
run_one() {
  local IN="$CONTRACTS_DATA/$SET/$1" SAVE_DIR="outputs/contracts_$SET/$1"
  "$PYTHON" index.py --corpus "$IN/memgraphrag_corpus.txt" --save-dir "$SAVE_DIR" \
    --llm-name "$LLM_MODEL" --llm-base-url "$LLM_BASE_URL" \
    --embedding-model "$EMB" --embedding-base-url "$EMBEDDING_BASE_URL/embeddings" --tokenizer "$EMBEDDING_MODEL" \
    --chunk-size 256 --chunk-overlap 32 --artifact-mode default --memory-max-workers 20 \
    --ontology-filter-mode percentile --ontology-low-percent 20 --conflict-min-confidence 0.85 \
    --synonymy-edge-topk 100 --synonymy-edge-threshold 0.8
  "$PYTHON" retrieval_dataset_test.py --questions "$IN/questions.json" \
    --save-dir "$SAVE_DIR" --output "$SAVE_DIR/qa_results.json" \
    --llm-name "$LLM_MODEL" --llm-base-url "$LLM_BASE_URL" \
    --embedding-model "$EMB" --embedding-base-url "$EMBEDDING_BASE_URL/embeddings" \
    --question-type all --sample-num 0 --qa-top-k 5 --qa-max-workers 5 \
    --skip-fact-rerank true --fact-similarity-threshold 0.6 --use-raw-threshold-filter true
}
for_each_contract memgraphrag "$SET" run_one
score_contracts memgraphrag "$SET"
