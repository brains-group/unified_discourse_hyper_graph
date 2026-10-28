#!/usr/bin/env bash
# MemGraphRAG on G-Medical (GraphRAG-Bench Medical as MemGraphRAG evaluates it; evaluation_scripts/GMEDICAL_EVAL_PLAN.md).
# Index + retrieval = this system's QuALITY code path (run_quality.sh), unchanged, pointed at the G-Medical inputs
# (evaluation_data/build_gmedical_inputs.py writes the same layout; medical.txt as one document, 2,062 questions), as the
# contracts harness does. Then the shared open-ended answer step (common/answer.py --dataset gmedical: HippoRAG 2 /
# MemGraphRAG one-shot CoT prompt) and common/score_gmedical.py (LLM-Acc, overall and per GraphRAG-Bench level).
# Usage: bash evaluation_scripts/memgraphrag/run_gmedical.sh
#   GMEDICAL_INPUTS  inputs dir (default evaluation_data/gmedical/inputs; smoke: evaluation_data/gmedical/smoke/inputs)
#   GMEDICAL_TAG     output name (default gmedical)
#   CONCURRENCY_MULTIPLIER  as in run_quality.sh
set -euo pipefail
source "$(dirname "$0")/../common/env.sh"
cd "$EVAL_ROOT/memgraphrag"
export OPENAI_API_KEY="$DEEPSEEK_API_KEY"
export GMEDICAL_INPUTS="${GMEDICAL_INPUTS:-$REPO_ROOT/evaluation_data/gmedical/inputs}"
export QUALITY_INPUTS="$GMEDICAL_INPUTS"      # the QuALITY code path reads this inputs dir
TAG="${GMEDICAL_TAG:-gmedical}"
PYTHON="${PYTHON:-python}"
SAVE_DIR="outputs/$TAG"
EMB="VLLM/$EMBEDDING_MODEL"          # routes to src/embedding_model/VLLM.py
"$PYTHON" index.py --corpus "$QUALITY_INPUTS/memgraphrag_corpus.txt" --save-dir "$SAVE_DIR" \
  --llm-name "$LLM_MODEL" --llm-base-url "$LLM_BASE_URL" \
  --embedding-model "$EMB" --embedding-base-url "$EMBEDDING_BASE_URL/embeddings" --tokenizer "$EMBEDDING_MODEL" \
  --chunk-size 256 --chunk-overlap 32 --artifact-mode default --memory-max-workers "$(scale_concurrency 20)" \
  --ontology-filter-mode percentile --ontology-low-percent 20 --conflict-min-confidence 0.85 \
  --synonymy-edge-topk 100 --synonymy-edge-threshold 0.8
"$PYTHON" retrieval_dataset_test.py --questions "$QUALITY_INPUTS/questions.json" \
  --save-dir "$SAVE_DIR" --output "$SAVE_DIR/qa_results.json" \
  --llm-name "$LLM_MODEL" --llm-base-url "$LLM_BASE_URL" \
  --embedding-model "$EMB" --embedding-base-url "$EMBEDDING_BASE_URL/embeddings" \
  --question-type all --sample-num 0 --qa-top-k 5 --qa-max-workers "$(scale_concurrency 5)" \
  --skip-fact-rerank true --fact-similarity-threshold 0.6 --use-raw-threshold-filter true
"$PYTHON" "$EVAL_ROOT/common/answer.py" --dataset gmedical --system memgraphrag --retrieval "$SAVE_DIR/qa_results.json" --qa-top-k 5 --workers "$(scale_concurrency 16)"
"$PYTHON" "$EVAL_ROOT/common/score_gmedical.py" --system memgraphrag --predictions "$SAVE_DIR/answers_memgraphrag_gmedical.json" --workers "$(scale_concurrency 16)"
