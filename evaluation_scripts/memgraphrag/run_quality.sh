#!/usr/bin/env bash
# MemGraphRAG on QuALITY (evaluation_scripts/QUALITY_EVAL_PLAN.md). Upstream input format: one .txt, which index.py
# cuts into 256/32-token windows itself. memgraphrag_corpus.txt = the 115 dev articles ("title\ntext") joined by a
# blank line; at most one window per article boundary (114 of ~3,000) holds the end of one article and the start of
# the next (no question spans articles). All index / retrieval flags are the same as run.sh. Its own QA is kept for
# reference; the reported answer comes from the shared multiple-choice step over the same top-5 windows.
# Usage: bash evaluation_scripts/memgraphrag/run_quality.sh
#   QUALITY_INPUTS  inputs dir (default evaluation_data/quality/inputs; build_quality_inputs.py)
#   QUALITY_TAG     output name (default quality -> outputs/quality)
#   CONCURRENCY_MULTIPLIER  scales memory workers (base 20, as run.sh), QA workers (base 5, upstream default) and
#                           answer/score workers (base 16). Its OpenIE pool is Python's default (32 here) and not configurable.
set -euo pipefail
source "$(dirname "$0")/../common/env.sh"
cd "$EVAL_ROOT/memgraphrag"
export OPENAI_API_KEY="$DEEPSEEK_API_KEY"
export QUALITY_INPUTS="${QUALITY_INPUTS:-$REPO_ROOT/evaluation_data/quality/inputs}"
TAG="${QUALITY_TAG:-quality}"
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
"$PYTHON" "$EVAL_ROOT/common/answer.py" --dataset quality --system memgraphrag --retrieval "$SAVE_DIR/qa_results.json" --qa-top-k 5 --workers "$(scale_concurrency 16)"
"$PYTHON" "$EVAL_ROOT/common/score_quality.py" --system memgraphrag --predictions "$SAVE_DIR/answers_memgraphrag_quality.json" --workers "$(scale_concurrency 16)"
