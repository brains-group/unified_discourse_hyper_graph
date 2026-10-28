#!/usr/bin/env bash
# LogicRAG on QuALITY (evaluation_scripts/QUALITY_EVAL_PLAN.md). LogicRAG has no chunker and formats every corpus
# entry as "Title: {title}. Content: {text}", so it gets chunks_logicrag.json: each dev article cut into
# 1200/100-token chunks by HyGRAG's chunker, title = article title on the first chunk only (option B).
# Questions carry their four options. Same settings as run.sh (3 rounds, top-5). Its own answer is kept for
# reference; the reported answer comes from the shared multiple-choice step over the same info_summary.
# Usage: bash evaluation_scripts/logicrag/run_quality.sh [limit, 0 = all]
#   QUALITY_INPUTS  inputs dir (default evaluation_data/quality/inputs; build_quality_inputs.py)
#   QUALITY_TAG     output name (default quality -> result/quality/, cache/quality/)
#   CONCURRENCY_MULTIPLIER  scales answer/score workers (base 16); LogicRAG itself answers one question at a time
set -euo pipefail
LIMIT="${1:-0}"
source "$(dirname "$0")/../common/env.sh"
cd "$EVAL_ROOT/logicrag"
export OPENAI_API_KEY="$DEEPSEEK_API_KEY" OPENAI_BASE_URL="$LLM_BASE_URL"
export LOGICRAG_LLM_MODEL="$LLM_MODEL" LOGICRAG_EMBEDDING_MODEL="$EMBEDDING_MODEL"
export LOGICRAG_CALLS_PER_MINUTE="${LOGICRAG_CALLS_PER_MINUTE:-600}"   # upstream default 20/min is an OpenAI-tier limit
export QUALITY_INPUTS="${QUALITY_INPUTS:-$REPO_ROOT/evaluation_data/quality/inputs}"
TAG="${QUALITY_TAG:-quality}"
# own result / embedding-cache dirs: LogicRAG keys its embedding cache only on the corpus size
export LOGICRAG_RESULT_DIR="$EVAL_ROOT/logicrag/result/$TAG" LOGICRAG_CACHE_DIR="$EVAL_ROOT/logicrag/cache/$TAG"
PYTHON="${PYTHON:-python}"
OUT="$LOGICRAG_RESULT_DIR/logicrag_quality.json"
"$PYTHON" run.py --dataset "$QUALITY_INPUTS/questions.json" --corpus "$QUALITY_INPUTS/chunks_logicrag.json" \
  --limit "$LIMIT" --max-rounds 3 --top-k 5 --output logicrag_quality.json
"$PYTHON" "$EVAL_ROOT/common/answer.py" --dataset quality --system logicrag --retrieval "$OUT" --workers "$(scale_concurrency 16)"
"$PYTHON" "$EVAL_ROOT/common/score_quality.py" --system logicrag --predictions "$LOGICRAG_RESULT_DIR/answers_logicrag_quality.json" --workers "$(scale_concurrency 16)"
