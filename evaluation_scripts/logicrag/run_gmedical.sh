#!/usr/bin/env bash
# LogicRAG on G-Medical (GraphRAG-Bench Medical as MemGraphRAG evaluates it; evaluation_scripts/GMEDICAL_EVAL_PLAN.md).
# Index + retrieval = this system's QuALITY code path (run_quality.sh), unchanged, pointed at the G-Medical inputs
# (evaluation_data/build_gmedical_inputs.py writes the same layout; medical.txt as one document, 2,062 questions), as the
# contracts harness does. Then the shared open-ended answer step (common/answer.py --dataset gmedical: HippoRAG 2 /
# MemGraphRAG one-shot CoT prompt) and common/score_gmedical.py (LLM-Acc, overall and per GraphRAG-Bench level).
# Usage: bash evaluation_scripts/logicrag/run_gmedical.sh
#   GMEDICAL_INPUTS  inputs dir (default evaluation_data/gmedical/inputs; smoke: evaluation_data/gmedical/smoke/inputs)
#   GMEDICAL_TAG     output name (default gmedical); optional first argument = question limit (smoke)
#   CONCURRENCY_MULTIPLIER  as in run_quality.sh
set -euo pipefail
LIMIT="${1:-0}"
source "$(dirname "$0")/../common/env.sh"
cd "$EVAL_ROOT/logicrag"
export OPENAI_API_KEY="$DEEPSEEK_API_KEY" OPENAI_BASE_URL="$LLM_BASE_URL"
export LOGICRAG_LLM_MODEL="$LLM_MODEL" LOGICRAG_EMBEDDING_MODEL="$EMBEDDING_MODEL"
export LOGICRAG_CALLS_PER_MINUTE="${LOGICRAG_CALLS_PER_MINUTE:-600}"   # upstream default 20/min is an OpenAI-tier limit
export GMEDICAL_INPUTS="${GMEDICAL_INPUTS:-$REPO_ROOT/evaluation_data/gmedical/inputs}"
export QUALITY_INPUTS="$GMEDICAL_INPUTS"      # the QuALITY code path reads this inputs dir
TAG="${GMEDICAL_TAG:-gmedical}"
# own result / embedding-cache dirs: LogicRAG keys its embedding cache only on the corpus size
export LOGICRAG_RESULT_DIR="$EVAL_ROOT/logicrag/result/$TAG" LOGICRAG_CACHE_DIR="$EVAL_ROOT/logicrag/cache/$TAG"
PYTHON="${PYTHON:-python}"
OUT="$LOGICRAG_RESULT_DIR/logicrag_quality.json"
"$PYTHON" run.py --dataset "$QUALITY_INPUTS/questions.json" --corpus "$QUALITY_INPUTS/chunks_logicrag.json" \
  --limit "$LIMIT" --max-rounds 3 --top-k 5 --output logicrag_quality.json
"$PYTHON" "$EVAL_ROOT/common/answer.py" --dataset gmedical --system logicrag --retrieval "$OUT" --workers "$(scale_concurrency 16)"
"$PYTHON" "$EVAL_ROOT/common/score_gmedical.py" --system logicrag --predictions "$LOGICRAG_RESULT_DIR/answers_logicrag_gmedical.json" --workers "$(scale_concurrency 16)"
