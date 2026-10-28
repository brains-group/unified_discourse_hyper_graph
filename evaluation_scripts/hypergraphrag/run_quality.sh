#!/usr/bin/env bash
# HyperGraphRAG (upstream evaluation/hypergraphrag, unmodified) on QuALITY (evaluation_scripts/QUALITY_EVAL_PLAN.md):
# rag.insert(one "title\ntext" string per dev article; HyperGraphRAG chunks each 1200/100 itself) -> hybrid-mode
# context for every question (with its options) -> shared multiple-choice answer step -> score_quality.py.
# Every LLM call is cached in outputs/<tag>/llm_cache.sqlite.
# Usage: bash evaluation_scripts/hypergraphrag/run_quality.sh [limit, 0 = all questions]
#   QUALITY_INPUTS  inputs dir (default evaluation_data/quality/inputs; build_quality_inputs.py)
#   QUALITY_TAG     output name (default quality -> outputs/quality)
#   CONCURRENCY_MULTIPLIER  scales index max_async (base 32/32), retrieval max_async (16/16) and questions in flight (32),
#                           all upstream values, and answer/score workers (base 16)
set -euo pipefail
LIMIT="${1:-0}"
source "$(dirname "$0")/../common/env.sh"
cd "$EVAL_ROOT/hypergraphrag"
export QUALITY_INPUTS="${QUALITY_INPUTS:-$REPO_ROOT/evaluation_data/quality/inputs}"
TAG="${QUALITY_TAG:-quality}"
PYTHON="${PYTHON:-python}"
OUT="$EVAL_ROOT/hypergraphrag/outputs/$TAG"
"$PYTHON" main.py --dataset quality --quality-inputs "$QUALITY_INPUTS" --out-dir "$OUT" --step all --limit "$LIMIT" \
  --index-llm-max-async "$(scale_concurrency 32)" --index-embedding-max-async "$(scale_concurrency 32)" \
  --retrieve-llm-max-async "$(scale_concurrency 16)" --retrieve-embedding-max-async "$(scale_concurrency 16)" \
  --retrieve-concurrency "$(scale_concurrency 32)"
"$PYTHON" "$EVAL_ROOT/common/answer.py" --dataset quality --system hypergraphrag --retrieval "$OUT/knowledge.json" --workers "$(scale_concurrency 16)"
"$PYTHON" "$EVAL_ROOT/common/score_quality.py" --system hypergraphrag --predictions "$OUT/answers_hypergraphrag_quality.json" --workers "$(scale_concurrency 16)"
