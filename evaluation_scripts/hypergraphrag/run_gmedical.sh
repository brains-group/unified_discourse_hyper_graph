#!/usr/bin/env bash
# HyperGraphRAG on G-Medical (GraphRAG-Bench Medical as MemGraphRAG evaluates it; evaluation_scripts/GMEDICAL_EVAL_PLAN.md).
# Index + retrieval = this system's QuALITY code path (run_quality.sh), unchanged, pointed at the G-Medical inputs
# (evaluation_data/build_gmedical_inputs.py writes the same layout; medical.txt as one document, 2,062 questions), as the
# contracts harness does. Then the shared open-ended answer step (common/answer.py --dataset gmedical: HippoRAG 2 /
# MemGraphRAG one-shot CoT prompt) and common/score_gmedical.py (LLM-Acc, overall and per GraphRAG-Bench level).
# Usage: bash evaluation_scripts/hypergraphrag/run_gmedical.sh
#   GMEDICAL_INPUTS  inputs dir (default evaluation_data/gmedical/inputs; smoke: evaluation_data/gmedical/smoke/inputs)
#   GMEDICAL_TAG     output name (default gmedical); optional first argument = question limit (smoke)
#   CONCURRENCY_MULTIPLIER  as in run_quality.sh
set -euo pipefail
LIMIT="${1:-0}"
source "$(dirname "$0")/../common/env.sh"
cd "$EVAL_ROOT/hypergraphrag"
export GMEDICAL_INPUTS="${GMEDICAL_INPUTS:-$REPO_ROOT/evaluation_data/gmedical/inputs}"
export QUALITY_INPUTS="$GMEDICAL_INPUTS"      # the QuALITY code path reads this inputs dir
TAG="${GMEDICAL_TAG:-gmedical}"
PYTHON="${PYTHON:-python}"
OUT="$EVAL_ROOT/hypergraphrag/outputs/$TAG"
"$PYTHON" main.py --dataset quality --quality-inputs "$QUALITY_INPUTS" --out-dir "$OUT" --step all --limit "$LIMIT" \
  --index-llm-max-async "$(scale_concurrency 32)" --index-embedding-max-async "$(scale_concurrency 32)" \
  --retrieve-llm-max-async "$(scale_concurrency 16)" --retrieve-embedding-max-async "$(scale_concurrency 16)" \
  --retrieve-concurrency "$(scale_concurrency 32)"
"$PYTHON" "$EVAL_ROOT/common/answer.py" --dataset gmedical --system hypergraphrag --retrieval "$OUT/knowledge.json" --workers "$(scale_concurrency 16)"
"$PYTHON" "$EVAL_ROOT/common/score_gmedical.py" --system hypergraphrag --predictions "$OUT/answers_hypergraphrag_gmedical.json" --workers "$(scale_concurrency 16)"
