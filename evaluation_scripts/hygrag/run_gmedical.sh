#!/usr/bin/env bash
# HyGRAG on G-Medical (GraphRAG-Bench Medical as MemGraphRAG evaluates it; evaluation_scripts/GMEDICAL_EVAL_PLAN.md).
# Index + retrieval = this system's QuALITY code path (run_quality.sh), unchanged, pointed at the G-Medical inputs
# (evaluation_data/build_gmedical_inputs.py writes the same layout; medical.txt as one document, 2,062 questions), as the
# contracts harness does. Then the shared open-ended answer step (common/answer.py --dataset gmedical: HippoRAG 2 /
# MemGraphRAG one-shot CoT prompt) and common/score_gmedical.py (LLM-Acc, overall and per GraphRAG-Bench level).
# Usage: bash evaluation_scripts/hygrag/run_gmedical.sh
#   GMEDICAL_INPUTS  inputs dir (default evaluation_data/gmedical/inputs; smoke: evaluation_data/gmedical/smoke/inputs)
#   GMEDICAL_TAG     output name (default gmedical)
#   CONCURRENCY_MULTIPLIER  as in run_quality.sh
set -euo pipefail
source "$(dirname "$0")/../common/env.sh"
HERE="$EVAL_ROOT/hygrag"
cd "$HERE"
export GMEDICAL_INPUTS="${GMEDICAL_INPUTS:-$REPO_ROOT/evaluation_data/gmedical/inputs}"
export QUALITY_INPUTS="$GMEDICAL_INPUTS"      # the QuALITY code path reads this inputs dir
TAG="${GMEDICAL_TAG:-gmedical}"
HY_PY="${HYGRAG_PYTHON:-$HOME/miniconda3/envs/hygrag/bin/python}"
PYTHON="${PYTHON:-python}"
OUT="$HERE/outputs/$TAG"
export HYGRAG_CONCURRENCY="$(scale_concurrency "${HYGRAG_CONCURRENCY:-500}")"
echo "[hygrag] concurrency $HYGRAG_CONCURRENCY"
"$HY_PY" prepare_data.py --dataset quality --quality-inputs "$QUALITY_INPUTS" --out-dir "data/$TAG"
# --allow-fallback: questions that hit a fallback stay flagged in contexts.json / run_report.json instead of stopping
# the run before the answer step (0 of 4,086 HotpotQA / MuSiQue / QuALITY questions were flagged)
"$HY_PY" main.py --dataset quality --data-dir "data/$TAG" --out-dir "$OUT" --questions "$QUALITY_INPUTS/questions.json" --allow-fallback
"$PYTHON" "$EVAL_ROOT/common/answer.py" --dataset gmedical --system hygrag --retrieval "$OUT/contexts.json" --workers "$(scale_concurrency 16)"
"$PYTHON" "$EVAL_ROOT/common/score_gmedical.py" --system hygrag --predictions "$OUT/answers_hygrag_gmedical.json" --workers "$(scale_concurrency 16)"
