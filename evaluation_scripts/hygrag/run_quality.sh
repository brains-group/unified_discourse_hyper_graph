#!/usr/bin/env bash
# HyGRAG (vendored in HyGRAG/, method HKGraphTreeLSH) on QuALITY (evaluation_scripts/QUALITY_EVAL_PLAN.md), the way
# DIGIMON/HyGRAG run their own QuALITY copy: Corpus.json = one row per article (here the 115 dev articles,
# context "title\ntext"), chunked by HyGRAG itself (1200/100 per article) into one pooled graph; every question
# (with its four options) is retrieved against the whole pool. Then the shared multiple-choice answer step and
# score_quality.py. Every LLM call is cached in outputs/<tag>/llm_cache.sqlite.
# Usage: bash evaluation_scripts/hygrag/run_quality.sh
#   QUALITY_INPUTS      inputs dir (default evaluation_data/quality/inputs; build_quality_inputs.py)
#   QUALITY_TAG         output name (default quality -> data/quality, outputs/quality)
#   HYGRAG_CONCURRENCY  concurrent LLM requests (base, default 500); CONCURRENCY_MULTIPLIER multiplies it and the
#                       answer/score workers (base 16)
set -euo pipefail
source "$(dirname "$0")/../common/env.sh"
HERE="$EVAL_ROOT/hygrag"
cd "$HERE"
export QUALITY_INPUTS="${QUALITY_INPUTS:-$REPO_ROOT/evaluation_data/quality/inputs}"
TAG="${QUALITY_TAG:-quality}"
HY_PY="${HYGRAG_PYTHON:-$HOME/miniconda3/envs/hygrag/bin/python}"
PYTHON="${PYTHON:-python}"
OUT="$HERE/outputs/$TAG"
export HYGRAG_CONCURRENCY="$(scale_concurrency "${HYGRAG_CONCURRENCY:-500}")"
echo "[hygrag] concurrency $HYGRAG_CONCURRENCY"
"$HY_PY" prepare_data.py --dataset quality --quality-inputs "$QUALITY_INPUTS" --out-dir "data/$TAG"
"$HY_PY" main.py --dataset quality --data-dir "data/$TAG" --out-dir "$OUT" --questions "$QUALITY_INPUTS/questions.json"
"$PYTHON" "$EVAL_ROOT/common/answer.py" --dataset quality --system hygrag --retrieval "$OUT/contexts.json" --workers "$(scale_concurrency 16)"
"$PYTHON" "$EVAL_ROOT/common/score_quality.py" --system hygrag --predictions "$OUT/answers_hygrag_quality.json" --workers "$(scale_concurrency 16)"
