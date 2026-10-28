#!/usr/bin/env bash
# HyGRAG (zjunet/HyGRAG @ fa1a57e, vendored in HyGRAG/) on evaluation_data/<dataset>:
# per-passage corpus (prepare_data.py) -> HyGRAG index + context per question (main.py)
# -> shared answer step (common/answer.py) -> score (common/score.py).
# Every LLM call is cached in outputs/<ds>/llm_cache.sqlite, so rerunning after a crash replays finished calls.
# One-time setup: conda env `hygrag` (Python 3.10, HyGRAG/requirements.txt with faiss-cpu; see UPSTREAM.md)
# Usage: bash evaluation_scripts/hygrag/run.sh hotpotqa|musique
#   HYGRAG_CONCURRENCY  concurrent LLM requests (default 500)
#   JUDGE=1             add LLM-Acc
set -euo pipefail
DATASET="${1:?usage: run.sh hotpotqa|musique}"
source "$(dirname "$0")/../common/env.sh"
HERE="$EVAL_ROOT/hygrag"
cd "$HERE"
HY_PY="${HYGRAG_PYTHON:-$HOME/miniconda3/envs/hygrag/bin/python}"
PYTHON="${PYTHON:-python}"
OUT="$HERE/outputs/$DATASET"
"$HY_PY" prepare_data.py --dataset "$DATASET"
"$HY_PY" main.py --dataset "$DATASET"
"$PYTHON" "$EVAL_ROOT/common/answer.py" --dataset "$DATASET" --system hygrag --retrieval "$OUT/contexts.json"
"$PYTHON" "$EVAL_ROOT/common/score.py" --dataset "$DATASET" --system hygrag \
  --predictions "$OUT/answers_hygrag_$DATASET.json" ${JUDGE:+--llm-judge --judge-model "$LLM_MODEL" --judge-base-url "$LLM_BASE_URL"}
