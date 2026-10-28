#!/usr/bin/env bash
# HyGRAG (vendored, method HKGraphTreeLSH) on the insurance contracts (CONTRACTS_EVAL_PLAN.md): Corpus.json = the
# contract as one row, chunked by HyGRAG itself (1200/100) into one graph per contract; _retrieve_relevant_contexts for
# every query (QA questions, then facts verbatim); then the shared answer step + judges (common/contracts_score.py).
# DeepSeek by default (env_contracts.sh; CONTRACTS_LLM=qwen for local Qwen). main.py sends DeepSeek's default temperature
# 1.0 explicitly on DeepSeek (HyGRAG always sends one), none on other endpoints, as in every other HyGRAG run.
# Usage: bash evaluation_scripts/hygrag/run_contracts.sh <short|long|smoke>     (CONTRACTS="a b" for some only)
#   HYGRAG_CONCURRENCY  concurrent LLM requests per contract (default 64 for the local server)
set -euo pipefail
SET="${1:?usage: run_contracts.sh <short|long|smoke>}"
source "$(dirname "$0")/../common/env_contracts.sh"
source "$EVAL_ROOT/common/contracts_lib.sh"
HERE="$EVAL_ROOT/hygrag"
cd "$HERE"
HY_PY="${HYGRAG_PYTHON:-$HOME/miniconda3/envs/hygrag/bin/python}"
export HYGRAG_CONCURRENCY="${HYGRAG_CONCURRENCY:-64}"
run_one() {
  local IN="$CONTRACTS_DATA/$SET/$1"
  "$HY_PY" prepare_data.py --dataset quality --quality-inputs "$IN" --out-dir "data/contracts_$SET/$1"
  "$HY_PY" main.py --dataset quality --data-dir "data/contracts_$SET/$1" --out-dir "$HERE/outputs/contracts_$SET/$1" \
    --questions "$IN/questions.json" --allow-fallback  # flagged queries stay recorded in contexts.json / run_report.json (as run_gmedical.sh)
}
for_each_contract hygrag "$SET" run_one
score_contracts hygrag "$SET"
