#!/usr/bin/env bash
# Launch the QuALITY benchmark (evaluation_scripts/QUALITY_EVAL_PLAN.md, QUALITY_IMPLEMENTATION.md) for every
# baseline in parallel: each job runs <system>/run_quality.sh (own index + retrieval over the 115 pooled dev
# articles -> common/answer.py --dataset quality -> common/score_quality.py).
# Logs: evaluation_scripts/run_logs/<system>_<tag>.log; stop with stop_all.sh.
#
# Needs the voyage-4-nano embedding endpoint (common/env.sh), default http://localhost:8001/v1. LogicRAG also
# loads its own SentenceTransformer copy on GPU $GPU (default 0). Each job gets its own process group
# (setsid + nohup), so it survives the terminal / tmux session closing.
# Usage: bash evaluation_scripts/launch_quality.sh [systems...]
#   default systems: hipporag2 memgraphrag logicrag hypergraphrag ms_graphrag hygrag
#   QUALITY_INPUTS  inputs dir (default evaluation_data/quality/inputs; build with evaluation_data/build_quality_inputs.py)
#   QUALITY_TAG     output / log name (default quality); e.g. a smoke test on the 6-article subset:
#     QUALITY_TAG=quality_smoke QUALITY_INPUTS=evaluation_data/subsets/quality_100q_seed0/inputs bash evaluation_scripts/launch_quality.sh
#   GRAPHRAG_CONCURRENCY (default 1200), HYGRAG_CONCURRENCY (default 500)
#   CONCURRENCY_MULTIPLIER  multiplies every system's LLM concurrency settings (default 1; see common/env.sh), e.g. 2 or 3
set -euo pipefail
EVAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$EVAL_ROOT/.." && pwd)"
if [ -z "${DEEPSEEK_API_KEY:-}" ] && [ -f "$REPO_ROOT/.env" ]; then
  DEEPSEEK_API_KEY="$(grep -E '^\s*DEEPSEEK_API_KEY\s*=' "$REPO_ROOT/.env" | head -1 | cut -d= -f2- | tr -d "\"' ")"
  export DEEPSEEK_API_KEY
fi
: "${DEEPSEEK_API_KEY:?export DEEPSEEK_API_KEY or put it in $REPO_ROOT/.env}"
export PYTHON="${PYTHON:-$HOME/miniconda3/envs/unh_eval/bin/python}"
export QUALITY_INPUTS="$(cd "${QUALITY_INPUTS:-$REPO_ROOT/evaluation_data/quality/inputs}" && pwd)"
export QUALITY_TAG="${QUALITY_TAG:-quality}"
export CONCURRENCY_MULTIPLIER="${CONCURRENCY_MULTIPLIER:-1}"
if [ $# -gt 0 ]; then SYSTEMS=("$@"); else SYSTEMS=(hipporag2 memgraphrag logicrag hypergraphrag ms_graphrag hygrag); fi
[ -f "$QUALITY_INPUTS/questions.json" ] || { echo "missing $QUALITY_INPUTS/questions.json: run evaluation_data/build_quality_inputs.py"; exit 1; }
url="${EMBEDDING_BASE_URL:-http://localhost:8001/v1}"
gpu="${GPU:-0}"
curl -sf -m 30 "$url/embeddings" -H 'Content-Type: application/json' \
  -d '{"model":"voyageai/voyage-4-nano","input":["ping"]}' > /dev/null || { echo "no embedding server at $url"; exit 1; }
mkdir -p "$EVAL_ROOT/run_logs"
echo "inputs: $QUALITY_INPUTS ($(grep -o '"questions": [0-9]*' "$QUALITY_INPUTS/manifest.json"), tag $QUALITY_TAG, concurrency x$CONCURRENCY_MULTIPLIER)"
for sys in "${SYSTEMS[@]}"; do
  [ -f "$EVAL_ROOT/$sys/run_quality.sh" ] || { echo "no $sys/run_quality.sh"; exit 1; }
  log="$EVAL_ROOT/run_logs/${sys}_${QUALITY_TAG}.log"
  EMBEDDING_BASE_URL="$url" CUDA_VISIBLE_DEVICES="$gpu" \
    setsid nohup bash "$EVAL_ROOT/$sys/run_quality.sh" > "$log" 2>&1 < /dev/null &
  echo "$!" > "$EVAL_ROOT/run_logs/${sys}_${QUALITY_TAG}.pid"
  echo "started $sys (pid $!, GPU $gpu, embeddings $url) -> $log"
done
