#!/usr/bin/env bash
# Launch HippoRAG 2, MemGraphRAG, LogicRAG and HyperGraphRAG on HotpotQA and MuSiQue in parallel (8 jobs).
# Each job runs its system's run.sh (own index + retrieval -> common/answer.py -> common/score.py with the
# LLM judge). Logs: evaluation_scripts/run_logs/<system>_<dataset>.log
#
# Needs one voyage-4-nano embedding endpoint (vLLM batches concurrent requests; see common/env.sh for the
# command), by default http://localhost:8001/v1. Jobs run on GPU $GPU (default 0); only LogicRAG uses a GPU,
# for its own in-process SentenceTransformer copy of voyage-4-nano. Each job gets its own process group
# (setsid + nohup), so it keeps running after the terminal / tmux session closes; stop them with stop_all.sh.
# Usage: bash evaluation_scripts/launch_all.sh [systems...]   (default: the four above; ms_graphrag / hygrag only when named,
#   e.g. `launch_all.sh ms_graphrag`; for UNH use unh/launch_unh.sh (sequential evaluations; unh/NOTES.md); their run.sh use their own conda envs; concurrency: GRAPHRAG_CONCURRENCY
#   (default 1200), HYGRAG_CONCURRENCY (default 500))
set -euo pipefail
EVAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$EVAL_ROOT/.." && pwd)"
if [ -z "${DEEPSEEK_API_KEY:-}" ] && [ -f "$REPO_ROOT/.env" ]; then
  DEEPSEEK_API_KEY="$(grep -E '^\s*DEEPSEEK_API_KEY\s*=' "$REPO_ROOT/.env" | head -1 | cut -d= -f2- | tr -d "\"' ")"
  export DEEPSEEK_API_KEY
fi
: "${DEEPSEEK_API_KEY:?export DEEPSEEK_API_KEY or put it in $REPO_ROOT/.env}"
export PYTHON="${PYTHON:-$HOME/miniconda3/envs/unh_eval/bin/python}"
export JUDGE="${JUDGE-1}"
if [ $# -gt 0 ]; then SYSTEMS=("$@"); else SYSTEMS=(hipporag2 memgraphrag logicrag hypergraphrag); fi
url="${EMBEDDING_BASE_URL:-http://localhost:8001/v1}"
gpu="${GPU:-0}"
curl -sf -m 30 "$url/embeddings" -H 'Content-Type: application/json' \
  -d '{"model":"voyageai/voyage-4-nano","input":["ping"]}' > /dev/null || { echo "no embedding server at $url"; exit 1; }
mkdir -p "$EVAL_ROOT/run_logs"
for ds in hotpotqa musique; do
  for sys in "${SYSTEMS[@]}"; do
    log="$EVAL_ROOT/run_logs/${sys}_${ds}.log"
    EMBEDDING_BASE_URL="$url" CUDA_VISIBLE_DEVICES="$gpu" \
      setsid nohup bash "$EVAL_ROOT/$sys/run.sh" "$ds" > "$log" 2>&1 < /dev/null &
    echo "$!" > "$EVAL_ROOT/run_logs/${sys}_${ds}.pid"
    echo "started $sys $ds (pid $!, GPU $gpu, embeddings $url) -> $log"
  done
done
