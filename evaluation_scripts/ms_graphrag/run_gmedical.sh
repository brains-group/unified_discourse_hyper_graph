#!/usr/bin/env bash
# MS GraphRAG on G-Medical (GraphRAG-Bench Medical as MemGraphRAG evaluates it; evaluation_scripts/GMEDICAL_EVAL_PLAN.md).
# Index + retrieval = this system's QuALITY code path (run_quality.sh), unchanged, pointed at the G-Medical inputs
# (evaluation_data/build_gmedical_inputs.py writes the same layout; medical.txt as one document, 2,062 questions), as the
# contracts harness does. Then the shared open-ended answer step (common/answer.py --dataset gmedical: HippoRAG 2 /
# MemGraphRAG one-shot CoT prompt) and common/score_gmedical.py (LLM-Acc, overall and per GraphRAG-Bench level).
# Usage: bash evaluation_scripts/ms_graphrag/run_gmedical.sh
#   GMEDICAL_INPUTS  inputs dir (default evaluation_data/gmedical/inputs; smoke: evaluation_data/gmedical/smoke/inputs)
#   GMEDICAL_TAG     output name (default gmedical)
#   CONCURRENCY_MULTIPLIER  as in run_quality.sh
set -euo pipefail
source "$(dirname "$0")/../common/env.sh"
HERE="$EVAL_ROOT/ms_graphrag"
cd "$HERE"
export GMEDICAL_INPUTS="${GMEDICAL_INPUTS:-$REPO_ROOT/evaluation_data/gmedical/inputs}"
export QUALITY_INPUTS="$GMEDICAL_INPUTS"      # the QuALITY code path reads this inputs dir
TAG="${GMEDICAL_TAG:-gmedical}"
GR_BIN="${GRAPHRAG_BIN:-$HOME/miniconda3/envs/ms_graphrag/bin}"
PYTHON="${PYTHON:-python}"
ROOT="$HERE/caches/$TAG"
OUT="$HERE/outputs/$TAG"
mkdir -p "$ROOT/input" "$OUT"

PROXY_PORT="${PROXY_PORT:-18803}"
export GRAPHRAG_CONCURRENCY="$(scale_concurrency "${GRAPHRAG_CONCURRENCY:-1200}")"
export GRAPHRAG_LLM_API_BASE="http://127.0.0.1:$PROXY_PORT/v1"
# LiteLLM's aiohttp client allows 500 connections per host by default (see run.sh)
export AIOHTTP_CONNECTOR_LIMIT="$((GRAPHRAG_CONCURRENCY + 100))" AIOHTTP_CONNECTOR_LIMIT_PER_HOST="$((GRAPHRAG_CONCURRENCY + 100))"

"$GR_BIN/python" deepseek_proxy.py --port "$PROXY_PORT" --stats "$OUT/proxy_stats.json" &
PROXY_PID=$!
# the proxy is already stopped before answering; under set -e a failing kill in the trap would turn exit 0 into 1
trap 'kill $PROXY_PID 2>/dev/null || true; wait $PROXY_PID 2>/dev/null || true' EXIT
for _ in $(seq 50); do
  (echo > "/dev/tcp/127.0.0.1/$PROXY_PORT") 2>/dev/null && break
  sleep 0.2
done

# Workspace: stock prompts from `graphrag init`, then our settings.yaml (as run.sh)
if [ ! -f "$ROOT/prompts/local_search_system_prompt.txt" ]; then
  "$GR_BIN/graphrag" init --root "$ROOT" --force -m gpt-4.1 -e text-embedding-3-large < /dev/null
  rm -f "$ROOT/.env"
fi
cp settings.yaml "$ROOT/settings.yaml"
# input/ = exactly the article files of this inputs dir
rsync -a --delete --include='*.txt' --exclude='*' "$QUALITY_INPUTS/ms_graphrag_input/" "$ROOT/input/"
echo "[run] $(ls "$ROOT/input" | wc -l) article files in $ROOT/input"

echo "[run] graphrag index  (concurrency $GRAPHRAG_CONCURRENCY, proxy :$PROXY_PORT)"
"$GR_BIN/graphrag" index --root "$ROOT"
"$GR_BIN/python" query.py --dataset quality --root "$ROOT" --out "$OUT/local_search_contexts.json"
kill $PROXY_PID 2>/dev/null || true   # answer/score talk to DeepSeek directly
"$PYTHON" "$EVAL_ROOT/common/answer.py" --dataset gmedical --system ms_graphrag --retrieval "$OUT/local_search_contexts.json" --workers "$(scale_concurrency 16)"
"$PYTHON" "$EVAL_ROOT/common/score_gmedical.py" --system ms_graphrag --predictions "$OUT/answers_ms_graphrag_gmedical.json" --workers "$(scale_concurrency 16)"
