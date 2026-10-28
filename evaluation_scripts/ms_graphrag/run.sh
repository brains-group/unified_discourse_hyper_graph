#!/usr/bin/env bash
# Microsoft GraphRAG 3.2.0 (stock `graphrag init` settings, see UPSTREAM.md) on evaluation_data/<dataset>:
# graphrag index over <ds>_scrambled.txt -> local-search context per question (query.py)
# -> shared answer step (common/answer.py) -> score (common/score.py).
# DeepSeek calls go through deepseek_proxy.py (thinking off, no temperature, json_schema -> json_object),
# which this script starts and stops. Every LLM call is cached in caches/<ds>/cache/, so rerunning after a
# crash replays finished calls for free.
# One-time setup: conda create -n ms_graphrag python=3.12 && conda run -n ms_graphrag pip install -r requirements.txt
# Usage: bash evaluation_scripts/ms_graphrag/run.sh hotpotqa|musique
#   GRAPHRAG_CONCURRENCY  concurrent LLM / embedding requests per job (default 1200)
#   PROXY_PORT            local proxy port (default 18801 hotpotqa, 18802 musique)
#   JUDGE=1               add LLM-Acc
set -euo pipefail
DATASET="${1:?usage: run.sh hotpotqa|musique}"
source "$(dirname "$0")/../common/env.sh"
HERE="$EVAL_ROOT/ms_graphrag"
cd "$HERE"
GR_BIN="${GRAPHRAG_BIN:-$HOME/miniconda3/envs/ms_graphrag/bin}"
PYTHON="${PYTHON:-python}"
ROOT="$HERE/caches/$DATASET"
OUT="$HERE/outputs/$DATASET"
mkdir -p "$ROOT/input" "$OUT"

case "$DATASET" in hotpotqa) default_port=18801 ;; musique) default_port=18802 ;; *) echo "unknown dataset $DATASET"; exit 1 ;; esac
PROXY_PORT="${PROXY_PORT:-$default_port}"
export GRAPHRAG_CONCURRENCY="${GRAPHRAG_CONCURRENCY:-1200}"
export GRAPHRAG_LLM_API_BASE="http://127.0.0.1:$PROXY_PORT/v1"
# LiteLLM's aiohttp client allows 1000 connections / 500 per host by default; every call goes to one host
# (the proxy), so without this graphrag would stall at 500 in flight whatever concurrent_requests says.
export AIOHTTP_CONNECTOR_LIMIT="$((GRAPHRAG_CONCURRENCY + 100))" AIOHTTP_CONNECTOR_LIMIT_PER_HOST="$((GRAPHRAG_CONCURRENCY + 100))"

"$GR_BIN/python" deepseek_proxy.py --port "$PROXY_PORT" --stats "$OUT/proxy_stats.json" &
PROXY_PID=$!
trap 'kill $PROXY_PID 2>/dev/null; wait $PROXY_PID 2>/dev/null || true' EXIT
for _ in $(seq 50); do
  (echo > "/dev/tcp/127.0.0.1/$PROXY_PORT") 2>/dev/null && break
  sleep 0.2
done

# Workspace: stock prompts from `graphrag init`, then our settings.yaml (init output + [unh_v1] endpoint lines)
if [ ! -f "$ROOT/prompts/local_search_system_prompt.txt" ]; then
  "$GR_BIN/graphrag" init --root "$ROOT" --force -m gpt-4.1 -e text-embedding-3-large < /dev/null
  rm -f "$ROOT/.env"  # init writes GRAPHRAG_API_KEY=<API_KEY>; settings.yaml does not use it
fi
cp settings.yaml "$ROOT/settings.yaml"
SRC="$REPO_ROOT/evaluation_data/$DATASET/${DATASET}_scrambled.txt"
cmp -s "$SRC" "$ROOT/input/${DATASET}_scrambled.txt" || cp "$SRC" "$ROOT/input/${DATASET}_scrambled.txt"

echo "[run] graphrag index  (concurrency $GRAPHRAG_CONCURRENCY, proxy :$PROXY_PORT)"
"$GR_BIN/graphrag" index --root "$ROOT"
"$GR_BIN/python" query.py --dataset "$DATASET"
kill $PROXY_PID 2>/dev/null || true   # answer/score talk to DeepSeek directly
"$PYTHON" "$EVAL_ROOT/common/answer.py" --dataset "$DATASET" --system ms_graphrag \
  --retrieval "$OUT/local_search_contexts.json"
"$PYTHON" "$EVAL_ROOT/common/score.py" --dataset "$DATASET" --system ms_graphrag \
  --predictions "$OUT/answers_ms_graphrag_$DATASET.json" ${JUDGE:+--llm-judge --judge-model "$LLM_MODEL" --judge-base-url "$LLM_BASE_URL"}
