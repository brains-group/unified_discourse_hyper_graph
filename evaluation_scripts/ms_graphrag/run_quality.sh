#!/usr/bin/env bash
# Microsoft GraphRAG 3.2.0 (same stock settings.yaml as run.sh) on QuALITY (evaluation_scripts/QUALITY_EVAL_PLAN.md):
# input/ = one <article_id>.txt per dev article ("title\ntext"; graphrag's native input, each file one document,
# chunked 1200/100 per document) -> graphrag index -> local-search context for every question (with its options,
# query.py) -> shared multiple-choice answer step -> score_quality.py.
# DeepSeek calls go through deepseek_proxy.py (thinking off, no temperature, json_schema -> json_object), started and
# stopped here. Every LLM call is cached in caches/<tag>/cache/.
# Usage: bash evaluation_scripts/ms_graphrag/run_quality.sh
#   QUALITY_INPUTS        inputs dir (default evaluation_data/quality/inputs; build_quality_inputs.py)
#   QUALITY_TAG           output name (default quality -> caches/quality, outputs/quality)
#   GRAPHRAG_CONCURRENCY  concurrent LLM / embedding requests (base, default 1200); CONCURRENCY_MULTIPLIER multiplies
#                         it and the answer/score workers (base 16)
#   PROXY_PORT            local proxy port (default 18803; run.sh uses 18801 / 18802)
set -euo pipefail
source "$(dirname "$0")/../common/env.sh"
HERE="$EVAL_ROOT/ms_graphrag"
cd "$HERE"
export QUALITY_INPUTS="${QUALITY_INPUTS:-$REPO_ROOT/evaluation_data/quality/inputs}"
TAG="${QUALITY_TAG:-quality}"
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
trap 'kill $PROXY_PID 2>/dev/null; wait $PROXY_PID 2>/dev/null || true' EXIT
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
"$PYTHON" "$EVAL_ROOT/common/answer.py" --dataset quality --system ms_graphrag --retrieval "$OUT/local_search_contexts.json" --workers "$(scale_concurrency 16)"
"$PYTHON" "$EVAL_ROOT/common/score_quality.py" --system ms_graphrag --predictions "$OUT/answers_ms_graphrag_quality.json" --workers "$(scale_concurrency 16)"
