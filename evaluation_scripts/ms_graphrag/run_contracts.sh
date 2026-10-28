#!/usr/bin/env bash
# Microsoft GraphRAG 3.2.0 (stock settings.yaml, as run.sh / run_quality.sh) on the insurance contracts
# (CONTRACTS_EVAL_PLAN.md): one workspace per contract, input/ = the contract as one .txt (chunked 1200/100 by graphrag)
# -> graphrag index -> local-search context for every query (QA questions, then facts verbatim; query.py) -> shared
# answer step + judges (common/contracts_score.py). Calls go through deepseek_proxy.py (one per contract) to the local
# LLM (env_contracts.sh: DeepSeek by default): it drops any temperature, maps json_schema -> json_object, and adds
# thinking: disabled for DeepSeek, as in run.sh / run_quality.sh.
# Usage: bash evaluation_scripts/ms_graphrag/run_contracts.sh <short|long|smoke>     (CONTRACTS="a b" for some only)
#   GRAPHRAG_CONCURRENCY  concurrent requests per contract (default 64 for the local server)
#   PROXY_PORT_BASE       proxy port of the first contract (default 18820; contract i uses base + i)
set -euo pipefail
SET="${1:?usage: run_contracts.sh <short|long|smoke>}"
source "$(dirname "$0")/../common/env_contracts.sh"
source "$EVAL_ROOT/common/contracts_lib.sh"
HERE="$EVAL_ROOT/ms_graphrag"
cd "$HERE"
GR_BIN="${GRAPHRAG_BIN:-$HOME/miniconda3/envs/ms_graphrag/bin}"
export GRAPHRAG_CONCURRENCY="${GRAPHRAG_CONCURRENCY:-64}"
export AIOHTTP_CONNECTOR_LIMIT="$((GRAPHRAG_CONCURRENCY + 100))" AIOHTTP_CONNECTOR_LIMIT_PER_HOST="$((GRAPHRAG_CONCURRENCY + 100))"
run_one() {
  local c=$1 IN="$CONTRACTS_DATA/$SET/$1" ROOT="$HERE/caches/contracts_$SET/$1" OUT="$HERE/outputs/contracts_$SET/$1"
  local port=$(( ${PROXY_PORT_BASE:-18820} + $2 ))
  mkdir -p "$ROOT/input" "$OUT"
  export GRAPHRAG_LLM_API_BASE="http://127.0.0.1:$port/v1"
  "$GR_BIN/python" deepseek_proxy.py --port "$port" --stats "$OUT/proxy_stats.json" &
  local proxy=$!
  trap "kill $proxy 2>/dev/null || true" EXIT
  for _ in $(seq 50); do (echo > "/dev/tcp/127.0.0.1/$port") 2>/dev/null && break; sleep 0.2; done
  if [ ! -f "$ROOT/prompts/local_search_system_prompt.txt" ]; then
    "$GR_BIN/graphrag" init --root "$ROOT" --force -m gpt-4.1 -e text-embedding-3-large < /dev/null
    rm -f "$ROOT/.env"
  fi
  cp settings.yaml "$ROOT/settings.yaml"
  rsync -a --delete --include='*.txt' --exclude='*' "$IN/ms_graphrag_input/" "$ROOT/input/"
  "$GR_BIN/graphrag" index --root "$ROOT"
  "$GR_BIN/python" query.py --dataset quality --root "$ROOT" --questions "$IN/questions.json" \
    --out "$OUT/local_search_contexts.json"
}
for_each_contract ms_graphrag "$SET" run_one
score_contracts ms_graphrag "$SET"
