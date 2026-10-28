#!/usr/bin/env bash
# LogicRAG on the insurance contracts (CONTRACTS_EVAL_PLAN.md). LogicRAG has no chunker, so its corpus is the
# 1200/100-token chunks HyGRAG's chunker cuts from the contract (chunks_logicrag.json, title ""). Same settings as
# run.sh (3 rounds, top-5). Every query (QA questions, then facts verbatim) runs through LogicRAG; the context passed on
# is its final info_summary (what its own generate_answer reads), for QA and for facts. Then the shared answer step +
# judges (common/contracts_score.py). DeepSeek by default (env_contracts.sh; CONTRACTS_LLM=qwen for local Qwen).
# LogicRAG answers one query at a time, so each contract's queries are split into shards of LOGICRAG_SHARD_SIZE that
# run as separate processes, LOGICRAG_PROCS at a time over all contracts (contracts_shards.py), then merged in order.
# Usage: bash evaluation_scripts/logicrag/run_contracts.sh <short|long|smoke>     (CONTRACTS="a b" for some only)
#   LOGICRAG_SHARD_SIZE  queries per process (default 40)
#   LOGICRAG_PROCS       processes at once (default 12; each loads voyage-4-nano, ~2 GB, on LOGICRAG_GPU, default 1)
set -euo pipefail
SET="${1:?usage: run_contracts.sh <short|long|smoke>}"
source "$(dirname "$0")/../common/env_contracts.sh"
source "$EVAL_ROOT/common/contracts_lib.sh"
HERE="$EVAL_ROOT/logicrag"
cd "$HERE"
export OPENAI_API_KEY="$DEEPSEEK_API_KEY" OPENAI_BASE_URL="$LLM_BASE_URL"
export LOGICRAG_LLM_MODEL="$LLM_MODEL" LOGICRAG_EMBEDDING_MODEL="$EMBEDDING_MODEL"
export LOGICRAG_CALLS_PER_MINUTE="${LOGICRAG_CALLS_PER_MINUTE:-600}"
export CUDA_VISIBLE_DEVICES="${LOGICRAG_GPU:-1}"
SHARD_SIZE="${LOGICRAG_SHARD_SIZE:-40}"
LOGDIR="$EVAL_ROOT/run_logs/contracts"
mkdir -p "$LOGDIR" "$EVAL_ROOT/contracts_results/$SET/logicrag"

run_shard() {  # run_shard <contract> <shard>
  local c=$1 s=$2 SD="$HERE/result/contracts_$SET/$1/shards"
  "$PYTHON" "$HERE/contracts_shards.py" done "$SD/$s.questions.json" "$SD/$s.json" && return 0
  LOGICRAG_RESULT_DIR="$SD" LOGICRAG_CACHE_DIR="$HERE/cache/contracts_$SET/$c/$s" \
    "$PYTHON" run.py --dataset "$SD/$s.questions.json" --corpus "$CONTRACTS_DATA/$SET/$c/chunks_logicrag.json" \
      --limit 0 --max-rounds 3 --top-k 5 --output "$s.json" > "$LOGDIR/logicrag_${SET}_${c}_$s.log" 2>&1
}
export -f run_shard
export SET HERE LOGDIR CONTRACTS_DATA

t0=$(date +%s)
jobs_file="$(mktemp)"
for c in $(contracts_of "$SET"); do
  SD="$HERE/result/contracts_$SET/$c/shards"
  for s in $("$PYTHON" contracts_shards.py split "$CONTRACTS_DATA/$SET/$c/questions.json" "$SD" "$SHARD_SIZE"); do
    echo "$c $s" >> "$jobs_file"
  done
done
echo "[logicrag/$SET] $(wc -l < "$jobs_file") shards, ${LOGICRAG_PROCS:-12} at a time (logs: run_logs/contracts/logicrag_${SET}_*)"
fail=0
xargs -P "${LOGICRAG_PROCS:-12}" -L 1 bash -c 'run_shard "$0" "$1" || { echo "[logicrag] $0 $1 FAILED"; exit 1; }' < "$jobs_file" || fail=1
rm -f "$jobs_file"
for c in $(contracts_of "$SET"); do
  "$PYTHON" contracts_shards.py merge "$CONTRACTS_DATA/$SET/$c/questions.json" "$HERE/result/contracts_$SET/$c/shards" \
    "$HERE/result/contracts_$SET/$c/logicrag.json" || fail=1
done
echo "{\"set\": \"$SET\", \"all_contracts_seconds\": $(( $(date +%s) - t0 ))}" > "$EVAL_ROOT/contracts_results/$SET/logicrag/timing_all.json"
[ "$fail" = 0 ] || { echo "[logicrag/$SET] some shards failed; rerun resumes (finished shards are kept)"; exit 1; }
score_contracts logicrag "$SET"
