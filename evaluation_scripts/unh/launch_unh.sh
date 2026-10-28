#!/usr/bin/env bash
# UNH on HotpotQA and MuSiQue (250-question subsets), unattended:
#   1. both constructions start at once (in parallel);
#   2. as soon as one finishes, its evaluation runs: view full, then view hypergraph (retrieve -> shared answer step
#      -> score with the LLM judge);
#   3. evaluations never overlap: if the other construction finishes while an evaluation is running, its
#      evaluation waits and starts when the running one is done (order = order in which constructions finish);
#   4. at the end, collect_results.py --subset prints all seven systems on the same 250 questions.
# Detaches itself (setsid + nohup), so it survives the terminal / tmux closing. Logs in evaluation_scripts/run_logs/:
#   unh_pipeline.log (this orchestration; unh_pipeline_<ds>.log when datasets are named), unh_construct_<ds>.log, unh_evaluate_<ds>.log. Stop: stop_all.sh.
# A rerun resumes: construction from its checkpoint, retrieval from contexts*.json, answers from reader_cache.sqlite.
#
# Usage: bash evaluation_scripts/unh/launch_unh.sh [datasets...]      (default: hotpotqa musique)
#   GPU_HOTPOTQA / GPU_MUSIQUE   GPU for that dataset's construction and evaluation (embedding models only;
#                       defaults 0 / 1). GPU sets the default for any dataset without its own (default 0).
#   UNH_CONCURRENCY     construction --max-workers and --max-sub-workers per dataset (default 2300)
#   UNH_MAX_WORKERS / UNH_MAX_SUB_WORKERS   set the two separately (override UNH_CONCURRENCY)
#   SEQUENTIAL=1        fully sequential: construct + evaluate the first dataset, then the next (default: constructions
#                       in parallel, evaluations queued one at a time)
#   UNH_QUESTIONS=all   all questions on the full corpus (outputs/<ds>_full); default: the SUBSET_N subset
#   UNH_VIEWS / UNH_PROMPTS  views (default "full hypergraph discourse") / answer prompts (default "hipporag2 unh_short")
#   UNH_CORPUS          subset (default) | full     SUBSET_N / SUBSET_SEED: which subset files (default 250 / 0)
#   JUDGE=0             skip the LLM judge
set -euo pipefail
UNH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_ROOT="$(cd "$UNH_DIR/.." && pwd)"
REPO_ROOT="$(cd "$EVAL_ROOT/.." && pwd)"
LOGS="$EVAL_ROOT/run_logs"
mkdir -p "$LOGS"

# one orchestration log / pid per invocation: unh_pipeline.log for the default (both datasets), else
# unh_pipeline_<ds>[_<ds>].log, so a second launch (e.g. musique while hotpotqa evaluates) never clobbers the first
TAG="unh_pipeline"; [ $# -gt 0 ] && TAG="unh_pipeline_$(IFS=_; echo "$*")"

if [ -z "${UNH_FOREGROUND:-}" ]; then
  if [ -z "${DEEPSEEK_API_KEY:-}" ] && [ -f "$REPO_ROOT/.env" ]; then
    DEEPSEEK_API_KEY="$(grep -E '^\s*DEEPSEEK_API_KEY\s*=' "$REPO_ROOT/.env" | head -1 | cut -d= -f2- | tr -d "\"' ")"
    export DEEPSEEK_API_KEY
  fi
  : "${DEEPSEEK_API_KEY:?export DEEPSEEK_API_KEY or put it in $REPO_ROOT/.env}"
  url="${EMBEDDING_BASE_URL:-http://localhost:8001/v1}"
  UNH_FOREGROUND=1 setsid nohup bash "$0" "$@" > "$LOGS/$TAG.log" 2>&1 < /dev/null &
  echo "$!" > "$LOGS/$TAG.pid"
  echo "started UNH pipeline (pid $!; GPUs: hotpotqa ${GPU_HOTPOTQA:-0}, musique ${GPU_MUSIQUE:-1}) -> $LOGS/$TAG.log"
  echo "  construction logs: $LOGS/unh_construct_<ds>.log   evaluation logs: $LOGS/unh_evaluate_<ds>.log"
  exit 0
fi

export PYTHON="${PYTHON:-$HOME/miniconda3/envs/unh_eval/bin/python}"
# one GPU per dataset: GPU_<DATASET> (hotpotqa 0, musique 1), else GPU, else 0
gpu_of() {
  local v="GPU_${1^^}"
  if [ -n "${!v:-}" ]; then echo "${!v}"
  elif [ "$1" = hotpotqa ]; then echo 0
  elif [ "$1" = musique ]; then echo 1
  else echo "${GPU:-0}"; fi
}
export JUDGE="${JUDGE:-1}"
N="${SUBSET_N:-250}"; SEED="${SUBSET_SEED:-0}"
if [ $# -gt 0 ]; then DATASETS=("$@"); else DATASETS=(hotpotqa musique); fi
ts() { date '+%F %T'; }
ALL_QUESTIONS=0; [ "${UNH_QUESTIONS:-subset}" = all ] && ALL_QUESTIONS=1
subset_of() {  # the SUBSET value pipeline.sh gets for a dataset
  if [ "$ALL_QUESTIONS" = 1 ]; then echo none; else echo "$REPO_ROOT/evaluation_data/$1/subsets/${1}_${N}_seed${SEED}.json"; fi
}
if [ "$ALL_QUESTIONS" = 1 ]; then
  echo "[$(ts)] all questions on the full corpus (UNH_QUESTIONS=all) -> outputs/<ds>_full"
else
  for ds in "${DATASETS[@]}"; do
    sub="$(subset_of "$ds")"
    [ -f "$sub" ] || { echo "[$(ts)] missing $sub: python evaluation_scripts/common/make_subset.py --dataset $ds --n $N --seed $SEED"; exit 1; }
  done
fi

failed=0
evaluate() {
  echo "[$(ts)] construction done: $1 -> evaluating (views: ${UNH_VIEWS:-full hypergraph discourse}; prompts: ${UNH_PROMPTS:-hipporag2 unh_short})"
  if CUDA_VISIBLE_DEVICES="$(gpu_of "$1")" SUBSET="$(subset_of "$1")" \
       bash "$UNH_DIR/pipeline.sh" evaluate "$1" > "$LOGS/unh_evaluate_$1.log" 2>&1; then
    echo "[$(ts)] evaluation done: $1"
  else
    echo "[$(ts)] evaluation had failures: $1 (see unh_evaluate_$1.log)"; failed=1
  fi
}

if [ "${SEQUENTIAL:-0}" = 1 ]; then
  # fully sequential: construct ds1 -> evaluate ds1 -> construct ds2 -> evaluate ds2 (one job at a time)
  for ds in "${DATASETS[@]}"; do
    echo "[$(ts)] construction started: $ds (sequential, GPU $(gpu_of "$ds")) -> $LOGS/unh_construct_$ds.log"
    if CUDA_VISIBLE_DEVICES="$(gpu_of "$ds")" SUBSET="$(subset_of "$ds")" \
         bash "$UNH_DIR/pipeline.sh" construct "$ds" > "$LOGS/unh_construct_$ds.log" 2>&1; then
      evaluate "$ds"
    else
      echo "[$(ts)] construction FAILED: $ds (see unh_construct_$ds.log). Rerun to resume from its checkpoint."; failed=1
    fi
  done
else
  declare -A DS_OF_PID
  for ds in "${DATASETS[@]}"; do
    CUDA_VISIBLE_DEVICES="$(gpu_of "$ds")" SUBSET="$(subset_of "$ds")" \
      bash "$UNH_DIR/pipeline.sh" construct "$ds" > "$LOGS/unh_construct_$ds.log" 2>&1 &
    DS_OF_PID[$!]="$ds"
    echo "[$(ts)] construction started: $ds (pid $!, GPU $(gpu_of "$ds")) -> $LOGS/unh_construct_$ds.log"
  done
  remaining=${#DATASETS[@]}
  while [ "$remaining" -gt 0 ]; do
    # wait for whichever construction finishes next (evaluations below run in this same loop, so never in parallel)
    if wait -n -p done_pid "${!DS_OF_PID[@]}"; then rc=0; else rc=$?; fi
    ds="${DS_OF_PID[$done_pid]}"; unset "DS_OF_PID[$done_pid]"; remaining=$((remaining - 1))
    if [ "$rc" -ne 0 ]; then
      echo "[$(ts)] construction FAILED: $ds (exit $rc; see unh_construct_$ds.log). Rerun to resume from its checkpoint."
      failed=1; continue
    fi
    evaluate "$ds"
  done
fi

if [ "$ALL_QUESTIONS" = 1 ]; then
  "$PYTHON" "$EVAL_ROOT/common/collect_results.py" || true
else
  subs=()
  for ds in "${DATASETS[@]}"; do subs+=("$(subset_of "$ds")"); done
  "$PYTHON" "$EVAL_ROOT/common/collect_results.py" --subset "${subs[@]}" || true
fi
echo "[$(ts)] UNH pipeline finished (failures: $failed)"
exit $failed
