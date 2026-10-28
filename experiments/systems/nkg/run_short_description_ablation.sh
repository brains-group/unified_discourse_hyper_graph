#!/usr/bin/env bash
# UNH short-description ablation on the existing 100-question HotpotQA / MuSiQue subset graphs.
# See short_description_ablation.py for what changes (fact->fact edges keep only their short label).
#
# For each dataset it evaluates two conditions on the SAME graph, questions, engine, retrieval settings and LLM:
#   long   original graph, original layout (neutral: label inline + long descriptions appended) = every previous run
#   short  description replaced by the label everywhere, --no-neutral-layout = only the label in scoring and context
#   short_strict  (only when asked for: 2nd arg short_strict or all) short + entity->fact edge labels emptied; in these
#          July graphs those labels hold ~11-word text, mostly a copy of the target fact sentence
# Retrieval settings = the reference runs' (copied from their evaluation_results.json metadata):
#   hotpotqa  sample_exp/nkg/hotpotqa/run_v2 (best HotpotQA run, F1 62.5 with deepseek): dag engine
#   musique   sample_exp/nkg/musique/run_v1  (Qwen3-30B, F1 31.2): new engine
# Results: sample_exp/nkg/<ds>/short_desc_ablation/{long,short,short_strict}/evaluation_results[_N].json (never overwritten).
#
# Usage: bash experiments/systems/nkg/run_short_description_ablation.sh [hotpotqa|musique|all] [long|short|short_strict|both|all]
#   both = long + short (default); all = long + short + short_strict
#   LLM_API_BASE  default http://localhost:8005/v1   (Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 served by vLLM)
#   LLM_MODEL     default Qwen/Qwen3-30B-A3B-Instruct-2507-FP8
#   GPU           GPU for the retrieval models (voyage-4-nano bi-encoder + cross-encoder), default 0
#   WORKERS       questions in parallel, default 5
#   PYTHON        default the `contracts` conda env
set -euo pipefail
DATASETS="${1:-all}"; CONDITIONS="${2:-both}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO"
PYTHON="${PYTHON:-$HOME/miniconda3/envs/contracts/bin/python}"
API_BASE="${LLM_API_BASE:-http://localhost:8005/v1}"
MODEL="${LLM_MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507-FP8}"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
WORKERS="${WORKERS:-5}"
WIKI="/home/taluka/discourse_hyper_graph/evaluation_data/wiki_data"   # the question files these graphs were built for

curl -sf -m 10 "$API_BASE/models" > /dev/null || { echo "no LLM server at $API_BASE"; exit 1; }

run_ds() {
  local ds="$1" src qfile; local -a flags
  case "$ds" in
    hotpotqa)
      src=sample_exp/nkg/hotpotqa/run_v2; qfile="$WIKI/hotpot_questions_100.jsonl"
      flags=(--engine dag --top-k-seeds 40 --max-depth 1 --beam-width 6 --final-top-k 40 --retrieval-node-mode all
             --beam-mmr-lambda 0.85 --final-mmr-lambda 0.85 --search-mode beam_search
             --max-fork-branches 3 --parallel-fallback-chain-threshold 2) ;;
    musique)
      src=sample_exp/nkg/musique/run_v1; qfile="$WIKI/musique_questions_100.jsonl"
      flags=(--engine new --top-k-seeds 12 --max-depth 3 --beam-width 6 --final-top-k 45 --retrieval-node-mode all
             --goal-lookahead 1 --no-use-island-and-bridge --island-pool-size 75 --island-max-bridge-hops 1) ;;
    *) echo "unknown dataset $ds"; exit 1 ;;
  esac
  local root="sample_exp/nkg/$ds/short_desc_ablation"
  "$PYTHON" -m experiments.systems.nkg.short_description_ablation --source-run "$src" --out-root "$root"
  case "$CONDITIONS" in short_strict|all)
    "$PYTHON" -m experiments.systems.nkg.short_description_ablation --source-run "$src" --out-root "$root" --strict ;; esac
  for cond in long short short_strict; do
    case "$CONDITIONS:$cond" in
      both:long|both:short|all:*|"$cond":"$cond") ;;
      *) continue ;;
    esac
    local layout=(--neutral-layout); [ "$cond" != long ] && layout=(--no-neutral-layout)
    echo "=== $ds / $cond  ($MODEL @ $API_BASE, retrieval on GPU $CUDA_VISIBLE_DEVICES)"
    "$PYTHON" -m experiments.systems.nkg.hqa_musique_evaluation --mode evaluation --dataset "$ds" \
      --questions-file "$qfile" --run-dir "$root/$cond" \
      --provider local_vllm --model "$MODEL" --api-base "$API_BASE" \
      "${flags[@]}" --deduplicate-paths "${layout[@]}" \
      --max-context-tokens 2000 --answer-max-tokens 128 --max-workers "$WORKERS"
  done
}

if [ "$DATASETS" = all ]; then run_ds hotpotqa; run_ds musique; else run_ds "$DATASETS"; fi
"$PYTHON" -m experiments.systems.nkg.short_description_ablation_report || true
