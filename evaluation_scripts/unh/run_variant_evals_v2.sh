#!/usr/bin/env bash
# UNH retrieval-variant evaluations on the 250-question subsets, strictly one step after another (tmux pane).
# Like run_variant_evals.sh, plus a per-step deadline: before EVERY answer+judge step, if DeepSeek would not finish it
# before DEADLINE (default 21:00, when DeepSeek's rate doubles; estimate = now + STEP_EST_MIN minutes, default 10), that
# step is answered AND judged by Qwen3-30B-A3B instead. The Qwen server is the user's own vLLM command, started in tmux
# session "llm" on QWEN_GPU (default: the GPU not used by the evaluation's retrieval) the first time it is needed.
#   phase 1: for each dataset in DATASETS: retrieve all views (one graph embedding; skipped when the contexts are
#            complete), then per view answer (shared HippoRAG 2 prompt) + judge  -> rows unh_<TAG> / unh_<view>_<TAG>
#            (+ "_qwen" when Qwen did it). A view whose scores file already exists is skipped.
#   phase 2: every view with complete contexts: UNH's own August prompt (answer.py --prompt unh_short) + judge
#            -> rows ..._unhprompt (DeepSeek) or ..._unhprompt_qwen. Skipped when its scores file already exists.
#   then collect_results.py --subset for both datasets; stops the Qwen server if this script started it.
# Env: TAG (required), FINAL_TOP_K, TOP_K_SEEDS, SOURCE_PASSAGES, BEAM_MMR, DROP_EMPTY (default 1),
#      DATASETS (default "hotpotqa musique"), VIEWS (default "full hypergraph"), GPU_HOTPOTQA / GPU_MUSIQUE (0 / 1),
#      DEADLINE, STEP_EST_MIN, QWEN_GPU, QWEN_PORT (8005).  Log: run_logs/unh_variant_<TAG>.log (appended).
set -uo pipefail
: "${TAG:?set TAG}"
UNH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_ROOT="$(cd "$UNH_DIR/.." && pwd)"
REPO_ROOT="$(cd "$EVAL_ROOT/.." && pwd)"
if [ -z "${DEEPSEEK_API_KEY:-}" ]; then
  DEEPSEEK_API_KEY="$(grep -E '^\s*DEEPSEEK_API_KEY\s*=' "$REPO_ROOT/.env" | head -1 | cut -d= -f2- | tr -d "\"' ")"
  export DEEPSEEK_API_KEY
fi
source "$EVAL_ROOT/common/env.sh"
export HF_HUB_OFFLINE=1
PY="${PYTHON:-$HOME/miniconda3/envs/unh_eval/bin/python}"
LOG="$EVAL_ROOT/run_logs/unh_variant_${TAG}.log"
DEADLINE="${DEADLINE:-21:00}"
QPORT="${QWEN_PORT:-8005}"
QMODEL="Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
QURL="http://localhost:$QPORT/v1"
QWEN_STARTED_HERE=0
ts() { date '+%F %T'; }
say() { echo "[$(ts)] $*" | tee -a "$LOG"; }

ARGS=(--tag "$TAG")
[ -n "${FINAL_TOP_K:-}" ] && ARGS+=(--final-top-k "$FINAL_TOP_K")
[ -n "${TOP_K_SEEDS:-}" ] && ARGS+=(--top-k-seeds "$TOP_K_SEEDS")
[ -n "${SOURCE_PASSAGES:-}" ] && ARGS+=(--source-passages "$SOURCE_PASSAGES")
[ -n "${BEAM_MMR:-}" ] && ARGS+=(--beam-mmr-lambda "$BEAM_MMR")
[ "${DROP_EMPTY:-1}" = 1 ] && ARGS+=(--drop-empty-entities)
say "v2: variant $TAG: ${ARGS[*]} | deadline $DEADLINE for DeepSeek answer/judge steps, else Qwen"

gpu_of() { if [ "$1" = musique ]; then echo "${GPU_MUSIQUE:-1}"; else echo "${GPU_HOTPOTQA:-0}"; fi; }
vtag() { if [ "$2" = full ]; then echo "$TAG"; else echo "${2}_$TAG"; fi; }
ctx_of() { if [ "$2" = full ]; then echo "contexts_$TAG.json"; else echo "contexts_${2}_$TAG.json"; fi; }
ctx_complete() {
  "$PY" -c "import json,sys; r=json.load(open(sys.argv[1])); sys.exit(0 if len(r)==250 and all(x['status']=='ok' for x in r) else 1)" \
    "$UNH_DIR/outputs/${1}_250_seed0/$(ctx_of "$1" "$2")" 2>/dev/null
}
use_qwen_now() {  # true when a DeepSeek step started now would not end before DEADLINE
  [ "$(date -d "+${STEP_EST_MIN:-10} min" +%s)" -ge "$(date -d "today $DEADLINE" +%s)" ]
}
ensure_qwen() {
  curl -sf "$QURL/models" >/dev/null && return 0
  local g="${QWEN_GPU:-}"
  if [ -z "$g" ]; then  # the GPU not used by the evaluation's retrieval (musique -> 1, so Qwen -> 0)
    if [ "${GPU_MUSIQUE:-1}" = 1 ]; then g=0; else g=1; fi
  fi
  say "starting the user's vLLM Qwen server in tmux 'llm' on GPU $g (port $QPORT)"
  tmux send-keys -t llm C-u
  tmux send-keys -t llm "CUDA_VISIBLE_DEVICES=$g python3 -m vllm.entrypoints.openai.api_server   --model $QMODEL    --port $QPORT   --gpu-memory-utilization 0.91   --max-model-len 128000   --tensor-parallel-size 1  --trust-remote-code" Enter
  QWEN_STARTED_HERE=1
  for _ in $(seq 1 120); do curl -sf "$QURL/models" >/dev/null && { say "Qwen server ready"; return 0; }; sleep 10; done
  say "Qwen server did not come up in 20 min (see tmux llm)"; return 1
}

answer_and_score() {  # ds view prompt base_suffix   (backend chosen here, per step)
  local ds=$1 view=$2 prompt=$3 suffix=$4 out="$UNH_DIR/outputs/${1}_250_seed0" sub="$REPO_ROOT/evaluation_data/$1/subsets/${1}_250_seed0.json"
  local model="$LLM_MODEL" base="$LLM_BASE_URL" keyenv=DEEPSEEK_API_KEY backend=deepseek
  if use_qwen_now; then
    ensure_qwen || return 1
    model="$QMODEL"; base="$QURL"; keyenv=QWEN_API_KEY; backend=qwen; suffix="${suffix}_qwen"
  fi
  local name="unh_$(vtag "$ds" "$view")$suffix"
  if [ -f "$out/scores_${name}_$ds.json" ]; then say "$ds $view ($prompt, $backend): already scored, skipping"; return 0; fi
  say "$ds $view: answer + judge ($prompt prompt, $backend)"
  local think=1; [ "$backend" = qwen ] && think=0   # vLLM: no DeepSeek 'thinking' field (Instruct-2507 does not think)
  QWEN_API_KEY=EMPTY LLM_DISABLE_THINKING=$think "$PY" "$EVAL_ROOT/common/answer.py" --dataset "$ds" --system unh \
      --subset "$sub" --prompt "$prompt" --model "$model" --base-url "$base" --api-key-env "$keyenv" \
      --retrieval "$out/$(ctx_of "$ds" "$view")" --out "$out/answers_${name}_$ds.json" >> "$LOG" 2>&1 || return 1
  QWEN_API_KEY=EMPTY LLM_DISABLE_THINKING=$think "$PY" "$EVAL_ROOT/common/score.py" --dataset "$ds" --system unh \
      --subset "$sub" --llm-judge --judge-model "$model" --judge-base-url "$base" --judge-api-key-env "$keyenv" \
      --predictions "$out/answers_${name}_$ds.json" --out "$out/scores_${name}_$ds.json" >> "$LOG" 2>&1 || return 1
  say "$ds $view ($prompt, $backend) -> $name: $("$PY" -c "import json;print(json.load(open('$out/scores_${name}_$ds.json'))['metrics_percent'])")"
}

scored_any() {  # ds view base_suffix: scored with either backend already?
  local out="$UNH_DIR/outputs/${1}_250_seed0" name="unh_$(vtag "$1" "$2")$3"
  [ -f "$out/scores_${name}_$1.json" ] || [ -f "$out/scores_${name}_qwen_$1.json" ]
}

# ---- phase 1 ----
for ds in ${DATASETS:-hotpotqa musique}; do
  sub="$REPO_ROOT/evaluation_data/$ds/subsets/${ds}_250_seed0.json"
  views="${VIEWS:-full hypergraph}"
  need=""
  for view in $views; do ctx_complete "$ds" "$view" || need="$need $view"; done
  if [ -n "$need" ]; then
    need="${need# }"
    say "phase 1: $ds: retrieve views ${need// /,} (one graph embedding; query planner on DeepSeek)"
    (cd "$UNH_DIR" && CUDA_VISIBLE_DEVICES="$(gpu_of "$ds")" "$PY" main.py --dataset "$ds" --subset "$sub" \
        --out-dir "$UNH_DIR/outputs/${ds}_250_seed0" --step retrieve --view "${need// /,}" "${ARGS[@]}" >> "$LOG" 2>&1) \
      || say "phase 1: $ds: retrieval reported failures"
  fi
  for view in $views; do
    ctx_complete "$ds" "$view" || { say "phase 1: $ds $view: contexts incomplete; skipping (rerun resumes)"; continue; }
    scored_any "$ds" "$view" "" && { say "phase 1: $ds $view: already scored, skipping"; continue; }
    answer_and_score "$ds" "$view" hipporag2 "" || say "phase 1: $ds $view: answer/judge FAILED"
  done
done

# ---- phase 2: UNH's own prompt on the same contexts ----
for ds in hotpotqa musique; do
  for view in ${VIEWS:-full hypergraph}; do
    ctx_complete "$ds" "$view" || continue
    scored_any "$ds" "$view" "_unhprompt" && { say "phase 2: $ds $view: already scored, skipping"; continue; }
    answer_and_score "$ds" "$view" unh_short "_unhprompt" || say "phase 2: $ds $view: answer/judge FAILED"
  done
done

"$PY" "$EVAL_ROOT/common/collect_results.py" --subset "$REPO_ROOT"/evaluation_data/{hotpotqa,musique}/subsets/*_250_seed0.json >> "$LOG" 2>&1
if [ "$QWEN_STARTED_HERE" = 1 ]; then tmux send-keys -t llm C-c; say "stopped the Qwen server (tmux llm)"; fi
say "all done"
