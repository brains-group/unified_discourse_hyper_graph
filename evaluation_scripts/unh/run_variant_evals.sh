#!/usr/bin/env bash
# UNH retrieval-variant evaluations on the 250-question subsets, strictly one after another (meant for a tmux pane).
#   phase 1: hotpotqa full, hotpotqa hypergraph, musique full, musique hypergraph
#            each = main.py --step retrieve (variant flags, --tag) -> common/answer.py (shared HippoRAG 2 prompt)
#            -> common/score.py with the LLM judge
#            (both views of a dataset are retrieved by one main.py process: one graph embedding)
#   phase 2: the same four contexts answered with UNH's own August prompt (answer.py --prompt unh_short), then scored.
#            No new retrieval. Backend: DeepSeek if phase 2 can finish before DEADLINE (default 21:00; DeepSeek's rate
#            doubles after it), estimated as now + PHASE2_DEEPSEEK_MIN (default 30) minutes; otherwise Qwen3-30B-A3B
#            (vLLM on a free GPU, answer AND judge, rows suffixed _qwen; also the shared prompt on Qwen, for the
#            prompt-vs-model split); if no GPU is free, phase 2 is skipped.
#   then:    collect_results.py --subset for both datasets.
# Rows in collect_results: unh_<TAG> / unh_hypergraph_<TAG> (phase 1) and ..._unhprompt (phase 2).
# Env: TAG (required), FINAL_TOP_K, TOP_K_SEEDS, SOURCE_PASSAGES, BEAM_MMR, DROP_EMPTY (1/0, default 1),
#      DATASETS (default "hotpotqa musique"), VIEWS (default "full hypergraph"), GPU_HOTPOTQA / GPU_MUSIQUE (0 / 1).
# Log: run_logs/unh_variant_<TAG>.log (+ each step's output).
set -uo pipefail
: "${TAG:?set TAG, e.g. TAG=k20p3_mmr75}"
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
ts() { date '+%F %T'; }
say() { echo "[$(ts)] $*" | tee -a "$LOG"; }

ARGS=(--tag "$TAG")
[ -n "${FINAL_TOP_K:-}" ] && ARGS+=(--final-top-k "$FINAL_TOP_K")
[ -n "${TOP_K_SEEDS:-}" ] && ARGS+=(--top-k-seeds "$TOP_K_SEEDS")
[ -n "${SOURCE_PASSAGES:-}" ] && ARGS+=(--source-passages "$SOURCE_PASSAGES")
[ -n "${BEAM_MMR:-}" ] && ARGS+=(--beam-mmr-lambda "$BEAM_MMR")
[ "${DROP_EMPTY:-1}" = 1 ] && ARGS+=(--drop-empty-entities)
say "variant $TAG: ${ARGS[*]}"

gpu_of() { if [ "$1" = musique ]; then echo "${GPU_MUSIQUE:-1}"; else echo "${GPU_HOTPOTQA:-0}"; fi; }
vtag() { if [ "$2" = full ]; then echo "$TAG"; else echo "${2}_$TAG"; fi; }          # dataset view -> row suffix
ctx_of() { if [ "$2" = full ]; then echo "contexts_$TAG.json"; else echo "contexts_${2}_$TAG.json"; fi; }

answer_and_score() {  # ds view prompt suffix [model base_url api_key_env]
  local ds=$1 view=$2 prompt=$3 suffix=$4 out="$UNH_DIR/outputs/${1}_250_seed0" sub="$REPO_ROOT/evaluation_data/$1/subsets/${1}_250_seed0.json"
  local model="${5:-$LLM_MODEL}" base="${6:-$LLM_BASE_URL}" keyenv="${7:-DEEPSEEK_API_KEY}"
  local name="unh_$(vtag "$ds" "$view")$suffix"
  "$PY" "$EVAL_ROOT/common/answer.py" --dataset "$ds" --system unh --subset "$sub" --prompt "$prompt" \
      --model "$model" --base-url "$base" --api-key-env "$keyenv" \
      --retrieval "$out/$(ctx_of "$ds" "$view")" --out "$out/answers_${name}_$ds.json" >> "$LOG" 2>&1 || return 1
  "$PY" "$EVAL_ROOT/common/score.py" --dataset "$ds" --system unh --subset "$sub" --llm-judge \
      --judge-model "$model" --judge-base-url "$base" --judge-api-key-env "$keyenv" \
      --predictions "$out/answers_${name}_$ds.json" --out "$out/scores_${name}_$ds.json" >> "$LOG" 2>&1 || return 1
  say "$ds $view ($prompt): $("$PY" -c "import json;print(json.load(open('$out/scores_${name}_$ds.json'))['metrics_percent'])")"
}

done_pairs=()
for ds in ${DATASETS:-hotpotqa musique}; do
  sub="$REPO_ROOT/evaluation_data/$ds/subsets/${ds}_250_seed0.json"
  views="${VIEWS:-full hypergraph}"
  say "phase 1: $ds: retrieve views ${views// /,} (one graph embedding)"
  (cd "$UNH_DIR" && CUDA_VISIBLE_DEVICES="$(gpu_of "$ds")" "$PY" main.py --dataset "$ds" --subset "$sub" \
      --out-dir "$UNH_DIR/outputs/${ds}_250_seed0" --step retrieve --view "${views// /,}" "${ARGS[@]}" >> "$LOG" 2>&1) \
    || say "phase 1: $ds: retrieval reported failures (views without a complete contexts file are skipped)"
  for view in $views; do
    if ! "$PY" -c "import json,sys; r=json.load(open(sys.argv[1])); sys.exit(0 if len(r)==250 and all(x['status']=='ok' for x in r) else 1)" \
         "$UNH_DIR/outputs/${ds}_250_seed0/$(ctx_of "$ds" "$view")" 2>/dev/null; then
      say "phase 1: $ds $view: contexts incomplete; skipping its answer/score (rerun resumes)"; continue
    fi
    say "phase 1: $ds $view: answer + score (shared HippoRAG 2 prompt, DeepSeek)"
    if answer_and_score "$ds" "$view" hipporag2 ""; then done_pairs+=("$ds:$view"); else say "phase 1: $ds $view: answer/score FAILED"; fi
  done
done

# ---- phase 2 ----
deadline="${DEADLINE:-21:00}"
finish_est=$(date -d "+${PHASE2_DEEPSEEK_MIN:-30} min" +%s); deadline_s=$(date -d "today $deadline" +%s)
if [ "${#done_pairs[@]}" -eq 0 ]; then
  say "phase 2 skipped: no phase-1 evaluation completed"
elif [ "$finish_est" -lt "$deadline_s" ] && [ "${PHASE2_BACKEND:-auto}" != qwen ]; then
  say "phase 2 on DeepSeek (estimated end $(date -d @"$finish_est" +%H:%M) < $deadline)"
  for pair in "${done_pairs[@]}"; do
    ds=${pair%%:*}; view=${pair##*:}
    say "phase 2: $ds $view: answer + score (UNH's own prompt, DeepSeek)"
    answer_and_score "$ds" "$view" unh_short "_unhprompt" || say "phase 2: $ds $view: answer/score FAILED"
  done
else
  say "phase 2 would end after $deadline on DeepSeek (estimate $(date -d @"$finish_est" +%H:%M)): using Qwen3-30B-A3B locally"
  QGPU=""
  for g in $(nvidia-smi --query-gpu=index --format=csv,noheader); do
    used=$(nvidia-smi -i "$g" --query-gpu=memory.used --format=csv,noheader,nounits)
    procs=$(nvidia-smi -i "$g" --query-compute-apps=pid --format=csv,noheader | grep -c .)
    if [ "$used" -lt 2000 ] && [ "$procs" -eq 0 ]; then QGPU=$g; break; fi
  done
  QPORT="${QWEN_PORT:-8005}"
  if [ -z "$QGPU" ]; then
    say "phase 2 skipped: no free GPU for the Qwen server"
  elif ss -ltn | grep -q ":$QPORT "; then
    say "phase 2 skipped: port $QPORT already in use (set QWEN_PORT)"
  else
    QMODEL="Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
    say "starting vLLM $QMODEL on GPU $QGPU port $QPORT"
    CUDA_VISIBLE_DEVICES=$QGPU setsid nohup "$HOME/miniconda3/envs/unh_eval/bin/vllm" serve "$QMODEL" --port "$QPORT" \
        --max-model-len 32768 --gpu-memory-utilization 0.85 > "$EVAL_ROOT/run_logs/unh_qwen_vllm.log" 2>&1 < /dev/null &
    QPID=$!
    for _ in $(seq 1 180); do curl -sf "http://localhost:$QPORT/v1/models" >/dev/null && break; sleep 10; done
    if ! curl -sf "http://localhost:$QPORT/v1/models" >/dev/null; then
      say "phase 2 skipped: Qwen server did not come up (see run_logs/unh_qwen_vllm.log)"
    else
      export QWEN_API_KEY=EMPTY LLM_DISABLE_THINKING=0   # vLLM: no DeepSeek 'thinking' field; Instruct-2507 does not think
      for pair in "${done_pairs[@]}"; do
        ds=${pair%%:*}; view=${pair##*:}
        say "phase 2: $ds $view: answer + score (UNH's own prompt, Qwen)"
        answer_and_score "$ds" "$view" unh_short "_unhprompt_qwen" "$QMODEL" "http://localhost:$QPORT/v1" QWEN_API_KEY \
          || say "phase 2: $ds $view (unh prompt, Qwen): FAILED"
        say "phase 2: $ds $view: answer + score (shared HippoRAG 2 prompt, Qwen)"
        answer_and_score "$ds" "$view" hipporag2 "_qwen" "$QMODEL" "http://localhost:$QPORT/v1" QWEN_API_KEY \
          || say "phase 2: $ds $view (shared prompt, Qwen): FAILED"
      done
      export LLM_DISABLE_THINKING=1
    fi
    kill -- "-$QPID" 2>/dev/null || kill "$QPID" 2>/dev/null; say "stopped the Qwen server"
  fi
fi

"$PY" "$EVAL_ROOT/common/collect_results.py" --subset "$REPO_ROOT"/evaluation_data/{hotpotqa,musique}/subsets/*_250_seed0.json >> "$LOG" 2>&1
say "all done"
