#!/usr/bin/env bash
# One UNH stage for one dataset (called by launch_unh.sh and run.sh; usable by hand).
#   bash evaluation_scripts/unh/pipeline.sh construct hotpotqa|musique
#       prepare (corpus) + construct + shorten (no-op for label-only graphs)
#   bash evaluation_scripts/unh/pipeline.sh evaluate hotpotqa|musique
#       for view in UNH_VIEWS: retrieve once -> every prompt in UNH_PROMPTS, concurrently, on that same context:
#       common/answer.py -> common/score.py (LLM judge unless JUDGE=0)
# Env: SUBSET      ids file (default evaluation_data/<ds>/subsets/<ds>_250_seed0.json); SUBSET=none = ALL questions
#                  (needs UNH_CORPUS=full; set automatically). Output dir: outputs/<subset name>, outputs/<ds>_fullcorpus
#                  (full corpus, subset questions) or outputs/<ds>_full (full corpus, all questions).
#      UNH_CORPUS  subset | full (default subset; forced to full with SUBSET=none)
#      UNH_VIEWS   default "full hypergraph discourse"
#      UNH_PROMPTS default "hipporag2 unh_short": the shared HippoRAG 2 prompt (rows unh / unh_hypergraph) and UNH's own
#                  August prompt (answer.py --prompt unh_short; rows unh_unhprompt / unh_hypergraph_unhprompt)
#      PYTHON, and common/env.sh's LLM / embedding settings.
#      Smoke tests: UNH_OUT_ROOT (default evaluation_scripts/unh/outputs) and UNH_MAIN_ARGS (extra main.py flags,
#      e.g. "--limit-docs 60 --limit-q 3"; also how to override retrieval / construction defaults).
set -euo pipefail
STAGE="${1:?usage: pipeline.sh construct|evaluate <dataset>}"
DATASET="${2:?usage: pipeline.sh construct|evaluate <dataset>}"
source "$(dirname "$0")/../common/env.sh"
cd "$EVAL_ROOT/unh"
PYTHON="${PYTHON:-python}"
SUBSET="${SUBSET:-$REPO_ROOT/evaluation_data/$DATASET/subsets/${DATASET}_250_seed0.json}"
CORPUS="${UNH_CORPUS:-subset}"
OUT_ROOT="${UNH_OUT_ROOT:-$EVAL_ROOT/unh/outputs}"
SUBARGS=()
if [ "$SUBSET" = none ]; then
  CORPUS=full; OUT="$OUT_ROOT/${DATASET}_full"
else
  SUBARGS=(--subset "$SUBSET")
  if [ "$CORPUS" = full ]; then OUT="$OUT_ROOT/${DATASET}_fullcorpus"; else OUT="$OUT_ROOT/$(basename "$SUBSET" .json)"; fi
fi
read -r -a EXTRA <<< "${UNH_MAIN_ARGS:-}"
MAIN=("$PYTHON" main.py --dataset "$DATASET" "${SUBARGS[@]}" --corpus "$CORPUS" --out-dir "$OUT" "${EXTRA[@]}")
ts() { date '+%F %T'; }

case "$STAGE" in
  construct)
    echo "[$(ts)] [unh $DATASET] construct -> $OUT"
    "${MAIN[@]}" --step prepare
    "${MAIN[@]}" --step construct
    "${MAIN[@]}" --step shorten
    echo "[$(ts)] [unh $DATASET] construct done" ;;
  evaluate)
    status=0
    for view in ${UNH_VIEWS:-full hypergraph discourse}; do
      vtag=""; [ "$view" = full ] || vtag="_$view"
      ctx="$OUT/contexts.json"; [ "$view" = full ] || ctx="$OUT/contexts_$view.json"
      echo "[$(ts)] [unh $DATASET] evaluate view=$view"
      if ! "${MAIN[@]}" --step retrieve --view "$view"; then
        echo "[$(ts)] [unh $DATASET] retrieval ($view) had failures; skipping its answer/score (rerun resumes)"; status=1; continue
      fi
      # the view's contexts were retrieved once (above); every prompt answers from that same contexts file, and the
      # prompts' answer + judge steps run concurrently (API-only, no GPU)
      pids=(); logs=()
      for prompt in ${UNH_PROMPTS:-hipporag2 unh_short}; do
        ptag=""; [ "$prompt" = hipporag2 ] || ptag="_unhprompt"
        tag="unh${vtag}${ptag}"
        (
          "$PYTHON" "$EVAL_ROOT/common/answer.py" --dataset "$DATASET" --system unh "${SUBARGS[@]}" --prompt "$prompt" \
            --retrieval "$ctx" --out "$OUT/answers_${tag}_$DATASET.json"
          judge=(); [ "${JUDGE:-1}" = 0 ] || judge=(--llm-judge --judge-model "$LLM_MODEL" --judge-base-url "$LLM_BASE_URL")
          "$PYTHON" "$EVAL_ROOT/common/score.py" --dataset "$DATASET" --system unh "${SUBARGS[@]}" \
            --predictions "$OUT/answers_${tag}_$DATASET.json" --out "$OUT/scores_${tag}_$DATASET.json" "${judge[@]}"
          echo "[$(ts)] [unh $DATASET] view=$view prompt=$prompt scored -> $OUT/scores_${tag}_$DATASET.json"
        ) > "$OUT/answer_score_${tag}.log" 2>&1 &
        pids+=("$!"); logs+=("$OUT/answer_score_${tag}.log")
      done
      for pid in "${pids[@]}"; do wait "$pid" || status=1; done
      cat "${logs[@]}"
    done
    exit $status ;;
  *) echo "unknown stage $STAGE"; exit 1 ;;
esac
