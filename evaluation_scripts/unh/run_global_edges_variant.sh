#!/usr/bin/env bash
# Rebuild ONLY the global cross-chunk fact->fact edges of an existing UNH run, then evaluate it against the source run.
#   1. fork_merged.sh SRC -> DST: copies the merged checkpoint (after extraction, same-chunk fact->fact edges, entity
#      dedup and fact dedup; before global edges), so DST keeps SRC's facts, entities and same-chunk edges exactly
#   2. construct DST: global edges with --edge-mode clustered_fast and the global edge prompt $GLOBAL_PROMPT
#      (default event = BulkGlobalFactEdgesEvent, the compact RED / Hong et al. prompt); clustered round 2 only and
#      --clustered-max-facts-per-prompt as in SRC; $CONCURRENCY LLM calls at once
#   3. evaluate SRC and DST with the same PPR retriever (t2000_f40_dmp06, every view), the shared reader
#      (common/answer.py, hipporag2 prompt) and common/score.py (EM / F1 / Contain-Acc / LLM-Acc)
# Usage: bash evaluation_scripts/unh/run_global_edges_variant.sh [dataset] [src_run] [dst_run]
#   defaults: musique, outputs/musique_250_seed0, outputs/musique_250_seed0_fast_event
# Env: CONCURRENCY (default 1200), GLOBAL_PROMPT (default event), MIN_CLUSTER_SIZE (clustered_fast min cluster size;
#      unset = off), UNH_GPU (default 1), VIEWS (default "full
#      hypergraph discourse"), RETRIEVAL_WORKERS (default 50), SKIP_SRC_EVAL=1 (do not evaluate SRC)
set -euo pipefail
cd "$(dirname "$0")/../.."
REPO_ROOT="$PWD"
if [ -z "${DEEPSEEK_API_KEY:-}" ]; then
  export DEEPSEEK_API_KEY="$(grep -E '^\s*DEEPSEEK_API_KEY\s*=' .env | head -1 | cut -d= -f2- | tr -d "\"' ")"
fi
source evaluation_scripts/common/env.sh
export CUDA_VISIBLE_DEVICES="${UNH_GPU:-1}"
PYTHON="${PYTHON:-/home/taluka/miniconda3/envs/unh_eval/bin/python}"
DS="${1:-musique}"
SRC="$EVAL_ROOT/unh/${2:-outputs/${DS}_250_seed0}"
DST="$EVAL_ROOT/unh/${3:-outputs/${DS}_250_seed0_fast_${GLOBAL_PROMPT:-event}}"
SUBSET="$REPO_ROOT/evaluation_data/$DS/subsets/${DS}_250_seed0.json"
CONC="${CONCURRENCY:-1200}"
ts() { date '+%F %T'; }

# the source run's global-edge settings that DST keeps (only the clustering mode and the prompt change)
read -r MAXF R3 <<< "$("$PYTHON" -c "
import json; a = json.load(open('$SRC/construction_config.json'))['args']
print(a['clustered_max_facts_per_prompt'], int(a['clustered_round3']))")"
R3FLAG=--no-round3; [ "$R3" = 1 ] && R3FLAG=--round3

[ -d "$DST" ] || bash "$EVAL_ROOT/unh/fork_merged.sh" "$SRC" "$DST"
echo "[$(ts)] construct $DST: clustered_fast, global prompt ${GLOBAL_PROMPT:-event}, max facts/prompt $MAXF, concurrency $CONC"
"$PYTHON" "$EVAL_ROOT/unh/main.py" --dataset "$DS" --subset "$SUBSET" --corpus subset --out-dir "$DST" --step construct \
  --edge-mode clustered_fast --global-fact-edge-prompt "${GLOBAL_PROMPT:-event}" --clustered-max-facts-per-prompt "$MAXF" \
  "$R3FLAG" --concurrency "$CONC" ${MIN_CLUSTER_SIZE:+--fast-cluster-min-size "$MIN_CLUSTER_SIZE"}

PPR_ARGS='{"top_k_seeds": 30, "max_depth": 4, "beam_width": 6, "final_top_k": 80, "context_tokens": 2000, "force_top_facts": 40, "damping": 0.6}'
VIEWS="${VIEWS:-full hypergraph discourse}"
RUNS=("$DST"); [ "${SKIP_SRC_EVAL:-0}" = 1 ] || RUNS+=("$SRC")
for OUT in "${RUNS[@]}"; do
  echo "[$(ts)] retrieve $OUT (PPR t2000_f40_dmp06, tag ppr)"
  "$PYTHON" "$EVAL_ROOT/unh/main.py" --dataset "$DS" --subset "$SUBSET" --corpus subset --out-dir "$OUT" --step retrieve \
    --view "${VIEWS// /,}" --retriever ppr --ppr-args "$PPR_ARGS" --workers "${RETRIEVAL_WORKERS:-50}" --tag ppr
  for view in $VIEWS; do
    if [ "$view" = full ]; then ctx="$OUT/contexts_ppr.json"; tag=unh_ppr; else ctx="$OUT/contexts_${view}_ppr.json"; tag="unh_ppr_$view"; fi
    "$PYTHON" "$EVAL_ROOT/common/answer.py" --dataset "$DS" --system unh --subset "$SUBSET" --prompt hipporag2 \
      --retrieval "$ctx" --out "$OUT/answers_${tag}_$DS.json"
    "$PYTHON" "$EVAL_ROOT/common/score.py" --dataset "$DS" --system unh --subset "$SUBSET" \
      --predictions "$OUT/answers_${tag}_$DS.json" --out "$OUT/scores_${tag}_$DS.json" \
      --llm-judge --judge-model "$LLM_MODEL" --judge-base-url "$LLM_BASE_URL"
  done
done

"$PYTHON" - "$SRC" "$DST" "$DS" "$VIEWS" <<'EOF'
import json, sys
from pathlib import Path
src, dst, ds, views = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4].split()
print(f"{'run':40} {'view':11} {'EM':>5} {'F1':>5} {'Contain':>7} {'LLM-Acc':>7}")
for run in (src, dst):
    for v in views:
        f = run / f"scores_unh_ppr{'' if v == 'full' else '_' + v}_{ds}.json"
        if not f.exists():
            continue
        pe = json.load(open(f))["per_example"]
        m = {k: 100 * sum(p[k] for p in pe) / len(pe) for k in ("EM", "F1", "Contain-Acc", "LLM-Acc") if k in pe[0]}
        print(f"{run.name:40} {v:11} {m.get('EM', 0):5.1f} {m.get('F1', 0):5.1f} {m.get('Contain-Acc', 0):7.1f} {m.get('LLM-Acc', 0):7.1f}")
EOF
echo "[$(ts)] done"
