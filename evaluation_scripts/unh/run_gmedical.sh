#!/usr/bin/env bash
# UNH on G-Medical (GraphRAG-Bench Medical as MemGraphRAG evaluates it; evaluation_scripts/GMEDICAL_EVAL_PLAN.md).
# Same protocol as unh/run_quality.sh, on the G-Medical inputs (evaluation_data/build_gmedical_inputs.py):
#   construct  ONE graph over $GMEDICAL_INPUTS/documents.json (one document: the corpus verbatim, as every baseline),
#              chunked by UNH's chunker (600/50, unh/main.py --dataset quality code path),
#              UNH's construction defaults with --edge-mode clustered_fast
#   retrieve   PPR retriever (no LLM), the HotpotQA-selected setting t2000_f40_dmp06, unchanged, for every view
#              (full -> system unh, hypergraph -> unh_hypergraph, discourse -> unh_discourse)
#   answer     the shared open-ended answer step (common/answer.py --dataset gmedical), then common/score_gmedical.py
# Usage: bash evaluation_scripts/unh/run_gmedical.sh
#   GMEDICAL_INPUTS   inputs dir (default evaluation_data/gmedical/inputs; smoke: evaluation_data/gmedical/smoke/inputs)
#   GMEDICAL_TAG      output name (default gmedical -> unh/outputs/gmedical)
#   UNH_VIEWS         views (default "full hypergraph discourse")
#   UNH_CONCURRENCY   construction --max-workers / --max-sub-workers (default 2300)
#   UNH_RETRIEVAL_WORKERS  questions retrieved in parallel (default 50)
#   UNH_GPU           GPU for the embedding / cross-encoder models (default 1)
#   UNH_FACT_PROMPT, UNH_FACT_EDGE_PROMPT, UNH_GLOBAL_FACT_EDGE_PROMPT  construction prompts, as in run_quality.sh
#                     (default: the HotpotQA / MuSiQue defaults)
#   UNH_CLUSTER_MIN   clustered_fast: merge clusters below this many facts (--fast-cluster-min-size; default 0 = off)
#   UNH_CLUSTER_MAX   clustered_fast: max facts per cluster / edge prompt (--clustered-max-facts-per-prompt; default 50)
set -euo pipefail
source "$(dirname "$0")/../common/env.sh"
cd "$REPO_ROOT"
export CUDA_VISIBLE_DEVICES="${UNH_GPU:-1}"
PYTHON="${PYTHON:-python}"
export GMEDICAL_INPUTS="${GMEDICAL_INPUTS:-$REPO_ROOT/evaluation_data/gmedical/inputs}"
INPUTS="$GMEDICAL_INPUTS"
TAG="${GMEDICAL_TAG:-gmedical}"
OUT="$EVAL_ROOT/unh/outputs/$TAG"
mkdir -p "$OUT"

# same retrieval setting as the HotpotQA / MuSiQue 1,000-question runs and QuALITY (sweep_ppr.py variant t2000_f40_dmp06)
PPR_ARGS='{"top_k_seeds": 30, "max_depth": 4, "beam_width": 6, "final_top_k": 80, "context_tokens": 2000, "force_top_facts": 40, "damping": 0.6}'
# --dataset quality = the per-document inputs code path (documents.json, UNH chunks each document on its own)
MAIN=("$PYTHON" "$EVAL_ROOT/unh/main.py" --dataset quality --inputs "$INPUTS" --out-dir "$OUT")

"${MAIN[@]}" --step construct --edge-mode clustered_fast --concurrency "${UNH_CONCURRENCY:-2300}" \
  --fact-prompt "${UNH_FACT_PROMPT:-default}" --fact-edge-prompt "${UNH_FACT_EDGE_PROMPT:-default}" \
  ${UNH_GLOBAL_FACT_EDGE_PROMPT:+--global-fact-edge-prompt "$UNH_GLOBAL_FACT_EDGE_PROMPT"} \
  --fast-cluster-min-size "${UNH_CLUSTER_MIN:-0}" --clustered-max-facts-per-prompt "${UNH_CLUSTER_MAX:-50}"
VIEWS="${UNH_VIEWS:-full hypergraph discourse}"
"${MAIN[@]}" --step retrieve --view "${VIEWS// /,}" --retriever ppr --ppr-args "$PPR_ARGS" \
  --workers "${UNH_RETRIEVAL_WORKERS:-50}"

for view in $VIEWS; do
  case "$view" in
    full) ctx="$OUT/contexts.json"; sys=unh ;;
    hypergraph) ctx="$OUT/contexts_hypergraph.json"; sys=unh_hypergraph ;;
    discourse) ctx="$OUT/contexts_discourse.json"; sys=unh_discourse ;;
  esac
  "$PYTHON" "$EVAL_ROOT/common/answer.py" --dataset gmedical --system unh --questions "$INPUTS/questions.json" \
    --retrieval "$ctx" --out "$OUT/answers_${sys}_gmedical.json" --workers "$(scale_concurrency 16)"
  "$PYTHON" "$EVAL_ROOT/common/score_gmedical.py" --system "$sys" --questions "$INPUTS/questions.json" \
    --predictions "$OUT/answers_${sys}_gmedical.json" --out "$OUT/scores_${sys}_gmedical.json" \
    --workers "$(scale_concurrency 16)"
done
