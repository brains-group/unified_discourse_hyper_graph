#!/usr/bin/env bash
# UNH on QuALITY (QUALITY_EVAL_PLAN.md; pooling decision in QUALITY_POOLING_FAIRNESS.md):
#   construct  ONE pooled graph over every article of $QUALITY_INPUTS/documents.json (as every baseline), each article
#              chunked on its own by UNH's chunker (600/50, --input-mode per_document), UNH's construction defaults
#              (unh/main.py, as HotpotQA/MuSiQue) with --edge-mode clustered_fast (as MuSiQue)
#   retrieve   PPR retriever (no LLM), the HotpotQA-selected setting t2000_f40_dmp06, unchanged, for every view
#              (full -> system unh, hypergraph -> unh_hypergraph, discourse -> unh_discourse); query = question + options
#   answer     the shared multiple-choice answer step (common/answer.py --dataset quality), then common/score_quality.py
# Usage: bash evaluation_scripts/unh/run_quality.sh
#   QUALITY_INPUTS    inputs dir (default evaluation_data/quality/inputs; build_quality_inputs.py)
#   QUALITY_TAG       output name (default quality -> unh/outputs/quality; anything else -> unh/outputs/<tag>_quality)
#   QUALITY_ARTICLES  smoke tests: keep only the first N articles of QUALITY_INPUTS (and their questions); the reduced
#                     inputs are written to <out>/inputs and used for every step
#   UNH_VIEWS         views (default "full hypergraph discourse")
#   UNH_CONCURRENCY   construction --max-workers / --max-sub-workers (default 2300)
#   UNH_RETRIEVAL_WORKERS  questions retrieved in parallel (default 50)
#   UNH_GPU           GPU for the embedding / cross-encoder models (default 1)
#   UNH_FACT_PROMPT   construction fact-sentence prompt: default | detailed (ChunkFactsDetailed; default: default)
#   UNH_FACT_EDGE_PROMPT  construction fact->fact edge prompt (intra-chunk + global): default | general | event
#                     (Bulk*FactEdgesEvent, intra-chunk calls also see the chunk text; default: default)
#   UNH_GLOBAL_FACT_EDGE_PROMPT  global cross-chunk edge prompt only: default | general | event (BulkGlobalFactEdgesEvent;
#                     unset: UNH_FACT_EDGE_PROMPT)
set -euo pipefail
source "$(dirname "$0")/../common/env.sh"
cd "$REPO_ROOT"
export CUDA_VISIBLE_DEVICES="${UNH_GPU:-1}"
PYTHON="${PYTHON:-python}"
INPUTS="${QUALITY_INPUTS:-$REPO_ROOT/evaluation_data/quality/inputs}"
TAG="${QUALITY_TAG:-quality}"
if [ "$TAG" = quality ]; then OUT="$EVAL_ROOT/unh/outputs/quality"; else OUT="$EVAL_ROOT/unh/outputs/${TAG}_quality"; fi
mkdir -p "$OUT"

if [ -n "${QUALITY_ARTICLES:-}" ]; then  # smoke: the first N articles and only their questions
  "$PYTHON" - "$INPUTS" "$OUT/inputs" "$QUALITY_ARTICLES" <<'EOF'
import json, sys
from pathlib import Path
src, dst, n = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
docs = json.load(open(src / "documents.json"))[:n]
keep = {d["article_id"] for d in docs}
qs = [q for q in json.load(open(src / "questions.json")) if q["article_id"] in keep]
dst.mkdir(parents=True, exist_ok=True)
json.dump(docs, open(dst / "documents.json", "w"), ensure_ascii=False, indent=1)
json.dump(qs, open(dst / "questions.json", "w"), ensure_ascii=False, indent=1)
print(f"[run_quality] smoke inputs: {len(docs)} articles, {len(qs)} questions -> {dst}")
EOF
  INPUTS="$OUT/inputs"
fi

# same retrieval setting as the HotpotQA / MuSiQue 1,000-question runs (sweep_ppr.py variant t2000_f40_dmp06)
PPR_ARGS='{"top_k_seeds": 30, "max_depth": 4, "beam_width": 6, "final_top_k": 80, "context_tokens": 2000, "force_top_facts": 40, "damping": 0.6}'
MAIN=("$PYTHON" "$EVAL_ROOT/unh/main.py" --dataset quality --inputs "$INPUTS" --out-dir "$OUT")

"${MAIN[@]}" --step construct --edge-mode clustered_fast --concurrency "${UNH_CONCURRENCY:-2300}" \
  --fact-prompt "${UNH_FACT_PROMPT:-default}" --fact-edge-prompt "${UNH_FACT_EDGE_PROMPT:-default}" \
  ${UNH_GLOBAL_FACT_EDGE_PROMPT:+--global-fact-edge-prompt "$UNH_GLOBAL_FACT_EDGE_PROMPT"}
VIEWS="${UNH_VIEWS:-full hypergraph discourse}"
"${MAIN[@]}" --step retrieve --view "${VIEWS// /,}" --retriever ppr --ppr-args "$PPR_ARGS" \
  --workers "${UNH_RETRIEVAL_WORKERS:-50}"

for view in $VIEWS; do
  case "$view" in
    full) ctx="$OUT/contexts.json"; sys=unh ;;
    hypergraph) ctx="$OUT/contexts_hypergraph.json"; sys=unh_hypergraph ;;
    discourse) ctx="$OUT/contexts_discourse.json"; sys=unh_discourse ;;
  esac
  "$PYTHON" "$EVAL_ROOT/common/answer.py" --dataset quality --system unh --questions "$INPUTS/questions.json" \
    --retrieval "$ctx" --out "$OUT/answers_${sys}_quality.json" --workers "$(scale_concurrency 16)"
  "$PYTHON" "$EVAL_ROOT/common/score_quality.py" --system "$sys" --questions "$INPUTS/questions.json" \
    --predictions "$OUT/answers_${sys}_quality.json" --out "$OUT/scores_${sys}_quality.json" \
    --workers "$(scale_concurrency 16)"
done
