#!/usr/bin/env bash
# UNH (nkg) on the insurance contracts (CONTRACTS_EVAL_PLAN.md): one graph per contract (the contract is one document,
# chunked by UNH itself, 600/50), then every retrieval view for every query (QA questions, then facts verbatim) with the
# PPR retriever (no LLM call), the HotpotQA-selected setting t2000_f40_dmp06 used on HotpotQA / MuSiQue / QuALITY /
# G-Medical: full -> system "unh", hypergraph -> "unh_hypergraph", discourse -> "unh_discourse". Then the shared answer
# step + judges (common/contracts_score.py) for each view.
# DeepSeek by default (env_contracts.sh; CONTRACTS_LLM=qwen for local Qwen).
# Usage: bash evaluation_scripts/unh/run_contracts.sh <short|long|smoke>     (CONTRACTS="a b" for some only)
#   UNH_CONCURRENCY   construction --max-workers / --max-sub-workers per contract (default 64)
#   UNH_GPU           GPU for the embedding / cross-encoder models (default 1)
#   UNH_VIEWS         views (default "full hypergraph discourse")
#   UNH_PPR_ARGS      PPR retriever arguments (default t2000_f40_dmp06 below)
#   UNH_EDGE_MODE, UNH_FACT_PROMPT, UNH_FACT_EDGE_PROMPT, UNH_GLOBAL_FACT_EDGE_PROMPT, UNH_CLUSTER_MIN, UNH_CLUSTER_MAX
#                     construction knobs, as in run_gmedical.sh (defaults: unh/main.py's, i.e. the HotpotQA / MuSiQue
#                     construction: edge mode clustered, default prompts)
set -euo pipefail
SET="${1:?usage: run_contracts.sh <short|long|smoke>}"
source "$(dirname "$0")/../common/env_contracts.sh"
source "$EVAL_ROOT/common/contracts_lib.sh"
cd "$REPO_ROOT"
export UNH_CONCURRENCY="${UNH_CONCURRENCY:-64}"
export CUDA_VISIBLE_DEVICES="${UNH_GPU:-1}"
VIEWS="${UNH_VIEWS:-full hypergraph discourse}"
# same retrieval setting as the HotpotQA / MuSiQue 1,000-question runs, QuALITY and G-Medical (sweep_ppr.py t2000_f40_dmp06)
PPR_ARGS="${UNH_PPR_ARGS:-{\"top_k_seeds\": 30, \"max_depth\": 4, \"beam_width\": 6, \"final_top_k\": 80, \"context_tokens\": 2000, \"force_top_facts\": 40, \"damping\": 0.6\}}"
export VIEWS PPR_ARGS
run_one() {
  local MAIN=("$PYTHON" "$EVAL_ROOT/unh/main.py" --dataset contracts --inputs "$CONTRACTS_DATA/$SET/$1"
              --out-dir "$EVAL_ROOT/unh/outputs/contracts_$SET/$1")
  "${MAIN[@]}" --step construct --edge-mode "${UNH_EDGE_MODE:-clustered}" \
    --fact-prompt "${UNH_FACT_PROMPT:-default}" --fact-edge-prompt "${UNH_FACT_EDGE_PROMPT:-default}" \
    ${UNH_GLOBAL_FACT_EDGE_PROMPT:+--global-fact-edge-prompt "$UNH_GLOBAL_FACT_EDGE_PROMPT"} \
    ${UNH_CLUSTER_MIN:+--fast-cluster-min-size "$UNH_CLUSTER_MIN"} ${UNH_CLUSTER_MAX:+--clustered-max-facts-per-prompt "$UNH_CLUSTER_MAX"}
  "${MAIN[@]}" --step retrieve --view "${VIEWS// /,}" --retriever ppr --ppr-args "$PPR_ARGS"
}
for_each_contract unh "$SET" run_one
for view in $VIEWS; do
  case "$view" in full) sys=unh ;; hypergraph) sys=unh_hypergraph ;; discourse) sys=unh_discourse ;; esac
  score_contracts "$sys" "$SET"
done
