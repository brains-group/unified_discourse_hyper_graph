# Helpers for the <system>/run_contracts.sh scripts (insurance contracts, CONTRACTS_EVAL_PLAN.md). Source after env_contracts.sh.
#   CONTRACTS      space-separated contract names (default: every contract of the set)
#   CONTRACT_JOBS  contracts built / retrieved in parallel (default 5, env_contracts.sh)
CONTRACTS_DATA="$REPO_ROOT/evaluation_data/contracts"

contracts_of() {  # contracts_of <set>: the set's contracts (or $CONTRACTS)
  if [ -n "${CONTRACTS:-}" ]; then echo "$CONTRACTS"; return; fi
  "$PYTHON" -c "import json,sys; print(' '.join(json.load(open(sys.argv[1]))['contracts']))" "$CONTRACTS_DATA/$1/contracts.json"
}

# for_each_contract <system> <set> <fn>: runs `fn <contract> <index>` for every contract, CONTRACT_JOBS at a time, each
# in a subshell with `set -e`, output in run_logs/contracts/<system>_<set>_<contract>.log. Writes
# contracts_results/<set>/<system>/timing_<contract>.json (wall seconds of index + retrieval). Returns 1 if any failed.
for_each_contract() {
  local system=$1 set=$2 fn=$3 i=0 c failed=0
  local logdir="$EVAL_ROOT/run_logs/contracts" resdir="$EVAL_ROOT/contracts_results/$set/$system"
  mkdir -p "$logdir" "$resdir"
  local -A pids=()
  for c in $(contracts_of "$set"); do
    while [ "$(jobs -rp | wc -l)" -ge "${CONTRACT_JOBS:-5}" ]; do wait -n || true; done
    (
      set -euo pipefail
      t0=$(date +%s)
      "$fn" "$c" "$i"
      printf '{"contract": "%s", "index_and_retrieval_seconds": %s}\n' "$c" "$(( $(date +%s) - t0 ))" > "$resdir/timing_$c.json"
    ) > "$logdir/${system}_${set}_$c.log" 2>&1 &
    pids[$c]=$!
    echo "[$system/$set] $c started (log: run_logs/contracts/${system}_${set}_$c.log)"
    i=$((i + 1))
  done
  for c in "${!pids[@]}"; do
    if wait "${pids[$c]}"; then echo "[$system/$set] $c done"; else echo "[$system/$set] $c FAILED (see run_logs/contracts/${system}_${set}_$c.log)"; failed=1; fi
  done
  return $failed
}

score_contracts() {  # score_contracts <system> <set>: shared answer step + judges over the set's contracts
  "$PYTHON" "$EVAL_ROOT/common/contracts_score.py" --system "$1" --set "$2" ${CONTRACTS:+--contracts $CONTRACTS}
}
