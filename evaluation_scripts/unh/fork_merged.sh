#!/usr/bin/env bash
# Start a new UNH run directory from another run's merged checkpoint (after extraction + entity dedup + fact dedup,
# before the cross-chunk fact->fact edges), so a different edge clustering can be built without redoing extraction
# or dedup:
#   bash evaluation_scripts/unh/fork_merged.sh outputs/hotpotqa_full outputs/hotpotqa_full_fast
#   python main.py --dataset hotpotqa --corpus full --out-dir outputs/hotpotqa_full_fast --step construct \
#       --edge-mode clustered_fast [--fast-cluster-leiden-passes 2 ...]
# Copies data/corpus.jsonl and checkpoint/{chunk_manifest.json, extraction_progress.json, merged.graphml,
# merged_state.json}; NOT construction_config.json (the new run saves its own, so changing --edge-mode or the
# --fast-cluster-* flags is a fresh config there, not a wipe) and NOT checkpoint/rounds/ (edges are rebuilt).
# Construction flags other than the edge ones must match the source run, or the merged graph does not reflect them.
set -euo pipefail
SRC="${1:?usage: fork_merged.sh SRC_RUN_DIR DST_RUN_DIR}"
DST="${2:?usage: fork_merged.sh SRC_RUN_DIR DST_RUN_DIR}"
[ -f "$SRC/checkpoint/merged.graphml" ] || { echo "no $SRC/checkpoint/merged.graphml"; exit 1; }
[ -e "$DST" ] && { echo "$DST already exists"; exit 1; }
mkdir -p "$DST/checkpoint" "$DST/data"
cp "$SRC/data/corpus.jsonl" "$DST/data/"
for f in chunk_manifest.json extraction_progress.json merged.graphml merged_state.json; do
  cp "$SRC/checkpoint/$f" "$DST/checkpoint/"
done
echo "forked $SRC -> $DST (merged checkpoint; edges will be rebuilt)"
