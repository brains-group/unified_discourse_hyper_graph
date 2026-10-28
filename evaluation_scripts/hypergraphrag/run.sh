#!/usr/bin/env bash
# HyperGraphRAG (upstream evaluation/hypergraphrag, unmodified) on evaluation_data/<dataset>:
# index (one doc per passage) -> hybrid-mode context retrieval -> shared answer step (common/answer.py) -> score.
# Every LLM call is cached in outputs/<dataset>/llm_cache.sqlite, so rerunning after a crash is cheap.
# Usage: bash evaluation_scripts/hypergraphrag/run.sh hotpotqa|musique [limit, 0 = all questions]
#   READER=upstream  also answer with upstream's <think>/<answer> prompt (predictions_upstream_reader.json)
#   JUDGE=1          add LLM-Acc
set -euo pipefail
DATASET="${1:?usage: run.sh hotpotqa|musique [limit]}"
LIMIT="${2:-0}"
source "$(dirname "$0")/../common/env.sh"
cd "$EVAL_ROOT/hypergraphrag"
PYTHON="${PYTHON:-python}"
OUT="outputs/$DATASET"
"$PYTHON" main.py --dataset "$DATASET" --step all --limit "$LIMIT"
"$PYTHON" "$EVAL_ROOT/common/answer.py" --dataset "$DATASET" --system hypergraphrag --retrieval "$OUT/knowledge.json"
"$PYTHON" "$EVAL_ROOT/common/score.py" --dataset "$DATASET" --system hypergraphrag \
  --predictions "$OUT/answers_hypergraphrag_$DATASET.json" ${JUDGE:+--llm-judge --judge-model "$LLM_MODEL" --judge-base-url "$LLM_BASE_URL"}
if [ "${READER:-}" = "upstream" ]; then
  "$PYTHON" main.py --dataset "$DATASET" --step qa-upstream --limit "$LIMIT"
  "$PYTHON" "$EVAL_ROOT/common/score.py" --dataset "$DATASET" --system hypergraphrag \
    --predictions "$OUT/predictions_upstream_reader.json" --out "$OUT/scores_upstream_reader_hypergraphrag_$DATASET.json"
fi
