#!/usr/bin/env bash
# MemGraphRAG (official code/index.py + code/retrieval_dataset_test.py) on evaluation_data/<dataset>.
# NOTE: index.py reads the whole corpus .txt as ONE string and cuts 256-token windows with 32 overlap
# across passage boundaries (upstream behaviour, kept on purpose). <dataset>.txt is stored in question
# order, so a window can hold several gold passages of one question; by default this runner therefore
# indexes <dataset>_scrambled.txt (same passages, shuffled; evaluation_data/make_scrambled_txt.py).
# CORPUS_TXT=<path> overrides it (e.g. the original <dataset>.txt).
# Its own QA is kept for reference; the reported answers come from common/answer.py over the same top-5 docs.
# Upstream run_index.sh / run_retrieval_test.sh are kept unchanged as reference templates.
# Usage: bash evaluation_scripts/memgraphrag/run.sh hotpotqa|musique      (JUDGE=1 adds LLM-Acc)
set -euo pipefail
DATASET="${1:?usage: run.sh hotpotqa|musique}"
source "$(dirname "$0")/../common/env.sh"
cd "$EVAL_ROOT/memgraphrag"
export OPENAI_API_KEY="$DEEPSEEK_API_KEY"
PYTHON="${PYTHON:-python}"
SAVE_DIR="outputs/$DATASET"
EMB="VLLM/$EMBEDDING_MODEL"          # routes to src/embedding_model/VLLM.py
CORPUS_TXT="${CORPUS_TXT:-$REPO_ROOT/evaluation_data/$DATASET/${DATASET}_scrambled.txt}"
[ -f "$CORPUS_TXT" ] || { echo "missing $CORPUS_TXT: python evaluation_data/make_scrambled_txt.py --dataset $DATASET"; exit 1; }
echo "[memgraphrag] corpus: $CORPUS_TXT"
"$PYTHON" index.py --corpus "$CORPUS_TXT" --save-dir "$SAVE_DIR" \
  --llm-name "$LLM_MODEL" --llm-base-url "$LLM_BASE_URL" \
  --embedding-model "$EMB" --embedding-base-url "$EMBEDDING_BASE_URL/embeddings" --tokenizer "$EMBEDDING_MODEL" \
  --chunk-size 256 --chunk-overlap 32 --artifact-mode default --memory-max-workers 20 \
  --ontology-filter-mode percentile --ontology-low-percent 20 --conflict-min-confidence 0.85 \
  --synonymy-edge-topk 100 --synonymy-edge-threshold 0.8
# qa-top-k 5 = paper setting (upstream CLI default is 10); other flags = upstream run_retrieval_test.sh values
"$PYTHON" retrieval_dataset_test.py --questions "$REPO_ROOT/evaluation_data/$DATASET/$DATASET.json" \
  --save-dir "$SAVE_DIR" --output "$SAVE_DIR/qa_results.json" \
  --llm-name "$LLM_MODEL" --llm-base-url "$LLM_BASE_URL" \
  --embedding-model "$EMB" --embedding-base-url "$EMBEDDING_BASE_URL/embeddings" \
  --question-type all --sample-num 0 --qa-top-k 5 \
  --skip-fact-rerank true --fact-similarity-threshold 0.6 --use-raw-threshold-filter true
"$PYTHON" "$EVAL_ROOT/common/score.py" --dataset "$DATASET" --system memgraphrag \
  --predictions "$SAVE_DIR/qa_results.json" --out "$SAVE_DIR/scores_native_memgraphrag_$DATASET.json"
"$PYTHON" "$EVAL_ROOT/common/answer.py" --dataset "$DATASET" --system memgraphrag --retrieval "$SAVE_DIR/qa_results.json" --qa-top-k 5
"$PYTHON" "$EVAL_ROOT/common/score.py" --dataset "$DATASET" --system memgraphrag \
  --predictions "$SAVE_DIR/answers_memgraphrag_$DATASET.json" ${JUDGE:+--llm-judge --judge-model "$LLM_MODEL" --judge-base-url "$LLM_BASE_URL"}
