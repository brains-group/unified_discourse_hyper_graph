#!/usr/bin/env bash
# LogicRAG (official run.py / RAGEvaluator) on evaluation_data/<dataset>, SentenceTransformer embeddings.
# Its own final answer is kept for reference; the reported answer comes from the shared answer step
# (common/answer.py) reading the same info_summary LogicRAG's generate_answer() reads.
# Usage: bash evaluation_scripts/logicrag/run.sh hotpotqa|musique [limit, 0 = all]      (JUDGE=1 adds LLM-Acc)
set -euo pipefail
DATASET="${1:?usage: run.sh hotpotqa|musique [limit]}"
LIMIT="${2:-0}"
source "$(dirname "$0")/../common/env.sh"
cd "$EVAL_ROOT/logicrag"
export OPENAI_API_KEY="$DEEPSEEK_API_KEY" OPENAI_BASE_URL="$LLM_BASE_URL"
export LOGICRAG_LLM_MODEL="$LLM_MODEL" LOGICRAG_EMBEDDING_MODEL="$EMBEDDING_MODEL"
export LOGICRAG_CALLS_PER_MINUTE="${LOGICRAG_CALLS_PER_MINUTE:-600}"   # upstream default 20/min is an OpenAI-tier limit
PYTHON="${PYTHON:-python}"
OUT="logicrag_${DATASET}.json"
"$PYTHON" run.py --dataset "$REPO_ROOT/evaluation_data/$DATASET/$DATASET.json" \
  --corpus "$REPO_ROOT/evaluation_data/$DATASET/${DATASET}_corpus.json" \
  --limit "$LIMIT" --max-rounds 3 --top-k 5 --output "$OUT"
"$PYTHON" "$EVAL_ROOT/common/score.py" --dataset "$DATASET" --system logicrag \
  --predictions "result/$OUT" --out "result/scores_native_logicrag_$DATASET.json"
"$PYTHON" "$EVAL_ROOT/common/answer.py" --dataset "$DATASET" --system logicrag --retrieval "result/$OUT"
"$PYTHON" "$EVAL_ROOT/common/score.py" --dataset "$DATASET" --system logicrag \
  --predictions "result/answers_logicrag_$DATASET.json" ${JUDGE:+--llm-judge --judge-model "$LLM_MODEL" --judge-base-url "$LLM_BASE_URL"}
