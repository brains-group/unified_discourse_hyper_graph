#!/usr/bin/env bash
# HippoRAG 2 (official main.py) on evaluation_data/<dataset>: index + retrieval (+ its own QA, kept for reference),
# then the shared answer step (common/answer.py) and common/score.py.
# Usage: bash evaluation_scripts/hipporag2/run.sh hotpotqa|musique      (JUDGE=1 adds LLM-Acc)
set -euo pipefail
DATASET="${1:?usage: run.sh hotpotqa|musique}"
source "$(dirname "$0")/../common/env.sh"
cd "$EVAL_ROOT/hipporag2"
export OPENAI_API_KEY="$DEEPSEEK_API_KEY"      # HippoRAG's OpenAI client reads OPENAI_API_KEY
PYTHON="${PYTHON:-python}"
OUT="outputs/$DATASET"
"$PYTHON" main.py --dataset "$DATASET" \
  --llm_base_url "$LLM_BASE_URL" --llm_name "$LLM_MODEL" \
  --embedding_provider vllm --embedding_name "$EMBEDDING_MODEL" \
  --embedding_base_url "$EMBEDDING_BASE_URL/embeddings" \
  --force_index_from_scratch "${HIPPORAG_FORCE_INDEX:-false}"   # true = rebuild the graph, reusing saved embeddings/OpenIE
# native answers (HippoRAG.qa, same prompt, max_new_tokens 2048): EM/F1 only, for reference
"$PYTHON" "$EVAL_ROOT/common/score.py" --dataset "$DATASET" --system hipporag2 \
  --predictions "$OUT/predictions.json" --out "$OUT/scores_native_hipporag2_$DATASET.json"
# shared answer step over retrieved_docs[:5] -> the reported numbers
"$PYTHON" "$EVAL_ROOT/common/answer.py" --dataset "$DATASET" --system hipporag2 --retrieval "$OUT/predictions.json"
"$PYTHON" "$EVAL_ROOT/common/score.py" --dataset "$DATASET" --system hipporag2 \
  --predictions "$OUT/answers_hipporag2_$DATASET.json" ${JUDGE:+--llm-judge --judge-model "$LLM_MODEL" --judge-base-url "$LLM_BASE_URL"}
