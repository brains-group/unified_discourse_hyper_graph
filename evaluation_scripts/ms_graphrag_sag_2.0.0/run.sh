#!/usr/bin/env bash
# Microsoft GraphRAG (graphrag==2.0.0) via SAG-Benchmark's harness: prepare -> index -> local-search context
# -> HippoRAG 2 reader prompt (Step_3) -> common/score.py.
# One-time setup:  cd evaluation_scripts/ms_graphrag && uv sync --frozen   (creates .venv; required for the src/ patch layer)
# Usage: bash evaluation_scripts/ms_graphrag/run.sh hotpotqa|musique
set -euo pipefail
DATASET="${1:?usage: run.sh hotpotqa|musique}"
source "$(dirname "$0")/../common/env.sh"
cd "$EVAL_ROOT/ms_graphrag"
export LLM_API_KEY="$DEEPSEEK_API_KEY" EMBEDDING_API_KEY="${EMBEDDING_API_KEY:-EMPTY}"
export EMBEDDING_MODEL_NAME="$EMBEDDING_MODEL" EMBEDDING_DIM="${EMBEDDING_DIM:-2048}"
export GRAPHRAG_DISABLE_THINKING="${GRAPHRAG_DISABLE_THINKING:-0}"   # 1 only for vLLM-served Qwen3 (sends chat_template_kwargs)
PY=".venv/bin/python"
"$PY" reproduce/Step_0_prepare_dataset.py --dataset "$DATASET"
"$PY" reproduce/Step_1_build_index.py --dataset "$DATASET"
"$PY" reproduce/Step_2_retrieve.py --dataset "$DATASET"
"$PY" reproduce/Step_3_qa_eval.py --dataset "$DATASET"
"$PY" "$EVAL_ROOT/common/score.py" --dataset "$DATASET" --system ms_graphrag \
  --predictions "outputs/$DATASET/predictions.json" ${JUDGE:+--llm-judge --judge-model "$LLM_MODEL" --judge-base-url "$LLM_BASE_URL"}
