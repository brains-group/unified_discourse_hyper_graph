#!/usr/bin/env bash
# HippoRAG 2 on QuALITY (evaluation_scripts/QUALITY_EVAL_PLAN.md): index the 1200/100-token chunks HyGRAG's chunker
# makes from the 115 dev articles (HippoRAG 2 has no chunker), retrieve for every question (with its options),
# then the shared multiple-choice answer step (common/answer.py) and common/score_quality.py.
# Usage: bash evaluation_scripts/hipporag2/run_quality.sh
#   QUALITY_INPUTS  inputs dir (default evaluation_data/quality/inputs; build_quality_inputs.py)
#   QUALITY_TAG     output name (default quality -> outputs/quality; anything else -> outputs/<tag>_quality)
#   CONCURRENCY_MULTIPLIER  scales OpenIE workers (base 8, upstream default) and answer/score workers (base 16)
set -euo pipefail
source "$(dirname "$0")/../common/env.sh"
cd "$EVAL_ROOT/hipporag2"
export OPENAI_API_KEY="$DEEPSEEK_API_KEY"      # HippoRAG's OpenAI client reads OPENAI_API_KEY
export QUALITY_INPUTS="${QUALITY_INPUTS:-$REPO_ROOT/evaluation_data/quality/inputs}"
TAG="${QUALITY_TAG:-quality}"
PYTHON="${PYTHON:-python}"
if [ "$TAG" = quality ]; then SAVE=outputs; OUT=outputs/quality; else SAVE="outputs/$TAG"; OUT="outputs/${TAG}_quality"; fi
"$PYTHON" main.py --dataset quality --quality_inputs "$QUALITY_INPUTS" --save_dir "$SAVE" \
  --llm_base_url "$LLM_BASE_URL" --llm_name "$LLM_MODEL" \
  --embedding_provider vllm --embedding_name "$EMBEDDING_MODEL" \
  --embedding_base_url "$EMBEDDING_BASE_URL/embeddings" \
  --openie_max_workers "$(scale_concurrency 8)" \
  --force_index_from_scratch "${HIPPORAG_FORCE_INDEX:-false}"
"$PYTHON" "$EVAL_ROOT/common/answer.py" --dataset quality --system hipporag2 --retrieval "$OUT/predictions.json" --workers "$(scale_concurrency 16)"
"$PYTHON" "$EVAL_ROOT/common/score_quality.py" --system hipporag2 --predictions "$OUT/answers_hipporag2_quality.json" --workers "$(scale_concurrency 16)"
