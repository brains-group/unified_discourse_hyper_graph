#!/usr/bin/env bash
# HippoRAG 2 on G-Medical (GraphRAG-Bench Medical as MemGraphRAG evaluates it; evaluation_scripts/GMEDICAL_EVAL_PLAN.md).
# Index + retrieval = this system's QuALITY code path (run_quality.sh), unchanged, pointed at the G-Medical inputs
# (evaluation_data/build_gmedical_inputs.py writes the same layout; medical.txt as one document, 2,062 questions), as the
# contracts harness does. Then the shared open-ended answer step (common/answer.py --dataset gmedical: HippoRAG 2 /
# MemGraphRAG one-shot CoT prompt) and common/score_gmedical.py (LLM-Acc, overall and per GraphRAG-Bench level).
# Usage: bash evaluation_scripts/hipporag2/run_gmedical.sh
#   GMEDICAL_INPUTS  inputs dir (default evaluation_data/gmedical/inputs; smoke: evaluation_data/gmedical/smoke/inputs)
#   GMEDICAL_TAG     output name (default gmedical) -> outputs/<tag>_quality (HippoRAG appends the dataset label)
#   CONCURRENCY_MULTIPLIER  as in run_quality.sh
set -euo pipefail
source "$(dirname "$0")/../common/env.sh"
cd "$EVAL_ROOT/hipporag2"
export OPENAI_API_KEY="$DEEPSEEK_API_KEY"      # HippoRAG's OpenAI client reads OPENAI_API_KEY
export GMEDICAL_INPUTS="${GMEDICAL_INPUTS:-$REPO_ROOT/evaluation_data/gmedical/inputs}"
export QUALITY_INPUTS="$GMEDICAL_INPUTS"      # the QuALITY code path reads this inputs dir
TAG="${GMEDICAL_TAG:-gmedical}"
PYTHON="${PYTHON:-python}"
SAVE="outputs/$TAG"; OUT="outputs/${TAG}_quality"
"$PYTHON" main.py --dataset quality --quality_inputs "$QUALITY_INPUTS" --save_dir "$SAVE" \
  --llm_base_url "$LLM_BASE_URL" --llm_name "$LLM_MODEL" \
  --embedding_provider vllm --embedding_name "$EMBEDDING_MODEL" \
  --embedding_base_url "$EMBEDDING_BASE_URL/embeddings" \
  --openie_max_workers "$(scale_concurrency 8)" \
  --force_index_from_scratch "${HIPPORAG_FORCE_INDEX:-false}"
"$PYTHON" "$EVAL_ROOT/common/answer.py" --dataset gmedical --system hipporag2 --retrieval "$OUT/predictions.json" --workers "$(scale_concurrency 16)"
"$PYTHON" "$EVAL_ROOT/common/score_gmedical.py" --system hipporag2 --predictions "$OUT/answers_hipporag2_gmedical.json" --workers "$(scale_concurrency 16)"
