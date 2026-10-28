# Shared model settings for every evaluation_scripts/* runner. Source it: `source evaluation_scripts/common/env.sh`
# Override any value by exporting it before sourcing.
: "${DEEPSEEK_API_KEY:?export DEEPSEEK_API_KEY first}"
# DeepSeek V4.1 Flash; "deepseek-v4-flash" is temporarily routed to the same model (api-docs.deepseek.com/updates)
export LLM_MODEL="${LLM_MODEL:-deepseek-flash}"
# DeepSeek V4/V4.1 think by default; every client here sends thinking: disabled unless this is 0
export LLM_DISABLE_THINKING="${LLM_DISABLE_THINKING:-1}"
# Temperature: empty = no client sends one (DeepSeek default, 1.0). Set a number to send it on every call.
export LLM_TEMPERATURE="${LLM_TEMPERATURE-}"
export LLM_BASE_URL="${LLM_BASE_URL:-https://api.deepseek.com}"
# Embeddings: an OpenAI-compatible server hosting voyageai/voyage-4-nano under that exact model id.
# vLLM (0.19, unh_eval env) - the architecture override is required; vLLM picks up MEAN pooling from the
# sentence-transformers config by itself (checked 2026-09-30: cosine vs SentenceTransformer mean 0.9997):
#   CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server --model voyageai/voyage-4-nano --port 8001 \
#     --gpu-memory-utilization 0.4 --max-model-len 32000 --trust-remote-code --runner pooling --convert embed \
#     --enforce-eager --hf-overrides '{"architectures": ["VoyageQwen3BidirectionalEmbedModel"]}'
# Without --hf-overrides, vLLM 0.19 fails at load ("no parameter named 'linear'"). Fallback, serial and slower:
#   python evaluation_scripts/common/st_embedding_server.py --model voyageai/voyage-4-nano --port 8001
export EMBEDDING_MODEL="${EMBEDDING_MODEL:-voyageai/voyage-4-nano}"
export EMBEDDING_BASE_URL="${EMBEDDING_BASE_URL:-http://localhost:8001/v1}"
export EVAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export REPO_ROOT="$(cd "$EVAL_ROOT/.." && pwd)"
# [unh_v1] Concurrency multiplier for the run_quality.sh scripts: every LLM concurrency setting a script passes
# (HyGRAG max_concurrent, HyperGraphRAG max_async, HippoRAG 2 OpenIE workers, MemGraphRAG memory / QA workers,
# MS GraphRAG concurrent_requests, answer / score workers) is its base value x CONCURRENCY_MULTIPLIER (default 1,
# fractions allowed, rounded, at least 1). The base is the per-system variable when set (e.g. HYGRAG_CONCURRENCY),
# else the default. Concurrency only changes speed, not what is sent.
export CONCURRENCY_MULTIPLIER="${CONCURRENCY_MULTIPLIER:-1}"
scale_concurrency() { awk -v b="$1" -v m="$CONCURRENCY_MULTIPLIER" 'BEGIN { v = int(b * m + 0.5); if (v < 1) v = 1; print v }'; }
