# HyperGraphRAG: upstream provenance

- Source: https://github.com/LHRLAB/HyperGraphRAG @ d587cdf8c3fe2be7719557f845324cb3a321f5e2 (MIT, see LICENSE). Paper: Luo et al., "HyperGraphRAG: Retrieval-Augmented Generation via Hypergraph-Structured Knowledge Representation", NeurIPS 2025.
- Copied:
  - `hypergraphrag/` = upstream `evaluation/hypergraphrag/`, **unmodified**. This is the copy the paper's evaluation scripts import. It differs from the repo-root `hypergraphrag/` package in two ways: its `openai_*` defaults point at `https://vip.apiyi.com/v1` (unused here, because we pass our own LLM and embedding functions), and when keyword extraction finds no entities or no hyper-relations it falls back to the raw question instead of returning `fail_response`.
  - `upstream_scripts/` = upstream `evaluation/script_insert.py`, `script_hypergraphrag.py`, and `get_generation.py`, kept as reference. They are not run.
  - `requirements.txt`, `LICENSE`.
- Not the same as `<repo>/hypergraphrag/`, which is a copy of the upstream repo-root package. `main.py` asserts that it imports the copy in this folder.

## How `main.py` maps to upstream

| Step | Upstream | Here |
|---|---|---|
| 1. Index | `script_insert.py`: `HyperGraphRAG(embedding_func_max_async=32, llm_model_max_async=32)`, then `rag.insert(contexts)` with up to 10 retries 10 s apart | same settings and retries. Input = one document per corpus passage, `title\ntext`, the same as HippoRAG 2. |
| 2. Retrieve | `script_hypergraphrag.py`: `aquery(q, QueryParam(only_need_context=True))`, 32 questions in flight, constructor defaults | same; writes `outputs/<ds>/knowledge.json` |
| 3. Generate | `get_generation.py`: its own `<think>/<answer>` prompt on gpt-4o-mini | `common/answer.py --system hypergraphrag`, the answer step shared by every system: HippoRAG 2 `rag_qa_musique` prompt, temperature 0, max_tokens 2048, with `knowledge.json`'s context as one doc. `main.py --step qa-upstream` runs get_generation's prompt verbatim, for reference. |
| 4. Score | `get_score.py` (their own metrics) | `common/score.py --system hypergraphrag` |

Upstream defaults kept (all from the `HyperGraphRAG` dataclass and `QueryParam`):
- chunk_token_size 1200, overlap 100, tiktoken `gpt-4o-mini`
- entity_extract_max_gleaning 2; entity summaries capped at 500 tokens
- JsonKV / NanoVectorDB (cosine threshold 0.2) / NetworkX storage
- query mode `hybrid`, top_k 60, max_token_for_{text_unit,global_context,local_context} 4000
- LLM calls send no temperature and no max_tokens, except the upstream max_tokens=500 for summaries
- tenacity retry: 3 attempts on RateLimitError / APIConnectionError / Timeout

## Local changes (all in `main.py`, none in the package)

1. **Models.** The LLM is `LLM_MODEL` @ `LLM_BASE_URL` instead of gpt-4o-mini. Embeddings are `EMBEDDING_MODEL` @ `EMBEDDING_BASE_URL` instead of text-embedding-3-small. `embedding_dim` is read from the server (upstream hard-codes 1536). Upstream reads `llm_model_name` for nothing else.
2. **Thinking off.** `extra_body={"thinking": {"type": "disabled"}}` goes on every chat call unless `LLM_DISABLE_THINKING=0`. DeepSeek V4 / V4.1 think by default.
3. **LLM call cache** (`outputs/<ds>/llm_cache.sqlite`). Upstream caches nothing during indexing, so a crash loses every extraction call. Cached responses are keyed on the full request, and empty responses are never cached.
4. **Retry hygiene.**
   - Each insert retry builds a fresh `HyperGraphRAG`. Upstream's `limit_async_func_call` never frees a slot when a call raises, so reusing the instance can hang.
   - Extraction tasks left running by the failed attempt are cancelled first.
5. **QueryParam per question.** Upstream's `_build_query_context` mutates `param.mode`. Upstream also creates a new one per call, via the default argument.
6. **Retrieval status.** Each question in `knowledge.json` gets a status: `ok`, `fail_response` (no keywords were extracted; upstream then hands "Sorry, I'm not able to provide an answer…" to the reader as knowledge, and so does this harness), `no_context`, or `error`. The step exits non-zero if any question has `error`.

## QuALITY (`run_quality.sh`, 2026-10-01; see ../QUALITY_IMPLEMENTATION.md)
- `main.py --dataset quality` loads `quality/inputs/documents.json`, one entry per dev article. `step_index` inserts `"title\ntext"` per article, exactly as for passages.
- HyperGraphRAG chunks each document itself (upstream 1200/100, tiktoken gpt-4o-mini). The upstream package is unchanged.
- Questions (with their four options) go to `aquery(only_need_context=True)` as before. `common/answer.py --dataset quality` reads `knowledge`.

## Insurance contracts (`run_contracts.sh`, 2026-10-09; see ../CONTRACTS_EVAL_PLAN.md)
- One index per contract: the contract is one document (`documents.json`, chunked 1200/100 by HyperGraphRAG), read by the inputs-dir loader (`--dataset quality --quality-inputs <contract dir>`). Hybrid-mode context for every query. No code change.
- `[unh_v1]` `main.py`: the LLM client timeout reads `LLM_TIMEOUT` (default 300 s, unchanged). `common/env_qwen.sh` sets 1800 s, because on the local Qwen server a 6-7k-token extraction can exceed 300 s under load, and a timed-out call is retried from scratch.
