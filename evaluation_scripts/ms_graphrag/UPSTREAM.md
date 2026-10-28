# Microsoft GraphRAG 3.2.0: what runs and what differs from stock

- Upstream: https://github.com/microsoft/graphrag, PyPI `graphrag==3.2.0` (released 2026-09-23, the latest). MIT.
- Integration: the released package, installed unmodified in its own conda env (`ms_graphrag`, Python 3.12; see
  `requirements.txt`). No graphrag source is vendored or patched.
- Replaces the SAG-Benchmark harness around `graphrag==2.0.0`, archived unchanged in `../ms_graphrag_sag_2.0.0/`
  (never run; see `EVAL_PLAN.md` for why it was replaced and how other papers ran GraphRAG).

## Pipeline (`run.sh <dataset>`)

1. **Workspace** `caches/<ds>/`: `graphrag init` writes the stock prompts; `settings.yaml` is copied in. The
   input folder holds one file, `evaluation_data/<ds>/<ds>_scrambled.txt`.
2. **Index**: `graphrag index --root caches/<ds>`, the standard method with stock prompts. Every LLM call is
   cached in `caches/<ds>/cache/`, so rerunning after a crash replays finished calls for free.
3. **Context** (`query.py`): the local-search engine is built with the same calls as `graphrag query --method local`,
   using community level 2 (the CLI default). For each question it runs `context_builder.build_context(...)`,
   the first half of `LocalSearch.search()`, and saves `context_chunks` to `outputs/<ds>/local_search_contexts.json`.
   That is the exact text graphrag's own answer prompt would contain. GraphRAG's own answer is not generated.
4. **Answer**: `common/answer.py --system ms_graphrag`, with the context as one document (shared prompt and settings).
5. **Score**: `common/score.py` computes EM/F1, plus LLM-Acc with `JUDGE=1`.

## Settings changed from `graphrag init` (`diff settings.init.yaml settings.yaml`)

| Setting | Stock | Here | Why |
|---|---|---|---|
| `completion_models.*.model` / `api_base` / `api_key` | gpt-4.1, OpenAI | `deepseek-flash` via `deepseek_proxy.py` | our LLM |
| `embedding_models.*.model` / `api_base` | text-embedding-3-large | `voyageai/voyage-4-nano` on the vLLM server (`EMBEDDING_BASE_URL`) | our embedding model |
| `vector_store.vector_size` | 3072 | 2048 | voyage-4-nano output size |
| `concurrent_requests` | 25 | `GRAPHRAG_CONCURRENCY` (run.sh default 1200) | throughput only |

Everything else is stock:
- chunks of 1200/100 tokens (o200k_base);
- entity types organization/person/geo/event, `max_gleanings` 1;
- description summaries of up to 500 tokens; claims off;
- Leiden clustering with `max_cluster_size` 10 and `use_lcc: true`;
- community reports with `max_length` 2000 and `max_input_length` 8000;
- local search with `top_k_entities` 10, `top_k_relationships` 10, `text_unit_prop` 0.5, `community_prop` 0.15
  and `max_context_tokens` 12000.

## DeepSeek proxy (`deepseek_proxy.py`)

run.sh starts a local proxy for indexing and stops it before the answer step. It forwards every chat request to
`https://api.deepseek.com` unchanged except for three rewrites, and counts each in `outputs/<ds>/proxy_stats.json`:

1. adds `thinking: {"type": "disabled"}` (DeepSeek V4.1 thinks by default), unless `LLM_DISABLE_THINKING=0`;
2. drops `temperature`, so DeepSeek uses its default, unless `LLM_TEMPERATURE` is set. graphrag 3.2 indexing sends
   no temperature; this only guards against it;
3. `response_format: json_schema` becomes `json_object`. graphrag 3.x asks for a JSON schema only in
   `create_community_reports`, and DeepSeek rejects `json_schema` with HTTP 400 (microsoft/graphrag#2200). The stock
   community-report prompt already lists the JSON fields, and graphrag still validates each reply against its own
   pydantic model.

The `Authorization` header passes through unchanged; the proxy holds no key.

## Client connection limit

LiteLLM's aiohttp client allows 1000 connections in total and 500 per host. All calls go to one host (the proxy), so
graphrag would stall at 500 in flight. run.sh sets `AIOHTTP_CONNECTOR_LIMIT` and `AIOHTTP_CONNECTOR_LIMIT_PER_HOST`
to `GRAPHRAG_CONCURRENCY + 100`. Load test (2026-09-30, mock upstream with 3 s replies, 1200 calls): peak 500 in
flight with the defaults, 1200 with run.sh's values.

## Known stock behaviour: entities outside the largest component

With `use_lcc: true`, only the largest connected component of the entity graph is clustered into communities.
Local search only uses entities that have a community (`read_indexer_entities` filters on community level), so
entities outside that component can never be retrieved. A question whose best-matching entities are all outside it
gets an empty context, and the reader then answers without context, as graphrag's own answer call would.
`query.py` prints `entities_searchable` next to the total and counts empty contexts. In the 200-passage smoke test,
805 of 1318 entities were searchable and 2 of 10 contexts were empty. A full corpus is more connected, so both
numbers should be better there. This is left at the stock default.

## Smoke test (2026-09-30)

Setup: the first 200 passages of `hotpotqa.txt`, 10 questions, concurrency 50.

- Index: 25 text units, 1318 entities, 1682 relationships, 193 communities and 193 community reports. 502 LLM calls,
  all HTTP 200; 200 of them were json_schema requests rewritten to json_object; failure rate 0; 3.5 minutes.
- Answer and score: all 10 answered, none truncated; EM 40, F1 59.0, LLM-Acc 60 (10 questions only).

## QuALITY (`run_quality.sh`, 2026-10-01; see ../QUALITY_IMPLEMENTATION.md)
- Same `settings.yaml` (stock + endpoints) and the same proxy, on port 18803.
- `input/` = one `<article_id>.txt` per dev article (`"title\ntext"`), synced from `quality/inputs/ms_graphrag_input/`. This is graphrag's native input: the stock text reader makes each file one document, and stock chunking cuts 1200/100 tokens per document.
- `query.py --dataset quality` reads questions with their four options from `quality/inputs/questions.json`. Everything else is as for HotpotQA/MuSiQue: local-search `build_context` text goes to `common/answer.py --dataset quality`.

## Insurance contracts (`run_contracts.sh`, 2026-10-09; see ../CONTRACTS_EVAL_PLAN.md)
- One workspace per contract (`caches/contracts_<set>/<contract>/`), `input/` = the contract as one `.txt`. `query.py --dataset quality --questions <contract queries>` builds the local-search contexts. Calls go through `deepseek_proxy.py` (one per contract, ports 18820+) to the local Qwen server; with `LLM_DISABLE_THINKING=0` it adds no `thinking` field, still drops any temperature and maps json_schema to json_object. Stock settings, concurrency 64. No code change.
