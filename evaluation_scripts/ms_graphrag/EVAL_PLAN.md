# Microsoft GraphRAG on HotpotQA / MuSiQue: how others ran it, and what we should do

Status: implemented 2026-09-30 as option A (graphrag 3.2.0 + proxy, temperature stripped, one scrambled `.txt`, no native answers). See `UPSTREAM.md`. The old harness is in `../ms_graphrag_sag_2.0.0/`.

## 1. What we have now

`evaluation_scripts/ms_graphrag/` is the SAG-Benchmark harness (Zleap-AI/SAG-Benchmark @ 942f498) around **`graphrag==2.0.0` (Feb 2025)**. The latest release is **3.2.0 (Sep 23 2026)**. Where this setup departs from stock GraphRAG:

| | Our SAG copy | Stock GraphRAG default |
|---|---|---|
| version | 2.0.0 (fnllm backend) | 3.2.0 (LiteLLM backend, monorepo rewrite) |
| input | CSV, one row per passage, from `<ds>_corpus.json` | whatever you feed it (text files by default) |
| chunking | 1000 / 100 tokens, grouped by passage | 1200 / 100 tokens |
| JSON handling | `model_supports_json: false` (loose parsing) | structured output (JSON schema) required in 3.x |
| thinking | a patch layer copies files into site-packages; its thinking flag is vLLM-only (`chat_template_kwargs`), so with DeepSeek **thinking stays on** | n/a |
| temperature | graphrag 2.0 sends `temperature: 0` on every call; Step 3 also sends 0 | goes against our rule of not sending a temperature |
| retrieval | local search `build_context` only (community level 2), whole context as one doc | local search |
| answer | our own Step 3 reader, not `common/answer.py` | graphrag writes its own answer |

Your suspicion is right: it is an older engine with a few non-default settings. It also breaks two of our cross-system rules (thinking off, no temperature sent).

## 2. How others ran MS GraphRAG on these benchmarks

| Source | Implementation | Mode | Settings | Answer step | Reported |
|---|---|---|---|---|---|
| **HippoRAG 2** (Gutiérrez et al. 2025, App. G.2 / Table 14). Same 1000-question HotpotQA/MuSiQue files as ours. | Microsoft code, "default hyperparameters and prompts" | **Local** | chunks 1200/100, report max 2000, max input 8000, max cluster 10, response type "Short phrase", top-k 60 | GraphRAG answers, then "the same QA prompt that HippoRAG 2 adopts… is applied to rephrase the original response" | F1 **68.6** HotpotQA / **38.5** MuSiQue (HippoRAG 2: 75.5 / 48.6), Llama-3.3-70B reader |
| **RAG vs. GraphRAG** (Han et al. 2025) | Microsoft code (Edge et al.) | Local and Global | 256-token chunks for all methods, GPT-4o-mini graph build | **Retrieval saved, then one unified generation script for every method** | HotpotQA F1 61.66 (local, Llama-3.1-8B). Global "often struggles on QA" |
| **DIGIMON** (Zhou et al., VLDB'25) | **Reimplemented** in their own framework, not Microsoft code | Local (global "not applicable" to specific QA) | 1200-token chunks, Llama-3-8B, BGE-M3 | own generator; Accuracy = gold contained in output | Local GraphRAG below vanilla RAG and HippoRAG on both datasets |
| **SAG-Benchmark** (what we vendored) | Microsoft `graphrag==2.0.0` | Local, `build_context` only | chunks 1000/100 per passage | SAG's own reader over the context | n/a |
| LinearRAG, MemGraphRAG, LogicRAG | no MS GraphRAG code released; LinearRAG's table has no MS GraphRAG row | | | | |
| GraphRAG-Bench (Xiang et al.) | MS-GraphRAG local and global | | | | Novel/Medical corpora only, no HotpotQA/MuSiQue |
| A-RAG repo | lists GraphRAG baseline scripts as "roadmap", not released | | | | |

Takeaways:
1. **Local search** is what everyone uses for these datasets. Global search is for corpus-level sense-making and does badly on factoid multi-hop QA.
2. **Stock defaults** are the norm (HippoRAG 2 says so outright). Nobody tunes GraphRAG for HotpotQA.
3. The answer step is always **made uniform** in one of two ways:
   - **(a) keep GraphRAG's context and run a shared reader over it** (RAG-vs-GraphRAG, SAG);
   - **(b) let GraphRAG answer, then rephrase that answer with the shared QA prompt** (HippoRAG 2).
4. Expect GraphRAG to land clearly below HippoRAG 2 on both datasets. If ours doesn't, look for a bug or leakage.

## 3. Your plan, checked

> install GraphRAG, completely default, feed the scrambled HotpotQA/MuSiQue text, let it build (caching is built in), then local search only, then our answer step + EM/F1 + judge.

This matches the literature and is the right design. Notes:

- **"Completely default" can't be literal.** A few settings have to be filled in, and none of them affect method behaviour:
  - which LLM (DeepSeek endpoint, model, key);
  - which embedding endpoint (our vLLM voyage-4-nano on :8001);
  - `vector_size: 2048` (voyage-4-nano; the default of 3072 is for OpenAI text-embedding-3-large);
  - input type `text` pointing at the scrambled file;
  - `concurrent_requests`, which only changes speed.

  Everything else stays at the `graphrag init` defaults: chunks 1200/100, entity types org/person/geo/event, gleanings 1, claims off, max cluster 10, local search `top_k_entities` 10, `text_unit_prop` 0.5, `community_prop` 0.15, `max_context_tokens` 12000, community level 2.
- **Caching:** yes. Every LLM call is cached under `cache/` (JSON by default; 3.2 adds SQLite). Re-running `graphrag index` after a crash replays cached calls at no cost.
- **Scrambled `.txt`:** a single file becomes one GraphRAG "document", so the default 1200-token chunks run across passage boundaries. That is exactly why the scrambled order matters. It is the same treatment MemGraphRAG gets, so gold passages are not co-located. (The alternative is one document per passage, which is leak-free without scrambling. It's also fine, but further from "just feed it the text".)
- **Local search only:** agreed.

## 4. The one real obstacle: DeepSeek + graphrag 3.x

graphrag 3.x requires **structured output (JSON schema)**. `create_community_reports` calls the model with `response_format=CommunityReportResponse` and has no fallback. **DeepSeek rejects `json_schema`** ("This response_format type is unavailable now"). This is open upstream: microsoft/graphrag#2200 (3.0.1, `deepseek/deepseek-chat` fails at community reports) and #2571. A LiteLLM fix (PR #41242, now #42245) is not in a release. Graph extraction, description summaries and local search are plain text, so community reports is the only step that breaks.

Options:

| | What | Pros | Cons |
|---|---|---|---|
| **A (recommended)** | **graphrag 3.2.0, untouched, behind a tiny local OpenAI-compatible proxy** (~60 lines) that forwards to DeepSeek and only: (1) adds `thinking: disabled`; (2) drops `temperature`, per our cross-system rule; (3) rewrites `response_format: json_schema` to `json_object`. GraphRAG's community-report prompt already spells out the JSON fields, and GraphRAG still validates the reply against its own pydantic model. | Latest engine. No patches to graphrag. The same three rules we apply to every other system. Every rewrite is visible in one file and logged. | Schema is not enforced server-side. A malformed report is handled by GraphRAG's own error path (that community gets no report). We count those and report the number. |
| B | graphrag **2.7.2** (last 2.x, Mar 2026) with `model_supports_json: false` (loose JSON), plus the same proxy for thinking/temperature | 2.x's JSON path already works with DeepSeek | Not the current version. Otherwise same result as A |
| C | Keep the current 2.0.0 SAG harness and fix the thinking flag | Least work | Oldest engine, non-default chunking and CSV input, SAG-specific patch layer. This is what you don't trust |

## 5. Proposed pipeline (option A)

New folder `evaluation_scripts/ms_graphrag/` (the SAG copy is archived to `ms_graphrag_sag_2.0.0/`, not deleted):

1. **Env:** a separate uv/venv with `graphrag==3.2.0` pinned. Python 3.11–3.13; graphrag's deps conflict with the other systems, so it gets its own env.
2. **Proxy:** `deepseek_proxy.py` on localhost, started by `run.sh`, with the three rewrites above.
3. **Workspace:** `graphrag init --root caches/<ds>`, then edit only the required fields from §3, keeping the diff against the `init` file in `UPSTREAM.md`. Copy `evaluation_data/<ds>/<ds>_scrambled.txt` into `caches/<ds>/input/`.
4. **Index:** `graphrag index --root caches/<ds>` (standard method, stock prompts). Crash → rerun; the cache makes it free.
5. **Query:** `query.py` builds the stock local-search engine once and, for each of the 1000 questions, calls `search()` with default parameters. It saves both:
   - `context_text`: the exact context GraphRAG's own answer call reads;
   - GraphRAG's native answer (default response type), as the reference row, like `scores_native_*` for the other systems.

   This adds one extra LLM call per question.
6. **Answer:** `common/answer.py --system ms_graphrag`, with the context as one doc. This is option (a) from §2, the same way HyperGraphRAG's `only_need_context` blob and LogicRAG's `info_summary` are read. `answer.py` gets a small `ms_graphrag` loader.
7. **Score:** `common/score.py` with EM/F1/LLM-judge as usual, plus a native score for reference. `collect_results.py` picks both up automatically.

Code this needs: `deepseek_proxy.py`, `query.py`, `run.sh`, `UPSTREAM.md`, and a ~5-line `ms_graphrag` branch in `common/answer.py`. Nothing inside graphrag is modified.

Before any full run: a smoke test on ~200 passages and 10 questions to confirm community reports parse through the proxy, the embeddings dimension matches, and caching resumes.

## 6. Decisions for you

1. **Version / DeepSeek route:** A (3.2.0 + proxy), B (2.7.2), or C (keep 2.0.0)?
2. **Temperature:** graphrag 2.x sends `temperature: 0` by default. Our cross-system rule is to send none. The proposal strips it in the proxy so GraphRAG matches the others. Keep that?
3. **Input:** one scrambled `.txt` (your plan, same as MemGraphRAG), or one document per passage?
4. **Native answers:** also save GraphRAG's own local-search answer as a reference row (≈1000 extra calls per dataset)?

## Sources

- HippoRAG 2, *From RAG to Memory*, App. G.2 / Table 14: https://arxiv.org/html/2502.14802v2
- Han et al., *RAG vs. GraphRAG: A Systematic Evaluation and Key Insights*: https://arxiv.org/html/2502.11371v3
- Zhou et al., *In-depth Analysis of Graph-based RAG in a Unified Framework* (DIGIMON): https://arxiv.org/html/2503.04338
- Xiang et al., *When to use Graphs in RAG* (GraphRAG-Bench): https://arxiv.org/html/2506.05690v2
- LinearRAG: https://arxiv.org/html/2510.10114
- SAG-Benchmark: https://github.com/Zleap-AI/SAG-Benchmark
- A-RAG: https://github.com/Ayanami0730/arag
- graphrag releases: https://pypi.org/project/graphrag/ ; defaults: https://github.com/microsoft/graphrag/blob/main/packages/graphrag/graphrag/config/defaults.py ; config: https://microsoft.github.io/graphrag/config/yaml/ , https://microsoft.github.io/graphrag/config/models/ ; local search: https://microsoft.github.io/graphrag/query/local_search/
- DeepSeek + community reports: https://github.com/microsoft/graphrag/issues/2200 , https://github.com/microsoft/graphrag/issues/2571 , https://github.com/BerriAI/litellm/pull/41242
