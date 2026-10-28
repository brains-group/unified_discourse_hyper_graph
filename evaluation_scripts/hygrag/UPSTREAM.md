# HyGRAG: provenance and every local change

- Upstream: https://github.com/zjunet/HyGRAG @ fa1a57e ("release"), a fork of DIGIMON (JayLZhou/GraphRAG). Paper:
  *A Unified Framework for Context-Aware and Relation-Aware Graph RAG* (WWW'26). The README declares MIT; the repo
  ships no LICENSE file.
- `HyGRAG/` is a copy of every tracked file except DIGIMON's own datasets (`Data/<ds>/{Corpus,Question}.json`,
  94 MB) and the paper PDF. Method: `HKGraphTreeLSH` (graph `hk_graph_tree`, retriever `hk_tree`, flat search).
- Goal: HyGRAG runs as published. Only housekeeping is changed: which LLM and embedding it calls, keys, paths,
  concurrency, crash recovery, and checks that make its silent failures visible. Planning notes and the
  embedding-size tests: `EVAL_PLAN.md`.

## Pipeline (`run.sh <dataset>`)

1. `prepare_data.py`: `evaluation_data/<ds>/<ds>_corpus.json` → `data/<ds>/Corpus.json`, DIGIMON's own input
   format (JSONL, one document per row). **One row per passage**, `context = "title\ntext"`. HyGRAG chunks every
   document on its own (1200/100 tokens, never across documents), so each passage is one chunk. There is no big
   corpus file and no cross-passage windows; the scrambled `.txt` is not used or needed.
2. `main.py`:
   - Loads the corpus with HyGRAG's `RAGQueryDataset.get_corpus()` and builds the index with `GraphRAG.insert()`.
   - Runs the build checks (below).
   - For each question, saves `HKGraphTreeQuery._retrieve_relevant_contexts(question)`, the exact context HyGRAG's
     own `generation_qa` reads, to `outputs/<ds>/contexts.json`. Query time makes no LLM calls.
3. `common/answer.py --system hygrag`, with the context as one doc, then `common/score.py`.

## Changes inside `HyGRAG/` (tagged `[unh_v1]`)

1. `Core/Provider/OpenaiApi.py`:
   - `_cons_kwargs` sends `extra_body={"thinking": {"type": "disabled"}}` unless `LLM_DISABLE_THINKING=0`
     (DeepSeek V4.1 thinks by default).
   - `_achat_completion` checks the response cache first.
2. `Core/Provider/ResponseCache.py` (new): an sqlite cache keyed on the exact request (all kwargs except
   `timeout`). Only non-empty replies are stored; enabled by `HYGRAG_LLM_CACHE`. Upstream records nothing and
   saves the graph only at the end, so a crash would lose every call. A rerun now replays finished calls for free.
   - Side effect: `_simple_entity_extraction` sends the same NER prompt a second time per chunk, after the first
     NER pass has finished, so the second call always hits the cache (200 of 200 in the smoke test).
   - Upstream ran at temperature 0, where both calls return the same entities anyway.
3. `Core/Graph/HKGraphTree.py` `_named_entity_recognition`: entities are kept as text. A bare number is turned into
   its string (`1571` → `"1571"`) and other non-text values are dropped, the same rule as HippoRAG 2's fix.
   - Why: upstream `clean_str` passes non-strings through unchanged, so a numeric entity became an `int` node id and
     `_convert_to_homogeneous_graph` failed on `node_id.startswith`.
   - The build's `try/except` hid the crash and left no index (MuSiQue full run, 2026-10-01: 1 of 11,656 NER
     replies). The post-build checks stopped the run before querying.
4. `Option/Config2.yaml`: the user config HyGRAG's README says to fill in.
   - llm: `deepseek-flash` at `https://api.deepseek.com`, `max_concurrent` 500, temperature 1.0.
   - embedding: `openai` type pointed at the vLLM voyage-4-nano server, with **no `dimensions`** (the server rejects
     it; native 2048).
   - api_key: a non-empty placeholder, because `Config.default()` validates the key at import. `main.py` sets the
     real key from `DEEPSEEK_API_KEY` at runtime, so it is never written to disk.
   - The upstream values are kept in the file's header comment.

## Method config `config/HKGraphTreeLSH.yaml`

A copy of `HyGRAG/Option/Ours/HKGraphTreeLSH.yaml` with one line changed (`diff` shows only this):
- `cleora_dim: 2048` (upstream 1024 = the size of bge-m3, the paper's embedding model).
- HyGRAG truncates node embeddings to `cleora_dim` but not query embeddings. At 1024 with voyage-4-nano, FAISS
  would hold 1024-dim nodes and get 2048-dim queries; every search fails silently and returns nothing. This was
  verified by running HyGRAG's own methods (EVAL_PLAN.md §8).

## Settings `main.py` applies at runtime (housekeeping)

- `llm`: key, base URL, model from the env; `max_concurrent` = `HYGRAG_CONCURRENCY` (default 500).
- `llm.temperature` = `LLM_TEMPERATURE` or 1.0 (DeepSeek's default; HyGRAG always sends one).
- `embedding`: as in Config2 above.
- `working_dir` = `outputs/<ds>/work`.
- `graph.faiss_index_path` = `outputs/<ds>/work/faiss_index`. The YAML's `./faiss_index/quality` is one path shared
  by every dataset, so one run could load another dataset's indexes.

## Silent-failure checks (`main.py`)

HyGRAG wraps the whole graph build in `try/except` and logs; it replaces failed embeddings with random vectors,
failed or empty community summaries with placeholder text, and failed searches with a fallback retrieval.
`main.py` counts each of these log messages (`run_report.json`) and checks the built index:
- a hierarchy exists;
- the chunk, entity and community FAISS indexes are non-empty and the size of the embeddings;
- there is one chunk vector per chunk.

Any failed check or fatal message aborts before querying. Questions that hit a fallback or failed search are
flagged in `contexts.json`, and the run exits non-zero unless `--allow-fallback`. Unparseable NER/OpenIE JSON is
upstream behaviour (no entities for that chunk) and is only counted.

## Environment

- conda env `hygrag`, Python 3.10.16 (as in `experiment.yml`), `requirements.unh.txt`: upstream
  `requirements.txt` with three changes:
  - `faiss-gpu==1.7.2` → `faiss-cpu==1.7.4`: no wheel for this setup; same API; the indexes are CPU-side.
  - `umap==0.1.1` dropped: not on PyPI; `umap-learn==0.5.7` provides `umap`; only RAPTOR-style graphs import it.
  - `setuptools==75.8.0`, the version in `experiment.yml`; newer releases drop `pkg_resources`, which
    `llama_index.legacy` imports.
- HyGRAG's own logger also writes to `HyGRAG/Result/HyGRAG/experiment/Logs/` (gitignored).

## Upstream behaviour kept on purpose

- The chunk text never includes DIGIMON's separate `title` field; that is why the title is in `context`.
- Community summaries are capped at `max_tokens=300` while the prompt asks for "max 300 words".
- 2 HotpotQA passages exceed 1200 tokens and become 2 chunks each.
- llama_index's OpenAI embedding client replaces `"\n"` with `" "` before embedding (the upstream `openai`
  embedding path).
- Unknown YAML keys are ignored: `enable_ir_cot`, `max_communities_in_context`, and so on have no effect for
  `hk_tree`.
- Retrieval: 1 community + 4 chunks, 5 entities, top-5 relations (`top_k` 5).

## Smoke test (2026-09-30)

Setup: first 200 HotpotQA passages, 10 questions, concurrency 50, real DeepSeek + vLLM.
- Index in 82 s: 200 chunks, 2,707 entities, 64 communities over 2 levels. All three FAISS indexes are 2048-dim.
- Every fatal counter is 0 and there were 0 JSON parse failures. LLM cache: 464 calls made, 200 hits (the
  duplicate NER).
- 10/10 contexts (2.2–5.1k chars), none flagged, none empty.
- `answer.py` → `score.py`: EM 40, F1 45.9, LLM-Acc 50 (10 questions over a 200-passage corpus, so not
  meaningful).

## QuALITY (`run_quality.sh`, 2026-10-01; see ../QUALITY_IMPLEMENTATION.md)
- `prepare_data.py --dataset quality` writes DIGIMON's own QuALITY layout: `Corpus.json` with one row per article (the 115 dev articles, `context = "title\ntext"`), and `Question.json` whose `question` holds the four options. HyGRAG's own QuALITY copy is the same layout, but it holds 265 train + dev articles.
- HyGRAG chunks each article itself (1200/100) and builds one pooled graph. `main.py --dataset quality` saves `_retrieve_relevant_contexts(question with options)` for every question.
- The silent-failure checks are unchanged. No code under `HyGRAG/` is changed.
- Upstream note: `HyGRAG/Core/Utils/Evaluation.py` does not compile (line 505 has curly quotes), so upstream `main.py`, which imports it, cannot reach its QuALITY scorer as shipped. The extractor prompt is the same in `run_evaluation.py`. Our scorer copies it from the source text: `common/hygrag_close_extract_prompt.py`.

## Insurance contracts (`run_contracts.sh`, 2026-10-09; see ../CONTRACTS_EVAL_PLAN.md)
- One graph per contract: `Corpus.json` = the contract as one row (HyGRAG chunks it 1200/100), contexts for every query. Read by the inputs-dir loader (`--dataset quality`).
- `[unh_v1]` temperature for non-DeepSeek endpoints: HyGRAG always sent a temperature. `main.py` still sends 1.0 for DeepSeek (DeepSeek's default, as before) or `LLM_TEMPERATURE` when set; for any other endpoint (local Qwen) it sets `None`, and `HyGRAG/Core/Provider/OpenaiApi.py _cons_kwargs` now drops a `None` temperature, so the server's default applies, as for every other system.
