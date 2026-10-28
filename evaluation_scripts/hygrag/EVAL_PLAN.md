# HyGRAG on HotpotQA / MuSiQue: how it works, how to wire it in, what can go wrong

Status: implemented and smoke-tested 2026-09-30; see `UPSTREAM.md` (the plan below is kept as written). The repo was cloned on 2026-09-24 at `unh_v1/HyGRAG` (zjunet/HyGRAG @ fa1a57e,
unmodified; `git status` clean). Everything below was checked against that code.

## 1. What HyGRAG is

- Paper: *A Unified Framework for Context-Aware and Relation-Aware Graph RAG* (WWW'26, ZJU).
- The repo **is** a fork of DIGIMON (JayLZhou/GraphRAG). HyGRAG adds a graph type `hk_graph_tree`
  (`Core/Graph/HKGraphTree.py`), a retriever (`Core/Retriever/HKGraphTreeRetriever.py`) and a query class
  (`Core/Query/HKGraphTreeQuery.py`). The DIGIMON framework ships inside the repo; there is nothing separate to install.
- Method config: `Option/Ours/HKGraphTreeLSH.yaml`. Models: `Option/Config2.yaml`. Entry point:
  `main.py -opt <yaml> -dataset_name <ds>`.

## 2. Simplified pipeline

**Ingestion is per passage, not the big text file.** DIGIMON reads `Data/<ds>/Corpus.json`: JSONL with one row per
document, `{"title", "context"}`. It chunks each document **separately** (1200/100 tokens, never across documents), so
for HotpotQA and MuSiQue each passage is one chunk. That makes the scrambled `.txt` unnecessary: no chunk can hold two
passages, so gold passages can never be co-located. We would convert `evaluation_data/<ds>/<ds>_corpus.json` into that
JSONL.

```
<ds>_corpus.json (one row per passage)  ->  Corpus.json (JSONL {title, context})
   |
   v  HyGRAG index (GraphRAG.insert)
   1. chunk each passage alone (1 passage = 1 chunk)
   2. per chunk, 3 LLM calls: NER -> OpenIE triples (HippoRAG-style), then the same NER prompt again
      (used for chunk-chunk links)
   3. hybrid graph: entity nodes + chunk nodes; edges = triples, entity->chunk, chunk<->chunk if >= 3 shared entities
   4. embed every node (entity = "name. description", chunk = its text)
   5. "Cleora": 2 rounds of neighbour averaging of those embeddings (drives clustering only)
   6. LSH clustering (16 hyperplanes, clusters of 5-50) -> one LLM summary per cluster (<= 300 tokens)
      -> embed summaries -> repeat, up to 4 levels
   7. FAISS HNSW indexes over the text embeddings: entities / chunks / communities -> saved to disk
   |
   v  per question (no LLM call; the query-entity extraction step is commented out in the code)
   8. embed the question -> FAISS flat search over all levels -> 1 community summary + 4 chunks + 5 entities
      + top-5 relations among them
   9. context string with sections: Community / Entities / Key Relationships / Documents
   |
   v  common/answer.py (context as one doc)  ->  common/score.py (EM / F1 / LLM judge)
```

Step 9 is a separate function, `HKGraphTreeQuery._retrieve_relevant_contexts(question) -> str`. That is the clean
plug-in point, the same role `only_need_context` plays for HyperGraphRAG and `build_context` for MS GraphRAG.
HyGRAG's own answer call (`generation_qa`, a long-form "analyze the evidence…" prompt) would simply not run.

## 3. Can it use DeepSeek, our embeddings, and the shared answer and score steps?

Yes to all three, but each needs a small change, and some upstream defaults would silently produce a wrong
benchmark if left as they are.

| Piece | Upstream | What we'd need |
|---|---|---|
| **LLM** | `OpenaiApi` (openai SDK) with a configurable `base_url` | Point it at DeepSeek. The client sends `temperature: 0.0` on every call (config default) and has no way to disable thinking, so it needs a small `[unh_v1]` patch in `_cons_kwargs` (thinking off; temperature per our rule), as in the other systems. It also sends `stop: ["[/INST]", "<<SYS>>"]` and `max_tokens` 4096, both harmless. |
| **Embeddings** | `hf` (llama_index HuggingFaceEmbedding, hard-coded to `cuda:1`, no `trust_remote_code`) or `openai` (llama_index `OpenAIEmbedding` 0.3.1) | Use our vLLM voyage-4-nano endpoint, identical to the other systems. llama_index's `OpenAIEmbedding` checks model names against OpenAI's list and very likely rejects `voyageai/voyage-4-nano`, so expect a small OpenAI-compatible adapter (to be confirmed in a smoke test). Don't send `dimensions`. |
| **Embedding size** | `cleora_dim: 1024` = the size of bge-m3, the paper's embedding model | **Must become 2048** (voyage-4-nano). The code truncates longer embeddings to `cleora_dim` for the node index, while question embeddings stay full length. At 1024, FAISS would get 1024-dim nodes and 2048-dim queries. The search fails, the error is caught, and it silently switches to `_fallback_retrieval` for every question. |
| **Answer step** | its own long-form prompt; paper metric = "gold is a substring of the output" | Skip it. Save the step-9 context and run `common/answer.py --system hygrag`, the same as HyperGraphRAG and MS GraphRAG. |
| **Scoring** | `Core/Utils/Evaluation.py`; `main.py` only scores the **first 500** questions | Skip it. Our own runner covers all 1000; then `common/score.py`. |
| **Concurrency** | one semaphore, `llm.max_concurrent` (20) | Set it (e.g. 1200). The openai SDK's connection pool defaults to 1000, so check actual throughput in a smoke test. |
| **Environment** | pinned old deps (torch 2.6, faiss-gpu 1.7.2, llama-index 0.12, openai 1.70) | Its own conda env, like MS GraphRAG. |

## 4. Risks to benchmark authenticity (must be handled)

HyGRAG's code swallows errors in several places. A run can "finish" and produce a broken or empty index, or
fallback retrieval for every question, with no crash. Each point below needs a guard or a count in our runner.

1. **The whole index build is inside `try/except` that only logs** (`HKGraphTree._build_graph`, then "✅ construction
   finished"). Any error in any step leaves a partial or empty graph, and querying carries on. *Guard:* after the
   build, check that the hierarchy and all three FAISS indexes exist and that node/community counts are sensible;
   otherwise abort.
2. **No LLM response cache and no checkpoint.** The graph is only saved at the very end, so a crash (or an error the
   `except` above hides) throws away every extraction call. *Fix:* an additive sqlite response cache in the LLM
   client, as we did for HyperGraphRAG, so a rerun replays finished calls for free.
3. **Embedding failures become random vectors** (`_initialize_text_embeddings`: `np.random.normal` on any exception,
   and for nodes with no text). *Guard:* count those warnings; abort if any happen for a non-empty node.
4. **Retrieval failures are silent.** `_retrieve_relevant_contexts` returns the string `"Context retrieval failed:
   …"`, and the flat search drops to `_fallback_retrieval` on problems. Both would reach the reader as "context".
   *Guard:* the runner flags those strings and fallback calls and reports counts; the full run should have 0.
5. **Embedding size mismatch** (section 3): set `cleora_dim` = 2048 and check that the FAISS index size equals the
   question-embedding size before querying.
6. **Unparseable NER/OpenIE JSON is treated as "no entities/triples"** (`prase_json_from_response` never raises).
   This is upstream behaviour and we keep it, but count it (HippoRAG 2 had 4–5 such chunks per dataset).
7. **Stale artifacts.** `graph.force: False` reuses whatever sits under `working_dir/<index_name>`. A smoke-test
   index could be picked up by the full run. *Guard:* separate working directories per dataset and per smoke test.
8. **Unknown YAML keys are silently ignored** (e.g. `enable_ir_cot`, `max_communities_in_context` do nothing for
   `hk_tree`). No action needed; document it so nobody thinks IR-CoT is on.
9. **Content dedup.** DIGIMON drops documents with identical text (2 of 9,811 HotpotQA passages share text under
   different titles). With the title in the content (decision 1) nothing is lost.

## 5. Decisions for you

1. **Title in the text?** DIGIMON keeps the title only as metadata, so the LLM never sees it (the chunk is the
   paragraph only). Every other system in our benchmark indexes `title` + text: HippoRAG 2 and HyperGraphRAG use
   `title\ntext`, and MemGraphRAG and MS GraphRAG get the `.txt` that includes titles.
   - (a) Authentic DIGIMON ingestion: `context` = text only.
   - (b) **Recommended:** `context` = `title\ntext`, the same input the other systems get. The method is unchanged;
     this is a data-format choice.
2. **Temperature.** Upstream sends `temperature: 0.0` (the paper ran Llama-3.1-8B greedy). Our cross-system rule is
   to send none. Strip it, as for the other four?
3. **Response cache.** Recommended, for crash recovery. Side effect: the second NER call per chunk (same prompt)
   becomes a cache hit and returns the same entities. That matches upstream's temperature-0 intent, where the two
   calls give nearly identical results, and saves about 1/3 of indexing calls. Without a cache, at DeepSeek's default
   temperature, the two NER calls would differ.
4. **Concurrency** for the build (e.g. 1200).

## 6. What implementing it would involve (later, after approval)

- `evaluation_scripts/hygrag/` with HyGRAG vendored unmodified plus `UPSTREAM.md`.
- `[unh_v1]` patches:
  - LLM client: thinking off, temperature rule, sqlite cache.
  - An embedding adapter for the vLLM endpoint.
- A config YAML: HKGraphTreeLSH values unchanged except `cleora_dim: 2048`, the models and `max_concurrent`.
- `prepare_data.py`: `<ds>_corpus.json` → `Corpus.json` JSONL; questions → `Question.json`.
- `main.py` runner:
  - build the index, then run the post-build checks from section 4;
  - for all 1000 questions, `_retrieve_relevant_contexts` → `contexts.json`, with failure counters;
  - then `common/answer.py` and `common/score.py`.
- A smoke test (about 200 passages, 10 questions) before any full run.

Rough cost per dataset:
- about 3 calls per passage (roughly 30k for HotpotQA and 35k for MuSiQue; about 20k with the cache, decision 3);
- plus one summary per LSH cluster (a few thousand);
- no LLM calls at query time.

## 7. Reference: paper numbers (not comparable to ours)

The paper uses DIGIMON's own question sets and corpora: HotpotQA with 3,702 questions over a 66k-passage corpus, and
MuSiQue with 3,000 questions from a different split than ours. It scores the first 500 questions with
substring-containment accuracy on long-form output, using Llama-3.1-8B. HyGRAG reports 68.72 (HotpotQA) and 35.87
(MuSiQue) "accuracy". Our numbers will use the HippoRAG 2 question sets, the shared reader and EM/F1/LLM-Acc, so only
the relative ranking against our other systems is meaningful.

## 8. Update 2026-09-30: user decisions and embedding-dimension test

Decisions:
- title in the text (`title\ntext`);
- temperature 1.0 (identical to DeepSeek's default, and to not sending one);
- our own sqlite response cache, so a rerun replays every finished call for free.

**Embedding dimension (tested, no code written):**
- HyGRAG does not reject 2048-dim embeddings. The only hard-coded 1024 on its path is `cleora_dim` (default 1024 =
  bge-m3). Cleora averaging, LSH (hyperplanes are drawn in the data's dimension) and the three FAISS indexes
  (`_create_single_faiss_index(embedding_dim)`, taken from the data) all adapt to the embedding size.
  `Core/Index/FaissIndex.py` hard-codes 1024 but is only used when `use_*_vdb` is on, which is off for HyGRAG.
- The upstream failure at 1024: node embeddings are truncated to `cleora_dim` (`_initialize_text_embeddings`), but the
  question embedding (`_embed_text`) is not. So FAISS would get 1024-dim nodes and 2048-dim queries.
- voyage-4-nano is Matryoshka-trained (model card: 2048/1024/512/256). Our vLLM server rejects `dimensions` with HTTP
  400 ("does not support matryoshka representation"), because it was started without a Matryoshka override.
- Client-side truncation of the vLLM 2048 output to its first 1024 values, renormalised, equals SentenceTransformer's
  own `truncate_dim=1024`: cosine 0.99992–0.99996 on 5 strings, the same agreement as vLLM vs ST at 2048.
- Retrieval cost of fewer dimensions (HotpotQA, 200 random questions × all 9,811 passages, `title\ntext`, cosine):

  | dim | gold-passage recall@5 | top-5 overlap with 2048 |
  |---|---|---|
  | 2048 | 0.835 | 1.000 |
  | 1024 | 0.823 | 0.872 |
  | 512 | 0.807 | 0.816 |

Options:
- **A (recommended): `cleora_dim: 2048`.** A one-value YAML change, no code. HyGRAG then sees exactly the same
  embeddings as the other four systems. It also keeps the paper's relationship (`cleora_dim` = embedding size;
  1024 there because bge-m3 is 1024).
- B: 1024 everywhere. This needs either restarting vLLM with a Matryoshka override (it would interrupt the running
  jobs) or truncating in our embedding adapter, for nodes and queries alike. It works, but gives HyGRAG weaker
  embeddings than the other systems (−1.2 recall@5 points above).

**Check run (2026-09-30).** HyGRAG's own methods (`_initialize_text_embeddings`, `_embed_text`, `_build_faiss_index`,
`_create_single_faiss_index`, `_search_single_index`, `search_similar_*`, extracted verbatim from
`HKGraphTree.py`) were run on real voyage-4-nano vectors from the vLLM server: 4 passages, 3 entities, 3 questions.
- `cleora_dim: 1024`: stored vectors and indexes are 1024-dim, queries 2048. Every search logs "Failed to search chunk
  index" and returns `[]` (the silent failure).
- `cleora_dim: 2048`: indexes are 2048-dim and all 3 questions return the correct passage first. Option A confirmed.

## 9. Pre-implementation check (2026-09-30)

Settled:
- `title\ntext` per passage;
- temperature 1.0, via config `llm.temperature`, no patch needed;
- our sqlite response cache;
- `cleora_dim: 2048` (verified).

Found in this pass; the implementation must handle these:
1. **`embedding.dimensions` must be removed from Config2.** Upstream sets `dimensions: 1024`, and
   `EmbeddingFactory._try_set_model_and_batch_size` sends it with every request. Our vLLM server rejects any
   `dimensions` with HTTP 400. Every node embedding would fail and silently become a random vector
   (`_initialize_text_embeddings`).
2. **The embedding adapter must send `" "` for empty strings.** vLLM returns 400 on `""`, the same issue fixed for
   HippoRAG 2 and MemGraphRAG.
3. **Community-summary failures are silent.** An LLM error or empty reply becomes the placeholder text "Community at
   level X with entities and relationships from N members". That placeholder is embedded and can be retrieved as
   context. Failed summary embeddings fall back to the mean of member embeddings. *Guard:* count these; the full run
   should have 0.
4. Upstream behaviour, kept but documented:
   - summaries are capped at `max_tokens=300` while the prompt asks for "max 300 words", so some are cut
     mid-sentence;
   - 2 HotpotQA passages exceed 1200 tokens and become 2 chunks each; the title is only in the first chunk;
   - the NER prompt is sent twice per chunk; the second call is a cache hit.
5. Reproducibility: `random.seed` / `np.random.seed(42)` are set once in `HKGraphTree.__init__` (LSH hyperplanes).
6. Query time makes no LLM calls: the query-entity extraction is commented out, and the only `aask` calls are in
   `generation_qa` / `generation_summary`, which we skip.
7. Environment: llama-index is not installed in any env yet. The pinned `faiss-gpu==1.7.2` has no wheels for recent
   Python; use `faiss-cpu` (same API; HyGRAG's indexes are CPU-side). Whether llama_index `OpenAIEmbedding` accepts
   `voyageai/voyage-4-nano` gets checked at install time. The plan is a small adapter either way (item 2).

Still open: **concurrency** for the index build. The DeepSeek account limit (2500) is shared with any jobs still
running.
