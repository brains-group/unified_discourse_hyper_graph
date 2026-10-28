# QuALITY benchmark: implementation and per-system protocol

Built 2026-10-01 from the decisions in [QUALITY_EVAL_PLAN.md](QUALITY_EVAL_PLAN.md). Models are the same as for HotpotQA/MuSiQue:
- **LLM:** DeepSeek V4.1 Flash (`deepseek-flash`), with thinking disabled and no temperature sent.
- **Embeddings:** voyage-4-nano, served by vLLM on port 8001; LogicRAG instead loads it in-process with SentenceTransformer.

No system's internal code was changed.

## 1. The protocol in one paragraph

- **Data:** QuALITY's official dev split, **115 articles, 2,086 questions** (1,065 flagged HARD).
- **Index:** each system builds **one pooled index over all 115 articles**, as HyGRAG/DIGIMON do. Every question is retrieved against the whole pool, and nothing tells a system which article a question is about.
- **Query:** the question with its four options (`"…?\nA: …\nB: …\nC: …\nD: …"`), the same string HyGRAG/DIGIMON use.
- **Retrieval:** each system keeps its own retrieval and context size, as on HotpotQA/MuSiQue.
- **Answer:** one shared answer step for every system. The prompt holds the context and the question, and asks for **the letter only**.
- **Scoring:**
  - The metric is HyGRAG's: accuracy of the chosen letter against the gold letter, reported overall and on HARD and EASY.
  - A reply that isn't a bare letter goes to HyGRAG's own LLM option extractor, copied verbatim.

## 2. Files

| File | What it does |
|---|---|
| `evaluation_data/build_quality_inputs.py` | Builds every system's input from `evaluation_data/quality/{quality.json,quality_corpus.json}` into `quality/inputs/` (or `<subset>/inputs/` with `--subset`). |
| `common/prompts.py` (added at the end) | `build_mcq_messages`, the shared prompt; `parse_option_letter`, which reads a bare-letter reply. |
| `common/hygrag_close_extract_prompt.py` (new) | HyGRAG's `CLOSE_EXTRACT_OPTION_PORMPT`, byte-identical, extracted from the source text. |
| `common/answer.py` | New `--dataset quality` mode: same context loaders as before, the MCQ prompt instead of HippoRAG's prompt, and questions from `questions.json`. |
| `common/score_quality.py` (new) | Accuracy / Acc-HARD / Acc-EASY, extractor fallback, diagnostic counts. |
| `common/collect_results.py` | Also prints a QuALITY table from every `scores_<system>_quality.json`. |
| `<system>/run_quality.sh` (new, ×6) | One system end to end: index → contexts → `answer.py --dataset quality` → `score_quality.py`. |
| `launch_quality.sh` (new) | Starts all six `run_quality.sh` jobs in the background (`setsid nohup`), with logs in `run_logs/<system>_<tag>.log`. |
| `stop_all.sh` | Now also stops `*_quality*` jobs. |
| Loaders in `hipporag2/main.py`, `hypergraphrag/main.py`, `ms_graphrag/query.py`, `hygrag/{prepare_data,main}.py` | A `quality` choice that reads `quality/inputs/`. Each system's `UPSTREAM.md` has a QuALITY section. |

Two environment variables control every QuALITY script:
- `QUALITY_INPUTS`: the inputs directory. Default `evaluation_data/quality/inputs`.
- `QUALITY_TAG`: the output name. Default `quality`. The smoke test used `quality_smoke`.

## 3. Inputs (`build_quality_inputs.py`)

Every article is the document `"title\ntext"`. The title is put before the article once. It is not repeated on every chunk.

| File | Used by | Contents |
|---|---|---|
| `questions.json` | all | 2,086 rows. Fields: `id`; `question` (the question with options, exactly DIGIMON's format); `question_text`; `options`; `answer` (the option text); `answer_letter`; `difficult` (official HARD flag); `article_id`; `title`. Gold letters: A 491, B 528, C 539, D 528. |
| `documents.json` | HyGRAG, HyperGraphRAG | 115 × `{article_id, title, text}`; the runners index `title\ntext`. |
| `chunks.json` | HippoRAG 2 | 652 chunks. `title\ntext` cut with HyGRAG's own chunking loop (`Core/Chunk/Tokensize.py`): tiktoken gpt-3.5-turbo, 1,200-token windows starting every 1,100 tokens, each article on its own, decoded and stripped. Only chunk 0 of an article carries the title. |
| `chunks_logicrag.json` | LogicRAG | 651 chunks. The article text without the title line, cut the same way. `title` = article title on chunk 0, `""` on later chunks (option B), so LogicRAG reads `Title: The Girl in His Mind. Content: …` once per article. |
| `memgraphrag_corpus.txt` | MemGraphRAG | The 115 `title\ntext` documents joined by a blank line. |
| `ms_graphrag_input/<article_id>.txt` | MS GraphRAG | 115 files, one `title\ntext` each. |
| `manifest.json` | — | Counts and SHA-256 of every file. |

Two details about the chunks:
- **Tail chunks.** HyGRAG's chunking loop starts a window every 1,100 tokens until the end of the text, so an article whose last window ends just past a multiple of 1,100 gets a tiny tail chunk (the smallest is 4 tokens). We copy this exactly, so HippoRAG 2 and LogicRAG get the same units HyGRAG builds internally.
- **Same text as HyGRAG's copy.** The article text matches HyGRAG's own QuALITY copy apart from whitespace, for all 115 articles.

## 4. How each system is evaluated

Each system indexes with its own method, retrieves with its own method, and passes the context below to the shared answer step.

| System | What it indexes | Who chunks | Query | Context passed to the answer step |
|---|---|---|---|---|
| **HyGRAG** | `Corpus.json`: 115 rows, `context = title\ntext` (DIGIMON's own QuALITY layout) | HyGRAG, 1200/100 per article, then one pooled graph | question + options | `HKGraphTreeQuery._retrieve_relevant_contexts` (1 community + 4 chunks, 5 entities, 5 relations) |
| **HyperGraphRAG** | `rag.insert`: 115 strings `title\ntext` | HyperGraphRAG, 1200/100 per document | question + options | `aquery(only_need_context=True)`, hybrid mode |
| **MS GraphRAG** | `input/`: 115 `.txt` files | graphrag, 1200/100 per document (o200k) | question + options | local-search `build_context` text |
| **MemGraphRAG** | one `.txt`, the 115 documents joined | MemGraphRAG, 256/32-token windows over the whole file | question + options | top-5 windows (`--qa-top-k 5`) |
| **HippoRAG 2** | 652 pre-cut chunks, one passage each | pre-chunked with HyGRAG's chunker (HippoRAG 2 has none) | question + options | top-5 passages (`rag.retrieve`; HippoRAG's own QA call is skipped) |
| **LogicRAG** | 651 pre-cut chunks (option B titles) | pre-chunked with HyGRAG's chunker (LogicRAG has none) | question + options | `info_summary` (what its own `generate_answer` reads), 3 rounds, top-5 |

All index and retrieval settings are the ones used for HotpotQA/MuSiQue. MemGraphRAG's windows cross article boundaries, which is upstream behaviour. At most 114 of about 3,000 windows mix the end of one article with the start of the next, and no QuALITY question spans articles.

## 5. The shared answer step (`common/answer.py --dataset quality`)

```
system: You answer multiple-choice questions about a text, using the context you are given.
user:   Context:
        <the system's context; several documents are separated by a blank line>

        Question:
        <question>
        A: …
        B: …
        C: …
        D: …

        Answer the question to the best of your ability using the context. Reply with only the letter of the
        correct option (A, B, C, or D).
```

- **Settings, same as the HotpotQA reader:** model `deepseek-flash`, thinking disabled, no temperature sent, max_tokens 2048 (a letter-only reply stops after a token or two), 16 workers.
- **Cache:** every call is cached in `reader_cache.sqlite` next to the output.
- **Parsing:** `pred_answer` is the letter when the reply is a bare letter. `parse_option_letter` also accepts `B.`, `(B)`, `B: 10 hours`, `Answer: B` and `**B**`; only capital A–D count. Any other reply is stored as `""`.
- **Question check:** every row's question string is checked against `questions.json` before answering, so a misaligned file stops the run.

## 6. Scoring (`common/score_quality.py`)

1. **Letter.** The letter comes from the reply. If the reply is not a bare letter, it goes to **HyGRAG's option extractor**: `CLOSE_EXTRACT_OPTION_PORMPT` verbatim, sent to the same LLM. The JSON reply holds `predict`, which is a letter or `"-1"`. Extractor calls are cached in `extract_cache.sqlite`.
2. **Comparison.** Correct means the letter is one of A–D and equals `answer_letter`. Overall accuracy is reported as **Accuracy**, alongside **Acc-HARD** and **Acc-EASY**.
3. **Counts in the output:**
   - `letter_from_reply`
   - `extractor_used`
   - `extractor_key_model_output`
   - `no_option`
   - `empty_context`
   - `mean_context_chars`
   - the predicted-letter distribution, which shows position bias

Two HyGRAG scorer bugs are deliberately not copied, and both are counted:
- **Normalisation.** HyGRAG compares letters after SQuAD normalisation, which deletes the article "a". Gold "A" then normalises to the empty string, so an empty extraction also counts as correct. Our comparison is strict.
- **JSON key.** The extractor prompt's FORMAT section names the key `model_output`, but HyGRAG reads `predict`. Following the FORMAT section crashed into "-1" upstream. We read either key.

Also found: upstream `Core/Utils/Evaluation.py` does not compile (line 505 has curly quotes: `featurize_model_name=”gpt2-large”`), so upstream `main.py` cannot reach its own QuALITY scorer as shipped. The identical prompt is in `run_evaluation.py`, which presumably produced the paper's numbers.

## 7. What differs from HyGRAG's own QuALITY runs, and why

| | HyGRAG paper / code | Here | Why |
|---|---|---|---|
| Articles in the pool | 265 (train 150 + dev 115) | 115 (dev) | official split with HARD flags; under half the build cost (your decision) |
| Questions scored | `main.py`: the first 500 (all dev, about 39 articles) | all 2,086 dev questions | full split |
| Answer prompt | each method's own prompt; HyGRAG's is a long, reasoned answer that may say "Insufficient information" | one shared prompt, letter only | the answer step is identical for every system; differences come from retrieval |
| Letter extraction | always the LLM extractor | parse the bare letter; the extractor only as fallback | the prompt asks for a letter; the extractor is still HyGRAG's, verbatim |
| Letter comparison | normalised (gold "A" == "") | strict | bug |
| Title | not in the chunk text (`title` field dropped) | `title\ntext` per article, so it appears in the first chunk | same as our HotpotQA/MuSiQue setup |
| LLM / embeddings | Llama-3.1-8B / bge-m3 | deepseek-flash / voyage-4-nano | our benchmark's models |

## 8. How to run

```bash
# inputs (already built; deterministic, so rebuilding gives identical files and SHA-256 in manifest.json)
python evaluation_data/build_quality_inputs.py
python evaluation_data/build_quality_inputs.py --subset subsets/quality_100q_seed0     # smoke inputs

# full benchmark: all six systems in parallel (needs the voyage-4-nano server on :8001)
bash evaluation_scripts/launch_quality.sh
bash evaluation_scripts/launch_quality.sh hygrag ms_graphrag                              # some systems only

# faster: multiply every system's LLM concurrency settings (common/env.sh scale_concurrency; speed only)
CONCURRENCY_MULTIPLIER=3 GPU=1 bash evaluation_scripts/launch_quality.sh hipporag2 memgraphrag logicrag hypergraphrag hygrag

# one system in the foreground
bash evaluation_scripts/hygrag/run_quality.sh

# results table
python evaluation_scripts/common/collect_results.py
```

**Concurrency multiplier.** `CONCURRENCY_MULTIPLIER` (default 1) multiplies each setting below by the given factor. Fractions are allowed; results are rounded and never below 1.

| System | Base values multiplied |
|---|---|
| HyGRAG | `HYGRAG_CONCURRENCY` (500) |
| HyperGraphRAG | index `max_async` 32/32; retrieval `max_async` 16/16; questions in flight 32 |
| HippoRAG 2 | OpenIE workers (8) |
| MemGraphRAG | memory workers (20); QA workers (5) |
| MS GraphRAG | `GRAPHRAG_CONCURRENCY` (1200) |
| Answer and score steps | workers (16) |

Some limits are not multiplied:
- LogicRAG answers one question at a time.
- HippoRAG 2's retrieval loop is sequential.
- MemGraphRAG's OpenIE pool is Python's default (32 on this machine).

Approximate peak requests in flight for the five systems other than MS GraphRAG:

| Multiplier | Peak in flight |
|---|---|
| ×1 | ~600 |
| ×2 | ~1,170 |
| ×3 | ~1,750 |

**Outputs:**

| System | Directory | Retrieval file |
|---|---|---|
| HippoRAG 2 | `hipporag2/outputs/quality/` | `predictions.json` |
| LogicRAG | `logicrag/result/quality/` | `logicrag_quality.json` |
| MemGraphRAG | `memgraphrag/outputs/quality/` | `qa_results.json` |
| HyperGraphRAG | `hypergraphrag/outputs/quality/` | `knowledge.json` |
| MS GraphRAG | `ms_graphrag/outputs/quality/` (workspace in `caches/quality/`) | `local_search_contexts.json` |
| HyGRAG | `hygrag/outputs/quality/` | `contexts.json` |

Each directory also holds `answers_<system>_quality.json` and `scores_<system>_quality.json`.

**Projected construction cost for the full dev set,** from the HotpotQA/MuSiQue ratios:

| System | Input tokens | Output tokens |
|---|---|---|
| HippoRAG 2 | 5M | 1.8M |
| HyGRAG | 7M | 2.2M |
| MS GraphRAG | 24M | 11M |
| MemGraphRAG | 77M | 8M |
| HyperGraphRAG | 87M | 9M |

The answer step adds 2,086 calls per system.

## 9. Smoke test

2026-10-01. Setup:
- `subsets/quality_100q_seed0`: 6 articles, 113 questions (55 HARD), 32k corpus tokens.
- Real DeepSeek and vLLM voyage-4-nano.
- Command: `QUALITY_TAG=quality_smoke QUALITY_INPUTS=evaluation_data/subsets/quality_100q_seed0/inputs bash evaluation_scripts/launch_quality.sh` (concurrency 200).

**All six ran end to end with no errors.**
- Every question matched `questions.json`.
- All 678 answer-step replies were bare letters, so HyGRAG's extractor was never needed (0 calls).
- No answer was left without an option.

| System | Index built | Accuracy | Acc-HARD | Acc-EASY | Mean context (chars) | Empty contexts |
|---|---|---|---|---|---|---|
| HippoRAG 2 | 32 chunk passages | 88.50 | 83.64 | 93.10 | 23,993 | 0 |
| HyGRAG | 32 chunks, 752 entities, 19 communities, 2 levels; every build check passed, 0 flagged questions | 87.61 | 78.18 | 96.55 | 22,017 | 0 |
| HyperGraphRAG | 6 docs → 32 chunks, 2,710 nodes, 1,479 hyperedges | 82.30 | 69.09 | 94.83 | 67,159 | 0 |
| MemGraphRAG | 145 windows (136 unique) | 82.30 | 69.09 | 94.83 | 5,557 | 0 |
| LogicRAG | 32 chunks (option B titles) | 81.42 | 69.09 | 93.10 | 604 (summary) | 0 |
| MS GraphRAG | 6 docs → 32 text units, 288 entities, 34 communities | 54.87 | 45.45 | 63.79 | 21,297 | **16** |

These numbers only show that the pipeline works. With 6 articles in the pool, finding the right story is trivial, and some systems' contexts cover a large share of the whole corpus (HyperGraphRAG returns about 67k of the corpus's 140k characters). Only the full 115-article run is meaningful.

### Finding: MS GraphRAG's stock `use_lcc` hides most articles in a pooled QuALITY index

- **The mechanism.** Local search only sees entities that belong to a community. With the stock `cluster_graph.use_lcc: true`, only the largest connected component of the entity graph is clustered. On HotpotQA this cost a few empty contexts (see `ms_graphrag/UPSTREAM.md`). QuALITY articles are separate stories that share almost no entities, so the graph falls apart into roughly one component per article, and the largest component covers only a few of them.
- **What the smoke test showed.** Only 118 of 288 entities were searchable:

  | Article | Entities | Searchable | Questions | Empty contexts | Correct |
  |---|---|---|---|---|---|
  | 99905 | 106 | 86 | 19 | 0 | 17 |
  | 23563 | 28 | 28 | 19 | 0 | 14 |
  | 50818 | 75 | 6 | 18 | 7 | 6 |
  | 60745 | 15 | 1 | 20 | 0 | 7 |
  | 51362 | 34 | 0 | 17 | 3 | 6 |
  | 63523 | 35 | 0 | 20 | 6 | 12 |

  Questions on the four barely-searchable articles got empty contexts or context from other stories.
- **For the full run.** With 115 articles, common entities ("Earth", "Mars") may link more stories together, so the size of the effect is unknown until then. `query.py` prints `entities_searchable` for that run.
- **What we do now.** We keep the stock setting, as for HotpotQA/MuSiQue, and report the count. Setting `use_lcc: false` would change a method setting, which is the user's decision, not housekeeping.

