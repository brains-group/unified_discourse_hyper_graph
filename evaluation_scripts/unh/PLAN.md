# UNH in the shared HotpotQA / MuSiQue benchmark: plan

Status 2026-10-02: decided and implemented, then revised (label-only edges, round 3 and fact dedup off, subset corpus, sequential evaluation queue, hypergraph view). [NOTES.md](NOTES.md) is the current reference; parts of this plan, the full-corpus sections in particular, are superseded.

Goal: evaluate UNH (`nkg`) exactly like the six baselines in `evaluation_scripts/`:
- the same data (HippoRAG 2 release, `evaluation_data/<ds>/`);
- the same models (deepseek-flash with thinking off; voyage-4-nano);
- the same shared answer step (`common/answer.py`);
- the same scorer and judge (`common/score.py`: EM, F1, LLM-Acc).

The only difference is the question count: a fixed 250-question subset per dataset.

## 1. Folder

`evaluation_scripts/unh/` sits next to `hipporag2/`, `hygrag/` and the rest. It reuses `common/` unchanged except for the small additions in §4.

| File | Purpose |
|---|---|
| `PLAN.md` | this plan |
| `main.py` | construction + retrieval for one dataset → `outputs/<ds>/contexts.json` |
| `run.sh <ds>` | `main.py` → `common/answer.py --system unh` → `common/score.py --system unh` (with the judge) |
| `NOTES.md` | the exact UNH configuration used, and every deviation from UNH's own defaults (like `UPSTREAM.md` for the others) |

`main.py` imports `nkg` directly: `build_index`, `Graph`, and the chosen retriever. It doesn't go through `experiments/systems/nkg/hqa_musique_evaluation.py`, which reads the old `wiki_data` JSONL format and has its own answer prompt.

## 2. Ingestion: one passage = one chunk

- Corpus: **the whole** `evaluation_data/<ds>/<ds>_corpus.json`. HotpotQA has 9,811 passages and MuSiQue 11,656. This is the same corpus every baseline indexed.
- Each passage is the string `title\ntext`, the same string HippoRAG 2, HyperGraphRAG and HyGRAG index.
- `build_index(chunks=[...], chunk_list_inputs=False)` is the existing `prepared_chunks` mode. Each passage is one chunk with no re-chunking and no concatenation. No scrambled `.txt` is used, and no chunk can span two passages, which removes the leak the old `wiki_data` builds had.
- Construction checkpointing goes under `outputs/<ds>/checkpoint/`, so a crash resumes. This is UNH's own mechanism.

## 3. The 250-question subset: how to keep it fair

### 3a. Choosing the questions

- **The subset is stratified and seeded,** and lives in a file every system shares: `evaluation_data/<ds>/subsets/<ds>_250_seed0.json`. It is a list of question ids, in dataset order, plus a manifest with the seed and counts.
- **HotpotQA:** proportional by `type`. That gives bridge 203 and comparison 47 (the full set has 811 and 189).
- **MuSiQue:** proportional by hop type (`2hop`, `3hop1`, `3hop2`, `4hop1`, `4hop2`, `4hop3`). That gives about 129 / 61 / 18 / 27 / 7 / 8 (the full set has 518 / 243 / 73 / 108 / 27 / 31).
- **A seeded random sample within each stratum, not the first 250.** The dataset files aren't randomly ordered.

### 3b. Same corpus for UNH

- **UNH indexes the full corpus, not just the 250 questions' passages.** Every baseline searched all 9,811 / 11,656 passages, distractors from the other 750 questions included. A graph built only from the subset's passages (about 2,500) would make retrieval much easier, and its score would not be comparable.
- Only the questions are subset, never the corpus.

### 3c. The baselines don't need rerunning

- Every baseline's `scores_<system>_<ds>.json` already has per-question EM, F1, Contain-Acc and LLM-Acc (`per_example`), all from the same answer step and judge.
- Their 250-question scores are just the averages over the subset ids. This means:
  - no new API calls;
  - identical answers to the 1,000-question runs;
  - all seven systems scored on exactly the same 250 questions, with the same reader prompt, scorer and judge model.

### 3d. Don't tune on the 250

- The baselines ran with their default settings. If UNH's configuration (engine, layout, short vs long labels, context budget) is chosen by looking at its score on these 250 questions, its number is optimistically biased.
- So: fix UNH's configuration first. If any tuning is needed, use a separate dev set (`<ds>_dev100_seed1`) drawn from the other 750 questions. Then run the 250 once.
- The earlier `sample_exp` tuning used the old `wiki_data` questions. That's fine, as long as none of the 250 were tuned on.

### 3e. Report the uncertainty

- At n=250, one question is 0.4 points, and a 95% interval on F1 is roughly ±4–5 points.
- The comparison table should give a paired bootstrap interval for UNH minus each baseline, over the same 250 questions. That's cheap, because all per-question scores exist.

## 4. Changes to `common/` (small, harness only)

| File | Change |
|---|---|
| `common/make_subset.py` (new) | Writes `evaluation_data/<ds>/subsets/<ds>_250_seed0.json` (stratified, seeded) and a manifest. |
| `answer.py` | Register `unh`: `contexts.json` → `context_text` as one document, as for `ms_graphrag` and `hygrag`. New `--subset` (an ids file): dataset samples are filtered to those ids in dataset order before the existing position and question-string check. |
| `score.py` | Register `unh`. Same `--subset` filter. The LLM-Acc judge runs only on the 250. |
| `collect_results.py` | New `--subset <ids file>`. It recomputes every system's EM, F1 and LLM-Acc from `per_example` on those ids, adds UNH, and prints the table with bootstrap intervals. |
| `ctxlen.py` / `llmcost.py` (from the scratchpad) | Move into `common/` and add UNH: context tokens per question, and construction LLM calls and tokens. UNH already records these in `construction_results.json`. |

Nothing else in `common/` changes. Baseline outputs are only read, never rewritten.

## 5. UNH run settings (same models as the baselines)

| Setting | Value | Note |
|---|---|---|
| LLM | `deepseek-flash` via `https://api.deepseek.com`, thinking disabled | `nkg.utils.config.configure_dspy` already disables thinking for `deepseek/` models |
| Temperature | **decision D3** | UNH's `configure_dspy` sends 0.1. Baselines send none (DeepSeek default 1.0), except HyGRAG, which sends 1.0. |
| Embeddings | voyage-4-nano | UNH loads it in-process with SentenceTransformer, like LogicRAG. Same model as the vLLM endpoint (cosine 0.9997). |
| Cross-encoder | `cross-encoder/ms-marco-MiniLM-L6-v2` | part of UNH's retrieval |
| Answer | `common/answer.py`: HippoRAG 2 prompt, max_tokens 2048, thinking off | UNH's own short-answer prompt is not used (same rule as every baseline) |
| Judge | `common/score.py --llm-judge` (LinearRAG prompt, deepseek-flash) | same as the baselines |

## 6. Cost and time

The July HotpotQA subset build (988 passages, DeepSeek) used 8.2M input / 1.7M output tokens in 2,989 calls and took 33 min. Scaled to the full corpora (about 10× and about 12×):

| Dataset | Passages | Projected input tokens | Projected output tokens | Comparable baselines |
|---|---|---|---|---|
| HotpotQA | 9,811 | about 80M | about 17M | MemGraphRAG 159M / HyperGraphRAG 162M input |
| MuSiQue | 11,656 | about 100M | about 20M | |

These are rough. Cross-chunk edge clustering and deduplication may scale worse than linearly, and the current defaults (round 3 on, `fact_batch_size` 15) differ from the July build.

Retrieval plus answering for 250 questions per dataset costs little.

## 7. Decisions needed before implementation

| # | Decision | Options | Recommendation |
|---|---|---|---|
| D1 | Construction parameters | current `_CONSTRUCTION_DEFAULTS`, or the July build's (`fact_batch_size` 40, dedup threshold 0.35, cluster 72, round 3 off, `same_chunk_min_facts` 5) | **Current defaults**, plus fact dedup on. They're UNH's stated defaults; the July values were tuning. |
| D2 | Retrieval | old engine with `--default-retrieval-args` and short labels (best so far: F1 73.4 on the old subset), or the new engine (UNH's default `--engine`), or dag | Pick one before the test run (see §3d). The best so far is old engine + short labels, which needs a short-label graph. Here that would be built directly, by writing labels only at query time or using the description → label copy. |
| D3 | Temperature | UNH's 0.1, or not sent | **Not sent**, like the baselines. |
| D4 | Context budget | UNH's 2,000-token cap, or no cap | Keep UNH's own cap: each baseline keeps its own budget (LogicRAG about 70 tokens, HyperGraphRAG about 21.5k). Report context tokens per question in the table. |
| D5 | Subset | stratified seeded 250 (§3a), or the first 250 | Stratified seeded. |
| D6 | Concurrency | `--max-workers` / `--max-sub-workers` for construction | Raise these well above the July values (4 and 10); DeepSeek allows high concurrency. Exact numbers to set after a smoke test. |

## 8. Order of work (after approval)

1. Write `common/make_subset.py` and the two subset files. Check the strata counts and that the ids exist.
2. Add the `common/` changes (`unh` registration, `--subset`). Confirm that `collect_results.py --subset` reproduces each baseline's full-set numbers when given all 1,000 ids.
3. Write `unh/main.py` and `run.sh`. Smoke test on 200 passages and 10 questions, the same size as the baseline smoke tests.
4. Run the full-corpus construction on HotpotQA, then MuSiQue. Each is a long job, so it needs your explicit go-ahead.
5. Run retrieval on the 250, then the shared answer step, then scoring with the judge, then the subset table for all seven systems.
