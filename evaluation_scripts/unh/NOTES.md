# UNH (nkg) in evaluation_scripts: configuration and every deviation

Built 2026-10-02 from [PLAN.md](PLAN.md), then revised the same day at the user's request:
- label-only fact→fact edges;
- round 3 off and fact dedup off;
- a subset corpus;
- a sequential evaluation queue;
- a hypergraph view.

## How to run

```bash
bash evaluation_scripts/unh/launch_unh.sh            # both datasets; detaches; logs in evaluation_scripts/run_logs/unh_*.log
bash evaluation_scripts/unh/run.sh hotpotqa          # one dataset in the foreground
bash evaluation_scripts/stop_all.sh                  # stop
```

`launch_unh.sh` does the following:
1. Starts both constructions at once.
2. As each one finishes, evaluates it: view **full**, then view **hypergraph**. Each view is retrieve → shared answer step → score with the LLM judge.
3. Never runs two evaluations at once. A dataset whose construction finishes during another evaluation waits for it.
4. Ends with `collect_results.py --subset`, which prints all systems on the same 250 questions.

UNH needs no embedding server: voyage-4-nano and the cross-encoder are loaded in-process, on `GPU` (default 0).

## Steps (`pipeline.sh construct|evaluate <ds>` → `main.py`)

| Step | What happens | Output (`outputs/<ds>_250_seed0/`) |
|---|---|---|
| prepare | `evaluation_data/<ds>/subsets/<ds>_250_seed0_corpus.json`: the passages of the 250 subset questions (HotpotQA 2,488 passages; MuSiQue 4,025) → `data/corpus.jsonl`. `UNH_CORPUS=full` uses the whole corpus instead. | `data/corpus.jsonl` |
| construct | UNH's own CLI (`hqa_musique_evaluation.py --mode construction --input-mode prepared_chunks`): one passage = one chunk, UNH chunk string `"Title: {title}\n\n{text}"`, resumable from `checkpoint/` | `kg.graphml`, `construction_results.json` (tokens and calls per stage), `construction_config.json` |
| shorten | no-op for label-only graphs, which already store each label as the edge's `description` | — |
| retrieve (full) | old engine, mode `all` | `contexts.json`, `retrieval_report_full.json` |
| retrieve (hypergraph) | old engine, mode `hypergraph`: entity seeds, entity↔fact edges only (no fact→fact edges, so edge labels play no part) | `contexts_hypergraph.json`, `retrieval_report_hypergraph.json` |
| answer / score | `common/answer.py` / `common/score.py --system unh --subset … --llm-judge` | `answers_unh_<ds>.json`, `scores_unh_<ds>.json`; hypergraph: `answers_unh_hypergraph_<ds>.json`, `scores_unh_hypergraph_<ds>.json` (row `unh_hypergraph` in `collect_results`) |

## Settings

| Setting | Value |
|---|---|
| Construction | UNH's `_CONSTRUCTION_DEFAULTS` (fact dedup on, with its defaults), except: `--no-clustered-round3`, `--no-edge-descriptions` (`main.py --round3/--edge-descriptions` switch them back on; `--no-fact-dedup` turns fact dedup off), `--dedup-max-cluster-size 72` (UNH default 64), `--dedup-similarity-threshold 0.5` (default 0.75), `--same-chunk-min-facts 3` (default 2); all three are `main.py` flags |
| Fact→fact edges | label-only prompts (`BulkIntraChunkFactEdgesLabelOnly`, `BulkGlobalFactEdgesLabelOnly`): the LLM writes only `label_forward`, `label_backward` and `relevance_score`, with no description sentence. Each label is stored as that direction's `description` too, so scoring, path strings and context all see the label. |
| Kept | concise chunk summaries (context in the edge prompts); answered and follow-up questions (round-2 clustering); topics |
| LLM | `deepseek-flash`, thinking disabled, **no temperature sent** |
| Concurrency | construction `--max-workers 500 --max-sub-workers 500` (`UNH_CONCURRENCY`) per dataset; HotpotQA on GPU 0, MuSiQue on GPU 1 (`GPU_HOTPOTQA`, `GPU_MUSIQUE`), each dataset's evaluation on its own GPU. 1,000 is the OpenAI client's connection limit per process (litellm → `openai`, `max_connections=1000`), so it is the most calls one construction can have in flight; each construction is its own process with its own limit. Chunk extraction runs `--max-workers` passages at once; entity dedup and the cross-chunk edge rounds run `--max-sub-workers` calls at once. Retrieval: 16 questions at a time |
| Token accounting | construction: `construction_results.json` → `token_breakdown` per stage (`prompt`, `completion`, `calls`, `cache_hit`) and `llm_prompt_cache_hit_tokens`. Retrieval: `retrieval_report_<view>.json` → `query_llm_usage_this_run`. Answer step: `answers_<tag>_<ds>_usage.json` and a `usage` field per answer. Judge: `scores_<tag>_<ds>.json` → `judge_usage`. `cache_hit` = prompt tokens served by DeepSeek's automatic prompt cache (it is always on and cannot be switched; DSPy's local cache stays off, because a local cache hit makes no API call and reports no usage) |
| Retrieval | old engine, `DEFAULT_RETRIEVAL_ARGS_OLD` (top_k_seeds 9, max_depth 3, beam_width 6, final_top_k 20, MMR 0.6 / 0.6, beam_search, deduplicate_paths), `neutral_layout` off, **no context cap** |
| Questions | `evaluation_data/<ds>/subsets/<ds>_250_seed0.json`: stratified and seeded, from the HippoRAG 2 question files |
| Answer / judge | shared HippoRAG 2 prompt / LinearRAG judge, the same as all baselines |

## ⚠ Comparability: subset corpus vs full corpus

- **Corpus size.** The six baselines indexed the **full** corpus: 9,811 HotpotQA / 11,656 MuSiQue passages, the union of all 1,000 questions' passages. UNH (default) indexes only the 250 subset questions' own passages: 2,488 (25%) / 4,025 (35%).
- **What stays the same.** Every subset question still has its own gold passages and its own distractors, which is HotpotQA's "distractor" setting.
- **What changes.** The passages belonging to the other 750 questions are gone. For UNH that's 75% / 65% fewer cross-question distractors than the baselines faced, so UNH's numbers are not strictly comparable to the baselines' 250-question numbers.
- **Options for a strictly comparable table:**
  - (a) Run UNH with `UNH_CORPUS=full`.
  - (b) Rebuild the baselines on the subset corpora: cheaper than their full builds, about 25–35% of their cost.

## Changes outside `unh/`

| File | Change |
|---|---|
| `nkg/index/extraction/extract_relations.py` | New `LabelOnlyFactEdgePair`, `BulkIntraChunkFactEdgesLabelOnly` and `BulkGlobalFactEdgesLabelOnly`: the Neutral prompts without the description, plus an explicit instruction that the labels are the relationship's only text, the backward-label rule (CAUSES ↔ CAUSED_BY; symmetric labels stay the same) and the existing `FACT_FACT_LABEL_GUIDANCE`. |
| `nkg/index/construction/construct_edges.py` | `_bulk_fact_edge_constructor` / `_fact_edge_descriptions` helpers. `edge_descriptions` threaded through the bulk same-chunk (sequential and parallel), `construct_initial_edges`, `construct_edges_during_merge` and clustered edge functions. Default `True` (unchanged behaviour). |
| `nkg/index/construction/build_index.py` | `edge_descriptions` parameter (default `True`), passed to the per-chunk worker and the cross-chunk edge step. Recorded in `token_breakdown.hyperparams`. Raises if `False` with an `edge_mode` other than `clustered` (the only mode with label-only prompts). |
| `experiments/systems/nkg/hqa_musique_evaluation.py` | `--edge-descriptions/--no-edge-descriptions`, `--fact-dedup/--no-fact-dedup` (new `_CONSTRUCTION_DEFAULTS` keys, both default `True`), `--temperature`. Resume treats a key missing from an older saved config as unchanged when it's at its default. |
| `experiments/systems/nkg/short_description_ablation.py` | `make_short_label_graph` function (identical output) |
| `common/make_subset.py` | also writes `<subset>_corpus.json` |
| `common/answer.py`, `score.py`, `collect_results.py` | `unh` registered; `--subset`; subset table with paired bootstrap CIs |
| `stop_all.sh`, `launch_all.sh` | stop the UNH pipeline; point to `launch_unh.sh` |

## Smoke tests (2026-10-02)

**First version** (full-corpus mode, descriptions on, round 3 and fact dedup on): 150 HotpotQA passages, 5 questions. Everything worked end to end.

| Measure | Value |
|---|---|
| Construction LLM tokens | 1.95M input / 1.26M output, 1,127 calls |
| Round 3 | 613k tokens |
| Fact dedup | 430k tokens |

**Label-only prompts, tested directly against DeepSeek:**
- They return `LabelOnlyFactEdgePair` objects (`source_id`, `target_id`, `label_forward`, `label_backward`, `relevance_score`), for example `RESULTS_IN / RESULTS_FROM` and `PROVIDES_CONTEXT_FOR / CONTEXTUALIZED_BY`.
- They store each label as that edge's `description`.

**Current version.** Run as `UNH_FOREGROUND=1 UNH_OUT_ROOT=<scratch> UNH_MAIN_ARGS="--limit-docs 60 --limit-q 3" launch_unh.sh`: both datasets, 60 subset-corpus passages and 3 questions each, real DeepSeek, GPU 0.

- **Orchestration.**
  - Both constructions started at 07:37:38.
  - MuSiQue finished at 07:39:53, so its evaluation ran first.
  - HotpotQA finished at 07:39:59, *during* MuSiQue's evaluation. It waited, and started its evaluation at 07:41:35 when MuSiQue's ended.
  - The run then printed the subset table and finished with 0 failures.
- **Construction settings** (recorded in both builds): `edge_descriptions=False`, `run_fact_dedup=False`, `clustered_round3=False`, `temperature=none`, 2000/2000 workers.
  - The fact dedup and round 3 stages used 0 tokens.
  - Every fact→fact edge has `description == label` (3,258 / 3,258 and 2,552 / 2,552); for example `PRECEDES`, `ELABORATES_ON`, `CORROBORATES`.
- **Views.**
  - Both views: 3/3 contexts, 0 errors, about 3–5k characters each.
  - Full view: has fact→fact label arrows.
  - Hypergraph view: has none (entity↔fact only).
  - Answers and scores (with the judge) were written for both views. The scores are meaningless at 3 questions over 60 passages.

**Measured construction cost per passage (new configuration):**

| Dataset | Input tokens | Output tokens |
|---|---|---|
| HotpotQA | 7,951 | 4,752 |
| MuSiQue | 7,156 | 4,023 |

The first version used about 13,000 input and 8,400 output tokens per passage.

**Projected construction cost:**

| | HotpotQA | MuSiQue |
|---|---|---|
| Subset corpus (default) | 2,488 passages: about **20M in / 12M out** | 4,025 passages: about **29M in / 16M out** |
| Full corpus | about 78M in / 47M out | about 83M in / 47M out |

Round-2 clustering grows with corpus size, so treat these as lower bounds.
