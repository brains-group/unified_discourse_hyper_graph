# Round-2 fact clustering: why it took 2 h 40 min, and how to fix it

Run: HotpotQA full corpus (`outputs/hotpotqa_full`), launched 2026-10-06 07:54.
Graph at this stage: **109,066 facts**, 47,022 entities and 9,811 chunks.

## 1. Timeline of this run

| stage | ended | duration |
|---|---|---|
| extraction (9,811 chunks) | 08:20 | ~26 min |
| merge, entity dedup and fact dedup → `checkpoint/merged.graphml` | 08:51 | ~31 min (entity dedup 24 min, fact dedup 5 min) |
| **round 2: embed facts, build the similarity graph, run Leiden** | **11:33** | **~2 h 42 min** |
| round 2: 6,713 bulk LLM calls | 11:40: 6,706/6,713 done | ~7 min so far (the last few are the giant clusters, see section 3) |

The clustering itself was not stuck; it was slow. It ran with no output from 08:51 until `round_2_clusters.json` was written at 11:33.

## 2. Where the time goes

I benchmarked at the same scale: 109,066 facts and 1024-dimension voyage-4-nano vectors. The embeddings were synthetic and clustered, and the benchmark used GPU 1. The script is in the scratchpad: `bench_cluster.py`.

| step | time |
|---|---|
| **current `_build_sim_graph` loop** (one fact at a time) | **62 ms/fact → ~1.9 h for 109k facts** |
| ↳ 4 mat-vecs over the 109k×1024 matrices + full `argsort` + building 109k-element Python lists | 35 ms/fact |
| ↳ `sample_results` (rebuilds numpy arrays from those 109k-item lists, keeps 12) | 17 ms/fact |
| ↳ `dict(zip(all_ids, all_scores))` (109k-entry dict, 12 entries used) | 10 ms/fact |
| same scores computed as **one batched GPU top-k** (`weighted_cosine_topk`) | **5.3 s total** |
| building the networkx similarity graph from the top-k | 1.8 s |
| `graspologic.hierarchical_leiden(max_cluster_size=50)` | **2.4 s** |

**The answer: Leiden is not the slow part; choosing the top-k is.**

`construct_edges_between_facts_clustered._build_sim_graph` (`nkg/index/construction/construct_edges.py:1301`) does the following for each of the 109k facts:
- calls `get_relevant_seeds_by_id(get_all=True)`, which scores the fact against all 109k facts;
- fully sorts those 109k scores;
- turns them into Python lists and a dict;
- keeps 12.

That is O(N²) work done with Python-object overhead. Each per-fact mat-vec also re-reads about 1.8 GB of embeddings, which makes it memory-bound. Batched as matrix–matrix products on the GPU, the same arithmetic takes seconds.

The remaining ~40 min of the 2 h 42 min (not measured separately) is mostly in `init_fact_embeddings`. It:
- loads a new SentenceTransformer;
- re-encodes 5 text fields for every fact (sentence, topics, entity roles, answered questions, follow-up questions), about 545k texts;
- re-encodes fact sentences that fact dedup had just embedded with the same model.

## 3. A second problem: we use every Leiden level, not the final clusters

`hierarchical_leiden` returns one entry per (node, level). graspologic's documentation describes it as "a log of state at each level", and the real partition is the entries with `is_final_cluster=True`. All three of our callers group *every* entry by cluster id:

- `construct_edges.py:1215` (global fact edges, rounds 1–3)
- `fact_deduplication.py:405`
- `entity_deduplication.py:359`

The effect is visible in this run's `round_2_clusters.json`:

| | value |
|---|---|
| clusters | 6,713 (920 have more than 50 facts; the cap is 50) |
| largest clusters | 9,635 / 9,447 / 6,219 / 6,092 / … facts in **one prompt** |
| fact-lines sent to the LLM | 343,042 for 109,053 facts (**3.1× each fact**); 108,971 facts are in more than one cluster, up to 5 |
| `max_tokens=100000` truncation warnings | the giant prompts (also seen in entity dedup, with `Error resolving cluster` validation errors) |

Edge yield per completed cluster:

| cluster size | clusters done | mean edge rows per cluster |
|---|---|---|
| 2–50 | 5,793 | 18.4 |
| 51–200 | 729 | 94.8 |
| 201–1,000 | 157 | 349 |
| 1,001+ | 23 | 439 |

So **about 55% of round-2 edges come from the oversized, overlapping clusters**. Fixing this bug therefore changes the graph, not just the cost. The 250-subset results (and probably the August ones) were built with this behaviour.

## 4. Plan

### Fix A: batched similarity graph. Same output, ~1,000× faster (recommended now)

Replace the per-fact loop in `_build_sim_graph` with one call to the existing `nkg/utils/math_utils.weighted_cosine_topk`. It already does GPU chunked top-k, is already used by entity and fact dedup, and supports the asymmetric questions field:

```python
fields  = [sent, topic, entities, follow_up]          # query side
targets = [sent, topic, entities, answered]           # key side (questions_mode follow_up_to_answered)
idx, scores, _, _ = weighted_cosine_topk(fields, [w_s, w_t, w_e, w_q], k=top_k - 1, pair_with=targets)
```

- Build the edges from `idx` and `scores`, skip pairs already in the graph, and use the same `max(0.01, score)` weights.
- `top_k - 1`: the old loop took the top 12 including the fact itself, then dropped self, giving 11 neighbours.
- Round 3's `skip_percent_top_k` cannot use plain top-k: it skips the top 33% of all N facts. It needs `torch.kthvalue` or a sort per row block on the GPU. That is also a few seconds, but round 3 is off at the moment.
- **Verification before use:** run the old and new builders on the 250-subset HotpotQA graph and check that the edge sets match. Ties can differ in float rounding, as they did for the dedup change, where the edges came out identical.
- Expected round-2 clustering time: **under 1 minute** instead of ~2 h, excluding embedding.

### Fix B: stop re-encoding (saves most of the remaining ~40 min)

- Reuse the model already loaded for dedup. Today `init_fact_embeddings` loads a second copy, without `trust_remote_code`.
- Pass the fact-sentence vectors from fact dedup straight through.
- Save the five fact embedding matrices next to `merged.graphml` (`fact_embeddings.npz`), so a resume does not re-encode.

### Fix C: use only final Leiden clusters (changes the graph, so it needs a decision)

- Group by `is_final_cluster` (one line in each of the 3 callers), then use `_combine_small_clusters` to pack small clusters up to the prompt cap, as dedup already does.
- Effect: every prompt has at most 50 facts, there are no truncated 9k-fact prompts, and fact-lines sent drop ~3.1× (343k → ~109k).
- Risk: it removes the ~55% of round-2 edges that came from the overlapping coarse clusters. To keep similar connectivity but bounded, run Leiden twice: two seeds, or two resolutions. Each fact then lands in about 2 clusters of 50 or fewer facts, and the existing edge-pair check prevents duplicate edges. That gives about 2× fact-lines instead of 3.1×, with no giant prompts.
- Check retrieval with a ≤10-passage plumbing test, then the 250-subset answer-in-context sweep (`sweep_retrieval.py`).

### Is there a better clustering than Leiden?

Not for speed: Leiden takes 2.4 s here. Alternatives:
- capped k-means on the concatenated weighted embedding;
- mutual-kNN connected components;
- greedy agglomerative merging under a size cap.

All would be similar in speed and lose the graph structure (the existing-edge exclusion, and the weighting by neighbour rank), so **keep Leiden**. The only real gains are A, B and C.

## 5. Resuming from `merged.graphml`

**Yes.** `checkpoint/merged.graphml` (08:51) is the graph after extraction, merge, entity dedup and fact dedup. Its token and time totals are in `merged_state.json`. If you rerun the same `launch_unh.sh` command, it:
1. loads `merged.graphml` and skips extraction and both dedups;
2. finds `rounds/round_2_clusters.json`, so it does not re-embed or re-cluster;
3. replays the edges in `round_2_edges.jsonl` and sends only the clusters not yet done (and any that failed).

Caveats:
- **Construction arguments must be unchanged.** `main.py` passes `-y`. If any construction flag differs from `construction_config.json` (for example `--clustered-max-facts-per-prompt` or a fact-dedup setting), the run **wipes the entire checkpoint without asking**, including the 9,811 chunk extractions, and starts from scratch. Code changes do not trigger a wipe; only flag differences do.
  - Plan: drop `-y` from `main.py`, so a mismatch aborts instead of wiping.
  - Or make invalidation stage-aware: an edge-stage flag change should only clear `rounds/`.
- After Fixes A and B, a resume *mid-round* still uses the saved cluster list, so they cost nothing there. To re-cluster an existing graph with new code (for example Fix C on HotpotQA), delete only `checkpoint/rounds/` and keep `merged.graphml`.

## 6. Interaction with the run in progress

- The launcher is `SEQUENTIAL=1`: HotpotQA construction, then HotpotQA evaluation, then **MuSiQue construction in a new process**. MuSiQue will therefore load whatever `nkg` code is on disk when it starts.
- **Fixes A and B** do not change output (once parity is verified), so they are safe to apply before MuSiQue construction starts. That saves about 1–2 h there.
- **Fix C** changes the graph. Applying it only to MuSiQue would make the two datasets' graphs inconsistent. Either:
  - apply it after this run, for both; or
  - apply it now and rebuild HotpotQA round 2 from `merged.graphml` (delete `rounds/`; about 10 min of LLM calls, with fewer tokens than this time).

## 7. Implemented: `edge_mode="clustered_fast"` (2026-10-06)

The new module is `nkg/index/construction/fast_fact_clustering.py`. The legacy `clustered` mode is unchanged and is still the default. Select the new mode with:

```
main.py --edge-mode clustered_fast [--fast-cluster-embed-loader remote_code|legacy] \
        [--fast-cluster-leiden-levels final|all] [--fast-cluster-leiden-passes N]
```

The defaults are `remote_code`, `final` and `1`.

`fork_merged.sh SRC DST` copies a run's post-dedup checkpoint into a new run directory, so the new mode can be tried on an existing graph without redoing extraction or dedup.

### Parity check: 250-subset HotpotQA merged graph, 28,212 facts

| check | legacy | fast |
|---|---|---|
| embeddings (`legacy` loader) | 135 s | 133 s, identical |
| similarity graph | 534 s (18.9 ms/fact) | **1.4 s** |
| similarity-graph edges | 192,172 | 192,171 |

- Only 3 edges are legacy-only and 2 fast-only. These are float-rounding ties at rank 12.
- Leiden on those two nearly identical graphs gives 1,800 vs 1,788 all-level clusters. Leiden is sensitive to tiny input changes, so the fast `all` mode reproduces the legacy behaviour statistically, not bit for bit.

Fact-slots are the total number of fact lines sent to the LLM:

| clustering | clusters | largest | fact-slots |
|---|---|---|---|
| legacy all levels | 1,800 | 2,512 (259 over 50) | 76,165 (2.70 per fact) |
| `final`, 1 pass | 1,531 | 49 | 28,210 (1.00 per fact) |
| `final`, 2 passes | 2,733 | 49 | 51,535 (1.83 per fact) |

The real voyage model (`remote_code`, 2,048-dim) shares only **72%** of each fact's top-12 neighbours with the legacy causal-Qwen3 load (1,024-dim).

The 10-passage end-to-end construction smoke test (`clustered_fast`, defaults) passed: 137 facts, 6 clusters (largest 30), 6 LLM calls, 75 s.

### Full HotpotQA run (legacy mode)

- 6 round-2 clusters failed: their output was truncated and failed validation.
- All 6 were parent-level clusters of 522 to 6,092 facts, so no fact lost its own final cluster.
- They stay in the checkpoint as failed and would be retried on a resume.
