# HippoRAG 2: the graph and how Personalized PageRank (PPR) is used

Sources:
- the vendored code in `evaluation_scripts/hipporag2/src/hipporag/` (upstream `OSU-NLP-Group/HippoRAG` @ `1438aba`), mainly `HippoRAG.py`;
- the graphs our benchmark built: `outputs/{hotpotqa,musique}/deepseek-flash_voyageai_voyage-4-nano/graph.pickle`;
- a probe that re-runs HippoRAG's seeding and PPR on 100 HotpotQA questions. It is in the scratchpad: `hr_ppr_probe.py`.

Line numbers below refer to `src/hipporag/HippoRAG.py`.

---

## 0. The short version

1. **The graph is not "just triples".** Its nodes are **entity phrases** (the subjects and objects of the triples) and **passages**. It has three kinds of undirected, weighted edges:
   - **fact**: subject–object, one per extracted triple. The predicate is *not* in the graph.
   - **passage**: passage–entity, linking a passage to every entity in its triples.
   - **synonym**: entity–entity, when the entity embeddings have cosine ≥ 0.8.

   Facts (triples) are not nodes. They live only in a vector store and are used to *pick the seeds*.
2. **PPR runs once per query, over the whole graph** (about 107k nodes and 370k edges for HotpotQA). It is solved exactly with igraph's PRPACK and returns a score for every passage. Passages are ranked by that score.
3. **The seeds are not "top-k nodes by similarity" in the plain sense.** The pipeline is:
   - embed the query, match it against **facts**, and keep the top 5 facts;
   - an **LLM filter** removes irrelevant facts ("recognition memory");
   - the subject and object entities of the surviving facts become seeds, at most 5 entities;
   - **every passage** is also a seed, with a small weight (its dense-retrieval score × 0.05).
4. **The transition matrix is not balanced by edge type at all.** It is plain weighted-degree normalisation, P(u→v) = w(u,v) / Σ w(u,·). Almost every edge weight is ≈ 1:
   - fact edges = number of times that subject–object pair was extracted, median 1;
   - passage edges = 1;
   - synonym edges = cosine, 0.8–1.0.

   All the balancing happens in the **restart (reset) vector** and the **damping of 0.5**:
   - entity seeds are divided by how many passages mention the entity (an IDF-like penalty);
   - passage seeds are scaled by 0.05;
   - damping 0.5 makes walks very short.
5. **Measured on our HotpotQA graph:** the 5 entity seeds hold only **~2.6%** of the total restart mass; the 9,811 weak passage seeds hold the other ~97%. Yet the entity seeds produce **~68%** of the final score of the top-5 passages, because their mass is concentrated on 5 nodes while the passage mass is spread thinly over the whole corpus (section 5).

---

## 1. Indexing: what goes into the graph

### 1.1 OpenIE per passage

For each passage (chunk), the LLM does two steps (`information_extraction/openie_openai.py`):
1. **NER** extracts named entities.
2. **Triple extraction** extracts `(subject, predicate, object)` triples, conditioned on those entities.

Every triple element is normalised by `text_processing` (`utils/misc_utils.py:91`): casefold, punctuation → space, whitespace collapsed. So `"Chris Evans (actor)"` becomes `"chris evans actor"`. Identical strings anywhere in the corpus therefore become **the same node**. That is the only entity resolution, apart from synonym edges.

### 1.2 Three embedding stores (`index`, l.488)

| store | content | used for |
|---|---|---|
| `chunk_embeddings` | passage text | dense passage retrieval (DPR) scores → passage seeds |
| `entity_embeddings` | each unique subject/object phrase | building synonym edges (indexing only) |
| `fact_embeddings` | `str((s, p, o))` of each unique triple | query → fact matching → entity seeds |

The fact text is literally the Python tuple string, for example `"('chris evans', 'known for', 'superhero roles')"`. Our run uses voyage-4-nano with 2,048 dims for all three.

### 1.3 Nodes (`add_new_nodes`, l.1510)

- **Entity (phrase) nodes:** every distinct subject or object string (`extract_entity_nodes`, `misc_utils.py:122`). NER entities that never appear in a triple do **not** become nodes.
- **Passage nodes:** one per chunk.
- **Not nodes:** predicates and facts.

### 1.4 Edges

All edges are undirected (`is_directed_graph=False`), and each pair of nodes has one edge. A pair can carry several roles at once, for example `fact+synonym`.

| edge | built in | between | weight |
|---|---|---|---|
| **fact** | `add_fact_edges` (l.1168) | subject ↔ object of each triple | number of extractions of this (subject, object) pair across the corpus (the predicate is ignored, so two different relations between the same pair add up) |
| **passage** | `add_passage_edges` (l.1219) | passage ↔ every entity in that passage's triples | 1.0 |
| **synonym** | `add_synonymy_edges` (l.1267) | entity ↔ entity | cosine of the entity embeddings |

How synonym edges are built:
- KNN over all entity embeddings (k=2047, on the GPU), keeping neighbours with cosine ≥ **0.8**, at most 100 per node.
- Entities with 2 or fewer alphanumeric characters are skipped.

When a pair has several roles, `add_new_edges` (l.1540) sets the weight to `max(fact count, 1.0 if passage, synonym score)`.

Passages are never connected to each other directly. Two passages are linked only through a shared entity (passage → entity → passage), or through entity → synonym → entity.

### 1.5 What our graphs actually look like

| | HotpotQA | MuSiQue |
|---|---|---|
| nodes | 107,479 (9,811 passages + 97,668 entities) | 113,993 (11,656 + 102,337) |
| edges | 370,184 | 423,934 |
| passage edges | 148,232 | 158,219 |
| fact edges | 133,680 | 138,139 |
| synonym edges | 85,219 | 125,196 |
| fact+synonym | 3,053 | 2,380 |
| fact weight, median / p99 / max | 1 / 2 / 28 | 1 / 3 / 29 |
| synonym weight, median / p90 | 0.835 / 0.904 | 0.834 / 0.902 |
| entity degree, median / p99 / max | 3 / 36 / 1,465 | 4 / 62 / 2,195 |
| passage degree, median / p99 | 14 / 39 | 12 / 39 |
| unique facts (fact store) | 144,261 | n/a |

The biggest hubs are generic entities: `american` (degree 1,465), `united states` (1,377), `2010`, `england`, `new york city`. Keep these in mind for section 4.

Illustrative sketch of a small piece of the graph (the structure is real; the specific nodes and weights are made up):

```
   [passage: "The Newcomers (film) ... starring ... Chris Evans"]
        |  passage edge (w=1)                         \ passage edge (w=1)
   (the newcomers) ──fact (w=1)── (chris evans) ─────(kate bosworth) ...
                                     |  \
                     synonym (w=0.87) |   passage edge (w=1)
                                     |      \
                             (chris evans actor)   [passage: "Chris Evans (actor) ... known for his
                                                               superhero roles as the Marvel Comics ..."]
```

---

## 2. Query time: the whole retrieval pipeline (`retrieve`, l.693)

Every query runs these steps:

```
query ──embed──► cosine with ALL fact vectors ──min-max──► top-5 facts (linking_top_k=5)
                                                              │
                                     LLM filter ("recognition memory", DSPy prompt, rerank.py)
                                                              │ keeps 0..5 facts
                        ┌─────────────────────────────────────┴───────────────┐
                 0 facts left                                          ≥1 fact left
                        │                                                      │
              plain dense passage retrieval                  graph_search_with_fact_entities (l.1997)
              (no graph at all)                               ├─ entity seeds from the facts' subjects/objects
                                                              ├─ passage seeds from DPR × 0.05 (ALL passages)
                                                              └─ PPR over the whole graph, damping 0.5
                                                                     │
                                                         rank all passages by PPR score
                                                                     │
                                       top 200 kept (retrieval_top_k); top 5 go to the reader (qa_top_k)
```

### 2.1 Fact scoring (`get_fact_scores`, l.1881)

- `scores = fact_embeddings · q` over all ~144k facts, then **min-max normalised** to [0, 1] across all facts. The best fact always gets 1.0 and the worst 0.0.
- In upstream with NV-Embed, the query is embedded with the instruction "Given a question, retrieve relevant triplet facts that matches this question." Our vLLM embedder sends the raw question without the instruction.

### 2.2 Recognition memory (`rerank_facts`, l.2113 → `rerank.py`)

- The top 5 facts by cosine go to the LLM with a DSPy-optimised few-shot prompt (`filter_llama3.3-70B-Instruct.json`). The prompt asks the LLM to return only the facts relevant to the question.
- The returned facts are fuzzy-matched back to the candidates with `difflib`.
- The filter can only **remove or reorder** facts. It can never add one.
- If nothing survives, HippoRAG falls back to plain DPR for that query. The graph is not used at all.

### 2.3 Seeds = the PPR reset vector (`graph_search_with_fact_entities`, l.1997)

This is the heart of the method. A weight vector over **all** graph nodes is built from two parts.

**(a) Entity (phrase) seeds.** For each surviving fact `(s, p, o)` with normalised score `f`:

```
for phrase in (s, o):                         # the predicate is ignored
    w[phrase] += f / (#passages that contain phrase)    # ent_node_to_chunk_ids
    n[phrase] += 1
w = w / n                                      # average over the facts mentioning that phrase
keep only the top-5 phrases by w (get_top_k_weights), zero the rest
```

- Dividing by the number of passages that contain the phrase is an **inverse-document-frequency penalty**. An entity in 1 passage keeps its full score; `united states`, in hundreds of passages, gets almost nothing. This stops hubs from dominating.
- There are at most 5 facts, so at most 10 phrases, cut to the top 5. In practice ~5 entity seeds per query (measured mean 4.98).

**(b) Passage seeds.** DPR scores every passage (`passage_emb · q`, min-max normalised to [0, 1]) and gives **every passage node** the weight `0.05 × dpr` (`passage_node_weight=0.05`). Only the single lowest-scoring passage gets exactly 0.

**(c) The reset vector** is `phrase_weights + passage_weights`. igraph normalises it to sum to 1.

### 2.4 PPR itself (`run_ppr`, l.2166)

```python
self.graph.personalized_pagerank(
    vertices=range(N), damping=0.5, directed=False,
    weights='weight', reset=reset_prob, implementation='prpack')
doc_scores = pagerank_scores[passage_node_idxs]      # only passage nodes are read out
```

- Passages are sorted by score. The top 200 are kept, and the top 5 go to the QA reader.
- Entity nodes also receive PPR scores, but those scores are thrown away.
- **It runs once per query, over the entire graph.** It is not a local subgraph expansion around the seeds. PRPACK solves the linear system directly (no fixed iteration count), so the result is the exact stationary distribution.

---

## 3. What PPR computes, in plain words

Imagine a random walker on the graph:

1. It starts at a node drawn from the reset vector: one of the ~5 seed entities, or (more often, see section 5) a random passage weighted by its DPR score.
2. At each step, with probability **0.5** (the damping) it moves to a neighbour, and with probability **0.5** it **teleports back** to a fresh start node drawn from the reset vector.
3. When it moves, it chooses neighbour `v` of `u` with probability **w(u,v) / Σ_x w(u,x)**.

A node's PPR score is the fraction of time the walker spends there in the long run. Written as an equation, with `r` the normalised reset vector and `W` the row-normalised weighted adjacency matrix:

```
π = 0.5 · r + 0.5 · Wᵀ π          ⇒          π = 0.5 · Σ_k (0.5 · Wᵀ)^k · r
```

So the score is a sum over walks of length k, each discounted by 0.5^k:
- **k = 0 (50% of the mass):** the restart itself. A passage's own DPR seed counts directly.
- **k = 1 (25%):** nodes one hop from a seed. For an entity seed, those are the passages that mention it, plus its fact and synonym neighbours.
- **k = 2 (12.5%):** two hops, for example seed entity → passage → another entity in that passage, or seed entity → fact edge → entity → … The mass halves at every hop.

**Damping 0.5 is very low.** Classic PageRank uses 0.85, which gives an average walk of ~5.7 steps; here the average walk is **1 step**. HippoRAG uses PPR as a **"1–2 hop neighbourhood smoother"** around the seeds, not as a global centrality measure.

Multi-hop happens like this. A seed entity from fact 1 spreads to the passages that mention it (1 hop). A seed entity from fact 2 spreads to *its* passages. A passage that mentions both seeds, or sits next to both, collects mass from both and rises to the top. That is the "associative" behaviour of the hippocampal-index analogy.

---

## 4. "How do they balance the transition matrix?" They don't. The balancing is all on the seed side.

### 4.1 The transition matrix is just weighted-degree normalisation

`P(u→v) = w(u,v) / strength(u)`. There is no per-type normalisation and no learned weights. With the weights in section 1.5:

**From a passage node** (median 14 entity neighbours, all w=1): uniform, ≈ 1/14 to each entity it mentions.

**From an entity node:** its neighbours are a mix of:
- passages (w=1 each);
- fact neighbours (w = extraction count, almost always 1);
- synonym neighbours (w ≈ 0.8–0.9).

So it is **almost uniform over all neighbours, regardless of type**. A typical entity with degree 3 might send 1/3 to its passage, 1/3 to its fact partner and 1/3 to a synonym. Repeated triples, for example the same (subject, object) pair extracted from 5 passages, give a heavier fact edge.

**Hubs:** from `american` (degree 1,465), the walker spreads its mass so thinly that it carries almost nothing to any single neighbour. Hubs do still *collect* mass, because many nodes point at them. That is harmless, though, because only passage scores are read out.

The undirected projection is deliberate (`directed=False`, l.2196). Mass flows both ways along every fact, so a question can reach an entity from either its subject or its object side.

### 4.2 Where the real balancing happens

| knob | value | effect |
|---|---|---|
| IDF on entity seeds | `fact_score / #passages(entity)` | rare, specific entities seed strongly; generic ones barely at all |
| `linking_top_k` | 5 | at most 5 facts → at most 5 entity seeds |
| LLM filter | removes irrelevant facts | removes wrong seeds; if all are removed, the graph is skipped entirely |
| `passage_node_weight` | 0.05 | the size of each passage's DPR seed |
| damping | 0.5 | short walks: the graph only reaches ~1–2 hops from the seeds |
| min-max normalisation | facts and passages both mapped to [0, 1] | puts fact and passage scores on the same scale before the 0.05 factor |

---

## 5. Measured: who actually drives the ranking?

PPR is **linear in the reset vector**. That makes it possible to split the final score exactly into "the part from walks that started at entity seeds" and "the part from walks that started at passage seeds".

I re-ran HippoRAG's exact seeding and PPR on the saved HotpotQA graph for the first 100 questions, with these settings:
- the stored passage and fact vectors;
- the query encoded locally with the same voyage-4-nano model (the check reproduced the stored passage vectors at cosine 1.0000);
- the top-5 facts by cosine **without** the LLM filter (it can only remove facts, so this is the "filter kept everything" case).

| quantity (100 questions) | mean | median |
|---|---|---|
| entity seed mass (sum of weights) | 3.45 | 3.70 |
| passage seed mass (sum of 9,811 × 0.05 × dpr) | 132.1 | 129.9 |
| **share of restart mass on entities** | **2.6%** | 2.8% |
| **share of the top-5 passages' final score that comes from entity-seeded walks** | **68%** | 73% |
| Recall@5, full HippoRAG PPR | 0.830 | |
| Recall@5, plain DPR | 0.825 | |
| Recall@5, PPR from entity seeds only | 0.760 | |
| Recall@5, PPR from passage seeds only | 0.810 | |
| top-5 passages shared with DPR's top-5 | 3.2 of 5 | |

How to read this:

- **The passage seeds act as a broad, flat prior.** Each passage's DPR seed is tiny, but there are ~10k of them, so they carry ~97% of the restart mass. Their effect is close to "start from DPR". On its own it gives almost DPR's Recall@5 (0.81 vs 0.825).
- **The entity seeds are a sharp signal.** Only ~2.6% of the mass, but concentrated on ~5 nodes. Each of those nodes is adjacent to only a few passages, because of the IDF penalty and because specific entities have low degree. That mass lands on a handful of passages, so it **decides the top of the ranking**: ~68% of the top-5's score comes from entity-seeded walks.
- **Together** they beat both parts alone: 0.830 vs 0.76 and 0.81. On this sample the gain over DPR is small, and about 3 of the top-5 passages are the same as DPR's. The graph's main contribution is **reordering a few passages near the top**, pulling in a bridge passage that mentions a seed entity but was not lexically similar to the question.
- The 0.05 is effectively a **mixing knob between "trust DPR" and "trust the graph"**. Raising it would push HippoRAG toward plain DPR; lowering it would make the ranking depend more on the 5 entity seeds. The paper tunes it, and 0.05 is their chosen value.

For reference, the full HippoRAG run on our 1,000 HotpotQA questions (`outputs/hotpotqa/upstream_metrics.json`) reached **Recall@5 = 0.880** with the LLM filter on. The 100-question probe above is a smaller, filter-free sample, so its absolute numbers are lower. Use it for the *split*, not the level.

---

## 6. Answers to your questions

**"It's just normal triples, right, and something to do with chunks?"**
- Triples give **subject–object edges between entity nodes**. The predicate is dropped from the graph and survives only inside the fact embedding used for seeding.
- Chunks are **first-class nodes**, linked to every entity in their triples.
- Synonym edges (cosine ≥ 0.8) connect entity strings that are different but similar.
- Facts are never nodes.

**"How do they balance the transition probability matrix?"**
- They don't, beyond the standard row normalisation by weighted degree. Weights are ≈1 for passage and fact edges and ≈0.83 for synonym edges, so moves are almost uniform over neighbours.
- The balance is set by the **reset vector**:
  - IDF-penalised entity seeds;
  - DPR × 0.05 passage seeds on every passage;
  - and by **damping 0.5**, which keeps walks to ~1–2 hops.

**"Do they run PPR for every query from a set of top-k nodes selected by similarity first?"**
- **Yes, PPR runs once per query on the whole graph.**
- The seeds are selected **indirectly**:
  1. query → top-5 **facts** by embedding similarity;
  2. LLM filter;
  3. those facts' subject and object entities, at most 5, weighted by fact score / passage frequency.
- **All passages** are added as weak seeds by DPR score.
- The query is never matched to entity nodes directly. That was HippoRAG 1, which ran NER on the query and linked the named entities to nodes. HippoRAG 2 replaced it with query-to-fact matching.
- If the filter removes every fact, there is no PPR, just DPR.

---

## 7. Relevance to UNH (brief)

- HippoRAG's graph is shallow, and its PPR is effectively a 1–2-hop diffusion added on top of DPR. Most of its multi-hop power comes from **(i) fact-level seeding with an LLM filter** and **(ii) passage nodes sitting directly in the graph**, so a seed entity reaches its passages in one hop.
- UNH's beam search over facts is a much more targeted multi-hop traversal. HippoRAG's advantage is that **every passage is scored for every question in one exact linear solve**. On our graph that is fast and needs no per-hop decisions.
- If UNH ever wants a cheap global reranker, the HippoRAG recipe maps over directly:
  - restart = IDF-weighted seed facts or entities, plus a small DPR prior on chunks;
  - damping ≈ 0.5;
  - read out the chunk-node scores.
