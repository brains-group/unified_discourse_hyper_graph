# Hop-guided PPR for UNH: review of the idea and an implementation plan

Status: design only, nothing implemented. Written 2026-10-10. It builds on:
- the current `nkg/retrieval/ppr_engine.py`: RRF fact relevance plus coverage path selection;
- the 250-question HotpotQA results so far.

## 0. Summary

- **The core idea is sound,** and it fits PPR better than beam search does:
  - give PPR a plan that says *where to start* and *how to move at each hop*;
  - let the graph structure do the rest.
- **Two premises need correcting** (section 2):
  1. **Discourse edges are not drowned out.** On our graphs they already carry about **half** of the walk mass leaving a fact. What does penalise entity routes is the **step count**: an entity hop takes 2 PPR steps, so it is damped twice.
  2. **In UNH, the relation you want to follow ("grandfather of", "born on") is a fact node, not an edge.** Fact→fact labels are rhetorical (CORROBORATES, ELABORATES_ON, PRECEDES, …). "Guide the edges" therefore means: bias **entity → fact** moves toward facts that express the hop's relation, and **fact → entity** moves toward entities of the expected type. Only discourse intents (cause, time order, contrast) map onto fact→fact labels.
- **The "broad later hops" problem has a clean solution:**
  - Never put hop-2+ characteristics into the **restart** vector; that is what would highlight every grandfather in the graph.
  - Put them into the **transitions**, or into **restarts that are multiplied by the previous hop's PPR scores**.
  - Mass only leaves the seeds, so a transition bias only matters where the walk actually goes. It is anchored automatically.
- **PPR has no memory of how many hops it has taken.** A single PPR cannot express "prefer *grandfather* moves only after the astronaut hop". The fix is a **cascade**: one short PPR per hop, each restarting from the previous hop's results, weighted by how well each node matches the next hop. A layered (staged) graph is the exact version of the same thing.
- **This needs one planner call per question,** as UNH's current retriever already makes. Before building the planner, we can measure the mechanism's ceiling for free: MuSiQue ships the gold hop decomposition of every question (section 7, E2).

---

## 1. Where we are

On 250 HotpotQA questions with the full graph and no LLM, measured as answer in context:

| | answer in context | chars | distinct facts | s/question |
|---|---|---|---|---|
| official UNH (planner + beam search) | 79.6% | 3,822 | 18.1 | 385 |
| PPR v1 `base` | 65.6% | 4,167 | 17.9 | 7.6 |
| PPR `cov_s30`: RRF relevance + coverage selection, 30 path starts | **80.4%** | 4,707 | 34.8 | **2.2** |
| PPR `cov_s30_d2`: paths of up to 2 facts | 80.0% | **4,052** | 32.4 | 2.0 |
| PPR `cov_relppr`: PPR-only relevance | 76.8% | 4,704 | 34.9 | 0.9 |

What the diagnostic showed:
- **The candidate pool was never the problem.** The PPR top 150 facts contain the answer 85.6–88% of the time.
- **PPR is worse than the cross-encoder at the very top** (top 10: 54% vs 59%) but better at breadth (top 40: 78% vs 73%).
- **PPR ranks chunks better than plain embedding retrieval** (top 5: 87% vs 80%).

Hop guidance targets the remaining gap: getting the *later-hop* facts (the bridge entity's fact, the final attribute) into the top 10–20, not just somewhere in the top 150.

---

## 2. Checking the premises against the real graphs

### 2.1 How the walk splits at a fact node

The current PPR graph is undirected. Fact→fact weight = the LLM edge score (0.3–0.9); entity–fact and chunk–fact edges = 1. The split of a fact's outgoing walk mass:

| | HotpotQA | MuSiQue |
|---|---|---|
| to discourse (fact→fact) neighbours, mean | **49.2%** | **43.3%** |
| to entities, mean | 31.3% | 34.9% |
| to its chunk, mean | 19.5% | 21.8% |
| discourse share p10 / p50 / p90 | 0.29 / 0.50 / 0.69 | 0.23 / 0.43 / 0.63 |
| facts with no discourse edge | 0.4% | 0.8% |
| facts per entity, median / p90 / p99 / max | 2 / 10 / 26 / 5,017 | 2 / 8 / 30 / 1,584 |

Why this differs from the intuition:
- **PPR normalises per node.** What matters is the *local* mix of edge types at each node, not the global edge counts.
- Every fact has about 4.5 discourse neighbours but only about 1.8 entities and 1 chunk, so discourse already gets about half the mass. (The discourse *view* was fast for a different reason: beam search over sparse fact→fact edges.)
- A global rule such as "all discourse edges together get 40–50%" already holds, roughly.

**What actually penalises entity routes** is not the weights:

1. **Step count.** A bridge through an entity is fact → entity → fact: 2 steps, each surviving with probability d = 0.5, so ×0.25. A discourse hop is ×0.5.
   - A 3-hop chain through entities is 6 steps, ≈ 1.6% of the mass, against 12.5% through discourse edges.
   - This is the main reason PPR finds 1-hop neighbours well and 3-hop answers poorly. It is the mathematically important point in this design.
2. **Leakage into chunks.** About 20% of the mass at every fact goes to its chunk, then back to the other facts of the same chunk. This largely duplicates the `same_chunk` discourse edges, so it is a same-passage diffusion, not new information.
3. **Hubs.** An entity with 5,017 facts spreads whatever reaches it over 5,017 facts, about 0 each. With guidance this flips: a hub becomes a *router* that sends mass to the one fact matching the hop's relation (section 4.3).

### 2.2 The relation lives in the fact

Edge vocabularies on the MuSiQue graph (HotpotQA is the same):

| edge | distinct labels | most common |
|---|---|---|
| fact→fact | 10,023 | CORROBORATES, PROVIDES_CONTEXT_FOR, ELABORATED_BY, ELABORATES_ON, CONTEXTUALIZED_BY, CONTRASTS_WITH, PRECEDES, FOLLOWS |
| fact→entity | 71,372 | the entity's role in the fact: "starring actor", "album being described", "main subject of the text", … |
| entity→fact, chunk→fact | 1 (empty) | – |

A question like "the grandfather of X" is answered by a **fact node** whose sentence says "Y is the grandfather of X", plus the entity Y with a role such as "grandfather". The hypergraph view makes this explicit: an entity→fact move *is* the choice of a relation.

The guidance signals per hop therefore come from:

| plan field | matched against (embeddings already cached in `EfficientRetriever`) | biases |
|---|---|---|
| relation ("is the grandfather of") | fact sentence, answered_questions | entity → fact moves, and the next hop's restart |
| target type ("person", "date", "company") | entity name + role, fact→entity role label | fact → entity moves |
| answer form (date / number / person / place) | regex on the fact sentence (dates, numbers) + entity role similarity | final ranking |
| discourse intent (cause, time order, contrast) | the 10,023 fact→fact label strings | fact → fact moves |

---

## 3. The mathematical constraints the design must respect

1. **The restart vector says where mass comes from; transitions say how it moves.** Hop-1 (the specific anchor) belongs in the restart. Hops 2..K must not be, because a restart on "grandfather" nodes creates mass at every grandfather in the corpus. This is exactly the failure you described.
2. **A transition bias is anchored automatically.** π(v) is the sum over walks from the restart nodes to v. Boosting the weight of every grandfather-type move changes nothing for walks that never get near those moves. A biased edge far from the seeds receives almost no mass, so biasing transitions globally is safe.
3. **Row normalisation makes biases relative.** Multiplying a node's outgoing weights by relevance only changes the choice *among that node's neighbours*. A node with no relevant neighbour still passes its mass on in full. To stop mass flowing into irrelevant regions, the continuation probability itself must depend on relevance, through relevance-gated damping (4.3), or through the cascade's multiplicative gating (4.2).
4. **PPR is memoryless.** The transition from a node cannot depend on how many hops have been taken. "Hop-2 moves only after hop 1" therefore needs either:
   - a **layered** state space (node, stage), or
   - a **cascade** of PPRs.

   A single biased PPR still helps, but it applies every hop's bias at every step.
5. **The damping budget limits depth.** With d = 0.5, mass decays by half per step. The cascade sidesteps this: each hop is its own short PPR with its own restart, so depth no longer compounds the decay.
6. **Biased PPR is not symmetric.** Once transitions are query-biased, P is no longer the row-normalised version of a symmetric matrix. Power iteration with Pᵀ still converges (P is stochastic and the restart term keeps it ergodic); we just cannot use the W·(π/deg) shortcut. Per question that means one element-wise re-weighting of ~2M edge values and the usual ~30 sparse multiplications per PPR: milliseconds.

---

## 4. Design

### 4.1 The hop plan (one LLM call; replaces UNH's current planner call)

```json
{"structure": "chain",                       // chain | parallel (comparison) | dag
 "anchors": [{"text": "first astronaut to walk on the Moon", "type": "person"}],
 "hops": [
   {"relation": "is the grandfather of / paternal grandfather", "target_type": "person",
    "discourse": null},
   {"relation": "was born on (date of birth)", "target_type": "date", "discourse": null}],
 "answer_form": "date"}
```

- **Anchors** are hop 0: the specific, high-precision part. They are matched against entity names and fact sentences (cross-encoder plus cosine, as seeds today).
- Each **hop** describes the *edge you want to take next* and *what you expect to land on*, never the answer itself. This is the "characteristics, not the literal answer" idea.
- **Comparison questions** (`parallel`, about 20% of HotpotQA) have 2 anchors and the same hop applied to each. Each anchor group gets an equal share of the restart mass, so one does not drown the other.
- This replaces, not adds to, the existing `QueryPlan` call. The cost stays at about one call per question. The current planner's `target_edge_labels` / `target_edge_semantics` are unordered; the change here is that the hops are **ordered**.

### 4.2 Cascaded PPR (implement first)

```
r0 = restart on anchor matches (facts + entities)            # specific
π0 = PPR(r0, P_struct + bias(hop 1))
for k = 1..K:
    m_k = match(node, hop k)                                  # global, broad, in [0, 1]
    r_k = normalise( π_{k-1} ⊙ (ε + m_k) )                    # broad × proximity to hop k-1  →  anchored
    π_k = PPR(r_k, P_struct + bias(hop k+1))                  # short walk from hop-k landing nodes
final = π_K ⊙ (ε + answer_form_match)   blended with plain PPR:  λ·final + (1-λ)·π_plain
```

- **This is the "grandfather connected to *this* astronaut" requirement.** `m_k` alone is broad; multiplying by `π_{k-1}` keeps only matches near the previous hop's mass.
- **Each stage restarts,** so a 3-hop answer is not damped to 1.6%.
- **ε (about 0.05–0.1) and λ (about 0.5–0.7) make a wrong plan degrade gracefully** to today's PPR instead of failing.
- **Damping per stage can be lower** (0.3–0.5): each stage only needs 1–2 steps.
- **Cost:** K+1 PPRs (K ≤ 4), each about 30 sparse multiplications over ~2M non-zeros. That is ≪ 1 s, so retrieval stays at about 1–2 s per question.

### 4.3 Transition bias (used inside every stage)

For a move u→v of type t (discourse, fact→entity, entity→fact, chunk):

```
P(u→v) = Σ_t α_t(u) · w(u,v) · b_t(v | hop) / Σ_{v' of type t} w(u,v') · b_t(v' | hop)
b_t(v | hop) = exp(β · match_t(v, hop))        (softmax-style; β about 3–5, tuned)
```

- **α_t(u)**, renormalised over the types present at u, is the **edge-type balance**. It is the right place for "discourse gets 40–50%", but it is applied per node, so it means what it says.
  - Default α = current behaviour.
  - Sweep α_chunk lower (it duplicates same-chunk discourse edges) and α_disc vs α_ent per question type: bridge questions → entity moves, cause/time questions → discourse moves.
- **Relevance-gated damping** (optional): continue with probability d·g(u), where g = how well u's best outgoing move matches the hop, and restart otherwise. This stops mass from leaking through irrelevant regions; plain row normalisation cannot do that (3.3).
- **Entity projection** (optional; answers 2.1-1): replace fact→entity→fact by a direct fact→fact move with weight 1/(#facts of the entity), so an entity bridge costs 1 step like a discourse hop. Run it as an ablation, because the entity node is also where the target-type bias acts.

### 4.4 The exact version: layered PPR (only if the cascade shows ordering errors)

- The state is (node, stage k). The restart is only at stage 0.
- From stage k, a move into a node that matches hop k+1 advances to stage k+1 with probability σ(match); otherwise the walk stays at stage k.
- The answer is the stage-K mass. This is a random walk constrained by a small automaton, as in the Path Ranking Algorithm or regular-path-constrained PPR.
- It is exact where the cascade is approximate (it never applies hop-3 bias during hop 1), at (K+1)× the matrix size: still cheap.

### 4.5 From stage scores to UNH paths

- The cascade gives per-stage node scores π_0..π_K. A path is a **backtrace**: start from a top stage-K fact, step to the neighbour u with the highest π_{K-1}(u)·P(u→v), and so on back to an anchor (Viterbi-like).
- That gives hop-ordered paths (anchor → bridge → attribute), which then go into the existing coverage selector and `format_paths_deduplicated`.
- This replaces the beam search, and the context stays in UNH path form.

---

## 5. How this improves on the original plan

1. **Drop the global "discourse = 40–50%" weighting.** Use per-node type weights α instead. They already sit at about 45–50% for discourse; tune them per question type.
2. **Fix the real imbalance,** which is the step count and damping, with the cascade or entity projection.
3. **Highlight relations through facts and entity roles, not fact→fact labels.** Use fact→fact labels only for discourse intents.
4. **Put only hop 0 in the restart.** Later hops go into transitions and into restarts gated by the previous hop.
5. **Add an ε floor and a λ blend** with plain PPR, so plan errors cost little: there is no LogicRAG-style verification loop to catch them.
6. **Handle comparison questions explicitly,** with parallel anchors and equal restart shares.
7. **Measure the ceiling with gold decompositions** before investing in planner prompts.

---

## 6. Risks

| risk | mitigation |
|---|---|
| the planner gets the hop order or relation wrong | ε, λ blend; check the oracle plan (E2) against the LLM plan (E3) to separate mechanism errors from planner errors |
| match scores are badly calibrated (cosine bunches near 0.3–0.6) | rank- or softmax-based `m_k` (top-N per hop), β tuned on 100 questions and then held fixed |
| a hub entity receives a boosted move to a wrong but similar fact | the cascade's ⊙ π_{k-1} gating; cap entity-seed mass by IDF as today |
| an answer-form regex (dates/numbers) over-fires | only a multiplier (ε + m), never a filter |
| the extra LLM call breaks the "no LLM" property of the current PPR retriever | keep both: `ppr` (no LLM) and `ppr_guided` (one planner call); report both |
| tuning on the 250 subset overfits it | tune on 100 questions, report on the other 150, then the full 1,000 |

---

## 7. Experiments (cheap to expensive; GPU 0; one run at a time)

| id | what | LLM | question |
|---|---|---|---|
| E1 | per-node type weights α (discourse/entity/chunk grid) + entity projection, on the current `cov_s30` | none | do structure-only fixes help? |
| E2 | **oracle hop plan on MuSiQue**: each sub-question of `question_decomposition` as hop k's relation text (with "#1" placeholders and gold sub-answers masked) → cascaded PPR | none | the ceiling: does hop guidance get later-hop facts into the top 20? |
| E3 | LLM hop planner (DeepSeek, plans cached to disk, one call per question) + cascade, on HotpotQA and MuSiQue | 1 call/q | how much of E2's gain does a real plan keep? |
| E4 | layered PPR instead of the cascade | as E3 | only if E3 shows hop-order errors |
| E5 | reader F1 (`answer.py` + `score.py`) for the best 2 configurations vs `cov_s30` and the official run | reader only | the goal metric |

Metrics per run:
- answer in context;
- **supporting-passage recall** (MuSiQue `paragraph_support_idx`, HotpotQA supporting titles), which tells which hop was missed;
- distinct facts, context characters, seconds per question.

E2 isolates the mechanism. If gold decompositions don't raise MuSiQue recall, no planner will.

## 8. What I need from you

1. Approval of the plan schema (4.1), and whether the planner may use DeepSeek per question (E3), or Qwen on :8000.
2. Whether to run E1 and E2 first, which need no LLM, after the other session's sweep finishes. That keeps to one sweep at a time on GPU 0.
