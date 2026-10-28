# QuALITY: one pooled graph or one graph per article? A fairness review for UNH

Date: 2026-10-10.

**Question:** QuALITY's articles are independent stories and no question spans two articles. Should UNH build one pooled graph over all 115 dev articles, or one graph per article, combined by disjoint union? And how did the baselines and the published protocols handle this?

## 1. Short answer

- **Every baseline in our benchmark was built as one pooled index over all 115 articles.** Most of them merge entities across stories by name. This is verified in their code (section 2).
- **There are two established QuALITY protocols in the literature** (section 3):
  - **per-article** (original QuALITY, RAPTOR): each question is answered from its own article;
  - **pooled corpus** (DIGIMON / HyGRAG): one index over all articles, and retrieval must find the right story.
- **Our harness follows the pooled protocol**, as recorded in `QUALITY_IMPLEMENTATION.md`.
- **Recommendation:** build UNH as **one pooled graph** for the main result, exactly as the baselines were built. A per-article UNH graph would use document boundaries that no baseline used during construction, which is a fairness exposure. If you want the per-article number, report it only as a clearly labelled ablation (section 5).

## 2. What each baseline actually did on QuALITY (verified in code)

| System | Index | Cross-story behaviour (evidence) |
|---|---|---|
| HippoRAG 2 | one index over 652 chunks | Entity nodes are exact normalised strings, so "john" in every story is **one node**. Synonym edges (cosine ≥ 0.8) also link similar names across stories (`HippoRAG.py`: `add_fact_edges`, `add_synonymy_edges`). |
| HyGRAG | one `digimon.insert(corpus)` over all articles (`HyGRAG/main.py:74`) | Nodes are keyed by `entity_name` and merged, with descriptions re-summarised across sources (`Core/Graph/BaseGraph.py:72`, `_merge_nodes_then_upsert`). |
| HyperGraphRAG | one `rag.insert` over 115 strings | The same LightRAG-style merge by entity name (`hypergraphrag/operate.py:167`). |
| MS GraphRAG | one workspace over 115 `.txt` files | Entities are merged by `(title, type)` across all text units (`graphrag/index/operations/extract_graph/extract_graph.py:100`). Communities can span stories. |
| MemGraphRAG | one file of all 115 articles | HippoRAG-style string nodes plus synonym edges (`MemGraphRAG.py:1441`). Its 256-token windows even cross article boundaries (≤114 of ~3,000 windows). |
| LogicRAG | one chunk index over 651 chunks | No entity graph, so nothing to merge. Retrieval is pooled over all chunks. |

So **no baseline separated stories at construction time.** The ones with entity graphs all conflate same-named characters across stories. That's part of what the pooled protocol tests.

## 3. How published work builds QuALITY

| Work | Protocol | Evidence |
|---|---|---|
| **QuALITY** (Pang et al., NAACL 2022) | per-article reading comprehension: each question comes with its own article | dataset design |
| **RAPTOR** (Sarthi et al., ICLR 2024) | **per-article**: one tree per story, and each question is retrieved from its own story's tree | "The RAPTOR tree is built for each of these stories"; token cost and build time are reported per document length (arXiv 2401.18059, §4 and Appendix) |
| **DIGIMON** (Zhou et al., PVLDB 2025): "In-depth Analysis of Graph-based RAG in a Unified Framework" | **pooled**: QuALITY is listed as one corpus of 1,522,566 tokens, 4,609 questions and 265 documents (Table 4). Each method builds its graph from "the corpus D" | arXiv 2503.04338, §7.1, Table 4 |
| **HyGRAG** (WWW 2026; a DIGIMON fork) | **pooled**: DIGIMON's QuALITY copy (train + dev, 265 articles), one `insert` | `HyGRAG/main.py`; `Data/quality` |
| **Plan\*RAG** (Verma et al., 2024) | **no QuALITY evaluation found.** The paper evaluates multi-hop QA (HotpotQA and others) with a planning DAG; it doesn't build a graph index over a corpus | arXiv 2410.20753; Microsoft Research page |

No paper we found built per-article graphs for *some* systems and a pooled graph for others. Within a paper, every system follows the same protocol: RAPTOR's comparisons are all per-article, and DIGIMON's are all pooled.

## 4. Why per-article UNH would be a fairness exposure

- **Pooled UNH conflates stories, just as the baselines do.** Entity dedup, fact dedup and cross-chunk fact edges could link "John" in story A with "John" in story B. That is the same kind of cross-story noise HippoRAG, HyGRAG, HyperGraphRAG, MS GraphRAG and MemGraphRAG already have through name-based merging.
- **UNH's consolidation may even conflate less than theirs.** Its entity dedup asks the LLM, with each entity's role, whether two entities are the same, while HippoRAG merges identical strings unconditionally. So pooled UNH isn't obviously disadvantaged.
- **A per-article union removes that noise for UNH only.** It uses document boundaries at construction, which every baseline received (separate documents) but none used. Even with retrieval still pooled, a reviewer could reasonably say UNH was given a cleaner graph by the protocol, not by its method.
- **Making it symmetric would be expensive and would change the task.** It would mean per-article indexes for all six baselines: 115 builds each, with MS GraphRAG's cost alone at about 24M input tokens. And if every system were per-article, retrieval should then also be per-article (the RAPTOR protocol). That's a different benchmark from the one we've already run four baselines on.

## 5. Recommendation

1. **Main result: one pooled UNH graph over all 115 dev articles.**
   - Same input as the baselines: 115 separate `title\ntext` documents.
   - UNH's own chunker (600/50) within each article, so chunks never straddle articles. This matches HyGRAG, HyperGraphRAG and MS GraphRAG, which also chunk per document.
   - UNH's normal global consolidation (`clustered_fast`).
   - PPR `t2000_f40_dmp06` unchanged, all three views.

   This is exactly the protocol every baseline followed. There's nothing to defend.
2. **Optional ablation, clearly labelled:** per-article UNH graphs combined by disjoint union, retrieved pooled with no article ID. Report it as "document-scoped consolidation", next to the main pooled number, with the disclosure that it uses document boundaries at construction time. Its value is scientific: it measures how much cross-story conflation hurts a consolidating graph. It's not a headline result.
3. **Don't mix protocols in one table.** If a per-article (RAPTOR-style) comparison is ever wanted, every system has to be rebuilt and evaluated per article.

## 6. Implementation note

UNH node IDs are random UUIDs (`nkg/models/Graph.py`), so the ablation's disjoint union is collision-free. The pooled main build needs only a per-document chunking mode. Today `concat_file` would let chunks cross article boundaries, and `prepared_chunks` would treat each whole article as one chunk.

## Sources

- RAPTOR: arXiv 2401.18059 (ICLR 2024), §4 "Evaluation Datasets"; layer ablation ("The RAPTOR tree is built for each of these stories"); Appendix "Scalability".
- DIGIMON: arXiv 2503.04338 / PVLDB 18(13):5623–5637, §7.1 and Table 4.
- Plan\*RAG: arXiv 2410.20753; https://www.microsoft.com/en-us/research/publication/planrag-efficient-test-time-planning-for-retrieval-augmented-generation/
- Code: `HyGRAG/main.py`, `HyGRAG/Core/Graph/BaseGraph.py`, `evaluation_scripts/hypergraphrag/hypergraphrag/operate.py`, graphrag `extract_graph.py`, `evaluation_scripts/hipporag2/src/hipporag/HippoRAG.py`, `evaluation_scripts/memgraphrag/src/MemGraphRAG.py`.
