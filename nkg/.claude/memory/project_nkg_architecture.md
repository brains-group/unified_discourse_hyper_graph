---
name: NKG Architecture Overview
description: Full architecture of the Networked Knowledge Graph (nkg) package - data flow, DSPy signatures, Pydantic models, and how extraction, node construction, and edge construction fit together.
type: project
---

# NKG (Networked Knowledge Graph) Architecture

**Why:** The user is building a unified discourse + hypergraph architecture. Understanding the full pipeline is essential for any optimization or extension work.

**How to apply:** Reference this when modifying extraction, graph construction, or retrieval. Every change must keep `build_index_from_file` and `build_index_from_directory` working end-to-end.

---

## Directory Structure

```
nkg/
├── models/
│   ├── index_objects.py     — Pydantic graph node types (EntityFingerprint, Fact, Chunk)
│   └── Graph.py             — NetworkX DiGraph wrapper with UUID-keyed node dicts
├── index/
│   ├── extraction/
│   │   ├── extract_entities.py       — DSPy signatures for entity fingerprinting
│   │   ├── extract_facts.py          — DSPy signatures + modules for fact assembly
│   │   ├── extract_chunk_features.py — DSPy signatures + ChunkAssembler module
│   │   └── extract_relations.py      — DSPy signatures for edge construction
│   └── construction/
│       ├── construct_nodes.py        — initialize_graph_from_text()
│       ├── construct_edges.py        — all edge construction functions
│       └── build_index.py            — build_index_from_file / _from_directory
├── deduplication/
│   └── entity_deduplication.py      — GraphDeduplicator (embedding cluster + LLM merge)
├── retrieval/                        — retrieval engine, planner, scorer, traversal
└── utils/                            — chunking, config, math_utils, general
```

## Core Data Models (index_objects.py)

- **EntityFingerprint** (frozen): `name`, `type`, `role` (3-8 word micro-role), `relational_anchors: Tuple[str,...]`
- **Fact** (frozen): `name`, `sentence`, `macro_topics`, `chunk_topics`, `answered_questions`, `follow_up_questions`, `entities: List[EntityFingerprint]`
- **Chunk** (frozen): `name`, `text`, `summary`, `topics`, `entities: List[EntityFingerprint]`, `facts: List[Fact]`

## Extraction Pipeline (per chunk)

1. `ChunkDescription` DSPy sig → name, topics, summary
2. `ChunkFacts` DSPy sig → fact_sentences: list[str]
3. `ExtractChunkEntities` DSPy sig → chunk_entities: list[EntityFingerprint]  (once per chunk)
4. `BatchedFactAssembler.forward(source_text, fact_sentences, chunk_entities, batch_size)`:
   - Formats numbered facts (1..N within each batch) + entity_names list
   - `BatchedFactMetaExtractor` DSPy sig → list[FactBatchItem] (with fact_id, metadata, entity_names subset)
   - `_link_entity_names_to_fingerprints()`: exact string match then TF-IDF cosine similarity fallback
   - Returns list[Fact] with only the relevant subset of chunk entities per fact

## Graph Node Types

- **chunk** nodes: Chunk Pydantic object
- **fact** nodes: Fact Pydantic object  
- **entity** nodes: EntityFingerprint Pydantic object

## Edge Types

- `chunk_fact` (chunk → fact): created in `graph.add_chunk()`
- `fact_entity` (fact → entity): created in `graph.add_fact()` via `add_entities(fact.entities)`
- `entity_fact` (entity → fact): created by `EntityFactEdge` DSPy sig in `construct_initial_edges()`
- `fact_fact` (fact ↔ fact): created by `FactEdge` DSPy sig in `construct_edges_during_merge()` using multi-dimensional embeddings + rank fusion
- `chunk_chunk` (chunk ↔ chunk): optional, created in non-linear mode

## Key Graph Methods

- `graph.add_chunk(chunk, cascading=True)` — adds chunk, cascades to facts and entities
- `graph.add_fact(fact, cascading=True)` — adds fact, adds entities, creates fact_entity edges
- `graph.init_fact_embeddings(retrieval_model)` — 4 embedding dimensions: sentence, topics, entity roles, follow-up questions
- `graph.get_relevant_seeds(fact, get_all=True)` — rank fusion across 4 dims for retrieval

## Top-Level Entry Points

- `build_index_from_file(filepath, ...)` — chunks file, parallel chunk processing, global merge, linear fact edges
- `build_index_from_directory(directory, ...)` — pyramid merge-sort across multiple files

## Performance Design

- Entity extraction: once per chunk (not once per fact) → shared entity pool
- Fact extraction: batched (N facts per LLM call, configurable via `fact_batch_size`)
- Entity-fact edges: parallel ThreadPoolExecutor
- Fact-fact edges: multi-dimensional vector search + parallel LLM calls
- Deduplication: embedding clustering (Leiden algorithm) then LLM cluster resolution
