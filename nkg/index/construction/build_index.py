import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

import dspy

from nkg.models.Graph import Graph
from nkg.index.construction.construct_nodes import initialize_graph_from_text
from nkg.index.construction.construct_edges import construct_initial_edges, construct_edges_during_merge
from nkg.index.construction.checkpointing import ConstructionCheckpoint
from nkg.utils.general import batch_list  # Adjust import path if needed
from nkg.utils.config import configure_dspy
from nkg.utils.chunking import chunk_text_by_tokens
from nkg.deduplication.entity_deduplication import GraphDeduplicator
from nkg.deduplication.fact_deduplication import GraphFactDeduplicator
from nkg.utils.token_tracker import HISTORY_SIZE_FOR_TOKEN_COUNTS, usage_tokens

import warnings
warnings.filterwarnings("ignore")


def _stage_tokens(history_idx: int) -> dict:
    """Sums DSPy LM history from history_idx onward into a token-count dict."""
    try:
        entries = dspy.settings.lm.history[history_idx:]
        prompt, completion, cache_hit = 0, 0, 0
        for e in entries:
            p, c, hit = usage_tokens(e.get("usage"))
            prompt += p
            completion += c
            cache_hit += hit
        return {"prompt": prompt, "completion": completion, "total": prompt + completion, "calls": len(entries),
                "cache_hit": cache_hit}
    except Exception:
        return {"prompt": 0, "completion": 0, "total": 0, "calls": 0, "cache_hit": 0}


def process_text_batch(texts: List[str], chunk_size: int, overlap: int, edge_threshold: float, max_workers: int, fact_batch_size: int = 5) -> Graph:
    """
    Worker function for the first layer of the pyramid.
    Takes a batch of texts, initializes graphs for each, merges them,
    and then constructs all edges within the merged local graph.
    """
    merged_graph = Graph()

    # 1. Initialize and merge graphs for all texts in this batch
    for text in texts:
        local_graph = initialize_graph_from_text(text, chunk_size=chunk_size, overlap=overlap, fact_batch_size=fact_batch_size)
        merged_graph.merge(local_graph)

    # 2. Construct edges across the newly merged graph
    construct_initial_edges(merged_graph, threshold=edge_threshold, max_workers=max_workers)

    return merged_graph


def process_graph_batch(graphs: List[Graph], edge_threshold: float, max_workers:int) -> Graph:
    """
    Worker function for the upper layers of the pyramid.
    Takes a batch of already-built graphs, merges them,
    and constructs new edges bridging the disparate graphs.
    """
    merged_graph = Graph()

    # 1. Merge all graphs in this batch together
    for graph in graphs:
        merged_graph.merge(graph)

    # 2. Construct edges (this will find the new cross-graph relationships
    # because the internal ones were already processed and flagged)
    construct_edges_during_merge(merged_graph, threshold=edge_threshold, max_workers=max_workers)

    return merged_graph


def build_index_from_directory(
        directory: str,
        batch_size: int = 5,
        max_workers: int = 4,
        chunk_size: int = 600,
        overlap: int = 50,
        edge_threshold: float = 0.5,
        max_sub_workers=10,
        fact_batch_size: int = 5
) -> Graph:
    """
    Reads all text files in a directory and builds a unified knowledge graph
    using a parallel, hierarchical divide-and-conquer ("merge sort" style) approach.
    """
    # Step 1: Read all text files from the directory
    texts = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                texts.append(f.read())

    if not texts:
        print(f"No .txt files found in {directory}")
        return Graph()

    print(f"Found {len(texts)} files. Starting initial extraction phase...")

    # Step 2: Initial Batching - Layer 1 of the pyramid
    text_batches = batch_list(texts, max_batch_size=batch_size)
    current_graphs = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all text batches to the worker pool
        futures = {
            executor.submit(process_text_batch, batch, chunk_size, overlap, edge_threshold, max_sub_workers, fact_batch_size): i
            for i, batch in enumerate(text_batches)
        }

        for future in as_completed(futures):
            try:
                resulting_graph = future.result()
                current_graphs.append(resulting_graph)
                print(f"Completed initial text batch. Current active sub-graphs: {len(current_graphs)}")
            except Exception as e:
                print(f"Error processing text batch: {e}")

    # Step 3: Hierarchical Merging - Upper layers of the pyramid
    iteration = 1
    while len(current_graphs) > 1:
        print(f"\n--- Starting Merge Iteration {iteration} ---")
        print(f"Graphs to merge: {len(current_graphs)}")

        # Batch the current list of graphs
        graph_batches = batch_list(current_graphs, max_batch_size=batch_size)
        next_level_graphs = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_graph_batch, batch, edge_threshold,max_sub_workers): i
                for i, batch in enumerate(graph_batches)
            }

            for future in as_completed(futures):
                try:
                    resulting_graph = future.result()
                    next_level_graphs.append(resulting_graph)
                    print(f"Merged graph batch completed. Sub-graphs remaining in next level: {len(next_level_graphs)}")
                except Exception as e:
                    print(f"Error processing graph batch: {e}")

        # Move up the pyramid
        current_graphs = next_level_graphs
        iteration += 1

    print("\nIndexing complete! Returning unified Knowledge Graph.")
    # Return the single remaining unified graph
    return current_graphs[0]


def process_single_chunk(chunk_text: str, edge_threshold: float, max_sub_workers: int,
                          fact_batch_size: int = 5, use_cot: bool = True,
                          neutral_description: bool = False, concise_summary: bool = False,
                          combined_entity_extraction: bool = False,
                          full_entity_profile: bool = True,
                          generate_entity_relation_labels: bool = False,
                          same_chunk_min_facts: int = 2,
                          edge_descriptions: bool = True,
                          fact_prompt: str = "default",
                          fact_edge_prompt: str = "default") -> Graph:
    """
    Worker function to process an individual chunk of text.
    Extracts the nodes (Chunks, Facts, Entities) and constructs the
    internal local edges (Entity->Fact, Fact->Fact within the same chunk).
    """
    local_graph = initialize_graph_from_text(
        chunk_text, chunk_size=999999, overlap=0, fact_batch_size=fact_batch_size,
        use_cot=use_cot, concise_summary=concise_summary,
        combined_entity_extraction=combined_entity_extraction,
        full_entity_profile=full_entity_profile,
        generate_entity_relation_labels=generate_entity_relation_labels,
        fact_prompt=fact_prompt
    )
    construct_initial_edges(local_graph, threshold=edge_threshold, max_workers=max_sub_workers,
                             mode="linear", use_cot=use_cot, neutral_description=neutral_description,
                             same_chunk_min_facts=same_chunk_min_facts, edge_descriptions=edge_descriptions,
                             fact_edge_prompt=fact_edge_prompt)
    return local_graph


# def build_index_from_file(
#         filepath: str,
#         model_name: str = "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
#         max_workers: int = 4,
#         chunk_size: int = 600,
#         overlap: int = 50,
#         edge_threshold: float = 0.5,
#         top_k: int = 10,
#         max_sub_workers: int = 10,
#         fact_batch_size: int = 15,
#         edge_mode: str = "linear",
#         run_deduplication: bool = True,
#         dedup_model_name: str = "all-MiniLM-L6-v2",
#         dedup_sim_weights: dict = None,
#         dedup_iterations: int = 1,
#         dedup_max_cluster_size: int = 64,
# ) -> Graph:
#     """
#     Reads a single text file, breaks it into chunks via tokenizer, processes chunks
#     in parallel to build local subgraphs, merges them into a global graph, runs
#     entity deduplication, then constructs cross-chunk fact-to-fact edges.
#
#     edge_mode controls the global fact edge construction strategy:
#       "linear"    — one LLM call per candidate fact pair (original method)
#       "clustered" — groups facts into clusters via similarity + Leiden, then one
#                     bulk LLM call per cluster (much fewer LLM calls total)
#
#     Deduplication parameters:
#       run_deduplication    — set False to skip deduplication entirely
#       dedup_model_name     — sentence-transformers model used for entity similarity
#       dedup_sim_weights    — dict with keys "name", "role", "type" controlling the
#                              blend of embedding dimensions (default: {"name":0.6, "role":0.3, "type":0.1})
#       dedup_iterations     — number of deduplicate-then-re-cluster passes
#       dedup_max_cluster_size — max Leiden cluster size during deduplication
#     """
#     # Step 1: Read the text file
#     if not os.path.exists(filepath):
#         print(f"Error: File not found at {filepath}")
#         return Graph()
#
#     with open(filepath, 'r', encoding='utf-8') as f:
#         full_text = f.read()
#
#     if not full_text.strip():
#         print("Error: The provided file is empty.")
#         return Graph()
#
#     print(f"Reading file: {filepath}")
#
#     # Step 2: Chunk the text by tokens
#     print(f"Chunking text using tokenizer for model: {model_name}...")
#     text_chunks = chunk_text_by_tokens(
#         model=model_name,
#         text=full_text,
#         chunk_size=chunk_size,
#         overlap=overlap
#     )
#     print(f"Created {len(text_chunks)} chunks. Starting parallel node extraction...")
#
#     # Step 3: Process chunks in parallel to build localized graphs
#     local_graphs = []
#     with ThreadPoolExecutor(max_workers=max_workers) as executor:
#         futures = {
#             executor.submit(process_single_chunk, chunk, edge_threshold, max_sub_workers, fact_batch_size): i
#             for i, chunk in enumerate(text_chunks)
#         }
#
#         for future in as_completed(futures):
#             try:
#                 resulting_graph = future.result()
#                 local_graphs.append(resulting_graph)
#                 print(f"Completed chunk processing. Extracted nodes for {len(local_graphs)}/{len(text_chunks)} chunks.")
#             except Exception as e:
#                 print(f"Error processing chunk subgraph: {e}")
#
#     # Step 4: Global Merge
#     # We combine all subgraphs into one master graph BEFORE deduplication and cross-chunk
#     # comparisons so the vector search has access to the entire document's context.
#     print("\n--- Merging all local subgraphs into a Unified Global Graph ---")
#     global_graph = Graph()
#     for local_graph in local_graphs:
#         global_graph.merge(local_graph)
#
#     print(f"Merge complete. Base Graph Size: {global_graph.network.number_of_nodes()} nodes, "
#           f"{global_graph.network.number_of_edges()} edges")
#
#     # Step 5: Entity Deduplication (before global edges so deduped entities feed into vector search)
#     if run_deduplication:
#         print(f"\n--- Running Entity Deduplication ({dedup_iterations} iteration(s)) ---")
#         from sentence_transformers import SentenceTransformer
#         retrieval_model = SentenceTransformer(dedup_model_name)
#         deduplicator = GraphDeduplicator(retrieval_model=retrieval_model)
#         global_graph = deduplicator.deduplicate(
#             global_graph,
#             iterations=dedup_iterations,
#             max_workers=max_sub_workers,
#             max_cluster_size=dedup_max_cluster_size,
#             sim_weights=dedup_sim_weights,
#         )
#         print(f"After deduplication: {global_graph.network.number_of_nodes()} nodes, "
#               f"{global_graph.network.number_of_edges()} edges")
#
#     # Step 6: Global Cross-Chunk Edge Construction
#     print(f"\n--- Constructing global cross-chunk edges (mode={edge_mode}, top_k={top_k}) ---")
#     construct_edges_during_merge(
#         global_graph,
#         threshold=edge_threshold,
#         max_workers=max_sub_workers,
#         mode=edge_mode,
#         top_k=top_k
#     )
#
#     print("\nIndexing complete! Returning unified Knowledge Graph.")
#     return global_graph
#
#
# def build_index_from_list(
#         chunks: List[str],
#         model_name: str = "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
#         max_workers: int = 4,
#         edge_threshold: float = 0.5,
#         top_k: int = 10,
#         max_sub_workers: int = 10,
#         fact_batch_size: int = 15,
#         edge_mode: str = "linear",
#         build_global_edges: bool = True,
#         existing_graph: Graph = None,
#         run_deduplication: bool = True,
#         dedup_model_name: str = "all-MiniLM-L6-v2",
#         dedup_sim_weights: dict = None,
#         dedup_iterations: int = 1,
#         dedup_max_cluster_size: int = 64,
# ) -> Graph:
#     """
#     Takes a user-provided list of text chunks (strings) and incorporates them into a
#     knowledge graph. Unlike build_index_from_file, no chunking is performed — the caller
#     controls the chunk boundaries.
#
#     If existing_graph is provided, new nodes and edges are merged into it and the updated
#     graph is returned. If no existing_graph is given, a fresh graph is created.
#
#     build_global_edges controls whether cross-chunk fact-to-fact edge construction runs
#     after the merge. Set to False when absorbing chunks incrementally (e.g. in a
#     checkpoint pipeline) and call construct_edges_during_merge manually at the end.
#
#     dedup_iterations — number of deduplicate-then-re-cluster passes.
#     """
#     if not chunks:
#         print("Error: The provided chunk list is empty.")
#         return existing_graph if existing_graph is not None else Graph()
#
#     print(f"Processing {len(chunks)} user-provided chunks. Starting parallel node extraction...")
#
#     # Step 1: Process chunks in parallel to build localized subgraphs
#     local_graphs = []
#     with ThreadPoolExecutor(max_workers=max_workers) as executor:
#         futures = {
#             executor.submit(process_single_chunk, chunk, edge_threshold, max_sub_workers, fact_batch_size): i
#             for i, chunk in enumerate(chunks)
#         }
#
#         for future in as_completed(futures):
#             try:
#                 resulting_graph = future.result()
#                 local_graphs.append(resulting_graph)
#                 print(f"Completed chunk processing. Extracted nodes for {len(local_graphs)}/{len(chunks)} chunks.")
#             except Exception as e:
#                 print(f"Error processing chunk subgraph: {e}")
#
#     # Step 2: Merge all local subgraphs into the target graph (existing or new)
#     print("\n--- Merging local subgraphs ---")
#     global_graph = existing_graph if existing_graph is not None else Graph()
#     for local_graph in local_graphs:
#         global_graph.merge(local_graph)
#
#     print(f"Merge complete. Graph size: {global_graph.network.number_of_nodes()} nodes, "
#           f"{global_graph.network.number_of_edges()} edges")
#
#     # Step 3: Entity Deduplication
#     if run_deduplication:
#         print(f"\n--- Running Entity Deduplication ({dedup_iterations} iteration(s)) ---")
#         from sentence_transformers import SentenceTransformer
#         retrieval_model = SentenceTransformer(dedup_model_name)
#         deduplicator = GraphDeduplicator(retrieval_model=retrieval_model)
#         global_graph = deduplicator.deduplicate(
#             global_graph,
#             iterations=dedup_iterations,
#             max_workers=max_sub_workers,
#             max_cluster_size=dedup_max_cluster_size,
#             sim_weights=dedup_sim_weights,
#         )
#         print(f"After deduplication: {global_graph.network.number_of_nodes()} nodes, "
#               f"{global_graph.network.number_of_edges()} edges")
#
#     # Step 4: Global Cross-Chunk Edge Construction (optional)
#     if build_global_edges:
#         print(f"\n--- Constructing global cross-chunk edges (mode={edge_mode}, top_k={top_k}) ---")
#         construct_edges_during_merge(
#             global_graph,
#             threshold=edge_threshold,
#             max_workers=max_sub_workers,
#             mode=edge_mode,
#             top_k=top_k
#         )
#
#     print("\nIndexing complete! Returning Knowledge Graph.")
#     return global_graph


def build_index(
        filepath: str = None,
        chunks: List[str] = None,
        model_name: str = "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
        max_workers: int = 4,
        chunk_size: int = 600,
        overlap: int = 50,
        chunk_list_inputs: bool = False,
        edge_threshold: float = 0.5,
        top_k: int = 10,
        max_sub_workers: int = 10,
        fact_batch_size: int = 15,
        edge_mode: str = "linear",
        build_global_edges: bool = True,
        existing_graph: Graph = None,
        run_deduplication: bool = True,
        dedup_model_name: str = "all-MiniLM-L6-v2",
        dedup_sim_weights: dict = None,
        dedup_iterations: int = 1,
        dedup_max_cluster_size: int = 64,
        dedup_run_exact: bool = True,
        dedup_run_semhash: bool = True,
        dedup_semhash_threshold: float = 0.95,
        dedup_run_llm: bool = True,
        dedup_similarity_threshold: float = 0.75,
        # ── Fact deduplication / contradiction resolution (nkg/deduplication/fact_deduplication.py) ──
        # Runs after entity dedup, before global fact-fact edge construction, so contradictory/
        # duplicate/differently-scoped facts are cleaned up before spending LLM calls building
        # cross-chunk edges on them. See GraphFactDeduplicator.deduplicate for the algorithm.
        run_fact_dedup: bool = True,
        fact_dedup_iterations: int = 1,
        fact_dedup_top_k: int = 12,
        fact_dedup_threshold: float = 0.8,
        fact_dedup_cluster_size: int = 8,
        fact_dedup_sim_weights: dict = None,
        use_cot: bool = True,
        neutral_description: bool = False,
        concise_summary: bool = False,
        combined_entity_extraction: bool = False,
        full_entity_profile: bool = True,
        generate_entity_relation_labels: bool = False,
        clustered_round1: bool = True,
        clustered_round2: bool = True,
        clustered_round3: bool = True,
        clustered_max_facts_per_prompt: int = 20,
        retrieval_model_name: str = "all-MiniLM-L6-v2",
        # ── Construction-phase similarity weights (per round of global edge clustering) ──
        # Each dict has keys: "sent", "topic", "entity", "questions". None = use default.
        construction_r1_weights: dict = None,  # default: entity-only {sent:0, topic:0, entity:1, questions:0}
        construction_r2_weights: dict = None,  # default: balanced   {sent:1, topic:1, entity:1, questions:1}
        construction_r3_weights: dict = None,  # default: balanced w/ lower entity {sent:1, topic:1, entity:0.5, questions:1}
        # ── Asymmetric "questions" matching mode used by the global edge clustering rounds above ──
        # "follow_up_to_answered"      — this fact's follow-up questions vs. other facts' answered questions (default)
        # "follow_up_to_fact_sentence" — this fact's follow-up questions vs. other facts' sentences
        # "fact_sentence_to_follow_up" — this fact's sentence vs. other facts' follow-up questions
        questions_mode: str = "follow_up_to_answered",
        # ── Round 3 (middle-k) "skip" fraction for global edge clustering ──
        # Fraction of top-ranked facts to skip before taking the next top_k, per fact, when
        # building round 3's similarity graph. 0.0 disables skipping entirely (round 3 then
        # draws from the same top-k band as round 2, minus whatever already has an edge).
        middle_k_skip_percent: float = 0.33,
        # ── Skip the intra-chunk fact-edge LLM call for chunks with too few facts ──
        same_chunk_min_facts: int = 2,
        # ── Fact->fact edge text. True (default) = each edge gets a description sentence plus short labels.
        #    False = label-only prompts (no description is generated; each edge's `description` attribute
        #    holds its short label), so retrieval sees short labels only. Applies to the bulk same-chunk and
        #    clustered cross-chunk edges, i.e. edge_mode="clustered". ──
        edge_descriptions: bool = True,
        # ── edge_mode="clustered_fast" only (nkg/index/construction/fast_fact_clustering.py) ──
        # embed_loader: "remote_code" (the real voyage embedding model) | "legacy" (SentenceTransformer(name) as
        # Graph.init_fact_embeddings loads it); leiden_levels: "final" (disjoint clusters <= the prompt cap) | "all"
        # (legacy: every hierarchy level); leiden_passes: Leiden runs with different seeds, final clusters pooled.
        fast_cluster_embed_loader: str = "remote_code",
        fast_cluster_leiden_levels: str = "final",
        fast_cluster_leiden_passes: int = 1,
        # clustered_fast: clusters below this many facts are merged into their most strongly connected cluster while
        # the result stays <= clustered_max_facts_per_prompt (fast_fact_clustering.merge_small_clusters); 0 = off
        fast_cluster_min_size: int = 0,
        # fact-sentence extraction prompt (extract_chunk_features.FACT_PROMPTS): "default" (ChunkFacts) or "detailed"
        # (ChunkFactsDetailed: the text's wording and every time / cause / attitude / attribution / condition cue)
        fact_prompt: str = "default",
        # fact->fact edge prompt (label-only), intra-chunk and global: "default" (Bulk*FactEdgesLabelOnly), "general"
        # (Bulk*FactEdgesGeneral: one relation vocabulary for narrative and expository text) or "event"
        # (Bulk*FactEdgesEvent: RED / Hong et al. relation families + WILDCARD, specific labels; the intra-chunk call
        # also gets the chunk's text). general / event need edge_descriptions=False
        fact_edge_prompt: str = "default",
        # global cross-chunk fact->fact edge prompt: None (default) = fact_edge_prompt; "event" = BulkGlobalFactEdgesEvent
        # (RED / Hong et al. relation families + WILDCARD, specific labels; needs edge_descriptions=False and
        # edge_mode clustered / clustered_fast). Intra-chunk edges keep fact_edge_prompt.
        global_fact_edge_prompt: str = None,
        # ── Resumable checkpointing. None (default) = no checkpointing, unchanged behavior. ──
        # When set, build_index() persists progress under checkpoint_dir after every chunk,
        # after the merge+dedup step, and after every cluster within each global-edge round,
        # so a crashed/killed run can be resumed by calling build_index() again with the same
        # checkpoint_dir (and otherwise-identical arguments) instead of starting over.
        checkpoint_dir: str = None,
) -> tuple:
    """
    Unified entry point for building a knowledge graph from either a text file or a
    pre-chunked list of strings. Exactly one of filepath or chunks must be provided.

    filepath — path to a .txt file; text is chunked via chunk_text_by_tokens using
               chunk_size and overlap before processing.

    chunks   — caller-supplied list of strings. chunk_size/overlap are ignored unless
               chunk_list_inputs=True, in which case any string that exceeds chunk_size
               tokens is split into sub-chunks (with the given overlap) before processing.

    build_global_edges — when False, skips construct_edges_during_merge. Useful for
                         incremental checkpoint pipelines where global edges should only
                         be built after all batches have been absorbed.

    existing_graph — if provided, new nodes/edges are merged into it and the updated
                     graph is returned. A fresh Graph() is created when omitted. Only consulted
                     on a fresh run — if checkpoint_dir already has a merged checkpoint, that
                     checkpoint (which already reflects the original existing_graph) is used instead.

    generate_entity_relation_labels — controls how entity→fact edge labels are produced:
        False (default) — no extra LLM output for this; every entity linked to a fact gets an
                           empty entity→fact edge label (rendered as a bare "(participates in
                           event)" at retrieval time rather than a fabricated description).
        True             — the LLM generates a strict 2-4 word UPPERCASE relation_label per
                           entity per fact (e.g., IS_POLICYHOLDER, INITIATED_ACTION).

    checkpoint_dir — see the parameter comment above. Resuming requires calling build_index()
                     with the same filepath/chunks, chunk_size/overlap, and edge-construction
                     parameters as the original run; callers are responsible for ensuring that
                     (nkg.index.construction.checkpointing.ConstructionCheckpoint will raise if the
                     resolved chunk list doesn't match what's on disk, but it can't detect every
                     possible parameter change on its own).

    run_fact_dedup / fact_dedup_* — controls nkg.deduplication.fact_deduplication.GraphFactDeduplicator,
                     which runs immediately after entity deduplication (same checkpointed "merged"
                     stage) and before global fact-fact edge construction. It clusters facts whose
                     sentence/entity/topic similarity clears fact_dedup_threshold (top fact_dedup_top_k
                     neighbors per fact) and asks the LLM — shown each cluster's original source chunk
                     text — to remove/rewrite/merge facts that are duplicate, contradictory, or
                     differently-scoped restatements of each other. See that module's docstring for the
                     conflict taxonomy (credited to MemGraphRAG) and merge/rewire mechanics.
        fact_dedup_iterations   — number of deduplicate-then-re-cluster passes.
        fact_dedup_top_k        — max similarity-graph neighbors considered per fact (default 12).
        fact_dedup_threshold    — min normalized similarity [0,1] required to draw an edge (default 0.8).
        fact_dedup_cluster_size — max facts grouped into one LLM call (default 8).
        fact_dedup_sim_weights  — dict with keys "sentence", "entity", "topic" (default
                                  {"sentence": 0.7, "entity": 0.15, "topic": 0.15}).
    """
    # --- Input validation ---
    if fact_edge_prompt not in ("default", "general", "event"):
        raise ValueError(f"fact_edge_prompt must be 'default', 'general' or 'event', got {fact_edge_prompt!r}")
    if fact_edge_prompt in ("general", "event") and edge_descriptions:
        raise ValueError(f"fact_edge_prompt={fact_edge_prompt!r} is label-only: use it with edge_descriptions=False")
    if global_fact_edge_prompt not in (None, "default", "general", "event"):
        raise ValueError(f"global_fact_edge_prompt must be None, 'default', 'general' or 'event', got {global_fact_edge_prompt!r}")
    if global_fact_edge_prompt in ("general", "event") and edge_descriptions:
        raise ValueError(f"global_fact_edge_prompt={global_fact_edge_prompt!r} is label-only: use it with edge_descriptions=False")
    if global_fact_edge_prompt == "event" and edge_mode not in ("clustered", "clustered_fast"):
        raise ValueError("global_fact_edge_prompt='event' is implemented for edge_mode='clustered'/'clustered_fast'")
    if not edge_descriptions and edge_mode not in ("clustered", "clustered_fast"):
        raise ValueError("edge_descriptions=False is implemented for edge_mode='clustered'/'clustered_fast' (bulk label-only "
                         f"prompts); edge_mode={edge_mode!r} would still generate descriptions.")
    if filepath is not None and chunks is not None:
        raise ValueError("Provide either filepath or chunks, not both.")
    if filepath is None and chunks is None:
        raise ValueError("One of filepath or chunks must be provided.")

    # --- Resolve the final list of chunks to process ---
    if filepath is not None:
        if not os.path.exists(filepath):
            print(f"Error: File not found at {filepath}")
            return existing_graph if existing_graph is not None else Graph()

        with open(filepath, 'r', encoding='utf-8') as f:
            full_text = f.read()

        if not full_text.strip():
            print("Error: The provided file is empty.")
            return existing_graph if existing_graph is not None else Graph()

        print(f"Reading file: {filepath}")
        print(f"Chunking text (chunk_size={chunk_size}, overlap={overlap})...")
        text_chunks = chunk_text_by_tokens(
            model=model_name,
            text=full_text,
            chunk_size=chunk_size,
            overlap=overlap,
        )
        print(f"Created {len(text_chunks)} chunks.")
    else:
        if not chunks:
            print("Error: The provided chunk list is empty.")
            return existing_graph if existing_graph is not None else Graph()

        if chunk_list_inputs:
            print(f"chunk_list_inputs=True — checking {len(chunks)} strings against chunk_size={chunk_size}...")
            text_chunks = []
            for s in chunks:
                sub = chunk_text_by_tokens(
                    model=model_name,
                    text=s,
                    chunk_size=chunk_size,
                    overlap=overlap,
                )
                text_chunks.extend(sub)
            print(f"Expanded to {len(text_chunks)} chunks after splitting oversized strings.")
        else:
            text_chunks = chunks
            print(f"Processing {len(text_chunks)} user-provided chunks (no re-chunking).")

    print("Starting parallel node extraction...")

    _zero = {"prompt": 0, "completion": 0, "total": 0, "calls": 0, "cache_hit": 0}
    # every stage's token count slices dspy.settings.lm.history; DSPy's default cap (10,000 calls) would drop the
    # oldest calls of a larger build and undercount every stage
    if dspy.settings.max_history_size < HISTORY_SIZE_FOR_TOKEN_COUNTS:
        dspy.configure(max_history_size=HISTORY_SIZE_FOR_TOKEN_COUNTS)
    ckpt = ConstructionCheckpoint(checkpoint_dir)
    ckpt.verify_or_init_chunk_manifest(text_chunks)

    if ckpt.has_final():
        print("\n--- Checkpoint: global edges already complete — loading final graph ---")
        final_graph, final_token_breakdown = ckpt.load_final()
        print("\nIndexing complete! Returning Knowledge Graph (from checkpoint).")
        return final_graph, final_token_breakdown

    if ckpt.has_merged():
        # --- Checkpoint: chunk extraction + dedup already done, skip straight to global edges ---
        print("\n--- Checkpoint: chunks already merged + deduplicated — loading merged graph ---")
        global_graph, merged_state = ckpt.load_merged()
        extraction_tokens = ckpt.extraction_tokens_total()
        dedup_tokens = merged_state.get("dedup_tokens", _zero)
        fact_dedup_tokens = merged_state.get("fact_dedup_tokens", _zero)
        extraction_sub: dict = {}
    else:
        # --- Step 1: Process chunks in parallel to build localized subgraphs ---
        done_indices = ckpt.chunk_done_indices()
        local_graphs = {i: ckpt.load_chunk_graph(i) for i in done_indices}
        pending_indices = [i for i in range(len(text_chunks)) if i not in done_indices]
        if done_indices:
            print(f"--- Checkpoint: {len(done_indices)}/{len(text_chunks)} chunks already extracted, "
                  f"resuming remaining {len(pending_indices)} ---")

        try: _h_extraction_start = len(dspy.settings.lm.history)
        except Exception: _h_extraction_start = 0
        _extraction_t0 = time.time()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_single_chunk, text_chunks[i], edge_threshold, max_sub_workers, fact_batch_size, use_cot, neutral_description, concise_summary, combined_entity_extraction, full_entity_profile, generate_entity_relation_labels, same_chunk_min_facts, edge_descriptions, fact_prompt, fact_edge_prompt): i
                for i in pending_indices
            }

            for future in as_completed(futures):
                idx = futures[future]
                try:
                    resulting_graph = future.result()
                    local_graphs[idx] = resulting_graph
                    ckpt.save_chunk_graph(idx, resulting_graph)
                    print(f"Completed chunk processing. Extracted nodes for {len(local_graphs)}/{len(text_chunks)} chunks.")
                except Exception as e:
                    print(f"Error processing chunk subgraph: {e}")

        attempt_extraction_tokens = _stage_tokens(_h_extraction_start)
        try: _h_extraction_end = len(dspy.settings.lm.history)
        except Exception: _h_extraction_end = _h_extraction_start
        ckpt.add_extraction_tokens_and_time(attempt_extraction_tokens, time.time() - _extraction_t0)
        extraction_tokens = ckpt.extraction_tokens_total() if ckpt.enabled else attempt_extraction_tokens

        # Post-hoc sub-stage breakdown for chunk_extraction: classify each DSPy history entry
        # added during THIS attempt by its [[ ## field_name ## ]] prompt markers. On a resumed
        # run this only covers the chunks processed in this process (see build_index docstring).
        try:
            from nkg.utils.token_tracker import breakdown_from_history
            extraction_sub = breakdown_from_history(_h_extraction_start, _h_extraction_end)
        except Exception:
            extraction_sub = {}

        # When checkpointing is enabled, a chunk that failed this attempt must NOT be silently
        # dropped — that would merge/finalize (and checkpoint!) a permanently incomplete graph,
        # with no way to ever retry the missing chunk. Fail loudly instead so a rerun resumes
        # and retries exactly the still-missing indices. (Checkpointing disabled preserves the
        # original best-effort behavior for other build_index() callers.)
        if ckpt.enabled:
            still_missing = [i for i in range(len(text_chunks)) if i not in local_graphs]
            if still_missing:
                raise RuntimeError(
                    f"{len(still_missing)}/{len(text_chunks)} chunk(s) failed extraction this "
                    f"attempt (indices {still_missing}) — see errors above. Not proceeding to "
                    f"merge with an incomplete chunk set. Re-run with the same checkpoint_dir to "
                    f"retry just the missing chunk(s)."
                )

        # --- Step 2: Merge all local subgraphs into the target graph (deterministic, by chunk index) ---
        print("\n--- Merging local subgraphs ---")
        global_graph = existing_graph if existing_graph is not None else Graph()
        for i in sorted(local_graphs):
            global_graph.merge(local_graphs[i])

        print(f"Merge complete. Graph size: {global_graph.network.number_of_nodes()} nodes, "
              f"{global_graph.network.number_of_edges()} edges")

        # --- Step 3: Entity Deduplication (atomic — a crash here restarts dedup, not extraction) ---
        try: _h_dedup_start = len(dspy.settings.lm.history)
        except Exception: _h_dedup_start = 0
        _dedup_t0 = time.time()

        if run_deduplication:
            print(f"\n--- Running Entity Deduplication ({dedup_iterations} iteration(s)) ---")
            from sentence_transformers import SentenceTransformer
            retrieval_model = SentenceTransformer(dedup_model_name, trust_remote_code=True)
            deduplicator = GraphDeduplicator(retrieval_model=retrieval_model, use_cot=use_cot)
            global_graph = deduplicator.deduplicate(
                global_graph,
                run_exact_dedup=dedup_run_exact,
                run_semhash_dedup=dedup_run_semhash,
                semhash_threshold=dedup_semhash_threshold,
                run_llm_dedup=dedup_run_llm,
                iterations=dedup_iterations,
                max_workers=max_sub_workers,
                max_cluster_size=dedup_max_cluster_size,
                sim_weights=dedup_sim_weights,
                similarity_threshold=dedup_similarity_threshold,
            )
            print(f"After deduplication: {global_graph.network.number_of_nodes()} nodes, "
                  f"{global_graph.network.number_of_edges()} edges")

        dedup_tokens = _stage_tokens(_h_dedup_start)
        dedup_time_sec = time.time() - _dedup_t0

        # --- Step 3.5: Fact Deduplication / Contradiction Resolution (same atomic checkpoint
        # stage as entity dedup — a crash here also restarts from the merge, not extraction) ---
        try: _h_fact_dedup_start = len(dspy.settings.lm.history)
        except Exception: _h_fact_dedup_start = 0
        _fact_dedup_t0 = time.time()

        if run_fact_dedup:
            print(f"\n--- Running Fact Deduplication ({fact_dedup_iterations} iteration(s)) ---")
            if run_deduplication:
                fact_retrieval_model = retrieval_model  # reuse the model already loaded for entity dedup
            else:
                from sentence_transformers import SentenceTransformer
                fact_retrieval_model = SentenceTransformer(dedup_model_name, trust_remote_code=True)
            fact_deduplicator = GraphFactDeduplicator(retrieval_model=fact_retrieval_model, use_cot=use_cot)
            global_graph = fact_deduplicator.deduplicate(
                global_graph,
                iterations=fact_dedup_iterations,
                top_k=fact_dedup_top_k,
                threshold=fact_dedup_threshold,
                cluster_size=fact_dedup_cluster_size,
                max_workers=max_sub_workers,
                sim_weights=fact_dedup_sim_weights,
            )
            print(f"After fact deduplication: {global_graph.network.number_of_nodes()} nodes, "
                  f"{global_graph.network.number_of_edges()} edges")

        fact_dedup_tokens = _stage_tokens(_h_fact_dedup_start)
        fact_dedup_time_sec = time.time() - _fact_dedup_t0

        ckpt.save_merged(global_graph, dedup_tokens, dedup_time_sec,
                         fact_dedup_tokens=fact_dedup_tokens, fact_dedup_time_sec=fact_dedup_time_sec)

    # --- Step 4: Global Cross-Chunk Edge Construction (optional) ---
    global_round_tokens: dict = {}
    if build_global_edges:
        print(f"\n--- Constructing global cross-chunk edges (mode={edge_mode}, top_k={top_k}) ---")
        result = construct_edges_during_merge(
            global_graph,
            threshold=edge_threshold,
            max_workers=max_sub_workers,
            mode=edge_mode,
            top_k=top_k,
            use_cot=use_cot,
            neutral_description=neutral_description,
            use_round1=clustered_round1,
            use_round2=clustered_round2,
            use_round3=clustered_round3,
            clustered_max_facts_per_prompt=clustered_max_facts_per_prompt,
            retrieval_model=retrieval_model_name,
            r1_weights=construction_r1_weights,
            r2_weights=construction_r2_weights,
            r3_weights=construction_r3_weights,
            questions_mode=questions_mode,
            middle_k_skip_percent=middle_k_skip_percent,
            checkpoint=ckpt,
            edge_descriptions=edge_descriptions,
            fact_edge_prompt=global_fact_edge_prompt or fact_edge_prompt,
            fast_cluster_options={"embed_loader": fast_cluster_embed_loader,
                                  "leiden_levels": fast_cluster_leiden_levels,
                                  "leiden_passes": fast_cluster_leiden_passes,
                                  "min_cluster_size": fast_cluster_min_size},
        )
        if isinstance(result, dict):
            global_round_tokens = result

    # --- Build per-stage token breakdown ---
    r1 = global_round_tokens.get("round1", _zero)
    r2 = global_round_tokens.get("round2", _zero)
    r3 = global_round_tokens.get("round3", _zero)
    total_prompt     = extraction_tokens["prompt"]     + dedup_tokens["prompt"]     + fact_dedup_tokens["prompt"]     + r1["prompt"]     + r2["prompt"]     + r3["prompt"]
    total_completion = extraction_tokens["completion"] + dedup_tokens["completion"] + fact_dedup_tokens["completion"] + r1["completion"] + r2["completion"] + r3["completion"]
    total_calls      = extraction_tokens["calls"]      + dedup_tokens["calls"]      + fact_dedup_tokens["calls"]      + r1["calls"]      + r2["calls"]      + r3["calls"]
    total_cache_hit  = sum(t.get("cache_hit", 0) for t in (extraction_tokens, dedup_tokens, fact_dedup_tokens, r1, r2, r3))

    _sub_stages = [
        "chunk_description",
        "chunk_fact_extraction",
        "chunk_meta_combined",
        "chunk_entity_extraction",
        "fact_metadata_batch",
        "same_chunk_fact_edges",
        "chunk_extraction_other",
    ]

    token_breakdown = {
        "chunk_extraction":    extraction_tokens,
        "chunk_extraction_sub": {k: extraction_sub.get(k, _zero) for k in _sub_stages
                                 if extraction_sub.get(k, _zero)["calls"] > 0},
        "entity_deduplication": dedup_tokens,
        "fact_deduplication": fact_dedup_tokens,
        "global_edges_round1": r1,
        "global_edges_round2": r2,
        "global_edges_round3": r3,
        "total": {
            "prompt":     total_prompt,
            "completion": total_completion,
            "total":      total_prompt + total_completion,
            "calls":      total_calls,
            "cache_hit":  total_cache_hit,
        },
        "hyperparams": {
            "edge_mode": edge_mode,
            "fast_cluster_options": ({"embed_loader": fast_cluster_embed_loader,
                                      "leiden_levels": fast_cluster_leiden_levels,
                                      "leiden_passes": fast_cluster_leiden_passes,
                                      **({"min_cluster_size": fast_cluster_min_size} if fast_cluster_min_size else {})}
                                     if edge_mode == "clustered_fast" else None),
            "edge_descriptions": edge_descriptions,
            "fact_prompt": fact_prompt,
            "fact_edge_prompt": fact_edge_prompt,
            "global_fact_edge_prompt": global_fact_edge_prompt or fact_edge_prompt,
            "combined_entity_extraction": combined_entity_extraction,
            "full_entity_profile": full_entity_profile,
            "generate_entity_relation_labels": generate_entity_relation_labels,
            "dedup_run_exact": dedup_run_exact,
            "dedup_run_semhash": dedup_run_semhash,
            "dedup_semhash_threshold": dedup_semhash_threshold,
            "dedup_run_llm": dedup_run_llm,
            "dedup_similarity_threshold": dedup_similarity_threshold,
            "run_fact_dedup": run_fact_dedup,
            "fact_dedup_iterations": fact_dedup_iterations,
            "fact_dedup_top_k": fact_dedup_top_k,
            "fact_dedup_threshold": fact_dedup_threshold,
            "fact_dedup_cluster_size": fact_dedup_cluster_size,
            "fact_dedup_sim_weights": fact_dedup_sim_weights or {"sentence": 0.7, "entity": 0.15, "topic": 0.15},
            "questions_mode": questions_mode,
            "middle_k_skip_percent": middle_k_skip_percent,
            "construction_r1_weights": construction_r1_weights or {"sent": 0.0, "topic": 0.0, "entity": 1.0, "questions": 0.0},
            "construction_r2_weights": construction_r2_weights or {"sent": 1.0, "topic": 1.0, "entity": 1.0, "questions": 1.0},
            "construction_r3_weights": construction_r3_weights or {"sent": 1.0, "topic": 1.0, "entity": 0.5, "questions": 1.0},
        },
    }

    ckpt.save_final(global_graph, token_breakdown)

    print("\nIndexing complete! Returning Knowledge Graph.")
    return global_graph, token_breakdown


if __name__ == "__main__":
    configure_dspy(max_tokens=45000)

    import litellm
    litellm.request_timeout = 120

    # Deduplication, global edge construction, and the full pipeline are now all done inside
    # build_index_from_file. Pass dedup_* params to control deduplication behavior.
    final_graph, _ = build_index(
        filepath="./evaluation_data/ten_contracts_dataset/contracts/contract_7_term_with_riders.txt",
        model_name="Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
        max_workers=100,
        chunk_size=600,
        overlap=50,
        edge_threshold=0.5,
        top_k=12,
        max_sub_workers=10,
        edge_mode="clustered",
        run_deduplication=True,
        dedup_model_name="voyageai/voyage-4-nano",
        dedup_iterations=1,
        dedup_max_cluster_size=64,
        dedup_sim_weights={"name": 0.5, "role": 0.3, "type": 0.2},
        fact_batch_size=10
    )

    print(f"\nFinal Graph Size: {final_graph.network.number_of_nodes()} nodes, "
          f"{final_graph.network.number_of_edges()} edges")
    print(f"Average Entity Edges: {final_graph.avg_entity_edges()}")

    final_graph.export_graph("./kg_outputs/c7riders_voyage.graphml")


# if __name__ == "__main__":
#     # Example usage:
#     # Build the index from a folder called "data", processing 3 files/graphs per batch, using 4 threads.
#     configure_dspy(max_tokens=45000)
#
#     import litellm
#
#     # Tell litellm to wait up to 2 minutes for a response
#     litellm.request_timeout = 120
#
#     final_graph = build_index_from_directory(
#         directory="./td",
#         batch_size=2,
#         max_workers=200,
#         chunk_size=600,
#         overlap=50,
#         edge_threshold=0.5,
#         max_sub_workers=50
#     )
#
#     final_graph.export_graph("./kg_outputs/td_gemma.graphml")
#     final_graph = Graph()
#     final_graph.load_graph("kg_outputs/td_gemma.graphml")
#
#     print(
#         f"Final Graph Size: {final_graph.network.number_of_nodes()} nodes, {final_graph.network.number_of_edges()} edges")
#     print(f"Average Entity Edges: {final_graph.avg_entity_edges()}")
#
#     from nkg.deduplication.entity_deduplication import GraphDeduplicator
#     from sentence_transformers import SentenceTransformer
#     retrieval_model = SentenceTransformer("all-MiniLM-L6-v2")
#     deduplicator = GraphDeduplicator(retrieval_model=retrieval_model)
#     final_graph = deduplicator.deduplicate(final_graph, max_workers=100)
#
#     print(
#         f"Final Graph Size: {final_graph.network.number_of_nodes()} nodes, {final_graph.network.number_of_edges()} edges")
#     print(f"Average Entity Edges: {final_graph.avg_entity_edges()}")
#
#     # Save the resulting graph if desired
#     final_graph.export_graph("./kg_outputs/td_gemma_dedup.graphml")