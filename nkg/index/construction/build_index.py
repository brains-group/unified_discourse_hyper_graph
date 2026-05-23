import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

from nkg.models.Graph import Graph
from nkg.index.construction.construct_nodes import initialize_graph_from_text
from nkg.index.construction.construct_edges import construct_initial_edges, construct_edges_during_merge
from nkg.utils.general import batch_list  # Adjust import path if needed
from nkg.utils.config import configure_dspy
from nkg.utils.chunking import chunk_text_by_tokens
from nkg.deduplication.entity_deduplication import GraphDeduplicator

import warnings
warnings.filterwarnings("ignore")


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


def process_single_chunk(chunk_text: str, edge_threshold: float, max_sub_workers: int, fact_batch_size: int = 5) -> Graph:
    """
    Worker function to process an individual chunk of text.
    Extracts the nodes (Chunks, Facts, Entities) and constructs the
    internal local edges (Entity->Fact, Fact->Fact within the same chunk).
    """
    # We pass a massive chunk_size to bypass re-chunking, since we pre-chunked the text.
    local_graph = initialize_graph_from_text(chunk_text, chunk_size=999999, overlap=0, fact_batch_size=fact_batch_size)
    construct_initial_edges(local_graph, threshold=edge_threshold, max_workers=max_sub_workers, mode="linear")
    return local_graph


def build_index_from_file(
        filepath: str,
        model_name: str = "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
        max_workers: int = 4,
        chunk_size: int = 600,
        overlap: int = 50,
        edge_threshold: float = 0.5,
        top_k: int = 10,
        max_sub_workers: int = 10,
        fact_batch_size: int = 15,
        edge_mode: str = "linear",
        run_deduplication: bool = True,
        dedup_model_name: str = "all-MiniLM-L6-v2",
        dedup_sim_weights: dict = None,
        dedup_iterations: int = 1,
        dedup_max_cluster_size: int = 64,
) -> Graph:
    """
    Reads a single text file, breaks it into chunks via tokenizer, processes chunks
    in parallel to build local subgraphs, merges them into a global graph, runs
    entity deduplication, then constructs cross-chunk fact-to-fact edges.

    edge_mode controls the global fact edge construction strategy:
      "linear"    — one LLM call per candidate fact pair (original method)
      "clustered" — groups facts into clusters via similarity + Leiden, then one
                    bulk LLM call per cluster (much fewer LLM calls total)

    Deduplication parameters:
      run_deduplication    — set False to skip deduplication entirely
      dedup_model_name     — sentence-transformers model used for entity similarity
      dedup_sim_weights    — dict with keys "name", "role", "type" controlling the
                             blend of embedding dimensions (default: {"name":0.6, "role":0.3, "type":0.1})
      dedup_iterations     — number of deduplicate-then-re-cluster passes
      dedup_max_cluster_size — max Leiden cluster size during deduplication
    """
    # Step 1: Read the text file
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        return Graph()

    with open(filepath, 'r', encoding='utf-8') as f:
        full_text = f.read()

    if not full_text.strip():
        print("Error: The provided file is empty.")
        return Graph()

    print(f"Reading file: {filepath}")

    # Step 2: Chunk the text by tokens
    print(f"Chunking text using tokenizer for model: {model_name}...")
    text_chunks = chunk_text_by_tokens(
        model=model_name,
        text=full_text,
        chunk_size=chunk_size,
        overlap=overlap
    )
    print(f"Created {len(text_chunks)} chunks. Starting parallel node extraction...")

    # Step 3: Process chunks in parallel to build localized graphs
    local_graphs = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_single_chunk, chunk, edge_threshold, max_sub_workers, fact_batch_size): i
            for i, chunk in enumerate(text_chunks)
        }

        for future in as_completed(futures):
            try:
                resulting_graph = future.result()
                local_graphs.append(resulting_graph)
                print(f"Completed chunk processing. Extracted nodes for {len(local_graphs)}/{len(text_chunks)} chunks.")
            except Exception as e:
                print(f"Error processing chunk subgraph: {e}")

    # Step 4: Global Merge
    # We combine all subgraphs into one master graph BEFORE deduplication and cross-chunk
    # comparisons so the vector search has access to the entire document's context.
    print("\n--- Merging all local subgraphs into a Unified Global Graph ---")
    global_graph = Graph()
    for local_graph in local_graphs:
        global_graph.merge(local_graph)

    print(f"Merge complete. Base Graph Size: {global_graph.network.number_of_nodes()} nodes, "
          f"{global_graph.network.number_of_edges()} edges")

    # Step 5: Entity Deduplication (before global edges so deduped entities feed into vector search)
    if run_deduplication:
        print(f"\n--- Running Entity Deduplication ({dedup_iterations} iteration(s)) ---")
        from sentence_transformers import SentenceTransformer
        retrieval_model = SentenceTransformer(dedup_model_name)
        deduplicator = GraphDeduplicator(retrieval_model=retrieval_model)
        global_graph = deduplicator.deduplicate(
            global_graph,
            iterations=dedup_iterations,
            max_workers=max_sub_workers,
            max_cluster_size=dedup_max_cluster_size,
            sim_weights=dedup_sim_weights,
        )
        print(f"After deduplication: {global_graph.network.number_of_nodes()} nodes, "
              f"{global_graph.network.number_of_edges()} edges")

    # Step 6: Global Cross-Chunk Edge Construction
    print(f"\n--- Constructing global cross-chunk edges (mode={edge_mode}, top_k={top_k}) ---")
    construct_edges_during_merge(
        global_graph,
        threshold=edge_threshold,
        max_workers=max_sub_workers,
        mode=edge_mode,
        top_k=top_k
    )

    print("\nIndexing complete! Returning unified Knowledge Graph.")
    return global_graph


if __name__ == "__main__":
    configure_dspy(max_tokens=45000)

    import litellm
    litellm.request_timeout = 120

    # Deduplication, global edge construction, and the full pipeline are now all done inside
    # build_index_from_file. Pass dedup_* params to control deduplication behavior.
    final_graph = build_index_from_file(
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
        dedup_model_name="google/embeddinggemma-300m",
        dedup_iterations=1,
        dedup_max_cluster_size=64,
        dedup_sim_weights={"name": 0.5, "role": 0.3, "type": 0.2},
        fact_batch_size=10
    )

    print(f"\nFinal Graph Size: {final_graph.network.number_of_nodes()} nodes, "
          f"{final_graph.network.number_of_edges()} edges")
    print(f"Average Entity Edges: {final_graph.avg_entity_edges()}")

    final_graph.export_graph("./kg_outputs/c7riders_7.graphml")


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