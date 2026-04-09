import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

from nkg.models.Graph import Graph
from nkg.index.construction.construct_nodes import initialize_graph_from_text
from nkg.index.construction.construct_edges import construct_all_edges, construct_edges_during_merge
from nkg.utils.general import batch_list  # Adjust import path if needed
from nkg.utils.config import configure_dspy

import warnings
warnings.filterwarnings("ignore")


def process_text_batch(texts: List[str], chunk_size: int, overlap: int, edge_threshold: float, max_workers: int) -> Graph:
    """
    Worker function for the first layer of the pyramid.
    Takes a batch of texts, initializes graphs for each, merges them,
    and then constructs all edges within the merged local graph.
    """
    merged_graph = Graph()

    # 1. Initialize and merge graphs for all texts in this batch
    for text in texts:
        local_graph = initialize_graph_from_text(text, chunk_size=chunk_size, overlap=overlap)
        merged_graph.merge(local_graph)

    # 2. Construct edges across the newly merged graph
    construct_all_edges(merged_graph, threshold=edge_threshold, max_workers=max_workers)

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
        max_sub_workers=10
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
            executor.submit(process_text_batch, batch, chunk_size, overlap, edge_threshold, max_sub_workers): i
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


if __name__ == "__main__":
    # Example usage:
    # Build the index from a folder called "data", processing 3 files/graphs per batch, using 4 threads.
    configure_dspy(max_tokens=45000)

    import litellm

    # Tell litellm to wait up to 2 minutes for a response
    litellm.request_timeout = 120

    final_graph = build_index_from_directory(
        directory="./td",
        batch_size=2,
        max_workers=200,
        chunk_size=600,
        overlap=50,
        edge_threshold=0.4,
        max_sub_workers=60
    )

    final_graph.export_graph("./kg_outputs/td1.graphml")
    print(
        f"Final Graph Size: {final_graph.network.number_of_nodes()} nodes, {final_graph.network.number_of_edges()} edges")
    print(f"Average Entity Edgse: {final_graph.avg_entity_edges()}")

    from nkg.deduplication.entity_deduplication import GraphDeduplicator
    from sentence_transformers import SentenceTransformer
    retrieval_model = SentenceTransformer("all-MiniLM-L6-v2")
    deduplicator = GraphDeduplicator(retrieval_model=retrieval_model)
    final_graph = deduplicator.deduplicate(final_graph, max_workers=100)

    print(
        f"Final Graph Size: {final_graph.network.number_of_nodes()} nodes, {final_graph.network.number_of_edges()} edges")
    print(f"Average Entity Edgse: {final_graph.avg_entity_edges()}")

    # Save the resulting graph if desired
    final_graph.export_graph("./kg_outputs/td2.graphml")