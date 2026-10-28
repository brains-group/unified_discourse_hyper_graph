from nkg.models.Graph import Graph
from nkg.models.index_objects import *
import networkx as nx
from concurrent.futures import ThreadPoolExecutor, as_completed
import dspy
from nkg.utils.chunking import *
from nkg.index.extraction.extract_chunk_features import ChunkAssembler

def initialize_graph_from_text(text: str, chunk_size=600, overlap=50, fact_batch_size: int = 5,
                               use_cot: bool = True, concise_summary: bool = False,
                               combined_entity_extraction: bool = False,
                               full_entity_profile: bool = True,
                               generate_entity_relation_labels: bool = False,
                               fact_prompt: str = "default") -> Graph:
    """
    Takes a text string, divides into chunks, and creates a graph from the chunks.

    This graph doesn't include higher-order edges such as edges between:
    chunks to chunks, facts to other facts, entities to facts.
    """
    chunks = chunk_text_by_tokens(text=text, chunk_size=chunk_size, overlap=overlap)
    graph = Graph()

    chunk_assembler = ChunkAssembler(use_cot=use_cot, concise_summary=concise_summary,
                                     combined_entity_extraction=combined_entity_extraction,
                                     full_entity_profile=full_entity_profile,
                                     generate_entity_relation_labels=generate_entity_relation_labels,
                                     fact_prompt=fact_prompt)
    for chunk in chunks:
        chunk_instance = chunk_assembler(source_text=chunk, fact_batch_size=fact_batch_size)
        graph.add_chunk(chunk_instance)

    print("Initializing graph...")
    return graph





