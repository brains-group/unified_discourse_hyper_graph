from nkg.models.Graph import Graph
from nkg.models.index_objects import *
import networkx as nx
from concurrent.futures import ThreadPoolExecutor, as_completed
import dspy
from nkg.utils.chunking import *
from nkg.index.extraction.extract_chunk_features import ChunkAssembler

def initialize_graph_from_text(text: str, chunk_size=600, overlap=50, fact_batch_size: int = 5) -> Graph:
    """
    Takes a text string, divides into chunks, and creates a graph from the chunks.

    This graph doesn't include higher-order edges such as edges between:
    chunks to chunks, facts to other facts, entities to facts.
    """
    chunks = chunk_text_by_tokens(text=text, chunk_size=chunk_size, overlap=overlap)
    graph = Graph()

    chunk_assembler = ChunkAssembler()
    for chunk in chunks:
        chunk_instance = chunk_assembler(source_text=chunk, fact_batch_size=fact_batch_size)
        graph.add_chunk(chunk_instance)

    print("Initializing graph...")
    return graph





