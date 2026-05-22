from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import dspy

from nkg.models.index_objects import Chunk, Fact
from nkg.index.extraction.extract_entities import extract_entity_fingerprints
from pydantic import BaseModel, Field

class ChunkEdge(dspy.Signature):
    __doc__ = """
    Evaluate the narrative or logical relationship between chunk_summary_1 and chunk_summary_2.
    Provide bidirectional analysis using the following constraints:
    - description_1_2: Explain how chunk 2 builds upon, contrasts, or relates to chunk 1 from chunk 1's perspective (1-2 sentences).
    - description_2_1: Explain how chunk 1 builds upon, contrasts, or relates to chunk 2 from chunk 2's perspective (1-2 sentences).
    - label_1_2 & label_2_1: Provide a strict 2-4 word UPPERCASE relationship label for each direction (e.g., PROVIDES_EXAMPLE, CONTRADICTS, ELABORATES_ON, PRECEDES).
    - relevance_score: Provide a float between 0.0 and 1.0 indicating how strongly these chunks are logically connected (1.0 = highly connected, 0.0 = unrelated).
    """

    chunk_summary_1: str = dspy.InputField()
    chunk_summary_2: str = dspy.InputField()

    # Replaced single description with directed descriptions
    description_1_2: str = dspy.OutputField(
        desc="How chunk 2 builds upon, contrasts, or relates to chunk 1 from chunk 1's perspective.")
    description_2_1: str = dspy.OutputField(
        desc="How chunk 1 builds upon, contrasts, or relates to chunk 2 from chunk 2's perspective.")

    label_1_2: str = dspy.OutputField()
    label_2_1: str = dspy.OutputField()

    relevance_score: float = dspy.OutputField()

class ChunkFactEdge(dspy.Signature):
    __doc__ = """
    Determine the specific functional role that the provided fact_sentence plays within the broader context of its parent chunk_summary.
    Output MUST be a strict 2-4 word UPPERCASE string describing this structural relationship (e.g., SUPPORTS_CLAIM, PROVIDES_METRIC, DEFINES_TERM, LISTS_EXCLUSION, ETC.).
    """

    chunk_summary: str = dspy.InputField()
    fact_sentence: str = dspy.InputField()
    short_edge_label: str = dspy.OutputField()

class CandidateFacts(dspy.Signature):
    __doc__ = """
    You are a strict relevance filter. You are given two chunk summaries, a description of their relationship (rich_edge_description), and a fact_dict mapping UUIDs to Fact Sentences.
    Your task is to identify which specific facts from the fact_dict directly participate in or support the described relationship between the two chunks.
    CRITICAL INSTRUCTION: The output candidate_facts MUST be a list containing ONLY the exact dictionary keys (the UUID strings) of the relevant facts. Do NOT output the text of the sentences. Do not hallucinate keys.
    """

    chunk_summary_1: str = dspy.InputField()
    chunk_summary_2: str = dspy.InputField()
    rich_edge_description: str = dspy.InputField()
    fact_dict: dict = dspy.InputField()
    candidate_facts: list[str] = dspy.OutputField()

class FactEdge(dspy.Signature):
    __doc__ = """
    Analyze the cross-chunk logical relationship between fact_sentence_1 and fact_sentence_2, using their respective chunk summaries for background context.
    - description_1_2 & description_2_1: Provide concise (1-2 sentence) bidirectional descriptions of how the facts interact, complement, or contradict each other.
    - label_1_2 & label_2_1: Provide strict 2-4 word UPPERCASE relationship labels (e.g., CORROBORATES, DEPENDS_ON, CONFLICTS_WITH).
    - relevance_score: A float between 0.0 and 1.0 indicating relationship strength. Output 0.0 if they are completely independent.
    """
    # inputs
    fact_sentence_1: str = dspy.InputField()
    chunk_summary_1: str = dspy.InputField()
    fact_sentence_2: str = dspy.InputField()
    chunk_summary_2: str = dspy.InputField()

    # outputs
    description_1_2: str = dspy.OutputField()
    description_2_1: str = dspy.OutputField()
    label_1_2: str = dspy.OutputField()
    label_2_1: str = dspy.OutputField()
    relevance_score: float = dspy.OutputField()

class SameChunkFactEdge(dspy.Signature):
    __doc__ = """
    Analyze the intra-chunk relationship between fact_sentence_1 and fact_sentence_2, which both belong to the provided shared chunk_summary.
    - description_1_2 & description_2_1: Provide concise (1-2 sentence) bidirectional descriptions explaining the local flow of logic between these two facts.
    - label_1_2 & label_2_1: Provide strict 2-4 word UPPERCASE relationship labels (e.g., CAUSES, PRECEDES, EXEMPLIFIES, MODIFIES).
    - relevance_score: A float between 0.0 and 1.0 indicating their degree of logical dependency.
    If there is no relationship between fact sentence 1 and fact sentence 2, you can return empty strings for the descriptions and labels with a relevance score of 0.
    """
    # inputs
    chunk_summary: str = dspy.InputField()
    fact_sentence_1: str = dspy.InputField()
    fact_sentence_2: str = dspy.InputField()

    # outputs
    description_1_2: str = dspy.OutputField()
    description_2_1: str = dspy.OutputField()
    label_1_2: str = dspy.OutputField()
    label_2_1: str = dspy.OutputField()
    relevance_score: float = dspy.OutputField()

class IntraChunkFactEdgePair(BaseModel):
    """Represents a single directed edge pair between two facts identified by their temporary integer IDs."""
    source_id: int = Field(
        ...,
        description="The integer ID of the source fact in this relationship."
    )
    target_id: int = Field(
        ...,
        description="The integer ID of the target fact in this relationship. Must differ from source_id."
    )
    description_forward: str = Field(
        ...,
        description="1-2 sentence description of how the target fact relates to or flows from the source fact (source → target direction)."
    )
    description_backward: str = Field(
        ...,
        description="1-2 sentence description of how the source fact relates to or flows from the target fact (target → source direction)."
    )
    label_forward: str = Field(
        ...,
        description="Strict 2-4 word UPPERCASE relationship label for the source → target direction (e.g., CAUSES, PRECEDES, SUPPORTS, EXEMPLIFIES, MODIFIES)."
    )
    label_backward: str = Field(
        ...,
        description="Strict 2-4 word UPPERCASE relationship label for the target → source direction."
    )
    relevance_score: float = Field(
        ...,
        description="Float between 0.0 and 1.0 indicating how strongly these two facts are logically connected."
    )


class BulkIntraChunkFactEdges(dspy.Signature):
    __doc__ = """
    You are given a chunk summary and a numbered list of fact sentences that all come from that same chunk.
    Identify every pair of facts that has a meaningful logical or narrative relationship (causal, sequential, supportive, clarifying, contradictory, etc.).
    For each relevant pair, emit an IntraChunkFactEdgePair using their integer IDs (source_id and target_id).
    IMPORTANT: source_id and target_id are the integer labels from the numbered list — they are NOT about specific content.
      - description_forward: how the target fact relates to the source fact, from the source's perspective (1-2 sentences).
      - description_backward: how the source fact relates to the target fact, from the target's perspective (1-2 sentences).
      - label_forward & label_backward: strict 2-4 word UPPERCASE labels for each direction (e.g., CAUSES, PRECEDES, EXEMPLIFIES, MODIFIES, CORROBORATES, CONFLICTS_WITH).
      - relevance_score: float 0.0–1.0 indicating logical dependency strength.
    Only emit pairs with a non-zero relevance_score. Skip completely unrelated pairs.
    """

    chunk_summary: str = dspy.InputField(desc="Summary of the chunk that all of the following facts belong to.")
    numbered_facts: str = dspy.InputField(
        desc="Numbered list of fact sentences, one per line in the format: <integer_id>: <fact sentence>"
    )

    fact_edges: list[IntraChunkFactEdgePair] = dspy.OutputField(
        desc="List of IntraChunkFactEdgePair objects for every relevant pair of facts found."
    )


class BulkGlobalFactEdges(dspy.Signature):
    __doc__ = """
    You are given a list of facts drawn from across multiple chunks of a document, each assigned a sequential integer ID.
    Each fact is presented alongside its source chunk context so you can understand its setting.
    Identify every pair of facts that has a meaningful cross-context logical or narrative relationship
    (causal, sequential, supportive, clarifying, contradictory, etc.).
    For each relevant pair, emit an IntraChunkFactEdgePair using their integer IDs (source_id and target_id).
    IMPORTANT: source_id and target_id are the integer labels from the numbered list — they are NOT about specific content.
      - description_forward: how the target fact relates to or follows from the source fact (source → target, 1-2 sentences).
      - description_backward: how the source fact relates to or follows from the target fact (target → source, 1-2 sentences).
      - label_forward & label_backward: strict 2-4 word UPPERCASE labels for each direction (e.g., CORROBORATES, DEPENDS_ON, CONFLICTS_WITH, PRECEDES, SUPPORTS).
      - relevance_score: float 0.0–1.0 indicating relationship strength.
    Only emit pairs with a non-zero relevance_score. Skip completely unrelated pairs.
    """

    numbered_facts: str = dspy.InputField(
        desc="Numbered list of facts, one per line in the format: <integer_id>: [Context: <chunk summary>] <fact sentence>"
    )

    fact_edges: list[IntraChunkFactEdgePair] = dspy.OutputField(
        desc="List of IntraChunkFactEdgePair objects for every relevant pair of facts found."
    )


class EntityFactRelationship(BaseModel):
    entity_name: str = Field(
        ...,
        description="The entity name of the entity whose relationship."
    )
    relation_name: str = Field(
        ...,
        description="The relationship between the entity and the fact or event."
    )

class EntityFactEdge(dspy.Signature):
    ___doc__ = """
    Analyze how the specific entities in the EntityNames list participate in the provided fact_sentence.
    For EACH entity in the list, generate an EntityFactRelationship object.
    CRITICAL CONSTRAINTS:
    1. entity_name: This MUST exactly match the string provided in the EntityNames input array. Do not alter casing or spelling.
    2. relation_name: Provide a strict 2-4 word UPPERCASE label defining the entity's precise action, state, or role within this specific fact (e.g., INITIATED_ACTION, SUFFERED_LOSS, IS_SUBJECT_OF, LOCATED_AT).
    You must output a relationship for every entity provided in the list.
    """

    #inputs
    fact_name: str = dspy.InputField()
    fact_sentence: str = dspy.InputField()
    EntityNames: list[str] = dspy.InputField()

    #outputs
    relations: list[EntityFactRelationship] = dspy.OutputField()



def main():
    pass

if __name__ == "__main__":
    main()
