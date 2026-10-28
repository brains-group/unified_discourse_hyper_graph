from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import dspy

from nkg.models.index_objects import Chunk, Fact
from nkg.index.extraction.extract_entities import extract_entity_fingerprints
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Shared guidance for every fact<->fact label_forward/label_backward (or
# label_1_2/label_2_1) field below. Fact-fact edges kept getting mislabeled
# with entity-attribute-style verbs (FOUNDED, BORN, NAMED_AFTER, BIRTH_DATE)
# that describe one entity's role *within* a single fact rather than the
# relationship *between* the two fact sentences -- e.g. "Esquire was founded
# in 1933." -[FOUNDED]-> "Esquire is an American men's magazine." makes no
# sense as a label, because FOUNDED doesn't describe how these two sentences
# relate to each other at all. This block is inlined into every relevant
# signature's docstring so the instruction travels with the actual prompt.
# ---------------------------------------------------------------------------
FACT_FACT_LABEL_GUIDANCE = """
    CRITICAL RULE FOR THE SHORT LABEL(S): the label must name the relationship BETWEEN THE TWO
    FACT SENTENCES AS WHOLES -- never an attribute, role, or action belonging to a single entity
    mentioned inside one of them. Test every label by reading it as
    "fact_sentence_1 --[LABEL]--> fact_sentence_2": if the label doesn't actually explain HOW
    fact 2 relates to fact 1 (adds detail to it, is caused by it, contradicts it, restates it,
    comes before/after it, etc.), it is wrong -- even if it looks like a plausible 2-4 word
    UPPERCASE tag. The relationship between the two facts is the single most important thing to
    get right here; a wrong or nonsensical label is worse than a conservative but correct one.

    Do NOT use entity-attribute-style verbs or nouns as a fact-fact label -- these describe one
    entity's role within a single fact, not a relationship between two facts, and belong on
    entity-level edges elsewhere in the pipeline, never here: FOUNDED, BORN, NAMED_AFTER,
    EMPLOYED, LOCATED_AT, BIRTH_DATE, FOUNDING_YEAR, DIED, OWNED_BY, and similar.

    BAD (do not do this):
    - "Esquire was founded in 1933." --[FOUNDED]--> "Esquire is an American men's magazine."
      Wrong: FOUNDED restates a fact about Esquire; it says nothing about how fact 2 relates to
      fact 1 (fact 2 is actually just background identifying what Esquire is).
    - "Esquire was founded in 1933." --[FOUNDING_YEAR]-->
      "Esquire was founded under the guidance of founders Arnold Gingrich, David A. Smart and
      Henry L. Jackson."
      Wrong: FOUNDING_YEAR is a topic tag copied from fact 1's own content, not a relationship --
      it doesn't say that fact 2 adds detail (the founders' names) to the event fact 1 states.
    - "Australia is a location." --[SAME_CATEGORY]--> "Cyprus is a location."
      Wrong: sharing a shallow surface category is not a meaningful relationship. If two facts
      only share something this thin, give the pair a relevance_score near 0.0 (or skip it
      entirely per this signature's own instructions) instead of inventing a label for it.

    GOOD (do this):
    - "Esquire was founded in 1933." --[ELABORATES_ON]-->
      "Esquire was founded under the guidance of founders Arnold Gingrich, David A. Smart and
      Henry L. Jackson."
      Right: names that fact 2 adds concrete detail to the founding event stated in fact 1.
    - "The company's revenue grew 20% in 2023." --[CAUSES]-->
      "The company opened three new factories in 2024."
      Right: names the causal link between the two events.
    - "The treaty was signed in 1919." --[CONTRADICTS]-->
      "Historians dispute whether the treaty was ever formally signed."
      Right: names the disagreement between the two sentences.

    Reuse a small set of precise categories that genuinely fit the pair (CAUSES, PRECEDES,
    ELABORATES_ON, CONTRADICTS, CORROBORATES, EXEMPLIFIES, RESULTS_IN, CONTRASTS_WITH,
    PROVIDES_CONTEXT_FOR, RESTATES) rather than inventing a new one-off label for every pair.
    """


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
    """ + FACT_FACT_LABEL_GUIDANCE
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
    """ + FACT_FACT_LABEL_GUIDANCE
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
        description="Strict 2-4 word UPPERCASE label naming the RELATIONSHIP BETWEEN THE TWO FACT "
                    "SENTENCES for the source → target direction (e.g., CAUSES, PRECEDES, "
                    "ELABORATES_ON, CORROBORATES, CONTRADICTS, EXEMPLIFIES). Never an entity's "
                    "attribute/role/action (FOUNDED, BORN, NAMED_AFTER, EMPLOYED are WRONG here) and "
                    "never a topic tag copied from one fact's own content -- it must describe how "
                    "the two sentences relate to each other, not restate either one."
    )
    label_backward: str = Field(
        ...,
        description="Same rule as label_forward, for the target → source direction."
    )
    relevance_score: float = Field(
        ...,
        description="Float between 0.0 and 1.0 indicating how strongly these two facts are logically connected."
    )


class NeutralFactEdgePair(BaseModel):
    """Holds a single neutral bridging description and directional labels for a fact pair."""
    source_id: int = Field(
        ...,
        description="The integer ID of the source fact in this relationship."
    )
    target_id: int = Field(
        ...,
        description="The integer ID of the target fact in this relationship. Must differ from source_id."
    )
    description: str = Field(
        ...,
        description="Exactly 1 sentence that neutrally bridges both facts, expressing the logical or narrative connection between them without favoring either direction."
    )
    label_forward: str = Field(
        ...,
        description="Strict 2-4 word UPPERCASE label naming the RELATIONSHIP BETWEEN THE TWO FACT "
                    "SENTENCES for the source → target direction (e.g., CAUSES, PRECEDES, "
                    "ELABORATES_ON, CORROBORATES, CONTRADICTS, EXEMPLIFIES). Never an entity's "
                    "attribute/role/action (FOUNDED, BORN, NAMED_AFTER, EMPLOYED are WRONG here) and "
                    "never a topic tag copied from one fact's own content -- it must describe how "
                    "the two sentences relate to each other, not restate either one."
    )
    label_backward: str = Field(
        ...,
        description="Same rule as label_forward, for the target → source direction."
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
    """ + FACT_FACT_LABEL_GUIDANCE

    chunk_summary: str = dspy.InputField(desc="Summary of the chunk that all of the following facts belong to.")
    numbered_facts: str = dspy.InputField(
        desc="Numbered list of fact sentences, one per line in the format: <integer_id>: <fact sentence>"
    )

    fact_edges: list[IntraChunkFactEdgePair] = dspy.OutputField(
        desc="List of IntraChunkFactEdgePair objects for every relevant pair of facts found."
    )


class BulkIntraChunkFactEdgesNeutral(dspy.Signature):
    __doc__ = """
    You are given a chunk summary and a numbered list of fact sentences that all come from that same chunk.
    Identify every pair of facts that has a meaningful logical or narrative relationship (causal, sequential, supportive, clarifying, contradictory, etc.).
    For each relevant pair, emit a NeutralFactEdgePair using their integer IDs (source_id and target_id).
    IMPORTANT: source_id and target_id are the integer labels from the numbered list — they are NOT about specific content.
      - description: Write exactly 1 sentence that neutrally bridges both facts, capturing the key logical or narrative insight that connects them without favoring either direction.
      - label_forward & label_backward: strict 2-4 word UPPERCASE labels for each direction (e.g., CAUSES, PRECEDES, EXEMPLIFIES, MODIFIES, CORROBORATES, CONFLICTS_WITH).
      - relevance_score: float 0.0–1.0 indicating logical dependency strength.
    Only emit pairs with a non-zero relevance_score. Skip completely unrelated pairs.
    """ + FACT_FACT_LABEL_GUIDANCE

    chunk_summary: str = dspy.InputField(desc="Summary of the chunk that all of the following facts belong to.")
    numbered_facts: str = dspy.InputField(
        desc="Numbered list of fact sentences, one per line in the format: <integer_id>: <fact sentence>"
    )

    fact_edges: list[NeutralFactEdgePair] = dspy.OutputField(
        desc="List of NeutralFactEdgePair objects for every relevant pair of facts found."
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
    """ + FACT_FACT_LABEL_GUIDANCE

    numbered_facts: str = dspy.InputField(
        desc="Numbered list of facts, one per line in the format: <integer_id>: [Context: <chunk summary>] <fact sentence>"
    )

    fact_edges: list[IntraChunkFactEdgePair] = dspy.OutputField(
        desc="List of IntraChunkFactEdgePair objects for every relevant pair of facts found."
    )


class BulkGlobalFactEdgesNeutral(dspy.Signature):
    __doc__ = """
    You are given a list of facts drawn from across multiple chunks of a document, each assigned a sequential integer ID.
    Each fact is presented alongside its source chunk context so you can understand its setting.
    Identify every pair of facts that has a meaningful cross-context logical or narrative relationship
    (causal, sequential, supportive, clarifying, contradictory, etc.).
    For each relevant pair, emit a NeutralFactEdgePair using their integer IDs (source_id and target_id).
    IMPORTANT: source_id and target_id are the integer labels from the numbered list — they are NOT about specific content.
      - description: Write exactly 1 sentence that neutrally bridges both facts, capturing the key logical or narrative insight that connects them across their different contexts without favoring either direction.
      - label_forward & label_backward: strict 2-4 word UPPERCASE labels for each direction (e.g., CORROBORATES, DEPENDS_ON, CONFLICTS_WITH, PRECEDES, SUPPORTS).
      - relevance_score: float 0.0–1.0 indicating relationship strength.
    Only emit pairs with a non-zero relevance_score. Skip completely unrelated pairs.
    """ + FACT_FACT_LABEL_GUIDANCE

    numbered_facts: str = dspy.InputField(
        desc="Numbered list of facts, one per line in the format: <integer_id>: [Context: <chunk summary>] <fact sentence>"
    )

    fact_edges: list[NeutralFactEdgePair] = dspy.OutputField(
        desc="List of NeutralFactEdgePair objects for every relevant pair of facts found."
    )


# ---------------------------------------------------------------------------------------------
# Label-only fact->fact edges (build_index(edge_descriptions=False)): the same relationship
# extraction as the Neutral signatures above, but the LLM writes no description sentence -- only
# the two directional UPPERCASE labels and the score. The labels are then the relationship's only
# text anywhere (construction stores each label as that edge's `description` too), so retrieval
# runs on short labels only.
# ---------------------------------------------------------------------------------------------
class LabelOnlyFactEdgePair(BaseModel):
    """Directional short labels and a strength score for one fact pair -- no description sentence."""
    source_id: int = Field(
        ...,
        description="The integer ID of the source fact in this relationship."
    )
    target_id: int = Field(
        ...,
        description="The integer ID of the target fact in this relationship. Must differ from source_id."
    )
    label_forward: str = Field(
        ...,
        description="Strict 2-4 word UPPERCASE label naming the RELATIONSHIP BETWEEN THE TWO FACT "
                    "SENTENCES for the source → target direction (e.g., CAUSES, PRECEDES, "
                    "ELABORATES_ON, CORROBORATES, CONTRADICTS, EXEMPLIFIES). This label is the only "
                    "description the relationship will ever have, so it must be specific enough to "
                    "stand on its own. Never an entity's attribute/role/action (FOUNDED, BORN, "
                    "NAMED_AFTER, EMPLOYED are WRONG here) and never a topic tag copied from one fact's "
                    "own content -- it must describe how the two sentences relate to each other."
    )
    label_backward: str = Field(
        ...,
        description="Same rule as label_forward, for the target → source direction: the same "
                    "relationship read the other way (e.g., CAUSES ↔ CAUSED_BY, PRECEDES ↔ FOLLOWS, "
                    "ELABORATES_ON ↔ ELABORATED_BY; symmetric relationships such as CORROBORATES or "
                    "CONTRADICTS keep the same label)."
    )
    relevance_score: float = Field(
        ...,
        description="Float between 0.0 and 1.0 indicating how strongly these two facts are logically connected."
    )


class BulkIntraChunkFactEdgesLabelOnly(dspy.Signature):
    __doc__ = """
    You are given a chunk summary and a numbered list of fact sentences that all come from that same chunk.
    Identify every pair of facts that has a meaningful logical or narrative relationship (causal, sequential, supportive, clarifying, contradictory, etc.).
    For each relevant pair, emit a LabelOnlyFactEdgePair using their integer IDs (source_id and target_id).
    IMPORTANT: source_id and target_id are the integer labels from the numbered list — they are NOT about specific content.
      - label_forward: strict 2-4 word UPPERCASE label naming how the target fact relates to the source fact (source → target), e.g., CAUSES, PRECEDES, EXEMPLIFIES, MODIFIES, CORROBORATES, CONFLICTS_WITH.
      - label_backward: strict 2-4 word UPPERCASE label for the same relationship read in the target → source direction (e.g., CAUSED_BY, FOLLOWS, EXEMPLIFIED_BY, MODIFIED_BY; symmetric relationships such as CORROBORATES or CONFLICTS_WITH keep the same label).
      - relevance_score: float 0.0–1.0 indicating logical dependency strength.
    Do NOT write any description, explanation, or sentence: the two labels are the only text this relationship will
    ever have, so choose them carefully -- each must name the relationship precisely on its own.
    Only emit pairs with a non-zero relevance_score. Skip completely unrelated pairs.
    """ + FACT_FACT_LABEL_GUIDANCE

    chunk_summary: str = dspy.InputField(desc="Summary of the chunk that all of the following facts belong to.")
    numbered_facts: str = dspy.InputField(
        desc="Numbered list of fact sentences, one per line in the format: <integer_id>: <fact sentence>"
    )

    fact_edges: list[LabelOnlyFactEdgePair] = dspy.OutputField(
        desc="List of LabelOnlyFactEdgePair objects for every relevant pair of facts found."
    )


class BulkGlobalFactEdgesLabelOnly(dspy.Signature):
    __doc__ = """
    You are given a list of facts drawn from across multiple chunks of a document, each assigned a sequential integer ID.
    Each fact is presented alongside its source chunk context so you can understand its setting.
    Identify every pair of facts that has a meaningful cross-context logical or narrative relationship
    (causal, sequential, supportive, clarifying, contradictory, etc.).
    For each relevant pair, emit a LabelOnlyFactEdgePair using their integer IDs (source_id and target_id).
    IMPORTANT: source_id and target_id are the integer labels from the numbered list — they are NOT about specific content.
      - label_forward: strict 2-4 word UPPERCASE label naming how the target fact relates to the source fact across their contexts (source → target), e.g., CORROBORATES, DEPENDS_ON, CONFLICTS_WITH, PRECEDES, SUPPORTS.
      - label_backward: strict 2-4 word UPPERCASE label for the same relationship read in the target → source direction (e.g., SUPPORTED_BY, FOLLOWS, DEPENDED_ON_BY; symmetric relationships such as CORROBORATES or CONFLICTS_WITH keep the same label).
      - relevance_score: float 0.0–1.0 indicating relationship strength.
    Do NOT write any description, explanation, or sentence: the two labels are the only text this relationship will
    ever have, so choose them carefully -- each must name the relationship precisely on its own.
    Only emit pairs with a non-zero relevance_score. Skip completely unrelated pairs.
    """ + FACT_FACT_LABEL_GUIDANCE

    numbered_facts: str = dspy.InputField(
        desc="Numbered list of facts, one per line in the format: <integer_id>: [Context: <chunk summary>] <fact sentence>"
    )

    fact_edges: list[LabelOnlyFactEdgePair] = dspy.OutputField(
        desc="List of LabelOnlyFactEdgePair objects for every relevant pair of facts found."
    )


# ---------------------------------------------------------------------------------------------
# General fact->fact edges (build_index(fact_edge_prompt="general"), label-only): one relation vocabulary for any
# kind of text -- narrative (time, motive, dialogue, change of state) and expository / argumentative (evidence,
# example, definition, concession) -- and a link only when one fact explains, answers, enables or follows from the
# other, not merely because both are about the same topic or scene. The relevance score is the edge weight
# retrieval walks on, so it is defined by how much one fact is needed to understand the other.
# ---------------------------------------------------------------------------------------------
GENERAL_FACT_EDGE_GUIDANCE = """
    WHEN TO LINK TWO FACTS: only when one fact helps explain, answer, enable, constrain or follow from the other --
    i.e. a reader who knows fact 1 understands fact 2 better (why it happened, when, what led to it, what it led to,
    who it answers, what it supports or contradicts). Do NOT link two facts just because they mention the same
    person, place, topic or scene; shared entities are already connected elsewhere. Do NOT emit "X adds detail to
    X" links between two facts that merely describe the same thing from different angles unless one is genuinely
    needed to interpret the other. Fewer, meaningful links are better than many vague ones.

    RELATION VOCABULARY (label_forward, source -> target; pick the most specific that is true; label_backward is the
    same relation read target -> source, given in brackets; symmetric ones keep the same label):
      Time and sequence
        PRECEDES [FOLLOWS]                 source happens before target (state the order the text gives)
        DURING [DURING]                    the two happen at the same time
      Cause, motive and purpose
        CAUSES [CAUSED_BY]                 source brings about target
        ENABLES [ENABLED_BY]               source makes target possible
        PREVENTS [PREVENTED_BY]            source stops or blocks target
        MOTIVATES [MOTIVATED_BY]           source (a goal, fear, belief, feeling, order) is why someone does target
        IN_ORDER_TO [PURPOSE_OF]           target is the goal that source's action serves
        RESULTS_IN [RESULTS_FROM]          target is the outcome or consequence of source
      Dialogue, attribution and perspective
        RESPONDS_TO [ANSWERED_BY]          target is a reply, reaction or answer to source (speech or action)
        CLAIMS_ABOUT [CLAIMED_BY]          source is someone's statement, belief or report about target
        REVEALS [REVEALED_BY]              source discloses or explains something hidden or unclear in target
        CONTRADICTS [CONTRADICTS]          the two cannot both be true, or one is a lie, mistake or reversal of the other
      Change of state
        CHANGES_STATE [STATE_CHANGED_BY]   source turns the situation, condition or relationship of target into a new one
      Evidence and exposition
        SUPPORTS [SUPPORTED_BY]            source is evidence, a reason or an argument for target
        EXAMPLE_OF [EXEMPLIFIED_BY]        source is a specific case of the general statement in target
        DEFINES [DEFINED_BY]               source explains what a term or thing in target is
        CONDITION_FOR [DEPENDS_ON]         target holds only if source holds
        CONCEDES [CONCEDED_BY]             source acknowledges a limitation or exception to target
        ELABORATES_ON [ELABORATED_BY]      source adds new information that target needs to be fully understood
      Comparison
        CONTRASTS_WITH [CONTRASTS_WITH]    the two differ in a way that matters
        PARALLELS [PARALLELS]              the two are alike in a way that matters
    Use another 2-4 word UPPERCASE label only when none of these fits; it must still describe how the two facts
    relate, never an attribute of one entity (FOUNDED, BORN, NAMED_AFTER, LOCATED_AT are wrong here).

    RELEVANCE SCORE (0.0-1.0, the strength retrieval uses): 0.9-1.0 one fact is essential to understanding or
    answering a question about the other (the cause of an event, the reply to a question, the evidence for a claim,
    the motive for an action); 0.6-0.8 a clear, useful connection; 0.3-0.5 a weak but real connection. If you would
    score a pair below 0.3, do not emit it.
    """


class GeneralFactEdgePair(LabelOnlyFactEdgePair):
    """LabelOnlyFactEdgePair (stored the same way: each direction's label is its description) with the general
    relation vocabulary."""
    label_forward: str = Field(
        ...,
        description="2-4 word UPPERCASE relation for source → target from the RELATION VOCABULARY (e.g., CAUSES, "
                    "MOTIVATES, RESPONDS_TO, PRECEDES, SUPPORTS, EXAMPLE_OF, CONTRADICTS). It is the only text the "
                    "relationship will ever have, so it must name how the two facts relate precisely."
    )
    label_backward: str = Field(
        ...,
        description="The same relation read target → source, as given in brackets in the RELATION VOCABULARY "
                    "(e.g., CAUSED_BY, MOTIVATED_BY, ANSWERED_BY, FOLLOWS, SUPPORTED_BY, EXEMPLIFIED_BY); symmetric "
                    "relations (CONTRADICTS, CONTRASTS_WITH, PARALLELS, DURING) keep the same label."
    )
    relevance_score: float = Field(
        ...,
        description="0.0-1.0: how much one fact is needed to understand or answer a question about the other "
                    "(see RELEVANCE SCORE). Pairs below 0.3 are not emitted."
    )


class BulkIntraChunkFactEdgesGeneral(dspy.Signature):
    __doc__ = """
    You are given a chunk summary and a numbered list of fact sentences that all come from that same chunk of a text
    (any kind of text: a story, an article, a report, an argument).
    Find the pairs of facts where one helps explain, answer, enable or follow from the other, and for each emit a
    GeneralFactEdgePair using their integer IDs (source_id and target_id).
    IMPORTANT: source_id and target_id are the integer labels from the numbered list — they are NOT about specific content.
      - label_forward / label_backward: the relation in each direction, from the RELATION VOCABULARY below.
      - relevance_score: per the RELEVANCE SCORE rubric below.
    Do NOT write any description or explanation: the two labels are the only text the relationship will ever have.
    """ + GENERAL_FACT_EDGE_GUIDANCE

    chunk_summary: str = dspy.InputField(desc="Summary of the chunk that all of the following facts belong to.")
    numbered_facts: str = dspy.InputField(
        desc="Numbered list of fact sentences, one per line in the format: <integer_id>: <fact sentence>"
    )

    fact_edges: list[GeneralFactEdgePair] = dspy.OutputField(
        desc="List of GeneralFactEdgePair objects, one per meaningfully related pair of facts."
    )


class BulkGlobalFactEdgesGeneral(dspy.Signature):
    __doc__ = """
    You are given facts drawn from different chunks of a corpus (any kind of text: stories, articles, reports,
    arguments), each with a sequential integer ID and shown alongside its source chunk context. Facts from different
    documents or unrelated contexts usually should NOT be linked: link two facts only when, in context, one helps
    explain, answer, enable or follow from the other (the same event, person or argument continued elsewhere).
    For each such pair emit a GeneralFactEdgePair using their integer IDs (source_id and target_id).
    IMPORTANT: source_id and target_id are the integer labels from the numbered list — they are NOT about specific content.
      - label_forward / label_backward: the relation in each direction, from the RELATION VOCABULARY below.
      - relevance_score: per the RELEVANCE SCORE rubric below.
    Do NOT write any description or explanation: the two labels are the only text the relationship will ever have.
    """ + GENERAL_FACT_EDGE_GUIDANCE

    numbered_facts: str = dspy.InputField(
        desc="Numbered list of facts, one per line in the format: <integer_id>: [Context: <chunk summary>] <fact sentence>"
    )

    fact_edges: list[GeneralFactEdgePair] = dspy.OutputField(
        desc="List of GeneralFactEdgePair objects, one per meaningfully related pair of facts."
    )



# ---------------------------------------------------------------------------------------------
# Event-relation fact->fact edges (build_index(fact_edge_prompt="event") for intra-chunk + global edges, or
# global_fact_edge_prompt="event" for global cross-chunk edges only; label-only): the compact prompt of evaluation_scripts/unh/FACT_EDGE_EVENT_PROMPT_COMPACT.md. Relation families from Richer
# Event Description (O'Gorman et al. 2016) and Hong et al. (2016) -- contingency, inheritance, expansion, comparison,
# temporality, bridging -- plus WILDCARD; labels name what the pair reveals (CAUSES_TWO_DAY_BLACKOUT, not CAUSES).
# Few-shot examples use invented names absent from HotpotQA / MuSiQue / QuALITY.
# ---------------------------------------------------------------------------------------------
EVENT_FACT_EDGE_GUIDANCE = """
    CORE TEST: mentally merge the two facts into one sentence. Link them only if the merged sentence tells you
    something new: why something happened, what it led to, what made it possible or stopped it, when it happened
    relative to something else, what larger event it belongs to, who an unnamed person is, or that two reports cannot
    both be true. If the merge is just "A, and also B", do not link. Sharing a person, place, topic, category or
    coincidental date is NOT enough, because shared entities are already linked elsewhere. Facts from different
    documents or unrelated contexts that only share a name are not linked. Fewer, meaningful edges beat many vague
    ones.

    RELATION FAMILIES (choose the one most informative relation per pair; cause > precondition > time order):
      CONTINGENCY  cause (given A, B was inevitable); precondition/enable (without A, no B, including implicit
                   conditions with no "because" in the text); prevent (A is why B did not happen); motive (a feeling,
                   belief or goal in A is why someone does B); purpose (A is done in order to achieve B).
                   If B merely presupposes A by definition (a "remarriage" presupposes a marriage), it is not a cause.
      INHERITANCE  same event with complementary details (only with the same modality and polarity: "planned to
                   sell" is not "sold"); subevent (B is one part of the larger event A; a more specific event is
                   usually a subevent, not the same event); recurrence (B repeats A); generic habit vs. one occurrence.
      EXPANSION    instance (B is a case or member of the general statement A); conjunction (two parts of one larger
                   event with similar roles); disjunction (two parts of one larger event with opposing roles).
      COMPARISON   contradiction (cannot both be true); negation (B ends A: a job lost, a promise broken); competing
                   accounts of the same event; concession (an expected outcome failed); a meaningful contrast.
      TEMPORALITY  before, immediately before, during/contains, begins on, ends on, overlap, simultaneous. Use only
                   when the timing itself matters, and never in addition to a causal edge.
      BRIDGING     part/whole; set/member; identity claim (an unnamed person matched to a named one: say "suspect"
                   unless the text confirms it); plan, threat or hope -> what actually happened.
      WILDCARD     clearly and strongly related, but no family fits: a clue and what it reveals, a question and its
                   answer, a statement that exposes a lie, a rule that decides someone's outcome. Invent a label
                   naming exactly what is revealed. If you cannot name it, emit nothing.

    LABELS: UPPERCASE_WITH_UNDERSCORES, 2-6 words: the relation word plus WHAT it is about. Never use a bare relation
    name like CAUSES, MOTIVATES, CONTRADICTS, PRECEDES or ELABORATES_ON.
      CAUSES_TWO_DAY_BLACKOUT (not CAUSES), FEAR_MOTIVATES_FLIGHT (not MOTIVATES), CONTRADICTS_CAR_COLOR (not
      CONTRADICTS).
    Use content words, not full names. label_forward reads source -> target. label_backward is the same relation read
    target -> source and is just as specific (CAUSES_TWO_DAY_BLACKOUT <-> BLACKOUT_CAUSED_BY_STORM); symmetric
    relations may repeat the label. Never use an attribute of one entity (FOUNDED, BORN, LOCATED_AT, EMPLOYED_BY) as a
    label.

    SCORE (0.0-1.0) = how much new understanding the combination gives:
      0.9-1.0  answers a why/how/who/what-next question neither fact answers alone
      0.6-0.8  a clear, useful gain
      0.3-0.5  weak but real
      below 0.3  do not emit

    GOOD EXAMPLES
      CONTINGENCY  "The storm knocked down Brindlewick's power lines." + "Brindlewick had no electricity for two
                   days." -> CAUSES_TWO_DAY_BLACKOUT / BLACKOUT_CAUSED_BY_STORM (0.95)
                   "Mara Okafor learned her brother had secretly sold the family farm." + "Mara Okafor did not speak
                   to her brother for a year." -> ANGER_MOTIVATES_YEAR_OF_SILENCE / SILENCE_MOTIVATED_BY_FARM_SALE (0.95)
      INHERITANCE  "Brindlewick held a three-day harvest festival." + "A pie contest was held on the festival's second
                   day." -> INCLUDES_PIE_CONTEST / SUBEVENT_OF_HARVEST_FESTIVAL (0.7)
      EXPANSION    "Every shop on Dovecote Lane closed during the strike." + "The Dovecote Lane bakery shut on
                   Monday." -> INSTANCE_BAKERY_CLOSES_IN_STRIKE / PART_OF_STRIKE_CLOSURES (0.75)
      COMPARISON   "Doctors expected the treatment to cure Halvard's fever." + "Halvard's fever returned a week
                   later." -> EXPECTED_CURE_FAILED / SHOWS_TREATMENT_FAILED (0.9)
      TEMPORALITY  "The siege of the fortress lasted six months." + "The commander's son was born in the fortress
                   during the siege." -> CONTAINS_SONS_BIRTH / BORN_DURING_SIEGE (0.7)
      BRIDGING     "A masked man robbed the Quenby bank." + "Police arrested Karl Reiss as a suspect in the Quenby
                   robbery." -> SUSPECT_FOR_MASKED_ROBBER / MASKED_ROBBER_SUSPECTED_REISS (0.9)
      WILDCARD     "The anonymous letter was written in green ink." + "Only the governor's secretary used green ink."
                   -> INK_POINTS_TO_SECRETARY / SECRETARY_LIKELY_WROTE_LETTER (0.95)

    BAD EXAMPLES
      x Shared entity only: "Mara Okafor grew up in Brindlewick." + "Mara Okafor owns a grey cat." -> no edge.
      x Restatement: "The bakery opened in 1990." + "The bakery has been open since 1990." -> no edge (a duplicate).
      x Vague label: storm -> blackout labelled CAUSES. Use CAUSES_TWO_DAY_BLACKOUT.
      x Subevent called the same event: "A fire spread through the old town." + "A fire destroyed the Dovecote Lane
        bakery." -> not SAME_FIRE, but INCLUDES_BAKERY_DESTROYED / SUBEVENT_OF_OLD_TOWN_FIRE.
      x Different modality as the same event: "Karl Reiss threatened to burn the barn." + "The barn burned down." ->
        not SAME_EVENT, but THREAT_PRECEDES_BARN_FIRE (or THREAT_CARRIED_OUT only if the text says Reiss set it).
    """

EVENT_RELATION_FAMILIES = ("CONTINGENCY", "INHERITANCE", "EXPANSION", "COMPARISON", "TEMPORALITY", "BRIDGING",
                           "WILDCARD")


class EventFactEdgePair(LabelOnlyFactEdgePair):
    """LabelOnlyFactEdgePair (stored the same way: each direction's label is its description) plus the relation
    family, stored on both directed edges as `relation_family`."""
    relation_family: str = Field(
        ...,
        description="One of CONTINGENCY, INHERITANCE, EXPANSION, COMPARISON, TEMPORALITY, BRIDGING, WILDCARD."
    )
    label_forward: str = Field(
        ...,
        description="Specific 2-6 word UPPERCASE_WITH_UNDERSCORES label, source -> target: the relation word plus WHAT "
                    "it is about (e.g., CAUSES_TWO_DAY_BLACKOUT, ANGER_MOTIVATES_YEAR_OF_SILENCE), never a bare "
                    "relation name. It is the only text the relationship will ever have."
    )
    label_backward: str = Field(
        ...,
        description="The same relation read target -> source, just as specific (e.g., BLACKOUT_CAUSED_BY_STORM); "
                    "symmetric relations may repeat label_forward."
    )
    relevance_score: float = Field(
        ...,
        description="0.0-1.0: how much new understanding combining the two facts gives (see SCORE). Pairs below 0.3 "
                    "are not emitted."
    )


class BulkIntraChunkFactEdgesEvent(dspy.Signature):
    __doc__ = """
    You link fact sentences in a knowledge graph. An edge means: "read these two facts together, because combined
    they reveal something neither reveals alone."
    You are given one passage of a text (chunk_text), its summary, and a numbered list of fact sentences extracted
    from that passage. All the facts come from the same passage, so they belong to the same story, argument or
    report: this is the best chance to capture how its events, motives, statements and details connect. Read the
    passage to see what it states or clearly implies about how the facts relate (why something happened, what it led
    to, who answers whom, what came before or after, what contradicts what), including links the passage implies
    without a connective word. Every edge must connect two facts FROM THE NUMBERED LIST; the passage is context only.
    For each pair that passes the CORE TEST, emit one EventFactEdgePair using their integer IDs (source_id and
    target_id, never equal; they are labels from the numbered list, not content): relation_family, label_forward,
    label_backward and relevance_score, as defined below.
    Do NOT write any description or explanation: the two labels are the only text the relationship will ever have.
    """ + EVENT_FACT_EDGE_GUIDANCE

    chunk_text: str = dspy.InputField(desc="The passage all of the following facts were extracted from.")
    chunk_summary: str = dspy.InputField(desc="Summary of the passage.")
    numbered_facts: str = dspy.InputField(
        desc="Numbered list of fact sentences, one per line in the format: <integer_id>: <fact sentence>"
    )

    fact_edges: list[EventFactEdgePair] = dspy.OutputField(
        desc="List of EventFactEdgePair objects, one per pair of facts that passes the CORE TEST."
    )

class BulkGlobalFactEdgesEvent(dspy.Signature):
    __doc__ = """
    You link fact sentences in a knowledge graph. An edge means: "read these two facts together, because combined
    they reveal something neither reveals alone."
    You are given facts drawn from different chunks of a corpus, each with a sequential integer ID and shown
    alongside its source chunk context. For each pair that passes the CORE TEST, emit one EventFactEdgePair using
    their integer IDs (source_id and target_id, never equal; they are labels from the numbered list, not content):
    relation_family, label_forward, label_backward and relevance_score, as defined below.
    Do NOT write any description or explanation: the two labels are the only text the relationship will ever have.
    """ + EVENT_FACT_EDGE_GUIDANCE

    numbered_facts: str = dspy.InputField(
        desc="Numbered list of facts, one per line in the format: <integer_id>: [Context: <chunk summary>] <fact sentence>"
    )

    fact_edges: list[EventFactEdgePair] = dspy.OutputField(
        desc="List of EventFactEdgePair objects, one per pair of facts that passes the CORE TEST."
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
    __doc__ = """
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
