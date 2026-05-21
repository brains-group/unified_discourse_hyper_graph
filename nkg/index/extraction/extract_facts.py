from typing import List
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import dspy

from nkg.models.index_objects import Fact, EntityFingerprint


# ---------------------------------------------------------------------------
# Original single-fact extraction (kept for reference / standalone use)
# ---------------------------------------------------------------------------

class FactMetaExtractor(dspy.Signature):
    __doc__ = """
    Analyze the specific fact_sentence within the broader context of its parent source_text.
    Extract precise structural metadata using the following STRICT CONSTRAINTS:

    1. name: Generate a concise, descriptive 3-5 word title summarizing the core claim, rule, or event of the fact.
    2. chunk_topics: Extract a list of 2-4 highly specific thematic keywords or short phrases that connect this fact to the broader text.
    3. answered_questions: Extract a list of 2-4 questions that this fact answer.
    4. follow_up_questions: Extract a list of 2-4 follow up quesions that a user may have after reading this fact.

    CRITICAL INSTRUCTION: You MUST output at least one valid string in the chunk_topics list. Do NOT return an empty list. Do NOT hallucinate concepts, external facts, or metadata not explicitly supported by the provided text.
    """

    source_text: str = dspy.InputField()
    fact_sentence: str = dspy.InputField()
    name: str = dspy.OutputField()
    chunk_topics: List[str] = dspy.OutputField()
    answered_questions: List[str] = dspy.OutputField()
    follow_up_questions: List[str] = dspy.OutputField()


class FactAssembler(dspy.Module):
    """Single-fact assembler kept for standalone / legacy use."""

    def __init__(self):
        self.fact_meta_extractor = dspy.ChainOfThought(FactMetaExtractor)

    def forward(self, source_text: str, fact_sentence: str, chunk_entities: list[EntityFingerprint]) -> Fact:
        fact_meta_info = self.fact_meta_extractor(source_text=source_text, fact_sentence=fact_sentence)

        fact_instance = Fact(
            name=fact_meta_info.name,
            sentence=fact_sentence,
            macro_topics=[],
            chunk_topics=fact_meta_info.chunk_topics,
            entities=chunk_entities,
            answered_questions=fact_meta_info.answered_questions,
            follow_up_questions=fact_meta_info.follow_up_questions
        )

        return fact_instance


# ---------------------------------------------------------------------------
# Batched fact extraction
# ---------------------------------------------------------------------------

class FactBatchItem(BaseModel):
    """Holds metadata for one fact within a batch LLM call."""

    fact_id: int = Field(
        ...,
        description="Sequential ID of the fact within the current batch (starts at 1)."
    )
    name: str = Field(
        ...,
        description="Concise 3-5 word descriptive title for this fact."
    )
    chunk_topics: List[str] = Field(
        ...,
        description="2-4 specific thematic keywords connecting this fact to the chunk. NEVER empty."
    )
    answered_questions: List[str] = Field(
        ...,
        description="2-4 questions that this fact directly answers."
    )
    follow_up_questions: List[str] = Field(
        ...,
        description="2-4 questions a reader might have after reading this fact."
    )
    entity_names: List[str] = Field(
        ...,
        description="Subset of the provided entity_names that appear in this specific fact. May be empty."
    )


class BatchedFactMetaExtractor(dspy.Signature):
    __doc__ = """
    You are analyzing a batch of fact sentences extracted from a document chunk.

    You are given:
    - source_text: The original chunk for context.
    - numbered_facts: Fact sentences with sequential IDs, e.g. "1. <sentence>\\n2. <sentence>\\n..."
    - entity_names: ALL entity names found in this chunk.

    For EACH numbered fact output one FactBatchItem with:
    - fact_id: The number assigned to that fact (1, 2, 3, ...).
    - name: A concise 3-5 word title.
    - chunk_topics: 2-4 thematic keywords. NEVER empty.
    - answered_questions: 2-4 questions this fact answers.
    - follow_up_questions: 2-4 follow-up questions a reader might have.
    - entity_names: Choose ONLY from the provided entity_names — pick the subset that appear in this specific fact.

    You MUST return exactly one result per numbered fact. Do NOT skip any fact IDs.
    """

    source_text: str = dspy.InputField()
    numbered_facts: str = dspy.InputField(
        desc="Fact sentences numbered 1 to N, e.g. '1. <sentence>\\n2. <sentence>'"
    )
    entity_names: list[str] = dspy.InputField(
        desc="All entity names from this chunk. Choose the relevant subset for each fact."
    )
    fact_results: list[FactBatchItem] = dspy.OutputField(
        desc="One FactBatchItem per numbered fact, keyed by fact_id matching the input number."
    )


# ---------------------------------------------------------------------------
# Entity linking helpers
# ---------------------------------------------------------------------------

def _find_closest_entity(entity_name: str, chunk_entities: list[EntityFingerprint]) -> EntityFingerprint:
    """Return the closest EntityFingerprint using character n-gram TF-IDF cosine similarity."""
    candidate_names = [ef.name for ef in chunk_entities]
    all_names = [entity_name] + candidate_names

    try:
        vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 3))
        tfidf_matrix = vectorizer.fit_transform(all_names)
        scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
        best_idx = int(scores.argmax())
        return chunk_entities[best_idx]
    except Exception:
        return chunk_entities[0]


def _link_entity_names_to_fingerprints(
    entity_names: list[str],
    chunk_entities: list[EntityFingerprint]
) -> list[EntityFingerprint]:
    """
    Map entity name strings from LLM output to EntityFingerprint objects.
    Tries exact string match first; falls back to cosine similarity.
    Returns a deduplicated list preserving order.
    """
    if not chunk_entities or not entity_names:
        return []

    name_to_fingerprint = {ef.name: ef for ef in chunk_entities}

    seen = set()
    linked = []
    for name in entity_names:
        ef = name_to_fingerprint.get(name) or _find_closest_entity(name, chunk_entities)
        if ef not in seen:
            seen.add(ef)
            linked.append(ef)

    return linked


# ---------------------------------------------------------------------------
# Batched fact assembler
# ---------------------------------------------------------------------------

class BatchedFactAssembler(dspy.Module):
    """
    Assembles a list of Fact objects from a list of fact sentences in batches.

    Instead of one LLM call per fact, a single call processes `batch_size` facts
    at once. Each fact is given a sequential ID (1..N, restarting per batch) so
    the LLM can reference facts without repeating their full text.

    Entity assignment uses the shared chunk_entities pool:
    the LLM chooses which entity names appear in each fact, and we link those
    names back to the actual EntityFingerprint objects via exact match or cosine
    similarity fallback.
    """

    def __init__(self):
        self.fact_extractor = dspy.ChainOfThought(BatchedFactMetaExtractor)

    def forward(
        self,
        source_text: str,
        fact_sentences: list[str],
        chunk_entities: list[EntityFingerprint],
        batch_size: int = 5
    ) -> list[Fact]:
        entity_names = [ef.name for ef in chunk_entities]
        all_facts = []

        for i in range(0, len(fact_sentences), batch_size):
            batch = fact_sentences[i:i + batch_size]
            numbered_facts = "\n".join(f"{j + 1}. {sentence}" for j, sentence in enumerate(batch))

            result = self.fact_extractor(
                source_text=source_text,
                numbered_facts=numbered_facts,
                entity_names=entity_names
            )

            # Key results by their sequential ID so we can stitch them back to sentences
            id_to_result = {item.fact_id: item for item in result.fact_results}

            for j, sentence in enumerate(batch):
                fact_id = j + 1
                item = id_to_result.get(fact_id)

                if item is None:
                    # LLM missed this ID — create a minimal fallback fact
                    fact = Fact(
                        name=sentence[:50],
                        sentence=sentence,
                        macro_topics=[],
                        chunk_topics=["unknown"],
                        entities=[],
                        answered_questions=[],
                        follow_up_questions=[]
                    )
                else:
                    linked_entities = _link_entity_names_to_fingerprints(item.entity_names, chunk_entities)
                    fact = Fact(
                        name=item.name,
                        sentence=sentence,
                        macro_topics=[],
                        chunk_topics=item.chunk_topics,
                        entities=linked_entities,
                        answered_questions=item.answered_questions,
                        follow_up_questions=item.follow_up_questions
                    )

                all_facts.append(fact)

        return all_facts


def main():
    pass

if __name__ == "__main__":
    main()
