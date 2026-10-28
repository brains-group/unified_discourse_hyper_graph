import dspy
from nkg.models.index_objects import Chunk, EntityFingerprint
from nkg.index.extraction.extract_facts import BatchedFactAssembler
from nkg.index.extraction.extract_entities import extract_chunk_entities, ShortEntityItem


class ChunkMeta(dspy.Signature):
    __doc__ = """
    Extract structural metadata and core facts from the source_text :

    - summary: Provide a dense, 2-3 sentence summary of the core information.
    - fact_sentences: Extract a list of self-contained factual sentences directly from the text. Ensure each sentence represents a complete clear claim, rule, event, or other set of information.
    """

    source_text: str = dspy.InputField()
    summary: str = dspy.OutputField()
    fact_sentences: list[str] = dspy.OutputField()


class ChunkDescription(dspy.Signature):
    __doc__ = """
    Extract structural metadata and core facts from the source_text :

    - summary: Provide a dense, 3-4 sentence summary of the core information.
    """

    source_text: str = dspy.InputField()
    summary: str = dspy.OutputField()


class ChunkFacts(dspy.Signature):
    __doc__ = """
    You are an expert intelligence analyst. Your goal is to achieve 100% exhaustive recall of all factual information contained in the source_text.

    -Steps-
    1. Scan the text and identify every distinct Entity (person, organization, concept, document, location) and every distinct Event/Action.
    2. For EACH Entity and Event identified, extract the specific, atomic facts associated with them.

    Each fact sentence must be:
    * Atomic: It must contain exactly one complete thought, rule, or action. Break compound sentences into multiple separate facts.
    * Self-Contained: It must make perfect sense completely out of context. Resolve all pronouns (change "He signed it" to "John Doe signed Policy 123").
    * Grounded: It must be explicitly stated in the text. Do not infer or summarize external knowledge.

    CRITICAL: You must extract EVERYTHING.
    For example, if a chunk has 15 distinct facts, do not stop after 3 or 4 facts, instead you must output the 15 fact sentences.
    """

    source_text: str = dspy.InputField()
    fact_sentences: list[str] = dspy.OutputField()


class ChunkFactsDetailed(dspy.Signature):
    __doc__ = """
    You are an expert annotator building a knowledge base that must answer detailed questions about the source_text
    WITHOUT access to the original text. Anything you leave out or blur is lost forever, so preserve every detail and
    cue, not just the main claims.

    -Steps-
    1. Read the whole source_text. Identify every person, group, place, object, document and concept, and every
       event, action, statement, thought and state.
    2. For each of them, write fact sentences that together preserve ALL of the information in the text, in the
       order it appears.

    Each fact sentence must be:
    * Faithful, not paraphrased: keep the text's own wording, names, numbers and key phrases wherever possible.
      Never generalize, never soften, never summarize several details into one vague statement.
    * Self-contained: resolve every pronoun and reference to the explicit name of the person, place or thing, so the
      sentence makes sense entirely out of context.
    * Complete with its cues. Keep, inside the sentence, every qualifier that changes its meaning:
        - time: dates, times of day, durations, ages, how long ago, how long something lasted, what happened
          before / after / while;
        - cause and purpose: why something happened, what someone wanted, intended, hoped or feared;
        - attitude and inner state: emotions, opinions, beliefs, suspicions, motives, as the text states them;
        - attribution: who said, thought, claimed, ordered or wrote it, and to whom; keep direct quotes verbatim
          when they carry meaning;
        - condition, contrast and uncertainty: if / unless / although / but / only / never / almost / apparently /
          seems / probably, and every negation;
        - quantities and specifics: amounts, counts, distances, prices, sizes, colors, names of things.
    * Atomic but not stripped: one event, claim or state per sentence, but do NOT split off a cause, time, condition
      or attribution from the event it belongs to. Prefer one longer faithful sentence over two that lose the link.
    * Grounded: only what the text says or directly states as implied by a character or narrator. No outside
      knowledge, no speculation of your own.

    Also cover:
    * Dialogue: what each speaker says, asks, refuses, admits or promises, and how they react.
    * Descriptions: how people, places and objects are described, including their appearance, role and status.
    * Sequence: when the text orders events (first, then, later, finally, years earlier), state the order.

    CRITICAL: Extract EVERYTHING. A dense passage can need 30 or more fact sentences; do not stop early, and do not
    drop small details because they seem minor. Small details are often exactly what a question asks about.
    """

    source_text: str = dspy.InputField()
    fact_sentences: list[str] = dspy.OutputField()


FACT_PROMPTS = {"default": ChunkFacts, "detailed": ChunkFactsDetailed}


class ChunkMetaConcise(dspy.Signature):
    __doc__ = """
    Extract structural metadata and core facts from the source_text :

    - summary: Provide a single sentence that broadly describes the main subject or purpose of this chunk without detailing specific facts.
    - fact_sentences: Extract a list of self-contained factual sentences directly from the text. Ensure each sentence represents a complete clear claim, rule, event, or other set of information.
    """

    source_text: str = dspy.InputField()
    summary: str = dspy.OutputField()
    fact_sentences: list[str] = dspy.OutputField()


class ChunkDescriptionConcise(dspy.Signature):
    __doc__ = """
    Extract structural metadata and core facts from the source_text :

    - summary: Provide a single sentence that broadly describes the main subject or purpose of this chunk without detailing specific facts.
    """

    source_text: str = dspy.InputField()
    summary: str = dspy.OutputField()


class ChunkDescriptionWithEntities(dspy.Signature):
    __doc__ = """
    Extract from source_text:
    - summary: 3-4 dense sentences covering the core information.
    - entities: Every named entity. For each:
        - name: Canonical name. Resolve pronouns (e.g., "you" → "Policyholder").
        - type: One of: PERSON, ORGANIZATION, LOCATION, DATE, POLICY, COVERAGE, DOCUMENT, CONCEPT.
        - role: 3-8 word micro-role describing its function in this specific text.
        - relational_anchors: Specific identifiers (policy numbers, codes, IDs). Empty list if none.
    Only include what is explicitly stated in the text.
    """
    source_text: str = dspy.InputField()
    summary: str = dspy.OutputField()
    entities: list[EntityFingerprint] = dspy.OutputField()


class ChunkDescriptionConciseWithEntities(dspy.Signature):
    __doc__ = """
    Extract from source_text:
    - summary: 1 sentence broadly describing the main subject or purpose of the chunk.
    - entities: Every named entity. For each:
        - name: Canonical name. Resolve pronouns (e.g., "you" → "Policyholder").
        - type: One of: PERSON, ORGANIZATION, LOCATION, DATE, POLICY, COVERAGE, DOCUMENT, CONCEPT.
        - role: 3-8 word micro-role describing its function in this specific text.
        - relational_anchors: Specific identifiers (policy numbers, codes, IDs). Empty list if none.
    Only include what is explicitly stated in the text.
    """
    source_text: str = dspy.InputField()
    summary: str = dspy.OutputField()
    entities: list[EntityFingerprint] = dspy.OutputField()


class ChunkDescriptionWithEntitiesShort(dspy.Signature):
    __doc__ = """
    Extract from source_text:
    - summary: 3-4 dense sentences covering the core information.
    - entities: Every named entity. For each:
        - name: Canonical name. Resolve pronouns (e.g., "you" → "Policyholder").
        - role: 3-8 word micro-role describing its function in this specific text.
    Only include what is explicitly stated in the text. Do not include type or relational anchors.
    """
    source_text: str = dspy.InputField()
    summary: str = dspy.OutputField()
    entities: list[ShortEntityItem] = dspy.OutputField()


class ChunkDescriptionConciseWithEntitiesShort(dspy.Signature):
    __doc__ = """
    Extract from source_text:
    - summary: 1 sentence broadly describing the main subject or purpose of the chunk.
    - entities: Every named entity. For each:
        - name: Canonical name. Resolve pronouns (e.g., "you" → "Policyholder").
        - role: 3-8 word micro-role describing its function in this specific text.
    Only include what is explicitly stated in the text. Do not include type or relational anchors.
    """
    source_text: str = dspy.InputField()
    summary: str = dspy.OutputField()
    entities: list[ShortEntityItem] = dspy.OutputField()


class ChunkMetaWithEntities(dspy.Signature):
    __doc__ = """
    Extract from source_text:
    - summary: 2-3 dense sentences covering the core information.
    - fact_sentences: Every distinct factual claim, rule, or event as a self-contained sentence. Resolve all pronouns.
    - entities: Every named entity. For each:
        - name: Canonical name. Resolve pronouns (e.g., "you" → "Policyholder").
        - type: One of: PERSON, ORGANIZATION, LOCATION, DATE, POLICY, COVERAGE, DOCUMENT, CONCEPT.
        - role: 3-8 word micro-role describing its function in this specific text.
        - relational_anchors: Specific identifiers (policy numbers, codes, IDs). Empty list if none.
    Only include what is explicitly stated in the text.
    """
    source_text: str = dspy.InputField()
    summary: str = dspy.OutputField()
    fact_sentences: list[str] = dspy.OutputField()
    entities: list[EntityFingerprint] = dspy.OutputField()


class ChunkMetaConciseWithEntities(dspy.Signature):
    __doc__ = """
    Extract from source_text:
    - summary: 1 sentence broadly describing the main subject or purpose of the chunk.
    - fact_sentences: Every distinct factual claim, rule, or event as a self-contained sentence. Resolve all pronouns.
    - entities: Every named entity. For each:
        - name: Canonical name. Resolve pronouns (e.g., "you" → "Policyholder").
        - type: One of: PERSON, ORGANIZATION, LOCATION, DATE, POLICY, COVERAGE, DOCUMENT, CONCEPT.
        - role: 3-8 word micro-role describing its function in this specific text.
        - relational_anchors: Specific identifiers (policy numbers, codes, IDs). Empty list if none.
    Only include what is explicitly stated in the text.
    """
    source_text: str = dspy.InputField()
    summary: str = dspy.OutputField()
    fact_sentences: list[str] = dspy.OutputField()
    entities: list[EntityFingerprint] = dspy.OutputField()


class ChunkMetaWithEntitiesShort(dspy.Signature):
    __doc__ = """
    Extract from source_text:
    - summary: 2-3 dense sentences covering the core information.
    - fact_sentences: Every distinct factual claim, rule, or event as a self-contained sentence. Resolve all pronouns.
    - entities: Every named entity. For each:
        - name: Canonical name. Resolve pronouns (e.g., "you" → "Policyholder").
        - role: 3-8 word micro-role describing its function in this specific text.
    Only include what is explicitly stated in the text. Do not include type or relational anchors.
    """
    source_text: str = dspy.InputField()
    summary: str = dspy.OutputField()
    fact_sentences: list[str] = dspy.OutputField()
    entities: list[ShortEntityItem] = dspy.OutputField()


class ChunkMetaConciseWithEntitiesShort(dspy.Signature):
    __doc__ = """
    Extract from source_text:
    - summary: 1 sentence broadly describing the main subject or purpose of the chunk.
    - fact_sentences: Every distinct factual claim, rule, or event as a self-contained sentence. Resolve all pronouns.
    - entities: Every named entity. For each:
        - name: Canonical name. Resolve pronouns (e.g., "you" → "Policyholder").
        - role: 3-8 word micro-role describing its function in this specific text.
    Only include what is explicitly stated in the text. Do not include type or relational anchors.
    """
    source_text: str = dspy.InputField()
    summary: str = dspy.OutputField()
    fact_sentences: list[str] = dspy.OutputField()
    entities: list[ShortEntityItem] = dspy.OutputField()


class ChunkAssembler(dspy.Module):
    def __init__(self, use_cot: bool = True, concise_summary: bool = False,
                 combined_entity_extraction: bool = False, full_entity_profile: bool = True,
                 generate_entity_relation_labels: bool = False, fact_prompt: str = "default"):
        """fact_prompt: which signature extracts the fact sentences (double extraction): "default" (ChunkFacts) or
        "detailed" (ChunkFactsDetailed: keeps the text's wording and every time / cause / attitude / attribution /
        condition cue inside the fact, for detail-heavy text such as narratives)."""
        if fact_prompt not in FACT_PROMPTS:
            raise ValueError(f"fact_prompt must be one of {list(FACT_PROMPTS)}, got {fact_prompt!r}")
        self.use_cot = use_cot
        self.combined_entity_extraction = combined_entity_extraction
        self.full_entity_profile = full_entity_profile
        self.generate_entity_relation_labels = generate_entity_relation_labels

        predict = dspy.ChainOfThought if use_cot else dspy.Predict

        if combined_entity_extraction and full_entity_profile:
            desc_sig = ChunkDescriptionConciseWithEntities if concise_summary else ChunkDescriptionWithEntities
            meta_sig = ChunkMetaConciseWithEntities if concise_summary else ChunkMetaWithEntities
        elif combined_entity_extraction and not full_entity_profile:
            desc_sig = ChunkDescriptionConciseWithEntitiesShort if concise_summary else ChunkDescriptionWithEntitiesShort
            meta_sig = ChunkMetaConciseWithEntitiesShort if concise_summary else ChunkMetaWithEntitiesShort
        else:
            desc_sig = ChunkDescriptionConcise if concise_summary else ChunkDescription
            meta_sig = ChunkMetaConcise if concise_summary else ChunkMeta

        self.chunk_description_extractor = predict(desc_sig)
        self.chunk_meta_extractor = predict(meta_sig)
        self.chunk_fact_extractor = predict(FACT_PROMPTS[fact_prompt])

    def forward(self, source_text: str, double_extraction=True, fact_batch_size: int = 5) -> Chunk:
        if double_extraction:
            chunk_description = self.chunk_description_extractor(source_text=source_text)
            chunk_facts = self.chunk_fact_extractor(source_text=source_text)
            summary = chunk_description.summary
            fact_sentences = chunk_facts.fact_sentences
            raw_entities = chunk_description.entities if self.combined_entity_extraction else None
        else:
            chunk_meta_info = self.chunk_meta_extractor(source_text=source_text)
            summary = chunk_meta_info.summary
            fact_sentences = chunk_meta_info.fact_sentences
            raw_entities = chunk_meta_info.entities if self.combined_entity_extraction else None

        if self.combined_entity_extraction and raw_entities is not None:
            if self.full_entity_profile:
                chunk_entities = raw_entities  # already list[EntityFingerprint]
            else:
                chunk_entities = [
                    EntityFingerprint(name=item.name, type="", role=item.role,
                                      relational_anchors=(summary,))
                    for item in raw_entities
                ]
        else:
            chunk_entities = extract_chunk_entities(
                text=source_text, use_cot=self.use_cot,
                full_profile=self.full_entity_profile, chunk_summary=summary
            )

        fact_assembler = BatchedFactAssembler(
            use_cot=self.use_cot,
            generate_entity_relation_labels=self.generate_entity_relation_labels
        )
        facts = fact_assembler(
            source_text=source_text,
            fact_sentences=fact_sentences,
            chunk_entities=chunk_entities,
            batch_size=fact_batch_size
        )

        return Chunk(
            text=source_text,
            summary=summary,
            entities=chunk_entities,
            facts=facts
        )


def main():
    pass

if __name__ == "__main__":
    main()
