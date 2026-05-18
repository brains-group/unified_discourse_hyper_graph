from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import dspy

from nkg.models.index_objects import Fact
from nkg.index.extraction.extract_entities import extract_entity_fingerprints

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
    def __init__(self):
        self.fact_meta_extractor = dspy.ChainOfThought(FactMetaExtractor)

    def forward(self, source_text: str, fact_sentence: str) -> Fact:
        fact_meta_info = self.fact_meta_extractor(source_text=source_text, fact_sentence=fact_sentence)
        entity_fingerprints = extract_entity_fingerprints(text=source_text)

        fact_instance = Fact(
            name=fact_meta_info.name,
            sentence=fact_sentence,
            macro_topics=[],
            chunk_topics=fact_meta_info.chunk_topics,
            entities=entity_fingerprints,
            answered_questions=fact_meta_info.answered_questions,
            follow_up_questions=fact_meta_info.follow_up_questions
        )

        return fact_instance

def main():
    pass

if __name__ == "__main__":
    main()
