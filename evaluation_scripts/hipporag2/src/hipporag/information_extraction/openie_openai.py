import json
from dataclasses import dataclass
from typing import Dict, Any, List, TypedDict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from ..prompts import PromptTemplateManager
from ..utils.logging_utils import get_logger
from ..utils.llm_utils import fix_broken_generated_json, filter_invalid_triples
from ..utils.misc_utils import TripleRawOutput, NerRawOutput
from ..llm.openai_gpt import CacheOpenAI

logger = get_logger(__name__)


class ChunkInfo(TypedDict):
    num_tokens: int
    content: str
    chunk_order: List[Tuple]
    full_doc_ids: List[str]


@dataclass
class LLMInput:
    chunk_id: str
    input_message: List[Dict]


def _extract_json_list_field(response: str, field_name: str) -> List:
    decoder = json.JSONDecoder()
    for start_index, character in enumerate(response):
        if character != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(response[start_index:])
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict) or field_name not in payload:
            continue
        value = payload[field_name]
        if not isinstance(value, list):
            raise ValueError(f"OpenIE response field {field_name!r} must be a list.")
        return value
    raise ValueError(f"OpenIE response does not contain a valid JSON object with {field_name!r}.")


def _extract_ner_from_response(real_response):
    return _extract_json_list_field(real_response, "named_entities")


class OpenIE:
    def __init__(self, llm_model: CacheOpenAI, max_workers: int = 8, ner_max_tokens: int = 512, triple_max_tokens: int = 2048):
        # Init prompt template manager
        if max_workers < 1:
            raise ValueError("max_workers must be at least 1.")
        if ner_max_tokens < 1 or triple_max_tokens < 1:
            raise ValueError("OpenIE token limits must be at least 1.")
        self.prompt_template_manager = PromptTemplateManager(role_mapping={"system": "system", "user": "user", "assistant": "assistant"})
        self.llm_model = llm_model
        self.max_workers = max_workers
        self.ner_max_tokens = ner_max_tokens
        self.triple_max_tokens = triple_max_tokens

    def ner(self, chunk_key: str, passage: str, refresh: bool = False) -> NerRawOutput:  # [unh_v1] refresh
        # PREPROCESSING
        ner_input_message = self.prompt_template_manager.render(name='ner', passage=passage)
        raw_response = ""
        metadata = {}
        try:
            # LLM INFERENCE
            inference_kwargs = {"max_new_tokens": self.ner_max_tokens}
            if refresh:
                inference_kwargs["_unh_refresh"] = True  # [unh_v1] retry: ask again instead of reusing the cache
            response_format = getattr(getattr(self.llm_model, "global_config", None), "response_format", None)
            if response_format is not None:
                inference_kwargs["response_format"] = response_format
            raw_response, metadata, cache_hit = self.llm_model.infer(
                messages=ner_input_message,
                **inference_kwargs,
            )
            metadata['cache_hit'] = cache_hit
            if metadata.get('finish_reason') == 'length':
                real_response = fix_broken_generated_json(raw_response)
            else:
                real_response = raw_response
            extracted_entities = _extract_ner_from_response(real_response)
            # [unh_v1] models sometimes emit a bare number (e.g. 1963) in the entity list; keep it as text, as
            # filter_invalid_triples already does for triples. Non-scalar items (null, objects, lists) are dropped.
            extracted_entities = [e if isinstance(e, str) else str(e) for e in extracted_entities
                                  if isinstance(e, (str, int, float)) and not isinstance(e, bool)]
            unique_entities = list(dict.fromkeys(extracted_entities))

        except Exception as e:
            # For any other unexpected exceptions, log them and return with the error message
            logger.warning(e)
            metadata.update({'error': str(e)})
            return NerRawOutput(
                chunk_id=chunk_key,
                response=raw_response,  # Store the error message in metadata
                unique_entities=[],
                metadata=metadata  # Store the error message in metadata
            )

        return NerRawOutput(
            chunk_id=chunk_key,
            response=raw_response,
            unique_entities=unique_entities,
            metadata=metadata
        )

    def triple_extraction(self, chunk_key: str, passage: str, named_entities: List[str], refresh: bool = False) -> TripleRawOutput:  # [unh_v1] refresh
        def _extract_triples_from_response(real_response):
            return _extract_json_list_field(real_response, "triples")

        # PREPROCESSING
        messages = self.prompt_template_manager.render(
            name='triple_extraction',
            passage=passage,
            named_entity_json=json.dumps({"named_entities": named_entities})
        )

        raw_response = ""
        metadata = {}
        try:
            # LLM INFERENCE
            inference_kwargs = {"max_new_tokens": self.triple_max_tokens}
            if refresh:
                inference_kwargs["_unh_refresh"] = True  # [unh_v1] retry: ask again instead of reusing the cache
            response_format = getattr(getattr(self.llm_model, "global_config", None), "response_format", None)
            if response_format is not None:
                inference_kwargs["response_format"] = response_format
            raw_response, metadata, cache_hit = self.llm_model.infer(
                messages=messages,
                **inference_kwargs,
            )
            metadata['cache_hit'] = cache_hit
            if metadata.get('finish_reason') == 'length':
                real_response = fix_broken_generated_json(raw_response)
            else:
                real_response = raw_response
            extracted_triples = _extract_triples_from_response(real_response)
            triplets = filter_invalid_triples(triples=extracted_triples)

        except Exception as e:
            logger.warning(f"Exception for chunk {chunk_key}: {e}")
            metadata.update({'error': str(e)})
            return TripleRawOutput(
                chunk_id=chunk_key,
                response=raw_response,
                metadata=metadata,
                triples=[]
            )

        # Success
        return TripleRawOutput(
            chunk_id=chunk_key,
            response=raw_response,
            metadata=metadata,
            triples=triplets
        )

    def openie(self, chunk_key: str, passage: str) -> Dict[str, Any]:
        ner_output = self.ner(chunk_key=chunk_key, passage=passage)
        triple_output = self.triple_extraction(chunk_key=chunk_key, passage=passage, named_entities=ner_output.unique_entities)
        return {"ner": ner_output, "triplets": triple_output}

    def batch_openie(self, chunks: Dict[str, ChunkInfo]) -> Tuple[Dict[str, NerRawOutput], Dict[str, TripleRawOutput]]:
        """
        Conduct batch OpenIE synchronously using multi-threading which includes NER and triple extraction.

        Args:
            chunks (Dict[str, ChunkInfo]): chunks to be incorporated into graph. Each key is a hashed chunk 
            and the corresponding value is the chunk info to insert.

        Returns:
            Tuple[Dict[str, NerRawOutput], Dict[str, TripleRawOutput]]:
                - A dict with keys as the chunk ids and values as the NER result instances.
                - A dict with keys as the chunk ids and values as the triple extraction result instances.
        """

        # Extract passages from the provided chunks
        chunk_passages = {chunk_key: chunk["content"] for chunk_key, chunk in chunks.items()}

        ner_results_list = []
        logical_prompt_tokens = 0
        logical_completion_tokens = 0
        billable_prompt_tokens = 0
        billable_completion_tokens = 0
        num_cache_hit = 0

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Create NER futures for each chunk
            ner_futures = {
                executor.submit(self.ner, chunk_key, passage): chunk_key
                for chunk_key, passage in chunk_passages.items()
            }

            pbar = tqdm(as_completed(ner_futures), total=len(ner_futures), desc="NER")
            for future in pbar:
                result = future.result()
                ner_results_list.append(result)
                # Update metrics based on the metadata from the result
                metadata = result.metadata
                prompt_tokens = metadata.get('prompt_tokens', 0)
                completion_tokens = metadata.get('completion_tokens', 0)
                logical_prompt_tokens += prompt_tokens
                logical_completion_tokens += completion_tokens
                if metadata.get('cache_hit'):
                    num_cache_hit += 1
                else:
                    billable_prompt_tokens += prompt_tokens
                    billable_completion_tokens += completion_tokens

                pbar.set_postfix({
                    'billable_prompt_tokens': billable_prompt_tokens,
                    'billable_completion_tokens': billable_completion_tokens,
                    'logical_prompt_tokens': logical_prompt_tokens,
                    'logical_completion_tokens': logical_completion_tokens,
                    'num_cache_hit': num_cache_hit
                })

        # [unh_v1] upstream raises if any chunk failed. Retry each failed chunk once with a fresh call; chunks that
        # still fail keep no entities (they stay in the index as passages) and are listed in the log.
        for i, result in enumerate(ner_results_list):
            if result.metadata.get("error"):
                ner_results_list[i] = self.ner(result.chunk_id, chunk_passages[result.chunk_id], refresh=True)
        failed_ner_chunk_ids = [result.chunk_id for result in ner_results_list if result.metadata.get("error")]
        if failed_ner_chunk_ids:
            logger.warning(f"[unh_v1] NER still failed after one retry for {len(failed_ner_chunk_ids)} chunk(s), continuing without their entities: {failed_ner_chunk_ids}")

        triple_results_list = []
        logical_prompt_tokens, logical_completion_tokens = 0, 0
        billable_prompt_tokens, billable_completion_tokens, num_cache_hit = 0, 0, 0
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Create triple extraction futures for each chunk
            re_futures = {
                executor.submit(self.triple_extraction, ner_result.chunk_id,
                                chunk_passages[ner_result.chunk_id],
                                ner_result.unique_entities): ner_result.chunk_id
                for ner_result in ner_results_list
            }
            # Collect triple extraction results with progress bar
            pbar = tqdm(as_completed(re_futures), total=len(re_futures), desc="Extracting triples")
            for future in pbar:
                result = future.result()
                triple_results_list.append(result)
                metadata = result.metadata
                prompt_tokens = metadata.get('prompt_tokens', 0)
                completion_tokens = metadata.get('completion_tokens', 0)
                logical_prompt_tokens += prompt_tokens
                logical_completion_tokens += completion_tokens
                if metadata.get('cache_hit'):
                    num_cache_hit += 1
                else:
                    billable_prompt_tokens += prompt_tokens
                    billable_completion_tokens += completion_tokens
                pbar.set_postfix({
                    'billable_prompt_tokens': billable_prompt_tokens,
                    'billable_completion_tokens': billable_completion_tokens,
                    'logical_prompt_tokens': logical_prompt_tokens,
                    'logical_completion_tokens': logical_completion_tokens,
                    'num_cache_hit': num_cache_hit
                })

        # [unh_v1] same policy as NER: one fresh retry, then continue without that chunk's triples
        ner_by_id = {r.chunk_id: r for r in ner_results_list}
        for i, result in enumerate(triple_results_list):
            if result.metadata.get("error"):
                triple_results_list[i] = self.triple_extraction(result.chunk_id, chunk_passages[result.chunk_id],
                                                                ner_by_id[result.chunk_id].unique_entities, refresh=True)
        failed_triple_chunk_ids = [result.chunk_id for result in triple_results_list if result.metadata.get("error")]
        if failed_triple_chunk_ids:
            logger.warning(f"[unh_v1] Triple extraction still failed after one retry for {len(failed_triple_chunk_ids)} chunk(s), continuing without their triples: {failed_triple_chunk_ids}")

        ner_results_dict = {res.chunk_id: res for res in ner_results_list}
        triple_results_dict = {res.chunk_id: res for res in triple_results_list}

        return ner_results_dict, triple_results_dict
