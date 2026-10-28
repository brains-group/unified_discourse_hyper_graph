"""
Post-hoc token classification for DSPy pipeline stages.

After a parallel processing phase completes, scan the DSPy history entries added
during that phase and classify each one by the [[ ## field_name ## ]] markers
that ChatAdapter embeds in every system message.  Because classification runs
*after* all threads have finished, there are no race conditions.
"""
import dspy

# DSPy keeps only the last settings.max_history_size (default 10,000) LM calls, and every token count in
# construction slices that history by index, so a longer build would silently stop counting. build_index()
# raises the cap to this before it starts.
HISTORY_SIZE_FOR_TOKEN_COUNTS = 10_000_000


def usage_tokens(u) -> tuple:
    """(prompt, completion, cache_hit) tokens of one LM usage record (dict or object). cache_hit = the prompt
    tokens served from the provider's prompt cache (DeepSeek: prompt_cache_hit_tokens; OpenAI-style:
    prompt_tokens_details.cached_tokens); 0 when the provider does not report it."""
    if u is None:
        return 0, 0, 0
    get = (lambda k: u.get(k)) if isinstance(u, dict) else (lambda k: getattr(u, k, None))
    hit = get("prompt_cache_hit_tokens")
    if hit is None:
        details = get("prompt_tokens_details")
        if details is not None:
            hit = details.get("cached_tokens") if isinstance(details, dict) else getattr(details, "cached_tokens", None)
    return int(get("prompt_tokens") or 0), int(get("completion_tokens") or 0), int(hit or 0)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def _get_system_content(entry: dict) -> str:
    messages = entry.get("messages") or []
    for msg in messages:
        if msg.get("role") == "system":
            content = msg.get("content", "")
            return content if isinstance(content, str) else ""
    # Fallback: concatenate all content strings
    parts = []
    for msg in messages:
        c = msg.get("content", "")
        if isinstance(c, str):
            parts.append(c)
    return " ".join(parts)


def _classify_entry(entry: dict) -> str:
    """
    Map a single DSPy history entry to the pipeline sub-stage that produced it.

    Classification order matters — most-specific markers first.
    """
    s = _get_system_content(entry)

    # Entity deduplication
    if "[[ ## cluster_records ## ]]" in s:
        return "entity_deduplication"

    # Same-chunk fact edges  (BulkIntraChunkFactEdges / Neutral)
    if "[[ ## chunk_summary ## ]]" in s and "[[ ## numbered_facts ## ]]" in s:
        return "same_chunk_fact_edges"

    # Fact metadata batch  (BatchedFactMetaExtractor)
    if "[[ ## entity_names ## ]]" in s and "[[ ## numbered_facts ## ]]" in s:
        return "fact_metadata_batch"

    # Global bulk fact edges  (BulkGlobalFactEdges / Neutral) — numbered_facts only
    if "[[ ## numbered_facts ## ]]" in s:
        return "global_bulk_fact_edges"

    # Chunk entity extraction  (ExtractChunkEntities)
    if "[[ ## entities ## ]]" in s:
        return "chunk_entity_extraction"

    # Chunk combined: summary + fact_sentences in one call  (ChunkMeta / ChunkMetaConcise)
    if "[[ ## summary ## ]]" in s and "[[ ## fact_sentences ## ]]" in s:
        return "chunk_meta_combined"

    # Chunk fact extraction  (ChunkFacts)
    if "[[ ## fact_sentences ## ]]" in s:
        return "chunk_fact_extraction"

    # Chunk description  (ChunkDescription / ChunkDescriptionConcise)
    if "[[ ## summary ## ]]" in s:
        return "chunk_description"

    return "chunk_extraction_other"


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _zero() -> dict:
    return {"prompt": 0, "completion": 0, "total": 0, "calls": 0, "cache_hit": 0}


def _add_usage(bucket: dict, entry: dict) -> None:
    p, c, hit = usage_tokens(entry.get("usage") or {})
    bucket["prompt"] += p
    bucket["completion"] += c
    bucket["total"] += p + c
    bucket["calls"] += 1
    bucket["cache_hit"] += hit


def breakdown_from_history(start_idx: int, end_idx: int) -> dict:
    """
    Classify every DSPy history entry in [start_idx, end_idx) into a sub-stage
    and return a dict of {stage_name: {prompt, completion, total, calls, cache_hit}}.
    """
    result: dict = {}
    try:
        entries = dspy.settings.lm.history[start_idx:end_idx]
    except Exception:
        return result

    for entry in entries:
        stage = _classify_entry(entry)
        if stage not in result:
            result[stage] = _zero()
        _add_usage(result[stage], entry)

    return result
