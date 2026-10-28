import time
from nkg.models.Graph import Graph
from nkg.models.index_objects import *
import networkx as nx
from concurrent.futures import ThreadPoolExecutor, as_completed
import dspy
from nkg.index.extraction.extract_relations import *
from nkg.utils.general import batch_list
from nkg.utils.math_utils import sample_results
from nkg.index.construction.checkpointing import ConstructionCheckpoint
from nkg.utils.token_tracker import usage_tokens


def _bulk_fact_edge_constructor(kind: str, use_cot: bool, neutral_description: bool, edge_descriptions: bool = True,
                                fact_edge_prompt: str = "default"):
    """DSPy module for bulk fact->fact edges. kind: "intra" (same chunk) or "global" (clustered cross-chunk).
    edge_descriptions=False -> the label-only signatures (no description sentence is generated).
    fact_edge_prompt="general" -> the general label-only signatures (extract_relations.GENERAL_FACT_EDGE_GUIDANCE:
    one relation vocabulary for narrative and expository text, links only where one fact explains the other).
    fact_edge_prompt="event" -> BulkIntraChunkFactEdgesEvent (also given the chunk's text) / BulkGlobalFactEdgesEvent
    (extract_relations.EVENT_FACT_EDGE_GUIDANCE: RED / Hong et al. relation families + WILDCARD, specific labels,
    few-shot), label-only."""
    if fact_edge_prompt == "event":
        if edge_descriptions:
            raise ValueError("fact_edge_prompt='event' is label-only: use it with edge_descriptions=False")
        sig = BulkIntraChunkFactEdgesEvent if kind == "intra" else BulkGlobalFactEdgesEvent
    elif fact_edge_prompt == "general":
        if edge_descriptions:
            raise ValueError("fact_edge_prompt='general' is label-only: use it with edge_descriptions=False")
        sig = BulkIntraChunkFactEdgesGeneral if kind == "intra" else BulkGlobalFactEdgesGeneral
    elif fact_edge_prompt != "default":
        raise ValueError(f"fact_edge_prompt must be 'default', 'general' or 'event', got {fact_edge_prompt!r}")
    elif not edge_descriptions:
        sig = BulkIntraChunkFactEdgesLabelOnly if kind == "intra" else BulkGlobalFactEdgesLabelOnly
    elif neutral_description:
        sig = BulkIntraChunkFactEdgesNeutral if kind == "intra" else BulkGlobalFactEdgesNeutral
    else:
        sig = BulkIntraChunkFactEdges if kind == "intra" else BulkGlobalFactEdges
    return (dspy.ChainOfThought if use_cot else dspy.Predict)(sig)


def _fact_edge_descriptions(edge_pair, neutral_description: bool):
    """(forward, backward) text stored in each edge's `description` attribute. A label-only pair has no
    description, so each direction stores its own label there: retrieval reads `description` for scoring,
    path strings and context, and therefore sees only the short label."""
    if isinstance(edge_pair, LabelOnlyFactEdgePair):
        return edge_pair.label_forward, edge_pair.label_backward
    if neutral_description:
        return edge_pair.description, edge_pair.description
    return edge_pair.description_forward, edge_pair.description_backward


def _edge_extra(edge_pair) -> dict:
    """Extra attributes stored on both directed edges: the relation family of an event-prompt pair."""
    if isinstance(edge_pair, EventFactEdgePair):
        return {"relation_family": str(edge_pair.relation_family).strip().upper()}
    return {}


def _intra_chunk_inputs(graph: Graph, chunk_id, fact_edge_prompt: str) -> dict:
    """Context inputs of an intra-chunk bulk call besides numbered_facts: the chunk summary, plus the chunk's own text
    for the event prompt (BulkIntraChunkFactEdgesEvent)."""
    chunk = graph.chunks[chunk_id]
    inputs = {"chunk_summary": chunk.summary}
    if fact_edge_prompt == "event":
        inputs["chunk_text"] = chunk.text
    return inputs
from graspologic.partition import hierarchical_leiden

def _count_tokens_from(history_idx: int) -> dict:
    """Sums DSPy LM history from history_idx onward into a token-count dict."""
    try:
        entries = dspy.settings.lm.history[history_idx:]
        prompt, completion, cache_hit = 0, 0, 0
        for e in entries:
            p, c, hit = usage_tokens(e.get("usage"))
            prompt += p
            completion += c
            cache_hit += hit
        return {"prompt": prompt, "completion": completion, "total": prompt + completion, "calls": len(entries),
                "cache_hit": cache_hit}
    except Exception:
        return {"prompt": 0, "completion": 0, "total": 0, "calls": 0, "cache_hit": 0}


def construct_edges_between_chunks(graph: Graph, use_cot: bool = True):
    chunk_ids = graph.get_chunk_ids()
    chunk_edge_constructor = (dspy.ChainOfThought if use_cot else dspy.Predict)(ChunkEdge)

    for i in range(len(chunk_ids)):
        for j in range(i + 1, len(chunk_ids)):
            chunk1_id = chunk_ids[i]
            chunk2_id = chunk_ids[j]

            # check if there exists an edge between chunk 1 and chunk 2
            if graph.has_chunk_chunk_edge(chunk1_id, chunk2_id):
                continue

            # if there does not exist an edge between chunk 1 and chunk 2, then you can create an edge

            # get chunk summaries
            chunk1_summary = graph.chunks[chunk1_id].summary
            chunk2_summary = graph.chunks[chunk2_id].summary

            # llm call for edge info
            edge_info = chunk_edge_constructor(chunk_summary_1=chunk1_summary, chunk_summary_2=chunk2_summary)

            # add edges between chunks in both directions since this is directed graph with different descriptions
            graph.network.add_edge(chunk1_id, chunk2_id,
                                   description=edge_info.description_1_2,
                                   label=edge_info.label_1_2,
                                   edge_type="chunk_chunk",
                                   score=edge_info.relevance_score,
                                   fact_comparison=False)

            graph.network.add_edge(chunk2_id, chunk1_id,
                                   description=edge_info.description_2_1,
                                   label=edge_info.label_2_1,
                                   edge_type="chunk_chunk",
                                   score=edge_info.relevance_score,
                                   fact_comparison=False)

    print("Finished constructing edges between chunks")


# 1. Define a pure worker function that ONLY talks to the LLM and returns data
def _chunk_edge_worker(chunk_edge_constructor, c1_id, c1_summary, c2_id, c2_summary):
    # This is the slow part. It happens in parallel.
    edge_info = chunk_edge_constructor(chunk_summary_1=c1_summary, chunk_summary_2=c2_summary)

    # Return the IDs and the result. DO NOT modify the graph here.
    return c1_id, c2_id, edge_info


def construct_edges_between_chunks_parallel(graph: Graph, max_workers: int = 10, use_cot: bool = True):
    chunk_ids = graph.get_chunk_ids()
    chunk_edge_constructor = (dspy.ChainOfThought if use_cot else dspy.Predict)(ChunkEdge)

    # 2. Gather all the tasks upfront
    tasks = []
    for i in range(len(chunk_ids)):
        for j in range(i + 1, len(chunk_ids)):
            chunk1_id = chunk_ids[i]
            chunk2_id = chunk_ids[j]

            # Filter out existing edges BEFORE sending to threads
            if not graph.has_chunk_chunk_edge(chunk1_id, chunk2_id):
                tasks.append((
                    chunk1_id, graph.chunks[chunk1_id].summary,
                    chunk2_id, graph.chunks[chunk2_id].summary
                ))

    # 3. Execute the slow LLM calls in parallel
    print(f"Executing {len(tasks)} chunk comparisons across {max_workers} threads...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks to the pool
        futures = {
            executor.submit(_chunk_edge_worker, chunk_edge_constructor, t[0], t[1], t[2], t[3]): t
            for t in tasks
        }

        # 4. Safely update the graph in the main thread as results finish
        for future in as_completed(futures):
            try:
                c1_id, c2_id, edge_info = future.result()

                # Because this loop is in the main thread, networkx mutations are safe
                graph.network.add_edge(c1_id, c2_id,
                                       description=edge_info.description_1_2,
                                       label=edge_info.label_1_2,
                                       score=edge_info.relevance_score,
                                       edge_type="chunk_chunk",
                                       fact_comparison=False)

                graph.network.add_edge(c2_id, c1_id,
                                       description=edge_info.description_2_1,
                                       label=edge_info.label_2_1,
                                       score=edge_info.relevance_score,
                                       edge_type="chunk_chunk",
                                       fact_comparison=False)
            except Exception as e:
                print(f"Error processing chunk edge: {e}")

    print("Finished constructing edges between chunks")

def construct_edges_between_same_chunk_facts(graph: Graph, use_cot: bool = True):
    fact_edge_constructor = (dspy.ChainOfThought if use_cot else dspy.Predict)(SameChunkFactEdge)

    for chunk_id in graph.get_chunk_ids():
        fact_ids = graph.get_fact_ids(by_chunk_id=chunk_id)
        for i in range(len(fact_ids)):
            for j in range(i + 1, len(fact_ids)):
                fact_id1 = fact_ids[i]
                fact_id2 = fact_ids[j]

                # check if edge data already exists
                edge_data = graph.network.get_edge_data(fact_id1, fact_id2)
                if not edge_data is None:
                    continue

                # get info
                chunk_summary = graph.chunks[chunk_id].summary
                fact_sentence_1 = graph.facts[fact_id1].sentence
                fact_sentence_2 = graph.facts[fact_id2].sentence

                # call llm to construct edge info
                edge_info = fact_edge_constructor(chunk_summary=chunk_summary,
                                                  fact_sentence_1=fact_sentence_1,
                                                  fact_sentence_2=fact_sentence_2)

                # create the edges
                graph.network.add_edge(fact_id1, fact_id2,
                                       description=edge_info.description_1_2,
                                       label=edge_info.label_1_2,
                                       edge_type="fact_fact",
                                       score=edge_info.relevance_score)

                graph.network.add_edge(fact_id2, fact_id1,
                                       description=edge_info.description_2_1,
                                       label=edge_info.label_2_1,
                                       edge_type="fact_fact",
                                       score=edge_info.relevance_score)

    print("Finished constructing edges between facts in the same chunk")


def _same_chunk_fact_worker(fact_edge_constructor, chunk_summary, f1_id, f1_sentence, f2_id, f2_sentence):
    # This is the slow I/O bound part that happens in parallel
    edge_info = fact_edge_constructor(
        chunk_summary=chunk_summary,
        fact_sentence_1=f1_sentence,
        fact_sentence_2=f2_sentence
    )

    # Return the IDs and the generated info. DO NOT modify the graph here.
    return f1_id, f2_id, edge_info


def construct_edges_between_same_chunk_facts_parallel(graph: Graph, max_workers: int = 10, use_cot: bool = True):
    fact_edge_constructor = (dspy.ChainOfThought if use_cot else dspy.Predict)(SameChunkFactEdge)

    # 2. Gather all the tasks upfront
    tasks = []

    for chunk_id in graph.get_chunk_ids():
        fact_ids = graph.get_fact_ids(by_chunk_id=chunk_id)
        chunk_summary = graph.chunks[chunk_id].summary

        for i in range(len(fact_ids)):
            for j in range(i + 1, len(fact_ids)):
                fact_id1 = fact_ids[i]
                fact_id2 = fact_ids[j]

                # Filter out existing edges BEFORE sending to threads
                if not graph.network.has_edge(fact_id1, fact_id2):
                    fact_sentence_1 = graph.facts[fact_id1].sentence
                    fact_sentence_2 = graph.facts[fact_id2].sentence

                    # Pack the required data into a tuple
                    tasks.append((
                        chunk_summary,
                        fact_id1, fact_sentence_1,
                        fact_id2, fact_sentence_2
                    ))

    # 3. Execute the slow LLM calls in parallel
    print(f"Executing {len(tasks)} intra-chunk fact comparisons across {max_workers} threads...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks to the pool
        futures = {
            executor.submit(
                _same_chunk_fact_worker,
                fact_edge_constructor,
                t[0], t[1], t[2], t[3], t[4]
            ): t
            for t in tasks
        }

        # 4. Safely update the graph in the main thread as results finish
        for future in as_completed(futures):
            try:
                f1_id, f2_id, edge_info = future.result()

                # Because this loop is in the main thread, networkx mutations are safe
                graph.network.add_edge(
                    f1_id, f2_id,
                    description=edge_info.description_1_2,
                    label=edge_info.label_1_2,
                    edge_type="fact_fact",
                    score=edge_info.relevance_score
                )

                graph.network.add_edge(
                    f2_id, f1_id,
                    description=edge_info.description_2_1,
                    label=edge_info.label_2_1,
                    edge_type="fact_fact",
                    score=edge_info.relevance_score
                )
            except Exception as e:
                print(f"Error processing intra-chunk fact edge: {e}")

    print("Finished constructing edges between facts in the same chunk")


def construct_edges_between_same_chunk_facts_bulk(graph: Graph, use_cot: bool = True, neutral_description: bool = False, same_chunk_min_facts: int = 2, edge_descriptions: bool = True, fact_edge_prompt: str = "default"):
    """
    Constructs intra-chunk fact-to-fact edges using a single LLM call per chunk.

    Instead of one LLM call per fact pair, all facts in a chunk are passed together
    using sequential integer IDs. The LLM identifies relevant pairs and emits all
    edges at once. Integer IDs are then mapped back to UUIDs to build graph edges.

    same_chunk_min_facts — a chunk needs at least this many facts before the LLM call
    is made at all; chunks below the threshold are skipped entirely (no edges, no call).
    """
    bulk_edge_constructor = _bulk_fact_edge_constructor("intra", use_cot, neutral_description, edge_descriptions, fact_edge_prompt)

    for chunk_id in graph.get_chunk_ids():
        fact_ids = graph.get_fact_ids(by_chunk_id=chunk_id)

        if len(fact_ids) < max(2, same_chunk_min_facts):
            continue

        chunk_inputs = _intra_chunk_inputs(graph, chunk_id, fact_edge_prompt)

        # Assign sequential integer IDs and build the numbered facts prompt string.
        # We use ints instead of UUIDs because LLMs are far more reliable at selecting
        # and referencing small integers than full UUID strings.
        int_to_uuid = {}
        numbered_lines = []
        for idx, fact_id in enumerate(fact_ids, start=1):
            int_to_uuid[idx] = fact_id
            numbered_lines.append(f"{idx}: {graph.facts[fact_id].sentence}")

        numbered_facts_str = "\n".join(numbered_lines)

        try:
            result = bulk_edge_constructor(
                **chunk_inputs,
                numbered_facts=numbered_facts_str
            )

            for edge_pair in result.fact_edges:
                src_int = edge_pair.source_id
                tgt_int = edge_pair.target_id

                # Validate that the LLM used real integer IDs from the list
                if src_int not in int_to_uuid or tgt_int not in int_to_uuid:
                    continue
                if src_int == tgt_int:
                    continue
                if edge_pair.relevance_score <= 0.0:
                    continue

                fact_id1 = int_to_uuid[src_int]
                fact_id2 = int_to_uuid[tgt_int]

                if graph.network.has_edge(fact_id1, fact_id2):
                    continue

                desc, desc_back = _fact_edge_descriptions(edge_pair, neutral_description)

                graph.network.add_edge(
                    fact_id1, fact_id2,
                    description=desc,
                    label=edge_pair.label_forward,
                    edge_type="fact_fact",
                    edge_sub_type="same_chunk",
                    score=edge_pair.relevance_score,
                    **_edge_extra(edge_pair)
                )
                graph.network.add_edge(
                    fact_id2, fact_id1,
                    description=desc_back,
                    label=edge_pair.label_backward,
                    edge_type="fact_fact",
                    edge_sub_type="same_chunk",
                    score=edge_pair.relevance_score,
                    **_edge_extra(edge_pair)
                )

        except Exception as e:
            print(f"Error constructing bulk intra-chunk edges for chunk {chunk_id}: {e}")

    print("Finished constructing bulk intra-chunk fact edges.")


def _bulk_same_chunk_fact_worker(bulk_edge_constructor, chunk_id, chunk_inputs, int_to_uuid, numbered_facts_str):
    """Worker: runs one bulk LLM call for a chunk and returns raw results for the main thread to add to the graph.
    chunk_inputs: _intra_chunk_inputs (chunk_summary, and chunk_text for the event prompt)."""
    result = bulk_edge_constructor(
        **chunk_inputs,
        numbered_facts=numbered_facts_str
    )
    return chunk_id, int_to_uuid, result


def construct_edges_between_same_chunk_facts_bulk_parallel(graph: Graph, max_workers: int = 10, use_cot: bool = True, neutral_description: bool = False, same_chunk_min_facts: int = 2, edge_descriptions: bool = True, fact_edge_prompt: str = "default"):
    """
    Parallel version of construct_edges_between_same_chunk_facts_bulk.
    One LLM call per chunk, all calls run in parallel across a thread pool.

    same_chunk_min_facts — a chunk needs at least this many facts before the LLM call
    is made at all; chunks below the threshold are skipped entirely (no edges, no call).
    """
    bulk_edge_constructor = _bulk_fact_edge_constructor("intra", use_cot, neutral_description, edge_descriptions, fact_edge_prompt)

    tasks = []
    for chunk_id in graph.get_chunk_ids():
        fact_ids = graph.get_fact_ids(by_chunk_id=chunk_id)
        if len(fact_ids) < max(2, same_chunk_min_facts):
            continue

        chunk_inputs = _intra_chunk_inputs(graph, chunk_id, fact_edge_prompt)
        int_to_uuid = {}
        numbered_lines = []
        for idx, fact_id in enumerate(fact_ids, start=1):
            int_to_uuid[idx] = fact_id
            numbered_lines.append(f"{idx}: {graph.facts[fact_id].sentence}")

        tasks.append((chunk_id, chunk_inputs, int_to_uuid, "\n".join(numbered_lines)))

    print(f"Executing {len(tasks)} bulk intra-chunk fact evaluations across {max_workers} threads...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_bulk_same_chunk_fact_worker, bulk_edge_constructor, t[0], t[1], t[2], t[3]): t
            for t in tasks
        }

        for future in as_completed(futures):
            try:
                chunk_id, int_to_uuid, result = future.result()

                for edge_pair in result.fact_edges:
                    src_int = edge_pair.source_id
                    tgt_int = edge_pair.target_id

                    if src_int not in int_to_uuid or tgt_int not in int_to_uuid:
                        continue
                    if src_int == tgt_int:
                        continue
                    if edge_pair.relevance_score <= 0.0:
                        continue

                    fact_id1 = int_to_uuid[src_int]
                    fact_id2 = int_to_uuid[tgt_int]

                    if graph.network.has_edge(fact_id1, fact_id2):
                        continue

                    desc, desc_back = _fact_edge_descriptions(edge_pair, neutral_description)

                    graph.network.add_edge(
                        fact_id1, fact_id2,
                        description=desc,
                        label=edge_pair.label_forward,
                        edge_type="fact_fact",
                        edge_sub_type="same_chunk",
                        score=edge_pair.relevance_score,
                        **_edge_extra(edge_pair)
                    )
                    graph.network.add_edge(
                        fact_id2, fact_id1,
                        description=desc_back,
                        label=edge_pair.label_backward,
                        edge_type="fact_fact",
                        edge_sub_type="same_chunk",
                        score=edge_pair.relevance_score,
                        **_edge_extra(edge_pair)
                    )
            except Exception as e:
                print(f"Error processing bulk intra-chunk fact edge: {e}")

    print("Finished constructing bulk intra-chunk fact edges (parallel).")


def construct_edges_between_all_facts(graph: Graph, threshold: float = 0.5, use_cot: bool = True):
    """
    Evaluates highly related chunks and creates deep fact-to-fact relationships
    between them to build cross-chunk context within the knowledge graph.
    """
    # Create DSPy constructors
    fact_edge_constructor = (dspy.ChainOfThought if use_cot else dspy.Predict)(FactEdge)
    candidate_selector = (dspy.ChainOfThought if use_cot else dspy.Predict)(CandidateFacts)

    chunk_ids = graph.get_chunk_ids()

    # Pairwise comparison between all chunks
    for i in range(len(chunk_ids)):
        for j in range(i + 1, len(chunk_ids)):
            chunk1_id = chunk_ids[i]
            chunk2_id = chunk_ids[j]

            # Get edge data between the chunks
            edge_data_1_to_2 = graph.network.get_edge_data(chunk1_id, chunk2_id)
            edge_data_2_to_1 = graph.network.get_edge_data(chunk2_id, chunk1_id)

            # If there does not exist edge data between chunks, skip
            if edge_data_1_to_2 is None or edge_data_2_to_1 is None:
                continue

            # Check if we have already cross-examined facts between these chunks
            if edge_data_1_to_2.get("fact_comparison") is True:
                continue

            # If the edge between chunks has a score above the threshold
            score_1_to_2 = edge_data_1_to_2.get("score", 0.0)

            if score_1_to_2 >= threshold:
                # Extract chunk summaries
                chunk1_summary = graph.chunks[chunk1_id].summary
                chunk2_summary = graph.chunks[chunk2_id].summary

                # Get dictionary of fact IDs to sentences for each chunk
                fact_dict_1 = graph.get_fact_ids(by_chunk_id=chunk1_id, mode="dict")
                fact_dict_2 = graph.get_fact_ids(by_chunk_id=chunk2_id, mode="dict")

                # Extract rich edge descriptions from the chunk-to-chunk level
                desc_1_to_2 = edge_data_1_to_2.get("description", "")
                desc_2_to_1 = edge_data_2_to_1.get("description", "")

                # Select candidate facts via DSPy selector for Chunk 1
                cand_facts_1_result = candidate_selector(
                    chunk_summary_1=chunk1_summary,
                    chunk_summary_2=chunk2_summary,
                    rich_edge_description=desc_1_to_2,
                    fact_dict=fact_dict_1
                )

                # Select candidate facts via DSPy selector for Chunk 2
                # (Swapping summaries so it views the relationship from Chunk 2's perspective)
                cand_facts_2_result = candidate_selector(
                    chunk_summary_1=chunk2_summary,
                    chunk_summary_2=chunk1_summary,
                    rich_edge_description=desc_2_to_1,
                    fact_dict=fact_dict_2
                )

                # Filter the LLM output list for fact IDs that actually exist in the graph
                valid_cand_facts_1 = [fid for fid in cand_facts_1_result.candidate_facts if fid in graph.facts]
                valid_cand_facts_2 = [fid for fid in cand_facts_2_result.candidate_facts if fid in graph.facts]

                # Pairwise comparison between the candidate facts of chunk 1 and chunk 2
                for fact1_id in valid_cand_facts_1:
                    for fact2_id in valid_cand_facts_2:

                        # Between the candidate facts, check if there already exists an edge
                        if graph.network.has_edge(fact1_id, fact2_id):
                            continue

                        # Extract important info needed for the LLM call
                        fact1_sentence = graph.facts[fact1_id].sentence
                        fact2_sentence = graph.facts[fact2_id].sentence

                        # Call LLM via DSPy constructor to create edge info
                        fact_edge_info = fact_edge_constructor(
                            fact_sentence_1=fact1_sentence,
                            chunk_summary_1=chunk1_summary,
                            fact_sentence_2=fact2_sentence,
                            chunk_summary_2=chunk2_summary
                        )

                        # Create an edge both ways between each fact
                        graph.network.add_edge(
                            fact1_id, fact2_id,
                            description=fact_edge_info.description_1_2,
                            label=fact_edge_info.label_1_2,
                            edge_type="fact_fact",
                            score=fact_edge_info.relevance_score
                        )

                        graph.network.add_edge(
                            fact2_id, fact1_id,
                            description=fact_edge_info.description_2_1,
                            label=fact_edge_info.label_2_1,
                            edge_type="fact_fact",
                            score=fact_edge_info.relevance_score
                        )

                # Finally, set the attribute fact_comparison to True to prevent redundant future checks
                graph.network[chunk1_id][chunk2_id]["fact_comparison"] = True
                graph.network[chunk2_id][chunk1_id]["fact_comparison"] = True
    print("Finished constructing edges between all facts.")


def _candidate_worker(candidate_selector, c1_id, c2_id, c1_sum, c2_sum, desc_1_2, desc_2_1, dict_1, dict_2):
    """Worker for Phase 1: Retrieves candidate facts for both directions of a chunk pair."""
    cand_1_res = candidate_selector(
        chunk_summary_1=c1_sum,
        chunk_summary_2=c2_sum,
        rich_edge_description=desc_1_2,
        fact_dict=dict_1
    )

    cand_2_res = candidate_selector(
        chunk_summary_1=c2_sum,
        chunk_summary_2=c1_sum,
        rich_edge_description=desc_2_1,
        fact_dict=dict_2
    )

    return c1_id, c2_id, cand_1_res.candidate_facts, cand_2_res.candidate_facts


def _cross_chunk_fact_worker(fact_edge_constructor, f1_id, f2_id, f1_sent, f2_sent, c1_sum, c2_sum):
    """Worker for Phase 2: Generates the edge description between two candidate facts."""
    edge_info = fact_edge_constructor(
        fact_sentence_1=f1_sent,
        chunk_summary_1=c1_sum,
        fact_sentence_2=f2_sent,
        chunk_summary_2=c2_sum
    )
    return f1_id, f2_id, edge_info


# --- MAIN PARALLELIZED FUNCTION ---

def construct_edges_between_all_facts_parallel(graph: Graph, threshold: float = 0.5, max_workers: int = 10, use_cot: bool = True):
    """
    Evaluates highly related chunks and creates deep fact-to-fact relationships
    between them to build cross-chunk context within the knowledge graph.
    """
    fact_edge_constructor = (dspy.ChainOfThought if use_cot else dspy.Predict)(FactEdge)
    candidate_selector = (dspy.ChainOfThought if use_cot else dspy.Predict)(CandidateFacts)

    chunk_ids = graph.get_chunk_ids()
    candidate_tasks = []

    # ==========================================
    # PREPARATION: Gather chunk pairs for Phase 1
    # ==========================================
    for i in range(len(chunk_ids)):
        for j in range(i + 1, len(chunk_ids)):
            chunk1_id = chunk_ids[i]
            chunk2_id = chunk_ids[j]

            edge_data_1_to_2 = graph.network.get_edge_data(chunk1_id, chunk2_id)
            edge_data_2_to_1 = graph.network.get_edge_data(chunk2_id, chunk1_id)

            if edge_data_1_to_2 is None or edge_data_2_to_1 is None:
                continue
            if edge_data_1_to_2.get("fact_comparison") is True:
                continue

            score_1_to_2 = edge_data_1_to_2.get("score", 0.0)
            if score_1_to_2 >= threshold:
                candidate_tasks.append((
                    chunk1_id, chunk2_id,
                    graph.chunks[chunk1_id].summary,
                    graph.chunks[chunk2_id].summary,
                    edge_data_1_to_2.get("description", ""),
                    edge_data_2_to_1.get("description", ""),
                    graph.get_fact_ids(by_chunk_id=chunk1_id, mode="dict"),
                    graph.get_fact_ids(by_chunk_id=chunk2_id, mode="dict")
                ))

    # ==========================================
    # PHASE 1: Parallel Candidate Fact Selection
    # ==========================================
    candidate_results = []
    print(f"Phase 1: Executing {len(candidate_tasks)} Candidate Fact selections across {max_workers} threads...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_candidate_worker, candidate_selector, *t): t
            for t in candidate_tasks
        }
        for future in as_completed(futures):
            try:
                candidate_results.append(future.result())
            except Exception as e:
                print(f"Error processing candidate selector: {e}")

    # ==========================================
    # PREPARATION: Gather fact pairs for Phase 2
    # ==========================================
    fact_edge_tasks = []
    chunks_to_mark = []  # We track this to mark them as completed later

    for c1_id, c2_id, cands_1, cands_2 in candidate_results:
        # Save these chunk IDs so we can flag them as completed at the end
        chunks_to_mark.append((c1_id, c2_id))

        # Filter valid facts directly from the LLM outputs
        valid_1 = [fid for fid in cands_1 if fid in graph.facts]
        valid_2 = [fid for fid in cands_2 if fid in graph.facts]

        c1_summary = graph.chunks[c1_id].summary
        c2_summary = graph.chunks[c2_id].summary

        for f1_id in valid_1:
            for f2_id in valid_2:
                # Filter out existing edges before doing LLM math
                if not graph.network.has_edge(f1_id, f2_id):
                    fact_edge_tasks.append((
                        f1_id, f2_id,
                        graph.facts[f1_id].sentence,
                        graph.facts[f2_id].sentence,
                        c1_summary, c2_summary
                    ))

    # ==========================================
    # PHASE 2: Parallel Cross-Chunk Fact Linking
    # ==========================================
    print(f"Phase 2: Executing {len(fact_edge_tasks)} Cross-Chunk Fact evaluations across {max_workers} threads...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_cross_chunk_fact_worker, fact_edge_constructor, *t): t
            for t in fact_edge_tasks
        }
        for future in as_completed(futures):
            try:
                f1_id, f2_id, edge_info = future.result()

                # Safely write to NetworkX in the main thread
                graph.network.add_edge(
                    f1_id, f2_id,
                    description=edge_info.description_1_2,
                    label=edge_info.label_1_2,
                    edge_type="fact_fact",
                    score=edge_info.relevance_score
                )
                graph.network.add_edge(
                    f2_id, f1_id,
                    description=edge_info.description_2_1,
                    label=edge_info.label_2_1,
                    edge_type="fact_fact",
                    score=edge_info.relevance_score
                )
            except Exception as e:
                print(f"Error processing cross-chunk fact edge: {e}")

    # ==========================================
    # CLEANUP: Mark chunk comparisons as complete
    # ==========================================
    for c1_id, c2_id in chunks_to_mark:
        graph.network[c1_id][c2_id]["fact_comparison"] = True
        graph.network[c2_id][c1_id]["fact_comparison"] = True

    print("Finished constructing edges between all facts.")

def construct_edges_between_entities_and_facts(graph: Graph, entity_batch_size=5, use_cot: bool = True):
    """
    Constructs edges going from Entities back to their parent Facts.
    Uses batched LLM calls to generate the specific relationship labels.
    """
    # 1. Create DSPy constructor
    entity_fact_edge_constructor = (dspy.ChainOfThought if use_cot else dspy.Predict)(EntityFactEdge)

    # 2. Collect all fact IDs
    fact_ids = graph.get_fact_ids()

    for fact_id in fact_ids:
        # 3. Get name and sentence for this fact
        fact = graph.facts[fact_id]
        fact_sentence = fact.sentence
        fact_name = fact_sentence[:80]

        # 4. Find the set of entities that the fact is connected to
        connected_entity_ids = []
        for _, target_node, edge_data in graph.network.out_edges(fact_id, data=True):
            if edge_data.get("edge_type") == "fact_entity":
                connected_entity_ids.append(target_node)

        if not connected_entity_ids:
            continue

        # 5. Batch the entities using the general utils function
        entity_batches = batch_list(connected_entity_ids, max_batch_size=entity_batch_size)

        for batch in entity_batches:
            filtered_entities = []
            name_to_id_map = {}

            # 6. Filter entities by checking if the reverse edge (Entity -> Fact) already exists
            for ent_id in batch:
                if not graph.network.has_edge(ent_id, fact_id):
                    filtered_entities.append(ent_id)
                    # Keep dictionary mapping entity name to entity ID for the LLM
                    ent_name = graph.entities[ent_id].name
                    name_to_id_map[ent_name] = ent_id

            # Skip if all entities in this batch already have reverse edges
            if not filtered_entities:
                continue

            # 7. Call LLM to create edge info
            entity_names_list = list(name_to_id_map.keys())

            try:
                edge_info = entity_fact_edge_constructor(
                    fact_name=fact_name,
                    fact_sentence=fact_sentence,
                    EntityNames=entity_names_list
                )

                # 8. Create the edges using the mapped IDs
                for relation in edge_info.relations:
                    generated_name = relation.entity_name
                    relation_label = relation.relation_name

                    # Ensure the LLM didn't hallucinate a slightly different name
                    if generated_name in name_to_id_map:
                        mapped_ent_id = name_to_id_map[generated_name]

                        graph.network.add_edge(
                            mapped_ent_id,
                            fact_id,
                            edge_type="entity_fact",
                            label=relation_label
                        )
            except Exception as e:
                # Always good to catch exceptions per batch so one bad LLM output doesn't crash the loop
                print(f"Error generating entity-fact edges for fact {fact_name}: {e}")

    print("Finished constructing edges between entities and facts.")


def _entity_fact_worker(entity_fact_edge_constructor, fact_id, fact_name, fact_sentence, entity_names_list,
                        name_to_id_map):
    """
    Worker for isolating the LLM call.
    It passes through the fact_id and the name_to_id_map so the main thread
    has everything it needs to safely reconstruct the graph edges.
    """
    edge_info = entity_fact_edge_constructor(
        fact_name=fact_name,
        fact_sentence=fact_sentence,
        EntityNames=entity_names_list
    )

    # Return the mapped data alongside the DSPy output
    return fact_id, name_to_id_map, edge_info


# --- MAIN PARALLELIZED FUNCTION ---

def construct_edges_between_entities_and_facts_parallel(graph: Graph, entity_batch_size: int = 5, max_workers: int = 10, use_cot: bool = True):
    """
    Constructs edges going from Entities back to their parent Facts.
    Uses threaded, batched LLM calls to generate the specific relationship labels.
    """
    entity_fact_edge_constructor = (dspy.ChainOfThought if use_cot else dspy.Predict)(EntityFactEdge)
    fact_ids = graph.get_fact_ids()

    tasks = []

    # ==========================================
    # PREPARATION: Gather tasks
    # ==========================================
    for fact_id in fact_ids:
        fact = graph.facts[fact_id]
        fact_sentence = fact.sentence
        fact_name = fact_sentence[:80]

        # Find the set of entities that the fact is connected to
        connected_entity_ids = []
        for _, target_node, edge_data in graph.network.out_edges(fact_id, data=True):
            if edge_data.get("edge_type") == "fact_entity":
                connected_entity_ids.append(target_node)

        if not connected_entity_ids:
            continue

        # Batch the entities
        entity_batches = batch_list(connected_entity_ids, max_batch_size=entity_batch_size)

        for batch in entity_batches:
            name_to_id_map = {}

            # Filter entities by checking if the reverse edge already exists
            for ent_id in batch:
                if not graph.network.has_edge(ent_id, fact_id):
                    ent_name = graph.entities[ent_id].name
                    name_to_id_map[ent_name] = ent_id

            # If all entities in this batch are already linked, skip
            if not name_to_id_map:
                continue

            # Add to the task list
            entity_names_list = list(name_to_id_map.keys())
            tasks.append((
                fact_id,
                fact_name,
                fact_sentence,
                entity_names_list,
                name_to_id_map
            ))

    # ==========================================
    # EXECUTION: Parallel LLM Calls
    # ==========================================
    print(f"Executing {len(tasks)} Entity-Fact batches across {max_workers} threads...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(_entity_fact_worker, entity_fact_edge_constructor, *t): t
            for t in tasks
        }

        # ==========================================
        # RESOLUTION: Safe Graph Mutation
        # ==========================================
        for future in as_completed(futures):
            try:
                fact_id, name_to_id_map, edge_info = future.result()

                for relation in edge_info.relations:
                    generated_name = relation.entity_name
                    relation_label = relation.relation_name

                    # Ensure the LLM didn't hallucinate a slightly different name
                    if generated_name in name_to_id_map:
                        mapped_ent_id = name_to_id_map[generated_name]

                        # Safely mutate the graph in the main thread
                        graph.network.add_edge(
                            mapped_ent_id,
                            fact_id,
                            edge_type="entity_fact",
                            label=relation_label
                        )
            except Exception as e:
                # Catching exceptions dynamically per task prevents a single bad prompt from failing the run
                print(f"Error generating entity-fact edges: {e}")

    print("Finished constructing edges between entities and facts.")

def _get_parent_chunk_summary(graph: Graph, fact_id: str) -> str:
    """
    Traverses incoming edges to a Fact node to find its parent Chunk,
    returning the chunk's summary for LLM context.
    """
    # Look at edges coming INTO the fact
    for source_node, target_node, edge_data in graph.network.in_edges(fact_id, data=True):
        if edge_data.get("edge_type") == "chunk_fact":
             # We found the parent chunk, return its summary
            return graph.chunks[source_node].summary
    return ""


def construct_edges_between_facts_linear(graph: Graph, top_k: int, skip_percent: float = 0.25,
                                         retrieval_model: str = "all-MiniLM-L6-v2", use_cot: bool = True):
    """
    Constructs fact-to-fact edges in linear time using multi-dimensional vector search
    and rank-based sampling.
    """
    print("Initializing fact embeddings for vector search...")
    graph.init_fact_embeddings(retrieval_model=retrieval_model)

    fact_edge_constructor = (dspy.ChainOfThought if use_cot else dspy.Predict)(FactEdge)
    fact_ids = graph.get_fact_ids()

    print(f"Evaluating {len(fact_ids)} facts linearly...")
    for source_id in fact_ids:
        source_fact = graph.facts[source_id]

        # 1. Retrieve all facts ranked by multi-dimensional relevance
        all_ids, all_scores = graph.get_relevant_seeds_by_id(source_id, get_all=True)

        # 2. Sample Top-K (Immediate Relevance)
        top_ids, _ = sample_results(all_ids, all_scores, strategy="top_k", k=top_k)

        # 3. Sample Middle-K (Discovery / Skipping top percentile)
        mid_ids, _ = sample_results(
            all_ids, all_scores,
            strategy="skip_percent_top_k",
            k=top_k,
            skip_top_percent=skip_percent
        )

        # 4. Combine, deduplicate, and remove self-references
        candidate_targets = set(top_ids + mid_ids)
        if source_id in candidate_targets:
            candidate_targets.remove(source_id)

        source_chunk_summary = _get_parent_chunk_summary(graph, source_id)

        # 5. Evaluate top-k candidates first, then mid-k only candidates
        # Tracking them separately so we can tag the edge_sub_type accurately.
        top_set = set(top_ids) - {source_id}
        mid_only_set = (set(mid_ids) - {source_id}) - top_set

        for target_id, edge_sub_type in [(tid, "global_topk_linear") for tid in top_set] + \
                                         [(tid, "global_midk_linear") for tid in mid_only_set]:
            if graph.network.has_edge(source_id, target_id):
                continue

            target_fact = graph.facts[target_id]
            target_chunk_summary = _get_parent_chunk_summary(graph, target_id)

            edge_info = fact_edge_constructor(
                fact_sentence_1=source_fact.sentence,
                chunk_summary_1=source_chunk_summary,
                fact_sentence_2=target_fact.sentence,
                chunk_summary_2=target_chunk_summary
            )

            graph.network.add_edge(
                source_id, target_id,
                description=edge_info.description_1_2,
                label=edge_info.label_1_2,
                edge_type="fact_fact",
                edge_sub_type=edge_sub_type,
                score=edge_info.relevance_score
            )
            graph.network.add_edge(
                target_id, source_id,
                description=edge_info.description_2_1,
                label=edge_info.label_2_1,
                edge_type="fact_fact",
                edge_sub_type=edge_sub_type,
                score=edge_info.relevance_score
            )

    print("Finished constructing linear fact edges.")


def construct_edges_between_facts_linear_parallel(graph: Graph, top_k: int, skip_percent: float = 0.25,
                                                  retrieval_model: str = "all-MiniLM-L6-v2", max_workers: int = 10,
                                                  use_cot: bool = True):
    """
    Parallel version of linear edge construction. Computes vector search
    in the main thread, then delegates DSPy LLM calls to a thread pool.
    """
    print("Initializing fact embeddings for parallel vector search...")
    graph.init_fact_embeddings(retrieval_model=retrieval_model)

    fact_edge_constructor = (dspy.ChainOfThought if use_cot else dspy.Predict)(FactEdge)
    fact_ids = graph.get_fact_ids()

    tasks = []
    seen_pairs = set()  # Tracks unique comparisons to prevent redundant bidirectional checks

    # ==========================================
    # PHASE 1: Fast Vector Search & Sampling
    # ==========================================
    # Tasks are 7-tuples: (src_id, tgt_id, src_sent, tgt_sent, src_chunk, tgt_chunk, edge_sub_type)
    # Top-k and mid-k pairs are tracked separately so we can tag edge_sub_type accurately.
    print("Calculating vector similarities and generating task queue...")
    for source_id in fact_ids:
        source_fact = graph.facts[source_id]

        all_ids, all_scores = graph.get_relevant_seeds_by_id(source_id, get_all=True)
        top_ids, _ = sample_results(all_ids, all_scores, strategy="top_k", k=top_k)
        mid_ids, _ = sample_results(all_ids, all_scores, strategy="skip_percent_top_k", k=top_k,
                                    skip_top_percent=skip_percent)

        top_set = set(top_ids) - {source_id}
        mid_only_set = (set(mid_ids) - {source_id}) - top_set

        source_chunk_summary = _get_parent_chunk_summary(graph, source_id)

        for target_id, edge_sub_type in [(tid, "global_topk_linear") for tid in top_set] + \
                                         [(tid, "global_midk_linear") for tid in mid_only_set]:
            pair_key = tuple(sorted([source_id, target_id]))
            if pair_key not in seen_pairs and not graph.network.has_edge(source_id, target_id):
                seen_pairs.add(pair_key)
                target_fact = graph.facts[target_id]
                target_chunk_summary = _get_parent_chunk_summary(graph, target_id)
                tasks.append((
                    source_id, target_id,
                    source_fact.sentence, target_fact.sentence,
                    source_chunk_summary, target_chunk_summary,
                    edge_sub_type
                ))

    # ==========================================
    # PHASE 2: Parallel LLM Execution
    # ==========================================
    # Worker only receives the first 6 elements (not edge_sub_type).
    # We retrieve edge_sub_type from the task via the futures dict after completion.
    print(f"Executing {len(tasks)} Fact-to-Fact evaluations across {max_workers} threads...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_cross_chunk_fact_worker, fact_edge_constructor, *t[:6]): t
            for t in tasks
        }

        for future in as_completed(futures):
            try:
                task = futures[future]
                edge_sub_type = task[6]
                f1_id, f2_id, edge_info = future.result()

                graph.network.add_edge(
                    f1_id, f2_id,
                    description=edge_info.description_1_2,
                    label=edge_info.label_1_2,
                    edge_type="fact_fact",
                    edge_sub_type=edge_sub_type,
                    score=edge_info.relevance_score
                )
                graph.network.add_edge(
                    f2_id, f1_id,
                    description=edge_info.description_2_1,
                    label=edge_info.label_2_1,
                    edge_type="fact_fact",
                    edge_sub_type=edge_sub_type,
                    score=edge_info.relevance_score
                )
            except Exception as e:
                print(f"Error processing parallel linear fact edge: {e}")

    print("Finished constructing parallel linear fact edges.")


def _bulk_global_fact_worker(bulk_edge_constructor, int_to_uuid, numbered_facts_str):
    """Worker: runs one bulk LLM call for a cluster and returns raw results for the main thread."""
    result = bulk_edge_constructor(numbered_facts=numbered_facts_str)
    return int_to_uuid, result


def _run_bulk_fact_edge_llm(graph: Graph, indexed_cluster_lists: list, bulk_edge_constructor, max_workers: int, edge_sub_type: str, neutral_description: bool = False, on_cluster_done=None):
    """
    Prepares numbered-fact strings for each cluster, runs parallel LLM calls,
    then writes the resulting edges to the graph (main thread only).
    Called once per clustering round with the appropriate edge_sub_type tag.

    indexed_cluster_lists — list of (cluster_idx, cluster_fact_ids) tuples. cluster_idx is the
    cluster's position in the round's full cluster list (stable across resumes) so a checkpoint
    can track exactly which clusters are done; callers filter out already-completed clusters
    before calling this.

    on_cluster_done(cluster_idx, edges) — optional callback invoked in the main thread right
    after a cluster's edges are written into `graph`. `edges` is the list of
    {source, target, description, label, edge_type, edge_sub_type, score} dicts just added
    (both directions), possibly empty if the cluster produced no qualifying edges. Called
    exactly once per successfully-processed cluster (not called if the LLM call itself raised).
    """
    tasks = []
    for cluster_idx, cluster_fact_ids in indexed_cluster_lists:
        int_to_uuid = {}
        numbered_lines = []
        for idx, fact_id in enumerate(cluster_fact_ids, start=1):
            int_to_uuid[idx] = fact_id
            fact = graph.facts[fact_id]
            chunk_summary = _get_parent_chunk_summary(graph, fact_id)
            numbered_lines.append(f"{idx}: [Context: {chunk_summary}] {fact.sentence}")
        tasks.append((cluster_idx, int_to_uuid, "\n".join(numbered_lines)))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_bulk_global_fact_worker, bulk_edge_constructor, t[1], t[2]): t[0]
            for t in tasks
        }
        for future in as_completed(futures):
            cluster_idx = futures[future]
            try:
                int_to_uuid, result = future.result()
                new_edges = []
                for edge_pair in result.fact_edges:
                    src_int = edge_pair.source_id
                    tgt_int = edge_pair.target_id

                    if src_int not in int_to_uuid or tgt_int not in int_to_uuid:
                        continue
                    if src_int == tgt_int:
                        continue
                    if edge_pair.relevance_score <= 0.0:
                        continue

                    fact_id1 = int_to_uuid[src_int]
                    fact_id2 = int_to_uuid[tgt_int]

                    if graph.network.has_edge(fact_id1, fact_id2):
                        continue

                    desc, desc_back = _fact_edge_descriptions(edge_pair, neutral_description)
                    # event prompt (EventFactEdgePair): the relation family travels with both directed edges
                    extra = _edge_extra(edge_pair)

                    graph.network.add_edge(
                        fact_id1, fact_id2,
                        description=desc,
                        label=edge_pair.label_forward,
                        edge_type="fact_fact",
                        edge_sub_type=edge_sub_type,
                        score=edge_pair.relevance_score,
                        **extra
                    )
                    graph.network.add_edge(
                        fact_id2, fact_id1,
                        description=desc_back,
                        label=edge_pair.label_backward,
                        edge_type="fact_fact",
                        edge_sub_type=edge_sub_type,
                        score=edge_pair.relevance_score,
                        **extra
                    )
                    new_edges.append({"source": fact_id1, "target": fact_id2, "description": desc,
                                       "label": edge_pair.label_forward, "edge_type": "fact_fact",
                                       "edge_sub_type": edge_sub_type, "score": edge_pair.relevance_score, **extra})
                    new_edges.append({"source": fact_id2, "target": fact_id1, "description": desc_back,
                                       "label": edge_pair.label_backward, "edge_type": "fact_fact",
                                       "edge_sub_type": edge_sub_type, "score": edge_pair.relevance_score, **extra})

                if on_cluster_done is not None:
                    on_cluster_done(cluster_idx, new_edges)
            except Exception as e:
                print(f"Error processing clustered fact edge batch (cluster {cluster_idx}): {e}")


_ZERO_TOKENS = {"prompt": 0, "completion": 0, "total": 0, "calls": 0, "cache_hit": 0}


def _add_tokens(a: dict, b: dict) -> dict:
    return {
        "prompt":     a.get("prompt", 0) + b.get("prompt", 0),
        "completion": a.get("completion", 0) + b.get("completion", 0),
        "total":      a.get("total", 0) + b.get("total", 0),
        "calls":      a.get("calls", 0) + b.get("calls", 0),
        "cache_hit":  a.get("cache_hit", 0) + b.get("cache_hit", 0),
    }


def _process_clustered_round(graph: Graph, round_num: int, use_round: bool, sim_graph_builder, max_cluster_size: int,
                             bulk_edge_constructor, max_workers: int, edge_sub_type: str, neutral_description: bool,
                             checkpoint, cluster_fn=None) -> dict:
    """
    Handles checkpoint resume/replay + Leiden clustering + bulk LLM edge construction for one
    round of clustered global fact-edge construction. Returns this round's cumulative token
    dict (summed across every resume attempt, via `checkpoint`; just this attempt's tokens
    when checkpointing is disabled).

    sim_graph_builder — zero-arg callable that builds and returns this round's nx.Graph of
    candidate fact pairs. Only called when there's no already-saved cluster list to resume from,
    so a resumed run never rebuilds embeddings/re-clusters work that's already been done.

    cluster_fn — optional callable(sim_graph) -> list of cluster fact-id lists, replacing the
    built-in grouping below (used by fast_fact_clustering, edge_mode="clustered_fast"). None =
    the built-in grouping, which keeps every hierarchical_leiden level (see fast_fact_clustering).
    """
    if not use_round:
        print(f"Round {round_num}: skipped (use_round{round_num}=False).")
        return dict(_ZERO_TOKENS)

    if checkpoint.round_status(round_num) == "done":
        print(f"Round {round_num}: already complete (checkpoint) — replaying saved edges.")
        checkpoint.replay_round_edges(round_num, graph)
        return checkpoint.round_tokens(round_num)

    before_tokens = checkpoint.round_tokens(round_num)

    if checkpoint.has_cluster_list(round_num):
        cluster_lists = checkpoint.load_cluster_list(round_num)
        completed_before = checkpoint.completed_cluster_indices(round_num)
        checkpoint.replay_round_edges(round_num, graph)
        print(f"Round {round_num}: resuming — {len(completed_before)}/{len(cluster_lists)} clusters already done.")
    else:
        print(f"Round {round_num}: building similarity graph...")
        sim_graph = sim_graph_builder()
        print(f"Round {round_num} sim graph: {sim_graph.number_of_edges()} candidate edges.")
        if cluster_fn is not None:
            cluster_lists = cluster_fn(sim_graph)
        elif sim_graph.number_of_edges() > 0:
            clusters_raw = hierarchical_leiden(sim_graph, max_cluster_size=max_cluster_size, random_seed=42)
            clusters = {}
            for node in clusters_raw:
                clusters.setdefault(node.cluster, []).append(node.node)
            cluster_lists = [c for c in clusters.values() if len(c) > 1]
        else:
            cluster_lists = []
        checkpoint.save_cluster_list(round_num, cluster_lists)
        completed_before = set()
        print(f"Round {round_num}: {len(cluster_lists)} clusters → running {len(cluster_lists)} bulk LLM calls...")

    remaining = [(i, c) for i, c in enumerate(cluster_lists) if i not in completed_before]
    if not remaining:
        checkpoint.mark_round_done(round_num)
        return before_tokens

    succeeded = set()

    def _on_done(cluster_idx, edges):
        succeeded.add(cluster_idx)
        checkpoint.append_cluster_edges(round_num, cluster_idx, edges)

    try: _h = len(dspy.settings.lm.history)
    except Exception: _h = 0
    _t0 = time.time()
    _run_bulk_fact_edge_llm(graph, remaining, bulk_edge_constructor, max_workers,
                            edge_sub_type=edge_sub_type, neutral_description=neutral_description,
                            on_cluster_done=_on_done)
    attempt_tokens = _count_tokens_from(_h)
    attempt_time = time.time() - _t0
    checkpoint.add_round_tokens_and_time(round_num, attempt_tokens, attempt_time)

    total_done = completed_before | succeeded
    if total_done == set(range(len(cluster_lists))):
        checkpoint.mark_round_done(round_num)
    else:
        missing = set(range(len(cluster_lists))) - total_done
        print(f"Round {round_num}: {len(missing)} cluster(s) failed this attempt and will be retried on next resume.")

    return _add_tokens(before_tokens, attempt_tokens)


def construct_edges_between_facts_clustered(graph: Graph, top_k: int = 10, skip_percent: float = 0.33,
                                            max_cluster_size: int = 20, max_workers: int = 10,
                                            retrieval_model: str = "all-MiniLM-L6-v2", use_cot: bool = True,
                                            neutral_description: bool = False,
                                            use_round1: bool = True, use_round2: bool = True, use_round3: bool = True,
                                            edge_descriptions: bool = True, fact_edge_prompt: str = "default",
                                            r1_weights: dict = None, r2_weights: dict = None, r3_weights: dict = None,
                                            questions_mode: str = "follow_up_to_answered",
                                            checkpoint=None):
    """
    Builds global cross-chunk fact-to-fact edges using 3 rounds of similarity-based
    clustering followed by one bulk LLM call per cluster.

    Round 1 — Entity-based (entity_weight=1.0, all others=0.0), top_k:
        Groups facts that share the same entities/roles. Catches relationships
        between facts about the same real-world objects even if their sentences
        are phrased very differently.

    Round 2 — Balanced top_k (entity_weight=0.5, all others=1.0):
        Groups the most semantically similar facts overall, with entity overlap
        as a secondary signal.

    Round 3 — Balanced middle_k (same weights as Round 2, skipping the top percentile):
        Discovery round — surfaces less-obvious but still meaningful connections
        that were too similar to appear in the top-k (already covered) but still
        sit in a plausible relevance band.

    In all rounds, fact pairs that already share an edge in the graph are excluded
    from the similarity graph before clustering, so no LLM work is duplicated.
    Edges created in Round 1 are naturally excluded by Round 2's filter, and so on.
    """
    checkpoint = checkpoint if checkpoint is not None else ConstructionCheckpoint(None)

    print("Initializing fact embeddings for clustered edge construction...")
    graph.init_fact_embeddings(retrieval_model=retrieval_model)

    # Resolve per-round similarity weights, falling back to documented defaults.
    _r1 = r1_weights or {"sent": 0.0, "topic": 0.0, "entity": 1.0, "questions": 0.0}
    _r2 = r2_weights or {"sent": 1.0, "topic": 1.0, "entity": 1.0, "questions": 1.0}
    _r3 = r3_weights or {"sent": 1.0, "topic": 1.0, "entity": 0.5, "questions": 1.0}

    fact_ids = graph.get_fact_ids()
    bulk_edge_constructor = _bulk_fact_edge_constructor("global", use_cot, neutral_description, edge_descriptions, fact_edge_prompt)

    def _build_sim_graph(weights: dict, strategy: str, skip_top_percent: float = 0.0) -> nx.Graph:
        sim_graph = nx.Graph()
        sim_graph.add_nodes_from(fact_ids)
        for fact_id in fact_ids:
            all_ids, all_scores = graph.get_relevant_seeds_by_id(
                fact_id, get_all=True,
                sent_weight=weights["sent"], topic_weight=weights["topic"],
                entity_weight=weights["entity"], questions_weight=weights["questions"],
                questions_mode=questions_mode
            )
            id_to_score = dict(zip(all_ids, all_scores))
            if strategy == "top_k":
                candidate_ids, _ = sample_results(all_ids, all_scores, strategy="top_k", k=top_k)
            else:
                candidate_ids, _ = sample_results(all_ids, all_scores, strategy="skip_percent_top_k",
                                                  k=top_k, skip_top_percent=skip_top_percent)
            for candidate_id in candidate_ids:
                if candidate_id == fact_id:
                    continue
                if graph.network.has_edge(fact_id, candidate_id):
                    continue
                sim_graph.add_edge(fact_id, candidate_id, weight=max(0.01, float(id_to_score[candidate_id])))
        return sim_graph

    # ==========================================
    # ROUND 1: Entity-based clustering → top_k
    # ==========================================
    round1_tokens = _process_clustered_round(
        graph, 1, use_round1,
        sim_graph_builder=lambda: _build_sim_graph(_r1, "top_k"),
        max_cluster_size=max_cluster_size, bulk_edge_constructor=bulk_edge_constructor,
        max_workers=max_workers, edge_sub_type="global_topk_entity_cluster",
        neutral_description=neutral_description, checkpoint=checkpoint,
    )

    # ==========================================
    # ROUND 2: Balanced top-k clustering
    # ==========================================
    round2_tokens = _process_clustered_round(
        graph, 2, use_round2,
        sim_graph_builder=lambda: _build_sim_graph(_r2, "top_k"),
        max_cluster_size=max_cluster_size, bulk_edge_constructor=bulk_edge_constructor,
        max_workers=max_workers, edge_sub_type="global_topk_balanced",
        neutral_description=neutral_description, checkpoint=checkpoint,
    )

    # ==========================================
    # ROUND 3: Balanced middle-k clustering (skips top percentile)
    # ==========================================
    round3_tokens = _process_clustered_round(
        graph, 3, use_round3,
        sim_graph_builder=lambda: _build_sim_graph(_r3, "middle_k", skip_top_percent=skip_percent),
        max_cluster_size=max_cluster_size, bulk_edge_constructor=bulk_edge_constructor,
        max_workers=max_workers, edge_sub_type="global_middlek_balanced_cluster",
        neutral_description=neutral_description, checkpoint=checkpoint,
    )

    print("Finished constructing clustered global fact edges.")
    return {"round1": round1_tokens, "round2": round2_tokens, "round3": round3_tokens}


def construct_initial_edges(graph: Graph, threshold: float = 0.5, max_workers: int = 10, mode: str = "linear", use_cot: bool = True, neutral_description: bool = False, same_chunk_min_facts: int = 2, edge_descriptions: bool = True, fact_edge_prompt: str = "default"):
    if mode == "linear":
        # Entity→fact edges are created during fact extraction (BatchedFactAssembler) using
        # entity_relation_labels stored on each Fact, so no separate entity edge step is needed here.
        if max_workers == 1:
            construct_edges_between_same_chunk_facts_bulk(graph, use_cot=use_cot, neutral_description=neutral_description, same_chunk_min_facts=same_chunk_min_facts, edge_descriptions=edge_descriptions, fact_edge_prompt=fact_edge_prompt)
        else:
            construct_edges_between_same_chunk_facts_bulk_parallel(graph, max_workers=max_workers, use_cot=use_cot, neutral_description=neutral_description, same_chunk_min_facts=same_chunk_min_facts, edge_descriptions=edge_descriptions, fact_edge_prompt=fact_edge_prompt)
    else:
        if max_workers == 1:
            construct_edges_between_entities_and_facts(graph, entity_batch_size=5, use_cot=use_cot)
            construct_edges_between_chunks(graph, use_cot=use_cot)
            construct_edges_between_same_chunk_facts(graph, use_cot=use_cot)
            construct_edges_between_all_facts(graph, threshold, use_cot=use_cot)
        else:
            construct_edges_between_entities_and_facts_parallel(graph, entity_batch_size=5, max_workers=max_workers, use_cot=use_cot)
            construct_edges_between_chunks_parallel(graph, max_workers=max_workers, use_cot=use_cot)
            construct_edges_between_same_chunk_facts_parallel(graph, max_workers=max_workers, use_cot=use_cot)
            construct_edges_between_all_facts_parallel(graph, threshold, max_workers=max_workers, use_cot=use_cot)

def construct_edges_during_merge(graph: Graph, threshold: float = 0.5, max_workers: int = 10, mode: str = "linear", top_k: int = 10, use_cot: bool = True, neutral_description: bool = False,
                                 use_round1: bool = True, use_round2: bool = True, use_round3: bool = True,
                                 clustered_max_facts_per_prompt: int = 20,
                                 retrieval_model: str = "all-MiniLM-L6-v2",
                                 r1_weights: dict = None, r2_weights: dict = None, r3_weights: dict = None,
                                 questions_mode: str = "follow_up_to_answered",
                                 middle_k_skip_percent: float = 0.33,
                                 checkpoint=None, edge_descriptions: bool = True, fact_edge_prompt: str = "default",
                                 fast_cluster_options: dict = None):
    if mode == "clustered_fast":
        # fast_fact_clustering: batched GPU similarity graph, cached embeddings, Leiden-level choice.
        # fast_cluster_options: {"embed_loader", "leiden_levels", "leiden_passes"} (module defaults when absent).
        from nkg.index.construction.fast_fact_clustering import construct_edges_between_facts_clustered_fast
        return construct_edges_between_facts_clustered_fast(graph, top_k=top_k, skip_percent=middle_k_skip_percent, max_workers=max_workers, use_cot=use_cot, neutral_description=neutral_description, edge_descriptions=edge_descriptions, fact_edge_prompt=fact_edge_prompt,
                                                            use_round1=use_round1, use_round2=use_round2, use_round3=use_round3,
                                                            max_cluster_size=clustered_max_facts_per_prompt,
                                                            retrieval_model=retrieval_model,
                                                            r1_weights=r1_weights, r2_weights=r2_weights, r3_weights=r3_weights,
                                                            questions_mode=questions_mode,
                                                            checkpoint=checkpoint, **(fast_cluster_options or {}))
    if mode == "linear":
        if max_workers == 1:
            construct_edges_between_facts_linear(graph, top_k=top_k, skip_percent=middle_k_skip_percent, use_cot=use_cot, retrieval_model=retrieval_model)
        else:
            construct_edges_between_facts_linear_parallel(graph, top_k=top_k, skip_percent=middle_k_skip_percent, max_workers=max_workers, use_cot=use_cot, retrieval_model=retrieval_model)
    elif mode == "clustered":
        # One LLM call per cluster instead of one per fact-pair.
        # max_workers controls parallel LLM calls across clusters within each round.
        return construct_edges_between_facts_clustered(graph, top_k=top_k, skip_percent=middle_k_skip_percent, max_workers=max_workers, use_cot=use_cot, neutral_description=neutral_description, edge_descriptions=edge_descriptions, fact_edge_prompt=fact_edge_prompt,
                                                       use_round1=use_round1, use_round2=use_round2, use_round3=use_round3,
                                                       max_cluster_size=clustered_max_facts_per_prompt,
                                                       retrieval_model=retrieval_model,
                                                       r1_weights=r1_weights, r2_weights=r2_weights, r3_weights=r3_weights,
                                                       questions_mode=questions_mode,
                                                       checkpoint=checkpoint)
    else:
        if max_workers == 1:
            construct_edges_between_chunks(graph, use_cot=use_cot)
            construct_edges_between_all_facts(graph, threshold, use_cot=use_cot)
        else:
            construct_edges_between_chunks_parallel(graph, max_workers=max_workers, use_cot=use_cot)
            construct_edges_between_all_facts_parallel(graph, threshold, max_workers=max_workers, use_cot=use_cot)