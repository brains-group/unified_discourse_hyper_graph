"""
Fact-level deduplication and contradiction resolution for the nkg knowledge graph.

Companion to entity_deduplication.py's GraphDeduplicator, but instead of merging
duplicate *entities* it clusters *facts* that a similarity search flags as plausibly
duplicate, contradictory, or overlapping, and asks an LLM -- grounded in the ORIGINAL
CHUNK TEXT the facts were extracted from, not chunk summaries -- to keep, remove,
rewrite, or merge them.

Credit: the conflict taxonomy used in FactClusterResolution below (duplicate / mutual
exclusivity / granularity mismatch, and the "specify or combine, don't just delete"
handling of granularity conflicts) is adapted from MemGraphRAG's conflict-detection and
conflict-resolution pipeline (XMUDeepLIT/MemGraphRAG, KDD'26,
https://github.com/XMUDeepLIT/MemGraphRAG -- see code/src/MemGraphRAG.py
`detect_memory_conflicts`/`resolve_memory_conflicts` and code/src/prompts/prompt.py
`PROMPT['conflict_detection']`/`PROMPT['conflict_resolution']`).

MemGraphRAG operates on OpenIE (head, relation, tail) triples and resolves a conflict
by rewriting a triple's *relation* text (e.g. appending a time or granularity
qualifier onto the predicate), and it never merges two triples into one. nkg facts are
full natural-language sentences carrying their own entity list and multi-chunk
provenance, so this module is a from-scratch dspy re-implementation, not a port: it
adds a genuine multi-fact *merge* action (impossible for a bare triple) that unions the
source facts' chunk/entity graph connections and metadata onto one new fact node, and
resolves via a single dspy.Signature per cluster instead of MemGraphRAG's separate
detection/resolution calls.
"""

import uuid
from typing import List, Dict, Any, Optional

import dspy
import numpy as np
import networkx as nx
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from nkg.utils.math_utils import weighted_cosine_topk
from concurrent.futures import ThreadPoolExecutor, as_completed
from graspologic.partition import hierarchical_leiden

from nkg.models.Graph import Graph
from nkg.models.index_objects import Fact


# ==========================================
# DSPY SIGNATURES & MODELS
# ==========================================

class FactResolution(BaseModel):
    fact_ids: List[int] = Field(
        description="The numbered IDs (from the numbered_facts list) this action applies to. "
                    "Exactly one ID for 'remove' or 'modify'; two or more for 'merge'."
    )
    action: str = Field(description="One of 'remove', 'modify', or 'merge'.")
    resolved_sentence: Optional[str] = Field(
        default=None,
        description="Required for 'modify' and 'merge': the new, precise fact sentence. "
                    "Omit (null) for 'remove'."
    )
    conflict_type: str = Field(
        description="What kind of issue this is: 'duplicate', 'mutual_exclusivity', "
                    "'granularity', 'false_claim', or 'other'."
    )
    reason: str = Field(description="Brief explanation, citing the source text where relevant.")


class FactClusterResolution(dspy.Signature):
    """
    You are curating a knowledge graph. You are given the ORIGINAL SOURCE TEXT the
    candidate facts were extracted from, and a numbered list of facts that a similarity
    search flagged as plausibly duplicate, contradictory, or overlapping. Decide what,
    if anything, to do with each one.

    Facts can conflict in different ways -- identify which applies before acting:
    - duplicate: two facts state the same thing in different words. Merge them into one
      clean sentence (or remove the weaker/less precise phrasing).
    - mutual_exclusivity: the facts make genuinely incompatible claims that cannot both
      be true as stated (e.g. two different birth years for the same person). Check the
      source text and remove whichever is not actually supported by it. If a closer look
      shows the facts are compatible once qualified (e.g. two different presidents each
      correctly tied to their own distinct term number, or two events at different
      times), that is NOT mutual exclusivity -- treat it as granularity instead and do
      not delete anything.
    - granularity: both facts are true but state the claim at different levels of
      specificity or scope (e.g. "born in Los Angeles" vs. "born in the USA"; "the 45th
      president" vs. "the 46th president" naming two different people who each really
      held that distinct term). These are compatible, not contradictions. Resolve by
      either (a) modifying each fact so it states its specific scope explicitly, so nothing
      reads as contradicting the other, or (b) merging them into one fact that states the
      full, unambiguous picture (e.g. "born in Los Angeles, California, USA"). Prefer
      merging when the two facts are about the exact same underlying claim; prefer
      modifying in place when they are otherwise distinct facts that just need
      disambiguating qualifiers.
    - false_claim: the source text does not support the fact at all (extraction error).
      Remove it.

    Only emit an action for facts you are actually changing -- never emit an action for
    a fact that should stay exactly as it is; every fact not mentioned in your output is
    kept unchanged automatically. A merge may combine two OR MORE facts into a single
    richer sentence when they describe the same underlying claim at complementary levels
    of detail or in redundant phrasing. Do not invent facts the source text does not
    support. When uncertain whether something is a real conflict, prefer to leave the
    facts alone rather than guessing.
    """
    source_text: str = dspy.InputField(
        desc="The original chunk text (concatenated across chunks, each labeled, if the "
             "cluster spans more than one) that the candidate facts were extracted from."
    )
    numbered_facts: str = dspy.InputField(
        desc="Numbered candidate fact sentences to evaluate, each tagged with which "
             "chunk(s) in source_text it came from."
    )
    resolutions: List[FactResolution] = dspy.OutputField(
        desc="Actions to take. Empty list if every fact is fine as-is."
    )


# ==========================================
# MODULE-LEVEL HELPERS
# ==========================================

# Default weights for the 3 similarity dimensions used to build the fact-dedup similarity
# graph. Sentence similarity dominates per spec; entity/topic split the remaining weight
# evenly by default -- override via sim_weights for a different split.
FACT_SIM_WEIGHTS = {"sentence": 0.7, "entity": 0.15, "topic": 0.15}


def _fact_entity_repr(fact: Fact) -> str:
    """Space-joined entity names -- identity signal for whether two facts are about the same
    real-world people/places/things, which is what matters for contradiction detection
    (as opposed to the functional 'role' embedding used for fact_fact edge construction)."""
    names = [e.name for e in fact.entities]
    return " ".join(names) if names else "unknown"


def _fact_topic_repr(fact: Fact) -> str:
    topics = list(fact.macro_topics) + list(fact.chunk_topics)
    return " ".join(topics) if topics else "unknown"


def _get_parent_chunk_ids(graph: Graph, fact_id: str) -> List[str]:
    """Every chunk this fact is directly linked from via a chunk_fact edge. A fact produced
    by an earlier merge iteration can have more than one parent chunk."""
    chunk_ids = []
    for source_node, _, edge_data in graph.network.in_edges(fact_id, data=True):
        if edge_data.get("edge_type") == "chunk_fact" and source_node in graph.chunks:
            chunk_ids.append(source_node)
    return chunk_ids


def _dedup_preserve_order(items: list) -> list:
    return list(dict.fromkeys(items))


def _merge_optional_lists(lists: List[Optional[list]]) -> Optional[list]:
    """Union several Optional[List[str]] fields. None only if every input was None."""
    if all(l is None for l in lists):
        return None
    merged = []
    for l in lists:
        merged.extend(l or [])
    return _dedup_preserve_order(merged)


def _apply_remove(fact_id: str, graph: Graph) -> None:
    if graph.network.has_node(fact_id):
        graph.network.remove_node(fact_id)
    graph.facts.pop(fact_id, None)


def _apply_modify(fact_id: str, new_sentence: str, graph: Graph) -> None:
    old = graph.facts[fact_id]
    new_fact = Fact(
        sentence=new_sentence,
        macro_topics=old.macro_topics,
        chunk_topics=old.chunk_topics,
        answered_questions=old.answered_questions,
        follow_up_questions=old.follow_up_questions,
        entities=old.entities,
        entity_relation_labels=old.entity_relation_labels,
    )
    graph.facts[fact_id] = new_fact
    graph.network.nodes[fact_id]["data"] = new_fact


def _rewire_fact_group(group_ids: List[str], merged_fact: Fact, graph: Graph) -> str:
    """
    Create a single new fact node holding merged_fact, redirect every edge touching any
    node in group_ids onto the new node (chunk_fact edges from every parent chunk of every
    merged fact, entity_fact/fact_entity edges to every entity any of them referenced,
    fact_fact edges to any other neighboring fact), then delete the old fact nodes.

    Edges where BOTH endpoints fall inside group_ids (e.g. an existing fact_fact edge
    between two facts that are themselves being merged) are dropped rather than rewired,
    since redirecting both ends onto the same new node would create a meaningless self-loop.

    Mirrors entity_deduplication.py's _rewire_group surgical-rewiring approach.
    """
    new_id = str(uuid.uuid4())
    group_set = set(group_ids)
    graph.facts[new_id] = merged_fact
    graph.network.add_node(new_id, data=merged_fact, type="fact")

    for old_node in group_ids:
        for u, _, edge_data in list(graph.network.in_edges(old_node, data=True)):
            if u in group_set:
                continue
            graph.network.add_edge(u, new_id, **edge_data)
        for _, v, edge_data in list(graph.network.out_edges(old_node, data=True)):
            if v in group_set:
                continue
            graph.network.add_edge(new_id, v, **edge_data)
        graph.network.remove_node(old_node)
        del graph.facts[old_node]

    return new_id


def _apply_merge(fact_ids: List[str], new_sentence: str, graph: Graph) -> Optional[str]:
    """Concatenates every metadata field across the source facts (deduplicated, order
    preserved) onto one new Fact, then rewires all of their graph connections onto it."""
    facts = [graph.facts[fid] for fid in fact_ids if fid in graph.facts]
    if len(facts) < 2:
        return None

    merged_fact = Fact(
        sentence=new_sentence,
        macro_topics=_dedup_preserve_order([t for f in facts for t in f.macro_topics]),
        chunk_topics=_dedup_preserve_order([t for f in facts for t in f.chunk_topics]),
        answered_questions=_merge_optional_lists([f.answered_questions for f in facts]),
        follow_up_questions=_merge_optional_lists([f.follow_up_questions for f in facts]),
        entities=_dedup_preserve_order([e for f in facts for e in f.entities]),
        entity_relation_labels=tuple(
            _dedup_preserve_order([pair for f in facts for pair in f.entity_relation_labels])
        ),
    )
    valid_ids = [fid for fid in fact_ids if fid in graph.facts]
    return _rewire_fact_group(valid_ids, merged_fact, graph)


# ==========================================
# DEDUPLICATOR CLASS
# ==========================================

class GraphFactDeduplicator:
    def __init__(self, retrieval_model: SentenceTransformer, use_cot: bool = True):
        self.retrieval_model = retrieval_model
        self.resolver = (dspy.ChainOfThought if use_cot else dspy.Predict)(FactClusterResolution)

    # ------------------------------------------------------------------
    # Similarity graph construction
    # ------------------------------------------------------------------

    def _build_similarity_graph(self, graph: Graph, top_k: int = 12, threshold: float = 0.8,
                                sim_weights: dict = None) -> nx.Graph:
        """
        Embeds each fact's sentence, entity names, and topics; computes a weighted
        combined cosine similarity; min-max normalizes it to [0, 1] across distinct fact
        pairs (so the most similar pair of facts in this corpus scores 1.0 and the least
        similar pair scores 0.0); then, per fact, draws an edge to each of its top_k
        nearest neighbors that clears `threshold`. A fact with no neighbor above
        threshold gets no edges at all (top_k stops mattering for it) and ends up an
        isolated node -- Leiden puts it in its own singleton cluster, which the caller
        filters out.
        """
        weights = sim_weights if sim_weights is not None else FACT_SIM_WEIGHTS

        fact_ids = list(graph.facts.keys())
        if len(fact_ids) < 2:
            return nx.Graph()

        sentences = [graph.facts[fid].sentence or "unknown" for fid in fact_ids]
        entity_reprs = [_fact_entity_repr(graph.facts[fid]) for fid in fact_ids]
        topic_reprs = [_fact_topic_repr(graph.facts[fid]) for fid in fact_ids]

        embs_sentence = self.retrieval_model.encode(sentences, show_progress_bar=False)
        embs_entity = self.retrieval_model.encode(entity_reprs, show_progress_bar=False)
        embs_topic = self.retrieval_model.encode(topic_reprs, show_progress_bar=False)

        # Weighted sentence/entity/topic cosine, top_k neighbours per fact, computed in row chunks (no N x N
        # matrix: at full-corpus scale that is ~65 GB per matrix). The min-max normalisation over all distinct
        # pairs uses the global off-diagonal min/max from the same pass; it preserves each row's ranking, so the
        # top_k neighbours are the same as the dense version's (up to exact ties).
        n = len(fact_ids)
        top_idx, top_scores, lo, hi = weighted_cosine_topk(
            [embs_sentence, embs_entity, embs_topic],
            [weights["sentence"], weights["entity"], weights["topic"]], top_k)
        spread = hi - lo
        # Every distinct pair is equally similar (most often: only 2 facts total, so there's
        # only one distinct pairwise value; can also happen late in the last dedup iteration
        # once few candidate facts remain). There's no relative signal to normalize against,
        # so treat every pair as maximally similar rather than defaulting to 0 -- silently
        # zeroing everything out would disable edge creation exactly when the graph is small.
        normalized = np.ones_like(top_scores) if spread <= 1e-12 else (top_scores - lo) / spread

        sim_graph = nx.Graph()
        sim_graph.add_nodes_from(fact_ids)

        for i in range(n):
            for j, score in zip(top_idx[i], normalized[i]):
                if score >= threshold:
                    sim_graph.add_edge(fact_ids[i], fact_ids[int(j)], weight=float(score))

        return sim_graph

    @staticmethod
    def _combine_small_clusters(clusters: List[List[str]], max_cluster_size: int) -> List[List[str]]:
        """
        Greedily combines small clusters together so long as the combined size stays
        within max_cluster_size. Reduces the number of separate LLM calls needed.
        Same approach as entity_deduplication.py's helper of the same name.
        """
        clusters_sorted = sorted(clusters, key=len)
        combined: List[List[str]] = []
        i = 0
        while i < len(clusters_sorted):
            bucket = list(clusters_sorted[i])
            j = i + 1
            while j < len(clusters_sorted) and len(bucket) + len(clusters_sorted[j]) <= max_cluster_size:
                bucket.extend(clusters_sorted[j])
                j += 1
            combined.append(bucket)
            i = j
        return combined

    # ------------------------------------------------------------------
    # LLM cluster resolution
    # ------------------------------------------------------------------

    def _resolve_cluster(self, cluster_fact_ids: List[str], graph: Graph) -> List[Dict[str, Any]]:
        """
        Worker: grounds one cluster's facts in their original parent chunk text, sends
        them to the LLM, and returns validated {"action", "fact_ids": [uuid,...],
        "resolved_sentence"} dicts. Read-only against `graph` -- safe to call from a
        thread pool; the caller applies mutations sequentially afterward.
        """
        int_to_uuid: Dict[int, str] = {}
        fact_chunk_ids: Dict[int, List[str]] = {}
        chunk_text_by_id: Dict[str, str] = {}

        for idx, fact_id in enumerate(cluster_fact_ids, start=1):
            int_to_uuid[idx] = fact_id
            parent_chunk_ids = _get_parent_chunk_ids(graph, fact_id)
            fact_chunk_ids[idx] = parent_chunk_ids
            for cid in parent_chunk_ids:
                chunk_text_by_id.setdefault(cid, graph.chunks[cid].text)

        ordered_chunk_ids = sorted(chunk_text_by_id.keys())
        chunk_num = {cid: i + 1 for i, cid in enumerate(ordered_chunk_ids)}

        if ordered_chunk_ids:
            source_text = "\n\n".join(
                f"=== Chunk {chunk_num[cid]} ===\n{chunk_text_by_id[cid]}" for cid in ordered_chunk_ids
            )
        else:
            source_text = "(no source chunk text found)"

        fact_lines = []
        for idx, fact_id in enumerate(cluster_fact_ids, start=1):
            chunk_tags = ",".join(str(chunk_num[cid]) for cid in fact_chunk_ids[idx]) or "?"
            fact_lines.append(f"{idx}: [Chunk {chunk_tags}] {graph.facts[fact_id].sentence}")
        numbered_facts = "\n".join(fact_lines)

        try:
            result = self.resolver(source_text=source_text, numbered_facts=numbered_facts)
        except Exception as e:
            print(f"Error resolving fact cluster: {e}")
            return []

        resolutions = []
        seen_ids = set()  # a fact can only be touched by one action per cluster
        for res in result.resolutions:
            action = (res.action or "").strip().lower()
            valid_ids = [int_to_uuid[i] for i in res.fact_ids if i in int_to_uuid and int_to_uuid[i] not in seen_ids]

            if action == "remove":
                for fid in valid_ids:
                    resolutions.append({"action": "remove", "fact_ids": [fid], "resolved_sentence": None,
                                        "conflict_type": res.conflict_type, "reason": res.reason})
                    seen_ids.add(fid)
            elif action == "modify" and valid_ids and res.resolved_sentence:
                fid = valid_ids[0]
                resolutions.append({"action": "modify", "fact_ids": [fid], "resolved_sentence": res.resolved_sentence,
                                    "conflict_type": res.conflict_type, "reason": res.reason})
                seen_ids.add(fid)
            elif action == "merge" and len(valid_ids) >= 2 and res.resolved_sentence:
                resolutions.append({"action": "merge", "fact_ids": valid_ids, "resolved_sentence": res.resolved_sentence,
                                    "conflict_type": res.conflict_type, "reason": res.reason})
                seen_ids.update(valid_ids)

        return resolutions

    # ------------------------------------------------------------------
    # One deduplicate-then-recluster pass
    # ------------------------------------------------------------------

    def _run_iteration(self, graph: Graph, top_k: int, threshold: float, sim_weights: dict,
                       cluster_size: int, max_workers: int) -> dict:
        sim_graph = self._build_similarity_graph(graph, top_k=top_k, threshold=threshold, sim_weights=sim_weights)
        if sim_graph.number_of_edges() == 0:
            return {"removed": 0, "modified": 0, "merged": 0, "had_clusters": False,
                    "clusters_evaluated": 0, "facts_considered": 0, "actions": []}

        clusters_raw = hierarchical_leiden(sim_graph, max_cluster_size=cluster_size, random_seed=42)
        clusters: Dict[Any, List[str]] = {}
        for node in clusters_raw:
            clusters.setdefault(node.cluster, []).append(node.node)

        candidate_clusters = [c for c in clusters.values() if len(c) > 1]
        if not candidate_clusters:
            return {"removed": 0, "modified": 0, "merged": 0, "had_clusters": False,
                    "clusters_evaluated": 0, "facts_considered": 0, "actions": []}

        candidate_clusters = self._combine_small_clusters(candidate_clusters, cluster_size)
        facts_considered = sum(len(c) for c in candidate_clusters)
        print(f"Fact dedup: {len(candidate_clusters)} cluster(s) / {facts_considered} fact(s) to evaluate "
              f"(cluster_size<={cluster_size}).")

        all_resolutions: List[Dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self._resolve_cluster, cluster, graph): cluster for cluster in candidate_clusters}
            for future in as_completed(futures):
                all_resolutions.extend(future.result())

        removed = modified = merged = 0
        touched: set = set()
        actions_log: List[Dict[str, Any]] = []
        for res in all_resolutions:
            fact_ids = [fid for fid in res["fact_ids"] if graph.network.has_node(fid) and fid not in touched]
            if not fact_ids:
                continue
            # Grab the human-readable sentence(s) BEFORE mutating/removing the node(s).
            before_sentences = [graph.facts[fid].sentence for fid in fact_ids if fid in graph.facts]
            log_entry = {
                "action": res["action"], "conflict_type": res.get("conflict_type", "other"),
                "reason": res.get("reason", ""), "before_sentences": before_sentences,
                "resolved_sentence": res.get("resolved_sentence"),
            }
            if res["action"] == "remove":
                _apply_remove(fact_ids[0], graph)
                removed += 1
                touched.add(fact_ids[0])
                actions_log.append(log_entry)
            elif res["action"] == "modify":
                _apply_modify(fact_ids[0], res["resolved_sentence"], graph)
                modified += 1
                touched.add(fact_ids[0])
                actions_log.append(log_entry)
            elif res["action"] == "merge" and len(fact_ids) >= 2:
                _apply_merge(fact_ids, res["resolved_sentence"], graph)
                merged += 1
                touched.update(fact_ids)
                actions_log.append(log_entry)

        return {"removed": removed, "modified": modified, "merged": merged, "had_clusters": True,
                "clusters_evaluated": len(candidate_clusters), "facts_considered": facts_considered,
                "actions": actions_log}

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def deduplicate(self, graph: Graph, iterations: int = 1, top_k: int = 12, threshold: float = 0.8,
                    cluster_size: int = 8, max_workers: int = 15, sim_weights: dict = None,
                    iteration_stats: list = None) -> Graph:
        """
        Repeatedly builds a fact-similarity graph (top_k neighbors above `threshold`,
        weighted per FACT_SIM_WEIGHTS), Leiden-clusters it into groups of at most
        `cluster_size` facts, and asks an LLM -- shown each cluster's original source
        chunk text -- to remove/modify/merge facts that are duplicate, contradictory, or
        differently-scoped restatements of each other.

        Stops when either `iterations` passes have run, or a pass produces no cluster of
        size > 1 (nothing left worth reviewing at the current threshold/top_k).

        iteration_stats -- if a list is passed, each iteration's stats dict (removed/
        modified/merged/had_clusters/clusters_evaluated/facts_considered/actions, where
        "actions" is a per-resolution log with conflict_type/reason/before_sentences/
        resolved_sentence) is appended to it in order, so callers can report exact
        counts and audit *why* each change was made without parsing stdout.
        """
        for iteration in range(iterations):
            print(f"\n--- Fact Dedup Iteration {iteration + 1}/{iterations} ---")
            stats = self._run_iteration(graph, top_k=top_k, threshold=threshold, sim_weights=sim_weights,
                                        cluster_size=cluster_size, max_workers=max_workers)
            if iteration_stats is not None:
                iteration_stats.append(stats)
            print(f"Iteration {iteration + 1}: removed={stats['removed']} modified={stats['modified']} "
                  f"merged={stats['merged']}")
            if not stats["had_clusters"]:
                print("No fact clusters above the similarity threshold -- stopping fact deduplication early.")
                break

        if hasattr(graph, 'label_edges'):
            graph.label_edges()

        return graph


def main():
    from nkg.utils.config import configure_dspy
    configure_dspy(max_tokens=35000)

    import warnings
    warnings.filterwarnings("ignore")

    graph = Graph()
    graph.load_graph("./kg_outputs/td.graphml")
    graph.label_edges()
    retrieval_model = SentenceTransformer("all-MiniLM-L6-v2")
    deduplicator = GraphFactDeduplicator(retrieval_model=retrieval_model)
    print(f"Before fact deduplication: {len(graph.facts)} facts.")
    graph = deduplicator.deduplicate(graph, max_workers=15)
    print(f"After fact deduplication: {len(graph.facts)} facts.")
    graph.export_graph("./kg_outputs/td_fact_deduplicated.graphml")


if __name__ == "__main__":
    main()
