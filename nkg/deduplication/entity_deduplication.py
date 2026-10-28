import uuid
import json
import random
import unicodedata
import inflect
import numpy as np
import networkx as nx
import dspy
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from nkg.utils.math_utils import weighted_cosine_topk
from concurrent.futures import ThreadPoolExecutor, as_completed
from graspologic.partition import hierarchical_leiden

from nkg.models.Graph import Graph
from nkg.models.index_objects import EntityFingerprint


# ==========================================
# DSPY SIGNATURES & MODELS
# ==========================================

class MergedEntity(BaseModel):
    duplicate_names: List[str] = Field(
        description="The names of the entities from the cluster that are duplicates of each other."
    )
    alias_name: str = Field(description="The normalized name for the merged entity.")
    alias_type: str = Field(description="The high-level ontological category (e.g., PERSON, ORGANIZATION).")
    alias_role: str = Field(description="A concise 3-8 word micro-role defining its function.")
    alias_anchors: List[str] = Field(
        description="A combined, deduplicated list of relational anchors from the merged entities.")


class ClusterResolution(dspy.Signature):
    """
    Given a cluster of entity records (JSON), identify subsets that refer to the EXACT same real-world entity.

    For each duplicate group:
    - List the entity names that are duplicates in duplicate_names.
    - Produce a single normalized alias representing them.

    Only merge entities that are unambiguously identical. Different entities that are merely related (e.g., a person and their company) must NOT be merged. Return an empty list if no duplicates exist.
    """
    cluster_records: str = dspy.InputField(desc="JSON of entity records with their names, types, roles, and anchors.")
    merged_entities: List[MergedEntity] = dspy.OutputField(desc="Merged entity groups. Empty list if no duplicates.")


# ==========================================
# MODULE-LEVEL HELPERS
# ==========================================

# Default weights for the 3 embedding arrays used in the LLM-dedup similarity graph
SIM_WEIGHTS = {"name": 0.6, "role": 0.3, "type": 0.1}

_inflect_engine = inflect.engine()


def _normalize_for_storage(name: str) -> str:
    """Lowercase + NFKC unicode normalization. This is what gets stored in the entity profile."""
    return unicodedata.normalize("NFKC", name).strip().lower()


def _normalize_for_grouping(name: str) -> str:
    """Lowercase + NFKC + per-token singularization. Used as the exact-match grouping key only."""
    name = _normalize_for_storage(name)
    tokens = []
    for tok in name.split():
        sing = _inflect_engine.singular_noun(tok)
        tokens.append(sing if isinstance(sing, str) and sing else tok)
    return " ".join(tokens)


def _centroid_nearest_idx(entity_ids: list, graph: Graph, retrieval_model) -> int:
    """Return the index into entity_ids whose role embedding is nearest to the group centroid."""
    roles = [graph.entities[eid].role for eid in entity_ids]
    embs = retrieval_model.encode(roles, show_progress_bar=False)
    centroid = embs.mean(axis=0, keepdims=True)
    sims = cosine_similarity(centroid, embs)[0]
    return int(np.argmax(sims))


def _rewire_group(group_ids: list, alias_fp: EntityFingerprint, graph: Graph):
    """
    Create a single new alias node with alias_fp, rewire every edge from every node in
    group_ids to the alias, then delete all old nodes.
    """
    alias_id = str(uuid.uuid4())
    graph.entities[alias_id] = alias_fp
    graph.network.add_node(alias_id, type="entity", data=alias_fp)

    for old_node in group_ids:
        for u, _, edge_data in list(graph.network.in_edges(old_node, data=True)):
            graph.network.add_edge(u, alias_id, **edge_data)
        for _, v, edge_data in list(graph.network.out_edges(old_node, data=True)):
            graph.network.add_edge(alias_id, v, **edge_data)
        graph.network.remove_node(old_node)
        del graph.entities[old_node]


# ==========================================
# DEDUPLICATOR CLASS
# ==========================================

class GraphDeduplicator:
    def __init__(self, retrieval_model: SentenceTransformer, use_cot: bool = True):
        self.retrieval_model = retrieval_model
        self.resolver = (dspy.ChainOfThought if use_cot else dspy.Predict)(ClusterResolution)

    # ------------------------------------------------------------------
    # Phase 1: Exact-name deduplication after normalization
    # ------------------------------------------------------------------

    def _phase1_exact_dedup(self, graph: Graph) -> int:
        """
        Normalize all entity names to lowercase NFKC, group by singularized key,
        merge each group using centroid-nearest role selection, random type and anchors.
        """
        print("Phase 1: Normalizing entity names and grouping exact matches...")

        # Step 1: Update every entity's stored name to the normalized form
        for eid in list(graph.entities.keys()):
            if not graph.network.has_node(eid):
                continue
            fp = graph.entities[eid]
            stored = _normalize_for_storage(fp.name)
            if stored != fp.name:
                new_fp = EntityFingerprint(name=stored, type=fp.type, role=fp.role,
                                           relational_anchors=fp.relational_anchors)
                graph.entities[eid] = new_fp
                graph.network.nodes[eid]["data"] = new_fp

        # Step 2: Group by singularized key (catches plural/singular variants)
        groups: Dict[str, List[str]] = {}
        for eid in graph.entities:
            if not graph.network.has_node(eid):
                continue
            key = _normalize_for_grouping(graph.entities[eid].name)
            groups.setdefault(key, []).append(eid)

        # Step 3: Merge groups of 2+
        merged = 0
        for entity_ids in groups.values():
            entity_ids = [eid for eid in entity_ids if graph.network.has_node(eid)]
            if len(entity_ids) <= 1:
                continue

            winner_idx = _centroid_nearest_idx(entity_ids, graph, self.retrieval_model)
            winner = graph.entities[entity_ids[winner_idx]]
            rand_ent = graph.entities[random.choice(entity_ids)]

            alias_fp = EntityFingerprint(
                name=winner.name,
                type=rand_ent.type,
                role=winner.role,
                relational_anchors=rand_ent.relational_anchors,
            )
            _rewire_group(entity_ids, alias_fp, graph)
            merged += len(entity_ids)

        print(f"Phase 1 complete: merged {merged} entities via exact-name match.")
        return merged

    # ------------------------------------------------------------------
    # Phase 2: SemHash near-duplicate deduplication
    # ------------------------------------------------------------------

    def _phase2_semhash_dedup(self, graph: Graph, threshold: float = 0.95) -> int:
        """
        Run SemHash on the (already-normalized) entity names to catch near-duplicate
        strings that survived exact matching. Uses centroid-nearest role selection,
        random type and anchors.
        """
        print(f"Phase 2: SemHash deduplication (threshold={threshold})...")
        from semhash import SemHash

        entity_ids = [eid for eid in graph.entities if graph.network.has_node(eid)]
        if not entity_ids:
            return 0

        names = [graph.entities[eid].name for eid in entity_ids]

        # name → [eid] lookup (should be 1:1 after phase 1)
        name_to_ids: Dict[str, List[str]] = {}
        for eid, name in zip(entity_ids, names):
            name_to_ids.setdefault(name, []).append(eid)

        # SemHash expects unique strings
        unique_names = list(dict.fromkeys(names))
        semhash_index = SemHash.from_records(records=unique_names)
        result = semhash_index.self_deduplicate(threshold=threshold)

        # Build dup → direct winner mapping (result.filtered = items that were removed as duplicates)
        dup_to_winner: Dict[str, str] = {}
        for dup_record in result.filtered:
            dup_name = dup_record.record
            if not dup_record.duplicates or not dup_record.duplicates[0]:
                continue
            winner_name = dup_record.duplicates[0][0]
            dup_to_winner[dup_name] = winner_name

        # Resolve transitive chains: A→B→C all end up at C
        def resolve_winner(name: str) -> str:
            visited = set()
            while name in dup_to_winner and name not in visited:
                visited.add(name)
                name = dup_to_winner[name]
            return name

        # Collect entity ID groups keyed by their ultimate winner name
        groups_by_winner: Dict[str, List[str]] = {}
        for dup_name in dup_to_winner:
            winner_name = resolve_winner(dup_name)
            if winner_name not in groups_by_winner:
                groups_by_winner[winner_name] = list(name_to_ids.get(winner_name, []))
            for eid in name_to_ids.get(dup_name, []):
                if eid not in groups_by_winner[winner_name]:
                    groups_by_winner[winner_name].append(eid)

        merged = 0
        for entity_ids in groups_by_winner.values():
            entity_ids = [eid for eid in entity_ids if graph.network.has_node(eid)]
            if len(entity_ids) <= 1:
                continue

            winner_idx = _centroid_nearest_idx(entity_ids, graph, self.retrieval_model)
            winner = graph.entities[entity_ids[winner_idx]]
            rand_ent = graph.entities[random.choice(entity_ids)]

            alias_fp = EntityFingerprint(
                name=winner.name,
                type=rand_ent.type,
                role=winner.role,
                relational_anchors=rand_ent.relational_anchors,
            )
            _rewire_group(entity_ids, alias_fp, graph)
            merged += len(entity_ids)

        print(f"Phase 2 complete: merged {merged} entities via SemHash.")
        return merged

    # ------------------------------------------------------------------
    # Phase 3: LLM deduplication (Leiden clustering + LLM confirmation)
    # ------------------------------------------------------------------

    def _build_similarity_graph(self, graph: Graph, top_k: int = 15, threshold: float = 0.75,
                                sim_weights: dict = None) -> nx.Graph:
        """
        Embeds name, role, and type; computes a weighted combined cosine similarity;
        builds a sparse undirected graph for Leiden clustering.
        """
        print("Building similarity matrix for clustering...")

        weights = sim_weights if sim_weights is not None else SIM_WEIGHTS

        entity_ids = list(graph.entities.keys())
        if not entity_ids:
            return nx.Graph()

        names = [graph.entities[eid].name or "unknown" for eid in entity_ids]
        roles = [graph.entities[eid].role or "unknown" for eid in entity_ids]
        types = [graph.entities[eid].type or "unknown" for eid in entity_ids]

        embs_name = self.retrieval_model.encode(names, show_progress_bar=False)
        embs_role = self.retrieval_model.encode(roles, show_progress_bar=False)
        embs_type = self.retrieval_model.encode(types, show_progress_bar=False)

        # Weighted name/role/type cosine, top_k neighbours per entity, computed in row chunks (no N x N matrix:
        # at full-corpus scale that is ~200 GB per matrix). Same edges as the dense version (up to exact ties).
        top_idx, top_scores, _, _ = weighted_cosine_topk(
            [embs_name, embs_role, embs_type], [weights["name"], weights["role"], weights["type"]], top_k)

        sim_graph = nx.Graph()
        sim_graph.add_nodes_from(entity_ids)

        for i in range(len(entity_ids)):
            for j, score in zip(top_idx[i], top_scores[i]):
                if score >= threshold:
                    sim_graph.add_edge(entity_ids[i], entity_ids[int(j)], weight=float(score))

        return sim_graph

    @staticmethod
    def _combine_small_clusters(clusters: List[List[str]], max_cluster_size: int) -> List[List[str]]:
        """
        Greedily combines small clusters together so long as the combined size stays
        within max_cluster_size. This cuts down the number of separate LLM calls
        (and their repeated prompt overhead) needed to resolve the same entities.
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

    def _resolve_cluster(self, cluster_ids: List[str], graph: Graph) -> List[Dict[str, Any]]:
        """
        Worker: sends one cluster's entities to the LLM and parses confirmed merge groups.
        """
        name_to_ids: Dict[str, List[str]] = {}
        for eid in cluster_ids:
            name_to_ids.setdefault(graph.entities[eid].name, []).append(eid)

        records = []
        for eid in cluster_ids:
            ent = graph.entities[eid]
            records.append({
                "name": ent.name,
                "type": ent.type,
                "role": ent.role,
                "anchors": list(ent.relational_anchors)
            })

        try:
            result = self.resolver(cluster_records=json.dumps(records, indent=2))
            resolved_groups = []
            for merged in result.merged_entities:
                valid_dups = []
                for name in merged.duplicate_names:
                    valid_dups.extend(name_to_ids.get(name, []))
                valid_dups = list(dict.fromkeys(valid_dups))
                if len(valid_dups) > 1:
                    alias_fp = EntityFingerprint(
                        name=merged.alias_name,
                        type=merged.alias_type,
                        role=merged.alias_role,
                        relational_anchors=merged.alias_anchors,
                    )
                    resolved_groups.append({"alias": alias_fp, "duplicates": valid_dups})
            return resolved_groups
        except Exception as e:
            print(f"Error resolving cluster: {e}")
            return []

    def _phase3_llm_dedup(self, graph: Graph, iterations: int, max_workers: int,
                          max_cluster_size: int, sim_weights: dict,
                          similarity_threshold: float = 0.75) -> int:
        """
        Leiden clustering on embedding similarity, then parallel LLM confirmation,
        then surgical rewiring. Runs for `iterations` passes.
        """
        merged_total = 0
        for iteration in range(iterations):
            print(f"\n--- LLM Dedup Iteration {iteration + 1} ---")

            sim_graph = self._build_similarity_graph(graph, threshold=similarity_threshold, sim_weights=sim_weights)
            if len(sim_graph.edges) == 0:
                print("Similarity graph has no edges above threshold. Skipping.")
                break

            clusters_raw = hierarchical_leiden(sim_graph, max_cluster_size=max_cluster_size, random_seed=42)
            clusters: Dict[str, List[str]] = {}
            for node in clusters_raw:
                clusters.setdefault(node.cluster, []).append(node.node)

            candidate_clusters = [c for c in clusters.values() if len(c) > 1]
            candidate_clusters = self._combine_small_clusters(candidate_clusters, max_cluster_size)
            print(f"Found {len(candidate_clusters)} clusters for LLM evaluation "
                  f"(after combining small clusters to reduce LLM calls).")

            print(f"Executing LLM deduplication across {max_workers} threads...")
            resolution_results = []
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(self._resolve_cluster, cluster, graph): cluster
                    for cluster in candidate_clusters
                }
                for future in as_completed(futures):
                    resolution_results.extend(future.result())

            print("Surgically rewiring graph edges...")
            merged_in_iter = 0
            for res in resolution_results:
                alias_fp = res["alias"]
                duplicates = [d for d in res["duplicates"] if graph.network.has_node(d)]
                if len(duplicates) <= 1:
                    continue
                _rewire_group(duplicates, alias_fp, graph)
                merged_in_iter += len(duplicates)

            print(f"Removed {merged_in_iter} duplicate nodes in iteration {iteration + 1}.")
            merged_total += merged_in_iter

        return merged_total

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def deduplicate(self, graph: Graph,
                    # Phase 1 & 2 controls
                    run_exact_dedup: bool = True,
                    run_semhash_dedup: bool = True,
                    semhash_threshold: float = 0.95,
                    # Phase 3 controls (LLM)
                    run_llm_dedup: bool = True,
                    iterations: int = 1,
                    max_workers: int = 15,
                    max_cluster_size: int = 64,
                    sim_weights: dict = None,
                    similarity_threshold: float = 0.75) -> Graph:
        """
        Three-phase entity deduplication:

        Phase 1 — Exact-name: normalize all entity names to lowercase NFKC, group by
                   singularized key, merge each group with centroid-nearest role selection.

        Phase 2 — SemHash: run SemHash on the normalized names to catch near-duplicates
                   that survived phase 1 (e.g., abbreviations, minor typos). Same merge
                   strategy as phase 1.

        Phase 3 — LLM: Leiden clustering on (name, role, type) embedding similarity
                   (edges drawn between entities whose combined similarity >= similarity_threshold),
                   parallel LLM confirmation of each cluster, surgical rewiring.
        """
        if run_exact_dedup:
            self._phase1_exact_dedup(graph)

        if run_semhash_dedup:
            self._phase2_semhash_dedup(graph, threshold=semhash_threshold)

        if run_llm_dedup:
            self._phase3_llm_dedup(graph, iterations=iterations, max_workers=max_workers,
                                   max_cluster_size=max_cluster_size, sim_weights=sim_weights,
                                   similarity_threshold=similarity_threshold)

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
    deduplicator = GraphDeduplicator(retrieval_model=retrieval_model)
    print(f"Before deduplication: {len(graph.entities)} entities.")
    graph = deduplicator.deduplicate(graph, max_workers=100)
    print(f"After deduplication: {len(graph.entities)} entities.")
    graph.export_graph("./kg_outputs/td_deduplicated.graphml")


if __name__ == "__main__":
    main()
