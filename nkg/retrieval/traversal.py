import numpy as np
from copy import deepcopy
from typing import List, Dict, Tuple
from sentence_transformers import CrossEncoder

from nkg.models.Graph import Graph
from nkg.utils.math_utils import *
from nkg.retrieval.scoring import *

class Path:
    """Tracks the state of a traversal branch to prevent loops and monitor depth."""

    def __init__(self, start_node_id: str, is_fact: bool):
        self.current_node = start_node_id
        self.visited_nodes = {start_node_id}
        self.fact_depth = 1 if is_fact else 0
        self.full_depth = 1

        # History stores the path taken.
        # Format: list of tuples (source_id, edge_type, edge_data, target_id)
        self.history = []

    # def branch(self, target_node_id: str, is_fact: bool, edge_type: str, edge_data: dict) -> 'Path':
    #     """Creates a new Path object representing a step forward."""
    #     new_path = deepcopy(self)
    #     new_path.visited_nodes.add(target_node_id)
    #     new_path.current_node = target_node_id
    #     new_path.history.append((self.current_node, edge_type, edge_data, target_node_id))
    #
    #     if is_fact:
    #         new_path.fact_depth += 1
    #
    #     return new_path

    def branch(self, target_node_id: str, is_fact: bool, edge_type: str, edge_data: dict) -> 'Path':
        """Creates a new Path object representing a step forward without slow deepcopy."""
        # 1. Instantiate a new base path
        new_path = Path(start_node_id=target_node_id, is_fact=is_fact)

        # 2. Manually copy over the standard Python sets and lists (Lightweight & Fast)
        new_path.visited_nodes = set(self.visited_nodes)
        new_path.visited_nodes.add(target_node_id)

        new_path.fact_depth = self.fact_depth + (1 if is_fact else 0)
        new_path.full_depth = self.full_depth + 1

        new_path.history = list(self.history)
        new_path.history.append((self.current_node, edge_type, edge_data, target_node_id))

        return new_path


# ==========================================
# STRING ASSEMBLY HELPERS
# ==========================================

def _get_node_representation(graph: Graph, node_id: str) -> str:
    """Returns the formatted string representation for a Fact or Entity."""
    if node_id in graph.facts:
        return f'"{graph.facts[node_id].sentence}"'
    elif node_id in graph.entities:
        ent = graph.entities[node_id]
        return f"[{ent.name} ({ent.role})]"
    return "[Unknown Node]"


def assemble_path_string(graph: Graph, path: Path) -> str:
    """
    Translates a Path object's history into a highly readable narrative string
    for the Cross-Encoder to evaluate.
    """
    if not path.history:
        return _get_node_representation(graph, path.current_node)

    # Start with the representation of the very first node
    first_node_id = path.history[0][0]
    assembled_str = _get_node_representation(graph, first_node_id)

    for _, edge_type, edge_data, target_id in path.history:
        target_rep = _get_node_representation(graph, target_id)

        if edge_type == "fact_fact":
            desc = edge_data.get("description", "is related to")
            assembled_str += f" -> {desc} -> {target_rep}"

        elif edge_type == "fact_entity":
            label = edge_data.get("label", "has property")
            assembled_str += f" -> (has {label}) -> {target_rep}"

        elif edge_type == "entity_fact":
            label = edge_data.get("label", "participates in")
            assembled_str += f" -> (participates in event: {label}) -> {target_rep}"

    return assembled_str


# ==========================================
# TRAVERSAL & EXPANSION
# ==========================================

def expand_paths(
        engine,  # The Retriever engine (holds graph, embeddings, and planner outputs)
        seeds: List[str],
        plan: object,
        max_depth: int = 3,
        total_depth: int = 3,
        beam_width: int = 3,
        string_output=False,
        mode="all"
) -> List[Path] | List[str]:
    """
    Executes a Bounded Beam Search with Local MMR to explore the graph.
    """
    graph = engine.graph
    active_paths = []
    completed_paths = []
    valid_edge_types = []

    # use the mode to determine what type of edge traversals you will except
    if mode == "all":
        valid_edge_types = ["fact_fact", "entity_fact", "fact_entity"]
    elif mode == "hypergraph":
        valid_edge_types = ["entity_fact", "fact_entity"]
    elif mode == "discourse":
        valid_edge_types = ["fact_fact"]


    # Initialize Active Paths
    for seed in seeds:
        is_fact = seed in graph.facts
        active_paths.append(Path(start_node_id=seed, is_fact=is_fact))

    # Pre-fetch the query plan embeddings for scoring edges
    plan_labels_embs = engine.retrieval_model.encode(plan.target_edge_labels, show_progress_bar=False) if plan.target_edge_labels else np.array(
        [])
    plan_semantics_embs = engine.retrieval_model.encode(
        plan.target_edge_semantics,show_progress_bar=False) if plan.target_edge_semantics else np.array([])
    plan_broad_embs = engine.retrieval_model.encode(plan.broad_anchors,show_progress_bar=False) if plan.broad_anchors else np.array([])

    while active_paths:
        next_active = []

        for path in active_paths:
            current_id = path.current_node

            # Look up outgoing edges
            outgoing_edges = list(graph.network.out_edges(current_id, data=True))

            valid_candidates = []
            candidate_scores = []
            candidate_diversity_embs = []  # Used for Local MMR

            for u, v, edge_data in outgoing_edges:
                edge_type = edge_data.get("edge_type")

                # Filter 1: Loop prevention
                if v in path.visited_nodes:
                    continue

                # Filter 2: Ignore irrelevant edge types (like chunk_fact)
                if edge_type not in valid_edge_types:
                    continue

                edge_tuple = (u, v)
                score = 0.0
                diversity_emb = None

                # Score based on edge type using the engine's pre-computed embeddings
                if edge_type == "fact_fact":
                    score = score_fact_fact_edge(
                        plan_labels_embs, plan_semantics_embs,
                        engine.ff_label_embs[edge_tuple],
                        engine.ff_desc_embs[edge_tuple],
                        edge_data.get("score", 0.0)
                    )
                    diversity_emb = engine.ff_desc_embs[edge_tuple][0]  # Use desc for diversity

                elif edge_type == "entity_fact":
                    score = score_entity_fact_edge(
                        plan_broad_embs, plan_semantics_embs,
                        engine.ef_label_embs[edge_tuple]
                    )
                    diversity_emb = engine.ef_label_embs[edge_tuple][0]

                elif edge_type == "fact_entity":
                    score = score_fact_entity_edge(
                        plan_broad_embs,
                        engine.fe_label_embs[edge_tuple]
                    )
                    diversity_emb = engine.fe_label_embs[edge_tuple][0]

                valid_candidates.append((v, edge_type, edge_data))
                candidate_scores.append(score)
                candidate_diversity_embs.append(diversity_emb)

            # Handle Dead Ends
            if not valid_candidates:
                completed_paths.append(path)
                continue

            # Local MMR Execution (Select top 'beam_width' edges)
            selected_edge_indices = compute_mmr(
                candidate_scores=candidate_scores,
                candidate_embeddings=np.array(candidate_diversity_embs),
                top_k=beam_width,
                lambda_param=0.5  # Balance local relevance and local diversity
            )

            # Branch the path for the selected edges
            for idx in selected_edge_indices:
                target_node, e_type, e_data = valid_candidates[idx]
                is_fact = target_node in graph.facts

                new_path = path.branch(target_node, is_fact, e_type, e_data)

                # Check Depth
                if new_path.fact_depth >= max_depth or new_path.full_depth >= total_depth:
                    completed_paths.append(new_path)
                else:
                    next_active.append(new_path)

        active_paths = next_active

    if string_output:
        assembled_strings = [assemble_path_string(engine.graph, p) for p in completed_paths]
        return assembled_strings

    return completed_paths

from collections import defaultdict


def expand_paths_blind(engine, seeds, max_depth=3, mode="all", total_depth=None):
    """
    Blind expansion: starting from each seed, follow ALL valid edges outward up to
    max_depth (fact hops) without any scoring or beam pruning during traversal.
    The full set of completed paths is then passed to the global cross-encoder +
    MMR ranking stage.

    mode controls which edge types are followed:
      "all"        → fact_fact, entity_fact, fact_entity
      "discourse"  → fact_fact only
      "hypergraph" → entity_fact, fact_entity only
    """
    graph = engine.graph
    active_paths = [Path(seed, seed in graph.facts) for seed in seeds]
    completed_paths = []

    # total_depth caps full hop count so entity-heavy graphs don't explode
    if total_depth is None:
        total_depth = max_depth * 2

    valid_edge_types = {
        "all": {"fact_fact", "entity_fact", "fact_entity"},
        "hypergraph": {"entity_fact", "fact_entity"},
        "discourse": {"fact_fact"},
    }[mode]

    # Cache out-edges per node so multiple paths sharing a frontier node
    # only pay the NetworkX adjacency lookup once per BFS level.
    while active_paths:
        next_active = []
        edge_cache = {}

        for path in active_paths:
            current_id = path.current_node
            if current_id not in edge_cache:
                edge_cache[current_id] = list(graph.network.out_edges(current_id, data=True))

            found_any = False
            for u, v, edge_data in edge_cache[current_id]:
                edge_type = edge_data.get("edge_type")

                if edge_type not in valid_edge_types:
                    continue
                if v in path.visited_nodes:
                    continue

                found_any = True
                is_fact = v in graph.facts
                new_path = path.branch(v, is_fact, edge_type, edge_data)

                if new_path.fact_depth >= max_depth or new_path.full_depth >= total_depth:
                    completed_paths.append(new_path)
                else:
                    next_active.append(new_path)

            if not found_any:
                completed_paths.append(path)

        active_paths = next_active

    return completed_paths


def expand_paths_batched(engine, seeds, plan, max_depth=3, beam_width=3, mode="all", total_depth=None, mmr_lambda=0.5):
    graph = engine.graph
    active_paths = [Path(seed, seed in graph.facts) for seed in seeds]
    completed_paths = []

    # total_depth caps full traversal hops (including entity nodes) so paths through
    # entity-heavy graphs cannot grow unboundedly before fact_depth reaches max_depth.
    if total_depth is None:
        total_depth = max_depth * 2

    valid_edge_types = {
        "all": {"fact_fact", "entity_fact", "fact_entity"},
        "hypergraph": {"entity_fact", "fact_entity"},
        "discourse": {"fact_fact"},
    }[mode]

    plan_labels_norm = normalize_rows(engine.retrieval_model.encode(plan.target_edge_labels, show_progress_bar=False)) if plan.target_edge_labels else np.zeros((0, engine.emb_dim), dtype=np.float32)
    plan_sem_norm = normalize_rows(engine.retrieval_model.encode(plan.target_edge_semantics, show_progress_bar=False)) if plan.target_edge_semantics else np.zeros((0, engine.emb_dim), dtype=np.float32)
    plan_broad_norm = normalize_rows(engine.retrieval_model.encode(plan.broad_anchors, show_progress_bar=False)) if plan.broad_anchors else np.zeros((0, engine.emb_dim), dtype=np.float32)

    while active_paths:
        frontier = []
        next_active = []
        # Cache out_edges per node for this iteration.
        # Multiple active paths sharing the same current_node pay the NetworkX
        # adjacency lookup cost only once instead of once per path.
        edge_cache = {}

        for path_idx, path in enumerate(active_paths):
            current_id = path.current_node
            if current_id not in edge_cache:
                edge_cache[current_id] = list(graph.network.out_edges(current_id, data=True))
            outgoing_edges = edge_cache[current_id]

            found_any = False
            for u, v, edge_data in outgoing_edges:
                edge_type = edge_data.get("edge_type")
                if edge_type not in valid_edge_types:
                    continue
                if v in path.visited_nodes:
                    continue

                found_any = True
                frontier.append({
                    "path_idx": path_idx,
                    "target": v,
                    "edge_type": edge_type,
                    "edge_data": edge_data,
                    "edge_tuple": (u, v),
                })

            if not found_any:
                completed_paths.append(path)

        if not frontier:
            break

        ff_idx, ef_idx, fe_idx = [], [], []
        for i, item in enumerate(frontier):
            if item["edge_type"] == "fact_fact":
                ff_idx.append(i)
            elif item["edge_type"] == "entity_fact":
                ef_idx.append(i)
            else:
                fe_idx.append(i)

        scores = np.zeros(len(frontier), dtype=np.float32)
        diversity_vecs = [None] * len(frontier)

        if ff_idx:
            ff_edges = [frontier[i]["edge_tuple"] for i in ff_idx]
            ff_label = normalize_rows(np.vstack([engine.ff_label_embs[e][0] for e in ff_edges]))
            ff_desc = normalize_rows(np.vstack([engine.ff_desc_embs[e][0] for e in ff_edges]))
            ff_llm = np.array([frontier[i]["edge_data"].get("score", 0.0) for i in ff_idx], dtype=np.float32)
            ff_scores = batch_score_fact_fact_edges(plan_labels_norm, plan_sem_norm, ff_label, ff_desc, ff_llm)

            for j, i in enumerate(ff_idx):
                scores[i] = ff_scores[j]
                diversity_vecs[i] = ff_desc[j]

        if ef_idx:
            ef_edges = [frontier[i]["edge_tuple"] for i in ef_idx]
            ef_label = normalize_rows(np.vstack([engine.ef_label_embs[e][0] for e in ef_edges]))
            ef_scores = batch_score_entity_fact_edges(plan_broad_norm, plan_sem_norm, ef_label)

            for j, i in enumerate(ef_idx):
                scores[i] = ef_scores[j]
                diversity_vecs[i] = ef_label[j]

        if fe_idx:
            fe_edges = [frontier[i]["edge_tuple"] for i in fe_idx]
            fe_label = normalize_rows(np.vstack([engine.fe_label_embs[e][0] for e in fe_edges]))
            fe_scores = batch_score_fact_entity_edges(plan_broad_norm, fe_label)

            for j, i in enumerate(fe_idx):
                scores[i] = fe_scores[j]
                diversity_vecs[i] = fe_label[j]

        by_path = defaultdict(list)
        for i, item in enumerate(frontier):
            by_path[item["path_idx"]].append((i, item))

        for path_idx, items in by_path.items():
            path = active_paths[path_idx]
            cand_scores = [scores[i] for i, _ in items]
            cand_embs = np.vstack([diversity_vecs[i] for i, _ in items])


            chosen_local = compute_mmr(
                candidate_scores=cand_scores,
                candidate_embeddings=cand_embs,
                top_k=beam_width,
                lambda_param=mmr_lambda
            )

            for local_idx in chosen_local:
                global_i, item = items[local_idx]
                target = item["target"]
                e_type = item["edge_type"]
                e_data = item["edge_data"]
                is_fact = target in graph.facts

                new_path = path.branch(target, is_fact, e_type, e_data)

                # Stop if we've hit the desired fact depth OR if the total hop count
                # has reached the absolute ceiling (prevents explosion through entity nodes).
                if new_path.fact_depth >= max_depth or new_path.full_depth >= total_depth:
                    completed_paths.append(new_path)
                else:
                    next_active.append(new_path)

        active_paths = next_active

    return completed_paths


# ==========================================
# FINAL GLOBAL RANKING
# ==========================================

def rank_paths_global(
        engine,
        query: str,
        completed_paths: List[Path],
        cross_encoder: CrossEncoder,
        final_top_k: int = 5,
        mmr_lambda: float = 0.6,
        verbose=False
) -> Tuple[List[str], List[Path]]:
    """
    Assembles the paths into text, scores them against the query using a Cross-Encoder,
    and returns the final diverse set of logical answers using Global MMR.
    """
    if not completed_paths:
        return [], []

    # 1. Assemble Paths
    assembled_strings = [assemble_path_string(engine.graph, p) for p in completed_paths]

    # 2. Score with Cross Encoder (Relevance)
    ce_inputs = [[query, path_str] for path_str in assembled_strings]
    if verbose:
        print(f"Running Cross-Encoder on {len(ce_inputs)} assembled paths...")
    ce_scores = cross_encoder.predict(ce_inputs)

    # 3. Embed the assembled strings (Diversity)
    # This ensures we don't return 5 paths that are essentially identical variations of each other.
    path_embeddings = engine.retrieval_model.encode(assembled_strings, show_progress_bar=False)

    # 4. Global MMR Execution
    selected_indices = compute_mmr(
        candidate_scores=list(ce_scores),
        candidate_embeddings=path_embeddings,
        top_k=final_top_k,
        lambda_param=mmr_lambda
    )

    final_strings = [assembled_strings[i] for i in selected_indices]
    final_paths = [completed_paths[i] for i in selected_indices]

    return final_strings, final_paths

def expand_paths_precomputed(
        engine,
        seeds: List[str],
        plan_labels_embs: np.ndarray,
        plan_semantics_embs: np.ndarray,
        plan_broad_embs: np.ndarray,
        max_depth: int = 3,
        beam_width: int = 3,
        mode = "all"
) -> List[Path]:
    """Same as expand_paths, but uses precomputed embeddings to bypass LLM/GPU."""
    graph = engine.graph
    active_paths = []
    completed_paths = []
    valid_edge_types = []

    # use the mode to determine what type of edge traversals you will except
    if mode == "all":
        valid_edge_types = ["fact_fact", "entity_fact", "fact_entity"]
    elif mode == "hypergraph":
        valid_edge_types = ["entity_fact", "fact_entity"]
    elif mode == "discourse":
        valid_edge_types = ["fact_fact"]

    for seed in seeds:
        is_fact = seed in graph.facts
        active_paths.append(Path(start_node_id=seed, is_fact=is_fact))

    while active_paths:
        next_active = []

        for path in active_paths:
            current_id = path.current_node

            # Look up outgoing edges
            outgoing_edges = list(graph.network.out_edges(current_id, data=True))

            valid_candidates = []
            candidate_scores = []
            candidate_diversity_embs = []  # Used for Local MMR

            for u, v, edge_data in outgoing_edges:
                edge_type = edge_data.get("edge_type")

                # Filter 1: Loop prevention
                if v in path.visited_nodes:
                    continue

                # Filter 2: Ignore irrelevant edge types (like chunk_fact)
                if edge_type not in valid_edge_types:
                    continue

                edge_tuple = (u, v)
                score = 0.0
                diversity_emb = None

                # Score based on edge type using the engine's pre-computed embeddings
                if edge_type == "fact_fact":
                    score = score_fact_fact_edge(
                        plan_labels_embs, plan_semantics_embs,
                        engine.ff_label_embs[edge_tuple],
                        engine.ff_desc_embs[edge_tuple],
                        edge_data.get("score", 0.0)
                    )
                    diversity_emb = engine.ff_desc_embs[edge_tuple][0]  # Use desc for diversity

                elif edge_type == "entity_fact":
                    score = score_entity_fact_edge(
                        plan_broad_embs, plan_semantics_embs,
                        engine.ef_label_embs[edge_tuple]
                    )
                    diversity_emb = engine.ef_label_embs[edge_tuple][0]

                elif edge_type == "fact_entity":
                    score = score_fact_entity_edge(
                        plan_broad_embs,
                        engine.fe_label_embs[edge_tuple]
                    )
                    diversity_emb = engine.fe_label_embs[edge_tuple][0]

                valid_candidates.append((v, edge_type, edge_data))
                candidate_scores.append(score)
                candidate_diversity_embs.append(diversity_emb)

            # Handle Dead Ends
            if not valid_candidates:
                completed_paths.append(path)
                continue

            # Local MMR Execution (Select top 'beam_width' edges)
            selected_edge_indices = compute_mmr(
                candidate_scores=candidate_scores,
                candidate_embeddings=np.array(candidate_diversity_embs),
                top_k=beam_width,
                lambda_param=0.5  # Balance local relevance and local diversity
            )

            # Branch the path for the selected edges
            for idx in selected_edge_indices:
                target_node, e_type, e_data = valid_candidates[idx]
                is_fact = target_node in graph.facts

                new_path = path.branch(target_node, is_fact, e_type, e_data)

                # Check Depth
                if new_path.fact_depth >= max_depth:
                    completed_paths.append(new_path)
                else:
                    next_active.append(new_path)

        active_paths = next_active

    return completed_paths

def rank_paths_global_precomputed(
        completed_paths: List[Path],
        assembled_strings: List[str],
        ce_scores: List[float],
        path_embeddings: np.ndarray,
        final_top_k: int = 5
) -> Tuple[List[str], List[Path]]:
    """Same as rank_paths_global, but uses precomputed scores/embeddings."""
    if not completed_paths or not assembled_strings:
        return [], []

    selected_indices = compute_mmr(
        candidate_scores=ce_scores,
        candidate_embeddings=path_embeddings,
        top_k=final_top_k,
        lambda_param=0.6
    )

    final_strings = [assembled_strings[i] for i in selected_indices]
    final_paths = [completed_paths[i] for i in selected_indices]

    return final_strings, final_paths