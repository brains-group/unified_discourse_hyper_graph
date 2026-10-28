import numpy as np
from copy import deepcopy
from typing import List, Dict, Tuple, Optional
from sentence_transformers import CrossEncoder

from nkg.models.Graph import Graph
from nkg.utils.math_utils import *
from nkg.retrieval_2.scoring import *


def compute_mmr_batched(
        group_scores: List[np.ndarray],
        group_embeddings: List[np.ndarray],
        top_k: int,
        lambda_param: float = 0.5,
) -> List[List[int]]:
    """
    Runs MMR independently for many candidate groups (e.g. one group per active beam-search
    path) in a single batched call, instead of one compute_mmr() call — and one small matmul —
    per group.

    Equivalent to calling compute_mmr(group_scores[g], group_embeddings[g], top_k, lambda_param)
    once for every group g, but replaces the G separate small matmuls plus G separate inner
    top_k loops with:
      - one batched matmul over a zero-padded (G, K_max, D) tensor
      - top_k vectorized selection steps shared across every group at once (instead of G * top_k
        individual numpy calls)

    Groups are ragged (different candidate counts per path). Shorter groups are padded and
    masked off so padding is never selected, and a group naturally stops contributing once all
    of its real candidates have been picked (returning fewer than top_k indices for that group).

    Returns a list of length G; entry g is the list of *local* selected indices into
    group_scores[g] / group_embeddings[g] (same index semantics as compute_mmr's return value).
    """
    G = len(group_scores)
    if G == 0:
        return []

    sizes = np.array([len(s) for s in group_scores], dtype=np.intp)
    K_max = int(sizes.max()) if sizes.size else 0
    if K_max == 0:
        return [[] for _ in range(G)]

    D = next((np.atleast_2d(e).shape[1] for e in group_embeddings if len(e) > 0), None)
    if D is None:
        return [[] for _ in range(G)]

    padded_scores = np.full((G, K_max), -np.inf, dtype=np.float32)
    padded_embs = np.zeros((G, K_max, D), dtype=np.float32)
    valid_mask = np.zeros((G, K_max), dtype=bool)

    for g in range(G):
        k = int(sizes[g])
        if k == 0:
            continue
        padded_scores[g, :k] = np.asarray(group_scores[g], dtype=np.float32)
        padded_embs[g, :k, :] = np.atleast_2d(group_embeddings[g]).astype(np.float32)
        valid_mask[g, :k] = True

    top_k = min(top_k, K_max)
    g_idx = np.arange(G)

    if lambda_param == 1.0:
        # Pure relevance — no diversity term, so we don't even need the pairwise
        # similarity tensor. One vectorized sort covers every group at once.
        order = np.argsort(-padded_scores, axis=1)
        return [order[g, :min(top_k, int(sizes[g]))].tolist() for g in range(G)]

    # Single batched matmul replaces G separate embs @ embs.T calls.
    sim_tensor = padded_embs @ padded_embs.transpose(0, 2, 1)  # (G, K_max, K_max)

    selected_mask = np.zeros((G, K_max), dtype=bool)
    running_max_sim = np.full((G, K_max), -np.inf, dtype=np.float32)
    output: List[List[int]] = [[] for _ in range(G)]

    for step in range(top_k):
        if step == 0:
            # First pick per group: pure relevance, same as compute_mmr's first iteration.
            step_scores = np.where(valid_mask, padded_scores, -np.inf)
        else:
            # Padded slots hold -inf; 0 * -inf briefly produces NaN here whenever lambda_param
            # is exactly 0.0, but every such slot is immediately overwritten by -inf below
            # (valid_mask is False there), so the NaN never affects the result.
            with np.errstate(invalid="ignore"):
                mmr = lambda_param * padded_scores - (1.0 - lambda_param) * running_max_sim
            step_scores = np.where(valid_mask & ~selected_mask, mmr, -np.inf)

        best_idx = np.argmax(step_scores, axis=1)            # (G,) — one vectorized pick for every group
        best_val = step_scores[g_idx, best_idx]
        pick_valid = np.isfinite(best_val)                    # False once a group has run out of real candidates

        for g in np.where(pick_valid)[0]:
            output[g].append(int(best_idx[g]))

        selected_mask[g_idx[pick_valid], best_idx[pick_valid]] = True

        # Update running max-similarity-to-selected for next step's diversity term.
        newly_sim = sim_tensor[g_idx, :, best_idx]            # (G, K_max) sim of every candidate to this step's pick
        running_max_sim = np.where(pick_valid[:, None], np.maximum(running_max_sim, newly_sim), running_max_sim)

    return output


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
# DEDUPLICATION & NEUTRAL LAYOUT HELPERS
# ==========================================

def _get_path_nodes(path: Path) -> List[str]:
    """Extract the ordered list of node IDs from a Path's history."""
    if not path.history:
        return [path.current_node]
    nodes = [path.history[0][0]]
    for _, _, _, target in path.history:
        nodes.append(target)
    return nodes


def _build_edge_lookup(path: Path) -> Dict[Tuple[str, str], Tuple[str, dict]]:
    """Map (source_id, target_id) -> (edge_type, edge_data) for every step in the path."""
    return {(src, tgt): (edge_type, edge_data) for src, edge_type, edge_data, tgt in path.history}


def _format_node_sequence(
    graph: Graph,
    nodes: List[str],
    edge_lookup: Dict[Tuple[str, str], Tuple[str, dict]],
    neutral_layout: bool,
) -> str:
    """
    Format a contiguous sequence of node IDs into a context string.

    Standard layout: long description inlined between each pair of fact nodes.

    Neutral layout: short edge label inlined, long descriptions collected and
    appended as a 'Logical Conclusions' block. This gives the LLM the high-level
    path direction up front and the detailed reasoning below, reducing cognitive
    load from long inline descriptions while preserving all information.

    Entity→fact and fact→entity edges are formatted identically in both layouts
    since they don't carry long descriptions.
    """
    if not nodes:
        return ""

    result = _get_node_representation(graph, nodes[0])
    conclusions: List[str] = []

    for i in range(len(nodes) - 1):
        src, tgt = nodes[i], nodes[i + 1]
        entry = edge_lookup.get((src, tgt))
        tgt_rep = _get_node_representation(graph, tgt)

        if entry is None:
            result += f" -> {tgt_rep}"
            continue

        edge_type, edge_data = entry

        if edge_type == "fact_fact":
            if neutral_layout:
                label = edge_data.get("label", "relates to")
                result += f" -> {label} -> {tgt_rep}"
                desc = edge_data.get("description", "")
                if desc:
                    conclusions.append(desc)
            else:
                desc = edge_data.get("description", "is related to")
                result += f" -> {desc} -> {tgt_rep}"

        elif edge_type == "fact_entity":
            label = edge_data.get("label", "has property")
            result += f" -> (has {label}) -> {tgt_rep}"

        elif edge_type == "entity_fact":
            label = edge_data.get("label", "participates in")
            result += f" -> (participates in event: {label}) -> {tgt_rep}"

        else:
            result += f" -> {tgt_rep}"

    if neutral_layout and conclusions:
        result += "\nLogical Conclusions:\n" + "\n".join(conclusions)

    return result


def _extract_bursts(nodes: List[str], edge_cache: set) -> List[List[str]]:
    """
    Scan a node sequence and return contiguous bursts where every consecutive
    edge is absent from edge_cache. Updates edge_cache in-place for each novel
    edge immediately, so repeated edges within the same path are suppressed too.
    """
    bursts: List[List[str]] = []
    current_burst: Optional[List[str]] = None

    for i in range(len(nodes) - 1):
        a, b = nodes[i], nodes[i + 1]
        if (a, b) not in edge_cache and (b, a) not in edge_cache:
            # Novel edge: extend or start a burst and cache it right away
            if current_burst is None:
                current_burst = [a, b]
            else:
                current_burst.append(b)
            edge_cache.add((a, b))
            edge_cache.add((b, a))
        else:
            # Cached edge: close the current burst if one is open
            if current_burst is not None:
                bursts.append(current_burst)
                current_burst = None

    if current_burst is not None:
        bursts.append(current_burst)

    return bursts


def format_path(graph: Graph, path: Path, neutral_layout: bool = False) -> str:
    """Format a single Path into a context string. Supports neutral_layout mode."""
    nodes = _get_path_nodes(path)
    edge_lookup = _build_edge_lookup(path)
    return _format_node_sequence(graph, nodes, edge_lookup, neutral_layout)


def format_paths_deduplicated(
    graph: Graph,
    ranked_paths: List[Path],
    neutral_layout: bool = False,
) -> str:
    """
    Convert a priority-ranked list of Paths into a deduplicated context string.

    Algorithm:
    - Process paths in clusters. The first (highest-priority) path in each cluster
      is the "base" and is always printed in full.
    - After printing the base, all remaining paths are partitioned into those sharing
      at least one node with the base ("related") and those that don't ("unrelated").
    - Related paths are sorted by node-overlap with the base (most overlap first) so
      the most similar paths are deduplicated against the most complete cache state.
    - Each related path is then scanned for "bursts" — contiguous runs of consecutive
      edges not yet in the cache — and only those bursts are printed.
    - The edge cache is updated in-place as each burst edge is encountered, preventing
      repeats across paths AND within the same path.
    - Unrelated paths become the candidates for the next cluster.

    Both neutral_layout and standard layout are supported; see _format_node_sequence.
    """
    if not ranked_paths:
        return ""

    edge_cache: set = set()
    context_parts: List[str] = []

    # Pre-compute per-path data once to avoid repeated work in the main loop
    path_data = []
    for p in ranked_paths:
        nodes = _get_path_nodes(p)
        path_data.append({
            "nodes": nodes,
            "node_set": set(nodes),
            "edge_lookup": _build_edge_lookup(p),
        })

    remaining = list(path_data)

    while remaining:
        base = remaining.pop(0)

        # Print the base fully using the requested layout
        base_str = _format_node_sequence(graph, base["nodes"], base["edge_lookup"], neutral_layout)
        if base_str:
            context_parts.append(base_str)

        # Cache all edges from the base path (both directions)
        for i in range(len(base["nodes"]) - 1):
            a, b = base["nodes"][i], base["nodes"][i + 1]
            edge_cache.add((a, b))
            edge_cache.add((b, a))

        # Partition remaining paths by node overlap with the base
        related: List[Tuple[int, dict]] = []
        unrelated: List[dict] = []
        for pd in remaining:
            overlap = len(pd["node_set"] & base["node_set"])
            if overlap > 0:
                related.append((overlap, pd))
            else:
                unrelated.append(pd)

        # Most-overlapping paths first: deduplication removes shared edges earliest,
        # maximising the chance that later paths contribute only unique information
        related.sort(key=lambda x: x[0], reverse=True)

        for _, pd in related:
            bursts = _extract_bursts(pd["nodes"], edge_cache)
            for burst_nodes in bursts:
                burst_str = _format_node_sequence(
                    graph, burst_nodes, pd["edge_lookup"], neutral_layout
                )
                if burst_str:
                    context_parts.append(burst_str)

        # Unrelated paths seed the next cluster
        remaining = unrelated

    return "\n\n".join(context_parts)


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

            valid_candidates = []
            candidate_scores = []
            candidate_diversity_embs = []  # Used for Local MMR

            for v, edge_type, edge_data in engine.adj.get(current_id, []):
                # Filter 1: Loop prevention
                if v in path.visited_nodes:
                    continue

                # Filter 2: Ignore irrelevant edge types (like chunk_fact)
                if edge_type not in valid_edge_types:
                    continue

                edge_tuple = (current_id, v)
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

    while active_paths:
        next_active = []

        for path in active_paths:
            current_id = path.current_node

            found_any = False
            for v, edge_type, edge_data in engine.adj.get(current_id, []):
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

        for path_idx, path in enumerate(active_paths):
            current_id = path.current_node

            found_any = False
            for v, edge_type, edge_data in engine.adj.get(current_id, []):
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
                    "edge_tuple": (current_id, v),
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

        # One padded-batch MMR call covers every active path's local edge selection at this
        # depth level at once — replaces what used to be a separate compute_mmr() call (and a
        # separate small matmul) per path. See compute_mmr_batched for details.
        path_indices = list(by_path.keys())
        group_scores = [np.array([scores[i] for i, _ in by_path[pidx]], dtype=np.float32) for pidx in path_indices]
        group_embs = [np.array([diversity_vecs[i] for i, _ in by_path[pidx]], dtype=np.float32) for pidx in path_indices]

        chosen_per_group = compute_mmr_batched(
            group_scores=group_scores,
            group_embeddings=group_embs,
            top_k=beam_width,
            lambda_param=mmr_lambda
        )

        for pidx, chosen_local in zip(path_indices, chosen_per_group):
            items = by_path[pidx]
            path = active_paths[pidx]

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

            valid_candidates = []
            candidate_scores = []
            candidate_diversity_embs = []  # Used for Local MMR

            for v, edge_type, edge_data in engine.adj.get(current_id, []):
                # Filter 1: Loop prevention
                if v in path.visited_nodes:
                    continue

                # Filter 2: Ignore irrelevant edge types (like chunk_fact)
                if edge_type not in valid_edge_types:
                    continue

                edge_tuple = (current_id, v)
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