"""
Beam search traversal driven by goal coverage instead of MMR.

Structurally similar to the old engine's batched beam search (group the
frontier by edge type, score in bulk, then pick per-node): what changes is
(1) which goal(s) are "active" at the current depth (see
goal_plan.active_goal_indices) and (2) branch selection uses
coverage.select_by_goal_coverage over a (candidates x active_goals) matrix
instead of MMR over a (candidates x candidates) similarity matrix.

The traversal is depth-synchronized: every active path advances exactly one
hop per outer-loop iteration (classic BFS-by-layer), which is what makes it
valid to compute a single shared active-goal window per iteration rather than
recomputing it per path.
"""
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from nkg.search.content_scoring import (
    score_entity_fact_candidates,
    score_fact_entity_candidates,
    score_fact_fact_candidates,
)
from nkg.search.coverage import select_by_goal_coverage
from nkg.search.goal_plan import EmbeddedGoalPlan, active_goal_indices
from nkg.search.trace import TraceRecorder, edge_label, node_label

VALID_EDGE_TYPES_BY_MODE = {
    "all": {"fact_fact", "entity_fact", "fact_entity"},
    "hypergraph": {"entity_fact", "fact_entity"},
    "discourse": {"fact_fact"},
}


class SearchPath:
    """Tracks one traversal branch: visited nodes (loop guard), depth counters, and history."""

    __slots__ = ("current_node", "visited_nodes", "fact_depth", "full_depth", "history")

    def __init__(self, start_node_id: str, is_fact: bool):
        self.current_node = start_node_id
        self.visited_nodes: frozenset = frozenset((start_node_id,))
        self.fact_depth = 1 if is_fact else 0
        self.full_depth = 1
        self.history: List[Tuple[str, str, dict, str]] = []  # (src, edge_type, edge_data, tgt)

    def branch(self, target_node_id: str, is_fact: bool, edge_type: str, edge_data: dict) -> "SearchPath":
        """Returns a new SearchPath one hop further along; never mutates self."""
        new_path = SearchPath(target_node_id, is_fact)
        new_path.visited_nodes = self.visited_nodes | {target_node_id}
        new_path.fact_depth = self.fact_depth + (1 if is_fact else 0)
        new_path.full_depth = self.full_depth + 1
        new_path.history = self.history + [(self.current_node, edge_type, edge_data, target_node_id)]
        return new_path


def expand_frontier(
    engine,
    seeds: List[str],
    goal_plan: EmbeddedGoalPlan,
    max_depth: int = 3,
    beam_width: int = 3,
    mode: str = "all",
    goal_lookahead: int = 1,
    total_depth: Optional[int] = None,
    tracer: Optional[TraceRecorder] = None,
) -> List[SearchPath]:
    """
    Bounded beam search: from each seed, repeatedly gather every valid
    outgoing edge across the whole active frontier, score it in bulk against
    the goal(s) active at the current depth, then keep the top `beam_width`
    per path via goal-coverage selection. Stops a path when it hits a dead
    end, `max_depth` fact-hops, or `total_depth` total hops (the latter caps
    growth through long runs of entity nodes).

    tracer: if given and enabled, records every candidate edge considered
    from every node at every depth (with its score and whether it was kept),
    plus dead ends -- see trace.py. No-op (and no extra cost beyond one
    attribute check) when omitted or disabled.
    """
    graph = engine.graph
    valid_edge_types = VALID_EDGE_TYPES_BY_MODE[mode]
    total_depth = total_depth or max_depth * 2
    tracer = tracer or TraceRecorder(enabled=False)

    active_paths = [SearchPath(seed, seed in graph.facts) for seed in seeds]
    completed_paths: List[SearchPath] = []
    depth = 0

    while active_paths:
        goal_idx = active_goal_indices(depth, goal_plan.goals, lookahead=goal_lookahead)
        goal_sentence_window = goal_plan.sentence_mat[goal_idx]
        goal_keyterms_window = goal_plan.keyterms_mat[goal_idx]

        frontier = []
        for path_idx, path in enumerate(active_paths):
            found_any = False
            for v, edge_type, edge_data in engine.adj.get(path.current_node, []):
                if edge_type not in valid_edge_types or v in path.visited_nodes:
                    continue
                found_any = True
                frontier.append({
                    "path_idx": path_idx,
                    "target": v,
                    "edge_type": edge_type,
                    "edge_data": edge_data,
                    "edge_tuple": (path.current_node, v),
                })
            if not found_any:
                completed_paths.append(path)
                if tracer.enabled:
                    tracer.log("dead_end", depth=depth, node=node_label(graph, path.current_node))

        if not frontier:
            break

        scores = _score_frontier(engine, frontier, goal_sentence_window, goal_keyterms_window)

        by_path: Dict[int, List[int]] = defaultdict(list)
        for i, item in enumerate(frontier):
            by_path[item["path_idx"]].append(i)

        next_active = []
        for path_idx, item_indices in by_path.items():
            path = active_paths[path_idx]
            local_scores = scores[item_indices]  # (n_local_candidates, len(goal_idx))
            chosen_local = select_by_goal_coverage(local_scores, beam_width)

            if tracer.enabled:
                chosen_set = set(chosen_local)
                tracer.log(
                    "hop_decision",
                    depth=depth,
                    goal_indices=list(goal_idx),
                    from_node=node_label(graph, path.current_node),
                    candidates=[
                        {
                            "target": node_label(graph, frontier[gi]["target"]),
                            "edge_type": frontier[gi]["edge_type"],
                            "edge": edge_label(frontier[gi]["edge_type"], frontier[gi]["edge_data"]),
                            "scores": scores[gi].tolist(),
                            "chosen": local_pos in chosen_set,
                        }
                        for local_pos, gi in enumerate(item_indices)
                    ],
                )

            for local_pos in chosen_local:
                item = frontier[item_indices[local_pos]]
                target, e_type, e_data = item["target"], item["edge_type"], item["edge_data"]
                new_path = path.branch(target, target in graph.facts, e_type, e_data)

                if new_path.fact_depth >= max_depth or new_path.full_depth >= total_depth:
                    completed_paths.append(new_path)
                else:
                    next_active.append(new_path)

        active_paths = next_active
        depth += 1

    return completed_paths


def _score_frontier(
    engine,
    frontier: List[dict],
    goal_sentence_window: np.ndarray,
    goal_keyterms_window: np.ndarray,
) -> np.ndarray:
    """
    Scores every candidate edge in the frontier against the active goal
    window, batched by edge type so each type's score function runs as one
    vectorized matrix op instead of once per candidate.
    """
    n_goals = goal_sentence_window.shape[0]
    scores = np.zeros((len(frontier), n_goals), dtype=np.float32)

    ff_idx = [i for i, item in enumerate(frontier) if item["edge_type"] == "fact_fact"]
    ef_idx = [i for i, item in enumerate(frontier) if item["edge_type"] == "entity_fact"]
    fe_idx = [i for i, item in enumerate(frontier) if item["edge_type"] == "fact_entity"]

    if ff_idx:
        edges = [frontier[i]["edge_tuple"] for i in ff_idx]
        label_mat = np.vstack([engine.ff_label_embs[e] for e in edges])
        desc_mat = np.vstack([engine.ff_desc_embs[e] for e in edges])
        construction_scores = np.array([frontier[i]["edge_data"].get("score", 0.0) for i in ff_idx], dtype=np.float32)
        dest_mat = engine.fact_sent_mat[[engine.fact_row[frontier[i]["target"]] for i in ff_idx]]

        scores[ff_idx] = score_fact_fact_candidates(
            label_mat, desc_mat, construction_scores, dest_mat, goal_sentence_window, goal_keyterms_window
        )

    if ef_idx:
        edges = [frontier[i]["edge_tuple"] for i in ef_idx]
        label_mat = np.vstack([engine.ef_label_embs[e] for e in edges])
        dest_mat = engine.fact_sent_mat[[engine.fact_row[frontier[i]["target"]] for i in ef_idx]]

        scores[ef_idx] = score_entity_fact_candidates(label_mat, dest_mat, goal_sentence_window)

    if fe_idx:
        dest_rows = [engine.entity_row[frontier[i]["target"]] for i in fe_idx]
        name_mat = engine.entity_name_mat[dest_rows]
        role_mat = engine.entity_role_mat[dest_rows]
        penalty = engine.entity_hub_penalty[dest_rows]

        scores[fe_idx] = score_fact_entity_candidates(
            name_mat, role_mat, goal_keyterms_window, goal_sentence_window, penalty
        )

    return scores
