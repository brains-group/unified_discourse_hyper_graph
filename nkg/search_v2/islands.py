"""
Optional D5 candidate generation: "island and bridge".

select_seeds already scores every fact/entity in the graph against the query
-- that cost is paid regardless of pool size. Ordinary retrieval then throws
away everything outside the top handful of seeds. This module instead keeps a
much larger pool of independently-relevant nodes ("islands") and looks for
pairs that are *already* directly connected by a real edge in the graph
("bridges") -- a plain dict lookup against the adjacency map built at
construction time, with no new LLM or embedding calls. Two independently
relevant nodes that happen to be graph-neighbors is high-confidence,
essentially-free context: both ends already cleared the relevance bar on
their own, and the graph is telling you they belong to the same narrative
thread.

This is strictly additive and optional: it supplements the main beam search
with extra candidate paths, and only runs when the caller passes
use_island_and_bridge=True to GoalSearchEngine.retrieve().

search_v2: greedy extension scoring reads from the same cached NodeGoalScores
the rest of the engine uses (see content_scoring.py / engine.py) instead of
gathering edge label/description embeddings per candidate.
"""
from typing import List

import numpy as np

from nkg.search_v2.content_scoring import (
    NodeGoalScores,
    score_fact_entity_candidates,
    score_fact_fact_candidates,
)
from nkg.search_v2.goal_plan import EmbeddedGoalPlan, active_goal_indices
from nkg.search_v2.traversal import VALID_EDGE_TYPES_BY_MODE, SearchPath
from nkg.search_v2.trace import TraceRecorder, node_label


def find_island_bridges(
    engine,
    island_node_ids: List[str],
    goal_plan: EmbeddedGoalPlan,
    node_goal_scores: NodeGoalScores,
    mode: str = "all",
    max_bridge_hops: int = 1,
    tracer: TraceRecorder = None,
) -> List[SearchPath]:
    """
    Build one SearchPath per direct edge found between two nodes in
    island_node_ids, then optionally extend each bridge by up to
    max_bridge_hops further hops using the same content-aware edge scoring as
    the main beam search (greedy best-successor only -- this pass is meant to
    be a cheap bonus, not a second full beam search).
    """
    valid_edge_types = VALID_EDGE_TYPES_BY_MODE[mode]
    island_set = set(island_node_ids)
    graph = engine.graph
    tracer = tracer or TraceRecorder(enabled=False)

    bridges: List[SearchPath] = []
    seen_pairs = set()

    for u in island_node_ids:
        for v, edge_type, edge_data in engine.adj.get(u, []):
            if edge_type not in valid_edge_types or v not in island_set or v == u:
                continue
            pair = (u, v) if u < v else (v, u)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)

            bridge = SearchPath(u, u in graph.facts)
            bridge = bridge.branch(v, v in graph.facts, edge_type, edge_data)
            bridges.append(bridge)

    if max_bridge_hops > 0:
        bridges = [
            _extend_greedily(engine, p, goal_plan, node_goal_scores, valid_edge_types, max_bridge_hops)
            for p in bridges
        ]

    if tracer.enabled:
        tracer.log(
            "island_bridges",
            bridges=[
                {"nodes": [node_label(graph, n) for n in _bridge_nodes(p)]}
                for p in bridges
            ],
        )

    return bridges


def _bridge_nodes(path: SearchPath) -> List[str]:
    if not path.history:
        return [path.current_node]
    return [path.history[0][0]] + [tgt for _, _, _, tgt in path.history]


def _extend_greedily(
    engine,
    path: SearchPath,
    goal_plan: EmbeddedGoalPlan,
    node_goal_scores: NodeGoalScores,
    valid_edge_types: set,
    hops_left: int,
) -> SearchPath:
    """
    Follows the single best-scoring untried edge at most `hops_left` times.
    The goal window restarts at index 0 for each extension step regardless of
    how deep the bridge itself already is -- a deliberate simplification,
    since a bridge is a short free bonus rather than a planned traversal.
    """
    graph = engine.graph
    for hop in range(hops_left):
        goal_idx = active_goal_indices(hop, goal_plan.goals, lookahead=1)

        candidates = [
            (v, et, ed) for v, et, ed in engine.adj.get(path.current_node, [])
            if et in valid_edge_types and v not in path.visited_nodes
        ]
        if not candidates:
            break

        best_v, best_et, best_ed, best_score = None, None, None, -np.inf
        for v, et, ed in candidates:
            score = _score_single_edge(engine, v, et, ed, goal_idx, node_goal_scores)
            if score > best_score:
                best_v, best_et, best_ed, best_score = v, et, ed, score

        path = path.branch(best_v, best_v in graph.facts, best_et, best_ed)

    return path


def _score_single_edge(engine, v, edge_type, edge_data, goal_idx, node_goal_scores: NodeGoalScores) -> float:
    """Same lookup-based scoring the main beam search uses, applied to one candidate at a time."""
    if edge_type == "fact_fact":
        dest_scores = node_goal_scores.fact_scores[engine.fact_row[v]][None, goal_idx]
        construction_score = np.array([edge_data.get("score", 0.0)], dtype=np.float32)
        result = score_fact_fact_candidates(dest_scores, construction_score)
    elif edge_type == "entity_fact":
        result = node_goal_scores.fact_scores[engine.fact_row[v]][None, goal_idx]
    else:  # fact_entity
        row = engine.entity_row[v]
        dest_scores = node_goal_scores.entity_scores[row][None, goal_idx]
        penalty = engine.entity_hub_penalty[row][None]
        result = score_fact_entity_candidates(dest_scores, penalty)

    return float(result.max())
