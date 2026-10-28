"""
Content-aware scoring for nodes, edges, and paths against embedded goals.

search_v2 fork: nkg/search compared destination-node content (its sentence,
name, role) rather than just edge metadata (label/description) -- that was
the fix over nkg/retrieval's scorers, which only ever looked at edge
metadata. This fork goes one step further: the (node x goal) content score
for *every* node in the graph is already computed once, up front, by
select_seeds (score_facts_against_goals / score_entities_against_goals below,
unchanged) -- so there is no reason to re-derive a node's content-vs-goal
score again during traversal. Edge-scoring here is now a lookup into that
cached (N, G) matrix (see NodeGoalScores and engine._score_all_nodes_against_goals)
instead of a fresh embedding gather + matmul at every hop.

The edge label/description embeddings the old traversal scorers also
compared are dropped entirely: they were already the minority weight next to
dest-content (15%+20% vs 50% for fact_fact, 25% vs 75% for entity_fact), and
the query planner has no edge-specific field to match them against in the
first place -- they were being scored against the same generic goal sentence
as everything else. Construction-time signals that are NOT node content (the
fact_fact edge's static LLM relevance score, and the fact_entity hub
penalty) are still per-edge/per-node O(1) lookups, so those are kept.

All scoring functions return an (N, G) matrix -- one row per candidate, one
column per goal -- so results feed directly into coverage.select_by_goal_coverage.
"""
from dataclasses import dataclass

import numpy as np

# Each node-scoring function's weights sum to 1.0 within its own weighted sum.
# The hub-degree penalty applied to fact_entity lookups is a separate
# post-hoc dampening factor, not part of that budget.
FACT_NODE_WEIGHTS = {
    "sentence": 0.35, "questions": 0.20, "keyterms": 0.15, "chunk_topics": 0.15, "entity_names": 0.15,
}
ENTITY_NODE_WEIGHTS = {"name": 0.45, "role": 0.35, "anchors": 0.20}
# Same dest_content:construction_score ratio the old engine used (0.50:0.15),
# renormalized to 1.0 now that the label/desc terms are gone.
FACT_FACT_EDGE_WEIGHTS = {"dest_content": 0.77, "construction_score": 0.23}


@dataclass
class NodeGoalScores:
    """
    Every entity's and every fact's own content scored against every goal --
    computed once per query (engine._score_all_nodes_against_goals) and
    reused for seeding, beam traversal, and island-and-bridge alike, instead
    of being recomputed piecemeal at every hop.
    """
    entity_scores: np.ndarray  # (N_entity, G)
    fact_scores: np.ndarray    # (N_fact, G)


def score_facts_against_goals(
    fact_sent_mat: np.ndarray,           # (N, D)
    fact_questions_mat: np.ndarray,      # (N, D)
    has_questions: np.ndarray,           # (N,) bool
    fact_chunk_topics_mat: np.ndarray,   # (N, D)
    has_chunk_topics: np.ndarray,        # (N,) bool
    fact_entity_names_mat: np.ndarray,   # (N, D)
    has_entity_names: np.ndarray,        # (N,) bool
    goal_sentence_mat: np.ndarray,       # (G, D)
    goal_keyterms_mat: np.ndarray,       # (G, D)
) -> np.ndarray:
    """
    Score every fact's own content against every goal. Feeds NodeGoalScores,
    which is used for both seeding and (via a row lookup) traversal edge
    scoring for fact_fact / entity_fact hops.

    chunk_topics and entity_names are short, proper-noun-ish phrases (the
    topics of the chunk the fact came from, and the names of the entities the
    fact mentions), so both are matched against the goal's key_terms rather
    than its full sentence, the same way the fact's own keyterms match works.
    """
    sentence_scores = fact_sent_mat @ goal_sentence_mat.T   # (N, G)
    questions_scores = fact_questions_mat @ goal_sentence_mat.T
    keyterms_scores = fact_sent_mat @ goal_keyterms_mat.T
    chunk_topics_scores = fact_chunk_topics_mat @ goal_keyterms_mat.T
    entity_names_scores = fact_entity_names_mat @ goal_keyterms_mat.T

    w = FACT_NODE_WEIGHTS
    # Facts missing an optional field (questions, chunk topics, entity names)
    # redistribute that weight onto the sentence match instead of scoring
    # against a meaningless zero vector.
    q_mask = has_questions[:, None].astype(np.float32)
    c_mask = has_chunk_topics[:, None].astype(np.float32)
    n_mask = has_entity_names[:, None].astype(np.float32)

    w_sentence = (
        w["sentence"]
        + w["questions"] * (1.0 - q_mask)
        + w["chunk_topics"] * (1.0 - c_mask)
        + w["entity_names"] * (1.0 - n_mask)
    )
    w_questions = w["questions"] * q_mask
    w_chunk_topics = w["chunk_topics"] * c_mask
    w_entity_names = w["entity_names"] * n_mask

    return (
        w_sentence * sentence_scores
        + w_questions * questions_scores
        + w["keyterms"] * keyterms_scores
        + w_chunk_topics * chunk_topics_scores
        + w_entity_names * entity_names_scores
    )


def score_entities_against_goals(
    entity_name_mat: np.ndarray,     # (N, D)
    entity_role_mat: np.ndarray,     # (N, D)
    entity_anchor_mat: np.ndarray,   # (N, D)
    has_anchors: np.ndarray,         # (N,) bool
    goal_sentence_mat: np.ndarray,   # (G, D)
    goal_keyterms_mat: np.ndarray,   # (G, D)
) -> np.ndarray:
    """
    Score every entity's own content against every goal. Feeds NodeGoalScores,
    which is used for both seeding and (via a row lookup) fact_entity
    traversal edge scoring.
    """
    name_scores = entity_name_mat @ goal_keyterms_mat.T     # names are proper nouns -> match key_terms
    role_scores = entity_role_mat @ goal_sentence_mat.T     # roles are descriptive -> match the full goal sentence
    anchor_scores = entity_anchor_mat @ goal_keyterms_mat.T

    w = ENTITY_NODE_WEIGHTS
    a_mask = has_anchors[:, None].astype(np.float32)
    w_name = w["name"] + w["anchors"] * (1.0 - a_mask) * 0.5
    w_role = w["role"] + w["anchors"] * (1.0 - a_mask) * 0.5
    w_anchors = w["anchors"] * a_mask

    return w_name * name_scores + w_role * role_scores + w_anchors * anchor_scores


def score_fact_fact_candidates(
    dest_content_scores: np.ndarray,   # (N, G) -- looked up from NodeGoalScores.fact_scores
    construction_scores: np.ndarray,   # (N,) static relevance score assigned at graph-construction time
) -> np.ndarray:
    """
    Score candidate fact->fact hops. dest_content_scores is a pure lookup (no
    matmul); construction_scores is a per-edge O(1) dict-value gather. No
    edge label/description embeddings are compared at all in search_v2.
    """
    w = FACT_FACT_EDGE_WEIGHTS
    construction_col = construction_scores.astype(np.float32)[:, None]  # (N, 1), broadcasts across goal columns
    return w["dest_content"] * dest_content_scores + w["construction_score"] * construction_col


def score_fact_entity_candidates(
    dest_content_scores: np.ndarray,  # (N, G) -- looked up from NodeGoalScores.entity_scores
    hub_penalty: np.ndarray,          # (N,) in (0, 1], see hub_degree_penalty()
) -> np.ndarray:
    """
    Score candidate fact->entity hops -- i.e. stepping onto an entity as a
    pivot for the next hop. dest_content_scores already encodes the
    destination entity's name/role/anchors (type is intentionally never
    considered, see NodeGoalScores/score_entities_against_goals); this just
    applies the hub-degree penalty so generic, extremely-well-connected
    entities (e.g. a contract's boilerplate defined terms) are less
    attractive pivots than rare, specific ones.
    """
    return dest_content_scores * hub_penalty[:, None]


def hub_degree_penalty(out_degrees: np.ndarray) -> np.ndarray:
    """
    Dampens an entity's attractiveness as a traversal pivot in proportion to
    how many outgoing edges it has. A generic defined term touching 180 facts
    is a bad thing to route through -- it explodes the search into
    mostly-unrelated territory. log1p keeps the penalty gentle for ordinary
    entities (degree ~1-5) while still meaningfully suppressing extreme hubs
    (degree > 50).
    """
    return 1.0 / (1.0 + np.log1p(out_degrees.astype(np.float32)))


def score_paths_against_goals(
    path_embeddings: np.ndarray,      # (N, D) already-normalized assembled-path embeddings
    goal_sentence_mat: np.ndarray,    # (G, D)
    normalized_relevance: np.ndarray, # (N,) cross-encoder relevance, min-max scaled to [0, 1]
) -> np.ndarray:
    """
    Final-ranking score matrix: one column per goal, plus one extra "overall
    relevance" column carrying the cross-encoder's whole-query judgement (the
    only point in the pipeline that reads full assembled path text rather
    than isolated fields). Unchanged from nkg/search -- this is a one-time
    final step over a handful of assembled paths, not a per-hop cost, so it
    isn't what search_v2's caching targets.

    The goal columns are gated by normalized_relevance before being handed to
    coverage selection. Without this, the relevance column -- min-max scaled
    so the single best candidate hits exactly 1.0 -- gets fully "covered"
    after the very first pick (uncovered = 1 - 1.0 = 0), so every remaining
    slot is chosen using only the goal-embedding columns. Those compare a
    full assembled path against a short, abstract goal sentence and don't
    discriminate well: topically-similar but cross-encoder-irrelevant paths
    (including ones the cross-encoder scores negative) routinely beat
    near-duplicates of the actually-relevant path once the "relevance" box
    is checked. Multiplying goal_scores by normalized_relevance means an
    irrelevant candidate scores near zero on every column, not just the one
    that saturates after pick one, so it can't win a slot on embedding
    similarity alone.
    """
    goal_scores = np.clip(path_embeddings @ goal_sentence_mat.T, 0.0, 1.0)  # (N, G)
    goal_scores = goal_scores * normalized_relevance[:, None]
    return np.concatenate([goal_scores, normalized_relevance[:, None]], axis=1)
