"""
Content-aware scoring for nodes, edges, and paths against embedded goals.

The single most important fix here relative to the old
nkg/retrieval/scoring.py: every edge-scoring function below looks at the
*destination node's own content* (its sentence, name, role), not just the
edge's metadata (its label/description). Audit of the old traversal-time
scorers found they only ever compared edge metadata to the query -- never
the node being hopped to -- which is why disabling local beam pruning barely
moved answer accuracy: the pruning signal was only loosely related to what
actually matters for the answer.

All scoring functions return an (N, G) matrix -- one row per candidate, one
column per goal -- so results feed directly into coverage.select_by_goal_coverage.
"""
import numpy as np

# Each function's weights sum to 1.0 within that function's own weighted sum.
# The hub-degree penalty applied in score_fact_entity_candidates is a
# separate post-hoc dampening factor, not part of that budget.
FACT_NODE_WEIGHTS = {
    "sentence": 0.35, "questions": 0.20, "keyterms": 0.15, "chunk_topics": 0.15, "entity_names": 0.15,
}
ENTITY_NODE_WEIGHTS = {"name": 0.45, "role": 0.35, "anchors": 0.20}
FACT_FACT_EDGE_WEIGHTS = {"label": 0.15, "desc": 0.20, "dest_content": 0.50, "construction_score": 0.15}
ENTITY_FACT_EDGE_WEIGHTS = {"label": 0.25, "dest_content": 0.75}
# No "label" term here: construction sets the fact->entity edge label to the
# destination entity's bare ontological type (e.g. "PERSON"), and type is
# deliberately excluded from search-time scoring -- only the entity's own
# name and role (both already-informative free text) are used.
FACT_ENTITY_EDGE_WEIGHTS = {"dest_name": 0.45, "dest_role": 0.55}


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
    Score every fact's own content against every goal. Used both for seeding
    and (via a gathered destination row) for scoring fact_fact / entity_fact
    traversal edges by what the destination fact actually says.

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
    Score every entity's own content against every goal. Used both for
    seeding and (via a gathered destination row) for scoring fact_entity
    traversal edges by the destination entity's actual identity.
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
    edge_label_mat: np.ndarray,       # (N, D)
    edge_desc_mat: np.ndarray,        # (N, D)
    construction_scores: np.ndarray,  # (N,) static relevance score assigned at graph-construction time
    dest_fact_sent_mat: np.ndarray,   # (N, D) -- the destination fact's OWN sentence embedding
    goal_sentence_mat: np.ndarray,    # (G, D)
    goal_keyterms_mat: np.ndarray,    # (G, D)
) -> np.ndarray:
    """
    Score candidate fact->fact hops. The dominant term is dest_content: does
    the destination fact actually say something relevant to the goal? Edge
    label/description (what the construction-time LLM said the relationship
    *is*) and the static construction-time relevance score are kept as
    secondary signals, not the primary one.
    """
    label_scores = edge_label_mat @ goal_sentence_mat.T
    desc_scores = edge_desc_mat @ goal_sentence_mat.T
    dest_scores = 0.5 * (dest_fact_sent_mat @ goal_sentence_mat.T + dest_fact_sent_mat @ goal_keyterms_mat.T)
    construction_col = construction_scores.astype(np.float32)[:, None]  # (N, 1), broadcasts across goal columns

    w = FACT_FACT_EDGE_WEIGHTS
    return (w["label"] * label_scores + w["desc"] * desc_scores
            + w["dest_content"] * dest_scores + w["construction_score"] * construction_col)


def score_entity_fact_candidates(
    edge_label_mat: np.ndarray,      # (N, D)
    dest_fact_sent_mat: np.ndarray,  # (N, D)
    goal_sentence_mat: np.ndarray,   # (G, D)
) -> np.ndarray:
    """
    Score candidate entity->fact hops (an entity pivot fanning out into the
    facts it participates in). The old scorer only ever compared the edge
    label; this adds the destination fact's own content as the dominant signal.
    """
    label_scores = edge_label_mat @ goal_sentence_mat.T
    dest_scores = dest_fact_sent_mat @ goal_sentence_mat.T

    w = ENTITY_FACT_EDGE_WEIGHTS
    return w["label"] * label_scores + w["dest_content"] * dest_scores


def score_fact_entity_candidates(
    dest_entity_name_mat: np.ndarray,  # (N, D)
    dest_entity_role_mat: np.ndarray,  # (N, D)
    goal_keyterms_mat: np.ndarray,     # (G, D)
    goal_sentence_mat: np.ndarray,     # (G, D)
    hub_penalty: np.ndarray,           # (N,) in (0, 1], see hub_degree_penalty()
) -> np.ndarray:
    """
    Score candidate fact->entity hops -- i.e. stepping onto an entity as a
    pivot for the next hop. The old scorer only compared the edge label, which
    construction sets to the entity's bare ontological type (e.g. "PERSON")
    -- it never looked at *which* person, and type is too coarse a signal to
    be useful for relevance anyway. This scores the destination entity's
    actual name and role instead (type is intentionally not considered at
    all), and multiplies by a hub-degree penalty so generic,
    extremely-well-connected entities (e.g. a contract's boilerplate defined
    terms) are less attractive pivots than rare, specific ones -- hubs fan out
    into mostly-irrelevant territory.
    """
    name_scores = dest_entity_name_mat @ goal_keyterms_mat.T
    role_scores = dest_entity_role_mat @ goal_sentence_mat.T

    w = FACT_ENTITY_EDGE_WEIGHTS
    raw = w["dest_name"] * name_scores + w["dest_role"] * role_scores
    return raw * hub_penalty[:, None]


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
    than isolated fields).

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
