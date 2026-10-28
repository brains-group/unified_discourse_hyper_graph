"""
Goal-coverage candidate selection.

Replaces Maximal Marginal Relevance (MMR) everywhere in the new search engine.
MMR trades relevance against diversity by penalizing candidates that resemble
ones already picked -- it needs a full (candidates x candidates) similarity
matrix, and it has no notion of *why* two candidates are similar.

Goal-coverage instead greedily builds a set that maximizes how well the
query's sub-goals (see goal_plan.py) are covered in total. Diversity falls out
for free: once a goal is well covered by a selected candidate, picking another
near-duplicate of it earns much less additional reward, so selection is
naturally pulled toward whichever goal is still under-served. This also only
needs a (candidates x goals) score matrix rather than a (candidates x
candidates) one.

This is a probabilistic (soft) variant of the classic weighted maximum
coverage problem: each score is treated as the probability that a candidate
"satisfies" a goal, and a goal's coverage is 1 minus the probability that
*none* of the selected candidates satisfy it. An earlier version of this
function used hard per-goal maximums (F(S) = sum_g max_i score[i, g]) instead
-- that turned out to fully saturate after the first pick: once one candidate
dominated a goal, every other candidate for that goal scored an identical
zero marginal gain, so ranking among them degraded to array order rather than
relevance order. That silently broke the common case, since most queries plan
to exactly one goal (see goal_plan.GoalPlan). The soft version below never
fully saturates, so it still recovers plain top-k-by-score when there is only
one goal, while continuing to prefer under-covered goals over already-served
ones when there are several.

Greedy selection of a monotone submodular function is proven (1 - 1/e) ~= 63%
optimal (Nemhauser, Wolsey & Fisher, 1978); this soft-coverage objective is
submodular for the same reason facility-location/probabilistic-coverage
objectives are (see e.g. Lin & Bilmes, 2011, "A Class of Submodular Functions
for Document Summarization").
"""
import numpy as np


def select_by_goal_coverage(score_matrix: np.ndarray, top_k: int) -> list[int]:
    """
    Greedily pick up to top_k rows (candidates) from score_matrix (N, G) to
    maximize F(S) = sum over goals g of [1 - prod_{i in S} (1 - relevance[i, g])],
    where relevance is score_matrix clipped to [0, 1] (a negative cosine
    similarity means "unrelated", not "anti-coverage").

    Tracks, per goal, the running probability that goal is still unsatisfied
    by anything selected so far (`uncovered`, starting at 1.0 for every
    goal). The marginal gain of adding candidate i is
    `sum_g uncovered[g] * relevance[i, g]`: a goal that's still mostly
    uncovered rewards a relevant new candidate almost fully, while a goal
    that's nearly saturated rewards even a perfect match with very little.
    Because `uncovered` only asymptotically approaches zero, two candidates
    that both target the same goal are still ranked by their own relevance
    relative to each other -- diversity across goals is favored without
    collapsing same-goal ranking to an arbitrary tie.

    With a single goal column this reduces to ordinary top-k by score (up to
    the `uncovered` scaling factor, which is identical for every remaining
    candidate at a given step and so never changes the ranking), so
    single-goal (simple) queries behave like plain relevance ranking.

    Returns: candidate row indices into score_matrix, in selection order.
    """
    if score_matrix.size == 0:
        return []

    n_candidates = score_matrix.shape[0]
    top_k = min(top_k, n_candidates)

    relevance = np.clip(score_matrix, 0.0, 1.0).astype(np.float32)
    uncovered = np.ones(score_matrix.shape[1], dtype=np.float32)  # per-goal probability still unsatisfied
    remaining = np.ones(n_candidates, dtype=bool)
    selected: list[int] = []

    for _ in range(top_k):
        idx = np.where(remaining)[0]

        gains = relevance[idx] @ uncovered  # (|idx|,) -- marginal coverage gain per candidate

        best = int(idx[np.argmax(gains)])
        selected.append(best)
        uncovered = uncovered * (1.0 - relevance[best])
        remaining[best] = False

    return selected
