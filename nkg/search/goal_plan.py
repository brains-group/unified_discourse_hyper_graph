"""
Depth-aligned query planning.

The old planner (nkg/retrieval/planner.py) writes one flat, unordered bag of
target entities/topics/anchors and reuses it unchanged at every hop of the
traversal. That is the single biggest source of "the query plan is too
broad": a 3-hop question ends up scored against the exact same static bag at
hop 1 as at hop 3.

This planner instead asks the LLM to do what a person does silently before
they start looking something up: write down the ordered list of things
they'd need to find, one sub-goal at a time. For "what famous alumni did
Einstein share his alma mater with?" that's [1] "the school Einstein
attended", [2] "other famous alumni of that school". Traversal then scores
primarily against whichever sub-goal is "active" at the current depth (see
active_goal_indices below and its use in traversal.py), so the search
direction actually shifts as it goes deeper -- at no extra LLM cost, since
the whole list still comes from one upfront call, exactly like the old
single-shot plan.
"""
from dataclasses import dataclass

import dspy
import numpy as np
from pydantic import BaseModel, Field

from nkg.utils.math_utils import normalize_rows


class SubGoal(BaseModel):
    goal: str = Field(
        description="One thing that needs to be found or confirmed to answer the query, "
                    "phrased as a natural sentence (e.g. 'the school Albert Einstein attended')."
    )
    key_terms: list[str] = Field(
        default_factory=list,
        description="1-3 precise proper nouns or role phrases central to just this goal "
                    "(e.g. ['Albert Einstein', 'alma mater']). Empty if the goal is purely "
                    "thematic and names nothing specific."
    )
    depends_on_previous: bool = Field(
        description="True if this goal can only be resolved after the previous goal's answer "
                    "is known (e.g. needs 'the school' before it can ask about alumni). False "
                    "if this goal could be checked independently, in any order."
    )


class GoalPlan(dspy.Signature):
    """
    You are planning how to search a knowledge graph to answer user_query.
    user_query is already a clean, well-formed query — use it as-is. Do NOT
    rewrite or rephrase it.

    Think like someone mentally mapping out a lookup path before starting,
    not like someone guessing at random links. Break the query into an
    ORDERED sequence of sub-goals -- the sequence of things that need to be
    found, in the order a person would look them up.

    CRITICAL CONSTRAINTS:
    1. Simple, single-fact queries get exactly ONE goal. Do not invent extra
       steps for questions that only need one lookup.
    2. Only add a second/third goal when the query genuinely requires
       chaining through an intermediate fact to reach the answer (true
       multi-hop questions).
    3. Precision over volume: key_terms is 1-3 sharp terms, never a sentence.
    4. Do not hallucinate specifics the query didn't mention -- if a goal has
       no obvious proper noun, leave key_terms empty rather than inventing one.
    5. Each goal's text is matched by semantic similarity directly against
       sentences from the source documents, so it must read like a statement
       of the fact being sought, not like an instruction to yourself. Never
       phrase a goal as a task or command (avoid verbs like "investigate",
       "find", "determine", "look up", "check"). Write it the way the answer
       itself would be phrased in a document, so it sits close in meaning to
       the sentences it needs to match.
    """
    user_query: str = dspy.InputField()

    goals: list[SubGoal] = dspy.OutputField(
        desc="Ordered sub-goals, from the first thing to look up to the last. Length 1 for "
             "simple queries; length 2-4 for genuine multi-hop queries."
    )


@dataclass
class EmbeddedGoalPlan:
    """
    A GoalPlan with every text field pre-embedded and L2-normalized, so every
    downstream scoring step is a plain dot product.

    sentence_mat / keyterms_mat: (num_goals, D) -- one row per goal, in order.
        keyterms_mat's row is the embedding of that goal's key_terms joined
        into a single string (e.g. "Albert Einstein | alma mater"), falling
        back to the goal's own sentence embedding when it has no key_terms.
        Joining into one string and embedding once (rather than embedding
        each term separately and averaging the vectors) keeps every
        downstream comparison a single matmul, and lets the embedding model
        itself capture the relationship between the terms instead of
        blending independently-computed vectors.
    goals: the original SubGoal list (for depends_on_previous / text access).
    raw_plan: the raw dspy Prediction returned by the planner (has
        .reasoning, .goals) -- kept around purely for tracing/debugging
        (see trace.py), not read by any scoring code.
    """
    goals: list[SubGoal]
    sentence_mat: np.ndarray
    keyterms_mat: np.ndarray
    raw_plan: object = None


def generate_and_embed_goal_plan(
    query: str,
    retrieval_model,
    planner: dspy.Module = None,
) -> EmbeddedGoalPlan:
    """
    Runs the one-shot GoalPlan LLM call and embeds every field. This is the
    only LLM call the whole retrieval pipeline makes.
    """
    planner = planner or dspy.ChainOfThought(GoalPlan)
    plan = planner(user_query=query)

    goals = plan.goals or [SubGoal(goal=query, key_terms=[], depends_on_previous=False)]

    sentence_mat = normalize_rows(retrieval_model.encode([g.goal for g in goals], show_progress_bar=False)).astype(np.float32)

    keyterms_mat = sentence_mat.copy()  # default: fall back to the goal sentence itself
    keyterm_rows = [i for i, g in enumerate(goals) if g.key_terms]
    if keyterm_rows:
        joined = [" | ".join(goals[i].key_terms) for i in keyterm_rows]
        joined_embs = normalize_rows(retrieval_model.encode(joined, show_progress_bar=False))
        for row, i in enumerate(keyterm_rows):
            keyterms_mat[i] = joined_embs[row]

    return EmbeddedGoalPlan(
        goals=goals,
        sentence_mat=sentence_mat,
        keyterms_mat=keyterms_mat.astype(np.float32),
        raw_plan=plan,
    )


def active_goal_indices(depth: int, goals: list[SubGoal], lookahead: int = 1) -> list[int]:
    """
    Which goal(s) should traversal score against at a given depth?

    The primary goal is simply goals[depth], clamped to the last goal once
    traversal runs deeper than the plan anticipated. In addition, peek ahead
    up to `lookahead` further goals that are marked independent
    (depends_on_previous=False) -- those can be picked up opportunistically
    whenever a relevant edge appears, rather than waiting for their "turn."
    Goals gated on a previous goal are left out until it's their turn, since
    scoring against them before their prerequisite is found is just noise.
    """
    n = len(goals)
    primary = min(depth, n - 1)
    indices = {primary}
    for j in range(primary + 1, min(primary + 1 + lookahead, n)):
        if not goals[j].depends_on_previous:
            indices.add(j)
    return sorted(indices)
