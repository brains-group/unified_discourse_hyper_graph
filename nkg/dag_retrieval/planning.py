"""
Top-level DAG planning: decompose a query into an ordered graph of atomic
subqueries before any searching happens.

This is deliberately the *only* thing this signature does. It does not pick
search entities/topics/filters and it does not resolve ambiguous references
("that university", "the common actor") -- those are the job of
refinement.RefineSubquery, run per-node once a node's dependencies have
actually been searched. Separating the two means the DAG shape is decided
once, up front, in one cheap call, while disambiguation always has real
retrieved context to work from instead of guessing at plan time.
"""
import dspy
from pydantic import BaseModel, Field


class PlanNode(BaseModel):
    id: str = Field(description="Short unique id for this subquery, e.g. 'q1', 'q2'.")
    query: str = Field(
        description="A plain, standalone-sounding subquery in natural language. It may contain "
                    "ambiguous references (e.g. 'that university', 'the common actor') that can "
                    "only be resolved once its dependencies' answers are known -- do not try to "
                    "resolve them here, a later step handles that."
    )
    depends_on: list[str] = Field(
        default_factory=list,
        description="ids of nodes whose answers must be known before this subquery's ambiguous "
                    "terms can be resolved and it can be searched. Empty if this subquery can be "
                    "searched immediately on its own. May list more than one id (e.g. a subquery "
                    "that needs the results of two independent earlier subqueries)."
    )


class DAGPlan(dspy.Signature):
    """
    You are planning how to search a knowledge graph to answer user_query, by breaking it into
    a DAG (not necessarily a simple chain) of atomic subqueries.

    Think like a person who can't answer the question in one lookup and has to plan a sequence of
    smaller lookups first. Follow this procedure in order:

    STEP 1 -- Inventory every constraint already given in user_query. List every named entity,
    date, number, and descriptive condition stated in the question, exactly as written. Nothing in
    this list may be dropped or simplified away later -- a constraint that gets lost during
    decomposition makes every subquery search for the wrong, far-too-generic thing.

    STEP 2 -- Identify what is actually unknown. For each piece of the question that isn't a named
    constraint, ask: "can this be searched right now, using only the constraints from Step 1?"
        - If YES for everything -- there is only one unknown and it's directly searchable -- this
          is a single-fact query: emit exactly ONE node with no dependencies, and its query text
          must include EVERY constraint from Step 1 verbatim (or lightly reworded, but never
          simplified away). A generic-sounding one-node query that dropped a constraint is a
          planning failure, not a simplification.
        - If a piece of the question can only be named/phrased once ANOTHER unknown is resolved
          first (it reads as "that X", "the same Y", "this person", or is logically an attribute of
          an answer you don't have yet) -- this is a BRIDGE: it needs its own node that depends_on
          the node producing that prerequisite answer.
        - If there are two or more independent unknowns that each stand on their own (neither one's
          wording depends on the other's answer) and the question wants them checked side by side --
          this is a COMPARISON: give each one its own node with NO dependency between them, and stop
          there. Do NOT add a further node that just re-asks the original comparison ("do X and Y
          share Z?") -- that extra node has nothing new to search for; the comparison itself is made
          by reading the sibling nodes' results side by side once they're both retrieved, not by
          another graph lookup.

    STEP 3 -- Decide whether a fan-in node is actually needed. Only add a node that depends_on
    MULTIPLE earlier nodes when answering user_query requires a genuinely NEW graph lookup that can
    only be phrased after combining those nodes' answers (e.g. "what is the capital of the country
    where both director A and director B were born?" -- once node A and node B each independently
    find a birth country, a NEW lookup, "what is the capital of that shared country", only becomes
    nameable once both are known, so it correctly depends_on both). If instead the final step is
    just comparing, intersecting, or reading off the sibling nodes' answers with no further lookup
    (e.g. "were A and B both French?", "which of A's roles and B's roles overlap?", "was the actor
    common to A and B named X?" when both are independently searchable lists) -- do NOT create that
    node; stop at the independent siblings from Step 2, exactly like a comparison.

    STEP 4 -- Write each node's query text so it captures ONLY the new question being resolved at
    that step. Do not restate or re-embed a prior node's full question text inside a later node --
    the depends_on edge already records that ordering; a later node's job is to ask the next NEW
    thing, using a placeholder for whatever the dependency will supply, not to repeat everything
    asked so far.

    STEP 5 -- Sanity-check every dependency. For each node with depends_on set, confirm it truly
    cannot be phrased or searched without that prerequisite's answer (it contains an unresolvable
    placeholder reference). Do not invent a dependency that isn't truly required -- every
    dependency delays that node's execution until its prerequisites finish, and a dependency that
    isn't real just adds latency and a chance for the prerequisite's answer to be wrong to also
    corrupt this node.

    WORKED EXAMPLES:
    - "What year was the magazine that Stanley Bing spent a decade at founded in?" -- one bridge
      chain: q1 (no deps) "What magazine did Stanley Bing spend a decade at?"; q2 (depends_on q1)
      "What year was that magazine founded in?" -- q2 does not repeat "Stanley Bing" or "a decade",
      it only asks the new thing (the founding year) using a placeholder for q1's answer.
    - "Were both Roger Donaldson and André Cayatte French filmmakers?" -- comparison, two sibling
      nodes, no third node: q1 (no deps) "What nationality is Roger Donaldson?"; q2 (no deps) "What
      nationality is André Cayatte?". The actual yes/no comparison is made once both are retrieved,
      not by a third "were both French?" node.
    - "What Russian Marxist revolutionary born in 1875 was part of the Vpered organization?" -- this
      is one unknown person checked against two independent constraints, so treat it like a
      comparison rather than forcing a single vague node: q1 (no deps) "Which Russian Marxist
      revolutionaries were born in 1875?"; q2 (no deps) "Who was part of the Vpered organization?".
      The answer is whichever name appears in both retrieved results.
    - "Who is the English professional footballer who is the niece of a former footballer born in
      the same place as where he plays?" -- there is no independently-nameable bridge entity here
      (the "former footballer" is only described, never named, and is itself defined relative to the
      answer), so this cannot be cleanly decomposed: it must stay ONE node, and that node's query
      text must carry the FULL compound description verbatim. Do NOT simplify it down to a generic
      category question like "Who is an English professional footballer?" -- that drops every
      constraint that made the question answerable at all.
    - "Which TV series, written by the creators of Show X, is set in Location Y?" -- proper 3-step
      bridge chain, each step adding exactly one new piece: q1 (no deps) "Who created Show X?"; q2
      (depends_on q1) "What TV series were written by these creators?"; q3 (depends_on q2) "Which of
      these TV series is set in Location Y?". Notice q2 does not repeat "Show X" or re-ask who the
      creators are, and q3 does not repeat the creators or "written by" -- each step only adds the
      one new fact needed on top of what the previous step already established.

    Do not try to resolve ambiguous references yourself; just record the dependency and leave the
    ambiguous wording in place, a later step resolves it using the dependency's actual retrieved
    answer. Do not specify search parameters (entities, topics, filters) here either -- just the
    plain question text and the dependency structure.
    """
    user_query: str = dspy.InputField()

    nodes: list[PlanNode] = dspy.OutputField(
        desc="The DAG of subqueries needed to answer user_query. Length 1 for simple queries."
    )


def sanitize_and_level(nodes: list[PlanNode]) -> tuple[dict[str, PlanNode], dict[str, int]]:
    """
    Validates a raw LLM-produced node list into a clean DAG and computes each node's level
    (0 = no dependencies, otherwise 1 + max(level of its dependencies)).

    Defensive because this is untrusted LLM output at a system boundary: drops depends_on ids
    that don't exist or self-reference, and breaks any cycle by clearing dependencies for
    whichever nodes are left stuck once no more progress can be made (rare, but a cycle here
    would infinite-loop leveling otherwise).
    """
    by_id = {n.id: n for n in nodes}
    for n in nodes:
        n.depends_on = [d for d in n.depends_on if d in by_id and d != n.id]

    levels: dict[str, int] = {}
    remaining = dict(by_id)
    while remaining:
        ready = [nid for nid, n in remaining.items() if all(d in levels for d in n.depends_on)]
        if not ready:
            # Cycle among whatever's left: break it by dropping their dependencies outright.
            for n in remaining.values():
                n.depends_on = []
            continue
        for nid in ready:
            deps = remaining[nid].depends_on
            levels[nid] = 1 + max((levels[d] for d in deps), default=-1)
            del remaining[nid]

    return by_id, levels
