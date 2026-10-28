"""
Assembling the DAG's per-node retrieval results into the strings the LLM
actually sees -- one for feeding a node's dependency context into the
refinement step, one for the final answer-generation context.
"""
from typing import Dict, List

from .planning import PlanNode


def format_prerequisite_context(dep_executions: list) -> str:
    """Formats a node's already-resolved dependency results for the refinement LLM call."""
    if not dep_executions:
        return ""
    blocks = [
        f'Subquery "{ex.original_query}" found:\n{ex.context_string}'
        for ex in dep_executions
    ]
    return "\n\n".join(blocks)


def format_dag_context(by_id: Dict[str, PlanNode], levels: Dict[str, int], executions: list) -> str:
    """
    Assembles the final context string handed to the answer-generation LLM.

    Shows the original plan structure once at the top (for orientation), then walks every
    executed node deepest-level-first, printing each execution's refined query and retrieved
    context exactly once. A node that forked into several branches contributes one entry per
    branch at its own level; a node with no forked ancestors is never repeated, so shared
    prefixes of the DAG are not duplicated across branches. Deliberately omits *why* any query
    was refined the way it was -- the generation LLM sees the resolved question and its evidence,
    not the disambiguation reasoning that produced it.

    Deepest-first (not shallowest-first) is deliberate: the caller's context budget is enforced
    by truncating this string from the *end* (see truncate_tokens in
    experiments/utils/multihop_utils.py), so whatever is printed last is what gets cut first
    under a tight token budget. The deepest level holds the actually-disambiguated subquery that
    answers the original question -- each refined_query already reads as a standalone, resolved
    question (that's what refinement.RefineSubquery is for), so it doesn't depend on the shallower
    levels' raw context to be understood. The shallower levels and the plan header are the most
    dispensable content, so they're placed where truncation will reach first.

    Executions with level < 0 are the optional raw-fallback branch (see
    engine.DAGRetriever.retrieve's parallel_fallback_chain_threshold) -- an independent search of
    the original, un-decomposed query, kept as a safety net against a DAG that mis-resolved a
    placeholder. It is placed in its own section right after the header, ahead of every real DAG
    level, so it is among the least likely content to be cut if the budget is tight: it's meant to
    survive precisely the case where the DAG's own (possibly wrong) resolution is what should lose
    the truncation race, not what should protect it.
    """
    plan_lines = []
    for nid in sorted(by_id.keys(), key=lambda n: levels[n]):
        node = by_id[nid]
        dep_str = f" (depends on: {', '.join(node.depends_on)})" if node.depends_on else ""
        plan_lines.append(f"- {nid}: {node.query}{dep_str}")
    header = "Query plan:\n" + "\n".join(plan_lines)

    fallback_execs = [ex for ex in executions if ex.level < 0]
    by_level: Dict[int, List] = {}
    for ex in executions:
        if ex.level >= 0:
            by_level.setdefault(ex.level, []).append(ex)

    sections = [header]

    if fallback_execs:
        parts = ["Independent fallback search (original question, searched directly as a safety net):"]
        for ex in fallback_execs:
            body = ex.context_string or "(No relevant information found.)"
            parts.append(f"Subquery: {ex.refined_query}\n{body}")
        sections.append("\n\n".join(parts))

    for level in sorted(by_level.keys(), reverse=True):
        parts = [f"Level {level}:"]
        for ex in sorted(by_level[level], key=lambda e: e.branch_id):
            body = ex.context_string or "(No relevant information found.)"
            parts.append(f"Subquery: {ex.refined_query}\n{body}")
        sections.append("\n\n".join(parts))

    return "\n\n".join(sections)
