"""
DAGRetriever: a planning layer on top of nkg.retrieval.Retriever.

Where the plain Retriever answers one query in one shot, DAGRetriever first
breaks the query into a DAG of atomic subqueries (planning.DAGPlan), then
walks that DAG level by level. Before searching a node it refines and
disambiguates that node's subquery text using the *actual retrieved context*
of its dependencies (refinement.RefineSubquery) -- resolving references like
"that university" into the concrete value found upstream -- and dynamically
forks into parallel branches whenever a dependency's context reveals
multiple distinct valid resolutions for the same placeholder.

Every actual graph search is delegated to the wrapped Retriever's own
retrieve(): this class only decides *what* to search and *in what order*,
and stitches the per-node results back into one final context string. It
does not duplicate any traversal/scoring/formatting logic from
nkg.retrieval.
"""
import itertools
from dataclasses import dataclass, field
from typing import List

import dspy

from nkg.retrieval.engine import Retriever
from .planning import DAGPlan, PlanNode, sanitize_and_level
from .refinement import RefineSubquery, RefinedSubquery
from .formatting import format_dag_context, format_prerequisite_context


@dataclass
class NodeExecution:
    """One concrete run of a DAG node -- a node forked into N branches produces N of these."""
    node_id: str
    branch_id: str
    level: int
    original_query: str
    refined_query: str
    context_string: str
    raw_paths: list = field(default_factory=list)
    parent_branch_ids: List[str] = field(default_factory=list)


@dataclass
class DAGTrace:
    """
    Everything about one DAGRetriever.retrieve(return_raw=True) call needed to inspect or render
    both the as-planned DAG and what actually ran, from a single consistent planner invocation.
    """
    by_id: dict            # node_id -> PlanNode, the plan as produced by dag_planner (pre-execution)
    levels: dict            # node_id -> int, same leveling used to order execution
    executions: List[NodeExecution]  # every concrete run, including every fork branch
    is_fallback: bool       # True if the plan collapsed to one node and execution skipped the DAG walk


class DAGRetriever:
    def __init__(
        self,
        retriever: Retriever,
        dag_planner: dspy.Module = None,
        refiner: dspy.Module = None,
    ):
        self.retriever = retriever
        self.dag_planner = dag_planner or dspy.ChainOfThought(DAGPlan)
        self.refiner = refiner or dspy.ChainOfThought(RefineSubquery)

    def retrieve(
        self,
        query: str,
        top_k_seeds: int = 9,
        max_depth: int = 1,
        beam_width: int = 3,
        final_top_k: int = 5,
        beam_mmr_lambda: float = 0.5,
        final_mmr_lambda: float = 0.6,
        mode: str = "all",
        search_mode: str = "beam_search",
        deduplicate_paths: bool = True,
        neutral_layout: bool = False,
        max_fork_branches: int = 6,
        verbose: bool = False,
        return_raw: bool = False,
        fallback_top_k_seeds: int = 9,
        fallback_max_depth: int = 3,
        fallback_beam_width: int = 6,
        fallback_final_top_k: int = 20,
        parallel_fallback_chain_threshold: int = 2,
    ):
        """
        Plans query into a DAG of subqueries, then executes it level by level, refining and
        searching each node with the wrapped Retriever, and returns one assembled context string
        (or (context_string, executions) if return_raw=True).

        top_k_seeds/max_depth default to 9/1 rather than the plain Retriever's own 20/3: each DAG
        node is already one focused sub-lookup -- the multi-hop reasoning lives in the DAG
        structure itself -- so a shallow, narrow search per *decomposed* node is intentional, not
        a shortcut. max_fork_branches caps how many dependency combinations / disambiguation
        forks a single node may expand into, so a pathological plan can't runaway the LLM call
        count.

        fallback_*: when the plan collapses to a single node with no dependencies -- i.e. the
        query didn't actually need decomposing, which is the common case for atomic lookups --
        there is nothing to disambiguate and the DAG layer adds no value over calling the wrapped
        Retriever directly. Rather than still pay for a DAG-shallow max_depth=1 search (which
        regresses relative to just asking the old engine), this delegates straight to it using
        these fuller, single-query-appropriate defaults (matching the old engine's own
        recommended settings) instead of the per-DAG-node args above, and skips the DAG-plan
        header/level-walk wrapper in the output entirely since there's no structure to show.

        parallel_fallback_chain_threshold: a short DAG (longest dependency chain <= this many
        nodes) is exactly where a single bad refinement/execution step is most costly -- e.g. a
        2-node bridge whose second node resolves its placeholder to the wrong entity has nothing
        else in the DAG to fall back on, and the wrong resolution silently becomes the only
        evidence the generation LLM sees. As a safety net, whenever the executed DAG's longest
        chain is at or under this threshold, the original, un-decomposed user_query is ALSO
        searched directly (using the fallback_* args above) as one extra independent branch,
        merged into the final context alongside the DAG's own results -- so a correct answer the
        raw query would have found on its own isn't lost just because the DAG mis-resolved a
        placeholder. Set to 0 to disable. Does not apply to the single-node case above, which
        already *is* this same raw-query search.
        """
        plan = self.dag_planner(user_query=query)
        raw_nodes = plan.nodes or [PlanNode(id="q1", query=query, depends_on=[])]
        by_id, levels = sanitize_and_level(raw_nodes)

        if len(by_id) == 1:
            only_node = next(iter(by_id.values()))
            ctx, paths = self.retriever.retrieve(
                query=only_node.query,
                top_k_seeds=fallback_top_k_seeds,
                max_depth=fallback_max_depth,
                beam_width=fallback_beam_width,
                final_top_k=fallback_final_top_k,
                beam_mmr_lambda=beam_mmr_lambda,
                final_mmr_lambda=final_mmr_lambda,
                mode=mode,
                search_mode=search_mode,
                deduplicate_paths=deduplicate_paths,
                neutral_layout=neutral_layout,
                return_raw_paths=True,
                precomputed_plan=None,
                verbose=verbose,
            )
            if return_raw:
                execution = NodeExecution(
                    node_id=only_node.id, branch_id=only_node.id, level=0,
                    original_query=only_node.query, refined_query=only_node.query,
                    context_string=ctx, raw_paths=paths, parent_branch_ids=[],
                )
                trace = DAGTrace(by_id=by_id, levels=levels, executions=[execution], is_fallback=True)
                return ctx, trace
            return ctx

        order = sorted(by_id.keys(), key=lambda nid: levels[nid])

        executions_by_node = {}

        for nid in order:
            node = by_id[nid]
            level = levels[nid]
            deps = node.depends_on

            if not deps:
                dep_combos = [[]]
            else:
                dep_combos = list(itertools.product(*(executions_by_node[d] for d in deps)))
                dep_combos = dep_combos[:max_fork_branches]

            node_results = []
            for combo_idx, combo in enumerate(dep_combos):
                combo = list(combo)
                combo_prefix = f"{nid}#{combo_idx}" if len(dep_combos) > 1 else nid

                if not deps:
                    # Source node: nothing to disambiguate yet, so skip the refinement LLM call
                    # and let the wrapped Retriever's own QueryPlan extract parameters directly.
                    refined_list = [RefinedSubquery(query=node.query)]
                    auto_plan = True
                else:
                    prereq_context = format_prerequisite_context(combo)
                    refinement = self.refiner(
                        overall_query=query,
                        subquery=node.query,
                        prerequisite_context=prereq_context,
                    )
                    refined_list = (refinement.refined or [RefinedSubquery(query=node.query)])[:max_fork_branches]
                    auto_plan = False

                for r_idx, refined in enumerate(refined_list):
                    branch_id = combo_prefix if len(refined_list) == 1 else f"{combo_prefix}.{r_idx}"
                    ctx, paths = self.retriever.retrieve(
                        query=refined.query,
                        top_k_seeds=top_k_seeds,
                        max_depth=max_depth,
                        beam_width=beam_width,
                        final_top_k=final_top_k,
                        beam_mmr_lambda=beam_mmr_lambda,
                        final_mmr_lambda=final_mmr_lambda,
                        mode=mode,
                        search_mode=search_mode,
                        deduplicate_paths=deduplicate_paths,
                        neutral_layout=neutral_layout,
                        return_raw_paths=True,
                        precomputed_plan=None if auto_plan else refined,
                        verbose=verbose,
                    )
                    node_results.append(NodeExecution(
                        node_id=nid,
                        branch_id=branch_id,
                        level=level,
                        original_query=node.query,
                        refined_query=refined.query,
                        context_string=ctx,
                        raw_paths=paths,
                        parent_branch_ids=[ex.branch_id for ex in combo],
                    ))

            executions_by_node[nid] = node_results

        all_executions = [ex for execs in executions_by_node.values() for ex in execs]

        chain_length = max(levels.values()) + 1
        if 0 < parallel_fallback_chain_threshold and chain_length <= parallel_fallback_chain_threshold:
            fb_ctx, fb_paths = self.retriever.retrieve(
                query=query,
                top_k_seeds=fallback_top_k_seeds,
                max_depth=fallback_max_depth,
                beam_width=fallback_beam_width,
                final_top_k=fallback_final_top_k,
                beam_mmr_lambda=beam_mmr_lambda,
                final_mmr_lambda=final_mmr_lambda,
                mode=mode,
                search_mode=search_mode,
                deduplicate_paths=deduplicate_paths,
                neutral_layout=neutral_layout,
                return_raw_paths=True,
                precomputed_plan=None,
                verbose=verbose,
            )
            all_executions.append(NodeExecution(
                node_id="__raw_fallback__", branch_id="raw_fallback", level=-1,
                original_query=query, refined_query=query,
                context_string=fb_ctx, raw_paths=fb_paths, parent_branch_ids=[],
            ))

        context_string = format_dag_context(by_id, levels, all_executions)

        if return_raw:
            trace = DAGTrace(by_id=by_id, levels=levels, executions=all_executions, is_fallback=False)
            return context_string, trace
        return context_string
