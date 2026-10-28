#!/usr/bin/env python3
"""
NKG Retrieval Speed Benchmark: nkg/search (v1) vs nkg/search_v2

Pure wall-clock comparison of the two GoalSearchEngine implementations --
NOT a quality/accuracy evaluation, and no LLM-as-judge is involved.

The one real LLM call each engine normally makes (query planning) is
replaced with a stub planner that returns a pre-built, already-embedded-free
fake plan instantly. This means:
  - No network/LLM latency is included in the timed region at all.
  - Both engines see the *same* sequence of fake plans in the same order, so
    neither one benefits or suffers from LLM output variance -- the only
    thing that can differ between engines is their own internal pipeline.

Timing starts immediately before each retrieve() call (equivalent to "right
after the plan is received from the LLM", since the stub planner returns
essentially instantaneously) and stops when retrieve() returns, so it covers
exactly: goal embedding, seeding, beam traversal, (optional island/bridge),
cross-encoder final ranking, and path formatting.

Usage:
    python -m experiments.systems.nkg.speed_benchmark \\
        --construction-dir "token_reduction_experiments/insurance/nkg/double_round_2/no_cot/neutral_desc/concise_summary/remove_cache/clustered_40/fact_batch_20/yes_qt/string_dedup/llm_dedup_50/ten_contracts_dataset/contract_2_whole_life" \\
        --num-queries 50
"""
import argparse
import os
import statistics
import time

from dotenv import load_dotenv
from sentence_transformers import CrossEncoder, SentenceTransformer

from nkg.models.Graph import Graph
from nkg.search.engine import GoalSearchEngine as GoalSearchEngineV1
from nkg.search.goal_plan import SubGoal as SubGoalV1
from nkg.search_v2.engine import GoalSearchEngine as GoalSearchEngineV2
from nkg.search_v2.goal_plan import SubGoal as SubGoalV2

KG_FILENAME = "kg.graphml"

DEFAULT_RETRIEVAL_ARGS = {
    "top_k_seeds":            10,
    "max_depth":              3,
    "beam_width":             6,
    "final_top_k":            20,
    "goal_lookahead":         1,
    "mode":                   "all",
    "use_island_and_bridge":  False,
    "island_pool_size":       75,
    "island_max_bridge_hops": 1,
    "deduplicate_paths":      True,
    "neutral_layout":         True,
}

# A handful of representative fake query plans (single-hop, dependent
# multi-hop, independent multi-hop), phrased like insurance-contract facts.
# Content realism only matters for exercising realistic traversal shape --
# this is not a quality eval, so the plans don't need to match any real QA.
FAKE_PLANS = [
    {
        "query": "What is the death benefit amount payable under the whole life policy?",
        "goals": [
            {"goal": "the death benefit amount payable under the whole life policy",
             "key_terms": ["death benefit", "whole life policy"], "depends_on_previous": False},
        ],
    },
    {
        "query": "What grace period is allowed for late premium payments?",
        "goals": [
            {"goal": "the grace period allowed for late premium payments",
             "key_terms": ["grace period", "premium payment"], "depends_on_previous": False},
        ],
    },
    {
        "query": "Who issued the policy, and what is that company's claims contact address?",
        "goals": [
            {"goal": "the insurance company that issued the policy",
             "key_terms": ["insurance company", "policy issuer"], "depends_on_previous": False},
            {"goal": "the contact address of that insurance company for filing claims",
             "key_terms": ["contact address", "claims"], "depends_on_previous": True},
        ],
    },
    {
        "query": "What is the cash surrender value schedule, and what is the policy loan interest rate?",
        "goals": [
            {"goal": "the cash surrender value schedule of the policy",
             "key_terms": ["cash surrender value"], "depends_on_previous": False},
            {"goal": "the policy loan interest rate",
             "key_terms": ["policy loan", "interest rate"], "depends_on_previous": False},
        ],
    },
    {
        "query": "How can a beneficiary designation be changed, and what documentation is required?",
        "goals": [
            {"goal": "the named beneficiary designation on the policy",
             "key_terms": ["beneficiary designation"], "depends_on_previous": False},
            {"goal": "the conditions under which a beneficiary designation can be changed",
             "key_terms": ["change beneficiary"], "depends_on_previous": True},
            {"goal": "the documentation required to change the beneficiary",
             "key_terms": ["documentation", "beneficiary change"], "depends_on_previous": True},
        ],
    },
    {
        "query": "Under what conditions does the policy lapse?",
        "goals": [
            {"goal": "the conditions that cause the policy to lapse",
             "key_terms": ["policy lapse"], "depends_on_previous": False},
        ],
    },
]


class FakePlan:
    """Duck-types the object dspy.ChainOfThought(GoalPlan)(...) would normally return."""
    def __init__(self, goals):
        self.goals = goals
        self.reasoning = "stub -- no LLM call made"


class StubPlanner:
    """Drop-in replacement for engine.planner: returns a fixed plan instantly, no LLM call."""
    def __init__(self, goals):
        self._plan = FakePlan(goals)

    def __call__(self, user_query=None, **kwargs):
        return self._plan


def build_fake_plans(subgoal_cls):
    return [
        {
            "query": spec["query"],
            "goals": [subgoal_cls(**g) for g in spec["goals"]],
        }
        for spec in FAKE_PLANS
    ]


def build_engine(engine_cls, bi_encoder, cross_encoder, kg_path, label):
    print(f"[{label}] Constructing engine (one-time content-matrix precompute)...")
    t0 = time.perf_counter()
    engine = engine_cls(retrieval_model=bi_encoder, cross_encoder=cross_encoder, graph_path=kg_path)
    print(f"[{label}] Engine built in {time.perf_counter() - t0:.2f}s "
          f"({len(engine.all_entity_ids)} entities, {len(engine.all_fact_ids)} facts)")
    return engine


def run_benchmark(engine, plans, num_queries, retrieval_args, label):
    times = []
    for i in range(num_queries):
        plan_spec = plans[i % len(plans)]
        engine.planner = StubPlanner(plan_spec["goals"])

        t0 = time.perf_counter()
        engine.retrieve(query=plan_spec["query"], **retrieval_args)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        print(f"[{label}] query {i + 1:>3}/{num_queries}  plan={i % len(plans)}  {elapsed:.4f}s")
    return times


def summarize(times, label):
    return {
        "label": label,
        "n": len(times),
        "total_sec": sum(times),
        "mean_sec": statistics.mean(times),
        "median_sec": statistics.median(times),
        "stdev_sec": statistics.stdev(times) if len(times) > 1 else 0.0,
        "min_sec": min(times),
        "max_sec": max(times),
    }


def print_summary(summary):
    print(f"\n=== {summary['label']} ({summary['n']} queries) ===")
    print(f"  total:  {summary['total_sec']:.2f}s")
    print(f"  mean:   {summary['mean_sec']:.4f}s")
    print(f"  median: {summary['median_sec']:.4f}s")
    print(f"  stdev:  {summary['stdev_sec']:.4f}s")
    print(f"  min:    {summary['min_sec']:.4f}s")
    print(f"  max:    {summary['max_sec']:.4f}s")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--construction-dir", type=str, required=True,
                         help="Directory containing a pre-built kg.graphml.")
    parser.add_argument("--num-queries", type=int, default=50)
    parser.add_argument("--retrieval-model", type=str, default="voyageai/voyage-4-nano")
    parser.add_argument("--cross-encoder-model", type=str, default="cross-encoder/ms-marco-MiniLM-L6-v2")
    args = parser.parse_args()

    load_dotenv()

    kg_path = os.path.join(args.construction_dir, KG_FILENAME)
    if not os.path.exists(kg_path):
        raise FileNotFoundError(f"No {KG_FILENAME} found under {args.construction_dir}")

    print(f"[Setup] Loading models ({args.retrieval_model}, {args.cross_encoder_model})...")
    bi_encoder = SentenceTransformer(args.retrieval_model, trust_remote_code=True)
    cross_encoder = CrossEncoder(args.cross_encoder_model)

    engine_v1 = build_engine(GoalSearchEngineV1, bi_encoder, cross_encoder, kg_path, "search (v1)")
    engine_v2 = build_engine(GoalSearchEngineV2, bi_encoder, cross_encoder, kg_path, "search_v2")

    plans_v1 = build_fake_plans(SubGoalV1)
    plans_v2 = build_fake_plans(SubGoalV2)

    print(f"\nRunning {args.num_queries} queries per engine, cycling through {len(FAKE_PLANS)} fake plans...\n")

    times_v1 = run_benchmark(engine_v1, plans_v1, args.num_queries, DEFAULT_RETRIEVAL_ARGS, "search (v1)")
    times_v2 = run_benchmark(engine_v2, plans_v2, args.num_queries, DEFAULT_RETRIEVAL_ARGS, "search_v2")

    summary_v1 = summarize(times_v1, "search (v1)")
    summary_v2 = summarize(times_v2, "search_v2")

    print_summary(summary_v1)
    print_summary(summary_v2)

    speedup = summary_v1["mean_sec"] / summary_v2["mean_sec"] if summary_v2["mean_sec"] > 0 else float("nan")
    print(f"\n=== search_v2 is {speedup:.2f}x the speed of search (v1) (mean per-query, higher = search_v2 faster) ===")


if __name__ == "__main__":
    main()
