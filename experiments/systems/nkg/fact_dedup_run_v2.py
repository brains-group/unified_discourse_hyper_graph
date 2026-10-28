#!/usr/bin/env python3
"""
fact_dedup_run_v2.py
=====================
Creates run_v2 from an existing NKG construction run (run_v1) WITHOUT rebuilding the
knowledge graph: copies run_v1's essential artifacts (corpus.txt, kg.graphml,
construction_config.json, construction_results.json) into a new run directory, then
runs ONLY nkg.deduplication.fact_deduplication.GraphFactDeduplicator against the copied
graph -- removing/rewriting/merging duplicate, contradictory, or differently-scoped
facts -- and overwrites run_v2/kg.graphml with the result.

The original run_v1 directory is never touched. run_v2/kg_pre_fact_dedup.graphml is a
backup of the graph exactly as copied from run_v1, before fact dedup ran, so the two
can be diffed/compared.

run_v2 ends up with the same construction_results.json / kg.graphml shape that
experiments/systems/nkg/hqa_musique_evaluation.py --mode evaluation expects, plus a new
run_v2/fact_dedup_results.json report and a "fact_deduplication" block appended into
construction_results.json (with its token cost folded into the running totals) -- so
`--mode evaluation --run-dir run_v2` works exactly like it does against run_v1.

EXAMPLE USAGE
-------------
  python -m experiments.systems.nkg.fact_dedup_run_v2 \\
      --source-run-dir sample_exp/nkg/hotpotqa/run_v1 \\
      --target-run-dir sample_exp/nkg/hotpotqa/run_v2 \\
      --iterations 3 --top-k 12 --threshold 0.8
"""

import os
import sys
import json
import time
import shutil
import argparse
from collections import defaultdict
from datetime import datetime

import dspy
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from nkg.utils.config import configure_dspy
from nkg.models.Graph import Graph
from nkg.deduplication.fact_deduplication import GraphFactDeduplicator


# =============================================================================
# PROVIDER CONFIGURATION (mirrors hqa_musique_evaluation.py's ProviderConfig)
# =============================================================================

PROVIDER_DEFAULTS = {
    "local_vllm": {"model": "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8", "api_base": "http://localhost:8000/v1", "api_key": "EMPTY"},
    "deepseek":   {"model": "deepseek-v4-flash", "api_base": "https://api.deepseek.com", "api_key": ""},
}


def resolve_provider(args):
    base = PROVIDER_DEFAULTS[args.provider]
    model = args.model or base["model"]
    api_base = args.api_base or base["api_base"]
    api_key = args.api_key or base["api_key"]
    if args.provider == "deepseek" and not api_key:
        api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        if not api_key:
            print("[WARN] provider=deepseek: no api-key found. Set DEEPSEEK_API_KEY env var or pass --api-key.")
    return {"provider": args.provider, "model": model, "api_base": api_base, "api_key": api_key}


def dspy_model_string(cfg: dict) -> str:
    if cfg["provider"] == "local_vllm":
        return f"openai/{cfg['model']}"
    if cfg["provider"] == "deepseek":
        return f"deepseek/{cfg['model']}"
    return cfg["model"]


# =============================================================================
# TOKEN TRACKING (same approach as build_index.py's _stage_tokens)
# =============================================================================

def _stage_tokens(history_idx: int) -> dict:
    try:
        entries = dspy.settings.lm.history[history_idx:]
        prompt, completion = 0, 0
        for e in entries:
            u = e.get("usage")
            if u is None:
                continue
            if isinstance(u, dict):
                prompt += int(u.get("prompt_tokens", 0) or 0)
                completion += int(u.get("completion_tokens", 0) or 0)
            else:
                prompt += int(getattr(u, "prompt_tokens", 0) or 0)
                completion += int(getattr(u, "completion_tokens", 0) or 0)
        return {"prompt": prompt, "completion": completion, "total": prompt + completion, "calls": len(entries)}
    except Exception:
        return {"prompt": 0, "completion": 0, "total": 0, "calls": 0}


def _add_tokens(a: dict, b: dict) -> dict:
    return {k: a.get(k, 0) + b.get(k, 0) for k in ("prompt", "completion", "total", "calls")}


# =============================================================================
# COPY run_v1 -> run_v2 (essential artifacts only -- NOT the 150MB+ construction
# checkpoint/ or dag_report_*/ directories, which are irrelevant once the graph is
# already built and are not needed by --mode evaluation)
# =============================================================================

ESSENTIAL_FILES = ["corpus.txt", "construction_config.json", "construction_results.json", "kg.graphml"]
FACT_DEDUP_RESULTS_FILE = "fact_dedup_results.json"
KG_PRE_DEDUP_FILENAME = "kg_pre_fact_dedup.graphml"
KG_FILENAME = "kg.graphml"


def copy_run_dir(source_dir: str, target_dir: str, force: bool) -> bool:
    """Returns True if a fresh copy+dedup should proceed, False if already done and not --force."""
    marker = os.path.join(target_dir, FACT_DEDUP_RESULTS_FILE)
    if os.path.exists(marker) and not force:
        print(f"[FactDedup] {marker} already exists -- fact deduplication already ran for "
              f"{target_dir}. Pass --force to redo it from a fresh copy of {source_dir}.")
        return False

    os.makedirs(target_dir, exist_ok=True)
    print(f"[FactDedup] Copying essential artifacts: {source_dir} -> {target_dir}")
    for fname in ESSENTIAL_FILES:
        src = os.path.join(source_dir, fname)
        dst = os.path.join(target_dir, fname)
        if not os.path.exists(src):
            raise FileNotFoundError(f"Expected source file missing: {src}")
        shutil.copy2(src, dst)
        print(f"[FactDedup]   {fname:<28} {os.path.getsize(dst):>14,} bytes")

    backup_path = os.path.join(target_dir, KG_PRE_DEDUP_FILENAME)
    shutil.copy2(os.path.join(target_dir, KG_FILENAME), backup_path)
    print(f"[FactDedup]   Backed up pre-dedup graph -> {backup_path}")
    return True


# =============================================================================
# VERBOSE REPORT
# =============================================================================

def graph_stats(graph: Graph) -> dict:
    return {
        "n_nodes": graph.network.number_of_nodes(),
        "n_edges": graph.network.number_of_edges(),
        "n_entities": len(graph.entities),
        "n_facts": len(graph.facts),
        "n_chunks": len(graph.chunks),
    }


def print_report(before: dict, after: dict, iteration_stats: list, token_stats: dict,
                 elapsed_sec: float, params: dict, target_dir: str, lineage_info: dict = None):
    print("\n" + "=" * 78)
    print("FACT DEDUPLICATION REPORT")
    print("=" * 78)
    print(f"Run dir:         {target_dir}")
    print(f"Parameters:      iterations={params['iterations']}  top_k={params['top_k']}  "
          f"threshold={params['threshold']}  cluster_size={params['cluster_size']}")
    print(f"Sim weights:     {params['sim_weights']}")
    print(f"Retrieval model: {params['dedup_model_name']}")
    print(f"LLM:             {params['provider']}/{params['model']}  (api_base={params['api_base']})")
    print()
    print(f"{'':20}{'BEFORE':>12}{'AFTER':>12}{'DELTA':>12}")
    for key, label in [("n_facts", "Facts"), ("n_entities", "Entities"), ("n_nodes", "Graph nodes"),
                       ("n_edges", "Graph edges")]:
        b, a = before[key], after[key]
        print(f"{label:20}{b:>12,}{a:>12,}{a - b:>+12,}")
    print()

    if lineage_info:
        n_prior = lineage_info["n_hops_before_this_one"]
        orig_facts = lineage_info["original_n_facts"]
        pct_total = 100.0 * (orig_facts - after["n_facts"]) / orig_facts if orig_facts else 0.0
        pct_this_hop = 100.0 * (before["n_facts"] - after["n_facts"]) / before["n_facts"] if before["n_facts"] else 0.0
        prior_tok = lineage_info["cumulative_token_usage_before_this_hop"]["total"]
        print(f"CUMULATIVE across the full lineage ({n_prior} prior hop(s): "
              f"{', '.join(lineage_info['prior_hops'])}, then this hop):")
        print(f"  Original construction facts:  {orig_facts:,}")
        print(f"  Facts after this hop:         {after['n_facts']:,}")
        print(f"  Cumulative reduction:         {orig_facts - after['n_facts']:,}  ({pct_total:.1f}%)")
        print(f"  This hop's own reduction:     {before['n_facts'] - after['n_facts']:,}  ({pct_this_hop:.1f}%)")
        print(f"  Fact-dedup tokens before this hop: {prior_tok:,}")
        print(f"  Fact-dedup tokens this hop:        {token_stats['total']:,}")
        print(f"  Fact-dedup tokens cumulative:      {prior_tok + token_stats['total']:,}")
        print()

    total_removed  = sum(s["removed"]  for s in iteration_stats)
    total_modified = sum(s["modified"] for s in iteration_stats)
    total_merged   = sum(s["merged"]   for s in iteration_stats)
    total_clusters = sum(s["clusters_evaluated"] for s in iteration_stats)
    total_considered = sum(s["facts_considered"] for s in iteration_stats)
    all_actions = [a for s in iteration_stats for a in s["actions"]]
    facts_absorbed_into_merges = sum(len(a["before_sentences"]) for a in all_actions if a["action"] == "merge")

    print(f"Iterations run:              {len(iteration_stats)} / {params['iterations']}"
          f"{'  (stopped early -- no cluster > size 1 left)' if len(iteration_stats) < params['iterations'] else ''}")
    print(f"Clusters evaluated (total):  {total_clusters}")
    print(f"Facts considered (total):    {total_considered}  (a fact can be re-considered across iterations)")
    print(f"Facts REMOVED:               {total_removed}")
    print(f"Facts MODIFIED in place:     {total_modified}")
    print(f"MERGE operations:            {total_merged}  "
          f"({facts_absorbed_into_merges} original facts -> {total_merged} new merged facts, "
          f"net {facts_absorbed_into_merges - total_merged:+d})")
    print()

    print("Per-iteration breakdown:")
    print(f"  {'#':>3}  {'clusters':>8}  {'considered':>10}  {'removed':>7}  {'modified':>8}  {'merged':>6}  had_clusters")
    for i, s in enumerate(iteration_stats, 1):
        print(f"  {i:>3}  {s['clusters_evaluated']:>8}  {s['facts_considered']:>10}  "
              f"{s['removed']:>7}  {s['modified']:>8}  {s['merged']:>6}  {s['had_clusters']}")
    print()

    if all_actions:
        by_type = defaultdict(lambda: defaultdict(int))
        for a in all_actions:
            by_type[a["conflict_type"]][a["action"]] += 1
        print("Breakdown by conflict_type (as classified by the LLM):")
        for ctype in sorted(by_type):
            counts = by_type[ctype]
            print(f"  {ctype:<20} " + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))
        print()

        sample_n = min(15, len(all_actions))
        print(f"Sample actions (first {sample_n} of {len(all_actions)}):")
        for a in all_actions[:sample_n]:
            print(f"  [{a['action'].upper():>6}] ({a['conflict_type']}) {a['reason']}")
            for sent in a["before_sentences"]:
                print(f"      - {sent}")
            if a["resolved_sentence"]:
                print(f"      -> {a['resolved_sentence']}")
    else:
        print("No actions were taken -- no fact pair ever cleared the similarity threshold, "
              "or every cluster the LLM reviewed judged the facts fine as-is.")
    print()

    print(f"Token usage (fact dedup LLM calls only):")
    print(f"  prompt tokens:     {token_stats['prompt']:>12,}")
    print(f"  completion tokens: {token_stats['completion']:>12,}")
    print(f"  total tokens:      {token_stats['total']:>12,}")
    print(f"  LLM calls:         {token_stats['calls']:>12,}")
    print(f"Wall time: {elapsed_sec:.1f}s")
    print("=" * 78)


# =============================================================================
# MAIN
# =============================================================================

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source-run-dir", type=str, default="sample_exp/nkg/hotpotqa/run_v1")
    p.add_argument("--target-run-dir", type=str, default="sample_exp/nkg/hotpotqa/run_v2")
    p.add_argument("--iterations",   type=int,   default=3)
    p.add_argument("--top-k",        type=int,   default=12)
    p.add_argument("--threshold",    type=float, default=0.8)
    p.add_argument("--cluster-size", type=int,   default=8)
    p.add_argument("--max-workers",  type=int,   default=15)
    p.add_argument("--dedup-model-name", type=str, default="voyageai/voyage-4-nano")
    p.add_argument("--provider", type=str, default="deepseek", choices=list(PROVIDER_DEFAULTS))
    p.add_argument("--model",    type=str, default=None)
    p.add_argument("--api-base", type=str, default=None)
    p.add_argument("--api-key",  type=str, default="")
    p.add_argument("--max-tokens", type=int, default=40000)
    p.add_argument("--num-retries", type=int, default=1)
    p.add_argument("--use-cot", action=argparse.BooleanOptionalAction, default=False,
                   help="Match run_v1's construction_config.json use_cot=false by default.")
    p.add_argument("--force", action="store_true",
                   help="Re-copy from --source-run-dir and redo fact dedup even if "
                        "--target-run-dir already has a fact_dedup_results.json.")
    args = p.parse_args()

    load_dotenv()
    cfg = resolve_provider(args)

    print(f"\n{'=' * 78}")
    print(f"  NKG FACT DEDUPLICATION")
    print(f"  source : {args.source_run_dir}")
    print(f"  target : {args.target_run_dir}")
    print(f"  provider/model : {cfg['provider']}/{cfg['model']}")
    print(f"  params : iterations={args.iterations} top_k={args.top_k} threshold={args.threshold} "
          f"cluster_size={args.cluster_size}")
    print(f"{'=' * 78}\n")

    proceed = copy_run_dir(args.source_run_dir, args.target_run_dir, force=args.force)
    if not proceed:
        sys.exit(0)

    # Peek at the just-copied construction_results.json for prior fact-dedup lineage (e.g. this
    # is run_v2 -> run_v3, and run_v2's file already carries the run_v1 -> run_v2 hop) so the
    # final report can show the cumulative picture across the whole chain, not just this hop.
    lineage_info = None
    cr_path_peek = os.path.join(args.target_run_dir, "construction_results.json")
    try:
        with open(cr_path_peek, "r", encoding="utf-8") as f:
            _peek = json.load(f)
        _c = _peek.get("construction", {})
        _hist = _c.get("fact_deduplication_history", [])
        if _hist:
            lineage_info = {
                "n_hops_before_this_one": len(_hist),
                "original_n_facts": _c.get("n_facts_construction_only", _hist[0]["before"]["n_facts"]),
                "original_n_entities": _c.get("n_entities_construction_only", _hist[0]["before"]["n_entities"]),
                "prior_hops": [h["hop"] for h in _hist],
                "cumulative_token_usage_before_this_hop": {
                    "prompt": sum(h["token_usage"]["prompt"] for h in _hist),
                    "completion": sum(h["token_usage"]["completion"] for h in _hist),
                    "total": sum(h["token_usage"]["total"] for h in _hist),
                    "calls": sum(h["token_usage"]["calls"] for h in _hist),
                },
            }
    except Exception as e:
        print(f"[FactDedup] (no usable prior lineage found: {e})")

    kg_path = os.path.join(args.target_run_dir, KG_FILENAME)
    print(f"\n[FactDedup] Loading graph: {kg_path}")
    t_load0 = time.time()
    graph = Graph()
    graph.load_graph(kg_path)
    graph.label_edges()
    print(f"[FactDedup] Loaded in {time.time() - t_load0:.1f}s")

    before = graph_stats(graph)
    print(f"[FactDedup] Before: {before['n_facts']:,} facts, {before['n_entities']:,} entities, "
          f"{before['n_nodes']:,} nodes, {before['n_edges']:,} edges")

    configure_dspy(
        model=dspy_model_string(cfg), api_base=cfg["api_base"], api_key=cfg["api_key"],
        max_tokens=args.max_tokens, cache=False, num_retries=args.num_retries,
    )

    print(f"[FactDedup] Loading retrieval model: {args.dedup_model_name}")
    retrieval_model = SentenceTransformer(args.dedup_model_name, trust_remote_code=True)

    deduplicator = GraphFactDeduplicator(retrieval_model=retrieval_model, use_cot=args.use_cot)

    try:
        _h_start = len(dspy.settings.lm.history)
    except Exception:
        _h_start = 0
    t0 = time.time()

    iteration_stats = []
    graph = deduplicator.deduplicate(
        graph,
        iterations=args.iterations,
        top_k=args.top_k,
        threshold=args.threshold,
        cluster_size=args.cluster_size,
        max_workers=args.max_workers,
        iteration_stats=iteration_stats,
    )

    elapsed_sec = time.time() - t0
    token_stats = _stage_tokens(_h_start)

    after = graph_stats(graph)

    params = {
        "iterations": args.iterations, "top_k": args.top_k, "threshold": args.threshold,
        "cluster_size": args.cluster_size, "sim_weights": {"sentence": 0.7, "entity": 0.15, "topic": 0.15},
        "dedup_model_name": args.dedup_model_name, "provider": cfg["provider"], "model": cfg["model"],
        "api_base": cfg["api_base"],
    }
    print_report(before, after, iteration_stats, token_stats, elapsed_sec, params, args.target_run_dir,
                lineage_info=lineage_info)

    # ---- Save the deduplicated graph back over run_v2/kg.graphml ----
    graph.export_graph(kg_path)
    print(f"\n[FactDedup] Saved deduplicated KG -> {kg_path}  ({os.path.getsize(kg_path):,} bytes)")

    # ---- Write a dedicated fact_dedup_results.json report ----
    fact_dedup_report = {
        "run_id": f"nkg_factdedup_{datetime.now().strftime('%Y%m%dT%H%M%S')}",
        "timestamp": datetime.now().isoformat(),
        "source_run_dir": os.path.abspath(args.source_run_dir),
        "target_run_dir": os.path.abspath(args.target_run_dir),
        "lineage_info": lineage_info,
        "params": params,
        "before": before,
        "after": after,
        "totals": {
            "iterations_run": len(iteration_stats),
            "iterations_requested": args.iterations,
            "clusters_evaluated": sum(s["clusters_evaluated"] for s in iteration_stats),
            "facts_considered": sum(s["facts_considered"] for s in iteration_stats),
            "removed": sum(s["removed"] for s in iteration_stats),
            "modified": sum(s["modified"] for s in iteration_stats),
            "merged": sum(s["merged"] for s in iteration_stats),
        },
        "iteration_stats": iteration_stats,
        "token_usage": token_stats,
        "execution_time_sec": round(elapsed_sec, 2),
    }
    fact_dedup_results_path = os.path.join(args.target_run_dir, FACT_DEDUP_RESULTS_FILE)
    with open(fact_dedup_results_path, "w", encoding="utf-8") as f:
        json.dump(fact_dedup_report, f, indent=2, default=str)
    print(f"[FactDedup] Saved report -> {fact_dedup_results_path}")

    # ---- Update construction_results.json in place: post-dedup counts + rolled-up token totals,
    # so --mode evaluation --run-dir <target> (which reads llm_*_tokens from this file for its
    # "totals.all_tokens") correctly includes the fact-dedup cost. Original construction-only
    # token/count baselines are preserved under nested "*_construction_only" keys for
    # transparency -- and since this script is chained (run_v1 -> run_v2 -> run_v3 -> ...), those
    # baseline keys are set ONLY the first time a lineage goes through fact dedup (guarded by
    # "not in construction" below); on every later hop the true original-construction numbers
    # that were already stamped in the copied construction_results.json are left untouched rather
    # than being overwritten with the previous hop's post-dedup numbers. Likewise
    # "fact_deduplication" is now a per-lineage LIST (fact_deduplication_history) that each hop
    # appends to, instead of a single dict each hop would otherwise clobber. ----
    cr_path = os.path.join(args.target_run_dir, "construction_results.json")
    with open(cr_path, "r", encoding="utf-8") as f:
        construction_results = json.load(f)

    construction = construction_results["construction"]
    if "llm_prompt_tokens_construction_only" not in construction:
        # First fact-dedup hop ever applied to this lineage -- snapshot the true pre-dedup
        # construction baseline before it gets overwritten below.
        construction["n_nodes_construction_only"]    = construction["n_nodes"]
        construction["n_edges_construction_only"]    = construction["n_edges"]
        construction["n_entities_construction_only"] = construction["n_entities"]
        construction["n_facts_construction_only"]    = construction["n_facts"]
        construction["llm_prompt_tokens_construction_only"]     = construction["llm_prompt_tokens"]
        construction["llm_completion_tokens_construction_only"] = construction["llm_completion_tokens"]
        construction["llm_total_tokens_construction_only"]      = construction["llm_total_tokens"]
        construction["llm_api_calls_construction_only"]         = construction["llm_api_calls"]

    construction["n_nodes"]    = after["n_nodes"]
    construction["n_edges"]    = after["n_edges"]
    construction["n_entities"] = after["n_entities"]
    construction["n_facts"]    = after["n_facts"]

    construction["llm_prompt_tokens"]     += token_stats["prompt"]
    construction["llm_completion_tokens"] += token_stats["completion"]
    construction["llm_total_tokens"]      += token_stats["total"]
    construction["llm_api_calls"]         += token_stats["calls"]
    construction["execution_time_sec"] = round(construction.get("execution_time_sec", 0.0) + elapsed_sec, 2)

    if "fact_deduplication_history" not in construction and "fact_deduplication" in construction:
        # This lineage's PRIOR hop was written by an older version of this script that stored a
        # single "fact_deduplication" dict instead of a list -- migrate it in as history[0] so it
        # survives instead of being silently clobbered by construction["fact_deduplication"] = ...
        # further down.
        legacy = dict(construction["fact_deduplication"])
        legacy.setdefault("hop", "(legacy record predating fact_deduplication_history)")
        legacy.setdefault("timestamp", construction_results.get("timestamp", "unknown"))
        construction["fact_deduplication_history"] = [legacy]

    this_hop = {
        "hop": f"{os.path.basename(os.path.normpath(args.source_run_dir))} -> "
               f"{os.path.basename(os.path.normpath(args.target_run_dir))}",
        "timestamp": datetime.now().isoformat(),
        "params": params,
        "before": before,
        "after": after,
        "totals": fact_dedup_report["totals"],
        "token_usage": token_stats,
        "execution_time_sec": round(elapsed_sec, 2),
        "report_path": os.path.abspath(fact_dedup_results_path),
    }
    history = construction.setdefault("fact_deduplication_history", [])
    history.append(this_hop)
    construction["fact_deduplication"] = this_hop  # latest hop, kept for single-hop convenience/back-compat

    lineage = construction_results["metadata"].setdefault("run_lineage", [])
    lineage.append({
        "run_dir": os.path.abspath(args.target_run_dir),
        "parent_run_dir": os.path.abspath(args.source_run_dir),
        "fact_dedup_params": params,
        "timestamp": datetime.now().isoformat(),
    })
    construction_results["metadata"]["parent_run_dir"] = os.path.abspath(args.source_run_dir)
    construction_results["metadata"]["kg_pre_fact_dedup_path"] = os.path.abspath(
        os.path.join(args.target_run_dir, KG_PRE_DEDUP_FILENAME)
    )

    with open(cr_path, "w", encoding="utf-8") as f:
        json.dump(construction_results, f, indent=2, default=str)
    print(f"[FactDedup] Updated -> {cr_path}  (post-dedup counts + rolled-up token totals)")

    print(f"\n[FactDedup] DONE. {args.target_run_dir} is ready for evaluation.")


if __name__ == "__main__":
    main()
