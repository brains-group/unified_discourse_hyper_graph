#!/usr/bin/env python3
"""
NKG Retrieval Deep-Trace Debugger

Runs GoalSearchEngine.retrieve(..., trace=True) for one or more questions from
a contract's QA file and prints/saves a full account of every decision the
retrieval pipeline made: the goal plan (incl. whether the planner's LLM call
silently retried), every seed candidate's score, every edge considered at
every traversal hop and whether it was kept, and every final-ranking
candidate's cross-encoder score.

The QA pairs' "evidence" field (the source-text span that actually answers
the question) is cross-referenced against the graph's fact nodes by word
overlap, so you can see exactly where the evidence dropped out of the
pipeline: never reachable at all, reachable but not seeded, seeded but never
selected during traversal, or reached but cut at final ranking.

---------------------------------------------------------------------
Example (trace one question):
    python3 -m experiments.systems.nkg.trace_retrieval \\
        --construction-dir ".../ten_contracts_dataset/contract_1_term_life" \\
        --eval-data-dir ./evaluation_data \\
        --question-index 0 --default-retrieval-args

Example (trace every CAT2 question, save full JSON+text report):
    python3 -m experiments.systems.nkg.trace_retrieval \\
        --construction-dir ".../contract_1_term_life" \\
        --eval-data-dir ./evaluation_data \\
        --category CAT2_MULTI_HOP --all --out-dir ./trace_reports
---------------------------------------------------------------------
"""

import os
import re
import sys
import json
import argparse
from datetime import datetime

from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer

from nkg.utils.config import configure_dspy
from nkg.search.engine import GoalSearchEngine
from experiments.evaluate_mine import ask_llm_query, score_llm_answer
from experiments.utils.chunking import truncate_context
from experiments.systems.nkg.insurance_evaluation import (
    CONSTRUCTION_JSON,
    KG_FILENAME,
    resolve_paths_for_contract,
    default_retrieval_args,
)


# =============================================================================
# EVIDENCE <-> FACT-NODE MATCHING
# =============================================================================
_WORD_RE = re.compile(r"[a-z0-9]+")


def _word_set(text: str) -> set:
    return set(_WORD_RE.findall((text or "").lower()))


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def rank_facts_by_evidence_overlap(graph, evidence: str, top_n: int = 5):
    """Every fact node's sentence, ranked by word-overlap against the evidence text."""
    evidence_words = _word_set(evidence)
    if not evidence_words:
        return []
    scored = [
        (fid, fact.sentence, _jaccard(evidence_words, _word_set(fact.sentence)))
        for fid, fact in graph.facts.items()
    ]
    scored.sort(key=lambda t: t[2], reverse=True)
    return scored[:top_n]


# =============================================================================
# TRACE -> HUMAN-READABLE REPORT
# =============================================================================
def find_stage_mentions(tracer, fact_text: str) -> dict:
    """Where (if anywhere) a specific fact's text shows up across the trace stages."""
    mentions = {"seeded": None, "hop_appearances": [], "final_ranked": None}

    for e in tracer.stage("seeding"):
        for c in e["candidates"]:
            if c["text"] == fact_text:
                mentions["seeded"] = {"selected": c["selected"], "rank": c["selection_rank"], "scores": c["scores"]}

    for e in tracer.stage("hop_decision"):
        for c in e["candidates"]:
            if c["target"] == fact_text:
                mentions["hop_appearances"].append({
                    "depth": e["depth"], "from": e["from_node"], "chosen": c["chosen"], "scores": c["scores"],
                })

    for e in tracer.stage("final_ranking"):
        for c in e["candidates"]:
            if fact_text in c["path_text"]:
                mentions["final_ranked"] = {
                    "selected": c["selected"], "rank": c["selection_rank"],
                    "cross_encoder_score": c["cross_encoder_score"],
                }

    return mentions


def format_report(question_record: dict, max_candidates_per_hop: int = 25) -> str:
    lines = []
    w = lines.append

    w("=" * 100)
    w(f"QUESTION: {question_record['question']}")
    w(f"CATEGORY: {question_record.get('category', 'UNKNOWN')}")
    w(f"EXPECTED ANSWER: {question_record['expected_answer']}")
    w(f"EVIDENCE: {question_record.get('evidence', '(none provided)')}")
    w("=" * 100)

    gp = question_record["goal_plan"]
    w("\n--- GOAL PLAN ---")
    w(f"rewritten_query: {gp['rewritten_query']}")
    if gp.get("reasoning"):
        w(f"reasoning: {gp['reasoning']}")
    for i, g in enumerate(gp["goals"]):
        w(f"  [{i}] {g['goal']}  (key_terms={g['key_terms']}, depends_on_previous={g['depends_on_previous']})")
    n_calls = len(gp["lm_calls"])
    total_tok = sum(c["total_tokens"] for c in gp["lm_calls"])
    w(f"planner LLM calls: {n_calls}  (total_tokens={total_tok})"
      + ("   <-- 2+ calls means the ChatAdapter's structured-output parse failed "
         "and silently retried with JSONAdapter (see dspy.adapters.chat_adapter)" if n_calls > 1 else ""))

    w("\n--- EVIDENCE CROSS-REFERENCE ---")
    for fid, sentence, overlap in question_record["evidence_matches"]:
        w(f"\n  candidate evidence fact (word-overlap={overlap:.2f}): {sentence!r}")
        mentions = question_record["evidence_mentions"][fid]
        if mentions["seeded"] is None:
            w("    seeding      : NOT a candidate (score too low to make the seed pool, or below top_k_seeds)")
        else:
            s = mentions["seeded"]
            w(f"    seeding      : candidate, selected={s['selected']}"
              + (f" (rank {s['rank']})" if s["selected"] else "") + f", scores={s['scores']}")
        if not mentions["hop_appearances"]:
            w("    traversal    : never appeared as an outgoing edge target from any active path")
        else:
            for h in mentions["hop_appearances"]:
                w(f"    traversal    : depth={h['depth']} from={h['from']!r} chosen={h['chosen']} scores={h['scores']}")
        if mentions["final_ranked"] is None:
            w("    final_ranking: never assembled into a candidate path")
        else:
            f = mentions["final_ranked"]
            w(f"    final_ranking: selected={f['selected']}"
              + (f" (rank {f['rank']})" if f["selected"] else "")
              + f", cross_encoder_score={f['cross_encoder_score']:.3f}")

    w("\n--- SEEDING (top candidates by score) ---")
    seeding_candidates = question_record["seeding_candidates"]
    top_seeded = sorted(seeding_candidates, key=lambda c: max(c["scores"]), reverse=True)[:max_candidates_per_hop]
    for c in top_seeded:
        mark = f"SELECTED(rank {c['selection_rank']})" if c["selected"] else "-"
        w(f"  [{mark:>16s}] scores={['%.3f' % s for s in c['scores']]}  {c['text'][:90]!r}")

    w(f"\n--- TRAVERSAL ({len(question_record['hop_events'])} hop decisions, "
      f"{question_record['n_dead_ends']} dead ends) ---")
    for h in question_record["hop_events"]:
        w(f"\n  depth={h['depth']} goal_indices={h['goal_indices']} from: {h['from_node'][:90]!r}")
        cands = h["candidates"][:max_candidates_per_hop]
        for c in cands:
            mark = "KEEP" if c["chosen"] else "drop"
            w(f"    [{mark}] ({c['edge_type']}, {c['edge']!r}) -> {c['target'][:80]!r}  scores={['%.3f' % s for s in c['scores']]}")
        if len(h["candidates"]) > max_candidates_per_hop:
            w(f"    ... +{len(h['candidates']) - max_candidates_per_hop} more candidates not shown")

    w("\n--- FINAL RANKING ---")
    final_candidates = question_record["final_candidates"]
    for c in sorted(final_candidates, key=lambda c: c["cross_encoder_score"], reverse=True)[:max_candidates_per_hop]:
        mark = f"SELECTED(rank {c['selection_rank']})" if c["selected"] else "-"
        w(f"  [{mark:>16s}] ce={c['cross_encoder_score']:.3f}  {c['path_text'][:100]!r}")

    w("\n--- RETRIEVED CONTEXT (what the generation LLM actually saw) ---")
    w(question_record["retrieved_context"])

    w("\n--- GENERATION ---")
    w(f"LLM answer: {question_record['llm_answer']}")
    w(f"Judge score: {question_record['score']}")
    w("=" * 100)

    return "\n".join(lines)


# =============================================================================
# MAIN TRACE DRIVER
# =============================================================================
def trace_one_question(pair, retriever, retrieval_args, tokenizer, max_context_tokens, graph):
    question = pair["question"]
    expected = pair["answer"]
    evidence = pair.get("evidence", "")

    (raw_context, paths), tracer = retriever.retrieve(
        query=question, return_raw_paths=True, trace=True, **retrieval_args,
    )
    truncated_context, context_tokens = truncate_context(raw_context, tokenizer, max_context_tokens)
    llm_answer = ask_llm_query(question=question, context=truncated_context)
    score = score_llm_answer(question=question, llm_answer=llm_answer, correct_answer=expected)

    goal_plan_events = tracer.stage("goal_plan")
    seeding_events = tracer.stage("seeding")
    hop_events = tracer.stage("hop_decision")
    final_events = tracer.stage("final_ranking")

    evidence_matches = rank_facts_by_evidence_overlap(graph, evidence, top_n=3) if evidence else []
    evidence_mentions = {fid: find_stage_mentions(tracer, sentence) for fid, sentence, _ in evidence_matches}

    return {
        "question": question,
        "expected_answer": expected,
        "category": pair.get("category", "UNKNOWN"),
        "evidence": evidence,
        "goal_plan": goal_plan_events[0] if goal_plan_events else {},
        "seeding_candidates": seeding_events[0]["candidates"] if seeding_events else [],
        "hop_events": hop_events,
        "n_dead_ends": len(tracer.stage("dead_end")),
        "final_candidates": final_events[0]["candidates"] if final_events else [],
        "evidence_matches": evidence_matches,
        "evidence_mentions": evidence_mentions,
        "retrieved_context": truncated_context,
        "context_tokens": context_tokens,
        "llm_answer": llm_answer,
        "score": score,
        "raw_events": tracer.events,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Deep-trace NKG's new search engine for specific QA questions, cross-referenced "
                    "against the QA pairs' evidence field."
    )
    parser.add_argument("--construction-dir", type=str, required=True,
                        help="Directory containing construction.json + kg.graphml for one contract.")
    parser.add_argument("--eval-data-dir", type=str, required=True,
                        help="Root evaluation_data/ directory, used to resolve the contract's QA file.")
    parser.add_argument("--question-index", type=int, default=0,
                        help="Which question (0-based, in qa_pairs order) to trace. Ignored if --all.")
    parser.add_argument("--all", action="store_true", help="Trace every question in the QA file.")
    parser.add_argument("--category", type=str, default=None,
                        help="Only trace questions in this category (e.g. CAT2_MULTI_HOP).")
    parser.add_argument("--limit", type=int, default=None, help="Cap the number of questions traced.")
    parser.add_argument("--retrieval-model", type=str, default="voyageai/voyage-4-nano")
    parser.add_argument("--cross-encoder-model", type=str, default="cross-encoder/ms-marco-MiniLM-L6-v2")
    parser.add_argument("--tokenizer-model", type=str, default="Qwen/Qwen3-30B-A3B-Instruct-2507-FP8")
    parser.add_argument("--max-context-tokens", type=int, default=12000)
    parser.add_argument("--judge-max-tokens", type=int, default=8000)
    parser.add_argument("--default-retrieval-args", action="store_true",
                        help="Use the new engine's hardcoded default retrieval args.")
    parser.add_argument("--top-k-seeds", type=int, default=10)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--beam-width", type=int, default=6)
    parser.add_argument("--final-top-k", type=int, default=20)
    parser.add_argument("--goal-lookahead", type=int, default=1)
    parser.add_argument("--mode", type=str, default="all", choices=["all", "hypergraph", "discourse"])
    parser.add_argument("--use-island-and-bridge", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-candidates-per-hop", type=int, default=25,
                        help="Cap on how many candidates are printed per hop/seeding/ranking event "
                            "in the text report (the saved JSON always has everything).")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="If given, saves a .txt report and a .json raw-trace dump per question here. "
                            "Otherwise just prints to stdout.")
    args = parser.parse_args()

    construction_dir = os.path.abspath(args.construction_dir)
    eval_data_dir = os.path.abspath(args.eval_data_dir)

    with open(os.path.join(construction_dir, CONSTRUCTION_JSON), "r", encoding="utf-8") as f:
        construction_record = json.load(f)
    contract_path = construction_record["contract_file"]
    qa_path, _fact_path = resolve_paths_for_contract(contract_path, eval_data_dir)

    with open(qa_path, "r", encoding="utf-8") as f:
        qa_pairs = json.load(f).get("qa_pairs", [])

    if args.category:
        qa_pairs = [p for p in qa_pairs if p.get("category") == args.category]
    if args.all or args.category:
        selected_pairs = qa_pairs[: args.limit] if args.limit else qa_pairs
    else:
        selected_pairs = [qa_pairs[args.question_index]]

    if not selected_pairs:
        print("[ERROR] No matching questions found.")
        sys.exit(1)

    if args.default_retrieval_args:
        retrieval_args = default_retrieval_args("new")
    else:
        retrieval_args = {
            "top_k_seeds": args.top_k_seeds, "max_depth": args.max_depth, "beam_width": args.beam_width,
            "final_top_k": args.final_top_k, "goal_lookahead": args.goal_lookahead, "mode": args.mode,
            "use_island_and_bridge": args.use_island_and_bridge, "island_pool_size": 75,
            "island_max_bridge_hops": 1, "deduplicate_paths": True, "neutral_layout": True,
        }

    configure_dspy(max_tokens=args.judge_max_tokens)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_model)

    print(f"[*] Loading models (retrieval={args.retrieval_model}, cross_encoder={args.cross_encoder_model})...")
    bi_encoder = SentenceTransformer(args.retrieval_model, trust_remote_code=True)
    cross_encoder = CrossEncoder(args.cross_encoder_model)

    kg_path = os.path.join(construction_dir, KG_FILENAME)
    print(f"[*] Loading KG from {kg_path}...")
    retriever = GoalSearchEngine(retrieval_model=bi_encoder, cross_encoder=cross_encoder, graph_path=kg_path)

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)

    for i, pair in enumerate(selected_pairs):
        print(f"\n[*] Tracing question {i + 1}/{len(selected_pairs)}: {pair['question'][:70]!r}")
        record = trace_one_question(
            pair, retriever, retrieval_args, tokenizer, args.max_context_tokens, retriever.graph,
        )
        report_text = format_report(record, max_candidates_per_hop=args.max_candidates_per_hop)
        print(report_text)

        if args.out_dir:
            stem = f"trace_{i:03d}_{datetime.now().strftime('%H%M%S')}"
            with open(os.path.join(args.out_dir, f"{stem}.txt"), "w", encoding="utf-8") as f:
                f.write(report_text)
            with open(os.path.join(args.out_dir, f"{stem}.json"), "w", encoding="utf-8") as f:
                json.dump(record, f, indent=2, default=str)
            print(f"\n[*] Saved report -> {args.out_dir}/{stem}.{{txt,json}}")


if __name__ == "__main__":
    main()
