#!/usr/bin/env python3
"""
kggen_multihop_eval.py
======================

Single-script KGGen evaluation harness for HotpotQA and MuSiQue benchmarks.
Two modes — selected by --mode:

  construction  : builds a global knowledge graph over the corpus.jsonl
                  produced by prepare_{hotpotqa,musique}_corpus.py.
                  Auto-resumes from the latest checkpoint inside --run-dir,
                  or starts fresh if the directory is empty.

  evaluation    : loads the run_dir/final-checkpoint state and answers every
                  question in questions.jsonl, scoring with the *official*
                  HotpotQA / MuSiQue EM and F1 functions.


-------------------------------------------------------------------------------
EXAMPLES
-------------------------------------------------------------------------------
# Construction (resumable). First call starts fresh; subsequent calls resume
# from the latest checkpoint inside the run directory automatically.
python kggen_multihop_eval.py \\
    --mode construction \\
    --dataset hotpotqa \\
    --corpus-file ./data/hotpot_corpus_1k.jsonl \\
    --run-dir ./results/hotpotqa/kggen/run1 \\
    --checkpoint-interval 100 \\
    --batch-size 5 \\
    --max-workers 10

# Same idea for MuSiQue:
python kggen_multihop_eval.py \\
    --mode construction \\
    --dataset musique \\
    --corpus-file ./data/musique_corpus_1k.jsonl \\
    --run-dir ./results/musique/kggen/run1 \\
    --checkpoint-interval 100

# Evaluation (only works after final-checkpoint has been produced):
python kggen_multihop_eval.py \\
    --mode evaluation \\
    --dataset hotpotqa \\
    --questions-file ./data/hotpot_questions_1k.jsonl \\
    --run-dir ./results/hotpotqa/kggen/run1 \\
    --retrieval-k 10 \\
    --retrieval-depth 2 \\
    --max-context-tokens 2000


-------------------------------------------------------------------------------
DIRECTORY LAYOUT
-------------------------------------------------------------------------------
results/hotpotqa/kggen/run1/
    checkpoint-1/
        metadata.json
        absorbed_paragraph_ids.json
        graph.json              <- cumulative aggregated KG after first N paragraphs
    checkpoint-2/
        metadata.json
        absorbed_paragraph_ids.json
        graph.json              <- cumulative aggregated KG after 2N paragraphs
    final-checkpoint/
        metadata.json
        absorbed_paragraph_ids.json
        graph.json              <- final aggregated KG (all paragraphs)
    evaluation_results.json     <- created in --mode evaluation (outside checkpoint dirs)
"""

import os
import sys
import json
import time
import uuid
import argparse
import traceback
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import litellm
from transformers import AutoTokenizer

# ---- Project imports (adjust path to your project layout if needed) ----
# The CustomKGGen wrapper from wrappers_1.py needs to be importable.
# Either drop wrappers_1.py next to this script, or adjust the import below.
from .wrappers import CustomKGGen        # fallback for package-style imports

from experiments.utils.multihop_utils import (
    HOTPOTQA, MUSIQUE, SUPPORTED_DATASETS,
    load_corpus_jsonl, load_questions_jsonl,
    find_all_checkpoints, find_latest_checkpoint, find_final_checkpoint,
    next_checkpoint_num, save_checkpoint, load_checkpoint,
    FINAL_CHECKPOINT_NAME,
    GlobalTokenAccumulator, attach_token_tracker, detach_token_tracker,
    add_token_counters,
    call_llm_for_answer, score_em, score_f1, truncate_tokens, safe_div,
)


# =============================================================================
# CONSTRUCTION
# =============================================================================
def run_construction(args):
    """
    Builds a global KGGen graph over corpus.jsonl, checkpointing every
    args.checkpoint_interval paragraphs.

    Each checkpoint-N/ contains:
        metadata.json               — cumulative construction stats
        absorbed_paragraph_ids.json — all paragraph IDs processed so far
        graph.json                  — CUMULATIVE aggregated graph at this point

    Resume logic: the latest checkpoint's graph.json is loaded and reconstructed
    as a KGGen Graph object, then used as the "base" in the next aggregate() call
    alongside the new block's subgraphs. This means aggregate() runs once per
    block (not just once at the end), but each checkpoint is a fully usable graph.

    Token / time tracking is cumulative across blocks; each new checkpoint's
    metadata.json carries the running totals — previous checkpoint JSONs are
    never modified.
    """
    os.makedirs(args.run_dir, exist_ok=True)

    # ---- 0. Refuse if already done ----
    if find_final_checkpoint(args.run_dir):
        print(f"[!] final-checkpoint already exists at {args.run_dir}")
        print(f"[!] Construction is complete. Delete final-checkpoint/ to rebuild.")
        return

    # ---- 1. Load corpus ----
    print(f"[Construction] Loading corpus: {args.corpus_file}")
    corpus = load_corpus_jsonl(args.corpus_file)
    print(f"[Construction]   {len(corpus):,} paragraphs in corpus")

    # ---- 2. Determine resume state ----
    latest = find_latest_checkpoint(args.run_dir)
    if latest:
        print(f"[Construction] Resuming from: {latest}")
        prev_metadata, absorbed_ids, base_graph = load_checkpoint(latest)
        cumulative_tokens = prev_metadata.get(
            "cumulative_construction_tokens",
            {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "api_calls": 0},
        )
        cumulative_time  = float(prev_metadata.get("cumulative_construction_time_sec", 0.0))
        print(f"[Construction]   {len(absorbed_ids):,} paragraphs already absorbed, "
              f"base graph has {len(base_graph.entities):,} entities")
    else:
        print(f"[Construction] No checkpoint found — starting fresh.")
        absorbed_ids     = []
        base_graph       = None
        cumulative_tokens = {"prompt_tokens": 0, "completion_tokens": 0,
                             "total_tokens": 0, "api_calls": 0}
        cumulative_time  = 0.0

    absorbed_set = set(absorbed_ids)

    # ---- 3. Filter remaining paragraphs (sorted by ID for determinism) ----
    remaining = [p for pid, p in sorted(corpus.items()) if pid not in absorbed_set]
    if not remaining:
        print("[Construction] All paragraphs already absorbed.")
    else:
        print(f"[Construction] {len(remaining):,} paragraphs remaining "
              f"({len(absorbed_set):,} already absorbed of {len(corpus):,} total)")

    # ---- 4. Initialize KGGen ----
    os.environ["OPENAI_API_BASE"] = args.api_base
    os.environ["OPENAI_API_KEY"]  = args.api_key
    kg = CustomKGGen(
        model=f"openai/{args.model}",
        api_key=args.api_key,
        api_base=args.api_base,
        retrieval_model=args.embed_model,
    )

    # ---- 5. Process in checkpoint-sized blocks ----
    def _generate_one(para: dict):
        pid, title, text = para["id"], para["title"], para["full_context"]
        try:
            g = kg.generate(
                input_data=text,
                context=f"This is a Wikipedia paragraph about: {title}",
            )
            return pid, g
        except Exception as e:
            print(f"[Construction WARN] {pid} ({title[:40]}) → {e}")
            return pid, None

    checkpoint_num = next_checkpoint_num(args.run_dir)

    while remaining:
        block     = remaining[:args.checkpoint_interval]
        remaining = remaining[args.checkpoint_interval:]

        print(f"\n[Construction] --- checkpoint-{checkpoint_num} block "
              f"({len(block):,} paragraphs) ---")
        block_t0 = time.time()

        block_tracker = GlobalTokenAccumulator()
        attach_token_tracker(block_tracker)

        new_subgraphs  = []
        new_absorbed   = []
        try:
            with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                future_map = {executor.submit(_generate_one, p): p["id"] for p in block}
                done = 0
                for future in as_completed(future_map):
                    pid, g = future.result()
                    done += 1
                    if g is not None:
                        new_subgraphs.append(g)
                        new_absorbed.append(pid)
                    if done % max(1, len(block) // 5) == 0 or done == len(block):
                        print(f"[Construction]   {done}/{len(block)} done")
        finally:
            detach_token_tracker()

        # ---- Aggregate: base_graph (previous checkpoint) + new subgraphs ----
        graphs_to_merge = (([base_graph] if base_graph is not None else [])
                           + new_subgraphs)
        if not graphs_to_merge:
            print(f"[Construction WARN] No graphs to aggregate for block — skipping.")
            checkpoint_num += 1
            continue

        print(f"[Construction] Aggregating {len(graphs_to_merge)} graphs "
              f"({1 if base_graph else 0} base + {len(new_subgraphs)} new)...")
        agg_t0      = time.time()
        agg_tracker = GlobalTokenAccumulator()
        attach_token_tracker(agg_tracker)
        try:
            aggregated = kg.aggregate(graphs_to_merge)
        finally:
            detach_token_tracker()
        agg_time   = round(time.time() - agg_t0, 2)
        agg_tokens = agg_tracker.snapshot()

        block_time   = round(time.time() - block_t0, 2)
        block_tokens = add_token_counters(block_tracker.snapshot(), agg_tokens)

        absorbed_ids.extend(new_absorbed)
        cumulative_tokens = add_token_counters(cumulative_tokens, block_tokens)
        cumulative_time   = round(cumulative_time + block_time, 2)

        post_entities = len(aggregated.entities)
        post_edges    = len(getattr(aggregated, "edges",
                            getattr(aggregated, "relations",
                            getattr(aggregated, "triples", []))))

        metadata = {
            "checkpoint_id":                       f"checkpoint-{checkpoint_num}",
            "checkpoint_num":                      checkpoint_num,
            "timestamp":                           datetime.now().isoformat(),
            "dataset":                             args.dataset,
            "model":                               args.model,
            "embedding_model":                     args.embed_model,
            "api_base":                            args.api_base,
            "corpus_file":                         os.path.abspath(args.corpus_file),
            "total_paragraphs_in_corpus":          len(corpus),
            "paragraphs_absorbed_so_far":          len(absorbed_ids),
            "paragraphs_in_this_block":            len(new_absorbed),
            "paragraphs_failed_in_block":          len(block) - len(new_absorbed),
            "cumulative_graph_entities":           post_entities,
            "cumulative_graph_edges":              post_edges,
            "cumulative_construction_tokens":      cumulative_tokens,
            "this_block_tokens":                   block_tokens,
            "cumulative_construction_time_sec":    cumulative_time,
            "this_block_time_sec":                 block_time,
            "construction_params": {
                "checkpoint_interval": args.checkpoint_interval,
                "max_workers":         args.max_workers,
            },
        }
        cp_path = save_checkpoint(
            run_dir=args.run_dir,
            checkpoint_num=checkpoint_num,
            metadata=metadata,
            absorbed_paragraph_ids=absorbed_ids,
            aggregated_graph=aggregated,
            kg_instance=kg,
            final=False,
        )
        print(f"[Construction] checkpoint-{checkpoint_num} saved → {cp_path}")
        print(f"[Construction]   {len(absorbed_ids):,}/{len(corpus):,} paras absorbed | "
              f"{post_entities:,} entities | "
              f"{cumulative_tokens['total_tokens']:,} cumulative tokens | "
              f"{cumulative_time}s cumulative time")

        base_graph     = aggregated   # carry forward for next block
        checkpoint_num += 1

    # ---- 6. Write final-checkpoint (no additional aggregation needed) ----
    # The last checkpoint-N already has the fully aggregated graph. We copy its
    # state into final-checkpoint and flag it as complete.
    if base_graph is None:
        print("[Construction FATAL] No graph was produced; cannot write final-checkpoint.")
        return

    # Read back the last checkpoint's metadata to carry into final-checkpoint
    latest = find_latest_checkpoint(args.run_dir)
    last_meta, last_absorbed, _ = load_checkpoint(latest)

    final_metadata = {
        **last_meta,
        "checkpoint_id":  FINAL_CHECKPOINT_NAME,
        "checkpoint_num": None,
        "timestamp":      datetime.now().isoformat(),
        "is_final":       True,
        "note": (
            "Node embeddings are NOT persisted here. They are recomputed at "
            "evaluation time from graph.json using kg.generate_embeddings(). "
            "This avoids large binary files and keeps the checkpoint directory "
            "fully human-readable."
        ),
    }
    final_cp = save_checkpoint(
        run_dir=args.run_dir,
        checkpoint_num=0,
        metadata=final_metadata,
        absorbed_paragraph_ids=last_absorbed,
        aggregated_graph=base_graph,
        kg_instance=kg,
        final=True,
    )
    print(f"\n[Construction] FINAL CHECKPOINT → {final_cp}")
    print(f"[Construction]   graph.json contains {len(base_graph.entities):,} entities")
    print(f"[Construction]   total tokens: {cumulative_tokens['total_tokens']:,}")
    print(f"[Construction]   total wall time: {cumulative_time}s")


# =============================================================================
# EVALUATION
# =============================================================================
def _eval_one_question(
    q: dict,
    kg,
    nx_graph,
    node_embeddings,
    tokenizer,
    args,
) -> dict:
    """
    Retrieves context for one question and asks the LLM to answer it.
    Returns a per-question result dict including EM/F1 against the gold answer.
    """
    question     = q["question"]
    gold         = q["answer"]
    aliases      = q.get("answer_aliases", [])      # MuSiQue only
    qtype        = q.get("type", "unknown")          # bridge/comparison or 2hop/3hop/4hop
    qlevel       = q.get("level", None)             # easy/medium/hard for HotpotQA
    qid          = q.get("question_id", "")

    llm_answer   = ""
    em           = 0.0
    f1           = 0.0
    precision    = 0.0
    recall       = 0.0
    raw_context  = ""
    trunc_ctx    = ""
    ctx_tokens   = 0
    status       = "SUCCESS"
    error_msg    = ""

    try:
        # 1. Retrieve KG-based context
        _, _, raw_context = kg.retrieve(
            query=question,
            node_embeddings=node_embeddings,
            graph=nx_graph,
            k=args.retrieval_k,
            depth=args.retrieval_depth,
        )

        # 2. Truncate to max_context_tokens
        trunc_ctx, ctx_tokens = truncate_tokens(
            raw_context, tokenizer, args.max_context_tokens
        )

        # 3. Generate answer
        llm_answer = call_llm_for_answer(
            model=args.model,
            api_base=args.api_base,
            api_key=args.api_key,
            question=question,
            context=trunc_ctx,
            max_tokens=args.answer_max_tokens,
            temperature=0.0,
        )

        # 4. Score
        em = score_em(llm_answer, gold, aliases=aliases, dataset=args.dataset)
        f1, precision, recall = score_f1(llm_answer, gold, aliases=aliases, dataset=args.dataset)

    except Exception as e:
        status    = "ERROR"
        error_msg = str(e)
        print(f"[Eval ERROR] {qid}: {e}")

    return {
        "question_id":              qid,
        "question":                 question,
        "expected_answer":          gold,
        "expected_answer_aliases":  aliases,
        "type":                     qtype,
        "level":                    qlevel,
        "llm_answer":               llm_answer,
        "em":                       float(em),
        "f1":                       float(f1),
        "precision":                float(precision),
        "recall":                   float(recall),
        "retrieved_context":        trunc_ctx,
        "raw_context_length_chars": len(raw_context),
        "context_tokens":           ctx_tokens,
        "max_context_tokens":       args.max_context_tokens,
        "status":                   status,
        **({"error": error_msg} if error_msg else {}),
    }


def _aggregate_by_key(results: list, key: str) -> dict:
    """Group results by `key` and compute mean EM/F1 + count per group."""
    groups = defaultdict(list)
    for r in results:
        if r["status"] != "SUCCESS":
            continue
        k = r.get(key)
        if k is None:
            continue
        groups[k].append(r)
    out = {}
    for k, rs in groups.items():
        n = len(rs)
        out[str(k)] = {
            "count":     n,
            "em":        safe_div(sum(r["em"] for r in rs), n),
            "f1":        safe_div(sum(r["f1"] for r in rs), n),
            "precision": safe_div(sum(r["precision"] for r in rs), n),
            "recall":    safe_div(sum(r["recall"] for r in rs), n),
        }
    return out


def run_evaluation(args):
    """
    Loads the final-checkpoint from args.run_dir, runs answer generation for
    every question in args.questions_file, scores EM/F1 with the official
    dataset metrics, and writes evaluation_results.json at the run_dir root.

    Node embeddings are recomputed from the graph.json in final-checkpoint
    rather than loaded from disk. This keeps the checkpoint directory fully
    human-readable JSON with no binary blobs, and the recomputation is fast
    (sentence-transformer forward pass only — no LLM calls).
    """
    final_cp = find_final_checkpoint(args.run_dir)
    if not final_cp:
        print(f"[Evaluation FATAL] No final-checkpoint at {args.run_dir}. "
              f"Run --mode construction first.")
        sys.exit(1)

    print(f"[Evaluation] Loading final checkpoint: {final_cp}")
    final_metadata, absorbed_ids, combined_graph = load_checkpoint(final_cp)
    print(f"[Evaluation]   graph.json loaded: "
          f"{len(combined_graph.entities):,} entities, "
          f"{len(absorbed_ids):,} paragraphs were absorbed")

    # ---- Rehydrate KGGen for retrieve() and embedding generation ----
    os.environ["OPENAI_API_BASE"] = args.api_base
    os.environ["OPENAI_API_KEY"]  = args.api_key
    kg = CustomKGGen(
        model=f"openai/{args.model}",
        api_key=args.api_key,
        api_base=args.api_base,
        retrieval_model=args.embed_model,
    )
    nx_graph = kg.to_nx(combined_graph)

    print(f"[Evaluation] Computing node embeddings from graph.json "
          f"(sentence-transformer only, no LLM calls)...")
    emb_t0 = time.time()
    node_embeddings, _ = kg.generate_embeddings(combined_graph)
    emb_time = round(time.time() - emb_t0, 2)
    print(f"[Evaluation]   {len(node_embeddings):,} node embeddings in {emb_time}s")

    # ---- Load questions ----
    print(f"[Evaluation] Loading questions: {args.questions_file}")
    questions = load_questions_jsonl(args.questions_file)
    print(f"[Evaluation]   {len(questions):,} questions to evaluate")

    # ---- Tokenizer for context truncation ----
    print(f"[Evaluation] Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # ---- Run in parallel — token tracker sees only answer-gen calls ----
    eval_tracker = GlobalTokenAccumulator()
    attach_token_tracker(eval_tracker)

    eval_t0 = time.time()
    results  = []
    try:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            future_map = {
                executor.submit(_eval_one_question, q, kg, nx_graph,
                                node_embeddings, tokenizer, args): i
                for i, q in enumerate(questions)
            }
            done = 0
            for future in as_completed(future_map):
                r = future.result()
                results.append(r)
                done += 1
                if done % max(1, len(questions) // 20) == 0 or done == len(questions):
                    n_ok = sum(1 for x in results if x["status"] == "SUCCESS")
                    running_em = safe_div(sum(x["em"] for x in results if x["status"] == "SUCCESS"), n_ok)
                    running_f1 = safe_div(sum(x["f1"] for x in results if x["status"] == "SUCCESS"), n_ok)
                    print(f"[Evaluation] {done}/{len(questions)}  "
                          f"running EM={running_em:.4f} F1={running_f1:.4f}")
    finally:
        detach_token_tracker()

    eval_time   = round(time.time() - eval_t0, 2)
    eval_tokens = eval_tracker.snapshot()

    # ---- Aggregate metrics ----
    successes = [r for r in results if r["status"] == "SUCCESS"]
    errors    = [r for r in results if r["status"] != "SUCCESS"]
    n = len(successes)

    overall = {
        "em":        safe_div(sum(r["em"]        for r in successes), n),
        "f1":        safe_div(sum(r["f1"]        for r in successes), n),
        "precision": safe_div(sum(r["precision"] for r in successes), n),
        "recall":    safe_div(sum(r["recall"]    for r in successes), n),
    }
    by_type  = _aggregate_by_key(successes, "type")
    by_level = _aggregate_by_key(successes, "level") if args.dataset == HOTPOTQA else None

    # ---- Build final output JSON ----
    eval_output = {
        "run_id":    f"kggen_eval_{uuid.uuid4().hex[:8]}",
        "timestamp": datetime.now().isoformat(),
        "system":    "KGGEN",
        "dataset":   args.dataset,

        "metadata": {
            "corpus_file":           final_metadata.get("corpus_file"),
            "questions_file":        os.path.abspath(args.questions_file),
            "run_dir":               os.path.abspath(args.run_dir),
            "final_checkpoint_path": os.path.abspath(final_cp),
            "model":                 args.model,
            "embedding_model":       args.embed_model,
            "api_base":              args.api_base,
            "retrieval_params":      {"k": args.retrieval_k, "depth": args.retrieval_depth},
            "max_context_tokens":    args.max_context_tokens,
            "answer_max_tokens":     args.answer_max_tokens,
            "max_workers":           args.max_workers,
            "embedding_recompute_time_sec": emb_time,
        },

        "construction": final_metadata,

        "evaluation": {
            "total_questions":     len(questions),
            "successful":          len(successes),
            "errored":             len(errors),
            "evaluation_time_sec": eval_time,
            "evaluation_tokens":   eval_tokens,
            "overall":             overall,
            "by_type":             by_type,
            **({"by_level": by_level} if by_level is not None else {}),
            "results":             results,
        },

        "totals": {
            "all_tokens": add_token_counters(
                final_metadata.get("cumulative_construction_tokens", {}),
                eval_tokens,
            ),
            "all_time_sec": round(
                float(final_metadata.get("cumulative_construction_time_sec", 0.0))
                + eval_time, 2
            ),
        },
    }

    out_path = os.path.join(args.run_dir, "evaluation_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(eval_output, f, indent=2, default=str)

    # ---- Terminal summary ----
    print(f"\n[Evaluation] DONE in {eval_time}s")
    print(f"[Evaluation]   wrote → {out_path}")
    print(f"[Evaluation]   overall EM={overall['em']}  F1={overall['f1']}  "
          f"(n={n}, errors={len(errors)})")
    print(f"[Evaluation]   by type:")
    for k, v in sorted(by_type.items()):
        print(f"     {k:>12}: EM={v['em']:.4f}  F1={v['f1']:.4f}  (n={v['count']})")
    if by_level:
        print(f"[Evaluation]   by level:")
        for k, v in sorted(by_level.items()):
            print(f"     {k:>12}: EM={v['em']:.4f}  F1={v['f1']:.4f}  (n={v['count']})")
    print(f"[Evaluation]   eval tokens: {eval_tokens['total_tokens']:,}  "
          f"({eval_tokens['api_calls']:,} calls)")


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="KGGen evaluation on HotpotQA / MuSiQue — single script"
    )

    # ---- Mode & dataset ----
    parser.add_argument("--mode", required=True, choices=["construction", "evaluation"],
                        help="construction: build the KG (resumable). "
                             "evaluation: run QA on a built KG.")
    parser.add_argument("--dataset", required=True, choices=list(SUPPORTED_DATASETS),
                        help="Which benchmark we're evaluating on. Controls "
                             "scoring functions (HotpotQA yes/no rule vs MuSiQue aliases) "
                             "and the stratification categories.")

    # ---- File inputs ----
    parser.add_argument("--corpus-file",    type=str,
                        help="Path to corpus.jsonl (construction mode).")
    parser.add_argument("--questions-file", type=str,
                        help="Path to questions.jsonl (evaluation mode).")
    parser.add_argument("--run-dir",        type=str, required=True,
                        help="Directory holding checkpoints. "
                             "Construction resumes from latest checkpoint inside this dir, "
                             "or starts fresh if empty. Evaluation reads final-checkpoint here.")

    # ---- Models & API ----
    parser.add_argument("--model",       type=str,
                        default="Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
                        help="LLM model name (used for both KG extraction and answer generation).")
    parser.add_argument("--embed-model", type=str,
                        default="google/embeddinggemma-300m",
                        help="Sentence-transformer model for node embeddings.")
    parser.add_argument("--api-base",    type=str, default="http://localhost:8000/v1")
    parser.add_argument("--api-key",     type=str, default="EMPTY")

    # ---- Construction-only ----
    parser.add_argument("--checkpoint-interval", type=int, default=100,
                        help="Paragraphs absorbed between consecutive checkpoints.")
    parser.add_argument("--batch-size",   type=int, default=5,
                        help="(legacy; max_workers controls parallelism in this script)")

    # ---- Evaluation-only ----
    parser.add_argument("--retrieval-k",        type=int, default=10,
                        help="Top-K seed nodes for KGGen retrieval.")
    parser.add_argument("--retrieval-depth",    type=int, default=2,
                        help="BFS depth for KGGen context expansion.")
    parser.add_argument("--max-context-tokens", type=int, default=2000,
                        help="Max tokens of retrieved context fed to the LLM per question.")
    parser.add_argument("--answer-max-tokens",  type=int, default=128,
                        help="max_tokens for the answer-generation LLM call. "
                             "HotpotQA/MuSiQue answers are short — 128 is plenty.")

    # ---- Shared ----
    parser.add_argument("--max-workers", type=int, default=10,
                        help="Thread-pool size for parallel paragraph extraction "
                             "(construction) or parallel question answering (evaluation).")

    args = parser.parse_args()

    # ---- Validate ----
    if args.mode == "construction" and not args.corpus_file:
        parser.error("--mode construction requires --corpus-file")
    if args.mode == "evaluation" and not args.questions_file:
        parser.error("--mode evaluation requires --questions-file")

    # ---- Dispatch ----
    print(f"\n{'=' * 65}")
    print(f"  KGGEN {args.mode.upper()} — {args.dataset.upper()}")
    print(f"  run-dir: {args.run_dir}")
    print(f"  model:   {args.model}")
    print(f"{'=' * 65}\n")

    if args.mode == "construction":
        try:
            run_construction(args)
        except KeyboardInterrupt:
            print("\n[Construction] Interrupted by user. "
                  "Latest checkpoint is preserved — rerun the same command to resume.")
        except Exception as e:
            print(f"\n[Construction FATAL] {e}")
            traceback.print_exc()
            sys.exit(1)
    else:  # evaluation
        run_evaluation(args)


if __name__ == "__main__":
    main()