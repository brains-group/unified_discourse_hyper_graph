#!/usr/bin/env python3
"""
KGGen Insurance Contract Evaluation — Single Script
Runs graph construction followed by QA and/or Fact evaluation.

Single contract:
    python kggen_eval.py \
        --contract path/to/contract.txt \
        --qa-file path/to/qa.json \
        --fact-file path/to/facts.json \
        --out-dir ./results

All contracts in both dataset sections:
    python kggen_eval.py \
        --all \
        --eval-data-dir ./evaluation_data \
        --out-dir ./results

Only ten_contracts_dataset:
    python kggen_eval.py \
        --ten-contracts-dataset \
        --eval-data-dir ./evaluation_data \
        --out-dir ./results

Only long_form_contracts:
    python kggen_eval.py \
        --long-form-contracts \
        --eval-data-dir ./evaluation_data \
        --out-dir ./results

All other flags (model, chunking, retrieval, etc.) apply in every mode.
"""

import os
import sys
import json
import time
import uuid
import argparse
import threading
import traceback
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import litellm
from litellm.integrations.custom_logger import CustomLogger
from transformers import AutoTokenizer

# ---------- Project imports ----------
from experiments.evaluate_mine import ask_llm_query, score_llm_answer, evaluate_fact
from experiments.utils.chunking import truncate_context
from nkg.utils.config import configure_dspy
from nkg.utils.chunking import chunk_text_by_tokens
from .wrappers import CustomKGGen


# =============================================================================
# GLOBAL TOKEN ACCUMULATOR
# Re-declared here (not imported) because it hooks into LiteLLM callbacks
# specifically during construction and is not a general shared utility.
# =============================================================================
class GlobalTokenAccumulator(CustomLogger):
    """Thread-safe LiteLLM hook that sums token usage across all API calls."""

    def __init__(self):
        super().__init__()
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.api_calls = 0
        self._lock = threading.Lock()

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        usage = (
            response_obj.get("usage")
            if isinstance(response_obj, dict)
            else getattr(response_obj, "usage", None)
        )
        if usage is None:
            return
        if isinstance(usage, dict):
            p = usage.get("prompt_tokens", 0)
            c = usage.get("completion_tokens", 0)
            t = usage.get("total_tokens", p + c)
        else:
            p = getattr(usage, "prompt_tokens", 0)
            c = getattr(usage, "completion_tokens", 0)
            t = getattr(usage, "total_tokens", p + c)
        with self._lock:
            self.prompt_tokens += p
            self.completion_tokens += c
            self.total_tokens += t
            self.api_calls += 1

    def snapshot(self):
        with self._lock:
            return {
                "llm_prompt_tokens": self.prompt_tokens,
                "llm_completion_tokens": self.completion_tokens,
                "llm_total_tokens": self.total_tokens,
                "llm_api_calls": self.api_calls,
            }


# =============================================================================
# DATASET PATH RESOLUTION
# Handles the two different naming conventions across the two dataset folders:
#
#   ten_contracts_dataset/
#     contracts/  contract_1_term_life.txt
#     facts/      facts_contract_1_term_life.json   <- prefixed with "facts_"
#     qa_pairs/   qa_contract_1_term_life.json       <- prefixed with "qa_"
#
#   long_form_contracts/
#     contracts/  td_contract.txt
#     facts/      td_contract.json                  <- same name as contract
#     qa_pairs/   qa_td_contract.json               <- prefixed with "qa_"
# =============================================================================
DATASET_SUBDIRS = ["ten_contracts_dataset", "long_form_contracts"]


def resolve_paths_for_contract(contract_path: str, dataset_root: str):
    """
    Given a contract .txt path that lives somewhere under dataset_root,
    locate the matching qa_pairs and facts JSON files.

    Returns (qa_path, fact_path) or raises FileNotFoundError.
    """
    base     = os.path.splitext(os.path.basename(contract_path))[0]
    # Walk up to find which dataset sub-directory this contract lives in
    section_dir = None
    for subdir in DATASET_SUBDIRS:
        candidate = os.path.join(dataset_root, subdir)
        if contract_path.startswith(os.path.abspath(candidate)):
            section_dir = candidate
            break

    if section_dir is None:
        raise ValueError(
            f"Contract {contract_path} does not appear to be under any known "
            f"dataset sub-directory inside {dataset_root}."
        )

    qa_path   = os.path.join(section_dir, "qa_pairs", f"qa_{base}.json")
    # Facts may be stored with or without a "facts_" prefix
    fact_path = os.path.join(section_dir, "facts", f"facts_{base}.json")
    if not os.path.exists(fact_path):
        fact_path = os.path.join(section_dir, "facts", f"{base}.json")

    if not os.path.exists(qa_path):
        raise FileNotFoundError(f"Missing QA file:   {qa_path}")
    if not os.path.exists(fact_path):
        raise FileNotFoundError(f"Missing Fact file: {fact_path}")

    return qa_path, fact_path


def collect_all_contracts(eval_data_dir: str, subdirs=None):
    """
    Scans the given dataset sub-directories (defaults to all) and returns a
    list of (contract_path, qa_path, fact_path) tuples for every .txt found.
    """
    if subdirs is None:
        subdirs = DATASET_SUBDIRS
    triplets = []
    for subdir in subdirs:
        contracts_dir = os.path.join(eval_data_dir, subdir, "contracts")
        if not os.path.isdir(contracts_dir):
            print(f"[WARN] Contracts directory not found: {contracts_dir} — skipping.")
            continue
        for fname in sorted(os.listdir(contracts_dir)):
            if not fname.endswith(".txt"):
                continue
            if fname in ["2025_pdp_contract.txt", "all_state_auto_insurance.txt"]:
                continue
            contract_path = os.path.abspath(os.path.join(contracts_dir, fname))
            try:
                qa_path, fact_path = resolve_paths_for_contract(
                    contract_path, os.path.abspath(eval_data_dir)
                )
                triplets.append((contract_path, qa_path, fact_path))
            except FileNotFoundError as e:
                print(f"[WARN] Skipping {fname}: {e}")
    return triplets


# =============================================================================
# GRAPH CONSTRUCTION
# =============================================================================
def build_graph(
    file_path: str,
    model: str,
    retrieval_model: str,
    api_base: str,
    api_key: str,
    chunk_size: int,
    chunk_overlap: int,
    batch_size: int,
    max_workers: int,
):
    """
    Reads the contract, chunks it, generates KGGen sub-graphs in parallel,
    aggregates, and pre-computes node embeddings.

    Returns:
        kg               — CustomKGGen instance (reused for retrieval)
        combined_graph   — aggregated KGGen Graph object
        nx_graph         — NetworkX DiGraph
        node_embeddings  — pre-computed embedding dict
        metrics          — dict with construction statistics
    """
    token_tracker = GlobalTokenAccumulator()
    litellm.callbacks       = [token_tracker]
    litellm.success_callback = [token_tracker]
    os.environ["OPENAI_API_BASE"] = api_base
    os.environ["OPENAI_API_KEY"]  = api_key

    t0 = time.time()

    kg = CustomKGGen(
        model=f"openai/{model}",
        api_key=api_key,
        api_base=api_base,
        retrieval_model=retrieval_model,
    )

    print(f"[Construction] Reading: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        full_text = f.read()

    try:
        total_document_tokens = litellm.token_counter(
            model=f"openai/{model}", text=full_text
        )
    except Exception:
        total_document_tokens = len(full_text.split())
        print("[WARN] Precise token count unavailable; using word count as proxy.")

    print(f"[Construction] Document tokens: {total_document_tokens}")

    chunks       = chunk_text_by_tokens(
        model=model, text=full_text, chunk_size=chunk_size, overlap=chunk_overlap
    )
    total_chunks = len(chunks)
    print(
        f"[Construction] Chunks: {total_chunks} "
        f"| Batch size: {batch_size} | Workers: {max_workers}"
    )

    batches = [chunks[i : i + batch_size] for i in range(0, total_chunks, batch_size)]

    def process_batch(batch_texts):
        results = []
        for text in batch_texts:
            try:
                g = kg.generate(input_data=text, context="This is an insurance contract.")
                if g is not None:
                    results.append(g)
            except Exception as e:
                print(f"[WARN] Chunk generation error: {e}")
        return results

    giant_graph_list = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(process_batch, b): i for i, b in enumerate(batches)
        }
        for future in as_completed(future_map):
            bi = future_map[future]
            try:
                sub = future.result()
                giant_graph_list.extend(sub)
                print(
                    f"[Construction] Batch {bi+1}/{len(batches)} "
                    f"— {len(sub)} graphs produced"
                )
            except Exception as e:
                print(f"[Construction] Batch {bi+1} error: {e}")

    if not giant_graph_list:
        raise RuntimeError("No sub-graphs were generated. Construction failed.")

    pre_entities = sum(len(g.entities) for g in giant_graph_list)
    pre_edges    = sum(
        len(getattr(g, "edges", getattr(g, "relations", getattr(g, "triples", []))))
        for g in giant_graph_list
    )

    print(f"[Construction] Aggregating {len(giant_graph_list)} sub-graphs...")
    combined_graph = kg.aggregate(giant_graph_list)

    post_entities    = len(combined_graph.entities)
    post_edges_list  = getattr(
        combined_graph,
        "edges",
        getattr(combined_graph, "relations", getattr(combined_graph, "triples", [])),
    )
    post_edges = len(post_edges_list)

    print("[Construction] Computing node embeddings...")
    node_embeddings, _ = kg.generate_embeddings(combined_graph)
    nx_graph           = kg.to_nx(combined_graph)

    execution_time = round(time.time() - t0, 2)

    # Detach tracker before evaluation starts so its counts stay clean
    litellm.callbacks        = []
    litellm.success_callback = []

    token_snap = token_tracker.snapshot()

    metrics = {
        "total_document_tokens": total_document_tokens,
        "total_chunks":          total_chunks,
        "chunk_size":            chunk_size,
        "chunk_overlap":         chunk_overlap,
        "batch_size":            batch_size,
        "max_workers":           max_workers,
        "pre_entities":          pre_entities,
        "post_entities":         post_entities,
        "entity_dedup_ratio":    round(post_entities / pre_entities, 4) if pre_entities else 1.0,
        "pre_edges":             pre_edges,
        "post_edges":            post_edges,
        "edge_dedup_ratio":      round(post_edges / pre_edges, 4) if pre_edges else 1.0,
        **token_snap,
        "execution_time_sec":    execution_time,
    }

    print(
        f"[Construction] Done in {execution_time}s "
        f"| Entities {pre_entities}→{post_entities} "
        f"| Edges {pre_edges}→{post_edges} "
        f"| LLM tokens {token_snap['llm_total_tokens']}"
    )
    return kg, combined_graph, nx_graph, node_embeddings, metrics


# =============================================================================
# QA EVALUATION WORKER
# =============================================================================
def _eval_qa_item(
    pair, kg, node_embeddings, nx_graph,
    tokenizer, max_context_tokens, retrieval_params
):
    q        = pair["question"]
    a        = pair["answer"]
    category = pair.get("category", "UNKNOWN")
    evidence = pair.get("evidence", "")

    llm_answer        = ""
    score             = 0.0
    raw_context       = ""
    truncated_context = ""
    context_tokens    = 0
    status            = "SUCCESS"
    error_msg         = ""

    try:
        _, _, raw_context = kg.retrieve(
            query=q,
            node_embeddings=node_embeddings,
            graph=nx_graph,
            **retrieval_params,
        )
        truncated_context, context_tokens = truncate_context(
            raw_context, tokenizer, max_context_tokens
        )
        llm_answer = ask_llm_query(question=q, context=truncated_context)
        score      = score_llm_answer(
            question=q, llm_answer=llm_answer, correct_answer=a
        )
    except Exception as e:
        status    = "ERROR"
        error_msg = str(e)
        print(f"[QA ERROR] {q[:60]}... → {e}")

    return {
        "question":                 q,
        "expected_answer":          a,
        "evidence":                 evidence,
        "category":                 category,
        "llm_answer":               llm_answer,
        "score":                    float(score),
        "judge_decision":           float(score),   # 0.0 / 0.5 / 1.0
        "retrieved_context":        truncated_context,
        "raw_context_length_chars": len(raw_context),
        "context_tokens":           context_tokens,
        "max_context_tokens":       max_context_tokens,
        "status":                   status,
        **({"error": error_msg} if error_msg else {}),
    }


# =============================================================================
# FACT EVALUATION WORKER
# =============================================================================
def _eval_fact_item(
    fact_dict, kg, node_embeddings, nx_graph,
    tokenizer, max_context_tokens, retrieval_params
):
    fact    = fact_dict["fact"]
    fact_id = fact_dict.get("id")

    score             = 0
    raw_context       = ""
    truncated_context = ""
    context_tokens    = 0
    status            = "SUCCESS"
    error_msg         = ""

    try:
        _, _, raw_context = kg.retrieve(
            query=fact,
            node_embeddings=node_embeddings,
            graph=nx_graph,
            **retrieval_params,
        )
        truncated_context, context_tokens = truncate_context(
            raw_context, tokenizer, max_context_tokens
        )
        score = evaluate_fact(context=truncated_context, correct_answer=fact)
    except Exception as e:
        status    = "ERROR"
        error_msg = str(e)
        print(f"[FACT ERROR] {fact[:60]}... → {e}")

    return {
        "fact_id":                  fact_id,
        "fact":                     fact,
        "score":                    float(score),
        "judge_decision":           int(score),   # binary: 1 recalled / 0 not
        "retrieved_context":        truncated_context,
        "raw_context_length_chars": len(raw_context),
        "context_tokens":           context_tokens,
        "max_context_tokens":       max_context_tokens,
        "status":                   status,
        **({"error": error_msg} if error_msg else {}),
    }


# =============================================================================
# RUN QA EVALUATION
# =============================================================================
def run_qa_evaluation(
    qa_file, kg, node_embeddings, nx_graph,
    tokenizer, max_context_tokens, retrieval_params, max_workers
):
    print(f"\n[QA Eval] Loading: {qa_file}")
    with open(qa_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    pairs = data.get("qa_pairs", [])
    if not pairs:
        print("[QA Eval] No qa_pairs found in file.")
        return None

    print(f"[QA Eval] {len(pairs)} questions | {max_workers} workers")
    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(
                _eval_qa_item, p, kg, node_embeddings, nx_graph,
                tokenizer, max_context_tokens, retrieval_params
            ): i
            for i, p in enumerate(pairs)
        }
        done = 0
        for future in as_completed(future_map):
            res = future.result()
            results.append(res)
            done += 1
            print(
                f"[QA Eval] {done}/{len(pairs)} "
                f"cat={res['category']} score={res['score']}"
            )

    by_category = defaultdict(list)
    for r in results:
        by_category[r["category"]].append(r["score"])

    category_summary = {
        cat: {
            "avg_score": round(sum(scores) / len(scores), 4),
            "count":     len(scores),
            "scores":    scores,
        }
        for cat, scores in by_category.items()
    }

    all_scores      = [r["score"] for r in results]
    non_cat5_scores = [r["score"] for r in results if r["category"] != "CAT5_UNANSWERABLE"]
    overall_excl    = round(sum(non_cat5_scores) / len(non_cat5_scores), 4) if non_cat5_scores else 0.0
    overall_incl    = round(sum(all_scores) / len(all_scores), 4) if all_scores else 0.0

    print(f"[QA Eval] Overall avg (excl CAT5): {overall_excl}")
    for cat, s in category_summary.items():
        print(f"  {cat}: {s['avg_score']}  (n={s['count']})")

    return {
        "overall_avg_score_excl_cat5": overall_excl,
        "overall_avg_score_incl_cat5": overall_incl,
        "total_questions":             len(pairs),
        "by_category":                 category_summary,
        "results":                     results,
    }


# =============================================================================
# RUN FACT EVALUATION
# =============================================================================
def run_fact_evaluation(
    fact_file, kg, node_embeddings, nx_graph,
    tokenizer, max_context_tokens, retrieval_params, max_workers
):
    print(f"\n[Fact Eval] Loading: {fact_file}")
    with open(fact_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    facts = data.get("facts", [])
    if not facts:
        print("[Fact Eval] No facts found in file.")
        return None

    print(f"[Fact Eval] {len(facts)} facts | {max_workers} workers")
    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(
                _eval_fact_item, fd, kg, node_embeddings, nx_graph,
                tokenizer, max_context_tokens, retrieval_params
            ): i
            for i, fd in enumerate(facts)
        }
        done = 0
        for future in as_completed(future_map):
            res = future.result()
            results.append(res)
            done += 1
            if done % 10 == 0 or done == len(facts):
                running_avg = round(sum(r["score"] for r in results) / len(results), 4)
                print(f"[Fact Eval] {done}/{len(facts)} running recall: {running_avg}")

    all_scores     = [r["score"] for r in results]
    overall_avg    = round(sum(all_scores) / len(all_scores), 4) if all_scores else 0.0
    total_recalled = sum(1 for s in all_scores if s >= 1.0)

    print(
        f"[Fact Eval] Recall: {overall_avg} "
        f"({total_recalled}/{len(facts)} facts recalled)"
    )

    return {
        "overall_recall_score": overall_avg,
        "total_facts":          len(facts),
        "total_recalled":       total_recalled,
        "results":              results,
    }


# =============================================================================
# SINGLE CONTRACT PIPELINE  (construction + evaluation for one contract)
# =============================================================================
def run_single_contract(
    contract_path: str,
    qa_file,            # str or None
    fact_file,          # str or None
    out_dir: str,
    model: str,
    embed_model: str,
    api_base: str,
    api_key: str,
    chunk_size: int,
    chunk_overlap: int,
    batch_size: int,
    max_workers: int,
    retrieval_k: int,
    retrieval_depth: int,
    max_context_tokens: int,
    judge_max_tokens: int,
    tokenizer,          # pre-loaded AutoTokenizer, shared across contracts
):
    """
    Runs the full KGGen pipeline (construction + QA/Fact evaluation) for a
    single contract. Writes graph.json and results.json to a dedicated
    sub-directory under out_dir.
    """
    # ---- Output directory: {contract}_{qa|fact|qa_fact}_{timestamp} ----
    contract_name    = os.path.splitext(os.path.basename(contract_path))[0]
    eval_tag         = "_".join(
        (["qa"] if qa_file else []) + (["fact"] if fact_file else [])
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir   = os.path.join(out_dir, f"{contract_name}_{eval_tag}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    retrieval_params = {"k": retrieval_k, "depth": retrieval_depth}

    print("\n" + "=" * 65)
    print(f" CONTRACT: {contract_name}")
    print(f" Output  : {run_dir}")
    print("=" * 65)

    # ---- Phase 1: Construction ----
    print("\n[*] PHASE 1 — GRAPH CONSTRUCTION")
    try:
        kg, combined_graph, nx_graph, node_embeddings, construction_metrics = build_graph(
            file_path=contract_path,
            model=model,
            retrieval_model=embed_model,
            api_base=api_base,
            api_key=api_key,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            batch_size=batch_size,
            max_workers=max_workers,
        )
    except Exception as e:
        print(f"[FATAL] Construction failed for {contract_name}: {e}")
        traceback.print_exc()
        return   # Skip to the next contract instead of killing the whole run

    graph_path = os.path.join(run_dir, "graph.json")
    CustomKGGen.export_graph(combined_graph, graph_path)
    print(f"[*] Graph exported → {graph_path}")

    # ---- Phase 2: Evaluation ----
    qa_eval_result   = None
    fact_eval_result = None

    if qa_file:
        print("\n[*] PHASE 2a — QA EVALUATION")
        qa_eval_result = run_qa_evaluation(
            qa_file=qa_file,
            kg=kg,
            node_embeddings=node_embeddings,
            nx_graph=nx_graph,
            tokenizer=tokenizer,
            max_context_tokens=max_context_tokens,
            retrieval_params=retrieval_params,
            max_workers=max_workers,
        )

    if fact_file:
        print("\n[*] PHASE 2b — FACT RECALL EVALUATION")
        fact_eval_result = run_fact_evaluation(
            fact_file=fact_file,
            kg=kg,
            node_embeddings=node_embeddings,
            nx_graph=nx_graph,
            tokenizer=tokenizer,
            max_context_tokens=max_context_tokens,
            retrieval_params=retrieval_params,
            max_workers=max_workers,
        )

    # ---- Save results ----
    final_output = {
        "run_id":    f"kggen_{uuid.uuid4().hex[:8]}",
        "timestamp": timestamp,
        "system":    "KGGEN",

        "metadata": {
            "contract_file":      os.path.abspath(contract_path),
            "qa_file":            qa_file,
            "fact_file":          fact_file,
            "llm_model":          model,
            "embedding_model":    embed_model,
            "api_base":           api_base,
            "retrieval_params":   retrieval_params,
            "max_context_tokens": max_context_tokens,
            "judge_max_tokens":   judge_max_tokens,
            "chunking": {
                "chunk_size":    chunk_size,
                "chunk_overlap": chunk_overlap,
            },
            "graph_export_path":  graph_path,
        },

        "construction":    construction_metrics,
        "qa_evaluation":   qa_eval_result,
        "fact_evaluation": fact_eval_result,
    }

    results_path = os.path.join(run_dir, "results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2, default=str)

    # ---- Terminal summary ----
    cm = construction_metrics
    print(f"\n  Graph    → {graph_path}")
    print(f"  Results  → {results_path}")
    print(f"  Document tokens : {cm['total_document_tokens']}")
    print(f"  Chunks          : {cm['total_chunks']}  ({cm['chunk_size']} tok, {cm['chunk_overlap']} overlap)")
    print(f"  Entities        : {cm['pre_entities']} → {cm['post_entities']}  (dedup {cm['entity_dedup_ratio']})")
    print(f"  Edges           : {cm['pre_edges']} → {cm['post_edges']}  (dedup {cm['edge_dedup_ratio']})")
    print(f"  LLM tokens used : {cm['llm_total_tokens']}  ({cm['llm_api_calls']} calls)")
    print(f"  Wall time       : {cm['execution_time_sec']}s")
    if qa_eval_result:
        print(f"  QA (excl CAT5)  : {qa_eval_result['overall_avg_score_excl_cat5']}")
        for cat, s in qa_eval_result["by_category"].items():
            print(f"    {cat}: {s['avg_score']}  (n={s['count']})")
    if fact_eval_result:
        print(
            f"  Fact Recall     : {fact_eval_result['overall_recall_score']}  "
            f"({fact_eval_result['total_recalled']}/{fact_eval_result['total_facts']})"
        )


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="KGGen: single-script construction + QA/Fact evaluation"
    )

    # ---- Mode: single contract, a specific dataset section, or all ----
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--contract", type=str,
                      help="Path to a single contract .txt file")
    mode.add_argument("--all", action="store_true",
                      help="Evaluate every contract in both dataset sections under --eval-data-dir")
    mode.add_argument("--ten-contracts-dataset", action="store_true",
                      help="Evaluate only the ten_contracts_dataset section under --eval-data-dir")
    mode.add_argument("--long-form-contracts", action="store_true",
                      help="Evaluate only the long_form_contracts section under --eval-data-dir")

    # ---- Files for single-contract mode ----
    parser.add_argument("--qa-file",   type=str, default=None,
                        help="QA JSON path (single-contract mode; must contain 'qa_pairs')")
    parser.add_argument("--fact-file", type=str, default=None,
                        help="Fact JSON path (single-contract mode; must contain 'facts')")

    # ---- Directory for multi-contract modes ----
    parser.add_argument("--eval-data-dir", type=str, default=None,
                        help="Root evaluation_data/ directory (required for --all, "
                             "--ten-contracts-dataset, --long-form-contracts)")

    # ---- Output ----
    parser.add_argument("--out-dir", type=str, default="./results",
                        help="Base directory for all output")

    # ---- Models & API ----
    parser.add_argument("--model",       type=str,
                        default="Qwen/Qwen3-30B-A3B-Instruct-2507-FP8")
    parser.add_argument("--embed-model", type=str,
                        default="google/embeddinggemma-300m")
    parser.add_argument("--api-base",    type=str,
                        default="http://localhost:8000/v1")
    parser.add_argument("--api-key",     type=str, default="EMPTY")

    # ---- Construction hyperparams ----
    parser.add_argument("--chunk-size",    type=int, default=600)
    parser.add_argument("--chunk-overlap", type=int, default=50)
    parser.add_argument("--batch-size",    type=int, default=5,
                        help="Chunks per parallel batch during construction")
    parser.add_argument("--max-workers",   type=int, default=10,
                        help="Thread-pool size (construction and evaluation share this)")

    # ---- Retrieval hyperparams ----
    parser.add_argument("--retrieval-k",        type=int, default=10,
                        help="Top-K seed nodes for KGGen retrieval")
    parser.add_argument("--retrieval-depth",    type=int, default=2,
                        help="BFS depth for KGGen context expansion")
    parser.add_argument("--max-context-tokens", type=int, default=2000,
                        help="Max tokens fed to the LLM judge per query")

    # ---- Judge LLM ----
    parser.add_argument("--judge-max-tokens", type=int, default=8000,
                        help="max_tokens for the DSPy judge LM")

    args = parser.parse_args()

    # ---- Validate argument combinations ----
    multi_contract_mode = args.all or args.ten_contracts_dataset or args.long_form_contracts
    if multi_contract_mode and not args.eval_data_dir:
        parser.error("--all, --ten-contracts-dataset, and --long-form-contracts all require --eval-data-dir")
    if args.contract and not args.qa_file and not args.fact_file:
        parser.error("Single-contract mode requires at least one of --qa-file or --fact-file")

    # ---- Shared setup (done once regardless of how many contracts) ----
    configure_dspy(max_tokens=args.judge_max_tokens)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # ---- Resolve which dataset sub-directories to scan ----
    if args.all:
        subdirs_to_scan = DATASET_SUBDIRS
        mode_label = "--all"
    elif args.ten_contracts_dataset:
        subdirs_to_scan = ["ten_contracts_dataset"]
        mode_label = "--ten-contracts-dataset"
    elif args.long_form_contracts:
        subdirs_to_scan = ["long_form_contracts"]
        mode_label = "--long-form-contracts"
    else:
        subdirs_to_scan = None   # single-contract mode, not used below

    # ---- Collect contracts to process ----
    if multi_contract_mode:
        print(f"\n[*] {mode_label} mode: scanning {args.eval_data_dir}")
        triplets = collect_all_contracts(args.eval_data_dir, subdirs=subdirs_to_scan)
        if not triplets:
            print("[ERROR] No contracts found. Check --eval-data-dir.")
            sys.exit(1)
        print(f"[*] Found {len(triplets)} contracts to evaluate.\n")
        for i, (contract_path, qa_path, fact_path) in enumerate(triplets, 1):
            print(f"\n{'#' * 65}")
            print(f"# CONTRACT {i}/{len(triplets)}: {os.path.basename(contract_path)}")
            print(f"{'#' * 65}")
            run_single_contract(
                contract_path=contract_path,
                qa_file=qa_path,
                fact_file=fact_path,
                out_dir=args.out_dir,
                model=args.model,
                embed_model=args.embed_model,
                api_base=args.api_base,
                api_key=args.api_key,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
                batch_size=args.batch_size,
                max_workers=args.max_workers,
                retrieval_k=args.retrieval_k,
                retrieval_depth=args.retrieval_depth,
                max_context_tokens=args.max_context_tokens,
                judge_max_tokens=args.judge_max_tokens,
                tokenizer=tokenizer,
            )
    else:  # --contract (single)
        run_single_contract(
            contract_path=args.contract,
            qa_file=args.qa_file,
            fact_file=args.fact_file,
            out_dir=args.out_dir,
            model=args.model,
            embed_model=args.embed_model,
            api_base=args.api_base,
            api_key=args.api_key,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            batch_size=args.batch_size,
            max_workers=args.max_workers,
            retrieval_k=args.retrieval_k,
            retrieval_depth=args.retrieval_depth,
            max_context_tokens=args.max_context_tokens,
            judge_max_tokens=args.judge_max_tokens,
            tokenizer=tokenizer,
        )


if __name__ == "__main__":
    main()