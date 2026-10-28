#!/usr/bin/env python3
"""
Naive RAG Insurance Contract Evaluation

Dense retrieval baseline for the insurance contract dataset. Chunks each
contract, embeds via a vLLM-served embedding model, builds an in-memory
FAISS flat index, then runs QA and Fact evaluation using the same DSPy
judge functions as all other systems.

No separate construction / evaluation modes — the full pipeline (embed +
index + evaluate) runs in one pass per contract, matching the single-pass
structure of the KGGen and HyperGraphRAG insurance evaluation scripts.

Single contract:
    python insurance_evaluation.py \
        --contract path/to/contract.txt \
        --qa-file  path/to/qa.json \
        --fact-file path/to/facts.json \
        --out-dir  ./results

All contracts in both dataset sections:
    python insurance_evaluation.py \
        --all \
        --eval-data-dir ./evaluation_data \
        --out-dir ./results

Only ten_contracts_dataset:
    python insurance_evaluation.py \
        --ten-contracts-dataset \
        --eval-data-dir ./evaluation_data \
        --out-dir ./results

Only long_form_contracts:
    python insurance_evaluation.py \
        --long-form-contracts \
        --eval-data-dir ./evaluation_data \
        --out-dir ./results
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

import faiss
import numpy as np
import litellm
from transformers import AutoTokenizer

from experiments.evaluate_mine import ask_llm_query, score_llm_answer, evaluate_fact
from experiments.utils.chunking import truncate_context
from nkg.utils.config import configure_dspy
from nkg.utils.chunking import chunk_text_by_tokens


DATASET_SUBDIRS = ["ten_contracts_dataset", "long_form_contracts"]


# =============================================================================
# TEE STREAM — mirrors all stdout to a log file
# =============================================================================

class TeeStream:
    def __init__(self, log_path: str, original_stream):
        self._file   = open(log_path, "w", encoding="utf-8", buffering=1)
        self._stream = original_stream
        self._lock   = threading.Lock()

    def write(self, data):
        with self._lock:
            self._stream.write(data)
            self._file.write(data)

    def flush(self):
        self._stream.flush()
        self._file.flush()

    def close(self):
        self._file.close()

    def fileno(self):
        return self._stream.fileno()

    def isatty(self):
        return self._stream.isatty()


# =============================================================================
# ERROR TRACKER
# =============================================================================

class ErrorTracker:
    def __init__(self):
        self._lock       = threading.Lock()
        self._by_section = defaultdict(int)

    def record(self, section: str):
        with self._lock:
            self._by_section[section] += 1

    def summary(self) -> dict:
        with self._lock:
            by_section = dict(self._by_section)
            return {"total_errors": sum(by_section.values()), "by_section": by_section}


# =============================================================================
# DATASET PATH RESOLUTION  (same logic as other insurance eval scripts)
# =============================================================================

def resolve_paths_for_contract(contract_path: str, dataset_root: str):
    base        = os.path.splitext(os.path.basename(contract_path))[0]
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
    fact_path = os.path.join(section_dir, "facts", f"facts_{base}.json")
    if not os.path.exists(fact_path):
        fact_path = os.path.join(section_dir, "facts", f"{base}.json")
    if not os.path.exists(qa_path):
        raise FileNotFoundError(f"Missing QA file:   {qa_path}")
    if not os.path.exists(fact_path):
        raise FileNotFoundError(f"Missing Fact file: {fact_path}")
    return qa_path, fact_path


def collect_all_contracts(eval_data_dir: str, subdirs=None):
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
# EMBEDDING HELPERS
# =============================================================================

def _embed_batch(texts: list, embed_model: str, embed_api_base: str,
                 api_key: str) -> list:
    """Embed a list of texts in one LiteLLM call. Returns list of float vectors."""
    resp = litellm.embedding(
        model=f"openai/{embed_model}",
        input=texts,
        api_base=embed_api_base,
        api_key=api_key,
    )
    return [d["embedding"] for d in resp.data]


def _l2_normalize(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return arr / norms


def embed_texts(texts: list, embed_model: str, embed_api_base: str,
                api_key: str, batch_size: int = 64) -> np.ndarray:
    """
    Embed all texts in batches. Returns L2-normalized float32 array
    of shape (len(texts), embedding_dim). Inner-product on L2-normalized
    vectors equals cosine similarity, matching FAISS IndexFlatIP semantics.
    """
    all_vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        vecs  = _embed_batch(batch, embed_model, embed_api_base, api_key)
        all_vecs.extend(vecs)
    arr = np.array(all_vecs, dtype=np.float32)
    return _l2_normalize(arr)


def embed_one(text: str, embed_model: str, embed_api_base: str,
              api_key: str) -> np.ndarray:
    """Embed a single text. Returns L2-normalized 1D float32 array."""
    vecs = _embed_batch([text], embed_model, embed_api_base, api_key)
    arr  = np.array([vecs[0]], dtype=np.float32)
    return _l2_normalize(arr)[0]


# =============================================================================
# CONSTRUCTION  (chunk → embed → FAISS index)
# =============================================================================

def build_index(
    file_path: str,
    embed_model: str,
    embed_api_base: str,
    api_key: str,
    model: str,
    chunk_size: int,
    chunk_overlap: int,
):
    """
    Reads the contract, chunks it, embeds each chunk via vLLM, and builds
    an in-memory FAISS IndexFlatIP.

    Returns:
        index   — faiss.IndexFlatIP (L2-normalised vectors, cosine sim)
        chunks  — list[str], one entry per FAISS vector (same order)
        metrics — dict with construction statistics
    """
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

    chunks = chunk_text_by_tokens(
        model=model, text=full_text, chunk_size=chunk_size, overlap=chunk_overlap
    )
    print(f"[Construction] Chunks: {len(chunks)} "
          f"(chunk_size={chunk_size}, overlap={chunk_overlap})")

    print(f"[Construction] Embedding {len(chunks)} chunks via vLLM ...")
    t0 = time.time()
    chunk_embs = embed_texts(chunks, embed_model, embed_api_base, api_key)
    embed_time = round(time.time() - t0, 2)
    print(f"[Construction] Embedding done in {embed_time}s  "
          f"shape={chunk_embs.shape}")

    dim   = chunk_embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(chunk_embs)
    print(f"[Construction] FAISS index: {index.ntotal} vectors  dim={dim}")

    metrics = {
        "total_document_tokens": total_document_tokens,
        "chunk_size":            chunk_size,
        "chunk_overlap":         chunk_overlap,
        "total_chunks":          len(chunks),
        "embedding_dim":         dim,
        "embed_time_sec":        embed_time,
        "execution_time_sec":    embed_time,  # construction IS embedding for naive RAG
        "_embedding_note": (
            "Embedding tokens are not tracked — vLLM embedding responses "
            "do not reliably report token usage."
        ),
    }
    return index, chunks, metrics


# =============================================================================
# RETRIEVAL
# =============================================================================

def retrieve_context(
    query: str,
    index,
    chunks: list,
    embed_model: str,
    embed_api_base: str,
    api_key: str,
    top_k: int,
) -> str:
    """
    Embed the query, search FAISS for the top-k nearest chunks, and
    concatenate their texts separated by '---'.
    """
    q_vec = embed_one(query, embed_model, embed_api_base, api_key)
    _D, I = index.search(q_vec.reshape(1, -1), top_k)
    retrieved = [chunks[idx] for idx in I[0] if 0 <= idx < len(chunks)]
    return "\n\n---\n\n".join(retrieved)


# =============================================================================
# QA EVALUATION WORKER
# =============================================================================

def _eval_qa_item(
    pair, index, chunks, embed_model, embed_api_base, api_key, top_k,
    tokenizer, max_context_tokens, error_tracker: ErrorTracker,
):
    q        = pair["question"]
    a        = pair["answer"]
    category = pair.get("category", "UNKNOWN")
    evidence = pair.get("evidence", "")

    llm_answer = score = 0.0
    raw_context = truncated_context = ""
    context_tokens = 0
    status = "SUCCESS"

    current_section = "qa_evaluation.context_retrieval"
    try:
        raw_context = retrieve_context(
            q, index, chunks, embed_model, embed_api_base, api_key, top_k
        )
        truncated_context, context_tokens = truncate_context(
            raw_context, tokenizer, max_context_tokens
        )

        current_section = "qa_evaluation.llm_answer_generation"
        llm_answer = ask_llm_query(question=q, context=truncated_context)

        current_section = "qa_evaluation.llm_judge_scoring"
        score = score_llm_answer(question=q, llm_answer=llm_answer, correct_answer=a)
    except Exception as e:
        status = "ERROR"
        error_tracker.record(current_section)
        print(f"[QA ERROR | {current_section}] {q[:60]}... → {e}")

    return {
        "question":                 q,
        "expected_answer":          a,
        "evidence":                 evidence,
        "category":                 category,
        "llm_answer":               llm_answer,
        "score":                    float(score),
        "judge_decision":           float(score),
        "retrieved_context":        truncated_context,
        "raw_context_length_chars": len(raw_context),
        "context_tokens":           context_tokens,
        "max_context_tokens":       max_context_tokens,
        "status":                   status,
    }


# =============================================================================
# FACT EVALUATION WORKER
# =============================================================================

def _eval_fact_item(
    fact_dict, index, chunks, embed_model, embed_api_base, api_key, top_k,
    tokenizer, max_context_tokens, error_tracker: ErrorTracker,
):
    fact    = fact_dict["fact"]
    fact_id = fact_dict.get("id")

    score = 0
    raw_context = truncated_context = ""
    context_tokens = 0
    status = "SUCCESS"

    current_section = "fact_evaluation.context_retrieval"
    try:
        raw_context = retrieve_context(
            fact, index, chunks, embed_model, embed_api_base, api_key, top_k
        )
        truncated_context, context_tokens = truncate_context(
            raw_context, tokenizer, max_context_tokens
        )

        current_section = "fact_evaluation.llm_judge_scoring"
        score = evaluate_fact(context=truncated_context, correct_answer=fact)
    except Exception as e:
        status = "ERROR"
        error_tracker.record(current_section)
        print(f"[FACT ERROR | {current_section}] {fact[:60]}... → {e}")

    return {
        "fact_id":                  fact_id,
        "fact":                     fact,
        "score":                    float(score),
        "judge_decision":           int(score),
        "retrieved_context":        truncated_context,
        "raw_context_length_chars": len(raw_context),
        "context_tokens":           context_tokens,
        "max_context_tokens":       max_context_tokens,
        "status":                   status,
    }


# =============================================================================
# RUN QA EVALUATION
# =============================================================================

def run_qa_evaluation(
    qa_file, index, chunks, embed_model, embed_api_base, api_key, top_k,
    tokenizer, max_context_tokens, max_workers, error_tracker: ErrorTracker,
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
                _eval_qa_item, p, index, chunks, embed_model, embed_api_base,
                api_key, top_k, tokenizer, max_context_tokens, error_tracker,
            ): i
            for i, p in enumerate(pairs)
        }
        done = 0
        for future in as_completed(future_map):
            res = future.result()
            results.append(res)
            done += 1
            print(f"[QA Eval] {done}/{len(pairs)} cat={res['category']} score={res['score']}")

    by_category = defaultdict(list)
    for r in results:
        by_category[r["category"]].append(r["score"])

    category_summary = {
        cat: {"avg_score": round(sum(s) / len(s), 4), "count": len(s), "scores": s}
        for cat, s in by_category.items()
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
    fact_file, index, chunks, embed_model, embed_api_base, api_key, top_k,
    tokenizer, max_context_tokens, max_workers, error_tracker: ErrorTracker,
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
                _eval_fact_item, fd, index, chunks, embed_model, embed_api_base,
                api_key, top_k, tokenizer, max_context_tokens, error_tracker,
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

    print(f"[Fact Eval] Recall: {overall_avg} ({total_recalled}/{len(facts)} facts recalled)")

    return {
        "overall_recall_score": overall_avg,
        "total_facts":          len(facts),
        "total_recalled":       total_recalled,
        "results":              results,
    }


# =============================================================================
# SINGLE CONTRACT PIPELINE
# =============================================================================

def run_single_contract(
    contract_path: str,
    qa_file,
    fact_file,
    out_dir: str,
    model: str,
    embed_model: str,
    embed_api_base: str,
    api_key: str,
    chunk_size: int,
    chunk_overlap: int,
    top_k: int,
    max_context_tokens: int,
    judge_max_tokens: int,
    max_workers: int,
    tokenizer,
):
    contract_name = os.path.splitext(os.path.basename(contract_path))[0]
    eval_tag      = "_".join(
        (["qa"] if qa_file else []) + (["fact"] if fact_file else [])
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir   = os.path.join(out_dir, f"{contract_name}_{eval_tag}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    log_path   = os.path.join(run_dir, "run.log")
    tee        = TeeStream(log_path, sys.stdout)
    sys.stdout = tee

    error_tracker = ErrorTracker()

    print("\n" + "=" * 65)
    print(f" CONTRACT: {contract_name}")
    print(f" Output  : {run_dir}")
    print("=" * 65)

    # ---- Phase 1: Construction (chunk + embed + FAISS index) ----
    print("\n[*] PHASE 1 — CONSTRUCTION (CHUNK + EMBED + INDEX)")
    try:
        index, chunks, construction_metrics = build_index(
            file_path=contract_path,
            embed_model=embed_model,
            embed_api_base=embed_api_base,
            api_key=api_key,
            model=model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    except Exception as e:
        print(f"[FATAL] Construction failed for {contract_name}: {e}")
        traceback.print_exc()
        sys.stdout = tee._stream
        tee.close()
        return

    # ---- Phase 2: Evaluation ----
    qa_eval_result   = None
    fact_eval_result = None

    if qa_file:
        print("\n[*] PHASE 2a — QA EVALUATION")
        qa_eval_result = run_qa_evaluation(
            qa_file=qa_file,
            index=index,
            chunks=chunks,
            embed_model=embed_model,
            embed_api_base=embed_api_base,
            api_key=api_key,
            top_k=top_k,
            tokenizer=tokenizer,
            max_context_tokens=max_context_tokens,
            max_workers=max_workers,
            error_tracker=error_tracker,
        )

    if fact_file:
        print("\n[*] PHASE 2b — FACT RECALL EVALUATION")
        fact_eval_result = run_fact_evaluation(
            fact_file=fact_file,
            index=index,
            chunks=chunks,
            embed_model=embed_model,
            embed_api_base=embed_api_base,
            api_key=api_key,
            top_k=top_k,
            tokenizer=tokenizer,
            max_context_tokens=max_context_tokens,
            max_workers=max_workers,
            error_tracker=error_tracker,
        )

    error_summary = error_tracker.summary()

    final_output = {
        "run_id":    f"naive_rag_{uuid.uuid4().hex[:8]}",
        "timestamp": timestamp,
        "system":    "NAIVE_RAG",

        "metadata": {
            "contract_file":      os.path.abspath(contract_path),
            "qa_file":            qa_file,
            "fact_file":          fact_file,
            "llm_model":          model,
            "embedding_model":    embed_model,
            "embed_api_base":     embed_api_base,
            "retrieval_params": {
                "top_k":              top_k,
                "max_context_tokens": max_context_tokens,
            },
            "chunking": {
                "chunk_size":    chunk_size,
                "chunk_overlap": chunk_overlap,
            },
            "judge_max_tokens": judge_max_tokens,
        },

        "construction":    construction_metrics,
        "qa_evaluation":   qa_eval_result,
        "fact_evaluation": fact_eval_result,
        "error_summary":   error_summary,
    }

    results_path = os.path.join(run_dir, "results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2, default=str)

    # ---- Terminal summary ----
    cm = construction_metrics
    print(f"\n  Log     → {log_path}")
    print(f"  Results → {results_path}")
    print(f"  Document tokens : {cm['total_document_tokens']}")
    print(f"  Chunks          : {cm['total_chunks']}  "
          f"({cm['chunk_size']} tok, {cm['chunk_overlap']} overlap)")
    print(f"  Embedding dim   : {cm['embedding_dim']}")
    print(f"  Embed time      : {cm['embed_time_sec']}s")
    print(f"  top_k           : {top_k}")
    if qa_eval_result:
        print(f"  QA (excl CAT5)  : {qa_eval_result['overall_avg_score_excl_cat5']}")
        for cat, s in qa_eval_result["by_category"].items():
            print(f"    {cat}: {s['avg_score']}  (n={s['count']})")
    if fact_eval_result:
        print(f"  Fact Recall     : {fact_eval_result['overall_recall_score']}  "
              f"({fact_eval_result['total_recalled']}/{fact_eval_result['total_facts']})")
    print(f"  Errors (total)  : {error_summary['total_errors']}")
    if error_summary["by_section"]:
        for section, count in error_summary["by_section"].items():
            print(f"    {section}: {count}")

    sys.stdout = tee._stream
    tee.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Naive RAG — insurance contract QA / Fact evaluation"
    )

    # ---- Mode ----
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--contract", type=str,
                      help="Path to a single contract .txt file")
    mode.add_argument("--all", action="store_true",
                      help="Evaluate every contract in both dataset sections")
    mode.add_argument("--ten-contracts-dataset", action="store_true",
                      help="Evaluate only the ten_contracts_dataset section")
    mode.add_argument("--long-form-contracts", action="store_true",
                      help="Evaluate only the long_form_contracts section")

    parser.add_argument("--qa-file",       type=str, default=None)
    parser.add_argument("--fact-file",     type=str, default=None)
    parser.add_argument("--eval-data-dir", type=str, default=None)
    parser.add_argument("--out-dir",       type=str, default="./results")

    # ---- LLM (judge) ----
    parser.add_argument("--model", type=str,
                        default="Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
                        help="LLM model for DSPy judge (also used for token counting).")
    parser.add_argument("--api-base", type=str,
                        default="http://localhost:8000/v1")
    parser.add_argument("--api-key",  type=str, default="EMPTY")
    parser.add_argument("--judge-max-tokens", type=int, default=8000)

    # ---- Embedding ----
    parser.add_argument("--embed-model", type=str,
                        default="google/gemma-embedding-300m",
                        help="Embedding model served by vLLM.")
    parser.add_argument("--embed-api-base", type=str,
                        default="http://localhost:8001/v1",
                        help="vLLM embedding endpoint.")

    # ---- Chunking ----
    parser.add_argument("--chunk-size",    type=int, default=600)
    parser.add_argument("--chunk-overlap", type=int, default=50)

    # ---- Retrieval ----
    parser.add_argument("--top-k",              type=int, default=5,
                        help="Number of chunks to retrieve per query.")
    parser.add_argument("--max-context-tokens", type=int, default=2000,
                        help="Token budget for context passed to the judge LLM.")

    # ---- Parallelism ----
    parser.add_argument("--max-workers", type=int, default=10,
                        help="Thread-pool size for QA/Fact evaluation.")

    args = parser.parse_args()

    # ---- Validate ----
    multi_mode = args.all or args.ten_contracts_dataset or args.long_form_contracts
    if multi_mode and not args.eval_data_dir:
        parser.error("--all, --ten-contracts-dataset, and --long-form-contracts require --eval-data-dir")
    if args.contract and not args.qa_file and not args.fact_file:
        parser.error("Single-contract mode requires at least one of --qa-file or --fact-file")

    configure_dspy(
        model=f"openai/{args.model}",
        api_base=args.api_base,
        api_key=args.api_key,
        max_tokens=args.judge_max_tokens,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    if args.all:
        subdirs_to_scan = DATASET_SUBDIRS
    elif args.ten_contracts_dataset:
        subdirs_to_scan = ["ten_contracts_dataset"]
    elif args.long_form_contracts:
        subdirs_to_scan = ["long_form_contracts"]
    else:
        subdirs_to_scan = None

    shared = dict(
        out_dir=args.out_dir,
        model=args.model,
        embed_model=args.embed_model,
        embed_api_base=args.embed_api_base,
        api_key=args.api_key,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        top_k=args.top_k,
        max_context_tokens=args.max_context_tokens,
        judge_max_tokens=args.judge_max_tokens,
        max_workers=args.max_workers,
        tokenizer=tokenizer,
    )

    print(f"\n{'=' * 65}")
    print(f"  NAIVE RAG — INSURANCE EVALUATION")
    print(f"  model         : {args.model}")
    print(f"  embed-model   : {args.embed_model}  @  {args.embed_api_base}")
    print(f"  chunk         : {args.chunk_size} tok / {args.chunk_overlap} overlap")
    print(f"  top_k         : {args.top_k}")
    print(f"  max-ctx-tok   : {args.max_context_tokens}")
    print(f"  max-workers   : {args.max_workers}")
    print(f"  out-dir       : {args.out_dir}")
    print(f"{'=' * 65}\n")

    if multi_mode:
        triplets = collect_all_contracts(args.eval_data_dir, subdirs=subdirs_to_scan)
        if not triplets:
            print("[ERROR] No contracts found. Check --eval-data-dir.")
            sys.exit(1)
        print(f"[*] Found {len(triplets)} contracts to evaluate.\n")
        for i, (contract_path, qa_path, fact_path) in enumerate(triplets, 1):
            print(f"\n{'#' * 65}")
            print(f"# CONTRACT {i}/{len(triplets)}: {os.path.basename(contract_path)}")
            print(f"{'#' * 65}")
            try:
                run_single_contract(
                    contract_path=contract_path,
                    qa_file=qa_path,
                    fact_file=fact_path,
                    **shared,
                )
            except Exception as e:
                print(f"[FATAL] Contract {contract_path} failed: {e}")
                traceback.print_exc()
                continue
    else:
        run_single_contract(
            contract_path=args.contract,
            qa_file=args.qa_file,
            fact_file=args.fact_file,
            **shared,
        )


if __name__ == "__main__":
    main()
