#!/usr/bin/env python3
"""
HyperGraphRAG Insurance Contract Evaluation — Single Script
Runs graph construction followed by QA and/or Fact evaluation.

NOTE on retrieval modes: this version of HyperGraphRAG only implements
mode="hybrid" in aquery(). Passing any other mode will silently produce
no response. Use --retrieval-mode hybrid (the default).

Single contract:
    python hgrag_eval.py \
        --contract path/to/contract.txt \
        --qa-file path/to/qa.json \
        --fact-file path/to/facts.json \
        --out-dir ./results

All contracts in both dataset sections:
    python hgrag_eval.py \
        --all \
        --eval-data-dir ./evaluation_data \
        --out-dir ./results

Only ten_contracts_dataset:
    python hgrag_eval.py \
        --ten-contracts-dataset \
        --eval-data-dir ./evaluation_data \
        --out-dir ./results

Only long_form_contracts:
    python hgrag_eval.py \
        --long-form-contracts \
        --eval-data-dir ./evaluation_data \
        --out-dir ./results

All other flags (model, chunking, retrieval, etc.) apply in every mode.

ARCHITECTURE NOTE
-----------------
This script runs the **entire pipeline inside a single event loop**, started
once via asyncio.run(main_async(...)) at the top of main(). This is required
to avoid LiteLLM's GLOBAL_LOGGING_WORKER reusing an already-awaited coroutine
across multiple event loops, which previously produced:
    RuntimeError: cannot reuse already awaited coroutine
    Task was destroyed but it is pending!

Concurrency for QA/Fact evaluation is achieved with asyncio.Semaphore +
asyncio.gather, NOT a ThreadPoolExecutor. Sync DSPy judge calls
(ask_llm_query, score_llm_answer, evaluate_fact) are off-loaded to the
default executor via loop.run_in_executor() so they don't block the loop.
"""

import os
import sys
import json
import time
import uuid
import asyncio
import argparse
import threading
import traceback
from datetime import datetime
from collections import defaultdict

import litellm

# =============================================================================
# DISABLE LITELLM ASYNC LOGGING WORKER
# =============================================================================
# Clearing callbacks alone is not sufficient — LiteLLM's GLOBAL_LOGGING_WORKER
# is a module-level singleton that stores its coroutine in self._task and
# tries to reschedule it whenever an async LiteLLM call happens on a new
# event loop. Across loops, the same coroutine object gets re-scheduled,
# producing "cannot reuse already awaited coroutine".
#
# We do two things:
#   1. Empty every callback list so nothing is enqueued in the first place.
#   2. Monkey-patch the global worker's start/enqueue/init methods to no-ops.
# =============================================================================
litellm.success_callback         = []
litellm.failure_callback         = []
litellm._async_success_callback  = []
litellm._async_failure_callback  = []
litellm.callbacks                = []
litellm.input_callback           = []
litellm.service_callback         = []

try:
    from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER

    def _noop(*_a, **_kw):
        return None

    async def _noop_async(*_a, **_kw):
        return None

    GLOBAL_LOGGING_WORKER.ensure_initialized = _noop
    GLOBAL_LOGGING_WORKER.enqueue            = _noop
    GLOBAL_LOGGING_WORKER.start              = _noop
    GLOBAL_LOGGING_WORKER._worker_loop       = _noop_async
    GLOBAL_LOGGING_WORKER._task              = None
    GLOBAL_LOGGING_WORKER._queue             = None
except (ImportError, AttributeError):
    pass

from transformers import AutoTokenizer

# ---------- Project imports ----------
from experiments.evaluate_mine import ask_llm_query, score_llm_answer, evaluate_fact
from experiments.utils.chunking import truncate_context
from nkg.utils.config import configure_dspy

from hypergraphrag import HyperGraphRAG
from hypergraphrag.base import QueryParam


# =============================================================================
# DATASET PATH RESOLUTION
# =============================================================================
DATASET_SUBDIRS = ["ten_contracts_dataset", "long_form_contracts"]


# =============================================================================
# TEE STREAM — duplicates all stdout to a log file
# =============================================================================
class TeeStream:
    """Wraps sys.stdout so prints go to both the console and a log file."""
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
# ERROR TRACKER — records pipeline section only (not error content)
# =============================================================================
class ErrorTracker:
    """
    Counter for errors per pipeline section.

    Sections:
        graph_construction
        qa_evaluation.context_retrieval
        qa_evaluation.llm_answer_generation
        qa_evaluation.llm_judge_scoring
        fact_evaluation.context_retrieval
        fact_evaluation.llm_judge_scoring
    """

    def __init__(self):
        self._lock       = threading.Lock()
        self._by_section = defaultdict(int)

    def record(self, section: str):
        with self._lock:
            self._by_section[section] += 1

    def summary(self) -> dict:
        with self._lock:
            by_section = dict(self._by_section)
            return {
                "total_errors": sum(by_section.values()),
                "by_section":   by_section,
            }


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
# ASYNC TOKEN TRACKER
# =============================================================================
class AsyncTokenTracker:
    """Thread-safe manual token accumulator for async LiteLLM wrappers."""

    def __init__(self):
        self.prompt_tokens     = 0
        self.completion_tokens = 0
        self.api_calls         = 0
        self._lock             = threading.Lock()

    def record(self, usage):
        if usage is None:
            return
        if isinstance(usage, dict):
            p = usage.get("prompt_tokens", 0)
            c = usage.get("completion_tokens", 0)
        else:
            p = getattr(usage, "prompt_tokens", 0)
            c = getattr(usage, "completion_tokens", 0)
        with self._lock:
            self.prompt_tokens     += p
            self.completion_tokens += c
            self.api_calls         += 1

    @property
    def total_tokens(self):
        return self.prompt_tokens + self.completion_tokens

    def snapshot(self):
        with self._lock:
            return {
                "llm_prompt_tokens":     self.prompt_tokens,
                "llm_completion_tokens": self.completion_tokens,
                "llm_total_tokens":      self.total_tokens,
                "llm_api_calls":         self.api_calls,
            }


# =============================================================================
# RETRY HELPER for transient vLLM connection errors
# =============================================================================
# Under bursty load (10 concurrent queries × multiple LLM/embedding calls each)
# vLLM can briefly refuse or drop connections. The error surfaces as
# litellm.APIConnectionError with repr "Connection error." We retry these
# transparently with exponential backoff + jitter so a single bad TCP handshake
# doesn't kill an entire QA/Fact item.
import random  # noqa: E402  (intentional: kept near retry helper for clarity)

# Build the tuple of exception classes we should retry on. Some of these
# attributes don't exist in older LiteLLM versions, so we fall back to a
# generic Exception class via getattr() — which means "this entry won't catch
# anything specific". The OSError catch-all at the end handles raw socket
# errors that bubble up from httpx underneath LiteLLM.
_TRANSIENT_ERRORS = tuple({
    getattr(litellm, "APIConnectionError",      None),
    getattr(litellm, "APIError",                None),
    getattr(litellm, "Timeout",                 None),
    getattr(litellm, "RateLimitError",          None),
    getattr(litellm, "ServiceUnavailableError", None),
    getattr(litellm, "InternalServerError",     None),
    ConnectionError,
    asyncio.TimeoutError,
    OSError,
} - {None})


async def _with_retries(coro_factory, label: str,
                        max_retries: int, base_delay: float = 1.0,
                        cap_delay: float = 30.0):
    """
    Calls coro_factory() up to (max_retries + 1) times. On a transient error,
    sleeps base_delay * 2**attempt seconds (capped at cap_delay) plus jitter,
    then retries. After max_retries retries, the last exception is re-raised
    so the caller's error-tracking logic kicks in.

    coro_factory is a zero-arg callable returning a coroutine — we re-call it
    each attempt because awaited coroutines can't be reused.
    """
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            return await coro_factory()
        except _TRANSIENT_ERRORS as e:
            last_exc = e
            if attempt == max_retries:
                break
            delay = min(base_delay * (2 ** attempt), cap_delay) + random.uniform(0, 1)
            print(f"[RETRY | {label}] attempt {attempt+1}/{max_retries+1} "
                  f"failed: {type(e).__name__}: {e} — sleeping {delay:.1f}s")
            await asyncio.sleep(delay)
    raise last_exc


# =============================================================================
# ASYNC LLM WRAPPERS
# =============================================================================
def make_llm_wrappers(model: str, embed_model: str,
                      api_base: str, embed_api_base: str,
                      api_key: str, tracker: AsyncTokenTracker,
                      embed_dim: int = 2560,
                      max_retries: int = 5,
                      request_timeout: float = 120.0):
    """Returns (complete_func, embed_func) bound to endpoints + tracker.

    Both wrappers retry transient connection errors (see _with_retries).
    request_timeout is forwarded to LiteLLM/httpx so a hung connection fails
    fast enough to retry rather than blocking forever.
    """

    async def local_vllm_complete(prompt: str, **kwargs) -> str:
        def _factory():
            return litellm.acompletion(
                model=f"openai/{model}",
                messages=[{"role": "user", "content": prompt}],
                api_base=api_base,
                api_key=api_key,
                temperature=0.0,
                max_tokens=kwargs.get("max_tokens", 8192),
                timeout=request_timeout,
            )
        response = await _with_retries(_factory, "completion", max_retries)
        tracker.record(getattr(response, "usage", None))
        return response.choices[0].message.content

    async def local_vllm_embed(texts: list, **kwargs) -> list:
        def _factory():
            return litellm.aembedding(
                model=f"openai/{embed_model}",
                input=texts,
                api_base=embed_api_base,
                api_key=api_key,
                timeout=request_timeout,
            )
        response = await _with_retries(_factory, "embedding", max_retries)
        tracker.record(getattr(response, "usage", None))
        return [d["embedding"] for d in response.data]

    local_vllm_embed.embedding_dim = embed_dim
    return local_vllm_complete, local_vllm_embed


# =============================================================================
# GRAPH CONSTRUCTION
# =============================================================================
async def _build_graph_async(
    file_path: str,
    working_dir: str,
    model: str,
    embed_model: str,
    api_base: str,
    embed_api_base: str,
    api_key: str,
    chunk_size: int,
    chunk_overlap: int,
    max_gleanings: int,
    embed_dim: int,
    llm_max_async: int,
    embedding_max_async: int,
    llm_max_token_size: int,
    llm_max_retries: int,
    request_timeout: float,
    error_tracker: "ErrorTracker",
):
    """
    Inserts the contract into a fresh HyperGraphRAG index and returns
    (kg_engine, construction_metrics).
    """
    tracker = AsyncTokenTracker()
    complete_fn, embed_fn = make_llm_wrappers(
        model=model,
        embed_model=embed_model,
        api_base=api_base,
        embed_api_base=embed_api_base,
        api_key=api_key,
        tracker=tracker,
        embed_dim=embed_dim,
        max_retries=llm_max_retries,
        request_timeout=request_timeout,
    )

    t0 = time.time()

    kg = HyperGraphRAG(
        working_dir=working_dir,
        llm_model_func=complete_fn,
        llm_model_name=model,
        llm_model_max_token_size=llm_max_token_size,
        llm_model_max_async=llm_max_async,
        embedding_func=embed_fn,
        embedding_batch_num=16,
        embedding_func_max_async=embedding_max_async,
        chunk_token_size=chunk_size,
        chunk_overlap_token_size=chunk_overlap,
        tiktoken_model_name="gpt-4o-mini",
        kv_storage="JsonKVStorage",
        vector_storage="NanoVectorDBStorage",
        graph_storage="NetworkXStorage",
        entity_extract_max_gleaning=max_gleanings,
        # Disable LLM caching during construction so all LLM calls flow
        # through our wrapper and get tracked.
        enable_llm_cache=False,
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
    print(f"[Construction] Running ainsert (chunk_size={chunk_size}, "
          f"overlap={chunk_overlap}, gleanings={max_gleanings})...")

    try:
        await kg.ainsert(full_text)
    except Exception:
        error_tracker.record("graph_construction")
        raise

    execution_time = round(time.time() - t0, 2)

    try:
        nx_graph   = kg.chunk_entity_relation_graph._graph
        n_entities = len(nx_graph.nodes)
        n_edges    = len(nx_graph.edges)
    except Exception as e:
        error_tracker.record("graph_construction")
        print(f"[WARN] Could not read graph metrics: {e}")
        n_entities, n_edges = 0, 0

    try:
        n_chunks = len(kg.text_chunks._data)
    except AttributeError:
        try:
            chunk_file = os.path.join(working_dir, "kv_store", "text_chunks.json")
            with open(chunk_file, "r") as f:
                n_chunks = len(json.load(f))
        except Exception:
            n_chunks = 0

    token_snap = tracker.snapshot()

    artifact_files = []
    if os.path.isdir(working_dir):
        for fname in sorted(os.listdir(working_dir)):
            fpath = os.path.join(working_dir, fname)
            if os.path.isfile(fpath):
                artifact_files.append({
                    "file":       fname,
                    "size_bytes": os.path.getsize(fpath),
                })

    metrics = {
        "total_document_tokens": total_document_tokens,
        "chunk_size":            chunk_size,
        "chunk_overlap":         chunk_overlap,
        "max_gleanings":         max_gleanings,
        "total_chunks":          n_chunks,
        "entities":              n_entities,
        "edges":                 n_edges,
        **token_snap,
        "execution_time_sec":    execution_time,
        "artifact_files":        artifact_files,
    }

    print(
        f"[Construction] Done in {execution_time}s "
        f"| Entities: {n_entities} | Edges: {n_edges} "
        f"| Chunks: {n_chunks} | LLM tokens: {token_snap['llm_total_tokens']}"
    )
    return kg, metrics


# =============================================================================
# QA EVALUATION WORKER (async)
# =============================================================================
async def _eval_qa_item_async(
    pair, kg, retrieval_params,
    tokenizer, max_context_tokens, error_tracker: ErrorTracker,
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

    loop = asyncio.get_running_loop()
    current_section = "qa_evaluation.context_retrieval"
    try:
        param       = QueryParam(**retrieval_params)
        # aquery directly — we're already inside an async context, so we
        # avoid the sync kg.query() wrapper which would create a nested loop.
        raw_context = await kg.aquery(query=q, param=param)
        raw_context = raw_context if isinstance(raw_context, str) else str(raw_context)
        truncated_context, context_tokens = truncate_context(
            raw_context, tokenizer, max_context_tokens
        )

        current_section = "qa_evaluation.llm_answer_generation"
        # DSPy calls are sync (blocking). Run them in the default executor
        # so they don't stall the asyncio event loop.
        llm_answer = await loop.run_in_executor(
            None, ask_llm_query, q, truncated_context
        )

        current_section = "qa_evaluation.llm_judge_scoring"
        score = await loop.run_in_executor(
            None, score_llm_answer, q, llm_answer, a
        )

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
# FACT EVALUATION WORKER (async)
# =============================================================================
async def _eval_fact_item_async(
    fact_dict, kg, retrieval_params,
    tokenizer, max_context_tokens, error_tracker: ErrorTracker,
):
    fact    = fact_dict["fact"]
    fact_id = fact_dict.get("id")

    score             = 0
    raw_context       = ""
    truncated_context = ""
    context_tokens    = 0
    status            = "SUCCESS"

    loop = asyncio.get_running_loop()
    current_section = "fact_evaluation.context_retrieval"
    try:
        param       = QueryParam(**retrieval_params)
        raw_context = await kg.aquery(query=fact, param=param)
        raw_context = raw_context if isinstance(raw_context, str) else str(raw_context)
        truncated_context, context_tokens = truncate_context(
            raw_context, tokenizer, max_context_tokens
        )

        current_section = "fact_evaluation.llm_judge_scoring"
        score = await loop.run_in_executor(
            None, evaluate_fact, truncated_context, fact
        )

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
# RUN QA EVALUATION (async, semaphore-bounded gather)
# =============================================================================
async def run_qa_evaluation_async(
    qa_file, kg, retrieval_params,
    tokenizer, max_context_tokens, max_workers, error_tracker: ErrorTracker,
):
    print(f"\n[QA Eval] Loading: {qa_file}")
    with open(qa_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    pairs = data.get("qa_pairs", [])
    if not pairs:
        print("[QA Eval] No qa_pairs found in file.")
        return None

    print(f"[QA Eval] {len(pairs)} questions | concurrency={max_workers}")

    sem       = asyncio.Semaphore(max_workers)
    completed = {"n": 0}
    total     = len(pairs)
    print_lock = asyncio.Lock()

    async def _bounded(p):
        async with sem:
            res = await _eval_qa_item_async(
                p, kg, retrieval_params,
                tokenizer, max_context_tokens, error_tracker,
            )
        async with print_lock:
            completed["n"] += 1
            print(
                f"[QA Eval] {completed['n']}/{total} "
                f"cat={res['category']} score={res['score']}"
            )
        return res

    results = await asyncio.gather(*[_bounded(p) for p in pairs])

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
# RUN FACT EVALUATION (async, semaphore-bounded gather)
# =============================================================================
async def run_fact_evaluation_async(
    fact_file, kg, retrieval_params,
    tokenizer, max_context_tokens, max_workers, error_tracker: ErrorTracker,
):
    print(f"\n[Fact Eval] Loading: {fact_file}")
    with open(fact_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    facts = data.get("facts", [])
    if not facts:
        print("[Fact Eval] No facts found in file.")
        return None

    print(f"[Fact Eval] {len(facts)} facts | concurrency={max_workers}")

    sem       = asyncio.Semaphore(max_workers)
    completed = {"n": 0}
    total     = len(facts)
    print_lock = asyncio.Lock()
    running_scores = []

    async def _bounded(fd):
        async with sem:
            res = await _eval_fact_item_async(
                fd, kg, retrieval_params,
                tokenizer, max_context_tokens, error_tracker,
            )
        async with print_lock:
            completed["n"] += 1
            running_scores.append(res["score"])
            if completed["n"] % 10 == 0 or completed["n"] == total:
                running_avg = round(sum(running_scores) / len(running_scores), 4)
                print(f"[Fact Eval] {completed['n']}/{total} running recall: {running_avg}")
        return res

    results = await asyncio.gather(*[_bounded(fd) for fd in facts])

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
# SINGLE CONTRACT PIPELINE (async)
# =============================================================================
async def run_single_contract_async(
    contract_path: str,
    qa_file,
    fact_file,
    out_dir: str,
    model: str,
    embed_model: str,
    api_base: str,
    embed_api_base: str,
    api_key: str,
    chunk_size: int,
    chunk_overlap: int,
    max_gleanings: int,
    embed_dim: int,
    llm_max_async: int,
    embedding_max_async: int,
    llm_max_token_size: int,
    llm_max_retries: int,
    request_timeout: float,
    max_workers: int,
    retrieval_mode: str,
    retrieval_top_k: int,
    max_context_tokens: int,
    judge_max_tokens: int,
    tokenizer,
):
    contract_name = os.path.splitext(os.path.basename(contract_path))[0]
    eval_tag      = "_".join(
        (["qa"] if qa_file else []) + (["fact"] if fact_file else [])
    )
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir     = os.path.join(out_dir, f"{contract_name}_{eval_tag}_{timestamp}")
    working_dir = os.path.join(run_dir, "hgrag_index")
    os.makedirs(working_dir, exist_ok=True)

    log_path   = os.path.join(run_dir, "run.log")
    tee        = TeeStream(log_path, sys.stdout)
    sys.stdout = tee

    error_tracker = ErrorTracker()

    retrieval_params = {
        "mode":              retrieval_mode,
        "top_k":             retrieval_top_k,
        "only_need_context": True,
    }

    print("\n" + "=" * 65)
    print(f" CONTRACT: {contract_name}")
    print(f" Output  : {run_dir}")
    print("=" * 65)

    # ---- Phase 1: Construction ----
    print("\n[*] PHASE 1 — GRAPH CONSTRUCTION")
    try:
        kg, construction_metrics = await _build_graph_async(
            file_path=contract_path,
            working_dir=working_dir,
            model=model,
            embed_model=embed_model,
            api_base=api_base,
            embed_api_base=embed_api_base,
            api_key=api_key,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            max_gleanings=max_gleanings,
            embed_dim=embed_dim,
            llm_max_async=llm_max_async,
            embedding_max_async=embedding_max_async,
            llm_max_token_size=llm_max_token_size,
            llm_max_retries=llm_max_retries,
            request_timeout=request_timeout,
            error_tracker=error_tracker,
        )
    except Exception as e:
        print(f"[FATAL] Construction failed for {contract_name}: {e}")
        traceback.print_exc()
        sys.stdout = tee._stream
        tee.close()
        return

    print(f"[*] Index stored in → {working_dir}")
    print("[*] Continuing in same event loop with same engine — no reload needed.")

    # ---- Phase 2: Evaluation (same kg, same loop) ----
    qa_eval_result   = None
    fact_eval_result = None

    if qa_file:
        print("\n[*] PHASE 2a — QA EVALUATION")
        qa_eval_result = await run_qa_evaluation_async(
            qa_file=qa_file,
            kg=kg,
            retrieval_params=retrieval_params,
            tokenizer=tokenizer,
            max_context_tokens=max_context_tokens,
            max_workers=max_workers,
            error_tracker=error_tracker,
        )

    if fact_file:
        print("\n[*] PHASE 2b — FACT RECALL EVALUATION")
        fact_eval_result = await run_fact_evaluation_async(
            fact_file=fact_file,
            kg=kg,
            retrieval_params=retrieval_params,
            tokenizer=tokenizer,
            max_context_tokens=max_context_tokens,
            max_workers=max_workers,
            error_tracker=error_tracker,
        )

    error_summary = error_tracker.summary()

    # ---- Save results ----
    final_output = {
        "run_id":    f"hgrag_{uuid.uuid4().hex[:8]}",
        "timestamp": timestamp,
        "system":    "HYPERGRAPHRAG",

        "metadata": {
            "contract_file":      os.path.abspath(contract_path),
            "qa_file":            qa_file,
            "fact_file":          fact_file,
            "llm_model":          model,
            "embedding_model":    embed_model,
            "api_base":           api_base,
            "embed_api_base":     embed_api_base,
            "llm_max_async":      llm_max_async,
            "embedding_max_async": embedding_max_async,
            "llm_max_token_size": llm_max_token_size,
            "llm_max_retries":    llm_max_retries,
            "request_timeout":    request_timeout,
            "retrieval_params":   retrieval_params,
            "max_context_tokens": max_context_tokens,
            "judge_max_tokens":   judge_max_tokens,
            "chunking": {
                "chunk_size":    chunk_size,
                "chunk_overlap": chunk_overlap,
                "max_gleanings": max_gleanings,
            },
            "index_dir": working_dir,
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
    print(f"\n  Index    → {working_dir}")
    print(f"  Log      → {log_path}")
    print(f"  Results  → {results_path}")
    print(f"  Document tokens  : {cm['total_document_tokens']}")
    print(f"  Chunks           : {cm['total_chunks']}  "
          f"({cm['chunk_size']} tok, {cm['chunk_overlap']} overlap, "
          f"{cm['max_gleanings']} gleanings)")
    print(f"  Entities         : {cm['entities']}")
    print(f"  Edges            : {cm['edges']}")
    print(f"  LLM tokens used  : {cm['llm_total_tokens']}  ({cm['llm_api_calls']} calls)")
    print(f"  Wall time        : {cm['execution_time_sec']}s")
    if cm.get("artifact_files"):
        print(f"  Artifacts ({len(cm['artifact_files'])} files):")
        for af in cm["artifact_files"]:
            print(f"    {af['file']:45s}  {af['size_bytes']:>10,} bytes")
    if qa_eval_result:
        print(f"  QA (excl CAT5)   : {qa_eval_result['overall_avg_score_excl_cat5']}")
        for cat, s in qa_eval_result["by_category"].items():
            print(f"    {cat}: {s['avg_score']}  (n={s['count']})")
    if fact_eval_result:
        print(
            f"  Fact Recall      : {fact_eval_result['overall_recall_score']}  "
            f"({fact_eval_result['total_recalled']}/{fact_eval_result['total_facts']})"
        )
    # FIX: was previously `error_summary["by_phase"]` which raised KeyError —
    # ErrorTracker.summary() returns by_section, so use that.
    print(f"\n  Errors (total)   : {error_summary['total_errors']}")
    if error_summary["by_section"]:
        for section, count in error_summary["by_section"].items():
            print(f"    {section}: {count}")

    sys.stdout = tee._stream
    tee.close()


# =============================================================================
# MAIN ASYNC ENTRYPOINT
# =============================================================================
async def main_async(args, tokenizer, shared_kwargs):
    multi_contract_mode = args.all or args.ten_contracts_dataset or args.long_form_contracts

    if multi_contract_mode:
        if args.all:
            subdirs_to_scan = DATASET_SUBDIRS
            mode_label      = "--all"
        elif args.ten_contracts_dataset:
            subdirs_to_scan = ["ten_contracts_dataset"]
            mode_label      = "--ten-contracts-dataset"
        else:
            subdirs_to_scan = ["long_form_contracts"]
            mode_label      = "--long-form-contracts"

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
            try:
                await run_single_contract_async(
                    contract_path=contract_path,
                    qa_file=qa_path,
                    fact_file=fact_path,
                    tokenizer=tokenizer,
                    **shared_kwargs,
                )
            except Exception as e:
                # Don't let one bad contract crash the entire batch run.
                print(f"[FATAL] Contract {contract_path} failed: {e}")
                traceback.print_exc()
                continue
    else:
        await run_single_contract_async(
            contract_path=args.contract,
            qa_file=args.qa_file,
            fact_file=args.fact_file,
            tokenizer=tokenizer,
            **shared_kwargs,
        )


def main():
    parser = argparse.ArgumentParser(
        description="HyperGraphRAG: single-script construction + QA/Fact evaluation"
    )

    # ---- Mode ----
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
                        help="QA JSON path (single-contract mode)")
    parser.add_argument("--fact-file", type=str, default=None,
                        help="Fact JSON path (single-contract mode)")

    # ---- Directory for multi-contract modes ----
    parser.add_argument("--eval-data-dir", type=str, default=None,
                        help="Root evaluation_data/ directory")

    # ---- Output ----
    parser.add_argument("--out-dir", type=str, default="./results",
                        help="Base directory for all output")

    # ---- LLM model ----
    parser.add_argument("--model",          type=str,
                        default="Qwen/Qwen3-30B-A3B-Instruct-2507-FP8")
    parser.add_argument("--embed-model",    type=str,
                        default="Qwen/Qwen3-Embedding-4B")
    parser.add_argument("--api-base",       type=str,
                        default="http://localhost:8000/v1")
    parser.add_argument("--embed-api-base", type=str,
                        default="http://localhost:8001/v1")
    parser.add_argument("--api-key",        type=str, default="EMPTY")
    parser.add_argument("--embed-dim",      type=int, default=2560)

    # ---- Construction hyperparams ----
    parser.add_argument("--chunk-size",    type=int, default=800)
    parser.add_argument("--chunk-overlap", type=int, default=50)
    parser.add_argument("--max-gleanings", type=int, default=1)
    parser.add_argument("--llm-max-async", type=int, default=4,
                        help="Max concurrent internal LLM calls per HGRAG operation. "
                             "Effective concurrency against vLLM is "
                             "max-workers × llm-max-async — keep the product reasonable.")
    parser.add_argument("--embedding-max-async", type=int, default=8,
                        help="Max concurrent embedding calls. HGRAG's default is 16 but "
                             "embedding bursts are a common cause of vLLM Connection errors "
                             "during retrieval — lower this if you see embedding-side retries.")
    parser.add_argument("--llm-max-token-size", type=int, default=20000)
    parser.add_argument("--llm-max-retries", type=int, default=5,
                        help="Times to retry a failed completion or embedding call before "
                             "giving up. Uses exponential backoff with jitter (1s, 2s, 4s, "
                             "8s, 16s — capped at 30s). Set to 0 to disable.")
    parser.add_argument("--request-timeout", type=float, default=120.0,
                        help="Per-request timeout in seconds for LiteLLM/httpx. Forwarded "
                             "to litellm.acompletion/aembedding so a hung connection trips "
                             "the retry path instead of blocking forever.")
    parser.add_argument("--max-workers",   type=int, default=10,
                        help="asyncio.Semaphore size for QA/Fact evaluation. "
                             "(Previously a ThreadPoolExecutor; now a semaphore so all "
                             "work happens in one event loop, avoiding LiteLLM "
                             "LoggingWorker coroutine-reuse bugs.)")

    # ---- Retrieval hyperparams ----
    parser.add_argument("--retrieval-mode",  type=str, default="hybrid",
                        choices=["hybrid"])
    parser.add_argument("--retrieval-top-k",    type=int, default=10)
    parser.add_argument("--max-context-tokens", type=int, default=2000)

    # ---- Judge LLM ----
    parser.add_argument("--judge-max-tokens", type=int, default=8000)

    args = parser.parse_args()

    # ---- Validate ----
    multi_contract_mode = args.all or args.ten_contracts_dataset or args.long_form_contracts
    if multi_contract_mode and not args.eval_data_dir:
        parser.error("--all, --ten-contracts-dataset, and --long-form-contracts all require --eval-data-dir")
    if args.contract and not args.qa_file and not args.fact_file:
        parser.error("Single-contract mode requires at least one of --qa-file or --fact-file")

    # ---- Shared setup ----
    configure_dspy(max_tokens=args.judge_max_tokens)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    shared_kwargs = dict(
        out_dir=args.out_dir,
        model=args.model,
        embed_model=args.embed_model,
        api_base=args.api_base,
        embed_api_base=args.embed_api_base,
        api_key=args.api_key,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        max_gleanings=args.max_gleanings,
        embed_dim=args.embed_dim,
        llm_max_async=args.llm_max_async,
        embedding_max_async=args.embedding_max_async,
        llm_max_token_size=args.llm_max_token_size,
        llm_max_retries=args.llm_max_retries,
        request_timeout=args.request_timeout,
        max_workers=args.max_workers,
        retrieval_mode=args.retrieval_mode,
        retrieval_top_k=args.retrieval_top_k,
        max_context_tokens=args.max_context_tokens,
        judge_max_tokens=args.judge_max_tokens,
    )

    # ---- ONE event loop for the entire script ----
    asyncio.run(main_async(args, tokenizer, shared_kwargs))


if __name__ == "__main__":
    main()