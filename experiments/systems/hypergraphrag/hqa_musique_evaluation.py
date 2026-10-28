#!/usr/bin/env python3
"""
hgrag_multihop_eval.py
======================
HyperGraphRAG evaluation harness for HotpotQA and MuSiQue.

Two modes (--mode):
  construction  Build the global HyperGraphRAG index over corpus.jsonl.
                Auto-resumes from the latest checkpoint if one exists.
  evaluation    Run QA answer generation over questions.jsonl using the
                built index, scoring EM/F1 with official dataset metrics.

DIRECTORY LAYOUT
----------------
  run_dir/
      hgrag_index/              <- HyperGraphRAG working_dir (auto-managed,
      |                            always the live current state on disk)
      checkpoint-1/
          metadata.json
          absorbed_paragraph_ids.json
      checkpoint-2/
          ...
      final-checkpoint/
          metadata.json
          absorbed_paragraph_ids.json
      evaluation_results.json

CHECKPOINT DESIGN
-----------------
HyperGraphRAG writes its full index to disk automatically via JsonKVStorage,
NanoVectorDBStorage, and NetworkXStorage — there is NO graph.json export step.
The hgrag_index/ directory IS the checkpoint for graph state.  Our checkpoint-N
folders only track metadata + absorbed_paragraph_ids so we know which paragraphs
to skip on resume.  On restart, HyperGraphRAG is pointed at the same working_dir
and its internal filter_keys deduplication prevents re-inserting known docs.

PROVIDERS (--provider)
----------------------
  local_vllm (default)
      model    : Qwen/Qwen3-30B-A3B-Instruct-2507-FP8
      api-base : http://localhost:8000/v1
      api-key  : EMPTY
  deepseek
      model    : deepseek-v4-flash  (current prod model, May 2026)
      api-base : https://api.deepseek.com
      api-key  : DEEPSEEK_API_KEY env var  or  --api-key
  Override any field: --model / --api-base / --api-key

THINKING MODE (--thinking / --reasoning-effort)
-----------------------------------------------
Disabled by default.  When --thinking is set:
  DeepSeek : passes extra_body={"thinking": {"type": "enabled"}} to the API.
             Reasoning is returned in reasoning_content (separate field),
             so .content is always the clean final answer.

RATE LIMITING (--construction-rpm / --eval-rpm)
-----------------------------------------------
HyperGraphRAG has no native rate limiting.  This script inserts paragraphs
sequentially and wraps each ainsert / aquery with an async token-bucket limiter.
  0 = disabled (default; fine for local vLLM).
  Recommended for DeepSeek free tier: 10 (construction), 50 (eval).
  Note: each ainsert makes ~5-10 internal LLM calls.  10 RPM here
  means ~10 paragraphs/minute, i.e. 1-2 hours for 1000 paragraphs.

EXAMPLES
--------
  # Local vLLM, no rate limit
  python hgrag_multihop_eval.py \\
      --mode construction --provider local_vllm \\
      --dataset hotpotqa --corpus-file ./data/hotpot_corpus_1k.jsonl \\
      --run-dir ./results/hotpotqa/hgrag/run1 \\
      --checkpoint-interval 50 --llm-max-async 4

  # DeepSeek v4 flash, 10 RPM construction
  python hgrag_multihop_eval.py \\
      --mode construction --provider deepseek \\
      --dataset hotpotqa --corpus-file ./data/hotpot_corpus_1k.jsonl \\
      --run-dir ./results/hotpotqa/hgrag/run1_ds \\
      --checkpoint-interval 25 --construction-rpm 10

  # Evaluation with thinking enabled
  python hgrag_multihop_eval.py \\
      --mode evaluation --provider deepseek \\
      --dataset hotpotqa --questions-file ./data/hotpot_questions_1k.jsonl \\
      --run-dir ./results/hotpotqa/hgrag/run1_ds \\
      --thinking --reasoning-effort medium --eval-rpm 50
"""

import os, sys, json, re, time, uuid, asyncio, threading, argparse, traceback, shutil, random, resource
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

import litellm
from transformers import AutoTokenizer

from experiments.evaluate_mine import (
    configure_dspy,
    ask_llm_query,
    ask_llm_query_short,
)

# =============================================================================
# DISABLE LITELLM ASYNC LOGGING WORKER (required to avoid coroutine-reuse
# RuntimeError across asyncio.run() calls — see insurance eval for full notes)
# =============================================================================
litellm.success_callback        = []
litellm.failure_callback        = []
litellm._async_success_callback = []
litellm._async_failure_callback = []
litellm.callbacks               = []
litellm.input_callback          = []
litellm.service_callback        = []

try:
    from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
    def _noop(*_a, **_kw): return None
    async def _noop_async(*_a, **_kw): return None
    GLOBAL_LOGGING_WORKER.ensure_initialized = _noop
    GLOBAL_LOGGING_WORKER.enqueue            = _noop
    GLOBAL_LOGGING_WORKER.start              = _noop
    GLOBAL_LOGGING_WORKER._worker_loop       = _noop_async
    GLOBAL_LOGGING_WORKER._task              = None
    GLOBAL_LOGGING_WORKER._queue             = None
except (ImportError, AttributeError):
    pass

from hypergraphrag import HyperGraphRAG
from hypergraphrag.base import QueryParam

from experiments.utils.multihop_utils import (
    HOTPOTQA, MUSIQUE, SUPPORTED_DATASETS,
    load_corpus_jsonl, load_questions_jsonl,
    find_all_checkpoints, find_latest_checkpoint, find_final_checkpoint,
    next_checkpoint_num, FINAL_CHECKPOINT_NAME,
    score_em, score_f1, truncate_tokens, safe_div, add_token_counters,
    next_eval_results_path,
)

# =============================================================================
# PROVIDER CONFIGS
# =============================================================================

@dataclass
class ProviderConfig:
    provider: str
    model:    str
    api_base: str
    api_key:  str


PROVIDER_DEFAULTS = {
    "local_vllm": ProviderConfig(
        provider="local_vllm",
        model="Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
        api_base="http://localhost:8000/v1",
        api_key="EMPTY",
    ),
    "deepseek": ProviderConfig(
        # deepseek-v4-flash: current production model (May 2026).
        # Legacy alias deepseek-chat retires 2026-07-24.
        provider="deepseek",
        model="deepseek-v4-flash",
        api_base="https://api.deepseek.com",
        api_key="",
    ),
}


def resolve_provider(args) -> ProviderConfig:
    base     = PROVIDER_DEFAULTS[args.provider]
    model    = args.model    or base.model
    api_base = args.api_base or base.api_base
    api_key  = args.api_key  or base.api_key
    if args.provider == "deepseek" and not api_key:
        api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        if not api_key:
            print("[WARN] provider=deepseek: no api-key found. "
                  "Set DEEPSEEK_API_KEY env var or pass --api-key.")
    return ProviderConfig(provider=args.provider, model=model,
                          api_base=api_base, api_key=api_key)


def litellm_model_string(cfg: ProviderConfig) -> str:
    """
    LiteLLM routing string.
    local_vllm: 'openai/<model>' (non-OpenAI compat endpoint prefix)
    deepseek:   'deepseek/<model>' (provider prefix required by LiteLLM)
    """
    if cfg.provider == "local_vllm":
        return f"openai/{cfg.model}"
    if cfg.provider == "deepseek":
        return f"deepseek/{cfg.model}"
    return cfg.model


# =============================================================================
# THINKING MODE
# Per DeepSeek docs: pass extra_body={"thinking": {"type": "enabled"}} via
# the OpenAI SDK (or LiteLLM's extra_body kwarg).  DeepSeek returns reasoning
# in response.choices[0].message.reasoning_content (separate from .content),
# so .content is always the clean final answer — no stripping needed.
# When thinking is OFF, extra_body is simply not passed.
# =============================================================================

def thinking_extra_body(thinking_enabled: bool) -> dict:
    """Returns the extra_body dict for LiteLLM when thinking is on, else {}."""
    if not thinking_enabled:
        return {}
    return {"extra_body": {"thinking": {"type": "enabled"}}}


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

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "prompt_tokens":     self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens":      self.total_tokens,
                "api_calls":         self.api_calls,
            }

    def add_snapshot(self, snap: dict) -> dict:
        """Merge a snapshot dict into this tracker and return new cumulative snapshot."""
        with self._lock:
            self.prompt_tokens     += snap.get("prompt_tokens", 0)
            self.completion_tokens += snap.get("completion_tokens", 0)
            self.api_calls         += snap.get("api_calls", 0)
            return self.snapshot()

# =============================================================================
# RETRY HELPER (verbatim from insurance eval — handles transient vLLM / API errors)
# =============================================================================

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
# ASYNC RATE LIMITER (token-bucket, event-loop-friendly)
# =============================================================================

class AsyncRateLimiter:
    """
    Async token-bucket rate limiter.  rpm=0 disables entirely.

    Bucket starts full (allows initial burst).  Refills at rpm/60 tokens/sec.
    Each acquire() suspends (asyncio.sleep) without blocking the event loop.

    Use as async context manager:   async with limiter: await api_call()
    """
    def __init__(self, rpm: int = 0):
        self.rpm      = rpm
        self.disabled = rpm <= 0
        if not self.disabled:
            self._cap  = float(rpm)
            self._tok  = float(rpm)
            self._rate = rpm / 60.0
            self._last = time.monotonic()
            self._lock = asyncio.Lock()

    async def acquire(self):
        if self.disabled:
            return
        while True:
            async with self._lock:
                now = time.monotonic()
                self._tok  = min(self._cap, self._tok + (now - self._last) * self._rate)
                self._last = now
                if self._tok >= 1.0:
                    self._tok -= 1.0
                    return
            await asyncio.sleep(1.0 / self._rate)

    async def __aenter__(self):  await self.acquire(); return self
    async def __aexit__(self, *_): pass
    def __repr__(self):
        return "AsyncRateLimiter(disabled)" if self.disabled \
               else f"AsyncRateLimiter(rpm={self.rpm})"


# =============================================================================
# LLM WRAPPER FACTORY
# Returns (complete_fn, embed_fn) for a given provider + thinking config.
# complete_fn is passed to HyperGraphRAG as llm_model_func.
# embed_fn    is passed to HyperGraphRAG as embedding_func.
# Both are also used directly in the evaluation answer-generation call.
# =============================================================================

def make_llm_wrappers(
    cfg: ProviderConfig,
    tracker: AsyncTokenTracker,
    thinking_enabled: bool,
    embed_model: str,
    embed_backend: str,      # "vllm" or "ollama"
    embed_api_base: str,
    embed_dim: int,
    llm_max_retries: int,
    request_timeout: float,
    rate_limiter: "AsyncRateLimiter | None" = None,
):
    """
    Returns (complete_fn, embed_fn).

    complete_fn : HyperGraphRAG's internal LLM for entity extraction / queries.
                  Signature: async (prompt, **kwargs) -> str.
    embed_fn    : HyperGraphRAG's embedding function.
                  Signature: async (texts: list, **kwargs) -> list[list[float]].

    Answer generation is handled separately via DSPy (ask_llm_query /
    ask_llm_query_short from experiments.evaluate_mine) so it uses the same
    prompts and signatures as the KGGen evaluation harness.

    Thinking (DeepSeek only):
        extra_body={"thinking": {"type": "enabled"}} is passed when --thinking
        is set.  DeepSeek returns reasoning in reasoning_content (separate field),
        so .content is always the clean answer — no stripping needed.

    Embedding backends:
        vllm  : model string prefixed with 'openai/', hits embed_api_base.
        ollama: model string as-is, hits http://127.0.0.1:11434/v1 with
                api_key='ollama' (standard Ollama OpenAI-compat endpoint).
    """
    model_str = litellm_model_string(cfg)
    xbody     = thinking_extra_body(thinking_enabled)   # {} or extra_body dict

    # ---- LLM completion wrapper ----
    async def complete_fn(prompt: str, **kwargs) -> str:
        if rate_limiter is not None:
            await rate_limiter.acquire()
        def _factory():
            return litellm.acompletion(
                model=model_str,
                messages=[{"role": "user", "content": prompt}],
                api_base=cfg.api_base,
                api_key=cfg.api_key,
                temperature=0.0,
                max_tokens=kwargs.get("max_tokens", 4096),
                timeout=request_timeout,
                **xbody,
            )
        response = await _with_retries(_factory, "completion", llm_max_retries)
        tracker.record(getattr(response, "usage", None))
        return response.choices[0].message.content or ""

    # ---- Embedding function ----
    # vllm  : LiteLLM needs 'openai/<model>' prefix for non-OpenAI compat endpoints.
    # ollama: bare model name, fixed base URL + api_key per Ollama's OpenAI compat API.
    if embed_backend == "ollama":
        # Ollama serves an OpenAI-compatible /v1 endpoint — same pattern as vllm.
        # LiteLLM needs the 'openai/' prefix to route it as an OpenAI-compat call.
        embed_str      = f"openai/{embed_model}"   # e.g. "openai/embeddinggemma:300m"
        embed_base_url = "http://127.0.0.1:11434/v1"
        embed_key      = "ollama"
    else:  # vllm
        embed_str      = f"openai/{embed_model}"
        embed_base_url = embed_api_base
        embed_key      = cfg.api_key

    async def embed_fn(texts: list, **kwargs) -> list:
        def _factory():
            return litellm.aembedding(
                model=embed_str,
                input=texts,
                api_base=embed_base_url,
                api_key=embed_key,
                timeout=request_timeout,
            )
        response = await _with_retries(_factory, "embedding", llm_max_retries)
        tracker.record(getattr(response, "usage", None))
        return [d["embedding"] for d in response.data]

    embed_fn.embedding_dim = embed_dim

    return complete_fn, embed_fn

# =============================================================================
# HGRAG-SPECIFIC CHECKPOINT FUNCTIONS
# Unlike KGGen, there is NO graph export step.  The HyperGraphRAG working_dir
# is always up-to-date on disk.  Checkpoint folders only store metadata +
# absorbed_paragraph_ids so we know what to skip on resume.
# =============================================================================

def hgrag_save_checkpoint(
    cp_dir: str,
    metadata: dict,
    absorbed_paragraph_ids: list,
) -> str:
    os.makedirs(cp_dir, exist_ok=True)
    with open(os.path.join(cp_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)
    with open(os.path.join(cp_dir, "absorbed_paragraph_ids.json"),
              "w", encoding="utf-8") as f:
        json.dump(absorbed_paragraph_ids, f, indent=2)
    return cp_dir


def hgrag_load_checkpoint(cp_dir: str):
    """Returns (metadata, absorbed_paragraph_ids)."""
    with open(os.path.join(cp_dir, "metadata.json"), encoding="utf-8") as f:
        metadata = json.load(f)
    with open(os.path.join(cp_dir, "absorbed_paragraph_ids.json"),
              encoding="utf-8") as f:
        absorbed = json.load(f)
    return metadata, absorbed


# =============================================================================
# CONSTRUCTION
# =============================================================================

async def run_construction(args, cfg: ProviderConfig):
    """
    Builds a global HyperGraphRAG index over corpus.jsonl, checkpointing
    every --checkpoint-interval paragraphs.  Auto-resumes if a checkpoint
    already exists in run_dir.
    """
    os.makedirs(args.run_dir, exist_ok=True)

    if find_final_checkpoint(args.run_dir):
        print("[!] final-checkpoint already exists. Delete it to rebuild.")
        return

    # HyperGraphRAG's working_dir lives inside run_dir.  It is created once
    # and persists across resume runs — it IS the live graph state.
    hgrag_index_dir = os.path.join(args.run_dir, "hgrag_index")

    print(f"[Construction] Loading corpus: {args.corpus_file}")
    corpus = load_corpus_jsonl(args.corpus_file)
    print(f"[Construction]   {len(corpus):,} paragraphs")

    # ---- Resume state ----
    latest = find_latest_checkpoint(args.run_dir)
    if latest:
        print(f"[Construction] Resuming from: {latest}")
        prev_meta, absorbed_ids = hgrag_load_checkpoint(latest)
        cum_tokens = prev_meta.get(
            "cumulative_construction_tokens",
            {"prompt_tokens": 0, "completion_tokens": 0,
             "total_tokens": 0, "api_calls": 0})
        cum_time = float(prev_meta.get("cumulative_construction_time_sec", 0.0))
        print(f"[Construction]   {len(absorbed_ids):,} paragraphs already absorbed")

        # Restore hgrag_index from the checkpoint's snapshot.
        # This guards against a corrupted hgrag_index/ left by a crash mid-write:
        # the snapshot in checkpoint-N/ is always the last fully-flushed state.
        cp_index_snapshot = os.path.join(latest, "hgrag_index")
        if os.path.isdir(cp_index_snapshot):
            print(f"[Construction] Restoring hgrag_index from checkpoint snapshot ...")
            if os.path.exists(hgrag_index_dir):
                shutil.rmtree(hgrag_index_dir)
            shutil.copytree(cp_index_snapshot, hgrag_index_dir)
            print(f"[Construction]   hgrag_index restored from {cp_index_snapshot}")
        else:
            print(f"[Construction WARN] Checkpoint has no hgrag_index snapshot "
                  f"(old-style checkpoint); using existing hgrag_index/ as-is.")
    else:
        print("[Construction] Starting fresh.")
        absorbed_ids = []
        cum_tokens   = {"prompt_tokens": 0, "completion_tokens": 0,
                        "total_tokens": 0, "api_calls": 0}
        cum_time     = 0.0

    absorbed_set = set(absorbed_ids)
    remaining    = [p for pid, p in sorted(corpus.items())
                    if pid not in absorbed_set]
    print(f"[Construction] {len(remaining):,} paragraphs remaining "
          f"({len(absorbed_set):,}/{len(corpus):,} already absorbed)")

    # ---- Token tracker for this run (will accumulate across all blocks) ----
    run_tracker = AsyncTokenTracker()

    construction_limiter = AsyncRateLimiter(rpm=args.construction_rpm)
    print(f"[Construction] {construction_limiter}")

    # ---- Build LLM wrappers ----
    complete_fn, embed_fn = make_llm_wrappers(
        cfg=cfg,
        tracker=run_tracker,
        thinking_enabled=args.thinking,
        embed_model=args.embed_model,
        embed_backend=args.embed_backend,
        embed_api_base=args.embed_api_base,
        embed_dim=args.embed_dim,
        llm_max_retries=args.llm_max_retries,
        request_timeout=args.request_timeout,
        rate_limiter=construction_limiter,
    )

    # ---- Create (or reopen) HyperGraphRAG ----
    # Pointing at the same working_dir lets HyperGraphRAG see existing data.
    kg = HyperGraphRAG(
        working_dir=hgrag_index_dir,
        llm_model_func=complete_fn,
        llm_model_name=cfg.model,
        llm_model_max_token_size=args.llm_max_token_size,
        llm_model_max_async=args.llm_max_async,
        embedding_func=embed_fn,
        embedding_batch_num=32,
        embedding_func_max_async=args.embedding_max_async,
        chunk_token_size=args.chunk_size,
        chunk_overlap_token_size=args.chunk_overlap,
        tiktoken_model_name="gpt-4o-mini",
        kv_storage="JsonKVStorage",
        vector_storage="NanoVectorDBStorage",
        graph_storage="NetworkXStorage",
        entity_extract_max_gleaning=args.max_gleanings,
        enable_llm_cache=False,
    )

    # Rate control for construction is handled internally by HyperGraphRAG via
    # llm_model_max_async.  Transient 429s are caught by _with_retries() inside
    # complete_fn.  No external per-paragraph limiter is used here.
    if args.thinking:
        print("[Construction] Thinking ON")

    ckpt_num = next_checkpoint_num(args.run_dir)

    while remaining:
        block     = remaining[:args.checkpoint_interval]
        remaining = remaining[args.checkpoint_interval:]

        print(f"\n[Construction] --- checkpoint-{ckpt_num}: "
              f"{len(block)} paragraphs ---")
        t0 = time.time()

        blk_start_tokens = run_tracker.snapshot()

        # ── Batch insert ──────────────────────────────────────────────────────
        # Pass the entire block as a list in ONE ainsert() call.
        #
        # Why this matters for correctness:
        #   ainsert(list) feeds all docs into a single extract_entities() pass.
        #   Entity extraction runs concurrently per chunk INSIDE the library
        #   (bounded by llm_model_max_async), then ALL results are collected
        #   before any _merge_nodes_then_upsert() is called.  This means an
        #   entity like "Barack Obama" seen across 10 paragraphs in the block
        #   gets ALL its descriptions accumulated and merged in one shot.
        #
        #   Calling ainsert() concurrently on individual paragraphs (the old
        #   approach) created a read-modify-write race: two concurrent calls
        #   could both read "entity X not in graph yet", independently call the
        #   LLM for a summary, then overwrite each other — losing descriptions.
        #
        # _insert_done() is called by ainsert()'s own finally block once the
        # call completes; we do NOT call it manually.
        block_pids  = [p["id"]           for p in block]
        block_texts = [p["full_context"] for p in block]

        print(f"[Construction]   Inserting batch of {len(block_texts)} paragraphs ...")
        insert_ok = False
        for attempt in range(args.llm_max_retries + 1):
            try:
                await kg.ainsert(block_texts)
                insert_ok = True
                break
            except Exception as e:
                if attempt < args.llm_max_retries:
                    delay = min(2.0 * (2 ** attempt), 60.0)
                    print(f"[Construction WARN] Batch attempt "
                          f"{attempt + 1}/{args.llm_max_retries + 1} failed: "
                          f"{type(e).__name__}: {e} — retrying in {delay:.1f}s")
                    await asyncio.sleep(delay)
                else:
                    print(f"[Construction ERROR] Batch insert failed after "
                          f"{args.llm_max_retries + 1} attempts — block skipped.")

        new_absorbed = block_pids if insert_ok else []

        snap = run_tracker.snapshot()
        print(f"[Construction]   Batch {'OK' if insert_ok else 'FAILED (skipped)'}. "
              f" tokens_this_run={snap['total_tokens']:,}")

        # Compute per-block token delta
        end_snap   = run_tracker.snapshot()
        blk_tokens = {
            k: end_snap[k] - blk_start_tokens[k]
            for k in ("prompt_tokens", "completion_tokens", "total_tokens", "api_calls")
        }
        blk_time = round(time.time() - t0, 2)

        absorbed_ids.extend(new_absorbed)
        cum_tokens = add_token_counters(cum_tokens, blk_tokens)
        cum_time   = round(cum_time + blk_time, 2)

        # Read graph stats
        try:
            nx_g      = kg.chunk_entity_relation_graph._graph
            n_entities = len(nx_g.nodes)
            n_edges    = len(nx_g.edges)
        except Exception:
            n_entities = n_edges = -1

        meta = {
            "checkpoint_id":                    f"checkpoint-{ckpt_num}",
            "checkpoint_num":                   ckpt_num,
            "timestamp":                        datetime.now().isoformat(),
            "dataset":                          args.dataset,
            "provider":                         cfg.provider,
            "model":                            cfg.model,
            "embedding_model":                  args.embed_model,
            "embedding_backend":                args.embed_backend,
            "api_base":                         cfg.api_base,
            "hgrag_index_dir":                  os.path.abspath(hgrag_index_dir),
            "corpus_file":                      os.path.abspath(args.corpus_file),
            "total_paragraphs_in_corpus":       len(corpus),
            "paragraphs_absorbed_so_far":       len(absorbed_ids),
            "paragraphs_in_this_block":         len(new_absorbed),
            "paragraphs_failed_in_block":       len(block) - len(new_absorbed),
            "cumulative_graph_entities":        n_entities,
            "cumulative_graph_edges":           n_edges,
            "cumulative_construction_tokens":   cum_tokens,
            "this_block_tokens":                blk_tokens,
            "cumulative_construction_time_sec": cum_time,
            "this_block_time_sec":              blk_time,
            "construction_params": {
                "checkpoint_interval":  args.checkpoint_interval,
                "llm_max_async":        args.llm_max_async,
                "embedding_max_async":  args.embedding_max_async,
                "chunk_size":           args.chunk_size,
                "chunk_overlap":        args.chunk_overlap,
                "max_gleanings":        args.max_gleanings,
                "construction_rpm":     args.construction_rpm,
                "thinking":             args.thinking,
                "delete_old_checkpoints": args.delete_old_checkpoints,
            },
        }

        cp_name = FINAL_CHECKPOINT_NAME if not remaining else f"checkpoint-{ckpt_num}"
        cp_dir  = os.path.join(args.run_dir, cp_name)
        os.makedirs(cp_dir, exist_ok=True)

        # ── Step 1: snapshot hgrag_index into the checkpoint directory ────────
        # This MUST happen before writing metadata and before deleting any old
        # checkpoint, so that the new checkpoint-N folder is a fully self-
        # contained, restorable unit from the moment it is created.
        cp_index_dir = os.path.join(cp_dir, "hgrag_index")
        t_snap = time.time()
        if os.path.exists(cp_index_dir):
            shutil.rmtree(cp_index_dir)
        shutil.copytree(hgrag_index_dir, cp_index_dir)
        snap_secs = round(time.time() - t_snap, 2)
        print(f"[Construction] Snapshotted hgrag_index → {cp_index_dir}  ({snap_secs}s)")

        # ── Step 2: write metadata + absorbed_ids ────────────────────────────
        # The checkpoint is now a complete, self-contained unit on disk.
        hgrag_save_checkpoint(cp_dir, meta, absorbed_ids)
        print(f"[Construction] Saved {cp_dir}")
        print(f"[Construction]   absorbed={len(absorbed_ids):,}/{len(corpus):,}  "
              f"entities={n_entities}  edges={n_edges}  "
              f"tokens={cum_tokens['total_tokens']:,}  time={cum_time}s")

        # ── Step 3: ONLY NOW delete old checkpoints ───────────────────────────
        # The new checkpoint (including its hgrag_index snapshot) is fully on
        # disk, so it is safe to discard the previous one.
        if args.delete_old_checkpoints and not (cp_name == FINAL_CHECKPOINT_NAME):
            for old_num, old_path in find_all_checkpoints(args.run_dir):
                if old_num < ckpt_num:
                    shutil.rmtree(old_path, ignore_errors=True)
                    print(f"[Construction] Deleted old checkpoint: {old_path}")

        ckpt_num += 1

    # If while loop finished without writing final-checkpoint (already done above
    # when remaining became empty mid-loop), ensure it exists.
    if not find_final_checkpoint(args.run_dir):
        latest   = find_latest_checkpoint(args.run_dir)
        if latest:
            last_meta, last_abs = hgrag_load_checkpoint(latest)
            final_meta = {
                **last_meta,
                "checkpoint_id":  FINAL_CHECKPOINT_NAME,
                "checkpoint_num": None,
                "timestamp":      datetime.now().isoformat(),
                "is_final":       True,
            }
            hgrag_save_checkpoint(
                os.path.join(args.run_dir, FINAL_CHECKPOINT_NAME),
                final_meta, last_abs,
            )

    final_cp = find_final_checkpoint(args.run_dir)
    print(f"\n[Construction] FINAL -> {final_cp}")
    print(f"[Construction]   hgrag_index -> {hgrag_index_dir}")
    print(f"[Construction]   total tokens: {cum_tokens['total_tokens']:,}  "
          f"time: {cum_time}s")

# =============================================================================
# EVALUATION
# =============================================================================

async def _eval_one_question(
    q: dict, kg: HyperGraphRAG,
    tokenizer, args, cfg: ProviderConfig,
    limiter: AsyncRateLimiter, eval_tracker: AsyncTokenTracker,
) -> dict:
    qid      = q.get("question_id", "")
    question = q["question"]
    gold     = q["answer"]
    aliases  = q.get("answer_aliases", [])
    qtype    = q.get("type", "unknown")
    qlevel   = q.get("level", None)

    llm_ans = raw_ctx = trunc_ctx = ""
    em = f1 = prec = rec = 0.0
    ctx_tok = 0
    status  = "SUCCESS"
    err     = ""

    loop = asyncio.get_running_loop()

    try:
        # 1. Retrieve context via HyperGraphRAG hybrid query
        async with limiter:
            param   = QueryParam(
                mode="hybrid",
                top_k=args.retrieval_top_k,
                only_need_context=True,
            )
            raw_ctx = await kg.aquery(query=question, param=param)
            raw_ctx = raw_ctx if isinstance(raw_ctx, str) else str(raw_ctx)

        # 2. Truncate to token budget
        trunc_ctx, ctx_tok = truncate_tokens(raw_ctx, tokenizer, args.max_context_tokens)

        # 3. Generate answer via DSPy (same signatures as KGGen evaluation).
        #    DSPy calls are synchronous; run_in_executor keeps the event loop free.
        if args.short_answer:
            llm_ans = await loop.run_in_executor(
                None, ask_llm_query_short, question, trunc_ctx
            )
        else:
            llm_ans = await loop.run_in_executor(
                None, ask_llm_query, question, trunc_ctx, args.chain_of_thought
            )

        # 4. Score with official EM / F1 metrics
        em            = score_em(llm_ans, gold, aliases=aliases, dataset=args.dataset)
        f1, prec, rec = score_f1(llm_ans, gold, aliases=aliases, dataset=args.dataset)

    except Exception as e:
        status = "ERROR"
        err    = str(e)
        print(f"[Eval ERROR] {qid}: {e}")

    return {
        "question_id":              qid,
        "question":                 question,
        "expected_answer":          gold,
        "expected_answer_aliases":  aliases,
        "type":                     qtype,
        "level":                    qlevel,
        "llm_answer":               llm_ans,
        "em":                       float(em),
        "f1":                       float(f1),
        "precision":                float(prec),
        "recall":                   float(rec),
        "retrieved_context":        trunc_ctx,
        "raw_context_length_chars": len(raw_ctx),
        "context_tokens":           ctx_tok,
        "max_context_tokens":       args.max_context_tokens,
        "status":                   status,
        **({"error": err} if err else {}),
    }


def _aggregate_by_key(results: list, key: str) -> dict:
    groups = defaultdict(list)
    for r in results:
        if r["status"] == "SUCCESS" and r.get(key) is not None:
            groups[r[key]].append(r)
    return {
        str(k): {
            "count":     len(rs),
            "em":        safe_div(sum(r["em"]        for r in rs), len(rs)),
            "f1":        safe_div(sum(r["f1"]        for r in rs), len(rs)),
            "precision": safe_div(sum(r["precision"] for r in rs), len(rs)),
            "recall":    safe_div(sum(r["recall"]    for r in rs), len(rs)),
        }
        for k, rs in groups.items()
    }


async def run_evaluation(args, cfg: ProviderConfig):
    """
    Loads the final-checkpoint, reopens HyperGraphRAG from hgrag_index/,
    and runs async QA evaluation with semaphore-bounded concurrency.
    """
    final_cp = find_final_checkpoint(args.run_dir)
    if not final_cp:
        print("[Evaluation FATAL] No final-checkpoint found. "
              "Run --mode construction first.")
        sys.exit(1)

    final_meta, absorbed_ids = hgrag_load_checkpoint(final_cp)
    hgrag_index_dir = final_meta.get(
        "hgrag_index_dir",
        os.path.join(args.run_dir, "hgrag_index"),
    )
    print(f"[Evaluation] Loading HyperGraphRAG from: {hgrag_index_dir}")
    print(f"[Evaluation]   {len(absorbed_ids):,} paragraphs were absorbed")

    # ---- Rebuild LLM wrappers (complete_fn + embed_fn for HGRAG queries) ----
    eval_tracker = AsyncTokenTracker()
    complete_fn, embed_fn = make_llm_wrappers(
        cfg=cfg,
        tracker=eval_tracker,
        thinking_enabled=args.thinking,
        embed_model=args.embed_model,
        embed_backend=args.embed_backend,
        embed_api_base=args.embed_api_base,
        embed_dim=args.embed_dim,
        llm_max_retries=args.llm_max_retries,
        request_timeout=args.request_timeout,
    )

    # ---- Reopen HyperGraphRAG (points at same working_dir — reads existing data) ----
    kg = HyperGraphRAG(
        working_dir=hgrag_index_dir,
        llm_model_func=complete_fn,
        llm_model_name=cfg.model,
        llm_model_max_token_size=args.llm_max_token_size,
        llm_model_max_async=args.llm_max_async,
        embedding_func=embed_fn,
        embedding_batch_num=32,
        embedding_func_max_async=args.embedding_max_async,
        chunk_token_size=args.chunk_size,
        chunk_overlap_token_size=args.chunk_overlap,
        tiktoken_model_name="gpt-4o-mini",
        kv_storage="JsonKVStorage",
        vector_storage="NanoVectorDBStorage",
        graph_storage="NetworkXStorage",
        entity_extract_max_gleaning=args.max_gleanings,
        enable_llm_cache=False,
    )

    # ---- Load questions ----
    print(f"[Evaluation] Loading questions: {args.questions_file}")
    questions = load_questions_jsonl(args.questions_file)
    print(f"[Evaluation]   {len(questions):,} questions")

    # ---- Tokenizer ----
    try:
        tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    except Exception:
        import tiktoken

        class _TiktokenWrapper:
            def __init__(self, enc):
                self._enc = enc
            def encode(self, text, **kwargs):
                return self._enc.encode(text)
            def decode(self, ids, **kwargs):
                return self._enc.decode(ids)

        tokenizer = _TiktokenWrapper(tiktoken.get_encoding("cl100k_base"))
        print("[Evaluation] HuggingFace tokenizer unavailable; "
              "falling back to tiktoken cl100k_base.")

    # ---- Configure DSPy for answer generation ----
    # Uses the same signatures and prompts as the KGGen evaluation harness
    # (ShortAnswerQuestion or AnswerQuestion from experiments.evaluate_mine).
    configure_dspy(
        model=litellm_model_string(cfg),
        api_base=cfg.api_base,
        api_key=cfg.api_key,
    )
    if args.short_answer:
        ans_label = "ShortAnswer (span-style, EM/F1 optimised)"
    elif args.chain_of_thought:
        ans_label = "ChainOfThought (sentence-style)"
    else:
        ans_label = "Predict (sentence-style)"
    print(f"[Evaluation] DSPy configured  model={litellm_model_string(cfg)}  "
          f"answer_module={ans_label}")

    # ---- Rate limiter and concurrency semaphore ----
    limiter = AsyncRateLimiter(rpm=args.eval_rpm)
    sem     = asyncio.Semaphore(args.max_workers)
    print(f"[Evaluation] {limiter}  concurrency={args.max_workers}")
    if args.thinking:
        print("[Evaluation] Thinking ON")

    eval_t0  = time.time()
    results  = []
    done_cnt = {"n": 0}

    async def _bounded(q):
        async with sem:
            r = await _eval_one_question(
                q, kg, tokenizer, args, cfg, limiter, eval_tracker
            )
        done_cnt["n"] += 1
        n  = done_cnt["n"]
        if n % max(1, len(questions) // 20) == 0 or n == len(questions):
            ok = [x for x in results if x["status"] == "SUCCESS"]
            nok = len(ok) + (1 if r["status"] == "SUCCESS" else 0)
            cur_em = safe_div(sum(x["em"] for x in ok) + r.get("em", 0.0), nok)
            cur_f1 = safe_div(sum(x["f1"] for x in ok) + r.get("f1", 0.0), nok)
            print(f"[Evaluation] {n}/{len(questions)}  "
                  f"EM={cur_em:.4f}  F1={cur_f1:.4f}")
        return r

    results = list(await asyncio.gather(*[_bounded(q) for q in questions]))

    eval_time   = round(time.time() - eval_t0, 2)
    eval_tokens = eval_tracker.snapshot()

    ok  = [r for r in results if r["status"] == "SUCCESS"]
    bad = [r for r in results if r["status"] != "SUCCESS"]
    n   = len(ok)

    overall  = {
        "em":                    safe_div(sum(r["em"]                    for r in ok), n),
        "f1":                    safe_div(sum(r["f1"]                    for r in ok), n),
        "precision":             safe_div(sum(r["precision"]             for r in ok), n),
        "recall":                safe_div(sum(r["recall"]                for r in ok), n),
        "avg_context_tokens":    safe_div(sum(r["context_tokens"]        for r in ok), n),
        "avg_raw_context_chars": safe_div(sum(r["raw_context_length_chars"] for r in ok), n),
    }
    by_type  = _aggregate_by_key(ok, "type")
    by_level = _aggregate_by_key(ok, "level") if args.dataset == HOTPOTQA else None

    output = {
        "run_id":    f"hgrag_eval_{uuid.uuid4().hex[:8]}",
        "timestamp": datetime.now().isoformat(),
        "system":    "HYPERGRAPHRAG",
        "dataset":   args.dataset,
        "metadata": {
            "questions_file":          os.path.abspath(args.questions_file),
            "run_dir":                 os.path.abspath(args.run_dir),
            "hgrag_index_dir":         os.path.abspath(hgrag_index_dir),
            "final_checkpoint_path":   os.path.abspath(final_cp),
            "provider":                cfg.provider,
            "model":                   cfg.model,
            "embedding_model":         args.embed_model,
            "embedding_backend":       args.embed_backend,
            "api_base":                cfg.api_base,
            "retrieval_params": {
                "mode":  "hybrid",
                "top_k": args.retrieval_top_k,
            },
            "max_context_tokens":      args.max_context_tokens,
            "answer_module":           ans_label,
            "short_answer":            args.short_answer,
            "chain_of_thought":        args.chain_of_thought,
            "max_workers":             args.max_workers,
            "eval_rpm":                args.eval_rpm,
            "thinking":                args.thinking,
        },
        "construction":  final_meta,
        "evaluation": {
            "total_questions":     len(questions),
            "successful":          n,
            "errored":             len(bad),
            "evaluation_time_sec": eval_time,
            "evaluation_tokens":   eval_tokens,
            "overall":             overall,
            "by_type":             by_type,
            **({"by_level": by_level} if by_level else {}),
            "results":             results,
        },
        "totals": {
            "all_tokens": add_token_counters(
                final_meta.get("cumulative_construction_tokens", {}),
                eval_tokens,
            ),
            "all_time_sec": round(
                float(final_meta.get("cumulative_construction_time_sec", 0.0))
                + eval_time, 2
            ),
        },
    }

    out_path = next_eval_results_path(args.run_dir)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n[Evaluation] DONE in {eval_time}s  ->  {out_path}")
    print(f"[Evaluation]   EM={overall['em']:.4f}  F1={overall['f1']:.4f}  "
          f"n={n}  errors={len(bad)}")
    print(f"[Evaluation]   avg_context_tokens={overall['avg_context_tokens']:.0f}  "
          f"avg_raw_chars={overall['avg_raw_context_chars']:.0f}  "
          f"max_context_tokens={args.max_context_tokens}")
    print(f"[Evaluation]   answer_module={ans_label}")
    print(f"[Evaluation]   by type:")
    for k, v in sorted(by_type.items()):
        print(f"     {k:>12}: EM={v['em']:.4f}  F1={v['f1']:.4f}  (n={v['count']})")
    if by_level:
        print(f"[Evaluation]   by level:")
        for k, v in sorted(by_level.items()):
            print(f"     {k:>12}: EM={v['em']:.4f}  F1={v['f1']:.4f}  (n={v['count']})")
    print(f"[Evaluation]   tokens={eval_tokens['total_tokens']:,}  "
          f"calls={eval_tokens['api_calls']:,}")

# =============================================================================
# MAIN
# =============================================================================

async def main_async(args, cfg: ProviderConfig):
    if args.mode == "construction":
        try:
            await run_construction(args, cfg)
        except KeyboardInterrupt:
            print("\n[Construction] Interrupted. "
                  "Latest checkpoint preserved — rerun to resume.")
        except Exception as e:
            print(f"\n[Construction FATAL] {e}")
            traceback.print_exc()
            sys.exit(1)
    else:
        await run_evaluation(args, cfg)


def _raise_fd_limit(target: int = 8192):
    """Raise the process file-descriptor soft limit to avoid EMFILE errors
    when many ainsert coroutines open JSON storage files concurrently."""
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        new_soft = min(target, hard) if hard != resource.RLIM_INFINITY else target
        if new_soft > soft:
            resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))
            print(f"[Init] Raised fd limit: {soft} → {new_soft}")
    except Exception as e:
        print(f"[Init WARN] Could not raise fd limit: {e}")


def main():
    _raise_fd_limit(8192)
    p = argparse.ArgumentParser(
        description="HyperGraphRAG evaluation on HotpotQA / MuSiQue",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ---- Mode & dataset ----
    p.add_argument("--mode",    required=True, choices=["construction", "evaluation"])
    p.add_argument("--dataset", required=True, choices=list(SUPPORTED_DATASETS))

    # ---- Provider ----
    p.add_argument(
        "--provider", default="local_vllm", choices=list(PROVIDER_DEFAULTS),
        help=("LLM provider. local_vllm (default) or deepseek. "
              "Override defaults with --model/--api-base/--api-key."),
    )

    # ---- Files ----
    p.add_argument("--corpus-file",    type=str,
                   help="Path to corpus.jsonl (construction mode).")
    p.add_argument("--questions-file", type=str,
                   help="Path to questions.jsonl (evaluation mode).")
    p.add_argument("--run-dir",        type=str, required=True,
                   help="Run directory. Holds checkpoint-N/ folders and "
                        "hgrag_index/ (auto-created).")

    # ---- Model overrides ----
    p.add_argument("--model",       type=str, default="",
                   help="Override provider default model name.")
    p.add_argument("--embed-model", type=str,
                   default="voyageai/voyage-4-nano",
                   help="Embedding model name. For ollama use the Ollama model tag "
                        "(e.g. 'embeddinggemma:300m'). For vllm use the HF model path.")
    p.add_argument("--embed-backend", type=str, default="vllm",
                   choices=["vllm", "ollama"],
                   help=("Embedding server backend. "
                         "vllm (default): hits --embed-api-base with openai/ prefix. "
                         "ollama: hits http://127.0.0.1:11434/v1 with api_key='ollama', "
                         "model name passed as-is (no prefix)."))
    p.add_argument("--embed-api-base", type=str, default="http://localhost:8001/v1",
                   help="Embedding API base URL (vllm only; ignored for ollama).")
    p.add_argument("--api-base",    type=str, default="",
                   help="Override provider default LLM API base URL.")
    p.add_argument("--api-key",     type=str, default="",
                   help="Override provider default API key.")
    p.add_argument("--embed-dim",   type=int, default=1024,
                   help="Embedding dimension (must match --embed-model).")

    # ---- Thinking mode ----
    p.add_argument(
        "--thinking", action="store_true", default=False,
        help=("Enable DeepSeek thinking mode. Passes "
              "extra_body={'thinking': {'type': 'enabled'}} to the API. "
              "Reasoning is returned in reasoning_content (separate from content), "
              "so the final answer in .content is always clean. "
              "Only meaningful with --provider deepseek. Default: OFF."),
    )

    # ---- Rate limiting ----
    p.add_argument(
        "--construction-rpm", type=int, default=0, metavar="N",
        help=("Max paragraphs/minute during construction. 0=unlimited (default). "
              "Each ainsert makes ~5-10 internal LLM calls. "
              "Recommended for DeepSeek free tier: 10."),
    )
    p.add_argument(
        "--eval-rpm", type=int, default=0, metavar="N",
        help=("Max aquery calls/minute during evaluation. 0=unlimited. "
              "Each aquery makes ~3-5 internal LLM calls. "
              "Recommended for DeepSeek free tier: 50."),
    )

    # ---- Construction hyperparams ----
    p.add_argument("--insert-concurrency", type=int, default=32,
                   help="[DEPRECATED — no longer used] Previously capped concurrent "
                        "ainsert calls. Insertion is now done as a single batch call "
                        "per checkpoint block; internal concurrency is controlled by "
                        "--llm-max-async. Argument is accepted but ignored.")
    p.add_argument("--checkpoint-interval", type=int, default=50,
                   help="Save a checkpoint every N paragraphs. Default: 50. "
                        "Smaller = safer against crashes; larger = less I/O.")
    p.add_argument("--delete-old-checkpoints", action="store_true",
                   help="Delete previous checkpoint-N folders after saving "
                        "a new one. Keeps only the latest on disk. "
                        "final-checkpoint is never deleted.")
    p.add_argument("--chunk-size",          type=int, default=1200,
                   help="HyperGraphRAG internal chunk token size.")
    p.add_argument("--chunk-overlap",       type=int, default=100)
    p.add_argument("--max-gleanings",       type=int, default=2,
                   help="Entity extraction gleaning passes per chunk.")
    p.add_argument("--llm-max-async",       type=int, default=16,
                   help="Max concurrent internal LLM calls within HyperGraphRAG. "
                        "Lower this for external APIs to reduce 429s.")
    p.add_argument("--embedding-max-async", type=int, default=16)
    p.add_argument("--llm-max-token-size",  type=int, default=32768)
    p.add_argument("--llm-max-retries",     type=int, default=5,
                   help="Retry attempts for transient connection errors "
                        "(exponential backoff, capped at 30s).")
    p.add_argument("--request-timeout",     type=float, default=120.0,
                   help="Per-request timeout in seconds.")

    # ---- Evaluation hyperparams ----
    p.add_argument("--retrieval-top-k",     type=int, default=60,
                   help="top_k for HyperGraphRAG hybrid retrieval.")
    p.add_argument("--max-context-tokens",  type=int, default=12000,
                   help="Max tokens of retrieved context passed to the answer LLM. "
                        "Truncated to this budget after HGRAG retrieval. "
                        "Per-question and average context lengths are logged in results.")
    p.add_argument("--short-answer", action=argparse.BooleanOptionalAction, default=True,
                   help="Use ShortAnswerQuestion (span-style, no explanation) for answer "
                        "generation via DSPy (default: on). Matches the KGGen harness. "
                        "Designed for EM/F1 benchmarks where the gold is a short phrase. "
                        "Pass --no-short-answer to fall back to the sentence-style "
                        "AnswerQuestion prompt (controlled by --chain-of-thought).")
    p.add_argument("--chain-of-thought", action=argparse.BooleanOptionalAction, default=True,
                   help="Use dspy.ChainOfThought for answer generation (default: on). "
                        "Only applies when --no-short-answer is set. "
                        "Pass --no-chain-of-thought to use dspy.Predict instead.")
    p.add_argument("--max-workers",         type=int, default=10,
                   help="Asyncio semaphore size for concurrent question evaluation.")

    args = p.parse_args()

    # ---- Validate ----
    if args.mode == "construction" and not args.corpus_file:
        p.error("--mode construction requires --corpus-file")
    if args.mode == "evaluation" and not args.questions_file:
        p.error("--mode evaluation requires --questions-file")

    cfg = resolve_provider(args)

    rpm_val = args.construction_rpm if args.mode == "construction" else args.eval_rpm
    rpm_str = str(rpm_val) if rpm_val > 0 else "unlimited"

    print(f"\n{'='*65}")
    print(f"  HGRAG {args.mode.upper()} -- {args.dataset.upper()}")
    print(f"  provider : {cfg.provider}")
    print(f"  model    : {cfg.model}")
    print(f"  api-base : {cfg.api_base}")
    print(f"  embed    : {args.embed_model} ({args.embed_backend})")
    print(f"  run-dir  : {args.run_dir}")
    print(f"  RPM      : {rpm_str} ({args.mode})")
    if args.mode == "construction":
        print(f"  ckpt every : {args.checkpoint_interval} paragraphs")
        print(f"  delete old : {'yes' if args.delete_old_checkpoints else 'no'}")
    print(f"  thinking : {'ON' if args.thinking else 'OFF'}")
    print(f"{'='*65}\n")

    # Single event loop for the entire script (avoids LiteLLM logging worker bugs)
    asyncio.run(main_async(args, cfg))


if __name__ == "__main__":
    main()