#!/usr/bin/env python3
"""
graphrag_multihop_eval.py
=========================
GraphRAG 2.7.0 evaluation on HotpotQA and MuSiQue.

Two-mode script:
  construction   Materializes corpus.jsonl → corpus.txt → runs `graphrag index`.
                 Saves construction_results.json in --run-dir.
                 Re-runs are skipped if that file already exists
                 (pass --force-rebuild to override).

  evaluation     Loads the built parquet artifacts, answers questions in parallel,
                 scores with official EM / F1, saves evaluation_results.json.

PROVIDERS (--provider)
----------------------
  local_vllm (default)
      model    : Qwen/Qwen3-30B-A3B-Instruct-2507-FP8
      api-base : http://localhost:8000/v1
      api-key  : EMPTY
  deepseek
      model    : deepseek-v4-flash
      api-base : https://api.deepseek.com
      api-key  : DEEPSEEK_API_KEY env var  or  --api-key

  Override any field: --model / --api-base / --api-key

EMBEDDING BACKEND (--embed-backend)
------------------------------------
  vllm (default)
      Hits --embed-api-base (default http://localhost:8001/v1) with api_key
      inherited from the LLM provider.  Use for vLLM-served embed models.
  ollama
      Hits http://127.0.0.1:11434/v1 with api_key='ollama' (Ollama's
      OpenAI-compat endpoint).  --embed-api-base is ignored.
      Use --embed-model with an Ollama model tag (e.g. 'embeddinggemma:300m').

CHUNKING
--------
GraphRAG's default is 1200 tokens / 100 overlap.
Pass --chunk-size / --chunk-overlap to match the values used by the other
systems in your cross-system comparison.

CORPUS FORMAT
-------------
Each unique paragraph in corpus.jsonl is written as its own .txt file under
<run-dir>/corpus_files/, then copied into <run-dir>/workspace/input/.
Each file is formatted as:

    Title: <title>

    <paragraph text>

One file per paragraph prevents GraphRAG's chunker from creating chunks that
straddle two unrelated Wikipedia articles, keeping entity extraction clean.

EXAMPLE USAGE
-------------
  # Build the index (vLLM embed, default)
  python graphrag_multihop_eval.py \\
      --mode construction --dataset hotpotqa \\
      --corpus-file ./data/hotpot_corpus_1k.jsonl \\
      --run-dir ./results/hotpotqa/graphrag/run1

  # Run evaluation
  python graphrag_multihop_eval.py \\
      --mode evaluation --dataset hotpotqa \\
      --questions-file ./data/hotpot_questions_1k.jsonl \\
      --run-dir ./results/hotpotqa/graphrag/run1 \\
      --max-workers 20

  # DeepSeek LLM + Ollama embeddings (embeddinggemma:300m)
  python graphrag_multihop_eval.py \\
      --mode construction --provider deepseek --dataset hotpotqa \\
      --corpus-file ./data/hotpot_corpus_1k.jsonl \\
      --run-dir ./results/hotpotqa/graphrag/run_ds_ollama \\
      --embed-backend ollama --embed-model embeddinggemma:300m \\
      --concurrent-requests 60

  # DeepSeek provider with custom chunk size
  python graphrag_multihop_eval.py \\
      --mode construction --provider deepseek --dataset musique \\
      --corpus-file ./data/musique_corpus_1k.jsonl \\
      --run-dir ./results/musique/graphrag/run_ds \\
      --chunk-size 600 --chunk-overlap 50

GRAPHRAG VERSION
----------------
Targets graphrag==2.7.0 (new models: block in settings.yaml,
ContextBuilderResult return type from build_context, LanceDBVectorStore
takes VectorStoreSchemaConfig). Adjust the imports below for newer versions.
"""

import os
import sys
import json
import time
import uuid
import shutil
import textwrap
import argparse
import subprocess
import threading
import traceback
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict, deque

from dotenv import load_dotenv
load_dotenv()
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import litellm
from transformers import AutoTokenizer

# ── shared utilities (same module as KGGen / HyperGraphRAG scripts) ──────────
from experiments.utils.multihop_utils import (
    HOTPOTQA, MUSIQUE, SUPPORTED_DATASETS,
    load_corpus_jsonl, load_questions_jsonl,
    GlobalTokenAccumulator, attach_token_tracker, detach_token_tracker,
    call_llm_for_answer, score_em, score_f1, truncate_tokens, safe_div,
    add_token_counters,
)

# ── GraphRAG 2.7.0 ───────────────────────────────────────────────────────────
from graphrag.query.indexer_adapters import (
    read_indexer_entities,
    read_indexer_relationships,
    read_indexer_reports,
    read_indexer_text_units,
)
from graphrag.query.context_builder.entity_extraction import EntityVectorStoreKey
from graphrag.query.structured_search.local_search.mixed_context import LocalSearchMixedContext
from graphrag.config.models.vector_store_schema_config import VectorStoreSchemaConfig
from graphrag.vector_stores.lancedb import LanceDBVectorStore
from graphrag.tokenizer.get_tokenizer import get_tokenizer as graphrag_get_tokenizer


# =============================================================================
# PROVIDER CONFIGURATION
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
        provider="deepseek",
        model="deepseek-v4-flash",   # current prod model (May 2026); legacy alias deepseek-chat retires 2026-07-24
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


def _eval_model_string(cfg: ProviderConfig) -> str:
    """
    LiteLLM routing string for synchronous completion calls during evaluation.
    local_vllm → 'openai/<model>'  (OpenAI-compatible endpoint)
    deepseek   → 'deepseek/<model>'
    """
    if cfg.provider == "local_vllm":
        return f"openai/{cfg.model}"
    if cfg.provider == "deepseek":
        return f"deepseek/{cfg.model}"
    return cfg.model


# =============================================================================
# LITELLM EMBEDDING WRAPPER (GraphRAG EmbeddingModel Protocol)
# Copied verbatim from graphrag_eval.py — satisfies the Protocol used
# by LocalSearchMixedContext.  embed() is the hot path called per query.
# =============================================================================
class LiteLLMEmbedder:
    """Wraps LiteLLM embedding behind GraphRAG's EmbeddingModel Protocol."""

    config = None   # Protocol attribute; never read during local search

    def __init__(self, model: str, api_base: str, api_key: str,
                 timeout: float = 120.0, max_retries: int = 5):
        self._model       = f"openai/{model}"
        self._api_base    = api_base
        self._api_key     = api_key
        self._timeout     = timeout
        self._max_retries = max_retries

    def embed(self, text: str, **_kwargs) -> list:
        resp = litellm.embedding(
            model=self._model, input=[text],
            api_base=self._api_base, api_key=self._api_key,
            timeout=self._timeout, num_retries=self._max_retries,
        )
        return resp.data[0]["embedding"]

    def embed_batch(self, text_list: list, **_kwargs) -> list:
        resp = litellm.embedding(
            model=self._model, input=text_list,
            api_base=self._api_base, api_key=self._api_key,
            timeout=self._timeout, num_retries=self._max_retries,
        )
        return [d["embedding"] for d in resp.data]

    async def aembed(self, text: str, **_kwargs) -> list:
        resp = await litellm.aembedding(
            model=self._model, input=[text],
            api_base=self._api_base, api_key=self._api_key,
            timeout=self._timeout, num_retries=self._max_retries,
        )
        return resp.data[0]["embedding"]

    async def aembed_batch(self, text_list: list, **_kwargs) -> list:
        resp = await litellm.aembedding(
            model=self._model, input=text_list,
            api_base=self._api_base, api_key=self._api_key,
            timeout=self._timeout, num_retries=self._max_retries,
        )
        return [d["embedding"] for d in resp.data]


OLLAMA_EMBED_BASE_URL = "http://127.0.0.1:11434/v1"
OLLAMA_EMBED_API_KEY  = "ollama"


def _resolve_embed_endpoint(args, cfg: "ProviderConfig") -> tuple:
    """
    Returns (embed_api_base, embed_api_key) for the chosen --embed-backend.
    ollama: fixed Ollama base URL + api_key='ollama'; --embed-api-base ignored.
    vllm  : uses --embed-api-base and the LLM provider's api_key.
    """
    if args.embed_backend == "ollama":
        return OLLAMA_EMBED_BASE_URL, OLLAMA_EMBED_API_KEY
    return args.embed_api_base, cfg.api_key


# =============================================================================
# CORPUS MATERIALIZATION
# Converts JSONL corpus (title + paragraph fields) → one .txt file per
# paragraph so GraphRAG never chunks across unrelated Wikipedia articles.
# =============================================================================

def write_paragraph_files(corpus: dict, output_dir: str) -> int:
    """
    Write each paragraph in `corpus` (id → entry dict) to its own .txt file
    inside `output_dir`.  Each file is formatted as:

        Title: <title>

        <paragraph text>

    Returns total character count written across all files.
    """
    os.makedirs(output_dir, exist_ok=True)
    total_chars = 0
    for pid, entry in corpus.items():
        safe_name = pid.replace(os.sep, "_") + ".txt"
        content = f"Title: {entry['title']}\n\n{entry['paragraph']}"
        fpath = os.path.join(output_dir, safe_name)
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(content)
        total_chars += len(content)
    return total_chars


# =============================================================================
# GRAPHRAG WORKSPACE SETUP
# =============================================================================

def setup_graphrag_workspace(
    workspace_dir: str,
    corpus_files_dir: str,
    model: str,
    embed_model: str,
    api_base: str,
    embed_api_base: str,
    llm_api_key: str,
    embed_api_key: str,
    chunk_size: int,
    chunk_overlap: int,
    max_gleanings: int,
    concurrent_requests: int,
    community_max_length: int,
    batch_max_tokens: int = 1900,
    llm_rpm: int = 1000,
    llm_tpm: int = 100000,
    embed_rpm: int = 10000,
    embed_tpm: int = 10000000,
):
    """
    1. `graphrag init`  – creates workspace skeleton + default prompts.
    2. Copies corpus.txt into input/.
    3. Overwrites settings.yaml with our vLLM / DeepSeek / Ollama endpoints.
    llm_api_key  : API key for the chat model (provider key or 'EMPTY').
    embed_api_key: API key for the embedding model ('ollama' for Ollama backend).
    """
    os.makedirs(workspace_dir, exist_ok=True)

    settings_yaml_path = os.path.join(workspace_dir, "settings.yaml")
    if os.path.exists(settings_yaml_path):
        print(f"[Construction] Workspace already initialized — skipping graphrag init, reusing cache.")
    else:
        print(f"[Construction] graphrag init → {workspace_dir}")
        init_result = subprocess.run(
            ["python", "-m", "graphrag", "init", "--root", workspace_dir],
            capture_output=True, text=True,
        )
        if init_result.returncode != 0:
            raise RuntimeError(
                f"graphrag init failed (exit {init_result.returncode}):\n"
                f"STDOUT: {init_result.stdout}\nSTDERR: {init_result.stderr}"
            )

    # Copy per-paragraph .txt files into input/
    input_dir = os.path.join(workspace_dir, "input")
    os.makedirs(input_dir, exist_ok=True)
    for fname in os.listdir(corpus_files_dir):
        if fname.endswith(".txt"):
            shutil.copy(os.path.join(corpus_files_dir, fname),
                        os.path.join(input_dir, fname))

    # --- settings.yaml -------------------------------------------------------
    # entity_types are broadened for Wikipedia-style knowledge vs. insurance
    # domain.  chunk_size defaults to 1200 (GraphRAG default) for fair
    # comparison.  community_max_length at 1500 cl100k tokens is a safe ceiling
    # for embed models with SentencePiece tokenizers (2048-token limit).
    settings_content = textwrap.dedent(f"""\
        encoding_model: cl100k_base
        skip_workflows: []
        models:
          default_chat_model:
            api_key: {llm_api_key}
            type: chat
            model_provider: openai
            model: {model}
            encoding_model: cl100k_base
            api_base: {api_base}
            tokens_per_minute: {llm_tpm}
            requests_per_minute: {llm_rpm}
            max_retries: 5
            concurrent_requests: {concurrent_requests}
          default_embedding_model:
            api_key: {embed_api_key}
            type: embedding
            model_provider: openai
            model: {embed_model}
            encoding_model: cl100k_base
            api_base: {embed_api_base}
            tokens_per_minute: {embed_tpm}
            requests_per_minute: {embed_rpm}
            concurrent_requests: {concurrent_requests}
        chunks:
          size: {chunk_size}
          overlap: {chunk_overlap}
          group_by_columns: [id]
        entity_extraction:
          prompt: "prompts/entity_extraction.txt"
          entity_types: [organization, person, geo, event]
          max_gleanings: {max_gleanings}
        summarize_descriptions:
          prompt: "prompts/summarize_descriptions.txt"
          max_length: 500
          max_input_tokens: 4000
        claim_extraction:
          enabled: false
        community_reports:
          prompt: "prompts/community_report.txt"
          max_length: {community_max_length}
          max_input_length: 8000
        cluster_graph:
          max_cluster_size: 10
        embed_graph:
          enabled: false
        embed_text:
          enabled: true
          batch_max_tokens: {batch_max_tokens}
    """)
    with open(os.path.join(workspace_dir, "settings.yaml"), "w", encoding="utf-8") as f:
        f.write(settings_content)


# =============================================================================
# BEST-EFFORT TOKEN ACCOUNTING (GraphRAG cache directory)
# GraphRAG uses its own LLM client (fnllm / litellm provider) so we cannot
# inject a LiteLLM callback during construction.  We walk cache/ after the
# subprocess completes and pull token counts from the cached JSON files.
# Path: data["result"]["response"]["usage"] (GraphRAG 2.7.0 litellm provider)
# =============================================================================

def _parse_graphrag_cache(workspace_dir: str) -> tuple:
    """Returns (api_calls, prompt_tokens, completion_tokens)."""
    cache_dir = os.path.join(workspace_dir, "cache")
    api_calls = prompt_tokens = completion_tokens = 0
    unrecognised = 0
    _sample_printed = False

    if not os.path.exists(cache_dir):
        return api_calls, prompt_tokens, completion_tokens

    for root, _dirs, files in os.walk(cache_dir):
        for fname in files:
            fpath = os.path.join(root, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                continue
            api_calls += 1

            if not _sample_printed:
                _sample_printed = True
                print(f"[Token Tracking] Cache sample ({fname}):")
                print(f"  top keys: {list(data.keys()) if isinstance(data, dict) else type(data)}")
                usage_val = (data.get("result", {}).get("response", {}).get("usage")
                             if isinstance(data, dict) else None)
                print(f"  result.response.usage: {usage_val}")

            usage = None
            for path in (
                ("result", "response", "usage"),   # GraphRAG 2.7.0 litellm — primary
                ("result", "usage"),               # fnllm fallback
                ("usage",),                        # legacy flat
                ("metadata", "usage"),
                ("output", "usage"),
            ):
                node, ok = data, True
                for key in path:
                    if not isinstance(node, dict) or key not in node:
                        ok = False; break
                    node = node[key]
                if ok and isinstance(node, dict):
                    usage = node; break

            if usage:
                prompt_tokens     += int(usage.get("prompt_tokens", 0) or 0)
                completion_tokens += int(usage.get("completion_tokens", 0) or 0)
            else:
                unrecognised += 1

    if unrecognised:
        print(f"[Token Tracking] {unrecognised}/{api_calls} cache files had no "
              "recognisable usage field (normal for embedding entries).")
    return api_calls, prompt_tokens, completion_tokens


# =============================================================================
# GRAPH CONSTRUCTION
# =============================================================================

CONSTRUCTION_RESULTS_FILE = "construction_results.json"
WORKSPACE_SUBDIR          = "workspace"


def _workspace_dir(run_dir: str) -> str:
    return os.path.join(run_dir, WORKSPACE_SUBDIR)


def _output_dir(run_dir: str) -> str:
    return os.path.join(_workspace_dir(run_dir), "output")


def run_construction(args, cfg: ProviderConfig):
    os.makedirs(args.run_dir, exist_ok=True)

    # ── Guard: skip if already built ─────────────────────────────────────────
    results_path = os.path.join(args.run_dir, CONSTRUCTION_RESULTS_FILE)
    if os.path.exists(results_path) and not args.force_rebuild:
        print(f"[Construction] {CONSTRUCTION_RESULTS_FILE} already exists — skipping.")
        print(f"[Construction] Pass --force-rebuild to rebuild from scratch.")
        return

    if args.force_rebuild:
        workspace_to_remove = _workspace_dir(args.run_dir)
        if os.path.exists(workspace_to_remove):
            print("[Construction] --force-rebuild: removing previous workspace.")
            shutil.rmtree(workspace_to_remove, ignore_errors=True)
        if os.path.exists(results_path):
            os.remove(results_path)

    workspace = _workspace_dir(args.run_dir)

    # ── Load corpus ───────────────────────────────────────────────────────────
    print(f"[Construction] Loading corpus: {args.corpus_file}")
    corpus = load_corpus_jsonl(args.corpus_file)
    print(f"[Construction]   {len(corpus):,} unique paragraphs")

    # ── Write one .txt file per paragraph ────────────────────────────────────
    corpus_files_dir = os.path.join(args.run_dir, "corpus_files")
    print(f"[Construction] Writing {len(corpus):,} paragraph files → {corpus_files_dir}")
    total_chars = write_paragraph_files(corpus, corpus_files_dir)
    total_words = total_chars // 5  # rough estimate
    print(f"[Construction]   {total_chars:,} chars  (~{total_words:,} words)")

    # ── Token count the corpus ────────────────────────────────────────────────
    try:
        all_text = " ".join(
            open(os.path.join(corpus_files_dir, f), encoding="utf-8").read()
            for f in os.listdir(corpus_files_dir) if f.endswith(".txt")
        )
        total_doc_tokens = litellm.token_counter(
            model=f"openai/{cfg.model}", text=all_text
        )
    except Exception:
        total_doc_tokens = total_chars // 4  # cl100k_base ~4 chars/token
        print("[WARN] LiteLLM token counter unavailable; using chars/4 estimate.")
    print(f"[Construction]   ~{total_doc_tokens:,} tokens in corpus")

    # ── graphrag init + settings.yaml ─────────────────────────────────────────
    embed_api_base, embed_api_key = _resolve_embed_endpoint(args, cfg)
    setup_graphrag_workspace(
        workspace_dir=workspace,
        corpus_files_dir=corpus_files_dir,
        model=cfg.model,
        embed_model=args.embed_model,
        api_base=cfg.api_base,
        embed_api_base=embed_api_base,
        llm_api_key=cfg.api_key,
        embed_api_key=embed_api_key,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        max_gleanings=args.max_gleanings,
        concurrent_requests=args.concurrent_requests,
        community_max_length=args.community_max_length,
        batch_max_tokens=args.batch_max_tokens,
        llm_rpm=args.llm_rpm,
        llm_tpm=args.llm_tpm,
        embed_rpm=args.embed_rpm,
        embed_tpm=args.embed_tpm,
    )

    # ── graphrag index ─────────────────────────────────────────────────────────
    log_path = os.path.join(args.run_dir, "graphrag_index.log")
    print(f"[Construction] Running `graphrag index` (streaming output; log → {log_path})")
    t0 = time.time()
    tail_buf = deque(maxlen=40)   # last 40 lines for error reporting

    proc = subprocess.Popen(
        ["python", "-c", "import patch_graphrag; from graphrag.cli.main import app; app(prog_name='graphrag')", "index", "--root", workspace],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # merge stderr into stdout stream
        text=True,
        bufsize=1,
    )
    with open(log_path, "w", encoding="utf-8") as log_f:
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_f.write(line)
            log_f.flush()
            tail_buf.append(line.rstrip())
    proc.wait()
    execution_time = round(time.time() - t0, 2)

    # ── Handle non-zero exit ───────────────────────────────────────────────────
    # GraphRAG 2.7.0 can exit with code 1 due to LiteLLM LoggingWorker timeout
    # errors even when the pipeline completes and writes all required artifacts.
    # We tolerate exit code 1 if the critical parquet files are present; if any
    # required artifact is missing we still raise a hard error.
    out_dir = _output_dir(args.run_dir)
    if proc.returncode != 0:
        REQUIRED = ["entities.parquet", "relationships.parquet", "text_units.parquet"]
        missing_required = [
            f for f in REQUIRED if not os.path.exists(os.path.join(out_dir, f))
        ]
        if missing_required:
            raise RuntimeError(
                f"[Construction FATAL] graphrag index returned exit {proc.returncode}.\n"
                f"Missing required artifacts: {missing_required}\n"
                f"Last 40 lines (also in {log_path}):\n" + "\n".join(tail_buf)
            )
        print(
            f"[Construction WARN] graphrag index exited {proc.returncode} but all "
            f"required parquet artifacts are present. Continuing.\n"
            f"  (This is a known graphrag/LiteLLM LoggingWorker timeout issue.)\n"
            f"  Check {log_path} for details."
        )
        if not os.path.exists(os.path.join(out_dir, "community_reports.parquet")):
            print(
                "[Construction WARN] community_reports.parquet is missing — "
                "community context will be disabled in evaluation (community_prop=0)."
            )

    # ── Parse parquet metrics ─────────────────────────────────────────────────

    def _safe_count(name):
        p = os.path.join(out_dir, name)
        if os.path.exists(p):
            try:
                return len(pd.read_parquet(p))
            except Exception as e:
                print(f"[WARN] Could not read {name}: {e}")
        return 0

    n_entities    = _safe_count("entities.parquet")
    n_edges       = _safe_count("relationships.parquet")
    n_chunks      = _safe_count("text_units.parquet")
    n_communities = _safe_count("communities.parquet")
    n_reports     = _safe_count("community_reports.parquet")
    n_documents   = _safe_count("documents.parquet")

    artifact_files = []
    if os.path.isdir(out_dir):
        for fname in sorted(os.listdir(out_dir)):
            fpath = os.path.join(out_dir, fname)
            if os.path.isfile(fpath):
                artifact_files.append({"file": fname,
                                       "size_bytes": os.path.getsize(fpath)})

    # ── Token accounting from cache ───────────────────────────────────────────
    api_calls, prompt_tokens, completion_tokens = _parse_graphrag_cache(workspace)

    construction_results = {
        "run_id":    f"graphrag_construction_{uuid.uuid4().hex[:8]}",
        "timestamp": datetime.now().isoformat(),
        "system":    "GRAPHRAG",
        "dataset":   args.dataset,
        "metadata": {
            "corpus_file":        os.path.abspath(args.corpus_file),
            "corpus_paragraphs":  len(corpus),
            "corpus_files_dir":   os.path.abspath(corpus_files_dir),
            "run_dir":           os.path.abspath(args.run_dir),
            "workspace_dir":     os.path.abspath(workspace),
            "output_dir":        os.path.abspath(out_dir),
            "provider":          cfg.provider,
            "model":             cfg.model,
            "embed_model":       args.embed_model,
            "api_base":          cfg.api_base,
            "embed_api_base":    args.embed_api_base,
            "construction_params": {
                "chunk_size":          args.chunk_size,
                "chunk_overlap":       args.chunk_overlap,
                "max_gleanings":       args.max_gleanings,
                "concurrent_requests": args.concurrent_requests,
                "community_max_length": args.community_max_length,
            },
        },
        "construction": {
            "total_corpus_chars":      total_chars,
            "total_corpus_tokens":     total_doc_tokens,
            "total_chunks":            n_chunks,
            "entities":                n_entities,
            "edges":                   n_edges,
            "communities":             n_communities,
            "community_reports":       n_reports,
            "documents":               n_documents,
            "llm_prompt_tokens":       prompt_tokens,
            "llm_completion_tokens":   completion_tokens,
            "llm_total_tokens":        prompt_tokens + completion_tokens,
            "llm_api_calls":           api_calls,
            "execution_time_sec":      execution_time,
            "artifact_files":          artifact_files,
            "_token_tracking_note": (
                "Token counts walked from GraphRAG's cache/ directory. "
                "GraphRAG uses its own LLM client (not bare LiteLLM), so "
                "callbacks cannot be injected during indexing."
            ),
        },
    }

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(construction_results, f, indent=2, default=str)

    # ── Summary ───────────────────────────────────────────────────────────────
    c = construction_results["construction"]
    print(f"\n[Construction] DONE  {execution_time}s")
    print(f"[Construction]   Paragraphs  : {len(corpus):,}")
    print(f"[Construction]   Corpus chars: {total_chars:,}  (~{total_doc_tokens:,} tokens)")
    print(f"[Construction]   Chunks      : {n_chunks}  ({args.chunk_size} tok, "
          f"{args.chunk_overlap} overlap)")
    print(f"[Construction]   Entities    : {n_entities}  Edges: {n_edges}")
    print(f"[Construction]   Communities : {n_communities}  Reports: {n_reports}")
    print(f"[Construction]   LLM tokens  : {c['llm_total_tokens']:,}  "
          f"({api_calls} cache entries)")
    print(f"[Construction]   Results     → {results_path}")


# =============================================================================
# CONTEXT BUILDER SETUP  (same API as insurance eval script)
# =============================================================================

def setup_local_context_builder(
    run_dir: str,
    embed_model: str,
    embed_api_base: str,
    embed_api_key: str,
    community_level: int,
    embed_timeout: float = 120.0,
    embed_max_retries: int = 5,
) -> LocalSearchMixedContext:
    out_dir = _output_dir(run_dir)

    for required in ["entities.parquet", "relationships.parquet", "text_units.parquet"]:
        p = os.path.join(out_dir, required)
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"GraphRAG output missing: {p}\n"
                "Run --mode construction first."
            )

    entity_df       = pd.read_parquet(os.path.join(out_dir, "entities.parquet"))
    relationship_df = pd.read_parquet(os.path.join(out_dir, "relationships.parquet"))
    text_unit_df    = pd.read_parquet(os.path.join(out_dir, "text_units.parquet"))

    community_path = os.path.join(out_dir, "communities.parquet")
    community_df   = (pd.read_parquet(community_path)
                      if os.path.exists(community_path) else pd.DataFrame())

    report_path = os.path.join(out_dir, "community_reports.parquet")
    if os.path.exists(report_path) and len(community_df) > 0:
        report_df = pd.read_parquet(report_path)
        reports   = read_indexer_reports(report_df, community_df, community_level)
    else:
        reports = []
        print("[WARN] No community reports found — community_prop forced to 0.")

    entities      = read_indexer_entities(entity_df, community_df, community_level)
    relationships = read_indexer_relationships(relationship_df)
    text_units    = read_indexer_text_units(text_unit_df)

    # LanceDB vector store (API changed in graphrag 2.7.0)
    description_embedding_store = LanceDBVectorStore(
        vector_store_schema_config=VectorStoreSchemaConfig(
            index_name="default-entity-description"
        )
    )
    description_embedding_store.connect(db_uri=os.path.join(out_dir, "lancedb"))

    text_embedder = LiteLLMEmbedder(
        model=embed_model,
        api_base=embed_api_base,
        api_key=embed_api_key,
        timeout=embed_timeout,
        max_retries=embed_max_retries,
    )

    tokenizer = graphrag_get_tokenizer()   # cl100k_base TiktokenTokenizer

    context_builder = LocalSearchMixedContext(
        entities=entities,
        entity_text_embeddings=description_embedding_store,
        text_embedder=text_embedder,
        text_units=text_units,
        community_reports=reports,
        relationships=relationships,
        tokenizer=tokenizer,
        embedding_vectorstore_key=EntityVectorStoreKey.ID,
    )

    print(f"[Retrieval] Loaded {len(entities)} entities, "
          f"{len(relationships)} relationships, "
          f"{len(text_units)} text units, "
          f"{len(reports)} community reports.")
    return context_builder


def _retrieve_context(context_builder: LocalSearchMixedContext,
                      query: str, retrieval_params: dict) -> str:
    """
    Calls build_context() and returns a plain string.
    In graphrag 2.7.0 this returns a ContextBuilderResult with .context_chunks
    (str | list[str]), not a (str, dict) tuple as in 0.3.x.
    """
    result = context_builder.build_context(
        query=query, conversation_history=None, **retrieval_params
    )
    ctx = result.context_chunks
    if isinstance(ctx, list):
        return "\n\n".join(str(x) for x in ctx)
    return ctx if isinstance(ctx, str) else str(ctx)


# =============================================================================
# RATE LIMITER (thread-safe token-bucket, for eval phase)
# =============================================================================

class RateLimiter:
    """
    Thread-safe token-bucket rate limiter for synchronous (threaded) calls.
    rpm=0 disables entirely.  Refills at rpm/60 tokens per second.
    """
    def __init__(self, rpm: int = 0):
        self.rpm      = rpm
        self.disabled = rpm <= 0
        if not self.disabled:
            self._cap    = float(rpm)
            self._tokens = float(rpm)
            self._rate   = rpm / 60.0
            self._last   = time.monotonic()
            self._lock   = threading.Lock()

    def acquire(self):
        if self.disabled:
            return
        while True:
            with self._lock:
                now = time.monotonic()
                self._tokens = min(
                    self._cap,
                    self._tokens + (now - self._last) * self._rate
                )
                self._last = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
            time.sleep(1.0 / self._rate)

    def __enter__(self):  self.acquire(); return self
    def __exit__(self, *_): pass
    def __repr__(self):
        return ("RateLimiter(disabled)" if self.disabled
                else f"RateLimiter(rpm={self.rpm})")


# =============================================================================
# EVALUATION WORKER
# =============================================================================

def _eval_one(q: dict, context_builder: LocalSearchMixedContext,
              retrieval_params: dict, tokenizer, args, cfg: ProviderConfig,
              limiter: "RateLimiter") -> dict:
    qid     = q.get("question_id", "")
    question= q["question"]
    gold    = q["answer"]
    aliases = q.get("answer_aliases", [])
    qtype   = q.get("type", "unknown")
    qlevel  = q.get("level", None)

    llm_ans = raw_ctx = trunc_ctx = ""
    em = f1 = prec = rec = 0.0
    ctx_tok = 0
    status = "SUCCESS"; err = ""

    try:
        raw_ctx = _retrieve_context(context_builder, question, retrieval_params)

        trunc_ctx, ctx_tok = truncate_tokens(raw_ctx, tokenizer, args.max_context_tokens)

        # call_llm_for_answer always prepends "openai/" internally, so pass the
        # raw model name (cfg.model), NOT _eval_model_string() which would add
        # a provider prefix first and cause a double-prefix like
        # "openai/deepseek/deepseek-v4-flash".
        with limiter:
            llm_ans = call_llm_for_answer(
                model=cfg.model,
                api_base=cfg.api_base,
                api_key=cfg.api_key,
                question=question,
                context=trunc_ctx,
                max_tokens=args.answer_max_tokens,
                temperature=0.0,
            )

        em           = score_em(llm_ans, gold, aliases=aliases, dataset=args.dataset)
        f1, prec, rec = score_f1(llm_ans, gold, aliases=aliases, dataset=args.dataset)

    except Exception as e:
        status = "ERROR"; err = str(e)
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


def _agg(results: list, key: str) -> dict:
    """Aggregate EM/F1 by a grouping key (type or level)."""
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


# =============================================================================
# EVALUATION
# =============================================================================

def run_evaluation(args, cfg: ProviderConfig):
    # ── Guard: construction must be done ─────────────────────────────────────
    results_path = os.path.join(args.run_dir, CONSTRUCTION_RESULTS_FILE)
    if not os.path.exists(results_path):
        print(f"[Evaluation FATAL] {CONSTRUCTION_RESULTS_FILE} not found in "
              f"{args.run_dir}. Run --mode construction first.")
        sys.exit(1)

    with open(results_path, "r", encoding="utf-8") as f:
        construction_meta = json.load(f)
    print(f"[Evaluation] Found construction results from "
          f"{construction_meta.get('timestamp', '?')}")

    # ── Load questions ────────────────────────────────────────────────────────
    print(f"[Evaluation] Loading questions: {args.questions_file}")
    questions = load_questions_jsonl(args.questions_file)
    print(f"[Evaluation]   {len(questions):,} questions")

    # ── Context builder ───────────────────────────────────────────────────────
    print("[Evaluation] Loading GraphRAG parquet artifacts...")
    embed_api_base, embed_api_key = _resolve_embed_endpoint(args, cfg)
    context_builder = setup_local_context_builder(
        run_dir=args.run_dir,
        embed_model=args.embed_model,
        embed_api_base=embed_api_base,
        embed_api_key=embed_api_key,
        community_level=args.community_level,
    )

    # Library defaults for reference (build_context function-signature defaults).
    # These differ from the official notebook example in max_tokens and community_prop.
    _LIBRARY_DEFAULTS = {
        "max_tokens": 8000, "community_prop": 0.25, "text_unit_prop": 0.5,
        "top_k_mapped_entities": 10, "top_k_relationships": 10,
        "include_entity_rank": False, "include_relationship_weight": False,
    }
    # Script defaults (used when --default-retrieval-params is NOT set).
    _SCRIPT_DEFAULTS = {
        "max_tokens": 12000, "community_prop": 0.15, "text_unit_prop": 0.5,
        "top_k_mapped_entities": 10, "top_k_relationships": 10,
    }

    # Collect only the retrieval flags the user explicitly typed on the CLI.
    # argparse defaults are None, so a non-None value means it was explicitly passed.
    _explicit = {}
    if args.retrieval_max_tokens       is not None: _explicit["max_tokens"]            = args.retrieval_max_tokens
    if args.retrieval_top_k_entities   is not None: _explicit["top_k_mapped_entities"] = args.retrieval_top_k_entities
    if args.retrieval_top_k_relationships is not None: _explicit["top_k_relationships"]= args.retrieval_top_k_relationships
    if args.text_unit_prop             is not None: _explicit["text_unit_prop"]        = args.text_unit_prop
    if args.community_prop             is not None:
        _explicit["community_prop"] = max(args.community_prop, 0.0)

    if args.default_retrieval_params:
        # Start from nothing (library defaults apply for anything not in retrieval_params),
        # then layer in only the flags the user explicitly passed.
        retrieval_params = _explicit
        _defaulted = {k: v for k, v in _LIBRARY_DEFAULTS.items() if k not in _explicit}
        print("[Evaluation] --default-retrieval-params enabled.")
        print(f"[Evaluation]   Library defaults in effect : {_defaulted}")
        if _explicit:
            print(f"[Evaluation]   CLI overrides applied      : {_explicit}")
    else:
        # Use script defaults for anything not explicitly passed.
        retrieval_params = {
            "max_tokens":            _explicit.get("max_tokens",            _SCRIPT_DEFAULTS["max_tokens"]),
            "top_k_mapped_entities": _explicit.get("top_k_mapped_entities", _SCRIPT_DEFAULTS["top_k_mapped_entities"]),
            "top_k_relationships":   _explicit.get("top_k_relationships",   _SCRIPT_DEFAULTS["top_k_relationships"]),
            "text_unit_prop":        _explicit.get("text_unit_prop",        _SCRIPT_DEFAULTS["text_unit_prop"]),
            "community_prop":        max(_explicit.get("community_prop",    _SCRIPT_DEFAULTS["community_prop"]), 0.0),
        }

    # ── Tokenizer (for context truncation before LLM call) ───────────────────
    try:
        tokenizer = AutoTokenizer.from_pretrained(cfg.model)
        print(f"[Evaluation] Tokenizer: {cfg.model}")
    except Exception:
        try:
            import tiktoken

            class _TiktokenWrapper:
                def __init__(self, enc): self._enc = enc
                def encode(self, t, **kw): return self._enc.encode(t)
                def decode(self, ids, **kw): return self._enc.decode(ids)

            tokenizer = _TiktokenWrapper(tiktoken.get_encoding("cl100k_base"))
            print("[Evaluation] HuggingFace tokenizer unavailable; "
                  "falling back to tiktoken cl100k_base.")
        except Exception:
            tokenizer = None
            print("[Evaluation WARN] No tokenizer available — context truncation disabled.")

    # ── Token tracking for evaluation LLM calls ───────────────────────────────
    tracker = GlobalTokenAccumulator()
    attach_token_tracker(tracker)

    # ── Eval rate limiter ─────────────────────────────────────────────────────
    eval_limiter = RateLimiter(rpm=args.eval_rpm)
    print(f"[Evaluation] Rate limiter: {eval_limiter}")

    # ── Parallel evaluation ───────────────────────────────────────────────────
    print(f"[Evaluation] Starting evaluation: {len(questions):,} questions, "
          f"{args.max_workers} workers")
    t0 = time.time()
    results = []

    try:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            fmap = {
                executor.submit(_eval_one, q, context_builder, retrieval_params,
                                tokenizer, args, cfg, eval_limiter): i
                for i, q in enumerate(questions)
            }
            done = 0
            for future in as_completed(fmap):
                r = future.result()
                results.append(r)
                done += 1
                # Progress every 5% or final
                if done % max(1, len(questions) // 20) == 0 or done == len(questions):
                    ok = [x for x in results if x["status"] == "SUCCESS"]
                    n  = len(ok)
                    print(f"[Evaluation] {done}/{len(questions)}  "
                          f"EM={safe_div(sum(x['em'] for x in ok), n):.4f}  "
                          f"F1={safe_div(sum(x['f1'] for x in ok), n):.4f}  "
                          f"err={done - n}")
    finally:
        detach_token_tracker()

    eval_t  = round(time.time() - t0, 2)
    eval_tk = tracker.snapshot()

    # ── Aggregate metrics ─────────────────────────────────────────────────────
    ok  = [r for r in results if r["status"] == "SUCCESS"]
    bad = [r for r in results if r["status"] != "SUCCESS"]
    n   = len(ok)

    overall = {
        "em":        safe_div(sum(r["em"]        for r in ok), n),
        "f1":        safe_div(sum(r["f1"]        for r in ok), n),
        "precision": safe_div(sum(r["precision"] for r in ok), n),
        "recall":    safe_div(sum(r["recall"]    for r in ok), n),
    }
    by_type  = _agg(ok, "type")
    by_level = _agg(ok, "level") if args.dataset == HOTPOTQA else None

    # ── Totals: construction + evaluation ─────────────────────────────────────
    const_tokens = {
        "prompt_tokens":     construction_meta["construction"]["llm_prompt_tokens"],
        "completion_tokens": construction_meta["construction"]["llm_completion_tokens"],
        "total_tokens":      construction_meta["construction"]["llm_total_tokens"],
        "api_calls":         construction_meta["construction"]["llm_api_calls"],
    }
    all_tokens = add_token_counters(const_tokens, eval_tk)

    # ── Build output ──────────────────────────────────────────────────────────
    output = {
        "run_id":    f"graphrag_eval_{uuid.uuid4().hex[:8]}",
        "timestamp": datetime.now().isoformat(),
        "system":    "GRAPHRAG",
        "dataset":   args.dataset,
        "metadata": {
            "questions_file":   os.path.abspath(args.questions_file),
            "run_dir":          os.path.abspath(args.run_dir),
            "provider":         cfg.provider,
            "model":            cfg.model,
            "embed_model":      args.embed_model,
            "api_base":         cfg.api_base,
            "embed_api_base":   args.embed_api_base,
            "retrieval_params": retrieval_params,
            "max_context_tokens":  args.max_context_tokens,
            "answer_max_tokens":   args.answer_max_tokens,
            "community_level":     args.community_level,
            "max_workers":         args.max_workers,
        },
        "construction": construction_meta["construction"],
        "evaluation": {
            "total_questions":    len(questions),
            "successful":         n,
            "errored":            len(bad),
            "evaluation_time_sec": eval_t,
            "evaluation_tokens":  eval_tk,
            "overall":            overall,
            "by_type":            by_type,
            **({"by_level": by_level} if by_level else {}),
            "results":            results,
        },
        "totals": {
            "all_tokens":    all_tokens,
            "all_time_sec":  round(
                float(construction_meta["construction"]["execution_time_sec"]) + eval_t, 2
            ),
        },
    }

    eval_path = os.path.join(args.run_dir, "evaluation_results.json")
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)

    # ── Terminal summary ──────────────────────────────────────────────────────
    print(f"\n[Evaluation] DONE  {eval_t}s  →  {eval_path}")
    print(f"[Evaluation]   n={n}  err={len(bad)}  "
          f"EM={overall['em']}  F1={overall['f1']}")
    for k, v in sorted(by_type.items()):
        print(f"    type {k:>12}: EM={v['em']:.4f}  F1={v['f1']:.4f}  (n={v['count']})")
    if by_level:
        for k, v in sorted(by_level.items()):
            print(f"   level {k:>12}: EM={v['em']:.4f}  F1={v['f1']:.4f}  (n={v['count']})")
    print(f"[Evaluation]   eval tokens={eval_tk['total_tokens']:,}  "
          f"calls={eval_tk['api_calls']:,}")
    print(f"[Evaluation]   total (construction+eval) tokens="
          f"{all_tokens['total_tokens']:,}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    p = argparse.ArgumentParser(
        description="GraphRAG 2.7.0 evaluation on HotpotQA / MuSiQue",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ---- Mode & dataset ----
    p.add_argument("--mode",    required=True, choices=["construction", "evaluation"],
                   help="construction: build index. evaluation: run QA scoring.")
    p.add_argument("--dataset", required=True, choices=list(SUPPORTED_DATASETS),
                   help="hotpotqa or musique (controls EM/F1 scorer and output keys).")

    # ---- Core paths ----
    p.add_argument("--run-dir",        type=str, required=True,
                   help="Working directory. Construction writes workspace/ here; "
                        "evaluation reads from the same directory.")
    p.add_argument("--corpus-file",    type=str, default=None,
                   help="(construction) Path to corpus.jsonl from the prep script.")
    p.add_argument("--questions-file", type=str, default=None,
                   help="(evaluation) Path to questions.jsonl from the prep script.")

    # ---- Provider ----
    p.add_argument("--provider", default="local_vllm",
                   choices=list(PROVIDER_DEFAULTS),
                   help="LLM provider preset. Override with --model/--api-base/--api-key.")
    p.add_argument("--model",    type=str, default="",
                   help="Override provider default LLM model name.")
    p.add_argument("--api-base", type=str, default="",
                   help="Override provider default LLM API base URL.")
    p.add_argument("--api-key",  type=str, default="",
                   help="Override provider default API key.")

    # ---- Embedding ----
    p.add_argument("--embed-backend", type=str, default="vllm",
                   choices=["vllm", "ollama"],
                   help=("Embedding server backend. "
                         "vllm (default): hits --embed-api-base with the provider api_key. "
                         "ollama: hits http://127.0.0.1:11434/v1 with api_key='ollama'; "
                         "--embed-api-base is ignored."))
    p.add_argument("--embed-model",    type=str, default="voyageai/voyage-4-nano",
                   help=("Embedding model. For vllm use the HF model path; "
                         "for ollama use the Ollama model tag (e.g. 'embeddinggemma:300m')."))
    p.add_argument("--embed-api-base", type=str, default="http://localhost:8001/v1",
                   help="Embedding API base URL (vllm only; ignored for ollama).")

    # ---- Construction ----
    p.add_argument("--chunk-size",          type=int, default=1200,
                   help="GraphRAG chunk size in tokens (default: 1200, GraphRAG default). "
                        "Lower for cross-system comparison with KGGen/HyperGraphRAG.")
    p.add_argument("--chunk-overlap",       type=int, default=100,
                   help="GraphRAG chunk overlap in tokens (default: 100).")
    p.add_argument("--max-gleanings",       type=int, default=1,
                   help="Entity extraction max gleanings (default: 1).")
    p.add_argument("--concurrent-requests", type=int, default=25,
                   help="GraphRAG indexer concurrent_requests for LLM + embedding calls.")
    p.add_argument("--community-max-length", type=int, default=2000,
                   help="community_reports.max_length in tokens (default: 2000, GraphRAG default).")
    p.add_argument("--batch-max-tokens",    type=int, default=1900,
                   help="embed_text.batch_max_tokens in settings.yaml (default: 1900). "
                        "GraphRAG package default is 8191.")
    p.add_argument("--llm-rpm",  type=int, default=1000,
                   help="requests_per_minute for the LLM in settings.yaml "
                        "(default: 1000; matches DeepSeek free-tier). "
                        "For local vLLM set high (e.g. 10000).")
    p.add_argument("--llm-tpm",  type=int, default=100000,
                   help="tokens_per_minute for the LLM in settings.yaml "
                        "(default: 100000).")
    p.add_argument("--embed-rpm", type=int, default=10000,
                   help="requests_per_minute for the embedding model in settings.yaml "
                        "(default: 10000; effectively unlimited for local vLLM).")
    p.add_argument("--embed-tpm", type=int, default=10000000,
                   help="tokens_per_minute for the embedding model in settings.yaml "
                        "(default: 10000000; effectively unlimited for local vLLM).")
    p.add_argument("--force-rebuild", action="store_true",
                   help="Delete and rebuild the workspace even if construction_results.json "
                        "already exists.")

    # ---- Evaluation ----
    p.add_argument("--max-workers",    type=int, default=20,
                   help="Thread-pool size for parallel question evaluation.")
    p.add_argument("--eval-rpm",       type=int, default=0,
                   help="Rate limit for evaluation LLM calls (requests/min). "
                        "0 = disabled (default; fine for local vLLM). "
                        "Set e.g. 500 when using DeepSeek to avoid hitting API limits.")
    p.add_argument("--retrieval-top-k-entities",      type=int, default=None,
                   help="Number of entities retrieved from the vector store (script default: 10; "
                        "library default: 10). When --default-retrieval-params is set, "
                        "explicitly passing this flag still overrides the library default.")
    p.add_argument("--retrieval-top-k-relationships", type=int, default=None,
                   help="Number of out-of-network relationships pulled into context (script default: 10; "
                        "library default: 10). When --default-retrieval-params is set, "
                        "explicitly passing this flag still overrides the library default.")
    p.add_argument("--retrieval-max-tokens",          type=int, default=None,
                   help="GraphRAG's internal context builder token budget (script default: 12000; "
                        "library default: 8000). When --default-retrieval-params is set, "
                        "explicitly passing this flag still overrides the library default.")
    p.add_argument("--community-level",               type=int, default=2,
                   help="Community hierarchy level for report retrieval.")
    p.add_argument("--text-unit-prop",                type=float, default=None,
                   help="Fraction of context budget for raw text chunks (script default: 0.5; "
                        "library default: 0.5). When --default-retrieval-params is set, "
                        "explicitly passing this flag still overrides the library default.")
    p.add_argument("--community-prop",                type=float, default=None,
                   help="Fraction of context budget for community summaries (script default: 0.15; "
                        "library default: 0.25; auto-zeroed when no reports exist). "
                        "When --default-retrieval-params is set, explicitly passing this flag "
                        "still overrides the library default.")
    p.add_argument("--default-retrieval-params", action="store_true",
                   help="Use the library's built-in build_context defaults for any retrieval param "
                        "not explicitly passed on the command line (max_tokens=8000, "
                        "community_prop=0.25, include_entity_rank=False, "
                        "include_relationship_weight=False). Any --retrieval-* / --text-unit-prop "
                        "/ --community-prop flags passed alongside this are still respected.")
    p.add_argument("--use-graphrag-defaults", action="store_true",
                   help="Override all KG-quality parameters to exact GraphRAG package defaults: "
                        "chunk_size=1200, chunk_overlap=100, community_max_length=2000, "
                        "batch_max_tokens=8191, retrieval_max_tokens=12000, community_prop=0.15, "
                        "max_context_tokens=12000. Also implies --default-retrieval-params. "
                        "Infrastructure params (concurrent_requests, RPM/TPM, etc.) are unchanged. "
                        "entity_types use the GraphRAG default [organization, person, geo, event].")
    p.add_argument("--max-context-tokens",            type=int, default=2000,
                   help="Token limit for context passed to the answer-generation LLM.")
    p.add_argument("--answer-max-tokens",             type=int, default=128,
                   help="max_tokens for the answer-generation LLM call. "
                        "HotpotQA/MuSiQue answers are short — 128 is plenty.")

    args = p.parse_args()

    # ---- Apply pure GraphRAG defaults if requested ----
    if args.use_graphrag_defaults:
        args.chunk_size           = 1200
        args.chunk_overlap        = 100
        args.max_gleanings        = 1
        args.community_max_length = 2000
        args.batch_max_tokens     = 8191
        args.max_context_tokens   = 12000
        # Implies --default-retrieval-params but with graphrag defaults for
        # the retrieval flags (so they override the library's own defaults too)
        args.default_retrieval_params    = True
        args.retrieval_max_tokens        = 12000
        args.community_prop              = 0.15
        args.text_unit_prop              = 0.5
        args.retrieval_top_k_entities    = 10
        args.retrieval_top_k_relationships = 10
        print("[*] --use-graphrag-defaults: all KG-quality params set to GraphRAG package defaults.")

    # ---- Validate ----
    if args.mode == "construction" and not args.corpus_file:
        p.error("--mode construction requires --corpus-file")
    if args.mode == "evaluation" and not args.questions_file:
        p.error("--mode evaluation requires --questions-file")

    cfg = resolve_provider(args)

    # ---- Banner ----
    print(f"\n{'=' * 65}")
    print(f"  GRAPHRAG {args.mode.upper()} — {args.dataset.upper()}")
    print(f"  provider  : {cfg.provider}")
    print(f"  model     : {cfg.model}")
    print(f"  api-base  : {cfg.api_base}")
    _embed_base = (OLLAMA_EMBED_BASE_URL if args.embed_backend == "ollama"
                   else args.embed_api_base)
    print(f"  embed     : {args.embed_model}  @  {_embed_base}  [{args.embed_backend}]")
    print(f"  run-dir   : {args.run_dir}")
    if args.mode == "construction":
        print(f"  chunk     : {args.chunk_size} tok / {args.chunk_overlap} overlap")
        print(f"  gleanings : {args.max_gleanings}")
        print(f"  concurrent: {args.concurrent_requests}")
        print(f"  llm RPM   : {args.llm_rpm}  TPM: {args.llm_tpm}")
        print(f"  embed RPM : {args.embed_rpm}  TPM: {args.embed_tpm}")
        if args.force_rebuild:
            print(f"  [!] --force-rebuild: existing workspace will be deleted")
    else:
        print(f"  eval RPM  : {args.eval_rpm if args.eval_rpm > 0 else 'unlimited'}")
    print(f"{'=' * 65}\n")

    # ---- Dispatch ----
    if args.mode == "construction":
        try:
            run_construction(args, cfg)
        except KeyboardInterrupt:
            print("\n[Construction] Interrupted. Workspace preserved in run-dir.")
        except Exception as e:
            print(f"\n[Construction FATAL] {e}")
            traceback.print_exc()
            sys.exit(1)
    else:
        try:
            run_evaluation(args, cfg)
        except KeyboardInterrupt:
            print("\n[Evaluation] Interrupted.")
        except Exception as e:
            print(f"\n[Evaluation FATAL] {e}")
            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    main()