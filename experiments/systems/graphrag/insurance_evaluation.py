#!/usr/bin/env python3
"""
Microsoft GraphRAG Insurance Contract Evaluation — Single Script
Targets graphrag==2.7.0.

Runs graph construction (via the graphrag CLI) followed by QA and/or Fact
evaluation, using the SAME judge prompts as the KGGen and HyperGraphRAG
scripts so the three systems can be compared fairly.

Single contract:
    python graphrag_eval.py \
        --contract path/to/contract.txt \
        --qa-file path/to/qa.json \
        --fact-file path/to/facts.json \
        --out-dir ./results

All contracts in both dataset sections:
    python graphrag_eval.py \
        --all \
        --eval-data-dir ./evaluation_data \
        --out-dir ./results

Only ten_contracts_dataset:
    python graphrag_eval.py \
        --ten-contracts-dataset \
        --eval-data-dir ./evaluation_data \
        --out-dir ./results

Only long_form_contracts:
    python graphrag_eval.py \
        --long-form-contracts \
        --eval-data-dir ./evaluation_data \
        --out-dir ./results

ARCHITECTURE
------------
1. Construction: subprocess `python -m graphrag index --root <workspace>`,
   parse parquet artifacts for graph metrics. Token usage extracted best-effort
   from GraphRAG's cache/ directory (GraphRAG uses fnllm not LiteLLM so we
   can't hook callbacks directly).

2. Retrieval: load parquet artifacts, instantiate LocalSearchMixedContext with
   a LiteLLMEmbedder wrapper. Call build_context() which returns a
   ContextBuilderResult with context_chunks — NO LLM call, just retrieval.

3. Evaluation: context_chunks → truncate_context → ask_llm_query →
   score_llm_answer / evaluate_fact. Same DSPy judge as KGGen and HyperGraphRAG.

API CHANGES vs OLD SCRIPT (graphrag 0.3.x → 2.7.0)
---------------------------------------------------
- graphrag.query.llm.oai.*  REMOVED entirely. Embedding uses the
  EmbeddingModel Protocol from graphrag.language_model.protocol.base.
  We implement this with LiteLLMEmbedder below.
- LanceDBVectorStore() now takes VectorStoreSchemaConfig(index_name=...)
  instead of collection_name=.
- LocalSearchMixedContext() uses tokenizer= (graphrag Tokenizer)
  instead of token_encoder= (raw tiktoken object).
- build_context() returns ContextBuilderResult with .context_chunks
  instead of a plain (str, dict) tuple.
- tiktoken is no longer imported directly; get_tokenizer() handles it.
"""

import asyncio
import os
import sys
import json
import time
import uuid
import shutil
import argparse
import subprocess
import threading
import textwrap
import traceback
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import litellm
from transformers import AutoTokenizer

# ---------- Project imports ----------
from experiments.evaluate_mine import ask_llm_query, score_llm_answer, evaluate_fact
from experiments.utils.chunking import truncate_context
from nkg.utils.config import configure_dspy

# ---------- GraphRAG 2.7.0 imports ----------
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
# LITELLM EMBEDDING WRAPPER — implements graphrag.language_model EmbeddingModel
# =============================================================================
# In graphrag 2.7.0, LocalSearchMixedContext.text_embedder must satisfy the
# EmbeddingModel Protocol (graphrag.language_model.protocol.base).
# The only method actually called during retrieval is the synchronous embed(),
# via map_query_to_entities → text_embedder=lambda t: text_embedder.embed(t).
# We implement all four methods of the protocol for completeness, but embed()
# is the one that matters for correctness.
class LiteLLMEmbedder:
    """
    Wraps LiteLLM embedding behind GraphRAG's EmbeddingModel Protocol.
    The `config` attribute is set to None because it is declared in the
    Protocol but never accessed internally during local search retrieval.
    """

    config = None   # Protocol requires the attribute; value is never read

    def __init__(self, model: str, api_base: str, api_key: str,
                 timeout: float = 120.0, max_retries: int = 5):
        self._model    = f"openai/{model}"
        self._api_base = api_base
        self._api_key  = api_key
        self._timeout  = timeout
        self._max_retries = max_retries

    # ---- synchronous methods (called by LocalSearchMixedContext) ----

    def embed(self, text: str, **_kwargs) -> list[float]:
        """Embed a single string. This is the hot path called per query."""
        response = litellm.embedding(
            model=self._model,
            input=[text],
            api_base=self._api_base,
            api_key=self._api_key,
            timeout=self._timeout,
            num_retries=self._max_retries,
        )
        return response.data[0]["embedding"]

    def embed_batch(self, text_list: list[str], **_kwargs) -> list[list[float]]:
        response = litellm.embedding(
            model=self._model,
            input=text_list,
            api_base=self._api_base,
            api_key=self._api_key,
            timeout=self._timeout,
            num_retries=self._max_retries,
        )
        return [d["embedding"] for d in response.data]

    # ---- async methods (part of the Protocol; not called during context build) ----

    async def aembed(self, text: str, **_kwargs) -> list[float]:
        response = await litellm.aembedding(
            model=self._model,
            input=[text],
            api_base=self._api_base,
            api_key=self._api_key,
            timeout=self._timeout,
            num_retries=self._max_retries,
        )
        return response.data[0]["embedding"]

    async def aembed_batch(self, text_list: list[str], **_kwargs) -> list[list[float]]:
        response = await litellm.aembedding(
            model=self._model,
            input=text_list,
            api_base=self._api_base,
            api_key=self._api_key,
            timeout=self._timeout,
            num_retries=self._max_retries,
        )
        return [d["embedding"] for d in response.data]


# =============================================================================
# DATASET PATH RESOLUTION  (identical to kggen_eval.py / hgrag_eval.py)
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
# ERROR TRACKER
# =============================================================================
class ErrorTracker:
    """
    Counter for errors per pipeline section.

    Sections:
        graph_construction
        context_builder_setup
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
    # Contracts that are skipped in kggen_eval.py for known compatibility reasons.
    # Keeping the same exclusion list ensures all three systems are evaluated on
    # an identical set of contracts for fair comparison.
    SKIP_CONTRACTS = {} #{"2025_pdp_contract.txt", "all_state_auto_insurance.txt"}
    triplets = []
    for subdir in subdirs:
        contracts_dir = os.path.join(eval_data_dir, subdir, "contracts")
        if not os.path.isdir(contracts_dir):
            print(f"[WARN] Contracts directory not found: {contracts_dir} — skipping.")
            continue
        for fname in sorted(os.listdir(contracts_dir)):
            if not fname.endswith(".txt"):
                continue
            if fname in SKIP_CONTRACTS:
                print(f"[WARN] Skipping {fname}: excluded by SKIP_CONTRACTS list.")
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
# WORKSPACE SETUP — graphrag init + custom settings.yaml
# =============================================================================
def setup_graphrag_workspace(workspace_dir: str, source_file: str,
                             model: str, embed_model: str,
                             api_base: str, embed_api_base: str, api_key: str,
                             chunk_size: int, chunk_overlap: int,
                             max_gleanings: int, concurrent_requests: int):
    """
    1. graphrag init  — creates the workspace skeleton + default prompts.
    2. Copy contract into input/.
    3. Overwrite settings.yaml pointing at our vLLM endpoints.
    """
    os.makedirs(workspace_dir, exist_ok=True)

    print(f"[Construction] Initializing GraphRAG workspace at {workspace_dir}")
    init_result = subprocess.run(
        ["python", "-m", "graphrag", "init", "--root", workspace_dir],
        capture_output=True, text=True,
    )
    if init_result.returncode != 0:
        raise RuntimeError(
            f"graphrag init failed (exit {init_result.returncode}):\n"
            f"STDOUT: {init_result.stdout}\nSTDERR: {init_result.stderr}"
        )

    input_dir = os.path.join(workspace_dir, "input")
    os.makedirs(input_dir, exist_ok=True)
    shutil.copy(source_file, os.path.join(input_dir, os.path.basename(source_file)))

    settings_content = textwrap.dedent(f"""\
        encoding_model: cl100k_base
        skip_workflows: []
        models:
          default_chat_model:
            api_key: {api_key}
            type: chat
            model_provider: openai
            model: {model}
            encoding_model: cl100k_base
            api_base: {api_base}
            tokens_per_minute: 100000
            requests_per_minute: 1000
            max_retries: 5
            concurrent_requests: {concurrent_requests}
            max_tokens: 10000
          default_embedding_model:
            api_key: {api_key}
            type: embedding
            model_provider: openai
            model: {embed_model}
            encoding_model: cl100k_base
            api_base: {embed_api_base}
            tokens_per_minute: 100000
            requests_per_minute: 1000
            concurrent_requests: {concurrent_requests}
        chunks:
          # Default: size=1200, overlap=100.
          # Set to match the chunk_size/overlap used by KGGen and HyperGraphRAG
          # for fair cross-system comparison.
          size: {chunk_size}
          overlap: {chunk_overlap}
          group_by_columns: [id]
        entity_extraction:
          prompt: "prompts/entity_extraction.txt"
          entity_types: [organization, person, geo, event]
          max_gleanings: {max_gleanings}
        summarize_descriptions:
          prompt: "prompts/summarize_descriptions.txt"
          max_length: 500        # default: 500 — unchanged
          max_input_tokens: 4000 # default: 4000 — unchanged
        claim_extraction:
          enabled: false
        community_reports:
          prompt: "prompts/community_report.txt"
          # Default: max_length=2000, max_input_length=8000.
          # max_length reduced from 2000 to 1500: the GraphRAG default assumes
          # OpenAI text-embedding-3-small (8191-token limit). Our model
          # (google/embeddinggemma-300m) uses SentencePiece and has a 2048-token
          # limit. SentencePiece tokenises legal text ~5-20% more aggressively
          # than cl100k_base, so 2000 cl100k tokens → ~2100-2400 SentencePiece
          # tokens → ContextWindowExceededError on long-form contracts.
          # 1500 cl100k tokens → ~1575-1800 SentencePiece tokens → safely under 2048.
          max_length: 1500
          max_input_length: 8000 # default: 8000 — unchanged
        cluster_graph:
          max_cluster_size: 10   # default: 10 — unchanged
        embed_graph:
          enabled: false         # default: false — node2vec embeddings are not
                                 # used by local search; enabling adds cost with
                                 # no benefit for this evaluation.
        embed_text:
          enabled: true
          # batch_max_tokens default is 8191 (sized for OpenAI text-embedding-3-small).
          # Lowered to 1700 as a safety net: the TokenTextSplitter uses cl100k_base
          # to count tokens, but the embedding model uses SentencePiece. Any text
          # over 1700 cl100k tokens is split and its chunk embeddings averaged,
          # keeping each individual embed request under the 2048 SentencePiece limit.
          batch_max_tokens: 1700
    """)

    with open(os.path.join(workspace_dir, "settings.yaml"), "w", encoding="utf-8") as f:
        f.write(settings_content)


# =============================================================================
# BEST-EFFORT TOKEN ACCOUNTING from GraphRAG's cache directory
# =============================================================================
# GraphRAG 2.7.0 cache file format — traced through the full call chain:
#
#   graphrag.language_model.providers.litellm.request_wrappers.with_cache
#     → cache.set(cache_key, {"response": response.model_dump()})
#   graphrag.cache.JsonPipelineCache.set
#     → writes: {"result": value, **(debug_data or {})}
#             = {"result": {"response": <OpenAI ModelResponse.model_dump()>}}
#
# So the full path to token usage is:
#   data["result"]["response"]["usage"]["prompt_tokens"]
#   data["result"]["response"]["usage"]["completion_tokens"]
#
# Embedding cache files have the same envelope but the OpenAI
# CreateEmbeddingResponse model_dump() has no "usage" field, so they
# contribute to api_calls but 0 to tokens — which is correct.
#
# NOTE on "reading from cache": each run uses a fresh timestamped workspace_dir
# so there is no cross-run cache reuse. The cache/ directory fills up during
# the current indexing run. Reading those files afterwards is the only way to
# get token counts because GraphRAG's LiteLLM provider does not expose a
# callback hook the way bare LiteLLM does.
def _parse_graphrag_cache(workspace_dir: str):
    cache_dir = os.path.join(workspace_dir, "cache")
    api_calls, prompt_tokens, completion_tokens = 0, 0, 0
    unrecognised = 0
    _printed_sample = False   # print the structure of the first file once for diagnostics

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

            # One-time structural diagnostic — shows top-level keys and whether
            # usage is where we expect it. Helps catch future schema shifts.
            if not _printed_sample:
                _printed_sample = True
                top_keys = list(data.keys()) if isinstance(data, dict) else type(data).__name__
                result_keys = list(data.get("result", {}).keys()) if isinstance(data.get("result"), dict) else "N/A"
                response_keys = list(data.get("result", {}).get("response", {}).keys()) \
                    if isinstance(data.get("result", {}).get("response"), dict) else "N/A"
                usage_val = data.get("result", {}).get("response", {}).get("usage")
                print(f"[Token Tracking] Cache file sample ({fname}):")
                print(f"  top-level keys    : {top_keys}")
                print(f"  result keys       : {result_keys}")
                print(f"  response keys     : {response_keys}")
                print(f"  result.response.usage: {usage_val}")

            usage = None
            # Search paths ordered most-likely-first.
            # Primary path for graphrag 2.7.0 litellm provider:
            #   result → response → usage
            # Fallbacks cover fnllm-based runs and older 0.3.x-style entries.
            for path in (
                ("result", "response", "usage"),   # GraphRAG 2.7.0 litellm — PRIMARY
                ("result", "usage"),               # GraphRAG fnllm provider fallback
                ("usage",),                        # flat / very old legacy
                ("metadata", "usage"),             # legacy variant
                ("output", "usage"),               # legacy variant
            ):
                node, ok = data, True
                for key in path:
                    if not isinstance(node, dict) or key not in node:
                        ok = False
                        break
                    node = node[key]
                if ok and isinstance(node, dict):
                    usage = node
                    break

            if usage:
                prompt_tokens     += int(usage.get("prompt_tokens", 0) or 0)
                completion_tokens += int(usage.get("completion_tokens", 0) or 0)
            else:
                unrecognised += 1

    if unrecognised:
        print(f"[Token Tracking] {unrecognised}/{api_calls} cache files had no recognisable "
              "usage field (normal for embedding entries; unexpected for chat entries).")
    return api_calls, prompt_tokens, completion_tokens


# =============================================================================
# GRAPH CONSTRUCTION
# =============================================================================
def build_graph(
    file_path: str,
    workspace_dir: str,
    model: str,
    embed_model: str,
    api_base: str,
    embed_api_base: str,
    api_key: str,
    chunk_size: int,
    chunk_overlap: int,
    max_gleanings: int,
    concurrent_requests: int,
    error_tracker: ErrorTracker,
):
    """Runs `graphrag index`, parses parquet artifacts, returns (output_dir, metrics)."""
    try:
        setup_graphrag_workspace(
            workspace_dir=workspace_dir,
            source_file=file_path,
            model=model,
            embed_model=embed_model,
            api_base=api_base,
            embed_api_base=embed_api_base,
            api_key=api_key,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            max_gleanings=max_gleanings,
            concurrent_requests=concurrent_requests,
        )
    except Exception:
        error_tracker.record("graph_construction")
        raise

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
    print("[Construction] Running `graphrag index` (output captured below)...")

    t0 = time.time()
    # Capture stdout+stderr so output flows through Python sys.stdout
    # (TeeStream) into run.log. Without capture, subprocess uses the OS-level
    # file descriptor and bypasses TeeStream entirely — on failure you only
    # see "exit code 1" with no indication of what actually went wrong.
    index_result = subprocess.run(
        ["python", "-m", "graphrag", "index", "--root", workspace_dir],
        capture_output=True,
        text=True,
    )
    execution_time = round(time.time() - t0, 2)

    # Always print captured output so it lands in run.log via TeeStream
    if index_result.stdout:
        print(index_result.stdout, end="")
    if index_result.stderr:
        # Mirror stderr to stdout so TeeStream picks it up for run.log
        print("[graphrag stderr]\n" + index_result.stderr)

    if index_result.returncode != 0:
        error_tracker.record("graph_construction")
        tail_lines = (index_result.stdout + "\n" + index_result.stderr).strip().split("\n")
        tail = "\n".join(tail_lines[-40:])
        raise RuntimeError(
            f"graphrag index returned exit code {index_result.returncode}.\n"
            f"Last 40 lines of output:\n{tail}"
        )

    output_dir = os.path.join(workspace_dir, "output")

    # ---- Parse parquet artifacts for graph metrics ----
    n_entities = n_edges = n_chunks = n_communities = n_reports = n_documents = 0
    artifact_files = []
    try:
        if os.path.isdir(output_dir):
            for fname in sorted(os.listdir(output_dir)):
                fpath = os.path.join(output_dir, fname)
                if os.path.isfile(fpath):
                    artifact_files.append({"file": fname,
                                           "size_bytes": os.path.getsize(fpath)})

        def _safe_count(name):
            p = os.path.join(output_dir, name)
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

    except Exception as e:
        error_tracker.record("graph_construction")
        print(f"[WARN] Could not read graph metrics: {e}")

    api_calls, prompt_tokens, completion_tokens = _parse_graphrag_cache(workspace_dir)

    metrics = {
        "total_document_tokens": total_document_tokens,
        "chunk_size":            chunk_size,
        "chunk_overlap":         chunk_overlap,
        "max_gleanings":         max_gleanings,
        "total_chunks":          n_chunks,
        "entities":              n_entities,
        "edges":                 n_edges,
        "communities":           n_communities,
        "community_reports":     n_reports,
        "documents":             n_documents,
        "llm_prompt_tokens":     prompt_tokens,
        "llm_completion_tokens": completion_tokens,
        "llm_total_tokens":      prompt_tokens + completion_tokens,
        "llm_api_calls":         api_calls,
        "execution_time_sec":    execution_time,
        "artifact_files":        artifact_files,
        "_token_tracking_note": (
            "Tokens extracted by walking GraphRAG's cache/ directory. "
            "GraphRAG uses fnllm (not LiteLLM) so callbacks cannot be injected; "
            "token counts are 0 if the cache schema omits usage info."
        ),
    }

    print(
        f"[Construction] Done in {execution_time}s | "
        f"Entities: {n_entities} | Edges: {n_edges} | "
        f"Chunks: {n_chunks} | Communities: {n_communities} | Reports: {n_reports}"
    )
    return output_dir, metrics


# =============================================================================
# CONTEXT BUILDER SETUP
# =============================================================================
def setup_local_context_builder(
    output_dir: str,
    embed_model: str,
    embed_api_base: str,
    api_key: str,
    community_level: int,
    embed_timeout: float = 120.0,
    embed_max_retries: int = 5,
):
    """
    Load parquet artifacts and instantiate LocalSearchMixedContext.

    Key changes vs old 0.3.x code:
      - LanceDBVectorStore now takes VectorStoreSchemaConfig(index_name=...)
        instead of collection_name=...
      - text_embedder is our LiteLLMEmbedder (satisfies EmbeddingModel Protocol)
      - tokenizer= uses graphrag's own get_tokenizer() instead of raw tiktoken
      - build_context() returns ContextBuilderResult, not a (str, dict) tuple
    """
    entity_path       = os.path.join(output_dir, "entities.parquet")
    relationship_path = os.path.join(output_dir, "relationships.parquet")
    text_unit_path    = os.path.join(output_dir, "text_units.parquet")
    community_path    = os.path.join(output_dir, "communities.parquet")
    report_path       = os.path.join(output_dir, "community_reports.parquet")

    for required in [entity_path, relationship_path, text_unit_path]:
        if not os.path.exists(required):
            raise FileNotFoundError(
                f"GraphRAG output missing: {required} — "
                "indexing probably did not complete successfully."
            )

    entity_df       = pd.read_parquet(entity_path)
    relationship_df = pd.read_parquet(relationship_path)
    text_unit_df    = pd.read_parquet(text_unit_path)

    community_df = (pd.read_parquet(community_path)
                    if os.path.exists(community_path) else pd.DataFrame())

    if os.path.exists(report_path) and len(community_df) > 0:
        report_df = pd.read_parquet(report_path)
        reports = read_indexer_reports(report_df, community_df, community_level)
    else:
        reports = []
        print("[WARN] No community reports found — community_prop will be forced to 0.")

    entities      = read_indexer_entities(entity_df, community_df, community_level)
    relationships = read_indexer_relationships(relationship_df)
    text_units    = read_indexer_text_units(text_unit_df)

    # ---- LanceDB vector store (API changed in 2.7.0) ----
    # Old:  LanceDBVectorStore(collection_name="default-entity-description")
    # New:  LanceDBVectorStore(VectorStoreSchemaConfig(index_name=...))
    description_embedding_store = LanceDBVectorStore(
        vector_store_schema_config=VectorStoreSchemaConfig(
            index_name="default-entity-description"
        )
    )
    description_embedding_store.connect(
        db_uri=os.path.join(output_dir, "lancedb")
    )

    # ---- Our LiteLLM embedding wrapper ----
    text_embedder = LiteLLMEmbedder(
        model=embed_model,
        api_base=embed_api_base,
        api_key=api_key,
        timeout=embed_timeout,
        max_retries=embed_max_retries,
    )

    # ---- graphrag's own tokenizer (replaces raw tiktoken in 0.3.x) ----
    tokenizer = graphrag_get_tokenizer()   # defaults to cl100k_base TiktokenTokenizer

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

    print(
        f"[Retrieval] Loaded {len(entities)} entities, {len(relationships)} "
        f"relationships, {len(text_units)} text units, {len(reports)} reports."
    )
    return context_builder


def _retrieve_context(context_builder, query: str, retrieval_params: dict) -> str:
    """
    Call LocalSearchMixedContext.build_context() and return a plain string.

    In graphrag 2.7.0, build_context() returns a ContextBuilderResult dataclass
    (not a (str, dict) tuple as in 0.3.x). The context is in .context_chunks
    which is typed str | list[str].
    """
    result = context_builder.build_context(
        query=query, conversation_history=None, **retrieval_params
    )
    # graphrag 2.7.0: result is ContextBuilderResult
    ctx = result.context_chunks
    if isinstance(ctx, list):
        return "\n\n".join(str(x) for x in ctx)
    return ctx if isinstance(ctx, str) else str(ctx)


# =============================================================================
# QA EVALUATION WORKER
# =============================================================================
def _eval_qa_item(
    pair, context_builder, retrieval_params,
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
        raw_context = _retrieve_context(context_builder, q, retrieval_params)
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
    fact_dict, context_builder, retrieval_params,
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
        raw_context = _retrieve_context(context_builder, fact, retrieval_params)
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
    qa_file, context_builder, retrieval_params,
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
                _eval_qa_item, p, context_builder, retrieval_params,
                tokenizer, max_context_tokens, error_tracker,
            ): i for i, p in enumerate(pairs)
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
    fact_file, context_builder, retrieval_params,
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
                _eval_fact_item, fd, context_builder, retrieval_params,
                tokenizer, max_context_tokens, error_tracker,
            ): i for i, fd in enumerate(facts)
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
    api_base: str,
    embed_api_base: str,
    api_key: str,
    chunk_size: int,
    chunk_overlap: int,
    max_gleanings: int,
    concurrent_requests: int,
    max_workers: int,
    retrieval_top_k_entities: int,
    retrieval_top_k_relationships: int,
    retrieval_max_tokens: int,
    community_level: int,
    text_unit_prop: float,
    community_prop: float,
    max_context_tokens: int,
    judge_max_tokens: int,
    tokenizer,
):
    contract_name = os.path.splitext(os.path.basename(contract_path))[0]
    eval_tag      = "_".join(
        (["qa"] if qa_file else []) + (["fact"] if fact_file else [])
    )
    timestamp     = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir       = os.path.join(out_dir, f"{contract_name}_{eval_tag}_{timestamp}")
    workspace_dir = os.path.join(run_dir, "graphrag_workspace")
    os.makedirs(run_dir, exist_ok=True)

    log_path   = os.path.join(run_dir, "run.log")
    tee        = TeeStream(log_path, sys.stdout)
    sys.stdout = tee

    error_tracker = ErrorTracker()

    retrieval_params = {
        "text_unit_prop":              text_unit_prop,
        "community_prop":              community_prop,
        "top_k_mapped_entities":       retrieval_top_k_entities,
        "top_k_relationships":         retrieval_top_k_relationships,
        "max_context_tokens":          retrieval_max_tokens,
        "include_entity_rank":         True,
        "include_relationship_weight": True,
        "include_community_rank":      False,
        "return_candidate_context":    False,
        "use_community_summary":       False,
    }

    print("\n" + "=" * 65)
    print(f" CONTRACT: {contract_name}")
    print(f" Output  : {run_dir}")
    print("=" * 65)

    # ---- Phase 1: Construction ----
    print("\n[*] PHASE 1 — GRAPH CONSTRUCTION")
    try:
        output_dir, construction_metrics = build_graph(
            file_path=contract_path,
            workspace_dir=workspace_dir,
            model=model,
            embed_model=embed_model,
            api_base=api_base,
            embed_api_base=embed_api_base,
            api_key=api_key,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            max_gleanings=max_gleanings,
            concurrent_requests=concurrent_requests,
            error_tracker=error_tracker,
        )
    except Exception as e:
        print(f"[FATAL] Construction failed for {contract_name}: {e}")
        traceback.print_exc()
        sys.stdout = tee._stream
        tee.close()
        return

    print(f"[*] Output stored in → {output_dir}")

    # ---- Phase 2: Build context-only retrieval engine ----
    print("\n[*] PHASE 2 — BUILDING CONTEXT-ONLY RETRIEVAL ENGINE")
    try:
        context_builder = setup_local_context_builder(
            output_dir=output_dir,
            embed_model=embed_model,
            embed_api_base=embed_api_base,
            api_key=api_key,
            community_level=community_level,
        )
        if not getattr(context_builder, "community_reports", None):
            retrieval_params["community_prop"] = 0.0
            print("[*] community_prop forced to 0 (no community reports loaded).")
    except Exception as e:
        error_tracker.record("context_builder_setup")
        print(f"[FATAL] Context builder setup failed for {contract_name}: {e}")
        traceback.print_exc()
        sys.stdout = tee._stream
        tee.close()
        return

    # ---- Phase 3: Evaluation ----
    qa_eval_result   = None
    fact_eval_result = None

    if qa_file:
        print("\n[*] PHASE 3a — QA EVALUATION")
        qa_eval_result = run_qa_evaluation(
            qa_file=qa_file,
            context_builder=context_builder,
            retrieval_params=retrieval_params,
            tokenizer=tokenizer,
            max_context_tokens=max_context_tokens,
            max_workers=max_workers,
            error_tracker=error_tracker,
        )

    if fact_file:
        print("\n[*] PHASE 3b — FACT RECALL EVALUATION")
        fact_eval_result = run_fact_evaluation(
            fact_file=fact_file,
            context_builder=context_builder,
            retrieval_params=retrieval_params,
            tokenizer=tokenizer,
            max_context_tokens=max_context_tokens,
            max_workers=max_workers,
            error_tracker=error_tracker,
        )

    error_summary = error_tracker.summary()

    final_output = {
        "run_id":    f"graphrag_{uuid.uuid4().hex[:8]}",
        "timestamp": timestamp,
        "system":    "GRAPHRAG",

        "metadata": {
            "contract_file":       os.path.abspath(contract_path),
            "qa_file":             qa_file,
            "fact_file":           fact_file,
            "llm_model":           model,
            "embedding_model":     embed_model,
            "api_base":            api_base,
            "embed_api_base":      embed_api_base,
            "concurrent_requests": concurrent_requests,
            "retrieval_params":    retrieval_params,
            "community_level":     community_level,
            "max_context_tokens":  max_context_tokens,
            "judge_max_tokens":    judge_max_tokens,
            "chunking": {
                "chunk_size":    chunk_size,
                "chunk_overlap": chunk_overlap,
                "max_gleanings": max_gleanings,
            },
            "workspace_dir": workspace_dir,
            "output_dir":    output_dir,
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
    print(f"\n  Workspace → {workspace_dir}")
    print(f"  Output    → {output_dir}")
    print(f"  Log       → {log_path}")
    print(f"  Results   → {results_path}")
    print(f"  Document tokens   : {cm['total_document_tokens']}")
    print(f"  Chunks            : {cm['total_chunks']}  "
          f"({cm['chunk_size']} tok, {cm['chunk_overlap']} overlap, "
          f"{cm['max_gleanings']} gleanings)")
    print(f"  Entities          : {cm['entities']}")
    print(f"  Edges             : {cm['edges']}")
    print(f"  Communities       : {cm['communities']}")
    print(f"  Community reports : {cm['community_reports']}")
    print(f"  LLM tokens (cache): {cm['llm_total_tokens']}  ({cm['llm_api_calls']} cache entries)")
    print(f"  Wall time         : {cm['execution_time_sec']}s")
    if cm.get("artifact_files"):
        print(f"  Artifacts ({len(cm['artifact_files'])} files):")
        for af in cm["artifact_files"]:
            print(f"    {af['file']:45s}  {af['size_bytes']:>10,} bytes")
    if qa_eval_result:
        print(f"  QA (excl CAT5)    : {qa_eval_result['overall_avg_score_excl_cat5']}")
        for cat, s in qa_eval_result["by_category"].items():
            print(f"    {cat}: {s['avg_score']}  (n={s['count']})")
    if fact_eval_result:
        print(f"  Fact Recall       : {fact_eval_result['overall_recall_score']}  "
              f"({fact_eval_result['total_recalled']}/{fact_eval_result['total_facts']})")
    print(f"\n  Errors (total)    : {error_summary['total_errors']}")
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
        description="GraphRAG 2.7.0: single-script construction + QA/Fact evaluation"
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

    # ---- Models & API ----
    parser.add_argument("--model",          type=str,
                        default="Qwen/Qwen3-30B-A3B-Instruct-2507-FP8")
    parser.add_argument("--embed-model",    type=str,
                        default="Qwen/Qwen3-Embedding-4B")
    parser.add_argument("--api-base",       type=str,
                        default="http://localhost:8000/v1")
    parser.add_argument("--embed-api-base", type=str,
                        default="http://localhost:8001/v1")
    parser.add_argument("--api-key",        type=str, default="EMPTY")

    # ---- Construction ----
    parser.add_argument("--chunk-size",          type=int,   default=600)
    parser.add_argument("--chunk-overlap",       type=int,   default=50)
    parser.add_argument("--max-gleanings",       type=int,   default=1)
    parser.add_argument("--concurrent-requests", type=int,   default=25,
                        help="Indexer concurrent_requests (chat and embedding).")

    # ---- Evaluation parallelism ----
    parser.add_argument("--max-workers", type=int, default=10,
                        help="Thread-pool size for QA/Fact evaluation.")

    # ---- Retrieval ----
    parser.add_argument("--retrieval-top-k-entities",      type=int,   default=10)
    parser.add_argument("--retrieval-top-k-relationships", type=int,   default=10)
    parser.add_argument("--retrieval-max-tokens",          type=int,   default=8000,
                        help="Context builder's internal token budget (before truncation).")
    parser.add_argument("--community-level",               type=int,   default=2)
    parser.add_argument("--text-unit-prop",                type=float, default=0.5)
    parser.add_argument("--community-prop",                type=float, default=0.1)
    parser.add_argument("--max-context-tokens",            type=int,   default=2000,
                        help="Final context budget passed to the LLM judge.")

    # ---- Judge LLM ----
    parser.add_argument("--judge-max-tokens", type=int, default=8000)

    args = parser.parse_args()

    # ---- Validate ----
    multi_mode = args.all or args.ten_contracts_dataset or args.long_form_contracts
    if multi_mode and not args.eval_data_dir:
        parser.error("--all, --ten-contracts-dataset, and --long-form-contracts require --eval-data-dir")
    if args.contract and not args.qa_file and not args.fact_file:
        parser.error("Single-contract mode requires at least one of --qa-file or --fact-file")

    configure_dspy(max_tokens=args.judge_max_tokens)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    if args.all:
        subdirs_to_scan = DATASET_SUBDIRS;  mode_label = "--all"
    elif args.ten_contracts_dataset:
        subdirs_to_scan = ["ten_contracts_dataset"];  mode_label = "--ten-contracts-dataset"
    elif args.long_form_contracts:
        subdirs_to_scan = ["long_form_contracts"];  mode_label = "--long-form-contracts"
    else:
        subdirs_to_scan = None

    shared = dict(
        out_dir=args.out_dir,
        model=args.model,
        embed_model=args.embed_model,
        api_base=args.api_base,
        embed_api_base=args.embed_api_base,
        api_key=args.api_key,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        max_gleanings=args.max_gleanings,
        concurrent_requests=args.concurrent_requests,
        max_workers=args.max_workers,
        retrieval_top_k_entities=args.retrieval_top_k_entities,
        retrieval_top_k_relationships=args.retrieval_top_k_relationships,
        retrieval_max_tokens=args.retrieval_max_tokens,
        community_level=args.community_level,
        text_unit_prop=args.text_unit_prop,
        community_prop=args.community_prop,
        max_context_tokens=args.max_context_tokens,
        judge_max_tokens=args.judge_max_tokens,
        tokenizer=tokenizer,
    )

    if multi_mode:
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