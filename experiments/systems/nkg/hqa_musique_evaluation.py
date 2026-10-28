#!/usr/bin/env python3
"""
nkg_multihop_eval.py
=====================
NKG evaluation harness for HotpotQA and MuSiQue — mirrors the two-mode
construction/evaluation interface used by the GraphRAG / KGGen / HyperGraphRAG
multihop eval scripts (experiments/systems/{graphrag,kggen,hypergraphrag}/
hqa_musique_evaluation.py), and reuses the corpus/questions JSONL schema and
EM/F1 scoring from experiments/utils/multihop_utils.py.

Two modes:
  construction  Builds one NKG over the corpus with nkg.index.construction.build_index.
                Saves kg.graphml + construction_results.json in --run-dir.
                Skipped if construction_results.json already exists
                (pass --force-rebuild to override).
  evaluation    Loads the saved kg.graphml, answers questions in parallel via
                the NKG retrieval engine + LLM, scores with official EM/F1,
                saves evaluation_results.json, or evaluation_results_1.json,
                evaluation_results_2.json, etc. if that file already exists (never overwrites
                a prior run's results).

INPUT MODE (--input-mode)
--------------------------
  concat_file (default) — all corpus paragraphs are concatenated into one
      corpus.txt, separated by a document-boundary marker, and NKG's own
      token-based chunker (chunk_size/overlap) splits it into chunks. Chunk
      boundaries may therefore straddle two unrelated paragraphs.
  prepared_chunks — each corpus paragraph is passed to build_index() as its
      own final chunk (chunks=[...], chunk_list_inputs=False) with NO further
      token-based splitting. Use this when you want chunk boundaries to
      exactly match paragraph boundaries.

CHECKPOINTING / RESUME
-----------------------
Construction is checkpointed under --run-dir/checkpoint/ (chunk-level, then
merge+dedup, then per-cluster within each global-edge round — see
nkg/index/construction/checkpointing.py). Re-running the same
`--mode construction --run-dir X` command resumes automatically instead of
starting over. You do NOT need to re-pass construction flags on resume —
they're loaded from --run-dir/construction_config.json. If you pass a
construction flag that conflicts with the saved config (other than
--max-workers/--max-sub-workers, which may always change), you'll be warned
and asked to confirm before checkpoint progress is discarded and construction
restarts from scratch with the new values.

RETRIEVAL ENGINE (--engine)
----------------------------
  new (default) — nkg.search.engine.GoalSearchEngine (goal-coverage redesign)
  old           — nkg.retrieval.engine.Retriever (original MMR-based engine)
  dag           — nkg.dag_retrieval.DAGRetriever (decomposes the query into a DAG of
                   subqueries, refining/disambiguating and forking as it goes, and
                   searches each subquery with the old engine)

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

  Note: chunking uses a separate --chunk-tokenizer-model (default a Qwen HF
  tokenizer) so that --provider deepseek doesn't try to load a HuggingFace
  tokenizer named after a non-HF model id.

STRATIFIED SUBSETS (e.g. "100 questions instead of 1000")
------------------------------------------------------------
Already supported upstream — no flags needed here. Generate a smaller
stratified sample directly from the prep scripts:
  python hqa_prep.py     --input ... --sample-size 100 --seed 42 \\
      --out-corpus ./data/hotpot_corpus_100.jsonl --out-questions ./data/hotpot_questions_100.jsonl
  python musique_prep.py --input ... --sample-size 100 --seed 42 \\
      --out-corpus ./data/musique_corpus_100.jsonl --out-questions ./data/musique_questions_100.jsonl
Then point --corpus-file/--questions-file at the *_100.jsonl files as usual.

EXAMPLE USAGE
-------------
  # Build the index
  python -m experiments.systems.nkg.hqa_musique_evaluation \\
      --mode construction --dataset hotpotqa \\
      --corpus-file ./data/hotpot_corpus_1k.jsonl \\
      --run-dir ./results/hotpotqa/nkg/run1

  # Resume an interrupted construction run — no construction flags needed
  python -m experiments.systems.nkg.hqa_musique_evaluation \\
      --mode construction --dataset hotpotqa --run-dir ./results/hotpotqa/nkg/run1

  # Run evaluation (new engine, default retrieval args)
  python -m experiments.systems.nkg.hqa_musique_evaluation \\
      --mode evaluation --dataset hotpotqa \\
      --questions-file ./data/hotpot_questions_1k.jsonl \\
      --run-dir ./results/hotpotqa/nkg/run1 \\
      --default-retrieval-args --max-workers 20

  # DeepSeek provider, old engine, custom chunk size
  python -m experiments.systems.nkg.hqa_musique_evaluation \\
      --mode construction --provider deepseek --dataset musique \\
      --corpus-file ./data/musique_corpus_1k.jsonl \\
      --run-dir ./results/musique/nkg/run_ds \\
      --chunk-size 600 --chunk-overlap 50
"""

import os
import sys
import json
import time
import uuid
import argparse
import traceback
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer

from dotenv import load_dotenv

from nkg.utils.config import configure_dspy
from nkg.index.construction.build_index import build_index
from nkg.index.construction.checkpointing import ConstructionCheckpoint
from nkg.retrieval.engine import Retriever
from nkg.search.engine import GoalSearchEngine
from nkg.dag_retrieval import DAGRetriever

from experiments.utils.multihop_utils import (
    HOTPOTQA, MUSIQUE, SUPPORTED_DATASETS,
    load_corpus_jsonl, load_questions_jsonl,
    GlobalTokenAccumulator, attach_token_tracker, detach_token_tracker,
    call_llm_for_answer, score_em, score_f1, truncate_tokens, safe_div,
    add_token_counters, next_eval_results_path,
)


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
        model="deepseek-v4-flash",
        api_base="https://api.deepseek.com",
        api_key="",
    ),
}


def resolve_provider(args) -> ProviderConfig:
    provider = args.provider or "local_vllm"
    base     = PROVIDER_DEFAULTS[provider]
    model    = args.model    or base.model
    api_base = args.api_base or base.api_base
    api_key  = args.api_key  or base.api_key
    if provider == "deepseek" and not api_key:
        api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        if not api_key:
            print("[WARN] provider=deepseek: no api-key found. "
                  "Set DEEPSEEK_API_KEY env var or pass --api-key.")
    return ProviderConfig(provider=provider, model=model,
                          api_base=api_base, api_key=api_key)


def resolve_temperature(value):
    """--temperature: a number, or 'none' = do not send a temperature (dspy.LM omits a None temperature)."""
    if value is None or str(value).strip().lower() in ("none", ""):
        return None
    return float(value)


def dspy_model_string(cfg: ProviderConfig) -> str:
    """DSPy/LiteLLM routing string, e.g. 'openai/Qwen/...' or 'deepseek/deepseek-v4-flash'."""
    if cfg.provider == "local_vllm":
        return f"openai/{cfg.model}"
    if cfg.provider == "deepseek":
        return f"deepseek/{cfg.model}"
    return cfg.model


# =============================================================================
# ENGINE SELECTION (same defaults as insurance_evaluation.py)
# =============================================================================

ENGINE_CHOICES = ["new", "old", "dag"]
DEFAULT_ENGINE = "new"

DEFAULT_RETRIEVAL_ARGS_OLD = {
    "top_k_seeds":        9,
    "max_depth":          3,
    "beam_width":         6,
    "final_top_k":        20,
    "mode":               "all",
    "beam_mmr_lambda":    0.6,
    "final_mmr_lambda":   0.6,
    "search_mode":        "beam_search",
    "deduplicate_paths":  True,
    "neutral_layout":     True,
}

DEFAULT_RETRIEVAL_ARGS_NEW = {
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

# Hardcoded retrieval defaults for the DAG engine (nkg.dag_retrieval.DAGRetriever, wrapping the
# OLD engine). max_depth=1 (vs. the old/new engines' 3) is intentional: each DAG node is already
# one focused sub-lookup -- the multi-hop reasoning lives in the DAG structure itself.
DEFAULT_RETRIEVAL_ARGS_DAG = {
    "top_k_seeds":        9,
    "max_depth":          1,
    "beam_width":         6,
    "final_top_k":        20,
    "mode":               "all",
    "beam_mmr_lambda":    0.6,
    "final_mmr_lambda":   0.6,
    "search_mode":        "beam_search",
    "deduplicate_paths":  True,
    "neutral_layout":     True,
    "max_fork_branches":  6,
    "parallel_fallback_chain_threshold": 2,
}


def default_retrieval_args(engine: str) -> dict:
    if engine == "new":
        return dict(DEFAULT_RETRIEVAL_ARGS_NEW)
    if engine == "dag":
        return dict(DEFAULT_RETRIEVAL_ARGS_DAG)
    return dict(DEFAULT_RETRIEVAL_ARGS_OLD)


def build_retriever(engine: str, bi_encoder, cross_encoder, kg_path: str):
    if engine == "new":
        return GoalSearchEngine(
            retrieval_model=bi_encoder,
            cross_encoder=cross_encoder,
            graph_path=kg_path,
        )
    if engine == "dag":
        inner = Retriever(
            retrieval_model=bi_encoder,
            cross_encoder=cross_encoder,
            graph_path=kg_path,
            entity_weights={"name": 0.6, "role": 0.4, "anchors": 0.0},
            fact_weights={"sentence": 0.8, "questions": 0.0, "chunk": 0.2},
        )
        return DAGRetriever(inner)
    return Retriever(
        retrieval_model=bi_encoder,
        cross_encoder=cross_encoder,
        graph_path=kg_path,
        entity_weights={"name": 0.6, "role": 0.4, "anchors": 0.0},
        fact_weights={"sentence": 0.8, "questions": 0.0, "chunk": 0.2},
    )


# =============================================================================
# CORPUS MATERIALIZATION
# NKG's build_index() takes either a single filepath or a pre-chunked list of
# strings. In --input-mode concat_file, all paragraphs in the corpus are
# concatenated into one .txt document, separated by a document boundary
# marker so chunk-level context stays legible even if a chunk straddles two
# paragraphs. In --input-mode prepared_chunks, each paragraph becomes its own
# final chunk and no concatenated file is written.
# =============================================================================

def write_corpus_file(corpus: dict, out_path: str) -> int:
    """
    Writes every paragraph in `corpus` (id -> entry dict) to a single
    concatenated .txt file at out_path. Returns total character count.
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    parts = []
    for _pid, entry in sorted(corpus.items()):
        parts.append(f"Title: {entry['title']}\n\n{entry['paragraph']}")
    content = "\n\n---\n\n".join(parts)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)
    return len(content)


def corpus_to_prepared_chunks(corpus: dict) -> list:
    """One paragraph -> one final chunk string, in stable (sorted-id) order."""
    return [f"Title: {entry['title']}\n\n{entry['paragraph']}" for _pid, entry in sorted(corpus.items())]


# =============================================================================
# CONSTRUCTION CONFIG PERSISTENCE / RESUME
# Lets a resumed `--mode construction --run-dir X` invocation omit every
# construction flag: any flag left unset on the CLI (default=None below)
# falls back to what's saved in run-dir/construction_config.json, which in
# turn falls back to the hardcoded defaults in _CONSTRUCTION_DEFAULTS on a
# genuinely fresh run. Flags explicitly re-passed with a *different* value
# than what's saved trigger a warn+confirm+restart-from-scratch flow (see
# _resolve_and_persist_construction_args). --max-workers/--max-sub-workers
# are pure concurrency knobs and are exempt from all of this.
# =============================================================================

CONSTRUCTION_CONFIG_FILE = "construction_config.json"

_CONSTRUCTION_DEFAULTS = {
    "provider": "local_vllm",
    "model": "",
    "api_base": "",
    "input_mode": "concat_file",
    "chunk_size": 600,
    "chunk_overlap": 50,
    "chunk_tokenizer_model": "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
    "edge_threshold": 0.5,
    "top_k": 12,
    "fact_batch_size": 15,
    "edge_mode": "clustered",
    "deduplication": True,
    "dedup_model_name": "voyageai/voyage-4-nano",
    "dedup_iterations": 1,
    "dedup_max_cluster_size": 64,
    "dedup_run_exact": True,
    "dedup_run_semhash": True,
    "dedup_semhash_threshold": 0.95,
    "dedup_run_llm": True,
    "dedup_similarity_threshold": 0.75,
    "use_cot": False,
    "neutral_description": True,
    "concise_summary": True,
    "combined_entity_extraction": True,
    "full_entity_profile": False,
    "clustered_round1": False,
    "clustered_round2": True,
    "clustered_round3": True,
    "clustered_max_facts_per_prompt": 40,
    "middle_k_skip_percent": 0.33,
    "global_edge_retrieval_model": "voyageai/voyage-4-nano",
    "use_questions_weight": True,
    "questions_mode": "follow_up_to_answered",
    "same_chunk_min_facts": 2,
    "generate_entity_relation_labels": False,
    "dedup_sim_weight_name": 0.6,
    "dedup_sim_weight_role": 0.3,
    "dedup_sim_weight_type": 0.1,
    # fact->fact edge descriptions (False = label-only edge prompts) and post-entity-dedup fact dedup;
    # both default to build_index()'s own defaults
    "edge_descriptions": True,
    "run_fact_dedup": True,
    # fact dedup: similarity-graph neighbours per fact and facts per LLM dedup call (build_index's defaults)
    "fact_dedup_top_k": 12,
    "fact_dedup_cluster_size": 8,
    # edge_mode="clustered_fast" only (nkg/index/construction/fast_fact_clustering.py); build_index's defaults
    "fast_cluster_embed_loader": "remote_code",
    "fast_cluster_leiden_levels": "final",
    "fast_cluster_leiden_passes": 1,
    "fast_cluster_min_size": 0,
    "fact_prompt": "default",
    "fact_edge_prompt": "default",
    "global_fact_edge_prompt": None,
}

# max_workers/max_sub_workers/num_retries/construction_max_tokens are deliberately
# excluded — pure execution/robustness knobs, always taken fresh from the CLI (or
# its normal argparse default), never diffed/warned about, never persisted as part
# of the resumable config.


def _load_construction_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_construction_config(path: str, resolved_args: dict, override_history: list):
    payload = {
        "saved_at": datetime.now().isoformat(),
        "args": resolved_args,
        "override_history": override_history,
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)


def _diff_construction_args(saved_args: dict, resolved_args: dict) -> dict:
    diffs = {}
    for k, new_v in resolved_args.items():
        if k not in saved_args and _CONSTRUCTION_DEFAULTS.get(k, object()) == new_v:
            continue  # a key added to the defaults after this run was saved, at its default: not a change
        old_v = saved_args.get(k, "<unset>")
        if old_v != new_v:
            diffs[k] = (old_v, new_v)
    return diffs


def _confirm(assume_yes: bool, prompt: str) -> bool:
    if assume_yes:
        return True
    if not sys.stdin.isatty():
        return False
    try:
        resp = input(prompt).strip().lower()
    except EOFError:
        return False
    return resp in ("y", "yes")


def _resolve_and_persist_construction_args(args) -> None:
    """
    Merges CLI-supplied construction args with any saved construction_config.json
    in args.run_dir and mutates `args` in place with the fully-resolved values, so
    the rest of the script can just read args.* as usual. See the module docstring
    for the resume/override-warning behavior.
    """
    results_path   = os.path.join(args.run_dir, CONSTRUCTION_RESULTS_FILE)
    config_path    = os.path.join(args.run_dir, CONSTRUCTION_CONFIG_FILE)
    checkpoint_dir = os.path.join(args.run_dir, "checkpoint")

    if os.path.exists(results_path) and not args.force_rebuild:
        return  # already complete; run_construction() will print the skip message

    saved = _load_construction_config(config_path) if os.path.exists(config_path) else None
    saved_args = (saved or {}).get("args", {})

    resolved = {}
    for key, default in _CONSTRUCTION_DEFAULTS.items():
        cli_val = getattr(args, key)
        if cli_val is not None:
            resolved[key] = cli_val
        elif key in saved_args:
            resolved[key] = saved_args[key]
        else:
            resolved[key] = default

    if args.corpus_file:
        resolved["corpus_file"] = os.path.abspath(args.corpus_file)
    elif "corpus_file" in saved_args:
        resolved["corpus_file"] = saved_args["corpus_file"]
    else:
        resolved["corpus_file"] = None  # main() checks this after we return

    resolved["dataset"] = args.dataset

    for key, value in resolved.items():
        if hasattr(args, key):
            setattr(args, key, value)

    if args.force_rebuild:
        print("[Construction] --force-rebuild: wiping any existing checkpoint/config, starting fresh.")
        ConstructionCheckpoint(checkpoint_dir).wipe()
        if os.path.exists(config_path):
            os.remove(config_path)
        _save_construction_config(config_path, resolved, override_history=[])
        return

    if saved is None:
        _save_construction_config(config_path, resolved, override_history=[])
        return

    diffs = _diff_construction_args(saved_args, resolved)
    if not diffs:
        print(f"[Construction] Resuming from {args.run_dir} — construction args match the saved config.")
        return

    print(f"\n[Construction] WARNING: construction arguments differ from the checkpoint "
          f"already in {args.run_dir}:\n")
    for k in sorted(diffs):
        old_v, new_v = diffs[k]
        print(f"    {k}: {old_v!r}  ->  {new_v!r}")
    print("\n[Construction] Continuing will DISCARD all checkpoint progress in this run-dir "
          "and restart construction from scratch using the new values above.\n")

    if not _confirm(args.yes, "Continue and restart from scratch? [y/N] "):
        print("[Construction] Aborted. Re-run with matching arguments to resume, or pass "
              "--force-rebuild if you intend to start over.")
        sys.exit(1)

    override_entry = {
        "timestamp": datetime.now().isoformat(),
        "changed": {k: {"old": v[0], "new": v[1]} for k, v in diffs.items()},
    }
    history = (saved.get("override_history") or []) + [override_entry]
    ConstructionCheckpoint(checkpoint_dir).wipe()
    _save_construction_config(config_path, resolved, override_history=history)


# =============================================================================
# GRAPH CONSTRUCTION
# =============================================================================

CONSTRUCTION_RESULTS_FILE = "construction_results.json"
KG_FILENAME               = "kg.graphml"


def _print_token_breakdown(breakdown: dict):
    lw = 30
    def row(label, d, indent=0):
        prefix = "  " * indent
        print(f"    {prefix}{label:<{lw - 2 * indent}} total={d.get('total', 0):>9,}  "
              f"(prompt={d.get('prompt', 0):>9,}, compl={d.get('completion', 0):>9,}, calls={d.get('calls', 0):>4})")
    print("[Construction] Token breakdown by stage:")
    row("chunk_extraction",     breakdown.get("chunk_extraction",    {}))
    sub = breakdown.get("chunk_extraction_sub", {})
    for sub_name, sub_data in sub.items():
        row(sub_name, sub_data, indent=1)
    row("entity_deduplication", breakdown.get("entity_deduplication", {}))
    row("global_edges_round1",  breakdown.get("global_edges_round1", {}))
    row("global_edges_round2",  breakdown.get("global_edges_round2", {}))
    row("global_edges_round3",  breakdown.get("global_edges_round3", {}))
    print(f"    {'─' * (lw + 58)}")
    row("TOTAL",                breakdown.get("total", {}))


def run_construction(args, cfg: ProviderConfig):
    os.makedirs(args.run_dir, exist_ok=True)

    results_path = os.path.join(args.run_dir, CONSTRUCTION_RESULTS_FILE)
    if os.path.exists(results_path) and not args.force_rebuild:
        print(f"[Construction] {CONSTRUCTION_RESULTS_FILE} already exists — skipping.")
        print(f"[Construction] Pass --force-rebuild to rebuild from scratch.")
        return

    checkpoint_dir = os.path.join(args.run_dir, "checkpoint")
    config_path    = os.path.join(args.run_dir, CONSTRUCTION_CONFIG_FILE)

    print(f"[Construction] Loading corpus: {args.corpus_file}")
    corpus = load_corpus_jsonl(args.corpus_file)
    print(f"[Construction]   {len(corpus):,} unique paragraphs")

    if args.input_mode == "per_document":
        print(f"[Construction]   input-mode=per_document — each document is chunked on its own "
              f"({args.chunk_size}/{args.chunk_overlap} tokens), no chunk crosses documents.")
        prepared_chunks = corpus_to_prepared_chunks(corpus)
        total_chars = sum(len(c) for c in prepared_chunks)
        corpus_txt_path = None
        filepath_arg, chunks_arg = None, prepared_chunks
    elif args.input_mode == "prepared_chunks":
        print(f"[Construction]   input-mode=prepared_chunks — each paragraph becomes exactly "
              f"one final chunk, no further token-based splitting.")
        prepared_chunks = corpus_to_prepared_chunks(corpus)
        total_chars = sum(len(c) for c in prepared_chunks)
        corpus_txt_path = None
        filepath_arg, chunks_arg = None, prepared_chunks
    else:
        corpus_txt_path = os.path.join(args.run_dir, "corpus.txt")
        total_chars = write_corpus_file(corpus, corpus_txt_path)
        print(f"[Construction]   Wrote concatenated corpus → {corpus_txt_path}  "
              f"({total_chars:,} chars)")
        filepath_arg, chunks_arg = corpus_txt_path, None

    questions_w = 1.0 if args.use_questions_weight else 0.0
    r2_weights = {"sent": 1.0, "topic": 1.0, "entity": 1.0, "questions": questions_w}
    r3_weights = {"sent": 1.0, "topic": 1.0, "entity": 1.0, "questions": questions_w}
    dedup_sim_weights = {
        "name": args.dedup_sim_weight_name,
        "role": args.dedup_sim_weight_role,
        "type": args.dedup_sim_weight_type,
    }

    configure_dspy(
        model=dspy_model_string(cfg), api_base=cfg.api_base, api_key=cfg.api_key,
        max_tokens=args.construction_max_tokens, cache=False, num_retries=args.num_retries,
        temperature=resolve_temperature(args.temperature),
    )

    print(f"[Construction] Building NKG "
          f"(chunk_size={args.chunk_size}, overlap={args.chunk_overlap}, "
          f"edge_mode={args.edge_mode}, dedup={args.deduplication}, "
          f"checkpoint_dir={checkpoint_dir})...")

    t0 = time.time()
    graph, token_breakdown = build_index(
        filepath=filepath_arg,
        chunks=chunks_arg,
        chunk_list_inputs=args.input_mode == "per_document",  # per_document: build_index chunks each document
        model_name=args.chunk_tokenizer_model,
        max_workers=args.max_workers,
        chunk_size=args.chunk_size,
        overlap=args.chunk_overlap,
        edge_threshold=args.edge_threshold,
        top_k=args.top_k,
        max_sub_workers=args.max_sub_workers,
        fact_batch_size=args.fact_batch_size,
        edge_mode=args.edge_mode,
        run_deduplication=args.deduplication,
        dedup_model_name=args.dedup_model_name,
        dedup_iterations=args.dedup_iterations,
        dedup_max_cluster_size=args.dedup_max_cluster_size,
        use_cot=args.use_cot,
        neutral_description=args.neutral_description,
        concise_summary=args.concise_summary,
        clustered_round1=args.clustered_round1,
        clustered_round2=args.clustered_round2,
        clustered_round3=args.clustered_round3,
        clustered_max_facts_per_prompt=args.clustered_max_facts_per_prompt,
        middle_k_skip_percent=args.middle_k_skip_percent,
        retrieval_model_name=args.global_edge_retrieval_model,
        construction_r2_weights=r2_weights,
        construction_r3_weights=r3_weights,
        questions_mode=args.questions_mode,
        dedup_run_exact=args.dedup_run_exact,
        dedup_run_semhash=args.dedup_run_semhash,
        dedup_semhash_threshold=args.dedup_semhash_threshold,
        dedup_run_llm=args.dedup_run_llm,
        dedup_similarity_threshold=args.dedup_similarity_threshold,
        combined_entity_extraction=args.combined_entity_extraction,
        full_entity_profile=args.full_entity_profile,
        generate_entity_relation_labels=args.generate_entity_relation_labels,
        dedup_sim_weights=dedup_sim_weights,
        same_chunk_min_facts=args.same_chunk_min_facts,
        edge_descriptions=args.edge_descriptions,
        run_fact_dedup=args.run_fact_dedup,
        fact_dedup_top_k=args.fact_dedup_top_k,
        fact_dedup_cluster_size=args.fact_dedup_cluster_size,
        fast_cluster_embed_loader=args.fast_cluster_embed_loader,
        fast_cluster_leiden_levels=args.fast_cluster_leiden_levels,
        fast_cluster_leiden_passes=args.fast_cluster_leiden_passes,
        fast_cluster_min_size=args.fast_cluster_min_size,
        fact_prompt=args.fact_prompt,
        fact_edge_prompt=args.fact_edge_prompt,
        global_fact_edge_prompt=args.global_fact_edge_prompt,
        checkpoint_dir=checkpoint_dir,
    )
    process_execution_time = round(time.time() - t0, 2)

    # Cumulative wall time across every resume attempt for this run-dir (a resumed run's own
    # time.time() delta only covers what THIS process did, not earlier crashed attempts).
    _ckpt = ConstructionCheckpoint(checkpoint_dir)
    execution_time = round(
        _ckpt.extraction_time_total() + _ckpt.dedup_time_total()
        + sum(_ckpt.round_time(n) for n in (1, 2, 3)),
        2,
    ) or process_execution_time
    total_tok = token_breakdown.get("total", {})

    n_nodes    = graph.network.number_of_nodes()
    n_edges    = graph.network.number_of_edges()
    n_entities = len(graph.entities)
    n_facts    = len(graph.facts)

    print(
        f"[Construction] Done in {execution_time}s (this attempt: {process_execution_time}s) "
        f"| Nodes: {n_nodes} | Edges: {n_edges} "
        f"| Entities: {n_entities} | Facts: {n_facts} "
        f"| LLM tokens: {total_tok.get('total', 0):,} ({total_tok.get('calls', 0)} calls)"
    )
    _print_token_breakdown(token_breakdown)

    kg_path = os.path.join(args.run_dir, KG_FILENAME)
    graph.export_graph(kg_path)
    print(f"[Construction] Saved KG → {kg_path}  ({os.path.getsize(kg_path):,} bytes)")

    construction_args = {
        "input_mode":                    args.input_mode,
        "model_name":                    cfg.model,
        "chunk_tokenizer_model":         args.chunk_tokenizer_model,
        "max_workers":                   args.max_workers,
        "chunk_size":                    args.chunk_size,
        "chunk_overlap":                 args.chunk_overlap,
        "edge_threshold":                args.edge_threshold,
        "top_k":                         args.top_k,
        "max_sub_workers":               args.max_sub_workers,
        "fact_batch_size":               args.fact_batch_size,
        "edge_mode":                     args.edge_mode,
        "same_chunk_min_facts":          args.same_chunk_min_facts,
        "edge_descriptions":             args.edge_descriptions,
        "run_fact_dedup":                args.run_fact_dedup,
        "fact_dedup_top_k":              args.fact_dedup_top_k,
        "fact_dedup_cluster_size":       args.fact_dedup_cluster_size,
        "fast_cluster_embed_loader":     args.fast_cluster_embed_loader,
        "fast_cluster_leiden_levels":    args.fast_cluster_leiden_levels,
        "fast_cluster_leiden_passes":    args.fast_cluster_leiden_passes,
        "fast_cluster_min_size":         args.fast_cluster_min_size,
        "fact_prompt":                   args.fact_prompt,
        "fact_edge_prompt":              args.fact_edge_prompt,
        "global_fact_edge_prompt":       args.global_fact_edge_prompt or args.fact_edge_prompt,
        "run_deduplication":             args.deduplication,
        "dedup_model_name":              args.dedup_model_name,
        "dedup_iterations":              args.dedup_iterations,
        "dedup_max_cluster_size":        args.dedup_max_cluster_size,
        "dedup_run_exact":               args.dedup_run_exact,
        "dedup_run_semhash":             args.dedup_run_semhash,
        "dedup_semhash_threshold":       args.dedup_semhash_threshold,
        "dedup_run_llm":                 args.dedup_run_llm,
        "dedup_similarity_threshold":    args.dedup_similarity_threshold,
        "dedup_sim_weights":             dedup_sim_weights,
        "generate_entity_relation_labels": args.generate_entity_relation_labels,
        "use_cot":                       args.use_cot,
        "neutral_description":           args.neutral_description,
        "concise_summary":               args.concise_summary,
        "combined_entity_extraction":    args.combined_entity_extraction,
        "full_entity_profile":           args.full_entity_profile,
        "clustered_round1":              args.clustered_round1,
        "clustered_round2":              args.clustered_round2,
        "clustered_round3":              args.clustered_round3,
        "clustered_max_facts_per_prompt": args.clustered_max_facts_per_prompt,
        "middle_k_skip_percent":         args.middle_k_skip_percent,
        "global_edge_retrieval_model":   args.global_edge_retrieval_model,
        "use_questions_weight":          args.use_questions_weight,
        "questions_mode":                args.questions_mode,
        "construction_max_tokens":       args.construction_max_tokens,
        "temperature":                   args.temperature,
        "num_retries":                   args.num_retries,
    }

    saved_config = _load_construction_config(config_path) if os.path.exists(config_path) else {}

    construction_results = {
        "run_id":    f"nkg_construction_{uuid.uuid4().hex[:8]}",
        "timestamp": datetime.now().isoformat(),
        "system":    "NKG",
        "dataset":   args.dataset,
        "metadata": {
            "corpus_file":         os.path.abspath(args.corpus_file),
            "corpus_paragraphs":   len(corpus),
            "corpus_txt_path":     os.path.abspath(corpus_txt_path) if corpus_txt_path else None,
            "run_dir":             os.path.abspath(args.run_dir),
            "kg_path":             os.path.abspath(kg_path),
            "provider":            cfg.provider,
            "model":               cfg.model,
            "api_base":            cfg.api_base,
            "construction_args":   construction_args,
            "checkpointing": {
                "checkpoint_dir":    os.path.abspath(checkpoint_dir),
                "config_path":       os.path.abspath(config_path),
                "override_history":  saved_config.get("override_history", []),
            },
        },
        "construction": {
            "total_corpus_chars":    total_chars,
            "n_nodes":               n_nodes,
            "n_edges":               n_edges,
            "n_entities":            n_entities,
            "n_facts":               n_facts,
            "execution_time_sec":    execution_time,
            "kg_size_bytes":         os.path.getsize(kg_path),
            "token_breakdown":       token_breakdown,
            "llm_prompt_tokens":     total_tok.get("prompt", 0),
            "llm_completion_tokens": total_tok.get("completion", 0),
            "llm_total_tokens":      total_tok.get("total", 0),
            "llm_api_calls":         total_tok.get("calls", 0),
            "llm_prompt_cache_hit_tokens": total_tok.get("cache_hit", 0),
        },
    }

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(construction_results, f, indent=2, default=str)

    print(f"\n[Construction] DONE  {execution_time}s")
    print(f"[Construction]   Results → {results_path}")


# =============================================================================
# EVALUATION WORKER
# =============================================================================

def _eval_one(q: dict, retriever, retrieval_args: dict, tokenizer, args, cfg: ProviderConfig) -> dict:
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
        raw_ctx = retriever.retrieve(query=question, **retrieval_args)
        raw_ctx = raw_ctx if isinstance(raw_ctx, str) else str(raw_ctx)

        trunc_ctx, ctx_tok = truncate_tokens(raw_ctx, tokenizer, args.max_context_tokens)

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

def run_evaluation(args, cfg: ProviderConfig, retrieval_args: dict):
    results_path = os.path.join(args.run_dir, CONSTRUCTION_RESULTS_FILE)
    kg_path      = os.path.join(args.run_dir, KG_FILENAME)
    if not os.path.exists(results_path) or not os.path.exists(kg_path):
        print(f"[Evaluation FATAL] {CONSTRUCTION_RESULTS_FILE} / {KG_FILENAME} not found in "
              f"{args.run_dir}. Run --mode construction first.")
        sys.exit(1)

    with open(results_path, "r", encoding="utf-8") as f:
        construction_meta = json.load(f)
    print(f"[Evaluation] Found construction results from "
          f"{construction_meta.get('timestamp', '?')}")

    print(f"[Evaluation] Loading questions: {args.questions_file}")
    questions = load_questions_jsonl(args.questions_file)
    print(f"[Evaluation]   {len(questions):,} questions")

    print(f"[Evaluation] Loading retrieval models ({args.retrieval_model})...")
    bi_encoder    = SentenceTransformer(args.retrieval_model, trust_remote_code=True)
    cross_encoder = CrossEncoder(args.cross_encoder_model)

    print(f"[Evaluation] Loading KG from {kg_path} (engine={args.engine})...")
    retriever = build_retriever(args.engine, bi_encoder, cross_encoder, kg_path)

    configure_dspy(
        model=dspy_model_string(cfg), api_base=cfg.api_base, api_key=cfg.api_key,
        max_tokens=args.judge_max_tokens, num_retries=args.num_retries,
        temperature=resolve_temperature(args.temperature),
    )

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_model)
        print(f"[Evaluation] Tokenizer: {args.tokenizer_model}")
    except Exception:
        import tiktoken

        class _TiktokenWrapper:
            def __init__(self, enc): self._enc = enc
            def encode(self, t, **kw): return self._enc.encode(t)
            def decode(self, ids, **kw): return self._enc.decode(ids)

        tokenizer = _TiktokenWrapper(tiktoken.get_encoding("cl100k_base"))
        print("[Evaluation] HuggingFace tokenizer unavailable; "
              "falling back to tiktoken cl100k_base.")

    tracker = GlobalTokenAccumulator()
    attach_token_tracker(tracker)

    print(f"[Evaluation] Starting evaluation: {len(questions):,} questions, "
          f"{args.max_workers} workers")
    t0 = time.time()
    results = []

    try:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            fmap = {
                executor.submit(_eval_one, q, retriever, retrieval_args, tokenizer, args, cfg): i
                for i, q in enumerate(questions)
            }
            done = 0
            for future in as_completed(fmap):
                r = future.result()
                results.append(r)
                done += 1
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

    const_tokens = {
        "prompt_tokens":     construction_meta["construction"]["llm_prompt_tokens"],
        "completion_tokens": construction_meta["construction"]["llm_completion_tokens"],
        "total_tokens":      construction_meta["construction"]["llm_total_tokens"],
        "api_calls":         construction_meta["construction"]["llm_api_calls"],
    }
    all_tokens = add_token_counters(const_tokens, eval_tk)

    output = {
        "run_id":    f"nkg_eval_{uuid.uuid4().hex[:8]}",
        "timestamp": datetime.now().isoformat(),
        "system":    "NKG",
        "dataset":   args.dataset,
        "metadata": {
            "questions_file":     os.path.abspath(args.questions_file),
            "run_dir":            os.path.abspath(args.run_dir),
            "provider":           cfg.provider,
            "model":              cfg.model,
            "api_base":           cfg.api_base,
            "engine":             args.engine,
            "retrieval_args":     retrieval_args,
            "retrieval_model":    args.retrieval_model,
            "cross_encoder_model": args.cross_encoder_model,
            "max_context_tokens": args.max_context_tokens,
            "answer_max_tokens":  args.answer_max_tokens,
            "max_workers":        args.max_workers,
        },
        "construction": construction_meta["construction"],
        "evaluation": {
            "total_questions":     len(questions),
            "successful":          n,
            "errored":             len(bad),
            "evaluation_time_sec": eval_t,
            "evaluation_tokens":   eval_tk,
            "overall":             overall,
            "by_type":             by_type,
            **({"by_level": by_level} if by_level else {}),
            "results":             results,
        },
        "totals": {
            "all_tokens":   all_tokens,
            "all_time_sec": round(
                float(construction_meta["construction"]["execution_time_sec"]) + eval_t, 2
            ),
        },
    }

    eval_path = next_eval_results_path(args.run_dir)
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)

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
        description="NKG evaluation on HotpotQA / MuSiQue",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ---- Mode & dataset ----
    p.add_argument("--mode",    required=True, choices=["construction", "evaluation"])
    # [contracts]/[quality] construction-only labels for evaluation_scripts/unh (CONTRACTS_EVAL_PLAN.md, QUALITY_EVAL_PLAN.md)
    p.add_argument("--dataset", required=True, choices=list(SUPPORTED_DATASETS) + ["contracts", "quality"])

    # ---- Core paths ----
    p.add_argument("--run-dir",        type=str, required=True,
                   help="Working directory. Construction writes kg.graphml + "
                        "construction_results.json + construction_config.json + checkpoint/ "
                        "here; evaluation reads from it.")
    p.add_argument("--corpus-file",    type=str, default=None,
                   help="(construction) Path to corpus.jsonl from the prep script. Optional "
                        "when resuming a construction run that already has a "
                        "construction_config.json in --run-dir.")
    p.add_argument("--questions-file", type=str, default=None,
                   help="(evaluation) Path to questions.jsonl from the prep script.")

    # ---- Provider ----
    # default=None (not "local_vllm"/"") so an unset flag can fall back to a resumed run's
    # saved config instead of silently overriding it — see _resolve_and_persist_construction_args.
    p.add_argument("--provider", default=None, choices=list(PROVIDER_DEFAULTS))
    p.add_argument("--model",    type=str, default=None)
    p.add_argument("--api-base", type=str, default=None)
    p.add_argument("--api-key",  type=str, default="")

    # ---- Construction args (mirrors insurance_evaluation.py) ----
    # All default=None below (except --max-workers/--max-sub-workers/--force-rebuild): an unset
    # flag resolves to the saved construction_config.json value on resume, or the hardcoded
    # default in _CONSTRUCTION_DEFAULTS on a fresh run. See _resolve_and_persist_construction_args.
    p.add_argument("--input-mode", type=str, default=None, choices=["concat_file", "prepared_chunks", "per_document"],
                   help="concat_file (default): concatenate the corpus into one .txt and let NKG "
                        "chunk it by token count. prepared_chunks: one paragraph = one final "
                        "chunk, no further token-based splitting. per_document: each paragraph (document) "
                        "is chunked on its own by NKG's chunker (--chunk-size/--chunk-overlap), so no chunk "
                        "crosses a document boundary (QuALITY: one document = one article).")
    p.add_argument("--max-workers",        type=int,   default=4,
                   help="Concurrency knob — always taken from the CLI (or this default), never "
                        "diffed against a saved config, safe to change freely on resume.")
    p.add_argument("--chunk-size",         type=int,   default=None)
    p.add_argument("--chunk-overlap",      type=int,   default=None)
    p.add_argument("--chunk-tokenizer-model", type=str, default=None,
                   help="HuggingFace tokenizer used for token-based chunking (concat_file mode). "
                        "Independent of --model/--provider so e.g. --provider deepseek doesn't "
                        "try to load a HF tokenizer named after a non-HF model id.")
    p.add_argument("--edge-threshold",     type=float, default=None)
    p.add_argument("--top-k",              type=int,   default=None)
    p.add_argument("--max-sub-workers",    type=int,   default=10,
                   help="Concurrency knob — same exemption as --max-workers.")
    p.add_argument("--fact-batch-size",    type=int,   default=None)
    p.add_argument("--edge-mode",          type=str,   default=None,
                   choices=["linear", "clustered", "clustered_fast"],
                   help="clustered_fast = nkg/index/construction/fast_fact_clustering.py (batched GPU similarity "
                        "graph, cached embeddings, --fast-cluster-* options); default clustered.")
    p.add_argument("--same-chunk-min-facts", type=int, default=None,
                   help="A chunk needs at least this many facts before the intra-chunk "
                        "fact-edge LLM call is made at all; chunks below the threshold are "
                        "skipped entirely (default: 2).")
    p.add_argument("--deduplication", action=argparse.BooleanOptionalAction, default=None,
                   help="Run entity deduplication (default: on). Pass --no-deduplication to disable.")
    p.add_argument("--dedup-model-name",   type=str, default=None)
    p.add_argument("--dedup-iterations",   type=int, default=None)
    p.add_argument("--dedup-max-cluster-size", type=int, default=None)
    p.add_argument("--dedup-run-exact", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--dedup-run-semhash", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--dedup-semhash-threshold", type=float, default=None)
    p.add_argument("--dedup-run-llm", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--dedup-similarity-threshold", type=float, default=None)
    p.add_argument("--dedup-sim-weight-name", type=float, default=None,
                   help="Entity-dedup similarity blend weight for name (default: 0.6).")
    p.add_argument("--dedup-sim-weight-role", type=float, default=None,
                   help="Entity-dedup similarity blend weight for role (default: 0.3).")
    p.add_argument("--dedup-sim-weight-type", type=float, default=None,
                   help="Entity-dedup similarity blend weight for entity type (default: 0.1).")
    p.add_argument("--generate-entity-relation-labels", action=argparse.BooleanOptionalAction, default=None,
                   help="Have the LLM generate a strict 2-4 word UPPERCASE relation label per "
                        "entity per fact, instead of reusing the fact's own sentence as the "
                        "entity->fact edge label (default: off — costs extra LLM output).")
    p.add_argument("--use-cot", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--edge-descriptions", action=argparse.BooleanOptionalAction, default=None,
                   help="Fact->fact edges get a description sentence plus short labels (default: on). "
                        "--no-edge-descriptions = label-only edge prompts; each edge's `description` holds its label.")
    p.add_argument("--fact-dedup", dest="run_fact_dedup", action=argparse.BooleanOptionalAction, default=None,
                   help="Fact deduplication after entity dedup, before cross-chunk edges (default: on).")
    p.add_argument("--fact-dedup-top-k", type=int, default=None,
                   help="Fact dedup: nearest neighbours per fact in its similarity graph (default 12).")
    p.add_argument("--fact-dedup-cluster-size", type=int, default=None,
                   help="Fact dedup: max facts per LLM dedup call (default 8).")
    p.add_argument("--fast-cluster-embed-loader", type=str, default=None, choices=["remote_code", "legacy"],
                   help="--edge-mode clustered_fast: 'remote_code' (default) loads the embedding model with "
                        "trust_remote_code=True (the real voyage-4-nano); 'legacy' loads it as the clustered mode does.")
    p.add_argument("--fast-cluster-leiden-levels", type=str, default=None, choices=["final", "all"],
                   help="--edge-mode clustered_fast: 'final' (default) = Leiden's final clusters only (disjoint, "
                        "<= --clustered-max-facts-per-prompt); 'all' = every hierarchy level, as the clustered mode.")
    p.add_argument("--fast-cluster-min-size", type=int, default=None,
                   help="--edge-mode clustered_fast: clusters below this many facts are merged into their most strongly "
                        "connected cluster while the result stays <= --clustered-max-facts-per-prompt (default 0 = off).")
    p.add_argument("--fast-cluster-leiden-passes", type=int, default=None,
                   help="--edge-mode clustered_fast: Leiden runs with seeds 42, 43, ...; final clusters pooled, so "
                        "each fact is in up to this many clusters (default 1).")
    p.add_argument("--fact-prompt", type=str, default=None, choices=["default", "detailed"],
                   help="fact-sentence extraction prompt: default (ChunkFacts) or detailed (ChunkFactsDetailed: keeps "
                        "the text's wording and every time / cause / attitude / attribution / condition cue inside "
                        "each fact; for detail-heavy text such as narratives)")
    p.add_argument("--fact-edge-prompt", type=str, default=None, choices=["default", "general", "event"],
                   help="fact->fact edge prompt (label-only): default (Bulk*FactEdgesLabelOnly) or general "
                        "(Bulk*FactEdgesGeneral: one relation vocabulary for narrative and expository text -- time, "
                        "motive, dialogue, change of state, evidence, example -- and a link only where one fact "
                        "explains, answers or follows from the other; needs --no-edge-descriptions)")
    p.add_argument("--global-fact-edge-prompt", type=str, default=None, choices=["default", "general", "event"],
                   help="global cross-chunk fact->fact edge prompt (default: --fact-edge-prompt). event = "
                        "BulkGlobalFactEdgesEvent (RED / Hong et al. relation families + WILDCARD, specific labels, "
                        "few-shot; needs --no-edge-descriptions and --edge-mode clustered / clustered_fast)")
    p.add_argument("--neutral-description", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--concise-summary", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--combined-entity-extraction", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--full-entity-profile", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--clustered-round1", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--clustered-round2", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--clustered-round3", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--clustered-max-facts-per-prompt", type=int, default=None)
    p.add_argument("--middle-k-skip-percent", type=float, default=None)
    p.add_argument("--global-edge-retrieval-model", type=str, default=None)
    p.add_argument("--use-questions-weight", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--questions-mode", type=str, default=None,
                   choices=["follow_up_to_answered", "follow_up_to_fact_sentence",
                            "fact_sentence_to_follow_up"])
    p.add_argument("--temperature", type=str, default="0.1",
                   help="Temperature for every DSPy LLM call (construction and query-time). Default 0.1 "
                        "(nkg.utils.config.configure_dspy's default). 'none' = not sent (provider default; "
                        "DeepSeek: 1.0), as in evaluation_scripts/.")
    p.add_argument("--construction-max-tokens", type=int, default=100000,
                   help="max_tokens for construction LLM calls (default 100000). Excluded from "
                        "the resume config diff/warning, like --max-workers and --num-retries.")
    p.add_argument("--num-retries", type=int, default=1,
                   help="dspy.LM retries on a failed/truncated generation before giving up on "
                        "that call (default 1). Applies to both construction and evaluation LM "
                        "calls. Lower this (e.g. 0) if degenerate/truncated generations on hard "
                        "inputs are wasting time retrying rather than being skipped quickly. "
                        "Excluded from the resume config diff/warning, like --max-workers.")
    p.add_argument("--force-rebuild", action="store_true",
                   help="Rebuild even if construction_results.json already exists — wipes any "
                        "existing checkpoint/construction_config.json first.")
    p.add_argument("-y", "--yes", action="store_true",
                   help="Auto-confirm the resume-with-different-args prompt (for non-interactive "
                        "runs). Without this, a detected arg conflict on a non-TTY stdin aborts "
                        "instead of hanging.")

    # ---- Retrieval / engine ----
    p.add_argument("--engine", type=str, default=DEFAULT_ENGINE, choices=ENGINE_CHOICES,
                   help="'new' (GoalSearchEngine, default), 'old' (MMR-based Retriever), or "
                        "'dag' (DAGRetriever: decomposes the query into a DAG of subqueries and "
                        "searches each with the old engine).")
    p.add_argument("--retrieval-model",     type=str, default="voyageai/voyage-4-nano")
    p.add_argument("--cross-encoder-model", type=str, default="cross-encoder/ms-marco-MiniLM-L6-v2")
    p.add_argument("--default-retrieval-args", action="store_true",
                   help="Start from hardcoded defaults for --engine; explicit CLI "
                        "overrides below still apply on top.")
    p.add_argument("--deduplicate-paths", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--neutral-layout", action=argparse.BooleanOptionalAction, default=None,
                   help="Context layout for fact->fact edges. Default (unset) = True, as before: short label inline + "
                        "long descriptions in a 'Logical Conclusions' block. --no-neutral-layout = the edge's "
                        "`description` inline (used by short_description_ablation.py, where description == label).")
    p.add_argument("--top-k-seeds",      type=int,   default=None)
    p.add_argument("--max-depth",        type=int,   default=None)
    p.add_argument("--beam-width",       type=int,   default=None)
    p.add_argument("--final-top-k",      type=int,   default=None)
    p.add_argument("--retrieval-node-mode", type=str, default=None)
    p.add_argument("--beam-mmr-lambda",  type=float, default=None, help="[old/dag engine only]")
    p.add_argument("--final-mmr-lambda", type=float, default=None, help="[old/dag engine only]")
    p.add_argument("--search-mode",      type=str,   default=None,
                   choices=["beam_search", "blind_expansion"], help="[old/dag engine only]")
    p.add_argument("--max-fork-branches", type=int, default=None,
                   help="[dag engine only] Cap on how many dependency-combination / "
                        "disambiguation forks a single DAG node may expand into")
    p.add_argument("--parallel-fallback-chain-threshold", type=int, default=None,
                   help="[dag engine only] When the executed DAG's longest dependency chain is "
                        "at or under this many nodes, also search the original un-decomposed "
                        "query directly as an extra safety-net branch merged into the final "
                        "context. Set to 0 to disable.")
    p.add_argument("--goal-lookahead", type=int, default=None, help="[new engine only]")
    p.add_argument("--use-island-and-bridge", action=argparse.BooleanOptionalAction,
                   default=None, help="[new engine only]")
    p.add_argument("--island-pool-size", type=int, default=None, help="[new engine only]")
    p.add_argument("--island-max-bridge-hops", type=int, default=None, help="[new engine only]")

    # ---- Evaluation ----
    p.add_argument("--tokenizer-model", type=str, default="Qwen/Qwen3-30B-A3B-Instruct-2507-FP8")
    p.add_argument("--judge-max-tokens", type=int, default=8000)
    p.add_argument("--max-context-tokens", type=int, default=2000)
    p.add_argument("--answer-max-tokens",  type=int, default=128)

    args = p.parse_args()

    if args.mode == "evaluation" and not args.questions_file:
        p.error("--mode evaluation requires --questions-file")

    if args.mode == "construction":
        _resolve_and_persist_construction_args(args)
        already_done = os.path.exists(os.path.join(args.run_dir, CONSTRUCTION_RESULTS_FILE)) \
                        and not args.force_rebuild
        if not args.corpus_file and not already_done:
            p.error("--mode construction requires --corpus-file (or an existing "
                     "construction_config.json in --run-dir to resume from)")

    load_dotenv()
    cfg = resolve_provider(args)

    # ---- Build retrieval args dict (engine-specific) ----
    CLI_RETRIEVAL_FALLBACKS = {
        "top_k_seeds": 9, "max_depth": 3, "beam_width": 6, "final_top_k": 20,
        "retrieval_node_mode": "all", "beam_mmr_lambda": 1.0, "final_mmr_lambda": 0.6,
        "search_mode": "beam_search", "goal_lookahead": 1,
        "use_island_and_bridge": False, "island_pool_size": 75, "island_max_bridge_hops": 1,
        "max_fork_branches": 6, "parallel_fallback_chain_threshold": 2,
    }
    # DAG's per-node search is intentionally shallow -- max_depth=1 rather than the old/new
    # engines' 3 -- so it gets its own fallback for that one key.
    DAG_MAX_DEPTH_FALLBACK = 1

    def cli_val(name):
        v = getattr(args, name)
        if v is not None:
            return v
        if name == "max_depth" and args.engine == "dag":
            return DAG_MAX_DEPTH_FALLBACK
        return CLI_RETRIEVAL_FALLBACKS[name]

    ENGINE_OVERRIDABLE = {
        "new": ["goal_lookahead", "use_island_and_bridge", "island_pool_size", "island_max_bridge_hops"],
        "old": ["beam_mmr_lambda", "final_mmr_lambda", "search_mode"],
        "dag": ["beam_mmr_lambda", "final_mmr_lambda", "search_mode", "max_fork_branches",
                "parallel_fallback_chain_threshold"],
    }

    if args.default_retrieval_args:
        retrieval_args = default_retrieval_args(args.engine)
        retrieval_args["deduplicate_paths"] = args.deduplicate_paths
        overridable = ["top_k_seeds", "max_depth", "beam_width", "final_top_k", "retrieval_node_mode"]
        overridable += ENGINE_OVERRIDABLE[args.engine]
        for name in overridable:
            if getattr(args, name) is not None:
                key = "mode" if name == "retrieval_node_mode" else name
                retrieval_args[key] = getattr(args, name)
    elif args.engine == "new":
        retrieval_args = {
            "top_k_seeds":            cli_val("top_k_seeds"),
            "max_depth":              cli_val("max_depth"),
            "beam_width":             cli_val("beam_width"),
            "final_top_k":            cli_val("final_top_k"),
            "goal_lookahead":         cli_val("goal_lookahead"),
            "mode":                   cli_val("retrieval_node_mode"),
            "use_island_and_bridge":  cli_val("use_island_and_bridge"),
            "island_pool_size":       cli_val("island_pool_size"),
            "island_max_bridge_hops": cli_val("island_max_bridge_hops"),
            "deduplicate_paths":      args.deduplicate_paths,
            "neutral_layout":         True,
        }
    elif args.engine == "dag":
        retrieval_args = {
            "top_k_seeds":       cli_val("top_k_seeds"),
            "max_depth":         cli_val("max_depth"),
            "beam_width":        cli_val("beam_width"),
            "final_top_k":       cli_val("final_top_k"),
            "mode":              cli_val("retrieval_node_mode"),
            "beam_mmr_lambda":   cli_val("beam_mmr_lambda"),
            "final_mmr_lambda":  cli_val("final_mmr_lambda"),
            "search_mode":       cli_val("search_mode"),
            "max_fork_branches": cli_val("max_fork_branches"),
            "parallel_fallback_chain_threshold": cli_val("parallel_fallback_chain_threshold"),
            "deduplicate_paths": args.deduplicate_paths,
            "neutral_layout":    True,
        }
    else:
        retrieval_args = {
            "top_k_seeds":      cli_val("top_k_seeds"),
            "max_depth":        cli_val("max_depth"),
            "beam_width":       cli_val("beam_width"),
            "final_top_k":      cli_val("final_top_k"),
            "mode":             cli_val("retrieval_node_mode"),
            "beam_mmr_lambda":  cli_val("beam_mmr_lambda"),
            "final_mmr_lambda": cli_val("final_mmr_lambda"),
            "search_mode":      cli_val("search_mode"),
            "deduplicate_paths": args.deduplicate_paths,
            "neutral_layout":    True,
        }

    if args.neutral_layout is not None:
        retrieval_args["neutral_layout"] = args.neutral_layout

    # ---- Banner ----
    print(f"\n{'=' * 65}")
    print(f"  NKG {args.mode.upper()} — {args.dataset.upper()}")
    print(f"  provider  : {cfg.provider}")
    print(f"  model     : {cfg.model}")
    print(f"  api-base  : {cfg.api_base}")
    print(f"  run-dir   : {args.run_dir}")
    if args.mode == "construction":
        print(f"  input-mode: {args.input_mode}")
        print(f"  chunk     : {args.chunk_size} tok / {args.chunk_overlap} overlap")
        print(f"  edge-mode : {args.edge_mode}")
        if args.force_rebuild:
            print(f"  [!] --force-rebuild: existing results will be overwritten")
    else:
        print(f"  engine    : {args.engine}")
        print(f"  retrieval : {retrieval_args}")
    print(f"{'=' * 65}\n")

    if args.mode == "construction":
        try:
            run_construction(args, cfg)
        except KeyboardInterrupt:
            print("\n[Construction] Interrupted. Checkpoint preserved — rerun the same command to resume.")
        except Exception as e:
            print(f"\n[Construction FATAL] {e}")
            traceback.print_exc()
            sys.exit(1)
    else:
        try:
            run_evaluation(args, cfg, retrieval_args)
        except KeyboardInterrupt:
            print("\n[Evaluation] Interrupted.")
        except Exception as e:
            print(f"\n[Evaluation FATAL] {e}")
            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    main()
