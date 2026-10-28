#!/usr/bin/env python3
"""
NKG Insurance Contract Evaluation

Three modes:
  construction  — build and save the knowledge graph for each contract
  evaluation    — run QA/Fact evaluation on already-built knowledge graphs
  auto          — construction + evaluation in one go (like other eval scripts)

Construction mode saves a kg.graphml and construction.json per contract under:
    out_dir/contract_name/

Evaluation mode scans a target directory for construction directories, skips any
already evaluated under the same (eval_type, engine, retrieval settings), and
saves eval JSON output.

Retrieval engine: --engine {new,old,dag}, default "new".
  new — nkg.search.GoalSearchEngine   (goal-coverage redesign; see nkg/search/)
  old — nkg.retrieval.Retriever       (original MMR-based engine)
  dag — nkg.dag_retrieval.DAGRetriever (decomposes the query into a DAG of subqueries,
        refining/disambiguating and forking as it goes, and searches each subquery
        with the old engine; see nkg/dag_retrieval/)
Output files are tagged with the engine name (eval_{type}_{engine}_{hash}.json)
so both engines can be evaluated against the same construction dirs without
overwriting each other's results.

---------------------------------------------------------------------
Construction examples:
    python insurance_evaluation.py --mode construction \\
        --ten-contracts-dataset --eval-data-dir ./evaluation_data \\
        --out-dir ./latest_results/insurance/nkg/ten_contracts_dataset

    python insurance_evaluation.py --mode construction \\
        --contract evaluation_data/ten_contracts_dataset/contracts/contract_1.txt \\
        --out-dir ./latest_results/insurance/nkg

Evaluation examples:
    # Evaluate all KGs under a parent directory with the new engine (default)
    python insurance_evaluation.py --mode evaluation \\
        --target-dir ./latest_results/insurance/nkg/ten_contracts_dataset \\
        --eval-data-dir ./evaluation_data \\
        --eval-type both --default-retrieval-args

    # Same, but with the old engine, for an A/B comparison
    python insurance_evaluation.py --mode evaluation \\
        --target-dir ./latest_results/insurance/nkg/ten_contracts_dataset \\
        --eval-data-dir ./evaluation_data \\
        --eval-type both --default-retrieval-args --engine old

    # Evaluate a single KG directory
    python insurance_evaluation.py --mode evaluation \\
        --target-dir ./latest_results/insurance/nkg/ten_contracts_dataset/contract_1 \\
        --eval-data-dir ./evaluation_data --eval-type qa

Auto mode example:
    python insurance_evaluation.py --mode auto \\
        --ten-contracts-dataset --eval-data-dir ./evaluation_data \\
        --out-dir ./latest_results/insurance/nkg/ten_contracts_dataset
---------------------------------------------------------------------
"""

import os
import sys
import json
import time
import uuid
import hashlib
import argparse
import threading
import traceback
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer

import dspy

from nkg.utils.config import configure_dspy
from nkg.index.construction.build_index import build_index
from nkg.retrieval.engine import Retriever
from nkg.search.engine import GoalSearchEngine
from nkg.dag_retrieval import DAGRetriever
from experiments.evaluate_mine import ask_llm_query, score_llm_answer, evaluate_fact
from experiments.utils.chunking import truncate_context


# =============================================================================
# CONSTANTS
# =============================================================================
DATASET_SUBDIRS  = ["ten_contracts_dataset", "long_form_contracts"]
KG_FILENAME      = "kg.graphml"
CONSTRUCTION_JSON = "construction.json"

ENGINE_CHOICES = ["new", "old", "dag"]
DEFAULT_ENGINE = "new"

# Hardcoded retrieval defaults for the OLD engine (nkg.retrieval.engine.Retriever).
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

# Hardcoded retrieval defaults for the NEW engine (nkg.search.engine.GoalSearchEngine).
# Sizing kept in the same ballpark as the old defaults above for a fair A/B comparison;
# island_and_bridge (D5) is left off by default since it's a strictly opt-in bolt-on.
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
# OLD engine). max_depth=1 (vs. the old/new engines' 3) is intentional, not a smaller version of
# the same knob: each DAG node is already one focused sub-lookup, so a shallow, narrow search per
# node is the point -- the multi-hop reasoning lives in the DAG structure itself.
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
}


def default_retrieval_args(engine: str) -> dict:
    if engine == "new":
        return dict(DEFAULT_RETRIEVAL_ARGS_NEW)
    if engine == "dag":
        return dict(DEFAULT_RETRIEVAL_ARGS_DAG)
    return dict(DEFAULT_RETRIEVAL_ARGS_OLD)


def build_retriever(engine: str, bi_encoder, cross_encoder, kg_path: str):
    """Constructs the selected retrieval engine, pointed at an already-built KG on disk."""
    if engine == "new":
        return GoalSearchEngine(
            retrieval_model=bi_encoder,
            cross_encoder=cross_encoder,
            graph_path=kg_path,
        )
    if engine == "dag":
        # DAGRetriever is a planning layer that wraps a plain old-engine Retriever -- it decides
        # *what*/*in what order* to search, and delegates every actual graph search to it.
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
        entity_weights={"name": 0.6, "role": 0.4, "anchors": 0.0},  # default: 0.4, 0.3, 0.3
        fact_weights={"sentence": 0.8, "questions": 0.0, "chunk": 0.2},  # default: 0.5, 0.3, 0.2
    )


# =============================================================================
# TEE STREAM — duplicates stdout to a log file
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
            return {"total_errors": sum(by_section.values()), "by_section": by_section}


# =============================================================================
# DATASET PATH HELPERS
# =============================================================================
def resolve_paths_for_contract(contract_path: str, dataset_root: str):
    base        = os.path.splitext(os.path.basename(contract_path))[0]
    section_dir = None
    for subdir in DATASET_SUBDIRS:
        candidate = os.path.join(dataset_root, subdir)
        if os.path.abspath(contract_path).startswith(os.path.abspath(candidate)):
            section_dir = candidate
            break

    if section_dir is None:
        raise ValueError(
            f"Contract {contract_path} is not under any known dataset sub-directory in {dataset_root}."
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
# RETRIEVAL ARGS HELPERS
# =============================================================================
def retrieval_args_hash(retrieval_args: dict) -> str:
    serialized = json.dumps(retrieval_args, sort_keys=True)
    return hashlib.md5(serialized.encode()).hexdigest()[:10]


def eval_output_filename(eval_type: str, engine: str, retrieval_args: dict) -> str:
    h = retrieval_args_hash(retrieval_args)
    return f"eval_{eval_type}_{engine}_{h}.json"


def is_construction_dir(path: str) -> bool:
    return (
        os.path.isdir(path)
        and os.path.exists(os.path.join(path, CONSTRUCTION_JSON))
        and os.path.exists(os.path.join(path, KG_FILENAME))
    )


# =============================================================================
# GRAPH CONSTRUCTION
# =============================================================================
def _print_token_breakdown(breakdown: dict):
    """Prints a formatted per-stage token breakdown table."""
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


def build_graph(
    contract_path: str,
    out_dir: str,
    construction_args: dict,
    error_tracker: ErrorTracker,
    construction_max_tokens: int = 55000,
) -> tuple:
    """
    Builds an NKG for a contract file, saves kg.graphml + construction.json.
    Returns (graph, construction_record).
    """
    configure_dspy(max_tokens=construction_max_tokens, cache=False)

    os.makedirs(out_dir, exist_ok=True)

    print(f"[Construction] Reading: {contract_path}")
    with open(contract_path, "r", encoding="utf-8") as f:
        full_text = f.read()
    total_document_chars = len(full_text)

    print(f"[Construction] Document chars: {total_document_chars}")
    print(
        f"[Construction] Building NKG "
        f"(chunk_size={construction_args['chunk_size']}, "
        f"overlap={construction_args['overlap']}, "
        f"edge_mode={construction_args['edge_mode']}, "
        f"dedup={construction_args['run_deduplication']}, "
        f"max_tokens={construction_max_tokens})..."
    )

    use_questions_weight = construction_args.get("use_questions_weight", True)
    questions_w = 1.0 if use_questions_weight else 0.0
    construction_r2_weights = {"sent": 1.0, "topic": 1.0, "entity": 1.0, "questions": questions_w}
    construction_r3_weights = {"sent": 1.0, "topic": 1.0, "entity": 1.0, "questions": questions_w}

    t0 = time.time()
    try:
        graph, token_breakdown = build_index(
            filepath=contract_path,
            model_name=construction_args["model_name"],
            max_workers=construction_args["max_workers"],
            chunk_size=construction_args["chunk_size"],
            overlap=construction_args["overlap"],
            edge_threshold=construction_args["edge_threshold"],
            top_k=construction_args["top_k"],
            max_sub_workers=construction_args["max_sub_workers"],
            fact_batch_size=construction_args["fact_batch_size"],
            edge_mode=construction_args["edge_mode"],
            run_deduplication=construction_args["run_deduplication"],
            dedup_model_name=construction_args["dedup_model_name"],
            dedup_iterations=construction_args["dedup_iterations"],
            dedup_max_cluster_size=construction_args["dedup_max_cluster_size"],
            use_cot=construction_args.get("use_cot", False),
            neutral_description=construction_args.get("neutral_description", True),
            concise_summary=construction_args.get("concise_summary", True),
            clustered_round1=construction_args.get("clustered_round1", False),
            clustered_round2=construction_args.get("clustered_round2", True),
            clustered_round3=construction_args.get("clustered_round3", True),
            clustered_max_facts_per_prompt=construction_args.get("clustered_max_facts_per_prompt", 40),
            middle_k_skip_percent=construction_args.get("middle_k_skip_percent", 0.33),
            retrieval_model_name=construction_args.get("retrieval_model_name", "voyageai/voyage-4-nano"),
            construction_r2_weights=construction_r2_weights,
            construction_r3_weights=construction_r3_weights,
            questions_mode=construction_args.get("questions_mode", "follow_up_to_answered"),
            dedup_run_exact=construction_args.get("dedup_run_exact", True),
            dedup_run_semhash=construction_args.get("dedup_run_semhash", True),
            dedup_semhash_threshold=construction_args.get("dedup_semhash_threshold", 0.95),
            dedup_run_llm=construction_args.get("dedup_run_llm", True),
            dedup_similarity_threshold=construction_args.get("dedup_similarity_threshold", 0.75),
            combined_entity_extraction=construction_args.get("combined_entity_extraction", True),
            full_entity_profile=construction_args.get("full_entity_profile", False),
        )
    except Exception:
        error_tracker.record("graph_construction")
        raise

    execution_time = round(time.time() - t0, 2)
    total_tok      = token_breakdown.get("total", {})

    n_nodes    = graph.network.number_of_nodes()
    n_edges    = graph.network.number_of_edges()
    n_entities = len(graph.entities)
    n_facts    = len(graph.facts)

    print(
        f"[Construction] Done in {execution_time}s "
        f"| Nodes: {n_nodes} | Edges: {n_edges} "
        f"| Entities: {n_entities} | Facts: {n_facts} "
        f"| LLM tokens: {total_tok.get('total', 0):,} ({total_tok.get('calls', 0)} calls)"
    )
    _print_token_breakdown(token_breakdown)

    kg_path = os.path.join(out_dir, KG_FILENAME)
    graph.export_graph(kg_path)
    print(f"[Construction] Saved KG → {kg_path}  ({os.path.getsize(kg_path):,} bytes)")

    metrics = {
        "total_document_chars":  total_document_chars,
        "n_nodes":               n_nodes,
        "n_edges":               n_edges,
        "n_entities":            n_entities,
        "n_facts":               n_facts,
        "execution_time_sec":    execution_time,
        "kg_size_bytes":         os.path.getsize(kg_path),
        # Per-stage breakdown (detailed)
        "token_breakdown":       token_breakdown,
        # Top-level totals kept for backward compatibility with analysis notebooks
        "llm_prompt_tokens":     total_tok.get("prompt", 0),
        "llm_completion_tokens": total_tok.get("completion", 0),
        "llm_total_tokens":      total_tok.get("total", 0),
        "llm_api_calls":         total_tok.get("calls", 0),
    }

    construction_record = {
        "system":            "NKG",
        "timestamp":         datetime.now().strftime("%Y%m%d_%H%M%S"),
        "contract_file":     os.path.abspath(contract_path),
        "construction_args": construction_args,
        "metrics":           metrics,
    }

    construction_json_path = os.path.join(out_dir, CONSTRUCTION_JSON)
    with open(construction_json_path, "w", encoding="utf-8") as f:
        json.dump(construction_record, f, indent=2, default=str)
    print(f"[Construction] Saved construction record → {construction_json_path}")

    return graph, construction_record


# =============================================================================
# QA EVALUATION WORKER
# =============================================================================
def _eval_qa_item(
    pair, retriever, retrieval_args, tokenizer, max_context_tokens, error_tracker,
):
    q        = pair["question"]
    a        = pair["answer"]
    category = pair.get("category", "UNKNOWN")
    evidence = pair.get("evidence", "")

    llm_answer = ""
    score = 0.0
    raw_context = truncated_context = ""
    context_tokens = 0
    status = "SUCCESS"

    current_section = "qa_evaluation.context_retrieval"
    try:
        raw_context = retriever.retrieve(query=q, **retrieval_args)
        raw_context = raw_context if isinstance(raw_context, str) else str(raw_context)
        truncated_context, context_tokens = truncate_context(raw_context, tokenizer, max_context_tokens)

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
    fact_dict, retriever, retrieval_args, tokenizer, max_context_tokens, error_tracker,
):
    fact    = fact_dict["fact"]
    fact_id = fact_dict.get("id")

    score = 0
    raw_context = truncated_context = ""
    context_tokens = 0
    status = "SUCCESS"

    current_section = "fact_evaluation.context_retrieval"
    try:
        raw_context = retriever.retrieve(query=fact, **retrieval_args)
        raw_context = raw_context if isinstance(raw_context, str) else str(raw_context)
        truncated_context, context_tokens = truncate_context(raw_context, tokenizer, max_context_tokens)

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
    qa_file, retriever, retrieval_args, tokenizer, max_context_tokens, max_workers, error_tracker,
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
                _eval_qa_item, p, retriever, retrieval_args,
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
    fact_file, retriever, retrieval_args, tokenizer, max_context_tokens, max_workers, error_tracker,
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
                _eval_fact_item, fd, retriever, retrieval_args,
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
# EVALUATE A SINGLE CONSTRUCTION DIRECTORY
# =============================================================================
def evaluate_construction_dir(
    construction_dir: str,
    eval_data_dir: str,
    eval_type: str,
    engine: str,
    retrieval_args: dict,
    retrieval_model_name: str,
    cross_encoder_model_name: str,
    tokenizer,
    max_context_tokens: int,
    max_workers_eval: int,
    force_evaluation: bool,
):
    """
    Loads the KG in construction_dir, resolves qa/fact files via construction.json,
    runs evaluation, and saves the eval JSON output.
    Skips if the same (eval_type, engine, retrieval_args) was already evaluated unless force_evaluation.
    """
    construction_json_path = os.path.join(construction_dir, CONSTRUCTION_JSON)
    kg_path                = os.path.join(construction_dir, KG_FILENAME)

    if not os.path.exists(construction_json_path) or not os.path.exists(kg_path):
        print(f"[SKIP] {construction_dir}: missing construction.json or kg.graphml")
        return

    out_filename = eval_output_filename(eval_type, engine, retrieval_args)
    out_path     = os.path.join(construction_dir, out_filename)
    if os.path.exists(out_path) and not force_evaluation:
        print(f"[SKIP] Already evaluated ({out_filename}): {construction_dir}")
        return

    with open(construction_json_path, "r", encoding="utf-8") as f:
        construction_record = json.load(f)

    contract_path = construction_record["contract_file"]
    print(f"\n[Eval] Contract  : {os.path.basename(contract_path)}")
    print(f"[Eval] Directory : {construction_dir}")

    # Resolve QA / Fact files from the eval data directory
    qa_file = fact_file = None
    try:
        resolved_qa, resolved_fact = resolve_paths_for_contract(contract_path, eval_data_dir)
        if eval_type in ("qa", "both"):
            qa_file = resolved_qa
        if eval_type in ("fact", "both"):
            fact_file = resolved_fact
    except (ValueError, FileNotFoundError) as e:
        print(f"[WARN] Could not resolve eval files for {os.path.basename(contract_path)}: {e}")

    if not qa_file and not fact_file:
        print(f"[SKIP] No evaluation files resolved for {os.path.basename(contract_path)}")
        return

    # Load models and retriever
    print(f"[Eval] Loading retrieval models ({retrieval_model_name})...")
    bi_encoder    = SentenceTransformer(retrieval_model_name, trust_remote_code=True)
    cross_encoder = CrossEncoder(cross_encoder_model_name)

    print(f"[Eval] Loading KG from {kg_path} (engine={engine})...")
    retriever = build_retriever(engine, bi_encoder, cross_encoder, kg_path)

    error_tracker    = ErrorTracker()
    qa_eval_result   = None
    fact_eval_result = None

    if qa_file:
        print("\n[*] QA EVALUATION")
        qa_eval_result = run_qa_evaluation(
            qa_file=qa_file,
            retriever=retriever,
            retrieval_args=retrieval_args,
            tokenizer=tokenizer,
            max_context_tokens=max_context_tokens,
            max_workers=max_workers_eval,
            error_tracker=error_tracker,
        )

    if fact_file:
        print("\n[*] FACT RECALL EVALUATION")
        fact_eval_result = run_fact_evaluation(
            fact_file=fact_file,
            retriever=retriever,
            retrieval_args=retrieval_args,
            tokenizer=tokenizer,
            max_context_tokens=max_context_tokens,
            max_workers=max_workers_eval,
            error_tracker=error_tracker,
        )

    error_summary = error_tracker.summary()

    final_output = {
        "run_id":             f"nkg_eval_{uuid.uuid4().hex[:8]}",
        "timestamp":          datetime.now().strftime("%Y%m%d_%H%M%S"),
        "system":             "NKG",
        "engine":             engine,
        "eval_type":          eval_type,
        "construction":       construction_record,
        "retrieval_args":     retrieval_args,
        "max_context_tokens": max_context_tokens,
        "qa_evaluation":      qa_eval_result,
        "fact_evaluation":    fact_eval_result,
        "error_summary":      error_summary,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2, default=str)

    print(f"\n[Eval] Saved → {out_path}")
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


# =============================================================================
# AUTO MODE — single contract (construction + evaluation)
# =============================================================================
def run_auto_contract(
    contract_path: str,
    qa_file,
    fact_file,
    out_dir: str,
    eval_type: str,
    engine: str,
    construction_args: dict,
    retrieval_args: dict,
    retrieval_model_name: str,
    cross_encoder_model_name: str,
    tokenizer,
    max_context_tokens: int,
    max_workers_eval: int,
    construction_max_tokens: int = 55000,
    judge_max_tokens: int = 8000,
):
    contract_name = os.path.splitext(os.path.basename(contract_path))[0]
    eval_tag      = "_".join(
        (["qa"]   if qa_file   and eval_type in ("qa",   "both") else [])
        + (["fact"] if fact_file and eval_type in ("fact", "both") else [])
    ) or "none"
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

    # ---- Phase 1: Construction ----
    print("\n[*] PHASE 1 — GRAPH CONSTRUCTION")
    try:
        _graph, construction_record = build_graph(
            contract_path=contract_path,
            out_dir=run_dir,
            construction_args=construction_args,
            error_tracker=error_tracker,
            construction_max_tokens=construction_max_tokens,
        )
    except Exception as e:
        print(f"[FATAL] Construction failed for {contract_name}: {e}")
        traceback.print_exc()
        sys.stdout = tee._stream
        tee.close()
        return

    # ---- Phase 2: Initialize Retriever from saved KG ----
    # Reconfigure DSPy with judge token budget before evaluation begins.
    print(f"\n[*] PHASE 2 — INITIALIZING RETRIEVER (engine={engine})")
    configure_dspy(max_tokens=judge_max_tokens)
    kg_path = os.path.join(run_dir, KG_FILENAME)
    try:
        bi_encoder    = SentenceTransformer(retrieval_model_name, trust_remote_code=True)
        cross_encoder = CrossEncoder(cross_encoder_model_name)
        retriever = build_retriever(engine, bi_encoder, cross_encoder, kg_path)
    except Exception as e:
        print(f"[FATAL] Retriever init failed for {contract_name}: {e}")
        traceback.print_exc()
        sys.stdout = tee._stream
        tee.close()
        return

    # ---- Phase 3: Evaluation ----
    qa_eval_result   = None
    fact_eval_result = None

    if qa_file and eval_type in ("qa", "both"):
        print("\n[*] PHASE 3a — QA EVALUATION")
        qa_eval_result = run_qa_evaluation(
            qa_file=qa_file,
            retriever=retriever,
            retrieval_args=retrieval_args,
            tokenizer=tokenizer,
            max_context_tokens=max_context_tokens,
            max_workers=max_workers_eval,
            error_tracker=error_tracker,
        )

    if fact_file and eval_type in ("fact", "both"):
        print("\n[*] PHASE 3b — FACT RECALL EVALUATION")
        fact_eval_result = run_fact_evaluation(
            fact_file=fact_file,
            retriever=retriever,
            retrieval_args=retrieval_args,
            tokenizer=tokenizer,
            max_context_tokens=max_context_tokens,
            max_workers=max_workers_eval,
            error_tracker=error_tracker,
        )

    error_summary = error_tracker.summary()

    final_output = {
        "run_id":             f"nkg_{uuid.uuid4().hex[:8]}",
        "timestamp":          timestamp,
        "system":             "NKG",
        "engine":             engine,
        "eval_type":          eval_type,
        "construction":       construction_record,
        "retrieval_args":     retrieval_args,
        "max_context_tokens": max_context_tokens,
        "qa_evaluation":      qa_eval_result,
        "fact_evaluation":    fact_eval_result,
        "error_summary":      error_summary,
    }

    results_path = os.path.join(run_dir, "results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2, default=str)

    # ---- Terminal summary ----
    cm = construction_record["metrics"]
    print(f"\n  Log      → {log_path}")
    print(f"  Results  → {results_path}")
    print(f"  Document chars   : {cm['total_document_chars']}")
    print(f"  Nodes            : {cm['n_nodes']}  (Entities: {cm['n_entities']}, Facts: {cm['n_facts']})")
    print(f"  Edges            : {cm['n_edges']}")
    if "token_breakdown" in cm:
        _print_token_breakdown(cm["token_breakdown"])
    else:
        print(f"  LLM tokens       : {cm['llm_total_tokens']}  "
              f"(prompt: {cm['llm_prompt_tokens']}, completion: {cm['llm_completion_tokens']}, "
              f"calls: {cm['llm_api_calls']})")
    print(f"  Wall time        : {cm['execution_time_sec']}s")
    if qa_eval_result:
        print(f"  QA (excl CAT5)   : {qa_eval_result['overall_avg_score_excl_cat5']}")
        for cat, s in qa_eval_result["by_category"].items():
            print(f"    {cat}: {s['avg_score']}  (n={s['count']})")
    if fact_eval_result:
        print(f"  Fact Recall      : {fact_eval_result['overall_recall_score']}  "
              f"({fact_eval_result['total_recalled']}/{fact_eval_result['total_facts']})")
    print(f"\n  Errors (total)   : {error_summary['total_errors']}")
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
        description="NKG Insurance Contract Evaluation — construction / evaluation / auto modes"
    )

    # ---- Mode ----
    parser.add_argument(
        "--mode", type=str, required=True,
        choices=["construction", "evaluation", "auto"],
        help="construction: build KGs only. evaluation: evaluate existing KGs. auto: both."
    )

    # ---- Dataset selection (construction + auto) ----
    dataset = parser.add_mutually_exclusive_group()
    dataset.add_argument("--contract", type=str,
                         help="Path to a single contract .txt file")
    dataset.add_argument("--all", action="store_true",
                         help="All contracts in both dataset sections under --eval-data-dir")
    dataset.add_argument("--ten-contracts-dataset", action="store_true",
                         help="Only ten_contracts_dataset section under --eval-data-dir")
    dataset.add_argument("--long-form-contracts", action="store_true",
                         help="Only long_form_contracts section under --eval-data-dir")

    parser.add_argument(
        "--exclude-contract", action="append", dest="exclude_contracts",
        metavar="NAME", default=[],
        help="Contract filename (stem, no extension) to exclude. Can be specified multiple times.",
    )

    # ---- Single contract qa/fact overrides (auto mode only) ----
    parser.add_argument("--qa-file",   type=str, default=None,
                        help="QA JSON for single-contract auto mode (if not using --eval-data-dir resolution)")
    parser.add_argument("--fact-file", type=str, default=None,
                        help="Fact JSON for single-contract auto mode (if not using --eval-data-dir resolution)")

    # ---- Directories ----
    parser.add_argument("--eval-data-dir", type=str, default=None,
                        help="Root evaluation_data/ directory (required for multi-contract and evaluation modes)")
    parser.add_argument("--out-dir", type=str, default="./results",
                        help="Output root for construction and auto modes")
    parser.add_argument("--target-dir", type=str, default=None,
                        help="Directory to evaluate (evaluation mode). "
                             "If it contains construction.json + kg.graphml, evaluate it directly. "
                             "Otherwise scan subdirectories.")

    # ---- Eval type ----
    parser.add_argument("--eval-type", type=str, default="both",
                        choices=["qa", "fact", "both"],
                        help="Which evaluation(s) to run (evaluation and auto modes)")

    # ---- Construction args ----
    parser.add_argument("--model-name", type=str,
                        default="Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
                        help="Model name used by DSPy (LLM extraction) and text chunking tokenizer")
    parser.add_argument("--max-workers",        type=int,   default=4,
                        help="Worker threads for parallel chunk processing")
    parser.add_argument("--chunk-size",         type=int,   default=600)
    parser.add_argument("--overlap",            type=int,   default=50)
    parser.add_argument("--edge-threshold",     type=float, default=0.5)
    parser.add_argument("--top-k",              type=int,   default=12,
                        help="Top-K similar facts considered for global cross-chunk edges")
    parser.add_argument("--max-sub-workers",    type=int,   default=10,
                        help="Sub-worker threads for edge construction and deduplication")
    parser.add_argument("--fact-batch-size",    type=int,   default=15,
                        help="Number of facts per LLM extraction batch")
    parser.add_argument("--edge-mode",          type=str,   default="clustered",
                        choices=["linear", "clustered"],
                        help="Global cross-chunk fact edge construction strategy")
    parser.add_argument("--no-deduplication",   action="store_true",
                        help="Skip entity deduplication step")
    parser.add_argument("--dedup-model-name",   type=str,   default="voyageai/voyage-4-nano",
                        help="Sentence-transformers model for entity deduplication similarity")
    parser.add_argument("--dedup-iterations",   type=int,   default=1)
    parser.add_argument("--dedup-max-cluster-size", type=int, default=64)
    parser.add_argument("--dedup-run-exact", action=argparse.BooleanOptionalAction, default=True,
                        help="Run phase 1 exact-name entity dedup (default: on)")
    parser.add_argument("--dedup-run-semhash", action=argparse.BooleanOptionalAction, default=True,
                        help="Run phase 2 SemHash near-duplicate entity dedup (default: on)")
    parser.add_argument("--dedup-semhash-threshold", type=float, default=0.95,
                        help="SemHash near-duplicate similarity threshold (default: 0.95)")
    parser.add_argument("--dedup-run-llm", action=argparse.BooleanOptionalAction, default=True,
                        help="Run phase 3 LLM-confirmed entity dedup (default: on)")
    parser.add_argument("--dedup-similarity-threshold", type=float, default=0.75,
                        help="Similarity-graph edge threshold for phase 3 LLM dedup clustering (default: 0.75)")
    parser.add_argument("--use-cot", action=argparse.BooleanOptionalAction, default=False,
                        help="Enable chain-of-thought in DSPy calls during construction (default: off)")
    parser.add_argument("--neutral-description", action=argparse.BooleanOptionalAction, default=True,
                        help="Use a single neutral edge description instead of forward+backward (default: on)")
    parser.add_argument("--concise-summary", action=argparse.BooleanOptionalAction, default=True,
                        help="Use 1-sentence chunk summaries instead of 3-4 sentences (default: on)")
    parser.add_argument("--combined-entity-extraction", action=argparse.BooleanOptionalAction, default=True,
                        help="Extract entities in one combined LLM call instead of per-entity calls (default: on)")
    parser.add_argument("--full-entity-profile", action=argparse.BooleanOptionalAction, default=False,
                        help="Extract full entity profiles (type/role/anchors) instead of names only (default: off)")
    parser.add_argument("--clustered-round1", action=argparse.BooleanOptionalAction, default=False,
                        help="Run entity-based clustering round for global edges (default: off)")
    parser.add_argument("--clustered-round2", action=argparse.BooleanOptionalAction, default=True,
                        help="Run balanced top-k clustering round for global edges (default: on)")
    parser.add_argument("--clustered-round3", action=argparse.BooleanOptionalAction, default=True,
                        help="Run balanced middle-k clustering round for global edges (default: on)")
    parser.add_argument("--clustered-max-facts-per-prompt", type=int, default=40,
                        help="Max facts per cluster LLM prompt for global edges (default: 40)")
    parser.add_argument("--middle-k-skip-percent", type=float, default=0.33,
                        help="Fraction of top-ranked facts to skip before taking the next top_k, per fact, "
                             "when building round 3's (middle-k) similarity graph. 0.0 disables skipping "
                             "(default: 0.33)")
    parser.add_argument("--global-edge-retrieval-model", type=str, default="voyageai/voyage-4-nano",
                        help="Sentence-transformers model for global fact edge similarity search (default: voyageai/voyage-4-nano)")
    parser.add_argument("--use-questions-weight", action=argparse.BooleanOptionalAction, default=True,
                        help="Include the answered/follow-up questions similarity dimension in global-edge "
                             "clustering rounds 2 and 3 (default: on)")
    parser.add_argument("--questions-mode", type=str, default="follow_up_to_answered",
                        choices=["follow_up_to_answered", "follow_up_to_fact_sentence", "fact_sentence_to_follow_up"],
                        help="Asymmetric matching strategy for the questions similarity dimension: "
                             "'follow_up_to_answered' matches this fact's follow-up questions against other "
                             "facts' answered questions (default); 'follow_up_to_fact_sentence' matches this "
                             "fact's follow-up questions against other facts' sentences; "
                             "'fact_sentence_to_follow_up' matches this fact's sentence against other facts' "
                             "follow-up questions.")

    # ---- Retrieval models ----
    parser.add_argument("--retrieval-model", type=str,
                        default="voyageai/voyage-4-nano",
                        help="Sentence-transformers bi-encoder for graph retrieval")
    parser.add_argument("--cross-encoder-model", type=str,
                        default="cross-encoder/ms-marco-MiniLM-L6-v2",
                        help="Cross-encoder model for path re-ranking")

    # ---- Engine selection ----
    parser.add_argument("--engine", type=str, default=DEFAULT_ENGINE, choices=ENGINE_CHOICES,
                        help="Which retrieval engine to use: 'new' (nkg.search.GoalSearchEngine, "
                             "goal-coverage redesign, default), 'old' (nkg.retrieval.Retriever, "
                             "MMR-based), or 'dag' (nkg.dag_retrieval.DAGRetriever, decomposes the "
                             "query into a DAG of subqueries and searches each with the old engine).")

    # ---- Retrieval query args ----
    parser.add_argument("--default-retrieval-args", action="store_true",
                        help="Start from the hardcoded default retrieval parameters for the selected "
                             "--engine. Any retrieval flag below that is also explicitly passed on the "
                             "command line overrides its corresponding default.")
    parser.add_argument("--deduplicate-paths", action=argparse.BooleanOptionalAction, default=True,
                        help="Cluster retrieved paths by node overlap and keep only the top path per "
                             "cluster (default: on). Pass --no-deduplicate-paths to keep all paths.")
    parser.add_argument("--top-k-seeds",      type=int,   default=None)
    parser.add_argument("--max-depth",        type=int,   default=None)
    parser.add_argument("--beam-width",       type=int,   default=None)
    parser.add_argument("--final-top-k",      type=int,   default=None)
    parser.add_argument("--retrieval-node-mode", type=str, default=None,
                        help="Mode passed to retriever.retrieve() controlling which node/edge types are used")

    # ---- Retrieval query args: old/dag engines (dag delegates each subquery's search to the
    # old engine, so these apply to it too) ----
    parser.add_argument("--beam-mmr-lambda",  type=float, default=None,
                        help="[old/dag engine only] Local per-hop MMR relevance/diversity trade-off")
    parser.add_argument("--final-mmr-lambda", type=float, default=None,
                        help="[old/dag engine only] Final global MMR relevance/diversity trade-off")
    parser.add_argument("--search-mode",      type=str,   default=None,
                        choices=["beam_search", "blind_expansion"],
                        help="[old/dag engine only] Graph expansion strategy")

    # ---- Retrieval query args: dag engine only (nkg.dag_retrieval.DAGRetriever) ----
    parser.add_argument("--max-fork-branches", type=int, default=None,
                        help="[dag engine only] Cap on how many dependency-combination / "
                             "disambiguation forks a single DAG node may expand into")

    # ---- Retrieval query args: new engine only (nkg.search.GoalSearchEngine) ----
    parser.add_argument("--goal-lookahead", type=int, default=None,
                        help="[new engine only] How many independent upcoming goals traversal may "
                             "opportunistically score against, in addition to the current depth's goal")
    parser.add_argument("--use-island-and-bridge", action=argparse.BooleanOptionalAction, default=None,
                        help="[new engine only] Optional D5 stage: also builds ready-made chains from "
                             "directly-connected pairs of independently high-relevance nodes (default: off)")
    parser.add_argument("--island-pool-size", type=int, default=None,
                        help="[new engine only, island-and-bridge] Size of the relevance-ranked pool "
                             "searched for bridges")
    parser.add_argument("--island-max-bridge-hops", type=int, default=None,
                        help="[new engine only, island-and-bridge] Extra hops each bridge may be "
                             "greedily extended by")

    # ---- Evaluation settings ----
    parser.add_argument("--tokenizer-model", type=str,
                        default="Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
                        help="HuggingFace tokenizer for context truncation")
    parser.add_argument("--max-context-tokens",       type=int, default=12000)
    parser.add_argument("--judge-max-tokens",         type=int, default=8000,
                        help="Max tokens for the DSPy judge LLM (evaluation)")
    parser.add_argument("--construction-max-tokens",  type=int, default=40000,
                        help="Max tokens for the DSPy LLM during KG construction")
    parser.add_argument("--max-workers-eval",   type=int, default=10,
                        help="Thread-pool size for QA/Fact evaluation workers")
    parser.add_argument("--force-evaluation",   action="store_true",
                        help="Re-run evaluation even if output for the same settings already exists")

    args = parser.parse_args()

    # ---- Validate ----
    multi_contract = args.all or args.ten_contracts_dataset or args.long_form_contracts

    if args.mode in ("construction", "auto"):
        if not args.contract and not multi_contract:
            parser.error(
                f"--mode {args.mode} requires one of: --contract, --all, "
                "--ten-contracts-dataset, --long-form-contracts"
            )
        if multi_contract and not args.eval_data_dir:
            parser.error(
                "--all, --ten-contracts-dataset, and --long-form-contracts require --eval-data-dir"
            )

    if args.mode == "evaluation":
        if not args.target_dir:
            parser.error("--mode evaluation requires --target-dir")
        if not args.eval_data_dir:
            parser.error("--mode evaluation requires --eval-data-dir to resolve QA/Fact files")

    # For auto single-contract mode, require either eval-data-dir or explicit qa/fact files
    if args.mode == "auto" and args.contract:
        has_explicit_files = args.qa_file or args.fact_file
        if not has_explicit_files and not args.eval_data_dir:
            parser.error(
                "Auto mode with --contract requires either --eval-data-dir or "
                "--qa-file / --fact-file"
            )

    # ---- Build retrieval args dict (engine-specific) ----
    # CLI retrieval flags default to None so we can tell "not passed" apart from
    # "explicitly passed" — this lets --default-retrieval-args be combined with
    # explicit overrides (e.g. --default-retrieval-args --final-top-k 40) without
    # those overrides silently being dropped from the args dict (and therefore from
    # its hash, which is what determines whether a run is treated as "already evaluated").
    CLI_RETRIEVAL_FALLBACKS = {
        "top_k_seeds":            9,
        "max_depth":              3,
        "beam_width":             6,
        "final_top_k":            20,
        "retrieval_node_mode":    "all",
        "beam_mmr_lambda":        1.0,
        "final_mmr_lambda":       0.6,
        "search_mode":            "beam_search",
        "goal_lookahead":         1,
        "use_island_and_bridge":  False,
        "island_pool_size":       75,
        "island_max_bridge_hops": 1,
        "max_fork_branches":      6,
    }
    # DAG's per-node search is intentionally shallow (each node is one focused sub-lookup, not a
    # multi-hop search in its own right) -- max_depth=1 rather than the old/new engines' 3.
    DAG_MAX_DEPTH_FALLBACK = 1

    def cli_val(name):
        """args.<name> if explicitly passed on the command line, else its normal default."""
        v = getattr(args, name)
        if v is not None:
            return v
        if name == "max_depth" and args.engine == "dag":
            return DAG_MAX_DEPTH_FALLBACK
        return CLI_RETRIEVAL_FALLBACKS[name]

    ENGINE_OVERRIDABLE = {
        "new": ["goal_lookahead", "use_island_and_bridge", "island_pool_size", "island_max_bridge_hops"],
        "old": ["beam_mmr_lambda", "final_mmr_lambda", "search_mode"],
        "dag": ["beam_mmr_lambda", "final_mmr_lambda", "search_mode", "max_fork_branches"],
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
        print(f"[*] Using default retrieval args for engine='{args.engine}' "
              f"(with explicit CLI overrides applied): {retrieval_args}")
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
            "deduplicate_paths": args.deduplicate_paths,
            "neutral_layout":    True,
        }
    else:  # args.engine == "old"
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
            "neutral_layout": True,
        }

    # ---- Build construction args dict ----
    construction_args = {
        "model_name":                    args.model_name,
        "max_workers":                   args.max_workers,
        "chunk_size":                    args.chunk_size,
        "overlap":                       args.overlap,
        "edge_threshold":                args.edge_threshold,
        "top_k":                         args.top_k,
        "max_sub_workers":               args.max_sub_workers,
        "fact_batch_size":               args.fact_batch_size,
        "edge_mode":                     args.edge_mode,
        "run_deduplication":             not args.no_deduplication,
        "dedup_model_name":              args.dedup_model_name,
        "dedup_iterations":              args.dedup_iterations,
        "dedup_max_cluster_size":        args.dedup_max_cluster_size,
        "dedup_run_exact":               args.dedup_run_exact,
        "dedup_run_semhash":             args.dedup_run_semhash,
        "dedup_semhash_threshold":       args.dedup_semhash_threshold,
        "dedup_run_llm":                 args.dedup_run_llm,
        "dedup_similarity_threshold":    args.dedup_similarity_threshold,
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
        "retrieval_model_name":           args.global_edge_retrieval_model,
        "use_questions_weight":          args.use_questions_weight,
        "questions_mode":                args.questions_mode,
    }

    # ---- DSPy configuration — mode-specific ----
    # construction: high token budget for LLM extraction
    # evaluation:   judge token budget (lower, answers are short)
    # auto:         build_graph() handles construction config internally;
    #               run_auto_contract() reconfigures for judge before eval
    if args.mode == "evaluation":
        configure_dspy(max_tokens=args.judge_max_tokens)
    else:
        # construction and auto: configure for construction now; auto reconfigures for judge later
        configure_dspy(max_tokens=args.construction_max_tokens, cache=False)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_model)

    # ---- Collect contracts (construction + auto) ----
    if args.mode in ("construction", "auto"):
        if args.contract:
            # Single contract — qa/fact resolved later per mode
            contract_path = os.path.abspath(args.contract)
            if args.eval_data_dir:
                try:
                    qa_path, fact_path = resolve_paths_for_contract(
                        contract_path, os.path.abspath(args.eval_data_dir)
                    )
                except (ValueError, FileNotFoundError) as e:
                    if args.mode == "auto" and not (args.qa_file or args.fact_file):
                        parser.error(f"Could not resolve qa/fact files: {e}")
                    qa_path = fact_path = None
            else:
                qa_path = fact_path = None

            # Explicit overrides for auto mode
            if args.mode == "auto":
                if args.qa_file:
                    qa_path = args.qa_file
                if args.fact_file:
                    fact_path = args.fact_file

            triplets = [(contract_path, qa_path, fact_path)]
        else:
            if args.all:
                subdirs_to_scan = DATASET_SUBDIRS
            elif args.ten_contracts_dataset:
                subdirs_to_scan = ["ten_contracts_dataset"]
            else:
                subdirs_to_scan = ["long_form_contracts"]

            triplets = collect_all_contracts(
                os.path.abspath(args.eval_data_dir), subdirs=subdirs_to_scan
            )
            if args.exclude_contracts:
                excluded_stems = {e.lower() for e in args.exclude_contracts}
                before = len(triplets)
                triplets = [
                    t for t in triplets
                    if os.path.splitext(os.path.basename(t[0]))[0].lower() not in excluded_stems
                ]
                print(f"[*] Excluded {before - len(triplets)} contract(s): {args.exclude_contracts}")
            if not triplets:
                print("[ERROR] No contracts found. Check --eval-data-dir.")
                sys.exit(1)
            print(f"[*] Found {len(triplets)} contracts.\n")

    # ==========================================================================
    # CONSTRUCTION MODE
    # ==========================================================================
    if args.mode == "construction":
        for i, (contract_path, _qa, _fact) in enumerate(triplets, 1):
            contract_name    = os.path.splitext(os.path.basename(contract_path))[0]
            contract_out_dir = os.path.join(args.out_dir, contract_name)

            print(f"\n{'#' * 65}")
            print(f"# CONSTRUCTION {i}/{len(triplets)}: {contract_name}")
            print(f"{'#' * 65}")

            os.makedirs(contract_out_dir, exist_ok=True)
            log_path   = os.path.join(contract_out_dir, "construction.log")
            tee        = TeeStream(log_path, sys.stdout)
            sys.stdout = tee

            error_tracker = ErrorTracker()
            try:
                build_graph(
                    contract_path=contract_path,
                    out_dir=contract_out_dir,
                    construction_args=construction_args,
                    error_tracker=error_tracker,
                    construction_max_tokens=args.construction_max_tokens,
                )
            except Exception as e:
                print(f"[FATAL] Construction failed for {contract_name}: {e}")
                traceback.print_exc()
            finally:
                sys.stdout = tee._stream
                tee.close()

    # ==========================================================================
    # EVALUATION MODE
    # ==========================================================================
    elif args.mode == "evaluation":
        target_dir = os.path.abspath(args.target_dir)
        eval_data_dir = os.path.abspath(args.eval_data_dir)

        if is_construction_dir(target_dir):
            dirs_to_eval = [target_dir]
            print(f"[*] Single construction directory: {target_dir}")
        else:
            dirs_to_eval = []
            for name in sorted(os.listdir(target_dir)):
                sub = os.path.join(target_dir, name)
                if is_construction_dir(sub):
                    dirs_to_eval.append(sub)
            if not dirs_to_eval:
                print(f"[ERROR] No construction directories found under {target_dir}")
                sys.exit(1)
            print(f"[*] Found {len(dirs_to_eval)} construction directories to evaluate.")

        for i, cdir in enumerate(dirs_to_eval, 1):
            print(f"\n{'#' * 65}")
            print(f"# EVAL {i}/{len(dirs_to_eval)}: {os.path.basename(cdir)}")
            print(f"{'#' * 65}")
            try:
                evaluate_construction_dir(
                    construction_dir=cdir,
                    eval_data_dir=eval_data_dir,
                    eval_type=args.eval_type,
                    engine=args.engine,
                    retrieval_args=retrieval_args,
                    retrieval_model_name=args.retrieval_model,
                    cross_encoder_model_name=args.cross_encoder_model,
                    tokenizer=tokenizer,
                    max_context_tokens=args.max_context_tokens,
                    max_workers_eval=args.max_workers_eval,
                    force_evaluation=args.force_evaluation,
                )
            except Exception as e:
                print(f"[FATAL] Evaluation failed for {cdir}: {e}")
                traceback.print_exc()
                continue

    # ==========================================================================
    # AUTO MODE
    # ==========================================================================
    elif args.mode == "auto":
        for i, (contract_path, qa_path, fact_path) in enumerate(triplets, 1):
            print(f"\n{'#' * 65}")
            print(f"# AUTO {i}/{len(triplets)}: {os.path.basename(contract_path)}")
            print(f"{'#' * 65}")
            try:
                run_auto_contract(
                    contract_path=contract_path,
                    qa_file=qa_path,
                    fact_file=fact_path,
                    out_dir=args.out_dir,
                    eval_type=args.eval_type,
                    engine=args.engine,
                    construction_args=construction_args,
                    retrieval_args=retrieval_args,
                    retrieval_model_name=args.retrieval_model,
                    cross_encoder_model_name=args.cross_encoder_model,
                    tokenizer=tokenizer,
                    max_context_tokens=args.max_context_tokens,
                    max_workers_eval=args.max_workers_eval,
                    construction_max_tokens=args.construction_max_tokens,
                    judge_max_tokens=args.judge_max_tokens,
                )
            except Exception as e:
                print(f"[FATAL] Contract {contract_path} failed: {e}")
                traceback.print_exc()
                continue


if __name__ == "__main__":
    main()


