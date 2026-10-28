#!/usr/bin/env python3
"""
NKG Retrieval Hyperparameter Random Search

Given a dataset directory of already-built KGs (the same construction_dir
layout produced by insurance_evaluation.py's --mode construction: each
sub-directory holds construction.json + kg.graphml), this randomly samples
retrieval-argument combinations for the selected --engine, evaluates every
contract under that parameter set, and reports the combination with the
highest average score across all contracts.

Each contract's KG is loaded and embedded exactly once (retrieval_args only
affect .retrieve() calls, not graph construction), so the expensive part of
setup happens once regardless of --n-trials. Progress is written to --out
after every trial, so an interrupted run still leaves a usable partial log.

---------------------------------------------------------------------
Example:
    python3 -m experiments.systems.nkg.optimize_retrieval \\
        --target-dir "token_reduction_experiments/insurance/nkg/.../ten_contracts_dataset" \\
        --eval-data-dir ./evaluation_data \\
        --engine new --eval-type qa \\
        --n-trials 40 --max-workers-eval 30
---------------------------------------------------------------------
"""

import os
import sys
import json
import time
import random
import argparse
import traceback
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer

from nkg.utils.config import configure_dspy
from experiments.systems.nkg.insurance_evaluation import (
    KG_FILENAME,
    CONSTRUCTION_JSON,
    ENGINE_CHOICES,
    DEFAULT_ENGINE,
    build_retriever,
    resolve_paths_for_contract,
    is_construction_dir,
    ErrorTracker,
    _eval_qa_item,
    _eval_fact_item,
)


# =============================================================================
# HYPERPARAMETER SEARCH SPACES
# =============================================================================
# Params not listed here are held fixed (see *_FIXED_ARGS below) because
# varying them has no meaningful effect on answer quality (e.g. deduplicate_paths).
NEW_ENGINE_PARAM_SPACE = {
    "top_k_seeds":            [5, 8, 10, 12, 15, 20],
    "max_depth":               [1, 2, 3, 4],
    "beam_width":               [2, 3, 4, 6, 8, 10],
    "final_top_k":               [5, 10, 15, 20, 25, 30],
    "goal_lookahead":            [0, 1, 2],
    "mode":                      ["all", "hypergraph", "discourse"],
    "use_island_and_bridge":     [False, True],
    "island_pool_size":          [30, 50, 75, 100, 150],
    "island_max_bridge_hops":    [1, 2],
}
NEW_ENGINE_FIXED_ARGS = {"deduplicate_paths": True, "neutral_layout": True}

OLD_ENGINE_PARAM_SPACE = {
    "top_k_seeds":       [5, 8, 10, 12, 15, 20],
    "max_depth":         [1, 2, 3, 4],
    "beam_width":        [2, 3, 4, 6, 8, 10],
    "final_top_k":       [5, 10, 15, 20, 25, 30],
    "mode":              ["all", "hypergraph", "discourse"],
    "beam_mmr_lambda":   [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    "final_mmr_lambda":  [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    "search_mode":       ["beam_search", "blind_expansion"],
}
OLD_ENGINE_FIXED_ARGS = {"deduplicate_paths": True, "neutral_layout": True}


def sample_retrieval_args(engine: str, rng: random.Random) -> dict:
    space = NEW_ENGINE_PARAM_SPACE if engine == "new" else OLD_ENGINE_PARAM_SPACE
    fixed = NEW_ENGINE_FIXED_ARGS if engine == "new" else OLD_ENGINE_FIXED_ARGS
    sampled = {k: rng.choice(v) for k, v in space.items()}
    sampled.update(fixed)
    return sampled


# =============================================================================
# SCORING — reuses the same per-item worker functions as insurance_evaluation.py
# (identical LLM-answer + judge-scoring logic), just without the file-loading
# and JSON-summary wrapper, so a subsampled item list can be passed in directly.
# =============================================================================
def _score_qa_items(pairs, retriever, retrieval_args, tokenizer, max_context_tokens, max_workers, error_tracker):
    if not pairs:
        return None
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(_eval_qa_item, p, retriever, retrieval_args, tokenizer, max_context_tokens, error_tracker)
            for p in pairs
        ]
        results = [f.result() for f in as_completed(futures)]
    non_cat5 = [r["score"] for r in results if r["category"] != "CAT5_UNANSWERABLE"]
    return round(sum(non_cat5) / len(non_cat5), 4) if non_cat5 else 0.0


def _score_fact_items(facts, retriever, retrieval_args, tokenizer, max_context_tokens, max_workers, error_tracker):
    if not facts:
        return None
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(_eval_fact_item, fd, retriever, retrieval_args, tokenizer, max_context_tokens, error_tracker)
            for fd in facts
        ]
        results = [f.result() for f in as_completed(futures)]
    scores = [r["score"] for r in results]
    return round(sum(scores) / len(scores), 4) if scores else 0.0


# =============================================================================
# DATASET DISCOVERY
# =============================================================================
def discover_construction_dirs(target_dir: str) -> list:
    if is_construction_dir(target_dir):
        return [target_dir]
    return [
        os.path.join(target_dir, name)
        for name in sorted(os.listdir(target_dir))
        if is_construction_dir(os.path.join(target_dir, name))
    ]


def load_contracts(cdirs, eval_data_dir, eval_type, engine, bi_encoder, cross_encoder, rng,
                    qa_sample_size, fact_sample_size):
    """
    Builds one retriever per contract and pre-loads (optionally subsampled) QA/fact
    items once, so every trial's parameter set is scored on identical data and no
    contract is re-embedded per trial.
    """
    contracts = []
    for cdir in cdirs:
        with open(os.path.join(cdir, CONSTRUCTION_JSON), "r", encoding="utf-8") as f:
            record = json.load(f)
        contract_path = record["contract_file"]
        name = os.path.splitext(os.path.basename(contract_path))[0]

        try:
            qa_path, fact_path = resolve_paths_for_contract(contract_path, eval_data_dir)
        except (ValueError, FileNotFoundError) as e:
            print(f"[WARN] Skipping {name}: {e}")
            continue

        qa_pairs, facts = [], []
        if eval_type in ("qa", "both"):
            with open(qa_path, "r", encoding="utf-8") as f:
                qa_pairs = json.load(f).get("qa_pairs", [])
            if qa_sample_size and len(qa_pairs) > qa_sample_size:
                qa_pairs = rng.sample(qa_pairs, qa_sample_size)
        if eval_type in ("fact", "both"):
            with open(fact_path, "r", encoding="utf-8") as f:
                facts = json.load(f).get("facts", [])
            if fact_sample_size and len(facts) > fact_sample_size:
                facts = rng.sample(facts, fact_sample_size)

        print(f"[*] Loading KG for {name} (engine={engine})...")
        kg_path = os.path.join(cdir, KG_FILENAME)
        retriever = build_retriever(engine, bi_encoder, cross_encoder, kg_path)

        contracts.append({"name": name, "retriever": retriever, "qa_pairs": qa_pairs, "facts": facts})

    return contracts


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Random search over NKG retrieval hyperparameters across all contracts in a dataset directory."
    )
    parser.add_argument("--target-dir", type=str, required=True,
                        help="Directory of construction dirs (construction.json + kg.graphml), or a single one.")
    parser.add_argument("--eval-data-dir", type=str, required=True,
                        help="Root evaluation_data/ directory, used to resolve QA/Fact files per contract.")
    parser.add_argument("--engine", type=str, default=DEFAULT_ENGINE, choices=ENGINE_CHOICES,
                        help="Which retrieval engine's hyperparameter space to search.")
    parser.add_argument("--eval-type", type=str, default="qa", choices=["qa", "fact", "both"],
                        help="Which evaluation(s) contribute to the per-trial score.")
    parser.add_argument("--n-trials", type=int, default=30,
                        help="Number of randomly sampled parameter combinations to try.")
    parser.add_argument("--qa-sample-size", type=int, default=None,
                        help="Cap on QA pairs used per contract (same subsample reused every trial). Default: use all.")
    parser.add_argument("--fact-sample-size", type=int, default=None,
                        help="Cap on facts used per contract (same subsample reused every trial). Default: use all.")
    parser.add_argument("--retrieval-model", type=str, default="voyageai/voyage-4-nano")
    parser.add_argument("--cross-encoder-model", type=str, default="cross-encoder/ms-marco-MiniLM-L6-v2")
    parser.add_argument("--tokenizer-model", type=str, default="Qwen/Qwen3-30B-A3B-Instruct-2507-FP8")
    parser.add_argument("--max-context-tokens", type=int, default=12000)
    parser.add_argument("--judge-max-tokens", type=int, default=8000)
    parser.add_argument("--max-workers-eval", type=int, default=10,
                        help="Thread-pool size for QA/Fact evaluation workers, per contract per trial.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling and subsampling.")
    parser.add_argument("--out", type=str, default=None,
                        help="Output JSON path for the trial log + best params. "
                             "Default: <target-dir>/optimize_<engine>_<eval-type>_<timestamp>.json")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    configure_dspy(max_tokens=args.judge_max_tokens)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_model)

    target_dir = os.path.abspath(args.target_dir)
    eval_data_dir = os.path.abspath(args.eval_data_dir)

    cdirs = discover_construction_dirs(target_dir)
    if not cdirs:
        print(f"[ERROR] No construction directories found under {target_dir}")
        sys.exit(1)
    print(f"[*] Found {len(cdirs)} construction directories.")

    print(f"[*] Loading models (retrieval={args.retrieval_model}, cross_encoder={args.cross_encoder_model})...")
    bi_encoder = SentenceTransformer(args.retrieval_model, trust_remote_code=True)
    cross_encoder = CrossEncoder(args.cross_encoder_model)

    contracts = load_contracts(
        cdirs, eval_data_dir, args.eval_type, args.engine, bi_encoder, cross_encoder, rng,
        args.qa_sample_size, args.fact_sample_size,
    )
    if not contracts:
        print("[ERROR] No usable contracts (QA/Fact files not resolvable).")
        sys.exit(1)

    out_path = args.out or os.path.join(
        target_dir,
        f"optimize_{args.engine}_{args.eval_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )

    trials = []
    best = None

    for trial_i in range(1, args.n_trials + 1):
        retrieval_args = sample_retrieval_args(args.engine, rng)
        print(f"\n{'=' * 65}\nTRIAL {trial_i}/{args.n_trials}: {retrieval_args}\n{'=' * 65}")

        per_contract = {}
        t0 = time.time()
        for c in contracts:
            error_tracker = ErrorTracker()
            qa_score = fact_score = None
            try:
                if c["qa_pairs"]:
                    qa_score = _score_qa_items(
                        c["qa_pairs"], c["retriever"], retrieval_args, tokenizer,
                        args.max_context_tokens, args.max_workers_eval, error_tracker,
                    )
                if c["facts"]:
                    fact_score = _score_fact_items(
                        c["facts"], c["retriever"], retrieval_args, tokenizer,
                        args.max_context_tokens, args.max_workers_eval, error_tracker,
                    )
            except Exception as e:
                print(f"[ERROR] Trial {trial_i} contract {c['name']} failed: {e}")
                traceback.print_exc()

            component_scores = [s for s in (qa_score, fact_score) if s is not None]
            combined = round(sum(component_scores) / len(component_scores), 4) if component_scores else 0.0
            per_contract[c["name"]] = {
                "qa_score": qa_score, "fact_score": fact_score, "combined": combined,
                "errors": error_tracker.summary(),
            }
            print(f"  {c['name']}: qa={qa_score} fact={fact_score} combined={combined}")

        contract_avgs = [v["combined"] for v in per_contract.values()]
        avg_score = round(sum(contract_avgs) / len(contract_avgs), 4) if contract_avgs else 0.0
        elapsed = round(time.time() - t0, 1)
        print(f"[Trial {trial_i}] avg_score={avg_score}  ({elapsed}s)")

        trial_record = {
            "trial": trial_i,
            "retrieval_args": retrieval_args,
            "avg_score": avg_score,
            "per_contract": per_contract,
            "elapsed_sec": elapsed,
        }
        trials.append(trial_record)

        if best is None or avg_score > best["avg_score"]:
            best = trial_record
            print(f"[*] New best: avg_score={avg_score}")

        # Saved after every trial so an interrupted run still leaves a usable log.
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({
                "engine": args.engine,
                "eval_type": args.eval_type,
                "seed": args.seed,
                "n_trials_completed": trial_i,
                "n_trials_requested": args.n_trials,
                "contracts": [c["name"] for c in contracts],
                "best": best,
                "trials": trials,
            }, f, indent=2, default=str)

    print(f"\n{'#' * 65}\n BEST PARAMETERS (avg_score={best['avg_score']})\n{'#' * 65}")
    print(json.dumps(best["retrieval_args"], indent=2))
    print(f"\nFull log saved -> {out_path}")


if __name__ == "__main__":
    main()
