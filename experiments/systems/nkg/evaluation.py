import os
import json
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer, CrossEncoder

# Your custom architecture imports
from nkg.utils.config import configure_dspy
from nkg.retrieval.engine import Retriever
from experiments.evaluate_mine import QAEvaluator, FactEvaluator


# ==========================================
# 1. Context Truncation Utility
# ==========================================
def truncate_context(text: str, tokenizer: AutoTokenizer, max_tokens: int) -> tuple[str, int]:
    """Truncates retrieved text to exactly X tokens to ensure fair comparisons."""
    if not text:
        return "", 0
    tokens = tokenizer.encode(text)
    total_retrieved = len(tokens)
    if total_retrieved > max_tokens:
        return tokenizer.decode(tokens[:max_tokens]), max_tokens
    return text, total_retrieved


# ==========================================
# 2. Parallel Processing Logic
# ==========================================
def evaluate_qa_item(pair: dict, retriever: Retriever, evaluator: QAEvaluator,
                     tokenizer: AutoTokenizer, max_tokens: int) -> tuple[float, str]:
    q = pair["question"]
    a = pair["answer"]
    category = pair.get("category", "UNKNOWN_CATEGORY")

    try:
        # Custom Retrieval Parameters
        raw_context = retriever.retrieve(
            query=q,
            top_k_seeds=10,
            max_depth=3,
            beam_width=6,
            final_top_k=25,
            mode="hypergraph"
        )

        truncated_context, tokens_used = truncate_context(raw_context, tokenizer, max_tokens)

        # DSPy Evaluation
        ans_result = evaluator.answer(context=truncated_context, question=q).answer
        score = evaluator.evaluator(question=q, correct_answer=a, llm_answer=ans_result).score

        final_score = float(score) if score else 0.0
        print(f"[QA] Scored {final_score} | Cat: {category} | Tokens: {tokens_used}/{max_tokens}")
        return final_score, category

    except Exception as e:
        print(f"Error evaluating question '{q}': {e}")
        return 0.0, category


def evaluate_fact_item(fact_dict: dict, retriever: Retriever, evaluator: FactEvaluator,
                       tokenizer: AutoTokenizer, max_tokens: int) -> tuple[float, str]:
    fact = fact_dict["fact"]

    try:
        raw_context = retriever.retrieve(
            query=fact,
            top_k_seeds=8,
            max_depth=3,
            beam_width=8,
            final_top_k=30
        )

        truncated_context, tokens_used = truncate_context(raw_context, tokenizer, max_tokens)

        score = evaluator(context=truncated_context, correct_answer=fact)
        final_score = float(score) if score else 0.0

        print(f"[FACT] Scored {final_score} | Tokens: {tokens_used}/{max_tokens}")
        return final_score, "FACT_RECALL"

    except Exception as e:
        print(f"Error evaluating fact '{fact}': {e}")
        return 0.0, "FACT_RECALL"


# ==========================================
# 3. Main Runner & Aggregation
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Evaluate Custom NKG QA/Fact Performance")
    parser.add_argument("--graph-path", type=str, required=True, help="Path to your GraphML file")
    parser.add_argument("--data", type=str, required=True, help="Path to QA or Fact JSON dataset")
    parser.add_argument("--type", type=str, choices=["QA", "FACT"], required=True)

    # Fair Comparison Constraints
    parser.add_argument("--max-context-tokens", type=int, default=2000)
    parser.add_argument("--tokenizer-model", type=str, default="Qwen/Qwen3-30B-A3B-Instruct-2507-FP8")
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()

    # 1. Initialize DSPy
    configure_dspy(max_tokens=40000)

    # 2. Setup Tokenizer & Local Retrieval Models
    print(f"Loading Tokenizer: {args.tokenizer_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_model)

    print("Loading Local Embedding Models...")
    #bi_encoder = SentenceTransformer("all-MiniLM-L6-v2")
    bi_encoder = SentenceTransformer("google/embeddinggemma-300m")
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    # 3. Initialize Custom Retriever
    print(f"Loading existing Knowledge Graph from {args.graph_path}...")
    retriever = Retriever(
        retrieval_model=bi_encoder,
        cross_encoder=cross_encoder,
        graph_path=args.graph_path
    )

    # 4. Load Data
    with open(args.data, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 5. Execute Evaluation
    category_scores = defaultdict(lambda: {"total_score": 0.0, "count": 0})
    total_global_score = 0.0
    total_items = 0

    print(f"\nStarting {args.type} Evaluation with {args.workers} workers...")

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = []

        if args.type == "QA":
            evaluator = QAEvaluator()
            items = data.get("qa_pairs", [])
            for pair in items:
                futures.append(executor.submit(
                    evaluate_qa_item, pair, retriever, evaluator, tokenizer, args.max_context_tokens
                ))

        elif args.type == "FACT":
            evaluator = FactEvaluator()
            items = data.get("facts", [])
            for fact_dict in items:
                futures.append(executor.submit(
                    evaluate_fact_item, fact_dict, retriever, evaluator, tokenizer, args.max_context_tokens
                ))

        # Aggregate as they finish
        for future in as_completed(futures):
            score, category = future.result()

            # Global Aggregation
            total_global_score += score
            total_items += 1

            # Categorical Aggregation
            category_scores[category]["total_score"] += score
            category_scores[category]["count"] += 1

    # 6. Final Reporting
    print("\n" + "=" * 50)
    print(f" FINAL {args.type} EVALUATION RESULTS")
    print("=" * 50)

    if total_items > 0:
        global_avg = total_global_score / total_items
        print(f"OVERALL AVERAGE SCORE: {global_avg:.4f}  (Total Items: {total_items})")
        print("-" * 50)

        for cat, stats in sorted(category_scores.items()):
            cat_avg = stats["total_score"] / stats["count"]
            print(f"Category: {cat.ljust(25)} | Avg Score: {cat_avg:.4f} | Count: {stats['count']}")
    else:
        print("No items evaluated.")

    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()