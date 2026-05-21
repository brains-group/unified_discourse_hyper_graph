import os
import json
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import AutoTokenizer

from experiments.utils.chunking import truncate_context
from nkg.utils.config import configure_dspy
from experiments.evaluate_mine import ask_llm_query, score_llm_answer, evaluate_fact, QAEvaluator, FactEvaluator
from experiments.utils.evaluation_tracker import EvaluationTracker

import mlflow
from experiments.systems.kggen.wrappers import CustomKGGen

@mlflow.trace(name="Evaluate_Single_QA")
def evaluate_qa_pair(pair: dict, context: str, max_tokens: int, tokenizer: AutoTokenizer):
    q = pair["question"]
    a = pair["answer"]
    category = pair.get("category", "UNKNOWN_CATEGORY")
    llm_answer = ""
    score = 0
    try:
        # apply max tokens limit
        truncated_context, tokens_used = truncate_context(context, tokenizer, max_tokens)

        # get answer and score
        llm_answer = ask_llm_query(question=q, context=truncated_context)
        score = score_llm_answer(question=q, correct_answer=a, llm_answer=llm_answer)

        return {
            "question": q,
            "correct_answer": a,
            "category": category,
            "llm_answer": llm_answer,
            "score": score,
            "max_tokens": max_tokens,
            "tokenizer": tokenizer.name_or_path,
            "context_tokens": tokens_used,
            "status": "SUCCESS"
        }
    except Exception as e:
        return {
            "question": q,
            "correct_answer": a,
            "category": category,
            "llm_answer": llm_answer,
            "score": score,
            "max_tokens": max_tokens,
            "tokenizer": tokenizer.name_or_path,
            "context_tokens": tokens_used,
            "status": "ERROR"
        }

# --- QA PROCESSING ---
def process_qa_pair(pair: dict, kg, node_embeddings: dict, nx_graph, tokenizer: AutoTokenizer, max_tokens: int,
                    retrieval_kwargs: dict):
    q = pair["question"]
    a = pair["answer"]
    category = pair.get("category", "UNKNOWN_CATEGORY")

    try:
        # KGGen returns: top_nodes, context_set, context_text
        _, _, raw_context = kg.retrieve(
            query=q,
            node_embeddings=node_embeddings,
            graph=nx_graph,
            **retrieval_kwargs
        )

        result = evaluate_qa_pair(pair=pair,
                                  context=raw_context,
                                  max_tokens=max_tokens,
                                  tokenizer=tokenizer)

        if result["status"] == "SUCCESS":
            print(f"[QA] Scored {result['score']} | Cat: {category} | Tokens: {result['context_tokens']}/{max_tokens}")
        elif result["status"] == "ERROR":
            print(f"Error evaluating QA")
    except Exception as e:
        print(f"Error evaluating question '{q}': {e}")


# --- FACT PROCESSING ---
@mlflow.trace(name="Evaluate_Single_Fact")
def evaluate_single_fact(fact_dict: dict, context: str, max_tokens: int, tokenizer: AutoTokenizer):
    fact = fact_dict["fact"]
    score = 0.0
    tokens_used = 0

    try:
        # apply max tokens limit
        truncated_context, tokens_used = truncate_context(context, tokenizer, max_tokens)

        # get score using the provided evaluate_fact function
        score = evaluate_fact(context=truncated_context, correct_answer=fact)

        return {
            "fact": fact,
            "category": "FACT_RECALL",  # Hardcoded since facts don't have categories in your JSON
            "score": float(score),
            "max_tokens": max_tokens,
            "tokenizer": tokenizer.name_or_path,
            "context_tokens": tokens_used,
            "status": "SUCCESS"
        }
    except Exception as e:
        return {
            "fact": fact,
            "category": "FACT_RECALL",
            "score": float(score),
            "max_tokens": max_tokens,
            "tokenizer": tokenizer.name_or_path,
            "context_tokens": tokens_used,
            "status": "ERROR"
        }


# --- FACT PROCESSING ---
def process_fact(fact_dict: dict, kg, node_embeddings: dict, nx_graph, tokenizer: AutoTokenizer, max_tokens: int,
                 retrieval_kwargs: dict):
    fact = fact_dict["fact"]

    try:
        # KGGen returns: top_nodes, context_set, context_text
        _, _, raw_context = kg.retrieve(
            query=fact,  # The query is the fact itself
            node_embeddings=node_embeddings,
            graph=nx_graph,
            **retrieval_kwargs
        )

        result = evaluate_single_fact(fact_dict=fact_dict,
                                      context=raw_context,
                                      max_tokens=max_tokens,
                                      tokenizer=tokenizer)

        if result["status"] == "SUCCESS":
            print(f"[FACT] Scored {result['score']} | Tokens: {result['context_tokens']}/{max_tokens}")
        elif result["status"] == "ERROR":
            print(f"Error evaluating Fact")

    except Exception as e:
        print(f"Error retrieving context for fact '{fact}': {e}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate KGGen QA/Fact Performance")
    parser.add_argument("--kg-run-id", type=str, required=True)
    parser.add_argument("--graph-path", type=str, required=True, help="Path to graph.json")
    parser.add_argument("--data", type=str, required=True, help="Path to QA or Fact JSON")
    parser.add_argument("--type", type=str, choices=["QA", "FACT"], required=True)
    parser.add_argument("--out-dir", type=str, default="./experiments/results")

    parser.add_argument("--max-context-tokens", type=int, default=2000)
    parser.add_argument("--llm-model", type=str, default="Qwen/Qwen3-30B-A3B-Instruct-2507-FP8")
    parser.add_argument("--embed-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()

    configure_dspy(max_tokens=8000)
    tokenizer = AutoTokenizer.from_pretrained(args.llm_model)


    print("Initializing KGGen Engine...")
    # Make sure to pass the retrieval model here so SentenceTransformer loads it
    kg = CustomKGGen(
        model=f"openai/{args.llm_model}",
        api_key="EMPTY",
        api_base="http://localhost:8000/v1",
        retrieval_model=args.embed_model
    )

    print(f"Loading Graph from {args.graph_path}...")
    graph_obj = kg.from_file(args.graph_path)
    nx_graph = kg.to_nx(graph_obj)

    print("Pre-computing Graph Node Embeddings for Retrieval...")
    node_embeddings, _ = kg.generate_embeddings(graph_obj)

    # KGGen specifically uses 'k' for top nodes. (Depth is hardcoded to 2 in their source).
    retrieval_hyperparams = {"k": 20, "depth": 3}

    tracker = EvaluationTracker(
        eval_type=args.type,
        kg_method="KGGen",
        kg_run_id=args.kg_run_id,
        dataset_path=args.data,
        graph_path=args.graph_path,
        context_max_tokens=args.max_context_tokens,
        retrieval_hyperparams=retrieval_hyperparams
    )

    with open(args.data, "r", encoding="utf-8") as f:
        data = json.load(f)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = []
        if args.type == "QA":
            evaluator = QAEvaluator()
            items = data.get("qa_pairs", [])
            for pair in items:
                futures.append(executor.submit(
                    process_qa_pair, pair, kg, node_embeddings, nx_graph,
                    evaluator, tokenizer, args.max_context_tokens,
                    retrieval_hyperparams, tracker
                ))
        elif args.type == "FACT":
            evaluator = FactEvaluator()
            items = data.get("facts", [])
            for fact_dict in items:
                futures.append(executor.submit(
                    process_fact, fact_dict, kg, node_embeddings, nx_graph,
                    evaluator, tokenizer, args.max_context_tokens,
                    retrieval_hyperparams, tracker
                ))

        for future in as_completed(futures):
            future.result()

    tracker.save_report(args.out_dir)


if __name__ == "__main__":
    main()