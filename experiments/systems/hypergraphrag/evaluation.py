import os
import json
import argparse
import litellm
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import AutoTokenizer

# DSPy and Tracker Imports
from nkg.utils.config import configure_dspy
from experiments.evaluate_mine import QAEvaluator, ResponseEvaluator  # <-- Update this import path
from experiments.utils.evaluation_tracker import EvaluationTracker

# HyperGraphRAG Imports
from hypergraphrag import HyperGraphRAG
from hypergraphrag.base import QueryParam


# ==========================================
# 1. Local vLLM Wrappers for Internal Retrieval
# ==========================================
# HyperGraphRAG needs these to perform its internal graph traversals and community
# summaries during the 'hybrid' or 'global' search phases.

async def local_vllm_complete(prompt: str, **kwargs) -> str:
    model_name = kwargs.get("model_name", "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8")
    response = await litellm.acompletion(
        model=f"openai/{model_name}",
        messages=[{"role": "user", "content": prompt}],
        api_base="http://localhost:8000/v1",
        api_key="EMPTY",
        temperature=0.0,
        max_tokens=kwargs.get("max_tokens", 4000)
    )
    return response.choices[0].message.content


async def local_vllm_embed(texts: list[str], **kwargs) -> list[list[float]]:
    response = await litellm.aembedding(
        model="openai/Qwen/Qwen3-Embedding-4B",
        input=texts,
        api_base="http://localhost:8001/v1",
        api_key="EMPTY"
    )
    return [data["embedding"] for data in response.data]


# Monkey-patch embedding dimension for NanoVectorDB
local_vllm_embed.embedding_dim = 2560


# ==========================================
# 2. Context Truncation Utility
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
# 3. Parallel Processing Logic
# ==========================================
def process_qa_pair(pair: dict, kg: HyperGraphRAG, evaluator: QAEvaluator,
                    tokenizer: AutoTokenizer, max_tokens: int,
                    retrieval_kwargs: dict, tracker: EvaluationTracker):
    q = pair["question"]
    a = pair["answer"]
    category = pair.get("category", "UNKNOWN_CATEGORY")

    try:
        # HyperGraphRAG utilizes a QueryParam dataclass.
        # only_need_context=True forces it to return the raw retrieved chunks/nodes
        # instead of generating a final LLM answer.
        param = QueryParam(**retrieval_kwargs)
        response = kg.query(query=q, param=param)

        # Ensure we are extracting the string context
        raw_context = response if isinstance(response, str) else str(response)

        # Truncate to standardized token length
        truncated_context, tokens_used = truncate_context(raw_context, tokenizer, max_tokens)

        # Execute DSPy Judge
        ans_result = evaluator.answer(context=truncated_context, question=q).answer
        score = evaluator.evaluator(question=q, correct_answer=a, llm_answer=ans_result).score

        final_score = float(score) if score else 0.0

        tracker.add_record(
            question_or_fact=q,
            category=category,
            ground_truth=a,
            llm_answer=ans_result,
            score=final_score,
            retrieved_tokens=tokens_used,
            raw_context=truncated_context
        )
        print(f"[QA] Scored {final_score} | Cat: {category} | Tokens: {tokens_used}/{max_tokens}")
    except Exception as e:
        print(f"Error evaluating question '{q}': {e}")


def process_fact(fact_dict: dict, kg: HyperGraphRAG, evaluator: ResponseEvaluator,
                 tokenizer: AutoTokenizer, max_tokens: int,
                 retrieval_kwargs: dict, tracker: EvaluationTracker):
    fact = fact_dict["fact"]

    try:
        param = QueryParam(**retrieval_kwargs)
        response = kg.query(query=fact, param=param)
        raw_context = response if isinstance(response, str) else str(response)

        truncated_context, tokens_used = truncate_context(raw_context, tokenizer, max_tokens)

        score = evaluator(context=truncated_context, correct_answer=fact)
        final_score = float(score) if score else 0.0

        tracker.add_record(
            question_or_fact=fact,
            category="FACT_RECALL",
            ground_truth=fact,
            llm_answer=str(final_score),
            score=final_score,
            retrieved_tokens=tokens_used,
            raw_context=truncated_context
        )
        print(f"[FACT] Scored {final_score} | Tokens: {tokens_used}/{max_tokens}")
    except Exception as e:
        print(f"Error evaluating fact '{fact}': {e}")


# ==========================================
# 4. Main Runner
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Evaluate HyperGraphRAG QA/Fact Performance")
    parser.add_argument("--kg-run-id", type=str, required=True, help="Run ID from construction phase")
    parser.add_argument("--graph-dir", type=str, required=True, help="Path to the HyperGraphRAG working directory")
    parser.add_argument("--data", type=str, required=True, help="Path to QA or Fact JSON")
    parser.add_argument("--type", type=str, choices=["QA", "FACT"], required=True)
    parser.add_argument("--out-dir", type=str, default="./experiments/results")

    # Standardization arguments
    parser.add_argument("--max-context-tokens", type=int, default=2000)
    parser.add_argument("--tokenizer-model", type=str, default="Qwen/Qwen3-30B-A3B-Instruct-2507-FP8")
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()

    configure_dspy(max_tokens=8000)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_model)

    print(f"Loading existing HyperGraphRAG index from {args.graph_dir}...")
    # Initializing HyperGraphRAG with the exact same working_dir automatically loads it
    kg = HyperGraphRAG(
        working_dir=args.graph_dir,
        llm_model_func=local_vllm_complete,
        llm_model_name=args.tokenizer_model,
        embedding_func=local_vllm_embed,
        embedding_batch_num=16,
        kv_storage="JsonKVStorage",
        vector_storage="NanoVectorDBStorage",
        graph_storage="NetworkXStorage"
    )

    # Note: 'only_need_context=True' stops HyperGraphRAG from wasting GPU cycles
    # generating a final answer, returning the raw text string of the context instead.
    retrieval_hyperparams = {
        "mode": "hybrid",
        "top_k": 10,
        "only_need_context": True
    }

    tracker = EvaluationTracker(
        eval_type=args.type,
        kg_method="HyperGraphRAG",
        kg_run_id=args.kg_run_id,
        dataset_path=args.data,
        graph_path=args.graph_dir,
        context_max_tokens=args.max_context_tokens,
        retrieval_hyperparams=retrieval_hyperparams
    )

    with open(args.data, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Starting parallel evaluation with {args.workers} workers...")

    # ThreadPoolExecutor plays perfectly with HyperGraphRAG's 'always_get_an_event_loop()'
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = []
        if args.type == "QA":
            evaluator = QAEvaluator()
            items = data.get("qa_pairs", [])
            for pair in items:
                futures.append(executor.submit(
                    process_qa_pair, pair, kg, evaluator, tokenizer,
                    args.max_context_tokens, retrieval_hyperparams, tracker
                ))
        elif args.type == "FACT":
            evaluator = ResponseEvaluator()
            items = data.get("facts", [])
            for fact_dict in items:
                futures.append(executor.submit(
                    process_fact, fact_dict, kg, evaluator, tokenizer,
                    args.max_context_tokens, retrieval_hyperparams, tracker
                ))

        for future in as_completed(futures):
            future.result()

    tracker.save_report(args.out_dir)


if __name__ == "__main__":
    main()