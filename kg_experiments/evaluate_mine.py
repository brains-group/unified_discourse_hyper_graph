import os
import json
import logging
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any

import dspy
from sentence_transformers import SentenceTransformer, CrossEncoder

# Update these imports based on your new architecture
from nkg.utils.config import configure_dspy
from nkg.utils.general import batch_list
from nkg.retrieval.engine import Retriever
from nkg.index.construction.build_index import build_index_from_directory  # Assuming you have a wrapper for this

# ==========================================
# LOGGING CONFIGURATION
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("KGEvaluator")


# ==========================================
# DSPY SIGNATURES & MODULES
# ==========================================
class EvaluateResponse(dspy.Signature):
    """Determine whether the context contains the information stated in the correct answer. Respond with 1 if yes, 0 if no."""
    context: str = dspy.InputField(desc="The context retrieved from the Knowledge Graph")
    correct_answer: str = dspy.InputField(desc="The correct fact to check for")
    evaluation: int = dspy.OutputField(desc="1 if context contains the correct answer, 0 otherwise")


class AnswerQuestion(dspy.Signature):
    """Use the provided context to answer the question in 1-2 sentences."""
    context: str = dspy.InputField(desc="The context retrieved from the Knowledge Graph")
    question: str = dspy.InputField(desc="The question to answer")
    answer: str = dspy.OutputField(desc="The answer to the question based ONLY on the context")


class EvaluateQA(dspy.Signature):
    """
    Judge whether the LLM's answer to the question is correct based on the ground truth correct_answer.
    Return 1.0 if fully correct, 0.5 if partially correct/missing a detail, and 0.0 if wrong.
    """
    question: str = dspy.InputField(desc="The question that was asked")
    correct_answer: str = dspy.InputField(desc="The ground truth correct answer")
    llm_answer: str = dspy.InputField(desc="The LLM generated answer to judge")
    score: float = dspy.OutputField(desc="Your float score (1.0, 0.5, or 0.0)")


class ResponseEvaluator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.evaluate = dspy.ChainOfThought(EvaluateResponse)

    def forward(self, context: str, correct_answer: str) -> int:
        result = self.evaluate(context=context, correct_answer=correct_answer)
        try:
            return int(result.evaluation)
        except ValueError:
            return 0


class QAEvaluator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.answer = dspy.ChainOfThought(AnswerQuestion)
        self.evaluator = dspy.ChainOfThought(EvaluateQA)

    def forward(self, question: str, context: str, correct_answer: str) -> float:
        ans_result = self.answer(context=context, question=question).answer
        eval_result = self.evaluator(question=question, correct_answer=correct_answer, llm_answer=ans_result).score
        try:
            return float(eval_result)
        except ValueError:
            return 0.0


# ==========================================
# WORKER FUNCTIONS
# ==========================================
def process_fact_batch(batch: List[str], retriever: Retriever) -> float:
    """Processes a batch of facts and returns the sum of their scores."""
    evaluator = ResponseEvaluator()
    batch_score = 0
    for fact in batch:
        context = retriever.retrieve(query=fact, top_k_seeds=8, max_depth=3, beam_width=5, final_top_k=8)
        print("Retrieved fact context")
        score = evaluator(context=context, correct_answer=fact)
        print(f"Scored fact: {score}")
        batch_score += score

    batch_size = len(batch)
    print(f"Fact Recall Batch Complete: {batch_score}/{batch_size} correct")
    return batch_score


def process_qa_batch(batch: List[Dict[str, str]], retriever: Retriever) -> float:
    """Processes a batch of QA pairs and returns the sum of their scores."""
    evaluator = QAEvaluator()
    batch_score = 0.0
    for pair in batch:
        q = pair["question"]
        a = pair["answer"]
        context = retriever.retrieve(query=q, top_k_seeds=8, max_depth=3, beam_width=5, final_top_k=8)
        print("Retrieved qa context")
        score = evaluator(question=q, context=context, correct_answer=a)
        print(f"Scored qa: {score}")
        batch_score += score
    batch_size = len(batch)
    print(f"QA Batch Complete: {batch_score}/{batch_size} correct")
    return batch_score


# ==========================================
# MAIN EVALUATION RUNNERS
# ==========================================
def run_fact_recall_eval(retriever: Retriever, filepath: str, max_workers: int=100) -> Dict[str, Any]:
    logger.info(f"Starting Fact Recall Evaluation using data from {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    facts = data.get("facts", [])
    if not facts:
        logger.warning("No facts found in dataset.")
        return {"recall_score": 0.0, "total_facts": 0}

    batched_facts = batch_list(facts, max_batch_size=3)
    total_score = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_fact_batch, batch, retriever): batch for batch in batched_facts}
        completed = 0

        for future in as_completed(futures):
            try:
                score = future.result()
                total_score += score
                completed += 1
                logger.info(
                    f"[Fact Eval] Batch {completed}/{len(batched_facts)} completed. Running Score: {total_score}")
            except Exception as e:
                logger.error(f"Error processing fact batch: {e}")

    avg_score = total_score / len(facts)
    logger.info(f"Fact Recall Evaluation Complete. Average Score: {avg_score:.4f}")
    return {"recall_score": avg_score, "total_facts": len(facts)}


def run_qa_eval(retriever: Retriever, filepath: str, max_workers: int = 100) -> Dict[str, Any]:
    logger.info(f"Starting QA Evaluation using data from {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    pairs = data.get("pairs", [])
    if not pairs:
        logger.warning("No QA pairs found in dataset.")
        return {"qa_score": 0.0, "total_pairs": 0}

    batched_pairs = batch_list(pairs, max_batch_size=5)
    total_score = 0.0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_qa_batch, batch, retriever): batch for batch in batched_pairs}
        completed = 0

        for future in as_completed(futures):
            try:
                score = future.result()
                total_score += score
                completed += 1
                logger.info(f"[QA Eval] Batch {completed}/{len(batched_pairs)} completed. Running Score: {total_score}")
            except Exception as e:
                logger.error(f"Error processing QA batch: {e}")

    avg_score = total_score / len(pairs)
    logger.info(f"QA Evaluation Complete. Average Score: {avg_score:.4f}")
    return {"qa_score": avg_score, "total_pairs": len(pairs)}


from nkg.utils.eval_utils import generate_all_plans_threaded, batch_embed_plans, batch_cross_encode_and_embed
from nkg.retrieval.traversal import rank_paths_global_precomputed
from nkg.retrieval.planner import generate_query_plan


def run_batched_evaluation(
        eval_type: str,  # "fact" or "qa"
        dataset_filepath: str,
        retriever: Retriever,
        max_workers: int = 100
):
    logger.info(f"--- Starting Batched Evaluation for {eval_type} ---")

    # 1. Load Data into a Query Dictionary format: { q_id : query_string }
    with open(dataset_filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    queries_dict = {}
    ground_truths = {}

    if eval_type == "fact":
        items = data.get("facts", [])
        for i, fact in enumerate(items):
            q_id = f"fact_{i}"
            queries_dict[q_id] = fact
            ground_truths[q_id] = fact
    else:
        items = data.get("pairs", [])
        for i, pair in enumerate(items):
            q_id = f"qa_{i}"
            queries_dict[q_id] = pair["question"]
            ground_truths[q_id] = pair["answer"]

    # PHASE 1: Batched LLM Planning
    all_plans = generate_all_plans_threaded(queries_dict, generate_query_plan, max_workers=50)

    # PHASE 2: Batched GPU Embedding
    plan_embeddings_dict = batch_embed_plans(all_plans, retriever.retrieval_model)

    # PHASE 3: Parallel RAM Graph Traversal
    logger.info("Traversing graph in RAM for all queries concurrently...")
    all_raw_paths_objects = {}
    all_raw_paths_strings = {}

    def _traverse_worker(q_id, plan_embs):
        try:
            paths, strings = retriever.get_raw_paths_precomputed(
                plan_embs=plan_embs, top_k_seeds=8, max_depth=3, beam_width=5
            )
            return q_id, paths, strings
        except Exception as e:
            logger.error(f"Traversal error for {q_id}: {e}")
            return q_id, [], []

    # Use max_workers=os.cpu_count() or a fixed high number since NumPy handles concurrent read-only math efficiently
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(_traverse_worker, q_id, plan_embs): q_id
            for q_id, plan_embs in plan_embeddings_dict.items()
        }

        tmp = 0
        for future in as_completed(futures):
            q_id, paths, strings = future.result()
            all_raw_paths_objects[q_id] = paths
            all_raw_paths_strings[q_id] = strings
            tmp += 1
            print(f"Received paths for {tmp} queries")

    # PHASE 4: Batched GPU Cross-Encoding and Bi-Encoding
    ce_scores_dict, path_embs_dict = batch_cross_encode_and_embed(
        queries_dict, all_raw_paths_strings, retriever.retrieval_model, retriever.cross_encoder
    )

    # PHASE 5: RAM Global MMR & Context Assembly
    logger.info("Applying Global MMR and finalizing contexts...")
    final_contexts = {}
    tmp = 0
    for q_id in queries_dict.keys():
        if q_id not in ce_scores_dict or not ce_scores_dict[q_id]:
            final_contexts[q_id] = ""
            continue

        final_strings, _ = rank_paths_global_precomputed(
            completed_paths=all_raw_paths_objects[q_id],
            assembled_strings=all_raw_paths_strings[q_id],
            ce_scores=ce_scores_dict[q_id],
            path_embeddings=path_embs_dict[q_id],
            final_top_k=8
        )
        final_contexts[q_id] = "\n\n".join(final_strings)
        tmp += 1

        if tmp % 100 == 0:
            print(f"Retrieved {tmp} contexts")

    # PHASE 6: Batched LLM Judging
    logger.info("Executing LLM Evaluators on final contexts...")
    total_score = 0

    if eval_type == "fact":
        evaluator = ResponseEvaluator()

        def _eval_worker(q_id):
            return evaluator(context=final_contexts[q_id], correct_answer=ground_truths[q_id])
    else:
        evaluator = QAEvaluator()

        def _eval_worker(q_id):
            return evaluator(question=queries_dict[q_id], context=final_contexts[q_id],
                             correct_answer=ground_truths[q_id])

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_eval_worker, q_id): q_id for q_id in queries_dict.keys()}
        completed = 0
        for future in as_completed(futures):
            try:
                total_score += future.result()
                completed += 1
                if completed % 10 == 0:
                    logger.info(f"Evaluation Progress: {completed}/{len(queries_dict)}")
            except Exception as e:
                logger.error(f"Evaluation Worker Failed: {e}")

    avg_score = total_score / len(queries_dict)
    logger.info(f"Evaluation Complete! Average Score: {avg_score:.4f}")

    return {"score": avg_score, "total_items": len(queries_dict)}


# ==========================================
# ENTRY POINT & CLI PARSING
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Evaluate Knowledge Graph Retrieval System")

    # Core Parameters
    parser.add_argument("--exp-name", type=str, required=True,
                        help="Name of the experiment (e.g., 'v1_chunking_600_50')")
    parser.add_argument("--out-file", type=str, required=True, help="Path to save the JSON results")
    parser.add_argument("--workers", type=int, default=10, help="Number of concurrent threads")

    # Graph Source (Mutually Exclusive: Provide a pre-built GraphML OR an input directory to build it)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--graph-path", type=str, help="Path to an existing GraphML file to load")
    group.add_argument("--input-dir", type=str, help="Directory of text files to build the graph from scratch")

    # Dataset Paths
    parser.add_argument("--fact-data", type=str, default=None, help="Path to Fact Recall JSON dataset")
    parser.add_argument("--qa-data", type=str, default=None, help="Path to QA JSON dataset")

    args = parser.parse_args()

    # 1. Initialize DSPy
    configure_dspy(max_tokens=35000)
    logger.info(f"Initializing Experiment: {args.exp_name}")

    # 2. Setup Retrieval Models
    logger.info("Loading embedding models...")
    bi_encoder = SentenceTransformer("all-MiniLM-L6-v2")
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    # 3. Initialize Retriever
    if args.graph_path:
        logger.info(f"Loading existing Knowledge Graph from {args.graph_path}")
        retriever = Retriever(retrieval_model=bi_encoder, cross_encoder=cross_encoder, graph_path=args.graph_path)
    else:
        logger.info(f"Building Knowledge Graph from text files in {args.input_dir}")
        # Note: adjust the chunk sizes or parameters via arguments if needed
        built_graph = build_index_from_directory(directory=args.input_dir)
        retriever = Retriever(retrieval_model=bi_encoder, cross_encoder=cross_encoder, graph=built_graph)

    # 4. Run Evaluations
    results = {
        "experiment_name": args.exp_name,
        "graph_source": args.graph_path if args.graph_path else args.input_dir,
        "fact_recall": None,
        "qa_performance": None
    }

    if args.fact_data:
        #results["fact_recall"] = run_fact_recall_eval(retriever, args.fact_data, args.workers)
        results["fact_recall"] = run_batched_evaluation("fact", args.fact_data, retriever, args.workers)

    if args.qa_data:
        #results["qa_performance"] = run_qa_eval(retriever, args.qa_data, args.workers)
        results["qa_performance"] = run_batched_evaluation("qa", args.qa_data, retriever, args.workers)

    if not args.fact_data and not args.qa_data:
        logger.error("No datasets provided for evaluation. Please use --fact-data or --qa-data.")
        return

    # 5. Save Results
    os.makedirs(os.path.dirname(args.out_file) or ".", exist_ok=True)
    with open(args.out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    logger.info(f"Experiment results saved successfully to {args.out_file}")

if __name__ == "__main__":
    main()