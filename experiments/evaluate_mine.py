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
class EvaluateFact(dspy.Signature):
    """You are a strict fact-verification evaluator.

    Your task is to determine whether the **correct_answer** is explicitly or inferably supported by the **context**.

    Rules:
    - Output 1 if the correct_answer is clearly stated, paraphrased, or can be directly inferred from the context.
    - Output 0 if the correct_answer is absent, contradicted, or cannot be inferred from the context.
    - Do NOT use outside knowledge. Base your judgment solely on the provided context.
    - Partial matches or vague relevance are NOT sufficient — the fact must be substantively supported.
    """

    context: str = dspy.InputField(desc="The context retrieved from the Knowledge Graph. This is the only source of truth.")
    correct_answer: str = dspy.InputField(desc="The specific fact or claim to verify against the context.")
    evaluation: int = dspy.OutputField(desc="Binary judgment: 1 if the context substantively supports the correct_answer, 0 if it does not.")

class AnswerQuestion(dspy.Signature):
    """You are a precise question-answering system grounded strictly in retrieved knowledge.

    Your task is to answer the question using ONLY the information provided in the context.

    Rules:
    - Answer in 1-2 concise sentences. Do not over-explain.
    - Base your answer solely on the context. Do NOT use outside knowledge or make assumptions.
    - If the context does not contain enough information to answer, respond with: "The context does not contain enough information to answer this question."
    - Do not fabricate facts, infer beyond what is stated, or hedge unnecessarily.
    - Prefer direct, factual language over vague or speculative phrasing.
    """

    context: str = dspy.InputField(desc="The context retrieved from the Knowledge Graph. Treat this as the sole source of truth.")
    question: str = dspy.InputField(desc="The question to answer. Answer this and only this — do not address tangential points.")
    answer: str = dspy.OutputField(desc="A 1-2 sentence factual answer derived strictly from the context. Do not guess anything.")

class EvaluateQA(dspy.Signature):
    """You are a rigorous but fair answer-quality judge.

    Your task is to score an LLM-generated answer by comparing it against the ground truth correct_answer
    for a given question.

    Scoring rubric:
    - 1.0 — Fully correct: the answer captures all key facts from the correct_answer without contradiction.
             Paraphrasing and different word order are acceptable as long as meaning is preserved.
    - 0.5 — Partially correct: the answer contains the core idea but is missing a meaningful detail,
             is overly vague, or includes one minor inaccuracy alongside correct information.
    - 0.0 — Wrong or irrelevant: the answer contradicts the correct_answer, is entirely off-topic,
             or contains no substantively correct information.

    Rules:
    - Judge based on semantic meaning, not surface wording. Synonyms and paraphrases count as correct.
    - Do NOT penalize for stylistic differences, verbosity, or hedging language unless it obscures correctness.
    - Do NOT award partial credit for answers that happen to mention correct keywords without a coherent answer.
    - If the correct_answer has multiple distinct facts, partial omission of one warrants 0.5, omission of most warrants 0.0.
    - Base your judgment solely on the question and correct_answer provided — do not use outside knowledge.
    """

    question: str = dspy.InputField(
        desc="The original question that was asked."
    )
    correct_answer: str = dspy.InputField(
        desc="The ground truth answer. This is the reference for correctness — treat it as authoritative."
    )
    llm_answer: str = dspy.InputField(
        desc="The LLM-generated answer to evaluate. Compare this semantically against the correct_answer."
    )
    score: float = dspy.OutputField(
        desc="Your score: 1.0 (fully correct), 0.5 (partially correct), or 0.0 (wrong). No other values are valid."
    )


class FactEvaluator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.evaluate = dspy.ChainOfThought(EvaluateFact)

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

def ask_llm_query(question: str, context: str) -> str:
    responder = dspy.ChainOfThought(AnswerQuestion)
    return responder(context=context, question=question).answer

def score_llm_answer(question: str, llm_answer: str, correct_answer: str) -> float:
    evaluator = dspy.ChainOfThought(EvaluateQA)
    eval_result = evaluator(question=question, correct_answer=correct_answer, llm_answer=llm_answer).score
    try:
        return float(eval_result)
    except ValueError:
        return 0.0

def evaluate_fact(context: str, correct_answer: str) -> float:
    evaluator = dspy.ChainOfThought(EvaluateFact)
    result = evaluator(context=context, correct_answer=correct_answer)
    try:
        return int(result.evaluation)
    except ValueError:
        return 0


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
        results["fact_recall"] = run_fact_recall_eval(retriever, args.fact_data, args.workers)

    if args.qa_data:
        results["qa_performance"] = run_qa_eval(retriever, args.qa_data, args.workers)

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