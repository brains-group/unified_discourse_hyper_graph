import time
import json
import mlflow
from abc import ABC, abstractmethod
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import AutoTokenizer

# Assuming your DSPy imports and truncation utils are available
from experiments.evaluate_mine import ask_llm_query, score_llm_answer, evaluate_fact
from experiments.utils.chunking import truncate_context


class BaseKGEvaluator(ABC):
    def __init__(self, construction_run_id: str, system_name: str, llm_model: str, max_tokens: int = 2000):
        self.construction_run_id = construction_run_id
        self.system_name = system_name
        self.max_tokens = max_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model)

        # --- Aggregation Trackers ---
        self.metrics = {"qa_total": 0, "fact_total": 0}
        self.qa_scores = []
        self.fact_scores = []
        self.context_token_counts = []
        self.qa_categories = defaultdict(list)  # Tracks scores per category

    @abstractmethod
    def retrieve_context(self, query: str, **kwargs) -> str:
        """CHILD CLASS MUST IMPLEMENT THIS."""
        pass

    # ==========================================
    # TRACED EVALUATION LOGIC
    # ==========================================
    @mlflow.trace(name="Evaluate_Single_QA")
    def _evaluate_single_qa(self, pair: dict, **retrieval_kwargs):
        q = pair["question"]
        a = pair["answer"]
        category = pair.get("category", "UNKNOWN_CATEGORY")

        try:
            # 1. System-Specific Retrieval
            raw_context = self.retrieve_context(q, **retrieval_kwargs)

            # 2. Enforce Token Limits
            truncated_context, tokens_used = truncate_context(raw_context, self.tokenizer, self.max_tokens)

            # 3. DSPy LLM Calls
            llm_answer = ask_llm_query(question=q, context=truncated_context)
            score = score_llm_answer(question=q, correct_answer=a, llm_answer=llm_answer)

            return {"status": "SUCCESS", "score": score, "category": category, "tokens": tokens_used}

        except Exception as e:
            return {"status": "ERROR", "error": str(e), "category": category}

    @mlflow.trace(name="Evaluate_Single_Fact")
    def _evaluate_single_fact(self, fact_dict: dict, **retrieval_kwargs):
        fact = fact_dict["fact"]

        try:
            raw_context = self.retrieve_context(fact, **retrieval_kwargs)
            truncated_context, tokens_used = truncate_context(raw_context, self.tokenizer, self.max_tokens)
            score = evaluate_fact(context=truncated_context, correct_answer=fact)

            return {"status": "SUCCESS", "score": float(score), "tokens": tokens_used}

        except Exception as e:
            return {"status": "ERROR", "error": str(e)}

    # ==========================================
    # MULTI-THREADED EXECUTION
    # ==========================================
    def run_qa_evaluation(self, qa_path: str, max_workers: int = 10, **retrieval_kwargs):
        with open(qa_path, 'r') as f:
            pairs = json.load(f).get("qa_pairs", [])

        self.metrics["qa_total"] = len(pairs)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self._evaluate_single_qa, p, **retrieval_kwargs) for p in pairs]

            for future in as_completed(futures):
                res = future.result()
                if res["status"] == "SUCCESS":
                    self.qa_scores.append(res["score"])
                    self.qa_categories[res["category"]].append(res["score"])
                    self.context_token_counts.append(res["tokens"])

    def run_fact_evaluation(self, fact_path: str, max_workers: int = 10, **retrieval_kwargs):
        with open(fact_path, 'r') as f:
            facts = json.load(f).get("facts", [])

        self.metrics["fact_total"] = len(facts)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self._evaluate_single_fact, f, **retrieval_kwargs) for f in facts]

            for future in as_completed(futures):
                res = future.result()
                if res["status"] == "SUCCESS":
                    self.fact_scores.append(res["score"])
                    self.context_token_counts.append(res["tokens"])

    # ==========================================
    # MLFLOW RECORD CREATION (The Foreign Key Link)
    # ==========================================
    # Inside BaseKGEvaluator
    def log_results_to_mlflow(self, dataset_domain: str, dataset_section: str, document_name: str,
                              retrieval_kwargs: dict):
        """Creates a NEW evaluation run and links it to the construction run."""

        avg_qa = sum(self.qa_scores) / len(self.qa_scores) if self.qa_scores else 0.0
        avg_fact = sum(self.fact_scores) / len(self.fact_scores) if self.fact_scores else 0.0
        avg_tokens = sum(self.context_token_counts) / len(
            self.context_token_counts) if self.context_token_counts else 0.0

        mlflow.set_experiment("KG_Evaluation")

        with mlflow.start_run(run_name=f"EVAL_{self.system_name}_{document_name}"):
            # --- THE FOREIGN KEY & GLOBAL IDENTIFIERS ---
            mlflow.set_tag("construction_run_id", self.construction_run_id)
            mlflow.set_tag("system", self.system_name)
            mlflow.set_tag("dataset_domain", dataset_domain)
            mlflow.set_tag("dataset_section", dataset_section)  # <-- ADDED THIS
            mlflow.set_tag("document_name", document_name)

            # --- LOG PARAMETERS ---
            mlflow.log_param("max_context_tokens", self.max_tokens)
            prefixed_params = {f"search_{k}": v for k, v in retrieval_kwargs.items()}
            mlflow.log_params(prefixed_params)

            # --- LOG METRICS ---
            metrics_to_log = {
                "avg_qa_score": avg_qa,
                "avg_fact_score": avg_fact,
                "avg_context_tokens": avg_tokens,
                "total_qa_tested": self.metrics["qa_total"],
                "total_facts_tested": self.metrics["fact_total"]
            }

            for category, scores in self.qa_categories.items():
                cat_avg = sum(scores) / len(scores) if scores else 0.0
                metrics_to_log[f"cat_score_{category}"] = cat_avg

            mlflow.log_metrics(metrics_to_log)