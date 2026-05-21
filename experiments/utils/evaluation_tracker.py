import os
import json
import time
import uuid
import threading
from typing import Dict, Any, List
from pymongo import MongoClient

def resolve_ground_truth_paths(document_name: str, base_dataset_dir: str, dataset_section: str):
    """
    Maps a source document to its QA and Fact JSONs,
    routing through the correct dataset_section folder.
    """
    base_name = os.path.splitext(document_name)[0]

    # 1. Normalize the section tag to match the physical lowercase folder names
    section_folder = dataset_section.lower()
    section_dir = os.path.join(base_dataset_dir, section_folder)

    # 2. Construct QA Path (Consistent across both datasets)
    qa_path = os.path.join(section_dir, "qa_pairs", f"qa_{base_name}.json")

    # 3. Construct Fact Path (Handles naming differences between datasets)
    fact_path_standard = os.path.join(section_dir, "facts", f"{base_name}.json")
    fact_path_prefixed = os.path.join(section_dir, "facts", f"facts_{base_name}.json")

    # Fallback logic for fact files
    if os.path.exists(fact_path_standard):
        fact_path = fact_path_standard
    else:
        fact_path = fact_path_prefixed

    # 4. Validate existence to prevent silent failures
    if not os.path.exists(qa_path):
        raise FileNotFoundError(f"Missing QA dataset: {qa_path}")
    if not os.path.exists(fact_path):
        raise FileNotFoundError(f"Missing Fact dataset: {fact_path}")

    return qa_path, fact_path


class EvaluationTracker:
    """Thread-safe tracker that logs comprehensive QA/Fact data to MongoDB."""

    def __init__(self, eval_type: str, kg_method: str, kg_run_id: str, dataset_path: str,
                 graph_path: str, context_max_tokens: int, retrieval_hyperparams: Dict[str, Any],
                 mongo_uri: str = "mongodb://localhost:27017/", db_name: str = "kg_benchmarks"):

        self.eval_type = eval_type.upper()
        self.eval_run_id = f"eval_{self.eval_type.lower()}_{uuid.uuid4().hex[:8]}"
        self.kg_method = kg_method
        self.kg_run_id = kg_run_id
        self.dataset_path = dataset_path
        self.graph_path = graph_path
        self.context_max_tokens = context_max_tokens
        self.retrieval_hyperparams = retrieval_hyperparams

        self.start_time = time.time()
        self.lock = threading.Lock()

        # Connect to MongoDB
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.eval_collection = self.db["evaluation_runs"]

        self.detailed_logs: List[Dict[str, Any]] = []
        self.source_metadata = self._lookup_construction_metadata()

    def _lookup_construction_metadata(self) -> Dict[str, Any]:
        """Looks up the original document info. (You can also move construction logs to Mongo later!)"""
        log_file = "benchmark_logs.jsonl"
        if not os.path.exists(log_file):
            return {"document_name": "UNKNOWN"}

        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    if data.get("run_id") == self.kg_run_id:
                        return data.get("source", {"document_name": "UNKNOWN"})
                except json.JSONDecodeError:
                    continue
        return {"document_name": "UNKNOWN"}

    def add_record(self, question_or_fact: str, category: str, ground_truth: str,
                   llm_answer: str, score: float, retrieved_tokens: int, raw_context: str):
        """Thread-safely appends a detailed evaluation record."""
        record = {
            "query": question_or_fact,
            "category": category,
            "ground_truth_evidence": ground_truth,
            "llm_output": llm_answer,
            "score": score,
            "retrieved_context_tokens": retrieved_tokens,
            "raw_context": raw_context
        }
        with self.lock:
            self.detailed_logs.append(record)

    def _calculate_metrics(self) -> Dict[str, Any]:
        if not self.detailed_logs:
            return {}

        total_score = sum(log["score"] for log in self.detailed_logs)
        total_questions = len(self.detailed_logs)

        categories = {}
        for log in self.detailed_logs:
            cat = log["category"]
            if cat not in categories:
                categories[cat] = {"total_score": 0.0, "count": 0}
            categories[cat]["total_score"] += log["score"]
            categories[cat]["count"] += 1

        category_metrics = {}
        for cat, data in categories.items():
            category_metrics[cat] = {
                "average_score": round(data["total_score"] / data["count"], 4),
                "total_items": data["count"]
            }

        return {
            "overall_average_score": round(total_score / total_questions, 4),
            "total_items_evaluated": total_questions,
            "by_category": category_metrics
        }

    def save_report(self):
        """Compiles the final payload and pushes it directly to MongoDB."""
        end_time = time.time()

        document = {
            "eval_run_id": self.eval_run_id,
            "eval_type": self.eval_type,
            "timestamp_start": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(self.start_time)),
            "timestamp_end": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(end_time)),
            "execution_time_sec": round(end_time - self.start_time, 2),

            # Metadata for easy querying
            "kg_method": self.kg_method,
            "kg_run_id": self.kg_run_id,
            "source_document": self.source_metadata.get("document_name", "UNKNOWN"),
            "context_max_tokens": self.context_max_tokens,

            # Deep params
            "retrieval_hyperparams": self.retrieval_hyperparams,
            "dataset_path": self.dataset_path,
            "graph_path": self.graph_path,

            # Results
            "metrics": self._calculate_metrics(),
            "detailed_logs": self.detailed_logs
        }

        # Insert into MongoDB
        try:
            self.eval_collection.insert_one(document)
            print(f"\n[{self.eval_type} Eval Tracker] Successfully pushed results to MongoDB!")
            print(f"Overall Score: {document['metrics'].get('overall_average_score', 0)}")
        except Exception as e:
            print(f"\n[FATAL] Failed to push to MongoDB: {e}")
            # Fallback to local disk if Mongo crashes
            with open(f"{self.eval_run_id}_fallback_report.json", "w") as f:
                # Remove the mongo _id before saving to json if it was injected
                if "_id" in document:
                    del document["_id"]
                json.dump(document, f)