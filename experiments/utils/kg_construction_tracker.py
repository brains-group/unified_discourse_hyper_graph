import time
import litellm
import threading
import json

class BenchmarkTracker:
    """Catches all LiteLLM calls globally and safely aggregates metrics across threads."""

    def __init__(self, run_id, method, file_path, hyperparams):
        self.run_id = run_id
        self.kg_method = method
        self.file_path = file_path
        self.hyperparameters = hyperparams
        self.start_time = time.time()

        # Locks ensure parallel threads don't overwrite each other's token counts
        self.lock = threading.Lock()
        self.total_input = 0
        self.total_completion = 0
        self.total_calls = 0

    def litellm_callback(self, kwargs, completion_response, start_time, end_time):
        """This fires automatically every time LiteLLM finishes an API call."""
        usage = getattr(completion_response, 'usage', None)
        if usage:
            with self.lock:
                self.total_input += getattr(usage, 'prompt_tokens', 0)
                self.total_completion += getattr(usage, 'completion_tokens', 0)
                self.total_calls += 1

    def save_log(self, export_dir, document_name, total_chunks, total_document_tokens, graph_metrics=None):
        """Dumps the final aggregated run metrics to the JSONL file."""
        end_time = time.time()

        if graph_metrics is None:
            graph_metrics = {}

        log_entry = {
            "run_id": self.run_id,
            "timestamp_start": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(self.start_time)),
            "timestamp_end": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(end_time)),
            "total_execution_time_sec": round(end_time - self.start_time, 2),
            "kg_method": self.kg_method,
            "source": {
                "document_name": document_name,
                "document_path": self.file_path,
                "total_chunks": total_chunks,
                "total_document_tokens": total_document_tokens # <-- Moved here
            },
            "export": {
                "export_directory": export_dir
            },
            "token_metrics": {
                "total_input_tokens": self.total_input,
                "total_completion_tokens": self.total_completion,
                "total_tokens_used": self.total_input + self.total_completion,
                "total_llm_api_calls": self.total_calls
            },
            "graph_metrics": graph_metrics, # Now strictly output metrics
            "hyperparameters": self.hyperparameters
        }

        # Append to benchmark file securely
        log_file_path = "benchmark_logs.jsonl"
        with open(log_file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")

        print(f"\n[Benchmarking] Run logged successfully to {log_file_path}")