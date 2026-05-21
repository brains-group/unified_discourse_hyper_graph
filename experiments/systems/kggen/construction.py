import os
import time
import json
import uuid
import shutil
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

import mlflow
import litellm
import dspy

from .wrappers import CustomKGGen
from kg_gen.models import Graph
from nkg.utils.chunking import chunk_text_by_tokens

from litellm.integrations.custom_logger import CustomLogger
import litellm
import threading
from mlflow.tracking import MlflowClient

# ==========================================
# 1. Thread-Safe Global Token Accumulator (MLflow-Proof)
# ==========================================
class GlobalTokenAccumulator(CustomLogger):
    """Silently catches and sums all tokens via LiteLLM's official CustomLogger API."""

    def __init__(self):
        super().__init__()
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.api_calls = 0
        self.lock = threading.Lock()

    # We use the built-in log_success_event instead of a custom callback name
    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        # 1. Safely extract usage, whether response_obj is a dict or an object
        if isinstance(response_obj, dict):
            usage = response_obj.get('usage')
        else:
            usage = getattr(response_obj, 'usage', None)

        if usage:
            with self.lock:
                # 2. Safely extract tokens, whether usage is a dict or an object
                if isinstance(usage, dict):
                    p_tokens = usage.get('prompt_tokens', 0)
                    c_tokens = usage.get('completion_tokens', 0)
                    t_tokens = usage.get('total_tokens', p_tokens + c_tokens)
                else:
                    p_tokens = getattr(usage, 'prompt_tokens', 0)
                    c_tokens = getattr(usage, 'completion_tokens', 0)
                    t_tokens = getattr(usage, 'total_tokens', p_tokens + c_tokens)

                self.prompt_tokens += p_tokens
                self.completion_tokens += c_tokens
                self.total_tokens += t_tokens
                self.api_calls += 1

# ==========================================
# 2. Parallel Processing Logic
# ==========================================
def process_batch(batch_texts, kg):
    """Processes a batch of texts sequentially within a single thread."""
    batch_graphs = []
    for text in batch_texts:
        try:
            graph = kg.generate(input_data=text,
                                context="This is an insurance contract.")
            if graph is not None:
                batch_graphs.append(graph)
        except Exception as e:
            print(f"Error generating graph for a text segment: {e}")
    return batch_graphs


# ==========================================
# 3. Main MLflow Pipeline Runner
# ==========================================
def run_kggen_pipeline(file_path: str,
                             export_directory: str,
                             model: str, retrieval_model: str,
                             dataset_domain: str, dataset_section: str, # <-- NEW PARAMS
                             batch_size: int = 5, max_workers: int = 10,
                             chunk_size: int = 600, chunk_overlap: int = 50,):
    start_time = time.time()

    # 1. FORCE DISABLE CACHE FOR ACCURATE TOKEN TRACKING
    dspy.settings.configure(cache=False)

    hyperparams = {
        "batch_size": batch_size,
        "max_workers": max_workers,
    }

    document_name = os.path.basename(file_path)
    run_id = f"run_kggen_{uuid.uuid4().hex[:8]}"
    workspace_dir = os.path.join(export_directory, run_id)
    os.makedirs(workspace_dir, exist_ok=True)

    # 2. Setup Global Token Tracking
    token_tracker = GlobalTokenAccumulator()
    litellm.callbacks = [token_tracker]
    litellm.success_callback = [token_tracker]

    os.environ["OPENAI_API_BASE"] = "http://localhost:8000/v1"
    os.environ["OPENAI_API_KEY"] = "EMPTY"

    # 3. Start MLflow Run
    # 3. Handle MLflow Experiment Lifecycle
    experiment_name = "KG_Construction"
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        # Experiment does not exist, create it
        mlflow.create_experiment(experiment_name)
    elif experiment.lifecycle_stage == "deleted":
        # Experiment exists but was deleted, restore it
        print(f"Restoring deleted MLflow experiment: {experiment_name}")
        client.restore_experiment(experiment.experiment_id)

    # Now safely set the experiment
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=f"KGGen_{dataset_domain}_{dataset_section}_{document_name}"):
        # 1. TAG THE RUN (Crucial for querying later)
        mlflow.set_tags({
            "dataset_domain": dataset_domain,  # e.g., "INSURANCE_CONTRACTS"
            "dataset_section": dataset_section,  # e.g., "LONG_FORM_CONTRACTS"
            "kg_method": "KGGEN",
            "status": "IN_PROGRESS"
        })

        # 2. Log normal parameters
        mlflow.log_params(hyperparams)
        mlflow.log_param("document_name", document_name)
        mlflow.log_param("chunk_size", chunk_size)
        mlflow.log_param("chunk_overlap", chunk_overlap)
        mlflow.log_param("model", model)
        mlflow.log_param("embedding_model", retrieval_model)
        mlflow.log_param("system", "KGGEN")

        try:
            kg = CustomKGGen(
                model=f"openai/{model}",
                api_key="EMPTY",
                retrieval_model=retrieval_model
            )

            print(f"Reading document: {file_path}")
            with open(file_path, 'r', encoding='utf-8') as f:
                full_text = f.read()

            total_document_tokens = litellm.token_counter(model=f"openai/{model}", text=full_text)
            mlflow.log_metric("total_document_tokens", total_document_tokens)

            print("Chunking text...")
            chunks = chunk_text_by_tokens(
                model=model,
                text=full_text,
                chunk_size=chunk_size,
                overlap=chunk_overlap
            )

            total_chunks = len(chunks)
            mlflow.log_metric("total_chunks", total_chunks)

            batches = [chunks[i:i + batch_size] for i in range(0, len(chunks), batch_size)]
            giant_graph_list = []

            print(f"Starting parallel generation with {len(batches)} batches...")
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_batch = {
                    executor.submit(process_batch, batch, kg): i
                    for i, batch in enumerate(batches)
                }

                for future in as_completed(future_to_batch):
                    batch_index = future_to_batch[future]
                    try:
                        graphs_from_batch = future.result()
                        giant_graph_list.extend(graphs_from_batch)
                        print(f"Batch {batch_index + 1}/{len(batches)} completed. ({len(graphs_from_batch)} graphs)")
                    except Exception as exc:
                        print(f"Batch {batch_index + 1} generated an exception: {exc}")

            # Calculate Pre-Merge Metrics
            pre_entities = 0
            pre_edges = 0
            for sub_graph in giant_graph_list:
                if sub_graph:
                    pre_entities += len(sub_graph.entities)
                    edges = getattr(sub_graph, 'edges',
                                    getattr(sub_graph, 'relations', getattr(sub_graph, 'triples', [])))
                    pre_edges += len(edges)

            combined_graph = None
            if giant_graph_list:
                print("\nAggregating sub-graphs...")
                combined_graph = kg.aggregate(giant_graph_list)

                post_entities = len(combined_graph.entities)
                post_edges_list = getattr(combined_graph, 'edges',
                                          getattr(combined_graph, 'relations', getattr(combined_graph, 'triples', [])))
                post_edges = len(post_edges_list)

                entity_dedup_ratio = (post_entities / pre_entities) if pre_entities > 0 else 1.0
                edge_dedup_ratio = (post_edges / pre_edges) if pre_edges > 0 else 1.0

                # --- LOG GRAPH METRICS TO MLFLOW ---
                mlflow.log_metrics({
                    "pre_entities": pre_entities,
                    "post_entities": post_entities,
                    "entity_dedup_ratio": round(entity_dedup_ratio, 4),
                    "pre_edges": pre_edges,
                    "post_edges": post_edges,
                    "edge_dedup_ratio": round(edge_dedup_ratio, 4)
                })

                # Save Graph JSON locally and push to MLflow Artifacts
                graph_export_path = os.path.join(workspace_dir, "graph.json")
                CustomKGGen.export_graph(combined_graph, graph_export_path)
                mlflow.log_artifact(graph_export_path, artifact_path="knowledge_graph")

            else:
                raise ValueError("No graphs were generated. Pipeline failed.")

            # --- LOG GLOBAL TOKEN USAGE TO MLFLOW ---
            mlflow.log_metrics({
                "llm_prompt_tokens": token_tracker.prompt_tokens,
                "llm_completion_tokens": token_tracker.completion_tokens,
                "llm_total_tokens": token_tracker.total_tokens,
                "llm_api_calls": token_tracker.api_calls,
                "execution_time_sec": round(time.time() - start_time, 2)
            })

            # Clean up callback to avoid memory leaks
            litellm.success_callback = []
            litellm.callbacks = []
            mlflow.set_tag("status", "SUCCESS")
            print(f"\n[SUCCESS] Pipeline complete. View results in MLflow UI.")

            return run_id, workspace_dir

        except Exception as e:

            # 1. Log the failure to MLflow

            mlflow.set_tag("status", "FAILED")

            mlflow.log_param("error_message", str(e))

            # 2. Print it to the terminal so you aren't flying blind!

            print(f"\n[FATAL ERROR] Pipeline failed: {e}")

            traceback.print_exc()

            litellm.success_callback = []
            litellm.callbacks = []
            #token_tracker.cleanup()

            # 3. Return a tuple of two Nones to satisfy the unpacking

            return None, None


# ==========================================
# Execution
# ==========================================
if __name__ == "__main__":
    from dotenv import load_dotenv
    import asyncio

    load_dotenv()
    model_name = os.getenv("MODEL_NAME", "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8")
    embedding_model = os.getenv("EMBEDDING_MODEL", "google/embeddinggemma-300m")

    target_file = "./evaluation_data/long_form_contracts/contracts/td_contract.txt"
    export_dir = "./experiments/data/graphs/kggen"

    # We use asyncio.run because you will eventually call this from your async Orchestrator!
    run_id, graph_dir = run_kggen_pipeline(target_file,
                           export_dir,
                           model=model_name,
                           retrieval_model=embedding_model,
                           dataset_domain="INSURANCE_CONTRACTS",
                           dataset_section="LONG_FORM_CONTRACTS",
                           max_workers=30,
                           batch_size=2,
                           )