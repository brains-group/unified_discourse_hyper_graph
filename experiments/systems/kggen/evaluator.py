from experiments.base_evaluator import *

class KGGenEvaluator(BaseKGEvaluator):
    def __init__(self, construction_run_id, llm_model, max_tokens, kg_instance, node_embeddings, nx_graph):
        super().__init__(construction_run_id, "KGGEN", llm_model, max_tokens)
        self.kg = kg_instance
        self.node_embeddings = node_embeddings
        self.nx_graph = nx_graph

    def retrieve_context(self, query: str, **kwargs) -> str:
        # KGGen returns: top_nodes, context_set, raw_context
        _, _, raw_context = self.kg.retrieve(
            query=query,
            node_embeddings=self.node_embeddings,
            graph=self.nx_graph,
            **kwargs
        )
        return raw_context


import itertools
from .wrappers import CustomKGGen
from experiments.utils.evaluation_tracker import resolve_ground_truth_paths
from dotenv import load_dotenv
import os
def sweep_kggen_eval_parameters(construction_run_id: str, document_name: str, base_dataset_dir: str, dataset_section: str, dataset_domain: str):
    """
    Given a single KG construction run, tests multiple retrieval parameter
    combinations and logs them as separate linked evaluation runs.
    """
    # 1. Resolve Data Paths
    qa_path, fact_path = resolve_ground_truth_paths(
        document_name=document_name,
        base_dataset_dir=base_dataset_dir,
        dataset_section=dataset_section
    )

    # 2. Initialize System (Download Graph from MLflow Artifacts)
    load_dotenv()
    api_base = os.getenv("API_BASE", "http://localhost:8000/v1")
    model_name = os.getenv("MODEL_NAME", "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8")
    emb_model=os.getenv("EMBEDDING_MODEL", "google/embeddinggemma-300m")

    kg = CustomKGGen(
        model=f"openai/{model_name}",
        api_key="EMPTY",
        api_base="http://localhost:8000/v1",
        retrieval_model=emb_model,
    )
    graph_path = mlflow.artifacts.download_artifacts(run_id=construction_run_id,
                                                     artifact_path="knowledge_graph/graph.json")
    graph_obj = kg.from_file(graph_path)
    nx_graph = kg.to_nx(graph_obj)
    node_embeddings, _ = kg.generate_embeddings(graph_obj)

    # 3. Define the Hyperparameter Grid
    # Example: Testing different depths and top_K for KGGen
    search_k_values = [10]
    search_depth_values = [2]
    max_token_limits = [2000]

    combinations = list(itertools.product(search_k_values, search_depth_values, max_token_limits))

    print(
        f"Starting Parameter Sweep: {len(combinations)} evaluation runs mapped to Construction Run {construction_run_id}")

    # 4. Execute the Sweep
    for k, depth, max_tokens in combinations:
        print(f"\n--- Running Eval: K={k}, Depth={depth}, MaxTokens={max_tokens} ---")

        evaluator = KGGenEvaluator(
            construction_run_id=construction_run_id,
            llm_model="Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
            max_tokens=max_tokens,
            kg_instance=kg,
            node_embeddings=node_embeddings,
            nx_graph=nx_graph
        )

        search_params = {"k": k, "depth": depth}

        evaluator.run_qa_evaluation(qa_path, max_workers=10, **search_params)
        evaluator.run_fact_evaluation(fact_path, max_workers=10, **search_params)

        evaluator.log_results_to_mlflow(
            dataset_domain="INSURANCE_CONTRACTS",
            document_name=document_name,
            retrieval_kwargs=search_params,
            dataset_section=dataset_section
        )