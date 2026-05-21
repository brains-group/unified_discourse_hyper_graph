import mlflow
from typing import List, Callable, Dict, Optional
from mlflow.tracking import MlflowClient

class EvaluationOrchestrator:
    def __init__(self, base_dataset_dir: str, construction_experiment_name: str = "KG_Construction"):
        self.base_dataset_dir = base_dataset_dir
        self.construction_experiment_name = construction_experiment_name
        self.system_registry: Dict[str, Callable] = {}

    def register_system(self, system_name: str, sweep_function: Callable):
        """Registers a system and its associated sweep function."""
        self.system_registry[system_name.upper()] = sweep_function
        print(f"Registered system: {system_name.upper()}")

    def _build_mlflow_query(self,
                            systems: Optional[List[str]] = None,
                            run_ids: Optional[List[str]] = None,
                            status: str = "SUCCESS",
                            ignore_evaluated: bool = True) -> str:
        """Constructs the MLflow SQL-like query string based on provided filters."""
        query_parts = []

        if status:
            query_parts.append(f"tags.status = '{status}'")

        if ignore_evaluated:
            # Assumes your sweep script tags the construction run once finished
            query_parts.append("tags.evaluation_status != 'COMPLETED'")

        if systems:
            sys_str = "','".join([s.upper() for s in systems])
            query_parts.append(f"tags.kg_method IN ('{sys_str}')")

        if run_ids:
            # MLflow uses run_id directly in the query
            id_str = "','".join(run_ids)
            query_parts.append(f"run_id IN ('{id_str}')")

        return " and ".join(query_parts)

    def run_evaluations(self,
                        target_systems: Optional[List[str]] = None,
                        target_run_ids: Optional[List[str]] = None,
                        require_status: str = "SUCCESS",
                        skip_already_evaluated: bool = True):
        """
        Queries MLflow for construction runs matching the filters and triggers
        their registered sweep functions.
        """
        mlflow.set_experiment(self.construction_experiment_name)

        # 1. Build Query and Fetch Runs
        query = self._build_mlflow_query(
            systems=target_systems,
            run_ids=target_run_ids,
            status=require_status,
            ignore_evaluated=skip_already_evaluated
        )

        print(f"Executing MLflow Query: {query}")
        df_runs = mlflow.search_runs(filter_string=query)

        if df_runs.empty:
            print("No construction runs found matching the criteria.")
            return

        print(f"Found {len(df_runs)} runs to evaluate.")

        # 2. Iterate and Execute
        for index, row in df_runs.iterrows():
            run_id = row["run_id"]

            # Safely extract tags/params (Pandas returns NaN for missing dict keys)
            system = str(row.get("tags.kg_method", "")).upper()
            doc_name = row.get("params.document_name")
            dataset_domain = row.get("tags.dataset_domain", "UNKNOWN_DOMAIN")
            dataset_section = row.get("tags.dataset_section", "UNKNOWN_SECTION")

            print(f"\n=======================================================")
            print(f"Targeting Run: {run_id} | System: {system} | Doc: {doc_name}")
            print(f"=======================================================")

            # Route to the correct sweep function
            if system not in self.system_registry:
                print(f"[WARNING] System '{system}' is not registered in the orchestrator. Skipping.")
                continue

            sweep_func = self.system_registry[system]

            try:
                # Execute the specific sweep
                sweep_func(
                    construction_run_id=run_id,
                    document_name=doc_name,
                    base_dataset_dir=self.base_dataset_dir,
                    dataset_domain=dataset_domain,
                    dataset_section=dataset_section
                )

                # --- THE FIX ---
                # Safely tag the construction run using the MLflow Client!
                # This edits the run directly in the database without opening a context manager
                # and bypasses the active experiment conflict.
                client = MlflowClient()
                client.set_tag(run_id, "evaluation_status", "COMPLETED")

                print(f"[*] Successfully marked construction run {run_id} as EVALUATION COMPLETED.")

            except Exception as e:
                print(f"[ERROR] Failed during sweep for run {run_id}: {e}")

# run_pipeline.py
from .systems.kggen.evaluator import sweep_kggen_eval_parameters
#from hgrag_eval import sweep_hgrag_parameters

if __name__ == "__main__":
    # 1. Initialize Orchestrator
    orchestrator = EvaluationOrchestrator(
        base_dataset_dir="./evaluation_data",
        construction_experiment_name="KG_Construction"
    )

    # 2. Register your systems
    orchestrator.register_system("KGGEN", sweep_kggen_eval_parameters)
    #orchestrator.register_system("HYPERGRAPHRAG", sweep_hgrag_parameters)

    # --- SCENARIO A: Run everything that hasn't been evaluated yet ---
    # orchestrator.run_evaluations()

    # --- SCENARIO B: Only evaluate KGGEN runs ---
    # orchestrator.run_evaluations(target_systems=["KGGEN"])

    # --- SCENARIO C: I want to re-evaluate two specific runs because I changed my dataset ---
    orchestrator.run_evaluations(
        #target_run_ids=["run_id_abc123", "run_id_xyz987"],
        skip_already_evaluated=False  # Force it to run even if tagged COMPLETED
    )