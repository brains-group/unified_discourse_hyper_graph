import os
import time
import uuid
import shutil
import subprocess
import pandas as pd
import litellm
from pathlib import Path
import textwrap

from experiments.utils.kg_construction_tracker import BenchmarkTracker


def setup_graphrag_workspace(workspace_dir: str, source_file: str, model_name: str, chunk_size: int,
                             chunk_overlap: int):
    """Creates the GraphRAG directory structure and injects the configuration."""
    input_dir = os.path.join(workspace_dir, "input")
    os.makedirs(input_dir, exist_ok=True)

    # Copy the target file into GraphRAG's input folder
    shutil.copy(source_file, os.path.join(input_dir, os.path.basename(source_file)))

    # Updated settings.yaml for GraphRAG v0.3.0+
    # Notice the shift from `llm:` to `models:` and `default_chat_model:`
    # Updated settings.yaml: explicitly bypassing tiktoken inference
    # Ensure this triple-quote block is completely flush to the left margin!
    # Using textwrap to strip IDE indentation
    # Using textwrap to strip IDE indentation
    settings_content = textwrap.dedent(f"""\
            encoding_model: cl100k_base
            skip_workflows: []
            models:
              default_chat_model:
                api_key: EMPTY
                type: chat
                model_provider: openai       # <-- THE FIX
                model: {model_name}
                encoding_model: cl100k_base
                api_base: http://localhost:8000/v1
                tokens_per_minute: 100000 
                requests_per_minute: 1000 
                max_retries: 3
                concurrent_requests: 50
                max_tokens: 10000
              default_embedding_model:
                api_key: EMPTY
                type: embedding              # <-- Matched to new schema
                model_provider: openai       # <-- THE FIX
                model: Qwen/Qwen3-Embedding-4B
                encoding_model: cl100k_base
                api_base: http://localhost:8001/v1
                tokens_per_minute: 100000
                requests_per_minute: 1000
                concurrent_requests: 50
            chunks:
              size: {chunk_size}
              overlap: {chunk_overlap}
              group_by_columns: [id]
            entity_extraction:
              prompt: "prompts/entity_extraction.txt"
              entity_types: [organization,person,geo,event]
              max_gleanings: 1
            summarize_descriptions:
              prompt: "prompts/summarize_descriptions.txt"
              max_length: 500
            claim_extraction:
              enabled: false
            community_reports:
              prompt: "prompts/community_report.txt"
              max_length: 2000
              max_input_length: 8000
            cluster_graph:
              max_cluster_size: 10
            embed_graph:
              enabled: true
            embed_text:
              enabled: true
            """)

    with open(os.path.join(workspace_dir, "settings.yaml"), "w", encoding="utf-8") as f:
        f.write(settings_content)


# def get_latest_graphrag_output_dir(workspace_dir: str) -> str:
#     """Finds the timestamped folder GraphRAG just created, explicitly ignoring lancedb."""
#     output_base = os.path.join(workspace_dir, "output")
#
#     valid_subdirs = []
#     for d in os.listdir(output_base):
#         dir_path = os.path.join(output_base, d)
#         # Only consider directories that physically contain an "artifacts" folder
#         if os.path.isdir(dir_path) and os.path.exists(os.path.join(dir_path, "artifacts")):
#             valid_subdirs.append(dir_path)
#
#     if not valid_subdirs:
#         raise FileNotFoundError(f"Could not find any timestamped artifact folders in {output_base}")
#
#     # Grab the most recent valid timestamp folder
#     latest_subdir = max(valid_subdirs, key=os.path.getmtime)
#     return os.path.join(latest_subdir, "artifacts")

def get_latest_graphrag_output_dir(workspace_dir: str) -> str:
    """In GraphRAG v0.3.0+, outputs are saved directly in the 'output' directory."""
    output_dir = os.path.join(workspace_dir, "output")

    if not os.path.exists(output_dir):
        raise FileNotFoundError(f"Could not find the output directory at {output_dir}")

    return output_dir


def run_graphrag_pipeline(file_path: str, export_directory: str, model: str):
    # Setup Hyperparameters
    hyperparams = {
        "chunk_size": 600,
        "chunk_overlap": 50,
        "max_gleanings": 1,
        "model": model,
    }

    document_name = os.path.basename(file_path)
    run_id = f"run_graphrag_{uuid.uuid4().hex}"
    workspace_dir = f"./graphrag_workspace_{run_id}"

    tracker = BenchmarkTracker(run_id, "GraphRAG", file_path, hyperparams)

    # 1. Create empty workspace and run INIT first!
    print(f"Initializing GraphRAG prompts in {workspace_dir}...")
    os.makedirs(workspace_dir, exist_ok=True)
    subprocess.run(["python", "-m", "graphrag", "init", "--root", workspace_dir], capture_output=True)

    # 2. NOW inject our custom settings and input file
    print(f"Injecting custom configuration...")
    setup_graphrag_workspace(workspace_dir, file_path, model, hyperparams["chunk_size"], hyperparams["chunk_overlap"])

    with open(file_path, 'r', encoding='utf-8') as f:
        full_text = f.read()
    total_document_tokens = litellm.token_counter(model=f"openai/{model}", text=full_text)

    # 3. Run the actual pipeline (Removed the init command from here)
    print("Executing GraphRAG Indexing Pipeline (This may take a while)...")
    process = subprocess.run(["python", "-m", "graphrag", "index", "--root", workspace_dir], capture_output=False)

    if process.returncode != 0:
        print("GraphRAG Pipeline Failed!")
        return None

    # 4. Extract Metrics from Parquet Files
    print("Pipeline complete. Extracting metrics...")
    artifacts_dir = get_latest_graphrag_output_dir(workspace_dir)

    try:
        # Based on the new GraphRAG output structure, read the direct parquet files
        df_entities = pd.read_parquet(os.path.join(artifacts_dir, "entities.parquet"))
        df_edges = pd.read_parquet(os.path.join(artifacts_dir, "relationships.parquet"))

        # GraphRAG v0.3.0+ consolidates outputs, so base/final are the same here
        pre_entities = len(df_entities)
        post_entities = len(df_entities)
        pre_edges = len(df_edges)
        post_edges = len(df_edges)

        # Total Chunks processed by GraphRAG
        df_chunks = pd.read_parquet(os.path.join(artifacts_dir, "text_units.parquet"))
        total_chunks = len(df_chunks)

        # Token Tracking Extraction
        # GraphRAG stores telemetry in the stats.json or within the parquet files.
        tracker.inject_manual_tokens(input_tokens=0, completion_tokens=0, api_calls=0)

    except Exception as e:
        print(f"Error reading GraphRAG artifacts: {e}")
        pre_entities, post_entities, pre_edges, post_edges, total_chunks = 0, 0, 0, 0, 0

    entity_dedup_ratio = (post_entities / pre_entities) if pre_entities > 0 else 1.0
    edge_dedup_ratio = (post_edges / pre_edges) if pre_edges > 0 else 1.0

    graph_metrics = {
        "pre_entities": pre_entities,
        "post_entities": post_entities,
        "entity_dedup_ratio": round(entity_dedup_ratio, 4),
        "pre_edges": pre_edges,
        "post_edges": post_edges,
        "edge_dedup_ratio": round(edge_dedup_ratio, 4)
    }

    # 5. Save Logs and Cleanup
    os.makedirs(export_directory, exist_ok=True)
    # Move the final graph artifacts to your official export directory
    shutil.copytree(artifacts_dir, os.path.join(export_directory, run_id))

    tracker.save_log(export_directory, document_name, total_chunks, total_document_tokens, graph_metrics)

    print(f"GraphRAG run complete. Extracted {post_entities} Entities and {post_edges} Edges.")
    return run_id


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    target_file = "./cleaned_td.txt"
    export_dir = "./experiments/data/graphs/graphrag"
    model_name = os.getenv("MODEL_NAME", "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8")

    run_graphrag_pipeline(target_file, export_dir, model_name)