import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple
from sentence_transformers import SentenceTransformer, CrossEncoder

def generate_all_plans_threaded(
        queries_dict: Dict[str, str],
        planner_func,
        max_workers: int = 50
) -> Dict[str, object]:
    """
    PHASE 1: Generates Query Plans for all queries concurrently.
    queries_dict: { "q_id": "The actual query string" }
    """
    print(f"Generating {len(queries_dict)} query plans across {max_workers} threads...")
    plans = {}

    def _worker(q_id, query_str):
        try:
            # Assuming planner_func returns the plan object directly
            plan = planner_func(query_str)
            return q_id, plan
        except Exception as e:
            print(f"Error planning query {q_id}: {e}")
            return q_id, None

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_worker, q_id, q_str): q_id for q_id, q_str in queries_dict.items()}
        for future in as_completed(futures):
            q_id, plan = future.result()
            if plan:
                plans[q_id] = plan

    return plans


def batch_embed_plans(
        plans_dict: Dict[str, object],
        bi_encoder: SentenceTransformer
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    PHASE 2: Flattens all lists from all plans into a single array, encodes in one pass,
    and carefully maps them back into a nested dictionary.
    """
    print("Flattening and batch embedding all query plans...")
    flat_strings = []
    tracking = {}
    current_idx = 0

    fields_to_embed = [
        "rewritten_query", "target_topics", "target_entities",
        "broad_anchors", "target_edge_labels", "target_edge_semantics"
    ]

    # 1. Flatten
    for q_id, plan in plans_dict.items():
        tracking[q_id] = {}
        for field in fields_to_embed:
            # Get the attribute. If it's a single string (like rewritten_query), wrap in list.
            val = getattr(plan, field, [])
            if isinstance(val, str):
                val = [val]
            if not val:  # Handle None or empty lists
                val = []

            num_items = len(val)
            flat_strings.extend(val)
            tracking[q_id][field] = (current_idx, current_idx + num_items)
            current_idx += num_items

    # 2. Encode
    print(f"Encoding {len(flat_strings)} total plan components...")
    all_embs = bi_encoder.encode(flat_strings, show_progress_bar=True)

    # 3. Unflatten
    plan_embeddings = {}
    for q_id in plans_dict:
        plan_embeddings[q_id] = {}
        for field in fields_to_embed:
            start, end = tracking[q_id][field]
            # NumPy slicing safety: If start == end, it naturally returns an empty array np.array([])
            plan_embeddings[q_id][field] = all_embs[start:end]

    return plan_embeddings


def batch_cross_encode_and_embed(
        queries_dict: Dict[str, str],
        paths_dict: Dict[str, List[str]],
        bi_encoder: SentenceTransformer,
        cross_encoder: CrossEncoder
) -> Tuple[Dict[str, List[float]], Dict[str, np.ndarray]]:
    """
    PHASE 4: Takes raw assembled path strings, flattens them, scores via CE,
    embeds via Bi-Encoder, and unflattens.
    """
    print("Batch processing Cross-Encoder and Diversity Embeddings...")
    ce_inputs = []
    bi_inputs = []
    tracking = {}
    current_idx = 0

    # 1. Flatten
    for q_id, path_strings in paths_dict.items():
        num_paths = len(path_strings)
        query_str = queries_dict[q_id]

        for p_str in path_strings:
            ce_inputs.append([query_str, p_str])
            bi_inputs.append(p_str)

        tracking[q_id] = (current_idx, current_idx + num_paths)
        current_idx += num_paths

    if not ce_inputs:
        return {}, {}

    # 2. Process
    print(f"Running Cross-Encoder on {len(ce_inputs)} pairs...")
    ce_scores_flat = cross_encoder.predict(ce_inputs, show_progress_bar=True)

    print(f"Embedding {len(bi_inputs)} paths for diversity MMR...")
    bi_embs_flat = bi_encoder.encode(bi_inputs, show_progress_bar=True)

    # 3. Unflatten
    ce_scores_dict = {}
    path_embs_dict = {}

    for q_id, (start, end) in tracking.items():
        # Cross encoder returns a 1D array of floats, we convert slice to list
        ce_scores_dict[q_id] = list(ce_scores_flat[start:end])
        path_embs_dict[q_id] = bi_embs_flat[start:end]

    return ce_scores_dict, path_embs_dict