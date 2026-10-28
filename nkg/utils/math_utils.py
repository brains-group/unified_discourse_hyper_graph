import numpy as np


def cosine_similarity_single(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """
    Computes the cosine similarity between two single embedding vectors.
    Assumes pre-normalized (unit-length) inputs — uses dot product directly.
    """
    return float(np.dot(np.atleast_1d(emb1).ravel(), np.atleast_1d(emb2).ravel()))


def max_pooled_list_similarity(query_embs: np.ndarray, target_embs: np.ndarray) -> float:
    """
    Computes the S_list similarity between a list of query embeddings and target embeddings.
    Assumes pre-normalized inputs — uses dot product instead of sklearn cosine_similarity.
    """
    if query_embs is None or target_embs is None:
        return 0.0

    if len(query_embs) == 0 or len(target_embs) == 0:
        return 0.0

    sim_matrix = query_embs @ target_embs.T   # (Q, T) — valid for unit-norm vectors
    max_sims = np.max(sim_matrix, axis=1)
    return float(np.mean(max_sims))


def compute_mmr(
        candidate_scores: list[float],
        candidate_embeddings: np.ndarray,
        top_k: int,
        lambda_param: float = 0.5
) -> list[int]:
    """
    Selects top_k items using Maximal Marginal Relevance (MMR).
    Balances relevance (candidate_scores) with diversity (candidate_embeddings).

    Assumes pre-normalized embeddings — uses dot products for all similarity
    computations and a vectorized numpy inner loop (no Python for-loop per candidate).

    lambda_param: 1.0 means pure relevance (standard top-k). 0.0 means pure diversity.
    Returns: A list of indices corresponding to the selected candidates.
    """
    if len(candidate_scores) == 0:
        return []

    scores = np.array(candidate_scores, dtype=np.float32)
    embs = np.atleast_2d(candidate_embeddings).astype(np.float32)

    top_k = min(top_k, len(scores))

    selected_indices = []
    # Boolean mask replaces Python list + O(N) .remove() — flipping a bit is O(1)
    mask = np.ones(len(scores), dtype=bool)

    # Pre-compute the NxN similarity matrix via dot product.
    # Valid because embeddings are pre-normalized (unit vectors → dot = cosine).
    sim_matrix = embs @ embs.T

    for _ in range(top_k):
        unsel = np.where(mask)[0]

        if not selected_indices:
            # First pick: highest raw relevance among all candidates
            best_idx = int(unsel[np.argmax(scores[unsel])])
        else:
            sel = np.array(selected_indices, dtype=np.intp)
            # For every unselected candidate: max similarity to any already-selected item.
            # sim_matrix[np.ix_(unsel, sel)] → shape (|unsel|, |sel|)
            # .max(axis=1)                   → shape (|unsel|,)
            # Entire inner loop replaced by two numpy ops — no Python iteration.
            max_sim_to_selected = sim_matrix[np.ix_(unsel, sel)].max(axis=1)
            mmr_scores = lambda_param * scores[unsel] - (1.0 - lambda_param) * max_sim_to_selected
            best_idx = int(unsel[np.argmax(mmr_scores)])

        selected_indices.append(best_idx)
        mask[best_idx] = False   # O(1) — replaces O(N) list.remove()

    return selected_indices


def normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = np.atleast_2d(x)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, 1e-12, None)

def batch_mean_cos(plan_norm: np.ndarray, cand_norm: np.ndarray) -> np.ndarray:
    if plan_norm is None or cand_norm is None or len(plan_norm) == 0 or len(cand_norm) == 0:
        return np.zeros(len(cand_norm), dtype=np.float32)
    return (plan_norm @ cand_norm.T).mean(axis=0).astype(np.float32)


import math


def sample_results(
        ids: list[str],
        scores: list[float],
        strategy: str = "top_k",
        k: int = 8,
        threshold: float = 0.5,
        skip_top_percent: float = 0.25,
        is_descending: bool = True
) -> tuple[list[str], list[float]]:
    """
    Samples fact IDs based on strict rank/index positioning.

    Args:
        is_descending: Set True if data is sorted Best-to-Worst (Highest score at index 0).
        skip_top_percent: Decimal (0.0 to 1.0) representing the percentage of top items to skip or isolate.
    """
    ids_arr = np.array(ids)
    scores_arr = np.array(scores)

    # 1. Standardize Sort Order (Force Best-to-Worst internally)
    # If the user passes Worst-to-Best data, we reverse it immediately.
    if not is_descending:
        ids_arr = ids_arr[::-1]
        scores_arr = scores_arr[::-1]

    total_items = len(ids_arr)

    # 2. Execute Strategy
    if strategy == "top_k":
        return ids_arr[:k].tolist(), scores_arr[:k].tolist()

    # RANK-BASED: The "Middle K" / "Skip Top 25%" fix
    elif strategy == "skip_percent_top_k":
        # Calculates the exact array index based on list length
        start_idx = math.floor(total_items * skip_top_percent)

        if start_idx >= total_items:
            return [], []

        selected = ids_arr[start_idx: start_idx + k]
        selected_scores = scores_arr[start_idx: start_idx + k]
        return selected.tolist(), selected_scores.tolist()

    # 🎯 VALUE-BASED: Grab items above a strict hard similarity score
    elif strategy == "random_threshold":
        valid_indices = np.where(scores_arr >= threshold)[0]

        if len(valid_indices) == 0:
            return [], []

        sample_size = min(k, len(valid_indices))
        selected_idx = np.random.choice(valid_indices, size=sample_size, replace=False)
        return ids_arr[selected_idx].tolist(), scores_arr[selected_idx].tolist()

    # 🎲 RANK-BASED: Grab random items from the elite top X%
    elif strategy == "random_top_percent":
        pool_size = math.floor(total_items * skip_top_percent)

        if pool_size == 0:
            return [], []

        sample_size = min(k, pool_size)
        selected_idx = np.random.choice(pool_size, size=sample_size, replace=False)
        return ids_arr[selected_idx].tolist(), scores_arr[selected_idx].tolist()

    else:
        raise ValueError(f"Invalid strategy: {strategy}")

def weighted_cosine_topk(embeddings: list, weights: list, k: int, pair_with: list = None, chunk_rows: int = 4096):
    """Top-k neighbours of every row under a weighted sum of cosine similarities, without building the N x N matrix.

    score(i, j) = sum_m weights[m] * cos(embeddings[m][i], pair_with[m][j])   (pair_with defaults to embeddings)
    Self pairs (i == j) are excluded. Rows are scored in chunks (on the GPU when torch + CUDA are available), so
    memory is O(chunk_rows * N) instead of O(N^2). Zero vectors stay zero (cosine 0), as in sklearn.

    Returns (top_idx [N, k'] int64, top_scores [N, k'] float32, off_diag_min, off_diag_max), k' = min(k, N - 1),
    each row sorted by descending score; off_diag_min / off_diag_max are over all i != j pairs (for min-max
    normalisation, which preserves the per-row ranking).
    """
    pair_with = pair_with if pair_with is not None else embeddings
    n = len(embeddings[0])
    kk = max(0, min(k, n - 1))
    if n == 0 or kk == 0:
        return np.zeros((n, 0), dtype=np.int64), np.zeros((n, 0), dtype=np.float32), 0.0, 0.0

    def unit(m):
        m = np.asarray(m, dtype=np.float32)
        norms = np.linalg.norm(m, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return m / norms

    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        torch, device = None, "numpy"

    top_idx = np.zeros((n, kk), dtype=np.int64)
    top_scores = np.zeros((n, kk), dtype=np.float32)
    lo, hi = float("inf"), float("-inf")
    if torch is not None:
        A = [torch.from_numpy(unit(m)).to(device) for m in embeddings]
        B = A if pair_with is embeddings else [torch.from_numpy(unit(m)).to(device) for m in pair_with]
        for s in range(0, n, chunk_rows):
            e = min(n, s + chunk_rows)
            block = sum(w * (a[s:e] @ b.T) for w, a, b in zip(weights, A, B))
            rows = torch.arange(e - s, device=block.device)
            block[rows, rows + s] = float("-inf")  # exclude self from top-k and from max
            hi = max(hi, float(block.max()))
            block[rows, rows + s] = float("inf")   # exclude self from min
            lo = min(lo, float(block.min()))
            block[rows, rows + s] = float("-inf")
            vals, idx = torch.topk(block, kk, dim=1)
            top_idx[s:e] = idx.cpu().numpy()
            top_scores[s:e] = vals.float().cpu().numpy()
            del block
        return top_idx, top_scores, lo, hi
    A = [unit(m) for m in embeddings]
    B = A if pair_with is embeddings else [unit(m) for m in pair_with]
    for s in range(0, n, chunk_rows):
        e = min(n, s + chunk_rows)
        block = sum(w * (a[s:e] @ b.T) for w, a, b in zip(weights, A, B))
        rows = np.arange(e - s)
        block[rows, rows + s] = -np.inf
        hi = max(hi, float(block.max()))
        block[rows, rows + s] = np.inf
        lo = min(lo, float(block.min()))
        block[rows, rows + s] = -np.inf
        part = np.argpartition(-block, kk - 1, axis=1)[:, :kk]
        order = np.argsort(-np.take_along_axis(block, part, axis=1), axis=1)
        top_idx[s:e] = np.take_along_axis(part, order, axis=1)
        top_scores[s:e] = np.take_along_axis(block, top_idx[s:e], axis=1)
    return top_idx, top_scores, lo, hi
