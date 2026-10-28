"""
Fast clustered global fact->fact edge construction: build_index(edge_mode="clustered_fast").

Same pipeline as construct_edges.construct_edges_between_facts_clustered (per round: fact similarity graph ->
hierarchical Leiden -> one bulk LLM call per cluster, with the same checkpoint / resume / replay through
construct_edges._process_clustered_round), with three differences, each a parameter:

1. Similarity graph (always on). The legacy builder calls Graph.get_relevant_seeds_by_id(get_all=True) once per fact:
   4 matrix-vector products over every fact, a full argsort, and N-element Python lists / dict, to keep top_k
   (~62 ms per fact, ~1.9 h at 109k facts). Here the same weighted score
       score(i, j) = w_s cos(sent_i, sent_j) + w_t cos(topic_i, topic_j) + w_e cos(ent_i, ent_j) + w_q cos(Q_i, K_j)
   is one matrix product of concatenated, weight-scaled unit vectors, scored in row blocks on the GPU, with top-k
   (or the round-3 middle band) taken per block (seconds). The candidate rule is the legacy one: rank over all
   facts INCLUDING the fact itself, take positions [start, start + top_k) with start = floor(N * skip_percent)
   (0 for top-k), drop self, drop pairs that already have an edge; edge weight max(0.01, score).

2. Embeddings (embed_loader). Graph.init_fact_embeddings loads the model with SentenceTransformer(name) WITHOUT
   trust_remote_code; for voyageai/voyage-4-nano that silently loads a plain causal Qwen3Model (1024-d) instead of
   the bidirectional embedding model (2048-d) that dedup and retrieval use.
     "remote_code" (default) — SentenceTransformer(name, trust_remote_code=True), the real embedding model
     "legacy"                — SentenceTransformer(name), identical vectors to the legacy path
   The five field matrices are cached in the checkpoint dir (fact_field_embeddings_<loader>.npz, keyed by a hash of
   the model, loader and field texts), so a resume or a re-cluster of the same graph does not re-encode.

3. Leiden levels (leiden_levels, leiden_passes). graspologic.hierarchical_leiden returns one entry per (node, level):
   a node in a level-0 community larger than max_cluster_size appears again at level 1, and so on; only entries with
   is_final_cluster=True form the partition. The legacy code groups every entry, so each fact is sent in its final
   cluster AND in all of its ancestor communities (HotpotQA full: 6,713 clusters, largest 9,635 facts, 3.1 prompts
   per fact).
     "final" (default) — final clusters only: disjoint, each <= max_cluster_size facts
     "all"             — the legacy grouping (every level)
   leiden_passes > 1 repeats Leiden with seeds random_seed, random_seed + 1, ... and adds each pass's final
   clusters (identical clusters kept once), so each fact appears in up to `passes` different <= max_cluster_size
   clusters: bounded-size overlap in place of the legacy nested overlap.

embed_loader="legacy", leiden_levels="all", leiden_passes=1 reproduces the legacy clusters (up to float rounding
at score ties), which is how this module is checked against the legacy builder.
"""
import hashlib
import math
import os
import time
from datetime import datetime

import networkx as nx
import numpy as np
from graspologic.partition import hierarchical_leiden

from nkg.index.construction.checkpointing import ConstructionCheckpoint
from nkg.index.construction.construct_edges import _bulk_fact_edge_constructor, _process_clustered_round
from nkg.models.Graph import Graph

FIELDS = ("sent", "topic", "entity", "answered", "follow_up")


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def fact_field_texts(graph: Graph):
    """The five per-fact texts, formatted exactly as Graph.init_fact_embeddings formats them."""
    fact_ids = list(graph.facts.keys())
    texts = {f: [] for f in FIELDS}
    for fact in graph.facts.values():
        texts["sent"].append(fact.sentence)
        texts["topic"].append(str([x + " " for x in fact.chunk_topics]))
        texts["entity"].append(str([x.role + " " for x in fact.entities]))
        texts["answered"].append(str([x + " " for x in fact.answered_questions]))
        texts["follow_up"].append(str([x + " " for x in fact.follow_up_questions]))
    return fact_ids, texts


def encode_fact_fields(graph: Graph, model_name: str, embed_loader: str = "remote_code", cache_dir: str = None,
                       batch_size: int = 32):
    """(fact_ids, {field: float32 [n_facts, dim]}). Loaded from cache_dir when the cached texts/model match."""
    if embed_loader not in ("remote_code", "legacy"):
        raise ValueError(f"embed_loader must be 'remote_code' or 'legacy', got {embed_loader!r}")
    fact_ids, texts = fact_field_texts(graph)
    h = hashlib.sha1(f"{model_name}|{embed_loader}|{batch_size}".encode())
    for f in FIELDS:
        for t in texts[f]:
            h.update(t.encode("utf-8", "surrogatepass"))
            h.update(b"\x00")
    key = h.hexdigest()
    path = os.path.join(cache_dir, f"fact_field_embeddings_{embed_loader}.npz") if cache_dir else None
    if path and os.path.exists(path):
        try:
            data = np.load(path, allow_pickle=False)
            if str(data["key"]) == key:
                print(f"[{_ts()}] Fact field embeddings: loaded from {path}")
                return fact_ids, {f: data[f] for f in FIELDS}
        except Exception as exc:  # noqa: BLE001 — a bad cache file is just re-encoded
            print(f"[{_ts()}] Fact field embeddings: ignoring unreadable cache {path} ({exc!r})")

    from sentence_transformers import SentenceTransformer
    t0 = time.time()
    if embed_loader == "remote_code":
        model = SentenceTransformer(model_name, trust_remote_code=True)
    else:
        model = SentenceTransformer(model_name)
    emb = {}
    for f in FIELDS:
        emb[f] = np.asarray(model.encode(texts[f], batch_size=batch_size, show_progress_bar=False), dtype=np.float32)
    print(f"[{_ts()}] Fact field embeddings: encoded {len(fact_ids)} facts x {len(FIELDS)} fields "
          f"(dim {emb['sent'].shape[1] if len(fact_ids) else 0}, loader {embed_loader}) in {time.time() - t0:.0f}s")
    del model
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass
    if path:
        os.makedirs(cache_dir, exist_ok=True)
        tmp = path + ".tmp.npz"
        np.savez(tmp, key=np.array(key), **emb)
        os.replace(tmp, path)
    return fact_ids, emb


def _unit(m: np.ndarray) -> np.ndarray:
    """Row-normalised float32 copy; zero rows stay zero (as Graph._unit_fact_embeddings / sklearn)."""
    m = np.asarray(m, dtype=np.float32)
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


def _query_key_fields(weights: dict, questions_mode: str):
    """[(weight, query_field, key_field)] for score(i, j) = sum w * cos(query_field_i, key_field_j)."""
    if questions_mode == "follow_up_to_fact_sentence":
        q_pair = ("follow_up", "sent")       # this fact's follow-ups vs other facts' sentences
    elif questions_mode == "fact_sentence_to_follow_up":
        q_pair = ("sent", "follow_up")       # this fact's sentence vs other facts' follow-ups
    else:
        q_pair = ("follow_up", "answered")   # follow_up_to_answered (legacy default)
    terms = [(weights["sent"], "sent", "sent"), (weights["topic"], "topic", "topic"),
             (weights["entity"], "entity", "entity"), (weights["questions"], *q_pair)]
    return [(float(w), q, k) for w, q, k in terms if w != 0]


def fact_candidates(emb: dict, weights: dict, questions_mode: str, top_k: int, skip_top_percent: float = 0.0,
                    chunk_rows: int = 2048):
    """Per fact (row i), the facts at ranks [start, start + top_k) of the weighted score over ALL facts, self
    included (the legacy get_relevant_seeds_by_id(get_all=True) + sample_results rule), start =
    floor(N * skip_top_percent). Returns (idx int64 [N, k'], scores float32 [N, k']), rows sorted by descending score."""
    terms = _query_key_fields(weights, questions_mode)
    n = len(emb["sent"])
    start = math.floor(n * skip_top_percent) if skip_top_percent > 0 else 0
    width = max(0, min(top_k, n - start))
    if n == 0 or width == 0 or not terms:
        return np.zeros((n, 0), dtype=np.int64), np.zeros((n, 0), dtype=np.float32)
    unit = {f: _unit(emb[f]) for f in {f for _, q, k in terms for f in (q, k)}}
    # one matrix product: [w_1 q_1 | w_2 q_2 | ...] @ [k_1 | k_2 | ...]^T = sum_m w_m q_m k_m^T
    Q = np.concatenate([w * unit[q] for w, q, _ in terms], axis=1)
    K = np.concatenate([unit[k] for _, _, k in terms], axis=1)
    top_idx = np.zeros((n, width), dtype=np.int64)
    top_scores = np.zeros((n, width), dtype=np.float32)
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        torch = None
    if torch is not None:
        prev_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False  # full float32, as the numpy legacy path
        try:
            Qt, Kt = torch.from_numpy(Q).to(device), torch.from_numpy(K).to(device)
            for s in range(0, n, chunk_rows):
                block = Qt[s:s + chunk_rows] @ Kt.T
                vals, idx = torch.topk(block, start + width, dim=1)
                top_idx[s:s + chunk_rows] = idx[:, start:].cpu().numpy()
                top_scores[s:s + chunk_rows] = vals[:, start:].float().cpu().numpy()
                del block, vals, idx
            del Qt, Kt
            if device == "cuda":
                torch.cuda.empty_cache()
        finally:
            torch.backends.cuda.matmul.allow_tf32 = prev_tf32
        return top_idx, top_scores
    for s in range(0, n, chunk_rows):
        block = Q[s:s + chunk_rows] @ K.T
        order = np.argsort(-block, axis=1, kind="stable")[:, start:start + width]
        top_idx[s:s + chunk_rows] = order
        top_scores[s:s + chunk_rows] = np.take_along_axis(block, order, axis=1)
    return top_idx, top_scores


def build_fact_sim_graph(graph: Graph, fact_ids: list, top_idx: np.ndarray, top_scores: np.ndarray) -> nx.Graph:
    """The legacy similarity graph from per-fact candidates: nodes in fact order, edges added in fact order (a pair
    seen from both ends keeps the later weight, as in the legacy loop), self and already-connected pairs skipped."""
    sim_graph = nx.Graph()
    sim_graph.add_nodes_from(fact_ids)
    network = graph.network
    for i, fact_id in enumerate(fact_ids):
        for j, score in zip(top_idx[i].tolist(), top_scores[i].tolist()):
            if j == i:
                continue
            candidate_id = fact_ids[j]
            if network.has_edge(fact_id, candidate_id):
                continue
            sim_graph.add_edge(fact_id, candidate_id, weight=max(0.01, float(score)))
    return sim_graph


def merge_small_clusters(clusters: list, sim_graph: nx.Graph, min_size: int, max_size: int,
                         node_vectors: dict = None) -> list:
    """Greedy merge of clusters below min_size (smallest first) into the cluster they are most strongly connected to:
    connection = summed similarity-graph edge weight between the two clusters; any connection counts, however weak.
    A merge is allowed only if the result has <= max_size facts. A small cluster with no allowed connected partner
    goes to the allowed cluster with the most similar centroid (node_vectors: fact id -> unit vector), if given.
    A merged cluster still below min_size goes back in the queue, so merging continues until every cluster has
    >= min_size facts or no allowed merge is left. Cluster member order: the absorbing cluster's members first."""
    import heapq
    members = {i: list(c) for i, c in enumerate(clusters)}
    where = {n: i for i, c in members.items() for n in c}
    nbr = {i: {} for i in members}
    for u, v, d in sim_graph.edges(data=True):
        a, b = where.get(u), where.get(v)
        if a is None or b is None or a == b:
            continue
        w = float(d.get("weight", 1.0))
        nbr[a][b] = nbr[a].get(b, 0.0) + w
        nbr[b][a] = nbr[b].get(a, 0.0) + w
    cent = None
    if node_vectors is not None:
        cent = {}
        for i, c in members.items():
            vs = [node_vectors[n] for n in c if n in node_vectors]
            cent[i] = (np.sum(vs, axis=0), len(vs)) if vs else (None, 0)
    heap = [(len(c), i) for i, c in members.items() if len(c) < min_size]
    heapq.heapify(heap)
    while heap:
        size, a = heapq.heappop(heap)
        if a not in members or len(members[a]) != size:
            continue  # merged away, or a stale entry
        fits = lambda b: b != a and b in members and size + len(members[b]) <= max_size  # noqa: E731
        cands = [(w, -len(members[b]), b) for b, w in nbr[a].items() if fits(b)]
        if cands:
            b = max(cands)[2]
        elif cent is not None and cent[a][0] is not None:
            va = cent[a][0] / np.linalg.norm(cent[a][0])
            best, b = None, None
            for j in members:
                if fits(j) and cent[j][0] is not None:
                    sim = float(va @ (cent[j][0] / np.linalg.norm(cent[j][0])))
                    if best is None or sim > best:
                        best, b = sim, j
            if b is None:
                continue
        else:
            continue
        # merge a into b
        members[b].extend(members.pop(a))
        for c, w in nbr.pop(a).items():
            if c == b:
                nbr[b].pop(a, None)
                continue
            nbr[c].pop(a, None)
            nbr[b][c] = nbr[b].get(c, 0.0) + w
            nbr[c][b] = nbr[c].get(b, 0.0) + w
        if cent is not None:
            (sa, na), (sb, nb) = cent.pop(a), cent[b]
            cent[b] = ((sb if sa is None else sa if sb is None else sa + sb), na + nb)
        if len(members[b]) < min_size:
            heapq.heappush(heap, (len(members[b]), b))
    return list(members.values())


def leiden_fact_clusters(sim_graph: nx.Graph, max_cluster_size: int, leiden_levels: str = "final",
                         leiden_passes: int = 1, random_seed: int = 42, min_cluster_size: int = 0,
                         node_vectors: dict = None) -> list:
    """Cluster lists (each of >= 2 facts) from hierarchical Leiden; see the module docstring for the options.
    min_cluster_size > 0 (leiden_levels "final" only): each pass's final clusters, singletons included, go through
    merge_small_clusters before the >= 2 filter; 0 (default) = unchanged."""
    if leiden_levels not in ("final", "all"):
        raise ValueError(f"leiden_levels must be 'final' or 'all', got {leiden_levels!r}")
    if min_cluster_size and leiden_levels != "final":
        raise ValueError("min_cluster_size needs leiden_levels='final' (the 'all' grouping is nested, not a partition)")
    if min_cluster_size > max_cluster_size:
        raise ValueError(f"min_cluster_size {min_cluster_size} > max_cluster_size {max_cluster_size}")
    if sim_graph.number_of_edges() == 0:
        return []
    out, seen = [], set()
    for p in range(max(1, leiden_passes)):
        raw = hierarchical_leiden(sim_graph, max_cluster_size=max_cluster_size, random_seed=random_seed + p)
        clusters = {}
        for entry in raw:
            if leiden_levels == "final" and not entry.is_final_cluster:
                continue
            clusters.setdefault(entry.cluster, []).append(entry.node)
        if min_cluster_size > 0:
            clusters = dict(enumerate(merge_small_clusters(list(clusters.values()), sim_graph, min_cluster_size,
                                                           max_cluster_size, node_vectors)))
        for members in clusters.values():
            if len(members) < 2:
                continue
            if leiden_passes > 1:
                key = frozenset(members)
                if key in seen:
                    continue
                seen.add(key)
            out.append(members)
    return out


def construct_edges_between_facts_clustered_fast(graph: Graph, top_k: int = 10, skip_percent: float = 0.33,
                                                 max_cluster_size: int = 20, max_workers: int = 10,
                                                 retrieval_model: str = "all-MiniLM-L6-v2", use_cot: bool = True,
                                                 neutral_description: bool = False,
                                                 use_round1: bool = True, use_round2: bool = True,
                                                 use_round3: bool = True, edge_descriptions: bool = True,
                                                 fact_edge_prompt: str = "default",
                                                 r1_weights: dict = None, r2_weights: dict = None,
                                                 r3_weights: dict = None,
                                                 questions_mode: str = "follow_up_to_answered",
                                                 checkpoint=None,
                                                 embed_loader: str = "remote_code",
                                                 leiden_levels: str = "final", leiden_passes: int = 1,
                                                 min_cluster_size: int = 0):
    """Drop-in for construct_edges_between_facts_clustered (same rounds, weights, prompts, checkpoints and return
    value); see the module docstring for what differs. min_cluster_size > 0: clusters below it are merged
    (merge_small_clusters; strongest similarity-graph connection, else nearest sentence-embedding centroid) while the
    result stays <= max_cluster_size; 0 (default) = the plain Leiden partition."""
    checkpoint = checkpoint if checkpoint is not None else ConstructionCheckpoint(None)
    _r1 = r1_weights or {"sent": 0.0, "topic": 0.0, "entity": 1.0, "questions": 0.0}
    _r2 = r2_weights or {"sent": 1.0, "topic": 1.0, "entity": 1.0, "questions": 1.0}
    _r3 = r3_weights or {"sent": 1.0, "topic": 1.0, "entity": 0.5, "questions": 1.0}
    print(f"[{_ts()}] Clustered-fast fact edges: embed_loader={embed_loader}, leiden_levels={leiden_levels}, "
          f"leiden_passes={leiden_passes}, max_cluster_size={max_cluster_size}, min_cluster_size={min_cluster_size}, "
          f"top_k={top_k}")
    bulk_edge_constructor = _bulk_fact_edge_constructor("global", use_cot, neutral_description, edge_descriptions,
                                                        fact_edge_prompt)

    state = {}  # embeddings are computed lazily: a resume whose rounds all have saved cluster lists never needs them

    def _embeddings():
        if "emb" not in state:
            state["fact_ids"], state["emb"] = encode_fact_fields(graph, retrieval_model, embed_loader,
                                                                 cache_dir=getattr(checkpoint, "dir", None))
        return state["fact_ids"], state["emb"]

    def _sim_graph(weights: dict, skip_top_percent: float = 0.0) -> nx.Graph:
        fact_ids, emb = _embeddings()
        t0 = time.time()
        idx, scores = fact_candidates(emb, weights, questions_mode, top_k, skip_top_percent)
        t1 = time.time()
        sim_graph = build_fact_sim_graph(graph, fact_ids, idx, scores)
        print(f"[{_ts()}] Similarity graph: candidates {t1 - t0:.1f}s, graph {time.time() - t1:.1f}s")
        return sim_graph

    def _cluster(sim_graph: nx.Graph) -> list:
        t0 = time.time()
        node_vectors = None
        if min_cluster_size > 0:  # centroid fallback for small clusters with no allowed connected partner
            fact_ids, emb = _embeddings()
            sent = np.asarray(emb["sent"], dtype=np.float32)
            sent = sent / np.maximum(np.linalg.norm(sent, axis=1, keepdims=True), 1e-12)
            node_vectors = dict(zip(fact_ids, sent))
        clusters = leiden_fact_clusters(sim_graph, max_cluster_size, leiden_levels, leiden_passes,
                                        min_cluster_size=min_cluster_size, node_vectors=node_vectors)
        sizes = [len(c) for c in clusters]
        print(f"[{_ts()}] Leiden ({leiden_levels}, {leiden_passes} pass(es)): {len(clusters)} clusters, "
              f"max {max(sizes) if sizes else 0} facts, min {min(sizes) if sizes else 0}, "
              f"below min_cluster_size {sum(x < min_cluster_size for x in sizes)}, {sum(sizes)} fact-slots, "
              f"{time.time() - t0:.1f}s")
        return clusters

    rounds = [
        (1, use_round1, lambda: _sim_graph(_r1), "global_topk_entity_cluster"),
        (2, use_round2, lambda: _sim_graph(_r2), "global_topk_balanced"),
        (3, use_round3, lambda: _sim_graph(_r3, skip_top_percent=skip_percent), "global_middlek_balanced_cluster"),
    ]
    tokens = {}
    for num, use, builder, sub_type in rounds:
        tokens[f"round{num}"] = _process_clustered_round(
            graph, num, use, sim_graph_builder=builder, max_cluster_size=max_cluster_size,
            bulk_edge_constructor=bulk_edge_constructor, max_workers=max_workers, edge_sub_type=sub_type,
            neutral_description=neutral_description, checkpoint=checkpoint, cluster_fn=_cluster,
        )
    print("Finished constructing clustered-fast global fact edges.")
    return tokens
