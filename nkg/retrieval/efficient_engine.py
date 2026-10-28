"""
EfficientRetriever: the old-engine Retriever (nkg/retrieval/engine.py) with the same retrieval, at a fraction of the
memory and start-up time. Retrieval logic (query plan, seed scores, beam search, cross-encoder ranking, formatting) is
the parent's; only how embeddings are stored and how MMR is computed differ:

1. MMR without the N x N matrix (mmr_select). compute_mmr builds candidates x candidates similarities up front; seed
   selection runs MMR over EVERY entity and fact, so on the full HotpotQA graph (156k candidates) that is a ~97 GB
   matrix per question. mmr_select keeps each candidate's running max similarity to the picks so far, one
   matrix-vector product per pick: the same picks (up to float rounding), O(N) memory. Used for seed selection, the
   beam-search steps and the final path ranking.
2. One vector per distinct string. Every text the retriever embeds (entity names / roles / anchors, fact sentences /
   answered questions / chunk topics, fact->fact edge descriptions and labels, entity<->fact edge labels) is encoded
   once per distinct string; fields and edges hold row indices into that one matrix. The parent embeds every edge's
   text separately (label-only graphs: each fact->fact edge twice, description == label), ~1.37M vectors on the full
   HotpotQA graph for ~0.1M distinct strings.
3. On-disk embedding cache (embedding_cache_path): the distinct-string matrix is saved once and reloaded by every
   later process on the same graph (each view of an evaluation runs in its own process), keyed by the model name and
   the strings themselves, so a changed graph or model never reuses stale vectors.
4. Vectorised seed scoring: the per-entity / per-fact Python loops (and the per-question copy of every candidate's
   diversity vector) become a few matrix products over shared, read-only matrices; ragged fields (anchors, chunk
   topics) are max-reduced per node with np.maximum.reduceat. Scores are combined in float64 in the parent's order of
   operations, so they equal the parent's Python-float scores up to float32 dot-product rounding.

Empty strings get a zero vector (cosine 0), as in the parent. get_seeds_precomputed is supported; the per-node dicts
entity_embs / fact_embs are not built (nothing in the retrieval path reads them).
"""
import hashlib
import os
import threading
import time
import types

import numpy as np

from nkg.retrieval import traversal as _traversal
from nkg.retrieval.engine import Retriever
from nkg.utils.math_utils import normalize_rows


def mmr_select(candidate_scores, candidate_embeddings=None, top_k: int = 10, lambda_param: float = 0.5,
               sim_to=None) -> list:
    """compute_mmr's selection without the candidates x candidates matrix.

    Either candidate_embeddings ([N, D], unit-norm rows) or sim_to (callable(i) -> similarities [N] of every
    candidate to candidate i) supplies the similarities. Same rule as compute_mmr: first pick = highest score; then
    argmax over unpicked of lambda * score - (1 - lambda) * max similarity to the picks."""
    scores = np.array(candidate_scores, dtype=np.float32)
    n = len(scores)
    if n == 0:
        return []
    if sim_to is None:
        embs = np.atleast_2d(np.asarray(candidate_embeddings, dtype=np.float32))
        sim_to = lambda i: embs @ embs[i]  # noqa: E731
    top_k = min(top_k, n)
    mask = np.ones(n, dtype=bool)
    max_sim = None
    selected = []
    for _ in range(top_k):
        unsel = np.where(mask)[0]
        if not selected:
            best = int(unsel[np.argmax(scores[unsel])])
        else:
            mmr_scores = lambda_param * scores[unsel] - (1.0 - lambda_param) * max_sim[unsel]
            best = int(unsel[np.argmax(mmr_scores)])
        selected.append(best)
        mask[best] = False
        if len(selected) < top_k:
            sims = np.asarray(sim_to(best), dtype=np.float32)
            max_sim = sims if max_sim is None else np.maximum(max_sim, sims)
    return selected


def _with_mmr(fn):
    """A copy of a traversal function whose compute_mmr is mmr_select (same code, globals copied: the original
    function and every other user of traversal are untouched)."""
    g = dict(fn.__globals__)
    g["compute_mmr"] = mmr_select
    return types.FunctionType(fn.__code__, g, fn.__name__, fn.__defaults__, fn.__closure__)


_expand_paths_batched = _with_mmr(_traversal.expand_paths_batched)
_rank_paths_global = _with_mmr(_traversal.rank_paths_global)
_expand_paths_precomputed = _with_mmr(_traversal.expand_paths_precomputed)


def _segment_max_mean(sims: np.ndarray, flat_idx: np.ndarray, starts: np.ndarray) -> np.ndarray:
    """sims [Q, U]; flat_idx = the concatenated rows of each node's ragged list (nodes with an empty list excluded),
    starts = each node's first position in flat_idx. Per node: mean over the Q query vectors of the max over its list
    (the parent's (Q x L).max(axis=1).mean()), float32."""
    g = sims[:, flat_idx]                                  # [Q, total]
    mx = np.maximum.reduceat(g, starts, axis=1)            # [Q, n_nodes]
    return np.ascontiguousarray(mx.T).mean(axis=1)         # per node, like the parent's 1-D .mean()


class EfficientRetriever(Retriever):
    """Retriever (old engine) with the memory / speed changes in the module docstring. Same constructor plus
    embedding_cache_path (None = no cache) and encode_batch_size."""

    def __init__(self, *args, embedding_cache_path: str = None, model_name: str = None, encode_batch_size: int = 32,
                 **kwargs):
        self._embedding_cache_path = embedding_cache_path
        self._model_name = model_name or ""
        self._encode_batch_size = encode_batch_size
        self._lazy_lock = threading.Lock()
        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------------ embeddings
    def _initialize_embeddings(self, verbose=False):
        g = self.graph
        self.all_entity_ids = list(g.entities.keys())
        self.all_fact_ids = list(g.facts.keys())

        # every text, in a deterministic order, per field (lists for ragged fields)
        ent_name = [e.name for e in g.entities.values()]
        ent_role = [e.role for e in g.entities.values()]
        ent_anchors = [list(e.relational_anchors or []) for e in g.entities.values()]
        fact_sent = [f.sentence for f in g.facts.values()]
        fact_q = [" | ".join(f.answered_questions) if f.answered_questions else None for f in g.facts.values()]
        fact_topics = [list(f.chunk_topics or []) for f in g.facts.values()]
        ff_edges, ff_descs, ff_labels, ef_edges, ef_labels, fe_edges, fe_labels = [], [], [], [], [], [], []
        for u, v, data in g.network.edges(data=True):
            if u in g.facts and v in g.facts:
                ff_edges.append((u, v))
                ff_descs.append(data.get("description", "") or "unlabeled")
                ff_labels.append(data.get("label", "") or "unlabeled")
            elif data.get("edge_type") == "entity_fact":
                ef_edges.append((u, v))
                ef_labels.append(data.get("label", "") or "unlabeled")
            elif data.get("edge_type") == "fact_entity":
                fe_edges.append((u, v))
                fe_labels.append(data.get("label", "") or "unlabeled")

        def _norm(s):
            return s if (s and s.strip()) else ""   # empty / whitespace-only -> the zero row

        strings = {""}
        for coll in (ent_name, ent_role, fact_sent, ff_descs, ff_labels, ef_labels, fe_labels):
            strings.update(_norm(s) for s in coll)
        for lists in (ent_anchors, fact_topics):
            for lst in lists:
                strings.update(_norm(s) for s in lst)
        strings.update(_norm(s) for s in fact_q if s is not None)
        uniq = sorted(strings)                      # "" sorts first -> row 0
        row = {s: i for i, s in enumerate(uniq)}
        self.U = self._load_or_encode(uniq)         # [n_strings, D] unit rows, row 0 = zeros
        self.emb_dim = self.U.shape[1]
        n_total = (len(ent_name) + len(ent_role) + sum(map(len, ent_anchors)) + len(fact_sent)
                   + sum(s is not None for s in fact_q) + sum(map(len, fact_topics))
                   + 2 * len(ff_edges) + len(ef_edges) + len(fe_edges))
        print(f"[EfficientRetriever] {len(uniq)} distinct strings for {n_total} embedded fields/edges "
              f"({self.U.nbytes / 1e9:.2f} GB)", flush=True)

        idx = lambda coll: np.fromiter((row[_norm(s)] for s in coll), dtype=np.int64, count=len(coll))  # noqa: E731

        def ragged(lists):
            """(flat row indices, starts of non-empty lists, mask of nodes with a non-empty list)"""
            lens = np.fromiter((len(x) for x in lists), dtype=np.int64, count=len(lists))
            flat = np.fromiter((row[_norm(s)] for lst in lists for s in lst), dtype=np.int64, count=int(lens.sum()))
            offsets = np.concatenate([[0], np.cumsum(lens)[:-1]]) if len(lens) else np.zeros(0, dtype=np.int64)
            has = lens > 0
            return flat, offsets[has].astype(np.int64), has

        # matrices the seed scores read every query (shared, read-only)
        self.ent_name_rows, self.ent_role_rows, self.fact_sent_rows = idx(ent_name), idx(ent_role), idx(fact_sent)
        self.entity_name_mat = self.U[self.ent_name_rows]
        self.entity_role_mat = self.U[self.ent_role_rows]
        self.fact_sent_mat = self.U[self.fact_sent_rows]
        self.anchor_flat, self.anchor_starts, self.ent_has_anchors = ragged(ent_anchors)
        self.topic_flat, self.topic_starts, self.fact_has_topics = ragged(fact_topics)
        self.fact_has_questions = np.array([q is not None for q in fact_q], dtype=bool)
        self.fact_q_rows = np.array([row[_norm(q)] if q is not None else 0 for q in fact_q], dtype=np.int64)
        # ragged-field sub-matrices (distinct rows only), built on first use
        self._sub = {}
        self._ragged_rows = {"anchors": self.anchor_flat, "topics": self.topic_flat}

        # edges: (1, D) views into U, as the parent's dicts (the traversal code reads engine.ff_label_embs[e][0])
        self._edge_rows = (ff_edges, idx(ff_descs), idx(ff_labels), ef_edges, idx(ef_labels), fe_edges, idx(fe_labels))

    def _load_or_encode(self, uniq: list) -> np.ndarray:
        h = hashlib.sha1(f"{self._model_name}|batch{self._encode_batch_size}".encode())
        for s in uniq:
            h.update(s.encode("utf-8", "surrogatepass"))
            h.update(b"\x00")
        key = h.hexdigest()
        path = self._embedding_cache_path
        if path and os.path.exists(path):
            try:
                with np.load(path, allow_pickle=False) as data:
                    if str(data["key"]) == key:
                        U = data["U"]
                        print(f"[EfficientRetriever] embeddings loaded from {path}", flush=True)
                        return U
                print(f"[EfficientRetriever] {path} is for another graph/model; re-encoding", flush=True)
            except Exception as exc:  # noqa: BLE001 — unreadable cache: re-encode
                print(f"[EfficientRetriever] ignoring unreadable cache {path} ({exc!r})", flush=True)
        t0 = time.time()
        enc = self.retrieval_model.encode(uniq[1:], batch_size=self._encode_batch_size, show_progress_bar=True)
        U = np.zeros((len(uniq), enc.shape[1]), dtype=np.float32)
        U[1:] = enc
        del enc
        U = normalize_rows(U)
        print(f"[EfficientRetriever] encoded {len(uniq) - 1} strings in {time.time() - t0:.0f}s", flush=True)
        if path:
            tmp = path + ".tmp.npz"
            np.savez(tmp, key=np.array(key), U=U)
            os.replace(tmp, path)
            print(f"[EfficientRetriever] embeddings cached -> {path}", flush=True)
        return U

    def _initialize_edge_embeddings(self, verbose=False):
        ff_edges, ff_d, ff_l, ef_edges, ef_l, fe_edges, fe_l = self._edge_rows
        U = self.U
        for i, e in enumerate(ff_edges):
            self.ff_desc_embs[e] = U[ff_d[i]:ff_d[i] + 1]
            self.ff_label_embs[e] = U[ff_l[i]:ff_l[i] + 1]
        for i, e in enumerate(ef_edges):
            self.ef_label_embs[e] = U[ef_l[i]:ef_l[i] + 1]
        for i, e in enumerate(fe_edges):
            self.fe_label_embs[e] = U[fe_l[i]:fe_l[i] + 1]
        del self._edge_rows

    def _ragged_sub(self, name: str):
        """(distinct-row sub-matrix, positions of the flat list in it) for a ragged field; built once."""
        if name not in self._sub:
            with self._lazy_lock:
                if name not in self._sub:
                    rows = self._ragged_rows[name]
                    uniq_rows, inv = np.unique(rows, return_inverse=True)
                    self._sub[name] = (self.U[uniq_rows], inv.astype(np.int64))
        return self._sub[name]

    # ------------------------------------------------------------------ seeds
    def _seed_candidates(self, rewrite, targets, broad, topics, use_entities: bool, use_facts: bool,
                         entity_diversity: str):
        """(candidate ids, float64 scores, list of (diversity matrix, row offset)) in the parent's candidate order
        (entities, then facts)."""
        ids, parts, div = [], [], []
        if use_entities and len(self.all_entity_ids):
            n = len(self.all_entity_ids)
            name_v = ((targets @ self.entity_name_mat.T).max(axis=0).astype(np.float32)
                      if len(targets) else np.zeros(n, dtype=np.float32)).astype(np.float64)
            role_v = ((broad @ self.entity_role_mat.T).max(axis=0).astype(np.float32)
                      if len(broad) else np.zeros(n, dtype=np.float32)).astype(np.float64)
            w = self.entity_weights
            redistribute = w["anchors"] / 2
            score = (w["name"] + redistribute) * name_v + (w["role"] + redistribute) * role_v
            if len(targets) and self.ent_has_anchors.any() and w["anchors"] != 0:
                # (with an anchors weight of 0 both of the parent's branches give w_name*name + w_role*role)
                sub, pos = self._ragged_sub("anchors")
                anchor_v = _segment_max_mean(targets @ sub.T, pos, self.anchor_starts).astype(np.float64)
                has = self.ent_has_anchors
                score[has] = w["name"] * name_v[has] + w["role"] * role_v[has] + w["anchors"] * anchor_v
            ids.extend(self.all_entity_ids)
            parts.append(score)
            div.append(self.entity_role_mat if entity_diversity == "role" else self.entity_name_mat)
        if use_facts and len(self.all_fact_ids):
            n = len(self.all_fact_ids)
            sent_v = (rewrite @ self.fact_sent_mat.T).flatten().astype(np.float32).astype(np.float64)
            w = self.fact_weights
            w_s = np.full(n, w["sentence"], dtype=np.float64)
            w_q = np.full(n, w["questions"], dtype=np.float64)
            w_c = np.full(n, w["chunk"], dtype=np.float64)
            q_v = np.zeros(n, dtype=np.float64)
            no_q = ~self.fact_has_questions
            w_s[no_q] = w_s[no_q] + w_q[no_q]
            w_q[no_q] = 0.0
            if w["questions"] != 0 and self.fact_has_questions.any():
                hq = self.fact_has_questions
                q_v[hq] = (self.U[self.fact_q_rows[hq]] @ rewrite[0]).astype(np.float32)
            chunk_v = np.zeros(n, dtype=np.float64)
            has_c = self.fact_has_topics if len(topics) else np.zeros(n, dtype=bool)
            if has_c.any():
                sub, pos = self._ragged_sub("topics")
                chunk_v[has_c] = _segment_max_mean(topics @ sub.T, pos, self.topic_starts).astype(np.float64)
            no_c = ~has_c
            w_s[no_c] = w_s[no_c] + w_c[no_c]
            w_c[no_c] = 0.0
            ids.extend(self.all_fact_ids)
            parts.append(w_s * sent_v + w_q * q_v + w_c * chunk_v)
            div.append(self.fact_sent_mat)
        scores = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float64)
        return ids, scores, div

    @staticmethod
    def _sim_to(div):
        """callable(i) -> similarity of every candidate to candidate i, over the stacked diversity blocks."""
        offs = np.cumsum([0] + [len(m) for m in div])

        def sim_to(i):
            b = int(np.searchsorted(offs, i, side="right") - 1)
            v = div[b][i - offs[b]]
            return np.concatenate([m @ v for m in div])
        return sim_to

    def _encode_norm(self, texts):
        if not texts:
            return np.zeros((0, self.emb_dim), dtype=np.float32)
        return normalize_rows(self.retrieval_model.encode(texts, show_progress_bar=False))

    def get_seeds(self, query: str, top_k: int = 10, lambda_mmr: float = 0.6, verbose=False, mode="all",
                  precomputed_plan: object = None):
        plan = precomputed_plan if precomputed_plan is not None else self.planner(user_query=query)
        rewrite = self._encode_norm([query])
        targets = self._encode_norm(plan.target_entities)
        broad = self._encode_norm(plan.broad_anchors)
        topics = self._encode_norm(plan.target_topics)
        # the parent scores entities and facts whenever mode is truthy ("all", "hypergraph", "discourse")
        ids, scores, div = self._seed_candidates(rewrite, targets, broad, topics, bool(mode), bool(mode), "role")
        selected = mmr_select(scores.tolist(), top_k=top_k, lambda_param=lambda_mmr, sim_to=self._sim_to(div))
        return [ids[i] for i in selected], plan

    def get_seeds_precomputed(self, plan_embs_dict: dict, top_k: int = 10, lambda_mmr: float = 0.6, mode="all"):
        z = np.zeros((0, self.emb_dim), dtype=np.float32)
        ids, scores, div = self._seed_candidates(
            plan_embs_dict.get("rewritten_query", z), plan_embs_dict.get("target_entities", z),
            plan_embs_dict.get("broad_anchors", z), plan_embs_dict.get("target_topics", z),
            mode in ("all", "hypergraph"), mode in ("all", "discourse"), "name")
        selected = mmr_select(scores.tolist(), top_k=top_k, lambda_param=lambda_mmr, sim_to=self._sim_to(div))
        return [ids[i] for i in selected]

    # ------------------------------------------------------------------ traversal / ranking with mmr_select
    def retrieve(self, query: str, top_k_seeds: int = 20, max_depth: int = 3, beam_width: int = 3,
                 final_top_k: int = 5, beam_mmr_lambda: float = 0.5, final_mmr_lambda: float = 0.6,
                 return_raw_paths: bool = False, verbose=False, mode="all", search_mode: str = "beam_search",
                 deduplicate_paths: bool = True, neutral_layout: bool = False, precomputed_plan: object = None,
                 source_passages: int = 0):
        """Parent's retrieve (same steps and arguments) with mmr_select in beam search and final ranking."""
        if search_mode not in ("beam_search", "blind_expansion"):
            raise ValueError(f"Unknown search_mode '{search_mode}'. Choose 'beam_search' or 'blind_expansion'.")
        seeds, plan = self.get_seeds(query, top_k=top_k_seeds, mode=mode, precomputed_plan=precomputed_plan)
        if not seeds:
            print("No relevant seeds found.")
            return ""
        if search_mode == "beam_search":
            completed_paths = _expand_paths_batched(engine=self, seeds=seeds, plan=plan, max_depth=max_depth,
                                                    beam_width=beam_width, mode=mode, mmr_lambda=beam_mmr_lambda)
        else:
            completed_paths = _traversal.expand_paths_blind(engine=self, seeds=seeds, max_depth=max_depth, mode=mode)
        final_strings, final_paths = _rank_paths_global(engine=self, query=query, completed_paths=completed_paths,
                                                        cross_encoder=self.cross_encoder, final_top_k=final_top_k,
                                                        mmr_lambda=final_mmr_lambda)
        if deduplicate_paths:
            context_string = _traversal.format_paths_deduplicated(self.graph, final_paths, neutral_layout)
        elif neutral_layout:
            context_string = "\n\n".join(_traversal.format_path(self.graph, p, neutral_layout=True)
                                         for p in final_paths)
        else:
            context_string = "\n\n".join(final_strings)
        if source_passages > 0:
            passages = self.source_passages_for(final_paths, source_passages)
            if passages:
                context_string = context_string + "\n\nSource passages:\n\n" + "\n\n".join(passages)
        if return_raw_paths:
            return context_string, final_paths
        return context_string

    def get_raw_paths_precomputed(self, plan_embs: dict, top_k_seeds: int = 8, max_depth: int = 3,
                                  beam_width: int = 3) -> tuple:
        seeds = self.get_seeds_precomputed(plan_embs, top_k=top_k_seeds)
        if not seeds:
            return [], []
        completed_paths = _expand_paths_precomputed(
            engine=self, seeds=seeds,
            plan_labels_embs=plan_embs.get("target_edge_labels", np.array([])),
            plan_semantics_embs=plan_embs.get("target_edge_semantics", np.array([])),
            plan_broad_embs=plan_embs.get("broad_anchors", np.array([])),
            max_depth=max_depth, beam_width=beam_width)
        return completed_paths, [_traversal.assemble_path_string(self.graph, p) for p in completed_paths]
