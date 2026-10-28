"""
PPRRetriever: UNH retrieval with NO LLM call at query time. HippoRAG 2-style Personalized PageRank (PPR) over the whole
UNH graph picks the relevant facts; UNH-style paths are then built through them and formatted exactly as the old
engine formats its paths (traversal.format_paths_deduplicated).

Per question:
  1. Embed the question once (the retriever's bi-encoder, as EfficientRetriever embeds it).
  2. Fact seeds: cosine with every fact sentence -> top `fact_candidates`; the cross-encoder re-ranks them (replaces
     the LLM query planner and HippoRAG's LLM fact filter) -> top `seed_facts`. Seed weight = the fact's min-max
     normalised cosine (HippoRAG's fact score).
  3. Entity seeds (HippoRAG's phrase seeds): each seed fact's entities get its weight / #chunks the entity occurs in
     (IDF-like: generic entities seed almost nothing), averaged over the seed facts that mention them.
  4. Chunk prior (HippoRAG's passage seeds): every chunk gets passage_node_weight * min-max(cosine(question, chunk)).
  5. PPR on the undirected, weighted graph (facts, entities, chunks; fact->fact / entity<->fact / chunk->fact edges,
     per-type weights, fact->fact scaled by the edge's LLM score), damping 0.5, power iteration to convergence.
  6. Fact relevance (fact_relevance="rrf"): reciprocal-rank fusion of the PPR rank (best in depth) and the
     cross-encoder rank over the top `ce_fact_pool` facts by cosine and by PPR (best at the very top).
  7. Paths: the `relevant_facts` most relevant facts form the candidate set. Starting from the top `top_k_seeds` of
     them, a beam search extends each path to an unvisited candidate fact, either directly over a fact->fact edge or
     through a shared entity (fact -> entity -> fact), choosing by relevance: `beam_width` children per path,
     `global_beam` paths kept per depth, up to `max_depth` facts. Hubs cost nothing (only candidate facts are
     considered) and the path count is bounded.
     Selection (path_select="coverage"): greedily the path that adds the most relevance through facts not yet in the
     context, up to `final_top_k` paths and a `context_tokens` budget, so the context covers many distinct
     relevant facts in few characters. force_top_facts=N then adds, for each of the N most relevant facts not yet
     covered, its best candidate path (or the fact alone). ppr_rounds=2 first mixes in a second PPR seeded from the
     round-1 top facts and their entities (bridge facts one hop beyond what the question matches). path_select="ce" (first version): the cross-encoder scores the top
     `ce_path_cap` paths (or the PPR mass, path_score), then MMR with fact-set overlap.
  8. Context: format_paths_compact (layout="compact": a fact's sentence and an entity's role are written only at
     their first mention, later mentions use the fact's number / the entity's name; entity hops carry no labels,
     which only repeated the role or the next fact's sentence), or traversal.format_paths_deduplicated
     (layout="legacy"); the lowest-priority paths are dropped until it fits `context_tokens`. Optionally the text of the top
     `ppr_passages` chunks by PPR score, or of the chunks of the path facts (`source_passages`, as the old engine).

Views (mode): "all" = every edge type; "hypergraph" = entity<->fact only (paths hop through entities);
"discourse" = fact->fact only. chunk->fact edges are in the PPR graph for every view (chunks are never path nodes).

Everything the per-question work reads is built once in the constructor (sparse matrices, index arrays, chunk
embeddings, cached next to the graph like EfficientRetriever's); questions can be retrieved from several threads.
"""
import hashlib
import os
import time

import numpy as np
import scipy.sparse as sp

from nkg.retrieval import traversal as _traversal
from nkg.retrieval.efficient_engine import EfficientRetriever, mmr_select
from nkg.retrieval.traversal import Path
from nkg.utils.math_utils import normalize_rows

DEFAULT_PPR_ARGS = {
    "damping": 0.5,               # probability of following an edge (igraph / HippoRAG convention)
    "passage_node_weight": 0.05,  # chunk prior scale (HippoRAG 2)
    "fact_candidates": 50,        # facts by cosine handed to the cross-encoder
    "seed_facts": 5,              # facts kept as PPR seeds (HippoRAG linking_top_k)
    "ce_seed_rerank": True,       # cross-encoder picks the seed facts (False: top cosine)
    "fact_seed_weight": 1.0,      # scale of the fact seeds
    "entity_seed_weight": 1.0,    # scale of the entity seeds (0 = none)
    "edge_weights": {"fact_fact": 1.0, "entity_fact": 1.0, "chunk_fact": 1.0},
    "ff_use_score": True,         # fact->fact edge weight *= its LLM score
    "relevant_facts": 150,        # candidate facts for paths (top PPR)
    "global_beam": 60,            # paths kept per depth
    "path_score": "ce",           # "ce" | "ppr" | "ce+ppr"
    "ppr_passages": 0,            # append the top-N chunks by PPR score
    "fact_relevance": "rrf",      # fact relevance for paths: "rrf" (CE rank + PPR rank) | "ppr" | "ce"
    "ce_fact_pool": 100,          # facts scored by the cross-encoder: top cosine + top PPR (each this many)
    "rrf_k": 20,                  # reciprocal-rank-fusion constant
    "path_select": "coverage",    # "coverage": greedy new-relevant-fact gain | "ce": cross-encoder + MMR (first version)
    "ce_path_cap": 100,           # path_select="ce": only the top paths by relevance go to the cross-encoder
    "context_tokens": 2000,       # token budget for the formatted paths (0 = none); lowest-priority paths dropped
    "budget_tokenizer": "Qwen/Qwen3-30B-A3B-Instruct-2507",  # tokenizer the budget is counted with
    "layout": "compact",          # "compact": each fact / entity written in full once (format_paths_compact)
                                  # | "legacy": traversal.format_paths_deduplicated, as the old engine
    "force_top_facts": 0,         # after coverage selection, every top-N relevant fact not yet in the context gets
                                  # its best candidate path (or itself alone); these come last, so a budget cuts them first
    "ppr_rounds": 1,              # 2: second PPR seeded from the round-1 top `round2_seed_facts` facts (and their
                                  # entities), mixed into the PPR scores before the relevance fusion (reaches bridge facts)
    "round2_seed_facts": 10,
    "round2_mix": 1.0,            # weight of the round-2 scores (each round normalised to sum 1)
    "ppr_max_iter": 60,
    "ppr_tol": 1e-8,
}

_VIEW_PPR_EDGES = {"all": ("fact_fact", "entity_fact", "chunk_fact"),
                   "hypergraph": ("entity_fact", "chunk_fact"),
                   "discourse": ("fact_fact", "chunk_fact")}
_VIEW_HOPS = {"all": (True, True), "hypergraph": (False, True), "discourse": (True, False)}  # (fact->fact, via entity)


def format_paths_compact(graph, ranked_paths: list) -> str:
    """Paths in priority order, one block per path; an edge already written by an earlier path is not written again
    (the path is split into its novel runs, as format_paths_deduplicated does). Facts are numbered at their first
    mention: (F3) "sentence"; later mentions are just (F3). Entities are [name (role)] at their first mention, then
    [name]. Fact->fact edges show their label (-> LABEL ->); entity hops are plain arrows (their labels repeat the
    role or the next fact's sentence)."""
    fact_no, seen_ent, edge_cache, blocks = {}, set(), set(), []

    def node(n):
        if n in graph.facts:
            if n in fact_no:
                return f"(F{fact_no[n]})"
            fact_no[n] = len(fact_no) + 1
            return f'(F{fact_no[n]}) "{graph.facts[n].sentence}"'
        if n in graph.entities:
            e = graph.entities[n]
            if n in seen_ent or not (e.role or "").strip():
                seen_ent.add(n)
                return f"[{e.name}]"
            seen_ent.add(n)
            return f"[{e.name} ({e.role})]"
        return "[?]"

    for p in ranked_paths:
        nodes = _traversal._get_path_nodes(p)
        edges = _traversal._build_edge_lookup(p)
        runs = [nodes] if len(nodes) == 1 else _traversal._extract_bursts(nodes, edge_cache)
        if len(nodes) == 1 and nodes[0] in fact_no:
            continue
        for run in runs:
            out = node(run[0])
            for u, v in zip(run, run[1:]):
                et, data = edges.get((u, v), (None, {}))
                label = (data.get("label") or "").strip() if et == "fact_fact" else ""
                out += f" -> {label} -> {node(v)}" if label else f" -> {node(v)}"
            blocks.append(out)
    return "\n".join(blocks)


def _minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    return np.ones_like(x) if hi - lo == 0 else (x - lo) / (hi - lo)


class PPRRetriever(EfficientRetriever):
    """EfficientRetriever's graph / embeddings plus the PPR index. Extra constructor arguments: chunk_cache_path
    (chunk-text embeddings, None = no cache) and ppr_args (defaults for the DEFAULT_PPR_ARGS knobs)."""

    def __init__(self, *args, chunk_cache_path: str = None, ppr_args: dict = None, **kwargs):
        self._chunk_cache_path = chunk_cache_path
        self.ppr_args = {**DEFAULT_PPR_ARGS, **(ppr_args or {})}
        super().__init__(*args, **kwargs)
        self._build_ppr_index()

    # ------------------------------------------------------------------ index (once)
    def _build_ppr_index(self):
        t0 = time.time()
        g = self.graph
        self.fact_ids = self.all_fact_ids                       # rows of fact_sent_mat
        self.ent_ids = self.all_entity_ids
        self.chunk_ids = list(g.chunks.keys())
        nf, ne, nc = len(self.fact_ids), len(self.ent_ids), len(self.chunk_ids)
        self.n_nodes = nf + ne + nc
        self.f_off, self.e_off, self.c_off = 0, nf, nf + ne
        fpos = {f: i for i, f in enumerate(self.fact_ids)}
        epos = {e: i for i, e in enumerate(self.ent_ids)}
        cpos = {c: i for i, c in enumerate(self.chunk_ids)}

        ff_u, ff_v, ff_s, ef_e, ef_f, cf_c, cf_f = [], [], [], [], [], [], []
        for u, v, data in g.network.edges(data=True):
            et = data.get("edge_type")
            if et == "fact_fact" and u in fpos and v in fpos:
                ff_u.append(fpos[u]); ff_v.append(fpos[v])
                s = data.get("score")
                ff_s.append(float(s) if s is not None else 1.0)
            elif et == "entity_fact" and u in epos and v in fpos:
                ef_e.append(epos[u]); ef_f.append(fpos[v])
            elif et == "fact_entity" and u in fpos and v in epos:
                ef_e.append(epos[v]); ef_f.append(fpos[u])
            elif et == "chunk_fact" and u in cpos and v in fpos:
                cf_c.append(cpos[u]); cf_f.append(fpos[v])
        ff_u, ff_v, ff_s = np.array(ff_u, np.int64), np.array(ff_v, np.int64), np.array(ff_s, np.float64)
        ef_e, ef_f = np.array(ef_e, np.int64), np.array(ef_f, np.int64)
        cf_c, cf_f = np.array(cf_c, np.int64), np.array(cf_f, np.int64)

        # binary, symmetric (deduplicated) structure per type; weights applied per view in _ppr_matrix
        def sym(rows, cols, n_rows, n_cols, vals=None):
            vals = np.ones(len(rows)) if vals is None else vals
            m = sp.csr_matrix((vals, (rows, cols)), shape=(n_rows, n_cols))
            m.sum_duplicates()
            return m
        ff = sym(ff_u, ff_v, nf, nf, ff_s)
        ff = ff.maximum(ff.T).tocsr()                            # undirected; an edge stored both ways keeps max score
        self.ff_score = ff                                       # facts x facts, value = LLM score
        ef = sym(ef_e, ef_f, ne, nf)
        ef.data[:] = 1.0                                         # entity<->fact stored both ways in the graph
        self.ef = ef.tocsr()                                     # entities x facts
        self.fe = self.ef.T.tocsr()                              # facts x entities
        cf = sym(cf_c, cf_f, nc, nf)
        cf.data[:] = 1.0
        self.cf = cf.tocsr()                                     # chunks x facts
        self.fc = self.cf.T.tocsr()                              # facts x chunks
        self.ent_n_facts = np.diff(self.ef.indptr)
        # #distinct chunks per entity (HippoRAG's ent_node_to_chunk_ids): entity -> facts -> chunks
        ec = (self.ef @ self.fc).tocsr()
        self.ent_n_chunks = np.maximum(np.diff(ec.indptr), 1)
        self._ppr_mats = {}

        self.chunk_mat = self._load_or_encode_chunks([g.chunks[c].text for c in self.chunk_ids])
        print(f"[PPRRetriever] {nf} facts, {ne} entities, {nc} chunks; edges ff={ff.nnz // 2} ef={self.ef.nnz} "
              f"cf={self.cf.nnz}; index built in {time.time() - t0:.0f}s", flush=True)

    def _load_or_encode_chunks(self, texts: list) -> np.ndarray:
        h = hashlib.sha1(f"{self._model_name}|chunks|batch{self._encode_batch_size}".encode())
        for s in texts:
            h.update(s.encode("utf-8", "surrogatepass"))
            h.update(b"\x00")
        key = h.hexdigest()
        path = self._chunk_cache_path
        if path and os.path.exists(path):
            with np.load(path, allow_pickle=False) as data:
                if str(data["key"]) == key:
                    print(f"[PPRRetriever] chunk embeddings loaded from {path}", flush=True)
                    return data["C"]
            print(f"[PPRRetriever] {path} is for another graph/model; re-encoding", flush=True)
        t0 = time.time()
        nonempty = [i for i, s in enumerate(texts) if s and s.strip()]
        C = np.zeros((len(texts), self.emb_dim), dtype=np.float32)
        if nonempty:
            C[nonempty] = self.retrieval_model.encode([texts[i] for i in nonempty], batch_size=self._encode_batch_size,
                                                      show_progress_bar=True)
        C = normalize_rows(C)
        print(f"[PPRRetriever] encoded {len(nonempty)} chunks in {time.time() - t0:.0f}s", flush=True)
        if path:
            tmp = path + ".tmp.npz"
            np.savez(tmp, key=np.array(key), C=C)
            os.replace(tmp, path)
        return C

    def _ppr_matrix(self, mode: str, edge_weights: dict, ff_use_score: bool):
        """(W, inverse weighted degree) of the undirected node graph for a view; cached per configuration."""
        key = (mode, tuple(sorted(edge_weights.items())), ff_use_score)
        if key not in self._ppr_mats:
            with self._lazy_lock:
                if key not in self._ppr_mats:
                    nf, ne, nc = len(self.fact_ids), len(self.ent_ids), len(self.chunk_ids)
                    types = _VIEW_PPR_EDGES[mode]
                    zero = lambda r, c: sp.csr_matrix((r, c))  # noqa: E731
                    ff = zero(nf, nf)
                    if "fact_fact" in types and edge_weights.get("fact_fact", 0):
                        ff = self.ff_score.copy()
                        if not ff_use_score:
                            ff.data[:] = 1.0
                        ff = ff * edge_weights["fact_fact"]
                    fe = self.fe * edge_weights.get("entity_fact", 0) if "entity_fact" in types else zero(nf, ne)
                    fc = self.fc * edge_weights.get("chunk_fact", 0) if "chunk_fact" in types else zero(nf, nc)
                    W = sp.bmat([[ff, fe, fc],
                                 [fe.T, None, None],
                                 [fc.T, None, None]], format="csr")
                    W.eliminate_zeros()
                    deg = np.asarray(W.sum(axis=1)).ravel()
                    inv = np.zeros_like(deg)
                    inv[deg > 0] = 1.0 / deg[deg > 0]
                    self._ppr_mats[key] = (W, inv)
        return self._ppr_mats[key]

    def _tokenizer(self, name: str):
        if getattr(self, "_tok", (None, None))[0] != name:
            with self._lazy_lock:
                if getattr(self, "_tok", (None, None))[0] != name:
                    from transformers import AutoTokenizer
                    self._tok = (name, AutoTokenizer.from_pretrained(name))
        return self._tok[1]

    # ------------------------------------------------------------------ PPR
    def personalized_pagerank(self, reset: np.ndarray, mode: str = "all", damping: float = 0.5,
                              edge_weights: dict = None, ff_use_score: bool = True, max_iter: int = 60,
                              tol: float = 1e-8) -> np.ndarray:
        """pi = (1 - d) r + d P^T pi with P = D^-1 W (W symmetric -> P^T pi = W (pi / deg)); mass reaching a node
        with no edges restarts from r. Same fixed point as igraph's personalized_pagerank(damping=d, directed=False)."""
        W, inv = self._ppr_matrix(mode, edge_weights or self.ppr_args["edge_weights"], ff_use_score)
        r = reset / reset.sum()
        dangling = inv == 0
        pi = r.copy()
        for _ in range(max_iter):
            nxt = damping * (W @ (pi * inv))
            nxt += (1.0 - damping + damping * pi[dangling].sum()) * r
            if np.abs(nxt - pi).sum() < tol:
                pi = nxt
                break
            pi = nxt
        return pi

    def _reset_vector(self, query: str, qv: np.ndarray, a: dict):
        """(reset vector over all nodes, seed fact rows, fact cosine scores, {fact row: cross-encoder score})."""
        nf = len(self.fact_ids)
        cos = self.fact_sent_mat @ qv
        k = min(a["fact_candidates"], nf)
        cand = np.argpartition(-cos, k - 1)[:k]
        ce_cache = {}
        if a["ce_seed_rerank"]:
            ce = self._ce_facts(query, cand, ce_cache)
            order = cand[np.argsort(-ce, kind="stable")]
        else:
            order = cand[np.argsort(-cos[cand], kind="stable")]
        seeds = order[: a["seed_facts"]]
        fscore = _minmax(cos)                                    # HippoRAG: min-max over all facts
        reset = np.zeros(self.n_nodes)
        reset[self.f_off + seeds] = a["fact_seed_weight"] * fscore[seeds]
        if a["entity_seed_weight"] and len(self.ent_ids):
            w = np.zeros(len(self.ent_ids))
            n = np.zeros(len(self.ent_ids))
            for f in seeds:
                ents = self.fe.indices[self.fe.indptr[f]:self.fe.indptr[f + 1]]
                w[ents] += fscore[f] / self.ent_n_chunks[ents]
                n[ents] += 1
            has = n > 0
            reset[self.e_off:self.c_off][has] = a["entity_seed_weight"] * w[has] / n[has]
        if a["passage_node_weight"] and len(self.chunk_ids):
            reset[self.c_off:] = a["passage_node_weight"] * _minmax(self.chunk_mat @ qv)
        return reset, seeds, cos, ce_cache

    def _ce_facts(self, query: str, rows, cache: dict) -> np.ndarray:
        """Cross-encoder scores (question, fact sentence) for fact rows; scores already in cache are reused."""
        todo = [int(i) for i in rows if int(i) not in cache]
        if todo:
            sc = self.cross_encoder.predict([[query, self.graph.facts[self.fact_ids[i]].sentence] for i in todo],
                                            show_progress_bar=False)
            cache.update(zip(todo, map(float, sc)))
        return np.array([cache[int(i)] for i in rows], dtype=np.float64)

    def _second_round(self, pf: np.ndarray, rel: np.ndarray, mode: str, a: dict) -> np.ndarray:
        """PPR scores mixed with a second PPR whose seeds are the round-1 top facts by relevance (weight = their
        normalised relevance) and their entities (IDF-weighted as in _reset_vector): mass one hop beyond the facts
        the question matches, i.e. toward bridge facts."""
        k = min(a["round2_seed_facts"], len(rel))
        top = np.argpartition(-rel, k - 1)[:k]
        w = rel[top] / rel[top].max()
        reset = np.zeros(self.n_nodes)
        reset[self.f_off + top] = w
        ent_w = np.zeros(len(self.ent_ids))
        ent_n = np.zeros(len(self.ent_ids))
        for f, wf in zip(top, w):
            ents = self.fe.indices[self.fe.indptr[f]:self.fe.indptr[f + 1]]
            ent_w[ents] += wf / self.ent_n_chunks[ents]
            ent_n[ents] += 1
        has = ent_n > 0
        reset[self.e_off:self.c_off][has] = a["entity_seed_weight"] * ent_w[has] / ent_n[has]
        pf2 = self.personalized_pagerank(reset, mode, a["damping"], a["edge_weights"], a["ff_use_score"],
                                         a["ppr_max_iter"], a["ppr_tol"])[self.f_off:self.e_off]
        return pf / max(pf.sum(), 1e-300) + a["round2_mix"] * pf2 / max(pf2.sum(), 1e-300)

    def _fact_relevance(self, query: str, cos: np.ndarray, pf: np.ndarray, ce_cache: dict, a: dict) -> np.ndarray:
        """Relevance of every fact for path building. rrf: 1/(k + PPR rank) + 1/(k + cross-encoder rank), the
        cross-encoder scoring the top ce_fact_pool facts by cosine and by PPR (PPR is the better ranking in depth,
        the cross-encoder at the very top); facts outside the pool get only the PPR term."""
        if a["fact_relevance"] == "ppr":
            return pf / pf.max() if pf.max() > 0 else pf
        n = len(pf)
        m = min(a["ce_fact_pool"], n)
        pool = np.union1d(np.argpartition(-cos, m - 1)[:m], np.argpartition(-pf, m - 1)[:m])
        ce = self._ce_facts(query, pool, ce_cache)
        ce_rank = np.full(n, np.inf)
        ce_rank[pool[np.argsort(-ce, kind="stable")]] = np.arange(len(pool))
        if a["fact_relevance"] == "ce":
            rel = np.zeros(n)
            rel[pool] = 1.0 / (a["rrf_k"] + ce_rank[pool])
            return rel
        ppr_rank = np.empty(n)
        ppr_rank[np.argsort(-pf, kind="stable")] = np.arange(n)
        return 1.0 / (a["rrf_k"] + ppr_rank) + 1.0 / (a["rrf_k"] + ce_rank)

    # ------------------------------------------------------------------ paths
    def _edge_data(self, u: str, v: str) -> dict:
        net = self.graph.network
        d = net.get_edge_data(u, v)
        return d if d is not None else (net.get_edge_data(v, u) or {})

    def _build_paths(self, pf: np.ndarray, mode: str, start_k: int, beam_width: int, global_beam: int,
                     max_depth: int, n_relevant: int):
        """Beam search over the top-n_relevant facts by PPR score. Returns completed paths as
        (fact row list, list of (kind, entity row or -1)) with kind 'ff' / 'ent' per hop."""
        use_ff, use_ent = _VIEW_HOPS[mode]
        nf = len(pf)
        n_relevant = min(n_relevant, nf)
        rel = np.argpartition(-pf, n_relevant - 1)[:n_relevant]
        rel = rel[np.argsort(-pf[rel], kind="stable")]
        in_rel = np.zeros(nf, dtype=bool)
        in_rel[rel] = True

        nbrs = {}  # fact row -> {fact row: (kind, entity row)} within the candidate set; ff preferred over entity hops

        def neighbours(f):
            if f in nbrs:
                return nbrs[f]
            out = {}
            if use_ent:
                ents = self.fe.indices[self.fe.indptr[f]:self.fe.indptr[f + 1]]
                # most specific shared entity first, so it wins for a fact reachable through several
                for e in ents[np.argsort(self.ent_n_facts[ents], kind="stable")[::-1]]:
                    fs = self.ef.indices[self.ef.indptr[e]:self.ef.indptr[e + 1]]
                    for g2 in fs[in_rel[fs]]:
                        if g2 != f:
                            out[int(g2)] = ("ent", int(e))
            if use_ff:
                fs = self.ff_score.indices[self.ff_score.indptr[f]:self.ff_score.indptr[f + 1]]
                for g2 in fs[in_rel[fs]]:
                    out[int(g2)] = ("ff", -1)
            nbrs[f] = out
            return out

        beam = [(float(pf[f]), [int(f)], []) for f in rel[:start_k]]
        completed = []
        while beam:
            children = []
            for score, facts, hops in beam:
                if len(facts) >= max_depth:
                    completed.append((score, facts, hops))
                    continue
                opts = [(g2, h) for g2, h in neighbours(facts[-1]).items() if g2 not in facts]
                if not opts:
                    completed.append((score, facts, hops))
                    continue
                opts.sort(key=lambda x: -pf[x[0]])
                for g2, h in opts[:beam_width]:
                    children.append((score + float(pf[g2]), facts + [g2], hops + [h]))
            children.sort(key=lambda x: -x[0])
            completed.extend(children[global_beam:])  # pruned children still count as (shorter) candidates
            beam = children[:global_beam]
        return completed

    def _to_path(self, facts: list, hops: list) -> Path:
        fid = self.fact_ids
        p = Path(fid[facts[0]], True)
        for (kind, e), f2 in zip(hops, facts[1:]):
            cur = p.current_node
            if kind == "ff":
                p = p.branch(fid[f2], True, "fact_fact", self._edge_data(cur, fid[f2]))
            else:
                eid = self.ent_ids[e]
                p = p.branch(eid, False, "fact_entity", self._edge_data(cur, eid))
                p = p.branch(fid[f2], True, "entity_fact", self._edge_data(eid, fid[f2]))
        return p

    # ------------------------------------------------------------------ public API
    def retrieve(self, query: str, top_k_seeds: int = 12, max_depth: int = 4, beam_width: int = 6,
                 final_top_k: int = 20, final_mmr_lambda: float = 0.6, mode: str = "all",
                 deduplicate_paths: bool = True, neutral_layout: bool = False, source_passages: int = 0,
                 return_raw_paths: bool = False, return_debug: bool = False, **kwargs):
        """Same output as Retriever.retrieve (a context string of formatted paths). The old engine's planner /
        beam-MMR arguments (beam_mmr_lambda, search_mode, precomputed_plan, verbose) are accepted and ignored;
        DEFAULT_PPR_ARGS keys override self.ppr_args for this call."""
        a = {**self.ppr_args, **{k: v for k, v in kwargs.items() if k in DEFAULT_PPR_ARGS}}
        qv = self._encode_norm([query])[0]
        reset, seeds, cos, ce_cache = self._reset_vector(query, qv, a)
        pi = self.personalized_pagerank(reset, mode, a["damping"], a["edge_weights"], a["ff_use_score"],
                                        a["ppr_max_iter"], a["ppr_tol"])
        pf = pi[self.f_off:self.e_off]
        rel = self._fact_relevance(query, cos, pf, ce_cache, a)
        if a["ppr_rounds"] >= 2:
            pf = self._second_round(pf, rel, mode, a)
            rel = self._fact_relevance(query, cos, pf, ce_cache, a)
        raw = self._build_paths(rel, mode, top_k_seeds, beam_width, a["global_beam"], max_depth, a["relevant_facts"])
        fact_sets = [set(facts) for _, facts, _ in raw]

        if a["path_select"] == "coverage":
            # greedy: the path adding the most relevance through facts not yet in the context; ties -> fewer facts
            covered, chosen = set(), []
            remaining = set(range(len(raw)))
            while remaining and len(chosen) < final_top_k:
                best, best_key = None, None
                for i in remaining:
                    new = fact_sets[i] - covered
                    key = (sum(rel[f] for f in new), -len(raw[i][1]))
                    if best_key is None or key > best_key:
                        best, best_key = i, key
                if best_key[0] <= 0:
                    break
                chosen.append(best)
                covered |= fact_sets[best]
                remaining.discard(best)
            if a["force_top_facts"]:
                k = min(a["force_top_facts"], len(rel))
                top = np.argpartition(-rel, k - 1)[:k]
                for f in top[np.argsort(-rel[top], kind="stable")]:
                    f = int(f)
                    if f in covered:
                        continue
                    with_f = [i for i in remaining if f in fact_sets[i]]
                    if with_f:
                        best = max(with_f, key=lambda i: (sum(rel[x] for x in fact_sets[i] - covered), -len(raw[i][1])))
                    else:
                        raw.append((float(rel[f]), [f], []))
                        fact_sets.append({f})
                        best = len(raw) - 1
                    chosen.append(best)
                    covered |= fact_sets[best]
                    remaining.discard(best)
            paths = {i: self._to_path(raw[i][1], raw[i][2]) for i in chosen}
            strings = {i: _traversal.assemble_path_string(self.graph, paths[i]) for i in chosen}
        else:
            # cross-encoder over the top ce_path_cap paths by relevance, then MMR with fact-set overlap
            cap = sorted(range(len(raw)), key=lambda i: -raw[i][0])[: a["ce_path_cap"]]
            paths = {i: self._to_path(raw[i][1], raw[i][2]) for i in cap}
            strings = {i: _traversal.assemble_path_string(self.graph, paths[i]) for i in cap}
            ce = np.asarray(self.cross_encoder.predict([[query, strings[i]] for i in cap], show_progress_bar=False),
                            dtype=np.float64)
            ppr_score = np.array([raw[i][0] for i in cap], dtype=np.float64)
            score = (_minmax(ce) if a["path_score"] == "ce" else _minmax(ppr_score) if a["path_score"] == "ppr"
                     else 0.5 * _minmax(ce) + 0.5 * _minmax(ppr_score))

            def sim_to(j):
                si = fact_sets[cap[j]]
                return np.array([len(si & fact_sets[i]) / len(si | fact_sets[i]) for i in cap], dtype=np.float32)
            chosen = [cap[j] for j in mmr_select(score.tolist(), top_k=final_top_k, lambda_param=final_mmr_lambda,
                                                 sim_to=sim_to)]

        def render(sel):
            fp = [paths[i] for i in sel]
            if a["layout"] == "compact":
                return format_paths_compact(self.graph, fp)
            if deduplicate_paths:
                return _traversal.format_paths_deduplicated(self.graph, fp, neutral_layout)
            if neutral_layout:
                return "\n\n".join(_traversal.format_path(self.graph, p, neutral_layout=True) for p in fp)
            return "\n\n".join(strings[i] for i in sel)
        context = render(chosen)
        if a["context_tokens"]:
            tok = self._tokenizer(a["budget_tokenizer"])
            while len(chosen) > 1 and len(tok(context)["input_ids"]) > a["context_tokens"]:
                chosen = chosen[:-1]                             # drop the lowest-priority path until within budget
                context = render(chosen)
        final_paths = [paths[i] for i in chosen]

        passages = []
        if a["ppr_passages"] > 0:
            pc = pi[self.c_off:]
            top = np.argsort(-pc, kind="stable")[: a["ppr_passages"]]
            passages = [self.graph.chunks[self.chunk_ids[i]].text for i in top]
        elif source_passages > 0:
            passages = self.source_passages_for(final_paths, source_passages)
        if passages:
            context = context + "\n\nSource passages:\n\n" + "\n\n".join(passages)

        if return_debug:
            return context, {"seed_facts": [self.fact_ids[i] for i in seeds], "n_candidate_paths": len(raw),
                             "final_paths": final_paths,
                             "top_chunks": [self.chunk_ids[i] for i in np.argsort(-pi[self.c_off:])[:10]]}
        if return_raw_paths:
            return context, final_paths
        return context
