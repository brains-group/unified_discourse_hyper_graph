"""
GuidedPPRRetriever: PPRRetriever (nkg/retrieval/ppr_engine.py) with two additions from
evaluation_scripts/unh/PPR_HOP_GUIDED_PLAN.md. Path building, fact relevance (CE + PPR rank fusion), coverage selection,
compact layout and the token budget are PPRRetriever's, unchanged.

1. Typed transitions (plan section 4.3; experiment E1). The walk at a fact node first picks an edge TYPE, then a
   neighbour of that type:
       P(u -> v) = alpha_t(u) * w(u,v) b(v) / sum_{v' of type t} w(u,v') b(v')
   alpha_t(u) = type_alpha[t] renormalised over the types u has. type_alpha=None keeps the original walk (types in
   proportion to their weighted degree, i.e. plain row normalisation). b(v) is a per-node bias on fact targets
   (1 unless a hop plan biases the walk). entity_transit=True makes fact -> entity -> fact ONE step (an entity is
   passed through without damping or restart), so an entity bridge costs the same damping as a discourse edge.
   Computed with transposed blocks and per-block row sums, no per-query matrix is built:
       x_t = pi_u * alpha_t(u) / s_t(u)  (s_t = B_t @ b),   mass into v = b(v) * (B_t^T x_t)(v).

2. Cascaded hop-guided PPR (plan section 4.2; experiment E2 = oracle hops from MuSiQue's question_decomposition,
   E3 = an LLM plan). hops = [hop 1 text, hop 2 text, ...] (each describes what that hop's fact says, not its answer):
       stage 0: restart from the facts matching hop 1 (cross-encoder over the top cosine facts, as PPRRetriever seeds)
                + their entities; walk biased toward facts matching hop 1
       stage k: restart r_k = normalise(pi_{k-1}[facts] * (eps + m_k)), m_k = match(fact, hop k)
                walk biased toward hop k+1 (the last stage toward hop K)
       guided fact score = sum_k normalise(pi_{k-1}[facts] * (eps + m_k))   (every hop's landing facts)
       pf = lam * guided + (1 - lam) * normalise(plain PPR facts)  -> PPRRetriever's fact relevance and paths
   m_k(f) = exp(guide_beta * (cos(hop k, fact f) - max_f cos)): 1 for the best-matching fact, ~0 far below it.
   Multiplying by pi_{k-1} anchors a broad hop ("grandfather of ...") to the previous hop's neighbourhood; every
   stage restarts, so depth does not compound the damping.
"""
import threading

import numpy as np

from nkg.retrieval.ppr_engine import DEFAULT_PPR_ARGS, PPRRetriever, _minmax

GUIDED_ARGS = {
    "type_alpha": None,        # {"fact_fact": a, "entity_fact": b, "chunk_fact": c} or None (original walk)
    "entity_transit": False,   # fact -> entity -> fact as one step
    "hops": None,              # list of hop texts (cascade) or None
    "hop_seed_mode": "cascade",  # "cascade" | "multiseed" (restart from every hop's seeds at once, no cascade)
    "guide_lambda": 0.5,       # weight of the guided score vs plain PPR
    "guide_eps": 0.1,          # floor of the hop match (a wrong hop degrades to plain PPR)
    "guide_beta": 20.0,        # sharpness of the hop match
    "guide_bias": True,        # bias each stage's walk toward the next hop's facts
    "stage_damping": None,     # damping inside cascade stages (None = damping)
}


class GuidedPPRRetriever(PPRRetriever):
    def __init__(self, *args, **kwargs):
        self._tl = threading.local()
        super().__init__(*args, **kwargs)
        self._typed = {}

    # ------------------------------------------------------------------ typed transitions
    def _blocks(self, mode: str, ff_use_score: bool):
        """Per-view blocks with their transposes: FF (facts x facts), FE (facts x entities), FC (facts x chunks)."""
        key = (mode, ff_use_score)
        if key not in self._typed:
            with self._lazy_lock:
                if key not in self._typed:
                    from nkg.retrieval.ppr_engine import _VIEW_PPR_EDGES
                    types = _VIEW_PPR_EDGES[mode]
                    ff = self.ff_score.copy() if "fact_fact" in types else None
                    if ff is not None and not ff_use_score:
                        ff.data[:] = 1.0
                    fe = self.fe if "entity_fact" in types else None
                    fc = self.fc if "chunk_fact" in types else None
                    blk = {}
                    for name, m in (("fact_fact", ff), ("entity_fact", fe), ("chunk_fact", fc)):
                        if m is not None and m.nnz:
                            blk[name] = (m.tocsr(), m.T.tocsr())
                    self._typed[key] = blk
        return self._typed[key]

    def _typed_ppr(self, reset: np.ndarray, mode: str, damping: float, a: dict, fact_bias: np.ndarray = None):
        """pi = (1 - d) r + d * T(pi), T = the typed walk in the module docstring (plain row normalisation when
        type_alpha is None and fact_bias is None). Mass that cannot move restarts from r."""
        nf, ne = len(self.fact_ids), len(self.ent_ids)
        fo, eo, co = self.f_off, self.e_off, self.c_off
        blk = self._blocks(mode, a["ff_use_score"])
        ew = a["edge_weights"]
        b = np.ones(nf) if fact_bias is None else fact_bias
        # row sums per type at fact nodes (biased where the target is a fact), and the reverse rows
        s, wt = {}, {}
        for t, (B, _) in blk.items():
            s[t] = B @ b if t == "fact_fact" else np.asarray(B.sum(axis=1)).ravel()
            wt[t] = ew.get(t, 1.0) * s[t]
        alpha = a["type_alpha"]
        if alpha is None:   # original walk: type share proportional to its (biased) weighted degree
            tot = sum(wt.values())
            share = {t: np.divide(wt[t], tot, out=np.zeros(nf), where=tot > 0) for t in blk}
        else:
            raw = {t: alpha.get(t, 0.0) * (s[t] > 0) for t in blk}
            tot = sum(raw.values())
            share = {t: np.divide(raw[t], tot, out=np.zeros(nf), where=tot > 0) for t in blk}
        coef = {t: np.divide(share[t], s[t], out=np.zeros(nf), where=s[t] > 0) for t in blk}
        # reverse rows: entity -> facts, chunk -> facts (biased by b, normalised per entity / chunk)
        rs = {}
        for t in ("entity_fact", "chunk_fact"):
            if t in blk:
                rs[t] = blk[t][1] @ b                              # (entities or chunks) x facts @ b
        r = reset / reset.sum()
        pi = r.copy()
        transit = a["entity_transit"] and "entity_fact" in blk
        for _ in range(a["ppr_max_iter"]):
            pf = pi[fo:eo]
            nxt = np.zeros_like(pi)
            if "fact_fact" in blk:
                nxt[fo:eo] += b * (blk["fact_fact"][1] @ (pf * coef["fact_fact"]))
            for t, off, n in (("entity_fact", eo, ne), ("chunk_fact", co, len(self.chunk_ids))):
                if t not in blk:
                    continue
                B, BT = blk[t]
                into = BT @ (pf * coef[t])                          # fact -> entity / chunk
                src = pi[off:off + n]
                if t == "entity_fact" and transit:
                    src = src + into                                # pass through the entity in the same step
                else:
                    nxt[off:off + n] += into
                back = np.divide(src, rs[t], out=np.zeros(n), where=rs[t] > 0)
                nxt[fo:eo] += b * (B @ back)                        # entity / chunk -> facts
            moved = nxt.sum()
            nxt = damping * nxt + (1.0 - damping * moved) * r       # unmoved mass (dangling) restarts too
            if np.abs(nxt - pi).sum() < a["ppr_tol"]:
                pi = nxt
                break
            pi = nxt
        return pi

    def personalized_pagerank(self, reset, mode="all", damping=0.5, edge_weights=None, ff_use_score=True,
                              max_iter=60, tol=1e-8):
        cfg = getattr(self._tl, "cfg", None)
        if cfg is None or (cfg["type_alpha"] is None and not cfg["entity_transit"]):
            return super().personalized_pagerank(reset, mode, damping, edge_weights, ff_use_score, max_iter, tol)
        a = {**cfg, "edge_weights": edge_weights or self.ppr_args["edge_weights"], "ff_use_score": ff_use_score,
             "ppr_max_iter": max_iter, "ppr_tol": tol}
        return self._typed_ppr(reset, mode, damping, a)

    # ------------------------------------------------------------------ hop cascade
    def _hop_match(self, text: str) -> np.ndarray:
        v = self._encode_norm([text])[0]
        cos = self.fact_sent_mat @ v
        return np.exp(self._tl.cfg["guide_beta"] * (cos - cos.max())), cos

    def _seed_reset(self, query: str, cos: np.ndarray, a: dict) -> np.ndarray:
        """PPRRetriever's seeding for one text: cross-encoder over its top cosine facts -> seed facts + entities."""
        k = min(a["fact_candidates"], len(cos))
        cand = np.argpartition(-cos, k - 1)[:k]
        ce = self._ce_facts(query, cand, {})
        seeds = cand[np.argsort(-ce, kind="stable")][: a["seed_facts"]]
        fscore = _minmax(cos)
        reset = np.zeros(self.n_nodes)
        reset[self.f_off + seeds] = fscore[seeds]
        w = np.zeros(len(self.ent_ids)); n = np.zeros(len(self.ent_ids))
        for f in seeds:
            ents = self.fe.indices[self.fe.indptr[f]:self.fe.indptr[f + 1]]
            w[ents] += fscore[f] / self.ent_n_chunks[ents]
            n[ents] += 1
        has = n > 0
        reset[self.e_off:self.c_off][has] = a["entity_seed_weight"] * w[has] / n[has]
        return reset

    def _guided_facts(self, mode: str, a: dict) -> np.ndarray:
        cfg = self._tl.cfg
        hops = cfg["hops"]
        d = cfg["stage_damping"] or a["damping"]
        aa = {**a, **cfg}
        ms = [self._hop_match(h) for h in hops]
        if cfg["hop_seed_mode"] == "multiseed":
            reset = sum(self._seed_reset(h, cos, aa) for h, (_, cos) in zip(hops, ms))
            pi = self._typed_ppr(reset, mode, d, aa)
            pf = pi[self.f_off:self.e_off]
            return pf / pf.sum()
        eps = cfg["guide_eps"]
        reset = self._seed_reset(hops[0], ms[0][1], aa)
        guided = np.zeros(len(self.fact_ids))
        for k in range(len(hops)):
            bias = (eps + ms[min(k, len(hops) - 1)][0]) if cfg["guide_bias"] else None
            pi = self._typed_ppr(reset, mode, d, aa, fact_bias=bias)
            land = pi[self.f_off:self.e_off] * (eps + ms[k][0])      # stage-k landing facts (hop k matched)
            land /= max(land.sum(), 1e-300)
            guided += land
            if k + 1 < len(hops):
                reset = np.zeros(self.n_nodes)
                reset[self.f_off:self.e_off] = land * (eps + ms[k + 1][0])   # anchored hop k+1 restart
        return guided / guided.sum()

    def _fact_relevance(self, query, cos, pf, ce_cache, a):
        cfg = getattr(self._tl, "cfg", None)
        if cfg is not None and cfg.get("hops") and not getattr(self._tl, "in_guided", False):
            self._tl.in_guided = True
            try:
                g = self._guided_facts(self._tl.mode, a)
            finally:
                self._tl.in_guided = False
            lam = cfg["guide_lambda"]
            pf = lam * g + (1.0 - lam) * pf / max(pf.sum(), 1e-300)
        return super()._fact_relevance(query, cos, pf, ce_cache, a)

    # ------------------------------------------------------------------ API
    def retrieve(self, query: str, mode: str = "all", **kwargs):
        """PPRRetriever.retrieve plus the GUIDED_ARGS keys (popped here, visible to the overrides above)."""
        self._tl.cfg = {**GUIDED_ARGS, **{k: kwargs.pop(k) for k in list(kwargs) if k in GUIDED_ARGS}}
        self._tl.mode = mode
        try:
            return super().retrieve(query, mode=mode, **kwargs)
        finally:
            self._tl.cfg = None
