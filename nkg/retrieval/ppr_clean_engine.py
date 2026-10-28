"""
CleanPPRRetriever: AnswerReservePPRRetriever (nkg/retrieval/ppr_answer_engine.py) with two context-cleaning options.
The main PPR engine (nkg/retrieval/ppr_engine.py) is not modified; everything here is overrides in this subclass.

Why: in the MuSiQue min-20 contexts about half of the 2,000-token budget went to filler: unrelated single facts
appended by force_top_facts (e.g. nine strangers' birth dates under a birth-date question), and hub facts that rank
as relevant for almost any question (ISO 3166 facts appeared in 79 of 250 contexts).

1. force_mode (how force_top_facts adds the top-N relevant facts not yet in the context):
     "all"        PPRRetriever's rule: the best candidate path containing the fact, or the fact alone
     "path"       only through a candidate path; a fact that no candidate path contains is not added alone
     "connected"  only through a candidate path that shares a node (fact or entity) with the context selected before
                  the forcing step, i.e. it extends the existing chains instead of opening an unrelated one
2. hub_gamma: the PPR score of every fact is divided by its global PageRank ** hub_gamma before the relevance fusion
   (PPR rank + cross-encoder rank). Global PageRank = PPR with a uniform restart on the same graph and damping, so a
   fact that is central for every query (a hub) loses rank, and a fact that is central for THIS query keeps it
   (pi_q / pi ratio, the standard query-specific vs global centrality correction). 0 = off.
3. passages_first: with ppr_passages / source_passages, the source passages are written before the fact paths
   ("Source passages:" ... then "Fact paths:" ...) instead of after them. Same content, only the order changes.
4. passage_next: with ppr_passages=N, each of the top-N chunks by PPR score also brings the chunk that follows it in
   its document (passage_next=1), as if the corpus had been cut into chunks twice as long. The next chunk is the
   following chunk in corpus order, confirmed by the chunker's token overlap (a chunk's last characters reappear at
   the start of the next one; none across a document boundary; built once per engine). Each run of consecutive selected chunks is merged into one passage with the overlap removed; passages are
   ordered by their best chunk's PPR rank. A chunk with no successor (end of a document) brings nothing.
"""
import threading

import numpy as np

from nkg.retrieval.ppr_answer_engine import AnswerReservePPRRetriever
from nkg.retrieval.ppr_engine import PPRRetriever

CLEAN_ARGS = {
    "force_mode": "all",   # "all" | "path" | "connected"
    "hub_gamma": 0.0,      # PPR score / global PageRank ** hub_gamma (0 = off)
    "passages_first": False,  # source passages before the fact paths
    "passage_next": 0,        # 1: each top PPR passage also brings the chunk after it, merged (see 4.)
}
_PROBE = slice(-80, -30)  # inside the 50-token chunker overlap, away from the decode boundary at the end
PASSAGES_SEP = "\n\nSource passages:\n\n"  # PPRRetriever.retrieve appends passages after this


class CleanPPRRetriever(AnswerReservePPRRetriever):

    def __init__(self, *args, **kwargs):
        self._gpr = {}
        self._gpr_lock = threading.Lock()
        super().__init__(*args, **kwargs)

    def retrieve(self, query: str, mode: str = "all", **kwargs):
        self._tl.clean = {**CLEAN_ARGS, **{k: kwargs.pop(k) for k in list(kwargs) if k in CLEAN_ARGS}}
        self._tl.clean_mode = mode
        first = self._tl.clean["passages_first"]
        n_next = self._tl.clean["passage_next"]
        try:
            if n_next:
                out = self._retrieve_with_next(query, mode, n_next, **kwargs)
            else:
                out = super().retrieve(query, mode=mode, **kwargs)
        finally:
            self._tl.clean = None
        if first:
            ctx, rest = (out[0], out[1:]) if isinstance(out, tuple) else (out, None)
            if PASSAGES_SEP in ctx:
                facts, passages = ctx.split(PASSAGES_SEP, 1)
                ctx = "Source passages:\n\n" + passages + "\n\nFact paths:\n\n" + facts
            out = (ctx, *rest) if rest is not None else ctx
        return out

    # ------------------------------------------------------------------ passage + next chunk
    def _next_chunk(self) -> dict:
        """chunk index -> (index of the chunk that follows it, characters of the follower to skip). The graph keeps
        chunks in corpus order, so the follower of chunk i is chunk i+1 when the chunker's overlap confirms it (a
        document boundary has none). Searching all chunks instead would mis-link exact duplicate chunks."""
        if getattr(self, "_succ", None) is None:
            with self._gpr_lock:
                if getattr(self, "_succ", None) is None:
                    texts = [self.graph.chunks[c].text for c in self.chunk_ids]
                    succ = {}
                    for i in range(len(texts) - 1):
                        a, b = texts[i], texts[i + 1]
                        probe = a[_PROBE]
                        k = b.find(probe, 0, 600) if len(a) >= 200 and len(probe) >= 40 else -1
                        if k >= 0:  # A ends with probe + tail; B continues after probe + tail
                            succ[i] = (i + 1, k + len(probe) + len(a[_PROBE.stop:]))
                    self._succ = succ
        return self._succ

    def _retrieve_with_next(self, query, mode, n_next, **kwargs):
        n = kwargs.pop("ppr_passages", self.ppr_args["ppr_passages"])
        want_paths = kwargs.pop("return_raw_paths", False)
        ctx, dbg = super().retrieve(query, mode=mode, ppr_passages=0, return_debug=True, **kwargs)
        cpos = {c: i for i, c in enumerate(self.chunk_ids)}
        top = [cpos[c] for c in dbg["top_chunks"][:n]]
        succ = self._next_chunk()
        rank = {}
        for r, i in enumerate(top):                      # each top chunk and its n_next followers
            rank.setdefault(i, r)
            for _ in range(n_next):
                if i not in succ:
                    break
                i = succ[i][0]
                rank.setdefault(i, r)
        pred = {j: i for i, (j, _) in succ.items()}
        heads = [i for i in rank if pred.get(i) not in rank]  # first chunk of each run of consecutive selected chunks
        blocks = []
        for h in heads:
            i, text, best, seen = h, self.graph.chunks[self.chunk_ids[h]].text, rank[h], {h}
            while i in succ and succ[i][0] in rank and succ[i][0] not in seen:
                j, skip = succ[i]
                text += self.graph.chunks[self.chunk_ids[j]].text[skip:]
                best, i = min(best, rank[j]), j
                seen.add(j)
            blocks.append((best, text))
        passages = [txt for _, txt in sorted(blocks, key=lambda x: x[0])]
        if passages:
            ctx = ctx + PASSAGES_SEP + "\n\n".join(passages)
        return (ctx, dbg["final_paths"]) if want_paths else ctx

    def _clean(self):
        return getattr(self._tl, "clean", None) or CLEAN_ARGS

    # ------------------------------------------------------------------ hub penalty
    def _global_fact_pagerank(self, mode: str, a: dict) -> np.ndarray:
        key = (mode, a["damping"], tuple(sorted(a["edge_weights"].items())), a["ff_use_score"])
        if key not in self._gpr:
            with self._gpr_lock:
                if key not in self._gpr:
                    pi = PPRRetriever.personalized_pagerank(self, np.ones(self.n_nodes), mode, a["damping"],
                                                            a["edge_weights"], a["ff_use_score"], 200, 1e-12)
                    g = pi[self.f_off:self.e_off]
                    self._gpr[key] = np.maximum(g, g[g > 0].min() if (g > 0).any() else 1e-12)
        return self._gpr[key]

    def _fact_relevance(self, query, cos, pf, ce_cache, a):
        gamma = self._clean()["hub_gamma"]
        if gamma:
            pf = pf / self._global_fact_pagerank(getattr(self._tl, "clean_mode", "all"), a) ** gamma
        return super()._fact_relevance(query, cos, pf, ce_cache, a)

    # ------------------------------------------------------------------ forced facts
    def _path_nodes(self, raw_path) -> set:
        _, facts, hops = raw_path
        return {("f", f) for f in facts} | {("e", e) for kind, e in hops if kind == "ent"}

    def _force_top(self, rel, raw, fact_sets, chosen, covered, remaining, a):
        how = self._clean()["force_mode"]
        if how == "all":
            return super()._force_top(rel, raw, fact_sets, chosen, covered, remaining, a)
        context_nodes = set().union(*(self._path_nodes(raw[i]) for i in chosen)) if chosen else set()
        k = min(a["force_top_facts"], len(rel))
        top = np.argpartition(-rel, k - 1)[:k]
        for f in top[np.argsort(-rel[top], kind="stable")]:
            f = int(f)
            if f in covered:
                continue
            with_f = [i for i in remaining if f in fact_sets[i]]
            if how == "connected":
                with_f = [i for i in with_f if self._path_nodes(raw[i]) & context_nodes]
            if not with_f:
                continue                                   # no (connected) path: not added alone
            best = max(with_f, key=lambda i: (sum(rel[x] for x in fact_sets[i] - covered), -len(raw[i][1])))
            chosen.append(best)
            covered |= fact_sets[best]
            remaining.discard(best)
