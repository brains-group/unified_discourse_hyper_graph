"""
AnswerReservePPRRetriever: GuidedPPRRetriever (nkg/retrieval/ppr_guided_engine.py) plus answer-candidate paths with a
reserved share of the token budget. No LLM, no plan, paths only.

Why: on the 250-question subsets the answer-bearing fact is often reached but lost. Fact relevance scores each fact
against the WHOLE question, and the deepest fact of a chain (the answer) only resembles the end of the question, so
it ranks low (MuSiQue: 19.6% of questions have their answer fact outside the top 150, 11.2% inside it but not
selected into the 2,000-token context).

What changes (endpoint_score set; otherwise GuidedPPRRetriever.retrieve, unchanged):
  1. Endpoint scoring. Every candidate path with >= 2 facts (the top `endpoint_cap` by path relevance) is scored by
     its two ends: the first fact (where the chain starts, which matches the named part of the question) and the
     last fact (the answer candidate):
        pair: cross-encoder(question, first fact + " " + last fact)      (one score for the two ends together)
        sum:  minmax(CE(question, first fact)) + minmax(CE(question, last fact))
        last: CE(question, last fact) only (control)
     The best path per last fact is kept.
  2. Reserved budget. The top `answer_paths` endpoint-scored paths go first into the context, up to
     `answer_reserve` x context_tokens; the coverage selection (PPRRetriever's rule) then fills the rest, counting the
     reserved facts as covered. When the context is over budget, only non-reserved paths are dropped.
  0. complete_paths=True (before 1): the beam search starts from the top `top_k_seeds` facts and always extends to
     the most relevant neighbours, so many pool facts (relevance rank ~40-150) end no path at all; such a fact
     could never be an answer candidate. Each pool fact no beam path contains gets one extra path: the shortest
     chain (breadth-first over the pool, fact->fact edges or a shared entity, as the beam search moves) from the
     start facts to it, at most max_depth facts long. complete_max caps how many (the highest-relevance pool facts
     first); complete_in_coverage=False lets them into the context only through the answer reservation (2), so they
     cannot crowd the coverage selection.
  3. force_top_facts (PPRRetriever's option) is applied after the coverage selection, as there: every top-N relevant
     fact not yet in the context gets its best candidate path (or itself alone), last, so the budget cuts them first.
"""
import numpy as np

from collections import deque

from nkg.retrieval.ppr_engine import _VIEW_HOPS, DEFAULT_PPR_ARGS, format_paths_compact
from nkg.retrieval.ppr_guided_engine import GUIDED_ARGS, GuidedPPRRetriever

ANSWER_ARGS = {
    "endpoint_score": None,    # None (off) | "pair" | "sum" | "last"
    "endpoint_cap": 150,       # multi-fact paths scored by their endpoints (top by path relevance)
    "answer_paths": 5,         # endpoint-ranked paths placed first
    "answer_reserve": 0.25,    # share of context_tokens they may use
    "complete_paths": False,   # every pool fact no beam path reached gets a path back to its nearest start fact
    "complete_max": 0,         # at most this many completion paths, highest-relevance pool facts first (0 = all)
    "complete_in_coverage": True,  # False: completion paths enter only through the answer reservation
}


class AnswerReservePPRRetriever(GuidedPPRRetriever):

    def retrieve(self, query: str, mode: str = "all", **kwargs):
        ans = {**ANSWER_ARGS, **{k: kwargs.pop(k) for k in list(kwargs) if k in ANSWER_ARGS}}
        if not ans["endpoint_score"]:
            return super().retrieve(query, mode=mode, **kwargs)
        self._tl.cfg = {**GUIDED_ARGS, **{k: kwargs.pop(k) for k in list(kwargs) if k in GUIDED_ARGS}}
        self._tl.mode = mode
        try:
            return self._retrieve_reserved(query, mode, ans, **kwargs)
        finally:
            self._tl.cfg = None

    def _complete_paths(self, raw, rel, mode, start_k, max_depth, n_relevant, max_paths=0):
        """Shortest chains from the start facts to every pool fact no beam path contains (module docstring, 0)."""
        use_ff, use_ent = _VIEW_HOPS[mode]
        nf = len(rel)
        n_relevant = min(n_relevant, nf)
        pool = np.argpartition(-rel, n_relevant - 1)[:n_relevant]
        pool = pool[np.argsort(-rel[pool], kind="stable")]
        in_pool = np.zeros(nf, dtype=bool)
        in_pool[pool] = True
        reached = set().union(*(set(f) for _, f, _ in raw)) if raw else set()

        def neighbours(f):  # as PPRRetriever._build_paths: most specific shared entity, fact->fact edges preferred
            out = {}
            if use_ent:
                ents = self.fe.indices[self.fe.indptr[f]:self.fe.indptr[f + 1]]
                for e in ents[np.argsort(self.ent_n_facts[ents], kind="stable")[::-1]]:
                    fs = self.ef.indices[self.ef.indptr[e]:self.ef.indptr[e + 1]]
                    for g2 in fs[in_pool[fs]]:
                        if g2 != f:
                            out[int(g2)] = ("ent", int(e))
            if use_ff:
                fs = self.ff_score.indices[self.ff_score.indptr[f]:self.ff_score.indptr[f + 1]]
                for g2 in fs[in_pool[fs]]:
                    out[int(g2)] = ("ff", -1)
            return out

        parent = {int(f): None for f in pool[:start_k]}          # BFS from all start facts, best first
        depth = {f: 1 for f in parent}
        queue = deque(parent)
        while queue:
            f = queue.popleft()
            if depth[f] >= max_depth:
                continue
            for g2, hop in sorted(neighbours(f).items(), key=lambda x: -rel[x[0]]):
                if g2 not in parent:
                    parent[g2] = (f, hop)
                    depth[g2] = depth[f] + 1
                    queue.append(g2)
        out = []
        for f in pool:
            f = int(f)
            if f in reached or parent.get(f) is None:
                continue
            facts, hops = [f], []
            while parent[facts[0]] is not None:
                prev, hop = parent[facts[0]]
                facts.insert(0, prev)
                hops.insert(0, hop)
            out.append((float(sum(rel[x] for x in facts)), facts, hops))
            if max_paths and len(out) >= max_paths:
                break
        return out

    def _force_top(self, rel, raw, fact_sets, chosen, covered, remaining, a):
        """PPRRetriever.retrieve's force_top_facts: every top-N relevant fact not yet in the context gets its best
        candidate path (or itself alone), appended to chosen (in place)."""
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

    def _endpoint_scores(self, query, raw, rel, ans, ce_cache):
        """{path index: endpoint score} for the best path per last fact among the top multi-fact paths."""
        multi = [i for i in range(len(raw)) if len(raw[i][1]) >= 2]
        multi = sorted(multi, key=lambda i: -sum(rel[f] for f in raw[i][1]))[: ans["endpoint_cap"]]
        if not multi:
            return {}
        sent = lambda f: self.graph.facts[self.fact_ids[f]].sentence  # noqa: E731
        how = ans["endpoint_score"]
        if how == "pair":
            sc = np.asarray(self.cross_encoder.predict(
                [[query, sent(raw[i][1][0]) + " " + sent(raw[i][1][-1])] for i in multi], show_progress_bar=False),
                dtype=np.float64)
        else:
            ends = sorted({raw[i][1][-1] for i in multi} | ({raw[i][1][0] for i in multi} if how == "sum" else set()))
            ce = dict(zip(ends, self._ce_facts(query, ends, ce_cache)))
            vals = np.array(list(ce.values()))
            lo, hi = vals.min(), vals.max()
            nrm = {f: (v - lo) / (hi - lo) if hi > lo else 1.0 for f, v in ce.items()}
            sc = np.array([nrm[raw[i][1][-1]] + (nrm[raw[i][1][0]] if how == "sum" else 0.0) for i in multi])
        best = {}
        for i, s in zip(multi, sc):
            last = raw[i][1][-1]
            if last not in best or s > best[last][1]:
                best[last] = (i, float(s))
        return {i: s for i, s in best.values()}

    def _retrieve_reserved(self, query, mode, ans, top_k_seeds=12, max_depth=4, beam_width=6, final_top_k=20,
                           return_raw_paths=False, **kwargs):
        a = {**self.ppr_args, **{k: v for k, v in kwargs.items() if k in DEFAULT_PPR_ARGS}}
        qv = self._encode_norm([query])[0]
        reset, seeds, cos, ce_cache = self._reset_vector(query, qv, a)
        pi = self.personalized_pagerank(reset, mode, a["damping"], a["edge_weights"], a["ff_use_score"],
                                        a["ppr_max_iter"], a["ppr_tol"])
        pf = pi[self.f_off:self.e_off]
        rel = self._fact_relevance(query, cos, pf, ce_cache, a)
        raw = self._build_paths(rel, mode, top_k_seeds, beam_width, a["global_beam"], max_depth, a["relevant_facts"])
        n_beam = len(raw)
        if ans["complete_paths"]:
            raw = raw + self._complete_paths(raw, rel, mode, top_k_seeds, max_depth, a["relevant_facts"],
                                             ans["complete_max"])
        fact_sets = [set(f) for _, f, _ in raw]
        paths = {}

        def path(i):
            if i not in paths:
                paths[i] = self._to_path(raw[i][1], raw[i][2])
            return paths[i]

        tok = self._tokenizer(a["budget_tokenizer"]) if a["context_tokens"] else None
        ntok = lambda sel: len(tok(format_paths_compact(self.graph, [path(i) for i in sel]))["input_ids"])  # noqa: E731

        # 1-2. answer-candidate paths, first, within their share of the budget
        end = self._endpoint_scores(query, raw, rel, ans, ce_cache)
        reserved, covered = [], set()
        for i in sorted(end, key=lambda i: -end[i])[: ans["answer_paths"]]:
            if tok is not None and reserved and ntok(reserved + [i]) > ans["answer_reserve"] * a["context_tokens"]:
                break
            reserved.append(i)
            covered |= fact_sets[i]

        # coverage selection (PPRRetriever's rule) for the rest
        chosen = list(reserved)
        remaining = set(range(len(raw) if ans["complete_in_coverage"] else n_beam)) - set(reserved)
        while remaining and len(chosen) < final_top_k:
            best, best_key = None, None
            for i in remaining:
                key = (sum(rel[f] for f in fact_sets[i] - covered), -len(raw[i][1]))
                if best_key is None or key > best_key:
                    best, best_key = i, key
            if best_key[0] <= 0:
                break
            chosen.append(best)
            covered |= fact_sets[best]
            remaining.discard(best)
        if a["force_top_facts"]:
            self._force_top(rel, raw, fact_sets, chosen, covered, remaining, a)

        context = format_paths_compact(self.graph, [path(i) for i in chosen])
        if tok is not None:
            while len(chosen) > len(reserved) and len(tok(context)["input_ids"]) > a["context_tokens"]:
                chosen = chosen[:-1]                       # drop the lowest-priority non-reserved path
                context = format_paths_compact(self.graph, [path(i) for i in chosen])
        final_paths = [path(i) for i in chosen]
        if return_raw_paths:
            return context, final_paths
        return context
