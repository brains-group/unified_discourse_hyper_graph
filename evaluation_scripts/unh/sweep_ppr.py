#!/usr/bin/env python3
"""PPR retriever (nkg/retrieval/ppr_engine.py) sweep on an existing full-corpus UNH graph. No LLM is called here.

    python evaluation_scripts/unh/sweep_ppr.py --dataset hotpotqa --variants base,nochunk --limit-q 250

The graph and embeddings are loaded once; every variant retrieves the same questions (default: the 250 subset
evaluation_data/<ds>/subsets/<ds>_250_seed0.json, in dataset order). For each variant:
  - outputs/<ds>_full/ppr_sweep/contexts_<variant>.json   (main.py's contexts format, so answer.py reads it as is)
  - answer-in-context %  (a normalised gold answer occurs in the normalised context; common/metrics)
  - mean context length and seconds per question
The same numbers are computed for the existing official contexts of the full run (contexts.json etc.), restricted
to the same questions, as the reference.  F1 needs the shared answer step: answer.py / score.py on these files.
"""
import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DATA_ROOT = REPO / "evaluation_data"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE.parent / "common"))

VIEW_MODE = {"full": "all", "hypergraph": "hypergraph", "discourse": "discourse"}
VIEW_CTX = {"full": "contexts.json", "hypergraph": "contexts_hypergraph.json", "discourse": "contexts_discourse.json"}

# variant -> overrides of the retrieve() arguments (DEFAULT_PPR_ARGS keys and the path / context arguments)
VARIANTS = {
    "base":      {},
    "nochunk":   {"passage_node_weight": 0.0},
    "noent":     {"entity_seed_weight": 0.0},
    "noce":      {"ce_seed_rerank": False},
    "pprrank":   {"path_score": "ppr"},
    "mixrank":   {"path_score": "ce+ppr"},
    "pw02":      {"passage_node_weight": 0.2},
    "seeds10":   {"seed_facts": 10},
    "d3":        {"max_depth": 3},
    "k30":       {"final_top_k": 30},
    "k10":       {"final_top_k": 10},
    "nodedup":   {"deduplicate_paths": False},
    "src3":      {"source_passages": 3},
    "ppr3":      {"ppr_passages": 3},
    "ppr5":      {"ppr_passages": 5},
    "d85":       {"damping": 0.85},
    "noscore":   {"ff_use_score": False},
    # v1 = the first version of path building/selection (cross-encoder over all paths); the names above were run
    # with it before the coverage selection existed and now need {"path_select": "ce", "fact_relevance": "ppr"}
    "v1base":    {"path_select": "ce", "fact_relevance": "ppr", "ce_path_cap": 100000},
    # v2: fused fact relevance (CE + PPR ranks) + coverage selection
    "cov":       {},
    "cov_s30":   {"top_k_seeds": 30},
    "cov_s40_b3": {"top_k_seeds": 40, "beam_width": 3},
    "cov_s30_d3": {"top_k_seeds": 30, "max_depth": 3},
    "cov_s30_d2": {"top_k_seeds": 30, "max_depth": 2},
    "cov_relppr": {"top_k_seeds": 30, "fact_relevance": "ppr"},
    "cov_relce":  {"top_k_seeds": 30, "fact_relevance": "ce"},
    "cerank_rrf": {"top_k_seeds": 30, "path_select": "ce"},
    "cov_s30_k30": {"top_k_seeds": 30, "final_top_k": 30},
    "cov_s30_k10": {"top_k_seeds": 30, "final_top_k": 10},
    # (cov_s30_c2000/c3000/c4000 were run with a character budget, since replaced by context_tokens)
    "cov_s30_ppr3": {"top_k_seeds": 30, "ppr_passages": 3},
    # v3: compact layout (default) + token budget (context_tokens, default 2000); cov* above ran with layout legacy
    "c_s30":          {"top_k_seeds": 30},
    "c_s30_d2":       {"top_k_seeds": 30, "max_depth": 2},
    "c_s30_t1000":    {"top_k_seeds": 30, "final_top_k": 80, "context_tokens": 1000},
    "c_s30_t1500":    {"top_k_seeds": 30, "final_top_k": 80, "context_tokens": 1500},
    "c_s30_t2000":    {"top_k_seeds": 30, "final_top_k": 80, "context_tokens": 2000},
    "c_s30_d2_t1000": {"top_k_seeds": 30, "max_depth": 2, "final_top_k": 80, "context_tokens": 1000},
    "c_s30_d2_t1500": {"top_k_seeds": 30, "max_depth": 2, "final_top_k": 80, "context_tokens": 1500},
    "c_s30_d2_t2000": {"top_k_seeds": 30, "max_depth": 2, "final_top_k": 80, "context_tokens": 2000},
    "t1000_dmp06":    {"top_k_seeds": 30, "final_top_k": 80, "context_tokens": 1000, "damping": 0.6},
    "t1000_dmp07":    {"top_k_seeds": 30, "final_top_k": 80, "context_tokens": 1000, "damping": 0.7},
    "t1000_force40":  {"top_k_seeds": 30, "final_top_k": 80, "context_tokens": 1000, "force_top_facts": 40},
    "t1000_r2":       {"top_k_seeds": 30, "final_top_k": 80, "context_tokens": 1000, "ppr_rounds": 2},
    "t1000_all":      {"top_k_seeds": 30, "final_top_k": 80, "context_tokens": 1000, "damping": 0.6,
                       "force_top_facts": 40, "ppr_rounds": 2},
    "t2000_all":      {"top_k_seeds": 30, "final_top_k": 80, "context_tokens": 2000, "damping": 0.6,
                       "force_top_facts": 40, "ppr_rounds": 2},
    # v4: what drives t2000_all (forcing vs damping vs round 2), and the smallest budget that keeps it
    "t2000_f40":      {"top_k_seeds": 30, "final_top_k": 80, "context_tokens": 2000, "force_top_facts": 40},
    "t2000_f60":      {"top_k_seeds": 30, "final_top_k": 80, "context_tokens": 2000, "force_top_facts": 60},
    "t2000_f80":      {"top_k_seeds": 30, "final_top_k": 80, "context_tokens": 2000, "force_top_facts": 80},
    "t2000_f40_dmp06": {"top_k_seeds": 30, "final_top_k": 80, "context_tokens": 2000, "force_top_facts": 40,
                        "damping": 0.6},
    # the selected setting + the top-5 chunks by PPR score appended after the paths (G-Medical / contracts, 2026-10-11)
    "t2000_f40_dmp06_p5": {"top_k_seeds": 30, "final_top_k": 80, "context_tokens": 2000, "force_top_facts": 40,
                           "damping": 0.6, "ppr_passages": 5},
    "t2000_f40_r2":   {"top_k_seeds": 30, "final_top_k": 80, "context_tokens": 2000, "force_top_facts": 40,
                       "ppr_rounds": 2},
    "t1500_f40":      {"top_k_seeds": 30, "final_top_k": 80, "context_tokens": 1500, "force_top_facts": 40},
    "t1500_f60":      {"top_k_seeds": 30, "final_top_k": 80, "context_tokens": 1500, "force_top_facts": 60},
    "t1500_all":      {"top_k_seeds": 30, "final_top_k": 80, "context_tokens": 1500, "damping": 0.6,
                       "force_top_facts": 40, "ppr_rounds": 2},
}
BASE_ARGS = {"top_k_seeds": 12, "max_depth": 4, "beam_width": 6, "final_top_k": 20, "final_mmr_lambda": 0.6,
             "deduplicate_paths": True, "neutral_layout": False, "source_passages": 0}


def parse_variant(name: str) -> dict:
    """A VARIANTS name, or ad-hoc 'key=value+key=value' (values parsed as JSON when possible)."""
    if name in VARIANTS:
        return dict(VARIANTS[name])
    out = {}
    for part in name.split("+"):
        k, v = part.split("=", 1)
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=["hotpotqa", "musique"])
    ap.add_argument("--variants", default="base")
    ap.add_argument("--view", default="full", choices=list(VIEW_MODE))
    ap.add_argument("--subset", default=None, help="ids file (default: <ds>_250_seed0)")
    ap.add_argument("--limit-q", type=int, default=0)
    ap.add_argument("--all-questions", action="store_true",
                    help="all of <ds>.json instead of the subset; files / summary keys get the suffix _q<N>")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out-dir", default=None, help="run dir holding kg.graphml (default outputs/<ds>_full)")
    ap.add_argument("--retrieval-model", default="voyageai/voyage-4-nano")
    ap.add_argument("--cross-encoder-model", default="cross-encoder/ms-marco-MiniLM-L6-v2")
    ap.add_argument("--overwrite", action="store_true", help="re-retrieve variants whose contexts file exists")
    args = ap.parse_args()

    from metrics import get_gold_answers, normalize_answer
    from nkg.retrieval.ppr_engine import PPRRetriever
    from sentence_transformers import CrossEncoder, SentenceTransformer

    out = Path(args.out_dir) if args.out_dir else HERE / "outputs" / f"{args.dataset}_full"
    kg = out / "kg.graphml"
    sub = Path(args.subset) if args.subset else DATA_ROOT / args.dataset / "subsets" / f"{args.dataset}_250_seed0.json"
    samples = json.load(open(DATA_ROOT / args.dataset / f"{args.dataset}.json", encoding="utf-8"))
    if not args.all_questions:
        keep = set(json.load(open(sub))["ids"])
        samples = [s for s in samples if s.get("id", s.get("_id")) in keep]
    if args.limit_q:
        samples = samples[: args.limit_q]
    qid = lambda s: s.get("id", s.get("_id"))  # noqa: E731
    golds = {qid(s): [normalize_answer(g) for g in get_gold_answers(s)] for s in samples}

    def n_facts(ctx):  # distinct fact sentences (formatted as "...") in the path part of a context
        return len(set(re.findall(r'"([^"]{15,})"', ctx.split("\n\nSource passages:")[0])))

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-30B-A3B-Instruct-2507")  # as DEFAULT_PPR_ARGS budget_tokenizer

    def n_tokens(ctx):
        return len(tok(ctx)["input_ids"])

    def edge_shares(rows):
        """Mean per question of the share of path edges that are discourse (fact->fact) vs entity (fact<->entity)
        edges, and of hops (an entity hop = 2 edges = 1 hop); questions whose paths have no edge are skipped."""
        e, h = [], []
        for r in rows:
            ff, en = r["edges_discourse"], r["edges_entity"]
            if ff + en:
                e.append(ff / (ff + en))
                h.append(ff / (ff + en / 2))
        mean = lambda x: round(100 * sum(x) / len(x), 1) if x else None  # noqa: E731
        return {"pct_edges_discourse": mean(e), "pct_edges_entity": None if not e else round(100 - mean(e), 1),
                "pct_hops_discourse": mean(h), "pct_hops_entity": None if not h else round(100 - mean(h), 1),
                "mean_edges_per_question": round(sum(r["edges_discourse"] + r["edges_entity"] for r in rows) / len(rows), 1)}

    def hit(i, ctx):
        n = normalize_answer(ctx)
        return any(g and g in n for g in golds[i])

    sweep_dir = out / "ppr_sweep"
    sweep_dir.mkdir(exist_ok=True)
    summary_path = sweep_dir / f"summary_{args.view}.json"
    summary = json.load(open(summary_path)) if summary_path.exists() else {}

    ref = out / VIEW_CTX[args.view]
    if ref.exists():
        rows = {r["id"]: r for r in json.load(open(ref, encoding="utf-8"))}
        rr = [rows[qid(s)] for s in samples if qid(s) in rows]
        summary["official" + (f"_q{len(rr)}" if args.all_questions else "")] = {"questions": len(rr),
                               "answer_in_context_pct": round(100 * sum(hit(r["id"], r["context_text"]) for r in rr) / max(1, len(rr)), 1),
                               "mean_context_chars": round(sum(len(r["context_text"]) for r in rr) / max(1, len(rr))),
                               "mean_distinct_facts": round(sum(n_facts(r["context_text"]) for r in rr) / max(1, len(rr)), 1),
                               "mean_tokens": round(sum(n_tokens(r["context_text"]) for r in rr) / max(1, len(rr))),
                               "max_tokens": max(n_tokens(r["context_text"]) for r in rr),
                               "mean_seconds": round(sum(r.get("seconds", 0) for r in rr) / max(1, len(rr)), 2)}

    bi = SentenceTransformer(args.retrieval_model, trust_remote_code=True)
    ce = CrossEncoder(args.cross_encoder_model)
    slug = args.retrieval_model.replace("/", "__")
    t0 = time.time()
    retriever = PPRRetriever(
        retrieval_model=bi, cross_encoder=ce, graph_path=str(kg),
        entity_weights={"name": 0.6, "role": 0.4, "anchors": 0.0},
        fact_weights={"sentence": 0.8, "questions": 0.0, "chunk": 0.2}, drop_empty_entities=True,
        embedding_cache_path=str(out / f"kg.{slug}.dropempty.retriever_emb.npz"), model_name=args.retrieval_model,
        chunk_cache_path=str(out / f"kg.{slug}.chunk_emb.npz"))
    print(f"[sweep_ppr] retriever ready in {time.time() - t0:.0f}s", flush=True)

    suffix = f"_q{len(samples)}" if args.all_questions else ""
    for name in args.variants.split(","):
        kw = {**BASE_ARGS, **parse_variant(name), "mode": VIEW_MODE[args.view]}
        name = name + suffix
        cpath = sweep_dir / (f"contexts_{name}.json" if args.view == "full" else f"contexts_{args.view}_{name}.json")
        if cpath.exists() and not args.overwrite:
            print(f"[sweep_ppr] {name}: {cpath.name} exists, skipped (--overwrite to redo)", flush=True)
            continue

        def one(s):
            t = time.time()
            ctx, paths = retriever.retrieve(query=s["question"], return_raw_paths=True, **kw)
            # distinct edges of the final paths (each written once in the context), by type; an entity hop
            # (fact -> entity -> fact) is two edges (fact_entity + entity_fact) and one hop
            seen = {}
            for p in paths:
                for u, et, _, v in p.history:
                    seen.setdefault(frozenset((u, v)), et)
            n_ff = sum(et == "fact_fact" for et in seen.values())
            n_ent = sum(et in ("fact_entity", "entity_fact") for et in seen.values())
            return {"id": qid(s), "question": s["question"], "context_text": ctx, "context_chars": len(ctx),
                    "status": "ok", "error": None, "seconds": round(time.time() - t, 3),
                    "edges_discourse": n_ff, "edges_entity": n_ent}

        done, t1 = {}, time.time()
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(one, s) for s in samples]
            for f in as_completed(futs):
                r = f.result()
                done[r["id"]] = r
        rows = [done[qid(s)] for s in samples]
        json.dump(rows, open(cpath, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        summary[name] = {"args": {k: v for k, v in kw.items()},
                         "questions": len(rows),
                         "answer_in_context_pct": round(100 * sum(hit(r["id"], r["context_text"]) for r in rows) / len(rows), 1),
                         "mean_context_chars": round(sum(r["context_chars"] for r in rows) / len(rows)),
                         "mean_distinct_facts": round(sum(n_facts(r["context_text"]) for r in rows) / len(rows), 1),
                         "mean_tokens": round(sum(n_tokens(r["context_text"]) for r in rows) / len(rows)),
                         "max_tokens": max(n_tokens(r["context_text"]) for r in rows),
                         **edge_shares(rows),
                         "mean_seconds": round(sum(r["seconds"] for r in rows) / len(rows), 3),
                         "wall_seconds": round(time.time() - t1, 1)}
        json.dump(summary, open(summary_path, "w"), indent=2)
        m = summary[name]
        print(f"[sweep_ppr] {name:10s} answer-in-context {m['answer_in_context_pct']:5.1f}%  "
              f"edges disc/ent {m['pct_edges_discourse']}/{m['pct_edges_entity']}%  tokens {m['mean_tokens']:5d} (max {m['max_tokens']:5d})  chars {m['mean_context_chars']:6d}  facts {m['mean_distinct_facts']:5.1f}  {m['mean_seconds']:.2f}s/q  wall {m['wall_seconds']:.0f}s", flush=True)

    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "args"} for k, v in summary.items()}, indent=1))


if __name__ == "__main__":
    main()
