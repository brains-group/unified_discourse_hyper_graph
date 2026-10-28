#!/usr/bin/env python3
"""E1 / E2 of PPR_HOP_GUIDED_PLAN.md on an existing full-corpus UNH graph (nkg/retrieval/ppr_guided_engine.py).
No LLM is called here.

    python evaluation_scripts/unh/sweep_ppr_guided.py --dataset musique --variants g_base,a_ent,orc_mask

E1  typed transitions: per-node edge-type shares (type_alpha), entity hops as one step (entity_transit), damping.
E2  oracle hop plan (MuSiQue only): each question's gold `question_decomposition` sub-questions as the hop texts of
    the cascade, with the "#k" references masked (orc_*) or replaced by the gold intermediate answers (gold_*: an
    upper bound that leaks the bridge entities; never a result, only a ceiling).

Every variant starts from BASE (PPRRetriever's current best path settings: 30 path starts, compact layout, token
budget) and retrieves the same questions (default: the 250 subset). Per variant:
  ppr_guided/contexts_<variant>.json   (main.py's contexts format, so answer.py reads it as is)
  answer-in-context %, supporting-passage recall (a gold supporting passage counts as retrieved when at least one
  fact of its chunk is in the final paths: % of questions with ALL supports, and the mean fraction), context
  characters and seconds per question; the official run's contexts on the same questions as the reference.
"""
import argparse
import json
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

BASE = {"top_k_seeds": 30, "final_top_k": 80, "context_tokens": 2000}
T_ENT = {"fact_fact": 0.3, "entity_fact": 0.55, "chunk_fact": 0.15}
T_DISC = {"fact_fact": 0.55, "entity_fact": 0.3, "chunk_fact": 0.15}
T_EQ = {"fact_fact": 0.5, "entity_fact": 0.5, "chunk_fact": 0.0}
VARIANTS = {
    # E1
    "g_base":      {},
    "d06":         {"damping": 0.6},
    "d07":         {"damping": 0.7},
    "a_ent":       {"type_alpha": T_ENT},
    "a_disc":      {"type_alpha": T_DISC},
    "a_nochunk":   {"type_alpha": T_EQ},
    "transit":     {"entity_transit": True},
    "transit_d06": {"entity_transit": True, "damping": 0.6},
    "transit_d07": {"entity_transit": True, "damping": 0.7},
    "a_ent_transit_d06": {"type_alpha": T_ENT, "entity_transit": True, "damping": 0.6},
    # E2 (MuSiQue): oracle hops
    "orc_multiseed": {"hops": "mask", "hop_seed_mode": "multiseed"},
    "orc_mask":      {"hops": "mask"},
    "orc_mask_l07":  {"hops": "mask", "guide_lambda": 0.7},
    "orc_mask_l03":  {"hops": "mask", "guide_lambda": 0.3},
    "orc_mask_nobias": {"hops": "mask", "guide_bias": False},
    "orc_mask_sd07": {"hops": "mask", "stage_damping": 0.7},
    "orc_mask_transit": {"hops": "mask", "entity_transit": True},
    "gold_mask":     {"hops": "gold"},
    # answer-candidate paths scored by their endpoints, with a reserved budget share (ppr_answer_engine.py), on top of
    # the best structural settings per dataset (HotpotQA a_ent_transit_d06, MuSiQue transit_d06; see ANSWER_BASE)
    "ar_pair_r25_n5":  {"endpoint_score": "pair", "answer_reserve": 0.25, "answer_paths": 5},
    "ar_pair_r40_n10": {"endpoint_score": "pair", "answer_reserve": 0.40, "answer_paths": 10},
    "ar_sum_r25_n5":   {"endpoint_score": "sum", "answer_reserve": 0.25, "answer_paths": 5},
    "ar_sum_r40_n10":  {"endpoint_score": "sum", "answer_reserve": 0.40, "answer_paths": 10},
    "ar_last_r25_n5":  {"endpoint_score": "last", "answer_reserve": 0.25, "answer_paths": 5},
    # + completed paths: every pool fact the beam search missed gets a chain from the start facts
    "arc_pair_r25_n5":  {"endpoint_score": "pair", "answer_reserve": 0.25, "answer_paths": 5, "complete_paths": True},
    "arc_pair_r40_n10": {"endpoint_score": "pair", "answer_reserve": 0.40, "answer_paths": 10, "complete_paths": True},
    "arc_sum_r25_n5":   {"endpoint_score": "sum", "answer_reserve": 0.25, "answer_paths": 5, "complete_paths": True},
    "arc_sum_r40_n10":  {"endpoint_score": "sum", "answer_reserve": 0.40, "answer_paths": 10, "complete_paths": True},
    "arc_last_r40_n10": {"endpoint_score": "last", "answer_reserve": 0.40, "answer_paths": 10, "complete_paths": True},
    # capped completion on top of the force-top-40 PPR setting (FT40: other session's best reader run); no ANSWER_BASE
    "ft40":            {"force_top_facts": 40, "damping": 0.6},
    "ft40_cc5_res":    {"force_top_facts": 40, "damping": 0.6, "endpoint_score": "pair", "answer_reserve": 0.25,
                        "answer_paths": 5, "complete_paths": True, "complete_max": 5, "complete_in_coverage": False},
    "ft40_call_res":   {"force_top_facts": 40, "damping": 0.6, "endpoint_score": "pair", "answer_reserve": 0.25,
                        "answer_paths": 5, "complete_paths": True, "complete_in_coverage": False},
    "ft40_cc10":       {"force_top_facts": 40, "damping": 0.6, "endpoint_score": "pair", "answer_reserve": 0.25,
                        "answer_paths": 5, "complete_paths": True, "complete_max": 10},
    "ft40_cc20":       {"force_top_facts": 40, "damping": 0.6, "endpoint_score": "pair", "answer_reserve": 0.25,
                        "answer_paths": 5, "complete_paths": True, "complete_max": 20},
    # context cleaning (ppr_clean_engine.py) on top of ft40_cc5_res
    "cl_fpath":        {"force_mode": "path"},
    "cl_fconn":        {"force_mode": "connected"},
    "cl_hub05":        {"hub_gamma": 0.5},
    "cl_hub1":         {"hub_gamma": 1.0},
    "cl_fconn_hub05":  {"force_mode": "connected", "hub_gamma": 0.5},
}
CLEAN_BASE = "ft40_cc5_res"
ANSWER_BASE = {"hotpotqa": "a_ent_transit_d06", "musique": "transit_d06"}


def resolve_variant(name: str, dataset: str) -> dict:
    """A VARIANTS entry with the base it builds on (ar_/arc_: ANSWER_BASE; cl_: CLEAN_BASE)."""
    v = dict(VARIANTS[name])
    if name.startswith(("ar_", "arc_")):
        v = {**VARIANTS[ANSWER_BASE[dataset]], **v}
    if name.startswith("cl_"):
        v = {**VARIANTS[CLEAN_BASE], **v}
    return v


def hop_texts(sample, how: str):
    """Sub-questions of a MuSiQue question_decomposition; '#k' masked ('it') or replaced by the gold answer k."""
    dec = sample.get("question_decomposition") or []
    out = []
    for d in dec:
        q = d["question"].replace(">>", " ")
        if how == "gold":
            q = re.sub(r"#(\d+)", lambda m: dec[int(m.group(1)) - 1]["answer"] if int(m.group(1)) <= len(dec) else "it", q)
        else:
            q = re.sub(r"#\d+", "it", q)
        out.append(" ".join(q.split()))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=["hotpotqa", "musique"])
    ap.add_argument("--variants", default="g_base")
    ap.add_argument("--subset", default=None)
    ap.add_argument("--limit-q", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    from metrics import get_gold_answers, normalize_answer
    from nkg.retrieval.ppr_clean_engine import CleanPPRRetriever
    from sentence_transformers import CrossEncoder, SentenceTransformer

    out = Path(args.out_dir) if args.out_dir else HERE / "outputs" / f"{args.dataset}_full"
    sub = Path(args.subset) if args.subset else DATA_ROOT / args.dataset / "subsets" / f"{args.dataset}_250_seed0.json"
    keep = set(json.load(open(sub))["ids"])
    qid = lambda s: s.get("id", s.get("_id"))  # noqa: E731
    samples = [s for s in json.load(open(DATA_ROOT / args.dataset / f"{args.dataset}.json", encoding="utf-8"))
               if qid(s) in keep]
    if args.limit_q:
        samples = samples[: args.limit_q]
    golds = {qid(s): [normalize_answer(g) for g in get_gold_answers(s)] for s in samples}

    bi = SentenceTransformer("voyageai/voyage-4-nano", trust_remote_code=True)
    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")
    slug = "voyageai__voyage-4-nano"
    t0 = time.time()
    r = CleanPPRRetriever(
        retrieval_model=bi, cross_encoder=ce, graph_path=str(out / "kg.graphml"),
        entity_weights={"name": 0.6, "role": 0.4, "anchors": 0.0},
        fact_weights={"sentence": 0.8, "questions": 0.0, "chunk": 0.2}, drop_empty_entities=True,
        embedding_cache_path=str(out / f"kg.{slug}.dropempty.retriever_emb.npz"), model_name="voyageai/voyage-4-nano",
        chunk_cache_path=str(out / f"kg.{slug}.chunk_emb.npz"))
    print(f"[guided] retriever ready in {time.time() - t0:.0f}s", flush=True)

    # gold supporting passages -> chunk rows (HotpotQA: by title; MuSiQue: by paragraph text, titles repeat)
    norm = lambda t: " ".join(re.sub(r"[^a-z0-9 ]", " ", t.lower()).split())  # noqa: E731
    by_title, by_text = {}, {}
    for i, c in enumerate(r.chunk_ids):
        text = r.graph.chunks[c].text.strip()
        if text.startswith("Title:"):                       # UNH chunk text: "Title: <title>\n\n<paragraph>"
            text = text[len("Title:"):].strip()
        head, _, body = text.partition("\n")
        by_title.setdefault(norm(head), i)
        by_text.setdefault(norm(body.strip() or text)[:150], i)
    supports, unmatched = {}, 0
    for s in samples:
        if args.dataset == "hotpotqa":
            rows = {by_title.get(norm(t)) for t, _ in s["supporting_facts"]}
        else:
            rows = {by_text.get(norm(p["paragraph_text"])[:150]) for p in s["paragraphs"] if p["is_supporting"]}
        unmatched += None in rows
        supports[qid(s)] = {x for x in rows if x is not None}
    print(f"[guided] supporting passages matched to chunks; {unmatched} questions with an unmatched support", flush=True)
    chunk_row = {c: i for i, c in enumerate(r.chunk_ids)}

    gdir = out / "ppr_guided"
    gdir.mkdir(exist_ok=True)
    spath = gdir / "summary.json"
    summary = json.load(open(spath)) if spath.exists() else {}

    def score_rows(rows):
        n = len(rows)
        hit = sum(any(g and g in normalize_answer(x["context_text"]) for g in golds[x["id"]]) for x in rows)
        full = sum(bool(supports[x["id"]]) and supports[x["id"]] <= set(x.get("chunks", [])) for x in rows)
        frac = sum(len(supports[x["id"]] & set(x.get("chunks", []))) / max(1, len(supports[x["id"]])) for x in rows)
        return {"questions": n, "answer_in_context_pct": round(100 * hit / n, 1),
                "all_supports_pct": round(100 * full / n, 1), "support_recall_pct": round(100 * frac / n, 1),
                "mean_context_chars": round(sum(len(x["context_text"]) for x in rows) / n)}

    official = out / "contexts.json"
    if official.exists():
        rows = {x["id"]: x for x in json.load(open(official, encoding="utf-8"))}
        rr = [{"id": qid(s), "context_text": rows[qid(s)]["context_text"]} for s in samples if qid(s) in rows]
        m = score_rows(rr)
        m.pop("all_supports_pct"), m.pop("support_recall_pct")         # official contexts keep no fact ids
        summary["official"] = m

    for name in args.variants.split(","):
        cpath = gdir / f"contexts_{name}.json"
        if cpath.exists() and not args.overwrite:
            print(f"[guided] {name}: exists, skipped", flush=True)
            continue
        v = dict(VARIANTS[name])
        if name.startswith(("ar_", "arc_")):
            v = {**VARIANTS[ANSWER_BASE[args.dataset]], **v}
        if name.startswith("cl_"):
            v = {**VARIANTS[CLEAN_BASE], **v}
        hops_how = v.pop("hops", None)
        if hops_how and args.dataset != "musique":
            print(f"[guided] {name}: oracle hops need MuSiQue, skipped", flush=True)
            continue
        kw = {**BASE, **v, "mode": "all", "return_raw_paths": True}

        def one(s):
            t = time.time()
            extra = {"hops": hop_texts(s, hops_how)} if hops_how else {}
            ctx, paths = r.retrieve(query=s["question"], **kw, **extra)
            facts = set()
            for p in paths:
                nodes = [p.history[0][0]] + [h[3] for h in p.history] if p.history else [p.current_node]
                facts.update(n for n in nodes if n in r.graph.facts)
            chunks = sorted({chunk_row[c] for f in facts for c in r.graph.network.predecessors(f) if c in chunk_row})
            return {"id": qid(s), "question": s["question"], "context_text": ctx, "context_chars": len(ctx),
                    "status": "ok", "error": None, "seconds": round(time.time() - t, 3), "chunks": chunks,
                    "hops": extra.get("hops")}

        done, t1 = {}, time.time()
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for f in as_completed([ex.submit(one, s) for s in samples]):
                x = f.result()
                done[x["id"]] = x
        rows = [done[qid(s)] for s in samples]
        json.dump(rows, open(cpath, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        m = score_rows(rows)
        m.update(mean_seconds=round(sum(x["seconds"] for x in rows) / len(rows), 2),
                 wall_seconds=round(time.time() - t1), args={k: v for k, v in kw.items() if k != "return_raw_paths"},
                 hops=hops_how)
        summary[name] = m
        json.dump(summary, open(spath, "w"), indent=2)
        print(f"[guided] {name:18s} AIC {m['answer_in_context_pct']:5.1f}%  all-supports {m['all_supports_pct']:5.1f}%  "
              f"support-recall {m['support_recall_pct']:5.1f}%  chars {m['mean_context_chars']:5d}  "
              f"{m['mean_seconds']:.2f}s/q", flush=True)


if __name__ == "__main__":
    main()
