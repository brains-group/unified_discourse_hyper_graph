#!/usr/bin/env python3
"""Retrieval-only sweep for UNH on an existing graph: does the gold answer appear in the context, and how long is it?

    python evaluation_scripts/unh/sweep_retrieval.py --dataset hotpotqa --limit-q 100

No answer step and no judge: for every question the query plan (UNH's one query-time LLM call) is made ONCE and
shared by every variant (Retriever.retrieve(precomputed_plan=...)), so variants differ only in their retrieval knobs.
Metric: answer-in-context = some gold answer (common/metrics.get_gold_answers, normalised as in scoring) occurs in
the normalised context string. Questions are the first --limit-q of the 250 subset, in dataset order.

Variants (old engine, DEFAULT_RETRIEVAL_ARGS_OLD, neutral_layout off, view full):
  keep_k20       the official run's settings, empty-name entities kept
  k20            empty-name entities dropped at query time (Retriever(drop_empty_entities=True))
  k20_p3/k20_p5  + the source text of the top 3 / 5 chunks of the facts in the final paths (source_passages)
  k40            final_top_k 40, top_k_seeds 15
  k40_p5         k40 + 5 source passages
Output: outputs/<ds>_250_seed0/sweep/sweep_<ds>_<n>q.json (summary + per-question hits / context lengths).
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DATA_ROOT = REPO / "evaluation_data"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE.parent / "common"))

VARIANTS = {
    "keep_k20": dict(drop=False, final_top_k=20, top_k_seeds=9, source_passages=0),
    "k20":      dict(drop=True, final_top_k=20, top_k_seeds=9, source_passages=0),
    "k20_p3":   dict(drop=True, final_top_k=20, top_k_seeds=9, source_passages=3),
    "k20_p5":   dict(drop=True, final_top_k=20, top_k_seeds=9, source_passages=5),
    "k40":      dict(drop=True, final_top_k=40, top_k_seeds=15, source_passages=0),
    "k40_p5":   dict(drop=True, final_top_k=40, top_k_seeds=15, source_passages=5),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=["hotpotqa", "musique"])
    ap.add_argument("--limit-q", type=int, default=100)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out-dir", default=None, help="run dir holding kg.graphml (default outputs/<ds>_250_seed0)")
    ap.add_argument("--model", default=os.environ.get("LLM_MODEL", "deepseek-flash"))
    ap.add_argument("--base-url", default=os.environ.get("LLM_BASE_URL", "https://api.deepseek.com"))
    args = ap.parse_args()

    import experiments.systems.nkg.hqa_musique_evaluation as H
    from metrics import get_gold_answers, normalize_answer
    from nkg.retrieval.engine import Retriever
    from sentence_transformers import CrossEncoder, SentenceTransformer

    out = Path(args.out_dir) if args.out_dir else HERE / "outputs" / f"{args.dataset}_250_seed0"
    kg = out / "kg.graphml"
    ids = json.load(open(DATA_ROOT / args.dataset / "subsets" / f"{args.dataset}_250_seed0.json"))["ids"]
    keep = set(ids)
    samples = [s for s in json.load(open(DATA_ROOT / args.dataset / f"{args.dataset}.json", encoding="utf-8"))
               if s.get("id", s.get("_id")) in keep][: args.limit_q]
    variants = {v: VARIANTS[v] for v in args.variants.split(",")}

    H.configure_dspy(model=f"deepseek/{args.model}", api_base=args.base_url, api_key=os.environ["DEEPSEEK_API_KEY"],
                     max_tokens=8000, num_retries=1, temperature=None)
    bi = SentenceTransformer("voyageai/voyage-4-nano", trust_remote_code=True)
    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")
    weights = dict(entity_weights={"name": 0.6, "role": 0.4, "anchors": 0.0},
                   fact_weights={"sentence": 0.8, "questions": 0.0, "chunk": 0.2})
    retrievers = {}
    for drop in sorted({v["drop"] for v in variants.values()}):
        t0 = time.time()
        retrievers[drop] = Retriever(retrieval_model=bi, cross_encoder=ce, graph_path=str(kg),
                                     drop_empty_entities=drop, **weights)
        print(f"[sweep] retriever drop_empty_entities={drop} ready in {time.time() - t0:.0f}s "
              f"(dropped {retrievers[drop].dropped_empty_entities} entities)", flush=True)

    base = dict(H.DEFAULT_RETRIEVAL_ARGS_OLD)
    base.update(neutral_layout=False, mode="all")

    def one(s):
        q = s["question"]
        golds = [normalize_answer(g) for g in get_gold_answers(s)]
        plan = retrievers[next(iter(retrievers))].planner(user_query=q)
        row = {"id": s.get("id", s.get("_id"))}
        for name, v in variants.items():
            a = dict(base, final_top_k=v["final_top_k"], top_k_seeds=v["top_k_seeds"])
            ctx = retrievers[v["drop"]].retrieve(query=q, precomputed_plan=plan,
                                                 source_passages=v["source_passages"], **a)
            ctx = ctx if isinstance(ctx, str) else str(ctx)
            norm = normalize_answer(ctx)
            row[name] = {"hit": any(g and g in norm for g in golds), "chars": len(ctx)}
        return row

    rows, errors, t0 = [], 0, time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(one, s) for s in samples]
        for i, f in enumerate(as_completed(futs), 1):
            try:
                rows.append(f.result())
            except Exception as exc:  # noqa: BLE001
                errors += 1
                print(f"[sweep] error: {exc!r}", flush=True)
            if i % 10 == 0 or i == len(futs):
                print(f"[sweep] {i}/{len(futs)}  errors={errors}  {time.time() - t0:.0f}s", flush=True)

    summary = {}
    for name in variants:
        hits = [r[name]["hit"] for r in rows]
        chars = [r[name]["chars"] for r in rows]
        summary[name] = {**variants[name], "answer_in_context_pct": round(100 * sum(hits) / max(1, len(hits)), 1),
                         "mean_context_chars": round(sum(chars) / max(1, len(chars)))}
    (out / "sweep").mkdir(exist_ok=True)
    path = out / "sweep" / f"sweep_{args.dataset}_{args.limit_q}q.json"
    json.dump({"dataset": args.dataset, "questions": len(rows), "errors": errors, "summary": summary, "rows": rows},
              open(path, "w"), indent=2)
    print(f"\n{args.dataset}: {len(rows)} questions, {errors} errors -> {path}")
    for name, m in summary.items():
        print(f"  {name:10s} answer-in-context {m['answer_in_context_pct']:5.1f}%   context {m['mean_context_chars']:6d} chars")


if __name__ == "__main__":
    main()
