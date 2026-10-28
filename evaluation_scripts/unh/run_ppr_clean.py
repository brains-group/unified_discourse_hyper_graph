#!/usr/bin/env python3
"""Retrieve every question of a dataset with a sweep_ppr_guided.py variant (CleanPPRRetriever,
nkg/retrieval/ppr_clean_engine.py; the main PPR engine is not used) and write main.py's contexts format, so
common/answer.py and the scorers read it unchanged. No LLM is called here.

    python evaluation_scripts/unh/run_ppr_clean.py --dataset musique --run-dir evaluation_scripts/unh/outputs/musique_full \
        --variant cl_hub1 --workers 32
    -> <run-dir>/contexts_<variant>.json   (view full; every question of evaluation_data/<ds>/<ds>.json, in order)
    gmedical / quality: --questions <inputs>/questions.json (default evaluation_data/gmedical/inputs/questions.json)
"""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=["hotpotqa", "musique", "gmedical", "quality"])
    ap.add_argument("--run-dir", required=True, help="holds kg.graphml (and the embedding caches)")
    ap.add_argument("--variant", required=True, help="a sweep_ppr_guided.py VARIANTS name")
    ap.add_argument("--questions", default=None)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--limit-q", type=int, default=0)
    args = ap.parse_args()

    from sweep_ppr_guided import BASE, resolve_variant
    from nkg.retrieval.ppr_clean_engine import CleanPPRRetriever
    from sentence_transformers import CrossEncoder, SentenceTransformer

    out = Path(args.run_dir)
    if args.dataset in ("gmedical", "quality"):
        qpath = Path(args.questions or REPO / "evaluation_data" / args.dataset / "inputs" / "questions.json")
        samples = json.load(open(qpath, encoding="utf-8"))
    else:
        samples = json.load(open(REPO / "evaluation_data" / args.dataset / f"{args.dataset}.json", encoding="utf-8"))
    if args.limit_q:
        samples = samples[: args.limit_q]
    qid = lambda s: s.get("id", s.get("_id"))  # noqa: E731
    kw = {**BASE, **resolve_variant(args.variant, "musique" if args.dataset != "hotpotqa" else "hotpotqa"),
          "mode": "all"}
    slug = "voyageai__voyage-4-nano"
    t0 = time.time()
    r = CleanPPRRetriever(
        retrieval_model=SentenceTransformer("voyageai/voyage-4-nano", trust_remote_code=True),
        cross_encoder=CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2"), graph_path=str(out / "kg.graphml"),
        entity_weights={"name": 0.6, "role": 0.4, "anchors": 0.0},
        fact_weights={"sentence": 0.8, "questions": 0.0, "chunk": 0.2}, drop_empty_entities=True,
        embedding_cache_path=str(out / f"kg.{slug}.dropempty.retriever_emb.npz"), model_name="voyageai/voyage-4-nano",
        chunk_cache_path=str(out / f"kg.{slug}.chunk_emb.npz"))
    print(f"[run_ppr_clean] retriever ready in {time.time() - t0:.0f}s; {len(samples)} questions; args={kw}", flush=True)
    tok = r._tokenizer(kw.get("budget_tokenizer", r.ppr_args["budget_tokenizer"]))

    def one(s):
        t = time.time()
        try:
            ctx = r.retrieve(query=s["question"], **kw)
            return {"id": qid(s), "question": s["question"], "context_text": ctx, "context_chars": len(ctx),
                    "status": "ok", "error": None, "seconds": round(time.time() - t, 2),
                    "context_tokens": len(tok(ctx)["input_ids"])}
        except Exception as exc:  # noqa: BLE001
            return {"id": qid(s), "question": s["question"], "context_text": "", "context_chars": 0,
                    "status": "error", "error": repr(exc), "seconds": round(time.time() - t, 2)}

    done, t1 = {}, time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, f in enumerate(as_completed([ex.submit(one, s) for s in samples]), 1):
            x = f.result()
            done[x["id"]] = x
            if i % 100 == 0 or i == len(samples):
                print(f"[run_ppr_clean] {i}/{len(samples)}  errors={sum(v['status'] != 'ok' for v in done.values())}  "
                      f"{time.time() - t1:.0f}s", flush=True)
    rows = [done[qid(s)] for s in samples]
    path = out / f"contexts_{args.variant}.json"
    json.dump(rows, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    ok = [x for x in rows if x["status"] == "ok"]
    print(f"[run_ppr_clean] wrote {path}: {len(ok)}/{len(rows)} ok, mean {sum(x['context_chars'] for x in ok) / max(1, len(ok)):.0f} "
          f"chars, {sum(x['context_tokens'] for x in ok) / max(1, len(ok)):.0f} tokens, {sum(x['seconds'] for x in ok) / max(1, len(ok)):.2f} s/q", flush=True)
    if len(ok) < len(rows):
        sys.exit(f"{len(rows) - len(ok)} questions failed")


if __name__ == "__main__":
    main()
