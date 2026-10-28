#!/usr/bin/env python3
"""A fixed question subset of evaluation_data/<ds>/<ds>.json (the HippoRAG 2 release every system was run on).

    python evaluation_scripts/common/make_subset.py --dataset hotpotqa --n 250 --seed 0
    -> evaluation_data/hotpotqa/subsets/hotpotqa_250_seed0.json

Stratified: each stratum gets a share proportional to its size in the full 1,000 (largest-remainder rounding),
and within a stratum the questions are a seeded random sample. Strata:
  hotpotqa  `type` (bridge / comparison)
  musique   hop type = the id prefix (2hop, 3hop1, 3hop2, 4hop1, 4hop2, 4hop3)

The file lists the ids in dataset order; answer.py / score.py / collect_results.py take it with --subset.
Re-running with the same arguments writes the same file (it refuses to overwrite a different one unless --force).

Also writes <name>_corpus.json: the passages that belong to the subset's questions (HotpotQA: each question's
10 context paragraphs, gold + distractors; MuSiQue: its ~20 paragraphs), in the order of <ds>_corpus.json and
in the same {title, text} format. The full corpus is exactly the union of all 1,000 questions' passages, so this
is the subset's share of it. NOTE: the six baselines indexed the full corpus; a system indexed on the subset
corpus searches fewer cross-question distractors (see evaluation_scripts/unh/NOTES.md).
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

DATA_ROOT = Path(__file__).resolve().parents[2] / "evaluation_data"


def qid(sample: dict) -> str:
    return sample.get("id", sample.get("_id"))


def stratum(dataset: str, sample: dict) -> str:
    return sample["type"] if dataset == "hotpotqa" else qid(sample).split("__")[0]


def allocate(counts: dict, n: int) -> dict:
    total = sum(counts.values())
    quota = {k: n * v / total for k, v in counts.items()}
    alloc = {k: int(q) for k, q in quota.items()}
    for k in sorted(quota, key=lambda k: (-(quota[k] - alloc[k]), k))[: n - sum(alloc.values())]:
        alloc[k] += 1
    return alloc


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=["hotpotqa", "musique"])
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exclude", default=None, help="another subset file whose ids must not be used (e.g. a dev set)")
    ap.add_argument("--name", default=None, help="default: <ds>_<n>_seed<seed>")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    samples = json.load(open(DATA_ROOT / args.dataset / f"{args.dataset}.json", encoding="utf-8"))
    excluded = set(json.load(open(args.exclude))["ids"]) if args.exclude else set()
    pool = [s for s in samples if qid(s) not in excluded]
    by_stratum = {}
    for s in pool:
        by_stratum.setdefault(stratum(args.dataset, s), []).append(qid(s))
    full_counts = Counter(stratum(args.dataset, s) for s in samples)
    alloc = allocate(dict(full_counts), args.n)

    rng = random.Random(args.seed)
    chosen = set()
    for k in sorted(alloc):
        if alloc[k] > len(by_stratum.get(k, [])):
            raise SystemExit(f"stratum {k}: need {alloc[k]}, only {len(by_stratum.get(k, []))} available")
        chosen.update(rng.sample(sorted(by_stratum[k]), alloc[k]))
    ids = [qid(s) for s in samples if qid(s) in chosen]  # dataset order

    name = args.name or f"{args.dataset}_{args.n}_seed{args.seed}"
    out = DATA_ROOT / args.dataset / "subsets" / f"{name}.json"
    payload = {
        "dataset": args.dataset, "source": f"evaluation_data/{args.dataset}/{args.dataset}.json (HippoRAG 2 release)",
        "n": len(ids), "seed": args.seed, "stratified_by": "type" if args.dataset == "hotpotqa" else "hop type (id prefix)",
        "strata_full": dict(sorted(full_counts.items())), "strata_subset": dict(sorted(alloc.items())),
        "excluded_from": args.exclude, "ids": ids,
    }
    if out.exists() and not args.force and json.load(open(out))["ids"] != ids:
        raise SystemExit(f"{out} exists with different ids; pass --force to replace it")

    # the subset's own passages, mapped onto the shared corpus by exact (title, text)
    corpus = json.load(open(DATA_ROOT / args.dataset / f"{args.dataset}_corpus.json", encoding="utf-8"))
    index = {(d["title"], d["text"]): i for i, d in enumerate(corpus)}
    keep_idx, unmatched = set(), 0
    for s in samples:
        if qid(s) not in chosen:
            continue
        paras = ([(t, "".join(sents)) for t, sents in s["context"]] if args.dataset == "hotpotqa"
                 else [(p["title"], p["paragraph_text"]) for p in s["paragraphs"]])
        for p in paras:
            if p in index:
                keep_idx.add(index[p])
            else:
                unmatched += 1
    if unmatched:
        raise SystemExit(f"{unmatched} subset passages are not in {args.dataset}_corpus.json")
    sub_corpus = [corpus[i] for i in sorted(keep_idx)]
    corpus_out = out.parent / f"{name}_corpus.json"
    payload["corpus_file"] = corpus_out.name
    payload["corpus_passages"] = len(sub_corpus)
    payload["full_corpus_passages"] = len(corpus)

    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(payload, open(out, "w"), indent=1)
    json.dump(sub_corpus, open(corpus_out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in payload.items() if k != "ids"}, indent=1))
    print(f"wrote {out} and {corpus_out}")


if __name__ == "__main__":
    main()
