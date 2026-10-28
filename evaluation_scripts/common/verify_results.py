#!/usr/bin/env python3
"""Re-derive every reported baseline number from the per-question records, for checking the tables by hand.

    ~/miniconda3/envs/unh_eval/bin/python evaluation_scripts/common/verify_results.py [--skip-tokens]

For each (dataset, system) it opens the scores file, recomputes the metrics as plain means over `per_example`
(or, for contracts, over every contract's scores_qa.json), prints them next to the stored summary value, and
flags any mismatch. Context size = mean cl100k tokens of the context the shared reader saw (answers_*.json
`retrieved_docs`, joined with blank lines; contracts: the stored `context`). Read-only: writes nothing.
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SYSTEMS = ["hipporag2", "logicrag", "memgraphrag", "hypergraphrag", "ms_graphrag", "hygrag"]


def out_dir(system, ds):
    if system == "logicrag":
        return ROOT / "logicrag/result" / ("" if ds in ("hotpotqa", "musique") else ds)
    if system == "hipporag2" and ds in ("quality", "gmedical"):
        return ROOT / "hipporag2/outputs" / ("quality" if ds == "quality" else "gmedical_quality")
    return ROOT / system / "outputs" / ds


_enc = None


def ctx_tokens(answers_path):
    global _enc
    if _enc is None:
        import tiktoken
        _enc = tiktoken.get_encoding("cl100k_base")
    rows = json.load(open(answers_path))
    docs = [r.get("retrieved_docs") or [] for r in rows]
    return sum(len(_enc.encode("\n\n".join(map(str, d)))) for d in docs) / max(len(docs), 1)


def mean(xs):
    xs = [float(x) for x in xs if x is not None]
    return 100 * sum(xs) / len(xs) if xs else None


def check(label, stored, recomputed):
    ok = stored is None or recomputed is None or abs(stored - recomputed) < 0.06
    return f"{label} {recomputed:.2f}" + ("" if ok else f" (!! stored {stored})")


def multihop(ds, tokens):
    print(f"\n## {ds}  (scores_<sys>_{ds}.json; metrics = mean of per_example)")
    for s in SYSTEMS:
        d = out_dir(s, ds)
        f = d / f"scores_{s}_{ds}.json"
        if not f.exists():
            print(f"  {s:14s} MISSING {f.relative_to(ROOT)}")
            continue
        data = json.load(open(f))
        pe, stored = data["per_example"], data["metrics_percent"]
        parts = [check(m, stored.get(m), mean(r[m] for r in pe)) for m in ("EM", "F1", "Contain-Acc", "LLM-Acc")]
        a = d / f"answers_{s}_{ds}.json"
        if tokens and a.exists():
            parts.append(f"ctx_tok {ctx_tokens(a):.0f}")
        print(f"  {s:14s} n={len(pe):5d}  " + "  ".join(parts) + f"   [{f.relative_to(ROOT)}]")


def quality(tokens):
    print("\n## quality  (scores_<sys>_quality.json; Accuracy = mean(correct), HARD = difficult==1)")
    for s in SYSTEMS:
        d = out_dir(s, "quality")
        f = d / f"scores_{s}_quality.json"
        if not f.exists():
            print(f"  {s:14s} MISSING {f.relative_to(ROOT)}")
            continue
        data = json.load(open(f))
        pe, st = data["per_example"], data["metrics_percent"]
        parts = [check("Acc", st["Accuracy"], mean(r["correct"] for r in pe)),
                 check("HARD", st["Acc-HARD"], mean(r["correct"] for r in pe if r["difficult"])),
                 check("EASY", st["Acc-EASY"], mean(r["correct"] for r in pe if not r["difficult"]))]
        a = d / f"answers_{s}_quality.json"
        if tokens and a.exists():
            parts.append(f"ctx_tok {ctx_tokens(a):.0f}")
        print(f"  {s:14s} n={len(pe):5d}  " + "  ".join(parts) + f"   [{f.relative_to(ROOT)}]")


def gmedical(tokens):
    print("\n## gmedical  (scores_<sys>_gmedical.json; LLM-Acc = mean of per_example, per level; macro = mean of levels)")
    levels = ["Fact Retrieval", "Complex Reasoning", "Contextual Summarize", "Creative Generation"]
    for s in SYSTEMS:
        d = out_dir(s, "gmedical")
        f = d / f"scores_{s}_gmedical.json"
        if not f.exists():
            print(f"  {s:14s} MISSING {f.relative_to(ROOT)}")
            continue
        data = json.load(open(f))
        pe, st = data["per_example"], data["metrics_percent"]
        per_level = {lv: mean(r["LLM-Acc"] for r in pe if r["level"] == lv) for lv in levels}
        parts = [check("All", st["LLM-Acc"], mean(r["LLM-Acc"] for r in pe))]
        parts += [check(lv.split()[0], st.get(f"LLM-Acc {lv}"), per_level[lv]) for lv in levels]
        parts.append(check("Macro", st.get("LLM-Acc macro (mean of levels)"), sum(per_level.values()) / 4))
        a = d / f"answers_{s}_gmedical.json"
        if tokens and a.exists():
            parts.append(f"ctx_tok {ctx_tokens(a):.0f}")
        print(f"  {s:14s} n={len(pe):5d}  " + "  ".join(parts) + f"   [{f.relative_to(ROOT)}]")


def contracts(set_name):
    cats = ["CAT1_ATOMIC_LOOKUP", "CAT2_RELATIONAL_LOOKUP", "CAT3_CONDITIONAL_REASONING", "CAT4_MULTIHOP_REASONING",
            "CAT5_UNANSWERABLE"]
    root = ROOT / "contracts_results" / set_name
    if not root.exists():
        return
    print(f"\n## contracts {set_name}  (contracts_results/{set_name}/<sys>/<contract>/scores_qa.json; "
          "score = mean judge score; Macro = mean of per-contract scores)")
    for s in SYSTEMS:
        sd = root / s
        summ = sd / "summary.json"
        if not summ.exists():
            print(f"  {s:14s} MISSING {summ.relative_to(ROOT)}")
            continue
        st = json.load(open(summ))["qa"]
        rows, per_contract = [], []
        for f in sorted(sd.glob("*/scores_qa.json")):
            data = json.load(open(f))
            r = data["per_question"] if isinstance(data, dict) and "per_question" in data else data
            rows += r
            per_contract.append(mean(x["score"] for x in r))
        parts = [check("All", st["pooled"]["score_all"], mean(x["score"] for x in rows)),
                 check("exCAT5", st["pooled"]["score_excl_cat5"], mean(x["score"] for x in rows if x["category"] != cats[4]))]
        parts += [check(c[:4], st["pooled"]["by_category"][c]["score"], mean(x["score"] for x in rows if x["category"] == c))
                  for c in cats]
        parts.append(check("Macro", st["macro"]["score_all"], sum(per_contract) / len(per_contract)))
        parts.append(f"ctx_tok {sum(x['context_tokens'] for x in rows) / len(rows):.0f}")
        print(f"  {s:14s} n={len(rows):5d}  " + "  ".join(parts) + f"   [{summ.relative_to(ROOT)}]")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-tokens", action="store_true", help="skip the context-token counts (faster)")
    args = ap.parse_args()
    tokens = not args.skip_tokens
    for ds in ("hotpotqa", "musique"):
        multihop(ds, tokens)
    quality(tokens)
    gmedical(tokens)
    contracts("long_qa")


if __name__ == "__main__":
    main()
