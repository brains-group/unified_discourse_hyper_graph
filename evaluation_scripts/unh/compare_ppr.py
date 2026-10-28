#!/usr/bin/env python3
"""Comparison table: UNH PPR retriever vs the old UNH retriever (planner + beam search, every view) vs every baseline,
on the same questions, all answered by the shared reader (common/answer.py, hipporag2 prompt) and scored by
common/score.py (EM / F1 / Contain-Acc / LLM-Acc).

    python evaluation_scripts/unh/compare_ppr.py --dataset hotpotqa --ppr-tag t2000_f40_dmp06_q1000
    python evaluation_scripts/unh/compare_ppr.py --dataset hotpotqa --ppr-tag t2000_f40_dmp06 --subset 250

Context tokens = what each system hands the reader (answers file `retrieved_docs`, joined), counted with the Qwen3
tokenizer the PPR budget uses. For UNH PPR the discourse / entity shares of the path edges come from the sweep summary.
Writes outputs/<ds>_full/ppr_sweep/comparison_<tag>.{json,md}.
"""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVAL = HERE.parent
DATA = EVAL.parent / "evaluation_data"

BASELINES = {  # name -> score file (shared answer step), relative to evaluation_scripts/
    "HippoRAG 2": "hipporag2/outputs/{ds}/scores_hipporag2_{ds}.json",
    "LogicRAG": "logicrag/result/scores_logicrag_{ds}.json",
    "MemGraphRAG": "memgraphrag/outputs/{ds}/scores_memgraphrag_{ds}.json",
    "HyperGraphRAG": "hypergraphrag/outputs/{ds}/scores_hypergraphrag_{ds}.json",
    "MS GraphRAG": "ms_graphrag/outputs/{ds}/scores_ms_graphrag_{ds}.json",
    "HyGRAG": "hygrag/outputs/{ds}/scores_hygrag_{ds}.json",
}
OLD_UNH = {
    "UNH old retriever (full view)": "unh/outputs/{ds}_full/scores_unh_{ds}.json",
    "UNH old retriever (hypergraph view)": "unh/outputs/{ds}_full/scores_unh_hypergraph_{ds}.json",
    "UNH old retriever (discourse view)": "unh/outputs/{ds}_full/scores_unh_discourse_{ds}.json",
}
METRICS = ("EM", "F1", "Contain-Acc", "LLM-Acc")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=["hotpotqa", "musique"])
    ap.add_argument("--ppr-tag", required=True, help="variant name of the PPR run, e.g. t2000_f40_dmp06_q1000")
    ap.add_argument("--subset", type=int, default=0, help="250: restrict every system to the 250 subset")
    ap.add_argument("--views", default="full", help="PPR views to include, e.g. full,hypergraph,discourse "
                    "(scores_unh_ppr[_<view>]_<tag>_<ds>.json; path stats from summary_<view>.json)")
    args = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-30B-A3B-Instruct-2507")

    ds = args.dataset
    sweep = EVAL / "unh" / "outputs" / f"{ds}_full" / "ppr_sweep"
    keep = None
    if args.subset:
        keep = set(json.load(open(DATA / ds / "subsets" / f"{ds}_{args.subset}_seed0.json"))["ids"])
    vname = {"full": "full view", "hypergraph": "UNH-H, hypergraph view", "discourse": "UNH-D, discourse view"}
    views = args.views.split(",")
    systems = {f"UNH PPR ({vname[v]})": str(sweep / f"scores_unh_ppr{'' if v == 'full' else '_' + v}_{args.ppr_tag}_{ds}.json")
               for v in views}
    systems.update({k: str(EVAL / v.format(ds=ds)) for k, v in OLD_UNH.items()})
    systems.update({k: str(EVAL / v.format(ds=ds)) for k, v in BASELINES.items()})
    questions = {s.get("id", s.get("_id")): s["question"].strip() for s in json.load(open(DATA / ds / f"{ds}.json"))}

    rows = []
    for name, f in systems.items():
        if not Path(f).exists():
            print(f"[compare] missing {f}; skipped")
            continue
        r = json.load(open(f))
        pe = [p for p in r["per_example"] if keep is None or p["id"] in keep]
        row = {"system": name, "n": len(pe)}
        row.update({m: round(100 * sum(p[m] for p in pe) / len(pe), 1) for m in METRICS if m in pe[0]})
        ids = {p["id"] for p in pe}
        qset = {questions[i] for i in ids}
        ans = json.load(open(r["predictions"]))
        ctx = [x for x in ans if (x.get("id") in ids) or (x.get("id") is None and x["question"].strip() in qset)]
        t = sorted(len(tok("\n\n".join(x["retrieved_docs"]))["input_ids"]) for x in ctx)
        row.update({"context_tokens_mean": round(sum(t) / len(t)), "context_tokens_max": t[-1]})
        rows.append(row)

    edge = {}
    for v in views:
        summ = json.load(open(sweep / f"summary_{v}.json")).get(args.ppr_tag, {})
        edge[v] = {k: summ.get(k) for k in ("pct_edges_discourse", "pct_edges_entity", "pct_hops_discourse",
                                             "pct_hops_entity", "mean_edges_per_question", "mean_distinct_facts",
                                             "answer_in_context_pct", "mean_tokens", "mean_seconds")}
    rows.sort(key=lambda x: -x.get("F1", 0))
    out = {"dataset": ds, "subset": args.subset or None, "ppr_tag": args.ppr_tag, "ppr_path_stats": edge, "rows": rows}
    stem = (f"comparison_{args.ppr_tag}" + ("" if views == ["full"] else "_" + "_".join(views))
            + (f"_subset{args.subset}" if args.subset else ""))
    json.dump(out, open(sweep / f"{stem}.json", "w"), indent=2)
    md = [f"| system | n | EM | F1 | Contain-Acc | LLM-Acc | context tokens mean / max |", "|---|---|---|---|---|---|---|"]
    for x in rows:
        md.append(f"| {x['system']} | {x['n']} | {x.get('EM')} | {x.get('F1')} | {x.get('Contain-Acc')} | "
                  f"{x.get('LLM-Acc')} | {x['context_tokens_mean']} / {x['context_tokens_max']} |")
    md.append("")
    for v, e in edge.items():
        md.append(f"UNH PPR {v} path stats: {e}")
    (sweep / f"{stem}.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
