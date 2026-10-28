#!/usr/bin/env python3
"""Insurance-contracts result tables (CONTRACTS_EVAL_PLAN.md) from contracts_results/<set>/<system>/summary.json.

    python evaluation_scripts/common/collect_contracts.py               # short and long (and smoke, if present)
    python evaluation_scripts/common/collect_contracts.py --sets smoke

Per set, written to contracts_results/<set>/tables.md and printed:
  QA            rows = systems; CAT1..CAT5 (pooled over every question of the set), All, All excl. CAT5,
                Macro (mean of the per-contract scores), mean context tokens. Scores: mean judge score x 100
                (1.0 / 0.5 / 0.0 per question).
  Fact recall   rows = systems; Pooled, Macro, one column per contract, mean context tokens.
  Per contract  the QA table for each contract.
Also contracts_results/<set>/qa.csv and facts.csv.
"""
import argparse
import csv
import json
from pathlib import Path

RESULTS_ROOT = Path(__file__).resolve().parents[1] / "contracts_results"
CATS = [("CAT1", "CAT1_ATOMIC_LOOKUP"), ("CAT2", "CAT2_RELATIONAL_LOOKUP"), ("CAT3", "CAT3_CONDITIONAL_REASONING"),
        ("CAT4", "CAT4_MULTIHOP_REASONING"), ("CAT5", "CAT5_UNANSWERABLE")]
ORDER = ["hipporag2", "logicrag", "memgraphrag", "hypergraphrag", "ms_graphrag", "hygrag", "unh", "unh_hypergraph", "unh_discourse"]
NOTES = {
    "short": "Each short contract is 2-3 chunks of 1,200 tokens, so HippoRAG 2's and LogicRAG's top-5 is the whole contract.",
    "long": "2025_pdp_contract is ~5 chunks of 1,200 tokens, so HippoRAG 2's and LogicRAG's top-5 is about the whole contract.",
}
NOTES["long_qa"] = NOTES["long"]


def fmt(x):
    return "-" if x is None else (f"{x:.1f}" if isinstance(x, float) else str(x))


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(fmt(v) for v in r) + " |" for r in rows]
    return "\n".join(out)


def collect(set_name: str):
    root = RESULTS_ROOT / set_name
    summaries = {}
    for p in sorted(root.glob("*/summary.json")):
        s = json.load(open(p))
        summaries[s["system"]] = s
    if not summaries:
        return None
    systems = [s for s in ORDER if s in summaries] + sorted(set(summaries) - set(ORDER))
    has_facts = any(s["facts"]["pooled"]["n"] for s in summaries.values())  # QA-only sets (long_qa) have none
    contracts = json.load(open(Path(__file__).resolve().parents[2] / "evaluation_data" / "contracts" / set_name / "contracts.json"))["contracts"]

    qa_header = ["System"] + [c for c, _ in CATS] + ["All", "All excl. CAT5", "Macro", "Ctx tokens", "n"]
    qa_rows = []
    for sysname in systems:
        s = summaries[sysname]
        p, m = s["qa"]["pooled"], s["qa"]["macro"]
        name = sysname + ("" if s.get("complete_set", True) else " (partial)")
        qa_rows.append([name] + [p["by_category"][k]["score"] for _, k in CATS]
                       + [p["score_all"], p["score_excl_cat5"], m["score_all"], p["mean_context_tokens"], p["n"]])

    fact_header = ["System", "Pooled", "Macro"] + contracts + ["Ctx tokens", "n"]
    fact_rows = []
    for sysname in systems:
        s = summaries[sysname]
        per = s["per_contract"]
        fact_rows.append([sysname, s["facts"]["pooled"]["recall"], s["facts"]["macro"]["recall"]]
                         + [per[c]["facts"]["recall"] if c in per else None for c in contracts]
                         + [s["facts"]["pooled"]["mean_context_tokens"], s["facts"]["pooled"]["n"]])

    parts = [f"# Insurance contracts: {set_name} set\n",
             "Scores in percent. QA: mean judge score (1.0 / 0.5 / 0.0) per question; Pooled = over every question of the "
             "set, Macro = mean of the per-contract scores. Fact recall: share of facts the judge finds supported by the "
             "retrieved context. Ctx tokens: mean retrieved-context length (cl100k).",
             f"\n_{NOTES[set_name]}_\n" if set_name in NOTES else "",
             "## QA\n", md_table(qa_header, qa_rows),
             *(["\n## Fact recall\n", md_table(fact_header, fact_rows)] if has_facts else []),
             "\n## QA per contract\n"]
    for c in contracts:
        rows = []
        for sysname in systems:
            per = summaries[sysname]["per_contract"].get(c)
            if not per:
                continue
            q = per["qa"]
            rows.append([sysname] + [q["by_category"][k]["score"] for _, k in CATS]
                        + [q["score_all"], q["score_excl_cat5"], per["facts"]["recall"], q["mean_context_tokens"], q["n"]])
        if rows:
            parts += [f"### {c}\n", md_table(["System"] + [c_ for c_, _ in CATS] + ["All", "All excl. CAT5", "Fact recall",
                                                                                   "Ctx tokens", "n"], rows), ""]
    diag = []
    for sysname in systems:
        s = summaries[sysname]
        p, f = s["qa"]["pooled"], s["facts"]["pooled"]
        diag.append([sysname, p["empty_contexts"] + f["empty_contexts"], p["abstained_on_answerable"],
                     p["judge_unparsed"] + f["judge_unparsed"], p["reader_hit_max_tokens"], p["reader_no_answer_tag"]])
    parts += ["## Diagnostics\n", md_table(["System", "Empty contexts", "Abstained on CAT1-4", "Judge unparsed",
                                           "Reader hit max_tokens", "Reader without 'Answer:'"], diag)]
    text = "\n".join(parts) + "\n"
    (root / "tables.md").write_text(text)
    for name, header, rows in (("qa.csv", qa_header, qa_rows),) + ((("facts.csv", fact_header, fact_rows),) if has_facts else ()):
        with open(root / name, "w", newline="") as fh:
            csv.writer(fh).writerows([header] + rows)
    return text


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", nargs="*", default=["short", "long", "long_qa", "smoke", "smoke_qa"])
    args = ap.parse_args()
    for s in args.sets:
        text = collect(s)
        if text:
            print(text)
            print(f"(written to {RESULTS_ROOT / s / 'tables.md'})\n")


if __name__ == "__main__":
    main()
