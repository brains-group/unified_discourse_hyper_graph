#!/usr/bin/env python3
"""Collect every scores_*.json written by common/score.py into one table.

    python evaluation_scripts/common/collect_results.py            # markdown table + results_table.csv

Where each system's files live (all written by its run.sh):
  hipporag2/outputs/<ds>/      memgraphrag/outputs/<ds>/      logicrag/result/      hypergraphrag/outputs/<ds>/
    scores_<system>_<ds>.json         reported numbers: shared answer step (common/answer.py)
    scores_native_<system>_<ds>.json  the system's own answer step, for reference (EM/F1 only)
    answers_<system>_<ds>.json        per question: answer, raw reader output, the context docs it read

Each scores file = {"dataset", "system", "num_questions", "metrics_percent": {EM, F1, Contain-Acc, LLM-Acc},
"judge_model", "per_example": [...]}.
"""
import argparse
import csv
import json
import re
from pathlib import Path

EVAL_ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("hotpotqa", "musique")
PATTERN = re.compile(r"scores_(?:(native|upstream_reader)_)?(\w+?)_(hotpotqa|musique)\.json$")
QUALITY_PATTERN = re.compile(r"scores_(\w+?)_quality\.json$")  # common/score_quality.py


def subset_table(root: Path, subset_files, metrics, n_boot: int = 10000, seed: int = 0):
    """[unh_v1] Every system's shared-answer-step scores restricted to the same question ids (common/make_subset.py),
    recomputed from each scores file's per_example (no new answers or judge calls). Paired bootstrap 95% intervals
    for UNH minus each baseline, over the same questions."""
    import random
    rows_out = []
    for sub_path in subset_files:
        sub = json.load(open(sub_path))
        ds, ids = sub["dataset"], sub["ids"]
        per_system = {}
        for path in sorted(root.rglob(f"scores_*_{ds}.json")):
            m = PATTERN.search(path.name)
            if "run_logs" in path.parts or not m or m.group(1):  # shared answer step only
                continue
            data = json.load(open(path, encoding="utf-8"))
            by_id = {r["id"]: r for r in data["per_example"]}
            missing = [i for i in ids if i not in by_id]
            if missing:
                print(f"  [skip] {path.relative_to(root)}: {len(missing)} of the {len(ids)} subset questions are missing")
                continue
            if m.group(2) in per_system:
                print(f"  [warn] several scores files for {m.group(2)} {ds}; using {path.relative_to(root)}")
            per_system[m.group(2)] = ([by_id[i] for i in ids], path)
        print(f"\n## {ds}: {len(ids)}-question subset {Path(sub_path).name} "
              f"(strata {sub.get('strata_subset')}); same questions, answer step and judge for every system\n")
        header = ["system"] + metrics + (["UNH − system, F1 [95% CI]", "UNH − system, LLM-Acc [95% CI]"] if "unh" in per_system else [])
        print("| " + " | ".join(header) + " |")
        print("|" + "---|" * len(header))
        rng = random.Random(seed)
        boot_idx = [[rng.randrange(len(ids)) for _ in ids] for _ in range(n_boot)]

        def mean(xs):
            return 100 * sum(xs) / len(xs)

        def paired(a, b, key):
            if any(key not in r for r in a + b):
                return "–"
            d = [x[key] - y[key] for x, y in zip(a, b)]
            boots = sorted(mean([d[i] for i in idx]) for idx in boot_idx)
            return f"{mean(d):+.1f} [{boots[int(0.025 * n_boot)]:+.1f}, {boots[int(0.975 * n_boot)]:+.1f}]"

        order = sorted(per_system, key=lambda k: -mean([r["F1"] for r in per_system[k][0]]))
        for system in order:
            rs, path = per_system[system]
            cells = [system] + [f"{mean([r[k] for r in rs]):.2f}" if all(k in r for r in rs) else "–" for k in metrics]
            if "unh" in per_system:
                u = per_system["unh"][0]
                cells += (["—", "—"] if system == "unh" else [paired(u, rs, "F1"), paired(u, rs, "LLM-Acc")])
            print("| " + " | ".join(cells) + " |")
            rows_out.append({"dataset": ds, "subset": Path(sub_path).name, "system": system, "n": len(rs),
                             **{k: round(mean([r[k] for r in rs]), 2) for k in metrics if all(k in r for r in rs)},
                             "file": str(path.relative_to(root))})
    return rows_out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(EVAL_ROOT), help="directory to search (default: evaluation_scripts/)")
    ap.add_argument("--metrics", nargs="+", default=["EM", "F1", "LLM-Acc"])
    ap.add_argument("--csv", default=None, help="default: <root>/results_table.csv")
    ap.add_argument("--include-native", action="store_true", help="also list each system's own answer step")
    ap.add_argument("--subset", nargs="+", default=None,
                    help="[unh_v1] subset ids file(s) from common/make_subset.py: print every system's scores on exactly "
                         "those questions (recomputed from per_example) with paired bootstrap CIs vs UNH; writes "
                         "<root>/results_table_subset.csv")
    args = ap.parse_args()
    if args.subset:
        root = Path(args.root)
        rows = subset_table(root, args.subset, args.metrics)
        if rows:
            out = root / "results_table_subset.csv"
            with open(out, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["dataset", "subset", "system", "n"] + args.metrics + ["file"], extrasaction="ignore")
                w.writeheader()
                w.writerows(rows)
            print(f"\nrows: {out}")
        return

    root = Path(args.root)
    rows, quality = [], []
    for path in sorted(root.rglob("scores_*.json")):
        if "run_logs" in path.parts:  # aborted runs are archived there
            continue
        q = QUALITY_PATTERN.search(path.name)
        if q:
            data = json.load(open(path, encoding="utf-8"))
            quality.append({"system": q.group(1), "n": data["num_questions"], **data["metrics_percent"],
                            "extractor_used": data["counts"]["extractor_used"], "no_option": data["counts"]["no_option"],
                            "file": str(path.relative_to(root))})
            continue
        m = PATTERN.search(path.name)
        if not m:
            continue
        kind, system, dataset = m.group(1) or "shared", m.group(2), m.group(3)
        data = json.load(open(path, encoding="utf-8"))
        rows.append({"system": system, "answer_step": kind, "dataset": dataset, "n": data["num_questions"],
                     "judge": data.get("judge_model"), **data["metrics_percent"], "file": str(path.relative_to(root))})
    if quality:
        print("QuALITY (dev, pooled; accuracy %, common/score_quality.py)\n")
        print("| system | Accuracy | Acc-HARD | Acc-EASY | n | extractor used | no option | file |")
        print("|---|---|---|---|---|---|---|---|")
        for r in sorted(quality, key=lambda r: r["file"]):
            print(f"| {r['system']} | {r['Accuracy']} | {r['Acc-HARD']} | {r['Acc-EASY']} | {r['n']} | "
                  f"{r['extractor_used']} | {r['no_option']} | {r['file']} |")
        print()
    if not rows:
        if quality:
            return
        raise SystemExit(f"no scores_*.json under {root}")

    out_csv = Path(args.csv) if args.csv else root / "results_table.csv"
    fields = ["system", "answer_step", "dataset", "n", "EM", "F1", "Contain-Acc", "LLM-Acc", "judge", "file"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    # markdown: one row per (system, answer step), columns = dataset x metric
    keys = sorted({(r["system"], r["answer_step"]) for r in rows if args.include_native or r["answer_step"] == "shared"})
    by = {(r["system"], r["answer_step"], r["dataset"]): r for r in rows}
    header = ["system"] + (["answer step"] if args.include_native else []) + \
             [f"{ds} {m}" for ds in DATASETS for m in args.metrics] + ["n (hotpotqa / musique)"]
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for system, kind in keys:
        cells = [system] + ([kind] if args.include_native else [])
        for ds in DATASETS:
            r = by.get((system, kind, ds))
            cells += [f"{r[m]:.2f}" if r and m in r else "–" for m in args.metrics]
        cells.append(" / ".join(str(by[(system, kind, ds)]["n"]) if (system, kind, ds) in by else "–" for ds in DATASETS))
        print("| " + " | ".join(cells) + " |")
    judges = sorted({r["judge"] for r in rows if r["judge"]})
    print(f"\nLLM judge: {', '.join(judges) or 'none'}   |   all rows: {out_csv}")


if __name__ == "__main__":
    main()
