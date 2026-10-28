#!/usr/bin/env python3
"""PPR parameter sweep on a QuALITY UNH graph: paths only (ppr_passages 0, no chunk text), retrieval query = question +
options (as in run_quality.sh), the shared letter-only answer step and score_quality.py per variant.

    python evaluation_scripts/unh/sweep_ppr_quality.py                       # every variant, full view
    python evaluation_scripts/unh/sweep_ppr_quality.py --variants base,f60,f80 --views full,discourse
    python evaluation_scripts/unh/sweep_ppr_quality.py --list

Each variant = BASE (the HotpotQA-selected t2000_f40_dmp06) + its overrides. Files go next to the graph's official ones,
tagged pq_<variant>: contexts[_<view>]_pq_<variant>.json, answers_<sys>_pq_<variant>_quality.json,
scores_<sys>_pq_<variant>_quality.json; the summary is <out>/ppr_quality_sweep.{json,md}. Finished retrievals / reader
calls are reused (main.py resumes by id; answer.py caches), so re-running only fills in what is missing.
Needs DEEPSEEK_API_KEY exported; the GPU comes from CUDA_VISIBLE_DEVICES (default here: 1).
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVAL = HERE.parent
DEFAULT_OUT = HERE / "outputs" / "quality_smoke10_detailed_event_quality"

# t2000_f40_dmp06: the setting used for HotpotQA / MuSiQue / QuALITY so far
BASE = {"top_k_seeds": 30, "max_depth": 4, "beam_width": 6, "final_top_k": 80, "context_tokens": 2000,
        "force_top_facts": 40, "damping": 0.6, "ppr_passages": 0}

VARIANTS = {  # name -> overrides of BASE (paths only throughout: ppr_passages stays 0)
    "base": {},
    # more of the top relevant facts forced into the context
    "f60": {"force_top_facts": 60},
    "f80": {"force_top_facts": 80, "final_top_k": 120},
    # more path start facts
    "s50": {"top_k_seeds": 50},
    "s50_f60": {"top_k_seeds": 50, "force_top_facts": 60},
    # more PPR seed facts (cross-encoder-picked restart facts; default 5)
    "seed10": {"seed_facts": 10},
    "seed20": {"seed_facts": 20, "fact_candidates": 100},
    # wider candidate pool for paths and for the cross-encoder
    "rel300_f60": {"relevant_facts": 300, "ce_fact_pool": 200, "force_top_facts": 60},
    # damping
    "dmp05": {"damping": 0.5},
    "dmp07": {"damping": 0.7},
    # second PPR round seeded from the round-1 top facts
    "r2": {"ppr_rounds": 2},
    # shallower / deeper paths
    "d3": {"max_depth": 3},
    "d5_b8": {"max_depth": 5, "beam_width": 8},
    # combined, with a bigger budget so the extra facts fit
    "wide_t3000": {"top_k_seeds": 50, "seed_facts": 10, "relevant_facts": 300, "ce_fact_pool": 200,
                   "force_top_facts": 80, "final_top_k": 150, "context_tokens": 3000},
}
VIEWS = {"full": ("contexts", "unh"), "hypergraph": ("contexts_hypergraph", "unh_hypergraph"),
         "discourse": ("contexts_discourse", "unh_discourse")}


def run(cmd, log):
    with open(log, "a") as f:
        f.write("\n$ " + " ".join(cmd) + "\n")
        f.flush()
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode:
        sys.exit(f"[sweep] failed ({r.returncode}): {' '.join(cmd[:4])} ... see {log}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT), help="UNH QuALITY output dir holding kg.graphml and inputs/")
    ap.add_argument("--variants", default="all", help="comma list of variant names, or all")
    ap.add_argument("--views", default="full", help="comma list of full,hypergraph,discourse")
    ap.add_argument("--workers", type=int, default=50, help="questions retrieved in parallel")
    ap.add_argument("--reader-workers", type=int, default=16)
    ap.add_argument("--limit-q", type=int, default=0, help="smoke: first N questions only")
    ap.add_argument("--list", action="store_true", help="print the variants and exit")
    args = ap.parse_args()

    if args.list:
        for k, v in VARIANTS.items():
            print(f"{k:12} {json.dumps({**BASE, **v})}")
        return
    names = list(VARIANTS) if args.variants == "all" else args.variants.split(",")
    unknown = [n for n in names if n not in VARIANTS]
    if unknown:
        sys.exit(f"unknown variants {unknown}; see --list")
    views = args.views.split(",")
    out = Path(args.out_dir)
    inputs = out / "inputs"
    questions = inputs / "questions.json"
    if not (out / "kg.graphml").exists() or not questions.exists():
        sys.exit(f"{out} needs kg.graphml and inputs/questions.json (a run_quality.sh output dir)")
    if not os.environ.get("DEEPSEEK_API_KEY"):
        sys.exit("export DEEPSEEK_API_KEY first")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
    py = sys.executable
    log = out / "ppr_quality_sweep.log"

    rows = []
    for name in names:
        ppr = {**BASE, **VARIANTS[name]}
        tag = f"pq_{name}"
        cmd = [py, str(HERE / "main.py"), "--dataset", "quality", "--inputs", str(inputs), "--out-dir", str(out),
               "--step", "retrieve", "--view", ",".join(views), "--retriever", "ppr", "--ppr-args", json.dumps(ppr),
               "--workers", str(args.workers), "--tag", tag]
        if args.limit_q:
            cmd += ["--limit-q", str(args.limit_q)]
        print(f"[sweep] {name}: retrieve", flush=True)
        run(cmd, log)
        for view in views:
            ctx_stem, sys_name = VIEWS[view]
            ctx = out / f"{ctx_stem}_{tag}.json"
            ans = out / f"answers_{sys_name}_{tag}_quality.json"
            sc = out / f"scores_{sys_name}_{tag}_quality.json"
            run([py, str(EVAL / "common" / "answer.py"), "--dataset", "quality", "--system", "unh", "--questions",
                 str(questions), "--retrieval", str(ctx), "--out", str(ans), "--workers", str(args.reader_workers)], log)
            run([py, str(EVAL / "common" / "score_quality.py"), "--system", sys_name, "--questions", str(questions),
                 "--predictions", str(ans), "--out", str(sc), "--workers", str(args.reader_workers)], log)
            m = json.load(open(sc))["metrics_percent"]
            c = json.load(open(ctx))
            tok = [x["context_tokens"] for x in c if x.get("status") == "ok"]
            row = {"variant": name, "view": view, "n": len(c), "Accuracy": m["Accuracy"], "Acc-HARD": m["Acc-HARD"],
                   "Acc-EASY": m["Acc-EASY"], "ctx_tokens_mean": round(sum(tok) / max(len(tok), 1)),
                   "ctx_tokens_max": max(tok, default=0), "ppr_args": ppr}
            rows.append(row)
            print(f"[sweep] {name:12} {view:10} acc {row['Accuracy']:5.1f}  HARD {row['Acc-HARD']:5.1f}  "
                  f"EASY {row['Acc-EASY']:5.1f}  tokens {row['ctx_tokens_mean']}", flush=True)

    json.dump(rows, open(out / "ppr_quality_sweep.json", "w"), indent=2)
    md = ["| variant | view | n | Accuracy | Acc-HARD | Acc-EASY | context tokens mean / max | overrides |",
          "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['variant']} | {r['view']} | {r['n']} | {r['Accuracy']:.1f} | {r['Acc-HARD']:.1f} | "
                  f"{r['Acc-EASY']:.1f} | {r['ctx_tokens_mean']} / {r['ctx_tokens_max']} | "
                  f"{json.dumps(VARIANTS[r['variant']])} |")
    (out / "ppr_quality_sweep.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
