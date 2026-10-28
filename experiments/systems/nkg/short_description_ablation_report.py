#!/usr/bin/env python3
"""Compare the long vs short edge-description conditions of short_description_ablation.py.

    python -m experiments.systems.nkg.short_description_ablation_report

For each dataset, reads the newest evaluation_results*.json in sample_exp/nkg/<ds>/short_desc_ablation/{long,short,short_strict}/
and prints EM / F1 overall and by question type, context tokens per question (after the 2,000-token cap),
raw context length before the cap, and answer-time LLM usage. Also writes short_desc_ablation/comparison.json.
"""
import glob
import json
import os
import statistics as st

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "sample_exp", "nkg")


def newest(path_glob):
    files = glob.glob(path_glob)
    return max(files, key=os.path.getmtime) if files else None


def summarize(path):
    d = json.load(open(path))
    ev = d["evaluation"]
    ok = [r for r in ev["results"] if r["status"] == "SUCCESS"]
    return {
        "file": os.path.relpath(path, ROOT),
        "model": d["metadata"]["model"], "engine": d["metadata"]["engine"],
        "neutral_layout": d["metadata"]["retrieval_args"].get("neutral_layout"),
        "n": ev["successful"], "errored": ev["errored"],
        "EM": round(100 * ev["overall"]["em"], 2), "F1": round(100 * ev["overall"]["f1"], 2),
        "by_type": {k: {"n": v["count"], "EM": round(100 * v["em"], 2), "F1": round(100 * v["f1"], 2)}
                    for k, v in sorted(ev["by_type"].items())},
        "context_tokens_mean": round(st.mean(r["context_tokens"] for r in ok)) if ok else None,
        "context_hit_cap": sum(r["context_tokens"] >= r["max_context_tokens"] for r in ok),
        "raw_context_chars_mean": round(st.mean(r["raw_context_length_chars"] for r in ok)) if ok else None,
        "eval_llm_calls": ev["evaluation_tokens"]["api_calls"],
        "eval_prompt_tokens": ev["evaluation_tokens"]["prompt_tokens"],
    }


def main():
    out = {}
    for ds in ("hotpotqa", "musique"):
        base = os.path.join(ROOT, ds, "short_desc_ablation")
        rows = {c: newest(os.path.join(base, c, "evaluation_results*.json")) for c in ("long", "short", "short_strict")}
        if not any(rows.values()):
            continue
        out[ds] = {c: summarize(p) for c, p in rows.items() if p}
        print(f"\n## {ds}")
        print("| condition | model | engine | n (err) | EM | F1 | context tok/q | hit 2k cap | raw context chars/q | eval calls |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        for c, s in out[ds].items():
            print(f"| {c} | {s['model'].split('/')[-1]} | {s['engine']} | {s['n']} ({s['errored']}) | {s['EM']} | {s['F1']} | "
                  f"{s['context_tokens_mean']} | {s['context_hit_cap']} | {s['raw_context_chars_mean']:,} | {s['eval_llm_calls']} |")
        types = sorted({t for s in out[ds].values() for t in s["by_type"]})
        print("\nF1 by type: " + "; ".join(
            f"{t}: " + " vs ".join(f"{c} {s['by_type'].get(t, {}).get('F1', '–')}" for c, s in out[ds].items()) for t in types))
        for c, s in out[ds].items():
            print(f"  {c}: {s['file']}")
        json.dump(out[ds], open(os.path.join(base, "comparison.json"), "w"), indent=2)
    if not out:
        print("no ablation results yet")


if __name__ == "__main__":
    main()
