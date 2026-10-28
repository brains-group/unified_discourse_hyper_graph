#!/usr/bin/env python3
"""Score any system's HotpotQA / MuSiQue output with the same metrics the papers report.

    python evaluation_scripts/common/score.py --dataset musique --system hipporag2 \
        --predictions evaluation_scripts/hipporag2/outputs/musique/predictions.json \
        [--llm-judge --judge-model deepseek-v4-flash --judge-base-url https://api.deepseek.com]

Metrics (see metrics.py / prompts.py for provenance):
  EM, F1              HippoRAG 2 / MemGraphRAG (gold = answer + answer_aliases, max)
  Contain-Acc         LinearRAG "Str-Acc" in the MemGraphRAG table (gold = primary answer)
  LLM-Acc             LinearRAG judge prompt, only with --llm-judge (gold = primary answer)
  Recall@k            only with --recall-k, only for systems whose retrieved units are corpus passages
                      (hipporag2, logicrag); MemGraphRAG retrieves token chunks and
                      MS GraphRAG / HyperGraphRAG a graph context, so passage recall is not
                      defined for them.

Predictions are aligned to evaluation_data/<dataset>/<dataset>.json by position, and every
question string is checked against the dataset. A prefix of the dataset (e.g. --limit runs) is fine.

LLM-Acc caveat: the papers used gpt-4o-mini as judge. Report which judge model you used.
"""
import argparse
import json
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import metrics as M  # noqa: E402
from prompts import JUDGE_SYSTEM_PROMPT, judge_user_prompt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "evaluation_data"
SYSTEMS = ("hipporag2", "logicrag", "memgraphrag", "ms_graphrag", "hypergraphrag", "hygrag", "unh")
PASSAGE_LEVEL = {"hipporag2", "logicrag"}


def load_predictions(system: str, path: str, corpus: list) -> list:
    """Return [{question, pred_answer, retrieved_docs}] in dataset order."""
    data = json.load(open(path, encoding="utf-8"))
    rows = []
    if isinstance(data, list):
        # [unh_v1] common/answer.py output (any system) and the hipporag2 / ms_graphrag / hypergraphrag files
        for r in data:
            rows.append({"question": r["question"], "pred_answer": r["pred_answer"], "retrieved_docs": r.get("retrieved_docs") or []})
    elif system in ("hipporag2", "ms_graphrag", "hypergraphrag"):
        for r in data:
            rows.append({"question": r["question"], "pred_answer": r["pred_answer"], "retrieved_docs": r.get("retrieved_docs") or []})
    elif system == "memgraphrag":
        for r in data["results"]:
            rows.append({"question": r["question"], "pred_answer": r["predicted_answer"], "retrieved_docs": r.get("retrieved_docs") or []})
    elif system == "logicrag":
        # LogicRAG formats passages as "Title: {title}. Content: {text}" (src/models/base_rag.py); map back to title\ntext
        to_hippo = {f"Title: {d['title']}. Content: {d['text']}": f"{d['title']}\n{d['text']}" for d in corpus}
        for r in data["results"]:
            docs = [to_hippo.get(c, c) for c in r.get("contexts") or []]
            rows.append({"question": r["question"], "pred_answer": r["answer"], "retrieved_docs": docs})
    else:
        raise ValueError(system)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=["hotpotqa", "musique"])
    ap.add_argument("--system", required=True, choices=SYSTEMS)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--out", default=None, help="default: <predictions dir>/scores_<system>_<dataset>.json")
    ap.add_argument("--subset", default=None,
                    help="ids file from common/make_subset.py: score only those questions (predictions must hold "
                         "exactly them, in dataset order). To score an existing full-run file on a subset, use "
                         "collect_results.py --subset instead.")
    ap.add_argument("--recall-k", type=int, nargs="*", default=[],
                    help="passage Recall@k for hipporag2 / logicrag native outputs, e.g. --recall-k 2 5 (off by default)")
    ap.add_argument("--llm-judge", action="store_true")
    ap.add_argument("--judge-model", default=os.getenv("JUDGE_MODEL", "deepseek-v4-flash"))
    ap.add_argument("--judge-base-url", default=os.getenv("JUDGE_BASE_URL", "https://api.deepseek.com"))
    ap.add_argument("--judge-api-key-env", default="DEEPSEEK_API_KEY", help="name of the env var holding the judge key")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    samples = json.load(open(DATA_ROOT / args.dataset / f"{args.dataset}.json", encoding="utf-8"))
    corpus = json.load(open(DATA_ROOT / args.dataset / f"{args.dataset}_corpus.json", encoding="utf-8"))
    if args.subset:
        keep = set(json.load(open(args.subset))["ids"])
        samples = [s for s in samples if s.get("id", s.get("_id")) in keep]
        if len(samples) != len(keep):
            sys.exit(f"{len(keep) - len(samples)} subset ids are not in {args.dataset}.json")
    preds = load_predictions(args.system, args.predictions, corpus)
    if len(preds) > len(samples):
        sys.exit(f"{len(preds)} predictions but only {len(samples)} questions")
    for i, (p, s) in enumerate(zip(preds, samples)):
        if p["question"].strip() != s["question"].strip():
            sys.exit(f"question mismatch at index {i}: {p['question']!r} vs {s['question']!r}")
    samples = samples[: len(preds)]

    judge = None
    judge_usage = {"api_calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "prompt_cache_hit_tokens": 0}
    judge_lock = threading.Lock()
    if args.llm_judge:
        from openai import OpenAI
        client = OpenAI(api_key=os.environ[args.judge_api_key_env], base_url=args.judge_base_url)

        def judge(pred, gold):  # LinearRAG LLM_Model.infer: max_tokens=2000
            resp = client.chat.completions.create(
                model=args.judge_model, max_tokens=2000,
                # [unh_v1] temperature: LLM_TEMPERATURE if set, else not sent (provider default). LinearRAG sends 0.
                **({"temperature": float(os.environ["LLM_TEMPERATURE"])} if os.environ.get("LLM_TEMPERATURE") else {}),
                # [unh_v1] DeepSeek V4/V4.1 think by default; disable unless LLM_DISABLE_THINKING=0
                **({"extra_body": {"thinking": {"type": "disabled"}}} if os.environ.get("LLM_DISABLE_THINKING", "1") != "0" else {}),
                messages=[{"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                          {"role": "user", "content": judge_user_prompt(pred, gold)}])
            u = resp.usage.model_dump() if resp.usage is not None else {}
            with judge_lock:  # [unh_v1] judge token usage, reported in the scores file
                judge_usage["api_calls"] += 1
                judge_usage["prompt_tokens"] += int(u.get("prompt_tokens") or 0)
                judge_usage["completion_tokens"] += int(u.get("completion_tokens") or 0)
                judge_usage["prompt_cache_hit_tokens"] += int(u.get("prompt_cache_hit_tokens")
                                                              or (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
            # [unh_v1] LinearRAG compares .strip().lower() == "correct"; punctuation is also dropped here so
            # "Correct." counts ("incorrect" still never matches)
            return 1.0 if re.sub(r"[^a-z]", "", (resp.choices[0].message.content or "").lower()) == "correct" else 0.0

    per_example = []
    for p, s in zip(preds, samples):
        golds = M.get_gold_answers(s)
        row = {"id": s.get("id", s.get("_id")), "question": s["question"], "gold": s["answer"],
               "gold_answers": golds, "pred_answer": p["pred_answer"],
               "EM": M.exact_match(golds, p["pred_answer"]), "F1": M.f1(golds, p["pred_answer"]),
               "Contain-Acc": float(M.contain(p["pred_answer"], s["answer"]))}
        if args.system in PASSAGE_LEVEL and args.recall_k:
            row.update(M.recall_at_k(M.get_gold_docs(s, args.dataset), p["retrieved_docs"], args.recall_k))
        per_example.append(row)

    if judge is not None:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            scores = list(ex.map(lambda r: judge(r["pred_answer"], r["gold"]), per_example))
        for r, sc in zip(per_example, scores):
            r["LLM-Acc"] = sc

    keys = [k for k in per_example[0] if k in ("EM", "F1", "Contain-Acc", "LLM-Acc") or k.startswith("Recall@")]
    summary = {k: round(100 * sum(r[k] for r in per_example) / len(per_example), 2) for k in keys}
    result = {"dataset": args.dataset, "system": args.system, "predictions": os.path.abspath(args.predictions),
              "num_questions": len(per_example), "subset": os.path.abspath(args.subset) if args.subset else None,
              "metrics_percent": summary,
              "judge_model": args.judge_model if judge else None,
              "judge_usage": judge_usage if judge else None, "per_example": per_example}
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.predictions)), f"scores_{args.system}_{args.dataset}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: result[k] for k in ("dataset", "system", "num_questions", "metrics_percent", "judge_model")}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
