#!/usr/bin/env python3
"""Score any system's G-Medical answers (common/answer.py --dataset gmedical output).

    python evaluation_scripts/common/score_gmedical.py --system hipporag2 \
        --predictions evaluation_scripts/hipporag2/outputs/gmedical_quality/answers_hipporag2_gmedical.json

Metric = MemGraphRAG's G-Bench metric: LLM-Acc only ("since golden answers consist of lengthy descriptive statements,
we only evaluate using LLM-Acc", MemGraphRAG §5). The judge prompt is LinearRAG's (prompts.judge_user_prompt, the
same as common/score.py --llm-judge): generated answer vs gold answer, reply 'correct' or 'incorrect'. The generated
answer is answer.py's pred_answer (text after "Answer:", as MemGraphRAG / HippoRAG 2 parse it).

Reported overall and per GraphRAG-Bench level: type1 Fact Retrieval, type2 Complex Reasoning, type3 Contextual
Summarize, type4 Creative Generation (the overall number is dominated by type1, 53% of the questions).
Judge calls are cached in judge_cache.sqlite next to the predictions; the judge model is recorded in the output
(MemGraphRAG used gpt-4o-mini; ours defaults to the DeepSeek model every other step uses).
"""
import argparse
import hashlib
import json
import os
import re
import sqlite3
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompts import JUDGE_SYSTEM_PROMPT, judge_user_prompt  # noqa: E402

DATA_ROOT = Path(__file__).resolve().parents[2] / "evaluation_data"
SYSTEMS = ("hipporag2", "logicrag", "memgraphrag", "ms_graphrag", "hypergraphrag", "hygrag", "unh", "unh_hypergraph",
           "unh_discourse")
LEVELS = {"type1": "Fact Retrieval", "type2": "Complex Reasoning", "type3": "Contextual Summarize",
          "type4": "Creative Generation"}


class Judge:
    """LinearRAG's LLM-Acc judge, cached. max_tokens 4096 (LinearRAG LLM_Model.infer sends 2000; raised with the
    G-Medical reader budget, user decision 2026-10-11; the reply is one word, so it only removes a cap)."""

    def __init__(self, model, base_url, api_key, cache_path: Path):
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=300, max_retries=5)
        self.params = {"model": model, "max_tokens": 4096}
        if os.environ.get("LLM_TEMPERATURE"):  # else not sent (provider default); LinearRAG sends 0
            self.params["temperature"] = float(os.environ["LLM_TEMPERATURE"])
        if os.environ.get("LLM_DISABLE_THINKING", "1") != "0":
            self.params["extra_body"] = {"thinking": {"type": "disabled"}}
        self.conn = sqlite3.connect(str(cache_path), check_same_thread=False, timeout=120)
        self.conn.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, response TEXT)")
        self.conn.commit()
        self.lock = threading.Lock()
        self.api_calls = 0

    def __call__(self, pred: str, gold: str):
        request = {**self.params, "messages": [{"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                                               {"role": "user", "content": judge_user_prompt(pred, gold)}]}
        key = hashlib.sha256(json.dumps(request, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
        with self.lock:
            row = self.conn.execute("SELECT response FROM cache WHERE key = ?", (key,)).fetchone()
        if row is not None:
            raw = row[0]
        else:
            raw = self.client.chat.completions.create(**request).choices[0].message.content or ""
            with self.lock:
                self.api_calls += 1
                if raw:
                    self.conn.execute("INSERT OR REPLACE INTO cache (key, response) VALUES (?, ?)", (key, raw))
                    self.conn.commit()
        # LinearRAG: .strip().lower() == "correct"; punctuation also dropped, as common/score.py
        return raw, 1.0 if re.sub(r"[^a-z]", "", raw.lower()) == "correct" else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--system", required=True, choices=SYSTEMS)
    ap.add_argument("--predictions", required=True, help="answer.py --dataset gmedical output")
    ap.add_argument("--questions", default=None, help="default: $GMEDICAL_INPUTS/questions.json or "
                    "evaluation_data/gmedical/inputs/questions.json")
    ap.add_argument("--out", default=None, help="default: <predictions dir>/scores_<system>_gmedical.json")
    ap.add_argument("--judge-model", default=os.getenv("JUDGE_MODEL", os.getenv("LLM_MODEL", "deepseek-v4-flash")))
    ap.add_argument("--judge-base-url", default=os.getenv("JUDGE_BASE_URL", os.getenv("LLM_BASE_URL", "https://api.deepseek.com")))
    ap.add_argument("--judge-api-key-env", default="DEEPSEEK_API_KEY")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    qpath = args.questions or Path(os.environ.get("GMEDICAL_INPUTS", DATA_ROOT / "gmedical" / "inputs")) / "questions.json"
    questions = json.load(open(qpath, encoding="utf-8"))
    preds = json.load(open(args.predictions, encoding="utf-8"))
    if len(preds) != len(questions):
        sys.exit(f"{len(preds)} predictions but {len(questions)} questions in {qpath}")
    for i, (p, q) in enumerate(zip(preds, questions)):
        if p["question"].strip() != q["question"].strip():
            sys.exit(f"question mismatch at index {i}: {p['question']!r} vs {q['question']!r}")

    judge = Judge(args.judge_model, args.judge_base_url, os.environ[args.judge_api_key_env],
                  Path(args.predictions).resolve().parent / "judge_cache.sqlite")
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        verdicts = list(tqdm(ex.map(lambda pq: judge(pq[0]["pred_answer"], pq[1]["answer"]), zip(preds, questions)),
                             total=len(preds), desc=f"judge {args.system}"))

    per_example = [{"id": q["id"], "question_type": q["question_type"], "level": q["level"], "question": q["question"],
                    "gold": q["answer"], "pred_answer": p["pred_answer"], "judge_reply": raw, "LLM-Acc": sc,
                    "context_chars": sum(len(d) for d in p.get("retrieved_docs") or [])}
                   for p, q, (raw, sc) in zip(preds, questions, verdicts)]
    acc = lambda rows: round(100 * sum(r["LLM-Acc"] for r in rows) / len(rows), 2) if rows else None  # noqa: E731
    by_type = {t: [r for r in per_example if r["question_type"] == t] for t in LEVELS}
    metrics = {"LLM-Acc": acc(per_example),
               **{f"LLM-Acc {LEVELS[t]}": acc(rows) for t, rows in by_type.items()},
               "LLM-Acc macro (mean of levels)": round(sum(acc(r) for r in by_type.values() if r)
                                                       / max(1, sum(bool(r) for r in by_type.values())), 2)}
    counts = {"questions": len(per_example), **{f"n {LEVELS[t]}": len(r) for t, r in by_type.items()},
              "empty_pred": sum(not r["pred_answer"].strip() for r in per_example),
              "judge_reply_not_correct_or_incorrect": sum(re.sub(r"[^a-z]", "", r["judge_reply"].lower())
                                                          not in ("correct", "incorrect") for r in per_example),
              "mean_context_chars": round(sum(r["context_chars"] for r in per_example) / len(per_example)),
              "judge_api_calls_this_run": judge.api_calls}
    result = {"dataset": "gmedical", "system": args.system, "predictions": os.path.abspath(args.predictions),
              "questions": os.path.abspath(qpath), "num_questions": len(per_example), "metrics_percent": metrics,
              "counts": counts, "judge_model": args.judge_model, "per_example": per_example}
    out = args.out or Path(args.predictions).resolve().parent / f"scores_{args.system}_gmedical.json"
    json.dump(result, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(json.dumps({k: result[k] for k in ("system", "num_questions", "metrics_percent", "counts", "judge_model")}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
