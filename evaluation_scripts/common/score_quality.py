#!/usr/bin/env python3
"""Score any system's QuALITY answers (common/answer.py --dataset quality output).

    python evaluation_scripts/common/score_quality.py --system hygrag \
        --predictions evaluation_scripts/hygrag/outputs/quality/answers_hygrag_quality.json

Metric = HyGRAG/DIGIMON's QuALITY metric (Core/Utils/Evaluation.py close_eval): accuracy of the chosen option
letter against the gold letter. Reported as a percentage, overall and on QuALITY's official HARD / EASY
questions (`difficult` flag: most speed validators answered wrongly).

How the chosen letter is found:
  1. The shared answer prompt asks for the letter only, and answer.py already parsed bare-letter replies
     (pred_answer).
  2. Any other reply goes to HyGRAG's own option extractor: CLOSE_EXTRACT_OPTION_PORMPT verbatim
     (hygrag_close_extract_prompt.py), sent to the same LLM, JSON reply, "predict" = the letter or "-1".
     Calls are cached in extract_cache.sqlite next to the predictions.

Two upstream scoring bugs are not copied (see QUALITY_EVAL_PLAN.md §1):
  - HyGRAG compares letters after SQuAD normalisation, which deletes the article "a", so gold "A" also matches an
    empty extraction. Here the comparison is strict: one of A-D, equal to the gold letter.
  - The extractor prompt's FORMAT section names the key "model_output" while HyGRAG reads "predict"; a reply
    that follows the FORMAT section crashed into "-1" upstream. Here either key is read.
Both are counted in the output (extractor_used, extractor_key_model_output) so their effect is visible.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hygrag_close_extract_prompt import CLOSE_EXTRACT_OPTION_PROMPT  # noqa: E402
from prompts import parse_option_letter  # noqa: E402

DATA_ROOT = Path(__file__).resolve().parents[2] / "evaluation_data"
SYSTEMS = ("hipporag2", "logicrag", "memgraphrag", "ms_graphrag", "hypergraphrag", "hygrag", "unh", "unh_hypergraph", "unh_discourse")


class Extractor:
    """HyGRAG's LLM option extractor, cached."""

    def __init__(self, model, base_url, api_key, cache_path: Path):
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=300, max_retries=5)
        self.params = {"model": model}
        if os.environ.get("LLM_TEMPERATURE"):
            self.params["temperature"] = float(os.environ["LLM_TEMPERATURE"])
        if os.environ.get("LLM_DISABLE_THINKING", "1") != "0":
            self.params["extra_body"] = {"thinking": {"type": "disabled"}}
        self.conn = sqlite3.connect(str(cache_path), check_same_thread=False)
        self.conn.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, response TEXT)")
        self.lock = threading.Lock()

    def __call__(self, question: str, model_output: str):
        prompt = CLOSE_EXTRACT_OPTION_PROMPT.format(question=question, model_output=model_output)
        request = {**self.params, "messages": [{"role": "user", "content": prompt}]}
        key = hashlib.sha256(json.dumps(request, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
        with self.lock:
            row = self.conn.execute("SELECT response FROM cache WHERE key = ?", (key,)).fetchone()
        if row is not None:
            raw = row[0]
        else:
            raw = self.client.chat.completions.create(**request).choices[0].message.content or ""
            if raw:
                with self.lock:
                    self.conn.execute("INSERT OR REPLACE INTO cache VALUES (?, ?)", (key, raw))
                    self.conn.commit()
        found = re.search(r"\{.*\}", raw, re.S)
        try:
            obj = json.loads(found.group(0)) if found else {}
        except json.JSONDecodeError:
            obj = {}
        key_used = "predict" if "predict" in obj else "model_output" if "model_output" in obj else None
        value = str(obj.get(key_used, "-1")).strip().strip(".()").upper() if key_used else "-1"
        return (value if value in ("A", "B", "C", "D") else None), key_used, raw


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--system", required=True, choices=SYSTEMS)
    ap.add_argument("--predictions", required=True, help="answers_<system>_quality.json from common/answer.py")
    ap.add_argument("--questions", default=None,
                    help="default: $QUALITY_INPUTS/questions.json or evaluation_data/quality/inputs/questions.json")
    ap.add_argument("--out", default=None, help="default: <predictions dir>/scores_<system>_quality.json")
    ap.add_argument("--model", default=os.environ.get("LLM_MODEL", "deepseek-flash"))
    ap.add_argument("--base-url", default=os.environ.get("LLM_BASE_URL", "https://api.deepseek.com"))
    ap.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    qpath = args.questions or Path(os.environ.get("QUALITY_INPUTS", DATA_ROOT / "quality" / "inputs")) / "questions.json"
    samples = json.load(open(qpath, encoding="utf-8"))
    preds = json.load(open(args.predictions, encoding="utf-8"))
    if len(preds) > len(samples):
        sys.exit(f"{len(preds)} predictions but only {len(samples)} questions")
    for i, (p, s) in enumerate(zip(preds, samples)):
        if p["question"].strip() != s["question"].strip():
            sys.exit(f"question mismatch at index {i}: {p['question'][:80]!r} vs {s['question'][:80]!r}")
    samples = samples[: len(preds)]

    rows = []
    for p, s in zip(preds, samples):
        letter = p["pred_answer"] if p["pred_answer"] in ("A", "B", "C", "D") else parse_option_letter(p.get("raw_response", ""))
        docs = p.get("retrieved_docs") or []
        rows.append({"id": s["id"], "article_id": s["article_id"], "difficult": s["difficult"],
                     "gold": s["answer_letter"], "pred_letter": letter, "letter_source": "reply" if letter else None,
                     "raw_response": p.get("raw_response", ""), "context_chars": sum(len(d or "") for d in docs)})

    todo = [r for r, s in zip(rows, samples) if r["pred_letter"] is None]
    if todo:
        extractor = Extractor(args.model, args.base_url, os.environ[args.api_key_env],
                              Path(args.predictions).resolve().parent / "extract_cache.sqlite")
        qtext = {s["id"]: s["question"] for s in samples}
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            results = list(ex.map(lambda r: extractor(qtext[r["id"]], r["raw_response"]), todo))
        for r, (letter, key_used, raw) in zip(todo, results):
            r.update(pred_letter=letter, letter_source="hygrag_extractor", extractor_key=key_used, extractor_raw=raw)

    for r in rows:
        r["correct"] = float(r["pred_letter"] == r["gold"])

    def acc(subset):
        return round(100 * sum(r["correct"] for r in subset) / len(subset), 2) if subset else None

    hard = [r for r in rows if r["difficult"]]
    easy = [r for r in rows if not r["difficult"]]
    summary = {"Accuracy": acc(rows), "Acc-HARD": acc(hard), "Acc-EASY": acc(easy)}
    counts = {
        "letter_from_reply": sum(r["letter_source"] == "reply" for r in rows),
        "extractor_used": len(todo),
        "extractor_key_model_output": sum(r.get("extractor_key") == "model_output" for r in rows),
        "no_option": sum(r["pred_letter"] is None for r in rows),
        "empty_context": sum(r["context_chars"] == 0 for r in rows),
        "mean_context_chars": round(sum(r["context_chars"] for r in rows) / len(rows)),
        "hard_questions": len(hard), "easy_questions": len(easy),
        "pred_letter_distribution": {k: sum(r["pred_letter"] == k for r in rows) for k in "ABCD"},
    }
    result = {"dataset": "quality", "system": args.system, "predictions": os.path.abspath(args.predictions),
              "questions": os.path.abspath(qpath), "num_questions": len(rows), "metrics_percent": summary,
              "counts": counts, "extractor_model": args.model if todo else None, "judge_model": None,
              "per_example": rows}
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.predictions)), f"scores_{args.system}_quality.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: result[k] for k in ("dataset", "system", "num_questions", "metrics_percent", "counts")}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
