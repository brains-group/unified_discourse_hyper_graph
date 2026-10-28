#!/usr/bin/env python3
"""Step 3 — QA over the GraphRAG local-search context.

[unh_v1] Replaces SAG-Benchmark's Step_3 (kept for reference as Step_3_qa_eval.sag_upstream.py.txt).
Upstream shelled out to SAG's own scripts/run_qa_benchmark.py, which needs SAG's whole pipeline
package and uses SAG's own zero-shot reader prompt. Here the reader is the HippoRAG 2 /
MemGraphRAG `rag_qa_musique` one-shot prompt (evaluation_scripts/common/prompts.py), with the same
generation settings as HippoRAG 2 (temperature 0, max_tokens 2048) and the same "Wikipedia Title: "
formatting of each retrieved doc that SAG used. So MS GraphRAG is read exactly like the other systems.

Input : outputs/<ds>/response/graphrag_<ds>_result.json   (Step 2)
Output: outputs/<ds>/predictions.json   -> score with evaluation_scripts/common/score.py --system ms_graphrag
"""
import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from _common import OUTPUT_ROOT, PROJECT_ROOT, load_env, log

sys.path.insert(0, str(PROJECT_ROOT.parent / "common"))
from prompts import build_rag_qa_messages, extract_answer  # noqa: E402


def main():
    load_env()
    ap = argparse.ArgumentParser(description="Step 3: QA with the HippoRAG 2 reader prompt")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--qa-top-k", type=int, default=5, help="GraphRAG yields one context doc per question, so this is a no-op")
    ap.add_argument("--max-concurrency", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from openai import OpenAI
    client = OpenAI(api_key=os.environ["LLM_API_KEY"], base_url=os.environ["LLM_BASE_URL"])
    model = os.environ["LLM_MODEL"]

    ds = args.dataset
    src = OUTPUT_ROOT / ds / "response" / f"graphrag_{ds}_result.json"
    rows = json.load(open(src, encoding="utf-8"))
    if args.limit:
        rows = rows[: args.limit]

    def answer(row):
        messages = build_rag_qa_messages(row["question"], row["retrieved_docs"], args.qa_top_k)
        resp = client.chat.completions.create(model=model, messages=messages, temperature=0, max_tokens=args.max_tokens)
        text = resp.choices[0].message.content or ""
        return text, extract_answer(text)

    log(f"[qa] {ds}: {len(rows)} questions, model={model}")
    with ThreadPoolExecutor(max_workers=args.max_concurrency) as ex:
        outputs = list(ex.map(answer, rows))

    predictions = [
        {"id": row["dataset_sample_id"], "question": row["question"], "pred_answer": pred,
         "raw_response": raw, "retrieved_docs": row["retrieved_docs"]}
        for row, (raw, pred) in zip(rows, outputs)
    ]
    out = OUTPUT_ROOT / ds / "predictions.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(predictions, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    log(f"[qa] wrote {out}")
    log(f"score: python {PROJECT_ROOT.parent / 'common' / 'score.py'} --dataset {ds} --system ms_graphrag --predictions {out}")


if __name__ == "__main__":
    main()
