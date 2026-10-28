#!/usr/bin/env python3
"""Write evaluation_data/<ds> in the DIGIMON / HyGRAG input format.

    python prepare_data.py --dataset hotpotqa            # -> data/hotpotqa/{Corpus.json,Question.json}

HyGRAG (a DIGIMON fork) reads Data/<ds>/Corpus.json: JSONL, one document per row, {"title", "context"}.
DocChunk chunks every document on its own (1200/100 tokens, never across documents), so each passage here
is its own document and, being shorter than 1200 tokens (all but 2 HotpotQA passages), its own chunk.

context = "title\\ntext", the same passage text every other system in evaluation_scripts indexes. DIGIMON
keeps `title` only as metadata that the LLM never sees, so the title goes into `context` as well.

QuALITY (--dataset quality): one row per dev article from <quality inputs>/documents.json, context = "title\ntext"
(HyGRAG chunks each article 1200/100 itself, as for DIGIMON's own QuALITY copy); questions from questions.json,
where `question` already carries the four options, as in DIGIMON's Question.json.

Question.json: JSONL {"id", "question", "answer"} in dataset order. HyGRAG's runner here only reads the
question; answers are scored by common/score.py against evaluation_data.
"""
import argparse
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA_ROOT = HERE.parents[1] / "evaluation_data"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=["hotpotqa", "musique", "quality"])
    ap.add_argument("--quality-inputs", default=os.environ.get("QUALITY_INPUTS", str(DATA_ROOT / "quality" / "inputs")),
                    help="--dataset quality: dir with documents.json / questions.json (evaluation_data/build_quality_inputs.py)")
    ap.add_argument("--out-dir", default=None, help="default: data/<dataset>")
    ap.add_argument("--limit-docs", type=int, default=0, help="first N passages only (smoke tests)")
    ap.add_argument("--limit-q", type=int, default=0, help="first N questions only (smoke tests)")
    args = ap.parse_args()

    if args.dataset == "quality":  # [unh_v1] documents.json rows are {article_id, title, text}: one per article
        corpus = json.load(open(Path(args.quality_inputs) / "documents.json", encoding="utf-8"))
        questions = json.load(open(Path(args.quality_inputs) / "questions.json", encoding="utf-8"))
    else:
        corpus = json.load(open(DATA_ROOT / args.dataset / f"{args.dataset}_corpus.json", encoding="utf-8"))
        questions = json.load(open(DATA_ROOT / args.dataset / f"{args.dataset}.json", encoding="utf-8"))
    if args.limit_docs:
        corpus = corpus[: args.limit_docs]
    if args.limit_q:
        questions = questions[: args.limit_q]

    out = Path(args.out_dir) if args.out_dir else HERE / "data" / args.dataset
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "Corpus.json", "w", encoding="utf-8") as f:
        for doc in corpus:
            f.write(json.dumps({"title": doc["title"], "context": f"{doc['title']}\n{doc['text']}"}, ensure_ascii=False) + "\n")
    with open(out / "Question.json", "w", encoding="utf-8") as f:
        for q in questions:
            answer = q.get("answer", q.get("answers", ""))
            f.write(json.dumps({"id": q.get("id", q.get("_id")), "question": q["question"], "answer": answer},
                               ensure_ascii=False, default=str) + "\n")
    print(f"wrote {out}: {len(corpus)} documents (one row each), {len(questions)} questions")


if __name__ == "__main__":
    main()
