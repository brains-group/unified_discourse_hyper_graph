#!/usr/bin/env python3
"""G-Medical (GraphRAG-Bench Medical, as MemGraphRAG evaluates it) in the input format of every system in
evaluation_scripts/. Same layout as build_quality_inputs.py, so every runner's QuALITY code path reads it unchanged
(as the contracts harness already does); only the answer and score steps differ (open-ended answers, LLM-Acc).

    python evaluation_data/build_gmedical_inputs.py                 # all 2,062 questions, ONE document -> gmedical/inputs/
    python evaluation_data/build_gmedical_inputs.py --per-line --out evaluation_data/gmedical/inputs_per_line
                                                                    # alternative: 41 documents, one per distinct line
    python evaluation_data/build_gmedical_inputs.py --smoke         # 2 guidelines, their first 3 questions per type
                                                                    #   -> gmedical/smoke/inputs/ (about 4.3k words)

Source = MemGraphRAG's own copy (MemGraphRAG/dataset/medical/): medical.txt (one NCCN patient guideline per line,
44 lines, 3 of them exact duplicates: AML, CML, ALL) and question.json ([{"source": "nccn", "questions":
{"type1": [...], ..., "type4": [...]}}], 2,062 questions). MemGraphRAG itself reads medical.txt as one string and
cuts 256/32-token windows across guideline boundaries.

Corpus decision (user, 2026-10-11): ONE document = medical.txt verbatim (all 44 lines, duplicates kept, line breaks
kept), exactly the official GraphRAG-Bench Datasets/Corpus/medical.json `context` string (byte-identical) and what
MemGraphRAG ingests; every system's own chunker cuts across the line breaks. The lines are roughly one NCCN booklet
excerpt each, but the benchmark defines no document boundaries (some lines are fragments, e.g. a 220-token glossary
piece), so none are imposed. --per-line instead writes the 41 distinct lines as separate documents.
--smoke (needs per-line structure to pick 2 guidelines) always writes per-line documents.
What each system gets, written to <out>/:
  questions.json         every runner, answer.py, score_gmedical.py: {id, question, answer, question_type
                         (type1..type4), level (Fact Retrieval / Complex Reasoning / Contextual Summarize /
                         Creative Generation), source, evidence}, in MemGraphRAG's order (type1, type2, type3, type4)
  documents.json         HyGRAG, HyperGraphRAG, UNH: {article_id, title: "", text}; they chunk it themselves
  chunks.json            HippoRAG 2 (no chunker of its own): the document(s) cut exactly as MemGraphRAG's index.py and
                         GraphRAG-Bench's Examples/run_hipporag2.py cut them: 256/32 windows of the embedding model's HF
                         tokenizer, each passage prefixed "<idx>:" (user decision 2026-10-11; on QuALITY these systems
                         got HyGRAG's 1200/100, the setup that benchmark mirrors). One document -> the same 991 windows
                         MemGraphRAG indexes.
  chunks_logicrag.json   LogicRAG (no chunker either): the same chunks, {title: "", text}
  memgraphrag_corpus.txt MemGraphRAG: the document text(s), one per line (one document: medical.txt itself)
  ms_graphrag_input/     MS GraphRAG: <article_id>.txt, one per document
  manifest.json          counts and SHA-256 of every file
"""
import argparse
import hashlib
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from build_quality_inputs import sha256

CHUNK_TOKENS, CHUNK_OVERLAP = 256, 32   # MemGraphRAG run_index.sh / GraphRAG-Bench run_hipporag2.py
TOKENIZER = "voyageai/voyage-4-nano"    # the embedding model's tokenizer, as both use (memgraphrag/run_gmedical.sh)


def split_text(text: str, tokenizer, chunk_size: int, overlap: int) -> list:
    """MemGraphRAG code/index.py split_text, verbatim (GraphRAG-Bench's run_hipporag2.py split_text is the same)."""
    previous_limit = getattr(tokenizer, "model_max_length", None)
    try:
        tokenizer.model_max_length = int(1e9)
        token_ids = tokenizer.encode(text, add_special_tokens=False)
    finally:
        if previous_limit is not None:
            tokenizer.model_max_length = previous_limit
    chunks = []
    step = chunk_size - overlap
    for start in range(0, len(token_ids), step):
        chunk = token_ids[start:start + chunk_size]
        if not chunk:
            break
        chunks.append(tokenizer.decode(chunk, skip_special_tokens=True, clean_up_tokenization_spaces=True))
        if start + chunk_size >= len(token_ids):
            break
    return chunks

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "MemGraphRAG" / "dataset" / "medical"
SMOKE_SOURCES = ("65_Adrenal Tumors_processed", "17_Kidney Cancer_processed")  # small, unambiguous guidelines
LEVELS = {"type1": "Fact Retrieval", "type2": "Complex Reasoning", "type3": "Contextual Summarize",
          "type4": "Creative Generation"}


def infer_source_docs(docs: list, questions: list) -> dict:
    """question `source` -> index of the guideline its evidence sentences overlap most (word overlap, >3 letters).
    A heuristic (medical.txt carries no names): used only to pick smoke documents, never as gold."""
    words = lambda t: {w for w in re.sub(r"[^a-z0-9 ]", " ", t.lower()).split() if len(w) > 3}  # noqa: E731
    vocab = [words(d["text"]) for d in docs]
    votes = defaultdict(Counter)
    for q in questions:
        for e in q["evidence"] if isinstance(q["evidence"], list) else [q["evidence"]]:
            w = words(e)
            if w:
                votes[q["source"]][max(range(len(docs)), key=lambda i: len(w & vocab[i]))] += 1
    return {src: c.most_common(1)[0][0] for src, c in votes.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true", help=f"only the guidelines of {SMOKE_SOURCES} and their questions")
    ap.add_argument("--per-type", type=int, default=None, help="keep the first N questions of each type "
                    "(default: all; 3 per source with --smoke)")
    ap.add_argument("--per-line", action="store_true", help="41 documents, one per distinct line, instead of one document")
    ap.add_argument("--out", default=None, help="default: gmedical/inputs, or gmedical/smoke/inputs with --smoke")
    args = ap.parse_args()
    out = Path(args.out) if args.out else HERE / "gmedical" / ("smoke/inputs" if args.smoke else "inputs")

    raw = (SRC / "medical.txt").read_text(encoding="utf-8")
    lines = [l.strip() for l in raw.split("\n") if l.strip()]
    seen, docs = set(), []
    for line in lines:
        h = hashlib.md5(line.encode("utf-8")).hexdigest()
        if h not in seen:
            seen.add(h)
            docs.append({"article_id": f"med_{len(docs):02d}", "title": "", "text": line})

    groups = json.load(open(SRC / "question.json", encoding="utf-8"))[0]["questions"]
    questions = []
    for qtype in ("type1", "type2", "type3", "type4"):
        for q in groups[qtype]:
            questions.append({"id": q["id"], "question": q["question"].strip(), "answer": q["answer"],
                              "question_type": qtype, "level": LEVELS[qtype], "source": q["source"],
                              "evidence": q["evidence"]})
    assert len({q["id"] for q in questions}) == len(questions), "duplicate question ids"
    source_docs = infer_source_docs(docs, questions)
    if args.smoke:
        keep_docs = {source_docs[s] for s in SMOKE_SOURCES}
        docs = [d for i, d in enumerate(docs) if i in keep_docs]
        questions = [q for q in questions if q["source"] in SMOKE_SOURCES]
    cap = args.per_type if args.per_type is not None else (3 if args.smoke else 0)
    if cap:
        taken = Counter()
        kept = []
        for q in questions:
            key = (q["source"], q["question_type"]) if args.smoke else q["question_type"]
            if taken[key] < cap:
                taken[key] += 1
                kept.append(q)
        questions = kept

    if not (args.per_line or args.smoke):  # ONE document: the corpus verbatim (official medical.json context)
        docs = [{"article_id": "medical", "title": "", "text": raw.strip("\n")}]

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    chunks = []
    for d in docs:  # one document: split_text(medical.txt as index.py reads it) -> MemGraphRAG's own windows
        text = d["text"] + "\n" if d["article_id"] == "medical" else d["text"]  # = memgraphrag_corpus.txt
        for i, piece in enumerate(split_text(text, tokenizer, CHUNK_TOKENS, CHUNK_OVERLAP)):
            chunks.append({"article_id": d["article_id"], "chunk_index": i, "title": "",
                           "text": f"{len(chunks)}:{piece}"})  # "<idx>:" as MemGraphRAG / GraphRAG-Bench docs

    if out.exists():
        shutil.rmtree(out)
    (out / "ms_graphrag_input").mkdir(parents=True)
    dump = lambda obj, name: json.dump(obj, open(out / name, "w", encoding="utf-8"), ensure_ascii=False, indent=1)  # noqa: E731
    dump(questions, "questions.json")
    dump(docs, "documents.json")
    dump(chunks, "chunks.json")
    dump([{"title": c["title"], "text": c["text"]} for c in chunks], "chunks_logicrag.json")
    (out / "memgraphrag_corpus.txt").write_text("\n".join(d["text"] for d in docs) + "\n", encoding="utf-8")
    for d in docs:
        (out / "ms_graphrag_input" / f"{d['article_id']}.txt").write_text(d["text"], encoding="utf-8")

    files = ["questions.json", "documents.json", "chunks.json", "chunks_logicrag.json", "memgraphrag_corpus.txt"]
    manifest = {"source": str(SRC), "source_lines": len(lines), "distinct_guidelines": len(seen),
                "layout": "per_line" if (args.per_line or args.smoke) else "one_document (medical.txt verbatim)",
                "smoke_sources": list(SMOKE_SOURCES) if args.smoke else None, "documents": len(docs),
                "document_words": sum(len(d["text"].split()) for d in docs), "chunks": len(chunks),
                "chunking (HippoRAG 2, LogicRAG)": f"{CHUNK_TOKENS}/{CHUNK_OVERLAP} {TOKENIZER} tokens, MemGraphRAG split_text", "questions": len(questions),
                "questions_per_type": {t: sum(q["question_type"] == t for q in questions) for t in LEVELS},
                "inferred_source_to_document (heuristic, not gold)": {s: f"med_{i:02d}" for s, i in sorted(source_docs.items())},
                "sha256": {f: sha256(out / f) for f in files}}
    json.dump(manifest, open(out / "manifest.json", "w"), indent=2)
    print(json.dumps({k: v for k, v in manifest.items() if k != "sha256" and not k.startswith("inferred")}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
