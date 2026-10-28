#!/usr/bin/env python3
"""QuALITY (official dev split) in the input format of every system in evaluation_scripts/.

    python evaluation_data/build_quality_inputs.py                                   # full dev: quality/inputs/
    python evaluation_data/build_quality_inputs.py --subset subsets/quality_100q_seed0 # smoke: <subset>/inputs/

Protocol (evaluation_scripts/QUALITY_EVAL_PLAN.md): one pooled index per system over all articles, every
question asked against the whole pool with its four options, one shared letter-only answer step.

Every article is the document "title\\ntext" (the article text itself usually starts with the title too). What
each system gets, written to <out>/:

  questions.json         every runner, answer.py, score_quality.py. `question` = question + options, exactly as
                         HyGRAG/DIGIMON's Question.json ("...?\\nA: ...\\nB: ...\\nC: ...\\nD: ..."); gold `answer_letter`,
                         `answer` (option text), `difficult` (official HARD flag), `article_id`.
  documents.json         HyGRAG, HyperGraphRAG: one entry per article {article_id, title, text}; the runners
                         index f"{title}\\n{text}" and chunk it themselves (1200/100 tokens, per document).
  chunks.json            HippoRAG 2 (no chunker of its own): "title\\ntext" cut with HyGRAG/DIGIMON's own chunking
                         loop (Core/Chunk/Tokensize.py: tiktoken gpt-3.5-turbo, 1200 tokens, a new window every
                         1100 tokens, each article on its own, decoded and stripped). Only an article's first chunk
                         carries the title. {article_id, chunk_index, title: "", text}
  chunks_logicrag.json   LogicRAG (no chunker; formats every unit as "Title: {title}. Content: {text}"): the article
                         text WITHOUT the title line, cut the same way; `title` = article title on chunk 0 and ""
                         on later chunks, so the title appears once per article, as for every other system.
  memgraphrag_corpus.txt MemGraphRAG (reads one .txt and cuts 256/32-token windows itself): the documents joined by
                         a blank line, in corpus order.
  ms_graphrag_input/     MS GraphRAG (one document per .txt file, chunked per document): <article_id>.txt = "title\\ntext".
  manifest.json          counts and SHA-256 of every file.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import tiktoken

HERE = Path(__file__).resolve().parent
CHUNK_TOKENS, OVERLAP = 1200, 100  # HyGRAG config/HKGraphTreeLSH.yaml chunk_token_size / chunk_overlap_token_size
ENC = tiktoken.encoding_for_model("gpt-3.5-turbo")  # HyGRAG token_model: gpt-3.5-turbo (cl100k_base)


def digimon_chunks(text: str) -> list:
    """HyGRAG Core/Chunk/DocChunk.build_chunks + Tokensize.chunking_by_token_size for one document."""
    tokens = ENC.encode(text.strip(), disallowed_special=())
    windows = [tokens[start:start + CHUNK_TOKENS] for start in range(0, len(tokens), CHUNK_TOKENS - OVERLAP)]
    return [chunk.strip() for chunk in ENC.decode_batch(windows)]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subset", default=None, help="a subsets/quality_* dir (relative to evaluation_data/); default: full dev")
    ap.add_argument("--out", default=None, help="default: quality/inputs or <subset>/inputs")
    args = ap.parse_args()

    articles = json.load(open(HERE / "quality" / "quality_corpus.json", encoding="utf-8"))
    if args.subset:
        src = HERE / args.subset
        questions = json.load(open(src / "questions.json", encoding="utf-8"))
        wanted = {q["article_id"] for q in questions}
        articles = [a for a in articles if a["article_id"] in wanted]
        out = Path(args.out) if args.out else src / "inputs"
    else:
        questions = json.load(open(HERE / "quality" / "quality.json", encoding="utf-8"))
        out = Path(args.out) if args.out else HERE / "quality" / "inputs"
    ids = [a["article_id"] for a in articles]
    assert len(ids) == len(set(ids)), "duplicate article_id"
    missing = {q["article_id"] for q in questions} - set(ids)
    assert not missing, f"questions refer to articles not in the corpus: {missing}"
    out.mkdir(parents=True, exist_ok=True)

    qs = [{"id": q["id"], "question": q["question_with_options"], "question_text": q["question"],
           "options": q["options"], "answer": q["answer"], "answer_letter": q["answer_letter"],
           "difficult": bool(q["difficult"]), "article_id": q["article_id"], "title": q["title"]} for q in questions]
    for q in qs:
        assert q["answer_letter"] in "ABCD" and q["question"].startswith(q["question_text"])

    docs = [{"article_id": a["article_id"], "title": a["title"], "text": a["text"]} for a in articles]
    chunks, chunks_logic = [], []
    for a in articles:
        for i, c in enumerate(digimon_chunks(f"{a['title']}\n{a['text']}")):
            chunks.append({"article_id": a["article_id"], "chunk_index": i, "title": "", "text": c})
        for i, c in enumerate(digimon_chunks(a["text"])):
            chunks_logic.append({"article_id": a["article_id"], "chunk_index": i,
                                 "title": a["title"] if i == 0 else "", "text": c})

    def dump(name, obj):
        json.dump(obj, open(out / name, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    dump("questions.json", qs)
    dump("documents.json", docs)
    dump("chunks.json", chunks)
    dump("chunks_logicrag.json", chunks_logic)
    (out / "memgraphrag_corpus.txt").write_text("\n\n".join(f"{a['title']}\n{a['text']}" for a in articles), encoding="utf-8")
    ms_dir = out / "ms_graphrag_input"
    shutil.rmtree(ms_dir, ignore_errors=True)
    ms_dir.mkdir()
    for a in articles:
        (ms_dir / f"{a['article_id']}.txt").write_text(f"{a['title']}\n{a['text']}", encoding="utf-8")

    files = ["questions.json", "documents.json", "chunks.json", "chunks_logicrag.json", "memgraphrag_corpus.txt"]
    manifest = {
        "source": args.subset or "quality (official QuALITY v1.0.1 dev)",
        "articles": len(docs), "questions": len(qs), "hard_questions": sum(q["difficult"] for q in qs),
        "chunks": len(chunks), "chunks_logicrag": len(chunks_logic),
        "chunker": "HyGRAG Tokensize.chunking_by_token_size: tiktoken gpt-3.5-turbo, 1200 tokens, stride 1100, per article",
        "answer_letters": {k: sum(q["answer_letter"] == k for q in qs) for k in "ABCD"},
        "sha256": {f: sha256(out / f) for f in files},
        "ms_graphrag_input_files": len(list(ms_dir.glob("*.txt"))),
    }
    json.dump(manifest, open(out / "manifest.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in manifest.items() if k != "sha256"}, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
