#!/usr/bin/env python3
"""Insurance contracts in the input format of every system in evaluation_scripts/ (CONTRACTS_EVAL_PLAN.md).

    python evaluation_data/build_contract_inputs.py            # short + long + smoke -> evaluation_data/contracts/<set>/

Sets:
  short   ten_contracts_dataset/       10 synthetic life-insurance policies (1.3k-3.1k tokens each)
  long    long_form_contracts/         5 real contracts (6k-49k tokens each)
  smoke   one contract from each set (contract_1_term_life, 2025_pdp_contract), 2 QA per category + 10 facts each
  long_qa, smoke_qa   QA only (user decision 2026-10-11: no fact recall): the long / smoke set with every QA
          question and no fact queries; same documents and chunks, so every runner and contracts_score.py take the
          set name unchanged (no fact rows -> nothing fact-judged)

Protocol: one graph per contract per system; every QA question and every fact (verbatim) is a retrieval query
against its own contract's graph only. Each contract gets its own inputs dir, laid out exactly like a QuALITY inputs
dir (build_quality_inputs.py), so every runner's existing inputs-dir loader (`--dataset quality`) reads it unchanged:

  questions.json         every query of the contract, QA first then facts:
                         {id, kind: "qa"|"fact", question (= the query string), answer (QA gold; the fact itself for
                         facts), answer_letter: "" (field the inputs-dir loaders expect), category, evidence, fact_id,
                         contract}. ids: "<contract>/qa/<n>", "<contract>/fact/<fact id>".
  documents.json         HyGRAG, HyperGraphRAG: the whole contract as one document {article_id, title: "", text}.
  chunks.json            HippoRAG 2 (no chunker): the contract cut with HyGRAG's own chunker (tiktoken gpt-3.5-turbo,
                         1200 tokens, a new window every 1100), as for QuALITY.
  chunks_logicrag.json   LogicRAG (no chunker): the same chunks, title "".
  memgraphrag_corpus.txt MemGraphRAG: the contract text (it cuts its own 256/32-token windows).
  ms_graphrag_input/     MS GraphRAG: <contract>.txt.
  contract.txt           UNH: the contract text (UNH chunks it itself, 600/50).
  manifest.json          counts and SHA-256 of every file.

The text is the contract .txt with its `<!-- image -->` lines (PDF-conversion placeholders) removed; otherwise
byte-identical for every system. There is no title: a contract is one document, so `title` is "" everywhere.
"""
import hashlib
import json
import re
import shutil
from pathlib import Path

import tiktoken

HERE = Path(__file__).resolve().parent
OUT_ROOT = HERE / "contracts"
CHUNK_TOKENS, OVERLAP = 1200, 100  # HyGRAG config/HKGraphTreeLSH.yaml chunk_token_size / chunk_overlap_token_size
ENC = tiktoken.encoding_for_model("gpt-3.5-turbo")  # HyGRAG token_model: gpt-3.5-turbo (cl100k_base)
CATEGORIES = ["CAT1_ATOMIC_LOOKUP", "CAT2_RELATIONAL_LOOKUP", "CAT3_CONDITIONAL_REASONING",
              "CAT4_MULTIHOP_REASONING", "CAT5_UNANSWERABLE"]

SETS = {
    "short": {"dir": HERE / "ten_contracts_dataset", "qa": "qa_pairs/qa_{c}.json", "facts": "facts/facts_{c}.json"},
    "long": {"dir": HERE / "long_form_contracts", "qa": "qa_pairs/qa_{c}.json", "facts": "facts/{c}.json"},
}
SMOKE = [("short", "contract_1_term_life"), ("long", "2025_pdp_contract")]
SMOKE_QA_PER_CATEGORY, SMOKE_FACTS = 2, 10


def digimon_chunks(text: str) -> list:
    """HyGRAG Core/Chunk/DocChunk.build_chunks + Tokensize.chunking_by_token_size for one document."""
    tokens = ENC.encode(text.strip(), disallowed_special=())
    windows = [tokens[start:start + CHUNK_TOKENS] for start in range(0, len(tokens), CHUNK_TOKENS - OVERLAP)]
    return [chunk.strip() for chunk in ENC.decode_batch(windows)]


def clean(text: str) -> str:
    text = "\n".join(line for line in text.splitlines() if line.strip() != "<!-- image -->")
    return re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def contract_names(set_name: str) -> list:
    """Contracts in a stable order: by file name, numerically for contract_<n>_*."""
    names = [p.stem for p in (SETS[set_name]["dir"] / "contracts").glob("*.txt")]
    key = lambda n: (int(m.group(1)), n) if (m := re.match(r"contract_(\d+)_", n)) else (0, n)  # noqa: E731
    return sorted(names, key=key)


def load_contract(set_name: str, c: str):
    spec = SETS[set_name]
    text = clean((spec["dir"] / "contracts" / f"{c}.txt").read_text(encoding="utf-8"))
    qa = json.load(open(spec["dir"] / spec["qa"].format(c=c), encoding="utf-8"))["qa_pairs"]
    facts = json.load(open(spec["dir"] / spec["facts"].format(c=c), encoding="utf-8"))["facts"]
    return text, qa, facts


def write_contract(out: Path, c: str, text: str, qa: list, facts: list, source: str):
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    queries = []
    for n, q in enumerate(qa):
        assert q["category"] in CATEGORIES, q["category"]
        queries.append({"id": f"{c}/qa/{n:04d}", "kind": "qa", "question": q["question"].strip(),
                        "answer": q["answer"], "answer_letter": "", "category": q["category"],
                        "evidence": q.get("evidence", ""), "fact_id": None, "contract": c})
    for f in facts:
        queries.append({"id": f"{c}/fact/{f['id']}", "kind": "fact", "question": f["fact"].strip(),
                        "answer": f["fact"], "answer_letter": "", "category": "FACT", "evidence": "",
                        "fact_id": f["id"], "contract": c})
    ids = [q["id"] for q in queries]
    assert len(ids) == len(set(ids)), f"{c}: duplicate query ids"

    chunks = digimon_chunks(text)

    def dump(name, obj):
        json.dump(obj, open(out / name, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    dump("questions.json", queries)
    dump("documents.json", [{"article_id": c, "title": "", "text": text}])
    dump("chunks.json", [{"article_id": c, "chunk_index": i, "title": "", "text": t} for i, t in enumerate(chunks)])
    dump("chunks_logicrag.json", [{"article_id": c, "chunk_index": i, "title": "", "text": t} for i, t in enumerate(chunks)])
    (out / "memgraphrag_corpus.txt").write_text(text, encoding="utf-8")
    (out / "contract.txt").write_text(text, encoding="utf-8")
    (out / "ms_graphrag_input").mkdir()
    (out / "ms_graphrag_input" / f"{c}.txt").write_text(text, encoding="utf-8")
    files = ["questions.json", "documents.json", "chunks.json", "chunks_logicrag.json", "memgraphrag_corpus.txt",
             "contract.txt", f"ms_graphrag_input/{c}.txt"]
    manifest = {"contract": c, "source": source, "tokens_cl100k": len(ENC.encode(text, disallowed_special=())),
                "chars": len(text), "chunks_1200": len(chunks),
                "qa": len(qa), "facts": len(facts),
                "qa_by_category": {k: sum(q["category"] == k for q in qa) for k in CATEGORIES},
                "sha256": {f: sha256(out / f) for f in files}}
    dump("manifest.json", manifest)
    return manifest


def main():
    for set_name in ("short", "long"):
        rows = []
        for c in contract_names(set_name):
            text, qa, facts = load_contract(set_name, c)
            rows.append(write_contract(OUT_ROOT / set_name / c, c, text, qa, facts, str(SETS[set_name]["dir"].name)))
        json.dump({"set": set_name, "contracts": [r["contract"] for r in rows],
                   "totals": {k: sum(r[k] for r in rows) for k in ("tokens_cl100k", "qa", "facts", "chunks_1200")},
                   "per_contract": [{k: v for k, v in r.items() if k != "sha256"} for r in rows]},
                  open(OUT_ROOT / set_name / "contracts.json", "w"), indent=1)
        print(f"{set_name}: {len(rows)} contracts, {sum(r['qa'] for r in rows)} QA, {sum(r['facts'] for r in rows)} facts, "
              f"{sum(r['tokens_cl100k'] for r in rows):,} tokens")

    rows = []
    for set_name, c in SMOKE:
        text, qa, facts = load_contract(set_name, c)
        qa = [q for cat in CATEGORIES for q in [x for x in qa if x["category"] == cat][:SMOKE_QA_PER_CATEGORY]]
        rows.append(write_contract(OUT_ROOT / "smoke" / c, c, text, qa, facts[:SMOKE_FACTS], f"smoke from {set_name}"))
    json.dump({"set": "smoke", "contracts": [r["contract"] for r in rows],
               "totals": {k: sum(r[k] for r in rows) for k in ("tokens_cl100k", "qa", "facts", "chunks_1200")},
               "per_contract": [{k: v for k, v in r.items() if k != "sha256"} for r in rows]},
              open(OUT_ROOT / "smoke" / "contracts.json", "w"), indent=1)
    print(f"smoke: {len(rows)} contracts, {sum(r['qa'] for r in rows)} QA, {sum(r['facts'] for r in rows)} facts")

    # QA-only variants: the same contracts and texts, the QA questions only (no fact queries)
    for qa_set, src_set, picks in (("long_qa", "long", [("long", c) for c in contract_names("long")]), ("smoke_qa", "smoke", SMOKE)):
        rows = []
        for set_name, c in picks:
            text, qa, _ = load_contract(set_name, c)
            if src_set == "smoke":
                qa = [q for cat in CATEGORIES for q in [x for x in qa if x["category"] == cat][:SMOKE_QA_PER_CATEGORY]]
            rows.append(write_contract(OUT_ROOT / qa_set / c, c, text, qa, [], f"{qa_set} (QA only) from {set_name}"))
        json.dump({"set": qa_set, "contracts": [r["contract"] for r in rows],
                   "totals": {k: sum(r[k] for r in rows) for k in ("tokens_cl100k", "qa", "facts", "chunks_1200")},
                   "per_contract": [{k: v for k, v in r.items() if k != "sha256"} for r in rows]},
                  open(OUT_ROOT / qa_set / "contracts.json", "w"), indent=1)
        print(f"{qa_set}: {len(rows)} contracts, {sum(r['qa'] for r in rows)} QA, no facts")


if __name__ == "__main__":
    main()
