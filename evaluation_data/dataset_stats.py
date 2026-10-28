#!/usr/bin/env python3
"""Corpus and question size of every dataset in evaluation_data/, in tokens, with a rough construction-cost projection.

    python evaluation_data/dataset_stats.py              # markdown tables
    python evaluation_data/dataset_stats.py --json out.json

Tokens are counted with tiktoken cl100k_base (the tokenizer HyGRAG/DIGIMON and most of our harnesses use).
DeepSeek's own tokenizer differs somewhat, so treat token counts as close estimates.

Projection: input/output LLM tokens spent during index construction per corpus token, measured on our full
HotpotQA and MuSiQue runs (2026-09-30, deepseek-flash; evaluation_scripts caches). Projection = corpus tokens x
that ratio (mean of the two datasets). It is only an order-of-magnitude guide for other corpora: per-call prompt
overhead is amortised differently for long documents (QuALITY articles, contracts) than for ~130-token Wikipedia
passages, and MemGraphRAG's cost grows with the number of extracted facts.
"""
import argparse
import glob
import json
import statistics as st
from pathlib import Path

import tiktoken

HERE = Path(__file__).resolve().parent
ENC = tiktoken.get_encoding("cl100k_base")

# construction tokens per corpus token: (input, output); corpus = 1,299,764 (HotpotQA) / 1,348,035 (MuSiQue) tokens
MEASURED = {
    "hipporag2": ((9_247_043 / 1_299_764 + 10_495_687 / 1_348_035) / 2, (3_591_407 / 1_299_764 + 3_749_472 / 1_348_035) / 2),
    "hygrag": ((13_771_886 / 1_299_764 + 15_258_761 / 1_348_035) / 2, (4_328_048 / 1_299_764 + 4_518_913 / 1_348_035) / 2),
    "ms_graphrag": ((47_370_497 / 1_299_764 + 49_327_181 / 1_348_035) / 2, (21_401_995 / 1_299_764 + 22_425_312 / 1_348_035) / 2),
    "memgraphrag": ((158_704_552 / 1_299_764 + 153_453_450 / 1_348_035) / 2, (16_514_806 / 1_299_764 + 16_051_494 / 1_348_035) / 2),
    "hypergraphrag": ((161_932_289 / 1_299_764 + 189_948_058 / 1_348_035) / 2, (17_567_770 / 1_299_764 + 19_591_573 / 1_348_035) / 2),
}


def ntok(s: str) -> int:
    return len(ENC.encode(s, disallowed_special=()))


def chunks_1200(tokens: int) -> int:
    return 1 if tokens <= 1200 else 1 + -(-(tokens - 1200) // 1100)


def stats(name, docs, questions, note=""):
    d = [ntok(t) for t in docs]
    q = [ntok(t) for t in questions]
    return {"dataset": name, "docs": len(d), "corpus_tokens": sum(d), "doc_tokens_mean": round(st.mean(d)),
            "doc_tokens_median": round(st.median(d)), "doc_tokens_max": max(d),
            "chunks_1200_100": sum(chunks_1200(x) for x in d), "questions": len(q),
            "question_tokens_mean": round(st.mean(q), 1) if q else 0, "note": note}


def load_all():
    rows = []
    for ds in ("hotpotqa", "musique"):
        corpus = json.load(open(HERE / ds / f"{ds}_corpus.json"))
        qs = json.load(open(HERE / ds / f"{ds}.json"))
        rows.append(stats(ds, [f"{c['title']}\n{c['text']}" for c in corpus], [q["question"] for q in qs], "HippoRAG 2 release, per passage"))
    corpus = json.load(open(HERE / "popqa" / "popqa_corpus.json"))
    qs = json.load(open(HERE / "popqa" / "popqa.json"))
    rows.append(stats("popqa (full)", [f"{c['title']}\n{c['text']}" for c in corpus], [q["question"] for q in qs], "DIGIMON copy, long-tail 1,399"))
    corpus = json.load(open(HERE / "quality" / "quality_corpus.json"))
    qs = json.load(open(HERE / "quality" / "quality.json"))
    rows.append(stats("quality (dev)", [c["text"] for c in corpus], [q["question_with_options"] for q in qs], "official dev, per article"))
    digi = HERE.parent / "HyGRAG" / "Data" / "quality"
    if digi.exists():
        corpus = [json.loads(l) for l in open(digi / "Corpus.json")]
        qs = [json.loads(l) for l in open(digi / "Question.json")]
        rows.append(stats("quality (DIGIMON train+dev pooled)", [c["context"] for c in corpus], [q["question"] for q in qs], "as in HyGRAG paper"))
    for sub in sorted((HERE / "subsets").glob("*")):
        corpus = json.load(open(sub / "corpus.json"))
        qs = json.load(open(sub / "questions.json"))
        texts = [f"{c['title']}\n{c['text']}" if sub.name.startswith("popqa") else c["text"] for c in corpus]
        rows.append(stats(f"subset {sub.name}", texts, [q.get("question_with_options", q["question"]) for q in qs], "seeded subset"))
    for ds in ("long_form_contracts", "ten_contracts_dataset"):
        docs = [open(f, encoding="utf-8").read() for f in sorted(glob.glob(str(HERE / ds / "contracts" / "*.txt")))]
        qs = [qa["question"] for f in sorted(glob.glob(str(HERE / ds / "qa_pairs" / "*.json"))) for qa in json.load(open(f))["qa_pairs"]]
        rows.append(stats(ds, docs, qs, "insurance contracts, per contract"))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    rows = load_all()
    print("| dataset | docs | corpus tokens | tokens/doc mean / median / max | chunks @1200/100 | questions | tokens/question |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['dataset']} | {r['docs']:,} | {r['corpus_tokens']:,} | {r['doc_tokens_mean']:,} / {r['doc_tokens_median']:,} / "
              f"{r['doc_tokens_max']:,} | {r['chunks_1200_100']:,} | {r['questions']:,} | {r['question_tokens_mean']} |")
    systems = list(MEASURED)
    print("\nProjected construction LLM tokens, input / output (millions), from our HotpotQA+MuSiQue ratios:")
    print("| dataset | " + " | ".join(systems) + " |")
    print("|---|" + "---|" * len(systems))
    for r in rows:
        cells = [f"{r['corpus_tokens'] * MEASURED[s][0] / 1e6:,.1f} / {r['corpus_tokens'] * MEASURED[s][1] / 1e6:,.1f}" for s in systems]
        print(f"| {r['dataset']} | " + " | ".join(cells) + " |")
    print("\nratios used (input, output per corpus token): " + ", ".join(f"{s} {a:.1f}/{b:.1f}" for s, (a, b) in MEASURED.items()))
    if args.json:
        json.dump({"rows": rows, "ratios": MEASURED}, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
