#!/usr/bin/env python3
"""Write <ds>/<ds>_scrambled.txt: the same passages as <ds>/<ds>.txt, in a shuffled order.

<ds>.txt (MemGraphRAG's index.py input) stores the corpus in question order, so the passages of
one question sit next to each other. MemGraphRAG cuts 256-token windows across passage
boundaries, which puts all gold passages of a question in one window for ~31% of HotpotQA
questions. Shuffling the passage order breaks that adjacency while keeping every passage
byte-for-byte unchanged.

Format is identical to <ds>.txt: "\\n\\n".join(title + "\\n" + text) + "\\n\\n", built from
<ds>_corpus.json. The shuffle uses a fixed seed, so the output is reproducible.

    python evaluation_data/make_scrambled_txt.py                      # hotpotqa, seed 0
    python evaluation_data/make_scrambled_txt.py --dataset musique --seed 0
"""
import argparse
import json
import random
from pathlib import Path

DATA_ROOT = Path(__file__).resolve().parent


def block(doc: dict) -> str:
    return f"{doc['title']}\n{doc['text']}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="hotpotqa", choices=["hotpotqa", "musique"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="default: <ds>/<ds>_scrambled.txt")
    args = ap.parse_args()

    ds_dir = DATA_ROOT / args.dataset
    corpus = json.load(open(ds_dir / f"{args.dataset}_corpus.json", encoding="utf-8"))
    blocks = [block(d) for d in corpus]

    # The original .txt must be exactly the unshuffled join; otherwise the formats would differ.
    original = (ds_dir / f"{args.dataset}.txt").read_text(encoding="utf-8")
    if "\n\n".join(blocks) + "\n\n" != original:
        raise SystemExit(f"{args.dataset}.txt is not the plain join of {args.dataset}_corpus.json; format check failed")

    order = list(range(len(blocks)))
    random.Random(args.seed).shuffle(order)
    scrambled = "\n\n".join(blocks[i] for i in order) + "\n\n"

    # Same passages, same count, only the order changed.
    assert sorted(blocks[i] for i in order) == sorted(blocks)
    assert len(scrambled) == len(original)

    out = Path(args.out) if args.out else ds_dir / f"{args.dataset}_scrambled.txt"
    out.write_text(scrambled, encoding="utf-8")
    moved = sum(1 for pos, i in enumerate(order) if pos != i)
    print(f"wrote {out}: {len(blocks)} passages, seed {args.seed}, {moved} moved from their original position")


if __name__ == "__main__":
    main()
