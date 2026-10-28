#!/usr/bin/env python3
"""Check evaluation_data/ against the SHA-256 digests of the HippoRAG 2 release.

The .json files must be byte-identical to https://huggingface.co/datasets/osunlp/HippoRAG_2
(== OSU-NLP-Group/HippoRAG reproduce/dataset == XMUDeepLIT/MemGraphRAG dataset/, see
evaluation_data/README.md). The .txt files are MemGraphRAG's, which equal
"\\n\\n".join(title + "\\n" + text) + "\\n\\n" over the corpus; that identity is re-checked here too.
"""
import hashlib
import json
import sys
from pathlib import Path

DATA_ROOT = Path(__file__).resolve().parents[2] / "evaluation_data"
EXPECTED = {
    "hotpotqa/hotpotqa.json": "3ad9c0bcbf93f41d7004ca6007049c904d2605046b314a2f7ecfb379c64cba6d",
    "hotpotqa/hotpotqa_corpus.json": "9333647b922382776cd2cb02893b390d77984df85a91bfa8be411284978aca7d",
    "hotpotqa/hotpotqa.txt": "18391e2db6b9e85b814d3dfc98f47fb6232f9818655d26a55d8bd22b7dbcbea3",
    "musique/musique.json": "98ed4e21d3076532f6388d42320fb809599c63a0d8dffca8ece5e41922be6b46",
    "musique/musique_corpus.json": "73157a03ce3f0b1a5673dd5dc12bb970c24976dbffc688af9eecdd758c97ffcb",
    "musique/musique.txt": "5ede356cff41a5dd1327118f65e49c8fe542c6e117ee9064de34d94ab8a18e71",
}


def main() -> int:
    ok = True
    for rel, digest in EXPECTED.items():
        got = hashlib.sha256((DATA_ROOT / rel).read_bytes()).hexdigest()
        status = "OK " if got == digest else "BAD"
        ok &= got == digest
        print(f"{status} {rel}")
    for ds in ("hotpotqa", "musique"):
        corpus = json.load(open(DATA_ROOT / ds / f"{ds}_corpus.json", encoding="utf-8"))
        rebuilt = "\n\n".join(d["title"] + "\n" + d["text"] for d in corpus) + "\n\n"
        same = rebuilt == (DATA_ROOT / ds / f"{ds}.txt").read_text(encoding="utf-8")
        ok &= same
        print(f"{'OK ' if same else 'BAD'} {ds}.txt == corpus rendered as title\\ntext blocks")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
