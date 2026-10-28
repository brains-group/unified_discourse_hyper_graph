#!/usr/bin/env python3
"""[unh_v1] Insurance contracts (run_contracts.sh): LogicRAG answers one query at a time, so each contract's queries
are split into shards that run as separate processes (same corpus, same settings), then merged back in query order.

    python contracts_shards.py split <questions.json> <shard dir> <shard size>   -> prints the shard names
    python contracts_shards.py merge <questions.json> <shard dir> <out.json>
    python contracts_shards.py done <shard questions.json> <shard result.json>    -> exit 0 if the shard is complete
"""
import json
import sys
from pathlib import Path


def split(questions, shard_dir, size):
    qs = json.load(open(questions, encoding="utf-8"))
    shard_dir = Path(shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)
    for k in range(0, len(qs), size):
        name = f"shard_{k // size:03d}"
        json.dump(qs[k:k + size], open(shard_dir / f"{name}.questions.json", "w", encoding="utf-8"), ensure_ascii=False)
        print(name)


def merge(questions, shard_dir, out):
    qs = json.load(open(questions, encoding="utf-8"))
    results = []
    for f in sorted(Path(shard_dir).glob("shard_*.questions.json")):
        name = f.name.replace(".questions.json", "")
        res = Path(shard_dir) / f"{name}.json"
        if not res.exists():
            sys.exit(f"missing shard result {res}")
        results += json.load(open(res, encoding="utf-8"))["results"]
    if len(results) != len(qs) or any(r["question"].strip() != q["question"].strip() for r, q in zip(results, qs)):
        sys.exit(f"merged {len(results)} results do not line up with the {len(qs)} queries of {questions}")
    missing = sum(r.get("info_summary") is None for r in results)
    json.dump({"results": results, "merged_from": str(shard_dir), "missing_info_summary": missing},
              open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"merged {len(results)} results -> {out} ({missing} without info_summary)")


if __name__ == "__main__":
    if sys.argv[1] == "split":
        split(sys.argv[2], sys.argv[3], int(sys.argv[4]))
    elif sys.argv[1] == "done":
        res = Path(sys.argv[3])
        ok = res.exists() and len(json.load(open(res)).get("results", [])) == len(json.load(open(sys.argv[2])))
        sys.exit(0 if ok else 1)
    elif sys.argv[1] == "merge":
        merge(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        sys.exit(__doc__)
