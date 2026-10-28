#!/usr/bin/env python3
"""
prepare_hotpotqa_corpus.py
==========================

Reads a HotpotQA JSON file (e.g. hotpot_dev_distractor_v1.json) and produces
two JSONL artifacts:

    corpus.jsonl    — one row per UNIQUE paragraph (deduped by Wikipedia title)
                      fields: id, title, paragraph, full_context, n_sentences
    questions.jsonl — one row per question, with references to corpus IDs
                      fields: question_id, question, answer, type, level,
                              candidate_paragraph_ids,    (the 10 distractor pool)
                              supporting_paragraph_ids,   (the 2 gold paragraphs)
                              supporting_titles,
                              supporting_facts            ([{paragraph_id, title, sentence_idx}, ...])

Usage:
    # Full dev set (all ~7405 questions)
    python prepare_hotpotqa_corpus.py \\
        --input ./data/hotpot_dev_distractor_v1.json \\
        --out-corpus    ./data/hotpot_corpus.jsonl \\
        --out-questions ./data/hotpot_questions.jsonl

    # Reproducible 1000-question stratified sample
    python prepare_hotpotqa_corpus.py \\
        --input ./data/hotpot_dev_distractor_v1.json \\
        --sample-size 1000 \\
        --seed 42 \\
        --out-corpus    ./data/hotpot_corpus_1k.jsonl \\
        --out-questions ./data/hotpot_questions_1k.jsonl

Sampling notes:
    - Stratification is by (type, level) — e.g. (bridge, hard), (comparison, medium).
      This preserves the dataset's natural difficulty and type distribution in the sample.
    - Remaining questions after proportional allocation are filled by random draw from
      the largest strata (avoids off-by-one rounding issues).
    - The corpus output contains ONLY the paragraphs referenced by the sampled questions,
      so it is a valid closed-world subset for KG construction and retrieval evaluation.
    - With --seed and --sample-size fixed, output is fully deterministic across machines.

Other notes:
    - Deduplication key is the Wikipedia title.
    - Paragraph IDs are stable across runs only if --id-format=hash is used.
      The default 'sequential' format gives clean para_000001-style IDs that
      depend on iteration order; pass --id-format=hash if you need cross-run stability.
    - Output is JSONL (one JSON object per line).
"""

import os
import json
import random
import hashlib
import argparse
from pathlib import Path
from collections import Counter, defaultdict


def make_paragraph_id(title: str, idx: int, fmt: str) -> str:
    """Build a paragraph ID using the chosen format."""
    if fmt == "hash":
        h = hashlib.md5(title.encode("utf-8")).hexdigest()[:12]
        return f"para_{h}"
    elif fmt == "sequential":
        return f"para_{idx:06d}"
    else:
        raise ValueError(f"Unknown id-format: {fmt}")


def stratified_sample(data: list, sample_size: int, seed: int) -> list:
    """
    Stratified sample by (type, level).
    Allocates proportionally, distributing any remainder to the largest strata.
    Returns a shuffled list of length min(sample_size, len(data)).
    """
    if sample_size >= len(data):
        print(f"    sample-size ({sample_size:,}) >= dataset size ({len(data):,}); "
              f"using full dataset")
        return data

    # Group by stratum key
    strata = defaultdict(list)
    for q in data:
        key = (q.get("type", "unknown"), q.get("level", "unknown"))
        strata[key].append(q)

    total = len(data)
    allocations = {}
    for key, members in strata.items():
        allocations[key] = max(1, round(len(members) / total * sample_size))

    # Adjust total to exactly hit sample_size
    rng = random.Random(seed)
    allocated = sum(allocations.values())
    diff = sample_size - allocated
    # Sort keys by stratum size descending so the largest strata absorb rounding
    sorted_keys = sorted(strata.keys(), key=lambda k: len(strata[k]), reverse=True)
    i = 0
    while diff != 0:
        key = sorted_keys[i % len(sorted_keys)]
        cap = len(strata[key])
        if diff > 0 and allocations[key] < cap:
            allocations[key] += 1
            diff -= 1
        elif diff < 0 and allocations[key] > 1:
            allocations[key] -= 1
            diff += 1
        i += 1
        if i > 10 * len(sorted_keys):
            break  # safety valve; shouldn't happen

    sampled = []
    for key, members in strata.items():
        n = min(allocations[key], len(members))
        sampled.extend(rng.sample(members, n))

    rng.shuffle(sampled)

    # Print stratum breakdown
    sampled_counts = Counter((q.get("type", "unknown"), q.get("level", "unknown"))
                              for q in sampled)
    print(f"    Stratified sample breakdown (type, level) → count:")
    for k in sorted(sampled_counts):
        print(f"      {k}: {sampled_counts[k]:,} "
              f"(of {len(strata[k]):,} in full set)")
    return sampled


def main():
    parser = argparse.ArgumentParser(
        description="Build a deduplicated paragraph corpus from HotpotQA"
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Path to hotpot_dev_distractor_v1.json (or train file)")
    parser.add_argument("--out-corpus", type=str, default="./hotpot_corpus.jsonl",
                        help="Output path for corpus JSONL")
    parser.add_argument("--out-questions", type=str, default="./hotpot_questions.jsonl",
                        help="Output path for questions JSONL")
    parser.add_argument("--id-format", type=str, default="sequential",
                        choices=["sequential", "hash"],
                        help="sequential = para_000001 etc.  hash = para_<md5(title)[:12]>")
    # ---- Sampling args ----
    parser.add_argument("--sample-size", type=int, default=None,
                        help="Number of questions to sample (default: all). "
                             "Sampling is stratified by (type, level). "
                             "The corpus output will contain only paragraphs "
                             "referenced by the sampled questions.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible sampling (default: 42). "
                             "Has no effect if --sample-size is not set.")
    args = parser.parse_args()

    # ---- Load ----
    print(f"[*] Loading {args.input}")
    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"    {len(data):,} questions loaded")

    # ---- Optional stratified sampling ----
    if args.sample_size is not None:
        print(f"[*] Stratified sampling {args.sample_size:,} questions "
              f"(seed={args.seed})...")
        data = stratified_sample(data, args.sample_size, args.seed)
        print(f"    {len(data):,} questions in sample")

    # ---- Deduplicate paragraphs across sampled questions only ----
    # paragraphs maps title -> {id, title, paragraph, full_context, n_sentences}
    paragraphs = {}
    print("[*] Deduplicating paragraphs by title...")
    for q in data:
        for ctx in q["context"]:
            title, sentences = ctx[0], ctx[1]
            if title in paragraphs:
                continue
            paragraph    = " ".join(s.strip() for s in sentences).strip()
            full_context = f"{title}\n\n{paragraph}"
            pid          = make_paragraph_id(title, len(paragraphs) + 1, args.id_format)
            paragraphs[title] = {
                "id":           pid,
                "title":        title,
                "paragraph":    paragraph,
                "full_context": full_context,
                "n_sentences":  len(sentences),
            }
    print(f"    {len(paragraphs):,} unique paragraphs after dedup")

    # ---- Build title -> id lookup ----
    title_to_id = {t: p["id"] for t, p in paragraphs.items()}

    # ---- Build questions manifest ----
    print("[*] Building questions manifest...")
    questions = []
    for q in data:
        # 10 distractor candidates per question
        candidate_ids = [title_to_id[ctx[0]] for ctx in q["context"] if ctx[0] in title_to_id]

        # Supporting facts: list of [title, sentence_idx]
        supporting       = []
        supporting_pids  = []   # ordered, deduped
        supporting_titles = []  # ordered, deduped
        for sf in q.get("supporting_facts", []):
            sf_title, sf_idx = sf[0], sf[1]
            if sf_title not in title_to_id:
                continue
            pid = title_to_id[sf_title]
            supporting.append({
                "paragraph_id": pid,
                "title":        sf_title,
                "sentence_idx": sf_idx,
            })
            if pid not in supporting_pids:
                supporting_pids.append(pid)
            if sf_title not in supporting_titles:
                supporting_titles.append(sf_title)

        questions.append({
            "question_id":              q.get("_id", f"q_{len(questions):06d}"),
            "question":                 q["question"],
            "answer":                   q["answer"],
            "type":                     q.get("type", "unknown"),
            "level":                    q.get("level", "unknown"),
            "candidate_paragraph_ids":  candidate_ids,
            "supporting_paragraph_ids": supporting_pids,
            "supporting_titles":        supporting_titles,
            "supporting_facts":         supporting,
        })

    # ---- Stats ----
    total_chars    = sum(len(p["full_context"]) for p in paragraphs.values())
    avg_para_chars = total_chars / len(paragraphs) if paragraphs else 0
    avg_candidates = (
        sum(len(q["candidate_paragraph_ids"]) for q in questions) / len(questions)
        if questions else 0
    )
    type_counts  = Counter(q["type"]  for q in questions)
    level_counts = Counter(q["level"] for q in questions)

    print()
    print("=" * 60)
    print(" CORPUS STATS")
    print("=" * 60)
    if args.sample_size is not None:
        print(f"  Sample size (requested)  : {args.sample_size:,}")
        print(f"  Sample size (actual)     : {len(questions):,}")
        print(f"  Random seed              : {args.seed}")
    print(f"  Total questions          : {len(questions):,}")
    print(f"  Unique paragraphs        : {len(paragraphs):,}")
    print(f"  Avg candidates/question  : {avg_candidates:.1f}")
    print(f"  Avg paragraph chars      : {avg_para_chars:.0f}")
    print(f"  Total corpus chars       : {total_chars:,}")
    print(f"  Approx tokens (chars/4)  : {total_chars // 4:,}")
    print(f"  By type                  : {dict(type_counts)}")
    print(f"  By level                 : {dict(level_counts)}")
    print("=" * 60)

    # ---- Write outputs ----
    Path(args.out_corpus).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_corpus, "w", encoding="utf-8") as f:
        for p in paragraphs.values():
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"\n[*] Wrote corpus    → {args.out_corpus}")

    Path(args.out_questions).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_questions, "w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    print(f"[*] Wrote questions → {args.out_questions}")


if __name__ == "__main__":
    main()