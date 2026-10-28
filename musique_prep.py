#!/usr/bin/env python3
"""
prepare_musique_corpus.py
=========================

Reads a MuSiQue JSONL file and produces two JSONL artifacts in the same
schema as prepare_hotpotqa_corpus.py so that the downstream KG construction
and evaluation scripts are identical for both datasets.

    corpus.jsonl    — one row per UNIQUE paragraph (deduped by title)
                      fields: id, title, paragraph, full_context, n_sentences

    questions.jsonl — one row per question
                      fields: question_id, question, answer, type, level,
                              candidate_paragraph_ids,
                              supporting_paragraph_ids,
                              supporting_titles,
                              supporting_facts,
                              answer_aliases,       (MuSiQue-specific)
                              answerable            (MuSiQue-specific)

Official download (CC BY 4.0):
    git clone https://github.com/StonyBrookNLP/musique
    cd musique && bash download_data.sh
    # Files land in data/
    # Use: data/musique_ans_v1.0_dev.jsonl   (answerable dev, ~2417 examples)
    #      data/musique_ans_v1.0_train.jsonl  (answerable train)
    #      data/musique_full_v1.0_dev.jsonl   (answerable + unanswerable dev)

Usage:
    # Full answerable dev set (~2417 questions)
    python prepare_musique_corpus.py \\
        --input ./data/musique_ans_v1.0_dev.jsonl \\
        --out-corpus    ./data/musique_corpus.jsonl \\
        --out-questions ./data/musique_questions.jsonl

    # Reproducible 1000-question stratified sample
    python prepare_musique_corpus.py \\
        --input ./data/musique_ans_v1.0_dev.jsonl \\
        --sample-size 1000 \\
        --seed 42 \\
        --out-corpus    ./data/musique_corpus_1k.jsonl \\
        --out-questions ./data/musique_questions_1k.jsonl

    # Include unanswerable questions (non-standard)
    python prepare_musique_corpus.py \\
        --input ./data/musique_full_v1.0_dev.jsonl \\
        --include-unanswerable \\
        --out-corpus    ./data/musique_full_corpus.jsonl \\
        --out-questions ./data/musique_full_questions.jsonl

Sampling notes:
    - Stratification is by hop type (2hop / 3hop / 4hop), derived from the
      question ID prefix. This preserves the dataset's hop-count distribution.
    - The corpus output contains ONLY the paragraphs referenced by the sampled
      questions — a valid closed-world subset for KG construction and retrieval.
    - With --seed and --sample-size fixed, output is fully deterministic.

Key structural differences from HotpotQA handled here:
    - Input is already JSONL (one JSON object per line), not a single JSON array.
    - Paragraphs arrive as full text in 'paragraph_text', not a sentence list.
      n_sentences is estimated by splitting on sentence-ending punctuation.
    - Supporting paragraphs are flagged with is_supporting=True; no sentence index.
      supporting_facts entries use sentence_idx: null to keep schema consistent.
    - answer_aliases list may be non-empty — preserved in questions.jsonl.
    - answerable boolean is preserved — filter to True for standard eval.
    - Question type (2hop / 3hop / 4hop) is derived from the id prefix.
"""

import os
import re
import json
import random
import hashlib
import argparse
from pathlib import Path
from collections import Counter, defaultdict


def make_paragraph_id(title: str, idx: int, fmt: str) -> str:
    if fmt == "hash":
        h = hashlib.md5(title.encode("utf-8")).hexdigest()[:12]
        return f"para_{h}"
    elif fmt == "sequential":
        return f"para_{idx:06d}"
    else:
        raise ValueError(f"Unknown id-format: {fmt}")


def estimate_n_sentences(text: str) -> int:
    """
    Rough sentence count by splitting on sentence-ending punctuation.
    MuSiQue gives full paragraph text rather than a pre-split sentence list,
    so this is an estimate only — used for consistency with the HotpotQA schema.
    """
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return max(1, len([p for p in parts if p]))


def derive_hop_type(question_id: str) -> str:
    """
    MuSiQue ids are prefixed with '2hop__', '3hop__', '4hop__'.
    Returns '2hop', '3hop', '4hop', or 'unknown'.
    """
    for prefix in ("2hop", "3hop", "4hop"):
        if question_id.startswith(prefix):
            return prefix
    return "unknown"


def load_jsonl(path: str):
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def stratified_sample(data: list, sample_size: int, seed: int) -> list:
    """
    Stratified sample by hop type (2hop / 3hop / 4hop).
    Allocates proportionally, distributing any remainder to the largest strata.
    Returns a shuffled list of length min(sample_size, len(data)).
    """
    if sample_size >= len(data):
        print(f"    sample-size ({sample_size:,}) >= dataset size ({len(data):,}); "
              f"using full dataset")
        return data

    # Group by hop type
    strata = defaultdict(list)
    for q in data:
        key = derive_hop_type(q.get("id", ""))
        strata[key].append(q)

    total = len(data)
    allocations = {}
    for key, members in strata.items():
        allocations[key] = max(1, round(len(members) / total * sample_size))

    # Adjust to exactly hit sample_size
    rng = random.Random(seed)
    allocated = sum(allocations.values())
    diff = sample_size - allocated
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
            break  # safety valve

    sampled = []
    for key, members in strata.items():
        n = min(allocations[key], len(members))
        sampled.extend(rng.sample(members, n))

    rng.shuffle(sampled)

    # Print stratum breakdown
    sampled_counts = Counter(derive_hop_type(q.get("id", "")) for q in sampled)
    print(f"    Stratified sample breakdown (hop type) → count:")
    for k in sorted(sampled_counts):
        print(f"      {k}: {sampled_counts[k]:,} "
              f"(of {len(strata[k]):,} in full set)")
    return sampled


def main():
    parser = argparse.ArgumentParser(
        description="Build a deduplicated paragraph corpus from MuSiQue"
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Path to musique_ans_v1.0_dev.jsonl or similar")
    parser.add_argument("--out-corpus", type=str, default="./musique_corpus.jsonl")
    parser.add_argument("--out-questions", type=str, default="./musique_questions.jsonl")
    parser.add_argument("--id-format", type=str, default="sequential",
                        choices=["sequential", "hash"])
    parser.add_argument("--include-unanswerable", action="store_true",
                        help="Include unanswerable questions (non-standard). "
                             "Default: only answerable=True questions are kept.")
    # ---- Sampling args ----
    parser.add_argument("--sample-size", type=int, default=None,
                        help="Number of questions to sample (default: all). "
                             "Sampling is stratified by hop type (2hop/3hop/4hop). "
                             "The corpus output will contain only paragraphs "
                             "referenced by the sampled questions.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible sampling (default: 42). "
                             "Has no effect if --sample-size is not set.")
    args = parser.parse_args()

    # ---- Load ----
    print(f"[*] Loading {args.input}")
    raw = load_jsonl(args.input)
    print(f"    {len(raw):,} total entries loaded")

    # Filter answerability
    if not args.include_unanswerable:
        data = [e for e in raw if e.get("answerable", True)]
        print(f"    {len(data):,} answerable entries kept "
              f"({len(raw) - len(data):,} unanswerable dropped)")
    else:
        data = raw
        print(f"    Keeping all entries including unanswerable")

    # ---- Optional stratified sampling ----
    if args.sample_size is not None:
        print(f"[*] Stratified sampling {args.sample_size:,} questions "
              f"(seed={args.seed})...")
        data = stratified_sample(data, args.sample_size, args.seed)
        print(f"    {len(data):,} questions in sample")

    # ---- Deduplicate paragraphs across sampled questions only ----
    paragraphs = {}   # title -> corpus entry dict
    print("[*] Deduplicating paragraphs by title...")
    for entry in data:
        for para in entry.get("paragraphs", []):
            title = para["title"]
            if title in paragraphs:
                continue
            text         = para["paragraph_text"].strip()
            full_context = f"{title}\n\n{text}"
            pid          = make_paragraph_id(title, len(paragraphs) + 1, args.id_format)
            paragraphs[title] = {
                "id":           pid,
                "title":        title,
                "paragraph":    text,
                "full_context": full_context,
                "n_sentences":  estimate_n_sentences(text),
            }
    print(f"    {len(paragraphs):,} unique paragraphs after dedup")

    title_to_id = {t: p["id"] for t, p in paragraphs.items()}

    # ---- Build questions manifest ----
    print("[*] Building questions manifest...")
    questions = []
    for entry in data:
        paras = entry.get("paragraphs", [])

        candidate_ids = [
            title_to_id[p["title"]]
            for p in paras
            if p["title"] in title_to_id
        ]

        supporting_pids   = []
        supporting_titles = []
        supporting_facts  = []
        for para in paras:
            if not para.get("is_supporting", False):
                continue
            title = para["title"]
            if title not in title_to_id:
                continue
            pid = title_to_id[title]
            if pid not in supporting_pids:
                supporting_pids.append(pid)
            if title not in supporting_titles:
                supporting_titles.append(title)
            # MuSiQue has no sentence-level index — use null to match HotpotQA schema
            supporting_facts.append({
                "paragraph_id": pid,
                "title":        title,
                "sentence_idx": None,
            })

        hop_type = derive_hop_type(entry.get("id", ""))

        questions.append({
            # ---- Fields shared with HotpotQA schema ----
            "question_id":              entry.get("id", f"q_{len(questions):06d}"),
            "question":                 entry["question"],
            "answer":                   entry["answer"],
            "type":                     hop_type,
            "level":                    None,
            "candidate_paragraph_ids":  candidate_ids,
            "supporting_paragraph_ids": supporting_pids,
            "supporting_titles":        supporting_titles,
            "supporting_facts":         supporting_facts,
            # ---- MuSiQue-specific fields ----
            "answer_aliases":           entry.get("answer_aliases", []),
            "answerable":               entry.get("answerable", True),
        })

    # ---- Stats ----
    total_chars    = sum(len(p["full_context"]) for p in paragraphs.values())
    avg_para_chars = total_chars / len(paragraphs) if paragraphs else 0
    avg_candidates = (
        sum(len(q["candidate_paragraph_ids"]) for q in questions) / len(questions)
        if questions else 0
    )
    type_counts = Counter(q["type"] for q in questions)
    alias_count = sum(1 for q in questions if q["answer_aliases"])

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
    print(f"  Questions with aliases   : {alias_count:,}")
    print(f"  By hop type              : {dict(type_counts)}")
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