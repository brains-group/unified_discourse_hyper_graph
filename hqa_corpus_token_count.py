#!/usr/bin/env python3
"""
count_corpus_tokens.py
======================

Counts tokens in a JSONL HotpotQA corpus produced by prepare_hotpotqa_corpus.py

Usage:
    python count_corpus_tokens.py \
        --input ./data/hotpot_corpus.jsonl

Optional:
    --field paragraph
    --encoding cl100k_base

Default field is:
    full_context

Requires:
    pip install tiktoken
"""

import json
import argparse
from pathlib import Path

import tiktoken


def main():
    parser = argparse.ArgumentParser(
        description="Count tokens in a JSONL corpus using tiktoken"
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to corpus JSONL file"
    )

    parser.add_argument(
        "--field",
        type=str,
        default="full_context",
        choices=["paragraph", "full_context"],
        help="Which text field to tokenize"
    )

    parser.add_argument(
        "--encoding",
        type=str,
        default="cl100k_base",
        help="tiktoken encoding name"
    )

    args = parser.parse_args()

    corpus_path = Path(args.input)

    if not corpus_path.exists():
        raise FileNotFoundError(f"File not found: {corpus_path}")

    print(f"[*] Loading tokenizer: {args.encoding}")
    enc = tiktoken.get_encoding(args.encoding)

    total_tokens = 0
    total_paragraphs = 0

    min_tokens = float("inf")
    max_tokens = 0

    print(f"[*] Reading corpus: {corpus_path}")

    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)

            text = row.get(args.field, "")
            n_tokens = len(enc.encode(text))

            total_tokens += n_tokens
            total_paragraphs += 1

            min_tokens = min(min_tokens, n_tokens)
            max_tokens = max(max_tokens, n_tokens)

    avg_tokens = (
        total_tokens / total_paragraphs
        if total_paragraphs > 0 else 0
    )

    print()
    print("=" * 60)
    print(" TOKEN STATS")
    print("=" * 60)
    print(f"  Field tokenized        : {args.field}")
    print(f"  Encoding               : {args.encoding}")
    print(f"  Total paragraphs       : {total_paragraphs:,}")
    print(f"  Total corpus tokens    : {total_tokens:,}")
    print(f"  Average tokens/para    : {avg_tokens:.2f}")
    print(f"  Min tokens/para        : {min_tokens:,}")
    print(f"  Max tokens/para        : {max_tokens:,}")
    print("=" * 60)


if __name__ == "__main__":
    main()