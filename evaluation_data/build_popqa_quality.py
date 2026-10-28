#!/usr/bin/env python3
"""Build PopQA and QuALITY in the evaluation_data layout, plus small seeded subsets for quick iteration.

    python evaluation_data/build_popqa_quality.py                         # full sets + subsets of 100 and 200
    python evaluation_data/build_popqa_quality.py --sizes 50 100 200 --seed 1

Full sets (same layout as hotpotqa/ and musique/):
  popqa/popqa.json, popqa/popqa_corpus.json
      Source: DIGIMON's PopQA copy (HyGRAG/Data/Popqa): the 1,399-question long-tail subset (as in Self-RAG), with
      33,595 retrieved Wikipedia passages pooled into one corpus. The corpus is stored in question order, one block
      of retrieved passages per question (about 25 each). Each question records its block [start, end) and whether
      the block was located reliably (its subject found in the first passage title/text, 5-60 passages).
  quality/quality.json, quality/quality_corpus.json
      Source: the official QuALITY v1.0.1 release (nyu-mll/quality, copied to quality/official/): the dev split,
      115 articles / 2,086 questions (1,065 HARD); the test split has no public labels. DIGIMON's copy pools
      train+dev (265 articles / 4,609 questions) and drops the article id and the HARD flag; its article texts
      match the official ones (528/530 rows identical).

Subsets in subsets/<ds>_<N>q_seed<seed>/ (for quick iteration only; scores are not comparable to full-set scores):
  popqa:   N questions sampled per relation type in proportion (questions are stored sorted by relation, so a
           plain slice would be one relation), only from reliably located blocks; corpus = those questions' blocks.
  quality: whole dev articles in seeded random order until >= N questions; every question of a chosen article is
           kept (a per-article graph is built anyway), so N is a minimum.
  Each subset is written in three formats:
    questions.json + corpus.json        evaluation_data layout ([{title, text}] corpus), for evaluation_scripts
    unh_questions.jsonl + unh_corpus.jsonl   experiments/utils/multihop_utils.py schema (UNH / nkg)
    digimon/Question.json + Corpus.json      DIGIMON / HyGRAG schema
  plus manifest.json (selection, counts, seed).
"""
import argparse
import collections
import json
import random
import re
import shutil
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DIGIMON = REPO / "HyGRAG" / "Data"
QUALITY_OFFICIAL_SRC = Path("/tmp/claude-1024/-home-taluka-unh-v1/93063a08-6c59-4bde-ad06-a12fe8b70006/scratchpad/quality")

POPQA_TEMPLATES = [  # PopQA paper, Appendix Table 2
    ("occupation", r"What is (.+?)'s occupation\?"), ("place of birth", r"In what city was (.+?) born\?"),
    ("genre", r"What genre is (.+?)\?"), ("father", r"Who is the father of (.+?) ?\?"),
    ("country", r"In what country is (.+?)\?"), ("producer", r"Who was the producer of (.+?) ?\?"),
    ("director", r"Who was the director of (.+?) ?\?"), ("capital of", r"What is (.+?) the capital of\?"),
    ("screenwriter", r"Who was the screenwriter for (.+?) ?\?"), ("composer", r"Who was the composer of (.+?) ?\?"),
    ("color", r"What color is (.+?) ?\?"), ("religion", r"What is the religion of (.+?) ?\?"),
    ("sport", r"What sport does (.+?) play\?"), ("author", r"Who is the author of (.+?) ?\?"),
    ("mother", r"Who is the mother of (.+?) ?\?"), ("capital", r"What is the capital of (.+?) ?\?"),
]


def jsonl(path):
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def norm_key(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return " " + re.sub(r"[^a-z0-9]+", " ", s.lower()).strip() + " "


# ---------------------------------------------------------------- PopQA
def build_popqa(out: Path):
    corpus = jsonl(DIGIMON / "Popqa" / "Corpus.json")
    raw_q = jsonl(DIGIMON / "Popqa" / "Question.json")
    titles = [norm_key(c["title"]) for c in corpus]
    texts = [norm_key(c["title"] + " " + c["context"]) for c in corpus]
    questions, starts, found, prev = [], [], [], -1
    for i, q in enumerate(raw_q):
        relation, subject = next(((r, m.group(1).strip()) for r, p in POPQA_TEMPLATES
                                  if (m := re.match(p, q["question"].strip()))), (None, None))
        key = norm_key(subject)
        window = range(prev + 1, min(prev + 120, len(corpus)))
        j = next((k for k in window if key in titles[k]), None)
        how = "title"
        if j is None:
            j, how = next((k for k in window if key in texts[k]), None), "text"
        if j is None:
            j, how = prev + 1, "not found"
        starts.append(j); found.append(how); prev = j
        questions.append({"id": f"popqa_{i:04d}", "question": q["question"], "answer": q["answers"][0],
                          "answer_aliases": q["answers"], "relation": relation, "subject": subject})
    ends = starts[1:] + [len(corpus)]
    for i, q in enumerate(questions):
        size = ends[i] - starts[i]
        nxt_ok = i + 1 == len(questions) or found[i + 1] != "not found"
        q.update(block_start=starts[i], block_end=ends[i],
                 block_reliable=found[i] != "not found" and nxt_ok and 5 <= size <= 60)
    passages = [{"title": c["title"], "text": c["context"].strip()} for c in corpus]
    out.mkdir(parents=True, exist_ok=True)
    json.dump(questions, open(out / "popqa.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump(passages, open(out / "popqa_corpus.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    reliable = sum(q["block_reliable"] for q in questions)
    print(f"popqa: {len(questions)} questions, {len(passages)} passages; {reliable} blocks located reliably")
    return questions, passages


def subset_popqa(questions, passages, n, seed):
    rng = random.Random(seed)
    pool = collections.defaultdict(list)
    for q in questions:
        if q["block_reliable"]:
            pool[q["relation"]].append(q)
    total = sum(len(v) for v in pool.values())
    # proportional allocation (largest remainder), at least 1 per relation when n allows
    quota = {r: n * len(v) / total for r, v in pool.items()}
    take = {r: int(x) for r, x in quota.items()}
    for r in sorted(quota, key=lambda r: quota[r] - take[r], reverse=True)[: n - sum(take.values())]:
        take[r] += 1
    chosen = []
    for r, v in sorted(pool.items()):
        chosen += rng.sample(v, min(take[r], len(v)))
    chosen.sort(key=lambda q: q["id"])
    corpus_idx = sorted({k for q in chosen for k in range(q["block_start"], q["block_end"])})
    return chosen, [passages[k] for k in corpus_idx], {"relations": {r: take[r] for r in sorted(take)}}


# ---------------------------------------------------------------- QuALITY
def build_quality(out: Path):
    off = out / "official"
    off.mkdir(parents=True, exist_ok=True)
    for split in ("train", "dev", "test"):
        name = f"QuALITY.v1.0.1.htmlstripped.{split}"
        if not (off / name).exists():
            if not (QUALITY_OFFICIAL_SRC / name).exists():
                raise SystemExit(f"missing {name}: download QuALITY.v1.0.1.zip from https://github.com/nyu-mll/quality "
                                 f"(data/v1.0.1) and unzip the htmlstripped files into {off}")
            shutil.copy2(QUALITY_OFFICIAL_SRC / name, off / name)
    rows = jsonl(off / "QuALITY.v1.0.1.htmlstripped.dev")
    articles, questions = {}, []
    for row in rows:  # one row per (article, question set); each article has 2 sets from different writers
        articles.setdefault(row["article_id"], {"title": row["title"], "text": row["article"].strip(),
                                                "article_id": row["article_id"], "source": row["source"]})
        for q in row["questions"]:
            gold = q["gold_label"]  # 1-based index into options
            letters = "ABCD"
            stem = q["question"].strip()
            questions.append({
                "id": q["question_unique_id"], "question": stem,
                "question_with_options": stem + "\n" + "\n".join(f"{letters[k]}: {o}" for k, o in enumerate(q["options"])),
                "options": q["options"], "answer": q["options"][gold - 1], "answer_letter": letters[gold - 1],
                "answer_aliases": [q["options"][gold - 1]], "difficult": bool(q["difficult"]),
                "article_id": row["article_id"], "set_unique_id": row["set_unique_id"], "title": row["title"],
                "source": row["source"]})
    corpus = list(articles.values())
    json.dump(questions, open(out / "quality.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump(corpus, open(out / "quality_corpus.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"quality: dev {len(corpus)} articles, {len(questions)} questions ({sum(q['difficult'] for q in questions)} HARD)")
    return questions, corpus


def subset_quality(questions, corpus, n, seed):
    rng = random.Random(seed)
    order = [a["article_id"] for a in corpus]
    rng.shuffle(order)
    by_article = collections.defaultdict(list)
    for q in questions:
        by_article[q["article_id"]].append(q)
    chosen_articles, chosen = [], []
    for aid in order:
        if len(chosen) >= n:
            break
        chosen_articles.append(aid); chosen += by_article[aid]
    keep = set(chosen_articles)
    return chosen, [a for a in corpus if a["article_id"] in keep], {"articles": chosen_articles}


# ---------------------------------------------------------------- writers
def write_subset(ds, qs, corpus, meta, out: Path, seed):
    out.mkdir(parents=True, exist_ok=True)
    (out / "digimon").mkdir(exist_ok=True)
    json.dump(qs, open(out / "questions.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump([{"title": c["title"], "text": c["text"]} for c in corpus],
              open(out / "corpus.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    with open(out / "unh_corpus.jsonl", "w", encoding="utf-8") as f:
        for k, c in enumerate(corpus):
            f.write(json.dumps({"id": f"{ds}_p{k:05d}", "title": c["title"], "paragraph": c["text"],
                                "full_context": f"{c['title']}\n{c['text']}"}, ensure_ascii=False) + "\n")
    with open(out / "unh_questions.jsonl", "w", encoding="utf-8") as f:
        for q in qs:
            rec = {"question_id": q["id"], "question": q.get("question_with_options", q["question"]),
                   "answer": q["answer"], "answer_aliases": q["answer_aliases"],
                   "type": q.get("relation") or ("hard" if q.get("difficult") else "easy"), "level": ds}
            if "answer_letter" in q:
                rec.update(answer_letter=q["answer_letter"], options=q["options"], article_id=q["article_id"])
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with open(out / "digimon" / "Corpus.json", "w", encoding="utf-8") as f:
        for k, c in enumerate(corpus):
            f.write(json.dumps({"title": c["title"], "context": c["text"], "id": k}, ensure_ascii=False) + "\n")
    with open(out / "digimon" / "Question.json", "w", encoding="utf-8") as f:
        for q in qs:
            rec = {"question": q.get("question_with_options", q["question"]), "answer": "|".join(q["answer_aliases"])}
            if "answer_letter" in q:
                rec.update(options=dict(zip("ABCD", q["options"])), answer_idx=q["answer_letter"])
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    manifest = {"dataset": ds, "seed": seed, "questions": len(qs), "corpus_docs": len(corpus), **meta}
    if ds == "quality":
        manifest["hard_questions"] = sum(q["difficult"] for q in qs)
    json.dump(manifest, open(out / "manifest.json", "w"), indent=1)
    print(f"  subset {out.name}: {len(qs)} questions, {len(corpus)} docs")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", type=int, nargs="+", default=[100, 200])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pq, pc = build_popqa(HERE / "popqa")
    qq, qc = build_quality(HERE / "quality")
    for n in args.sizes:
        qs, corpus, meta = subset_popqa(pq, pc, n, args.seed)
        write_subset("popqa", qs, corpus, meta, HERE / "subsets" / f"popqa_{n}q_seed{args.seed}", args.seed)
        qs, corpus, meta = subset_quality(qq, qc, n, args.seed)
        write_subset("quality", qs, corpus, meta, HERE / "subsets" / f"quality_{n}q_seed{args.seed}", args.seed)


if __name__ == "__main__":
    main()
