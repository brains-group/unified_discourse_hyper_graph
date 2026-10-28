"""Metric functions copied from the papers' code (logic unchanged, framework plumbing removed).

EM / F1 / Recall@k — HippoRAG 2 (src/hipporag/evaluation/{qa_eval,retrieval_eval}.py,
                     src/hipporag/utils/eval_utils.py); MemGraphRAG ships the same classes.
                     Gold = [answer] + answer_aliases, aggregated with max (HippoRAG main.py).
Contain-Acc        — LinearRAG src/evaluate.py::calculate_contain (the paper's "Str-Acc").
                     Uses LinearRAG's normalize_answer, which is the same SQuAD normalizer plus a
                     None guard. Gold = the dataset's primary `answer` field, as in LinearRAG.
"""
import re
import string
from collections import Counter
from typing import Dict, List


def normalize_answer(answer) -> str:
    if answer is None:
        return ""
    if not isinstance(answer, str):
        answer = str(answer)

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(answer.lower())))


# ---- HippoRAG 2 QAExactMatch / QAF1Score (per example, aggregation_fn = max) ----
def exact_match(gold_list: List[str], predicted: str) -> float:
    return max(1.0 if normalize_answer(g) == normalize_answer(predicted) else 0.0 for g in gold_list)


def _f1(gold: str, predicted: str) -> float:
    gold_tokens = normalize_answer(gold).split()
    predicted_tokens = normalize_answer(predicted).split()
    common = Counter(predicted_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(predicted_tokens)
    recall = 1.0 * num_same / len(gold_tokens)
    return 2 * (precision * recall) / (precision + recall)


def f1(gold_list: List[str], predicted: str) -> float:
    return max(_f1(g, predicted) for g in gold_list)


# ---- LinearRAG Evaluator.calculate_contain ----
def contain(pre_answers: str, gold_ans: str) -> int:
    if pre_answers is None or pre_answers == "" or (isinstance(pre_answers, str) and pre_answers.strip() == ""):
        return 0
    if gold_ans is None or gold_ans == "" or (isinstance(gold_ans, str) and gold_ans.strip() == ""):
        return 0
    s1 = normalize_answer(pre_answers)
    s2 = normalize_answer(gold_ans)
    return 1 if s2 in s1 else 0


# ---- HippoRAG 2 RetrievalRecall (per example) ----
def recall_at_k(gold_docs: List[str], retrieved_docs: List[str], k_list: List[int]) -> Dict[str, float]:
    out = {}
    for k in k_list:
        relevant = set(retrieved_docs[:k]) & set(gold_docs)
        out[f"Recall@{k}"] = len(relevant) / len(set(gold_docs)) if gold_docs else 0.0
    return out


# ---- HippoRAG 2 main.py gold helpers ----
def get_gold_answers(sample: dict) -> List[str]:
    answer = sample["answer"]
    answers = [answer] if isinstance(answer, str) else list(answer)
    answers.extend(sample.get("answer_aliases", []))
    return list(dict.fromkeys(answers))


def get_gold_docs(sample: dict, dataset_name: str) -> List[str]:
    if "supporting_facts" in sample:
        gold_titles = {item[0] for item in sample["supporting_facts"]}
        supporting = [item for item in sample["context"] if item[0] in gold_titles]
        separator = "" if dataset_name.startswith("hotpotqa") else " "
        docs = [item[0] + "\n" + separator.join(item[1]) for item in supporting]
    else:
        paragraphs = [p for p in sample["paragraphs"] if p.get("is_supporting", True)]
        docs = [p["title"] + "\n" + (p["text"] if "text" in p else p["paragraph_text"]) for p in paragraphs]
    return list(dict.fromkeys(docs))
