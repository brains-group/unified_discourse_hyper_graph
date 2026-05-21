"""
multihop_eval_utils.py
======================

Shared utilities for evaluating KG-based RAG systems (KGGen, GraphRAG,
HypergraphRAG, etc.) on HotpotQA and MuSiQue benchmark datasets.

Includes:
    - Exact match (EM) and F1 functions verbatim from the official
      HotpotQA evaluation script (hotpot_evaluate_v1.py).
    - Alias-aware variants for MuSiQue (max over gold + answer_aliases).
    - Checkpoint scan / save / load helpers for resumable construction.
    - Corpus / questions JSONL loaders matching the schema produced by
      prepare_hotpotqa_corpus.py and prepare_musique_corpus.py.
    - GlobalTokenAccumulator (LiteLLM hook) reused from insurance eval.
    - Answer-extraction prompt and a thread-safe LLM-call helper.

Author notes:
    - HotpotQA and MuSiQue both derive their normalization from SQuAD's
      reference implementation, so the inner normalize_answer is shared.
    - HotpotQA adds special "yes/no/noanswer" handling that MuSiQue lacks.
      MuSiQue, in turn, supports answer_aliases that HotpotQA lacks.
      Both behaviors are preserved here under separate function names.
"""

import os
import re
import json
import time
import string
import threading
from pathlib import Path
from collections import Counter
from typing import Optional, Tuple, List, Dict, Any, Any

import litellm
from litellm.integrations.custom_logger import CustomLogger


# =============================================================================
# DATASET CONSTANTS
# =============================================================================
HOTPOTQA = "hotpotqa"
MUSIQUE  = "musique"
SUPPORTED_DATASETS = (HOTPOTQA, MUSIQUE)


# =============================================================================
# OFFICIAL HOTPOTQA NORMALIZATION (verbatim from hotpot_evaluate_v1.py)
# https://github.com/hotpotqa/hotpot/blob/master/hotpot_evaluate_v1.py
# =============================================================================
def normalize_answer(s: str) -> str:
    """
    Standard SQuAD/HotpotQA/MuSiQue answer normalization:
    lowercase, strip punctuation, drop articles (a/an/the), collapse whitespace.
    """
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


# =============================================================================
# OFFICIAL HOTPOTQA F1 / EM
# =============================================================================
def hotpot_f1_score(prediction: str, ground_truth: str) -> Tuple[float, float, float]:
    """
    Returns (f1, precision, recall). Verbatim from hotpot_evaluate_v1.py.

    Special rule: if EITHER side normalizes to 'yes', 'no', or 'noanswer'
    and they don't match exactly, the score is zero. This avoids partial
    credit for answering 'no' to a 'yes' question via spurious token overlap.
    """
    normalized_prediction   = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0.0, 0.0, 0.0)

    if (normalized_prediction in ["yes", "no", "noanswer"]
            and normalized_prediction != normalized_ground_truth):
        return ZERO_METRIC
    if (normalized_ground_truth in ["yes", "no", "noanswer"]
            and normalized_prediction != normalized_ground_truth):
        return ZERO_METRIC

    prediction_tokens   = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common              = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same            = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall    = 1.0 * num_same / len(ground_truth_tokens)
    f1        = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def hotpot_exact_match(prediction: str, ground_truth: str) -> float:
    """Returns 1.0 if normalized strings match exactly, else 0.0."""
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


# =============================================================================
# MUSIQUE F1 / EM — same formulation as HotpotQA but takes max over aliases
# https://github.com/StonyBrookNLP/musique/blob/main/metrics/answer.py
# =============================================================================
def _musique_token_f1(prediction: str, ground_truth: str) -> Tuple[float, float, float]:
    """
    MuSiQue's reference F1 — same SQuAD-style tokenization, but without
    HotpotQA's yes/no zero-out rule. Returns (f1, precision, recall).
    """
    gold_toks = normalize_answer(ground_truth).split()
    pred_toks = normalize_answer(prediction).split()

    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either side is empty, exact match logic decides
        match = float(gold_toks == pred_toks)
        return match, match, match

    common   = Counter(gold_toks) & Counter(pred_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0, 0.0, 0.0
    precision = 1.0 * num_same / len(pred_toks)
    recall    = 1.0 * num_same / len(gold_toks)
    f1        = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def musique_exact_match(prediction: str, gold: str, aliases: Optional[List[str]] = None) -> float:
    """Max EM over (gold + aliases)."""
    candidates = [gold] + list(aliases or [])
    return max(hotpot_exact_match(prediction, g) for g in candidates)


def musique_f1_score(prediction: str, gold: str,
                     aliases: Optional[List[str]] = None) -> Tuple[float, float, float]:
    """Returns the F1/precision/recall triple from whichever (gold | alias) maximises F1."""
    candidates = [gold] + list(aliases or [])
    best = (0.0, 0.0, 0.0)
    for g in candidates:
        f1, p, r = _musique_token_f1(prediction, g)
        if f1 > best[0]:
            best = (f1, p, r)
    return best


# =============================================================================
# UNIFIED SCORING ENTRY POINTS
# Use these two functions in the eval loop regardless of dataset.
# =============================================================================
def score_em(prediction: str, gold: str,
             aliases: Optional[List[str]] = None, dataset: str = HOTPOTQA) -> float:
    if dataset == HOTPOTQA:
        return hotpot_exact_match(prediction, gold)
    elif dataset == MUSIQUE:
        return musique_exact_match(prediction, gold, aliases)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def score_f1(prediction: str, gold: str,
             aliases: Optional[List[str]] = None,
             dataset: str = HOTPOTQA) -> Tuple[float, float, float]:
    if dataset == HOTPOTQA:
        return hotpot_f1_score(prediction, gold)
    elif dataset == MUSIQUE:
        return musique_f1_score(prediction, gold, aliases)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


# =============================================================================
# CORPUS / QUESTIONS LOADERS
# Matches the JSONL schema produced by prepare_{hotpotqa,musique}_corpus.py
# =============================================================================
def load_corpus_jsonl(path: str) -> Dict[str, dict]:
    """
    Returns dict: paragraph_id -> {id, title, paragraph, full_context, n_sentences}
    """
    corpus = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            corpus[entry["id"]] = entry
    return corpus


def load_questions_jsonl(path: str) -> List[dict]:
    """Returns a list of question dicts (preserves file order)."""
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


# =============================================================================
# GRAPH RECONSTRUCTION
# KGGen's natural export format is JSON (via export_graph / graph.json).
# We reconstruct the Graph object from that JSON so we can pass it back into
# kg.aggregate() on resume. The attribute set mirrors what CustomKGGen.export_graph
# writes: entities, relations, edges, entity_clusters, edge_clusters.
# =============================================================================
def reconstruct_graph(graph_dict: dict):
    """
    Rebuilds a KGGen Graph object from the dict produced by export_graph().

    Tries the two most common kg_gen import paths. If neither is available
    (unusual), falls back to a types.SimpleNamespace that exposes the same
    attributes — kg.aggregate() accesses attributes directly so it still works.

    NOTE: if kg.aggregate() in your version of kg_gen does isinstance(g, Graph)
    type-checking rather than duck-typing, the SimpleNamespace fallback will
    fail. In that case pin down the correct import and replace the try/except.
    """
    try:
        from kg_gen.models import Graph          # newer kg_gen versions
    except ImportError:
        try:
            from kg_gen import Graph             # older kg_gen versions
        except ImportError:
            import types
            Graph = None                        # use SimpleNamespace below

    kwargs = dict(
        entities=set(graph_dict.get("entities") or []),
        relations=set(graph_dict.get("relations") or []),
        edges=[tuple(e) for e in (graph_dict.get("edges") or [])],
        entity_clusters={
            k: set(v) for k, v in (graph_dict.get("entity_clusters") or {}).items()
        } or None,
        edge_clusters={
            k: set(v) for k, v in (graph_dict.get("edge_clusters") or {}).items()
        } or None,
    )

    if Graph is not None:
        try:
            return Graph(**kwargs)
        except TypeError:
            pass   # Graph constructor differs; fall through to SimpleNamespace

    import types
    g = types.SimpleNamespace()
    for k, v in kwargs.items():
        setattr(g, k, v)
    return g


# =============================================================================
# CHECKPOINT MANAGEMENT
# Layout under run_dir:
#
#     checkpoint-1/              <- first N paragraphs absorbed and aggregated
#         metadata.json
#         absorbed_paragraph_ids.json
#         graph.json             <- aggregated KGGen graph exported as JSON
#
#     checkpoint-2/
#         metadata.json
#         absorbed_paragraph_ids.json
#         graph.json
#
#     final-checkpoint/          <- all paragraphs absorbed; evaluation-ready
#         metadata.json
#         absorbed_paragraph_ids.json
#         graph.json
#
# Every checkpoint's graph.json is a CUMULATIVE aggregated state (not just the
# delta for that block). On resume we load the latest graph.json, reconstruct
# the Graph object, and treat it as the base to merge new subgraphs into.
#
# Node embeddings are NOT persisted — they are cheap to recompute from graph.json
# at evaluation time (sentence-transformer forward pass, no LLM calls).
# =============================================================================
CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")
FINAL_CHECKPOINT_NAME = "final-checkpoint"


def find_all_checkpoints(run_dir: str) -> List[Tuple[int, str]]:
    """Sorted list of (checkpoint_num, full_path) for all checkpoint-N dirs."""
    if not os.path.isdir(run_dir):
        return []
    out = []
    for name in os.listdir(run_dir):
        m = CHECKPOINT_RE.match(name)
        if m and os.path.isdir(os.path.join(run_dir, name)):
            out.append((int(m.group(1)), os.path.join(run_dir, name)))
    out.sort(key=lambda x: x[0])
    return out


def find_latest_checkpoint(run_dir: str) -> Optional[str]:
    """Path of the highest-numbered checkpoint-N folder, or None."""
    cps = find_all_checkpoints(run_dir)
    return cps[-1][1] if cps else None


def find_final_checkpoint(run_dir: str) -> Optional[str]:
    """Path of final-checkpoint/ if it exists, else None."""
    p = os.path.join(run_dir, FINAL_CHECKPOINT_NAME)
    return p if os.path.isdir(p) else None


def next_checkpoint_num(run_dir: str) -> int:
    """Returns the next checkpoint number to use (1 if none exist yet)."""
    cps = find_all_checkpoints(run_dir)
    return (cps[-1][0] + 1) if cps else 1


def save_checkpoint(
    run_dir: str,
    checkpoint_num: int,
    metadata: dict,
    absorbed_paragraph_ids: List[str],
    aggregated_graph,           # KGGen Graph object — exported via export_graph
    kg_instance,                # CustomKGGen instance (owns export_graph)
    final: bool = False,
) -> str:
    """
    Persists a checkpoint to run_dir/checkpoint-N (or run_dir/final-checkpoint).

    Writes three files:
        metadata.json               — cumulative construction stats
        absorbed_paragraph_ids.json — which paragraph IDs have been processed
        graph.json                  — aggregated KGGen graph in export_graph format

    On resume, load_checkpoint() reads all three and uses reconstruct_graph()
    to turn graph.json back into a KGGen Graph object.

    Args:
        run_dir:               parent run directory.
        checkpoint_num:        N in "checkpoint-N"; ignored when final=True.
        metadata:              dict to dump as metadata.json.
        absorbed_paragraph_ids: cumulative list of processed paragraph IDs.
        aggregated_graph:      the KGGen Graph aggregated so far (or final).
        kg_instance:           CustomKGGen instance (provides export_graph).
        final:                 if True, writes to final-checkpoint/.

    Returns:
        The checkpoint folder path that was created.
    """
    folder = FINAL_CHECKPOINT_NAME if final else f"checkpoint-{checkpoint_num}"
    cp_dir = os.path.join(run_dir, folder)
    os.makedirs(cp_dir, exist_ok=True)

    with open(os.path.join(cp_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)

    with open(os.path.join(cp_dir, "absorbed_paragraph_ids.json"), "w", encoding="utf-8") as f:
        json.dump(absorbed_paragraph_ids, f, indent=2)

    graph_json_path = os.path.join(cp_dir, "graph.json")
    kg_instance.export_graph(aggregated_graph, graph_json_path)

    return cp_dir


def load_checkpoint(cp_dir: str) -> Tuple[dict, List[str], Any]:
    """
    Loads a checkpoint from cp_dir.

    Returns:
        (metadata, absorbed_paragraph_ids, graph)
        where graph is a reconstructed KGGen Graph object ready for use as
        the base in the next kg.aggregate() call.
    """
    with open(os.path.join(cp_dir, "metadata.json"), "r", encoding="utf-8") as f:
        metadata = json.load(f)
    with open(os.path.join(cp_dir, "absorbed_paragraph_ids.json"), "r", encoding="utf-8") as f:
        absorbed = json.load(f)
    with open(os.path.join(cp_dir, "graph.json"), "r", encoding="utf-8") as f:
        graph_dict = json.load(f)
    graph = reconstruct_graph(graph_dict)
    return metadata, absorbed, graph


# =============================================================================
# GLOBAL TOKEN ACCUMULATOR — same one used in insurance eval
# =============================================================================
class GlobalTokenAccumulator(CustomLogger):
    """Thread-safe LiteLLM hook that sums token usage across all API calls."""

    def __init__(self):
        super().__init__()
        self.prompt_tokens     = 0
        self.completion_tokens = 0
        self.total_tokens      = 0
        self.api_calls         = 0
        self._lock             = threading.Lock()

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        usage = (
            response_obj.get("usage")
            if isinstance(response_obj, dict)
            else getattr(response_obj, "usage", None)
        )
        if usage is None:
            return
        if isinstance(usage, dict):
            p = usage.get("prompt_tokens", 0)
            c = usage.get("completion_tokens", 0)
            t = usage.get("total_tokens", p + c)
        else:
            p = getattr(usage, "prompt_tokens", 0)
            c = getattr(usage, "completion_tokens", 0)
            t = getattr(usage, "total_tokens", p + c)
        with self._lock:
            self.prompt_tokens     += p
            self.completion_tokens += c
            self.total_tokens      += t
            self.api_calls         += 1

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "prompt_tokens":     self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens":      self.total_tokens,
                "api_calls":         self.api_calls,
            }

    def reset(self):
        with self._lock:
            self.prompt_tokens     = 0
            self.completion_tokens = 0
            self.total_tokens      = 0
            self.api_calls         = 0


def attach_token_tracker(tracker: GlobalTokenAccumulator):
    """Hooks the tracker into LiteLLM's success callbacks."""
    litellm.callbacks        = [tracker]
    litellm.success_callback = [tracker]


def detach_token_tracker():
    litellm.callbacks        = []
    litellm.success_callback = []


def add_token_counters(a: dict, b: dict) -> dict:
    """Element-wise add two token snapshots (prompt/completion/total/api_calls)."""
    keys = ("prompt_tokens", "completion_tokens", "total_tokens", "api_calls")
    return {k: int(a.get(k, 0)) + int(b.get(k, 0)) for k in keys}


# =============================================================================
# ANSWER PROMPT & LLM CALL
# Short-answer extraction prompt that matches what HotpotQA/MuSiQue expect:
# concise span-like answers (entity / date / yes-no), no chain-of-thought.
# =============================================================================
SYSTEM_PROMPT = (
    "You answer multi-hop questions using only the provided context. "
    "Output the shortest possible answer phrase — a single entity, date, "
    "number, or yes/no. Do not include any explanation, restate the question, "
    "or add extra words."
)

USER_PROMPT_TEMPLATE = """Context:
{context}

Question: {question}

Answer:"""


def make_answer_messages(question: str, context: str) -> list:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",
         "content": USER_PROMPT_TEMPLATE.format(context=context, question=question)},
    ]


def call_llm_for_answer(
    model: str,
    api_base: str,
    api_key: str,
    question: str,
    context: str,
    max_tokens: int = 128,
    temperature: float = 0.0,
) -> str:
    """
    Synchronous LiteLLM call. Thread-safe (LiteLLM internal locks).
    Returns the raw LLM string with surrounding whitespace stripped.
    """
    resp = litellm.completion(
        model=f"openai/{model}",
        messages=make_answer_messages(question, context),
        api_base=api_base,
        api_key=api_key,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return resp.choices[0].message.content.strip()


# =============================================================================
# MISC HELPERS
# =============================================================================
def safe_div(num, denom, ndigits=4):
    return round(num / denom, ndigits) if denom else 0.0


def truncate_tokens(text: str, tokenizer, max_tokens: int) -> Tuple[str, int]:
    """
    Truncate `text` so the encoded length <= max_tokens. Returns (text, n_tokens).
    Falls back gracefully if the tokenizer fails for any reason.
    """
    try:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) <= max_tokens:
            return text, len(ids)
        kept = ids[:max_tokens]
        return tokenizer.decode(kept, skip_special_tokens=True), len(kept)
    except Exception:
        # Word-count fallback (~1 word ≈ 1.3 tokens, conservative)
        words = text.split()
        cap   = int(max_tokens * 0.75)
        if len(words) <= cap:
            return text, len(words)
        return " ".join(words[:cap]), cap