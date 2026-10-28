"""
Resumable on-disk checkpoint state for build_index().

ConstructionCheckpoint owns everything under <checkpoint_dir>/ needed to resume a crashed
construction run without redoing already-finished work:

    chunk_manifest.json         num_chunks + a hash per source chunk (detects corpus/chunking drift)
    chunks/chunk_NNNNN.graphml  one local subgraph per source chunk, saved as it finishes
    extraction_progress.json    completed chunk indices + cumulative tokens/time
    merged.graphml              graph after ALL chunks are merged + deduplicated (pre global-edges)
    merged_state.json           dedup tokens/time
    rounds/round_N_clusters.json   cluster assignments for round N (computed once, cached)
    rounds/round_N_edges.jsonl      one line per finished cluster's edges (append-only journal)
    rounds/round_N_state.json       status + completed cluster indices + cumulative tokens/time
    final.graphml + final_state.json   marks "global edges complete"

Every method is a safe no-op when constructed with checkpoint_dir=None, so callers can invoke
this object unconditionally instead of branching on whether checkpointing is enabled.
"""
import os
import json
import shutil
import hashlib
from typing import List, Optional

from nkg.models.Graph import Graph

_ZERO_TOKENS = {"prompt": 0, "completion": 0, "total": 0, "calls": 0, "cache_hit": 0}


def _add_tokens(a: dict, b: dict) -> dict:
    return {
        "prompt":     a.get("prompt", 0) + b.get("prompt", 0),
        "completion": a.get("completion", 0) + b.get("completion", 0),
        "total":      a.get("total", 0) + b.get("total", 0),
        "calls":      a.get("calls", 0) + b.get("calls", 0),
        "cache_hit":  a.get("cache_hit", 0) + b.get("cache_hit", 0),
    }


def _read_json(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path, data):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    os.replace(tmp, path)


def _chunk_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


class ConstructionCheckpoint:

    def __init__(self, checkpoint_dir: Optional[str]):
        self.enabled = checkpoint_dir is not None
        self.dir = checkpoint_dir
        if self.enabled:
            os.makedirs(self._chunks_dir(), exist_ok=True)
            os.makedirs(self._rounds_dir(), exist_ok=True)

    # ---- paths ----
    def _chunks_dir(self):               return os.path.join(self.dir, "chunks")
    def _manifest_path(self):            return os.path.join(self.dir, "chunk_manifest.json")
    def _extraction_progress_path(self): return os.path.join(self.dir, "extraction_progress.json")
    def _chunk_graph_path(self, idx):    return os.path.join(self._chunks_dir(), f"chunk_{idx:05d}.graphml")
    def _merged_graph_path(self):        return os.path.join(self.dir, "merged.graphml")
    def _merged_state_path(self):        return os.path.join(self.dir, "merged_state.json")
    def _rounds_dir(self):               return os.path.join(self.dir, "rounds")
    def _round_clusters_path(self, n):   return os.path.join(self._rounds_dir(), f"round_{n}_clusters.json")
    def _round_edges_path(self, n):      return os.path.join(self._rounds_dir(), f"round_{n}_edges.jsonl")
    def _round_state_path(self, n):      return os.path.join(self._rounds_dir(), f"round_{n}_state.json")
    def _final_graph_path(self):         return os.path.join(self.dir, "final.graphml")
    def _final_state_path(self):         return os.path.join(self.dir, "final_state.json")

    # ================= wipe =================
    def wipe(self):
        if not self.enabled:
            return
        if os.path.isdir(self.dir):
            shutil.rmtree(self.dir, ignore_errors=True)
        os.makedirs(self._chunks_dir(), exist_ok=True)
        os.makedirs(self._rounds_dir(), exist_ok=True)

    # ================= Phase A: chunk extraction =================
    def verify_or_init_chunk_manifest(self, text_chunks: List[str]):
        """Validates the resolved chunk list against the saved manifest, if any. Creates the
        manifest on a fresh run. Raises if an existing manifest disagrees with the freshly
        computed chunks (corpus or chunking params changed without going through the
        eval script's wipe-on-config-change flow)."""
        if not self.enabled:
            return
        hashes = [_chunk_hash(c) for c in text_chunks]
        existing = _read_json(self._manifest_path())
        if existing is None:
            _write_json(self._manifest_path(), {"num_chunks": len(text_chunks), "chunk_hashes": hashes})
            return
        if existing.get("num_chunks") != len(text_chunks) or existing.get("chunk_hashes") != hashes:
            raise RuntimeError(
                f"Checkpoint chunk manifest at {self._manifest_path()} does not match the "
                f"freshly-computed chunk list (expected {existing.get('num_chunks')} chunks, "
                f"got {len(text_chunks)}, or chunk content differs). The corpus file or chunking "
                f"parameters changed since this checkpoint was created. Wipe the checkpoint "
                f"directory to start over."
            )

    def chunk_done_indices(self) -> set:
        if not self.enabled:
            return set()
        prog = _read_json(self._extraction_progress_path(), default={})
        return set(prog.get("completed_indices", []))

    def load_chunk_graph(self, idx: int) -> Graph:
        g = Graph()
        g.load_graph(self._chunk_graph_path(idx))
        return g

    def save_chunk_graph(self, idx: int, graph: Graph):
        if not self.enabled:
            return
        graph.export_graph(self._chunk_graph_path(idx))
        prog = _read_json(self._extraction_progress_path(), default={
            "completed_indices": [], "cumulative_tokens": dict(_ZERO_TOKENS), "cumulative_time_sec": 0.0,
        })
        done = set(prog.get("completed_indices", []))
        done.add(idx)
        prog["completed_indices"] = sorted(done)
        _write_json(self._extraction_progress_path(), prog)

    def add_extraction_tokens_and_time(self, tokens: dict, time_sec: float):
        if not self.enabled:
            return
        prog = _read_json(self._extraction_progress_path(), default={
            "completed_indices": [], "cumulative_tokens": dict(_ZERO_TOKENS), "cumulative_time_sec": 0.0,
        })
        prog["cumulative_tokens"] = _add_tokens(prog.get("cumulative_tokens", _ZERO_TOKENS), tokens)
        prog["cumulative_time_sec"] = round(prog.get("cumulative_time_sec", 0.0) + time_sec, 3)
        _write_json(self._extraction_progress_path(), prog)

    def extraction_tokens_total(self) -> dict:
        if not self.enabled:
            return dict(_ZERO_TOKENS)
        return _read_json(self._extraction_progress_path(), default={}).get("cumulative_tokens", dict(_ZERO_TOKENS))

    def extraction_time_total(self) -> float:
        if not self.enabled:
            return 0.0
        return float(_read_json(self._extraction_progress_path(), default={}).get("cumulative_time_sec", 0.0))

    # ================= Phase B: merge + dedup =================
    def has_merged(self) -> bool:
        return self.enabled and os.path.exists(self._merged_graph_path())

    def load_merged(self):
        g = Graph()
        g.load_graph(self._merged_graph_path())
        state = _read_json(self._merged_state_path(), default={})
        return g, state

    def save_merged(self, graph: Graph, dedup_tokens: dict, dedup_time_sec: float,
                    fact_dedup_tokens: dict = None, fact_dedup_time_sec: float = 0.0):
        if not self.enabled:
            return
        graph.export_graph(self._merged_graph_path())
        _write_json(self._merged_state_path(), {
            "dedup_tokens": dedup_tokens,
            "dedup_time_sec": round(dedup_time_sec, 3),
            "fact_dedup_tokens": fact_dedup_tokens or dict(_ZERO_TOKENS),
            "fact_dedup_time_sec": round(fact_dedup_time_sec, 3),
        })

    def dedup_time_total(self) -> float:
        if not self.enabled:
            return 0.0
        return float(_read_json(self._merged_state_path(), default={}).get("dedup_time_sec", 0.0))

    def fact_dedup_time_total(self) -> float:
        if not self.enabled:
            return 0.0
        return float(_read_json(self._merged_state_path(), default={}).get("fact_dedup_time_sec", 0.0))

    # ================= Phase C: global-edge rounds =================
    def _round_state(self, n: int) -> dict:
        return _read_json(self._round_state_path(n), default={
            "status": "pending", "completed_cluster_indices": [],
            "cumulative_tokens": dict(_ZERO_TOKENS), "cumulative_time_sec": 0.0,
        })

    def round_status(self, n: int) -> str:
        if not self.enabled:
            return "pending"
        return self._round_state(n).get("status", "pending")

    def has_cluster_list(self, n: int) -> bool:
        return self.enabled and os.path.exists(self._round_clusters_path(n))

    def save_cluster_list(self, n: int, cluster_lists: List[list]):
        if not self.enabled:
            return
        _write_json(self._round_clusters_path(n), cluster_lists)
        state = self._round_state(n)
        state["status"] = "in_progress"
        _write_json(self._round_state_path(n), state)

    def load_cluster_list(self, n: int) -> List[list]:
        return _read_json(self._round_clusters_path(n), default=[])

    def completed_cluster_indices(self, n: int) -> set:
        if not self.enabled:
            return set()
        return set(self._round_state(n).get("completed_cluster_indices", []))

    def append_cluster_edges(self, n: int, cluster_idx: int, edges: List[dict]):
        """Journals a finished cluster's edges and marks it complete. Call from the main thread
        right after the cluster's edges are written into the in-memory graph."""
        if not self.enabled:
            return
        with open(self._round_edges_path(n), "a", encoding="utf-8") as f:
            f.write(json.dumps({"cluster_idx": cluster_idx, "edges": edges}, default=str) + "\n")
        state = self._round_state(n)
        done = set(state.get("completed_cluster_indices", []))
        done.add(cluster_idx)
        state["completed_cluster_indices"] = sorted(done)
        state["status"] = "in_progress"
        _write_json(self._round_state_path(n), state)

    def add_round_tokens_and_time(self, n: int, tokens: dict, time_sec: float):
        if not self.enabled:
            return
        state = self._round_state(n)
        state["cumulative_tokens"] = _add_tokens(state.get("cumulative_tokens", _ZERO_TOKENS), tokens)
        state["cumulative_time_sec"] = round(state.get("cumulative_time_sec", 0.0) + time_sec, 3)
        _write_json(self._round_state_path(n), state)

    def round_tokens(self, n: int) -> dict:
        if not self.enabled:
            return dict(_ZERO_TOKENS)
        return self._round_state(n).get("cumulative_tokens", dict(_ZERO_TOKENS))

    def round_time(self, n: int) -> float:
        if not self.enabled:
            return 0.0
        return float(self._round_state(n).get("cumulative_time_sec", 0.0))

    def replay_round_edges(self, n: int, graph: Graph):
        """Re-adds every journaled edge for round n onto `graph`. Idempotent (add_edge overwrites)."""
        if not self.enabled:
            return
        path = self._round_edges_path(n)
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                for edge in entry.get("edges", []):
                    src, tgt = edge["source"], edge["target"]
                    attrs = {k: v for k, v in edge.items() if k not in ("source", "target")}
                    graph.network.add_edge(src, tgt, **attrs)

    def mark_round_done(self, n: int):
        if not self.enabled:
            return
        state = self._round_state(n)
        state["status"] = "done"
        _write_json(self._round_state_path(n), state)

    # ================= Phase D: final =================
    def has_final(self) -> bool:
        return self.enabled and os.path.exists(self._final_graph_path())

    def load_final(self):
        g = Graph()
        g.load_graph(self._final_graph_path())
        state = _read_json(self._final_state_path(), default={})
        return g, state.get("token_breakdown", {})

    def save_final(self, graph: Graph, token_breakdown: dict):
        if not self.enabled:
            return
        graph.export_graph(self._final_graph_path())
        _write_json(self._final_state_path(), {"token_breakdown": token_breakdown})
