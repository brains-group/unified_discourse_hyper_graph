"""
Structured, opt-in tracing for GoalSearchEngine.retrieve().

Every stage (goal planning, seeding, beam traversal, island-and-bridge, final
ranking) can optionally record exactly what it saw and chose -- every
candidate edge considered from every node, its score against the active
goal(s), and whether it was kept or dropped -- so a specific query's behavior
can be inspected after the fact instead of guessed at. Tracing is purely
observational: nothing here changes scoring or selection.

Disabled by default and passed as a Null Object (TraceRecorder(enabled=False))
rather than None, so every call site can unconditionally call tracer.log(...)
without an `if trace:` check at every call site. When disabled, .log() is a
no-op and costs one attribute check.
"""
from dataclasses import dataclass, field
from typing import Any, List

import dspy

from nkg.models.Graph import Graph


def node_label(graph: Graph, node_id: str) -> str:
    """Short human-readable text for a node, for trace/report output."""
    if node_id in graph.facts:
        return graph.facts[node_id].sentence
    if node_id in graph.entities:
        ent = graph.entities[node_id]
        return f"{ent.name} ({ent.role})"
    return f"<unknown node: {node_id}>"


def edge_label(edge_type: str, edge_data: dict) -> str:
    """Short human-readable text for an edge, for trace/report output."""
    if edge_type == "fact_fact":
        return edge_data.get("label") or edge_data.get("description") or "relates to"
    if edge_type == "fact_entity":
        return f"has {edge_data.get('label', 'property')}"
    if edge_type == "entity_fact":
        return f"participates in: {edge_data.get('label', 'unknown')}"
    return edge_type


def lm_history_len() -> int:
    """Current length of dspy's global LM call history, or -1 if no LM is configured."""
    try:
        return len(dspy.settings.lm.history)
    except Exception:
        return -1


@dataclass
class TraceRecorder:
    """Collects structured events during one retrieve() call."""

    enabled: bool = True
    events: List[dict] = field(default_factory=list)

    def log(self, stage: str, **data: Any) -> None:
        if self.enabled:
            self.events.append({"stage": stage, **data})

    def lm_calls_since(self, start_idx: int) -> List[dict]:
        """
        Per-call token usage for every DSPy LM call made since start_idx. A
        planning stage that shows 2+ calls here means the adapter's primary
        format failed to parse and it silently fell back to a second full
        LLM call (see dspy.adapters.chat_adapter.ChatAdapter) -- direct
        evidence of where extra latency/tokens are going, rather than a guess.
        """
        if not self.enabled or start_idx < 0:
            return []
        try:
            entries = dspy.settings.lm.history[start_idx:]
        except Exception:
            return []
        calls = []
        for e in entries:
            u = e.get("usage") or {}
            calls.append({
                "prompt_tokens": u.get("prompt_tokens", 0),
                "completion_tokens": u.get("completion_tokens", 0),
                "total_tokens": u.get("total_tokens", 0),
            })
        return calls

    def stage(self, name: str) -> List[dict]:
        return [e for e in self.events if e["stage"] == name]
