"""
Turns ranked SearchPath objects into the final context string handed to the
generation LLM. Carried over in spirit from the old retrieval engine's
formatting helpers, adapted to SearchPath -- this part of the pipeline was
never what the redesign targeted, so its behavior is preserved rather than
rewritten.
"""
from typing import Dict, List, Optional, Tuple

from nkg.models.Graph import Graph
from nkg.search_v2.traversal import SearchPath


def _node_repr(graph: Graph, node_id: str) -> str:
    if node_id in graph.facts:
        return f'"{graph.facts[node_id].sentence}"'
    if node_id in graph.entities:
        ent = graph.entities[node_id]
        return f"[{ent.name} ({ent.role})]"
    return "[Unknown Node]"


def assemble_path_string(graph: Graph, path: SearchPath) -> str:
    """Flattens a path's full history into one narrative string, for the cross-encoder."""
    if not path.history:
        return _node_repr(graph, path.current_node)

    result = _node_repr(graph, path.history[0][0])
    for _, edge_type, edge_data, target_id in path.history:
        target_rep = _node_repr(graph, target_id)
        if edge_type == "fact_fact":
            result += f" -> {edge_data.get('description', 'is related to')} -> {target_rep}"
        elif edge_type == "fact_entity":
            result += f" -> (has {edge_data.get('label', 'property')}) -> {target_rep}"
        elif edge_type == "entity_fact":
            result += f" -> (participates in event: {edge_data.get('label', 'unknown')}) -> {target_rep}"
        else:
            result += f" -> {target_rep}"
    return result


def _path_nodes(path: SearchPath) -> List[str]:
    if not path.history:
        return [path.current_node]
    return [path.history[0][0]] + [tgt for _, _, _, tgt in path.history]


def _edge_lookup(path: SearchPath) -> Dict[Tuple[str, str], Tuple[str, dict]]:
    return {(src, tgt): (edge_type, edge_data) for src, edge_type, edge_data, tgt in path.history}


def _format_node_sequence(
    graph: Graph,
    nodes: List[str],
    edge_lookup: Dict[Tuple[str, str], Tuple[str, dict]],
    neutral_layout: bool,
) -> str:
    """
    Standard layout inlines the full edge description between each pair of
    fact nodes. Neutral layout instead inlines a short label and collects the
    long descriptions into a trailing "Logical Conclusions" block, so the LLM
    sees the high-level path shape before the detailed reasoning. Entity
    edges are formatted identically in both layouts since they carry no long
    description.
    """
    if not nodes:
        return ""

    result = _node_repr(graph, nodes[0])
    conclusions: List[str] = []

    for i in range(len(nodes) - 1):
        src, tgt = nodes[i], nodes[i + 1]
        entry = edge_lookup.get((src, tgt))
        tgt_rep = _node_repr(graph, tgt)

        if entry is None:
            result += f" -> {tgt_rep}"
            continue

        edge_type, edge_data = entry
        if edge_type == "fact_fact":
            if neutral_layout:
                result += f" -> {edge_data.get('label', 'relates to')} -> {tgt_rep}"
                desc = edge_data.get("description", "")
                if desc:
                    conclusions.append(desc)
            else:
                result += f" -> {edge_data.get('description', 'is related to')} -> {tgt_rep}"
        elif edge_type == "fact_entity":
            result += f" -> (has {edge_data.get('label', 'property')}) -> {tgt_rep}"
        elif edge_type == "entity_fact":
            result += f" -> (participates in event: {edge_data.get('label', 'unknown')}) -> {tgt_rep}"
        else:
            result += f" -> {tgt_rep}"

    if neutral_layout and conclusions:
        result += "\nLogical Conclusions:\n" + "\n".join(conclusions)

    return result


def format_path(graph: Graph, path: SearchPath, neutral_layout: bool = False) -> str:
    """Formats a single path into a context string."""
    return _format_node_sequence(graph, _path_nodes(path), _edge_lookup(path), neutral_layout)


def _extract_bursts(nodes: List[str], edge_cache: set) -> List[List[str]]:
    """
    Scans a node sequence and returns contiguous bursts where every
    consecutive edge is absent from edge_cache, updating edge_cache in place
    as each novel edge is found so repeats within the same path are also
    suppressed.
    """
    bursts: List[List[str]] = []
    current: Optional[List[str]] = None

    for i in range(len(nodes) - 1):
        a, b = nodes[i], nodes[i + 1]
        if (a, b) not in edge_cache and (b, a) not in edge_cache:
            current = [a, b] if current is None else current + [b]
            edge_cache.add((a, b))
            edge_cache.add((b, a))
        else:
            if current is not None:
                bursts.append(current)
                current = None

    if current is not None:
        bursts.append(current)

    return bursts


def format_paths_deduplicated(graph: Graph, ranked_paths: List[SearchPath], neutral_layout: bool = False) -> str:
    """
    Converts a priority-ranked list of paths into a deduplicated context
    string:
    - Paths are processed in clusters. The highest-priority remaining path
      becomes the cluster's "base" and is always printed in full.
    - Remaining paths are split into those sharing a node with the base
      ("related") and those that don't ("unrelated"); related paths are
      sorted by node-overlap with the base (most overlap first) so
      deduplication happens against the most complete cache state first.
    - Each related path is scanned for "bursts" -- contiguous runs of edges
      not already printed -- and only those bursts are added.
    - Unrelated paths seed the next cluster.
    """
    if not ranked_paths:
        return ""

    edge_cache: set = set()
    context_parts: List[str] = []

    path_data = [
        {"nodes": _path_nodes(p), "node_set": set(_path_nodes(p)), "edge_lookup": _edge_lookup(p)}
        for p in ranked_paths
    ]
    remaining = list(path_data)

    while remaining:
        base = remaining.pop(0)

        base_str = _format_node_sequence(graph, base["nodes"], base["edge_lookup"], neutral_layout)
        if base_str:
            context_parts.append(base_str)

        for i in range(len(base["nodes"]) - 1):
            a, b = base["nodes"][i], base["nodes"][i + 1]
            edge_cache.add((a, b))
            edge_cache.add((b, a))

        related: List[Tuple[int, dict]] = []
        unrelated: List[dict] = []
        for pd in remaining:
            overlap = len(pd["node_set"] & base["node_set"])
            if overlap > 0:
                related.append((overlap, pd))
            else:
                unrelated.append(pd)

        related.sort(key=lambda x: x[0], reverse=True)

        for _, pd in related:
            for burst_nodes in _extract_bursts(pd["nodes"], edge_cache):
                burst_str = _format_node_sequence(graph, burst_nodes, pd["edge_lookup"], neutral_layout)
                if burst_str:
                    context_parts.append(burst_str)

        remaining = unrelated

    return "\n\n".join(context_parts)
