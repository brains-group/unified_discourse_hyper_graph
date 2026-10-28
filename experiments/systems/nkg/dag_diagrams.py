"""
Matplotlib renderers for nkg.dag_retrieval structures, used by dag_report.py.

Two diagrams per question:
  - render_plan_dag():       the as-planned query DAG (PlanNode.depends_on), before any
                              searching happened.
  - render_execution_dag():  every concrete NodeExecution (including every fork branch a
                              node expanded into), colored by which original plan node it
                              came from, with edges drawn from each branch's actual
                              parent_branch_ids -- so forks/joins in the real execution are
                              visible exactly as they happened.

Both share the same node_id -> color assignment (build_node_color_map) so a branch like
"q2#0" in the execution diagram is visually traceable back to "q2" in the plan diagram.
"""
import re
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

_PALETTE = plt.get_cmap("tab10").colors
_NO_RESULTS_MARKERS = ("(no relevant information found.)", "")

_BOX_WIDTH_IN = 2.6
_LEVEL_GAP_IN = 1.1
_BOX_GAP_IN = 0.35
_WRAP_CHARS = 32
_HEADER_FONTSIZE = 9
_BODY_FONTSIZE = 7.5
_LINE_HEIGHT_IN = 0.155
_PAD_IN = 0.12


def _natural_key(s: str):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def build_node_color_map(node_ids) -> dict:
    ordered = sorted(set(node_ids), key=_natural_key)
    return {nid: _PALETTE[i % len(_PALETTE)] for i, nid in enumerate(ordered)}


def _wrap(text: str, width: int = _WRAP_CHARS, max_lines: int = 6) -> list:
    lines = textwrap.wrap(text or "", width=width) or [""]
    if len(lines) > max_lines:
        lines = lines[: max_lines - 1] + [lines[max_lines - 1][: width - 1] + "…"]
    return lines


def _box_height(n_body_lines: int) -> float:
    return 2 * _PAD_IN + _LINE_HEIGHT_IN * (n_body_lines + 1.3)


def _is_empty_result(context_string: str) -> bool:
    return (context_string or "").strip().lower() in _NO_RESULTS_MARKERS


def _draw_layered_dag(boxes: list, edges: list, out_path: str, fig_title: str = None):
    """
    boxes: list of dicts: {id, level, header, body_lines, color, empty(bool)}
    edges: list of (src_id, dst_id)
    """
    by_level = {}
    for b in boxes:
        by_level.setdefault(b["level"], []).append(b)
    for lvl in by_level:
        by_level[lvl].sort(key=lambda b: _natural_key(b["id"]))

    heights = {b["id"]: _box_height(len(b["body_lines"])) for b in boxes}

    n_levels = max(by_level) + 1 if by_level else 1
    max_row_width = max(
        len(row) * _BOX_WIDTH_IN + (len(row) - 1) * _BOX_GAP_IN for row in by_level.values()
    )

    positions = {}
    y_cursor = 0.0
    row_height_by_level = {}
    for level in sorted(by_level.keys()):
        row = by_level[level]
        row_height = max(heights[b["id"]] for b in row)
        row_height_by_level[level] = row_height
        row_width = len(row) * _BOX_WIDTH_IN + (len(row) - 1) * _BOX_GAP_IN
        x_start = (max_row_width - row_width) / 2.0
        for i, b in enumerate(row):
            x = x_start + i * (_BOX_WIDTH_IN + _BOX_GAP_IN)
            positions[b["id"]] = (x, y_cursor)
        y_cursor += row_height + _LEVEL_GAP_IN

    total_height = y_cursor - _LEVEL_GAP_IN
    title_pad = 0.5 if fig_title else 0.0
    fig_w = max_row_width + 2 * _PAD_IN
    fig_h = total_height + 2 * _PAD_IN + title_pad

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.invert_yaxis()
    ax.axis("off")
    if fig_title:
        ax.text(fig_w / 2, 0.15, fig_title, ha="center", va="top",
                 fontsize=12, fontweight="bold")

    by_id = {b["id"]: b for b in boxes}
    box_rect = {}
    for b in boxes:
        x, y = positions[b["id"]]
        x += _PAD_IN
        y += title_pad + _PAD_IN
        w = _BOX_WIDTH_IN
        h = heights[b["id"]]
        face = b["color"]
        edge = "#b91c1c" if b.get("empty") else tuple(c * 0.6 for c in b["color"][:3])
        linewidth = 2.2 if b.get("forked") else 1.3
        rect = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            linewidth=linewidth,
            edgecolor=edge,
            facecolor=(*face[:3], 0.22) if len(face) >= 3 else face,
        )
        ax.add_patch(rect)
        box_rect[b["id"]] = (x, y, w, h)

        ax.text(x + w / 2, y + _PAD_IN, b["header"], ha="center", va="top",
                 fontsize=_HEADER_FONTSIZE, fontweight="bold", color="#111827")
        body = "\n".join(b["body_lines"])
        ax.text(x + w / 2, y + _PAD_IN + _LINE_HEIGHT_IN * 1.3, body,
                 ha="center", va="top", fontsize=_BODY_FONTSIZE, color="#1f2937")

    for src, dst in edges:
        if src not in box_rect or dst not in box_rect:
            continue
        sx, sy, sw, sh = box_rect[src]
        dx, dy, dw, dh = box_rect[dst]
        start = (sx + sw / 2, sy + sh)
        end = (dx + dw / 2, dy)
        arrow = FancyArrowPatch(
            start, end,
            connectionstyle="arc3,rad=0.12",
            arrowstyle="-|>", mutation_scale=14,
            linewidth=1.1, color="#6b7280", zorder=1,
        )
        ax.add_patch(arrow)

    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return out_path


def render_plan_dag(by_id: dict, levels: dict, out_path: str, title: str = "Initial DAG Plan"):
    """by_id/levels as produced by nkg.dag_retrieval.planning.sanitize_and_level."""
    color_map = build_node_color_map(by_id.keys())
    boxes = []
    for nid, node in by_id.items():
        body_lines = _wrap(node.query)
        boxes.append({
            "id": nid, "level": levels[nid], "header": nid,
            "body_lines": body_lines, "color": color_map[nid],
            "forked": False, "empty": False,
        })
    edges = [(dep, node.id) for node in by_id.values() for dep in node.depends_on]
    return _draw_layered_dag(boxes, edges, out_path, fig_title=title)


def render_execution_dag(executions: list, out_path: str, title: str = "Executed DAG (all fork branches)"):
    """
    executions: list of nkg.dag_retrieval.engine.NodeExecution.

    Every fork branch of every node is drawn as its own box, colored by its original
    node_id (matching render_plan_dag's colors), with edges from each branch's actual
    parent_branch_ids. A node_id with >1 branch is outlined thicker ("forked"); a branch
    whose retrieval came back empty is outlined red and flagged, since that's exactly the
    kind of failure this report exists to surface.
    """
    node_id_counts = {}
    for ex in executions:
        node_id_counts[ex.node_id] = node_id_counts.get(ex.node_id, 0) + 1
    color_map = build_node_color_map(node_id_counts.keys())

    boxes = []
    for ex in executions:
        body_lines = _wrap(ex.refined_query)
        n_paths = len(ex.raw_paths or [])
        empty = _is_empty_result(ex.context_string)
        stat = "⚠ no results" if empty else f"{n_paths} path(s) retrieved"
        body_lines = body_lines + ["", stat]
        boxes.append({
            "id": ex.branch_id, "level": ex.level,
            "header": f"{ex.node_id}  [{ex.branch_id}]" if ex.branch_id != ex.node_id else ex.node_id,
            "body_lines": body_lines, "color": color_map[ex.node_id],
            "forked": node_id_counts[ex.node_id] > 1, "empty": empty,
        })
    edges = [(parent, ex.branch_id) for ex in executions for parent in ex.parent_branch_ids]
    return _draw_layered_dag(boxes, edges, out_path, fig_title=title)
