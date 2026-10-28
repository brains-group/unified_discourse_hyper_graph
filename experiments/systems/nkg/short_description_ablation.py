#!/usr/bin/env python3
"""
short_description_ablation.py
==============================
Ablation: how does UNH (nkg) do when fact->fact edges carry ONLY their short label
(e.g. "ENSURES", "IS_BASIS_FOR") and never their long description (a ~17-word sentence such as
"The legal deposit obligation ensures the British National Bibliography is the most
comprehensive listing of UK titles.")?

The long description is used in three places at query time:
  1. edge scoring: both engines embed every fact->fact edge's description (old/dag engine
     weight 0.5, new engine 0.20) next to its label to decide which edges to follow;
  2. the cross-encoder path strings (assemble_path_string inlines the description);
  3. the context handed to the answer LLM (standard layout inlines it; the neutral layout used in
     every previous run puts the label inline and appends all descriptions as
     "Logical Conclusions").

Rather than touching the retrieval code, this script writes a copy of an already-built graph in
which every fact->fact edge's `description` is replaced by its `label`. Evaluated with
--no-neutral-layout (description inline), all three places then see only the short label; nothing
else in the graph (facts, entities, entity roles, other edge types, scores) changes. No LLM call is
needed: the graph is not rebuilt.

  python -m experiments.systems.nkg.short_description_ablation \\
      --source-run sample_exp/nkg/hotpotqa/run_v2 \\
      --out-root   sample_exp/nkg/hotpotqa/short_desc_ablation

writes
  <out-root>/short/kg.graphml                 the short-label graph
  <out-root>/short/construction_results.json  copied (evaluation mode requires it)
  <out-root>/long/{kg.graphml,construction_results.json}  symlinks to the source run, so the
                                              baseline's evaluation_results_N.json land here and
                                              not in the source run dir
  <out-root>/ablation_info.json               what was changed (counts, a few examples)
With --strict it also writes <out-root>/short_strict/: the same, plus every entity->fact edge label emptied.
In these July graphs those labels hold ~11-word text (over half are a copy of the target fact's sentence),
although current construction leaves them empty unless generate_entity_relation_labels=True; an empty label
renders as a bare "(participates in event)", which is what current code produces.
Then evaluate with hqa_musique_evaluation.py (see run_short_description_ablation.sh).
"""
import argparse
import json
import os
import random
import shutil
from collections import Counter
from datetime import datetime

import networkx as nx

KG = "kg.graphml"
CONSTRUCTION_RESULTS = "construction_results.json"
CONSTRUCTION_CONFIG = "construction_config.json"


def make_short_label_graph(src_kg: str, dst_kg: str, strict: bool = False) -> dict:
    """Write a copy of src_kg in which every fact_fact edge's description is its short label (and, with
    strict, every entity_fact label is emptied). Nothing else changes. Returns counts and a few examples.
    Also used by evaluation_scripts/unh/main.py."""
    g = nx.read_graphml(src_kg)
    edge_types = Counter(d.get("edge_type") for _, _, d in g.edges(data=True))
    with_desc = Counter(d.get("edge_type") for _, _, d in g.edges(data=True) if d.get("description"))
    changed, missing_label, examples = 0, 0, []
    ff = [(u, v, d) for u, v, d in g.edges(data=True) if d.get("edge_type") == "fact_fact"]
    random.Random(0).shuffle(ff)
    for u, v, d in ff:
        label = (d.get("label") or "").strip()
        if not label:
            missing_label += 1
            label = "relates to"  # the placeholder the formatter itself uses when a label is missing
        if len(examples) < 5:
            examples.append({"label": label, "long_description_removed": d.get("description", "")})
        d["description"] = label
        changed += 1
    leftover = [t for t, n in with_desc.items() if t != "fact_fact" and n]
    if leftover:
        raise SystemExit(f"edge types other than fact_fact carry a description: {leftover}; extend this script")
    ef_emptied = 0
    if strict:
        for _, _, d in g.edges(data=True):
            if d.get("edge_type") == "entity_fact" and d.get("label"):
                d["label"] = ""
                ef_emptied += 1
    nx.write_graphml(g, dst_kg)
    return {"edge_types": dict(edge_types), "edges_with_description_before": dict(with_desc),
            "fact_fact_edges_changed": changed, "fact_fact_edges_without_label": missing_label,
            "entity_fact_labels_emptied": ef_emptied, "examples": examples}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source-run", required=True, help="a finished construction run dir (kg.graphml + construction_results.json)")
    ap.add_argument("--out-root", required=True, help="creates <out-root>/long and <out-root>/short")
    ap.add_argument("--force", action="store_true", help="rewrite the variant kg.graphml if it exists")
    ap.add_argument("--strict", action="store_true",
                    help="write <out-root>/short_strict/ instead: short labels AND empty entity->fact labels")
    args = ap.parse_args()

    src = os.path.abspath(args.source_run)
    for f in (KG, CONSTRUCTION_RESULTS):
        if not os.path.exists(os.path.join(src, f)):
            raise SystemExit(f"{os.path.join(src, f)} not found")
    long_dir = os.path.join(args.out_root, "long")
    short_dir = os.path.join(args.out_root, "short_strict" if args.strict else "short")
    os.makedirs(long_dir, exist_ok=True)
    os.makedirs(short_dir, exist_ok=True)

    # baseline: symlinks to the unchanged source graph
    for f in (KG, CONSTRUCTION_RESULTS, CONSTRUCTION_CONFIG):
        target, link = os.path.join(src, f), os.path.join(long_dir, f)
        if os.path.exists(target) and not os.path.lexists(link):
            os.symlink(target, link)

    short_kg = os.path.join(short_dir, KG)
    if os.path.exists(short_kg) and not args.force:
        print(f"[ablation] {short_kg} exists; keeping it (--force to rewrite)")
        return

    stats = make_short_label_graph(os.path.join(src, KG), short_kg, strict=args.strict)
    for f in (CONSTRUCTION_RESULTS, CONSTRUCTION_CONFIG):
        if os.path.exists(os.path.join(src, f)):
            shutil.copy2(os.path.join(src, f), os.path.join(short_dir, f))

    info = {
        "created": datetime.now().isoformat(),
        "source_run": src,
        "variant": os.path.basename(short_dir),
        "change": "every fact_fact edge: description := label (short label only)"
                  + ("; every entity_fact label emptied" if args.strict else "; nothing else changed"),
        **stats,
        "evaluate_with": {"long/": "original graph, original (neutral) layout, as in every previous run",
                          "short/": "this graph with --no-neutral-layout, so only the label appears anywhere"},
    }
    json.dump(info, open(os.path.join(args.out_root, f"ablation_info_{os.path.basename(short_dir)}.json"), "w"), indent=2)
    print(json.dumps({k: info[k] for k in ("source_run", "variant", "edge_types", "fact_fact_edges_changed",
                                           "fact_fact_edges_without_label", "entity_fact_labels_emptied")}, indent=2))
    print(f"[ablation] wrote {short_kg}")


if __name__ == "__main__":
    main()
