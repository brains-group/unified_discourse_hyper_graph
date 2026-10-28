"""
GoalSearchEngine: the new nkg query engine.

Replaces nkg/retrieval end to end with a philosophy change, not a speed
optimization (see the D1-D4 redesign plan in project memory):

  1. Query planning decomposes the query into an ORDERED list of sub-goals
     instead of one flat bag of fields reused unchanged at every hop
     (goal_plan.py).
  2. Every selection step -- seeds, beam branches, and the final ranked set
     -- picks candidates by greedy goal *coverage* instead of MMR, so
     diversity is a side effect of covering distinct sub-goals rather than a
     separate embedding-dissimilarity penalty (coverage.py).
  3. Edge scoring during traversal was audited to be blind to destination-node
     content (only edge labels/descriptions were ever compared to the query);
     every hop now re-checks what the destination node actually says
     (content_scoring.py).
  4. High-degree "hub" entities (generic defined terms touching hundreds of
     facts) are down-weighted as traversal pivots instead of being treated as
     equally good hops as rare, specific entities (content_scoring.py).

All four cost exactly one LLM call per query, identical to the old engine. A
fifth, strictly optional stage ("island and bridge", islands.py) is available
via use_island_and_bridge=True on retrieve().
"""
from typing import List, Tuple, Union

import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

from nkg.models.Graph import Graph
from nkg.search.content_scoring import (
    hub_degree_penalty,
    score_entities_against_goals,
    score_facts_against_goals,
    score_paths_against_goals,
)
from nkg.search.coverage import select_by_goal_coverage
from nkg.search.formatting import assemble_path_string, format_path, format_paths_deduplicated
from nkg.search.goal_plan import EmbeddedGoalPlan, GoalPlan, generate_and_embed_goal_plan
from nkg.search.islands import find_island_bridges
from nkg.search.trace import TraceRecorder, lm_history_len, node_label
from nkg.search.traversal import VALID_EDGE_TYPES_BY_MODE, SearchPath, expand_frontier
from nkg.utils.math_utils import normalize_rows

import dspy

# Which node kinds participate in seeding for a given traversal mode. Mirrors
# VALID_EDGE_TYPES_BY_MODE in traversal.py: "discourse" traversal only ever
# follows fact_fact edges, so only facts are useful seeds, and symmetrically
# for "hypergraph".
_SEED_NODE_KINDS = {
    "all": {"entity", "fact"},
    "hypergraph": {"entity"},
    "discourse": {"fact"},
}


class GoalSearchEngine:
    """
    Public API mirrors the old Retriever's shape (same constructor style, a
    single `.retrieve()` entry point) so it's a drop-in replacement in
    evaluation harnesses -- only the internals implement the goal-coverage
    redesign.
    """

    def __init__(
        self,
        retrieval_model: SentenceTransformer,
        cross_encoder: CrossEncoder,
        graph: Graph = None,
        graph_path: str = None,
    ):
        self.retrieval_model = retrieval_model
        self.cross_encoder = cross_encoder
        self.planner = dspy.ChainOfThought(GoalPlan)

        if graph is not None:
            self.graph = graph
        elif graph_path is not None:
            self.graph = Graph()
            self.graph.load_graph(graph_path)
            self.graph.label_edges()
        else:
            raise ValueError("Must provide either a graph object or a graph_path.")

        self._initialize_content_matrices()
        self._initialize_edge_matrices()
        self._build_adj_dict()
        self._initialize_hub_penalty()

    # ------------------------------------------------------------------
    # Graph -> matrix precomputation (done once at construction time)
    # ------------------------------------------------------------------

    def _initialize_content_matrices(self, verbose: bool = False):
        """
        Flattens every entity/fact text field into one big batch-encode call,
        then reassembles the results into contiguous (N, D) matrices keyed by
        row-index maps (entity_row / fact_row). Ragged per-node fields
        (answered_questions, chunk_topics, fact entity names) are joined into
        a single string per node and embedded once, with a companion boolean
        mask, so every downstream comparison is a matmul rather than a
        per-node Python loop. relational_anchors is the exception: it's
        mean-pooled instead of joined, since anchors are full free-text spans
        rather than short terms and concatenating them would blur into one
        run-on sentence.
        """
        entity_ids = list(self.graph.entities.keys())
        fact_ids = list(self.graph.facts.keys())
        self.all_entity_ids = entity_ids
        self.all_fact_ids = fact_ids
        self.entity_row = {eid: i for i, eid in enumerate(entity_ids)}
        self.fact_row = {fid: i for i, fid in enumerate(fact_ids)}

        flat_strings: List[str] = []

        def push(s: str) -> int:
            flat_strings.append(s)
            return len(flat_strings) - 1

        entity_name_i = [push(self.graph.entities[eid].name) for eid in entity_ids]
        entity_role_i = [push(self.graph.entities[eid].role) for eid in entity_ids]

        anchor_spans = []  # (start, count) or None, per entity
        for eid in entity_ids:
            anchors = self.graph.entities[eid].relational_anchors
            if anchors:
                start = len(flat_strings)
                for a in anchors:
                    push(a)
                anchor_spans.append((start, len(anchors)))
            else:
                anchor_spans.append(None)

        fact_sent_i = [push(self.graph.facts[fid].sentence) for fid in fact_ids]

        questions_i = []  # index into flat_strings, or None, per fact
        for fid in fact_ids:
            questions = self.graph.facts[fid].answered_questions
            questions_i.append(push(" | ".join(questions)) if questions else None)

        chunk_topics_i = []  # index into flat_strings, or None, per fact
        for fid in fact_ids:
            topics = self.graph.facts[fid].chunk_topics
            chunk_topics_i.append(push(" | ".join(topics)) if topics else None)

        entity_names_i = []  # index into flat_strings, or None, per fact
        for fid in fact_ids:
            names = [e.name for e in self.graph.facts[fid].entities]
            entity_names_i.append(push(" | ".join(names)) if names else None)

        if verbose:
            print(f"Encoding {len(flat_strings)} text segments for goal-search content matrices...")
        all_embs = normalize_rows(self.retrieval_model.encode(flat_strings, show_progress_bar=verbose))
        self.emb_dim = all_embs.shape[1]

        n_ent, n_fact = len(entity_ids), len(fact_ids)

        self.entity_name_mat = all_embs[entity_name_i] if n_ent else np.zeros((0, self.emb_dim), dtype=np.float32)
        self.entity_role_mat = all_embs[entity_role_i] if n_ent else np.zeros((0, self.emb_dim), dtype=np.float32)

        self.entity_anchor_mat = np.zeros((n_ent, self.emb_dim), dtype=np.float32)
        self.has_anchors = np.zeros(n_ent, dtype=bool)
        for i, span in enumerate(anchor_spans):
            if span is not None:
                start, count = span
                self.entity_anchor_mat[i] = normalize_rows(all_embs[start:start + count].mean(axis=0, keepdims=True))[0]
                self.has_anchors[i] = True

        self.fact_sent_mat = all_embs[fact_sent_i] if n_fact else np.zeros((0, self.emb_dim), dtype=np.float32)

        self.fact_questions_mat = np.zeros((n_fact, self.emb_dim), dtype=np.float32)
        self.has_questions = np.zeros(n_fact, dtype=bool)
        for i, qi in enumerate(questions_i):
            if qi is not None:
                self.fact_questions_mat[i] = all_embs[qi]
                self.has_questions[i] = True

        self.fact_chunk_topics_mat = np.zeros((n_fact, self.emb_dim), dtype=np.float32)
        self.has_chunk_topics = np.zeros(n_fact, dtype=bool)
        for i, ci in enumerate(chunk_topics_i):
            if ci is not None:
                self.fact_chunk_topics_mat[i] = all_embs[ci]
                self.has_chunk_topics[i] = True

        self.fact_entity_names_mat = np.zeros((n_fact, self.emb_dim), dtype=np.float32)
        self.has_entity_names = np.zeros(n_fact, dtype=bool)
        for i, ni in enumerate(entity_names_i):
            if ni is not None:
                self.fact_entity_names_mat[i] = all_embs[ni]
                self.has_entity_names[i] = True

    def _initialize_edge_matrices(self, verbose: bool = False):
        """
        Embeds every edge's label (and, for fact_fact edges, description)
        once at construction time. Stored as {(src, tgt): 1D vector} dicts,
        gathered into batch matrices on demand during traversal.

        fact_entity edges are deliberately skipped here: construction sets
        their label to the destination entity's bare ontological type (e.g.
        "PERSON"), and search-time scoring for that edge type intentionally
        ignores type entirely in favor of the destination entity's own
        name/role (see content_scoring.score_fact_entity_candidates), so
        there is nothing useful to embed for them.
        """
        ff_edges, ff_descs, ff_labels = [], [], []
        ef_edges, ef_labels = [], []

        for u, v, data in self.graph.network.edges(data=True):
            if u in self.graph.facts and v in self.graph.facts:
                ff_edges.append((u, v))
                ff_descs.append(data.get("description") or "unlabeled")
                ff_labels.append(data.get("label") or "unlabeled")
            elif data.get("edge_type") == "entity_fact":
                ef_edges.append((u, v))
                ef_labels.append(data.get("label") or "unlabeled")

        self.ff_label_embs, self.ff_desc_embs = {}, {}
        if ff_edges:
            label_mat = normalize_rows(self.retrieval_model.encode(ff_labels, show_progress_bar=verbose))
            desc_mat = normalize_rows(self.retrieval_model.encode(ff_descs, show_progress_bar=verbose))
            for i, e in enumerate(ff_edges):
                self.ff_label_embs[e] = label_mat[i]
                self.ff_desc_embs[e] = desc_mat[i]

        self.ef_label_embs = {}
        if ef_edges:
            label_mat = normalize_rows(self.retrieval_model.encode(ef_labels, show_progress_bar=verbose))
            for i, e in enumerate(ef_edges):
                self.ef_label_embs[e] = label_mat[i]

    def _build_adj_dict(self):
        """
        Plain-dict adjacency, {node_id: [(neighbor_id, edge_type, edge_data), ...]},
        so every traversal step is an O(1) dict lookup rather than a NetworkX
        out_edges() call.
        """
        adj: dict = {}
        for u, v, data in self.graph.network.edges(data=True):
            adj.setdefault(u, []).append((v, data.get("edge_type"), data))
        self.adj = adj

    def _initialize_hub_penalty(self):
        """Per-entity traversal-pivot penalty, aligned to entity_row order. See content_scoring.hub_degree_penalty."""
        out_degree = np.array([len(self.adj.get(eid, [])) for eid in self.all_entity_ids], dtype=np.float32)
        self.entity_hub_penalty = hub_degree_penalty(out_degree)

    # ------------------------------------------------------------------
    # Retrieval stages
    # ------------------------------------------------------------------

    def select_seeds(
        self, goal_plan: EmbeddedGoalPlan, top_k: int = 20, mode: str = "all", tracer: TraceRecorder = None,
    ) -> List[str]:
        """Scores every eligible entity/fact against every goal, then picks top_k by goal coverage."""
        tracer = tracer or TraceRecorder(enabled=False)
        kinds = _SEED_NODE_KINDS[mode]
        candidate_ids: List[str] = []
        score_blocks: List[np.ndarray] = []

        if "entity" in kinds and self.all_entity_ids:
            candidate_ids.extend(self.all_entity_ids)
            score_blocks.append(score_entities_against_goals(
                self.entity_name_mat, self.entity_role_mat, self.entity_anchor_mat, self.has_anchors,
                goal_plan.sentence_mat, goal_plan.keyterms_mat,
            ))

        if "fact" in kinds and self.all_fact_ids:
            candidate_ids.extend(self.all_fact_ids)
            score_blocks.append(score_facts_against_goals(
                self.fact_sent_mat, self.fact_questions_mat, self.has_questions,
                self.fact_chunk_topics_mat, self.has_chunk_topics,
                self.fact_entity_names_mat, self.has_entity_names,
                goal_plan.sentence_mat, goal_plan.keyterms_mat,
            ))

        if not candidate_ids:
            return []

        score_matrix = np.vstack(score_blocks)
        selected = select_by_goal_coverage(score_matrix, top_k)

        if tracer.enabled:
            selected_rank = {i: rank for rank, i in enumerate(selected)}
            tracer.log(
                "seeding",
                goals=[g.goal for g in goal_plan.goals],
                candidates=[
                    {
                        "node_id": cid,
                        "text": node_label(self.graph, cid),
                        "scores": score_matrix[i].tolist(),
                        "selected": i in selected_rank,
                        "selection_rank": selected_rank.get(i),
                    }
                    for i, cid in enumerate(candidate_ids)
                ],
            )

        return [candidate_ids[i] for i in selected]

    def _rank_paths(
        self,
        query: str,
        goal_plan: EmbeddedGoalPlan,
        paths: List[SearchPath],
        final_top_k: int,
        verbose: bool = False,
        tracer: TraceRecorder = None,
    ) -> Tuple[List[str], List[SearchPath]]:
        """
        Cross-encoder scores each assembled path against the real query text
        (the only point in the pipeline that checks full path content), then
        picks the final set by goal coverage over [per-goal embedding
        similarity, normalized cross-encoder score] instead of MMR.
        """
        tracer = tracer or TraceRecorder(enabled=False)
        if not paths:
            return [], []

        assembled = [assemble_path_string(self.graph, p) for p in paths]

        if verbose:
            print(f"Running cross-encoder on {len(assembled)} assembled paths...")
        ce_scores = np.asarray(self.cross_encoder.predict([[query, s] for s in assembled]), dtype=np.float32)
        lo, hi = float(ce_scores.min()), float(ce_scores.max())
        normalized_relevance = (ce_scores - lo) / (hi - lo) if hi > lo else np.ones_like(ce_scores)

        path_embs = normalize_rows(self.retrieval_model.encode(assembled, show_progress_bar=False))

        score_matrix = score_paths_against_goals(path_embs, goal_plan.sentence_mat, normalized_relevance)
        selected = select_by_goal_coverage(score_matrix, final_top_k)

        if tracer.enabled:
            selected_rank = {i: rank for rank, i in enumerate(selected)}
            tracer.log(
                "final_ranking",
                goals=[g.goal for g in goal_plan.goals],
                candidates=[
                    {
                        "path_text": assembled[i],
                        "cross_encoder_score": float(ce_scores[i]),
                        "normalized_relevance": float(normalized_relevance[i]),
                        "goal_coverage_scores": score_matrix[i].tolist(),
                        "selected": i in selected_rank,
                        "selection_rank": selected_rank.get(i),
                    }
                    for i in range(len(paths))
                ],
            )

        return [assembled[i] for i in selected], [paths[i] for i in selected]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k_seeds: int = 20,
        max_depth: int = 3,
        beam_width: int = 3,
        final_top_k: int = 5,
        goal_lookahead: int = 1,
        mode: str = "all",
        use_island_and_bridge: bool = False,
        island_pool_size: int = 75,
        island_max_bridge_hops: int = 1,
        deduplicate_paths: bool = True,
        neutral_layout: bool = False,
        return_raw_paths: bool = False,
        verbose: bool = False,
        trace: bool = False,
    ) -> Union[str, Tuple[str, List[SearchPath]], Tuple[Union[str, Tuple[str, List[SearchPath]]], TraceRecorder]]:
        """
        Main public entry point. Plans once, traverses, ranks, and formats a
        single context string for the generation LLM.

        mode:                   Which edge/node types traversal may use.
                                "all" (fact_fact + entity_fact + fact_entity),
                                "hypergraph" (entity<->fact edges only), or
                                "discourse" (fact_fact edges only).
        goal_lookahead:         How many independent (not depends_on_previous)
                                upcoming goals traversal may opportunistically
                                score against, in addition to the depth's
                                primary goal. See goal_plan.active_goal_indices.
        use_island_and_bridge:  Optional D5 stage (off by default). When True,
                                also builds ready-made chains from pairs of
                                independently high-relevance nodes that are
                                already directly connected in the graph, and
                                merges them into the candidate pool before
                                final ranking. Pure addition -- never replaces
                                the beam-search paths, only supplements them.
        island_pool_size:       (island-and-bridge only) size of the relevance-
                                ranked candidate pool searched for bridges.
        island_max_bridge_hops: (island-and-bridge only) how many extra hops
                                each bridge may be greedily extended by.
        deduplicate_paths:      If True, paths are clustered by node overlap and
                                only novel edge "bursts" from lower-priority
                                paths are included, cutting repetition.
        neutral_layout:         If True, fact->fact edges render with a short
                                inline label and their long descriptions are
                                collected into a trailing "Logical Conclusions"
                                block instead of being inlined.
        trace:                  If True, also returns a TraceRecorder (as the
                                last element of a tuple, alongside whatever
                                return_raw_paths would already produce) with a
                                full structured record of the goal plan (incl.
                                dspy LM call count/token usage -- 2+ calls
                                means the ChatAdapter's JSON parse failed and
                                silently fell back to a second full LLM call),
                                every seed candidate's per-goal score, every
                                edge considered at every traversal hop and
                                whether it was kept, and every final-ranking
                                candidate's cross-encoder + coverage scores.
                                See trace.py. Off by default; costs materially
                                more memory/time when on, since score matrices
                                get copied into plain-Python trace records.
        """
        if mode not in VALID_EDGE_TYPES_BY_MODE:
            raise ValueError(f"Unknown mode '{mode}'. Choose one of {sorted(VALID_EDGE_TYPES_BY_MODE)}.")

        tracer = TraceRecorder(enabled=trace)

        if verbose:
            print(f"\n--- GoalSearchEngine retrieving for: '{query}' (mode={mode}) ---")

        lm_start = lm_history_len() if trace else -1
        goal_plan = generate_and_embed_goal_plan(query, self.retrieval_model, planner=self.planner)
        if verbose:
            print(f"Planned {len(goal_plan.goals)} goal(s): {[g.goal for g in goal_plan.goals]}")
        if trace:
            raw = goal_plan.raw_plan
            tracer.log(
                "goal_plan",
                reasoning=getattr(raw, "reasoning", None),
                goals=[
                    {"goal": g.goal, "key_terms": g.key_terms, "depends_on_previous": g.depends_on_previous}
                    for g in goal_plan.goals
                ],
                lm_calls=tracer.lm_calls_since(lm_start),
            )

        seeds = self.select_seeds(goal_plan, top_k=top_k_seeds, mode=mode, tracer=tracer)
        if not seeds:
            result = ("", []) if return_raw_paths else ""
            return (result, tracer) if trace else result

        completed_paths = expand_frontier(
            engine=self,
            seeds=seeds,
            goal_plan=goal_plan,
            max_depth=max_depth,
            beam_width=beam_width,
            mode=mode,
            goal_lookahead=goal_lookahead,
            tracer=tracer,
        )
        if verbose:
            print(f"Beam search produced {len(completed_paths)} candidate paths.")

        if use_island_and_bridge:
            island_seeds = self.select_seeds(goal_plan, top_k=island_pool_size, mode=mode)
            bridge_paths = find_island_bridges(
                engine=self,
                island_node_ids=island_seeds,
                goal_plan=goal_plan,
                mode=mode,
                max_bridge_hops=island_max_bridge_hops,
                tracer=tracer,
            )
            if verbose:
                print(f"Island-and-bridge added {len(bridge_paths)} candidate paths.")
            completed_paths = completed_paths + bridge_paths

        final_strings, final_paths = self._rank_paths(
            query, goal_plan, completed_paths, final_top_k, verbose=verbose, tracer=tracer,
        )

        if deduplicate_paths:
            context = format_paths_deduplicated(self.graph, final_paths, neutral_layout)
        elif neutral_layout:
            context = "\n\n".join(format_path(self.graph, p, neutral_layout=True) for p in final_paths)
        else:
            context = "\n\n".join(final_strings)

        if verbose:
            print("--- Retrieval complete ---")

        result = (context, final_paths) if return_raw_paths else context
        return (result, tracer) if trace else result


def main():
    from nkg.utils.config import configure_dspy

    configure_dspy(max_tokens=30000)

    bi_encoder = SentenceTransformer("Qwen/Qwen3-Embedding-4B")
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    engine = GoalSearchEngine(
        retrieval_model=bi_encoder,
        cross_encoder=cross_encoder,
        graph_path="./my_knowledge_graph.graphml",
    )

    context = engine.retrieve(
        query="What are the liability limits for the main policyholder's vehicle?",
        top_k_seeds=10,
        max_depth=2,
        beam_width=3,
        final_top_k=4,
    )
    print("Context to inject into generation prompt:\n")
    print(context)


if __name__ == "__main__":
    main()
