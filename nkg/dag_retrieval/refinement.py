"""
Per-node refinement: run once a DAG node's dependencies have real retrieved
context, right before that node is searched.

This is where ambiguous references get resolved ("that university" -> "the
University of Zurich") and where the same structured search parameters the
old QueryPlan (nkg/retrieval/planner.py) would have produced -- entities,
anchors, topics, edge labels/semantics -- get filled in for this specific,
now-disambiguated node. Source nodes (no dependencies) skip this call
entirely and let the wrapped Retriever's own internal QueryPlan handle
extraction, since there is nothing to disambiguate yet.
"""
import dspy
from pydantic import BaseModel, Field


class RefinedSubquery(BaseModel):
    query: str = Field(
        description="The standalone, unambiguous search query -- every resolvable placeholder "
                    "reference replaced by the concrete name/value found in prerequisite_context."
    )
    target_entities: list[str] = Field(
        default_factory=list,
        description="Specific named instances (e.g. 'John Doe', 'Policy 123'). Exact proper "
                    "nouns/IDs only."
    )
    broad_anchors: list[str] = Field(
        default_factory=list,
        description="Role-based entities or concepts (e.g. 'policyholder', 'alumnus'). Do not "
                    "overlap with target_entities."
    )
    target_topics: list[str] = Field(
        default_factory=list,
        description="Broad thematic categories this subquery touches upon."
    )
    target_edge_labels: list[str] = Field(
        default_factory=list,
        description="Strict 2-4 word UPPERCASE_WITH_UNDERSCORES categorical relation labels the "
                    "answer likely sits behind (e.g. 'ALUMNUS_OF')."
    )
    target_edge_semantics: list[str] = Field(
        default_factory=list,
        description="Rich descriptive phrases of the narrative relationship needed to connect "
                    "the dots."
    )


class RefineSubquery(dspy.Signature):
    """
    You are about to execute ONE subquery from a larger search-plan DAG. Some of its wording may
    be a placeholder reference (e.g. "that university", "the common actor") that only became
    knowable once its prerequisite subqueries were answered. prerequisite_context holds what was
    actually retrieved for each of those prerequisites (empty if this subquery has none).

    Do two things:
    1. Rewrite subquery into a standalone, unambiguous natural-language query with every
       resolvable reference replaced by the concrete name/value found in prerequisite_context.
       If prerequisite_context is empty, the subquery is already standalone -- keep its wording
       as-is.
    2. Extract the structured search parameters for it: target_entities (exact proper nouns),
       broad_anchors (role/type terms), target_topics (broad themes), target_edge_labels (short
       UPPERCASE relation labels), target_edge_semantics (descriptive relation phrases). Limit
       every list to 1-3 precise items; leave a list empty rather than inventing filler.

    If prerequisite_context reveals MULTIPLE distinct, equally plausible values for the same
    placeholder (e.g. three different shared actors, not one), output one refined entry PER
    distinct value instead of arbitrarily picking one -- this lets the search branch and check
    every possibility. Only fork when the context genuinely contains multiple concrete
    resolutions; the ordinary case outputs exactly one entry.
    """
    overall_query: str = dspy.InputField(desc="The user's original, full question.")
    subquery: str = dspy.InputField(
        desc="This node's original subquery text, possibly with unresolved placeholder references."
    )
    prerequisite_context: str = dspy.InputField(
        desc="Retrieved answer-context for each of this subquery's prerequisites, formatted for "
             "reading. Empty string if this subquery has no prerequisites."
    )

    refined: list[RefinedSubquery] = dspy.OutputField(
        desc="Normally one entry; more than one only when genuinely forking on multiple resolved "
             "values."
    )
