from sentence_transformers import SentenceTransformer
from typing import Optional, Dict
from kg_gen import KGGen
import json
import os
import dspy


class CustomKGGen(KGGen):
    """A wrapper class that fixes the hardcoded depth limitation in KGGen's retrieval."""

    def init_model(self, *args, **kwargs):
        """
        Intercepts the pip version's hardcoded setup, lets it finish,
        and then forcefully rips the cache out of the DSPy object.
        """
        # 1. Let the original KGGen pip version do its hardcoded setup
        super().init_model(*args, **kwargs)

        # Initialize dspy LM with current settings
        if self.api_key:
            self.lm = dspy.LM(
                model=self.model,
                api_key=self.api_key,
                reasoning={"effort": self.reasoning_effort}
                if self.reasoning_effort
                else None,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                api_base=self.api_base,
                cache=False,
                model_type="responses" if self.model.startswith("openai/") else "chat",
            )
        else:
            self.lm = dspy.LM(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                api_base=self.api_base,
                reasoning={"effort": self.reasoning_effort}
                if self.reasoning_effort
                else None,
                cache=False,
                model_type="responses" if self.model.startswith("openai/") else "chat",
            )

    def retrieve(
            self,
            query: str,
            node_embeddings: dict,
            graph,  # nx.DiGraph
            model: Optional[SentenceTransformer] = None,
            k: int = 8,
            depth: int = 2,  # <-- WE ADDED THE DEPTH PARAMETER HERE
            verbose: bool = False,
    ):
        model = self._parse_embedding_model(model)
        top_nodes = self.retrieve_relevant_nodes(query, node_embeddings, model, k)
        context = set()

        for node, _ in top_nodes:
            # <-- WE PASS THE DEPTH DOWN TO RETRIEVE_CONTEXT HERE
            node_context = self.retrieve_context(node, graph, depth=depth)

            if verbose:
                print(f"Context for node {node}: {node_context}")
            context.update(node_context)

        context_text = " ".join(context)
        if verbose:
            print(f"Combined context: '{context_text}'\n---")

        return top_nodes, context, context_text

    def extract_token_usage_from_history(self):
        """
        Aggressively hunts for token usage in DSPy's history log,
        handling dicts, objects, and nested lists.
        """
        total_prompt = 0
        total_completion = 0
        total_all = 0

        if not hasattr(self, 'lm') or not self.lm or getattr(self.lm, 'history', None) is None:
            return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        for entry in self.lm.history:
            usage = None

            # 1. Look in the top-level entry
            if isinstance(entry, dict):
                usage = entry.get("usage")

                # 2. Look inside the 'response' key (could be a dict, object, or list)
                if not usage and "response" in entry:
                    resp = entry["response"]

                    # If response is a dict
                    if isinstance(resp, dict):
                        usage = resp.get("usage")

                    # If response is a raw vLLM/LiteLLM object
                    elif hasattr(resp, "usage"):
                        usage = resp.usage

                    # If response is a list (newer DSPy versions sometimes do this)
                    elif isinstance(resp, list) and len(resp) > 0:
                        first_item = resp[0]
                        if isinstance(first_item, dict):
                            usage = first_item.get("usage")
                        elif hasattr(first_item, "usage"):
                            usage = first_item.usage

            # 3. Extract the tokens if we found the usage block!
            if usage:
                # Handle dictionary usage blocks
                if isinstance(usage, dict):
                    total_prompt += usage.get("prompt_tokens", 0)
                    total_completion += usage.get("completion_tokens", 0)
                    total_all += usage.get("total_tokens", 0)

                # Handle object usage blocks
                else:
                    total_prompt += getattr(usage, "prompt_tokens", 0)
                    total_completion += getattr(usage, "completion_tokens", 0)
                    total_all += getattr(usage, "total_tokens", 0)

        # 4. Fallback: Some DSPy LM instances track tokens directly on the LM object
        if total_all == 0:
            total_prompt = getattr(self.lm, "prompt_tokens", 0)
            total_completion = getattr(self.lm, "completion_tokens", 0)
            total_all = total_prompt + total_completion

        return {
            "prompt_tokens": total_prompt,
            "completion_tokens": total_completion,
            "total_tokens": total_all,
        }

    @staticmethod
    def export_graph(graph, output_path: str):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        graph_dict = {
            "entities": list(graph.entities),
            "relations": list(graph.relations),
            "edges": list(graph.edges),
            "entity_clusters": {k: list(v) for k, v in
                                graph.entity_clusters.items()} if graph.entity_clusters else None,
            "edge_clusters": {k: list(v) for k, v in graph.edge_clusters.items()} if graph.edge_clusters else None
        }
        with open(output_path, "w") as f:
            json.dump(graph_dict, f, indent=2)