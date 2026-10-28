"""[unh_v1] OpenAI-compatible /v1/embeddings client (e.g. a vLLM-served voyageai/voyage-4-nano).

Ported from HippoRAG 2 (OSU-NLP-Group/HippoRAG @ 1438aba, src/hipporag/embedding_model/VLLM.py),
which MemGraphRAG is forked from. Upstream MemGraphRAG only supports local HF checkpoints.

Select it with an embedding model name that starts with "VLLM/", e.g.
    --embedding-model VLLM/voyageai/voyage-4-nano --embedding-base-url http://localhost:8001/v1/embeddings
Like HippoRAG's version, the `instruction` kwarg is accepted and ignored.
"""
from typing import List

import numpy as np
import requests
from tqdm import tqdm

from ..utils.config_utils import BaseConfig
from .base import BaseEmbeddingModel


class VLLMEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, global_config: BaseConfig, embedding_model_name: str) -> None:
        super().__init__(global_config=global_config)

        self.model_id = embedding_model_name.removeprefix("VLLM/")
        self.batch_size = global_config.embedding_batch_size
        self.base_url = global_config.embedding_base_url
        if not self.base_url:
            raise ValueError("embedding_base_url is required for VLLM/ embedding models")
        self.timeout = global_config.embedding_request_timeout
        self.embedding_dim = None

    def encode(self, texts: List[str]) -> np.ndarray:
        # [unh_v1] vLLM returns 400 for "" (a few extracted entities are empty); embed them as " " instead
        texts = [text if text else " " for text in texts]
        payload = {"model": self.model_id, "input": texts}
        response = requests.post(self.base_url, json=payload, timeout=self.timeout)
        response.raise_for_status()
        rows = sorted(response.json()["data"], key=lambda row: row.get("index", 0))
        return np.array([row["embedding"] for row in rows], dtype=np.float32)

    def batch_encode(self, texts: List[str], **kwargs) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        if len(texts) <= self.batch_size:
            results = self.encode(texts)
        else:
            chunks = []
            for i in tqdm(range(0, len(texts), self.batch_size), desc="Batch Encoding"):
                chunks.append(self.encode(texts[i:i + self.batch_size]))
            results = np.concatenate(chunks)
        if self.embedding_dim is None and len(results):
            self.embedding_dim = results.shape[1]
        if self.global_config.embedding_return_as_normalized:
            results = results / np.linalg.norm(results, axis=1, keepdims=True)
        return results
