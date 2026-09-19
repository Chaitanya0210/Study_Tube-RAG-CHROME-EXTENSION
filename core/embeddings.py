"""Local Sentence-Transformers embedding wrapper with caching and model configuration."""

import logging
from functools import lru_cache
from typing import List, Optional

import numpy as np

from core.config import get_config

logger = logging.getLogger(__name__)

# Forward reference / type hinting without hard import at load time
_MODEL_INSTANCE = None


class EmbeddingModel:
    """Wrapper around SentenceTransformer with caching and normalization."""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or get_config().embedding_model
        logger.info("Initializing SentenceTransformer embedding model: %s", self.model_name)
        
        # Lazy import to keep module import times snappy
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(self.model_name)

    def _prepare_query_text(self, text: str) -> str:
        """Add prompt prefix for models that require it (like e5 or bge)."""
        lower = self.model_name.lower()
        if "e5" in lower and not text.startswith("query: "):
            return f"query: {text}"
        return text

    def _prepare_doc_text(self, text: str) -> str:
        """Add passage prefix for models that require it (like e5)."""
        lower = self.model_name.lower()
        if "e5" in lower and not text.startswith("passage: "):
            return f"passage: {text}"
        return text

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute normalized vector embeddings for a list of document chunks."""
        if not texts:
            return []
        prepared = [self._prepare_doc_text(t) for t in texts]
        embeddings = self._model.encode(
            prepared,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Compute normalized vector embedding for a search query."""
        prepared = self._prepare_query_text(text)
        embedding = self._model.encode(
            prepared,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return embedding.tolist()


@lru_cache(maxsize=4)
def get_embedding_model(model_name: Optional[str] = None) -> EmbeddingModel:
    """Singleton getter for the embedding model, cached per model name."""
    name = model_name or get_config().embedding_model
    return EmbeddingModel(model_name=name)
