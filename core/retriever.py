"""Vector similarity retriever with Maximal Marginal Relevance (MMR) re-ranking."""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from core.config import get_config
from core.embeddings import EmbeddingModel, get_embedding_model
from core.splitter import Chunk
from core.vectorstore import VectorStoreManager

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    """A retrieved chunk paired with its similarity score."""
    chunk: Chunk
    score: float  # Cosine similarity (higher is more similar, typically 0 to 1)


def compute_mmr(
    query_embedding: np.ndarray,
    doc_embeddings: np.ndarray,
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> List[int]:
    """Select top_k diverse documents using Maximal Marginal Relevance (MMR).
    
    Formula:
    argmax_{d in U} [ lambda * Sim(d, q) - (1 - lambda) * max_{s in S} Sim(d, s) ]
    
    query_embedding: normalized 1D vector (dim,)
    doc_embeddings: normalized 2D array (n_docs, dim)
    """
    n_docs = len(doc_embeddings)
    if n_docs <= top_k:
        # Return all indices sorted by query similarity
        sims = np.dot(doc_embeddings, query_embedding)
        return list(np.argsort(sims)[::-1])

    # Precalculate similarities to query: shape (n_docs,)
    query_sims = np.dot(doc_embeddings, query_embedding)

    # Precalculate pairwise similarities among docs: shape (n_docs, n_docs)
    doc_sims = np.dot(doc_embeddings, doc_embeddings.T)

    selected: List[int] = []
    unselected = set(range(n_docs))

    # Pick the top similarity document first
    first_pick = int(np.argmax(query_sims))
    selected.append(first_pick)
    unselected.remove(first_pick)

    # Iteratively pick remaining documents
    while len(selected) < top_k and unselected:
        best_idx = -1
        best_score = -float("inf")

        for idx in unselected:
            sim_to_query = query_sims[idx]
            # Max similarity to any already-selected document
            max_sim_to_selected = max(doc_sims[idx, s] for s in selected)
            mmr_score = (lambda_param * sim_to_query) - ((1.0 - lambda_param) * max_sim_to_selected)

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        if best_idx != -1:
            selected.append(best_idx)
            unselected.remove(best_idx)
        else:
            break

    return selected


class Retriever:
    """Retrieves relevant transcript chunks for a video using MMR similarity search."""

    def __init__(
        self,
        vector_store: Optional[VectorStoreManager] = None,
        embedding_model: Optional[EmbeddingModel] = None,
    ):
        self.vector_store = vector_store or VectorStoreManager()
        self.embedding_model = embedding_model or get_embedding_model()

    def retrieve(
        self,
        video_id: str,
        query: str,
        top_k: Optional[int] = None,
        mmr_lambda: Optional[float] = None,
    ) -> List[RetrievedChunk]:
        """Search the video collection for relevant chunks using MMR."""
        cfg = get_config()
        k = top_k or cfg.top_k
        lam = mmr_lambda if mmr_lambda is not None else cfg.mmr_lambda

        col_name = self.vector_store.get_collection_name(video_id)
        try:
            col = self.vector_store.client.get_collection(name=col_name)
        except Exception as e:
            logger.warning("Collection for video %s not found: %s", video_id, e)
            return []

        # Number of candidates to fetch before MMR re-ranking
        candidate_k = min(max(k * 3, 10), max(1, col.count()))
        if candidate_k == 0:
            return []

        query_vec = self.embedding_model.embed_query(query)

        # Query Chroma with embeddings included
        results = col.query(
            query_embeddings=[query_vec],
            n_results=candidate_k,
            include=["documents", "metadatas", "embeddings", "distances"],
        )

        docs = results["documents"][0] if results.get("documents") else []
        metas = results["metadatas"][0] if results.get("metadatas") else []
        ids = results["ids"][0] if results.get("ids") else []
        embeddings = results["embeddings"][0] if results.get("embeddings") else []
        distances = results["distances"][0] if results.get("distances") else []

        if not docs:
            return []

        # Convert embeddings to numpy array
        doc_embeddings = np.array(embeddings, dtype=np.float32)
        q_vec = np.array(query_vec, dtype=np.float32)

        # Apply MMR
        selected_indices = compute_mmr(
            query_embedding=q_vec,
            doc_embeddings=doc_embeddings,
            top_k=min(k, len(docs)),
            lambda_param=lam,
        )

        retrieved: List[RetrievedChunk] = []
        for idx in selected_indices:
            meta = metas[idx]
            # Convert cosine distance to cosine similarity (since cosine distance = 1 - sim)
            dist = distances[idx] if idx < len(distances) else 0.0
            similarity = max(0.0, min(1.0, 1.0 - dist))

            chunk = Chunk(
                chunk_id=ids[idx],
                video_id=meta.get("video_id", video_id),
                text=docs[idx],
                start_seconds=float(meta.get("start_seconds", 0.0)),
                end_seconds=float(meta.get("end_seconds", 0.0)),
                timestamp=meta.get("timestamp", "00:00"),
                metadata=meta,
            )
            retrieved.append(RetrievedChunk(chunk=chunk, score=round(float(similarity), 4)))

        logger.info("Retrieved %d MMR-ranked chunks for video %s", len(retrieved), video_id)
        return retrieved
