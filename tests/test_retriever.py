"""Tests for vector store indexing, MMR algorithm, and retriever logic."""

import numpy as np
import pytest

from core.retriever import Retriever, compute_mmr
from core.splitter import Chunk
from core.vectorstore import VectorStoreManager, sanitize_collection_name


class MockEmbeddingModel:
    """Fast, deterministic embedding mock for unit tests."""

    def __init__(self, dim: int = 4):
        self.dim = dim

    def embed_documents(self, texts):
        res = []
        for i, t in enumerate(texts):
            # Create synthetic unit vector based on length and index
            vec = np.zeros(self.dim, dtype=np.float32)
            vec[i % self.dim] = 1.0
            res.append(vec.tolist())
        return res

    def embed_query(self, text):
        vec = np.zeros(self.dim, dtype=np.float32)
        vec[0] = 1.0  # Points to dim 0
        return vec.tolist()


def test_sanitize_collection_name():
    assert sanitize_collection_name("dQw4w9WgXcQ") == "yt_dQw4w9WgXcQ"
    assert sanitize_collection_name("vid-123_abc") == "yt_vid-123_abc"
    # Special chars replaced
    assert sanitize_collection_name("vid@#$123") == "yt_vid___123"


def test_compute_mmr_diversity():
    # Query vector along dimension 0
    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    # Candidate docs (all unit length):
    # Doc 0: identical to query (sim to query = 1.0)
    # Doc 1: query sim = 0.90, but very redundant with doc 0 (sim to doc 0 = 0.95)
    # Doc 2: query sim = 0.80, but diverse from doc 0 (sim to doc 0 = 0.40)
    docs = np.array([
        [1.0, 0.0, 0.0],         # doc 0: sim to query = 1.0, sim to doc 0 = 1.0
        [0.90, 0.4358899, 0.0],  # doc 1: sim to query = 0.90, sim to doc 0 = 0.90
        [0.80, 0.10, 0.5916],    # doc 2: sim to query = 0.80, sim to doc 0 = 0.80
    ], dtype=np.float32)

    # With lambda=0.3 (favoring diversity):
    # doc 0 selected first.
    # doc 1 score = 0.3 * 0.90 - 0.7 * 0.90 = -0.36
    # If doc 3 has sim to query = 0.70, and sim to doc 0 = 0.10:
    # doc 3 score = 0.3 * 0.70 - 0.7 * 0.10 = 0.21 - 0.07 = +0.14
    docs = np.array([
        [1.0, 0.0, 0.0],        # doc 0: identical to query
        [0.92, 0.39, 0.0],       # doc 1: sim to doc 0 is ~0.92
        [0.60, 0.80, 0.0],       # doc 2: sim to query is 0.60, sim to doc 0 is 0.60
        [0.70, 0.0, 0.7141],     # doc 3: sim to query is 0.70, sim to doc 0 is 0.70
    ], dtype=np.float32)
    docs = docs / np.linalg.norm(docs, axis=1, keepdims=True)

    # Let's test with explicit vectors where doc 2 is diverse
    v_query = np.array([1.0, 0.0], dtype=np.float32)
    v_docs = np.array([
        [1.0, 0.0],          # doc 0: sim to query = 1.0, sim to doc 0 = 1.0
        [0.95, 0.31225],     # doc 1: sim to query = 0.95, sim to doc 0 = 0.95
        [0.70, 0.71414],     # doc 2: sim to query = 0.70, sim to doc 0 = 0.70
    ], dtype=np.float32)
    v_docs = v_docs / np.linalg.norm(v_docs, axis=1, keepdims=True)

    # With lambda=0.3:
    # Pick 0: doc 0 (highest sim)
    # Doc 1 score: 0.3 * 0.95 - 0.7 * 0.95 = -0.38
    # Doc 2 score: 0.3 * 0.70 - 0.7 * 0.70 = -0.28
    # Since -0.28 > -0.38, doc 2 is selected as the second pick!
    selected = compute_mmr(v_query, v_docs, top_k=2, lambda_param=0.3)
    assert selected[0] == 0
    assert selected[1] == 2


def test_vector_store_indexing_and_cache(tmp_path):
    manager = VectorStoreManager(persist_directory=str(tmp_path))
    mock_embeddings = MockEmbeddingModel()
    video_id = "test_vid_01"

    assert not manager.is_video_indexed(video_id)

    chunks = [
        Chunk(
            chunk_id=f"{video_id}_0",
            video_id=video_id,
            text="First chunk about Python programming.",
            start_seconds=0.0,
            end_seconds=15.0,
            timestamp="00:00",
        ),
        Chunk(
            chunk_id=f"{video_id}_1",
            video_id=video_id,
            text="Second chunk about machine learning.",
            start_seconds=15.0,
            end_seconds=30.0,
            timestamp="00:15",
        ),
    ]

    count = manager.index_chunks(video_id, chunks, embedding_model=mock_embeddings)
    assert count == 2
    assert manager.is_video_indexed(video_id)

    # Re-indexing with force=False should skip and return cached count
    count2 = manager.index_chunks(video_id, chunks, embedding_model=mock_embeddings, force=False)
    assert count2 == 2

    # Verify retrieval
    retriever = Retriever(vector_store=manager, embedding_model=mock_embeddings)
    results = retriever.retrieve(video_id=video_id, query="Python", top_k=2)

    assert len(results) == 2
    assert results[0].chunk.timestamp in ["00:00", "00:15"]
    assert 0.0 <= results[0].score <= 1.0
