"""ChromaDB vector store manager with persistent storage and video cache checks."""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings

from core.config import get_config
from core.embeddings import EmbeddingModel, get_embedding_model
from core.splitter import Chunk

logger = logging.getLogger(__name__)


def sanitize_collection_name(video_id: str) -> str:
    """Format video_id into a valid Chroma collection name (3-63 chars, [a-zA-Z0-9._-])."""
    # Replace non-alphanumeric/underscore/dash with underscore
    cleaned = re.sub(r"[^a-zA-Z0-9._-]", "_", video_id)
    name = f"yt_{cleaned}"
    # Chroma requires name length between 3 and 63
    if len(name) > 63:
        name = name[:63]
    return name


class VectorStoreManager:
    """Manages persistent ChromaDB collections per YouTube video."""

    def __init__(self, persist_directory: Optional[str] = None):
        cfg = get_config()
        self.persist_dir = Path(persist_directory or cfg.chroma_dir).resolve()
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Initializing ChromaDB PersistentClient at %s", self.persist_dir)
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )

    def get_collection_name(self, video_id: str) -> str:
        """Return the sanitized collection name for a video."""
        return sanitize_collection_name(video_id)

    def is_video_indexed(self, video_id: str) -> bool:
        """Check whether the video is already indexed with stored chunks."""
        col_name = self.get_collection_name(video_id)
        try:
            col = self.client.get_collection(name=col_name)
            count = col.count()
            logger.info("Collection '%s' exists with %d documents", col_name, count)
            return count > 0
        except Exception:
            return False

    def index_chunks(
        self,
        video_id: str,
        chunks: List[Chunk],
        embedding_model: Optional[EmbeddingModel] = None,
        force: bool = False,
    ) -> int:
        """Index a list of chunks into the video's Chroma collection.
        
        Skips indexing if already indexed and force is False.
        Returns the number of documents in the collection.
        """
        if not chunks:
            logger.warning("No chunks provided for video %s", video_id)
            return 0

        col_name = self.get_collection_name(video_id)

        if not force and self.is_video_indexed(video_id):
            logger.info("Video %s is already indexed in collection %s. Skipping.", video_id, col_name)
            col = self.client.get_collection(name=col_name)
            return col.count()

        # If forced and collection exists, reset it
        try:
            self.client.delete_collection(name=col_name)
            logger.info("Deleted existing collection %s for fresh re-indexing", col_name)
        except Exception:
            pass

        collection = self.client.create_collection(
            name=col_name,
            metadata={"video_id": video_id, "hnsw:space": "cosine"},
        )

        model = embedding_model or get_embedding_model()
        doc_texts = [c.text for c in chunks]
        embeddings = model.embed_documents(doc_texts)
        ids = [c.chunk_id for c in chunks]
        metadatas = [c.to_metadata() for c in chunks]

        # Chroma recommends adding in batches if large
        batch_size = 200
        for i in range(0, len(chunks), batch_size):
            end = i + batch_size
            collection.add(
                ids=ids[i:end],
                documents=doc_texts[i:end],
                embeddings=embeddings[i:end],
                metadatas=metadatas[i:end],
            )

        logger.info("Successfully indexed %d chunks for video %s", len(chunks), video_id)
        return collection.count()

    def get_all_chunks(self, video_id: str) -> List[Chunk]:
        """Retrieve all stored chunks for a video, ordered by start time."""
        col_name = self.get_collection_name(video_id)
        try:
            col = self.client.get_collection(name=col_name)
        except Exception:
            return []

        results = col.get(include=["documents", "metadatas"])
        docs = results.get("documents") or []
        metas = results.get("metadatas") or []
        ids = results.get("ids") or []

        chunks: List[Chunk] = []
        for cid, doc, meta in zip(ids, docs, metas):
            chunks.append(
                Chunk(
                    chunk_id=cid,
                    video_id=meta.get("video_id", video_id),
                    text=doc,
                    start_seconds=float(meta.get("start_seconds", 0.0)),
                    end_seconds=float(meta.get("end_seconds", 0.0)),
                    timestamp=meta.get("timestamp", "00:00"),
                    metadata=meta,
                )
            )

        # Sort chronologically by start timestamp
        chunks.sort(key=lambda c: c.start_seconds)
        return chunks

    def delete_video_index(self, video_id: str) -> None:
        """Delete collection for a specific video."""
        col_name = self.get_collection_name(video_id)
        try:
            self.client.delete_collection(name=col_name)
            logger.info("Deleted collection %s", col_name)
        except Exception as e:
            logger.debug("Failed or collection does not exist: %s", e)
