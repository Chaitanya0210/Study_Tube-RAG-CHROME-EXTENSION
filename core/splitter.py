"""Transcript chunker with character offset tracking and timestamp mapping."""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.transcript import TranscriptResult, TranscriptSegment

logger = logging.getLogger(__name__)


def format_timestamp(seconds: float) -> str:
    """Format seconds into 'mm:ss' or 'hh:mm:ss' string."""
    total_seconds = max(0, int(seconds))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


@dataclass
class Chunk:
    """A semantic text chunk with timing metadata."""
    chunk_id: str
    video_id: str
    text: str
    start_seconds: float
    end_seconds: float
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_metadata(self) -> Dict[str, Any]:
        """Convert timing and video details into a Chroma-compatible metadata dict."""
        meta = {
            "video_id": self.video_id,
            "chunk_id": self.chunk_id,
            "start_seconds": float(self.start_seconds),
            "end_seconds": float(self.end_seconds),
            "timestamp": self.timestamp,
        }
        meta.update(self.metadata)
        return meta


def split_transcript(
    transcript: TranscriptResult,
    chunk_size: int = 900,
    chunk_overlap: int = 150,
) -> List[Chunk]:
    """Split transcript segments into overlapping chunks with precise start/end timestamps."""
    if not transcript.segments:
        return []

    # Build continuous text with segment span tracking
    full_text_parts: List[str] = []
    # (start_char_idx, end_char_idx, start_sec, end_sec)
    spans: List[tuple[int, int, float, float]] = []
    current_char = 0

    for seg in transcript.segments:
        text = seg.text.strip()
        if not text:
            continue
        full_text_parts.append(text)
        start_char = current_char
        end_char = current_char + len(text)
        spans.append((start_char, end_char, seg.start, seg.end))
        current_char = end_char + 1  # accounting for space separator

    full_text = " ".join(full_text_parts)
    if not full_text:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
    )
    raw_chunks = splitter.split_text(full_text)

    chunks: List[Chunk] = []
    search_cursor = 0

    for idx, raw_chunk in enumerate(raw_chunks):
        chunk_text = raw_chunk.strip()
        if not chunk_text:
            continue

        # Find position of chunk in full_text
        # Search backwards slightly to handle overlap safely
        start_search = max(0, search_cursor - (chunk_overlap * 2))
        pos = full_text.find(chunk_text, start_search)
        if pos == -1:
            # Fallback: search anywhere after start_search
            pos = full_text.find(chunk_text[:30], start_search)
            if pos == -1:
                pos = start_search

        chunk_start_char = pos
        chunk_end_char = pos + len(chunk_text)
        search_cursor = pos

        # Find overlapping segments
        overlapping_spans = [
            span for span in spans
            if span[1] >= chunk_start_char and span[0] <= chunk_end_char
        ]

        if overlapping_spans:
            start_sec = overlapping_spans[0][2]
            end_sec = overlapping_spans[-1][3]
        else:
            # Fallback to closest segment
            start_sec = transcript.segments[0].start
            end_sec = transcript.segments[-1].end

        formatted = format_timestamp(start_sec)
        chunk_id = f"{transcript.video_id}_{idx}"

        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                video_id=transcript.video_id,
                text=chunk_text,
                start_seconds=start_sec,
                end_seconds=end_sec,
                timestamp=formatted,
                metadata={
                    "language": transcript.language,
                    "is_generated": transcript.is_generated,
                },
            )
        )

    logger.info("Split transcript for %s into %d chunks", transcript.video_id, len(chunks))
    return chunks
