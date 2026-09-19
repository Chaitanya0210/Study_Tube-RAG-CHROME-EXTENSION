"""Tests for transcript parsing, video ID extraction, and timestamped chunk splitting."""

import pytest

from core.splitter import Chunk, format_timestamp, split_transcript
from core.transcript import TranscriptResult, TranscriptSegment, extract_video_id


def test_format_timestamp():
    assert format_timestamp(0) == "00:00"
    assert format_timestamp(45) == "00:45"
    assert format_timestamp(65) == "01:05"
    assert format_timestamp(765) == "12:45"
    assert format_timestamp(3600) == "01:00:00"
    assert format_timestamp(3665) == "01:01:05"
    assert format_timestamp(7325) == "02:02:05"


def test_extract_video_id():
    # Standard watch URL
    assert extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    # With additional parameters
    assert extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=120s&ab_channel=RickAstley") == "dQw4w9WgXcQ"
    # Short youtu.be URL
    assert extract_video_id("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert extract_video_id("https://youtu.be/dQw4w9WgXcQ?t=42") == "dQw4w9WgXcQ"
    # Embed URL
    assert extract_video_id("https://www.youtube.com/embed/dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    # Shorts URL
    assert extract_video_id("https://www.youtube.com/shorts/dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    # Bare ID
    assert extract_video_id("dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    # Invalid strings
    assert extract_video_id("https://google.com") is None
    assert extract_video_id("") is None
    assert extract_video_id("not_a_valid_id") is None


def test_split_transcript_basic():
    segments = [
        TranscriptSegment(text="Welcome to the lecture on quantum physics.", start=0.0, duration=4.5),
        TranscriptSegment(text="Today we will discuss wave-particle duality and the double-slit experiment.", start=4.5, duration=5.5),
        TranscriptSegment(text="Light behaves both as continuous waves and discrete particles called photons.", start=10.0, duration=6.0),
        TranscriptSegment(text="This was first demonstrated by Thomas Young in his famous 1801 experiment.", start=16.0, duration=5.0),
        TranscriptSegment(text="Later, Albert Einstein explained the photoelectric effect using Planck's quantum hypothesis.", start=21.0, duration=7.0),
    ]
    transcript = TranscriptResult(
        video_id="test1234567",
        language="en",
        is_generated=False,
        segments=segments,
    )

    # Use small chunk size to force multiple chunks
    chunks = split_transcript(transcript, chunk_size=120, chunk_overlap=30)
    assert len(chunks) > 1

    # Verify first chunk
    first_chunk = chunks[0]
    assert first_chunk.video_id == "test1234567"
    assert first_chunk.start_seconds == 0.0
    assert first_chunk.timestamp == "00:00"
    assert "Welcome to the lecture" in first_chunk.text

    # Verify all chunks have valid start/end timestamps and metadata
    for chunk in chunks:
        assert chunk.start_seconds >= 0.0
        assert chunk.end_seconds >= chunk.start_seconds
        assert chunk.chunk_id.startswith("test1234567_")
        assert ":" in chunk.timestamp
        meta = chunk.to_metadata()
        assert meta["video_id"] == "test1234567"
        assert meta["timestamp"] == chunk.timestamp


def test_split_transcript_empty():
    transcript = TranscriptResult(
        video_id="empty123456",
        language="en",
        is_generated=False,
        segments=[],
    )
    chunks = split_transcript(transcript)
    assert chunks == []
