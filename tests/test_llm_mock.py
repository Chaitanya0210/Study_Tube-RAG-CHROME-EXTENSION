"""Unit tests for prompt formatting and mocked LLM / summarizer execution."""

import json
from typing import Generator, List, Optional
import pytest

from core.llm import BaseLLM
from core.prompts import (
    build_key_concepts_prompt,
    build_quiz_prompt,
    build_rag_prompt,
)
from core.splitter import Chunk
from core.summarizer import VideoSummarizer


class MockLLM(BaseLLM):
    """Mock LLM that returns predictable responses for testing."""

    def __init__(self, response_text: str = "Mock response with timestamp [01:23]"):
        self.response_text = response_text
        self.recorded_prompts: List[str] = []

    def generate(self, prompt: str, system_instruction: Optional[str] = None) -> str:
        self.recorded_prompts.append(prompt)
        return self.response_text

    def generate_stream(
        self, prompt: str, system_instruction: Optional[str] = None
    ) -> Generator[str, None, None]:
        self.recorded_prompts.append(prompt)
        words = self.response_text.split()
        for word in words:
            yield word + " "


def test_build_rag_prompt_depth_variations():
    chunks = [
        Chunk(
            chunk_id="v1_0",
            video_id="v1",
            text="Photosynthesis converts light into chemical energy.",
            start_seconds=60.0,
            end_seconds=75.0,
            timestamp="01:00",
        )
    ]

    # Test Simple depth
    simple_prompt = build_rag_prompt("How does photosynthesis work?", chunks, depth="Simple")
    assert "EXPLANATION LEVEL: SIMPLE" in simple_prompt
    assert "[Timestamp: 01:00]" in simple_prompt
    assert "Photosynthesis converts light" in simple_prompt

    # Test In-depth depth
    in_depth_prompt = build_rag_prompt("How does photosynthesis work?", chunks, depth="In-depth")
    assert "EXPLANATION LEVEL: IN-DEPTH" in in_depth_prompt
    assert "Beyond the video" in in_depth_prompt


def test_build_quiz_prompt_schema():
    chunks = [
        Chunk(
            chunk_id="v1_0",
            video_id="v1",
            text="The speed of light in vacuum is approximately 300,000 km per second.",
            start_seconds=120.0,
            end_seconds=135.0,
            timestamp="02:00",
        )
    ]
    quiz_prompt = build_quiz_prompt(chunks)
    assert "5-question multiple choice quiz" in quiz_prompt
    assert "schema" in quiz_prompt
    assert "[02:00]" in quiz_prompt


def test_mock_llm_generation():
    llm = MockLLM("The answer is found at [05:42].")
    chunks = [
        Chunk(
            chunk_id="v1_0",
            video_id="v1",
            text="Neural networks use backpropagation.",
            start_seconds=342.0,
            end_seconds=360.0,
            timestamp="05:42",
        )
    ]
    prompt = build_rag_prompt("What does the video say about neural networks?", chunks)
    response = llm.generate(prompt)
    assert "[05:42]" in response
    assert len(llm.recorded_prompts) == 1

    # Streaming test
    stream = list(llm.generate_stream(prompt))
    assert "".join(stream).strip() == "The answer is found at [05:42]."


def test_summarizer_map_reduce_with_mock():
    mock_llm = MockLLM("Summary segment note.")
    summarizer = VideoSummarizer(llm=mock_llm, batch_size=2)

    # 5 chunks with batch_size=2 results in 3 map calls and 1 reduce call = 4 LLM calls
    chunks = [
        Chunk(
            chunk_id=f"v_{i}",
            video_id="v",
            text=f"Transcript content part {i}.",
            start_seconds=float(i * 30),
            end_seconds=float((i + 1) * 30),
            timestamp=f"0{i}:00",
        )
        for i in range(5)
    ]

    final_summary = summarizer.summarize(chunks)
    assert final_summary == "Summary segment note."
    # 3 map calls + 1 reduce call = 4 total calls to LLM
    assert len(mock_llm.recorded_prompts) == 4
