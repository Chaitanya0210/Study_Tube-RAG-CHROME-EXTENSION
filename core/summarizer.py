"""Map-Reduce summarizer for long video transcripts."""

import logging
from typing import Generator, List, Optional

from core.llm import BaseLLM, get_llm
from core.prompts import build_map_summary_prompt, build_reduce_summary_prompt
from core.splitter import Chunk

logger = logging.getLogger(__name__)


class VideoSummarizer:
    """Performs hierarchical map-reduce summarization across transcript chunks."""

    def __init__(self, llm: Optional[BaseLLM] = None, batch_size: int = 6):
        self.llm = llm or get_llm()
        self.batch_size = batch_size

    def summarize(self, chunks: List[Chunk]) -> str:
        """Run full map-reduce pipeline synchronously and return final summary."""
        if not chunks:
            return "No transcript content available to summarize."

        if len(chunks) <= self.batch_size:
            # Short video: single reduce pass
            combined_text = "\n".join(f"[{c.timestamp}] {c.text}" for c in chunks)
            prompt = f"Please summarize this video transcript thoroughly, citing timestamps:\n\n{combined_text}"
            return self.llm.generate(prompt)

        # 1. MAP STEP: Summarize batches
        section_summaries: List[str] = []
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i : i + self.batch_size]
            start_time = batch[0].timestamp
            end_time = batch[-1].timestamp
            batch_text = "\n".join(f"[{c.timestamp}] {c.text}" for c in batch)

            map_prompt = build_map_summary_prompt(batch_text, start_time, end_time)
            logger.info("Summarizing batch %d to %d (timestamps %s - %s)", i, i + len(batch), start_time, end_time)
            summary_part = self.llm.generate(map_prompt)
            section_summaries.append(f"### Segment {start_time} - {end_time}\n{summary_part}")

        # 2. REDUCE STEP: Synthesize all section summaries
        reduce_prompt = build_reduce_summary_prompt(section_summaries)
        logger.info("Executing reduce synthesis over %d segment summaries", len(section_summaries))
        final_summary = self.llm.generate(reduce_prompt)
        return final_summary

    def summarize_stream(self, chunks: List[Chunk]) -> Generator[str, None, None]:
        """Run map-reduce and stream the final reduction into the UI."""
        if not chunks:
            yield "No transcript content available to summarize."
            return

        if len(chunks) <= self.batch_size:
            combined_text = "\n".join(f"[{c.timestamp}] {c.text}" for c in chunks)
            prompt = f"Please summarize this video transcript thoroughly, citing timestamps:\n\n{combined_text}"
            yield from self.llm.generate_stream(prompt)
            return

        # Map step
        section_summaries: List[str] = []
        num_batches = (len(chunks) + self.batch_size - 1) // self.batch_size
        yield f"⏳ Analyzing long video across {num_batches} sections...\n\n"

        for idx, i in enumerate(range(0, len(chunks), self.batch_size), 1):
            batch = chunks[i : i + self.batch_size]
            start_time = batch[0].timestamp
            end_time = batch[-1].timestamp
            batch_text = "\n".join(f"[{c.timestamp}] {c.text}" for c in batch)

            map_prompt = build_map_summary_prompt(batch_text, start_time, end_time)
            summary_part = self.llm.generate(map_prompt)
            section_summaries.append(f"### Segment {start_time} - {end_time}\n{summary_part}")

        # Reduce step (streamed)
        yield "🎯 Synthesizing final comprehensive study summary...\n\n"
        reduce_prompt = build_reduce_summary_prompt(section_summaries)
        yield from self.llm.generate_stream(reduce_prompt)
