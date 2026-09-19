"""Prompt templates for RAG chat, depth modes, summarization, and quiz generation."""

from typing import Any, Dict, List, Optional
from core.splitter import Chunk

SYSTEM_INSTRUCTION = """You are StudyTube, an expert AI study assistant embedded in YouTube.
Your mission is to help the user learn and deeply understand the video content.

Core Rules:
1. Ground your answers primarily in the provided video transcript context.
2. ALWAYS cite timestamps in the format [mm:ss] or [hh:mm:ss] when referencing video moments so the user can jump directly to that point in the player. Example: "The instructor introduces the concept at [03:15]."
3. If the answer is not mentioned in or cannot be deduced from the video transcript, state clearly: "This topic is not covered in the video."
4. When the user asks for more depth, or when the depth level is set to 'In-depth', explain the video's content first, then add a clearly labeled section titled:
### Beyond the video
In this section, provide additional real-world context, technical depth, or related concepts that enrich the user's understanding.
5. Tone and clarity: Be encouraging, pedagogical, and precise.
"""

DEPTH_GUIDELINES = {
    "Simple": (
        "EXPLANATION LEVEL: SIMPLE\n"
        "- Explain like the user is a curious beginner.\n"
        "- Use intuitive real-world analogies and clear, conversational language.\n"
        "- Keep explanations concise and avoid dense jargon unless immediately defined."
    ),
    "Standard": (
        "EXPLANATION LEVEL: STANDARD\n"
        "- Provide a balanced, structured, university-level study response.\n"
        "- Use clear headings and bullet points where helpful.\n"
        "- Define key terms and clearly explain mechanisms."
    ),
    "In-depth": (
        "EXPLANATION LEVEL: IN-DEPTH\n"
        "- Deliver an exhaustive, technically rigorous breakdown.\n"
        "- Detail the underlying principles, edge cases, formulas/code (if applicable), and theoretical context.\n"
        "- Include the '### Beyond the video' section to expand beyond what the speaker covered."
    ),
}


def build_rag_prompt(
    question: str,
    retrieved_chunks: List[Any],
    chat_history: Optional[List[Dict[str, str]]] = None,
    depth: str = "Standard",
) -> str:
    """Construct full prompt with context, history, depth instruction, and user query."""
    depth_instruction = DEPTH_GUIDELINES.get(depth, DEPTH_GUIDELINES["Standard"])

    # Format context chunks with timestamps
    context_blocks = []
    for idx, item in enumerate(retrieved_chunks, 1):
        # Handle both RetrievedChunk or raw Chunk
        chunk = getattr(item, "chunk", item)
        score = getattr(item, "score", None)
        score_str = f" (relevance: {score:.2f})" if score is not None else ""
        context_blocks.append(
            f"--- Context Chunk {idx} [Timestamp: {chunk.timestamp}]{score_str} ---\n{chunk.text}"
        )

    context_str = "\n\n".join(context_blocks) if context_blocks else "No relevant context found."

    # Format recent chat history (last 4-6 turns)
    history_str = ""
    if chat_history:
        recent = chat_history[-6:]
        turns = []
        for msg in recent:
            role = "Student" if msg.get("role") == "user" else "Assistant"
            turns.append(f"{role}: {msg.get('content', '')}")
        history_str = "Recent Conversation History:\n" + "\n".join(turns) + "\n\n"

    prompt = f"""{depth_instruction}

Video Transcript Context:
{context_str}

{history_str}Current Student Question: {question}

Please answer the student's question adhering strictly to the Core Rules and cited timestamps."""
    return prompt


def build_key_concepts_prompt(chunks: List[Chunk]) -> str:
    """Construct prompt for extracting key concepts with timestamp references."""
    context_blocks = [
        f"[{c.timestamp}] {c.text}"
        for c in chunks[:15]  # Sample representative chunks
    ]
    context_str = "\n\n".join(context_blocks)

    return f"""From the following video transcript excerpts, extract the 5 to 7 most important Core Concepts discussed in the video.

Transcript excerpts:
{context_str}

Format your output clearly with:
- **Concept Name**: Definition and explanation grounded in the video. Include the earliest timestamp where it is introduced (e.g. `[02:14]`).
- Key takeaway or formula/principle.
"""


def build_quiz_prompt(chunks: List[Chunk]) -> str:
    """Construct prompt to generate a 5-question multiple-choice quiz with answers."""
    context_blocks = [
        f"[{c.timestamp}] {c.text}"
        for c in chunks[:20]
    ]
    context_str = "\n\n".join(context_blocks)

    return f"""Based ONLY on the following video transcript snippets, generate a high-quality 5-question multiple choice quiz to test student comprehension.

Transcript snippets:
{context_str}

Output the quiz strictly in valid JSON format matching this exact schema:
[
  {{
    "question": "Question text here?",
    "options": {{
      "A": "Option text",
      "B": "Option text",
      "C": "Option text",
      "D": "Option text"
    }},
    "correct_answer": "A",
    "explanation": "Explanation of why this is correct, citing the timestamp [mm:ss] from the video."
  }}
]
Return ONLY the raw JSON array. Do not wrap in markdown backticks or commentary."""


def build_map_summary_prompt(chunk_batch_text: str, start_time: str, end_time: str) -> str:
    """Map prompt: summarize a consecutive section of video chunks."""
    return f"""Summarize the following transcript segment (spanning {start_time} to {end_time}) into 2-3 concise bullet points highlighting key points and timestamped events:

Transcript:
{chunk_batch_text}

Summary bullet points:"""


def build_reduce_summary_prompt(section_summaries: List[str]) -> str:
    """Reduce prompt: combine section summaries into a comprehensive study summary."""
    combined = "\n".join(section_summaries)
    return f"""You are summarizing a full educational video. Synthesize these chronological section notes into a cohesive, structured study summary:

Section notes:
{combined}

Structure your response with:
1. **Executive Overview**: 2-3 sentences summarizing the main thesis/goal of the video.
2. **Timeline Breakdown**: Chronological section milestones with timestamps.
3. **Core Takeaways**: 3-5 high-impact bullet points for quick review.
"""
