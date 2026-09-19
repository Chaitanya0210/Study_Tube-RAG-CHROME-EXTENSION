"""StudyTube: Streamlit AI Study Assistant for YouTube (Side-Panel Optimized)."""

import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path so 'core' can be imported anywhere
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from core.config import get_config
from core.llm import GeminiLLM, LLMError, get_llm
from core.prompts import (
    SYSTEM_INSTRUCTION,
    build_key_concepts_prompt,
    build_quiz_prompt,
    build_rag_prompt,
)
from core.retriever import Retriever
from core.splitter import format_timestamp, split_transcript
from core.summarizer import VideoSummarizer
from core.transcript import TranscriptError, extract_video_id, fetch_transcript
from core.vectorstore import VectorStoreManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("studytube.app")

# Page Configuration
st.set_page_config(
    page_title="StudyTube",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS for YouTube styling, compact side panel layout, and timestamp badges
st.markdown(
    """
<style>
    /* Remove default Streamlit top padding in side panel */
    .block-container {
        padding-top: 0.8rem;
        padding-bottom: 2rem;
        padding-left: 0.8rem;
        padding-right: 0.8rem;
    }
    /* Compact headers */
    h1, h2, h3, h4 {
        margin-top: 0.1rem;
        margin-bottom: 0.3rem;
    }
    /* Timestamp jump badge */
    a.timestamp-badge {
        display: inline-block;
        background-color: #EBF5FF;
        color: #0066CC !important;
        padding: 2px 7px;
        margin: 0 2px;
        border-radius: 5px;
        font-size: 0.85em;
        font-weight: 600;
        text-decoration: none !important;
        border: 1px solid #BCE0FD;
        cursor: pointer;
        transition: all 0.15s ease-in-out;
    }
    a.timestamp-badge:hover {
        background-color: #0066CC;
        color: #FFFFFF !important;
        border-color: #0066CC;
        transform: translateY(-1px);
    }
    /* Quiz styling */
    .quiz-card {
        background: #F8F9FA;
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 12px;
    }
    /* Status pills */
    .status-pill {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .status-ready { background-color: #DEF7EC; color: #03543F; }
    .status-indexing { background-color: #FEF08A; color: #713F12; }
    .status-error { background-color: #FDE8E8; color: #9B1C1C; }
</style>
""",
    unsafe_allow_html=True,
)

# JavaScript Bridge to send seek message to parent window (Chrome extension side panel)
st.html(
    """
<script>
(function() {
    if (!window._studyTubeBridgeInstalled) {
        window._studyTubeBridgeInstalled = true;
        document.addEventListener('click', function(e) {
            const link = e.target.closest('a[href^="#seek:"]');
            if (link) {
                e.preventDefault();
                const href = link.getAttribute('href');
                const seconds = parseFloat(href.replace('#seek:', ''));
                if (!isNaN(seconds)) {
                    console.log("[StudyTube Bridge] Seeking to seconds:", seconds);
                    window.top.postMessage({ type: 'STUDYTUBE_SEEK', seconds: seconds }, '*');
                }
            }
        }, true);
    }
})();
</script>
"""
)


def timestamp_to_seconds(ts_str: str) -> float:
    """Convert [mm:ss] or [hh:mm:ss] to float seconds."""
    clean = ts_str.strip("[]() ")
    parts = clean.split(":")
    try:
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        elif len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    except Exception:
        pass
    return 0.0


def render_clickable_timestamps(text: str) -> str:
    """Replace raw [mm:ss] or [hh:mm:ss] with clickable seek links."""
    pattern = r"\[(\d{1,2}:\d{2}(?::\d{2})?)\]"

    def replace_match(match: re.Match) -> str:
        ts_str = match.group(1)
        secs = timestamp_to_seconds(ts_str)
        return f'<a href="#seek:{secs}" class="timestamp-badge" title="Jump to {ts_str}">⏱️ {ts_str}</a>'

    return re.sub(pattern, replace_match, text)


@st.cache_resource
def get_vector_store() -> VectorStoreManager:
    return VectorStoreManager()


@st.cache_resource
def get_retriever_instance() -> Retriever:
    return Retriever(vector_store=get_vector_store())


# Initialize components
config = get_config()
vector_store = get_vector_store()
retriever = get_retriever_instance()
llm = get_llm()

# Read video_url from query parameters
query_params = st.query_params
video_url = query_params.get("video_url", "")

# Fallback UI if no video URL is provided in query params
if not video_url:
    st.markdown("### 🎓 StudyTube Assistant")
    st.info("No active YouTube video detected. Please open a YouTube video.")
    video_url = st.text_input(
        "Or paste a YouTube URL to test:",
        placeholder="https://www.youtube.com/watch?v=...",
    )

video_id = extract_video_id(video_url) if video_url else None

if not video_id:
    st.warning("Please open a valid YouTube video to begin studying.")
    st.stop()

# Index Status Handling
is_indexed = vector_store.is_video_indexed(video_id)
chunk_count = 0

if not is_indexed:
    status_placeholder = st.empty()
    status_placeholder.markdown(
        '<span class="status-pill status-indexing">⏳ Indexing transcript...</span>',
        unsafe_allow_html=True,
    )
    try:
        with st.spinner("Extracting transcript & building index..."):
            transcript = fetch_transcript(video_id)
            chunks = split_transcript(
                transcript,
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
            )
            vector_store.index_chunks(video_id, chunks)
            st.session_state[f"chunks_{video_id}"] = chunks
            chunk_count = len(chunks)
            status_placeholder.empty()
            st.rerun()
    except TranscriptError as e:
        status_placeholder.markdown(
            '<span class="status-pill status-error">❌ Index Failed</span>',
            unsafe_allow_html=True,
        )
        st.error(f"Transcript Notice: {str(e)}")
        st.stop()
    except Exception as e:
        status_placeholder.markdown(
            '<span class="status-pill status-error">❌ Error</span>',
            unsafe_allow_html=True,
        )
        st.error(f"Unexpected indexing error: {str(e)}")
        st.stop()
else:
    # Ensure chunks are cached in session state
    if f"chunks_{video_id}" not in st.session_state:
        st.session_state[f"chunks_{video_id}"] = vector_store.get_all_chunks(video_id)
    chunk_count = len(st.session_state[f"chunks_{video_id}"])

all_chunks = st.session_state.get(f"chunks_{video_id}", [])

# Header Bar (Top of Side Panel)
header_col1, header_col2 = st.columns([2, 1])
with header_col1:
    st.markdown(f"**Video:** `{video_id}`")
with header_col2:
    st.markdown(
        f'<span class="status-pill status-ready">✅ Ready ({chunk_count} chunks)</span>',
        unsafe_allow_html=True,
    )

# Study Controls Toolbar (In-line Expander for Side Panel)
with st.expander("⚙️ Study Tools & Explanation Depth", expanded=False):
    depth_choice = st.radio(
        "Explanation Level",
        ["Simple", "Standard", "In-depth"],
        index=1,
        horizontal=True,
        help="Simple: Beginner analogies | Standard: Clear notes | In-depth: Advanced theory & beyond the video",
    )

    btn_col1, btn_col2, btn_col3 = st.columns(3)
    with btn_col1:
        action_summarize = st.button("📋 Summary", use_container_width=True)
    with btn_col2:
        action_concepts = st.button("💡 Concepts", use_container_width=True)
    with btn_col3:
        action_quiz = st.button("📝 Quiz", use_container_width=True)

    if st.button("🔄 Clear Chat History", use_container_width=True):
        st.session_state[f"chat_history_{video_id}"] = []
        st.rerun()

# Ensure chat history exists
chat_key = f"chat_history_{video_id}"
if chat_key not in st.session_state:
    st.session_state[chat_key] = []

# Handle Quick Actions
if action_summarize:
    summarizer = VideoSummarizer(llm=llm)
    with st.chat_message("assistant"):
        placeholder = st.empty()
        stream_chunks = []
        try:
            for text_chunk in summarizer.summarize_stream(all_chunks):
                stream_chunks.append(text_chunk)
                placeholder.markdown(
                    render_clickable_timestamps("".join(stream_chunks)),
                    unsafe_allow_html=True,
                )
            final_text = "".join(stream_chunks)
            st.session_state[chat_key].append({"role": "assistant", "content": final_text})
        except LLMError as e:
            st.error(str(e))

elif action_concepts:
    prompt = build_key_concepts_prompt(all_chunks)
    with st.chat_message("assistant"):
        placeholder = st.empty()
        stream_chunks = []
        try:
            for text_chunk in llm.generate_stream(prompt):
                stream_chunks.append(text_chunk)
                placeholder.markdown(
                    render_clickable_timestamps("".join(stream_chunks)),
                    unsafe_allow_html=True,
                )
            final_text = "".join(stream_chunks)
            st.session_state[chat_key].append({"role": "assistant", "content": final_text})
        except LLMError as e:
            st.error(str(e))

elif action_quiz:
    with st.spinner("Generating 5-question comprehension quiz..."):
        prompt = build_quiz_prompt(all_chunks)
        try:
            raw_quiz = llm.generate(prompt)
            cleaned_json = re.sub(
                r"^```json\s*|^```\s*|```$", "", raw_quiz.strip(), flags=re.MULTILINE
            )
            quiz_data = json.loads(cleaned_json)
            st.session_state[f"quiz_{video_id}"] = quiz_data
            st.session_state[f"quiz_submitted_{video_id}"] = False
        except Exception as e:
            st.error(f"Failed to generate quiz: {str(e)}")

# Render Interactive Quiz if active
quiz_key = f"quiz_{video_id}"
if quiz_key in st.session_state:
    quiz_list = st.session_state[quiz_key]
    is_submitted = st.session_state.get(f"quiz_submitted_{video_id}", False)

    with st.expander("📝 Video Comprehension Quiz", expanded=True):
        user_answers = {}
        for q_idx, item in enumerate(quiz_list, 1):
            st.markdown(f"**Q{q_idx}. {item.get('question')}**")
            options = item.get("options", {})
            choice_keys = list(options.keys())
            formatted_choices = [f"{k}: {options[k]}" for k in choice_keys]

            selected = st.radio(
                f"Select answer for Q{q_idx}:",
                formatted_choices,
                key=f"q_{video_id}_{q_idx}",
                disabled=is_submitted,
            )
            selected_key = selected.split(":")[0] if selected else ""
            user_answers[q_idx] = selected_key

            if is_submitted:
                correct = item.get("correct_answer", "")
                explanation = item.get("explanation", "")
                if selected_key == correct:
                    st.success(f"✅ Correct! ({correct})")
                else:
                    st.error(f"❌ Incorrect. Correct answer: **{correct}**")
                st.markdown(
                    render_clickable_timestamps(f"**Explanation:** {explanation}"),
                    unsafe_allow_html=True,
                )
            st.markdown("---")

        q_col1, q_col2 = st.columns([1, 1])
        with q_col1:
            if not is_submitted and st.button(
                "Submit Quiz", type="primary", use_container_width=True
            ):
                st.session_state[f"quiz_submitted_{video_id}"] = True
                st.rerun()
        with q_col2:
            if st.button("Close Quiz", use_container_width=True):
                del st.session_state[quiz_key]
                st.rerun()

# Display Conversation History
if not st.session_state[chat_key]:
    st.info("👋 **Video transcript indexed!** Ask any question below to study with timestamps, or open **Study Tools** above for a summary.")

for message in st.session_state[chat_key]:
    with st.chat_message(message["role"]):
        st.markdown(
            render_clickable_timestamps(message["content"]),
            unsafe_allow_html=True,
        )

# Chat Input & RAG Execution
user_query = st.chat_input("Ask a question about this video...")
if user_query:
    st.session_state[chat_key].append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        retrieved = retriever.retrieve(video_id=video_id, query=user_query)

        rag_prompt = build_rag_prompt(
            question=user_query,
            retrieved_chunks=retrieved,
            chat_history=st.session_state[chat_key][:-1],
            depth=depth_choice if "depth_choice" in locals() else "Standard",
        )

        stream_chunks = []
        try:
            for chunk in llm.generate_stream(rag_prompt):
                stream_chunks.append(chunk)
                placeholder.markdown(
                    render_clickable_timestamps("".join(stream_chunks)),
                    unsafe_allow_html=True,
                )
            assistant_response = "".join(stream_chunks)
            st.session_state[chat_key].append(
                {"role": "assistant", "content": assistant_response}
            )
        except LLMError as e:
            placeholder.error(str(e))
        except Exception as e:
            placeholder.error(f"Unexpected error: {str(e)}")
