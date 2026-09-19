# StudyTube 🎓

> **Turn any YouTube video into an interactive study session right from your browser's side panel.**

StudyTube is an open-source Chrome extension (Manifest V3) that embeds a conversational AI study assistant alongside YouTube. As you watch lectures, tutorials, or podcasts, StudyTube extracts the transcript, indexes it into a local vector database, and lets you ask questions, generate summaries, and test yourself with quizzes—all with clickable timestamps that seek the YouTube video to the exact second.

Best of all, it runs **100% on free tools**: Google Gemini Flash (free tier) and local Sentence-Transformers. No paid API subscriptions, no external database accounts, and no cloud hosting needed.

---

## ✨ Features

- **Chrome Side Panel Integration**: Automatically detects when you're watching a YouTube video and syncs the assistant without leaving the page.
- **Timestamped Citations & 1-Click Video Seeking**: Every factual claim cites timestamps (like `[04:12]`). Clicking the timestamp badge instantly jumps the YouTube player to that moment.
- **Local RAG Pipeline with ChromaDB**: Transcripts are chunked with character-offset tracking to preserve precise start and end times, then stored in an on-disk ChromaDB collection.
- **Diversity-Aware MMR Search**: Uses Maximal Marginal Relevance (MMR) rather than simple similarity search, preventing repetitive chunks from cluttering your context.
- **Local Sentence-Transformers**: Embeddings run locally on your machine (`BAAI/bge-small-en-v1.5` by default, or multilingual models for Hindi and other languages).
- **Flexible Depth Modes**:
  - **Simple**: Beginner-friendly explanations using intuitive analogies.
  - **Standard**: Well-structured, academic-style study notes.
  - **In-depth**: Exhaustive technical breakdowns with an extra *Beyond the video* section.
- **Map-Reduce Summarizer**: Handles long lectures (>1-2 hours) by recursively summarizing chunk batches, avoiding context window limits.
- **Interactive Comprehension Quizzes**: Generates 5 multiple-choice questions from the video with immediate scoring, answers, and timestamped explanations.
- **Robust Multi-Language Fallback**: Prefers manual subtitles, automatically falls back to generated captions, and supports both English and Hindi.

---

## 🏗️ How It Works

```
                                  +---------------------------------------+
                                  |         Chrome Side Panel UI          |
                                  |    (sidepanel.html / sidepanel.js)    |
                                  +-------------------+-------------------+
                                                      |
                                   (iframe: localhost:8501/?embed=true)
                                                      v
+-----------------------------------+     +-----------------------------------+
|          YouTube Tab              |     |          Streamlit App            |
|   (content.js player control)     |<====|       (app/streamlit_app.py)      |
+-----------------------------------+     +-----------------+-----------------+
                                                            |
                                                            v
                                  +---------------------------------------+
                                  |             Core Engine               |
                                  |  - youtube-transcript-api (en/hi)     |
                                  |  - Offset-aware chunk splitter        |
                                  |  - Local SentenceTransformers         |
                                  |  - ChromaDB PersistentClient          |
                                  |  - MMR Retriever (lambda = 0.7)       |
                                  |  - Map-Reduce Summarizer              |
                                  |  - Gemini Flash via google-genai SDK  |
                                  +---------------------------------------+
```

1. When you open a YouTube video, the Chrome extension's background worker detects the video ID and notifies the side panel.
2. The side panel embeds the local Streamlit application in an iframe (`localhost:8501`).
3. Streamlit fetches the captions, splits them into semantic chunks with exact start/end timestamps, generates local embeddings, and stores them in `./chroma_db`.
4. When you ask a question or click **Summarize**, the app performs MMR retrieval and streams an answer from Gemini Flash.
5. When you click a timestamp link (`⏱️ 12:45`), a JavaScript bridge sends a `postMessage` to the extension, which signals the YouTube tab content script to seek `video.currentTime` and resume playback.

---

## 🚀 Getting Started

### Prerequisites
- **Python 3.11+**
- **Google Chrome** (with Manifest V3 side panel support)
- A free **Google Gemini API Key** from [Google AI Studio](https://aistudio.google.com/)

---

### Step 1: Clone the Repository

```bash
git clone https://github.com/Chaitanya0210/Study_Tube-RAG-CHROME-EXTENSION.git
cd Study_Tube-RAG-CHROME-EXTENSION
```

### Step 2: Set Up Python Environment

Create and activate a virtual environment:

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

### Step 3: Configure Your API Key

Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

Open `.env` in any text editor and paste your free Gemini API key:
```ini
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-flash-latest
```

### Step 4: Run the Backend Server

Start the Streamlit application:
```bash
streamlit run app/streamlit_app.py
```
Your backend is now running at `http://localhost:8501`.

### Step 5: Load the Chrome Extension

1. Open Chrome and navigate to `chrome://extensions`.
2. Turn on **Developer mode** in the top-right corner.
3. Click **Load unpacked** (top-left).
4. Select the **`extension`** folder inside this repository (`Study_Tube-RAG-CHROME-EXTENSION/extension`).
5. Pin the **StudyTube** extension to your browser toolbar.

---

## 🎯 Usage

1. Go to any educational video on [YouTube](https://www.youtube.com).
2. Click the **StudyTube** icon in your browser toolbar to open the side panel.
3. The video will be indexed automatically (`Ready (X chunks)`).
4. **Chat**: Type questions like *"What is the main argument in this lecture?"* or *"Explain the formula shown at 10 minutes."*
5. **Jump**: Click any `[mm:ss]` timestamp in the chat response to jump the video directly to that point.
6. **Study Tools**: Expand the **Study Tools & Explanation Depth** menu at the top to:
   - Generate an executive **Summary**
   - Extract **Key Concepts**
   - Take a self-grading 5-question **Quiz**
   - Switch between **Simple**, **Standard**, and **In-depth** modes.

---

## ⚙️ Configuration Options (`.env`)

| Variable | Default | Purpose |
| :--- | :--- | :--- |
| `GEMINI_API_KEY` | *Required* | API key from Google AI Studio |
| `GEMINI_MODEL` | `gemini-flash-latest` | Fast streaming model for free-tier usage |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Local embedding model (or `intfloat/multilingual-e5-small`) |
| `CHUNK_SIZE` | `900` | Target character size per chunk |
| `CHUNK_OVERLAP` | `150` | Character overlap between consecutive chunks |
| `TOP_K` | `5` | Number of context chunks retrieved per query |
| `MMR_LAMBDA` | `0.7` | MMR balance (1.0 = pure similarity, 0.0 = maximal diversity) |
| `CHROMA_DIR` | `./chroma_db` | Storage path for persistent vector database |

---

## 🧪 Running Tests

StudyTube includes unit tests for chunk splitting, timestamp offset math, vector storage, and MMR retrieval:

```bash
python -m pytest -v
```

---

## 📂 Project Structure

```
Study_Tube-RAG-CHROME-EXTENSION/
├── .streamlit/
│   └── config.toml             # Streamlit server & embedding configuration
├── extension/
│   ├── manifest.json           # Manifest V3 (sidePanel, tabs, scripting, CSP)
│   ├── background.js           # Background service worker (tab tracking)
│   ├── sidepanel.html          # Side panel frame container + offline fallback
│   ├── sidepanel.js            # Video URL sync, server health check, seek bridge
│   ├── content.js              # YouTube player controller (seek & play HUD)
│   └── icons/                  # 16x16, 48x48, 128x128 extension icons
├── core/
│   ├── config.py               # .env loader
│   ├── transcript.py           # Caption fetching (manual + auto, en/hi fallback)
│   ├── splitter.py             # RecursiveCharacterTextSplitter + timestamp mapping
│   ├── embeddings.py           # Local Sentence-Transformers singleton
│   ├── vectorstore.py          # ChromaDB PersistentClient per-video storage
│   ├── retriever.py            # Similarity search + MMR re-ranking
│   ├── prompts.py              # Prompt templates (RAG, depth, quiz, summary)
│   ├── llm.py                  # google-genai SDK wrapper + retry & backoff
│   └── summarizer.py           # Map-reduce chunk summarizer for long videos
├── app/
│   └── streamlit_app.py        # Streamlit side panel UI, quiz, and actions
├── tests/
│   ├── test_splitter.py        # Splitter & timestamp alignment tests
│   ├── test_retriever.py       # ChromaDB & MMR retrieval tests
│   └── test_llm_mock.py        # Prompt & LLM streaming mock tests
├── scripts/
│   ├── generate_icons.py       # Icon generator script
│   └── verify_e2e.py           # End-to-end verification across 3 video profiles
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🛡️ License

This project is licensed under the [MIT License](LICENSE).
