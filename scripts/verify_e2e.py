"""End-to-end verification script on 3 target YouTube videos.

1. English lecture (3Blue1Brown - Neural Networks: aircAruvnKk)
2. Hindi video (CodeWithHarry Python in Hindi: 7wnove7K-ZQ)
3. Long video > 1 hour (Andrej Karpathy - Let's build GPT from scratch: kCc8FmEb1nY, 1h 56m)
"""

import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.splitter import split_transcript
from core.transcript import fetch_transcript, extract_video_id
from core.vectorstore import VectorStoreManager
from core.retriever import Retriever
from core.embeddings import get_embedding_model

TEST_VIDEOS = [
    {
        "category": "English Lecture",
        "url": "https://www.youtube.com/watch?v=aircAruvnKk",  # 3Blue1Brown - What is a neural network?
        "description": "3Blue1Brown - Neural Networks (English)",
        "query": "What is a neuron and what are its weights and biases?",
    },
    {
        "category": "Hindi Educational Video",
        "url": "https://www.youtube.com/watch?v=7wnove7K-ZQ",  # CodeWithHarry Python Tutorial
        "description": "Python Tutorial in Hindi",
        "query": "Python programming basic concepts",
    },
    {
        "category": "Long Video (> 1 Hour)",
        "url": "https://www.youtube.com/watch?v=kCc8FmEb1nY",  # Andrej Karpathy - Let's build GPT (1h 56m)
        "description": "Andrej Karpathy - Let's build GPT: from scratch (1h 56m)",
        "query": "self-attention mechanism and multi-head attention",
    },
]

def run_e2e():
    print("=" * 70)
    print("StudyTube End-to-End Verification Across 3 Video Profiles")
    print("=" * 70)

    vector_store = VectorStoreManager(persist_directory="./chroma_db_test")
    embedding_model = get_embedding_model()
    retriever = Retriever(vector_store=vector_store, embedding_model=embedding_model)

    results = []

    for item in TEST_VIDEOS:
        cat = item["category"]
        url = item["url"]
        desc = item["description"]
        query = item["query"]
        video_id = extract_video_id(url)

        print(f"\n[{cat}] Testing: {desc} (ID: {video_id})")
        t0 = time.time()
        test_res = {"category": cat, "description": desc, "video_id": video_id}

        # 1. Fetch transcript
        try:
            transcript = fetch_transcript(video_id)
            print(f"  [OK] Transcript retrieved: {len(transcript.segments)} segments (lang: {transcript.language}, auto: {transcript.is_generated})")
            test_res["segments"] = len(transcript.segments)
            test_res["language"] = transcript.language
            test_res["is_generated"] = transcript.is_generated
            test_res["transcript_success"] = True
        except Exception as e:
            print(f"  [FAIL] Transcript retrieval failed: {e}")
            test_res["transcript_success"] = False
            test_res["error"] = str(e)
            results.append(test_res)
            continue

        # 2. Split into timestamped chunks
        try:
            chunks = split_transcript(transcript, chunk_size=900, chunk_overlap=150)
            first_ts = chunks[0].timestamp if chunks else "N/A"
            last_ts = chunks[-1].timestamp if chunks else "N/A"
            print(f"  [OK] Split into {len(chunks)} chunks (span: {first_ts} -> {last_ts})")
            test_res["chunks"] = len(chunks)
            test_res["time_range"] = f"{first_ts} -> {last_ts}"
            test_res["split_success"] = True
        except Exception as e:
            print(f"  [FAIL] Splitting failed: {e}")
            test_res["split_success"] = False
            test_res["error"] = str(e)
            results.append(test_res)
            continue

        # 3. Vector indexing in ChromaDB
        try:
            doc_count = vector_store.index_chunks(video_id, chunks, embedding_model=embedding_model, force=True)
            print(f"  [OK] Indexed in ChromaDB: {doc_count} documents in collection '{vector_store.get_collection_name(video_id)}'")
            test_res["indexed_docs"] = doc_count
            test_res["index_success"] = True
        except Exception as e:
            print(f"  [FAIL] Vector indexing failed: {e}")
            test_res["index_success"] = False
            test_res["error"] = str(e)
            results.append(test_res)
            continue

        # 4. Retrieval with MMR
        try:
            retrieved = retriever.retrieve(video_id, query=query, top_k=3)
            print(f"  [OK] MMR Retrieval for '{query}':")
            for idx, r in enumerate(retrieved, 1):
                clean_preview = r.chunk.text[:80].encode('ascii', errors='replace').decode('ascii')
                print(f"     [{idx}] Timestamp [{r.chunk.timestamp}] (Score: {r.score:.3f}): {clean_preview}...")
            test_res["retrieved_count"] = len(retrieved)
            test_res["top_timestamp"] = retrieved[0].chunk.timestamp if retrieved else "N/A"
            test_res["retrieval_success"] = True
        except Exception as e:
            print(f"  [FAIL] Retrieval failed: {e}")
            test_res["retrieval_success"] = False
            test_res["error"] = str(e)
            results.append(test_res)
            continue

        duration = time.time() - t0
        test_res["duration_sec"] = round(duration, 2)
        test_res["overall_status"] = "PASSED"
        print(f"  [OK] Video {video_id} passed all tests in {duration:.2f}s")
        results.append(test_res)

    print("\n" + "=" * 70)
    print("End-to-End Verification Summary")
    print("=" * 70)
    for r in results:
        status = r.get("overall_status", "FAILED")
        print(f"- {r['category']} ({r['video_id']}): {status}")
        if status == "PASSED":
            print(f"    Chunks: {r['chunks']}, Time: {r['time_range']}, Top citation: [{r['top_timestamp']}] in {r['duration_sec']}s")
        else:
            print(f"    Error: {r.get('error', 'Unknown')}")

    return results

if __name__ == "__main__":
    run_e2e()
