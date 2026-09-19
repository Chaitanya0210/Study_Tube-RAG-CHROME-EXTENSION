"""YouTube transcript fetcher with multi-language fallback and error handling."""

import logging
import re
from dataclasses import dataclass
from typing import List, Optional
from urllib.parse import parse_qs, urlparse

from youtube_transcript_api import (
    CouldNotRetrieveTranscript,
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
    YouTubeTranscriptApi,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TranscriptSegment:
    """Individual transcript snippet with timing information."""
    text: str
    start: float
    duration: float

    @property
    def end(self) -> float:
        """End timestamp in seconds."""
        return self.start + self.duration


@dataclass(frozen=True)
class TranscriptResult:
    """Complete transcript payload with metadata."""
    video_id: str
    language: str
    is_generated: bool
    segments: List[TranscriptSegment]

    @property
    def full_text(self) -> str:
        """Concatenated full text."""
        return " ".join(seg.text for seg in self.segments)


class TranscriptError(Exception):
    """User-friendly transcript exception."""
    pass


def extract_video_id(url_or_id: str) -> Optional[str]:
    """Extract 11-character YouTube video ID from various URL formats or raw ID."""
    if not url_or_id:
        return None

    cleaned = url_or_id.strip()

    # If it's already an 11-char ID
    if re.fullmatch(r"[a-zA-Z0-9_-]{11}", cleaned):
        return cleaned

    parsed = urlparse(cleaned)
    hostname = parsed.hostname or ""

    if "youtube.com" in hostname:
        if parsed.path == "/watch":
            qs = parse_qs(parsed.query)
            v = qs.get("v")
            if v and len(v[0]) == 11:
                return v[0]
        elif parsed.path.startswith(("/embed/", "/v/", "/shorts/")):
            parts = parsed.path.split("/")
            if len(parts) >= 3 and len(parts[2]) == 11:
                return parts[2]
    elif "youtu.be" in hostname:
        path = parsed.path.lstrip("/")
        if len(path) >= 11:
            return path[:11]

    # Fallback regex match
    match = re.search(r"(?:v=|\/embed\/|\/shorts\/|youtu\.be\/)([a-zA-Z0-9_-]{11})", cleaned)
    if match:
        return match.group(1)

    return None


def fetch_transcript(video_url_or_id: str, preferred_languages: Optional[List[str]] = None) -> TranscriptResult:
    """Fetch transcripts for a video, preferring manual captions over auto-generated.
    
    Supports English and Hindi by default.
    Raises TranscriptError with friendly user messages on failure.
    """
    video_id = extract_video_id(video_url_or_id)
    if not video_id:
        raise TranscriptError(f"Invalid YouTube URL or video ID: '{video_url_or_id}'")

    langs = preferred_languages or ["en", "en-US", "en-GB", "hi"]
    logger.info("Fetching transcript for video %s with preferred languages: %s", video_id, langs)

    try:
        # Support both current v1.x (instance.list) and legacy (class.list_transcripts) interfaces
        if hasattr(YouTubeTranscriptApi, "list_transcripts"):
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        else:
            api = YouTubeTranscriptApi()
            transcript_list = api.list(video_id)
    except TranscriptsDisabled:
        raise TranscriptError("Subtitles/transcripts are disabled for this video by the creator.")
    except VideoUnavailable:
        raise TranscriptError("This YouTube video is unavailable or set to private.")
    except CouldNotRetrieveTranscript as e:
        logger.warning("Could not retrieve transcript list: %s", e)
        raise TranscriptError("Could not retrieve transcript from YouTube. The video may be age-restricted or blocked.")
    except Exception as e:
        logger.error("Unexpected error retrieving transcript list for %s: %s", video_id, e, exc_info=True)
        raise TranscriptError(f"Failed to connect to YouTube transcript service: {str(e)}")

    # 1. Try manual captions in preferred order
    selected_transcript = None
    try:
        selected_transcript = transcript_list.find_manually_created_transcript(langs)
    except NoTranscriptFound:
        logger.debug("No manual transcript found among %s for video %s", langs, video_id)

    # 2. Fall back to auto-generated captions
    if not selected_transcript:
        try:
            selected_transcript = transcript_list.find_generated_transcript(langs)
        except NoTranscriptFound:
            logger.debug("No generated transcript found among %s for video %s", langs, video_id)

    # 3. Fall back to any available transcript (including translation if needed)
    if not selected_transcript:
        available = list(transcript_list)
        if not available:
            raise TranscriptError("No captions or transcripts were found for this video in any language.")
        
        candidate = available[0]
        if candidate.is_translatable:
            logger.info("Translating transcript from %s to en for video %s", candidate.language_code, video_id)
            try:
                selected_transcript = candidate.translate("en")
            except Exception:
                selected_transcript = candidate
        else:
            selected_transcript = candidate

    # Fetch segment payload
    try:
        raw_segments = selected_transcript.fetch()
    except Exception as e:
        logger.error("Failed to fetch raw transcript segments for %s: %s", video_id, e)
        raise TranscriptError(f"Failed to download caption segments: {str(e)}")

    def _get_val(seg_obj, key, default):
        if hasattr(seg_obj, key):
            return getattr(seg_obj, key)
        if isinstance(seg_obj, dict):
            return seg_obj.get(key, default)
        return default

    segments = []
    for raw in raw_segments:
        text_val = str(_get_val(raw, "text", "")).strip()
        if text_val:
            start_val = float(_get_val(raw, "start", 0.0))
            dur_val = float(_get_val(raw, "duration", 0.0))
            segments.append(TranscriptSegment(text=text_val, start=start_val, duration=dur_val))

    if not segments:
        raise TranscriptError("Transcript was found, but it contained no valid spoken text.")

    return TranscriptResult(
        video_id=video_id,
        language=selected_transcript.language_code,
        is_generated=selected_transcript.is_generated,
        segments=segments,
    )
