"""Configuration loader for StudyTube."""

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from dotenv import load_dotenv

# Automatically load .env from project root
_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
if _ENV_PATH.exists():
    load_dotenv(dotenv_path=_ENV_PATH)
else:
    load_dotenv()


@dataclass(frozen=True)
class AppConfig:
    """Application configuration container."""
    gemini_api_key: str
    gemini_model: str = "gemini-2.5-flash"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    chunk_size: int = 900
    chunk_overlap: int = 150
    top_k: int = 5
    mmr_lambda: float = 0.7
    chroma_dir: str = "./chroma_db"

    @property
    def is_gemini_configured(self) -> bool:
        """Check if a non-placeholder Gemini API key is provided."""
        return bool(self.gemini_api_key and not self.gemini_api_key.startswith("your_"))


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """Load configuration from environment variables with safe defaults."""
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()
    embedding_model = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5").strip()
    
    try:
        chunk_size = int(os.getenv("CHUNK_SIZE", "900"))
    except ValueError:
        chunk_size = 900

    try:
        chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "150"))
    except ValueError:
        chunk_overlap = 150

    try:
        top_k = int(os.getenv("TOP_K", "5"))
    except ValueError:
        top_k = 5

    try:
        mmr_lambda = float(os.getenv("MMR_LAMBDA", "0.7"))
    except ValueError:
        mmr_lambda = 0.7

    chroma_dir = os.getenv("CHROMA_DIR", "./chroma_db").strip()

    return AppConfig(
        gemini_api_key=api_key,
        gemini_model=model,
        embedding_model=embedding_model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k=top_k,
        mmr_lambda=mmr_lambda,
        chroma_dir=chroma_dir,
    )
