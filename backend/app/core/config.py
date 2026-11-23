"""
Central configuration for SDLcraft backend.
Reads from environment variables or .env file.
"""
from __future__ import annotations
import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent.parent  # backend/

load_dotenv()  # load from current directory .env if present
load_dotenv(BASE_DIR / ".env")  # load backend/.env if present
load_dotenv(BASE_DIR.parent / ".env")  # load project root .env if present

# ── LLM ─────────────────────────────────────────────────────────────────────
LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "http://127.0.0.1:1234/v1")
LLM_API_KEY: str = os.getenv("LLM_API_KEY", "lm-studio")
LLM_MODEL: str = os.getenv("LLM_MODEL", "local-model")
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))

# ── Vector Store ─────────────────────────────────────────────────────────────
CHROMA_PERSIST_DIR: str = os.getenv(
    "CHROMA_PERSIST_DIR",
    str(BASE_DIR / "data" / "chroma_store")
)
CHROMA_COLLECTION: str = os.getenv("CHROMA_COLLECTION", "sdlcraft_documents")

# ── Embedding ────────────────────────────────────────────────────────────────
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# ── Storage ──────────────────────────────────────────────────────────────────
UPLOAD_DIR: Path = BASE_DIR / "uploads"
OUTPUT_DIR: Path = BASE_DIR / "output"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# ── API ──────────────────────────────────────────────────────────────────────
API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
API_PORT: int = int(os.getenv("API_PORT", "8000"))
DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
