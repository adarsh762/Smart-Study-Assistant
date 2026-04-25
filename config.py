"""
Centralized configuration for Smart Study Assistant.
Loads environment variables from .env file.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
load_dotenv(Path(__file__).resolve().parent / ".env")

# ── Environment Mode ──
ENV = os.getenv("ENV", "development").lower()
IS_PRODUCTION = ENV == "production"

# ── Paths ──
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = str(BASE_DIR / "data")
INDEX_DIR = str(BASE_DIR / "index")

# ── API Keys (fail-fast validation) ──
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
if not GROQ_API_KEY or GROQ_API_KEY == "your_groq_api_key_here":
    print(
        "\n[ERROR] GROQ_API_KEY is not set or is still the placeholder.\n"
        "  1. Get a free key at https://console.groq.com/keys\n"
        "  2. Edit .env and set: GROQ_API_KEY=gsk_your_actual_key\n"
    )
    # Don't exit in case user only wants to test the UI
    # The LLM module will raise a clear error on first call

# ── Model Settings ──
EMBED_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL: str = "llama-3.3-70b-versatile"
LLM_MAX_TOKENS: int = 512
LLM_TEMPERATURE: float = 0.0

# ── Chunking ──
CHUNK_SIZE: int = 1000
CHUNK_OVERLAP: int = 200

# ── Retrieval ──
TOP_K: int = 3
MAX_TOP_K: int = 10          # Upper bound for user-supplied top_k
N_NEIGHBORS: int = 5

# ── Security ──
MAX_UPLOAD_SIZE_MB: int = 20  # Max PDF file size in MB
RATE_LIMIT_CHAT: int = 30     # Max chat requests per minute per IP
RATE_LIMIT_UPLOAD: int = 5    # Max upload requests per minute per IP

# ── CORS ──
ALLOWED_ORIGINS: list = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:7860,http://127.0.0.1:7860"
).split(",")

# ── Server ──
# Railway/Render require 0.0.0.0; local dev uses 127.0.0.1
HOST: str = os.getenv("HOST", "0.0.0.0" if IS_PRODUCTION else "127.0.0.1")
PORT: int = int(os.getenv("PORT", "7860"))

# ── Ensure directories exist ──
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)
