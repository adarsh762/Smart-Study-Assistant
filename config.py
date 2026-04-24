"""
Centralized configuration for Smart Study Assistant.
Loads environment variables from .env file.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
load_dotenv(Path(__file__).resolve().parent / ".env")

# ── Paths ──
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = str(BASE_DIR / "data")
INDEX_DIR = str(BASE_DIR / "index")

# ── API Keys ──
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# ── Model Settings ──
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama-3.3-70b-versatile"
LLM_MAX_TOKENS = 512
LLM_TEMPERATURE = 0.0

# ── Chunking ──
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# ── Retrieval ──
TOP_K = 3
N_NEIGHBORS = 5

# ── Ensure directories exist ──
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)
