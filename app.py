"""
Smart Study Assistant — Main Application
FastAPI backend + Gradio UI

Run: python app.py
"""

import os
import logging
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Dict

import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware

import gradio as gr

from config import (
    DATA_DIR, INDEX_DIR, IS_PRODUCTION, ALLOWED_ORIGINS,
    MAX_UPLOAD_SIZE_MB, MAX_TOP_K, RATE_LIMIT_CHAT,
    RATE_LIMIT_UPLOAD, HOST, PORT,
)
from core.pdf_loader import load_pdfs, validate_pdf_magic
from core.embeddings import build_index, load_index
from core.retriever import retrieve
from core.llm import ask, sanitize_input

# ── Logging ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════
# Global State (thread-safe)
# ═══════════════════════════════════════════
_index_lock = threading.Lock()
embeddings = None
docs = None
nn_index = None


def initialize_index() -> None:
    """Load existing index or build from PDFs."""
    global embeddings, docs, nn_index

    with _index_lock:
        # Try loading persisted index
        result = load_index(INDEX_DIR)
        if result is not None:
            embeddings, docs, nn_index = result
            return

        # Build from PDFs
        pdf_docs = load_pdfs(DATA_DIR)
        if pdf_docs:
            embeddings, docs, nn_index = build_index(pdf_docs, INDEX_DIR)
        else:
            logger.warning(
                "No PDFs found. Upload PDFs via the UI or API to get started."
            )


def rebuild_index() -> int:
    """Rebuild the index from all PDFs in data directory (thread-safe)."""
    global embeddings, docs, nn_index

    with _index_lock:
        pdf_docs = load_pdfs(DATA_DIR)
        if not pdf_docs:
            raise ValueError("No PDFs found in data directory.")
        embeddings, docs, nn_index = build_index(pdf_docs, INDEX_DIR)
        return len(pdf_docs)


# ═══════════════════════════════════════════
# Rate Limiter (in-memory, per-IP)
# ═══════════════════════════════════════════
class RateLimiter:
    """Simple in-memory sliding-window rate limiter."""

    def __init__(self) -> None:
        self._requests: Dict[str, List[float]] = defaultdict(list)

    def is_allowed(self, key: str, max_requests: int, window_seconds: int = 60) -> bool:
        now = time.time()
        cutoff = now - window_seconds
        # Prune old entries
        self._requests[key] = [t for t in self._requests[key] if t > cutoff]
        if len(self._requests[key]) >= max_requests:
            return False
        self._requests[key].append(now)
        return True


_rate_limiter = RateLimiter()


def _get_client_ip(request: Request) -> str:
    """Extract client IP from request, respecting X-Forwarded-For."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


# ═══════════════════════════════════════════
# Security Headers Middleware
# ═══════════════════════════════════════════
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response: Response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        if IS_PRODUCTION:
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; "
                "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
                "font-src 'self' https://fonts.gstatic.com; "
                "script-src 'self' 'unsafe-inline'; "
                "img-src 'self' data:; "
                "connect-src 'self'"
            )
        return response


# ═══════════════════════════════════════════
# FastAPI App
# ═══════════════════════════════════════════
api = FastAPI(
    title="Smart Study Assistant API",
    version="1.0.0",
    # Disable docs in production
    docs_url=None if IS_PRODUCTION else "/docs",
    redoc_url=None if IS_PRODUCTION else "/redoc",
)

# Security headers
api.add_middleware(SecurityHeadersMiddleware)

# CORS — restricted origins
api.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)

# Serve frontend
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "frontend")


@api.get("/")
async def serve_frontend():
    """Serve the premium chat frontend."""
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(
            index_path,
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
        )
    return {"message": "Smart Study Assistant API is running."}


# ── Health Check ──
@api.get("/health")
async def health_check():
    """Health check endpoint for deployment platforms."""
    return {"status": "healthy", "indexed": docs is not None}


# ── Request/Response Models ──
class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=3, ge=1, le=MAX_TOP_K)


class ChatResponse(BaseModel):
    answer: str
    sources: List[dict]


# ── Chat Endpoint ──
@api.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest, request: Request):
    """Chat with the assistant using RAG."""
    # Rate limit
    client_ip = _get_client_ip(request)
    if not _rate_limiter.is_allowed(f"chat:{client_ip}", RATE_LIMIT_CHAT):
        raise HTTPException(
            status_code=429,
            detail="Too many requests. Please wait a moment before trying again.",
        )

    # Check index
    if embeddings is None or docs is None or nn_index is None:
        raise HTTPException(
            status_code=503,
            detail="No index loaded. Upload PDFs first.",
        )

    # Sanitize input (prompt injection check)
    try:
        clean_question = sanitize_input(req.question)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Retrieve and answer
    try:
        context = retrieve(clean_question, embeddings, docs, nn_index, top_k=req.top_k)
        answer = ask(clean_question, context)
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        logger.exception("Unexpected error in chat endpoint")
        raise HTTPException(status_code=500, detail="Internal server error.")

    sources = [
        {"source": c["source"], "chunk_id": c["chunk_id"], "score": c["score"]}
        for c in context
    ]
    return ChatResponse(answer=answer, sources=sources)


# ── Upload Endpoint ──
@api.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...), request: Request = None):
    """Upload a PDF and rebuild the index."""
    # Rate limit
    if request:
        client_ip = _get_client_ip(request)
        if not _rate_limiter.is_allowed(f"upload:{client_ip}", RATE_LIMIT_UPLOAD):
            raise HTTPException(
                status_code=429,
                detail="Too many uploads. Please wait before uploading again.",
            )

    # Validate extension
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    # Validate file size
    contents = await file.read()
    max_bytes = MAX_UPLOAD_SIZE_MB * 1024 * 1024
    if len(contents) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {MAX_UPLOAD_SIZE_MB} MB.",
        )

    # Validate PDF magic bytes
    if not contents.startswith(b"%PDF"):
        raise HTTPException(
            status_code=400,
            detail="File does not appear to be a valid PDF.",
        )

    # Sanitize filename — prevent path traversal
    safe_name = Path(file.filename).name  # Strip any directory components
    if not safe_name or safe_name.startswith("."):
        raise HTTPException(status_code=400, detail="Invalid filename.")

    dest = Path(DATA_DIR) / safe_name
    # Verify destination stays inside DATA_DIR
    dest_resolved = dest.resolve()
    data_dir_resolved = Path(DATA_DIR).resolve()
    if not str(dest_resolved).startswith(str(data_dir_resolved)):
        raise HTTPException(status_code=400, detail="Invalid filename.")

    # Write file
    try:
        with open(dest_resolved, "wb") as f:
            f.write(contents)
    except OSError as e:
        logger.error("Failed to save uploaded file: %s", e)
        raise HTTPException(status_code=500, detail="Failed to save file.")

    # Rebuild index — clean up file if rebuild fails
    try:
        count = rebuild_index()
        return {
            "message": f"Uploaded {safe_name}. Index rebuilt with {count} chunks."
        }
    except Exception as e:
        # Clean up orphaned file
        try:
            dest_resolved.unlink(missing_ok=True)
        except OSError:
            pass
        logger.exception("Index rebuild failed after upload")
        raise HTTPException(status_code=500, detail=str(e))


# ── Status Endpoint (sanitized — no internal model details in production) ──
@api.get("/api/status")
async def status():
    """Get current index status."""
    pdf_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".pdf")]

    response = {
        "indexed": docs is not None,
        "num_chunks": len(docs) if docs else 0,
        "pdf_count": len(pdf_files),
    }

    # Only expose file names and model details in development
    if not IS_PRODUCTION:
        response["pdf_files"] = pdf_files
        response["model"] = "llama-3.3-70b-versatile"
        response["embedder"] = "all-MiniLM-L6-v2"

    return response


# ═══════════════════════════════════════════
# Gradio UI
# ═══════════════════════════════════════════
def gradio_chat(user_input, history):
    """Gradio chat callback."""
    history = history or []
    query = (user_input or "").strip()

    if not query:
        return history, ""

    if embeddings is None or docs is None or nn_index is None:
        history.append(
            (query, "No PDFs indexed yet. Upload PDFs to the `data/` folder and restart.")
        )
        return history, ""

    try:
        clean_query = sanitize_input(query)
        context = retrieve(clean_query, embeddings, docs, nn_index, top_k=3)
        answer = ask(clean_query, context)

        # Add source citations
        sources_text = "\n\n---\n**Sources:** " + ", ".join(
            f"`{c['source']}` (relevance: {c['score']:.2f})" for c in context
        )
        answer += sources_text

    except ValueError as e:
        answer = f"Input error: {e}"
    except RuntimeError as e:
        answer = f"Service error: {e}"
    except Exception as e:
        logger.exception("Error in Gradio chat")
        answer = "An unexpected error occurred. Please try again."

    history.append((query, answer))
    return history, ""


with gr.Blocks(title="Smart Study Assistant") as gradio_app:
    gr.Markdown("# Smart Study Assistant\nAsk questions about your uploaded course materials.")

    chatbot = gr.Chatbot(label="EduBot", height=500)
    with gr.Row():
        txt = gr.Textbox(
            show_label=False,
            placeholder="Ask a question about your PDFs...",
            scale=9,
        )
        btn = gr.Button("Send", variant="primary", scale=1)

    state = gr.State([])

    txt.submit(gradio_chat, [txt, state], [chatbot, txt])
    btn.click(gradio_chat, [txt, state], [chatbot, txt])

# Mount Gradio on FastAPI
api = gr.mount_gradio_app(api, gradio_app, path="/gradio")


# ═══════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  Smart Study Assistant")
    print("=" * 50)

    initialize_index()

    print(f"\n[*] Starting server on {HOST}:{PORT}...")
    print(f"   Frontend:  http://localhost:{PORT}")
    print(f"   Gradio UI: http://localhost:{PORT}/gradio")
    if not IS_PRODUCTION:
        print(f"   API Docs:  http://localhost:{PORT}/docs")
    print(f"   Health:    http://localhost:{PORT}/health")
    print("=" * 50 + "\n")

    uvicorn.run(api, host=HOST, port=PORT)
