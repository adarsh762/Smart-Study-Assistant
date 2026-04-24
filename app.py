"""
Smart Study Assistant — Main Application
FastAPI backend + Gradio UI

Run: python app.py
"""

import os
import shutil
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

import gradio as gr

from config import DATA_DIR, INDEX_DIR
from core.pdf_loader import load_pdfs
from core.embeddings import build_index, load_index
from core.retriever import retrieve
from core.llm import ask

# ═══════════════════════════════════════════
# Global State
# ═══════════════════════════════════════════
embeddings = None
docs = None
nn_index = None


def initialize_index():
    """Load existing index or build from PDFs."""
    global embeddings, docs, nn_index

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
        print("No PDFs found. Upload PDFs via the UI or API to get started.")


def rebuild_index():
    """Rebuild the index from all PDFs in data directory."""
    global embeddings, docs, nn_index
    pdf_docs = load_pdfs(DATA_DIR)
    if not pdf_docs:
        raise ValueError("No PDFs found in data directory.")
    embeddings, docs, nn_index = build_index(pdf_docs, INDEX_DIR)
    return len(pdf_docs)


# ═══════════════════════════════════════════
# FastAPI App
# ═══════════════════════════════════════════
api = FastAPI(title="Smart Study Assistant API", version="1.0.0")

api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "frontend")


@api.get("/")
async def serve_frontend():
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Smart Study Assistant API is running. Visit /gradio for the Gradio UI."}


class ChatRequest(BaseModel):
    question: str
    top_k: int = 3


class ChatResponse(BaseModel):
    answer: str
    sources: List[dict]


@api.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    """Chat with the assistant using RAG."""
    if embeddings is None or docs is None or nn_index is None:
        raise HTTPException(status_code=503, detail="No index loaded. Upload PDFs first.")

    context = retrieve(req.question, embeddings, docs, nn_index, top_k=req.top_k)
    answer = ask(req.question, context)

    sources = [
        {"source": c["source"], "chunk_id": c["chunk_id"], "score": c["score"]}
        for c in context
    ]
    return ChatResponse(answer=answer, sources=sources)


@api.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload a PDF and rebuild the index."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    dest = os.path.join(DATA_DIR, file.filename)
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        count = rebuild_index()
        return {"message": f"Uploaded {file.filename}. Index rebuilt with {count} chunks."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@api.get("/api/status")
async def status():
    """Get current index status."""
    pdf_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".pdf")]
    return {
        "indexed": docs is not None,
        "num_chunks": len(docs) if docs else 0,
        "pdf_files": pdf_files,
        "model": "llama-3.3-70b-versatile",
        "embedder": "all-MiniLM-L6-v2",
    }


# ═══════════════════════════════════════════
# Gradio UI
# ═══════════════════════════════════════════
def gradio_chat(user_input, history):
    """Gradio chat callback."""
    history = history or []
    query = (user_input or "").strip()

    if not query:
        return history, ""

    if embeddings is None or docs is None:
        history.append((query, "No PDFs indexed yet. Upload PDFs to the `data/` folder and restart."))
        return history, ""

    try:
        context = retrieve(query, embeddings, docs, nn_index, top_k=3)
        answer = ask(query, context)

        # Add source citations
        sources_text = "\n\n---\n📚 **Sources:** " + ", ".join(
            f"`{c['source']}` (relevance: {c['score']:.2f})" for c in context
        )
        answer += sources_text

    except Exception as e:
        answer = f"Error: {e}"

    history.append((query, answer))
    return history, ""


gradio_css = """
.gradio-container { max-width: 900px !important; margin: auto !important; }
"""


with gr.Blocks(title="Smart Study Assistant") as gradio_app:
    gr.Markdown("# 🎓 Smart Study Assistant\nAsk questions about your uploaded course materials.")

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

    print("\n[*] Starting server...")
    print("   Frontend:  http://localhost:7860")
    print("   Gradio UI: http://localhost:7860/gradio")
    print("   API Docs:  http://localhost:7860/docs")
    print("=" * 50 + "\n")

    uvicorn.run(api, host="0.0.0.0", port=7860)
