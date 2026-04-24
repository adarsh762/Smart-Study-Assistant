<div align="center">

# 🎓 Smart Study Assistant

**RAG-powered educational chatbot that answers questions from your PDFs in real-time.**

Built with Groq Llama-3.3 · Sentence Transformers · FastAPI · Nearest Neighbor Search

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![Groq](https://img.shields.io/badge/Groq-Llama_3.3-orange?style=flat)](https://groq.com)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat)](LICENSE)

</div>

---

## ✨ Features

- **📄 PDF Upload** — Drag-and-drop PDF upload with automatic text extraction and chunking
- **🔍 RAG Retrieval** — Sentence Transformer embeddings + Nearest Neighbor search for context retrieval
- **🤖 Groq LLM** — Lightning-fast inference with Llama 3.3 70B via Groq API
- **💬 Real-time Chat** — Beautiful dark-themed chat interface with markdown rendering
- **📊 Source Citations** — Every answer includes relevance-scored source citations
- **🎨 Premium UI** — Custom-built frontend with animations, typing indicators, and responsive design

---

## 🏗️ Architecture

```mermaid
graph LR
    A[PDF Upload] --> B[Text Extraction<br>pypdf]
    B --> C[Chunking<br>1000 chars + overlap]
    C --> D[Embeddings<br>MiniLM-L6-v2]
    D --> E[Vector Index<br>NearestNeighbors]
    
    F[User Question] --> G[Query Embedding]
    G --> E
    E --> H[Top-K Context]
    H --> I[Groq LLM<br>Llama 3.3 70B]
    I --> J[Answer + Sources]
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/smart-study-assistant.git
cd smart-study-assistant
pip install -r requirements.txt
```

### 2. Set up API Key

```bash
cp .env.example .env
# Edit .env and add your Groq API key
# Get one free at https://console.groq.com/keys
```

### 3. Add PDFs

Drop your course PDFs into the `data/` folder.

### 4. Run

```bash
python app.py
```

Open your browser:
- **Frontend**: [http://localhost:7860](http://localhost:7860)
- **Gradio UI**: [http://localhost:7860/gradio](http://localhost:7860/gradio)
- **API Docs**: [http://localhost:7860/docs](http://localhost:7860/docs)

---

## 📁 Project Structure

```
smart-study-assistant/
├── app.py              # FastAPI + Gradio entry point
├── config.py           # Centralized settings
├── core/
│   ├── pdf_loader.py   # PDF parsing + chunking
│   ├── embeddings.py   # Sentence Transformer indexing
│   ├── retriever.py    # Nearest neighbor search
│   └── llm.py          # Groq API wrapper
├── frontend/
│   └── index.html      # Premium chat UI
├── data/               # PDF storage (gitignored)
├── index/              # Persisted embeddings (gitignored)
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## 🛠️ API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/chat` | Send a question, get RAG answer |
| `POST` | `/api/upload` | Upload a PDF file |
| `GET` | `/api/status` | Get index stats |

### Chat Example

```bash
curl -X POST http://localhost:7860/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is machine learning?", "top_k": 3}'
```

---

## ⚙️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **LLM** | Groq API (Llama 3.3 70B Versatile) |
| **Embeddings** | Sentence Transformers (all-MiniLM-L6-v2) |
| **Retrieval** | scikit-learn NearestNeighbors (cosine) |
| **PDF Parsing** | pypdf |
| **Backend** | FastAPI + Uvicorn |
| **Alt UI** | Gradio |
| **Frontend** | Vanilla HTML/CSS/JS |

---

## 📝 License

MIT License — feel free to use, modify, and distribute.

---

<div align="center">
  <sub>Built by <strong>Adarshdev Singh Pawar</strong> · Internship Capstone Project</sub>
</div>

