# Multimodal RAG — Local PDF Intelligence System

> Upload any PDF. Ask questions in natural language. Get precise, cited answers with supporting diagrams and tables — all running **100% offline on your own machine**.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.0-000000?style=flat&logo=flask&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-black?style=flat)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)
![No API Key](https://img.shields.io/badge/API%20Key-Not%20Required-brightgreen?style=flat)

---

## What This Does

Most RAG systems send your documents to the cloud. This one doesn't.

This project is a **fully local, multimodal Retrieval-Augmented Generation (RAG) system** that can read PDFs the way a human expert would — understanding text, interpreting tables, and describing diagrams — then answer questions about them with pinpoint accuracy and source citations.

| Capability | Detail |
|---|---|
| **Text Q&A** | Precise answers with `[Page X]` citations |
| **Table understanding** | Extracts and queries structured data from embedded tables |
| **Image/diagram retrieval** | BLIP-captioned visuals returned alongside answers |
| **Hallucination guard** | Auto-rejects answers containing numbers not found in source chunks |
| **Dynamic FAQ** | Generates 5 suggested questions per document using the LLM |
| **Async processing** | PDF processing runs in a background thread; UI polls live progress |
| **Offline** | Zero internet after setup. No API keys. No telemetry. |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         USER BROWSER                         │
│                    Streamlit UI  :8501                       │
└─────────────────────────┬───────────────────────────────────┘
                           │  HTTP (REST)
┌─────────────────────────▼───────────────────────────────────┐
│                     FLASK BACKEND  :8000                     │
│                                                              │
│  ┌─────────────┐   ┌──────────────┐   ┌──────────────────┐ │
│  │  PDF Parser  │   │   Retriever  │   │   Chat Handler   │ │
│  │  PyMuPDF    │   │              │   │                  │ │
│  │  pdfplumber │──▶│  BM25 (40%)  │──▶│  3-Stage Router  │ │
│  │  BLIP DCAP  │   │  + Vectors   │   │  Stage 1: Visual │ │
│  └─────────────┘   │    (60%)     │   │  Stage 2: Q&A    │ │
│                     │  + Reranker  │   │  Stage 3: Concept│ │
│  ┌─────────────┐   └──────────────┘   └────────┬─────────┘ │
│  │  Embeddings │                                │           │
│  │  MiniLM-L6  │   ┌──────────────┐            │           │
│  │  (local)    │   │   Verifier   │◀───────────┘           │
│  └─────────────┘   │ (anti-hallu- │                        │
│                     │  cination)   │                        │
│  ┌─────────────┐   └──────┬───────┘                        │
│  │  Reranker   │          │                                 │
│  │ ms-marco    │   ┌──────▼───────┐                        │
│  │  (local)    │   │   Ollama LLM │                        │
│  └─────────────┘   │  qwen2.5:1.5b│                        │
│                     │   :11434     │                        │
└─────────────────────┴──────────────┴────────────────────────┘
              Everything runs on your local machine 🔒
```

---

## How the Retrieval Works

This system uses a **3-stage hybrid pipeline** — not a simple vector search:

### Stage 1 — Hybrid Search
Every query is scored by two signals and combined:
- **BM25 (40%)** — exact keyword matching, fast and precise for technical terms
- **Vector similarity (60%)** — semantic search using `sentence-transformers/all-MiniLM-L6-v2` embeddings

### Stage 2 — Cross-Encoder Reranking
The top 20 candidates from Stage 1 are re-scored by `cross-encoder/ms-marco-MiniLM-L-6-v2`, a neural model that reads the query and each chunk *together* — far more accurate than cosine similarity alone.

### Stage 3 — Intent Router
The query is classified into one of three modes:

| Mode | Trigger | Behaviour |
|---|---|---|
| **STAGE1_VISUAL** | "image", "diagram", "figure", "show me" | Returns the closest matching visual + brief description |
| **STAGE2_QUESTION** | "what", "how", "why", or a `?` | Precise Q&A with strict document-grounding rules |
| **STAGE3_GENERAL** | Topic words (no question keyword) | Full concept extraction — definition, components, specs, examples |

### Hallucination Guard
After the LLM generates an answer, a **numeric verifier** checks every number in the answer against the source chunks. If any number cannot be found in the retrieved passages, the answer is rejected and the user is warned — preventing silent hallucinations in technical documents.

---

## Tech Stack

| Component | Technology |
|---|---|
| **Backend API** | Flask 3.0 |
| **Frontend UI** | Streamlit 1.35 |
| **LLM (text generation)** | Ollama — `qwen2.5:1.5b` (local) |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` (local) |
| **Reranker** | `cross-encoder/ms-marco-MiniLM-L-6-v2` (local) |
| **Image captioning** | `Salesforce/blip-image-captioning-base` (local) |
| **Keyword search** | BM25Okapi (`rank-bm25`) |
| **PDF parsing** | PyMuPDF + pdfplumber |
| **Database** | SQLite (chat logging) |
| **Concurrency** | Python threading (async PDF processing) |

---

## Features

- **Multimodal** — understands text, embedded tables, raster images, and vector diagrams all in one pipeline
- **Hybrid BM25 + vector retrieval** with neural reranking for higher precision than pure vector search
- **3-stage intent routing** — the system adapts its retrieval strategy and LLM prompt based on what you're asking
- **Async PDF processing** — upload a 300-page PDF and the UI shows live progress without blocking the server
- **Hallucination detection** — numeric verifier rejects fabricated facts before they reach the user
- **Query spell-correction** — typos in technical terms are auto-corrected against the document vocabulary
- **Dynamic FAQ** — LLM generates 5 suggested questions after each PDF upload
- **Chat history** — conversations are saved per session as JSON
- **Engine cache** — processed PDFs are cached to disk (`engine_cache.pkl`) for instant reload on restart
- **Windows-compatible** — includes MAX_PATH workarounds and `use_fast=False` tokenizer fixes

---

## Project Structure

```
multimodel-rag/
├── app.py            # Flask backend — all AI/retrieval/PDF logic
├── frontend.py       # Streamlit UI — talks to backend over HTTP
├── db.py             # SQLite chat logger
├── requirements.txt  # Python dependencies
└── engine_cache.pkl  # Auto-generated cache (gitignored)
```

---

## Setup & Installation

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com) installed

### 1 — Clone & create environment
```bash
git clone https://github.com/desaipiyush67-design/multimodel-rag.git
cd multimodel-rag

python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 2 — Install Python dependencies
```bash
pip install -r requirements.txt
```

### 3 — Pull the LLM (one-time, needs internet)
```bash
ollama pull qwen2.5:1.5b
```

> The embedding, reranker, and BLIP models download automatically on first run via Hugging Face.

---

## Running

Open **3 terminals** in the project folder (with `venv` activated in each).

**Terminal 1 — Ollama**
```bash
ollama serve
```
> Skip if Ollama is already running (you'll see "address already in use").

**Terminal 2 — Flask Backend**
```bash
python app.py
```
Wait for `[WARMUP] Done.` — models are loaded and ready.

**Terminal 3 — Streamlit Frontend**
```bash
streamlit run frontend.py
# Windows fallback:
python -m streamlit run frontend.py
```

Open **http://localhost:8501** in your browser.

---

## Usage

1. Click **Upload PDF** in the sidebar and select your file
2. Click **Process PDF** — a live progress bar shows extraction status
3. Once ready, ask anything:
   - `"What is the maximum flow rate?"` → precise Q&A with page citations
   - `"Show me the wiring diagram"` → returns the matching diagram
   - `"DEWATS"` → full concept extraction (definition, components, specs, examples)
4. Click any **FAQ question** to ask it instantly

---

## API Reference

The Flask backend is a standalone REST API — you can integrate it with any frontend.

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/status` | Engine state, chunk/image counts, FAQ, ready flag |
| `GET` | `/faq` | Current 5 FAQ questions for the loaded PDF |
| `POST` | `/process_pdf` | Upload PDF (multipart `file` field) — returns `job_id` immediately |
| `GET` | `/process_progress` | Poll processing progress: `{state, percent, desc, result}` |
| `POST` | `/chat` | `{query: str, history: list}` → `{history: list, n_visuals: int}` |
| `POST` | `/reset_chat` | Acknowledge chat reset (stateless — history is managed by client) |

### Example: Chat
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the retention time?", "history": []}'
```

### Example: Upload PDF
```bash
curl -X POST http://localhost:8000/process_pdf \
  -F "file=@your_document.pdf"
# Returns: {"job_id": "abc123", "state": "running"}

# Then poll:
curl http://localhost:8000/process_progress
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `API_URL` | `http://localhost:8000` | Backend URL (set in `frontend.py` env) |
| `CHAT_DB_PATH` | `chat_logs.db` | Path for SQLite chat log database |

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `❌ Ollama not running` | Run `ollama serve` in a separate terminal |
| `streamlit: command not found` (Windows) | Use `python -m streamlit run frontend.py` |
| Port 8000 not responding | Ensure the backend started and shows `[WARMUP] Done.` |
| `❌ Backend unreachable` in UI | Backend not started, or wrong port. Set `API_URL=http://127.0.0.1:8000` |
| Upload fails with 413 error | PDF exceeds 200 MB cap — raise `MAX_CONTENT_LENGTH` in `app.py` |
| Slow on large PDFs | Expected — BLIP captions up to 220 diagrams locally on CPU |
| CUDA out of memory | Set `DEVICE = "cpu"` at the top of `app.py` |
| `Model not found` error | Run `ollama pull qwen2.5:1.5b` |
| Stale answers from old PDF | Delete `engine_cache.pkl` and re-upload your PDF |

---

## Design Decisions

**Why Flask instead of FastAPI?**
Flask is simpler to run as a single-process server with `python app.py`, which is important here because the ML models and parsed PDF are kept **in-process memory** — multiple workers would each load their own copies and not share state. Flask + threading is the right fit.

**Why BM25 + vectors instead of vectors alone?**
Technical documents contain exact model numbers, register names, and units that semantic search misses. BM25 ensures exact-match terms are always retrieved. The hybrid score (60% vector + 40% BM25) outperforms either method alone on technical PDFs.

**Why a cross-encoder reranker?**
Bi-encoder embeddings (used for fast ANN search) score query and document independently. A cross-encoder reads them *together*, giving much higher accuracy at the cost of speed — acceptable for the small candidate set (~20 chunks) after the initial retrieval.

---

## License

MIT — free to use, modify, and distribute.

---

*Built with PyMuPDF · pdfplumber · sentence-transformers · Ollama · BLIP · Flask · Streamlit*
