# Multimodal PDF Assistant — Local (Upgraded)

Run a powerful multimodal PDF Q&A assistant fully on your own machine.  
**No internet required after setup. No API keys. No data leaves your PC.**

---

## ⚡ What's Upgraded

| Feature | Old Version | New Version |
|---|---|---|
| **UI Framework** | Gradio | Streamlit |
| **Architecture** | Single script | FastAPI backend + Streamlit frontend |
| **Text Generation** | HF Inference API ☁️ (online) | Ollama — runs locally 🖥️ (offline) |
| **Image Captioning** | HF Inference API ☁️ (online) | BLIP model — runs locally 🖥️ (offline) |
| **API Key needed?** | ✅ Yes (HF_TOKEN) | ❌ No |
| **Internet needed?** | ✅ Yes (for generation) | ❌ No (fully offline) |
| **Default port** | 7860 | Frontend: 8501, Backend: 8000 |

---

## Setup

```bash
# 1. Create & activate a virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install Ollama
# Download from https://ollama.com and install it

# 4. Pull the AI model (only once, needs internet)
ollama pull qwen2.5:1.5b
```

---

## Run

You need **3 terminals** open at the same time:

**Terminal 1 — Ollama (AI Brain)**
```bash
ollama serve
```
> If you see "address already in use" — Ollama is already running, skip this step.

**Terminal 2 — Backend (FastAPI)**
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```
Wait until you see `[WARMUP] Done.`

**Terminal 3 — Frontend (Streamlit)**
```bash
streamlit run frontend.py
```

Open **http://localhost:8501** in your browser.

---

## How It Works

```
You (Browser)
    ↕  Streamlit (frontend.py) — port 8501
    ↕  FastAPI   (app.py)      — port 8000
    ↕  Ollama    (qwen2.5)     — port 11434
All running locally on your PC 🔒
```

- **Streamlit** — the chat UI you see in the browser
- **FastAPI** — receives your question, searches the PDF, talks to Ollama
- **Ollama** — the local AI model that generates answers

---

## What Runs Where

| Component | Where |
|---|---|
| PDF parsing | Local (PyMuPDF, pdfplumber) |
| Embeddings | Local (sentence-transformers) |
| Reranker | Local (cross-encoder) |
| Text generation | Local (Ollama — qwen2.5:1.5b) |
| Image captioning | Local (BLIP model) |

---

## Files

- `app.py` — FastAPI backend (PDF processing, retrieval, AI logic)
- `frontend.py` — Streamlit frontend (chat UI)
- `db.py` — SQLite chat logger (`chat_logs.db`)
- `requirements.txt` — Python dependencies

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `❌ Ollama not running` | Run `ollama serve` in a separate terminal |
| Port 8000 not responding | Make sure `uvicorn app:app` is running |
| Slow on large PDFs | Normal — BLIP captions up to 220 images locally |
| CUDA out of memory | Set `DEVICE = "cpu"` in `app.py` |
| Model not found | Run `ollama pull qwen2.5:1.5b` |
