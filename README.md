# Multimodal PDF Assistant — Local (Upgraded)

Run a powerful multimodal PDF Q&A assistant fully on your own machine.
**No internet required after setup. No API keys. No data leaves your PC.**

---

## ⚡ What's Upgraded

| Feature | Old Version | New Version |
|---|---|---|
| **UI Framework** | Gradio | Streamlit |
| **Architecture** | Single script | Flask backend + Streamlit frontend |
| **Backend Framework** | — | Flask + Waitress (WSGI) |
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
# Windows (PowerShell):
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
# (must include: flask, flask-cors, waitress, streamlit, requests,
#  sentence-transformers, rank-bm25, transformers, torch, pymupdf,
#  pdfplumber, pillow, numpy, ollama)

# 3. Install Ollama
# Download from https://ollama.com and install it

# 4. Pull the AI model (only once, needs internet)
ollama pull qwen2.5:1.5b
```

---

## Run

You need **3 terminals** open at the same time.

**Terminal 1 — Ollama (AI Brain)**
```bash
ollama serve
```
> If you see "address already in use" — Ollama is already running, skip this step.

**Terminal 2 — Backend (Flask, served by Waitress)**
```bash
# Production / recommended (works on Windows, Mac, Linux):
waitress-serve --host=0.0.0.0 --port=8000 --threads=4 app:app

# Or quick dev mode (shows a "development server" warning — fine for local use):
python app.py
```
Wait until you see `[WARMUP] Done.` and `Serving on http://0.0.0.0:8000`.

> **Always run with a single process.** The ML models and the parsed PDF live
> in process memory, so multiple workers would each load their own copies
> and would not share state. Use threads (`--threads 4`) for concurrency, not
> workers.

**Terminal 3 — Frontend (Streamlit)**
```bash
streamlit run frontend.py

# If 'streamlit' is not recognized on Windows, use:
python -m streamlit run frontend.py
```

Open **http://localhost:8501** in your browser.

---

## How It Works

```
You (Browser)
    ↕  Streamlit (frontend.py) — port 8501
    ↕  Flask    (app.py)      — port 8000   (served by Waitress)
    ↕  Ollama   (qwen2.5)     — port 11434
All running locally on your PC 🔒
```

- **Streamlit** — the chat UI you see in the browser
- **Flask** — receives your question, searches the PDF, talks to Ollama
- **Waitress** — the production WSGI server that runs the Flask app
- **Ollama** — the local AI model that generates answers

---

## Backend Endpoints

The Flask backend exposes the same routes the frontend uses:

| Method | Route | Purpose |
|---|---|---|
| `GET` | `/status` | Engine state, FAQ, ready flag |
| `GET` | `/faq` | Current FAQ questions |
| `POST` | `/process_pdf` | Multipart upload of a PDF |
| `POST` | `/chat` | JSON `{query, history}` → `{history, n_visuals}` |
| `POST` | `/reset_chat` | No-op acknowledgement |

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

- `app.py` — Flask backend (PDF processing, retrieval, AI logic)
- `frontend.py` — Streamlit frontend (chat UI)
- `db.py` — SQLite chat logger (`chat_logs.db`)
- `requirements.txt` — Python dependencies
- `engine_cache.pkl` — auto-generated cache of the last processed PDF
  (delete it to force a fresh upload on next startup)

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `❌ Ollama not running` | Run `ollama serve` in a separate terminal |
| `streamlit : The term ... is not recognized` (Windows) | Use `python -m streamlit run frontend.py` |
| `WARNING: This is a development server...` | Expected when using `python app.py`. Switch to `waitress-serve ... app:app` to silence it |
| Port 8000 not responding | Make sure the Flask backend is running and you see `Serving on http://0.0.0.0:8000` |
| `❌ Backend unreachable` in the UI | Backend not started, wrong port, or firewall. Override with `API_URL=http://127.0.0.1:8000` before launching Streamlit |
| Upload fails with 413 | PDF is larger than 200 MB cap. Raise `MAX_CONTENT_LENGTH` in `app.py` |
| PDF processing times out | Only happens behind reverse proxies. Increase the proxy/gunicorn timeout; Waitress has no default request timeout |
| Slow on large PDFs | Normal — BLIP captions up to 220 images locally |
| CUDA out of memory | Set `DEVICE = "cpu"` in `app.py` |
| Model not found | Run `ollama pull qwen2.5:1.5b` |
| Stale answers / wrong PDF | Delete `engine_cache.pkl` and re-upload |
