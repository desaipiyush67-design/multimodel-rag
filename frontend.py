"""
Frontend — Streamlit UI for the Multimodal PDF Assistant.
Talks to the FastAPI backend (app.py) over HTTP.

Run backend first:   uvicorn app:app --host 0.0.0.0 --port 8000
Then run frontend:   streamlit run frontend.py
"""

import os
import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

# =============================================================
# UI / CSS  (identical to the original Streamlit app)
# =============================================================
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@400;600&display=swap');

:root {
  --bg-0: #07090f; --bg-1: #0b1020; --bg-2: #0f172a;
  --surface: rgba(17, 24, 39, 0.72);
  --border: rgba(148, 163, 184, 0.16);
  --border-strong: rgba(148, 163, 184, 0.28);
  --text: #f1f5f9; --text-dim: #cbd5e1; --muted: #94a3b8;
  --primary: #6366f1; --primary-2: #8b5cf6; --accent: #22d3ee;
  --shadow-lg: 0 30px 80px -20px rgba(0,0,0,0.6);
  --shadow-glow: 0 0 0 1px rgba(99,102,241,0.35), 0 20px 60px -12px rgba(99,102,241,0.45);
  --radius-lg: 20px;
}
* { box-sizing: border-box; }
html, body, .gradio-container {
  background:
    radial-gradient(1200px 600px at 10% -10%, rgba(99,102,241,0.18), transparent 60%),
    radial-gradient(900px 500px at 110% 10%, rgba(34,211,238,0.12), transparent 55%),
    radial-gradient(800px 600px at 50% 120%, rgba(139,92,246,0.14), transparent 60%),
    var(--bg-0) !important;
  color: var(--text) !important;
  font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
}
.gradio-container { max-width: 100% !important; padding: 0 !important; }
#sidebar {
  background: linear-gradient(180deg, rgba(11,16,32,0.92), rgba(11,16,32,0.78)) !important;
  border-right: 1px solid var(--border) !important;
  padding: 24px 20px !important; min-height: 100vh !important;
  backdrop-filter: blur(20px) saturate(140%);
}
.sidebar-label {
  color: var(--muted) !important; font-size: 10.5px !important;
  font-weight: 800 !important; letter-spacing: 0.14em !important;
  text-transform: uppercase !important; margin: 18px 0 10px !important;
}
#main-chat { padding: 28px 32px !important; }
#chatbot {
  background: linear-gradient(180deg, rgba(15,23,42,0.6), rgba(15,23,42,0.4)) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-lg) !important;
  box-shadow: var(--shadow-lg) !important;
  overflow: hidden !important;
}
#chatbot, #chatbot * { color: var(--text) !important; }
#chatbot .message.user {
  background: linear-gradient(135deg, var(--primary), var(--primary-2)) !important;
  color: #fff !important; border-radius: 18px 18px 4px 18px !important;
}
#chatbot .message.bot, #chatbot .message.assistant {
  background: rgba(255,255,255,0.05) !important;
  border: 1px solid var(--border) !important;
  border-radius: 18px 18px 18px 4px !important;
}
#input-container {
  background: linear-gradient(180deg, rgba(15,23,42,0.95), rgba(15,23,42,0.85)) !important;
  border: 1px solid var(--border-strong) !important;
  border-radius: 22px !important; padding: 10px 10px 10px 18px !important;
  margin-top: 18px !important;
}
#input-container:focus-within {
  border-color: rgba(99,102,241,0.55) !important;
  box-shadow: var(--shadow-glow) !important;
}
#query-box textarea {
  background: transparent !important; border: none !important;
  color: var(--text) !important; font-size: 15px !important; resize: none !important;
}
#send-btn {
  background: linear-gradient(135deg, var(--primary), var(--primary-2)) !important;
  color: #fff !important; border: none !important;
  border-radius: 14px !important; height: 44px !important; min-width: 44px !important;
}
button { border-radius: 12px !important; font-weight: 700 !important; }
button.primary, button[variant="primary"] {
  background: linear-gradient(135deg, var(--primary), var(--primary-2)) !important;
  color: #fff !important; border: none !important;
}
.faq-btn {
  width: 100% !important; text-align: left !important; white-space: normal !important;
  padding: 12px 14px !important; margin: 6px 0 !important;
  background: rgba(99,102,241,0.10) !important;
  border: 1px solid rgba(99,102,241,0.28) !important;
  color: var(--text) !important; font-size: 13px !important;
  line-height: 1.4 !important; border-radius: 12px !important;
}
.faq-btn:hover {
  background: rgba(99,102,241,0.22) !important;
  border-color: rgba(99,102,241,0.55) !important;
}
#chatbot table {
  background: #ffffff !important; color: #0f172a !important;
  border-radius: 12px !important; border-collapse: separate !important;
  border-spacing: 0 !important; margin: 10px 0 !important;
}
#chatbot table * { color: #0f172a !important; }
#chatbot th, #chatbot td { padding: 10px 12px !important; border-bottom: 1px solid #e2e8f0 !important; }
#chatbot img { background: #ffffff !important; border-radius: 10px !important; object-fit: contain !important; }
"""


# =============================================================
# Backend helpers
# =============================================================
def api_status():
    try:
        r = requests.get(f"{API_URL}/status", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"status": f"❌ Backend unreachable: {e}", "badge": "❌ Offline",
                "faq": ["", "", "", "", ""], "ready": False}


def api_process_pdf(file_bytes: bytes, filename: str):
    files = {"file": (filename, file_bytes, "application/pdf")}
    r = requests.post(f"{API_URL}/process_pdf", files=files, timeout=60 * 60)
    r.raise_for_status()
    return r.json()


def api_chat(query: str, history):
    r = requests.post(
        f"{API_URL}/chat",
        json={"query": query, "history": history},
        timeout=60 * 10,
    )
    r.raise_for_status()
    return r.json()


# =============================================================
# Streamlit UI
# =============================================================
st.set_page_config(page_title="Multimodal PDF Assistant", layout="wide", initial_sidebar_state="expanded")
st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)

ss = st.session_state
if "initialized" not in ss:
    ss.initialized = True
    ss.history = []
    info = api_status()
    ss.status = info.get("status", "📤 Upload PDF")
    ss.badge = info.get("badge", "⚠️ Pending")
    ss.faq = info.get("faq", ["", "", "", "", ""])
    if info.get("ready"):
        ss.history = [{"role": "assistant",
                       "content": "👋 **Welcome back!** Ready to analyze your document."}]
    else:
        ss.history = [{"role": "assistant",
                       "content": "👋 **Hello!** Upload a PDF to begin."}]

# ----- Sidebar -----
with st.sidebar:
    if st.button("➕ New Chat", use_container_width=True):
        ss.history = []
        st.rerun()

    st.markdown("### 📄 Document")
    pdf_file = st.file_uploader("Upload PDF", type=["pdf"], label_visibility="collapsed")
    if st.button("🚀 Process PDF", type="primary", use_container_width=True):
        if not pdf_file:
            st.warning("⚠️ No file uploaded")
        else:
            with st.spinner("Processing PDF on backend..."):
                try:
                    result = api_process_pdf(pdf_file.read(), pdf_file.name)
                    ss.status = result.get("status", "✅ Engine Ready")
                    ss.badge = result.get("badge", "✔️ Active")
                    ss.faq = result.get("faq", ["", "", "", "", ""])
                    ss.history = [{"role": "assistant",
                                   "content": result.get("message", "✅ Ready.")}]
                except Exception as e:
                    st.error(f"❌ {e}")
            st.rerun()

    st.markdown(ss.status)
    st.markdown(ss.badge)

    with st.expander("📖 GUIDE", expanded=False):
        st.markdown(
            "- **Q&A (Stage 2)**: ask 'what/how/why' for precise facts.\n"
            "- **Visual (Stage 1)**: mention 'image/diagram/figure' to retrieve a visual.\n"
            "- **Concept (Stage 3)**: type a topic for full extraction."
        )

    with st.expander("❓ FAQ (click a question to ask)", expanded=False):
        for i, q in enumerate(ss.faq):
            if q:
                if st.button(q, key=f"faq_{i}", use_container_width=True):
                    ss._pending_query = q
                    st.rerun()

# ----- Main chat area -----
st.markdown("## 💬 Multimodal PDF Assistant")

_fragment = getattr(st, "fragment", None) or getattr(st, "experimental_fragment", None)


def _chat_area():
    chat_container = st.container()
    with chat_container:
        for msg in ss.history:
            role = msg.get("role", "assistant")
            with st.chat_message(role):
                st.markdown(msg.get("content", ""), unsafe_allow_html=True)

    pending = ss.pop("_pending_query", None)
    query = st.chat_input("Message PDF Assistant...", key="main_chat_input")
    query_to_run = pending or query

    if query_to_run and query_to_run.strip():
        ss.history.append({"role": "user", "content": query_to_run})
        with st.spinner("Thinking..."):
            try:
                resp = api_chat(query_to_run, ss.history[:-1])
                ss.history = resp["history"]
            except Exception as e:
                ss.history.append({"role": "assistant",
                                   "content": f"❌ Backend error: {e}"})
        st.rerun()


if _fragment:
    _chat_area = _fragment(_chat_area)

_chat_area()
