import os
import re
import io
import time
import base64
import traceback
import pickle
import concurrent.futures
import difflib
from typing import List, Dict

import fitz  # PyMuPDF
import pdfplumber
import gradio as gr
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from huggingface_hub import InferenceClient

import db  # SQLite Logging Module

# =============================================================
# CONFIGURATION
# =============================================================
HF_MODEL_GEN = "Qwen/Qwen2.5-7B-Instruct"
HF_MODEL_CAPTION = "Salesforce/blip-image-captioning-base"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K_CHUNKS = 12
RERANK_TOP_N = 8

_models = {}

def get_embed_model():
    if "embed" not in _models:
        print("[INIT] Loading Embedding Model ...")
        _models["embed"] = SentenceTransformer(EMBED_MODEL_NAME)
    return _models["embed"]

def get_rerank_model():
    if "rerank" not in _models:
        print("[INIT] Loading Reranker Model ...")
        _models["rerank"] = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _models["rerank"]

def get_hf_client(model: str = HF_MODEL_GEN):
    token = os.environ.get("HF_TOKEN")
    if not token or not token.strip():
        return None
    return InferenceClient(model=model, token=token.strip())

# =============================================================
# DYNAMIC FAQ GENERATION (NEW)
# =============================================================
def generate_pdf_faq(chunks):
    """Generate 5 document-specific Q&As. Always returns a clean string."""
    try:
        if not chunks:
            return "_FAQ unavailable — no content extracted._"
        text_parts = []
        for c in chunks:
            if isinstance(c, dict):
                t = c.get("text", "")
                if isinstance(t, str) and t.strip():
                    text_parts.append(t.strip())
        if not text_parts:
            return "_FAQ unavailable — no readable text in document._"
        sample = text_parts[:: max(1, len(text_parts) // 8)][:8]
        context = "\n\n---\n\n".join(sample)[:6000]
        prompt = (
            "Based ONLY on the document excerpts below, generate exactly 5 FAQs "
            "(questions only, no answers) that a reader might have. "
            "Format strictly as Markdown:\n\n"
            "**Q1:** <question>\n\n"
            "**Q2:** <question>\n\n"
            "**Q3:** <question>\n\n"
            "**Q4:** <question>\n\n"
            "**Q5:** <question>\n\n"
            "Do NOT include answers. Do NOT output Python lists or JSON. Only Markdown questions.\n\n"
            f"DOCUMENT EXCERPTS:\n{context}"
        )

        client = get_hf_client()
        if not client:
            return "_FAQ unavailable — HF_TOKEN missing._"
        resp = client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=700, temperature=0.3,
        )
        out = resp.choices[0].message.content.strip()
        if out.startswith("[") and "'text'" in out:
            return "_FAQ generation failed — please re-process the PDF._"
        return out or "_FAQ unavailable._"
    except Exception as e:
        print(f"[FAQ generation error] {e}")
        return "_FAQ unavailable — generation error._"

# =============================================================
# UTILS
# =============================================================
def clean_text(text: str) -> str:
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def table_to_markdown(table: list) -> str:
    if not table or len(table) < 1:
        return ""
    cleaned = [[str(c).replace("\n", " ").strip() if c else "" for c in row] for row in table]
    if all(all(c == "" for c in r) for r in cleaned):
        return ""
    header = cleaned[0]
    ncols = len(header)
    md = "| " + " | ".join(header) + " |\n"
    md += "| " + " | ".join(["---"] * ncols) + " |\n"
    for row in cleaned[1:]:
        while len(row) < ncols:
            row.append("")
        md += "| " + " | ".join(row[:ncols]) + " |\n"
    return md.strip()

def pil_to_base64(img: Image.Image, max_px: int = 800) -> str:
    buf = io.BytesIO()
    img = img.copy()
    img.thumbnail((max_px, max_px))
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()

def merge_rects(rects, gap: float = 35):
    """Merge nearby rectangles so vector parts become one visual crop."""
    if not rects:
        return []

    rects = [fitz.Rect(r) for r in rects]
    merged = []

    for r in rects:
        added = False
        expanded_r = r + (-gap, -gap, gap, gap)

        for i, m in enumerate(merged):
            expanded_m = m + (-gap, -gap, gap, gap)

            if expanded_r.intersects(expanded_m):
                merged[i] = m | r
                added = True
                break

        if not added:
            merged.append(r)

    changed = True
    while changed:
        changed = False
        new_merged = []

        for r in merged:
            added = False
            expanded_r = r + (-gap, -gap, gap, gap)

            for i, m in enumerate(new_merged):
                expanded_m = m + (-gap, -gap, gap, gap)

                if expanded_r.intersects(expanded_m):
                    new_merged[i] = m | r
                    added = True
                    changed = True
                    break

            if not added:
                new_merged.append(r)

        merged = new_merged

    return merged

def extract_vector_visuals_from_page(page, page_no: int, current_concept: str):
    """
    Extract vector drawings/callouts by detecting drawing boxes,
    merging nearby vector regions, and rendering them as image crops.
    """
    vector_items = []

    try:
        drawings = page.get_drawings()
    except Exception:
        return vector_items

    rects = []

    for d in drawings:
        rect = d.get("rect")
        if not rect:
            continue

        rect = fitz.Rect(rect)

        # skip tiny lines/noise
        if rect.width < 40 or rect.height < 30:
            continue

        # skip almost full-page background/vector noise
        page_area = page.rect.width * page.rect.height
        rect_area = rect.width * rect.height
        if rect_area > page_area * 0.65:
            continue

        rects.append(rect)

    merged_rects = merge_rects(rects, gap=45)

    for idx, rect in enumerate(merged_rects):
        try:
            # add margin so bubble text / nearby visual is not cut
            crop_rect = rect + (-25, -25, 25, 25)
            crop_rect = crop_rect & page.rect

            if crop_rect.width < 80 or crop_rect.height < 60:
                continue

            # avoid huge noisy crops
            page_area = page.rect.width * page.rect.height
            crop_area = crop_rect.width * crop_rect.height
            if crop_area > page_area * 0.55:
                continue

            # render vector region into image
            pix = page.get_pixmap(clip=crop_rect, dpi=200, alpha=False)
            pil_img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")

            # skip blank/low information crops
            if pil_img.width < 80 or pil_img.height < 60:
                continue
            if np.array(pil_img).std() < 3:
                continue

            surrounding_text = page.get_text("text", clip=crop_rect).strip()

            vector_items.append({
                "page": page_no,
                "base64": pil_to_base64(pil_img),
                "caption": f"Vector visual / callout / diagram: {surrounding_text[:1000]}",
                "concept": current_concept,
                "image": pil_img,
            })

        except Exception:
            continue

    return vector_items

# =============================================================
# EXTRACTION
# =============================================================
def extract_pdf_comprehensive(path: str, progress=gr.Progress()):
    all_chunks: List[Dict] = []
    images: List[Dict] = []
    progress(0.1, desc="Opening PDF...")
    doc = fitz.open(path)
    current_concept = "Intro/Overview"
    total_pages = len(doc)

    for pn in range(total_pages):
        progress((pn + 1) / total_pages * 0.7, desc=f"Processing Page {pn+1}/{total_pages}...")
        page = doc[pn]

        # 1. TABLE EXTRACTION (fitz)
        try:
            tabs = page.find_tables()
            for tab in tabs:
                md = table_to_markdown(tab.extract())
                if len(md) > 20:
                    all_chunks.append({
                        "page": pn + 1,
                        "text": f"### [TABLE DATA]\n{md}",
                        "type": "table",
                        "concept": current_concept,
                    })
        except Exception:
            pass

        # 2. TEXT EXTRACTION
        try:
            blocks = page.get_text("dict", flags=11)["blocks"]
        except Exception:
            blocks = []
        page_text_acc = []
        for b in blocks:
            if b.get("type") == 0:
                block_text = ""
                for line in b.get("lines", []):
                    for span in line.get("spans", []):
                        txt = clean_text(span.get("text", ""))
                        if not txt:
                            continue
                        if ("bold" in span.get("font", "").lower() or span.get("size", 0) > 12) and len(txt) < 100:
                            current_concept = txt
                        block_text += txt + " "
                if len(block_text.strip()) > 15:
                    page_text_acc.append(block_text.strip())

        full_page_text = " ".join(page_text_acc)
        words = full_page_text.split()
        for i in range(0, len(words), 350):
            chunk_words = words[i: i + 450]
            if len(chunk_words) < 10:
                continue
            all_chunks.append({
                "page": pn + 1,
                "text": " ".join(chunk_words),
                "type": "text",
                "concept": current_concept,
            })

        # 3. IMAGE EXTRACTION
        try:
            for img_info in page.get_images(full=True):
                xref = img_info[0]
                base_img = doc.extract_image(xref)
                img_data = base_img["image"]
                try:
                    pil_img = Image.open(io.BytesIO(img_data)).convert("RGB")
                except Exception:
                    continue
                if pil_img.width < 50 or pil_img.height < 50:
                    continue
                if np.array(pil_img).std() < 2:
                    continue
                rects = page.get_image_rects(xref)
                surrounding_text = ""
                if rects:
                    search_rect = rects[0] + (-50, -50, 50, 200)
                    surrounding_text = page.get_text("text", clip=search_rect).strip()
                images.append({
                    "page": pn + 1,
                    "base64": pil_to_base64(pil_img),
                    "caption": f"Diagram: {surrounding_text[:1000]}",
                    "concept": current_concept,
                    "image": pil_img,
                })
        except Exception:
            pass

        # 4. VECTOR DRAWING / CALLOUT EXTRACTION
        try:
            vector_visuals = extract_vector_visuals_from_page(
                page=page,
                page_no=pn + 1,
                current_concept=current_concept
            )

            for vv in vector_visuals:
                duplicate = False
                for im in images:
                    if im.get("page") == vv.get("page") and im.get("caption") == vv.get("caption"):
                        duplicate = True
                        break

                if not duplicate:
                    images.append(vv)

        except Exception:
            pass

    # Fallback Table Extraction (pdfplumber)
    progress(0.8, desc="Finalizing tables...")
    try:
        with pdfplumber.open(path) as pdf:
            for idx, p in enumerate(pdf.pages):
                for tbl in p.extract_tables() or []:
                    md = table_to_markdown(tbl)
                    if len(md) > 30 and not any(md[:50] in c["text"] for c in all_chunks if c["type"] == "table"):
                        all_chunks.append({
                            "page": idx + 1,
                            "text": f"### [TABLE DATA]\n{md}",
                            "type": "table",
                            "concept": "Appendix",
                        })
    except Exception:
        pass

    # Parallel AI Captioning
    def process_image(img_info):
        try:
            client = get_hf_client(HF_MODEL_CAPTION)
            if client is None:
                return img_info
            buf = io.BytesIO()
            img_info['image'].save(buf, format="JPEG")
            res = client.image_to_text(buf.getvalue())
            cap = res.generated_text if hasattr(res, "generated_text") else str(res)
            cap = cap.strip()
            if cap:
                img_info['caption'] = f"{img_info['caption']} | AI View: {cap}"
        except Exception:
            pass
        return img_info

    if images:
        progress(0.9, desc="AI Captioning technical diagrams...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            images = list(executor.map(process_image, images))
        for im in images:
            all_chunks.append({
                "page": im['page'],
                "text": f"### [TECHNICAL DIAGRAM]\n{im['caption']}",
                "type": "image_meta",
                "concept": im['concept'],
            })
            im.pop('image', None)

    doc.close()
    progress(1.0, desc="Extraction Complete ✓")
    return all_chunks, images

# =============================================================
# HYBRID RETRIEVAL
# =============================================================
def hybrid_retrieve(query: str, chunks: List[Dict], embs: np.ndarray, bm25: BM25Okapi, top_k: int = 15):
    if not chunks or embs is None:
        return []
    model = get_embed_model()
    q_emb = model.encode([query], normalize_embeddings=True).astype("float32")
    vs = np.dot(embs, q_emb.T).squeeze()
    if vs.ndim == 0:
        vs = np.array([float(vs)])
    stop_words = {"what", "is", "the", "a", "an", "of", "and", "in", "to", "for", "with", "on", "by", "at"}
    query_tokens = [w for w in query.lower().split() if w not in stop_words]
    bs = np.array(bm25.get_scores(query_tokens), dtype="float32")

    def normalize(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-9) if x.max() > x.min() else np.zeros_like(x)

    scores = 0.6 * normalize(vs) + 0.4 * normalize(bs)
    top_indices = np.argsort(scores)[::-1][:top_k]
    candidate_chunks = [chunks[int(i)] for i in top_indices]

    reranker = get_rerank_model()
    pairs = [[query, c["text"][:600]] for c in candidate_chunks]
    rerank_scores = reranker.predict(pairs)
    final_indices = np.argsort(rerank_scores)[::-1]
    sorted_chunks = [candidate_chunks[int(i)] for i in final_indices]

    results = []
    for i, chunk in enumerate(sorted_chunks):
        if i < 6 or rerank_scores[final_indices[i]] > -2.0:
            results.append(chunk)
    return results if results else [sorted_chunks[0]]

# =============================================================
# CHAT HANDLER
# =============================================================
def chat_handler(query, history, chunks, images, embs, bm25):
    if history is None:
        history = []
    if not chunks or embs is None:
        history.append({"role": "assistant", "content": "⚠️ Please upload and process a PDF document first."})
        return history, "", []

    try:
        # Spelling correction
        doc_vocab = set()
        for c in chunks:
            doc_vocab.add(c['concept'].lower())
            for word in re.findall(r'\b[a-zA-Z]{3,}\b', c['text'].lower()):
                doc_vocab.add(word)

        corrected_words = []
        for word in query.lower().split():
            matches = difflib.get_close_matches(word, list(doc_vocab), n=1, cutoff=0.8)
            corrected_words.append(matches[0] if matches else word)
        search_query = " ".join(corrected_words)

        query_lower = search_query.lower()
        query_words_set = set(query_lower.split())
        visual_keywords = {"picture", "image", "diagram", "figure", "photo", "show", "visual", "pic", "draw", "schematic", "map", "chart", "graph"}
        question_keywords = {"what", "how", "where", "why", "when", "who", "which", "explain", "advantages", "disadvantages", "adv", "define", "describe", "difference", "compare"}

        is_stage1_visual = any(vk in query_words_set for vk in visual_keywords)
        is_stage2_question = any(qk in query_words_set for qk in question_keywords) or "?" in query_lower

        if is_stage1_visual:
            mode = "STAGE1_VISUAL"
            max_context = 7
            system_prompt = (
                "You are a strict technical assistant. You must ONLY use the provided CONTEXT.\n"
                "The user is asking for a specific visual (image, diagram, picture, etc.).\n"
                "1. Provide a very brief 1-2 sentence description of the requested visual based on the context.\n"
                "2. If the visual or topic is not in the context, say 'No info in document.'\n"
                "3. Cite with [Page X].\n"
                "4. NO OUTSIDE KNOWLEDGE. NO HALLUCINATION."
            )
        elif is_stage2_question:
            mode = "STAGE2_QUESTION"
            max_context = 7
            system_prompt = (
                "You are a strict technical assistant. You must ONLY use the provided CONTEXT.\n"
                "The user is asking a specific question.\n"
                "1. Answer ONLY what is specifically asked. Be concise and specific. Do NOT add extra unasked info.\n"
                "2. If the answer is not explicitly in the context, say 'No info in document.'\n"
                "3. Cite with [Page X].\n"
                "4. NO OUTSIDE KNOWLEDGE. NO HALLUCINATION."
            )
        else:
            mode = "STAGE3_GENERAL"
            max_context = 15
            system_prompt = (
                "You are a strict technical assistant. You must ONLY use the provided CONTEXT.\n"
                "The user is asking about a general topic.\n"
                "1. Extract and summarize EVERYTHING related to this topic from the context.\n"
                "2. If the topic is not in the context, say 'No info in document.'\n"
                "3. Use bullet points for readability.\n"
                "4. Cite with [Page X].\n"
                "5. NO OUTSIDE KNOWLEDGE. NO HALLUCINATION."
            )

        top_chunks = hybrid_retrieve(search_query, chunks, embs, bm25, top_k=20)
        reranker = get_rerank_model()
        text_pairs = [[search_query, c['text'][:600]] for c in top_chunks]
        text_scores = reranker.predict(text_pairs)
        scored_chunks = sorted(zip(top_chunks, text_scores), key=lambda x: x[1], reverse=True)
        final_top_chunks = [c for c, _ in scored_chunks[:max_context]]

        context_parts = [f"--- [SOURCE: Page {c['page']} | Section: {c['concept']}] ---\n{c['text']}" for c in final_top_chunks]
        context_str = "\n\n".join(context_parts)[:15000]

        messages = [{"role": "system", "content": system_prompt}]
        for turn in history[-6:]:
            content = str(turn.get("content", ""))
            for marker in ["### \U0001f4ca", "### \U0001f5bc", "<div", "Data Evidence", "Visual Evidence"]:
                if marker in content:
                    content = content.split(marker)[0]
            messages.append({"role": turn["role"], "content": content.strip()})
        messages.append({"role": "user", "content": f"CONTEXT:\n{context_str}\n\nUSER QUESTION: {query}"})

        client = get_hf_client()
        if not client:
            ans = "⚠️ **API Token Missing.** Please set `HF_TOKEN` in secrets."
        else:
            try:
                response = client.chat_completion(messages=messages, max_tokens=1000, temperature=0.0)
                raw = response.choices[0].message.content
                # Coerce list/dict responses into clean text
                if isinstance(raw, list):
                    parts = []
                    for item in raw:
                        if isinstance(item, dict):
                            parts.append(str(item.get("text", "")))
                        else:
                            parts.append(str(item))
                    ans = "\n".join(p for p in parts if p).strip()
                elif isinstance(raw, dict):
                    ans = str(raw.get("text", raw)).strip()
                else:
                    ans = str(raw).strip()
                # Clean literal \n escapes the model sometimes emits
                ans = ans.replace("\\n", "\n")
            except Exception as e:
                ans = f"❌ **Cloud Error:** {str(e)}"


        # Visuals / Tables per stage
        final_rel_imgs = []
        final_rel_tabs = []
        if "no info in document" not in ans.lower():
            if mode == "STAGE1_VISUAL" and images:
                img_pairs = [[search_query, im['caption']] for im in images]
                img_scores = reranker.predict(img_pairs)
                scored_imgs = sorted(zip(images, img_scores), key=lambda x: x[1], reverse=True)
                if scored_imgs and scored_imgs[0][1] > 1.7:
                    final_rel_imgs = [scored_imgs[0][0]]
            elif mode == "STAGE3_GENERAL":
                if images:
                    img_pairs = [[search_query, im['caption']] for im in images]
                    img_scores = reranker.predict(img_pairs)
                    scored_imgs = sorted(zip(images, img_scores), key=lambda x: x[1], reverse=True)
                    for im, score in scored_imgs:
                        if score > 1.7:
                            final_rel_imgs.append(im)
                rel_tabs = [c for c in chunks if c['type'] == 'table']
                if rel_tabs:
                    tab_pairs = [[search_query, t['text'][:500]] for t in rel_tabs]
                    tab_scores = reranker.predict(tab_pairs)
                    scored_tabs = sorted(zip(rel_tabs, tab_scores), key=lambda x: x[1], reverse=True)
                    for t, score in scored_tabs:
                        if score > 0.0:
                            final_rel_tabs.append(t)
        tables_md = ""
        if final_rel_tabs:
            tables_md = "\n\n### \U0001f4ca Data Evidence\n"
            for t in final_rel_tabs: tables_md = ""
            tables_md += f"<div style='background:#ffffff;padding:16px;border-radius:14px;border:1px solid #e2e8f0;margin-bottom:12px;box-shadow:0 8px 24px rgba(15,23,42,0.06);'>\n\n{t['text']}\n\n*(Source: Page {t['page']})*</div>\n\n"
        image_md = ""
        if final_rel_imgs:
            image_md += "\n\n### \U0001f5bc Visual Evidence\n<div style='display:flex;gap:15px;overflow-x:auto;padding-bottom:10px;'>"
            for im in final_rel_imgs:
                dc = im['caption'][:120] + "..." if len(im['caption']) > 120 else im['caption']
                image_md += f"<div style='flex:0 0 400px;background:#ffffff;padding:12px;border-radius:14px;border:1px solid #e2e8f0;box-shadow:0 8px 24px rgba(15,23,42,0.06);'><img src='data:image/jpeg;base64,{im['base64']}' style='width:100%;border-radius:10px;'><br><p style='font-size:12px;margin-top:8px;color:#64748b;'>Page {im['page']} | {dc}</p></div>"


        final_answer = ans + tables_md + image_md

        try:
            db.log_chat(query, final_answer, "PDF Document")
        except Exception:
            pass

        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": final_answer})
        return history, "", final_rel_imgs

    except Exception as e:
        traceback.print_exc()
        err = str(e)
        if "402" in err:
            msg = "❌ **Hugging Face Quota Exceeded.**"
        elif "429" in err:
            msg = "⚠️ **Rate Limit.** Try again in a moment."
        else:
            msg = f"❌ **System Error:** {err}"
        history.append({"role": "assistant", "content": msg})
        return history, "", []

# =============================================================
# UI
# =============================================================
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@400;600&display=swap');

:root {
  --bg-0: #07090f;
  --bg-1: #0b1020;
  --bg-2: #0f172a;
  --surface: rgba(17, 24, 39, 0.72);
  --surface-2: rgba(255, 255, 255, 0.04);
  --surface-hover: rgba(255, 255, 255, 0.08);
  --border: rgba(148, 163, 184, 0.16);
  --border-strong: rgba(148, 163, 184, 0.28);
  --text: #f1f5f9;
  --text-dim: #cbd5e1;
  --muted: #94a3b8;
  --primary: #6366f1;
  --primary-2: #8b5cf6;
  --accent: #22d3ee;
  --success: #22c55e;
  --warn: #f59e0b;
  --danger: #ef4444;
  --shadow-lg: 0 30px 80px -20px rgba(0,0,0,0.6);
  --shadow-glow: 0 0 0 1px rgba(99,102,241,0.35), 0 20px 60px -12px rgba(99,102,241,0.45);
  --radius-lg: 20px;
  --radius-md: 14px;
  --radius-sm: 10px;
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
  font-feature-settings: "cv11","ss01","ss03";
  -webkit-font-smoothing: antialiased;
}

.gradio-container {
  max-width: 100% !important;
  padding: 0 !important;
}

/* Subtle animated grid overlay */
.gradio-container::before {
  content: "";
  position: fixed; inset: 0;
  background-image:
    linear-gradient(rgba(148,163,184,0.04) 1px, transparent 1px),
    linear-gradient(90deg, rgba(148,163,184,0.04) 1px, transparent 1px);
  background-size: 42px 42px;
  pointer-events: none;
  z-index: 0;
  mask-image: radial-gradient(ellipse at center, black 40%, transparent 80%);
}

/* ===== Sidebar ===== */
#sidebar {
  background: linear-gradient(180deg, rgba(11,16,32,0.92), rgba(11,16,32,0.78)) !important;
  border-right: 1px solid var(--border) !important;
  padding: 24px 20px !important;
  min-height: 100vh !important;
  backdrop-filter: blur(20px) saturate(140%);
  -webkit-backdrop-filter: blur(20px) saturate(140%);
  position: relative;
  z-index: 1;
}

#sidebar::after {
  content: "";
  position: absolute; top: 0; right: 0; bottom: 0; width: 1px;
  background: linear-gradient(180deg, transparent, rgba(99,102,241,0.4), transparent);
}

.sidebar-label {
  color: var(--muted) !important;
  font-size: 10.5px !important;
  font-weight: 800 !important;
  letter-spacing: 0.14em !important;
  text-transform: uppercase !important;
  margin: 18px 0 10px !important;
  display: flex; align-items: center; gap: 8px;
}

/* ===== Main chat area ===== */
#main-chat {
  padding: 28px 32px !important;
  position: relative;
  z-index: 1;
}

/* ===== Chatbot ===== */
#chatbot {
  background: linear-gradient(180deg, rgba(15,23,42,0.6), rgba(15,23,42,0.4)) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-lg) !important;
  box-shadow: var(--shadow-lg) !important;
  overflow: hidden !important;
  backdrop-filter: blur(14px);
}

#chatbot, #chatbot * { color: var(--text) !important; }

#chatbot .message-wrap, #chatbot .message {
  animation: msgIn 0.35s cubic-bezier(.2,.8,.2,1);
}
@keyframes msgIn {
  from { opacity: 0; transform: translateY(6px); }
  to   { opacity: 1; transform: translateY(0); }
}

#chatbot .message.user {
  background: linear-gradient(135deg, var(--primary), var(--primary-2)) !important;
  color: #fff !important;
  border: 1px solid rgba(255,255,255,0.18) !important;
  border-radius: 18px 18px 4px 18px !important;
  box-shadow: 0 12px 30px -10px rgba(99,102,241,0.55) !important;
}

#chatbot .message.bot, #chatbot .message.assistant {
  background: rgba(255,255,255,0.05) !important;
  border: 1px solid var(--border) !important;
  border-radius: 18px 18px 18px 4px !important;
  backdrop-filter: blur(10px);
}

#chatbot .prose h1, #chatbot .prose h2, #chatbot .prose h3 {
  color: #fff !important;
  letter-spacing: -0.01em;
}
#chatbot .prose strong { color: var(--accent) !important; }
#chatbot .prose a { color: var(--accent) !important; text-decoration: none; border-bottom: 1px dashed currentColor; }

/* ===== Input bar ===== */
#input-container {
  background: linear-gradient(180deg, rgba(15,23,42,0.95), rgba(15,23,42,0.85)) !important;
  border: 1px solid var(--border-strong) !important;
  border-radius: 22px !important;
  padding: 10px 10px 10px 18px !important;
  margin-top: 18px !important;
  box-shadow: 0 20px 50px -20px rgba(0,0,0,0.6), inset 0 1px 0 rgba(255,255,255,0.04) !important;
  transition: border-color .2s, box-shadow .2s;
}
#input-container:focus-within {
  border-color: rgba(99,102,241,0.55) !important;
  box-shadow: var(--shadow-glow) !important;
}

#query-box textarea {
  background: transparent !important;
  border: none !important;
  color: var(--text) !important;
  font-size: 15px !important;
  font-weight: 500 !important;
  padding: 10px 6px !important;
  resize: none !important;
}
#query-box textarea::placeholder { color: var(--muted) !important; }

#send-btn {
  background: linear-gradient(135deg, var(--primary), var(--primary-2)) !important;
  color: #fff !important;
  border: none !important;
  border-radius: 14px !important;
  font-size: 18px !important;
  font-weight: 800 !important;
  height: 44px !important;
  min-width: 44px !important;
  box-shadow: 0 12px 28px -8px rgba(99,102,241,0.55) !important;
  transition: transform .15s ease, filter .15s ease, box-shadow .15s ease !important;
}
#send-btn:hover { transform: translateY(-1px) scale(1.03); filter: brightness(1.1); }
#send-btn:active { transform: translateY(0) scale(0.98); }

/* ===== Buttons ===== */
button {
  border-radius: 12px !important;
  font-weight: 700 !important;
  letter-spacing: 0.01em !important;
  transition: all .18s ease !important;
}
button.primary, button[variant="primary"] {
  background: linear-gradient(135deg, var(--primary), var(--primary-2)) !important;
  color: #fff !important; border: none !important;
  box-shadow: 0 10px 24px -10px rgba(99,102,241,0.6) !important;
}
button.primary:hover { transform: translateY(-1px); filter: brightness(1.08); }
button.secondary, button[variant="secondary"] {
  background: rgba(255,255,255,0.06) !important;
  color: var(--text) !important;
  border: 1px solid var(--border-strong) !important;
}
button.secondary:hover { background: rgba(255,255,255,0.10) !important; }

/* ===== File upload ===== */
.file-preview, .upload-container, [data-testid="file"] {
  background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.02)) !important;
  border: 1.5px dashed rgba(148,163,184,0.35) !important;
  border-radius: 16px !important;
  color: var(--text-dim) !important;
  transition: border-color .2s, background .2s;
}
.file-preview:hover, .upload-container:hover {
  border-color: rgba(99,102,241,0.55) !important;
  background: rgba(99,102,241,0.06) !important;
}

/* ===== Accordions ===== */
.gr-accordion, .accordion {
  background: rgba(255,255,255,0.04) !important;
  border: 1px solid var(--border) !important;
  border-radius: 14px !important;
  overflow: hidden !important;
  margin-top: 10px !important;
}
.gr-accordion summary, .accordion summary {
  padding: 12px 14px !important;
  font-weight: 700 !important;
  color: var(--text) !important;
}

/* ===== Status badges ===== */
.markdown:has(> p:only-child:first-child) { line-height: 1.5; }

/* ===== Tables (in chat answers) ===== */
#chatbot table {
  background: #ffffff !important;
  color: #0f172a !important;
  border-radius: 12px !important;
  overflow: hidden !important;
  border-collapse: separate !important;
  border-spacing: 0 !important;
  box-shadow: 0 10px 30px -10px rgba(0,0,0,0.4);
  margin: 10px 0 !important;
}
#chatbot table * { color: #0f172a !important; }
#chatbot thead { background: #eef2ff !important; }
#chatbot th, #chatbot td { padding: 10px 12px !important; border-bottom: 1px solid #e2e8f0 !important; }

/* ===== Code ===== */
code, pre {
  font-family: 'JetBrains Mono', monospace !important;
  background: rgba(2,6,23,0.7) !important;
  color: #e2e8f0 !important;
  border-radius: 10px !important;
  border: 1px solid var(--border) !important;
}
pre { padding: 14px !important; }

/* ===== Scrollbars ===== */
*::-webkit-scrollbar { width: 10px; height: 10px; }
*::-webkit-scrollbar-track { background: transparent; }
*::-webkit-scrollbar-thumb {
  background: linear-gradient(180deg, rgba(99,102,241,0.5), rgba(139,92,246,0.5));
  border-radius: 10px; border: 2px solid transparent; background-clip: padding-box;
}
*::-webkit-scrollbar-thumb:hover { background: linear-gradient(180deg, var(--primary), var(--primary-2)); background-clip: padding-box; }

/* ===== Inputs (general) ===== */
input, textarea, select {
  color: var(--text) !important;
}

/* ===== Misc text ===== */
label, span, p, h1, h2, h3, h4, h5, h6 { color: var(--text) !important; }
.markdown, .markdown * { color: var(--text-dim) !important; }
.markdown h1, .markdown h2, .markdown h3 { color: #fff !important; letter-spacing: -0.01em; }

/* ===== Loading shimmer ===== */
.progress-text, .eta-bar {
  background: linear-gradient(90deg, var(--primary), var(--accent), var(--primary-2)) !important;
  -webkit-background-clip: text !important;
  -webkit-text-fill-color: transparent !important;
  font-weight: 700 !important;
}

/* ===== Responsive ===== */
@media (max-width: 900px) {
  #sidebar { min-height: auto !important; padding: 16px !important; }
  #main-chat { padding: 16px !important; }
  #chatbot { height: 60vh !important; }
}
"""

def process_and_init(file_obj, progress=gr.Progress()):
    if not file_obj:
        return "⚠️ No file uploaded", [], [], None, None, "⚠️ Pending", [], "_Upload a PDF to see FAQs._"
    try:
        chunks, images = extract_pdf_comprehensive(file_obj.name, progress)
        progress(0.9, desc="Finalizing index...")
        model = get_embed_model()
        texts = [c["text"] for c in chunks]
        embs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False).astype("float32")
        bm25 = BM25Okapi([t.lower().split() for t in texts])
        try:
            with open("engine_cache.pkl", "wb") as f:
                pickle.dump({"chunks": chunks, "images": images, "embs": embs, "bm25": bm25}, f)
        except Exception:
            pass
        faq_text = generate_pdf_faq(chunks)
        return ("✅ Engine Ready", chunks, images, embs, bm25, "✔️ Active",
                [{"role": "assistant", "content": "✅ **System Initialized!** Ask about text, tables, or diagrams."}],
                faq_text)
    except Exception as e:
        traceback.print_exc()
        return (f"❌ Error: {str(e)}", [], [], None, None, "❌ Error",
                [{"role": "assistant", "content": f"❌ Error: {str(e)}"}],
                "_FAQ unavailable._")

def load_cached_engine():
    if os.path.exists("engine_cache.pkl"):
        try:
            with open("engine_cache.pkl", "rb") as f:
                d = pickle.load(f)
            faq_text = generate_pdf_faq(d["chunks"])
            return ("✅ Engine Ready (Cached)", d["chunks"], d["images"], d["embs"], d["bm25"], "✔️ Active",
                    [{"role": "assistant", "content": "👋 **Welcome back!** Ready to analyze your document."}],
                    faq_text)
        except Exception:
            pass
    return ("📤 Upload PDF", [], [], None, None, "⚠️ Pending",
            [{"role": "assistant", "content": "👋 **Hello!** Upload a PDF to begin."}],
            "_Upload a PDF to see FAQs._")

with gr.Blocks(css=CSS, title="Multimodal PDF Assistant") as demo:
    st_chunks, st_images, st_embs, st_bm25 = gr.State([]), gr.State([]), gr.State(None), gr.State(None)

    with gr.Row():
        with gr.Column(scale=1, elem_id="sidebar", min_width=280):
            new_chat_btn = gr.Button("➕ New Chat", variant="secondary")
            gr.Markdown("### 📄 Document", elem_classes="sidebar-label")
            pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"], container=False)
            process_btn = gr.Button("🚀 Process PDF", variant="primary")
            status_msg = gr.Markdown("Ready.")
            status_badge = gr.Markdown("")

            with gr.Accordion("📖 GUIDE", open=False):
                gr.Markdown(
                    "- **Q&A (Stage 2)**: ask 'what/how/why' for precise facts.\n"
                    "- **Visual (Stage 1)**: mention 'image/diagram/figure' to retrieve a visual.\n"
                    "- **Concept (Stage 3)**: type a topic for full extraction."
                )

            with gr.Accordion("❓ FAQ (from your PDF)", open=False):
                faq_box = gr.Markdown("_Upload a PDF to see FAQs._")

        with gr.Column(scale=4, elem_id="main-chat"):
            chatbot = gr.Chatbot(show_label=False, elem_id="chatbot", height=750, type="messages") \
                if "type" in gr.Chatbot.__init__.__code__.co_varnames else gr.Chatbot(show_label=False, elem_id="chatbot", height=750)
            with gr.Row(elem_id="input-container"):
                query_box = gr.Textbox(placeholder="Message PDF Assistant...", scale=10, container=False, elem_id="query-box")
                send_btn = gr.Button("▲", scale=1, elem_id="send-btn")

    new_chat_btn.click(lambda: ([], ""), None, [chatbot, query_box])

    demo.load(load_cached_engine, None,
              [status_msg, st_chunks, st_images, st_embs, st_bm25, status_badge, chatbot, faq_box])

    process_btn.click(process_and_init, [pdf_input],
                      [status_msg, st_chunks, st_images, st_embs, st_bm25, status_badge, chatbot, faq_box])

    def run_chat_flow(q, h, chunks, imgs, embs, bm25):
        if not q or not q.strip():
            return h, ""
        new_h, _, _ = chat_handler(q, h, chunks, imgs, embs, bm25)
        return new_h, ""

    send_btn.click(run_chat_flow, [query_box, chatbot, st_chunks, st_images, st_embs, st_bm25], [chatbot, query_box])
    query_box.submit(run_chat_flow, [query_box, chatbot, st_chunks, st_images, st_embs, st_bm25], [chatbot, query_box])

if __name__ == "__main__":
    db.init_db()
    demo.launch(server_name="0.0.0.0", server_port=7860)
