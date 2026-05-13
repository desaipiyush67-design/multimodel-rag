"""
Backend API — Multimodal PDF Assistant  (Flask version)

Run (dev):    python app.py
              # serves on http://0.0.0.0:8000

Run (prod):   gunicorn -w 1 -b 0.0.0.0:8000 app:app
              # IMPORTANT: use 1 worker — ENGINE state and loaded ML models
              # are kept in-process. Multiple workers would each load their
              # own copy of the models and would not share the uploaded PDF.

All PDF parsing / retrieval / LLM logic from the FastAPI version is preserved
verbatim. Only the HTTP layer has been swapped from FastAPI to Flask.
Endpoints and request/response shapes are identical, so the existing
Streamlit frontend (frontend.py) works without any change.
"""

import os
import re
import io
import time
import base64
import traceback
import pickle
import concurrent.futures
import difflib
import tempfile
import uuid
from typing import List, Dict, Optional

import fitz  # PyMuPDF
import pdfplumber
import numpy as np
from PIL import Image

import torch
import ollama

from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from transformers import BlipProcessor, BlipForConditionalGeneration

from flask import Flask, request, jsonify, abort
from flask_cors import CORS

import db  # SQLite Logging Module


# Lightweight no-op progress callable (replaces Streamlit progress).
class _NoopProgress:
    def __call__(self, *args, **kwargs):
        pass
    def __getattr__(self, _):
        return self.__call__


# =============================================================
# CONFIGURATION — LOCAL OLLAMA  (TUNED FOR LARGE PDFs ~300 pages)
# =============================================================
OLLAMA_MODEL = "qwen2.5:1.5b"
CAPTION_MODEL = "Salesforce/blip-image-captioning-base"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

TOP_K_CHUNKS = 20
RERANK_TOP_N = 12

MAX_IMAGES_TO_CAPTION = 220
CAPTION_WORKERS = 2
EMBED_BATCH_SIZE = 64

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CACHE_VERSION = "v22-smart-verifier"

_models = {}


def _load_embed():
    print("[INIT] Loading Embedding Model ...")
    return SentenceTransformer(EMBED_MODEL_NAME, device=DEVICE)

def _load_rerank():
    print("[INIT] Loading Reranker Model ...")
    return CrossEncoder(RERANK_MODEL_NAME, device=DEVICE, max_length=384)

def _load_blip():
    print("[INIT] Loading Local BLIP ...")
    processor = BlipProcessor.from_pretrained(CAPTION_MODEL)
    model = BlipForConditionalGeneration.from_pretrained(CAPTION_MODEL).to(DEVICE)
    model.eval()
    return processor, model

def get_embed_model():
    if "embed" not in _models:
        _models["embed"] = _load_embed()
    return _models["embed"]

def get_rerank_model():
    if "rerank" not in _models:
        _models["rerank"] = _load_rerank()
    return _models["rerank"]

def get_blip():
    if "blip_processor" not in _models:
        p, m = _load_blip()
        _models["blip_processor"] = p
        _models["blip_model"] = m
    return _models["blip_processor"], _models["blip_model"]

def warmup_models():
    try:
        print("[WARMUP] Loading models...")
        get_embed_model().encode(["warmup"], show_progress_bar=False)
        get_rerank_model().predict([["warmup", "warmup"]])
        print("[WARMUP] Done.")
    except Exception as e:
        print("[WARMUP] Skipped:", e)

def ollama_generate(messages, max_tokens=700, temperature=0.0):
    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=messages,
        options={
            "temperature": temperature,
            "num_predict": max_tokens,
            "num_ctx": 4096,
            "top_p": 0.9,
            "repeat_penalty": 1.1,
        }
    )
    return response["message"]["content"].strip()

# =============================================================
# DYNAMIC FAQ GENERATION
# =============================================================
def _parse_faq_questions(raw: str) -> list:
    questions = []
    question_starters = (
        "what", "how", "where", "why", "when", "who", "which",
        "does", "do", "is", "are", "can", "should", "would",
        "will", "explain", "describe", "list", "define", "name"
    )
    for line in raw.split("\n"):
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^\*+\s*", "", line)
        line = re.sub(r"^Q\s*\d+[:.\)\-]\s*", "", line, flags=re.I)
        line = re.sub(r"^Question\s*\d+[:.\)\-]\s*", "", line, flags=re.I)
        line = re.sub(r"^\d+[\.\)]\s*", "", line)
        line = re.sub(r"^[-•·]\s*", "", line)
        line = line.strip("* ").strip()
        if not line:
            continue
        if len(line) < 8 or len(line) > 280:
            continue
        if line.endswith("?"):
            questions.append(line)
        else:
            first_word = line.split()[0].lower().strip(".,;:")
            if first_word in question_starters:
                line = line.rstrip(".!,;:") + "?"
                questions.append(line)
        if len(questions) == 5:
            break
    return questions


def generate_pdf_faq_questions(chunks):
    try:
        if not chunks:
            return []
        text_parts = []
        for c in chunks:
            if isinstance(c, dict):
                t = c.get("text", "")
                if isinstance(t, str) and t.strip():
                    text_parts.append(t.strip())
        if not text_parts:
            return []
        sample = text_parts[:: max(1, len(text_parts) // 8)][:8]
        context = "\n\n---\n\n".join(sample)[:6000]

        prompt = (
            "You are an expert at writing FAQ questions for a technical PDF document.\n"
            "Read the DOCUMENT EXCERPTS below and write EXACTLY 5 standalone questions that a reader might ask.\n\n"
            "STRICT RULES:\n"
            "- Output ONLY the 5 questions. NO answers. NO explanations. NO numbering. NO bullet points.\n"
            "- Each question on its own line.\n"
            "- Each question must end with a question mark (?).\n"
            "- Each question must be specific to the content above (mention real terms/topics from it).\n"
            "- Do NOT output 'Q1:' or 'Question 1:' or any prefix.\n\n"
            f"DOCUMENT EXCERPTS:\n{context}\n\n"
            "Now write 5 questions, one per line, each ending with ?:"
        )

        all_questions = []
        for attempt in range(3):
            temp = 0.4 + attempt * 0.15
            raw = ollama_generate(
                [{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=temp,
            )
            parsed = _parse_faq_questions(raw)
            for q in parsed:
                if q.lower() not in {x.lower() for x in all_questions}:
                    all_questions.append(q)
                if len(all_questions) == 5:
                    break
            print(f"[FAQ] attempt {attempt+1}: got {len(parsed)} parsed, total unique = {len(all_questions)}")
            if len(all_questions) >= 5:
                break

        return all_questions[:5]
    except Exception as e:
        print(f"[FAQ generation error] {e}")
        return []

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


def pil_to_base64(img: Image.Image, max_px: int = 1200) -> str:
    buf = io.BytesIO()
    has_alpha = (img.mode in ("RGBA", "LA")) or (img.mode == "P" and "transparency" in img.info)

    if has_alpha:
        if img.mode != "RGBA":
            img = img.convert("RGBA")
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        out = bg
        fmt_ext = "png"
        out.thumbnail((max_px, max_px), Image.LANCZOS)
        out.save(buf, format="PNG", optimize=True)
    else:
        out = img if img.mode == "RGB" else img.convert("RGB")
        fmt_ext = "jpeg"
        out.thumbnail((max_px, max_px), Image.LANCZOS)
        out.save(buf, format="JPEG", quality=92, optimize=True)

    return f"{fmt_ext}|{base64.b64encode(buf.getvalue()).decode()}"

def caption_image_fast(pil_img: Image.Image) -> str:
    try:
        processor, model = get_blip()
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        inputs = processor(images=pil_img, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=40)
        return clean_text(processor.decode(output[0], skip_special_tokens=True))
    except Exception as e:
        print("[BLIP ERROR]", e)
        return "technical diagram or figure"

def merge_rects(rects, gap: float = 35):
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
        if rect.width < 40 or rect.height < 30:
            continue
        page_area = page.rect.width * page.rect.height
        rect_area = rect.width * rect.height
        if rect_area > page_area * 0.65:
            continue
        rects.append(rect)
    merged_rects = merge_rects(rects, gap=45)
    for idx, rect in enumerate(merged_rects):
        try:
            crop_rect = rect + (-25, -25, 25, 25)
            crop_rect = crop_rect & page.rect
            if crop_rect.width < 80 or crop_rect.height < 60:
                continue
            page_area = page.rect.width * page.rect.height
            crop_area = crop_rect.width * crop_rect.height
            if crop_area > page_area * 0.55:
                continue
            pix = page.get_pixmap(clip=crop_rect, dpi=200, alpha=False, colorspace=fitz.csRGB)
            pil_img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
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
# EXTRACTION  (memory-friendly for 300-page PDFs)
# =============================================================
def extract_pdf_comprehensive(path: str, progress=_NoopProgress()):
    all_chunks: List[Dict] = []
    images: List[Dict] = []
    progress(0.05, desc="Opening PDF...")
    doc = fitz.open(path)
    current_concept = "Intro/Overview"
    total_pages = len(doc)

    MAX_RASTER_IMAGES_PER_PAGE = 8
    MAX_VECTOR_VISUALS_PER_PAGE = 4
    HARD_IMAGE_CAP = 800

    for pn in range(total_pages):
        if pn % 5 == 0 or pn == total_pages - 1:
            progress(0.05 + (pn + 1) / total_pages * 0.65,
                     desc=f"Processing Page {pn+1}/{total_pages}...")
        page = doc[pn]

        try:
            tabs = page.find_tables()
            for tab in tabs:
                md = table_to_markdown(tab.extract())
                if len(md) > 20:
                    all_chunks.append({
                        "page": pn + 1, "text": f"### [TABLE DATA]\n{md}",
                        "type": "table", "concept": current_concept,
                    })
        except Exception:
            pass

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
                "page": pn + 1, "text": " ".join(chunk_words),
                "type": "text", "concept": current_concept,
            })

        page_img_count = 0
        try:
            for img_info in page.get_images(full=True):
                if page_img_count >= MAX_RASTER_IMAGES_PER_PAGE:
                    break
                if len(images) >= HARD_IMAGE_CAP:
                    break
                xref = img_info[0]
                smask = img_info[1]
                try:
                    pix = fitz.Pixmap(doc, xref)
                    if pix.colorspace is None or pix.colorspace.n not in (3,):
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                    if smask:
                        try:
                            mask_pix = fitz.Pixmap(doc, smask)
                            pix = fitz.Pixmap(pix, mask_pix)
                        except Exception:
                            pass
                    png_bytes = pix.tobytes("png")
                    pil_img = Image.open(io.BytesIO(png_bytes))
                    if pil_img.mode not in ("RGB", "RGBA"):
                        pil_img = pil_img.convert("RGBA" if "A" in pil_img.mode else "RGB")
                except Exception as e:
                    print(f"[IMG decode skip xref={xref}] {e}")
                    continue

                if pil_img.width < 50 or pil_img.height < 50:
                    continue
                check_img = pil_img.convert("RGB") if pil_img.mode != "RGB" else pil_img
                if np.array(check_img).std() < 2:
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
                page_img_count += 1
        except Exception:
            pass

        try:
            if len(images) < HARD_IMAGE_CAP:
                vector_visuals = extract_vector_visuals_from_page(
                    page=page, page_no=pn + 1, current_concept=current_concept
                )
                vv_count = 0
                for vv in vector_visuals:
                    if vv_count >= MAX_VECTOR_VISUALS_PER_PAGE:
                        break
                    if len(images) >= HARD_IMAGE_CAP:
                        break
                    duplicate = False
                    for im in images:
                        if im.get("page") == vv.get("page") and im.get("caption") == vv.get("caption"):
                            duplicate = True
                            break
                    if not duplicate:
                        images.append(vv)
                        vv_count += 1
        except Exception:
            pass

    progress(0.78, desc="Finalizing tables...")
    try:
        with pdfplumber.open(path) as pdf:
            for idx, p in enumerate(pdf.pages):
                for tbl in p.extract_tables() or []:
                    md = table_to_markdown(tbl)
                    if len(md) > 30 and not any(md[:50] in c["text"] for c in all_chunks if c["type"] == "table"):
                        all_chunks.append({
                            "page": idx + 1, "text": f"### [TABLE DATA]\n{md}",
                            "type": "table", "concept": "Appendix",
                        })
    except Exception:
        pass

    def process_image(img_info):
        try:
            cap = caption_image_fast(img_info["image"])
            if cap:
                img_info['caption'] = f"{img_info['caption']} | AI View: {cap}"
        except Exception:
            pass
        return img_info

    if images:
        progress(0.88, desc=f"Captioning up to {MAX_IMAGES_TO_CAPTION} diagrams with BLIP...")
        images_to_caption = images[:MAX_IMAGES_TO_CAPTION]
        images_not_captioned = images[MAX_IMAGES_TO_CAPTION:]
        with concurrent.futures.ThreadPoolExecutor(max_workers=CAPTION_WORKERS) as executor:
            images_to_caption = list(executor.map(process_image, images_to_caption))
        images = images_to_caption + images_not_captioned
        for im in images:
            all_chunks.append({
                "page": im['page'], "text": f"### [TECHNICAL DIAGRAM]\n{im['caption']}",
                "type": "image_meta", "concept": im['concept'],
            })
            im.pop('image', None)

    doc.close()
    progress(1.0, desc="Extraction Complete ✓")
    return all_chunks, images

# =============================================================
# HYBRID RETRIEVAL
# =============================================================
def hybrid_retrieve(query: str, chunks: List[Dict], embs: np.ndarray, bm25: BM25Okapi, top_k: int = 20):
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
        doc_vocab = set()
        doc_case_map = {}
        for c in chunks:
            doc_vocab.add(c['concept'].lower())
            for w in re.findall(r'\b[A-Za-z]{3,}\b', c.get('concept', '')):
                doc_case_map.setdefault(w.lower(), w)
            for word in re.findall(r'\b[a-zA-Z]{3,}\b', c['text'].lower()):
                doc_vocab.add(word)
                if len(doc_vocab) > 60000:
                    break
            for w in re.findall(r'\b[A-Z]{2,}\b', c['text']):
                doc_case_map.setdefault(w.lower(), w)
            if len(doc_vocab) > 60000:
                break

        corrected_words = []
        vocab_list = list(doc_vocab)
        for word in query.lower().split():
            clean_w = word.strip(".,?!;:'\"()[]{}<>")
            if clean_w in doc_vocab or len(clean_w) <= 4:
                corrected_words.append(doc_case_map.get(clean_w, clean_w))
                continue
            matches = difflib.get_close_matches(clean_w, vocab_list, n=1, cutoff=0.9)
            if matches:
                corrected_words.append(doc_case_map.get(matches[0], matches[0]))
            else:
                corrected_words.append(clean_w)
        search_query = " ".join(corrected_words)

        query_lower = search_query.lower()
        query_words_set = set(query_lower.split())
        visual_keywords = {"picture", "image", "diagram", "figure", "photo", "show", "visual", "pic", "draw", "schematic", "map", "chart", "graph"}
        question_keywords = {"what", "how", "where", "why", "when", "who", "which", "explain", "advantages", "disadvantages", "adv", "define", "describe", "difference", "compare"}
        table_query_words = {
            "table", "chart", "specification", "specifications", "spec", "specs",
            "terminal", "terminals", "quick", "checks", "fault", "batching",
            "flowmeter", "rate", "product", "relay", "modbus", "troubleshooting"
        }
        is_table_like_query = any(w in query_lower for w in table_query_words)

        is_stage1_visual = any(vk in query_words_set for vk in visual_keywords)
        is_stage2_question = any(qk in query_words_set for qk in question_keywords) or "?" in query_lower or is_table_like_query

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
            max_context = 8
            system_prompt = (
                "You are a STRICT document-grounded PDF assistant. Use ONLY the provided CONTEXT.\n"
                "Do not use memory, training knowledge, assumptions, or information from other documents.\n\n"
                "CRITICAL RULES:\n"
                "1. Every fact in the answer must be directly supported by the provided CONTEXT.\n"
                "2. If the exact answer is not clearly present in the CONTEXT, say exactly: No info in document.\n"
                "3. Do NOT invent product names, model numbers, relay names, register names, constants, capacities, or specifications.\n"
                "4. Do NOT mix information from different tables, pages, products, manuals, or sections.\n"
                "5. If the user asks about a table, answer ONLY from the table whose title/heading matches the user question.\n"
                "6. If multiple unrelated tables are present in CONTEXT, ignore all tables that do not match the asked topic.\n"
                "7. For product specifications, only report rows that appear under the Product specifications table.\n"
                "8. Copy numbers, units, names, headings, and technical terms exactly as written in the CONTEXT.\n"
                "9. Cite [Page X] after every bullet or fact.\n"
                "10. Keep the answer concise and do not add explanations beyond the document.\n\n"
                "ANSWER FORMAT:\n"
                "- Use bullet points for multiple facts.\n"
                "- If answering from a table, preserve the table topic and row meaning.\n"
                "- If unsure, say: No info in document."
            )
        else:
            mode = "STAGE3_GENERAL"
            max_context = 12
            system_prompt = (
                "You are a strict technical assistant. The user gave a TOPIC NAME and wants "
                "EVERYTHING the document says about it. Use ONLY the provided CONTEXT.\n\n"
                "MANDATORY STRUCTURE — produce ALL of these sections that apply:\n"
                "**Definition / Description:** 1-2 sentences quoting the document's definition.\n"
                "**Purpose / Use:** what is it used for, where it is applied.\n"
                "**Components / Units:** list EVERY component or treatment unit mentioned, with any specs.\n"
                "**Design Criteria / Parameters:** list EVERY numerical value, dimension, retention time, "
                "percentage, capacity, etc. that the document gives for THIS topic.\n"
                "**Examples / Locations:** real-world installations or examples named.\n"
                "**Advantages / Disadvantages:** any pros or cons mentioned.\n"
                "**Other Notes:** anything else the document says about the topic.\n\n"
                "STRICT RULES:\n"
                "1. BE EXHAUSTIVE. Do NOT skip details. If the document lists 5 components, list all 5. "
                "If it gives retention times for each, include them all.\n"
                "2. CRITICAL — TOPIC BOUNDARY: Only use facts from chunks whose section heading or main "
                "subject is THIS topic. If a chunk in the CONTEXT discusses a DIFFERENT system (e.g. "
                "twin drain values when asked about DEWATS), IGNORE it entirely. Do not blend systems.\n"
                "3. Numbers, units, and names must be COPIED VERBATIM from the context. Never round, "
                "never paraphrase a number.\n"
                "4. Cite [Page X] after every fact or bullet.\n"
                "5. If the topic is not in the context, output exactly: 'No info in document.'\n"
                "6. NO OUTSIDE KNOWLEDGE. NO HALLUCINATION. NO MIXING TOPICS."
            )

        top_chunks = hybrid_retrieve(search_query, chunks, embs, bm25, top_k=25)
        reranker = get_rerank_model()
        text_pairs = [[search_query, c['text'][:600]] for c in top_chunks]
        text_scores = reranker.predict(text_pairs)
        scored_chunks = sorted(zip(top_chunks, text_scores), key=lambda x: x[1], reverse=True)

        STOP = {"what","is","the","a","an","of","and","in","to","for","with","on","by","at",
                "are","tell","me","about","please","explain","describe","define","show",
                "give","do","does","can","you","i","my","our","this","that","these","those",
                "how","why","when","where","which","who","whom","whose","used","use",
                "especially","particularly","mainly","also","there","they","them","its",
                "be","been","was","were","has","have","had","will","would","could","should"}
        raw_key_terms = [w for w in re.findall(r"[a-zA-Z]{3,}", search_query.lower()) if w not in STOP]
        key_terms = []
        for w in raw_key_terms:
            if w not in key_terms:
                key_terms.append(w)
        query_phrase = " ".join(key_terms)

        is_table_query = is_table_like_query

        generic_spec_words = {
            "product", "products", "specification", "specifications", "spec", "specs",
            "table", "tables", "chart", "charts", "data", "datasheet", "datasheets"
        }
        non_generic_terms = [t for t in key_terms if t not in generic_spec_words]
        is_generic_spec_query = (
            is_table_query
            and len(key_terms) > 0
            and len(non_generic_terms) == 0
        )

        def normalized_text(value):
            value = str(value or "").lower()
            value = re.sub(r"[^a-z0-9]+", " ", value)
            return re.sub(r"\s+", " ", value).strip()

        def chunk_search_blob(chunk):
            return normalized_text(f"{chunk.get('concept', '')} {chunk.get('text', '')}")

        def phrase_matches_blob(blob):
            if not query_phrase:
                return False
            phrase_re = r"\b" + r"\s+".join(re.escape(t) + r"s?" for t in key_terms) + r"\b"
            return re.search(phrase_re, blob) is not None

        def term_variants(term):
            variants = {term}
            if term.endswith("ies") and len(term) > 4:
                variants.add(term[:-3] + "y")
            if term.endswith("es") and len(term) > 4:
                variants.add(term[:-2])
            if term.endswith("s") and len(term) > 3:
                variants.add(term[:-1])
            else:
                variants.add(term + "s")
                variants.add(term + "es")
            return variants

        def key_term_hits(chunk):
            if not key_terms:
                return 1
            blob = chunk_search_blob(chunk)
            hits = 0
            for t in key_terms:
                if any(re.search(r"\b" + re.escape(v) + r"\b", blob) for v in term_variants(t)):
                    hits += 1
            return hits

        def chunk_heading_matches(chunk):
            if not key_terms:
                return True
            concept_l = normalized_text(chunk.get("concept", ""))
            return phrase_matches_blob(concept_l) or any(
                any(re.search(r"\b" + re.escape(v) + r"\b", concept_l) for v in term_variants(t))
                for t in key_terms
            )

        def lexical_score(chunk):
            blob = chunk_search_blob(chunk)
            score = key_term_hits(chunk)
            if phrase_matches_blob(blob):
                score += 4
            if chunk_heading_matches(chunk):
                score += 3
            if chunk.get("type") in {"table", "image_meta"}:
                score += 1
            return score

        def table_topic_matches(chunk):
            if not is_table_query or not key_terms:
                return True

            concept_l = normalized_text(chunk.get("concept", ""))
            text_l_raw = str(chunk.get("text", ""))
            text_l = text_l_raw.lower()
            blob = normalized_text(f"{chunk.get('concept', '')} {chunk.get('text', '')}")

            if is_generic_spec_query:
                exact_anchor = (
                    re.search(r"\bproduct\s+specifications?\b", concept_l) is not None
                    or re.search(r"\bproduct\s+specifications?\b", blob[:500]) is not None
                )
                has_spec_content = (
                    chunk.get("type") == "table"
                    or "parameter" in text_l
                    or "nominal" in text_l
                    or re.search(r"\d+\s*(?:mm|kg/cm|bar|°c|˚c|m/s|psi|hz|v|a|w)\b", text_l) is not None
                )
                return exact_anchor and has_spec_content

            hits = sum(1 for t in key_terms if re.search(r"\b" + re.escape(t) + r"s?\b", blob))
            required_hits = max(1, min(len(key_terms), 2))
            return hits >= required_hits

        if is_table_query:
            scored_chunks = [(c, s) for c, s in scored_chunks if table_topic_matches(c)]

        if mode == "STAGE2_QUESTION":
            RERANK_FLOOR = -1.5
            text_min = max(1, len(key_terms) // 2) if key_terms else 0
        elif mode == "STAGE3_GENERAL":
            RERANK_FLOOR = -2.0
            text_min = 1
        else:
            RERANK_FLOOR = -3.0
            text_min = 1 if key_terms else 0

        def chunk_passes(c, s):
            if not key_terms:
                return True
            lx = lexical_score(c)
            if lx >= max(2, min(len(key_terms), 2)):
                return True
            return s > RERANK_FLOOR and (chunk_heading_matches(c) or key_term_hits(c) >= text_min)

        filtered = [(c, s) for c, s in scored_chunks if chunk_passes(c, s)]

        fallback_chunks = []
        if key_terms:
            fallback_candidates = [(c, lexical_score(c)) for c in chunks]
            if is_table_query:
                fallback_candidates = [(c, lx) for c, lx in fallback_candidates if table_topic_matches(c)]
            fallback_candidates = [(c, lx) for c, lx in fallback_candidates if lx >= max(2, min(len(key_terms), 2))]
            fallback_candidates.sort(key=lambda x: (x[1], -int(x[0].get("page", 10**9))), reverse=True)
            fallback_chunks = [c for c, _ in fallback_candidates[:max_context * 3]]
            seen = {(id(c), c.get("page"), c.get("type"), c.get("text", "")[:80]) for c, _ in filtered}
            for c in fallback_chunks:
                key = (id(c), c.get("page"), c.get("type"), c.get("text", "")[:80])
                if key not in seen:
                    filtered.append((c, 0.0))
                    seen.add(key)

        if is_table_query:
            filtered = [(c, s) for c, s in filtered if table_topic_matches(c)]

        topic_missing = len(filtered) == 0
        if topic_missing:
            final_top_chunks = []
        else:
            final_top_chunks = [c for c, _ in filtered[:max_context]]

        final_rel_imgs = []
        final_rel_tabs = []
        if not topic_missing:
            if images and mode in {"STAGE1_VISUAL", "STAGE3_GENERAL"}:
                img_pairs = [[search_query, im['caption']] for im in images]
                img_scores = reranker.predict(img_pairs)
                scored_imgs = sorted(zip(images, img_scores), key=lambda x: x[1], reverse=True)
                limit = 1 if mode == "STAGE1_VISUAL" else 10
                floor = 1.0 if mode == "STAGE1_VISUAL" else 0.2
                final_rel_imgs = [im for im, score in scored_imgs[:limit] if score > floor]

            if mode == "STAGE3_GENERAL":
                rel_tabs = [c for c in chunks if c['type'] == 'table']
                if rel_tabs:
                    tab_pairs = [[search_query, t['text'][:500]] for t in rel_tabs]
                    tab_scores = reranker.predict(tab_pairs)
                    scored_tabs = sorted(zip(rel_tabs, tab_scores), key=lambda x: x[1], reverse=True)
                    final_rel_tabs = [t for t, score in scored_tabs[:8] if score > -0.5]

        if topic_missing:
            context_str = "(No relevant passages were found in the document for this query.)"
        else:
            if mode == "STAGE2_QUESTION":
                char_limit = 6000
            elif mode == "STAGE3_GENERAL":
                char_limit = 16000
            else:
                char_limit = 9000
            context_parts = [f"--- [SOURCE: Page {c['page']} | Section: {c['concept']}] ---\n{c['text']}" for c in final_top_chunks]
            context_str = "\n\n".join(context_parts)[:char_limit]

        messages = [{"role": "system", "content": system_prompt}]
        messages.append({"role": "user", "content": f"CONTEXT:\n{context_str}\n\nQUESTION: {query}"})

        if topic_missing:
            ans = "No info in document."
        else:
            if mode == "STAGE2_QUESTION":
                llm_max = 450
            elif mode == "STAGE3_GENERAL":
                llm_max = 1200
            else:
                llm_max = 700
            ans = ollama_generate(messages, max_tokens=llm_max, temperature=0.0)

            if is_table_query and ans and "no info in document" not in ans.lower():
                context_l_for_guard = context_str.lower()
                hallucination_markers = [
                    "mitsubishi", "programmable controller", "input relay", "output relay",
                    "internal relay", "latch relay", "link relay", "annunciator",
                    "step relay", "data register", "special register", "file register",
                    "index register", "interrupt pointer", "module access device",
                    "cpu module", "din rail", "mac address", "ph sensor", "orp sensor",
                    "tds sensor", "rtd", "conductivity sensor"
                ]
                bad_markers = [m for m in hallucination_markers if m in ans.lower() and m not in context_l_for_guard]
                if bad_markers:
                    print(f"[TABLE GUARD] Rejected hallucinated answer markers: {bad_markers}")
                    ans = "No info in document."

            if mode == "STAGE2_QUESTION" and ans and "no info in document" not in ans.lower():
                cited_pages = set(re.findall(r"\[Page\s*(\d+)\]", ans, flags=re.I))
                verify_chunks = list(final_top_chunks)
                if cited_pages:
                    for c in chunks:
                        if str(c.get("page", "")) in cited_pages and c not in verify_chunks:
                            verify_chunks.append(c)
                verify_text = " ".join(c["text"] for c in verify_chunks).lower()

                answer_numbers = re.findall(r"\d+(?:[.,]\d+)*", ans)
                missing_numbers = []
                for n in answer_numbers:
                    n_low = n.lower()
                    if re.search(r"\[page\s*" + re.escape(n_low) + r"\]", ans, flags=re.I):
                        continue
                    if re.search(r"(?:figure|table|section|chapter|appendix|fig\.?)\s*" + re.escape(n_low),
                                 ans, flags=re.I):
                        continue
                    if re.search(r"(?<!\d)" + re.escape(n_low) + r"(?!\d)", verify_text):
                        continue
                    missing_numbers.append(n)

                if missing_numbers:
                    print(f"[VERIFIER] Numbers in answer NOT found in on-topic chunks: {missing_numbers}")
                    print(f"[VERIFIER] Verified against {len(verify_chunks)} chunks "
                          f"({len(verify_text)} chars). Cited pages: {cited_pages or 'none'}")
                    ans = (
                        "⚠️ **Could not verify answer against the document.**\n\n"
                        f"The model produced a number ({', '.join(missing_numbers)}) "
                        "that does not appear in the relevant source passages. "
                        "This may indicate a hallucination. Please rephrase your question "
                        "or check the document directly."
                    )

        tables_md = ""
        if final_rel_tabs:
            tables_md = "\n\n### 📊 Data Evidence\n"
            for t in final_rel_tabs:
                tables_md += f"\n{t['text']}\n\n*(Source: Page {t['page']})*\n"

        image_md = ""
        if final_rel_imgs:
            image_md += "\n\n### 🖼 Visual Evidence\n\n<div style='display:flex;flex-wrap:wrap;gap:14px;'>"
            for im in final_rel_imgs:
                dc = im['caption'][:120] + "..." if len(im['caption']) > 120 else im['caption']
                b64_data = im['base64']
                if "|" in b64_data and len(b64_data.split("|", 1)[0]) <= 5:
                    fmt_ext, b64_payload = b64_data.split("|", 1)
                else:
                    fmt_ext, b64_payload = "jpeg", b64_data
                image_md += (
                    "<div style='flex:1 1 280px;max-width:360px;background:#fff;border-radius:10px;padding:8px;'>"
                    f"<a href='data:image/{fmt_ext};base64,{b64_payload}' target='_blank'>"
                    f"<img src='data:image/{fmt_ext};base64,{b64_payload}' "
                    "style='width:100%;height:auto;border-radius:8px;display:block;' />"
                    "</a>"
                    f"<div style='font-size:12px;color:#0f172a;margin-top:6px;'>Page {im['page']} | {dc}</div>"
                    "</div>"
                )
            image_md += "</div>\n"

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
        if "connection refused" in err.lower() or "failed to connect" in err.lower():
            msg = "❌ **Ollama not running.** Open another terminal and run `ollama serve`."
        else:
            msg = f"❌ **System Error:** {err}"
        history.append({"role": "assistant", "content": msg})
        return history, "", []


# =============================================================
# ENGINE STATE  (in-process; one PDF loaded at a time)
# =============================================================
ENGINE = {
    "chunks": [],
    "images": [],
    "embs": None,
    "bm25": None,
    "status": "📤 Upload PDF",
    "badge": "⚠️ Pending",
    "faq": ["", "", "", "", ""],
}


def _build_engine_from_pdf(path: str, progress=_NoopProgress()):
    chunks, images = extract_pdf_comprehensive(path, progress)
    progress(0.92, desc="Building search index...")
    model = get_embed_model()
    texts = [c["text"] for c in chunks]
    embs = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=EMBED_BATCH_SIZE,
    ).astype("float32")
    bm25 = BM25Okapi([t.lower().split() for t in texts])
    try:
        with open("engine_cache.pkl", "wb") as f:
            pickle.dump({
                "version": CACHE_VERSION,
                "chunks": chunks, "images": images, "embs": embs, "bm25": bm25
            }, f)
    except Exception:
        pass

    questions = generate_pdf_faq_questions(chunks)
    while len(questions) < 5:
        questions.append("")
    return chunks, images, embs, bm25, questions[:5]


def _try_load_cache():
    if not os.path.exists("engine_cache.pkl"):
        return False
    try:
        with open("engine_cache.pkl", "rb") as f:
            d = pickle.load(f)
        if d.get("version") != CACHE_VERSION:
            try:
                os.remove("engine_cache.pkl")
            except Exception:
                pass
            return False
        ENGINE["chunks"] = d["chunks"]
        ENGINE["images"] = d["images"]
        ENGINE["embs"] = d["embs"]
        ENGINE["bm25"] = d["bm25"]
        ENGINE["status"] = "✅ Engine Ready (Cached)"
        ENGINE["badge"] = "✔️ Active"
        questions = generate_pdf_faq_questions(d["chunks"])
        while len(questions) < 5:
            questions.append("")
        ENGINE["faq"] = questions[:5]
        return True
    except Exception:
        return False


# =============================================================
# FLASK APP
# =============================================================
app = Flask(__name__)
# Allow ~200 MB PDF uploads (matches the "300-page PDF" comment above).
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024

# Permissive CORS — same policy as the FastAPI version.
CORS(app, resources={r"/*": {"origins": "*"}})


def _run_startup_once():
    """Replaces FastAPI's @app.on_event('startup')."""
    try:
        db.init_db()
    except Exception:
        pass
    try:
        warmup_models()
    except Exception:
        pass
    _try_load_cache()


# Run startup work exactly once, at import time, so it happens
# whether the app is launched via `python app.py` or via gunicorn.
_STARTUP_DONE = False
if not _STARTUP_DONE:
    _run_startup_once()
    _STARTUP_DONE = True


@app.get("/status")
def status():
    return jsonify({
        "status": ENGINE["status"],
        "badge": ENGINE["badge"],
        "n_chunks": len(ENGINE["chunks"]),
        "n_images": len(ENGINE["images"]),
        "faq": ENGINE["faq"],
        "ready": ENGINE["embs"] is not None and len(ENGINE["chunks"]) > 0,
    })


@app.get("/faq")
def faq():
    return jsonify({"faq": ENGINE["faq"]})


@app.post("/process_pdf")
def process_pdf():
    if "file" not in request.files:
        return jsonify({"detail": "No file part named 'file' in the request."}), 400
    file = request.files["file"]
    if not file or not file.filename:
        return jsonify({"detail": "Empty file upload."}), 400
    if not file.filename.lower().endswith(".pdf"):
        return jsonify({"detail": "Only PDF files are accepted."}), 400

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    try:
        file.save(tmp.name)
        tmp.close()
        chunks, images, embs, bm25, questions = _build_engine_from_pdf(tmp.name)
        ENGINE["chunks"] = chunks
        ENGINE["images"] = images
        ENGINE["embs"] = embs
        ENGINE["bm25"] = bm25
        ENGINE["status"] = "✅ Engine Ready"
        ENGINE["badge"] = "✔️ Active"
        ENGINE["faq"] = questions
        return jsonify({
            "status": ENGINE["status"],
            "badge": ENGINE["badge"],
            "n_chunks": len(chunks),
            "n_images": len(images),
            "faq": questions,
            "message": f"✅ **System Initialized!** Indexed {len(chunks)} chunks and {len(images)} visuals.",
        })
    except Exception as e:
        traceback.print_exc()
        ENGINE["status"] = f"❌ Error: {e}"
        ENGINE["badge"] = "❌ Error"
        return jsonify({"detail": str(e)}), 500
    finally:
        try:
            os.remove(tmp.name)
        except Exception:
            pass


@app.post("/chat")
def chat():
    data = request.get_json(silent=True) or {}
    query = data.get("query")
    history = data.get("history")
    if not isinstance(query, str) or not query.strip():
        return jsonify({"detail": "Field 'query' is required and must be a non-empty string."}), 400
    if history is not None and not isinstance(history, list):
        return jsonify({"detail": "Field 'history' must be a list if provided."}), 400

    new_history, _, rel_imgs = chat_handler(
        query,
        history or [],
        ENGINE["chunks"],
        ENGINE["images"],
        ENGINE["embs"],
        ENGINE["bm25"],
    )
    return jsonify({"history": new_history, "n_visuals": len(rel_imgs)})


@app.post("/reset_chat")
def reset_chat():
    return jsonify({"ok": True})


# Friendly handler for oversize uploads, mirroring FastAPI's 413 behavior.
@app.errorhandler(413)
def _too_large(_e):
    return jsonify({"detail": "Uploaded file is too large."}), 413


if __name__ == "__main__":
    # Single-process dev server. For production use gunicorn with -w 1
    # so the ML models and ENGINE are loaded only once.
    app.run(host="0.0.0.0", port=8000, debug=False, threaded=True)
