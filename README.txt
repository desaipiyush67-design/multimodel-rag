
# Multimodal PDF RAG Assistant

## Overview
A local multimodal Retrieval-Augmented Generation (RAG) system for analyzing technical PDF documents using:
- Text extraction
- Table extraction
- Diagram/image retrieval
- Visual captioning
- Semantic search
- Hybrid reranking
- Local Ollama LLM inference

Built with:
- Gradio
- Ollama
- SentenceTransformers
- BLIP
- PyMuPDF
- pdfplumber

---

# Features

## Text Question Answering
Ask factual questions directly from PDFs.

Examples:
- What is the principle of operation of the electromagnetic flowmeter?
- What are the storage instructions?

---

## Table Extraction & Retrieval
Supports:
- Product specifications
- Troubleshooting tables
- Terminal configuration tables
- Relay settings
- Communication settings
- Flow rate tables

---

## Visual / Diagram Retrieval
Supports:
- Flow diagrams
- Technical drawings
- Wiring diagrams
- Troubleshooting visuals
- Installation figures

The system:
- extracts visuals from PDFs
- captions them using BLIP
- reranks them against the query
- displays relevant visual evidence

---

# Architecture

## Components

| Component | Purpose |
|---|---|
| PyMuPDF | PDF parsing + image extraction |
| pdfplumber | Table extraction |
| SentenceTransformer | Embedding generation |
| BM25 | Keyword retrieval |
| CrossEncoder | Reranking |
| BLIP | Visual captioning |
| Ollama + Qwen | Answer generation |
| Gradio | UI |

---

# Models Used

## LLM
qwen2.5:1.5b

## Embedding Model
sentence-transformers/all-MiniLM-L6-v2

## Reranker
cross-encoder/ms-marco-MiniLM-L-6-v2

## Image Captioning
Salesforce/blip-image-captioning-base

---

# Installation

## Create Virtual Environment

Windows:
python -m venv venv
venv\Scripts\activate

---

## Install Requirements

pip install -r requirements.txt

---

# Install Ollama

https://ollama.com

Pull model:
ollama pull qwen2.5:1.5b

---

# Install BLIP Model

python -c "from transformers import BlipProcessor, BlipForConditionalGeneration; BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-base'); BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-base')"

---

# Run Application

Start Ollama:
ollama serve

Run app:
python app.py

Application URL:
http://127.0.0.1:7860

---

# Supported Query Types

## Stage 1 — Visual Queries
Examples:
- Show the Flow Tube Local Earthing diagram.

## Stage 2 — Factual Questions
Examples:
- What are the storage instructions?

## Stage 3 — Topic Extraction
Examples:
- DEWATS
- Product specifications
- Troubleshooting the transmitter section

---

# Large PDF Support

Optimized for:
- ~300 page PDFs
- Large technical manuals
- Heavy engineering documentation

Current limits:
- HARD_IMAGE_CAP = 800
- MAX_IMAGES_TO_CAPTION = 220

---

# Hallucination Prevention

Includes:
- strict document grounding
- table-topic filtering
- reranker validation
- lexical topic guard
- number verifier
- hallucination marker rejection

Fallback:
No info in document.

---

# GPU Support

Automatically detects CUDA availability.

Recommended:
- 4GB+ VRAM GPU

---

# UI

Built with:
- Gradio
- Dark glassmorphism theme
- Chat interface
- FAQ buttons
- Visual evidence viewer

---

# Notes

- Fully local deployment
- No cloud APIs required
- Designed for edge deployment
- Works offline after model download
- Suitable for industrial documentation QA
