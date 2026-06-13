"""
================================================================================
  MAC RAG CHATBOT — INDEX BUILDER (PDF-AWARE)
  Project   : Efficient Chatbot for Maharaja Agrasen College (MAC)
  Author    : Kartikey Tiwari, B.Sc. (Hons.) Electronics, VIII Sem
  Supervisor: Prof. Amit Pundir

  CHANGES FROM PREVIOUS VERSION:
    1. Binary URL guard — JPEG/image sources never enter the index
    2. PDF-aware chunking — respects page/section boundaries
    3. HTML dual chunking — preserved (250w + 80w)
    4. Chunk-level deduplication — eliminates exact duplicates
    5. kind field propagated from final_pages.json to meta.json
================================================================================
"""

import json
import re
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

INPUT_FILE = "data/final_pages.json"
INDEX_FILE = "data/faiss.index"
META_FILE  = "data/meta.json"

model = SentenceTransformer("all-MiniLM-L6-v2")

BINARY_EXTENSIONS = (
    ".jpeg", ".jpg", ".png", ".gif", ".webp",
    ".zip", ".rar", ".mp4", ".mp3", ".svg",
)

def is_binary_url(url: str) -> bool:
    return any(url.lower().split("?")[0].endswith(ext) for ext in BINARY_EXTENSIONS)

def is_pdf_url(url: str) -> bool:
    return url.lower().split("?")[0].endswith(".pdf")


# =========================
# URL CLASSIFICATION
# =========================

def classify_url(url: str) -> str:
    url = url.lower()
    if url.endswith(".pdf"):
        return "pdf"
    if "contact" in url:
        return "contact"
    if "about" in url:
        return "about"
    if "faculty" in url:
        return "faculty"
    if "department" in url:
        return "department"
    return "general"


# =========================
# CHUNKING
# =========================

def chunk_text_html(text: str) -> list:
    """
    Dual chunking for HTML: large context chunks (250w) + small fact chunks (80w).
    Deduplication applied within each page.
    """
    words  = text.split()
    chunks = []
    seen   = set()

    for size, min_w in [(250, 50), (80, 20)]:
        for i in range(0, len(words), size):
            chunk = " ".join(words[i:i + size])
            if len(chunk.split()) >= min_w and chunk not in seen:
                chunks.append(chunk)
                seen.add(chunk)

    return chunks


def chunk_text_pdf(text: str) -> list:
    """
    PDF-aware chunking: splits on double newlines (natural page/section
    boundaries preserved during extraction), then uses sliding window
    for oversized sections.
    Produces semantically denser chunks than blind word-count slicing.
    """
    chunks = []
    seen   = set()

    # Split on paragraph/page boundaries
    sections = [s.strip() for s in re.split(r"\n{2,}", text) if s.strip()]

    for section in sections:
        words = section.split()

        if len(words) < 20:
            continue

        if len(words) <= 250:
            if section not in seen:
                chunks.append(section)
                seen.add(section)
        else:
            # Sliding window with 50-word overlap
            for i in range(0, len(words), 200):
                chunk = " ".join(words[i:i + 250])
                if len(chunk.split()) >= 30 and chunk not in seen:
                    chunks.append(chunk)
                    seen.add(chunk)

    return chunks


# =========================
# BUILD INDEX
# =========================

def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_chunks  = []
    seen_chunks = set()     # global deduplication across all pages
    skipped_bin = 0
    skipped_dup = 0

    for item in data:
        url  = item.get("url", "")
        text = item.get("text", "").strip()
        kind = item.get("kind", "html")

        # Guard: skip binary image sources
        if is_binary_url(url):
            skipped_bin += 1
            continue

        # Choose chunking strategy by content type
        if kind == "pdf" or is_pdf_url(url):
            raw_chunks = chunk_text_pdf(text)
            chunk_type = "pdf"
        else:
            raw_chunks = chunk_text_html(text)
            chunk_type = classify_url(url)

        for chunk in raw_chunks:
            text_clean  = chunk.strip()
            fingerprint = text_clean.lower()

            if fingerprint in seen_chunks:
                skipped_dup += 1
                continue
            seen_chunks.add(fingerprint)

            all_chunks.append({
                "text"  : text_clean,
                "source": url,
                "type"  : chunk_type
            })

    print(f"\nPages processed      : {len(data)}")
    print(f"Binary pages skipped : {skipped_bin}")
    print(f"Duplicate chunks skip: {skipped_dup}")
    print(f"HTML chunks          : {sum(1 for c in all_chunks if c['type'] != 'pdf')}")
    print(f"PDF chunks           : {sum(1 for c in all_chunks if c['type'] == 'pdf')}")
    print(f"Clean chunks to embed: {len(all_chunks)}")

    texts = [c["text"] for c in all_chunks]

    print("\nGenerating embeddings...\n")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)

    dim   = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings, dtype="float32"))

    faiss.write_index(index, INDEX_FILE)

    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False)

    print(f"\n✅ Index built: {len(all_chunks)} clean chunks")
    print(f"   Saved: {INDEX_FILE}")
    print(f"   Saved: {META_FILE}")


if __name__ == "__main__":
    main()
