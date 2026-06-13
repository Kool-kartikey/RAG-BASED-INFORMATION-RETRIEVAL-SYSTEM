import json
import faiss
import numpy as np
import os
import time
from sentence_transformers import SentenceTransformer

META_FILE  = "data/meta.json"
INDEX_FILE = "data/faiss.index"

print("[INDEXER] Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")


def rebuild_index() -> dict:
    """
    Rebuilds FAISS index from current meta.json.
    Called after crawl or chunk deletion.

    Returns:
        {
            "chunks_indexed": int,
            "time_taken": str
        }
    """
    start = time.time()

    if not os.path.exists(META_FILE):
        raise FileNotFoundError(f"{META_FILE} not found. Run crawler first.")

    print("[INDEXER] Loading chunks from meta.json...")
    with open(META_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    if not chunks:
        raise ValueError("meta.json is empty. Nothing to index.")

    texts = [c["text"] for c in chunks]
    print(f"[INDEXER] Embedding {len(texts)} chunks...")

    embeddings = model.encode(
        texts,
        show_progress_bar = True,
        normalize_embeddings = False
    )

    dim   = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings, dtype="float32"))

    faiss.write_index(index, INDEX_FILE)

    elapsed = round(time.time() - start, 2)
    print(f"[INDEXER] Done. {len(chunks)} chunks indexed in {elapsed}s")

    return {
        "chunks_indexed": len(chunks),
        "time_taken"    : f"{elapsed}s"
    }


if __name__ == "__main__":
    result = rebuild_index()
    print(f"\nIndex rebuilt: {result['chunks_indexed']} chunks in {result['time_taken']}")