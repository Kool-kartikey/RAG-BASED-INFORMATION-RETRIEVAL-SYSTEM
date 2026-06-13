import json
import os
import uuid
from datetime import datetime

CRAWL_LOG_FILE = "data/crawl_log.json"


def _load_log() -> list:
    if not os.path.exists(CRAWL_LOG_FILE):
        return []
    with open(CRAWL_LOG_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_log(log: list):
    os.makedirs("data", exist_ok=True)
    with open(CRAWL_LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)


def create_crawl_entry(subdomain: str, max_pages: int) -> str:
    """Creates new crawl entry. Returns crawl_id."""
    crawl_id = str(uuid.uuid4())[:8].upper()
    entry = {
        "crawl_id"    : crawl_id,
        "subdomain"   : subdomain,
        "max_pages"   : max_pages,
        "pages_found" : 0,
        "chunks_added": 0,
        "status"      : "running",
        "started_at"  : datetime.now().isoformat(),
        "finished_at" : None,
    }
    log = _load_log()
    log.append(entry)
    _save_log(log)
    print(f"[CRAWL LOG] Created: {crawl_id} for {subdomain}")
    return crawl_id


def update_crawl_entry(crawl_id: str, pages_found: int,
                       chunks_added: int, status: str = "done"):
    """Updates crawl entry when finished."""
    log = _load_log()
    for entry in log:
        if entry["crawl_id"] == crawl_id:
            entry["pages_found"]  = pages_found
            entry["chunks_added"] = chunks_added
            entry["status"]       = status
            entry["finished_at"]  = datetime.now().isoformat()
            break
    _save_log(log)
    print(f"[CRAWL LOG] Updated: {crawl_id} — {pages_found} pages, {chunks_added} chunks")


def get_all_crawls() -> list:
    """Returns all crawls newest first."""
    log = _load_log()
    return sorted(log, key=lambda x: x["started_at"], reverse=True)


def get_crawl_by_id(crawl_id: str) -> dict:
    """Returns single crawl or None."""
    for entry in _load_log():
        if entry["crawl_id"] == crawl_id:
            return entry
    return None


def delete_crawl_by_id(crawl_id: str) -> int:
    """Deletes crawl + removes its chunks from meta.json."""
    log = _load_log()
    log = [e for e in log if e["crawl_id"] != crawl_id]
    _save_log(log)

    meta_path = "data/meta.json"
    if not os.path.exists(meta_path):
        return 0

    with open(meta_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    original      = len(chunks)
    chunks        = [c for c in chunks if c.get("crawl_id") != crawl_id]
    deleted_count = original - len(chunks)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)

    print(f"[CRAWL LOG] Deleted {crawl_id} — removed {deleted_count} chunks")
    return deleted_count