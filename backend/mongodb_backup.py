import json
import os
from datetime import datetime
from pymongo import MongoClient, UpdateOne

# =========================
# CONNECTION
# =========================

MONGO_URI = "mongodb://localhost:27017"
DB_NAME   = "college_chatbot_db"

def get_db():
    client = MongoClient(MONGO_URI)
    return client[DB_NAME]


# =========================
# SYNC CHUNKS
# =========================

def sync_chunks():
    """
    Reads meta.json → upserts all chunks to MongoDB.
    Uses source+text hash as unique key to avoid duplicates.
    """
    meta_path = "data/meta.json"
    if not os.path.exists(meta_path):
        print("[MONGO] meta.json not found. Skipping chunks sync.")
        return 0

    with open(meta_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    if not chunks:
        print("[MONGO] No chunks to sync.")
        return 0

    db         = get_db()
    collection = db["chunks"]

    operations = []
    for chunk in chunks:
        operations.append(
            UpdateOne(
                # Match by source + first 100 chars of text
                {
                    "source"      : chunk.get("source", ""),
                    "text_preview": chunk.get("text", "")[:100]
                },
                {
                    "$set": {
                        "text"      : chunk.get("text", ""),
                        "source"    : chunk.get("source", ""),
                        "type"      : chunk.get("type", "general"),
                        "crawl_id"  : chunk.get("crawl_id", ""),
                        "updated_at": datetime.now().isoformat()
                    },
                    "$setOnInsert": {
                        "text_preview": chunk.get("text", "")[:100],
                        "created_at"  : datetime.now().isoformat()
                    }
                },
                upsert=True
            )
        )

    if operations:
        result = collection.bulk_write(operations)
        print(f"[MONGO] Chunks synced: "
              f"{result.upserted_count} new, "
              f"{result.modified_count} updated")
        return len(chunks)

    return 0


# =========================
# SYNC CRAWL LOGS
# =========================

def sync_crawl_logs():
    """
    Reads crawl_log.json → upserts to MongoDB crawl_logs collection.
    """
    log_path = "data/crawl_log.json"
    if not os.path.exists(log_path):
        print("[MONGO] crawl_log.json not found. Skipping.")
        return 0

    with open(log_path, "r", encoding="utf-8") as f:
        crawls = json.load(f)

    if not crawls:
        print("[MONGO] No crawl logs to sync.")
        return 0

    db         = get_db()
    collection = db["crawl_logs"]

    operations = []
    for crawl in crawls:
        operations.append(
            UpdateOne(
                {"crawl_id": crawl["crawl_id"]},
                {
                    "$set": {
                        "crawl_id"    : crawl["crawl_id"],
                        "subdomain"   : crawl.get("subdomain", ""),
                        "max_pages"   : crawl.get("max_pages", 0),
                        "pages_found" : crawl.get("pages_found", 0),
                        "chunks_added": crawl.get("chunks_added", 0),
                        "status"      : crawl.get("status", "unknown"),
                        "started_at"  : crawl.get("started_at", ""),
                        "finished_at" : crawl.get("finished_at", ""),
                        "synced_at"   : datetime.now().isoformat()
                    }
                },
                upsert=True
            )
        )

    if operations:
        result = collection.bulk_write(operations)
        print(f"[MONGO] Crawl logs synced: "
              f"{result.upserted_count} new, "
              f"{result.modified_count} updated")

    return len(crawls)


# =========================
# SYNC SCHEDULES
# =========================

def sync_schedules():
    """
    Reads schedule_log.json → upserts to MongoDB schedules collection.
    """
    log_path = "data/schedule_log.json"
    if not os.path.exists(log_path):
        print("[MONGO] schedule_log.json not found. Skipping.")
        return 0

    with open(log_path, "r", encoding="utf-8") as f:
        schedules = json.load(f)

    if not schedules:
        print("[MONGO] No schedules to sync.")
        return 0

    db         = get_db()
    collection = db["schedules"]

    operations = []
    for schedule in schedules:
        operations.append(
            UpdateOne(
                {"job_id": schedule["job_id"]},
                {
                    "$set": {
                        "job_id"     : schedule["job_id"],
                        "subdomain"  : schedule.get("subdomain", ""),
                        "frequency"  : schedule.get("frequency", ""),
                        "max_pages"  : schedule.get("max_pages", 100),
                        "created_at" : schedule.get("created_at", ""),
                        "synced_at"  : datetime.now().isoformat()
                    }
                },
                upsert=True
            )
        )

    if operations:
        result = collection.bulk_write(operations)
        print(f"[MONGO] Schedules synced: "
              f"{result.upserted_count} new, "
              f"{result.modified_count} updated")

    return len(schedules)


# =========================
# SYNC CHAT LOGS
# =========================

def log_chat(question: str, answer: str, intent: str = "general"):
    """
    Logs every chatbot query + answer to MongoDB.
    Called from chatbot.py after every response.
    """
    try:
        db         = get_db()
        collection = db["chat_logs"]
        collection.insert_one({
            "question"  : question,
            "answer"    : answer,
            "intent"    : intent,
            "timestamp" : datetime.now().isoformat(),
            "date"      : datetime.now().strftime("%Y-%m-%d")
        })
    except Exception as e:
        print(f"[MONGO] Chat log error: {e}")


# =========================
# FULL SYNC
# =========================

def sync_all():
    """
    Syncs everything — chunks, crawl logs, schedules.
    Call this after any crawl or index rebuild.
    """
    print("\n[MONGO] Starting full sync...")
    print(f"[MONGO] Connected to: {MONGO_URI}/{DB_NAME}")

    chunks_count   = sync_chunks()
    crawls_count   = sync_crawl_logs()
    schedule_count = sync_schedules()

    print(f"[MONGO] Sync complete:")
    print(f"  Chunks    : {chunks_count}")
    print(f"  Crawls    : {crawls_count}")
    print(f"  Schedules : {schedule_count}")
    return {
        "chunks"   : chunks_count,
        "crawls"   : crawls_count,
        "schedules": schedule_count
    }


# =========================
# TEST CONNECTION
# =========================

def test_connection():
    """Tests MongoDB connection."""
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
        client.server_info()
        print(f"[MONGO] ✅ Connected to {MONGO_URI}")
        print(f"[MONGO] Database: {DB_NAME}")
        db = client[DB_NAME]
        for name in db.list_collection_names():
            count = db[name].count_documents({})
            print(f"  {name}: {count} documents")
        return True
    except Exception as e:
        print(f"[MONGO] ❌ Connection failed: {e}")
        return False


# =========================
# RUN DIRECTLY
# =========================

if __name__ == "__main__":
    print("MAC College RAG — MongoDB Sync Tool")
    print("=" * 40)

    if test_connection():
        sync_all()
    else:
        print("Fix MongoDB connection first.")