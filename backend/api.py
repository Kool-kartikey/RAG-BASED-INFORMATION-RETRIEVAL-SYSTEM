import sys
import os
import json

sys.path.append(os.path.dirname(__file__))

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel
from typing import Optional, List
import secrets

from backend.retriever import search
from backend.llm import generate_answer
from chatbot import get_answer

from crawl_log import get_all_crawls, get_crawl_by_id, delete_crawl_by_id
from scheduler import start_scheduler, schedule_crawl_job, get_scheduled_jobs, remove_scheduled_job
from indexer import rebuild_index


# ─────────────────────────────────────────
# AUTH
# ─────────────────────────────────────────
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "mac2024"

security = HTTPBasic()

def verify_admin(credentials: HTTPBasicCredentials = Depends(security)):
    correct_user = secrets.compare_digest(credentials.username, ADMIN_USERNAME)
    correct_pass = secrets.compare_digest(credentials.password, ADMIN_PASSWORD)
    if not (correct_user and correct_pass):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid admin credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


# ─────────────────────────────────────────
# APP
# ─────────────────────────────────────────
app = FastAPI(
    title="MAC College RAG System API",
    version="3.0.0"
)

# FIX: allow_origins must be explicit (not wildcard) when using credentials
# (HTTP Basic Auth sends an Authorization header — browsers block wildcard + credentials)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080",
        "http://127.0.0.1:8080",
    ],
    allow_credentials=True,   # required for Basic Auth headers to pass
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]

class CrawlRequest(BaseModel):
    subdomain: str
    max_pages: Optional[int] = 100

class ScheduleRequest(BaseModel):
    subdomain: str
    frequency: str
    max_pages: Optional[int] = 100

class DeleteCrawlRequest(BaseModel):
    crawl_id: str

class DebugRequest(BaseModel):
    question: str
    top_k: int = 5


# ─────────────────────────────────────────
# MONGODB HELPER
# ─────────────────────────────────────────
def mongo_sync():
    try:
        from mongodb_backup import sync_all
        sync_all()
        print("[API] MongoDB sync complete.")
    except Exception as e:
        print(f"[API] MongoDB sync skipped: {e}")


# ─────────────────────────────────────────
# PUBLIC ROUTES
# ─────────────────────────────────────────
@app.get("/")
def root():
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return {"message": "MAC College RAG API running."}


@app.get("/college_banner.jpg")
def serve_banner():
    if os.path.exists("college_banner.jpg"):
        return FileResponse("college_banner.jpg", media_type="image/jpeg")
    raise HTTPException(status_code=404, detail="Banner not found")


@app.post("/ask", response_model=QueryResponse)
def ask(request: QueryRequest):
    question = request.question.strip()

    if not question:
        return QueryResponse(answer="Please ask a question.", sources=[])

    try:
        # Route through chatbot.py — guardrails + RAG in one call
        answer = get_answer(question)

        # Extract sources separately for the response model
        # (guardrail rejections have no sources)
        try:
            chunks = search(question, k=5)
            sources = list(set([
                c.get("source", "unknown") for c in chunks
            ]))
        except Exception:
            sources = []

        return QueryResponse(answer=answer, sources=sources)

    except Exception as e:
        print(f"[API ERROR] {e}")
        return QueryResponse(
            answer="Something went wrong. Please try again.",
            sources=[]
        )


@app.post("/debug")
async def debug_retrieve(request: DebugRequest):
    chunks = search(request.question, k=request.top_k)
    return {
        "question": request.question,
        "top_k": request.top_k,
        "chunks": [
            {
                "rank": i + 1,
                "text": c.get("text", ""),
                "source": c.get("source", "unknown"),
                "score": round(c.get("score", 0.0), 6)
            }
            for i, c in enumerate(chunks)
        ]
    }


@app.get("/health")
def health():
    return {"status": "ok", "model": "Llama 3.2 3B", "college": "MAC"}


# ─────────────────────────────────────────
# ADMIN ROUTES
# ─────────────────────────────────────────
@app.get("/admin/status")
def admin_status(username: str = Depends(verify_admin)):
    chunk_count = 0
    if os.path.exists("data/meta.json"):
        with open("data/meta.json", "r", encoding="utf-8") as f:
            chunk_count = len(json.load(f))
    return {
        "total_chunks": chunk_count,
        "total_crawls": len(get_all_crawls()),
        "scheduled_jobs": len(get_scheduled_jobs()),
        "model": "Llama 3.2 3B Instruct",
        "index_exists": os.path.exists("data/faiss.index"),
    }


@app.post("/admin/crawl/start")
def start_crawl(request: CrawlRequest, username: str = Depends(verify_admin)):
    try:
        from scraper.hybrid_crawler import run_crawl
        crawl_id = run_crawl(url=request.subdomain, max_pages=request.max_pages)
        mongo_sync()
        return {
            "status": "success",
            "crawl_id": crawl_id,
            "subdomain": request.subdomain,
            "message": "Crawl complete. Click Rebuild Index to activate new data."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/admin/crawl/schedule")
def schedule_crawl(request: ScheduleRequest, username: str = Depends(verify_admin)):
    if request.frequency not in ["daily", "weekly", "monthly"]:
        raise HTTPException(status_code=400, detail="frequency must be daily, weekly, or monthly")
    try:
        job_id = schedule_crawl_job(
            subdomain=request.subdomain,
            frequency=request.frequency,
            max_pages=request.max_pages
        )
        mongo_sync()
        return {"status": "success", "job_id": job_id, "frequency": request.frequency}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/admin/crawl/history")
def crawl_history(username: str = Depends(verify_admin)):
    return {"crawls": get_all_crawls()}


@app.delete("/admin/crawl/delete")
def delete_crawl(request: DeleteCrawlRequest, username: str = Depends(verify_admin)):
    if not get_crawl_by_id(request.crawl_id):
        raise HTTPException(status_code=404, detail=f"Crawl ID '{request.crawl_id}' not found")
    try:
        deleted = delete_crawl_by_id(request.crawl_id)
        mongo_sync()
        return {
            "status": "success",
            "crawl_id": request.crawl_id,
            "chunks_deleted": deleted,
            "message": "Deleted. Run Rebuild Index to apply changes."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/admin/index/rebuild")
def rebuild_faiss(username: str = Depends(verify_admin)):
    try:
        result = rebuild_index()
        mongo_sync()
        return {
            "status": "success",
            "chunks_indexed": result["chunks_indexed"],
            "time_taken": result["time_taken"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/admin/schedule/list")
def list_schedules(username: str = Depends(verify_admin)):
    return {"scheduled_jobs": get_scheduled_jobs()}


@app.delete("/admin/schedule/remove/{job_id}")
def remove_schedule(job_id: str, username: str = Depends(verify_admin)):
    try:
        remove_scheduled_job(job_id)
        mongo_sync()
        return {"status": "success", "job_id": job_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────
# RESET
# ─────────────────────────────────────────
@app.delete("/admin/data/reset")
def reset_all_data(username: str = Depends(verify_admin)):
    try:
        if os.path.exists("data/faiss.index"):
            os.remove("data/faiss.index")

        with open("data/meta.json", "w", encoding="utf-8") as f:
            json.dump([], f)

        with open("data/raw_pages.json", "w", encoding="utf-8") as f:
            json.dump([], f)

        try:
            from mongodb_backup import get_db
            db = get_db()
            db["chunks"].delete_many({})
            db["embeddings"].delete_many({})
        except Exception:
            pass

        from backend.retriever import reload_index
        reload_index()

        return {
            "status": "reset_complete",
            "message": "System cleared. Run crawl + rebuild."
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────
# STARTUP
# ─────────────────────────────────────────
@app.on_event("startup")
def on_startup():
    start_scheduler()
    print("[API] Started. Admin: admin / mac2024")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)