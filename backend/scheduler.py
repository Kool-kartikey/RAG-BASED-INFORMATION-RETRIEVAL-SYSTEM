import uuid
import json
import os
from datetime import datetime

SCHEDULE_LOG_FILE = "data/schedule_log.json"

# ─────────────────────────────────────────
# Try to import APScheduler
# If not installed → use dummy scheduler
# ─────────────────────────────────────────
try:
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.jobstores.memory import MemoryJobStore
    USE_APSCHEDULER = True
except ImportError:
    USE_APSCHEDULER = False
    print("[SCHEDULER] APScheduler not installed. Using dummy scheduler.")
    print("[SCHEDULER] Run: pip install apscheduler")


# ─────────────────────────────────────────
# GLOBAL SCHEDULER
# ─────────────────────────────────────────
if USE_APSCHEDULER:
    scheduler = BackgroundScheduler(
        jobstores = {"default": MemoryJobStore()},
        timezone  = "Asia/Kolkata"
    )
else:
    scheduler = None


# ─────────────────────────────────────────
# SCHEDULE LOG (persists to disk)
# ─────────────────────────────────────────

def _load_schedule_log() -> list:
    if not os.path.exists(SCHEDULE_LOG_FILE):
        return []
    with open(SCHEDULE_LOG_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_schedule_log(log: list):
    os.makedirs("data", exist_ok=True)
    with open(SCHEDULE_LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)


# ─────────────────────────────────────────
# CRAWL JOB FUNCTION
# ─────────────────────────────────────────

def _run_scheduled_crawl(subdomain: str, max_pages: int):
    """Runs when a scheduled job fires."""
    print(f"\n[SCHEDULER] Running scheduled crawl for: {subdomain}")
    try:
        from pipeline.hybrid_crawler import run_crawl
        crawl_id = run_crawl(url=subdomain, max_pages=max_pages)
        print(f"[SCHEDULER] Crawl done. ID: {crawl_id}")

        from indexer import rebuild_index
        result = rebuild_index()
        print(f"[SCHEDULER] Index rebuilt: {result['chunks_indexed']} chunks")
    except Exception as e:
        print(f"[SCHEDULER ERROR] {e}")


# ─────────────────────────────────────────
# PUBLIC FUNCTIONS
# ─────────────────────────────────────────

def start_scheduler():
    """Call once when API starts."""
    if not USE_APSCHEDULER:
        print("[SCHEDULER] Skipping — APScheduler not available.")
        return

    if not scheduler.running:
        scheduler.start()

        # Restore saved jobs from disk
        saved_jobs = _load_schedule_log()
        for job in saved_jobs:
            try:
                _register_job(
                    job_id    = job["job_id"],
                    subdomain = job["subdomain"],
                    frequency = job["frequency"],
                    max_pages = job.get("max_pages", 100)
                )
            except Exception as e:
                print(f"[SCHEDULER] Failed to restore job {job['job_id']}: {e}")

        print(f"[SCHEDULER] Started. {len(saved_jobs)} jobs restored.")


def _register_job(job_id: str, subdomain: str, frequency: str, max_pages: int):
    """Registers job with APScheduler."""
    if not USE_APSCHEDULER:
        return

    trigger_map = {
        "daily"   : {"trigger": "cron", "hour": 2, "minute": 0},
        "weekly"  : {"trigger": "cron", "day_of_week": "mon", "hour": 2},
        "monthly" : {"trigger": "cron", "day": 1, "hour": 2},
    }
    trigger_config = trigger_map.get(frequency, trigger_map["weekly"])

    scheduler.add_job(
        func             = _run_scheduled_crawl,
        id               = job_id,
        kwargs           = {"subdomain": subdomain, "max_pages": max_pages},
        replace_existing = True,
        **trigger_config
    )


def schedule_crawl_job(subdomain: str, frequency: str, max_pages: int = 100) -> str:
    """Adds a new scheduled crawl. Returns job_id."""
    job_id = str(uuid.uuid4())[:8].upper()

    if USE_APSCHEDULER:
        _register_job(job_id, subdomain, frequency, max_pages)

    log = _load_schedule_log()
    log.append({
        "job_id"     : job_id,
        "subdomain"  : subdomain,
        "frequency"  : frequency,
        "max_pages"  : max_pages,
        "created_at" : datetime.now().isoformat(),
    })
    _save_schedule_log(log)

    print(f"[SCHEDULER] Job {job_id} scheduled ({frequency}) for {subdomain}")
    return job_id


def get_scheduled_jobs() -> list:
    """Returns all scheduled jobs with next run time."""
    log = _load_schedule_log()

    if not USE_APSCHEDULER or not scheduler.running:
        return [{**entry, "next_run": "scheduler offline"} for entry in log]

    job_map = {j.id: j for j in scheduler.get_jobs()}

    result = []
    for entry in log:
        job      = job_map.get(entry["job_id"])
        next_run = str(job.next_run_time) if job and job.next_run_time else "unknown"
        result.append({**entry, "next_run": next_run})

    return result


def remove_scheduled_job(job_id: str):
    """Removes a scheduled job from scheduler + disk."""
    if USE_APSCHEDULER and scheduler.running:
        try:
            if scheduler.get_job(job_id):
                scheduler.remove_job(job_id)
        except Exception as e:
            print(f"[SCHEDULER] Error removing job: {e}")

    log = _load_schedule_log()
    log = [j for j in log if j["job_id"] != job_id]
    _save_schedule_log(log)

    print(f"[SCHEDULER] Job {job_id} removed.")