"""
================================================================================
  MAC RAG CHATBOT — HUMAN ANNOTATION TOOL
  Project   : Efficient Chatbot for Maharaja Agrasen College (MAC)
  Author    : Kartikey Tiwari, B.Sc. (Hons.) Electronics, VII Sem
  Supervisor: Prof. Amit Pundir

  PURPOSE:
    Produces binary relevance labels (0/1) for retrieved chunks,
    enabling computation of Hit Rate @K and MRR — standard IR metrics
    per Voorhees (TREC 1999) and Baeza-Yates & Ribeiro-Neto (1999).

  ANNOTATION RULE (memorise this):
    Mark 1 (RELEVANT) if the chunk contains information necessary or
    sufficient to correctly answer the query.
    Mark 0 (NOT RELEVANT) if removing it would not affect answer quality.

  USAGE:
    1. Add /debug endpoint to api.py  (already done)
    2. Start your FastAPI server: python api.py
    3. Run: python annotate.py
    4. Press 1 (relevant), 0 (not relevant), s (skip/unsure)
    5. Progress auto-saves after every chunk — safe to Ctrl+C and resume

  OUTPUT:
    annotations.json  — full annotation record (used by benchmark_eval.py)
    annotation_log.txt — human-readable summary
================================================================================
"""

import json
import os
import sys
import time
import requests
from datetime import datetime
from colorama import Fore, Back, Style, init

init(autoreset=True)

# ── CONFIG ────────────────────────────────────────────────────────────────────
API_BASE_URL    = "http://localhost:8000"
DEBUG_ENDPOINT  = f"{API_BASE_URL}/debug"
DATASET_FILE    = "eval_dataset.json"
ANNOTATIONS_FILE= "annotations.json"
LOG_FILE        = "annotation_log.txt"
TOP_K           = 5          # chunks to retrieve per query
REQUEST_TIMEOUT = 30         # seconds

# ── CATEGORY COLOURS ──────────────────────────────────────────────────────────
CAT_COLOUR = {
    "factual"      : Fore.CYAN,
    "listing"      : Fore.GREEN,
    "entity"       : Fore.MAGENTA,
    "admission_fee": Fore.YELLOW,
    "off_topic"    : Fore.RED,
    "absent"       : Fore.RED,
}

# ── HELPERS ───────────────────────────────────────────────────────────────────

def clear():
    os.system("cls" if os.name == "nt" else "clear")


def print_header(current: int, total: int, annotated_chunks: int):
    print(Fore.CYAN + "=" * 70)
    print(Fore.CYAN + "  MAC RAG — HUMAN ANNOTATION TOOL")
    print(Fore.CYAN + "  Maharaja Agrasen College, University of Delhi")
    print(Fore.CYAN + "=" * 70)
    pct = int((current / total) * 100) if total else 0
    bar = ("█" * (pct // 5)).ljust(20)
    print(f"  Query progress : [{Fore.GREEN}{bar}{Style.RESET_ALL}] "
          f"{current}/{total} queries ({pct}%)")
    print(f"  Chunks labelled: {annotated_chunks}")
    print(Fore.CYAN + "=" * 70 + "\n")


def print_annotation_rule():
    print(Fore.WHITE + "  ANNOTATION RULE:")
    print("  Mark " + Fore.GREEN + "1 (RELEVANT)" + Style.RESET_ALL +
          " if chunk helps answer the query.")
    print("  Mark " + Fore.RED + "0 (NOT RELEVANT)" + Style.RESET_ALL +
          " if chunk is noise for this query.")
    print("  Press " + Fore.YELLOW + "s" + Style.RESET_ALL +
          " to skip/flag as unsure (you can review later).")
    print()


def check_server() -> bool:
    try:
        r = requests.get(f"{API_BASE_URL}/health", timeout=10)
        return r.status_code == 200
    except Exception:
        return False


def load_dataset(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["queries"]


def load_annotations(path: str) -> dict:
    """Load existing annotations (supports resume)."""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "metadata": {
            "project"      : "MAC RAG Chatbot Benchmark",
            "annotator"    : "Kartikey Tiwari",
            "started_at"   : datetime.now().isoformat(),
            "annotation_rule": (
                "Chunk marked RELEVANT (1) if it contains information "
                "necessary or sufficient to correctly answer the query. "
                "NOT RELEVANT (0) otherwise. "
                "Protocol follows Voorhees (TREC 1999)."
            ),
        },
        "queries": {}
    }


def save_annotations(data: dict, path: str):
    data["metadata"]["last_saved"] = datetime.now().isoformat()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def fetch_chunks(question: str, top_k: int) -> list:
    """Call /debug endpoint to get retrieved chunks."""
    try:
        r = requests.post(
            DEBUG_ENDPOINT,
            json={"question": question, "top_k": top_k},
            timeout=REQUEST_TIMEOUT
        )
        if r.status_code == 200:
            return r.json().get("chunks", [])
        else:
            print(Fore.RED + f"  /debug returned HTTP {r.status_code}")
            return []
    except requests.exceptions.Timeout:
        print(Fore.RED + "  Request timed out.")
        return []
    except Exception as e:
        print(Fore.RED + f"  Error: {e}")
        return []


def get_keypress(valid_keys: list) -> str:
    """Read a single keypress (cross-platform)."""
    while True:
        try:
            if os.name == "nt":
                import msvcrt
                key = msvcrt.getwch().lower()
            else:
                import tty, termios
                fd = sys.stdin.fileno()
                old = termios.tcgetattr(fd)
                try:
                    tty.setraw(fd)
                    key = sys.stdin.read(1).lower()
                finally:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old)
            if key in valid_keys:
                return key
        except Exception:
            # Fallback for environments without raw mode
            line = input().strip().lower()
            if line in valid_keys:
                return line


def annotate_query(query: dict, annotations: dict,
                   query_num: int, total: int) -> int:
    """
    Annotate all chunks for a single query.
    Returns number of chunks newly annotated.
    """
    qid      = query["id"]
    category = query["category"]
    q_text   = query["query"]
    gt       = query["ground_truth"]
    expected = query["expected_behavior"]

    # Already fully annotated? Skip.
    if qid in annotations["queries"]:
        existing = annotations["queries"][qid]
        if existing.get("completed", False):
            print(Fore.YELLOW + f"  [{qid}] Already annotated — skipping.")
            time.sleep(0.3)
            return 0

    # Fetch chunks
    print(Fore.CYAN + f"\n  Fetching chunks from /debug...")
    chunks = fetch_chunks(q_text, TOP_K)

    if not chunks:
        print(Fore.RED + "  No chunks returned. Skipping this query.")
        annotations["queries"][qid] = {
            "query"     : q_text,
            "category"  : category,
            "ground_truth": gt,
            "expected"  : expected,
            "chunks"    : [],
            "completed" : True,
            "note"      : "No chunks returned by retriever",
        }
        save_annotations(annotations, ANNOTATIONS_FILE)
        return 0

    # Special rule for off_topic and absent queries
    auto_label = None
    if expected in ("reject", "no_data"):
        auto_label = 0
        print(Fore.YELLOW +
              f"\n  AUTO-LABEL: This is an {expected.upper()} query.")
        print(Fore.YELLOW +
              "  All chunks will be marked 0 (not relevant) automatically.")
        print(Fore.YELLOW +
              "  This is correct — no MAC website chunk should answer this.")
        print()

    chunk_records = []
    newly_annotated = 0

    for i, chunk in enumerate(chunks):
        rank   = chunk.get("rank", i + 1)
        text   = chunk.get("text", "").strip()
        source = chunk.get("source", "unknown")
        score  = chunk.get("score", 0.0)

        clear()
        total_chunks_done = sum(
            len(q.get("chunks", []))
            for q in annotations["queries"].values()
        ) + len(chunk_records)
        print_header(query_num, total, total_chunks_done)

        # Query info panel
        cat_col = CAT_COLOUR.get(category, Fore.WHITE)
        print(f"  {cat_col}[{qid}]{Style.RESET_ALL}  "
              f"Category: {cat_col}{category}{Style.RESET_ALL}  "
              f"Expected: {expected}")
        print()
        print(Fore.WHITE + "  QUERY:")
        print(f"  {q_text}")
        print()
        print(Fore.WHITE + "  GROUND TRUTH:")
        print(Fore.GREEN + f"  {gt}")
        print()
        print(Fore.CYAN + f"  ── CHUNK {rank} of {len(chunks)} "
              + "─" * (48 - len(str(rank))) )
        print(f"  Source : {Fore.YELLOW}{source}{Style.RESET_ALL}")
        print(f"  Score  : {score:.6f}")
        print()

        # Display chunk text (wrapped at 66 chars)
        words = text.split()
        line  = "  │  "
        for word in words:
            if len(line) + len(word) + 1 > 70:
                print(line)
                line = "  │  " + word + " "
            else:
                line += word + " "
        if line.strip():
            print(line)
        print()

        # Auto-label for off_topic / absent
        if auto_label is not None:
            label  = auto_label
            note   = "auto-labelled 0: off_topic/absent query"
            print(Fore.YELLOW + f"  → Auto-labelled: 0 (not relevant)\n")
        else:
            print_annotation_rule()
            print("  Your label for this chunk: ", end="", flush=True)
            key = get_keypress(["1", "0", "s"])

            if key == "1":
                label = 1
                note  = ""
                print(Fore.GREEN + "1 — RELEVANT ✓")
            elif key == "0":
                label = 0
                note  = ""
                print(Fore.RED + "0 — NOT RELEVANT ✗")
            else:
                label = -1
                note  = "flagged for review"
                print(Fore.YELLOW + "s — SKIPPED (flagged for review)")

        chunk_records.append({
            "rank"  : rank,
            "source": source,
            "score" : score,
            "text"  : text[:300],   # truncate for storage
            "label" : label,
            "note"  : note,
        })
        newly_annotated += 1
        time.sleep(0.15)

    # Save this query's annotations
    annotations["queries"][qid] = {
        "query"       : q_text,
        "category"    : category,
        "ground_truth": gt,
        "expected"    : expected,
        "chunks"      : chunk_records,
        "completed"   : True,
        "annotated_at": datetime.now().isoformat(),
    }
    save_annotations(annotations, ANNOTATIONS_FILE)

    # Confirmation
    relevant_count = sum(1 for c in chunk_records if c["label"] == 1)
    print()
    print(Fore.GREEN + f"  ✓ Query [{qid}] saved — "
          f"{relevant_count}/{len(chunk_records)} chunks marked relevant.")
    print(Fore.CYAN + "  Press any key for next query...", end="", flush=True)
    get_keypress(["0","1","2","3","4","5","6","7","8","9",
                  "a","b","c","d","e","f","g","h","i","j","k","l","m",
                  "n","o","p","q","r","s","t","u","v","w","x","y","z",
                  " ","\r","\n"])
    return newly_annotated


def generate_log(annotations: dict, queries: list):
    """Write a human-readable summary to annotation_log.txt."""
    lines = [
        "=" * 70,
        "  MAC RAG CHATBOT — ANNOTATION LOG",
        f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 70,
        "",
        f"  Annotation rule: {annotations['metadata']['annotation_rule']}",
        "",
        f"  {'ID':<6} {'Category':<14} {'Chunks':<8} "
        f"{'Relevant':<10} {'Precision':<12} {'Status'}",
        "  " + "-" * 58,
    ]

    total_chunks   = 0
    total_relevant = 0

    for q in queries:
        qid = q["id"]
        if qid not in annotations["queries"]:
            lines.append(f"  {qid:<6} {q['category']:<14} {'—':<8} "
                         f"{'—':<10} {'—':<12} PENDING")
            continue

        rec      = annotations["queries"][qid]
        chunks   = rec.get("chunks", [])
        n        = len(chunks)
        relevant = sum(1 for c in chunks if c["label"] == 1)
        prec     = f"{relevant/n*100:.0f}%" if n else "—"
        status   = "DONE" if rec.get("completed") else "PARTIAL"

        total_chunks   += n
        total_relevant += relevant

        lines.append(f"  {qid:<6} {q['category']:<14} {n:<8} "
                     f"{relevant:<10} {prec:<12} {status}")

    lines += [
        "",
        "  " + "─" * 58,
        f"  Total chunks annotated : {total_chunks}",
        f"  Total relevant chunks  : {total_relevant}",
        f"  Overall precision      : "
        f"{total_relevant/total_chunks*100:.1f}%" if total_chunks else "—",
        "",
        "=" * 70,
    ]

    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(Fore.GREEN + f"\n  📄 Log saved to: {LOG_FILE}")


def show_resume_status(annotations: dict, queries: list):
    """Show which queries are done and which are pending."""
    done    = set(annotations["queries"].keys())
    pending = [q for q in queries if q["id"] not in done]
    print(Fore.CYAN + "\n  RESUME STATUS")
    print(f"  Completed : {len(done)} / {len(queries)} queries")
    print(f"  Remaining : {len(pending)} queries")
    if pending:
        print(f"  Next up   : [{pending[0]['id']}] {pending[0]['query'][:50]}")
    print()


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    clear()
    print(Fore.CYAN + "=" * 70)
    print(Fore.CYAN + "  MAC RAG — HUMAN ANNOTATION TOOL")
    print(Fore.CYAN + "  Maharaja Agrasen College, University of Delhi")
    print(Fore.CYAN + "=" * 70 + "\n")

    # Server check
    print("  Checking server...", end=" ", flush=True)
    if not check_server():
        print(Fore.RED + "FAILED")
        print(Fore.RED + "\n  FastAPI server not reachable at " + API_BASE_URL)
        print("  Start it with: python api.py\n")
        sys.exit(1)
    print(Fore.GREEN + "OK ✓")

    # Dataset
    if not os.path.exists(DATASET_FILE):
        print(Fore.RED + f"\n  eval_dataset.json not found.")
        print("  Place it in the same directory as annotate.py.\n")
        sys.exit(1)

    queries     = load_dataset(DATASET_FILE)
    annotations = load_annotations(ANNOTATIONS_FILE)

    show_resume_status(annotations, queries)

    # Instructions
    print(Fore.WHITE + "  HOW THIS WORKS:")
    print("  • For each query, your retriever fetches the top-5 chunks")
    print("  • You read each chunk and press 1 (relevant) or 0 (not relevant)")
    print("  • Off-topic and absent queries are auto-labelled 0")
    print("  • Progress is saved after EVERY chunk — safe to Ctrl+C anytime")
    print("  • Resume by running this script again\n")
    print(Fore.CYAN + "  Press ENTER to begin annotation...", end="", flush=True)

    try:
        input()
    except KeyboardInterrupt:
        print("\n  Cancelled.")
        sys.exit(0)

    # Main annotation loop
    total_annotated = 0
    already_done    = sum(
        1 for q in queries
        if q["id"] in annotations["queries"]
        and annotations["queries"][q["id"]].get("completed", False)
    )

    for i, query in enumerate(queries):
        qid = query["id"]

        # Skip already completed
        if (qid in annotations["queries"] and
                annotations["queries"][qid].get("completed", False)):
            continue

        try:
            n = annotate_query(
                query, annotations,
                query_num=already_done + total_annotated + 1,
                total=len(queries)
            )
            total_annotated += 1
        except KeyboardInterrupt:
            print(Fore.YELLOW + "\n\n  Interrupted — progress saved. "
                  "Run again to resume.")
            generate_log(annotations, queries)
            sys.exit(0)

    # Done
    clear()
    print_header(len(queries), len(queries),
                 sum(len(q.get("chunks", []))
                     for q in annotations["queries"].values()))

    print(Fore.GREEN + "  ALL 30 QUERIES ANNOTATED ✓\n")
    print("  Your annotations are saved in: annotations.json")
    print("  This file is the input to:     benchmark_eval.py\n")

    generate_log(annotations, queries)

    print(Fore.CYAN + "\n  Next step: run python benchmark_eval.py")
    print(Fore.CYAN + "  (make sure LM Studio is running first)\n")


if __name__ == "__main__":
    main()
