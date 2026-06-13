"""
================================================================================
  MAC RAG CHATBOT — DATA CLEANING PIPELINE (PDF-AWARE)
  Project   : Efficient Chatbot for Maharaja Agrasen College (MAC)
  Author    : Kartikey Tiwari, B.Sc. (Hons.) Electronics, VIII Sem
  Supervisor: Prof. Amit Pundir

  WHAT THIS HANDLES:
    HTML pages:
      - Binary image/JPEG sources                  (blocked)
      - Google Translate language list              (stripped)
      - Google Translate toolbar                    (stripped)
      - Accessibility widget boilerplate            (stripped)
      - Navigation header bleed                     (stripped)
      - Footer / social share / breadcrumbs         (stripped)
      - Exact duplicate pages                       (deduplicated)
      - Pages too short after cleaning              (dropped)

    PDF pages:
      - Passed through as-is (already cleaned by hybrid_crawler.py)
      - NOT blocked (.pdf is legitimate content — previous version was wrong)
      - Short PDFs still dropped if under MIN_CONTENT_WORDS
================================================================================
"""

import json
import re
import unicodedata

INPUT_FILE  = "data/raw_pages.json"
OUTPUT_FILE = "data/final_pages.json"

MIN_CONTENT_WORDS = 30

# True binary file extensions — block entirely. NOTE: .pdf NOT in this list.
BINARY_EXTENSIONS = (
    ".jpeg", ".jpg", ".png", ".gif", ".webp",
    ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".zip", ".rar", ".mp4", ".mp3", ".svg", ".ico",
)

def is_binary_source(url: str) -> bool:
    return any(url.lower().split("?")[0].endswith(ext) for ext in BINARY_EXTENSIONS)

def is_binary_text(text: str) -> bool:
    if not text:
        return True
    non_printable = sum(
        1 for c in text[:500]
        if unicodedata.category(c) in ("Cc", "Cs") and c not in "\n\r\t"
    )
    return non_printable > 10

def is_pdf_source(url: str) -> bool:
    return url.lower().split("?")[0].endswith(".pdf")

def count_real_words(text: str) -> int:
    return len(re.findall(r"[a-zA-Z\u0900-\u097F]{3,}", text))

# ── HTML NOISE PATTERNS ───────────────────────────────────────────────────────

GT_LANGUAGE_LIST = re.compile(
    r"Abkhaz\s+Acehnese.*?(?:Zulu|Zapotec|Zaza)",
    re.IGNORECASE | re.DOTALL
)
GT_TOOLBAR = re.compile(
    r"(?:Original text\s+)?Rate this translation.*?"
    r"(?:Screen Reader|Toggle high contrast|Decrease text spacing)",
    re.IGNORECASE | re.DOTALL
)
ACCESSIBILITY_WIDGET = re.compile(
    r"(?:Play\s+Pause\s+Stop|Increase\s+Decrease\s+Reset|"
    r"Dyslexic\s+font|Gray\s+hues|Enlarged\s+cursor|"
    r"(?:Increase|Decrease)\s+text\s+spacing|"
    r"Toggle\s+high\s+contrast|Screen\s+Reader).*?(?:\n|$)",
    re.IGNORECASE
)
NAV_HEADER = re.compile(
    r"STAFF\s+LOGIN.*?(?:हिंदी|Hindi|Select\s+Language)",
    re.IGNORECASE | re.DOTALL
)
NAV_LINKS = re.compile(
    r"(?:Skip\s+to\s+(?:main\s+)?content|"
    r"Home\s+About\s+(?:Us\s+)?(?:MAC|College))",
    re.IGNORECASE
)
FOOTER = re.compile(
    r"(?:Copyright\s+©|Designed\s+(?:and\s+)?(?:Developed|Maintained)\s+by|"
    r"Best\s+viewed\s+in|Website\s+maintained\s+by).*?(?:\n|$)",
    re.IGNORECASE
)
SOCIAL_SHARE = re.compile(
    r"(?:Share\s+on\s+(?:Facebook|Twitter|WhatsApp|LinkedIn)|"
    r"Follow\s+us\s+on|Like\s+us\s+on\s+Facebook).*?(?:\n|$)",
    re.IGNORECASE
)
BREADCRUMB = re.compile(
    r"(?:You\s+are\s+here|Home\s*[»>]).*?(?:\n|$)",
    re.IGNORECASE
)


def clean_html_text(text: str) -> str:
    if is_binary_text(text):
        return ""
    text = GT_LANGUAGE_LIST.sub(" ", text)
    text = GT_TOOLBAR.sub(" ", text)
    text = ACCESSIBILITY_WIDGET.sub(" ", text)
    text = NAV_HEADER.sub(" ", text)
    text = NAV_LINKS.sub(" ", text)
    text = FOOTER.sub(" ", text)
    text = SOCIAL_SHARE.sub(" ", text)
    text = BREADCRUMB.sub(" ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {3,}", " ", text)
    return text.strip()


# ── QUALITY FILTER ────────────────────────────────────────────────────────────

def is_bad_page(text: str, url: str) -> tuple:
    text_lower = text.lower()
    if count_real_words(text) < MIN_CONTENT_WORDS:
        return True, "too_short"
    if text_lower.count("s.no") > 3:
        return True, "sno_table"
    if text_lower.count("view profile") > 8:
        return True, "view_profile_dump"
    if is_binary_text(text):
        return True, "binary_content"
    if "abkhaz" in text_lower or "dzongkha" in text_lower:
        return True, "gt_residue"
    return False, ""


# ── DEDUPLICATION ─────────────────────────────────────────────────────────────

def deduplicate(pages: list) -> tuple:
    seen, clean, dups = set(), [], 0
    for page in pages:
        fp = page["text"].strip().lower()
        if fp in seen:
            dups += 1
            continue
        seen.add(fp)
        clean.append(page)
    return clean, dups


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Loading {INPUT_FILE}...")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    print(f"Input pages: {len(raw_data)}\n")

    final = []
    stats = {
        "binary_source": 0, "binary_text": 0,
        "too_short": 0, "sno_table": 0,
        "view_profile_dump": 0, "gt_residue": 0,
        "pdf_kept": 0, "html_kept": 0,
    }

    for item in raw_data:
        url  = item.get("url", "")
        text = item.get("text", "")
        kind = item.get("kind", "html")

        if is_binary_source(url):
            stats["binary_source"] += 1
            continue

        if is_binary_text(text):
            stats["binary_text"] += 1
            continue

        # PDFs: already cleaned by crawler — quality-filter only
        if is_pdf_source(url) or kind == "pdf":
            bad, reason = is_bad_page(text, url)
            if bad:
                stats[reason] = stats.get(reason, 0) + 1
                continue
            final.append({"url": url, "text": text, "kind": "pdf"})
            stats["pdf_kept"] += 1
            continue

        # HTML: full cleaning pass
        cleaned = clean_html_text(text)
        bad, reason = is_bad_page(cleaned, url)
        if bad:
            stats[reason] = stats.get(reason, 0) + 1
            continue

        final.append({"url": url, "text": cleaned, "kind": "html"})
        stats["html_kept"] += 1

    final, dup_count = deduplicate(final)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)

    print("=" * 55)
    print("  CLEANING REPORT")
    print("=" * 55)
    print(f"  Input pages              : {len(raw_data)}")
    print(f"  Dropped — binary URL     : {stats['binary_source']}")
    print(f"  Dropped — binary text    : {stats['binary_text']}")
    print(f"  Dropped — too short      : {stats['too_short']}")
    print(f"  Dropped — S.No. tables   : {stats['sno_table']}")
    print(f"  Dropped — profile dumps  : {stats['view_profile_dump']}")
    print(f"  Dropped — GT residue     : {stats.get('gt_residue', 0)}")
    print(f"  Dropped — duplicates     : {dup_count}")
    print("-" * 55)
    print(f"  HTML pages kept          : {stats['html_kept']}")
    print(f"  PDF pages kept           : {stats['pdf_kept']}")
    print(f"  TOTAL CLEAN PAGES        : {len(final)}")
    print("=" * 55)
    print(f"\nSaved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
