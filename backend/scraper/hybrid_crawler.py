"""
================================================================================
  MAC RAG CHATBOT — HYBRID CRAWLER
  Project   : Efficient Chatbot for Maharaja Agrasen College (MAC)
  Author    : Kartikey Tiwari, B.Sc. (Hons.) Electronics, VIII Sem
  Supervisor: Prof. Amit Pundir

  WHAT'S IMPROVED OVER PREVIOUS VERSION:
    1. PDF-specific text cleaning (removes repeated headers/footers,
       page numbers, watermarks, column-merge artifacts)
    2. PDF link discovery from JS-rendered pages via Playwright
    3. PDF page-aware chunking — respects natural page boundaries
    4. raw_pages.json now stores kind="pdf"/"html" for downstream use
    5. is_valid() updated — PDFs pass through, only true binary images blocked
================================================================================
"""

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from playwright.sync_api import sync_playwright
import json
import time
import hashlib
import os
import sys
import re
import unicodedata
from collections import Counter

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    fitz = None
    PYMUPDF_AVAILABLE = False
    print("[WARNING] PyMuPDF not installed. PDFs will be skipped.")
    print("          Install with: pip install pymupdf")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SITEMAP_URL = "https://mac.du.ac.in/sitemap.aspx"
DOMAIN      = "mac.du.ac.in"
OUTPUT_FILE = "data/raw_pages.json"
META_FILE   = "data/meta.json"
PDF_DIR     = "data/pdfs"
HEADERS     = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36"
}
MAX_PAGES        = 400
PDF_MIN_WORDS    = 30     # PDFs with fewer real words are skipped
HTML_MIN_WORDS   = 50     # HTML pages with fewer words are skipped

# True binary extensions — skip entirely (not PDFs)
SKIP_EXTENSIONS = (
    ".jpg", ".jpeg", ".png", ".gif", ".webp",
    ".mp4", ".mp3", ".avi", ".mov",
    ".zip", ".rar", ".exe", ".msi",
    ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".svg", ".ico", ".woff", ".woff2", ".ttf",
)

os.makedirs(PDF_DIR, exist_ok=True)


# =========================
# HELPERS
# =========================

def normalize(url):
    return url.split("#")[0].rstrip("/")


def is_pdf_url(url: str) -> bool:
    return url.lower().split("?")[0].endswith(".pdf")


def is_valid(url: str) -> bool:
    """
    Returns True if URL should be crawled.
    PDFs are valid. Only true binary non-text files are rejected.
    """
    parsed = urlparse(url)
    if DOMAIN not in parsed.netloc:
        return False
    url_clean = url.lower().split("?")[0]
    if any(url_clean.endswith(ext) for ext in SKIP_EXTENSIONS):
        return False
    return True


def hash_text(text: str) -> str:
    return hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest()


def count_real_words(text: str) -> int:
    """Counts alphabetic words only — filters token garbage."""
    return len(re.findall(r"[a-zA-Z\u0900-\u097F]{3,}", text))


# =========================
# HTML TEXT CLEANING
# =========================

def clean_html_text(text: str) -> str:
    """
    Cleans HTML-extracted text.
    Removes Google Translate toolbars, accessibility widgets,
    nav headers, and normalises whitespace.
    """
    if not text:
        return ""

    text = text.replace("\x00", " ")
    text = unicodedata.normalize("NFKC", text)

    # Google Translate language dropdown list
    text = re.sub(
        r"Abkhaz\s+Acehnese.*?(?:Zulu|Zapotec|Zaza)",
        " ", text, flags=re.IGNORECASE | re.DOTALL
    )
    # Google Translate toolbar buttons
    text = re.sub(
        r"(?:Original text\s+)?Rate this translation.*?"
        r"(?:Screen Reader|Toggle high contrast|Decrease text spacing)",
        " ", text, flags=re.IGNORECASE | re.DOTALL
    )
    # Accessibility widget text
    text = re.sub(
        r"(?:Play\s+Pause\s+Stop|Increase\s+Decrease\s+Reset|"
        r"Dyslexic\s+font|Gray\s+hues|Enlarged\s+cursor|"
        r"(?:Increase|Decrease)\s+text\s+spacing|"
        r"Toggle\s+high\s+contrast|Screen\s+Reader).*?(?:\n|$)",
        " ", text, flags=re.IGNORECASE
    )
    # Navigation header
    text = re.sub(
        r"STAFF\s+LOGIN.*?(?:हिंदी|Hindi|Select\s+Language)",
        " ", text, flags=re.IGNORECASE | re.DOTALL
    )
    # Breadcrumbs
    text = re.sub(
        r"(?:You\s+are\s+here|Home\s*[»>]).*?(?:\n|$)",
        " ", text, flags=re.IGNORECASE
    )
    # Footer
    text = re.sub(
        r"(?:Copyright\s+©|Designed\s+(?:and\s+)?(?:Developed|Maintained)\s+by|"
        r"Best\s+viewed\s+in|Website\s+maintained\s+by).*?(?:\n|$)",
        " ", text, flags=re.IGNORECASE
    )
    # Repeated punctuation
    text = re.sub(r"([•▪◦·])\1+", r"\1", text)

    # Normalise whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_text_bs(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    text = " ".join(soup.get_text(separator=" ", strip=True).split())
    return clean_html_text(text)


# =========================
# PDF TEXT CLEANING
# =========================

def detect_repeated_lines(pages_text: list, threshold: float = 0.6) -> set:
    """
    Detects lines that appear on more than `threshold` fraction of pages.
    These are almost certainly headers/footers/watermarks.
    Returns a set of such lines (lowercased, stripped) to remove.
    """
    if len(pages_text) < 3:
        return set()

    line_counts = Counter()
    for page_text in pages_text:
        # Consider only short lines (< 15 words) as potential header/footer
        for line in page_text.splitlines():
            line_stripped = line.strip()
            if line_stripped and len(line_stripped.split()) < 15:
                line_counts[line_stripped.lower()] += 1

    repeated = set()
    n_pages = len(pages_text)
    for line, count in line_counts.items():
        if count / n_pages >= threshold:
            repeated.add(line)

    return repeated


def clean_pdf_page(text: str, repeated_lines: set) -> str:
    """
    Cleans a single PDF page's text:
    - Removes detected header/footer lines
    - Fixes hyphenation line-breaks
    - Removes standalone page numbers
    - Normalises whitespace
    """
    if not text:
        return ""

    text = text.replace("\x00", " ")
    text = unicodedata.normalize("NFKC", text)

    # Fix PDF hyphenation: "exam-\nnation" → "examination"
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)

    # Remove lines that match detected repeated header/footer
    cleaned_lines = []
    for line in text.splitlines():
        if line.strip().lower() in repeated_lines:
            continue
        cleaned_lines.append(line)
    text = "\n".join(cleaned_lines)

    # Remove standalone page numbers (line containing only a number)
    text = re.sub(r"^\s*\d{1,4}\s*$", "", text, flags=re.MULTILINE)

    # Remove watermark-style repeated single words
    text = re.sub(r"(\b\w+\b)(\s+\1){4,}", r"\1", text, flags=re.IGNORECASE)

    # Fix column-merge artifacts (two columns smashed together)
    # e.g., "Name Department" on same line when they should be separate facts
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)

    # Normalise whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def extract_text_pdf_pages(filepath: str) -> list:
    """
    Extracts text per-page from a PDF using PyMuPDF.
    Returns list of (page_num, cleaned_text) tuples.
    Applies PDF-specific cleaning including header/footer removal.
    """
    if not PYMUPDF_AVAILABLE:
        return []

    try:
        doc = fitz.open(filepath)
        if doc.page_count == 0:
            return []

        # First pass: extract raw text per page
        raw_pages = []
        for page_num, page in enumerate(doc, 1):
            raw_text = page.get_text("text")
            if raw_text.strip():
                raw_pages.append((page_num, raw_text))

        if not raw_pages:
            return []

        # Detect repeated header/footer lines across all pages
        repeated = detect_repeated_lines([t for _, t in raw_pages])

        # Second pass: clean each page
        cleaned_pages = []
        for page_num, raw_text in raw_pages:
            cleaned = clean_pdf_page(raw_text, repeated)
            if count_real_words(cleaned) >= 10:  # at least 10 real words
                cleaned_pages.append((page_num, cleaned))

        doc.close()
        return cleaned_pages

    except Exception as e:
        print(f"  [PDF extraction failed] {e}")
        return []


def extract_text_pdf(filepath: str) -> str:
    """
    Full PDF text as a single string (for delta hashing and raw_pages.json).
    """
    pages = extract_text_pdf_pages(filepath)
    if not pages:
        return ""
    return "\n\n".join(text for _, text in pages)


# =========================
# PDF DOWNLOAD
# =========================

def pdf_filename_from_url(url: str) -> str:
    return hashlib.md5(url.encode("utf-8", errors="ignore")).hexdigest() + ".pdf"


def download_pdf(url: str) -> str:
    """
    Downloads a PDF and returns its local filepath.
    Returns empty string on failure.
    """
    try:
        filepath = os.path.join(PDF_DIR, pdf_filename_from_url(url))
        if os.path.exists(filepath) and os.path.getsize(filepath) > 1024:
            return filepath   # already downloaded and non-empty

        res = requests.get(url, headers=HEADERS, timeout=30, stream=True)
        if res.status_code != 200:
            print(f"  [PDF] HTTP {res.status_code}: {url}")
            return ""

        content_type = res.headers.get("Content-Type", "").lower()
        if "pdf" not in content_type and not is_pdf_url(url):
            return ""

        with open(filepath, "wb") as f:
            for chunk in res.iter_content(chunk_size=8192):
                f.write(chunk)

        if os.path.getsize(filepath) < 1024:   # less than 1KB = likely error page
            os.remove(filepath)
            return ""

        return filepath

    except Exception as e:
        print(f"  [PDF download failed] {e}")
        return ""


# =========================
# PLAYWRIGHT HELPERS
# =========================

def get_text_playwright(url: str) -> str:
    """Fallback for JS-heavy pages that BS4 can't read."""
    try:
        print(f"  [Playwright fallback] {url}")
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, timeout=15000)
            page.wait_for_load_state("networkidle")
            html = page.content()
            browser.close()
            return extract_text_bs(html)
    except Exception as e:
        print(f"  [Playwright failed] {e}")
        return ""


def get_links_playwright(url: str) -> tuple:
    """
    Extracts both page links AND PDF links from a JS-rendered page.
    Returns (html_links, pdf_links).
    """
    html_links = []
    pdf_links  = []
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, timeout=15000)
            page.wait_for_load_state("networkidle")
            links = page.query_selector_all("a")
            for link in links:
                href = link.get_attribute("href")
                if not href:
                    continue
                full = normalize(urljoin(url, href))
                if not full.startswith("http"):
                    continue
                if is_pdf_url(full) and DOMAIN in full:
                    pdf_links.append(full)
                elif is_valid(full):
                    html_links.append(full)
            browser.close()
    except Exception:
        pass
    return html_links, pdf_links


# =========================
# STEP 1 — SITEMAP
# =========================

def get_sitemap_urls() -> tuple:
    """
    Returns (html_urls, pdf_urls) found in the sitemap.
    """
    html_urls = set()
    pdf_urls  = set()
    try:
        res = requests.get(SITEMAP_URL, headers=HEADERS, timeout=20)
        soup = BeautifulSoup(res.text, "html.parser")
        for link in soup.find_all("a", href=True):
            href = normalize(urljoin(SITEMAP_URL, link["href"]))
            if is_pdf_url(href) and DOMAIN in href:
                pdf_urls.add(href)
            elif is_valid(href):
                html_urls.add(href)
        print(f"Sitemap HTML URLs: {len(html_urls)}")
        print(f"Sitemap PDF  URLs: {len(pdf_urls)}")
    except Exception as e:
        print(f"[Sitemap failed] {e}")
    return list(html_urls), list(pdf_urls)


# =========================
# STEP 2 — BFS EXPANSION
# =========================

def is_important(url: str) -> bool:
    keywords = ["department", "faculty", "course", "programme",
                "notice", "download", "circular", "pdf"]
    return any(k in url.lower() for k in keywords)


def expand_links(html_urls: list) -> tuple:
    """
    BFS expansion from important HTML pages.
    Returns (expanded_html_urls, discovered_pdf_urls).
    """
    expanded  = set(html_urls)
    pdf_found = set()

    for url in html_urls:
        if not is_important(url):
            continue
        try:
            res = requests.get(url, headers=HEADERS, timeout=10)
            soup = BeautifulSoup(res.text, "html.parser")
            for link in soup.find_all("a", href=True):
                href = normalize(urljoin(url, link["href"]))
                if is_pdf_url(href) and DOMAIN in href:
                    pdf_found.add(href)
                elif is_valid(href):
                    expanded.add(href)
        except Exception:
            continue

    print(f"After BFS expansion — HTML: {len(expanded)}, PDFs found: {len(pdf_found)}")
    return list(expanded), list(pdf_found)


# =========================
# STEP 3 — PLAYWRIGHT LINKS
# =========================

def get_browser_links(html_urls: list) -> tuple:
    """
    Uses Playwright to extract links (including PDFs) from JS-rendered pages.
    Returns (expanded_html_urls, discovered_pdf_urls).
    """
    new_html = set(html_urls)
    new_pdfs = set()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        for url in html_urls:
            if not is_important(url):
                continue
            try:
                page.goto(url, timeout=15000)
                page.wait_for_load_state("networkidle")
                links = page.query_selector_all("a")
                for link in links:
                    href = link.get_attribute("href")
                    if not href:
                        continue
                    full = normalize(urljoin(url, href))
                    if is_pdf_url(full) and DOMAIN in full:
                        new_pdfs.add(full)
                    elif is_valid(full):
                        new_html.add(full)
            except Exception:
                continue
        browser.close()

    print(f"After Playwright — HTML: {len(new_html)}, PDFs: {len(new_pdfs)}")
    return list(new_html), list(new_pdfs)


# =========================
# EXISTING DATA LOADER
# =========================

def load_existing_hashes() -> set:
    if not os.path.exists(META_FILE):
        return set()
    with open(META_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    hashes = set()
    for chunk in chunks:
        text = chunk.get("text", "")
        if text:
            hashes.add(hash_text(text))
    print(f"[DELTA] Loaded {len(hashes)} existing chunk hashes")
    return hashes


def load_existing_sources() -> set:
    if not os.path.exists(META_FILE):
        return set()
    with open(META_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return set(c.get("source", "") for c in chunks)


# =========================
# CRAWL SINGLE SUBDOMAIN
# =========================

def crawl_subdomain(
    start_url: str,
    max_pages: int = 100,
    existing_hashes: set = None
) -> list:
    """
    Crawls a subdomain with:
    - BS4 first (fast), Playwright fallback for JS pages
    - Full PDF discovery + extraction with per-page cleaning
    - Delta check: skips unchanged pages
    Returns ONLY new/changed pages.
    """
    if existing_hashes is None:
        existing_hashes = set()

    visited      = set()
    html_queue   = [normalize(start_url)]
    pdf_queue    = []
    results      = []
    page_hashes  = set()

    skipped_unchanged = 0
    skipped_thin      = 0
    skipped_pdf_fail  = 0

    def add_links_from_soup(soup, base_url):
        for link in soup.find_all("a", href=True):
            href = normalize(urljoin(base_url, link["href"]))
            if is_pdf_url(href) and DOMAIN in href and href not in visited:
                if href not in pdf_queue:
                    pdf_queue.append(href)
            elif is_valid(href) and href not in visited:
                if href not in html_queue:
                    html_queue.append(href)

    # ── Process HTML pages ───────────────────────────────────────────
    while html_queue and len(results) < max_pages:
        url = html_queue.pop(0)

        if url in visited:
            continue
        visited.add(url)

        if DOMAIN not in url:
            continue

        try:
            print(f"  [HTML] {url}")
            res = requests.get(url, headers=HEADERS, timeout=10)

            if res.status_code != 200:
                continue

            text = extract_text_bs(res.text)

            # Playwright fallback for thin pages
            if count_real_words(text) < HTML_MIN_WORDS:
                print(f"  BS4 thin → Playwright fallback")
                text = get_text_playwright(url)
                # Also extract PDF links via Playwright
                _, pw_pdfs = get_links_playwright(url)
                for p in pw_pdfs:
                    if p not in visited and p not in pdf_queue:
                        pdf_queue.append(p)

            if count_real_words(text) < HTML_MIN_WORDS:
                skipped_thin += 1
                continue

            page_hash = hash_text(text)

            if page_hash in existing_hashes:
                print(f"  ⟳ Unchanged — skipping")
                skipped_unchanged += 1
                soup = BeautifulSoup(res.text, "html.parser")
                add_links_from_soup(soup, url)
                continue

            if page_hash in page_hashes:
                continue

            print(f"  ✅ NEW: {url}")
            results.append({
                "url" : url,
                "text": text,
                "hash": page_hash,
                "kind": "html"
            })
            page_hashes.add(page_hash)

            soup = BeautifulSoup(res.text, "html.parser")
            add_links_from_soup(soup, url)

            if len(soup.find_all("a", href=True)) < 5:
                pw_html, pw_pdfs = get_links_playwright(url)
                for h in pw_html:
                    if h not in visited:
                        html_queue.append(h)
                for p in pw_pdfs:
                    if p not in visited and p not in pdf_queue:
                        pdf_queue.append(p)

            time.sleep(0.3)

        except Exception as e:
            print(f"  Error: {e}")
            continue

    # ── Process PDF pages ─────────────────────────────────────────────
    print(f"\n[PDF PHASE] Processing {len(pdf_queue)} discovered PDFs...")
    for url in pdf_queue:
        if url in visited:
            continue
        visited.add(url)

        try:
            print(f"  [PDF] {url}")
            pdf_path = download_pdf(url)

            if not pdf_path:
                skipped_pdf_fail += 1
                print(f"  ✗ Download failed")
                continue

            text = extract_text_pdf(pdf_path)

            if count_real_words(text) < PDF_MIN_WORDS:
                skipped_thin += 1
                print(f"  ✗ Too thin after extraction")
                continue

            page_hash = hash_text(text)

            if page_hash in existing_hashes:
                print(f"  ⟳ Unchanged PDF — skipping")
                skipped_unchanged += 1
                continue

            if page_hash in page_hashes:
                continue

            print(f"  ✅ NEW PDF: {url} ({count_real_words(text)} words)")
            results.append({
                "url" : url,
                "text": text,
                "hash": page_hash,
                "kind": "pdf"
            })
            page_hashes.add(page_hash)
            time.sleep(0.5)

        except Exception as e:
            print(f"  Error: {e}")
            continue

    print(f"\n[CRAWL SUMMARY]")
    print(f"  Pages visited     : {len(visited)}")
    print(f"  New/changed HTML  : {sum(1 for r in results if r['kind']=='html')}")
    print(f"  New/changed PDFs  : {sum(1 for r in results if r['kind']=='pdf')}")
    print(f"  Unchanged skipped : {skipped_unchanged}")
    print(f"  Thin skipped      : {skipped_thin}")
    print(f"  PDF fail          : {skipped_pdf_fail}")

    return results


# =========================
# CHUNK HELPER
# =========================

def chunk_text_html(text: str) -> list:
    """
    Dual chunking for HTML content.
    Large (250w) + Small (80w) with deduplication.
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


def chunk_text_pdf(text: str, source_url: str) -> list:
    """
    PDF-aware chunking: splits on double newlines (natural page/section
    boundaries), then applies sliding window if section is too large.
    Produces cleaner, denser chunks than blind word-count slicing.
    """
    chunks = []
    seen   = set()

    # Split on double newline (page/section boundary preserved by extractor)
    sections = [s.strip() for s in re.split(r"\n{2,}", text) if s.strip()]

    for section in sections:
        words = section.split()

        if len(words) < 20:
            continue    # too short to be useful

        if len(words) <= 250:
            # Section fits in one chunk — use as-is
            if section not in seen:
                chunks.append(section)
                seen.add(section)
        else:
            # Section too large — sliding window with overlap
            for i in range(0, len(words), 200):
                chunk = " ".join(words[i:i + 250])
                if len(chunk.split()) >= 30 and chunk not in seen:
                    chunks.append(chunk)
                    seen.add(chunk)

    return chunks


# =========================
# RUN_CRAWL — INCREMENTAL
# =========================

def run_crawl(url: str, max_pages: int = 100) -> str:
    """
    Incremental crawl entry point called by api.py.
    Now fully PDF-aware.
    """
    from crawl_log import create_crawl_entry, update_crawl_entry

    print(f"\n[CRAWLER] Starting INCREMENTAL crawl for: {url}")
    print(f"[CRAWLER] Max pages: {max_pages}")
    print(f"[CRAWLER] PDF support: {'YES (PyMuPDF)' if PYMUPDF_AVAILABLE else 'NO (install pymupdf)'}")

    existing_hashes = load_existing_hashes()
    print(f"[CRAWLER] Existing chunk hashes: {len(existing_hashes)}")

    crawl_id = create_crawl_entry(subdomain=url, max_pages=max_pages)

    try:
        new_pages = crawl_subdomain(
            start_url      = url,
            max_pages      = max_pages,
            existing_hashes= existing_hashes
        )

        if not new_pages:
            print("[CRAWLER] No new or changed pages found.")
            update_crawl_entry(crawl_id, 0, 0, status="done")
            return crawl_id

        new_chunks   = []
        seen_chunks  = set()

        for page in new_pages:
            kind = page.get("kind", "html")
            if kind == "pdf":
                raw_chunks = chunk_text_pdf(page["text"], page["url"])
            else:
                raw_chunks = chunk_text_html(page["text"])

            for chunk in raw_chunks:
                fp = chunk.strip().lower()
                if fp in seen_chunks:
                    continue
                seen_chunks.add(fp)
                new_chunks.append({
                    "text"    : chunk.strip(),
                    "source"  : page["url"],
                    "type"    : kind,
                    "crawl_id": crawl_id
                })

        existing_chunks = []
        if os.path.exists(META_FILE):
            with open(META_FILE, "r", encoding="utf-8") as f:
                existing_chunks = json.load(f)

        all_chunks = existing_chunks + new_chunks

        with open(META_FILE, "w", encoding="utf-8") as f:
            json.dump(all_chunks, f, ensure_ascii=False)

        # Also save raw_pages for clean_data.py + build_rag.py pipeline
        existing_raw = []
        if os.path.exists(OUTPUT_FILE):
            with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
                existing_raw = json.load(f)

        existing_raw_urls = {p["url"] for p in existing_raw}
        new_raw = [
            {"url": p["url"], "text": p["text"], "kind": p.get("kind","html")}
            for p in new_pages
            if p["url"] not in existing_raw_urls
        ]
        all_raw = existing_raw + new_raw
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(all_raw, f, indent=2, ensure_ascii=False)

        update_crawl_entry(
            crawl_id   = crawl_id,
            pages_found= len(new_pages),
            chunks_added=len(new_chunks),
            status     = "done"
        )

        html_count = sum(1 for p in new_pages if p.get("kind") == "html")
        pdf_count  = sum(1 for p in new_pages if p.get("kind") == "pdf")

        print(f"\n[CRAWLER] INCREMENTAL CRAWL COMPLETE")
        print(f"  Crawl ID      : {crawl_id}")
        print(f"  New HTML pages: {html_count}")
        print(f"  New PDF pages : {pdf_count}")
        print(f"  New chunks    : {len(new_chunks)}")
        print(f"  Total chunks  : {len(all_chunks)}")

        return crawl_id

    except Exception as e:
        print(f"[CRAWLER ERROR] {e}")
        update_crawl_entry(crawl_id, 0, 0, status="failed")
        return crawl_id


# =========================
# FULL SITE CRAWL (main)
# =========================

def crawl_full(html_urls: list, pdf_urls: list) -> list:
    """
    Full crawl — processes HTML and PDF URL lists.
    Used by main() for initial full-site crawl.
    """
    results = []
    hashes  = set()

    # HTML pages
    for i, url in enumerate(html_urls):
        if len(results) >= MAX_PAGES:
            break
        try:
            print(f"[{i+1}] {url}")
            res = requests.get(url, headers=HEADERS, timeout=10)
            if res.status_code != 200:
                continue

            text = extract_text_bs(res.text)
            if count_real_words(text) < HTML_MIN_WORDS:
                text = get_text_playwright(url)
            if count_real_words(text) < HTML_MIN_WORDS:
                continue

            h = hash_text(text)
            if h in hashes:
                continue

            results.append({"url": url, "text": text, "kind": "html"})
            hashes.add(h)
            time.sleep(0.3)

        except Exception:
            continue

    # PDF pages
    print(f"\n[PDF PHASE] {len(pdf_urls)} PDFs to process...")
    for i, url in enumerate(pdf_urls):
        try:
            print(f"[PDF {i+1}] {url}")
            pdf_path = download_pdf(url)
            if not pdf_path:
                continue

            text = extract_text_pdf(pdf_path)
            if count_real_words(text) < PDF_MIN_WORDS:
                continue

            h = hash_text(text)
            if h in hashes:
                continue

            results.append({"url": url, "text": text, "kind": "pdf"})
            hashes.add(h)
            time.sleep(0.5)

        except Exception:
            continue

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    html_n = sum(1 for r in results if r["kind"] == "html")
    pdf_n  = sum(1 for r in results if r["kind"] == "pdf")
    print(f"\nFINAL DATASET: {len(results)} pages ({html_n} HTML, {pdf_n} PDF)")
    return results


# =========================
# MAIN PIPELINE
# =========================

def main():
    print("=" * 60)
    print("  MAC RAG CRAWLER — Full Site Crawl")
    print(f"  PyMuPDF: {'available' if PYMUPDF_AVAILABLE else 'NOT INSTALLED — PDFs will be skipped'}")
    print("=" * 60 + "\n")

    html_urls, pdf_urls = get_sitemap_urls()
    html_urls, more_pdfs = expand_links(html_urls)
    pdf_urls = list(set(pdf_urls + more_pdfs))

    html_urls, browser_pdfs = get_browser_links(html_urls)
    pdf_urls = list(set(pdf_urls + browser_pdfs))

    print(f"\nFinal URL counts — HTML: {len(html_urls)}, PDF: {len(pdf_urls)}")
    crawl_full(html_urls, pdf_urls)


if __name__ == "__main__":
    main()
