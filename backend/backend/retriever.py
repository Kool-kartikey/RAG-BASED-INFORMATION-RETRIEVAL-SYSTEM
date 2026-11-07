import faiss
import json
import numpy as np
import re
import os
import sys
import math
from sentence_transformers import SentenceTransformer

# ==============================================================================
# PATH & MODULE RESOLUTION (ROBUST ACROSS WORKING DIRECTORIES)
# ==============================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(CURRENT_DIR)
for p in [CURRENT_DIR, BACKEND_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from backend.llm import generate_answer, is_existence_query
except ImportError:
    from llm import generate_answer, is_existence_query


def resolve_path(relative_path: str) -> str:
    candidates = [
        os.path.join(BACKEND_DIR, relative_path),
        os.path.join(os.getcwd(), relative_path),
        os.path.join(os.getcwd(), "backend", relative_path),
        os.path.join(os.getcwd(), "repo", "backend", relative_path),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return os.path.join(BACKEND_DIR, relative_path)

INDEX_FILE = resolve_path("data/faiss.index")
META_FILE = resolve_path("data/meta.json")

TOP_K_CANDIDATES = 60
TOP_K_DEFAULT = 5
TOP_K_LISTING = 40
MIN_SCORE = 0.05

HIGH_VALUE_URLS = {
    "contactus.aspx": 0.70,
    "departments.aspx": 0.60,
    "courses.aspx": 0.60,
    "principal.aspx": 0.50,
    "about.aspx": 0.35,
    "mission.aspx": 0.30,
    "contact": 0.35,
    "admission": 0.25,
    "commerce-intro": 0.20,
    "history-intro": 0.20,
    "mathematics-intro": 0.20,
    "chemistry-intro": 0.20,
    "electronics-intro": 0.30,
    "physical-education": 0.20,
    "english-intro": 0.30,
    "computersci-intro": 0.20,
    "journalism-intro": 0.20,
    "hindi-intro": 0.20,
    "physics-intro": 0.20,
    "political-science": 0.20,
    "business-economics-intro": 0.35,
    "economics-intro": 0.20,
    "biology-intro": 0.20,
    "electronics.aspx": 0.45,
    "commerce.aspx": 0.45,
    "computersci.aspx": 0.45,
    "mathematics.aspx": 0.45,
    "physics.aspx": 0.45,
    "english.aspx": 0.45,
    "hindi.aspx": 0.40,
    "history.aspx": 0.40,
    "journalism.aspx": 0.40,
}

LISTING_KEYWORDS = [
    "how many", "list", "all", "what are", "what courses",
    "what departments", "what programmes",
    "tell me all", "show all", "departments are",
    "courses are", "programmes are", "programs are",
    "which departments", "which courses", "which programmes",
    "all departments", "all courses", "all programmes",
    "all faculty", "all teachers", "all professors", "all staff",
    "list faculty", "faculty of department", "faculty in department",
    "faculty members", "teaching staff", "professors in", "teachers in",
]

YESNO_KEYWORDS = [
    "does mac offer", "does mac have", "is there a",
    "do they offer", "is there", "does the college offer",
    "does the college have", "can i study", "is bsc",
    "is ba ", "is bcom", "is msc"
]

print("[INIT] Loading dense embedding model (all-MiniLM-L6-v2)...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print(f"[INIT] Loading FAISS index from: {INDEX_FILE}")
index = faiss.read_index(INDEX_FILE)

print(f"[INIT] Loading metadata from: {META_FILE}")
with open(META_FILE, "r", encoding="utf-8") as f:
    metadata = json.load(f)

# ==============================================================================
# BM25 LEXICAL ENGINE (SPARSE RETRIEVAL)
# ==============================================================================
def bm25_tokenize(text: str) -> list:
    """Extract lowercased alphanumeric words for sparse term matching."""
    return [w for w in re.findall(r"\b\w+\b", text.lower()) if len(w) > 1]

print("[INIT] Indexing corpus for BM25 lexical retrieval...")
tokenized_corpus = [bm25_tokenize(doc.get("text", "")) for doc in metadata]

try:
    from rank_bm25 import BM25Okapi
    bm25 = BM25Okapi(tokenized_corpus)
    print(f"[INIT] BM25Okapi initialized across {len(tokenized_corpus)} chunks.")
except Exception as e:
    print(f"[INIT] Note: rank_bm25 fallback engine activated ({e})")
    class FallbackBM25Okapi:
        def __init__(self, corpus, k1=1.5, b=0.75):
            self.k1 = k1
            self.b = b
            self.corpus_size = len(corpus)
            self.avgdl = sum(len(x) for x in corpus) / self.corpus_size if self.corpus_size > 0 else 1.0
            self.doc_freqs = []
            self.idf = {}
            nd = {}
            for document in corpus:
                frequencies = {}
                for word in document:
                    frequencies[word] = frequencies.get(word, 0) + 1
                self.doc_freqs.append(frequencies)
                for word in frequencies:
                    nd[word] = nd.get(word, 0) + 1
            for word, freq in nd.items():
                self.idf[word] = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1)

        def get_scores(self, query_tokens):
            scores = np.zeros(self.corpus_size)
            for q in query_tokens:
                q_idf = self.idf.get(q, 0.0)
                if q_idf <= 0:
                    continue
                for idx, freqs in enumerate(self.doc_freqs):
                    if q in freqs:
                        freq = freqs[q]
                        doc_len = sum(freqs.values())
                        num = freq * (self.k1 + 1)
                        denom = freq + self.k1 * (1 - self.b + self.b * (doc_len / self.avgdl))
                        scores[idx] += q_idf * (num / denom)
            return scores
    bm25 = FallbackBM25Okapi(tokenized_corpus)

# ==============================================================================
# CROSS-ENCODER NEURAL RERANKER (STAGE 2)
# ==============================================================================
reranker = None
try:
    from sentence_transformers import CrossEncoder
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    print("[INIT] Cross-Encoder reranker (ms-marco-MiniLM-L-6-v2) ready.")
except Exception as e:
    print(f"[INIT] Cross-Encoder deferred to RRF ranking ({e})")

print(f"[INIT] Ready. Hybrid retrieval pipeline loaded with {len(metadata)} chunks.\n")

INTENT_MAP = {
    "principal": {
        "keywords": [
            "principal", "head of college", "head of institution",
            "who is the principal", "institution head",
            "college head", "incharge", "officiating"
        ],
        "boost": ["principal", "head of institution"],
    },
    "faculty": {
        "keywords": [
            "faculty", "professor", "teacher", "lecturer",
            "staff", "dr.", "prof.", "phd", "assistant professor"
        ],
        "boost": ["faculty", "professor", "lecturer", "staff"],
    },
    "department": {
        "keywords": [
            "department", "dept", "division", "programme",
            "computer science", "commerce", "arts", "science",
            "courses offered", "course", "courses", "how many",
            "offered", "available", "bcom", "bsc", "ba",
            "bachelor", "undergraduate", "fyup", "program",
            "programmes", "programs", "study", "what courses",
            "what departments", "what programmes",
            "which courses", "which departments",
            "electronics", "does mac offer", "journalism",
            "all departments", "all courses", "english", "hindi"
        ],
        "boost": ["department", "programme", "course", "undergraduate"],
    },
    "admission": {
        "keywords": ["admission", "apply", "eligibility", "cutoff", "merit", "entrance", "enrollment", "form"],
        "boost": ["admission", "eligibility", "apply"],
    },
    "fee": {
        "keywords": ["fee", "fees", "cost", "tuition", "payment", "scholarship", "charges", "financial"],
        "boost": ["fee structure", "tuition", "charges"],
    },
    "contact": {
        "keywords": [
            "contact", "phone", "email", "address", "location", "office",
            "helpline", "reach", "mobile", "telephone", "number", "where is",
            "how can", "how to contact", "vasundhara", "website", "official website"
        ],
        "boost": ["contact", "address", "phone number", "email"],
    },
    "exam": {
        "keywords": ["exam", "examination", "result", "timetable", "schedule", "datesheet", "marks", "grade"],
        "boost": ["examination", "result", "timetable"],
    },
    "identity": {
        "keywords": [
            "full form", "stands for", "full name",
            "what is mac", "about mac", "about college",
            "what is maharaja", "history", "established",
            "founded", "since when", "affiliation",
            "accreditation", "naac", "vision", "mission",
            "what is b.a.", "what is b.sc", "what is b.com",
            "what is bsc", "what is bcom", "what is ba ",
            "describe the", "tell me about the course",
            "what is bbe", "what is bbec", "bbe"
        ],
        "boost": ["maharaja agrasen college", "affiliated", "undergraduate"],
    },
}


def detect_intent(query):
    query_lower = query.lower()
    scores = {}
    for intent, config in INTENT_MAP.items():
        score = sum(1 for kw in config["keywords"] if kw in query_lower)
        if score > 0:
            scores[intent] = score
    if not scores:
        return "general", []
    best = max(scores, key=scores.get)
    return best, INTENT_MAP[best]["boost"]


def is_listing_query(query):
    return any(kw in query.lower() for kw in LISTING_KEYWORDS)


def is_yesno_query(query):
    return any(kw in query.lower() for kw in YESNO_KEYWORDS)


def get_specific_subject(query):
    subjects = [
        "electronics", "computer science", "mathematics",
        "physics", "chemistry", "commerce", "english",
        "hindi", "history", "political science", "economics",
        "journalism", "business economics", "physical education",
        "biology", "bbe"
    ]
    q = query.lower()
    for subject in subjects:
        if subject in q:
            return subject
    return None


def expand_query(query, intent, boost_terms):
    if intent == "general" or not boost_terms:
        return query
    # Avoid diluting acronym queries like BBE or specific courses
    if any(acr in query.lower() for acr in ["bbe", "naac", "b.a. english", "electronics"]):
        return query
    return f"{query} {' '.join(boost_terms[:2])}"


def l2_to_similarity(l2_distance):
    return 1.0 / (1.0 + float(l2_distance))


def apply_url_boost(source, intent, score):
    source_lower = (source or "").lower()

    for pattern, boost in HIGH_VALUE_URLS.items():
        if pattern in source_lower:
            score += boost
            break

    if "faculties_detail" in source_lower and intent not in ["faculty", "principal"]:
        score -= 0.20

    if "macerp" in source_lower or "online payment" in source_lower.replace(" ", ""):
        score -= 0.80

    if "short-term-courses" in source_lower and intent == "department":
        score -= 0.60

    if "short-term-courses" in source_lower and intent == "contact":
        score -= 0.50

    if any(p in source_lower for p in ["notice-general", "student_council"]):
        score -= 0.25

    return score


def apply_intent_boost(text, source, intent, score, query=""):
    text_lower = (text or "").lower()
    source_lower = (source or "").lower()

    if intent == "principal":
        if "principal" in text_lower:
            score += 0.40
        if re.search(r"(dr\.|prof\.|mr\.|ms\.)", text_lower):
            score += 0.10
        if "principal.aspx" in source_lower:
            score += 0.30
        if "faculties_detail" in source_lower:
            score -= 0.30

    elif intent == "faculty":
        if any(w in text_lower for w in ["faculty", "professor", "lecturer"]):
            score += 0.20
        if "faculties_detail" in source_lower:
            score += 0.35
        if "faculty=faculty" in source_lower:
            score += 0.25
        query_lower = query.lower()
        if any(w in query_lower for w in ["dr.", "prof."]):
            skip = {"who", "is", "are", "dr", "prof", "the", "a", "an", "?"}
            name_words = [
                w.strip(".,?") for w in query_lower.split()
                if w.strip(".,?") not in skip and len(w.strip(".,?")) > 2
            ]
            if name_words and any(nw in text_lower for nw in name_words):
                score += 0.40

    elif intent == "department":
        if any(w in text_lower for w in ["department", "programme", "course", "undergraduate", "bachelor", "fyup"]):
            score += 0.25
        if any(w in text_lower for w in ["bcom", "bsc", "ba ", "btech", "bbe"]):
            score += 0.25
        if "departments.aspx" in source_lower:
            score += 0.25
        if "courses.aspx" in source_lower:
            score += 0.30
        if query:
            query_lower_d = query.lower()
            SUBJECTS = [
                "electronics", "commerce", "mathematics", "physics",
                "chemistry", "computer", "english", "hindi", "history",
                "journalism", "economics", "political", "biology",
                "business economics", "physical education", "bbe"
            ]
            for subj in SUBJECTS:
                if subj in query_lower_d and (subj in source_lower or subj in text_lower):
                    score += 0.45
                    break
        if is_yesno_query(query):
            subject = get_specific_subject(query)
            if subject and (subject in source_lower or subject in text_lower):
                score += 0.50

    elif intent == "admission":
        if any(w in text_lower for w in ["admission", "eligibility"]):
            score += 0.20

    elif intent == "fee":
        if any(w in text_lower for w in ["fee", "fees", "charges", "hostel"]):
            score += 0.25

    elif intent == "contact":
        if any(w in text_lower for w in ["address", "vasundhara", "110096", "+91", "phone", "telephone", "email", "principal@", "contact"]):
            score += 0.40
        if "contactus.aspx" in source_lower:
            score += 0.30

    elif intent == "exam":
        if any(w in text_lower for w in ["exam", "result", "timetable"]):
            score += 0.20

    elif intent == "identity":
        if any(w in text_lower for w in ["maharaja agrasen", "established", "affiliated", "university of delhi", "accredited", "naac", "vision", "mission"]):
            score += 0.35
        if "bbe" in query.lower() and "bbe" in text_lower:
            score += 0.50
        COURSE_INTRO_PATTERNS = [
            "commerce-intro", "english-intro", "hindi-intro",
            "journalism-intro", "business-economics-intro",
            "electronics-intro", "mathematics-intro",
            "physics-intro", "chemistry-intro", "biology-intro",
            "computersci-intro", "political-science", "degree_details"
        ]
        if any(p in source_lower for p in COURSE_INTRO_PATTERNS):
            score += 0.40

    return score


def _fallback_results(results, k):
    if not results:
        return []
    k = max(1, int(k))
    return results[:k]


# ==============================================================================
# RECIPROCAL RANK FUSION (RRF) & HYBRID RETRIEVAL
# ==============================================================================
def reciprocal_rank_fusion(dense_ranks: list, sparse_ranks: list, k_rrf=60) -> dict:
    """
    RRF Score: sum( 1.0 / (k_rrf + rank) )
    dense_ranks: list of idx
    sparse_ranks: list of idx
    """
    rrf_scores = {}
    for rank, idx in enumerate(dense_ranks):
        rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 / (k_rrf + rank + 1))
    for rank, idx in enumerate(sparse_ranks):
        rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 / (k_rrf + rank + 1))
    return rrf_scores


def search(query, k=None):
    intent, boost_terms = detect_intent(query)
    print(f"[INTENT] {intent}")

    listing = is_listing_query(query)
    yesno = is_yesno_query(query)

    # is_existence_query already imported at top level
    existence = is_existence_query(query)

    if k is None:
        if existence:
            k = TOP_K_DEFAULT
        else:
            k = TOP_K_LISTING if listing else TOP_K_DEFAULT
    k = max(1, int(k))

    print(f"[SEARCH] Querying hybrid index (listing={listing}, yesno={yesno}, existence={existence})")

    # ── 1. Dense FAISS Retrieval ──────────────────────────────────────────────
    expanded = expand_query(query, intent, boost_terms)
    q_emb = model.encode([expanded], normalize_embeddings=False)
    D, I = index.search(np.array(q_emb, dtype="float32"), TOP_K_CANDIDATES)
    dense_candidates = [int(idx) for idx in I[0] if idx != -1]

    # ── 2. Sparse Lexical BM25 Retrieval ──────────────────────────────────────
    query_tokens = bm25_tokenize(query)
    if query_tokens:
        bm25_scores = bm25.get_scores(query_tokens)
        top_bm25_indices = np.argsort(bm25_scores)[::-1][:TOP_K_CANDIDATES]
        sparse_candidates = [int(idx) for idx in top_bm25_indices if bm25_scores[idx] > 0]
    else:
        sparse_candidates = []

    # ── 3. Reciprocal Rank Fusion (RRF) ───────────────────────────────────────
    fused_scores = reciprocal_rank_fusion(dense_candidates, sparse_candidates, k_rrf=60)
    if not fused_scores:
        print("[SEARCH] No candidates found across Dense and Sparse retrieval.")
        return []

    # Apply URL & Intent domain boosts to fused candidates
    scored_candidates = []
    for idx, rrf_score in fused_scores.items():
        item = metadata[idx]
        text = item.get("text", "")
        source = item.get("source", "unknown") or "unknown"

        # Scale RRF score to [0, 1] baseline and add domain bonuses
        score = float(rrf_score) * 15.0
        score = apply_url_boost(source, intent, score)
        score = apply_intent_boost(text, source, intent, score, query)

        if len(text.split()) < 20:
            score -= 0.10

        scored_candidates.append({
            "idx": idx,
            "text": text,
            "source": source,
            "type": item.get("type", "general"),
            "score": round(score, 4),
        })

    scored_candidates.sort(key=lambda x: x["score"], reverse=True)

    # ── 4. Cross-Encoder Neural Reranking (Top 15 Pool) ───────────────────────
    top_pool = scored_candidates[:15]
    if reranker is not None and top_pool:
        try:
            pairs = [[query, doc["text"][:500]] for doc in top_pool]
            ce_scores = reranker.predict(pairs)
            for i, doc in enumerate(top_pool):
                # Blend Cross-Encoder score (logit) with domain score
                ce_val = float(ce_scores[i])
                doc["score"] = round(doc["score"] + (ce_val * 0.5), 4)
            top_pool.sort(key=lambda x: x["score"], reverse=True)
            scored_candidates = top_pool + scored_candidates[15:]
        except Exception as e:
            print(f"[RERANK] Warning during Cross-Encoder: {e}")

    filtered = [r for r in scored_candidates if r["score"] >= MIN_SCORE]
    if not filtered:
        filtered = _fallback_results(scored_candidates, k)

    # ── 5. Source Deduplication ───────────────────────────────────────────────
    max_per_source = 1 if intent == "principal" else 2
    seen_sources = {}
    deduplicated = []
    for r in filtered:
        src = r["source"]
        count = seen_sources.get(src, 0)
        if count < max_per_source:
            deduplicated.append(r)
            seen_sources[src] = count + 1

    top = deduplicated[:k]
    print(f"[SEARCH] After hybrid fusion & reranking: using top {len(top)} chunks.")
    for i, r in enumerate(top):
        print(f"  [{i+1}] score={r['score']} | {r['source']}")

    return top


def answer(query):
    results = search(query)

    if not results:
        return "No data found"

    if not any(r.get("text", "").strip() for r in results):
        return "No data found"

    return generate_answer(query, results)


def reload_index():
    global index, metadata, tokenized_corpus, bm25
    print("[RELOAD] Reloading FAISS and metadata...")
    index = faiss.read_index(INDEX_FILE)
    with open(META_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    tokenized_corpus = [bm25_tokenize(doc.get("text", "")) for doc in metadata]
    try:
        from rank_bm25 import BM25Okapi
        bm25 = BM25Okapi(tokenized_corpus)
    except Exception:
        pass
    print(f"[RELOAD] Complete. {len(metadata)} chunks active.")


if __name__ == "__main__":
    print("College Hybrid RAG Chatbot — testing query\n")
    sample_queries = [
        "What is BBE course?",
        "Does MAC offer B.A. English?",
        "Who is the principal?",
        "What is the NAAC grade of MAC?"
    ]
    for q in sample_queries:
        print(f"\nQ: {q}")
        chunks = search(q, k=3)
        for c in chunks:
            print(f"  -> {c['source']} | Score: {c['score']}")
