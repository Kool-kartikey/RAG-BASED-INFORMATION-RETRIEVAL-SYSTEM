import faiss
import json
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from backend.llm import generate_answer

INDEX_FILE = "data/faiss.index"
META_FILE = "data/meta.json"

TOP_K_CANDIDATES = 80
TOP_K_DEFAULT = 5
TOP_K_LISTING = 40
MIN_SCORE = 0.10

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
    "english-intro": 0.20,
    "computersci-intro": 0.20,
    "journalism-intro": 0.20,
    "hindi-intro": 0.20,
    "physics-intro": 0.20,
    "political-science": 0.20,
    "business-economics-intro": 0.20,
    "economics-intro": 0.20,
    "biology-intro": 0.20,
    # Department faculty listing pages (confirmed in meta.json)
    "electronics.aspx":          0.45,
    "commerce.aspx":             0.45,
    "computersci.aspx":          0.45,
    "mathematics.aspx":          0.45,
    "physics.aspx":              0.45,
    "english.aspx":              0.40,
    "hindi.aspx":                0.40,
    "history.aspx":              0.40,
    "journalism.aspx":           0.40,
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

print("[INIT] Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("[INIT] Loading FAISS index...")
index = faiss.read_index(INDEX_FILE)

print("[INIT] Loading metadata...")
with open(META_FILE, "r", encoding="utf-8") as f:
    metadata = json.load(f)

print(f"[INIT] Ready. {len(metadata)} chunks loaded.\n")


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
            "all departments", "all courses"
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
            # Course description queries — "what is B.A. English"
            "what is b.a.", "what is b.sc", "what is b.com",
            "what is bsc", "what is bcom", "what is ba ",
            "describe the", "tell me about the course",
            "what is bbe", "what is bbec",
        ],
        "boost": ["maharaja agrasen college", "established", "affiliated"],
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
        "biology"
    ]
    q = query.lower()
    for subject in subjects:
        if subject in q:
            return subject
    return None


def expand_query(query, intent, boost_terms):
    if intent == "general" or not boost_terms:
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

    # Penalise ERP/payment portal — never relevant for informational queries
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
        # Boost faculty detail pages for faculty queries
        # These pages contain individual faculty name, dept, email
        if "faculties_detail" in source_lower:
            score += 0.35
        # Boost department faculty listing pages
        if "faculty=faculty" in source_lower:
            score += 0.25
        # If query contains a specific name, boost chunks that contain it
        query_lower = query.lower()
        if any(w in query_lower for w in ["dr.", "prof."]):
            # Extract potential name words from query (skip dr./prof./who/is)
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
        if any(w in text_lower for w in ["bcom", "bsc", "ba ", "btech"]):
            score += 0.15
        if "departments.aspx" in source_lower:
            score += 0.25
        if "courses.aspx" in source_lower:
            score += 0.25
        # Boost department-specific pages when subject appears in both query and source
        if query:
            query_lower_d = query.lower()
            SUBJECTS = [
                "electronics", "commerce", "mathematics", "physics",
                "chemistry", "computer", "english", "hindi", "history",
                "journalism", "economics", "political", "biology",
                "business economics", "physical education",
            ]
            for subj in SUBJECTS:
                if subj in query_lower_d and subj in source_lower:
                    score += 0.45
                    break
        if is_yesno_query(query):
            subject = get_specific_subject(query)
            if subject and subject in source_lower:
                score += 0.50
            if subject and subject in text_lower:
                score += 0.20

    elif intent == "admission":
        if any(w in text_lower for w in ["admission", "eligibility"]):
            score += 0.20

    elif intent == "fee":
        if any(w in text_lower for w in ["fee", "fees", "charges"]):
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
        # Boost course intro pages for course description queries
        COURSE_INTRO_PATTERNS = [
            "commerce-intro", "english-intro", "hindi-intro",
            "journalism-intro", "business-economics-intro",
            "electronics-intro", "mathematics-intro",
            "physics-intro", "chemistry-intro", "biology-intro",
            "computersci-intro", "political-science",
        ]
        if any(p in source_lower for p in COURSE_INTRO_PATTERNS):
            score += 0.40

    return score


def _fallback_results(results, k):
    if not results:
        return []
    k = max(1, int(k))
    return results[:k]


def search(query, k=None):
    intent, boost_terms = detect_intent(query)
    print(f"[INTENT] {intent}")

    listing = is_listing_query(query)
    yesno = is_yesno_query(query)

    # Existence queries (does MAC offer X, is X available) must use
    # TOP_K_DEFAULT — not TOP_K_LISTING — to prevent LLM timeout
    from backend.llm import is_existence_query
    existence = is_existence_query(query)

    if k is None:
        if existence:
            k = TOP_K_DEFAULT
        else:
            k = TOP_K_LISTING if listing else TOP_K_DEFAULT
    k = max(1, int(k))

    print(f"[SEARCH] Using top {k} chunks (listing={listing}, yesno={yesno}, existence={existence})")

    expanded = expand_query(query, intent, boost_terms)
    q_emb = model.encode([expanded], normalize_embeddings=False)
    D, I = index.search(np.array(q_emb, dtype="float32"), TOP_K_CANDIDATES)

    results = []
    for dist, idx in zip(D[0], I[0]):
        if idx == -1:
            continue

        item = metadata[idx]
        text = item.get("text", "")
        source = item.get("source", "unknown") or "unknown"

        score = l2_to_similarity(dist)
        score = apply_url_boost(source, intent, score)
        score = apply_intent_boost(text, source, intent, score, query)

        if len(text.split()) < 25:
            score -= 0.10

        results.append({
            "text": text,
            "source": source,
            "type": item.get("type", "general"),
            "score": round(score, 4),
        })

    if not results:
        print("[SEARCH] No candidates found.")
        return []

    results.sort(key=lambda x: x["score"], reverse=True)
    filtered = [r for r in results if r["score"] >= MIN_SCORE]

    if not filtered:
        print("[SEARCH] No chunk crossed threshold; returning best raw candidates.")
        filtered = _fallback_results(results, k)

    print(f"[SEARCH] {len(filtered)} chunks selected.")
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
    print(f"[SEARCH] After dedup: {len(deduplicated)}. Using top {len(top)}.")
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


if __name__ == "__main__":
    print("College RAG Chatbot — type 'exit' to quit\n")
    while True:
        query = input("Ask: ").strip()
        if not query:
            continue
        if query.lower() == "exit":
            break
        print("\nAnswer:")
        print(answer(query))
        print("\n" + "=" * 60)