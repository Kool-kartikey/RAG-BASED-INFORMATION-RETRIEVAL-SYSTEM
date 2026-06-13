import sys
import os
import re

sys.path.append(os.path.join(os.path.dirname(__file__), "backend"))

from retriever import answer as rag_answer

# =========================
# STATIC RESPONSES
# Pre-installed for stable institutional facts that rarely change.
# Bypasses RAG entirely — guaranteed accurate, zero latency.
# =========================

STATIC_CONTACT = """Contact Details — Maharaja Agrasen College:
- Address : Maharaja Agrasen College, Vasundhara Enclave, Delhi – 110096
- Phone   : +91-11-22610563
- Email   : principal@mac.du.ac.in
- Website : https://mac.du.ac.in"""

STATIC_ABOUT = """About Maharaja Agrasen College (MAC):
- Full Name   : Maharaja Agrasen College
- Affiliation : University of Delhi
- Address     : Vasundhara Enclave, Delhi – 110096
- Phone       : +91-11-22610563
- Email       : principal@mac.du.ac.in
- Website     : https://mac.du.ac.in
- NAAC Grade  : A"""

STATIC_DEPARTMENTS = """Departments at Maharaja Agrasen College (MAC):
1.  Physics
2.  Chemistry
3.  Mathematics
4.  Biology
5.  Electronics
6.  Computer Science
7.  Commerce
8.  English
9.  Hindi
10. Political Science
11. Economics
12. History
13. Business Economics
14. Journalism
15. Physical Education"""

STATIC_COURSES = """Courses offered at Maharaja Agrasen College (MAC):

ARTS:
1. B.A.(H) Business Economics (BBE)
2. B.A.(H) English
3. B.A.(H) Journalism
4. B.A.(H) Hindi
5. B.A.(H) Political Science

COMMERCE:
6. B.Com.(H)

SCIENCE:
7. B.Sc.(Physical Science)
8. B.Sc.(H) Electronics
9. B.Sc.(G) Mathematics"""

STATIC_COURSES_ARTS = """Arts courses at Maharaja Agrasen College (MAC):
1. B.A.(H) Business Economics (BBE)
2. B.A.(H) English
3. B.A.(H) Journalism
4. B.A.(H) Hindi
5. B.A.(H) Political Science"""

STATIC_COURSES_COMMERCE = """Commerce courses at Maharaja Agrasen College (MAC):
1. B.Com.(H)"""

STATIC_COURSES_SCIENCE = """Science courses at Maharaja Agrasen College (MAC):
1. B.Sc.(Physical Science)
2. B.Sc.(H) Electronics
3. B.Sc.(G) Mathematics"""

STATIC_STREAMS = """Maharaja Agrasen College (MAC) offers undergraduate programmes in three streams:
1. Arts
2. Commerce
3. Science"""

STATIC_PRINCIPAL = """Principal of Maharaja Agrasen College (MAC):
- Name : Prof. Sanjeev Kumar Tiwari
- Role : Principal, Maharaja Agrasen College, University of Delhi"""

# =========================
# STATIC TRIGGERS
# Checked before RAG — fastest response path.
# Order matters: stream-specific courses checked before full course list.
# Principal checked first to avoid RAG ambiguity.
# =========================

STATIC_TRIGGERS = {
    "principal": {
        "patterns": [
            "who is the principal",
            "principal of mac",
            "principal of maharaja",
            "principal of the college",
            "name of the principal",
            "name of principal",
            "current principal",
            "college principal",
            "who heads mac",
            "who heads the college",
            "head of mac",
            "head of the college",
        ],
        "response": STATIC_PRINCIPAL,
    },
    "contact": {
        "patterns": [
            "contact detail", "contact info", "contact us",
            "phone number", "phone of mac", "phone of college",
            "email of mac", "email of college", "email id of mac",
            "email id", "contact email", "official email",
            "address of mac", "address of college", "address of maharaja",
            "how to contact", "how to reach", "reach the college",
            "helpline", "contact number",
            "pin code", "pincode", "postal code", "zip code",
            "vasundhara enclave", "which area", "what area",
            "locality of mac", "location of mac",
            "official website of mac", "official website of maharaja",
            "website of mac", "website of maharaja agrasen",
            "mac website", "mac college website",
            "mac.du.ac.in", "what is the website",
            "link of mac", "url of mac",
        ],
        "response": STATIC_CONTACT,
    },
    "about": {
        "patterns": [
            "about mac", "about maharaja agrasen", "about the college",
            "tell me about mac", "what is mac",
            "what is maharaja agrasen college",
            "information about mac", "info about mac",
            "naac grade", "naac accreditation", "naac of mac",
            "is mac naac", "is mac accredited", "naac status",
            "overview of mac", "introduction to mac",
        ],
        "response": STATIC_ABOUT,
    },
    "streams": {
        "patterns": [
            "streams at mac", "streams in mac", "streams available",
            "what streams", "which streams", "how many streams",
            "streams offered", "streams does mac offer",
        ],
        "response": STATIC_STREAMS,
    },
    "departments": {
        "patterns": [
            "all departments", "list departments", "list all departments",
            "what are the departments", "which departments",
            "departments in mac", "departments at mac",
            "departments of mac", "how many departments",
            "what departments does mac have", "all dept",
            "departments available", "departments offered",
        ],
        "response": STATIC_DEPARTMENTS,
    },
    "courses_arts": {
        "patterns": [
            "arts courses", "arts stream courses", "courses in arts",
            "list arts courses", "all arts courses",
            "what are the arts courses", "which arts courses",
            "courses for arts", "arts programmes", "ba courses",
            "b.a. courses", "arts stream at mac",
        ],
        "response": STATIC_COURSES_ARTS,
    },
    "courses_commerce": {
        "patterns": [
            "commerce courses", "commerce stream courses",
            "courses in commerce", "list commerce courses",
            "all commerce courses", "what are the commerce courses",
            "which commerce courses", "courses for commerce",
            "commerce programmes", "bcom courses", "b.com courses",
            "commerce stream at mac", "list only the commerce stream",
        ],
        "response": STATIC_COURSES_COMMERCE,
    },
    "courses_science": {
        "patterns": [
            "science courses", "science stream courses",
            "courses in science", "list science courses",
            "all science courses", "what are the science courses",
            "which science courses", "courses for science",
            "science programmes", "bsc courses", "b.sc courses",
            "science stream at mac",
        ],
        "response": STATIC_COURSES_SCIENCE,
    },
    "courses": {
        "patterns": [
            "all courses", "list courses", "list all courses",
            "what are the courses", "which courses",
            "courses in mac", "courses at mac", "courses offered",
            "courses of mac", "all programmes", "list programmes",
            "what programmes", "available courses", "available programmes",
            "undergraduate courses", "ug courses", "all ug",
            "what courses does mac offer", "courses available at mac",
            "all undergraduate", "list undergraduate",
        ],
        "response": STATIC_COURSES,
    },
}


def check_static_response(query: str):
    """
    Returns a pre-installed static response if the query matches
    a known stable topic. Returns None otherwise.
    """
    ql = query.lower().strip()
    for _, config in STATIC_TRIGGERS.items():
        for pattern in config["patterns"]:
            if pattern in ql:
                return config["response"]
    return None


# =========================
# MAC COLLEGE GUARDRAILS
# =========================

COLLEGE_TOPICS = [
    # People
    "principal", "director", "hod", "faculty", "professor",
    "teacher", "lecturer", "staff", "dean", "registrar",
    "dr.", "prof.",

    # Academics
    "department", "course", "programme", "subject", "syllabus",
    "semester", "exam", "examination", "timetable",
    "datesheet", "attendance",

    # Admissions
    "admission", "apply", "eligibility", "cutoff", "merit",
    "enrollment",

    # Finance
    "fee", "fees", "scholarship", "tuition", "charges",

    # College life
    "hostel", "canteen", "sports", "fest",
    "nss", "ncc", "committee", "placement",

    # College identity
    "college", "mac", "maharaja", "agrasen",
    "delhi university", "university of delhi",
    "campus", "contact", "address", "email",
    "website", "location", "vision", "mission",
    "iqac", "full form", "stands for",
    "accreditation", "naac",
    "does mac", "does the college",
    "mac.du.ac.in",
]

SHORT_TOKENS = {"mac", "nss", "ncc", "hod", "fee"}

BLOCKED_TOPICS = [
    # Weather
    "weather", "temperature", "rain", "sunny", "forecast",
    "climate today", "humidity",

    # News
    "breaking news", "today's headline",

    # Sports unrelated to college
    "cricket score", "ipl", "football match", "fifa",
    "cricket team", "cricket player", "cricket captain",
    "nba", "match score", "live score",

    # Entertainment
    "movie", "film", "web series", "netflix",
    "song", "music", "album", "actor", "actress",
    "despacito", "bollywood", "hollywood",

    # Food
    "recipe", "restaurant near", "food delivery",
    "how do i cook", "how to cook",

    # Finance / crypto
    "stock price", "share price", "bitcoin", "crypto",
    "sensex", "nifty", "forex",

    # Politics
    "prime minister", "president of india",
    "election result", "parliament",
    "narendra modi", "rahul gandhi", "amit shah",
    "who is the cm", "who is the pm",
    "chief minister of",

    # Personal / social
    "my girlfriend", "my boyfriend",
    "relationship advice", "horoscope", "astrology",
    "joke", "funny", "meme",

    # Other institutions
    "bits pilani", "jnu", "jadavpur",
    "anna university", "harvard", "oxford",

    # Coding / AI
    "write a ", "write me a ", "write code",
    "debug my", "code for me", "program for me",
    "script for me", "chatgpt", "openai", "gemini",
    "translate this", "summarize this",

    # Celebrities
    "elon musk", "shah rukh khan", "virat kohli",
    "who is the captain", "captain of india",

    # General knowledge / science
    "machine learning", "deep learning", "artificial intelligence",
    "black hole", "solar system", "quantum", "relativity",
    "photosynthesis", "water cycle",
    "capital of", "currency of", "population of",
    "who won the", "who is the president",
    "what is the capital", "history of india",
    "cryptocurrency",
]

OFF_TOPIC_RESPONSE = (
    "I can only answer questions related to "
    "Maharaja Agrasen College (MAC). Please ask me about "
    "admissions, departments, faculty, courses, fees, "
    "events, or any other college-related topic."
)


def is_mac_related(query: str) -> tuple:
    """
    Three-pass guardrail. Returns (is_allowed: bool, reason: str).

    Pass 1 — Hard block  : BLOCKED_TOPICS → reject immediately.
    Pass 2 — Allowlist   : COLLEGE_TOPICS match → allow.
    Pass 3 — Lenient     : send to RAG, returns No data found naturally.
    """
    query_lower = query.lower().strip()

    # Pass 1: Hard block
    for blocked in BLOCKED_TOPICS:
        bs = blocked.strip()
        if len(bs) <= 4 and " " not in bs:
            if re.search(r"\b" + re.escape(bs) + r"\b", query_lower):
                return False, f"BLOCKED: '{blocked}'"
        else:
            if blocked in query_lower:
                return False, f"BLOCKED: '{blocked}'"

    # Pass 2: College allowlist
    for topic in COLLEGE_TOPICS:
        if topic in SHORT_TOKENS:
            if re.search(r"\b" + re.escape(topic) + r"\b", query_lower):
                return True, f"ALLOWED: '{topic}'"
        else:
            if topic in query_lower:
                return True, f"ALLOWED: '{topic}'"

    # Pass 3: Lenient fallback
    return True, "ALLOWED: lenient fallback"


# =========================
# MONGODB CHAT LOGGER
# =========================

def log_to_mongo(query: str, response: str):
    """Silently logs to MongoDB. Never breaks main flow."""
    try:
        from mongodb_backup import log_chat
        log_chat(question=query, answer=response)
    except Exception:
        pass


# =========================
# MAIN ANSWER FUNCTION
# =========================

def get_answer(query: str) -> str:
    """Single entry point used by CLI and FastAPI."""
    query = query.strip()

    if not query or len(query) < 3:
        return "Please ask a proper question."

    if len(query) > 300:
        return "Please keep your question under 300 characters."

    # ── Step 1: Static response check (fastest path) ──────────────────
    static = check_static_response(query)
    if static:
        print(f"[STATIC] Returning pre-installed response")
        log_to_mongo(query, static)
        return static

    # ── Step 2: Guardrail check ───────────────────────────────────────
    allowed, reason = is_mac_related(query)
    print(f"[GUARDRAIL] {reason}")

    if not allowed:
        log_to_mongo(query, OFF_TOPIC_RESPONSE)
        return OFF_TOPIC_RESPONSE

    # ── Step 3: Course not-offered check ─────────────────────────────
    from backend.llm import is_not_offered_query
    if is_not_offered_query(query):
        response = "No data found"
        print(f"[NOT_OFFERED] Query matches known non-MAC course")
        log_to_mongo(query, response)
        return response

    # ── Step 4: RAG pipeline ──────────────────────────────────────────
    try:
        response = rag_answer(query) or "No data found"
        log_to_mongo(query, response)
        return response
    except Exception as e:
        print(f"[ERROR] {e}")
        return "Something went wrong. Please try again."


# =========================
# CLI INTERFACE
# =========================

BANNER = """
╔══════════════════════════════════════════════════════╗
║       Maharaja Agrasen College - RAG Chatbot         ║
║              Powered by Llama 3.2 3B                 ║
║         Data Source: mac.du.ac.in                    ║
╚══════════════════════════════════════════════════════╝
Type your question and press Enter.
Type 'exit' to quit | Type 'help' for sample questions.
"""

HELP_TEXT = """
Sample Questions:
  -> Who is the principal?
  -> What departments are available?
  -> What courses does the college offer?
  -> What is the address of the college?
  -> How can I contact the college?
  -> What is the full form of MAC?
"""


def chat():
    print(BANNER)
    while True:
        try:
            query = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye!")
            break

        if not query:
            continue
        if query.lower() in ["exit", "quit", "bye"]:
            print("\nGoodbye!")
            break
        if query.lower() == "help":
            print(HELP_TEXT)
            continue

        print("\nBot:", get_answer(query), "\n")
        print("=" * 60)


if __name__ == "__main__":
    chat()