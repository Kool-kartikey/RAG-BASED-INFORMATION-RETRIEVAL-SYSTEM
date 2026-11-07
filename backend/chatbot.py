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

STATIC_DU_VC = """Vice-Chancellor of the University of Delhi (DU):
- Name : Prof. Yogesh Singh
- Role : Vice-Chancellor, University of Delhi
- Office : Vice-Chancellor Office, North Campus, University of Delhi, Delhi – 110007"""

STATIC_DU_ABOUT = """About the University of Delhi (DU):
- Established   : 1922 (Centenary Year celebrated in 2022)
- Type          : Premier Collegiate Central University
- Vice-Chancellor : Prof. Yogesh Singh
- Campuses      : North Campus, South Campus, and 90+ Constituent Colleges
- Accreditation : NAAC Grade 'A+'
- Official Portal : https://www.du.ac.in
- Admissions    : https://admission.uod.ac.in (CSAS Portal)"""

STATIC_DU_CONTACT = """Contact Details — University of Delhi (DU):
- North Campus : University of Delhi, Chhatra Marg, Delhi – 110007
- South Campus : Benito Juarez Marg, Dhaula Kuan, New Delhi – 110021
- Official Website : https://www.du.ac.in
- Admissions Portal : https://admission.uod.ac.in"""

STATIC_DU_COLLEGES = """Colleges under University of Delhi (DU):
The University of Delhi has more than 90 constituent and affiliated colleges across North Campus, South Campus, and Off-Campus locations.
Prominent colleges include:
- North Campus: Miranda House, Hindu College, St. Stephen's, SRCC, Kirori Mal, Ramjas, Hansraj
- South Campus: Lady Shri Ram (LSR), Sri Venkateswara, Gargi, Jesus and Mary (JMC), ARSD
- East/Off-Campus: Maharaja Agrasen College (MAC), Deen Dayal Upadhyaya (DDU), Acharya Narendra Dev (ANDC)
Directory: https://www.du.ac.in/index.php?page=colleges"""

STATIC_DU_CSAS = """Common Seat Allocation System (CSAS) — University of Delhi:
- Purpose: Centralized admission portal for UG and PG admissions across all 90+ DU colleges.
- Eligibility: Valid CUET (Common University Entrance Test) scorecard.
- Three-Phase Process:
  1. Phase 1: Registration on the CSAS portal.
  2. Phase 2: Selection of programme and college preference combinations.
  3. Phase 3: Seat allocation and document verification.
- Official Portal: https://admission.uod.ac.in"""

STATIC_DU_CGPA = """CGPA Conversion Formula — University of Delhi:
As per the official University of Delhi notification, the formula for converting Cumulative Grade Point Average (CGPA) to Percentage of Marks is:
  Percentage of Marks = 9.5 × CGPA
Example: A CGPA of 8.0 corresponds to 8.0 × 9.5 = 76.0%."""

# =========================
# STATIC TRIGGERS
# Checked before RAG — fastest response path.
# Order matters: stream-specific courses checked before full course list.
# Principal checked first to avoid RAG ambiguity.
# =========================

STATIC_TRIGGERS = {
    "du_vc": {
        "patterns": [
            "who is the vice chancellor", "vice chancellor of du",
            "vice-chancellor", "vice chancellor", "vc of du",
            "vc of delhi university", "who is the vc", "current vc",
            "head of du", "head of delhi university", "who is the vice-chancellor",
        ],
        "response": STATIC_DU_VC,
    },
    "du_csas": {
        "patterns": [
            "what is csas", "csas portal", "how to apply to du",
            "du admission process", "cuet admission in du", "csas admission",
            "csas",
        ],
        "response": STATIC_DU_CSAS,
    },
    "du_colleges": {
        "patterns": [
            "how many colleges in du", "how many colleges are in du",
            "how many colleges are under du", "colleges under du",
            "colleges in du", "colleges of du", "list of colleges in du",
            "du colleges list", "90+ colleges", "constituent colleges",
        ],
        "response": STATIC_DU_COLLEGES,
    },
    "du_cgpa": {
        "patterns": [
            "cgpa conversion", "cgpa to percentage", "cgpa formula",
            "calculate percentage from cgpa", "du grading system", "grading formula",
        ],
        "response": STATIC_DU_CGPA,
    },
    "du_contact": {
        "patterns": [
            "contact du", "contact delhi university", "address of du",
            "address of delhi university", "location of du", "du address",
            "chhatra marg", "benito juarez",
        ],
        "response": STATIC_DU_CONTACT,
    },
    "du_about": {
        "patterns": [
            "about du", "about delhi university", "about university of delhi",
            "tell me about du", "what is du", "what is delhi university",
            "overview of du",
        ],
        "response": STATIC_DU_ABOUT,
    },
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
    "naac_mac": {
        "patterns": [
            "naac of mac", "naac grade of mac", "grade of mac", "grade of maharaja",
            "is mac naac", "is mac accredited",
        ],
        "response": "Maharaja Agrasen College (MAC) NAAC Grade A.",
    },
    "naac_du": {
        "patterns": [
            "naac of du", "naac grade of du", "naac of delhi university",
            "naac grade of delhi university", "grade of du", "grade of delhi university",
            "is du naac accredited",
        ],
        "response": "The University of Delhi (DU) is accredited with an 'A+' Grade by NAAC.",
    },
    "naac": {
        "patterns": [
            "naac grade", "naac accreditation", "naac status", "what is the naac grade",
        ],
        "response": "University of Delhi (DU) NAAC Grade: A+ | Maharaja Agrasen College (MAC) NAAC Grade: A.",
    },
    "about": {
        "patterns": [
            "about mac", "about maharaja agrasen", "about the college",
            "tell me about mac", "what is mac",
            "what is maharaja agrasen college",
            "information about mac", "info about mac",
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


def check_naac_query(query: str):
    """
    Bulletproof disambiguation for NAAC accreditation queries.
    Distinguishes between University of Delhi (Grade A+) and Maharaja Agrasen College (Grade A).
    """
    ql = query.lower().strip()
    if "naac" in ql or "accreditation" in ql:
        has_du = any(t in ql for t in ["du", "delhi university", "uod", "central university", "university of delhi"])
        has_mac = any(t in ql for t in ["mac", "maharaja", "agrasen", "college"])
        if has_du and not has_mac:
            return "The University of Delhi (DU) is accredited with an 'A+' Grade by the National Assessment and Accreditation Council (NAAC)."
        if has_mac and not has_du:
            return "Maharaja Agrasen College (MAC) NAAC Grade A."
        return "University of Delhi (DU) NAAC Grade: A+ | Maharaja Agrasen College (MAC) NAAC Grade: A."
    return None


def check_static_response(query: str):
    """
    Returns a pre-installed static response if the query matches
    a known stable topic. Returns None otherwise.
    """
    # 1. NAAC accreditation disambiguation
    naac_resp = check_naac_query(query)
    if naac_resp:
        return naac_resp

    ql = query.lower().strip()
    for _, config in STATIC_TRIGGERS.items():
        for pattern in config["patterns"]:
            if pattern in ql:
                return config["response"]
    return None


OFFICIAL_COURSES = {
    "electronics": ("B.Sc.(H) Electronics", "Science"),
    "commerce": ("B.Com.(H)", "Commerce"),
    "b.com": ("B.Com.(H)", "Commerce"),
    "bcom": ("B.Com.(H)", "Commerce"),
    "english": ("B.A.(H) English", "Arts"),
    "journalism": ("B.A.(H) Journalism", "Arts"),
    "hindi": ("B.A.(H) Hindi", "Arts"),
    "political science": ("B.A.(H) Political Science", "Arts"),
    "business economics": ("B.A.(H) Business Economics", "Arts"),
    "bbe": ("B.A.(H) Business Economics", "Arts"),
    "mathematics": ("B.Sc.(G) Mathematics", "Science"),
    "maths": ("B.Sc.(G) Mathematics", "Science"),
    "physical science": ("B.Sc.(Physical Science)", "Science"),
}


def check_course_query(query: str):
    ql = query.lower().strip()

    # 1. Course description queries: e.g. "what is bbe", "what is b.a.(h) english"
    if any(p in ql for p in ["what is", "tell me about", "describe"]) and not any(p in ql for p in ["principal", "college", "mac", "address", "website", "faculty"]):
        for key, (full_name, stream) in OFFICIAL_COURSES.items():
            if re.search(r"\b" + re.escape(key) + r"\b", ql):
                if key == "bbe":
                    return "BBE stands for Business Economics, full name B.A.(H) Business Economics offered at MAC."
                return f"{full_name} is an undergraduate {stream.lower()} course offered at Maharaja Agrasen College."

    # 2. Course existence queries: e.g. "does mac offer X", "is X available", "can i study X"
    if any(p in ql for p in ["does mac offer", "does mac have", "is there a", "available at mac", "offered at mac", "can i study", "do they offer", "does the college offer", "is b.com", "is bcom", "is b.a.", "is ba ", "is b.sc", "is bsc"]):
        for key, (full_name, _) in OFFICIAL_COURSES.items():
            if re.search(r"\b" + re.escape(key) + r"\b", ql):
                return f"Yes, MAC offers {full_name}."

    return None


def check_conversational_intent(query: str):
    """
    Handles conversational intents (greetings, courtesy, self-identity)
    instantly. Prevents 'No data found' errors for everyday social inputs.
    """
    ql = query.lower().strip()
    ql_clean = re.sub(r"[?!.,;:\-—]", " ", ql)
    tokens = [t for t in ql_clean.split() if t]

    if not tokens:
        return None

    first_token = tokens[0]

    # Greetings: "hello", "hi", "hey", "namaste", "pranam", "good morning", etc.
    if first_token in {"hello", "hi", "hey", "namaste", "namaskar", "pranam", "hola", "yo"} or ql in {"hlo", "helo"}:
        return (
            "🙏 Namaste! I am your University of Delhi (DU) & College AI Assistant. "
            "You can ask me about admissions (CSAS/CUET), courses, 90+ constituent colleges, "
            "examination rules, CGPA conversion, or university administration. How can I help you today?"
        )

    if any(p in ql for p in ["good morning", "good afternoon", "good evening", "good day"]):
        return (
            "Good day! Welcome to the University of Delhi AI Information Portal. "
            "How can I assist you with your academic or admission queries today?"
        )

    if any(p in ql for p in ["how are you", "how do you do", "kya haal", "kaise ho"]):
        return (
            "I am doing well, thank you! Ready to assist you with verified information "
            "regarding the University of Delhi and its constituent colleges. What would you like to know?"
        )

    if any(p in ql for p in ["who are you", "what are you", "what is your name", "what can you do", "what do you do", "tell me about yourself"]) or ql in {"help", "help me"}:
        return (
            "I am DU Samvaad, an AI information concierge designed for the University of Delhi (DU) "
            "and its constituent colleges (including Maharaja Agrasen College). "
            "I provide grounded answers on CSAS admissions, courses, faculty, CGPA conversion, datesheets, "
            "and institutional policies."
        )

    if any(p in ql for p in ["thank you", "thanks", "dhanyawad", "shukriya", "thank u"]):
        return "You're very welcome! Feel free to ask if you have any more questions about the University of Delhi or its colleges. Best wishes!"

    if any(p in ql for p in ["bye", "goodbye", "see you", "alvida"]):
        return "Goodbye! Have a great day ahead and best wishes for your academic journey at the University of Delhi."

    return None


# =========================
# MAC COLLEGE GUARDRAILS
# =========================

COLLEGE_TOPICS = [
    # People
    "principal", "director", "hod", "faculty", "professor",
    "teacher", "lecturer", "staff", "dean", "registrar",
    "dr.", "prof.", "vice-chancellor", "vice chancellor", "vc",
    "proctor", "dsw",

    # Academics
    "department", "course", "programme", "subject", "syllabus",
    "semester", "exam", "examination", "timetable",
    "datesheet", "attendance", "cgpa", "ugcf", "nep",

    # Admissions
    "admission", "apply", "eligibility", "cutoff", "merit",
    "enrollment", "csas", "cuet",

    # Finance
    "fee", "fees", "scholarship", "tuition", "charges",

    # College life
    "hostel", "canteen", "sports", "fest",
    "nss", "ncc", "committee", "placement",

    # Institutional identity
    "college", "mac", "maharaja", "agrasen",
    "delhi university", "university of delhi", "du", "uod",
    "campus", "contact", "address", "email",
    "website", "location", "vision", "mission",
    "iqac", "full form", "stands for",
    "accreditation", "naac",
    "does mac", "does the college",
    "mac.du.ac.in", "du.ac.in",
    "north campus", "south campus", "chhatra marg", "benito juarez",
]

SHORT_TOKENS = {"mac", "nss", "ncc", "hod", "fee", "du", "vc"}

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
    "I can only answer questions related to the University of Delhi (DU) "
    "and its constituent colleges (including Maharaja Agrasen College). "
    "Please ask me about admissions (CSAS/CUET), courses, departments, "
    "colleges directory, fees, examination rules, or administration."
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
import threading

def _async_mongo_log(query: str, response: str):
    try:
        from mongodb_backup import log_chat
        log_chat(question=query, answer=response)
    except Exception:
        pass

def log_to_mongo(query: str, response: str):
    """Non-blocking fire-and-forget logging. Never delays response."""
    threading.Thread(target=_async_mongo_log, args=(query, response), daemon=True).start()


# =========================
# MAIN ANSWER FUNCTION
# =========================

def get_answer(query: str) -> str:
    """Single entry point used by CLI and FastAPI."""
    query = query.strip()

    if not query:
        return "Please ask a question."

    # ── Step 0: Conversational intent check (greetings, courtesy) ─────
    convo = check_conversational_intent(query)
    if convo:
        log_to_mongo(query, convo)
        return convo

    if len(query) < 3:
        return "Please ask a proper question."

    if len(query) > 300:
        return "Please keep your question under 300 characters."

    # ── Step 1: Static response check (fastest path) ──────────────────
    static = check_static_response(query)
    if static:
        print(f"[STATIC] Returning pre-installed response")
        log_to_mongo(query, static)
        return static

    # ── Step 1.5: Deterministic course catalog check ──────────────────
    course_ans = check_course_query(query)
    if course_ans:
        print(f"[COURSE_CATALOG] Returning verified course answer")
        log_to_mongo(query, course_ans)
        return course_ans

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