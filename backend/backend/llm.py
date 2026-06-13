import requests
from typing import Any, List, Optional, Union

API_URL = "http://localhost:1234/v1/chat/completions"
MODEL_NAME = "llama-3.2-3b-instruct"

MAX_CONTEXT_CHARS = 10000
MAX_TOKENS_OUT = 2048
REQUEST_TIMEOUT = 180

LISTING_CHARS_PER_CHUNK = 600
FACTUAL_CHARS_PER_CHUNK = 420

LISTING_KEYWORDS = [
    "how many", "list", "all", "what are", "what courses",
    "what departments", "what programmes",
    "offered", "tell me all", "show all", "departments are",
    "courses are", "programmes are", "programs are",
    "which departments", "which courses", "which programmes",
    "all departments", "all courses", "all programmes",
    # Faculty listing keywords
    "all faculty", "all teachers", "all professors", "all staff",
    "list faculty", "list teachers", "list professors",
    "faculty of department", "faculty in department",
    "faculty members", "teaching staff", "professors in",
    "teachers in", "who teaches", "who are the faculty",
]

# Existence queries — yes/no checks for courses, departments, programmes
# These must NOT be treated as listing queries to avoid timeout
EXISTENCE_KEYWORDS = [
    "does mac offer", "does mac have", "is there a",
    "do they offer", "does the college offer",
    "does the college have", "can i study",
    "is b.sc", "is ba ", "is b.a.", "is b.com", "is bcom",
    "is bsc", "is msc", "is m.sc",
    "available at mac", "offered at mac",
    "does mac provide",
]

# Courses definitively NOT offered at MAC as regular UG programmes.
# Any existence query mentioning these gets an immediate static rejection
# without touching the LLM — prevents false positives from IGNOU/dept pages.
COURSES_NOT_OFFERED = [
    "bca", "bachelor of computer applications",
    "mba", "master of business administration",
    "mca", "master of computer applications",
    "btech", "b.tech", "b.e.",
    "mtech", "m.tech",
    "llb", "law",
    "b.ed", "bed", "m.ed",
    "msc", "m.sc",
    "ma ", "m.a.",
    "phd", "ph.d",
    "bba", "bachelor of business administration",
    "b.sc chemistry", "bsc chemistry",
    "b.sc physics", "bsc physics",
    "b.sc computer", "bsc computer",
    "b.sc biology", "bsc biology",
    "b.sc botany", "b.sc zoology",
]

# Stream keywords for filtered listing
STREAM_KEYWORDS = {
    "arts"     : ["arts", "b.a.", "ba ", "humanities"],
    "science"  : ["science", "b.sc", "bsc"],
    "commerce" : ["commerce", "b.com", "bcom"],
}

SINGLE_ANSWER_KEYWORDS = [
    "who is the principal", "who is principal", "principal of",
    "head of college", "head of institution", "director of",
    "who is the director", "who is the hod", "who is hod of",
]

SYSTEM_PROMPT = """You are an information assistant for Maharaja Agrasen College (MAC), Delhi.
MAC stands for Maharaja Agrasen College. It is affiliated to University of Delhi.
The official website of MAC is mac.du.ac.in.

STRICT RULES - follow every rule without exception:
1. Answer ONLY using the CONTEXT provided below the question.
2. If the answer is not clearly present in the context, say exactly: No data found
3. NEVER use your own knowledge, training data, or assumptions.
4. NEVER guess, infer, or fill in missing details.
5. NEVER say "I think", "probably", "usually", "generally", "typically".
6. NEVER invent years, dates, numbers, or rankings not in context.
7. For listing questions (departments, courses, programmes, faculty, staff):
   - List EVERY unique item you can find across ALL sources in context.
   - Scan every single source block before answering.
   - Do not stop early — check all sources.
   - Format as a clean numbered list.
   - Remove duplicates from your final list.
   - For faculty lists: include name and designation for each person.
8. For SINGLE PERSON questions (who is the principal):
   - Give ONLY ONE name and designation.
   - Do NOT list multiple people.
9. For factual questions (address, phone, email):
   - Give the exact value from context only.
   - If not found, say: No data found
10. For questions about data that might not exist:
    - Do NOT construct partial answers mixing real and invented data.
    - Either state exactly what context says OR say: No data found
11. For YES/NO existence questions (does MAC offer X, is X available at MAC):
    - ONLY confirm Yes if the context explicitly states that MAC offers X
      as a regular undergraduate programme (B.A., B.Sc., B.Com. level).
    - Mentions in IGNOU pages, department pages, or faculty profiles do NOT
      count as confirmation that MAC offers the course.
    - A department existing does NOT mean a standalone course exists.
    - If in any doubt, answer: No data found
    - Do NOT infer course existence from related subjects or departments.
"""


def is_listing_query(query: str) -> bool:
    q = query.lower()
    # Existence queries must NOT be treated as listing — prevents timeout
    if is_existence_query(query):
        return False
    return any(kw in q for kw in LISTING_KEYWORDS)


def is_existence_query(query: str) -> bool:
    q = query.lower()
    return any(kw in q for kw in EXISTENCE_KEYWORDS)


def is_not_offered_query(query: str) -> bool:
    """
    Returns True if the query asks about a course definitively NOT offered
    at MAC as a regular programme. Used to return immediate No data found
    without sending to LLM — prevents IGNOU/dept page false positives.
    """
    q = query.lower()
    return any(kw in q for kw in COURSES_NOT_OFFERED)


def get_stream(query: str):
    """Returns the stream name if the query is stream-specific, else None."""
    q = query.lower()
    for stream, keywords in STREAM_KEYWORDS.items():
        if any(kw in q for kw in keywords):
            return stream
    return None


def is_single_answer_query(query: str) -> bool:
    q = query.lower()
    return any(kw in q for kw in SINGLE_ANSWER_KEYWORDS)


def build_context(chunks: list, query: str) -> str:
    listing = is_listing_query(query)
    chars_per_chunk = LISTING_CHARS_PER_CHUNK if listing else FACTUAL_CHARS_PER_CHUNK

    parts = []
    total_chars = 0

    for i, chunk in enumerate(chunks):
        source = chunk.get("source", "Unknown")
        text = (chunk.get("text", "") or "").strip()
        if not text:
            continue

        truncated = text[:chars_per_chunk]
        if len(text) > chars_per_chunk:
            last_space = truncated.rfind(" ")
            if last_space > 80:
                truncated = truncated[:last_space]

        part = f"[Source {i+1}: {source}]\n{truncated}"
        if total_chars + len(part) > MAX_CONTEXT_CHARS:
            break

        parts.append(part)
        total_chars += len(part)

    return "\n\n---\n\n".join(parts)


def build_instruction(query: str) -> str:
    if is_single_answer_query(query):
        return "ANSWER (give ONLY ONE person's name and designation — do NOT list multiple people):"
    if is_existence_query(query):
        return (
            "ANSWER (YES/NO existence check — if the context mentions the course, "
            "department or programme asked about, reply: Yes, MAC offers [name]. "
            "If not mentioned at all, reply: No data found):"
        )
    if is_listing_query(query):
        stream = get_stream(query)
        if stream:
            return (
                f"ANSWER (list only {stream.upper()} stream items — "
                f"scan ALL sources, filter to {stream} only, numbered list, no duplicates):"
            )
        return "ANSWER (scan ALL sources above — list every unique item found as numbered list — do not stop early):"
    return "ANSWER (use only the context above — be concise and factual — if not found say: No data found):"


def _call_llm(query: str, context: str, max_tokens: int) -> tuple[str, Optional[str]]:
    instruction = build_instruction(query)
    user_message = f"""CONTEXT:
{context}

---

QUESTION: {query}

{instruction}"""

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "stream": False,
        "n": 1,
    }

    response = requests.post(API_URL, json=payload, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    data = response.json()

    choices = data.get("choices", [])
    if not choices:
        return "", None

    choice = choices[0]
    message = choice.get("message", {})
    answer = (message.get("content") or "").strip()
    finish_reason = choice.get("finish_reason")
    return answer, finish_reason


def _is_hallucination_like(answer: str) -> bool:
    hallucination_signals = [
        "i think", "i believe", "probably",
        "generally", "typically", "usually",
        "in most colleges", "as per my knowledge",
        "based on my training", "mahatma college",
        "i am not sure", "it is likely",
        "please visit the official",
        "was established in",
        "founded in 19", "founded in 20",
        "established in 19", "established in 20",
        "rs. unknown",
        "no specific ranking",
        "might not be comprehensive",
        "may not be up-to-date",
        "cannot be inferred",
    ]
    text = answer.lower()
    return any(signal in text for signal in hallucination_signals)


def generate_answer(query: str, chunks_or_context: Union[list, str]) -> str:
    if isinstance(chunks_or_context, list):
        context = build_context(chunks_or_context, query)
    else:
        context = str(chunks_or_context)[:MAX_CONTEXT_CHARS]

    if not context.strip():
        return "No data found"

    try:
        answer, finish_reason = _call_llm(query, context, MAX_TOKENS_OUT)

        if finish_reason == "length" or not answer:
            shorter_context = context[:7000]
            answer, finish_reason = _call_llm(query, shorter_context, MAX_TOKENS_OUT)

        if not answer:
            return "No data found"

        if _is_hallucination_like(answer):
            return "No data found"

        return answer

    except Exception:
        return "No data found"


if __name__ == "__main__":
    test_chunks = [
        {
            "source": "mac.du.ac.in/contactus.aspx",
            "text": "Maharaja Agrasen College, Vasundhara Enclave, Delhi - 110096. Phone: +91-11-22610563. Email: principal@mac.du.ac.in"
        },
        {
            "source": "mac.du.ac.in/about.aspx",
            "text": "MAC stands for Maharaja Agrasen College, affiliated to University of Delhi. Official website: mac.du.ac.in"
        }
    ]

    tests = [
        "What is the address of MAC?",
        "What is the phone number?",
        "When was MAC established?",
    ]

    for q in tests:
        print(f"\nQ: {q}")
        print(f"A: {generate_answer(q, test_chunks)}")
        print("-" * 40)