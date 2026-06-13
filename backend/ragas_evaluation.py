"""
RAGAS Evaluation Script — MAC College RAG Chatbot
===================================================
Version 2.0 — Verified Ground Truths
All ground truths verified directly from mac.du.ac.in

Test Set: 35 queries across 7 categories
- Factual          : 5 queries
- Listing          : 5 queries
- Off-Topic        : 5 queries
- Entity           : 5 queries
- Negative         : 5 queries
- Multi-hop        : 5 queries
- Novel Contribution: 5 queries

SETUP (run once in your venv):
    pip install ragas datasets langchain langchain-community --break-system-packages

USAGE:
    1. Start LM Studio → load Llama 3.2 3B → Start Server
    2. Start FastAPI: python api.py
    3. Run: python ragas_evaluation.py

OUTPUT (saved in data/ragas_results/):
    - ragas_results_[timestamp].json
    - ragas_summary_[timestamp].csv
    - ragas_report_[timestamp].txt
"""

import sys
import os
import json
import time
import requests
import csv
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────
API_BASE_URL = "http://localhost:8000"
RESULTS_DIR  = "data/ragas_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Exact off-topic rejection response from chatbot.py
OFF_TOPIC_RESPONSE = (
    "I can only answer questions related to "
    "Maharaja Agrasen College (MAC). Please ask me about "
    "admissions, departments, faculty, courses, fees, "
    "events, or any other college-related topic."
)

# ─────────────────────────────────────────
# TEST SET — 35 VERIFIED QUESTIONS
# All ground truths verified from mac.du.ac.in
# ─────────────────────────────────────────
TEST_SET = [

    # ══════════════════════════════════════
    # CATEGORY 1: FACTUAL (5 queries)
    # Tests precision of single-fact retrieval
    # ══════════════════════════════════════
    {
        "id"          : "F01",
        "category"    : "Factual",
        "question"    : "What is the full form of MAC?",
        "ground_truth": "MAC stands for Maharaja Agrasen College.",
    },
    {
        "id"          : "F02",
        "category"    : "Factual",
        "question"    : "Which university is Maharaja Agrasen College affiliated to?",
        "ground_truth": "Maharaja Agrasen College is affiliated to the University of Delhi.",
    },
    {
        "id"          : "F03",
        "category"    : "Factual",
        "question"    : "What is the official website of Maharaja Agrasen College?",
        "ground_truth": "The official website of Maharaja Agrasen College is mac.du.ac.in.",
    },
    {
        "id"          : "F04",
        "category"    : "Factual",
        "question"    : "What is the address of Maharaja Agrasen College?",
        "ground_truth": "Maharaja Agrasen College is located at Vasundhara Enclave, Delhi - 110096.",
    },
    {
        "id"          : "F05",
        "category"    : "Factual",
        "question"    : "What accreditation has Maharaja Agrasen College received?",
        "ground_truth": "Maharaja Agrasen College is accredited with NAAC A Grade.",
    },

    # ══════════════════════════════════════
    # CATEGORY 2: LISTING (5 queries)
    # Tests completeness of multi-item retrieval
    # ══════════════════════════════════════
    {
        "id"          : "L01",
        "category"    : "Listing",
        "question"    : "What arts courses are offered at Maharaja Agrasen College?",
        "ground_truth": (
            "Maharaja Agrasen College offers the following arts courses: "
            "B.A.(H) Business Economics (BBE), B.A.(H) English, "
            "B.A.(H) Journalism, B.A.(H) Hindi, and B.A.(H) Political Science."
        ),
    },
    {
        "id"          : "L02",
        "category"    : "Listing",
        "question"    : "What science courses are offered at Maharaja Agrasen College?",
        "ground_truth": (
            "Maharaja Agrasen College offers the following science courses: "
            "B.Sc.(Physical Science) with Chemistry and Physics, "
            "B.Sc.(H) Electronics, and B.Sc.(G) Mathematics."
        ),
    },
    {
        "id"          : "L03",
        "category"    : "Listing",
        "question"    : "What are all the courses offered at Maharaja Agrasen College?",
        "ground_truth": (
            "Maharaja Agrasen College offers the following courses: "
            "Arts: B.A.(H) Business Economics (BBE), B.A.(H) English, B.A.(H) Journalism, "
            "B.A.(H) Hindi, B.A.(H) Political Science. "
            "Commerce: B.Com.(H). "
            "Science: B.Sc.(Physical Science) with Chemistry and Physics, "
            "B.Sc.(H) Electronics, B.Sc.(G) Mathematics."
        ),
    },
    {
        "id"          : "L04",
        "category"    : "Listing",
        "question"    : "What departments are available at Maharaja Agrasen College?",
        "ground_truth": (
            "Maharaja Agrasen College has the following departments: "
            "Business Economics, English, Economics, Hindi, History, Journalism, "
            "Physical Education, Political Science, Commerce, Biology, Chemistry, "
            "Computer Science, Electronics, Mathematics, and Physics."
        ),
    },
    {
        "id"          : "L05",
        "category"    : "Listing",
        "question"    : "What are the contact details of Maharaja Agrasen College?",
        "ground_truth": (
            "Maharaja Agrasen College contact details: "
            "Address: Vasundhara Enclave, Delhi - 110096. "
            "Email: principal@mac.du.ac.in. "
            "Phone: +91-11-22610563."
        ),
    },

    # ══════════════════════════════════════
    # CATEGORY 3: OFF-TOPIC REJECTION (5 queries)
    # Tests domain guardrail accuracy
    # Expected: exact OFF_TOPIC_RESPONSE for all
    # ══════════════════════════════════════
    {
        "id"          : "O01",
        "category"    : "Off-Topic",
        "question"    : "Who is the Prime Minister of India?",
        "ground_truth": OFF_TOPIC_RESPONSE,
    },
    {
        "id"          : "O02",
        "category"    : "Off-Topic",
        "question"    : "What is the weather in Delhi today?",
        "ground_truth": OFF_TOPIC_RESPONSE,
    },
    {
        "id"          : "O03",
        "category"    : "Off-Topic",
        "question"    : "Who is Elon Musk?",
        "ground_truth": OFF_TOPIC_RESPONSE,
    },
    {
        "id"          : "O04",
        "category"    : "Off-Topic",
        "question"    : "What is the score of today's cricket match?",
        "ground_truth": OFF_TOPIC_RESPONSE,
    },
    {
        "id"          : "O05",
        "category"    : "Off-Topic",
        "question"    : "Can you write a Python program for me?",
        "ground_truth": OFF_TOPIC_RESPONSE,
    },

    # ══════════════════════════════════════
    # CATEGORY 4: ENTITY/CONTEXTUAL (5 queries)
    # Tests semantic understanding of institutional entities
    # ══════════════════════════════════════
    {
        "id"          : "E01",
        "category"    : "Entity",
        "question"    : "What is the vision of Maharaja Agrasen College?",
        "ground_truth": (
            "The vision of Maharaja Agrasen College is: Pursuit of knowledge, "
            "innovation and research through holistic and transformative education "
            "to nurture future leaders."
        ),
    },
    {
        "id"          : "E02",
        "category"    : "Entity",
        "question"    : "What is IQAC at Maharaja Agrasen College?",
        "ground_truth": (
            "IQAC stands for Internal Quality Assurance Cell. It is a body "
            "responsible for maintaining and enhancing the quality of academic "
            "and administrative functions at Maharaja Agrasen College."
        ),
    },
    {
        "id"          : "E03",
        "category"    : "Entity",
        "question"    : "What is the commerce course offered at MAC and what does it focus on?",
        "ground_truth": (
            "MAC offers B.Com.(H) under the Department of Commerce. "
            "The Commerce programme emphasizes an interactive teacher-learner "
            "environment and prepares students for careers in entrepreneurship, "
            "marketing, e-commerce, advertising, insurance, and civil services."
        ),
    },
    {
        "id"          : "E04",
        "category"    : "Entity",
        "question"    : "What is the phone number of Maharaja Agrasen College?",
        "ground_truth": "The phone number of Maharaja Agrasen College is +91-11-22610563.",
    },
    {
        "id"          : "E05",
        "category"    : "Entity",
        "question"    : "What is the email address of Maharaja Agrasen College?",
        "ground_truth": "The email address of Maharaja Agrasen College is principal@mac.du.ac.in.",
    },

    # ══════════════════════════════════════
    # CATEGORY 5: NEGATIVE/ABSENT DATA (5 queries)
    # Tests hallucination guard
    # Expected: No data found for all
    # ══════════════════════════════════════
    {
        "id"          : "N01",
        "category"    : "Negative",
        "question"    : "When was Maharaja Agrasen College established?",
        "ground_truth": "No data found",
    },
    {
        "id"          : "N02",
        "category"    : "Negative",
        "question"    : "What is the mobile number of the HOD of Electronics department?",
        "ground_truth": "No data found",
    },
    {
        "id"          : "N03",
        "category"    : "Negative",
        "question"    : "What is the NIRF ranking of MAC?",
        "ground_truth": "No data found",
    },
    {
        "id"          : "N04",
        "category"    : "Negative",
        "question"    : "What is the hostel fee at MAC?",
        "ground_truth": "No data found",
    },
    {
        "id"          : "N05",
        "category"    : "Negative",
        "question"    : "What is the salary of MAC professors?",
        "ground_truth": "No data found",
    },

    # ══════════════════════════════════════
    # CATEGORY 6: MULTI-HOP/COMPLEX (5 queries)
    # Tests reasoning across multiple retrieved chunks
    # ══════════════════════════════════════
    {
        "id"          : "M01",
        "category"    : "Multi-hop",
        "question"    : "What science courses does MAC offer and which subjects do they cover?",
        "ground_truth": (
            "MAC offers three science courses: B.Sc.(Physical Science) which covers "
            "Chemistry and Physics as combined subjects, B.Sc.(H) Electronics which "
            "is a standalone Honours programme in Electronics, and B.Sc.(G) Mathematics "
            "which is a General programme."
        ),
    },
    {
        "id"          : "M02",
        "category"    : "Multi-hop",
        "question"    : "How can a student contact Maharaja Agrasen College for admission related queries?",
        "ground_truth": (
            "A student can contact Maharaja Agrasen College at: "
            "Address: Vasundhara Enclave, Delhi - 110096. "
            "Email: principal@mac.du.ac.in. "
            "Phone: +91-11-22610563. "
            "The official website mac.du.ac.in also has admission related information."
        ),
    },
    {
        "id"          : "M03",
        "category"    : "Multi-hop",
        "question"    : "What humanities and social science programmes are offered at MAC?",
        "ground_truth": (
            "MAC offers the following humanities and social science programmes: "
            "B.A.(H) English, B.A.(H) Journalism, B.A.(H) Hindi, and "
            "B.A.(H) Political Science. The college also has departments of "
            "Economics, History, and Physical Education though these do not "
            "have standalone Honours courses."
        ),
    },
    {
        "id"          : "M04",
        "category"    : "Multi-hop",
        "question"    : "What is the relationship between the Economics department and courses at MAC?",
        "ground_truth": (
            "MAC has a Department of Economics but does not offer a standalone "
            "B.A.(H) Economics course. Economics is offered as part of the "
            "B.A.(H) Business Economics (BBE) programme under the "
            "Business Economics department."
        ),
    },
    {
        "id"          : "M05",
        "category"    : "Multi-hop",
        "question"    : "What programmes does the Department of Commerce offer at MAC and what are its activities?",
        "ground_truth": (
            "The Department of Commerce at MAC offers B.Com.(H). The department "
            "is active in organising talks, discussions, workshops, and their "
            "two-day annual festival CRUSADE. It has cells including Financial "
            "Literacy Cell, Life Skill Cell, Wellness Cell, Training and Placement "
            "Cell, Alumni Cell, and Community Service Cell."
        ),
    },

    # ══════════════════════════════════════
    # CATEGORY 7: NOVEL CONTRIBUTIONS (5 queries)
    # Tests system's specific technical contributions
    # Directly ties evaluation to paper's novel claims
    # ══════════════════════════════════════
    {
        "id"          : "NC01",
        "category"    : "Novel-Contribution",
        "question"    : "What departments does MAC have for science?",
        "ground_truth": (
            "MAC has the following science departments: Biology, Chemistry, "
            "Computer Science, Electronics, Mathematics, and Physics."
        ),
    },
    {
        "id"          : "NC02",
        "category"    : "Novel-Contribution",
        "question"    : "List all departments available at Maharaja Agrasen College",
        "ground_truth": (
            "Maharaja Agrasen College has the following departments: "
            "Business Economics, English, Economics, Hindi, History, Journalism, "
            "Physical Education, Political Science, Commerce, Biology, Chemistry, "
            "Computer Science, Electronics, Mathematics, and Physics."
        ),
    },
    {
        "id"          : "NC03",
        "category"    : "Novel-Contribution",
        "question"    : "What is the B.Sc. Electronics programme at MAC?",
        "ground_truth": (
            "MAC offers B.Sc.(H) Electronics under the Department of Electronics. "
            "It is a Honours programme affiliated to the University of Delhi."
        ),
    },
    {
        "id"          : "NC04",
        "category"    : "Novel-Contribution",
        "question"    : "Does MAC offer a course in journalism?",
        "ground_truth": (
            "Yes, MAC offers B.A.(H) Journalism under the Department of Journalism."
        ),
    },
    {
        "id"          : "NC05",
        "category"    : "Novel-Contribution",
        "question"    : "What is the B.Com programme at Maharaja Agrasen College?",
        "ground_truth": (
            "MAC offers B.Com.(H) under the Department of Commerce. "
            "It is affiliated to the University of Delhi and prepares students "
            "for careers in entrepreneurship, marketing, e-commerce, advertising, "
            "insurance, and civil services."
        ),
    },
]


# ─────────────────────────────────────────
# STEP 1: GET ANSWER FROM API
# ─────────────────────────────────────────
def get_answer_from_api(question: str) -> str:
    """Calls the FastAPI /ask endpoint and returns the answer."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/ask",
            json    = {"question": question},
            timeout = 120
        )
        if response.status_code == 200:
            return response.json().get("answer", "No data found")
        else:
            print(f"  [API ERROR] Status {response.status_code}")
            return "API Error"
    except requests.exceptions.Timeout:
        print(f"  [TIMEOUT] Question timed out after 120s")
        return "Timeout"
    except Exception as e:
        print(f"  [ERROR] {e}")
        return "Connection Error"


# ─────────────────────────────────────────
# STEP 2: GET CONTEXTS FROM RETRIEVER
# ─────────────────────────────────────────
def get_contexts(question: str) -> list:
    """
    Gets retrieved chunks directly from FAISS retriever.
    Returns list of context strings for RAGAS evaluation.
    Off-topic queries return empty context since guardrail
    blocks them before retrieval.
    """
    try:
        from backend.retriever import search
        results = search(question)
        if not results:
            return ["No relevant context found."]
        return [r["text"] for r in results]
    except Exception as e:
        print(f"  [RETRIEVER ERROR] {e}")
        return ["Context retrieval failed."]


# ─────────────────────────────────────────
# STEP 3: RUN ALL QUESTIONS
# ─────────────────────────────────────────
def run_test_set() -> list:
    """Runs all 35 questions and collects answers + contexts."""
    print("\n" + "="*60)
    print("  MAC RAG Chatbot — RAGAS Evaluation v2.0")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Total queries: {len(TEST_SET)}")
    print("="*60 + "\n")

    results      = []
    category_counts = {}

    for i, item in enumerate(TEST_SET, 1):
        cat = item["category"]
        category_counts[cat] = category_counts.get(cat, 0) + 1

        print(f"[{i:02d}/{len(TEST_SET)}] [{cat}] {item['question'][:55]}...")

        start_time = time.time()
        answer     = get_answer_from_api(item["question"])
        contexts   = get_contexts(item["question"])
        elapsed    = round(time.time() - start_time, 2)

        print(f"         Answer   : {answer[:75]}...")
        print(f"         Contexts : {len(contexts)} chunks | Time: {elapsed}s\n")

        results.append({
            "id"           : item["id"],
            "category"     : cat,
            "question"     : item["question"],
            "ground_truth" : item["ground_truth"],
            "answer"       : answer,
            "contexts"     : contexts,
            "response_time": elapsed,
        })

        # Small delay to avoid overwhelming LM Studio on CPU
        time.sleep(2)

    print(f"\n[DONE] All {len(TEST_SET)} questions completed.\n")
    return results


# ─────────────────────────────────────────
# STEP 4: RUN RAGAS
# ─────────────────────────────────────────
def run_ragas(results: list) -> dict:
    """Formats results for RAGAS and computes scores."""
    print("[RAGAS] Preparing dataset...")

    try:
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )
        from datasets import Dataset
        from langchain_community.chat_models import ChatOpenAI
        from langchain_community.embeddings import HuggingFaceEmbeddings

        ragas_data = {
            "question"    : [r["question"]     for r in results],
            "answer"      : [r["answer"]        for r in results],
            "contexts"    : [r["contexts"]      for r in results],
            "ground_truth": [r["ground_truth"]  for r in results],
        }

        dataset = Dataset.from_dict(ragas_data)

        print("[RAGAS] Connecting to local LM Studio as judge LLM...")

        # Use local Llama 3.2 3B via LM Studio as judge
        judge_llm = ChatOpenAI(
            model_name      = "llama-3.2-3b-instruct",
            openai_api_base = "http://localhost:1234/v1",
            openai_api_key  = "not-needed",
            temperature     = 0.0,
        )

        # Use same MiniLM embeddings as your system
        embeddings = HuggingFaceEmbeddings(
            model_name = "all-MiniLM-L6-v2"
        )

        print("[RAGAS] Running evaluation — this will take several minutes on CPU...")
        print("[RAGAS] Metrics: Faithfulness, Answer Relevancy, Context Precision, Context Recall\n")

        ragas_scores = evaluate(
            dataset,
            metrics    = [
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            ],
            llm        = judge_llm,
            embeddings = embeddings,
        )

        print("\n[RAGAS] Evaluation complete.")
        return dict(ragas_scores)

    except ImportError as e:
        print(f"[RAGAS ERROR] Missing package: {e}")
        print("Run: pip install ragas datasets langchain langchain-community --break-system-packages")
        return {}
    except Exception as e:
        print(f"[RAGAS ERROR] {e}")
        return {}


# ─────────────────────────────────────────
# STEP 5: SAVE RESULTS
# ─────────────────────────────────────────
def save_results(results: list, ragas_scores: dict):
    """Saves all results in multiple formats for academic use."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── 1. Raw JSON ──
    json_path = f"{RESULTS_DIR}/ragas_results_{timestamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp"    : timestamp,
            "system"       : {
                "embedding_model" : "all-MiniLM-L6-v2",
                "dimensions"      : 384,
                "vector_index"    : "FAISS IndexFlatL2",
                "llm"             : "Llama 3.2 3B Instruct",
                "data_source"     : "mac.du.ac.in",
                "total_chunks"    : 2244,
                "pages_scraped"   : 200,
                "hardware"        : "CPU-only, 8GB RAM",
            },
            "total_queries": len(results),
            "ragas_scores" : ragas_scores,
            "results"      : results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n[SAVED] Raw results     : {json_path}")

    # ── 2. Summary CSV ──
    csv_path = f"{RESULTS_DIR}/ragas_summary_{timestamp}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "ID", "Category", "Question",
            "Answer (truncated)", "Response Time (s)", "Contexts Retrieved"
        ])
        for r in results:
            writer.writerow([
                r["id"],
                r["category"],
                r["question"],
                r["answer"][:100] + "..." if len(r["answer"]) > 100 else r["answer"],
                r["response_time"],
                len(r["contexts"]),
            ])
    print(f"[SAVED] Summary CSV     : {csv_path}")

    # ── 3. Academic Report ──
    report_path = f"{RESULTS_DIR}/ragas_report_{timestamp}.txt"
    with open(report_path, "w", encoding="utf-8") as f:

        f.write("="*70 + "\n")
        f.write("RAGAS EVALUATION REPORT — MAC COLLEGE RAG CHATBOT\n")
        f.write(f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*70 + "\n\n")

        f.write("SYSTEM CONFIGURATION\n")
        f.write("-"*40 + "\n")
        f.write("Embedding Model  : all-MiniLM-L6-v2 (384 dimensions)\n")
        f.write("Vector Index     : FAISS IndexFlatL2\n")
        f.write("LLM              : Llama 3.2 3B Instruct (LM Studio)\n")
        f.write("Top-K Retrieval  : 40 candidates\n")
        f.write("Hardware         : CPU-only, 8GB RAM\n")
        f.write("Data Source      : mac.du.ac.in\n")
        f.write("Pages Scraped    : 200\n")
        f.write("Total Chunks     : 2244\n\n")

        f.write("TEST SET DESIGN\n")
        f.write("-"*40 + "\n")
        f.write("Total Queries         : 35\n")
        f.write("Categories            : 7\n")
        f.write("  Factual             : 5 queries\n")
        f.write("  Listing             : 5 queries\n")
        f.write("  Off-Topic           : 5 queries\n")
        f.write("  Entity              : 5 queries\n")
        f.write("  Negative            : 5 queries\n")
        f.write("  Multi-hop           : 5 queries\n")
        f.write("  Novel-Contribution  : 5 queries\n\n")

        f.write("Ground Truth Source   : Verified from mac.du.ac.in\n")
        f.write("Evaluation Framework  : RAGAS (Es et al., 2023)\n")
        f.write("Judge LLM             : Llama 3.2 3B (local, zero cost)\n\n")

        # RAGAS scores
        if ragas_scores:
            f.write("RAGAS SCORES\n")
            f.write("-"*40 + "\n")
            for metric, score in ragas_scores.items():
                bar = "█" * int(score * 20)
                f.write(f"  {metric:<25}: {score:.4f}  {bar}\n")
            f.write("\n")

        # Response time by category
        f.write("RESPONSE TIME BY CATEGORY\n")
        f.write("-"*40 + "\n")
        categories = {}
        for r in results:
            cat = r["category"]
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(r["response_time"])

        for cat, times in sorted(categories.items()):
            avg  = sum(times) / len(times)
            mins = min(times)
            maxs = max(times)
            f.write(f"  {cat:<22}: avg={avg:.1f}s  min={mins:.1f}s  max={maxs:.1f}s\n")

        overall = sum(r["response_time"] for r in results) / len(results)
        f.write(f"  {'Overall':<22}: avg={overall:.1f}s\n\n")

        # Off-topic rejection
        f.write("OFF-TOPIC REJECTION ANALYSIS\n")
        f.write("-"*40 + "\n")
        off_topic = [r for r in results if r["category"] == "Off-Topic"]
        rejected  = [
            r for r in off_topic
            if "only answer questions" in r["answer"].lower()
        ]
        rate = len(rejected) / len(off_topic) * 100
        f.write(f"  Queries submitted  : {len(off_topic)}\n")
        f.write(f"  Correctly rejected : {len(rejected)}\n")
        f.write(f"  Rejection rate     : {rate:.1f}%\n\n")

        # Hallucination guard
        f.write("HALLUCINATION GUARD ANALYSIS\n")
        f.write("-"*40 + "\n")
        negative    = [r for r in results if r["category"] == "Negative"]
        no_data     = [
            r for r in negative
            if "no data found" in r["answer"].lower()
        ]
        hallucinated = len(negative) - len(no_data)
        f.write(f"  Absent-data queries    : {len(negative)}\n")
        f.write(f"  Correctly no data      : {len(no_data)}\n")
        f.write(f"  Hallucinated responses : {hallucinated}\n")
        f.write(f"  Hallucination rate     : {hallucinated/len(negative)*100:.1f}%\n\n")

        # Full Q&A log
        f.write("FULL QUESTION-ANSWER LOG\n")
        f.write("="*70 + "\n")
        for r in results:
            f.write(f"\n[{r['id']}] [{r['category']}]\n")
            f.write(f"Q        : {r['question']}\n")
            f.write(f"A        : {r['answer']}\n")
            f.write(f"Expected : {r['ground_truth']}\n")
            f.write(f"Time     : {r['response_time']}s | Contexts: {len(r['contexts'])}\n")
            f.write("-"*50 + "\n")

    print(f"[SAVED] Academic report : {report_path}")


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
def main():
    print("\n" + "="*60)
    print("  MAC RAG Chatbot — RAGAS Evaluation Pipeline v2.0")
    print("  35 Verified Queries | 7 Categories")
    print("="*60)
    print("\n  Pre-flight checklist:")
    print("  1. LM Studio running with Llama 3.2 3B")
    print("  2. FastAPI running (python api.py)")
    print("  3. data/faiss.index exists")
    print("  4. data/meta.json has 2244 chunks\n")

    # Check API health
    try:
        health = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if health.status_code == 200:
            print("[OK] FastAPI is reachable.\n")
        else:
            print("[WARNING] FastAPI returned unexpected status.\n")
    except Exception:
        print("[ERROR] FastAPI not reachable at http://localhost:8000")
        print("        Start with: python api.py")
        sys.exit(1)

    # Run all questions
    results = run_test_set()

    # Backup raw results immediately
    backup_path = f"{RESULTS_DIR}/raw_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(backup_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[BACKUP] Raw results saved to {backup_path}\n")

    # Run RAGAS
    ragas_scores = run_ragas(results)

    # Save everything
    save_results(results, ragas_scores)

    print("\n" + "="*60)
    print("  EVALUATION COMPLETE")
    print(f"  All files saved in: {RESULTS_DIR}/")
    print("="*60 + "\n")

    # Print quick summary to terminal
    if ragas_scores:
        print("RAGAS SCORES SUMMARY:")
        print("-"*40)
        for metric, score in ragas_scores.items():
            print(f"  {metric:<25}: {score:.4f}")
        print()


if __name__ == "__main__":
    main()