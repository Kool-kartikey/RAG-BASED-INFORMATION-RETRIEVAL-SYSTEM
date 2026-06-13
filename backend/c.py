import time
import sys
import os

# Ensure the root directory is in the path for internal imports
sys.path.append(os.getcwd())

try:
    from backend.retriever import answer
except ImportError as e:
    print(f"CRITICAL ERROR: {e}")
    print("Ensure you renamed line 6 in retriever.py to: from backend.llm import generate_answer")
    sys.exit(1)

# Categorized queries for better analysis
QUERY_DATA = {
    "FACTUAL": [
        "Who is the principal of MAC?",
        "What is the address of MAC?",
        "When was MAC established?",
        "What is the full form of MAC?",
        "Who is the HOD of Computer Science?"
    ],
    "LISTING": [
        "What departments are there?",
        "What courses does MAC offer?",
        "What programmes are available?",
        "List the faculty of CS department",
        "What committees does MAC have?"
    ],
    "OFF-TOPIC": [
        "What is the weather today?",
        "Who is the Prime Minister of India?",
        "Tell me a joke",
        "What is Bitcoin?",
        "How do I cook pasta?"
    ],
    "ENTITY": [
        "Who is Prof. Sanjeev Kumar Tiwari?",
        "Tell me about the Commerce department",
        "What is IQAC?",
        "Who is the placement coordinator?",
        "What is the NSS committee?"
    ]
}

results_summary = []

print(f"{'='*60}")
print(f"{'MAC RAG-CHATBOT PERFORMANCE TEST':^60}")
print(f"{'='*60}\n")

for category, queries in QUERY_DATA.items():
    print(f"--- CATEGORY: {category} ---")
    for q in queries:
        start_time = time.time()
        
        try:
            # The core function call
            response = answer(q)
            status = "SUCCESS"
        except Exception as e:
            response = f"ERROR: {str(e)}"
            status = "FAILED"
            
        elapsed = round(time.time() - start_time, 2)
        results_summary.append((category, q, elapsed, status))
        
        print(f"Q: {q}")
        print(f"A: {response}")
        print(f"Time: {elapsed}s | Status: {status}")
        print("-" * 20)
    print("\n")

# --- FINAL SUMMARY TABLE ---
print(f"{'='*60}")
print(f"{'FINAL SUMMARY REPORT':^60}")
print(f"{'='*60}")
print(f"{'Category':<12} | {'Time (s)':<8} | {'Query'}")
print("-" * 60)

total_time = 0
for cat, q, t, s in results_summary:
    total_time += t
    short_q = (q[:45] + '..') if len(q) > 45 else q
    print(f"{cat:<12} | {t:<8.2f} | {short_q}")

print("-" * 60)
print(f"Total Queries: {len(results_summary)}")
print(f"Total Execution Time: {round(total_time, 2)}s")
print(f"Average Time per Query: {round(total_time/len(results_summary), 2)}s")
print(f"{'='*60}")