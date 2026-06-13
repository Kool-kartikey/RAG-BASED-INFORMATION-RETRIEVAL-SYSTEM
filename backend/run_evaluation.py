import requests
import json
import time

API_URL = "http://localhost:8000/ask"

QUERIES = [
    # ───────────── Factual (5) ─────────────
    ("F01", "Factual", "What is the address of Maharaja Agrasen College?"),
    ("F02", "Factual", "Who is the principal of Maharaja Agrasen College?"),
    ("F03", "Factual", "What is the official website of MAC?"),
    ("F04", "Factual", "What is the university affiliation of MAC?"),
    ("F05", "Factual", "What is the contact email of MAC?"),

    # ───────────── Listing (5) ─────────────
    ("L01", "Listing", "List all departments in Maharaja Agrasen College"),
    ("L02", "Listing", "List all undergraduate courses offered by MAC"),
    ("L03", "Listing", "List all societies and clubs in MAC"),
    ("L04", "Listing", "List facilities available on campus"),
    ("L05", "Listing", "List commerce-related courses at MAC"),

    # ───────────── Negative (5) ─────────────
    ("N01", "Negative", "What is the hostel fee of MAC?"),
    ("N02", "Negative", "What is the average placement salary at MAC?"),
    ("N03", "Negative", "Does MAC provide international exchange programs?"),
    ("N04", "Negative", "What is the ranking of MAC globally?"),
    ("N05", "Negative", "What is the CEO of Maharaja Agrasen College?"),

    # ───────────── Off-topic (5) ─────────────
    ("O01", "Off-Topic", "Who is the Prime Minister of India?"),
    ("O02", "Off-Topic", "What is machine learning?"),
    ("O03", "Off-Topic", "Explain black holes"),
    ("O04", "Off-Topic", "Who won the FIFA World Cup 2022?"),
    ("O05", "Off-Topic", "What is the capital of France?"),

    # ───────────── Multi-hop (5) ─────────────
    ("M01", "Multi-hop", "Which departments offer science courses at MAC?"),
    ("M02", "Multi-hop", "What facilities support science students at MAC?"),
    ("M03", "Multi-hop", "Which courses are related to commerce and management?"),
    ("M04", "Multi-hop", "What student activities are available apart from academics?"),
    ("M05", "Multi-hop", "Which departments are associated with humanities courses?"),

    # ───────────── Entity (5) ─────────────
    ("E01", "Entity", "What is NAAC accreditation?"),
    ("E02", "Entity", "What is University of Delhi?"),
    ("E03", "Entity", "What does MAC stand for?"),
    ("E04", "Entity", "What is B.Com course?"),
    ("E05", "Entity", "What is an undergraduate programme?")
]

OUTPUT_FILE = "evaluation_dataset.json"


def ask_api(question):
    try:
        response = requests.post(API_URL, json={"question": question}, timeout=60)
        data = response.json()
        return data.get("answer", ""), data.get("sources", [])
    except Exception as e:
        return f"ERROR: {str(e)}", []


def main():
    dataset = []

    print("Running evaluation queries...\n")

    for idx, (qid, category, query) in enumerate(QUERIES):
        print(f"[{idx+1}/{len(QUERIES)}] {qid} → {query}")

        answer, sources = ask_api(query)

        entry = {
            "ID": qid,
            "CATEGORY": category,
            "QUERY": query,
            "ANSWER": answer,
            "SOURCES": sources,
            "GROUND_TRUTH": ""
        }

        dataset.append(entry)

        time.sleep(1)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()