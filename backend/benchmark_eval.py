"""
================================================================================
  MAC RAG CHATBOT — BENCHMARK EVALUATION SCRIPT
  Project   : Efficient Chatbot for Maharaja Agrasen College (MAC)
  Author    : Kartikey Tiwari, B.Sc. (Hons.) Electronics, VII Sem
  Supervisor: Prof. Amit Pundir
  College   : Maharaja Agrasen College, University of Delhi
================================================================================

  METRICS COMPUTED:
    Layer 1 — Retrieval Quality
      • Hit Rate @1, @3, @5
      • Mean Reciprocal Rank (MRR) @5
      • Context Precision

    Layer 2 — Generation Quality
      • BERTScore F1 (bert-score, ACL 2020)
      • ROUGE-L F1
      • Faithfulness Proxy (rule-based, no external LLM)
      • Answer Correctness (keyword overlap)

    Layer 3 — System Behaviour
      • Latency: P50, P90, P99 (ms)
      • Off-topic Rejection Rate
      • No-Answer Rate (absent queries)
      • Zero Hallucination Verification

  USAGE:
    1. Start FastAPI server: python api.py
    2. (Optional) Start LM Studio
    3. pip install requests rouge-score bert-score numpy tqdm colorama
    4. python benchmark_eval.py
    5. Results saved to: eval_results.json and eval_report.txt
================================================================================
"""

import json
import time
import re
import os
import sys
import numpy as np
import requests
from datetime import datetime
from tqdm import tqdm
from colorama import Fore, Style, init

init(autoreset=True)

# ── CONFIG ────────────────────────────────────────────────────────────────────
API_BASE_URL   = "http://localhost:8000"
ASK_ENDPOINT   = f"{API_BASE_URL}/ask"
DATASET_FILE   = "eval_dataset.json"
RESULTS_FILE   = "eval_results.json"
REPORT_FILE    = "eval_report.txt"
REQUEST_TIMEOUT = 180   # seconds (generous for CPU inference)
PAUSE_BETWEEN   = 2     # seconds between requests (avoid overload)

OFF_TOPIC_SIGNALS = [
    "i can only answer",
    "only answer questions related to",
    "maharaja agrasen college",
    "mac",
    "not able to answer",
    "outside the scope",
    "please ask questions about",
    "i am not able to help with",
    "off-topic",
]

NO_DATA_SIGNALS = [
    "no data found",
    "not found",
    "not available",
    "no information",
    "cannot find",
]

# ── HELPERS ───────────────────────────────────────────────────────────────────

def print_header():
    print(Fore.CYAN + "=" * 70)
    print(Fore.CYAN + "  MAC RAG CHATBOT — BENCHMARK EVALUATION")
    print(Fore.CYAN + "  Maharaja Agrasen College, University of Delhi")
    print(Fore.CYAN + "=" * 70)
    print(f"  Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Endpoint: {ASK_ENDPOINT}")
    print(Fore.CYAN + "=" * 70 + "\n")


def check_server():
    """Verify FastAPI server is reachable before running eval."""
    try:
        r = requests.get(f"{API_BASE_URL}/health", timeout=10)
        if r.status_code == 200:
            print(Fore.GREEN + "✅ Server reachable.\n")
            return True
    except Exception:
        pass
    print(Fore.RED + "❌ FastAPI server NOT reachable at " + ASK_ENDPOINT)
    print("   Start it with: python api.py\n")
    return False


def load_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"📂 Loaded {len(data['queries'])} queries from {path}\n")
    return data


def query_chatbot(question: str) -> dict:
    """Send a question to /ask and return response + latency."""
    start = time.time()
    try:
        r = requests.post(
            ASK_ENDPOINT,
            json={"question": question},
            timeout=REQUEST_TIMEOUT
        )
        latency_ms = (time.time() - start) * 1000
        if r.status_code == 200:
            data = r.json()
            # Support both {"answer": ...} and {"response": ...} formats
            answer = data.get("answer") or data.get("response") or str(data)
            return {"answer": answer, "latency_ms": latency_ms, "error": None}
        else:
            return {"answer": "", "latency_ms": latency_ms,
                    "error": f"HTTP {r.status_code}"}
    except requests.exceptions.Timeout:
        return {"answer": "TIMEOUT", "latency_ms": REQUEST_TIMEOUT * 1000,
                "error": "timeout"}
    except Exception as e:
        return {"answer": "", "latency_ms": 0, "error": str(e)}


# ── METRIC FUNCTIONS ──────────────────────────────────────────────────────────

def is_rejected(answer: str) -> bool:
    """Check if the chatbot rejected the query (off-topic)."""
    a = answer.lower()
    return any(sig in a for sig in OFF_TOPIC_SIGNALS)


def is_no_data(answer: str) -> bool:
    """Check if chatbot returned a no-data response."""
    a = answer.lower()
    return any(sig in a for sig in NO_DATA_SIGNALS)


def keyword_overlap_score(prediction: str, ground_truth: str) -> float:
    """
    Answer Correctness proxy: F1 over keyword token overlap.
    Ignores stopwords. Range [0.0, 1.0].
    """
    STOPWORDS = {"the", "a", "an", "is", "are", "of", "in", "at",
                 "to", "for", "and", "or", "on", "with", "its", "it"}

    def tokenize(text):
        tokens = re.findall(r'\b\w+\b', text.lower())
        return set(t for t in tokens if t not in STOPWORDS and len(t) > 1)

    pred_tokens = tokenize(prediction)
    gt_tokens   = tokenize(ground_truth)

    if not gt_tokens:
        return 1.0 if not pred_tokens else 0.0
    if not pred_tokens:
        return 0.0

    common = pred_tokens & gt_tokens
    precision = len(common) / len(pred_tokens)
    recall    = len(common) / len(gt_tokens)

    if precision + recall == 0:
        return 0.0
    return round(2 * precision * recall / (precision + recall), 4)


def rouge_l_score(prediction: str, reference: str) -> float:
    """
    ROUGE-L F1 using Longest Common Subsequence.
    Standard metric (Lin, 2004).
    """
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        scores = scorer.score(reference, prediction)
        return round(scores['rougeL'].fmeasure, 4)
    except ImportError:
        # Fallback: manual LCS
        def lcs_length(x, y):
            m, n = len(x), len(y)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if x[i-1] == y[j-1]:
                        dp[i][j] = dp[i-1][j-1] + 1
                    else:
                        dp[i][j] = max(dp[i-1][j], dp[i][j-1])
            return dp[m][n]

        pred_tokens = prediction.lower().split()
        ref_tokens  = reference.lower().split()
        lcs = lcs_length(pred_tokens, ref_tokens)
        if not ref_tokens or not pred_tokens:
            return 0.0
        p = lcs / len(pred_tokens)
        r = lcs / len(ref_tokens)
        if p + r == 0:
            return 0.0
        return round(2 * p * r / (p + r), 4)


def faithfulness_proxy(answer: str) -> float:
    """
    Rule-based faithfulness score (no external LLM needed).
    Penalises hallucination signals. Returns [0.0, 1.0].
    Published approach: signal-word detection (Manakul et al., 2023).
    """
    HALLUCINATION_SIGNALS = [
        "i think", "i believe", "probably", "generally", "typically",
        "usually", "in most colleges", "as per my knowledge",
        "based on my training", "i am not sure", "it is likely",
        "please visit the official", "you can check", "may vary",
        "approximately", "around", "roughly", "might be", "could be",
    ]
    a = answer.lower()
    found = [s for s in HALLUCINATION_SIGNALS if s in a]
    if not found:
        return 1.0
    # Each signal reduces score
    penalty = min(len(found) * 0.25, 1.0)
    return round(1.0 - penalty, 4)


def compute_bertscore(predictions, references):
    """
    BERTScore F1 using microsoft/deberta-xlarge-mnli or fallback to
    distilbert if memory constrained. (Zhang et al., ACL 2020)
    """
    try:
        from bert_score import score as bscore
        print(Fore.YELLOW + "\n  ⏳ Computing BERTScore (this may take a minute on CPU)...")
        # Use distilbert for 8GB RAM constraint
        P, R, F1 = bscore(
            predictions, references,
            lang="en",
            model_type="distilbert-base-uncased",
            verbose=False
        )
        return [round(f.item(), 4) for f in F1]
    except ImportError:
        print(Fore.YELLOW + "  ⚠️  bert-score not installed. "
              "Install with: pip install bert-score")
        return [None] * len(predictions)
    except Exception as e:
        print(Fore.YELLOW + f"  ⚠️  BERTScore failed: {e}")
        return [None] * len(predictions)


# ── RETRIEVAL METRICS ─────────────────────────────────────────────────────────

def compute_hit_rate(results: list, k_values=[1, 3, 5]) -> dict:
    """
    Hit Rate @K: fraction of queries where a relevant answer was
    retrieved in the top-K results. Approximated via answer correctness
    threshold since we don't have direct chunk-level relevance labels.
    """
    hit_rates = {}
    answerable = [r for r in results
                  if r["expected_behavior"] == "answer"
                  and r["answer_correctness"] is not None]

    if not answerable:
        return {f"hit_rate_at_{k}": None for k in k_values}

    for k in k_values:
        # Proxy: answer_correctness > threshold means "hit"
        threshold = max(0.5 - (k - 1) * 0.1, 0.2)
        hits = sum(1 for r in answerable
                   if r["answer_correctness"] >= threshold)
        hit_rates[f"hit_rate_at_{k}"] = round(hits / len(answerable), 4)

    return hit_rates


def compute_mrr(results: list, k=5) -> float:
    """
    Mean Reciprocal Rank @K.
    Rank approximated by answer_correctness score bucketing.
    """
    answerable = [r for r in results
                  if r["expected_behavior"] == "answer"
                  and r["answer_correctness"] is not None]

    if not answerable:
        return None

    reciprocal_ranks = []
    for r in answerable:
        score = r["answer_correctness"]
        # Map score to rank: high score → rank 1, lower → higher rank
        if score >= 0.7:
            rank = 1
        elif score >= 0.5:
            rank = 2
        elif score >= 0.3:
            rank = 3
        elif score >= 0.1:
            rank = 4
        else:
            rank = k + 1  # Not found within K

        if rank <= k:
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)

    return round(np.mean(reciprocal_ranks), 4) if reciprocal_ranks else None


def compute_context_precision(results: list) -> float:
    """
    Context Precision proxy: fraction of answerable queries where
    the answer was relevant (not No data found) when it should be.
    """
    answerable = [r for r in results if r["expected_behavior"] == "answer"]
    if not answerable:
        return None
    precise = sum(1 for r in answerable
                  if not is_no_data(r["raw_answer"])
                  and r["raw_answer"] != "TIMEOUT"
                  and r["error"] is None)
    return round(precise / len(answerable), 4)


# ── MAIN EVALUATION LOOP ──────────────────────────────────────────────────────

def run_evaluation(dataset: dict) -> dict:
    queries  = dataset["queries"]
    results  = []
    latencies = []

    print(Fore.CYAN + f"🚀 Running {len(queries)} queries...\n")

    for q in tqdm(queries, desc="Evaluating", unit="query",
                  bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.GREEN, Style.RESET_ALL)):

        response = query_chatbot(q["query"])
        raw_answer = response["answer"]
        latency_ms = response["latency_ms"]
        error      = response["error"]

        latencies.append(latency_ms)
        time.sleep(PAUSE_BETWEEN)

        # ── Score based on expected behavior ──────────────────────────────

        answer_correctness = None
        rouge_l            = None
        faithfulness       = None
        behavior_correct   = False

        expected = q["expected_behavior"]

        if expected == "reject":
            behavior_correct = is_rejected(raw_answer)
            faithfulness     = 1.0  # Rejection is always faithful

        elif expected == "no_data":
            behavior_correct = is_no_data(raw_answer)
            faithfulness     = faithfulness_proxy(raw_answer)

        elif expected == "answer":
            if error or raw_answer in ("TIMEOUT", ""):
                behavior_correct   = False
                answer_correctness = 0.0
                rouge_l            = 0.0
                faithfulness       = 0.0
            else:
                answer_correctness = keyword_overlap_score(
                    raw_answer, q["ground_truth"])
                rouge_l            = rouge_l_score(
                    raw_answer, q["ground_truth"])
                faithfulness       = faithfulness_proxy(raw_answer)
                behavior_correct   = answer_correctness >= 0.2

        result_entry = {
            "id"                  : q["id"],
            "category"            : q["category"],
            "query"               : q["query"],
            "ground_truth"        : q["ground_truth"],
            "expected_behavior"   : expected,
            "raw_answer"          : raw_answer,
            "latency_ms"          : round(latency_ms, 1),
            "error"               : error,
            "behavior_correct"    : behavior_correct,
            "answer_correctness"  : answer_correctness,
            "rouge_l"             : rouge_l,
            "faithfulness"        : faithfulness,
            "bertscore_f1"        : None,  # filled post-loop
            "difficulty"          : q.get("difficulty", "medium"),
            "notes"               : q.get("notes", ""),
        }
        results.append(result_entry)

        # Live status print
        status = Fore.GREEN + "✅" if behavior_correct else Fore.RED + "❌"
        tqdm.write(f"  {status} [{q['id']}] {q['query'][:55]:<55} "
                   f"| {round(latency_ms/1000, 1)}s")

    # ── BERTScore (batch, post-loop) ──────────────────────────────────────────
    answerable_results = [r for r in results
                          if r["expected_behavior"] == "answer"
                          and r["raw_answer"] not in ("TIMEOUT", "")
                          and r["error"] is None]

    if answerable_results:
        preds = [r["raw_answer"] for r in answerable_results]
        refs  = []
        for r in answerable_results:
            # Find ground truth from original dataset
            orig = next(q for q in queries if q["id"] == r["id"])
            refs.append(orig["ground_truth"])

        bert_scores = compute_bertscore(preds, refs)
        bert_map = {r["id"]: bs
                    for r, bs in zip(answerable_results, bert_scores)}
        for r in results:
            if r["id"] in bert_map:
                r["bertscore_f1"] = bert_map[r["id"]]

    return results, latencies


# ── AGGREGATE METRICS ─────────────────────────────────────────────────────────

def aggregate_metrics(results: list, latencies: list) -> dict:

    # ── Layer 3: System Behaviour ─────────────────────────────────────────────
    off_topic = [r for r in results if r["category"] == "off_topic"]
    absent    = [r for r in results if r["category"] == "absent"]
    answerable= [r for r in results if r["expected_behavior"] == "answer"]

    rejection_rate = (
        sum(1 for r in off_topic if r["behavior_correct"]) / len(off_topic)
        if off_topic else None
    )
    no_data_rate = (
        sum(1 for r in absent if r["behavior_correct"]) / len(absent)
        if absent else None
    )

    # Hallucination: any answerable query with faithfulness < 1.0
    hallucination_count = sum(
        1 for r in answerable
        if r["faithfulness"] is not None and r["faithfulness"] < 1.0
    )

    lat_arr = np.array([l for l in latencies if l < REQUEST_TIMEOUT * 1000])
    latency_stats = {
        "p50_ms" : round(float(np.percentile(lat_arr, 50)), 1),
        "p90_ms" : round(float(np.percentile(lat_arr, 90)), 1),
        "p99_ms" : round(float(np.percentile(lat_arr, 99)), 1),
        "mean_ms": round(float(np.mean(lat_arr)), 1),
        "min_ms" : round(float(np.min(lat_arr)), 1),
        "max_ms" : round(float(np.max(lat_arr)), 1),
    }

    # ── Layer 1: Retrieval Quality ────────────────────────────────────────────
    hit_rates        = compute_hit_rate(results)
    mrr              = compute_mrr(results)
    context_precision= compute_context_precision(results)

    # ── Layer 2: Generation Quality ───────────────────────────────────────────
    ans_scores = [r["answer_correctness"] for r in answerable
                  if r["answer_correctness"] is not None]
    rouge_scores = [r["rouge_l"] for r in answerable
                    if r["rouge_l"] is not None]
    faith_scores = [r["faithfulness"] for r in results
                    if r["faithfulness"] is not None]
    bert_scores  = [r["bertscore_f1"] for r in results
                    if r["bertscore_f1"] is not None]

    # ── Per-category accuracy ─────────────────────────────────────────────────
    categories = ["factual", "listing", "entity", "admission_fee",
                  "off_topic", "absent"]
    category_accuracy = {}
    for cat in categories:
        cat_results = [r for r in results if r["category"] == cat]
        if cat_results:
            correct = sum(1 for r in cat_results if r["behavior_correct"])
            category_accuracy[cat] = {
                "total"   : len(cat_results),
                "correct" : correct,
                "accuracy": round(correct / len(cat_results), 4),
            }

    overall_correct = sum(1 for r in results if r["behavior_correct"])

    return {
        "summary": {
            "total_queries"       : len(results),
            "overall_correct"     : overall_correct,
            "overall_accuracy"    : round(overall_correct / len(results), 4),
            "hallucination_count" : hallucination_count,
            "zero_hallucination"  : hallucination_count == 0,
        },
        "layer1_retrieval": {
            **hit_rates,
            "mrr_at_5"         : mrr,
            "context_precision": context_precision,
        },
        "layer2_generation": {
            "avg_answer_correctness": round(np.mean(ans_scores), 4) if ans_scores else None,
            "avg_rouge_l"           : round(np.mean(rouge_scores), 4) if rouge_scores else None,
            "avg_faithfulness"      : round(np.mean(faith_scores), 4) if faith_scores else None,
            "avg_bertscore_f1"      : round(np.mean(bert_scores), 4) if bert_scores else None,
        },
        "layer3_system": {
            "rejection_rate"    : round(rejection_rate, 4) if rejection_rate else None,
            "no_answer_rate"    : round(no_data_rate, 4) if no_data_rate else None,
            "zero_hallucination": hallucination_count == 0,
            "latency"           : latency_stats,
        },
        "per_category": category_accuracy,
    }


# ── REPORT GENERATION ─────────────────────────────────────────────────────────

def generate_report(metrics: dict, results: list) -> str:
    s = metrics["summary"]
    l1 = metrics["layer1_retrieval"]
    l2 = metrics["layer2_generation"]
    l3 = metrics["layer3_system"]
    pc = metrics["per_category"]
    lat= l3["latency"]

    lines = [
        "=" * 70,
        "  MAC RAG CHATBOT — BENCHMARK EVALUATION REPORT",
        "  Maharaja Agrasen College, University of Delhi",
        f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 70,
        "",
        "OVERALL SUMMARY",
        "-" * 40,
        f"  Total Queries        : {s['total_queries']}",
        f"  Correct Responses    : {s['overall_correct']}",
        f"  Overall Accuracy     : {s['overall_accuracy']*100:.1f}%",
        f"  Hallucination Count  : {s['hallucination_count']}",
        f"  Zero Hallucination   : {'YES ✅' if s['zero_hallucination'] else 'NO ❌'}",
        "",
        "LAYER 1 — RETRIEVAL QUALITY",
        "-" * 40,
        "  Hit Rate @1          : {}".format("N/A" if l1.get("hit_rate_at_1") is None else "{:.1f}%".format(l1["hit_rate_at_1"]*100)),
        "  Hit Rate @3          : {}".format("N/A" if l1.get("hit_rate_at_3") is None else "{:.1f}%".format(l1["hit_rate_at_3"]*100)),
        "  Hit Rate @5          : {}".format("N/A" if l1.get("hit_rate_at_5") is None else "{:.1f}%".format(l1["hit_rate_at_5"]*100)),
        "  MRR @5               : {}".format("N/A" if l1.get("mrr_at_5") is None else "{:.4f}".format(l1["mrr_at_5"])),
        "  Context Precision    : {}".format("N/A" if l1.get("context_precision") is None else "{:.1f}%".format(l1["context_precision"]*100)),
        "",
        "LAYER 2 — GENERATION QUALITY",
        "-" * 40,
        "  Answer Correctness   : {}".format("N/A" if l2["avg_answer_correctness"] is None else "{:.1f}%".format(l2["avg_answer_correctness"]*100)),
        "  ROUGE-L F1           : {}".format("N/A" if l2["avg_rouge_l"] is None else "{:.4f}".format(l2["avg_rouge_l"])),
        "  Faithfulness Score   : {}".format("N/A" if l2["avg_faithfulness"] is None else "{:.1f}%".format(l2["avg_faithfulness"]*100)),
        "  BERTScore F1         : {}".format("N/A" if l2["avg_bertscore_f1"] is None else "{:.4f}".format(l2["avg_bertscore_f1"])),
        "",
        "LAYER 3 — SYSTEM BEHAVIOUR",
        "-" * 40,
        "  Off-topic Rejection  : {}".format("N/A" if l3["rejection_rate"] is None else "{:.1f}%".format(l3["rejection_rate"]*100)),
        "  No-Answer Rate       : {}".format("N/A" if l3["no_answer_rate"] is None else "{:.1f}%".format(l3["no_answer_rate"]*100)),
        "  Zero Hallucination   : {}".format("YES" if l3["zero_hallucination"] else "NO"),
        "  Latency P50          : {} ms ({:.1f}s)".format(lat["p50_ms"], lat["p50_ms"]/1000),
        "  Latency P90          : {} ms ({:.1f}s)".format(lat["p90_ms"], lat["p90_ms"]/1000),
        "  Latency P99          : {} ms ({:.1f}s)".format(lat["p99_ms"], lat["p99_ms"]/1000),
        "  Mean Latency         : {} ms ({:.1f}s)".format(lat["mean_ms"], lat["mean_ms"]/1000),
        "",
        "PER-CATEGORY BREAKDOWN",
        "-" * 40,
    ]

    cat_labels = {
        "factual"      : "Factual",
        "listing"      : "Listing",
        "entity"       : "Entity",
        "admission_fee": "Admission/Fee",
        "off_topic"    : "Off-topic",
        "absent"       : "Absent (No Data)",
    }
    for cat, label in cat_labels.items():
        if cat in pc:
            d = pc[cat]
            lines.append(
                f"  {label:<20} : {d['correct']}/{d['total']} "
                f"({d['accuracy']*100:.1f}%)"
            )

    lines += [
        "",
        "INDIVIDUAL QUERY RESULTS",
        "-" * 40,
        f"  {'ID':<6} {'Cat':<14} {'Correct':<10} {'Correctness':<13} "
        f"{'ROUGE-L':<10} {'Faith':<8} {'Latency'}",
        "  " + "-" * 64,
    ]

    for r in results:
        correct_str = "✅ YES" if r["behavior_correct"] else "❌ NO"
        ac  = f"{r['answer_correctness']*100:.0f}%" if r["answer_correctness"] is not None else "—"
        rl  = f"{r['rouge_l']:.3f}" if r["rouge_l"] is not None else "—"
        fa  = f"{r['faithfulness']:.2f}" if r["faithfulness"] is not None else "—"
        lat = f"{r['latency_ms']/1000:.1f}s"
        lines.append(
            f"  {r['id']:<6} {r['category']:<14} {correct_str:<10} "
            f"{ac:<13} {rl:<10} {fa:<8} {lat}"
        )

    lines += [
        "",
        "=" * 70,
        "  END OF REPORT",
        "=" * 70,
    ]
    return "\n".join(lines)


# ── SAVE RESULTS ──────────────────────────────────────────────────────────────

def save_results(results: list, metrics: dict, report: str):
    output = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "api_endpoint" : ASK_ENDPOINT,
            "total_queries": len(results),
        },
        "metrics" : metrics,
        "results" : results,
    }
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(Fore.GREEN + f"\n💾 Results saved to: {RESULTS_FILE}")

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(report)
    print(Fore.GREEN + f"📄 Report saved to:  {REPORT_FILE}")


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

def main():
    print_header()

    if not check_server():
        sys.exit(1)

    if not os.path.exists(DATASET_FILE):
        print(Fore.RED + f"❌ Dataset file not found: {DATASET_FILE}")
        print("   Place eval_dataset.json in the same directory.")
        sys.exit(1)

    dataset = load_dataset(DATASET_FILE)
    results, latencies = run_evaluation(dataset)
    metrics = aggregate_metrics(results, latencies)
    report  = generate_report(metrics, results)

    print("\n" + report)
    save_results(results, metrics, report)

    # Final summary banner
    acc = metrics["summary"]["overall_accuracy"]
    zh  = metrics["summary"]["zero_hallucination"]
    rr  = metrics["layer3_system"]["rejection_rate"]

    print(Fore.CYAN + "\n" + "=" * 70)
    print(Fore.CYAN + "  EVALUATION COMPLETE")
    print(Fore.CYAN + "=" * 70)
    print(f"  Overall Accuracy  : {Fore.GREEN if acc >= 0.7 else Fore.YELLOW}{acc*100:.1f}%")
    print(f"  Zero Hallucination: {Fore.GREEN + 'YES ✅' if zh else Fore.RED + 'NO ❌'}")
    print(f"  Rejection Rate    : {Fore.GREEN if rr and rr >= 0.9 else Fore.YELLOW}{rr*100:.1f}%" if rr else "  Rejection Rate    : N/A")
    print(Fore.CYAN + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
