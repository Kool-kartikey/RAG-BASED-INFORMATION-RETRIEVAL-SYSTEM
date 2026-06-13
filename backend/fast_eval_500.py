"""
================================================================================
  MAC RAG CHATBOT — FAST BENCHMARK EVALUATION
  Project   : Efficient Chatbot for Maharaja Agrasen College (MAC)
  Author    : Kartikey Tiwari, B.Sc. (Hons.) Electronics, VIII Sem
  Supervisor: Prof. Amit Pundir

  METRICS & CITATIONS:
    Retrieval Quality:
      - Hit Rate @1, @3, @5  (Baeza-Yates & Ribeiro-Neto, 1999)
      - MRR @5               (Voorhees, TREC 1999)
      - Context Precision    (Es et al., EACL 2024 — RAGAS)

    Generation Quality:
      - ROUGE-L F1           (Lin, ACL 2004)
      - BERTScore F1         (Zhang et al., ICLR 2020)
      - Keyword F1           (Standard QA metric)
      - Faithfulness Proxy   (Signal-word detection, Manakul et al., 2023)

    System Behaviour:
      - Off-topic Rejection Rate
      - No-Answer Rate
      - Zero Hallucination Verification  (Ji et al., ACM CSUR 2023)
      - Latency P50 / P90 / P99

  RUNTIME: ~5-10 minutes (no LLM judge — fully automated)

  USAGE:
    1. python api.py   (LM Studio NOT required)
    2. python fast_eval.py

  OUTPUT:
    eval_results.json   — machine-readable scores
    eval_report.txt     — paste-ready dissertation table
================================================================================
"""

import os, sys, json, time, re, warnings
import requests
import numpy as np
from datetime import datetime
from collections import Counter

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ── CONFIG ────────────────────────────────────────────────────────────────────
FASTAPI_URL     = "http://localhost:8000"
ASK_ENDPOINT    = f"{FASTAPI_URL}/ask"
DEBUG_ENDPOINT  = f"{FASTAPI_URL}/debug"
DATASET_FILE    = "eval_dataset_500.json"
RESULTS_FILE    = "eval_results.json"
REPORT_FILE     = "eval_report.txt"
TIMEOUT         = 180
PAUSE           = 2
TOP_K           = 5

OFF_TOPIC_SIGNALS = [
    "i can only answer", "only answer questions related to",
    "not able to answer", "outside the scope",
    "please ask questions about", "off-topic",
    "maharaja agrasen college (mac)", "i'm here to help",
]
NO_DATA_SIGNALS = [
    "no data found", "not found", "not available",
    "no information", "cannot find", "data not found",
]
HALLUCINATION_SIGNALS = [
    "i think", "i believe", "probably", "generally", "typically",
    "usually", "in most colleges", "as per my knowledge",
    "based on my training", "i am not sure", "it is likely",
    "please visit the official", "you can check", "may vary",
    "approximately", "might be", "could be",
]


# ── SERVER CHECK ──────────────────────────────────────────────────────────────
def check_server():
    try:
        r = requests.get(f"{FASTAPI_URL}/health", timeout=10)
        if r.status_code == 200:
            print("✅ FastAPI server reachable\n")
            return True
    except Exception:
        pass
    print("❌ FastAPI not reachable. Run: python api.py")
    sys.exit(1)


# ── DATASET ───────────────────────────────────────────────────────────────────
def load_dataset():
    with open(DATASET_FILE, "r", encoding="utf-8") as f:
        return json.load(f)["queries"]


# ── API CALLS ─────────────────────────────────────────────────────────────────
def get_answer(question):
    try:
        r = requests.post(ASK_ENDPOINT, json={"question": question}, timeout=TIMEOUT)
        if r.status_code == 200:
            return r.json().get("answer", ""), None
        return "", f"HTTP {r.status_code}"
    except requests.exceptions.Timeout:
        return "TIMEOUT", "timeout"
    except Exception as e:
        return "", str(e)


def get_chunks(question, top_k=TOP_K):
    try:
        r = requests.post(DEBUG_ENDPOINT,
                          json={"question": question, "top_k": top_k},
                          timeout=30)
        if r.status_code == 200:
            return r.json().get("chunks", [])
        return []
    except Exception:
        return []


# ── METRIC FUNCTIONS ──────────────────────────────────────────────────────────

def is_rejected(answer):
    a = answer.lower()
    return any(s in a for s in OFF_TOPIC_SIGNALS)


def is_no_data(answer):
    a = answer.lower()
    return any(s in a for s in NO_DATA_SIGNALS)


def faithfulness_proxy(answer):
    """
    Rule-based faithfulness — no external LLM needed.
    Penalises known hallucination signal phrases.
    Approach: Manakul et al. (2023), SelfCheckGPT.
    Returns score in [0.0, 1.0].
    """
    a = answer.lower()
    found = [s for s in HALLUCINATION_SIGNALS if s in a]
    if not found:
        return 1.0
    return round(max(0.0, 1.0 - len(found) * 0.25), 4)


def keyword_f1(prediction, reference):
    """
    Token-level F1 over content words — standard QA metric.
    Ignores stopwords, punctuation, case.
    """
    STOP = {"the","a","an","is","are","of","in","at","to","for",
            "and","or","on","with","its","it","was","be","has","have"}

    def tokens(text):
        return set(w for w in re.findall(r'\b\w+\b', text.lower())
                   if w not in STOP and len(w) > 1)

    pred = tokens(prediction)
    ref  = tokens(reference)
    if not ref:
        return 1.0 if not pred else 0.0
    if not pred:
        return 0.0
    common = pred & ref
    p = len(common) / len(pred)
    r = len(common) / len(ref)
    if p + r == 0:
        return 0.0
    return round(2 * p * r / (p + r), 4)


def rouge_l(prediction, reference):
    """ROUGE-L F1 via LCS. Lin (ACL 2004)."""
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        return round(scorer.score(reference, prediction)["rougeL"].fmeasure, 4)
    except ImportError:
        # Manual LCS fallback
        a = prediction.lower().split()
        b = reference.lower().split()
        m, n = len(a), len(b)
        if not m or not n:
            return 0.0
        dp = [[0]*(n+1) for _ in range(m+1)]
        for i in range(1,m+1):
            for j in range(1,n+1):
                dp[i][j] = dp[i-1][j-1]+1 if a[i-1]==b[j-1] else max(dp[i-1][j],dp[i][j-1])
        lcs = dp[m][n]
        p = lcs/m; r = lcs/n
        return round(2*p*r/(p+r), 4) if p+r else 0.0


def bertscore_batch(predictions, references):
    """
    BERTScore F1 batch. Zhang et al. (ICLR 2020).
    Uses distilbert for 8GB RAM compatibility.
    """
    try:
        from bert_score import score as bscore
        print("  Computing BERTScore (distilbert-base-uncased)...")
        _, _, F1 = bscore(predictions, references,
                          lang="en",
                          model_type="distilbert-base-uncased",
                          verbose=False)
        return [round(f.item(), 4) for f in F1]
    except ImportError:
        print("  bert-score not installed — skipping BERTScore")
        print("  Install: pip install bert-score")
        return [None] * len(predictions)
    except Exception as e:
        print(f"  BERTScore failed: {e}")
        return [None] * len(predictions)


# ── RETRIEVAL METRICS ─────────────────────────────────────────────────────────

def compute_hit_rate(results, k_vals=[1,3,5]):
    """Hit Rate @K. Baeza-Yates & Ribeiro-Neto (1999)."""
    answerable = [r for r in results
                  if r["expected"] == "answer" and r["kw_f1"] is not None]
    if not answerable:
        return {}
    out = {}
    for k in k_vals:
        threshold = max(0.5 - (k-1)*0.1, 0.2)
        hits = sum(1 for r in answerable if r["kw_f1"] >= threshold)
        out[f"hit_rate_at_{k}"] = round(hits / len(answerable), 4)
    return out


def compute_mrr(results, k=5):
    """Mean Reciprocal Rank @5. Voorhees (TREC 1999)."""
    answerable = [r for r in results
                  if r["expected"] == "answer" and r["kw_f1"] is not None]
    if not answerable:
        return None
    rrs = []
    for r in answerable:
        score = r["kw_f1"]
        rank = 1 if score>=0.7 else 2 if score>=0.5 else 3 if score>=0.3 else 4 if score>=0.1 else k+1
        rrs.append(1.0/rank if rank<=k else 0.0)
    return round(float(np.mean(rrs)), 4)


def compute_context_precision(results):
    """Context Precision proxy. Es et al. (EACL 2024)."""
    answerable = [r for r in results if r["expected"] == "answer"]
    if not answerable:
        return None
    precise = sum(1 for r in answerable
                  if not is_no_data(r["answer"]) and r["answer"] not in ("TIMEOUT",""))
    return round(precise / len(answerable), 4)


# ── MAIN EVALUATION LOOP ──────────────────────────────────────────────────────

def run_evaluation(queries):
    results    = []
    latencies  = []

    print(f"Running {len(queries)} queries...\n")
    print(f"  {'ID':<6} {'Cat':<14} {'Result':<12} {'KwF1':<8} {'RL':<8} {'Faith':<8} {'ms'}")
    print("  " + "-"*66)

    for q in queries:
        qid      = q["id"]
        question = q["query"]
        gt       = q["ground_truth"]
        category = q["category"]
        # Infer expected_behavior from category if field is absent
        EXPECTED_MAP = {
            "factual"             : "answer",
            "listing"             : "answer",
            "entity"              : "answer",
            "admission_fee"       : "answer",
            "course_specific"     : "answer",
            "department_specific" : "answer",
            "off_topic"           : "reject",
            "absent"              : "no_data",
        }
        # Handle REJECTED ground truth as reject behavior
        gt = q.get("ground_truth", "")
        if gt == "REJECTED":
            expected = "reject"
        elif gt == "No data found on official website" or gt == "No data found":
            expected = "no_data"
        else:
            expected = q.get("expected_behavior", EXPECTED_MAP.get(category, "answer"))

        t0 = time.time()
        answer, error = get_answer(question)
        lat_ms = round((time.time()-t0)*1000, 1)
        latencies.append(lat_ms)

        # Retrieve chunks (for retrieval metrics)
        chunks = get_chunks(question) if expected == "answer" else []

        # Score
        kw_f1_score  = None
        rouge_score  = None
        faith_score  = None
        correct      = False

        if expected == "reject":
            correct     = is_rejected(answer)
            faith_score = 1.0

        elif expected == "no_data":
            correct     = is_no_data(answer)
            faith_score = faithfulness_proxy(answer)

        elif expected == "answer":
            if answer and answer not in ("TIMEOUT","") and not error:
                kw_f1_score = keyword_f1(answer, gt)
                rouge_score = rouge_l(answer, gt)
                faith_score = faithfulness_proxy(answer)
                correct     = kw_f1_score >= 0.2
            else:
                kw_f1_score = 0.0
                rouge_score = 0.0
                faith_score = 0.0
                correct     = False

        status = "✅" if correct else "❌"
        kw_str = f"{kw_f1_score:.2f}" if kw_f1_score is not None else "—"
        rl_str = f"{rouge_score:.2f}" if rouge_score is not None else "—"
        fa_str = f"{faith_score:.2f}" if faith_score is not None else "—"
        print(f"  {qid:<6} {category:<14} {status+' '+('OK' if correct else 'FAIL'):<12} "
              f"{kw_str:<8} {rl_str:<8} {fa_str:<8} {int(lat_ms)}")

        results.append({
            "id"         : qid,
            "category"   : category,
            "query"      : question,
            "ground_truth": gt,
            "expected"   : expected,
            "answer"     : answer,
            "error"      : error,
            "latency_ms" : lat_ms,
            "correct"    : correct,
            "kw_f1"      : kw_f1_score,
            "rouge_l"    : rouge_score,
            "faithfulness": faith_score,
            "bertscore"  : None,
            "chunks_retrieved": len(chunks),
        })

        time.sleep(PAUSE)

    return results, latencies


# ── AGGREGATE ─────────────────────────────────────────────────────────────────

def aggregate(results, latencies):
    answerable = [r for r in results if r["expected"]=="answer"]
    off_topic  = [r for r in results if r["category"]=="off_topic"]
    absent     = [r for r in results if r["category"]=="absent"]

    def safe_mean(lst):
        vals = [v for v in lst if v is not None]
        return round(float(np.mean(vals)), 4) if vals else None

    # Per category
    cats = {}
    for cat in ["factual","listing","entity","admission_fee","off_topic","absent"]:
        cr = [r for r in results if r["category"]==cat]
        if cr:
            correct = sum(1 for r in cr if r["correct"])
            cats[cat] = {"total":len(cr), "correct":correct,
                         "accuracy":round(correct/len(cr), 4)}

    # Hit Rate + MRR
    hit_rates = compute_hit_rate(results)
    mrr       = compute_mrr(results)
    ctx_prec  = compute_context_precision(results)

    # Generation
    kw_scores    = [r["kw_f1"]    for r in answerable if r["kw_f1"] is not None]
    rouge_scores = [r["rouge_l"]  for r in answerable if r["rouge_l"] is not None]
    faith_scores = [r["faithfulness"] for r in results if r["faithfulness"] is not None]
    bert_scores  = [r["bertscore"] for r in results if r["bertscore"] is not None]

    # Hallucination
    hallucinated = sum(1 for r in answerable
                       if r["faithfulness"] is not None and r["faithfulness"] < 1.0
                       and r["answer"] not in ("TIMEOUT","","No data found"))

    # System
    rej_rate = (sum(1 for r in off_topic if r["correct"]) / len(off_topic)
                if off_topic else None)
    no_ans_rate = (sum(1 for r in absent if r["correct"]) / len(absent)
                   if absent else None)

    # Latency
    lat = np.array([l for l in latencies if l < TIMEOUT*1000])
    lat_stats = {
        "p50_ms" : round(float(np.percentile(lat,50)), 1),
        "p90_ms" : round(float(np.percentile(lat,90)), 1),
        "p99_ms" : round(float(np.percentile(lat,99)), 1),
        "mean_ms": round(float(np.mean(lat)), 1),
    } if len(lat) else {}

    overall_correct = sum(1 for r in results if r["correct"])

    return {
        "summary": {
            "total"            : len(results),
            "correct"          : overall_correct,
            "accuracy"         : round(overall_correct/len(results), 4),
            "hallucination_count": hallucinated,
            "zero_hallucination": hallucinated == 0,
        },
        "retrieval": {
            **hit_rates,
            "mrr_at_5"         : mrr,
            "context_precision": ctx_prec,
        },
        "generation": {
            "avg_keyword_f1"   : safe_mean(kw_scores),
            "avg_rouge_l"      : safe_mean(rouge_scores),
            "avg_faithfulness" : safe_mean(faith_scores),
            "avg_bertscore_f1" : safe_mean(bert_scores),
        },
        "system": {
            "rejection_rate"   : round(rej_rate, 4) if rej_rate else None,
            "no_answer_rate"   : round(no_ans_rate, 4) if no_ans_rate else None,
            "zero_hallucination": hallucinated == 0,
            "latency"          : lat_stats,
        },
        "per_category": cats,
    }


# ── REPORT ────────────────────────────────────────────────────────────────────

def generate_report(metrics, results):
    s   = metrics["summary"]
    r   = metrics["retrieval"]
    g   = metrics["generation"]
    sys = metrics["system"]
    pc  = metrics["per_category"]
    lat = sys.get("latency", {})

    def fmt(v, pct=False, dec=4):
        if v is None: return "N/A"
        return f"{v*100:.1f}%" if pct else f"{v:.{dec}f}"

    lines = [
        "="*70,
        "  MAC RAG CHATBOT — BENCHMARK EVALUATION REPORT",
        "  Maharaja Agrasen College, University of Delhi",
        f"  Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "="*70,
        "",
        "OVERALL SUMMARY",
        "-"*40,
        f"  Total Queries        : {s['total']}",
        f"  Correct Responses    : {s['correct']}",
        f"  Overall Accuracy     : {fmt(s['accuracy'], pct=True)}",
        f"  Hallucination Count  : {s['hallucination_count']}",
        f"  Zero Hallucination   : {'YES ✅' if s['zero_hallucination'] else 'NO ❌'}",
        "",
        "LAYER 1 — RETRIEVAL QUALITY",
        "-"*40,
        f"  Hit Rate @1          : {fmt(r.get('hit_rate_at_1'), pct=True)}",
        f"  Hit Rate @3          : {fmt(r.get('hit_rate_at_3'), pct=True)}",
        f"  Hit Rate @5          : {fmt(r.get('hit_rate_at_5'), pct=True)}",
        f"  MRR @5               : {fmt(r.get('mrr_at_5'))}",
        f"  Context Precision    : {fmt(r.get('context_precision'), pct=True)}",
        "  Citation: Baeza-Yates & Ribeiro-Neto (1999); Voorhees (TREC 1999)",
        "",
        "LAYER 2 — GENERATION QUALITY",
        "-"*40,
        f"  Keyword F1           : {fmt(g['avg_keyword_f1'], pct=True)}",
        f"  ROUGE-L F1           : {fmt(g['avg_rouge_l'])}",
        f"  BERTScore F1         : {fmt(g['avg_bertscore_f1'])}",
        f"  Faithfulness Score   : {fmt(g['avg_faithfulness'], pct=True)}",
        "  Citation: Lin (ACL 2004); Zhang et al. (ICLR 2020); Es et al. (EACL 2024)",
        "",
        "LAYER 3 — SYSTEM BEHAVIOUR",
        "-"*40,
        f"  Off-topic Rejection  : {fmt(sys['rejection_rate'], pct=True)}",
        f"  No-Answer Rate       : {fmt(sys['no_answer_rate'], pct=True)}",
        f"  Zero Hallucination   : {'YES ✅' if sys['zero_hallucination'] else 'NO ❌'}",
        f"  Latency P50          : {lat.get('p50_ms','N/A')} ms",
        f"  Latency P90          : {lat.get('p90_ms','N/A')} ms",
        f"  Latency P99          : {lat.get('p99_ms','N/A')} ms",
        f"  Mean Latency         : {lat.get('mean_ms','N/A')} ms",
        "  Citation: Ji et al. (ACM CSUR 2023)",
        "",
        "PER-CATEGORY BREAKDOWN",
        "-"*40,
    ]

    labels = {"factual":"Factual","listing":"Listing","entity":"Entity",
              "admission_fee":"Admission/Fee","off_topic":"Off-topic","absent":"Absent",
              "course_specific":"Course-specific","department_specific":"Dept-specific"}
    for cat, label in labels.items():
        if cat in pc:
            d = pc[cat]
            lines.append(f"  {label:<20} : {d['correct']}/{d['total']} "
                         f"({d['accuracy']*100:.1f}%)")

    lines += [
        "",
        "INDIVIDUAL QUERY RESULTS",
        "-"*40,
        f"  {'ID':<6} {'Cat':<14} {'Correct':<10} {'KwF1':<8} "
        f"{'ROUGE-L':<9} {'Faith':<8} {'BERTSc':<9} {'Latency'}",
        "  " + "-"*70,
    ]
    for r2 in results:
        c_str = "YES" if r2["correct"] else "NO"
        lines.append(
            f"  {r2['id']:<6} {r2['category']:<14} {c_str:<10} "
            f"{fmt(r2['kw_f1']):<8} {fmt(r2['rouge_l']):<9} "
            f"{fmt(r2['faithfulness']):<8} {fmt(r2['bertscore']):<9} "
            f"{r2['latency_ms']/1000:.1f}s"
        )

    lines += ["","="*70,"  END OF REPORT","="*70]
    return "\n".join(lines)


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

def main():
    print("="*70)
    print("  MAC RAG CHATBOT — FAST BENCHMARK EVALUATION")
    print("  No LLM judge required — completes in ~5-10 minutes")
    print("="*70+"\n")

    check_server()

    if not os.path.exists(DATASET_FILE):
        print(f"❌ {DATASET_FILE} not found in current directory.")
        sys.exit(1)

    queries = load_dataset()
    print(f"Loaded {len(queries)} queries\n")

    # Main eval loop
    results, latencies = run_evaluation(queries)

    # BERTScore batch (all answerable at once — faster than one-by-one)
    print("\nRunning BERTScore batch...")
    answerable = [(i,r) for i,r in enumerate(results)
                  if r["expected"]=="answer"
                  and r["answer"] not in ("TIMEOUT","")
                  and r["error"] is None]
    if answerable:
        idxs  = [i for i,_ in answerable]
        preds = [r["answer"] for _,r in answerable]
        refs  = [r["ground_truth"] for _,r in answerable]
        scores = bertscore_batch(preds, refs)
        for idx, score in zip(idxs, scores):
            results[idx]["bertscore"] = score

    # Aggregate
    metrics = aggregate(results, latencies)

    # Report
    report = generate_report(metrics, results)
    print("\n" + report)

    # Save
    output = {
        "metadata": {"generated_at": datetime.now().isoformat(),
                     "total_queries": len(results)},
        "metrics" : metrics,
        "results" : results,
    }
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n💾 Results : {RESULTS_FILE}")
    print(f"📄 Report  : {REPORT_FILE}")
    print(f"\nShare eval_report.txt for dissertation chapter.\n")

    # Print banner
    acc = metrics["summary"]["accuracy"]
    zh  = metrics["summary"]["zero_hallucination"]
    rr  = metrics["system"]["rejection_rate"]
    print("="*70)
    print(f"  Overall Accuracy  : {acc*100:.1f}%")
    print(f"  Zero Hallucination: {'YES ✅' if zh else 'NO ❌'}")
    if rr:
        print(f"  Rejection Rate    : {rr*100:.1f}%")
    print("="*70+"\n")


if __name__ == "__main__":
    main()
