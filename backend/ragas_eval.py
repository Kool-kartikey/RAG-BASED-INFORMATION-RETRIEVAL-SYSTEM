import json
import time
import asyncio
import requests
import os
import sys
from datetime import datetime
import warnings

warnings.filterwarnings("ignore", message="Core Pydantic V1")
os.environ["RAGAS_DO_NOT_TRACK"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

LM_STUDIO_BASE_URL = "http://localhost:1234/v1"
LM_STUDIO_MODEL = "llama-3.2-3b-instruct"
FASTAPI_BASE_URL = "http://localhost:8000"
ASK_ENDPOINT = f"{FASTAPI_BASE_URL}/ask"
DEBUG_ENDPOINT = f"{FASTAPI_BASE_URL}/debug"

DATASET_FILE = "eval_dataset.json"
RESULTS_FILE = "ragas_results.json"
REPORT_FILE = "ragas_report.txt"

REQUEST_TIMEOUT = 240
PAUSE_BETWEEN = 1
TOP_K_CHUNKS = 3
MAX_CONTEXT_CHARS_PER_CHUNK = 1200

RAGAS_CATEGORIES = {"factual", "listing", "entity", "absent"}


def check_imports():
    missing = []
    try:
        import ragas
    except ImportError:
        missing.append("ragas")
    try:
        from openai import AsyncOpenAI
    except ImportError:
        missing.append("openai")
    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        missing.append("langchain-openai")
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
    except ImportError:
        missing.append("langchain-community")
    if missing:
        print(f"Missing packages: {missing}")
        print(f"Install with: pip install {' '.join(missing)}")
        sys.exit(1)


check_imports()

from ragas import SingleTurnSample, EvaluationDataset, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings


def check_fastapi() -> bool:
    try:
        r = requests.get(f"{FASTAPI_BASE_URL}/health", timeout=10)
        return r.status_code == 200
    except Exception:
        return False


def check_lm_studio() -> bool:
    try:
        r = requests.get(f"{LM_STUDIO_BASE_URL}/models", timeout=10)
        return r.status_code == 200
    except Exception:
        return False


def load_dataset(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "queries" in data:
        return data["queries"]

    if isinstance(data, list):
        return data

    raise ValueError("Invalid dataset format. Expected a list or an object with key 'queries'.")


def get_answer(question: str) -> str:
    try:
        r = requests.post(
            ASK_ENDPOINT,
            json={"question": question},
            timeout=REQUEST_TIMEOUT
        )
        if r.status_code == 200:
            return r.json().get("answer", "")
        return ""
    except Exception as e:
        print(f"  /ask error: {e}")
        return ""


def get_chunks(question: str, top_k: int = TOP_K_CHUNKS) -> list:
    try:
        r = requests.post(
            DEBUG_ENDPOINT,
            json={"question": question, "top_k": top_k},
            timeout=REQUEST_TIMEOUT
        )
        if r.status_code == 200:
            chunks = r.json().get("chunks", [])
            cleaned = []
            for c in chunks:
                text = (c.get("text", "") or "").strip()
                if not text:
                    continue
                cleaned.append(text[:MAX_CONTEXT_CHARS_PER_CHUNK])
            return cleaned
        return []
    except Exception as e:
        print(f"  /debug error: {e}")
        return []


def init_ragas_llm():
    llm = ChatOpenAI(
        base_url=LM_STUDIO_BASE_URL,
        api_key="not-needed",
        model=LM_STUDIO_MODEL,
        temperature=0.0,
        max_tokens=2048,
        timeout=180,
        model_kwargs={"n": 1},
    )
    return LangchainLLMWrapper(llm)


def init_ragas_embeddings():
    emb = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )
    return LangchainEmbeddingsWrapper(emb)


def collect_ragas_data(queries: list) -> tuple:
    samples = []
    raw_records = []

    evaluable = [q for q in queries if q["category"] in RAGAS_CATEGORIES]
    print(f"\n  Collecting data for {len(evaluable)} evaluable queries...\n")

    for i, query in enumerate(evaluable):
        qid = query["id"]
        question = query["query"]
        gt = query["ground_truth"]

        print(f"  [{i+1}/{len(evaluable)}] [{qid}] {question[:60]}")

        t0 = time.time()
        answer = get_answer(question)
        latency_ms = (time.time() - t0) * 1000

        if not answer:
            print("    ✗ No answer received — skipping")
            continue

        chunks = get_chunks(question)

        if not chunks:
            print("    ✗ No chunks retrieved — skipping")
            continue

        print(f"    ✓ Answer: {answer[:80]}...")
        print(f"    ✓ Chunks: {len(chunks)}, Latency: {latency_ms:.0f}ms")

        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=chunks,
            reference=gt,
        )
        samples.append(sample)

        raw_records.append({
            "id": qid,
            "category": query["category"],
            "query": question,
            "ground_truth": gt,
            "answer": answer,
            "retrieved_contexts": chunks,
            "latency_ms": round(latency_ms, 1),
        })

        time.sleep(PAUSE_BETWEEN)

    return samples, raw_records


def compute_system_metrics(queries: list, raw_records: list) -> dict:
    import numpy as np

    OFF_TOPIC_SIGNALS = [
        "i can only answer", "only answer questions related to",
        "not able to answer", "outside the scope",
        "please ask questions about", "off-topic",
    ]
    NO_DATA_SIGNALS = [
        "no data found", "not found", "not available",
        "no information", "cannot find",
    ]

    off_topic_queries = [q for q in queries if q["category"] == "off_topic"]
    off_topic_correct = 0
    print(f"\n  Testing {len(off_topic_queries)} off-topic queries...")
    for q in off_topic_queries:
        answer = get_answer(q["query"])
        is_rejected = any(sig in answer.lower() for sig in OFF_TOPIC_SIGNALS)
        if is_rejected:
            off_topic_correct += 1
        print(f"    [{q['id']}] {'REJECTED' if is_rejected else 'NOT REJECTED'}: {answer[:60]}")
        time.sleep(1)

    absent_queries = [q for q in queries if q["category"] == "absent"]
    absent_correct = 0
    print(f"\n  Testing {len(absent_queries)} absent queries...")
    for q in absent_queries:
        answer = get_answer(q["query"])
        is_no_data = any(sig in answer.lower() for sig in NO_DATA_SIGNALS)
        if is_no_data:
            absent_correct += 1
        print(f"    [{q['id']}] {'NO DATA' if is_no_data else 'HALLUCINATED'}: {answer[:60]}")
        time.sleep(1)

    latencies = [r["latency_ms"] for r in raw_records]
    lat_stats = {}
    if latencies:
        lat_arr = np.array(latencies)
        lat_stats = {
            "p50_ms": round(float(np.percentile(lat_arr, 50)), 1),
            "p90_ms": round(float(np.percentile(lat_arr, 90)), 1),
            "p99_ms": round(float(np.percentile(lat_arr, 99)), 1),
            "mean_ms": round(float(np.mean(lat_arr)), 1),
        }

    return {
        "rejection_rate": round(off_topic_correct / len(off_topic_queries), 4) if off_topic_queries else None,
        "no_answer_rate": round(absent_correct / len(absent_queries), 4) if absent_queries else None,
        "zero_hallucination": absent_correct == len(absent_queries),
        "off_topic_detail": f"{off_topic_correct}/{len(off_topic_queries)}",
        "absent_detail": f"{absent_correct}/{len(absent_queries)}",
        "latency": lat_stats,
    }


def _set_if_exists(obj, name, value):
    try:
        setattr(obj, name, value)
    except Exception:
        pass


async def run_ragas(samples: list, evaluator_llm, evaluator_embeddings) -> dict:
    if not samples:
        return {}

    dataset = EvaluationDataset(samples=samples)

    faithfulness.llm = evaluator_llm
    answer_relevancy.llm = evaluator_llm
    answer_relevancy.embeddings = evaluator_embeddings
    context_precision.llm = evaluator_llm

    _set_if_exists(answer_relevancy, "strictness", 1)
    _set_if_exists(faithfulness, "strictness", 1)
    _set_if_exists(context_precision, "strictness", 1)

    print(f"\n  Running RAGAS on {len(samples)} samples...")
    print(f"  Judge LLM: {LM_STUDIO_MODEL} @ {LM_STUDIO_BASE_URL}\n")

    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=evaluator_llm,
    )

    return result


def generate_report(ragas_result, system_metrics: dict, raw_records: list) -> str:
    lat = system_metrics.get("latency", {})

    def safe_mean(key):
        try:
            scores = ragas_result[key]
            if scores is None:
                return None
            if hasattr(scores, "__iter__"):
                valid = [s for s in scores if s is not None]
                return round(sum(valid) / len(valid), 4) if valid else None
            return round(float(scores), 4)
        except Exception:
            return None

    faith = safe_mean("faithfulness")
    relevancy = safe_mean("answer_relevancy")
    precision = safe_mean("context_precision")

    lines = [
        "=" * 70,
        "  MAC RAG CHATBOT — RAGAS EVALUATION REPORT",
        "  Maharaja Agrasen College, University of Delhi",
        f"  Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"  Judge LLM : {LM_STUDIO_MODEL} (local)",
        "=" * 70,
        "",
        "LAYER 1 — RETRIEVAL QUALITY",
        "-" * 40,
        "  Context Precision    : {}".format(
            "N/A" if precision is None else f"{precision:.4f} ({precision*100:.1f}%)"),
        "",
        "LAYER 2 — GENERATION QUALITY",
        "-" * 40,
        "  Faithfulness         : {}".format(
            "N/A" if faith is None else f"{faith:.4f} ({faith*100:.1f}%)"),
        "  Answer Relevancy     : {}".format(
            "N/A" if relevancy is None else f"{relevancy:.4f} ({relevancy*100:.1f}%)"),
        "",
        "LAYER 3 — SYSTEM BEHAVIOUR",
        "-" * 40,
        "  Off-topic Rejection  : {} ({})".format(
            "N/A" if system_metrics["rejection_rate"] is None else f"{system_metrics['rejection_rate']*100:.1f}%",
            system_metrics.get("off_topic_detail", "")
        ),
        "  No-Answer Rate       : {} ({})".format(
            "N/A" if system_metrics["no_answer_rate"] is None else f"{system_metrics['no_answer_rate']*100:.1f}%",
            system_metrics.get("absent_detail", "")
        ),
        "  Zero Hallucination   : {}".format(
            "YES" if system_metrics["zero_hallucination"] else "NO"
        ),
        "  Latency P50          : {} ms".format(lat.get("p50_ms", "N/A")),
        "  Latency P90          : {} ms".format(lat.get("p90_ms", "N/A")),
        "  Latency P99          : {} ms".format(lat.get("p99_ms", "N/A")),
        "  Mean Latency         : {} ms".format(lat.get("mean_ms", "N/A")),
        "",
        "PER-QUERY RESULTS",
        "-" * 70,
        "  {:<6} {:<14} {:<55} {}".format("ID", "Category", "Query (truncated)", "Latency"),
        "  " + "-" * 66,
    ]

    for r in raw_records:
        lines.append("  {:<6} {:<14} {:<55} {}ms".format(
            r["id"],
            r["category"],
            r["query"][:53],
            int(r["latency_ms"])
        ))

    lines += [
        "",
        "=" * 70,
        "  DISSERTATION CITATION",
        "-" * 70,
        "  Es, S., James, J., Espinosa-Anke, L., & Schockaert, S. (2024).",
        "  RAGAS: Automated evaluation of retrieval augmented generation.",
        "  In Proceedings of EACL 2024 (System Demonstrations), pp. 150-158.",
        "  DOI: 10.18653/v1/2024.eacl-demo.16",
        "=" * 70,
    ]

    return "\n".join(lines)


async def main():
    print("=" * 70)
    print("  MAC RAG CHATBOT — RAGAS EVALUATION")
    print("  Maharaja Agrasen College, University of Delhi")
    print("=" * 70 + "\n")

    print("  Checking servers...")
    if not check_fastapi():
        print("  ❌ FastAPI not reachable. Start with: python api.py")
        sys.exit(1)
    print("  ✅ FastAPI OK")

    if not check_lm_studio():
        print("  ❌ LM Studio not reachable at localhost:1234")
        print("     Open LM Studio → load llama-3.2-3b-instruct → Start Server")
        sys.exit(1)
    print("  ✅ LM Studio OK\n")

    if not os.path.exists(DATASET_FILE):
        print(f"  ❌ {DATASET_FILE} not found.")
        sys.exit(1)

    queries = load_dataset(DATASET_FILE)
    print(f"  Loaded {len(queries)} queries from {DATASET_FILE}")

    print("\n  Initialising RAGAS evaluator LLM...")
    evaluator_llm = init_ragas_llm()
    print("  ✅ LLM wrapper ready")

    print("  Initialising embeddings for Answer Relevancy...")
    evaluator_embeddings = init_ragas_embeddings()
    print("  ✅ Embeddings ready\n")

    print("─" * 70)
    print("  PHASE 1: Collecting chatbot responses")
    print("─" * 70)
    samples, raw_records = collect_ragas_data(queries)
    print(f"\n  Collected {len(samples)} valid samples for RAGAS scoring.")

    print("\n" + "─" * 70)
    print("  PHASE 2: System behaviour metrics")
    print("─" * 70)
    system_metrics = compute_system_metrics(queries, raw_records)

    print("\n" + "─" * 70)
    print("  PHASE 3: RAGAS scoring")
    print("─" * 70)
    ragas_result = await run_ragas(samples, evaluator_llm, evaluator_embeddings)

    report = generate_report(ragas_result, system_metrics, raw_records)
    print("\n" + report)

    output = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "judge_llm": LM_STUDIO_MODEL,
            "total_samples": len(samples),
        },
        "ragas_scores": {
            "faithfulness": str(ragas_result.get("faithfulness", "N/A")),
            "answer_relevancy": str(ragas_result.get("answer_relevancy", "N/A")),
            "context_precision": str(ragas_result.get("context_precision", "N/A")),
        },
        "system_metrics": system_metrics,
        "raw_records": raw_records,
    }

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n  Saved: {RESULTS_FILE}")
    print(f"  Saved: {REPORT_FILE}\n")


if __name__ == "__main__":
    asyncio.run(main())