"""
Week-4 Custom Group-wise Multi-Model Evaluation
=================================================

What this script does:
1. Loads a ground-truth dataset (custom_eval_dataset.json) organized into 6 groups.
2. Runs every question against every model listed in MODELS, using the app's
   own /api/hallucination/check endpoint (this reuses the SAME retrieval +
   generation + grounding pipeline the real app uses, so results reflect the
   actual application, not a separate mock pipeline).
3. Computes 7 metrics per (question, model):
     1. Correctness       - embedding cosine similarity between the model's
                             answer and the ground_truth answer (nomic-embed-text)
     2. Faithfulness       - grounding_score returned by the app (% of answer
                             sentences supported by retrieved evidence)
     3. Hallucination Rate - 1 - faithfulness (proportion of unsupported claims)
     4. Retrieval Quality  - top-1 similarity score from the retrieval step
     5. Response Latency   - wall-clock seconds for the full request
     6. Token Usage        - approx. word count of (prompt context + answer),
                             used as a lightweight proxy since exact token
                             counts are not exposed by the endpoint
     7. Resource Usage     - peak CPU% and memory (MB) of the local `ollama`
                             process sampled during generation
4. Aggregates results by (group, model) and prints/saves which model wins
   each group on which metrics, with a short reason.

HOW TO RUN
----------
Prerequisites (all from earlier setup):
  - Ollama running (ollama serve) with all models pulled
  - Backend running: uvicorn app.main:app --host 0.0.0.0 --port 8000

Then, in a NEW terminal (with the backend venv activated):
    cd backend/evaluation
    python custom_group_evaluation.py

Results are written to:
    results/custom_group_evaluation_raw.json      (every single run)
    results/custom_group_evaluation_summary.json  (aggregated per group/model)
"""

import json
import time
import math
import os
import threading

import requests

try:
    import psutil
except ImportError:
    psutil = None
    print("[WARN] psutil not installed - resource usage metric will be skipped.")
    print("       Install with: pip install psutil")

# =========================================================
# CONFIG
# =========================================================

BACKEND_URL = "http://localhost:8000"
OLLAMA_URL = "http://localhost:11434"

MODELS = [
    "codellama:7b-instruct",
    "qwen2.5:1.5b",
    "phi3:mini",
]

EMBED_MODEL = "nomic-embed-text"

DATASET_PATH = os.path.join(os.path.dirname(__file__), "custom_eval_dataset.json")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
RAW_OUTPUT_PATH = os.path.join(RESULTS_DIR, "custom_group_evaluation_raw.json")
SUMMARY_OUTPUT_PATH = os.path.join(RESULTS_DIR, "custom_group_evaluation_summary.json")

TOP_K = 5


# =========================================================
# HELPERS
# =========================================================

def get_embedding(text):
    """Get an embedding vector from Ollama for correctness scoring."""
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/embeddings",
            json={"model": EMBED_MODEL, "prompt": text},
            timeout=60,
        )
        response.raise_for_status()
        return response.json().get("embedding", [])
    except Exception as e:
        print(f"[WARN] embedding failed: {e}")
        return []


def cosine_similarity(a, b):
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def find_ollama_process():
    """Find the running ollama process (server or a model runner) for resource sampling."""
    if psutil is None:
        return None
    for proc in psutil.process_iter(["pid", "name"]):
        try:
            name = (proc.info["name"] or "").lower()
            if "ollama" in name:
                return psutil.Process(proc.info["pid"])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return None


class ResourceSampler:
    """Samples CPU% and memory of the ollama process in a background thread
    while a request is in flight, and reports the peak values observed."""

    def __init__(self, interval=0.2):
        self.interval = interval
        self.peak_cpu = 0.0
        self.peak_mem_mb = 0.0
        self._stop_event = threading.Event()
        self._thread = None
        self._proc = find_ollama_process()

    def _run(self):
        if self._proc is None:
            return
        # Prime cpu_percent (first call always returns 0.0)
        try:
            self._proc.cpu_percent(interval=None)
        except Exception:
            pass
        while not self._stop_event.is_set():
            try:
                cpu = self._proc.cpu_percent(interval=None)
                mem_mb = self._proc.memory_info().rss / (1024 * 1024)
                self.peak_cpu = max(self.peak_cpu, cpu)
                self.peak_mem_mb = max(self.peak_mem_mb, mem_mb)
            except Exception:
                pass
            time.sleep(self.interval)

    def start(self):
        if psutil is None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        if psutil is None:
            return (0.0, 0.0)
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2)
        return (self.peak_cpu, self.peak_mem_mb)


def approx_token_count(text):
    """Lightweight proxy for token usage (~0.75 tokens per word for English text)."""
    words = len(text.split())
    return round(words / 0.75)


# =========================================================
# EVALUATION LOOP
# =========================================================

def run_single(question_item, model):
    question = question_item["question"]
    ground_truth = question_item["ground_truth"]

    sampler = ResourceSampler()
    sampler.start()

    start = time.time()
    try:
        response = requests.post(
            f"{BACKEND_URL}/api/hallucination/check",
            json={"question": question, "model": model, "top_k": TOP_K},
            timeout=180,
        )
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        latency = time.time() - start
        sampler.stop()
        print(f"    [ERROR] {model} on {question_item['id']}: {e}")
        return {
            "id": question_item["id"],
            "group": question_item["group"],
            "model": model,
            "question": question,
            "error": str(e),
            "latency_seconds": round(latency, 2),
        }

    latency = time.time() - start
    peak_cpu, peak_mem_mb = sampler.stop()

    answer = data.get("answer", "")
    grounding_score = data.get("grounding_score", 0)  # 0-100
    risk = data.get("risk", "UNKNOWN")
    retrieval_confidence = data.get("retrieval_confidence", 0)
    unsupported_claims = data.get("unsupported_claims", [])
    supported_claims = data.get("supported_claims", [])
    total_claims = len(supported_claims) + len(unsupported_claims)

    # --- Correctness: embedding similarity vs ground truth ---
    answer_emb = get_embedding(answer) if answer else []
    truth_emb = get_embedding(ground_truth)
    correctness = cosine_similarity(answer_emb, truth_emb)

    # --- Faithfulness / Hallucination ---
    faithfulness = grounding_score / 100.0
    hallucination_rate = (
        len(unsupported_claims) / total_claims if total_claims else (0.0 if answer else 1.0)
    )

    # --- Token usage (approx) ---
    token_usage = approx_token_count(question + " " + answer)

    return {
        "id": question_item["id"],
        "group": question_item["group"],
        "model": model,
        "question": question,
        "answer": answer,
        "ground_truth": ground_truth,
        "metrics": {
            "correctness": round(correctness, 4),
            "faithfulness": round(faithfulness, 4),
            "hallucination_rate": round(hallucination_rate, 4),
            "retrieval_quality": round(retrieval_confidence, 4),
            "latency_seconds": round(latency, 2),
            "token_usage_approx": token_usage,
            "peak_cpu_percent": round(peak_cpu, 1),
            "peak_memory_mb": round(peak_mem_mb, 1),
        },
        "risk": risk,
    }


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    print("=" * 70)
    print(f"Running {len(dataset)} questions x {len(MODELS)} models "
          f"= {len(dataset) * len(MODELS)} evaluations")
    print("=" * 70)

    all_results = []
    for model in MODELS:
        print(f"\n### MODEL: {model} ###")
        for item in dataset:
            print(f"  - [{item['group']}] {item['id']}: {item['question'][:60]}...")
            result = run_single(item, model)
            all_results.append(result)

    with open(RAW_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nRaw results saved to: {RAW_OUTPUT_PATH}")

    summary = aggregate(all_results)
    with open(SUMMARY_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {SUMMARY_OUTPUT_PATH}")

    print_report(summary)


# =========================================================
# AGGREGATION + REPORT
# =========================================================

METRIC_KEYS = [
    "correctness", "faithfulness", "hallucination_rate", "retrieval_quality",
    "latency_seconds", "token_usage_approx", "peak_cpu_percent", "peak_memory_mb",
]

# For these metrics, LOWER is better
LOWER_IS_BETTER = {"hallucination_rate", "latency_seconds", "token_usage_approx",
                   "peak_cpu_percent", "peak_memory_mb"}


def aggregate(all_results):
    groups = {}
    for r in all_results:
        if "metrics" not in r:
            continue
        g = r["group"]
        m = r["model"]
        groups.setdefault(g, {}).setdefault(m, {k: [] for k in METRIC_KEYS})
        for k in METRIC_KEYS:
            groups[g][m][k].append(r["metrics"][k])

    summary = {}
    for g, models in groups.items():
        summary[g] = {}
        for m, metric_lists in models.items():
            summary[g][m] = {
                k: round(sum(v) / len(v), 4) if v else None
                for k, v in metric_lists.items()
            }

        # Determine best model per metric for this group
        best_per_metric = {}
        for k in METRIC_KEYS:
            values = {m: summary[g][m][k] for m in summary[g] if summary[g][m][k] is not None}
            if not values:
                continue
            if k in LOWER_IS_BETTER:
                best_model = min(values, key=values.get)
            else:
                best_model = max(values, key=values.get)
            best_per_metric[k] = best_model
        summary[g]["_best_per_metric"] = best_per_metric

        # Overall winner: model that wins the most metrics in this group
        wins = {}
        for k, m in best_per_metric.items():
            wins[m] = wins.get(m, 0) + 1
        if wins:
            overall_winner = max(wins, key=wins.get)
            summary[g]["_overall_winner"] = overall_winner
            summary[g]["_reason"] = (
                f"{overall_winner} led on {wins[overall_winner]}/{len(best_per_metric)} "
                f"metrics in this group: "
                f"{', '.join(k for k, v in best_per_metric.items() if v == overall_winner)}."
            )

    return summary


def print_report(summary):
    print("\n" + "=" * 70)
    print("GROUP-WISE MODEL COMPARISON REPORT")
    print("=" * 70)
    for group, data in summary.items():
        print(f"\n### {group} ###")
        for model, metrics in data.items():
            if model.startswith("_"):
                continue
            print(f"  {model}:")
            for k in METRIC_KEYS:
                print(f"      {k:22s}: {metrics.get(k)}")
        if "_overall_winner" in data:
            print(f"  >>> BEST MODEL FOR THIS GROUP: {data['_overall_winner']}")
            print(f"      Why: {data['_reason']}")


if __name__ == "__main__":
    main()
