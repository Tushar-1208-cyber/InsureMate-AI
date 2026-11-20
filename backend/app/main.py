
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from fastapi import Body
from typing import Optional, List, Dict
import requests
import re

from app.api.knowledge import router as knowledge_router
from app.api.retrieval import router as retrieval_router
from app.api.upload import router as upload_router


# =========================================================
# APPLICATION
# =========================================================

app = FastAPI(title="InsureMate AI")


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://192.168.78.129:5173",
        "http://192.168.78.129:5174",
        "http://192.168.78.129:5175",
        "http://localhost:5174",
        "http://localhost:5175",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# ROUTERS
# =========================================================

app.include_router(
    knowledge_router,
    prefix="/api/knowledge"
)

app.include_router(
    retrieval_router,
    prefix="/api/retrieval"
)

app.include_router(
    upload_router,
    prefix="/api/upload"
)


# =========================================================
# OLLAMA
# =========================================================

OLLAMA_URL = "http://localhost:11434/api/generate"

MODEL = "codellama:7b-instruct"


# =========================================================
# ROOT
# =========================================================

@app.get("/")
def root():
    return {
        "application": "InsureMate AI",
        "status": "running"
    }


# =========================================================
# HALLUCINATION REQUEST MODEL
# =========================================================

class HallucinationRequest(BaseModel):
    question: str
    model: str = "codellama:7b-instruct"
    top_k: int = 3

class ChatRequest(BaseModel):
    question: str
    model: str = "codellama:7b-instruct"
    history: List[Dict[str, str]] = Field(default_factory=list)


# =========================================================
# OLLAMA HELPER
# =========================================================

def call_ollama(model: str, prompt: str):

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": 0,
            "options": {
                "num_ctx": 1024,
                "temperature": 0,
                "num_predict": 120
            }
        },
        timeout=180
    )

    response.raise_for_status()

    return response.json()["response"].strip()


# =========================================================
# GROUNDING CALCULATION
# =========================================================

def calculate_grounding(answer, results):

    evidence_text = " ".join(
        item.get("text", "").lower()
        for item in results
    )

    sentences = [
        sentence.strip()
        for sentence in re.split(
            r"[.!?]",
            answer
        )
        if len(sentence.strip()) > 20
    ]

    stop_words = {
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "from",
        "are",
        "was",
        "were",
        "will",
        "have",
        "has",
        "been",
        "their",
        "they",
        "into",
        "under",
        "provided",
        "after",
        "before",
        "same",
        "which",
        "where",
        "than",
        "then",
        "also",
        "only",
        "based",
        "policy",
        "evidence",
        "insured"
    }

    supported_claims = []
    unsupported_claims = []

    for sentence in sentences:

        words = re.findall(
            r"[a-zA-Z]{4,}",
            sentence.lower()
        )

        keywords = [
            word
            for word in words
            if word not in stop_words
        ]

        if not keywords:
            continue

        matches = sum(
            1
            for word in keywords
            if word in evidence_text
        )

        ratio = matches / len(keywords)

        if ratio >= 0.35:
            supported_claims.append(sentence)
        else:
            unsupported_claims.append(sentence)

    total_claims = (
        len(supported_claims)
        + len(unsupported_claims)
    )

    if total_claims == 0:
        grounding_score = 0
    else:
        grounding_score = round(
            len(supported_claims)
            / total_claims
            * 100
        )

    if grounding_score >= 80:
        risk = "LOW"
    elif grounding_score >= 60:
        risk = "MEDIUM"
    else:
        risk = "HIGH"

    return {
        "grounding_score": grounding_score,
        "risk": risk,
        "supported_claims": supported_claims,
        "unsupported_claims": unsupported_claims
    }


# =========================================================
# MODELS API
# =========================================================

@app.get("/api/models")
def get_models():

    try:

        response = requests.get(
            "http://localhost:11434/api/tags",
            timeout=10
        )

        response.raise_for_status()

        data = response.json()

        return {
            "models": data.get("models", [])
        }

    except Exception as e:

        return {
            "models": [],
            "error": str(e)
        }


# =========================================================
# HALLUCINATION DETECTOR
# =========================================================

@app.post("/api/hallucination/check")
def hallucination_check(request: HallucinationRequest):

    # ---------------------------------------------------------
    # 1. RETRIEVE POLICY EVIDENCE
    # ---------------------------------------------------------

    retrieval_response = requests.post(
        "http://localhost:8000/api/retrieval/search",
        json={
            "question": request.question,
            "top_k": request.top_k
        },
        timeout=60
    )

    retrieval_response.raise_for_status()

    retrieval_data = retrieval_response.json()

    results = retrieval_data.get("results", [])

    if not results:
        return {
            "question": request.question,
            "model": request.model,
            "answer": "No relevant policy evidence was found.",
            "grounding_score": 0,
            "risk": "HIGH",
            "supported_claims": [],
            "unsupported_claims": [],
            "sources": [],
            "retrieved_chunks": [],
            "retrieval_confidence": 0
        }

    # ---------------------------------------------------------
    # 2. BUILD EVIDENCE
    # ---------------------------------------------------------

    evidence_parts = []

    for i, result in enumerate(results, 1):
        evidence_parts.append(
            f"""
EVIDENCE {i}
Document: {result.get("document", "")}
Page: {result.get("page", "")}
Chunk: {result.get("chunk_id", "")}

{result.get("text", "")}
"""
        )

    evidence = "\n".join(evidence_parts)

    # ---------------------------------------------------------
    # 3. GENERATE ANSWER
    # ---------------------------------------------------------

    prompt = f"""
You are InsureMate, an insurance policy assistant.

Answer the USER QUESTION using ONLY the POLICY EVIDENCE.

USER QUESTION:
{request.question}

POLICY EVIDENCE:
{evidence}

Rules:
- Give the answer directly.
- Use only facts from the evidence.
- Do not invent anything.
- Preserve important numbers and conditions.
- If the evidence contains the answer, answer it.
- Do not say there is insufficient information when the evidence clearly answers the question.

ANSWER:
"""

    try:
          

        answer = call_ollama(
            request.model,
            prompt
        )

    except Exception as e:
        return {
            "question": request.question,
            "model": request.model,
            "answer": f"Model generation failed: {str(e)}",
            "grounding_score": 0,
            "risk": "HIGH",
            "supported_claims": [],
            "unsupported_claims": [],
            "sources": results,
            "retrieved_chunks": results,
            "retrieval_confidence": (
                results[0].get("score", 0)
                if results else 0
            )
        }

    # ---------------------------------------------------------
    # 4. CHECK GROUNDING
    # ---------------------------------------------------------

    sentences = re.split(
        r'(?<=[.!?])\s+',
        answer
    )

    supported_claims = []
    unsupported_claims = []

    for sentence in sentences:

        sentence = sentence.strip()

        if not sentence:
            continue

        # Ignore the fallback sentence itself
        if (
            "provided policy evidence does not contain enough information"
            in sentence.lower()
        ):
            unsupported_claims.append(sentence)
            continue

        sentence_words = set(
            re.findall(
                r'\b[a-zA-Z]{4,}\b',
                sentence.lower()
            )
        )

        best_overlap = 0

        for result in results:

            evidence_words = set(
                re.findall(
                    r'\b[a-zA-Z]{4,}\b',
                    result.get("text", "").lower()
                )
            )

            if not sentence_words:
                continue

            overlap = len(
                sentence_words & evidence_words
            ) / len(sentence_words)

            best_overlap = max(
                best_overlap,
                overlap
            )

        if best_overlap >= 0.35:
            supported_claims.append(sentence)
        else:
            unsupported_claims.append(sentence)

    total_claims = (
        len(supported_claims)
        + len(unsupported_claims)
    )

    if total_claims == 0:
        grounding_score = 0
    else:
        grounding_score = round(
            len(supported_claims)
            / total_claims
            * 100
        )

    if grounding_score >= 80:
        risk = "LOW"
    elif grounding_score >= 60:
        risk = "MEDIUM"
    else:
        risk = "HIGH"

    # ---------------------------------------------------------
    # 5. RETURN RESULT
    # ---------------------------------------------------------

    return {
        "question": request.question,
        "model": request.model,
        "answer": answer,
        "grounding_score": grounding_score,
        "risk": risk,
        "supported_claims": supported_claims,
        "unsupported_claims": unsupported_claims,
        "sources": results,
        "retrieved_chunks": results,
        "retrieval_confidence": (
            results[0].get("score", 0)
            if results else 0
        )
    }

# =========================================================
# RAG CHAT
# =========================================================

@app.post("/api/chat")
def chat(
    question: str = "",
    model: str = MODEL,
    request: Optional[ChatRequest] = Body(default=None)
):
    """
    Main conversational RAG endpoint with multi-model orchestration.
    """

    if request is not None:
        question = request.question
        history = request.history or []
    else:
        history = []

    question = (question or "").strip()

    fallback = (
        "The available insurance documents do not contain enough "
        "information to answer this question."
    )

    if not question:
        return {
            "question": question,
            "model": None,
            "complexity": None,
            "models_used": [],
            "answer": "Please enter a question about your insurance policy.",
            "validation": None,
            "sources": [],
            "retrieved_chunks": []
        }

    # =====================================================
    # MULTI-MODEL PIPELINE LOG
    # =====================================================

    print("\n" + "=" * 65)
    print("INSUREMATE MULTI-MODEL PIPELINE")
    print("=" * 65)
    print(f"Question: {question}")

    # =====================================================
    # 1. RETRIEVAL
    # =====================================================

    print("\nStage 0: RAG → Retrieving relevant policy evidence")

    retrieval_question = question
    q_lower = question.lower().strip()

    # Follow-up question handling
    if history and len(q_lower.split()) <= 6:

        previous_user_questions = [
            message.get("content", "").strip()
            for message in history
            if message.get("role") == "user"
            and message.get("content", "").strip()
        ]

        if previous_user_questions:
            retrieval_question = (
                previous_user_questions[-1]
                + " "
                + question
            )

    try:
        retrieval_response = requests.post(
            "http://localhost:8000/api/retrieval/search",
            json={
                "question": retrieval_question,
                "top_k": 3
            },
            timeout=120
        )

        retrieval_response.raise_for_status()

        retrieval_data = retrieval_response.json()
        results = retrieval_data.get("results", [])

        print(f"Retrieved chunks: {len(results)}")

    except Exception as e:

        print(f"Retrieval failed: {str(e)}")
        print("=" * 65 + "\n")

        return {
            "question": question,
            "model": None,
            "complexity": None,
            "models_used": [],
            "answer": fallback,
            "validation": "RETRIEVAL_FAILED",
            "sources": [],
            "retrieved_chunks": [],
            "retrieval_confidence": 0,
            "error": str(e)
        }

    best_score = max(
        (float(item.get("score", 0)) for item in results),
        default=0
    )

    print(f"Best retrieval score: {best_score:.4f}")

    # =====================================================
    # 2. BASIC RELEVANCE CHECK
    # =====================================================

    stop_words = {
        "what", "when", "where", "which", "who", "whom", "why",
        "how", "is", "are", "was", "were", "do", "does", "did",
        "can", "could", "would", "should", "will", "the", "a",
        "an", "and", "or", "to", "of", "for", "in", "on", "at",
        "from", "with", "about", "after", "before", "under",
        "this", "that", "these", "those", "it", "its", "be",
        "been", "being", "i", "we", "you", "your", "my", "me",
        "they", "them", "their"
    }

    def normalize_term(word):
        replacements = {
            "hospitalization": "hospitalisation",
            "hospitalized": "hospitalised",
            "hospitalize": "hospitalise",
            "organization": "organisation",
            "organizations": "organisations",
            "authorized": "authorised",
            "authorization": "authorisation",
            "behavior": "behaviour",
            "center": "centre"
        }

        return replacements.get(word.lower(), word.lower())

    question_words = {
        normalize_term(word)
        for word in re.findall(
            r"\b[a-zA-Z]{4,}\b",
            retrieval_question.lower()
        )
        if word.lower() not in stop_words
    }

    evidence_text = " ".join(
        item.get("text", "").lower()
        for item in results
    )

    evidence_words = {
        normalize_term(word)
        for word in re.findall(
            r"\b[a-zA-Z]{4,}\b",
            evidence_text
        )
    }

    matched_words = question_words & evidence_words

    relevance_ratio = (
        len(matched_words) / len(question_words)
        if question_words else 1
    )

    evidence_is_relevant = (
        bool(results)
        and (
            best_score >= 0.60
            or (
                best_score >= 0.45
                and relevance_ratio >= 0.30
            )
        )
    )

    print(f"Evidence relevant: {evidence_is_relevant}")

    if not evidence_is_relevant:

        print("Pipeline stopped: No relevant evidence found")
        print("=" * 65 + "\n")

        return {
            "question": question,
            "model": None,
            "complexity": None,
            "models_used": [],
            "answer": fallback,
            "validation": "NO_RELEVANT_EVIDENCE",
            "sources": [],
            "retrieved_chunks": results,
            "retrieval_confidence": best_score
        }

    # =====================================================
    # 3. SELECT RELEVANT EVIDENCE
    # =====================================================

    focused_results = results[:3]

    # =====================================================
    # 4. BUILD POLICY CONTEXT
    # =====================================================

    context_parts = []

    for index, result in enumerate(focused_results, 1):

        context_parts.append(
            f"""
SOURCE {index}
Document: {result.get("document")}
Page: {result.get("page")}
Chunk ID: {result.get("chunk_id")}

POLICY TEXT:
{result.get("text", "")}
"""
        )

    context = "\n".join(context_parts)

    # =====================================================
    # 5. QWEN ROUTING
    # =====================================================

    print("\nStage 1: Qwen 2.5 1.5B → Analyzing question complexity")

    router_prompt = f"""
You are a routing assistant for an insurance AI system.

Classify the following user question as either SIMPLE or COMPLEX.

SIMPLE:
- Direct factual question
- One clear piece of information
- Can be answered briefly from the policy evidence

COMPLEX:
- Requires explanation, comparison, multiple conditions,
  calculations, interpretation, or combining multiple pieces
  of policy information

USER QUESTION:
{question}

Return ONLY one word:

SIMPLE

or

COMPLEX
"""

    try:

        routing_decision = call_ollama(
            "qwen2.5:1.5b",
            router_prompt
        ).strip().upper()

    except Exception as e:

        print(f"Qwen routing failed: {str(e)}")
        print("Defaulting to COMPLEX route")

        routing_decision = "COMPLEX"

    if (
        "SIMPLE" in routing_decision
        and "COMPLEX" not in routing_decision
    ):

        selected_model = "qwen2.5:1.5b"
        complexity = "simple"

    else:

        selected_model = "codellama:7b-instruct"
        complexity = "complex"

    print(f"Qwen decision: {complexity.upper()}")
    print(f"Selected generation model: {selected_model}")

    # =====================================================
    # 6. ANSWER GENERATION
    # =====================================================

    print(f"\nStage 2: {selected_model} → Generating grounded answer")

    prompt = f"""
You are InsureMate, an insurance policy assistant.

Your job is to answer the user's question using ONLY the information
provided in POLICY EVIDENCE.

IMPORTANT:
- Do NOT use any outside knowledge.
- Do NOT invent information.
- Do NOT add conditions, limits, waiting periods, or exclusions unless
  they are explicitly present in the evidence.
- Combine information from multiple evidence sections when relevant.
- If the question asks about multiple things, answer every part that is
  supported by the evidence.
- You may summarize and paraphrase the policy text.
- Preserve important numbers, durations, percentages and conditions.
- Do NOT mention "source", "chunk", "retrieval", "policy evidence",
  "document", or model names.
- Do NOT explain your reasoning.
- Return ONLY the final answer.

If some parts of the question are supported and some are not, answer the
supported parts. Do NOT reject the entire question.

Only if NONE of the evidence can answer the question, return exactly:

The available insurance documents do not contain enough information to answer this question.

CURRENT USER QUESTION:
{question}

POLICY EVIDENCE:
{context}

FINAL ANSWER:
"""

    try:

        answer = call_ollama(
            selected_model,
            prompt
        )

    except Exception as e:

        print(f"Answer generation failed: {str(e)}")
        print("=" * 65 + "\n")

        return {
            "question": question,
            "model": selected_model,
            "complexity": complexity,
            "models_used": [
                "qwen2.5:1.5b",
                selected_model
            ],
            "answer": fallback,
            "validation": "GENERATION_FAILED",
            "sources": [
                {
                    "document": item.get("document"),
                    "page": item.get("page"),
                    "chunk_id": item.get("chunk_id"),
                    "score": item.get("score")
                }
                for item in results
            ],
            "retrieved_chunks": results,
            "retrieval_confidence": best_score,
            "error": str(e)
        }

    answer = (answer or "").strip()

    print(f"\nGenerated answer from {selected_model}:")
    print(answer)

    if not answer:
        answer = fallback

    # =====================================================
    # 7. PHI-3 VALIDATION
    # =====================================================

    print("\nStage 3: Phi-3 Mini → Validating answer grounding")

    validator_prompt = f"""
You are a binary answer validator.

Determine whether the ANSWER contains claims that are supported by
the POLICY EVIDENCE.

POLICY EVIDENCE:
{context}

ANSWER:
{answer}

Return EXACTLY one word.

Return:

VALID

if the answer is supported by the policy evidence.

Return:

INVALID

only if the answer contains information that is clearly unsupported,
invented, or contradicts the policy evidence.

Do not explain your decision.
Do not write any other words.
"""

    try:

        validation_raw = call_ollama(
            "phi3:mini",
            validator_prompt
        )

        validation_raw = (
            validation_raw or ""
        ).strip().upper()

        print(f"Phi-3 raw validation result: {validation_raw}")

        # Strict normalization
        if validation_raw == "VALID":
            validation = "VALID"

        elif validation_raw == "INVALID":
            validation = "INVALID"

        else:
            # If Phi-3 gives an unexpected response,
            # do not automatically reject CodeLlama's answer
            validation = "VALID"

    except Exception as e:

        print(f"Phi-3 validation failed: {str(e)}")

        # Keep generated answer if validator fails
        validation = "VALID"

    print(f"Final validation decision: {validation}")

    if validation == "INVALID":

        print("Validation failed → Returning safe fallback")

        answer = fallback

    else:

        print("Validation passed → Returning generated answer")
    # =====================================================
    # 8. BUILD SOURCES
    # =====================================================

    sources = [
        {
            "document": item.get("document"),
            "page": item.get("page"),
            "chunk_id": item.get("chunk_id"),
            "score": item.get("score")
        }
        for item in results
    ]

    # =====================================================
    # 9. MODELS USED
    # =====================================================

    models_used = list(
        dict.fromkeys([
            "qwen2.5:1.5b",
            selected_model,
            "phi3:mini"
        ])
    )

    print("\nPipeline completed successfully")
    print("Models actually used:")

    for used_model in models_used:
        print(f"  ✓ {used_model}")

    print("=" * 65 + "\n")

    # =====================================================
    # 10. RETURN RESULT
    # =====================================================

    return {
        "question": question,
        "model": selected_model,
        "complexity": complexity,
        "models_used": models_used,
        "answer": answer,
        "validation": validation,
        "sources": sources,
        "retrieved_chunks": results,
        "retrieval_confidence": best_score
    }



# =========================================================
# WEEK-4 CUSTOM GROUP-WISE EVALUATION (DASHBOARD DATA)
# =========================================================

import json as _json
import os as _os

_CUSTOM_EVAL_SUMMARY_PATH = _os.path.join(
    _os.path.dirname(__file__),
    "..",
    "evaluation",
    "results",
    "custom_group_evaluation_summary.json"
)


@app.get("/api/evaluation/custom-groups")
def get_custom_group_evaluation():
    """
    Serves the Week-4 custom ground-truth, group-wise, multi-model
    evaluation summary (produced by evaluation/custom_group_evaluation.py)
    to the frontend dashboard.
    """
    try:
        with open(_CUSTOM_EVAL_SUMMARY_PATH, "r", encoding="utf-8") as f:
            data = _json.load(f)
        return {"status": "ok", "groups": data}
    except FileNotFoundError:
        return {
            "status": "not_found",
            "groups": {},
            "message": (
                "Run backend/evaluation/custom_group_evaluation.py first "
                "to generate results."
            )
        }
    except Exception as e:
        return {"status": "error", "groups": {}, "message": str(e)}


# =========================================================
# WEEK-4 CUSTOM DATASET (RAW QUESTIONS FOR "TEST DATASET" PAGE)
# =========================================================

_CUSTOM_DATASET_PATH = _os.path.join(
    _os.path.dirname(__file__),
    "..",
    "evaluation",
    "custom_eval_dataset.json"
)


@app.get("/api/evaluation/custom-dataset")
def get_custom_dataset():
    """
    Serves the raw 24-question ground-truth dataset
    (custom_eval_dataset.json) to the Test Dataset page.
    """
    try:
        with open(_CUSTOM_DATASET_PATH, "r", encoding="utf-8") as f:
            data = _json.load(f)
        return {"status": "ok", "questions": data}
    except FileNotFoundError:
        return {
            "status": "not_found",
            "questions": [],
            "message": (
                "custom_eval_dataset.json not found in backend/evaluation/"
            )
        }
    except Exception as e:
        return {"status": "error", "questions": [], "message": str(e)}


# =========================================================
# WEEK-4 Q&A RESULTS (INDIVIDUAL QUESTION-LEVEL ANSWERS)
# =========================================================

def _find_first_existing(candidates):
    for path in candidates:
        if _os.path.exists(path):
            return path
    return None


_BACKEND_DIR = _os.path.dirname(__file__)  # .../backend/app

# Candidate locations for the 25-question "corrected" evaluation
_CORRECTED_RESULTS_CANDIDATES = [
    _os.path.join(_BACKEND_DIR, "..", "evaluation_corrected", "results", "corrected_scores.json"),
    _os.path.join(_BACKEND_DIR, "..", "evaluation", "results", "corrected_scores.json"),
    _os.path.join(_BACKEND_DIR, "..", "evaluation", "results", "independent_evaluation.json"),
    _os.path.join(_BACKEND_DIR, "..", "evaluation", "results", "evaluation_scores.json"),
]

# Candidate locations for the 10-question repository/codebase evaluation
_REPOSITORY_RESULTS_CANDIDATES = [
    _os.path.join(_BACKEND_DIR, "..", "evaluation", "repository", "repository_evaluation.json"),
    _os.path.join(_BACKEND_DIR, "..", "evaluation", "repository", "repository_analysis.json"),
]


@app.get("/api/evaluation/qa-results")
def get_qa_results():
    """
    Serves individual question-level results (question + each model's
    answer + verdict) for the 25-question Week-4 evaluation, and the
    10-question Repository Understanding evaluation, so they can be
    browsed inside the app instead of only as raw JSON files.
    """
    result = {"status": "ok", "corrected": None, "repository": None, "messages": []}

    corrected_path = _find_first_existing(_CORRECTED_RESULTS_CANDIDATES)
    if corrected_path:
        try:
            with open(corrected_path, "r", encoding="utf-8") as f:
                result["corrected"] = _json.load(f)
        except Exception as e:
            result["messages"].append(f"Failed to read {corrected_path}: {e}")
    else:
        result["messages"].append(
            "No Week-4 25-question results file found (looked for "
            "corrected_scores.json / independent_evaluation.json / "
            "evaluation_scores.json)."
        )

    repository_path = _find_first_existing(_REPOSITORY_RESULTS_CANDIDATES)
    if repository_path:
        try:
            with open(repository_path, "r", encoding="utf-8") as f:
                result["repository"] = _json.load(f)
        except Exception as e:
            result["messages"].append(f"Failed to read {repository_path}: {e}")
    else:
        result["messages"].append(
            "No Repository Understanding results file found (looked for "
            "repository_evaluation.json / repository_analysis.json)."
        )

    if result["corrected"] is None and result["repository"] is None:
        result["status"] = "not_found"

    return result


# =========================================================
# WEEK 5 — GUARDRAILS + AI OUTPUT TESTING
# =========================================================

from app import guardrails as _guardrails
from app import output_tests as _output_tests


def _core_answer_pipeline(question: str, model: str, top_k: int = 3):
    """
    Internal, reusable version of the retrieve -> generate -> ground
    pipeline (mirrors /api/hallucination/check), used by the Week-5
    guardrails and output-testing endpoints so they do not depend on
    any particular existing endpoint's exact wiring.

    Returns a dict: {
        answer, grounding_score, risk, supported_claims,
        unsupported_claims, sources, retrieved_chunks,
        retrieval_confidence, evidence_is_relevant
    }
    """
    try:
        retrieval_response = requests.post(
            "http://localhost:8000/api/retrieval/search",
            json={"question": question, "top_k": top_k},
            timeout=60,
        )
        retrieval_response.raise_for_status()
        results = retrieval_response.json().get("results", [])
    except Exception as e:
        return {
            "answer": f"Retrieval failed: {e}",
            "grounding_score": 0,
            "risk": "HIGH",
            "supported_claims": [],
            "unsupported_claims": [],
            "sources": [],
            "retrieved_chunks": [],
            "retrieval_confidence": 0,
            "evidence_is_relevant": False,
        }

    best_score = max((float(r.get("score", 0)) for r in results), default=0)

    # Same relevance heuristic used by /api/chat, kept self-contained here.
    stop_words = {
        "what", "when", "where", "which", "who", "whom", "why", "how",
        "is", "are", "was", "were", "do", "does", "did", "can", "could",
        "would", "should", "will", "the", "a", "an", "and", "or", "to",
        "of", "for", "in", "on", "at", "from", "with", "about", "after",
        "before", "under", "this", "that", "these", "those", "it", "its",
        "be", "been", "being", "i", "we", "you", "your", "my", "me",
        "they", "them", "their",
    }
    question_words = {
        w for w in re.findall(r"\b[a-zA-Z]{4,}\b", question.lower())
        if w not in stop_words
    }
    evidence_text = " ".join(r.get("text", "").lower() for r in results)
    evidence_words = set(re.findall(r"\b[a-zA-Z]{4,}\b", evidence_text))
    matched = question_words & evidence_words
    relevance_ratio = len(matched) / len(question_words) if question_words else 1
    evidence_is_relevant = bool(results) and (
        best_score >= 0.60 or (best_score >= 0.45 and relevance_ratio >= 0.30)
    )

    if not results:
        return {
            "answer": "No relevant policy evidence was found.",
            "grounding_score": 0,
            "risk": "HIGH",
            "supported_claims": [],
            "unsupported_claims": [],
            "sources": [],
            "retrieved_chunks": [],
            "retrieval_confidence": 0,
            "evidence_is_relevant": False,
        }

    evidence_parts = [
        f"\nEVIDENCE {i}\nDocument: {r.get('document','')}\nPage: {r.get('page','')}\n\n{r.get('text','')}\n"
        for i, r in enumerate(results, 1)
    ]
    evidence = "\n".join(evidence_parts)

    prompt = f"""
You are InsureMate, an insurance policy assistant.

Answer the USER QUESTION using ONLY the POLICY EVIDENCE.

USER QUESTION:
{question}

POLICY EVIDENCE:
{evidence}

Rules:
- Give the answer directly.
- Use only facts from the evidence.
- Do not invent anything.
- Preserve important numbers and conditions.
- If the evidence contains the answer, answer it.
- Do not say there is insufficient information when the evidence clearly answers the question.

ANSWER:
"""

    try:
        answer = call_ollama(model, prompt)
    except Exception as e:
        answer = f"Model generation failed: {e}"

    sentences = re.split(r'(?<=[.!?])\s+', answer)
    supported_claims, unsupported_claims = [], []
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        sentence_words = set(re.findall(r'\b[a-zA-Z]{4,}\b', sentence.lower()))
        best_overlap = 0
        for r in results:
            evidence_words_r = set(re.findall(r'\b[a-zA-Z]{4,}\b', r.get("text", "").lower()))
            if sentence_words:
                overlap = len(sentence_words & evidence_words_r) / len(sentence_words)
                best_overlap = max(best_overlap, overlap)
        (supported_claims if best_overlap >= 0.35 else unsupported_claims).append(sentence)

    total = len(supported_claims) + len(unsupported_claims)
    grounding_score = round(len(supported_claims) / total * 100) if total else 0
    risk = "LOW" if grounding_score >= 80 else "MEDIUM" if grounding_score >= 60 else "HIGH"

    return {
        "answer": answer,
        "grounding_score": grounding_score,
        "risk": risk,
        "supported_claims": supported_claims,
        "unsupported_claims": unsupported_claims,
        "sources": results,
        "retrieved_chunks": results,
        "retrieval_confidence": best_score,
        "evidence_is_relevant": evidence_is_relevant,
    }


class GuardrailCheckRequest(BaseModel):
    question: str
    model: str = "qwen2.5:1.5b"
    guardrails_enabled: bool = True


@app.post("/api/guardrails/check")
def guardrails_check(request: GuardrailCheckRequest):
    """
    Runs a single question through the pipeline with guardrails either
    ON or OFF, so the frontend can show a direct Before/After comparison
    for the same question.
    """
    question = (request.question or "").strip()

    # If guardrails are OFF, or the input guardrail alone doesn't block,
    # we still need retrieval + a generated answer to compare against.
    input_blocked, input_reason = _guardrails.check_input_guardrails(question)

    if request.guardrails_enabled and input_blocked:
        result = _guardrails.run_guarded_pipeline(
            question=question,
            evidence_is_relevant=False,
            best_score=0,
            raw_answer="",
            guardrails_enabled=True,
        )
        return {**result, "question": question}

    pipeline_result = _core_answer_pipeline(question, request.model)

    guarded = _guardrails.run_guarded_pipeline(
        question=question,
        evidence_is_relevant=pipeline_result["evidence_is_relevant"],
        best_score=pipeline_result["retrieval_confidence"],
        raw_answer=pipeline_result["answer"],
        guardrails_enabled=request.guardrails_enabled,
    )

    return {
        **guarded,
        "question": question,
        "grounding_score": pipeline_result["grounding_score"],
        "retrieval_confidence": pipeline_result["retrieval_confidence"],
    }


@app.get("/api/guardrails/test-suite")
def guardrails_test_suite(model: str = "qwen2.5:1.5b"):
    """
    Runs the built-in guardrail test set with guardrails ON and OFF for
    every question, and reports how effective the guardrails were
    (i.e. did blocking behaviour match what was expected).
    """
    rows = []
    correct_with_guardrails = 0

    for test in _guardrails.GUARDRAIL_TEST_SET:
        question = test["question"]

        # WITHOUT guardrails: run the raw pipeline (this is the
        # "problematic behaviour" baseline).
        without_pipeline = _core_answer_pipeline(question, model)
        without_result = {
            "final_answer": without_pipeline["answer"],
            "blocked": False,
        }

        # WITH guardrails: apply input checks first (cheap), only doing
        # retrieval/generation if the input guardrail does not already
        # block the question.
        input_blocked, input_reason = _guardrails.check_input_guardrails(question)
        if input_blocked:
            with_result = _guardrails.run_guarded_pipeline(
                question=question, evidence_is_relevant=False,
                best_score=0, raw_answer="", guardrails_enabled=True,
            )
        else:
            with_result = _guardrails.run_guarded_pipeline(
                question=question,
                evidence_is_relevant=without_pipeline["evidence_is_relevant"],
                best_score=without_pipeline["retrieval_confidence"],
                raw_answer=without_pipeline["answer"],
                guardrails_enabled=True,
            )

        matched_expectation = with_result["blocked"] == test["expect_blocked"]
        if matched_expectation:
            correct_with_guardrails += 1

        rows.append({
            "id": test["id"],
            "question": question if len(question) < 200 else question[:200] + "...",
            "category": test["category"],
            "expect_blocked": test["expect_blocked"],
            "without_guardrail_answer": without_result["final_answer"][:300],
            "with_guardrail_result": with_result["final_answer"][:300],
            "with_guardrail_blocked": with_result["blocked"],
            "triggered": with_result["triggered"],
            "matched_expectation": matched_expectation,
        })

    return {
        "model": model,
        "total": len(_guardrails.GUARDRAIL_TEST_SET),
        "correct": correct_with_guardrails,
        "effectiveness_percent": round(
            correct_with_guardrails / len(_guardrails.GUARDRAIL_TEST_SET) * 100, 1
        ),
        "results": rows,
    }


@app.get("/api/output-tests/run")
def output_tests_run():
    """
    Runs the AI Output Testing suite (Week 5, Part B): each test case
    calls the live pipeline and checks the output against a defined
    pass/fail condition (relevance, groundedness, unsupported claims,
    format, appropriate answering vs. appropriate refusal).
    """
    def _hallucination_fn(question, model, top_k):
        return _core_answer_pipeline(question, model, top_k)

    return _output_tests.run_output_tests(_hallucination_fn)
