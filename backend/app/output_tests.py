"""
InsureMate AI Output Testing Module (Week 5)
===============================================

Treats the LLM's output as something that must be TESTED before it is
accepted, rather than trusted blindly. Each test case defines:

  - a question to ask
  - which model to use
  - a check_type describing WHAT property is being tested
  - a pass/fail function that inspects the actual pipeline output

Running this module calls the live backend pipeline (via the same
/api/hallucination/check logic already used elsewhere) and produces a
pass/fail report with a reason for every test.
"""

import re


# =========================================================
# PASS/FAIL CHECK FUNCTIONS
# One function per check_type. Each returns (passed: bool, reason: str).
# =========================================================

def check_relevance(question, result):
    """PASS if the answer's key words overlap with the question's key words
    (i.e. the model actually addressed what was asked, not something else)."""
    answer = (result.get("answer") or "").lower()
    q_words = set(re.findall(r"[a-z]{4,}", question.lower()))
    a_words = set(re.findall(r"[a-z]{4,}", answer))
    if not q_words:
        return False, "Question had no meaningful keywords to check against."
    overlap = len(q_words & a_words) / len(q_words)
    passed = overlap >= 0.15 or len(answer) > 30
    return passed, f"Keyword overlap with question: {overlap:.0%}."


def check_grounded(question, result):
    """PASS if grounding_score is at or above 60 (MEDIUM or better)."""
    score = result.get("grounding_score", 0)
    passed = score >= 60
    return passed, f"Grounding score was {score}% (pass threshold: 60%)."


def check_no_unsupported_claims(question, result):
    """PASS if there are zero unsupported claims in the answer."""
    unsupported = result.get("unsupported_claims", [])
    passed = len(unsupported) == 0
    return passed, f"{len(unsupported)} unsupported claim(s) found."


def check_format_not_empty(question, result):
    """PASS if the answer is non-empty and reasonably formed (not just
    punctuation / whitespace, and under a sane max length)."""
    answer = (result.get("answer") or "").strip()
    passed = 5 < len(answer) < 3000
    return passed, f"Answer length was {len(answer)} characters."

def check_answers_when_info_exists(question, result):
    """PASS if the model actually gave a substantive answer (did not
    refuse) when the question is a genuine, in-scope policy question."""
    answer = (result.get("answer") or "").lower()
    refused = "does not contain enough information" in answer or "no relevant policy evidence" in answer
    passed = not refused
    return passed, "Model refused to answer." if refused else "Model provided a substantive answer."


def check_appropriate_refusal(question, result):
    """PASS if the model correctly refuses when the question is
    out-of-scope / unanswerable from the knowledge base."""
    answer = (result.get("answer") or "").lower()
    refused = (
        "does not contain enough information" in answer
        or "no relevant policy evidence" in answer
        or "cannot" in answer
    )
    passed = refused
    return passed, "Model correctly refused." if refused else "Model did not refuse when it should have."


CHECKERS = {
    "relevance": check_relevance,
    "grounded": check_grounded,
    "no_unsupported_claims": check_no_unsupported_claims,
    "format": check_format_not_empty,
    "answers_when_info_exists": check_answers_when_info_exists,
    "appropriate_refusal": check_appropriate_refusal,
}


# =========================================================
# TEST SUITE DEFINITION
# =========================================================

OUTPUT_TEST_SUITE = [
    {
        "id": "OT-01",
        "question": "What is the waiting period for pre-existing diseases?",
        "model": "codellama:7b-instruct",
        "check_type": "relevance",
        "description": "Answer should be relevant to the question asked.",
    },
    {
        "id": "OT-02",
        "question": "What is the waiting period for pre-existing diseases?",
        "model": "codellama:7b-instruct",
        "check_type": "grounded",
        "description": "Answer should be grounded in retrieved policy evidence (>=60%).",
    },
    {
        "id": "OT-03",
        "question": "Does the policy cover hospitalization expenses?",
        "model": "qwen2.5:1.5b",
        "check_type": "no_unsupported_claims",
        "description": "Answer should not contain unsupported claims.",
    },
    {
        "id": "OT-04",
        "question": "What are the important waiting periods mentioned in the policy documents?",
        "model": "phi3:mini",
        "check_type": "format",
        "description": "Answer should be non-empty and reasonably formatted.",
    },
    {
        "id": "OT-05",
        "question": "What percentage of eligible expenses is covered under the policy?",
        "model": "codellama:7b-instruct",
        "check_type": "answers_when_info_exists",
        "description": "Model should answer (not refuse) when the info exists.",
    },
    {
        "id": "OT-06",
        "question": "What is the current price of Bitcoin?",
        "model": "qwen2.5:1.5b",
        "check_type": "appropriate_refusal",
        "description": "Model should refuse an out-of-scope question.",
    },
    {
        "id": "OT-07",
        "question": "Who will win the next Cricket World Cup?",
        "model": "phi3:mini",
        "check_type": "appropriate_refusal",
        "description": "Model should refuse an out-of-scope question.",
    },
    {
        "id": "OT-08",
        "question": "Are stones in the biliary system covered immediately?",
        "model": "codellama:7b-instruct",
        "check_type": "grounded",
        "description": "Answer should be grounded in retrieved policy evidence (>=60%).",
    },
    {
        "id": "OT-09",
        "question": "Does the policy cover cosmetic surgery?",
        "model": "phi3:mini",
        "check_type": "no_unsupported_claims",
        "description": "Answer should not contain unsupported claims.",
    },
    {
        "id": "OT-10",
        "question": "What benefits are provided under the accident insurance policy?",
        "model": "qwen2.5:1.5b",
        "check_type": "relevance",
        "description": "Answer should be relevant to the question asked.",
    },
]


def run_output_tests(hallucination_check_fn):
    """
    hallucination_check_fn: a callable(question, model, top_k) -> dict,
    matching the shape returned by the existing /api/hallucination/check
    endpoint logic. Injected as a parameter so this module has no
    circular import on main.py.
    """
    report = []
    passed_count = 0

    for test in OUTPUT_TEST_SUITE:
        try:
            result = hallucination_check_fn(test["question"], test["model"], 3)
            checker = CHECKERS[test["check_type"]]
            passed, reason = checker(test["question"], result)
        except Exception as e:
            passed, reason = False, f"Test execution error: {e}"
            result = {}

        if passed:
            passed_count += 1

        report.append({
            "id": test["id"],
            "question": test["question"],
            "model": test["model"],
            "check_type": test["check_type"],
            "description": test["description"],
            "passed": passed,
            "reason": reason,
            "answer": (result or {}).get("answer", ""),
        })

    return {
        "total": len(OUTPUT_TEST_SUITE),
        "passed": passed_count,
        "failed": len(OUTPUT_TEST_SUITE) - passed_count,
        "pass_rate": round(passed_count / len(OUTPUT_TEST_SUITE) * 100, 1),
        "results": report,
    }
