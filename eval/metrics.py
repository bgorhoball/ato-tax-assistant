"""
RAG evaluation metrics for ato-tax-assistant.

All functions are pure (no LLM calls, no I/O) and operate on the raw
AskResult fields so they are cheap and deterministic to re-run.

Metric summary
--------------
answer_accuracy        Keyword-match proxy: fraction of expected_answer_contains
                       substrings found in the answer (case-insensitive).
                       For should_decline cases: 1.0 if decline language present,
                       0.0 otherwise.

retrieval_recall       Fraction of ground_truth_pages surfaced in the retrieved
                       document set.  None if no ground_truth_pages defined.

citation_faithfulness  Fraction of page numbers cited in the answer text that
                       are also present in the retrieved document set.
                       None if the answer makes no page citations.
                       NOTE: This is a shallow proxy — the future Layer-3
                       implementation should do verbatim quote matching.

hallucination_penalty  1.0 if none of expected_answer_excludes appear in the
                       answer, 0.0 if any do.  None if list is empty.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class CaseResult:
    """Evaluation result for a single test case."""

    case_id: str
    category: str
    question: str
    answer: str
    retrieved_pages: list[int]

    answer_accuracy: float | None  # None = not evaluable (no criteria defined)
    retrieval_recall: float | None
    citation_faithfulness: float | None
    hallucination_penalty: float | None  # 1.0 clean | 0.0 penalised | None N/A

    latency_s: float = 0.0
    error: str | None = None


# Phrases the model should use when it cannot find the answer in context.
_DECLINE_SIGNALS: tuple[str, ...] = (
    "not stated",
    "not provided",
    "not found",
    "not mentioned",
    "not specified",
    "not covered",
    "not included",
    "not explicitly",
    "not in the context",
    "not available in",
    "don't know",
    "do not know",
    "cannot find",
    "unable to find",
    "no information",
    "does not mention",
    "does not state",
    "does not provide",
    "does not include",
    "the context does not",
    "the document does not",
    "the provided context",
    "based on the context",  # often precedes "I cannot determine"
    "i cannot determine",
    "i'm unable",
    "i am unable",
)


def compute_answer_accuracy(answer: str, case) -> float | None:
    """
    For should_decline cases: 1.0 if a decline-signal phrase is in the answer.
    For regular cases: fraction of expected_answer_contains keywords present.
    Returns None if no evaluation criteria is defined.
    """
    a = answer.lower()

    if case.should_decline:
        return 1.0 if any(sig in a for sig in _DECLINE_SIGNALS) else 0.0

    if not case.expected_answer_contains:
        return None

    hits = sum(1 for kw in case.expected_answer_contains if kw.lower() in a)
    return round(hits / len(case.expected_answer_contains), 3)


def compute_retrieval_recall(retrieved_pages: list[int], case) -> float | None:
    """
    Fraction of ground_truth_pages that appear in the retrieved document set.
    Returns None if ground_truth_pages is empty (no verified ground truth).
    """
    if not case.ground_truth_pages:
        return None

    hits = sum(1 for p in case.ground_truth_pages if p in retrieved_pages)
    return round(hits / len(case.ground_truth_pages), 3)


def compute_citation_faithfulness(answer: str, retrieved_pages: list[int]) -> float | None:
    """
    Extracts page numbers cited in the answer text and checks whether all of
    them are present in the retrieved document set.

    Matches: "Page 51", "page 51", "p. 51", "(p.51)", "pages 51-53"
    Returns None if the answer contains no page citations.

    Upgrade path (Layer 3): replace regex page extraction with verbatim quote
    lookup against source chunk text for true citation faithfulness.
    """
    cited_nums = re.findall(
        r'\bp(?:age|gs?|\.)\s*(\d+)',
        answer,
        flags=re.IGNORECASE,
    )
    if not cited_nums:
        return None

    cited = {int(n) for n in cited_nums}
    retrieved = set(retrieved_pages)
    faithful = cited & retrieved  # cited pages that are actually retrieved

    return round(len(faithful) / len(cited), 3)


def compute_hallucination_penalty(answer: str, case) -> float | None:
    """
    1.0 if none of expected_answer_excludes appear in the answer (clean).
    0.0 if any forbidden string is present (fabricated fact detected).
    None if expected_answer_excludes is empty.
    """
    if not case.expected_answer_excludes:
        return None

    a = answer.lower()
    for excluded in case.expected_answer_excludes:
        if excluded.lower() in a:
            return 0.0
    return 1.0


def evaluate(answer: str, retrieved_pages: list[int], case, latency_s: float = 0.0) -> CaseResult:
    """Compute all metrics for a single case and return a CaseResult."""
    return CaseResult(
        case_id=case.id,
        category=case.category,
        question=case.question,
        answer=answer,
        retrieved_pages=retrieved_pages,
        answer_accuracy=compute_answer_accuracy(answer, case),
        retrieval_recall=compute_retrieval_recall(retrieved_pages, case),
        citation_faithfulness=compute_citation_faithfulness(answer, retrieved_pages),
        hallucination_penalty=compute_hallucination_penalty(answer, case),
        latency_s=latency_s,
    )
