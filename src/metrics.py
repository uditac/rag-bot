"""Scoring helpers for the evaluation set. Pure Python, no model calls."""

REFUSAL_MARKERS = ("don't know", "do not know", "cannot find", "not in the documents")


def retrieval_hit(retrieved_texts: list[str], keywords: list[str]) -> bool:
    """True if every keyword appears somewhere in the retrieved chunks."""
    blob = " ".join(retrieved_texts).lower()
    return all(k.lower() in blob for k in keywords)


def answer_correct(answer: str, keywords: list[str]) -> bool:
    """True if every expected keyword appears in the answer."""
    low = answer.lower()
    return all(k.lower() in low for k in keywords)


def is_refusal(answer: str) -> bool:
    low = answer.lower()
    return any(m in low for m in REFUSAL_MARKERS)


def score_case(case: dict, retrieved_texts: list[str], answer: str) -> dict:
    """Score one eval case.

    Answerable cases need the right facts retrieved AND in the answer.
    Unanswerable cases (expect_refusal) must be refused, not guessed.
    """
    if case.get("expect_refusal"):
        return {"retrieval_hit": None, "passed": is_refusal(answer)}
    keywords = case["keywords"]
    return {
        "retrieval_hit": retrieval_hit(retrieved_texts, keywords),
        "passed": answer_correct(answer, keywords),
    }


def summarize(scored: list[dict]) -> dict:
    answerable = [s for s in scored if s["retrieval_hit"] is not None]
    refusals = [s for s in scored if s["retrieval_hit"] is None]
    return {
        "total": len(scored),
        "retrieval_hit_rate": (
            sum(s["retrieval_hit"] for s in answerable) / len(answerable)
            if answerable else None
        ),
        "answer_accuracy": (
            sum(s["passed"] for s in answerable) / len(answerable)
            if answerable else None
        ),
        "correct_refusal_rate": (
            sum(s["passed"] for s in refusals) / len(refusals)
            if refusals else None
        ),
    }
