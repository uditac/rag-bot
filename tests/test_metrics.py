from metrics import is_refusal, score_case, summarize


def test_refusal_detection():
    assert is_refusal("I don't know based on the documents.")
    assert not is_refusal("The fee is 149 DKK.")


def test_answerable_case_passes_when_retrieved_and_answered():
    case = {"question": "q", "keywords": ["149"]}
    s = score_case(case, ["The fee is 149 DKK"], "It costs 149 DKK [faq.txt]")
    assert s == {"retrieval_hit": True, "passed": True}


def test_answerable_case_fails_when_retrieval_misses():
    case = {"question": "q", "keywords": ["149"]}
    s = score_case(case, ["unrelated text"], "I don't know based on the documents.")
    assert s["retrieval_hit"] is False and s["passed"] is False


def test_unanswerable_case_requires_refusal():
    case = {"question": "q", "expect_refusal": True}
    assert score_case(case, [], "I don't know based on the documents.")["passed"]
    assert not score_case(case, [], "It costs 99 DKK.")["passed"]


def test_summary_rates():
    scored = [
        {"retrieval_hit": True, "passed": True},
        {"retrieval_hit": True, "passed": False},
        {"retrieval_hit": None, "passed": True},
    ]
    s = summarize(scored)
    assert s["retrieval_hit_rate"] == 1.0
    assert s["answer_accuracy"] == 0.5
    assert s["correct_refusal_rate"] == 1.0
