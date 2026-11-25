from agent_metrics import check_answer, check_tools, score_case, summarize


def test_check_tools():
    assert check_tools(["get_member", "search_faq"], ["get_member"], [])
    assert not check_tools(["search_faq"], ["get_member"], [])
    assert not check_tools(["create_support_ticket"], [], ["create_support_ticket"])


def test_check_answer_all_and_any():
    assert check_answer("Fee is 149 DKK", keywords=["149"])
    assert not check_answer("Fee is 149 DKK", keywords=["149", "student"])
    assert check_answer("I can't help with that request.", any_keywords=["can't help", "cannot help"])
    assert not check_answer("Sure!", any_keywords=["can't help"])


def test_score_case_requires_both():
    case = {"expected_tools": ["search_faq"], "keywords": ["149"]}
    assert score_case(case, ["search_faq"], "149 DKK")["passed"]
    assert not score_case(case, [], "149 DKK")["passed"]
    assert not score_case(case, ["search_faq"], "unknown")["passed"]


def test_summarize():
    s = summarize([{"tools_ok": True, "answer_ok": True, "passed": True},
                   {"tools_ok": True, "answer_ok": False, "passed": False}])
    assert s["tool_selection"] == 1.0 and s["answer_quality"] == 0.5 and s["pass_rate"] == 0.5
    assert summarize([])["total"] == 0
