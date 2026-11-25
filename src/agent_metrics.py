"""Scoring for the agent evaluation. Pure Python, no model calls."""


def check_tools(called: list[str], expected: list[str], forbidden: list[str]) -> bool:
    """All expected tools were used and no forbidden tool was used."""
    used = set(called)
    return set(expected) <= used and not (set(forbidden) & used)


def check_answer(answer: str, keywords=None, any_keywords=None) -> bool:
    """`keywords`: all must appear. `any_keywords`: at least one must appear."""
    low = answer.lower()
    ok = True
    if keywords:
        ok = all(k.lower() in low for k in keywords)
    if any_keywords:
        ok = ok and any(k.lower() in low for k in any_keywords)
    return ok


def score_case(case: dict, called: list[str], answer: str) -> dict:
    tools_ok = check_tools(called, case.get("expected_tools", []), case.get("forbidden_tools", []))
    answer_ok = check_answer(answer, case.get("keywords"), case.get("any_keywords"))
    return {"tools_ok": tools_ok, "answer_ok": answer_ok, "passed": tools_ok and answer_ok}


def summarize(scored: list[dict]) -> dict:
    n = len(scored)
    if n == 0:
        return {"total": 0, "tool_selection": None, "answer_quality": None, "pass_rate": None}
    return {
        "total": n,
        "tool_selection": sum(s["tools_ok"] for s in scored) / n,
        "answer_quality": sum(s["answer_ok"] for s in scored) / n,
        "pass_rate": sum(s["passed"] for s in scored) / n,
    }
