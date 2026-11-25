"""Run the agent evaluation set.

  python src/evaluate_agent.py
"""
import argparse
import json
import os
from pathlib import Path

from agent import Agent
from agent_metrics import score_case, summarize
from rag import RagBot
from tools import Toolbox


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--questions", default="eval/agent_questions.json")
    args = parser.parse_args()

    cases = json.loads(Path(args.questions).read_text(encoding="utf-8"))

    bot = RagBot(db_path="db_eval")
    bot.index(args.data)

    eval_db = Path("db_agent_eval/members.sqlite")
    if eval_db.exists():
        eval_db.unlink()  # start from clean seed data so tickets do not pile up
    toolbox = Toolbox(str(eval_db), faq_search=bot.retrieve,
                      audit_path="logs/agent_eval_calls.jsonl")
    agent = Agent(toolbox, log_path="logs/agent_eval_runs.jsonl")

    scored, rows = [], []
    for case in cases:
        res = agent.run(case["question"])
        called = [c["name"] for c in res["tool_calls"]]
        s = score_case(case, called, res["answer"])
        scored.append(s)
        rows.append({"question": case["question"], "tools": called,
                     "answer": res["answer"], **s})
        mark = "PASS" if s["passed"] else "FAIL"
        print(f"[{mark}] {case['question']}\n        tools: {called or 'none'}")

    summary = summarize(scored)
    print("\nSummary")
    for key, value in summary.items():
        shown = f"{value:.0%}" if isinstance(value, float) else value
        print(f"  {key}: {shown}")

    Path("eval/last_agent_results.json").write_text(
        json.dumps({"summary": summary, "cases": rows}, indent=2), encoding="utf-8")
    toolbox.close()


if __name__ == "__main__":
    main()
