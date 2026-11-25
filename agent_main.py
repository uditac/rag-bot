"""Command line entry point for the member-service agent.

  python agent_main.py ask "What is the status of member M1003?"
  python agent_main.py chat

Run `python main.py index` first so the FAQ search has an index.
"""
import argparse
import sys

sys.path.insert(0, "src")
from agent import Agent  # noqa: E402
from rag import RagBot  # noqa: E402
from tools import Toolbox  # noqa: E402


def show(res: dict) -> None:
    print(f"\n{res['answer']}")
    used = ", ".join(c["name"] for c in res["tool_calls"]) or "none"
    print(f"\nTools used: {used}  |  steps: {res['steps']}  |  {res['latency_s']}s\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["ask", "chat"])
    parser.add_argument("question", nargs="?")
    args = parser.parse_args()

    bot = RagBot()
    toolbox = Toolbox("db_agent/members.sqlite", faq_search=bot.retrieve)
    agent = Agent(toolbox)

    if args.command == "ask":
        if not args.question:
            parser.error("ask needs a question")
        show(agent.run(args.question))
    else:
        print("Ask a question (empty line to quit). Demo member IDs: M1001 to M1004.")
        while True:
            q = input("> ").strip()
            if not q:
                break
            show(agent.run(q))


if __name__ == "__main__":
    main()
