"""Command line entry point.

  python main.py index            build the vector index from data/
  python main.py ask "question"   ask one question
  python main.py chat             interactive loop
"""
import argparse
import sys

sys.path.insert(0, "src")
from rag import RagBot  # noqa: E402


def show(result: dict) -> None:
    print(f"\n{result['answer']}")
    print(f"\nSources: {', '.join(result['sources'])}  ({result['latency_s']}s)\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["index", "ask", "chat"])
    parser.add_argument("question", nargs="?")
    parser.add_argument("--data", default="data")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=800)
    args = parser.parse_args()

    bot = RagBot(chunk_size=args.chunk_size, k=args.k)

    if args.command == "index":
        print(f"Indexed {bot.index(args.data)} chunks.")
    elif args.command == "ask":
        if not args.question:
            parser.error("ask needs a question")
        show(bot.answer(args.question))
    else:
        print("Ask a question (empty line to quit).")
        while True:
            q = input("> ").strip()
            if not q:
                break
            show(bot.answer(q))


if __name__ == "__main__":
    main()
