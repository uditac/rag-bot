"""Run the evaluation set and print retrieval and answer quality.

  python src/evaluate.py
  python src/evaluate.py --chunk-size 400 --k 6
"""
import argparse
import json
from pathlib import Path

from metrics import score_case, summarize
from rag import RagBot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--questions", default="eval/questions.json")
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--overlap", type=int, default=100)
    parser.add_argument("--k", type=int, default=4)
    args = parser.parse_args()

    cases = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    bot = RagBot(db_path="db_eval", chunk_size=args.chunk_size,
                 overlap=args.overlap, k=args.k)
    n_chunks = bot.index(args.data)
    print(f"chunk_size={args.chunk_size} overlap={args.overlap} k={args.k} "
          f"-> {n_chunks} chunks\n")

    scored, rows = [], []
    for case in cases:
        res = bot.answer(case["question"])
        s = score_case(case, [p["text"] for p in res["passages"]], res["answer"])
        scored.append(s)
        rows.append({"question": case["question"], "answer": res["answer"], **s})
        mark = "PASS" if s["passed"] else "FAIL"
        print(f"[{mark}] {case['question']}")

    summary = summarize(scored)
    print("\nSummary")
    for key, value in summary.items():
        shown = f"{value:.0%}" if isinstance(value, float) else value
        print(f"  {key}: {shown}")

    Path("eval/last_results.json").write_text(
        json.dumps({"settings": vars(args), "summary": summary, "cases": rows}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
