# Small RAG bot with evaluation

A compact retrieval-augmented generation (RAG) bot that answers questions from
your own documents, cites its sources, refuses to guess, and ships with a small
evaluation set so every change can be measured.

The sample data is a fictional member-union FAQ (`data/faq.txt`). Replace it with
your own `.txt` or `.md` files.

## How it works

1. **Chunk**: documents are split into overlapping chunks (`src/chunking.py`).
2. **Embed + store**: chunks are embedded locally with `all-MiniLM-L6-v2` and stored in ChromaDB.
3. **Retrieve**: the question is embedded and the `k` closest chunks are fetched.
4. **Answer**: Claude answers using only those chunks, cites the source file, and
   replies "I don't know based on the documents." when the answer is not there.
5. **Monitor**: every query is logged to `logs/queries.jsonl` (latency, token counts,
   sources, top retrieval distance, errors).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your-key        # Windows PowerShell: $env:ANTHROPIC_API_KEY="your-key"
```

The model defaults to `claude-sonnet-5-5`. Override it with `RAG_MODEL`.

## Usage

```bash
python main.py index                       # build the index from data/
python main.py ask "How much does a standard membership cost?"
python main.py chat                        # interactive loop
```

## Evaluation

`eval/questions.json` holds 12 questions: 10 with expected facts and 2 the
documents cannot answer (the bot must refuse, not guess).

```bash
python src/evaluate.py
python src/evaluate.py --chunk-size 400 --k 6    # compare settings
```

It reports:

- **retrieval_hit_rate**: were the needed facts in the retrieved chunks?
- **answer_accuracy**: did the answer contain the expected facts?
- **correct_refusal_rate**: did the bot refuse unanswerable questions?

Splitting retrieval from answer quality tells you whether a failure is a search
problem (fix chunking, `k`, embeddings) or a generation problem (fix the prompt).

### Results

Run the evaluation and record your numbers here. Example format:

| chunk_size | k | retrieval hit | accuracy | correct refusals |
|-----------:|--:|--------------:|---------:|-----------------:|
| 800 | 4 | _fill in_ | _fill in_ | _fill in_ |
| 400 | 6 | _fill in_ | _fill in_ | _fill in_ |

## Tests and CI

```bash
pip install -r requirements-dev.txt
pytest
```

Tests cover chunking and scoring, need no API key or model download, and run in
GitHub Actions on every push (`.github/workflows/ci.yml`).

## Known limitations

- Keyword matching in the evaluation is crude. A correct answer phrased differently
  can be marked wrong. A stronger version would use an LLM judge.
- Fixed-size chunking can cut a fact in half. Try heading- or paragraph-aware chunking and re-run the evaluation.
- Refusal relies on the prompt. A distance threshold on retrieval would add a second safeguard.
- Only `.txt` and `.md` files are loaded. PDFs need a text-extraction step.
- Do not put sensitive or personal data in `data/` unless you have checked how it is handled.

## Ideas for next steps

- Hybrid search (keyword + vector) and a reranker
- Turn it into an agent that decides when to search and can call a second tool
- Estimate cost per query from the logged token counts
- Add a small web UI
