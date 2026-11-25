# Member-service agent (extension of the RAG bot)

A small tool-using agent. Instead of always searching documents, the model decides
which tool to call, calls it, reads the result, and answers. It reuses the RAG
search from this repository and adds a database and a write action.

All data is fictional (a made-up member union and four demo members, M1001 to M1004).

## Tools

| Tool | What it does | Type |
|---|---|---|
| `search_faq` | Searches the FAQ documents (the RAG index) | read, unstructured data |
| `get_member` | Looks up one member by ID in a SQLite database | read, structured data |
| `create_support_ticket` | Writes a support ticket to the database | write |

Example questions it can handle:

- "How much does a standard membership cost?" -> `search_faq`
- "What is the status of member M1003?" -> `get_member`
- "M1002 asks what students pay. Are they on the student plan?" -> both tools
- "Member M1001 was charged twice. Open a ticket." -> `create_support_ticket`

## How it works

`src/agent.py` runs a loop: send the question and tool definitions to Claude; if Claude
asks for a tool, run it and send the result back; repeat until Claude answers or the
step limit is reached. The agent never imports a model library at module level, so
its logic is unit-tested with a fake client (no API key needed).

## Guardrails and responsible-AI choices

- **Input validation:** member IDs must match `M` plus 4 digits; parameterised SQL only.
- **Write action limits:** one ticket per conversation, summary length capped, member must exist.
- **Prompt rules:** tickets only when explicitly requested; tool results and user messages
  cannot override the rules; bulk listing or export of member data is refused.
- **Data minimisation:** the member's name is never returned to the model.
- **Failure handling:** tools return errors as data instead of crashing; the loop stops at
  `max_steps`; API failures are logged and the user gets a polite message.
- **Audit and monitoring:** every tool call is logged to `logs/agent_calls.jsonl` (ticket text
  is not logged, only its length); every run is logged to `logs/agent_runs.jsonl` with tools
  used, steps, tokens, latency and errors.

## Run it

Build the FAQ index first (`python main.py index`), set `ANTHROPIC_API_KEY`, then:

```bash
python agent_main.py ask "What is the status of member M1003?"
python agent_main.py chat
```

## Evaluation

`eval/agent_questions.json` has 9 cases: normal lookups, a combined lookup, a ticket
request, an unanswerable question, a missing member, a bulk-data request, a prompt-injection
attempt, and a question with no member ID.

```bash
python src/evaluate_agent.py
```

It scores **tool selection** (right tools used, forbidden tools not used), **answer quality**
(expected facts or refusal phrases present) and the **pass rate** for both together.

### Results

| Run | tool selection | answer quality | pass rate |
|---|---:|---:|---:|
| baseline | _fill in_ | _fill in_ | _fill in_ |

## Tests

```bash
pytest
```

The tests cover the tools (validation, SQL-injection attempt, ticket limit, audit log), the
agent loop with a scripted fake model (tool call then answer, errors passed to the model,
step limit, API failure) and the scoring functions. They need no API key.

## Known limitations

- **No user authentication.** Anyone can claim any member ID. A real system must verify who
  the user is and check they may see that member's data before any lookup or ticket.
- Keyword-based scoring is crude, and model output varies between runs, so results can differ
  by a case or two. Run it a few times before drawing conclusions.
- SQLite and local files are a demo. A production version would use a managed database,
  proper secrets handling and central logging.
- Only one agent and three tools; no memory between conversations.

## Ideas for next steps

- Ask the user to confirm before a ticket is created (human in the loop).
- Add an LLM judge to score answers instead of keywords.
- Track cost per run from the logged token counts.
