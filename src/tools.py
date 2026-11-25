"""Tools the agent can call: FAQ search, a member database, and a ticket writer.

Design rules:
- Tools never raise to the model. Problems come back as {"error": "..."} so the
  agent can explain them.
- Inputs are validated before they touch the database.
- Only the fields the agent needs are returned (the member's name is never exposed).
- Every call is written to an audit log.
"""
import json
import re
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

MEMBER_ID = re.compile(r"^M\d{4}$")
MAX_SUMMARY_CHARS = 300

SCHEMA = """
CREATE TABLE IF NOT EXISTS members (
    member_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    plan TEXT NOT NULL,
    fee_dkk REAL NOT NULL,
    status TEXT NOT NULL,
    cancellation_effective TEXT
);
CREATE TABLE IF NOT EXISTS tickets (
    ticket_id INTEGER PRIMARY KEY AUTOINCREMENT,
    member_id TEXT NOT NULL,
    summary TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""

# Entirely fictional demo data.
SEED_MEMBERS = [
    ("M1001", "Anna Madsen", "standard", 149.0, "active", None),
    ("M1002", "Jonas Berg", "student", 49.0, "active", None),
    ("M1003", "Sara Holm", "standard", 149.0, "cancelling", "2026-12-31"),
    ("M1004", "Mikkel Dahl", "parental_leave", 74.5, "active", None),
]

TOOL_SCHEMAS = [
    {
        "name": "search_faq",
        "description": (
            "Search the union's FAQ documents for policies and general information "
            "(fees, cancelling, legal help, opening hours, complaints, data protection). "
            "Use for any general question about how the union works."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "What to look up"}},
            "required": ["query"],
        },
    },
    {
        "name": "get_member",
        "description": (
            "Look up ONE member by member ID (format: M followed by 4 digits, e.g. M1001). "
            "Returns plan, monthly fee in DKK, status and cancellation date. "
            "Only call when the user has given a member ID."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"member_id": {"type": "string", "description": "e.g. M1001"}},
            "required": ["member_id"],
        },
    },
    {
        "name": "create_support_ticket",
        "description": (
            "Open a support ticket for a member. Only call when the user explicitly "
            "asks to open a ticket or file a complaint. At most one per conversation."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "member_id": {"type": "string", "description": "e.g. M1001"},
                "summary": {"type": "string", "description": "Short description of the issue"},
            },
            "required": ["member_id", "summary"],
        },
    },
]


def connect(db_path: str) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    if conn.execute("SELECT COUNT(*) FROM members").fetchone()[0] == 0:
        conn.executemany("INSERT INTO members VALUES (?,?,?,?,?,?)", SEED_MEMBERS)
        conn.commit()
    return conn


class Toolbox:
    def __init__(self, db_path, faq_search, audit_path="logs/agent_calls.jsonl",
                 max_tickets_per_run=1):
        self.conn = connect(db_path)
        self.faq_search = faq_search  # callable: query -> list of {"source", "text", ...}
        self.audit_path = Path(audit_path)
        self.max_tickets = max_tickets_per_run
        self.tickets_this_run = 0

    def new_run(self) -> None:
        self.tickets_this_run = 0

    def close(self) -> None:
        self.conn.close()

    # ---- the tools ------------------------------------------------------
    def search_faq(self, query):
        if not isinstance(query, str) or not query.strip():
            return {"error": "query must be a non-empty string"}
        hits = self.faq_search(query)[:3]
        return {"results": [{"source": h["source"], "text": h["text"]} for h in hits]}

    def get_member(self, member_id):
        if not isinstance(member_id, str) or not MEMBER_ID.match(member_id):
            return {"error": "invalid member id (expected M followed by 4 digits)"}
        row = self.conn.execute(
            "SELECT plan, fee_dkk, status, cancellation_effective "
            "FROM members WHERE member_id = ?",
            (member_id,),
        ).fetchone()
        if row is None:
            return {"error": "member not found"}
        return {"member_id": member_id, **dict(row)}

    def create_support_ticket(self, member_id, summary):
        if not isinstance(member_id, str) or not MEMBER_ID.match(member_id):
            return {"error": "invalid member id (expected M followed by 4 digits)"}
        if not isinstance(summary, str) or not summary.strip():
            return {"error": "summary must be a non-empty string"}
        if len(summary) > MAX_SUMMARY_CHARS:
            return {"error": f"summary too long (max {MAX_SUMMARY_CHARS} characters)"}
        if self.tickets_this_run >= self.max_tickets:
            return {"error": "ticket limit reached for this conversation"}
        exists = self.conn.execute(
            "SELECT 1 FROM members WHERE member_id = ?", (member_id,)
        ).fetchone()
        if exists is None:
            return {"error": "member not found"}
        cur = self.conn.execute(
            "INSERT INTO tickets (member_id, summary, created_at) VALUES (?,?,?)",
            (member_id, summary.strip(), datetime.now(timezone.utc).isoformat()),
        )
        self.conn.commit()
        self.tickets_this_run += 1
        return {"ticket_id": f"T-{cur.lastrowid}", "status": "created"}

    # ---- dispatch + audit -----------------------------------------------
    def dispatch(self, name: str, args: dict) -> dict:
        start = time.time()
        handlers = {
            "search_faq": self.search_faq,
            "get_member": self.get_member,
            "create_support_ticket": self.create_support_ticket,
        }
        try:
            if name not in handlers:
                out = {"error": f"unknown tool: {name}"}
            else:
                out = handlers[name](**(args or {}))
        except TypeError:
            out = {"error": "invalid arguments"}
        except Exception as exc:  # never let a tool crash the agent loop
            out = {"error": f"tool failed: {type(exc).__name__}"}
        self._audit(name, args, out, start)
        return out

    def _audit(self, name, args, out, start) -> None:
        logged = dict(args or {})
        if "summary" in logged:  # free text may contain personal data
            logged["summary"] = f"<{len(str(logged['summary']))} chars>"
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "tool": name,
            "args": logged,
            "ok": "error" not in out,
            "latency_s": round(time.time() - start, 4),
        }
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        with self.audit_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
