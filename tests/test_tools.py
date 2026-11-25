import json
from tools import Toolbox


def make_box(tmp_path, hits=None):
    hits = hits if hits is not None else [{"source": "faq.txt", "text": "Fee is 149 DKK", "distance": 0.1}]
    return Toolbox(str(tmp_path / "m.sqlite"), faq_search=lambda q: hits,
                   audit_path=str(tmp_path / "audit.jsonl"))


def test_get_member_found_hides_name(tmp_path):
    out = make_box(tmp_path).dispatch("get_member", {"member_id": "M1003"})
    assert out["status"] == "cancelling"
    assert out["cancellation_effective"] == "2026-12-31"
    assert "name" not in out


def test_get_member_not_found(tmp_path):
    out = make_box(tmp_path).dispatch("get_member", {"member_id": "M9999"})
    assert out == {"error": "member not found"}


def test_get_member_rejects_bad_id_and_sql_injection(tmp_path):
    box = make_box(tmp_path)
    assert "error" in box.dispatch("get_member", {"member_id": "1001"})
    assert "error" in box.dispatch("get_member", {"member_id": "M1001' OR '1'='1"})


def test_ticket_created_and_limited_per_run(tmp_path):
    box = make_box(tmp_path)
    first = box.dispatch("create_support_ticket", {"member_id": "M1001", "summary": "Double charge"})
    assert first["ticket_id"] == "T-1"
    second = box.dispatch("create_support_ticket", {"member_id": "M1001", "summary": "Again"})
    assert "limit" in second["error"]
    box.new_run()
    third = box.dispatch("create_support_ticket", {"member_id": "M1001", "summary": "New run"})
    assert third["ticket_id"] == "T-2"


def test_ticket_validation(tmp_path):
    box = make_box(tmp_path)
    assert "error" in box.dispatch("create_support_ticket", {"member_id": "M1001", "summary": " "})
    assert "error" in box.dispatch("create_support_ticket", {"member_id": "M1001", "summary": "x" * 301})
    assert box.dispatch("create_support_ticket", {"member_id": "M9999", "summary": "hi"}) == {"error": "member not found"}


def test_search_faq_returns_at_most_three(tmp_path):
    hits = [{"source": "f.txt", "text": str(i), "distance": 0.1} for i in range(5)]
    out = make_box(tmp_path, hits).dispatch("search_faq", {"query": "fee"})
    assert len(out["results"]) == 3


def test_unknown_tool_and_bad_arguments(tmp_path):
    box = make_box(tmp_path)
    assert "unknown tool" in box.dispatch("delete_everything", {})["error"]
    assert box.dispatch("get_member", {"wrong": "x"}) == {"error": "invalid arguments"}


def test_audit_log_written_without_ticket_text(tmp_path):
    box = make_box(tmp_path)
    box.dispatch("create_support_ticket", {"member_id": "M1001", "summary": "secret details"})
    line = (tmp_path / "audit.jsonl").read_text(encoding="utf-8").strip().splitlines()[-1]
    entry = json.loads(line)
    assert entry["tool"] == "create_support_ticket" and entry["ok"] is True
    assert "secret details" not in line
