from types import SimpleNamespace

from agent import Agent
from tools import Toolbox


def text(t):
    return SimpleNamespace(type="text", text=t)


def tool_use(tid, name, args):
    return SimpleNamespace(type="tool_use", id=tid, name=name, input=args)


def resp(content, stop):
    return SimpleNamespace(content=content, stop_reason=stop,
                           usage=SimpleNamespace(input_tokens=10, output_tokens=5))


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.messages = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def make_agent(tmp_path, responses, max_steps=6):
    box = Toolbox(str(tmp_path / "m.sqlite"),
                  faq_search=lambda q: [{"source": "faq.txt", "text": "149 DKK", "distance": 0.1}],
                  audit_path=str(tmp_path / "audit.jsonl"))
    client = FakeClient(responses)
    agent = Agent(box, client=client, max_steps=max_steps, log_path=str(tmp_path / "runs.jsonl"))
    return agent, client


def test_direct_answer_without_tools(tmp_path):
    agent, _ = make_agent(tmp_path, [resp([text("Please give me your member ID.")], "end_turn")])
    res = agent.run("What is my fee?")
    assert res["answer"] == "Please give me your member ID."
    assert res["tool_calls"] == [] and res["steps"] == 1


def test_tool_call_then_answer(tmp_path):
    agent, client = make_agent(tmp_path, [
        resp([text("Checking."), tool_use("t1", "get_member", {"member_id": "M1003"})], "tool_use"),
        resp([text("M1003 is cancelling, effective 2026-12-31.")], "end_turn"),
    ])
    res = agent.run("Status of M1003?")
    assert [c["name"] for c in res["tool_calls"]] == ["get_member"]
    assert "2026-12-31" in res["answer"]
    assert res["input_tokens"] == 20 and res["output_tokens"] == 10
    sent_back = [m for m in client.calls[-1]["messages"]
                 if m["role"] == "user" and isinstance(m["content"], list)]
    assert sent_back[0]["content"][0]["type"] == "tool_result"
    assert sent_back[0]["content"][0]["tool_use_id"] == "t1"


def test_tool_errors_are_passed_to_model_not_raised(tmp_path):
    agent, client = make_agent(tmp_path, [
        resp([tool_use("t1", "get_member", {"member_id": "M9999"})], "tool_use"),
        resp([text("Member not found.")], "end_turn"),
    ])
    res = agent.run("Status of M9999?")
    assert res["error"] is None
    result_block = [m for m in client.calls[-1]["messages"]
                    if m["role"] == "user" and isinstance(m["content"], list)][0]["content"][0]
    assert result_block["is_error"] is True


def test_stops_at_max_steps(tmp_path):
    looping = [resp([tool_use(f"t{i}", "search_faq", {"query": "fee"})], "tool_use") for i in range(2)]
    agent, client = make_agent(tmp_path, looping, max_steps=2)
    res = agent.run("loop forever")
    assert res["error"] == "max steps reached"
    assert res["steps"] == 2 and len(client.calls) == 2
    assert "went wrong" in res["answer"]


def test_api_failure_is_logged_not_raised(tmp_path):
    agent, _ = make_agent(tmp_path, [RuntimeError("boom")])
    res = agent.run("anything")
    assert "RuntimeError" in res["error"]
    assert (tmp_path / "runs.jsonl").exists()


def test_ignores_non_text_blocks_in_final_answer(tmp_path):
    thinking = SimpleNamespace(type="thinking", thinking="...")
    agent, _ = make_agent(tmp_path, [resp([thinking, text("Done.")], "end_turn")])
    assert agent.run("hi")["answer"] == "Done."
