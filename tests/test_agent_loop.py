"""Tests the agent loop with a scripted fake model - no API key or network needed."""
from types import SimpleNamespace as NS

from airport_agent.agent import AirportAgent


class FakeClient:
    """Mimics client.messages.create / client.beta.messages.create with canned responses."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
        self.messages = NS(create=self._create)
        self.beta = NS(messages=NS(create=self._create))

    def _create(self, **kwargs):
        # Snapshot the history: the agent keeps appending to the same list object.
        self.requests.append(dict(kwargs, messages=list(kwargs["messages"])))
        return self.responses.pop(0)


def tool_use(id_, name, args):
    return NS(type="tool_use", id=id_, name=name, input=args)


def text(t):
    return NS(type="text", text=t)


def test_loop_runs_tool_then_answers():
    client = FakeClient([
        NS(stop_reason="tool_use", content=[tool_use("t1", "find_airports", {"query": "santa ana"})]),
        NS(stop_reason="end_turn", content=[text("Santa Ana is SNA.")]),
    ])
    agent = AirportAgent(client=client)
    reply = agent.ask("What code is Santa Ana?")

    assert reply.text == "Santa Ana is SNA."
    assert [c.name for c in reply.tool_calls] == ["find_airports"]
    assert '"SNA"' in reply.tool_calls[0].output
    # The second request must carry the tool result, linked by id, in a user turn.
    tool_msg = client.requests[1]["messages"][-1]
    assert tool_msg["role"] == "user"
    assert tool_msg["content"][0]["type"] == "tool_result"
    assert tool_msg["content"][0]["tool_use_id"] == "t1"


def test_parallel_tool_calls_return_in_one_message():
    client = FakeClient([
        NS(stop_reason="tool_use", content=[tool_use("a", "get_airport_profile", {"iata": "LAX"}),
                                            tool_use("b", "get_airport_profile", {"iata": "ZZZ"})]),
        NS(stop_reason="end_turn", content=[text("done")]),
    ])
    agent = AirportAgent(client=client)
    reply = agent.ask("compare")
    results = client.requests[1]["messages"][-1]["content"]
    assert [r["tool_use_id"] for r in results] == ["a", "b"]
    assert results[1]["is_error"] is True          # bad code is reported back, not raised
    assert len(reply.tool_calls) == 2


def test_history_supports_follow_ups():
    client = FakeClient([
        NS(stop_reason="end_turn", content=[text("first")]),
        NS(stop_reason="end_turn", content=[text("second")]),
    ])
    agent = AirportAgent(client=client)
    agent.ask("q1")
    agent.ask("q2")
    sent = client.requests[1]["messages"]
    assert [m["role"] for m in sent] == ["user", "assistant", "user"]


def test_step_limit_forces_final_answer(monkeypatch):
    from airport_agent import agent as agent_module
    monkeypatch.setattr(agent_module.config, "MAX_AGENT_STEPS", 2)
    looping = NS(stop_reason="tool_use", content=[tool_use("x", "get_methodology", {})])
    client = FakeClient([looping, looping, NS(stop_reason="end_turn", content=[text("partial answer")])])
    reply = AirportAgent(client=client).ask("loop forever")
    assert reply.stop_reason == "max_steps"
    assert reply.text == "partial answer"
    assert client.requests[-1]["tool_choice"] == {"type": "none"}


def test_usage_cost_and_telemetry_are_recorded():
    from airport_agent import config
    usage = NS(input_tokens=1000, output_tokens=500, cache_creation_input_tokens=2000, cache_read_input_tokens=10000)
    client = FakeClient([NS(stop_reason="end_turn", content=[text("hi")], usage=usage)])
    reply = AirportAgent(client=client, model="claude-opus-5").ask("q")
    # 1000*5 + 500*25 + 2000*5*1.25 + 10000*5*0.1 = 35,000 per million tokens = $0.035
    assert reply.cost_usd == 0.035
    assert reply.usage.model_calls == 1
    log_lines = (config.LOG_DIR / "agent.log").read_text().splitlines()
    assert '"event": "answer"' in log_lines[-1]
