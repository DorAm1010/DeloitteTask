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


def test_live_grounding_flags_numbers_not_in_tool_outputs():
    client = FakeClient([
        NS(stop_reason="tool_use", content=[tool_use("t1", "find_airports", {"query": "santa ana"})]),
        NS(stop_reason="end_turn", content=[text("SNA had 5,590,354 passengers and a 97.3% load factor.")]),
        NS(stop_reason="end_turn", content=[text("As I said, 5,590,354 passengers.")]),
    ])
    agent = AirportAgent(client=client)
    first = agent.ask("How busy is Santa Ana?")
    assert first.ungrounded == ["97.3%"]       # invented; the passenger count is in the tool output
    assert first.grounding == 0.5
    follow_up = agent.ask("Remind me?")          # no new tools: earlier turns' data still counts
    assert follow_up.grounding == 1.0


def test_telemetry_summary_kpis():
    from airport_agent.telemetry import summarize
    client = FakeClient([
        NS(stop_reason="tool_use", content=[tool_use("t1", "get_airport_profile", {"iata": "ZZZ"})]),
        NS(stop_reason="end_turn", content=[text("No data for ZZZ.")]),
    ])
    AirportAgent(client=client).ask("q")
    s = summarize()
    assert s["answers"] == 1 and s["completion_rate"] == 1.0 and s["tool_error_rate"] == 1.0


class _FailingClient(FakeClient):
    """Returns scripted responses, then raises the given SDK exception."""

    def __init__(self, responses, exc):
        super().__init__(responses)
        self.exc = exc

    def _create(self, **kwargs):
        if not self.responses:
            raise self.exc
        return super()._create(**kwargs)


def _api_error(cls, status, message):
    import httpx2
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx2.Response(status, request=request)
    return cls(message, response=response, body={"error": {"message": message}})


def test_api_error_becomes_friendly_message_and_history_rolls_back():
    import anthropic
    import pytest
    from airport_agent.agent import AgentError
    exc = _api_error(anthropic.BadRequestError, 400, "Your credit balance is too low to access the Anthropic API.")
    # First call asks for a tool, second call fails: the dangling tool_use must not stay in history.
    client = _FailingClient([NS(stop_reason="tool_use", content=[tool_use("t1", "get_methodology", {})])], exc)
    agent = AirportAgent(client=client)
    with pytest.raises(AgentError) as info:
        agent.ask("How does scoring work?")
    assert info.value.status == 402 and "credits" in info.value.message
    assert agent.messages == []                  # rolled back; the next question starts clean


def test_auth_and_connection_errors_are_described():
    import anthropic
    from airport_agent.agent import describe_api_error
    auth = describe_api_error(_api_error(anthropic.AuthenticationError, 401, "invalid x-api-key"))
    assert auth.status == 401 and "ANTHROPIC_API_KEY" in auth.message
    import httpx2
    conn = describe_api_error(anthropic.APIConnectionError(request=httpx2.Request("POST", "https://x")))
    assert conn.status == 503 and "internet" in conn.message


def test_server_returns_json_error_instead_of_500(monkeypatch):
    import anthropic
    import pytest
    from fastapi import HTTPException
    from airport_agent import server
    exc = _api_error(anthropic.BadRequestError, 400, "Your credit balance is too low to access the Anthropic API.")
    monkeypatch.setattr(server, "AirportAgent", lambda: AirportAgent(client=_FailingClient([], exc)))
    server._sessions.clear()
    with pytest.raises(HTTPException) as info:
        server.chat(server.ChatRequest(message="hi"))
    assert info.value.status_code == 402 and "credits" in info.value.detail


def test_missing_api_key_gives_clear_message(monkeypatch, tmp_path):
    import pytest
    from airport_agent import config
    from airport_agent.agent import AgentError
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE", "ANTHROPIC_CONFIG_DIR",
                "ANTHROPIC_BASE_URL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))     # no `ant auth login` profile either
    agent = AirportAgent()                        # real SDK client, created lazily
    with pytest.raises(AgentError) as info:
        agent.ask("hi")
    assert info.value.status == 401 and str(config.ENV_FILE) in info.value.message
    assert agent.messages == []


def test_broken_credentials_profile_gives_same_message(monkeypatch):
    import pytest
    from airport_agent.agent import AgentError
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("ANTHROPIC_CONFIG_DIR", "/nonexistent")  # SDK fails while creating the client
    with pytest.raises(AgentError) as info:
        AirportAgent().ask("hi")
    assert info.value.status == 401 and "ANTHROPIC_API_KEY" in info.value.message


def test_retry_after_api_error_refers_to_the_failed_question():
    import anthropic
    import pytest
    import httpx2
    from airport_agent.agent import AgentError
    conn = anthropic.APIConnectionError(request=httpx2.Request("POST", "https://x"))
    agent = AirportAgent(client=_FailingClient([], conn))
    with pytest.raises(AgentError):
        agent.ask("How often is BOS delayed?")
    with pytest.raises(AgentError):
        agent.ask("try now")                     # still failing: keep the ORIGINAL question
    agent.client = FakeClient([NS(stop_reason="end_turn", content=[text("BOS answer")])])
    agent.ask("try now")
    sent = agent.client.requests[0]["messages"][0]["content"]
    assert "How often is BOS delayed?" in sent and sent.endswith("try now")
    assert agent.failed_question is None         # cleared after success


def test_progress_events_follow_the_loop():
    client = FakeClient([
        NS(stop_reason="tool_use", content=[tool_use("t1", "estimate_unmet_demand", {"iata": "sfo"})]),
        NS(stop_reason="end_turn", content=[text("done")]),
    ])
    events = []
    AirportAgent(client=client).ask("Unmet demand at SFO?", on_progress=events.append)
    assert [e["type"] for e in events] == ["thinking", "tool", "tool_done", "thinking"]
    assert events[1]["label"] == "Estimating unmet demand at SFO"
    assert events[2]["is_error"] is False


def test_broken_progress_listener_does_not_break_the_answer():
    def boom(_event):
        raise RuntimeError("listener crashed")
    client = FakeClient([NS(stop_reason="end_turn", content=[text("fine")])])
    assert AirportAgent(client=client).ask("hi", on_progress=boom).text == "fine"
