"""The agent: a Claude model in a loop with our tools.

How it works (the whole idea of an "agent" fits in `ask`):
  1. Send the conversation + tool definitions to Claude.
  2. If Claude answers with `stop_reason == "tool_use"`, it is asking us to
     run one or more tools. We run them and send the results back as a
     `tool_result` message, then go to 1.
  3. When Claude stops asking for tools, its text is the final answer.

The conversation history lives in `self.messages`, which is what makes
follow-up questions ("and what about Hartford?") work: the model sees the
earlier questions, tool results and answers.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

import anthropic

from . import config
from .grounding import grounding
from .telemetry import Usage, log_event
from .tools import TOOLS, run_tool

class AgentError(Exception):
    """A model call failed. `message` is safe to show the user; `status` is an HTTP status for the web API."""

    def __init__(self, message: str, status: int = 502):
        super().__init__(message)
        self.message, self.status = message, status


def _missing_key_error() -> AgentError:
    return AgentError(f"No Anthropic API key found. Put ANTHROPIC_API_KEY=... in {config.ENV_FILE} "
                      "(copy .env.example) or set it as an environment variable, then restart.", 401)


def describe_api_error(exc: Exception) -> AgentError:
    """Turn Anthropic SDK exceptions into a short, actionable message instead of a traceback."""
    detail = getattr(exc, "message", None) or str(exc)
    if isinstance(exc, anthropic.CredentialsError):  # e.g. a broken `ant auth login` profile
        return _missing_key_error()
    if isinstance(exc, anthropic.AuthenticationError):
        return AgentError("The Anthropic API key was rejected. Check ANTHROPIC_API_KEY in .env and restart.", 401)
    if isinstance(exc, anthropic.PermissionDeniedError):
        return AgentError(f"The API key isn't allowed to use this model or feature: {detail}", 403)
    if isinstance(exc, anthropic.BadRequestError) and "credit balance" in detail.lower():
        return AgentError("Your Anthropic account is out of credits. Add credits in the Anthropic console "
                          "(Plans & Billing), then try again.", 402)
    if isinstance(exc, anthropic.RateLimitError):
        return AgentError("Rate limit reached on the Anthropic API. Wait a minute and try again.", 429)
    if isinstance(exc, anthropic.APIConnectionError):
        return AgentError("Can't reach the Anthropic API. Check your internet connection.", 503)
    if isinstance(exc, anthropic.APIStatusError):
        return AgentError(f"The Anthropic API returned an error ({exc.status_code}): {detail}", 502)
    return AgentError(f"The model call failed: {detail}", 502)


@dataclass
class ToolCall:
    name: str
    input: dict
    output: str
    is_error: bool
    duration_ms: int = 0


@dataclass
class AgentReply:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str | None = None
    usage: Usage = field(default_factory=Usage)
    cost_usd: float | None = None
    latency_s: float = 0.0
    grounding: float | None = None          # share of numbers traced to tool outputs (None = no numbers)
    ungrounded: list[str] = field(default_factory=list)


class AirportAgent:
    def __init__(self, client: anthropic.Anthropic | None = None, model: str = config.MODEL):
        # Created lazily on first use, so a missing/broken credential becomes a clear AgentError in ask().
        self.client = client
        self.model = model
        self.messages: list[dict] = []
        self.session_id = str(uuid.uuid4())
        self.failed_question: str | None = None  # last question lost to an API error

    def reset(self) -> None:
        self.messages = []
        self.failed_question = None

    def _call_model(self, **extra):
        if self.client is None:
            self.client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment / .env
        kwargs = dict(
            model=self.model,
            max_tokens=16000,
            system=config.SYSTEM_PROMPT,  # prompts/system.md
            tools=TOOLS,
            messages=self.messages,
            output_config={"effort": config.EFFORT},
            # Caches the stable prefix (tools + system + earlier turns) so each
            # loop iteration only pays full price for the new tokens.
            cache_control={"type": "ephemeral"},
            **extra,
        )
        if config.USE_REFUSAL_FALLBACK:
            return self.client.beta.messages.create(
                betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs)
        return self.client.messages.create(**kwargs)

    def ask(self, question: str) -> AgentReply:
        """Answer one question. On an API failure, raises AgentError and leaves the history unchanged."""
        checkpoint = len(self.messages)
        original = question
        if self.failed_question:
            # The failed turn was rolled back, so tell the model about it: "try again" then means that question.
            question = (f'(My previous message failed because of an API error and got no answer: '
                        f'"{self.failed_question}")\n\n{question}')
        try:
            reply = self._ask(question)
            self.failed_question = None
            return reply
        except TypeError as exc:
            # The SDK raises TypeError (not an API error) at request time when it finds no credentials.
            if "authentication method" not in str(exc):
                raise
            del self.messages[checkpoint:]
            self.failed_question = self.failed_question or original
            raise _missing_key_error() from exc
        except anthropic.AnthropicError as exc:
            # Roll back the half-finished turn, so a dangling tool_use can't break the next request
            # and the failed question doesn't get merged into the next one.
            del self.messages[checkpoint:]
            self.failed_question = self.failed_question or original
            err = describe_api_error(exc)
            log_event("error", session=self.session_id, model=self.model, question=question,
                      status=err.status, error=err.message)
            raise err from exc

    def _ask(self, question: str) -> AgentReply:
        self.messages.append({"role": "user", "content": question})
        calls: list[ToolCall] = []
        usage = Usage()
        started = time.perf_counter()

        for _ in range(config.MAX_AGENT_STEPS):
            response = self._call_model()
            usage.add(response)
            # Keep the full content (text, thinking and tool_use blocks) - the
            # API needs it verbatim on the next request.
            self.messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "tool_use":
                results = []
                for block in response.content:
                    if block.type != "tool_use":
                        continue
                    t0 = time.perf_counter()
                    output, is_error = run_tool(block.name, dict(block.input))
                    ms = int((time.perf_counter() - t0) * 1000)
                    calls.append(ToolCall(block.name, dict(block.input), output, is_error, ms))
                    log_event("tool_call", session=self.session_id, tool=block.name, input=dict(block.input),
                              is_error=is_error, duration_ms=ms, output_chars=len(output))
                    results.append({"type": "tool_result", "tool_use_id": block.id,
                                    "content": output, "is_error": is_error})
                # All results for one turn go back in a single user message.
                self.messages.append({"role": "user", "content": results})
                continue

            text = "".join(b.text for b in response.content if b.type == "text").strip()
            if response.stop_reason == "refusal":
                text = text or "The model declined to answer this request."
            elif response.stop_reason == "max_tokens":
                text += "\n\n_(Answer truncated: output limit reached. Ask me to continue.)_"
            return self._finish(question, text, calls, response.stop_reason, usage, started)

        # Safety valve against runaway loops: force a final answer without tools.
        self.messages.append({"role": "user", "content": (
            "You have reached the tool-call limit. Answer now with what you have and say what is missing.")})
        response = self._call_model(tool_choice={"type": "none"})
        usage.add(response)
        self.messages.append({"role": "assistant", "content": response.content})
        text = "".join(b.text for b in response.content if b.type == "text").strip()
        return self._finish(question, text, calls, "max_steps", usage, started)

    def _tool_outputs(self) -> list[str]:
        """Every tool result in the conversation - follow-ups may cite earlier turns' data."""
        return [item["content"] for m in self.messages if m["role"] == "user" and isinstance(m["content"], list)
                for item in m["content"] if isinstance(item, dict) and item.get("type") == "tool_result"]

    def _finish(self, question: str, text: str, calls: list[ToolCall], stop_reason: str | None,
                usage: Usage, started: float) -> AgentReply:
        share, missing = grounding(text, self._tool_outputs())
        reply = AgentReply(text=text, tool_calls=calls, stop_reason=stop_reason, usage=usage,
                           cost_usd=usage.cost_usd(self.model),
                           latency_s=round(time.perf_counter() - started, 2),
                           grounding=share, ungrounded=missing)
        log_event("answer", session=self.session_id, model=self.model, question=question,
                  stop_reason=stop_reason, tools=[c.name for c in calls], latency_s=reply.latency_s,
                  cost_usd=reply.cost_usd, grounding=share, ungrounded=missing, **usage.to_dict())
        return reply
