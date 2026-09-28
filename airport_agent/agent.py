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

from dataclasses import dataclass, field

import anthropic

from . import config
from .tools import TOOLS, run_tool

SYSTEM_PROMPT = """You are an airport investment analyst assistant for a firm that invests in US airport \
modernization projects. You help analysts find airports where renovation/expansion is most likely to pay off \
because flight and passenger demand is outgrowing capacity.

How to work:
- Every number you state must come from a tool result in this conversation. Never estimate figures from memory. \
If the tools cannot answer, say so plainly and explain what data would be needed.
- Use score_airports for any ranking or comparison; it is the firm's deterministic scoring model. Explain results \
through its components (percentiles, weights, contributions) rather than inventing your own ranking logic.
- Resolve ambiguous places with find_airports. "LA" usually means LAX but the region has several airports; say \
which one you used. If a request is genuinely ambiguous, make a sensible assumption, state it, and offer the \
alternative.
- You may add qualitative context from curated_notes in tool results; label it as context, not data.

How to answer:
- Lead with the direct answer (a sentence or two), then the evidence as a short list or small table.
- Always include a brief "Assumptions & caveats" section: definitions used (e.g. long-haul threshold), data \
periods from data_vintage, coverage limits flagged by the tools, and what the model does not capture (costs, \
airport finances, regulation).
- Express uncertainty honestly: distinguish measured facts, derived estimates, and your interpretation.
- Keep it concise and skimmable; analysts will ask follow-ups.

Scope: US airports and public aviation data only. You do not give financial advice or predict returns; you \
identify and explain demand/capacity signals that inform investment screening."""


@dataclass
class ToolCall:
    name: str
    input: dict
    output: str
    is_error: bool


@dataclass
class AgentReply:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str | None = None


class AirportAgent:
    def __init__(self, client: anthropic.Anthropic | None = None, model: str = config.MODEL):
        self.client = client or anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
        self.model = model
        self.messages: list[dict] = []

    def reset(self) -> None:
        self.messages = []

    def _call_model(self, **extra):
        kwargs = dict(
            model=self.model,
            max_tokens=16000,
            system=SYSTEM_PROMPT,
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
        self.messages.append({"role": "user", "content": question})
        calls: list[ToolCall] = []

        for _ in range(config.MAX_AGENT_STEPS):
            response = self._call_model()
            # Keep the full content (text, thinking and tool_use blocks) - the
            # API needs it verbatim on the next request.
            self.messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "tool_use":
                results = []
                for block in response.content:
                    if block.type != "tool_use":
                        continue
                    output, is_error = run_tool(block.name, dict(block.input))
                    calls.append(ToolCall(block.name, dict(block.input), output, is_error))
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
            return AgentReply(text=text, tool_calls=calls, stop_reason=response.stop_reason)

        # Safety valve against runaway loops: force a final answer without tools.
        self.messages.append({"role": "user", "content": (
            "You have reached the tool-call limit. Answer now with what you have and say what is missing.")})
        response = self._call_model(tool_choice={"type": "none"})
        self.messages.append({"role": "assistant", "content": response.content})
        text = "".join(b.text for b in response.content if b.type == "text").strip()
        return AgentReply(text=text, tool_calls=calls, stop_reason="max_steps")
