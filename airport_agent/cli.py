"""Terminal chat:  python -m airport_agent.cli  (add --trace to print tool calls)."""
from __future__ import annotations

import sys

from .agent import AgentError, AirportAgent


def main() -> None:
    trace = "--trace" in sys.argv
    agent = AirportAgent()
    print("Airport Investment Agent - ask a question, 'reset' to start over, 'quit' to exit.\n")
    while True:
        try:
            question = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in {"quit", "exit"}:
            break
        if question.lower() == "reset":
            agent.reset()
            print("(conversation cleared)\n")
            continue
        if not question:
            continue
        try:
            reply = agent.ask(question)
        except AgentError as err:
            print(f"\nerror> {err.message}\n")
            continue
        if trace:
            for call in reply.tool_calls:
                print(f"  [tool] {call.name}({call.input}){' ERROR' if call.is_error else ''}")
        print(f"\nagent> {reply.text}\n")
        for call in reply.tool_calls:
            if call.name == "show_chart" and not call.is_error:
                print(f"  [chart: {call.input.get('chart')} for {call.input.get('iatas')} - open the web UI to see it]\n")


if __name__ == "__main__":
    main()
