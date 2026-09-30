"""Terminal chat:  python -m airport_agent.cli  (add --trace to print tool calls)."""
from __future__ import annotations

import sys
import threading
import time

from .agent import AgentError, AirportAgent


class LiveStatus:
    """Shows what the agent is doing while it works: a spinner line plus one line per tool call.

    The spinner only runs on a real terminal; when output is piped, just the step lines are printed.
    """
    FRAMES = "|/-\\"

    def __init__(self, stream=sys.stdout):
        self.stream = stream
        self.animate = stream.isatty()
        self.status = "Thinking"
        self.lock = threading.Lock()
        self.done = threading.Event()
        self.started = time.monotonic()
        self.thread = threading.Thread(target=self._spin, daemon=True)

    def __enter__(self):
        if self.animate:
            self.thread.start()
        return self

    def __exit__(self, *exc):
        self.done.set()
        if self.animate:
            self.thread.join()
        self._clear()

    def _clear(self):
        if self.animate:
            with self.lock:
                self.stream.write("\r\033[K")
                self.stream.flush()

    def _spin(self):
        i = 0
        while not self.done.wait(0.1):
            with self.lock:
                elapsed = int(time.monotonic() - self.started)
                self.stream.write(f"\r\033[K  {self.FRAMES[i % 4]} {self.status}... {elapsed}s")
                self.stream.flush()
            i += 1

    def __call__(self, event: dict) -> None:
        """Progress callback for AirportAgent.ask()."""
        if event["type"] == "thinking":
            self.status = "Thinking" if event["step"] == 0 else "Reviewing the results"
        elif event["type"] == "tool":
            self._clear()
            with self.lock:
                self.stream.write(f"  -> {event['label']}\n")
                self.stream.flush()
            self.status = event["label"]
        elif event["type"] == "tool_done" and event["is_error"]:
            self._clear()
            with self.lock:
                self.stream.write("     (that step failed; the agent will adjust)\n")
                self.stream.flush()


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
            with LiveStatus() as status:
                reply = agent.ask(question, on_progress=status)
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
