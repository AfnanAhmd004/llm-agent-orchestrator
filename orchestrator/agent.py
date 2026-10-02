"""A tool-using agent with a strict JSON action protocol and a step budget."""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from .llm import LLM
from .tools import Tool

PROTOCOL = (
    "Respond with exactly one JSON object and nothing else.\n"
    'To call a tool: {"tool": "<name>", "args": {...}}\n'
    'To finish:      {"final": "<answer>"}'
)


@dataclass
class TraceEvent:
    agent: str
    kind: str  # "llm" | "tool" | "error" | "final"
    content: str


@dataclass
class Agent:
    name: str
    role: str
    llm: LLM
    tools: list[Tool] = field(default_factory=list)
    max_steps: int = 8

    def system_prompt(self) -> str:
        tools = "\n".join(t.spec() for t in self.tools) or "(no tools)"
        return f"You are {self.name}. {self.role}\n\nTools:\n{tools}\n\n{PROTOCOL}"

    def run(self, task: str, trace: list[TraceEvent] | None = None) -> str:
        trace = trace if trace is not None else []
        by_name = {t.name: t for t in self.tools}
        messages = [{"role": "system", "content": self.system_prompt()}, {"role": "user", "content": task}]
        for _ in range(self.max_steps):
            reply = self.llm.complete(messages)
            trace.append(TraceEvent(self.name, "llm", reply))
            messages.append({"role": "assistant", "content": reply})
            try:
                action = _parse(reply)
            except ValueError as e:
                trace.append(TraceEvent(self.name, "error", str(e)))
                messages.append({"role": "user", "content": f"Invalid response: {e}. {PROTOCOL}"})
                continue
            if "final" in action:
                trace.append(TraceEvent(self.name, "final", str(action["final"])))
                return str(action["final"])
            name = action.get("tool")
            if name not in by_name:
                msg = f"unknown tool {name!r}; available: {sorted(by_name)}"
                trace.append(TraceEvent(self.name, "error", msg))
                messages.append({"role": "user", "content": msg})
                continue
            try:
                result = by_name[name](action.get("args", {}))
                obs = json.dumps(result, default=str)
                trace.append(TraceEvent(self.name, "tool", f"{name} -> {obs}"))
            except (ValueError, TypeError) as e:
                obs = f"tool error: {e}"
                trace.append(TraceEvent(self.name, "error", obs))
            messages.append({"role": "user", "content": f"Observation: {obs}"})
        raise RuntimeError(f"{self.name} exceeded {self.max_steps} steps")


def _parse(text: str) -> dict:
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object found")
    obj = json.loads(text[start : end + 1])
    if not isinstance(obj, dict) or not ({"final"} & obj.keys() or "tool" in obj):
        raise ValueError('object must contain "tool" or "final"')
    return obj
