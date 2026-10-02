"""Orchestration patterns: pipeline, planner/workers and reviewer loop."""
from __future__ import annotations

import json

from .agent import Agent, TraceEvent


def pipeline(agents: list[Agent], task: str, trace: list[TraceEvent] | None = None) -> str:
    """Each agent's answer becomes the next agent's input."""
    out = task
    for a in agents:
        out = a.run(out, trace)
    return out


def plan_and_execute(planner: Agent, workers: dict[str, Agent], task: str, trace: list[TraceEvent] | None = None) -> dict:
    """Planner returns a JSON list of {"worker": name, "task": text}; each step runs on that worker.

    Results of earlier steps are passed to later ones as context.
    """
    plan_text = planner.run(f"Break this into steps for workers {sorted(workers)}. "
                            f'Final answer must be a JSON list of {{"worker", "task"}}.\nTask: {task}', trace)
    steps = json.loads(plan_text)
    results = {}
    for i, step in enumerate(steps):
        worker = workers[step["worker"]]
        context = "\n".join(f"step {j}: {r}" for j, r in results.items())
        results[i] = worker.run(f"{step['task']}\n\nEarlier results:\n{context or '(none)'}", trace)
    return {"plan": steps, "results": results}


def review_loop(author: Agent, reviewer: Agent, task: str, max_rounds: int = 3,
                trace: list[TraceEvent] | None = None) -> tuple[str, int]:
    """Author drafts; reviewer replies APPROVE or feedback; repeat until approved or out of rounds."""
    draft = author.run(task, trace)
    for rnd in range(1, max_rounds + 1):
        verdict = reviewer.run(f"Task: {task}\nDraft:\n{draft}\nReply APPROVE or give specific feedback.", trace)
        if verdict.strip().upper().startswith("APPROVE"):
            return draft, rnd
        draft = author.run(f"Task: {task}\nPrevious draft:\n{draft}\nReviewer feedback:\n{verdict}\nRevise.", trace)
    return draft, max_rounds
