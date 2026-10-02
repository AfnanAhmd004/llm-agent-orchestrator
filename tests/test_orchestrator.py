import json

import pytest

from orchestrator import Agent, ScriptedLLM, pipeline, plan_and_execute, review_loop, tool


@tool("Add two integers.")
def add(a: int, b: int) -> int:
    return a + b


def final(x):
    return json.dumps({"final": x})


def test_tool_call_then_final():
    llm = ScriptedLLM([json.dumps({"tool": "add", "args": {"a": 2, "b": 3}}), final("5")])
    trace = []
    assert Agent("calc", "math", llm, [add]).run("2+3?", trace) == "5"
    assert any(e.kind == "tool" and "-> 5" in e.content for e in trace)
    assert "Observation: 5" in llm.calls[1][-1]["content"]


def test_invalid_json_is_recovered():
    llm = ScriptedLLM(["sure, the answer is 5", final("5")])
    trace = []
    assert Agent("a", "r", llm).run("q", trace) == "5"
    assert trace[1].kind == "error"


def test_bad_tool_args_are_reported_not_raised():
    llm = ScriptedLLM([json.dumps({"tool": "add", "args": {"a": "2", "b": 3}}), final("gave up")])
    trace = []
    Agent("a", "r", llm, [add]).run("q", trace)
    assert any(e.kind == "error" and "expects int" in e.content for e in trace)


def test_unknown_tool():
    llm = ScriptedLLM([json.dumps({"tool": "rm_rf", "args": {}}), final("ok")])
    trace = []
    Agent("a", "r", llm, [add]).run("q", trace)
    assert any("unknown tool" in e.content for e in trace)


def test_step_budget():
    llm = ScriptedLLM(["nope"] * 3)
    with pytest.raises(RuntimeError):
        Agent("a", "r", llm, max_steps=3).run("q")


def test_pipeline_passes_outputs():
    a = Agent("a", "r", ScriptedLLM([final("draft")]))
    b_llm = ScriptedLLM([final("polished")])
    b = Agent("b", "r", b_llm)
    assert pipeline([a, b], "start") == "polished"
    assert b_llm.calls[0][1]["content"] == "draft"


def test_plan_and_execute_routes_steps():
    plan = json.dumps([{"worker": "w1", "task": "t1"}, {"worker": "w2", "task": "t2"}])
    planner = Agent("p", "r", ScriptedLLM([final(plan)]))
    w2_llm = ScriptedLLM([final("r2")])
    out = plan_and_execute(planner, {"w1": Agent("w1", "r", ScriptedLLM([final("r1")])), "w2": Agent("w2", "r", w2_llm)}, "go")
    assert out["results"] == {0: "r1", 1: "r2"}
    assert "step 0: r1" in w2_llm.calls[0][1]["content"]


def test_review_loop_revises_until_approved():
    author = Agent("author", "r", ScriptedLLM([final("v1"), final("v2")]))
    reviewer = Agent("rev", "r", ScriptedLLM([final("add units"), final("APPROVE")]))
    draft, rounds = review_loop(author, reviewer, "write spec")
    assert draft == "v2" and rounds == 2
