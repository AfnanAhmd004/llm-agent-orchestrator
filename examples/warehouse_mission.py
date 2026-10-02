"""Planner + workers for a robot fleet: plan a pick mission, check stock, assign the nearest robot.

Runs offline with scripted model replies so the control flow is reproducible.
Swap `ScriptedLLM` for `AnthropicLLM(model=...)` to drive it with a real model.
"""
import json
import math

from orchestrator import Agent, ScriptedLLM, plan_and_execute, tool

INVENTORY = {"bracket-A7": {"shelf": "S3", "qty": 14}, "motor-M2": {"shelf": "S1", "qty": 0}}
SHELVES = {"S1": (2.0, 8.0), "S3": (12.0, 4.0)}
ROBOTS = {"amr-1": (0.0, 0.0), "amr-2": (10.0, 5.0)}


@tool("Look up stock level and shelf for a part number.")
def check_stock(part: str) -> dict:
    return INVENTORY.get(part, {"error": "unknown part"})


@tool("Return the robot closest to a shelf and its distance in metres.")
def nearest_robot(shelf: str) -> dict:
    sx, sy = SHELVES[shelf]
    name, (x, y) = min(ROBOTS.items(), key=lambda kv: math.dist(kv[1], (sx, sy)))
    return {"robot": name, "distance_m": round(math.dist((x, y), (sx, sy)), 2)}


planner = Agent("planner", "You decompose operations requests into steps for specialist workers.", ScriptedLLM([
    json.dumps({"final": json.dumps([
        {"worker": "inventory", "task": "Check stock for bracket-A7."},
        {"worker": "dispatch", "task": "Assign a robot to pick bracket-A7 from its shelf."},
    ])}),
]))
inventory = Agent("inventory", "You answer stock questions using tools.", ScriptedLLM([
    json.dumps({"tool": "check_stock", "args": {"part": "bracket-A7"}}),
    json.dumps({"final": "bracket-A7: 14 in stock on shelf S3"}),
]), tools=[check_stock])
dispatch = Agent("dispatch", "You assign robots to missions using tools.", ScriptedLLM([
    json.dumps({"tool": "nearest_robot", "args": {"shelf": "S3"}}),
    json.dumps({"final": "amr-2 dispatched to S3 (2.24 m away)"}),
]), tools=[nearest_robot])

trace = []
out = plan_and_execute(planner, {"inventory": inventory, "dispatch": dispatch}, "Pick one bracket-A7 for line 2", trace)
for e in trace:
    print(f"[{e.agent:<9}] {e.kind:<5} {e.content}")
print("\nresults:", out["results"])
