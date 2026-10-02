# llm-agent-orchestrator

A small, dependency-free framework for **multi-agent LLM systems**: tool-using agents with a strict JSON action protocol, plus three orchestration patterns (pipeline, planner/workers, author/reviewer). Every model call, tool call and error is recorded in a trace.

It is built to be **testable**: a `ScriptedLLM` backend replays fixed replies, so control flow, error handling and tool routing can be unit-tested without a model, an API key or any randomness.

## Core ideas

- **One action per turn.** The model replies with `{"tool": ..., "args": ...}` or `{"final": ...}`. Anything else is fed back as an error and the agent retries within its step budget.
- **Typed tools.** `@tool` builds a tool from a type-annotated function; arguments are checked for missing, extra and wrongly typed fields before the function runs.
- **Failure is data.** Bad JSON, unknown tools and tool exceptions become observations the model can react to, not crashes.
- **Bounded.** Every agent has `max_steps`; review loops have `max_rounds`.

## Patterns

| Pattern | Function | Use for |
|---|---|---|
| Pipeline | `pipeline([a, b, c], task)` | fixed multi-stage processing |
| Planner / workers | `plan_and_execute(planner, workers, task)` | decomposing a request across specialists |
| Author / reviewer | `review_loop(author, reviewer, task)` | drafting with a quality gate |

## Example: robot fleet dispatch

```bash
pip install -e ".[dev]"
python examples/warehouse_mission.py
```

```
[planner  ] final [{"worker": "inventory", ...}, {"worker": "dispatch", ...}]
[inventory] tool  check_stock -> {"shelf": "S3", "qty": 14}
[dispatch ] tool  nearest_robot -> {"robot": "amr-2", "distance_m": 2.24}
[dispatch ] final amr-2 dispatched to S3 (2.24 m away)
```

The example runs offline with scripted replies. To use a real model:

```python
from orchestrator import AnthropicLLM, Agent
agent = Agent("dispatch", "You assign robots to missions.", AnthropicLLM(model="<model-id>"), tools=[nearest_robot])
```

(`pip install -e ".[llm]"` and set `ANTHROPIC_API_KEY`.) Any other model works through `FunctionLLM(fn)`.

## Tests

Tool calls and observations, recovery from invalid JSON, typed-argument errors, unknown tools, step budgets, and the routing and context passing of each pattern.

## License

MIT
