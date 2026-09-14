"""
Agentic programs: harness-written workflows that own one rollout in code. Where a prompt can only ask the model to
delegate, verify or synthesize, a program does the coordination itself and leaves the model the leaf work.

A program is configuration: `AgentConfig.agentic_program = (ProgramClass, kwargs)`, serialized like a tool spec.
`Agent.run_program()` (what the rollout manager calls) constructs it with the initialized, unstarted agent and calls
`execute()`, which returns the main agent's own `AgentRunResult`. Inside `execute` a program may, before the first turn:

- `agent.augment_context(content)`: text and activation parts placed ahead of the task in the first user message;
- `agent.run_subagent(brief, max_duration=None)`: a solver in the same sandbox (delegation tools and program removed),
  its result appended to `subagent_results`;
- `agent.set_final_answer(answer)`: finish without a model turn (finish reason "programmed");
- `agent.run()`: the model loop on the (augmented) task.

A program that runs the main agent through `run()` and solvers through `run_subagent()` gets every trajectory recorded
for SFT/RL. A program that drives agents any other way must populate the run results itself. Programs never mutate the
config (the cache key is computed from it before and after the run) and never touch a started segment.
"""
from __future__ import annotations

import typing as t

if t.TYPE_CHECKING:
    from .agent import Agent
    from .agent_config import AgentRunResult


class AgenticProgram:
    name: str = ""

    def __init__(self, agent: "Agent", **kwargs):
        self.agent = agent

    def execute(self) -> "AgentRunResult":
        """Run the workflow; return `self.agent.run_results` (the main agent's own result object)."""
        raise NotImplementedError
