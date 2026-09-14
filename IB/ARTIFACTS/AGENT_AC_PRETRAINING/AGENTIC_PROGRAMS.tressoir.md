# Agentic programs

Harness-written workflows that own a rollout's structure in code: the program spawns solvers, shapes the main agent's context and runs the main agent, instead of a prompt the model must obey. Version 0.2 (2026-09-14) makes the program central: one program per config, `Agent.run_program()` as the entry point the rollout manager calls, `execute()` returning the main agent's run result, and lenient deserialization so training can read rows without the program, tool or environment classes present. The first target is one simple end-to-end test on a GPU node with the program written in the test file.

## Executive Summary

**Goal.** Give harness designers programmatic control of rollouts as a first-class part of the system, with a minimal surface: a program is `(class, kwargs)` on the config, the manager calls `agent.run_program()`, the program returns the main agent's `AgentRunResult`. No teacher kind, no library program, no share ratios in this pass; teacher campaigns keep working unchanged (a config without a program behaves exactly as today).

**Shape.**

| Piece | Where | What it does |
|---|---|---|
| `AgenticProgram` | `activation/agent/agentic_program.py` (new) | Base class: `__init__(agent, **kwargs)`, `execute() -> AgentRunResult`. One per config. |
| `AgentConfig.agentic_program` | `agent/agent_config.py` | `tuple[type[AgenticProgram], dict] \| None`, serialized like a tool spec. |
| `Agent.run_program()` | `agent/agent.py` | Runs the program when there is one, else `run()`. The rollout manager calls it. |
| Runtime API for programs | `agent/agent.py` | `augment_context(content)` (ahead of the task, pre-first-turn), `run_subagent(brief, max_duration=None)`, `set_final_answer(answer)`, `run()`. |
| Lenient deserialization | `agent/agent_config.py`, `agentic_program.py` | A class spec that does not resolve becomes a `MissingClass` placeholder that keeps the spec, re-serializes unchanged and raises `MissingClassError` when constructed. Tools, env setups and the program alike. |
| Cache key | `agent/rollout_caching.py` | `metadata["program"]` label appended when present; free-form digests include the program spec. |
| Development tree | `/workspace` (git) | Code is written in the workspace; the round builder stages product files into `IB/ARTIFACTS/AGENT_AC_PRETRAINING/slice4a/` on handoff readiness. |

**Data flow of the test program.** Manager → `agent.run_program()` → `SolveThenVerifyProgram.execute()` → `agent.run_subagent(task)` (same sandbox, delegation tools and program removed, bounded duration; result appended to `subagent_results`) → `agent.augment_context([solver line])` → `agent.run()` templates the first prompt with the solver line ahead of the task, the model checks with python and submits → `execute` returns `agent.run_results`; `run_program` sets the total duration and returns it. The row carries the solver's run in `subagent_results` and the injected line inside `prompt_messages`; the cache replays it like any row.

**Responsibilities.** A program that runs the main agent through `run()` and solvers through `run_subagent` gets every trajectory recorded for SFT/RL for free. A program that drives agents any other way must populate `run_results` (trajectory, prompt fields, `subagent_results`) itself; nothing checks it beyond `execute` returning the main agent's own `run_results` object.

**Boundaries.** Programs augment context before the first turn only (no parts-aware mid-run append). No recursion: `run_subagent` children carry no program and no delegation tools. The wall deadline clips the main run's `max_duration` as today; solver bounds are the program's job. `score` default flips to `None`.

## Accepted decisions

- **v0.1 (2026-09-11):** programs run before the first turn and change runtime state only; the program spec joins the cache key; subagent configs drop the program; `score` defaults to `None`; solver results injected the way the subagent tool returns them (text line, plus a `subagent_return` part when an AC model is configured).
- **v0.2 (2026-09-14, from chat):** one program per config (`agentic_program: (type, kwargs) | None`); `Agent.run_program()` is what the rollout manager calls and falls back to `run()`; `execute()` returns the main agent's `AgentRunResult`, and non-trivial workflows that bypass `run_subagent` populate the training trajectories themselves; no teacher kind and no library program yet, without closing the door for teacher campaigns; no share ratios; the first test is a `test_basic_agent.py`-style GPU test with the program class in the test file on a DeepMath problem of difficulty 2 or below; code is written in the workspace and synced to `IB/ARTIFACTS/...` on handoff readiness (canon).

## Requested Decisions

<article class="decision" data-tressoir-decision data-decision-state="unresolved" aria-labelledby="programs-serde-question">
  <header class="decision-header">
    <div>
      <h3 class="decision-title" id="programs-serde-question">How should deserialization treat a class spec (program, tool, env setup) whose class is not importable?</h3>
      <p class="decision-context">Goal from chat: SFT/RL reads rows without the env, tool or program classes present, but anything that tries to construct one errs. The recommendation refines your `serde_missing` idea; check one to resolve.</p>
    </div>
    <span class="decision-state" data-decision-indicator role="status" aria-live="polite">Unresolved</span>
  </header>
  <fieldset class="decision-options">
    <legend class="visually-hidden">Decision answers</legend>
    <label class="decision-option">
      <input type="checkbox" data-tressoir-input="programs.serde.placeholder">
      <span><strong>`MissingClass` placeholder carrying the spec (Recommended)</strong><small>Your idea, made round-trip safe and uniform. An unresolvable spec becomes a `MissingClass(spec)` object whose `__module__`/`__qualname__` are the spec's, so `_class_spec` re-serializes it unchanged (a row read for training and rewritten keeps its provenance), and whose `__call__` raises `MissingClassError(spec)`. Applied to the program, tools and env setups by `AgentConfig.deserialize` (lenient by default; `strict=True` for callers that need live classes). Live use fails at construction with the spec named: `Agent._initialize_tools`, `agent_env` setups, `run_program`.</small></span>
    </label>
    <label class="decision-option">
      <input type="checkbox" data-tressoir-input="programs.serde.function">
      <span><strong>A bare `serde_missing(*args, **kwargs)` function</strong><small>Simplest: the spec resolves to one shared function that raises. Loses the spec on re-serialization (every missing class serializes as `serde_missing`) and cannot name what was missing in the error.</small></span>
    </label>
    <label class="decision-option">
      <input type="checkbox" data-tressoir-input="programs.serde.lazy">
      <span><strong>Lazy specs, resolved on first use</strong><small>Configs keep raw specs and resolve when a tool/program/env setup is constructed. Cleanest semantics, but every reader of `config.tools` / `env_setups` (reporter, tests, subagent inheritance) must handle both forms; a larger change than the placeholder.</small></span>
    </label>
  </fieldset>
  <div class="field decision-feedback">
    <label for="programs-serde-response">Free Response</label>
    <textarea id="programs-serde-response" rows="2" data-tressoir-input="programs.serde.feedback" data-tressoir-autogrow="2:6" placeholder="Anything the choices miss…"></textarea>
  </div>
</article>

## Milestones

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">P0 — The workspace becomes the development tree</span>
    <span class="card-oneliner">Import the campaign-stable tree into /workspace, keep git there, point the round builder at it; IB/TMP keeps only disposable evidence.</span>
    <span class="card-badge">Planning</span>
  </summary>

#### Planning Overview

Per your instruction: code is written in the workspace (`/workspace`, the container's git copy of the repository) and reaches you through `IB/ARTIFACTS/AGENT_AC_PRETRAINING/slice4a/` (product files, `changes.patch`, manifest) whenever the round is rebuilt at handoff readiness. Steps: copy the work tree's product files (68 differ from the workspace snapshot) into `/workspace`; commit on `main` there with branch `campaign-stable` and the redo tag at the imported commit; switch `build_round.py` to read from `/workspace`; run the CPU test set from `/workspace`; retire `IB/TMP/AGENT_AC_PRETRAINING/work` (kept until the round from `/workspace` reproduces it byte for byte, then deleted). Canon gains the rule (Cross-cutting decisions).

#### Planned Changes

`IB/TMP/AGENT_AC_PRETRAINING/build_round.py`

```diff
-work = root / 'IB/TMP/AGENT_AC_PRETRAINING/work'
+work = root                                       # the workspace is the development tree; IB/TMP holds evidence only
```

`IB/CANON/ROOT_CANON.md · Cross-cutting decisions`

```diff
+- **Decision — Code lives in the workspace; artifacts carry it to the user.** Agents develop in `/workspace` (git: `main` for
+  the slice, a stable branch and tag for every tree a long run was launched from). `IB/TMP/` holds disposable evidence only
+  (logs, pulled caches, probes), never a second copy of the code. At handoff readiness the round builder stages the product
+  files, the patch against the source and a manifest into `IB/ARTIFACTS/<TASK>/<round>/`, and the round document lists them.
```

</details>

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">P1 — Program runtime, config and lenient serde</span>
    <span class="card-oneliner">The base class, the single-program config field, run_program and the runtime API, the cache key segment, and MissingClass deserialization.</span>
    <span class="card-badge">Planning</span>
  </summary>

#### Planning Overview

`run_program` is the one entry point: it constructs the program, calls `execute()`, checks the returned object is the agent's own `run_results`, records the total duration over the whole program (subagents included) and reports finish. A program exception ends the run with `finish_reason="error"` and the traceback in `score_feedback`, like a tool failure, so the manager and the cache treat it as a row. `run()` is unchanged for the model loop; it gains a `reported` guard so `run_program` can register the agent with the reporter before solvers start (children then appear under a known parent). The subagent tool's config inheritance moves to a module function shared with `run_subagent`; both drop delegation tools and the program. `augment_context` fills an agent slot that `first_user_messages()` places ahead of the task text; it asserts the agent has not started. `set_final_answer` sets the answer and `finish_reason="programmed"` for programs that never run the model. Deserialization is lenient per the decision above; `_class_spec` also maps `__main__` to the script's module name. The cache key appends `metadata["program"]` when present and the free-form digest includes the program spec, so a programmed variant never collides with the plain one and campaign keys are unchanged.

#### Planned Changes

`activation/agent/agentic_program.py` · new

```python
class AgenticProgram:
    """A harness-written workflow that owns one rollout. `execute` returns the main agent's own run_results; a program
    that runs the main agent through agent.run() and solvers through agent.run_subagent() gets every trajectory recorded;
    any other way of driving agents must populate run_results itself."""
    name: str = ""

    def __init__(self, agent: "Agent", **kwargs):
        self.agent = agent

    def execute(self) -> "AgentRunResult":
        raise NotImplementedError


class MissingClassError(RuntimeError):
    """A tool, env setup or program class named by a record is not importable here."""


class MissingClass:
    """Stands in for a class spec that did not resolve: keeps the spec (re-serializes unchanged), refuses construction."""
    def __init__(self, module: str, qualname: str, error: str):
        self.__module__, self.__qualname__, self.error = module, qualname, error
    def __call__(self, *args, **kwargs):
        raise MissingClassError(f"{self.__module__}:{self.__qualname__} is not importable here ({self.error}); "
                                "the record can be read and trained on, not run")
```

`activation/agent/agent_config.py · AgentConfig, serialize, deserialize, _class_spec, _resolve_class_spec`

```diff
@@ AgentConfig @@
     env_setups: dict[str, tuple[type["AgentEnvSetup"], dict]] = field(default_factory=dict)
+    agentic_program: tuple[type["AgenticProgram"], dict] | None = None   # the program that owns the rollout (run_program); None: run()
@@ serialize @@
-        data = {... if key not in ("dataset_task", "tools", "env_setups")}
+        data = {... if key not in ("dataset_task", "tools", "env_setups", "agentic_program")}
+        data["agentic_program"] = None if self.agentic_program is None else _class_spec(*self.agentic_program)
@@ deserialize(data, harness=None, strict=False) @@
-        tools = {name: None if spec is None else _resolve_class_spec(spec) ...}
+        resolve = lambda spec: _resolve_class_spec(spec, strict=strict)
+        tools = {name: None if spec is None else resolve(spec) ...}
+        setups = {name: resolve(spec) ...}
+        program_spec = data.pop("agentic_program", None)
+        program = None if program_spec is None else resolve(program_spec)
@@ _class_spec @@
-    return {"class": f"{cls.__module__}:{cls.__qualname__}", "kwargs": _jsonable(kwargs)}
+    module = cls.__module__
+    if module == "__main__":                                   # a class in a script run as `python -m pkg.script`
+        spec = getattr(sys.modules.get("__main__"), "__spec__", None)
+        module = spec.name if spec is not None and spec.name else module
+    return {"class": f"{module}:{cls.__qualname__}", "kwargs": _jsonable(kwargs)}   # a MissingClass carries the original names
@@ _resolve_class_spec(spec, strict=True) @@
-    cls = importlib.import_module(module_name) ...
+    try:
+        cls = importlib.import_module(module_name); for part in qualname.split("."): cls = getattr(cls, part)
+    except Exception as error:
+        if strict: raise
+        cls = MissingClass(module_name, qualname, f"{type(error).__name__}: {error}")
     return cls, dict(spec.get("kwargs") or {})
@@ AgentRunResult @@
-    score: float | None = 0.0
+    score: float | None = None
```

`activation/agent/agent.py · Agent`

```diff
@@ __init__ @@
+        self.program_content: list[dict] = []     # placed ahead of the task by a program, consumed once by first_user_messages()
+        self.reported = False
@@ first_user_messages @@
         prompt = self._offloaded_prompt(config.user_prompt)
+        lead = list(self.program_content)
         if not messages or messages[-1].get("role") != "user":
-            messages.append({"role": "user", "content": prompt})
+            messages.append({"role": "user", "content": (lead + [{"type": "text", "text": prompt}]) if lead else prompt})
         elif prompt:
             ...
-            messages[-1]["content"] = content + [{"type": "text", "text": prompt}]
+            messages[-1]["content"] = content + lead + [{"type": "text", "text": prompt}]
@@ run @@
-            if self.reporter is not None:
+            if self.reporter is not None and not self.reported:
                 self.reporter.report_agent_start(self)
+                self.reported = True
@@ new: the program entry point @@
+    def run_program(self) -> AgentRunResult:
+        """What the rollout manager calls: the config's program when there is one (its result is this agent's own
+        run_results, duration over the whole program), else run()."""
+        program = self.agent_config.agentic_program
+        if program is None:
+            return self.run()
+        cls, kwargs = program
+        results, start = self.run_results, time.time()
+        if self.reporter is not None and not self.reported:
+            self.reporter.report_agent_start(self); self.reported = True
+        try:
+            returned = cls(self, **kwargs).execute()
+            if returned is not results:
+                raise TypeError(f"{cls.__name__}.execute must return the main agent's own run_results")
+        except Exception:
+            results.finish_reason = "error"
+            results.score_feedback = traceback.format_exc()
+            print(f"Agent {self.agent_id[:8]} - program error:\n{results.score_feedback}", flush=True)
+        finally:
+            results.duration = time.time() - start
+            if not self.finished:
+                self.finished = True
+                if self.reporter is not None:
+                    self.reporter.report_agent_finish(self)
+        return results
+
+    def augment_context(self, content: list[dict]) -> None:
+        """Content (text and activation parts) placed ahead of the task in the first user message; before the first turn only."""
+        assert not self.started, "augment_context runs before the first turn"
+        self.program_content.extend(content)
+
+    def run_subagent(self, brief: str, max_duration: float | None = None, agent_name: str = "program_subagent") -> AgentRunResult:
+        """A solver in this agent's sandbox on `brief`: no delegation tools, no program; its result joins subagent_results."""
+        config = subagent_config_of(self, brief, agent_name=agent_name)
+        if max_duration is not None:
+            config.max_duration = min(float(config.max_duration), float(max_duration))
+        child = Agent(self.harness, config, agent_env=self.agent_env, parent_agent=self, reporter=self.reporter, seed=self.seed)
+        try:
+            result = child.run()
+        finally:
+            child.shutdown()                       # the env is the parent's; shutdown only releases the child's own state
+        with self.lock:
+            self.run_results.subagent_results.append(result)
+        return result
+
+    def set_final_answer(self, answer) -> None:
+        if answer is None or (isinstance(answer, str) and not answer.strip()):
+            raise ValueError("set_final_answer needs a non-empty answer")
+        self.run_results.answer = answer
+        self.run_results.finish_reason = "programmed"
```

`activation/agent/agent_tools.py · subagent_config_of (from SubagentTool._inherit_general_config)`

```diff
+def subagent_config_of(parent: "Agent", task: str = "", agent_name: str = "general_subagent") -> "AgentConfig":
+    """The caller's config as a focused subagent: same model, tools, env image, budgets and dataset task; the subagent note
+    appended; every delegation tool removed and no program (no recursion); no env setups (the env is the parent's)."""
+    parent_config = parent.agent_config
+    delegation = {name: None for name, tool in parent.tools.items() if isinstance(tool, SubagentTool)}
+    return replace(parent_config, agent_name=agent_name, user_prompt=task, messages_input=[], agentic_program=None,
+                   system_prompt=(parent_config.system_prompt or "") + SUBAGENT_SYSTEM_NOTE,
+                   tools={**parent_config.tools, **delegation}, env_setups={})
@@ SubagentTool._inherit_general_config @@
-        ... (body)
+        return subagent_config_of(self.agent)
```

`activation/agent/rollout_manager.py · _run_one`

```diff
-            result = agent.run()
+            result = agent.run_program()
```

`activation/agent/rollout_caching.py · config_key`

```diff
     label = config.metadata.get("template") if config.metadata else None
+    program = config.metadata.get("program") if config.metadata else None
     if task is not None and label:
-        return f"{task.dataset_id}/{task.task_id}/{label}"
+        return f"{task.dataset_id}/{task.task_id}/{label}" + (f"/{program}" if program else "")
-    digest = hashlib.sha256(json.dumps([config.system_prompt, ..., config.call_kwargs], ...)
+    program_spec = None if config.agentic_program is None else _class_spec(*config.agentic_program)
+    digest = hashlib.sha256(json.dumps([config.system_prompt, ..., config.call_kwargs, program_spec], ...)
```

Elided: `agent/__init__.py` exports (`AgenticProgram`, `MissingClass`, `MissingClassError`); the reporter's label for `programmed`; the `sys` import in `agent_config.py`.

</details>

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">P2 — The basic agentic-program test</span>
    <span class="card-oneliner">One GPU end-to-end test in the style of test_basic_agent.py, program class in the file, DeepMath difficulty-2 problem, plus a CPU serde test.</span>
    <span class="card-badge">Planning</span>
  </summary>

#### Planning Overview

The problem is DeepMath-103K train shard 0, difficulty 2.0, topic Number Theory → Factorization: "How many trailing zeroes are there in 100!?", answer 24, scored by math-verify. It is embedded as a `DatasetTask` so the test needs no network. The program, `SolveThenVerifyProgram`, runs one solver subagent on the plain task (bounded to 120 s), injects the solver's answer ahead of the task with an instruction to check it with python, and runs the main agent. The GPU test runs on a node with the 9B model through the rollout manager (so `run_program` is exercised where it matters), scores, and runs the same call again from the cache. The CPU test covers the placeholder deserialization and the cache key without an engine. Exact file below.

#### Planned Changes

`activation/tests/test_basic_agentic_program.py` · new, complete

```python
"""
One end-to-end agentic-program rollout on a GPU node, in the style of test_basic_agent.py: a program written in this
file runs a solver subagent on a simple DeepMath problem, puts the solver's answer ahead of the task, and runs the main
agent to verify and submit. Then the same request from the cache, and a CPU check that a record deserializes without
this module's program class.

    uv run sky exec --sync <node> -- uv run pytest activation/tests/test_basic_agentic_program.py --gpu --slow -s
"""
import json
import os
from dataclasses import replace

import pytest

from activation.agent import Agent, AgentConfig, AgentRunResult, RolloutReporter
from activation.agent.agent_config import _class_spec
from activation.agent.agentic_program import AgenticProgram, MissingClass, MissingClassError
from activation.agent.rollout_caching import config_key
from activation.common.data_syncing import resolve_path
from activation.dataset import ANSWER_RULES, DatasetTask, DatasetTaskKind, DatasetTaskMetricsKind, bare_prompt
from activation.harness import FREE_DEVICE, HarnessRuntime, HarnessRuntimeConfig, ModelConfig

MODEL_NAME = "qwen3.5-9b"
MODEL_ID = "Qwen/Qwen3.5-9B"

# DeepMath-103K (zwhe99/DeepMath-103K, train shard 0), difficulty 2.0, topic Number Theory -> Factorization.
QUESTION = "How many trailing zeroes are there in 100!?"
TRAILING_ZEROES = DatasetTask(
    task_id="trailing-zeroes-100", dataset_id="deepmath___fixture",
    task_datum={"question": QUESTION, "final_answer": "24", "difficulty": 2.0},
    reference_metrics_kind=DatasetTaskMetricsKind.MATH_VERIFY, gold_answer="24",
    agent_prompt=bare_prompt(QUESTION, ANSWER_RULES["math"]), task_kind=DatasetTaskKind.MATH,
)

BASE_AGENT_CONFIG = AgentConfig(
    # Default tools only (shell, python, parallel_tool_call, submit_answer). Thinking is off by the harness default.
    system_prompt=(
        "You are a careful problem solver working in a sandbox. Use the python or shell tools to compute "
        "rather than guessing. When you are done, call submit_answer exactly once with the final answer."
    ),
    user_prompt=TRAILING_ZEROES.agent_prompt,
    dataset_task=TRAILING_ZEROES,
    model_name=MODEL_NAME,
    call_kwargs={"sampling_params": {"max_tokens": 2048, "temperature": 0.7}},
    max_turns=10,
    max_tool_errors=3,
    max_duration=300,
)


class SolveThenVerifyProgram(AgenticProgram):
    """A solver subagent answers first; the main agent sees that answer ahead of the task, checks it and submits."""
    name = "solve_then_verify"

    def __init__(self, agent: Agent, solver_max_duration: float = 120.0):
        super().__init__(agent)
        self.solver_max_duration = solver_max_duration

    def execute(self) -> AgentRunResult:
        task = self.agent.agent_config.dataset_task
        solver = self.agent.run_subagent(task.agent_prompt, max_duration=self.solver_max_duration)
        self.agent.augment_context([{
            "type": "text",
            "text": (f"A first solver finished ({solver.finish_reason}, {solver.num_turns} turns) with the answer "
                     f"{solver.answer!r}. Check it with the python tool before you submit, and submit the correct value."),
        }])
        return self.agent.run()


PROGRAM_CONFIG = replace(
    BASE_AGENT_CONFIG,
    agentic_program=(SolveThenVerifyProgram, {"solver_max_duration": 120.0}),
    metadata={"program": SolveThenVerifyProgram.name},
)


def _make_harness() -> HarnessRuntime:
    return HarnessRuntime(HarnessRuntimeConfig(
        model_configs={MODEL_NAME: ModelConfig(MODEL_NAME, MODEL_ID)},
        agent_max_concurrent=4,
    ))


def _print_rollout(label: str, result: AgentRunResult) -> None:
    print(
        f"\n=== {label} seed={result.seed} finish={result.finish_reason} turns={result.num_turns} "
        f"tokens in/cached/out={result.num_input_tokens}/{result.num_cached_input_tokens}/{result.num_output_tokens} "
        f"duration={result.duration:.1f}s score={result.score} answer={result.answer!r}"
    )
    for step in result.trajectory:
        if step["role"] == "assistant":
            calls = ", ".join(f"{call['name']}({json.dumps(call['arguments'])[:80]})" for call in step["tool_calls"])
            print(f"  assistant: {step['content'][:200]!r} -> {calls}")
        else:
            for output in step["tool_call_results"]:
                print(f"  tool: {output[:200]!r}")


@pytest.mark.gpu
@pytest.mark.slow
def test_basic_agentic_program_solve_then_verify():
    """The program runs a solver, injects its answer, runs the main agent; scored; then the same call from the cache."""
    harness = _make_harness()
    reporter = RolloutReporter(
        str(resolve_path("AGENT_TEST/program")),
        title="Agent test: agentic program",
        description=f"{MODEL_ID}, solve-then-verify program on a DeepMath difficulty-2 problem, {PROGRAM_CONFIG.max_turns} turns max.",
    )
    try:
        results = harness.rollout_manager.perform_single_rollouts(
            [PROGRAM_CONFIG], seed=0, perform_scoring=True, caching_id="agent_test_program", reporter=reporter,
        )
        loads_before = len(harness.harness_stats.model_loading_times)
        cached = harness.rollout_manager.perform_single_rollouts(
            [PROGRAM_CONFIG], seed=0, perform_scoring=True, caching_id="agent_test_program",
        )
    finally:
        harness.loaded_models[MODEL_NAME].engine_to_device(FREE_DEVICE)

    assert len(results) == 1
    result = results[0]
    assert len(result.subagent_results) == 1, "the program runs exactly one solver"
    solver = result.subagent_results[0]
    _print_rollout("solver", solver)
    _print_rollout("main", result)

    # The solver: a plain run in the same sandbox, no program and no delegation tools, bounded to 120 s.
    assert solver.agent_config.agentic_program is None
    assert solver.agent_config.max_duration <= 120.0
    assert solver.finish_reason in ("submitted", "max_turns", "max_tool_errors", "max_duration", "no_tool_call")

    # The main agent: the solver's line sits ahead of the task text in the first user message.
    first_user = next(message for message in result.prompt_messages if message["role"] == "user")
    content = first_user["content"]
    assert isinstance(content, list) and len(content) >= 2, content
    assert "A first solver finished" in content[0]["text"]
    assert content[-1]["text"].startswith(QUESTION)
    assert result.finish_reason == "submitted", result.finish_reason
    assert any(
        call["name"] in ("python", "shell", "parallel_tool_call")
        for step in result.trajectory for call in step.get("tool_calls", [])
    ), "the main agent did not check with a tool"
    assert result.score == 1.0, (result.answer, result.score)
    assert result.duration >= solver.duration, "run_program measures the whole program"

    # The cache: the same call replays the row without touching the engine; the live program class is restored.
    assert len(harness.harness_stats.model_loading_times) == loads_before
    assert cached[0].answer == result.answer and cached[0].score == result.score
    assert cached[0].agent_config.agentic_program[0] is SolveThenVerifyProgram
    assert len(cached[0].subagent_results) == 1
    assert os.path.exists(os.path.join(reporter.report_folder, "report.tressoir.html"))
    print("\n=== harness stats ===")
    print(json.dumps(harness.harness_stats.summarize(), indent=2))


def test_record_deserializes_without_the_program_class():
    """A record whose program class is not importable reads back with a placeholder that refuses construction and
    re-serializes unchanged; the cache key of the programmed config differs from the plain one."""
    result = AgentRunResult(agent_config=PROGRAM_CONFIG, seed=0, finish_reason="submitted", answer="24", score=1.0)
    row = result.serialize()
    spec = row["agent_config"]["agentic_program"]
    assert spec["class"].endswith(":SolveThenVerifyProgram") and spec["kwargs"] == {"solver_max_duration": 120.0}

    row["agent_config"]["agentic_program"]["class"] = "activation.tests.nowhere:SolveThenVerifyProgram"
    restored = AgentRunResult.deserialize(row)
    placeholder, kwargs = restored.agent_config.agentic_program
    assert isinstance(placeholder, MissingClass) and kwargs == {"solver_max_duration": 120.0}
    with pytest.raises(MissingClassError, match="activation.tests.nowhere:SolveThenVerifyProgram"):
        placeholder(None)
    assert _class_spec(placeholder, kwargs)["class"] == "activation.tests.nowhere:SolveThenVerifyProgram"
    assert restored.answer == "24" and restored.score == 1.0                     # training reads the row as usual

    with pytest.raises(Exception):
        AgentRunResult.deserialize(row, strict=True)                            # a live run must not get a placeholder

    plain_key = config_key(BASE_AGENT_CONFIG)
    assert config_key(PROGRAM_CONFIG) != plain_key
    assert config_key(PROGRAM_CONFIG) == f"{plain_key}/{SolveThenVerifyProgram.name}" or plain_key not in config_key(PROGRAM_CONFIG)
```

Notes: `AgentRunResult.deserialize` gains `strict` and passes it to `AgentConfig.deserialize`. The last assertion covers both key forms (a labelled dataset task appends the program segment; an unlabelled one digests the spec). The GPU test needs the node (`ac-4a-camp` is idle) and runs after P1 lands.

</details>

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">P3 — Docs, round and canon</span>
    <span class="card-oneliner">Round rebuild from the workspace, canon entry moved from planned to applied, STATE update.</span>
    <span class="card-badge">TBD</span>
  </summary>

Rebuild the round from `/workspace` with the new files; update the canon decision to the landed shape and note any drift; record the test results in STATE. Teacher-kind integration and library programs are decided later, when we get there.

</details>
