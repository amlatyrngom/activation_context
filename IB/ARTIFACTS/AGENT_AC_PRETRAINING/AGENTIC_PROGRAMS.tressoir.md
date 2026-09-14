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

**Version 0.3 (P4).** Activation content becomes uniform: every tool result carries `activation_content` (produced always, rendered by an agent with an AC model, recorded either way), parts carry the system prompt and the tools, programs inject tool results through `run_tool` and `augment_context`, and one function rebuilds an AC-bearing message list from a base record for SFT.

| Record field | Written by | Holds |
|---|---|---|
| `TrajectoryStep.tool_results` | `append_tool_results` | the serialized results of a tool step, activation content included |
| `AgentRunResult.injected_input` | `augment_context` (programs; the parent for a subagent) | the results placed ahead of the task in the first user message |
| `TrajectoryStep.messages`, `prompt_messages` | as today | what the model saw, parts inline only when rendered |

## Accepted decisions

- **v0.1 (2026-09-11):** programs run before the first turn and change runtime state only; the program spec joins the cache key; subagent configs drop the program; `score` defaults to `None`; solver results injected the way the subagent tool returns them (text line, plus a `subagent_return` part when an AC model is configured).
- **v0.2 (2026-09-14, from chat):** one program per config (`agentic_program: (type, kwargs) | None`); `Agent.run_program()` is what the rollout manager calls and falls back to `run()`; `execute()` returns the main agent's `AgentRunResult`, and non-trivial workflows that bypass `run_subagent` populate the training trajectories themselves; no teacher kind and no library program yet, without closing the door for teacher campaigns; no share ratios; the first test is a `test_basic_agent.py`-style GPU test with the program class in the test file on a DeepMath problem of difficulty 2 or below; code is written in the workspace and synced to `IB/ARTIFACTS/...` on handoff readiness (canon).

**Accepted (2026-09-14, chat, folded into P4):** (a) subagents take their system prompt from their base config; (b) an activation part always carries the system prompt of the segment it compresses, and its tool definitions (natively through the side template, else appended to the system text): omitting them was a bug; (c) `ToolCallResult` returns `content` and `activation_content`, the latter produced always, rendered only by an agent with an AC model, and readable from the record for SFT; (d) `Agent.to_activation_context()` is the one builder of segment parts; (e) `augment_context` takes tool results, recorded under `AgentRunResult.injected_input`; (f) the subagent's index on its tool result is a record field only.

## Requested Decisions

None open.

**Accepted (2026-09-14, with the plan's approval, the recommended option): `enable_ac_communication` is dropped.** An agent with an AC model renders every channel; old rows that carry the field deserialize with it ignored.

## Milestones

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">P0 — The workspace becomes the development tree</span>
    <span class="card-oneliner">Done: /workspace holds the imported tree with git, the round builder reads from it.</span>
    <span class="card-badge">Review</span>
  </summary>

#### What landed

`/workspace` now holds the work tree's product files plus `activation/extensions/` from the source, committed as c374cf1 on `main` with branch `campaign-stable` and tag `redo-launch-2026-09-11` at that commit. `build_round.py` reads from `/workspace`; the round rebuilt from there (48 product files). The CPU suite passes from the workspace (44 passed, 1 skipped before P1). The canon rule is recorded under Cross-cutting decisions.

#### Drifts, challenges, and unplanned steps

The source repository turned out to be a pre-campaign snapshot: it holds an earlier integration of my round with your `@AI` notes (all already implemented) plus two new items, `agent_programs.py` (the prototype note, superseded by this plan) and the empty `extensions` package. It lacks the campaign-era changes (loaders, tuning, redo, persistent draw). The workspace therefore took the work tree as its base and added only `extensions/`; nothing from the source was lost. The workspace also carried five files present in neither tree (`fla_cache.py`, `fla_configs/`, two `ac_training_*` probes, `deepscaler_preview.py`); they were removed to match the source. `TMP/` was added to `.skyignore` so node uploads stay small (4.3 MB).

#### Validation

`diff -rq` between the work tree and the workspace: only `extensions/` differs. Tests from `/workspace`: 44 passed, 1 skipped.

#### Planning Overview (as planned)

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
    <span class="card-oneliner">Done: AgenticProgram, run_program, the runtime API, the cache key segment, MissingClass placeholders in common/utils.py.</span>
    <span class="card-badge">Review</span>
  </summary>

#### What landed

Commit abac685 on `main`. `activation/common/utils.py` (new): `MissingClassError`, `MissingClass` (keeps module and qualified name, `__call__` raises), `class_spec_name` (`__main__` mapped to the script's module), `resolve_class_name(name, strict)`. `activation/agent/agentic_program.py` (new): `AgenticProgram` with the contract in its docstring. `agent_config.py`: `agentic_program` field, serialized like a tool spec; `AgentConfig.deserialize(..., strict=False)` and `AgentRunResult.deserialize(..., strict=False)` resolve every spec leniently; `_class_spec` / `_resolve_class_spec` delegate to the shared helpers; `score` defaults to `None`. `agent.py`: `program_content` and `reported` slots; `first_user_messages()` places program content ahead of the task text; `run_program()`, `augment_context()`, `run_subagent()`, `set_final_answer()`; `run()` reports the start only once. `agent_tools.py`: `subagent_config_of()` shared by the subagent tool and `run_subagent`, dropping delegation tools and the program. `rollout_manager.py`: `_run_one` calls `run_program()`. `rollout_caching.py`: the program label segment and the program spec in the digest. `agent/__init__.py` exports `AgenticProgram`.

#### Drifts, challenges, and unplanned steps

- A class defined in a module run as `python -m` is importable under its module name, but that import creates a second module object, so identity (`is`) with the running module's class does not hold; the probe checks the name instead. Records read back by pytest-imported modules keep identity, which is what the GPU test asserts through the cache replay (the manager passes the live config).
- `run_program` records the start with the reporter before the program runs, so solvers appear under a known parent; `run()` therefore guards its own start report.
- No mid-run augmentation: `augment_context` asserts the agent has not started.

#### Validation

`activation/bench/agent_probes/agentic_program_probe.py` (CPU, tiny fixture): placeholder deserialization for the program and a tool, strict mode raising, unchanged re-serialization, cache key with and without the program segment, `__main__` mapping, programmed finish with the context ahead of the task, error handling for a raising program, a wrong return and a missing class: PASS. CPU suite from the workspace: 48 passed, 1 skipped (scoring, teacher study, tool shapes, deadline, redo, tuning, env setups, LOFT/Loong, dataset loading). 3b CPU checks pass.

#### Planning Overview (as planned)

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
    <span class="card-oneliner">Written as planned; job 11 failed on the engine-context check (fixed locally); rewritten in P4 on run_tool before the rerun.</span>
    <span class="card-badge">Implementing</span>
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
    <span class="card-title">P4 — Activation content everywhere</span>
    <span class="card-oneliner">Landed: activation content on every tool result, one rendering rule, parts with the frame, run_tool and injected input, the record and its conversion. GPU runs in flight.</span>
    <span class="card-badge">Review</span>
  </summary>

#### What landed

The five rules, as planned, in `activation/common/ac_parts.py` (`activation_part`, `resolve_activation_part`, `tools_in_system_text`, `template_takes_tools`), `agent_tools.py` (`ToolCallResult.activation_content`, `tool`, `subagent_index`, `serialize`/`deserialize`; `injected_content`; truncation, search and the subagent tool always build their parts; `segment_messages_of` and `SHARED_FIELDS` removed), `agent.py` (`to_activation_context`, `nested_in_context`, `render_activation`, `run_tool`, `run_subagent` over it, `augment_context(results)`, `first_user_messages` from `injected_input`, `append_tool_results` recording `tool_results`, `compact` through the same part builder), `agent_utils.py` (`tool_messages(calls, contents)`), `agent_config.py` (`TrajectoryStep.tool_results`, `AgentRunResult.injected_input`), the encoder (`EncodeRequest.tools`, cache key, `part_view_rows` / `encode` / `encode_async` with tools, native side templating with the system-text fallback), the study (segment parts with the frame; `_expand_tree` drops it; traj_qa parts carry the source trajectory's system prompt and tools when the loader kept them), and `agent_training_utils.activation_messages_of(run, config)`. `enable_ac_communication` is gone (old rows ignore it): an agent with an AC model renders every channel. The test program runs on `run_subagent` + `augment_context([solver, note])`; the probe covers `run_tool`, the truncation part with nested parent context, the record round trip and the conversion.

#### Drifts, challenges, and unplanned steps

- Before the first turn an agent has no segment, so a tool-output or search part produced by `run_tool` then holds the tool message alone (`nested_in_context`); the parent-context nesting starts with the first prompt. The subagent tool passes the parent's part even when the parent has no segment yet (a program's solver): the part then holds only the system message and the tools.
- The old content-list form of `augment_context` is gone; a plain string becomes a result of the pseudo-tool `program`, and every injected block is labelled `[tool]` in the first user message.
- The reporter and two older tests read `tool_call_results` (strings); it is still written next to `tool_results`, so nothing else moved.
- The `enable_ac_communication` decision was taken as recommended (drop) with the plan's approval; the field is removed rather than kept inert.
- The multi-line import in `ac_model_utils.py` was corrupted by the first patch (both 3b checks failed on a NameError) and repaired.
- Old campaign rows have neither `tool_results` nor `injected_input`, so `activation_messages_of` leaves them as they are. Deriving subagent parts for old rows from `subagent_results` is a separate record-upgrade step, not written.

#### Validation

- `activation/bench/agent_probes/agentic_program_probe.py`: PASS (serde, runtime, and the new activation-content checks: `run_tool` through the model's path, a 50k-char echo truncated with a `tool_output` part, a recorded tool step with a `parent_context` part whose messages start with the system message and carry the tools, the serialized record restored unchanged, `activation_messages_of` rendering both parts with `ac_name` and the configured ratios for a reader with an AC model and nothing for one without).
- CPU suite from `/workspace`: 48 passed, 13 skipped, 3 failed. The three failures are pre-existing and unrelated (`test_basic_engine` needs a live vLLM engine; `test_basic_harness` calls `simple_chat` with a stale signature and has a `harnes` typo).
- 3b CPU checks (`channels_cpu.py`, `harvest_cpu.py`): PASS with the new expectations (system-first parts with tools, the injected parent note, `subagent_index` on the tool step, search extras recorded raw and rendered with the ratios).
- GPU, `ac-4a-camp`: `test_basic_agentic_program.py` + `test_basic_agent.py` (one job) and `test_basic_agent_ac.py` (queued behind it) are running; results go here when they land.

#### Planning Overview (as planned)

Five rules, each one place in the code.

1. **A tool result is `content` plus `activation_content`.** `content` is what the model reads (text parts; one text part equals `output`). `activation_content` is a list of parts `{type, kind, messages, tools}` produced by every tool on every call, with no look at the agent's AC model: the full text of a truncated output, the search passages beyond `top_k` (retrieved unconditionally), the subagent's final segment. The parts carry no encoder settings; `ac_name` and the compression ratio are attached by whoever renders them, from the reader's config by `kind`.
2. **One rendering rule, on the agent.** `Agent.render_activation(parts)` deep-copies the parts, fills `ac_name` and `compression_target` by kind (recursively for nested parent context) and returns them when the agent has an AC model, `[]` otherwise. `append_tool_results` builds each tool message as rendered activation parts followed by the result's content (one ordering, replacing today's two). The first user message is built the same way from injected results (rule 4).
3. **Parts carry the frame.** `Agent.to_activation_context(kind)` returns a part whose `messages` are the agent's current segment **with its system message first** and whose `tools` are its tool definitions. Every channel that compresses a segment (subagent prompt and return, compaction, the parent context nested in tool-output and search parts) builds its part through it. The encoder templates a part with `tools=` natively (the side model is a Qwen); a template without tool support gets the definitions appended to the system text. The AC study's trajectory items (traj_qa) build their parts the same way from the source trajectory's system prompt; passage parts (rag_qa) have no frame to carry. Subagents take their system prompt from their base config: the general subagent's base config is the parent's config with the subagent note appended once, nothing else synthesized.
4. **Programs inject tool results.** `Agent.run_tool(tool, **arguments)` runs a configured tool by name, or a tool instance the program built, through `execute_tool_call`, so program calls and model calls yield identical results (truncation, sandbox files, error wrapping, activation content). `augment_context(results)` appends `ToolCallResult`s (a plain string becomes a result of the pseudo-tool `program`) to `run_results.injected_input` and nothing else. At `begin()` the first user message is `messages_input`, then one block per injected result (rendered activation parts, then a text part naming the tool and holding the output), then the task text. The subagent tool hands the parent's context to the child through the same call: `child.augment_context(ToolCallResult(tool="parent", output=SUBAGENT_INTRO, activation_content=[parent.to_activation_context("subagent_prompt")]))`.
5. **The record keeps activation content whether or not it was rendered.** `TrajectoryStep.tool_results` stores the serialized results (`tool`, `output`, `is_error`, `is_final`, `content`, `activation_content`, `subagent_index`); `AgentRunResult.injected_input` stores the injected ones. `messages` and `token_ids` stay what the model saw. One function, `activation_messages_of(run, config)` in `agent_training_utils.py`, rebuilds a run's messages with the parts rendered from those two fields, for SFT conversion of base runs; the trainer's row integration stays deferred as today.

Boundaries. Compaction keeps its own prompt construction (its part comes from `to_activation_context("compaction")`, and the compacted segment is already in `compactions`); the text-only fallback stays a trailing user message. `subagent_index` is a record field only, never rendered. `tool_call_results` (the old list of strings) stays readable on old records; new records write `tool_results`. The old content-list form of `augment_context` goes away (P2's test is rewritten). Record growth: the full text of truncated outputs, capped at `AC_OUTPUT_LIMIT_CHARS` as today, on about one call in a hundred in the campaign corpus; subagent segments are copied into the parent's tool step once.

Already applied from the P2 failure: `_is_context_overflow` ignores `AssertionError` (the engine-context check named `max_model_len` and was misfiled as `context_exceeded`); the program test sets `absolute_trajectory_cap=40_000` for the 9B engine's 52k context. `test_basic_agent.py` has the same latent mismatch and gets the same line.

**Data flow per channel**

| Channel | Producer | Part messages | Reader placement | Recorded in |
|---|---|---|---|---|
| truncated tool output | `_finalize_result` | parent context (nested part) + full output as a tool message | tool message, before the visible text | `tool_results[i].activation_content` |
| search extras | `SearchTool` | parent context + passages beyond `top_k` | tool message | same |
| subagent return | `SubagentTool` | child's final segment, system first, child's tools | tool message | same, with `subagent_index` |
| subagent prompt | `SubagentTool` → child | parent's segment so far, system first, parent's tools | child's first user message, before the brief | child's `injected_input` |
| program injection | `augment_context` | whatever the tool produced | first user message, before the task | `injected_input` |
| compaction | `compact()` | the finished segment, system first | the new segment's trailing user message | `compactions[k]` (unchanged) |

#### Planned Changes

`activation/agent/agent_tools.py · ToolCallResult, truncate_output, SearchTool, SubagentTool, ParallelCallTool`

```diff
@@ activation/agent/agent_tools.py — ToolCallResult @@
 @dataclass
 class ToolCallResult:
     output: str                                          # what the model sees as text (already truncated)
     is_error: bool = False
     is_final: bool = False                               # submit_answer sets it
-    content: list[dict] | None = None                    # ordered content parts (text and activation_context) when the result carries
-                                                         # activation context; None means [text(output)]. Exactly one text part equals output.
+    content: list[dict] | None = None                    # text parts; None means [text(output)]. Exactly one text part equals output.
+    activation_content: list[dict] = field(default_factory=list)   # parts {type, kind, messages, tools}: what the reader may see compressed;
+                                                         # produced always, rendered by the agent when it has an AC model, recorded either way
+    tool: str = ""                                       # the tool that produced it (set by the agent; "program" / "parent" for injections)
+    subagent_index: int | None = None                    # record only: the child's index in subagent_results (never rendered)
     compacted: bool = False
+
+    def serialize(self) -> dict: ...                     # the fields above, activation_content deep-copied
+    @classmethod
+    def deserialize(cls, data: dict) -> "ToolCallResult": ...
+
+
+def activation_part(kind: str, messages: list[dict], tools: list[dict] | None = None) -> dict:
+    """A part without encoder settings: the agent's render_activation fills ac_name and compression_target by kind."""
+    return {"type": AC_PART_TYPE, "kind": kind, "messages": messages, "tools": tools or []}

@@ activation/agent/agent_tools.py — truncate_output @@
-def truncate_output(env, text, call_id, limit=..., agent=None) -> tuple[str, dict | None]:
+def truncate_output(env, text, call_id, limit=..., agent=None) -> tuple[str, dict | None]:
     ...
-    if agent is None or agent.ac_model is None:
-        return truncated, None
-    config = agent.agent_config
-    part = ac_part([{"role": "user", "content": [agent.context_part(config.ac_subagent_ratio)]}, {"role": "tool", "content": text}],
-                   config.ac_model_name, config.ac_tool_output_ratio, kind="tool_output")
+    if agent is None:
+        return truncated, None
+    part = activation_part("tool_output", [{"role": "user", "content": [agent.to_activation_context("parent_context")]},
+                                           {"role": "tool", "content": text}])
     return truncated, part

@@ activation/agent/agent_tools.py — SearchTool.execute @@
-        extra = min(SEARCH_EXTRA_MAX, 2 * k) if self.agent.ac_model is not None else 0
+        extra = min(SEARCH_EXTRA_MAX, 2 * k)                                          # always retrieved: the part is recorded either way
         chunks = index.bm25_query_many_frozen([query], top_k=k + extra)[0]
         ...
-        content = None
-        if chunks[k:]:
-            part = ac_part([...], config.ac_model_name, config.ac_search_ratio, kind="search")
-            content = [{"type": "text", "text": output}, part]
-        return ToolCallResult(output=output, content=content)
+        activation = []
+        if chunks[k:]:
+            activation = [activation_part("search", [{"role": "user", "content": [self.agent.to_activation_context("parent_context")]},
+                                                     {"role": "tool", "content": blocks(chunks[k:])}])]
+        return ToolCallResult(output=output, activation_content=activation)

@@ activation/agent/agent_tools.py — SubagentTool.execute @@
         subagent_config.user_prompt = task
-        share_ac = parent_config.enable_ac_communication and parent.ac_model is not None
-        if share_ac:
-            for name in self.SHARED_FIELDS: ...
-            subagent_config.messages_input = [{"role": "user", "content": [{"type": "text", "text": SUBAGENT_INTRO}, ac_part(...)]}]
         subagent = Agent(...)
+        subagent.augment_context(ToolCallResult(tool="parent", output=SUBAGENT_INTRO,
+                                                activation_content=[parent.to_activation_context("subagent_prompt")]))
         result = subagent.simulated_run() if parent.step_mode else subagent.run()
         with parent.lock:
             parent.run_results.subagent_results.append(result)
+            index = len(parent.run_results.subagent_results) - 1
         output = (f"Subagent finished ({result.finish_reason}, {result.num_turns} turns). Answer: ...")
-        content = None
-        if share_ac:
-            final_segment = segment_messages_of(result)
-            content = [ac_part(final_segment, ..., kind="subagent_return"), {"type": "text", "text": output}]
-        return ToolCallResult(output=output, is_error=result.finish_reason == "error", content=content)
+        return ToolCallResult(output=output, is_error=result.finish_reason == "error", subagent_index=index,
+                              activation_content=[subagent.to_activation_context("subagent_return")])
+# segment_messages_of() and SHARED_FIELDS are removed; the AC fields of the child come from subagent_config_of (the parent's config).

@@ activation/agent/agent_tools.py — ParallelCallTool.execute @@
-        content: list[dict] = []
-        for call, result, labeled in zip(nested, results, outputs):
-            for item in tool_content(result): ...                                # interleaving text and parts by hand
-        ...
-        return ToolCallResult(output="\n\n".join(outputs), is_error=..., content=content if has_parts else None, is_final=...)
+        return ToolCallResult(output="\n\n".join(outputs), is_error=any(r.is_error for r in results),
+                              is_final=any(r.is_final for r in results),
+                              activation_content=[part for r in results for part in r.activation_content])   # members' parts in call order
```

`activation/agent/agent.py · to_activation_context(), render_activation(), run_tool(), run_subagent(), augment_context(), first_user_messages(), append_tool_results(), _finalize_result(), compact()`

```diff
@@ activation/agent/agent.py — segment as a part @@
-    def segment_messages(self) -> list[dict]:
-        """The current segment without the system message: what a part sees as 'the parent so far'."""
-        return [message for message in self.messages if message.get("role") != "system"]
-
-    def context_part(self, ratio: float) -> dict:
-        return ac_part(deepcopy(self.segment_messages()), self.agent_config.ac_model_name, ratio, kind="parent_context")
+    def to_activation_context(self, kind: str) -> dict:
+        """This agent's current segment as a part: system message first, then the segment verbatim; its tool definitions ride as `tools`."""
+        return activation_part(kind, deepcopy(self.messages), deepcopy(self.tool_definitions))
+
+    RATIO_BY_KIND = {"compaction": "ac_compaction_ratio", "tool_output": "ac_tool_output_ratio", "search": "ac_search_ratio",
+                     "subagent_prompt": "ac_subagent_ratio", "subagent_return": "ac_subagent_ratio", "parent_context": "ac_subagent_ratio"}
+
+    def render_activation(self, parts: list[dict]) -> list[dict]:
+        """The parts with this reader's encoder settings (recursively), or [] when this agent has no AC model."""
+        if self.ac_model is None or not parts:
+            return []
+        return [resolve_activation_part(part, self.agent_config) for part in parts]      # deep copy + ac_name + compression_target by kind

@@ activation/agent/agent.py — first_user_messages @@
     def first_user_messages(self) -> list[dict]:
-        """messages_input with the task appended as a trailing text part of the last user message (today's prompt when empty)."""
+        """messages_input, then one block per injected result (rendered parts, then '[tool]\n' + output), then the task text."""
         config = self.agent_config
         messages = deepcopy(config.messages_input)
         prompt = self._offloaded_prompt(config.user_prompt)
-        lead = list(self.program_content)
+        lead = injected_content(self.run_results.injected_input, self.render_activation)   # shared with activation_messages_of()
         ...

@@ activation/agent/agent.py — tool results @@
     def append_tool_results(self, calls, results, tool_seconds=None) -> None:
-        tool_messages = self.dialect.rendering.tool_messages(calls, results)
+        contents = [self.render_activation(result.activation_content) + tool_content(result) for result in results]
+        tool_messages = self.dialect.rendering.tool_messages(calls, contents)
         ...
-        step = TrajectoryStep(role="tool", ..., tool_call_results=[result.output for result in results], messages=tool_messages, ...)
+        step = TrajectoryStep(role="tool", ..., tool_results=[result.serialize() for result in results], messages=tool_messages, ...)

     def _finalize_result(self, call, result) -> ToolCallResult:
+        result.tool = call["name"]
         visible, part = truncate_output(self._env, result.output, call["id"], agent=self if not result.compacted else None)
         if visible != result.output:
-            ... result.content = ([part] if part is not None else []) + content
+            ... (the text part becomes `visible`)
+            result.activation_content = ([part] if part is not None else []) + list(result.activation_content)
         return result

@@ activation/agent/agent.py — programs @@
-    def augment_context(self, content: list[dict]) -> None:
-        assert not self.started, "augment_context runs before the first turn"
-        self.program_content.extend(deepcopy(content))
+    def augment_context(self, results: "ToolCallResult | str | list[ToolCallResult | str]") -> None:
+        """Tool results (or plain text, as results of the pseudo-tool "program") that open the first user message; before the first turn only."""
+        assert not self.started, "augment_context runs before the first turn"
+        for item in (results if isinstance(results, list) else [results]):
+            result = ToolCallResult(tool="program", output=item) if isinstance(item, str) else item
+            self.run_results.injected_input.append(result.serialize())
+
+    def run_tool(self, tool: "str | AgentTool", **arguments) -> ToolCallResult:
+        """A tool call made by a program: the configured tool `name`, or an instance the program built; same path as a model call."""
+        call = {"id": uuid.uuid4().hex[:8], "name": tool if isinstance(tool, str) else tool.name, "arguments": arguments}
+        if isinstance(tool, str):
+            return self.execute_tool_call(call)
+        try:
+            result = tool.execute(**tool.normalize_arguments(arguments))
+        except Exception as error:
+            result = ToolCallResult(output=f"{tool.name} failed: {type(error).__name__}: {error}", is_error=True)
+        return self._finalize_result(call, result)

-    def run_subagent(self, brief, max_duration=None, agent_name="program_subagent") -> AgentRunResult:
-        ... (builds the child, runs it, appends to subagent_results)
+    def run_subagent(self, brief, max_duration=None, agent_name="program_subagent") -> ToolCallResult:
+        """Convenience over run_tool: a subagent tool on this agent's inherited config, bounded to `max_duration`."""
+        base = subagent_config_of(self, agent_name=agent_name)
+        if max_duration is not None:
+            base.max_duration = min(float(base.max_duration), float(max_duration))
+        return self.run_tool(SubagentTool(self.harness, self, base_config=base), task=brief)

@@ activation/agent/agent.py — compact @@
-        if self.ac_model is not None:
-            tree = ac_part(deepcopy([inner] + segment), config.ac_model_name, config.ac_compaction_ratio, kind="compaction")
+        tree = self.render_activation([self.to_activation_context("compaction")])       # [] without an AC model
+        if tree:
-            after = {"role": "user", "content": [tree, {"type": "text", "text": "\n\n" + text}]}
+            after = {"role": "user", "content": tree + [{"type": "text", "text": "\n\n" + text}]}
```

`activation/agent/agent_utils.py · tool_messages()` — both renderings take the per-result content lists instead of results:

```diff
-    def tool_messages(self, calls: list[dict], results: list["ToolCallResult"]) -> list[dict]:
-        from .agent_tools import tool_content
-        return [{"role": "tool", "content": tool_content(result)} for result in results]
+    def tool_messages(self, calls: list[dict], contents: list[list[dict]]) -> list[dict]:
+        return [{"role": "tool", "content": content} for content in contents]
```

`activation/agent/agent_config.py · TrajectoryStep, AgentRunResult`

```diff
 class TrajectoryStep:
     ...
-    tool_call_results: list[str] = field(default_factory=list) # truncated outputs, same order as tool_calls
+    tool_call_results: list[str] = field(default_factory=list) # old records only: truncated outputs, same order as tool_calls
+    tool_results: list[dict] = field(default_factory=list)     # serialized ToolCallResults, same order as tool_calls: output, content,
+                                                               # activation_content (recorded whether or not it was rendered), subagent_index
 class AgentRunResult:
     ...
+    injected_input: list[dict] = field(default_factory=list)   # serialized ToolCallResults a program (or the parent, for a subagent)
+                                                               # placed ahead of the task in the first user message
```

`activation/common/ac_parts.py · activation_part(), resolve_activation_part()` — the raw part factory moves here (shared by tools, agent and study); `resolve_activation_part(part, config)` fills `ac_name` and `compression_target` by `kind` recursively. `ac_part()` stays for the study's passage parts.

`activation/ac_model/ac_model.py · part_view_rows(), encode(), encode_async(), row_cache_key()` — every entry takes the part's `tools` and passes them to `tokenize_with_parts(..., tools=tools)`; a side template without tool support gets the definitions rendered into the system message text (`InlineRendering.system_prompt`); the cache key includes the tools.

```diff
-    def part_view_rows(self, messages, compression_ratio=None) -> int:
+    def part_view_rows(self, messages, compression_ratio=None, tools=None) -> int:
         ...
-        ids, _ = tokenize_with_parts(self.side.tokenizer, messages, child_lengths, self.pad_id)
+        ids, _ = tokenize_with_parts(self.side.tokenizer, messages, child_lengths, self.pad_id, tools=self._side_tools(tools))
```

`activation/ac_model/ac_model_study.py · traj_qa items` — a trajectory window part carries the source run's system message first and its tool definitions (`activation_part("trajectory", [system] + window, tools)`); `_expand_parts` renders that system line into the teacher's transcript. Passage parts (rag_qa) are unchanged.

`activation/agent_training/agent_training_utils.py · activation_messages_of()`

```diff
+def activation_messages_of(run: AgentRunResult, config: AgentConfig) -> list[dict]:
+    """
+    The run's final segment as messages with every recorded activation part rendered for `config` (its ac_model_name and
+    ratios): the first user message rebuilt from injected_input, each tool step's message from its tool_results. Base runs
+    (no parts rendered at run time) come out AC-bearing; AC runs come out as recorded. Token ids are not produced here.
+    """
```

`activation/tests/test_basic_agentic_program.py · SolveThenVerifyProgram, the test` — the program on `run_tool`:

```diff
     def execute(self) -> AgentRunResult:
         task = self.agent.agent_config.dataset_task
-        solver = self.agent.run_subagent(task.agent_prompt, max_duration=self.solver_max_duration)
-        self.agent.augment_context([{"type": "text", "text": (f"A first solver finished ({solver.finish_reason}, ...")}])
+        solver = self.agent.run_subagent(task.agent_prompt, max_duration=self.solver_max_duration)   # a ToolCallResult
+        self.agent.augment_context([solver, "Check the solver's answer above with the python tool before you submit, and submit the correct value."])
         return self.agent.run()
```

and the assertions: one child in `subagent_results`, `solver.subagent_index == 0`, `solver.activation_content` is one `subagent_return` part whose messages start with the child's system message and whose `tools` name the child's tools; `injected_input` holds two entries; the first user message reads `[subagent]` block, then the program note, then the task text last; no `activation_context` part rendered (no AC model); the cached replay restores `injected_input` and the program class.

`activation/bench/agent_probes/agentic_program_probe.py · check_runtime(), check_conversion()` — on the tiny fixture: `run_tool("python", code=...)` from a program returns a finalized result with `tool == "python"`; a long output yields a `tool_output` part with nested parent context whose messages start with the system message; `activation_messages_of(run, ac_config)` on the recorded base run yields a first user message with a rendered `activation_context` part and a tool message with the `tool_output` part; the record's `messages` are unchanged.

`IB/TMP/SLICE3B/cpu_checks/channels_cpu.py, harvest_cpu.py` — updated to the new shapes: the child's first user message is one message (injected block + task) instead of intro-message plus task-message; parts start with a system message; the harvest teacher transcript gains the system line.

#### Validation plan

- CPU: the probe (runtime, conversion), the CPU suite, the 3b checks.
- GPU (`ac-4a-camp`, 9B): `test_basic_agentic_program.py`, then `test_basic_agent.py` with the cap line, then the slice-3b GPU channel test with an AC model so that rendered parts (system first, tools) still encode and the cache key changes are exercised.
- Record check on one campaign row: `activation_messages_of` on a base row with a subagent yields rendered `subagent_prompt` and `subagent_return` parts, and the truncated-output channel stays text-only on old rows (their full text was never recorded).

</details>

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">P3 — Docs, round and canon</span>
    <span class="card-oneliner">Round rebuild from the workspace, canon entry moved from planned to applied, STATE update.</span>
    <span class="card-badge">TBD</span>
  </summary>

Rebuild the round from `/workspace` with the new files; update the canon decision to the landed shape and note any drift; record the test results in STATE. Teacher-kind integration and library programs are decided later, when we get there.

</details>
