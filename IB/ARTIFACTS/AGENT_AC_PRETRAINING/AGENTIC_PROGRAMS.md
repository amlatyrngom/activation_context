# Agentic programs — agent-facing source

Projection: `AGENTIC_PROGRAMS.tressoir.md` (the complete human-facing surface; keep both in agreement). Subplan of slice 4a, folder `IB/ARTIFACTS/AGENT_AC_PRETRAINING/`. Status: Planning v0.2, 2026-09-14; one decision open (serde placeholder shape). Milestones: P0 workspace as development tree, P1 runtime + config + lenient serde, P2 the basic test (exact file in the projection), P3 docs.

## Version 0.2 (2026-09-14) — user refinements, all accepted

1. Programs are central: `Agent.run_program()` is the entry point the rollout manager calls (`_run_one`); it falls back to `run()` when the config has no program.
2. One program per config: `AgentConfig.agentic_program: tuple[type[AgenticProgram], dict] | None` (was a dict of many).
3. `AgenticProgram.execute()` returns the main agent's own `AgentRunResult`; `run_program` checks identity, records the duration over the whole program, reports finish. Programs that drive agents without `run()`/`run_subagent` must populate the trajectories for SFT/RL themselves.
4. No teacher kind, no library program (`ParallelSolveProgram` dropped), no share ratios in this pass; stay compatible with teacher campaigns (configs without a program are unchanged; keys unchanged without a program label).
5. The target is a `test_basic_agent.py`-style GPU end-to-end test with the program class in the test file, on a DeepMath problem of difficulty <= 2: "How many trailing zeroes are there in 100!?" (train shard 0, difficulty 2.0, Number Theory -> Factorization, answer 24), embedded as a DatasetTask (no network).
6. Workflow: code is written in `/workspace` (git) and synced to `IB/ARTIFACTS/AGENT_AC_PRETRAINING/slice4a/` by the round builder at handoff readiness; recorded in canon (P0).
7. Serde: lenient deserialization so training reads rows without the program/tool/env classes present, erring only on construction. Open decision in the projection: `MissingClass` placeholder carrying the spec (recommended; round-trips unchanged, names the missing class in the error, uniform over tools/env setups/program, `strict=True` for live callers) vs a bare `serde_missing` function vs lazy specs.

Superseded from v0.1: `agentic_programs` dict, `_run_programs` loop inside `run()`, `run_subagents` helper, `ParallelSolveProgram`, `PROGRAM_PARALLEL` teacher kind, the "no tolerant deserialization" rule (replaced by the placeholder decision once accepted).

## Settled by canon (do not reopen)

`CANON/ROOT_CANON.md` → Agent rollout decisions → "Agent programs (planned)":
programs are config (`agentic_programs: dict[str, tuple[class, kwargs]]`, tool-style spec and serialization); run inside `Agent.run()` after the clock starts and before `begin()`; runtime state only (extra first messages ahead of the task, `run_subagent`, `set_final_answer` with finish reason `programmed`); the program spec joins the cache key; subagent configs drop programs; serde keeps "same class exists at read time" with the single `__main__` → module-name normalization in `_class_spec`; keep prompt-elicited multi-agent teachers alongside programs; `score` defaults to `None`.

Names per the user (2026-09-11): class `AgenticProgram`, module `activation/agent/agentic_program.py`.

## Research notes (code as of commit 7b1002e in `IB/TMP/AGENT_AC_PRETRAINING/work`)

- `Agent.run()` (`agent/agent.py` ~369): `start = time.time()`; `self.begin()`; `reporter.report_agent_start(self)`; `_check_engine_context()`; loop. The hook goes after the clock start and the reporter registration, before `begin()`. `begin()` is idempotent (`if self.started: return`).
- `first_user_messages()` (~181): `messages_input` deep-copied; the offloaded task text appended as the trailing text part of the last user message, or as a new user message. Program content is inserted as content items before the task text in that message.
- `_begin_segment()` (~231): templates once; `prompt_token_ids`, `prompt_messages`, `prompt_ac_spans` recorded; rows encoded from parts (`_encode_parts`). Activation parts in program content are encoded here like any first-message part.
- `SubagentTool.execute` (`agent/agent_tools.py` ~340): builds the child config via `_inherit_general_config()` (`replace(parent_config, agent_name=..., user_prompt="", messages_input=[], system_prompt=+SUBAGENT_SYSTEM_NOTE, tools={**tools, **delegation-as-None}, env_setups={})`), shares `parent.agent_env`, runs `subagent.run()`, appends under `parent.lock` to `run_results.subagent_results`, returns text + optional `subagent_return` part via `segment_messages_of(result)` and `ac_part(..., kind="subagent_return")`. `run_subagent` reuses this by extracting `subagent_config_of(parent, task, agent_name)` as a module function and adding `agentic_programs={}`.
- `AgentConfig.serialize/deserialize` (`agent/agent_config.py` ~60–90): tools and env setups go through `_class_spec` / `_resolve_class_spec` (`module:Qualname` + kwargs; `importlib.import_module`). Programs follow the same path. The cache replay passes the live config (`AgentRunResult.deserialize(row, harness, agent_config=config)`), so no resolution at replay.
- `config_key` (`agent/rollout_caching.py` ~43): dataset task with `metadata["template"]` → `dataset_id/task_id/template`; else digest of prompts/model/lora/messages_input/ac/call_kwargs. Program label appended as a fourth segment when present; free-form digests include `agentic_programs` specs (they are in the serialized config; add them to the digest list).
- `AgentRunResult.score: float | None = 0.0` (~140). Flip to `None`; `Agent.score()` already writes `None` for unscored; reporter, summary, tuning skip `None`.
- Reporter (`agent/rollout_reporter.py` 73–92): `report_agent_start` registers the running item; children call the same with `parent_agent` set. Registering the parent before programs keeps the tree readable.
- Study (`agent_training/agent_training_teacher_study.py` ~193–216): `_config_for(name, task, teacher_kind, index)` builds the config with `metadata` labels; `draw_entry` records benchmark, dataset_id, task_id, teacher_kind, template; `_configs_for_entries` rebuilds. The program label derives from the teacher kind (`PROGRAMS[kind]`), so the draw record needs no new field, but recording `program` explicitly is harmless and clearer.
- Evidence for the approach (12 h campaign, `node_camp/campaign_12h/summary.json`, 4,341 runs): submitted-row mean score 0.707 for base, sequential and parallel alike; 30 min cuts 3% / 10% / 14%; per-sequence decode ~11 tok/s at ~125 live sequences; the redo at 16 agents/GPU doubled subagent speed (24.5 tok/s) and the redone rows score 0.49 where they scored 0.

## Plan (mirror of the projection)

P1 — Program runtime in the agent (Planning): `agentic_program.py` with `AgenticProgram` and `ParallelSolveProgram`; `AgentConfig.agentic_programs` + serde; `_class_spec` `__main__` normalization; `Agent.program_content`, `_run_programs`, `augment_context`, `run_subagent`, `run_subagents`, `set_final_answer`; `first_user_messages` places program content ahead of the task; `run()` hook and `programmed` early return (still calls `begin()` so the row records the prompt); `subagent_config_of` shared with the tool, drops programs; `config_key` program segment; `score` default `None`; tests `test_basic_agentic_program.py` (tiny fixture, `_submit` monkeypatched for the child run).

P2 — Teacher kind and node smoke run (Planning): `AgentTeacherKind.PROGRAM_PARALLEL` (weight 0 default), `SYNTHESIS` template for every task kind, `PROGRAMS` map in the study, `metadata["program"]`, draw record round trip, study test; node smoke `program_smoke_v1` (30 tasks, `--teacher-weights all:program_parallel=1`, `--agents-per-gpu 16`) after the redo run; read rows for parts/subagents/scores; harvest CPU check on the rows.

P3 — Docs, round, canon (TBD).

## Open decisions (in the projection)

1. Solver result injection: ACCEPTED 2026-09-11 (projection interaction `programs.output.mirror_tool`): mirror the subagent tool.
2. Solver time bound: fraction of the parent budget, `solver_share=0.6` (recommended) / absolute seconds.
3. Where exercised: teacher kind in the study (recommended) / separate probe script.

## Risks and notes

- Program subagents run before the parent's `begin()`, so the parent has no replica yet; children pick their own replica (as the tool's children do when the parent has none). Prefix-cache locality between parent and children is not a goal here.
- `run_subagents` uses a thread pool sized to the briefs; the manager's pool bound counts the parent only (as with the tool), so the engine sees pool x (1 + solvers) live sequences at worst. The smoke run uses 16 agents/GPU for that reason.
- A program's own wall time counts against the parent's `max_duration` (clock started first) and the wall deadline's clipping applies through the parent's clipped config.
- No mid-run augmentation; no recursion; no tolerant deserialization (canon).
