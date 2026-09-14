# Slice 4a hand-off

Everything the agent AC pre-training slice delivered as of 2026-09-14, in one place: where the code, data and documents are, what the system does now, how the records look, how to run it, what was validated, and what is open. The exact patch against your source is in the round document; this file is the map.

## Where things are

| What | Where | Notes |
|---|---|---|
| Development tree | `/workspace` (git, branch `main`, head `dce8a1c`) | `campaign-stable` branch and tag `redo-launch-2026-09-11` mark the code the 12 h campaign ran. `IB/TMP` is ignored by git. |
| Exact delta against `/source` | `IB/ARTIFACTS/AGENT_AC_PRETRAINING/slice4a/` (`changes.patch`, `source_manifest.json`, staged files) and `SLICE4A_ROUND.tressoir.md` | 56 product files. Regenerate with `IB/TMP/AGENT_AC_PRETRAINING/build_round.py` after any edit in `/workspace`. |
| Plans and reports | `PLAN.tressoir.md` (slice plan, M0–M4 in Review), `AGENTIC_PROGRAMS.tressoir.md` (P0–P4 in Review), `ROLLOUT_TUNING.tressoir.md`, `PROBE_REPORT.tressoir.md`, `PROBE_COMPARE.tressoir.md`, `TUNING_REPORT.tressoir.md` | Same folder as this file. Sources `.md` next to each projection. |
| Durable decisions | `IB/CANON/ROOT_CANON.md` | Workflow rule, agentic programs, activation content, `score` default, rollout cache rules. |
| Status | `IB/STATE.md` | Newest section first. |
| Teacher corpus | `IB/TMP/AGENT_AC_PRETRAINING/node_camp/ROLLOUTS/teacher_campaign_12h/` (`rollouts.jsonl` 4.7 GB, `draw.json`, `draw_tasks.jsonl`) | Also on the node under `~/activation_artifacts/SYNC/ROLLOUTS/teacher_campaign_12h/`. Reports in `node_camp/campaign_12h/` and `node_camp/redo_12h/`. |
| Tuning configurations | `IB/TMP/AGENT_AC_PRETRAINING/node_camp/TUNING/` (`campaign_27b_medium_v2.json`, `smoke_v1.json`) | Saved autotune ids; `--autotune-id` reloads them. |
| Node logs | `IB/TMP/AGENT_AC_PRETRAINING/node_*.log` | One per job; the remote copy is `~/sky_logs/<job>-sky-cmd/run.log`. |
| Test reports | `IB/TMP/AGENT_AC_PRETRAINING/node_camp/{program_test,ac_test,probe_v3,smoke_*}/` | Pulled `report.tressoir.html` folders. |
| Cloud node | `ac-4a-camp` (AWS ap-northeast-1, g7e.12xlarge, 2x RTX PRO 6000, autostop after idle) | Kept alive; its disk holds the campaign cache and the model weights. Tear down after pulling anything you still need. |
| CPU fixture | `IB/TMP/AGENT_ROLLOUTS/slice3_apply_20260909T103843Z/tiny_qwen35` | Tiny Qwen3.5 used by the probes and the 3b checks; no docker needed. |
| S3 upload | `IB/TMP/AGENT_AC_PRETRAINING/upload_campaign_to_s3.py` | Needs `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` and `boto3`; blocked here (no credentials visible to the agent container). |

## What the system does now

**Rollouts.** A `DatasetTask` (bare prompt with an answer rule, `task_kind`, optional env setups and a searchable corpus) becomes an `AgentConfig`; the rollout manager runs agents on threads against vLLM replicas, one sandbox per top-level agent, subagents in the parent's sandbox. Results are cached per `caching_id` in `rollouts.jsonl` (last row per `(config_key, seed)` wins; the key is `<dataset>/<task>/<template>` plus `/<program>` for labelled tasks, a config digest otherwise). A `RedoPolicy` re-rolls cached rows by finish reason and teacher kind; `--redo-only` runs nothing else. Draws are persistent (`draw_tasks.jsonl`) with a guard (`draw.json`), so a rerun replays the same task sequence even when a loader's task set drifts.

**Teacher campaign.** `activation/bench/canonical_training/generate_agent_teacher_trajectories.py` draws tasks over ten weighted benchmarks (MuSiQue, DAPO, DeepMath, LCA bug localization in Python/Java/Kotlin, NarrativeQA, LOFT, Loong, BRIGHT) and three teacher kinds (base, sequential multi-agent, parallel multi-agent) with the 27B oracle at medium reasoning, under a saved tuned pool configuration and a wall-clock deadline. The corpus below is its output.

**Agentic programs.** `AgentConfig.agentic_program = (ProgramClass, kwargs)`; the manager calls `Agent.run_program()`, which runs the program's `execute()` (must return the agent's own `run_results`) or `run()` without a program. Before the first turn a program may call `run_tool(tool, **args)` (a configured tool by name or an instance, same path as a model call), `run_subagent(brief, max_duration)` (a bounded subagent tool over the inherited config), `augment_context(results)` (tool results or text placed ahead of the task) and `set_final_answer(answer)`. Programs change runtime state only; the program label joins the cache key.

**Activation content.** Every `ToolCallResult` has `content` (what the model reads) and `activation_content` (parts `{kind, messages, tools}` every tool produces: the full text of a truncated output, search passages beyond `top_k`, a subagent's final segment). An agent with an AC model renders them (`render_activation` fills `ac_name` and the ratio by kind) ahead of the text; without one nothing is rendered. Parts carry the frame: the segment's system message first, and its tool definitions, templated natively on the side model. `Agent.to_activation_context(kind)` is the one builder of segment parts (before the first turn: the first prompt as it will be built). The record keeps activation content either way, so a base run converts to an AC-bearing message list with `activation_messages_of(run, config)`.

**Sandbox.** Created under a lock, eagerly by the manager before a live run; lazily elsewhere (cache-served rows, step mode, CPU probes, resume). Subagents receive the parent's.

## Record formats

`AgentRunResult` (one per agent run, children under `subagent_results`, earlier compaction segments under `compactions`):

| Field | Meaning |
|---|---|
| `agent_config` | The config, class specs serialized as `module:Qualname` plus kwargs; unresolvable classes deserialize to `MissingClass` placeholders (readable, not runnable) unless `strict=True`. |
| `prompt_token_ids`, `prompt_messages`, `prompt_ac_spans` | The segment's first prompt: tokens (placeholder runs at part positions), dialect messages with rendered parts inline, the spans. |
| `trajectory` | `TrajectoryStep` dicts: `role`, `content`, `tool_calls`, `tool_call_results` (strings), `tool_results` (serialized `ToolCallResult`s, new), `messages`, `ac_spans`, `token_ids`, `logprobs`, `think_end`, `timing`. |
| `injected_input` (new) | Serialized `ToolCallResult`s placed ahead of the task in the first user message (a program's `augment_context`; the parent's context for a subagent). |
| `answer`, `score` (`None` = unscored), `score_feedback`, `finish_reason`, `num_turns`, token counts, `duration`, `seed`, `source`, `lora_name`, `ac_model_name`, `ac_model_version` | As before; `score` now defaults to `None`. |

`ToolCallResult` (serialized): `tool`, `output`, `is_error`, `is_final`, `content`, `activation_content`, `subagent_index` (record only), `compacted`.

**Temporary format mismatch.** The corpus rows were written before this round: they have no `tool_results` and no `injected_input`, their parts (where an AC model was on, which the campaign never used) carried no system message or tools, and their configs carry the removed `enable_ac_communication` flag (ignored on read). New rows carry everything. `activation_messages_of` leaves old rows as they are. Whether to upgrade old rows (subagent parts can be rebuilt from `subagent_results`; truncated tool outputs and search extras cannot, their text was never recorded) or to regenerate is a decision that depends on what training the models need, so it is deliberately open.

## The teacher corpus

| Item | Value |
|---|---|
| Unique trajectories (top level) | 4,279 |
| Finish reasons | submitted 4,087, cut by the deadline 99, rest budget-limited or errors |
| Mean score by teacher kind | base 0.709, sequential 0.696, parallel 0.685 |
| Subagent runs inside the rows | 7,570 |
| Tool results / truncated | 127,143 / 1,118 (635 rows carry a truncation) |
| Run | job 8 (12 h, `campaign_12h`) plus the redo pass job 10 (`redo_12h`: max_duration and max_turns rows of the multi-agent kinds, 60 min and 250 calls, 16 agents per GPU) |
| Oracle | `unsloth/Qwen3.8-27B-NVFP4`, medium reasoning, 64k engine context, 50k trajectory cap |

The contention diagnosis (about 11 tokens per second per live sequence at about 125 live sequences; subagents run outside the pool bound; halving the pool doubled subagent speed) is in `PLAN.tressoir.md` M4 and `ROLLOUT_TUNING.tressoir.md`.

## How to run

All commands from `/workspace`. Local Python: `env UV_PYTHON_INSTALL_DIR=$HOME/.local/share/uv/python /workspace/.venv/bin/python`; on the node plain `uv run` (do not carry the local override there).

```bash
# CPU: suite, probe, 3b checks (no docker needed)
python -m pytest -q activation/tests --ignore=activation/tests/test_basic_engine.py --ignore=activation/tests/test_basic_harness.py
python -m activation.bench.agent_probes.agentic_program_probe
PYTHONPATH=. python IB/TMP/SLICE3B/cpu_checks/channels_cpu.py
PYTHONPATH=. python IB/TMP/SLICE3B/cpu_checks/harvest_cpu.py

# GPU tests on the node (the wrapper restarts a STOPPED cluster, uploads the tree, pulls --watch folders)
python activation/cloud/sky.py exec --watch '~/activation_artifacts/SYNC/AGENT_TEST/program/' IB/TMP/AGENT_AC_PRETRAINING/node_camp/program_test \
  --interval 60 ac-4a-camp -- uv run pytest activation/tests/test_basic_agentic_program.py activation/tests/test_basic_agent.py --gpu --slow -s
python activation/cloud/sky.py exec --watch '~/activation_artifacts/SYNC/AGENT_AC_TEST/' IB/TMP/AGENT_AC_PRETRAINING/node_camp/ac_test \
  --interval 120 ac-4a-camp -- uv run pytest activation/tests/test_basic_agent_ac.py --gpu --slow -s

# The campaign as it ran, and the redo pass
uv run python -m activation.bench.canonical_training.generate_agent_teacher_trajectories --total 20000 --seed 0 \
  --caching-id teacher_campaign_12h --report-folder AGENT_AC_PRETRAINING/campaign_12h --reasoning medium \
  --autotune-id campaign_27b_medium_v2 --max-wall-hours 12
uv run python -m activation.bench.canonical_training.generate_agent_teacher_trajectories --total 20000 --seed 0 \
  --caching-id teacher_campaign_12h --report-folder AGENT_AC_PRETRAINING/redo_12h --reasoning medium \
  --autotune-id campaign_27b_medium_v2 --max-duration 3600 --max-turns 250 --agents-per-gpu 16 \
  --redo-finish-reasons max_duration max_turns --redo-teacher-kinds sequential_multi_agent parallel_multi_agent --redo-only

# Node access and job control
ssh -F ~/.sky/generated/ssh/ac-4a-camp -o BatchMode=yes ac-4a-camp 'tail ~/sky_logs/<job>-sky-cmd/run.log'
# stop a job: pkill -f "generate_agent_teacher_trajectorie[s]" on the node, then kill the GPU pids from
#   nvidia-smi --query-compute-apps=pid --format=csv,noheader (orphaned EngineCore processes hold the GPUs)
# cancel through Sky from your shell (the agent cannot): sky cancel ac-4a-camp <job> --yes   (never --all)

# Upload the corpus to S3 (your shell, with AWS credentials in the environment)
cd /workspace && python IB/TMP/AGENT_AC_PRETRAINING/upload_campaign_to_s3.py
```

Two cache rules matter when rerunning a test after a code change: a finished row under the same caching id is replayed as is (finish reason and budgets are not in the key), so clear `~/activation_artifacts/SYNC/ROLLOUTS/<caching_id>/` on the node or use a new id; and a `RedoPolicy` only re-rolls the finish reasons it names.

## Validation status

| Check | Result |
|---|---|
| CPU suite (`activation/tests`, engine and harness files excluded) | 48 passed, 13 skipped |
| `test_basic_engine.py`, `test_basic_harness.py` | 3 pre-existing failures unrelated to this slice: the engine tests need a live vLLM engine, `simple_chat` is called with a stale signature, a `harnes` typo |
| Agentic-program probe | PASS (serde placeholders, cache key, `__main__` mapping, runtime, `run_tool`, truncation part with nested parent context, record round trip, conversion) |
| 3b CPU checks (channels, harvest) | PASS with system-first parts and tools |
| `test_basic_agentic_program.py` (9B, node) | passed, job 18: solver 24 in 2 turns, main verified and scored 1.0, cached replay |
| `test_basic_agent.py` (9B, node) | 2 passed, job 14 |
| `test_basic_agent_ac.py` (9B + AC model, node) | 3 passed, job 17: compaction, subagent, tool-output and search channels, engine probes on prompt embeddings, resume, harvest, text-only fallback |
| Teacher campaign | complete (above) |

## Open items and decisions still yours

- **Record formats.** The mismatch above: upgrade, regenerate, or accept old rows as text-only, once the training recipe is chosen.
- **S3 upload** of the corpus: script ready, needs your credentials.
- **12 h continuation** of the campaign: on hold by your decision; the redo flags and the persistent draw make it a rerun with a longer deadline.
- **Teacher kind for programs**: not added; programs are compatible with the campaign code but no template draws them yet.
- **Training integration**: `build_example` still refuses AC-bearing runs (fixed-row trainer integration deferred since 3b); `activation_messages_of` gives the message side, not token ids.
- **Old-row upgrade step** (subagent parts from `subagent_results`): not written.
- **Two stale tests** in the source (`test_basic_harness.py`): fix or drop.
- **Node**: `ac-4a-camp` is still up with the corpus cache on its disk.

## Lessons recorded

- Killing a job leaves `VLLM::EngineCore` processes holding the GPUs; kill them by pid before the next job or it waits for memory.
- Tar extraction of LCA checkouts must skip absolute symlinks; parallel fetches need a per-checkout lock.
- The 9B engine holds 52k tokens: the default 50k cap plus a 2,048-token turn does not fit; tests set the cap to 40,000. The engine-context assertion names `max_model_len` and was once misfiled as `context_exceeded`.
- `pkill -f` matches its own shell; use the `[s]` trick.
