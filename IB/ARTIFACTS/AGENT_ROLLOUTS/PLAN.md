# Agent rollouts — slice 1: end-to-end rollouts with caching on the harness (agent source of truth)

Written 2026-09-07 from the user's chat request and the state of `/source` versus the workspace.

## What the user asked (chat, 2026-09-07)

- Refocus on the paper's topic: harness–model co-design. The last two days (BRIGHT economics
  ablation, slice 1a) were retrieval micro-optimisation; the useful lesson is the scale needed for
  fine-tuning (a few hundred to a few thousand description examples reach a trained embedder).
- Of the slice-1a work, bring over only the genuine bug fixes.
- M0 = bring over the valuable piece: the Sky and data-sync changes.
- Every later milestone is agent-related only. Retrieval returns when it becomes a harness-design
  question, not a benchmark-percentage question.


## Decisions (round 3, 2026-09-07, from SLICE1_ROUND.interactions.json)

Free-form: "So for the memory limit, I think it should be passed into the env start kwargs right? I guess this
is the difference between hard-kill and soft per-process kill? If so, still take env kwargs, but do an 85%
uv limit if possible. The base agent config should specify this, or the global default you already have,
which is frankly what most runs will likely use." → `AgentConfig.env_memory_limit_mb: int | None = None`
(None = `HarnessRuntimeConfig.agent_env_memory_limit_mb`, 8192); `AgentEnv.__init__` adds `--memory <mb>m`
to `podman run` unless `env_args` has one, falls back without it (class flag `hard_limits_available`) when
podman refuses for lack of cgroups (Sky nodes); `_exec` wraps in `ulimit -v` at 85% of the limit. CPU
check passes; not rerun on a node (the two tests ran before this change; the exec wrapper is the same
shape as the one that carried the 4B probe). User also asked for an application handoff section in the
round document for their workspace agent.

## Decisions (round 2, 2026-09-07, from PLAN.interactions.json)

`agent.interfaces.approve` and `agent.interfaces.adjust` both true; feedback "Mostly lgtm. See two
points though." Free-form:

1. "Merge tools and tool_kwargs into tuple[AgentTool, dict]" → `AgentConfig.tools: dict[str,
   tuple[type[AgentTool], dict]]`; `tool_kwargs` removed; `serialize()` writes `[module:Qualname,
   kwargs]`, `deserialize()` resolves it; `Agent._initialize_tools` unpacks the pair.
2. "You should do a pattern replace of the dapo prompts, to require submit_answer instead of
   Answer:..." → `loaders/dapo_math.py`: `_HEAD_INSTRUCTION` / `_TAIL_INSTRUCTION` regexes over the
   dataset's instruction sentences, replaced by `AGENT_HEAD` / `AGENT_TAIL` (compute in the
   environment; finish with exactly one `submit_answer`). `agent_prompt_from_dapo()` is unit-checked
   on the CPU against real rows: both sentences matched on all of the first 200 rows and no "Answer:" survives.

Chat, same message: "You can directly apply the items and start. Ack back with the final diffs, and
the two test run reports btw. (if you encounter issues around model capability, bump to 27B or
something like that)."

## Decisions (round 1, 2026-09-07, from PLAN.interactions.json)

1. **M0 repair is destructive**: revert the user's uncommitted retrieval stubs to HEAD ("revert back to
   how it was before"); the staged slice-1a tree in `IB/ARTIFACTS/BRIGHT_ECON_ABLATION/activation/` is
   version-controlled in the user's workspace for later adaptation. Kept: the agent stubs, the bench
   move to `bench/agent_probes/`, `scoring.numeric_exact` stub, `DatasetTask` changes and export,
   `pyproject.toml` / `uv.lock` as the user left them. Reverted to HEAD: `dataset_manager.py`,
   `dataset_study.py`, `retrieval_ac.py`, `retrieval_model.py`, `retrieval_trainer.py`,
   `runtime_config.py` (the programmatic knobs and `cache_storage_dir` go; `resolve_path` replaces the
   latter), `vllm_wrapper.py` (M1 rewrites it), `tests/test_basic_dataset_study.py`; `dataset.py`
   partially (DataOrigin split and `programmatic_retrieval_examples` go, `NUMERIC_EXACT` and the
   `DatasetTask` changes stay). Deleted: `dataset_caching.py`, `dataset_study_prompts.py`,
   `dataset_bm25_index.py`, `dataset_study_interface.py`, `tests/test_programmatic_dataset_study.py`,
   `bench/bright_econ_ablation/`. `data_syncing.py` is replaced by the implementation.
2. **Podman only.** The user installed podman on their host (`sudo apt install podman`,
   `podman run docker.io/hello-world` works). Rollouts run on Sky nodes, so M2 also makes the Sky
   image podman-capable: `podman` + `fuse-overlayfs` in the Dockerfile, a storage.conf using
   fuse-overlayfs, and `config: docker: run_options: ["--privileged"]` in the task YAML (SkyPilot
   0.13 allows `('docker', 'run_options')` as a task-level override, checked in
   `sky/skylet/constants.py` and `sky/utils/schemas.py`). The image tag must change when the
   Dockerfile changes (today it is `uv-<lock digest>` only): the tag gains a short Dockerfile digest.
   No local-process backend.
3. **Unified async engine**: `AsyncLLM` per GPU, `submit` for agents, `chat` rebuilt on top.
4. **Tools**: the four defaults, the subagent tool, and a semantic search tool backed by BM25 for
   now (`DatasetIndex.bm25_query_many_frozen` over the task's dataset or a configured one).
5. Free-form: the plan must show the full tests so the interface can be judged; mid-review the user
   asked for the exact expected diff of the tests. The projection carries the full public interfaces
   (`## Interfaces`) and, in the M5 card, `diff -u` of the staged
   `/workspace/activation/tests/test_basic_agent.py` against the user's stub in `/source`.
6. Mid-review (chat): update the podman README for non-privileged use. The workspace `README.md`
   carries a rootless section (subuid/subgid, `podman system migrate`, the Ubuntu 24.04 AppArmor
   user-namespace fallback, a `podman info` check); it is an M0 card. DAPO: 1,791,700 rows are
   exactly 100 copies of 17,917 unique problems (streamed 60k rows: 17,917 unique, first repeat at
   row 17,918).

### Interface additions beyond the user's stubs (each flagged in the projection)

- `AgentConfig.tools: dict[str, tuple[type[AgentTool], dict]]` (round 2: class and constructor kwargs
  merged into one tuple; `tool_kwargs` is gone).
- `AgentRunResult.finish_reason` (`submitted`, `max_turns`, `max_tool_errors`, `max_duration`,
  `no_tool_call` (third tool-less reply in a row; each is nudged with "Call one of the provided tools or submit_answer when done."), `error`) and `AgentRunResult.seed`.
- `ToolCallResult.is_final` (set by `submit_answer`).
- `DatasetTask.agent_prompt: str` (what an agent is asked; the loader fills it; for DAPO the problem
  text without the dataset's "Answer:" instruction).
- `EngineChatOutput.cached_prompt_token_count`, `finish_reason`.
- `HarnessRuntimeConfig.agent_max_concurrent: int = 64`,
  `agent_env_default_image: str = "docker.io/library/python:3.12-slim"`.
- `RolloutManager.perform_*` take `reporter: RolloutReporter | None`.
- Reports and the rollout cache live under `data_syncing.resolve_path(...)`, so `sky exec --sync`
  carries them home while the run goes (the live view the user asked for in the DAPO test).

## Where the trees stand (facts, 2026-09-07)

- `/source` HEAD is `f66e80c Retrieval slice 1 ported` (2026-09-05 13:27). Its working tree carries
  the user's `@AI` notes of 2026-09-05 (the slice-1a spec: origins split, description mode,
  programmatic examples, P+V, `eval`, caching, sync) **and none of the 25 slice-1a cards**. The
  workspace holds the implemented slice 1a.
- New in `/source` since (2026-09-06 22:30 → 2026-09-07 16:04): `activation/agent/` (stubs with
  design notes: `agent_config.py`, `agent_env.py`, `agent_tools.py`, `agent.py`,
  `rollout_manager.py`, `rollout_reporter.py`, empty `rollout_caching.py`), `tests/test_basic_agent.py`
  (two key tests to write in full), `dataset/loaders/dapo_math.py` (stub), `scoring.numeric_exact`
  (stub), `DatasetTaskMetricsKind.NUMERIC_EXACT`, `DatasetTask.dataset_id` (renamed from
  `corpus_id`, `run_config` removed), `DatasetTask` exported, `vllm_wrapper.chat` note (ID
  stickiness, non-batched parallelism), `dataset_bm25_index.py` / `dataset_study_interface.py`
  (empty stubs), `test_programmatic_dataset_study.py` (stub).
- The user's own renames left `/source` inconsistent on the study path: `DataOrigin.SYNTHETIC` no
  longer exists but `dataset_study.py:282,309` and the study test still use it;
  `dataset_manager.label_study_examples` / `select_training_data` take `origins` but their bodies
  reference the removed `synthetic_only`; `pyproject.toml` has `peft` (the `>=0.20.0` pin was
  dropped by the user; the lock still resolves peft 0.20.0).
- Already in `/source` (no card needed): the `WindowedBytePooling` per-text window fix, the batching
  changes, `shuffle_fill_truncate`, `trust_remote_code`, `import typing as t`, report files mode 644,
  the bench `~` expansion, `runtime_config` knobs (`dataset_study_programmatic_*`, `cache_storage_dir`).
- Environment: the agent container has neither podman nor docker (`which` finds nothing). The Sky
  image (`activation/cloud/Dockerfile`) is a CUDA devel image with uv, ssh, rsync; no container
  runtime inside. vLLM 0.28.0; `vllm.v1.engine.async_llm.AsyncLLM.generate(prompt, sampling_params,
  request_id, *, lora_request, priority, data_parallel_rank, ...)` exists; `LLM.chat(tools=...,
  chat_template_kwargs=...)` exists.
- DAPO-Math-17k (`BytedTsinghua-SIA/DAPO-Math-17k`, verified via the datasets server): one `train`
  split with 1,791,700 rows (the 17k problems repeated for the DAPO run); fields `data_source`
  (`math_dapo`), `prompt` (one user message: instructions + problem + "Answer:" line request),
  `ability` (`MATH`), `reward_model` `{ground_truth: "34", style: "rule-lighteval/MATH_v2"}`,
  `extra_info.index` (uuid). Ground truths are integers as strings. Loading needs a dedupe on
  `extra_info.index` (or the prompt text) while streaming.
- `HarnessRuntime` owns `loaded_models`, `dataset_manager`, `module_manager`. `LoadedModel`
  engine path: `engine_to_device(TARGET)` frees every other model's engine (`engine_free_other_models`)
  and this model's HF copy, builds `VLLMWrapper(**engine_kwargs)`; `engine_chat_many` merges
  harness defaults (`temperature 1.0`, `max_tokens 1024`, `enable_thinking False`) → recommended →
  caller kwargs and calls `VLLMWrapper.chat` (one sync `vllm.LLM` per GPU, strided shards, threads).
  Base engine formula: `max_model_len 20_000`, fp8 KV, `gpu_memory_utilization 0.9`, `enable_lora`
  False. Raw HF models of other names are not freed by an engine load (co-habitation already holds
  when memory fits).
- `DatasetTask.score` unwraps `run_result.answer` (`scoring.unwrap_submit_answer`) and dispatches on
  `reference_metrics_kind`; `NUMERIC_EXACT` has no dispatch yet. `LoadedDataset.scorable_tasks:
  dict[str, DatasetTask]` exists.
- Tests are gated by markers (`slow`, `gpu`, `manual`, `ddp`) through `tests/conftest.py`.

## Classification of the slice-1a workspace delta

| workspace change | class | M0? |
| --- | --- | --- |
| `cloud/sky.py`: `exec --sync/--watch/--interval`, `sync` subcommand, push/pull thread, `ACTIVATION_SYNC_ROOT` export | the valuable piece | yes |
| `common/data_syncing.py`: `sync_root`, `resolve_path` | the valuable piece | yes |
| `retrieval_batching.embedded_text_length`: `min(len, None)` TypeError when `doc_embedding_input_limit_chars` is None (the GPU default) | genuine bug | yes |
| `dataset_study.generate_examples_labels`: a label pool over the engine context raised `VLLMValidationError` and killed the run; skip prompts over 18k tokens, run the rest of the batch | genuine bug | yes |
| study-path consistency after the user's renames (`SYNTHETIC_QA`, `origins` propagation, test) | repair of the user's own in-progress edits | decision 1 |
| label cache key per kind, seed-per-pass, `--study-only`, `--report-on-test`, unique report folders | fixes inside dropped features (cache, benches) | no |
| description / programmatic modes, prompts file, JSONL study cache, origins filter, `select_testing_data` | retrieval feature | no |
| P+V AC model, document instruction, `train` without validation + `eval`, baselines, reporter eval table | retrieval feature | no |
| `RedHatAI/Qwen3.5-4B-FP8-dynamic` recommended kwargs | knowledge, retrieval-side | no (agent milestones add their own model entries) |

## Design (slice 1 = full end-to-end rollouts + caching, no rejection sampling; slice 2 = harvesting + SFT)

### Components and owners

| component | file | role |
| --- | --- | --- |
| `AgentConfig`, `TrajectoryStep`, `AgentRunResult` | `agent/agent_config.py` | inputs, budgets, tools, the serialisable result |
| `AgentEnv` | `agent/agent_env.py` | sandbox: python / shell / files; podman backend, local-process backend |
| `AgentTool` + `ShellTool`, `PythonTool`, `SubmitAnswerTool`, `ParallelCallTool`, `SubagentTool` | `agent/agent_tools.py` | tool definitions (chat-template `tools`), execution, truncation, AC outputs |
| `Agent` | `agent/agent.py` | the turn loop; owns env when not given one; collects the trajectory and counters |
| `RolloutManager` | `agent/rollout_manager.py` | single / grouped rollouts, concurrency, seeds, scoring, caching |
| `RolloutCache` | `agent/rollout_caching.py` | JSONL of serialised results under the synced folder |
| `RolloutReporter` | `agent/rollout_reporter.py` | live page: progress, scores, active trajectories |
| engine serving path | `harness/vllm_wrapper.py`, `harness/loaded_model.py` | per-request submission, agent→replica pinning, batch `chat` on top |
| `DapoMathDataset`, `numeric_exact` | `dataset/loaders/dapo_math.py`, `dataset/scoring.py` | tasks and scoring |

### Data flow of one rollout

1. `RolloutManager.perform_*` expands configs × group into (config, seed) jobs, drops the ones the
   cache holds, submits the rest to a bounded thread pool (`agent_max_concurrent`).
2. Each job builds `Agent(harness, config, env=None, reporter)`; the agent creates its env
   (podman container `--rm`, or a local scratch dir), initialises tools from `config.tools` plus the
   four defaults.
3. Turn: messages (system, user, prior assistant/tool turns) → `loaded_model.engine_submit(messages,
   tools, sampling seed, agent_id)` → text → parse `<tool_call>` blocks → execute (parallel tool calls
   fan out on a small pool) → tool messages appended → reporter step. Stops on `submit_answer`,
   `max_turns`, `max_tool_errors`, `max_duration`.
4. `AgentRunResult` (answer, counters, trajectory, score) → scored by `config.dataset_task.score`
   when present → cache append (flushed) → reporter finish. The env is torn down when the agent owns it.

### Engine serving path (M1)

Rollouts cannot be batched synchronously: agents finish turns at different times. The engine must
accept requests one at a time from many threads and batch them itself (continuous batching), and a
given agent should keep hitting the same replica so its prefix cache pays off across turns.

Recommended: `VLLMWrapper` builds one `AsyncLLM` per visible GPU, each driven by its own event loop
on a background thread; `submit(token_ids, sampling_params, request_id, lora_request, replica) ->
concurrent.futures.Future[RequestOutput]`. `chat(conversations, **kwargs)` (the study path) becomes
"template every conversation with the tokenizer (`apply_chat_template(..., tools=, add_generation_prompt=True,
**chat_template_kwargs)`), submit all with strided replica assignment, wait, return in order" — same
signature and semantics, one engine kind. The agent path uses `submit` directly with
`replica = crc32(agent_id) % world_size`. `LoadedModel.engine_submit(messages, tools, sampling seed,
agent_id, lora_name) -> EngineChatOutput` wraps it with the same kwargs merge as `engine_chat_many`.
Alternative (decision 3): keep the sync `LLM` for the study path and add the async serving engine as
a second kind (`ModelConfig.engine_serving`), switching by free + reload. Rejected alternative: a
micro-batching thread over the sync `LLM` (a long generation blocks the whole batch; no continuous
batching between calls).

Recommended kwargs entry for `Qwen/Qwen3.5-9B`: `max_model_len 40_960` (compaction threshold is 32k
tokens), thinking off through `chat_template_kwargs`, `qwen_non_thinking` sampling. Co-habitation:
`engine_to_device` keeps raw HF models of other names; the rollout test passes
`engine_kwargs={"gpu_memory_utilization": 0.8}` when an embedding model must share the GPU. LoRA on
the serving engine (`lora_name` on `AgentConfig`) is plumbed (`enable_lora` from `max_loras`) but
untested in slice 1 (no adapter exists yet; slice 2).

### Agent environment (M2)

`AgentEnv(dockerfile_path=None, env_args=None, default_image=harness default)`, podman only:
`podman build` the dockerfile once per content hash (tag `activation-env-<hash>`), else the default
image; `podman run -d --rm --network none <env_args> <image> sleep infinity`; `run_python_code` =
`podman exec -i <id> python3 -` with the code on stdin; `run_shell` = `podman exec -i <id> bash -s`;
`write_file` = `podman exec -i <id> sh -c 'mkdir -p "$(dirname P)" && cat > P'`; `read_file` =
`podman exec <id> cat P`. Timeouts through `Popen.communicate(timeout)`, kill on expiry, partial
combined stdout/stderr returned with `ok=False`. `shutdown` = `podman rm -f` once (idempotent);
`__del__` calls it. `env_args` is a dict of extra `podman run` flags (`{"--env": "PYTHONHASHSEED=0"}`); resource limits need cgroups, which podman runs without inside Sky nodes.

Sky side (same milestone): the Dockerfile installs `podman fuse-overlayfs` and writes
`/etc/containers/storage.conf` (`driver = "overlay"`, `mount_program = "/usr/bin/fuse-overlayfs"`);
`_task_yaml` adds `config:
  docker:
    run_options: ["--privileged"]`; `IMAGE_ID` tag becomes
`uv-<lock digest>-df<8 hex of the Dockerfile>` so the image rebuilds. Validation: `podman run
docker.io/library/python:3.12-slim python3 -c 'print(1)'` inside a fresh node before any rollout.
Nothing env-related can run in this agent container (no podman); CPU checks of the agent loop use
a fake env and a fake engine.



- `AgentConfig` becomes a dataclass with the user's fields (`default_factory` for the dicts and the
  tools map); `tools` maps name → tool class, instantiated per agent with `(harness, agent)`.
- Tool definitions are OpenAI-style function schemas handed to the chat template's `tools`;
  Qwen3.5 answers with `<tool_call>{"name": ..., "arguments": {...}}</tool_call>` blocks (Hermes
  style), parsed with a tolerant regex + `json.loads`; a malformed call counts as a tool error and
  the model gets the parser's message back.
- Default tools: `shell(script, timeout)`, `python(code, timeout)`, `submit_answer(answer)`,
  `parallel_tool_call(calls: [{name, arguments}])` (runs the nested calls on a small pool, results
  concatenated in order). `SubagentTool(base_config, name, extra_description)`: runs a child `Agent`
  in the caller's env with `user_prompt = task`, appends its result to the parent's
  `subagent_results` under the parent's lock, returns the child's answer as the tool output and the
  child's trajectory as `ac_outputs["trajectory"]` (and receives the caller's trajectory as
  `ac_inputs`) unless `enable_ac_communication` is False. `SemanticSearchTool(dataset_id=None, top_k=5, max_chars=2000)` is BM25 for now over the task's
  dataset (or the configured one) through `DatasetIndex.bm25_query_many_frozen`, returning
  `[chunk_id]` + snippet per hit; the dense path returns with the retrieval rethink.
- `Agent.run()` produces `AgentRunResult`: `answer` (from `submit_answer`, else None), `num_turns`,
  token counters from `EngineChatOutput` (cached-input tokens from `RequestOutput.num_cached_tokens`
  when vLLM reports it), `num_ac_input_bytes/tokens` (sizes of the AC inputs carried), `duration`,
  `trajectory` (list of `TrajectoryStep` dicts), `score`, `score_feedback`, `subagent_results`.
  `serialize()`/`deserialize()` handle the nested config (tool classes by name, `dataset_task` by
  `dataset_id` + `task_id` + gold fields).
- Compaction is out of scope (slice 2 or later); `compaction_threshold_tokens` is carried.

### Model-specific details (M3, `agent/agent_utils.py`, added 2026-09-07 after the first node run)

The first node run failed both primes rollouts with `max_tool_errors`: Qwen3.5-9B's chat template
teaches the XML shape (`<tool_call><function=python><parameter=code>…</parameter></function></tool_call>`),
not Hermes JSON, and the parser only knew JSON. The user asked for the parsing and the rendering to be
adaptive and for the model-specific parts to live in one module. `ModelDialect.from_chat_template`
reads the model's habits off its chat template (which call shape it teaches; whether it renders
`tools` / `tool_calls` at all), `parse_tool_calls` tries every known `ToolCallFormat` per block (XML
function blocks, Hermes JSON; unterminated blocks are parsed too), text parameter values are coerced by
the tool schema, and `MessageRendering` / `InlineRendering` decide how the assistant turn, the tool
results and the tool list go back to the model. The agent loop holds no format knowledge.

### Rollout manager, cache, reporter (M4)

- `HarnessRuntime.rollout_manager = RolloutManager(self)` next to the dataset manager.
- `perform_single_rollouts(configs, seed, caching_id, perform_scoring)` and
  `perform_grouped_rollouts(configs, group_count, base_seed, perform_scoring, caching_id)`; member i
  of a group runs with seed `base_seed + i`; results in input order, groups as lists.
- Concurrency: `ThreadPoolExecutor(max_workers=harness_config.agent_max_concurrent)` (default 64);
  every agent thread submits its own engine requests; the engine batches.
- Cache: `RolloutCache(caching_id)` → one JSONL under `resolve_path(f"ROLLOUTS/{caching_id}")` (the
  synced folder, so results written on a node land locally and a later node starts with them). Row
  key = (`config_key`, `seed`) where `config_key` is `dataset_id/task_id` for dataset tasks, else a
  sha256 of the prompts + model. Rows are serialised results; a killed run keeps its finished agents;
  a group of 6 after a cached group of 4 runs only members 4 and 5.
- Reporter (`RolloutReporter(HtmlReporter)`): `report_agent_start(agent)`, `report_agent_step(agent)`,
  `report_agent_finish(agent)`; widgets: status (running / finished / cached, mean score, output
  tokens/s), a line plot (finished count and running mean score over time), a table of finished
  rollouts (task, seed, turns, tool calls, tokens, duration, score), and an "active trajectories"
  text widget per running agent (last 3 steps, capped at 4k chars). Rendering change in
  `common/reporting.py` (shared with training): the page is written once as a static shell with the
  first snapshot embedded, `report_data.json` is rewritten atomically on every `render`, and the
  page polls it every 2 s and morphs (fetch failure, e.g. `file://`, falls back to the embedded
  snapshot, so the page still works as a self-contained file at the end). Validate in the tressoir
  html viewer over `sky exec --watch`.

### DAPO-Math tasks and the two key tests (M5)

- `DapoMathDataset.load(harness, max_examples, seed=0)`: stream the train split, dedupe on
  `extra_info.index`, take the first `max_examples` unique problems; `DatasetTask(task_id=index,
  dataset_id="dapo_math", task_datum=row, reference_metrics_kind=NUMERIC_EXACT,
  gold_answer=ground_truth)` into `LoadedDataset.scorable_tasks`; no documents. The agent's user
  prompt is the problem text (the dataset's "Answer:" instruction stripped) with a line telling it to
  finish with `submit_answer`.
- `scoring.numeric_exact(pred, golds)`: last number in the prediction (commas, `$`, spaces removed;
  `\boxed{}` unwrapped), compared as an exact `Fraction` to each gold; 1.0 / 0.0. Dispatch for
  `NUMERIC_EXACT` in `DatasetTask.score`.
- `tests/test_basic_agent.py` in full (both `gpu` + `slow`): `BASE_AGENT_CONFIG` on `Qwen/Qwen3.5-9B`,
  default tools, `max_turns 10`, thinking off. `test_basic_agent_primes`: the 140th + 141st + 142nd
  primes (809 + 811 + 821 = 2441), group of 2, reporter under `IB/TMP/AGENT_TEST/primes/`, asserts
  both rollouts finish with a numeric answer, at least one python/shell call happened, report files
  exist; prints trajectories and whether the answer is 2441. `test_basic_agent_dapo`: load, shuffle
  with a seed, 2 tasks, grouped rollouts of 2 with the reporter, print scores and trajectories;
  problems need not be solved. Private CPU checks (fake engine returning scripted tool calls) live
  under `IB/TMP/AGENT_ROLLOUTS/` and are not part of the handoff.

## Validation plan

- M0: `python -m compileall`, imports of every touched module, `uv run sky --help` / `exec --help`,
  a `resolve_path` check on the host; the label guard and the batching fix are re-read against
  slice-1a evidence (both ran on nodes). The study test needs an engine (GPU) — not rerun for M0.
- M1: on a Sky node (RTX PRO 6000): `test_basic_engine` (unchanged behaviour of `chat`), a private
  concurrency probe (32 threads submitting, in-order results, distinct request ids, prefix-cache hit
  on the second turn of a pinned agent), then the study test once as the regression gate for the
  unified engine (decision 3).
- M2: podman on the user's host and inside a fresh Sky node (privileged container). M3: CPU checks
  here with a fake env and a fake engine (private, `IB/TMP/AGENT_ROLLOUTS/`).
- M4–M5: the two tests on a Sky node with `exec --sync --watch`; the live page morphing locally.

## Sandbox memory (added during the DAPO probe)

Ray's OOM killer ended the Qwen3.5-4B probe job when one agent's `python3 -` reached 47 GB RSS. No cgroups
under podman inside the node, so `AgentEnv._exec` wraps every command in `sh -c 'ulimit -v KB && exec "$@"'`;
`HarnessRuntimeConfig.agent_env_memory_limit_mb = 8192` (None disables). Applied to all runs after 18:35.

## Milestone status

Round 3 (2026-09-07): implemented, run on a node, handed off. M0 Review · M1 Review · M2 Review · M3 Review
· M4 Review · M5 Review · Slice 2 TBD. Handoff: `SLICE1_ROUND.tressoir.md` (revert block, exact cards vs
the reverted tree, staged `activation/` + `README.md`). Node runs: `ac-agent` four runs, the fourth
`2 passed in 255.82s` (torn down); `ac-dapo-4b` / `ac-dapo-9b` running the 100 × 2 DAPO probe. Mid-run
user notes applied: trajectories rendered on the report (running + last N, uncut copies as files),
`agent_utils.py` for model-specific parsing/rendering, the nudge sentence. Remaining gates: the study
test on the unified engine; rootless podman on the user's host.
