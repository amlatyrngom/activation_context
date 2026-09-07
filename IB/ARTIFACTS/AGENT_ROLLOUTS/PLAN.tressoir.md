# Agent rollouts, slice 1: end-to-end rollouts with caching

Version 3 (2026-09-07): implemented and run. M0 reverts your uncommitted retrieval stubs to the committed tree and brings over the Sky sync, the synced folder, the two genuine bug fixes and the rootless-podman README section. M1–M5 are the agent stack: one async engine kind with per-agent replica pinning, a podman-only sandbox (the Sky image and setup made podman-capable), the agent loop with the four default tools plus subagent and BM25 search, the rollout manager with a cache under the synced folder and a polling live page, and the DAPO-Math tasks with both key tests. Every milestone card now leads with its completion report; the exact per-file diffs and the staged tree are in `SLICE1_ROUND.tressoir.md` next to this plan. Your round-2 adjustments (tools as `(class, kwargs)` tuples; DAPO prompts requiring `submit_answer`) are in.

## Executive Summary

### What changed since version 1

| your answer | effect on the plan |
| --- | --- |
| M0 repair is destructive: revert to how it was | M0 checks out the committed versions of the retrieval files you annotated and deletes the untracked retrieval stubs; the slice-1a staged tree stays version-controlled in `IB/ARTIFACTS/BRIGHT_ECON_ABLATION/` for later. Your agent stubs, the bench move, `numeric_exact`, the `DatasetTask` changes, `pyproject.toml` and `uv.lock` are untouched. |
| podman only | No local-process backend. The Sky image gains `podman` + `fuse-overlayfs` and the task YAML a `--privileged` docker run option (SkyPilot 0.13 allows it per task), because rollouts run where the engine runs. Your README gets a rootless section (your mid-review note) so nothing on your host needs `sudo` after install. |
| unified async engine | One `AsyncLLM` per GPU; `submit` for agents, `chat` rebuilt on top with the same signature; the study test is the regression gate. |
| subagent tool, and search on BM25 for now | `SemanticSearchTool` queries the task's dataset (or a configured one) through the existing BM25 index and returns chunk ids with snippets; the dense path returns with the retrieval rethink. |
| show the full tests | `## Interfaces` below is the complete public surface as Python; the M5 card carries the exact diff of `test_basic_agent.py` against your stub. |
| the 1.79M rows | Verified by streaming: the first 17,917 rows are unique problems and row 17,918 is the first repeat; 1,791,700 = 100 × 17,917. The loader dedupes on `extra_info.index` and stops at `max_examples`. |

### The agent stack (slice 1 = full end-to-end rollouts + caching; slice 2 = harvesting + SFT)

| component | file | role |
| --- | --- | --- |
| `AgentConfig`, `TrajectoryStep`, `AgentRunResult` | `agent/agent_config.py` | inputs, budgets, tools, the serialisable result |
| `AgentEnv` | `agent/agent_env.py` | podman sandbox: python, shell, files; timeouts with partial output; idempotent shutdown |
| `AgentTool`, `ToolCallResult`, the default tools, `SubagentTool`, `SemanticSearchTool` | `agent/agent_tools.py` | function schemas for the chat template, execution, 20k-char truncation, activation-context outputs |
| `ModelDialect`, `ToolCallFormat` (`XmlFunctionFormat`, `HermesJsonFormat`), `MessageRendering` / `InlineRendering` | `agent/agent_utils.py` | everything model-specific: how calls are written (parsed adaptively per block), how turns are rendered back, the nudge; read off the chat template |
| `Agent` | `agent/agent.py` | the turn loop; owns its env unless given one; trajectory, counters, scoring |
| `RolloutManager`, `RolloutCache`, `RolloutReporter` | `agent/rollout_manager.py`, `rollout_caching.py`, `rollout_reporter.py` | single and grouped rollouts on a bounded pool, seeds, scoring, JSONL cache under the synced folder, live page |
| engine serving path | `harness/vllm_wrapper.py`, `harness/loaded_model.py` | `AsyncLLM` per GPU, `submit`, agent to replica pinning, `chat` on top |
| `DapoMathDataset`, `numeric_exact`, `DatasetTask.agent_prompt` | `dataset/loaders/dapo_math.py`, `dataset/scoring.py`, `dataset/dataset.py` | tasks and scoring |
| Sky image and task | `cloud/Dockerfile`, `cloud/sky.py` | podman inside the node's container; image tag follows the Dockerfile |

One rollout, end to end:

1. The rollout manager expands configs × group into (config, seed) jobs, drops the ones the cache already holds, and submits the rest to a bounded thread pool (`agent_max_concurrent`).
2. Each agent creates its env (`podman run -d --rm --network none`) and its tools: `shell`, `python`, `submit_answer`, `parallel_tool_call`, plus the ones in its config (`tools`: name to class and constructor kwargs).
3. Each turn: the messages and tool schemas go to `engine_submit` (one request; the engine batches across agents; the agent is pinned to one replica so its prefix cache serves the next turn), the reply's `<tool_call>` blocks are parsed and executed (parallel calls fan out), tool messages are appended, the reporter sees the step. The loop stops on `submit_answer`, `max_turns`, `max_tool_errors`, `max_duration`, or three replies in a row without a tool call (each nudged).
4. The result (answer, token counters, trajectory, `finish_reason`, score from `DatasetTask.score`) is appended to the cache and flushed; the env is torn down.

Where co-design shows in slice 1: `ac_inputs` on the config, `activations` on every trajectory step, `ac_outputs` on every tool result (the full tool output beyond the truncation, a subagent's whole trajectory) and the `num_ac_input_*` counters are carried as data. No model consumes them yet.

### Boundaries

- Engine and rollouts run on Sky nodes; podman validation runs on your host and inside a fresh node; this agent container has no podman and no GPU, so its checks use a fake env and a fake engine.
- No rejection sampling, harvesting, SFT or context compaction in slice 1 (`compaction_threshold_tokens` is carried, unused). LoRA on the serving engine is plumbed, untested until an adapter exists.
- Tests are the two you asked for; other validation is private under `IB/TMP/`.

## Accepted decisions

- **M0 repair (destructive).** Reverted to the committed tree: `dataset_manager.py`, `dataset_study.py`, `retrieval_ac.py`, `retrieval_model.py`, `retrieval_trainer.py`, `runtime_config.py`, `vllm_wrapper.py`, `tests/test_basic_dataset_study.py`; `dataset.py` partially (the `DataOrigin` split and `programmatic_retrieval_examples` go; `NUMERIC_EXACT` and the `DatasetTask` changes stay). Deleted: `dataset_caching.py`, `dataset_study_prompts.py`, `dataset_bm25_index.py`, `dataset_study_interface.py`, `tests/test_programmatic_dataset_study.py`, `bench/bright_econ_ablation/`. Your `cache_storage_dir` knob goes with `runtime_config.py`; `resolve_path` is its replacement.
- **Environment: podman only**, on your host (rootless) and inside Sky nodes (privileged container, podman + fuse-overlayfs in the image).
- **Engine: unified `AsyncLLM`.**
- **Tools: defaults + subagent + BM25 search.**

## Requested Decisions

No open decision this round. Round 3 (2026-09-07, on `SLICE1_ROUND.tressoir.md`) had one note, applied: the sandbox memory limit is an env start kwarg (`AgentConfig.env_memory_limit_mb`, default from the harness), a hard container limit where cgroups exist and a soft `ulimit -v` at 85% per process everywhere.

Round 2 (2026-09-07) approved the interfaces and the two tests with two adjustments, both applied:

- **`AgentConfig.tools: dict[str, tuple[type[AgentTool], dict]]`.** Class and constructor kwargs in one tuple; `tool_kwargs` is gone. `_initialize_tools` unpacks the pair and passes the kwargs to the tool's constructor.
- **DAPO prompts require `submit_answer`.** The loader pattern-replaces the dataset's head instruction ("Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer …") and tail ("Remember to put your answer on its own line after \"Answer:\".") with instructions to compute in the environment and finish with one `submit_answer` call; the problem text between them is untouched.

## Interfaces

The complete public surface of slice 1, as it will exist after M1–M5. Bodies are omitted; every signature and field is final unless you change it.

`activation/agent/agent_config.py`

```python
@dataclass
class AgentConfig:
    # Inputs
    system_prompt: str = ""
    user_prompt: str = ""
    ac_inputs: dict[str, object] = field(default_factory=dict)  # activation-context inputs, carried as data in slice 1
    dataset_task: "DatasetTask | None" = None                    # set when the run corresponds to a dataset task

    # Model calls
    model_name: str | None = None          # a harness model name; its engine serves the rollout
    lora_name: str | None = None           # adapter on the serving engine (plumbed, untested in slice 1)
    call_kwargs: dict | None = None        # engine chat kwargs, merged like engine_chat_many (sampling_params inside)

    # Environment. Every agent gets shell / python / parallel_tool_call / submit_answer.
    env_dockerfile_path: str | None = None # None: the harness default image
    env_args: dict[str, str] | None = None # extra `podman run` flags, e.g. {"--env": "PYTHONHASHSEED=0"}
    env_memory_limit_mb: int | None = None # round 3: sandbox memory; None = harness agent_env_memory_limit_mb (8192). --memory where cgroups exist, ulimit -v at 85% per process always
    tools: dict[str, tuple[type["AgentTool"], dict]] = field(default_factory=dict)  # name -> (class, constructor kwargs), added to the defaults
    enable_ac_communication: bool = False  # subagents exchange trajectories as activation context

    # Budgets
    max_turns: int = 20
    max_tool_errors: int = 5
    max_duration: float = 600              # seconds
    compaction_threshold_tokens: int = 32768  # carried; compaction is slice 2

    def serialize(self) -> dict: ...                       # tool classes by "module:Class", dataset_task by id + gold fields
    @staticmethod
    def deserialize(d: dict, harness: "HarnessRuntime") -> "AgentConfig": ...


@dataclass
class TrajectoryStep:
    """Simple formats the model expects (raw dicts, not our objects)."""
    role: str                                   # "assistant" or "tool"
    content: str                                # assistant text, or the concatenated tool outputs
    tool_calls: list[dict] = field(default_factory=list)        # [{"id", "name", "arguments"}]
    tool_call_results: list[str] = field(default_factory=list)  # truncated outputs, same order
    activations: dict[str, object] = field(default_factory=dict) # ac_outputs of this step's tools, keyed by call id


@dataclass
class AgentRunResult:
    agent_config: AgentConfig
    answer: t.Any = None                        # the submit_answer argument, None when never submitted
    num_turns: int = 0
    num_input_tokens: int = 0
    num_cached_input_tokens: int = 0            # prefix-cache hits reported by the engine
    num_output_tokens: int = 0
    num_ac_input_bytes: int = 0
    num_ac_input_tokens: int = 0
    duration: float = 0.0
    trajectory: list[dict] = field(default_factory=list)   # TrajectoryStep dicts, in order
    score: float = 0.0
    score_feedback: str | None = None
    subagent_results: list["AgentRunResult"] = field(default_factory=list)
    finish_reason: str = ""                     # added: submitted | max_turns | max_tool_errors | max_duration | no_tool_call | error
    seed: int = 0                               # added: the sampling seed of this rollout

    def serialize(self) -> dict: ...            # nested config, dataset task and subagent results included
    @staticmethod
    def deserialize(d: dict, harness: "HarnessRuntime") -> "AgentRunResult": ...
```

`activation/agent/agent_tools.py`

```python
@dataclass
class ToolCallResult:
    output: str                                  # what the model sees (already truncated)
    is_error: bool = False
    ac_outputs: dict[str, t.Any] = field(default_factory=dict)  # ac name -> ac input, e.g. "full_output"
    is_final: bool = False                       # added: submit_answer sets it


class AgentTool:
    name: str = ""
    description: str = ""
    parameters: dict = {}                        # JSON schema of the arguments
    def __init__(self, harness: "HarnessRuntime", agent: "Agent", **kwargs): ...
    def tool_definition(self) -> dict: ...       # {"type": "function", "function": {name, description, parameters}}
    def execute(self, **arguments) -> ToolCallResult: ...


def truncate_output(env: "AgentEnv", text: str, call_id: str, limit: int = 20_000) -> tuple[str, dict]:
    """head[:10000] ... [truncated; full output at /tmp/agent_outputs/<call_id>.txt] ... tail[-10000:];
    the full text is written into the env and returned as {"full_output": text}."""


class ShellTool(AgentTool):        # shell(script: str, timeout: int = 120)
class PythonTool(AgentTool):       # python(code: str, timeout: int = 120)
class SubmitAnswerTool(AgentTool): # submit_answer(answer: str) -> is_final
class ParallelCallTool(AgentTool): # parallel_tool_call(calls: list[{"name": str, "arguments": dict}]) -> outputs in order

class SemanticSearchTool(AgentTool):
    """semantic_search(query: str, top_k: int | None = None). BM25 for now over dataset_id (default:
    the config's dataset task's dataset); returns "[chunk_id]\n<snippet>" blocks, max_chars per hit."""
    def __init__(self, harness, agent, dataset_id: str | None = None, top_k: int = 5, max_chars: int = 2000): ...

class SubagentTool(AgentTool):
    """<name>(task: str). Runs a child Agent in the caller's env with user_prompt = task; the child's
    answer is the tool output, its trajectory goes to ac_outputs["trajectory"] and the caller's
    trajectory into the child's ac_inputs (both skipped when enable_ac_communication is False).
    The child's result is appended to the parent's subagent_results under the parent's lock."""
    def __init__(self, harness, agent, base_config: AgentConfig, extra_description: str = ""): ...
```

`activation/agent/agent_env.py`

```python
class AgentEnv:
    """One podman container per env: `podman run -d --rm --network none <env_args> <image> sleep infinity`.
    The image is the dockerfile built once per content hash, else the harness default image."""
    def __init__(self, dockerfile_path: str | None = None, env_args: dict[str, str] | None = None,
                 default_image: str = "docker.io/library/python:3.12-slim"): ...
    def run_python_code(self, code: str, timeout: int | None = None) -> tuple[str, bool]: ...  # combined output, ok
    def run_shell(self, script: str, timeout: int | None = None) -> tuple[str, bool]: ...
    def write_file(self, path: str, content: str) -> bool: ...
    def read_file(self, path: str) -> tuple[str, bool]: ...
    def shutdown(self) -> None: ...   # podman rm -f, idempotent; __del__ calls it
```

`activation/agent/agent_utils.py` (model-specific details, out of the loop)

```python
class ToolCallFormat:                 # one way of writing a call inside <tool_call> … </tool_call>
    name: str; reminder: str          # reminder = the shape, quoted back to the model on a parse error
    def matches(self, block: str) -> bool
    def parse(self, block: str, parameter_types) -> tuple[str, dict]   # (tool name, arguments) or ValueError
    def render(self, name: str, arguments: dict) -> str                # the block as the model writes it

class XmlFunctionFormat(ToolCallFormat)   # Qwen3.5 / Qwen3-Coder: <function=name><parameter=key>value</parameter></function>; text values coerced by the schema (integer, number, boolean, object, array)
class HermesJsonFormat(ToolCallFormat)    # Qwen3 / Qwen2.5 / Hermes: {"name": ..., "arguments": {...}}
FORMATS = [XmlFunctionFormat(), HermesJsonFormat()]

def parse_tool_calls(text, parameter_types=None, formats=None, preferred=None) -> tuple[str, list[dict]]
    # every block is parsed by the first format that recognises it (adaptive per block); an unterminated block
    # (max_tokens) is still parsed; a block no format accepts -> {"name": "parse_error", "arguments": {"raw", "error", "expected"}}
def parameter_types_of(tool_definitions) -> dict[tool, dict[param, type]]

class MessageRendering:               # default: structured messages, the chat template renders tools / tool_calls / tool results
    def system_prompt(self, system_prompt, tool_definitions) -> str
    def template_tools(self, tool_definitions) -> list[dict] | None    # what goes to apply_chat_template(tools=...)
    def assistant_message(self, content, calls) -> dict                # {"role": "assistant", "content", "tool_calls": [...]}
    def tool_messages(self, calls, results) -> list[dict]              # [{"role": "tool", "content": output}, ...]
class InlineRendering(MessageRendering)   # templates without tool support: tool list in the system prompt, calls written into the assistant text, results as one user message of <tool_response> blocks

@dataclass
class ModelDialect:
    preferred_format: ToolCallFormat; formats: list[ToolCallFormat]; rendering: MessageRendering
    nudge_message: str = "Use a tool, or call submit_answer with your final answer."
    @classmethod
    def from_chat_template(cls, chat_template: str | None) -> ModelDialect  # "<function=" -> XML; "<tool_call>" -> JSON; no tools/tool_calls in the template -> InlineRendering
    @classmethod
    def for_tokenizer(cls, tokenizer) -> ModelDialect
    def parse(self, text, parameter_types=None) -> tuple[str, list[dict]]
    def parse_error_message(self, arguments) -> str
# Agent.run(): self.dialect = ModelDialect.for_tokenizer(loaded_model.tokenizer); the loop only ever calls
# dialect.parse / rendering.assistant_message / rendering.tool_messages / rendering.system_prompt / rendering.template_tools.
```

`activation/agent/agent.py`

```python
class Agent:
    def __init__(self, harness: "HarnessRuntime", agent_config: AgentConfig, agent_env: AgentEnv | None = None,
                 parent_agent: "Agent | None" = None, reporter: "RolloutReporter | None" = None, seed: int = 0): ...
    agent_id: str            # uuid4 hex; pins the agent to one engine replica across turns
    lock: threading.Lock     # guards subagent_results appends
    tools: dict[str, AgentTool]
    run_results: AgentRunResult
    def run(self) -> AgentRunResult: ...   # the turn loop; fills run_results; never raises for a model or tool failure (finish_reason "error")
    def score(self) -> float: ...          # dataset_task.score(run_results) when a task is attached, else 0.0
    def shutdown(self) -> None: ...        # tears the env down only when this agent created it
```

`activation/agent/rollout_manager.py`, `rollout_caching.py`, `rollout_reporter.py`

```python
class RolloutManager:                      # harness.rollout_manager, next to dataset_manager
    def perform_single_rollouts(self, agent_configs: list[AgentConfig], seed: int = 0, caching_id: str | None = None,
                                perform_scoring: bool = True, reporter: "RolloutReporter | None" = None) -> list[AgentRunResult]: ...
    def perform_grouped_rollouts(self, agent_configs: list[AgentConfig], group_count: int, base_seed: int = 0,
                                 perform_scoring: bool = True, caching_id: str | None = None,
                                 reporter: "RolloutReporter | None" = None) -> list[list[AgentRunResult]]: ...
    # member i of a group runs with seed base_seed + i; results keep the input order


class RolloutCache:
    """rollouts.jsonl under resolve_path(f"ROLLOUTS/{caching_id}"): one serialized result per line, flushed per
    append, keyed by (config_key, seed). config_key is "<dataset_id>/<task_id>" for dataset tasks, else a sha256
    of the prompts and model. A group of 6 after a cached group of 4 runs members 4 and 5 only."""
    def __init__(self, caching_id: str | None): ...
    def get(self, config_key: str, seed: int) -> dict | None: ...
    def append(self, result: AgentRunResult) -> None: ...


class RolloutReporter(HtmlReporter):
    def __init__(self, report_folder: str, title: str, description: str = ""): ...
    def report_agent_start(self, agent: Agent) -> None: ...
    def report_agent_step(self, agent: Agent) -> None: ...    # statistics and the live trajectory come from agent.run_results
    def report_agent_finish(self, agent: Agent) -> None: ...
    # page: status (running / finished / cached, mean score, output tokens/s), a line plot over time,
    # a table of finished rollouts, and an "active: <agent>" text per running agent (last 3 steps, 4k chars)
```

`activation/harness/loaded_model.py`, `vllm_wrapper.py`, `runtime_config.py`

```python
@dataclass
class EngineChatOutput:
    text: str
    prompt_token_count: int
    output_token_count: int
    cached_prompt_token_count: int = 0        # added
    finish_reason: str = ""                   # added: "stop" | "length"

class LoadedModel:
    def engine_submit(self, messages: list[dict], tools: list[dict] | None = None, seed: int | None = None,
                      agent_id: str = "", lora_name: str | None = None, chat_kwargs: dict | None = None) -> EngineChatOutput: ...
    # one conversation for one agent; same kwargs merge as engine_chat_many; blocks the calling thread while the engine batches
    def engine_chat_many(self, conversations, lora_name=None, chat_kwargs=None) -> list[EngineChatOutput]: ...  # unchanged

class VLLMWrapper:
    def __init__(self, **engine_kwargs): ...   # one AsyncLLM per visible GPU, each on its own event-loop thread
    world_size: int
    def submit(self, prompt_token_ids: list[int], sampling_params, request_id: str, replica: int,
               lora_request=None) -> "concurrent.futures.Future[vllm.RequestOutput]": ...
    def chat(self, conversations: list, **chat_kwargs) -> list["vllm.RequestOutput"]: ...   # same signature: template, submit strided, wait, in order
    def shutdown(self) -> None: ...

@dataclass
class HarnessRuntimeConfig:
    ...
    # Agents
    agent_max_concurrent: int = 64                                      # added: thread pool bound for rollouts
    agent_env_default_image: str = "docker.io/library/python:3.12-slim" # added
```

`activation/dataset/dataset.py`, `scoring.py`, `loaders/dapo_math.py`

```python
@dataclass
class DatasetTask:
    task_id: str
    dataset_id: str
    task_datum: dict
    reference_metrics_kind: DatasetTaskMetricsKind
    gold_answer: str = ""
    gold_answer_aliases: list[str] = field(default_factory=list)
    agent_prompt: str = ""                    # added: what an agent is asked; the loader fills it
    def score(self, run_result, force_metric=None) -> float: ...   # NUMERIC_EXACT dispatches to scoring.numeric_exact

def numeric_exact(pred, golds) -> float: ...  # last number of pred (\boxed{}, commas, $ stripped) as an exact Fraction vs each gold

class DapoMathDataset:
    @classmethod
    def load(cls, harness: "HarnessRuntime", max_examples: int | None, seed: int = 0) -> LoadedDataset: ...
    # streams the train split, dedupes on extra_info.index, stops at max_examples unique problems;
    # scorable_tasks[index] = DatasetTask(..., NUMERIC_EXACT, gold_answer=reward_model.ground_truth,
    #     agent_prompt=<problem text without the dataset's "Answer:" instruction>); no documents
```

## Milestones

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">M0 — Revert the retrieval stubs; bring over the Sky sync, the synced folder, the two bug fixes and the podman README</span>
    <span class="card-oneliner">A shell block for the revert, then exact diff cards against the reverted tree.</span>
    <span class="card-badge">Review</span>
  </summary>

#### Completion report

**What landed.** The round document `SLICE1_ROUND.tressoir.md` (next to this plan) opens with the revert block (eight `git checkout`s, five file deletions, one folder) and then carries exact cards against the reverted tree: `cloud/sky.py` (`exec --sync/--watch/--interval`, the `sync` command, the pull thread; image tag `uv-<lock>-df<dockerfile>`; `--privileged` run option and a podman setup block), `cloud/Dockerfile` (podman, fuse-overlayfs, storage and engine configs), `common/data_syncing.py` (`sync_root`, `resolve_path`), `retrieval/retrieval_batching.py` (the `None` embedding limit), `dataset/dataset_study.py` (label pools over the engine context are skipped, on the committed file), `README.md` (rootless podman section). `pyproject.toml`, `uv.lock` and your agent stubs are untouched.

**Drifts.** The podman-in-node setup needed three fixes found on the node (non-interactive apt keeping `/etc/fuse.conf`; `runroot`/`graphroot` in `storage.conf`; `[containers] cgroups = "disabled"` because docker delegates no controllers). The image was not rebuilt from this container (no docker here); the setup block carried the runs on the fallback image, and your next `sky setup` without `--no-image-rebuild` builds the new tag.

**Validation.** `compileall` and imports here; the sync carried every node run of the day (report pages, trajectories and the rollout cache arrived under `IB/TMP/SYNC/` while the runs went); the setup's `podman run … python:3.12-slim` check printed `podman ok` on three fresh nodes.

#### Planning Overview (as planned)

The handoff is a round document with the revert block below first, then exact per-file cards against the tree as it stands after the revert, plus a staged tree next to it. Everything in the cards already ran (the sync carried three parallel probe nodes and the overnight caching runs; the two fixes were found by real failures on nodes). Nothing here touches `pyproject.toml`, `uv.lock` or your agent files.

#### Planned Changes (as planned)

`the revert (run in /source)`

```bash
cd /source
git checkout -- activation/dataset/dataset_manager.py activation/dataset/dataset_study.py \
  activation/retrieval/retrieval_ac.py activation/retrieval/retrieval_model.py activation/retrieval/retrieval_trainer.py \
  activation/harness/runtime_config.py activation/harness/vllm_wrapper.py activation/tests/test_basic_dataset_study.py
rm activation/dataset/dataset_caching.py activation/dataset/dataset_study_prompts.py \
   activation/dataset/dataset_bm25_index.py activation/dataset/dataset_study_interface.py \
   activation/tests/test_programmatic_dataset_study.py
rm -r activation/bench/bright_econ_ablation
# dataset.py is a card, not a checkout: it keeps NUMERIC_EXACT and your DatasetTask changes.
```

`activation/dataset/dataset.py · DataOrigin, LoadedDataset`

```diff
 class DataOrigin(StrEnum):
     NATIVE = auto()
     EXTERNAL = auto()
-    SYNTHETIC_QA = auto()
-    SYNTHETIC_DESCRIPTION = auto()
-    PROGRAMMATIC = auto()
+    SYNTHETIC = auto()
@@ LoadedDataset @@
-    programmatic_retrieval_examples: dict[str, LabeledRetrievalQAExample] = field(default_factory=dict)
-    """List of programmatic retrieval examples. Separated due to potentially large volume."""
 (NUMERIC_EXACT, DatasetTask.dataset_id and the removed run_config stay as you have them)
```

`activation/cloud/sky.py · exec_cmd(), sync(), _push_sync(), _pull_once(), _pull_loop()`

```diff
@@ constants @@
 LOCAL_DOWNLOAD_ROOT = PROJECT_ROOT / "IB" / "TMP"
+LOCAL_SYNC_ROOT = LOCAL_DOWNLOAD_ROOT / "SYNC"
+REMOTE_SYNC_ROOT = f"{REMOTE_ARTIFACTS}/SYNC"
+SYNC_ROOT_ENV = "ACTIVATION_SYNC_ROOT"
@@ sync helpers @@
+def _push_sync(record):   # local IB/TMP/SYNC -> node SYNC folder, update-only (-azu), so newer node files survive
+def _pull_once(record, pairs, quiet=False):   # each (remote folder, local folder) pulled once; nothing deleted locally
+def _pull_loop(record, pairs, interval_seconds, stop):   # background thread while the command runs
+def sync(name):   # one push and one pull, e.g. after a watcher died with its session
@@ exec_cmd @@
-# @AI: make this more practical by allow --watch  etc.
-def exec_cmd(name: str, cmd: str | list[str]) -> int:
+def exec_cmd(name, cmd, *, sync=False, watch_paths=None, interval_seconds=15.0) -> int:
+    """With sync: push IB/TMP/SYNC first, pull it back every interval while the command runs and once
+    more when it ends; the command sees ACTIVATION_SYNC_ROOT. watch_paths adds one more pulled pair."""
     ...
     remote_command = shlex.join(["env", "VLLM_WORKER_MULTIPROC_METHOD=spawn",
+            f"{SYNC_ROOT_ENV}={REMOTE_SYNC_ROOT}",
             "UV_NO_SYNC=0", ...])
+    puller thread started when pairs exist; try: sky exec ... finally: stop, join, final _pull_once
@@ argparse @@
+    exec_parser.add_argument("--sync", action="store_true", help="mirror IB/TMP/SYNC before, during and after (flags go before the name)")
+    exec_parser.add_argument("--watch", nargs=2, metavar=("REMOTE_FOLDER", "LOCAL_FOLDER"))
+    exec_parser.add_argument("--interval", type=float, default=15.0)
+    sync_parser = commands.add_parser("sync", help="push and pull IB/TMP/SYNC once")
```

`activation/common/data_syncing.py · sync_root(), resolve_path()`

```diff
-"""Helper class for coarse-grained but effective data syncing. ... (your spec)"""
+"""One folder, IB/TMP/SYNC/, is the synced working area ... code never cares where it runs."""
+SYNC_ROOT_ENV = "ACTIVATION_SYNC_ROOT"
+LOCAL_SYNC_ROOT = PROJECT_ROOT / "IB" / "TMP" / "SYNC"
+
+def sync_root() -> Path:
+    return Path(os.environ.get(SYNC_ROOT_ENV) or LOCAL_SYNC_ROOT)
+
+def resolve_path(relative: str, create: bool = True) -> Path:
+    """A folder (or file path) under the synced area, created on request; may not escape it."""
+    ... raises ValueError on absolute paths or '..'; mkdir the folder (or the file's parent)
```

`activation/retrieval/retrieval_batching.py · embedded_text_length()`

```diff
 def embedded_text_length(retrieval_model, text, is_query):
     prefix = len(retrieval_model.query_instruction) if is_query else 0
-    return min(len(text), retrieval_model.input_limit_chars) + prefix
+    limit = retrieval_model.input_limit_chars
+    return (len(text) if limit is None else min(len(text), limit)) + prefix
```

`activation/dataset/dataset_study.py · generate_examples_labels()` (against the reverted, committed file)

```diff
+from vllm.exceptions import VLLMValidationError
+LABEL_PROMPT_TOKEN_LIMIT = 18_000   # under the engine's 20k context, with room for the answer
@@ the label batch loop @@
-            outputs = loaded_model.engine_chat_many(conversations, chat_kwargs=chat_kwargs)
+            try:
+                outputs = loaded_model.engine_chat_many(conversations, chat_kwargs=chat_kwargs)
+            except VLLMValidationError:
+                # One pool over the context; keep the prompts under the limit and rerun the batch.
+                lengths = [len(tokenizer.apply_chat_template(c, tokenize=True, add_generation_prompt=True)) for c in conversations]
+                kept = [i for i, n in enumerate(lengths) if n <= LABEL_PROMPT_TOKEN_LIMIT]
+                print(f"... {len(batch) - len(kept)} prompts over {LABEL_PROMPT_TOKEN_LIMIT} tokens skipped (max {max(lengths)}).")
+                batch = [batch[i] for i in kept]
+                outputs = loaded_model.engine_chat_many([conversations[i] for i in kept], chat_kwargs=chat_kwargs) if kept else []
```

`README.md · podman section` (your mid-review note: easier non-privileged use; replaces your two-line block)

```diff
-Installing `podman`.
+Installing `podman` in rootless (non-privileged) mode. Agent environments are `podman run --rm --network none`
+containers created by the harness as your normal user; nothing needs `sudo` after this block.
 ```bash
-sudo apt update && sudo apt install podman -y
-podman run docker.io/hello-world
+sudo apt update && sudo apt install -y podman uidmap slirp4netns fuse-overlayfs
+# Rootless podman maps container users onto a range of sub-UIDs/GIDs owned by you.
+grep -q "^$USER:" /etc/subuid || sudo usermod --add-subuids 100000-165535 --add-subgids 100000-165535 "$USER"
+podman system migrate
+# Ubuntu 24.04 restricts unprivileged user namespaces through AppArmor. Podman ships a profile, but if
+# `podman run` fails with a user-namespace or "cannot clone" error, relax the restriction:
+#   sudo sysctl -w kernel.apparmor_restrict_unprivileged_userns=0
+#   echo 'kernel.apparmor_restrict_unprivileged_userns=0' | sudo tee /etc/sysctl.d/60-podman.conf
+podman info --format 'rootless={{.Host.Security.Rootless}} driver={{.Store.GraphDriverName}}'  # rootless=true driver=overlay
+podman run --rm docker.io/hello-world
 ```
```

</details>

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">M1 — Engine serving path: one request at a time, from many threads, pinned per agent</span>
    <span class="card-oneliner">AsyncLLM replicas behind VLLMWrapper.submit; chat rebuilt on top; LoadedModel.engine_submit; Qwen3.5-9B entries.</span>
    <span class="card-badge">Review</span>
  </summary>

#### Completion report

**What landed.** `harness/vllm_wrapper.py` rewritten around one `AsyncLLM` per visible GPU on its own event-loop thread (`_Replica`; the engine is constructed inside the loop), `submit(prompt_token_ids, sampling_params, request_id, replica, lora_request)` returning a concurrent future, `template()` for chat-template tokenisation with tools, `replica_for(agent_id, world_size)` (crc32), `chat()` rebuilt on top with the old signature; `recommended_*_kwargs` gain Qwen3.5-9B and Qwen3.5-4B (40,960 context, thinking off). `harness/loaded_model.py`: `engine_submit(messages, tools, seed, agent_id, lora_name, chat_kwargs) -> EngineChatOutput` (with `cached_prompt_token_count`, `finish_reason`), `_merged_chat_kwargs` shared by `engine_chat_many`.

**Drifts.** None in the interface. The study test was not rerun on the unified engine today (the node time went to the agent runs); it is the remaining regression gate for `chat()`.

**Validation.** Node runs: 2, 4 and 200-rollout groups submitting from up to 64 threads, results in order, `num_cached_tokens` reported per request (2,112–13,728 cached prompt tokens on the DAPO rollouts, 1,056 on a primes rollout sharing the group's prompt). Engine load 2.5 s for weights, ~70 s with compile and CUDA graphs.

#### Planning Overview (as planned)

`VLLMWrapper` builds one `AsyncLLM` per visible GPU, each driven by an event loop on a background thread; `submit` returns a future for the `RequestOutput`. `chat` keeps its signature: it templates each conversation with the tokenizer (`apply_chat_template(..., tools=..., add_generation_prompt=True, **chat_template_kwargs)`), submits with strided replicas and returns in order. `LoadedModel.engine_submit` merges kwargs exactly like `engine_chat_many`, sets the per-request seed, and pins the agent to `crc32(agent_id) % world_size` so the prefix cache of turn k serves turn k+1. Co-habitation: an engine load already keeps raw HF models of other names; a run that needs an embedding model on the same GPU lowers `gpu_memory_utilization` through `engine_kwargs`.

#### Planned Changes (as planned)

`activation/harness/vllm_wrapper.py · VLLMWrapper.__init__(), submit(), chat(), recommended_engine_kwargs(), recommended_chat_kwargs()`

```diff
-            self.replicas.append(vllm.LLM(**engine_kwargs))
+            engine = AsyncLLM.from_engine_args(AsyncEngineArgs(**engine_kwargs))
+            self.replicas.append(_Replica(engine, loop, thread))     # one event loop thread per replica
+
+    def submit(self, prompt_token_ids, sampling_params, request_id, replica, lora_request=None):
+        """One request; the replica's engine batches it with whatever else is in flight."""
+        return asyncio.run_coroutine_threadsafe(self.replicas[replica].collect(...), self.replicas[replica].loop)
+
     def chat(self, conversations, **chat_kwargs) -> list[vllm.RequestOutput]:
-        """strided shards, one thread per replica, in-order results."""
+        """Same signature as before: template, submit with strided replicas, wait, return in order."""
@@ recommended kwargs @@
+            "Qwen/Qwen3.5-9B": base | {"max_model_len": 40_960},              # 32k compaction threshold + answer room
+            "Qwen/Qwen3.5-9B": qwen_non_thinking | {"chat_template_kwargs": {"enable_thinking": False}},
```

`activation/harness/loaded_model.py · engine_submit(), EngineChatOutput`

```diff
+    cached_prompt_token_count: int = 0
+    finish_reason: str = ""
+    def engine_submit(self, messages, tools=None, seed=None, agent_id="", lora_name=None, chat_kwargs=None) -> EngineChatOutput:
+        """One conversation for one agent: same kwargs merge as engine_chat_many, per-request seed,
+        replica pinned by agent_id. Blocks the calling thread; the engine batches across threads."""
```

</details>

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">M2 — Agent environment on podman, on your host and inside Sky nodes</span>
    <span class="card-oneliner">AgentEnv over podman exec; the Sky image gains podman and the task a privileged run option; image tag follows the Dockerfile.</span>
    <span class="card-badge">Review</span>
  </summary>

#### Completion report

**What landed.** `agent/agent_env.py` as planned: podman only, `podman run -d --rm --network none <env_args> <image> sleep infinity`, `podman exec -i` with the code on stdin and an in-container `timeout -s KILL`, partial output on expiry, `write_file`/`read_file`, image build per Dockerfile hash (`localhost/activation-env-<sha12>`), idempotent `shutdown`; the harness default image is `docker.io/library/python:3.12-slim` (`agent_env_default_image`). `cloud/sky.py` and `cloud/Dockerfile` make the node's container podman-capable (M0 card).

**Drifts.** In the 100-problem probe one sandboxed Python grew to 47 GB and the node's OOM killer ended the job. Per your round-3 note the limit is an env start setting: `AgentConfig.env_memory_limit_mb` (None = harness `agent_env_memory_limit_mb`, 8192) becomes `podman run --memory` where cgroups exist (your rootless host; a hard kill) and `ulimit -v` at 85% of it on every exec (a `MemoryError` the model reads). Inside Sky nodes podman has no cgroups, so the first `--memory` refusal switches the hard limit off for the process and the soft limit carries. Rootless podman on your host is documented, not run from here.

**Validation.** Every rollout of the day ran in such a sandbox on the node (206 containers across the four runs, each torn down after its agent); the model hit `ModuleNotFoundError: sympy` and `NameError` inside them and recovered from the tracebacks, which is the tool loop working end to end. Podman-in-node smoke check in setup: `podman ok`.

#### Planning Overview (as planned)

One class, five operations, one backend. The dockerfile is built once per content hash (tag `activation-env-<hash>`), else the harness default image is used; the container runs detached with `--rm --network none` plus `env_args`; every operation is a `podman exec` with the code or script on stdin. Timeouts use `Popen.communicate(timeout)`; on expiry the process is killed and the partial combined output is returned with `ok=False`. Truncation is the tool layer's job.

Sky side: rollouts run where the engine runs, so the node's container must run podman. The Dockerfile installs `podman fuse-overlayfs` and writes a storage config that uses fuse-overlayfs (the node container's root filesystem is itself an overlay); the task YAML adds `config: docker: run_options: ["--privileged"]`, which SkyPilot 0.13 accepts per task. Today the image tag is `uv-<lock digest>`, so a Dockerfile change would not rebuild; the tag gains a short Dockerfile digest. First check on a fresh node: `podman run --rm docker.io/library/python:3.12-slim python3 -c 'print(1)'`.

#### Planned Changes (as planned)

`activation/agent/agent_env.py · AgentEnv`

```diff
 class AgentEnv:
-    def __init__(self, dockerfile_path: str|None=None):
+    def __init__(self, dockerfile_path=None, env_args=None, default_image="docker.io/library/python:3.12-slim"):
+        image = self._build_image(dockerfile_path) if dockerfile_path else default_image
+        self.container_id = podman run -d --rm --network none <env_args> image sleep infinity
-    def run_python_code(self, code, timeout): pass
+    def run_python_code(self, code, timeout=None) -> tuple[str, bool]:   # podman exec -i <id> python3 -  (code on stdin)
+    def run_shell(self, script, timeout=None) -> tuple[str, bool]:      # podman exec -i <id> bash -s
+    def write_file(self, path, content) -> bool                            # sh -c 'mkdir -p "$(dirname P)" && cat > P'
+    def read_file(self, path) -> tuple[str, bool]                          # cat P
+    def shutdown(self): podman rm -f once; __del__ calls it
+    def _exec(self, argv, stdin, timeout): Popen(..., stderr=STDOUT); communicate(timeout); on TimeoutExpired kill and return the partial output
```

`activation/cloud/Dockerfile`, `activation/cloud/sky.py · IMAGE_ID, _task_yaml()`

```diff
         openssh-server \
+        podman \
+        fuse-overlayfs \
         rsync \
+RUN printf '[storage]\ndriver = "overlay"\n[storage.options.overlay]\nmount_program = "/usr/bin/fuse-overlayfs"\n' > /etc/containers/storage.conf
@@ sky.py @@
-DEFAULT_IMAGE_ID = f"docker:{DEFAULT_IMAGE_REPOSITORY}:uv-{LOCK_DIGEST}"
+DEFAULT_IMAGE_ID = f"docker:{DEFAULT_IMAGE_REPOSITORY}:uv-{LOCK_DIGEST}-df{DOCKERFILE_DIGEST}"
@@ _task_yaml @@
+config:
+  docker:
+    run_options: ["--privileged"]   # podman inside the node's container (agent environments)
```

</details>

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">M3 — Agent loop, tools, trajectory and result</span>
    <span class="card-oneliner">The interfaces above, implemented: the turn loop with budgets, the four default tools, the subagent tool and BM25 search, serialisation.</span>
    <span class="card-badge">Review</span>
  </summary>

#### Completion report

**What landed.** `agent/agent_config.py` (`AgentConfig` with `tools: dict[str, tuple[type[AgentTool], dict]]` per your round-2 note, `TrajectoryStep`, `AgentRunResult` with `finish_reason`/`seed`, serialisation both ways), `agent/agent_tools.py` (schemas, `truncate_output`, `shell`/`python`/`submit_answer`/`parallel_tool_call`, `SemanticSearchTool` on BM25, `SubagentTool`), `agent/agent.py` (the turn loop with budgets, error capture, reporter hooks) and, new, `agent/agent_utils.py`: every model-specific detail in one place (`ToolCallFormat`s parsed adaptively per block, schema-typed coercion of text values, `MessageRendering`/`InlineRendering`, `ModelDialect.from_chat_template`).

**Drifts.** (1) The first node run failed both primes rollouts with `max_tool_errors`: Qwen3.5 writes `<function=…><parameter=…>` blocks, the parser only knew Hermes JSON. Hence `agent_utils.py` (your note: adaptive parsing and rendering, model specifics separated). (2) The nudge is your sentence, "Call one of the provided tools or submit_answer when done.", after every tool-less turn; the third in a row ends the run (was: one nudge). (3) `num_tool_calls` counted each call twice (assistant step and tool step); fixed. (4) `tool_kwargs` is gone (merged into the tuple). (5) A prompt over the model's context ends the run with `context_exceeded` (seen on the 4B probe at 42,945 tokens) instead of `error`.

**Validation.** Node: 206 rollouts, every one `submitted`, 0 parse errors after the format fix. CPU (`IB/TMP/AGENT_ROLLOUTS/cpu_check.py`, scripted engine, local fake env): both call formats, coercion, an unterminated block, dialect detection from a template, inline rendering, the parallel tool with a bogus call and a malformed block, the subagent exchanging trajectories as activation context, truncation with the full text carried as `ac_outputs`, serialisation round-trip, the three-nudge stop, the engine-error path.

#### Planning Overview (as planned)

Tool definitions are OpenAI-style function schemas passed as `tools` to the chat template; Qwen3.5 answers with `<tool_call>{"name": ..., "arguments": {...}}</tool_call>` blocks, parsed tolerantly; a malformed call is a tool error and the parser's message goes back to the model as that call's result. A reply with no tool call gets the nudge ("Call one of the provided tools or submit_answer when done."); the third such reply in a row ends the run with `no_tool_call`. Every tool result is cut to head 10k + tail 10k characters with the full text written to `/tmp/agent_outputs/<call_id>.txt` inside the env and carried as `ac_outputs["full_output"]`. `parallel_tool_call` runs its nested calls on a small pool and returns the outputs in order. The subagent tool runs a child agent in the caller's env with the task as its user prompt, exchanges trajectories as activation context unless `enable_ac_communication` is off, and registers the child's result under the parent's lock. The search tool builds the BM25 index of its dataset on first use through the dataset manager. `serialize()` / `deserialize()` round-trip the nested config (tool classes by `module:Class`) and the dataset task (dataset id, task id, gold fields, agent prompt).

#### Planned Changes (as planned)

`activation/agent/agent.py · Agent.run(), _execute_tool_calls(), score()`

```diff
     def run(self):
-        pass
+        messages = [system (config.system_prompt), user (config.user_prompt)]; start = time.time(); nudged = False
+        while True:
+            if turns >= max_turns: finish("max_turns"); break
+            if time.time() - start > max_duration: finish("max_duration"); break
+            output = loaded_model.engine_submit(messages, tools=self._tool_definitions(), seed=self.seed, agent_id=self.agent_id, lora_name=config.lora_name, chat_kwargs=config.call_kwargs)
+            calls = parse_tool_calls(output.text); counters += output tokens
+            if not calls: nudge once, else finish("no_tool_call"); continue/break
+            results = self._execute_tool_calls(calls)          # ToolCallResult per call; errors counted
+            trajectory += [assistant step, tool step with activations]; messages += assistant + tool messages
+            reporter.report_agent_step(self)
+            if any(result.is_final): finish("submitted"); break
+            if errors > max_tool_errors: finish("max_tool_errors"); break
+        run_results.duration / num_turns filled; reporter.report_agent_finish(self); return run_results
```

`activation/agent/agent_tools.py · parse_tool_calls(), truncate_output(), the tool classes`

```diff
+TOOL_CALL_PATTERN = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)
+def parse_tool_calls(text) -> list[dict]:   # [{"id": uuid8, "name", "arguments"}]; json errors become a "parse_error" entry
+def truncate_output(env, text, call_id, limit=20_000) -> tuple[str, dict]
 class ShellTool / PythonTool / SubmitAnswerTool / ParallelCallTool / SemanticSearchTool / SubagentTool: as in ## Interfaces
```

</details>

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">M4 — Rollout manager, cache under the synced folder, live reporter</span>
    <span class="card-oneliner">Single and grouped rollouts on a bounded pool; JSONL cache keyed by task and seed; a page that polls a JSON instead of being rewritten.</span>
    <span class="card-badge">Review</span>
  </summary>

#### Completion report

**What landed.** `agent/rollout_manager.py` (`perform_single_rollouts`, `perform_grouped_rollouts`, cache lookup before the pool, one engine load per call, `agent_max_concurrent` threads), `agent/rollout_caching.py` (JSONL under `resolve_path("ROLLOUTS/<caching_id>")`, keyed by `<dataset_id>/<task_id>` or a prompt hash, and the seed), `agent/rollout_reporter.py` + the `trajectories` widget in `common/reporting.py`: running rollouts first and open, then the last `finished_trajectories_shown` (≤100) complete ones, each with prompt, assistant steps, calls with arguments, tool results; page texts are cut and the uncut trajectory of every agent is written to `<report>/trajectories/<agent>.json` and linked. The page is written once and polls `report_data.json` (2 s), so it morphs without flicker.

**Drifts.** The reporter's `active-<id>` text blocks were replaced by the trajectories widget (your note). `reporter.report_folder` added because the approved test reads it. `RolloutCache` creates its own folder: `resolve_path` takes a dotted caching id for a file name and the probe's cache would have failed on the first append. Rows for cache hits do not appear on the page (the cached call in the DAPO test rendered nothing new, by design: nothing ran).

**Validation.** Node: `RolloutCache - 4 cached rollouts` on the DAPO test's second call with no engine load (asserted); the pages and trajectory files synced home every ~10 s during the runs (`IB/TMP/SYNC/AGENT_TEST/{primes,dapo}/`); the page's script passes `node --check`. CPU: 12 rollouts cached and replayed, group growth 2→3 reusing the two cached members, 14 trajectory files.

#### Planning Overview (as planned)

`harness.rollout_manager` sits next to the dataset manager. Jobs are (config, seed) pairs; member i of a group runs with `base_seed + i`; the cache row key is (`dataset_id/task_id` or a hash of the prompts and model, seed), so a group of 6 after a cached group of 4 runs members 4 and 5 only, and a killed run keeps every finished agent. The cache file lives under `resolve_path("ROLLOUTS/<caching_id>")`, which is why M0 comes first: rows written on a node land locally while the run goes, and the next node starts with them. The reporter gets your three hooks; the page shows progress, mean score, throughput, a table of finished rollouts and the last steps of every running agent. The rendering change is in the shared `HtmlReporter`, so training gets it too: the page is written once with the first snapshot embedded, `report_data.json` is rewritten atomically on each `render`, and the page polls it every 2 s (a failed fetch, e.g. over `file://`, falls back to the embedded snapshot, so the finished page still works alone). I will verify the polling in the tressoir html viewer over `sky exec --sync`.

#### Planned Changes (as planned)

`activation/agent/rollout_manager.py · RolloutManager._run_jobs()`

```diff
+    def _run_jobs(self, jobs, caching_id, perform_scoring, reporter):
+        cache = RolloutCache(caching_id); results = [cache.get(config_key(config), seed) for config, seed in jobs]
+        with ThreadPoolExecutor(max_workers=self.harness.harness_config.agent_max_concurrent) as pool:
+            futures = {pool.submit(self._run_one, config, seed, perform_scoring, reporter): i for i, (config, seed) in enumerate(jobs) if results[i] is None}
+            for future in as_completed(futures): results[futures[future]] = future.result(); cache.append(results[...])
+        return results   # input order
+    def _run_one(self, config, seed, perform_scoring, reporter):
+        agent = Agent(self.harness, config, reporter=reporter, seed=seed)
+        try: result = agent.run(); result.score = agent.score() if perform_scoring and config.dataset_task else 0.0
+        finally: agent.shutdown()
```

`activation/common/reporting.py · HtmlReporter.render()` and `activation/agent/rollout_reporter.py · RolloutReporter`

```diff
     def render(self, force=False):
-        write the whole self-contained page (data embedded) atomically
+        write report_data.json atomically on every render; write the page once (first snapshot embedded, a 2 s poll of report_data.json that morphs the widgets, fallback to the embedded data)
+class RolloutReporter(HtmlReporter): the three hooks; widgets: status, "rollouts over time" line plot, "finished rollouts" table, one "active: <agent>" text per running agent
```

`activation/harness/runtime.py, runtime_config.py`

```diff
+        self.rollout_manager = RolloutManager(self)
+    # Agents
+    agent_max_concurrent: int = 64
+    agent_env_default_image: str = "docker.io/library/python:3.12-slim"
```

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">M5 — DAPO-Math tasks, exact numeric scoring, the two key tests in full</span>
    <span class="card-oneliner">The loader that fills scorable_tasks, numeric_exact with its dispatch, and test_basic_agent.py exactly as it will be handed off.</span>
    <span class="card-badge">Review</span>
  </summary>

#### Completion report

**What landed.** `dataset/loaders/dapo_math.py` (`DapoMathDataset.load`: streaming dedupe on `extra_info.index`, `dataset_id = dapo_math_<n>`, scorable tasks with `NUMERIC_EXACT`; `agent_prompt_from_dapo` pattern-replaces the dataset's head and tail instructions with ones that require `submit_answer`, per your round-2 note), `dataset/scoring.py` (`numeric_exact`), `dataset/dataset.py` (`DatasetTask.agent_prompt`, dispatch), `tests/test_basic_agent.py` exactly as shown in the planned changes below plus nothing else.

**Drifts.** None in the tests. The pattern replace was checked against the first 200 dataset rows on the CPU: both instruction sentences matched on every row, no "Answer:" survives.

**Validation.** `2 passed in 255.82s` on `ac-agent` (RTX PRO 6000, Qwen3.5-9B): primes both `submitted` in 2 turns with 2441; DAPO 2 tasks × 2, all `submitted`, mean score 1.00, second call from the cache. Full output `IB/TMP/AGENT_ROLLOUTS/test_report_run4.txt`; the results table is in `SLICE1_ROUND.tressoir.md`. A 100-problem × group-2 probe per model (`bench/agent_probes/dapo_bench.py`, Qwen3.5-4B and 9B) is running as this is written; its summary goes to `IB/TMP/SYNC/DAPO_BENCH/<model>/summary.json`.

#### Planning Overview (as planned)

The loader streams the train split, dedupes on `extra_info.index` and stops at `max_examples` unique problems (17,917 exist; the file holds each 100 times). Each becomes a `DatasetTask` with `gold_answer = reward_model.ground_truth` (integers as strings), `NUMERIC_EXACT`, and `agent_prompt` = the problem text without the dataset's "Answer:" instruction. `numeric_exact` takes the last number of the prediction (`\boxed{}`, commas, dollar signs stripped) and compares it as an exact fraction. The tests run on `Qwen/Qwen3.5-9B` with thinking off (harness default), ten turns, group size 2, a reporter under the synced folder so `sky exec --sync` shows the page here while the node runs. The DAPO test also proves the cache: the same call again returns the same rollouts without loading the engine.

#### Planned Changes (as planned)

`activation/dataset/loaders/dapo_math.py · DapoMathDataset.load()`

```diff
+class DapoMathDataset:
+    @classmethod
+    def load(cls, harness, max_examples: int|None, seed: int = 0) -> LoadedDataset:
+        rows = load_dataset("BytedTsinghua-SIA/DAPO-Math-17k", split="train", streaming=True)
+        unique by extra_info.index until max_examples; DatasetTask(task_id=index, dataset_id="dapo_math", task_datum=row,
+            reference_metrics_kind=DatasetTaskMetricsKind.NUMERIC_EXACT, gold_answer=row["reward_model"]["ground_truth"],
+            agent_prompt=_problem_text(row["prompt"][0]["content"]))
+        return LoadedDataset(dataset_id="dapo_math", documents={}, scorable_tasks={...}, stats=...)
```

`activation/dataset/scoring.py · numeric_exact()` and `activation/dataset/dataset.py · DatasetTask.score()`

```diff
 def numeric_exact(pred, golds) -> float:
-    """@AI: exact numetic check. Returns 0.0/1.0"""
-    pass
+    value = _last_number(pred)            # \boxed{}, commas, $ and spaces stripped; Fraction
+    return 1.0 if value is not None and any(value == _as_fraction(g) for g in golds) else 0.0
+        if metric == DatasetTaskMetricsKind.NUMERIC_EXACT:
+            return scoring.numeric_exact(answer, golds)
```

`activation/tests/test_basic_agent.py` (exact expected diff against your stub; the staged file is next to the round document)

```diff
--- a/activation/tests/test_basic_agent.py
+++ b/activation/tests/test_basic_agent.py
@@ -1,22 +1,140 @@
+"""
+Two end-to-end agent rollouts on a GPU node.
+
+    uv run sky exec --sync <node> -- uv run pytest activation/tests/test_basic_agent.py --gpu --slow -s
+
+The reporters write under the synced folder, so IB/TMP/SYNC/AGENT_TEST/{primes,dapo}/report.tressoir.html
+morphs here every 15 s while the node runs.
+"""
+import json
+import os
+import random
+from dataclasses import replace
+
+import pytest
+
+from activation.agent import AgentConfig, RolloutReporter
+from activation.common.data_syncing import resolve_path
+from activation.dataset.loaders import DapoMathDataset
+from activation.harness import FREE_DEVICE, HarnessRuntime, HarnessRuntimeConfig, ModelConfig
+
+MODEL_NAME = "qwen3.5-9b"
+MODEL_ID = "Qwen/Qwen3.5-9B"
+
 BASE_AGENT_CONFIG = AgentConfig(
-    # AgentConfig with just the default tools.
-    # Based on Qwen/Qwen3.5-9B.
-    # Limit to 10 steps. No thinking.
+    # Default tools only (shell, python, parallel_tool_call, submit_answer). Thinking is off by the harness default.
+    system_prompt=(
+        "You are a careful problem solver working in a sandbox. Use the python or shell tools to compute "
+        "rather than guessing. When you are done, call submit_answer exactly once with the final answer."
+    ),
+    model_name=MODEL_NAME,
+    call_kwargs={"sampling_params": {"max_tokens": 2048, "temperature": 0.7}},
+    max_turns=10,
+    max_tool_errors=3,
+    max_duration=300,
 )
 
+
+def _make_harness() -> HarnessRuntime:
+    return HarnessRuntime(HarnessRuntimeConfig(
+        model_configs={MODEL_NAME: ModelConfig(MODEL_NAME, MODEL_ID)},
+        agent_max_concurrent=8,
+    ))
+
+
+def _print_rollout(result) -> None:
+    print(
+        f"\n=== rollout seed={result.seed} finish={result.finish_reason} turns={result.num_turns} "
+        f"tokens in/cached/out={result.num_input_tokens}/{result.num_cached_input_tokens}/{result.num_output_tokens} "
+        f"duration={result.duration:.1f}s score={result.score:.2f} answer={result.answer!r}"
+    )
+    for step in result.trajectory:
+        if step["role"] == "assistant":
+            calls = ", ".join(f"{call['name']}({json.dumps(call['arguments'])[:80]})" for call in step["tool_calls"])
+            print(f"  assistant: {step['content'][:200]!r} -> {calls}")
+        else:
+            for output in step["tool_call_results"]:
+                print(f"  tool: {output[:200]!r}")
+
+
+@pytest.mark.gpu
+@pytest.mark.slow
 def test_basic_agent_primes():
-    # @AI: key test: give it to me in full.
-    ... # Simple task to compute the sum of the 140th, 141st and 142nd prime numbers.
-    ... # rollout group size 2, with reporting. No attached dataset.
+    """A self-contained task, no dataset: a group of 2 rollouts with the live report."""
+    harness = _make_harness()
+    config = replace(
+        BASE_AGENT_CONFIG,
+        user_prompt="Compute the sum of the 140th, 141st and 142nd prime numbers (2 is the 1st prime).",
+    )
+    reporter = RolloutReporter(
+        str(resolve_path("AGENT_TEST/primes")),
+        title="Agent test: primes",
+        description=f"{MODEL_ID}, group of 2, {config.max_turns} turns max, default tools.",
+    )
+    try:
+        groups = harness.rollout_manager.perform_grouped_rollouts(
+            [config], group_count=2, base_seed=0, perform_scoring=False, reporter=reporter,
+        )
+    finally:
+        harness.loaded_models[MODEL_NAME].engine_to_device(FREE_DEVICE)
+
+    assert len(groups) == 1 and len(groups[0]) == 2
+    for result in groups[0]:
+        _print_rollout(result)
+        assert result.finish_reason == "submitted", result.finish_reason
+        assert any(
+            call["name"] in ("python", "shell", "parallel_tool_call")
+            for step in result.trajectory for call in step.get("tool_calls", [])
+        ), "no tool was used"
+        assert str(result.answer).replace(",", "").strip().lstrip("-").isdigit(), result.answer
+    answers = [int(str(result.answer).replace(",", "").strip()) for result in groups[0]]
+    print(f"answers {answers}; expected 2441 (809 + 811 + 821)")
+    assert 2441 in answers, answers  # at least one of the two rollouts gets it (drop this line to print only)
+    assert os.path.exists(os.path.join(reporter.report_folder, "report.tressoir.html"))
+    assert os.path.exists(os.path.join(reporter.report_folder, "report_data.json"))
+    print("\n=== harness stats ===")
+    print(json.dumps(harness.harness_stats.summarize(), indent=2))
 
+
+@pytest.mark.gpu
+@pytest.mark.slow
 def test_basic_agent_dapo():
-    # @AI: key test: give it to me in full.
-    harness = ...
-    dapo_math_dataset = ...
-    dataset_tasks = ... # Shuffle, then select 2
-    rollout_manager = harness.rollout_manager
-    rollout_reporter = ... # Allow me to live view the rollouts.
-    rollout_manager.perform_grouped_rollouts(
-        ..., # group size 2
+    """Two DAPO-Math tasks, a group of 2 each, scored; then the same call served from the cache."""
+    harness = _make_harness()
+    dataset = DapoMathDataset.load(harness, max_examples=64)
+    harness.dataset_manager.register_dataset(dataset)
+    tasks = list(dataset.scorable_tasks.values())
+    random.Random(0).shuffle(tasks)
+    tasks = tasks[:2]
+    configs = [replace(BASE_AGENT_CONFIG, user_prompt=task.agent_prompt, dataset_task=task) for task in tasks]
+    reporter = RolloutReporter(
+        str(resolve_path("AGENT_TEST/dapo")),
+        title="Agent test: DAPO-Math",
+        description=f"{MODEL_ID}, 2 tasks x group of 2, {BASE_AGENT_CONFIG.max_turns} turns max, default tools.",
     )
-    # Some display. Problem need not be solved.
\ No newline at end of file
+    try:
+        groups = harness.rollout_manager.perform_grouped_rollouts(
+            configs, group_count=2, base_seed=0, perform_scoring=True, caching_id="agent_test_dapo", reporter=reporter,
+        )
+        loads_before = len(harness.harness_stats.model_loading_times)
+        # The same request again: every rollout comes from the cache, the engine is not touched.
+        cached = harness.rollout_manager.perform_grouped_rollouts(
+            configs, group_count=2, base_seed=0, perform_scoring=True, caching_id="agent_test_dapo",
+        )
+    finally:
+        harness.loaded_models[MODEL_NAME].engine_to_device(FREE_DEVICE)
+
+    assert len(groups) == 2 and all(len(group) == 2 for group in groups)
+    for task, group in zip(tasks, groups):
+        print(f"\n### task {task.task_id}: gold {task.gold_answer}\n{task.agent_prompt[:300]}")
+        for result in group:
+            _print_rollout(result)  # the problem need not be solved
+            assert result.finish_reason in ("submitted", "max_turns", "max_tool_errors", "max_duration", "no_tool_call")
+            assert result.score in (0.0, 1.0)
+            assert result.agent_config.dataset_task.task_id == task.task_id
+    print(f"scores {[[result.score for result in group] for group in groups]}")
+    assert len(harness.harness_stats.model_loading_times) == loads_before
+    assert [[r.answer for r in group] for group in cached] == [[r.answer for r in group] for group in groups]
+    assert os.path.exists(os.path.join(reporter.report_folder, "report.tressoir.html"))
+    print("\n=== harness stats ===")
+    print(json.dumps(harness.harness_stats.summarize(), indent=2))
```

</details>

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">Slice 2 — Trajectory harvesting and SFT</span>
    <span class="card-oneliner">Your own note in rollout_manager.py; shaped after slice 1 runs.</span>
    <span class="card-badge">TBD</span>
  </summary>

Rejection sampling over grouped rollouts, harvesting the trajectories the cache holds, SFT with a LoRA on the serving engine, and context compaction. Not planned further until slice 1 has produced rollouts.

</details>

## Validation

What actually ran (2026-09-07):

- **Node `ac-agent`** (RTX PRO 6000, us-east-2, fallback image + podman setup): four runs of `pytest activation/tests/test_basic_agent.py --gpu --slow -s -x` through `sky exec --sync`. Run 1 was lost to my launcher; run 2 failed on the tool-call format (both primes rollouts `max_tool_errors`, fixed by `agent_utils.py`); run 3 solved primes (2 turns, 2441 twice) and failed on `reporter.report_folder`; run 4: `2 passed in 255.82s`, DAPO mean score 1.00, cache proven. Report: `IB/TMP/AGENT_ROLLOUTS/test_report_run4.txt`; pages `IB/TMP/SYNC/AGENT_TEST/{primes,dapo}/report.tressoir.html`.
- **CPU** (`IB/TMP/AGENT_ROLLOUTS/cpu_check.py`, scripted engine, local fake env, the real Qwen3.5 tokenizer and chat template): every agent path listed in the M3 and M4 reports; `CPU CHECK OK`.
- **Not rerun today:** the study test on the unified engine (M1's regression gate for `chat()`), rootless podman on your host (documented in the README section).
- **DAPO probe** (`bench/agent_probes/dapo_bench.py`, 100 problems × group 2, one node per model): Qwen3.5-4B accuracy 0.725, 9B 0.795; 17 mixed-outcome problems each, 29 mixed for at least one model, 58 solved by both every time, 10 by neither; failures are budget exhaustion (turns, tool errors, time) far more than wrong submissions. Full tables in `SLICE1_ROUND.tressoir.md`; pages under `IB/TMP/SYNC/DAPO_BENCH/`. Nodes torn down.
