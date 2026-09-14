# Agent training, slice 2: the handoff

Round 1 (2026-09-08). M0–M3 of `SLICE2_PLAN.tressoir.md` implemented and validated: the agent speaks tokens between turns (segments), the serving engine takes a LoRA per request and sleeps while the trainer holds the GPU, one clipped-surrogate trainer over weighted trajectories, four selection functions, and the two key tests, which passed on a fresh RTX PRO 6000 node (`ac-train`, torn down afterwards). Each milestone card carries its completion report (what landed, drifts, focused diffs, validation); this document is the application handoff, the test results, and the exact per-file deltas.

Application update (2026-09-08): the workspace application includes only `test_basic_agent_training.py` from the tests and probes. The dialect test and both probes stay under `slice2/` in IB for private validation; their diff cards are omitted below. `AgentRunResult.lora_name` is retained after checking the optional feedback: the config names the requested adapter, while the result records the adapter actually applied (`None` before the first exchange, even when the config names an adapter).

## Application handoff (for your workspace agent)

Order of work, in your checkout of `/source` (commit `9a5843e` plus your uncommitted agent_training stubs):

1. **The deletion first** (block below): the stub `agent_training_optimization.py` is renamed to `agent_training_utils.py`.
2. **Copy the implementation and the key test only**: `rsync -a --exclude=bench/ --exclude=tests/ IB/ARTIFACTS/AGENT_ROLLOUTS/slice2/activation/ activation/`, then `cp IB/ARTIFACTS/AGENT_ROLLOUTS/slice2/activation/tests/test_basic_agent_training.py activation/tests/`. Copy the staged `pyproject.toml` (`flash-linear-attention` and the PEFT minimum version) and `uv.lock`. Keep all other slice-2 tests and probes in IB. The cards are the fallback if a local file has drifted.
3. **CPU checks**: `uv run python -m compileall -q activation`; `uv run pytest IB/ARTIFACTS/AGENT_ROLLOUTS/slice2/activation/tests/test_agent_dialect.py` (2 passed, needs the Qwen3.5-4B and Qwen3-0.6B tokenizers); `uv run python IB/TMP/AGENT_ROLLOUTS/slice2_cpu_check.py` ends with `SLICE2 CPU CHECK OK` (Qwen3-0.6B, fabricated trajectories, three training rounds on CPU).
4. **Node**: `uv run sky exec --sync <node> -- uv run pytest activation/tests/test_basic_agent_training.py --gpu --slow -s -x`. Expected: `2 passed` in about 60–95 min; the pages appear under `IB/TMP/SYNC/AGENT_TRAINING_TEST/` while it runs; the adapters under `IB/TMP/SYNC/LORAS/dapo_group_mean/`. The node's venv picks up the new lock at the first `uv run` (`UV_NO_SYNC=0`), no image rebuild needed.
5. **Optional probes remain in IB**: `slice2/activation/bench/agent_probes/{lora_engine_probe,training_profile}.py` are reproducibility tools, excluded from the main-code application. Historical node probe results below remain evidence; no cloud runs are required for the local application.

The two tests now write `single_*` and `multi_*` report and cache prefixes (`single_rollouts_round0`, `single_train_round0`, `single_rollouts_round1`; `multi_rollouts_round{0,1,2}`, `multi_train_round{0,1}`) instead of the round 10–12 numbering of this run.

Things that look wrong but are not: the loss printed per step is near zero with group-mean weights (weights sum to zero per group and the ratios start at one; the gradient is what moves); the trajectory JSON files under `rollouts_round*/trajectories/` carry `score 0` because they are dumped before scoring (the report table has the scores); `EngineDeadError` lines at process exit are vLLM's shutdown noise after a `SUCCEEDED` job.

## Test results

`uv run sky exec --sync ac-train -- uv run pytest activation/tests/test_basic_agent_training.py --gpu --slow -s -x` on `Qwen/Qwen3.5-4B`, thinking off, default tools in `python:3.12-slim` sandboxes, LoRA `dapo_group_mean` rank 64, `group_mean_advantage`, two updates per round, learning rate 2e-5, clip 0.2. Result: `2 passed in 5698.01s (1:34:58)`. Cleaned pytest output `IB/TMP/AGENT_ROLLOUTS/training_tests_run1_report.txt` (full log `training_tests_run1.log`); pages and uncut trajectories `IB/TMP/SYNC/AGENT_TRAINING_TEST/{rollouts_round0,train_round0,rollouts_round1,rollouts_round10,train_round10,rollouts_round11,train_round11,rollouts_round12}/report.tressoir.html`; adapters `IB/TMP/SYNC/LORAS/dapo_group_mean/{round_000,round_001,latest}` (325 MB each); rollout caches `IB/TMP/SYNC/ROLLOUTS/agent_training_test_r{0,1,10,11,12}/`.

`activation/tests/test_basic_agent_training.py` · pytest output, run 1 (engine logs and sync lines removed)

```
activation/tests/test_basic_agent_training.py
qwen3.5-4b - Engine moved to target in 22.23s                       # round-0 rollouts, 128 in 8m13s, mean score 0.570
qwen3.5-4b - Engine asleep in 4.59s
qwen3.5-4b - Moved model in 1.73s                                   # base + fresh adapter on the GPU
AgentTrainer - round 0 step 1/2: loss -0.0417 ratio 1.000 clip 0.001 logprob -0.114 grad 0.043 122131 tokens in 163.6s
AgentTrainer - round 0 step 2/2: loss 0.0000 ratio 1.003 clip 0.004 logprob -0.116 grad 0.051 130670 tokens in 179.7s
AgentTrainer - round 0 done: items 43, examples 42 (1 over 16,384 tokens), 252,801 tokens, +7.75 / -6.875, 359.7 s, checkpoint LORAS/dapo_group_mean/round_000, peak 39.4 GB
qwen3.5-4b - Engine woke up in 0.40s                                # round-1 rollouts with the adapter, 8m52s, 0.656
accuracy before 0.570 -> after one round 0.656 (16 x 8; printed, not asserted)
.
qwen3.5-4b - Engine moved to target in 21.33s                       # test 2: round 10 (base) 33m52s (stalled, see below), 0.391
AgentTrainer - round 0 step 1/2: loss 0.0803 ratio 1.000 clip 0.002 logprob -0.130 grad 0.061 173418 tokens in 240.0s
AgentTrainer - round 0 step 2/2: loss -0.0798 ratio 0.997 clip 0.004 logprob -0.131 grad 0.069 177219 tokens in 262.5s
qwen3.5-4b - Engine woke up in 0.40s                                # round 11 with round_000, 8m48s, 0.578
AgentTrainer - round 1 step 1/2: loss -0.0937 ratio 1.000 clip 0.002 logprob -0.127 grad 0.040 215963 tokens in 276.3s
AgentTrainer - round 1 step 2/2: loss 0.0942 ratio 1.006 clip 0.008 logprob -0.122 grad 0.043 207033 tokens in 271.8s
qwen3.5-4b - Engine woke up in 0.41s                                # round 12 with round_001, 8m26s, 0.648
accuracy by round [0.391, 0.578, 0.648]; step-1 ratios [1.0, 1.0]
.
======================== 2 passed in 5698.01s (1:34:58) ========================
```

| phase | model | anypass@8 | mean@8 | allpass@8 | mixed groups | finish: submitted / max_turns / tool errors / duration | elapsed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| test 1, round 0 | base | 11/16 | 0.570 | 5/16 | 6 | 81 / 33 / 9 / 5 | 8m13s |
| test 1, round 1 | round_000 | 14/16 | 0.656 | 6/16 | 8 | 94 / 19 / 6 / 9 | 8m52s |
| test 2, round 10 | base, stalled | 10/16 | 0.391 | 3/16 | 7 | 57 / 14 / 8 / 48 | 33m52s |
| test 2, round 11 | round_000 | 12/16 | 0.578 | 3/16 | 9 | 83 / 18 / 5 / 22 | 8m48s |
| test 2, round 12 | round_001 | 12/16 | 0.648 | 8/16 | 4 | 95 / 20 / 8 / 4 | 8m26s |

What the run says about the harness: the mechanics hold end to end (segments and log-probs on every assistant step; group-mean weights summing to zero per group; the engine's and the trainer's probabilities agree at step 1, ratio 1.000 with 0.1–0.2% of tokens outside the clip; sleep 4.6 s, wake 0.4 s; checkpoints loadable by both PEFT and vLLM; two consecutive rounds through `exchange_lora`). The accuracies are not measurements: sixteen problems and one seed set move the base model alone across 0.57–0.66, and the trainer saw only 43–72 items per round. The direction is the expected one (fewer `max_turns`, more `submitted`) and nothing more should be read into it.

**The round-10 stall.** In that phase 48 rollouts finished in the first 150 s at the usual pace, then 25 in the next nineteen minutes, then the remaining 55 at normal pace; 48 runs ended with `max_duration` because a run's budget is only checked between turns. Cause: one rollout's Python call printed about 2 GB. The model saw 20k characters, but `truncate_output` kept the whole text as an activation-context output on the trajectory step, and the reporter re-serialised it into `trajectories/bb29dede.json` at every step (1.97 GB per write, `rollouts_round10` 19 GB on disk), the rollout cache wrote it (2.2 GB), and the JSON encoding held the GIL of the process that runs all 128 agents, so the engine sat idle. Fixed in this handoff: the sandbox caps captured output at 4 MB head and tail (`agent_env.py`, `OUTPUT_CAPTURE_LIMIT_CHARS`), and the activation-context copy and the env file at four times the 20k characters the model sees (`agent_tools.py`, `AC_OUTPUT_LIMIT_CHARS = 4 * TOOL_OUTPUT_LIMIT_CHARS`, your call). Not yet added: a per-request timeout, for the M5 driver. Delete `IB/TMP/SYNC/AGENT_TRAINING_TEST/rollouts_round10/trajectories/bb29dede.json*` and `IB/TMP/SYNC/ROLLOUTS/agent_training_test_r10/` if you want the 21 GB back.

## Training throughput: the poles in the tent

Measured on `ac-prof2` with `activation/bench/agent_probes/training_profile.py` on the cached round-0 rollouts (46 examples, 282k tokens, lengths 1.7k–12.6k, median 5.4k): the trainer's own step (collate, decoder with the adapter, fp32 head, clipped surrogate, backward) over six micro-batches per variant, one warm-up excluded, then the same list again (cold and warm passes agreed to 0.1 s everywhere, so no per-shape compile cost was measurable on this node). Logs `IB/TMP/AGENT_ROLLOUTS/training_profile_run{1,2}.log`, tables `IB/TMP/SYNC/TRAINING_PROFILE/`.

| variant | real tokens/s | padding | peak GB | what it changes |
| --- | --- | --- | --- | --- |
| padded_32k (the slice-2 test's configuration) | 2,983 | 4.2% | 41.3 | 2–4 examples packed to 32k tokens with a padding mask |
| **single** (one example per micro-batch, no mask) | **4,055** | 0 | 28.6 | sdpa takes its causal fast path; nothing wasted on pads |
| single_512 (length rounded up to 512) | 2,890 | 1.6% | 28.8 | 30% slower than single: rounding does not pay |
| single_2048 | 2,754 | 5.3% | 29.1 | |
| single_512_8k (fp32 head in 8k chunks) | 2,885 | 1.6% | 39.0 | the head chunk size is not a pole |
| single_eager (eager attention) | 2,843 | 1.6% | 55.3 | attention implementation is not a pole once the mask is gone |
| single_no_ckpt (no gradient checkpointing) | out of memory | | | at 12k tokens on a 96 GB card |

What the profiler says about the padded configuration: the attention layers ran PyTorch's memory-efficient kernel with an explicit mask (forward 10%, backward 23% of GPU time); dtype copies and elementwise work another 23%; the fp32 head's GEMMs about 5%; the gated-delta-net kernels themselves under 2%. Without the mask the attention rows leave the top of the table.

The poles, in order, and what was done:

1. **The fallback gated-delta-net kernels** (library): transformers ran its pure-PyTorch implementation for 24 of 32 layers because `fla` was not installed. `flash-linear-attention` added to `pyproject.toml`: step 179.7 → 74.6 s on the same tokens (2.4×).
2. **The padding mask** (`agent_training_utils.py`): packing examples under a token budget forced a 2-D padding mask, which sends `sdpa` off its flash path. `pad_free_micro_batches=True` (default): one example per micro-batch and no mask to the decoder, since right padding never reaches a real token under causal attention. 2,983 → 4,055 tokens/s on the same data (1.36×), peak memory 41 → 29 GB.
3. **Length rounding** tried and rejected: `length_multiple` stays a knob, default 1.
4. Not poles: the fp32 head chunk size, the attention implementation, and gradient checkpointing (needed for memory at these lengths).

Round-0 training in the key test, same 42 examples and 252,801 tokens each time:

| configuration | step 1 (122k tokens) | step 2 (131k tokens) | round | round tokens/s | peak GB |
| --- | --- | --- | --- | --- | --- |
| slice-2 test run 1: torch fallback kernels, packed 32k micro-batches | 163.6 s | 179.7 s | 359.7 s | 703 | 39.4 |
| run 2: `flash-linear-attention`, packed micro-batches | 137.5 s | 74.6 s | 230.8 s | 1,095 | 39.4 |
| run 3: fla + one example per micro-batch, no mask (the defaults now) | 70.8 s | 44.7 s | 126.4 s | 2,000 | 27.5 |

Step 2 is the steady state, 2,920 tokens/s; step 1 carries the process warm-up. The run-3 test passed end to end (`timed_round_run1.log`: round-0 rollouts from the cache, training, round-1 rollouts with the adapter at 0.625 after one round, `1 passed`), so the change is validated on the real path, not only in the probe.

At the sized round (1024 rollouts, about a third selected, roughly 2M training tokens) that is about ten minutes of training against an hour of generation. The remaining gap to the card's arithmetic is about 15%.

## The one deletion (run first, in your checkout)

`activation/agent_training/` · the renamed file

```bash
rm activation/agent_training/agent_training_optimization.py   # untracked stub; its role is now agent_training_utils.py
```

The cards below are exact deltas against `/source` as of your commit `9a5843e` plus your uncommitted prototype stubs; the implementation and key-test copies under `slice2/` next to this document are byte-for-byte what the cards produce. Use the filtered copy commands above; the extra tests and probes in that tree stay in IB. The generated `uv.lock` is supplied alongside the source.

## Diffs to apply

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">activation/agent/agent_utils.py</span>
    <span class="card-oneliner">The token-level dialect: prompt_token_ids, continuation_token_ids (template wrapper via a sentinel), join_continuation.</span>
    <span class="card-badge">Diff</span>
  </summary>

Exact delta vs `/source` for `activation/agent/agent_utils.py`:

```diff
--- a/activation/agent/agent_utils.py
+++ b/activation/agent/agent_utils.py
@@ -6,6 +6,16 @@ model's chat template, so the same loop drives a Qwen3.5 (XML function blocks),
 model (JSON in <tool_call>) or a template without tool support (everything inlined as text).
 Parsing is adaptive per block: every registered format is tried, the dialect only decides which
 one is shown in error messages and used when rendering inline.
+
+The dialect also owns the token-level view of a conversation (slice 2): the first prompt is templated
+once, then the agent appends its own sampled tokens verbatim and only the wrapper the template puts
+between a finished assistant turn and the next generation prompt (`continuation_token_ids`). The
+model therefore reads exactly the tokens it wrote, one trajectory is one training sequence, and
+prefix-cache hits are exact. `continuation_token_ids` renders the wrapper from the chat template with a
+sentinel in place of the assistant text, so it needs no assumptions about the template's wording. Note
+that templates re-render *history* in ways the model never saw while sampling (Qwen3 drops the empty
+`<think>` block of finished turns, Qwen3.5 once a user message follows); the segments keep the
+generation-prompt form the model actually read, which is what training must see.
 """
 from __future__ import annotations
 
@@ -19,6 +29,7 @@ if t.TYPE_CHECKING:
     from .agent_tools import ToolCallResult
 
 PARSE_ERROR_TOOL = "parse_error"
+ASSISTANT_SENTINEL = "\u241f TRESSOIR_ASSISTANT_TURN \u241f"   # never in real text; marks where the sampled tokens sit in a rendering
 TOOL_CALL_BLOCK = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)
 TOOL_CALL_OPEN = re.compile(r"<tool_call>(?!.*</tool_call>)(.*)$", re.S)   # an unterminated block (max_tokens hit)
 ParameterTypes = dict[str, dict[str, str]]   # tool name -> parameter name -> JSON schema type
@@ -243,3 +254,60 @@ class ModelDialect:
     def parse_error_message(self, arguments: dict) -> str:
         return (f"Could not parse the tool call ({arguments.get('error')}). Expected shape: "
                 f"{arguments.get('expected') or self.preferred_format.reminder}.")
+
+    # ------------------------------------------------------------------------------- tokens
+    def prompt_token_ids(self, tokenizer, messages: list[dict], tools: list[dict] | None = None,
+                         chat_template_kwargs: dict | None = None) -> list[int]:
+        """The first prompt of a conversation: the chat template, once, with the generation prompt open."""
+        return _template_ids(tokenizer, messages, tools, True, chat_template_kwargs)
+
+    def continuation_token_ids(self, tokenizer, messages_before: list[dict], new_messages: list[dict],
+                               tools: list[dict] | None = None, chat_template_kwargs: dict | None = None) -> list[int]:
+        """
+        The tokens the model reads between its own sampled text and its next turn: the end-of-turn
+        marker, `new_messages` (tool results, or the nudge) and the next generation prompt, exactly as
+        the chat template renders them. Rendered with a sentinel as the assistant content, so the
+        wrapper is independent of what the model wrote. Starts with the end-of-turn token(s); the caller
+        drops the first one when the sample already ended with it.
+        """
+        history = list(messages_before) + [{"role": "assistant", "content": ASSISTANT_SENTINEL}] + list(new_messages)
+        text = _template_text(tokenizer, history, tools, True, chat_template_kwargs)
+        index = text.rfind(ASSISTANT_SENTINEL)
+        if index < 0:
+            raise ValueError("the chat template did not render the assistant content verbatim; cannot derive the turn wrapper")
+        wrapper = text[index + len(ASSISTANT_SENTINEL):]
+        return list(tokenizer.encode(wrapper, add_special_tokens=False))
+
+    @staticmethod
+    def join_continuation(sampled_token_ids: list[int], wrapper_token_ids: list[int]) -> list[int]:
+        """The wrapper without its leading end-of-turn token when the sample already produced it."""
+        if sampled_token_ids and wrapper_token_ids and sampled_token_ids[-1] == wrapper_token_ids[0]:
+            return wrapper_token_ids[1:]
+        return wrapper_token_ids
+
+
+def _template_text(tokenizer, messages: list[dict], tools, add_generation_prompt: bool, chat_template_kwargs: dict | None) -> str:
+    return tokenizer.apply_chat_template(
+        _flatten(messages), tools=tools, add_generation_prompt=add_generation_prompt, tokenize=False, **(chat_template_kwargs or {}),
+    )
+
+
+def _template_ids(tokenizer, messages: list[dict], tools, add_generation_prompt: bool, chat_template_kwargs: dict | None) -> list[int]:
+    ids = tokenizer.apply_chat_template(
+        _flatten(messages), tools=tools, add_generation_prompt=add_generation_prompt, tokenize=True, **(chat_template_kwargs or {}),
+    )
+    if hasattr(ids, "keys"):                                                      # BatchEncoding / dict
+        ids = ids["input_ids"]
+    return list(ids)
+
+
+def _flatten(messages: list[dict]) -> list[dict]:
+    """Text-only content parts joined, the shape our templates take."""
+    out = []
+    for message in messages:
+        message = dict(message)
+        content = message.get("content")
+        if isinstance(content, list):
+            message["content"] = "".join(part.get("text", "") for part in content if isinstance(part, dict))
+        out.append(message)
+    return out
```

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">activation/agent/agent_config.py</span>
    <span class="card-oneliner">record_sampling; TrajectoryStep.token_ids / logprobs; AgentRunResult.prompt_token_ids / source / lora_name.</span>
    <span class="card-badge">Diff</span>
  </summary>

Exact delta vs `/source` for `activation/agent/agent_config.py`:

```diff
--- a/activation/agent/agent_config.py
+++ b/activation/agent/agent_config.py
@@ -28,6 +28,7 @@ class AgentConfig:
     model_name: str | None = None                             # A harness model name; its engine serves the rollout.
     lora_name: str | None = None                              # Adapter on the serving engine (plumbed, untested).
     call_kwargs: dict | None = None                           # Engine chat kwargs (sampling_params inside), merged like engine_chat_many.
+    record_sampling: bool = True                              # Ask the engine for the sampled token ids and their log-probs (training needs them).
 
     # Environment. Every agent gets the shell / python / parallel_tool_call / submit_answer tools.
     env_dockerfile_path: str | None = None                    # None: the harness default image.
@@ -88,11 +89,14 @@ class TrajectoryStep:
     """
     Contains simple formats the model expects (e.g., raw dicts rather than our objects).
     """
-    role: str                                                  # "assistant" or "tool"
+    role: str                                                  # "assistant" | "tool" | "user" (the nudge)
     content: str                                               # assistant text (tool-call blocks removed) or the joined tool outputs
     tool_calls: list[dict] = field(default_factory=list)       # [{"id", "name", "arguments"}]
     tool_call_results: list[str] = field(default_factory=list) # truncated outputs, same order as tool_calls
     activations: dict[str, object] = field(default_factory=dict)  # ac_outputs of this step's tools, keyed by call id
+    # Token segment (slice 2): the prompt the model read is AgentRunResult.prompt_token_ids + every step's token_ids in order.
+    token_ids: list[int] = field(default_factory=list)         # assistant: the sampled tokens verbatim; tool/user: the template's wrapper up to the next generation prompt
+    logprobs: list[float] = field(default_factory=list)        # assistant only: the engine's log-prob of each sampled token (pi_old); empty when not recorded
 
 
 @dataclass
@@ -110,8 +114,11 @@ class AgentRunResult:
     score: float = 0.0
     score_feedback: str | None = None
     subagent_results: list["AgentRunResult"] = field(default_factory=list)
-    finish_reason: str = ""                                    # submitted | max_turns | max_tool_errors | max_duration | no_tool_call | error
+    finish_reason: str = ""                                    # submitted | max_turns | max_tool_errors | max_duration | no_tool_call | context_exceeded | error
     seed: int = 0
+    prompt_token_ids: list[int] = field(default_factory=list)  # the first turn's prompt (system + task + tools, templated once)
+    source: str = "policy"                                     # "policy" | "hinted:<model>" | "oracle:<model>": who produced this run
+    lora_name: str | None = None                               # the adapter the engine actually applied (None: the base model)
 
     @property
     def num_tool_calls(self) -> int:
```

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">activation/agent/agent.py</span>
    <span class="card-oneliner">The loop keeps the token prefix: sampled tokens appended verbatim, dialect wrappers for tool results and the nudge (recorded as a user step), engine_submit_tokens with the adapter.</span>
    <span class="card-badge">Diff</span>
  </summary>

Exact delta vs `/source` for `activation/agent/agent.py`:

```diff
--- a/activation/agent/agent.py
+++ b/activation/agent/agent.py
@@ -1,7 +1,13 @@
 """
-The agent turn loop: messages -> one engine request -> tool calls -> tool results -> next turn, until
-the answer is submitted or a budget runs out. One Agent owns one AgentRunResult; the rollout manager
-runs many agents on threads and the engine batches their requests.
+The agent turn loop: prompt tokens -> one engine request -> tool calls -> tool results -> next turn,
+until the answer is submitted or a budget runs out. One Agent owns one AgentRunResult; the rollout
+manager runs many agents on threads and the engine batches their requests.
+
+The conversation is kept twice: as messages (the readable record for reports and the cache) and as
+the token prefix the engine actually reads. The prefix is templated once for the first turn; after
+that the agent appends its own sampled tokens verbatim and the dialect's wrapper for the tool results
+or the nudge, so the model sees exactly what it wrote, every step records its token segment, and one
+trajectory is one training sequence (agent_training).
 """
 from __future__ import annotations
 
@@ -48,6 +54,7 @@ class Agent:
         self.agent_id = uuid.uuid4().hex  # Pins the agent to one engine replica, so its prefix cache serves the next turn.
         self.lock = threading.Lock()      # Parallel subagent result appends.
         self.messages: list[dict] = []
+        self.prefix: list[int] = []        # the token prompt of the next turn
         self.tools: dict[str, AgentTool] = {}
         self.finished = False
         self._initialize_tools()
@@ -105,6 +112,11 @@ class Agent:
         no_tool_turns = 0
         errors = 0
         try:
+            loaded_model = self.loaded_model
+            tokenizer = loaded_model.tokenizer
+            template_kwargs = loaded_model.chat_template_kwargs(config.call_kwargs)
+            self.prefix = self.dialect.prompt_token_ids(tokenizer, self.messages, template_tools, template_kwargs)
+            results.prompt_token_ids = list(self.prefix)
             while True:
                 if results.num_turns >= config.max_turns:
                     results.finish_reason = "max_turns"
@@ -112,17 +124,22 @@ class Agent:
                 if time.time() - start > config.max_duration:
                     results.finish_reason = "max_duration"
                     break
-                output = self.loaded_model.engine_submit(
-                    self.messages, tools=template_tools, seed=self.seed + results.num_turns,
-                    agent_id=self.agent_id, lora_name=config.lora_name, chat_kwargs=config.call_kwargs,
+                output = loaded_model.engine_submit_tokens(
+                    self.prefix, seed=self.seed + results.num_turns, agent_id=self.agent_id, lora_name=config.lora_name,
+                    chat_kwargs=config.call_kwargs, record_sampling=config.record_sampling,
                 )
+                if results.num_turns == 0:
+                    results.lora_name = output.lora_name
                 results.num_turns += 1
                 results.num_input_tokens += output.prompt_token_count
                 results.num_cached_input_tokens += output.cached_prompt_token_count
                 results.num_output_tokens += output.output_token_count
                 content, calls = self.dialect.parse(output.text, parameter_types)
+                messages_before = list(self.messages)
                 self.messages.append(rendering.assistant_message(content, calls))
-                step = TrajectoryStep(role="assistant", content=content, tool_calls=calls)
+                sampled = list(output.token_ids)
+                step = TrajectoryStep(role="assistant", content=content, tool_calls=calls, token_ids=sampled, logprobs=list(output.logprobs or []))
+                self.prefix += sampled
                 if not calls:
                     # A turn without a call gets the nudge; several in a row end the run.
                     results.trajectory.append(asdict(step))
@@ -130,21 +147,29 @@ class Agent:
                     if no_tool_turns >= MAX_CONSECUTIVE_NO_TOOL_TURNS:
                         results.finish_reason = "no_tool_call"
                         break
-                    self.messages.append({"role": "user", "content": self.dialect.nudge_message})
+                    nudge = {"role": "user", "content": self.dialect.nudge_message}
+                    wrapper = self._continuation(tokenizer, messages_before, [nudge], template_tools, template_kwargs, sampled)
+                    self.messages.append(nudge)
+                    results.trajectory.append(asdict(TrajectoryStep(role="user", content=self.dialect.nudge_message, token_ids=wrapper)))
+                    self.prefix += wrapper
                     self._report_step()
                     continue
                 no_tool_turns = 0
                 call_results = self._execute_tool_calls(calls)
                 errors += sum(1 for result in call_results if result.is_error)
+                tool_messages = rendering.tool_messages(calls, call_results)
+                wrapper = self._continuation(tokenizer, messages_before, tool_messages, template_tools, template_kwargs, sampled)
                 tool_step = TrajectoryStep(
                     role="tool",
                     content="\n\n".join(result.output for result in call_results),
                     tool_calls=calls,
                     tool_call_results=[result.output for result in call_results],
                     activations={call["id"]: result.ac_outputs for call, result in zip(calls, call_results) if result.ac_outputs},
+                    token_ids=wrapper,
                 )
                 results.trajectory.extend([asdict(step), asdict(tool_step)])
-                self.messages.extend(rendering.tool_messages(calls, call_results))
+                self.messages.extend(tool_messages)
+                self.prefix += wrapper
                 self._report_step()
                 if any(result.is_final for result in call_results):
                     results.finish_reason = "submitted"
@@ -172,6 +197,12 @@ class Agent:
         if self.reporter is not None:
             self.reporter.report_agent_step(self)
 
+    def _continuation(self, tokenizer, messages_before: list[dict], new_messages: list[dict], template_tools, template_kwargs,
+                      sampled: list[int]) -> list[int]:
+        """The template's wrapper between the sampled turn and the next generation prompt (see agent_utils)."""
+        wrapper = self.dialect.continuation_token_ids(tokenizer, messages_before, new_messages, template_tools, template_kwargs)
+        return self.dialect.join_continuation(sampled, wrapper)
+
     def _count_ac_inputs(self):
         ac_inputs = self.agent_config.ac_inputs
         if not ac_inputs:
```

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">activation/agent/rollout_reporter.py</span>
    <span class="card-oneliner">Trajectory files skip the token lists; the nudge step renders.</span>
    <span class="card-badge">Diff</span>
  </summary>

Exact delta vs `/source` for `activation/agent/rollout_reporter.py`:

```diff
--- a/activation/agent/rollout_reporter.py
+++ b/activation/agent/rollout_reporter.py
@@ -151,7 +151,8 @@ def trajectory_item(agent: "Agent", state: str, folder: Path | None = None) -> d
         (folder / "trajectories").mkdir(parents=True, exist_ok=True)
         _write_atomic(folder / file, json.dumps({
             "agent_id": agent.agent_id, "state": state, "system_prompt": agent.agent_config.system_prompt,
-            "user_prompt": agent.agent_config.user_prompt, "trajectory": results.trajectory,
+            "user_prompt": agent.agent_config.user_prompt,
+            "trajectory": [{key: value for key, value in step.items() if key not in ("token_ids", "logprobs")} for step in results.trajectory],
             "answer": results.answer, "finish_reason": results.finish_reason, "score": results.score,
         }, indent=1, default=str))
     steps = []
@@ -163,12 +164,14 @@ def trajectory_item(agent: "Agent", state: str, folder: Path | None = None) -> d
                 "role": "assistant", "turn": turn, "content": _cut(step["content"], CONTENT_CHARS),
                 "calls": [{"name": call["name"], "arguments": _cut(_arguments(call["arguments"]), ARGUMENT_CHARS)} for call in step["tool_calls"]],
             })
-        else:
+        elif step["role"] == "tool":
             steps.append({
                 "role": "tool", "turn": turn, "content": "",
                 "results": [{"name": call["name"], "output": _cut(output, RESULT_CHARS)}
                             for call, output in zip(step["tool_calls"], step["tool_call_results"])],
             })
+        else:                                                                       # the nudge
+            steps.append({"role": "tool", "turn": turn, "content": "", "results": [{"name": "user", "output": _cut(step["content"], RESULT_CHARS)}]})
     return {
         "id": agent.agent_id, "title": _title(agent), "state": state, "stats": _stats(agent),
         "prompt": _cut(agent.agent_config.user_prompt, PROMPT_CHARS), "steps": steps, "file": file,
```

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">activation/agent/agent_env.py</span>
    <span class="card-oneliner">Captured command output capped at 4 MB head and tail (a runaway print produced 2 GB and stalled a rollout phase).</span>
    <span class="card-badge">Diff</span>
  </summary>

Exact delta vs `/source` for `activation/agent/agent_env.py`:

```diff
--- a/activation/agent/agent_env.py
+++ b/activation/agent/agent_env.py
@@ -12,6 +12,7 @@ import uuid
 from pathlib import Path
 
 DEFAULT_IMAGE = "docker.io/library/python:3.12-slim"
+OUTPUT_CAPTURE_LIMIT_CHARS = 4_000_000   # a command's combined output beyond this keeps its head and tail (a runaway print once produced 2 GB)
 DEFAULT_MEMORY_LIMIT_MB = 8192   # sandbox memory: `podman run --memory` where cgroups exist (hard kill), ulimit -v at 85% per process (MemoryError first)
 SOFT_LIMIT_FRACTION = 0.85
 EXEC_GRACE_SECONDS = 10   # how long the client waits past the in-container `timeout` before killing itself
@@ -95,10 +96,11 @@ class AgentEnv:
         except subprocess.TimeoutExpired:
             process.kill()
             output, _ = process.communicate()
-            return (output or "") + f"\n[timed out after {timeout} s]", False
+            return _cap_output(output or "") + f"\n[timed out after {timeout} s]", False
+        output = _cap_output(output or "")
         if process.returncode == 137 and timeout is not None:
-            return (output or "") + f"\n[timed out after {timeout} s]", False
-        return output or "", process.returncode == 0
+            return output + f"\n[timed out after {timeout} s]", False
+        return output, process.returncode == 0
 
     def run_python_code(self, code: str, timeout: int | None = None) -> tuple[str, bool]:
         """Runs python code (no session persistence). Returns combined stdout/stderr and ok; partial output on timeout."""
@@ -127,3 +129,11 @@ class AgentEnv:
             self.shutdown()
         except Exception:
             pass
+
+
+def _cap_output(text: str, limit: int = OUTPUT_CAPTURE_LIMIT_CHARS) -> str:
+    """Head and tail of a runaway output; nothing downstream (activation context, cache, reports) has to carry more."""
+    if len(text) <= limit:
+        return text
+    half = limit // 2
+    return f"{text[:half]}\n... [output capped: {len(text) - limit} chars dropped by the sandbox] ...\n{text[-half:]}"
```

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">activation/agent/agent_tools.py</span>
    <span class="card-oneliner">The activation-context copy of a long tool output capped at four times what the model sees.</span>
    <span class="card-badge">Diff</span>
  </summary>

Exact delta vs `/source` for `activation/agent/agent_tools.py`:

```diff
--- a/activation/agent/agent_tools.py
+++ b/activation/agent/agent_tools.py
@@ -20,6 +20,7 @@ if t.TYPE_CHECKING:
     from .agent import Agent
 
 TOOL_OUTPUT_LIMIT_CHARS = 20_000
+AC_OUTPUT_LIMIT_CHARS = 4 * TOOL_OUTPUT_LIMIT_CHARS   # kept as activation context (and written to the env file): 4x what the model sees, head and tail beyond
 TOOL_OUTPUT_DIR = "/tmp/agent_outputs"
 @dataclass
 class ToolCallResult:
@@ -33,6 +34,9 @@ def truncate_output(env: "AgentEnv | None", text: str, call_id: str, limit: int
     """Head + tail of a long output; the full text goes to a file in the env and out as activation context."""
     if len(text) <= limit:
         return text, {}
+    if len(text) > AC_OUTPUT_LIMIT_CHARS:
+        keep = AC_OUTPUT_LIMIT_CHARS // 2
+        text = f"{text[:keep]}\n... [{len(text) - AC_OUTPUT_LIMIT_CHARS} chars dropped] ...\n{text[-keep:]}"
     path = f"{TOOL_OUTPUT_DIR}/{call_id}.txt"
     written = env.write_file(path, text) if env is not None else False
     half = limit // 2
```

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">activation/harness/vllm_wrapper.py</span>
    <span class="card-oneliner">sleep / wake_up over the replicas (prefix cache reset on wake); enable_lora no longer forced off by the recommended kwargs.</span>
    <span class="card-badge">Diff</span>
  </summary>

Exact delta vs `/source` for `activation/harness/vllm_wrapper.py`:

```diff
--- a/activation/harness/vllm_wrapper.py
+++ b/activation/harness/vllm_wrapper.py
@@ -58,6 +58,10 @@ class _Replica:
             self._collect(prompt_token_ids, sampling_params, request_id, lora_request), self.loop,
         )
 
+    def run(self, coroutine):
+        """Run one engine coroutine on this replica's loop and wait for it."""
+        return asyncio.run_coroutine_threadsafe(coroutine, self.loop).result()
+
     def shutdown(self):
         try:
             self.engine.shutdown()
@@ -109,8 +113,7 @@ class VLLMWrapper:
             "limit_mm_per_prompt": {"image": 0, "video": 0},
             "kv_cache_dtype": "fp8",
             "gpu_memory_utilization": 0.9,
-            "enable_lora": False,
-        }
+        }   # LoRA support and sleep mode come from LoadedModel.engine_to_device (agents need both)
         mtp = {"speculative_config": {"method": "mtp", "num_speculative_tokens": 2}}
         known = {
             # Dense generator: MTP helps decode.
@@ -204,6 +207,17 @@ class VLLMWrapper:
             futures.append(self.submit(ids, sampling_params, replica=index % len(self.replicas), lora_request=lora_request))
         return [future.result() for future in futures]
 
+    def sleep(self, level: int = 1) -> None:
+        """vLLM sleep mode on every replica: level 1 offloads the weights to host RAM and frees the KV cache."""
+        for replica in self.replicas:
+            replica.run(replica.engine.sleep(level=level))
+
+    def wake_up(self) -> None:
+        """Back on the GPU; the prefix cache is reset since the KV cache was released."""
+        for replica in self.replicas:
+            replica.run(replica.engine.wake_up())
+            replica.run(replica.engine.reset_prefix_cache())
+
     def shutdown(self):
         """Best-effort shutdown of every replica's engine core."""
         for replica in self.replicas:
```

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">activation/harness/loaded_model.py</span>
    <span class="card-oneliner">EngineChatOutput with token ids, log-probs and the adapter; engine_submit_tokens; LoRARequest per request; engine_to_device(cpu) = sleep, target = wake; the model and the engine vacate the GPU for each other.</span>
    <span class="card-badge">Diff</span>
  </summary>

Exact delta vs `/source` for `activation/harness/loaded_model.py`:

```diff
--- a/activation/harness/loaded_model.py
+++ b/activation/harness/loaded_model.py
@@ -4,7 +4,7 @@ import torch
 from torch import nn
 import gc
 from copy import deepcopy
-from dataclasses import dataclass
+from dataclasses import dataclass, field
 
 from transformers import (
     AutoModel,
@@ -38,16 +38,29 @@ class EngineChatOutput:
     output_token_count: int
     cached_prompt_token_count: int = 0   # prefix-cache hits reported by the engine
     finish_reason: str = ""              # "stop" or "length"
+    token_ids: list[int] = field(default_factory=list)   # the sampled tokens (end-of-turn included when the model produced it)
+    logprobs: list[float] | None = None                  # log-prob of each sampled token when requested (SamplingParams.logprobs=0)
+    lora_name: str | None = None                         # the adapter the request carried (None: base)
 
     @staticmethod
-    def from_request_output(output: "vllm.RequestOutput") -> "EngineChatOutput":
+    def from_request_output(output: "vllm.RequestOutput", lora_name: str | None = None) -> "EngineChatOutput":
         completion = output.outputs[0]
+        token_ids = list(completion.token_ids)
+        logprobs = None
+        if completion.logprobs:
+            logprobs = []
+            for entry, token_id in zip(completion.logprobs, token_ids):
+                item = entry.get(token_id) if hasattr(entry, "get") else None
+                logprobs.append(float(item.logprob) if item is not None else float(next(iter(entry.values())).logprob))
         return EngineChatOutput(
             text=completion.text,
             prompt_token_count=len(output.prompt_token_ids or ()),
-            output_token_count=len(completion.token_ids),
+            output_token_count=len(token_ids),
             cached_prompt_token_count=output.num_cached_tokens or 0,
             finish_reason=completion.finish_reason or "",
+            token_ids=token_ids,
+            logprobs=logprobs,
+            lora_name=lora_name,
         )
 
 class LoadedModel:
@@ -128,8 +141,9 @@ class LoadedModel:
             return
 
         # Regular load/move.
-        print(f"{self.model_config.model_name} - Moving model to target.")
-        self.engine_to_device(FREE_DEVICE) # Avoid dual load.
+        print(f"{self.model_config.model_name} - Moving model to {device}.")
+        if device == TARGET_DEVICE:
+            self._vacate_engine_for_model() # Avoid dual residency on the GPU.
         if device != FREE_DEVICE and self.current_model_device == FREE_DEVICE:
             # Reload from disk.
             self.model = self.model_loader.from_pretrained(
@@ -138,8 +152,14 @@ class LoadedModel:
             )
             self.model.eval()
         # Finalize.
+        previous_device = self.current_model_device
         self.model.to(device)
         self.current_model_device = device
+        if previous_device.startswith("cuda") and not device.startswith("cuda"):
+            # Hand the memory back to the driver: PyTorch's caching allocator keeps freed blocks
+            # reserved, and vLLM's sleep-mode allocator grabs its share straight from CUDA on wake.
+            gc.collect()
+            torch.cuda.empty_cache()
         elapsed = time.time() - device_change_start_time
         print(f"{self.model_config.model_name} - Moved model in {elapsed:.2f}s")
         harness_stats.model_loading_times.append(
@@ -198,21 +218,38 @@ class LoadedModel:
         
 
     def engine_to_device(self, device: str):
-        """Move engine model to the given device."""
-        assert device != SOURCE_DEVICE, "Our vllm only supports gpu/free, not cpu."
+        """
+        Place the serving engine: TARGET builds it (or wakes a sleeping one), SOURCE ("cpu") puts it to
+        sleep (vLLM sleep level 1: weights offloaded to host RAM, KV cache released, compiled graphs
+        kept; the GPU is free for training), FREE shuts it down. Waking takes seconds; building takes
+        the full load. A request on a sleeping engine wakes it.
+        """
         device_change_start_time = time.time()
         harness_stats = self.harness.harness_stats
+        if device == SOURCE_DEVICE and not self.vllm_model:
+            return                                                       # nothing to put to sleep (also covers CPU-only hosts, where SOURCE == TARGET)
         if device == TARGET_DEVICE:
             if self.vllm_model:
+                if self.current_engine_device == SOURCE_DEVICE:
+                    self._vacate_model_for_engine()                      # the wake re-grabs the engine's whole GPU share
+                    if torch.cuda.is_available():
+                        torch.cuda.empty_cache()
+                    print(f"{self.model_config.model_name} - Engine waking up.")
+                    self.vllm_model.wake_up()
+                    self.current_engine_device = device
+                    elapsed = time.time() - device_change_start_time
+                    print(f"{self.model_config.model_name} - Engine woke up in {elapsed:.2f}s")
+                    harness_stats.model_loading_times.append((self.model_config.model_name, elapsed))
                 return
             self.engine_free_other_models()
-            self.model_to_device(FREE_DEVICE) # Avoid dual load.
+            self._vacate_model_for_engine()
             print(f"{self.model_config.model_name} - Engine moving to target.")
             engine_kwargs = dict(
                 model=self.model_config.model_id,
-                max_loras=4,
+                max_loras=self.harness.harness_config.max_loras,
                 max_lora_rank=self.harness.harness_config.max_lora_rank,
                 enable_lora=True,
+                enable_sleep_mode=True,
                 dtype=self.model_config.dtype or "auto",
             )
             if self.model_config.engine_kwargs is None:
@@ -228,6 +265,16 @@ class LoadedModel:
                 (self.model_config.model_name, elapsed)
             )
             return
+        if device == SOURCE_DEVICE:
+            if not self.vllm_model or self.current_engine_device == SOURCE_DEVICE:
+                return
+            print(f"{self.model_config.model_name} - Engine going to sleep (weights to host RAM).")
+            self.vllm_model.sleep(level=1)
+            self.current_engine_device = device
+            elapsed = time.time() - device_change_start_time
+            print(f"{self.model_config.model_name} - Engine asleep in {elapsed:.2f}s")
+            harness_stats.model_freeing_times.append((self.model_config.model_name, elapsed))
+            return
         if device == FREE_DEVICE:
             if not self.vllm_model:
                 return
@@ -243,6 +290,19 @@ class LoadedModel:
                 (self.model_config.model_name, elapsed)
             )
 
+    def _vacate_model_for_engine(self):
+        """The GPU belongs to the engine: a plain model is freed, a model carrying adapters moves to host RAM."""
+        if self.current_model_device == FREE_DEVICE:
+            return
+        if self.harness.module_manager.has_loras(self.model_config.model_name):
+            self.model_to_device(SOURCE_DEVICE)
+        else:
+            self.model_to_device(FREE_DEVICE)
+
+    def _vacate_engine_for_model(self):
+        """The GPU belongs to the model: a running engine sleeps (its weights stay in host RAM for a fast wake)."""
+        if self.vllm_model and self.current_engine_device == TARGET_DEVICE:
+            self.engine_to_device(SOURCE_DEVICE)
 
     def engine_free_other_models(self):
         """Free all other models to make space for this one."""
@@ -305,6 +365,11 @@ class LoadedModel:
         )
         return [EngineChatOutput.from_request_output(output) for output in request_outputs]
 
+    def chat_template_kwargs(self, chat_kwargs: dict|None) -> dict:
+        """The chat-template kwargs a request would use (harness defaults < recommended < explicit)."""
+        _, extra_kwargs = self._merged_chat_kwargs(chat_kwargs)
+        return dict(extra_kwargs.get("chat_template_kwargs") or {})
+
     def engine_submit(
         self,
         messages: list[dict],
@@ -313,20 +378,42 @@ class LoadedModel:
         agent_id: str = "",
         lora_name: str|None = None,
         chat_kwargs: dict|None = None,
+        record_sampling: bool = False,
+    ) -> EngineChatOutput:
+        """
+        One conversation for one agent: templated here, then submitted as tokens (engine_submit_tokens).
+        """
+        self.engine_to_device(TARGET_DEVICE)
+        _, extra_kwargs = self._merged_chat_kwargs(chat_kwargs, seed)
+        token_ids = self.vllm_model.template(messages, tools, True, extra_kwargs.get("chat_template_kwargs"))
+        return self.engine_submit_tokens(token_ids, seed=seed, agent_id=agent_id, lora_name=lora_name, chat_kwargs=chat_kwargs,
+                                         record_sampling=record_sampling)
+
+    def engine_submit_tokens(
+        self,
+        prompt_token_ids: list[int],
+        seed: int|None = None,
+        agent_id: str = "",
+        lora_name: str|None = None,
+        chat_kwargs: dict|None = None,
+        record_sampling: bool = False,
     ) -> EngineChatOutput:
         """
-        One conversation for one agent. Same kwargs merge as engine_chat_many, a per-request seed,
-        and the agent pinned to one replica (by agent_id) so its prefix cache serves the next turn.
-        Blocks the calling thread while the engine batches this request with everything in flight.
+        One request whose prompt the caller owns as token ids (the agent loop keeps its own prefix).
+        Same kwargs merge as engine_chat_many, a per-request seed, the agent pinned to one replica (by
+        agent_id) so its prefix cache serves the next turn, the adapter the module manager currently
+        exposes for `lora_name` (None until an exchange), and with `record_sampling` the sampled token
+        log-probs. Blocks the calling thread while the engine batches this request with everything in flight.
         """
         self.engine_to_device(TARGET_DEVICE)
-        # @AI: This now goes away. lora_name becomes supported.
-        assert lora_name is None, "LoRA on the serving engine is not wired yet (no adapter to test)."
-        sampling_params, extra_kwargs = self._merged_chat_kwargs(chat_kwargs, seed)
+        sampling_params, _ = self._merged_chat_kwargs(chat_kwargs, seed)
+        if record_sampling and sampling_params.logprobs is None:
+            sampling_params.logprobs = 0
+        lora_request = self.harness.module_manager.engine_lora_request(lora_name) if lora_name else None
         engine = self.vllm_model
-        token_ids = engine.template(messages, tools, True, extra_kwargs.get("chat_template_kwargs"))
-        future = engine.submit(token_ids, sampling_params, replica=VLLMWrapper.replica_for(agent_id, engine.world_size))
-        return EngineChatOutput.from_request_output(future.result())
+        future = engine.submit(list(prompt_token_ids), sampling_params, replica=VLLMWrapper.replica_for(agent_id, engine.world_size),
+                               lora_request=lora_request)
+        return EngineChatOutput.from_request_output(future.result(), lora_name=lora_name if lora_request is not None else None)
 
 
     def simple_chat(self, user_msg: str) -> str:
@@ -368,7 +455,7 @@ class LoadedModel:
     def decoder_forward(
         self,
         inputs_embeds: torch.Tensor,
-        attention_mask: torch.Tensor,
+        attention_mask: torch.Tensor | None,
         position_ids: torch.Tensor|None = None,
         lora_name: str|None = None,
     ) -> torch.Tensor:
```

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">activation/harness/module_manager.py</span>
    <span class="card-oneliner">Checkpoints: save_lora, exchange_lora, branch_lora, engine_lora_request; register_lora(checkpoint_path) and ensure_lora reload; free_lora with a save.</span>
    <span class="card-badge">Diff</span>
  </summary>

Exact delta vs `/source` for `activation/harness/module_manager.py`:

```diff
--- a/activation/harness/module_manager.py
+++ b/activation/harness/module_manager.py
@@ -7,16 +7,26 @@ after the base is loaded, further adapters of the same base are added by name on
 (the base weights are present once; each adapter only adds its own A/B matrices inside the
 wrapped linear layers), and the base weights stay frozen throughout. A forward pass names the
 adapter it wants (`lora_context`); with no name the adapters are disabled for that pass. Freeing
-a base with adapters attached is refused until `free_lora` has dropped them (checkpointing is not
-implemented yet). Engine and training use of a model remain mutually exclusive.
+a base with adapters attached is refused until `free_lora` has dropped them.
+
+Adapters travel between training and serving as checkpoints under the synced folder
+(`LORAS/<lora_name>/round_<n>` plus a `latest` copy): `save_lora` writes one, `exchange_lora` makes the
+engine load it on its next request naming the adapter (a fresh `LoRARequest` id per exchange, no engine
+restart), `branch_lora` copies one adapter's latest checkpoint into a new name, and `register_lora`'s
+`checkpoint_path` starts an adapter from a saved one. A freshly registered adapter is the identity
+(PEFT zero-initialises the B matrices), so round 0 of a training run samples the base model exactly.
 """
 import re
+import shutil
 import typing as t
 from contextlib import contextmanager
 from dataclasses import dataclass
+from pathlib import Path
 
 from torch import nn
 
+from activation.common.data_syncing import resolve_path
+
 from .hf_utils import TARGET_DEVICE, make_peft_lora_config
 
 if t.TYPE_CHECKING:
@@ -42,6 +52,14 @@ class LoraConfig:
     adapter_name: str = ""
     """The name PEFT knows the adapter by: lora_name with every character outside [0-9A-Za-z_] replaced, since PEFT
     uses it as a module name (no dots)."""
+    checkpoint_path: str | None = None
+    """Where the trainable adapter is loaded from when injected (None: fresh, zero-init B). Updated by save_lora."""
+    engine_checkpoint_path: str | None = None
+    """What the serving engine applies for this name (None: nothing yet, requests run the base). Set by exchange_lora."""
+    engine_version: int = 0
+    """Bumped by every exchange; the engine caches adapters by integer id, so a new version is a new LoRARequest."""
+    saved_rounds: int = 0
+    """Checkpoints written so far (round_000, round_001, ...)."""
 
     def __post_init__(self):
         self.adapter_name = self.adapter_name or re.sub(r"[^0-9A-Za-z_]", "_", self.lora_name)
@@ -56,6 +74,9 @@ class ModuleManager:
         self.retrieval_acs: dict[str, "StandardRetrievalACModel"] = dict()
         self.peft_models: dict[str, "PeftModel"] = dict()
         """One PEFT wrapper per base model name, present while at least one adapter is injected."""
+        self._engine_version_counter = 0
+        self._engine_int_ids: dict[str, int] = dict()
+        """Integer ids of the adapters' current engine views, unique across adapters and exchanges (vLLM caches by id)."""
 
     # ---------------------------------------------------------------- registration
 
@@ -67,13 +88,14 @@ class ModuleManager:
         alpha: int | None = None,
         dropout: float = 0.05,
         target_modules: list[str] | None = None,
-        checkpoint_path: str|None = None, # Support reloads. Use the data sync stuff to make it easy to share across runs.
+        checkpoint_path: str | None = None,
     ) -> LoraConfig:
         """
-        Record the adapter config; the adapter itself is injected on first use.
-        alpha None means 2 x rank; target_modules None means the seven block projections.
+        Record the adapter config; the adapter itself is injected on first use, fresh (zero-init B: the
+        identity) or from `checkpoint_path` (a PEFT adapter folder, absolute or relative to the synced
+        folder, e.g. "LORAS/<name>/latest"). alpha None means 2 x rank; target_modules None means the
+        seven block projections. A checkpoint is also what the engine serves after the first exchange.
         """
-        # @AI: make sure the lora are sufficiently close to zero-init.
         assert model_name in self.harness.loaded_models, f"Unknown model {model_name!r}."
         assert lora_name not in self.lora_configs, f"LoRA {lora_name!r} is already registered."
         max_rank = self.harness.harness_config.max_lora_rank
@@ -85,7 +107,10 @@ class ModuleManager:
             alpha=alpha if alpha is not None else 2 * rank,
             dropout=dropout,
             target_modules=list(target_modules or DEFAULT_LORA_TARGET_MODULES),
+            checkpoint_path=None if checkpoint_path is None else str(_checkpoint_folder(checkpoint_path)),
         )
+        if lora_config.checkpoint_path is not None:
+            assert Path(lora_config.checkpoint_path, "adapter_config.json").exists(), f"No adapter at {lora_config.checkpoint_path}."
         assert all(other.adapter_name != lora_config.adapter_name for other in self.lora_configs.values()), (
             f"LoRA {lora_name!r} maps to adapter name {lora_config.adapter_name!r}, which another LoRA already uses."
         )
@@ -121,19 +146,27 @@ class ModuleManager:
         name if missing, and return the wrapper. Base weights stay frozen. Which adapter a forward
         pass uses is decided per pass by `lora_context`.
         """
-        from peft import get_peft_model
+        from peft import PeftModel, get_peft_model
         lora_config = self.get_lora_config(lora_name)
         loaded_model = self.harness.loaded_models[lora_config.model_name]
         loaded_model.model_to_device(TARGET_DEVICE)
         peft_model = self.peft_models.get(lora_config.model_name)
         if peft_model is None:
             loaded_model.model.requires_grad_(False)
-            peft_model = get_peft_model(
-                loaded_model.model, make_peft_lora_config(lora_config), adapter_name=lora_config.adapter_name,
-            )
+            if lora_config.checkpoint_path is not None:
+                peft_model = PeftModel.from_pretrained(
+                    loaded_model.model, lora_config.checkpoint_path, adapter_name=lora_config.adapter_name, is_trainable=True,
+                )
+            else:
+                peft_model = get_peft_model(
+                    loaded_model.model, make_peft_lora_config(lora_config), adapter_name=lora_config.adapter_name,
+                )
             self.peft_models[lora_config.model_name] = peft_model
         elif lora_config.adapter_name not in peft_model.peft_config:
-            peft_model.add_adapter(lora_config.adapter_name, make_peft_lora_config(lora_config))
+            if lora_config.checkpoint_path is not None:
+                peft_model.load_adapter(lora_config.checkpoint_path, adapter_name=lora_config.adapter_name, is_trainable=True)
+            else:
+                peft_model.add_adapter(lora_config.adapter_name, make_peft_lora_config(lora_config))
         return peft_model
 
     @contextmanager
@@ -168,17 +201,18 @@ class ModuleManager:
 
     def free_lora(self, model_name: str, lora_name: str, checkpoint_path: str | None = None) -> None:
         """
-        Drop the adapter's weights from the base. When it was the last adapter of that base, the
-        PEFT wrapper is removed and the plain base modules are restored, so the base can be freed.
-        Checkpointing before the drop is not implemented yet (checkpoint_path must be None); the
-        registered config stays, so the adapter can be re-injected fresh.
+        Drop the adapter's weights from the base, saving them first when `checkpoint_path` is given
+        (see save_lora). When it was the last adapter of that base, the PEFT wrapper is removed and the
+        plain base modules are restored, so the base can be freed. The registered config stays, with
+        `checkpoint_path` pointing at the last save, so the adapter re-injects from there.
         """
-        assert checkpoint_path is None, "LoRA checkpointing is not implemented yet."
         lora_config = self.get_lora_config(lora_name)
         assert lora_config.model_name == model_name, f"LoRA {lora_name!r} belongs to {lora_config.model_name!r}, not {model_name!r}."
         peft_model = self.peft_models.get(model_name)
         if peft_model is None or lora_config.adapter_name not in peft_model.peft_config:
             return                                                                      # never injected
+        if checkpoint_path is not None:
+            self.save_lora(model_name, lora_name, checkpoint_path)
         if len(peft_model.peft_config) > 1:
             peft_model.delete_adapter(lora_config.adapter_name)
             return
@@ -186,10 +220,99 @@ class ModuleManager:
         loaded_model.model = peft_model.base_model.unload()                                # LoRA layers replaced back in place
         del self.peft_models[model_name]
 
+    # ---------------------------------------------------------------- checkpoints and the engine view
+
+    def checkpoint_folder(self, lora_name: str, round_index: int | None = None) -> Path:
+        """`LORAS/<lora_name>/round_<n>` under the synced folder (`latest` when round_index is None)."""
+        leaf = "latest" if round_index is None else f"round_{round_index:03d}"
+        return resolve_path(f"LORAS/{lora_name}/{leaf}", create=False)
+
+    def save_lora(self, model_name: str, lora_name: str, checkpoint_path: str | None = None) -> str:
+        """
+        Write the adapter (only this one) as a PEFT adapter folder that both `ensure_lora` and the
+        serving engine can load: `checkpoint_path`, or the next `LORAS/<lora_name>/round_<n>`; a
+        `latest` copy is refreshed next to the numbered folders. The registered config's
+        `checkpoint_path` now points at the save, so a re-injection continues from it.
+        """
+        lora_config = self.get_lora_config(lora_name)
+        assert lora_config.model_name == model_name, f"LoRA {lora_name!r} belongs to {lora_config.model_name!r}, not {model_name!r}."
+        peft_model = self.peft_models.get(model_name)
+        assert peft_model is not None and lora_config.adapter_name in peft_model.peft_config, f"LoRA {lora_name!r} is not injected."
+        folder = Path(checkpoint_path) if checkpoint_path is not None else self.checkpoint_folder(lora_name, lora_config.saved_rounds)
+        if checkpoint_path is not None and not folder.is_absolute():
+            folder = resolve_path(checkpoint_path, create=False)
+        if folder.exists():
+            shutil.rmtree(folder)
+        folder.mkdir(parents=True, exist_ok=True)
+        peft_model.save_pretrained(str(folder), selected_adapters=[lora_config.adapter_name])
+        nested = folder / lora_config.adapter_name                                          # PEFT nests non-default adapters
+        if nested.is_dir():
+            for item in nested.iterdir():
+                shutil.move(str(item), str(folder / item.name))
+            nested.rmdir()
+        assert (folder / "adapter_config.json").exists(), f"PEFT did not write an adapter at {folder}"
+        if checkpoint_path is None:
+            lora_config.saved_rounds += 1
+            latest = self.checkpoint_folder(lora_name)
+            if latest.exists():
+                shutil.rmtree(latest)
+            shutil.copytree(folder, latest)
+        lora_config.checkpoint_path = str(folder)
+        return str(folder)
+
+    def exchange_lora(self, model_name: str, lora_name: str, checkpoint_path: str | None = None) -> None:
+        """
+        Make the serving engine see the adapter: from `checkpoint_path`, or from its last save. The
+        next request naming `lora_name` carries a new LoRARequest (new integer id), which vLLM loads
+        from the folder without a restart; requests before the first exchange run the base model.
+        """
+        lora_config = self.get_lora_config(lora_name)
+        assert lora_config.model_name == model_name, f"LoRA {lora_name!r} belongs to {lora_config.model_name!r}, not {model_name!r}."
+        path = checkpoint_path if checkpoint_path is not None else lora_config.checkpoint_path
+        assert path is not None, f"LoRA {lora_name!r} has no checkpoint to exchange; call save_lora (or train) first."
+        folder = _checkpoint_folder(path)
+        assert (folder / "adapter_config.json").exists(), f"No adapter at {folder}."
+        lora_config.engine_checkpoint_path = str(folder)
+        lora_config.engine_version += 1
+        self._engine_version_counter += 1
+        self._engine_int_ids[lora_name] = self._engine_version_counter
 
-    def exchane_lora(self, model_name: str, lora_name: str, checkpoint_path: str|None = None) -> None:
-        pass # guarantees that a next call to the engine sees the checkpointed lora after training.
+    def branch_lora(self, model_name: str, src_lora_name: str, dest_lora_name: str) -> LoraConfig:
+        """
+        A new adapter that starts from another's latest save: the checkpoint is copied to
+        `LORAS/<dest>/round_000` (and `latest`), the config registered with the same shape, and the
+        engine view exchanged when the source had one.
+        """
+        src = self.get_lora_config(src_lora_name)
+        assert src.model_name == model_name, f"LoRA {src_lora_name!r} belongs to {src.model_name!r}, not {model_name!r}."
+        assert src.checkpoint_path is not None, f"LoRA {src_lora_name!r} has no checkpoint to branch from; save it first."
+        dest_round = self.checkpoint_folder(dest_lora_name, 0)
+        for target in (dest_round, self.checkpoint_folder(dest_lora_name)):
+            if target.exists():
+                shutil.rmtree(target)
+            shutil.copytree(src.checkpoint_path, target)
+        dest = self.register_lora(
+            dest_lora_name, model_name, rank=src.rank, alpha=src.alpha, dropout=src.dropout,
+            target_modules=list(src.target_modules), checkpoint_path=str(dest_round),
+        )
+        dest.saved_rounds = 1
+        if src.engine_checkpoint_path is not None:
+            self.exchange_lora(model_name, dest_lora_name, str(dest_round))
+        return dest
+
+    def engine_lora_request(self, lora_name: str):
+        """The vLLM LoRARequest for the adapter's current engine view, or None before the first exchange."""
+        lora_config = self.get_lora_config(lora_name)
+        if lora_config.engine_checkpoint_path is None:
+            return None
+        from vllm.lora.request import LoRARequest
+        return LoRARequest(
+            lora_name=f"{lora_config.adapter_name}_v{lora_config.engine_version}",
+            lora_int_id=self._engine_int_ids[lora_name],
+            lora_path=lora_config.engine_checkpoint_path,
+        )
 
 
-    def branch_lora(self, model_name: str, src_lora_name: str, dest_lora_name: str):
-        pass # branches a lora.
\ No newline at end of file
+def _checkpoint_folder(path: str) -> Path:
+    folder = Path(path)
+    return folder if folder.is_absolute() else resolve_path(path, create=False)
```

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">activation/agent_training/agent_training_config.py</span>
    <span class="card-oneliner">AgentTrainingConfig defaults and AgentTrainingStats.</span>
    <span class="card-badge">Diff</span>
  </summary>

Exact delta vs `/source` for `activation/agent_training/agent_training_config.py`:

```diff
--- a/activation/agent_training/agent_training_config.py
+++ b/activation/agent_training/agent_training_config.py
@@ -1,6 +1,77 @@
+"""
+Configuration and statistics of one agent-training round.
+
+`AgentTrainingConfig` carries standard, LoRA-appropriate defaults for on-policy clipped training:
+PPO clip 0.2 on the token ratio, a handful of gradient steps per round over one pass of the data
+(samples go stale after that), AdamW at 2e-5 with (0.9, 0.95) and no weight decay on the adapter,
+gradient-norm clip 1.0, bf16 model with the ratio in fp32, activation checkpointing, token-budgeted
+micro-batches with gradient accumulation, and dropout off (an on-policy ratio needs a deterministic
+policy). `AgentTrainingStats` records what a round did, per step, for the report and the tests.
+"""
+from dataclasses import dataclass, field
+
+
+@dataclass
 class AgentTrainingConfig:
-    pass # All the best practices for lrs, clipping, batching, etc.
+    # The objective.
+    clip_epsilon: float = 0.2                  # PPO clip on the token ratio pi_theta / pi_old
+    updates_per_round: int = 4                 # gradient steps per train() call: the data is split into this many mini-batches, one pass
+    max_example_tokens: int = 16_384           # longer trajectories are dropped (counted), never truncated
+    # Optimisation (LoRA values; a full fine-tune would sit two orders of magnitude lower).
+    learning_rate: float = 2e-5
+    weight_decay: float = 0.0
+    adam_betas: tuple[float, float] = (0.9, 0.95)
+    adam_eps: float = 1e-8
+    max_grad_norm: float = 1.0
+    # Memory.
+    micro_batch_tokens: int = 32_768          # token budget per micro-batch when examples are packed with padding (pad_free_micro_batches=False)
+    pad_free_micro_batches: bool = True       # one example per micro-batch, no padding mask: sdpa takes its causal (flash) path, nothing is wasted on pads
+    length_multiple: int = 1                  # sequence length rounded up to a multiple (tail pads carry no loss). 1: rounding to 512 cost 30% on the node (profile), and no per-shape compile cost was measurable           # padded tokens per forward/backward; gradient accumulation fills the mini-batch
+    logits_chunk_tokens: int = 2_048           # loss positions per lm_head chunk (the full logits of a long sequence would not fit)
+    gradient_checkpointing: bool = True
+    # Bookkeeping.
+    seed: int = 0
+    checkpoint_every_round: bool = True        # adapter saved under LORAS/<lora_name>/round_<n> (+ latest) after the round
 
 
+@dataclass
 class AgentTrainingStats:
-    pass
\ No newline at end of file
+    round_index: int = 0
+    items: int = 0                             # AgentTrainingItems received
+    examples: int = 0                          # sequences trained on
+    dropped_too_long: int = 0                  # over max_example_tokens
+    dropped_empty: int = 0                     # no token segments or no sampled tokens (a run recorded without sampling)
+    recomputed_old_logprobs: int = 0           # ignore_logprobs items whose pi_old came from a no-grad pass
+    total_tokens: int = 0                      # tokens read (prompts included)
+    assistant_tokens: int = 0                  # tokens that carried loss
+    steps: int = 0
+    loss: list[float] = field(default_factory=list)            # per step
+    mean_ratio: list[float] = field(default_factory=list)      # per step, pi_theta / pi_old over loss tokens (1.0 at step 1 when everything aligns)
+    clip_fraction: list[float] = field(default_factory=list)   # per step, share of loss tokens outside 1 +- epsilon
+    mean_logprob: list[float] = field(default_factory=list)    # per step, mean log pi_theta of the sampled tokens (an entropy proxy)
+    grad_norm: list[float] = field(default_factory=list)       # per step, before clipping
+    step_seconds: list[float] = field(default_factory=list)
+    reporting_loss: float | None = None        # the objective on reporting_data after the round, no gradient
+    positive_weight_sum: float = 0.0
+    negative_weight_sum: float = 0.0
+    duration_s: float = 0.0
+    checkpoint_path: str | None = None
+    peak_memory_bytes: int = 0
+
+    def summarize(self) -> dict:
+        """One flat dict for logs and tests."""
+        return {
+            "round_index": self.round_index, "items": self.items, "examples": self.examples,
+            "dropped_too_long": self.dropped_too_long, "dropped_empty": self.dropped_empty,
+            "recomputed_old_logprobs": self.recomputed_old_logprobs,
+            "total_tokens": self.total_tokens, "assistant_tokens": self.assistant_tokens, "steps": self.steps,
+            "first_loss": self.loss[0] if self.loss else None, "last_loss": self.loss[-1] if self.loss else None,
+            "first_mean_ratio": self.mean_ratio[0] if self.mean_ratio else None,
+            "first_clip_fraction": self.clip_fraction[0] if self.clip_fraction else None,
+            "mean_logprob_first": self.mean_logprob[0] if self.mean_logprob else None,
+            "mean_logprob_last": self.mean_logprob[-1] if self.mean_logprob else None,
+            "reporting_loss": self.reporting_loss,
+            "positive_weight_sum": self.positive_weight_sum, "negative_weight_sum": self.negative_weight_sum,
+            "duration_s": round(self.duration_s, 1), "tokens_per_s": round(self.total_tokens / self.duration_s, 1) if self.duration_s else 0.0,
+            "checkpoint_path": self.checkpoint_path, "peak_memory_gb": round(self.peak_memory_bytes / 2**30, 2),
+        }
```

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">activation/agent_training/agent_training_utils.py</span>
    <span class="card-oneliner">Examples from segments, token-budgeted micro-batches, padding, chunked fp32 log-probs, the clipped surrogate, dropout off.</span>
    <span class="card-badge">Diff</span>
  </summary>

New file vs `/source` for `activation/agent_training/agent_training_utils.py`:

```diff
new file mode 100644
--- /dev/null
+++ b/activation/agent_training/agent_training_utils.py
@@ -0,0 +1,183 @@
+"""
+The messy parts of agent training, kept out of the trainer: turning a trajectory into one training
+sequence with its loss mask and stored log-probs, packing sequences into token-budgeted micro-batches,
+padding, and the chunked log-prob computation that avoids materialising the full logits of a long
+sequence (16k positions x a 250k vocabulary would not fit).
+"""
+from __future__ import annotations
+
+import typing as t
+from dataclasses import dataclass
+
+import torch
+
+if t.TYPE_CHECKING:
+    from .agent_trainer import AgentTrainingItem
+
+
+@dataclass
+class TrainingExample:
+    """One trajectory as the model read it: the first prompt, then every step's token segment."""
+    token_ids: list[int]
+    loss_mask: list[bool]          # True at positions whose token carries loss (predicted from the previous position)
+    old_logprobs: list[float]      # aligned with token_ids; the engine's log-prob where loss_mask is True, 0.0 elsewhere
+    weight: float                  # the advantage A of the whole trajectory
+    item_index: int
+    needs_old_logprobs: bool       # teacher / hinted items (or runs recorded without log-probs): pi_old comes from a no-grad pass
+
+    @property
+    def num_loss_tokens(self) -> int:
+        return sum(self.loss_mask)
+
+
+def build_example(item: "AgentTrainingItem", item_index: int) -> TrainingExample | None:
+    """
+    The run's prompt_token_ids followed by each trajectory step's token_ids; assistant segments carry
+    loss on their sampled tokens (the ones with a recorded log-prob; all of them for ignore_logprobs
+    items). None when the run has no token segments or no sampled tokens.
+    """
+    run = item.run_results
+    token_ids = list(run.prompt_token_ids)
+    if not token_ids:
+        return None
+    loss_mask = [False] * len(token_ids)
+    old_logprobs = [0.0] * len(token_ids)
+    any_recorded = False
+    for step in run.trajectory:
+        segment = list(step.get("token_ids") or [])
+        if step.get("role") == "assistant":
+            logprobs = [float(value) for value in (step.get("logprobs") or [])]
+            recorded = min(len(logprobs), len(segment))
+            any_recorded = any_recorded or recorded > 0
+            sampled = len(segment) if item.ignore_logprobs or recorded == 0 else recorded
+            loss_mask += [True] * sampled + [False] * (len(segment) - sampled)
+            if item.ignore_logprobs or recorded == 0:
+                old_logprobs += [0.0] * len(segment)
+            else:
+                old_logprobs += logprobs[:sampled] + [0.0] * (len(segment) - sampled)
+        else:
+            loss_mask += [False] * len(segment)
+            old_logprobs += [0.0] * len(segment)
+        token_ids += segment
+    loss_mask[0] = False                                                   # the first token is never predicted
+    if not any(loss_mask):
+        return None
+    return TrainingExample(
+        token_ids=token_ids, loss_mask=loss_mask, old_logprobs=old_logprobs, weight=float(item.weight), item_index=item_index,
+        needs_old_logprobs=item.ignore_logprobs or not any_recorded,
+    )
+
+
+def micro_batches(examples: list[TrainingExample], token_budget: int, pad_free: bool = True) -> list[list[int]]:
+    """
+    Indices grouped into micro-batches, longest first. Pad-free: one example each (no padding, so the
+    attention layers run their causal fast path and no token is wasted). Otherwise examples are packed
+    under `token_budget` padded tokens per micro-batch (longest-first bins keep the padding small).
+    """
+    order = sorted(range(len(examples)), key=lambda index: -len(examples[index].token_ids))
+    if pad_free:
+        return [[index] for index in order]
+    groups: list[list[int]] = []
+    for index in order:
+        length = len(examples[index].token_ids)
+        for group in groups:
+            width = max(len(examples[i].token_ids) for i in group)
+            if max(width, length) * (len(group) + 1) <= token_budget:
+                group.append(index)
+                break
+        else:
+            groups.append([index])
+    return groups
+
+
+@dataclass
+class Collated:
+    input_ids: torch.Tensor        # [B, S]
+    attention_mask: torch.Tensor   # [B, S]
+    position_ids: torch.Tensor     # [B, S]
+    loss_mask: torch.Tensor        # [B, S] bool
+    old_logprobs: torch.Tensor     # [B, S] float32
+    weights: torch.Tensor          # [B] float32
+    example_indices: list[int]     # into the examples list, batch order
+    model_attention_mask: torch.Tensor | None = None   # what the decoder gets: None = causal only (right padding never reaches a real token)
+
+
+def collate(examples: list[TrainingExample], indices: list[int], pad_token_id: int, device, length_multiple: int = 1,
+            causal_only: bool | None = None) -> Collated:
+    """
+    Right-padded tensors on `device`. `attention_mask` marks the real tokens (statistics); the decoder
+    gets `model_attention_mask`: None when nothing but tail padding is present (`causal_only`, the
+    default for a single example), since under causal attention a real token never sees a later pad and
+    the pads carry no loss. `length_multiple` rounds the length up so the compiled kernels see few shapes.
+    """
+    length = max(len(examples[index].token_ids) for index in indices)
+    length = -(-length // max(1, length_multiple)) * max(1, length_multiple)
+    if causal_only is None:
+        causal_only = len(indices) == 1
+    rows, attention, loss, old, weights = [], [], [], [], []
+    for index in indices:
+        example = examples[index]
+        pad = length - len(example.token_ids)
+        rows.append(example.token_ids + [pad_token_id] * pad)
+        attention.append([1] * len(example.token_ids) + [0] * pad)
+        loss.append(example.loss_mask + [False] * pad)
+        old.append(example.old_logprobs + [0.0] * pad)
+        weights.append(example.weight)
+    attention_mask = torch.tensor(attention, dtype=torch.long)
+    return Collated(
+        input_ids=torch.tensor(rows, dtype=torch.long, device=device),
+        attention_mask=attention_mask.to(device),
+        position_ids=(attention_mask.cumsum(dim=1) - 1).clamp(min=0).to(device),
+        loss_mask=torch.tensor(loss, dtype=torch.bool, device=device),
+        old_logprobs=torch.tensor(old, dtype=torch.float32, device=device),
+        weights=torch.tensor(weights, dtype=torch.float32, device=device),
+        example_indices=list(indices),
+        model_attention_mask=None if causal_only else attention_mask.to(device),
+    )
+
+
+def sampled_logprobs(hidden: torch.Tensor, head: torch.nn.Module, batch: Collated, chunk_tokens: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
+    """
+    log pi_theta of every loss token: the decoder's hidden state at the previous position through the
+    output head, in chunks. The head runs in fp32 (its weight cast once per call; it is frozen, so no
+    gradient flows through the cast): bf16 logits of magnitude 20-30 are quantised in steps of 0.125-0.25,
+    which alone would put several percent of the tokens outside a 0.2 ratio clip at the first step.
+    Returns (logprobs [N], batch row of each [N], target ids [N]).
+    """
+    rows, positions = batch.loss_mask.nonzero(as_tuple=True)
+    states = hidden[rows, positions - 1]                                                   # [N, d]
+    targets = batch.input_ids[rows, positions]                                             # [N]
+    with torch.no_grad():
+        weight = head.weight.float()
+        bias = head.bias.float() if getattr(head, "bias", None) is not None else None
+    pieces = []
+    for start in range(0, states.shape[0], chunk_tokens):
+        logits = torch.nn.functional.linear(states[start:start + chunk_tokens].float(), weight, bias)   # [n, V] fp32
+        pieces.append(torch.log_softmax(logits, dim=-1).gather(-1, targets[start:start + chunk_tokens, None])[:, 0])
+    return torch.cat(pieces) if pieces else states.new_zeros(0, dtype=torch.float32), rows, targets
+
+
+def clipped_surrogate(logprobs: torch.Tensor, old_logprobs: torch.Tensor, weights: torch.Tensor, epsilon: float) -> tuple[torch.Tensor, dict[str, float]]:
+    """
+    Per token: min(s A, clip(s, 1-eps, 1+eps) A) with s = exp(logp - old), A the sample's weight.
+    Correct for negative weights (the PPO form). Returns the token terms and the ratio statistics.
+    """
+    ratio = torch.exp(logprobs - old_logprobs)
+    surrogate = torch.minimum(ratio * weights, ratio.clamp(1 - epsilon, 1 + epsilon) * weights)
+    with torch.no_grad():
+        stats = {
+            "mean_ratio": float(ratio.mean()) if ratio.numel() else 1.0,
+            "clip_fraction": float(((ratio - 1).abs() > epsilon).float().mean()) if ratio.numel() else 0.0,
+            "mean_logprob": float(logprobs.mean()) if logprobs.numel() else 0.0,
+        }
+    return surrogate, stats
+
+
+def disable_dropout(module: torch.nn.Module) -> int:
+    """Dropout layers to eval while the rest trains (checkpointing needs training mode). Returns how many."""
+    count = 0
+    for child in module.modules():
+        if isinstance(child, torch.nn.Dropout):
+            child.eval()
+            count += 1
+    return count
```

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">activation/agent_training/agent_training_selection.py</span>
    <span class="card-oneliner">group_mean_advantage (default), raft_positive, weighted_positive_negative, reinforce_reject; subagents unrolled.</span>
    <span class="card-badge">Diff</span>
  </summary>

New file vs `/source` for `activation/agent_training/agent_training_selection.py`:

```diff
new file mode 100644
--- /dev/null
+++ b/activation/agent_training/agent_training_selection.py
@@ -0,0 +1,71 @@
+"""
+Selection functions: one group of rollouts (the same task, several seeds) in, weighted training items
+out. The trainer only sees the weights, so the learning rule lives here:
+
+- `group_mean_advantage` (default): weight = reward - mean(group); uniform groups carry no
+  within-task information and yield nothing. GRPO's advantage without the std division (Dr. GRPO,
+  RLOO); pairs with group size 8 and oversampling to fill a batch with mixed groups (DAPO's dynamic sampling).
+- `raft_positive`: correct runs at +1, the rest dropped (RAFT / RAFT++ with the clipped ratio).
+- `weighted_positive_negative`: correct +0.1, incorrect -1 (W-REINFORCE's lambda = 0.1).
+- `reinforce_reject`: +1 / -1 on mixed groups only.
+
+Every function unrolls subagent runs with the parent's weight (a subagent's trajectory is judged by the
+answer it helped produce) and marks non-policy sources (`run.source != "policy"`) as `ignore_logprobs`.
+"""
+from __future__ import annotations
+
+import typing as t
+
+from activation.agent import AgentRunResult
+
+from .agent_trainer import AgentTrainingItem, unroll
+
+SelectionFunction = t.Callable[[list[AgentRunResult]], list[AgentTrainingItem]]
+
+
+def items_for(run: AgentRunResult, weight: float, model_name: str, lora_name: str, include_subagents: bool = True) -> list[AgentTrainingItem]:
+    """One item per (sub)agent run under `run`, all at `weight`; zero weights produce nothing."""
+    if weight == 0.0:
+        return []
+    runs = unroll(run) if include_subagents else [run]
+    return [
+        AgentTrainingItem(run_results=r, model_name=model_name, lora_name=lora_name, weight=float(weight), ignore_logprobs=r.source != "policy")
+        for r in runs
+    ]
+
+
+def group_mean_advantage(group: list[AgentRunResult], *, model_name: str, lora_name: str, include_subagents: bool = True) -> list[AgentTrainingItem]:
+    if not group:
+        return []
+    mean = sum(run.score for run in group) / len(group)
+    return [item for run in group for item in items_for(run, run.score - mean, model_name, lora_name, include_subagents)]
+
+
+def raft_positive(group: list[AgentRunResult], *, model_name: str, lora_name: str, include_subagents: bool = True) -> list[AgentTrainingItem]:
+    return [item for run in group if run.score >= 1.0 for item in items_for(run, 1.0, model_name, lora_name, include_subagents)]
+
+
+def weighted_positive_negative(
+    group: list[AgentRunResult], *, model_name: str, lora_name: str, positive_weight: float = 0.1, negative_weight: float = -1.0,
+    include_subagents: bool = True,
+) -> list[AgentTrainingItem]:
+    return [
+        item for run in group
+        for item in items_for(run, positive_weight if run.score >= 1.0 else negative_weight, model_name, lora_name, include_subagents)
+    ]
+
+
+def reinforce_reject(group: list[AgentRunResult], *, model_name: str, lora_name: str, include_subagents: bool = True) -> list[AgentTrainingItem]:
+    scores = {run.score >= 1.0 for run in group}
+    if len(scores) < 2:
+        return []
+    return weighted_positive_negative(group, model_name=model_name, lora_name=lora_name, positive_weight=1.0, negative_weight=-1.0,
+                                      include_subagents=include_subagents)
+
+
+SELECTION_FUNCTIONS = {
+    "group_mean": group_mean_advantage,
+    "raft": raft_positive,
+    "weighted": weighted_positive_negative,
+    "reinforce_rej": reinforce_reject,
+}
```

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">activation/agent_training/agent_training_reporter.py</span>
    <span class="card-oneliner">Per-step curves and the per-round table.</span>
    <span class="card-badge">Diff</span>
  </summary>

Exact delta vs `/source` for `activation/agent_training/agent_training_reporter.py`:

```diff
--- a/activation/agent_training/agent_training_reporter.py
+++ b/activation/agent_training/agent_training_reporter.py
@@ -0,0 +1,68 @@
+"""
+The agent-training report: per-step curves (loss, ratio and clip fraction, mean log-prob, gradient
+norm, throughput) across rounds, a per-round table, and a status strip. `HtmlReporter`
+(common/reporting.py) owns the page; the trainer calls initialize_round / report_step / report_round.
+"""
+from __future__ import annotations
+
+import typing as t
+
+from activation.common.reporting import HtmlReporter, format_seconds
+
+if t.TYPE_CHECKING:
+    from .agent_training_config import AgentTrainingConfig, AgentTrainingStats
+
+ROUND_COLUMNS = ["round", "items", "examples", "dropped", "tokens", "loss tokens", "steps", "first loss", "last loss",
+                 "step-1 ratio", "step-1 clip", "reporting loss", "+weights", "-weights", "duration", "checkpoint"]
+
+
+class AgentTrainingReporter(HtmlReporter):
+    def __init__(self, report_folder: str, title: str, description: str = ""):
+        super().__init__(report_folder, title, description, eyebrow="Agent training", refresh_seconds=10, min_render_interval_seconds=2.0)
+        self.global_step = 0
+        self.initialize_line_plot("loss", "Loss per step", "The clipped surrogate (negated), mean over the mini-batch's samples.", "step", "loss", ["loss"])
+        self.initialize_line_plot(
+            "ratio", "Ratio to the sampling policy",
+            "Mean pi_theta / pi_old over loss tokens (1.0 at the first step of a round when trainer and engine agree) and the share of tokens outside 1 +- epsilon.",
+            "step", "value", ["mean ratio", "clip fraction"],
+        )
+        self.initialize_line_plot("logprob", "Mean log-prob of the sampled tokens", "An entropy proxy: rising means the policy sharpens on its own samples.", "step", "log-prob", ["mean logprob"])
+        self.initialize_line_plot("grad_norm", "Gradient norm", "Before clipping.", "step", "norm", ["grad norm"])
+        self.initialize_line_plot("throughput", "Tokens per second", "Tokens read per second of step time (prompts included).", "step", "tokens/s", ["tokens/s"])
+        self.initialize_table("rounds", "Rounds", "One row per train() call.", ROUND_COLUMNS)
+        self.render(force=True)
+
+    @property
+    def report_folder(self) -> str:
+        return str(self.folder)
+
+    def initialize_round(self, round_index: int, config: "AgentTrainingConfig", items: int, examples: int, dropped: int, tokens: int) -> None:
+        self.set_status(round=str(round_index), items=str(items), examples=str(examples), dropped=str(dropped), tokens=str(tokens),
+                        steps_this_round=str(config.updates_per_round), learning_rate=f"{config.learning_rate:g}", clip_epsilon=f"{config.clip_epsilon:g}")
+        self.render(force=True)
+
+    def report_step(self, round_index: int, step: int, loss: float, mean_ratio: float, clip_fraction: float, mean_logprob: float,
+                    grad_norm: float, tokens: int, seconds: float) -> None:
+        self.global_step += 1
+        x = self.global_step
+        self.add_data_point("loss", {"x": x, "loss": loss})
+        self.add_data_point("ratio", {"x": x, "mean ratio": mean_ratio, "clip fraction": clip_fraction})
+        self.add_data_point("logprob", {"x": x, "mean logprob": mean_logprob})
+        self.add_data_point("grad_norm", {"x": x, "grad norm": grad_norm})
+        self.add_data_point("throughput", {"x": x, "tokens/s": tokens / seconds if seconds > 0 else 0.0})
+        self.set_status(step=f"{step} (round {round_index})", last_loss=f"{loss:.4f}", last_ratio=f"{mean_ratio:.3f}")
+        self.render(force=True)
+
+    def report_round(self, stats: "AgentTrainingStats") -> None:
+        self.add_data_point("rounds", {
+            "round": stats.round_index, "items": stats.items, "examples": stats.examples,
+            "dropped": stats.dropped_too_long + stats.dropped_empty, "tokens": stats.total_tokens, "loss tokens": stats.assistant_tokens,
+            "steps": stats.steps, "first loss": f"{stats.loss[0]:.4f}" if stats.loss else "-", "last loss": f"{stats.loss[-1]:.4f}" if stats.loss else "-",
+            "step-1 ratio": f"{stats.mean_ratio[0]:.3f}" if stats.mean_ratio else "-",
+            "step-1 clip": f"{stats.clip_fraction[0]:.3f}" if stats.clip_fraction else "-",
+            "reporting loss": "-" if stats.reporting_loss is None else f"{stats.reporting_loss:.4f}",
+            "+weights": f"{stats.positive_weight_sum:.2f}", "-weights": f"{stats.negative_weight_sum:.2f}",
+            "duration": format_seconds(stats.duration_s), "checkpoint": stats.checkpoint_path or "-",
+        })
+        self.set_status(round_duration=format_seconds(stats.duration_s))
+        self.render(force=True)
```

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">activation/agent_training/agent_trainer.py</span>
    <span class="card-oneliner">AgentTrainingItem, unroll, AgentTrainer.select_agent_runs / train.</span>
    <span class="card-badge">Diff</span>
  </summary>

Exact delta vs `/source` for `activation/agent_training/agent_trainer.py`:

```diff
--- a/activation/agent_training/agent_trainer.py
+++ b/activation/agent_training/agent_trainer.py
@@ -1,45 +1,264 @@
 """
-The general interface is to call train with a lora_name.
-Then to perform an exchange to make the weights visible to the engine.
+Agent training: weighted trajectories in, a trained LoRA checkpoint out.
 
-For now, until training itself becomes the issue (and not the outer loop of harness/model design, dataset study, etc.),
-this code should be as standard and "just works" as possible.
-Should follow all the standards that prevent unnecessary, several percentages loss in performance.
-But should avoid micro-optimizing the last 0.1-1%.
+The interface is `select_agent_runs` (a selection function turns each group of rollouts into weighted
+items) then `train(model_name, lora_name, ...)` for one round of gradient steps on that adapter, then
+`module_manager.exchange_lora` so the serving engine sees the checkpoint; evaluation is the rollout
+manager. Standard and "just works": the objective is the clipped surrogate
 
-The sibling optimization file provide the most common sense optimizations to keep this efficient (again without overkill).
+    L = -mean_i (1/|a_i|) sum_t min(s_t A_i, clip(s_t, 1-eps, 1+eps) A_i),   s_t = pi_theta(a_t|prefix) / pi_old(a_t|prefix)
+
+over the sampled assistant tokens of every trajectory (prompts and tool outputs masked), with A_i the
+item's weight. pi_old is the engine's stored log-prob of each sampled token; for `ignore_logprobs`
+items (teacher or hinted trajectories) it is the current policy at the start of the round, so the ratio
+starts at 1 and the first step is weighted supervised fine-tuning. RAFT++, group-mean advantages,
+Reinforce-Rej and W-REINFORCE are all selection functions on top of this one loss.
+
+One trajectory is one training sequence (the agent records its token segments), gradient
+checkpointing is on, micro-batches are packed by token budget and accumulated into
+`updates_per_round` mini-batch steps, and the GPU is shared with the serving engine through sleep
+mode: the engine sleeps while the model trains; the model moves to host RAM afterwards and the next
+rollout wakes the engine. The messy helpers live in agent_training_utils.
 """
+from __future__ import annotations
+
+import random
+import time
+import typing as t
+import uuid
+from dataclasses import dataclass, field
+
+import torch
+
+from activation.agent import AgentRunResult
+from activation.harness import SOURCE_DEVICE, TARGET_DEVICE
 
+from .agent_training_config import AgentTrainingConfig, AgentTrainingStats
+from .agent_training_reporter import AgentTrainingReporter
+from .agent_training_utils import (
+    Collated,
+    TrainingExample,
+    build_example,
+    clipped_surrogate,
+    collate,
+    disable_dropout,
+    micro_batches,
+    sampled_logprobs,
+)
+
+if t.TYPE_CHECKING:
+    from activation.harness import HarnessRuntime
+
+
+@dataclass
 class AgentTrainingItem:
-    run_results: AgentRunResults # NOTE: Only uses the top-level. Select should do the unrolling (allow selecting different loras for different subagents).
+    run_results: AgentRunResult          # one agent's run (top level or a subagent's; selection functions unroll)
     model_name: str
-    lora_name: str
-    weight: float
-    ignore_logprobs: bool = False # When true, ignore logprobs (for teacher trajectories).
-    ... # any other needed field, but tell me precisely.
+    lora_name: str                       # the adapter this item trains
+    weight: float                        # the advantage A: sign = direction, magnitude = strength; never 0
+    ignore_logprobs: bool = False        # teacher / hinted data: pi_old := the current policy at round start
+    group_key: str = field(default_factory=lambda: uuid.uuid4().hex[:8])   # select_agent_runs stamps one key per group
 
-class AgentTrainer:
-    def __init__(self, harness: HarnessRuntime):
-        pass
 
+def unroll(result: AgentRunResult) -> list[AgentRunResult]:
+    """The run and every subagent run beneath it, depth first."""
+    out = [result]
+    for child in result.subagent_results:
+        out.extend(unroll(child))
+    return out
 
+
+class AgentTrainer:
+    def __init__(self, harness: "HarnessRuntime", config: AgentTrainingConfig | None = None):
+        self.harness = harness
+        self.config = config or AgentTrainingConfig()
+        self.rounds_done: dict[str, int] = {}
+
+    # ----------------------------------------------------------------------------- selection
     def select_agent_runs(
         self,
-        run_results: list[list[AgentRunResults]],
-        selection_function: t.Callable[[list[AgentRunResults]], list[AgentTrainingItem]], # filter out all negatives, etc.
-    ) -> dict[tuple[str, str], AgentTrainingItem]:
-        pass
+        run_results: list[list[AgentRunResult]],
+        selection_function: t.Callable[[list[AgentRunResult]], list[AgentTrainingItem]],
+    ) -> dict[tuple[str, str], list[AgentTrainingItem]]:
+        """Applies the function to every group, stamps its items with one group_key, buckets by (model_name, lora_name)."""
+        selected: dict[tuple[str, str], list[AgentTrainingItem]] = {}
+        for group in run_results:
+            group_key = uuid.uuid4().hex[:8]
+            for item in selection_function(list(group)):
+                assert item.weight != 0.0, "selection functions must not emit zero-weight items"
+                item.group_key = group_key
+                selected.setdefault((item.model_name, item.lora_name), []).append(item)
+        return selected
 
+    # ----------------------------------------------------------------------------- training
     def train(
         self,
         model_name: str,
-        lora_name: str, # Now required.
+        lora_name: str,
         training_data: list[AgentTrainingItem],
-        reporting_data: list[AgentTrainingItem],
-        reporter: AgentTrainingReporter,
+        reporting_data: t.Sequence[AgentTrainingItem] = (),
+        reporter: AgentTrainingReporter | None = None,
     ) -> AgentTrainingStats:
-        pass
+        """
+        One round on `lora_name`: engine to sleep, base + adapter on the GPU, one clipped-surrogate pass
+        over the items in `updates_per_round` steps, the objective on `reporting_data` without gradient,
+        the adapter checkpointed, the model moved to host RAM. Does not exchange: the caller decides when
+        the engine sees the new adapter.
+        """
+        config = self.config
+        assert training_data, "No training data."
+        assert all(item.model_name == model_name and item.lora_name == lora_name for item in training_data), "items of another (model, lora)"
+        module_manager = self.harness.module_manager
+        loaded_model = self.harness.loaded_models[model_name]
+        round_index = self.rounds_done.get(lora_name, 0)
+        stats = AgentTrainingStats(round_index=round_index, items=len(training_data))
+        started = time.time()
+        torch.manual_seed(config.seed + round_index)
+
+        # Examples: one sequence per trajectory.
+        examples: list[TrainingExample] = []
+        for index, item in enumerate(training_data):
+            example = build_example(item, index)
+            if example is None:
+                stats.dropped_empty += 1
+            elif len(example.token_ids) > config.max_example_tokens:
+                stats.dropped_too_long += 1
+            else:
+                examples.append(example)
+        assert examples, f"every item was dropped ({stats.dropped_too_long} too long, {stats.dropped_empty} without sampled tokens)"
+        stats.examples = len(examples)
+        stats.total_tokens = sum(len(example.token_ids) for example in examples)
+        stats.assistant_tokens = sum(example.num_loss_tokens for example in examples)
+        stats.positive_weight_sum = sum(example.weight for example in examples if example.weight > 0)
+        stats.negative_weight_sum = sum(example.weight for example in examples if example.weight < 0)
+        reporting_examples = [example for example in (build_example(item, index) for index, item in enumerate(reporting_data))
+                              if example is not None and len(example.token_ids) <= config.max_example_tokens]
+        if reporter is not None:
+            reporter.initialize_round(round_index, config, len(training_data), len(examples), stats.dropped_too_long + stats.dropped_empty, stats.total_tokens)
+
+        # The GPU: engine asleep, base + adapter resident, checkpointing on, dropout off, base frozen.
+        if loaded_model.vllm_model is not None:
+            loaded_model.engine_to_device(SOURCE_DEVICE)                  # sleep: weights to host RAM, the GPU to the trainer
+        module_manager.ensure_lora(lora_name)
+        base = loaded_model.model
+        device = next(base.parameters()).device
+        base.train()
+        if config.gradient_checkpointing:
+            base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
+        disable_dropout(base)
+        parameters = module_manager.lora_parameters(lora_name)
+        assert parameters, f"LoRA {lora_name!r} has no trainable parameters"
+        optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate, betas=tuple(config.adam_betas), eps=config.adam_eps,
+                                      weight_decay=config.weight_decay)
+        pad_token_id = loaded_model.tokenizer.pad_token_id
+        if pad_token_id is None:
+            pad_token_id = loaded_model.tokenizer.eos_token_id or 0
+        if device.type == "cuda":
+            torch.cuda.reset_peak_memory_stats(device)
+        try:
+            # pi_old for items without usable engine log-probs: the current policy, before any step.
+            needing = [example for example in examples + reporting_examples if example.needs_old_logprobs]
+            if needing:
+                self._fill_old_logprobs(loaded_model, lora_name, needing, pad_token_id, device)
+                stats.recomputed_old_logprobs = sum(1 for example in examples if example.needs_old_logprobs)
+
+            # One pass, updates_per_round mini-batches, token-budgeted micro-batches with accumulation.
+            order = list(range(len(examples)))
+            random.Random(config.seed + round_index).shuffle(order)
+            steps = max(1, min(config.updates_per_round, len(order)))
+            per_step = -(-len(order) // steps)
+            for step in range(steps):
+                step_started = time.time()
+                mini = order[step * per_step:(step + 1) * per_step]
+                if not mini:
+                    break
+                optimizer.zero_grad(set_to_none=True)
+                totals = {"loss": 0.0, "mean_ratio": 0.0, "clip_fraction": 0.0, "mean_logprob": 0.0, "tokens": 0, "loss_tokens": 0}
+                for group in micro_batches([examples[index] for index in mini], config.micro_batch_tokens, config.pad_free_micro_batches):
+                    batch = collate(examples, [mini[i] for i in group], pad_token_id, device, config.length_multiple)
+                    loss, batch_stats, loss_tokens = self._objective(loaded_model, lora_name, batch, len(mini))
+                    loss.backward()
+                    totals["loss"] += float(loss.detach())
+                    for key in ("mean_ratio", "clip_fraction", "mean_logprob"):
+                        totals[key] += batch_stats[key] * loss_tokens
+                    totals["tokens"] += int(batch.attention_mask.sum())
+                    totals["loss_tokens"] += loss_tokens
+                grad_norm = float(torch.nn.utils.clip_grad_norm_(parameters, config.max_grad_norm))
+                optimizer.step()
+                divisor = max(1, totals["loss_tokens"])
+                seconds = time.time() - step_started
+                stats.steps += 1
+                stats.loss.append(totals["loss"])
+                stats.mean_ratio.append(totals["mean_ratio"] / divisor)
+                stats.clip_fraction.append(totals["clip_fraction"] / divisor)
+                stats.mean_logprob.append(totals["mean_logprob"] / divisor)
+                stats.grad_norm.append(grad_norm)
+                stats.step_seconds.append(seconds)
+                print(f"AgentTrainer - round {round_index} step {step + 1}/{steps}: loss {totals['loss']:.4f} ratio {stats.mean_ratio[-1]:.3f} "
+                      f"clip {stats.clip_fraction[-1]:.3f} logprob {stats.mean_logprob[-1]:.3f} grad {grad_norm:.3f} "
+                      f"{totals['tokens']} tokens in {seconds:.1f}s", flush=True)
+                if reporter is not None:
+                    reporter.report_step(round_index, step + 1, totals["loss"], stats.mean_ratio[-1], stats.clip_fraction[-1], stats.mean_logprob[-1],
+                                         grad_norm, totals["tokens"], seconds)
+
+            # The objective on the reporting items, no gradient, after the round.
+            if reporting_examples:
+                with torch.no_grad():
+                    total = 0.0
+                    for group in micro_batches(reporting_examples, config.micro_batch_tokens, config.pad_free_micro_batches):
+                        batch = collate(reporting_examples, group, pad_token_id, device, config.length_multiple)
+                        loss, _, _ = self._objective(loaded_model, lora_name, batch, len(reporting_examples))
+                        total += float(loss)
+                stats.reporting_loss = total
+
+            if config.checkpoint_every_round:
+                stats.checkpoint_path = module_manager.save_lora(model_name, lora_name)
+        finally:
+            del optimizer
+            base.eval()
+            if base.is_gradient_checkpointing:
+                base.gradient_checkpointing_disable()
+            if device.type == "cuda":
+                stats.peak_memory_bytes = int(torch.cuda.max_memory_allocated(device))
+            loaded_model.model_to_device(SOURCE_DEVICE)          # off the GPU; the next rollout wakes the engine
+            if torch.cuda.is_available():
+                torch.cuda.empty_cache()
+        self.rounds_done[lora_name] = round_index + 1
+        stats.duration_s = time.time() - started
+        if reporter is not None:
+            reporter.report_round(stats)
+            reporter.finish()
+        print(f"AgentTrainer - round {round_index} done: {stats.summarize()}", flush=True)
+        return stats
+
+    # ----------------------------------------------------------------------------- internals
+    def _hidden_and_logprobs(self, loaded_model, lora_name: str, batch: Collated) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
+        base = loaded_model.model
+        inputs_embeds = base.get_input_embeddings()(batch.input_ids)
+        hidden = loaded_model.decoder_forward(inputs_embeds, batch.model_attention_mask, batch.position_ids, lora_name)
+        return sampled_logprobs(hidden, base.get_output_embeddings(), batch, self.config.logits_chunk_tokens)
 
+    def _objective(self, loaded_model, lora_name: str, batch: Collated, num_examples: int) -> tuple[torch.Tensor, dict[str, float], int]:
+        """The negated clipped surrogate of this micro-batch, scaled so the mini-batch's total is a mean over its examples."""
+        logprobs, rows, _ = self._hidden_and_logprobs(loaded_model, lora_name, batch)
+        old = batch.old_logprobs[batch.loss_mask]
+        weights = batch.weights[rows]
+        surrogate, batch_stats = clipped_surrogate(logprobs, old, weights, self.config.clip_epsilon)
+        num_rows = batch.input_ids.shape[0]
+        per_sample = torch.zeros(num_rows, device=logprobs.device, dtype=torch.float32).index_add_(0, rows, surrogate)
+        counts = torch.zeros(num_rows, device=logprobs.device, dtype=torch.float32).index_add_(0, rows, torch.ones_like(surrogate))
+        loss = -(per_sample / counts.clamp(min=1)).sum() / max(1, num_examples)
+        return loss, batch_stats, int(logprobs.shape[0])
 
-    # I believe eval actually is just the rollout manager code.
-    # The code outside this calls exchange as needed.
\ No newline at end of file
+    @torch.no_grad()
+    def _fill_old_logprobs(self, loaded_model, lora_name: str, examples: list[TrainingExample], pad_token_id: int, device) -> None:
+        """pi_old := the current policy for examples without engine log-probs (teacher / hinted / unrecorded)."""
+        for group in micro_batches(examples, self.config.micro_batch_tokens, self.config.pad_free_micro_batches):
+            batch = collate(examples, group, pad_token_id, device, self.config.length_multiple)
+            logprobs, rows, _ = self._hidden_and_logprobs(loaded_model, lora_name, batch)
+            positions = batch.loss_mask.nonzero(as_tuple=True)[1]
+            for row_index, example_index in enumerate(batch.example_indices):
+                mask = rows == row_index
+                values = logprobs[mask].tolist()
+                for position, value in zip(positions[mask].tolist(), values):
+                    examples[example_index].old_logprobs[position] = value
```

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">activation/agent_training/__init__.py</span>
    <span class="card-oneliner">Package exports.</span>
    <span class="card-badge">Diff</span>
  </summary>

Exact delta vs `/source` for `activation/agent_training/__init__.py`:

```diff
--- a/activation/agent_training/__init__.py
+++ b/activation/agent_training/__init__.py
@@ -0,0 +1,11 @@
+from .agent_training_config import AgentTrainingConfig, AgentTrainingStats
+from .agent_training_reporter import AgentTrainingReporter
+from .agent_trainer import AgentTrainer, AgentTrainingItem, unroll
+from .agent_training_selection import (
+    SELECTION_FUNCTIONS,
+    SelectionFunction,
+    group_mean_advantage,
+    raft_positive,
+    reinforce_reject,
+    weighted_positive_negative,
+)
```

</details>




<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">activation/tests/test_basic_agent_training.py</span>
    <span class="card-oneliner">The two key tests, 16 x 8 on Qwen3.5-4B.</span>
    <span class="card-badge">Diff</span>
  </summary>

Exact delta vs `/source` for `activation/tests/test_basic_agent_training.py`:

```diff
--- a/activation/tests/test_basic_agent_training.py
+++ b/activation/tests/test_basic_agent_training.py
@@ -1,21 +1,139 @@
-def _raft_selection_function
+"""
+Slice 2 key tests: one training round on grouped rollouts, then two rounds with an exchange between
+them. Small on purpose (16 problems x group 8 on Qwen3.5-4B, ~128 rollouts per phase); the asserts are
+about mechanics (segments recorded, group-mean weights, step-1 ratio of 1, checkpoint, exchange seen by
+the engine), the accuracies are printed.
 
+  uv run pytest activation/tests/test_basic_agent_training.py --gpu --slow -s
+"""
+import json
+import os
+import random
+from dataclasses import replace
+
+import pytest
+
+from activation.agent import AgentConfig, RolloutReporter
+from activation.agent_training import AgentTrainer, AgentTrainingConfig, AgentTrainingReporter, group_mean_advantage
+from activation.common.data_syncing import resolve_path
+from activation.dataset.loaders import DapoMathDataset
+from activation.harness import FREE_DEVICE, HarnessRuntime, HarnessRuntimeConfig, ModelConfig
+
+MODEL_NAME, MODEL_ID, LORA = "qwen3.5-4b", "Qwen/Qwen3.5-4B", "dapo_group_mean"
+PROBLEMS, GROUP = 16, 8
+
+BASE = AgentConfig(
+    system_prompt=("You are a careful problem solver working in a sandbox. Use the python or shell tools to compute "
+                   "rather than guessing. When you are done, call submit_answer exactly once with the final answer."),
+    model_name=MODEL_NAME, lora_name=LORA,
+    call_kwargs={"sampling_params": {"max_tokens": 2048, "temperature": 0.7}},
+    max_turns=8, max_tool_errors=3, max_duration=300,
+)
+
+
+def _harness() -> HarnessRuntime:
+    harness = HarnessRuntime(HarnessRuntimeConfig(model_configs={MODEL_NAME: ModelConfig(MODEL_NAME, MODEL_ID)}, agent_max_concurrent=64))
+    harness.module_manager.register_lora(LORA, MODEL_NAME, rank=64)          # zero-init B: round 0 samples are the base model's
+    return harness
+
+
+def _tasks(harness):
+    dataset = DapoMathDataset.load(harness, max_examples=1000)
+    harness.dataset_manager.register_dataset(dataset)
+    tasks = list(dataset.scorable_tasks.values())
+    random.Random(0).shuffle(tasks)
+    return tasks[:PROBLEMS]
+
+
+def _selection_function(group):
+    return group_mean_advantage(group, model_name=MODEL_NAME, lora_name=LORA)
+
+
+def _rollouts(harness, configs, round_index, base_seed, prefix="single"):
+    reporter = RolloutReporter(str(resolve_path(f"AGENT_TRAINING_TEST/{prefix}_rollouts_round{round_index}")), f"training test ({prefix}), rollouts round {round_index}")
+    return harness.rollout_manager.perform_grouped_rollouts(configs, group_count=GROUP, base_seed=base_seed, perform_scoring=True,
+                                                            caching_id=f"agent_training_test_{prefix}_r{round_index}", reporter=reporter)
+
+
+def _accuracy(groups):
+    scores = [r.score for g in groups for r in g]
+    return sum(scores) / len(scores)
+
+
+def _release(harness):
+    """Adapters off the base (their checkpoints are on disk), then the base and the engine off the GPU."""
+    loaded = harness.loaded_models[MODEL_NAME]
+    loaded.engine_to_device(FREE_DEVICE)
+    harness.module_manager.free_lora(MODEL_NAME, LORA)
+    loaded.model_to_device(FREE_DEVICE)
+
+
+@pytest.mark.gpu
+@pytest.mark.slow
 def test_basic_agent_training():
-    # @AI: Show me this interface in full.
-    # I am tempted to say: use the same data for eval so we can see improvements, but it's not a big deal.
-    harness = ...
-    grouped_run_results = ...
-    training_data = agent_trainer.select_training_data(
-        grouped_run_results,
-        selection_function=
-    )
-    training_data, reporting_data = training_data[:90], training_data[90:100] # Or whatever
-    agent_r
-    pass
+    """Roll out, weight by group-mean advantage, train one round, exchange, roll out the same problems again with the adapter."""
+    harness = _harness()
+    tasks = _tasks(harness)
+    configs = [replace(BASE, user_prompt=t.agent_prompt, dataset_task=t) for t in tasks]
+    trainer = AgentTrainer(harness, AgentTrainingConfig(updates_per_round=2))
+    try:
+        groups = _rollouts(harness, configs, 0, base_seed=0)
+        for r in (r for g in groups for r in g):
+            assert r.prompt_token_ids and all(step["token_ids"] for step in r.trajectory), "segments not recorded"
+            assert all(0 < len(step["logprobs"]) <= len(step["token_ids"]) for step in r.trajectory if step["role"] == "assistant")
+            assert r.lora_name is None, "round 0 runs the base model (nothing exchanged yet)"
+        selected = trainer.select_agent_runs(groups, selection_function=_selection_function)
+        items = selected[(MODEL_NAME, LORA)]
+        assert items, "no mixed group in 16 x 8 (would be very unlucky at ~75% accuracy)"
+        assert any(item.weight > 0 for item in items) and any(item.weight < 0 for item in items)
+        by_group = {}
+        for item in items:
+            by_group.setdefault(item.group_key, []).append(item.weight)
+        assert all(abs(sum(weights)) < 1e-6 for weights in by_group.values()), "group-mean weights sum to zero per group"
+        split = max(1, int(len(items) * 0.9))
+        training_data, reporting_data = items[:split], items[split:]
+        reporter = AgentTrainingReporter(str(resolve_path("AGENT_TRAINING_TEST/single_train_round0")), "training test (single), round 0")
+        stats = trainer.train(MODEL_NAME, LORA, training_data, reporting_data, reporter)     # sleeps the engine
+        print(json.dumps(stats.summarize(), indent=1))
+        assert stats.steps == 2 and stats.examples > 0
+        assert stats.examples + stats.dropped_too_long + stats.dropped_empty == len(training_data)
+        assert 0.9 < stats.mean_ratio[0] < 1.1, stats.mean_ratio          # step 1: trainer and engine agree on pi_old (decision 3)
+        assert stats.clip_fraction[0] < 0.15, stats.clip_fraction    # bf16 engine (fp8 KV cache) vs bf16 trainer: a few % of tokens land outside the clip
+        assert stats.checkpoint_path and os.path.isdir(stats.checkpoint_path)
+        harness.module_manager.exchange_lora(MODEL_NAME, LORA)
+        again = _rollouts(harness, configs, 1, base_seed=1000)             # fresh seeds: not the cache, the adapter; wakes the engine
+        assert all(r.lora_name == LORA for g in again for r in g)
+        print(f"accuracy before {_accuracy(groups):.3f} -> after one round {_accuracy(again):.3f} (16 x 8; printed, not asserted)")
+    finally:
+        _release(harness)
+    assert os.path.exists(os.path.join(reporter.report_folder, "report.tressoir.html"))
 
 
+@pytest.mark.gpu
+@pytest.mark.slow
 def test_basic_agent_training_multi_rounds():
-    # Like the above, but proceeds for two rounds.
-    # To show multi-round, exchange, etc, in action.
-    # Slice 2 is still just lora, so the improvements should be clear?
-    pass
\ No newline at end of file
+    """Two rounds: rollouts -> train -> exchange -> rollouts with the adapter -> train -> exchange -> rollouts."""
+    harness = _harness()
+    tasks = _tasks(harness)
+    configs = [replace(BASE, user_prompt=t.agent_prompt, dataset_task=t) for t in tasks]
+    trainer = AgentTrainer(harness, AgentTrainingConfig(updates_per_round=2))
+    accuracies, ratios = [], []
+    try:
+        for round_index in range(2):
+            groups = _rollouts(harness, configs, round_index, base_seed=10_000 * (round_index + 1), prefix="multi")
+            accuracies.append(_accuracy(groups))
+            if round_index > 0:
+                assert all(r.lora_name == LORA for g in groups for r in g)
+            items = trainer.select_agent_runs(groups, _selection_function)[(MODEL_NAME, LORA)]
+            reporter = AgentTrainingReporter(str(resolve_path(f"AGENT_TRAINING_TEST/multi_train_round{round_index}")), f"training test (multi), round {round_index}")
+            stats = trainer.train(MODEL_NAME, LORA, items, reporter=reporter)
+            ratios.append(stats.mean_ratio[0])
+            harness.module_manager.exchange_lora(MODEL_NAME, LORA)
+        final = _rollouts(harness, configs, 2, base_seed=30_000, prefix="multi")
+        accuracies.append(_accuracy(final))
+    finally:
+        _release(harness)
+    print(f"accuracy by round {[round(a, 3) for a in accuracies]}; step-1 ratios {[round(r, 3) for r in ratios]}")
+    assert all(0.9 < r < 1.1 for r in ratios), ratios                     # round 2's samples came from the exchanged adapter
+    checkpoints = sorted(os.listdir(resolve_path(f"LORAS/{LORA}", create=False)))
+    assert "round_000" in checkpoints and "round_001" in checkpoints and "latest" in checkpoints
```

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">pyproject.toml</span>
    <span class="card-oneliner">flash-linear-attention: the Triton gated-delta-net kernels transformers picks up for Qwen3.5 training (then `uv lock`, or copy the staged uv.lock).</span>
    <span class="card-badge">Diff</span>
  </summary>

Exact delta vs `/source` for `pyproject.toml`:

```diff
--- a/pyproject.toml
+++ b/pyproject.toml
@@ -12,7 +12,8 @@ dependencies = [
     "datasets",
     "faiss-cpu",
     "bm25s",
-    "peft",
+    "peft>=0.20.0",
+    "flash-linear-attention",   # Triton gated-delta-net kernels for Qwen3.5 training (transformers imports `fla` when present)
 ]
 
 [project.scripts]
```

</details>
