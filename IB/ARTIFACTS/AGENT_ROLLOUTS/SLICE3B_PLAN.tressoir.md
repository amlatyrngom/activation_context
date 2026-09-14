# Slice 3b — activation context in the agent loop

Implemented and validated (v0.5, 2026-09-09). Activation context (AC) enters live agent rollouts through compaction, subagent exchange, large tool outputs and semantic search. Every milestone is in Review with its completion report above the retained plan; the exact delta against your source is staged in [SLICE3B_ROUND](SLICE3B_ROUND.tressoir.md) and applies on your say-so. M7's run-result conversion is done; training on the resulting items is deferred until later slices establish definitive recipes.

## Executive Summary

### Goal

An agent with an AC model configured runs end to end with all four channels, records everything needed to replay or train on the run without storing rows, and falls back to today's text behavior without one. One main-tree test exercises every channel with untrained rows through the real engine interface; M7 must convert our own rollouts into training items. Quality improvements from fitting on those items belong to later slices with definitive training recipes.

### The contract in one table

| Layer | Owns | Record |
| --- | --- | --- |
| `AgentConfig.messages_input` + `user_prompt` | the first user messages: ordered text and parts, task last | `AgentRunResult.prompt_messages` / `prompt_ac_spans` / `prompt_token_ids` |
| `ModelDialect.prompt_tokens` / `continuation_tokens` | parts rendered as sentinels, encoded as pad-id runs; spans returned | `TrajectoryStep.messages` / `ac_spans` / `token_ids` |
| `Agent._encode_parts` | rows through the AC rollout queue, cast once, kept for the run | never stored; `ac_model_name` / `ac_model_version` on the result |
| `Agent.prepare_request` → `engine_submit_tokens` | full-length tensor with zero rows at token positions plus the mask | engine prefix cache keyed by tokens and per-block row hashes |
| `ToolCallResult.content` | ordered `[part, text]` (tool output, subagent) or `[text, part]` (search) | tool step `messages` |
| `Agent.compact` | the finished segment into `compactions`; a new segment from the tree | `finish_reason="compacted"` per segment; selection unrolls all |

| Channel | Visible text | Part | Nested conditioning | Ratio |
| --- | --- | --- | --- | --- |
| Compaction | instructions + the model's summary | previous first user message verbatim, then the segment through the `compact` call | the task, innermost | 1/10 |
| Subagent, both directions | intro + task / "Subagent finished … Answer" | the parent's current segment / the child's final segment (its tree inside) | task-first | 1/20 |
| Large tool output | head/tail 20k chars + file notice | the full output (80k cap) | the parent's current segment at 1/20 | 1/40 |
| Semantic search | top_k passages | the next min(20, 2·top_k) passages | the parent's current segment at 1/20 | 1/20 |

### Milestone map

| Milestone | Delivers | Depends on |
| --- | --- | --- |
| M0 | mixed-prompt submission; `enable_prompt_embeds`; the payload probe and its gate | 3a1 applied |
| M1 | config, step and result contract; tool content; selection unroll; reporter | — |
| M2 | dialect with parts; the agent's rows, spans, request assembly, budgets | M0, M1 |
| M3 | `compact` tool and protocol | M2 |
| M4 | subagent exchange, step mode | M2 |
| M5 | tool-output and search parts | M2 |
| M6 | simulation utilities, the test file, node run | M3–M5 |
| M7 | required study-class conversion from run results to AC training items | M1–M5 records; M6 fixtures |

## Accepted decisions (from your interactions, 2026-09-09)

| # | Question | Accepted | Consequence |
| --- | --- | --- | --- |
| 1 | Tool-output part order | **part before the truncated text** | tool results are `[part, text]`; search stays `[text, part]` |
| 2 | Compaction visible text | **summary + instructions; previous first user message nested verbatim** ("this drift is fine for now") | the AC path and the text path share one shape; harvested items carry the summary |
| 3 | `messages_input` shape | **a list of dialect messages; the task appended as a trailing text part of the last user message** | same structure as AC prefixes and tool results |
| 4 | Step object | **`messages` + `ac_spans`, `activations` dropped** | resume, reporter, cache and training read one record |
| 5 | Parent conditioning | **the full current segment up to the calling turn** | one side-model pass per turn that adds a tool-output or search part |
| 6 | Engine path | **unpatched full-length baseline; M0 measures and gates the two plugin relaxations** | your question about the expected overhead is answered below |
| 7 | M7 scope | **run-result → AC training-item conversion is required in 3b; training itself is deferred** | implement and validate the converter now; fixed-row trainer integration and AC/Agent training runs wait for later definitive recipes |
| 8 | Harvesting location | **a section in `ActivationContextStudyGenerator`** | methods in `activation/ac_model/ac_model_study.py`, using the existing bound model/tokenizer |
| 9 | Synthetic-run helpers | **the existing utils module** | `SyntheticTurn` and `synthesize_agent` live in `activation/agent/agent_utils.py`; tests import them there |
| R1 | Rules diagram | **agreed, "reverify just to be sure"; the child's returned segment carries its recursive compactions** | reverified on CPU below; the returned part is the child's final segment whose first user message holds its tree |
| R2 | Absolute cap | **`finish_reason="trajectory_cap"`** | recorded like `context_exceeded`, never raised |

### Your two questions

**Expected engine overhead compared to everything else (decision 6).** Per turn of one agent on the 4B at a 30k-token prompt, the unpatched path adds: building the full-length bf16 tensor (150 MB, ~30 ms), shipping it to the engine core (~30–60 ms, out-of-band buffers), and SHA-256 hashing of every 16-token block at admission (~80–150 ms at 1–2 GB/s). The same turn otherwise spends: prefill of the new tokens after a prefix-cache hit (1–3k tokens, ~50–150 ms; a full 30k recompute on a miss is ~1–1.5 s), decode of 300–2,000 sampled tokens (5–40 s at 50–100 tokens/s per sequence while batched), tool execution (0.1–5 s), and one side-model encode when a part is added (~0.2–0.6 s for a 32k part). So for a single agent the embeds path is about 1–3% of a turn. The risk is different: hashing runs inside the engine-core process at request admission (`core.py:982` builds the `Request` with the block hasher), so it blocks the scheduler loop for everyone. With N agents each turning every T seconds the blocked fraction is roughly N × 0.12 s / T: about 6% at 8 agents, about 25% at 32, about 50% at 64 with T = 15 s. That is why M0 measures at 1/8/32/64 agents and why the gate is "engine-side overhead above ~20% of turn time at 32 agents". Host memory: transient full-length tensors at 64 agents reach 8–16 GB on a 64 GB host; also measured.

**The right training target from run results (decision 7).** Yes, with one correction on sign. The AC model's objective is a KL between the target model reading the *plain-text* prefix (teacher, no gradient) and the same target reading the *AC* prefix (student) on the *actual continuation*; that is exactly the 3a1 item shape, now built from our own rollouts. Two sources: (a) runs with AC enabled, where the student prefix is the real AC prompt of a compacted-later segment (or a child's real prompt, or a real tool-output part) and the teacher prefix is the same run expanded to text; (b) runs without AC, where the 3a1 generator synthesizes the cuts on our own trajectories exactly as it does on external ones. The completion is the recorded sampled turn. The weight multiplies a KL, which is nonnegative, so a negative weight would maximize divergence and is meaningless: the harvest rejects negative weights and takes the run's score (or a caller-provided nonnegative weight, e.g. successful runs only) as the item weight. Negative advantages belong to a different target, the policy gradient flowing through the rows into the AC modules via the agent trainer; that variant accepts negatives and is a later slice. This specifies the conversion contract only: implement item construction in 3b; defer training integration and optimization runs until later definitive recipe slices. Unscored mechanics fixtures pass an explicit nonnegative weight so default score zero cannot hide an empty converter.

### Reverification on CPU (`IB/TMP/SLICE3B/reverify_shapes.py`, tiny Qwen3.5 fixture, 3a1 part tokenizer)

- Compaction prompt `system, user(task), user([tree, text])` renders two consecutive user messages and one placeholder run whose span matches the requested row count.
- A second compaction nests the first tree verbatim and tokenizes at depth 2.
- The child prompt renders intro text, the parent part, and the task text in that order around one span.
- A tool message whose content is `[part, text]` tokenizes with one span; the nested parent part inside the tool part counts as one child.
- The current dialect wrapper flattens parts away (no sentinel in the rendered continuation): the dialect change in M2 is required, not optional.

## Requested Decisions

None open in this pass. The accepted records above replace the previous components; the rules artifact's two decisions are integrated as R1 and R2.

## Milestones

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">M0 — Engine interface and payload probe</span>
    <span class="card-oneliner">Mixed-prompt submission through the wrapper; the measurement that gates the vLLM relaxations.</span>
    <span class="card-badge">Review</span>
  </summary>
#### What landed

`VLLMWrapper.submit` and `_Replica.submit` take `prompt_embeds` + `prompt_is_token_ids` and build a vLLM `EmbedsPrompt` (import from `vllm.inputs`); `LoadedModel.engine_submit_tokens` passes them through; `engine_to_device` sets `enable_prompt_embeds=True` whenever an AC model is registered before the engine builds. The probe lives at `IB/ARTIFACTS/AGENT_ROLLOUTS/slice3b/probes/ac_payload_probe.py` (a copy rides in `activation/bench/agent_probes/` for the node run and is not in the patch).

#### Drifts, challenges, and unplanned steps

None in the interface. The probe measures lengths 8k/16k/32k/50k at concurrency 1/8/32 plus 16k at 64 (a 64 x 50k tensor set would hold 16 GB of host tensors at once); each point runs token-only, mixed, and mixed-repeat forms with `max_tokens=1`, and a `--max-tokens` flag was added to study decode-phase cache behavior. The probe also established the engine geometry the test relies on: prefix hits come in mamba-aligned blocks of 528 tokens with a bf16 KV cache and 1,056 with the harness default fp8 KV cache (`probe_short.json`, `probe_decode.json`, `ac_cache_diag.py`).

#### Focused actual diffs

`activation/harness/vllm_wrapper.py · _Replica.submit()`

```diff
@@ activation/harness/vllm_wrapper.py — _Replica.submit() @@
-    def submit(self, prompt_token_ids, sampling_params, request_id, lora_request=None) -> concurrent.futures.Future:
-        return asyncio.run_coroutine_threadsafe(
-            self._collect(prompt_token_ids, sampling_params, request_id, lora_request), self.loop,
-        )
+    def submit(self, prompt_token_ids, sampling_params, request_id, lora_request=None,
+               prompt_embeds=None, prompt_is_token_ids=None) -> concurrent.futures.Future:
+        if prompt_embeds is None:
+            prompt = vllm.TokensPrompt(prompt_token_ids=list(prompt_token_ids))
+        else:
+            prompt = EmbedsPrompt(prompt_embeds=prompt_embeds, prompt_token_ids=list(prompt_token_ids),
+                                  prompt_is_token_ids=list(prompt_is_token_ids))
+        return asyncio.run_coroutine_threadsafe(self._collect(prompt, sampling_params, request_id, lora_request), self.loop)
```

`activation/harness/loaded_model.py · engine_to_device()`

```diff
@@ activation/harness/loaded_model.py — engine_to_device() @@
+            if self.harness.module_manager.ac_models:
+                engine_kwargs.setdefault("enable_prompt_embeds", True)   # AC rows travel as prompt embeddings (mixed prompts)
```

#### Validation

Node payload probe: `ac_payload_probe.py` on one RTX PRO 6000 (Qwen3.5-4B, bf16 rows, 2,048 nonzero rows per prompt, `max_tokens=1`), `IB/TMP/SLICE3B/node/AGENT_AC_TEST/probe.json` and `probe_64.json`. Where the token-only and mixed cells share the same prefix-cache state, shipping the full-length tensor costs 1 to 3 percent of wall time at concurrency 8 to 64 (8k x 8: 1.95 s vs 2.01 s; 8k x 32: 5.79 vs 5.92; 50k x 8: 16.7 vs 17.2; 50k x 32: 66.8 vs 67.4; 16k x 64: 33.1 vs 33.3) and 120 to 240 ms per single request (16k x 1: 0.54 vs 0.66 s; 32k x 1: 1.20 vs 1.45 s), which is the serialization of 80 to 160 MB of rows. The cells whose two forms had different cached-token counts (16k x 8, 16k x 32, 32k x 32) are confounded by cross-cell prefix reuse and are not comparable. The repeated mixed prompt hits the prefix cache on every cell (cached tokens = prompt minus the last block, latency 3 to 20x lower) except 50k x 32, where 1.6M tokens exceed the KV budget and evict, which token prompts share. Host RSS grows with the rows in flight (2.4 GB at rest, 10.1 GB after 32 x 50k mixed requests, 7.3 GB after 64 x 16k) and never approached the 64 GB node. Verdict on the M0 gate: engine-side overhead is far below 20 percent at 32 agents and host memory is not under pressure; the vLLM plugin relaxations (short tensor to the last part, hashing only AC-bearing blocks) stay unimplemented.

---

*Planned changes (reference):*


#### Planning Overview

Two harness entry points accept rows and a mask and build a vLLM `EmbedsPrompt`; the engine is constructed with `enable_prompt_embeds=True` whenever an AC model is registered before it builds (next to the existing memory reservation). Rows are the model dtype at the target width on CPU; the mask marks the positions the caller embeds. The IB-only payload probe then measures the unpatched cost and decides on the two plugin relaxations (short tensor up to the last part; hash only AC-bearing blocks). Nothing in the agent depends on the probe's outcome: both relaxations change only what the wrapper ships.

#### Planned Changes

`activation/harness/vllm_wrapper.py · _Replica.submit(), VLLMWrapper.submit()`

```diff
@@ activation/harness/vllm_wrapper.py — imports @@
 import vllm
 from vllm.engine.arg_utils import AsyncEngineArgs
+from vllm.inputs import EmbedsPrompt
 from vllm.sampling_params import RequestOutputKind
@@ activation/harness/vllm_wrapper.py — _Replica._collect() / submit() @@
-    async def _collect(self, prompt_token_ids: list[int], sampling_params, request_id: str, lora_request) -> vllm.RequestOutput:
+    async def _collect(self, prompt, sampling_params, request_id: str, lora_request) -> vllm.RequestOutput:
         sampling_params = sampling_params.clone()
         sampling_params.output_kind = RequestOutputKind.FINAL_ONLY
         final = None
-        async for output in self.engine.generate(
-            vllm.TokensPrompt(prompt_token_ids=prompt_token_ids), sampling_params, request_id, lora_request=lora_request,
-        ):
+        async for output in self.engine.generate(prompt, sampling_params, request_id, lora_request=lora_request):
             final = output

-    def submit(self, prompt_token_ids, sampling_params, request_id, lora_request=None) -> concurrent.futures.Future:
-        return asyncio.run_coroutine_threadsafe(
-            self._collect(prompt_token_ids, sampling_params, request_id, lora_request), self.loop,
-        )
+    def submit(self, prompt_token_ids, sampling_params, request_id, lora_request=None,
+               prompt_embeds=None, prompt_is_token_ids=None) -> concurrent.futures.Future:
+        if prompt_embeds is None:
+            prompt = vllm.TokensPrompt(prompt_token_ids=list(prompt_token_ids))
+        else:
+            # Mixed prompt: the engine embeds every position whose mask is True itself and reads the shipped row elsewhere.
+            # The tensor must be full length (vLLM 0.28 rejects a length mismatch); zero rows sit at token positions.
+            prompt = EmbedsPrompt(prompt_embeds=prompt_embeds, prompt_token_ids=list(prompt_token_ids),
+                                  prompt_is_token_ids=list(prompt_is_token_ids))
+        return asyncio.run_coroutine_threadsafe(self._collect(prompt, sampling_params, request_id, lora_request), self.loop)
@@ activation/harness/vllm_wrapper.py — VLLMWrapper.submit() @@
     def submit(self, prompt_token_ids: list[int], sampling_params: vllm.SamplingParams, request_id: str | None = None,
-               replica: int = 0, lora_request=None) -> concurrent.futures.Future:
+               replica: int = 0, lora_request=None, prompt_embeds=None, prompt_is_token_ids=None) -> concurrent.futures.Future:
         request_id = request_id or uuid.uuid4().hex
-        return self.replicas[replica % len(self.replicas)].submit(prompt_token_ids, sampling_params, request_id, lora_request)
+        return self.replicas[replica % len(self.replicas)].submit(prompt_token_ids, sampling_params, request_id, lora_request,
+                                                                  prompt_embeds, prompt_is_token_ids)
```

`activation/harness/loaded_model.py · engine_to_device(), engine_submit_tokens()`

```diff
@@ activation/harness/loaded_model.py — engine_to_device() @@
             if self.harness.module_manager.ac_models and "gpu_memory_utilization" in engine_kwargs:
                 reservation = self.harness.harness_config.ac_engine_memory_reservation
                 engine_kwargs["gpu_memory_utilization"] = round(engine_kwargs["gpu_memory_utilization"] - reservation, 3)
                 print(f"{self.model_config.model_name} - Engine memory share lowered by {reservation} for the AC side model.")
+            if self.harness.module_manager.ac_models:
+                engine_kwargs.setdefault("enable_prompt_embeds", True)   # AC rows travel as prompt embeddings (mixed prompts)
@@ activation/harness/loaded_model.py — engine_submit_tokens() @@
     def engine_submit_tokens(
         self,
         prompt_token_ids: list[int],
         seed: int|None = None,
         agent_id: str = "",
         lora_name: str|None = None,
         chat_kwargs: dict|None = None,
         record_sampling: bool = False,
+        prompt_embeds: "torch.Tensor | None" = None,
+        prompt_is_token_ids: list[bool] | None = None,
     ) -> EngineChatOutput:
         """
         One request whose prompt the caller owns as token ids (the agent loop keeps its own prefix).
         ...
+        With `prompt_embeds` ([len(prompt_token_ids), d_model], the model dtype, on CPU) the positions whose
+        `prompt_is_token_ids` entry is False take their embedding from that tensor (activation-context rows);
+        every other position is embedded by the engine from its token id as usual.
         """
         ...
-        future = engine.submit(list(prompt_token_ids), sampling_params, replica=VLLMWrapper.replica_for(agent_id, engine.world_size),
-                               lora_request=lora_request)
+        future = engine.submit(list(prompt_token_ids), sampling_params, replica=VLLMWrapper.replica_for(agent_id, engine.world_size),
+                               lora_request=lora_request, prompt_embeds=prompt_embeds, prompt_is_token_ids=prompt_is_token_ids)
```

`IB/ARTIFACTS/AGENT_ROLLOUTS/slice3b/probes/ac_payload_probe.py` (IB-only, no diff card)

Zero-row full-length tensors with one nonzero span, prompt lengths 8k/16k/32k/50k, concurrency 1/8/32/64 through `engine_submit_tokens` with `max_tokens=1`; the same request twice for `cached_prompt_token_count`; a token-only control at every point; host RSS sampled; JSON plus a table under `IB/TMP/SYNC/AC_PAYLOAD_PROBE/`. Read-out: engine-side overhead per turn versus the control, its growth with concurrency, cache hits on the repeat. Gate: overhead above ~20% of turn time at 32 agents, or host memory pressure, triggers the plugin relaxations as a separate milestone.

</details>

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">M1 — Prompt and step contract</span>
    <span class="card-oneliner">`messages_input`, ordered content, spans on steps, compactions on results; selection, cache and reporter follow.</span>
    <span class="card-badge">Review</span>
  </summary>
#### What landed

`AgentConfig.messages_input` (the task appended as the trailing text part of the last user message), the four ratio fields, `TrajectoryStep.messages` + `ac_spans` (`activations` gone), `AgentRunResult.compactions`, `prompt_messages`, `prompt_ac_spans`, `ac_model_name`, `ac_model_version`, `num_ac_parts`/`num_ac_rows`, the new finish reasons; `ToolCallResult.content` + `compacted` with `tool_content()`; `unroll` over segments and the children attached to each; selection always includes compactions; `build_example` rejects AC-bearing runs until the fixed-row trainer lands; the reporter shows a rows line per part and the segment/part counts; the rollout cache key covers `messages_input` and `ac_model_name`; `AgentConfig.deserialize` drops a legacy `ac_inputs` key.

#### Drifts, challenges, and unplanned steps

The part primitives moved to a new `activation/common/ac_parts.py` (`AC_PART_TYPE`, `PART_SENTINEL`, `ac_part`, `is_ac_part`, `direct_parts`, `flatten_with_sentinels`, `encode_with_part_sentinels`, `COMPACTION_INSTRUCTIONS`) because `ac_model_study.py` imports the agent dialect: the agent could not import from `activation.ac_model` without a cycle. `ac_model_utils` and `ac_model_study` re-export the names, so 3a1 callers are unchanged. `ac_part` gained an optional `kind` (compaction, subagent_prompt, subagent_return, tool_output, search, parent_context) so the harvest expands each part by its channel instead of guessing from structure. The reporter's per-agent trajectory file omits `messages` (parts nest the parent segment; the run cache keeps the full record).

#### Focused actual diffs

`activation/agent/agent_config.py · AgentRunResult`

```diff
@@ activation/agent/agent_config.py — AgentRunResult @@
+    compactions: list["AgentRunResult"] = field(default_factory=list)   # this agent's earlier segments, oldest first (finish_reason "compacted")
     finish_reason: str = ""                                    # submitted | max_turns | max_tool_errors | max_duration | no_tool_call | context_exceeded
+                                                               # | trajectory_cap | compaction_refused | compacted (a segment) | simulated | error
     seed: int = 0
     prompt_token_ids: list[int] = field(default_factory=list)  # the segment's first prompt (system + first messages + tools, templated once)
+    prompt_messages: list[dict] = field(default_factory=list)  # the dialect messages of that prompt, parts inline (a compaction tree lives here)
+    prompt_ac_spans: list[dict] = field(default_factory=list)  # [{"start", "length"}] placeholder runs inside prompt_token_ids
+    ac_model_name: str | None = None                           # the encoder that produced the rows, and its version at run time
+    ac_model_version: int | None = None
```

`activation/common/ac_parts.py · ac_part()` (new module)

```python
def ac_part(messages: list[dict], ac_name: str, compression_target: float, kind: str | None = None) -> dict:
    part = {"type": AC_PART_TYPE, "ac_name": ac_name, "compression_target": compression_target, "messages": messages}
    if kind is not None:
        part["kind"] = kind
    return part
```

`activation/agent_training/agent_trainer.py · unroll()`

```diff
@@ activation/agent_training/agent_trainer.py — unroll() @@
-    out = [result]
-    for child in result.subagent_results:
-        out.extend(unroll(child))
+    out = []
+    for segment in list(result.compactions) + [result]:
+        out.append(segment)
+        for child in segment.subagent_results:
+            out.extend(unroll(child))
```

#### Validation

Serialization round trip and `unroll`/`items_for` counts in `channels_cpu.py`; `AgentRunResult.deserialize` reads the nested `compactions` with the live config. `IB/TMP/SLICE3B/cpu_checks/channels_cpu.py` (tiny Qwen3.5 fixture as side and target, no engine, no sandbox) passes in ~9 s: compaction after three long outputs, a refused call, a lone `compact` restarting the segment (9 recorded steps, tree of 10 messages, 64 rows at ratio 1/10 written over a pad run, bf16 rows), a second compaction nesting the first tree verbatim, `unroll` giving 3 segments, the subagent in step mode (child prompt `[text, part, text]`, parent tool step `[part, text]`), a 100k-char output truncated to <21k with an 80k-char part nested with the parent segment, search with 3 visible passages and a 6-passage part, the serialize/deserialize/resume round trip with identical prefix, spans and rows, the text-only compaction fallback, and the five-refusal cap ending the run with `compaction_refused`.

---

*Planned changes (reference):*


#### Planning Overview

The prompt and step contract. `AgentConfig.messages_input` replaces `ac_inputs`; four ratio fields replace the single one; `TrajectoryStep` records the dialect messages it appended and its placeholder spans; `AgentRunResult` records the segment's first messages and spans, the compacted segments, and the AC identity; `ToolCallResult` carries ordered content; selection unrolls compactions unconditionally; the reporter shows parts; the unfinished `CompactionTool` registration (3a1 finding A1) leaves the defaults until M3 lands the real tool. Cache rows and reports change with the step object; the deserializer accepts both the old `activations` key (ignored) and the new fields.

#### Planned Changes

`activation/agent/agent_config.py · AgentConfig, TrajectoryStep, AgentRunResult`

```diff
@@ activation/agent/agent_config.py — AgentConfig @@
     system_prompt: str = ""                                   # System prompt.
     user_prompt: str = ""                                     # User prompt (the task).
-    ac_inputs: dict[str, object] = field(default_factory=dict)  # Activation-context inputs; carried as data in slice 1.
+    messages_input: list[dict] = field(default_factory=list)  # Dialect messages ahead of the task: content may be ordered text and
+                                                              # activation_context parts. The task is appended as a trailing text part
+                                                              # of the last user message when given; empty = today's prompt.
     ...
-    enable_ac_communication: bool = False                     # Subagents exchange trajectories as activation context.
+    enable_ac_communication: bool = False                     # Subagents exchange trajectories as activation context (needs ac_model_name).

     # Budgets
     max_turns: int = 20
     max_tool_errors: int = 5
     max_duration: float = 600                                 # seconds
-    compaction_threshold_tokens: int = 32768                  # carried; compaction is slice 2. @AI: ...
-    absolute_trajectory_cap: int = 50_000 # @AI: Simply err if this is every reached.
-    ac_compaction_ratio: float|None = 1.0/16.0 # ~threshold*ratio-sized vector (slightly more due to nesting+text logic).
+    compaction_threshold_tokens: int = 32768                  # soft delta over the segment's start length; exceeded by at most one turn
+    absolute_trajectory_cap: int = 50_000                     # prefix length that ends the run with finish_reason "trajectory_cap"
+    # Activation-context ratios (rows per side token) of the four channels.
+    ac_compaction_ratio: float = 1.0 / 10.0
+    ac_subagent_ratio: float = 1.0 / 20.0                     # parent -> child prompt, child -> parent result, and the nested parent context
+    ac_tool_output_ratio: float = 1.0 / 40.0
+    ac_search_ratio: float = 1.0 / 20.0
@@ activation/agent/agent_config.py — AgentConfig.serialize() @@
-        data["ac_inputs"] = _jsonable(self.ac_inputs)
+        data["messages_input"] = _jsonable(self.messages_input)
@@ activation/agent/agent_config.py — TrajectoryStep @@
     role: str                                                  # "assistant" | "tool" | "user" (the nudge)
     content: str                                               # assistant text (tool-call blocks removed) or the joined tool outputs
     tool_calls: list[dict] = field(default_factory=list)       # [{"id", "name", "arguments"}]
     tool_call_results: list[str] = field(default_factory=list) # truncated outputs, same order as tool_calls
-    activations: dict[str, object] = field(default_factory=dict)  # ac_outputs of this step's tools, keyed by call id
+    messages: list[dict] = field(default_factory=list)         # the dialect messages this step appended (parts in order): one assistant
+                                                               # message, the tool messages, or the nudge
+    ac_spans: list[dict] = field(default_factory=list)         # [{"start", "length"}] placeholder runs inside token_ids, one per
+                                                               # activation_context part of `messages`, in order of appearance
     token_ids: list[int] = field(default_factory=list)
     logprobs: list[float] = field(default_factory=list)
@@ activation/agent/agent_config.py — AgentRunResult @@
-    num_ac_input_bytes: int = 0
-    num_ac_input_tokens: int = 0
+    num_ac_parts: int = 0                                      # parts encoded for this segment's prompts (nested children not counted)
+    num_ac_rows: int = 0                                       # rows those parts occupy in the prompt
     ...
     subagent_results: list["AgentRunResult"] = field(default_factory=list)
-    finish_reason: str = ""                                    # submitted | max_turns | ... | context_exceeded | error
+    compactions: list["AgentRunResult"] = field(default_factory=list)   # this agent's earlier segments, oldest first (finish_reason "compacted")
+    finish_reason: str = ""                                    # submitted | max_turns | max_tool_errors | max_duration | no_tool_call
+                                                               # | context_exceeded | trajectory_cap | compaction_refused | compacted | error
     seed: int = 0
     prompt_token_ids: list[int] = field(default_factory=list)  # the segment's first prompt (system + first messages + tools, templated once)
+    prompt_messages: list[dict] = field(default_factory=list)  # the dialect messages of that prompt (parts inline; a compaction tree lives here)
+    prompt_ac_spans: list[dict] = field(default_factory=list)  # placeholder runs inside prompt_token_ids
+    ac_model_name: str | None = None                           # the encoder that produced the rows, and its version at run time
+    ac_model_version: int | None = None
@@ activation/agent/agent_config.py — AgentRunResult.serialize() / deserialize() @@
-        data = {key: value for key, value in self.__dict__.items() if key not in ("agent_config", "subagent_results", "trajectory", "answer")}
+        data = {key: value for key, value in self.__dict__.items()
+                if key not in ("agent_config", "subagent_results", "compactions", "trajectory", "answer")}
         ...
         data["subagent_results"] = [result.serialize() for result in self.subagent_results]
+        data["compactions"] = [result.serialize() for result in self.compactions]
 ...
         children = [AgentRunResult.deserialize(child, harness) for child in data.pop("subagent_results", [])]
-        return AgentRunResult(agent_config=config, subagent_results=children, **data)
+        segments = [AgentRunResult.deserialize(segment, harness, agent_config=config) for segment in data.pop("compactions", [])]
+        return AgentRunResult(agent_config=config, subagent_results=children, compactions=segments, **data)
```

`activation/agent/agent_tools.py · ToolCallResult, tool_content()`

```diff
@@ activation/agent/agent_tools.py — ToolCallResult @@
 @dataclass
 class ToolCallResult:
     output: str                                          # what the model sees as text (already truncated)
     is_error: bool = False
-    ac_outputs: dict[str, t.Any] = field(default_factory=dict)  # activation context outputs (ac name -> ac input)
     is_final: bool = False                               # submit_answer sets it
+    content: list[dict] | None = None                    # ordered content parts (text and activation_context) when the result carries
+                                                         # activation context; None means [text(output)]. Exactly one text part equals output.
+    compacted: bool = False                              # the compaction tool restarted the segment: no tool message follows this result
+
+
+def tool_content(result: ToolCallResult) -> list[dict]:
+    return result.content if result.content is not None else [{"type": "text", "text": result.output}]
```

`activation/agent_training/agent_trainer.py · unroll()` and `agent_training_selection.py · items_for()`

```diff
@@ activation/agent_training/agent_trainer.py — unroll() @@
 def unroll(result: AgentRunResult) -> list[AgentRunResult]:
-    """The run and every subagent run beneath it, depth first."""
-    out = [result]
+    """The run, its compacted segments, and every subagent run beneath it (with theirs), depth first."""
+    out = []
-    for child in result.subagent_results:
-        out.extend(unroll(child))
+    for segment in list(result.compactions) + [result]:
+        out.append(segment)
+        for child in segment.subagent_results:
+            out.extend(unroll(child))
     return out
@@ activation/agent_training/agent_training_selection.py — items_for() @@
-    runs = unroll(run) if include_subagents else [run]
+    runs = unroll(run) if include_subagents else list(run.compactions) + [run]   # compactions are the same agent: always included
```

`activation/agent_training/agent_training_utils.py · build_example()`

```diff
@@ activation/agent_training/agent_training_utils.py — build_example() @@
     run = item.run_results
+    if run.prompt_ac_spans or any(step.get("ac_spans") for step in run.trajectory):
+        raise ValueError("Training on AC-bearing runs awaits the later fixed-row trainer integration")
     ...
```

The record-reader migration includes children attached to completed segments. Since new AC-row training is deferred, the existing training entry rejects AC-bearing sequences before treating placeholders as token embeddings; ordinary text-only training remains covered by its existing regression checks. This guard is compatibility work, not implementation of the deferred trainer path. Remaining example-building code is unchanged and omitted.

`activation/agent/rollout_reporter.py · trajectory_item()`

```diff
@@ activation/agent/rollout_reporter.py — trajectory_item() @@
         elif step["role"] == "tool":
             steps.append({
                 "role": "tool", "turn": turn, "content": "",
-                "results": [{"name": call["name"], "output": _cut(output, RESULT_CHARS)}
-                            for call, output in zip(step["tool_calls"], step["tool_call_results"])],
+                "results": [{"name": call["name"], "output": _cut(output, RESULT_CHARS) + _parts_note(step)}
+                            for call, output in zip(step["tool_calls"], step["tool_call_results"])],
             })
+
+
+def _parts_note(step: dict) -> str:
+    """One line per activation-context part of the step: rows and nesting depth, so the page shows what the model read as rows."""
+    spans = step.get("ac_spans") or []
+    if not spans:
+        return ""
+    return "\n" + "\n".join(f"[activation context part {index + 1}: {span['length']} rows]" for index, span in enumerate(spans))
```

Also in this milestone (no cards): `AgentConfig.deserialize` drops a legacy `ac_inputs` key; `rollout_caching.config_key` hashes `messages_input` too; the `_count_ac_inputs` counter in `agent.py` goes away with its fields; `DEFAULT_TOOLS` loses the unfinished `CompactionTool` entry until M3 replaces it.

</details>

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">M2 — Agent loop with parts</span>
    <span class="card-oneliner">Parts rendered as placeholder runs, encoded once, rows kept per agent, shipped every turn; budgets count rows.</span>
    <span class="card-badge">Review</span>
  </summary>
#### What landed

`ModelDialect.prompt_tokens` / `continuation_tokens` return `(ids, spans)` with parts rendered as sentinels and encoded as pad-id runs through the shared `encode_with_part_sentinels`; the text-only `prompt_token_ids` / `continuation_token_ids` remain as wrappers. `Agent` keeps `messages`, `prefix`, `spans`, `rows` (bf16 rows kept for the run), `segment_start`, `compaction_due`; `begin()` templates the first segment once (`first_user_messages()` builds it from `messages_input` + task); `record_assistant_turn` / `append_user_message` / `append_tool_results` are the loop's steps; `prepare_request()` assembles the full-length tensor and mask; `_submit()` ships them; `_after_append()` applies the cap (`trajectory_cap`) and the soft delta; `_check_engine_context()` asserts cap + max_tokens <= max_model_len.

#### Drifts, challenges, and unplanned steps

`_submit(chat_kwargs=None)` takes an override so `submit_probe(max_tokens=)` can sample short; `_continuation` shifts the spans left when `join_continuation` drops the wrapper's leading end-of-turn token (the sample already ended with it). Rows are `detach().to(bf16).cpu().contiguous()` per part; `row_dtype` reads `ModelConfig.dtype` (bf16 by default). `_pad_id` uses the target tokenizer's pad id. The InlineRendering wraps each result's content in `<tool_response>` text parts so parts survive templates without tool support.

#### Focused actual diffs

`activation/agent/agent.py · prepare_request(), _continuation()`

```python
    def prepare_request(self) -> tuple[list[int], torch.Tensor | None, list[bool] | None]:
        if not self.spans:
            return list(self.prefix), None, None
        embeds = torch.zeros(len(self.prefix), self.loaded_model.model_config.model_description.d_model, dtype=self.row_dtype)
        mask = [True] * len(self.prefix)
        for (start, end), rows in zip(self.spans, self.rows):
            embeds[start:end] = rows
            mask[start:end] = [False] * (end - start)
        return list(self.prefix), embeds, mask

    def _continuation(self, new_messages: list[dict]) -> tuple[list[int], list[tuple[int, int]]]:
        lengths = self._part_lengths(new_messages)
        wrapper, spans = self.dialect.continuation_tokens(self.loaded_model.tokenizer, self.messages, new_messages, self.template_tools,
                                                          self.template_kwargs, lengths, self._pad_id())
        joined = self.dialect.join_continuation(self.last_sampled, wrapper)
        shift = len(wrapper) - len(joined)                                            # a dropped leading end-of-turn token shifts the spans
        return joined, [(start - shift, end - shift) for start, end in spans]
```

`activation/agent/agent_utils.py · ModelDialect.continuation_tokens()`

```diff
@@ activation/agent/agent_utils.py — continuation_tokens() @@
-        history = list(messages_before) + [{"role": "assistant", "content": ASSISTANT_SENTINEL}] + list(new_messages)
+        history = (flatten_with_sentinels(messages_before, parts="drop") + [{"role": "assistant", "content": ASSISTANT_SENTINEL}]
+                   + flatten_with_sentinels(new_messages, parts="sentinel"))
         text = _template_text(tokenizer, history, tools, True, chat_template_kwargs, parts="drop")
         index = text.rfind(ASSISTANT_SENTINEL)
-        wrapper = text[index + len(ASSISTANT_SENTINEL):]
-        return list(tokenizer.encode(wrapper, add_special_tokens=False))
+        return encode_with_part_sentinels(tokenizer, text[index + len(ASSISTANT_SENTINEL):], list(part_lengths), pad_id)
```

#### Validation

Spans align with the encoded rows on every append (asserted in `_append_step` and `_begin_segment`); placeholder runs hold the pad id; the embeds tensor has the prefix length and as many masked positions as rows. `IB/TMP/SLICE3B/cpu_checks/channels_cpu.py` (tiny Qwen3.5 fixture as side and target, no engine, no sandbox) passes in ~9 s: compaction after three long outputs, a refused call, a lone `compact` restarting the segment (9 recorded steps, tree of 10 messages, 64 rows at ratio 1/10 written over a pad run, bf16 rows), a second compaction nesting the first tree verbatim, `unroll` giving 3 segments, the subagent in step mode (child prompt `[text, part, text]`, parent tool step `[part, text]`), a 100k-char output truncated to <21k with an 80k-char part nested with the parent segment, search with 3 visible passages and a 6-passage part, the serialize/deserialize/resume round trip with identical prefix, spans and rows, the text-only compaction fallback, and the five-refusal cap ending the run with `compaction_refused`. Node engine probes: `uv run pytest activation/tests/test_basic_agent_ac.py --gpu --slow -s` on `ac-slice3b-temp` (RTX PRO 6000, Qwen3.5-4B target with the fp8 KV cache, Qwen3.5-0.8B side, untrained AC rows), logs under `IB/TMP/SLICE3B/node/AGENT_AC_TEST/`. Simulated channels (`pytest_simulated.log`, 71 s): compaction prompt 1,259 tokens with rows at 391 positions, repeat cached 1,056; second compaction 910 tokens / 53 rows; subagent 944 tokens / 4 rows; a real 100k-character python output in the sandbox: 3,620 tokens / 251 rows, repeat cached 3,168; BRIGHT search: 2,161 tokens / 117 rows, repeat cached 2,112; every request accepted with the full-length bf16 tensor and the model answered sensibly on untrained rows (a python tool call, a search follow-up). With-AC rollouts (`pytest_rerun.log`, threshold 300 tokens): seed 0 submitted after 8 turns, 2 compactions, 2 parts / 223 rows, 4,224 cached prompt tokens on its last request; seed 1 after 4 turns, 1 compaction, 60 rows, 1,056 cached; both answered 2441 (the correct sum); the harvest produced 6 items from 12 candidates (6 turns before any part skipped). Text-only rollouts (`pytest.log`): both seeds submitted 2441 after 4 turns with 1 compaction each. Final clean run of the whole file after the test fixes (`pytest_final.log`): 3 passed in 170 s; in that run the with-AC seed 0 hit `max_turns` after 12 turns with 3 compactions and 355 rows (untrained rows: the model kept re-deriving the primes) while seed 1 submitted 2441 after one compaction, and the harvest yielded 10 items from 16 candidates.

---

*Planned changes (reference):*


#### Planning Overview

The loop keeps the conversation twice per step, as dialect messages with parts and as tokens with placeholder spans. The dialect renders parts as sentinels and encodes the pieces around them with pad-id runs (the same routine the AC tokenizer uses, factored out of `tokenize_with_parts`). The agent encodes each new part once through the AC rollout queue, casts the rows to the target dtype and keeps them for the run; every request assembles a full-length tensor and mask from the retained rows. Threshold and cap accounting run after every append; a run-start check asserts the cap plus the output budget fits the engine context. Without an AC model nothing changes except the step record.

#### Planned Changes

`activation/ac_model/ac_model_utils.py · encode_with_part_sentinels()` (factored out of `tokenize_with_parts`)

```diff
@@ activation/ac_model/ac_model_utils.py — tokenize_with_parts() @@
     text = tokenizer.apply_chat_template(flat, tools=tools, add_generation_prompt=add_generation_prompt, tokenize=False,
                                          **(chat_template_kwargs or {}))
-    ids: list[int] = []
-    spans: list[tuple[int, int]] = []
-    cursor = 0
-    for part_index, length in enumerate(part_lengths):
-        sentinel = PART_SENTINEL.format(index=part_index)
-        at = text.find(sentinel, cursor)
-        assert at >= 0, "the chat template did not render a part sentinel verbatim"
-        ids.extend(tokenizer.encode(text[cursor:at], add_special_tokens=False))
-        spans.append((len(ids), len(ids) + length))
-        ids.extend([pad_id] * length)
-        cursor = at + len(sentinel)
-    ids.extend(tokenizer.encode(text[cursor:], add_special_tokens=False))
-    return ids, spans
+    return encode_with_part_sentinels(tokenizer, text, part_lengths, pad_id)
+
+
+def encode_with_part_sentinels(tokenizer, text: str, part_lengths: list[int], pad_id: int) -> tuple[list[int], list[tuple[int, int]]]:
+    """Token ids of rendered text whose part sentinels become `pad_id` runs of the given lengths; (start, end) per run."""
+    ids: list[int] = []
+    spans: list[tuple[int, int]] = []
+    cursor = 0
+    for part_index, length in enumerate(part_lengths):
+        sentinel = PART_SENTINEL.format(index=part_index)
+        at = text.find(sentinel, cursor)
+        assert at >= 0, "the chat template did not render a part sentinel verbatim"
+        ids.extend(tokenizer.encode(text[cursor:at], add_special_tokens=False))
+        spans.append((len(ids), len(ids) + length))
+        ids.extend([pad_id] * length)
+        cursor = at + len(sentinel)
+    ids.extend(tokenizer.encode(text[cursor:], add_special_tokens=False))
+    return ids, spans
```

`activation/agent/agent_utils.py · ModelDialect.prompt_tokens(), continuation_tokens(), _flatten()`

```diff
@@ activation/agent/agent_utils.py — imports @@
+from activation.ac_model.ac_model_utils import PART_SENTINEL, encode_with_part_sentinels, is_ac_part
@@ activation/agent/agent_utils.py — ModelDialect tokens @@
-    def prompt_token_ids(self, tokenizer, messages, tools=None, chat_template_kwargs=None) -> list[int]:
-        """The first prompt of a conversation: the chat template, once, with the generation prompt open."""
-        return _template_ids(tokenizer, messages, tools, True, chat_template_kwargs)
+    def prompt_tokens(self, tokenizer, messages: list[dict], tools: list[dict] | None = None, chat_template_kwargs: dict | None = None,
+                      part_lengths: list[int] = (), pad_id: int = 0) -> tuple[list[int], list[tuple[int, int]]]:
+        """
+        The first prompt of a conversation: the chat template, once, with the generation prompt open. Every
+        activation_context part in `messages` renders as a sentinel and becomes a run of `part_lengths[k]`
+        placeholder ids; the (start, end) of each run is returned for the caller to write rows over.
+        """
+        text = _template_text(tokenizer, _flatten(messages, parts="sentinel"), tools, True, chat_template_kwargs)
+        return encode_with_part_sentinels(tokenizer, text, list(part_lengths), pad_id)

-    def continuation_token_ids(self, tokenizer, messages_before, new_messages, tools=None, chat_template_kwargs=None) -> list[int]:
+    def continuation_tokens(self, tokenizer, messages_before: list[dict], new_messages: list[dict], tools: list[dict] | None = None,
+                            chat_template_kwargs: dict | None = None, part_lengths: list[int] = (), pad_id: int = 0) -> tuple[list[int], list[tuple[int, int]]]:
         """
         The tokens the model reads between its own sampled text and its next turn ... (unchanged) ...
+        Parts inside `new_messages` become placeholder runs (spans relative to the returned wrapper); parts in
+        `messages_before` sit before the sentinel and only need to render as some text.
         """
-        history = list(messages_before) + [{"role": "assistant", "content": ASSISTANT_SENTINEL}] + list(new_messages)
+        history = (_flatten(messages_before, parts="drop") + [{"role": "assistant", "content": ASSISTANT_SENTINEL}]
+                   + _flatten(new_messages, parts="sentinel"))
         text = _template_text(tokenizer, history, tools, True, chat_template_kwargs)
         index = text.rfind(ASSISTANT_SENTINEL)
         if index < 0:
             raise ValueError("the chat template did not render the assistant content verbatim; cannot derive the turn wrapper")
-        wrapper = text[index + len(ASSISTANT_SENTINEL):]
-        return list(tokenizer.encode(wrapper, add_special_tokens=False))
+        return encode_with_part_sentinels(tokenizer, text[index + len(ASSISTANT_SENTINEL):], list(part_lengths), pad_id)
@@ activation/agent/agent_utils.py — _flatten() @@
-def _flatten(messages: list[dict]) -> list[dict]:
-    """Text-only content parts joined, the shape our templates take."""
+def _flatten(messages: list[dict], parts: str = "drop") -> list[dict]:
+    """Content lists joined into text for the template: text parts verbatim, activation_context parts as a numbered sentinel or dropped."""
     out = []
+    index = 0
     for message in messages:
         message = dict(message)
         content = message.get("content")
         if isinstance(content, list):
-            message["content"] = "".join(part.get("text", "") for part in content if isinstance(part, dict))
+            pieces = []
+            for part in content:
+                if is_ac_part(part):
+                    if parts == "sentinel":
+                        pieces.append(PART_SENTINEL.format(index=index))
+                        index += 1
+                elif isinstance(part, dict):
+                    pieces.append(part.get("text", ""))
+            message["content"] = "".join(pieces)
         out.append(message)
     return out
```

`_template_ids` and `template()` in the wrapper keep their text-only behavior for study callers.

`activation/agent/agent.py · Agent state, begin(), the append helpers, prepare_request(), _submit()`

```diff
@@ activation/agent/agent.py — imports and constants @@
+import torch
+from activation.ac_model.ac_model_utils import direct_parts
 ...
 MAX_CONSECUTIVE_NO_TOOL_TURNS = 3
+MAX_COMPACTION_REFUSALS = 5        # turns that ignore a due compaction before the run ends with compaction_refused
@@ activation/agent/agent.py — Agent.__init__() @@
         self.messages: list[dict] = []     # dialect messages of the current segment, parts inline
-        self.prefix: list[int] = []        # the token prompt of the next turn
+        self.prefix: list[int] = []        # the token prompt of the next turn (pad ids at part positions)
+        self.spans: list[tuple[int, int]] = []   # (start, end) of every part in the prefix, in order
+        self.rows: list[torch.Tensor] = []       # the rows of those parts, target dtype on CPU, kept for the run (identical bytes each turn)
+        self.segment_start = 0             # len(prompt_token_ids) of the current segment: the compaction delta counts from here
+        self.compaction_due = False
+        self.compaction_refusals = 0
+        self.no_tool_turns = 0
+        self.errors = 0
+        self.last_sampled: list[int] = []  # the previous assistant turn's tokens, for join_continuation
+        self.step_mode = False             # simulation: subagents do not run, nothing is submitted
+        self.started = False
@@ activation/agent/agent.py — AC accessors @@
+    @property
+    def ac_model(self):
+        name = self.agent_config.ac_model_name
+        return self.harness.module_manager.get_ac_model(name) if name else None
+
+    @property
+    def row_dtype(self) -> torch.dtype:
+        dtype = self.loaded_model.model_config.dtype
+        return dtype if isinstance(dtype, torch.dtype) else torch.bfloat16
+
+    def segment_messages(self) -> list[dict]:
+        """The current segment without the system message: what a part sees as 'the parent so far'."""
+        return [message for message in self.messages if message.get("role") != "system"]
+
+    def context_part(self, ratio: float) -> dict:
+        """The current segment as an activation_context part (task first; a compaction tree rides inside its first user message)."""
+        return ac_part(deepcopy(self.segment_messages()), self.agent_config.ac_model_name, ratio)
+
+    def _part_lengths(self, messages: list[dict]) -> list[int]:
+        ac_model = self.ac_model
+        return [ac_model.part_view_rows(part["messages"], part.get("compression_target")) for part in direct_parts(messages)] if ac_model else []
+
+    def _encode_parts(self, messages: list[dict]) -> list[torch.Tensor]:
+        """Rows of the parts in `messages`, in order, through the rollout queue (children first, cached), cast once to the target dtype."""
+        ac_model = self.ac_model
+        if ac_model is None:
+            assert not direct_parts(messages), "activation_context parts need agent_config.ac_model_name"
+            return []
+        futures = [ac_model.encode_async(part["messages"], part.get("compression_target")) for part in direct_parts(messages)]
+        rows = [future.result().to(dtype=self.row_dtype).cpu().contiguous() for future in futures]
+        self.run_results.num_ac_parts += len(rows)
+        self.run_results.num_ac_rows += sum(int(r.shape[0]) for r in rows)
+        return rows
@@ activation/agent/agent.py — begin() replaces the prompt setup at the top of run() @@
+    def first_user_messages(self) -> list[dict]:
+        """messages_input with the task appended as a trailing text part of the last user message (today's prompt when empty)."""
+        config = self.agent_config
+        messages = deepcopy(config.messages_input)
+        if not messages or messages[-1].get("role") != "user":
+            messages.append({"role": "user", "content": config.user_prompt})
+        elif config.user_prompt:
+            content = messages[-1]["content"]
+            content = [{"type": "text", "text": content}] if isinstance(content, str) else list(content)
+            messages[-1]["content"] = content + [{"type": "text", "text": config.user_prompt}]
+        return messages
+
+    def begin(self) -> None:
+        """Dialect, tool schemas and the first segment; templated once. Safe without an engine (tokenizer only)."""
+        config, results = self.agent_config, self.run_results
+        self.dialect = ModelDialect.for_tokenizer(self.loaded_model.tokenizer)
+        self.tool_definitions = self._tool_definitions()
+        self.parameter_types = parameter_types_of(self.tool_definitions)
+        self.template_tools = self.dialect.rendering.template_tools(self.tool_definitions)
+        self.template_kwargs = self.loaded_model.chat_template_kwargs(config.call_kwargs)
+        results.ac_model_name = config.ac_model_name
+        results.ac_model_version = self.ac_model.version if self.ac_model else None
+        system = self.dialect.rendering.system_prompt(config.system_prompt or "", self.tool_definitions)
+        self._begin_segment(([{"role": "system", "content": system}] if system else []) + self.first_user_messages())
+        self.started = True
+
+    def _begin_segment(self, messages: list[dict]) -> None:
+        """A fresh prompt for a new segment (run start, or after a compaction): rows for its parts, a new start length."""
+        results = self.run_results
+        tokenizer = self.loaded_model.tokenizer
+        lengths = self._part_lengths(messages)
+        ids, spans = self.dialect.prompt_tokens(tokenizer, messages, self.template_tools, self.template_kwargs, lengths, self._pad_id())
+        self.messages = list(messages)
+        self.prefix, self.spans, self.rows = list(ids), list(spans), self._encode_parts(messages)
+        self.segment_start = len(ids)
+        self.last_sampled = []
+        results.prompt_token_ids, results.prompt_messages = list(ids), deepcopy(messages)
+        results.prompt_ac_spans = [{"start": start, "length": end - start} for start, end in spans]
+        self._after_append()
@@ activation/agent/agent.py — append helpers (the loop body of run() moves into these) @@
+    def record_assistant_turn(self, content: str, calls: list[dict], token_ids: list[int], logprobs: list[float]) -> None:
+        message = self.dialect.rendering.assistant_message(content, calls)
+        step = TrajectoryStep(role="assistant", content=content, tool_calls=calls, messages=[message], token_ids=list(token_ids), logprobs=list(logprobs))
+        self.messages.append(message)
+        self.prefix += list(token_ids)
+        self.last_sampled = list(token_ids)
+        self.run_results.trajectory.append(asdict(step))
+
+    def append_user_message(self, text: str) -> None:
+        message = {"role": "user", "content": text}
+        wrapper, spans = self._continuation([message])
+        self._append_step(TrajectoryStep(role="user", content=text, messages=[message], token_ids=wrapper), [message], spans, [])
+
+    def append_tool_results(self, calls: list[dict], results: list[ToolCallResult]) -> None:
+        tool_messages = self.dialect.rendering.tool_messages(calls, results)          # content lists with parts, in order
+        wrapper, spans = self._continuation(tool_messages)
+        rows = self._encode_parts(tool_messages)
+        step = TrajectoryStep(role="tool", content="\n\n".join(result.output for result in results), tool_calls=calls,
+                              tool_call_results=[result.output for result in results], messages=tool_messages, token_ids=wrapper)
+        self._append_step(step, tool_messages, spans, rows)
+
+    def _continuation(self, new_messages: list[dict]) -> tuple[list[int], list[tuple[int, int]]]:
+        lengths = self._part_lengths(new_messages)
+        wrapper, spans = self.dialect.continuation_tokens(self.loaded_model.tokenizer, self.messages, new_messages, self.template_tools,
+                                                          self.template_kwargs, lengths, self._pad_id())
+        joined = self.dialect.join_continuation(self.last_sampled, wrapper)
+        shift = len(wrapper) - len(joined)                                            # a dropped leading end-of-turn token shifts the spans
+        return joined, [(start - shift, end - shift) for start, end in spans]
+
+    def _append_step(self, step: TrajectoryStep, messages: list[dict], spans: list[tuple[int, int]], rows: list[torch.Tensor]) -> None:
+        offset = len(self.prefix)
+        assert [int(r.shape[0]) for r in rows] == [end - start for start, end in spans], "placeholder runs differ from the encoded rows"
+        step.ac_spans = [{"start": start, "length": end - start} for start, end in spans]
+        self.messages.extend(messages)
+        self.prefix += step.token_ids
+        self.spans += [(offset + start, offset + end) for start, end in spans]
+        self.rows += rows
+        self.run_results.trajectory.append(asdict(step))
+        self._after_append()
+
+    def _after_append(self) -> None:
+        """Budgets that depend on the prefix length: the absolute cap ends the run, the soft delta makes compaction due."""
+        config, results = self.agent_config, self.run_results
+        if len(self.prefix) >= config.absolute_trajectory_cap and not results.finish_reason:
+            results.finish_reason = "trajectory_cap"
+        if len(self.prefix) - self.segment_start >= config.compaction_threshold_tokens:
+            self.compaction_due = True
+
+    def prepare_request(self) -> tuple[list[int], torch.Tensor | None, list[bool] | None]:
+        """The next engine request: the prefix, and with parts the full-length tensor (zero rows at token positions) plus its mask."""
+        if not self.spans:
+            return list(self.prefix), None, None
+        embeds = torch.zeros(len(self.prefix), self.loaded_model.model_config.model_description.d_model, dtype=self.row_dtype)
+        mask = [True] * len(self.prefix)
+        for (start, end), rows in zip(self.spans, self.rows):
+            embeds[start:end] = rows
+            mask[start:end] = [False] * (end - start)
+        return list(self.prefix), embeds, mask
+
+    def _submit(self):
+        config = self.agent_config
+        prefix, embeds, mask = self.prepare_request()
+        return self.loaded_model.engine_submit_tokens(
+            prefix, seed=self.seed + self.run_results.num_turns, agent_id=self.agent_id, lora_name=config.lora_name,
+            chat_kwargs=config.call_kwargs, record_sampling=config.record_sampling, prompt_embeds=embeds, prompt_is_token_ids=mask,
+        )
+
+    def _check_engine_context(self) -> None:
+        """The cap plus one turn's output must fit the engine: otherwise the cap could never be the binding budget."""
+        engine_kwargs = self.loaded_model.model_config.engine_kwargs or {}
+        max_len = engine_kwargs.get("max_model_len")
+        if max_len is None:
+            return
+        max_tokens = self.loaded_model._merged_chat_kwargs(self.agent_config.call_kwargs)[0].max_tokens or 0
+        assert self.agent_config.absolute_trajectory_cap + max_tokens <= max_len, (
+            f"absolute_trajectory_cap {self.agent_config.absolute_trajectory_cap} + max_tokens {max_tokens} exceeds max_model_len {max_len}")
```

The rewritten `run()` (shown in M3 with the protocol) calls `begin()`, `_check_engine_context()`, `_submit()`, `record_assistant_turn()`, `append_user_message()` and `append_tool_results()`; `_continuation()` replaces the old helper and `_count_ac_inputs()` is removed.

</details>

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">M3 — Compaction tool and protocol</span>
    <span class="card-oneliner">Soft threshold, refusals, lone `compact(summary)`, the nested tree, segment restart.</span>
    <span class="card-badge">Review</span>
  </summary>
#### What landed

`CompactionTool` (`compact(summary)`) is a default tool; `Agent.compact()` records the finished segment (trajectory, prompt, spans, AC identity, `num_turns`) under `compactions` with `finish_reason="compacted"` and begins `system + first user messages + user([tree, instructions + summary])`, the tree nesting the segment's first user message verbatim plus its messages through the compact call (text-only: `user("The conversation so far was compacted..." + summary)`). The loop honors only a lone `compact` once due, refuses other calls with `Refused <name>: <demand>` results (not errors), nudges call-less turns with the demand, and ends after `MAX_COMPACTION_REFUSALS = 5` with `compaction_refused`.

#### Drifts, challenges, and unplanned steps

The compact call's own turn has no tool step (the segment restarts instead), so a segment with n python turns and a refusal records 2n + 2 + 1 steps; the plan's test expected one more and was corrected. `COMPACTION_INSTRUCTIONS` moved to `activation/common/ac_parts.py` (re-exported by the study module).

#### Focused actual diffs

`activation/agent/agent.py · compact()`

```python
        record = AgentRunResult(
            agent_config=config, seed=self.seed, source=results.source, lora_name=results.lora_name, finish_reason="compacted",
            trajectory=results.trajectory, prompt_token_ids=results.prompt_token_ids, prompt_messages=results.prompt_messages,
            prompt_ac_spans=results.prompt_ac_spans, ac_model_name=results.ac_model_name, ac_model_version=results.ac_model_version,
            num_turns=sum(1 for step in results.trajectory if step["role"] == "assistant"),
        )
        results.compactions.append(record)
        results.trajectory = []
        text = COMPACTION_INSTRUCTIONS + (f"\n\nSummary written before compaction:\n{summary}" if summary.strip() else "")
        if self.ac_model is not None:
            tree = ac_part(deepcopy([inner] + segment), config.ac_model_name, config.ac_compaction_ratio, kind="compaction")
            after = {"role": "user", "content": [tree, {"type": "text", "text": "\n\n" + text}]}
        else:
            after = {"role": "user", "content": "The conversation so far was compacted.\n\n" + text}
        system = [message for message in self.messages if message.get("role") == "system"]
        self._begin_segment(system + self.first_user_messages() + [after])
```

#### Validation

`IB/TMP/SLICE3B/cpu_checks/channels_cpu.py` (tiny Qwen3.5 fixture as side and target, no engine, no sandbox) passes in ~9 s: compaction after three long outputs, a refused call, a lone `compact` restarting the segment (9 recorded steps, tree of 10 messages, 64 rows at ratio 1/10 written over a pad run, bf16 rows), a second compaction nesting the first tree verbatim, `unroll` giving 3 segments, the subagent in step mode (child prompt `[text, part, text]`, parent tool step `[part, text]`), a 100k-char output truncated to <21k with an 80k-char part nested with the parent segment, search with 3 visible passages and a 6-passage part, the serialize/deserialize/resume round trip with identical prefix, spans and rows, the text-only compaction fallback, and the five-refusal cap ending the run with `compaction_refused`. Node rollouts with a 300-token delta (with AC rows and text-only): `uv run pytest activation/tests/test_basic_agent_ac.py --gpu --slow -s` on `ac-slice3b-temp` (RTX PRO 6000, Qwen3.5-4B target with the fp8 KV cache, Qwen3.5-0.8B side, untrained AC rows), logs under `IB/TMP/SLICE3B/node/AGENT_AC_TEST/`. Simulated channels (`pytest_simulated.log`, 71 s): compaction prompt 1,259 tokens with rows at 391 positions, repeat cached 1,056; second compaction 910 tokens / 53 rows; subagent 944 tokens / 4 rows; a real 100k-character python output in the sandbox: 3,620 tokens / 251 rows, repeat cached 3,168; BRIGHT search: 2,161 tokens / 117 rows, repeat cached 2,112; every request accepted with the full-length bf16 tensor and the model answered sensibly on untrained rows (a python tool call, a search follow-up). With-AC rollouts (`pytest_rerun.log`, threshold 300 tokens): seed 0 submitted after 8 turns, 2 compactions, 2 parts / 223 rows, 4,224 cached prompt tokens on its last request; seed 1 after 4 turns, 1 compaction, 60 rows, 1,056 cached; both answered 2441 (the correct sum); the harvest produced 6 items from 12 candidates (6 turns before any part skipped). Text-only rollouts (`pytest.log`): both seeds submitted 2441 after 4 turns with 1 compaction each. Final clean run of the whole file after the test fixes (`pytest_final.log`): 3 passed in 170 s; in that run the with-AC seed 0 hit `max_turns` after 12 turns with 3 compactions and 355 rows (untrained rows: the model kept re-deriving the primes) while seed 1 submitted 2441 after one compaction, and the harvest yielded 10 items from 16 candidates.

---

*Planned changes (reference):*


#### Planning Overview

The compaction tool and the protocol from the rules artifact. `compact(summary)` is a default tool; the loop honors it only as a lone call once compaction is due, refuses everything else with a tool result naming `compact`, nudges call-less turns with the demand, and ends after five refused turns. `Agent.compact()` moves the finished segment into `compactions` and begins a new segment `system, user(first messages), user([tree, instructions + summary])`; the tree nests the finished segment's first user message verbatim followed by its messages through the `compact` call. Without an AC model the second user message carries the text only. `COMPACTION_INSTRUCTIONS` moves to `ac_model_utils.py` so both the study generator and the agent import it without pulling the dataset modules.

#### Planned Changes

`activation/agent/agent_tools.py · CompactionTool`

```diff
@@ activation/agent/agent_tools.py — CompactionTool @@
 class CompactionTool(AgentTool):
-    """
-    Explicit compaction tool.
-    ... (design notes) ...
-    """
-    def __init__(self, harness, agent, base_config: "AgentConfig"):
-        pass
+    """
+    Compacts the conversation into a new segment. With an AC model the finished segment becomes the recursive
+    tree part of the new prompt and the summary follows it as text; without one only the summary carries over.
+    The finished segment is recorded under the run's `compactions` (finish_reason "compacted").
+    """
+    name = "compact"
+    description = ("Compact the conversation so far: write a summary of what was done, what was learned and what remains, "
+                   "then continue from it. Call it alone, without other tools, when asked to compact.")
+    parameters = {
+        "type": "object",
+        "properties": {"summary": {"type": "string", "description": "A complete summary of the work so far for your future self."}},
+        "required": ["summary"],
+    }
+
+    def execute(self, summary: str = "") -> ToolCallResult:
+        self.agent.compact(str(summary or ""))
+        return ToolCallResult(output="Context compacted.", compacted=True)

 DEFAULT_TOOLS: dict[str, tuple[type[AgentTool], dict]] = {
     ShellTool.name: (ShellTool, {}),
     PythonTool.name: (PythonTool, {}),
     ParallelCallTool.name: (ParallelCallTool, {}),
     SubmitAnswerTool.name: (SubmitAnswerTool, {}),
-    CompactionTool.name: (CompactionTool, {})
+    CompactionTool.name: (CompactionTool, {}),
 }
```

`activation/agent/agent.py · Agent.compact(), the protocol helpers, run()`

```diff
@@ activation/agent/agent.py — compaction @@
+    COMPACTION_DEMAND = "The context threshold is reached: call compact(summary) now, alone, before any other tool."
+
+    def compact(self, summary: str) -> None:
+        """Close the current segment into run_results.compactions and start the next one from its tree (or the summary alone)."""
+        config, results = self.agent_config, self.run_results
+        first_user = next(i for i, message in enumerate(self.messages) if message.get("role") == "user")
+        inner, segment = self.messages[first_user], self.messages[first_user + 1:]          # verbatim; through the compact call
+        record = AgentRunResult(
+            agent_config=config, seed=self.seed, source=results.source, lora_name=results.lora_name, finish_reason="compacted",
+            trajectory=results.trajectory, prompt_token_ids=results.prompt_token_ids, prompt_messages=results.prompt_messages,
+            prompt_ac_spans=results.prompt_ac_spans, ac_model_name=results.ac_model_name, ac_model_version=results.ac_model_version,
+            num_turns=sum(1 for step in results.trajectory if step["role"] == "assistant"),
+        )
+        results.compactions.append(record)
+        results.trajectory = []
+        text = COMPACTION_INSTRUCTIONS + (f"\n\nSummary written before compaction:\n{summary}" if summary.strip() else "")
+        if self.ac_model is not None:
+            tree = ac_part(deepcopy([inner] + segment), config.ac_model_name, config.ac_compaction_ratio)
+            after = {"role": "user", "content": [tree, {"type": "text", "text": "\n\n" + text}]}
+        else:
+            after = {"role": "user", "content": "The conversation so far was compacted.\n\n" + text}
+        system = [message for message in self.messages if message.get("role") == "system"]
+        self._begin_segment(system + self.first_user_messages() + [after])
+        self.compaction_due, self.compaction_refusals = False, 0
+
+    @staticmethod
+    def _is_lone_compact(calls: list[dict]) -> bool:
+        return len(calls) == 1 and calls[0]["name"] == CompactionTool.name
+
+    def _refusal(self, call: dict) -> ToolCallResult:
+        return ToolCallResult(output=f"Refused {call['name']}: {self.COMPACTION_DEMAND}")
+
+    def _refused(self) -> bool:
+        """Counts a turn that ignored a due compaction; True when the run must end."""
+        self.compaction_refusals += 1
+        if self.compaction_refusals >= MAX_COMPACTION_REFUSALS:
+            self.run_results.finish_reason = "compaction_refused"
+            return True
+        return False
@@ activation/agent/agent.py — run() @@
     def run(self) -> AgentRunResult:
-        """Simple in/out run. ... Compaction is for later."""
+        """Simple in/out run. Never raises for a model or tool failure ... Compaction restarts the segment in place."""
         config, results = self.agent_config, self.run_results
         start = time.time()
-        self.dialect = ModelDialect.for_tokenizer(self.loaded_model.tokenizer)
-        ... (prompt setup, _count_ac_inputs) ...
+        self.begin()
         if self.reporter is not None:
             self.reporter.report_agent_start(self)
-        no_tool_turns = 0
-        errors = 0
         try:
-            loaded_model = self.loaded_model
-            ... (templating) ...
+            self._check_engine_context()
             while True:
                 if results.num_turns >= config.max_turns:
                     results.finish_reason = "max_turns"
                     break
                 if time.time() - start > config.max_duration:
                     results.finish_reason = "max_duration"
                     break
-                output = loaded_model.engine_submit_tokens(self.prefix, ...)
+                output = self._submit()
                 if results.num_turns == 0:
                     results.lora_name = output.lora_name
                 results.num_turns += 1
                 ... (token counters unchanged) ...
-                content, calls = self.dialect.parse(output.text, parameter_types)
-                messages_before = list(self.messages)
-                self.messages.append(rendering.assistant_message(content, calls))
-                sampled = list(output.token_ids)
-                step = TrajectoryStep(role="assistant", ...)
-                self.prefix += sampled
+                content, calls = self.dialect.parse(output.text, self.parameter_types)
+                self.record_assistant_turn(content, calls, list(output.token_ids), list(output.logprobs or []))
                 if not calls:
-                    ... (nudge via _continuation) ...
+                    self.no_tool_turns += 1
+                    if self.no_tool_turns >= MAX_CONSECUTIVE_NO_TOOL_TURNS:
+                        results.finish_reason = "no_tool_call"
+                        break
+                    self.append_user_message(self.COMPACTION_DEMAND if self.compaction_due else self.dialect.nudge_message)
+                    if (self.compaction_due and self._refused()) or results.finish_reason:
+                        break
+                    self._report_step()
                     continue
-                no_tool_turns = 0
-                call_results = self._execute_tool_calls(calls)
-                errors += sum(1 for result in call_results if result.is_error)
-                tool_messages = rendering.tool_messages(calls, call_results)
-                wrapper = self._continuation(...)
-                tool_step = TrajectoryStep(role="tool", ..., activations={...}, token_ids=wrapper)
-                results.trajectory.extend([asdict(step), asdict(tool_step)])
-                self.messages.extend(tool_messages)
-                self.prefix += wrapper
-                self._report_step()
+                self.no_tool_turns = 0
+                if self.compaction_due and not self._is_lone_compact(calls):
+                    self.append_tool_results(calls, [self._refusal(call) for call in calls])     # nothing runs until compact
+                    if self._refused() or results.finish_reason:
+                        break
+                    self._report_step()
+                    continue
+                call_results = self._execute_tool_calls(calls)
+                self.errors += sum(1 for result in call_results if result.is_error)
+                if any(result.compacted for result in call_results):
+                    self._report_step()                                                          # compact() began the new segment; no tool message follows
+                    continue
+                self.append_tool_results(calls, call_results)
+                self._report_step()
                 if any(result.is_final for result in call_results):
                     results.finish_reason = "submitted"
                     break
-                if errors > config.max_tool_errors:
+                if self.errors > config.max_tool_errors:
                     results.finish_reason = "max_tool_errors"
                     break
+                if results.finish_reason:                                                        # trajectory_cap, set by the append
+                    break
         except Exception as error:
             ... (unchanged) ...
```

The crossing turn's calls still run (the threshold is checked after their results are appended), the refusal results are not errors and do not count toward `max_tool_errors`, and `num_turns`, `max_duration` and `max_tool_errors` keep counting across segments.

</details>

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">M4 — Subagent exchange</span>
    <span class="card-oneliner">Parent segment into the child's prompt, child segment into the parent's result, step mode.</span>
    <span class="card-badge">Review</span>
  </summary>
#### What landed

`SubagentTool.execute` copies the AC fields to the child config, builds `messages_input = [user([intro, part(parent segment, kind subagent_prompt), ...])]` (the child appends the task), runs the child (or `simulated_run()` in step mode: prompt built, dummy answer `Agent <id> simulated run`, finish reason `simulated`), and returns `[part(child final segment, kind subagent_return), text(answer line)]`; `segment_messages_of(result)` is the child's prompt messages without the system message plus every step's messages, so its compaction tree rides inside.

#### Drifts, challenges, and unplanned steps

In step mode the child receives the parent's existing env object or none (`parent._env`), never `parent.agent_env`: the CPU check found the child starting a sandbox. Both directions require `enable_ac_communication` and an AC model on the parent.

#### Focused actual diffs

`activation/agent/agent_tools.py · SubagentTool.execute()`

```diff
@@ activation/agent/agent_tools.py — SubagentTool.execute() @@
-        share_ac = parent.agent_config.enable_ac_communication
-        if share_ac:
-            subagent_config.ac_inputs = dict(subagent_config.ac_inputs) | {"caller_trajectory": deepcopy(parent.run_results.trajectory)}
+        share_ac = parent_config.enable_ac_communication and parent.ac_model is not None
+        if share_ac:
+            for name in self.SHARED_FIELDS:
+                setattr(subagent_config, name, getattr(parent_config, name))
+            subagent_config.messages_input = [{"role": "user", "content": [
+                {"type": "text", "text": SUBAGENT_INTRO},
+                ac_part(deepcopy(parent.segment_messages()), parent_config.ac_model_name, parent_config.ac_subagent_ratio, kind="subagent_prompt"),
+            ]}]
 ...
-        result = subagent.run()
+        subagent.step_mode = parent.step_mode
+        result = subagent.simulated_run() if parent.step_mode else subagent.run()
 ...
-        ac_outputs = {"trajectory": result.trajectory} if share_ac else {}
-        return ToolCallResult(output=output, is_error=result.finish_reason == "error", ac_outputs=ac_outputs)
+        content = None
+        if share_ac:
+            final_segment = segment_messages_of(result)
+            content = [ac_part(final_segment, parent_config.ac_model_name, parent_config.ac_subagent_ratio, kind="subagent_return"),
+                       {"type": "text", "text": output}]
+        return ToolCallResult(output=output, is_error=result.finish_reason == "error", content=content)
```

#### Validation

`IB/TMP/SLICE3B/cpu_checks/channels_cpu.py` (tiny Qwen3.5 fixture as side and target, no engine, no sandbox) passes in ~9 s: compaction after three long outputs, a refused call, a lone `compact` restarting the segment (9 recorded steps, tree of 10 messages, 64 rows at ratio 1/10 written over a pad run, bf16 rows), a second compaction nesting the first tree verbatim, `unroll` giving 3 segments, the subagent in step mode (child prompt `[text, part, text]`, parent tool step `[part, text]`), a 100k-char output truncated to <21k with an 80k-char part nested with the parent segment, search with 3 visible passages and a 6-passage part, the serialize/deserialize/resume round trip with identical prefix, spans and rows, the text-only compaction fallback, and the five-refusal cap ending the run with `compaction_refused`. Node: `uv run pytest activation/tests/test_basic_agent_ac.py --gpu --slow -s` on `ac-slice3b-temp` (RTX PRO 6000, Qwen3.5-4B target with the fp8 KV cache, Qwen3.5-0.8B side, untrained AC rows), logs under `IB/TMP/SLICE3B/node/AGENT_AC_TEST/`. Simulated channels (`pytest_simulated.log`, 71 s): compaction prompt 1,259 tokens with rows at 391 positions, repeat cached 1,056; second compaction 910 tokens / 53 rows; subagent 944 tokens / 4 rows; a real 100k-character python output in the sandbox: 3,620 tokens / 251 rows, repeat cached 3,168; BRIGHT search: 2,161 tokens / 117 rows, repeat cached 2,112; every request accepted with the full-length bf16 tensor and the model answered sensibly on untrained rows (a python tool call, a search follow-up). With-AC rollouts (`pytest_rerun.log`, threshold 300 tokens): seed 0 submitted after 8 turns, 2 compactions, 2 parts / 223 rows, 4,224 cached prompt tokens on its last request; seed 1 after 4 turns, 1 compaction, 60 rows, 1,056 cached; both answered 2441 (the correct sum); the harvest produced 6 items from 12 candidates (6 turns before any part skipped). Text-only rollouts (`pytest.log`): both seeds submitted 2441 after 4 turns with 1 compaction each. Final clean run of the whole file after the test fixes (`pytest_final.log`): 3 passed in 170 s; in that run the with-AC seed 0 hit `max_turns` after 12 turns with 3 compactions and 355 rows (untrained rows: the model kept re-deriving the primes) while seed 1 submitted 2441 after one compaction, and the harvest yielded 10 items from 16 candidates.

---

*Planned changes (reference):*


#### Planning Overview

Both directions of the subagent exchange through `messages_input` and tool-result content. The child's prompt is one user message: intro text, the parent's current segment as a part at the subagent ratio, and the task appended by the child itself. The child's return is `[part(child final segment), text(answer line)]`; the child's final segment starts with its own first user messages, so any compaction tree it built rides inside the part. Step mode creates the child, gives it a dummy submitted answer and does not run it. Both directions need `enable_ac_communication` and an AC model; otherwise the exchange is text-only as today. The child inherits `ac_model_name` and the ratios from the parent's config so nested behavior is uniform.

#### Planned Changes

`activation/agent/agent_tools.py · SubagentTool.execute()`

```diff
@@ activation/agent/agent_tools.py — SubagentTool @@
+SUBAGENT_INTRO = "A parent agent spawned you. Its conversation so far is provided as activation context; your task follows.\n"
+
 class SubagentTool(AgentTool):
     def execute(self, task: str) -> ToolCallResult:
         from .agent import Agent
         parent = self.agent
+        parent_config = parent.agent_config
         subagent_config = deepcopy(self.base_config)
         if subagent_config.agent_name == "main":
             subagent_config.agent_name = "general_subagent"
         subagent_config.user_prompt = task
-        share_ac = parent.agent_config.enable_ac_communication
-        if share_ac:
-            subagent_config.ac_inputs = dict(subagent_config.ac_inputs) | {"caller_trajectory": deepcopy(parent.run_results.trajectory)}
+        share_ac = parent_config.enable_ac_communication and parent.ac_model is not None
+        if share_ac:
+            for name in ("ac_model_name", "enable_ac_communication", "ac_compaction_ratio", "ac_subagent_ratio", "ac_tool_output_ratio", "ac_search_ratio"):
+                setattr(subagent_config, name, getattr(parent_config, name))
+            subagent_config.messages_input = [{"role": "user", "content": [
+                {"type": "text", "text": SUBAGENT_INTRO},
+                parent.context_part(parent_config.ac_subagent_ratio),          # the parent's segment up to and including this delegating turn
+            ]}]                                                                # the child appends the task as the trailing text part
         subagent = Agent(harness=self.harness, agent_config=subagent_config, agent_env=parent.agent_env, parent_agent=parent,
                          reporter=parent.reporter, seed=parent.seed)
-        result = subagent.run()
+        subagent.step_mode = parent.step_mode
+        result = subagent.simulated_run() if parent.step_mode else subagent.run()
         with parent.lock:
             parent.run_results.subagent_results.append(result)
         output = (f"Subagent finished ({result.finish_reason}, {result.num_turns} turns). "
                   f"Answer: {result.answer if result.answer is not None else '(none)'}")
-        ac_outputs = {"trajectory": result.trajectory} if share_ac else {}
-        return ToolCallResult(output=output, is_error=result.finish_reason == "error", ac_outputs=ac_outputs)
+        content = None
+        if share_ac:
+            final_segment = segment_messages_of(result)                         # its first user messages (tree inside, if any) + every step
+            content = [ac_part(final_segment, parent_config.ac_model_name, parent_config.ac_subagent_ratio), {"type": "text", "text": output}]
+        return ToolCallResult(output=output, is_error=result.finish_reason == "error", content=content)
+
+
+def segment_messages_of(result: "AgentRunResult") -> list[dict]:
+    """A run's final segment as dialect messages without the system message: prompt messages, then each step's messages."""
+    messages = [message for message in result.prompt_messages if message.get("role") != "system"]
+    for step in result.trajectory:
+        messages.extend(step.get("messages") or [])
+    return deepcopy(messages)
```

`activation/agent/agent.py · Agent.simulated_run()`

```diff
@@ activation/agent/agent.py — simulated_run() @@
+    def simulated_run(self) -> AgentRunResult:
+        """Step mode: the prompt is built (parts encoded), nothing is sampled, the answer is a dummy submission."""
+        results = self.run_results
+        self.begin()
+        results.answer = f"Agent {self.agent_id[:8]} simulated run"
+        results.finish_reason = "simulated"
+        self.finished = True
+        return results
```

</details>

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">M5 — Large tool output and semantic search parts</span>
    <span class="card-oneliner">Full outputs and extra passages as parts, nested with the parent segment.</span>
    <span class="card-badge">Review</span>
  </summary>
#### What landed

`truncate_output(..., agent=)` returns the visible head/tail and, with an AC model, a `tool_output` part `[user[parent_context part], tool(full text, 80k cap)]`; `Agent._finalize_result` applies it to any tool's result (prepends the part, swaps the one text part for the visible text). `SemanticSearchTool` fetches `top_k + min(20, 2·top_k)` and returns `[text(top_k), part(extra, kind search)]`. `ParallelCallTool` concatenates each call's content in call order. `MessageRendering.tool_messages` emits content lists; `InlineRendering` wraps them in `<tool_response>` text parts.

#### Drifts, challenges, and unplanned steps

Content lists carry the invariant "exactly one text part equals `output`", which is what `_finalize_result` relies on to swap in the truncated text. Every nested `parent_context` part deep-copies the parent's current segment: rows are deduplicated by the row cache (identical messages hit the same key) but the JSON record grows quadratically with the number of parts in a segment; flagged for the next slice (a by-reference serialization would fix it).

#### Focused actual diffs

`activation/agent/agent.py · _finalize_result()`

```python
    def _finalize_result(self, call: dict, result: ToolCallResult) -> ToolCallResult:
        visible, part = truncate_output(self._env, result.output, call["id"], agent=self if not result.compacted else None)
        if visible != result.output:
            content = [dict(item) for item in tool_content(result)]
            for item in content:
                if item.get("type") == "text" and item.get("text") == result.output:
                    item["text"] = visible
            result.content = ([part] if part is not None else []) + content
            result.output = visible
        return result
```

`activation/agent/agent_tools.py · SemanticSearchTool.execute()`

```diff
@@ activation/agent/agent_tools.py — SemanticSearchTool.execute() @@
-        chunks = index.bm25_query_many_frozen([query], top_k=int(top_k or self.top_k))[0]
+        k = int(top_k or self.top_k)
+        config = self.agent.agent_config
+        extra = min(SEARCH_EXTRA_MAX, 2 * k) if self.agent.ac_model is not None else 0
+        chunks = index.bm25_query_many_frozen([query], top_k=k + extra)[0]
 ...
+        if chunks[k:]:
+            part = ac_part([{"role": "user", "content": [self.agent.context_part(config.ac_subagent_ratio)]},
+                            {"role": "tool", "content": blocks(chunks[k:])}], config.ac_model_name, config.ac_search_ratio, kind="search")
+            content = [{"type": "text", "text": output}, part]
+        return ToolCallResult(output=output, content=content)
```

#### Validation

`IB/TMP/SLICE3B/cpu_checks/channels_cpu.py` (tiny Qwen3.5 fixture as side and target, no engine, no sandbox) passes in ~9 s: compaction after three long outputs, a refused call, a lone `compact` restarting the segment (9 recorded steps, tree of 10 messages, 64 rows at ratio 1/10 written over a pad run, bf16 rows), a second compaction nesting the first tree verbatim, `unroll` giving 3 segments, the subagent in step mode (child prompt `[text, part, text]`, parent tool step `[part, text]`), a 100k-char output truncated to <21k with an 80k-char part nested with the parent segment, search with 3 visible passages and a 6-passage part, the serialize/deserialize/resume round trip with identical prefix, spans and rows, the text-only compaction fallback, and the five-refusal cap ending the run with `compaction_refused`. Node (a real 100k-char python output in the sandbox; BRIGHT search): `uv run pytest activation/tests/test_basic_agent_ac.py --gpu --slow -s` on `ac-slice3b-temp` (RTX PRO 6000, Qwen3.5-4B target with the fp8 KV cache, Qwen3.5-0.8B side, untrained AC rows), logs under `IB/TMP/SLICE3B/node/AGENT_AC_TEST/`. Simulated channels (`pytest_simulated.log`, 71 s): compaction prompt 1,259 tokens with rows at 391 positions, repeat cached 1,056; second compaction 910 tokens / 53 rows; subagent 944 tokens / 4 rows; a real 100k-character python output in the sandbox: 3,620 tokens / 251 rows, repeat cached 3,168; BRIGHT search: 2,161 tokens / 117 rows, repeat cached 2,112; every request accepted with the full-length bf16 tensor and the model answered sensibly on untrained rows (a python tool call, a search follow-up). With-AC rollouts (`pytest_rerun.log`, threshold 300 tokens): seed 0 submitted after 8 turns, 2 compactions, 2 parts / 223 rows, 4,224 cached prompt tokens on its last request; seed 1 after 4 turns, 1 compaction, 60 rows, 1,056 cached; both answered 2441 (the correct sum); the harvest produced 6 items from 12 candidates (6 turns before any part skipped). Text-only rollouts (`pytest.log`): both seeds submitted 2441 after 4 turns with 1 compaction each. Final clean run of the whole file after the test fixes (`pytest_final.log`): 3 passed in 170 s; in that run the with-AC seed 0 hit `max_turns` after 12 turns with 3 compactions and 355 rows (untrained rows: the model kept re-deriving the primes) while seed 1 submitted 2441 after one compaction, and the harvest yielded 10 items from 16 candidates.

---

*Planned changes (reference):*


#### Planning Overview

Large tool outputs and semantic search as parts. Truncation keeps today's visible head/tail and env file, and with an AC model adds a part holding the retained full output (80k-char cap) nested with the parent's current segment; the result content is `[part, text]` (decision 1). Search fetches `top_k + min(20, 2·top_k)`, shows `top_k`, and sends the rest as a part after the text, nested the same way. The parallel tool concatenates each call's content in call order. Tool messages carry content lists; the template flattens them with sentinels (M2).

#### Planned Changes

`activation/agent/agent_tools.py · truncate_output(), Agent.execute_tool_call()`

```diff
@@ activation/agent/agent_tools.py — truncate_output() @@
-def truncate_output(env, text, call_id, limit=TOOL_OUTPUT_LIMIT_CHARS) -> tuple[str, dict]:
-    """Head + tail of a long output; the full text goes to a file in the env and out as activation context."""
+def truncate_output(env, text, call_id, limit=TOOL_OUTPUT_LIMIT_CHARS, agent: "Agent | None" = None) -> tuple[str, dict | None]:
+    """Head + tail of a long output; the full text goes to a file in the env and, with an AC model, into a part (nested with the parent segment)."""
     if len(text) <= limit:
-        return text, {}
+        return text, None
     if len(text) > AC_OUTPUT_LIMIT_CHARS:
         keep = AC_OUTPUT_LIMIT_CHARS // 2
         text = f"{text[:keep]}\n... [{len(text) - AC_OUTPUT_LIMIT_CHARS} chars dropped] ...\n{text[-keep:]}"
     path = f"{TOOL_OUTPUT_DIR}/{call_id}.txt"
     written = env.write_file(path, text) if env is not None else False
     half = limit // 2
     note = f"full output written to {path}" if written else "full output kept as activation context"
     truncated = f"{text[:half]}\n... [truncated {len(text) - limit} chars; {note}] ...\n{text[-half:]}"
-    return truncated, {"full_output": text}
+    if agent is None or agent.ac_model is None:
+        return truncated, None
+    config = agent.agent_config
+    part = ac_part([{"role": "user", "content": [agent.context_part(config.ac_subagent_ratio)]}, {"role": "tool", "content": text}],
+                   config.ac_model_name, config.ac_tool_output_ratio)
+    return truncated, part
@@ activation/agent/agent.py — execute_tool_call() @@
-        output, ac_outputs = truncate_output(self._env if self._env is not None else None, result.output, call["id"])
-        result.output = output
-        if ac_outputs:
-            result.ac_outputs = dict(result.ac_outputs) | ac_outputs
-        return result
+        return self._finalize_result(call, result)
+
+    def _finalize_result(self, call: dict, result: ToolCallResult) -> ToolCallResult:
+        """Truncation and its part, applied to any tool's result (the one text part of a content list equals the visible output)."""
+        visible, part = truncate_output(self._env, result.output, call["id"], agent=self if not result.compacted else None)
+        if visible != result.output:
+            content = tool_content(result)
+            for item in content:
+                if item.get("type") == "text" and item.get("text") == result.output:
+                    item["text"] = visible
+            result.content = ([part] if part is not None else []) + content
+            result.output = visible
+        return result
```

`activation/agent/agent_tools.py · SemanticSearchTool.execute(), ParallelCallTool.execute()`

```diff
@@ activation/agent/agent_tools.py — SemanticSearchTool.execute() @@
         index = self.harness.dataset_manager._get_or_create_index(self.dataset_id)
         index.build_bm25_index()
-        chunks = index.bm25_query_many_frozen([query], top_k=int(top_k or self.top_k))[0]
+        k = int(top_k or self.top_k)
+        config = self.agent.agent_config
+        extra = min(20, 2 * k) if self.agent.ac_model is not None else 0                 # the model sees top_k; the next `extra` ride as rows
+        chunks = index.bm25_query_many_frozen([query], top_k=k + extra)[0]
         if not chunks:
             return ToolCallResult(output="No passage matched the query.")
-        blocks = [f"[{chunk.chunk_id}]\n{index.get_chunk_section(chunk)[:self.max_chars]}" for chunk in chunks]
-        return ToolCallResult(output="\n\n".join(blocks), ac_outputs={"chunk_ids": [chunk.chunk_id for chunk in chunks]})
+        def blocks(subset):
+            return "\n\n".join(f"[{chunk.chunk_id}]\n{index.get_chunk_section(chunk)[:self.max_chars]}" for chunk in subset)
+        output = blocks(chunks[:k])
+        content = None
+        if chunks[k:]:
+            part = ac_part([{"role": "user", "content": [self.agent.context_part(config.ac_subagent_ratio)]},
+                            {"role": "tool", "content": blocks(chunks[k:])}], config.ac_model_name, config.ac_search_ratio)
+            content = [{"type": "text", "text": output}, part]                                # text first, the extra passages after
+        return ToolCallResult(output=output, content=content)
@@ activation/agent/agent_tools.py — ParallelCallTool.execute() @@
-        outputs = [f"[{call['name']}] {result.output}" for call, result in zip(nested, results)]
-        ac_outputs = {call["id"]: result.ac_outputs for call, result in zip(nested, results) if result.ac_outputs}
-        return ToolCallResult(
-            output="\n\n".join(outputs), is_error=any(result.is_error for result in results),
-            ac_outputs=ac_outputs, is_final=any(result.is_final for result in results),
-        )
+        outputs = [f"[{call['name']}] {result.output}" for call, result in zip(nested, results)]
+        content: list[dict] = []
+        for call, result, labeled in zip(nested, results, outputs):
+            for item in tool_content(result):                                            # each call's parts in its own order
+                content.append({"type": "text", "text": labeled + "\n\n"} if item.get("type") == "text" else item)
+        has_parts = any(result.content is not None for result in results)
+        return ToolCallResult(
+            output="\n\n".join(outputs), is_error=any(result.is_error for result in results),
+            is_final=any(result.is_final for result in results), compacted=any(result.compacted for result in results),
+            content=content if has_parts else None,
+        )
```

`activation/agent/agent_utils.py · MessageRendering.tool_messages(), InlineRendering.tool_messages()`

```diff
@@ activation/agent/agent_utils.py — tool messages carry content lists @@
     def tool_messages(self, calls, results) -> list[dict]:
-        return [{"role": "tool", "content": result.output} for result in results]
+        return [{"role": "tool", "content": tool_content(result)} for result in results]
 ...
     def tool_messages(self, calls, results) -> list[dict]:               # InlineRendering
-        body = "\n".join(f"<tool_response>\n{result.output}\n</tool_response>" for result in results)
-        return [{"role": "user", "content": body}]
+        content: list[dict] = []
+        for result in results:
+            content += [{"type": "text", "text": "<tool_response>\n"}, *tool_content(result), {"type": "text", "text": "\n</tool_response>\n"}]
+        return [{"role": "user", "content": content}]
```

The parallel tool's visible text keeps the `[name] output` blocks; with parts it becomes the one text part of each call's content. A single tool exceeding the limit inside a parallel call gets its truncation part from `_finalize_result` on the nested call, before the concatenation.

</details>

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">M6 — Simulation utilities, the main-tree test, node validation</span>
    <span class="card-oneliner">`resume_from_run_result`, synthetic turns, `simulate_step`, the full test file.</span>
    <span class="card-badge">Review</span>
  </summary>
#### What landed

`SyntheticTurn` / `synthesize_agent` in `agent_utils.py` (runtime imports inside the function avoid the agent → utils cycle); `Agent.simulate_step()` executes the pending calls (refusals when due, compaction, subagents in step mode, real tools otherwise), appends, prepares the request and returns the inspection record; `Agent.submit_probe()` submits it once without recording; `Agent.resume_from_run_result()` rebuilds messages, prefix, spans and re-encoded rows from a record; `activation/tests/test_basic_agent_ac.py` with the three node tests.

#### Drifts, challenges, and unplanned steps

`simulate_step` calls `begin()` itself (idempotent through `started`). The test's rollout thresholds went from 1,500 to 300 tokens (a primes run is a few hundred tokens per turn, so 1,500 would rarely compact) and the text-only test asserts finish reasons in the allowed set plus at least one compaction instead of requiring a submission (model behavior after a compaction with untrained rows is not a mechanics assertion). The with-AC rollout test ends by harvesting items from its runs (M7). Three node-found test fixes: BRIGHT exposes labeled QA examples and no scorable tasks, so the search scenario takes the first example's query and names the dataset explicitly; the teardown frees the target and side adapters before the bases (the harness refuses to free a base carrying adapters); and the prefix-cache assertion applies only to prompts longer than one cache block (`ENGINE_CACHE_BLOCK_TOKENS = 1056`): a token-only 909-token prompt never hits on this engine either, so a shorter prompt proves nothing about the rows.

#### Focused actual diffs

`activation/agent/agent.py · simulate_step()`

```python
        last = results.trajectory[-1] if results.trajectory else None
        if last is not None and last["role"] == "assistant" and last["tool_calls"] and not results.finish_reason:
            calls = last["tool_calls"]
            if self.compaction_due and not self._is_lone_compact(calls):
                executed = [self._refusal(call) for call in calls]
                self.append_tool_results(calls, executed)
                self._refused()
            else:
                executed = self._execute_tool_calls(calls)
                if not any(result.compacted for result in executed):
                    self.append_tool_results(calls, executed)
        prefix, embeds, mask = self.prepare_request()
```

`activation/agent/agent_utils.py · synthesize_agent()`

```python
    for turn in turns:
        calls = [{"id": uuid.uuid4().hex[:8], "name": name, "arguments": dict(arguments)} for name, arguments in turn.calls]
        blocks = [agent.dialect.preferred_format.render(call["name"], call["arguments"]) for call in calls]
        text = "\n\n".join([turn.content] + blocks if turn.content else blocks)
        token_ids = list(tokenizer.encode(text + eot, add_special_tokens=False))
        agent.record_assistant_turn(turn.content, calls, token_ids, [])
        if calls and turn.outputs is not None:
            results = [agent._finalize_result(call, ToolCallResult(output=output)) for call, output in zip(calls, turn.outputs)]
            agent.append_tool_results(calls, results)
```

#### Validation

`IB/TMP/SLICE3B/cpu_checks/channels_cpu.py` (tiny Qwen3.5 fixture as side and target, no engine, no sandbox) passes in ~9 s: compaction after three long outputs, a refused call, a lone `compact` restarting the segment (9 recorded steps, tree of 10 messages, 64 rows at ratio 1/10 written over a pad run, bf16 rows), a second compaction nesting the first tree verbatim, `unroll` giving 3 segments, the subagent in step mode (child prompt `[text, part, text]`, parent tool step `[part, text]`), a 100k-char output truncated to <21k with an 80k-char part nested with the parent segment, search with 3 visible passages and a 6-passage part, the serialize/deserialize/resume round trip with identical prefix, spans and rows, the text-only compaction fallback, and the five-refusal cap ending the run with `compaction_refused`. Node test file: `uv run pytest activation/tests/test_basic_agent_ac.py --gpu --slow -s` on `ac-slice3b-temp` (RTX PRO 6000, Qwen3.5-4B target with the fp8 KV cache, Qwen3.5-0.8B side, untrained AC rows), logs under `IB/TMP/SLICE3B/node/AGENT_AC_TEST/`. Simulated channels (`pytest_simulated.log`, 71 s): compaction prompt 1,259 tokens with rows at 391 positions, repeat cached 1,056; second compaction 910 tokens / 53 rows; subagent 944 tokens / 4 rows; a real 100k-character python output in the sandbox: 3,620 tokens / 251 rows, repeat cached 3,168; BRIGHT search: 2,161 tokens / 117 rows, repeat cached 2,112; every request accepted with the full-length bf16 tensor and the model answered sensibly on untrained rows (a python tool call, a search follow-up). With-AC rollouts (`pytest_rerun.log`, threshold 300 tokens): seed 0 submitted after 8 turns, 2 compactions, 2 parts / 223 rows, 4,224 cached prompt tokens on its last request; seed 1 after 4 turns, 1 compaction, 60 rows, 1,056 cached; both answered 2441 (the correct sum); the harvest produced 6 items from 12 candidates (6 turns before any part skipped). Text-only rollouts (`pytest.log`): both seeds submitted 2441 after 4 turns with 1 compaction each. Final clean run of the whole file after the test fixes (`pytest_final.log`): 3 passed in 170 s; in that run the with-AC seed 0 hit `max_turns` after 12 turns with 3 compactions and 355 rows (untrained rows: the model kept re-deriving the primes) while seed 1 submitted 2441 after one compaction, and the harvest yielded 10 items from 16 candidates.

---

*Planned changes (reference):*


#### Planning Overview

The one-time test utilities and the main-tree test. `Agent.resume_from_run_result` rebuilds an agent from a serialized run (messages, prefix, spans, rows re-encoded). `activation/agent/agent_utils.py` gains a synthetic-run section that builds synthetic runs with the real dialect and tokenizer, turn by turn, with tool outputs given or left pending. `Agent.simulate_step()` executes the pending calls (subagents in step mode), encodes parts, prepares the next request and returns an inspection record without submitting. The test exercises every channel with untrained rows, submits one real mixed-prompt request per scenario (and a second to see the prefix cache hit), round-trips a resume, and runs one text-only compaction rollout end to end. Node: Qwen3.5-4B engine, 0.8B side model, `ac_dev` from the 3a1 checkpoint when `AC_CHECKPOINT` names it, untrained otherwise. Podman is needed for the real tools, as for the other agent tests.

#### Planned Changes

`activation/agent/agent.py · Agent.resume_from_run_result(), simulate_step()`

```diff
@@ activation/agent/agent.py — resume and simulate @@
+    @staticmethod
+    def resume_from_run_result(harness: "HarnessRuntime", run_result: AgentRunResult, reporter: "RolloutReporter | None" = None) -> "Agent":
+        """
+        An agent positioned exactly after the recorded steps of `run_result`'s current segment: messages, prefix and spans
+        from the record, rows re-encoded from the recorded parts (the record never stores rows). Counters continue.
+        """
+        agent = Agent(harness, run_result.agent_config, reporter=reporter, seed=run_result.seed)
+        agent.run_results = run_result
+        agent.dialect = ModelDialect.for_tokenizer(agent.loaded_model.tokenizer)
+        agent.tool_definitions = agent._tool_definitions()
+        agent.parameter_types = parameter_types_of(agent.tool_definitions)
+        agent.template_tools = agent.dialect.rendering.template_tools(agent.tool_definitions)
+        agent.template_kwargs = agent.loaded_model.chat_template_kwargs(run_result.agent_config.call_kwargs)
+        if agent.ac_model is not None and run_result.ac_model_version is not None and agent.ac_model.version != run_result.ac_model_version:
+            print(f"Agent {agent.agent_id[:8]} - resuming rows with AC version {agent.ac_model.version}, recorded {run_result.ac_model_version}", flush=True)
+        agent.messages = deepcopy(run_result.prompt_messages)
+        agent.prefix = list(run_result.prompt_token_ids)
+        agent.segment_start = len(agent.prefix)
+        spans = [(span["start"], span["start"] + span["length"]) for span in run_result.prompt_ac_spans]
+        parts_messages: list[dict] = list(run_result.prompt_messages)
+        for step in run_result.trajectory:
+            offset = len(agent.prefix)
+            agent.messages.extend(deepcopy(step["messages"]))
+            agent.prefix += list(step["token_ids"])
+            spans += [(offset + span["start"], offset + span["start"] + span["length"]) for span in step["ac_spans"]]
+            parts_messages += step["messages"]
+            if step["role"] == "assistant":
+                agent.last_sampled = list(step["token_ids"])
+        agent.spans = spans
+        agent.rows = agent._encode_parts(parts_messages)
+        assert [int(r.shape[0]) for r in agent.rows] == [end - start for start, end in spans], "recorded spans differ from the re-encoded rows"
+        agent.started = True
+        agent._after_append()
+        return agent
+
+    def simulate_step(self) -> dict:
+        """
+        One step without the engine. Executes the last assistant turn's pending calls (compaction, large outputs and
+        search for real; subagents in step mode; refusals when compaction is due), appends their results, and prepares
+        the next request. Returns what a test inspects; nothing is submitted.
+        """
+        self.step_mode = True
+        if not self.started:
+            self.begin()
+        results = self.run_results
+        executed = None
+        last = results.trajectory[-1] if results.trajectory else None
+        if last is not None and last["role"] == "assistant" and last["tool_calls"] and not results.finish_reason:
+            calls = last["tool_calls"]
+            if self.compaction_due and not self._is_lone_compact(calls):
+                executed = [self._refusal(call) for call in calls]
+                self.append_tool_results(calls, executed)
+                self._refused()
+            else:
+                executed = self._execute_tool_calls(calls)
+                if not any(result.compacted for result in executed):
+                    self.append_tool_results(calls, executed)
+        prefix, embeds, mask = self.prepare_request()
+        return {
+            "prefix_tokens": len(prefix), "spans": list(self.spans), "rows": [tuple(r.shape) for r in self.rows],
+            "embeds_shape": None if embeds is None else tuple(embeds.shape), "row_positions": None if mask is None else mask.count(False),
+            "compaction_due": self.compaction_due, "compactions": len(results.compactions), "finish_reason": results.finish_reason,
+            "results": executed, "messages": len(self.messages),
+        }
+
+    def submit_probe(self):
+        """Submit the prepared request once without recording anything: the engine interface check of the test."""
+        return self._submit()
```

`activation/agent/agent_utils.py · SyntheticTurn, synthesize_agent()` (new section; existing helpers omitted)

```python
# Synthetic runs (new section in the existing module).
"""
Synthetic runs for tests: turns are recorded through the same Agent methods the loop uses, with the real dialect and
tokenizer, so token segments, spans and rows are exactly what a rollout would record. A turn's tool outputs are either
given (recorded as tool results, truncation and parts included) or left pending for `Agent.simulate_step()` to execute.
One-time test support: the main-tree test is the only intended consumer.
"""
# Reuse existing typing, uuid and dataclass imports; extend TYPE_CHECKING only.
if t.TYPE_CHECKING:
    from .agent import Agent
    from .agent_config import AgentConfig
    from activation.harness import HarnessRuntime


@dataclass
class SyntheticTurn:
    content: str                                                   # the assistant's text (tool-call blocks are rendered from `calls`)
    calls: list[tuple[str, dict]] = field(default_factory=list)   # (tool name, arguments)
    outputs: list[str] | None = None                               # tool outputs to record; None leaves the calls pending


def synthesize_agent(harness: "HarnessRuntime", config: "AgentConfig", turns: list[SyntheticTurn], seed: int = 0) -> "Agent":
    """An agent whose run result holds `turns`, recorded as the loop would (sampled tokens = the rendered turn plus end-of-turn)."""
    # Agent imports this utils module; runtime imports here avoid a module cycle.
    from .agent import Agent
    from .agent_tools import ToolCallResult

    agent = Agent(harness, config, seed=seed)
    agent.step_mode = True
    agent.begin()
    tokenizer = agent.loaded_model.tokenizer
    eot = agent.loaded_model.model_config.model_description.eot_token or ""
    for turn in turns:
        calls = [{"id": uuid.uuid4().hex[:8], "name": name, "arguments": dict(arguments)} for name, arguments in turn.calls]
        blocks = [agent.dialect.preferred_format.render(call["name"], call["arguments"]) for call in calls]
        text = "\n\n".join([turn.content] + blocks) if turn.content else "\n\n".join(blocks)
        token_ids = tokenizer.encode(text + eot, add_special_tokens=False)
        agent.record_assistant_turn(turn.content, calls, token_ids, logprobs=[])
        if calls and turn.outputs is not None:
            assert len(turn.outputs) == len(calls), "one output per call"
            results = [agent._finalize_result(call, ToolCallResult(output=output)) for call, output in zip(calls, turn.outputs)]
            agent.append_tool_results(calls, results)
    return agent
```

`activation/tests/test_basic_agent_ac.py` (new; full code for review)

```python
"""
Slice 3b key test: activation context in the agent loop, on a GPU node.

    uv run sky exec --sync <node> -- uv run pytest activation/tests/test_basic_agent_ac.py --gpu --slow -s

Untrained AC rows are junk, so the simulated scenarios assert mechanics (parts, spans, rows, records, the engine
accepting mixed prompts and hitting its prefix cache), not answers. `AC_CHECKPOINT=AC_MODELS/ac_dev/epoch_002` loads a
3a1 checkpoint instead. The last test is a real text-only rollout with a low compaction threshold.
"""
import json
import os
from dataclasses import replace

import pytest
import torch

from activation.ac_model import ActivationContextModelConfig
from activation.ac_model.ac_model_utils import direct_parts
from activation.agent import Agent, AgentConfig, AgentRunResult, RolloutReporter, SemanticSearchTool, SubagentTool
from activation.agent.agent_utils import SyntheticTurn, synthesize_agent
from activation.common.data_syncing import resolve_path
from activation.dataset.loaders import BrightDataset
from activation.harness import FREE_DEVICE, HarnessRuntime, HarnessRuntimeConfig, ModelConfig

MODEL_NAME, MODEL_ID = "qwen3.5-4b", "Qwen/Qwen3.5-4B"
SIDE_NAME, SIDE_ID = "qwen3.5-0.8b", "Qwen/Qwen3.5-0.8B"
AC_NAME, SIDE_LORA, TARGET_LORA = "ac_dev", "ac_side", "ac_target"
SYSTEM = ("You are a careful problem solver working in a sandbox. Use the python or shell tools to compute rather than "
          "guessing. When you are done, call submit_answer exactly once with the final answer.")
BASE = AgentConfig(system_prompt=SYSTEM, model_name=MODEL_NAME, ac_model_name=AC_NAME, enable_ac_communication=True,
                   call_kwargs={"sampling_params": {"max_tokens": 1024, "temperature": 0.7}}, max_turns=12, max_tool_errors=3, max_duration=300)
LONG = "0123456789" * 120   # 1,200 chars per synthetic tool output


def _harness(with_ac: bool = True) -> HarnessRuntime:
    configs = {MODEL_NAME: ModelConfig(MODEL_NAME, MODEL_ID)}
    if with_ac:
        configs[SIDE_NAME] = ModelConfig(SIDE_NAME, SIDE_ID)
    harness = HarnessRuntime(HarnessRuntimeConfig(model_configs=configs, agent_max_concurrent=8))
    if with_ac:
        harness.module_manager.register_lora(TARGET_LORA, MODEL_NAME, rank=64)
        harness.module_manager.register_ac_model(ActivationContextModelConfig(
            AC_NAME, SIDE_NAME, SIDE_LORA, MODEL_NAME, TARGET_LORA, checkpoint_path=os.environ.get("AC_CHECKPOINT")))
    return harness


def _release(harness: HarnessRuntime) -> None:
    for name, loaded in harness.loaded_models.items():
        loaded.engine_to_device(FREE_DEVICE)
    if AC_NAME in harness.module_manager.ac_models:
        ac_model = harness.module_manager.get_ac_model(AC_NAME)
        ac_model.release()
        harness.module_manager.free_lora(SIDE_NAME, SIDE_LORA)
    for loaded in harness.loaded_models.values():
        loaded.model_to_device(FREE_DEVICE)


def _probe_engine(agent: Agent, label: str) -> None:
    """One real mixed-prompt request, then the same again: the engine accepts the rows and serves the prefix from its cache."""
    prefix, embeds, mask = agent.prepare_request()
    first = agent.submit_probe()
    second = agent.submit_probe()
    print(f"  [{label}] prompt {len(prefix)} tokens, rows at {0 if mask is None else mask.count(False)} positions, "
          f"cached {first.cached_prompt_token_count} -> {second.cached_prompt_token_count}, sampled {first.output_token_count} tokens")
    assert first.prompt_token_count == len(prefix)
    if embeds is not None:
        assert tuple(embeds.shape) == (len(prefix), agent.loaded_model.model_config.model_description.d_model)
        assert embeds.dtype == agent.row_dtype
    assert second.cached_prompt_token_count > 0, "the second identical request did not hit the prefix cache"


@pytest.mark.gpu
@pytest.mark.slow
def test_ac_agent_channels_simulated():
    """Compaction, subagent, large output and search through simulate_step; a resume round trip; one engine submit per scenario."""
    harness = _harness()
    bright = BrightDataset.load(harness, max_examples=20, domain="biology", max_corpus_documents=200)
    harness.dataset_manager.register_dataset(bright)
    task = next(iter(bright.scorable_tasks.values()))
    ac_model = harness.module_manager.get_ac_model(AC_NAME)
    harness.loaded_models[MODEL_NAME].ensure_engine_loaded()          # after the AC registration: enable_prompt_embeds and the memory reservation apply
    try:
        # --- compaction: three long tool outputs cross a 600-token delta; a python call is refused; a lone compact restarts the segment.
        config = replace(BASE, user_prompt="Sum the 140th to 142nd primes.", compaction_threshold_tokens=600)
        turns = [SyntheticTurn(f"Step {i}.", [("python", {"code": f"print({i})"})], [LONG]) for i in range(3)]
        agent = synthesize_agent(harness, config, turns)
        assert agent.compaction_due and not agent.run_results.compactions
        agent.record_assistant_turn("More work.", [{"id": "r1", "name": "python", "arguments": {"code": "print(4)"}}],
                                    agent.loaded_model.tokenizer.encode("More work."), [])
        state = agent.simulate_step()
        assert state["results"][0].output.startswith("Refused python") and agent.compaction_refusals == 1
        agent.record_assistant_turn("", [{"id": "c1", "name": "compact", "arguments": {"summary": "Printed 0..3; next sum the primes."}}],
                                    agent.loaded_model.tokenizer.encode("compact"), [])
        state = agent.simulate_step()
        results = agent.run_results
        assert state["compactions"] == 1 and results.compactions[0].finish_reason == "compacted"
        assert results.compactions[0].num_turns == 5 and len(results.compactions[0].trajectory) == 10
        assert results.trajectory == [] and not agent.compaction_due and agent.segment_start == len(agent.prefix)
        assert [m["role"] for m in results.prompt_messages] == ["system", "user", "user"]
        tree = direct_parts(results.prompt_messages)[0]
        assert tree["compression_target"] == config.ac_compaction_ratio and tree["messages"][0]["content"] == config.user_prompt
        assert len(tree["messages"]) == 1 + 10 and tree["messages"][-1]["tool_calls"][0]["function"]["name"] == "compact"
        assert results.prompt_messages[2]["content"][1]["text"].endswith("Printed 0..3; next sum the primes.")
        assert len(agent.spans) == 1 and agent.rows[0].shape[0] == ac_model.part_view_rows(tree["messages"], tree["compression_target"])
        assert state["embeds_shape"] == (len(agent.prefix), agent.rows[0].shape[1]) and state["row_positions"] == agent.rows[0].shape[0]
        _probe_engine(agent, "compaction")
        # A second compaction nests the first tree verbatim.
        agent.record_assistant_turn("", [{"id": "c2", "name": "compact", "arguments": {"summary": "again"}}], agent.loaded_model.tokenizer.encode("compact"), [])
        agent.compaction_due = True
        agent.simulate_step()
        outer = direct_parts(agent.run_results.prompt_messages)[0]
        assert direct_parts(outer["messages"])[0] == tree and len(agent.run_results.compactions) == 2

        # --- subagent in step mode: the child's prompt carries the parent segment; the parent's result carries the child's segment.
        child_base = replace(BASE, user_prompt="", tools={})
        config = replace(BASE, user_prompt="Delegate the prime search.", tools={"subagent": (SubagentTool, {"base_config": child_base})})
        agent = synthesize_agent(harness, config, [SyntheticTurn("Delegating.", [("subagent", {"task": "Find the 140th prime."})])])
        state = agent.simulate_step()
        child = agent.run_results.subagent_results[0]
        assert child.finish_reason == "simulated" and child.answer.endswith("simulated run")
        child_user = child.prompt_messages[1]
        assert [p.get("type") for p in child_user["content"]] == ["text", "activation_context", "text"]
        assert child_user["content"][2]["text"] == "Find the 140th prime." and child_user["content"][1]["messages"][0]["content"] == config.user_prompt
        assert child.prompt_ac_spans and child.prompt_ac_spans[0]["length"] == ac_model.part_view_rows(child_user["content"][1]["messages"], BASE.ac_subagent_ratio)
        tool_step = agent.run_results.trajectory[-1]
        assert [p.get("type") for p in tool_step["messages"][0]["content"]] == ["activation_context", "text"]
        assert tool_step["messages"][0]["content"][0]["messages"][0]["content"][1]["type"] == "activation_context"   # the child's segment starts with its part
        assert len(tool_step["ac_spans"]) == 1 and len(agent.spans) == 1
        _probe_engine(agent, "subagent")

        # --- large tool output: a real python call prints 100k chars; the part holds the full text nested with the parent segment.
        config = replace(BASE, user_prompt="Print a lot.")
        agent = synthesize_agent(harness, config, [SyntheticTurn("Printing.", [("python", {"code": "print('x' * 100000)"})])])
        state = agent.simulate_step()
        result = state["results"][0]
        assert len(result.output) < 21_000 and "truncated" in result.output and result.content is not None
        part, text = result.content
        assert part["type"] == "activation_context" and text["text"] == result.output
        assert part["compression_target"] == BASE.ac_tool_output_ratio and part["messages"][1]["role"] == "tool"
        assert len(part["messages"][1]["content"]) >= 80_000 and direct_parts(part["messages"])[0]["messages"][0]["content"] == config.user_prompt
        assert agent.run_results.trajectory[-1]["ac_spans"][0]["length"] == agent.rows[0].shape[0]
        _probe_engine(agent, "tool output")

        # --- semantic search: top_k visible, min(20, 2 top_k) extra as a part after the text.
        config = replace(BASE, user_prompt=task.agent_prompt, dataset_task=task, tools={"semantic_search": (SemanticSearchTool, {"top_k": 3})})
        agent = synthesize_agent(harness, config, [SyntheticTurn("Searching.", [("semantic_search", {"query": task.agent_prompt[:200]})])])
        state = agent.simulate_step()
        result = state["results"][0]
        assert result.output.count("\n[") + result.output.startswith("[") == 3 and result.content[0]["type"] == "text" and result.content[1]["type"] == "activation_context"
        assert result.content[1]["messages"][1]["content"].count("[") >= 6 and result.content[1]["compression_target"] == BASE.ac_search_ratio
        _probe_engine(agent, "search")

        # --- resume: the serialized record rebuilds the same prefix, spans and row shapes.
        data = json.loads(json.dumps(agent.run_results.serialize()))
        resumed = Agent.resume_from_run_result(harness, AgentRunResult.deserialize(data, harness))
        assert resumed.prefix == agent.prefix and resumed.spans == agent.spans
        assert [tuple(r.shape) for r in resumed.rows] == [tuple(r.shape) for r in agent.rows]
        assert resumed.messages == agent.messages and resumed.compaction_due == agent.compaction_due
        print("\n=== AC model stats ===\n" + json.dumps(ac_model.stats.summarize(), indent=1))
    finally:
        _release(harness)


@pytest.mark.gpu
@pytest.mark.slow
def test_ac_agent_rollout_with_ac():
    """A real rollout with AC rows and a low threshold: mechanics only (segments recorded, rows shipped, cache hits), the answer is printed."""
    harness = _harness()
    config = replace(BASE, user_prompt="Compute the sum of the 140th, 141st and 142nd prime numbers (2 is the 1st prime).",
                     compaction_threshold_tokens=1500)
    reporter = RolloutReporter(str(resolve_path("AGENT_AC_TEST/with_ac")), title="Agent AC test: primes with compaction")
    try:
        results = harness.rollout_manager.perform_grouped_rollouts([config], group_count=2, base_seed=0, perform_scoring=False, reporter=reporter)[0]
    finally:
        _release(harness)
    for result in results:
        print(f"\n=== seed {result.seed}: {result.finish_reason}, {result.num_turns} turns, {len(result.compactions)} compactions, "
              f"{result.num_ac_parts} parts / {result.num_ac_rows} rows, cached {result.num_cached_input_tokens}, answer {result.answer!r}")
        assert result.finish_reason in ("submitted", "max_turns", "max_tool_errors", "max_duration", "no_tool_call", "compaction_refused", "trajectory_cap")
        assert result.ac_model_name == AC_NAME and result.ac_model_version is not None
        for segment in result.compactions:
            assert segment.finish_reason == "compacted" and segment.prompt_token_ids and segment.trajectory
        if result.compactions:
            assert result.prompt_ac_spans and result.num_ac_rows > 0
    assert os.path.exists(os.path.join(reporter.report_folder, "report.tressoir.html"))


@pytest.mark.gpu
@pytest.mark.slow
def test_agent_compaction_text_only():
    """No AC model: the same protocol with the summary alone; at least one rollout compacts and still finishes."""
    harness = _harness(with_ac=False)
    config = replace(BASE, ac_model_name=None, enable_ac_communication=False, compaction_threshold_tokens=1500,
                     user_prompt="Compute the sum of the 140th, 141st and 142nd prime numbers (2 is the 1st prime).")
    reporter = RolloutReporter(str(resolve_path("AGENT_AC_TEST/text_only")), title="Agent AC test: text-only compaction")
    try:
        results = harness.rollout_manager.perform_grouped_rollouts([config], group_count=2, base_seed=0, perform_scoring=False, reporter=reporter)[0]
    finally:
        _release(harness)
    for result in results:
        print(f"\n=== seed {result.seed}: {result.finish_reason}, {result.num_turns} turns, {len(result.compactions)} compactions, answer {result.answer!r}")
        assert result.num_ac_parts == 0 and result.prompt_ac_spans == []
        for segment in result.compactions:
            assert segment.finish_reason == "compacted"
            assert isinstance(segment.prompt_messages[-1]["content"], str)
    assert any(result.compactions for result in results), "no rollout crossed the 1,500-token delta"
    assert any(result.finish_reason == "submitted" for result in results)
```

Validation plan for this milestone: CPU with the tiny fixture and a fake engine-free path (`synthesize_agent`, `simulate_step` without the probe, `resume_from_run_result`), then the node run of the three tests. The test's search scenario registers a 200-document BRIGHT corpus so the BM25 index builds in seconds.

</details>

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">M7 — Required run-result conversion to AC training items</span>
    <span class="card-oneliner">Harvesting in the study class; conversion is required, training waits for definitive recipes.</span>
    <span class="card-badge">Review</span>
  </summary>
#### What landed

`ActivationContextStudyGenerator.items_from_run_results(runs, weight_of=, seed=, completion_max_tokens=, text_run_kwargs=, items_per_text_run=)` with `harvest_report` counts; `_rollout_sources` visits a run and the subagents attached to each of its segments; `_items_from_ac_run` yields one item per assistant turn whose student prefix (the segment's real prompt + the steps before it) holds a part, teacher = `_expand_parts(student)` (a compaction tree → the history it holds with a synthetic `Context compacted.` tool result and without the instructions text; a `tool_output` part → the full output once; `search` → the extras after the visible passages; subagent parts → transcripts), completion = `_rollout_completion` (one terminal EOT stripped, no cap by default); `_items_from_text_run` converts a text-only run to a `DatasetDocument` (compaction messages dropped) and cuts it with `_compaction_item`, then swaps the recorded sampled tokens in for whole-turn completions; `_compaction_item` now records `target_position` / `mid_turn_cut`. Trainer integration (fixed rows in `AgentTrainer`) is deferred to the training-recipe slices, and `build_example` refuses AC-bearing runs meanwhile.

#### Drifts, challenges, and unplanned steps

The plan's `as_text` heuristics became explicit `kind` tags on parts. The teacher drops the task the tree repeats (the prompt already shows it). Mid-turn cuts of text-only runs keep the generator's re-encoded completion (`completion_source: reencoded`) because the recorded tokens cannot be split at a re-encoded boundary; whole-turn cuts take the recorded ids (`recorded`). Simulated children produce no assistant turn and therefore no item; a child that ran does.

#### Focused actual diffs

`activation/ac_model/ac_model_study.py · _items_from_ac_run()`

```python
            for t_index, step in enumerate(segment.trajectory):
                if step.get("role") == "assistant":
                    student = base + history
                    parts = direct_parts(student)
                    if not parts:
                        self._harvest_skip("no_parts")
                    else:
                        ids, complete = self._rollout_completion(step.get("token_ids") or [], completion_max_tokens)
                        ...
                        items.append(ActivationContextTrainingItem(
                            item_id=f"rollout:{source_key}:{k}:{t_index}", kind="compaction",
                            in_context_prefix=system + self._expand_parts(student), ac_prefix=deepcopy(system + student),
                            completion_text=self.tokenizer.decode(ids, clean_up_tokenization_spaces=False),
                            completion_token_ids=ids, completion_complete=complete, tools=tools, weight=weight,
                            dataset_id="agent_runs", doc_ids=[source_key], info={..., "channels": sorted({part.get("kind") or "unknown" for part in parts}), ...}))
                history += deepcopy(step.get("messages") or [])
```

`activation/ac_model/ac_model_study.py · _rollout_completion()`

```python
        eot_ids = self.tokenizer.encode(eot, add_special_tokens=False) if eot else []
        ids = list(recorded_ids)
        complete = bool(eot_ids) and len(ids) >= len(eot_ids) and ids[-len(eot_ids):] == eot_ids
        if complete:
            ids = ids[:-len(eot_ids)]
        if max_tokens is not None and len(ids) > max_tokens:
            return ids[:max_tokens], False
        return ids, complete
```

#### Validation

`IB/TMP/SLICE3B/cpu_checks/harvest_cpu.py` passes in ~13 s: a serialized AC run (compaction, then a 100k-char output, a search, a step-mode subagent, a final turn) yields 4 items whose channels grow from `[compaction]` to `[compaction, search, subagent_return, tool_output]`; segment 0's four turns are skipped as `no_parts`; the first teacher is `system, user, (assistant, tool) x4` ending in `Context compacted.` with no parts; the last teacher holds the 80k-char output once, the subagent transcript and the extra passages; negative weights raise, zero weights skip, explicit weights apply; a text-only run with one compaction yields 4 cut items with recorded completions; `ActivationContextTrainer.build_example` accepts every item (completion length + EOT when complete, part requests present). Node: the with-AC rollout test harvests its runs: `uv run pytest activation/tests/test_basic_agent_ac.py --gpu --slow -s` on `ac-slice3b-temp` (RTX PRO 6000, Qwen3.5-4B target with the fp8 KV cache, Qwen3.5-0.8B side, untrained AC rows), logs under `IB/TMP/SLICE3B/node/AGENT_AC_TEST/`. Simulated channels (`pytest_simulated.log`, 71 s): compaction prompt 1,259 tokens with rows at 391 positions, repeat cached 1,056; second compaction 910 tokens / 53 rows; subagent 944 tokens / 4 rows; a real 100k-character python output in the sandbox: 3,620 tokens / 251 rows, repeat cached 3,168; BRIGHT search: 2,161 tokens / 117 rows, repeat cached 2,112; every request accepted with the full-length bf16 tensor and the model answered sensibly on untrained rows (a python tool call, a search follow-up). With-AC rollouts (`pytest_rerun.log`, threshold 300 tokens): seed 0 submitted after 8 turns, 2 compactions, 2 parts / 223 rows, 4,224 cached prompt tokens on its last request; seed 1 after 4 turns, 1 compaction, 60 rows, 1,056 cached; both answered 2441 (the correct sum); the harvest produced 6 items from 12 candidates (6 turns before any part skipped). Text-only rollouts (`pytest.log`): both seeds submitted 2441 after 4 turns with 1 compaction each. Final clean run of the whole file after the test fixes (`pytest_final.log`): 3 passed in 170 s; in that run the with-AC seed 0 hit `max_turns` after 12 turns with 3 compactions and 355 rows (untrained rows: the model kept re-deriving the primes) while seed 1 submitted 2441 after one compaction, and the harvest yielded 10 items from 16 candidates.

---

*Planned changes (reference):*


#### Planning Overview

**Required for slice 3b: convert recorded run results into usable AC training items.** Implement `ActivationContextStudyGenerator.items_from_run_results()` and its private helpers as a new **run-result harvesting section of the existing study class**, using `self.harness`, `self.ac_name`, `self.ac_model`, `self.tokenizer` and the existing cut logic. This is a deliverable with acceptance checks; an implementation agent cannot mark M7 complete by deferring it.

| M7 work | Classification | Completion requirement |
| --- | --- | --- |
| Convert AC-enabled runs, their compacted segments and subagents into `ActivationContextTrainingItem` objects | **Required in 3b** | Preserve the actual student prompt, recover the corresponding plain-text teacher prefix, and retain the recorded continuation tokens and source provenance. Cover all four AC channels. |
| Convert text-only run results using the existing 3a1 cut generator | **Required in 3b** | Reconstruct source trajectory documents and reuse `_compaction_item`; do not silently skip a run because `ac_model_name` is unset. |
| Validate conversion and demonstrate consumption by existing item preparation | **Required in 3b** | Nonempty deterministic fixtures, serialization/source fidelity, template/length preparation and explicit skip counts. No optimizer step is needed. |
| Rebuild fixed AC rows inside AgentTrainer; add AC-bearing training batches; run Agent/AC optimization on harvested items | **Deferred to later training-recipe slices** | These are training work. Decide the definitive objectives, sampling, weighting and schedules there; they are not 3b acceptance gates. |

The 3a1 item contract supports KL self-distillation: teacher = plain-text prefix, student = actual AC prefix, completion = the turn the agent sampled. Constructing that item does not commit us to a definitive training recipe or require fitting either model. The earlier “deferrable” label applies **only to training itself**, never to conversion, its helpers or validation. The M1 migration of existing record readers remains required for compatibility, independently of the deferred new trainer path.

#### Planned Changes

`activation/ac_model/ac_model_study.py · ActivationContextStudyGenerator.items_from_run_results()`

```diff
@@ activation/ac_model/ac_model_study.py — imports / type-only imports @@
+import math
 if t.TYPE_CHECKING:
+    from ..agent.agent_config import AgentRunResult
     from ..harness import HarnessRuntime
@@ activation/ac_model/ac_model_study.py — ActivationContextStudyGenerator: run-result harvesting @@
+    def items_from_run_results(
+        self, runs: list["AgentRunResult"], *,
+        weight_of: t.Callable[["AgentRunResult"], float] | None = None,
+        seed: int = 0,
+    ) -> list[ActivationContextTrainingItem]:
+        """Build items from recorded runs without generation, model fitting or mutation of the records."""
+        rng = random.Random(seed)
+        items: list[ActivationContextTrainingItem] = []
+        # Candidate/accepted/skip counters and reporting omitted from this excerpt.
+        for root in runs:
+            weight = float(root.score if weight_of is None else weight_of(root))
+            if not math.isfinite(weight) or weight < 0:
+                raise ValueError("AC distillation item weights must be finite and nonnegative")
+            if weight == 0:
+                continue
+            # Visit each run once, including children attached to completed segments.
+            # Children inherit this root's selected weight; source_key includes their path.
+            for source_key, run in self._rollout_sources(root):
+                if run.ac_model_name is None:
+                    items.extend(self._items_from_text_run(run, source_key, weight, rng))
+                elif run.ac_model_name == self.ac_name:
+                    items.extend(self._items_from_ac_run(run, source_key, weight))
+                else:
+                    raise ValueError("Run uses a different AC model; select the matching study generator")
+        return items
```

The caller creates the existing `ActivationContextStudyGenerator(harness, ac_model_name)`, supplies in-memory or deserialized run results, and receives ordinary `ActivationContextTrainingItem` objects. It can save those items through the existing item format or pass them to preparation/evaluation later. No new harvest module or trainer-owned conversion path is introduced. Reuse the existing imports/class layout; only the relevant additions are shown.

`activation/ac_model/ac_model_study.py · ActivationContextStudyGenerator._items_from_ac_run()`

```python
    def _items_from_ac_run(self, run: "AgentRunResult", source_key: str,
                          weight: float) -> list[ActivationContextTrainingItem]:
        items = []
        # _rollout_item_candidates walks the real segment history in order.
        # It yields one candidate per eligible assistant turn, with all prior
        # AC parts represented in that candidate's student prefix.
        for candidate in self._rollout_item_candidates(run):
            items.append(ActivationContextTrainingItem(
                item_id=f"rollout:{source_key}:{candidate.segment_index}:{candidate.turn_index}",
                kind="compaction",  # existing continuation-distillation kind
                in_context_prefix=deepcopy(candidate.teacher_prefix),
                ac_prefix=deepcopy(candidate.student_prefix),
                completion_text=self.tokenizer.decode(candidate.completion_token_ids),
                completion_token_ids=list(candidate.completion_token_ids),
                completion_complete=candidate.completion_complete,
                tools=deepcopy(candidate.tools), weight=weight,
                info={"source": "agent_run", "source_key": source_key,
                      "segment": candidate.segment_index, "turn": candidate.turn_index,
                      "channels": candidate.channels, "score": run.score,
                      "ac_model_name": run.ac_model_name, "ac_model_version": run.ac_model_version},
            ))
        return items
```

The candidate record is a private implementation detail, shown to make the item mapping explicit; it does not add a public API or a module. Its completion fields are normalized from the recorded sampled IDs as below, not re-encoded from display text. The iterator/traversal bodies are omitted at Moderate detail, but **all cases in the following table are required implementation**, not optional follow-ups. Reuse the existing completion-boundary and source-reference logic where compatible; do not introduce a second general-purpose trajectory reformatter.

`activation/ac_model/ac_model_study.py · ActivationContextStudyGenerator._rollout_completion()`

```python
    def _rollout_completion(self, recorded_ids: list[int], max_tokens: int | None = None) -> tuple[list[int], bool]:
        """Recorded sampled ids as completion fields. No cap by default: a run's turn is bounded by its own sampling max_tokens."""
        eot = self.ac_model.target.model_config.model_description.eot_token or ""
        eot_ids = self.tokenizer.encode(eot, add_special_tokens=False)
        complete = bool(eot_ids) and recorded_ids[-len(eot_ids):] == eot_ids
        # The existing item preparer appends EOT when completion_complete=True.
        # Strip exactly one recorded terminator here so it is appended once.
        ids = list(recorded_ids[:-len(eot_ids)] if complete else recorded_ids)
        if max_tokens is not None and len(ids) > max_tokens:
            return ids[:max_tokens], False
        return ids, complete
```

Skip an empty payload with a recorded reason. Validate that preparation restores exactly the recorded completion sequence, or its explicitly capped token prefix; never infer a terminator for a capped/incomplete turn. This normalization applies to both AC-enabled and text-only sources.

| Recorded source | Student prefix | Teacher prefix / continuation |
| --- | --- | --- |
| Compacted-later segment | Its real first prompt, including visible summary/instructions and the recursive tree, then only steps before the sampled turn | The same source history expanded once to plain text; completion is that turn's recorded sampled tokens. |
| Subagent prompt and returned child context | The child's actual prompt with its parent part, or the parent's actual prompt after receiving the child's part | Expand the corresponding parent/child source segment and its compaction tree at the same position; use the next recorded assistant turn in that receiving agent. |
| Large tool output | The actual `[part, truncated text]` result and following visible history before the turn | Replace the represented output with the recorded full output once; do not append both its full and truncated copies. |
| Semantic search | Actual visible passages followed by the extra-passage part | Expand the recorded extra passages in order, preserving the visible passages once. |
| Text-only runs | A synthesized cut using existing 3a1 generation rules | Reconstructed original task/trajectory and recorded continuation; no new completion is sampled. |

Nested parts contain conditioning history as well as new content. Expansion must be channel-aware: it must not blindly concatenate that repeated history or silently discard a part. Preserve dialect messages, tool definitions, task text, partial-token boundaries and the actual completion end marker. Never add future messages to either prefix. Traverse children attached to every completed/current segment once, retaining child paths in deterministic source keys; seed alone is not a unique source identity. Return one item per eligible turn, recording multiple contributing channels rather than duplicating a turn per channel. Zero-weight and ineligible/over-budget candidates are counted and reported; incomplete source records get an explicit reason, not fabricated history. Use finite nonnegative selected weights, with a caller override for unscored fixtures; defining a training score/advantage recipe remains deferred.

`activation/ac_model/ac_model_study.py · ActivationContextStudyGenerator._items_from_text_run()`

```python
    def _items_from_text_run(self, run: "AgentRunResult", source_key: str,
                            weight: float, rng: random.Random) -> list[ActivationContextTrainingItem]:
        # Reconstruct original task + recorded messages over completed/current
        # segments, retaining source tool definitions and system instructions.
        document = self._rollout_document(run, source_key)
        # Call self._compaction_item with the existing 3a1 cut/ratio/budget
        # defaults and this deterministic RNG; retain source turn/token offsets.
        # Restore completion IDs from that recorded turn, then normalize its
        # recorded EOT with _rollout_completion (the cut generator's re-encoded
        # assistant_text and length-inferred completeness are not authoritative).
        # Apply weight and source_key to yielded items. Count shortfall/skips.
        ...
```

This helper body is intentionally abbreviated; implement it using the existing `_compaction_item` arguments/defaults from `generate_compaction_samples`, without a new cut implementation or silently widening its budgets. Keep provenance mapping from each reconstructed source turn to its original sampled IDs, recover the selected cut's token offset, and replace the generated completion fields with those recorded IDs and their actual end-marker state. If the exact mapping cannot be recovered, reject/resample that candidate with an explicit reason; do not silently accept a re-tokenized target. The existing 512-token completion / 72k teacher limits remain in force and completion truncation must update `completion_complete`. A model that has produced rows must match the study generator's target/tokenizer contract; conversion uses stored parts and source text and does not require re-encoding the historical AC checkpoint.

**M7 acceptance:** CPU fixtures must yield nonempty items for each channel and text-only runs, including a child attached to a completed segment. Check exact completion IDs, teacher/student source boundaries, recursive expansion without duplicate history, deterministic IDs, serialization round trip, finite nonnegative weights, explicit zero/negative/unscored behavior, and no input mutation. Run the existing item preparation/tokenization with the tiny fixture to establish compatibility, without an optimizer update. Convert runs saved by M6 as well; use `weight_of=lambda _: 1.0` for its unscored mechanics fixtures and report candidate/accepted/skip counts so an empty result cannot masquerade as success. M7 is incomplete until these conversions work; no definitive training recipe is required to finish it.

</details>

## Validation plan

- **CPU (tiny Qwen3.5 fixture, no engine):** the prompt-shape script already run; after M1–M5, `synthesize_agent` + `simulate_step` for all four channels with a CPU-registered AC model (0.8B side would not fit the 30 GB host with the 4B; the tiny fixture serves both roles as in the 3a1 CPU checks), serialization round trip through `AgentRunResult.serialize/deserialize`, `resume_from_run_result`, selection `unroll`, `ActivationContextStudyGenerator.items_from_run_results` on all four channels plus text-only runs, including children attached to completed segments. Validate exact source/completion tokens and existing item preparation without training. Existing record-reader regression checks remain required.
- **Node (RTX PRO 6000, 4B engine + 0.8B side):** M0 probe first; then `test_basic_agent_ac.py` (three tests); then the existing `test_basic_agent.py` and `test_basic_agent_training.py` once each, since the step record changed under them; M7 conversion on the saved AC-enabled and text-only run results with explicit fixture weights; report accepted and skipped items. No new AC-bearing trainer path or optimizer run is required in this slice.
- **Handoff:** the slice 3b round document with per-file diff cards against the 3a1 tree, IB-only probes under `slice3b/probes/`, no new public test file besides the one named here.

## Boundaries

Baseline is the 3a1 work tree; the agent package is identical to `/source`, so the agent delta stays clean once 3a1 lands. No vLLM modification unless M0's gate fires; the two relaxations then become their own milestone. Ratios 1/20 and 1/40 lie outside the 3a1-trained 1/8–1/16 range and are accepted for mechanics; the next AC training round widens the range. Run-result conversion, including AC-enabled and text-only sources, is mandatory in M7. New fixed-row AgentTrainer integration, AC/Agent optimization runs on these items, and policy-gradient training through rows belong to later slices where definitive training recipes are designed. M1 record-reader migrations and existing regression checks still belong to 3b.

## Status log

- 2026-09-09 v0.1: intent pass written from the user's notes; vLLM facts re-verified in the pinned source; rules diagram published.
- 2026-09-09 v0.2: seven plan decisions and two rules decisions integrated; the two questions answered; shapes reverified on CPU; every milestone moved to Planning with Moderate diffs; the full test file included in M6; initial M7 conversion outline added.
- 2026-09-09 v0.5: all milestones in Review with completion reports; CPU checks pass; node validation on a temporary RTX PRO 6000 (payload probe: 1 to 3 percent overhead at concurrency 8 to 64, gate passed; `test_basic_agent_ac.py`: 3 passed after three test-side fixes); the temporary node was torn down, `ac-fp4-probe` stays stopped with its disk; exact delta staged in SLICE3B_ROUND.
- 2026-09-09 v0.4: implementation started in `IB/TMP/SLICE3B/implementation/work` (all milestones Implementing); CPU channel and harvest checks pass; node run of the payload probe and the AC agent test in progress.
- 2026-09-09 v0.3: harvesting moved into the study class and synthetic-run helpers into agent_utils; M7 conversion and acceptance are mandatory, while training integration/recipes are deferred. Planned helper imports avoid a cycle; removed a vacuous assertion from the planned text-only test. Plan is handoff ready, with no open human decisions; product implementation is not claimed.
