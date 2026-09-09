# Activation context, slice 3b: implementation handoff

Slice 3b is implemented, validated on CPU and on a GPU node, and staged below as an exact delta against your source. Activation-context parts now flow through compaction, subagents, tool outputs and search in live rollouts, run records keep every segment, and run results convert to AC training items. The source itself is unchanged.

## What changed

- Activation-context parts (`{"type": "activation_context", ..., "kind"}`) ride inside dialect messages. The agent templates them as placeholder runs, encodes the rows once per part through the 3a1 encoder, and ships every request as a mixed token/row prompt (`EmbedsPrompt`) to vLLM with prefix caching intact.
- Compaction is a real tool. Past a soft token delta the loop accepts only a lone `compact(summary)`, refuses other calls with plain results, and ends after five refusals. The finished segment is recorded under `compactions`, and the next prompt is the system message, the first user messages, and one user message holding the segment as a tree (ratio 1/10) with instructions and the summary. Without an AC model the summary alone carries over.
- Subagents receive the parent's current segment as a part (ratio 1/20) and return their final segment as a part; truncated tool outputs keep the full text as a part (ratio 1/40) nested with the parent context; semantic search returns extra passages as a part (ratio 1/20).
- Run results keep every segment, its prompt messages, spans and the AC identity. `unroll` walks segments and their children. `simulate_step`, `synthesize_agent` and `resume_from_run_result` make the channels testable without sampling.
- `ActivationContextStudyGenerator.items_from_run_results` converts runs to KL self-distillation items: the teacher expands each part by its kind, the student keeps the real AC prefix, the completion is the recorded turn. Text-only runs go through the 3a1 cut generator with recorded completions on whole-turn cuts. Training on these items is deferred to the recipe slices; the example builder refuses AC-bearing runs meanwhile.

## Apply to your source

The implementation is staged under [slice3b](slice3b/README.md). The [product patch](slice3b/changes.patch) and cards below are exact deltas against `/source`. The source itself has not been modified. Check [source hashes](slice3b/source_manifest.json) before applying; preserve unrelated changes. The payload probe stays IB-only.

## Validation

- CPU (tiny Qwen3.5 fixture, no engine, no sandbox): [channels](../../TMP/SLICE3B/cpu_checks/channels_cpu.py) passes in 9 s: compaction after three long outputs, a refused call, a lone `compact` restarting the segment (9 recorded steps, tree of 10 messages, 64 bf16 rows at ratio 1/10), a second compaction nesting the first tree verbatim, `unroll` over 3 segments, a step-mode subagent in both directions, a 100k-character output truncated below 21k with an 80k part nested with the parent segment, search with 3 visible passages and a 6-passage part, the serialize/deserialize/resume round trip with identical prefix, spans and rows, the text-only fallback, and the five-refusal cap. [Harvest](../../TMP/SLICE3B/cpu_checks/harvest_cpu.py) passes in 12 s: 4 items across compaction, search, subagent return and tool output from a serialized AC run, teachers without parts and with the full output once, weights validated, 4 cut items with recorded completions from a text-only run, and the AC trainer's example builder accepting every item.
- Payload probe ([probe.json](../../TMP/SLICE3B/node/AGENT_AC_TEST/probe.json), [probe_64.json](../../TMP/SLICE3B/node/AGENT_AC_TEST/probe_64.json); Qwen3.5-4B on one RTX PRO 6000, 2,048 nonzero rows per prompt, one sampled token): where the token-only and mixed forms share the same cache state, the full-length row tensor costs 1 to 3 percent of wall time at concurrency 8 to 64 and 120 to 240 ms per single request at 16k to 32k tokens (row serialization). Repeats hit the prefix cache on every cell except 32 x 50k, where 1.6M tokens exceed the KV budget. Host RSS peaked at 10.1 GB. The M0 gate passes; the vLLM plugin relaxations stay unimplemented.
- Engine geometry: prefix hits come in mamba-aligned blocks, 528 tokens with a bf16 KV cache and 1,056 with the harness default fp8 cache ([probe_short.json](../../TMP/SLICE3B/node/AGENT_AC_TEST/probe_short.json), [probe_decode.json](../../TMP/SLICE3B/node/AGENT_AC_TEST/probe_decode.json), [diagnostic](slice3b/probes/ac_cache_diag.py)). Prompts shorter than one block never hit, token-only or mixed. The test's cache assertion is block-aware.
- Node tests, untrained rows: simulated channels ([log](../../TMP/SLICE3B/node/AGENT_AC_TEST/pytest_simulated.log)) accept every mixed request (compaction 1,259 tokens with 391 row positions, repeat cached 1,056; a real 100k-character sandbox output 3,620 tokens / 251 rows, cached 3,168; BRIGHT search 2,161 tokens / 117 rows, cached 2,112; second compaction and subagent prompts below one block) and the model answers sensibly on untrained rows. With-AC rollouts ([log](../../TMP/SLICE3B/node/AGENT_AC_TEST/pytest_rerun.log), 300-token delta): seed 0 submitted after 8 turns with 2 compactions and 223 rows, seed 1 after 4 turns with 1 compaction; both answered 2441 (correct); 6 harvested items from 12 candidates. Text-only rollouts: both seeds submitted 2441 with one compaction each. Final clean run of the whole file ([log](../../TMP/SLICE3B/node/AGENT_AC_TEST/pytest_final.log)): 3 passed in 170 s. In that run the with-AC seed 0 hit `max_turns` after 12 turns with 3 compactions and 355 rows while seed 1 submitted 2441 after one compaction, and the harvest yielded 10 items from 16 candidates: with untrained rows the model can lose the thread after a compaction, which is exactly the behavior the deferred training targets.
- Limits: rows are untrained, so nothing here measures AC quality; the runs show mechanics only. The test rollouts are two seeds of one task. Parts nest deep copies of the parent segment, so serialized records grow quadratically with the parts in a segment (the row cache deduplicates compute). Training on AC-bearing runs is deliberately refused until the fixed-row trainer integration of the recipe slices.

## Review and differences from the plan

- Part primitives moved to `activation/common/ac_parts.py` to avoid an import cycle between the study generator and the agent dialect; `ac_model_utils` and `ac_model_study` re-export them. Parts carry a `kind` so the harvest expands each channel explicitly rather than by structure.
- The compact call's own turn records no tool step: a compacted segment holds one step fewer than the plan's test expected.
- The node found three test-side defects and no product defects: BRIGHT has labeled examples, not scorable tasks; the teardown must free adapters before bases; and the cache assertion needed the block geometry above. The probe gained a `--max-tokens` flag and `Agent.submit_probe(max_tokens=)` an override while diagnosing the geometry.
- The retained node `ac-fp4-probe` could not restart for 12 minutes (42 capacity failures in us-east-1a); its start request was cancelled by ID, the disk kept, and a temporary node `ac-slice3b-temp` ran the validation (base image plus frozen-lock update, us-east-2). The temporary node was torn down after every result was pulled locally; `ac-fp4-probe` remains stopped with its disk.
- Deferred, as planned: fixed-row training in the agent trainer and the AC training recipes; the vLLM plugin relaxations (not needed at the measured overhead); by-reference serialization of nested parent contexts.

## Exact product diffs

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title"><a href="slice3b/activation/common/ac_parts.py">activation/common/ac_parts.py</a></span>
    <span class="card-oneliner">New shared part primitives: schema, kinds, sentinel split, placeholder encoding, compaction instructions.</span>
    <span class="card-badge">Diff</span>
  </summary>

`activation/common/ac_parts.py`

```diff
diff --git a/activation/common/ac_parts.py b/activation/common/ac_parts.py
new file mode 100644
--- /dev/null
+++ b/activation/common/ac_parts.py
@@ -0,0 +1,89 @@
+"""
+Activation-context parts as data: the part schema, the sentinel the chat template renders in place of
+a part, and the split-and-tokenize step that turns a rendered template into token ids with placeholder
+runs. Shared by the AC model (side tokenization) and the agent dialect (target prompts) without either
+importing the other.
+"""
+from __future__ import annotations
+
+import typing as t
+
+if t.TYPE_CHECKING:
+    from transformers import PreTrainedTokenizerBase
+
+AC_PART_TYPE = "activation_context"
+PART_SENTINEL = "⁣ACPART{index}⁣"     # rendered in place of a part by the chat template, split out before tokenizing
+COMPACTION_INSTRUCTIONS = (
+    "The conversation so far has been compacted into the activation context above. Continue the task "
+    "from exactly where it left off, as if the full conversation were still in front of you."
+)
+
+
+def ac_part(messages: list[dict], ac_name: str, compression_target: float, kind: str | None = None) -> dict:
+    """A part; `kind` names the channel that produced it (compaction, subagent_prompt, subagent_return, tool_output, search, parent_context)."""
+    part = {"type": AC_PART_TYPE, "ac_name": ac_name, "compression_target": compression_target, "messages": messages}
+    if kind is not None:
+        part["kind"] = kind
+    return part
+
+
+def is_ac_part(part: object) -> bool:
+    return isinstance(part, dict) and part.get("type") == AC_PART_TYPE
+
+
+def direct_parts(messages: list[dict]) -> list[dict]:
+    """The activation_context parts of these messages, in order of appearance (not their descendants)."""
+    parts = []
+    for message in messages:
+        content = message.get("content")
+        if isinstance(content, list):
+            parts.extend(part for part in content if is_ac_part(part))
+    return parts
+
+
+def flatten_with_sentinels(messages: list[dict], parts: str = "sentinel") -> list[dict]:
+    """
+    Content lists joined into the text the chat template takes: text parts verbatim, activation_context
+    parts as a numbered sentinel (`parts="sentinel"`) or dropped (`parts="drop"`, for history that only
+    needs to render as some text). Other keys of a message (tool_calls, ...) are kept.
+    """
+    out = []
+    index = 0
+    for message in messages:
+        message = dict(message)
+        content = message.get("content")
+        if isinstance(content, list):
+            pieces = []
+            for part in content:
+                if is_ac_part(part):
+                    if parts == "sentinel":
+                        pieces.append(PART_SENTINEL.format(index=index))
+                        index += 1
+                elif isinstance(part, dict):
+                    pieces.append(part.get("text", ""))
+                else:
+                    pieces.append(str(part))
+            message["content"] = "".join(pieces)
+        out.append(message)
+    return out
+
+
+def encode_with_part_sentinels(tokenizer: "PreTrainedTokenizerBase", text: str, part_lengths: list[int], pad_id: int) -> tuple[list[int], list[tuple[int, int]]]:
+    """
+    Token ids of a rendered template whose part sentinels become `pad_id` runs of the given lengths, plus
+    the (start, end) span of each run. The pieces between sentinels are tokenized without special tokens,
+    as the template's own tokenization would; the caller writes rows over the spans.
+    """
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

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title"><a href="slice3b/activation/ac_model/ac_model_utils.py">activation/ac_model/ac_model_utils.py</a></span>
    <span class="card-oneliner">Re-export the shared primitives; tokenize_with_parts uses the common split.</span>
    <span class="card-badge">Diff</span>
  </summary>

`activation/ac_model/ac_model_utils.py`

```diff
diff --git a/activation/ac_model/ac_model_utils.py b/activation/ac_model/ac_model_utils.py
--- a/activation/ac_model/ac_model_utils.py
+++ b/activation/ac_model/ac_model_utils.py
@@ -21,8 +21,9 @@
 if t.TYPE_CHECKING:
     from transformers import PreTrainedTokenizerBase
 
-AC_PART_TYPE = "activation_context"
-PART_SENTINEL = "⁣ACPART{index}⁣"     # rendered in place of a part by the chat template, split out before tokenizing
+from activation.common.ac_parts import (   # the part schema and the sentinel split live in common; re-exported here
+    AC_PART_TYPE, PART_SENTINEL, direct_parts, encode_with_part_sentinels, flatten_with_sentinels, is_ac_part,
+)
 
 
 # --------------------------------------------------------------------------------------- parts and keys
@@ -31,20 +32,6 @@
     if isinstance(messages, str):
         return [{"role": "user", "content": messages}]
     return list(messages)
-
-
-def is_ac_part(part: object) -> bool:
-    return isinstance(part, dict) and part.get("type") == AC_PART_TYPE
-
-
-def direct_parts(messages: list[dict]) -> list[dict]:
-    """The activation_context parts of these messages, in order of appearance (not their descendants)."""
-    parts = []
-    for message in messages:
-        content = message.get("content")
-        if isinstance(content, list):
-            parts.extend(part for part in content if is_ac_part(part))
-    return parts
 
 
 def canonical_json(value: object) -> str:
@@ -73,41 +60,14 @@
     renders one sentinel string per part; the text is split at the sentinels and the pieces tokenized
     (no special tokens, as the template's own tokenization). The caller writes rows over the spans.
     """
-    flat: list[dict] = []
-    index = 0
-    for message in messages:
-        message = dict(message)
-        content = message.get("content")
-        if isinstance(content, list):
-            pieces = []
-            for part in content:
-                if is_ac_part(part):
-                    pieces.append(PART_SENTINEL.format(index=index))
-                    index += 1
-                elif isinstance(part, dict):
-                    pieces.append(part.get("text", ""))
-                else:
-                    pieces.append(str(part))
-            message["content"] = "".join(pieces)
-        flat.append(message)
+    flat = flatten_with_sentinels(messages, parts="sentinel")
+    index = sum(1 for _ in direct_parts(messages))
     assert index == len(part_lengths), f"{index} parts in the messages, {len(part_lengths)} lengths given"
     if not any(message.get("role") == "user" and not str(message.get("content") or "").strip().startswith("<tool_response>") for message in flat):
         flat.insert(0, {"role": "user", "content": ""})                    # Qwen templates refuse a conversation without a user query (a mid-trajectory segment)
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
+    return encode_with_part_sentinels(tokenizer, text, list(part_lengths), pad_id)
 
 
 def sinusoidal_positions(length: int, d: int, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
```

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title"><a href="slice3b/activation/ac_model/ac_model_study.py">activation/ac_model/ac_model_study.py</a></span>
    <span class="card-oneliner">Cut targets recorded per item; run-result harvesting into AC training items.</span>
    <span class="card-badge">Diff</span>
  </summary>

`activation/ac_model/ac_model_study.py`

```diff
diff --git a/activation/ac_model/ac_model_study.py b/activation/ac_model/ac_model_study.py
--- a/activation/ac_model/ac_model_study.py
+++ b/activation/ac_model/ac_model_study.py
@@ -20,6 +20,7 @@
 from __future__ import annotations
 
 import json
+import math
 from copy import deepcopy
 from dataclasses import replace
 import random
@@ -30,25 +31,19 @@
 from ..dataset.loaders.trajectory_utils import submit_answer_definition
 from ..dataset.dataset_utils import message_text, render_messages
 from .ac_model_training import ActivationContextTrainingItem
-from .ac_model_utils import AC_PART_TYPE
+from .ac_model_utils import AC_PART_TYPE, direct_parts, is_ac_part
+from ..common.ac_parts import COMPACTION_INSTRUCTIONS, ac_part   # shared with the agent loop (agent_tools)
 
 if t.TYPE_CHECKING:
+    from ..agent.agent_config import AgentRunResult
     from ..harness import HarnessRuntime
 
-COMPACTION_INSTRUCTIONS = (
-    "The conversation so far has been compacted into the activation context above. Continue the task "
-    "from exactly where it left off, as if the full conversation were still in front of you."
-)
 TRAJ_QA_SYSTEM_PROMPT = "You answer questions about agent trajectories (transcripts of an agent's reasoning, tool calls and tool results)."
 TRAJ_QA_INSTRUCTIONS = "Answer the question from the trajectories above (some may be unrelated). Call submit_answer with the answer."
 RAG_QA_SYSTEM_PROMPT = "You answer questions from the passages you are given."
 RAG_QA_INSTRUCTIONS = "Answer the question from the passages above (some may be unrelated). Call submit_answer with the answer."
 MESSAGE_TOKEN_OVERHEAD = 8                    # template tokens per message, on top of its text (estimate)
 MIN_COMPACTION_TOKENS = 1024                  # a trajectory shorter than this (before its completion) yields no compaction item
-
-
-def ac_part(messages: list[dict], ac_name: str, compression_target: float) -> dict:
-    return {"type": AC_PART_TYPE, "ac_name": ac_name, "compression_target": compression_target, "messages": messages}
 
 
 # (It can't be perfect since we are taking external trajectories, but we should not kneecap ourselves; e.g., use things like submit answer where possible, etc.).
@@ -66,6 +61,7 @@
         self.tokenizer = self.ac_model.target.tokenizer
         self.dialect = ModelDialect.for_tokenizer(self.tokenizer)
         self._token_cache: dict[str, int] = {}
+        self.harvest_report: dict = {}             # counts of the last items_from_run_results call
 
     # ------------------------------------------------------------------------------------------ helpers
     def count_tokens(self, text: str) -> int:
@@ -291,6 +287,7 @@
             info={"depth": len(segments), "thresholds": thresholds, "ratio": ratio, "in_context_tokens": sum(tokens[:len(in_context_tail)]),
                   "immediate": immediate, "visible_suffix_tokens": visible_tokens,
                   "delay_turns": target_position - immediate_target,
+                  "target_position": target_position, "mid_turn_cut": last_offset is not None,   # the completion's body message; run harvesting maps it back
                   "observed_terminal_answer": (document.trajectory_kwargs or {}).get("observed_terminal_answer")},
         )
         prefix = self.tokenizer.apply_chat_template(item.in_context_prefix, tools=tools, add_generation_prompt=True, tokenize=True)
@@ -311,6 +308,238 @@
             ids = self.tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=True, tokenize=True)
             return len(ids["input_ids"] if hasattr(ids, "keys") else ids)
         return max(0, count(anchor + visible) - count(anchor))
+
+    # ------------------------------------------------------------------------------------------ run-result harvesting
+    def items_from_run_results(
+        self,
+        runs: list["AgentRunResult"],
+        *,
+        weight_of: t.Callable[["AgentRunResult"], float] | None = None,
+        seed: int = 0,
+        completion_max_tokens: int | None = None,
+        text_run_kwargs: dict | None = None,
+        items_per_text_run: int = 1,
+    ) -> list[ActivationContextTrainingItem]:
+        """
+        Self-distillation items from agent run results: the target reading a run as plain text (teacher)
+        against the same target reading the run's real activation-context prompt (student), on the turn
+        the agent actually sampled. Every run recorded with this AC model yields one item per assistant
+        turn whose prompt holds a part (compaction trees, subagent prompts and returns, tool-output and
+        search parts); the run's compacted segments and every subagent beneath it are visited. A run
+        without an AC model is converted to a trajectory document and cut by the 3a1 generator
+        (`text_run_kwargs` override its defaults), with the recorded sampled tokens as the completion
+        where the cut lands on a whole turn. Weights multiply a KL and must be finite and nonnegative;
+        `weight_of(run)` defaults to the run's score, zero-weight runs are skipped. Records are not
+        mutated; `harvest_report` holds candidate/accepted/skip counts.
+        """
+        rng = random.Random(seed)
+        report = self.harvest_report = {"roots": len(runs), "runs": 0, "candidates": 0, "accepted": 0, "skipped": {}}
+        items: list[ActivationContextTrainingItem] = []
+        for root in runs:
+            weight = float(root.score if weight_of is None else weight_of(root))
+            if not math.isfinite(weight) or weight < 0:
+                raise ValueError("AC distillation item weights must be finite and nonnegative: the KL target has no sign to flip "
+                                 "(negative advantages belong to the policy-gradient path)")
+            if weight == 0.0:
+                self._harvest_skip("zero_weight")
+                continue
+            for source_key, run in self._rollout_sources(root):
+                report["runs"] += 1
+                if run.ac_model_name is None:
+                    for index in range(items_per_text_run):
+                        items += self._items_from_text_run(run, f"{source_key}.{index}", weight, rng, completion_max_tokens, text_run_kwargs)
+                elif run.ac_model_name == self.ac_name:
+                    items += self._items_from_ac_run(run, source_key, weight, completion_max_tokens)
+                else:
+                    raise ValueError(f"run recorded with AC model {run.ac_model_name!r}; this generator is bound to {self.ac_name!r}")
+        report["accepted"] = len(items)
+        return items
+
+    def _harvest_skip(self, reason: str) -> None:
+        skipped = self.harvest_report["skipped"]
+        skipped[reason] = skipped.get(reason, 0) + 1
+
+    def _rollout_sources(self, root: "AgentRunResult", key: str | None = None) -> t.Iterator[tuple[str, "AgentRunResult"]]:
+        """The run, then every subagent attached to any of its segments (recursively), each once with a path key."""
+        key = f"{root.agent_config.agent_name}:{root.seed}" if key is None else key
+        yield key, root
+        for segment_index, segment in enumerate(list(root.compactions) + [root]):
+            for child_index, child in enumerate(segment.subagent_results):
+                yield from self._rollout_sources(child, f"{key}/s{segment_index}c{child_index}")
+
+    def _run_tools(self, run: "AgentRunResult") -> list[dict] | None:
+        """The tool definitions the run's prompts were templated with (None under the inline rendering, whose listing is in the system prompt)."""
+        from ..agent.agent import Agent
+        agent = Agent(self.harness, run.agent_config)
+        agent.dialect = self.dialect
+        agent._prepare_tools()
+        return agent.template_tools
+
+    def _rollout_completion(self, recorded_ids: list[int], max_tokens: int | None = None) -> tuple[list[int], bool]:
+        """
+        Recorded sampled ids as completion fields: exactly one terminal end-of-turn sequence is stripped
+        (the trainer appends it again when `completion_complete`), the rest is kept verbatim; a cap marks
+        the completion incomplete. No cap by default: a run's turn is bounded by its own sampling max_tokens.
+        """
+        eot = self.ac_model.target.model_config.model_description.eot_token or ""
+        eot_ids = self.tokenizer.encode(eot, add_special_tokens=False) if eot else []
+        ids = list(recorded_ids)
+        complete = bool(eot_ids) and len(ids) >= len(eot_ids) and ids[-len(eot_ids):] == eot_ids
+        if complete:
+            ids = ids[:-len(eot_ids)]
+        if max_tokens is not None and len(ids) > max_tokens:
+            return ids[:max_tokens], False
+        return ids, complete
+
+    def _items_from_ac_run(self, run: "AgentRunResult", source_key: str, weight: float, completion_max_tokens: int | None) -> list[ActivationContextTrainingItem]:
+        """One item per assistant turn whose student prefix (its segment's real prompt plus the steps before it) holds a part."""
+        tools = self._run_tools(run)
+        items: list[ActivationContextTrainingItem] = []
+        for k, segment in enumerate(list(run.compactions) + [run]):
+            system = [message for message in segment.prompt_messages if message.get("role") == "system"]
+            base = [message for message in segment.prompt_messages if message.get("role") != "system"]
+            history: list[dict] = []
+            for t_index, step in enumerate(segment.trajectory):
+                if step.get("role") == "assistant":
+                    student = base + history
+                    self.harvest_report["candidates"] += 1
+                    parts = direct_parts(student)
+                    if not parts:
+                        self._harvest_skip("no_parts")
+                    else:
+                        ids, complete = self._rollout_completion(step.get("token_ids") or [], completion_max_tokens)
+                        if not ids:
+                            self._harvest_skip("empty_completion")
+                        else:
+                            items.append(ActivationContextTrainingItem(
+                                item_id=f"rollout:{source_key}:{k}:{t_index}", kind="compaction",
+                                in_context_prefix=system + self._expand_parts(student), ac_prefix=deepcopy(system + student),
+                                completion_text=self.tokenizer.decode(ids, clean_up_tokenization_spaces=False),
+                                completion_token_ids=ids, completion_complete=complete, tools=tools, weight=weight,
+                                dataset_id="agent_runs", doc_ids=[source_key],
+                                info={"source": "agent_run", "source_key": source_key, "segment": k, "turn": t_index, "score": run.score,
+                                      "channels": sorted({part.get("kind") or "unknown" for part in parts}), "completion_source": "recorded",
+                                      "ac_model_name": run.ac_model_name, "ac_model_version": run.ac_model_version},
+                            ))
+                history += deepcopy(step.get("messages") or [])
+        return items
+
+    def _expand_parts(self, messages: list[dict]) -> list[dict]:
+        """
+        The teacher's view of messages with parts: text only, each part expanded once by its channel. A
+        compaction tree becomes the earlier history it holds (its instructions/summary text is dropped); a
+        tool-output part replaces the truncated text with the full output; search extras follow the visible
+        passages; a subagent's segment (either direction) becomes a transcript. No message of the future is added.
+        """
+        out: list[dict] = []
+        for message in messages:
+            content = message.get("content")
+            if not isinstance(content, list):
+                out.append(dict(message))
+                continue
+            tree = next((part for part in content if is_ac_part(part) and part.get("kind") == "compaction"), None)
+            if tree is not None:
+                expanded = self._expand_tree(tree)
+                if out and expanded and out[-1] == expanded[0]:
+                    out.pop()                                                       # the tree starts with the task the prompt already shows
+                out.extend(expanded)
+                continue
+            pieces: list[str] = []
+            drop_next_text = False
+            for part in content:
+                if is_ac_part(part):
+                    kind = part.get("kind")
+                    inner = part.get("messages") or []
+                    if kind == "tool_output":
+                        pieces.append(next((m.get("content", "") for m in inner if m.get("role") == "tool"), ""))
+                        drop_next_text = True                                       # the truncated copy would repeat the output
+                    elif kind == "search":
+                        pieces.append("\n\n" + next((m.get("content", "") for m in inner if m.get("role") == "tool"), ""))
+                    elif kind == "subagent_return":
+                        pieces.append("Subagent transcript:\n" + render_messages(self._expand_parts(inner)) + "\n\n")
+                    else:                                                           # subagent_prompt, parent_context, untagged: a transcript
+                        pieces.append("Parent transcript:\n" + render_messages(self._expand_parts(inner)) + "\n\n")
+                elif isinstance(part, dict):
+                    if drop_next_text and part.get("type") == "text":
+                        drop_next_text = False
+                        continue
+                    pieces.append(part.get("text", ""))
+                else:
+                    pieces.append(str(part))
+            out.append(dict(message, content="".join(pieces)))
+        return out
+
+    def _expand_tree(self, tree: dict) -> list[dict]:
+        """The history a compaction tree holds, as text: its first user message (recursively) then the segment through the compact call."""
+        inner = list(tree.get("messages") or [])
+        if not inner:
+            return []
+        expanded = self._expand_parts([inner[0]]) + self._expand_parts(inner[1:])
+        last = expanded[-1] if expanded else None
+        if last is not None and last.get("role") == "assistant" and last.get("tool_calls"):
+            expanded.append({"role": "tool", "content": "Context compacted."})        # the compact call's result, so the transcript stays well formed
+        return expanded
+
+    def _rollout_document(self, run: "AgentRunResult", key: str, tools: list[dict] | None) -> tuple[DatasetDocument | None, list[tuple[int, int] | None]]:
+        """A text-only run as a trajectory document (task first, every segment's steps as text, compaction messages dropped) with each body message's (segment, step) origin."""
+        segments = list(run.compactions) + [run]
+        first = segments[0].prompt_messages
+        task = next((message for message in first if message.get("role") == "user"), None)
+        if task is None:
+            return None, []
+        system = next((message.get("content") for message in first if message.get("role") == "system"), None)
+        trajectory = self._expand_parts([task])
+        origins: list[tuple[int, int] | None] = []
+        for k, segment in enumerate(segments):
+            for t_index, step in enumerate(segment.trajectory):
+                expanded = self._expand_parts(step.get("messages") or [])
+                trajectory += expanded
+                origins += [(k, t_index)] * len(expanded)
+            if k < len(segments) - 1 and trajectory and trajectory[-1].get("role") == "assistant" and trajectory[-1].get("tool_calls"):
+                trajectory.append({"role": "tool", "content": "Context compacted."})
+                origins.append(None)
+        document = DatasetDocument(doc_id=key, dataset_id="agent_runs", trajectory=trajectory,
+                                   trajectory_kwargs={"tools": tools, "system_prompt": system, "answer": run.answer}, modality=DataModality.TRAJECTORY)
+        return document, origins
+
+    def _items_from_text_run(self, run: "AgentRunResult", source_key: str, weight: float, rng: random.Random,
+                             completion_max_tokens: int | None, text_run_kwargs: dict | None) -> list[ActivationContextTrainingItem]:
+        """A text-only run cut by the 3a1 generator; a whole-turn completion takes the recorded sampled tokens, a mid-turn cut keeps the re-encoded rest of the turn."""
+        tools = self._run_tools(run)
+        document, origins = self._rollout_document(run, source_key, tools)
+        self.harvest_report["candidates"] += 1
+        if document is None:
+            self._harvest_skip("no_task")
+            return []
+        kwargs = dict(depth_range=(1, 2), threshold_range_tokens=(8192, 32768), ratios_range=(1.0 / 8.0, 1.0 / 16.0), completion_max_tokens=512,
+                      max_total_tokens=72_000, min_trajectory_tokens=MIN_COMPACTION_TOKENS, max_post_compaction_tokens=8192,
+                      immediate_continuation_ratio=0.25) | dict(text_run_kwargs or {})
+        immediate = rng.random() < kwargs.pop("immediate_continuation_ratio")
+        item = self._compaction_item(document, "agent_runs", rng, 0, 0, kwargs["depth_range"], kwargs["threshold_range_tokens"],
+                                     kwargs["ratios_range"], kwargs["completion_max_tokens"], kwargs["max_total_tokens"],
+                                     kwargs["min_trajectory_tokens"], kwargs["max_post_compaction_tokens"], immediate)
+        if item is None:
+            self._harvest_skip("no_cut")
+            return []
+        item.item_id = f"rollout_text:{source_key}"
+        item.weight = weight
+        item.doc_ids = [source_key]
+        item.info.update({"source": "agent_run_text", "source_key": source_key, "score": run.score})
+        position = item.info.get("target_position")
+        origin = origins[position] if position is not None and position < len(origins) else None
+        if item.info.get("mid_turn_cut") or origin is None:
+            item.info["completion_source"] = "reencoded" if item.info.get("mid_turn_cut") else "reencoded_unmapped"
+            return [item]
+        segment = (list(run.compactions) + [run])[origin[0]]
+        step = segment.trajectory[origin[1]]
+        ids, complete = self._rollout_completion(step.get("token_ids") or [], completion_max_tokens)
+        if not ids:
+            self._harvest_skip("empty_completion")
+            return []
+        item.completion_token_ids, item.completion_complete = ids, complete
+        item.completion_text = self.tokenizer.decode(ids, clean_up_tokenization_spaces=False)
+        item.info["completion_source"] = "recorded"
+        return [item]
 
     # ------------------------------------------------------------------------------------------ trajectory QA
     def _study_examples(self, dataset_id: str, num_samples: int, seed: int, modality: DataModality, caching_id: str | None) -> list[DatasetQAExample]:
```

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title"><a href="slice3b/activation/harness/vllm_wrapper.py">activation/harness/vllm_wrapper.py</a></span>
    <span class="card-oneliner">EmbedsPrompt submission for mixed token/row prompts.</span>
    <span class="card-badge">Diff</span>
  </summary>

`activation/harness/vllm_wrapper.py`

```diff
diff --git a/activation/harness/vllm_wrapper.py b/activation/harness/vllm_wrapper.py
--- a/activation/harness/vllm_wrapper.py
+++ b/activation/harness/vllm_wrapper.py
@@ -19,6 +19,7 @@
 import torch
 import vllm
 from vllm.engine.arg_utils import AsyncEngineArgs
+from vllm.inputs import EmbedsPrompt
 from vllm.sampling_params import RequestOutputKind
 from vllm.v1.engine.async_llm import AsyncLLM
 
@@ -42,21 +43,26 @@
 
         self.engine: AsyncLLM = asyncio.run_coroutine_threadsafe(construct(), self.loop).result()
 
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
         assert final is not None, f"request {request_id} produced no output"
         return final
 
-    def submit(self, prompt_token_ids, sampling_params, request_id, lora_request=None) -> concurrent.futures.Future:
-        return asyncio.run_coroutine_threadsafe(
-            self._collect(prompt_token_ids, sampling_params, request_id, lora_request), self.loop,
-        )
+    def submit(self, prompt_token_ids, sampling_params, request_id, lora_request=None,
+               prompt_embeds=None, prompt_is_token_ids=None) -> concurrent.futures.Future:
+        if prompt_embeds is None:
+            prompt = vllm.TokensPrompt(prompt_token_ids=list(prompt_token_ids))
+        else:
+            # Mixed prompt: the engine embeds every position whose mask is True itself and reads the shipped row
+            # elsewhere. The tensor must be full length (vLLM 0.28 rejects a length mismatch); zero rows sit at
+            # token positions. Needs `enable_prompt_embeds` on the engine (LoadedModel.engine_to_device).
+            prompt = EmbedsPrompt(prompt_embeds=prompt_embeds, prompt_token_ids=list(prompt_token_ids),
+                                  prompt_is_token_ids=list(prompt_is_token_ids))
+        return asyncio.run_coroutine_threadsafe(self._collect(prompt, sampling_params, request_id, lora_request), self.loop)
 
     def run(self, coroutine):
         """Run one engine coroutine on this replica's loop and wait for it."""
@@ -188,10 +194,15 @@
         return list(ids)
 
     def submit(self, prompt_token_ids: list[int], sampling_params: vllm.SamplingParams, request_id: str | None = None,
-               replica: int = 0, lora_request=None) -> concurrent.futures.Future:
-        """One request on one replica; the future resolves to the final vllm.RequestOutput."""
+               replica: int = 0, lora_request=None, prompt_embeds=None, prompt_is_token_ids=None) -> concurrent.futures.Future:
+        """
+        One request on one replica; the future resolves to the final vllm.RequestOutput. With `prompt_embeds`
+        ([len(prompt_token_ids), d_model], the model dtype, on CPU) and `prompt_is_token_ids` (True where the
+        engine embeds the token id itself) the request is a mixed prompt carrying activation-context rows.
+        """
         request_id = request_id or uuid.uuid4().hex
-        return self.replicas[replica % len(self.replicas)].submit(prompt_token_ids, sampling_params, request_id, lora_request)
+        return self.replicas[replica % len(self.replicas)].submit(prompt_token_ids, sampling_params, request_id, lora_request,
+                                                                  prompt_embeds, prompt_is_token_ids)
 
     def chat(self, conversations: list, **chat_kwargs) -> list[vllm.RequestOutput]:
         """Same signature as vllm.LLM.chat: template every conversation, submit strided over the replicas, return in order."""
```

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title"><a href="slice3b/activation/harness/loaded_model.py">activation/harness/loaded_model.py</a></span>
    <span class="card-oneliner">enable_prompt_embeds when an AC model is registered; pass-through of rows.</span>
    <span class="card-badge">Diff</span>
  </summary>

`activation/harness/loaded_model.py`

```diff
diff --git a/activation/harness/loaded_model.py b/activation/harness/loaded_model.py
--- a/activation/harness/loaded_model.py
+++ b/activation/harness/loaded_model.py
@@ -261,6 +261,8 @@
                 reservation = self.harness.harness_config.ac_engine_memory_reservation
                 engine_kwargs["gpu_memory_utilization"] = round(engine_kwargs["gpu_memory_utilization"] - reservation, 3)
                 print(f"{self.model_config.model_name} - Engine memory share lowered by {reservation} for the AC side model.")
+            if self.harness.module_manager.ac_models:
+                engine_kwargs.setdefault("enable_prompt_embeds", True)   # AC rows travel as prompt embeddings (mixed prompts)
             self.model_config.engine_kwargs = engine_kwargs
             self.vllm_model = VLLMWrapper(tokenizer=self.tokenizer, **engine_kwargs)
             self.current_engine_device = device
@@ -402,6 +404,8 @@
         lora_name: str|None = None,
         chat_kwargs: dict|None = None,
         record_sampling: bool = False,
+        prompt_embeds: "torch.Tensor | None" = None,
+        prompt_is_token_ids: list[bool] | None = None,
     ) -> EngineChatOutput:
         """
         One request whose prompt the caller owns as token ids (the agent loop keeps its own prefix).
@@ -409,6 +413,9 @@
         agent_id) so its prefix cache serves the next turn, the adapter the module manager currently
         exposes for `lora_name` (None until an exchange), and with `record_sampling` the sampled token
         log-probs. Blocks the calling thread while the engine batches this request with everything in flight.
+        With `prompt_embeds` ([len(prompt_token_ids), d_model], the model dtype, on CPU) the positions whose
+        `prompt_is_token_ids` entry is False take their embedding from that tensor (activation-context rows);
+        every other position is embedded by the engine from its token id as usual.
         """
         self.engine_to_device(TARGET_DEVICE)
         sampling_params, _ = self._merged_chat_kwargs(chat_kwargs, seed)
@@ -417,7 +424,7 @@
         lora_request = self.harness.module_manager.engine_lora_request(lora_name) if lora_name else None
         engine = self.vllm_model
         future = engine.submit(list(prompt_token_ids), sampling_params, replica=VLLMWrapper.replica_for(agent_id, engine.world_size),
-                               lora_request=lora_request)
+                               lora_request=lora_request, prompt_embeds=prompt_embeds, prompt_is_token_ids=prompt_is_token_ids)
         return EngineChatOutput.from_request_output(future.result(), lora_name=lora_name if lora_request is not None else None)
 
 
```

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title"><a href="slice3b/activation/agent/agent_config.py">activation/agent/agent_config.py</a></span>
    <span class="card-oneliner">messages_input, ratios, budgets, per-step messages and spans, segments and AC identity on run results.</span>
    <span class="card-badge">Diff</span>
  </summary>

`activation/agent/agent_config.py`

```diff
diff --git a/activation/agent/agent_config.py b/activation/agent/agent_config.py
--- a/activation/agent/agent_config.py
+++ b/activation/agent/agent_config.py
@@ -20,7 +20,9 @@
     # Inputs
     system_prompt: str = ""                                   # System prompt.
     user_prompt: str = ""                                     # User prompt (the task).
-    ac_inputs: dict[str, object] = field(default_factory=dict)  # Activation-context inputs; carried as data in slice 1.
+    messages_input: list[dict] = field(default_factory=list)  # Dialect messages ahead of the task: content may be ordered text and
+                                                              # activation_context parts. The task (user_prompt) is appended as a trailing
+                                                              # text part of the last user message; empty = today's prompt.
     dataset_task: "DatasetTask | None" = None                 # Set if this corresponds to a specific dataset task.
     agent_name: str = "main" # Used to tag trajectories with their subagent.
 
@@ -36,15 +38,19 @@
     env_args: dict[str, str] | None = None                    # Extra `podman run` flags, e.g. {"--env": "PYTHONHASHSEED=0"}.
     env_memory_limit_mb: int | None = None                    # Sandbox memory; None = the harness default (agent_env_memory_limit_mb). Hard container limit where cgroups exist, ulimit -v at 85% per process always.
     tools: dict[str, tuple[type["AgentTool"], dict]] = field(default_factory=dict)  # name -> (class, constructor kwargs), added to the defaults.
-    enable_ac_communication: bool = False                     # Subagents exchange trajectories as activation context.
+    enable_ac_communication: bool = False                     # Subagents exchange segments as activation context (needs ac_model_name).
 
     # Budgets
     max_turns: int = 20
     max_tool_errors: int = 5
     max_duration: float = 600                                 # seconds
-    compaction_threshold_tokens: int = 32768                  # carried; compaction is slice 2. @AI: This is a soft threshold btw. Can be temporarily exceed by one turn. It's also a delta on top of the starting length (system prompt, <task or any prev compactions, ac vector> + delta of this much with absolute cap below).
-    absolute_trajectory_cap: int = 50_000 # @AI: Simply err if this is every reached.
-    ac_compaction_ratio: float|None = 1.0/16.0 # ~threshold*ratio-sized vector (slightly more due to nesting+text logic).
+    compaction_threshold_tokens: int = 32768                  # soft delta over the segment's start length (system, first messages, tree); exceeded by at most one turn, then compaction is due
+    absolute_trajectory_cap: int = 50_000                     # prefix length that ends the run with finish_reason "trajectory_cap"
+    # Activation-context ratios (rows per side token) of the four channels.
+    ac_compaction_ratio: float = 1.0 / 10.0                   # the finished segment as the tree part of the next prompt
+    ac_subagent_ratio: float = 1.0 / 20.0                     # parent -> child prompt, child -> parent result, and the nested parent context of the parts below
+    ac_tool_output_ratio: float = 1.0 / 40.0                  # a truncated tool output's full text
+    ac_search_ratio: float = 1.0 / 20.0                       # semantic search passages beyond top_k
 
     def serialize(self) -> dict:
         data = {key: value for key, value in self.__dict__.items() if key not in ("dataset_task", "tools")}
@@ -52,7 +58,7 @@
             name: {"class": f"{cls.__module__}:{cls.__qualname__}", "kwargs": _jsonable(kwargs)}
             for name, (cls, kwargs) in self.tools.items()
         }
-        data["ac_inputs"] = _jsonable(self.ac_inputs)
+        data["messages_input"] = _jsonable(self.messages_input)
         task = self.dataset_task
         data["dataset_task"] = None if task is None else {
             "task_id": task.task_id, "dataset_id": task.dataset_id, "reference_metrics_kind": str(task.reference_metrics_kind),
@@ -65,6 +71,7 @@
         """The dataset task is looked up in the harness when it holds the dataset, else rebuilt from the stored fields."""
         from activation.dataset import DatasetTask, DatasetTaskMetricsKind
         data = dict(data)
+        data.pop("ac_inputs", None)                                                   # rows written before slice 3b
         tools = {}
         for name, spec in (data.pop("tools", None) or {}).items():
             module_name, _, qualname = spec["class"].partition(":")
@@ -96,7 +103,9 @@
     content: str                                               # assistant text (tool-call blocks removed) or the joined tool outputs
     tool_calls: list[dict] = field(default_factory=list)       # [{"id", "name", "arguments"}]
     tool_call_results: list[str] = field(default_factory=list) # truncated outputs, same order as tool_calls
-    activations: dict[str, object] = field(default_factory=dict)  # ac_outputs of this step's tools, keyed by call id
+    messages: list[dict] = field(default_factory=list)         # the dialect messages this step appended, activation_context parts inline and in
+                                                               # order: one assistant message, the tool messages, or the nudge
+    ac_spans: list[dict] = field(default_factory=list)         # [{"start", "length"}] placeholder runs inside token_ids, one per part of `messages`
     # Token segment (slice 2): the prompt the model read is AgentRunResult.prompt_token_ids + every step's token_ids in order.
     token_ids: list[int] = field(default_factory=list)         # assistant: the sampled tokens verbatim; tool/user: the template's wrapper up to the next generation prompt
     logprobs: list[float] = field(default_factory=list)        # assistant only: the engine's log-prob of each sampled token (pi_old); empty when not recorded
@@ -110,16 +119,22 @@
     num_input_tokens: int = 0
     num_cached_input_tokens: int = 0
     num_output_tokens: int = 0
-    num_ac_input_bytes: int = 0
-    num_ac_input_tokens: int = 0
+    num_ac_parts: int = 0                                      # parts encoded for this agent's prompts (nested children not counted)
+    num_ac_rows: int = 0                                       # rows those parts occupy in the prompts
     duration: float = 0.0
-    trajectory: list[dict] = field(default_factory=list)       # TrajectoryStep dicts, in order (a new activation type per step)
+    trajectory: list[dict] = field(default_factory=list)       # TrajectoryStep dicts of the current segment, in order
     score: float = 0.0
     score_feedback: str | None = None
     subagent_results: list["AgentRunResult"] = field(default_factory=list)
-    finish_reason: str = ""                                    # submitted | max_turns | max_tool_errors | max_duration | no_tool_call | context_exceeded | error
+    compactions: list["AgentRunResult"] = field(default_factory=list)   # this agent's earlier segments, oldest first (finish_reason "compacted")
+    finish_reason: str = ""                                    # submitted | max_turns | max_tool_errors | max_duration | no_tool_call | context_exceeded
+                                                               # | trajectory_cap | compaction_refused | compacted (a segment) | simulated | error
     seed: int = 0
-    prompt_token_ids: list[int] = field(default_factory=list)  # the first turn's prompt (system + task + tools, templated once)
+    prompt_token_ids: list[int] = field(default_factory=list)  # the segment's first prompt (system + first messages + tools, templated once)
+    prompt_messages: list[dict] = field(default_factory=list)  # the dialect messages of that prompt, parts inline (a compaction tree lives here)
+    prompt_ac_spans: list[dict] = field(default_factory=list)  # [{"start", "length"}] placeholder runs inside prompt_token_ids
+    ac_model_name: str | None = None                           # the encoder that produced the rows, and its version at run time
+    ac_model_version: int | None = None
     source: str = "policy"                                     # "policy" | "hinted:<model>" | "oracle:<model>": who produced this run
     lora_name: str | None = None                               # the adapter the engine actually applied (None: the base model)
 
@@ -129,11 +144,13 @@
         return sum(len(step.get("tool_calls", [])) for step in self.trajectory if step.get("role") == "assistant")
 
     def serialize(self) -> dict:
-        data = {key: value for key, value in self.__dict__.items() if key not in ("agent_config", "subagent_results", "trajectory", "answer")}
+        data = {key: value for key, value in self.__dict__.items()
+                if key not in ("agent_config", "subagent_results", "compactions", "trajectory", "answer")}
         data["agent_config"] = self.agent_config.serialize()
         data["answer"] = _jsonable(self.answer)
         data["trajectory"] = _jsonable(self.trajectory)
         data["subagent_results"] = [result.serialize() for result in self.subagent_results]
+        data["compactions"] = [result.serialize() for result in self.compactions]
         return data
 
     @staticmethod
@@ -144,7 +161,8 @@
         config_data = data.pop("agent_config")
         config = agent_config if agent_config is not None else AgentConfig.deserialize(config_data, harness)
         children = [AgentRunResult.deserialize(child, harness) for child in data.pop("subagent_results", [])]
-        return AgentRunResult(agent_config=config, subagent_results=children, **data)
+        segments = [AgentRunResult.deserialize(segment, harness, agent_config=config) for segment in data.pop("compactions", [])]
+        return AgentRunResult(agent_config=config, subagent_results=children, compactions=segments, **data)
 
 
 def _jsonable(value):
```

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title"><a href="slice3b/activation/agent/agent_utils.py">activation/agent/agent_utils.py</a></span>
    <span class="card-oneliner">Prompt and continuation tokens with placeholder runs; tool content rendering; synthetic runs.</span>
    <span class="card-badge">Diff</span>
  </summary>

`activation/agent/agent_utils.py`

```diff
diff --git a/activation/agent/agent_utils.py b/activation/agent/agent_utils.py
--- a/activation/agent/agent_utils.py
+++ b/activation/agent/agent_utils.py
@@ -11,7 +11,8 @@
 once, then the agent appends its own sampled tokens verbatim and only the wrapper the template puts
 between a finished assistant turn and the next generation prompt (`continuation_token_ids`). The
 model therefore reads exactly the tokens it wrote, one trajectory is one training sequence, and
-prefix-cache hits are exact. `continuation_token_ids` renders the wrapper from the chat template with a
+prefix-cache hits are exact. Activation-context parts inside messages render as sentinels and become
+placeholder runs whose spans the agent fills with rows (`prompt_tokens`, `continuation_tokens`). `continuation_token_ids` renders the wrapper from the chat template with a
 sentinel in place of the assistant text, so it needs no assumptions about the template's wording. Note
 that templates re-render *history* in ways the model never saw while sampling (Qwen3 drops the empty
 `<think>` block of finished turns, Qwen3.5 once a user message follows); the segments keep the
@@ -25,8 +26,13 @@
 import uuid
 from dataclasses import dataclass, field
 
+from activation.common.ac_parts import encode_with_part_sentinels, flatten_with_sentinels
+
 if t.TYPE_CHECKING:
+    from .agent import Agent
+    from .agent_config import AgentConfig
     from .agent_tools import ToolCallResult
+    from activation.harness import HarnessRuntime
 
 PARSE_ERROR_TOOL = "parse_error"
 ASSISTANT_SENTINEL = "\u241f TRESSOIR_ASSISTANT_TURN \u241f"   # never in real text; marks where the sampled tokens sit in a rendering
@@ -189,7 +195,8 @@
         return message
 
     def tool_messages(self, calls: list[dict], results: list["ToolCallResult"]) -> list[dict]:
-        return [{"role": "tool", "content": result.output} for result in results]
+        from .agent_tools import tool_content
+        return [{"role": "tool", "content": tool_content(result)} for result in results]     # content lists: parts in order
 
 
 class InlineRendering(MessageRendering):
@@ -214,8 +221,13 @@
         return {"role": "assistant", "content": "\n\n".join([content] + blocks if content else blocks)}
 
     def tool_messages(self, calls: list[dict], results: list["ToolCallResult"]) -> list[dict]:
-        body = "\n".join(f"<tool_response>\n{result.output}\n</tool_response>" for result in results)
-        return [{"role": "user", "content": body}]
+        from .agent_tools import tool_content
+        content: list[dict] = []
+        for index, result in enumerate(results):
+            content.append({"type": "text", "text": ("\n" if index else "") + "<tool_response>\n"})
+            content.extend(tool_content(result))
+            content.append({"type": "text", "text": "\n</tool_response>"})
+        return [{"role": "user", "content": content}]
 
 
 # --------------------------------------------------------------------------------------- dialect
@@ -258,25 +270,43 @@
     # ------------------------------------------------------------------------------- tokens
     def prompt_token_ids(self, tokenizer, messages: list[dict], tools: list[dict] | None = None,
                          chat_template_kwargs: dict | None = None) -> list[int]:
-        """The first prompt of a conversation: the chat template, once, with the generation prompt open."""
+        """The first prompt of a conversation: the chat template, once, with the generation prompt open (text-only content)."""
         return _template_ids(tokenizer, messages, tools, True, chat_template_kwargs)
+
+    def prompt_tokens(self, tokenizer, messages: list[dict], tools: list[dict] | None = None, chat_template_kwargs: dict | None = None,
+                      part_lengths: list[int] = (), pad_id: int = 0) -> tuple[list[int], list[tuple[int, int]]]:
+        """
+        The first prompt of a conversation: the chat template, once, with the generation prompt open.
+        Every activation_context part in `messages` renders as a sentinel and becomes a run of
+        `part_lengths[k]` placeholder ids; the (start, end) of each run is returned for the caller to
+        write rows over.
+        """
+        text = _template_text(tokenizer, messages, tools, True, chat_template_kwargs, parts="sentinel")
+        return encode_with_part_sentinels(tokenizer, text, list(part_lengths), pad_id)
 
     def continuation_token_ids(self, tokenizer, messages_before: list[dict], new_messages: list[dict],
                                tools: list[dict] | None = None, chat_template_kwargs: dict | None = None) -> list[int]:
+        """The wrapper of `continuation_tokens` for text-only messages."""
+        return self.continuation_tokens(tokenizer, messages_before, new_messages, tools, chat_template_kwargs)[0]
+
+    def continuation_tokens(self, tokenizer, messages_before: list[dict], new_messages: list[dict], tools: list[dict] | None = None,
+                            chat_template_kwargs: dict | None = None, part_lengths: list[int] = (), pad_id: int = 0) -> tuple[list[int], list[tuple[int, int]]]:
         """
         The tokens the model reads between its own sampled text and its next turn: the end-of-turn
         marker, `new_messages` (tool results, or the nudge) and the next generation prompt, exactly as
         the chat template renders them. Rendered with a sentinel as the assistant content, so the
         wrapper is independent of what the model wrote. Starts with the end-of-turn token(s); the caller
-        drops the first one when the sample already ended with it.
-        """
-        history = list(messages_before) + [{"role": "assistant", "content": ASSISTANT_SENTINEL}] + list(new_messages)
-        text = _template_text(tokenizer, history, tools, True, chat_template_kwargs)
+        drops the first one when the sample already ended with it. Parts inside `new_messages` become
+        placeholder runs (spans relative to the returned wrapper); parts in `messages_before` sit
+        before the sentinel and only need to render as some text.
+        """
+        history = (flatten_with_sentinels(messages_before, parts="drop") + [{"role": "assistant", "content": ASSISTANT_SENTINEL}]
+                   + flatten_with_sentinels(new_messages, parts="sentinel"))
+        text = _template_text(tokenizer, history, tools, True, chat_template_kwargs, parts="drop")
         index = text.rfind(ASSISTANT_SENTINEL)
         if index < 0:
             raise ValueError("the chat template did not render the assistant content verbatim; cannot derive the turn wrapper")
-        wrapper = text[index + len(ASSISTANT_SENTINEL):]
-        return list(tokenizer.encode(wrapper, add_special_tokens=False))
+        return encode_with_part_sentinels(tokenizer, text[index + len(ASSISTANT_SENTINEL):], list(part_lengths), pad_id)
 
     @staticmethod
     def join_continuation(sampled_token_ids: list[int], wrapper_token_ids: list[int]) -> list[int]:
@@ -286,28 +316,55 @@
         return wrapper_token_ids
 
 
-def _template_text(tokenizer, messages: list[dict], tools, add_generation_prompt: bool, chat_template_kwargs: dict | None) -> str:
+def _template_text(tokenizer, messages: list[dict], tools, add_generation_prompt: bool, chat_template_kwargs: dict | None,
+                   parts: str = "drop") -> str:
     return tokenizer.apply_chat_template(
-        _flatten(messages), tools=tools, add_generation_prompt=add_generation_prompt, tokenize=False, **(chat_template_kwargs or {}),
+        flatten_with_sentinels(messages, parts=parts), tools=tools, add_generation_prompt=add_generation_prompt, tokenize=False,
+        **(chat_template_kwargs or {}),
     )
 
 
 def _template_ids(tokenizer, messages: list[dict], tools, add_generation_prompt: bool, chat_template_kwargs: dict | None) -> list[int]:
     ids = tokenizer.apply_chat_template(
-        _flatten(messages), tools=tools, add_generation_prompt=add_generation_prompt, tokenize=True, **(chat_template_kwargs or {}),
+        flatten_with_sentinels(messages, parts="drop"), tools=tools, add_generation_prompt=add_generation_prompt, tokenize=True,
+        **(chat_template_kwargs or {}),
     )
     if hasattr(ids, "keys"):                                                      # BatchEncoding / dict
         ids = ids["input_ids"]
     return list(ids)
 
 
-def _flatten(messages: list[dict]) -> list[dict]:
-    """Text-only content parts joined, the shape our templates take."""
-    out = []
-    for message in messages:
-        message = dict(message)
-        content = message.get("content")
-        if isinstance(content, list):
-            message["content"] = "".join(part.get("text", "") for part in content if isinstance(part, dict))
-        out.append(message)
-    return out
+# --------------------------------------------------------------------------------------- synthetic runs (tests)
+@dataclass
+class SyntheticTurn:
+    """One assistant turn of a synthetic run: its text, its calls, and the tool outputs to record (None leaves the calls pending)."""
+    content: str
+    calls: list[tuple[str, dict]] = field(default_factory=list)   # (tool name, arguments)
+    outputs: list[str] | None = None                               # tool outputs to record; None: `Agent.simulate_step()` executes the calls
+
+
+def synthesize_agent(harness: "HarnessRuntime", config: "AgentConfig", turns: list[SyntheticTurn], seed: int = 0) -> "Agent":
+    """
+    An agent whose run result holds `turns`, recorded through the same methods the loop uses with the
+    real dialect and tokenizer, so token segments, spans and rows are exactly what a rollout would
+    record. The sampled tokens of a turn are the rendered turn (text and call blocks) plus the model's
+    end-of-turn token. One-time test support: the main-tree AC agent test is the intended consumer.
+    """
+    from .agent import Agent
+    from .agent_tools import ToolCallResult
+    agent = Agent(harness, config, seed=seed)
+    agent.step_mode = True
+    agent.begin()
+    tokenizer = agent.loaded_model.tokenizer
+    eot = agent.loaded_model.model_config.model_description.eot_token or ""
+    for turn in turns:
+        calls = [{"id": uuid.uuid4().hex[:8], "name": name, "arguments": dict(arguments)} for name, arguments in turn.calls]
+        blocks = [agent.dialect.preferred_format.render(call["name"], call["arguments"]) for call in calls]
+        text = "\n\n".join([turn.content] + blocks if turn.content else blocks)
+        token_ids = list(tokenizer.encode(text + eot, add_special_tokens=False))
+        agent.record_assistant_turn(turn.content, calls, token_ids, [])
+        if calls and turn.outputs is not None:
+            assert len(turn.outputs) == len(calls), "one output per call"
+            results = [agent._finalize_result(call, ToolCallResult(output=output)) for call, output in zip(calls, turn.outputs)]
+            agent.append_tool_results(calls, results)
+    return agent
```

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title"><a href="slice3b/activation/agent/agent_tools.py">activation/agent/agent_tools.py</a></span>
    <span class="card-oneliner">Result content lists, truncation part, search extras, subagent both directions, compact tool.</span>
    <span class="card-badge">Diff</span>
  </summary>

`activation/agent/agent_tools.py`

```diff
diff --git a/activation/agent/agent_tools.py b/activation/agent/agent_tools.py
--- a/activation/agent/agent_tools.py
+++ b/activation/agent/agent_tools.py
@@ -1,8 +1,11 @@
 """
 Defines tool calls.
 All tool results are truncated to ~20000 chars as head[:10000] ... [truncated and written to
-/tmp/agent_outputs/<id>.txt] ... tail[-10000:] (the env writes these files). When truncated, the
-full output travels as an activation-context output (`ac_outputs["full_output"]`).
+/tmp/agent_outputs/<id>.txt] ... tail[-10000:] (the env writes these files). A result's content is an
+ordered list of text and activation_context parts: with an AC model a truncated output's full text
+rides as a part before the visible text, a subagent's final segment before its answer line, and the
+semantic-search passages beyond top_k after the visible ones. Every such part nests the parent's
+current segment as its own part, so the encoder compresses the content in the parent's context.
 """
 from __future__ import annotations
 
@@ -13,8 +16,10 @@
 from copy import deepcopy
 from dataclasses import dataclass, field
 
+from activation.common.ac_parts import COMPACTION_INSTRUCTIONS, ac_part
+
 if t.TYPE_CHECKING:
-    from .agent_config import AgentConfig
+    from .agent_config import AgentConfig, AgentRunResult
     from .agent_env import AgentEnv
     from activation.harness import HarnessRuntime
     from .agent import Agent
@@ -22,18 +27,33 @@
 TOOL_OUTPUT_LIMIT_CHARS = 20_000
 AC_OUTPUT_LIMIT_CHARS = 4 * TOOL_OUTPUT_LIMIT_CHARS   # kept as activation context (and written to the env file): 4x what the model sees, head and tail beyond
 TOOL_OUTPUT_DIR = "/tmp/agent_outputs"
+SEARCH_EXTRA_MAX = 20                                 # passages beyond top_k that ride as activation context: min(this, 2 * top_k)
+SUBAGENT_INTRO = "A parent agent spawned you. Its conversation so far is provided as activation context; your task follows.\n"
+
+
 @dataclass
 class ToolCallResult:
-    output: str                                          # what the model sees (already truncated)
+    output: str                                          # what the model sees as text (already truncated)
     is_error: bool = False
-    ac_outputs: dict[str, t.Any] = field(default_factory=dict)  # activation context outputs (ac name -> ac input)
     is_final: bool = False                               # submit_answer sets it
-
-
-def truncate_output(env: "AgentEnv | None", text: str, call_id: str, limit: int = TOOL_OUTPUT_LIMIT_CHARS) -> tuple[str, dict]:
-    """Head + tail of a long output; the full text goes to a file in the env and out as activation context."""
+    content: list[dict] | None = None                    # ordered content parts (text and activation_context) when the result carries
+                                                         # activation context; None means [text(output)]. Exactly one text part equals output.
+    compacted: bool = False                              # the compaction tool restarted the segment: no tool message follows this result
+
+
+def tool_content(result: ToolCallResult) -> list[dict]:
+    return result.content if result.content is not None else [{"type": "text", "text": result.output}]
+
+
+def truncate_output(env: "AgentEnv | None", text: str, call_id: str, limit: int = TOOL_OUTPUT_LIMIT_CHARS,
+                    agent: "Agent | None" = None) -> tuple[str, dict | None]:
+    """
+    Head + tail of a long output; the full text goes to a file in the env and, with an AC model on the
+    agent, into an activation_context part (the full text as a tool message, nested with the parent's
+    current segment). Returns (visible text, part or None).
+    """
     if len(text) <= limit:
-        return text, {}
+        return text, None
     if len(text) > AC_OUTPUT_LIMIT_CHARS:
         keep = AC_OUTPUT_LIMIT_CHARS // 2
         text = f"{text[:keep]}\n... [{len(text) - AC_OUTPUT_LIMIT_CHARS} chars dropped] ...\n{text[-keep:]}"
@@ -42,7 +62,20 @@
     half = limit // 2
     note = f"full output written to {path}" if written else "full output kept as activation context"
     truncated = f"{text[:half]}\n... [truncated {len(text) - limit} chars; {note}] ...\n{text[-half:]}"
-    return truncated, {"full_output": text}
+    if agent is None or agent.ac_model is None:
+        return truncated, None
+    config = agent.agent_config
+    part = ac_part([{"role": "user", "content": [agent.context_part(config.ac_subagent_ratio)]}, {"role": "tool", "content": text}],
+                   config.ac_model_name, config.ac_tool_output_ratio, kind="tool_output")
+    return truncated, part
+
+
+def segment_messages_of(result: "AgentRunResult") -> list[dict]:
+    """A run's final segment as dialect messages without the system message: prompt messages, then each step's messages."""
+    messages = [message for message in result.prompt_messages if message.get("role") != "system"]
+    for step in result.trajectory:
+        messages.extend(step.get("messages") or [])
+    return deepcopy(messages)
 
 
 class AgentTool:
@@ -137,10 +170,17 @@
         with ThreadPoolExecutor(max_workers=min(8, len(nested))) as pool:
             results = list(pool.map(self.agent.execute_tool_call, nested))
         outputs = [f"[{call['name']}] {result.output}" for call, result in zip(nested, results)]
-        ac_outputs = {call["id"]: result.ac_outputs for call, result in zip(nested, results) if result.ac_outputs}
+        content: list[dict] = []
+        for call, result, labeled in zip(nested, results, outputs):
+            for item in tool_content(result):                                            # each call's parts in its own order
+                content.append({"type": "text", "text": labeled + "\n\n"} if item.get("type") == "text" else item)
+        if content and content[-1].get("type") == "text":
+            content[-1] = {"type": "text", "text": content[-1]["text"].rstrip("\n")}
+        has_parts = any(result.content is not None for result in results)
         return ToolCallResult(
             output="\n\n".join(outputs), is_error=any(result.is_error for result in results),
-            ac_outputs=ac_outputs, is_final=any(result.is_final for result in results),
+            is_final=any(result.is_final for result in results), compacted=any(result.compacted for result in results),
+            content=content if has_parts else None,
         )
 
 
@@ -169,26 +209,41 @@
             return ToolCallResult(output=f"No searchable corpus is registered for this task ({self.dataset_id}).", is_error=True)
         index = self.harness.dataset_manager._get_or_create_index(self.dataset_id)
         index.build_bm25_index()
-        chunks = index.bm25_query_many_frozen([query], top_k=int(top_k or self.top_k))[0]
+        k = int(top_k or self.top_k)
+        config = self.agent.agent_config
+        extra = min(SEARCH_EXTRA_MAX, 2 * k) if self.agent.ac_model is not None else 0    # the model sees top_k; the next `extra` ride as rows
+        chunks = index.bm25_query_many_frozen([query], top_k=k + extra)[0]
         if not chunks:
             return ToolCallResult(output="No passage matched the query.")
-        blocks = [f"[{chunk.chunk_id}]\n{index.get_chunk_section(chunk)[:self.max_chars]}" for chunk in chunks]
-        return ToolCallResult(output="\n\n".join(blocks), ac_outputs={"chunk_ids": [chunk.chunk_id for chunk in chunks]})
+
+        def blocks(subset) -> str:
+            return "\n\n".join(f"[{chunk.chunk_id}]\n{index.get_chunk_section(chunk)[:self.max_chars]}" for chunk in subset)
+
+        output = blocks(chunks[:k])
+        content = None
+        if chunks[k:]:
+            part = ac_part([{"role": "user", "content": [self.agent.context_part(config.ac_subagent_ratio)]},
+                            {"role": "tool", "content": blocks(chunks[k:])}], config.ac_model_name, config.ac_search_ratio, kind="search")
+            content = [{"type": "text", "text": output}, part]                                # text first, the extra passages after
+        return ToolCallResult(output=output, content=content)
 
 
 class SubagentTool(AgentTool):
     """
-    Simple in/out without a persistent handle. Runs in the same env as the caller. Gets the caller's
-    current trajectory as activation context and returns its own trajectory as activation context
-    (unless AC communication is disabled); its result is registered in the caller's subagent results.
-    Some subagents are general-purpose, others specific: all are registered as tools under their own
-    name, with `base_config` and `extra_description` as constructor kwargs.
+    Simple in/out without a persistent handle. Runs in the same env as the caller. With AC
+    communication the child's first user message carries the parent's current segment as a part
+    (intro text, part, task) and the child's final segment comes back as a part before the answer
+    line; the child's own compaction tree rides inside that segment. The result is registered in the
+    caller's subagent results. Some subagents are general-purpose, others specific: all are registered
+    as tools under their own name, with `base_config` and `extra_description` as constructor kwargs.
+    In step mode (simulation) the child is created with a dummy submitted answer and does not run.
     """
     parameters = {
         "type": "object",
         "properties": {"task": {"type": "string", "description": "A self-contained description of what the subagent should do and return."}},
         "required": ["task"],
     }
+    SHARED_FIELDS = ("ac_model_name", "enable_ac_communication", "ac_compaction_ratio", "ac_subagent_ratio", "ac_tool_output_ratio", "ac_search_ratio")
 
     def __init__(self, harness, agent, base_config: "AgentConfig", extra_description: str = ""):
         super().__init__(harness, agent)
@@ -199,59 +254,67 @@
     def execute(self, task: str) -> ToolCallResult:
         from .agent import Agent
         parent = self.agent
+        parent_config = parent.agent_config
         subagent_config = deepcopy(self.base_config)
         if subagent_config.agent_name == "main":
             subagent_config.agent_name = "general_subagent"
         subagent_config.user_prompt = task
-        share_ac = parent.agent_config.enable_ac_communication
-        
+        share_ac = parent_config.enable_ac_communication and parent.ac_model is not None
         if share_ac:
-            subagent_config.ac_inputs = dict(subagent_config.ac_inputs) | {"caller_trajectory": deepcopy(parent.run_results.trajectory)}
+            for name in self.SHARED_FIELDS:
+                setattr(subagent_config, name, getattr(parent_config, name))
+            subagent_config.messages_input = [{"role": "user", "content": [
+                {"type": "text", "text": SUBAGENT_INTRO},
+                ac_part(deepcopy(parent.segment_messages()), parent_config.ac_model_name, parent_config.ac_subagent_ratio, kind="subagent_prompt"),   # the parent's segment up to and including this delegating turn
+            ]}]                                                                # the child appends the task as the trailing text part
         subagent = Agent(
             harness=self.harness,
             agent_config=subagent_config,
-            agent_env=parent.agent_env,
+            agent_env=parent._env if parent.step_mode else parent.agent_env,   # step mode never starts a sandbox
             parent_agent=parent,
             reporter=parent.reporter,
             seed=parent.seed,
         )
-        result = subagent.run()
+        subagent.step_mode = parent.step_mode
+        result = subagent.simulated_run() if parent.step_mode else subagent.run()
         with parent.lock:
             parent.run_results.subagent_results.append(result)
         output = (f"Subagent finished ({result.finish_reason}, {result.num_turns} turns). "
                   f"Answer: {result.answer if result.answer is not None else '(none)'}")
-        ac_outputs = {"trajectory": result.trajectory} if share_ac else {}
-        return ToolCallResult(output=output, is_error=result.finish_reason == "error", ac_outputs=ac_outputs)
-
+        content = None
+        if share_ac:
+            final_segment = segment_messages_of(result)                         # its first user messages (tree inside, if any) + every step
+            content = [ac_part(final_segment, parent_config.ac_model_name, parent_config.ac_subagent_ratio, kind="subagent_return"),
+                       {"type": "text", "text": output}]
+        return ToolCallResult(output=output, is_error=result.finish_reason == "error", content=content)
 
 
 class CompactionTool(AgentTool):
     """
-    Explicit compaction tool.
-    In the absense of AC mode, just produces a string and moves on.
-    In AC mode, produces a string and moves the trajectory to a compaction list passed in as AC.
-    Has a side-effect of transforming the agent result (that may happen here in the agent class):
-    - The current trajectory effectively becomes a subagent_trajectories = [current trajectory].
-    - The new trajectory starts with initial user message and AC updated, but likely not at the config-level though (or maybe it should be, I don't know).
-    (It's the responsibility of trajectory selectors to unroll. New logic: auto-unroll any "main" agent trajectory, however deeply nested).
-
-    The compaction AC input is basically a kind of tree or linked list of nested trajectories:
-    AC {
-        AC {
-            ...
-            <2nd to last trajectory + text summary>,
-        } <- recursive.
-        <most recent trajectory + text summary>,
-    }, text summary <- last text summary visible whether AC is activated or not.
-    Caching should prevent repeated calls, but this happens infrequently enough that that might be fine in some cases.
-    """
-    def __init__(self, harness, agent, base_config: "AgentConfig"):
-        pass
+    Compacts the conversation into a new segment. With an AC model the finished segment becomes the
+    recursive tree part of the new prompt and the summary follows it as text; without one only the
+    summary carries over. The finished segment is recorded under the run's `compactions`
+    (finish_reason "compacted"). The loop demands a lone `compact` call once the segment's soft
+    threshold is reached (Agent.COMPACTION_DEMAND) and refuses every other call until then.
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
+
 
 DEFAULT_TOOLS: dict[str, tuple[type[AgentTool], dict]] = {
     ShellTool.name: (ShellTool, {}),
     PythonTool.name: (PythonTool, {}),
     ParallelCallTool.name: (ParallelCallTool, {}),
     SubmitAnswerTool.name: (SubmitAnswerTool, {}),
-    CompactionTool.name: (CompactionTool, {})
+    CompactionTool.name: (CompactionTool, {}),
 }
```

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title"><a href="slice3b/activation/agent/agent.py">activation/agent/agent.py</a></span>
    <span class="card-oneliner">Segments, rows, mixed-prompt requests, compaction protocol, budgets, simulate_step, resume.</span>
    <span class="card-badge">Diff</span>
  </summary>

`activation/agent/agent.py`

```diff
diff --git a/activation/agent/agent.py b/activation/agent/agent.py
--- a/activation/agent/agent.py
+++ b/activation/agent/agent.py
@@ -8,31 +8,46 @@
 that the agent appends its own sampled tokens verbatim and the dialect's wrapper for the tool results
 or the nudge, so the model sees exactly what it wrote, every step records its token segment, and one
 trajectory is one training sequence (agent_training).
+
+Activation context (slice 3b): messages may carry activation_context parts (a compaction tree, the
+parent's segment for a subagent, a large tool output, extra search passages). A part renders as a run
+of placeholder tokens in the prefix; its rows come from the AC model once (the rollout queue and cache
+batch and deduplicate them), are cast to the target dtype and kept for the run, and every request
+ships them as prompt embeddings next to the token ids. A run is a sequence of segments: the soft
+threshold makes compaction due, the model calls `compact(summary)` alone, the finished segment is
+recorded under `compactions` and the next one starts from its tree (or the summary alone without an
+AC model).
 """
 from __future__ import annotations
 
-import json
 import threading
 import time
 import traceback
 import typing as t
 import uuid
+from copy import deepcopy
 from dataclasses import asdict
+
+import torch
+
+from activation.common.ac_parts import COMPACTION_INSTRUCTIONS, ac_part, direct_parts
 
 from .agent_config import AgentConfig, AgentRunResult, TrajectoryStep
 from .agent_env import AgentEnv
-from .agent_tools import DEFAULT_TOOLS, AgentTool, ToolCallResult, truncate_output
+from .agent_tools import DEFAULT_TOOLS, AgentTool, CompactionTool, ToolCallResult, tool_content, truncate_output
 from .agent_utils import PARSE_ERROR_TOOL, ModelDialect, parameter_types_of
 
 MAX_CONSECUTIVE_NO_TOOL_TURNS = 3   # nudged after each; the run ends with no_tool_call at the third in a row
+MAX_COMPACTION_REFUSALS = 5         # turns that ignore a due compaction before the run ends with compaction_refused
 
 if t.TYPE_CHECKING:
     from activation.harness import HarnessRuntime
     from .rollout_reporter import RolloutReporter
 
 
-
 class Agent:
+    COMPACTION_DEMAND = "The context threshold is reached: call compact(summary) now, alone, before any other tool."
+
     def __init__(
         self,
         harness: "HarnessRuntime",
@@ -46,16 +61,30 @@
         self.agent_config = agent_config
         self.parent_agent = parent_agent
         self.reporter = reporter
-        self.dialect = ModelDialect()   # replaced by the model's own dialect when run() starts
+        self.dialect = ModelDialect()   # replaced by the model's own dialect when begin() runs
         self.seed = seed
         self.owns_env = agent_env is None
         self._env = agent_env
         self.run_results = AgentRunResult(agent_config=agent_config, seed=seed)
         self.agent_id = uuid.uuid4().hex  # Pins the agent to one engine replica, so its prefix cache serves the next turn.
         self.lock = threading.Lock()      # Parallel subagent result appends.
-        self.messages: list[dict] = []
-        self.prefix: list[int] = []        # the token prompt of the next turn
+        self.messages: list[dict] = []    # dialect messages of the current segment, parts inline
+        self.prefix: list[int] = []       # the token prompt of the next turn (placeholder ids at part positions)
+        self.spans: list[tuple[int, int]] = []   # (start, end) of every part in the prefix, in order
+        self.rows: list[torch.Tensor] = []       # the rows of those parts, target dtype on CPU, kept for the run (identical bytes each turn)
+        self.segment_start = 0            # len(prompt_token_ids) of the current segment: the compaction delta counts from here
+        self.compaction_due = False
+        self.compaction_refusals = 0
+        self.no_tool_turns = 0
+        self.errors = 0
+        self.last_sampled: list[int] = [] # the previous assistant turn's tokens, for join_continuation
+        self.step_mode = False            # simulation: subagents do not run, nothing is submitted
+        self.started = False
         self.tools: dict[str, AgentTool] = {}
+        self.tool_definitions: list[dict] = []
+        self.parameter_types: dict = {}
+        self.template_tools: list[dict] | None = None
+        self.template_kwargs: dict = {}
         self.finished = False
         self._initialize_tools()
 
@@ -88,98 +117,277 @@
         assert config.model_name in self.harness.loaded_models, f"unknown model {config.model_name!r}"
         return self.harness.loaded_models[config.model_name]
 
+    # ----------------------------------------------------------------------------- activation context
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
+        return ac_part(deepcopy(self.segment_messages()), self.agent_config.ac_model_name, ratio, kind="parent_context")
+
+    def _pad_id(self) -> int:
+        tokenizer = self.loaded_model.tokenizer
+        return tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (tokenizer.eos_token_id or 0)
+
+    def _part_lengths(self, messages: list[dict]) -> list[int]:
+        parts = direct_parts(messages)
+        if not parts:
+            return []
+        ac_model = self.ac_model
+        assert ac_model is not None, "activation_context parts need agent_config.ac_model_name"
+        return [ac_model.part_view_rows(part["messages"], part.get("compression_target")) for part in parts]
+
+    def _encode_parts(self, messages: list[dict]) -> list[torch.Tensor]:
+        """Rows of the parts in `messages`, in order, through the rollout queue (children first, cached), cast once to the target dtype."""
+        parts = direct_parts(messages)
+        if not parts:
+            return []
+        ac_model = self.ac_model
+        assert ac_model is not None, "activation_context parts need agent_config.ac_model_name"
+        futures = [ac_model.encode_async(part["messages"], part.get("compression_target")) for part in parts]
+        rows = [future.result().detach().to(dtype=self.row_dtype).cpu().contiguous() for future in futures]
+        self.run_results.num_ac_parts += len(rows)
+        self.run_results.num_ac_rows += sum(int(r.shape[0]) for r in rows)
+        return rows
+
+    # ----------------------------------------------------------------------------- segments
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
+        if self.started:
+            return
+        config, results = self.agent_config, self.run_results
+        self.dialect = ModelDialect.for_tokenizer(self.loaded_model.tokenizer)
+        self._prepare_tools()
+        results.ac_model_name = config.ac_model_name
+        results.ac_model_version = self.ac_model.version if self.ac_model is not None else None
+        system = self.dialect.rendering.system_prompt(config.system_prompt or "", self.tool_definitions)
+        self._begin_segment(([{"role": "system", "content": system}] if system else []) + self.first_user_messages())
+        self.started = True
+
+    def _prepare_tools(self) -> None:
+        self.tool_definitions = self._tool_definitions()
+        self.parameter_types = parameter_types_of(self.tool_definitions)
+        self.template_tools = self.dialect.rendering.template_tools(self.tool_definitions)
+        self.template_kwargs = self.loaded_model.chat_template_kwargs(self.agent_config.call_kwargs)
+
+    def _begin_segment(self, messages: list[dict]) -> None:
+        """A fresh prompt for a new segment (run start, or after a compaction): rows for its parts, a new start length."""
+        results = self.run_results
+        lengths = self._part_lengths(messages)
+        ids, spans = self.dialect.prompt_tokens(self.loaded_model.tokenizer, messages, self.template_tools, self.template_kwargs,
+                                                lengths, self._pad_id())
+        rows = self._encode_parts(messages)
+        assert [int(r.shape[0]) for r in rows] == [end - start for start, end in spans], "placeholder runs differ from the encoded rows"
+        self.messages = list(messages)
+        self.prefix, self.spans, self.rows = list(ids), list(spans), rows
+        self.segment_start = len(ids)
+        self.last_sampled = []
+        results.prompt_token_ids, results.prompt_messages = list(ids), deepcopy(messages)
+        results.prompt_ac_spans = [{"start": start, "length": end - start} for start, end in spans]
+        self._after_append()
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
+            tree = ac_part(deepcopy([inner] + segment), config.ac_model_name, config.ac_compaction_ratio, kind="compaction")
+            after = {"role": "user", "content": [tree, {"type": "text", "text": "\n\n" + text}]}
+        else:
+            after = {"role": "user", "content": "The conversation so far was compacted.\n\n" + text}
+        system = [message for message in self.messages if message.get("role") == "system"]
+        self._begin_segment(system + self.first_user_messages() + [after])
+        self.compaction_due, self.compaction_refusals = False, 0
+
+    # ----------------------------------------------------------------------------- appending steps
+    def record_assistant_turn(self, content: str, calls: list[dict], token_ids: list[int], logprobs: list[float]) -> None:
+        message = self.dialect.rendering.assistant_message(content, calls)
+        step = TrajectoryStep(role="assistant", content=content, tool_calls=calls, messages=[message], token_ids=list(token_ids),
+                              logprobs=list(logprobs))
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
+        """The template's wrapper between the sampled turn and the next generation prompt, with placeholder runs for the new parts."""
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
+    # ----------------------------------------------------------------------------- requests
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
+    def _submit(self, chat_kwargs: dict | None = None):
+        config = self.agent_config
+        prefix, embeds, mask = self.prepare_request()
+        return self.loaded_model.engine_submit_tokens(
+            prefix, seed=self.seed + self.run_results.num_turns, agent_id=self.agent_id, lora_name=config.lora_name,
+            chat_kwargs=config.call_kwargs if chat_kwargs is None else chat_kwargs, record_sampling=config.record_sampling,
+            prompt_embeds=embeds, prompt_is_token_ids=mask,
+        )
+
+    def submit_probe(self, max_tokens: int | None = None):
+        """Submit the prepared request once without recording anything: the engine-interface check of the AC agent test. `max_tokens` overrides the sampling length."""
+        chat_kwargs = None
+        if max_tokens is not None:
+            chat_kwargs = dict(self.agent_config.call_kwargs or {})
+            chat_kwargs["sampling_params"] = dict(chat_kwargs.get("sampling_params") or {}) | {"max_tokens": max_tokens}
+        return self._submit(chat_kwargs)
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
+
     # ----------------------------------------------------------------------------- run
     def run(self) -> AgentRunResult:
         """
         Simple in/out run. Populates run_results. Never raises for a model or tool failure: the run
-        ends with finish_reason "error" and the traceback in score_feedback. Compaction is for later.
+        ends with finish_reason "error" and the traceback in score_feedback. Compaction restarts the
+        segment in place; the earlier segments are in run_results.compactions.
         """
         config, results = self.agent_config, self.run_results
         start = time.time()
-        self.dialect = ModelDialect.for_tokenizer(self.loaded_model.tokenizer)
-        rendering = self.dialect.rendering
-        tool_definitions = self._tool_definitions()
-        parameter_types = parameter_types_of(tool_definitions)
-        template_tools = rendering.template_tools(tool_definitions)
-        self.messages = []
-        system_prompt = rendering.system_prompt(config.system_prompt or "", tool_definitions)
-        if system_prompt:
-            self.messages.append({"role": "system", "content": system_prompt})
-        self.messages.append({"role": "user", "content": config.user_prompt})
-        self._count_ac_inputs()
-        if self.reporter is not None:
-            self.reporter.report_agent_start(self)
-        no_tool_turns = 0
-        errors = 0
         try:
-            loaded_model = self.loaded_model
-            tokenizer = loaded_model.tokenizer
-            template_kwargs = loaded_model.chat_template_kwargs(config.call_kwargs)
-            self.prefix = self.dialect.prompt_token_ids(tokenizer, self.messages, template_tools, template_kwargs)
-            results.prompt_token_ids = list(self.prefix)
+            self.begin()
+            if self.reporter is not None:
+                self.reporter.report_agent_start(self)
+            self._check_engine_context()
             while True:
+                if results.finish_reason:                                                        # trajectory_cap, set by an append
+                    break
                 if results.num_turns >= config.max_turns:
                     results.finish_reason = "max_turns"
                     break
                 if time.time() - start > config.max_duration:
                     results.finish_reason = "max_duration"
                     break
-                output = loaded_model.engine_submit_tokens(
-                    self.prefix, seed=self.seed + results.num_turns, agent_id=self.agent_id, lora_name=config.lora_name,
-                    chat_kwargs=config.call_kwargs, record_sampling=config.record_sampling,
-                )
+                output = self._submit()
                 if results.num_turns == 0:
                     results.lora_name = output.lora_name
                 results.num_turns += 1
                 results.num_input_tokens += output.prompt_token_count
                 results.num_cached_input_tokens += output.cached_prompt_token_count
                 results.num_output_tokens += output.output_token_count
-                content, calls = self.dialect.parse(output.text, parameter_types)
-                messages_before = list(self.messages)
-                self.messages.append(rendering.assistant_message(content, calls))
-                sampled = list(output.token_ids)
-                step = TrajectoryStep(role="assistant", content=content, tool_calls=calls, token_ids=sampled, logprobs=list(output.logprobs or []))
-                self.prefix += sampled
+                content, calls = self.dialect.parse(output.text, self.parameter_types)
+                self.record_assistant_turn(content, calls, list(output.token_ids), list(output.logprobs or []))
                 if not calls:
-                    # A turn without a call gets the nudge; several in a row end the run.
-                    results.trajectory.append(asdict(step))
-                    no_tool_turns += 1
-                    if no_tool_turns >= MAX_CONSECUTIVE_NO_TOOL_TURNS:
+                    # A turn without a call gets the nudge (the compaction demand once due); several in a row end the run.
+                    self.no_tool_turns += 1
+                    if self.no_tool_turns >= MAX_CONSECUTIVE_NO_TOOL_TURNS:
                         results.finish_reason = "no_tool_call"
                         break
-                    nudge = {"role": "user", "content": self.dialect.nudge_message}
-                    wrapper = self._continuation(tokenizer, messages_before, [nudge], template_tools, template_kwargs, sampled)
-                    self.messages.append(nudge)
-                    results.trajectory.append(asdict(TrajectoryStep(role="user", content=self.dialect.nudge_message, token_ids=wrapper)))
-                    self.prefix += wrapper
+                    self.append_user_message(self.COMPACTION_DEMAND if self.compaction_due else self.dialect.nudge_message)
+                    if self.compaction_due and self._refused():
+                        break
                     self._report_step()
                     continue
-                no_tool_turns = 0
+                self.no_tool_turns = 0
+                if self.compaction_due and not self._is_lone_compact(calls):
+                    self.append_tool_results(calls, [self._refusal(call) for call in calls])     # nothing runs until compact
+                    if self._refused():
+                        break
+                    self._report_step()
+                    continue
                 call_results = self._execute_tool_calls(calls)
-                errors += sum(1 for result in call_results if result.is_error)
-                tool_messages = rendering.tool_messages(calls, call_results)
-                wrapper = self._continuation(tokenizer, messages_before, tool_messages, template_tools, template_kwargs, sampled)
-                tool_step = TrajectoryStep(
-                    role="tool",
-                    content="\n\n".join(result.output for result in call_results),
-                    tool_calls=calls,
-                    tool_call_results=[result.output for result in call_results],
-                    activations={call["id"]: result.ac_outputs for call, result in zip(calls, call_results) if result.ac_outputs},
-                    token_ids=wrapper,
-                )
-                results.trajectory.extend([asdict(step), asdict(tool_step)])
-                self.messages.extend(tool_messages)
-                self.prefix += wrapper
+                self.errors += sum(1 for result in call_results if result.is_error)
+                if any(result.compacted for result in call_results):
+                    self._report_step()                                                          # compact() began the new segment; no tool message follows
+                    continue
+                self.append_tool_results(calls, call_results)
                 self._report_step()
                 if any(result.is_final for result in call_results):
                     results.finish_reason = "submitted"
                     break
-                if errors > config.max_tool_errors:
+                if self.errors > config.max_tool_errors:
                     results.finish_reason = "max_tool_errors"
                     break
         except Exception as error:
             if _is_context_overflow(error):
-                # The conversation outgrew the model's context (no compaction in slice 1): a budget, not a bug.
+                # The conversation outgrew the model's context: a budget, not a bug.
                 results.finish_reason = "context_exceeded"
                 results.score_feedback = str(error)[:500]
             else:
@@ -193,26 +401,97 @@
                 self.reporter.report_agent_finish(self)
         return results
 
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
+
     def _report_step(self):
         if self.reporter is not None:
             self.reporter.report_agent_step(self)
 
-    def _continuation(self, tokenizer, messages_before: list[dict], new_messages: list[dict], template_tools, template_kwargs,
-                      sampled: list[int]) -> list[int]:
-        """The template's wrapper between the sampled turn and the next generation prompt (see agent_utils)."""
-        wrapper = self.dialect.continuation_token_ids(tokenizer, messages_before, new_messages, template_tools, template_kwargs)
-        return self.dialect.join_continuation(sampled, wrapper)
-
-    def _count_ac_inputs(self):
-        ac_inputs = self.agent_config.ac_inputs
-        if not ac_inputs:
-            return
-        text = json.dumps(ac_inputs, default=repr)
-        self.run_results.num_ac_input_bytes = len(text.encode("utf-8"))
-        try:
-            self.run_results.num_ac_input_tokens = len(self.loaded_model.tokenizer(text, add_special_tokens=False)["input_ids"])
-        except Exception:
-            self.run_results.num_ac_input_tokens = len(text) // 4
+    # ----------------------------------------------------------------------------- simulation and resume
+    def simulated_run(self) -> AgentRunResult:
+        """Step mode: the prompt is built (parts encoded), nothing is sampled, the answer is a dummy submission."""
+        results = self.run_results
+        self.step_mode = True
+        self.begin()
+        results.answer = f"Agent {self.agent_id[:8]} simulated run"
+        results.finish_reason = "simulated"
+        self.finished = True
+        return results
+
+    def simulate_step(self) -> dict:
+        """
+        One step without the engine. Executes the last assistant turn's pending calls (compaction, large
+        outputs and search for real; subagents in step mode; refusals when compaction is due), appends
+        their results, and prepares the next request. Returns what a test inspects; nothing is submitted.
+        """
+        self.step_mode = True
+        self.begin()
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
+    @staticmethod
+    def resume_from_run_result(harness: "HarnessRuntime", run_result: AgentRunResult, reporter: "RolloutReporter | None" = None) -> "Agent":
+        """
+        An agent positioned exactly after the recorded steps of `run_result`'s current segment: messages,
+        prefix and spans from the record, rows re-encoded from the recorded parts (the record never
+        stores rows). Counters continue from the record.
+        """
+        agent = Agent(harness, run_result.agent_config, reporter=reporter, seed=run_result.seed)
+        agent.run_results = run_result
+        agent.dialect = ModelDialect.for_tokenizer(agent.loaded_model.tokenizer)
+        agent._prepare_tools()
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
 
     # ----------------------------------------------------------------------------- tools
     def execute_tool_call(self, call: dict) -> ToolCallResult:
@@ -229,10 +508,18 @@
                 result = ToolCallResult(output=f"Bad arguments for {name}: {error}", is_error=True)
             except Exception as error:
                 result = ToolCallResult(output=f"{name} failed: {type(error).__name__}: {error}", is_error=True)
-        output, ac_outputs = truncate_output(self._env if self._env is not None else None, result.output, call["id"])
-        result.output = output
-        if ac_outputs:
-            result.ac_outputs = dict(result.ac_outputs) | ac_outputs
+        return self._finalize_result(call, result)
+
+    def _finalize_result(self, call: dict, result: ToolCallResult) -> ToolCallResult:
+        """Truncation and its part, applied to any tool's result (the one text part of a content list equals the visible output)."""
+        visible, part = truncate_output(self._env, result.output, call["id"], agent=self if not result.compacted else None)
+        if visible != result.output:
+            content = [dict(item) for item in tool_content(result)]
+            for item in content:
+                if item.get("type") == "text" and item.get("text") == result.output:
+                    item["text"] = visible
+            result.content = ([part] if part is not None else []) + content
+            result.output = visible
         return result
 
     def _execute_tool_calls(self, calls: list[dict]) -> list[ToolCallResult]:
```

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title"><a href="slice3b/activation/agent/rollout_caching.py">activation/agent/rollout_caching.py</a></span>
    <span class="card-oneliner">Cache key covers messages_input and the AC model name.</span>
    <span class="card-badge">Diff</span>
  </summary>

`activation/agent/rollout_caching.py`

```diff
diff --git a/activation/agent/rollout_caching.py b/activation/agent/rollout_caching.py
--- a/activation/agent/rollout_caching.py
+++ b/activation/agent/rollout_caching.py
@@ -22,7 +22,8 @@
     task = config.dataset_task
     if task is not None:
         return f"{task.dataset_id}/{task.task_id}"
-    digest = hashlib.sha256(json.dumps([config.system_prompt, config.user_prompt, config.model_name, config.lora_name]).encode()).hexdigest()
+    digest = hashlib.sha256(json.dumps([config.system_prompt, config.user_prompt, config.model_name, config.lora_name,
+                                        config.messages_input, config.ac_model_name], default=repr).encode()).hexdigest()
     return f"prompt/{digest[:16]}"
 
 
```

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title"><a href="slice3b/activation/agent/rollout_reporter.py">activation/agent/rollout_reporter.py</a></span>
    <span class="card-oneliner">Parts and rows in trajectory files and stats.</span>
    <span class="card-badge">Diff</span>
  </summary>

`activation/agent/rollout_reporter.py`

```diff
diff --git a/activation/agent/rollout_reporter.py b/activation/agent/rollout_reporter.py
--- a/activation/agent/rollout_reporter.py
+++ b/activation/agent/rollout_reporter.py
@@ -152,7 +152,8 @@
         _write_atomic(folder / file, json.dumps({
             "agent_id": agent.agent_id, "state": state, "system_prompt": agent.agent_config.system_prompt,
             "user_prompt": agent.agent_config.user_prompt,
-            "trajectory": [{key: value for key, value in step.items() if key not in ("token_ids", "logprobs")} for step in results.trajectory],
+            "trajectory": [{key: value for key, value in step.items() if key not in ("token_ids", "logprobs", "messages")} for step in results.trajectory],
+            "compactions": len(results.compactions), "prompt_ac_spans": results.prompt_ac_spans,
             "answer": results.answer, "finish_reason": results.finish_reason, "score": results.score,
         }, indent=1, default=str))
     steps = []
@@ -167,8 +168,8 @@
         elif step["role"] == "tool":
             steps.append({
                 "role": "tool", "turn": turn, "content": "",
-                "results": [{"name": call["name"], "output": _cut(output, RESULT_CHARS)}
-                            for call, output in zip(step["tool_calls"], step["tool_call_results"])],
+                "results": [{"name": call["name"], "output": _cut(output, RESULT_CHARS) + (_parts_note(step) if index == 0 else "")}
+                            for index, (call, output) in enumerate(zip(step["tool_calls"], step["tool_call_results"]))],
             })
         else:                                                                       # the nudge
             steps.append({"role": "tool", "turn": turn, "content": "", "results": [{"name": "user", "output": _cut(step["content"], RESULT_CHARS)}]})
@@ -178,6 +179,14 @@
     }
 
 
+def _parts_note(step: dict) -> str:
+    """One line per activation-context part of the step: its rows, so the page shows what the model read as rows."""
+    spans = step.get("ac_spans") or []
+    if not spans:
+        return ""
+    return "\n" + "\n".join(f"[activation context part {index + 1}: {span['length']} rows]" for index, span in enumerate(spans))
+
+
 def _title(agent: "Agent") -> str:
     task = agent.agent_config.dataset_task
     return _agent_label(agent) + (f" · task {task.task_id[:12]}" if task is not None else "") + f" · seed {agent.run_results.seed}"
@@ -186,6 +195,10 @@
 def _stats(agent: "Agent") -> str:
     results = agent.run_results
     parts = [f"turn {results.num_turns}", f"{results.num_input_tokens} in ({results.num_cached_input_tokens} cached) / {results.num_output_tokens} out"]
+    if results.compactions:
+        parts.append(f"{len(results.compactions)} compactions")
+    if results.num_ac_parts:
+        parts.append(f"{results.num_ac_parts} AC parts / {results.num_ac_rows} rows")
     if agent.finished:
         parts.append(results.finish_reason)
         parts.append(f"answer {str(results.answer)[:40]!r}")
```

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title"><a href="slice3b/activation/agent/__init__.py">activation/agent/__init__.py</a></span>
    <span class="card-oneliner">Export CompactionTool, SyntheticTurn, synthesize_agent.</span>
    <span class="card-badge">Diff</span>
  </summary>

`activation/agent/__init__.py`

```diff
diff --git a/activation/agent/__init__.py b/activation/agent/__init__.py
--- a/activation/agent/__init__.py
+++ b/activation/agent/__init__.py
@@ -9,8 +9,10 @@
     ParallelCallTool,
     SemanticSearchTool,
     SubagentTool,
+    CompactionTool,
 )
 from .agent import Agent
+from .agent_utils import SyntheticTurn, synthesize_agent
 from .rollout_caching import RolloutCache
 from .rollout_manager import RolloutManager
 from .rollout_reporter import RolloutReporter
```

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title"><a href="slice3b/activation/agent_training/agent_trainer.py">activation/agent_training/agent_trainer.py</a></span>
    <span class="card-oneliner">unroll walks segments and their children.</span>
    <span class="card-badge">Diff</span>
  </summary>

`activation/agent_training/agent_trainer.py`

```diff
diff --git a/activation/agent_training/agent_trainer.py b/activation/agent_training/agent_trainer.py
--- a/activation/agent_training/agent_trainer.py
+++ b/activation/agent_training/agent_trainer.py
@@ -71,10 +71,12 @@
 
 
 def unroll(result: AgentRunResult) -> list[AgentRunResult]:
-    """The run and every subagent run beneath it, depth first."""
-    out = [result]
-    for child in result.subagent_results:
-        out.extend(unroll(child))
+    """The run's compacted segments, the run, and every subagent run beneath it (with theirs), depth first."""
+    out = []
+    for segment in list(result.compactions) + [result]:
+        out.append(segment)
+        for child in segment.subagent_results:
+            out.extend(unroll(child))
     return out
 
 
```

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title"><a href="slice3b/activation/agent_training/agent_training_selection.py">activation/agent_training/agent_training_selection.py</a></span>
    <span class="card-oneliner">Selection always includes compacted segments.</span>
    <span class="card-badge">Diff</span>
  </summary>

`activation/agent_training/agent_training_selection.py`

```diff
diff --git a/activation/agent_training/agent_training_selection.py b/activation/agent_training/agent_training_selection.py
--- a/activation/agent_training/agent_training_selection.py
+++ b/activation/agent_training/agent_training_selection.py
@@ -10,7 +10,8 @@
 - `reinforce_reject`: +1 / -1 on mixed groups only.
 
 Every function unrolls subagent runs with the parent's weight (a subagent's trajectory is judged by the
-answer it helped produce) and marks non-policy sources (`run.source != "policy"`) as `ignore_logprobs`.
+answer it helped produce), always includes a run's compacted segments (the same agent's earlier
+sequences), and marks non-policy sources (`run.source != "policy"`) as `ignore_logprobs`.
 """
 from __future__ import annotations
 
@@ -27,7 +28,7 @@
     """One item per (sub)agent run under `run`, all at `weight`; zero weights produce nothing."""
     if weight == 0.0:
         return []
-    runs = unroll(run) if include_subagents else [run]
+    runs = unroll(run) if include_subagents else list(run.compactions) + [run]   # compacted segments are the same agent: always included
     return [
         AgentTrainingItem(run_results=r, model_name=model_name, lora_name=lora_name, weight=float(weight), ignore_logprobs=r.source != "policy")
         for r in runs
```

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title"><a href="slice3b/activation/agent_training/agent_training_utils.py">activation/agent_training/agent_training_utils.py</a></span>
    <span class="card-oneliner">Example builder refuses AC-bearing runs until the fixed-row trainer lands.</span>
    <span class="card-badge">Diff</span>
  </summary>

`activation/agent_training/agent_training_utils.py`

```diff
diff --git a/activation/agent_training/agent_training_utils.py b/activation/agent_training/agent_training_utils.py
--- a/activation/agent_training/agent_training_utils.py
+++ b/activation/agent_training/agent_training_utils.py
@@ -38,6 +38,10 @@
     items). None when the run has no token segments or no sampled tokens.
     """
     run = item.run_results
+    if run.prompt_ac_spans or any(step.get("ac_spans") for step in run.trajectory):
+        # Placeholder ids at part positions would be embedded as pad tokens: training on AC-bearing sequences
+        # needs the rows rebuilt from the recorded parts (deferred to the training-recipe slices).
+        raise ValueError("Training on AC-bearing runs awaits the fixed-row trainer integration")
     token_ids = list(run.prompt_token_ids)
     if not token_ids:
         return None
```

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title"><a href="slice3b/activation/tests/test_basic_agent_ac.py">activation/tests/test_basic_agent_ac.py</a></span>
    <span class="card-oneliner">New node test: simulated channels with engine probes, an AC rollout with harvest, a text-only compaction rollout.</span>
    <span class="card-badge">Diff</span>
  </summary>

`activation/tests/test_basic_agent_ac.py`

```diff
diff --git a/activation/tests/test_basic_agent_ac.py b/activation/tests/test_basic_agent_ac.py
new file mode 100644
--- /dev/null
+++ b/activation/tests/test_basic_agent_ac.py
@@ -0,0 +1,221 @@
+"""
+Slice 3b key test: activation context in the agent loop, on a GPU node.
+
+    uv run sky exec --sync <node> -- uv run pytest activation/tests/test_basic_agent_ac.py --gpu --slow -s
+
+Untrained AC rows are junk, so the simulated scenarios assert mechanics (parts, spans, rows, records, the engine
+accepting mixed prompts and hitting its prefix cache), not answers. `AC_CHECKPOINT=AC_MODELS/ac_dev/epoch_002` loads a
+3a1 checkpoint instead. The last two tests are real rollouts with a low compaction threshold: with AC rows, and text-only.
+"""
+import json
+import os
+from dataclasses import replace
+
+import pytest
+import torch
+
+from activation.ac_model import ActivationContextModelConfig, ActivationContextStudyGenerator
+from activation.common.ac_parts import direct_parts
+from activation.agent import Agent, AgentConfig, AgentRunResult, RolloutReporter, SemanticSearchTool, SubagentTool, SyntheticTurn, synthesize_agent
+from activation.agent.agent_tools import SEARCH_EXTRA_MAX
+from activation.common.data_syncing import resolve_path
+from activation.dataset.loaders import BrightDataset
+from activation.harness import FREE_DEVICE, HarnessRuntime, HarnessRuntimeConfig, ModelConfig
+
+MODEL_NAME, MODEL_ID = "qwen3.5-4b", "Qwen/Qwen3.5-4B"
+SIDE_NAME, SIDE_ID = "qwen3.5-0.8b", "Qwen/Qwen3.5-0.8B"
+AC_NAME, SIDE_LORA, TARGET_LORA = "ac_dev", "ac_side", "ac_target"
+ENGINE_CACHE_BLOCK_TOKENS = 1056   # prefix-cache block of the hybrid 4B engine with the fp8 KV cache (mamba-aligned); prompts at or below it cannot hit
+SYSTEM = ("You are a careful problem solver working in a sandbox. Use the python or shell tools to compute rather than "
+          "guessing. When you are done, call submit_answer exactly once with the final answer.")
+PRIMES = "Compute the sum of the 140th, 141st and 142nd prime numbers (2 is the 1st prime)."
+BASE = AgentConfig(system_prompt=SYSTEM, model_name=MODEL_NAME, ac_model_name=AC_NAME, enable_ac_communication=True,
+                   call_kwargs={"sampling_params": {"max_tokens": 1024, "temperature": 0.7}}, max_turns=12, max_tool_errors=3, max_duration=300)
+LONG = "0123456789" * 120   # 1,200 chars per synthetic tool output
+
+
+def _harness(with_ac: bool = True) -> HarnessRuntime:
+    configs = {MODEL_NAME: ModelConfig(MODEL_NAME, MODEL_ID)}
+    if with_ac:
+        configs[SIDE_NAME] = ModelConfig(SIDE_NAME, SIDE_ID)
+    harness = HarnessRuntime(HarnessRuntimeConfig(model_configs=configs, agent_max_concurrent=8))
+    if with_ac:
+        harness.module_manager.register_lora(TARGET_LORA, MODEL_NAME, rank=64)
+        harness.module_manager.register_ac_model(ActivationContextModelConfig(
+            AC_NAME, SIDE_NAME, SIDE_LORA, MODEL_NAME, TARGET_LORA, checkpoint_path=os.environ.get("AC_CHECKPOINT")))
+    return harness
+
+
+def _release(harness: HarnessRuntime) -> None:
+    for loaded in harness.loaded_models.values():
+        loaded.engine_to_device(FREE_DEVICE)
+    if AC_NAME in harness.module_manager.ac_models:
+        ac_model = harness.module_manager.get_ac_model(AC_NAME)
+        ac_model.release()
+        for model_name, lora_name in ((ac_model.config.target_model_name, ac_model.config.target_model_lora_name),
+                                      (ac_model.config.base_side_model_name, ac_model.config.base_side_model_lora_name)):
+            if harness.module_manager.has_loras(model_name):
+                harness.module_manager.free_lora(model_name, lora_name)       # adapters first: a base with adapters cannot be freed
+    for loaded in harness.loaded_models.values():
+        loaded.model_to_device(FREE_DEVICE)
+
+
+def _probe_engine(agent: Agent, label: str) -> None:
+    """One real mixed-prompt request, then the same again: the engine accepts the rows and serves the prefix from its cache."""
+    prefix, embeds, mask = agent.prepare_request()
+    first = agent.submit_probe()
+    second = agent.submit_probe()
+    print(f"  [{label}] prompt {len(prefix)} tokens, rows at {0 if mask is None else mask.count(False)} positions, "
+          f"cached {first.cached_prompt_token_count} -> {second.cached_prompt_token_count}, sampled {first.output_token_count} tokens: "
+          f"{first.text[:80]!r}")
+    assert first.prompt_token_count == len(prefix)
+    if embeds is not None:
+        assert tuple(embeds.shape) == (len(prefix), agent.loaded_model.model_config.model_description.d_model)
+        assert embeds.dtype == agent.row_dtype
+    # Prefix hits come in cache-aligned blocks: hybrid Qwen3.5 with the fp8 KV cache aligns at 1,056 tokens (528 with bf16), and a prompt
+    # shorter than one block can never hit (verified with token-only prompts on the node: 909 tokens -> 0, 1,254 tokens -> 1,056).
+    if len(prefix) > ENGINE_CACHE_BLOCK_TOKENS:
+        assert second.cached_prompt_token_count > 0, "the second identical request did not hit the prefix cache"
+
+
+@pytest.mark.gpu
+@pytest.mark.slow
+def test_ac_agent_channels_simulated():
+    """Compaction, subagent, large output and search through simulate_step; a resume round trip; one engine submit per scenario."""
+    harness = _harness()
+    bright = BrightDataset.load(harness, max_examples=20, domain="biology", max_corpus_documents=200)
+    harness.dataset_manager.register_dataset(bright)
+    example = next(iter(bright.labeled_qa_examples.values()))           # BRIGHT is a retrieval benchmark: labeled queries, no scorable tasks
+    ac_model = harness.module_manager.get_ac_model(AC_NAME)
+    harness.loaded_models[MODEL_NAME].ensure_engine_loaded()          # after the AC registration: enable_prompt_embeds and the memory reservation apply
+    tokenizer = harness.loaded_models[MODEL_NAME].tokenizer
+    try:
+        # --- compaction: three long tool outputs cross a 600-token delta; a python call is refused; a lone compact restarts the segment.
+        config = replace(BASE, user_prompt=PRIMES, compaction_threshold_tokens=600)
+        turns = [SyntheticTurn(f"Step {i}.", [("python", {"code": f"print({i})"})], [LONG]) for i in range(3)]
+        agent = synthesize_agent(harness, config, turns)
+        assert agent.compaction_due and not agent.run_results.compactions
+        agent.record_assistant_turn("More work.", [{"id": "r1", "name": "python", "arguments": {"code": "print(4)"}}], tokenizer.encode("More work."), [])
+        state = agent.simulate_step()
+        assert state["results"][0].output.startswith("Refused python") and agent.compaction_refusals == 1
+        agent.record_assistant_turn("", [{"id": "c1", "name": "compact", "arguments": {"summary": "Printed 0..3; next sum the primes."}}],
+                                    tokenizer.encode("compact"), [])
+        state = agent.simulate_step()
+        results = agent.run_results
+        assert state["compactions"] == 1 and results.compactions[0].finish_reason == "compacted"
+        assert results.compactions[0].num_turns == 5 and len(results.compactions[0].trajectory) == 9   # the compact call has no tool step
+        assert results.trajectory == [] and not agent.compaction_due and agent.segment_start == len(agent.prefix)
+        assert [m["role"] for m in results.prompt_messages] == ["system", "user", "user"]
+        tree = direct_parts(results.prompt_messages)[0]
+        assert tree["compression_target"] == config.ac_compaction_ratio and tree["messages"][0]["content"] == config.user_prompt
+        assert len(tree["messages"]) == 1 + 9 and tree["messages"][-1]["tool_calls"][0]["function"]["name"] == "compact"
+        assert results.prompt_messages[2]["content"][1]["text"].endswith("Printed 0..3; next sum the primes.")
+        assert len(agent.spans) == 1 and agent.rows[0].shape[0] == ac_model.part_view_rows(tree["messages"], tree["compression_target"])
+        assert state["embeds_shape"] == (len(agent.prefix), agent.rows[0].shape[1]) and state["row_positions"] == agent.rows[0].shape[0]
+        _probe_engine(agent, "compaction")
+        # A second compaction nests the first tree verbatim.
+        agent.record_assistant_turn("", [{"id": "c2", "name": "compact", "arguments": {"summary": "again"}}], tokenizer.encode("compact"), [])
+        agent.compaction_due = True
+        agent.simulate_step()
+        outer = direct_parts(agent.run_results.prompt_messages)[0]
+        assert direct_parts(outer["messages"])[0] == tree and len(agent.run_results.compactions) == 2
+        _probe_engine(agent, "compaction x2")
+
+        # --- subagent in step mode: the child's prompt carries the parent segment; the parent's result carries the child's segment.
+        child_base = replace(BASE, user_prompt="", tools={})
+        config = replace(BASE, user_prompt="Delegate the prime search.", tools={"subagent": (SubagentTool, {"base_config": child_base})})
+        agent = synthesize_agent(harness, config, [SyntheticTurn("Delegating.", [("subagent", {"task": "Find the 140th prime."})])])
+        state = agent.simulate_step()
+        child = agent.run_results.subagent_results[0]
+        assert child.finish_reason == "simulated" and child.answer.endswith("simulated run")
+        child_user = child.prompt_messages[1]
+        assert [p.get("type") for p in child_user["content"]] == ["text", "activation_context", "text"]
+        assert child_user["content"][2]["text"] == "Find the 140th prime." and child_user["content"][1]["messages"][0]["content"] == config.user_prompt
+        assert child.prompt_ac_spans and child.prompt_ac_spans[0]["length"] == ac_model.part_view_rows(child_user["content"][1]["messages"], BASE.ac_subagent_ratio)
+        tool_step = agent.run_results.trajectory[-1]
+        assert [p.get("type") for p in tool_step["messages"][0]["content"]] == ["activation_context", "text"]
+        assert tool_step["messages"][0]["content"][0]["messages"][0]["content"][1]["type"] == "activation_context"   # the child's segment starts with its part
+        assert len(tool_step["ac_spans"]) == 1 and len(agent.spans) == 1
+        _probe_engine(agent, "subagent")
+
+        # --- large tool output: a real python call prints 100k chars; the part holds the full text nested with the parent segment.
+        config = replace(BASE, user_prompt="Print a lot.")
+        agent = synthesize_agent(harness, config, [SyntheticTurn("Printing.", [("python", {"code": "print('x' * 100000)"})])])
+        state = agent.simulate_step()
+        result = state["results"][0]
+        assert len(result.output) < 21_000 and "truncated" in result.output and result.content is not None
+        part, text = result.content
+        assert part["type"] == "activation_context" and text["text"] == result.output
+        assert part["compression_target"] == BASE.ac_tool_output_ratio and part["messages"][1]["role"] == "tool"
+        assert len(part["messages"][1]["content"]) >= 80_000 and direct_parts(part["messages"])[0]["messages"][0]["content"] == config.user_prompt
+        assert agent.run_results.trajectory[-1]["ac_spans"][0]["length"] == agent.rows[0].shape[0]
+        _probe_engine(agent, "tool output")
+        agent.shutdown()
+
+        # --- semantic search: top_k visible, min(20, 2 top_k) extra as a part after the text.
+        config = replace(BASE, user_prompt=example.query, tools={"semantic_search": (SemanticSearchTool, {"top_k": 3, "dataset_id": bright.dataset_id})})
+        agent = synthesize_agent(harness, config, [SyntheticTurn("Searching.", [("semantic_search", {"query": example.query[:200]})])])
+        state = agent.simulate_step()
+        result = state["results"][0]
+        assert result.output.count("\n[") + result.output.startswith("[") == 3 and result.content[0]["type"] == "text" and result.content[1]["type"] == "activation_context"
+        extra = result.content[1]["messages"][1]["content"]
+        assert extra.count("\n[") + extra.startswith("[") == min(SEARCH_EXTRA_MAX, 6) and result.content[1]["compression_target"] == BASE.ac_search_ratio
+        _probe_engine(agent, "search")
+
+        # --- resume: the serialized record rebuilds the same prefix, spans and rows.
+        data = json.loads(json.dumps(agent.run_results.serialize()))
+        resumed = Agent.resume_from_run_result(harness, AgentRunResult.deserialize(data, harness))
+        assert resumed.prefix == agent.prefix and resumed.spans == agent.spans
+        assert [tuple(r.shape) for r in resumed.rows] == [tuple(r.shape) for r in agent.rows]
+        assert all(torch.equal(a, b) for a, b in zip(resumed.rows, agent.rows))
+        assert resumed.messages == agent.messages and resumed.compaction_due == agent.compaction_due
+        print("\n=== AC model stats ===\n" + json.dumps(ac_model.stats.summarize(), indent=1))
+    finally:
+        _release(harness)
+
+
+@pytest.mark.gpu
+@pytest.mark.slow
+def test_ac_agent_rollout_with_ac():
+    """A real rollout with AC rows and a low threshold: mechanics only (segments recorded, rows shipped, cache hits), the answer is printed. Ends with the harvest of AC training items from the runs."""
+    harness = _harness()
+    config = replace(BASE, user_prompt=PRIMES, compaction_threshold_tokens=300)          # a few hundred tokens per turn: compaction after ~2 turns
+    reporter = RolloutReporter(str(resolve_path("AGENT_AC_TEST/with_ac")), title="Agent AC test: primes with compaction")
+    try:
+        results = harness.rollout_manager.perform_grouped_rollouts([config], group_count=2, base_seed=0, perform_scoring=False, reporter=reporter)[0]
+        generator = ActivationContextStudyGenerator(harness, AC_NAME)
+        items = generator.items_from_run_results(results, weight_of=lambda run: 1.0)
+        print(f"\n=== harvested {len(items)} AC training items from {len(results)} runs: {generator.harvest_report}")
+    finally:
+        _release(harness)
+    for result in results:
+        print(f"\n=== seed {result.seed}: {result.finish_reason}, {result.num_turns} turns, {len(result.compactions)} compactions, "
+              f"{result.num_ac_parts} parts / {result.num_ac_rows} rows, cached {result.num_cached_input_tokens}, answer {result.answer!r}")
+        assert result.finish_reason in ("submitted", "max_turns", "max_tool_errors", "max_duration", "no_tool_call", "compaction_refused", "trajectory_cap")
+        assert result.ac_model_name == AC_NAME and result.ac_model_version is not None
+        for segment in result.compactions:
+            assert segment.finish_reason == "compacted" and segment.prompt_token_ids and segment.trajectory
+        if result.compactions:
+            assert result.prompt_ac_spans and result.num_ac_rows > 0
+    assert os.path.exists(os.path.join(reporter.report_folder, "report.tressoir.html"))
+
+
+@pytest.mark.gpu
+@pytest.mark.slow
+def test_agent_compaction_text_only():
+    """No AC model: the same protocol with the summary alone; at least one rollout compacts. Finish reasons are printed, not asserted beyond the allowed set."""
+    harness = _harness(with_ac=False)
+    config = replace(BASE, ac_model_name=None, enable_ac_communication=False, compaction_threshold_tokens=300, user_prompt=PRIMES)
+    reporter = RolloutReporter(str(resolve_path("AGENT_AC_TEST/text_only")), title="Agent AC test: text-only compaction")
+    try:
+        results = harness.rollout_manager.perform_grouped_rollouts([config], group_count=2, base_seed=0, perform_scoring=False, reporter=reporter)[0]
+    finally:
+        _release(harness)
+    for result in results:
+        print(f"\n=== seed {result.seed}: {result.finish_reason}, {result.num_turns} turns, {len(result.compactions)} compactions, answer {result.answer!r}")
+        assert result.num_ac_parts == 0 and result.prompt_ac_spans == []
+        for segment in result.compactions:
+            assert segment.finish_reason == "compacted"
+            assert isinstance(result.prompt_messages[-1]["content"], str)
+        assert result.finish_reason in ("submitted", "max_turns", "max_tool_errors", "max_duration", "no_tool_call", "compaction_refused", "trajectory_cap")
+    assert any(result.compactions for result in results), "no rollout crossed the 300-token delta"
```

</details>

