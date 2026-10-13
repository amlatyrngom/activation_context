# Trajectory campaign source handoff

The source package now supports question-aware and question-blind QA, plain teacher-trajectory SFT through the AC trainer, and bounded compaction targets from an immutable initial reader. It also carries the earlier reviewed cleanup and RAG/reporting fixes. `/source` remains untouched.

[Exact patch](trajectory_implementation/changes.patch) · [File hashes and validation](trajectory_implementation/manifest.json) · [Campaign plan](TRAJECTORY_COMPACTION_PLAN.md)

The package contains 20 changed source files against `/source`: product commit `50e21b45a557fc9633a470702860eb7b688f4361`, the compaction turn-count/ETA/cache-filename fixes, the shared compaction handoff prompt, and measured execution optimizations. The manifest records the exact bytes. Running source snapshots remain frozen. It supersedes the earlier consolidation patch for that baseline. Check each destination's `before_sha256` before applying; preserve divergent local changes. An already matching `after_sha256` needs no replacement.

```bash
git apply --check /path/to/trajectory_implementation/changes.patch
git apply /path/to/trajectory_implementation/changes.patch
```

The task behavior is concentrated in two existing files:

- `ac_model_study.py`: builds a normal trajectory QA training item from a caller-selected history window and fixed question/answer. The new `encoder_sees_question` boolean controls question visibility at every encoder level; trajectory QA defaults false and RAG defaults true. The reader always sees the question. Original task, system and tools remain in the encoded history. Full-context evaluation receives the same evidence. Recent-text controls enforce their actual token budget after decoding.
- `ac_model_training.py`: optional `teacher_cache_block_examples` bounds training-target memory. KL training snapshots the initial reader LoRA on the existing base model, persists its identity, and restores it together with optimizer state and penalty RNG. No-context preservation still targets the plain base. Per-target evaluation KL supports immediate-versus-later continuation measurements. Unsafe eager KL optimizer resume raises rather than silently changing teachers.

Compaction now resets the current segment’s turn count after moving its completed turns into the history, preventing premature exhaustion of the total turn budget. Resumed ETA uses only the updates completed during the measured interval. Focused CPU checks cover two consecutive compactions and the first/later epochs after resume.

The existing `ac_parts.py` owns the compaction prompt shared by the live tool and the private 27B summary producer. It requests Summary Overview, Progress So Far, Notes/Insights/Lessons To Preserve, Immediate Continuation, and DOs/DON'Ts. The latest change adds one sentence requesting the entire handoff under 500 words, prioritizing details needed to resume (prompt-only commit `10d6e55`). It retains the distinctions between guesses and observations, contradictory evidence, and instructions at their original scope. The existing tool description and compaction trigger reference that format; the resume cue works with text or AC. The word request does not guarantee the separate 1,536-token validation limit; outputs are not truncated to pass. The v7 pilot accepted three of four previously rejected held cases under the unchanged token limit. A brief source check found no new major task-state loss, while inherited research-summary overconfidence remains a limitation. This changes requested handoff content, not runtime context storage.

The approved next experiment trains AC to reproduce fixed oracle summaries. Its private producer and training worker use only original pre-cut history as encoder input: no future continuation, oracle summary, or compact call. Initial CPU checks cover target boundaries, portable bank reuse, fixed held order and zero-update restart/reporting. GPU proof and quality findings remain separate campaign artifacts. Summary NLL is reported separately from continuation NLL, with newly frozen controls under the updated resume cue. Ordinary runtime compact records include the summary call, so they cannot be harvested unchanged for this objective.

Study-cache filenames now retain existing valid names and shorten overlong keys with a hash of the complete original key. This fixes an observed filesystem failure after moving synthesis to a longer model path. Focused checks cover distinct long keys, legacy names and filesystem read/write. The running legacy producer uses an identical-model short-path alias for immediate recovery; its frozen source stays unchanged.

No new model architecture, utility module or public trajectory abstraction was introduced. Campaign-specific selection, synthesis, reporting and Slurm operation remain private bench scripts. Historical frozen producers remain reproducible.

A resumable compaction run needs its paired encoder/reader weights, optimizer checkpoint and the immutable `teacher_snapshots` directory referenced by the checkpoint manifest. Selected development pairs carry a shared frozen-teacher copy but intentionally omit optimizer state; they are evaluation/initialization exports, not exact training resumes.

Six tiny-decoder fixed-teacher/cache/resume tests pass, including uninterrupted-versus-resumed equivalence. Real-tokenizer checks cover question isolation and exact continuation targets. A focused check confirms that per-target KL has the correct target count and averages to the scalar KL. The patch parses, applies to an isolated baseline, reproduces all staged bytes and reverses cleanly.

The real 2B→9B L40S test completed four Adam updates across two processes, preserving teacher weights/outputs, plain-base targets, optimizer moments and penalty RNG. The full runner smoke completed eight examples, fixed controls, generated examples, plots and paired checkpoints. The first GPU runs used V3. Historical V6/V7 runs used the V4 product files before the current performance changes: a combined 32,716-token history + 8,185-token suffix passed on L40S with private Adam-moment parking, including two-process resume; the full eight-example runner passed frozen NLL/KL controls and immediate-token reporting. Larger GPUs are being profiled for throughput. These are execution checks, not quality findings.

Private V8 campaign work adds CPU preparation and a validated tokenized-example cache to avoid repeated long startup. It also fixes tuple/list normalization in the private runner’s JSON identity checks; this does not change the product patch. The actual GPU runner stop/reload check passed: two updates, process exit, optimizer resume and two further updates. Independent V7/V8 runs have identical initial forward predictions but slightly different first backward gradients, before any restart; this does not establish bitwise GPU equivalence. The reviewer found no cache/resume defect. See [scale review](TRAJECTORY_COMPACTION_SCALE_REVIEW.md) for measured startup, memory and throughput limits.

This package is a usable implementation checkpoint during the active 36-hour campaign. Later campaign fixes will receive an updated manifest; do not overlay historical patches afterward. The complete pre-cleanup workspace remains on `checkpoint/pre-trajectory-cleanup-20260923`, commit `600137587987f13f0055c06494b788638cd0745b`.

**Live deep-checkpoint limitation:** ordinary `Agent` rollouts currently accept input rows only. Golden deep checkpoints require complete per-layer interventions and explicitly reject that rows-only path. Current deep qualitative generations use the supported HF path. A private serial HF live-tool probe has executed all six fixed tool tasks with full per-layer payloads: frozen full history succeeds6/6, initialAC0/6 and the5.21M-token compaction pair1/6. These controlled fixtures do not establish natural-task quality. Production vLLM integration, deep-payload sidecar replay and concurrent deep rollouts are not implemented by this package. Training/evaluation evidence above remains valid.

The existing autotuner now measures complete SFT execution, including forward/backward, clipping, ordinary Adam and CPU/GPU moment transfers. Its optional execution policy separates examples per backward graph, reader batch size, encoder token budget, checkpoint thresholds and graph-local reuse of complete identical encodings. Logical eight-question updates, item weighting, penalty draws and singleton evaluation stay unchanged. Reuse does not regroup the dataset or cache across updates. Complex logic stays in existing utilities and autotuners; no new utility module is added.

The tuner runs on disposable models with fixed weights, allocated optimizer moments and fixed penalty flags. Every accepted candidate must match native loss and both gradient groups, remain within the memory limit, and pass the sampled workload including its long-history tail. Receipts record hardware, software, source and workload identity; a changed environment requires remeasurement. Reports distinguish execution timing from preparation, evaluation and checkpoint costs. Tuning is currently SFT-only; continuation KL retains its existing execution path.

Capacity probes also cover sequences just below each checkpoint threshold, where skipping checkpointing can use more memory than the longest sequence. These checks include final partial graphs and reader batches, keep real tail positions, and force the existing no-context penalty branch. They do not influence speed ranking. Cached receipts must contain their required capacity evidence. All 25 focused CPU tests pass. The original 8,000-question RTX and H200 singleton winners now also pass all three timing cases and their three relevant GPU capacity cases. Generic partial-batch coverage remains CPU-tested; this divisible-by-eight campaign does not establish GPU evidence for every possible batch remainder.

| Original-history fixture | Encoder / reader checkpoint thresholds | Median batch, seconds/question | 90th-percentile batch | Distinct 32k stress | Maximum GPU memory, including capacity checks |
|---|---|---:|---:|---:|---:|
| RTX PRO 6000 | 8,192 / 4,096 positions | 2.123 | 2.771 | 4.915 | 69.51 GiB |
| H200 | 16,384 / 4,096 positions | 1.582 | 1.859 | 3.300 | 104.69 GiB |

Both use singleton readers, no encoding reuse and CPU Adam moments. Sequences below the listed thresholds skip checkpointing. The median/p90 timing cases give planning references of 4.72/6.16 training hours on RTX and 3.52/4.13 on H200 for 8,000 questions × one epoch, excluding preparation, evaluation and saves. These are measured batches, not full-pass means or statistical intervals. The manifest records both receipt paths, hashes and policies. The receipts use initializer weights and do not validate a different resumed pair merely because its architecture matches.

The harness also avoids redundant full-model PEFT routing inside checkpointed layers. The original forward already has an outer adapter scope; recomputation retains its own scope. Fresh trainability and adapter-state snapshots preserve nested, disabled, merged and auxiliary-module behavior. Intervention position tensors are reused within a forward, and detailed diagnostic norms are computed only when reported.

Private validation includes 24 execution/Adam tests, 54 intervention tests, 77 adapter-scope tests, and nine tuner/progress checks. Native train/save/fresh-process-resume passed on L40S and RTX PRO, restoring exact weights, optimizer moments, clocks, penalty RNG, example order and frozen controls. GPU old/new adapter-scope comparisons passed on both devices for short histories and eight distinct 32k histories. These establish execution behavior, not task-quality improvement or bitwise equivalence between independent CUDA runs.

**Current batching limit:** reader batches of two/four fail the unchanged gradient gate despite close losses. They remain rejected execution-tuner candidates. A separate private B1/B2 learning experiment has been queued for the same next 512 fresh questions from one preserved checkpoint/Adam/RNG state; it does not fabricate a passed receipt or relax the numerical gate. No learning result is claimed here. Useful long histories stay intact when hardware and execution optimizations give adequate turnaround. See [performance findings](PERFORMANCE_FINDINGS.md).

The portable QA launcher now changes into its frozen source directory before loading subprocess workers. Its initial EC2 dependency/import failures occurred before any new optimizer update; they are environment setup failures, not additional training evidence. Portable migration, recovery transport and the learning experiment remain private campaign scripts outside these 20 product files.

## Ordinary-agent rehearsal and question visibility

`items_from_run_results(..., loss_kind="sft", text_only=True)` preserves ordinary context and retokenizes recorded assistant text, thinking and tool calls with the student tokenizer. Tool observations and user inputs are conditioning only. The default AC harvesting path keeps its recorded-token behavior. Text-only mode rejects AC-collected trajectories and creates no encoder requests; plain examples update only the reader LoRA. The teacher model must still be registered for recorded tool reconstruction.

Both QA builders expose `encoder_sees_question`; current defaults are preserved. Campaign runners bind non-default modes into preparation/resume identity, so old default runs remain resumable. The bank itself is unchanged and can provide both presentations.

Validation: six focused CPU contracts cover both tasks at depths1/2, unchanged targets/references, no answer injection, plain-reader updates and native Qwen thinking/tool masks. Three existing RAG layout/resume checks pass. RTX PRO checks pass for both QA layouts and an AC update followed by ordinary-agent SFT using populated optimizer state; encoder LoRA bytes stay fixed and reader LoRA changes. The initial GPU assertion compared CPU and CUDA tensors and was corrected to compare their CPU values; no product change was needed. Large-scale agent-SFT quality is not yet established.

## File changes against /source

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title"><a href="trajectory_implementation/activation/ac_model/ac_model.py">activation/ac_model/ac_model.py</a></span>
    <span class="card-oneliner">Retain training-only diagnostic collection; model architecture is unchanged by cleanup.</span>
  </summary>

````diff-python
diff --git a/activation/ac_model/ac_model.py b/activation/ac_model/ac_model.py
index c3313e2..33a0830 100644
--- a/activation/ac_model/ac_model.py
+++ b/activation/ac_model/ac_model.py
@@ -548,7 +548,7 @@ class ActivationContextModel:
                     for layer in self.intervention_layers:
                         source = modules.passage_attention[str(layer)](rows, keys)
                         deltas[layer] = modules.delta_heads[str(layer)](source)
-                        if self.mode == TRAINING:                              # does the delta re-inject the row? cosine with the input row, per layer; and the delta's RMS (0-d tensors, read at the trace point)
+                        if self.mode == TRAINING and torch.is_grad_enabled():   # does the delta re-inject the row? cosine with the input row, per layer; and the delta's RMS (0-d tensors, read at the trace point)
                             with torch.no_grad():
                                 cosine = F.cosine_similarity(deltas[layer].detach(), input_embeds.detach(), dim=-1).mean()
                             self._record_diagnostic("cosine", layer, cosine)
````

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title"><a href="trajectory_implementation/activation/ac_model/ac_model_reporter.py">activation/ac_model/ac_model_reporter.py</a></span>
    <span class="card-oneliner">Measure resumed ETA only from newly timed work; retain honest stop/penalty labels and completion samples.</span>
  </summary>

````diff-python
diff --git a/activation/ac_model/ac_model_reporter.py b/activation/ac_model/ac_model_reporter.py
index 6cfb6f7..e23fdac 100644
--- a/activation/ac_model/ac_model_reporter.py
+++ b/activation/ac_model/ac_model_reporter.py
@@ -33,6 +33,7 @@ class ActivationContextTrainingReporter(HtmlReporter):
         self.epoch_x = 0.0
         self.epoch_origin: int | None = None
         self.planned_updates: int | None = None
+        self._resume_skip_steps = 0
         self._interventions_reported = 0
         self.initialize_line_plot("gold_nll", "Gold-answer log-loss",
                                   "Negative log-likelihood per target token of the reference completion, item mean. "
@@ -88,6 +89,7 @@ class ActivationContextTrainingReporter(HtmlReporter):
     # ------------------------------------------------------------------------------------------ trainer callbacks
     def initialize_training(self, config: "ActivationContextTrainingConfig", stats: "ActivationContextTrainingStats") -> None:
         self._set_loss_kind(config.loss_kind)
+        self._resume_skip_steps = config.resume_skip_steps
         if self.epoch_origin is None:
             self.epoch_origin = stats.start_epoch - 1
         self.epoch_x = float(stats.start_epoch - 1 - self.epoch_origin)
@@ -111,12 +113,13 @@ class ActivationContextTrainingReporter(HtmlReporter):
             self.planned_updates = progress.global_step - ((progress.epoch - 1) * progress.steps + progress.step) + progress.epochs * progress.steps
         done = (progress.epoch - 1) * progress.steps + progress.step if progress.epoch else 0
         total = progress.epochs * progress.steps
-        remaining = (progress.elapsed_s / done * (total - done)) if done and total > done else None
+        measured_steps = progress.step - (self._resume_skip_steps if progress.epoch == 1 else 0)
+        remaining = (progress.epoch_elapsed_s / measured_steps * (total - done)) if measured_steps > 0 and total > done else None
         self.set_status(phase=phase, epoch=f"{progress.epoch} of {progress.epochs}", update=f"{progress.step} of {progress.steps} in this epoch",
                         global_updates=f"{progress.global_step}" + (f" of {self.planned_updates}" if self.planned_updates else ""),
-                        elapsed=_duration(progress.elapsed_s),
-                        remaining=("about " + _duration(remaining)) if remaining is not None else ("done" if done == total and total else "estimating"))
-        if phase == "training":
+                        train_call_elapsed=_duration(progress.elapsed_s),
+                        update_eta=("about " + _duration(remaining)) if remaining is not None else ("updates finished; finalizing" if done == total and total else "estimating"))
+        if phase not in ("reporting", "validation", "deltas_off"):
             self._clear_status("evaluation_items")
         self.render()
 
@@ -131,8 +134,8 @@ class ActivationContextTrainingReporter(HtmlReporter):
         if stats.penalty:
             if "penalty" not in self.widgets:
                 self.initialize_line_plot("penalty", "No-context penalty",
-                    "KL of the adapted reader to the base on the same items without their rows (unweighted, sampled examples); "
-                    "a rising curve means the reader improves without the rows.", "global update", "KL", ["train"])
+                    "KL(base no-context || current reader no-context), unweighted on sampled examples; "
+                    "rising values mean departure from prior behavior.", "global update", "KL", ["train"])
             self.add_data_point("penalty", {"x": progress.global_step, "train": stats.penalty[-1]})
             self.set_status(no_context_penalty=f"{stats.penalty[-1]:.4f}")
         if self.references:
@@ -225,7 +228,8 @@ class ActivationContextTrainingReporter(HtmlReporter):
         text = "\n\n".join(f"[{r['item_id']}] ({r['kind']}, global update {r['global_step']}, AC version {r['ac_version']})\n"
                            f"reference: {r['reference']}\n" + (f"teacher: {r['teacher']}\n" if r["teacher"] is not None else "") + f"student: {r['student']}" for r in rows)
         label = "Baseline" if progress.epoch == 0 else f"Epoch {progress.epoch_number - (self.epoch_origin or 0)}"
-        key = f"completions_{'baseline' if progress.epoch == 0 else 'epoch'}_{progress.epoch_number:03d}"
+        key = f"completions_update_{progress.global_step:06d}"
+        label += f" · update {progress.global_step}"
         self.set_text(key, f"{label} inspection samples", text)
         self.render()
 
@@ -241,7 +245,7 @@ class ActivationContextTrainingReporter(HtmlReporter):
         self.render(force=True)
 
     def report_training(self, stats: "ActivationContextTrainingStats") -> None:
-        self.set_status(phase="complete", peak_memory=f"{stats.peak_memory_bytes / 2**30:.2f} GB", elapsed=_duration(stats.duration_s), remaining="done")
+        self.set_status(phase=stats.stop_reason, peak_memory=f"{stats.peak_memory_bytes / 2**30:.2f} GB", train_call_elapsed=_duration(stats.duration_s), update_eta="finished" if stats.stop_reason == "complete" else "stopped")
         self._clear_status("evaluation_items")
         self.set_text("training_summary", "Training summary", json.dumps(stats.summarize(), indent=2, default=str))
         self.render(force=True)
````

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title"><a href="trajectory_implementation/activation/ac_model/ac_model_study.py">activation/ac_model/ac_model_study.py</a></span>
    <span class="card-oneliner">Toggle question visibility in both QA tasks; harvest text-only teacher SFT with preserved thinking/tool calls, while retaining matched references and recent-text budgets.</span>
  </summary>

````diff-python
diff --git a/activation/ac_model/ac_model_study.py b/activation/ac_model/ac_model_study.py
index be3ab47..19ba2cd 100644
--- a/activation/ac_model/ac_model_study.py
+++ b/activation/ac_model/ac_model_study.py
@@ -238,17 +238,24 @@ class ActivationContextStudyGenerator:
         self, runs: list["AgentRunResult"], *, loss_kind: t.Literal["kl", "sft"] = "kl",
         selection: t.Literal["score_1", "score_1_or_unscored", "all"] | None = None,
         weight_of: t.Callable[["AgentRunResult"], float] | None = None,
+        text_only: bool = False,
     ) -> list[ActivationContextTrainingItem]:
-        """One whole-history item per nonempty segment, preserving every selected recorded output."""
+        """Harvest whole segments; text-only SFT retains ordinary context and trains only the reader.
+
+        Text-only outputs are rendered with the student's tokenizer, including recorded thinking
+        and tool calls. The default AC path preserves the original sampled token IDs exactly.
+        """
         from ..agent_training.agent_training_utils import activation_messages_of
         from ..agent.rollout_caching import config_key
         if loss_kind not in ("kl", "sft"):
             raise ValueError(loss_kind)
+        if text_only and loss_kind != "sft":
+            raise ValueError("Text-only rollout rehearsal uses SFT")
         selection = selection or ("all" if loss_kind == "kl" else "score_1")
         if selection not in ("all", "score_1", "score_1_or_unscored"):
             raise ValueError(selection)
         report = self.harvest_report = {"roots": len(runs), "runs": 0, "candidates": 0, "accepted": 0,
-                                      "loss_kind": loss_kind, "selection": selection, "skipped": {}}
+                                      "loss_kind": loss_kind, "selection": selection, "text_only": text_only, "skipped": {}}
         items = []
         for root in runs:
             if selection != "all" and root.score != 1 and not (selection == "score_1_or_unscored" and root.score is None):
@@ -267,19 +274,26 @@ class ActivationContextStudyGenerator:
                 if source_model is None:
                     raise ValueError(f"Unknown recorded tokenizer source {run.agent_config.model_name!r}")
                 source_tokenizer = source_model.tokenizer
-                if (source_tokenizer.get_vocab() != self.tokenizer.get_vocab()
+                if not text_only and (source_tokenizer.get_vocab() != self.tokenizer.get_vocab()
                         or source_tokenizer.special_tokens_map != self.tokenizer.special_tokens_map
                         or source_tokenizer.chat_template != self.tokenizer.chat_template):
                     raise ValueError("Recorded assistant tokens require a compatible tokenizer and chat template")
                 tools = self._run_tools(run)
                 for segment_index, segment in enumerate([*run.compactions, run]):
                     report["candidates"] += 1
+                    if text_only and segment.ac_model_name:
+                        raise ValueError("Text-only rehearsal needs trajectories collected without AC")
+                    chat_kwargs = ({"enable_thinking": True} if text_only else
+                                   source_model.chat_template_kwargs(segment.agent_config.call_kwargs))
                     targets = []
                     for step in segment.trajectory:
                         assistants = [message for message in step.get("messages") or [] if message.get("role") == "assistant"]
                         if step.get("role") == "assistant":
-                            ids = list(step.get("token_ids") or [])
-                            if len(assistants) != 1 or not ids:
+                            if len(assistants) != 1:
+                                raise ValueError("Recorded assistant step needs exactly one assistant message")
+                            ids = (assistant_target_ids(self.tokenizer, assistants[0], tools, chat_kwargs)
+                                   if text_only else list(step.get("token_ids") or []))
+                            if not ids:
                                 raise ValueError("Recorded assistant step needs one message and its nonempty sampled token IDs; regenerate this run")
                             targets.append(ids)
                         elif assistants:
@@ -287,7 +301,7 @@ class ActivationContextStudyGenerator:
                     if not targets:
                         self._harvest_skip("empty_segment")
                         continue
-                    config = replace(segment.agent_config, ac_model_name=self.ac_name)
+                    config = replace(segment.agent_config, ac_model_name=None if text_only else self.ac_name)
                     student = activation_messages_of(segment, config)
                     start = len(segment.prompt_messages)
                     if segment.ac_model_name:
@@ -300,7 +314,7 @@ class ActivationContextStudyGenerator:
                         if any(len(direct_parts(messages)) != len(spans) for messages, spans in recorded):
                             raise ValueError("Recorded AC spans need matching raw activation parts; regenerate this run")
                     teacher_prefix = self._expand_parts(student[:start]) if loss_kind == "kl" else None
-                    if not segment.ac_model_name:
+                    if not segment.ac_model_name and not text_only:
                         # Text runs also compress their complete initial frame, retaining raw injected parts within it.
                         frame, suffix = student[:start], student[start:]
                         system = [message for message in frame if message.get("role") == "system"]
@@ -315,13 +329,15 @@ class ActivationContextStudyGenerator:
                         teacher = teacher_prefix + self._expand_parts(student[start:])
                         teacher_start = len(teacher_prefix)
                     item = ActivationContextTrainingItem(
-                        item_id=f"rollout:{source_key}:{segment_index}", kind="compaction",
+                        item_id=f"rollout:{source_key}:{segment_index}" + (":text_only" if text_only else ""),
+                        kind="agent_sft" if text_only else "compaction",
                         in_context_messages=teacher, ac_messages=deepcopy(student),
                         in_context_start=teacher_start, ac_start=start, assistant_token_ids=targets,
                         tools=tools, weight=weight, dataset_id="agent_runs", doc_ids=[root_key],
                         info={"source": "agent_run", "source_key": source_key, "root_key": root_key,
-                              "chat_template_kwargs": source_model.chat_template_kwargs(segment.agent_config.call_kwargs),
-                              "segment": segment_index, "score": root.score, "completion_source": "recorded",
+                              "chat_template_kwargs": chat_kwargs, "text_only": text_only,
+                              "segment": segment_index, "score": root.score,
+                              "completion_source": "recorded_text_student_tokens" if text_only else "recorded",
                               "channels": sorted({part.get("kind") or "unknown" for part in direct_parts(student)}),
                               "ac_model_name": segment.ac_model_name, "ac_model_version": segment.ac_model_version})
                     ActivationContextTrainer(self.harness, ActivationContextTrainingConfig(loss_kind=loss_kind)).build_example(self.ac_model, item)
@@ -486,6 +502,7 @@ class ActivationContextStudyGenerator:
         ratios_range: tuple[float, float] = (1.0 / 16.0, 1.0 / 32.0),
         seed: int = 0,
         caching_id: str | None = None,
+        *, encoder_sees_question: bool = False,
     ) -> list[ActivationContextTrainingItem]:
         """
         Study questions (`traj_qa`) on the dataset's trajectory chunks; per question a window of the
@@ -509,47 +526,70 @@ class ActivationContextStudyGenerator:
             distractors = [self._with_source_task(candidate, self._chunk_window(candidate, None, budget)) for candidate in candidates[:num_distractors]]
             ratio = self._sample_ratio(rng, ratios_range)
             depth = rng.randint(min(depth_range), max(depth_range))
-            question = example.query
-            def part_for(source: DatasetDocument, messages: list[dict]) -> dict:
-                kwargs = source.trajectory_kwargs or {}                              # the frame the trajectory ran with, when the loader kept it
-                frame = [{"role": "system", "content": kwargs["system_prompt"]}] if kwargs.get("system_prompt") else []
-                part = ac_part(frame + messages, self.ac_name, ratio)
-                if kwargs.get("tools"):
-                    part["tools"] = deepcopy(kwargs["tools"])
-                if depth >= 2:
-                    part = ac_part(frame + [messages[0], {"role": "user", "content": [part]}], self.ac_name, ratio)
+            gold_position = rng.randrange(num_distractors + 1)
+            item = self.trajectory_qa_item(document, window, example.query, example.gold_answers[0],
+                item_id=f"traj_qa:{dataset_id}:{seed}:{item_index}", compression_ratio=ratio, depth=depth,
+                distractors=list(zip(candidates[:num_distractors], distractors)), gold_position=gold_position,
+                encoder_sees_question=encoder_sees_question)
+            item.info.update(budget=budget, example_id=example.example_id, gold_chunk_text=chunk.chunk_text,
+                             window_tokens=self.count_tokens(render_messages(window)),
+                             window_over_budget=self.count_tokens(render_messages(window)) > budget)
+            items.append(item)
+        return items
+
+    def trajectory_qa_item(
+        self, document: DatasetDocument, messages: list[dict], question: str, answer: str, *,
+        item_id: str, compression_ratio: float = 1.0 / 16.0, depth: int = 1,
+        distractors: t.Sequence[tuple[DatasetDocument, list[dict]]] = (), gold_position: int = 0,
+        encoder_sees_question: bool = False,
+    ) -> ActivationContextTrainingItem:
+        """Pair a preselected history with a question; optionally expose it to every encoder level.
+
+        Select and freeze `messages` before synthesizing questions for a question-blind study. The
+        full-context reference receives the same source frames, histories and distractor order.
+        """
+        if not messages or not 0 <= gold_position <= len(distractors) or depth < 1:
+            raise ValueError("Trajectory QA needs a nonempty history, positive depth and valid source position")
+        sources = list(distractors)
+        sources.insert(gold_position, (document, messages))
+        prefix = f"Task: {question}\n\n"
+        suffix = f"{TRAJ_QA_INSTRUCTIONS}\nQuestion: {question}"
+        content = [{"type": "text", "text": prefix}]
+        full = [prefix]
+        encoder_question = [{"role": "user", "content": f"Question: {question}"}] if encoder_sees_question else []
+        for number, (source, history) in enumerate(sources, start=1):
+            kwargs = source.trajectory_kwargs or {}
+            history = self._with_source_task(source, history)
+            frame = self._system_messages(source)
+            part = ac_part(deepcopy(frame + history + encoder_question), self.ac_name, compression_ratio)
+            if kwargs.get("tools"):
+                part["tools"] = deepcopy(kwargs["tools"])
+            if depth >= 2:
+                part = ac_part(deepcopy(frame + [history[0]]) + [{"role": "user", "content": [part]}] + deepcopy(encoder_question),
+                               self.ac_name, compression_ratio)
                 if kwargs.get("tools"):
                     part["tools"] = deepcopy(kwargs["tools"])
-                return part
-
-            parts = [("gold", part_for(document, window))] + [("distractor", part_for(candidate, messages))
-                                                                for candidate, messages in zip(candidates[:num_distractors], distractors)]
-            rng.shuffle(parts)
-            content = [{"type": "text", "text": f"Task: {question}\n\n"}]
-            for number, (_, part) in enumerate(parts, start=1):
-                content.extend([{"type": "text", "text": f"Trajectory {number}:\n"}, part, {"type": "text", "text": "\n\n"}])
-            content.append({"type": "text", "text": f"{TRAJ_QA_INSTRUCTIONS}\nQuestion: {question}"})
-            source_frame = tools_in_system_text(self._system_messages(document) + window, (document.trajectory_kwargs or {}).get("tools"))
-            in_context_user = {"role": "user", "content": f"Trajectory:\n{render_messages(source_frame)}\n\n{TRAJ_QA_INSTRUCTIONS}\nQuestion: {question}"}
-            system = [{"role": "system", "content": TRAJ_QA_SYSTEM_PROMPT}]
-            output = self.dialect.rendering.assistant_message("", [{"id": "answer", "name": "submit_answer", "arguments": {"answer": example.gold_answers[0].strip()}}])
-            items.append(ActivationContextTrainingItem(
-                item_id=f"traj_qa:{dataset_id}:{seed}:{item_index}",
-                kind="traj_qa",
-                in_context_messages=system + [in_context_user, output],
-                ac_messages=system + [{"role": "user", "content": content}, deepcopy(output)],
-                in_context_start=len(system) + 1, ac_start=len(system) + 1,
-                assistant_token_ids=[assistant_target_ids(self.tokenizer, output, [submit_answer_definition()])],
-                tools=[submit_answer_definition()],
-                dataset_id=dataset_id,
-                doc_ids=[document.doc_id] + [candidate.doc_id for candidate in candidates[:num_distractors]],
-                info={"depth": depth, "budget": budget, "distractors": num_distractors, "ratio": ratio,
-                      "gold_position": next(i for i, (kind, _) in enumerate(parts) if kind == "gold"), "example_id": example.example_id,
-                      "gold_chunk_text": chunk.chunk_text, "answer": example.gold_answers[0],
-                      "window_tokens": self.count_tokens(render_messages(window)),
-                      "window_over_budget": self.count_tokens(render_messages(window)) > budget},
-            ))
-        return items
+            label = f"Trajectory {number}:\n"
+            content.extend([{"type": "text", "text": label}, part, {"type": "text", "text": "\n\n"}])
+            source_frame = tools_in_system_text(deepcopy(frame + history), kwargs.get("tools"))
+            full.extend([label, render_messages(source_frame), "\n\n"])
+        content.append({"type": "text", "text": suffix})
+        full.append(suffix)
+        tools = [submit_answer_definition()]
+        output = self.dialect.rendering.assistant_message("", [{"id": "answer", "name": "submit_answer", "arguments": {"answer": answer.strip()}}])
+        system = [{"role": "system", "content": TRAJ_QA_SYSTEM_PROMPT}]
+        return ActivationContextTrainingItem(
+            item_id=item_id, kind="traj_qa",
+            in_context_messages=system + [{"role": "user", "content": "".join(full)}, output],
+            ac_messages=system + [{"role": "user", "content": content}, deepcopy(output)],
+            in_context_start=len(system) + 1, ac_start=len(system) + 1,
+            assistant_token_ids=[assistant_target_ids(self.tokenizer, output, tools)], tools=tools,
+            dataset_id=document.dataset_id,
+            doc_ids=[document.doc_id] + [source.doc_id for source, _ in distractors],
+            info={"depth": depth, "distractors": len(distractors), "ratio": compression_ratio,
+                  "gold_position": gold_position, "question": question, "answer": answer.strip(),
+                  "matched_full_context": True, "encoder_sees_question": encoder_sees_question},
+        )
 
     # ------------------------------------------------------------------------------------------ RAG QA
     def generate_rag_qa_samples(
@@ -561,23 +601,38 @@ class ActivationContextStudyGenerator:
         ratios_range: tuple[float, float] = (1.0 / 16.0, 1.0 / 32.0),
         seed: int = 0,
         caching_id: str | None = None,
+        *, source_mode: t.Literal["chunk", "window", "document", "related_chunks"] = "chunk",
+        max_source_tokens: int | None = None, max_source_chunks: int = 3,
+        prompt_kind: t.Literal["qa", "code", "numeric", "constraints"] = "qa",
+        require_multiple_sources: bool = False, max_questions_per_source: int = 1,
+        study_context: str | None = None, progress: t.Callable[[dict], None] | None = None,
+        encoder_sees_question: bool = True,
     ) -> list[ActivationContextTrainingItem]:
-        """
-        Study questions on text chunks; the in-context prefix is the gold passage and the question,
-        the AC prefix one part holding the gold and bm25 distractor passages (the gold's document
-        excluded) shuffled up to the token budget. Depth 2 wraps that part once more.
-        """
+        """Synthesize questions, retain all selected evidence, and match AC/full-context distractors."""
         rng = random.Random(seed)
         index = self.harness.dataset_manager.dataset_indexes[dataset_id]
-        examples = self._study_examples(dataset_id, num_samples, seed, DataModality.TEXT, caching_id)
+        examples = self.harness.dataset_manager.synthesize_study_examples_qa(
+            dataset_id, num_samples, base_seed=seed, caching_id=caching_id, modality=DataModality.TEXT,
+            source_mode=source_mode, max_source_tokens=max_source_tokens, max_source_chunks=max_source_chunks,
+            prompt_kind=prompt_kind, require_multiple_sources=require_multiple_sources, max_questions_per_source=max_questions_per_source,
+            study_context=study_context, progress=progress)
         ranked = index.bm25_query_many_frozen([example.query for example in examples], top_k=64,
-                                              excluded_doc_ids=[list(example.positive_doc_ids or []) for example in examples]) if examples else []
+                                              excluded_doc_ids=[list(dict.fromkeys([*(example.positive_doc_ids or []),
+                                                  *(u["doc_id"] for u in (example.study_source or {}).get("units", []))])) for example in examples]) if examples else []
         items = []
         for item_index, (example, candidates) in enumerate(zip(examples, ranked)):
-            gold = index.chunks[example.positive_chunk_ids[0]]
+            if not example.gold_answers or not example.positive_chunk_ids:
+                continue
+            source = example.study_source
+            if source:
+                positives = [DatasetDocumentChunk(f"{example.example_id}:source:{i}", u["doc_id"], dataset_id, u["text"], 0)
+                             for i, u in enumerate(source["units"])]
+            else:
+                positives = [index.chunks[key] for key in example.positive_chunk_ids]
+            gold = positives[0]
             budget = rng.randint(min(distractors_token_range), max(distractors_token_range))
-            passages = [gold]
-            total = self.count_tokens(gold.chunk_text)
+            passages = list(positives)
+            total = sum(self.count_tokens(p.chunk_text) for p in positives)
             for candidate in candidates:
                 cost = self.count_tokens(candidate.chunk_text)
                 if total + cost > budget:
@@ -588,14 +643,14 @@ class ActivationContextStudyGenerator:
             ratio = self._sample_ratio(rng, ratios_range)
             depth = rng.randint(min(depth_range), max(depth_range))
             question = example.query
-            task_message = {"role": "user", "content": f"Task: {question}"}
+            encoder_task = [{"role": "user", "content": f"Task: {question}"}] if encoder_sees_question else []
             passages_text = "\n\n".join(f"Passage {number}:\n{passage.chunk_text}" for number, passage in enumerate(passages, start=1))
-            part = ac_part([task_message, {"role": "user", "content": passages_text}], self.ac_name, ratio)
+            part = ac_part(deepcopy(encoder_task) + [{"role": "user", "content": passages_text}], self.ac_name, ratio)
             if depth >= 2:
-                part = ac_part([task_message, {"role": "user", "content": [part]}], self.ac_name, ratio)
+                part = ac_part(deepcopy(encoder_task) + [{"role": "user", "content": [part]}], self.ac_name, ratio)
             content = [{"type": "text", "text": f"Task: {question}\n\nPassages:\n"}, part,
                        {"type": "text", "text": f"\n\n{RAG_QA_INSTRUCTIONS}\nQuestion: {question}"}]
-            in_context_user = {"role": "user", "content": f"Passage:\n{gold.chunk_text}\n\n{RAG_QA_INSTRUCTIONS}\nQuestion: {question}"}
+            in_context_user = {"role": "user", "content": f"Task: {question}\n\nPassages:\n{passages_text}\n\n{RAG_QA_INSTRUCTIONS}\nQuestion: {question}"}
             system = [{"role": "system", "content": RAG_QA_SYSTEM_PROMPT}]
             output = self.dialect.rendering.assistant_message("", [{"id": "answer", "name": "submit_answer", "arguments": {"answer": example.gold_answers[0].strip()}}])
             items.append(ActivationContextTrainingItem(
@@ -609,18 +664,21 @@ class ActivationContextStudyGenerator:
                 dataset_id=dataset_id,
                 doc_ids=sorted({passage.doc_id for passage in passages}),
                 info={"depth": depth, "budget": budget, "passages": len(passages), "ratio": ratio,
-                      "gold_position": passages.index(gold), "example_id": example.example_id, "gold_chunk_text": gold.chunk_text, "answer": example.gold_answers[0]},
+                      "gold_position": passages.index(gold), "example_id": example.example_id, "gold_chunk_text": "\n\n".join(p.chunk_text for p in positives), "answer": example.gold_answers[0],
+                      "passages_text": passages_text, "study_source": source, "matched_full_context": True,
+                      "encoder_sees_question": encoder_sees_question},
             ))
         return items
 
     def generate_reconstruction_samples(
         self,
         passages: list[dict],
-        ratio: float = 1.0 / 16.0,
+        ratio: float | None = None,
         variant: str = "reconstruct",
         cue_fraction: float = 0.2,
         seed: int = 0,
         system_prompt: str = RECONSTRUCTION_SYSTEM_PROMPT,
+        *, ratios_range: tuple[float, float] = (1.0 / 10.0, 1.0 / 30.0),
     ) -> list[ActivationContextTrainingItem]:
         """
         Dense, oracle-free items: the reader sees a passage only through its compressed rows and reproduces it.
@@ -641,6 +699,7 @@ class ActivationContextStudyGenerator:
             raise ValueError("variant must be 'reconstruct' or 'continue'")
         if not 0 < cue_fraction < 1:
             raise ValueError("cue_fraction must be in (0, 1)")
+        rng = random.Random(seed)
         items = []
         for index, passage in enumerate(passages):
             if passage.get("variant_only") not in (None, variant):
@@ -650,7 +709,9 @@ class ActivationContextStudyGenerator:
             if not text:
                 continue
             child = _first_child_span(segments) if segments else None
-            part_ratio = passage.get("ratio", ratio) if segments else ratio
+            part_ratio = passage.get("ratio", ratio)
+            if part_ratio is None:
+                part_ratio = self._sample_ratio(rng, ratios_range)
             nested_info = {"children": sum(1 for s in segments if s["kind"] == "child"), "child_ratios": [s["ratio"] for s in segments if s["kind"] == "child"]} if segments else {}
             if variant == "continue":
                 cut = _cue_cut(text, cue_fraction) if child is None else child[0] + _cue_cut(text[child[0]:child[1]], cue_fraction)
@@ -683,12 +744,18 @@ class ActivationContextStudyGenerator:
         return items
 
     def transform_for_eval_reference(self, items: list[ActivationContextTrainingItem],
-                                     kind: t.Literal["no_context", "recent_text"]) -> list[ActivationContextTrainingItem]:
+                                     kind: t.Literal["no_context", "recent_text", "full_context"]) -> list[ActivationContextTrainingItem]:
         """Preserve prefixes/suffixes and targets; QA's text reference is oracle gold, not retrieval."""
-        if kind not in ("no_context", "recent_text"):
+        if kind not in ("no_context", "recent_text", "full_context"):
             raise ValueError(kind)
         output = []
         for item in items:
+            if kind == "full_context":
+                if item.in_context_messages is None:
+                    raise ValueError(f"{item.item_id}: full context requires an in-context view")
+                output.append(replace(item, item_id=item.item_id + ":full_context", ac_messages=deepcopy(item.in_context_messages),
+                                      ac_start=item.in_context_start, info=dict(item.info, reference_label="full context, same distractors" if item.info.get("matched_full_context") else "in-context reference")))
+                continue
             messages = deepcopy(item.ac_messages)
             part_index = 0
             for message in messages:
@@ -708,7 +775,12 @@ class ActivationContextStudyGenerator:
                             ids = self.tokenizer.encode(item.info.get("gold_chunk_text", ""), add_special_tokens=False)[:rows]
                         else:
                             ids = []
-                        pieces.append(self.tokenizer.decode(ids))
+                        text = self.tokenizer.decode(ids)
+                        # A cut through a byte token can grow when decoded and tokenized again.
+                        while ids and len(self.tokenizer.encode(text, add_special_tokens=False)) > rows:
+                            ids = ids[1:] if item.kind == "compaction" else ids[:-1]
+                            text = self.tokenizer.decode(ids)
+                        pieces.append(text)
                     part_index += 1
                 message["content"] = "".join(pieces)
             output.append(replace(item, item_id=item.item_id + ":" + kind, ac_messages=messages,
````

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title"><a href="trajectory_implementation/activation/ac_model/ac_model_training.py">activation/ac_model/ac_model_training.py</a></span>
    <span class="card-oneliner">Preserve immutable teachers and exact resume state; apply measured SFT execution choices without changing the logical update or evaluation.</span>
  </summary>

````diff-python
diff --git a/activation/ac_model/ac_model_training.py b/activation/ac_model/ac_model_training.py
index 7eacc2e..cc143d7 100644
--- a/activation/ac_model/ac_model_training.py
+++ b/activation/ac_model/ac_model_training.py
@@ -5,6 +5,7 @@ import random
 import math
 import hashlib
 import json
+import os
 import shutil
 from fractions import Fraction
 from pathlib import Path
@@ -26,7 +27,8 @@ from ..common.data_syncing import resolve_path
 from .ac_model import ROLLOUT, TRAINING, ActivationContextModel
 from ..common.ac_parts import strip_ac_parts
 from .ac_model_utils import (ACOutput, EncodeRequest, direct_parts, prepare_training_history,
-                             validate_paired_histories, validate_part_capacity)
+                             validate_paired_histories, validate_part_capacity, validate_execution_policy,
+                             execution_groups, execution_batch_loss, AdamMomentPlacement)
 
 if t.TYPE_CHECKING:
     from ..harness import HarnessRuntime
@@ -62,6 +64,10 @@ class ActivationContextTrainingItem:
     info: dict = field(default_factory=dict)   # generator details (depth, ratio, budgets, gold position, ...)
 
 
+class _TrainingTimeLimit(Exception):
+    """Internal control flow: save a partial pair and leave the training residency."""
+
+
 @dataclass
 class ActivationContextTrainingConfig:
     """
@@ -92,10 +98,12 @@ class ActivationContextTrainingConfig:
     no_context_penalty_fraction: float = 0.25  # share of training examples carrying the penalty each update; the term is scaled by 1/fraction (unbiased)
     teacher_cache_top_k: int = 32              # stored teacher (KL) and no-context reference (penalty): top-k log-probs + the remainder mass per target position,
                                                # the gold token always among them; the KL is between the k+1-way coarsened distributions (a lower bound of the exact KL)
+    teacher_cache_block_examples: int | None = None  # opt-in bounded preparation; KL uses a persisted frozen reader adapter, references are kept for one block + fixed panels
     deltas_off_panel: bool = False             # deep models: evaluate the reporting items a second time at every panel with the rows kept and the deltas removed ("deltas_off"): held minus deltas_off is the deltas' contribution
     updates_per_epoch: int = 4                 # maximum gradient steps per epoch when examples_per_update is unset
     examples_per_update: int | None = None     # when set, overrides updates_per_epoch: ceil(items / examples_per_update) steps
     micro_batch_examples: int = 1              # SFT: examples per right-padded reader forward (1 = one example at a time)
+    execution_policy: dict | None = None       # measured training-only batching/residency policy from tune_ac_execution
     max_example_tokens: int = 72_000           # explicit error; no dropping or truncation
     logits_chunk_tokens: int | None = None     # completion positions per lm_head chunk
     fp32_head_matmul: bool = False
@@ -103,13 +111,19 @@ class ActivationContextTrainingConfig:
     gradient_checkpointing_min_tokens: int | None = None
     seed: int = 0
     num_epochs: int = 2
+    shuffle_examples: bool = True
+    max_duration_seconds: float | None = None  # Includes preparation/panels; reserve up to 60s for a paired save.
     reporting_interval: float = 0.1
     completion_samples: int = 4
     completion_max_new_tokens: int = 64
 
     def __post_init__(self) -> None:
+        if self.max_duration_seconds is not None and self.max_duration_seconds <= 0:
+            raise ValueError("max_duration_seconds must be positive")
         if self.loss_kind not in ("kl", "sft") or self.max_example_tokens < 1 or self.teacher_cache_top_k < 1:
             raise ValueError("Invalid loss kind, example capacity, or teacher cache size (the stored targets need a positive top-k)")
+        if self.teacher_cache_block_examples is not None and self.teacher_cache_block_examples < 1:
+            raise ValueError("teacher_cache_block_examples must be positive")
         if self.no_context_penalty_weight < 0 or not 0 < self.no_context_penalty_fraction <= 1:
             raise ValueError("no_context_penalty_weight must be nonnegative and no_context_penalty_fraction in (0, 1]")
         if self.learning_rate_target_lora < 0 or self.learning_rate_ac < 0:
@@ -124,6 +138,10 @@ class ActivationContextTrainingConfig:
             raise ValueError("micro_batch_examples must be positive")
         if self.micro_batch_examples > 1 and self.loss_kind != "sft":
             raise ValueError("micro_batch_examples > 1 is only implemented for loss_kind='sft'")
+        if self.execution_policy is not None:
+            validate_execution_policy(self.execution_policy)
+            if self.loss_kind != "sft" or self.micro_batch_examples != 1:
+                raise ValueError("execution_policy requires SFT and micro_batch_examples=1 (fixed evaluation forwards)")
         if self.completion_samples < 0 or self.completion_max_new_tokens < 1:
             raise ValueError("Invalid completion inspection budget")
 
@@ -160,6 +178,7 @@ class EvalItemSummary(t.TypedDict, total=False):
     target_logp: list[float]            # the student's log-prob of every target token, in order
     teacher_gold_nll: float | None      # the same under the stored teacher (KL only)
     teacher_target_logp: list[float] | None
+    target_kl: list[float] | None       # per-target coarsened KL, evaluation only
 
 
 class EvalKindSummary(t.TypedDict, total=False):
@@ -225,6 +244,7 @@ def reporting_boundaries(steps: int, interval: float) -> list[int]:
 
 @dataclass
 class ActivationContextTrainingStats:
+    stop_reason: str = "complete"
     loss_kind: str = "kl"
     loss: list[float] = field(default_factory=list)
     loss_by_kind: dict[str, list[float]] = field(default_factory=dict)
@@ -289,6 +309,7 @@ class ActivationContextTrainingStats:
         tokens = sum(self.step_tokens)
         seconds = sum(self.step_seconds)
         return {
+            "stop_reason": self.stop_reason,
             "loss_kind": self.loss_kind, "loss_first": self.loss[0] if self.loss else None,
             "loss_last": self.loss[-1] if self.loss else None,
             "loss_by_kind": {kind: sum(values) / len(values) for kind, values in self.loss_by_kind.items() if values},
@@ -337,9 +358,12 @@ class ActivationContextTrainer:
         self.teacher_cache: dict[tuple[str, str, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}   # prepared-token identity -> (ids [C, k], log-probs [C, k], log remainder [C]), CPU
         self.teacher_target_logp: dict[tuple[str, str, int], torch.Tensor] = {}   # same identity -> the stored teacher's log-prob of each target token [C], CPU
         self._teacher_texts: dict[str, str] = {}                                   # item id -> the stored teacher's greedy completion, sampled once at baseline
+        self._teacher_snapshot: dict | None = None                               # bounded KL: immutable adapter and identity, shared by all blocks and resumes
+        self._fixed_teacher_keys: set[tuple[str, str, int]] = set()                # reporting targets survive block eviction
         self._penalty_rng = random.Random(self.config.seed)                        # which training examples carry the no-context penalty
         self._penalty_term: torch.Tensor | None = None                             # set by _example_loss: the weighted penalty of the last example, if sampled
         self._last_hits: torch.Tensor | None = None                                # set by the chunked losses: argmax == target per target position
+        self._last_token_kl: torch.Tensor | None = None
         self._payload_bytes = 0                                                    # deep models: delta bytes handed to the reader (reset per train call)
         self._row_rms: dict[int, list[torch.Tensor]] = {}                          # deep models: residual RMS at the row positions per layer since the last trace (0-d tensors)
 
@@ -403,22 +427,62 @@ class ActivationContextTrainer:
         *,
         validation_data: t.Sequence[ActivationContextTrainingItem] = (),
         reset_optimizer: bool = False,
+        reporting_callback: t.Callable[[ActivationContextTrainingProgress, ActivationContextEvalSummary | None], None] | None = None,
     ) -> ActivationContextTrainingStats:
         """Complete epochs in one residency scope; optimizer/global step persist across calls."""
         self._training_ac_name = ac_model_name
         config = self.config
         if not training_data:
             raise ValueError("No eligible AC training data")
+        if config.loss_kind == "kl" and config.resume_optimizer and config.teacher_cache_block_examples is None:
+            raise ValueError("Resuming KL requires teacher_cache_block_examples and the original frozen teacher snapshot; "
+                             "refusing to rebuild targets from the updated reader")
         self.teacher_cache.clear()                 # teacher targets belong to this invocation only: the reader as it is at this call's start
         self.teacher_target_logp.clear()
         self._teacher_texts.clear()
+        self._teacher_snapshot = None
+        self._fixed_teacher_keys.clear()
         manager = self.harness.module_manager
         ac_model = manager.get_ac_model(ac_model_name)
+        if config.teacher_cache_block_examples is not None and config.resume_optimizer and not ac_model.config.checkpoint_path:
+            raise ValueError("Bounded-target resume requires an AC checkpoint and optimizer state")
         target = ac_model.target
         target_lora = ac_model.config.target_model_lora_name
         start_epoch = self._next_epoch_number(ac_model)
         stats = ActivationContextTrainingStats(loss_kind=config.loss_kind, start_epoch=start_epoch, items=len(training_data))
         started = time.monotonic()
+        deadline = (started + config.max_duration_seconds - min(60.0, config.max_duration_seconds * .1)
+                    if config.max_duration_seconds is not None else None)
+        self._training_deadline = deadline
+        epoch_number, completed_step, record = start_epoch, 0, None
+        def report_callback(progress, summary):
+            self._check_training_deadline()
+            if reporting_callback is None:
+                return
+            modes = [(base, base.training, base.is_gradient_checkpointing) for base in bases]
+            trainability = [(p, p.requires_grad) for base in [*bases, ac_model.modules] for p in base.parameters()]
+            residency = [(loaded, loaded.current_model_device) for loaded in dict.fromkeys([target, ac_model.side])]
+            previous_mode = ac_model.mode
+            try:
+                with ExitStack() as adapters, torch.no_grad():
+                    adapters.enter_context(manager.lora_context(target.model_config.model_name, target_lora))
+                    if ac_model.side is not target:
+                        adapters.enter_context(manager.lora_context(ac_model.side.model_config.model_name, ac_model.config.base_side_model_lora_name))
+                    reporting_callback(progress, summary)
+            finally:
+                for loaded, previous_device in residency:
+                    loaded.model_to_device(previous_device)
+                ac_model.modules.to(device)
+                ac_model.set_mode(previous_mode)
+                for base, training, checkpointed in modes:
+                    base.train(training)
+                    disable_dropout(base)
+                    if checkpointed:
+                        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
+                    elif base.is_gradient_checkpointing:
+                        base.gradient_checkpointing_disable()
+                for parameter, enabled in trainability:
+                    parameter.requires_grad_(enabled)
         if reporter is not None:
             reporter.set_status(phase="building training examples", examples=str(len(training_data)))
             reporter.render(force=True)
@@ -426,6 +490,8 @@ class ActivationContextTrainer:
         examples = [self.build_example(ac_model, item) for item in training_data]
         reporting_examples = [self.build_example(ac_model, item) for item in reporting_data]
         validation_examples = [self.build_example(ac_model, item) for item in validation_data]
+        if config.teacher_cache_block_examples is not None:
+            self._fixed_teacher_keys = {example.teacher_cache_key for example in [*reporting_examples, *validation_examples]}
         stats.examples = len(examples)
         stats.completion_tokens = sum(example.num_completion for example in examples)
         stats.teacher_tokens = sum(len(example.teacher_ids) for example in examples)
@@ -436,11 +502,13 @@ class ActivationContextTrainer:
         fixed_items = list(reporting_data[:config.completion_samples])
         kernel_scope = ExitStack()
         optimizer = None
+        moment_placement = None
         device = None
         bases = []
         if reporter is not None:
             reporter.initialize_training(config, stats)
         try:
+            self._check_training_deadline()
             placement_started = time.monotonic()
             if reporter is not None:
                 reporter.set_status(phase="model placement and kernel preparation")
@@ -462,17 +530,24 @@ class ActivationContextTrainer:
             self._target_gradient_checkpointing_min_tokens = variables["target_gradient_checkpointing_min_tokens"]
             ac_model._side_gradient_checkpointing_min_tokens = variables["side_gradient_checkpointing_min_tokens"]
             ac_model._training_gradient_checkpointing = config.gradient_checkpointing
+            if config.execution_policy is not None:
+                stats.autotuning["execution_policy"] = dict(config.execution_policy)
             self._check_capacity(ac_model, [*examples, *reporting_examples, *validation_examples])
             self._payload_bytes = 0
             if config.loss_kind == "kl":
-                # The teacher of this call is the reader as it is now; nothing after this line recomputes it.
+                # Eager mode stores every target now. Bounded mode preserves the same teacher as an immutable adapter.
+                if config.teacher_cache_block_examples is not None:
+                    self._prepare_teacher_snapshot(ac_model, start_epoch)
                 if reporter is not None:
                     reporter.set_status(phase="teacher targets")
                     reporter.render(force=True)
-                stats.teacher_targets = self._store_teacher_targets(ac_model, [*examples, *reporting_examples, *validation_examples], device, reporter)
+                teacher_examples = [*reporting_examples, *validation_examples] if self._teacher_snapshot else [*examples, *reporting_examples, *validation_examples]
+                stats.teacher_targets = self._store_teacher_targets(ac_model, teacher_examples, device, reporter,
+                        lora=self._teacher_snapshot["lora_name"] if self._teacher_snapshot else "reader")
                 if reporter is not None:
                     reporter.report_teacher_targets(stats.teacher_targets)
-            penalty_examples = [example.penalty for example in examples if example.penalty is not None]
+            penalty_examples = ([example.penalty for example in examples if example.penalty is not None]
+                                if config.teacher_cache_block_examples is None else [])
             if penalty_examples:
                 # The penalty's reference is the base with no adapter on the no-context history: fixed for the whole call, and kept next to
                 # the checkpoints so a resumed run skips the reference pass.
@@ -480,14 +555,16 @@ class ActivationContextTrainer:
                     reporter.set_status(phase="no-context reference")
                     reporter.render(force=True)
                 reference_path = resolve_path(f"AC_MODELS/{ac_model.name}/no_context_reference.pt", create=False)
+                reference_identity = {"model": target.model_config.model_id, "revision": getattr(target.model.config, "_commit_hash", None),
+                    "tokenizer": getattr(target.tokenizer, "name_or_path", None), "vocab": len(target.tokenizer), "reference": "base_no_context"}
                 if reference_path.exists():
-                    loaded = self._merge_teacher_targets(reference_path, {example.teacher_cache_key for example in penalty_examples})
+                    loaded = self._merge_teacher_targets(reference_path, {example.teacher_cache_key for example in penalty_examples}, reference_identity)
                     print(f"AC resume: {loaded} no-context references loaded from {reference_path}", flush=True)
                 stats.no_context_reference = self._store_teacher_targets(ac_model, penalty_examples, device, reporter, reporting_name="no-context reference", lora=None)
                 if stats.no_context_reference["examples"] or not reference_path.exists():
                     keys = {example.teacher_cache_key for example in penalty_examples}
                     reference_path.parent.mkdir(parents=True, exist_ok=True)
-                    torch.save({"top_k": config.teacher_cache_top_k, "targets": {k: v for k, v in self.teacher_cache.items() if k in keys},
+                    torch.save({"identity": reference_identity, "top_k": config.teacher_cache_top_k, "targets": {k: v for k, v in self.teacher_cache.items() if k in keys},
                                 "target_logp": {k: v for k, v in self.teacher_target_logp.items() if k in keys}}, reference_path)
             ac_parameters = ac_model.trainable_parameters()
             target_parameters = manager.lora_parameters(target_lora)
@@ -506,6 +583,8 @@ class ActivationContextTrainer:
                     saved_layout = [len(group["params"]) for group in state["optimizer"]["param_groups"]]
                     current_layout = [len(group["params"]) for group in groups]
                     if saved_layout == current_layout:
+                        if self._teacher_snapshot and state.get("teacher_snapshot_identity") != self._teacher_snapshot["identity"]:
+                            raise ValueError("Optimizer checkpoint does not match the frozen teacher snapshot")
                         optimizer.load_state_dict(state["optimizer"])
                         for group, wanted in zip(optimizer.param_groups, groups):   # load_state_dict also restores the saved hyper-parameters: the arm's rates win
                             group["lr"] = group["base_lr"] = wanted["lr"]
@@ -513,14 +592,23 @@ class ActivationContextTrainer:
                         optimizer_state_to(optimizer, device)
                         stats.resumed_from = str(saved)
                     else:
+                        if config.teacher_cache_block_examples is not None:
+                            raise ValueError("Bounded-target resume requires matching optimizer parameter groups")
                         # The group layout changed since the checkpoint (a checkpoint from the full-featured branch with extra groups, a migrated
                         # checkpoint): the moments cannot be mapped, so the optimizer starts fresh and only the update clock resumes. A requeued run never dies here.
                         print(f"AC resume: optimizer state discarded (layout changed: saved groups {saved_layout}, current {current_layout}); "
                               f"fresh moments, update clock resumed at {int(state['updates_done'])}", flush=True)
                         stats.resumed_from = f"{saved} (moments discarded: layout changed)"
                     self.updates_done[ac_model_name] = int(state["updates_done"])
+                    if "penalty_rng_state" in state:
+                        self._penalty_rng.setstate(state["penalty_rng_state"])
                     self.resumed.add(ac_model_name)
                     print(f"AC resume: optimizer moments and {state['updates_done']} updates restored from {saved}", flush=True)
+                elif config.teacher_cache_block_examples is not None:
+                    raise FileNotFoundError(f"Bounded-target resume requires optimizer state: {saved}")
+            if config.execution_policy is not None and config.execution_policy.get("optimizer_moments", "gpu") == "cpu":
+                moment_placement = AdamMomentPlacement(optimizer)
+                moment_placement.park()
             if device.type == "cuda":
                 torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True   # fp32 GEMMs (LoRA path, gates) on tensor cores
             if device.type == "cuda":
@@ -542,6 +630,8 @@ class ActivationContextTrainer:
                 stats.validation_baseline = self._evaluate(ac_model, list(validation_data), device, reporter, "validation")
                 if reporter is not None:
                     reporter.report_eval(baseline, "validation", stats.validation_baseline)
+            report_callback(baseline, stats.reporting)
+            self._check_training_deadline()
             stats.baseline_samples = self._sample_completions(ac_model, fixed_items, baseline)
             stats.baseline_seconds = time.monotonic() - phase
             if reporter is not None and stats.baseline_samples:
@@ -556,10 +646,23 @@ class ActivationContextTrainer:
                         self.updates_done.get(ac_model_name, 0), time.monotonic() - epoch_started, time.monotonic() - started)
                 record = ActivationContextEpochStats(progress(0))
                 order = list(range(len(examples)))
-                random.Random(config.seed + epoch_number).shuffle(order)
+                if config.shuffle_examples:
+                    random.Random(config.seed + epoch_number).shuffle(order)
+                completed_step = config.resume_skip_steps if epoch == 1 else 0
                 for step in range(1, steps + 1):
                     if epoch == 1 and step <= config.resume_skip_steps:
                         continue        # trained by the previous attempt (same seeded order); the mid-epoch checkpoint carries the moments and the update count
+                    self._check_training_deadline()
+                    if config.teacher_cache_block_examples is not None:
+                        mini = order[(step - 1) * per_step:step * per_step]
+                        needed = [examples[index] for index in mini]
+                        if any((config.loss_kind == "kl" and example.teacher_cache_key not in self.teacher_cache)
+                               or (example.penalty is not None and example.penalty.teacher_cache_key not in self.teacher_cache) for example in needed):
+                            block_steps = max(1, config.teacher_cache_block_examples // per_step)
+                            indexes = order[(step - 1) * per_step:(step - 1 + block_steps) * per_step]
+                            if reporter is not None:
+                                reporter.report_phase(progress(step - 1), "preparing next training block")
+                            self._prepare_teacher_block(ac_model, [examples[index] for index in indexes], device, reporter)
                     if reporter is not None:
                         reporter.report_phase(progress(step - 1), "training")
                     step_started = time.monotonic()
@@ -568,7 +671,13 @@ class ActivationContextTrainer:
                     optimizer.zero_grad(set_to_none=True)
                     totals = {"loss": 0.0, "agreement": 0.0, "tokens": 0, "penalty": []}
                     batched = config.micro_batch_examples > 1 and config.loss_kind == "sft"
-                    if batched:
+                    policy = config.execution_policy
+                    if policy is not None:
+                        # Draw in the original logical order, independent of execution grouping.
+                        penalty_flags = {index: bool(config.penalty_active and examples[index].penalty is not None
+                            and self._penalty_rng.random() < config.no_context_penalty_fraction) for index in mini}
+                        groups = [[mini[index] for index in group] for group in execution_groups([examples[index] for index in mini], policy)]
+                    elif batched:
                         mini = sorted(mini, key=lambda index: len(examples[index].student_ids))   # same examples, less padding per micro-batch
                         groups = [mini[i:i + config.micro_batch_examples] for i in range(0, len(mini), config.micro_batch_examples)]
                     else:
@@ -576,7 +685,19 @@ class ActivationContextTrainer:
                     for group in groups:
                         group_examples = [examples[index] for index in group]
                         example_started = time.monotonic()
-                        if batched:
+                        if policy is not None:
+                            old_side = ac_model._side_gradient_checkpointing_min_tokens
+                            old_target = self._target_gradient_checkpointing_min_tokens
+                            try:
+                                ac_model._side_gradient_checkpointing_min_tokens = policy.get("side_checkpoint_tokens", old_side)
+                                self._target_gradient_checkpointing_min_tokens = policy.get("target_checkpoint_tokens", old_target)
+                                results = execution_batch_loss(self, ac_model, group_examples, device,
+                                    [penalty_flags[index] for index in group], policy)
+                            finally:
+                                ac_model._side_gradient_checkpointing_min_tokens = old_side
+                                self._target_gradient_checkpointing_min_tokens = old_target
+                            outcomes = [(result["objective"], None, result["penalty"]) for result in results]
+                        elif batched:
                             outcomes = [(result["objective"], None, result["penalty"]) for result in self._batch_loss(ac_model, group_examples, device)]
                         else:
                             objective, agreement, _, _ = self._example_loss(ac_model, group_examples[0], device)
@@ -605,14 +726,15 @@ class ActivationContextTrainer:
                         for parameter in target_parameters:
                             parameter.grad = None
                     norm_ac = self._grad_norm(ac_parameters); norm_reader = self._grad_norm(target_parameters)
-                    head_norms = self._head_grad_norms(ac_model) if ac_model.is_deep else None       # before clipping, like norm_ac
+                    trace_due = ac_model.is_deep and self._intervention_trace_due(self.updates_done.get(ac_model_name, 0))
+                    head_norms = self._head_grad_norms(ac_model) if trace_due else None           # before clipping, only when consumed
                     grad_norm = float(torch.nn.utils.clip_grad_norm_(ac_parameters + ([] if reader_frozen else target_parameters), config.max_grad_norm))
                     set_learning_rates(optimizer, self.updates_done.get(ac_model_name, 0), config.warmup_updates,
                                        total_updates=planned_total, schedule=config.lr_schedule, final_fraction=config.lr_final_fraction)
                     stats.grad_norm_ac.append(norm_ac); stats.grad_norm_reader.append(norm_reader)
                     if self.updates_done.get(ac_model_name, 0) % 25 == 0:
                         stats.gate_trace.append({"update": self.updates_done.get(ac_model_name, 0), **{gate_names.get(id(parameter), str(index)): float(parameter.detach()) for index, parameter in enumerate(gate_parameters)}})
-                    if ac_model.is_deep and self._intervention_trace_due(self.updates_done.get(ac_model_name, 0)):
+                    if trace_due:
                         stats.intervention_trace.append(self._intervention_trace_entry(ac_model, self.updates_done.get(ac_model_name, 0), head_norms))
                     encoder_has_grad = any(parameter.grad is not None for parameter in ac_parameters)
                     try:
@@ -624,6 +746,7 @@ class ActivationContextTrainer:
                     seconds = time.monotonic() - step_started
                     record.training_seconds += seconds
                     stats.steps += 1
+                    completed_step = step
                     stats.loss.append(totals["loss"] / len(mini))
                     if config.loss_kind == "kl":
                         stats.kl.append(stats.loss[-1])
@@ -640,22 +763,25 @@ class ActivationContextTrainer:
                           f"{config.loss_kind} loss {stats.loss[-1]:.4f}, {seconds:.1f}s", flush=True)
                     if reporter is not None:
                         reporter.report_step(progress(step), stats)
+                    self._check_training_deadline()
                     if config.checkpoint_every_updates and config.checkpoint_every_epoch and step < steps and step % config.checkpoint_every_updates == 0:
                         self._save_partial(ac_model, epoch_number, step, steps)
-                    if step in boundaries and reporting_data:
+                    if step in boundaries and (reporting_data or reporting_callback is not None):
                         phase = time.monotonic()
                         if reporter is not None:
                             reporter.report_phase(progress(step), "reporting")
-                        record.reporting = self._evaluate(ac_model, list(reporting_data), device, reporter, "reporting")
+                        record.reporting = self._evaluate(ac_model, list(reporting_data), device, reporter, "reporting") if reporting_data else None
                         stats.reporting = record.reporting
-                        if reporter is not None:
+                        if reporter is not None and record.reporting is not None:
                             reporter.report_eval(progress(step), "reporting", record.reporting)
                         if deltas_off_panel:
                             record.deltas_off = self._evaluate(ac_model, list(reporting_data), device, reporter, "deltas_off", deltas_off=True)
                             stats.deltas_off = record.deltas_off
                             if reporter is not None:
                                 reporter.report_eval(progress(step), "deltas_off", record.deltas_off)
+                        report_callback(progress(step), record.reporting)
                         record.reporting_seconds += time.monotonic() - phase
+                self._check_training_deadline()
                 if validation_data:
                     phase = time.monotonic()
                     if reporter is not None:
@@ -688,9 +814,33 @@ class ActivationContextTrainer:
                 ac_model.epoch_number = epoch_number
             stats.teacher_cache_bytes = sum(sum(tensor.numel() * tensor.element_size() for tensor in entry) for entry in self.teacher_cache.values())
             stats.payload_bytes = self._payload_bytes
+        except _TrainingTimeLimit:
+            self._training_deadline = None
+            stats.stop_reason = "duration limit" if stats.steps else "duration limit; no updates"
+            if reporter is not None:
+                reporter.set_status(phase="saving partial checkpoint")
+                reporter.render(force=True)
+            saved_at = time.monotonic()
+            if stats.steps == 0 and config.resume_skip_steps:
+                # The incoming pair already owns its step and optimizer metadata. Do not relabel its weights as step zero.
+                stats.checkpoint_path = ac_model.config.checkpoint_path
+                stats.target_lora_checkpoint_path = manager.get_lora_config(target_lora).checkpoint_path
+                completed_step = config.resume_skip_steps
+            else:
+                stats.checkpoint_path, stats.target_lora_checkpoint_path = self._save_partial(ac_model, epoch_number, completed_step, steps)
+            stopped = ActivationContextTrainingProgress(len(stats.epochs) + 1, config.num_epochs, epoch_number, completed_step, steps,
+                self.updates_done.get(ac_model_name, 0), time.monotonic() - (epoch_started if record is not None else started), time.monotonic() - started)
+            record = record or ActivationContextEpochStats(stopped)
+            record.progress = stopped
+            record.checkpoint_path, record.target_lora_checkpoint_path = stats.checkpoint_path, stats.target_lora_checkpoint_path
+            record.checkpoint_seconds += time.monotonic() - saved_at
+            stats.epochs.append(record)
         finally:
+            self._training_deadline = None
             kernel_scope.close()
             cleanup_started = time.monotonic()
+            if moment_placement is not None:
+                moment_placement.close()
             if optimizer is not None:
                 optimizer_state_to(optimizer, SOURCE_DEVICE)
             for base in bases:
@@ -717,6 +867,95 @@ class ActivationContextTrainer:
             reporter.report_training(stats)
         return stats
 
+    def _check_training_deadline(self) -> None:
+        deadline = getattr(self, "_training_deadline", None)
+        if deadline is not None and time.monotonic() >= deadline:
+            raise _TrainingTimeLimit()
+
+    def _teacher_identity(self, ac_model: ActivationContextModel, folder: Path) -> dict:
+        files = sorted(path for path in folder.iterdir() if path.name == "adapter_config.json" or path.suffix in (".safetensors", ".bin"))
+        if not (folder / "adapter_config.json").is_file() or not any(path.suffix in (".safetensors", ".bin") for path in files):
+            raise FileNotFoundError(f"Frozen teacher adapter is incomplete: {folder}")
+        digest = hashlib.sha256()
+        for path in files:
+            digest.update(path.name.encode())
+            with path.open("rb") as stream:
+                for block in iter(lambda: stream.read(1024 * 1024), b""):
+                    digest.update(block)
+        target = ac_model.target
+        return {"model": target.model_config.model_id, "revision": getattr(target.model.config, "_commit_hash", None),
+                "tokenizer": getattr(target.tokenizer, "name_or_path", None), "vocab": len(target.tokenizer),
+                "adapter_sha256": digest.hexdigest(), "top_k": self.config.teacher_cache_top_k}
+
+    def _activate_teacher_snapshot(self, ac_model: ActivationContextModel, folder: Path, identity: dict) -> None:
+        if self._teacher_identity(ac_model, folder) != identity:
+            raise ValueError("Frozen teacher adapter/model/tokenizer identity does not match its checkpoint")
+        manager = self.harness.module_manager
+        source = manager.get_lora_config(ac_model.config.target_model_lora_name)
+        name = ac_model.name + "__teacher_" + identity["adapter_sha256"][:16]
+        if name not in manager.lora_configs:
+            manager.register_lora(name, source.model_name, rank=source.rank, alpha=source.alpha, dropout=source.dropout,
+                                  target_modules=source.target_modules, checkpoint_path=str(folder))
+        for parameter in manager.lora_parameters(name):
+            parameter.requires_grad_(False)
+        disable_dropout(manager.ensure_lora(name))
+        self._teacher_snapshot = {"folder": str(folder), "identity": identity, "lora_name": name, "ac_name": ac_model.name}
+
+    def _load_teacher_snapshot(self, ac_model: ActivationContextModel, checkpoint: Path) -> None:
+        path = checkpoint / "teacher_snapshot.json"
+        if not path.is_file():
+            raise FileNotFoundError(f"A frozen teacher snapshot is required; refusing to use the resumed reader: {path}")
+        data = json.loads(path.read_text())
+        if data.get("version") != 1:
+            raise ValueError("Unsupported frozen teacher snapshot version")
+        folder = (checkpoint / data["adapter_path"]).resolve()
+        self._activate_teacher_snapshot(ac_model, folder, data["identity"])
+
+    def _prepare_teacher_snapshot(self, ac_model: ActivationContextModel, epoch: int) -> None:
+        if self.config.resume_optimizer:
+            if not ac_model.config.checkpoint_path:
+                raise ValueError("Frozen-teacher resume requires an AC checkpoint")
+            self._load_teacher_snapshot(ac_model, Path(ac_model.config.checkpoint_path))
+            return
+        manager = self.harness.module_manager
+        source = manager.get_lora_config(ac_model.config.target_model_lora_name)
+        folder = resolve_path(f"AC_MODELS/{ac_model.name}/teacher_snapshots/epoch_{epoch:03d}_{time.time_ns()}", create=False)
+        previous_path = source.checkpoint_path
+        try:
+            manager.save_lora(source.model_name, source.lora_name, str(folder))
+        finally:
+            source.checkpoint_path = previous_path
+        self._activate_teacher_snapshot(ac_model, folder, self._teacher_identity(ac_model, folder))
+
+    def _prepare_teacher_block(self, ac_model: ActivationContextModel, examples: t.Sequence[_Example], device: torch.device,
+                               reporter: "ActivationContextTrainingReporter | None" = None) -> None:
+        # Fixed reporting entries stay resident; training and base-reference entries never accumulate across blocks.
+        for key in list(self.teacher_cache):
+            if key not in self._fixed_teacher_keys:
+                self.teacher_cache.pop(key)
+                self.teacher_target_logp.pop(key, None)
+        if self.config.loss_kind == "kl":
+            if self._teacher_snapshot is None:
+                raise RuntimeError("Bounded KL preparation requires its immutable teacher snapshot")
+            self._store_teacher_targets(ac_model, examples, device, reporter, lora=self._teacher_snapshot["lora_name"])
+        penalties = [example.penalty for example in examples if example.penalty is not None]
+        if penalties:
+            self._store_teacher_targets(ac_model, penalties, device, reporter, reporting_name="next block no-context reference", lora=None)
+
+    def _save_teacher_snapshot(self, folder: Path) -> None:
+        if self._teacher_snapshot is None:
+            return
+        data = {"version": 1, "adapter_path": os.path.relpath(self._teacher_snapshot["folder"], folder),
+                "identity": self._teacher_snapshot["identity"]}
+        temporary = folder / "teacher_snapshot.json.tmp"
+        temporary.write_text(json.dumps(data, indent=2) + "\n")
+        temporary.replace(folder / "teacher_snapshot.json")
+        # Small fixed panels remain directly readable with teacher='stored'. Arbitrary readouts use teacher='snapshot'.
+        keys = self._fixed_teacher_keys & self.teacher_cache.keys()
+        if keys:
+            torch.save({"top_k": self.config.teacher_cache_top_k, "targets": {key: self.teacher_cache[key] for key in keys},
+                        "target_logp": {key: self.teacher_target_logp[key] for key in keys}}, folder / "teacher_targets.pt")
+
     def _next_epoch_number(self, ac_model: ActivationContextModel) -> int:
         manager = self.harness.module_manager
         roots = [ac_model.checkpoint_folder().parent, manager.checkpoint_folder(ac_model.config.target_model_lora_name).parent]
@@ -736,7 +975,9 @@ class ActivationContextTrainer:
             raise FileExistsError(f"Epoch {epoch_number} already has checkpoint files")
         ac_path = ac_model.save(str(ac_folder), epoch_number=epoch_number)
         target_path = manager.save_lora(ac_model.target.model_config.model_name, target_lora, str(target_folder))
-        if self.teacher_cache and self.config.loss_kind == "kl":
+        if self._teacher_snapshot:
+            self._save_teacher_snapshot(ac_folder)
+        elif self.teacher_cache and self.config.loss_kind == "kl":
             self.save_teacher_targets(ac_folder / "teacher_targets.pt")     # the reference this run was trained against (an SFT run's cache only holds penalty references)
         for source, latest in ((ac_folder, ac_model.checkpoint_folder()), (target_folder, manager.checkpoint_folder(target_lora))):
             if latest.exists():
@@ -744,8 +985,16 @@ class ActivationContextTrainer:
             shutil.copytree(source, latest)
         optimizer = self.optimizers.get(ac_model.name)
         if optimizer is not None:               # resume material: moments + the update count (the LR schedule and the reader freeze window read it)
-            torch.save({"optimizer": optimizer.state_dict(), "updates_done": self.updates_done.get(ac_model.name, 0), "epoch_number": epoch_number},
+            torch.save({"optimizer": optimizer.state_dict(), "updates_done": self.updates_done.get(ac_model.name, 0), "epoch_number": epoch_number,
+                        "penalty_rng_state": self._penalty_rng.getstate(),
+                        "teacher_snapshot_identity": self._teacher_snapshot["identity"] if self._teacher_snapshot else None},
                        ac_folder / "optimizer.pt")
+        record = ac_folder.parent / "completed_epoch.json"
+        temporary = record.with_suffix(".json.tmp")
+        temporary.write_text(json.dumps({"epoch_number": epoch_number, "ac_checkpoint": ac_relative,
+                                        "target_lora_checkpoint": target_relative}, indent=2) + "\n")
+        temporary.replace(record)       # publish the complete pair before retiring any previous recovery checkpoint
+        if optimizer is not None:
             for previous in sorted(ac_folder.parent.glob("epoch_*/optimizer.pt")):
                 if previous.parent != ac_folder:
                     previous.unlink()           # only the newest epoch keeps its optimizer (about 2x the adapter size)
@@ -755,14 +1004,9 @@ class ActivationContextTrainer:
         partial_record = ac_folder.parent / "partial_epoch.json"
         if partial_record.exists() and json.loads(partial_record.read_text())["epoch_number"] <= epoch_number:
             partial_record.unlink()
-        record = ac_folder.parent / "completed_epoch.json"
-        temporary = record.with_suffix(".json.tmp")
-        temporary.write_text(json.dumps({"epoch_number": epoch_number, "ac_checkpoint": ac_relative,
-                                        "target_lora_checkpoint": target_relative}, indent=2) + "\n")
-        temporary.replace(record)       # only this record declares the numbered pair complete
         return ac_path, target_path
 
-    def _save_partial(self, ac_model: ActivationContextModel, epoch_number: int, step: int, steps: int) -> None:
+    def _save_partial(self, ac_model: ActivationContextModel, epoch_number: int, step: int, steps: int) -> tuple[str, str]:
         """Mid-epoch checkpoint: adapters as of `step` of `epoch_number` (saved with epoch_number - 1 so a resume restarts this epoch), optimizer.pt,
         and AC_MODELS/<name>/partial_epoch.json; earlier partials of the same epoch are removed."""
         manager = self.harness.module_manager
@@ -770,14 +1014,24 @@ class ActivationContextTrainer:
         leaf = f"epoch_{epoch_number:03d}_step_{step:05d}"
         ac_relative, target_relative = f"AC_MODELS/{ac_model.name}/{leaf}", f"LORAS/{target_lora}/{leaf}"
         ac_folder, target_folder = resolve_path(ac_relative, create=False), resolve_path(target_relative, create=False)
+        published = ac_folder.parent / "partial_epoch.json"
+        if published.exists():
+            pair = json.loads(published.read_text())
+            if pair.get("ac_checkpoint") == ac_relative and pair.get("target_lora_checkpoint") == target_relative:
+                if not ac_folder.exists() or not target_folder.exists():
+                    raise RuntimeError("Published partial checkpoint is missing one side")
+                return str(ac_folder), str(target_folder)  # Same completed update; keep the recoverable pair intact.
         for folder in (ac_folder, target_folder):
             if folder.exists():
                 shutil.rmtree(folder)
         ac_model.save(str(ac_folder), epoch_number=epoch_number - 1)
         manager.save_lora(ac_model.target.model_config.model_name, target_lora, str(target_folder))
+        self._save_teacher_snapshot(ac_folder)
         optimizer = self.optimizers.get(ac_model.name)
         if optimizer is not None:
-            torch.save({"optimizer": optimizer.state_dict(), "updates_done": self.updates_done.get(ac_model.name, 0), "epoch_number": epoch_number - 1, "step": step},
+            torch.save({"optimizer": optimizer.state_dict(), "updates_done": self.updates_done.get(ac_model.name, 0), "epoch_number": epoch_number - 1, "step": step,
+                        "penalty_rng_state": self._penalty_rng.getstate(),
+                        "teacher_snapshot_identity": self._teacher_snapshot["identity"] if self._teacher_snapshot else None},
                        ac_folder / "optimizer.pt")
         record = ac_folder.parent / "partial_epoch.json"
         temporary = record.with_suffix(".json.tmp")
@@ -789,10 +1043,13 @@ class ActivationContextTrainer:
                 if previous.name != leaf:
                     shutil.rmtree(previous)
         print(f"AC checkpoint: epoch {epoch_number} step {step}/{steps} saved to {ac_folder}", flush=True)
+        return str(ac_folder), str(target_folder)
 
-    def _merge_teacher_targets(self, path: Path, keys: set | None) -> int:
+    def _merge_teacher_targets(self, path: Path, keys: set | None, identity: dict | None = None) -> int:
         """Add the saved reference entries for `keys` (None: all) that are not stored yet (same top-k required). Returns how many were added."""
         data = torch.load(path, map_location="cpu", weights_only=False)
+        if identity is not None and data.get("identity") != identity:
+            return 0
         if data["top_k"] != self.config.teacher_cache_top_k:
             return 0
         added = 0
@@ -821,6 +1078,7 @@ class ActivationContextTrainer:
                 if base.is_gradient_checkpointing:
                     base.gradient_checkpointing_disable()
             for item in items:
+                self._check_training_deadline()
                 example = self.build_example(ac_model, item)
                 texts = {}
                 # The stored teacher is the reader at baseline: its completion is sampled once (before any update) and reused.
@@ -836,11 +1094,20 @@ class ActivationContextTrainer:
                         present = [(span, request) for span, request in zip(example.spans, example.part_requests) if span[1] <= prefix_length]
                         outputs = ac_model.encode_batch([request for _, request in present], with_layers=True) if present else []
                         embeds, payload = self._assemble_student(embeds, [span for span, _ in present], outputs, embeds.dtype)
-                    lora = ac_model.config.target_model_lora_name
+                    lora = self._teacher_snapshot["lora_name"] if side == "teacher" and self._teacher_snapshot else ac_model.config.target_model_lora_name
                     peft_model = manager.ensure_lora(lora)
                     budget = min(self.config.completion_max_new_tokens, capacity - prefix_length) if capacity else self.config.completion_max_new_tokens
                     generation = dict(max_new_tokens=budget, do_sample=False, use_cache=True,
                                       pad_token_id=target.tokenizer.pad_token_id if target.tokenizer.pad_token_id is not None else target.tokenizer.eos_token_id)
+                    # A chat turn can end before the model's end-of-text token.
+                    eot_token = target.model_config.model_description.eot_token
+                    if eot_token is not None:
+                        eos_ids = peft_model.generation_config.eos_token_id
+                        eos_ids = [eos_ids] if isinstance(eos_ids, int) else list(eos_ids or [])
+                        eot_id = target.tokenizer.convert_tokens_to_ids(eot_token)
+                        if eot_id not in eos_ids:
+                            eos_ids.append(eot_id)
+                        generation["eos_token_id"] = eos_ids
                     with manager.lora_context(target.model_config.model_name, lora):
                         if payload is not None:
                             output = generate_with_interventions(peft_model, inputs_embeds=embeds[None], interventions=[payload], **generation)
@@ -880,11 +1147,12 @@ class ActivationContextTrainer:
         Loss, agreement and target log-probs of the items without gradient (models placed as for training, moved back
         when `release`). KL items need stored teacher targets: `teacher="stored"` (default) fails when an item has none,
         so a reloaded reader is never silently made its own teacher; `teacher="current"` stores missing targets from the
-        reader as it is now (right before training, or after `load_teacher_targets`, that is the intended reference).
+        reader as it is now. `teacher="snapshot"` regenerates bounded blocks from a checkpoint's immutable adapter,
+        allowing arbitrary readouts without retaining all training targets in RAM or every checkpoint.
         A supplied reporter receives this evaluation under reporting_name.
         """
-        if teacher not in ("stored", "current"):
-            raise ValueError("teacher must be 'stored' or 'current'")
+        if teacher not in ("stored", "current", "snapshot"):
+            raise ValueError("teacher must be 'stored', 'current' or 'snapshot'")
         module_manager = self.harness.module_manager
         ac_model = module_manager.get_ac_model(ac_model_name)
         target = ac_model.target
@@ -896,9 +1164,14 @@ class ActivationContextTrainer:
             kernel_scope.enter_context(configure_fla_runtime(variables["fla_config"]))
             module_manager.ensure_lora(ac_model.config.target_model_lora_name)
             device = ac_model.prepare()
+            if teacher == "snapshot" and self.config.loss_kind == "kl":
+                if self._teacher_snapshot is None or self._teacher_snapshot["ac_name"] != ac_model.name:
+                    if not ac_model.config.checkpoint_path:
+                        raise ValueError("Snapshot evaluation requires an AC checkpoint")
+                    self._load_teacher_snapshot(ac_model, Path(ac_model.config.checkpoint_path))
             if teacher == "current" and self.config.loss_kind == "kl":
                 self._store_teacher_targets(ac_model, [self.build_example(ac_model, item) for item in items], device, reporter)
-            summary = self._evaluate(ac_model, items, device, reporter, reporting_name or reference_name or "evaluation")
+            summary = self._evaluate(ac_model, items, device, reporter, reporting_name or reference_name or "evaluation", teacher_snapshot=teacher == "snapshot")
             if reporter is not None:
                 reporter.report_eval(progress, reporting_name or reference_name or "evaluation", summary)
                 if reference_name is not None and summary["items"]:
@@ -915,7 +1188,8 @@ class ActivationContextTrainer:
     # ------------------------------------------------------------------------------------------ internals
     @torch.no_grad()
     def _evaluate(self, ac_model: ActivationContextModel, items: list[ActivationContextTrainingItem], device: torch.device,
-                  reporter: "ActivationContextTrainingReporter | None" = None, reporting_name: str = "evaluation", deltas_off: bool = False) -> ActivationContextEvalSummary:
+                  reporter: "ActivationContextTrainingReporter | None" = None, reporting_name: str = "evaluation", deltas_off: bool = False,
+                  teacher_snapshot: bool = False) -> ActivationContextEvalSummary:
         """`deltas_off`: a deep model's items scored with their rows kept and their per-layer deltas withheld (the deltas' contribution)."""
         if reporter is not None:
             reporter.report_evaluation_progress(reporting_name, 0, len(items))
@@ -939,12 +1213,14 @@ class ActivationContextTrainer:
                     "teacher_tokens": len(example.teacher_ids), "student_tokens": len(example.student_ids),
                     "gold_nll": float(-target_logp.mean()), "target_logp": [round(float(value), 4) for value in target_logp],
                     "teacher_gold_nll": float(-teacher_logp.mean()) if teacher_logp is not None else None,
-                    "teacher_target_logp": [round(float(value), 4) for value in teacher_logp] if teacher_logp is not None else None}
+                    "teacher_target_logp": [round(float(value), 4) for value in teacher_logp] if teacher_logp is not None else None,
+                    "target_kl": [float(value) for value in self._last_token_kl] if is_kl and self._last_token_kl is not None else None}
             rows: list[dict | None] = [None] * len(examples)
             done = 0
             if self.config.micro_batch_examples > 1 and not is_kl:
                 order = sorted(range(len(examples)), key=lambda index: len(examples[index].student_ids))
                 for start in range(0, len(order), self.config.micro_batch_examples):
+                    self._check_training_deadline()
                     group = order[start:start + self.config.micro_batch_examples]
                     for index, result in zip(group, self._batch_loss(ac_model, [examples[index] for index in group], device, with_penalty=False, deltas_off=deltas_off)):
                         rows[index] = row(examples[index], result["objective"], None, result["num"], result["target_logp"], result["hits"])
@@ -953,6 +1229,10 @@ class ActivationContextTrainer:
                         reporter.report_evaluation_progress(reporting_name, done, len(examples))
             else:
                 for index, example in enumerate(examples):
+                    self._check_training_deadline()
+                    if teacher_snapshot and is_kl and index % (self.config.teacher_cache_block_examples or 32) == 0:
+                        block = examples[index:index + (self.config.teacher_cache_block_examples or 32)]
+                        self._prepare_teacher_block(ac_model, [replace(entry, penalty=None) for entry in block], device)
                     loss, agreement, positions, target_logp = self._example_loss(ac_model, example, device, with_penalty=False, deltas_off=deltas_off)
                     rows[index] = row(example, loss, agreement, positions, target_logp, self._last_hits)
                     done += 1
@@ -1034,13 +1314,16 @@ class ActivationContextTrainer:
             reference_checkpointed = (config.gradient_checkpointing and torch.is_grad_enabled()
                                       and len(reference.student_ids) >= getattr(self, "_target_gradient_checkpointing_min_tokens", config.gradient_checkpointing_min_tokens or 0))
             hits = self._last_hits
+            token_kl = self._last_token_kl
             reference_states = target.decoder_forward(reference_embeds[None], None, lora_name=lora, gradient_checkpointing=reference_checkpointed)[0][reference.student_positions]
             penalty, _, _ = self._chunked_kl(None, reference_states, head, cached=cached_reference)
             self._last_hits = hits
+            self._last_token_kl = token_kl
             self._penalty_term = (config.no_context_penalty_weight / config.no_context_penalty_fraction) * penalty
         return loss, agreement, num, target_logp
 
-    def _batch_loss(self, ac_model: ActivationContextModel, batch: t.Sequence[_Example], device: torch.device, with_penalty: bool = True, deltas_off: bool = False) -> list[dict]:
+    def _batch_loss(self, ac_model: ActivationContextModel, batch: t.Sequence[_Example], device: torch.device, with_penalty: bool = True, deltas_off: bool = False,
+                    *, encoded_outputs: list[ACOutput] | None = None, penalty_flags: list[bool] | None = None) -> list[dict]:
         """
         SFT loss of several examples from one right-padded reader forward (and one encoder call for all their parts):
         per example {"objective", "num", "target_logp", "hits", "penalty"}, the values `_example_loss` gives one at a
@@ -1057,7 +1340,9 @@ class ActivationContextTrainer:
         head = base.get_output_embeddings()
         lora = ac_model.config.target_model_lora_name
         requests = [request for example in batch for request in example.part_requests]
-        outputs = ac_model.encode_batch(requests, with_layers=True) if requests else []
+        outputs = encoded_outputs if encoded_outputs is not None else (ac_model.encode_batch(requests, with_layers=True) if requests else [])
+        if len(outputs) != len(requests) or (penalty_flags is not None and len(penalty_flags) != len(batch)):
+            raise ValueError("Precomputed encodings and penalty decisions must match the batch")
         embeds, payloads, cursor_rows = [], [], 0
         for example in batch:
             student_ids = torch.tensor(example.student_ids, device=device)
@@ -1081,7 +1366,8 @@ class ActivationContextTrainer:
                             "target_logp": token_logp[cursor:cursor + num], "hits": hits[cursor:cursor + num], "penalty": None})
             cursor += num
         if with_penalty and config.penalty_active:
-            sampled = [index for index, example in enumerate(batch) if example.penalty is not None and self._penalty_rng.random() < config.no_context_penalty_fraction]
+            sampled = [index for index, example in enumerate(batch) if example.penalty is not None and
+                (penalty_flags[index] if penalty_flags is not None else self._penalty_rng.random() < config.no_context_penalty_fraction)]
             if sampled:
                 references = [batch[index].penalty for index in sampled]
                 cached = []
@@ -1236,6 +1522,7 @@ class ActivationContextTrainer:
             groups = [order[start:start + self.config.micro_batch_examples] for start in range(0, len(order), self.config.micro_batch_examples)]
         done = 0
         for group in groups:
+            self._check_training_deadline()
             if len(group) == 1:
                 ids = torch.tensor(group[0].teacher_ids, device=device, dtype=torch.long)
                 states_by_example = [target.decoder_forward(embedding(ids)[None], None, lora_name=lora, gradient_checkpointing=False)[0][group[0].teacher_positions]]
@@ -1330,6 +1617,7 @@ class ActivationContextTrainer:
         agree = 0.0
         num = student_states.shape[0]
         logps = []
+        token_kls = []
 
         hits = []
 
@@ -1343,7 +1631,10 @@ class ActivationContextTrainer:
             with torch.no_grad():
                 teacher_logp = F.log_softmax(head_logits(teacher_chunk, head, fp32), dim=-1)
             student_logp = F.log_softmax(head_logits(student_chunk, head, fp32), dim=-1)
-            kl_sum = (teacher_logp.exp() * (teacher_logp - student_logp)).sum(dim=-1).sum()
+            values = (teacher_logp.exp() * (teacher_logp - student_logp)).sum(dim=-1)
+            if not torch.is_grad_enabled():
+                token_kls.append(values.detach())
+            kl_sum = values.sum()
             agreement = (teacher_logp.argmax(dim=-1) == student_logp.argmax(dim=-1)).float().sum()
             return kl_sum, agreement.detach(), token_logp(student_logp, targets)
 
@@ -1352,7 +1643,10 @@ class ActivationContextTrainer:
             student_logp = F.log_softmax(head_logits(student_chunk, head, fp32), dim=-1)
             student_top = student_logp.gather(-1, ids.long())
             student_rest = torch.log(torch.clamp(1.0 - student_top.exp().sum(dim=-1), min=1e-12))
-            kl_sum = ((teacher_logp.exp() * (teacher_logp - student_top)).sum(dim=-1) + teacher_rest.exp() * (teacher_rest - student_rest)).sum()
+            values = (teacher_logp.exp() * (teacher_logp - student_top)).sum(dim=-1) + teacher_rest.exp() * (teacher_rest - student_rest)
+            if not torch.is_grad_enabled():
+                token_kls.append(values.detach())
+            kl_sum = values.sum()
             agreement = (ids[:, 0].long() == student_logp.argmax(dim=-1)).float().sum()
             return kl_sum, agreement.detach(), token_logp(student_logp, targets)
 
@@ -1372,6 +1666,7 @@ class ActivationContextTrainer:
             logps.append(chunk_logp)
         if target_ids is not None:
             self._last_hits = torch.cat(hits[:len(logps)]) if hits else target_ids.new_zeros((0,), dtype=torch.bool)
+        self._last_token_kl = torch.cat(token_kls) if token_kls else None
         return total / max(1, num), agree / max(1, num), (torch.cat(logps) if logps else student_states.new_zeros((0,), dtype=torch.float32))
 
     def _autotuning_variables(self, ac_model: ActivationContextModel) -> dict:
````

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title"><a href="trajectory_implementation/activation/ac_model/ac_model_utils.py">activation/ac_model/ac_model_utils.py</a></span>
    <span class="card-oneliner">Keep graph-local encoding reuse, execution grouping and ordinary Adam moment placement in existing utilities.</span>
  </summary>

````diff-python
diff --git a/activation/ac_model/ac_model_utils.py b/activation/ac_model/ac_model_utils.py
index 586c797..ca878df 100644
--- a/activation/ac_model/ac_model_utils.py
+++ b/activation/ac_model/ac_model_utils.py
@@ -555,7 +555,8 @@ class PassageAttention(nn.Module):
         normed = self.norm(passage)
         split = lambda tensor, n: tensor.view(n, self.heads, -1).transpose(0, 1)                   # [heads, n, r / heads]
         q, k, v = split(self.query(self.norm(rows)), rows.shape[0]), split(self.key(normed), passage.shape[0]), split(self.value(normed), passage.shape[0])
-        mixed = F.scaled_dot_product_attention(q, k, v).transpose(0, 1).reshape(rows.shape[0], -1)
+        # A batch dimension lets CUDA use memory-efficient attention for these FP32 readouts.
+        mixed = F.scaled_dot_product_attention(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)).squeeze(0).transpose(0, 1).reshape(rows.shape[0], -1)
         return rows + self.out(mixed)
 
 
@@ -642,6 +643,113 @@ class EncodeRequest(t.NamedTuple):
     tools: list[dict] | None = None            # the definitions the compressed segment ran with (part["tools"]); templated on the side
 
 
+def validate_execution_policy(policy: dict) -> None:
+    """An opt-in measured training policy; never changes the logical update size."""
+    allowed = {"graph_examples", "reader_batch", "encode_batch_tokens", "reuse_encodings",
+               "side_checkpoint_tokens", "target_checkpoint_tokens", "optimizer_moments"}
+    if set(policy) - allowed:
+        raise ValueError(f"Unknown AC execution settings: {sorted(set(policy) - allowed)}")
+    for name in ("graph_examples", "reader_batch", "encode_batch_tokens"):
+        if name in policy and (type(policy[name]) is not int or policy[name] < 1):
+            raise ValueError(f"{name} must be a positive integer")
+    for name in ("side_checkpoint_tokens", "target_checkpoint_tokens"):
+        if name in policy and (type(policy[name]) is not int or policy[name] < 0):
+            raise ValueError(f"{name} must be a nonnegative integer")
+    if "reuse_encodings" in policy and type(policy["reuse_encodings"]) is not bool:
+        raise ValueError("reuse_encodings must be a boolean")
+    if policy.get("optimizer_moments", "gpu") not in ("cpu", "gpu"):
+        raise ValueError("optimizer_moments must be cpu or gpu")
+
+
+def execution_groups(examples: t.Sequence, policy: dict) -> list[list[int]]:
+    """Bound live graphs inside this logical update; preserve its membership and order."""
+    size = policy.get("graph_examples", 1)
+    return [list(range(start, min(start + size, len(examples)))) for start in range(0, len(examples), size)]
+
+
+def execution_batch_loss(trainer, ac_model, examples: t.Sequence, device, penalty_flags: list[bool], policy: dict) -> list[dict]:
+    """Reuse complete differentiable encodings only until this group's single backward.
+
+    Reader batching is independent of encoder batching. Penalties are sampled by the
+    caller in logical item order, before grouping; every returned loss remains an item
+    mean. The caller applies item weights and the logical-update denominator.
+    """
+    if len(penalty_flags) != len(examples):
+        raise ValueError("One penalty decision is required per example")
+    requests, indices, known = [], [], {}
+    for example in examples:
+        positions = []
+        for request in example.part_requests:
+            # Exact ratio, tools and recursive content are all included. No persistent
+            # cache, detached tensors, or payload dictionaries cross a backward call.
+            key = canonical_json(request._asdict()) if policy.get("reuse_encodings", False) else None
+            if key is None or key not in known:
+                index = len(requests)
+                requests.append(request)
+                if key is not None:
+                    known[key] = index
+            else:
+                index = known[key]
+            positions.append(index)
+        indices.append(positions)
+    previous_budget = ac_model.config.encode_batch_max_tokens
+    try:
+        ac_model.config.encode_batch_max_tokens = policy.get("encode_batch_tokens", previous_budget)
+        outputs = ac_model.encode_batch(requests, with_layers=True) if requests else []
+    finally:
+        ac_model.config.encode_batch_max_tokens = previous_budget
+    results = []
+    size = policy.get("reader_batch", 1)
+    for start in range(0, len(examples), size):
+        stop = min(start + size, len(examples))
+        encoded = [outputs[index] for positions in indices[start:stop] for index in positions]
+        results.extend(trainer._batch_loss(ac_model, examples[start:stop], device,
+            encoded_outputs=encoded, penalty_flags=penalty_flags[start:stop]))
+    return results
+
+
+class AdamMomentPlacement:
+    """Park AdamW moments between updates without changing its step or clock tensors."""
+    def __init__(self, optimizer):
+        if not isinstance(optimizer, torch.optim.AdamW):
+            raise TypeError("Moment placement supports AdamW only")
+        if any(group.get("amsgrad") or group.get("capturable") or group.get("differentiable") for group in optimizer.param_groups):
+            raise ValueError("Moment placement requires ordinary non-AMSGrad AdamW")
+        self.optimizer = optimizer
+        self.handles = [optimizer.register_step_pre_hook(self._before), optimizer.register_step_post_hook(self._after)]
+
+    def _move(self, to_cpu: bool) -> None:
+        for parameter, state in self.optimizer.state.items():
+            if not state:
+                continue
+            if set(state) != {"step", "exp_avg", "exp_avg_sq"}:
+                raise ValueError("Unexpected AdamW state")
+            destination = torch.device("cpu") if to_cpu else parameter.device
+            for key in ("exp_avg", "exp_avg_sq"):
+                value = state[key]
+                if value.shape != parameter.shape:
+                    raise ValueError("AdamW moment/parameter shape mismatch")
+                if value.device != destination:
+                    state[key] = value.to(device=destination, non_blocking=False)
+
+    def park(self) -> None:
+        self._move(True)
+
+    def _before(self, optimizer, args, kwargs):
+        self._move(False)
+
+    def _after(self, optimizer, args, kwargs):
+        self.park()
+
+    def close(self) -> None:
+        try:
+            self.park()
+        finally:
+            for handle in self.handles:
+                handle.remove()
+            self.handles.clear()
+
+
 class EncodeQueue:
     """
     One worker thread that batches concurrent no-grad encodes: requests submitted from any thread are
````

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title"><a href="trajectory_implementation/activation/bench/agent_probes/cheap_synthetic_bench.py">activation/bench/agent_probes/cheap_synthetic_bench.py</a></span>
    <span class="card-oneliner">Use the shared questions-array contract; count only usable single-source QA results.</span>
  </summary>

````diff-python
diff --git a/activation/bench/agent_probes/cheap_synthetic_bench.py b/activation/bench/agent_probes/cheap_synthetic_bench.py
index 0fde34c..098172e 100644
--- a/activation/bench/agent_probes/cheap_synthetic_bench.py
+++ b/activation/bench/agent_probes/cheap_synthetic_bench.py
@@ -1,6 +1,6 @@
 """
 Throughput of small generators for a CHEAP_SYNTHETIC phase: the study prompt on real MS-MARCO
-chunks, one JSON {question, answer} per chunk, measured per model and engine concurrency. Writes
+chunks, at most one grounded question per chunk, measured per model and engine concurrency. Writes
 a live report (watch it with `sky watch`) and `cheap_synthetic_results.json` last.
 
     uv run sky exec <node> uv run python -m activation.bench.cheap_synthetic_bench \\
@@ -21,7 +21,7 @@ import torch
 from vllm.sampling_params import StructuredOutputsParams
 
 from activation.common.reporting import HtmlReporter
-from activation.dataset.dataset_study import STUDY_JSON_SCHEMA, make_study_prompt
+from activation.dataset.dataset_study_prompts import make_source_study_prompt
 from activation.dataset.dataset_utils import safe_truncate_embedding_chunk, shuffle_fill_truncate
 from activation.dataset.loaders import MsMarcoDataset
 from activation.harness import FREE_DEVICE, HarnessRuntime, HarnessRuntimeConfig, ModelConfig
@@ -38,13 +38,13 @@ def load_chunks(harness: HarnessRuntime, num_chunks: int, seed: int) -> list[str
 
 
 def conversations_for(snippets: list[str], chunk_input_limit: int) -> list[list[dict]]:
-    system_prompt, instructions, _ = make_study_prompt(None)
+    system_prompt, instructions, _ = make_source_study_prompt(None)
     out = []
     for snippet in snippets:
         snippet = safe_truncate_embedding_chunk(snippet, chunk_input_limit)
         out.append([
             {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
-            {"role": "user", "content": [{"type": "text", "text": f"# Corpus Snippet\n```\n{snippet}\n```\n{instructions}"}]},
+            {"role": "user", "content": [{"type": "text", "text": f"Source 0:\n{snippet}\n\n{instructions}"}]},
         ])
     return out
 
@@ -52,7 +52,7 @@ def conversations_for(snippets: list[str], chunk_input_limit: int) -> list[list[
 def chat_kwargs_for(structured: bool, thinking: bool, seed: int, max_tokens: int) -> dict:
     sampling = {"max_tokens": max_tokens, "seed": seed, "temperature": 0.7, "top_p": 0.9}
     if structured:
-        sampling["structured_outputs"] = StructuredOutputsParams(json=STUDY_JSON_SCHEMA)
+        sampling["structured_outputs"] = StructuredOutputsParams(json=make_source_study_prompt(None)[2])
     return {"sampling_params": sampling, "chat_template_kwargs": {"enable_thinking": thinking}}
 
 
@@ -82,10 +82,10 @@ def run_config(model_id: str, max_num_seqs: int, structured: bool, thinking: boo
             prompt_tokens += output.prompt_token_count
             output_tokens += output.output_token_count
             try:
-                qa = json.loads(output.text)
-                if not str(qa["question"]).strip() or not str(qa["answer"]).strip():
+                questions = json.loads(output.text)["questions"]
+                if len(questions) != 1 or not questions[0]["question"].strip() or not questions[0]["answer"].strip() or questions[0]["evidence_indices"] != [0]:
                     failures += 1
-            except (json.JSONDecodeError, KeyError, TypeError):
+            except (ValueError, KeyError, TypeError, AttributeError):
                 failures += 1
     elapsed = time.time() - start
     sample = outputs[0].text[:400] if outputs else ""
````

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title"><a href="trajectory_implementation/activation/common/reporting.py">activation/common/reporting.py</a></span>
    <span class="card-oneliner">Retain explicit UTC report timestamps.</span>
  </summary>

````diff-python
diff --git a/activation/common/reporting.py b/activation/common/reporting.py
index 2fcd991..e3a0401 100644
--- a/activation/common/reporting.py
+++ b/activation/common/reporting.py
@@ -624,7 +624,7 @@ class HtmlReporter:
             "refresh_seconds": self.refresh_seconds,
             "poll_seconds": self.poll_seconds,
             "finished": self.finished,
-            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
+            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
             "status": dict(self.status),
             "widgets": widgets,
         }
````

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title"><a href="trajectory_implementation/activation/dataset/dataset.py">activation/dataset/dataset.py</a></span>
    <span class="card-oneliner">Retain exact selected source units in QA records.</span>
  </summary>

````diff-python
diff --git a/activation/dataset/dataset.py b/activation/dataset/dataset.py
index 7007038..955a5c8 100644
--- a/activation/dataset/dataset.py
+++ b/activation/dataset/dataset.py
@@ -125,6 +125,7 @@ class DatasetQAExample:
     hard_negative_chunk_ids: list[str] | None = None
     excluded_doc_ids: list[str] | None = None
     oracle_labeled: bool = False
+    study_source: dict | None = None  # Exact selected source units/order/text for synthesized QA.
 
 
 @dataclass
````

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title"><a href="trajectory_implementation/activation/dataset/dataset_manager.py">activation/dataset/dataset_manager.py</a></span>
    <span class="card-oneliner">Retain the bounded synthesis options on the existing manager method.</span>
  </summary>

````diff-python
diff --git a/activation/dataset/dataset_manager.py b/activation/dataset/dataset_manager.py
index dcfc48a..418b23c 100644
--- a/activation/dataset/dataset_manager.py
+++ b/activation/dataset/dataset_manager.py
@@ -61,6 +61,10 @@ class DatasetManager:
         caching_id: str | None = None,
         modality: DataModality | None = None,
         *, progress: t.Callable[[dict], None] | None = None,
+        source_mode: t.Literal["chunk", "window", "document", "related_chunks"] = "chunk",
+        max_source_tokens: int | None = None, max_source_chunks: int = 3,
+        prompt_kind: t.Literal["qa", "code", "numeric", "constraints"] = "qa",
+        require_multiple_sources: bool = False, max_questions_per_source: int = 1,
     ) -> list[DatasetQAExample]:
         """
         Synthesize study questions from the dataset's chunks (see DatasetStudyGenerator.generate_examples_qa).
@@ -68,7 +72,9 @@ class DatasetManager:
         """
         study_generator = self._get_or_create_study_generator(dataset_id)
         seed = self.harness.harness_config.dataset_study_seed if base_seed is None else base_seed
-        return study_generator.generate_examples_qa(num_samples, seed, study_context, caching_id, modality, progress=progress)
+        return study_generator.generate_examples_qa(num_samples, seed, study_context, caching_id, modality, progress=progress, source_mode=source_mode,
+            max_source_tokens=max_source_tokens, max_source_chunks=max_source_chunks, prompt_kind=prompt_kind,
+            require_multiple_sources=require_multiple_sources, max_questions_per_source=max_questions_per_source)
 
 
     def label_study_examples(
````

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title"><a href="trajectory_implementation/activation/dataset/dataset_study.py">activation/dataset/dataset_study.py</a></span>
    <span class="card-oneliner">Retain programmable, cached synthesis; remove the dead old-row converter and prompt re-exports.</span>
  </summary>

````diff-python
diff --git a/activation/dataset/dataset_study.py b/activation/dataset/dataset_study.py
index 12e15a7..ba88260 100644
--- a/activation/dataset/dataset_study.py
+++ b/activation/dataset/dataset_study.py
@@ -6,6 +6,7 @@ gives a caching id (dataset_caching.py). Nothing is written into the dataset obj
 the examples back and decide what to keep.
 """
 import json
+import hashlib
 import random
 import time
 import typing as t
@@ -33,114 +34,7 @@ if t.TYPE_CHECKING:
     from activation.harness import HarnessRuntime
 
 
-STUDY_JSON_SCHEMA = {
-    "type": "object",
-    "properties": {
-        "has_valuable_question": {"type": "boolean"},
-        "question": {"type": ["string", "null"]},
-        "answer": {"type": ["string", "null"]},
-    },
-    "required": ["has_valuable_question", "question", "answer"],
-    "additionalProperties": False,
-}
-
-LABEL_JSON_SCHEMA = {
-    "type": "object",
-    "properties": {
-        "positives": {"type": "array", "items": {"type": "integer"}},
-        "negatives": {"type": "array", "items": {"type": "integer"}},
-    },
-    "required": ["positives", "negatives"],
-    "additionalProperties": False,
-}
-
-
-
-def make_study_prompt(study_context: str | None, modality: DataModality = DataModality.TEXT) -> tuple[str, str, dict]:
-    """
-    Returns system prompt, instructions, json schema. The trajectory variant asks about a fact inside
-    an agent's turns; both let the model declare a snippet worthless (`has_valuable_question: false`).
-    """
-    system_prompt = """
-# Core Guidelines
-- You are a study/exam-like questions formulator.
-- You are given a snippet from the corpus students are expected to study and understand.
-- From this, your goal is to generate a hard question (unanswerable with common knowledge or simple keyword look ups).
-    - This is IMPORTANT: the question should be **hard**, or it won't test the student's learning.
-    - The question should have an accompanying answer.
-- Your priority is: a hard question along with its expected, correct answer.
-- By default, keep your question short to medium-short unless the exam recommends longer questions.
-- When the snippet holds nothing worth asking about (boilerplate, navigation, an empty listing, pure formatting), set has_valuable_question to false and leave question and answer null.
-"""
-    if modality == DataModality.TRAJECTORY:
-        system_prompt += """
-# Trajectory Snippets
-- The snippet is part of an agent's conversation: its reasoning, its tool calls and the tool results.
-- Ask about a specific fact that appears in these turns (a value a tool returned, a decision the agent took, an error it saw, a file or quantity it found), phrased so someone who read the turns can answer it.
-- Never ask something answerable from the task statement alone, and never ask about the formatting of the transcript.
-"""
-
-    if study_context:
-        system_prompt += f"""
-# Exam Context
-The following emphasizes the kind of questions students are expected to handle.
-```
------
-{study_context}
------
-```
-Frame your questions in light of this.
-"""
-    system_prompt += """
-# Response Format
-Your answer must be formatted as a json object like:
-```json
-{
-    "has_valuable_question": true|false,
-    "question": "..."|null,
-    "answer": "..."|null
-}
-```
-"""
-    instructions = f"""
-# Instructions
-Generate the exam-like question for student's study.
-"""
-    return system_prompt, instructions, STUDY_JSON_SCHEMA
-
-
-def make_label_prompt() -> tuple[str, str, dict]:
-    """
-    Returns system prompt, instructions and json schema for the snippet labeling task.
-    """
-    system_prompt = """
-# Core Guidelines
-- You are a relevance judge determining what snippets from a corpus students should pay attention to answer a question.
-- You are given the study question, its reference answer, and numbered corpus snippets.
-- Label each snippet index as:
-    - **positive**: the snippet contains information that clearly helps answer the question.
-    - **negative**: the snippet contains information that clearly does not help answer the question.
-    - Leave out anything where you are unsure.
-- Be careful:
-    - The snippets are intentionally chosen to be *related* to the question without necessarily helping.
-    - Don't mark something as positive just because of shared vocabulary.
-- Every listed index must come from the snippet numbering; never list the same index on both sides.
-"""
-    system_prompt += """
-# Response Format
-Your answer must be formatted as a json object of snippet indices like:
-```json
-{
-    "positives": [0, 2],
-    "negatives": [3, 5, 6]
-}
-```
-Each is a list of indexes that should map to one of the given snippet indices.
-"""
-    instructions = """
-Label the snippets as positives / negatives for this question. Leave out ambiguous ones.
-"""
-    return system_prompt, instructions, LABEL_JSON_SCHEMA
+from .dataset_study_prompts import make_label_prompt, make_source_study_prompt
 
 
 class DatasetStudyGenerator:
@@ -181,126 +75,142 @@ class DatasetStudyGenerator:
         chat_kwargs["sampling_params"] = sampling_params_kwargs
         return chat_kwargs
 
-    def _example_from_row(self, row: dict, kind: str, base_seed: int) -> DatasetQAExample:
-        chunk = self.dataset_index.chunks[row["chunk_id"]]
-        return DatasetQAExample(
-            example_id=f"{self.dataset_id}:study:{kind}:{base_seed}:{row['index']}",
-            dataset_id=self.dataset_id,
-            query=row["question"],
-            gold_answers=[row["answer"]],
-            origin=SYNTHETIC,
-            split=DataSplit.TRAIN,
-            positive_doc_ids=[chunk.doc_id],
-            positive_chunk_ids=[chunk.chunk_id],
-        )
-
+    def _study_source(self, chunk, mode, max_chunks, max_tokens, tokenizer):
+        """Select once and retain exactly the text shown to the generator."""
+        doc = self.loaded_dataset.documents[chunk.doc_id]
+        chunks = sorted(doc.chunks.values(), key=lambda c: c.chunk_start)
+        if mode == "document":
+            if doc.text is None:
+                return None
+            selected = chunks
+            units = [{"doc_id": doc.doc_id, "chunk_ids": [c.chunk_id for c in selected], "text": doc.text}]
+        elif mode == "window":
+            position = next(i for i, c in enumerate(chunks) if c.chunk_id == chunk.chunk_id)
+            start = max(0, position - max_chunks // 2)
+            selected = chunks[start:start + max_chunks]
+            units = []
+            covered = selected[0].chunk_start
+            for candidate in selected:
+                end = candidate.chunk_start + len(candidate.chunk_text)
+                # Keep adjacent chunks independently citable without repeating their overlap.
+                text = (doc.text[covered:end] if doc.text is not None
+                        else candidate.chunk_text[max(0, covered - candidate.chunk_start):])
+                if text:
+                    units.append({"doc_id": doc.doc_id, "chunk_ids": [candidate.chunk_id], "text": text})
+                covered = max(covered, end)
+        elif mode == "related_chunks":
+            selected = [chunk] + self.dataset_index.bm25_query_many_frozen(
+                [chunk.chunk_text], top_k=max_chunks - 1, excluded_doc_ids=[[chunk.doc_id]])[0] if max_chunks > 1 else [chunk]
+            units = [{"doc_id": c.doc_id, "chunk_ids": [c.chunk_id], "text": c.chunk_text} for c in selected]
+        else:
+            text = safe_truncate_embedding_chunk(self.dataset_index.get_chunk_section(chunk), self.chunk_input_limit)
+            units = [{"doc_id": chunk.doc_id, "chunk_ids": [chunk.chunk_id], "text": text}]
+        text = "\n\n".join(f"Source {i}:\n{unit['text']}" for i, unit in enumerate(units))
+        if max_tokens is not None and len(tokenizer.encode(text, add_special_tokens=False)) > max_tokens:
+            return None  # Explicit skip, never silently shorten a whole document.
+        return {"mode": mode, "units": units, "text": text}
 
     def generate_examples_qa(
-        self,
-        num_samples: int,
-        base_seed: int = 0,
-        study_context: str | None = None,
-        caching_id: str | None = None,
-        modality: DataModality | None = None,
+        self, num_samples: int, base_seed: int = 0, study_context: str | None = None,
+        caching_id: str | None = None, modality: DataModality | None = None,
         *, progress: t.Callable[[dict], None] | None = None,
+        source_mode: t.Literal["chunk", "window", "document", "related_chunks"] = "chunk",
+        max_source_tokens: int | None = None, max_source_chunks: int = 3,
+        prompt_kind: t.Literal["qa", "code", "numeric", "constraints"] = "qa",
+        require_multiple_sources: bool = False, max_questions_per_source: int = 1,
     ) -> list[DatasetQAExample]:
-        """
-        One question per sampled chunk (num_samples chunks of the given modality, or of every modality
-        when None; the trajectory prompt variant for trajectory chunks), minus the chunks the model
-        declares junk. Cached rows (caching_id) are served first, the engine generates the tail with
-        seed `base_seed + batch index`; `study_context` is added to the prompt. Nothing is written into
-        the dataset: the examples are returned (origin SYNTHETIC, positive chunk = the source chunk).
-        """
-        loaded_model = self.harness.loaded_models[self.qa_model_name]
+        """Generate up to N questions per source attempt; retain exact evidence and cache whole requests."""
+        if source_mode not in ("chunk", "window", "document", "related_chunks"):
+            raise ValueError(f"Unknown source_mode: {source_mode}")
+        if max_source_chunks < 1 or max_questions_per_source < 1 or (max_source_tokens is not None and max_source_tokens < 1):
+            raise ValueError("Study source and question budgets must be positive")
+        model = self.harness.loaded_models[self.qa_model_name]
         stats = self.loaded_dataset.stats
-        kind = "traj_qa" if modality == DataModality.TRAJECTORY else "qa"
-        study_samples = self._sample_study_chunks(num_samples, base_seed, modality)
-        key = None
-        cached_rows: list[dict] = []
-        if caching_id is not None:
-            key = StudyCacheKey(caching_id, self.dataset_id, loaded_model.model_config.model_id, kind, base_seed)
-            for index, row in enumerate(self.cache.read(key)):
-                if index >= len(study_samples) or row.get("chunk_id") != study_samples[index].chunk_id:
-                    break                                                          # the sampler changed under the cache: stop trusting it
-                cached_rows.append(row)
-        examples: list[DatasetQAExample] = []
-        for row in cached_rows:
-            if row.get("junk"):
-                stats.study_num_junk_skipped += 1
-            else:
-                examples.append(self._example_from_row(row, kind, base_seed))
-        stats.study_num_cached += len(cached_rows)
-        pending = list(enumerate(study_samples))[len(cached_rows):]
-        def report(completed: int, phase: str) -> None:
-            if progress is not None:
-                progress({"phase": phase, "completed": completed, "total": len(study_samples),
-                          "accepted": len(examples), "cached": len(cached_rows),
-                          "rejected": stats.study_num_junk_skipped, "failed": stats.study_num_parse_failures})
-        report(len(cached_rows), "loading QA model" if pending else "QA complete")
-        if not pending:
-            return examples
-        # vllm batches a whole conversation list inside one chat() call, so a
-        # simple single-threaded loop needs no locks.
-        loaded_model.ensure_engine_loaded() # Keep the cold load out of the batch timings.
-        system_prompt, instructions, json_schema = make_study_prompt(study_context, modality or DataModality.TEXT)
-        total_generation_time = 0
-        reporting_interval = 0
-        for batch_index, batch_start in enumerate(range(0, len(pending), self.qa_batch_size)):
-            batch = pending[batch_start:batch_start + self.qa_batch_size]
-            chat_kwargs = self._make_engine_chat_kwargs(json_schema, base_seed + batch_index)
-            conversations = []
-            for _, chunk in batch:
-                snippet = self.dataset_index.get_chunk_section(chunk)
-                snippet = safe_truncate_embedding_chunk(snippet, self.chunk_input_limit)
-                conversations.append([
-                    {
-                        "role": "system", "content": [
-                            {"type": "text", "text": system_prompt},
-                        ]
-                    },
-                    {
-                        "role": "user", "content": [
-                            {"type": "text", "text": f"# Corpus Snippet\n```\n{snippet}\n```\n{instructions}"},
-                        ]
-                    },
-                ])
-            batch_start_time = time.time()
-            report(len(cached_rows) + batch_start, "generating QA")
-            outputs = loaded_model.engine_chat_many(conversations, chat_kwargs=chat_kwargs)
-            elapsed = time.time() - batch_start_time
-            total_generation_time += elapsed
-            stats.study_batch_latencies.append(elapsed)
+        system, instructions, schema = make_source_study_prompt(study_context, modality or DataModality.TEXT,
+            prompt_kind, max_questions_per_source, require_multiple_sources)
+        recipe = hashlib.sha256(json.dumps([system, instructions, schema, source_mode, max_source_tokens,
+            max_source_chunks, self.chat_kwargs, self.chunk_input_limit], sort_keys=True, default=str).encode()).hexdigest()[:16]
+        key = StudyCacheKey(caching_id, self.dataset_id, model.model_config.model_id, "qa_" + recipe, base_seed) if caching_id else None
+        cached = self.cache.read_by_id(key, "request_id") if key else {}
+        samples = self._sample_study_chunks(num_samples, base_seed, modality)
+        requests = []
+        skipped = 0
+        for index, chunk in enumerate(samples):
+            source = self._study_source(chunk, source_mode, max_source_chunks, max_source_tokens, model.tokenizer)
+            if source is None or (require_multiple_sources and len(source["units"]) < 2):
+                skipped += 1
+                continue
+            identity = hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()[:16]
+            requests.append({"request_id": f"{recipe}:{base_seed}:{index}:{identity}", "source": source})
+        rows = {r["request_id"]: cached[r["request_id"]] for r in requests if r["request_id"] in cached}
+        pending = [r for r in requests if r["request_id"] not in rows]
+        def report(phase):
+            if progress:
+                progress({"phase": phase, "completed": len(rows) + skipped, "total": len(samples),
+                          "accepted": sum(len(r["questions"]) for r in rows.values()), "cached": len(requests) - len(pending),
+                          "skipped_sources": skipped, "failed": stats.study_num_parse_failures})
+        report("loading QA model" if pending else "QA complete")
+        if pending:
+            model.ensure_engine_loaded()
+        for start in range(0, len(pending), self.qa_batch_size):
+            batch = pending[start:start + self.qa_batch_size]
+            conversations = [[{"role": "system", "content": system},
+                              {"role": "user", "content": r["source"]["text"] + "\n\n" + instructions}] for r in batch]
+            report("generating QA")
+            began = time.monotonic()
+            outputs = model.engine_chat_many(conversations, chat_kwargs=self._make_engine_chat_kwargs(schema, base_seed + start))
+            stats.study_batch_latencies.append(time.monotonic() - began)
+            if len(outputs) != len(batch):
+                raise RuntimeError("QA engine returned the wrong batch size")
             new_rows = []
-            for (index, chunk), output in zip(batch, outputs):
+            for request, output in zip(batch, outputs):
                 stats.study_prompt_tokens.append(output.prompt_token_count)
                 stats.study_output_tokens.append(output.output_token_count)
+                questions = []
                 try:
-                    qa_pair = json.loads(output.text)
-                    valuable = bool(qa_pair["has_valuable_question"])
-                    question = str(qa_pair.get("question") or "").strip()
-                    answer = str(qa_pair.get("answer") or "").strip()
-                except (json.JSONDecodeError, KeyError, TypeError):
-                    # Structured outputs make this rare (e.g. max_tokens cutoff). Not cached: a later pass retries.
+                    result = json.loads(output.text)
+                    seen = set()
+                    for q in result["questions"][:max_questions_per_source]:
+                        question, answer = q["question"].strip(), q["answer"].strip()
+                        try:
+                            # JSON permits lone surrogate escapes; cached/model text must be valid UTF-8.
+                            question.encode("utf-8")
+                            answer.encode("utf-8")
+                        except UnicodeEncodeError:
+                            stats.study_num_parse_failures += 1
+                            continue
+                        refs = sorted(set(q["evidence_indices"]))
+                        if not question or not answer or question.casefold() in seen:
+                            continue
+                        if not refs or any(type(i) is not int or not 0 <= i < len(request["source"]["units"]) for i in refs):
+                            continue
+                        if require_multiple_sources and len(refs) < 2:
+                            continue
+                        seen.add(question.casefold())
+                        questions.append({"question": question, "answer": answer, "evidence_indices": refs})
+                except (ValueError, KeyError, TypeError, AttributeError):
                     stats.study_num_parse_failures += 1
-                    continue
-                if not valuable or not question or not answer:
-                    stats.study_num_junk_skipped += 1
-                    new_rows.append({"index": index, "chunk_id": chunk.chunk_id, "junk": True})
-                    continue
-                row = {"index": index, "chunk_id": chunk.chunk_id, "question": question, "answer": answer}
+                # Cache empty/failed attempts too: stable source positions across restart.
+                row = dict(request, questions=questions)
+                rows[request["request_id"]] = row
                 new_rows.append(row)
-                examples.append(self._example_from_row(row, kind, base_seed))
-            if key is not None:
+                stats.study_num_junk_skipped += int(not questions)
+            if key:
                 self.cache.append(key, new_rows)
-            report(len(cached_rows) + batch_start + len(batch), "generating QA")
-            if reporting_interval <= 0:
-                print(
-                    f"{self.dataset_id} - Study ({kind}): {batch_start + len(batch)}/{len(pending)} chunks, "
-                    f"{len(examples)} questions, {stats.study_num_junk_skipped} junk. {total_generation_time:.2f}s."
-                )
-                reporting_interval = max(1, len(pending) // 20)
-            reporting_interval -= len(batch)
-        report(len(study_samples), "QA complete")
+            report("generating QA")
+        examples = []
+        for request in requests:
+            row = rows[request["request_id"]]
+            for child, qa in enumerate(row["questions"]):
+                supports = [row["source"]["units"][i] for i in qa["evidence_indices"]]
+                examples.append(DatasetQAExample(
+                    example_id=f"{self.dataset_id}:study:{row['request_id']}:{child}", dataset_id=self.dataset_id,
+                    query=qa["question"], gold_answers=[qa["answer"]], origin=SYNTHETIC, split=DataSplit.TRAIN,
+                    positive_doc_ids=list(dict.fromkeys(u["doc_id"] for u in supports)),
+                    positive_chunk_ids=list(dict.fromkeys(c for u in supports for c in u["chunk_ids"])),
+                    study_source=dict(row["source"], request_id=row["request_id"])))
+        stats.study_num_cached += len(requests) - len(pending)
+        report("QA complete")
         return examples
 
 
````

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title"><a href="trajectory_implementation/activation/dataset/dataset_study_prompts.py">activation/dataset/dataset_study_prompts.py</a></span>
    <span class="card-oneliner">One synthesis prompt and response schema; preserve complete study context and the unchanged labeling prompt.</span>
  </summary>

````diff-python
diff --git a/activation/dataset/dataset_study_prompts.py b/activation/dataset/dataset_study_prompts.py
new file mode 100644
index 0000000..8f63595
--- /dev/null
+++ b/activation/dataset/dataset_study_prompts.py
@@ -0,0 +1,101 @@
+"""Small study prompt presets; generation and caching stay in dataset_study."""
+from .dataset import DataModality
+
+LABEL_JSON_SCHEMA = {
+    "type": "object",
+    "properties": {
+        "positives": {"type": "array", "items": {"type": "integer"}},
+        "negatives": {"type": "array", "items": {"type": "integer"}},
+    },
+    "required": ["positives", "negatives"],
+    "additionalProperties": False,
+}
+
+
+
+def make_source_study_prompt(study_context: str | None, modality: DataModality = DataModality.TEXT,
+                             prompt_kind="qa", max_questions=1, require_multiple_sources=False) -> tuple[str, str, dict]:
+    """One response contract for single- and multiple-source study questions."""
+    presets = {
+        "qa": "Ask a concrete question whose answer depends on the supplied evidence.",
+        "code": "Ask about code behavior, a concrete input/output, an API condition, or a dependency. Do not invent missing implementation.",
+        "numeric": "Ask for a short calculation or quantitative conclusion grounded in the supplied numbers; include units in the answer when applicable.",
+        "constraints": "Ask for a decision that applies explicit rules or constraints in the evidence; make all needed conditions available.",
+    }
+    if prompt_kind not in presets:
+        raise ValueError(f"Unknown prompt_kind: {prompt_kind}")
+    system_prompt = """
+# Core Guidelines
+- You are a study/exam-like questions formulator.
+- You are given a snippet from the corpus students are expected to study and understand.
+- From this, your goal is to generate a hard question (unanswerable with common knowledge or simple keyword look ups).
+    - This is IMPORTANT: the question should be **hard**, or it won't test the student's learning.
+    - The question should have an accompanying answer.
+- Your priority is: a hard question along with its expected, correct answer.
+- By default, keep your question short to medium-short unless the exam recommends longer questions.
+- When the snippet holds nothing worth asking about (boilerplate, navigation, an empty listing, pure formatting), return an empty questions array.
+"""
+    if modality == DataModality.TRAJECTORY:
+        system_prompt += """
+# Trajectory Snippets
+- The snippet is part of an agent's conversation: its reasoning, its tool calls and the tool results.
+- Ask about a specific fact that appears in these turns (a value a tool returned, a decision the agent took, an error it saw, a file or quantity it found), phrased so someone who read the turns can answer it.
+- Never ask something answerable from the task statement alone, and never ask about the formatting of the transcript.
+"""
+
+    if study_context:
+        system_prompt += f"""
+# Exam Context
+The following emphasizes the kind of questions students are expected to handle.
+```
+-----
+{study_context}
+-----
+```
+Frame your questions in light of this.
+"""
+    system_prompt += "\n\n" + presets[prompt_kind] + f"\nReturn 0 to {max_questions} distinct, self-contained questions with concise correct answers. Return none for unusable evidence. Do not mention 'the snippet' or its numbering in questions."
+    if require_multiple_sources:
+        system_prompt += " Every question must require combining at least two numbered source units, and cite those units in evidence_indices. Skip if this cannot be done naturally."
+    else:
+        system_prompt += " Cite the source units supporting each answer in evidence_indices."
+    entry = {"type": "object", "properties": {"question": {"type": "string"}, "answer": {"type": "string"},
+             "evidence_indices": {"type": "array", "items": {"type": "integer"}}},
+             "required": ["question", "answer", "evidence_indices"], "additionalProperties": False}
+    schema = {"type": "object", "properties": {"questions": {"type": "array", "items": entry, "maxItems": max_questions}},
+              "required": ["questions"], "additionalProperties": False}
+    return system_prompt, "Return JSON with a questions array; evidence_indices use the zero-based Source numbers.", schema
+
+
+def make_label_prompt() -> tuple[str, str, dict]:
+    """
+    Returns system prompt, instructions and json schema for the snippet labeling task.
+    """
+    system_prompt = """
+# Core Guidelines
+- You are a relevance judge determining what snippets from a corpus students should pay attention to answer a question.
+- You are given the study question, its reference answer, and numbered corpus snippets.
+- Label each snippet index as:
+    - **positive**: the snippet contains information that clearly helps answer the question.
+    - **negative**: the snippet contains information that clearly does not help answer the question.
+    - Leave out anything where you are unsure.
+- Be careful:
+    - The snippets are intentionally chosen to be *related* to the question without necessarily helping.
+    - Don't mark something as positive just because of shared vocabulary.
+- Every listed index must come from the snippet numbering; never list the same index on both sides.
+"""
+    system_prompt += """
+# Response Format
+Your answer must be formatted as a json object of snippet indices like:
+```json
+{
+    "positives": [0, 2],
+    "negatives": [3, 5, 6]
+}
+```
+Each is a list of indexes that should map to one of the given snippet indices.
+"""
+    instructions = """
+Label the snippets as positives / negatives for this question. Leave out ambiguous ones.
+"""
+    return system_prompt, instructions, LABEL_JSON_SCHEMA
````

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title"><a href="trajectory_implementation/activation/agent/agent.py">activation/agent/agent.py</a></span>
    <span class="card-oneliner">Preserve the total turn budget when completed turns move into a compaction segment.</span>
  </summary>

````diff-python
diff --git a/activation/agent/agent.py b/activation/agent/agent.py
index 108cd7d..1d5e0bf 100644
--- a/activation/agent/agent.py
+++ b/activation/agent/agent.py
@@ -52,7 +52,8 @@ if t.TYPE_CHECKING:
 
 
 class Agent:
-    COMPACTION_DEMAND = "The context threshold is reached: call compact(summary) now, alone, before any other tool."
+    COMPACTION_DEMAND = ("The context threshold is reached: call compact(summary) now, alone, before any other tool. "
+                         "Use the five-section handoff format in its description.")
 
     def __init__(
         self,
@@ -290,6 +291,7 @@ class Agent:
         )
         results.compactions.append(record)
         results.trajectory = []
+        results.num_turns = 0
         text = COMPACTION_INSTRUCTIONS + (f"\n\nSummary written before compaction:\n{summary}" if summary.strip() else "")
         if tree:
             after = {"role": "user", "content": tree + [{"type": "text", "text": "\n\n" + text}]}
````

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title"><a href="trajectory_implementation/activation/dataset/dataset_caching.py">activation/dataset/dataset_caching.py</a></span>
    <span class="card-oneliner">Bound long filesystem cache keys while preserving existing valid filenames.</span>
  </summary>

````diff-python
diff --git a/activation/dataset/dataset_caching.py b/activation/dataset/dataset_caching.py
index 9d65462..8c7929b 100644
--- a/activation/dataset/dataset_caching.py
+++ b/activation/dataset/dataset_caching.py
@@ -10,6 +10,7 @@ it). A request beyond the cached rows reuses them and generates the tail, which
 are keyed by example id. A stable caching id from the caller is trusted; nothing else is checked.
 """
 import json
+import hashlib
 import re
 from dataclasses import dataclass
 from pathlib import Path
@@ -30,7 +31,13 @@ class StudyCacheKey:
         parts = [self.caching_id, self.dataset_id, self.kind, self.model_id, f"seed{self.seed}"]
         if self.label_model_id:
             parts.append(self.label_model_id)
-        return "__".join(re.sub(r"[^0-9A-Za-z_.-]+", "_", part) for part in parts) + ".jsonl"
+        stem = "__".join(re.sub(r"[^0-9A-Za-z_.-]+", "_", part) for part in parts)
+        if len(stem) + len(".jsonl") > 255:
+            # Local model snapshots can make a cache key exceed filesystem limits.
+            # Keep valid existing names; hash the complete original key when shortening.
+            suffix = hashlib.sha256(json.dumps(parts, ensure_ascii=False).encode("utf-8")).hexdigest()
+            stem = stem[:180] + "__" + suffix
+        return stem + ".jsonl"
 
 
 class StudyCache:
````

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title"><a href="trajectory_implementation/activation/common/ac_parts.py">activation/common/ac_parts.py</a></span>
    <span class="card-oneliner">Share a five-section handoff prompt requesting fewer than 500 words and a generic resume instruction across oracle generation and live compaction.</span>
  </summary>

````diff-python
diff --git a/activation/common/ac_parts.py b/activation/common/ac_parts.py
index e80745d..2302e64 100644
--- a/activation/common/ac_parts.py
+++ b/activation/common/ac_parts.py
@@ -13,9 +13,30 @@ if t.TYPE_CHECKING:
 
 AC_PART_TYPE = "activation_context"
 PART_SENTINEL = "⁣ACPART{index}⁣"     # rendered in place of a part by the chat template, split out before tokenizing
+COMPACTION_SUMMARY_PROMPT = """Write a concise handoff so the task can resume without rereading the conversation. Keep the entire handoff under 500 words, prioritizing details needed to resume. Preserve exact details and uncertainty. Keep guesses separate from observed results; searches are not confirmation. Preserve conflicting evidence. Use these five sections:
+
+# Summary Overview
+The current goal and overall situation.
+
+# Progress So Far
+[v] Completed work and results.
+[-] Work in progress or blocked.
+[ ] Work remaining.
+
+# Notes/Insights/Lessons To Preserve
+Key findings, decisions, and concrete details needed later.
+
+# Immediate Continuation
+The next action, including any blocker that must be resolved first.
+
+# DOs/DON'Ts
+Quote still-applicable constraints from the original task's system/user messages, preserving their scope. If absent, write "None".
+
+Update older notes to reflect the latest state. Avoid repetition; distinguish completed work from plans and attempts."""
+
 COMPACTION_INSTRUCTIONS = (
-    "The conversation so far has been compacted into the activation context above. Continue the task "
-    "from exactly where it left off, as if the full conversation were still in front of you."
+    "Continue the task from the compacted state above. Resume the next unfinished action; "
+    "do not treat plans or pending work as completed."
 )
 
 
````

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title"><a href="trajectory_implementation/activation/agent/agent_tools.py">activation/agent/agent_tools.py</a></span>
    <span class="card-oneliner">Use the shared handoff format in the existing compaction tool; preserve uncertainty and original instruction scope.</span>
  </summary>

````diff-python
diff --git a/activation/agent/agent_tools.py b/activation/agent/agent_tools.py
index 9df06be..19f5dc2 100644
--- a/activation/agent/agent_tools.py
+++ b/activation/agent/agent_tools.py
@@ -18,7 +18,7 @@ from concurrent.futures import ThreadPoolExecutor
 from copy import deepcopy
 from dataclasses import dataclass, field, replace
 
-from activation.common.ac_parts import COMPACTION_INSTRUCTIONS, activation_part
+from activation.common.ac_parts import COMPACTION_INSTRUCTIONS, COMPACTION_SUMMARY_PROMPT, activation_part
 
 if t.TYPE_CHECKING:
     from .agent_config import AgentConfig, AgentRunResult
@@ -451,11 +451,11 @@ class CompactionTool(AgentTool):
     threshold is reached (Agent.COMPACTION_DEMAND) and refuses every other call until then.
     """
     name = "compact"
-    description = ("Compact the conversation so far: write a summary of what was done, what was learned and what remains, "
-                   "then continue from it. Call it alone, without other tools, when asked to compact.")
+    description = ("Compact the conversation so far. Call it alone, without other tools, when asked to compact.\n\n"
+                   + COMPACTION_SUMMARY_PROMPT)
     parameters = {
         "type": "object",
-        "properties": {"summary": {"type": "string", "description": "A complete summary of the work so far for your future self."}},
+        "properties": {"summary": {"type": "string", "description": "A handoff using the five sections above."}},
         "required": ["summary"],
     }
 
````

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title"><a href="trajectory_implementation/activation/harness/hf_utils.py">activation/harness/hf_utils.py</a></span>
    <span class="card-oneliner">Avoid redundant PEFT routing and intervention-index transfers while preserving checkpoint recomputation and live adapter state.</span>
  </summary>

````diff-python
diff --git a/activation/harness/hf_utils.py b/activation/harness/hf_utils.py
index 3171f76..2e5e0cc 100644
--- a/activation/harness/hf_utils.py
+++ b/activation/harness/hf_utils.py
@@ -3,7 +3,7 @@ HF-facing code for model configuration parsing, message formatting, etc.
 Stores overly detailed/specific logic to keep the rest of code cleaner.
 """
 from collections import Counter
-from contextlib import contextmanager
+from contextlib import contextmanager, nullcontext
 import typing as t
 import torch
 import torch.nn.functional as F
@@ -46,7 +46,7 @@ _FFN_ONLY_NAMES = frozenset({"mlp", "moe"})
 
 @contextmanager
 def checkpoint_adapter_scope(model, adapter_context, enabled: bool | None = None):
-    """Bind each layer's recomputation to the adapter used for its original forward."""
+    """Bind recomputation to the adapter already scoped around the original decoder forward."""
     previous = [(module, module.gradient_checkpointing, getattr(module, "_gradient_checkpointing_func", None))
                 for module in model.modules() if hasattr(module, "gradient_checkpointing")]
     was_enabled = model.is_gradient_checkpointing
@@ -62,7 +62,7 @@ def checkpoint_adapter_scope(model, adapter_context, enabled: bool | None = None
             original = module._gradient_checkpointing_func
             def scoped(function, *args, _checkpoint=original, **kwargs):
                 return _checkpoint(function, *args, **(kwargs | {
-                    "use_reentrant": False, "context_fn": lambda: (adapter_context(), adapter_context())}))
+                    "use_reentrant": False, "context_fn": lambda: (nullcontext(), adapter_context())}))
             module._gradient_checkpointing_func = scoped
         yield
     finally:
@@ -74,6 +74,49 @@ def checkpoint_adapter_scope(model, adapter_context, enabled: bool | None = None
                 del module._gradient_checkpointing_func
 
 
+@contextmanager
+def peft_adapter_scope(model, adapter_name: str | None):
+    """Route an existing PEFT model without resetting an already-correct ordinary LoRA.
+
+    Trainability belongs to the optimizer: snapshot it live on every entry and restore
+    it even when routing was unchanged. No parameter/layer references survive this scope.
+    """
+    from peft.tuners.tuners_utils import BaseTunerLayer
+    from peft.utils.other import AuxiliaryTrainingWrapper
+
+    previous_adapter = model.active_adapter
+    trainability = [(parameter, parameter.requires_grad) for parameter in model.parameters()]
+    modules = list(model.modules())
+    layers = [(module, module.disable_adapters) for module in modules if isinstance(module, BaseTunerLayer)]
+    ordinary = not any(isinstance(module, AuxiliaryTrainingWrapper) for module in modules)
+    ordinary = ordinary and not any(layer.merged for layer, _ in layers)
+    adapters = tuple(model.peft_config)
+
+    def route(name):
+        names = [name] if isinstance(name, str) else list(name)
+        if (not ordinary or tuple(model.peft_config) != adapters or model.active_adapter != name
+                or model.base_model.active_adapter != name
+                or any(layer.active_adapters != names or layer.merged for layer, _ in layers)):
+            model.set_adapter(name)
+
+    try:
+        if adapter_name is not None:
+            route(adapter_name)
+        for layer, _ in layers:
+            if not ordinary or layer.disable_adapters != (adapter_name is None):
+                layer.enable_adapters(adapter_name is not None)
+        for parameter, trainable in trainability:
+            parameter.requires_grad_(trainable)
+        yield
+    finally:
+        route(previous_adapter)
+        for layer, disabled in layers:
+            if not ordinary or layer.disable_adapters != disabled:
+                layer.enable_adapters(not disabled)
+        for parameter, trainable in trainability:
+            parameter.requires_grad_(trainable)
+
+
 
 # --------------------------------------------------------------------------------------- deep AC inputs (interventions)
 INTERVENTIONS_KWARG = "interventions"
@@ -92,11 +135,25 @@ class BoundInterventions:
     """The payloads of one decoder forward bound to that forward's cache offset (set once by the text-model pre-hook);
     travels in the layer kwargs, so a checkpointed layer recomputes with exactly the same tensors."""
 
-    __slots__ = ("payloads", "offset")
+    __slots__ = ("payloads", "offset", "_indices")
 
     def __init__(self, payloads: list[Interventions | None], offset: int) -> None:
         self.payloads = payloads
         self.offset = offset
+        self._indices = {}
+
+    def indices(self, row: int, length: int, device: torch.device):
+        """Position metadata is fixed for this forward, including its checkpoint recomputation."""
+        source_positions = tuple(self.payloads[row]["positions"])
+        key = (source_positions, self.offset, length, device)
+        if key not in self._indices:
+            local = [position - self.offset for position in source_positions]
+            selected = [index for index, position in enumerate(local) if 0 <= position < length]
+            positions = torch.tensor([local[index] for index in selected], device=device, dtype=torch.long)
+            rows = (None if len(selected) == len(local)
+                    else torch.tensor(selected, device=device, dtype=torch.long))
+            self._indices[key] = positions, rows
+        return self._indices[key]
 
 
 def text_decoder(model) -> torch.nn.Module:
@@ -163,18 +220,17 @@ def install_intervention_hooks(model) -> None:
                 rows = payload["layer_inputs"].get(index)
                 if rows is None:
                     continue
-                positions = torch.as_tensor(payload["positions"], device=hidden.device, dtype=torch.long)
-                if positions.numel() != rows.shape[0]:
+                if len(payload["positions"]) != rows.shape[0]:
                     raise ValueError("positions and delta rows differ in count")
-                local = positions - bound.offset
-                keep = (local >= 0) & (local < length)
-                if not bool(keep.any()):
+                local, selected = bound.indices(row, length, hidden.device)
+                if local.numel() == 0:
                     continue
                 # Diagnostic for the producer: the residual RMS at this forward's row positions (the deltas' true denominator).
-                payload.setdefault("row_rms", {})[index] = hidden[row, local[keep]].detach().float().pow(2).mean(dim=-1).sqrt().mean()   # 0-d tensor: no host sync here
+                payload.setdefault("row_rms", {})[index] = hidden[row, local].detach().float().pow(2).mean(dim=-1).sqrt().mean()   # 0-d tensor: no host sync here
                 if output is None:
                     output = hidden.clone()
-                output[row].index_add_(0, local[keep], rows[keep].to(device=hidden.device, dtype=hidden.dtype))
+                rows = rows if selected is None else rows.index_select(0, selected)
+                output[row].index_add_(0, local, rows.to(device=hidden.device, dtype=hidden.dtype))
             return ((output if output is not None else hidden, *args[1:]), kwargs)
         return hook
     for index, layer in enumerate(decoder_layers(model)):
````

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title"><a href="trajectory_implementation/activation/harness/module_manager.py">activation/harness/module_manager.py</a></span>
    <span class="card-oneliner">Delegate adapter scoping to the existing harness utility with unchanged public behavior.</span>
  </summary>

````diff-python
diff --git a/activation/harness/module_manager.py b/activation/harness/module_manager.py
index 463befc..7b51063 100644
--- a/activation/harness/module_manager.py
+++ b/activation/harness/module_manager.py
@@ -28,7 +28,7 @@ from torch import nn
 
 from activation.common.data_syncing import resolve_path
 
-from .hf_utils import TARGET_DEVICE, make_peft_lora_config
+from .hf_utils import TARGET_DEVICE, make_peft_lora_config, peft_adapter_scope
 
 if t.TYPE_CHECKING:
     from peft import PeftModel
@@ -201,25 +201,8 @@ class ModuleManager:
         if peft_model is None:
             yield
             return
-        previous_adapter = peft_model.active_adapter
-        trainability = [(parameter, parameter.requires_grad) for parameter in peft_model.parameters()]
-        from peft.tuners.tuners_utils import BaseTunerLayer
-        layers = [(module, module.disable_adapters) for module in peft_model.modules() if isinstance(module, BaseTunerLayer)]
-        try:
-            if lora_name is not None:
-                peft_model.set_adapter(lora_config.adapter_name)
-            for layer, _ in layers:
-                layer.enable_adapters(lora_name is not None)
-            # PEFT routing changes trainability. Optimizer ownership, not the active adapter, controls it here.
-            for parameter, trainable in trainability:
-                parameter.requires_grad_(trainable)
+        with peft_adapter_scope(peft_model, None if lora_name is None else lora_config.adapter_name):
             yield
-        finally:
-            peft_model.set_adapter(previous_adapter)
-            for layer, disabled in layers:
-                layer.enable_adapters(not disabled)
-            for parameter, trainable in trainability:
-                parameter.requires_grad_(trainable)
 
     def lora_parameters(self, lora_name: str) -> list[nn.Parameter]:
         """The trainable parameters of that adapter only (injecting it if needed)."""
````

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title"><a href="trajectory_implementation/activation/autotuners/__init__.py">activation/autotuners/__init__.py</a></span>
    <span class="card-oneliner">Expose optional workload-specific AC execution tuning through the existing autotuner entry points.</span>
  </summary>

````diff-python
diff --git a/activation/autotuners/__init__.py b/activation/autotuners/__init__.py
index e544458..9f78ced 100644
--- a/activation/autotuners/__init__.py
+++ b/activation/autotuners/__init__.py
@@ -62,3 +62,16 @@ def configure_fla_runtime(bundle: dict | None):
     """Apply the returned catalog for a trainer's model-call scope; restore on exit."""
     from .fla import configure_fla_runtime as configure
     return configure(bundle)
+
+
+def tune_ac_execution(harness, ac_name: str, prepared_examples, training_config, *, seconds: float = 600,
+                      progress_callback=None) -> dict:
+    """Measure an SFT execution policy in a disposable GPU worker before model placement.
+
+    Returns ``policy`` and its measured ``receipt``. Does not change the caller's
+    config, examples, optimizer or weights; applying the policy is explicit.
+    The optional callback receives phase/case progress in the parent process.
+    """
+    from .core import tune_ac_execution as tune
+    return tune(harness, ac_name, prepared_examples, training_config, seconds=seconds,
+                progress_callback=progress_callback)
````

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title"><a href="trajectory_implementation/activation/autotuners/core.py">activation/autotuners/core.py</a></span>
    <span class="card-oneliner">Run isolated real-workload candidates with exact source/hardware identity, progress and persistent validated receipts.</span>
  </summary>

````diff-python
diff --git a/activation/autotuners/core.py b/activation/autotuners/core.py
index 91a391b..d2be7df 100644
--- a/activation/autotuners/core.py
+++ b/activation/autotuners/core.py
@@ -209,3 +209,322 @@ def resolve(model_configs, training_config, device, memory_budget_bytes, *, retu
                   "memory_budget_bytes": available}
     return {"fla_config": bundle, "logits_chunk_tokens": chunk if chunk is not None else automatic_chunk,
             "provenance": provenance}, threshold if threshold is not None else 0
+
+
+# Whole-model measurements are deliberately separate from the FLA kernel catalog.
+EXECUTION_SCHEMA = 2
+
+
+def execution_batches(records: list[dict], config) -> list[list[int]]:
+    """Two real update shapes and a distinct-history padding tail, never a data filter."""
+    import random
+    count = config.examples_per_update
+    if not count or count < 1:
+        raise ValueError("Execution tuning requires explicit examples_per_update")
+    order = list(range(len(records)))
+    if config.shuffle_examples:
+        random.Random(config.seed + 1).shuffle(order)
+    groups = [order[i:i + count] for i in range(0, len(order), count)]
+    groups.sort(key=lambda group: sum(records[i]["encoder_tokens"] for i in group))
+    selected = [groups[(len(groups) - 1) // 2], groups[int((len(groups) - 1) * .9)]]
+    tail, seen = [], set()
+    for index in sorted(order, key=lambda i: (records[i]["encoder_tokens"], records[i]["reader_tokens"]), reverse=True):
+        key = tuple(records[index]["request_keys"])
+        if key not in seen:
+            tail.append(index); seen.add(key)
+        if len(tail) == count:
+            break
+    if len(tail) < min(count, len(order)):
+        tail.extend(i for i in reversed(order) if i not in tail)
+        tail = tail[:count]
+    selected.append(tail)
+    return [list(group) for group in dict.fromkeys(tuple(group) for group in selected)]
+
+
+def execution_candidates(config) -> list[dict]:
+    """A small ordered ladder, not a Cartesian search. Every arm needs its own gate."""
+    baseline = dict(graph_examples=1, reader_batch=1, encode_batch_tokens=1,
+                    reuse_encodings=False, side_checkpoint_tokens=0,
+                    target_checkpoint_tokens=0, optimizer_moments="cpu")
+    # Reader batching and graph-local reuse leave encoder batch shape unchanged.
+    variants = [baseline, {**baseline, "optimizer_moments": "gpu"}]
+    variants.append({**baseline, "target_checkpoint_tokens": 4096})
+    variants.append({**variants[-1], "side_checkpoint_tokens": 8192})
+    for threshold in (16384, 32768):
+        variants.append({**variants[2], "side_checkpoint_tokens": threshold})
+    variants.append({**baseline, "graph_examples": 4, "reuse_encodings": True})
+    for batch in (2, 4):
+        variants.append({**baseline, "graph_examples": batch, "reader_batch": batch,
+                         "reuse_encodings": True})
+    variants.append({**variants[-1], "target_checkpoint_tokens": 4096})
+    variants.append({**variants[-1], "side_checkpoint_tokens": 8192})
+    variants.append({**variants[-1], "optimizer_moments": "gpu"})
+    # Encoder batching is deliberately last: prior shape changes failed parity.
+    variants.append({**variants[-1], "encode_batch_tokens": 16384})
+    count = config.examples_per_update
+    return [policy for policy in variants if policy["graph_examples"] <= count]
+
+
+def execution_capacity_cases(records: list[dict], config, policy: dict) -> list[dict]:
+    """Actual examples just below checkpoint boundaries; separate from speed sampling."""
+    update_sizes = [min(config.examples_per_update, len(records))]
+    if len(records) > config.examples_per_update and len(records) % config.examples_per_update:
+        update_sizes.append(len(records) % config.examples_per_update)
+    widths = list(dict.fromkeys(min(policy["graph_examples"], count - start)
+                  for count in update_sizes for start in range(0, count, policy["graph_examples"])))
+    cases = {}
+
+    def largest(scores, count):
+        # Avoid a repeated-history best case when graph-local reuse is enabled.
+        ranked = sorted(scores, key=lambda i: (scores[i], records[i]["encoder_tokens"],
+                                              records[i]["reader_tokens"], records[i]["penalty_tokens"],
+                                              str(records[i]["item"])), reverse=True)
+        result, seen = [], set()
+        for index in ranked:
+            key = tuple(records[index]["request_keys"])
+            if key not in seen:
+                result.append(index); seen.add(key)
+            if len(result) == count:
+                return result
+        return (result + [i for i in ranked if i not in result])[:count]
+
+    def add(indices, reason, width, penalty_indices=None, offset=0):
+        if not indices:
+            return
+        if len(indices) < width:
+            others = {i: records[i]["encoder_tokens"] + records[i]["reader_tokens"]
+                      for i in range(len(records)) if i not in indices}
+            remaining = largest(others, width - len(indices))
+            indices = remaining[:offset] + indices + remaining[offset:]
+        flags = [bool(config.penalty_active and records[i]["penalty_tokens"] and
+                      (penalty_indices is None or i in penalty_indices)) for i in indices]
+        key = (tuple(indices), tuple(flags))
+        if key not in cases:
+            cases[key] = {"indices": indices, "penalty_flags": flags, "reasons": []}
+        cases[key]["reasons"].append(reason)
+
+    for width in widths:
+        side = policy["side_checkpoint_tokens"]
+        if side:
+            scores = {i: max((length for length in record["encoder_sequence_tokens"] if length < side), default=0)
+                      for i, record in enumerate(records)}
+            add(largest({i: score for i, score in scores.items() if score}, width), f"side below {side}", width)
+        target = policy["target_checkpoint_tokens"]
+        if not target:
+            continue
+        # A final short update/graph/reader chunk can enter an uncheckpointed
+        # branch even when every full batch of the same examples checkpoints.
+        reader_widths = {min(policy["reader_batch"], width): 0}
+        if width > policy["reader_batch"] and width % policy["reader_batch"]:
+            reader_widths[width % policy["reader_batch"]] = width - width % policy["reader_batch"]
+        for reader_width, offset in reader_widths.items():
+            scores = {i: record["reader_tokens"] for i, record in enumerate(records)
+                      if record["reader_tokens"] * reader_width < target}
+            if len(scores) >= reader_width:
+                # Put the short reader case in the actual tail, not a full chunk.
+                indices = largest(scores, reader_width if offset else width)
+                add(indices, f"reader below {target}", width, offset=offset)
+            if config.penalty_active:
+                # A subset receives penalties, so each possible padded penalty
+                # width has its own boundary inside this reader chunk.
+                for count in range(1, reader_width + 1):
+                    scores = {i: record["penalty_tokens"] for i, record in enumerate(records)
+                              if 0 < record["penalty_tokens"] * count < target}
+                    if len(scores) >= count:
+                        indices = largest(scores, count)
+                        add(indices, f"penalty width {count} below {target}", width, set(indices), offset)
+    return list(cases.values())
+
+
+def read_execution_profile(path: Path, identity: dict) -> dict | None:
+    try:
+        payload = json.loads(path.read_text())
+        profile = payload["profile"]
+        if (payload["schema"] != SCHEMA or payload["sha256"] != digest(profile)
+                or profile["identity"] != identity or profile["kind"] != "ac_execution"
+                or profile["execution_schema"] != EXECUTION_SCHEMA
+                or not profile["validation"]["passed"]):
+            return None
+        winner = next(row for row in profile["candidates"] if row["policy"] == profile["policy"])
+        if (not winner["passed"] or not winner["cases"] or not all(row["passed"] for row in winner["cases"])
+                or len(winner["cases"]) != len(identity["cases"])):
+            return None
+        required = next(row["cases"] for row in identity["capacity_policies"] if row["policy"] == winner["policy"])
+        if ([row["capacity_case"] for row in winner["capacity_cases"]] != required
+                or not all(row["passed"] for row in winner["capacity_cases"])):
+            return None
+        return profile
+    except (OSError, ValueError, KeyError, TypeError, StopIteration):
+        return None
+
+
+def _wait_execution_worker(process, progress_path: Path, deadline: float, progress_callback=None):
+    """Forward new atomic progress records without resetting the worker deadline."""
+    previous = None
+    def update():
+        nonlocal previous
+        if progress_callback is None:
+            return
+        try:
+            current = progress_path.read_text()
+            progress = json.loads(current)
+        except (OSError, ValueError):
+            return
+        if current != previous:
+            progress_callback(progress)
+            previous = current
+    while True:
+        update()
+        remaining = deadline - time.monotonic()
+        if remaining <= 0:
+            raise subprocess.TimeoutExpired("AC execution worker", 0)
+        try:
+            process.wait(timeout=min(5., remaining))
+        except subprocess.TimeoutExpired:
+            continue
+        update()
+        return
+
+
+def tune_ac_execution(harness, ac_name, prepared_examples, training_config, *, seconds=600, progress_callback=None):
+    """Trusted local objects only; the child owns all GPU models and trial state."""
+    from dataclasses import asdict, replace
+    import pickle
+    import torch
+    from ..ac_model.ac_model_utils import direct_parts, tokenize_with_parts
+    from ..common.data_syncing import sync_root
+    if training_config.loss_kind != "sft":
+        raise ValueError("Whole-execution tuning currently supports SFT only")
+    if seconds <= 0 or not prepared_examples:
+        raise ValueError("Execution tuning needs examples and a positive budget")
+    manager = harness.module_manager
+    ac = manager.get_ac_model(ac_name)
+    if manager.peft_models or any(model.current_model_device.startswith("cuda") for model in harness.loaded_models.values()):
+        raise ValueError("Tune execution before placing/injecting parent models; the child needs exclusive GPU capacity")
+    examples = list(prepared_examples)
+    counts, sequences = {}, {}
+    def request_record(request):
+        key = digest(request._asdict())
+        if key not in counts:
+            children = [ac._child_request(part, request.compression_ratio) for part in direct_parts(request.messages)]
+            child_counts = [request_record(child) for child in children]
+            child_rows = [ac.num_view_rows(counts[k][0], child.compression_ratio) for k, child in zip(child_counts, children)]
+            ids, _ = tokenize_with_parts(ac.side.tokenizer, request.messages, child_rows, ac.pad_id, tools=request.tools)
+            length = len(ids) + ac.num_view_rows(len(ids), request.compression_ratio)
+            counts[key] = (len(ids), length + sum(counts[k][1] for k in child_counts))
+            sequences[key] = [length, *(value for child in child_counts for value in sequences[child])]
+        return key
+    records = []
+    for example in examples:
+        keys = [request_record(request) for request in example.part_requests]
+        records.append({"item": example.item.item_id, "request_keys": keys,
+                        "encoder_tokens": sum(counts[key][1] for key in keys),
+                        "encoder_sequence_tokens": [value for key in keys for value in sequences[key]],
+                        "reader_tokens": len(example.student_ids), "target_tokens": len(example.target_ids),
+                        "penalty_tokens": len(example.penalty.student_ids) if example.penalty else 0,
+                        "weight": example.item.weight,
+                        "reader_identity": digest([list(example.student_ids), list(example.student_positions), list(example.target_ids)]),
+                        "reference_identity": list(example.penalty.teacher_cache_key) if example.penalty else None})
+    batches = execution_batches(records, training_config)
+    capacity_cases, capacity_policies = [], []
+    for policy in execution_candidates(training_config):
+        selected_cases = []
+        for case in execution_capacity_cases(records, training_config, policy):
+            found = next((i for i, prior in enumerate(capacity_cases)
+                          if prior["indices"] == case["indices"] and prior["penalty_flags"] == case["penalty_flags"]), None)
+            if found is None:
+                found = len(capacity_cases); capacity_cases.append(case)
+            else:
+                capacity_cases[found]["reasons"] = list(dict.fromkeys(capacity_cases[found]["reasons"] + case["reasons"]))
+            selected_cases.append(found)
+        capacity_policies.append({"policy": policy, "cases": selected_cases})
+    indexes = list(dict.fromkeys(i for group in [*batches, *(case["indices"] for case in capacity_cases)] for i in group))
+    local = {index: position for position, index in enumerate(indexes)}
+    selected = [examples[i] for i in indexes]
+    model_names = {ac.config.base_side_model_name, ac.config.target_model_name}
+    harness_config = replace(harness.harness_config, model_configs={name: harness.harness_config.model_configs[name] for name in model_names})
+    def absolute_checkpoint(value):
+        if value is None:
+            return None
+        path = Path(value)
+        return str(path.resolve() if path.is_absolute() else resolve_path(str(path), create=False).resolve())
+    ac_config = replace(ac.config, checkpoint_path=absolute_checkpoint(ac.config.checkpoint_path))
+    loras = [replace(manager.get_lora_config(name), checkpoint_path=absolute_checkpoint(manager.get_lora_config(name).checkpoint_path))
+             for name in dict.fromkeys((ac.config.base_side_model_lora_name, ac.config.target_model_lora_name))]
+    device = torch.device("cuda", torch.cuda.current_device())
+    env = environment(device)
+    # Training knobs that affect graph shape/precision/objective, not elapsed-run state.
+    recipe = asdict(training_config)
+    for key in ("execution_policy", "max_duration_seconds", "resume_optimizer", "resume_skip_steps", "num_epochs",
+                "checkpoint_every_updates", "checkpoint_every_epoch", "reporting_interval", "completion_samples"):
+        recipe.pop(key, None)
+    def portable(value):
+        return json.loads(json.dumps(value, default=str, sort_keys=True))
+    checkpoints = {}
+    for checkpoint in [ac_config.checkpoint_path, *(lora.checkpoint_path for lora in loras)]:
+        if checkpoint:
+            checkpoints[checkpoint] = [(str(path.relative_to(checkpoint)), path.stat().st_size, path.stat().st_mtime_ns)
+                                       for path in sorted(Path(checkpoint).rglob("*")) if path.is_file()]
+    package = Path(__file__).parent.parent
+    sources = {name: hashlib.sha256((package / name).read_bytes()).hexdigest() for name in (
+        "ac_model/ac_model_training.py", "ac_model/ac_model_utils.py", "ac_model/ac_model.py",
+        "harness/hf_utils.py", "harness/loaded_model.py", "harness/module_manager.py", "agent_training/agent_training_utils.py")}
+    identity = {"kind": "ac_execution", "environment": env, "execution_schema": EXECUTION_SCHEMA, "execution_sources": sources,
+                "models": portable(asdict(harness_config)), "ac": portable(asdict(ac_config)), "checkpoints": portable(checkpoints),
+                "loras": portable([asdict(lora) for lora in loras]), "recipe": portable(recipe),
+                "workload": digest(records), "cases": [[records[i] for i in group] for group in batches],
+                "capacity_cases": [{**case, "records": [records[i] for i in case["indices"]]} for case in capacity_cases],
+                "capacity_policies": capacity_policies}
+    folder = resolve_path("AUTOTUNE/execution_profiles")
+    for path in sorted(folder.glob(f"{digest(identity)}-*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
+        profile = read_execution_profile(path, identity)
+        winner = next((row for row in profile["candidates"] if row["policy"] == profile["policy"]), None) if profile else None
+        free_bytes = torch.cuda.mem_get_info(device)[0]
+        if profile and max(row["peak_memory_bytes"] for row in winner["cases"] + winner["capacity_cases"]) < free_bytes * .96:
+            if progress_callback is not None:
+                progress_callback({"phase": "reusing measured policy", "completed_cases": len(winner["cases"]),
+                                   "total_cases": len(winner["cases"]), "elapsed_seconds": 0.})
+            return {"policy": profile["policy"], "receipt": {**profile, "origin": "cached", "path": str(path)}}
+    staging = resolve_path("AUTOTUNE/staging")
+    with tempfile.TemporaryDirectory(prefix="execution-", dir=staging) as temporary:
+        root = Path(temporary)
+        request = {"kind": "ac_execution", "identity": identity, "harness_config": harness_config,
+                   "loras": loras, "ac_config": ac_config,
+                   "ac_state": {key: value.detach().cpu() for key, value in ac.modules.state_dict().items()},
+                   "examples": selected, "batches": [[local[i] for i in group] for group in batches],
+                   "capacity_cases": [{**case, "indices": [local[i] for i in case["indices"]]} for case in capacity_cases],
+                   "capacity_policies": capacity_policies,
+                   "training_config": training_config, "seconds": float(seconds), "device": device.index,
+                   "autotune_root": str(sync_root().resolve()),
+                   "progress": str(root / "progress.json"),
+                   "sync_root": str(root / "isolated_run"), "output": str(root / "result.json")}
+        request_path = root / "request.pkl"
+        with request_path.open("wb") as handle:
+            pickle.dump(request, handle, protocol=pickle.HIGHEST_PROTOCOL)
+        log_path = resolve_path("AUTOTUNE/logs") / f"execution-{time.time_ns()}.log"
+        if progress_callback is not None:
+            progress_callback({"phase": "starting execution worker", "completed_cases": 0,
+                               "total_cases": len(batches) * (len(capacity_policies) + 2)
+                                              + 2 * len(capacity_cases) + sum(len(row["cases"]) for row in capacity_policies),
+                               "elapsed_seconds": 0.})
+        with log_path.open("w") as log:
+            process = subprocess.Popen([sys.executable, "-m", "activation.autotuners.worker", "--ac-execution", str(request_path)],
+                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
+            try:
+                _wait_execution_worker(process, Path(request["progress"]), time.monotonic() + float(seconds) + 10,
+                                       progress_callback)
+            except BaseException as error:
+                try:
+                    os.killpg(process.pid, signal.SIGKILL)
+                except ProcessLookupError:
+                    pass
+                process.wait()
+                if not isinstance(error, subprocess.TimeoutExpired):
+                    raise
+        profile = read_execution_profile(Path(request["output"]), identity)
+        if profile is None:
+            raise AutotuningUnavailable(f"No complete execution candidate passed within the budget; details {log_path}")
+        profile["worker_log"] = str(log_path)
+        profile["worker_exit_code"] = process.returncode
+        path = publish(profile, folder)
+        return {"policy": profile["policy"], "receipt": {**profile, "origin": "measured", "path": str(path)}}
````

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title"><a href="trajectory_implementation/activation/autotuners/worker.py">activation/autotuners/worker.py</a></span>
    <span class="card-oneliner">Measure complete forward/backward/Adam execution and reject candidates that change loss or gradients.</span>
  </summary>

````diff-python
diff --git a/activation/autotuners/worker.py b/activation/autotuners/worker.py
index e89e54b..c6132c1 100644
--- a/activation/autotuners/worker.py
+++ b/activation/autotuners/worker.py
@@ -120,5 +120,366 @@ def main():
         save("bounded_search", trial, error)
 
 
+
+
+def _execution_gradient_check(reference, actual):
+    """Compare every trainable gradient, including None ownership, by optimizer group."""
+    import math
+    import torch
+    if reference.keys() != actual.keys():
+        return {"passed": False, "reason": "gradient ownership changed"}
+    sums = {}
+    for key, first in reference.items():
+        second = actual[key]
+        if (first is None) != (second is None):
+            return {"passed": False, "reason": f"gradient presence changed: {key}"}
+        if first is None:
+            continue
+        if not torch.isfinite(first).all() or not torch.isfinite(second).all():
+            return {"passed": False, "reason": f"nonfinite gradient: {key}"}
+        values = sums.setdefault(key.split(":")[0], [0., 0., 0., 0.])
+        first, second = first.float(), second.float()
+        values[0] += float(first.square().sum())
+        values[1] += float(second.square().sum())
+        values[2] += float((first - second).square().sum())
+        values[3] += float((first * second).sum())
+    groups = {key: {"relative_l2": math.sqrt(delta / max(first, 1e-30)),
+                    "cosine": dot / math.sqrt(first * second) if first * second else float(first == second),
+                    "reference_norm": math.sqrt(first)}
+              for key, (first, second, delta, dot) in sums.items()}
+    return {"passed": bool(groups) and all(row["relative_l2"] <= .05 and row["cosine"] >= .995 for row in groups.values()),
+            "groups": groups}
+
+
+def _execution_progress(path: Path, phase: str, completed_cases: int, total_cases: int, started: float, **details):
+    payload = {"phase": phase, "completed_cases": completed_cases, "total_cases": total_cases,
+               "elapsed_seconds": time.monotonic() - started, **details}
+    temporary = path.with_suffix(".partial")
+    temporary.write_text(json.dumps(payload, allow_nan=False))
+    temporary.replace(path)
+
+
+def ac_execution_main(request_path):
+    """Disposable full-model SFT timing; trusted pickle is created by the parent API."""
+    import copy
+    from dataclasses import replace
+    import gc
+    import os
+    import pickle
+    import random
+    import traceback
+    import torch
+    from .core import digest, execution_candidates, SCHEMA, EXECUTION_SCHEMA
+    with Path(request_path).open("rb") as handle:
+        request = pickle.load(handle)
+    os.environ["ACTIVATION_SYNC_ROOT"] = request["sync_root"]
+    Path(request["sync_root"]).mkdir(parents=True, exist_ok=True)
+    torch.cuda.set_device(request["device"])
+    device = torch.device("cuda", request["device"])
+    torch.set_num_threads(min(8, os.cpu_count() or 1))
+    from ..harness import HarnessRuntime
+    from ..ac_model.ac_model import TRAINING
+    from ..ac_model.ac_model_training import ActivationContextTrainer
+    from ..ac_model.ac_model_utils import execution_groups, execution_batch_loss, AdamMomentPlacement
+    from ..agent_training.agent_training_utils import disable_dropout, optimizer_state_to, set_learning_rates
+    from . import get_ac_training_autotuning_variables, configure_fla_runtime
+    started = time.monotonic()
+    deadline = started + request["seconds"]
+    config = replace(request["training_config"], resume_optimizer=False, resume_skip_steps=0,
+                     micro_batch_examples=1, execution_policy=None, max_duration_seconds=None)
+    candidates = execution_candidates(config)
+    completed_cases = 0
+    capacity_cases = request["capacity_cases"]
+    capacity_policies = request["capacity_policies"]
+    if [row["policy"] for row in capacity_policies] != candidates:
+        raise ValueError("Execution worker candidate policies differ from the capacity request")
+    total_cases = (len(request["batches"]) * (len(candidates) + 2) + 2 * len(capacity_cases)
+                   + sum(len(row["cases"]) for row in capacity_policies))
+    progress_path = Path(request.get("progress", str(Path(request["output"]).with_name("progress.json"))))
+    def progress(phase, **details):
+        _execution_progress(progress_path, phase, completed_cases, total_cases, started, **details)
+    progress("model registration")
+    torch.manual_seed(config.seed)
+    harness = HarnessRuntime(request["harness_config"])
+    for lora in request["loras"]:
+        harness.module_manager.register_lora(lora.lora_name, lora.model_name, rank=lora.rank, alpha=lora.alpha,
+                                             dropout=lora.dropout, target_modules=lora.target_modules,
+                                             checkpoint_path=lora.checkpoint_path)
+    ac = harness.module_manager.register_ac_model(request["ac_config"])
+    ac.modules.load_state_dict(request["ac_state"], strict=True)
+    trainer = ActivationContextTrainer(harness, config)
+    trainer._training_deadline = None
+    # Reuse the shared kernel catalog without writing trial checkpoints into a run.
+    progress("kernel preparation")
+    os.environ["ACTIVATION_SYNC_ROOT"] = request["autotune_root"]
+    try:
+        variables = get_ac_training_autotuning_variables(ac.side.model_config, ac.target.model_config, config,
+                                                        ac_config=ac.config, device=device)
+    finally:
+        os.environ["ACTIVATION_SYNC_ROOT"] = request["sync_root"]
+    trainer._logits_chunk_tokens = variables["logits_chunk_tokens"]
+    progress("model placement")
+    harness.module_manager.ensure_lora(ac.config.target_model_lora_name)
+    ac.prepare()
+    ac.set_mode(TRAINING)
+    for base in dict.fromkeys((ac.side.model, ac.target.model)):
+        base.train(); disable_dropout(base)
+    ac._training_gradient_checkpointing = config.gradient_checkpointing
+    # Match train()'s FP32 LoRA/gate matmul mode, not the process default.
+    torch.backends.cuda.matmul.allow_tf32 = True
+    torch.backends.cudnn.allow_tf32 = True
+    examples = request["examples"]
+    batches = [[examples[i] for i in group] for group in request["batches"]]
+    trainer._check_capacity(ac, examples)
+    groups = [{"params": ac.trainable_parameters(), "lr": config.learning_rate_ac},
+              {"params": harness.module_manager.lora_parameters(ac.config.target_model_lora_name),
+               "lr": config.learning_rate_target_lora}]
+    optimizer = torch.optim.AdamW(groups, betas=config.adam_betas, eps=config.adam_eps, weight_decay=config.weight_decay)
+    for group in optimizer.param_groups:
+        group["base_lr"] = group["lr"]
+    set_learning_rates(optimizer, 0, config.warmup_updates, total_updates=config.schedule_total_updates,
+                       schedule=config.lr_schedule, final_fraction=config.lr_final_fraction)
+    params = {f"{group}:{index}": parameter for group, values in enumerate(optimizer.param_groups)
+              for index, parameter in enumerate(values["params"])}
+    placement = None
+    rng = random.Random(config.seed)
+    flags = [[bool(example.penalty is not None and config.penalty_active
+                   and rng.random() < config.no_context_penalty_fraction) for example in batch] for batch in batches]
+    records = []
+    baseline_gradients, baseline_losses = [], []
+    capacity_references = {}
+    frozen_weights = frozen_optimizer = None
+
+    def publish_result():
+        passing = [row for row in records if row["passed"]]
+        if not passing:
+            return
+        winner = min(passing, key=lambda row: row["seconds_per_question"])
+        profile = {"kind": "ac_execution", "execution_schema": EXECUTION_SCHEMA,
+                   "identity": request["identity"], "policy": winner["policy"], "candidates": records,
+                   "validation": {"passed": True, "loss_absolute_tolerance": .01, "gradient_relative_l2": .05,
+                                  "gradient_cosine": .995, "baseline_repeat": True},
+                   "sampling": {"batch_indexes": request["batches"], "penalty_flags": flags,
+                                "capacity_cases": capacity_cases,
+                                "scope": "sampled update shapes and distinct-history padding tail for speed; policy-specific checkpoint boundaries for capacity, with fixed existing penalties; no training-quality or full-resume claim"},
+                   "seconds": time.monotonic() - started,
+                   "limits": "Same supplied-workload SFT policy; unchanged reference/evaluation execution. Speed includes forward/backward/clipping/Adam/transfers; excludes capacity checks, reference preparation and reporting."}
+        payload = {"schema": SCHEMA, "profile": profile, "sha256": digest(profile)}
+        path = Path(request["output"])
+        temporary = path.with_suffix(".partial")
+        temporary.write_text(json.dumps(payload, allow_nan=False))
+        temporary.replace(path)
+
+    def clear_graphs():
+        optimizer.zero_grad(set_to_none=True)
+        trainer._penalty_term = None
+        trainer._last_hits = None
+        ac.invalidate_encoded_rows()
+        for value in ac.intervention_diagnostics.values():
+            value.clear()
+        trainer._row_rms.clear()
+        gc.collect()
+        torch.cuda.empty_cache()
+
+    def configure(policy, restore=True):
+        nonlocal placement
+        if placement is not None:
+            placement.close(); placement = None
+        clear_graphs()
+        if restore and frozen_weights is not None:
+            with torch.no_grad():
+                for key, parameter in params.items():
+                    parameter.copy_(frozen_weights[key])
+            # load_state_dict casts moments to the parameter device; clone CPU step tensors
+            # because Adam updates them in-place and the immutable snapshot must not alias.
+            state = {key: {**value, "step": value["step"].clone()} for key, value in frozen_optimizer["state"].items()}
+            optimizer.load_state_dict({"state": state, "param_groups": copy.deepcopy(frozen_optimizer["param_groups"])})
+        if policy["optimizer_moments"] == "cpu":
+            placement = AdamMomentPlacement(optimizer)
+            placement.park()
+        else:
+            optimizer_state_to(optimizer, device)
+        ac.config.encode_batch_max_tokens = policy["encode_batch_tokens"]
+        ac._side_gradient_checkpointing_min_tokens = policy["side_checkpoint_tokens"]
+        trainer._target_gradient_checkpointing_min_tokens = policy["target_checkpoint_tokens"]
+        torch.cuda.synchronize()
+
+    class FixedPenaltyDraws:
+        def __init__(self, batch, selected):
+            self.values = iter(0. if flag else 1. for example, flag in zip(batch, selected)
+                               if example.penalty is not None and config.penalty_active)
+        def random(self):
+            return next(self.values)
+
+    def measure(policy, batch, selected, *, native=False, capture=True):
+        optimizer.zero_grad(set_to_none=True)
+        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(device)
+        begin = time.monotonic()
+        objective_value = 0.
+        encodes_before, side_before = ac.stats.encodes, ac.stats.side_tokens
+        if native:
+            trainer._penalty_rng = FixedPenaltyDraws(batch, selected)
+            for example in batch:
+                objective, _, _, _ = trainer._example_loss(ac, example, device)
+                penalty = trainer._penalty_term
+                loss = (objective + (penalty if penalty is not None else 0.)) * (example.item.weight / len(batch))
+                loss.backward(); objective_value += float(loss.detach())
+                del objective, penalty, loss
+        else:
+            for group in execution_groups(batch, policy):
+                actual = [batch[i] for i in group]
+                outcomes = execution_batch_loss(trainer, ac, actual, device, [selected[i] for i in group], policy)
+                loss = sum((row["objective"] + (row["penalty"] if row["penalty"] is not None else 0.))
+                           * (example.item.weight / len(batch)) for example, row in zip(actual, outcomes))
+                loss.backward(); objective_value += float(loss.detach())
+                del outcomes, loss, actual
+        if config.learning_rate_target_lora == 0:
+            for parameter in groups[1]["params"]:
+                parameter.grad = None
+        torch.cuda.synchronize()
+        backward_seconds = time.monotonic() - begin
+        gradients = {key: parameter.grad.detach().cpu().clone() if parameter.grad is not None else None
+                     for key, parameter in params.items()} if capture else None
+        # CPU gradient copies/validation are excluded from measured execution time.
+        before_step = time.monotonic()
+        torch.nn.utils.clip_grad_norm_([p for p in params.values() if p.grad is not None], config.max_grad_norm)
+        optimizer.step()
+        ac.invalidate_encoded_rows()
+        torch.cuda.synchronize()
+        seconds = backward_seconds + time.monotonic() - before_step
+        result = {"seconds": seconds, "questions": len(batch), "objective": objective_value,
+                  "peak_memory_bytes": torch.cuda.max_memory_allocated(device),
+                  "encodes": ac.stats.encodes - encodes_before, "side_tokens": ac.stats.side_tokens - side_before}
+        return result, gradients
+
+    try:
+        with configure_fla_runtime(variables["fla_config"]):
+            # Singleton, plain-base no-context references stay independent of candidate batching.
+            progress("reference preparation")
+            trainer._store_teacher_targets(ac, [example.penalty for example in examples if example.penalty is not None],
+                                           device, lora=None)
+            configure(candidates[0], restore=False)
+            progress("optimizer warmup")
+            measure(candidates[0], batches[0], flags[0], native=True, capture=False)
+            # Every comparison starts at this identical, real post-Adam state.
+            frozen_weights = {key: parameter.detach().cpu().clone() for key, parameter in params.items()}
+            frozen_optimizer = copy.deepcopy(optimizer.state_dict())
+            for state in frozen_optimizer["state"].values():
+                for key, value in state.items():
+                    if torch.is_tensor(value):
+                        state[key] = value.detach().cpu().clone()
+            baseline_record = {"policy": candidates[0], "passed": True, "cases": [], "capacity_cases": [], "baseline": True}
+            for index, (batch, selected) in enumerate(zip(batches, flags)):
+                progress("native baseline", case=index + 1)
+                configure(candidates[0])
+                first, gradients = measure(candidates[0], batch, selected, native=True)
+                completed_cases += 1
+                progress("native repeat", case=index + 1)
+                configure(candidates[0])
+                repeated, repeated_gradients = measure(candidates[0], batch, selected, native=True)
+                comparison = _execution_gradient_check(gradients, repeated_gradients)
+                passed = comparison["passed"] and abs(first["objective"] - repeated["objective"]) <= .01
+                if not passed:
+                    raise RuntimeError(f"Native repeated baseline failed numerical gate: {comparison}")
+                completed_cases += 1
+                first.update({"case": index, "repeat": repeated, "gradient_check": comparison, "passed": True})
+                progress("singleton helper", case=index + 1)
+                configure(candidates[0])
+                helper, helper_gradients = measure(candidates[0], batch, selected)
+                helper_check = _execution_gradient_check(gradients, helper_gradients)
+                if not helper_check["passed"] or abs(helper["objective"] - first["objective"]) > .01:
+                    raise RuntimeError(f"Execution helper singleton failed native equivalence: {helper_check}")
+                completed_cases += 1
+                helper.update({"case": index, "native": first, "gradient_check": helper_check, "passed": True})
+                helper["passed"] = helper["peak_memory_bytes"] < torch.cuda.get_device_properties(device).total_memory * .96
+                baseline_record["passed"] = baseline_record["passed"] and helper["passed"]
+                baseline_record["cases"].append(helper)
+                baseline_gradients.append(gradients); baseline_losses.append(first["objective"])
+                del repeated_gradients, helper_gradients
+            baseline_record["seconds_per_question"] = sum(row["seconds"] for row in baseline_record["cases"]) / sum(map(len, batches))
+            records.append(baseline_record); publish_result()
+            for candidate_index, policy in enumerate(candidates[1:], 1):
+                if time.monotonic() >= deadline:
+                    break
+                record = {"policy": policy, "cases": [], "capacity_cases": [], "passed": True}
+                try:
+                    for index, (batch, selected) in enumerate(zip(batches, flags)):
+                        if time.monotonic() >= deadline:
+                            raise TimeoutError("Execution search budget reached")
+                        progress("candidate case", candidate=candidate_index, policy=policy, case=index + 1)
+                        configure(policy)
+                        row, gradients = measure(policy, batch, selected)
+                        comparison = _execution_gradient_check(baseline_gradients[index], gradients)
+                        row.update({"case": index, "gradient_check": comparison,
+                                    "loss_difference": abs(row["objective"] - baseline_losses[index])})
+                        row["passed"] = comparison["passed"] and row["loss_difference"] <= .01
+                        # Avoid choosing an arm with essentially no remaining capacity.
+                        row["passed"] = row["passed"] and row["peak_memory_bytes"] < torch.cuda.get_device_properties(device).total_memory * .96
+                        record["cases"].append(row)
+                        completed_cases += 1
+                        del gradients
+                        if not row["passed"]:
+                            record["passed"] = False
+                            break
+                    if record["passed"]:
+                        for case_index in capacity_policies[candidate_index]["cases"]:
+                            if time.monotonic() >= deadline:
+                                raise TimeoutError("Execution search budget reached before capacity validation")
+                            case = capacity_cases[case_index]
+                            batch = [examples[i] for i in case["indices"]]
+                            selected = case["penalty_flags"]
+                            if case_index not in capacity_references:
+                                progress("capacity native baseline", capacity_case=case_index, reasons=case["reasons"])
+                                configure(candidates[0])
+                                first, gradients = measure(candidates[0], batch, selected, native=True)
+                                completed_cases += 1
+                                progress("capacity native repeat", capacity_case=case_index)
+                                configure(candidates[0])
+                                repeated, repeated_gradients = measure(candidates[0], batch, selected, native=True)
+                                comparison = _execution_gradient_check(gradients, repeated_gradients)
+                                completed_cases += 1
+                                if not comparison["passed"] or abs(first["objective"] - repeated["objective"]) > .01:
+                                    raise RuntimeError(f"Capacity repeated baseline failed numerical gate: {comparison}")
+                                first.update({"repeat": repeated, "gradient_check": comparison})
+                                capacity_references[case_index] = (first, gradients)
+                                del repeated_gradients, gradients
+                            first, native_gradients = capacity_references[case_index]
+                            progress("candidate capacity", candidate=candidate_index, policy=policy,
+                                     capacity_case=case_index, reasons=case["reasons"])
+                            configure(policy)
+                            row, gradients = measure(policy, batch, selected)
+                            comparison = _execution_gradient_check(native_gradients, gradients)
+                            row.update({"capacity_case": case_index, "gradient_check": comparison,
+                                        "loss_difference": abs(row["objective"] - first["objective"]), "native": first})
+                            row["passed"] = (comparison["passed"] and row["loss_difference"] <= .01
+                                and row["peak_memory_bytes"] < torch.cuda.get_device_properties(device).total_memory * .96)
+                            record["capacity_cases"].append(row)
+                            completed_cases += 1
+                            del gradients
+                            if not row["passed"]:
+                                record["passed"] = False
+                                break
+                except (torch.cuda.OutOfMemoryError, TimeoutError) as error:
+                    record.update({"passed": False, "error": str(error), "error_type": type(error).__name__})
+                    error.__traceback__ = None  # release the failed forward's graph before the next restored candidate
+                    clear_graphs()
+                record["seconds_per_question"] = (sum(row["seconds"] for row in record["cases"]) / sum(map(len, batches))) if record["passed"] else None
+                records.append(record); publish_result()
+                progress("candidate complete", candidate=candidate_index, policy=policy, passed=record["passed"])
+                print(json.dumps({"policy": policy, "passed": record["passed"], "seconds_per_question": record["seconds_per_question"]}), flush=True)
+            progress("complete", tested_candidates=len(records), planned_candidates=len(candidates),
+                     budget_exhausted=time.monotonic() >= deadline)
+    except Exception as error:
+        progress("failed", error=str(error))
+        traceback.print_exc()
+        raise
+    finally:
+        if placement is not None:
+            placement.close()
+
+
 if __name__ == "__main__":
-    main()
+    if sys.argv[1:2] == ["--ac-execution"]:
+        ac_execution_main(sys.argv[2])
+    else:
+        main()
````

</details>

