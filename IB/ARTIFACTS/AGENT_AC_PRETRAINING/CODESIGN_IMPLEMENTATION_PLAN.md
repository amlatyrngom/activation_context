# AC co-design implementation — source

Accepted interfaces: CODESIGN_INTERFACE_PLAN.md. Accepted test files: CODESIGN_TEST_REVIEW.md. User authorized this full phase plus the third-file compatibility migration on 2026-09-15. Baseline/evidence: IB/TMP/AGENT_AC_PRETRAINING/codesign_implementation/. Follow the paired projection verbatim, classify review findings, and preserve prior unrelated changes.

## Human-facing projection

# AC and agent training — implementation

Implement the accepted interfaces in the existing objects, then verify their numerical behavior and produce the reviewed handoff. The user authorized this entire phase on 2026-09-15, including the minimal third-test compatibility edit.

## Executive Summary

AgentTrainer replays exact captured AC rows and updates only its reader adapter. ACTrainer re-encodes raw parts and jointly updates the encoder, side adapter and reader adapter with KL or SFT over all retained assistant outputs. JSONL writers save tensors beside their records and release completed result buffers only after successful publication.

| Area | Existing implementation to extend | Key check |
| --- | --- | --- |
| Capture and persistence | Agent, AgentRunResult, RolloutCache, agent_utils | Exact bytes, portable references, failed writes and retained aliases |
| Frozen-row policy training | TrainingExample, Collated, AgentTrainer | Independent clipped-loss and gradient oracle |
| Whole histories | AC training item, study generator, existing dialect | Independent token ledger; every assistant output once |
| KL/SFT | Current AC trainer and reporter | Real tiny-model objectives, joint gradients, no SFT teacher work |
| Shared base | ModuleManager, LoadedModel, existing HF utilities | Separate/shared adapter and recomputation parity |
| Handoff | Two complete mainline tests, one compatibility migration | Private proof matrix, exact diffs, honest validation limits |

No new utility module, storage class, registry, lazy tensor object or training destination abstraction. Active recipes migrate to the accepted item schema; historical snapshots, the teacher corpus, other workstreams, DDP and long quality experiments stay outside this phase.

## Requested Decisions

**Authorized in chat:** “Ok go for it.” Details, independent review, implementation, validation, re-review and handoff are one continuous phase. The only mainline test changes are the two accepted complete files plus the approved `base_dir` migration in `test_basic_agent_ac.py`. Routine review findings are resolved within this intent. A finding that requires changing an accepted public behavior or scope returns to the user.

The first independent review passed and implementation is authorized. The user additionally requested both full end-to-end files and private behavior probes on Sky. Those targeted GPU runs are included in this phase; a long quality campaign remains outside scope.

## Milestones

<details class="card" data-tressoir-markdown open>
  <summary><span class="card-title">M1 — Capture, persistence and frozen-row training</span><span class="card-oneliner">Keep storage work in existing utilities and preserve exact observations.</span><span class="card-badge">Completed</span></summary>

#### Completion report

**What landed:** Existing row dictionaries now capture the exact final CPU inputs. Existing agent utilities write portable tensor files, resolve/load/export references and release live buffers only after JSONL publication. AgentTrainer consumes those fixed rows through padding/packing and updates its reader adapter.

**Drifts and challenges:** Review tightened native missing-raw-data errors and append/flush failure handling. No utility module or storage class was added.

**Focused actual delta:**

```diff
-        self.run_results.trajectory.append(asdict(step))
+        self.run_results.trajectory.append(trajectory_step_dict(step))
```

The full per-file delta is supplied in the final co-design handoff; this excerpt omits surrounding unchanged agent logic.

**Validation:** Private persistence/backpressure/resume tests, independent policy loss/gradient/update proofs and strengthened GPU padded/packed BF16 checks passed. HotpotQA saved-row/cache replay passed both modes.

#### Planning Overview

Capture detached contiguous CPU tensors after the agent's final cast; avoid a second copy when rows already meet this contract. Each prompt/step span stores start, length and rows. Resume consumes captured rows and never calls the encoder; an unavailable encoder is valid for fixed-row training. Preserve span offsets through thinking removal, padded rows, packed sequence offsets and convolution spacers. Validate ordered nonoverlapping spans, bounds, row length/width/dtype and target-mask exclusion before an update; target dtype mismatch errors rather than silently changing the recorded observation.

Serialization operates only on known prompt/trajectory spans and nested run records (compactions/subagents), not arbitrary user/tool dictionaries. Copy containers without duplicating tensors. Hash contiguous CPU payload bytes together with dtype and shape; write one tensor-only file per content hash using a unique temporary file and atomic publication. Existing hash targets must validate; tensors load with weights_only=True on CPU and retain dtype. Serialize relative destination references, copying referenced source files into the destination when exporting. Deserialize resolves relative paths against base_dir without eager tensor loads; in-memory paths are absolute. Missing/corrupt files fail explicitly on access.

The original result retains its captured tensors until the JSONL append and flush succeed. Then release through its actual span dictionaries, recursively, so retained aliases/futures see references too. Cache rows are installed only after append success. Keep the rollout concurrency permit until save completes; disabled cache and deliberately unsaved deadline results retain their captured tensors. Inspection report data uses shape/dtype/path metadata without tensor contents. A torn append may leave orphan tensors; garbage collection is deferred.

#### Planned Changes

`activation/agent/agent_utils.py · row helpers`

```python
# Implement the accepted helpers here; algorithms above, bodies omitted.
capture_ac_spans(spans, rows)
trajectory_step_dict(step)
serialize_ac_rows(data, *, base_dir=None)
deserialize_ac_rows(data, *, base_dir=None)
release_ac_rows(result, serialized, *, base_dir)
load_ac_rows(span)
ac_spans_for_report(spans)
```

`activation/agent/agent.py · segment capture, step capture, resume_from_run_result`

```diff
- results.prompt_ac_spans = [{"start": start, "length": end - start} for start, end in spans]
+ results.prompt_ac_spans = capture_ac_spans(spans, rows)
- self.run_results.trajectory.append(asdict(step))
+ self.run_results.trajectory.append(trajectory_step_dict(step))
# Apply the same capture rule to step spans; resume loads each captured span.
# Surrounding existing agent code omitted.
```

`activation/agent/agent_config.py · AgentRunResult.serialize / deserialize`

```python
# Keep existing arguments; add keyword-only base_dir and delegate span conversion.
# Child/compaction serialization propagates the same destination.
# No tensor-bearing JSON serialization without a destination.
```

`activation/agent/rollout_caching.py · RolloutCache.append`

```python
# Serialize with self.path.parent; append/flush under the current lock;
# update cached row and release actual result rows only after success.
```

`activation/agent/rollout_manager.py · _run_jobs / _run_pending`

```python
# Cached deserialization receives cache.path.parent.
# Worker saves a non-deadline result before limiter.release(); remove outer append.
```

`activation/agent/rollout_reporter.py · report JSON construction`

```python
# Use metadata conversion rather than full tensor-bearing result serialization.
# Preserve current rendered trajectory detail; avoid asdict tensor copies.
```

`activation/agent_training/agent_training_utils.py · TrainingExample / Collated / build_example / collate`

```python
# Add ac_spans to the existing example and mapped row/offset/span entries to Collated.
# One existing-utility helper embeds tokens and overwrites only validated AC spans.
# Both policy passes consume that helper; no encoder/model lookup is required.
```

`activation/agent_training/agent_trainer.py · train / _hidden_and_logprobs`

```python
# Add keyword-only reset_optimizer=False. Discard this adapter's optimizer when set.
# Keep weights, rounds and warmup/update clocks. Use fixed-row batch embeddings.
# Preserve existing selection/model-lora buckets, old-logprob rules and clipped objective.
```

</details>

<details class="card" data-tressoir-markdown>
  <summary><span class="card-title">M2 — Paired histories and complete public windows</span><span class="card-oneliner">Reuse incremental dialect tokenization; supervise complete recorded histories.</span><span class="card-badge">Completed</span></summary>

#### Completion report

**What landed:** Paired histories have independent starts and target positions for every retained assistant output. Public windows end on complete turns; recorded roots/segments/children preserve exact outputs with stable grouping. Raw source tools and separate/inline reasoning remain visible.

**Drifts and challenges:** Final review found and corrected input-order-dependent root IDs, missing native raw-part acceptance, nested source-tool omissions and native inline-reasoning stripping.

**Focused actual delta:**

```diff
-            root_key = f"{root.agent_config.agent_name}:{root.seed}:{root_index}"
+            root_key = f"{config_key(root.agent_config)}:{root.seed}"
```

This shows the final-review correction; the handoff contains the full change against `/source`.

**Validation:** Native-template/history/QA/filter/error CPU proofs and both real HotpotQA modes passed. All three public-compaction/trajectory-QA/RAG-QA GPU cases passed.

#### Planning Overview

Replace the old completion-only item fields with the accepted two histories, independent starts and assistant_token_ids. An existing AC utility prepares one view: template its prefix once, append each authoritative assistant token sequence, and build intervening non-assistant wrappers with ModelDialect.continuation_tokens/join_continuation. Record state positions immediately before each target, adjusting part spans after a dropped duplicate EOT. Do not re-template earlier assistant outputs, which can discard reasoning. Include final input-only wrappers when recorded. Prefix assistant messages and nested transcripts never become targets.

Validate starts, suffix role/call ordering, assistant metadata equality, assistant count, nonempty integer IDs and provenance. Teacher and student have independent position arrays with identical target IDs. SFT validates/prepares only its student history; a teacher view may be absent. Current compatible recorded IDs remain authoritative, including incomplete tails; never append an invented terminal token. Public/QA assistant outputs are rendered once using the target dialect and actual EOT conventions, including reasoning and tool calls.

For each selected root, visit every existing compaction/current segment and its children once, using stable root origin keys to prevent held/train overlap. Default KL selects all with unit weights; SFT selects score exactly 1. Explicit filters override defaults. Validate root custom weights after selection; inherit to every child/segment, count zero weights and empty segments. Missing required raw data or incompatible target tokenizer raises with source identity.

For text records, rebuild channels using activation_messages_of and current initial-frame builders. Compress the complete recorded initial frame into an initial part before the first assistant target while retaining system/tool framing and the task; raw injected/search/subagent/tool parts remain at their original causal positions. Native AC histories retain their raw parts and get a channel-expanded teacher view. Expand prefixes separately from retained suffixes so historical assistant context cannot be mistaken for target messages. Keep current segment boundaries; delete the random-cut recorded-run conversion and obsolete arguments.

Reasoning-preservation migration also covers trajectory_chars and the existing dataset_utils.message_text/render_message/render_messages chain used by loader size filters, dataset_index._trajectory_pieces, BM25/study evidence and expanded teacher transcripts. Keep one consistent plain-text view containing reasoning exactly once. Source spans refer to that same combined text; a partial _source_window_messages slice must remove the structured reasoning field after assigning the sliced combined text, while a full-message slice retains its structured fields. Verify a reasoning-dominant fixture passes source-size eligibility, appears in indexed/study evidence and reconstructs chunk windows without omission or duplication. Reuse these existing helpers; change dataset_index only if its current delegation is insufficient.

For public compaction examples, normalize long splittable reasoning/text into complete assistant turns, inserting runtime-shaped Python continuation calls and matching tool responses. Preserve source order, reasoning and tool definitions; use unique call IDs and arguments. Choose compaction cuts only at the resulting complete-turn boundaries, then retain the maximal complete continuation fitting the all-role 8192 budget and total capacity. Count rendered wrappers/endings/tool replies exactly with the same incremental builder. All retained assistant tokens, including synthetic calls, are targets. Never split indivisible calls or discard their replies; report candidate/sample shortfalls explicitly. QA generators become one-assistant items; evaluation references keep the suffix and target IDs intact.

#### Planned Changes

`activation/ac_model/ac_model_training.py · ActivationContextTrainingItem / _Example`

```python
# Adopt the full field lists from CODESIGN_INTERFACE_PLAN M2/M3.
# _Example stores target IDs, teacher/student state positions, spans and requests.
```

`activation/ac_model/ac_model_utils.py · paired-history preparation helpers`

```python
# Keep incremental token assembly, target alignment and assistant rendering here.
# Use the existing ModelDialect; no new item class or utility module.
```

`activation/ac_model/ac_model_study.py · compaction, harvesting, QA, references`

```python
# Replace old one-completion/random-cut builders with whole-segment items.
# Remove immediate_continuation_ratio/completion_max_tokens from public compaction.
# Remove seed/text_run_kwargs/items_per_text_run/completion_max_tokens from harvesting.
# Preserve public QA entry points, root origins and source-aware split_items.
```

`activation/dataset/loaders/trajectory_utils.py · reformat_trajectory`

```python
# Preserve reasoning as its own assistant field for target-dialect rendering.
# Keep tool mapping, original observed answers and source frames.
```

`activation/dataset/dataset_utils.py · message_text / render_message / render_messages`

```python
# Include structured assistant reasoning exactly once in the shared plain-text view.
# trajectory_chars delegates to the same view; indexing offsets share that representation.
```

`activation/dataset/dataset_index.py · _trajectory_pieces; ac_model_study.py · _source_window_messages`

```python
# Audit source-span offsets against the shared combined text; keep existing delegation
# where already correct. Partial reconstructed slices must not duplicate reasoning.
```

`active AC recipe and autotuning callers · discovered by symbol search`

```python
# Migrate item construction/printing/cache identities to the accepted schema.
# Old serialized training items raise a regenerate-items error.
# Historical IB snapshots are not rewritten.
```

</details>

<details class="card" data-tressoir-markdown>
  <summary><span class="card-title">M3 — KL/SFT and shared-base correctness</span><span class="card-oneliner">Joint gradients, truthful reporting and adapter-safe recomputation.</span><span class="card-badge">Completed</span></summary>

#### Completion report

**What landed:** KL and literal SFT use all selected assistant targets, truthful objective-specific reports and explicit optimizer reset. Shared bases retain distinct adapter ownership through eager and checkpointed execution.

**Drifts and challenges:** Private gradients exposed PEFT trainability restoration; final review tightened mixed-reporter rejection and the drift forward's own checkpoint threshold. These preserve the accepted interface.

**Focused actual delta:**

```python
student_hidden = target.decoder_forward(student_embeds[None], None, lora_name=lora, gradient_checkpointing=checkpointed)[0]
```

This is the final student-forward call; surrounding loss preparation is omitted here and included in the handoff.

**Validation:** Dense CPU KL/CE gradients, optimizer/cache/checkpoint/report lifecycle and strengthened GPU independent FP32 references passed. All three shared side/reader checkpoint combinations matched eager gradients on BF16 Qwen3.5.

#### Planning Overview

Add loss_kind to the current config/stats/report flow. KL retains the exact/coarsened implementation but indexes all selected teacher/student positions. Its cache identity includes teacher tokens and selected positions, belongs to one train call, and is bypassed in evaluation. SFT uses chunked cross-entropy against target IDs and performs no teacher preparation, forward, cache or drift computation, including baseline evaluation and completion samples. KL/drift options incompatible with SFT fail at configuration. Report common loss/loss_kind plus CE or KL honestly; SFT KL/agreement fields are absent/None, not zero KL. Reject mixed modes in one reporter.

Preflight all training/reporting/validation examples before updates: configured 72k guard, actual target model capacity and each recursive side forward's capacity. Fail with item/view/count/limit, not a dropped or shortened item. Preserve per-item token means and the current item-weighted mean. Parameter ownership stays AC modules + side LoRA and reader LoRA. reset_optimizer clears the receiving trainer's complete optimizer state without resetting weights, epoch/round or warmup/update clocks. Saved epoch pairs and reader exchange ownership stay unchanged.

Allow side and reader to reference one registered LoadedModel with distinct actual adapter names. Equal model IDs in separate registrations remain separate. ModuleManager.lora_parameters returns the named adapter's parameters without selecting an unrelated adapter. lora_context saves/restores active adapters and trainability; PEFT set_adapter must not freeze a parameter group whose differentiable graph is still live. Ensure both adapters exist before assembling parameter groups, then keep both groups trainable for joint training.

LoadedModel.decoder_forward scopes every layer checkpoint callback to the adapter for that forward, using non-reentrant checkpoint context_fn for both original execution and recomputation. Put checkpoint-scoping mechanics in existing hf_utils, restore original callbacks on exit, and preserve existing threshold/configuration behavior. The recomputation context restores adapter state afterward, including nested/recursive calls. Shared-base placement/eval/cleanup operate on unique base objects; side and reader checkpoint thresholds are applied at their respective forwards so one cannot overwrite the other's policy. Test checkpointing off/on with asymmetric adapter weights and reversed forward order.

#### Planned Changes

`activation/ac_model/ac_model_training.py · config, preparation, train, eval, losses and samples`

```python
# loss_kind="kl" by default; keyword-only reset_optimizer=False on train.
# Prepared position arrays replace final-suffix slices.
# SFT invokes chunked CE; only KL invokes teacher/cache/drift branches.
# Preflight all data, preserve joint parameter groups and existing checkpoint ownership.
```

`activation/ac_model/ac_model_reporter.py · initialization and reporting`

```python
# Select CE/KL labels from loss_kind; common loss plots/stats remain interpretable.
# No fictitious SFT teacher sample or KL reference; mixed reporter modes raise.
```

`activation/ac_model/ac_model.py · constructor / recursive encoding / lifecycle`

```python
# Replace separate-LoadedModel guard with distinct side/reader adapter validation.
# Preserve recursive differentiable encoding, cache invalidation and checkpoint pair validation.
```

`activation/harness/module_manager.py · lora_context / lora_parameters`

```python
# Save/restore adapter selection and requires_grad flags; parameter ownership is explicit.
# Keep existing effective-name collision rejection and one-wrapper-per-base registry.
```

`activation/harness/hf_utils.py · checkpoint adapter scope`

```python
# Existing-module context helper wraps each active checkpoint callback with the
# forward's adapter context for original/recomputed execution, then restores callbacks.
```

`activation/harness/loaded_model.py · decoder_forward`

```python
# Add a small checkpoint-scope call around the current decoder invocation.
# Preserve the existing public signature and generation/engine behavior.
```

</details>

<details class="card" data-tressoir-markdown>
  <summary><span class="card-title">M4 — Private proofs, real examples and handoff</span><span class="card-oneliner">Prove numerical behavior without long learning runs; review the actual diff.</span><span class="card-badge">Completed</span></summary>

#### Completion report

**What landed:** Both complete mainline examples are implemented; private numerical and boundary proofs stay in IB/TMP. The [final handoff](CODESIGN_ROUND.tressoir.md) contains the 21 staged product files, checked patch, manifest and full test views.

**Drifts and challenges:** The user explicitly added Sky execution to the phase. A stalled local SSH agent temporarily blocked downloads; bypassing it restored access without affecting the node or tests. GPU fixture review strengthened unequal-length and independent-head oracles.

**Focused actual delta:** The final handoff shows both entire test files and all 21 product diffs against current /source. Forward/reverse patch checks and staged hash checks pass.

**Validation:** Plan review, implementation re-review, 47 private CPU tests, 15 focused existing regressions, both HotpotQA modes and four strengthened GPU proofs passed. All three public/QA cases passed on Sky; the node is paused after report recovery. Final independent artifact review passed with no findings.

#### Planning Overview

The [accepted test review](./CODESIGN_TEST_REVIEW.tressoir.md) defines nine proof groups. Private fixtures live only under IB/TMP/AGENT_AC_PRETRAINING/codesign_implementation. Use actual tiny PyTorch/PEFT/AC modules and independent numerical references for clipped loss, KL/CE, logits, gradients and optimizer updates; mock only external boundaries, forbidden encoder/teacher calls and failures. Use a nondegenerate asymmetric fixture with nonzero adapter B weights, multiple rows/targets/channels, recursive parts, unequal sequence lengths and clipping cases. Compare padded/packed layouts to independent separate execution; check row perturbation causality and no cross-example/future leakage.

Persistence checks use actual CPU tensor files with BF16/FP32 dtypes, hashes, moving/exporting folders and concurrent writes. Failure injection verifies original tensors survive failed file/JSONL publication, no tensor leaks through reports, actual retained aliases release after success, and delayed writes retain the concurrency permit. Test no-cache/deadline exceptions explicitly. Optimizer tests use consecutive real updates and phase reset; cache tests vary selected positions under equal teacher token sequences; paired checkpoint tests reject partial/mismatched records.

The two accepted mainline files use real public data/models/tools, no mocks/synthetic trajectories. The new file runs exactly four easy HotpotQA questions via a custom program through short KL/SFT training, paired reload and AC rollout/cache replay. Its explicit all filter makes training execute even with zero correct answers; default SFT filtering is covered privately. The existing third test receives only the approved folder argument migration. No quality-gain assertion.

After the implementation passes focused checks, obtain an independent review of correctness, proof coverage and every human-facing diff. Reclassify each finding and resolve real defects; rerun affected checks. Stage product files, an exact patch and manifest against the accepted baseline, and produce a paired handoff with complete per-file diffs plus both full final test files. Record actual commands/results and residual GPU/editor/environment limits. Preserve previous handoffs and all pre-existing user changes.

#### Planned Changes

`activation/tests/test_basic_agent_ac_training.py`

```python
# Apply the accepted complete proposal; adjust only for necessary implementation fixes.
```

`activation/tests/test_basic_ac_self_distillation.py`

```python
# Apply the accepted complete four-problem custom-program proposal.
```

`activation/tests/test_basic_agent_ac.py · existing serialization round trip`

```python
base_dir = resolve_path("AGENT_AC_TEST/roundtrip")
data = json.loads(json.dumps(agent.run_results.serialize(base_dir=base_dir)))
resumed = Agent.resume_from_run_result(harness, AgentRunResult.deserialize(data, harness, base_dir=base_dir))
# Rest of this existing test remains unchanged.
```

`IB/ARTIFACTS/AGENT_AC_PRETRAINING/CODESIGN_ROUND.tressoir.md · handoff`

```python
# Generate the current exact per-file deltas, source manifest and final full test views.
# Private validation source is excluded from the product patch.
```

</details>

## Validation and review status

First plan review passed. Implementation and independent re-review passed after correcting seven routine boundary issues: stable root identity, native missing raw parts, mixed reporter objectives, source tool frames, example delegation, drift checkpoint thresholds and inline reasoning. No user decision remains.

CPU validation: 47 private tests passed, plus 15 focused existing regressions. Sky job20 passed all five cases across both full mainline files, with no skips; strengthened private BF16 Qwen3.5 hybrid checks passed 4/4. No GPU product correction was needed. Tiny SFT quality worsened, and matched no-context KL was slightly better in all three small public/QA runs; no quality advantage is claimed.

The [handoff](CODESIGN_ROUND.tressoir.md) contains exact staged/source diffs and both full final files. Source/staged hashes, forward/reverse patch checks and artifact validation pass. Logs/reports/JUnit were recovered locally and ac-4a-camp is STOPPED, with its existing disk retained. Final independent handoff review passed with no findings; the authorized phase is complete and there is no open intent decision.
