# Harness/model co-design — settled intent

Your saved interactions and latest chat settle the training intent. The [interfaces and full test proposals](./CODESIGN_INTERFACE_PLAN.tressoir.md) were accepted in chat on 2026-09-15; the combined details/review/implementation/handoff phase awaits the user’s green light.

## Executive Summary

AgentTrainer keeps its clipped objective and trains the reader/agent LoRA on the exact rollout observation, including frozen captured AC rows. ACTrainer rebuilds differentiable rows from raw parts and jointly trains AC modules, side LoRA and reader/agent LoRA with KL or literal imitation SFT.

| Path | Data and supervision | Trainable parameters |
| --- | --- | --- |
| AgentTrainer | Recorded tokens and fixed rows; existing selection and clipped loss | Reader/agent LoRA |
| AC KL | Paired full-text and AC histories; all retained assistant targets; `all` default selection | AC modules, side LoRA, reader/agent LoRA |
| AC SFT | AC history and assistant targets; `score_1` default selection | Same joint AC parameter groups; no teacher forward |

Public generated continuations use the all-role 8192-token budget and complete turns. Converted run results retain the whole selected run across its existing causal segments, including recorded incomplete tails. Oversized recorded examples raise explicitly, as you confirmed. No 512-token training target cap remains.

## Requested Decisions

All intent controls are resolved. Their exact answers remain in the paired source; the linked interface document records chat acceptance and the remaining narrow test-compatibility exception.

| Decision | Accepted intent |
| --- | --- |
| Paired histories | A starting logical message index in each; matching retained role/order and identical assistant targets, with independently computed token offsets. Actual resets create separate pairs. |
| Training split | Encoder frozen in AgentTrainer; AC training offers KL and literal SFT. Existing agent target buckets and objective remain. |
| Reader learning | Both AC losses jointly train encoder modules, side LoRA and reader/agent LoRA so the reader learns to use compressed inputs. |
| Actual observations | Store the actual final-cast row tensors losslessly; retaining raw parts supports differentiable AC training. Main downside: storage/I/O. |
| Storage and ownership | JSONL and `saved_tensors/` share one folder. Agent captures rows; the writer supplies the destination, saves tensors before JSONL publication and releases result buffers after success. Serialized references are relative to the JSONL parent. Move/export the whole dataset. |
| Keep main files simple | Use existing `agent_utils.py` for tensor traversal, hashing, atomic writes, rebasing/loading, copy avoidance and release; create no new utilities file. Existing training utilities retain batching/masks. Main files keep small calls. Save before releasing the rollout slot; text-record growth and unsaved tensors remain explicit limits. |
| Test review checkpoint | Show complete updated `test_basic_agent_ac_training.py` and new `test_basic_ac_self_distillation.py` before the next stage; only these two mainline test files in this work’s delta. The latter uses four simple real HotpotQA problems, a custom AgenticProgram and complete short self-distillation/training/replay flow. |
| Private behavior checks | Synthetic trajectories, mocking and failure injection stay private. Tiny real-model numerical/gradient tests and a bounded GPU check establish execution contracts without long training; quality improvement is not a gate for the small mainline examples. |
| Public continuation calls | Split long splittable reasoning into complete assistant turns using runtime-shaped Python continuation calls. Supervise their assistant/control tokens exactly as ordinary emitted calls; mask tool outputs. Use distinct arguments as well as call IDs to avoid duplicate-call guards. |
| Public endings | Count all continuation tokens, including tool replies and wrappers, within 8192. End on complete turns; remove the old immediate/delayed one-target mixture and 512 cap. This does not implement runtime compaction. |
| Recorded-run coverage | Every selected recorded assistant output across segments/subagents; preserve complete initial frames, raw channels and existing incomplete tails. No synthetic turn splitting or continuation truncation during conversion. |
| Oversized recorded examples | Explicit error, with no shortening or silent dropping; this should be exceptional. The existing 72k guard and actual model capacities still apply. |
| Score selection | `score_1`, `score_1_or_unscored`, `all`; default KL `all`, SFT `score_1`. Unit default weights; finite nonnegative custom weights. Root selection/weight inherited by included children/segments. |
| Optional shared base | The same registered LoadedModel may serve side and reader with distinct LoRAs. Separate bases remain supported. The agent adapter equals the AC reader adapter. |
| Teacher behavior | Same current reader/adapter without gradient; retain the optional per-invocation first-visit top-k KL cache. SFT has no teacher requirement or old-policy ratio claim. |
| Scope | Minimal additions to current architecture. Target-key redesign and legacy conversion deferred. Interfaces precede detailed planning; the requested subagent confirms details afterward. No campaign, DDP or long experiment resumes. |

## Milestones

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">Next — Details, reviews, implementation and handoff</span>
    <span class="card-oneliner">The paired interface plan contains fields, signatures, ownership and data flow.</span>
    <span class="card-badge">TBD</span>
  </summary>

Use [CODESIGN_INTERFACE_PLAN.tressoir.md](./CODESIGN_INTERFACE_PLAN.tressoir.md). It incorporates the accepted intent and the request to prefer intuitive, non-intrusive changes. Interfaces and complete test proposals are accepted. The next green light is intended to cover details, the requested independent review, implementation, re-review and a handoff report in one pass; material findings that need a user decision return to the user.

</details>
