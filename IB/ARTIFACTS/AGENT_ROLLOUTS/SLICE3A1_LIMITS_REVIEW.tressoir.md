# Slice 3a1 · avoidable limits review

Fresh review of the revised AC design, datasets and agent training, after integrating your interactions. The strongest findings are concrete execution/data defects and an AC initialization risk. This report identifies corrections and experiments; it does not add unapproved implementation work.

**Overall verdict: `incorrect-or-missing`.** The accepted architecture is achievable. The reviewed plan still needs the dispositions below before it can support a claim that the learning path is sound. The earlier version-1 pass does not cover these findings.

## What changed in the plan

Stride **0** is the content-pooling default. A separate learned pooling path makes **V content-derived summary rows**, then adds **one shared marker vector across those rows plus positional encoding**. There is no extra marker position. Real subagent exchanges are deferred to a later slice. Epoch AC/LoRA checkpoints are eligible for regular incremental sync; healthy transfers have no default wall-time cutoff, stalled transfers retain partial progress and retry, and remaining unsynced data is reported.

The replan was written before the three new domain reviews ran. [Revised plan](SLICE3A1_PLAN.tressoir.md). Product source and cloud resources were not changed.

### Scope after the review discussion

The broad review identified possible limitations across several slices. It did not establish that all thirteen are prerequisites for 3a1. Following the user's scope correction, **prioritize AC initialization and source-data fidelity in 3a1**.

| Issue from the chat summary | 3a1 disposition |
| --- | --- |
| (1) Fresh-agent compaction-tool constructor failure — A1 | Defer to 3b with compaction/agent integration. The AC training plan does not require a fresh agent rollout. |
| (2) AC input initialization/scale — C1 | Fix as part of M2: preserve content relative to PE and random residual updates, then align content/summary inputs, shared marker and PE to the side decoder. Final RMS normalization alone is insufficient. |
| (3) Dataset text/trajectory corruption — D1, D2 | Fix in M1: preserve original continuation whitespace/token boundaries and assemble trajectory windows from original spans once, preserving turns and calls. Regenerate affected item caches. |
| (4) Mixed missing policy log-probs — A2 | Defer as an ingestion edge-case guard. Expert/hinted records already use `ignore_logprobs=True`; wholly unrecorded records also request reference recomputation. No normal producer of mixed recorded/unrecorded policy turns was identified. |
| (5) Seen-context QA evaluation — D4 | Keep the current question-disjoint split and its honest label. The user considers unseen-context evaluation low priority; no corpus repartitioning in 3a1. |

Keep the small M6 item-cache settings compatibility check with the data corrections, so old data or changed generation settings cannot silently invalidate their verification. Include full recursive backward in the already-planned capacity check. Broader adapter/sampling/action-window experiments and other agent-training review items remain later work, rather than new 3a1 prerequisites.

**Clarification of (4):** `items_for()` marks non-policy records with `ignore_logprobs=True`; `build_example()` then requests `_fill_old_logprobs()` before the agent trainer's first update. Those are the **student's round-start reference probabilities**, not probabilities required from the expert generator. The reproduced bug requires a non-ignored policy record with at least one recorded assistant turn and another unrecorded turn. A normal rollout requests recording consistently through its run-level `record_sampling` setting. AC distillation obtains teacher distributions through its own forward passes and does not depend on stored rollout log-probs. The previous chat summary overstated this edge case's current priority.

This records planning scope and recommendations; product implementation has not started.

## Broad review findings (original evidence)

“Confirmed” means source behavior or a small fixture establishes the fact; it does not imply measured learning harm. Priority reflects when the issue becomes relevant, not an estimate of how often it occurs.

| ID | Area | What can hold us back | Evidence | Smallest proposed response |
| --- | --- | --- | --- | --- |
| A1 | Agent execution | Fresh default agents fail before sampling | Confirmed constructor `TypeError` from unfinished `CompactionTool` | Remove its default registration until the later compaction slice; prove one uncached path works |
| C1 | AC initialization | Position/random residuals dominate content entering the side model | Actual mixer fixture: 43.5× token RMS; changing all token embeddings leaves cosine 0.99938 | Preserve the content signal at initialization, then align both branches, marker and PE to side-input scale |
| D1 | Continuation data | Mid-turn cuts alter recorded text and can alter token boundaries | `alpha` + recorded ` beta` becomes `alphabeta` after `.strip()` | Preserve whitespace and original target-token boundaries |
| D2 | Trajectory data | Window assembly duplicates text and creates extra turns | One 80-character message becomes 96 characters in three messages | Select source spans once, preserving original turns and calls; account for oversized indivisible calls |
| D5 | Experiment validity | A changed ratio, cut or study model can silently reuse old items | Cache keyed by kind/seed and accepted by existence | Enforce the proposed settings metadata before deciding whether generation is needed |
| A2 | Agent gradients | Mixed recorded/missing log-probs use false old probabilities | Missing turn gets old log-prob 0; negative gradient is exactly zero in fixture | Reject incomplete policy records or use an explicitly valid old-policy fallback |
| A4 | Agent credit | Engine/scorer failures are treated as wrong answers | Partial error run receives −0.5 and changes its group's baseline | Exclude invalid execution/scoring before computing group advantages; retain legitimate task failures |
| A5 | Agent budgets | Forced generation cutoffs look like voluntary no-tool turns | Three `length` stops become `no_tool_call`; cause is absent from steps | Preserve stop reasons and choose explicit continuation/truncation semantics |
| D3 | Supervision coverage | A 512-token prefix can never reach a late tool call or answer | Fixture call begins at token 1,801; only 43/1,100 cached Nemotron targets are complete turns | Keep the cap; decide whether to reserve a small share of action/end windows |
| D4 | Evaluation scope | Held QA often uses familiar trajectories | 135/200 held gold trajectories also supply training gold questions; 172/200 appear anywhere in train inputs | Label new-question/seen-context evaluation; split source pools before generation if unseen-context evidence is wanted |
| C2 | AC adaptation | Gated-delta token-mixing projections do not adapt | No LoRA wrappers in those projections; they occur in 18/24 side and 24/32 target layers | Make coverage explicit; compare one bounded side-adapter expansion before adopting it |
| C3 | AC capacity | A standalone encode can fit while full recursive training fails | Graphs from child/parent and student coexist; new stride-0 cost is unmeasured | Include one full depth-2 backward/step at the intended retained limit |
| A3 | External teacher data | Another tokenizer's IDs can be replayed into the student | Producer mismatch is accepted; recorded IDs are copied unchanged | Validate tokenization compatibility; reject unsupported records, defer conversion |

## Recommended disposition

**Current 3a1 corrections to propose:** C1, D1, D2 and the small D5 cache check. A1 belongs to 3b under the user's scope correction. Resolve C1's input-scale policy as part of implementing the already accepted summary architecture. They do not require a new trainer or a different teacher objective.

**Later agent work:** A2 is a mixed-record ingestion edge case with no normal producer identified, not an expert-data blocker; A4 and A5 also remain later agent-training items. A3 becomes a guard at the external-teacher ingestion boundary; cross-tokenizer conversion itself stays deferred. These are inherited agent issues surfaced by the requested broader review, and are not silently added to the AC implementation milestones.

**Explicit choices or small checks:** D3's action coverage, D4's generalization target, C2's adapter families and C3's full recursive capacity. No evidence supports immediately expanding every LoRA family, removing all caps, replacing the teacher, or launching a hyperparameter sweep.

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">AC design and training evidence</span>
    <span class="card-oneliner">Separate initialization facts, adaptation hypotheses and memory checks.</span>
    <span class="card-badge">3 findings</span>
  </summary>

### C1 — Preserve content before aligning side-input scale

`activation/ac_model/ac_model.py:327–363` reads side embeddings/recursive rows, adds a unit-amplitude sinusoid, applies ordinary random residual mixers, and supplies their outputs to the pretrained side decoder. The final target/recursive heads normalize their own outputs; they do not settle this earlier input path. `ac_model_utils.py:110–183` supplies positional RMS about 0.707 and ungated attention/FFN residuals.

The CPU fixture uses the actual source mixer at width 1024, two layers and 256 rows, with **synthetic** embeddings at the previously measured embedding RMS 0.019. Mixed-row RMS is 0.8275; replacing every embedding gives mean output cosine 0.99938. This demonstrates initialization dominated by position/random branches in that fixture, not a trained semantic failure. Scaling only PE leaves RMS 0.4603: the residual branches also need attention. Normalizing the final mixture cannot recover content already overwhelmed upstream.

The bounded correction needs two invariants: preserve content relative to PE/residuals, and align the resulting inputs to the side decoder. Consider small residual gains or near-identity output initialization; the exact gains remain unmeasured. Include ordinary and recursive inputs, both pooling branches, the shared marker and summary PE, so a late addition cannot undo alignment. Measure per-component RMS and content sensitivity, then finite gradients and learning in the planned short node check. Preserve the accepted V-row shape and shared marker.

### C2 — Decide hybrid LoRA coverage deliberately

`activation/harness/module_manager.py:39–40` calls the seven ordinary attention/MLP suffixes full coverage. Actual PEFT wrapping omits Qwen3.5's gated-delta `in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b` and `out_proj`. These blocks occupy three quarters of side/target layers. Their **MLPs still adapt**; the whole layers are not frozen.

Meta-tensor enumeration using cached production model dimensions gives this cost boundary:

| Adapter | Current LoRA parameters | Additional parameters for all five omitted families | Increase |
| --- | ---: | ---: | ---: |
| 0.8B side, rank 128 | 51,118,080 | 35,463,168 | 69% |
| 4B target, rank 64 | 84,934,656 | 44,924,928 | 53% |

These are parameter counts, not measured throughput or peak-memory increases. Current coverage is a plausible capacity limit, not a demonstrated reason learning cannot work. Record the selected families in configuration/checkpoint metadata. A small comparison of current coverage versus selected **side** gated-delta projections is a reasonable first test, holding data, steps, scale policy and rank fixed. A target change also requires serving/exchange compatibility. Do not change every model's global defaults or assign large ranks blindly to tiny gate outputs.

### C3 — Validate the whole recursive training shape

`ac_model.py:282,331` retains differentiable child outputs in their parent; `ac_model_training.py:268` backpropagates after the student loss is formed. Child and parent activations can coexist. Local mixers currently lack the base decoder's checkpoint wrapper. Stride 0 increases decoder rows but also removes content-pooler activations, so arithmetic alone does not prove an OOM.

Strengthen one already-planned capacity check: complete a depth-2 example at the intended largest retained shape, with a legal long visible suffix, actual checkpointing policy, both adapters, backward and optimizer step. Record each part's content/summary lengths and peak allocation. Standalone 8k/16k/32k encoder checks cannot validate the old whole-example cap. Also batch adaptive summary bins rather than launching one small GPU operation per summary row; this is an implementation clarification, not another framework.

</details>

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">Dataset and evaluation evidence</span>
    <span class="card-oneliner">Repair source fidelity and make supervision and held-out claims explicit.</span>
    <span class="card-badge">5 findings</span>
  </summary>

### D1 and D2 — Preserve the source text and conversation

`activation/ac_model/ac_model_study.py:177–218` cuts assistant text at character offsets and strips the remainder; `ac_model_training.py:170–176` tokenizes the partial/completion independently. The exact-method fixture reproduces lost inter-word whitespace. Preserve whitespace, use stripping only for emptiness validation, and retain a consistent original target-token split. Check a space, newline, within-word split and multibyte text. This is separate from the already-planned shared visible suffix correction.

`activation/dataset/dataset_index.py:41–80` creates overlapping retrieval chunks, while `ac_model_study.py:245–267` concatenates their `chunk_messages` into a trajectory window. The default overlap is 256 characters. The small 32/8 fixture proves duplicate text and invented message boundaries. Select spans from the original trajectory using provenance; do not remove useful overlap globally from retrieval or treat overlapping snippets as new turns.

The same assembly can exceed its advertised window budget: a long structured call is carried intact, and the central chunk is admitted before budget checking. A toy 32-token window admits 2,512 rendered tokens from a 20,000-character call. This proves the missing bound, **not its production frequency**. Account for indivisible calls explicitly and report/handle oversize cases; do not silently split JSON/tool arguments or silently claim the requested budget was met.

### D3 — Prefix continuation does not guarantee action recovery

`ac_model_study.py:84–89,210–218` renders reasoning before tool calls and takes the first 512 completion tokens. Moving to a later assistant turn still misses a call near that turn's end. In the **previous generator's cached items**, Nemotron has 43 complete turns, 89 tool-call opens and 43 closes among 1,100 items; Open-SWE has 420 complete turns, 561 opens and 435 closes among 1,100. Truncated reasoning and partial calls remain legitimate continuation targets. These counts establish limited action/end coverage, not invalid loss.

If this phase should teach action/outcome recovery, reserve a small measured share of actual action/terminal windows within the same cap. Supply preceding within-turn text consistently to teacher and student, preserving causality. If local continuation alone is intended, label that narrower objective. Check coverage by source and immediate/delayed/action category before a long training run.

### D4 — Decide what held-out QA should establish

The staged bench's `ac_training_bench.py:106–137` groups QA by study-question ID; `activation/dataset/dataset_study.py:184–195` does not make that a document identity. M6 retains question grouping explicitly. Applying its source-balanced split to 2,039 already-synced trajectory-QA items gives 1,839 train / 200 held: 135 held gold trajectories also supply train gold questions, and 172 appear somewhere in training inputs. IDs are dataset-qualified, so this is not a naming collision.

New questions about seen contexts are a valid, narrower evaluation. Do not describe it as unseen-trajectory generalization. If the latter is needed, partition source-document pools **before** question generation and distractor retrieval; restrict each side's evidence candidates to its own pool. Existing IDs plus a small allowed-document filter suffice. Do not create a large connected-component split after random distractors have linked the corpus.

### D5 — Enforce cache settings before reuse

The staged bench's `ac_training_bench.py:51–52,315–333` keys item files by kind/seed and treats existence as reuse eligibility. A fresh 3a1 namespace avoids old data once, but later ratio/threshold/model changes can still alias. The proposed settings sidecar must be compared **before** deciding whether to start a study engine. Include generator/schema revision, source selection, tokenizer, ratio/depth/cut/suffix/completion settings and study model/prompt. A digest in this cache filename or an enforced sidecar is sufficient; mismatch should select a fresh path or fail with the differing fields, preserving old caches.

The gold-only QA teacher with distractors on the student side remains an intentional oracle-guided retrieval/compression objective. Label it accordingly; it is not an isolated compression-loss measurement. Before using extended snippets (disabled by default), verify the supporting span reaches both branches. No unconditional teacher redesign follows from either observation. A suspected current-default Qwen thinking-prefix mismatch was checked and disproved.

</details>

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">Agent execution and learning evidence</span>
    <span class="card-oneliner">Fix invalid inputs and credit semantics before tuning the learning rule.</span>
    <span class="card-badge">5 findings</span>
  </summary>

### A1 — Keep the unfinished tool out of default construction

`activation/agent/agent_tools.py:248–256` registers `CompactionTool` with empty kwargs and an inherited empty name, despite requiring `base_config`. `agent.py:60,74–77` constructs every default tool before the run's exception handler. `rollout_manager.py:76` also constructs the agent before its handler. The actual source method raises the missing-argument `TypeError` with a default config. Cached results can bypass this path. Remove that one registration until compaction is implemented later; use a model-free construction check and an explicitly uncached live validation when authorized.

### A2 — Never disguise missing log-probs as probability one

`activation/agent_training/agent_training_utils.py:49–68` marks an unrecorded assistant segment for loss with old log-probs 0, but requests recomputation only when no segment had recorded values. A two-turn fixture therefore scores old values `[-2, 0]` against current `[-2, -2]`: before any update the missing negative-weight token is already clipped and receives zero gradient. This is conditional on mixed recording; its live frequency is unknown.

Reject incomplete policy records or supply a coherent, explicitly valid old-policy snapshot under a documented fallback. A current-student rescore is not automatically the original behavior policy. Define partial-array alignment rather than guessing. Fully recorded and wholly unrecorded examples are separate supported cases.

### A3 — Validate external teacher token compatibility

`agent_training_selection.py:26–33` stamps the requested student model on teacher/hinted records; `agent_training_utils.py:40–62` copies their producer's token IDs unchanged. The tiny fixture proves a differently named producer is accepted; it does **not** prove current Qwen callers have incompatible tokenizers. Different mappings can nevertheless corrupt text silently when IDs remain in range.

Make tokenizer/vocabulary compatibility a validated precondition of exact segment replay; vocabulary size alone is insufficient. Preserve producer/tokenizer provenance, and reject unsupported records. Keep real on-policy segments exact. A separate cross-tokenizer conversion path can remain later work.

### A4 — Separate invalid execution from task failure

`agent.py:180–188,247–250` can leave a partial error run or turn scoring errors into score zero. `agent_training_selection.py:37–41` computes a baseline across all members without validity filtering. A source-method fixture gives one submitted success +0.5 and an engine-disconnected partial run −0.5; both are trainable. Thus an external failure penalizes preceding actions and changes other members' credit.

Mark execution/scoring validity and filter invalid members before group advantages; report and replenish the shortfall. Wrong answers, model-caused tool mistakes and declared task budgets can still legitimately fail. Do not exclude every timeout or tool error indiscriminately.

### A5 — Preserve why generation stopped

`loaded_model.py:40,60` exposes the engine stop reason, but `agent.py:127–154` discards it and `agent_config.py:91–102` has no step field for it. A source-loop fixture returning `length` three times receives two synthetic user nudges and finishes as `no_tool_call`. The observed record cannot distinguish interruption from voluntary ending.

Store/report the engine reason and choose explicitly between same-turn continuation and a declared truncation/budget result. This does not establish that the current 1,024/2,048-token limits are wrong. Measure length-stop rates and repeat a small affected set with a larger cap before changing budgets.

**Experiments, not defects:** training rollouts currently inherit temperature 0.7, top-p 0.8, top-k 20 and presence penalty 1.5. The recorded/trainer likelihoods are raw-model probabilities in the inspected implementation, so step-one ratio near one does not establish that exploration uses the same distribution. Compare one explicit broader-sampling candidate only if learning warrants it; report reward, informative-group yield and cost per generated token. Per-trajectory mean loss also gives long runs smaller per-token coefficients; that is an explicit reduction choice, not a sign bug. No new critic or objective replacement is justified by this review.

The existing agent tests measure mechanics and adaptation on the same sixteen tasks with fresh seeds. Their reporting split and cached runs do not establish held-task generalization or uncached source execution. This limit was already documented; it is not a demand to turn the smoke test into a benchmark.

</details>

## Solver classification and validation

I independently checked the source paths and fixture evidence. All **13 primary findings** are `genuine` as defects, explicit boundary requirements, or consequential decisions/checks. That classification does **not** promote C2, D3, D4 or budget/sampling alternatives into proven quality fixes. C3 is a missing validation shape, not a reproduced OOM. A3 is conditional on incompatible external inputs. The oversized-call fixture establishes behavior at a toy budget, not observed production prevalence.

Batching summary bins is a `cheap-nit` clarification of the existing module implementation. Blanket LoRA expansion, a new teacher/critic, retokenizing on-policy trajectories, whole-completion generation, a global cache framework, and a connected-component corpus splitter would be `reviewer-overkill` here. The suspected current-default tokenizer-prefix mismatch was rejected after a direct check. The genuine scale/coverage questions remain open; the version-1 deferral is not used as a veto.

Three independent domain reviews and small CPU fixtures completed. They exercised actual source helpers/methods with small stand-ins, an already-local tokenizer, real PEFT wrapping on a tiny Qwen model, cached model configurations via meta tensors, and counts from previously synced datasets. No pretrained-model learning, GPU capacity run, cloud operation or model download occurred. All assertions in the three probes passed. Product source was not edited.

Raw review reports and reproducible probes are in `IB/TMP/SLICE3A1/review2/`: `ac_limits_review.md`, `data_limits_review.md`, `agent_limits_review.md`, their `*_probe.py` scripts and JSON/JSONL outputs. The reviewed version-2 baseline and accepted-interaction snapshot are preserved there. The final plan/projection share all seven milestone bodies; the Markdown checkers are clean and the six reviewed prototype-file hashes are unchanged. Custom-editor visual verification is unavailable in this container.

The user subsequently narrowed 3a1 to the corrections above and deprioritized the seen-context evaluation concern. The broad review verdict is historical evidence, not a requirement to resolve all thirteen findings in 3a1. This completes the requested identification pass. The verdict remains `incorrect-or-missing` because these proposed follow-ups have not been resolved or implemented; it is not represented as a clean implementation review.
