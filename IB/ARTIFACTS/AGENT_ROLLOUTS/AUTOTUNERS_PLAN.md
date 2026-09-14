# Autotuners: a small shared config service

Implementation plan based on the note in your source `activation/autotuners/__init__.py`. Trainers ask for needed configs; this agent-owned package handles finding, tuning, validating and reusing them. This approved extension is implemented, reviewed and included in the slice 3a1 handoff. The completion reports below distinguish measured results from conservative policies.

## Executive Summary

**Recommendation: task-specific getters over one small shared config service.** A trainer passes existing model/training config objects and receives a typed dictionary of execution settings. Autotuners derive the workload, detect the environment and build compatibility/cache keys internally. Keep trainer changes to a config request and ordinary configuration application. No trainer objective callbacks, parameter changes, probe scripts or cache management. Backend internals and their interfaces remain agent-owned, as requested.

| Concern | Recommended rule |
| --- | --- |
| Caller | Call a task-specific getter with existing model/training configs. Optional overrides cover constraints it cannot infer, such as selected device, adapter settings or reserved memory. The trainer does not construct kernel keys or environment fingerprints. |
| Reuse | Look in the sync-shared cache first, then committed presets; validate compatibility in both. A compatible hit does no benchmarking. |
| Internal cache identity | Autotuners detect hardware and relevant software versions, inspect model geometry, and derive kernel/candidate revisions and semantic flags. Keep that bookkeeping internal. Never key on a node name, model weights or dataset text. Memory-policy results additionally declare their workload and memory-budget assumptions. |
| Fast tuning | Target **about 60 seconds for the whole cold request**. Autotuners may extend to **180–300 seconds when justified** by cold compilation, useful coverage or correctness validation. Start with a few legal candidates/length buckets; record the reason for an extension and retain completed valid results. |
| Isolation | A spawned worker owns synthetic inputs and tuning state. Prefer calling before training-model placement; the service must not evict caller models or touch their weights, gradients, optimizer or RNG. |
| Miss or timeout | Use a validated compatible preset/fallback. If none exists, return an explicit unsupported/pending result; never silently start an unlimited sweep or invent a config. |
| Persistence | Small immutable result bundles under `resolve_path("AUTOTUNE")`; stable validated snapshots under `activation/autotuners/configs/` in version control. Native compiled binaries stay in their normal local caches. |

The task-specific facade derives an internal workload description; `autotuners` owns environment detection, workers and the config lifecycle. The trainer consumes settings in its own vocabulary. Illustrative interface:

`activation/autotuners/__init__.py · get_agent_training_autotuning_variables()`

```python
variables = get_agent_training_autotuning_variables(
    model_config,
    training_config,
)
# TypedDict fields could include:
# fla_config, gradient_checkpointing_min_tokens, logits_chunk_tokens
```

Pass the existing training config to provide context limits and selected execution modes without a long argument list. For memory-sensitive decisions, include existing adapter configuration or an explicit memory budget when it cannot be inferred; architecture alone cannot describe all co-resident models. An AC-specific getter accepts the side/target model configs and existing AC/training settings. Both getters use the same internal backends and cache. A future inference getter can follow the same pattern.

Use a TypedDict (or similarly small typed result) so callers get named settings without a loose string convention. Keep compatibility/coverage and provenance in result metadata or logs; distinguish measured, preset and heuristic values. Explicit user settings take precedence; an auto/None setting makes resolution intent clear. The service must not silently overwrite fixed training options. Internal workload descriptions contain metadata, never live training tensors or callables. No GPU work runs at import.

### What merits tuning now?

| Area | Current evidence in source | Recommendation |
| --- | --- | --- |
| FLA kernels | Existing tuning/export helper and ten committed configs; repeated tuning stalls were measured. | First bounded-search backend. |
| Activation checkpoint threshold | Shared helper uses 6,144 tokens on cards with at least 80 GB, otherwise always recomputes. | First additional policy to formalize; resolve by model/training/memory profile rather than card capacity alone. |
| Logits chunk size | Both trainers use 2,048 positions per output-head chunk. | Resolve alongside checkpointing; use presets first and a short memory/throughput probe when worthwhile. |
| Micro-batching, packing and attention options | Token budget, pad-free/packed modes, length rounding and FlexAttention block options already have hand-measured defaults. | Later targeted search; preserve current measured defaults and effective optimizer batches/loss semantics. |
| Inference engine settings | Model-specific vLLM memory/concurrency/speculation recommendations already exist. | Later separate task getter; serving and training have different workloads. |

FLA is the demonstrated kernel-tuning issue, not the only performance configuration worth centralizing. These other fields are candidates, not newly measured regressions. First delivery should expose agent/AC task getters resolving FLA plus checkpoint/logits policies; seed them with existing measurements or clearly labeled heuristics. An uncached full-model memory probe may not fit the short budget: reuse a conservative profile or report unresolved coverage, rather than claiming a kernel-only probe measured whole-trainer memory.

## Requested Decisions

**Recorded constraints:** task-specific getters take existing configs; hardware/software/cache identity is derived inside autotuners; non-invasive config requests from trainers; agent-owned internals/interfaces; shared reuse; small configs version-controlled; favor fast tuning over chasing the final 10–20% of performance. Target roughly 60 seconds, with user-authorized discretion to spend 3–5 times longer when needed. These come directly from your note and subsequent corrections. The user authorized implementation after this detailed plan receives review and any genuine findings are resolved. No additional approval is required for routine implementation or validation.

## Milestones

### M1 — Define task getters and migrate existing results (Completed)

Simple caller APIs; shared internal detection, cache and policy resolution.

#### Completion report

**What landed.** Typed Agent/AC getters accept existing configs and optional device/memory constraints. The shared core derives exact GPU/software/kernel/recipe identities, validates coverage and native candidate membership, and publishes immutable checksummed profiles atomically. Shared results precede committed presets. Explicit checkpoint/logits values win; config objects stay unchanged. The ten original JSON files moved byte-for-byte into `configs/legacy` as unvalidated seeds. Inspect/retune/promote commands and two validated RTX PRO 6000 profiles are staged for version control.

**Drifts and constraints.** Model metadata is read locally so hidden network retries cannot escape the tuning deadline. Harness registration already resolves metadata; the explicit retune CLI prepares it before tuning. The current adapter supports pinned FLA 0.5.2/Triton 3.7.1 Qwen3.5 BF16 gated-delta kernels; other linear recipes fail clearly. CPU/non-FLA calls return valid settings with kernel tuning not applicable. Visible heterogeneous GPU architectures are rejected because pinned FLA makes import-time architecture choices.

**Validation.** CPU fixtures passed explicit-setting preservation, unchanged caller state/RNG, schema/checksum/identity/coverage rejection, shared-before-preset lookup, concurrent immutable publication and a real watchdog termination. GPU validation covers both effective head geometries (16 and 32, K/V=128). Final committed profiles match the delivered code recipe and record full mode coverage and numerical errors. Complete file deltas are in [the 3a1 handoff](SLICE3A1_ROUND.tressoir.md).

**Original forward plan, retained for context.**

Keep this small: agent-training and AC-training getters over common environment/cache helpers, the first FLA backend and small checkpoint/logits policy resolvers. The facade reads model metadata internally; trainers do not package GPU/compiler/kernel identities or construct a generic WorkloadSpec. Only already-known constraints unavailable from model/training configs need optional input. Move the existing ten FLA JSON files (about 16 KB total) from `agent_training/fla_configs/` into the new package. Retain their provenance; missing legacy compatibility metadata means “seed to validate,” not a newly certified result.

Use validated shared records before committed presets, then tune missing coverage. Fingerprint relevant dependencies rather than the entire lockfile; record driver details for diagnosis without invalidating on irrelevant package changes. Atomic publication and a manifest/payload hash prevent partial records from becoming hits. Publish each completed result separately so timeout preserves useful work. Immutable run/result paths avoid two syncing machines overwriting one mutable cache; a local lock suppresses duplicate work on one host.

Provide a small inspect/retune/promote entry point. Promotion writes deterministic, compact JSON into the source handoff for version control; it is an agent action, without a new human review ceremony for individual tuning values or an automatic Git commit. Training never writes into the source tree.

#### Planning Overview

Expose typed task getters, keeping optional device/resource constraints high level. Cache identity comes from the selected GPU, relevant installed package versions, actual kernel source and candidate declarations, geometry/dtype, and recipe revision. A successful CUDA result means every required FLA mode for that request has a complete validated config bundle; CPU/non-FLA models receive an explicit not-applicable kernel result and valid execution settings. Unsupported environments or exhausted cold searches fail before training rather than returning an unvalidated config.

Store one immutable, checksummed JSON result per completed profile under the shared AUTOTUNE root. Probe progress/staging is separate and never eligible for a hit. Publish by same-directory temporary file plus rename. A compatible cache hit validates schema, payload digest, exact identity, required coverage and current legal candidate membership. Recheck after taking a per-profile local lock; another process may have completed it. Atomic immutable files make cross-machine duplicate tuning harmless. Small validated results can be promoted to committed presets; old GPU-only JSONs remain candidate seeds until verified under the new identity.

#### Planned Changes

`activation/autotuners/__init__.py · task getters and typed results`

```python
def get_agent_training_autotuning_variables(
    model_config: ModelConfig,
    training_config: AgentTrainingConfig,
    *, device: torch.device | None = None,
    memory_budget_bytes: int | None = None,
) -> AgentTrainingAutotuningVariables: ...

def get_ac_training_autotuning_variables(
    side_model_config: ModelConfig,
    target_model_config: ModelConfig,
    training_config: ActivationContextTrainingConfig,
    *, ac_config: ActivationContextModelConfig | None = None,
    device: torch.device | None = None,
    memory_budget_bytes: int | None = None,
) -> ACTrainingAutotuningVariables: ...
```

The result carries `fla_config`, `logits_chunk_tokens`, resolved checkpoint threshold(s), and concise provenance/timing. Types stay in this agent-owned package. Internal models/configs are inspected without loading a caller's weights or importing model code with broader trust than its existing ModelConfig allows.

`activation/autotuners/core.py · cache and execution-policy resolution`

```python
# Internal flow, with serialization and error details omitted.
identity = detect_environment_and_describe_models(model_configs, device)
result = read_compatible_shared_or_committed_result(identity, coverage)
if result is None:
    result = run_isolated_tuning(identity, coverage, flexible_budget)
validate_complete_result(result, identity, coverage)
return resolve_execution_variables(result, training_config, memory_budget)
```

Policy resolution preserves explicit non-None values, including threshold 0. `logits_chunk_tokens=None` selects auto while explicit positive sizes remain exact. Use small, labeled preset/heuristic policies first: positive no-checkpoint thresholds require a compatible full-model memory profile and headroom; otherwise use recomputation. Do not generalize the old single-model 6,144-token measurement to every model or to AC's combined residency. Resolve a conservative logits chunk from vocabulary/dtype and memory allowance; do not claim kernel-only measurements prove full-trainer capacity. No learning-rate, optimizer-batch or AC-ratio changes.

`activation/autotuners/__main__.py · inspect, retune, promote`

```text
python -m activation.autotuners inspect
python -m activation.autotuners retune <model-id>
python -m activation.autotuners promote <validated-result>
```

CLI argument details and module-private helpers omitted; promotion validates identity/payload/coverage and writes stable JSON without automatically committing. Canonical configs migrate into this package; old main-tree tuning entry points become narrow compatibility exports or are removed after their active callers migrate.


### M2 — Implement bounded tuning and execution policies (Completed)

FLA search first; conservative checkpoint/logits resolution with explicit provenance.

#### Completion report

**What landed.** A subprocess establishes complete validated baselines before a capped optional search, with a roughly 60-second target and 300-second parent watchdog. Eleven explicitly inventoried native autotuners retain their Config objects, pruning and reset hooks. The runtime scope bypasses FLA fuzzy/default loading, seeds exact native keys on every call, and rejects uncovered semantics before native benchmarking. Only inspected independent-row/batch axes permit reuse. Automatic checkpoint threshold is 0 until a compatible whole-model profile exists; automatic logits sizes bound vocabulary scratch against free/explicitly reserved memory. Both policies disclose their heuristic provenance.

**Drifts and fixes.** Independent review caught and resolved cross-thread autograd locking, recovery from a failed first legal candidate, and metadata network retries. Scope ownership and callback locks are separate; baseline discovery tries at most four legal candidates on native recoverable launch/compile failures. The optional search receives short native timing trials, then repeats numerical and long-sequence checks before publication. No whole-model throughput search or measured percentage of the global optimum is claimed.

**Validation.** A fresh physical RTX PRO 6000 compiled and validated both geometries in 60.8 seconds (before the final additional post-search long check); the final revision separately passed both geometry workers and public trainers. Forward/backward reference errors for the final profiles are below 0.005 relative RMS; dense, padded/batched, packed and final-state prefill modes pass. Long checks cover 2,048/8,192/32,768 tokens; fresh runtime checks also cover a nonmultiple 65,539 tokens. Private fixtures prove unknown-key rejection, callback-thread execution, state restoration and bounded alternate-candidate recovery. Exact timings, revision limits and profile hashes are in [the validation record](slice3a1/autotuner_validation_summary.json).

**Original forward plan, retained for context.**

Use native FLA/Triton kernels and their current legal candidate sets. Warm candidates before timing, separate startup/JIT from measured execution, check outputs and gradients, and preserve native reset/restore hooks. The 80–90% preference is a search-effort goal, not a claim that a tiny search proves a percentage of the global optimum.

Resolve checkpoint and logits settings from compatible existing profiles first. Any fresh memory/performance probe must model the relevant training path, adapter/gradient mode and concurrent side/target residency; isolated kernel timings cannot establish full-model headroom. Keep these probes owned by autotuners and within their budget. Label conservative rules as heuristics, reserve memory headroom, and preserve explicit overrides. Broader batching/attention and inference searches remain later work.

Choose a few representative length buckets from the requested workload; keep head dimensions, dtypes, layouts, gradient direction and semantic flags exact. Allow reuse across only declared, validated length ranges. Reject old configs outside the current pruned candidate set, including hardware-specific restrictions.

The time allowance covers the whole request, including startup/compile; report those phases separately. Treat 60 seconds as the fast-path target. The autotuner can select or extend an allowance up to 180–300 seconds when needed to finish useful cold compilation, coverage or correctness checks; it records why and does not request permission for that already-authorized discretion. Avoid automatically spending five minutes on every miss. A parent watchdog enforces the selected allowance and reaps the worker on expiry. An in-process timer can only limit new trial admission, so it must not be sold as a hard deadline. Publish only completed validated results; an all-invalid search produces an explicit failure.

**Do not promise bounded runtime from JSON alone.** Stock FLA strict/fuzzy cache misses can invoke a full native sweep. For the explicitly supported kernel set, keep the small compatibility/application bridge inside the FLA backend: supply a compatible validated fallback or report uncovered keys before that branch. Avoid a global replacement of `triton.autotune`; do not fork vendor kernels. Unsupported kernels remain explicit coverage gaps, not hidden guarantees.

#### Planning Overview

The first measured backend targets the pinned Qwen3.5 gated-delta path on CUDA. Derive effective geometry from model metadata: Qwen repeats query/key heads before FLA, so effective H equals the number of value heads (16 for the current side, 32 for target), with K/V=128. The synthetic worker covers ordinary forward/backward, inference prefill with a final state (used by greedy samples), and packed variable-length mode when requested. Side batching and padding are covered explicitly; the trainer does not enumerate kernel internals.

Keep an explicit inventory of the reached autotuner objects, including single-config kernels and ordinary Triton variants where applicable. Import/unwrap only those objects. An instance-local runtime guard resolves a validated record, intersects it with the current kernel's legal/pruned Config objects, seeds the native key, then calls native Autotuner.run. This bypasses FLA fuzzy/default globals and preserves initialized native hooks. Restore altered methods/caches in finally. Side/target bundles are unioned within one scope; supplement native keys with semantic flags omitted upstream and reseed on each call.

Only documented performance axes may reuse a config: l2norm's block-count NB, and batch count for a specifically verified batch-independent kernel. All head widths/counts, dtypes, layouts, initial/final-state and variable-length flags stay exact. Validate these reuse rules against several lengths/batch counts before enabling them. Missing semantic coverage raises before native autotuning; it never starts an unrestricted sweep on training tensors.

#### Planned Changes

`activation/autotuners/fla.py · inventory, legal-config selection and application scope`

```python
def configure_fla_runtime(bundle: FlaConfigBundle | None) -> ContextManager[None]: ...

# Sketch inside the known-kernel guard; locking/cleanup omitted.
key = native_key_and_semantic_flags(tuner, args, kwargs)
selection = bundle.lookup(key)
original_config = match_current_pruned_config(tuner, selection, args, kwargs)
tuner.cache[key.native] = original_config
return Autotuner.run(tuner, *args, **kwargs)
```

Do not globally replace triton.autotune, reconstruct Configs with lost hooks, retain a native cache from different hardware, or let a later FLA fuzzy lookup override the selected result. Instance/scope locking and finally cleanup preserve unrelated caller state. Upstream module/recipe drift invalidates the bundle and produces a clear unsupported error if the adapter cannot safely interpret it.

`activation/autotuners/worker.py · disposable synthetic validation and search`

```python
# Worker owns all tensors/RNG and uses native pruning/reset/restore hooks.
baseline = validate_one_legal_candidate_per_required_kernel_and_mode()
publish_complete_validated_baseline(baseline)
while budget_allows_useful_work():
    candidate = next_of_at_most_four_legal_candidates()
    validate_outputs_and_gradients(candidate)
    measure_after_warmup(candidate)
publish_best_complete_validated_result()
```

Use a short PyTorch reference for gated-delta outputs/gradients, then representative longer recipes and mode/batch transitions. Establish a complete usable baseline before spending budget on further optimization. Preserve original Config objects and native reset/restore hooks while benchmarking. Publish upgrades only after validation; failure/timeout can reuse an already completed result but never partial trial state. Start around 60 seconds and extend up to 180–300 when useful compilation/coverage requires it. Parent watchdog terminates/reaps the worker group; do not kill the caller's CUDA context. Shared results contain metadata/configs, not model weights, dataset text or compiled binaries.


### M3 — Connect both trainers and prove reuse (Completed)

Small call-site changes and one representative GPU check.

#### Completion report

**What landed.** AgentTrainer requests settings before model placement and applies the native scope before old-log-probability recomputation, through training/backward/reporting. AC train and public eval each obtain side/target settings and scope all preparation, recursive/batched execution, exact evaluations and greedy samples. Cleanup restores native state. Both configs support automatic logits sizes and both trainers expose resolved-setting provenance. The old fuzzy helper and obsolete generic checkpoint policy were removed after active IB profiling callers migrated.

**Drifts.** The retained GPU could not restart because of AWS capacity. Its restart request was canceled and it remains stopped with its disk intact. Validation used one temporary RTX PRO 6000 in Tokyo, with scoped artifact transfers; a broad 39 GB SYNC upload was canceled before submission. The temporary node was torn down after final scoped synchronization. This extension adds no public probe/test files; the two original 3a1 benchmark/profile tools remain IB-only.

**Validation.** Actual tiny CPU AC lifecycle passed 22 updates, teacher-cache reuse, persistent optimizer and exact paired reload; actual CPU Agent training passed. On the GPU, actual Qwen4B Agent dense/packed calls and side0.8B/target4B AC eval-before-train, one recursive/batched optimizer step, generated samples and eval-after all passed with finite metrics and no dropped examples. Separate fresh processes used shared and committed-preset hits with worker/native benchmark hooks set to raise, proving zero tuning on covered runtime calls. Earlier 3a1 capacity/quality evidence is retained; the larger real-data quality suite was not rerun for this execution-config change. Independent code and final handoff reviews are linked from [the handoff](SLICE3A1_ROUND.tressoir.md).

**Original forward plan, retained for context.**

Replace imports/calls from `agent_training.fla_cache` in AgentTrainer and ActivationContextTrainer with their task-specific getters; preserve a short compatibility shim while IB profiling callers migrate. Callers pass existing config objects and apply returned execution variables, respecting explicit overrides. They do not enumerate kernel shapes, discover software versions, implement searches or pass their objectives into the tuner. No training-loop redesign or automatic tuning of semantic settings such as AC compression ratio or learning rate.

Validate cache hit, invalidation, corrupt/partial records, concurrent publication, timeout and no-CUDA paths with private focused fixtures. On one GPU, demonstrate a cold request followed by a fresh-process/shared-cache hit with zero benchmarking for the same covered workload. Check forward/backward correctness, unchanged caller state and covered-key miss behavior. Verify the sync/export path and that another compatible node can select the same config; JIT compilation may still occur there. Keep probes in IB; publish config provenance, coverage, tuning seconds and hit/miss reasons.

#### Planning Overview

Both trainers resolve configs after their engine is asleep and preferably before loading training weights. They apply a narrow FLA configuration scope covering every model call, then restore it during existing cleanup. Calls still own model placement, parameters, optimizer and loss. Tune once per compatible profile; later calls and new compatible nodes reuse shared/preset results. The AC public eval path resolves and applies the same configs, so eval-first and post-training samples cannot fall into a separate untuned path.

#### Planned Changes

`activation/agent_training/agent_trainer.py · train() and head chunking`

```diff
- configure_fla_cache(device)
- min_tokens = checkpointing_min_tokens(config, device)
+ variables = get_agent_training_autotuning_variables(
+     loaded_model.model_config, config, device=training_device)
+ kernel_scope = configure_fla_runtime(variables['fla_config'])
+ min_tokens = variables['gradient_checkpointing_min_tokens']
+ self._resolved_logits_chunk_tokens = variables['logits_chunk_tokens']
+ # Apply scope before reference recomputation/forward/backward/reporting.
+ # Close it in finally; retain optimizer and model cleanup.
```

`activation/ac_model/ac_model_training.py · train(), eval(), KL head chunks`

```diff
- self._configure_kernels(device)
- min_tokens = checkpointing_min_tokens(config, device)
+ variables = get_ac_training_autotuning_variables(
+     ac_model.side.model_config, ac_model.target.model_config, config,
+     ac_config=ac_model.config, device=training_device)
+ kernel_scope = configure_fla_runtime(variables['fla_config'])
+ min_tokens = variables['target_gradient_checkpointing_min_tokens']
+ side_min_tokens = variables['side_gradient_checkpointing_min_tokens']
+ self._resolved_logits_chunk_tokens = variables['logits_chunk_tokens']
+ # Scope covers exact eval, training/backward and greedy inspection samples.
```

`activation/agent_training/agent_training_config.py · auto settings and stats`

```diff
- logits_chunk_tokens: int = 2_048
+ logits_chunk_tokens: int | None = None  # auto; positive explicit values win
+ # Stats retain resolved settings, result source and tuning seconds.
```

The AC config gets the same optional logits setting and resolved-setting stats. Explicit checkpoint thresholds and gradient-checkpointing on/off flags retain their meaning. Private helpers/profilers that bypass train() need an explicit resolved chunk size or a compatible conservative default; public train/eval always use the getter. Existing config objects remain unchanged.

`activation/agent_training/fla_cache.py; IB-only ac_training_profile.py / dump callers`

```diff
- # Caller-owned/manual dump-and-copy tuning path.
+ # Active callers use task getters and shared result publication/promotion.
+ # Retain only necessary compatibility exports; no second cache policy.
```

Moderate sketches above omit setup/cleanup placement and repetitive imports; final handoff will show complete per-file diffs against current source. No new public probe/test files are required.

Validation: CPU fixtures cover warm hits with zero worker/benchmark calls, corrupt/incompatible/partial cache misses, merge publication, timeout/reaping, CPU/non-FLA behavior, explicit overrides, device identity and no mutation of caller state. Native bridge fixtures check legal Config/hooks, side/target/packed transitions, supplemental flags and unknown-key rejection. On one GPU test cold-with-no-presets, fresh-process shared hit, committed-preset hit, invalidation, dense/packed/batched forward-backward correctness and no hidden runtime autotuning. Then run representative AgentTrainer and AC train/eval (including samples/recursive parts), recording returned settings and checking finite matching behavior. Use the existing node, sync results and pause it afterward. Retain earlier 3a1 evidence where unaffected; update the 3a1 patch, exact diff cards, completion reports and source/projection checks.


## Why this is the right first step

Today, the source `activation/autotuners/__init__.py` contains only the intended contract. Both trainers call `activation/agent_training/fla_cache.py`, which selects a directory primarily by GPU name and enables fuzzy loading when any JSON exists. Learning new configs and copying them back still depends on IB dump/profile scripts.

Inspection of pinned `fla-core`/`flash-linear-attention` 0.5.2 with Triton 3.7.1 found two reasons to formalize compatibility: numeric fuzzy matching can span more than length, and the JSON `triton_version` field is not enforced during lookup. These are reuse risks, not evidence that the currently measured configs produced wrong results.

Native facilities should do the kernel work: Triton exposes candidate pruning and reset/restore hooks, while FLA exposes config-cache modes. The backend should contain the glue around them. Upstream documentation can differ from this repository’s pinned release; verify the installed path before relying on newer disk-cache environment flags. [Triton autotuning API](https://triton-lang.org/main/python-api/generated/triton.autotune.html), [FLA cache controls](https://github.com/fla-org/flash-linear-attention/blob/main/ENVs.md).

This remains a small module and file-based cache. A distributed tuning service, database, universal search framework and automatic training-hyperparameter search would add scope without solving the current problem.

**Review and validation:** source note, both trainer call sites, current checkpoint/logits/batching/inference defaults, existing 16 KB presets, dump/profile paths and the pinned FLA implementation were inspected. An earlier independent trainer/backend review informed the worker boundary, budget, native-hook preservation and explicit miss handling; the subsequent task-specific interface correction is recorded above and the revised paired text is mechanically checked. The detailed plan received an independent pass with no blocking findings or new user decisions. One cheap API wording inconsistency was corrected. Implementation and validation are complete; see the completion reports and shared handoff. L2norm NB reuse is valid across positive native-supported row counts because NB is unused in the pinned kernel bodies; performance beyond measured lengths remains unmeasured. Dense cumsum batch reuse is similarly limited to independent batch replication, with all other dimensions/flags exact. Full validation results and actual implementation drifts are recorded in the completion reports.
