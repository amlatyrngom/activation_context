# Harness/model co-design — interface plan source

Status: interfaces and full test proposals accepted in chat; combined next phase authorized on 2026-09-15. The user authorized product implementation after the independent detailed-plan review on 2026-09-15. The user authorized this interface pass and requested minimal changes to existing architecture. Keep the source and projection synchronized; the complete human-facing projection follows verbatim.

## Execution notes

- Read and incorporated both generations of CODESIGN_PLAN interactions and the latest chat: "Ok just err. Should generally not happen though." The same message authorizes interface work and asks to avoid new abstractions.
- Accepted storage revision: capture rows in current span dictionaries; save under saved_tensors beside the writer’s JSONL; serialize relative references and resolve them on load. Utilities own tensor/copy/traversal/path logic. The main Agent, result methods, cache, manager and reporter contain small hooks. Keep raw parts independent, model/lora buckets unchanged, and existing trainer ownership.
- Current as-built source lacks row persistence, whole-continuation item preparation, SFT and same-base support. These snippets are proposals, not applied code. Audit target precision support before claiming exact row replay across a different dtype.
- Required implementation-review focus: full source frame retained on text-run conversion; complete assistant coverage even before later AC parts; teacher/student target offset pairing; old teacher cache key must include positions; no SFT teacher through baseline/sample/report paths; shared-adapter recomputation and trainability; optimizer cache resets at phase changes.
- The full-run text conversion should reuse initial frame/channel part construction, replacing the old random cut path. It must never concatenate across reset histories or duplicate assistant targets copied into context. Concrete construction and installed-library behavior belong to the detailed plan, not an unannounced interface expansion.
- Preserve existing product edits, TASK and historical handoffs. The user explicitly authorizes only test_basic_agent_ac_training.py and test_basic_ac_self_distillation.py for this work’s mainline test delta; all other added/changed tests are private. Full proposed files are review artifacts until implementation is agreed. No corpus conversion, cloud work or training launch is authorized by this planning pass.
- After the user reviews interfaces, create the detailed plan and invoke the specifically requested subagent reviewer. Notify the user if implementation findings require a public-interface change. No subagent review is claimed for this document.
- The supplied tools do not expose the custom Tressoir editor; static artifact validation is available. Record validation honestly.

## Storage revision accepted in chat

The user questioned external row_path portability, considered inline tensors and their retention cost, then evaluated a bundle consisting of <folder>/x.jsonl and <folder>/saved_tensors/. They accepted the folder-local writer-owned direction and instructed: "Ok put as many of the complexity in utils as possible, to avoid polluting main implementation files with this logic. You can update the plan now."

This revision replaces the global AC_ROWS proposal and Agent-time disk writes. Persist before releasing the rollout concurrency slot; release actual retained result tensors only after successful JSONL append. Capturing, saving/rebasing, report conversion and loading belong in activation/agent/agent_utils.py. serialize/deserialize gain base_dir; paths are resolved in memory so consumers need no extra root field. The storage-only revision originally awaited review with fresh decision keys; the subsequent full-test revision was accepted in chat on 2026-09-15. There were no saved interface interaction answers when this revision began.

Recovery and focused review/validation: `IB/TMP/AGENT_AC_PRETRAINING/interface_storage_20260914T203905Z/`. Preserve the separate VLLM_ACTIVATION_INPUTS workstream. Required later checks include move/export portability, concurrent save ordering, failed append preservation, no raw tensors retained by result aliases after save, no payload copies in asdict/reporting, and correct disabled-cache/deadline behavior. The save helpers operate only on owned row fields, not arbitrary tool payloads with similarly named keys.

## Latest test-review instruction

The user wants the full updated activation/tests/test_basic_agent_ac_training.py and a complete new test_basic_ac_self_distillation.py before the next stage; these are the only mainline/non-private test files for this change. The new file must demonstrate a custom AgenticProgram on four simple HotpotQA problems and the full self-distillation pipeline with training. Mainline tests use real calls and data. Private tests may construct synthetic trajectories, mock boundaries and inject failures; their code need not be shown. The user wants agreement that behavior can be tested without heavy training. They also explicitly rejected a new utilities file: use the existing activation/agent/agent_utils.py for row helpers.

Full proposed test files and their paired review surface are now provided, while implementation awaits the next-phase green light. This is stronger than a future promise to show the files: both files are complete code against the proposed APIs. The updated public-data file retains its three cases; the new example exercises KL and SFT with four roots, actual search/subagents, one epoch/two updates, paired reload, LoRA exchange and AC row persistence/replay. SFT uses explicit all for execution coverage and real scores; default score_1 behavior is private. No broad dataset quality or on-policy improvement is asserted. Independent numerical private tests, not mocked training results, prove loss/gradient ownership. Preserve root selection and whole-run target coverage.

Recovery and validation for this pass: `IB/TMP/AGENT_AC_PRETRAINING/interface_test_review_20260914T205207Z/`. No product test file has been written during this proposal pass. The new v3 interface controls encompass the test review; do not treat earlier blank controls as approval.


## Interface acceptance and next-phase boundary — 2026-09-15

The user accepted the interfaces and both full test proposals in chat on 2026-09-15. Their next green light is intended to cover details → independent review → implementation → re-review → acknowledgement with handoff report in one continuous phase. Resolve ordinary technical findings within the agreed intent; return for input when a finding requires a user decision or scope/public-behavior change. The user then green-lit the complete phase and the narrow third-test compatibility migration: “Ok go for it.”

## Human-facing projection

# Harness/model co-design — interfaces

Extend the current records, training items, generators and trainers to implement the accepted training split. The user accepted these interfaces and the full test proposals on 2026-09-15. The user authorized the combined next phase on 2026-09-15.

## Executive Summary

**The agent trainer learns from the exact observation captured during rollout. The AC trainer rebuilds compressed inputs and trains the encoder and reader together.** Both keep their current model names, LoRA names, entry points and checkpoint ownership.

| Existing surface | Proposed change | Producer → consumer |
| --- | --- | --- |
| `AgentRunResult.prompt_ac_spans`, `TrajectoryStep.ac_spans` | Captured `rows` become a relative `row_path` under the JSONL folder’s `saved_tensors/` after save | Agent capture → writer + utilities → resume / AgentTrainer |
| Existing `serialize` / `deserialize` | Keyword-only `base_dir`; row conversion delegated to the existing agent utility module | JSONL writer/reader → record serialization |
| `ActivationContextTrainingItem` | Two message histories, a start index in each, and assistant token sequences replace the one-completion fields | Study generator / direct caller → ACTrainer |
| `ActivationContextTrainingConfig` | `loss_kind="kl"` or `"sft"` | Training caller → preparation, loss, evaluation and reporting |
| `items_from_run_results` | Whole recorded segments; three built-in score filters with unit default weights | Selected roots → ordinary AC training items |
| `generate_compaction_samples` | Complete public-data turns within an all-role 8192-token continuation budget | Existing public trajectories → paired training items |
| Existing model configuration | Side and reader may name the same registered base; their LoRAs remain distinct | Harness configuration → AC model and ModuleManager |
| Both `train` methods | Optional `reset_optimizer=True` at a phase handoff | Training caller → receiving trainer's existing optimizer cache |

<pre aria-label="Training data and gradient flow">
Recorded tokens + AC spans → folder-local saved rows → AgentTrainer → reader LoRA
                                              clipped loss; rows fixed

Raw messages and parts → AC encoder + side LoRA → fresh rows → reader LoRA
                         gradients                           gradients
                       ACTrainer: KL or SFT on assistant tokens
                       KL teacher: full text, same reader, no gradient
</pre>

**Accepted boundaries:** all selected recorded assistant tokens are retained, including an already incomplete final turn. Existing compactions remain separate causal segments. An oversized recorded example raises an error. Public generated windows end on complete turns and include synthetic continuation calls as assistant targets. Raw parts remain available alongside saved rows. The agent LoRA and AC reader LoRA are the same adapter.

The accepted storage revision keeps the JSONL and its tensors together. The existing `agent_utils.py` owns capture/copy handling, tensor persistence, path resolution, loading and release. The main files contain brief integration calls. Training selection keeps its existing `(model_name, lora_name)` buckets; the AC training-item schema and the writer/reader `base_dir` keyword are the material caller changes.

## Requested Decisions

The [settled intent record](./CODESIGN_PLAN.tressoir.md) includes the accepted training behavior and folder-local tensor storage. **Accepted in chat:** tensors live in `saved_tensors/` beside the final JSONL; the writer owns the destination; persistence and loading complexity belongs in utilities. This revision integrates that decision. This pass also supplies the two full proposed mainline test files in the [test review](./CODESIGN_TEST_REVIEW.tressoir.md). The user has reviewed and accepted them. Their private-test behavior matrix establishes what can be checked without long training; private test code is excluded.

**Accepted in chat, 2026-09-15:** “This looks good to me.” The interfaces and both full test proposals are accepted.

The user accepted the interfaces and both full test proposals in chat on 2026-09-15. Their next green light is intended to cover details → independent review → implementation → re-review → acknowledgement with handoff report in one continuous phase. Resolve ordinary technical findings within the agreed intent; return for input when a finding requires a user decision or scope/public-behavior change. The user then green-lit the complete phase and the narrow third-test compatibility migration: “Ok go for it.”

**Authorized compatibility exception:** pass the round-trip folder as base_dir to the existing serialize/deserialize calls in `activation/tests/test_basic_agent_ac.py`. The user approved this narrow third-file migration; all other broad mainline-test changes remain limited to the two accepted complete files.

## Milestones

M1–M4 describe the proposed interface by coherent area. Their snippets show interface excerpts; unchanged members and implementation bodies are explicitly omitted. After the user green-lights the combined phase, the detailed-plan review gates implementation without another routine approval stop.

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">M1 — Capture rows and save them beside the trajectory</span>
    <span class="card-oneliner">The writer owns the folder; utilities handle tensor persistence and loading.</span>
    <span class="card-badge">Planning</span>
  </summary>

#### Planning Overview

**Accepted storage direction:** a saved trajectory dataset is one movable folder containing the JSONL and `saved_tensors/`. The Agent captures final-cast CPU rows without knowing a filesystem path. The existing JSONL writer supplies its parent directory to utilities, publishes each tensor there, appends the JSONL record, then replaces captured tensors in the retained result with file references. Resume and training load rows only when needed.

| Location | Responsibility |
| --- | --- |
| `activation/agent/agent_utils.py` — existing utility module | Capture span metadata; avoid tensor copies during step conversion; traverse recorded spans; hash/write/validate/load payloads; rebase references; release saved rows; prepare lightweight report metadata |
| `agent.py` | Call capture and step-conversion helpers; keep the normal live rows; call the row loader on resume |
| `agent_config.py` | Pass `base_dir` through existing serialization/deserialization and delegate tensor handling |
| `rollout_caching.py` | Derive `base_dir` from its JSONL path and own append success/failure; call the release helper only after successful publication |
| `rollout_manager.py` | Save a completed result before releasing its concurrency slot; pass the cache directory when loading |
| Existing training utilities / reporter | Call the row loader or metadata helper at their existing consumption points |

All proposed row helpers go in the existing `activation/agent/agent_utils.py`; create no new utilities file. Existing batching/mask helpers stay in `activation/agent_training/agent_training_utils.py`. These are functions over the current span dictionaries, run records and paths. Keep hashing, filesystem operations, recursive traversal and tensor conversion inside the existing utility module. Keep the existing decoder/training batching helpers where they already live. Add no storage class, lazy tensor wrapper, registry or new training destination object.

<pre aria-label="Destination and ownership flow">
caller caching_id → RolloutManager → RolloutCache.path
                                    &lt;folder&gt;/rollouts.jsonl
                                      ↓ path.parent
                                    &lt;folder&gt;/saved_tensors/&lt;digest&gt;.pt

Agent captures rows → writer delegates to utilities → JSONL append succeeds
                                                   → retained result releases rows
JSONL + its parent directory → references → minibatch / resume loads rows
</pre>

The current caller supplies `caching_id`; `RolloutCache` resolves `<sync_root>/ROLLOUTS/<caching_id>/rollouts.jsonl`. Any exporter writing `<folder>/x.jsonl` uses the same `base_dir=Path(jsonl_path).parent` rule. Folder configuration stays at the writer/reader boundary and is never threaded through the Agent, tools or model config.

#### Planned Changes

`activation/agent/agent_config.py · existing prompt_ac_spans / ac_spans dictionary contents`

```python
# Live, unsaved span: the exact final-cast, detached, contiguous CPU tensor.
{"start": 120, "length": 64, "rows": cpu_tensor}

# The same span in JSON, after tensor publication:
{"start": 120, "length": 64, "row_path": "saved_tensors/<digest>.pt"}

# In a loaded or successfully saved result, row_path is resolved to an
# absolute local path. Serializing again always rebases it into the new
# JSONL directory and emits a relative path. No runtime root field is added.
# Other record fields are unchanged; implementation omitted.
```

Consumers use `load_ac_rows(span)` for either representation. An unsaved span has `rows`; a successfully saved/loaded span has `row_path`. Ordinary serialized records never contain live tensors or absolute paths. The raw activation parts remain in the existing message fields, independently of captured row storage. Keep tensors out of raw nested part construction so copied source histories do not duplicate their payloads.

`activation/agent/agent_utils.py · proposed utility interfaces`

```python
def capture_ac_spans(
    spans: list[tuple[int, int]], rows: list[torch.Tensor],
) -> list[dict]: ...

def trajectory_step_dict(step: TrajectoryStep) -> dict: ...

def serialize_ac_rows(data: dict, *, base_dir: Path | None = None) -> dict: ...
def deserialize_ac_rows(data: dict, *, base_dir: Path | None = None) -> dict: ...

def release_ac_rows(
    result: AgentRunResult, serialized: dict, *, base_dir: Path,
) -> None: ...

def load_ac_rows(span: dict) -> torch.Tensor: ...
def ac_spans_for_report(spans: list[dict]) -> list[dict]: ...
# Bodies/imports omitted. Traversal, tensor codecs and atomic-file helpers
# remain private functions in this same module.
```

`capture_ac_spans` validates row counts and retains detached CPU values after the existing dtype cast. Captured rows are treated as immutable. `trajectory_step_dict` preserves the existing snapshot behavior for ordinary step fields while retaining the captured tensor buffers without `asdict` cloning them.

`serialize_ac_rows` handles the prompt and trajectory spans in one record; the existing recursive result serialization passes the same `base_dir` to children and earlier segments. It builds JSON-ready data without replacing or removing the original tensors. Each payload is one two-dimensional contiguous CPU tensor in `saved_tensors/<digest>.pt`; the digest covers dtype, shape and raw bytes. Identical payloads reuse a file within the destination folder. Existing references from another folder are copied into this destination and rebased, so exports preserve co-location as well as new saves.

Private utility helpers publish complete files atomically, validate existing digest collisions/content, and enforce file-publication ordering before a JSONL reference can be written. They preserve tensor dtype and values and use tensor-only loading on CPU (`weights_only=True`). Serialization with any captured or referenced row requires `base_dir`; omitting it raises a clear error rather than producing an absolute, dangling or stringified tensor reference. Text-only records keep working with the existing no-argument call. Legacy spans without captured payloads remain readable; loading them for exact replay raises explicitly.

`deserialize_ac_rows` validates that serialized references stay under the supplied folder's `saved_tensors/`, resolves them into in-memory paths, and does not eagerly load tensors. `load_ac_rows` returns the existing captured tensor or loads the referenced payload and verifies its digest, dtype, dimensionality and row count. Reader-specific width and lossless dtype compatibility are checked at consumption. Missing/corrupt data identifies the run/segment/span through the caller's existing example context; it is never regenerated silently. The utility does not maintain an unbounded loaded-tensor cache.

`release_ac_rows` walks the original result and the successfully serialized record together, including children/compactions. It removes the captured tensor fields from the actual retained result and installs resolved references. It performs no new file I/O; publication has already succeeded. A failed serialization or append must preserve the result's captured tensors. Completed results must not be concurrently mutated during save. Report helpers expose span offsets, dimensions/dtype when available and reference metadata without reading tensor files or dumping tensor values.

`activation/agent/agent.py · existing capture, compaction and resume hooks`

```python
# _begin_segment(), after the existing final dtype cast:
results.prompt_ac_spans = capture_ac_spans(spans, rows)

# _append_step():
step.ac_spans = capture_ac_spans(spans, rows)
self.run_results.trajectory.append(trajectory_step_dict(step))

# compact() preserves the existing recorded span dictionaries.
# resume_from_run_result() loads historical spans with load_ac_rows(span).
# Existing live self.rows, method signatures and surrounding logic omitted.
```

Historical rows remain fixed when the encoder changes. Resumed new observations use the configured live encoder and are captured normally. Preserve actual row values and existing AC name/version provenance. No file path is required for an unsaved in-memory run. After an Agent becomes unreachable its live buffers can be collected; `shutdown()` itself only closes the environment.

`activation/agent/agent_config.py · existing serialization methods`

```python
def serialize(self, *, base_dir: Path | None = None) -> dict: ...

@staticmethod
def deserialize(
    data: dict, harness: HarnessRuntime | None = None,
    agent_config: AgentConfig | None = None, strict: bool = False,
    *, base_dir: Path | None = None,
) -> AgentRunResult: ...
# Existing signatures gain a keyword-only base_dir. Bodies omitted.
# Delegate row conversion before generic _jsonable can turn a tensor into repr.
# Thread base_dir through existing child/compaction recursion.
```

`serialize(base_dir=...)` may write/copy payload files through the utility while producing its JSON dictionary. It leaves the input result intact. A writer that publishes that dictionary subsequently calls `release_ac_rows` after successful append; serializing for inspection alone never releases tensors. Deserialization resolves references relative to the JSONL parent. The same rule applies to explicit export/read callers; the model configuration and process working directory do not determine a saved row's location.

`activation/agent/rollout_caching.py · RolloutCache.append`

```python
base_dir = self.path.parent
row = result.serialize(base_dir=base_dir) | {"config_key": config_key(result.agent_config)}
with self.lock:
    # Existing append + flush, then update self.rows only after success.
    ...
release_ac_rows(result, row, base_dir=base_dir)
# Existing self.path is still derived from caching_id; surrounding guards omitted.
```

`append` keeps its signature and writer ownership. Utilities handle per-tensor file work; the existing cache lock protects JSONL publication and its in-memory index. Failed writes raise and leave captured data available. An interruption may leave unreferenced tensor files, which is preferable to publishing a reference to an unfinished file. Automatic orphan cleanup is deferred; shared references must never be deleted merely because one record was redone or removed.

`activation/agent/rollout_manager.py · _run_jobs / _run_pending`

```python
# _run_jobs(), when loading a cached record:
result = AgentRunResult.deserialize(
    row, self.harness, agent_config=config, base_dir=cache.path.parent,
)

# _run_pending.work(), before its existing finally: limiter.release():
if result.finish_reason != "deadline":
    cache.append(result)
return result
# Remove the outer completion-loop append; it now receives the saved result.
# Deadline/None handling and existing scheduling/metrics details omitted.
```

Keep the concurrency slot until saving and release of captured result rows completes. Both the ordinary and deadline-aware completion paths must pass this save point; do not leave an early return bypass. Completed futures, `results` and metrics may still reference the same result object, but its saved spans now contain paths. This prevents a separate backlog of completed unsaved tensors when writes are slower than rollouts. It does not make total RAM constant: active runs retain captured rows, the AC cache has its existing default 4 GiB per-model limit, and text/tokens/raw parts still accumulate in the results and JSONL cache. With caching disabled, or for deliberately unpersisted deadline results, tensors remain in memory until explicitly saved or released.

`activation/agent_training/agent_training_utils.py · TrainingExample / Collated`

```python
@dataclass
class TrainingExample:
    # Existing fields unchanged; omitted.
    ac_spans: list[dict] = field(default_factory=list)
    # start/length plus rows or resolved row_path; offsets are example-local.

@dataclass
class Collated:
    # Existing fields unchanged; omitted.
    ac_rows: list[tuple[int, int, torch.Tensor]] = field(default_factory=list)
    # (batch row, token start, detached rows), after packing/padding.
```

The existing `build_example` shifts span offsets as prompt/step tokens concatenate, including thinking-strip transformations. Removing or crossing an AC span is invalid. `collate` calls `load_ac_rows` for the current minibatch and accounts for packed offsets/spacers. Release those minibatch references after use. The original encoder need not be loaded. Keep offset/mask work in the existing training utilities; tensor files and path resolution belong in `agent_utils.py`.

`activation/agent_training/agent_trainer.py · existing embedding path`

```python
# _hidden_and_logprobs consumes Collated.ac_rows before decoder_forward.
# Both the training and no-gradient reference-logprob paths use those rows.
# AgentTrainingItem, (model_name, lora_name) selection and clipped loss stay.
# Existing method bodies omitted; train's keyword addition appears in M4.
```

`activation/agent/rollout_reporter.py · trajectory_item`

```python
# Use ac_spans_for_report(...) for prompt and per-step span metadata.
# The existing report JSON remains an inspection view; it never embeds tensor
# values or produces independently loadable relative row references.
# Existing entry points and surrounding rendering are omitted.
```

The transferable dataset is the JSONL plus `saved_tensors/`; moving that folder and reloading the JSONL preserves resolution. An actual dataset export uses the destination-parent serialization rule and copies all referenced payloads. An inspection report uses metadata only and does not publish a second tensor dataset. Main tradeoffs: file-count/read-write overhead, deduplication scoped to a folder, and possible orphan files after interrupted writes. These costs and the existing text-record memory growth remain visible; no broader streaming/cache-index redesign is included here.

</details>

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">M2 — Represent and build whole assistant continuations</span>
    <span class="card-oneliner">One training item holds one aligned causal segment, including multiple parts and targets.</span>
    <span class="card-badge">Planning</span>
  </summary>

#### Planning Overview

Revise `ActivationContextTrainingItem` directly. A start is a zero-based index into its history's top-level message list, inclusive. Prefix lengths may differ. Starting there, both histories must have matching role/message order and the same assistant target sequences; tool content may differ because the AC view carries parts. An actual compaction reset requires another item. Assistant transcripts nested inside parts or tool content are context, not additional top-level loss targets.

#### Planned Changes

`activation/ac_model/ac_model_training.py · ActivationContextTrainingItem`

```python
@dataclass
class ActivationContextTrainingItem:
    item_id: str
    kind: str
    in_context_messages: list[dict] | None
    ac_messages: list[dict]
    in_context_start: int
    ac_start: int
    assistant_token_ids: list[list[int]]
    tools: list[dict] | None = None
    weight: float = 1.0
    dataset_id: str = ""
    doc_ids: list[str] = field(default_factory=list)
    info: dict = field(default_factory=dict)
    # Full proposed field list; methods/imports omitted.
```

`assistant_token_ids` has one nonempty sequence per retained top-level assistant message, in order. It contains the actual target-tokenizer output, including emitted turn endings and assistant tool-call tokens. It excludes the generation-prompt tokens already supplied as input. For recorded runs, compatible sampled IDs are authoritative, including a recorded incomplete tail; metadata/text must describe those same outputs. Builders reject incompatible tokenizer provenance or malformed alignment. Public and QA builders render their intended complete assistant outputs once and supply the resulting IDs. Prefix assistant examples are context and are excluded from this list.

KL requires `in_context_messages`. SFT can use `None`, with `in_context_start=0`; it consumes only the AC history and target IDs. The item holds data, so an item with both views can be reused for either loss.

This replaces `in_context_prefix`, `ac_prefix`, `completion_text`, `completion_complete`, `teacher_partial_token_ids`, `completion_token_ids` and `teacher_partial_text`. Existing QA generators become single-assistant instances of the new item. `transform_for_eval_reference` changes the AC input while preserving starts, teacher and assistant targets; `split_items` keeps origin-based exclusion. Active constructors/readers and serialized item-cache identities must migrate together; old item payloads fail with a regenerate-items message. This phase does not add a legacy compatibility wrapper or change archived handoffs.

**Concrete alignment:** a teacher prefix of three messages and an AC prefix of two messages may have starts 3 and 2. Their retained suffixes can both be `assistant → tool → assistant`; their two assistant ID lists must match, while the tool message may contain compressed input. Each view gets its own token offsets. Equal role lists alone do not establish alignment: call IDs/order and assistant outputs must describe the same continuation.

`activation/ac_model/ac_model_study.py · items_from_run_results`

```python
def items_from_run_results(
    self,
    runs: list[AgentRunResult],
    *,
    loss_kind: Literal["kl", "sft"] = "kl",
    selection: Literal["score_1", "score_1_or_unscored", "all"] | None = None,
    weight_of: Callable[[AgentRunResult], float] | None = None,
) -> list[ActivationContextTrainingItem]: ...
# Complete proposed signature; implementation omitted.
```

`loss_kind` determines the default filter and whether a teacher view is necessary; it does not add a loss to the item. Pass the trainer config's value to configure this once. An explicit `selection` overrides the default: KL uses `all`, SFT uses `score_1`. The presets mean exactly score 1; score 1 or `None`; and every score. The default weight is 1. A supplied `weight_of(root)` runs after selection, must be finite and nonnegative, and applies to every included segment/child; zero is an explicit exclusion counted in `harvest_report`.

Selection applies to root results and is inherited by their included subagents and compacted segments. Traversal identifies each recorded segment once, preserving origin keys for held/train splits. Every recorded assistant output is targeted exactly once in that selected traversal; previous segments appearing inside a later compressed prompt remain context there. Segments with no assistant tokens are counted as empty. An empty selected set is reported and training raises a clear no-eligible-data error.

For an AC run, preserve its actual AC messages and raw parts, then expand the full-text teacher by channel. For a text run, build the compressed view from the complete recorded initial frame and available raw channel content, using existing part constructors and channel ratios. The initial frame supplies a causal part before the first assistant output; later recorded tool/subagent parts enter at their original positions. Keep the system/tool frame and task meaning available in both views. Prefix shape can change; every recorded assistant output remains a target. Each existing reset starts its own pair, using its recorded segment context. This replaces the old random-cut text conversion; `seed`, `completion_max_tokens`, `text_run_kwargs` and `items_per_text_run` are removed from this method.

Recorded conversion adds no synthetic continuation calls, new sampling windows or fabricated end tokens. It processes the whole selected run. Invalid/missing required raw content raises a source-specific error; older corpus repair remains separate. AC training reads raw parts and does not require the captured row files used by AgentTrainer.

`activation/ac_model/ac_model_study.py · generate_compaction_samples`

```python
def generate_compaction_samples(
    self,
    dataset_id: str,
    num_samples: int,
    seed: int = 0,
    depth_range: tuple[int, int] = (1, 2),
    threshold_range_tokens: tuple[int, int] = (8192, 32768),
    ratios_range: tuple[float, float] = (1.0 / 8.0, 1.0 / 16.0),
    max_post_compaction_tokens: int = 8192,
    max_total_tokens: int = 72_000,
    min_trajectory_tokens: int = MIN_COMPACTION_TOKENS,
    *,
    max_assistant_tokens: int | None = None,
) -> list[ActivationContextTrainingItem]: ...
# Complete proposed signature; implementation omitted.
```

The old immediate/delayed single-target mixture and `completion_max_tokens=512` disappear. `max_post_compaction_tokens` counts the entire retained continuation in the uncompressed target-tokenizer view: all roles, reasoning, control tokens, delimiters, calls and tool replies. Both views derive from that same normalized source window. All assistant spans inside it receive loss. This budget applies only here; it never clips recorded-run items.

`max_assistant_tokens=None` resolves to the target model's normal generation `max_tokens` setting; a caller with a per-answer override may pass that value. The generator also accounts for the window's 8192 limit and reserves the actual rendered size of the continuation call and turn endings. It splits long splittable assistant text into complete turns using ordinary `python(code)` calls such as `print("Continuing next turn 3")`, with unique call IDs and incrementing arguments. Both views see the same call and its literal tool response. Synthetic assistant/control tokens receive ordinary assistant loss, as they would if the model had emitted them; tool responses remain masked.

The window ends after a complete assistant output and, when necessary, its tool replies; it never invents a partial recorded target or dangles a retained tool call. Public sampling may report an ineligible candidate when an indivisible call/output cannot fit; source messages remain intact, and any sample-count shortfall is explicit. Preserve source text order and end markers. These are offline representation examples, with runtime-shaped turns; synthetic Python calls do not perform compaction or promise that an external tool history executes locally.

`activation/dataset/loaders/trajectory_utils.py · reformat_trajectory`

```python
def reformat_trajectory(
    normalized: list[dict], tools: list[dict] | None, *,
    reasoning: str = "keep", tool_map: dict | None = None,
) -> tuple[list[dict], dict]: ...
# Signature unchanged. Body changes preserve reasoning separately where the
# dialect needs it; tokenizer-dependent splitting belongs to the study generator.
```

Keep this shared normalizer and `ModelDialect` as the formatting route. Preserve system/tool definitions, source reasoning and observed answers. The existing Open-SWE, Nemotron and S1 loader entry points stay intact; no gold/reference data enters the trajectory and external tools retain their existing mappings. Native `submit_answer` behavior and decontamination remain. Runtime/dataset call formatting is reused for synthetic calls.

</details>

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">M3 — Add SFT to the current AC trainer</span>
    <span class="card-oneliner">Use the same parameter groups and training lifecycle with an explicit loss choice.</span>
    <span class="card-badge">Planning</span>
  </summary>

#### Planning Overview

The AC trainer remains the owner of AC modules, side LoRA and reader LoRA updates. It reconstructs every raw part with gradients, including recursive/multiple parts. Its prepared example gains positions for all assistant targets; teacher and student offsets are computed separately. SFT needs no teacher preparation, forward, distribution cache, old logprobs or drift forward.

#### Planned Changes

`activation/ac_model/ac_model_training.py · ActivationContextTrainingConfig / _Example`

```python
@dataclass
class ActivationContextTrainingConfig:
    loss_kind: Literal["kl", "sft"] = "kl"
    # Existing learning rates, epoch/execution settings and reporting knobs stay.
    max_example_tokens: int = 72_000  # exceeding the guard raises; never drops
    # Remaining existing fields omitted.

@dataclass
class _Example:
    item: ActivationContextTrainingItem
    teacher_ids: list[int] | None
    student_ids: list[int]
    teacher_positions: list[int] | None
    student_positions: list[int]
    target_ids: list[int]
    # Existing AC spans / part requests / teacher cache key retained; omitted.
    # Positions point to decoder states predicting the corresponding target_ids.
```

The current `build_example`, `train` and `eval` consume `config.loss_kind`. Teacher and student positions enumerate exactly the same target IDs, including the first token of each retained assistant output predicted from its preceding context. Prefixes, user/tool messages, AC placeholder rows and input-only generation-prompt tokens carry no loss. Preparation uses the dialect's incremental prompt/continuation path so re-templating cannot remove earlier sampled reasoning.

KL computes a token-mean KL within each item and the existing weighted mean across items. SFT substitutes token-mean cross-entropy against the same target IDs, with the same nonnegative item weighting. Teacher distributions come from the current reader and adapter without gradient. KL's optional top-k first-visit cache remains scoped to one `train` invocation; its key covers the complete teacher token sequence and selected positions. Evaluation recomputes exact current-adapter KL. SFT rejects a nonzero `teacher_cache_top_k` or `drift_term_weight` as incompatible config instead of silently introducing or ignoring a second objective.

Before updates, validate all training/reporting/validation items. KL checks prepared teacher and student lengths; SFT checks the student alone. Check individual encoder forwards against their supported capacity as well. A `ValueError` identifies the item, failing view, measured token count and configured/supported limit. An unexpected runtime OOM propagates with context. Neither case produces a shortened item or a silent skip. Keep the current 72k default guard; removing continuation truncation is not an increase in supported model capacity.

`activation/ac_model/ac_model_training.py · existing statistics and evaluation summaries`

```python
# Add to ActivationContextTrainingStats:
loss_kind: Literal["kl", "sft"]
loss: list[float]
loss_by_kind: dict[str, list[float]]
# Existing kl/agreement/drift histories stay for KL and are empty for SFT.

# Add loss_kind to ActivationContextEvalSummary; add loss: float | None
# to it, EvalItemSummary and EvalKindSummary. Existing kl/agreement become
# float | None in those summaries (None for SFT or empty sets).
# CompletionSample.teacher becomes str | None (None for SFT).
# Other fields stay; class bodies/default factories omitted.
```

`loss` is KL in KL mode and cross-entropy in SFT mode. The existing `completion_tokens` counter now counts all selected assistant targets. SFT records zero teacher tokens/work/cache entries; KL-specific agreement remains unset in SFT. Empty sets have no numeric loss. Retain `dropped_too_long` for report compatibility, always zero under the new error policy. `summarize()` adds `loss_kind`, `loss_first` and `loss_last`; existing KL-only keys remain truthful. Bounded completion samples use a complete prefix before one chosen assistant target from the item, preserve raw reference tokens, and omit the teacher generation in SFT. Their existing generation cap is an inspection limit, separate from training coverage.

`activation/ac_model/ac_model_reporter.py · existing reporter methods`

```python
# initialize_training / report_step / report_eval retain their signatures.
# Charts and tables use loss_kind and loss; KL agreement is shown only for KL.
# Existing reference method gains a defaulted keyword:
def report_reference(
    self, name: ReferenceName, loss: float, *,
    provenance: str = "initial adapter; held set",
    loss_kind: Literal["kl", "sft"] = "kl",
) -> None: ...
# Bodies omitted. Current keyword callers using kl= migrate to loss=.
```

A reporter's series belongs to one loss kind; use a fresh reporter when changing it so CE and KL points are not combined. Reusing a reporter with a different loss kind raises a clear error. Reusing a reporter with a different loss kind raises a clear error. Existing AC/reader paired epoch saves and completed-epoch publication remain the checkpoint boundary. This interface does not add a checkpoint type or persistent teacher cache.

</details>

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">M4 — Reuse a base and hand off the reader adapter</span>
    <span class="card-oneliner">Use existing configuration and explicit trainer calls to coordinate shared state.</span>
    <span class="card-badge">Planning</span>
  </summary>

#### Planning Overview

The AC config already names both registered models and both adapters. Let those model names be equal. One registered `LoadedModel` then serves both roles, while different registrations continue to own separate instances even when their model IDs match. The agent and AC reader select the same reader adapter; the side adapter must be distinct, including its effective PEFT adapter name.

#### Planned Changes

`activation/ac_model/ac_model.py · ActivationContextModelConfig / constructor`

```python
# Existing fields; illustrative configured values:
base_side_model_name="shared_base"
target_model_name="shared_base"
base_side_model_lora_name="encoder_lora"
target_model_lora_name="agent_lora"
# AgentConfig.model_name="shared_base", lora_name="agent_lora"
# No new sharing flag or model registration API; other fields omitted.
```

Replace the same-object rejection with validation of distinct side/reader adapters. Lifecycle operations deduplicate by object identity: placement, engine sleep, train/eval restoration and checkpoint configuration occur coherently for one base. Where role-specific execution settings cannot coexist on a shared object, use one compatible setting or raise a descriptive configuration error. This shares the Transformers base; the existing serving engine continues to own its inference weights.

`activation/harness/module_manager.py · existing adapter helpers`

```python
# Existing public signatures retained:
# lora_context(model_name, lora_name)
# lora_parameters(lora_name)
# ensure_lora / save_lora / exchange_lora
# Bodies omitted: bind the requested adapter for the complete operation and
# restore prior activation; enumerate trainable parameters by adapter identity.
```

Both side and reader parameters must remain trainable while their differentiable graphs are live. Every decoder forward and checkpoint recomputation must use the adapter that created that operation. A context only around the original forward is insufficient for recomputation. Detailed planning will specify the smallest change to the existing decoder/checkpoint scopes and validate it with the installed PEFT/PyTorch versions. Shared-base support includes differentiable AC training, frozen-row agent training and current recursive encoding; simply lifting the constructor guard would not implement this contract. Concurrent mutation/training of the same base remains outside the sequential trainer lifecycle.

`activation/agent_training/agent_trainer.py · AgentTrainer.train`

```python
def train(
    self, model_name: str, lora_name: str,
    training_data: list[AgentTrainingItem],
    reporting_data: Sequence[AgentTrainingItem] = (),
    reporter: AgentTrainingReporter | None = None,
    *, reset_optimizer: bool = False,
) -> AgentTrainingStats: ...
```

`activation/ac_model/ac_model_training.py · ActivationContextTrainer.train`

```python
def train(
    self, ac_model_name: str,
    training_data: list[ActivationContextTrainingItem],
    reporting_data: Sequence[ActivationContextTrainingItem] = (),
    reporter: ActivationContextTrainingReporter | None = None,
    *, validation_data: Sequence[ActivationContextTrainingItem] = (),
    reset_optimizer: bool = False,
) -> ActivationContextTrainingStats: ...
# Full signatures shown; bodies omitted.
```

Before creating/reusing Adam, `reset_optimizer=True` discards that trainer's cached optimizer for this destination. Parameter values, completed epochs/rounds and global update/warmup clocks continue. In the AC trainer this deliberately resets both parameter groups, including the encoder moments; that small cost keeps its existing single optimizer intact.

The caller passes the flag on the first call after another trainer has updated the same reader, after an in-place external weight load, or after changing the AC loss kind. Ordinary consecutive calls keep moments. Existing parameter-identity invalidation still handles reinjection. The caller already sequences train/checkpoint/exchange calls and also owns this handoff; forgetting the flag can reuse stale moments and is a caller error. Each destination's policy is independent. No new cross-trainer optimizer owner is needed.

`activation/ac_model/ac_model_training.py / activation/agent_training/agent_trainer.py · caller example using their existing constructors`

```python
ac_trainer = ActivationContextTrainer(harness, training_config)
agent_trainer = AgentTrainer(harness, agent_training_config)
items = study.items_from_run_results(
    runs, loss_kind=ac_trainer.config.loss_kind,
)
ac_trainer.train(ac_name, items, reset_optimizer=True)
# Existing reader-LoRA exchange is required before subsequent engine rollouts.
# Later, using data selected for the same reader destination:
agent_trainer.train(
    reader_model, reader_lora, selected_items, reset_optimizer=True,
)
# Existing module-manager checkpoint/exchange calls are omitted.
```

The example illustrates orchestration at the training caller; it does not move a training recipe into runtime or create a new orchestration class. Row snapshots retain the actual observation independently of subsequent reader or encoder changes. SFT on a run remains imitation; it does not claim an on-policy ratio or require generator logprobs.

</details>

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">M5 — Review both complete mainline test files</span>
    <span class="card-oneliner">Real-data examples publicly; deterministic behavior proofs privately.</span>
    <span class="card-badge">Planning</span>
  </summary>

#### Planning Overview

The user explicitly authorizes exactly two mainline test files in this work's delta: the updated `activation/tests/test_basic_agent_ac_training.py` and new `activation/tests/test_basic_ac_self_distillation.py`. Preserve unrelated existing tests. Keep mocks, synthetic trajectories and targeted failure injection in private IB/TMP tests. Use existing utilities only.

**Required checkpoint:** present both complete files, not selected snippets, before moving to the next stage. Full proposed files are available now in [CODESIGN_TEST_REVIEW.tressoir.md](./CODESIGN_TEST_REVIEW.tressoir.md), with native `.py` files beside it under `interface_tests/`. They target the proposed interfaces and are not yet applied or runtime-tested. Keep those full files synchronized with later implementation changes and show the final complete versions again at implementation handoff.

#### Planned Changes

`activation/tests/test_basic_agent_ac_training.py · full proposed file linked below`

```python
def test_ac_compaction() -> None: ...
def test_ac_trajectory_qa() -> None: ...
def test_ac_rag_qa() -> None: ...
# Overview only; every line of the proposed file is in the test review.
```

Keep the three existing real public-data cases and paired-checkpoint/matched-adapter reporting. Replace old prefix/completion/partial-tail reads with paired message histories and assistant-token lists. This is an update to the current example, with no mocking, synthetic training traces or new auxiliary test dependency. [Full native file](./interface_tests/test_basic_agent_ac_training.py).

`activation/tests/test_basic_ac_self_distillation.py · full proposed file linked below`

```python
class RetrieveThenSolveProgram(AgenticProgram):
    def execute(self) -> AgentRunResult: ...

@pytest.mark.parametrize("loss_kind", ["kl", "sft"])
def test_basic_ac_self_distillation(loss_kind: str) -> None: ...
# Overview only; all setup, program, training, replay and cleanup code is complete
# in the native file and test review, without placeholders.
```

The new file chooses exactly four short HotpotQA questions labeled easy from a bounded train-split prefix, independently of model outcomes. It registers the real corpus and BM25 search tool. The custom program retrieves evidence, runs one real solver subagent, injects both results, then runs the main agent. Gold answers remain scoring-only; supporting-title labels do not drive retrieval. The real text rollouts retain multiple raw channel parts and the original sampled targets.

For each KL/SFT mode: collect/score four root rollouts; convert their complete histories; verify assistant-token coverage; measure the initial objective; jointly train AC modules, side LoRA and reader LoRA for one epoch/two updates; report finite losses and paired saves; reload the completed pair and check the same objective; exchange the reader adapter; run the same four questions with AC; check saved tensors/cache replay and conversion of native AC histories. Fresh run/cache/checkpoint namespaces prevent an older run from satisfying the example. Same-problem score changes are printed as diagnostics, not held-out quality evidence.

The example explicitly uses `selection="all"` in both modes so all four problems exercise training even if no generated answer is correct. Scores are never fabricated. SFT's default `score_1` rejection selection, including the zero-eligible case, is checked privately. The existing default remains unchanged. A short self-distillation smoke is sufficient here; demonstrating useful frozen-row AgentTrainer learning belongs to a later quality experiment, while its exact mechanics are tested privately.

Private tests use synthetic histories and mocks for environment calls, forbidden-forward sentinels, filesystem failure injection and call counts. They use tiny real PyTorch/PEFT models and independent numerical references for losses, gradients and optimizer updates. The [test review](./CODESIGN_TEST_REVIEW.tressoir.md) lists required behavior and proof methods; it contains no private test source. [Full native self-distillation file](./interface_tests/test_basic_ac_self_distillation.py).

</details>

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">M6 — Implementation details and independent confirmation</span>
    <span class="card-oneliner">Proceed after the full-file review, with the requested subagent confirmation.</span>
    <span class="card-badge">Completed</span>
  </summary>

The combined phase is implemented and validated; [the handoff](CODESIGN_ROUND.tressoir.md) carries the final files and results. The user green-lit the combined phase on 2026-09-15. The implementation plan executes the following accepted sequence: detail the changes inside existing utilities and their small call sites; obtain the requested independent review, implement and validate, then re-review the result and return the handoff report with both full final test files. Use private checks for exact row replay and frozen-encoder ownership, independent clipped-loss/gradient references, padded/packed causality, whole-history KL/SFT targets and joint gradients, shared-base adapter/recomputation parity, root filters, teacher caches, optimizer reset clocks, paired checkpoints, reports and all storage failure/portability paths. Include one bounded GPU serving/precision check where CPU numerics cannot exercise the actual kernel/engine boundary; no long training is required to establish these contracts. The specifically requested subagent reviews the details and required behavior coverage. Bring any material public-interface change back to the user. Only the two agreed test files belong in the mainline delta; additional tests stay private. Legacy conversion, DDP, campaigns and long experiments remain deferred.

</details>

## Compatibility and validation

Existing public QA generator entry points, AgentTrainingItem selection, raw AC part dictionaries and model/LoRA registration remain. Rollout JSON entry points gain keyword-only `base_dir`; tensor-bearing callers must pass the JSONL parent and successful writers release captured rows through the utility. Text-only calls retain their defaults. AC item constructors and old compaction/harvest keyword callers require migration; diagnostic completion generation keeps its separate bounded cap. Source and active recipe references must be audited, including IB recipes, while historical handoff snapshots remain historical. These interface proposals are now implemented; current source/canon describe the accepted whole-history KL/SFT and optional shared-base behavior. The [implementation plan](CODESIGN_IMPLEMENTATION_PLAN.tressoir.md) tracks review and GPU validation.

The user explicitly approved the narrow third-file compatibility migration: update only the existing serialize/deserialize round trip in `test_basic_agent_ac.py` to pass base_dir. The two accepted complete files remain the only broader mainline-test changes; additional behavioral tests stay private.

At interface acceptance, this revision included full test proposals and plan updates only. All three artifact checks passed, along with paired-source/full-file agreement, Python syntax, local links, control keys, planned constructor arguments and the two-file/no-mock scope checks. Current call sites were audited against source; tracked product files and prior answers were preserved. Direct structured review: pass for the planning deliverable. Those were planning-only results. Implementation and the current CPU/GPU evidence are tracked in the linked implementation plan; this document retains the accepted forward design.
