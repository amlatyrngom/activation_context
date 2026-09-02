# Root Canon

A small index and cross-cutting memory for this project.

## Cross-cutting decisions and invariants

- **Convention — Preserve the mental model in incremental code fixes.** Start from the user's current code and make the smallest coherent behavioral change. Do not mix a bug fix with reformatting, import reordering, comment rewriting, renaming, defensive checks, or structural cleanup unless those edits are required for correctness. Pair-programming diffs are a human communication surface: show only intent-bearing changes, and isolate any unavoidable mechanical change so the human never has to search through churn for the real behavior. If a broader change is necessary, identify the concrete fault first and keep the behavioral delta easy to isolate. `activation/harness/hf_utils.py::adapt_message_format` is the reference example: preserve its existing shape when a focused correction is sufficient.
- **Convention — Tests are opt-in in pair-programming diffs.** Do not add test files or include test changes in the proposed handoff unless the user requests them. Private agent validation may live under `IB/TMP/`; report its result without making the human review a test implementation.
- **Invariant — Pair-programming handoffs receive a human-quality review.** Before publication, a reviewer checks correctness and whether every visible hunk communicates real intent to a human pair programmer. Remove incidental formatting/refactor noise and reject diffs whose behavior is difficult to distinguish from mechanical churn.
- **Invariant — Keep staged handoff folders clean.** Do not leave superseded source snapshots, `.orig` backups, patch rejects, or temporary working files in artifact handoff folders. Keep only the current handoff and its intentional review/support files; use `IB/TMP/` for disposable work.
- **Invariant — Keep interactive projections current.** When a handoff's staged source or accepted behavior changes, update its interactive `.tressoir.md` review surface in the same work slice. The projection must describe the current handoff and must not remain as a stale prior delta.

## SkyPilot workflow decisions

- **Invariant — Remote source has one exact owner.** `~/sky_workdir` is a disposable mirror owned by the local repository. Every `exec` resumes a stopped cluster with `--retry-until-up`, then performs the exact `upload` and may delete any remote-only or ignored path under that workdir.
- **Invariant — Retained remote state stays outside the source mirror.** Model/JIT caches belong beneath `/root/.cache` and checkpoints/results beneath `~/activation_artifacts`. Pause retains node-disk state; teardown destroys it, so durable results must be exported separately.
- **Convention — Transfer commands name their direction.** `upload` means local source to remote; `download` means remote artifacts to local. Downloads are explicit, non-deleting, preserve existing files by default, and confine in-project destinations to `IB/TMP`.
- **Decision — GPU dependencies are baked while source remains live.** The SkyPilot image supplies CUDA-devel, Python, and the frozen `uv.lock` environment outside the workdir. Setup validates CUDA and lock compatibility; remote commands import freshly uploaded source rather than reinstalling the project.
- **Decision — Docker images move through a registry.** SkyPilot pulls the project image from private ECR; it does not byte-upload a local Docker image. Registry credentials come from the standard AWS credential chain and must never be committed or baked.
- **Decision — Secrets are job-injected, not source-synced.** Keep project `.env` files excluded from the exact remote mirror, images, caches, and artifacts. When an existing `.env` is needed by a trusted remote command, pass it through SkyPilot `exec --secret-file`; never print values during validation. SSH encryption protects transport, while the remote job/node remains inside the secret trust boundary because job code can inspect its environment.
- **Decision — Remote vLLM commands use spawn-safe workers.** The Sky wrapper sets `VLLM_WORKER_MULTIPROC_METHOD=spawn` so vLLM does not fork after the parent process has inspected CUDA.

## Input-embedding decisions

- **Decision — Copy the primary embedding module, not only its weight.** Use an independent copy of `get_input_embeddings()` so target-defined lookup behavior such as Gemma 3 scaling is preserved and tied storage does not keep the full model alive. Freeze/eval the caller-owned copy. If the full model began at the free sentinel, restore it to free after copying; otherwise preserve its starting placement.
- **Decision — Keep embedding residency policy outside the helper.** The helper does not automatically cache or pin. In the audited 8B–32B fixtures, primary BF16 tables occupy 1.16–2.62 GiB and 2.37–10.54% of checkpoint parameters, so absolute GPU headroom decides whether permanent residency is worthwhile.
- **Invariant — `d_model` shape is assumed, not treated as semantic proof.** For side-model inputs, hold row layout and target width constant and evaluate learned representation compatibility. vLLM acceptance does not establish that side activations occupy the target model's input basis, scale, or distribution; use a trained and validated projector/adapter unless exact target-space rows are proven.
- **Decision — Audit the complete target input path per architecture.** Qwen 3/Qwen 3.5 use the primary input space; Gemma 3 adds scaling in its embedding module. Gemma 4 E2B additionally uses token-conditioned per-layer embeddings, so main `d_model` rows alone are not a complete text-equivalent carrier. Text-derived full prompts can preserve that path with meaningful companion target token IDs; non-text side rows require an explicit trained PLE/sentinel policy or richer carrier. Do not generalize E2B's path to Gemma 4 variants whose pinned configs do not declare it.

## Dataset loading and indexing decisions

- **Invariant — Labeled examples are first-class under budgets.** Loaders select examples as an unbiased prefix via `take_to_budget` (append-then-break, so it errs toward overfetching), with `filter_fn` for validity predicates and `cost_fn` available when fan-out (gold documents, passages) is the real resource. Never drop or skip examples because their documents are long or numerous.
- **Decision — Bound embedding cost by truncation, not document filtering.** Documents always load complete; `doc_embedding_input_limit_chars` applies `safe_truncate_embedding_chunk` (head + tail halves) to embedding inputs only, on both index-build and query paths. CPU tests run small limits (64–512); GPU runs use `None`. Filtering documents by length skews benchmark composition and drops examples.
- **Decision — Subsampled corpora hold the selected examples' gold documents.** BRIGHT loads golds only (other examples' golds are the natural distractors); `max_examples=None` means every labeled example. SciFact currently keeps seeded-sample distractors because claims have ~1 gold abstract each.
- **Decision — Benchmark exclusions are respected per query, not per corpus.** `LabeledRetrievalQAExample.excluded_doc_ids` carries native exclusions (BRIGHT `excluded_ids`, minus the `'N/A'` sentinel); `query_many_frozen` filters every chunk of an excluded document, over-fetching a flat +10 (can undershoot `top_k` in pathological cases, accepted).
- **Convention — Chunk ids are `"{doc_id}:{chunk_num}"`.** Pre-sized retrieval units (BRIGHT passages, MS-MARCO/NQ passages, SciFact abstracts, SciQ supports) load `atomic=True`. Chunks sort by length before batching so padding stays tight.
- **Convention — Public-dataset sanity is printed inspection plus reliable asserts.** The public loading test prints labeled queries with gold-marked top-k and per-dataset `DatasetStats.summarize()`; asserts cover only what is reliably true (example bounds, label integrity, exclusion respect, exact-chunk self-retrieval). No performance A/B in loading tests.

## Subsystem index

Create focused files such as `EDITOR_CANON.md`, `AUTH_CANON.md`, or `RELEASE_CANON.md` as needed, and link them here with a one-line description.

## Canon artifacts

Keep reviewed, validated configs, scripts, templates, fixtures, and other known-good reusable examples in `CANON_ARTIFACTS/`. Version-control them and document their usage and limits in that directory.

## Open questions

- Record unresolved questions only while they remain useful to future work.
