# Current code handoff

**Historical capture:** the current source-sync package is [CONSOLIDATION_HANDOFF.md](CONSOLIDATION_HANDOFF.md). The already-applied package below remains provenance only.

This captures the workspace **before** the proposed task-aware implementation, against the `/source` baseline captured on 2026-09-21. It lets you sync the existing code independently of approving the new experiment plan.

**Source sync observed at 15:25 UTC:** all 30 packaged changes are now reflected in `/source`, including the existing test caller migration and retired-module deletion. Their current hashes match the packaged destination, and the reverse patch check passes. This agent did not modify `/source`. Preserve this handoff as the captured before/after record; do not apply it again to the already-synced tree.

## What to use

- [Full review surface](CURRENT_CODE.tressoir.md): all 30 file differences, with exact hunks.
- [Patch](current_code/source.patch): 26 runtime modifications, two previously requested training tests, the required existing AC round-trip caller migration, and deletion of the retired `activation/agent/agent_programs.py`.
- [Full files and hashes](current_code/manifest.json): source and destination SHA-256 for each file; replacement files retain their `activation/...` paths under `current_code/`.
- `pyproject.toml` and `uv.lock` already match `/source`.

Apply from the destination repository only after checking that its baseline matches the manifest:

```bash
git apply --check /path/to/current_code/source.patch
git apply /path/to/current_code/source.patch
```

If you already incorporated part of an earlier handoff, use the full-file differences to integrate the remaining changes. Do not force the cumulative patch over a different baseline. No changes were made to `/source` here.

## Behavior in this snapshot

| Area | Existing changes carried by this handoff |
|---|---|
| AC model and utilities | Input rows plus residual interventions at configured reader layers; passage-attention readouts and dedicated heads; optional recursive keys and strided content pooling; compatible checkpoint migration. The new Golden runs explicitly disable content pooling. |
| AC study/trainer | Source-faithful trajectory conversion, whole retained targets, literal SFT, stored KL targets, base-reader no-context regularization, shared-base adapter ownership, padded batches, cosine scheduling, intermediate paired checkpoints and optimizer resume. |
| Captured agent rows | Exact CPU activation rows are recorded/replayed; serialization spills tensors beside the trajectory JSONL and loading resolves relative paths. Agent training consumes captured rows rather than silently re-encoding them. |
| Harness | Restored adapter routing/trainability across nested scopes and checkpoint recomputation, residual intervention hooks and model placement support. |
| Reporting | Phase-aware progress, gold NLL and named references, deep-channel diagnostics, and persistent live widgets rather than routine destructive page rebuilding. |
| Supporting changes | Study progress forwarding, retained reasoning text in trajectory renderings, spot-instance support, and existing autotuner compatibility fixes. |

The scope is cumulative: these changes span earlier completed rounds and the latest reconstruction work. They are not all new edits made for this plan. Bench scripts, experiment configs, one-off probes and other test changes are deliberately outside this product-code handoff. The two full training tests are `test_basic_agent_ac_training.py` and `test_basic_ac_self_distillation.py`, requested in the earlier co-design work. The handoff also includes the previously approved three-line `base_dir` migration in existing `test_basic_agent_ac.py`; omitting that caller fix would break its captured-row round trip. No new test suite is introduced.

## Known limits before task-aware runs

1. RAG generation still invokes synthesis, selects the first positive chunk and gives its text reference only the gold passage. Native support sets and matched full-context distractors are planned work.
2. The no-context penalty is anchored to the original base, using top-32 plus remainder KL. Its persisted cache needs model/reference provenance checks. The reporter's claim that rising penalty means improvement is incorrect and is included in the upcoming correction.
3. Existing QA splitting holds out questions, not necessarily supporting documents. The new benchmark must split source groups before constructing evidence bundles.
4. Reconstruction generation currently accepts a scalar ratio. The range default has not been implemented.
5. The deep residual path is established in the Hugging Face trainer. End-to-end deep serving through the separate vLLM fork is outside this handoff's validation claim; the proposed QA experiment uses the trainer's existing generation/evaluation path.
6. The current optional pooling/nested features are retained in source. Their presence does not make pooled checkpoints compatible with an unpooled Golden run.

## Validation and provenance

At capture, this package passed Python AST parsing, forward patch applicability against the then-current `/source`, reverse patch applicability against `/workspace`, and the Markdown artifact checker. After the observed source sync, all 30 `/source` paths match the packaged destination hashes and the reverse patch applies there as well. Staged contents and patch are tied to the manifest hashes. No product code was changed while packaging it.

Prior evidence, reused rather than represented as new tests: the earlier co-design completed both mainline end-to-end training files on GPUs; the reconstruction interventions/nested-material work records 20 passing CPU tests. The full current cumulative snapshot has **not** received a fresh GPU regression run. The new plan requires focused private proofs and a representative P2 restore/QA smoke before training. See [PLAN.tressoir.md](PLAN.tressoir.md).

Snapshots of P2 and P2s5 were separately archived with S3 SHA-256 verification; that establishes byte integrity, not a successful GPU restoration under today's code. See [CHECKPOINTS.md](CHECKPOINTS.md).
