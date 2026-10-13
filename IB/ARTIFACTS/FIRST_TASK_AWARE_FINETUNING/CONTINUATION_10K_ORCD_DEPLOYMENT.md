# ORCD alternative for the two 32k → fresh 10k continuations

**Reject L40S training with the frozen settings.** Comparable completed runs used
63.87 GiB (32,000 × 1), 61.69 GiB (16,000 × 2), and 62.03 GiB (16,000 × 1).
These are measured allocated GPU peaks, not model-size estimates; they exceed
L40S's nominal 48 GB capacity. All used the same microbatch 2 and checkpoint
threshold 1700. Smaller-batch or checkpointing changes are outside this deployment.

Prefer **one RTX PRO 6000 per arm** on `mit_preemptable`, with both arms on the
same hardware. Normal H200 is a command-line alternative if its queue is better;
current mixed-node availability does not establish a reliable start-time advantage.
Training requests remain 3 hours, 8 CPUs, 64 GiB host memory; readouts remain
45 minutes with the same CPU/host request. The training ceiling remains 10,200
seconds, and all scientific arm configurations remain byte-for-byte equivalent.

Staged under `~/orcd/pool/activation/taskaware/bench/`:

- `train_continuation_10k_orcd_v1.sbatch final_lr` or `rewarm`; optional `--resume`.
- `continuation_cohort_orcd_v1.sbatch final_lr` or `rewarm`.
- `CONTINUATION_10K_DESIGN_ORCD.json`, a separate deployment overlay.

For H200, pass `--partition=mit_normal_gpu --gres=gpu:h200:1` to `sbatch` for both
arms and their readouts. Defaults request `gpu:rtx_pro_6000:1`. No GPU job has
been submitted or CSAIL job cancelled by this task.

The original frozen design, runner, evaluator and source remain unchanged.
The overlay changes only run-root paths, the original ORCD pair-JSON hash,
platform/provenance text and added launcher hashes. It uses the already prepared
ORCD `continuation_10k_material/material_27b_10000.json` and
`continuation_32k_initial/pair.json`. No payload recopy, Adam change, new data or
new selection is involved. The copied parent panel and exact held-input
fingerprints remain enforced. Both arms preserve Adam and use their own complete
final pair; all final-run, source/config and 400-question NLL-replay guards remain.

CPU verification **23511022 completed successfully in 20 seconds**. It checked
every initial payload, the material/pair, source and helper hashes, and unchanged
scientific configurations. Training launchers require its successful
[verification receipt](CONTINUATION_10K_ORCD_VERIFICATION.json), also staged under
the remote `bench/` directory.

Suggested conservative priority fence for **both** continuation training jobs:
`afterok:23511022:23493337:23498370:23510987:23510988`.
This clears CPU verification, data-amount confirmation, both trained-layout
32k readouts (which depend on their own 8k readouts), and the frozen-32k resume
before admitting the two longer continuations. Recheck IDs after any repair.
Each continuation final100 job then depends only on its own successful training.
Keep the optional batching probe behind both continuation final100 readouts;
the campaign owner must verify it is still pending before changing dependencies.

Before replacement, the owner must reconfirm that CSAIL 2143664/2143666 and their
2143665/2143667 readouts are still pending and cancel those exact IDs, then submit
ORCD replacements once. Maintain **≤4 total ORCD campaign GPUs**, including
preemptable readouts, and **≤2 normal-partition GPUs**. H200 does not add a new
budget beyond that cap. No cancellation or queue modification is performed here.

Hashes, measured-memory sources and the CPU receipt are recorded in
[the deployment receipt](CONTINUATION_10K_ORCD_DEPLOYMENT.json).
