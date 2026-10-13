# 4B encoder feasibility — 2026-09-22

**P4 is selected for the queued 4B run.** This is an actual Qwen3.5-4B encoder paired with a Qwen3.5-9B reader, available on ORCD. It is a useful trained warm start, but its held reconstruction NLL **0.3232** misses the Golden `<0.2` gate and is worse than P2's **0.1064**. A subsequent comparison measures the performance of these two pretrained pairs; it cannot isolate encoder capacity.

| Checkpoint | Encoder → reader | Reconstruction tokens, approximately | Encoder / reader LR | Held gold NLL | Held token accuracy |
|---|---|---:|---|---:|---:|
| **P4** | **4B → 9B** | **48M** | **1e-4 / 2e-5** | **0.3232** | **90.19%** |
| P4s1 | 4B → 9B | 21M | 1e-4 / 2e-5 | 0.5224 | 85.05% |
| P4s5 | 4B → 9B | 21M | 5e-5 / 2e-5 | 0.3751 | 88.59% |
| P2 reference | 2B → 9B | 48M | 1e-4 / 2e-5 | 0.1064 | 96.50% |

The common held set contains 420 novel passages. P4 trained for one uninterrupted epoch over 77,568 passages (9,696 updates); its seen-training gold NLL is 0.2799. The short P4 variants underwent historical resumes, another reason to prefer P4. These old reconstruction runs predate the current synthesis-only QA policy; their mixed reconstruction corpus included historical math solutions and teacher outputs. No new native benchmark QA is introduced by this proposal.

## Saved identity and compatibility

Actual adapter configs and AC checkpoint metadata agree on `Qwen/Qwen3.5-4B` → `Qwen/Qwen3.5-9B`, widths 2560 → 4096, rank-128 LoRAs (alpha 256), architecture version 2, ratio 1/16, minimum 2 rows, maximum 4096 rows, and two mixer layers. No input/content pooling is configured. Deep passage-attention readouts feed residual heads at all eight full-attention reader layers, `[3,7,11,15,19,23,27,31]`. The historical internal reader alias `qwen4b` is retained because it is a registry name, not the actual base identity.

P4 predates the added input-row and recursive readouts: both complete sets of keys are absent. The current loader explicitly supports this case, initializing these readouts as identity while restoring the existing trained heads and deep attention. This is the same migration used for P2. The width-changing row head correctly has no identity skip. CPU checks confirmed the private runner's base identities, structural config and adapter checks against all three saved metadata records. **GPU restore/train/evaluate/paired-save check passed on RTX PRO 6000 (job23436039)**, using8 synthetic training and4 reporting questions. It completed with63.26GB reported peak memory. This verifies the current loader/migration and training path; it does not estimate QA quality, prove numerical equivalence to historical kernels, or guarantee the largest future batch fits.

P4's intact paired checkpoint is:

```text
/home/ngom/orcd/scratch/activation/KEEP/P4/
  AC_MODELS/rag_poc/epoch_001/
    ac_modules.pt
    side_lora/
  LORAS/rag_reader/epoch_001/
```

The three weight payloads are present and total 3.38 GB. The original P4 pair is now preserved locally and archived at `s3://general-bucket-aln/activation-context-data/checkpoints/reconstruction-20260922/P4/`. All 11 payload objects (3,383,849,998 bytes) and the final manifest passed S3 SHA-256/size verification. The archive contains original reconstruction weights and provenance, without optimizer state or public base models. [Verified manifest](CHECKPOINT_ARCHIVE_P4.json). A `PRESERVE` flag protects the original ORCD pair. P2/P2s5 remain separately archived.

## Narrow next run

Use **P4, 16,000 synthesized questions × 1 epoch, 27B-oracle material, ratio 1/16**, with the exact frozen material and reporting questions used by the matching P2 arm. Keep the current merged evidence layout, approximately 2k encoder tokens, up to three distractors, 8 examples/update, microbatch 2, encoder LR 5e-5, reader LR 2e-5, and the same loss, no-context penalty and schedule. Preserve P4's paired reader adapter; swapping in P2's reader would create another initialization experiment. Label the run `encoder4b_oracle27b_16k_r16`.

The local private runner [run_qa_encoder4b_v2.py](bench/run_qa_encoder4b_v2.py) is a separate sibling of v3. It selects the 4B side base, loads `QA_P4_ROOT`, checks both adapter identities/ranks before loading bases, and records the encoder/reader bases plus original checkpoint paths in the immutable run identity and visible recipe. It refuses fresh initialization and legacy runs lacking this explicit identity. It does not change v3, v4 or product files.

**Large job 23446884 is running on ORCD RTX node5103**, since about06:06UTC Sep22. It started after the 2B/27B-supervision/1/8 arm (23446877) completed and the frozen-base reference cache was verified. It reuses that slot and keeps the maximum at four active autonomous GPUs. Its dedicated wrapper sets `QA_P4_ROOT=/home/ngom/orcd/scratch/activation/KEEP/P4` and unsets the older campaign deadline for this user-authorized experiment only; it has a 7-hour Slurm limit and 6-hour training ceiling. The common material is frozen; [design and hashes](MODEL_COMPARISON_DESIGN.json).

## Verification records

- [Structured inventory](4B_ENCODER_INVENTORY.json): actual model IDs, saved architecture, payload sizes, local result digests and S3 scope.
- `IB/TMP/FIRST_TASK_AWARE_FINETUNING/encoder4b_inventory/remote_metadata.json`: bounded remote metadata receipts; only ~26 KB of checkpoint metadata was read per candidate, without tensor payloads.
- [CPU check](bench/check_encoder4b_config.py): real config dataclass with a fake harness, positive checks for P4/P4s1/P4s5, three adapter mismatch rejections and legacy-resume rejection; passed. Python compilation passed.
- Reconstruction results: `IB/TMP/SYNC/AC_RECON_ORCD/{P4,P4s1,P4s5,P2}/result.json`; historical context: `IB/ARTIFACTS/AGENT_AC_PRETRAINING/ac_recon/LEDGER.md`.

GPU smoke evidence: `../../TMP/SYNC/TASK_AWARE_QA/encoder4b_restore_smoke/result.json`; stop reasoncomplete, paired AC/reader checkpoints saved. The large run still starts from the originalP4 pair, not these smoke-trained weights.

## Initialized-optimizer memory check

Job **23442317 completed** on RTX PRO 6000 in 5m26s. Eight of the longest available real 27B source windows (5,387–6,114 source tokens; 45,793 total framed encoder tokens) were processed twice at microbatch 2. Both updates, final evaluation and paired saving completed; the second backward ran with Adam state allocated. Reported training peak was **61.06 GiB**, with zero dropped examples. This supports the planned short-context configuration; it does not certify arbitrary longer batches. These check weights are discarded as an initialization candidate: the large run starts from original P4. Evidence: `../../TMP/SYNC/TASK_AWARE_QA/encoder4b_memory_check_v2/result.json`.

The large-run v2 runner differs from the qualified v1 only in flat-table publication: nested cohort scores remain in their dedicated curves and raw JSON. The original 2B cohort baseline exposed the same table error before any training update. Focused tests reproduce that failure and pass on both corrected runners; the model and training paths are unchanged.

The original training and reference-copy dependencies completed. Continue with the exact optimizer state if this run is preempted; never substitute a weights-only continuation.

## Early task-aware results — 07:06 UTC

After **3,200 of 16,000 unique questions, first epoch** (20%), the 4B system improves both development cohorts relative to the matched 2B run. Both use the same 27B-generated training data and 1/16 compression.

| System | 9B-generated evaluation NLL | 27B-generated evaluation NLL | Learning NLL (32 fixed questions) |
|---|---:|---:|---:|
| 2B/P2 | 1.0774 | 0.9165 | 0.9064 |
| 4B/P4 | 0.9250 | 0.7906 | 0.8225 |

At 1,600 questions, the systems were broadly similar. This is an encouraging early learning trajectory, not a final size conclusion: the reconstruction checkpoints and initial paired reader adapters differ, and there is one training seed. The common 400 development questions are unchanged. The final fixed 100 response review and the original reserved confirmation remain pending.

Live report: `IB/TMP/SYNC/TASK_AWARE_QA/encoder4b_oracle27b_16k_r16/report.tressoir.html`.
