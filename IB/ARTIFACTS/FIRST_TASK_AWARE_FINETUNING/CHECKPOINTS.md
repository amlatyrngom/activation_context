# Starting checkpoints and durable storage

**Recommend P2, epoch 1**, as the first task-aware starting pair. It has the best completed result among the available unpooled 2B-encoder checkpoints and keeps the Golden architecture's principal choices.

| Candidate | Match and deviation | Recorded held reconstruction result | Decision |
|---|---|---|---|
| P2, epoch 1 | Qwen3.5-2B → 9B; rank-128 adapters; every full-attention layer; no content pooling; 1/16; about 48M tokens. Encoder LR was 1e-4; predates the added input/recursive readouts. | NLL 0.10637 nats/token; token accuracy 96.50%; 420 non-wiki items. | Primary; use the supported identity-readout migration and reset optimizer/schedule for QA. |
| P2s5, epoch 1 | Same sizes, ranks, channel and ratio; LR 5e-5; only about 21M tokens; also predates the added readouts. | NLL 0.38989; token accuracy 88.23%; 420 items. | Archived comparison/fallback for LR provenance; weaker starting performance, not the default. |
| F, partial step 2700 | Warm continuation of P2 with current readouts and LR 5e-5; cancelled before completion. | No completed final result from the continuation. | Keep on cluster, do not promote over the completed P2 pair on incomplete evidence. |
| E2 / E4 | Strided content pooling. | Different architecture; the matched F continuation was interrupted. | Excluded by frozen no-pooling requirement. |
| Older 4B-encoder / K/V-transfer runs | Different encoder or retired deep channel. | Some strong reconstruction scores under those designs. | Excluded as the starting point; low NLL alone does not establish transfer compatibility. |

These scores are reconstruction measurements, not evidence that QA already works. P2's `qwen4b` registry alias is historical: the checkpoint's actual model IDs and widths are 2B/9B, 2048/4096. Model identity must come from metadata, not that alias. P2's reader-specific heads/adapter cannot simply be loaded into a 2B, 4B or 27B reader. The **recipe and implementation** transfer; those reader checkpoints require their own trained compatible pairs.

## Archive completed

AWS account access was verified. The account already has `general-bucket-aln` in `us-east-2`, with all four public-access blocks enabled. Reused it as requested instead of creating another bucket:

```text
s3://general-bucket-aln/activation-context-data/checkpoints/reconstruction-20260921/
  manifest.json
  P2/
  P2s5/
```

Each candidate contains `AC_MODELS/rag_poc/completed_epoch.json`, its numbered `ac_modules.pt` and side LoRA, the paired numbered reader LoRA, `arm.json`, and `result.json`. Each pair is about 2.456 GB; together about 4.911 GB. Public base weights, optimizer moments, duplicate `latest` folders and reconstruction-specific no-context caches are excluded. This is an initialization archive for new training, not an exact optimizer-resume bundle for the old experiment.

Every object was uploaded with a supplied SHA-256 checksum, then checked using S3's returned checksum and byte count. The manifest was published last. The local [archive manifest](CHECKPOINT_ARCHIVE.json) records every object and digest. Objects use server-side AES256 encryption. No credentials were copied to either cluster.

Before the first QA run: download into a fresh run root, verify the manifest, load both adapters and AC modules, assert no pooling/full-attention layers/ranks, and reproduce a fixed reconstruction panel before and after the readout migration. Current loader migration support is source evidence; numerical equivalence and actual GPU restoration remain explicit validation gates.

Local copies: `IB/TMP/SYNC_RUNS/P2/` and `IB/TMP/SYNC_RUNS/P2s5/`. The originals remain on ORCD. No GPU jobs were launched for archival.

## Task-aware checkpoint archived — 2026-09-21

The selected **hill_control update 900** AC/reader pair is now verified in S3:

```text
s3://general-bucket-aln/activation-context-data/checkpoints/taskaware-20260921/hill_control/update_00900/
```

This is the lowest development answer NLL among that arm's reported checkpoints.
It is update 900 of 1000: epoch 2, update 400 of 500, after 7,200 presentations of the
4,000 training questions. It is **not the final update 1000 pair**. The separate
confirmation questions were unused for selection; these are development results.

| Fixed development panel, 100 questions | Answer-token NLL |
|---|---:|
| AC | 0.58714965 |
| Full context | 0.42591855 |
| No context | 1.88050563 |

All three views use the same reader LoRA at update 900; full context includes the
same evidence and distractors. The metric averages per-question gold-answer
token NLL and excludes tool syntax. Training started from P2 epoch 1 with the
2B encoder/9B reader, rank 128, no pooling and 1/16 compression.

The [QA archive manifest](QA_CHECKPOINT_ARCHIVE.json) records 14 payload objects
(2,490,086,374 bytes): `ac/`, `reader/`, original selection and panel 900 records,
frozen training configuration, material IDs/hash, run recipe and P2 initialization
provenance. Public bases, optimizer state, other selections and the full training
corpus are excluded. Use this pair for evaluation or initializing new training;
the archive does not contain optimizer-resume state.

Each object passed S3 PUT SHA256 validation and a matching HEAD checksum/size
check. Writes were create-only, so an unexpected existing object could not be
overwritten. The 6,833-byte manifest was uploaded and verified last, then the
local artifact was published. Its SHA256 is
`3efaa04bfe0ce8fe583b065772594288f9fc6a9fb3416cb49f19dbd271de0024`.
No GPU or cloud instance was launched for archival.

Local preserved pair:
`IB/TMP/FIRST_TASK_AWARE_FINETUNING/qa_checkpoint_archive/hill_control/update_00900/`.
The [completion receipt](../../TMP/FIRST_TASK_AWARE_FINETUNING/qa_checkpoint_archive/hill_control/update_00900/archive_receipt.json)
records the verified final manifest. The original protected ORCD pair remains.

## 27B-synthesized QA checkpoint archived — 2026-09-22

The completed **16,000 questions × 1 pass** `oracle_27b_16k_r16` pair is preserved
at `s3://general-bucket-aln/activation-context-data/checkpoints/taskaware-20260922/oracle_27b_16k_r16/epoch_001/`.
This is the exact final epoch pair, independently of the training selection record.
The encoder is Qwen3.5-2B and the reader is Qwen3.5-9B; **27B identifies the QA
synthesis teacher**, not the reader. Both rank-128 adapters started from P2 epoch 1.
Configured compression is 1/16; the archive also records aggregate row counts and
their small rounding difference.

Final development answer NLL is **0.6429122** on the 200 questions synthesized by
9B and **0.5636155** on the separate 200 synthesized by 27B. These are development
results; reserved confirmation had not been evaluated at archival.

The [verified archive manifest](QA_CHECKPOINT_ARCHIVE_27B16K_R16.json) records
18 payload/metadata objects totaling **2,496,309,177 bytes**, including completion,
training configuration, run identity/recipe, material IDs, the final panel, P2
provenance and the evaluator checksum receipt. All five model-file SHA256 hashes
match the completed GPU evaluator. Each S3 object passed create-only PUT and
server-checksum/size verification. The 10,089-byte manifest was published last;
its SHA256 is `58f1ad8b251ee4809ae05a4a184ebdc4d9d27a345433dd6e87df778f146857c4`.

Load `ac/` and `reader/` together. The original completion marker retains the
original run-relative paths; restore those paths only if reproducing that layout.
Base weights, optimizer state, other saves and the full training corpus are
excluded. This pair supports inference or new training with a fresh optimizer.
No new GPU or cloud instance was launched for archival. The original protected
ORCD pair remains, alongside the local preserved staging copy and its
[completion receipt](../../TMP/FIRST_TASK_AWARE_FINETUNING/qa_checkpoint_archive/oracle_27b_16k_r16/epoch_001/archive_receipt.json).

## 4B encoder comparison archived — 2026-09-22

The exact final `encoder4b_oracle27b_16k_r16/epoch_001` pair is verified at
`s3://general-bucket-aln/activation-context-data/checkpoints/taskaware-20260922/encoder4b_oracle27b_16k_r16/epoch_001/`.
It uses a **Qwen3.5-4B encoder and 9B reader**, rank-128 adapters initialized from
**original P4**, trained on **16,000 unique 27B-synthesized questions × 1 pass**
with 1/16 compression. The P4 reconstruction archive is linked in its provenance;
this is neither the P4 smoke-trained pair nor an intermediate QA checkpoint.

This preserves a useful larger-encoder comparison. Development answer NLL was
**0.6048205 / 0.5621804** on the 9B/27B question cohorts (200 each). On the shared
qualitative development cohort, the 4B system answered **60/87** supported-reference
questions correctly versus **66/87** for the 2B system. It is not the semantic-quality
winner; differing P4/P2 initialization and paired readers also prevent an isolated
encoder-size conclusion. Original confirmation had completed, but its scores were
not accessed or used for this archival selection; its protocol is unchanged.

The [verified manifest](QA_CHECKPOINT_ARCHIVE_ENCODER4B_27B16K_R16.json) records
**19 payload/metadata objects, 3,443,359,467 bytes**. CPU-only ORCD job **23471568**
completed staging and hashing in 20 seconds; all five model/config hashes matched
the completed evaluator, as did the final result and completion marker. Every S3
object passed create-only PUT and returned SHA256/size verification. The 11,582-byte
manifest was verified last; SHA256:
`743bc106599790c71d1a274385560d9e1f0ebfe54d1015e03767df26a17e8794`.

Restore `ac/` and `reader/` together with separately available base models.
Optimizer state, corpus, intermediate checkpoints and base weights are excluded;
this supports inference or fresh-optimizer training. Original archives are unchanged.
The protected original and local staging pair remain, with a
[completion receipt](../../TMP/FIRST_TASK_AWARE_FINETUNING/qa_checkpoint_archive/encoder4b_oracle27b_16k_r16/epoch_001/archive_receipt.json).
No GPU or cloud instance was launched for archival.


## 32,000-question final development candidate archived — 2026-09-22

The **preselected final 32,000 unique questions × 1 pass** pair is verified at
`s3://general-bucket-aln/activation-context-data/checkpoints/taskaware-20260922/oracle_27b_32k/epoch_001/`.
It uses the original P2 initialization, Qwen3.5-2B encoder → 9B reader,
rank-128 adapters and 1/16 compression, trained on 27B-synthesized questions.
The final epoch completion marker and paired paths passed the full-run guards.

Final development answer NLL is **0.5688311** across 400 questions
(**0.6059430 / 0.5317192** on the 9B/27B cohorts), versus **0.6032638** for
the earlier 16,000-question × 1-pass run with its shorter learning-rate schedule.
This is a **development candidate; the final 100-answer review was pending at
archival**, and no semantic-quality win is claimed. Reserved confirmation was
also pending and unused for selection. The lower observed NLL at 28,800 questions
(**0.5580611**) did not change selection: only the preselected final pair is archived.
`selected_checkpoint.json` retains that earlier best-panel provenance and must
not be used to locate the archived weights.

The [verified manifest](QA_CHECKPOINT_ARCHIVE_27B32K_FINAL.json) records
**19 payload/metadata files, 2,498,907,068 bytes**. CPU-only ORCD job **23497343**
completed source hashing and staging in 16 seconds. All five model/config payload
hashes and sizes matched the compute receipt after transfer; original result,
completion, configuration, material IDs/hash and final-panel records are included.
Every object passed create-only S3 PUT and returned SHA256/size verification.
The 10,941-byte manifest was verified last, with SHA256
`c52d8cbf140dbf491273ea1739119d14bd81d9746ae998a4b8d8a9d9f2356666`.

Restore `ac/` and `reader/` together with separately available base models.
Optimizer state, intermediate checkpoints, base weights and the full corpus are
excluded. The archive supports inference or fresh-optimizer training; it is not
an exact optimizer-resume bundle. No additional GPU restoration was performed;
the independent final answer readout was pending. Existing archives remain
unchanged. Original and local staging pairs stay protected, with a
[completion receipt](../../TMP/FIRST_TASK_AWARE_FINETUNING/qa_checkpoint_archive/oracle_27b_32k/epoch_001/archive_receipt.json).


## Constant-floor continuation final archived — 2026-09-23

The important final **32,000 questions × 1 + 10,000 fresh questions × 1** pair is
verified at `s3://general-bucket-aln/activation-context-data/checkpoints/taskaware-20260923/continue32k_10k_final_lr/epoch_001/`.
It retains the original P2 → completed 32k ancestry, Qwen3.5-2B encoder → 9B reader,
rank-128 adapters and 1/16 compression. This 10k stage inherited Adam and used constant
encoder/heads LR 5e-6, reader LR 2e-6. Stage clock 0→1250 is distinct from the inherited
Adam clocks starting 4000 (expected final 5250). It is not a 42k schedule planned from scratch.

Final held 400 answer NLL improved **0.5688311 → 0.5326864**. The complete 100-answer
source review found **71/87** correct versus the parent’s 63/87, using the unchanged
supported-reference denominator; all 100 examples remain in the review. These are
repeated development observations, not independent confirmation. Only the completed
final pair is archived; no intermediate best-panel weights were substituted.

The [verified manifest](QA_CHECKPOINT_ARCHIVE_CONTINUATION_FINAL_LR.json) contains
**26 payload/provenance objects, 2,493,427,939 bytes**. All five model/config hashes
and pinned completion/source/config records matched completed readout 23511288;
training 23511287 completed the full 10,000-question stage. Guarded direct rsync and
local hashing required no new CPU/GPU job. Every object passed create-only S3 PUT
and returned SHA256/size checks; the 12,532-byte manifest was verified last:
`922d272455a42fb29df6d3c9cbba3396ab8c8aa93b8b4992dafb9296d03a1fa2`.

Restore `ac/` and `reader/` together with separately available base models. Optimizer
state remains untouched on existing storage and is excluded from S3, along with
base weights, corpus and intermediate checkpoints. This archive supports inference
or fresh-optimizer fine-tuning, not exact Adam resume. Final remote/local pairs and
parent ancestry are preserved; existing archives are unchanged. The
[completion receipt](QA_CHECKPOINT_ARCHIVE_CONTINUATION_FINAL_LR_RECEIPT.json)
records the verified manifest, transfers and preservation flags.


## Final long-context adaptation archived — 2026-09-23

The important final **32,000 parent questions + 8,000 fresh questions ×1 pass** pair is
verified at `s3://general-bucket-aln/activation-context-data/checkpoints/taskaware-20260923/strong32k_piecewise_8k_8k/epoch_001/`. It has **40,000 lifetime unique questions**, original
P2 → completed 32k ancestry, a Qwen3.5-2B encoder → 9B reader, rank-128 adapters and
1/16 compression. The 8k stage trained with independent passages and **fresh Adam**;
it did not inherit the parent optimizer. It used encoder/heads LR 2e-5 and reader
LR 8e-6, 50-update warmup and cosine decay to 25%. The preselected final paired
`epoch_001` completed the full 8,000-question stage.

Source review gives **65/87 correct at 8k** and **60/87 at 32k**, versus the original
32k parent's 58/87 at 8k. All 100 questions remain visible, with the fixed 87/13
reference split; these are descriptive repeated-development results. This is the
best evaluated long-context final in the current campaign, not an independent
confirmation or universal-quality claim. Its 8,000 fresh questions overlap the
native 10,000-question continuation pool by design.

Full 400-question development AC NLL is **0.5968172** at 8k, with exact final-panel
replay. Selected-100 NLL is **0.6347504** at 8k and **0.6497265** at actual 32k;
the latter has no 400-question score. Both completed readouts use the same final
five model/config hashes. Own-reader full/no-context controls are NLL-only.

The [verified manifest](QA_CHECKPOINT_ARCHIVE_STRONG_LONG_FINAL.json) contains
**25 payload/provenance objects, 2,494,885,382 bytes**. CPU-only ORCD
job **23547063** performed the scoped source hashing/copying; local transfer checks
matched every pinned hash. All S3 objects passed create-only PUT and server SHA256/
size verification. The manifest was verified last, with SHA256
`38c38f19ca6b0dd03b3920ede938771b87456c051cb836d34c43075892b88664`.

Restore `ac/` and `reader/` together with separately available bases. Optimizer
state remains untouched on original storage and is excluded from S3, along with
base weights, corpus and intermediate saves. This supports inference or fresh-
optimizer fine-tuning, not exact Adam resume. Existing archives and preservation
flags remain unchanged; no pruning or GPU job was performed. The
[completion receipt](QA_CHECKPOINT_ARCHIVE_STRONG_LONG_FINAL_RECEIPT.json) records
staging, transfer and S3 verification. The [900-answer report](STRONG_LONG_COHORT100.tressoir.html)
retains the full source review.
