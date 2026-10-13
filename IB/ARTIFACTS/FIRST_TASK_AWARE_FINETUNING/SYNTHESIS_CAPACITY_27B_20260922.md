# 27B synthesis: source capacity

**The existing sources cover 36k and 64k published questions, but do not cover 200k at current yields.** Web runs short first; code, technical web and office also need extensions if acceptance rates persist. This is separate from the earlier 9B capacity result.

Snapshot: **22 September 2026, 10:09:06 UTC**; 32,000 balanced questions published and 32,554 accepted, including reusable surplus. The resumed producer retains its original source order, nine-family mix and grounding checks.

| Family | Raw windows remaining | More accepted questions needed for 200k | Lifetime / recent accepted per window | Extra windows needed at lower rate |
|---|---:|---:|---:|---:|
| code | 59,818 | 50,312 | 0.546 / 0.556 | 32,248 |
| math_arxiv | 78,815 | 20,118 | 1.201 / 1.198 | 0 |
| math_textbook | 22,445 | 13,395 | 0.814 / 0.813 | 0 |
| financial | 21,107 | 8,352 | 0.920 / 0.920 | 0 |
| web | 64,792 | 33,536 | 0.214 / 0.219 | 91,931 |
| technical_web | 13,487 | 8,369 | 0.344 / 0.343 | 10,904 |
| documents | 24,269 | 13,346 | 1.296 / 1.296 | 0 |
| legal | 29,153 | 9,963 | 1.591 / 1.591 | 0 |
| office | 16,313 | 10,055 | 0.579 / 0.575 | 1,172 |

Recent rates use the last 50 committed microchunks per family (1,280–3,200 windows; exact counts in JSON). These are observed throughput yields, not independent accuracy estimates. Counts credit committed surplus. Extrapolation assumes future source segments behave similarly.

For 36k and 64k, every family has sufficient raw capacity at both rates; the tightest 64k margin is **2.19×** the remaining accepted-question requirement. This does **not** certify 32k eligible training items after held-out request exclusions: the preparer must still gate exact eligible family counts.

For 200k, unchanged pools project balanced ceilings around **102–103k for web**,125k for technical web,141–143k for code and189k for office. Code alone needs about 50,312 more accepted questions from 59,818 windows: required yield 0.841/window versus 0.547 lifetime and 0.556 recent. The 9B bank's higher acceptance rate cannot be substituted here; the pipelines differ.

## Smallest practical extension proposal

Use the existing immutable family-extension files; preserve consumed source IDs/cursors, quotas and checking behavior. Round the projected shortages up modestly:

- **Web +110k raw windows:** continue the pinned MS MARCO raw-passage extractor beyond its old cap, reading only passage text.
- **Code +40k:** more distinct repository files with the existing extractor. The prior extension delivered 14,339 against a 24k cap, so another 40k is not proven available in the current cache; additional repository snapshots may be needed.
- **Technical web +15k:** continue the pinned raw-document subjects beyond their per-subject caps, retaining source identities.
- **Office +4k:** preserve 60/30/10 automation/business/procedural help (about 2,400/1,200/400 windows). Use remaining automation material and replenish the limiting business/help sources.

Prepare web first, then code/technical web; office is a smaller later top-up. These additions are now prepared and validated in a separate staging folder; see the completion receipt below. The initial capacity analysis did not acquire data. Recheck at 64k, and prepare additions before committing beyond about 90k published questions. Immediate 36k/64k production can proceed without these additions.

Provenance: [machine capacity receipt](SYNTHESIS_CAPACITY_27B_20260922.json), immutable local manifest snapshot `TMP/FIRST_TASK_AWARE_FINETUNING/storage_retention/synthesis_bank_27b_capacity_manifest_20260922.json` (SHA256 `c09848ffbb0e1e588f37e0e121f0685512ffda135343ba805071f279203846f4`), registered source-count receipt and all eight prior extension manifests. Counts/cursors/receipts and registered extension hashes agree. This was local metadata analysis only: no remote work, jobs, pruning, bank/source edits, question searches or overlap audit.

## Authorized extension preparation — complete and staged

**169,000 new raw windows (434,383,803 bytes) are ready**, with the requested exact counts:

| Family | New windows | SHA256 |
|---|---:|---|
| Web | 110,000 | `2ce75752f1ea496dce2004280f9caacc8c73cfb18472a4824cf6b475ded88df1` |
| Code | 40,000 | `44063dd854ee690fea2ddb3d1aa2c923607c7028666bab98bd84067aa5362300` |
| Technical web | 15,000 | `6a0858730984691de1755a39eff9c465a93fb593690893e231e947710f94d2b7` |
| Office | 4,000 | `4fcab6ce5e7fde890a28bb099d3b2d2758b05418c8542df05443aa80fe87f0a0` |

[Full completion/provenance receipt](RAW_SOURCE_EXTENSIONS_27B_20260922.json).
Remote staged files are under
`~/orcd/pool/activation/taskaware/bank_source_extensions_27b_20260922/published/`;
`completion.json` is the final combined receipt. Filenames are
`<family>__raw_27b_headroom_20260922_v2.jsonl`.

ORCD CPU job **23470299** used four CPUs and finished in **46 seconds**. Web,
code and technical web passed validation; the job exited 1 because the office
fallback repository was unavailable. Those successful files were kept unchanged.
The office recovery completed locally in 25.7 seconds using pinned Microsoft 365
administration documentation for the missing procedural help; no second Slurm
job was submitted. Its validated file was transferred through guarded rsync,
then published with a create-only hard link. The original failed-job receipt
remains for provenance; use `completion.json` for the completed four-file state.

The existing Pool/window helpers, source format and pinned raw-corpus extractors
were reused. Each file passed complete row parsing, exact count, content/file
SHA256, source provenance and uniqueness checks against the base, all eight
prior extensions and the other new files. Office is exactly **2,400 automation,
1,200 business, 400 procedural-help windows** in the existing interleave.
Code spans 12 additional repository snapshots pinned to commit SHAs. Technical
web adds 5,000 windows each from the existing robotics, economics and psychology
raw corpora; detailed app/topic distributions are in the receipt. Longer raw
business documents contributed previously unselected sections with original
document IDs retained; previously used exact windows were skipped. No native
question/answer fields or synthesis question banks were searched.

**Activated at 10:35 UTC by CPU job **23470885****, completed **0:0 in 6 seconds**.
All four JSONLs now have create-only hard links in the existing watched
`~/orcd/pool/activation/taskaware/bank_source_extensions/` directory. Full SHA256
and byte counts matched the completion receipt before and after linking;
all eight existing extension files retained their hashes, sizes and inodes.
[Activation receipt](RAW_SOURCE_EXTENSIONS_27B_ACTIVATION_20260922.json).

The producer was not interrupted and its code, bank and cursors were not edited.
At activation it had 34k published questions and had not yet registered the new
files. It will discover them at its next startup or source exhaustion, appending
after its existing extension order. No further acquisition is needed now.

At the earlier lower lifetime/recent 27B yields, available capacity projects at least
**214k balanced questions**, limited by code. This is an estimate: acceptance
on new repositories/topics can differ. Recheck at 64k published. No GPU allocation,
S3 write, pruning or training-queue change occurred. The staging root has PRESERVE.
