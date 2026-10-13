# AC execution performance — work in progress

## Measured starting point

The original L40S run presented eight questions sequentially per optimizer update. A measured40.35seconds/update is714questions/hour:11.2traininghours for8,000questions×1epoch, before preparation/panels. The actual frozen8k set averages14,080source tokens,14,960encoder positions including rows,1,332reader tokens,451no-history tokens and43supervised targets. Median source is11,527 and90th percentile31,935. This is a long-history workload, despite its short answers.

## How far from peak?

[NVIDIA specifies](https://www.nvidia.com/en-in/data-center/l40s/)362TFLOP/s denseBF16 forL40S (the larger advertised number uses sparsity). A rough operation count for this workload is200–300TFLOP/question: frozen-base forward/input-backward/checkpoint recomputation dominate, plus full-attention and trained adapters/readouts. Dividing by dense peak gives0.55–0.83seconds/question, versus about5.04seconds measured. This implies an approximate arithmetic lower bound of1.2–1.8hours/8k and roughly11–16% of that dense peak.

This is an estimate, not measured model-FLOP utilization or a feasible wall-time promise. Mixed dtypes, sequential layers/recurrences, memory traffic, checkpointing and small matrix shapes prevent peak arithmetic from being sustained. The gap is large enough to justify tuning; it does not establish a10× attainable speedup.

## Changes under test

- Separate encoder batch size, reader batch size and examples per backward graph.
- Reuse complete differentiable encodings inside one logical update when histories are identical.
- Measure checkpoint thresholds and CPU/GPU Adam-moment placement together with real allocated moments.
- Remove repeated intervention-position GPU synchronizations and unused per-layer diagnostic norms.
- Optional shorter-history curriculum, only if optimized execution on larger GPUs still misses useful turnaround. Preserve the original history distribution when feasible. The near4k/max8k producer is implemented but held; existing synthesis continues. Existing answers are never clipped beneath their evidence.

Execution policies retain the eight-question logical update, per-item means/weights, actual partial-batch denominator and preassigned penalty draws. Original held100 stays fixed. Shorter histories are explicitly a data-recipe change, separate from execution speedups.

## Validation and queue

CPU: 54 intervention tests, 24 execution/weighted-gradient/Adam tests, and tuner checks passed. Identical frozen 32-question fixtures include a short profile (median 4,073/max 8,099 source tokens) and original long histories including distinct 32k cases. Native baseline/repeat and both optimizer gradient groups must pass before any candidate can be selected; a winner then gets a real paired save/resume validation in fresh processes.

## First measured results (Sep24, 16:12 UTC)

| Same short fixture | Starting seconds/question | Fastest passing choice | Gain | Projected 8,000 questions × 1 epoch |
|---|---:|---:|---:|---:|
| L40S | 2.432 | 1.422 | 1.71× | 3.16 training hours |
| RTX PRO 6000 96GB | 2.435 | 0.835 | 2.92× | 1.85 training hours |

Both winners keep singleton readers and ordinary eight-question accumulation with CPU Adam moments. L40S skips reader checkpointing below 4,096 tokens. RTX additionally skips encoder checkpointing below 8,192 tokens; that choice exceeds L40S memory. Loss and both gradient groups pass all three measured cases. Full trainer save/fresh-process-resume subsequently passed on both devices. Projections exclude preparation, reporting and saves and come from demanding small fixtures, not a complete population benchmark.

Reader batches of two/four fail the unchanged gradient gate on both devices, despite close losses. They are not promoted. More aggressive uncheckpointed batches run out of memory. The fixture has more repeated histories than globally shuffled 8k training: expected within-update reuse there saves only about 0.13% of encoder calls. Current winners do not rely on reuse.

A separate real-PEFT CPU investigation found repeated full-model adapter setup inside every checkpointed layer. The guarded fix removes the redundant original-forward scope and avoids redundant adapter setters while preserving recomputation, live trainability and nested/disabled/merged behavior. All77 focused CPU checks passed.

Direct GPU A/B, with both models fully checkpointed and every other setting held fixed:

| Same eight questions | Before | After | Gain |
|---|---:|---:|---:|
| RTX PRO, short histories |19.22s|8.88s|2.16×|
| RTX PRO, distinct32k histories |44.85s|40.42s|1.11×|
| L40S, short histories |18.44s|10.59s|1.74×|
| L40S, distinct32k histories |78.59s|76.89s|1.02×|

Loss and both gradient groups pass unchanged gates in all four comparisons. Long histories still spend most time in model computation. These scope-only gains must not be multiplied blindly by the earlier checkpoint-policy gains; combined measurements are reported below. Native train/save/fresh-process-resume passed on both devices for the earlier selected policies.

The first reader diagnostic localizes a substantial shape effect: padding one reader from645 to729 positions produces roughly53% input-row gradient difference, while grouped output-head projection is effectively exact. CPU hybrid-model checks pass. Later bounded probes are recorded below; larger reader batches still fail the numerical gate.

Original-history probes have completed on RTX PRO and CSAIL H200. Live job state and current compute authorization belong in STATE.md; this findings document does not authorize or schedule nodes.

The combined readable report is `IB/TMP/SYNC/TRAJECTORY_COMPACTION/performance/report.tressoir.html`. Individual receipts synchronize under `{orcd,csail}/performance_*/{short,long}/`; RTX uses `performance_rtx_v1/` and `performance_rtx_v2/` beneath the same overview folder. Repeated refreshes change JSON only; HTML bytes and modification time were verified unchanged. Combined adapter-fix RTX measurements are complete:0.818seconds/question for the short fixture and4.513 for the long stress fixture. Completed H200 equivalents are0.771 and3.005seconds/question. These separate fixtures are not the original-distribution batch timings below.

## Original history distribution retained

The combined adapter fix and checkpoint policy reaches0.818seconds/question on the short RTX fixture (prior starting point2.435), with all numerical gates passing. The effects overlap; their isolated speedups cannot be multiplied. The original frozen8k workload was measured separately using its median and90th-percentile logical batches, plus a separate eight-history32k stress case. Stress timing is excluded from whole-pass planning estimates. Its source histories average14,080tokens; the32k stress fixture is not representative of every question.

The tuner now checks capacity just below checkpoint thresholds as well as the longest sequences. Final partial graph/reader batches have separate coverage. All25 CPU checks pass, and current8,000-example case selection matches frozen v3 exactly. RTX and H200 singleton winners now pass their three relevant GPU capacity cases as well as all three timing cases. Generic final partial graph/reader handling remains CPU-tested; the original8k set has no final logical remainder. The initializer receipts cannot be relabeled as validation of a different resumed checkpoint.

The follow-up RTX padding probe completed: turning TF32 off reduces, but does not eliminate, shape-dependent gradient differences. Padded singleton and batched-reader choices still fail the same gates. At fixed physical width, replacing future padding with random embeddings leaves prefix forward states bitwise equal. With random future embeddings the maximum future gradient is2.03e-12; zero padding instead has maximum4.56e-4 and L2.00762. Thus prefix forward invariance passes, but the backward residual at zero padding is unresolved. The first forward divergence when changing length is the first layer’s QKV projection (relativeL2≈0.00367), before its attention calculation; this points toward shape-dependent arithmetic rather than substantive future-token leakage on this fixture. The original-context8k run has therefore resumed with singleton readers, checkpointing and the validated adapter-scope fix; its optimizer and learning-rate schedule are retained.

Final original8k receipts include timing and policy-specific capacity:

| Device | Encoder / reader checkpoint threshold | Median batch seconds/question | 90th-percentile batch | Distinct32k stress | Peak memory including capacity |
|---|---|---:|---:|---:|---:|
| RTX PRO |8,192 /4,096 positions|2.123|2.771|4.915|69.51GiB|
| H200 |16,384 /4,096 positions|1.582|1.859|3.300|104.69GiB|

Both winners use singleton readers, no encoding reuse and CPU Adam moments. Each passes three timing and three relevant checkpoint-boundary capacity cases under the unchanged loss/both-gradient/memory gates. The median/p90 cases imply4.72/6.16traininghours on RTX and3.52/4.13 on H200 for8k questions as planning references, not measured population means or statistical intervals. Preparation, reporting and checkpoint time are additional. Full receipts: [RTX](../../TMP/SYNC/TRAJECTORY_COMPACTION/performance_original_qa_rtx_v2/result.json), [H200](../../TMP/SYNC/TRAJECTORY_COMPACTION/csail/performance_original_qa_h200_v3/result.json).

The precision investigation stops here without lowering the normal numerical gates. A separately labeled B1/B2 learning comparison is queued for512 identical fresh questions from the same paired checkpoint, optimizer, RNG and original schedule. Both hold graph size2, singleton encoder calls, full checkpointing and CPU Adam fixed. That experiment may measure a useful learning/speed tradeoff despite failed numerical equivalence; no learning result or execution-policy approval is claimed yet.
