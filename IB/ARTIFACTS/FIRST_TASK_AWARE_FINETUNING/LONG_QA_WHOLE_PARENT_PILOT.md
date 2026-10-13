# Whole-parent question synthesis pilot

The 9B pilot completed: **69 accepted questions from 75 parents and 221 candidates**, in 7m53s of job wall time. It resolves the obvious temporary-label problem, but it does **not** yet establish reliable multi-section supervision. Run a bounded 27B replay of the same 75 parents before scaling or expanding sources.

The generator saw complete natural parents, selected two or three exact support units, and produced at most three questions per parent. One checker judged premises, grounding, combined-evidence necessity and self-containment. Both passes used thinking off and presence penalty zero. Source spans and complete long context remain separately portable. Earlier arbitrary-span generation produced 17 accepted questions from 748 parents, including 10 label-dependent pairs; these are different attempts, not a controlled model comparison.

## Yield and rejections

| Domain | ~8k parents | ~16k parents | ~32k parents |
|---|---:|---:|---:|
| Code | 7 | 5 | 7 |
| Manuals | 2 | 1 | 0 |
| Legal | 7 | 11 | 11 |
| Math/scientific papers | 6 | 9 | 3 |
| Office | 0 | 0 | 0 |

Entries are accepted questions, from five parents per cell. In total, 38 parents yielded at least one acceptance. Actual natural lengths vary within each band; documents were not padded. Sixty parents are intact documents, while the 15 Office sources are explicitly partial, coherent reference dossiers.

Of 221 candidates, 20 failed evidence-index checks: **19 repeated the same index twice**, including 16 Office candidates; one code candidate used an invalid index. The remaining 201 reached the checker: 69 accepted, 132 rejected. Every semantic rejection had `needs_multiple_sections=false`; 115 failed only that flag. Seventeen also failed grounding, including 15 that also failed self-containment. These are checker decisions, not independently established rejection correctness. One of 75 generator outputs reached its token limit and was malformed. There were no temporary-label rejections, and a cheap scan found none of the named temporary-label patterns in all 69 accepted pairs.

Measured model-call time was 177.4s for generation plus 44.3s for checking; startup and other overhead account for the rest of job time. Do not extrapolate this tiny, domain-dependent yield into a large-bank production rate.

## Fixed 15-example manual check

Selected before quality review: smallest seeded question-ID hash in each of the 11 nonempty domain/length cells, plus four rows from unused parent groups using the same ordering. Four requested cells had no accepted examples. Exact selection, portable rows and aggregate receipts are in `IB/TMP/FIRST_TASK_AWARE_FINETUNING/long_qa_whole_parent_v1_review/`; full IDs and detailed evidence are in [LONG_QA_WHOLE_PARENT_PILOT_REVIEW.json](LONG_QA_WHOLE_PARENT_PILOT_REVIEW.json).

All 15 selected support spans exactly match their offsets in the complete parent. No sampled question or answer depends on temporary labels. Seven samples have a natural relationship between support units, but four of those have answer/support problems; seven can be answered from one cited unit, and one merely splits a short list at a chunk boundary. This is a descriptive sample, not a population quality estimate.

| ID suffix | Domain/band | Finding |
|---|---|---|
| `fb99d4323c1217230d02fe35` | Code/8k | Grounded caller/helper relationship: TimeDistributed handling plus inner-configuration extraction. |
| `27598e9d337932056bd72e9c` | Code/16k | **Incorrect conditions.** Swin tests `index+1`, and its final stage receives an extra normalization even when the user flag is false. The answer misses both. |
| `a34a41d5cd18e83823a51664` | Code/32k | Presence-check explanation overgeneralizes: required primitive fields are treated as present unconditionally. Additional citation mostly repeats the same comparison pattern. |
| `6aa597b12ca18a7c0b2da320` | Manuals/8k | Grounded LeakCanary options, but two units simply divide a short numbered list. No substantive distant dependency. |
| `81f94324ac3b8311c9728288` | Manuals/16k | Grounded comparison of speaker embedding versus similarity-score inputs. Two useful passages; much intervening context is printed embedding vectors. |
| `ef3ffca2979b863459dbd544` | Legal/8k | Douglas disclosure/due-diligence answer grounded; unit3 alone contains the full answer. |
| `d26fa755854364d1a98e1a71` | Legal/16k | Tax limitations/constitutional-privilege answer grounded; unit1 alone states the complete rationale. |
| `a4c533a286d9cc30a2ac5934` | Legal/32k | Natural NGA/NGPA relationship, but the WDA conclusion is absent from chosen support. It appears later in the complete parent. |
| `02289f3dfd15e2dd40107716` | Math/8k | **Incorrect suppression claim.** Source gives a square-root relation for the extra-dimensional expansion rate, not exponential suppression. |
| `286481d093a1357757a8594c` | Math/16k | Grounded, related comparison: strong-coupling visibility versus weak-coupling preservation of near-maximal concurrence. |
| `476705ffc7a942e822a8006f` | Math/32k | Complexity exponents supported conditionally, but “degree-4 torsion subgroup” is unsupported; theorem assumptions are omitted. |
| `7d35861ad177fd314686d225` | Code/16k | Byte Buddy redefine/rebase answer grounded; unit5 already contains both definitions. |
| `ae81593c84bb290b4a81072b` | Math/8k | Natural spectrum/intermittency comparison, but “depletion across all scales” overstates the source's modulation/energy-transfer result. |
| `9c0d9a2acfae9c2c432cfc62` | Legal/32k | Procedural explanation supported for this proceeding; question is broader than the opinion's qualifications. Unit6 alone supplies the explanation. |
| `f2d3063dcf5b92709227b5d1` | Legal/32k | Sale-intention/materiality result supported; “risk of liquidation” is imprecise. Source says disclosure reduces perceived risk of retaining shares and reveals liquidation profits. Unit10 alone suffices. |

For the Lovelock error, the exact support states `beta = ±sqrt(-(kappa1^-1 + kappa2^-1 alpha^2)/(10 kappa3^-1 alpha^2))`. Its exponential statement concerns growth of the **large dimensions**. For Swin, the missing interaction is between `_extra_norm(index)` using `i=index+1` and the caller forcing the final stage's flag. These are concrete failures for the stronger-model replay to probe.

## Next bounded step

Replay the **same 75 parents, same prompts and decoding settings on pinned 27B**, with no thinking. Preserve the 9B outputs and evaluate the two attempts separately. This tests whether stronger generation/checking improves support selection and exact reasoning without confounding a new source pool or prompt.

Do not force Office quotas: its zero yield and duplicate indices suggest these reference dossiers rarely need combined evidence under this recipe. Likewise, adjacent split lists and large numeric dumps must not be presented as demonstrated long-range reasoning. If the stronger-model replay is useful, process existing unused parents in interruptible batches, keeping parent-group splits and the actual achieved domain mix visible. No new validation framework or source-expansion campaign is warranted yet.

The completed 9B input is `long_qa_whole_parent_v1_pilot/parents_75.jsonl` under task TMP; SHA256 `7e3941fde695ddb3ba75151b1d9fc8cdfae4f789d99e57358b4467d6228565f6`. Worker SHA256 `0738c0744212e1911af5a000f472e635e7e24577794f628b0fd14cae4b318695`. The actual 9B tokenizer measured 6,495–32,228 generator prompt tokens; the 40,960-token context fits all parents plus the 1,024-token output allowance. Reports, accepted batches and raw generator/checker attempt receipts are synced at `IB/TMP/SYNC/TASK_AWARE_QA/long_qa_whole_parent_v1_pilot/`. Production workers and completed records were not modified during this review.

## 27B replay prepared

Separate immutable sibling `bench/generate_long_qa_whole_parent_27b_v1.py` preserves the completed 9B worker. It requires the prefetched local snapshot at revision `fc05daec18b0a78c049392ed2e771dde82bdf654`, uses dynamic FP8, context 40,960, max two sequences and prefill budget 8,192. The same generator/checker prompts, schemas, seed rule, output limits and temperatures are retained. Explicit top-p0.8/top-k20/min-p0 reproduce the 9B harness defaults that a local 27B path would otherwise lose. Both passes keep thinking off and presence penalty zero.

CPU checks used the **actual pinned 27B tokenizer**, copied without weights, on all 75 unchanged parents. Largest prompt plus output is 33,252 tokens. Fake-engine checks covered the pinned model path/configuration, both decoding recipes, exact-span publication, rejection, completed resume and changed-batch detection. Python compile and shell syntax passed. This is preparation evidence, not 27B semantic or GPU evidence.

Deployed job: `~/orcd/pool/activation/taskaware/jobs/long_qa_whole_parent_27b_v1_pilot.sbatch`: one RTX PRO6000, 45-minute hard bound, 2,400-second worker budget, two-parent batch publication, no automatic requeue. The root coordinator may place it after the throughput diagnostic, respecting the four-GPU cap and paired-training priority; **this preparation did not submit it**. Inputs and hash receipts are in task TMP `long_qa_whole_parent_27b_v1_pilot/`. New results will sync to `IB/TMP/SYNC/TASK_AWARE_QA/long_qa_whole_parent_27b_v1_pilot/`.

Root queued the same-parent 27B replay as job23431684 after checker-throughput job23429488; both exclude repeatedly preempted node5101. No additional GPU slot is requested beyond the existing sequential experiment slot.

The 27B replay completed on September 22 at 00:16:46 UTC: 73 accepted questions from 207 candidates and 75 parents. The fixed matched-parent review is recorded in [LONG_QA_WHOLE_PARENT_27B_REVIEW.md](LONG_QA_WHOLE_PARENT_27B_REVIEW.md). Temporary labels remained absent, but 11 of 15 sampled accepted pairs were answerable from one cited unit; stronger generation alone did not establish reliable multi-section supervision. One accepted Lovelock answer also contains an incorrect equation and JSON-decoded control characters. The original outputs remain unchanged.
