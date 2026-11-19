# Math synthesis recipe (§3.4): summary, 2026-11-19

The model is a 2B encoder (Qwen3.5-2B, Hidden Residuals) feeding a frozen Qwen3.5-9B reader. The reader reads the three solver runs
of a DeepMath problem through AC parts at compression r = 0.1 (r10) or 0.2 (r5) and writes the synthesis. Training data: 1,500
training problems with 3 solver runs each. Evaluation: dev = 200 problems; dev_hard = 180 fresh problems at difficulty ≥ 8.5 (80 at d8.5,
100 at ≥ 9; never trained on). Test sets dmext / dmhard are untouched until the user reviews.

Every number below is read from a file. Logs are in `math/RECIPE_RUNS.md`. The tables come from `recipe_summary_tables.py`, whose
output is `scores/recipe_summary_tables.md` (with CIs per band). GPU-h is allocated ORCD RTX PRO 6000 time from `sacct`, including
preempted segments.

## Pipeline: recon → stage 1 → stage 2 (hard8, random correct sample at ≥ 9)

Shared settings: reader frozen (LR 0), encoder LR 5e-5, warm-up 50, cosine, batch 8, one pass, 72,000-token item guard.

| stage | data | loss | updates | GPU-h |
|---|---|---|---|---|
| teacher samples | Untuned 9B reading the full runs (teacher-test prompt). Train: seeds 0–2, 4,500 samples. Dev seed 0: 200 proxy targets | — | — | 4.71 + 0.37 |
| recon (`M_lcrec`) | Reconstruct each training run from its AC part; r ∈ {0.1, 0.2} per run; 4,483 items | SFT | 560 | 4.03 (shared by both ratios) |
| stage 1 (`M_s1_r10`, `M_s1_r5c`) | Every stopped teacher sample; 4,431 items, 1,478 problems; student = AC view, teacher = full runs | KL | 553 | r10 10.23; r5 11.34 (+3.58 for two abandoned r5 starts) |
| hard8 samples | 5 more teacher samples (seeds 3–7) for the 1,063 training problems at ≥ 8.5; 5,315 samples | — | — | 5.46 |
| stage 2 (`M_s2hr_r10`, `M_s2hr_r5`) | Correct teacher samples only (teacher-test scorer). Below 8.5: the shortest correct of 3 plus any within 1.3×. d8.5: the same rule over 8. ≥ 9: k correct samples drawn at random from all 8 (k = size of the within-1.3× set, seeded by task id). 3,187 items, 1,275 problems; init = own stage-1 final | SFT | 398 | r10 4.91; r5 5.68 |

Final dev proxy, gap closed: stage 1 r10 0.816 / r5 0.871; stage 2 recipe r10 0.918 / r5 0.975.
On dev_hard ≥ 9, stage 2 cuts generated tokens (median) from 2,947 to 1,629 at r10 (−45 %) and from 3,129 to 2,227 at r5 (−29 %).

## Results: accuracy %, 3 seeds per problem

"excl." = sensitivity, label-mismatch problems excluded (see the caveat below): dev 187, dev_hard 171 (d8.5 76, ≥ 9 95).

| arm | dev | dev excl. | dev_hard all | d8.5 | ≥ 9 | dev_hard all excl. | d8.5 excl. | ≥ 9 excl. |
|---|---|---|---|---|---|---|---|---|
| single solver | 77.83 | 83.24 | 71.85 | 74.58 | 69.67 | 75.44 | 78.51 | 72.98 |
| majority of 3 | 81.00 | 86.63 | 73.33 | 76.25 | 71.00 | 77.19 | 80.26 | 74.74 |
| 9B results-only | 81.83 | 87.52 | 76.11 | 80.83 | 72.33 | 80.12 | 85.09 | 76.14 |
| 9B full runs | 82.33 | 88.06 | 77.78 | 80.83 | 75.33 | 81.68 | 85.09 | 78.95 |
| AC r10 stage 1 | 83.00 | 88.77 | 76.85 | 80.83 | 73.67 | 80.31 | 85.09 | 76.49 |
| **AC r10 stage 2 (recipe)** | 81.83 | 87.52 | 77.41 | 81.67 | 74.00 | 80.90 | 85.96 | 76.84 |
| AC r5 stage 1 | 82.67 | 88.41 | 77.59 | 81.25 | 74.67 | 81.29 | 85.53 | 77.89 |
| **AC r5 stage 2 (recipe)** | 82.33 | 88.06 | 77.04 | 80.00 | 74.67 | 80.51 | 84.21 | 77.54 |

Paired diffs (95 % bootstrap CI; sign-test p); none is significant:
- **Stage 2 recipe − own stage 1:**
  - dev: r10 −1.17 (−3.00, +0.33; p 0.17), r5 −0.33 (−1.83, +1.17; p 0.80)
  - dev_hard ≥ 9: r10 +0.33 (−3.00, +3.67; p 1.00), r5 +0.00 (−2.33, +2.33; p 1.00)
  - ≥ 9 excl.: r10 +0.35, r5 −0.35
- **Recipe − results-only at ≥ 9:** r10 +1.67 (−2.67, +6.00; p 0.46), r5 +2.33 (−2.33, +7.00; p 0.30).
- **9B full runs − results-only at ≥ 9:** +3.00 (−1.33, +7.67; p 0.16).
- **Goal not met:** AC is not clearly above results-only at ≥ 9. The AC arms sit between results-only and the full-runs teacher, with
  far fewer generated tokens: about 1.6k–3.1k vs results-only's 19.8k (median at ≥ 9).

## Tried and dropped

- **27B teacher:** on dev_hard ≥ 9, 27B − 9B full runs (same teacher-test path) was +0.67 (−4.67, +5.67; p 0.88). The bar for making
  27B targets was +3 with p < 0.1, so no 27B targets.
- **3-sample stage 2** (shortest correct of 3 plus within 1.3×; 2,229 items): it shortened the syntheses but lost about 1 pt at ≥ 9
  (r10 72.67 vs stage 1 73.67; r5 73.33 vs 74.67).
- **Shortest rule on the 8 samples (hard8 shortest):** at r5 it loses at ≥ 9: 71.33 vs stage 1 74.67 (−3.33, CI −6.33 to −0.67, p 0.08).
  Random − shortest is +3.33 (+0.33, +6.33; p 0.053). At r10 the two tie (74.00 each).
- **Rationalization (STaR-style):** 170 train problems at ≥ 8.5 with ≤ 1 correct of 8; 680 samples with the gold answer in the prompt.
  444 missed the gold and 199 failed the leak filter. That left 5 usable non-binary items on 4 problems (binary golds were excluded on
  purpose), so it was not trained.

## Caveat: label mismatch

Many DeepMath problems ask a yes/no question ("determine whether …") but have a non-binary gold (`f(x) = x`, `0`, `e^{O(N)}`). The
solvers and the teacher answer yes/no and are scored wrong.
- **Rule** (`scores/labelmismatch.json`): the gold is not Yes/No/True/False, and ≥ 2 of the 3 solver answers are.
- **Size:** 9 of 180 dev_hard problems (d8.5 4, d9 5; 8 have no correct solver run) and 13 of 200 dev problems (none with a correct
  solver run). Among the 91 training problems at ≥ 8.5 with ≤ 1 correct of 8 teacher samples and a non-binary gold, the teacher answers
  yes/no in ≥ 5 of 8 samples for 69.
- **Effect:** excluding them raises every arm by about 3–6 pts but leaves the gaps between arms nearly unchanged.
- **Status:** standard scoring is unchanged; the user decides.
