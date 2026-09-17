# RAG AC ablation — cross-arm results (2026-09-15)

Qwen3.5-4B reads an NQ passage compressed 1/16 into activation-context rows (AC student); the teacher is the same reader with the full passage, frozen at training start through stored top-256 targets; the control is the same reader with no passage. Every arm: 2816 training items × 2 epochs (704 updates of 8), 100 held-out items per panel, the same material, seed, initial adapters (hash `c2f0d3…`) and k. One arm per L40S GPU, one wave. Gold NLL is nats per target token of the gold answer call (lower is better); answer NLL restricts to the answer value tokens; exact KL is the full-vocabulary KL to the training-start teacher; EM is exact match after SQuAD normalization on greedy completions. Final exact controls over the 100 held items after the last epoch.

| arm | items×epochs | updates | gold NLL student | gold NLL teacher | gold NLL no ctx | gold NLL full text (trained reader) | answer NLL student | answer NLL teacher | answer NLL no ctx | exact KL student | exact KL no ctx | exact KL full text drift | EM student | EM teacher | train loss final | coverage | minutes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S · smoke (1e-4 / 2e-5, KL) | 256×2 | 64 | 0.575 | 0.132 | 0.571 | 0.117 | 1.905 | 0.364 | 1.886 | 0.467 | 0.467 | 0.011 | 0.042 | 0.500 |  | 1.000 | 21.8 |
| A · reference (1e-4 / 2e-5, KL) | 2816×2 | 704 | 0.584 | 0.132 | 0.586 | 0.140 | 1.945 | 0.364 | 1.948 | 0.466 | 0.466 | 0.019 | 0.060 | 0.510 |  | 1.000 | 108.3 |
| B · frozen reader (1e-4 / 0, KL) | 2816×2 | 704 | 0.581 | 0.132 | 0.620 | 0.132 | 1.904 | 0.364 | 2.070 | 0.473 | 0.507 | 0.000 | 0.060 | 0.510 |  | 1.000 | 108.2 |
| C · fast encoder (5e-4 / 2e-5, KL) | 2816×2 | 704 | 0.586 | 0.132 | 0.589 | 0.137 | 1.956 | 0.364 | 1.961 | 0.468 | 0.470 | 0.017 | 0.050 | 0.510 |  | 1.000 | 107.6 |
| D · SFT (1e-4 / 2e-5, gold answer) | 2816×2 | 704 | 0.538 | 0.132 | 0.538 | 0.084 | 1.790 | 0.364 | 1.786 | 0.525 | 0.525 | 0.098 | 0.070 |  |  |  | 91.8 |

## What the numbers say

**The compressed passage carries almost no information to held-out questions under this recipe.** In every arm with a trainable reader (A, C, D) the AC student and the no-context control end at the same held-out gold log-loss (A 0.584 vs 0.586; per-item mean difference 0.006, max 0.03) and the same exact KL to the teacher (0.466 vs 0.466). The held-out curves show why: the student's gold NLL drops from 0.677 to 0.564 within the first quarter epoch and then slowly rises to 0.584, while the no-context reader follows the same path (0.620 → 0.556 → 0.586). The gain is the reader learning the answer prior and format, not reading the passage. The teacher stays at 0.132 gold NLL and 0.51 exact match; the students reach 0.05–0.07.

**With the reader frozen (B) the AC does carry a little.** B's student beats its no-context control by 0.04 nats (0.581 vs 0.620) and 0.03 KL (0.473 vs 0.507), so the encoder alone transmits some passage signal that a co-trained reader then absorbs into the prior. B's full-training-set KL is 0.459 against 0.213 for A: the trainable reader fits the training passages (A held 0.466 vs training 0.213), so most of A's training progress is memorization of training items rather than a transferable reading skill.

**Learning rates and objective do not change the picture.** C (encoder 5e-4) is indistinguishable from A. D (SFT on the gold answer) reaches the best held gold NLL of the students (0.538, again equal to its own no-context control) at the cost of the largest reader drift (exact KL of the full-text reader to the original base 0.098 vs 0.019 for A).

**Reader drift is small in the KL arms**: 0.017–0.019 nats for the full-passage reader against the original base, 0 for B by construction.

## Run facts

| Arm | Duration | Targets pass | Baseline | Training per epoch | Held panels per epoch | Samples per epoch |
| --- | --- | --- | --- | --- | --- | --- |
| A | 108.3 min | 2916 views, 350 s | 622 s | 1554 / 1558 s | 149 / 148 s | 288 / 275 s |
| B | 108.2 min | 2916 views, 349 s | 626 s | 1548 / 1582 s | 145 / 149 s | 284 / 287 s |
| C | 107.6 min | 2916 views, 346 s | 608 s | 1562 / 1544 s | 150 / 149 s | 285 / 281 s |
| D | 91.8 min | none (SFT) | 359 s | 1554 / 1569 s | 125 / 126 s | 289 / 289 s |

The budget estimate was 7143 s (119 min) per arm from the 256-item smoke run; the arms took 108 min, so the calibration held with four processes on one node. Stored targets covered 99.98 % of the teacher mass (top-256). Reviewed source `rag-next-20260915T194308Z`, verified on the node against `training_source.json`. Checkpoints and stored targets (about 7.4 GB per arm) stayed on the node and were discarded with it; the pulled run folders (`IB/TMP/SYNC/RAG_AC_POC/{S,A,B,C,D}/`: report page, `report_data.json`, `training_stats.json`, `final_controls.json` with per-item rows, `full_training.json`, `result.json`, `console.log`) are the retained evidence. Held questions were re-generated for 25 of 100 held chunks when the material was re-prepared at 4096 training items; all arms and the smoke run used the same material.

## What this does and does not establish

It establishes that this recipe (one 4B reader for encoder and student, 1/16 compression, one passage, top-256 KL or SFT, 2 epochs over 2.8 k passages) does not produce a compressor whose output the reader uses on unseen passages, and that the co-trained reader converges to the no-context prior with or without the rows. It does not say whether more data, longer training, less compression, a separate side model, or an objective that penalizes the no-context solution would change that. B's small but consistent margin says the rows are not empty; the recipe simply gives the reader an easier route.
