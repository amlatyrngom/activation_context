# Compaction stage 2: plain cross-entropy on the teacher's shortest correct continuations (2026-11-17, user-approved)

Recipe position: FineWeb reconstruction → long-context reconstruction (`R_cmp_lcrec`) → stage 1 (KL to the full-prefix 9B on the
recorded continuations, one model per ratio) → **stage 2** (this). Reader frozen throughout.

Why: stage 1 can at best copy the full-context 9B on its own recorded continuations; it does not reward short or successful
behavior. Stage 2 trains the stage-1 student, through the rows, on the full-context 9B's **shortest correct** continuations, so the
selection (short and correct) is in the targets themselves (user, 2026-11-17: plain cross-entropy).

Data generation (no student involved):
- Prefixes: every 20k-cut prefix (`kit/cut_prefixes.py`, eval-200 rule) of the 13 stage-1 pools, minus `exclude_h7_dev206.json`
  (528 tasks): **3,985 prefixes** in three seed groups (one prefix per task per group): `prefixes_s2gen_c20_s0` 1,261,
  `_s1` 1,360, `_s2` 1,364 (S3 `compaction/pc_inputs/`; local `IB/TMP/PRE_REBUTTAL/s2gen/`). 2,593 of the original runs solved.
- Samples per prefix: 3 = the original recorded continuation + 2 new ones from the untuned 9B with the full prefix (the eval's
  `full` arm: budget 50,000, trajectory cap 90,000, max model length 100,000, max duration 6,000 s), sampling seeds 21 and 22.
- Selection: among the correct ones (score 1.0), the shortest, plus any within 1.3× of it (the H8a rule). Prefixes with no correct
  sample contribute nothing.
- Runner: `agent_eval/dev_chain.sh NODE prefixes_s2gen_c20_sG s2g_sG_r full 40 - 21 22` (tags `s2g_<group>_r_s<seed>`).

Training (not started): from the stage-1 per-ratio finals, plain cross-entropy on the selected continuation tokens (assistant
tokens), same trainer settings as stage 1 (encoder 5e-5, warm-up 50, cosine, batch 8, one pass), one model per ratio. Dev: agent
eval on dev-75 + dev-206.
