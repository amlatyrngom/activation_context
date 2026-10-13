# 32,000 synthesized questions × 1 epoch

The requested run uses a 2B encoder, 9B reader and 27B-generated answers, starting from the original P2 pair. At 1/16 compression, allow approximately 5.5–6 hours on one RTX PRO 6000, excluding queue and synthesis. Start as soon as the learning-rate comparison and data readiness resolve.

Keep the original 16,000 training questions unchanged and interleave another balanced 16,000. Keep the same 400 development questions, 32 learning questions and frozen distractor index. The new sources remain explicit required evidence even when absent from that index. A separate fresh 400-question confirmation excludes requests used in either original training pool and previous reservations.

Choose the recipe using the two matched 16,000 × 1 learning-rate comparisons: AC peak 5e-5 versus 2.5e-5 at 1/16 and 1/8; reader peak remains 2e-5. Prefer lower final development answer NLL, check both generator cohorts and actual answers, and retain 1/16 if the advantage of a more expensive recipe is marginal. Original five-arm confirmation scores do not guide this choice. No intermediate checkpoint cherry-picking.

The private selection fixture passed, including a reserved question appearing after its sibling in a later batch. The actual builder and tokenizers preserve existing inputs exactly and retain newly added sources outside the index. Focused review passed after requiring the other original training pool when reserving confirmation. Four preparation/launch files are staged on ORCD with matching hashes; product code and active frozen runners are unchanged. Exact eligible source-family counts gate material publication.

Evidence and file hashes: [QA_32K_PREPARATION.json](QA_32K_PREPARATION.json).

Material is ready: CPU job **23471384** completed in 56 seconds after the bank reached 35,000 published questions. The frozen run contains exactly 32,000 questions in the agreed nine-family mix; 16,000 are unchanged from the original pool. It includes about 668,000 answer tokens and 48.5 million required-source tokens before distractors. The [learning-rate decision](LR_DECISION.md) is complete: AC5e-5/reader2e-5 and1/16 compression. Repaired main job23479433 is allocated; initial23478943 stopped before training. See [Unicode mask repair](ANSWER_MASK_REPAIR.md); all32k/400 targets passed and the original16k/400 masks are identical. The run name is `oracle_27b_32k`.

The trainer can reuse matching fixed no-context references for the original 16,000 questions and compute the new ones. CPU preparation **23472004** stages that cache; model/reference identity checks remain in force.
