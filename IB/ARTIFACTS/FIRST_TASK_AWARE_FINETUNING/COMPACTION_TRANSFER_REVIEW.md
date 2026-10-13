# Continuation compaction → trajectory QA: 100-item review

The continuation-trained pair answers **48/100** correctly, compared with **40/100 initially**, **41/100 after 2,000 QA questions**, and **96/100 with frozen full history**. This is a modest positive semantic transfer signal: 14 gains and 6 losses against initial AC; 18 gains and 11 losses against QA tuning.

The checkpoint trained on 1,848 histories / approximately 5.21M supervised assistant tokens using **legacy fixed-teacher continuation KL**. It is not the new oracle-summary SFT experiment. These training budgets are not matched.

| Condition | Semantic correct | Complete submit | Correct + submit | Gold NLL |
|---|---:|---:|---:|---:|
| frozen full history | 96/100 | 98/100 | 95/100 | 0.299322 |
| frozen no history | 26/100 | 96/100 | 25/100 | 1.772159 |
| initial AC | 40/100 | 99/100 | 40/100 | 0.944860 |
| AC after 2000 questions | 41/100 | 99/100 | 41/100 | 0.908110 |
| Reader after 2000 questions, no history | 22/100 | 87/100 | 20/100 | 1.758753 |
| AC after 1848 continuation histories | 48/100 | 99/100 | 48/100 | 0.879622 |
| Reader after 1848 continuation histories, no history | 25/100 | 83/100 | 23/100 | 1.819729 |

Gains include the complete `services/gcs/data/collection.py` path (#34), `_keybindings` disposal (#46), the unmodified `/workspace/greenery` cause (#68), Infernal Arms (#70), linear approximation (#78), Davy Jones (#91), literal `<NA>` (#96), and the exact Tethys URL (#100). Losses include numerical facts, an equation solution, a coefficient, bridge/plate confusion and the LCCN. Coding improves most; math is 7/15 versus initial 8/15 and QA 2k 9/15.

The trained reader without history is 25/100 correct versus frozen 26/100, with only 83 completed submit calls versus frozen 96. The visible drift is chiefly more unfinished prose; its worse NLL does not by itself prove broad factual forgetting. AC retains 99 completed calls, so its semantic gains are not explained by improved formatting.

All 100 ordered IDs, roots, questions, gold answers and history lengths match. Fresh cohort NLLs are used above, not training-curve measurements from different hardware/kernel runs. Both encoder and reader were trained; the ablation does not isolate encoder learning. The 100 questions share 30 roots, and no significance or convergence claim is made.

Every new AC and no-history response was read. Prior reference concerns were preserved; selective source checks are not an exhaustive factual audit. The conflicting age item (#24) remains unknown and in the denominator. Equivalent wording/algebra is accepted; requiring literal wording for the borderline error template (#75) would reduce continuation AC to 47/100. No-history #62 is partial because it invents a rerun threshold absent from the source. Unfinished responses are recorded separately from semantic correctness; finish reasons were not saved.

[Interactive report](../../TMP/SYNC/TRAJECTORY_COMPACTION/qa_compaction_transfer_review/report.tressoir.html) · [Item-keyed evidence](../../TMP/SYNC/TRAJECTORY_COMPACTION/qa_compaction_transfer_review/evidence.json)
