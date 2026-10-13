# Experiments for the September 23 PI discussion

Updated **3:24pm EDT / 19:24 UTC**. Target: useful evidence by **2pm EDT / 18:00 UTC**. Compute is authorized on ORCD and CSAIL; **no EC2**. The large GPU jobs allocated at 16:56 after queueing. Intermediate panels are available; final answers will arrive after the meeting.

| Experiment | What it settles | Current state |
|---|---|---|
| Strong 42k checkpoint, one 32k encoder sequence, original 100 questions | Quality and inference fit without extra long-context training | Complete: **66/87 correct, NLL 0.6314**, versus matched passagewise **59/87, 0.6792**, and full text **81/87, 0.5744**. All 300 answers reviewed. |
| Same 42k + 960 fresh questions × 1 at monolithic 32k | Whether modest direct adaptation helps; short-context retention | **Complete: final held NLL 0.6059, learning 0.3874**, versus initial 0.6296 / 0.5116. Final decoded answers and short-context retention are queued as 23561844. Pending CSAIL duplicate and its readout canceled after fresh checks. |
| Same 42k + 8,000 fresh questions × 1, constant 2× terminal LR | Whether modestly higher sustained LR helps | **Training complete: held NLL 0.5229, learning 0.3502.** Final 100-answer readout 23549845 is pending Priority. |
| Same 42k + the same 8,000 × 1, cosine from 2× terminal LR to terminal LR | Compare sustained learning with a modest cooldown | **Training complete: held NLL 0.5118, learning 0.3199.** Final readout 23549848 pending Priority. All three initial held-400 panels exactly reproduce 0.532686410. |
| Same 42k + 960 fresh questions × 1 at native 2k, current LR versus 2× | Earlier local LR response | **Complete, including200 reviewed answers.** Terminal held NLL0.5496 /70of87 correct;2×0.5670 /71of87. [Matched report](PI_NATIVE960_LR_COHORT100.tressoir.html). |
| Original reconstruction retention after QA | Quantify reconstruction NLL/accuracy lost after specialization | **Complete: L40S 23572371,8m57s,exit0.** OriginalP2 NLL0.1064/96.48% accuracy; all three QA pairs ~1.08/73.6%, original420 passages. [Live report](../../TMP/SYNC/TASK_AWARE_QA/reconstruction_retention_v1/report.tressoir.html). |
| Resume 27B synthesis toward 100k, then 200k | Supply larger fresh-data runs | **ORCD RTX 23565367 pending**; original 53k bank plus 615 buffered preserved. Pending CSAIL duplicate 2182587 canceled before release. |

The completed 8,000 × 1 constant-terminal-LR run is the matched third large LR arm. All three use identical data/order, initial paired weights and inherited Adam state, with logical batch 8 / microbatch 2 on RTX. New encoder/reader rates are 1e-5 / 4e-6; the terminal rates are 5e-6 / 2e-6. These continuations test modest reheating and cooldown, not delaying decay throughout a new 100k-question run.

The shorter 960-question LR comparison uses microbatch 1 and full checkpointing on L40S. Its arms have identical initial controls. Their resolved numerical policies match and their preselected finals have been reviewed. It does not replace the 8k schedule comparison or establish convergence. An earlier attempt failed before training because of an incomplete inherited model-cache path; the replacements explicitly select the complete cache, with frozen training designs unchanged.

The 32k adaptation retains source-family proportions, the original 32 learning questions, and the exact 100 development questions/order. It uses 1/16 compression, one encoder sequence, up to 32,768 evidence tokens / 64 passages, logical batch 8 / microbatch 1, original 65,536 encoder packing and checkpointing. Adam state continues from the actual 42k checkpoint. Initial full/no-context baselines stay frozen; held AC and learning NLL appear every 96 questions. Final evaluation includes 100 decoded answers and native 2k retention against a same-host parent control. Only the completed final checkpoint is selected. Finalreadout23561844 isqueued aftertraining, withactual savedtraining headchunk explicitlyretained.

These are repeatedly inspected development questions. Reserved confirmation remains untouched. Most added long context is BM25 distractor material: this tests long-input handling and retrieval robustness, not yet reasoning across distributed required evidence. The longest evaluated input has 33,177 formatted source tokens plus 2,074 summary rows, or 35,251 encoder positions in one pass. Inference fit is established; the H200 adaptation has also completed its first training updates.

Live reports remain under `IB/TMP/SYNC/TASK_AWARE_QA/`:

- `report.tressoir.html`: overview.
- `pi42k_merged32k_fresh960_orcd_h200/report.tressoir.html`: monolithic 32k adaptation.
- `continue42k_8k_constant_2x/report.tressoir.html` and `continue42k_8k_cosine_2x_floor/report.tressoir.html`: large LR comparison.
- `pi42k_native2k_fresh960_floor_v1/report.tressoir.html` and `pi42k_native2k_fresh960_2x_v1/report.tressoir.html`: short LR comparison.

The H200 report alias now syncs its contents locally. A restart preserved the overview HTML; ordinary progress updates continue through JSON. Source trajectories, training configurations and metric values were not changed by the transport fix.

Other work: independent parent RTX readout23550350 completed in15m59 and reproduced monolithic32k66/87, NLL0.6305. Pending CSAIL parent readout 2182542 was canceled. Gentle-finalization 23546188 and its dependent readout remain queued; they perform no further training. A private optimizer-moment offload probe passed exact native two-update comparison and two actual 32k batches at about 38 GiB, but is not adopted and does not prove full-run or fresh-process resume behavior.

27B synthesis previously produced about 4.1–4.4k accepted questions/hour on RTX. Extending 53k to 100k is approximately 11 active GPU hours; 200k is approximately 36 hours, before queue/setup. H200 speed is unmeasured. Balanced 1k releases become usable immediately.

Frozen designs and submission receipts in this folder own exact settings, hashes and job history. [The PI evidence brief](PI_INSIGHTS_20260923.md) owns scientific conclusions. [The matched 300-answer report](PI_SAME_L40S_COHORT100.tressoir.html) contains the reviewed layout comparison. Native active monitoring continues separately from file transport.
