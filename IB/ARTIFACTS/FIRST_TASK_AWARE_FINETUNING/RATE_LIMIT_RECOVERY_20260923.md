# Results recovered after the Codex interruption

Checked Sep23,13:35–13:58UTC. Slurm jobs and report transport continued while Codex monitoring/review was unavailable. No GPU training was interrupted by the Codex limit. One training application independently reached its duration bound after saving its entire planned training stage; its missing final report is being recovered without training further.

## What we learned

| Final checkpoint | Training stage | Evidence at evaluation | Correct out of87 supported answers |
|---|---|---|---:|
| Original32k |32,000 unique×1 |8k independent passages |58 |
| Constant42k |+10,000 fresh×1, inherited Adam, constant terminal LR |8k independent passages |61 |
| Strong40k long |Original32k +8,000 fresh×1 trained at8k, fresh Adam |8k independent passages |65 |
| Constant42k |Same checkpoint |32k independent passages |58 |
| Strong40k long |Same checkpoint |32k independent passages |60 |
| Native constant42k |+10,000 fresh×1 |2k merged |71 |
| Native constant50k |+8,000 fresh×1 more |2k merged |70 |

The long-context stage helps:65 versus58 correct at8k, with11 gains and4 losses. Longer32k evidence remains harder:60 correct for the same final checkpoint. These32k inputs add retrieved distractors; they do not establish success on questions requiring evidence distributed across32k tokens.

The additional native8k questions improve held400 NLL slightly (.5327→.5271), while the same100 reviewed answers exchange five gains for six losses. There is no clear semantic gain from42k to50k here. Keep42k as the preferred native parent; do not automatically spend on a50k long-context readout. This small, repeatedly inspected development set does not prove universal convergence.

For the strong long-context checkpoint, held400 AC NLL is.5968 versus its current full-context reader.6000. On the selected100, AC is.6348 versus fullcontext.5939; at32k those become.6497 and.5663. Similar aggregate NLL is therefore not an answer-quality equivalence claim. The frozen full-context decoded baseline remains separate from these moving-reader NLL controls.

See [native continuation answers](CONTINUATION_COHORT100.tressoir.html) and [long-context answers](STRONG_LONG_COHORT100.tressoir.html). The original87 supported/13 reference-concern labels are fixed; new answers do not change them.

## Runtime opportunity

One actual8-question batch ran in20.17s rather than24.61s when the reader avoided recomputation, an18.1% reduction. Exact deterministic loss, outputs and all615 active gradients matched. Allocated memory rose49.81→62.79GiB. A few worst real batch shapes still need a memory check before future use; no production setting has changed. See [probe audit](READER_CHECKPOINT_GPU_RESULT_V1.md).

## Recovery in flight

Gentle constant42k +4,000 familiar questions×1 at8k completed alltraining and saved both adapters plus Adam5750. Its own duration limit interrupted the final panel; the dependent evaluator correctly rejected the incomplete publication.23546188 now completes reporting with optimizer steps forbidden and exact before/after weight/moment checks.23546190 runs the original100-answer readout only after successful finalization. This stage still has42k unique questions,46k presentations. Its final quality is pending.

The important strong final pair is selected for preservation/S3 archiving. Intermediate checkpoints and optimizers stay outside S3. The53k27B question bank and paused9B bank remain preserved; no synthesis or EC2 jobs were restarted.
