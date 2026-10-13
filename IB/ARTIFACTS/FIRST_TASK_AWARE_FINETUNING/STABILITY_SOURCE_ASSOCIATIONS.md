# Source and length associations with the original 27B gradient spikes

The available metadata does **not identify a convincing source-family or length explanation** for the original 27B-supervised 1/8 spikes. Associations are small, and the same logical batches usually have modest gradients in the 1/16 run. This supports continuing the controlled optimization investigations; it does not justify removing a source family or imposing a new length filter.

This analysis uses only the two completed original 16,000-question, one-pass runs. No confirmation questions, GPU jobs, remote operations, or live-source changes were involved. The single 568 MB training-material file was streamed; only compact training metadata was retained. [Evidence JSON](STABILITY_SOURCE_ASSOCIATIONS.json) contains hashes, all descriptive tables and metadata-only examples; [reproduction script](../../TMP/FIRST_TASK_AWARE_FINETUNING/stability_source_associations/analyze.py) records the procedure.

Exact logical batch membership is recoverable here. Both runs have identical saved material IDs and training configuration, start at epoch 1, have no resume or dropped examples, and complete 2,000 groups of eight. Recreating the frozen runner's domain interleave matches the saved IDs exactly. The frozen trainer then explicitly uses `Random(seed + epoch_number).shuffle(order)` before slicing groups of eight: seed 20260921 plus epoch 1. Sorting by student length occurs only inside each group. Python 3.11 and this host's Python 3.14 independently produce the same batch-ID hash, `45e1cfbe…6946dd0a`. This establishes logical membership, not identical physical microbatch pairing or no-context-penalty assignment across ratios.

The length limitation matters: stored `source_tokens` counts the **required source units before retrieval**, using the generator tokenizer. It excludes BM25 distractors, the repeated question and chat framing. Per-training-item total encoder-input lengths were not persisted, so this is a required-source-length analysis, not an exact total-evidence-length analysis. Stored `answer_tokens` similarly counts the bare answer, not its exact reader target-mask span.

All norms below are the existing AC-group preclip norms; the group contains encoder LoRA plus AC modules. A spike retains the earlier threshold of combined AC+reader norm >100. There are 174 such batches at 1/8 and two at 1/16.

| Association with AC norm, ranks centered within each 1,600-question window | 1/8 | 1/16 |
|---|---:|---:|
| Mean required-source length in the eight-example batch | .068 | .075 |
| Maximum required-source length | .021 | .044 |
| Mean bare-answer length | .091 | .137 |
| Maximum bare-answer length | .051 | .085 |
| Largest absolute source-family-count association | .082 | .089 |

Centering within fixed windows reduces confounding from the changing training stage; these remain exploratory correlations, with no significance or causal claim. The largest family associations are small: at 1/8, code count is +.082 and web count −.079. Every family appears in both spike and non-spike batches. For example, documents occur in 98 of 174 spike batches but also 888 non-spike batches; their presence corresponds to 9.9% versus 7.5% spike rates. This is insufficient to identify documents as a cause, especially with mixed eight-example batches and correlated family/length features.

The quartiles of **maximum required-source length** have nearly identical 1/8 spike rates: **8.6%, 9.2%, 8.5%, 8.5%**. Quartiles of mean answer length yield **5.8%, 9.6%, 10.3%, 9.0%**, a weak, non-monotonic pattern. Spike batches have median mean answer length 21.3 tokens versus 20.6 otherwise; median maximum required-source length is 2,928 versus 2,961 tokens. Required-source and answer-length distributions also remain similar across the pass while gradient volatility increases after 3,200 questions.

On the **same 174 logical batches** that spike at 1/8, the 1/16 AC norm has median **7.43**, versus 5.92 on other batches. Only **one** batch exceeds the >100 threshold in both runs. Whole-pass paired norm rank correlation is .297 and falls to .135 after ranking/centering within the fixed training windows. Neither matching membership nor this association separates the effects of learned state, compression ratio, physical microbatch grouping, or penalty assignment.

These metadata-only examples illustrate the spread; lengths are required-source/answer counts, not total encoder evidence:

| Questions seen | Pass | AC norm 1/8 / 1/16 | Maximum source / answer tokens | Families in the batch |
|---|---:|---:|---:|---|
| 6,016 | 37.6% | 3,788.9 / 3.56 | 2,601 / 43 | code 3, arXiv 3, textbook 1, web 1 |
| 8,336 | 52.1% | 6,006.2 / 4.76 | 2,606 / 36 | code 2, web 3, arXiv 1, documents 1, legal 1 |
| 11,680 | 73.0% | 5,228.8 / 7.87 | 3,377 / 20 | code 2, documents 2, office 2, arXiv 1, textbook 1 |
| 12,264 | 76.7% | 6,601.9 / 6.83 | 2,819 / 68 | code 3, financial 2, legal 1, textbook 1, web 1 |

No individual example can be blamed from an aggregated batch gradient. The evidence leaves total retrieved-evidence effects and narrower content properties unresolved. It does not support a source-pruning experiment or a new data-validation audit. Preserve the existing controlled learning-rate/clipping comparisons and the priority of the requested 32k work.
