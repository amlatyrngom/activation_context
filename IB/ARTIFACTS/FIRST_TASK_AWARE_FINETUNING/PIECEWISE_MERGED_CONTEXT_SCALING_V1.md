# Matched layouts at8k and32k evidence

Both models trained on the same4,000 questions once at8k evidence. The rows below score the same100 development questions. Inputs and total compressed-row counts match between layouts.

| Layout | AC NLL at8k | AC NLL at32k | Own full-context NLL at8k | Own full-context NLL at32k |
|---|---:|---:|---:|---:|
|Independent passages|1.0181|1.0253|0.5842|0.5447|
|Merged passages|1.3617|1.3636|0.5863|0.5573|

Both AC models show little additional NLL loss from the extra distractors. The passagewise model retains its substantial advantage. Mean compressed rows grow from502.9 to2055.3; no row cap was introduced.

At8k, source review gives34/87 fully correct for independent passages versus18/87 for merged passages, with18 versus2 exclusive wins. See[the400-answer comparison](PIECEWISE_COHORT100.tressoir.html). No actual32k answers were generated for these two small models.

These results concern distractor tolerance after8k training. They do not establish multi-document reasoning or semantic32k parity. The8k scoring used microbatch2 and32k used microbatch1; own full/no-context controls accompany each cell.
