# Initial trajectory synthesis check

The first producer published225 questions from60 histories:113 coding,70 research,42 math;194 training and31 held. The bank remains reusable.

A quick12-question inspection found concrete code renames, an observed test count, historical document facts and recorded mathematical intermediate values. Four cited examples were spot-checked against source messages; the stated code line,117-test result,30-person limit and simplified equation are present.

One math question supplied both numbers and asked for their GCD, making history unnecessary. The next prompt revision explicitly discourages answers computable from the question itself; the checker receives the same instruction. This is a prompt-quality refinement, not a reason to discard earlier portable data. Initial controls will expose such context-independent cases.

All first56 selected histories match their native encoder token counts exactly. Oracle prompts measure1,846–35,590 tokens. Native encoding preserves reasoning and structured tool calls. The isolated277,205-token warning was not reproduced by these selected prompts; its exact source remains unknown. The preflight guard did have a separate bug: it counted BatchEncoding fields, now corrected to count input_ids.

SourceV5 also shares the evidence prefix between generation and grouped checking, with actual cache counters and timings. Seven real-tokenizer cases share98.6–99.6% of checking-prompt tokens. GPU throughput and quality after this prompt-layout change still need measurement.
