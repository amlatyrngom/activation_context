# Complete development-cohort inspection

Inspect every one of the original100 development items, preserving their
recorded order and source/gold data. Compare four explicitly named views:

- Frozen P2 reader, full context: the reader snapshot from the start of QA
  training, with the original gold evidence and distractors.
- Frozen P2 reader, no context: that same unchanged reader with context removed.
- AC control update500: its saved paired encoder/heads and trained reader.
- AC control update900: its saved paired encoder/heads and trained reader.

The previous12-item report remains unchanged. Its trained-reader full/no-context
curves are not relabeled as frozen baselines.

`bench/evaluate_qualitative_cohort.py` loads one fixed checkpoint per process.
It verifies material/order hashes and records hashes of the actual saved weights.
NLL uses the complete100-item list with the original microbatch2 sorting and
batch partners. It compares the repeated NLL against the corresponding original
panel and retains any difference. Baselines compare with update0; AC views with
their respective update500/900 panels.

Each view also gets a fresh greedy decode with thinking disabled and a160-token
limit. Free generation starts before any answer token; the NLL measurement
instead conditions each gold token on previous gold tokens. Both are retained
separately, with raw responses, parsed answers, per-token losses, exact required
source units and full evidence including distractors. Completed rows persist
incrementally; restarting the same evaluator/input identity skips saved decodes.

The GPU script is prepared but submission belongs to the root agent. After it
finishes, inspect all100 items directly and write one HTML report with a concise
summary, descriptive counts for this development cohort and all100 item records.
Assess reference validity separately from each response: source-supported,
source-contradicted, ambiguous or not decidable from supplied evidence. Assess
responses for substantive correctness, omission, unsupported additions, refusal
or truncation; a wording mismatch alone is not an error. Explain representative
NLL/semantic disagreements using the recorded source and actual decodes.

No percentage inferred from the earlier selected12 examples will be presented as
a dataset-wide error rate. Even after inspecting all100, any counts describe this
development cohort, which was used for checkpoint selection; they are not a
confirmation/test estimate. Prompt repair and new synthesis are separate work.

Expected inputs: `TMP/SYNC/TASK_AWARE_QA/qualitative_cohort100/`.
New artifacts: `QUALITATIVE_COHORT100.evidence.json` and
`QUALITATIVE_COHORT100.tressoir.html`, beside this plan.
