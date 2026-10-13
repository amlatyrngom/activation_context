# Fresh42k→50k continuation: feasible without synthesis

Read-only assessment, Sep23 04:18UTC. **A balanced8,000 fresh-question stage is available now**, including already-committed surplus. Parent remains open between the completed constant-floor and rewarm42k checkpoints. No data/optimizer preparation, bank change, staging or job submission occurred.

The53 published batches contain9,863 eligible questions beyond the original32k and the shared10k continuation. Exact8k quotas are short by60 arXiv questions. The615 committed but unpublished surplus records contain611 eligible fresh questions, including115 arXiv; four arXiv rows fail the same existing corrupted-control-character filter. Combined eligible capacity is10,474.

| Family | Fresh published | Fresh surplus | Required for8k |
|---|---:|---:|---:|
| Code |3,120|111|2,400|
| arXiv math |900|115|960|
| Textbook math |772|58|640|
| Financial |503|74|400|
| Web |2,080|65|1,600|
| Technical web |520|38|400|
| Documents |779|37|640|
| Legal |574|95|480|
| Office |615|18|480|
| Total |9,863|611|8,000|

Use the published eligible pool plus at least60 eligible arXiv surplus rows. The surplus is already immutable JSONL, committed under the same manifest; it need not be republished or removed from the producer. A private selection can read/hash that accepted suffix and record exact microchunk provenance. Published-only capacity supports7,500 at the unchanged mix; there is no need to shrink the proposed stage or restart synthesis.

“Fresh” here means unique IDs and NFC/strip/casefold question text absent from all42k training questions. The existing1,900 question-only reservations and whole protected synthesis requests remain excluded. Source documents and other questions from earlier training requests may recur under the existing split policy; this is new-question continuation, not new-document evaluation. No confirmation questions were scored or confirmation-result files inspected.

## Minimal run

- Chosen own completed42k pair; preserve actual Adam5250 moments, reset only outer stage clock, and train8,000 new questions once: **50k unique questions,50k presentations**,1,000 stage updates, lifetime Adam6250.
- Keep native2k evidence/up to4 passages merged into one AC part,1/16 ratio, original65536 encoder budget, original precision/masks, microbatch2/logical8, joint clip1, base-only no-context KL1 on25%, and the parent native **checkpoint threshold1700**. Do not copy the long-context threshold0 unintentionally.
- Constant encoder5e-6/reader2e-6, no warmup or LR rewarming; originalseed and deterministic stage shuffle. Both parent choices trained on the same42k questions, so one frozen new8k set can support either.
- Keep original400 development questions/retrieval and the loss-independent100 selection. Choose a new balancedlearning32 from the new8k pool. Bind/replay the initial native400 against the chosen parent’s native readout. Use AC400+learning32 every800 questions; frozen initial controls plus current-reader full/no-context at0/50/100 can reuse the now-prepared sparse callback.
- Own completedfinal only; inherited-Adam/published-pair partial resumption. Final400 replay and the same100 actual greedy160/no-thinking/EOS-union AC answers. No new confirmation exposure or checkpoint selection.

Small private changes suffice: sibling of the existing10k continuation preparer/runner/evaluator, plus launchers. Change item/stage counts10k/1250→8k/1000 and parent/lifetime counters32k/4000→42k/5250; require6250 at own final. Extend selection to exclude both saved training-ID lists and read the committed surplus suffix using the producer’s per-family-prefix rule. Preserve all original product and frozen-run sources. Actual data selection and independent optimizer-copy verification belong on allocated CPU compute, with the existing tensor-exact clock-reset checks.

## Runtime and evidence limits

The completed constant-floor10k stage measured4,654.6s pure training, or3.724s per8 questions. Scaling gives **62.1min pure GPU training for8k**. That job also spent29.9min reporting and about14.8min in baseline/model placement/other work; total trainer time122.2min. Accordingly, approximately1h is credible for update compute, **not end-to-end runtime**. Budget roughly90–110min plus about4min final native100 readout, and a2h allocation with paired resume margin. Sparse current-reader reporting can reduce overhead; do not promise that saving before measurement. Fresh no-context base targets still need computation; old cached keys cannot substitute for new questions.

The availability pass streamed existing portable JSONL plus held400 metadata, retained only compact metadata, and finished in9.84s. It verified the exact unchanged manifest, all read published/surplus hashes, the42k training-ID union, and reproduced the prior selection receipt’s published residual counts. No large material JSON was loaded wholesale. Full counts, input hashes and exclusions are in `FRESH8K_BANK_AVAILABILITY_V1.json`; this is a feasibility result, not a frozen8k selection.
