# From 2k to 32k tokens of evidence

All six model/context combinations are complete. The same 100 development questions (50 per generator) were evaluated with each final paired reader at both evidence lengths. Each system trained on 16,000 unique 27B-generated questions × 1 epoch, using roughly 2k evidence tokens.

| System | AC at 2k | AC at 32k | Full at 2k | Full at 32k |
|---|---:|---:|---:|---:|
| 2B encoder, 1/16 | 0.6403 | 0.6969 | 0.6319 | 0.5484 |
| 2B encoder, 1/8 | 0.6789 | 0.7541 | 0.6190 | 0.5422 |
| 4B encoder, 1/16 | 0.6170 | 0.6788 | 0.6321 | 0.5539 |

Values are gold-answer NLL in nats/token, averaged equally over questions. More evidence helps the full-context reader but worsens average AC NLL for all three systems. The 4B system improves on 51 individual questions, but its losses are larger; the average alone does not describe every item. There are no generated-answer scores in this screen.

The required source remains whole. Additional BM25 distractors increase the budget from 2,048 tokens / up to 4 passages to 32,768 / up to 64. AC and full context receive identical evidence. This tests distractor robustness, not multi-document reasoning or training with long context. Merged passages and the saved 4,096-row cap remain unchanged; the HTML reports actual lengths, rows and cap hits.

Both lengths use microbatch one on RTX PRO 6000. Within each system, every no-context score is identical across lengths. The 4B/P4 initialization differs from the 2B/P2 pair in encoder and reader weights and reconstruction quality, so this is a comparison of trained systems, not an isolated capacity test.

Private worker `bench/evaluate_context_transfer.py` reuses the pinned builders, final-pair checks and fixed selection. All 100 IDs, cohort sizes, aggregate means, evidence fingerprints and checkpoint identities passed. ORCD jobs: 23461879 (2B) and 23466483 (4B). Earlier standalone browser checks cover the unchanged report structure; actual VS Code CSP/morph behavior remains untested.

[Read the report](CONTEXT_TRANSFER100.tressoir.html) · [Exact evidence](CONTEXT_TRANSFER100.evidence.json)
