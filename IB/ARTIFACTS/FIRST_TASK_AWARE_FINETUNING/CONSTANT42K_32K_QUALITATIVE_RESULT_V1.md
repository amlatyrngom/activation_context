# Constant-floor 42,000-question checkpoint: 8k → 32k inference

The same saved AC/reader pair gives **58/87 fully correct at 32k versus 61/87 at 8k**, with four gains and seven regressions. This is a small paired exploratory loss on a repeatedly inspected development cohort, not evidence that longer context is generally worse. Additional retrieved passages primarily test distractor robustness. No additional training occurred.

- Same 42,000 unique questions × 1 pass checkpoint: original 32,000 + 10,000 fresh, inherited Adam and constant floor learning rate. Readout job 23530004 completed in 16m42s.
- Fixed source split: 87 supported / 13 reference concerns. All 100 questions remain in NLL and examples; all 600 earlier answer records and source fields remain exactly unchanged.
- Supported counts at 32k: 58 correct, 10 partial, 18 incorrect, one existing contextual ambiguity. All 100: 61 correct, 13 partial, 24 incorrect, two ambiguous.
- Paired against the same weights at 8k: 54 correct in both, four only at 32k, seven only at 8k, 22 neither. All seven regressions become incorrect. Against the original 32,000-question checkpoint at 8k: seven gains / seven losses; this also changes weights.
- Generator cohorts: 31/42 correct on 9B-generated questions, 27/45 on 27B-generated; the 8k readout gives 32/42 and 29/45.
- Actual new context: 31,736–32,767 evidence tokens, 10–64 passages, 1,995–2,074 rows (mean 2,055.33). Ratio remains 1/16; all required sources retained whole; no row cap applied. Actual 32k input records and evidence are stored separately from earlier 8k inputs.
- **Selected 100 only:** AC NLL 0.678534864 versus 0.656747371 at 8k. Own reader full-context/no-context NLLs are 0.573443782 / 1.843845796. This readout has no 400-question mean or same-weight training-panel replay. Earlier frozen P2 decoded controls still use 8k evidence, not 32k.
- All 100 answers reached EOS; zero caps, trailing text or observed loops; 4,330 generated tokens. Greedy, thinking disabled, 160-token cap, EOS IDs 248044/248046. RTX PRO 6000; scoring micro-batch 1; full precision/runtime provenance retained.

The four gains recover the en-dash separator and frontend method, and remove unsupported additions from D-dimer and head-lice answers. Regressions change the ShardingSphere product, country intersection, division name, default-extras rule, coupon values, chore payment and anonymous-role enum. Pytest 2.8.4 is genuinely announced in actual 32k Passage 1; its generic question remains ambiguous under the existing treatment.

[Seven-view report](STRONG_LONG_COHORT100.tressoir.html) · [Evidence](STRONG_LONG_COHORT100.evidence.json). Private handoff, source checks, hashes and browser receipt: `IB/TMP/FIRST_TASK_AWARE_FINETUNING/constant42k_32k_review/`. Root spot-checks agreed on indices 1, 8, 31, 55 and 96. Standalone Chromium checks cover 700 cards, filters, actual evidence panels, mobile/light/dark and overflow; actual VSCode custom-editor CSP/live morphing was not exercised.
