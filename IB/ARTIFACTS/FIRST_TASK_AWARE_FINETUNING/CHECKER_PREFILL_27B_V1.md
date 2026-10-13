# 27B checker prefill throughput diagnostic

Completed GPU job23429488. **Keep the production2,048-token prefill budget.** The larger budget yielded only a3.8% gain in uncached checker call throughput and changed16 of128 distinct verdicts. This is not evidence of a semantic improvement. Production remains unchanged.

| Prefill budget | 256 measured calls: seconds | Requests/sec | Invalid JSON | Repeated decisions matching |
|---|---:|---:|---:|---:|
|2,048|44.02|5.82|0|128/128|
|8,192|42.40|6.04|0|128/128|

Both measured passes processed436,158 total prompt tokens with zero cached prompt tokens. Between settings,224/256 paired requests agreed; the same16 distinct cases changed in both repetitions (13 true→false,3 false→true). This demonstrates sensitivity across these engine configurations, but one fresh engine per condition does not isolate the numerical cause. No semantic correctness labels were used. The result cannot estimate whole-bank throughput improvement. Full timings/output decisions remain in the synced comparison.json and per-condition results.json files.

Test whether raising `max_num_batched_tokens` from 2,048 to 8,192 speeds up the existing 27B grounding checker. This measures **uncached checker throughput**, not whole-bank end-to-end throughput or semantic accuracy. The producer also spends time generating questions and can reuse evidence prefixes; this diagnostic deliberately excludes those effects.

## Fixed input and conditions

128 accepted portable-bank questions, selected before inspecting outputs from the first five balanced 27B releases. Seeded question-ID hashes, distinct source requests, proportionate interleaving; no new reference audit or correctness labels. Counts: code 38, arXiv 15, textbook 10, financial 7, web 26, technical web 6, manuals 10, legal 8, office 8. Accepted bank membership is not a correctness label.

`IB/TMP/FIRST_TASK_AWARE_FINETUNING/checker_prefill_27b_v1/cases_128.jsonl`

SHA256: `7714ae5f00d053dd06c1b98437a5d7a14f414e66706788e89c74a8946a05e321`. Adjacent `cases_manifest.json` records release hashes and selection. Full evidence, question, answer, source units and source receipts remain portable.

Both conditions use the exact deployed boolean prompt and pinned `Qwen/Qwen3.5-27B` revision `fc05daec18b0a78c049392ed2e771dde82bdf654`, dynamic FP8, context8,192, max64 sequences, GPU memory utilization0.85, speculative decoding off. Thinking off, temperature0, presence penalty0, top-p0.8, top-k20, min-p0, maximum48 output tokens, identical structured JSON schema. All explicit settings other than the prefill token budget match production. Prefix-cache and KV-dtype defaults remain those of the same deployed harness.

One fresh Python process/engine per condition, in order2,048 then8,192. Each runs one64-request warm-up, followed by **two measured passes through the same128 cases in64-request batches**:256 measured requests per condition,512 total. The prefix cache is explicitly reset before warm-up and every measured call; reset time is recorded separately and excluded from call timing. Any remaining within-batch cache hits are recorded, not assumed absent.

## Measurements and limits

- Fresh-engine load, warm-up and each steady call have separate timings. Fresh engine does not mean cold filesystem or compiler caches; the second process may reuse disk/compiler caches. Startup is excluded from throughput comparisons.
- Raw outputs, finish reasons, strict JSON validity, prompt/output/cached-token counts and paired decisions are saved. Agreement is compared across both conditions and across repeated passes within each condition. This is not semantic accuracy.
- Device-wide free/used memory is sampled every0.5seconds, including the engine subprocess. Existing harness KV metrics retain roughly2-second samples and cache capacity when available. These are sampled headroom measures, not exact allocator peaks; engine startup logs retain vLLM profiling/capacity information.
- Actual27B tokenizer lengths are checked before calls; evidence is never truncated. The cached9B tokenizer proxy gives262–3,854 prompt tokens,218,079 total for128 cases. Each repeated pass uses precisely the same texts/order.
- Small fixed replay, fixed condition order and one engine per condition make this a practical throughput diagnostic, not a broad performance benchmark. A gain here would justify a bounded producer setting change followed by observing real accepted questions/hour; it does not predict that gain directly.

## Prepared execution

`bench/checker_prefill_27b_v1.sbatch`: one RTX PRO6000,8CPUs,100GB host memory,30-minute hard limit, no automatic requeue. Earlier27B starts took217–302seconds; two starts plus this small replay should plausibly fit roughly10–20minutes, but fresh graph compilation can vary. A timeout leaves the partial raw results and logs; do not interpret an incomplete pair as a comparison.

Remote base: `~/orcd/pool/activation/taskaware/`.

- Frozen cases/manifest: `checker_prefill_27b_v1/`.
- Worker and helpers: `bench/`; job file: `jobs/checker_prefill_27b_v1.sbatch`.
- Output: `reports/checker_prefill_27b_v1/`, with `p2048/` and `p8192/` raw results, logs and individual reports; completed pair adds `comparison.json` and overview HTML.
- Expected synced local report: `IB/TMP/SYNC/TASK_AWARE_QA/checker_prefill_27b_v1/report.tressoir.html` once the existing report transport discovers it. Per-condition logs/results remain on ORCD even if the transport does not descend to their extra level.

Root schedules this only after the whole-parent pilot and within the existing four-GPU cap; the paired16k training experiment has priority. No job was submitted by this preparation task.

## CPU validation

Python compile and shell syntax passed. `bench/check_checker_prefill_27b_v1.py` checked exact production prompt/sampling, only-prefill configuration differences, case hashes/family counts, identical evidence/order and proxy token lengths. Two fake fresh engines exercised all ten64-request calls, ten prefix resets, both shutdowns, invalid outputs, changed decisions, matched-recipe rejection and actual report/result publication. This validates orchestration, not GPU performance. Receipt: adjacent `cpu_checks.json`; deployment hashes: adjacent `deployment_receipt.json` when staged.
