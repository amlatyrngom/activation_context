# Remaining 9B synthesis capacity

At **2026-09-22 08:22 UTC**, the bank has **147,000 balanced questions published**, **147,805 accepted** including reusable surplus, and **53,000 left to publish** toward 200,000. All nine registered source pools cover the remainder at both lifetime and recent yields. **No source expansion is needed now.** This completes the capacity recheck requested near 150k.

| Source family | Unconsumed windows | More accepted questions needed | Lifetime questions/window | Recent questions/window | Minimum future questions/window |
|---|---:|---:|---:|---:|---:|
| code | 29,386 | 15,677 | 0.920 | 0.919 | 0.533 |
| math_arxiv | 69,823 | 6,278 | 1.450 | 1.436 | 0.090 |
| math_textbook | 15,597 | 4,179 | 1.176 | 1.113 | 0.268 |
| financial | 17,011 | 2,600 | 1.257 | 1.208 | 0.153 |
| web | 59,896 | 10,437 | 0.842 | 0.849 | 0.174 |
| technical_web | 9,551 | 2,599 | 0.853 | 0.881 | 0.272 |
| documents | 18,861 | 4,180 | 1.585 | 1.562 | 0.222 |
| legal | 25,249 | 3,127 | 1.712 | 1.674 | 0.124 |
| office | 10,809 | 3,118 | 1.002 | 0.993 | 0.288 |

“Recent” means the last 50 committed microchunks per family, covering 1,600 source windows. Yields above one reflect the approved multiple-question mode on longer math/financial/document/legal/office sources. Required counts already credit committed surplus; every published release retains the fixed nine-family mix.

**Code remains the limiting family.** It needs 15,677 accepted questions from 29,386 windows: 0.533/window, versus 0.920 lifetime and 0.919 recently. Recent yield projects 1.72 times the remaining requirement. At lifetime yield, about 17,035 more windows are needed, leaving 12,351 at the target. Its current extrapolation limits the balanced bank to roughly 238k; this is an estimate, not a guarantee of future acceptance.

Keep the existing **14,339-window code extension**. Code has consumed 48,160 of the original 63,207 windows, leaving 15,047 before that extension. The original pool alone cannot supply all 15,677 required questions even at one accepted question per remaining window. The extension is registered and its metadata still matches the prior manifest. No repeating old sources or collecting extra sources is needed.

The other eight families have at least 3.24 times their remaining requirement at recent yields. Math-textbook yield has softened to 1.113/window, but its remaining-capacity multiple is still 4.15. Source segments can differ: **recheck at 180k published**, when code enters its extension, or if its recent yield falls substantially toward 0.6/window or exhaustion is reported.

## Storage and protection

No pruning is warranted. ORCD cached quota (08:06:56 UTC) is **412.6/1024 GB pool** and **824.6/1024 GB scratch**, leaving 611.4 and 199.4 GB. Local filesystem availability is **261.1 GiB**. No data or preservation marker was removed; reclaimed bytes: **0**. Existing ancestor/PRESERVE protections, AC/reader pairing and the newest approximately 250 GB rule remain. S3 is still restricted to important checkpoint pairs, not every save; this pass made no archive writes.

CPU job **23465658** completed **0:0 in 2 seconds**. It read the committed bank manifest and 13 explicit preservation paths, confirming protection on both banks/reports, frozen material, prior retained selections and all listed active/completed campaign roots. No weight/source tree scan, payload hashing, source/code change, new generation, GPU allocation or training-queue change occurred.

Evidence: [current machine snapshot](SYNTHESIS_CAPACITY_20260922.json), `TMP/FIRST_TASK_AWARE_FINETUNING/storage_retention/retention_capacity_20260922_0822.json`, `headroom_20260922_0822.json`, and `registered_source_counts_20260922.json`. The earlier 06:00 receipts remain under `*_0600.json`. All eight registered extension counts/hashes still agree with their existing manifest; raw payloads and question overlap were not re-audited.
