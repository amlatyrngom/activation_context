# Rollout tuning run: 2x RTX PRO 6000, medium effort

804 suite tasks (hotpotqa 134, musique 134, dapo_math 134, lca_bug_localization 134, quality 134, narrativeqa 134) on `unsloth/Qwen3.8-27B-NVFP4` (max_model_len 64000), two replicas, reasoning effort medium, autotune id `campaign_27b_medium_v2`, caching id `teacher_campaign_medium_s1`, seed 1, task order shuffled. One continuously fed queue: the first 803 tasks to finish ran under the defaults (phase A), then the autotuner chose the configuration below and the queue continued under its pool bound (phase B). Uncached pass 6,082 s in all. Every trajectory is in the campaign cache; the seed-0 campaign of the same 804 tasks (first tuning run, see below) is in `teacher_campaign_medium`.

## Chosen configuration

| Knob | Default (phase A) | Tuned (phase B) |
| --- | --- | --- |
| Agents per GPU | 40 | 33 |
| max_num_batched_tokens | engine default (8,192) | 16384 |
| gpu_memory_utilization | 0.9 | 0.92 |
| Speculative decoding (MTP) | on | off |
| Engine arguments | | deferred to the next run under this autotune id (agents were in flight) |

Rules that fired:

- kv budget: 0.8 x 1,593,600 / p90 prefix 28,686 = 44 -> 44 agents per GPU
- evicting: recompute 59%, waiting 45%, preempted 1 -> 33
- prefill-bound: prompt share 95% -> max_num_batched_tokens 16,384
- kv headroom: host memory p95 7% -> gpu_memory_utilization 0.92
- saturated: p95 kv 84%, 28 tok/s per replica -> speculative off

## Phase A against phase B

Each column is one window of the same queue: engine and host samples taken in it, and the runs that finished in it.

| Metric | probe | tuned |
| --- | --- | --- |
| Runs finished in the window | 803 | 1 |
| Window (s) | 5,224.40 | 4.20 |
| Tasks / h / GPU | 276.66 | 425.64 |
| Generated tokens / h / GPU | 1,350,403.25 | 16,770,054.91 |
| Engine decode tok/s per replica | 27.59 | 0.00 |
| KV usage p50 | 60.2% | 0.0% |
| KV usage p95 | 84.2% | 0.0% |
| Samples with a queue | 45.3% | 0.0% |
| Preemptions | 1 | 0 |
| Prefix hits | 30.8% | 16.8% |
| Recompute beyond alignment | 59.3% | 76.2% |
| Prefill share of engine tokens | 95.4% | 0.0% |
| Live prefix p50 | 7,076 | 16,931 |
| Live prefix p90 | 28,686 | 43,021 |
| Turn latency p50 (s) | 18.65 | 131.96 |
| Turn latency p90 (s) | 99.79 | 307.37 |
| Sandbox share of agent time | 11.8% | 27.9% |
| GPU utilization p50 | 98.5% | 0.0% |
| Host CPU p95 | 11.4% | 1.9% |
| Host memory p95 | 6.6% | 6.5% |
| Run duration p95 (s) | 1,997.38 | 4,224.22 |
| Mean score | 0.72 | 0.00 |

## Per benchmark per phase

Mean score / submitted share / mean total turns / mean generated tokens per run (phase membership by cache order, which is finish order).

| Benchmark | probe | tuned |
| --- | --- | --- |
| hotpotqa | 0.83 / 100% / 3 / 540 | - |
| musique | 0.69 / 99% / 7 / 2,161 | - |
| dapo_math | 0.78 / 90% / 5 / 9,751 | - |
| lca_bug_localization | 0.75 / 86% / 32 / 13,997 | 0.00 / 0% / 37 / 39,400 |
| quality | 0.84 / 100% / 5 / 1,138 | - |
| narrativeqa | 0.43 / 100% / 9 / 1,768 | - |

## Trajectory lengths

Total counts every compaction segment at its final size (prompt plus every step); generated counts assistant tokens, thinking included. Subagents are listed with their own means and folded into the joint figure.

### Seed 1 (this run, 804 tasks)

| Benchmark | Tasks | Total mean | Total p50 / p90 / max | Generated mean | Generated p50 / p90 / max | Subagents (total / generated mean) | Duration p50 / p90 / max (s) | Submitted | Score |
| --- | ---: | ---: | --- | ---: | --- | --- | --- | ---: | ---: |
| hotpotqa | 134 | 3,992 | 2,971 / 5,862 / 32,033 | 540 | 279 / 735 / 8,442 | 3 (17,251 / 2,327) | 23 / 62 / 972 | 100% | 0.83 |
| musique | 134 | 9,425 | 4,460 / 25,839 / 66,091 | 2,161 | 651 / 6,463 / 21,403 | 8 (86,544 / 17,896) | 50 / 539 / 4,202 | 99% | 0.69 |
| dapo_math | 134 | 11,851 | 6,120 / 25,920 / 73,537 | 9,751 | 4,412 / 24,576 / 63,012 | 33 (8,584 / 6,217) | 397 / 1,805 / 3,876 | 90% | 0.78 |
| lca_bug_localization | 134 | 46,664 | 15,166 / 136,177 / 265,095 | 14,186 | 5,301 / 45,429 / 78,975 | 30 (37,146 / 8,607) | 520 / 3,000 / 4,224 | 85% | 0.75 |
| quality | 134 | 7,348 | 6,784 / 13,186 / 18,186 | 1,138 | 687 / 2,768 / 5,491 | 15 (7,370 / 1,408) | 70 / 272 / 678 | 100% | 0.84 |
| narrativeqa | 134 | 11,962 | 9,006 / 21,697 / 79,203 | 1,768 | 966 / 4,116 / 12,454 | 17 (6,240 / 1,335) | 101 / 358 / 1,289 | 100% | 0.43 |

Joint per task with subagents folded in: 18,140 total tokens, 5,742 generated. Finish reasons: submitted 768, max_turns 14, no_tool_call 12, max_duration 8, trajectory_cap 1, max_tool_errors 1.

### Seed 0 (first tuning run, 804 tasks)

| Benchmark | Tasks | Total mean | Total p50 / p90 / max | Generated mean | Generated p50 / p90 / max | Subagents (total / generated mean) | Duration p50 / p90 / max (s) | Submitted | Score |
| --- | ---: | ---: | --- | ---: | --- | --- | --- | ---: | ---: |
| hotpotqa | 134 | 3,930 | 2,987 / 6,063 / 25,566 | 514 | 249 / 1,034 / 5,975 | 0 | 22 / 72 / 380 | 100% | 0.83 |
| musique | 134 | 11,226 | 5,132 / 25,688 / 151,253 | 2,463 | 856 / 6,909 / 34,073 | 10 (28,239 / 4,458) | 58 / 349 / 1,114 | 100% | 0.72 |
| dapo_math | 134 | 12,787 | 6,129 / 25,876 / 122,154 | 10,403 | 4,538 / 24,576 / 109,733 | 27 (15,558 / 12,062) | 216 / 837 / 3,399 | 89% | 0.76 |
| lca_bug_localization | 134 | 14,249 | 7,898 / 26,131 / 128,702 | 3,943 | 1,718 / 7,317 / 64,895 | 6 (6,237 / 1,820) | 3,601 / 3,690 / 3,796 | 49% | 0.46 |
| quality | 134 | 7,148 | 6,719 / 12,463 / 23,208 | 1,291 | 709 / 3,223 / 11,268 | 21 (7,862 / 1,583) | 44 / 153 / 388 | 100% | 0.84 |
| narrativeqa | 134 | 12,525 | 9,806 / 26,657 / 56,023 | 2,040 | 949 / 5,273 / 13,378 | 16 (8,776 / 2,043) | 96 / 340 / 528 | 100% | 0.38 |

Joint per task with subagents folded in: 11,611 total tokens, 3,999 generated. Finish reasons: submitted 720, max_duration 67, no_tool_call 12, max_tool_errors 3, max_turns 1, trajectory_cap 1.

## Projection for the campaign on 2x RTX PRO 6000

At the tuned window's rate of 426 tasks per hour per GPU (851 per hour on two GPUs, 33.5M generated tokens per hour) with this task mix and medium effort, after about 12 minutes of engine load:

| Run length | Trajectories | Generated tokens |
| --- | --- | --- |
| 6 h | about 4,937 | 195M |
| 12 h | about 10,045 | 396M |
| 24 h | about 20,260 | 798M |

Subagent runs are not counted as trajectories (they are cached inside their parent). The tuned window includes the run's final drain, so its rate is a lower bound of the steady state.

## Reading

Three runs fed this report: the first tuning run (seed 0, tasks in benchmark order, batch-then-drain scheduling), the seed-1 measurement run (shuffled order, one continuously fed queue), and the cache data of both.

**What the defaults do on the full mix.** With 40 agents per GPU on the shuffled mix the two engines are saturated: GPU utilization 98% at the median, KV cache 84% full at p95, a request queue in 45% of the samples, prefix-cache hits down to 31% and 59% of prompt tokens recomputed beyond block alignment. The run is 95% prefill by engine tokens, decode is 28 tokens per second per replica. The long-document and repository tasks carry p90 live prefixes of 29k tokens, so 40 agents per GPU do not fit in the 1.6M-token cache and evict each other. The whole-run rate was 277 tasks per hour per GPU (554 on two GPUs), and 400 to 500 per hour per GPU during the 40 minutes the pool was full; the difference is the drain of the last tasks.

**What the tuner chose.** From that window: 44 agents per GPU by the KV budget, cut to 33 by the eviction rule; batched tokens 16,384 because the run is prefill-bound; memory utilization 0.92 since the host has room; speculative decoding off because the cache is saturated and decode is a small share. The engine arguments could not be applied mid-run (agents were in flight) and are deferred to the next run under this autotune id; only the pool bound applied, to the last task. So phase B of this run is one task and says nothing. The tuned configuration is measured by the campaign run that starts from it (saved, applied before the engines load).

**What the first run taught, beyond the numbers.** Tasks were ordered by benchmark, so its probe saw only the cheap multi-hop and math tasks (1,650 to 1,970 tasks per hour on two GPUs) and its phase B met the repository tasks, where 78 threads each built their own 290,000-chunk BM25 index at once and blocked for an hour inside a tool call; 67 repository tasks ended at the 3,600 s cap with score 0 and were filtered from that cache (737 clean rollouts remain). Fixes landed: one build per index under a lock, indexes built before the threads, no passage index at all for repository tasks (the oracle used ripgrep 452 times and the index 158 times, with equal scores either way), a seeded shuffle of the task order, and the queue that never drains. The probe now fires on the first N finished tasks rather than the first N submitted, since one straggler among the latter delayed this run's tuning by an hour.

**Gains and losses of autotuning, as far as measured.** Gained: an instrumented answer to why the run is slow (prefix eviction under a saturated cache, not host or sandbox time: host CPU p95 11%, sandbox share 12%), a saved configuration that a later run applies from the start, and the probe cost is zero because the probe tasks are real rollouts. Lost: nothing in throughput yet, since the only tuned window was one task; the risk of the chosen configuration is that 33 agents per GPU underfill the GPUs on a cheap-task stretch. Both sides are settled by the first hour of the 12 h campaign, which runs under the saved configuration and reports the same window metrics; if its rate is below the defaults' 400 to 500 per hour per GPU, the campaign is relaunched on the defaults and resumes from its cache.

**Expected trajectories on two RTX PRO 6000 nodes at the defaults**, full mix, medium effort, continuously fed queue, engine load subtracted: about 4,500 in 6 h, 9,500 in 12 h and 19,000 in 24 h at the pool-full rate; 3,200, 6,500 and 13,000 at the whole-run average that includes a final drain. Subagents are 12% more agents on top and are not counted.

