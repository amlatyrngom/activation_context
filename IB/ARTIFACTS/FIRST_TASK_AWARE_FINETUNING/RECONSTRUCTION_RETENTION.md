# Reconstruction retained after QA fine-tuning

**Completed September 23, 19:39 UTC. QA specialization substantially degrades verbatim reconstruction; the latest continuations add almost no further degradation.**

All four paired checkpoints use the same **2B encoder → 9B reader**, original 420 held reconstruction passages, 1/16 compression, prompts, targets, numerical settings and L40S allocation. These are 60 passages each from SWE tool outputs, research tool outputs, teacher traces, code, documents, math and web. No optimization or new synthesis occurred.

| Paired checkpoint | Reconstruction NLL ↓ | Token accuracy ↑ |
|---|---:|---:|
| Original reconstruction P2 | **0.106406** | **96.48%** |
| 42,000 QA questions × 1 | 1.081234 | 73.57% |
| 50,000 QA questions × 1; mild cosine continuation | 1.079777 | 73.68% |
| 42,000 QA + 960 fresh examples with 32k inputs × 1 | 1.080822 | 73.63% |

At 42k, this is **+0.974828 nats/token and −22.91 percentage points** of accuracy. Every source group degrades. Adding 8k fresh QA questions or 960 long-context examples leaves reconstruction nearly unchanged, despite their QA-loss gains. This test does not locate when within the first 42k questions the loss arose.

| Source | Original P2 NLL | 42k QA NLL |
|---|---:|---:|
| SWE tool outputs | 0.0707 | 0.8334 |
| Research tool outputs | 0.1679 | 1.4531 |
| Teacher traces | 0.1405 | 1.2079 |
| Code | 0.0829 | 0.9591 |
| Documents | 0.1258 | 1.3109 |
| Math | 0.0572 | 0.6352 |
| Web | 0.1128 | 1.5047 |

NLL and accuracy are token-weighted over **279,584 assistant targets**, including 420 chat-end tokens. Accuracy is measured with the correct preceding tokens supplied. It does not imply exact free-generation copying. Four fixed short passages per checkpoint also received greedy reconstructions: all 16 stopped at EOS without hitting the 320-token cap. The QA-tuned outputs often shorten or change the passage substantially; the original P2 also makes smaller copying errors. These four examples per pair are illustrations, not an aggregate reconstruction success rate.

The comparison measures the complete encoder–reader pair under its original reconstruction interface. Encoder forgetting, reader adaptation, and their interaction remain unseparated. QA training conditions the encoder on a question, whereas this reconstruction task provides the passage alone; the test measures retention of the original task, not an isolated representation-capacity limit.

The freshly measured P2 baseline closely matches historical NLL 0.106369 / accuracy96.50%. V2's baseline exactly reproduced the first attempt. All420 item IDs, source/target fingerprints, row budgets, target-mask lengths, model revisions and runtime settings match across pairs. The corrected full check completed in **8m57s on one L40S**, exit0; peak allocated memory was25.30GiB for these reconstruction inputs. An earlier attempt stopped after P2 because of a tuple/list metadata-comparison bug, corrected without changing inference or scoring; its results remain preserved.

[Report with source breakdowns and all 16 actual reconstructions](../../TMP/SYNC/TASK_AWARE_QA/reconstruction_retention_v1/report.tressoir.html). [Verified results and provenance](RECONSTRUCTION_RETENTION_RESULT_V2.json). Exact settings: `bench/RECONSTRUCTION_RETENTION_V2.json`; execution receipt: `RECONSTRUCTION_RETENTION_SUBMISSION_V2.json`.
