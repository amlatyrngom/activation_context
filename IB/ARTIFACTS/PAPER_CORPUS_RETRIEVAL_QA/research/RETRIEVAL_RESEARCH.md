# Retrieval Q/A — Round 2 research

> **Pick: HotpotQA, with one corpus pooled from all of the dataset's paragraphs (509,315 passages).** MuSiQue is the optional second benchmark.
>
> HotpotQA passes your two-part rule:
> - **Recall:** with the 9B's search query, both gold passages are in the top 10 for 47% of questions and in the top 100 for 71% (+24 points).
> - **Accuracy:** the untuned 9B, with thinking on, solves 56% with the top 50 passages against 48% with the top 10 (+8 points; 20 questions gained, 4 lost; p = 0.002).
> - **One search is mostly enough:** both gold passages already come back in the top 100 for 71% of questions.
> - **Training data:** 90k training questions.

Written Oct 18 at ≈20:30Z by the retrieval research agent. All numbers below were measured here unless they are marked "(literature)".

## 1. What was measured

- **Recall (CPU, in the container):**
  - BM25 gold-passage recall at 5, 10, 20, 50, 100 and 200 passages, on 500 sampled evaluation questions per benchmark (100 for LOFT).
  - The BM25 is the repo's own (`activation/common/bm25.py`: bm25s defaults, English stop words, no stemming).
  - Two numbers per cutoff:
    - **mean recall:** the share of a question's gold passages found, averaged over questions;
    - **all-gold:** the share of questions with every gold passage found. This is what a multi-hop question needs.
- **Query expansion and reading (one H200 on CSAIL):**
  - The untuned Qwen3.5-9B writes one keyword search query per question, with thinking off.
  - BM25 retrieves with that query.
  - The 9B then answers the first 200 questions of each benchmark from the top-k passages, in two modes:
    - thinking on, for k = 10 and 50;
    - thinking off, for k = 0, 10, 20, 50 and 100.
  - Scoring is lenient accuracy: the gold answer or an alias appears in the "Final answer:" line, after SQuAD normalization. Exact match and F1 are in the score files.
  - Paired results compare the same questions at two cutoffs (two-sided sign test).

## 2. Ranked results

Expanded query = the 9B's query alone (as you specified). Recall is shown as mean / all-gold; accuracy is with thinking on.

| Benchmark (corpus) | Recall @10 | Recall @100 | Recall gap (mean / all-gold) | 9B top-10 | 9B top-50 | Accuracy gap (wins/losses, p) | Passes your rule? |
|---|---|---|---|---|---|---|---|
| **HotpotQA**, pooled (509k paragraphs) | 70.0 / 47.0 | 84.9 / 71.0 | +14.9 / **+24.0** | 48.0 | 56.0 | **+8.0** (+20/−4, p 0.002) | **Yes, on all-gold recall** |
| **MuSiQue**, pooled (102k) | 46.0 / 12.2 | 64.4 / 31.4 | +18.4 / +19.2 | 23.0 | 33.5 | **+10.5** (+27/−6, p < 0.001) | Borderline (recall gap just under 20) |
| 2WikiMultihopQA, pooled (430k) | 58.2 / 26.2 | 68.7 / 39.6 | +10.5 / +13.4 | 38.5 | 47.5 | +9.0 (+20/−2, p < 0.001) | No: recall gap too small |
| MultiHop-RAG (609 news articles, 7.3k chunks) | 51.8 / 25.0 | 89.4 / 73.2 | **+37.6 / +48.2** | 71.0 | 74.0 | +3.0 (+18/−12, p 0.36) | No: no passages already gives 66.5% |

Raw-question BM25 (CPU only, no 9B) on the other candidates:

| Candidate | Corpus | Questions | Recall @10 (mean / all-gold) | Recall @100 | Why not |
|---|---|---|---|---|---|
| QUEST (entity sets, about 10.5 gold documents each) | 325,505 Wikipedia pages (first 1,200 characters) | 323 dev | 8.2 / 1.2 | 19.9 / 4.3 | Recall too low even at 100; answers are 10-entity lists |
| LOFT QAMPARI, 1M version | 5,878 | 100 test | 66.4 / 39.0 | 92.8 / 82.0 | Only 100 test questions; tiny corpus |
| LOFT QUEST, 1M version | 2,858 | 100 | 45.3 / 22.0 | 75.5 / 56.0 | Same |
| LOFT HotpotQA, 1M version | 10,005 | 100 | 90.0 / 83.0 | 95.5 / 91.0 | Same, and no gap |
| LOFT MuSiQue, 1M version | 6,650 | 100 | 50.9 / 6.0 | 70.9 / 30.0 | Same |

Not measured, with the reason:

| Candidate | Why not |
|---|---|
| Open Natural Questions / TriviaQA | DPR Wikipedia, 21M passages: too large to index here. (Literature) BM25 answer recall at top 20 / top 100 is 59.1 / 73.7 on NQ and 66.9 / 76.7 on TriviaQA (DPR paper, Table 2). Jin et al. (ICLR 2025) find RAG accuracy keeps rising with more BM25 passages on NQ. A plausible single-hop second benchmark if a CSAIL CPU job builds the index. |
| HotpotQA full-wiki (5.2M abstracts) | The standard open setting. The index is too large for the container's memory; it could be built on a CSAIL CPU node. The pooled corpus above is the IRCoT-style construction IRCoT uses for MuSiQue and 2Wiki. |
| QAMPARI, full | Its Wikipedia 2021 corpus is large. Answers are lists of about 13 entities, and the published proofs are sentence snippets, not passages. |
| FRAMES (literature: 824 questions, 2–15 Wikipedia articles each) | No training split, and built for multi-step retrieval. |
| BrowseComp-Plus (literature: 100,195 documents averaging 5,179 words, 830 questions) | Documents far too long for 50 in text; agentic multi-round. |
| CRAG | Five web pages per question, not a shared corpus. |

What expansion does: on these benchmarks the 9B's query is no better than the raw question for BM25. HotpotQA top-100 all-gold is 71.0 expanded and 69.8 raw. Question + query together is slightly best (74.4). See the open questions.

## 3. Findings that shape the design

1. **The reader needs thinking.** With thinking off, extra passages barely help:

   | Thinking off, top-10 → top-100 | HotpotQA | MuSiQue | 2Wiki |
   |---|---|---|---|
   | Accuracy | 44.0 → 51.0 (+20/−6, p 0.009) | 19.0 → 22.5 (p 0.17) | 38.0 → 39.0 |

   With thinking on, top-50 adds 8–10.5 points on all three. Both the reader and the training targets should therefore think.
2. **MultiHop-RAG measures memory, not retrieval.**
   - 60% of its non-null questions are yes/no.
   - With no passages, the 9B (thinking off) already scores 66.5%. Top-100 reaches 73.5 (thinking off); thinking on, top-10 is 71.0 and top-50 is 74.0.
   - Its corpus is September–December 2023 news, which the 9B evidently saw in training.
   - It has the widest recall gap of all, and it still fails.
3. **MuSiQue needs more than one search.** Only 31% of its questions have every gold passage in the top 100, with no gain on 3-hop-2 and 4-hop questions. It still gains the most accuracy, because many 2-hop questions only need the bridge passage. Using it as the main benchmark would point toward your fallback, expand-search-expand-search.
4. **Passage lengths.** Mean prompt tokens per cutoff, from the reading runs:

   | Benchmark | Top 10 | Top 50 | Top 100 | Per passage |
   |---|---|---|---|---|
   | HotpotQA | 1.2k | 5.8k | 11.6k | ≈ 115 tokens |
   | MuSiQue | 1.3k | 6.3k | 12.4k | ≈ 125 tokens |

   Text top-100 plus about 3–6k tokens of thinking fits the 9B easily. At 1/10, top-100 is about 1.2k AC rows, the same reader cost as text top-10. At 1/20 it's about 580 rows.

## 4. Proposed setup (HotpotQA)

```text
question ──► 9B (thinking off) writes one query ──► BM25 over 509k pooled paragraphs ──► top-K passages
         └────────────────────────────────────────► reader 9B (thinking on): text top-k, or AC rows of top-K ──► "Final answer:"
```

- **Evaluation:**
  - 500 fresh validation questions, held out. At about 1.5 thinking reads per second on one H200, 500 is cheap, and more questions tighten paired tests.
  - The probe's 200 questions become the development set.
  - Metric: lenient accuracy, plus exact match and F1.
  - 3 sampling seeds (X2).
- **Training data (R5):**
  - Draw 12,000 training questions from the 90,447.
  - Same expansion and BM25.
  - The 9B reads the full text top-100 with thinking, and the reply is kept when correct.
  - Expected yield at the measured 56% (top-50): ≈ 6,500 examples. Top-100 wasn't read with thinking, but with thinking off top-100 beat top-50 by 3 points.
  - ≈ 2.2 h on one H200, from the measured ≈ 1.5 reads/s (400 thinking reads in 260 s).
- **Encoding unit (R3): one passage per AC part.** It is the fastest of the options:
  - The CPU row cache is keyed by content (`activation/ac_model/ac_model_utils.py:70`, `row_cache_key` = sha256 of messages + ratio + version).
  - So an uncontextual passage is encoded once, however many questions or cutoffs retrieve it. Top-10 is a subset of top-100, so a k sweep costs one encoding.
  - Short parts also batch well under `encode_batch_tokens`.
  - Contextual = question + one passage per part. That gets no cross-question reuse but still reuses across k.
  - At 1/20 a 115-token passage gives about 6 rows, above the 2-row minimum.
  - **Not verified:** whether the harness and vLLM path handle 100 AC parts in one prompt without overhead.
- **Starting pair:** `rowsonly_1000`, the 2B→9B pair from long-context reconstruction at 1/10, rows only. It is present on CSAIL at `/data/scratch/ngom/activation/paper_compaction_v1/pairs/rowsonly_1000` (listing checked Oct 18). The contextual and uncontextual arms both start from it and train separately (R4).
- **Arms:**
  - no retrieval;
  - text top-10 / 20 / 50 / 100;
  - reader LoRA with text top-10, trained on the same targets;
  - AC uncontextual top-100 at 1/10 and 1/20;
  - AC contextual top-100 at 1/10 and 1/20.
  - The main figure: x = passages retrieved, y = accuracy, with a line per arm.

**Run list.** These are estimates. Training speed was not measured for this input shape; I took the closest measured figure, 21 s per update on the H200 pair for trajectory Q/A, and assumed 20–30 s per update on one H200.

| Step | GPU | Estimate |
|---|---|---|
| Training targets: 12k expansions + BM25 (CPU) + 12k thinking reads of text top-100 | 1 H200 | ≈ 2.5 h |
| AC uncontextual 1/10 and 1/20; AC contextual 1/10 and 1/20 (4 runs, ≈ 6.5k examples ≈ 800 updates each) | 1 GPU each | ≈ 5–7 h each |
| Reader LoRA with text top-10 (no AC) | 1 GPU | ≈ 3 h |
| Evaluation: 500 questions × (5 text settings + LoRA + 4 AC arms × k ∈ {10, 50, 100}) × 3 seeds | 1 GPU | ≈ 6–8 h |
| **Total** | | **≈ 35–45 GPU-hours ≈ 15–20 h wall on 2–3 GPUs** |

## 5. Open questions for you

1. **Rule reading.** HotpotQA's recall gap is +24 on all-gold recall but +14.9 on mean recall. I judged it on all-gold, which is what a two-passage question needs. OK?
2. **MuSiQue as a second benchmark?**
   - For: largest accuracy gain (+10.5), the paper already uses MuSiQue, and 6.6k training examples are expected (≈ 33% of 19,938).
   - Against: one search finds all the evidence for only 31% of questions.
3. **Query.** The 9B's query alone is about equal to the raw question for BM25. Question + query is slightly better: all-gold top-100 is 74.4 against 71.0. Keep "query alone", which is faithful to the design, or use question + query?
4. **Corpus.** Pooled HotpotQA paragraphs (IRCoT-style) or the standard 5.2M full-wiki abstracts? Full-wiki needs a CSAIL CPU indexing job and would lower recall at every cutoff.
5. **Evaluation size:** 500, as proposed above?

## 6. Compute used and files

- **CSAIL** (`csail-shared-h200` partition; one GPU at a time):
  - Job 2704894 on aia-h200-13: 32 min, cancelled by me. Its thinking prompt cut half the replies at 4,096 tokens.
  - Job 2705230 on quanta-h200-1, 19:50–20:25Z: 35 min, completed. It ran the v2 reads.
- **CSAIL folder:** `/data/scratch/ngom/activation/paper_retrieval_qa_v1/probe_20261018/` (≈ 200 MB: shipped corpora, outputs, logs). New folder; nothing replaced or deleted.
- **Container scratch:** `IB/TMP/PAPER_RETRIEVAL_QA/`
  - `bm25_recall.py`: corpus builders and recall.
  - `ship/rq_probe.py` and `rq_probe2.py`: the GPU probe.
  - `score_reads.py`
  - `results/`: CPU recall.
  - `probe_out*/`: queries, recall, reads, scores.
  - `data/`: downloads, ≈ 750 MB.
- **Sources:**
  - DPR, arXiv 2004.04906, Table 2.
  - Long-Context LLMs Meet RAG, arXiv 2410.05983, §3: BM25 accuracy keeps rising with more passages; NQ recall@40 is 0.73 for BM25 and 0.90 for e5.
  - IRCoT, arXiv 2212.10509, App. A: the pooled corpora.
  - MultiHop-RAG, arXiv 2401.15391.
  - QUEST, arXiv 2305.11694.
  - BrowseComp-Plus, arXiv 2508.06600.
  - FRAMES, arXiv 2409.12941.
