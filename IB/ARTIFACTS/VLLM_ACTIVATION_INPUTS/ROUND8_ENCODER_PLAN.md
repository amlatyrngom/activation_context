# Round 8 — spend compute at write time: an iterative encoder and a lexical-anchor loss

Status: design, 01:20 UTC 09-17. Motivated by the user's request (01:08 UTC) for more original, impactful interventions now that
bandwidth is less constrained. To be design-reviewed and implemented in a worktree; POC review before applying.

## Where the loss lives (what the night established)

- The reader is not the limit. With the side model's K/V of every passage token spliced at the full-attention layers the
  reconstruction is exact (A32KVTF / NKVTF ≈ 0.000). With the same splice restricted to the 1/32 rows it is 0.30 (A32KVTA),
  vs 0.32 for the input-only reader. Everything between 0.30 and 0.00 is what the row budget costs: **how much of the passage
  the encoder packs into V rows** is the whole problem.
- Adding channels that carry the *same* rows in a better format buys 0.02 at 1/32 and nothing at 1/16. Dense supervision of
  the rows against the full-context reader (round 7) shapes the rows as intended and does not move the terminal loss in a
  warm pass. So the next lever is the encoder itself: make the rows better, with more compute at write time — which is free
  at inference (the reader / vLLM side is untouched) and cheap in training (the side pass is a fraction of the step).
- The residual errors are mostly exact identifiers and numbers (results finding 16): the rows lose *lexical* content first.

## Two encoder-side features, independent, both vLLM-neutral

### A. Iterative encoder (`encoder_passes` = 2, 3; default 1 = today)

Today (`ac_model.py: _encode_prepared_batch`): content embeddings → blocked-local mixer → (pooling) → V pooled summaries with a
marker and positions → **one** causal side-decoder pass over `[content ; summaries]` → the final V positions are the rows
(`target_head` into the reader space; with `kv_transfer` the rows' K/V at the chosen layers are captured in the same pass).

Pass k ≥ 2: run the side decoder again over `[content ; summaries_k]` where `summaries_k` are the previous pass's final row
states mapped back into the side input space — `unit_rows(prev_rows_side) + summary_marker + positions`, through a small
zero-init residual adapter (`refine_adapter`: d_side → d_side, LoRA-style zero-init so pass 2 starts as a plain re-read of the
rows) — plus the pass-1 summaries as a residual (so the input to pass 2 is `summaries_1 + refine(prev)`). The rows and the
transferred K/V come from the **last** pass. The mixer and pooling run once (shared content). Gradient flows through all passes
(non-reentrant checkpointing per pass as today). Cost: +1 side forward per extra pass; reader untouched; inference cost at the
reader identical; write-time cost ×passes (offline).

Why it could matter: the first pass's rows see the passage causally with the summaries at the end — every summary attends to
the whole passage but the summaries cannot see *each other's* final content when they are formed. A second pass lets each row
read what the other rows already captured and fill the gaps (and lets the K/V at the rows be computed from states that already
know the row assignment). This is the "encoder depth" lever that recurrent/iterative compressors use.

### B. Lexical-anchor auxiliary loss (`anchor_weight`, default 0 = off)

A writer-side auxiliary objective that supervises the rows directly on the content they lose first: from the rows, predict the
passage's **rare tokens**. Per part, take the set of passage token ids whose unigram frequency (from the training material,
computed once in preparation, cached) is below a quantile (`anchor_rare_quantile`, default 0.2 of the vocabulary mass — i.e.
identifiers, numbers, paths, hashes); build a multi-hot target over the vocabulary restricted to a sampled negative set
(`anchor_negatives`, default 2,048 random tokens + all positives) to keep it cheap; the rows (reader-space, after `target_head`)
go through a tiny head (`anchor_head`: mean over rows → d → vocab-slice logits via the reader's tied embedding matrix restricted
to the candidate set, no new large parameters) and a binary cross-entropy against the multi-hot target. Weight `anchor_weight`
(default 0.1 when on), constant. Bit-identical at 0. Inference untouched.

Why it could matter: the terminal loss gives the rows a weak, late signal about identifiers (they are few tokens among many);
this term is dense, cheap, and specific to the failure mode. It is the writer-side counterpart of round 7 with a target the rows
can actually satisfy at 1/32 (a set, not per-position activations).

## Bench keys and arms

Keys: `encoder_passes`, `anchor_weight`, `anchor_rare_quantile`, `anchor_negatives`.
Arms (1/32 warm from the K16r1 epoch-2 adapters, ND schedule, 8,192 × 1; twins `…n` on the K16 init with /root paths):

| arm | base | change |
|---|---|---|
| A32P2 | A32W (input-only) | encoder_passes 2 |
| A32P3 | A32W | encoder_passes 3 |
| A32AN | A32W | anchor_weight 0.1 |
| A32P2AN | A32W | passes 2 + anchor 0.1 |
| A32KVTAP2 | A32KVTA (kv_transfer, every layer) | encoder_passes 2 |
| NP2 | WCa (1/16) | encoder_passes 2 |

Read: held − control (A32W 0.340 / A32X 0.322; A32KVTA 0.322 / A32KVTAn 0.300; WCa 0.130 / WCn 0.122); for the anchor arms
also the anchor loss curve and the held-out accuracy on rare tokens (add a `rare_token_accuracy` panel: teacher-forced accuracy
restricted to target tokens in the rare set — that is the number the whole thing is about).

## Not in this round

Adaptive row placement (rows are pooled summaries, not placeholders; a content-adaptive pooling is a separate design), a
verbatim rare-token sidecar (changes what the ratio means), the seed-run version of round 7 (compute, not code).

## Addendum after the design review (01:25 UTC 09-17; DESIGN_REVIEW_8.md: SOUND WITH CHANGES)

- **A kept**, with the review's blockers: captures opened around the last pass only (the side capture contexts record the first forward only), `passes_off` panel and `refine_rms` trace, forked-RNG init and migrate-set discipline, refused with `kv_transfer_full` / `cross_attention`. Cost note corrected: pass 2 recomputes the content positions (side cost ×2); a cache-continued rows-only pass is the production form, later.
- **B dropped.** A bag of rare tokens is order-blind (the identifier errors are recombinations of pieces with identical bags) and a unigram rule never marks digits rare (Qwen tokenises digits singly, so 1798 → 1778 is a swap among the most frequent tokens). Replaced by the review's two surprisal-based features, both driven by the base model's no-context per-token surprisal the trainer already stores:
  - **B′ `rare_weight` β** — surprisal-weighted terminal loss, w_t = 1 + β·s_t/mean(s), normalised to mean 1 per example; metric unweighted; bit-identical at 0.
  - **C `pool_surprisal` γ** — summary-pooling bins drawn by cumulative surprisal instead of token count (rows per part unchanged) plus a zero-init surprisal bias on the pooling attention logits; the only lever that moves row capacity from cheap tokens to expensive ones. Bit-identical at 0.
  - Widget `rare_token_loss`: held-out loss restricted to the top-20 %-surprisal target tokens (from the stored per-token log-probs).
- **Arms**: A32P2, A32KVTAP2, A32SW (β 1), A32SP (γ 1), A32SPKVTA (transfer + γ 1), NSP (1/16, γ 1), and their `…n` twins. A32P3 only if A32P2 is ≥ 0.005 below control. A win must agree in sign on both inits.
