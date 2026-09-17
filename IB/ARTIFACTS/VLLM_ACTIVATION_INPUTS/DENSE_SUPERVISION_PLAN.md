# Round 7 — dense supervision: distil the full-context reader's per-layer attention outputs into the row-fed reader

Status: design, 22:45 UTC 09-16. Discussed with the user (22:15–22:40 UTC). To be reviewed (design review) and implemented in a worktree; apply only after a POC review.

## Why

Every deep variant so far (residual deltas, K/V deltas, row-over-passage attention; scaled, frozen and lora heads; learned K/V
slots) lands on the input-only control at 1/16 and 1/32 on both inits (results finding 18). The mechanism is a closed loop: the
warm reader does not read a new channel, so the channel's writer gets almost no gradient through it; the writer writes near-noise,
so the reader learns to read it even less. Zero-init heads make the channel neutral at init, not useful, and "write nothing" is a
stable point of the terminal loss. Round 6 (`kv_transfer`) attacks the reader side (native-format K/V). This round attacks the
gradient side: give the composition reader(writer(input)) a dense, fixed, per-layer target that does not depend on whether the
reader is currently listening, so "write nothing" is no longer a stable point and the rows are pushed toward delivering what the
passage would have delivered, whether or not the reader has started to use them.

## What is supervised

- **Teacher**: the base model with no adapters and no rows, on the full-context history (`in_context_messages`: the passage text
  in place of the placeholder run; the paired-history machinery of the KL mode, `validate_paired_histories`). No gradient.
- **Student**: the reader as trained (rows in place of the passage, side pass, any intervention target).
- **Supervised positions**: the target (output) tokens, which are the same tokens at aligned positions in both sequences (the
  passage tokens exist only in the teacher, the row positions only in the student). Alignment via the existing teacher/student
  start offsets.
- **Supervised tensor**: at each chosen full-attention reader layer, the attention block's output at the supervised positions
  (`self_attn` output after `o_proj`, i.e. the residual contribution of attention; option `distill_source "hidden"` = the
  residual stream after the layer instead). The target is the teacher's own tensor there: it does not depend on the student, so
  there is no shortcut (a student query that makes "rows-answer" and "passage-answer" trivially agree); the gradient flows into
  everything upstream in the student — the writer (encoder, row modules, side pass, K/V heads) and the reader LoRA at layers ≤ l.

## Loss

```
L = terminal (as now: SFT or KL, penalty unchanged)
  + λ(step) · mean_over_layers_l  mean_over_positions  ‖ A_student,l − A_teacher,l ‖² / rms_l²
```

- `rms_l` = the teacher tensor's RMS at layer l over the supervised positions of the current example (detached), so every layer's
  term is O(1) at the start and λ is interpretable (1 = parity with the terminal loss).
- Layers: `distill_layers` = explicit list, or `-1` = every full-attention layer (default: the intervention layers if
  interventions are configured, else every full-attention layer).
- **Schedule** `λ(step)`: hold `distill_weight` for the first `distill_hold` fraction of the run (default 0.25), decay linearly to
  `distill_floor` (default 0.05 × weight) at `distill_decay_end` (default 0.75), hold the floor to the end. The clock is the
  schedule clock already used for the LR (resume-safe). Log λ.
- Bit-identity: `distill_weight 0` (default) must leave every existing configuration byte-identical (no teacher pass, no hooks).

## Cost

One extra no-grad forward of the base model on the full-context history per example (bf16, adapters disabled, no checkpointing
needed), keeping only the hooked tensors at the chosen layers and supervised positions (≈ 8 × T × 2560 × 2 B). Expected ≈ +30–40 %
step time; no extra memory of note. Inference is untouched; nothing changes for vLLM.

## Widgets

- `distill_loss` — the normalised term per layer (trace every 5 updates for the first 200, then every update in the log).
- `distill_lambda` — λ per update.
- `rows_attention_share` (if cheap) — at the supervised positions, the share of attention mass that lands on the row positions,
  per chosen layer, computed on the held-out panel from a manual attention over the chosen layers' q/k for a few items. Tells
  whether the reader has started reading the rows and whether the gated (rather than clocked) schedule would have fired.

## Bench keys and arms

Keys: `distill_weight`, `distill_layers`, `distill_source`, `distill_hold`, `distill_decay_end`, `distill_floor`.
Arms (all 1/32 or 1/16 warm from the K16r1 epoch-2 adapters, ND schedule, 8,192 × 1; twins `…n` from the K16 adapters on
ac-recon-2 with the /root paths):

| arm | base | distill |
|---|---|---|
| A32XD | A32X (input-only, 1/32) | weight 1, all full-attention layers, attn_out |
| A32XD3 | A32X | weight 3 |
| A32XDH | A32X | weight 1, source hidden |
| NXD | WCa/WCn (input-only, 1/16) | weight 1 |
| A32KVTAD | A32KVTA (kv_transfer, every layer) | weight 1 |

Read: held − control (A32X 0.322 / A32W 0.340; WCn 0.122 / WCa 0.130), `distill_loss` trajectory per layer, and for the
kv_transfer arm `intervention_takeover`.

## Decision gate

If the round-6 ceilings (NKVTF / A32KVTF) do not beat the input-only control with a warm reader, this objective goes into the
seed run (stage 1 of the golden recipe, from scratch) rather than into a warm pass; the implementation is the same.
