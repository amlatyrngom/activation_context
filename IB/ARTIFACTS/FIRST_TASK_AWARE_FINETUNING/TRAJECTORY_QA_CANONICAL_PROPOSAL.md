> Superseded as a canonical proposal: the user clarified on September24 that the requested canonical section concerns completed RAG QA. The text below is only an unaccepted suggestion for the next trajectory-QA task.

# Trajectory Q/A Training
- Data: target 100k synthetic questions from coding, research and tool-use histories; hold out whole trajectories before synthesis.
- Oracle: Qwen3.5-27B, no reasoning; grounded answers with support spans, covering facts, errors, decisions, state and multi-turn dependencies. No benchmark QA targets.
- Models: start from a compatible reconstruction pair, 2B encoder → 9B reader; reuse reconstruction architecture, LoRAs and hyperparameters.
- Encoding: original task/system/tools + chronological history; future question only reaches the reader. Start with one turn-preserving 8k–32k window, depth 1; confirm on a small trajectory pilot.
- Compression: start 1/16, extend across 1/10–1/30 rows/source tokens; no pooling.
- Training: fixed-answer SFT, fresh optimizer, initially one pass over unique questions; retain no-context KL preservation.
- Evaluation: frozen initial-reader full/no-context baselines with matched evidence/distractors. Aim toward full-context answer quality; track held/learning NLL, grounded answers, drift and cost. Report every 10% in unique questions × epochs.
