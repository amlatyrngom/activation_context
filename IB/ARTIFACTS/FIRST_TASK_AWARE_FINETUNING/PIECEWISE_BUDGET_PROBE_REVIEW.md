# Encoder budget probe review

**Verdict: pass for the bounded GPU measurement; no budget change adopted.** The private helper changes only the existing encoder batch budget. Active training source, configuration, masks and checkpoint selection remain fixed.

The worker requires the piecewise arm’s own completed V2 final pair, matching frozen identity/source/material and full500-update Adam state. The numerical proof compares the same two recorded training cases, fixed actual penalty selections, all encoder outputs, every owned gradient and joint loss, including an unchanged repeat. It retains the prior deterministic gates; ordinary production timings run separately with matching flags. Any failed proof stops timing and preserves evidence.

Reviewed the four-update timing path against the actual pinned trainer38ca9885, separately from the real tiny-model CPU numerical check on the current workspace source. The inherited clock advances500→504, all four updates use the terminal rates, no baseline loss calls consume the recorded penalty stream, and all checkpoint saves including duration-triggered saves are disabled. Each condition starts from the same paired weights and Adam state; final hashes detect parent mutation. CPU numerical/negative-guard receipts are reused, not rerun without a code change.

Queue only after both own8k final readouts23498368 and23498373 succeed. This also gates on both training completions. Request45minutes, one RTX PRO GPU and96GiB host RAM; controller budget42minutes. The measured benefit remains unknown. Source-length geometry predicts less padding, but extra forward calls may offset the gain. No head optimization or held-question selection is part of this test.

Files: bench/probe_piecewise_budget_v1.py (819d0beed15dae7d487343e2c1578491d2e95fc0c4a4a8145e7bb66ba0be3938), launcher (d8fa44c5011a285a953740f6736b6e1f5b43c588f13a65e84e04be55922ce776). Detailed CPU/source-control-flow evidence: TMP/FIRST_TASK_AWARE_FINETUNING/PIECEWISE_BUDGET_CPU_CHECK.json.

Current scheduler recovery (23:20 UTC): original23509310 was cancelled after merged training preemption. Replacement23510989 waits for23498368 and23510986. Scientific protocol unchanged; see PIECEWISE_PREEMPTION_2315_RECOVERY.json.
