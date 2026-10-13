# Unicode answer-mask repair

The first32k attempt (23478943) stopped during input preparation, before training. Qwen’s NFC tokenizer represents `N` plus a combining tilde as `Ñ`; the benchmark asserted equality with the original, decomposed spelling.

The new builder normalizes only the expected string used by that assertion, using the tokenizer’s own normalizer. Original gold strings, target token IDs, decoded text, answer boundaries, offset mapping and exact token-ID round-trip checks remain unchanged. No questions are dropped.

CPU job23479177 completed in84seconds. All32,000 training answers and400 development answers passed; only one training answer needed normalization. Every original16,000 training and400 development answer mask matched exactly. A genuinely wrong answer still fails. No model weights or confirmation questions were read.

The main run restarts fresh from originalP2. The16k×2 companion continues with its unchanged original builder. Versioned evaluation copies apply the same assertion fix to future confirmation input preparation; their scoring and checkpoint-guard function bodies are unchanged. Thirteen existing guard checks pass. All original source files and the original frozen design are preserved.

The [amended design](DATA_AMOUNT_32K_EVALUATION_V2.json) records the parent digest, both actual training entrypoints and all updated helper hashes. Data, initialization, LR, schedule, ratio, development selection and reserved confirmation remain fixed. [Validation receipt](ANSWER_MASK_REPAIR_CHECK.json).
