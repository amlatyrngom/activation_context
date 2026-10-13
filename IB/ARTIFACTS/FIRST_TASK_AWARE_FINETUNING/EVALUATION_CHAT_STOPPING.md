# Learner chat stopping: confirmed and corrected

The learner's HF sample decoder continued after the model emitted its chat-end token. A real Qwen3.5-9B checkpoint reproduced the saved malformed sample exactly. Adding chat end to the inherited EOS IDs preserved the answer and removed the fabricated later conversation.

| Same saved pair and prompt | Inherited stopping | Inherited + chat end |
|---|---:|---:|
| EOS IDs | 248044 | 248044, 248046 |
| Generated tokens | 160 | 47 |
| Tokens after first chat end | 113 | 0 |
| Decode time, this one example | 9.99s | 2.33s |
| Reached 160-token limit | Yes | No |

The first 47 generated tokens, including `<|im_end|>`, matched exactly. Prompt embeddings were bitwise identical; the inherited output matched the original midpoint sample. Shared generation configuration was unchanged. The complete raw token streams and provenance are in [GPU evidence](EVALUATION_CHAT_STOPPING_GPU.json). This timing is a one-example diagnostic, not a throughput benchmark.

The remote cached Qwen3.5-9B snapshot `c202236235762e1c871ad0ccb60c8ee5ba337b9a` has no `generation_config.json`. HF therefore derives EOS 248044 (`<|endoftext|>`) from its model config. The tokenizer and existing `ModelDescription.eot_token` identify chat end as 248046 (`<|im_end|>`). The original trainer passed no explicit EOS setting, then decoded with special tokens removed, hiding the crossed turn boundaries.

The product change is nine added lines in `ActivationContextTrainer._sample_completions`: copy inherited EOS IDs, append the existing chat-end ID if absent, and pass that list to both ordinary and AC-intervention generation. There is no new option or helper, and shared generation configuration is not mutated. Teacher-forced NLL code is untouched.

Focused review and real tiny-model CPU checks passed for both generation paths, including inherited scalar/list/None values, no-chat-end fallback, exact prefix preservation and configuration restoration. These use controlled token sequences through actual PEFT/HF generation. Receipt: `TMP/FIRST_TASK_AWARE_FINETUNING/chat_eos_union_cpu.json`. The separate real9B argument A/B, ORCD job23454544, completed successfully in 2m49s using the original frozen trainer and saved `oracle_27b_16k_r16/selected/update_01000` pair. It performed no scoring or training.

All seven forthcoming fixed100 readout conditions now use the corrected stopping arguments and retain actual returned IDs, special-token text, EOS termination and token-budget flags. The private wrapper runs over the same pinned trainer; material, selection, checkpoints and NLL batching are unchanged. Deployed worker SHA-256: `167aa74528eeb642f3b0318dc43194ebd2b2ad30d30cb6a07cacab192b003f90`.

Running training snapshots and the already-started long-cross protocol retain their original decoder. Historical samples remain unedited. This fixes extra conversation generation, not the correctness of the first answer. Oracle synthesis uses a separate vLLM path; this investigation made no synthesis change.
