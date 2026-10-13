# 27B synthesis recovery — Sep22

Job 23424940 failed at 01:20 UTC after a generated JSON string decoded to the lone surrogate U+DCFF. `StudyCache.append` correctly refused to write that string as UTF-8. This was a text-format failure, not a disk/quota error or a model-engine failure; the later engine shutdown messages followed the exception.

The 15,000 published questions, committed surplus and valid cached prefix are intact. An independent reproduction confirmed that Python writes no bytes from the invalid JSONL row; preceding rows remain complete. No cache clearing, cursor reset, label repair or prompt change is needed.

The fix is seven lines in `activation/dataset/dataset_study.py`: validate each parsed question and answer as UTF-8 before adding it to the cache row. Reject an invalid pair, count a parse failure and retain valid siblings. An all-invalid generation is cached as an empty attempt, preserving restart behavior. Ordinary Unicode, including Japanese and emoji, and literal code escapes such as `\\udcff` remain valid.

Four focused private CPU checks passed: the existing source-selection/cache check, invalid-question and invalid-answer cases, and all-invalid attempt replay. They verify valid sibling preservation and that replay does not call the model again. The bounded review passed without additional changes.

New job **23441557** uses `train_ws_synthesis_unicode_v1`, copied from the exact 103-file training snapshot with only this parser changed. All shipped file hashes were verified; the original `train_ws_night_v1` is unchanged. The producer, prompt, recipe/cache IDs, target file and portable bank format remain unchanged. The active 9B generator and training jobs keep their original source snapshot.

The existing [implementation handoff](IMPLEMENTATION_HANDOFF.md) and [patch](implementation/changes.patch) now include this fix. Patch application to the frozen baseline reproduced every staged hash. Runtime confirmation: by01:35:36UTC the code-source cursor had advanced from the failed8640 block to8832, with the15k published bank intact. The resumed producer is generating and checking new questions normally.


For the 9B producer, the same fix is staged for its next natural requeue. Its separate snapshot `train_ws_9b_synthesis_unicode_v1` is cloned from that producer's own `train_ws`, not the newer training snapshot; a full small-source comparison verified that only `dataset_study.py` differs. `bench/bank9b_unicode_v1.sbatch` retains the original target, cache, batching and requeue behavior. The current segment continues unchanged.
