# Multi-GPU engine round 2

Diff cards on top of your ongoing adaptation (workspace resynced from `/source` first). They answer
your six `@AI` notes and fix three things the port left half-way: `engine_to_device` still builds a
bare `vllm.LLM` (so no replicas, and the free path calls a `shutdown()` that `vllm.LLM` lacks), the
`engine_kwargs is None` branch can never fire because `ModelConfig` defaults to `{}`, and the label
pass still runs on the study model. Every card is the exact delta of the workspace file against your
current `/source` tree; the staged copies under `IB/ARTIFACTS/ACTIVATION_CONTEXT_INIT/activation/`
match the cards. Your `harness/__init__.py` (exports `RECOMMENDED_BATCH_SIZE` only) and your
`engine_free_other_models()` placement are kept as they are and carry no card.

Round 1 of this handoff (the original M1–M3 cards plus the GPU gate results) is the previous
version of this file; those numbers are unchanged and summarized in `PLAN.tressoir.md`.

## Where the chat recommendation is applied

`VLLMWrapper.recommended_chat_kwargs(model_id)` is applied inside `LoadedModel.engine_chat_many`,
mirroring your engine-kwargs merge: harness defaults (`temperature 1.0`, `max_tokens 1024`) →
recommendation → the caller's `chat_kwargs`, key by key inside `sampling_params` and for the extra
chat kwargs. So `dataset_study_chat_kwargs=None` means "recommended" for both passes, each model gets
its own recommendation, and any explicit knob still wins. The study generator keeps adding its
prompt-specific parameters (`max_tokens 2048`, `seed`, `structured_outputs`).

## Renames, as applied

| Old | New | Note |
| --- | --- | --- |
| `dataset_study_model_name` | `dataset_study_qa_model_name` | |
| `dataset_study_batch_size` | `dataset_study_qa_batch_size` | |
| `dataset_stuby_label_model_name` | `dataset_study_label_model_name` | typo; `None` falls back to the QA model |
| `DatasetStudyGenerator.study_model_name` / `batch_size` | `qa_model_name` / `qa_batch_size` | |
| `DatasetStudyGenerator.generate_study_examples` | `generate_example_qa` | manager method name unchanged |
| test `STUDY_MODEL_NAME` | `QA_MODEL_NAME` | |

Not renamed, because both passes read them: `dataset_study_chunk_input_limit`,
`dataset_study_chat_kwargs`, `dataset_study_seed`, and the `study_*` stats fields. Say so if you want
the `qa_` prefix on the stats too; that touches `dataset.py` and `summarize()`.

## Validation

| Check | Result |
| --- | --- |
| `py_compile` of every touched module; no `@AI`, `STUDY_MODEL_NAME`, `study_model_name`, or old knob names left in `activation/` | OK |
| `recommended_chat_kwargs` table: known id → Qwen non-thinking sampling, unknown id → `{}` plus the note | OK |
| `engine_chat_many` merge order with the engine stubbed on CPU (private): bare call carries `temperature 0.7, top_p 0.8, top_k 20, min_p 0, presence_penalty 1.5, max_tokens 1024, enable_thinking False`; an explicit `temperature 0.3, max_tokens 5` overrides those two and keeps the rest | OK |
| `test_basic_dataset_loading.py -k bm25` on CPU | `2 passed in 21.48s` |
| Private labeling contract check (`IB/TMP/RETRIEVAL_TRAINING/private_label_contract_check.py`, knob names updated) | `CONTRACT CHECK OK`; log `IB/TMP/MULTI_GPU/contract_check_round2.log` |
| GPU study test | **Not re-run** for this round; the engine and chat kwargs that reach vLLM are the same values the four gated runs used, minus the sampling `min_p 0.0`, which is vLLM's default anyway. Say the word for a node. |

The IB-only training entry script (`IB/ARTIFACTS/RETRIEVAL_TRAINING/activation/training/scripts/train_bright_retriever.py`) follows the knob renames; no card, as before.

## Post-adaptation fixes (round 3)

Three cards against your adapted `/source`. Two of them would fail `test_basic_dataset_study` before
any engine loads: `HarnessRuntimeConfig` still spells the knob `dataset_stuby_label_model_name` while
the test passes `dataset_study_label_model_name` (a `TypeError` at construction), and the manager
calls `generate_examples_qa` / `generate_example_labels` while the generator defines
`generate_example_qa` / `generate_examples_labels` (an `AttributeError` on each pass). I took the
plural form for both generator methods, matching your `synthesize_study_examples_qa`. The third
restores the fallback to the QA model when the label knob is unset and removes the last `@AI` note.
Round 1 and round 2 cards above are already in your tree and stay for the record.

**Adapted (2026-09-04).** All three landed except the fallback: you kept `dataset_study_label_model_name` required (no `or self.qa_model_name`), so every config that labels must name a label model, the same one as the QA model if desired. Workspace and staged tree follow your tree; nothing is outstanding.

**GPU runs on the adapted tree (2026-09-04, fresh `ac-mgpu-final`, g7e.12xlarge us-east-1, torn down):**

| Leg | Result | Engine loads (QA / label) | Study output tok/s | Label prompt tok/s | Labels |
| --- | --- | --- | --- | --- | --- |
| 2× RTX PRO 6000, cold | `1 passed in 1617.87s` | 732.0 s / 766.1 s, two replicas each | 38.6 (JIT in first batch) | 30.4k | 64 pos / 276 neg / 0 ambiguous / 0 parse failures |
| 1× (`CUDA_VISIBLE_DEVICES=0`), warm | `1 passed in 118.71s` | 38.7 s / 56.3 s | 512.3 | 28.4k | 64 / 276 / 0 / 0 |

Residency asserts `[QA, LABEL]` / `[QA, LABEL]` held on both; engine and chat kwargs came entirely from the wrapper's recommendations (no `engine_kwargs`, no `dataset_study_chat_kwargs` in the test). Logs: `IB/TMP/MULTI_GPU/final_rtx{2,1}_study_test.log`.

The IB-only training entry script was stale twice over (`dataset_study_num_samples` no longer exists
as a knob, and it called `synthesize_study_examples` without a count); it now calls
`synthesize_study_examples_qa(dataset_id, args.study_num_samples)`. No card, as before.

Validation on CPU: `py_compile`, a sweep for leftover old names and notes, and the private labeling
contract check (`IB/TMP/MULTI_GPU/contract_check_round3.log`, `CONTRACT CHECK OK`).

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">activation/harness/runtime_config.py</span>
    <span class="card-oneliner">Label-model knob spelled the way the test passes it.</span>
    <span class="card-badge">Diff</span>
  </summary>

Also fixes the "recommeded" comment. Exact delta vs `/source`:

````diff-python
diff --git a/source/activation/harness/runtime_config.py b/activation/harness/runtime_config.py
index c670175..6c89dcd 100644
--- a/source/activation/harness/runtime_config.py
+++ b/activation/harness/runtime_config.py
@@ -26,7 +26,7 @@ class HarnessRuntimeConfig:
 
 
     # Dataset Study
-    dataset_study_qa_batch_size: int|None = None # None = recommeded value.
+    dataset_study_qa_batch_size: int|None = None # None = recommended value.
     dataset_study_qa_model_name: str|None = None
     dataset_study_chunk_input_limit: int|None = None # Limit for fast testing.
     dataset_study_chat_kwargs: dict|None = None
@@ -34,7 +34,7 @@ class HarnessRuntimeConfig:
     dataset_study_label_top_k: int = 10 # bm25 candidates per question.
     dataset_study_label_pool_max_chars: int = 32768 # Char budget of the snippet pool shown to the labeler.
     dataset_study_label_batch_size: int|None = None # None = recommended value.
-    dataset_stuby_label_model_name: str|None = None
+    dataset_study_label_model_name: str|None = None # None = the QA model labels too.
 
 @dataclass
 class HarnessStats:
````

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">activation/dataset/dataset_study.py</span>
    <span class="card-oneliner">`generate_examples_qa`, label fallback to the QA model, note removed.</span>
    <span class="card-badge">Diff</span>
  </summary>

Exact delta vs `/source`:

````diff-python
diff --git a/source/activation/dataset/dataset_study.py b/activation/dataset/dataset_study.py
index b7f4b84..60c324e 100644
--- a/source/activation/dataset/dataset_study.py
+++ b/activation/dataset/dataset_study.py
@@ -137,9 +137,8 @@ class DatasetStudyGenerator:
         self.dataset_id = loaded_dataset.dataset_id
         self.dataset_index = harness.dataset_manager.dataset_indexes[self.dataset_id]
         self.study_context = loaded_dataset.study_context
-        # @AI: Rename to qa_model_name, qa_batch_size, ... vs label_model_name, etc
         self.qa_model_name = harness.harness_config.dataset_study_qa_model_name
-        self.label_model_name = harness.harness_config.dataset_stuby_label_model_name
+        self.label_model_name = harness.harness_config.dataset_study_label_model_name or self.qa_model_name
         self.qa_batch_size = harness.harness_config.dataset_study_qa_batch_size or RECOMMENDED_BATCH_SIZE
         self.label_batch_size = harness.harness_config.dataset_study_label_batch_size or RECOMMENDED_BATCH_SIZE
         self.chunk_input_limit = harness.harness_config.dataset_study_chunk_input_limit
@@ -165,7 +164,7 @@ class DatasetStudyGenerator:
         return chat_kwargs
 
 
-    def generate_example_qa(self, num_samples: int) -> list[LabeledRetrievalQAExample]:
+    def generate_examples_qa(self, num_samples: int) -> list[LabeledRetrievalQAExample]:
         loaded_model = self.harness.loaded_models[self.qa_model_name]
         # vllm batches a whole conversation list inside one chat() call, so a
         # simple single-threaded loop needs no locks.
````

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">activation/dataset/dataset_manager.py</span>
    <span class="card-oneliner">`label_study_examples` calls the plural generator method.</span>
    <span class="card-badge">Diff</span>
  </summary>

Exact delta vs `/source`:

````diff-python
diff --git a/source/activation/dataset/dataset_manager.py b/activation/dataset/dataset_manager.py
index 9daaf1d..b652825 100644
--- a/source/activation/dataset/dataset_manager.py
+++ b/activation/dataset/dataset_manager.py
@@ -66,7 +66,7 @@ class DatasetManager:
         synthetic_only: bool = False,
     ):
         study_generator = self._get_or_create_study_generator(dataset_id)
-        return study_generator.generate_example_labels(num_samples, synthetic_only)
+        return study_generator.generate_examples_labels(num_samples, synthetic_only)
 
 
     def select_training_data(
````

</details>

## Diffs to apply (round 2)


<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">activation/harness/vllm_wrapper.py</span>
    <span class="card-oneliner">`recommended_chat_kwargs` completed.</span>
    <span class="card-badge">Diff</span>
  </summary>

Your "Complete me" note. Same shape as `recommended_engine_kwargs`: a table keyed by the four known ids, all pointing at the Qwen non-thinking sampling (Qwen3.6 and Qwen3.8 publish the same one; `min_p 0.0` added to your four values), an unknown id returns `{}` with a printed note so engine defaults apply, and a fresh dict comes back every call. Prompt-specific parameters stay with the caller, `enable_thinking=False` stays the engine default. Exact delta vs `/source`:

````diff-python
diff --git a/source/activation/harness/vllm_wrapper.py b/activation/harness/vllm_wrapper.py
index db709dd..20bc1fd 100644
--- a/source/activation/harness/vllm_wrapper.py
+++ b/activation/harness/vllm_wrapper.py
@@ -68,21 +68,33 @@ class VLLMWrapper:
     @staticmethod
     def recommended_chat_kwargs(model_id: str, is_for_training: bool = False) -> dict:
         """
-        Provides recommended chat kwargs for the given model.
-        Should be augmented with prompt-specific parameters.
+        Provides recommended chat kwargs for the given model, in `engine_chat_many` shape
+        (`sampling_params` is a dict of `vllm.SamplingParams` kwargs). Prompt-specific parameters
+        (`max_tokens`, `seed`, `structured_outputs`) belong to the caller; `enable_thinking=False`
+        is already the engine default. Unknown ids get no recommendation (engine defaults) and a
+        printed note; never raise.
         """
-        assert not is_for_training, "Recommended engine kwargs for training rolouts not yet set."
-        # @AI: Complete me.
-        return {
-            # Qwen3.8 instruct-mode sampling recommendation.
+        assert not is_for_training, "Recommended chat kwargs for training rollouts not yet set."
+        # Qwen3.6 / Qwen3.8 publish the same non-thinking sampling; presence_penalty curbs repetition.
+        qwen_non_thinking = {
             "sampling_params": {
                 "temperature": 0.7,
                 "top_p": 0.8,
                 "top_k": 20,
+                "min_p": 0.0,
                 "presence_penalty": 1.5,
             },
         }
-        
+        known = {
+            "unsloth/Qwen3.8-27B-NVFP4": qwen_non_thinking,
+            "Qwen/Qwen3.8-27B-FP8": qwen_non_thinking,
+            "nvidia/Qwen3.6-35B-A3B-NVFP4": qwen_non_thinking,
+            "Qwen/Qwen3.6-35B-A3B-FP8": qwen_non_thinking,
+        }
+        if model_id not in known:
+            print(f"{model_id} - No recommended chat kwargs; using engine defaults.")
+        return {key: dict(value) for key, value in known.get(model_id, {}).items()}
+
 
     def chat(self, conversations: list, **chat_kwargs) -> list[vllm.RequestOutput]:
         """Same signature as vllm.LLM.chat; strided shards, one thread per replica, in-order results."""
````

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">activation/harness/loaded_model.py</span>
    <span class="card-oneliner">Construct `VLLMWrapper` again; `engine_chat_many` applies the chat recommendation under explicit kwargs.</span>
    <span class="card-badge">Diff</span>
  </summary>

Two things. First, your `engine_to_device` builds `vllm.LLM` directly while the field is typed `VLLMWrapper` and the free path calls `.shutdown()`, which `vllm.LLM` does not have, so the first free would raise and no replicas would be built; the card restores `VLLMWrapper(**engine_kwargs)`. Second, `engine_chat_many` now merges harness defaults → `recommended_chat_kwargs(model_id)` → caller kwargs, both for the `sampling_params` dict and for the extra chat kwargs, mirroring what you did for engine kwargs. That is what makes `dataset_study_chat_kwargs=None` mean "recommended", and it is per model, so the labeler gets its own recommendation. Your `engine_free_other_models()` placement inside `engine_to_device` is kept. Exact delta vs `/source`:

````diff-python
diff --git a/source/activation/harness/loaded_model.py b/activation/harness/loaded_model.py
index 9d6241f..b807c6b 100644
--- a/source/activation/harness/loaded_model.py
+++ b/activation/harness/loaded_model.py
@@ -201,7 +201,7 @@ class LoadedModel:
                 self.model_config.engine_kwargs = VLLMWrapper.recommended_engine_kwargs(self.model_config.model_id)
             engine_kwargs.update(self.model_config.engine_kwargs)
             self.model_config.engine_kwargs = engine_kwargs
-            self.vllm_model = vllm.LLM(**engine_kwargs)
+            self.vllm_model = VLLMWrapper(**engine_kwargs)
             self.current_engine_device = device
             elapsed = time.time() - device_change_start_time
             print(f"{self.model_config.model_name} - Engine moved to target in {elapsed:.2f}s")
@@ -258,14 +258,17 @@ class LoadedModel:
         Batched engine chat. Returns texts with token counts for stats.
         """
         self.engine_to_device(TARGET_DEVICE)
+        # Harness defaults < recommended chat kwargs for this model < explicit chat kwargs.
         chat_kwargs = dict(chat_kwargs or {})
+        recommended_chat_kwargs = VLLMWrapper.recommended_chat_kwargs(self.model_config.model_id)
         default_sampling_params_kwargs = dict(
             temperature=1.0,
             max_tokens=1024,
         )
+        recommended_sampling_params_kwargs = recommended_chat_kwargs.pop("sampling_params", None) or dict()
         chat_sampling_params_kwargs = chat_kwargs.pop("sampling_params", None) or dict()
         sampling_params = vllm.SamplingParams(
-            **(default_sampling_params_kwargs | chat_sampling_params_kwargs)
+            **(default_sampling_params_kwargs | recommended_sampling_params_kwargs | chat_sampling_params_kwargs)
         )
         default_extra_kwargs = dict(
             chat_template_kwargs={"enable_thinking": False},
@@ -274,7 +277,7 @@ class LoadedModel:
             conversations,
             sampling_params=sampling_params,
             use_tqdm=False,
-            **(default_extra_kwargs | chat_kwargs)
+            **(default_extra_kwargs | recommended_chat_kwargs | chat_kwargs)
         )
         return [
             EngineChatOutput(
````

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">activation/harness/model_config.py</span>
    <span class="card-oneliner">`engine_kwargs` defaults to `None` so your recommendation branch can fire.</span>
    <span class="card-badge">Diff</span>
  </summary>

Your `engine_to_device` checks `if self.model_config.engine_kwargs is None`, but the dataclass default is `field(default_factory=dict)`, so a bare `ModelConfig` never gets the formula (the GPU runs I reported used my merge, not this branch). `None` is now the default and documented. `field` stays imported for `model_description`. Exact delta vs `/source`:

````diff-python
diff --git a/source/activation/harness/model_config.py b/activation/harness/model_config.py
index 291a245..67d4ca0 100644
--- a/source/activation/harness/model_config.py
+++ b/activation/harness/model_config.py
@@ -71,8 +71,8 @@ class ModelConfig:
     model_description: ModelDescription|None = None
     """Detailed Model Description. Auto-populated."""
 
-    engine_kwargs: dict[str, t.Any] = field(default_factory=dict)
-    """Additional arguments passed to the engine. Override default ones."""
+    engine_kwargs: dict[str, t.Any]|None = None
+    """Additional arguments passed to the engine. Override default ones. None: VLLMWrapper.recommended_engine_kwargs."""
 
 
     def pretty_format_description(self) -> str:
````

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">activation/harness/runtime_config.py</span>
    <span class="card-oneliner">QA knobs renamed; label-model typo fixed.</span>
    <span class="card-badge">Diff</span>
  </summary>

Your rename note. Only the two knobs that belong to the QA pass get the prefix: `dataset_study_qa_model_name`, `dataset_study_qa_batch_size`. `dataset_study_chunk_input_limit`, `dataset_study_chat_kwargs`, and `dataset_study_seed` are read by both passes (the pool builder truncates with the same limit, the label selection shuffles with the same seed), so they keep the shared name. `dataset_stuby_label_model_name` becomes `dataset_study_label_model_name`, which your test already passes. Exact delta vs `/source`:

````diff-python
diff --git a/source/activation/harness/runtime_config.py b/activation/harness/runtime_config.py
index 221a517..8d304f0 100644
--- a/source/activation/harness/runtime_config.py
+++ b/activation/harness/runtime_config.py
@@ -26,16 +26,15 @@ class HarnessRuntimeConfig:
 
 
     # Dataset Study
-    # @AI: rename to dataset_study_qa_batch_size, ...
-    dataset_study_batch_size: int|None = None # None = recommeded value.
-    dataset_study_model_name: str|None = None
+    dataset_study_qa_model_name: str|None = None
+    dataset_study_qa_batch_size: int|None = None # None = recommended value.
     dataset_study_chunk_input_limit: int|None = None # Limit for fast testing.
     dataset_study_chat_kwargs: dict|None = None
     dataset_study_seed: int = 0 # Seeds chunk sampling and generation sampling.
     dataset_study_label_top_k: int = 10 # bm25 candidates per question.
     dataset_study_label_pool_max_chars: int = 32768 # Char budget of the snippet pool shown to the labeler.
     dataset_study_label_batch_size: int|None = None # None = recommended value.
-    dataset_stuby_label_model_name: str|None = None
+    dataset_study_label_model_name: str|None = None # None = the QA model labels too.
 
 @dataclass
 class HarnessStats:
````

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">activation/dataset/dataset_study.py</span>
    <span class="card-oneliner">`qa_*` attributes, `generate_example_qa`, label pass on the label model.</span>
    <span class="card-badge">Diff</span>
  </summary>

Your two rename notes. `qa_model_name` / `qa_batch_size` versus `label_model_name` / `label_batch_size`; `label_model_name` falls back to the QA model when the knob is `None` (your line had dropped the fallback, which would have made `loaded_models[None]` fail for single-model setups). The label pass reads `loaded_models[self.label_model_name]`; your copy still used the study model, so the labeler never ran and the residency assert could not hold. Both batch sizes default to `RECOMMENDED_BATCH_SIZE` as you set them. Exact delta vs `/source`:

````diff-python
diff --git a/source/activation/dataset/dataset_study.py b/activation/dataset/dataset_study.py
index 2c0aef3..9f60204 100644
--- a/source/activation/dataset/dataset_study.py
+++ b/activation/dataset/dataset_study.py
@@ -137,10 +137,9 @@ class DatasetStudyGenerator:
         self.dataset_id = loaded_dataset.dataset_id
         self.dataset_index = harness.dataset_manager.dataset_indexes[self.dataset_id]
         self.study_context = loaded_dataset.study_context
-        # @AI: Rename to qa_model_name, qa_batch_size, ... vs label_model_name, etc
-        self.study_model_name = harness.harness_config.dataset_study_model_name
-        self.label_model_name = harness.harness_config.dataset_stuby_label_model_name
-        self.batch_size = harness.harness_config.dataset_study_batch_size or RECOMMENDED_BATCH_SIZE
+        self.qa_model_name = harness.harness_config.dataset_study_qa_model_name
+        self.label_model_name = harness.harness_config.dataset_study_label_model_name or self.qa_model_name
+        self.qa_batch_size = harness.harness_config.dataset_study_qa_batch_size or RECOMMENDED_BATCH_SIZE
         self.label_batch_size = harness.harness_config.dataset_study_label_batch_size or RECOMMENDED_BATCH_SIZE
         self.chunk_input_limit = harness.harness_config.dataset_study_chunk_input_limit
         self.chat_kwargs = harness.harness_config.dataset_study_chat_kwargs
@@ -165,9 +164,8 @@ class DatasetStudyGenerator:
         return chat_kwargs
 
 
-    # @AI: Rename me to generate_example_qa(...)
-    def generate_study_examples(self, num_samples: int) -> list[LabeledRetrievalQAExample]:
-        loaded_model = self.harness.loaded_models[self.study_model_name]
+    def generate_example_qa(self, num_samples: int) -> list[LabeledRetrievalQAExample]:
+        loaded_model = self.harness.loaded_models[self.qa_model_name]
         # vllm batches a whole conversation list inside one chat() call, so a
         # simple single-threaded loop needs no locks.
         loaded_model.ensure_engine_loaded() # Keep the cold load out of the batch timings.
@@ -179,8 +177,8 @@ class DatasetStudyGenerator:
         example_count = 0
         total_generation_time = 0
         reporting_interval = 0
-        for batch_start in range(0, len(study_samples), self.batch_size):
-            batch = study_samples[batch_start:batch_start + self.batch_size]
+        for batch_start in range(0, len(study_samples), self.qa_batch_size):
+            batch = study_samples[batch_start:batch_start + self.qa_batch_size]
             conversations = []
             for chunk in batch:
                 snippet = self.dataset_index.get_chunk_section(chunk)
@@ -362,7 +360,7 @@ class DatasetStudyGenerator:
         examples = self._select_examples_to_label(num_samples, synthetic_only)
         if not examples:
             return []
-        loaded_model = self.harness.loaded_models[self.study_model_name]
+        loaded_model = self.harness.loaded_models[self.label_model_name]
         loaded_model.ensure_engine_loaded() # Keep the cold load out of the batch timings.
         system_prompt, instructions, json_schema = make_label_prompt()
         chat_kwargs = self._make_engine_chat_kwargs(json_schema)
````

</details>

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">activation/dataset/dataset_manager.py</span>
    <span class="card-oneliner">Follows the generator rename.</span>
    <span class="card-badge">Diff</span>
  </summary>

The manager's `synthesize_study_examples` keeps its name and calls `generate_example_qa`. Exact delta vs `/source`:

````diff-python
diff --git a/source/activation/dataset/dataset_manager.py b/activation/dataset/dataset_manager.py
index ca14e04..eae8c8e 100644
--- a/source/activation/dataset/dataset_manager.py
+++ b/activation/dataset/dataset_manager.py
@@ -57,7 +57,7 @@ class DatasetManager:
     def synthesize_study_examples(self, dataset_id: str, num_samples: int):
         """Synthesize study examples"""
         study_generator = self._get_or_create_study_generator(dataset_id)
-        return study_generator.generate_study_examples(num_samples)
+        return study_generator.generate_example_qa(num_samples)
 
     def label_study_examples(
         self,
````

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">activation/tests/test_basic_dataset_study.py</span>
    <span class="card-oneliner">`QA_MODEL_NAME`, no chat kwargs, no batch constants.</span>
    <span class="card-badge">Diff</span>
  </summary>

Your two notes. `QA_MODEL_NAME` everywhere (config, residency asserts); the `dataset_study_chat_kwargs` block is dropped rather than set to `None` (same effect, one less line to read); the re-added `STUDY_BATCH_SIZE` / `STUDY_LABEL_BATCH_SIZE` constants were unused and go. Exact delta vs `/source`:

````diff-python
diff --git a/source/activation/tests/test_basic_dataset_study.py b/activation/tests/test_basic_dataset_study.py
index 86120ae..6153b45 100644
--- a/source/activation/tests/test_basic_dataset_study.py
+++ b/activation/tests/test_basic_dataset_study.py
@@ -15,14 +15,11 @@ from activation.harness import (
 pytestmark = pytest.mark.gpu
 
 STUDY_NUM_QUESTIONS = 16
-STUDY_BATCH_SIZE = 8
-STUDY_LABEL_BATCH_SIZE = 8
 if SUPPORTS_FP4:
-    # @AI: Rename to QA_MODEL_NAME.
-    STUDY_MODEL_NAME = "unsloth/Qwen3.8-27B-NVFP4"
+    QA_MODEL_NAME = "unsloth/Qwen3.8-27B-NVFP4"
     LABEL_MODEL_NAME = "nvidia/Qwen3.6-35B-A3B-NVFP4"
 else:
-    STUDY_MODEL_NAME = "Qwen/Qwen3.8-27B-FP8"
+    QA_MODEL_NAME = "Qwen/Qwen3.8-27B-FP8"
     LABEL_MODEL_NAME = "Qwen/Qwen3.6-35B-A3B-FP8"
 
 EMBEDDING_MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
@@ -32,21 +29,12 @@ BRIGHT_DOMAIN = "biology"
 def test_basic_dataset_study():
     harness_config = HarnessRuntimeConfig(
         model_configs={
-            STUDY_MODEL_NAME: ModelConfig(STUDY_MODEL_NAME, STUDY_MODEL_NAME),
+            QA_MODEL_NAME: ModelConfig(QA_MODEL_NAME, QA_MODEL_NAME),
             LABEL_MODEL_NAME: ModelConfig(LABEL_MODEL_NAME, LABEL_MODEL_NAME),
         },
-        dataset_study_model_name=STUDY_MODEL_NAME,
+        dataset_study_qa_model_name=QA_MODEL_NAME,
         dataset_study_label_model_name=LABEL_MODEL_NAME,
-        # @AI: set to None to use the recommended arguments.
-        dataset_study_chat_kwargs={
-            # Qwen3.8 instruct-mode sampling recommendation.
-            "sampling_params": {
-                "temperature": 0.7,
-                "top_p": 0.8,
-                "top_k": 20,
-                "presence_penalty": 1.5,
-            },
-        },
+        # Engine and chat kwargs come from VLLMWrapper's recommendations; batch sizes from the node.
     )
     harness = HarnessRuntime(harness_config)
     dataset = BrightDataset.load(
@@ -125,8 +113,8 @@ def test_basic_dataset_study():
     # Holds because this test never loads the HF model or the embedding layer (bm25 only).
     loads = [name for name, _ in harness.harness_stats.model_loading_times]
     frees = [name for name, _ in harness.harness_stats.model_freeing_times]
-    assert loads == [STUDY_MODEL_NAME, LABEL_MODEL_NAME], loads
-    assert frees == [STUDY_MODEL_NAME, LABEL_MODEL_NAME], frees
+    assert loads == [QA_MODEL_NAME, LABEL_MODEL_NAME], loads
+    assert frees == [QA_MODEL_NAME, LABEL_MODEL_NAME], frees
 
     print("\n=== harness stats ===")
     print(json.dumps(harness.harness_stats.summarize(), indent=2))
\ No newline at end of file
````

</details>
