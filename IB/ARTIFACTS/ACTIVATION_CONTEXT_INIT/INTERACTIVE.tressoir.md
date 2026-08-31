# LoadedModel refactor — minimal test repair

Your merged `LoadedModel` design is preserved. The handoff changes only two files: it restores the small `HarnessRuntime` convenience surface expected by the harness tests, fixes model reload and free-state bookkeeping inside `LoadedModel`, and accepts the engine test's existing `lora_name=None` call. Both requested test files pass unchanged.

## Outcome

| Boundary | Result |
| --- | --- |
| `test_basic_engine.py` | vLLM Qwen 3.5 chat passed and returned a response containing `hello`. |
| `test_basic_harness.py::test_basic_model_loading` | Metadata and direct Transformers generation passed; response was `Hello, World!`. |
| `test_basic_harness.py::test_basic_embedding_models` | Qwen 3 Embedding 4B shapes, normalization, and cosine-distance checks passed. |
| Complete GPU run | `3 passed in 350.92s` on one NVIDIA L40S. |
| Test changes | None. The two source test files were run as written. |
| Cloud cleanup | The disposable `ac-loaded-model-tests` cluster was torn down after the pass. |

## Scope kept deliberately narrow

- The single `LoadedModel` architecture, names, control flow, model selection, sampling settings, and harness configuration remain yours.
- No concurrency, locking, LoRA routing, message-schema redesign, or formatting cleanup is introduced.
- `lora_name` is accepted for compatibility with the existing test, but remains unused; this does not claim LoRA routing support.
- The source-shaped handoff contains only the two changed files. Superseded pre-refactor engine/harness copies and copied tests were removed from the handoff.

## Exact source delta

### [activation/harness/loaded_model.py](./activation/harness/loaded_model.py)

Make merged model reload, embedding-copy lifecycle, engine-free state, and the existing call signature behave correctly.

`activation/harness/loaded_model.py`
~~~diff-python
--- a/activation/harness/loaded_model.py
+++ b/activation/harness/loaded_model.py
@@ -100,7 +100,9 @@
         self.engine_to_device(FREE_DEVICE) # Avoid dual load.
         if device != FREE_DEVICE and self.current_model_device == FREE_DEVICE:
             # Reload from disk.
-            self.model = self.model_loader.from_pretrained()
+            self.model = self.model_loader.from_pretrained(
+                self.model_config.model_id, dtype=self.model_config.dtype
+            )
             self.model.eval()
         # Finalize.
         self.model.to(device)
@@ -121,19 +123,23 @@
             # Free.
             self.embedding_layer = None
             gc.collect()
-            if self.current_model_device.startswith("cuda"):
+            if self.current_embedding_layer_device.startswith("cuda"):
                 torch.cuda.empty_cache()
             self.current_embedding_layer_device = device
+            return
         # Bring over from scratch. Need to temporarily use the whole model.
         starting_model_device = self.current_model_device
         if starting_model_device == FREE_DEVICE:
             self.model_to_device(SOURCE_DEVICE) # Use cpu for efficiency.
-        self.embedding_layer = deepcopy(self.model.get_input_embeddings())
-        self.embedding_layer.requires_grad_(False)
-        self.embedding_layer.eval().to(device)
-        self.current_embedding_layer_device = device
-        if starting_model_device == FREE_DEVICE:
-            self.model_to_device(FREE_DEVICE) # Restore starting model device.
+        try:
+            assert self.model is not None
+            self.embedding_layer = deepcopy(self.model.get_input_embeddings())
+            self.embedding_layer.requires_grad_(False)
+            self.embedding_layer.eval().to(device)
+            self.current_embedding_layer_device = device
+        finally:
+            if starting_model_device == FREE_DEVICE:
+                self.model_to_device(FREE_DEVICE) # Restore starting model device.
@@ -160,9 +166,14 @@
             gc.collect()
             if torch.cuda.is_available():
                 torch.cuda.empty_cache()
+            self.current_engine_device = device

-    def simple_engine_chat(self, messages: list[dict]) -> str:
+    def simple_engine_chat(
+        self,
+        messages: list[dict],
+        lora_name: str|None = None,
+    ) -> str:
         """Simple function to test engine chat."""
~~~

### Why these lines are present

- Hugging Face reload needs the repository ID and the configured dtype; an argument-free `from_pretrained()` cannot reconstruct the model.
- Freeing an embedding copy must check the embedding copy's device, then return. Without the return, the free branch immediately falls through and tries to recreate/move the embedding layer to the `disk` sentinel.
- `finally` preserves your intended invariant: a full model temporarily loaded only to copy its embedding module returns to its original free state even if copying or transfer fails.
- Engine teardown now records the free state after dropping vLLM.
- The optional `lora_name` parameter matches the existing test call without changing current routing behavior.

### [activation/harness/runtime.py](./activation/harness/runtime.py)

Restore the two thin harness delegates used by the unchanged basic harness tests.

`activation/harness/runtime.py`
~~~diff-python
--- a/activation/harness/runtime.py
+++ b/activation/harness/runtime.py
@@ -62,3 +62,9 @@
             processor=processor,
             tokenizer=tokenizer,
         )
+
+    def simple_chat(self, model_name: str, user_msg: str) -> str:
+        return self.loaded_models[model_name].simple_chat(user_msg)
+
+    def simple_embed(self, model_name: str, text: str) -> torch.Tensor:
+        return self.loaded_models[model_name].simple_vector_embed(text)
~~~

These methods add no new behavior. They keep `HarnessRuntime` as the name-based convenience entry point while all loading, generation, and embedding work remains in your merged `LoadedModel`.

## Validation details

The preflight used the same staged files without downloading weights:

- both changed Python files parsed;
- all three tests collected from the two requested files;
- a small fake-model fixture verified independent embedding storage, restoration of a previously free full model, and a non-recreating free path.

The release run used the existing baked CUDA environment and unchanged tests:

`pytest(...)`
~~~bash
env VLLM_WORKER_MULTIPROC_METHOD=spawn \
  PYTHONPATH=/root/sky_workdir \
  uv run pytest -s \
  activation/tests/test_basic_engine.py \
  activation/tests/test_basic_harness.py
~~~

Result: `3 passed in 350.92s`. The Transformers `min_frames` / `max_frames` documentation messages and LoRA tower warnings were upstream logs, not failures.

## Apply surface

Copy or adapt exactly these two files:

1. [activation/harness/loaded_model.py](./activation/harness/loaded_model.py)
2. [activation/harness/runtime.py](./activation/harness/runtime.py)
