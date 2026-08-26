# Activation Context harness — minimal exact follow-up

This round incorporates your feedback as a small behavioral delta from the latest user-edited source, with the Qwen embedding terminal-token contract resolved across both proposed checkpoint sizes.

## Feedback resolved

- `adapt_message_format` is unchanged. Its existing structure, assertions, mutation behavior, and variable names remain visible exactly as you wrote them.
- The broad cleanup, defensive rewrites, `Any` annotation, `loaded` rename, forced `float32` conversion, and reduced test matrices are gone.
- `LoadedBaseModel.model` is typed as `PreTrainedModel`; the processor type admits both multimodal processors and text tokenizers.
- Embeddings still detach and move to CPU in their existing BF16 dtype. The norm assertions now create the comparison scalar in the embedding's own dtype.
- All four generative rows and both Qwen embedding rows are active in the default smoke-test lists.

## Qwen embedding readout: EOS, not EOT

The natural final token for direct Qwen3-Embedding input is `<|endoftext|>` (ID 151643), not `<|im_end|>` (ID 151645).

This is easy to misread from metadata:

| Checkpoint | Hugging Face config EOS | Tokenizer/chat EOS | Final token from direct tokenization |
| --- | --- | --- | --- |
| Qwen3-Embedding-0.6B | `<|endoftext|>` (151643) | `<|im_end|>` (151645) | `<|endoftext|>` (151643) |
| Qwen3-Embedding-4B | `<|im_end|>` (151645) | `<|im_end|>` (151645) | `<|endoftext|>` (151643) |

The two checkpoint configs disagree, but both embedding tokenizers append `<|endoftext|>` when special-token handling is enabled. The proposal therefore keeps `EmbeddingReadoutType.EOS_TOKEN` and, for a recognized EOS-readout embedding family, records the actual direct-input terminal token rather than trusting the checkpoint's generic EOS field. Manual input assembly should terminate with `<|endoftext|>`; `<|im_end|>` remains the chat end-of-turn token.

This matches Qwen's official [0.6B Transformers usage](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B#transformers-usage) and [4B Transformers usage](https://huggingface.co/Qwen/Qwen3-Embedding-4B#transformers-usage): tokenize direct text with special-token handling and pool the final active position.

## Why the remaining runtime lines exist

- The dictionary `_hf_models` is keyed by `model_id`, so indexing it with `model_name` is a real bug whenever an interface alias differs from the Hugging Face ID.
- The description assignment is needed only for the second and later `ModelConfig` aliases sharing one loaded base. `_load_model` populates the first config; without copying that discovered metadata, later alias configs remain `None`.
- Processor selection now follows the config shape. Text-only configs use `AutoTokenizer`; multimodal configs use `AutoProcessor`. This fixes the Gemma 3 1B failure because Hugging Face documents that checkpoint as text-only and demonstrates it with `AutoTokenizer` and `AutoModelForCausalLM`: [Gemma 3 documentation](https://huggingface.co/docs/transformers/model_doc/gemma3#notes).
- Your existing `enable_thinking=False` change is preserved as baseline and does not appear in the proposed diff. Configuration can come later.
- `return_tensor` becomes `return_tensors`; the singular spelling caused the tokenizer to return Python lists and produced the reported embedding-layer `TypeError`.

## Exact proposed source delta

The hunks below are relative to the latest `/source` snapshot. They contain every proposed source line; files with no delta are omitted.

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title"><a href="./activation/harness/model_config.py">activation/harness/model_config.py</a></span>
    <span class="card-oneliner">Fixes the dataclass list default without changing the surrounding model vocabulary.</span>
    <span class="card-badge">Changed</span>
  </summary>

`activation/harness/model_config.py`

~~~diff-python
@@ -48,7 +48,7 @@
     """
     d_model: int
     is_multimodal: bool
-    layer_descriptions: list[LayerDescription] = field(default=list)
+    layer_descriptions: list[LayerDescription] = field(default_factory=list)
     dtype: torch.dtype = torch.bfloat16
     eos_token: str | None = None
     eot_token: str | None = None
~~~

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title"><a href="./activation/harness/hf_utils.py">activation/harness/hf_utils.py</a></span>
    <span class="card-oneliner">Uses the terminal token emitted for direct embeddings and fixes right-padded batch indexing.</span>
    <span class="card-badge">Changed</span>
  </summary>

`activation/harness/hf_utils.py`

~~~diff-python
@@ -121,12 +121,17 @@
         )
 
     embedding_readout_type = canonical_embedding_readout_type(model_id)
+    eos_token = canonical_eos_token(hf_config, tokenizer)
+    if embedding_readout_type is EmbeddingReadoutType.EOS_TOKEN:
+        eos_token = tokenizer.convert_ids_to_tokens(
+            tokenizer("", add_special_tokens=True)["input_ids"][-1]
+        )
     return ModelDescription(
         d_model=text_config.hidden_size,
         is_multimodal=is_multimodal,
         dtype=dtype,
         layer_descriptions=layers,
-        eos_token=tokenizer.eos_token,
+        eos_token=eos_token,
         eot_token=canonical_eot_token(tokenizer),
         is_embedding_model=embedding_readout_type is not None,
         embedding_readout_type=embedding_readout_type,
@@ -172,7 +177,9 @@
             # Right-padded: select the last active index (equivalent to sum of 1s up to that point).
             padding_end_indices = attention_mask.sum(dim=1) - 1
             print(f"Padding end indices shape: {padding_end_indices.shape}")
-            batch_indices = torch.arange(padding_end_indices.shape[0], 1)
+            batch_indices = torch.arange(
+                padding_end_indices.shape[0], device=last_hidden_state.device
+            )
             pooled = last_hidden_state[batch_indices, padding_end_indices]
     else:
         # Avg: just average all active positions.
~~~

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title"><a href="./activation/harness/runtime.py">activation/harness/runtime.py</a></span>
    <span class="card-oneliner">Keeps the runtime shape while fixing aliases, processor selection, typing, and tensorization.</span>
    <span class="card-badge">Changed</span>
  </summary>

`activation/harness/runtime.py`

~~~diff-python
@@ -8,6 +8,8 @@
     AutoModelForCausalLM,
     AutoModelForMultimodalLM,
     AutoProcessor,
+    AutoTokenizer,
+    PreTrainedModel,
     PreTrainedTokenizerBase,
     ProcessorMixin,
 )
@@ -30,8 +32,8 @@
     def __init__(
         self,
         model_config: ModelConfig,
-        model: AutoModelForMultimodalLM|AutoModelForCausalLM|AutoModel,
-        processor: ProcessorMixin,
+        model: PreTrainedModel,
+        processor: ProcessorMixin|PreTrainedTokenizerBase,
         tokenizer: PreTrainedTokenizerBase,
     ):
         self.model_config = model_config
@@ -60,13 +62,17 @@
         for model_config in self.harness_runtime_config.model_configs.values():
             if model_config.model_id not in self._hf_models:
                 self._hf_models[model_config.model_id] = self._load_model(model_config)
-            self.loaded_models[model_config.model_name] = self._hf_models[model_config.model_name]
+            loaded_model = self._hf_models[model_config.model_id]
+            model_config.model_description = loaded_model.model_config.model_description
+            self.loaded_models[model_config.model_name] = loaded_model
 
     def _load_model(self, model_config: ModelConfig) -> LoadedBaseModel:
-        processor = AutoProcessor.from_pretrained(model_config.model_id)
-        if isinstance(processor, PreTrainedTokenizerBase):
+        hf_config = AutoConfig.from_pretrained(model_config.model_id)
+        if hf_config.get_text_config() is hf_config:
+            processor = AutoTokenizer.from_pretrained(model_config.model_id)
             tokenizer = processor
         else:
+            processor = AutoProcessor.from_pretrained(model_config.model_id)
             assert isinstance(processor, ProcessorMixin)
             tokenizer = processor.tokenizer
         model_config.model_description = model_description_from_hf(
@@ -134,7 +140,7 @@
         inputs = loaded_model.tokenizer(
             text,
             truncation=True,
-            return_tensor="pt",
+            return_tensors="pt",
         ).to(loaded_model.model.device)
         with torch.inference_mode():
             outputs = loaded_model.model(**inputs)
~~~

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title"><a href="./activation/tests/test_harness/test_basic_harness.py">activation/tests/test_harness/test_basic_harness.py</a></span>
    <span class="card-oneliner">Restores every model row and makes the existing embedding checks exercise the corrected contract.</span>
    <span class="card-badge">Changed</span>
  </summary>

`activation/tests/test_harness/test_basic_harness.py`

~~~diff-python
@@ -8,10 +8,10 @@
 
 # Model ID, Num Layers, D Model
 BASIC_SMALL_MODELS = [
-    # ("Qwen/Qwen3.5-0.8B", 24, 1024),
+    ("Qwen/Qwen3.5-0.8B", 24, 1024),
     ("Qwen/Qwen3-1.7B", 28, 2048),
-    # ("google/gemma-3-1b-it", 32, 1152),
-    # ("google/gemma-4-E2B-it", 35, 1536),
+    ("google/gemma-3-1b-it", 32, 1152),
+    ("google/gemma-4-E2B-it", 35, 1536),
 ]
 
 
@@ -19,7 +19,7 @@
 # Only use qwen for now.
 BASIC_SMALL_EMBEDDING_MODELS = [
     ("Qwen/Qwen3-Embedding-0.6B", 28, 1024, EmbeddingReadoutType.EOS_TOKEN, "<|endoftext|>"),
-    # ("Qwen/Qwen3-Embedding-4B", 36, 2560, EmbeddingReadoutType.EOS_TOKEN, "<|endoftext|>"),
+    ("Qwen/Qwen3-Embedding-4B", 36, 2560, EmbeddingReadoutType.EOS_TOKEN, "<|endoftext|>"),
 ]
 
 def test_basic_model_loading():
@@ -73,17 +73,17 @@
         assert description.d_model == expected_d_model
         assert description.is_embedding_model
         assert description.embedding_readout_type is expected_readout_type
-        # if expected_readout_type == EmbeddingReadoutType.EOS_TOKEN:
-        #     assert description.eos_token == expected_readout_token
-        # elif expected_readout_type == EmbeddingReadoutType.EOT_TOKEN:
-        #     assert description.eot_token == expected_readout_type
+        if expected_readout_type == EmbeddingReadoutType.EOS_TOKEN:
+            assert description.eos_token == expected_readout_token
+        elif expected_readout_type == EmbeddingReadoutType.EOT_TOKEN:
+            assert description.eot_token == expected_readout_token
         # Live tests.
         embedding1 = harness.simple_embed(model_id, text1)
         embedding2 = harness.simple_embed(model_id, text2)
         # Expected shape, normalization, and closeness.
         assert embedding1.shape == (expected_d_model,)
         assert embedding2.shape == (expected_d_model,)
-        assert torch.allclose(embedding1.norm(), torch.tensor(1.0), atol=1e-5)
-        assert torch.allclose(embedding2.norm(), torch.tensor(1.0), atol=1e-5)
+        assert torch.allclose(embedding1.norm(), embedding1.new_tensor(1.0), atol=1e-5)
+        assert torch.allclose(embedding2.norm(), embedding2.new_tensor(1.0), atol=1e-5)
         cosine_distance = 1.0 - torch.dot(embedding1, embedding2)
         assert cosine_distance.item() <= expected_max_cosine_distance_delta
~~~

</details>

## Validation

- All staged Python compiles.
- Pytest collects the two default weight-backed smoke tests. The already-created scratch environment is missing `python-dotenv`, although the current `/source/uv.lock` includes it; collection was repeated with an import-only stub rather than mutating the environment.
- Cached/public metadata checks passed for Qwen3.5-0.8B, Qwen3-1.7B, Qwen3-Embedding-0.6B, and Qwen3-Embedding-4B.
- Both embedding sizes emit direct terminal ID 151643; the 4B config discrepancy above was reproduced.
- A deterministic two-row right-padding fixture passes, and separate `ModelDescription` instances no longer share a layer list.
- A two-alias fixture loads one base, exposes it under both names, and populates both configs' descriptions.
- Qwen3-1.7B's rendered prompt ends with the explicit empty thinking block when `enable_thinking=False`.
- The Gemma config/processor probe could not be repeated in this unauthenticated session because the repositories are gated. The proposed split follows the official text-only Gemma 3 1B loading contract and your report that the multimodal Gemma model already works.
- Full model weights were not downloaded or executed in this pass.

## Durable review rule

Your `adapt_message_format` example is now recorded in [ROOT_CANON.md](../../CANON/ROOT_CANON.md): incremental fixes preserve the user's mental model and isolate required behavior from unrelated cleanup.
