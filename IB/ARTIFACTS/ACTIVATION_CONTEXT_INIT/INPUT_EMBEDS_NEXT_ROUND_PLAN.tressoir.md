# Secrets, Gemma 3, and side-model embeddings — completed

The approved batch is complete: `.env` reaches trusted remote jobs without entering the source mirror, Gemma 3 passes through the real engine, the primary embedding module can outlive a freed full model, and the side-activation conclusion is now based on same-shaped live controls.

## Outcome at a glance

| Question | Completed answer |
| --- | --- |
| How does `.env` reach the node? | `sky exec --secret-file <project>/.env`; the exact rsync mirror still excludes the file. |
| Did Gemma 3 work? | Yes. String/typed chat, exact target prompt rows, the cloned module, and the real project engine all passed on one L40S. |
| Can the embedding layer be independent? | Yes. The returned frozen module owns separate storage and a previously free full model is restored to free. |
| Is pinning cheap at 8B–32B? | Main BF16 tables measured 1.16–2.62 GiB and 2.37–10.54% of checkpoint parameters. It is plausible on large GPUs, but remains an explicit residency policy. |
| Are same-shaped side activations always safe? | No. A live 1024-wide side/target pair passed vLLM's carrier checks but occupied different learned coordinate spaces and generated different output. |

## Accepted execution

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">Remote secrets and vLLM process mode</span>
    <span class="card-oneliner">Inject secrets outside the mirror and make remote engine startup spawn-safe.</span>
    <span class="card-badge">Complete</span>
  </summary>

[Open the staged Sky wrapper](../SKYPILOT/activation/cloud/sky.py)

~~~diff-python
 ENV_FILE = PROJECT_ROOT / ".env"

 remote_command = shlex.join([
     "env",
     "VLLM_WORKER_MULTIPROC_METHOD=spawn",
     f"PYTHONPATH={REMOTE_WORKDIR}",
     *command,
 ])
+secret_file_args = (
+    ["--secret-file", str(ENV_FILE)] if ENV_FILE.is_file() else []
+)
 return _run_skypilot("exec", *secret_file_args, ...)
~~~

The live probe checked only presence. It also confirmed `/root/sky_workdir/.env` did not exist.

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">Independent primary embedding module</span>
    <span class="card-oneliner">Copy the module, freeze it, and preserve the full model's starting placement.</span>
    <span class="card-badge">Complete</span>
  </summary>

[Open `hf_utils.py`](./activation/harness/hf_utils.py) · [Open `runtime.py`](./activation/harness/runtime.py)

~~~python
def copy_input_embedding_layer(model, device=SOURCE_DEVICE):
    embedding_layer = deepcopy(model.get_input_embeddings())
    embedding_layer.requires_grad_(False)
    return embedding_layer.eval().to(device)

def extract_input_embedding_layer(self, device=SOURCE_DEVICE):
    starting_device = self.current_device
    if starting_device == FREE_DEVICE:
        self.to_device(SOURCE_DEVICE)
    try:
        return copy_input_embedding_layer(self.model, device)
    finally:
        if starting_device == FREE_DEVICE:
            self.to_device(FREE_DEVICE)
~~~

Gemma 3 live validation produced exact vLLM-scaled rows from the copied `Gemma3TextScaledWordEmbedding` after the full model returned to free.

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">Representation verdict</span>
    <span class="card-oneliner">The carrier accepts same-shaped rows; equivalence depends on the target's learned input space.</span>
    <span class="card-badge">Evidence-backed</span>
  </summary>

- Qwen 3 and Qwen 3.5 use their primary input rows directly: exact target rows are the positive contract; side rows need a trained and validated projector.
- Gemma 3 adds scaling in its input embedding module. Copying the module preserves that contract; raw unscaled rows do not.
- Gemma 4 12B/26B/31B use the main input table in the audited configs. Gemma 4 E2B additionally uses a large token-conditioned per-layer embedding path, so main rows alone are not a generally text-equivalent carrier for that variant.

The conclusion assumes the side tensors are already `d_model`-shaped. The observed risk is representation-space and architecture semantics—not a trivial width mismatch.

</details>

## Validation

- Focused local suite: `19 passed in 8.18s`.
- Gemma 3 engine response: `Hello, World!`.
- GPU: NVIDIA L40S; vLLM 0.28.0; Transformers 5.15.1.
- Final SkyPilot state: no clusters, jobs, or services.

## Apply surface

1. Copy [the staged `sky.py`](../SKYPILOT/activation/cloud/sky.py) into `activation/cloud/sky.py`.
2. Review and adapt [the helper](./activation/harness/hf_utils.py) and [the `LoadedBaseModel` lifecycle method](./activation/harness/runtime.py); `runtime.py` also contains the previously reviewed placement fixes.

Read the [cumulative HTML report](./INPUT_EMBEDS_REPORT.tressoir.html) for the full evidence, size table, and family-specific conclusion.

