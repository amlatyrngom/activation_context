"""Deep AC inputs (per-layer residual deltas read out through passage attention) on tiny CPU models: layer resolution, zero mode
unchanged, a deep model silent at init, gradients through checkpointed layers, padded-batch positions and causality, generation
without re-injection, checkpoints and resume, the deltas-off panel, the frozen-reader regime, and the reporter's trace widgets.

  PYTHONPATH=. .venv/bin/python -m pytest activation/tests/test_interventions.py -q -p no:cacheprovider

Reference data under `data/`: a tiny Qwen3 reader, and outputs, keys and a `modules` checkpoint written by the code
before deep inputs existed (tag ac-recon-20260916T132825Z), so zero mode is checked against what it replaced.
"""
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.modules.setdefault("fla", None)          # the kernels are GPU-only; the models fall back to torch
import pytest
import torch

from activation.ac_model import (ActivationContextModelConfig, ActivationContextStudyGenerator, ActivationContextTrainer,
                                 ActivationContextTrainingConfig)
from activation.ac_model.ac_model_training import ActivationContextTrainingProgress
from activation.ac_model.ac_model_utils import ACOutput, PassageAttention, resolve_intervention_layers
from activation.harness import HarnessRuntime, HarnessRuntimeConfig, ModelConfig
from activation.harness.hf_utils import (INTERVENTIONS_KWARG, BoundInterventions, decoder_layers, generate_with_interventions,
                                         install_intervention_hooks, text_decoder)

DATA = Path(__file__).resolve().parent / "data"
TINY_QWEN3 = DATA / "tiny_qwen3"
PASSAGES = [{"text": "alpha beta gamma delta evidence question answer first second third one two three", "doc_id": "d1", "source": "test"},
            {"text": "tool result beginning middle end context reasoning code search solver useful wrong yes no", "doc_id": "d2", "source": "test"},
            {"text": "four five six seven", "doc_id": "d3", "source": "test"}]
torch.set_num_threads(2)


# --------------------------------------------------------------------------------------------------- fixtures
def tiny_hybrid(folder: Path, hidden: int = 32, seed: int = 111) -> Path:
    """A 6-layer Qwen3.5 text model (full attention every third block: layers 2 and 5) with the tiny Qwen3 tokenizer."""
    from transformers import AutoTokenizer, Qwen3_5ForCausalLM, Qwen3_5TextConfig
    if (folder / "model.safetensors").is_file():
        return folder
    tokenizer = AutoTokenizer.from_pretrained(str(TINY_QWEN3))
    config = Qwen3_5TextConfig(vocab_size=len(tokenizer), num_hidden_layers=6, full_attention_interval=3, hidden_size=hidden, intermediate_size=2 * hidden,
                               num_attention_heads=4, num_key_value_heads=2, head_dim=8, linear_num_value_heads=4, linear_num_key_heads=2,
                               linear_key_head_dim=8, linear_value_head_dim=8, linear_conv_kernel_dim=4, max_position_embeddings=512,
                               pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id, tie_word_embeddings=False)
    config._attn_implementation = "sdpa"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = Qwen3_5ForCausalLM(config)
    folder.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(folder)
    tokenizer.save_pretrained(folder)
    return folder


def build(tmp_path: Path, model_dir: Path, *, frequency: int = 0, add_last: bool = True, seed: int = 1234,
          ac_checkpoint: str | None = None, reader_checkpoint: str | None = None, layers: list[int] | None = None, side_dir: Path | None = None):
    """The shared/reader base with a rank-4 adapter and a tiny AC model on top, in the same construction order every time
    (seeded: the AC modules draw first, the adapters draw at injection). `side_dir`: a separate (narrower) encoder base."""
    os.environ["ACTIVATION_SYNC_ROOT"] = str(tmp_path)
    torch.manual_seed(seed)
    models = {"reader": ModelConfig("reader", str(model_dir))}
    if side_dir is not None:
        models["side"] = ModelConfig("side", str(side_dir))
    h = HarnessRuntime(HarnessRuntimeConfig(model_configs=models))
    h.module_manager.register_lora("reader_lora", "reader", rank=4, dropout=0.0, checkpoint_path=reader_checkpoint)
    ac = h.module_manager.register_ac_model(ActivationContextModelConfig(
        "ac_private", "side" if side_dir is not None else "reader", "side_lora", "reader", "reader_lora", mixer_heads=4, mixer_window=4, side_lora_rank=4, max_view_rows=8,
        encode_batch_max_tokens=2048, checkpoint_path=ac_checkpoint, intervention_frequency=frequency, add_last_layer_intervention=add_last,
        intervention_layers=layers))
    h.module_manager.ensure_lora("reader_lora")
    ac.prepare()
    for loaded in h.loaded_models.values():
        loaded.model.float()
    return h, ac


def config(**overrides) -> ActivationContextTrainingConfig:
    base = dict(loss_kind="sft", num_epochs=1, examples_per_update=2, completion_samples=1, completion_max_new_tokens=4, teacher_cache_top_k=5,
                gradient_checkpointing=True, gradient_checkpointing_min_tokens=0, logits_chunk_tokens=3, learning_rate_ac=1e-2, learning_rate_target_lora=1e-2,
                warmup_updates=0, checkpoint_every_epoch=False, no_context_penalty_weight=1.0, no_context_penalty_fraction=1.0, reporting_interval=1.0)
    base.update(overrides)
    return ActivationContextTrainingConfig(**base)


def items_for(h, ac):
    return ActivationContextStudyGenerator(h, ac.name).generate_reconstruction_samples(PASSAGES, ratio=0.25)


def trainer_and_examples(h, ac, **overrides):
    trainer = ActivationContextTrainer(h, config(**overrides))
    items = items_for(h, ac)
    examples = [trainer.build_example(ac, item) for item in items]
    device = ac.prepare()
    h.module_manager.ensure_lora(ac.config.target_model_lora_name)
    if trainer.config.penalty_active:
        trainer._store_teacher_targets(ac, [example.penalty for example in examples], device, lora=None)
    ac.set_mode("training")                                                        # gradients through the encoder (rollout mode encodes under no_grad)
    return trainer, items, examples, device


def shared_state(ac) -> dict[str, torch.Tensor]:
    """Everything but the intervention modules: the AC modules' shared parameters, the side and reader adapters (cloned)."""
    modules = {name: tensor.clone() for name, tensor in ac.modules.state_dict().items() if not name.startswith(("delta_heads.", "passage_attention."))}
    side = {f"side:{index}": parameter.detach().clone() for index, parameter in enumerate(ac.harness.module_manager.lora_parameters(ac.config.base_side_model_lora_name))}
    reader = {f"reader:{index}": parameter.detach().clone() for index, parameter in enumerate(ac.harness.module_manager.lora_parameters(ac.config.target_model_lora_name))}
    return {**modules, **side, **reader}


def intervention_state(ac) -> dict[str, torch.Tensor]:
    return {name: tensor.clone() for name, tensor in ac.modules.state_dict().items() if name.startswith(("delta_heads.", "passage_attention."))}


def head_parameters(ac):
    return [parameter for name, parameter in ac.modules.named_parameters() if name.startswith("delta_heads.")]


def attention_parameters(ac):
    return [parameter for name, parameter in ac.modules.named_parameters() if name.startswith("passage_attention.")]


def open_heads(ac, std: float = 0.05) -> None:
    """Audible deltas: the zero-initialized down projections moved off zero."""
    with torch.no_grad():
        for head in ac.modules.delta_heads.values():
            head.down.weight.normal_(std=std)


def open_attention(ac, std: float = 0.05) -> None:
    """Moves the zero-initialized out projections off zero so the attention path carries gradient into q/k/v."""
    with torch.no_grad():
        for block in ac.modules.passage_attention.values():
            block.out.weight.normal_(std=std)


def side_lora_parameters(ac, layer: int | None = None) -> list[torch.nn.Parameter]:
    """The side adapter's trainable parameters (the manager grants trainability on request), optionally those of one side layer."""
    adapter = ac.config.base_side_model_lora_name
    granted = {id(parameter) for parameter in ac.harness.module_manager.lora_parameters(adapter)}
    return [parameter for name, parameter in ac.side.model.named_parameters()
            if id(parameter) in granted and (layer is None or f".layers.{layer}." in name)]


def fixed_request(ac):
    from activation.ac_model.ac_model_utils import EncodeRequest
    return EncodeRequest([{"role": "user", "content": PASSAGES[0]["text"]}], 0.25, False, None)


@pytest.fixture(scope="session")
def hybrid_dir(tmp_path_factory):
    return tiny_hybrid(tmp_path_factory.mktemp("models") / "tiny_hybrid")


# --------------------------------------------------------------------------------------------------- 1. resolution
def test_resolve_intervention_layers():
    full = [3, 7, 11, 15, 19, 23, 27, 31]                                          # Qwen3.5-4B: 32 layers, full attention every fourth block
    assert resolve_intervention_layers(32, 4, True, full) == [7, 11, 19, 23, 31]   # 6->7, 12->11, 19, 25 ties 23/27 -> 23, plus 31
    assert resolve_intervention_layers(32, 4, False, full) == [7, 11, 19, 23]
    assert resolve_intervention_layers(32, 4, True, None) == [6, 12, 19, 25, 31]   # no full-attention description: no snapping
    assert resolve_intervention_layers(6, 1, True, [2, 5]) == [2, 5]                # interior 3 snaps to 2 (distance 1)
    assert resolve_intervention_layers(6, 5, True, [2, 5]) == [2, 5]                # every interior point snaps onto the two blocks
    assert resolve_intervention_layers(2, 4, True, [0, 1]) == [1]                   # 0 excluded, duplicates collapse
    assert resolve_intervention_layers(32, 0, True, full) == []                     # zero disables the last-layer flag too
    assert resolve_intervention_layers(32, -1, True, full) == full                  # -1: every full-attention layer (the last-layer flag is moot)
    assert resolve_intervention_layers(6, -1, False, [0, 2, 5]) == [2, 5]           # 0 excluded
    with pytest.raises(ValueError):
        resolve_intervention_layers(32, -2, True, full)
    with pytest.raises(ValueError):
        resolve_intervention_layers(32, -1, True, [])
    assert ActivationContextModelConfig("a", "s", "sl", "t", "tl", intervention_frequency=-1).intervention_frequency == -1
    with pytest.raises(ValueError):
        ActivationContextModelConfig("a", "s", "sl", "t", "tl", intervention_frequency=-2)
    with pytest.raises(ValueError, match="frequency 0"):
        ActivationContextModelConfig("a", "s", "sl", "t", "tl", intervention_frequency=1, intervention_layers=[3])
    assert ActivationContextModelConfig("a", "s", "sl", "t", "tl", intervention_layers=[5, 3]).intervention_layers == [3, 5]
    assert PassageAttention(2560, 640).heads == 4 and PassageAttention(32, 8).heads == 1


def test_text_decoder_locator(hybrid_dir):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(str(hybrid_dir))
    assert text_decoder(model) is model.model and len(decoder_layers(model)) == 6
    try:
        from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration
        text = json.loads((hybrid_dir / "config.json").read_text())
        vision = {"depth": 1, "hidden_size": 32, "intermediate_size": 64, "num_heads": 4, "patch_size": 4, "temporal_patch_size": 1,
                  "spatial_merge_size": 1, "out_hidden_size": 32, "in_channels": 3, "num_position_embeddings": 16, "deepstack_visual_indexes": []}
        multimodal = Qwen3_5ForConditionalGeneration(Qwen3_5Config(text_config=text, vision_config=vision))
    except Exception as error:                                                     # noqa: BLE001 - the tiny vision tower is best effort
        pytest.skip(f"tiny multimodal Qwen3.5 not constructible here: {error!r}")
    layers = decoder_layers(multimodal)
    assert text_decoder(multimodal) is multimodal.model.language_model and len(layers) == 6
    assert all(type(layer).__name__ != "Qwen3_5VisionBlock" for layer in layers)


# --------------------------------------------------------------------------------------------------- 2. zero mode
def test_zero_mode_is_the_previous_model(tmp_path):
    h, ac = build(tmp_path, TINY_QWEN3)
    assert not ac.is_deep and ac.modules.delta_heads is None and ac.modules.passage_attention is None and ac.intervention_layers == []
    assert list(ac.modules.state_dict().keys()) == json.loads((DATA / "legacy_keys.json").read_text())
    legacy = torch.load(DATA / "legacy_ac_modules.pt", map_location="cpu", weights_only=False)
    ac.modules.load_state_dict(legacy["modules"], strict=True)                    # the pre-change checkpoint loads strictly
    ac.modules.target_head.skip = False                                           # the reference fixture predates the identity-skip row head (same keys, plain FFN)
    assert "interventions" not in legacy
    reference = torch.load(DATA / "legacy_outputs.pt", map_location="cpu", weights_only=False)
    ac.modules.load_state_dict(reference["modules"], strict=True)
    rows = ac.encode_batch([fixed_request(ac)])[0]
    assert torch.equal(rows, reference["rows"]), "zero-mode rows differ from the previous code's"
    ac.modules.target_head.skip = True                                            # back to the product's head (equal widths: identity skip)
    model = ac.target.model
    assert getattr(model, "_activation_intervention_hooks", None) is None, "hooks are installed only when a payload is passed"
    ids = torch.tensor(reference["ids"])
    hidden = ac.target.decoder_forward(model.get_input_embeddings()(ids)[None], None, lora_name=None)[0]
    assert torch.equal(hidden, reference["hidden"])
    assert isinstance(ac.encode_batch([fixed_request(ac)], with_layers=True)[0], ACOutput)
    assert ac.encode_batch([fixed_request(ac)], with_layers=True)[0].layer_inputs == {}
    assert ac.intervention_summary() == {}
    # An input-only checkpoint migrates into a deep model (shared modules loaded, the intervention modules at their seeded init); a deep checkpoint never loads as input-only.
    folder = ac.save(str(tmp_path / "zero"))
    deep_h, deep = build(tmp_path / "deep", TINY_QWEN3, frequency=1, seed=4321)
    fresh = intervention_state(deep)
    deep.load(folder)
    assert deep.is_deep and deep.intervention_layers == [1]
    assert torch.equal(deep.modules.target_head.ffn.up.weight, ac.modules.target_head.ffn.up.weight)
    assert all(torch.equal(tensor, deep.modules.state_dict()[name]) for name, tensor in fresh.items()), "the intervention modules keep their seeded init"


def test_pairing_shared_draws_are_identical(tmp_path):
    h0, zero = build(tmp_path / "zero", TINY_QWEN3, seed=77)
    after_zero = torch.rand(3)
    h1, deep = build(tmp_path / "deep", TINY_QWEN3, frequency=1, seed=77)
    after_deep = torch.rand(3)
    assert deep.intervention_layers == [1] and deep.is_deep
    shared = {name: tensor for name, tensor in deep.modules.state_dict().items() if not name.startswith(("delta_heads.", "passage_attention."))}
    assert shared.keys() == zero.modules.state_dict().keys()
    assert all(torch.equal(tensor, zero.modules.state_dict()[name]) for name, tensor in shared.items())
    reader0 = h0.module_manager.lora_parameters("reader_lora"); reader1 = h1.module_manager.lora_parameters("reader_lora")
    assert all(torch.equal(a, b) for a, b in zip(reader0, reader1)), "the adapters injected after the heads draw the same values"
    assert torch.equal(after_zero, after_deep), "the intervention modules' draws come from forked streams"
    heads_a = intervention_state(deep)
    _, again = build(tmp_path / "again", TINY_QWEN3, frequency=1, seed=77)
    assert all(torch.equal(tensor, again.modules.state_dict()[name]) for name, tensor in heads_a.items()), "the init is a function of the seed"


# --------------------------------------------------------------------------------------------------- 3. silent at init
def test_deep_model_is_silent_at_init_and_learns(tmp_path, hybrid_dir):
    h, ac = build(tmp_path, hybrid_dir, frequency=1, seed=31)
    h0, zero = build(tmp_path / "zero", hybrid_dir, seed=31)
    assert ac.intervention_layers == [2, 5] and set(ac.modules.delta_heads) == set(ac.modules.passage_attention) == {"2", "5"}
    for head in ac.modules.delta_heads.values():
        assert torch.equal(head.down.weight, torch.zeros_like(head.down.weight)) and torch.equal(head.down.bias, torch.zeros_like(head.down.bias))
    for block in ac.modules.passage_attention.values():
        assert torch.equal(block.out.weight, torch.zeros_like(block.out.weight)) and torch.equal(block.out.bias, torch.zeros_like(block.out.bias))
    summary = ac.intervention_summary()
    assert summary["layers"] == [2, 5] and summary["target"] == "residual" and summary["source"] == "passage_attention" and summary["attention_depth"] == "final"
    assert summary["head_parameters"] == sum(p.numel() for p in head_parameters(ac)) and summary["attention_parameters"] == sum(p.numel() for p in attention_parameters(ac))
    trainer, items, examples, device = trainer_and_examples(h, ac)
    with pytest.raises(RuntimeError, match="with_layers=True"):
        ac.encode_batch([fixed_request(ac)])                                      # a deep model refuses the rows-only API
    output = ac.encode_batch([fixed_request(ac)], with_layers=True)[0]
    assert set(output.layer_inputs) == {2, 5} and output.input_embeds.shape[1] == ac.d_target
    assert all(rows.shape == output.input_embeds.shape and torch.equal(rows, torch.zeros_like(rows)) for rows in output.layer_inputs.values()), "exactly zero deltas at init"
    assert torch.equal(output.input_embeds, zero.encode_batch([fixed_request(zero)])[0]), "the rows are the input-only model's (shared draws)"
    # The reader forward with the payload is bit-identical to the forward without it.
    embedding = ac.target.model.get_input_embeddings()
    example = examples[0]
    outputs = ac.encode_batch(example.part_requests, with_layers=True)
    with torch.no_grad():
        student = embedding(torch.tensor(example.student_ids))
        with_payload, payload = trainer._assemble_student(student, example.spans, outputs, student.dtype)
        without, none = trainer._assemble_student(student, example.spans, outputs, student.dtype, deltas_off=True)
        assert payload is not None and none is None and torch.equal(with_payload, without) and set(payload["layer_inputs"]) == {2, 5}
        hidden_on = ac.target.decoder_forward(with_payload[None], None, lora_name="reader_lora", interventions=[payload])
        hidden_off = ac.target.decoder_forward(without[None], None, lora_name="reader_lora")
    assert torch.equal(hidden_on, hidden_off), "a zero delta added in place leaves the reader's states bit-identical"
    # The head still learns from update 0: the down projection's gradient is non-zero (the up projection's is zero until down leaves zero).
    loss = trainer._example_loss(ac, example, device, with_penalty=False)[0]
    live = [(name, parameter) for name, parameter in ac.modules.named_parameters() if name.startswith("delta_heads.")]
    grads = dict(zip([name for name, _ in live], torch.autograd.grad(loss, [parameter for _, parameter in live], allow_unused=True)))
    for name, g in grads.items():
        if ".down." in name:
            assert g is not None and float(g.norm()) > 0, f"{name}: the head learns through its down projection at init"
        else:
            assert g is None or float(g.norm()) == 0, f"{name}: no gradient reaches up through a zero down projection"
    # At init the deltas-off panel is the panel itself (the zero delta is added in place).
    h.module_manager.ensure_lora("reader_lora")
    assert trainer._evaluate(ac, items, device, None, "deltas_off", deltas_off=True)["gold_nll"] == trainer._evaluate(ac, items, device, None, "reporting")["gold_nll"]
    # A training epoch: two optimizer groups (encoder, reader), the down projections move, the delta RMS is traced relative to the row RMS.
    trainer = ActivationContextTrainer(h, config(deltas_off_panel=True))
    before = intervention_state(ac)
    stats = trainer.train(ac.name, items, reporting_data=items)
    assert len(trainer.optimizers[ac.name].param_groups) == 2
    after = ac.modules.state_dict()
    assert all(not torch.equal(tensor, after[name]) for name, tensor in before.items() if name.startswith("delta_heads.") and ".down.weight" in name), "the heads train"
    first = stats.intervention_trace[0]
    assert first["update"] == 0 and set(first["delta_rms"]) == set(first["delta_rms_abs"]) == set(first["row_rms"]) == set(first["head_grad_norm"]) == {2, 5}
    assert all(value == 0.0 for value in first["delta_rms_abs"].values()) and all(value == 0.0 for value in first["delta_rms"].values()), "zero at init"
    assert all(first["row_rms"][layer] > 0 for layer in (2, 5)) and all(first["attention_out_norm"][layer] == 0.0 for layer in (2, 5))
    assert stats.deltas_off is not None and stats.deltas_off["items"] == len(items) and stats.payload_bytes > 0
    # Input-only models produce no deltas-off panel and no trace even when asked.
    stats0 = ActivationContextTrainer(h0, config(deltas_off_panel=True)).train(zero.name, items_for(h0, zero), reporting_data=items_for(h0, zero))
    assert stats0.deltas_off is None and stats0.intervention_trace == [] and stats0.payload_bytes == 0


# --------------------------------------------------------------------------------------------------- 4. gradients
def test_gradients_direct_vs_checkpointed(tmp_path, hybrid_dir):
    h, ac = build(tmp_path, hybrid_dir, frequency=1)
    assert ac.intervention_layers == [2, 5], "the interior point snaps to a full-attention block"
    trainer, items, examples, device = trainer_and_examples(h, ac)
    open_heads(ac)                                                                 # audible deltas: every head gets a real gradient
    open_attention(ac)                                                             # a zero out projection would leave q/k/v without gradient
    params = head_parameters(ac) + attention_parameters(ac) + [ac.modules.target_head.scale] + list(h.module_manager.lora_parameters("reader_lora")) + side_lora_parameters(ac)
    grads = {}
    for checkpointed in (False, True):
        trainer.config.gradient_checkpointing = checkpointed
        ac.config.side_gradient_checkpointing = checkpointed
        losses = [trainer._example_loss(ac, example, device, with_penalty=False)[0] for example in examples[:2]]
        grads[checkpointed] = torch.autograd.grad(sum(losses), params, allow_unused=True)
    for direct, recomputed in zip(grads[False], grads[True]):
        assert direct is not None and recomputed is not None
        torch.testing.assert_close(direct, recomputed, atol=1e-6, rtol=1e-5)
    assert all(float(g.norm()) > 0 for g in grads[True][:len(head_parameters(ac))]), "every delta head receives gradient"
    names = [name for name, _ in ac.modules.named_parameters() if name.startswith("passage_attention.")]
    block_grads = dict(zip(names, grads[True][len(head_parameters(ac)):len(head_parameters(ac)) + len(names)]))
    for name, g in block_grads.items():
        if name.endswith((".query.weight", ".key.weight", ".value.weight", ".out.weight")):
            assert float(g.norm()) > 0, f"{name} receives gradient through the attention path"
    assert any(float(g.norm()) > 0 for g in grads[True][-len(side_lora_parameters(ac)):]), "the side adapter receives gradient (lora_A's is zero at PEFT's zero init of lora_B)"
    # Two forwards with different payloads pending before one backward keep their own tensors.
    trainer.config.gradient_checkpointing = True
    ac.config.side_gradient_checkpointing = True
    loss_a = trainer._example_loss(ac, examples[0], device, with_penalty=False)[0]
    loss_b = trainer._example_loss(ac, examples[1], device, with_penalty=False)[0]      # a different example: its own payload tensors
    joint = torch.autograd.grad(loss_a + loss_b, params, allow_unused=True)
    assert all(torch.isfinite(g).all() for g in joint)
    single_a = torch.autograd.grad(trainer._example_loss(ac, examples[0], device, with_penalty=False)[0], params, allow_unused=True)
    single_b = torch.autograd.grad(trainer._example_loss(ac, examples[1], device, with_penalty=False)[0], params, allow_unused=True)
    for j, a, b in zip(joint, single_a, single_b):
        torch.testing.assert_close(j, a + b, atol=1e-6, rtol=1e-5)


# --------------------------------------------------------------------------------------------------- 5. padded batch
def test_padded_batch_positions_and_causality(tmp_path):
    torch.manual_seed(0)
    h, ac = build(tmp_path, TINY_QWEN3, frequency=1)
    assert ac.intervention_layers == [1]
    trainer, items, examples, device = trainer_and_examples(h, ac, micro_batch_examples=3)
    open_heads(ac)
    assert len({len(e.student_ids) for e in examples}) > 1, "the fixture must pad"
    single = [trainer._example_loss(ac, example, device, with_penalty=False) for example in examples]
    batched = trainer._batch_loss(ac, examples, device, with_penalty=False)
    for (loss, _, num, logp), result in zip(single, batched):
        assert result["num"] == num
        torch.testing.assert_close(result["objective"], loss, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(result["target_logp"], logp, atol=1e-4, rtol=1e-4)
    # Deltas on one row change that row only, and nothing before its first intervention position.
    target = ac.target
    embedding = target.model.get_input_embeddings()
    embeds, payloads = [], []
    for example in examples[:2]:
        outputs = ac.encode_batch(example.part_requests, with_layers=True)
        with torch.no_grad():
            student = embedding(torch.tensor(example.student_ids))
        student, payload = trainer._assemble_student(student, example.spans, outputs, student.dtype)
        embeds.append(student); payloads.append(payload)
    with torch.no_grad():
        plain = trainer._padded_forward(target, embeds, "reader_lora")
        one = trainer._padded_forward(target, embeds, "reader_lora", [None, payloads[1]])
        both = trainer._padded_forward(target, embeds, "reader_lora", payloads)
    assert torch.equal(plain[0], one[0]), "row 0 is untouched by row 1's payload"
    first = min(payloads[1]["positions"])
    assert torch.equal(plain[1, :first], one[1, :first]) and not torch.equal(plain[1, first:], one[1, first:])
    assert torch.equal(one[1], both[1]) and not torch.equal(plain[0], both[0])
    with pytest.raises(ValueError, match="payloads"):
        target.decoder_forward(torch.stack([embeds[0], embeds[0]]), None, lora_name="reader_lora", interventions=[payloads[0]])
    # deltas_off keeps the rows and withholds the payload.
    off = trainer._batch_loss(ac, examples[:1], device, with_penalty=False, deltas_off=True)[0]
    assert not torch.isclose(off["objective"], batched[0]["objective"])
    with torch.no_grad():
        student = embedding(torch.tensor(examples[0].student_ids))
    kept, payload = trainer._assemble_student(student, examples[0].spans, ac.encode_batch(examples[0].part_requests, with_layers=True), student.dtype, deltas_off=True)
    assert payload is None and not torch.equal(kept, student)


# --------------------------------------------------------------------------------------------------- 6. generation
def test_generation_matches_full_recompute(tmp_path, hybrid_dir):
    h, ac = build(tmp_path, hybrid_dir, frequency=1)
    trainer, items, examples, device = trainer_and_examples(h, ac)
    open_heads(ac)
    example = examples[0]
    target = ac.target
    manager = h.module_manager
    embedding = target.model.get_input_embeddings()
    prefix_length = example.student_positions[0] + 1
    with torch.no_grad():
        embeds = embedding(torch.tensor(example.student_ids[:prefix_length]))
    present = [(span, request) for span, request in zip(example.spans, example.part_requests) if span[1] <= prefix_length]
    outputs = ac.encode_batch([request for _, request in present], with_layers=True)
    with torch.no_grad():
        embeds, payload = trainer._assemble_student(embeds, [span for span, _ in present], outputs, embeds.dtype)
    peft_model = manager.ensure_lora("reader_lora")
    seen = []
    def spy(module, args, kwargs):                                                   # runs before the product hook: sees the bound payload
        bound = kwargs.get(INTERVENTIONS_KWARG)
        if isinstance(bound, BoundInterventions):
            applied = sum(int(((torch.as_tensor(p["positions"]) - bound.offset) >= 0).sum() - ((torch.as_tensor(p["positions"]) - bound.offset) >= args[0].shape[1]).sum())
                          for p in bound.payloads if p and 2 in p["layer_inputs"])
            seen.append((bound.offset, args[0].shape[1], applied, [list(p["positions"]) for p in bound.payloads if p]))
    install_intervention_hooks(peft_model.get_base_model())
    handle = decoder_layers(peft_model.get_base_model())[2].register_forward_pre_hook(spy, with_kwargs=True, prepend=True)
    steps = 5
    generation = dict(max_new_tokens=steps, do_sample=False, use_cache=True, pad_token_id=target.tokenizer.pad_token_id, min_new_tokens=steps)
    with torch.no_grad(), manager.lora_context("reader", "reader_lora"):
        cached = generate_with_interventions(peft_model, inputs_embeds=embeds[None], interventions=[payload], **generation)[0]
    handle.remove()
    assert len(cached) == steps
    assert seen[0][:3] == (0, prefix_length, len(payload["positions"])), "prefill: every delta row applied once"
    assert all(offset >= prefix_length and length == 1 and applied == 0 for offset, length, applied, _ in seen[1:]), "decode steps: nothing re-injected"
    # Oracle: a full-prefix recompute (no cache) at every step, the same payload on the prompt rows.
    head = target.model.get_output_embeddings()
    generated = []
    with torch.no_grad():
        for _ in range(steps):
            full = torch.cat([embeds, embedding(torch.tensor(generated, dtype=torch.long))], dim=0) if generated else embeds
            hidden = target.decoder_forward(full[None], None, lora_name="reader_lora", interventions=[payload])[0, -1]
            logits = head(hidden)
            if len(generated) < steps:                                             # min_new_tokens: generate masks the eos before the budget
                logits[target.tokenizer.eos_token_id] = float("-inf")
            generated.append(int(logits.argmax()))
    assert generated == cached.tolist()
    # The payload is heard (the last prompt position's logits move), and generate() itself rejects the raw kwarg.
    with torch.no_grad():
        with_deltas = head(target.decoder_forward(embeds[None], None, lora_name="reader_lora", interventions=[payload])[0, -1])
        without = head(target.decoder_forward(embeds[None], None, lora_name="reader_lora")[0, -1])
    assert not torch.allclose(with_deltas, without)
    with pytest.raises(ValueError, match="not used by the model"):
        with manager.lora_context("reader", "reader_lora"):
            peft_model.generate(inputs_embeds=embeds[None], interventions=[payload], **generation)
    with pytest.raises(ValueError, match="greedy"):
        generate_with_interventions(peft_model, inputs_embeds=embeds[None], interventions=[payload], num_beams=2, max_new_tokens=2)
    # The trainer's fidelity samples take the same path: one payload bound at prefill over the present spans' positions, none at decode.
    seen.clear()
    handle = decoder_layers(peft_model.get_base_model())[2].register_forward_pre_hook(spy, with_kwargs=True, prepend=True)
    try:
        trainer.config.completion_max_new_tokens = steps
        samples = trainer._sample_completions(ac, [items[0]], ActivationContextTrainingProgress(0, 1, 0, 0, 1, 0, 0.0, 0.0))
    finally:
        handle.remove()
    expected = [position for (start, end), _ in present for position in range(start, end)]
    assert samples and samples[0]["item_id"] == items[0].item_id and isinstance(samples[0]["student"], str)
    assert seen and seen[0][:3] == (0, prefix_length, len(expected)) and seen[0][3] == [expected], "_sample_completions: the payload is bound at prefill"
    assert 1 <= len(seen) <= steps and all(applied == 0 and payloads == [expected] for _, _, applied, payloads in seen[1:]), "decode steps carry the bound payload but add nothing"
    assert trainer._payload_bytes == 0, "sampling runs under no_grad: not counted as training payload"


# --------------------------------------------------------------------------------------------------- 7. checkpoints, resume
def test_checkpoint_roundtrip_and_resume(tmp_path, hybrid_dir):
    h, ac = build(tmp_path, hybrid_dir, frequency=1)
    items = items_for(h, ac)
    trainer = ActivationContextTrainer(h, config(checkpoint_every_epoch=True, deltas_off_panel=True))
    stats = trainer.train(ac.name, items, reporting_data=items)
    assert stats.intervention_trace and stats.intervention_trace[0]["update"] == 0
    assert set(stats.intervention_trace[0]["head_grad_norm"]) == {2, 5} and set(stats.intervention_trace[0]["row_rms"]) == {2, 5}
    assert stats.intervention_trace[0]["attention_out_norm"][2] == 0.0 and stats.intervention_trace[0]["attention_grad_norm"][2] >= 0.0
    assert stats.payload_bytes > 0 and stats.deltas_off is not None and stats.deltas_off["items"] == len(items)
    modules_before = intervention_state(ac)
    record = json.loads((tmp_path / "AC_MODELS" / ac.name / "completed_epoch.json").read_text())
    saved = tmp_path / "AC_MODELS" / ac.name / "epoch_001" / "optimizer.pt"
    moments = torch.load(saved, map_location="cpu", weights_only=False)
    assert len(moments["optimizer"]["state"]) >= len(head_parameters(ac)), "optimizer.pt carries moments for every head parameter"
    checkpoint = torch.load(tmp_path / record["ac_checkpoint"] / "ac_modules.pt", map_location="cpu", weights_only=False)
    assert checkpoint["interventions"]["layers"] == [2, 5] and checkpoint["interventions"]["attention_parameters"] > 0
    assert (checkpoint["interventions"]["source"], checkpoint["interventions"]["attention_depth"], checkpoint["interventions"]["target"], checkpoint["interventions"]["head_style"]) == ("passage_attention", "final", "residual", "lora")
    assert checkpoint["config"]["row_head_skip"] is True and any(name.startswith("passage_attention.") for name in checkpoint["modules"])
    # A new process: the epoch pair loads into a fresh deep model; the modules come back; the optimizer resumes.
    h2, ac2 = build(tmp_path / "second", hybrid_dir, frequency=1, seed=999,
                    ac_checkpoint=str(tmp_path / record["ac_checkpoint"]), reader_checkpoint=str(tmp_path / record["target_lora_checkpoint"]))
    for name, tensor in modules_before.items():
        assert torch.equal(tensor, ac2.modules.state_dict()[name]), name
    trainer2 = ActivationContextTrainer(h2, config(checkpoint_every_epoch=True, resume_optimizer=True, learning_rate_ac=3e-5, lr_schedule="constant"))
    stats2 = trainer2.train(ac2.name, items_for(h2, ac2))
    assert stats2.resumed_from == str(saved) and stats2.start_epoch == 2 and stats2.learning_rates[-1][0] == pytest.approx(3e-5)
    assert stats2.global_steps[0] == trainer.updates_done[ac.name] + 1
    assert all(id(parameter) in {id(p) for group in trainer2.optimizers[ac2.name].param_groups for p in group["params"]} for parameter in head_parameters(ac2))
    # A deep checkpoint never loads as input-only, nor into other layers.
    h3, zero = build(tmp_path / "zero", hybrid_dir)
    with pytest.raises(ValueError, match="intervention layers"):
        zero.load(str(tmp_path / record["ac_checkpoint"]))
    h4, other = build(tmp_path / "other", hybrid_dir, layers=[2])
    with pytest.raises(ValueError, match="intervention layers"):
        other.load(str(tmp_path / record["ac_checkpoint"]))
    # A deep checkpoint from the full-featured branch: its inert calibration state is dropped; another source or target is refused.
    legacy_folder = tmp_path / "legacy"
    legacy_folder.mkdir()
    for name in ("side_lora",):
        os.symlink(tmp_path / record["ac_checkpoint"] / name, legacy_folder / name)
    legacy = dict(checkpoint)
    legacy["modules"] = {**checkpoint["modules"], "intervention_calibration": torch.zeros(2), "interventions_calibrated": torch.tensor(True),
                         "delta_heads.2.relative": torch.tensor(0.05), "delta_heads.2.rms": torch.tensor(1.0), "delta_heads.5.relative": torch.tensor(0.05), "delta_heads.5.rms": torch.tensor(1.0)}
    torch.save(legacy, legacy_folder / "ac_modules.pt")
    h5, again = build(tmp_path / "again", hybrid_dir, frequency=1, seed=5)
    again.load(str(legacy_folder))
    assert all(torch.equal(tensor, again.modules.state_dict()[name]) for name, tensor in modules_before.items())
    legacy["interventions"] = {**checkpoint["interventions"], "source": "summary", "attention_depth": None}
    torch.save(legacy, legacy_folder / "ac_modules.pt")
    with pytest.raises(ValueError, match="not the model's design"):
        again.load(str(legacy_folder))


# --------------------------------------------------------------------------------------------------- 8. the read-out
def test_passage_attention_reads_the_final_side_states(tmp_path, hybrid_dir):
    h, ac = build(tmp_path, hybrid_dir, frequency=1)
    trainer, items, examples, device = trainer_and_examples(h, ac)
    open_heads(ac)
    # Closed block (zero out projection): the head reads the row's own final state.
    finals = []
    handle = text_decoder(ac.side.model).norm.register_forward_hook(lambda module, args, output: finals.append(output.detach().clone()))
    try:
        out = ac.encode_batch([fixed_request(ac)], with_layers=True)[0]
    finally:
        handle.remove()
    final = finals[-1][0].float()
    num_rows, length = out.input_embeds.shape[0], final.shape[0]
    with torch.no_grad():
        for layer in (2, 5):
            torch.testing.assert_close(out.layer_inputs[layer], ac.modules.delta_heads[str(layer)](final[length - num_rows:]), atol=1e-5, rtol=1e-5)
    # Opened: the row's final state queries the passage tokens' final (normed) states; the delta changes.
    open_attention(ac)
    finals.clear()
    handle = text_decoder(ac.side.model).norm.register_forward_hook(lambda module, args, output: finals.append(output.detach().clone()))
    try:
        opened = ac.encode_batch([fixed_request(ac)], with_layers=True)[0]
    finally:
        handle.remove()
    final = finals[-1][0].float()
    with torch.no_grad():
        for layer in (2, 5):
            expected = ac.modules.delta_heads[str(layer)](ac.modules.passage_attention[str(layer)](final[length - num_rows:], final[:length - num_rows]))
            torch.testing.assert_close(opened.layer_inputs[layer], expected, atol=1e-5, rtol=1e-5)
            assert not torch.allclose(opened.layer_inputs[layer], out.layer_inputs[layer]), "the opened attention path changes the head's input"
    # Training runs end to end with the trace fields.
    stats = ActivationContextTrainer(h, config()).train(ac.name, items)
    assert stats.intervention_trace[0]["attention_out_norm"][2] > 0 and stats.intervention_trace[0]["attention_grad_norm"][2] >= 0
    assert all(-1.0 <= stats.intervention_trace[0]["cosine"][layer] <= 1.0 for layer in (2, 5))


# --------------------------------------------------------------------------------------------------- 9. frozen reader
def test_frozen_reader_neither_steps_nor_carries_the_penalty(tmp_path, hybrid_dir):
    h, ac = build(tmp_path, hybrid_dir, frequency=1)
    items = items_for(h, ac)
    trainer = ActivationContextTrainer(h, config(learning_rate_target_lora=0.0))
    assert not trainer.config.penalty_active and trainer.build_example(ac, items[0]).penalty is None
    reader_before = [parameter.detach().clone() for parameter in h.module_manager.lora_parameters("reader_lora")]
    encoder_before = shared_state(ac)
    stats = trainer.train(ac.name, items, reporting_data=items)
    assert all(torch.equal(before, after) for before, after in zip(reader_before, h.module_manager.lora_parameters("reader_lora"))), "the reader adapter is untouched"
    assert stats.penalty == [] and all(count == 0 for count in stats.penalty_counts) and stats.no_context_reference == {}
    assert stats.learning_rates[-1][1] == 0.0 and all(norm == 0.0 for norm in stats.grad_norm_reader)
    assert any(not torch.equal(tensor, value) for name, value in shared_state(ac).items() for tensor in [encoder_before[name]] if name.startswith("side:") or name.startswith("mixer.")), "the encoder trains"
    with pytest.raises(ValueError, match="nonnegative"):
        config(learning_rate_target_lora=-1.0)


# --------------------------------------------------------------------------------------------------- 10. reporter
def test_reporter_takes_deep_trace_entries(tmp_path):
    from activation.ac_model.ac_model_reporter import ActivationContextTrainingReporter
    from activation.ac_model.ac_model_training import ActivationContextTrainingStats
    reporter = ActivationContextTrainingReporter(str(tmp_path / "report"), title="test")
    stats = ActivationContextTrainingStats(loss_kind="sft")
    def entry(update):
        return {"update": update, "head_grad_norm": {7: 1.0, 31: 2.0}, "row_rms": {7: 0.5, 31: None}, "cosine": {7: 0.1, 31: 0.2},
                "attention_out_norm": {7: 0.0, 31: 0.01}, "attention_grad_norm": {7: 0.3, 31: 0.4}, "delta_rms": {7: 0.01, 31: None}, "delta_rms_abs": {7: 0.005, 31: 0.001}}
    for step in (25, 50, 75, 100):
        stats.loss.append(1.0); stats.step_tokens.append(100); stats.step_seconds.append(1.0); stats.grad_norm.append(1.0); stats.learning_rates.append([1e-4, 2e-5])
        stats.intervention_trace.append(entry(step))
        reporter.report_step(ActivationContextTrainingProgress(1, 1, 1, step, 100, step, 1.0, 1.0), stats)
    assert {"intervention_grads", "intervention_row_rms", "intervention_cosine", "intervention_attention", "intervention_delta_rms"} <= set(reporter.widgets)
    delta_series = {series["name"]: series for series in reporter.widgets["intervention_delta_rms"]["series"]}
    assert set(delta_series) == {"layer 7"} and delta_series["layer 7"]["y"] == [0.01] * 4, "None values are skipped, the ratio is plotted"
    attention_series = {series["name"]: series for series in reporter.widgets["intervention_attention"]["series"]}
    assert set(attention_series) == {"layer 7", "layer 31"} and attention_series["layer 31"]["y"] == [0.01] * 4


def test_unequal_widths_separate_encoder_base(tmp_path):
    """A narrower encoder base (16 wide) feeding a 32-wide reader: the row head has no identity skip, the deep path trains,
    the checkpoint round-trips, and a checkpoint of the shared-base model refuses to load (different bases and widths)."""
    reader_dir = tiny_hybrid(tmp_path / "reader_model"); side_dir = tiny_hybrid(tmp_path / "side_model", hidden=16, seed=222)
    h, ac = build(tmp_path / "run", reader_dir, frequency=-1, side_dir=side_dir)
    assert (ac.d_side, ac.d_target) == (16, 32) and not ac.modules.target_head.skip and ac.intervention_layers == [2, 5]
    assert ac.side.model is not ac.target.model
    output = ac.encode_batch([fixed_request(ac)], with_layers=True)[0]
    assert output.input_embeds.shape[-1] == 32 and all(delta.shape[-1] == 32 for delta in output.layer_inputs.values()) and set(output.layer_inputs) == {2, 5}
    trainer, items, examples, device = trainer_and_examples(h, ac)
    open_heads(ac); open_attention(ac)
    before = intervention_state(ac); shared_before = shared_state(ac)
    stats = trainer.train("ac_private", items, reporting_data=items[:1])
    assert stats.epochs and all(math.isfinite(value) for value in stats.loss)
    after = intervention_state(ac)
    assert any(not torch.equal(before[name], after[name]) for name in before), "the intervention modules trained"
    assert any(not torch.equal(shared_before[name], shared_state(ac)[name]) for name in shared_before), "the shared modules trained"
    folder = ac.save(str(tmp_path / "ckpt"))
    h2, twin = build(tmp_path / "twin", reader_dir, frequency=-1, side_dir=side_dir, seed=99)
    twin.load(folder)
    assert all(torch.equal(tensor, twin.modules.state_dict()[name]) for name, tensor in ac.modules.state_dict().items())
    assert torch.allclose(twin.encode_batch([fixed_request(twin)], with_layers=True)[0].input_embeds, ac.encode_batch([fixed_request(ac)], with_layers=True)[0].input_embeds, atol=1e-5)
    h3, shared = build(tmp_path / "shared", reader_dir, frequency=-1)
    with pytest.raises(ValueError, match="identities or widths"):
        shared.load(folder)
