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
import torch.nn.functional as F

from activation.ac_model import (ActivationContextModelConfig, ActivationContextStudyGenerator, ActivationContextTrainer,
                                 ActivationContextTrainingConfig)
from activation.ac_model.ac_model_training import ActivationContextTrainingProgress
from activation.ac_model.ac_model import READOUT_MODULE_PREFIXES
from activation.ac_model.ac_model_utils import ACOutput, EncodeRequest, PassageAttention, resolve_intervention_layers
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
          ac_checkpoint: str | None = None, reader_checkpoint: str | None = None, layers: list[int] | None = None, side_dir: Path | None = None,
          **config_overrides):
    """The shared/reader base with a rank-4 adapter and a tiny AC model on top, in the same construction order every time
    (seeded: the AC modules draw first, the adapters draw at injection). `side_dir`: a separate (narrower) encoder base;
    `config_overrides`: further ActivationContextModelConfig fields (recursive_keys, input_pooling_stride, ...)."""
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
        intervention_layers=layers, **config_overrides))
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
    keys = list(ac.modules.state_dict().keys())
    readout_keys = [name for name in keys if name.startswith(READOUT_MODULE_PREFIXES)]
    assert [name for name in keys if name not in readout_keys] == json.loads((DATA / "legacy_keys.json").read_text())
    assert readout_keys and all(name.startswith(("row_readout.", "recursive_readout.")) for name in readout_keys)
    legacy = torch.load(DATA / "legacy_ac_modules.pt", map_location="cpu", weights_only=False)
    loaded = ac.modules.load_state_dict(legacy["modules"], strict=False)          # the pre-change checkpoint: only the read-outs (identity at init) are new
    assert sorted(loaded.missing_keys) == sorted(readout_keys) and not loaded.unexpected_keys
    ac.modules.target_head.skip = False                                           # the reference fixture predates the identity-skip row head (same keys, plain FFN)
    assert "interventions" not in legacy
    reference = torch.load(DATA / "legacy_outputs.pt", map_location="cpu", weights_only=False)
    loaded = ac.modules.load_state_dict(reference["modules"], strict=False)
    assert sorted(loaded.missing_keys) == sorted(readout_keys) and not loaded.unexpected_keys
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


def test_linear_skip_row_head(tmp_path, capsys):
    """`row_head_linear_skip`: with unequal widths the row head adds W·state to its zero-started FFN, W the least-squares map from the
    side base's input embeddings to the reader's over the shared vocabulary, filled at the first placement. A fresh row is then the
    normalized mapped state at the reader's embedding scale; the map trains and round-trips; a checkpoint without the skip refuses
    to load into a model with it and the reverse; equal widths refuse the option."""
    reader_dir = tiny_hybrid(tmp_path / "reader_model"); side_dir = tiny_hybrid(tmp_path / "side_model", hidden=16, seed=222)
    h, ac = build(tmp_path / "run", reader_dir, frequency=-1, side_dir=side_dir, row_head_linear_skip=True)
    head = ac.modules.target_head
    assert not head.skip and head.linear is not None and head.linear.weight.shape == (32, 16)
    assert "embedding map" in capsys.readouterr().out
    weight = head.linear.weight.detach().clone()
    assert torch.isfinite(weight).all() and float(weight.abs().max()) > 0
    side_embeddings = ac.side.model.get_input_embeddings().weight.detach().float()
    reader_embeddings = ac.target.model.get_input_embeddings().weight
    assert side_embeddings.shape[0] == reader_embeddings.shape[0]
    # The tiny bases' random tables share no structure (the fit is whatever 16 directions explain); a reader table that is a linear
    # image of the side's is recovered: the map is that image, the fit cosine ~1.
    torch.manual_seed(7); image = torch.randn(16, 32) * 0.05
    with torch.no_grad():
        reader_embeddings.copy_(side_embeddings @ image)
        ac.modules.scales_initialized.fill_(False)
    ac._ensure_scales()
    fit = float(capsys.readouterr().out.split("mean cosine of the fit ")[1].split(")")[0])
    weight = head.linear.weight.detach().clone()
    assert fit > 0.98 and torch.allclose(weight.T, image, atol=1e-3, rtol=1e-2), fit
    states = torch.randn(5, 16)
    mapped = states @ weight.T                                                                        # the FFN starts at zero: the row is the mapped state, RMS-normalized
    expected = mapped / mapped.pow(2).mean(dim=-1, keepdim=True).add(1e-6).sqrt() * head.scale
    torch.testing.assert_close(head(states), expected, atol=1e-5, rtol=1e-4)
    plain_h, plain = build(tmp_path / "plain", reader_dir, frequency=-1, side_dir=side_dir)
    assert plain.modules.target_head.linear is None and "linear.weight" not in dict(plain.modules.named_parameters())
    assert torch.equal(plain.modules.summary_marker, ac.modules.summary_marker)                     # the skip's Linear draws under a fork: the twin draws as before
    assert torch.equal(plain.modules.target_head.ffn.up.weight, head.ffn.up.weight)
    assert not torch.allclose(plain.encode_batch([fixed_request(plain)], with_layers=True)[0].input_embeds, ac.encode_batch([fixed_request(ac)], with_layers=True)[0].input_embeds)
    trainer, items, examples, device = trainer_and_examples(h, ac)
    open_heads(ac)
    stats = trainer.train("ac_private", items, reporting_data=items[:1])
    assert stats.epochs and all(math.isfinite(value) for value in stats.loss)
    assert not torch.equal(head.linear.weight.detach(), weight), "the map trains"
    folder = ac.save(str(tmp_path / "ckpt"))
    twin_h, twin = build(tmp_path / "twin", reader_dir, frequency=-1, side_dir=side_dir, seed=99, row_head_linear_skip=True)
    twin.load(folder)
    assert torch.equal(twin.modules.target_head.linear.weight, head.linear.weight)
    assert torch.allclose(twin.encode_batch([fixed_request(twin)], with_layers=True)[0].input_embeds, ac.encode_batch([fixed_request(ac)], with_layers=True)[0].input_embeds, atol=1e-5)
    with pytest.raises(ValueError, match="row_head_linear_skip"):
        plain.load(folder)
    plain_folder = plain.save(str(tmp_path / "plain_ckpt"))
    with pytest.raises(ValueError, match="row_head_linear_skip"):
        twin.load(plain_folder)
    with pytest.raises(ValueError, match="unequal"):
        build(tmp_path / "equal", reader_dir, frequency=-1, row_head_linear_skip=True)


# --------------------------------------------------------------------------------------------------- 12. read-outs everywhere, nested parts
def nested_request(child_text: str, ratio: float = 0.25) -> EncodeRequest:
    """A two-level part: parent text around one activation_context child (its own messages, its own compression target)."""
    return EncodeRequest([{"role": "user", "content": [{"type": "text", "text": "parent begins here"},
                                                       {"type": "activation_context", "compression_target": ratio, "messages": [{"role": "user", "content": child_text}]},
                                                       {"type": "text", "text": "parent ends here"}]}], ratio, False, None)


def open_readouts(ac, seed: int = 5, std: float = 0.05) -> None:
    """Every read-out's zero output projection and every delta head's zero down projection moved off zero (the same values for the same seed)."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        with torch.no_grad():
            blocks = [ac.modules.row_readout, ac.modules.recursive_readout, *(ac.modules.passage_attention.values() if ac.modules.passage_attention is not None else [])]
            for block in blocks:
                block.out.weight.normal_(std=std)
            for head in (ac.modules.delta_heads.values() if ac.modules.delta_heads is not None else []):
                head.down.weight.normal_(std=std)


def test_nested_child_keys(tmp_path, hybrid_dir):
    h_off, off = build(tmp_path / "off", hybrid_dir, frequency=1, seed=61)
    h_on, on = build(tmp_path / "on", hybrid_dir, frequency=1, seed=61, recursive_keys=True)
    assert not off.config.recursive_keys and on.config.recursive_keys and on.intervention_summary()["recursive_keys"] is True
    off.set_mode("training"); on.set_mode("training")                              # no rollout cache: every encode below sees the current weights
    for block in (on.modules.row_readout, on.modules.recursive_readout):
        assert torch.equal(block.out.weight, torch.zeros_like(block.out.weight)), "read-outs are the identity at init"
    # Closed read-outs: the key set is irrelevant, both models agree on a nested request (and the child result carries its states).
    child = on.encode_batch([EncodeRequest([{"role": "user", "content": PASSAGES[1]["text"]}], 0.25, True, None)], with_layers=True)[0]
    assert child.states is not None and child.states.shape == (child.input_embeds.shape[0], on.d_side) and child.layer_inputs == {}
    a, b = off.encode_batch([nested_request(PASSAGES[1]["text"])], with_layers=True)[0], on.encode_batch([nested_request(PASSAGES[1]["text"])], with_layers=True)[0]
    torch.testing.assert_close(a.input_embeds, b.input_embeds); torch.testing.assert_close(a.layer_inputs[2], b.layer_inputs[2])
    # Closed read-outs are trainable: the zero output projections receive gradient from the outputs they gate.
    out0 = on.encode_batch([nested_request(PASSAGES[1]["text"])], with_layers=True)[0]
    (out0.input_embeds.square().sum() + sum(delta.square().sum() for delta in out0.layer_inputs.values())).backward()
    for block in (on.modules.row_readout, on.modules.recursive_readout):
        assert block.out.weight.grad is not None and block.out.weight.grad.abs().sum() > 0, "a closed read-out must still receive gradient"
    on.modules.zero_grad(set_to_none=True)
    # Opened read-outs (the same values in both): the child keys change the parent's rows and deltas; a flat request stays identical.
    open_readouts(off); open_readouts(on)
    a, b = off.encode_batch([nested_request(PASSAGES[1]["text"])], with_layers=True)[0], on.encode_batch([nested_request(PASSAGES[1]["text"])], with_layers=True)[0]
    assert not torch.allclose(a.input_embeds, b.input_embeds) and not torch.allclose(a.layer_inputs[2], b.layer_inputs[2])
    fa, fb = off.encode_batch([fixed_request(off)], with_layers=True)[0], on.encode_batch([fixed_request(on)], with_layers=True)[0]
    torch.testing.assert_close(fa.input_embeds, fb.input_embeds); torch.testing.assert_close(fa.layer_inputs[5], fb.layer_inputs[5])
    # The parent text fixed, the child text changed: the parent's rows and deltas change.
    c = on.encode_batch([nested_request(PASSAGES[0]["text"])], with_layers=True)[0]
    assert not torch.allclose(b.input_embeds, c.input_embeds) and not torch.allclose(b.layer_inputs[5], c.layer_inputs[5])
    # Gradients reach the child level: the recursive read-out and the child's side states feed the parent's outputs.
    out = on.encode_batch([nested_request(PASSAGES[1]["text"])], with_layers=True)[0]
    (out.input_embeds.square().sum() + sum(delta.square().sum() for delta in out.layer_inputs.values())).backward()
    assert on.modules.recursive_readout.query.weight.grad is not None and on.modules.recursive_readout.query.weight.grad.abs().sum() > 0
    assert on.modules.row_readout.query.weight.grad is not None and on.modules.row_readout.query.weight.grad.abs().sum() > 0
    assert on.modules.recursive_adapter.ffn.up.weight.grad is not None


def test_readout_migration(tmp_path, hybrid_dir, capsys):
    """A checkpoint written before the row read-outs existed loads with them fresh (identity) and reproduces its outputs."""
    h, ac = build(tmp_path, hybrid_dir, frequency=1, seed=8)
    before = ac.encode_batch([fixed_request(ac)], with_layers=True)[0]
    folder = Path(ac.save(str(tmp_path / "saved")))
    state = torch.load(folder / "ac_modules.pt", map_location="cpu", weights_only=False)
    removed = {name: tensor for name, tensor in state["modules"].items() if name.startswith(READOUT_MODULE_PREFIXES)}
    assert removed
    state["modules"] = {name: tensor for name, tensor in state["modules"].items() if name not in removed}
    torch.save(state, folder / "ac_modules.pt")
    h2, ac2 = build(tmp_path / "second", hybrid_dir, frequency=1, seed=9)
    ac2.load(str(folder))
    assert "Row read-outs fresh" in capsys.readouterr().out
    after = ac2.encode_batch([fixed_request(ac2)], with_layers=True)[0]
    torch.testing.assert_close(after.input_embeds, before.input_embeds); torch.testing.assert_close(after.layer_inputs[2], before.layer_inputs[2])
    # A partial set of read-out weights is not a migration.
    name, tensor = next(iter(removed.items()))
    state["modules"][name] = tensor
    torch.save(state, folder / "ac_modules.pt")
    with pytest.raises(ValueError, match="do not match"):
        ac2.load(str(folder))


def test_content_pooling(tmp_path, hybrid_dir, capsys):
    with pytest.raises(ValueError, match="stride"):
        ActivationContextModelConfig("x", "s", "sl", "t", "tl", input_pooling_stride=5, input_pooling_window=4)
    h0, plain = build(tmp_path / "plain", hybrid_dir, frequency=1, seed=12)
    h, ac = build(tmp_path / "pooled", hybrid_dir, frequency=1, seed=12, input_pooling_stride=2, input_pooling_window=4)
    assert plain.modules.pooling is None and ac.modules.pooling is not None and ac.modules.pooling.stride == 2
    plain.profile_phases = ac.profile_phases = True
    out = ac.encode_batch([fixed_request(ac)], with_layers=True)[0]
    plain.encode_batch([fixed_request(plain)], with_layers=True)
    (pooled_len,), (plain_len,) = ac.stats.decoder_sequences, plain.stats.decoder_sequences
    assert pooled_len < plain_len, (pooled_len, plain_len)
    assert "pooling" in ac.stats.phase_seconds
    # Round trip into a model of the same shape; a stride-0 checkpoint is refused by a stride-2 model and vice versa.
    folder = ac.save(str(tmp_path / "saved"))
    h2, again = build(tmp_path / "again", hybrid_dir, frequency=1, seed=99, input_pooling_stride=2, input_pooling_window=4)
    again.load(folder)
    back = again.encode_batch([fixed_request(again)], with_layers=True)[0]
    torch.testing.assert_close(back.input_embeds, out.input_embeds); torch.testing.assert_close(back.layer_inputs[5], out.layer_inputs[5])
    # An unpooled checkpoint migrates into a pooled model (the strided arms warm-start this way): the pooler is fresh, everything else loads.
    plain_folder = plain.save(str(tmp_path / "plain_saved"))
    h3, warm = build(tmp_path / "warm", hybrid_dir, frequency=1, seed=77, input_pooling_stride=2, input_pooling_window=4)
    pooler_before = warm.modules.pooling.ffn.up.weight.clone()
    warm.load(plain_folder)
    assert "Content pooling fresh" in capsys.readouterr().out
    torch.testing.assert_close(warm.modules.pooling.ffn.up.weight, pooler_before)                      # fresh: the pooler keeps its init
    torch.testing.assert_close(warm.modules.target_head.ffn.up.weight, plain.modules.target_head.ffn.up.weight)   # the rest is the checkpoint's
    # A pooled checkpoint into an unpooled model, and a stride change, stay refused.
    with pytest.raises(ValueError, match="input_pooling_stride"):
        plain.load(folder)
    h4, other = build(tmp_path / "other", hybrid_dir, frequency=1, seed=78, input_pooling_stride=4, input_pooling_window=4)
    with pytest.raises(ValueError, match="input_pooling_stride"):
        other.load(folder)
    # The pooler's parameters receive gradient.
    ac.set_mode("training")
    graded = ac.encode_batch([fixed_request(ac)], with_layers=True)[0]
    (graded.input_embeds.square().sum() + sum(delta.square().sum() for delta in graded.layer_inputs.values())).backward()
    pooling_params = [(name, p) for name, p in ac.modules.named_parameters() if name.startswith("pooling.") and p.requires_grad]
    assert pooling_params and all(p.grad is not None and p.grad.abs().sum() > 0 for _, p in pooling_params), [n for n, p in pooling_params if p.grad is None]
    ac.modules.zero_grad(set_to_none=True)
    # Batch padding does not change a request's result: alone or beside a longer request, the same outputs.
    short, long = fixed_request(ac), EncodeRequest([{"role": "user", "content": PASSAGES[1]["text"] + " " + PASSAGES[0]["text"]}], 0.25, True, None)
    with torch.no_grad():
        alone = ac.encode_batch([short], with_layers=True)[0]
        padded = ac.encode_batch([short, long], with_layers=True)[0]
    torch.testing.assert_close(padded.input_embeds, alone.input_embeds, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(padded.layer_inputs[5], alone.layer_inputs[5], atol=1e-5, rtol=1e-4)
    # Training runs end to end with the pooler in the graph.
    stats = ActivationContextTrainer(h, config()).train(ac.name, items_for(h, ac))
    assert stats.intervention_trace


def test_interleaved_summaries(tmp_path, hybrid_dir, capsys):
    """`summary_layout="interleaved"`: each summary sits in the causal side pass right after its pooling bin, so a summary's row
    depends on its own bin and the bins before it only (the appended layout's rows see the whole content). Same modules: an
    appended checkpoint migrates into an interleaved model, the reverse is refused."""
    from activation.ac_model.ac_model_utils import interleaved_positions, summary_bins
    cpu = torch.device("cpu")
    for length, num_rows in ((10, 3), (2, 3), (1, 1), (17, 5), (64, 8)):
        tokens, summaries = interleaved_positions(length, num_rows, cpu)
        assert sorted(torch.cat([tokens, summaries]).tolist()) == list(range(length + num_rows))       # a permutation of the decoder positions
        _, ends = summary_bins(length, num_rows, cpu)
        for k in range(num_rows):                                                                       # after its bin's last token, before the next token
            assert summaries[k] > tokens[ends[k] - 1] and (ends[k] == length or summaries[k] < tokens[ends[k]])
        assert torch.all(summaries[1:] > summaries[:-1]) and torch.all(tokens[1:] > tokens[:-1])
    with pytest.raises(ValueError, match="summary_layout"):
        ActivationContextModelConfig("x", "s", "sl", "t", "tl", summary_layout="sideways")
    with pytest.raises(ValueError, match="content pooling"):
        ActivationContextModelConfig("x", "s", "sl", "t", "tl", summary_layout="interleaved", input_pooling_stride=2)
    h0, appended = build(tmp_path / "appended", hybrid_dir, frequency=1, seed=12)
    h, ac = build(tmp_path / "interleaved", hybrid_dir, frequency=1, seed=12, summary_layout="interleaved")
    assert torch.equal(ac.modules.summary_marker, appended.modules.summary_marker)                     # the layout draws nothing
    words = (PASSAGES[0]["text"] + " " + PASSAGES[1]["text"] + " " + PASSAGES[2]["text"]).split()
    text = " ".join(words * 4)
    changed = " ".join((words * 4)[:-4] + ["one", "two", "three", "alpha"])                          # the last bin's tokens differ, the token count does not
    tokenizer = ac.side.tokenizer
    assert len(tokenizer(text, add_special_tokens=False).input_ids) == len(tokenizer(changed, add_special_tokens=False).input_ids) > 100
    requests = [EncodeRequest([{"role": "user", "content": body}], 0.25, False, None) for body in (text, changed)]
    appended.profile_phases = ac.profile_phases = True
    with torch.no_grad():
        ours = [ac.encode_batch([request], with_layers=True)[0] for request in requests]
        theirs = [appended.encode_batch([request], with_layers=True)[0] for request in requests]
    assert ac.stats.decoder_sequences == appended.stats.decoder_sequences                             # the same content and summaries, differently placed
    assert ours[0].input_embeds.shape == theirs[0].input_embeds.shape == (8, appended.d_target)
    assert not torch.allclose(ours[0].input_embeds, theirs[0].input_embeds)
    # At init the read-outs are identity, so a row is its summary's final state: interleaved, the first summary cannot see the last bin.
    torch.testing.assert_close(ours[0].input_embeds[0], ours[1].input_embeds[0])
    assert not torch.allclose(ours[0].input_embeds[-1], ours[1].input_embeds[-1])
    assert not torch.allclose(theirs[0].input_embeds[0], theirs[1].input_embeds[0])
    # Batch padding does not change a request's result (beside a 130-token request the tiny hybrid's padded pass drifts by ~1e-4 on
    # rows of RMS 0.02 in either layout: the reference is the appended model's own drift).
    short = fixed_request(ac)
    with torch.no_grad():
        alone = ac.encode_batch([short], with_layers=True)[0]
        padded = ac.encode_batch([short, requests[0]], with_layers=True)[0]
        alone_appended = appended.encode_batch([short], with_layers=True)[0]
        padded_appended = appended.encode_batch([short, requests[0]], with_layers=True)[0]
    drift = (padded_appended.input_embeds - alone_appended.input_embeds).abs().max().item()
    torch.testing.assert_close(padded.input_embeds, alone.input_embeds, atol=max(2 * drift, 1e-6), rtol=0)
    torch.testing.assert_close(padded.layer_inputs[5], alone.layer_inputs[5], atol=1e-5, rtol=1e-4)
    # Checkpoints: the layout is saved; a round trip reproduces the outputs; an appended checkpoint migrates in; the reverse is refused.
    folder = ac.save(str(tmp_path / "saved"))
    assert torch.load(Path(folder) / "ac_modules.pt", map_location="cpu", weights_only=False)["config"]["summary_layout"] == "interleaved"
    h2, again = build(tmp_path / "again", hybrid_dir, frequency=1, seed=99, summary_layout="interleaved")
    again.load(folder)
    with torch.no_grad():
        back = again.encode_batch([requests[0]], with_layers=True)[0]
    torch.testing.assert_close(back.input_embeds, ours[0].input_embeds); torch.testing.assert_close(back.layer_inputs[5], ours[0].layer_inputs[5])
    appended_folder = appended.save(str(tmp_path / "appended_saved"))
    h3, warm = build(tmp_path / "warm", hybrid_dir, frequency=1, seed=77, summary_layout="interleaved")
    warm.load(appended_folder)
    assert "Summaries interleaved" in capsys.readouterr().out
    torch.testing.assert_close(warm.modules.summary_marker, appended.modules.summary_marker)
    with pytest.raises(ValueError, match="summary_layout"):
        appended.load(folder)
    # Gradient reaches the marker and the summary pooling through the interleaved sequence; training runs end to end.
    ac.set_mode("training")
    graded = ac.encode_batch([requests[0]], with_layers=True)[0]
    (graded.input_embeds.square().sum() + sum(delta.square().sum() for delta in graded.layer_inputs.values())).backward()
    assert ac.modules.summary_marker.grad is not None and ac.modules.summary_marker.grad.abs().sum() > 0
    assert ac.modules.summary_pooling.ffn.up.weight.grad is not None and ac.modules.summary_pooling.ffn.up.weight.grad.abs().sum() > 0
    ac.modules.zero_grad(set_to_none=True)
    stats = ActivationContextTrainer(h, config()).train(ac.name, items_for(h, ac))
    assert stats.intervention_trace


def test_density_weighted_bins(tmp_path, hybrid_dir, capsys):
    """`bin_density_weight` w > 0: the summary bins hold equal cumulative token weight 1 + w·dense (digits, code/math symbols,
    ac_model_utils.is_dense_piece) instead of equal token counts, so notation gets finer rows at the same V; w = 0 is the
    equal-token model unchanged, prose without dense tokens pools the same bins at any w, and a checkpoint moves between weights
    with a note (same modules)."""
    from activation.ac_model.ac_model_utils import interleaved_positions, is_dense_piece, summary_bins
    assert [is_dense_piece(p) for p in (" the", " 12", " x=", "—", "é", " α", " (", "�", " →")] == [False, True, True, False, False, False, False, False, True]
    cpu = torch.device("cpu")
    for length, num_rows in ((10, 3), (17, 5), (64, 8), (2, 3), (1, 1)):
        plain = summary_bins(length, num_rows, cpu)
        uniform = summary_bins(length, num_rows, cpu, torch.ones(length))
        assert torch.equal(plain[0], uniform[0]) and torch.equal(plain[1], uniform[1])             # uniform weights: the equal-token bins
        heavy = torch.ones(length)
        heavy[: length // 2] = 3.
        starts, ends = summary_bins(length, num_rows, cpu, heavy)
        assert starts[0] == 0 and ends[-1] == length and torch.all(ends > starts)                  # cover the input, nonempty
        assert torch.all(starts[1:] >= starts[:-1]) and torch.all(ends[1:] >= ends[:-1]) and torch.all(starts[1:] <= ends[:-1])
        if num_rows < length:
            widths = (ends - starts).float()
            assert widths[: num_rows // 2].mean() <= widths[num_rows // 2:].mean()                  # the heavier first half gets the narrower bins
        tokens, summaries = interleaved_positions(length, num_rows, cpu, heavy)
        assert sorted(torch.cat([tokens, summaries]).tolist()) == list(range(length + num_rows))
        for k in range(num_rows):
            assert summaries[k] > tokens[ends[k] - 1] and (ends[k] == length or summaries[k] < tokens[ends[k]])
    with pytest.raises(ValueError, match="bin_density_weight"):
        ActivationContextModelConfig("x", "s", "sl", "t", "tl", bin_density_weight=-1.)
    with pytest.raises(ValueError, match="one weight per content token"):
        summary_bins(10, 3, cpu, torch.ones(9))
    h0, plain_model = build(tmp_path / "plain", hybrid_dir, frequency=1, seed=12, summary_layout="interleaved")
    h, dense = build(tmp_path / "dense", hybrid_dir, frequency=1, seed=12, summary_layout="interleaved", bin_density_weight=1.)
    assert torch.equal(dense.modules.summary_marker, plain_model.modules.summary_marker)             # the weight draws nothing
    tokenizer = dense.side.tokenizer                                                               # the tiny vocabulary: a few words and the numbers 0-63
    prose = " ".join(["alpha beta gamma delta evidence question answer first second third one two three four five six seven eight nine zero"
                      " yes no tool result beginning middle end context reasoning code search solver useful wrong"] * 4)
    ids = tokenizer(prose, add_special_tokens=False).input_ids
    assert len(ids) > 100 and not any(i in tokenizer.all_special_ids or is_dense_piece(tokenizer.decode([i])) for i in ids)
    numbers = " ".join(str(i * 37 % 64) for i in range(48)) + " " + " ".join(prose.split()[:64])   # dense tokens concentrated in the first half
    requests = [EncodeRequest([{"role": "user", "content": body}], 0.25, False, None) for body in (prose, numbers)]
    with torch.no_grad():
        ours = [dense.encode_batch([request], with_layers=True)[0] for request in requests]
        theirs = [plain_model.encode_batch([request], with_layers=True)[0] for request in requests]
    assert dense._dense_vocab is not None and dense._dense_vocab.sum() > 0
    torch.testing.assert_close(ours[0].input_embeds, theirs[0].input_embeds)                        # no dense token: the same bins
    assert ours[1].input_embeds.shape == theirs[1].input_embeds.shape and not torch.allclose(ours[1].input_embeds, theirs[1].input_embeds)   # digits: the bins move
    folder = dense.save(str(tmp_path / "saved"))
    assert torch.load(Path(folder) / "ac_modules.pt", map_location="cpu", weights_only=False)["config"]["bin_density_weight"] == 1.
    h2, again = build(tmp_path / "again", hybrid_dir, frequency=1, seed=99, summary_layout="interleaved", bin_density_weight=1.)
    again.load(folder)
    with torch.no_grad():
        back = again.encode_batch([requests[1]], with_layers=True)[0]
    torch.testing.assert_close(back.input_embeds, ours[1].input_embeds)
    torch.testing.assert_close(back.layer_inputs[5], ours[1].layer_inputs[5])
    plain_model.load(folder)                                                                        # a weighted checkpoint into an equal-token model: noted, not refused
    assert "density weight 0.0" in capsys.readouterr().out
    dense.set_mode("training")
    graded = dense.encode_batch([requests[1]], with_layers=True)[0]
    (graded.input_embeds.square().sum() + sum(delta.square().sum() for delta in graded.layer_inputs.values())).backward()
    assert dense.modules.summary_pooling.ffn.up.weight.grad is not None and dense.modules.summary_pooling.ffn.up.weight.grad.abs().sum() > 0
    dense.modules.zero_grad(set_to_none=True)
    stats = ActivationContextTrainer(h, config()).train(dense.name, items_for(h, dense))
    assert stats.intervention_trace
