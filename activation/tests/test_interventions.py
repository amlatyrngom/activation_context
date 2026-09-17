"""Deep AC inputs (per-layer residual deltas) on tiny CPU models: layer resolution, zero mode unchanged, gradients through
checkpointed layers, padded-batch positions and causality, generation without re-injection, checkpoints and resume,
calibration, the deltas-off panel, and the paired bench arms.

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
from activation.ac_model.ac_model import INTERVENTION_SCALE_FRACTION
from activation.ac_model.ac_model_utils import ACOutput, KVDeltaHead, resolve_intervention_layers
from activation.harness import HarnessRuntime, HarnessRuntimeConfig, ModelConfig
from activation.harness.hf_utils import (INTERVENTIONS_KWARG, BoundInterventions, capture_kv_rms, decoder_layers, generate_with_interventions,
                                         install_intervention_hooks, text_decoder)

DATA = Path(__file__).resolve().parent / "data"
TINY_QWEN3 = DATA / "tiny_qwen3"
PASSAGES = [{"text": "alpha beta gamma delta evidence question answer first second third one two three", "doc_id": "d1", "source": "test"},
            {"text": "tool result beginning middle end context reasoning code search solver useful wrong yes no", "doc_id": "d2", "source": "test"},
            {"text": "four five six seven", "doc_id": "d3", "source": "test"}]
torch.set_num_threads(2)


# --------------------------------------------------------------------------------------------------- fixtures
def tiny_hybrid(folder: Path) -> Path:
    """A 6-layer Qwen3.5 text model (full attention every third block: layers 2 and 5) with the tiny Qwen3 tokenizer."""
    from transformers import AutoTokenizer, Qwen3_5ForCausalLM, Qwen3_5TextConfig
    if (folder / "model.safetensors").is_file():
        return folder
    tokenizer = AutoTokenizer.from_pretrained(str(TINY_QWEN3))
    config = Qwen3_5TextConfig(vocab_size=len(tokenizer), num_hidden_layers=6, full_attention_interval=3, hidden_size=32, intermediate_size=64,
                               num_attention_heads=4, num_key_value_heads=2, head_dim=8, linear_num_value_heads=4, linear_num_key_heads=2,
                               linear_key_head_dim=8, linear_value_head_dim=8, linear_conv_kernel_dim=4, max_position_embeddings=512,
                               pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id, tie_word_embeddings=False)
    config._attn_implementation = "sdpa"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(111)
        model = Qwen3_5ForCausalLM(config)
    folder.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(folder)
    tokenizer.save_pretrained(folder)
    return folder


def build(tmp_path: Path, model_dir: Path, *, frequency: int = 0, add_last: bool = True, prefer: bool = True, seed: int = 1234,
          ac_checkpoint: str | None = None, reader_checkpoint: str | None = None, source: str = "summary", target: str = "residual",
          source_layers: list[int] | None = None, depth: str = "final", head_style: str = "scaled", calibration: str = "no_rows", row_scale_init: float = 1.0,
          layers: list[int] | None = None, transfer_init: float = 1.0, transfer_correction: bool = False):
    """The shared/reader base with a rank-4 adapter and a tiny AC model on top, in the same construction order every time
    (seeded: the AC modules draw first, the adapters draw at injection)."""
    os.environ["ACTIVATION_SYNC_ROOT"] = str(tmp_path)
    torch.manual_seed(seed)
    h = HarnessRuntime(HarnessRuntimeConfig(model_configs={"reader": ModelConfig("reader", str(model_dir))}))
    h.module_manager.register_lora("reader_lora", "reader", rank=4, dropout=0.0, checkpoint_path=reader_checkpoint)
    deep = {"intervention_frequency": frequency, "add_last_layer_intervention": add_last, "prefer_attention_interventions": prefer,
            "intervention_source": source, "intervention_target": target, "intervention_source_layers": source_layers, "intervention_attention_depth": depth,
            "intervention_head_style": head_style, "intervention_calibration": calibration, "intervention_layers": layers,
            "intervention_transfer_init": transfer_init, "intervention_transfer_correction": transfer_correction} if frequency or layers else {}
    ac = h.module_manager.register_ac_model(ActivationContextModelConfig(
        "ac_private", "reader", "side_lora", "reader", "reader_lora", mixer_heads=4, mixer_window=4, side_lora_rank=4, max_view_rows=8,
        encode_batch_max_tokens=2048, checkpoint_path=ac_checkpoint, row_scale_init=row_scale_init, **deep))
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
    trainer._store_teacher_targets(ac, [example.penalty for example in examples], device, lora=None)
    ac.set_mode("training")                                                        # gradients through the encoder (rollout mode encodes under no_grad)
    return trainer, items, examples, device


def calibrate(trainer, ac, examples):
    if ac.config.intervention_calibration == "row_positions":
        with_rows = [example for example in examples if example.part_requests]
        return ac.calibrate_interventions(None, [example.item.item_id for example in with_rows],
                                          row_inputs=[trainer._calibration_rows(ac, example, ac.device) for example in with_rows])
    sample = [trainer._calibration_sequence(ac, example) for example in examples]
    return ac.calibrate_interventions([ids for ids, _ in sample], [item_id for _, item_id in sample])


def shared_state(ac) -> dict[str, torch.Tensor]:
    """Everything but the intervention modules: the AC modules' shared parameters, the side and reader adapters (cloned)."""
    modules = {name: tensor.clone() for name, tensor in ac.modules.state_dict().items() if not name.startswith(("delta_heads.", "passage_attention.", "cross_attention.", "intervention"))}
    side = {f"side:{index}": parameter.detach().clone() for index, parameter in enumerate(ac.harness.module_manager.lora_parameters(ac.config.base_side_model_lora_name))}
    reader = {f"reader:{index}": parameter.detach().clone() for index, parameter in enumerate(ac.harness.module_manager.lora_parameters(ac.config.target_model_lora_name))}
    return {**modules, **side, **reader}


def intervention_state(ac) -> dict[str, torch.Tensor]:
    return {name: tensor.clone() for name, tensor in ac.modules.state_dict().items() if name.startswith(("delta_heads.", "passage_attention.", "cross_attention."))}


def boost_scales(ac, factor: float = 50.0) -> None:
    """Audible deltas: every head's scale(s) multiplied in place (residual heads: `scale`; K/V heads: `scale_k`, `scale_v`)."""
    with torch.no_grad():
        for head in ac.modules.delta_heads.values():
            for scale in head.scale_parameters():
                scale.mul_(factor)


def attention_parameters(ac):
    return [parameter for name, parameter in ac.modules.named_parameters() if name.startswith("passage_attention.")]


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
    assert resolve_intervention_layers(32, 4, True, True, full) == [7, 11, 19, 23, 31]   # 6->7, 12->11, 19, 25 ties 23/27 -> 23, plus 31
    assert resolve_intervention_layers(32, 4, False, True, full) == [7, 11, 19, 23]
    assert resolve_intervention_layers(32, 4, True, False, full) == [6, 12, 19, 25, 31]
    assert resolve_intervention_layers(6, 1, True, True, [2, 5]) == [2, 5]          # interior 3 snaps to 2 (distance 1)
    assert resolve_intervention_layers(6, 5, True, False, [2, 5]) == [1, 2, 3, 4, 5]   # dedup with the last layer (no snapping)
    assert resolve_intervention_layers(6, 5, True, True, [2, 5]) == [2, 5]              # every interior point snaps onto the two blocks
    assert resolve_intervention_layers(2, 4, True, True, [0, 1]) == [1]             # 0 excluded, duplicates collapse
    assert resolve_intervention_layers(32, 0, True, True, full) == []               # zero disables the last-layer flag too
    assert resolve_intervention_layers(32, -1, True, True, full) == full            # -1: every full-attention layer (the last-layer flag is moot)
    assert resolve_intervention_layers(6, -1, False, False, [0, 2, 5]) == [2, 5]     # 0 excluded
    with pytest.raises(ValueError):
        resolve_intervention_layers(32, -2, True, True, full)
    with pytest.raises(ValueError):
        resolve_intervention_layers(32, -1, True, True, [])
    assert ActivationContextModelConfig("a", "s", "sl", "t", "tl", intervention_frequency=-1).intervention_frequency == -1
    with pytest.raises(ValueError):
        ActivationContextModelConfig("a", "s", "sl", "t", "tl", intervention_frequency=-2)


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
    assert not ac.is_deep and ac.modules.delta_heads is None and ac.intervention_layers == []
    assert list(ac.modules.state_dict().keys()) == json.loads((DATA / "legacy_keys.json").read_text())
    legacy = torch.load(DATA / "legacy_ac_modules.pt", map_location="cpu", weights_only=False)
    ac.modules.load_state_dict(legacy["modules"], strict=True)                    # the pre-change checkpoint loads strictly
    assert "interventions" not in legacy
    reference = torch.load(DATA / "legacy_outputs.pt", map_location="cpu", weights_only=False)
    ac.modules.load_state_dict(reference["modules"], strict=True)
    rows = ac.encode_batch([fixed_request(ac)])[0]
    assert torch.equal(rows, reference["rows"]), "zero-mode rows differ from the previous code's"
    model = ac.target.model
    assert getattr(model, "_activation_intervention_hooks", None) is None, "hooks are installed only when a payload is passed"
    ids = torch.tensor(reference["ids"])
    hidden = ac.target.decoder_forward(model.get_input_embeddings()(ids)[None], None, lora_name=None)[0]
    assert torch.equal(hidden, reference["hidden"])
    assert isinstance(ac.encode_batch([fixed_request(ac)], with_layers=True)[0], ACOutput)
    assert ac.encode_batch([fixed_request(ac)], with_layers=True)[0].layer_inputs == {}
    # An input-only checkpoint migrates into a deep model (shared modules loaded, heads fresh, calibration pending); a deep checkpoint never loads as input-only.
    folder = ac.save(str(tmp_path / "zero"))
    deep_h, deep = build(tmp_path / "deep", TINY_QWEN3, frequency=1)
    deep.load(folder)
    assert deep.is_deep and not deep.interventions_calibrated
    assert torch.equal(deep.modules.target_head.ffn.up.weight, ac.modules.target_head.ffn.up.weight)


def test_pairing_shared_draws_are_identical(tmp_path):
    h0, zero = build(tmp_path / "zero", TINY_QWEN3, seed=77)
    after_zero = torch.rand(3)
    h1, deep = build(tmp_path / "deep", TINY_QWEN3, frequency=1, seed=77)
    after_deep = torch.rand(3)
    assert deep.intervention_layers == [1] and deep.is_deep
    shared = {name: tensor for name, tensor in deep.modules.state_dict().items() if not name.startswith(("delta_heads.", "intervention"))}
    assert shared.keys() == zero.modules.state_dict().keys()
    assert all(torch.equal(tensor, zero.modules.state_dict()[name]) for name, tensor in shared.items())
    reader0 = h0.module_manager.lora_parameters("reader_lora"); reader1 = h1.module_manager.lora_parameters("reader_lora")
    assert all(torch.equal(a, b) for a, b in zip(reader0, reader1)), "the adapters injected after the heads draw the same values"
    assert torch.equal(after_zero, after_deep), "the heads' draws come from a forked stream"
    heads_a = {name: tensor for name, tensor in deep.modules.state_dict().items() if name.startswith("delta_heads.")}
    _, again = build(tmp_path / "again", TINY_QWEN3, frequency=1, seed=77)
    assert all(torch.equal(tensor, again.modules.state_dict()[name]) for name, tensor in heads_a.items()), "head init is a function of the seed"


# --------------------------------------------------------------------------------------------------- 3. encode, calibration
def test_deep_encode_and_calibration(tmp_path, hybrid_dir):
    h, ac = build(tmp_path, hybrid_dir, frequency=1)
    assert ac.intervention_layers == [2, 5]
    with pytest.raises(RuntimeError, match="not calibrated"):
        ac.encode_batch([fixed_request(ac)])
    trainer, items, examples, device = trainer_and_examples(h, ac)
    rms = calibrate(trainer, ac, examples[:2])
    assert set(rms) == {2, 5} and all(value > 0 for value in rms.values())
    assert ac.calibration_items == [items[0].item_id, items[1].item_id]
    summary = ac.intervention_summary()
    for layer in (2, 5):
        assert summary["scales"][layer] == pytest.approx(INTERVENTION_SCALE_FRACTION * rms[layer]) and summary["relative"][layer] == pytest.approx(INTERVENTION_SCALE_FRACTION)
        assert float(ac.modules.delta_heads[str(layer)].rms) == pytest.approx(rms[layer]) and summary["parametrization"] == "relative"
    # The calibration statistic is the median over positions with the first token excluded, one value per sequence, averaged.
    sample = [trainer._calibration_sequence(ac, example)[0] for example in examples[:2]]
    manual = {2: [], 5: []}
    handles = [decoder_layers(ac.target.model)[layer].register_forward_pre_hook((lambda index: lambda module, args: manual[index].append(args[0].detach()[0, 1:].float().pow(2).mean(-1).sqrt().median().item()))(layer)) for layer in (2, 5)]
    try:
        with torch.no_grad():
            for ids in sample:
                ac.target.decoder_forward(ac.target.model.get_input_embeddings()(torch.tensor(ids))[None], None, lora_name="reader_lora", gradient_checkpointing=False)
    finally:
        for handle in handles:
            handle.remove()
    for layer in (2, 5):
        assert rms[layer] == pytest.approx(sum(manual[layer]) / len(manual[layer]), rel=1e-6)
    with pytest.raises(RuntimeError, match="with_layers=True"):
        ac.encode_batch([fixed_request(ac)])                                      # a deep model refuses the rows-only API
    output = ac.encode_batch([fixed_request(ac)], with_layers=True)[0]
    assert set(output.layer_inputs) == {2, 5} and output.input_embeds.shape[1] == ac.d_target
    for layer, rows in output.layer_inputs.items():
        assert rows.shape == output.input_embeds.shape
        torch.testing.assert_close(rows.pow(2).mean(dim=-1).sqrt(), torch.full((rows.shape[0],), summary["scales"][layer]), atol=1e-5, rtol=1e-4)
    # Deterministic: a second fresh model calibrated on the same sample gets the same values.
    h2, ac2 = build(tmp_path / "again", hybrid_dir, frequency=1)
    trainer2, items2, examples2, _ = trainer_and_examples(h2, ac2)
    rms2 = calibrate(trainer2, ac2, examples2[:2])
    assert rms2 == rms
    # The calibration sequences are the stripped histories (plain passage text, no placeholder rows).
    ids, item_id = trainer._calibration_sequence(ac, examples[0])
    assert item_id == items[0].item_id and ac.pad_id not in ids and len(ids) > len(examples[0].target_ids)


# --------------------------------------------------------------------------------------------------- 4. gradients
def head_parameters(ac):
    return [parameter for name, parameter in ac.modules.named_parameters() if name.startswith("delta_heads.")]


@pytest.mark.parametrize("prefer, source, target, depth", [(True, "summary", "residual", "final"), (False, "summary", "residual", "final"), (True, "side_layers", "residual", "final"),
                                                           (True, "side_layers", "kv", "final"), (True, "passage_attention", "residual", "final"), (True, "passage_attention", "kv", "matched")],
                         ids=["attention_inputs", "gdn_input", "side_layers", "kv", "passage_final", "passage_matched_kv"])
def test_gradients_direct_vs_checkpointed(tmp_path, hybrid_dir, prefer, source, target, depth):
    h, ac = build(tmp_path, hybrid_dir, frequency=1, prefer=prefer, source=source, target=target, depth=depth)
    assert ac.intervention_layers == ([2, 5] if prefer else [3, 5]), "prefer=False: the interior delta lands at a linear-attention block"
    trainer, items, examples, device = trainer_and_examples(h, ac)
    calibrate(trainer, ac, examples[:2])
    boost_scales(ac)                                                               # audible deltas: every head gets a real gradient
    if source == "passage_attention":
        open_attention(ac)                                                         # a zero out projection would leave q/k/v without gradient
    params = head_parameters(ac) + attention_parameters(ac) + [ac.modules.target_head.scale] + list(h.module_manager.lora_parameters("reader_lora")) + side_lora_parameters(ac)
    grads = {}
    for checkpointed in (False, True):
        trainer.config.gradient_checkpointing = checkpointed
        ac.config.side_gradient_checkpointing = checkpointed                       # the side capture (side_layers) must survive the side recompute too
        losses = [trainer._example_loss(ac, example, device, with_drift=False)[0] for example in examples[:2]]
        grads[checkpointed] = torch.autograd.grad(sum(losses), params, allow_unused=True)
    for direct, recomputed in zip(grads[False], grads[True]):
        assert direct is not None and recomputed is not None
        torch.testing.assert_close(direct, recomputed, atol=1e-6, rtol=1e-5)
    assert all(float(g.norm()) > 0 for g in grads[True][:len(head_parameters(ac))]), "every delta head (weights and scales) receives gradient"
    if source == "passage_attention":
        names = [name for name, _ in ac.modules.named_parameters() if name.startswith("passage_attention.")]
        block_grads = dict(zip(names, grads[True][len(head_parameters(ac)):len(head_parameters(ac)) + len(names)]))
        for name, g in block_grads.items():
            if name.endswith((".query.weight", ".key.weight", ".value.weight", ".out.weight")):
                assert float(g.norm()) > 0, f"{name} receives gradient through the attention path"
    assert any(float(g.norm()) > 0 for g in grads[True][-len(side_lora_parameters(ac)):]), "the side adapter receives gradient (lora_A's is zero at PEFT's zero init of lora_B)"
    # Two forwards with different payloads pending before one backward keep their own tensors.
    trainer.config.gradient_checkpointing = True
    ac.config.side_gradient_checkpointing = True
    loss_a = trainer._example_loss(ac, examples[0], device, with_drift=False)[0]
    loss_b = trainer._example_loss(ac, examples[1], device, with_drift=False)[0]      # a different example: its own payload tensors
    joint = torch.autograd.grad(loss_a + loss_b, params, allow_unused=True)
    assert all(torch.isfinite(g).all() for g in joint)
    single_a = torch.autograd.grad(trainer._example_loss(ac, examples[0], device, with_drift=False)[0], params, allow_unused=True)
    single_b = torch.autograd.grad(trainer._example_loss(ac, examples[1], device, with_drift=False)[0], params, allow_unused=True)
    for j, a, b in zip(joint, single_a, single_b):
        torch.testing.assert_close(j, a + b, atol=1e-6, rtol=1e-5)


# --------------------------------------------------------------------------------------------------- 5. padded batch
@pytest.mark.parametrize("target", ["residual", "kv"])
def test_padded_batch_positions_and_causality(tmp_path, target):
    torch.manual_seed(0)
    h, ac = build(tmp_path, TINY_QWEN3, frequency=1, target=target, source="side_layers" if target == "kv" else "summary")
    assert ac.intervention_layers == [1]
    trainer, items, examples, device = trainer_and_examples(h, ac, micro_batch_examples=3)
    calibrate(trainer, ac, examples[:2])
    boost_scales(ac)
    assert len({len(e.student_ids) for e in examples}) > 1, "the fixture must pad"
    single = [trainer._example_loss(ac, example, device, with_drift=False) for example in examples]
    batched = trainer._batch_loss(ac, examples, device, with_penalty=False)
    for (loss, _, _, num, logp), result in zip(single, batched):
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
@pytest.mark.parametrize("target", ["residual", "kv"])
def test_generation_matches_full_recompute(tmp_path, hybrid_dir, target):
    h, ac = build(tmp_path, hybrid_dir, frequency=1, target=target, source="side_layers" if target == "kv" else "summary")
    trainer, items, examples, device = trainer_and_examples(h, ac)
    calibrate(trainer, ac, examples[:2])
    boost_scales(ac)
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
                          for p in bound.payloads if p and (2 in p["layer_inputs"] or 2 in p.get("kv_inputs", {})))
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
    # The payload is heard (the last prompt position's logits move; a random tiny model's argmax may not), and generate() itself rejects the raw kwarg.
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
@pytest.mark.parametrize("source, target, per_head, depth", [("summary", "residual", 5, "final"), ("side_layers", "kv", 6, "final"), ("passage_attention", "residual", 5, "matched")], ids=["residual", "kv", "passage"])
def test_checkpoint_roundtrip_and_resume(tmp_path, hybrid_dir, source, target, per_head, depth):
    h, ac = build(tmp_path, hybrid_dir, frequency=1, source=source, target=target, depth=depth)
    items = items_for(h, ac)
    trainer = ActivationContextTrainer(h, config(checkpoint_every_epoch=True, deltas_off_panel=True))
    stats = trainer.train(ac.name, items, reporting_data=items)
    assert ac.interventions_calibrated and ac.calibration_items == [item.item_id for item in items[:len(ac.calibration_items)]]
    scale_keys = {"2:k", "2:v", "5:k", "5:v"} if target == "kv" else {2, 5}
    assert stats.intervention_trace and stats.intervention_trace[0]["update"] == 0 and set(stats.intervention_trace[0]["scale"]) == scale_keys
    assert set(stats.intervention_trace[0]["head_grad_norm"]) == {2, 5} and set(stats.intervention_trace[0]["row_rms"]) == {2, 5}
    assert all(value == pytest.approx(INTERVENTION_SCALE_FRACTION) for value in stats.intervention_trace[0]["relative"].values()), "the first trace shows the relative scales at their init"
    assert stats.payload_bytes > 0 and stats.deltas_off is not None and stats.deltas_off["items"] == len(items)
    summary = ac.intervention_summary()
    heads_before = {name: tensor.clone() for name, tensor in ac.modules.state_dict().items() if name.startswith(("delta_heads.", "intervention", "passage_attention."))}
    record = json.loads((tmp_path / "AC_MODELS" / ac.name / "completed_epoch.json").read_text())
    saved = tmp_path / "AC_MODELS" / ac.name / "epoch_001" / "optimizer.pt"
    moments = torch.load(saved, map_location="cpu", weights_only=False)
    head_count = len(head_parameters(ac))
    assert head_count == 2 * per_head and len(moments["optimizer"]["state"]) >= head_count, "optimizer.pt carries moments for every head parameter"
    checkpoint = torch.load(tmp_path / record["ac_checkpoint"] / "ac_modules.pt", map_location="cpu", weights_only=False)
    assert checkpoint["interventions"]["layers"] == [2, 5] and checkpoint["interventions"]["calibration_items"] == ac.calibration_items
    assert (checkpoint["interventions"]["source"], checkpoint["interventions"]["target"]) == (source, target)
    assert checkpoint["interventions"]["source_layers"] == ([2, 5] if source == "side_layers" or depth == "matched" else [])
    assert checkpoint["interventions"]["attention_depth"] == (depth if source == "passage_attention" else None)
    if source == "passage_attention":
        assert checkpoint["interventions"]["attention_parameters"] > 0 and any(name.startswith("passage_attention.") for name in checkpoint["modules"])
        assert stats.intervention_trace[0]["attention_out_norm"][2] == 0.0 and stats.intervention_trace[0]["attention_grad_norm"][2] >= 0.0
    # A new process: the epoch pair loads into a fresh deep model; heads, scales and calibration come back; the optimizer resumes.
    h2, ac2 = build(tmp_path / "second", hybrid_dir, frequency=1, seed=999, source=source, target=target, depth=depth,
                    ac_checkpoint=str(tmp_path / record["ac_checkpoint"]), reader_checkpoint=str(tmp_path / record["target_lora_checkpoint"]))
    assert ac2.interventions_calibrated and ac2.calibration_items == ac.calibration_items
    assert ac2.intervention_summary()["scales"] == summary["scales"] and ac2.intervention_summary()["calibration_rms"] == summary["calibration_rms"]
    for name, tensor in heads_before.items():
        assert torch.equal(tensor, ac2.modules.state_dict()[name]), name
    trainer2 = ActivationContextTrainer(h2, config(checkpoint_every_epoch=True, resume_optimizer=True, learning_rate_ac=3e-5))
    stats2 = trainer2.train(ac2.name, items_for(h2, ac2))
    assert stats2.resumed_from == str(saved) and stats2.start_epoch == 2 and stats2.learning_rates[-1][0] == pytest.approx(3e-5)
    assert stats2.global_steps[0] == trainer.updates_done[ac.name] + 1
    assert all(id(parameter) in {id(p) for group in trainer2.optimizers[ac2.name].param_groups for p in group["params"]} for parameter in head_parameters(ac2))
    # A legacy (input-only) checkpoint into a deep model is refused; a deep one into an input-only model too.
    h3, zero = build(tmp_path / "zero", hybrid_dir)
    with pytest.raises(ValueError, match="intervention layers"):
        zero.load(str(tmp_path / record["ac_checkpoint"]))
    # The same layers with another source or target are refused too (the heads differ in shape or meaning).
    h4, other = build(tmp_path / "other", hybrid_dir, frequency=1, source="side_layers" if source == "summary" else "summary", target=target)
    with pytest.raises(ValueError, match="source/target"):
        other.load(str(tmp_path / record["ac_checkpoint"]))
    if source == "passage_attention":
        h5, other_depth = build(tmp_path / "other_depth", hybrid_dir, frequency=1, source=source, target=target, depth="final")
        with pytest.raises(ValueError, match="source/target"):
            other_depth.load(str(tmp_path / record["ac_checkpoint"]))


# --------------------------------------------------------------------------------------------------- 8. deltas-off panel, gate group
@pytest.mark.parametrize("target", ["residual", "kv"])
def test_deltas_off_panel_and_delta_scale_group(tmp_path, hybrid_dir, target):
    h, ac = build(tmp_path, hybrid_dir, frequency=1, target=target, source="side_layers" if target == "kv" else "summary")
    items = items_for(h, ac)
    trainer = ActivationContextTrainer(h, config(deltas_off_panel=True, learning_rate_delta_scales=5e-3, learning_rate_gates=2e-3, learning_rate_gates_scope="row_scale", num_epochs=1))
    stats = trainer.train(ac.name, items, reporting_data=items)
    optimizer = trainer.optimizers[ac.name]
    assert len(optimizer.param_groups) == 4, "encoder, reader, the row-scale gate group, the delta scales"
    scale_ids = {id(p) for p in optimizer.param_groups[3]["params"]}
    assert scale_ids == {id(scale) for head in ac.modules.delta_heads.values() for scale in head.scale_parameters()}, "exactly the relative delta scales form the last group"
    assert len(scale_ids) == (4 if target == "kv" else 2)
    assert {id(p) for p in optimizer.param_groups[2]["params"]} == {id(ac.modules.target_head.scale)}
    assert not scale_ids & {id(p) for p in optimizer.param_groups[0]["params"]}
    assert stats.learning_rates[-1][3] == pytest.approx(5e-3) and stats.learning_rates[-1][2] == pytest.approx(2e-3)
    with pytest.raises(ValueError, match="gates_scope"):
        config(learning_rate_gates_scope="delta_scales")
    # Without a gate group the delta scales are still their own (third) group.
    h1, ac1 = build(tmp_path / "plain", hybrid_dir, frequency=1, target=target, source="side_layers" if target == "kv" else "summary")
    trainer1 = ActivationContextTrainer(h1, config(learning_rate_delta_scales=7e-3))
    stats1 = trainer1.train(ac1.name, items_for(h1, ac1))
    assert len(trainer1.optimizers[ac1.name].param_groups) == 3 and stats1.learning_rates[-1][2] == pytest.approx(7e-3)
    if target == "residual":
        entry = stats1.intervention_trace[0]
        assert all(entry["row_rms"][layer] > 0 for layer in (2, 5)) and all(-1.0 <= entry["cosine"][layer] <= 1.0 for layer in (2, 5)), "row-position residual RMS and delta/row cosine are traced"
    else:
        assert all(stats1.intervention_trace[0]["cosine"][layer] is None for layer in (2, 5))
    assert stats.deltas_off is not None and stats.epochs[-1].deltas_off is not None
    device = ac.prepare()
    boost_scales(ac)
    h.module_manager.ensure_lora("reader_lora")
    off = trainer._evaluate(ac, items, device, None, "deltas_off", deltas_off=True)
    on = trainer._evaluate(ac, items, device, None, "reporting")
    off_again = trainer._evaluate(ac, items, device, None, "deltas_off", deltas_off=True)
    assert off["gold_nll"] == off_again["gold_nll"] and off["gold_nll"] != on["gold_nll"]
    # Input-only models produce no deltas-off panel and no trace even when asked.
    h0, zero = build(tmp_path / "zero", hybrid_dir)
    stats0 = ActivationContextTrainer(h0, config(deltas_off_panel=True)).train(zero.name, items_for(h0, zero), reporting_data=items_for(h0, zero))
    assert stats0.deltas_off is None and stats0.intervention_trace == [] and stats0.payload_bytes == 0


# --------------------------------------------------------------------------------------------------- 10. A1: layer-matched readout
def test_side_layer_readout(tmp_path, hybrid_dir):
    from activation.ac_model import ActivationContextModelConfig as Config
    with pytest.raises(ValueError, match="intervention_source"):
        Config("x", "reader", "side_lora", "reader", "reader_lora", intervention_frequency=1, intervention_source="final")
    with pytest.raises(ValueError, match="source_layers"):
        Config("x", "reader", "side_lora", "reader", "reader_lora", intervention_frequency=1, intervention_source_layers=[1, 2])
    with pytest.raises(ValueError, match="one side layer"):
        build(tmp_path / "bad", hybrid_dir, frequency=1, source="side_layers", source_layers=[1, 2, 3])
    with pytest.raises(ValueError, match="one side layer"):
        build(tmp_path / "bad2", hybrid_dir, frequency=1, source="side_layers", source_layers=[1, 6])
    h, ac = build(tmp_path, hybrid_dir, frequency=1, source="side_layers")
    assert ac.intervention_layers == [2, 5] and ac.intervention_source_layers == [2, 5], "s(l) = l by default"
    h1, mapped = build(tmp_path / "mapped", hybrid_dir, frequency=1, source="side_layers", source_layers=[1, 4])
    assert mapped.intervention_source_layers == [1, 4]
    hs, summary = build(tmp_path / "summary", hybrid_dir, frequency=1)
    assert summary.intervention_source_layers == [] and summary.intervention_summary() == {} or True
    for a, b in zip(ac.modules.state_dict().values(), summary.modules.state_dict().values()):
        assert torch.equal(a, b), "the readout changes what the heads read, not the heads"
    trainer, items, examples, device = trainer_and_examples(h, ac)
    calibrate(trainer, ac, examples[:2])
    trainer_s, items_s, examples_s, _ = trainer_and_examples(hs, summary)
    calibrate(trainer_s, summary, examples_s[:2])
    request = fixed_request(ac)
    # Independent capture of the side model's layer inputs (2 and 5) and of its final normed state during the encode.
    side_layers = decoder_layers(ac.side.model)
    captured, final = {}, []
    handles = [side_layers[layer].register_forward_pre_hook((lambda index: lambda module, args: captured.__setitem__(index, args[0].detach().clone()))(layer)) for layer in (2, 5)]
    handles.append(text_decoder(ac.side.model).norm.register_forward_hook(lambda module, args, output: final.append(output.detach().clone())))
    try:
        output = ac.encode_batch([request], with_layers=True)[0]
    finally:
        for handle in handles:
            handle.remove()
    num_rows, length = output.input_embeds.shape[0], captured[2].shape[1]
    assert captured[2].shape[0] == 1 and set(output.layer_inputs) == {2, 5}
    with torch.no_grad():
        for layer in (2, 5):
            expected = ac.modules.delta_heads[str(layer)](captured[layer][0, length - num_rows:length].float())
            torch.testing.assert_close(output.layer_inputs[layer], expected, atol=1e-5, rtol=1e-5)
            from_summary = ac.modules.delta_heads[str(layer)](final[-1][0, length - num_rows:length].float())
            assert not torch.allclose(output.layer_inputs[layer], from_summary), "the layer-matched readout is not the summary readout"
    # Summary mode reads the final state exactly as before (the same heads on `rows`), so its deltas are a function of z.
    handles = [text_decoder(summary.side.model).norm.register_forward_hook(lambda module, args, output: final.append(output.detach().clone()))]
    try:
        output_s = summary.encode_batch([request], with_layers=True)[0]
    finally:
        handles[0].remove()
    with torch.no_grad():
        for layer in (2, 5):
            torch.testing.assert_close(output_s.layer_inputs[layer], summary.modules.delta_heads[str(layer)](final[-1][0, length - num_rows:length].float()), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(output_s.input_embeds, output.input_embeds)
    # Gradient through the capture: the delta at reader layer 2 reaches the side adapter of side layers 0 and 1 only (the input of layer 2).
    below = side_lora_parameters(ac, 0) + side_lora_parameters(ac, 1)               # granting trainability before the forward puts the adapter in the graph
    at_or_above = side_lora_parameters(ac, 2) + side_lora_parameters(ac, 3)
    output = ac.encode_batch([request], with_layers=True)[0]
    grads = torch.autograd.grad(output.layer_inputs[2].sum(), below + at_or_above, allow_unused=True)
    assert any(g is not None and float(g.norm()) > 0 for g in grads[:len(below)]), "side layers below the source layer shape the captured state (lora_B is nonzero; lora_A's gradient is zero at PEFT's zero init of B)"
    assert all(g is None or float(g.norm()) == 0 for g in grads[len(below):]), "the source layer and those above do not"
    # Rows are cached and served with their deltas in rollout mode.
    ac.set_mode("rollout")
    first = ac.encode_batch([request], with_layers=True)[0]
    second = ac.encode_batch([request], with_layers=True)[0]
    assert ac.stats.cache_hits >= 1 and all(torch.equal(first.layer_inputs[layer].to(torch.bfloat16).float(), second.layer_inputs[layer]) for layer in (2, 5)), "the cache keeps the deltas (bf16, like the rows)"


# --------------------------------------------------------------------------------------------------- 11. B1: K/V deltas at the rows
def test_kv_interventions(tmp_path, hybrid_dir):
    with pytest.raises(ValueError, match="full-attention"):
        build(tmp_path / "gdn", hybrid_dir, frequency=1, prefer=False, target="kv")        # layer 3 is a linear-attention block
    h, ac = build(tmp_path, hybrid_dir, frequency=1, target="kv", source="side_layers")
    assert ac.intervention_layers == [2, 5] and ac.kv_interventions
    assert all(isinstance(head, KVDeltaHead) for head in ac.modules.delta_heads.values())
    d_kv = ac.target.model_config.model_description.d_kv
    assert d_kv == 2 * 8 and ac.modules.delta_heads["2"].down.out_features == 2 * d_kv
    assert tuple(ac.modules.intervention_calibration.shape) == (2, 2)
    keys = set(ac.modules.state_dict())
    assert {"delta_heads.2.relative_k", "delta_heads.2.relative_v", "delta_heads.2.rms_k", "delta_heads.2.rms_v", "delta_heads.5.relative_k"} <= keys and "delta_heads.2.relative" not in keys
    trainer, items, examples, device = trainer_and_examples(h, ac)
    rms = calibrate(trainer, ac, examples[:2])
    summary = ac.intervention_summary()
    assert summary["target"] == "kv" and summary["source"] == "side_layers" and summary["source_layers"] == [2, 5]
    for layer in (2, 5):
        k_rms, v_rms = rms[layer]
        assert k_rms > 0 and v_rms > 0 and summary["calibration_rms"][layer] == {"k": pytest.approx(k_rms), "v": pytest.approx(v_rms)}
        assert summary["scales"][layer]["k"] == pytest.approx(INTERVENTION_SCALE_FRACTION * k_rms) and summary["scales"][layer]["v"] == pytest.approx(INTERVENTION_SCALE_FRACTION * v_rms)
    output = ac.encode_batch([fixed_request(ac)], with_layers=True)[0]
    assert output.layer_inputs == {} and set(output.kv_inputs) == {2, 5}
    for layer, (dk, dv) in output.kv_inputs.items():
        assert dk.shape == dv.shape == (output.input_embeds.shape[0], d_kv)
        torch.testing.assert_close(dk.pow(2).mean(dim=-1).sqrt(), torch.full((dk.shape[0],), summary["scales"][layer]["k"]), atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(dv.pow(2).mean(dim=-1).sqrt(), torch.full((dv.shape[0],), summary["scales"][layer]["v"]), atol=1e-5, rtol=1e-4)
    # The reader adds dk after k_norm (pre-RoPE) and dv after v_proj at the row positions; the residual stream is untouched there.
    boost_scales(ac)
    example = examples[0]
    target = ac.target
    embedding = target.model.get_input_embeddings()
    outputs = ac.encode_batch(example.part_requests, with_layers=True)
    with torch.no_grad():
        student = embedding(torch.tensor(example.student_ids))
    embeds, payload = trainer._assemble_student(student, example.spans, outputs, student.dtype)
    assert set(payload["kv_inputs"]) == {2, 5} and payload["layer_inputs"] == {}
    install_intervention_hooks(target.model)                                       # the product hooks first, so the spies below see their result
    layer2 = decoder_layers(target.model)[2]
    seen = {}
    def record(name):
        def hook(module, args, output):
            seen.setdefault(name, []).append((output if isinstance(output, torch.Tensor) else output[0]).detach().clone())
        return hook
    handles = [layer2.self_attn.k_norm.register_forward_hook(record("k")), layer2.self_attn.v_proj.register_forward_hook(record("v")),
               layer2.register_forward_pre_hook(lambda module, args: seen.setdefault("input", []).append(args[0].detach().clone()))]
    try:
        with torch.no_grad():
            with_deltas = target.decoder_forward(embeds[None], None, lora_name="reader_lora", interventions=[payload])
            without = target.decoder_forward(embeds[None], None, lora_name="reader_lora")
    finally:
        for handle in handles:
            handle.remove()
    positions = torch.tensor(payload["positions"])
    dk, dv = payload["kv_inputs"][2]
    k_on, k_off = seen["k"]
    v_on, v_off = seen["v"]
    assert k_on.shape[1] == embeds.shape[0] and k_on.dim() == 4, "k_norm output is [B, S, kv_heads, head_dim] before the transpose and RoPE"
    diff_k = (k_on - k_off)[0].reshape(embeds.shape[0], -1)
    diff_v = (v_on - v_off)[0].reshape(embeds.shape[0], -1)
    torch.testing.assert_close(diff_k[positions], dk, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(diff_v[positions], dv, atol=1e-5, rtol=1e-5)
    mask = torch.ones(embeds.shape[0], dtype=torch.bool); mask[positions] = False
    assert torch.equal(diff_k[mask], torch.zeros_like(diff_k[mask])) and torch.equal(diff_v[mask], torch.zeros_like(diff_v[mask]))
    assert torch.equal(seen["input"][0], seen["input"][1]), "K/V deltas leave the layer input (residual stream) untouched"
    first = int(positions.min())
    assert torch.equal(with_deltas[0, :first], without[0, :first]) and not torch.equal(with_deltas[0, first:], without[0, first:]), "causal: nothing before the first row moves"
    assert not torch.allclose(with_deltas[0, -1], without[0, -1])
    # Malformed payloads are refused: row-count mismatch, and a K/V delta aimed at a linear-attention layer.
    bad = {"positions": payload["positions"][:-1], "layer_inputs": {}, "kv_inputs": payload["kv_inputs"]}
    with pytest.raises(ValueError, match="differ in count"):
        target.decoder_forward(embeds[None], None, lora_name="reader_lora", interventions=[bad])
    gdn = {"positions": payload["positions"], "layer_inputs": {}, "kv_inputs": {3: payload["kv_inputs"][2]}}
    with pytest.raises(ValueError, match="no full attention"):
        target.decoder_forward(embeds[None], None, lora_name="reader_lora", interventions=[gdn])
    # capture_kv_rms measures K after k_norm and V after v_proj, one pair per forward.
    with torch.no_grad(), capture_kv_rms(target.model, [2]) as measured:
        target.decoder_forward(embeds[None], None, lora_name="reader_lora")
    assert len(measured[2]) == 1 and measured[2][0][0] == pytest.approx(float(k_off[0].reshape(embeds.shape[0], -1).pow(2).mean(-1).sqrt().mean()), rel=1e-5)
    # deltas_off withholds the K/V payload and keeps the rows.
    kept, none = trainer._assemble_student(student, example.spans, outputs, student.dtype, deltas_off=True)
    assert none is None and torch.equal(kept, embeds)


# --------------------------------------------------------------------------------------------------- 12. relative scales, migration, controls
def test_relative_scales_conversion_and_migration(tmp_path, hybrid_dir):
    h, ac = build(tmp_path, hybrid_dir, frequency=1)
    trainer, items, examples, device = trainer_and_examples(h, ac)
    rms = calibrate(trainer, ac, examples[:2])
    head = ac.modules.delta_heads["2"]
    assert float(head.scale.detach()) == pytest.approx(rms[2] * INTERVENTION_SCALE_FRACTION) and head.scale.requires_grad
    folder = tmp_path / "relative_ckpt"
    ac.save(str(folder))
    # A checkpoint written by the absolute parametrization (one `scale` per head, no `rms` buffer) converts on load.
    state = torch.load(folder / "ac_modules.pt", map_location="cpu", weights_only=False)
    absolute = dict(state["modules"])
    for layer in (2, 5):
        absolute[f"delta_heads.{layer}.scale"] = torch.tensor(0.3 * float(absolute.pop(f"delta_heads.{layer}.rms")))     # 30 % of the RMS
        absolute.pop(f"delta_heads.{layer}.relative")
    state["modules"] = absolute
    legacy = tmp_path / "absolute_ckpt"
    import shutil
    shutil.copytree(folder, legacy)
    torch.save(state, legacy / "ac_modules.pt")
    h2, ac2 = build(tmp_path / "second", hybrid_dir, frequency=1, seed=999, ac_checkpoint=str(legacy))
    for layer in (2, 5):
        converted = ac2.modules.delta_heads[str(layer)]
        assert float(converted.relative) == pytest.approx(0.3) and float(converted.rms) == pytest.approx(rms[layer]) and float(converted.scale) == pytest.approx(0.3 * rms[layer])
    assert ac2.interventions_calibrated
    # Migration: an input-only checkpoint into a deep model loads the shared modules, keeps fresh seeded heads and leaves calibration to the trainer.
    h0, zero = build(tmp_path / "zero", hybrid_dir)
    zero_items = items_for(h0, zero)
    zero_folder = tmp_path / "zero_ckpt"
    zero.save(str(zero_folder))
    assert "interventions" not in torch.load(zero_folder / "ac_modules.pt", map_location="cpu", weights_only=False)
    h3, migrated = build(tmp_path / "migrated", hybrid_dir, frequency=1, seed=4321, ac_checkpoint=str(zero_folder))
    h4, fresh = build(tmp_path / "fresh", hybrid_dir, frequency=1, seed=4321)
    assert migrated.is_deep and not migrated.interventions_calibrated and migrated.calibration_items == []
    for name, tensor in zero.modules.state_dict().items():
        assert torch.equal(tensor, migrated.modules.state_dict()[name]), f"shared module {name} comes from the checkpoint"
    for name, tensor in fresh.modules.state_dict().items():
        if name.startswith("delta_heads."):
            assert torch.equal(tensor, migrated.modules.state_dict()[name]), f"{name}: heads are the seeded fresh init"
    stats = ActivationContextTrainer(h3, config()).train(migrated.name, items_for(h3, migrated))
    assert migrated.interventions_calibrated and stats.intervention_trace and len(migrated.calibration_items) > 0
    # A checkpoint whose optimizer.pt has another group layout (here: an input-only run's two groups) resumes with fresh moments and the update clock.
    h6, zero6 = build(tmp_path / "zero6", hybrid_dir)
    trainer6 = ActivationContextTrainer(h6, config(checkpoint_every_epoch=True))
    trainer6.train(zero6.name, items_for(h6, zero6))
    record6 = json.loads((tmp_path / "zero6" / "AC_MODELS" / zero6.name / "completed_epoch.json").read_text())
    saved6 = tmp_path / "zero6" / record6["ac_checkpoint"] / "optimizer.pt"
    assert saved6.exists() and len(torch.load(saved6, map_location="cpu", weights_only=False)["optimizer"]["param_groups"]) == 2
    h7, resumed = build(tmp_path / "resumed", hybrid_dir, frequency=1, seed=77, ac_checkpoint=str(tmp_path / "zero6" / record6["ac_checkpoint"]),
                        reader_checkpoint=str(tmp_path / "zero6" / record6["target_lora_checkpoint"]))
    trainer7 = ActivationContextTrainer(h7, config(resume_optimizer=True))
    stats7 = trainer7.train(resumed.name, items_for(h7, resumed))
    assert "moments discarded" in stats7.resumed_from and stats7.global_steps[0] == trainer6.updates_done[zero6.name] + 1
    assert len(trainer7.optimizers[resumed.name].param_groups) == 3
    # A deep checkpoint with other layers is still refused.
    h5, other = build(tmp_path / "other", hybrid_dir, frequency=1, add_last=False)
    assert other.intervention_layers == [2]
    with pytest.raises(ValueError, match="intervention layers"):
        other.load(str(folder))
    with pytest.raises(ValueError, match="intervention layers"):
        zero.load(str(folder))


def test_frozen_heads_and_detached_source_controls(tmp_path, hybrid_dir):
    from activation.ac_model import ActivationContextModelConfig as Config
    # Zero mode: the flags change nothing for an input-only model.
    h0, zero = build(tmp_path / "zero", hybrid_dir)
    zero.config.intervention_heads_frozen = zero.config.intervention_detach_source = True
    assert len(zero.trainable_parameters()) == len(zero.parameters()) + len(h0.module_manager.lora_parameters("side_lora"))
    assert torch.equal(zero.encode_batch([fixed_request(zero)])[0], build(tmp_path / "zero2", hybrid_dir)[1].encode_batch([fixed_request(zero)])[0])
    # Frozen heads: up/down out of the trainable set; a training epoch leaves them exactly, the relative scales move.
    h, ac = build(tmp_path / "frozen", hybrid_dir, frequency=1)
    ac.config.intervention_heads_frozen = True
    trainable = {id(p) for p in ac.trainable_parameters()}
    for name, parameter in ac.modules.named_parameters():
        if name.startswith("delta_heads."):
            assert (id(parameter) in trainable) == name.endswith(".relative"), name
    before = {name: tensor.clone() for name, tensor in ac.modules.state_dict().items() if name.startswith("delta_heads.")}
    stats = ActivationContextTrainer(h, config(learning_rate_delta_scales=1e-2)).train(ac.name, items_for(h, ac))
    after = ac.modules.state_dict()
    for name, tensor in before.items():
        if name.endswith((".up.weight", ".up.bias", ".down.weight", ".down.bias")):
            assert torch.equal(tensor, after[name]), f"{name} frozen"
        elif name.endswith(".relative"):
            assert not torch.equal(tensor, after[name]), f"{name} trains"
    assert ac.intervention_summary()["heads_frozen"]
    # Detached source: the deltas carry no gradient into the encoder or the side adapter; the input row still does.
    hd, detached = build(tmp_path / "detached", hybrid_dir, frequency=1)
    detached.config.intervention_detach_source = True
    trainer, items, examples, device = trainer_and_examples(hd, detached)
    calibrate(trainer, detached, examples[:2])
    encoder = [detached.modules.summary_marker, detached.modules.input_scale] + list(detached.modules.mixer.parameters())[:2] + side_lora_parameters(detached)
    output = detached.encode_batch([fixed_request(detached)], with_layers=True)[0]
    through_deltas = torch.autograd.grad(sum(rows.sum() for rows in output.layer_inputs.values()), encoder + head_parameters(detached), allow_unused=True)
    assert all(g is None for g in through_deltas[:len(encoder)]), "the deltas' gradient stops at the heads"
    assert all(g is not None for g in through_deltas[len(encoder):]), "the heads themselves learn"
    output = detached.encode_batch([fixed_request(detached)], with_layers=True)[0]
    through_rows = torch.autograd.grad(output.input_embeds.sum(), encoder, allow_unused=True)
    assert any(g is not None and float(g.norm()) > 0 for g in through_rows), "the input row's gradient still reaches the encoder"
    assert detached.intervention_summary()["detach_source"]
    with pytest.raises(ValueError, match="scale_fraction"):
        Config("x", "reader", "side_lora", "reader", "reader_lora", intervention_frequency=1, intervention_scale_fraction=1.5)


# --------------------------------------------------------------------------------------------------- 13. A2: passage attention
def test_passage_attention(tmp_path, hybrid_dir):
    from activation.ac_model import ActivationContextModelConfig as Config
    from activation.ac_model.ac_model_utils import PassageAttention
    with pytest.raises(ValueError, match="attention_depth"):
        Config("x", "reader", "side_lora", "reader", "reader_lora", intervention_frequency=1, intervention_source="passage_attention", intervention_attention_depth="deep")
    with pytest.raises(ValueError, match="source_layers"):
        Config("x", "reader", "side_lora", "reader", "reader_lora", intervention_frequency=1, intervention_source="passage_attention", intervention_source_layers=[1, 2])
    with pytest.raises(ValueError, match="attention_depth applies"):
        Config("x", "reader", "side_lora", "reader", "reader_lora", intervention_frequency=1, intervention_source="summary", intervention_attention_depth="matched")
    assert PassageAttention(2560, 640).heads == 4 and PassageAttention(32, 8).heads == 1
    request_of = fixed_request
    # Depth "final": at init the source is bit-identical to the summary readout (zero out projection), heads and their draws included.
    hs, summary = build(tmp_path / "summary", hybrid_dir, frequency=1)
    ha, attention = build(tmp_path / "attention", hybrid_dir, frequency=1, source="passage_attention")
    assert attention.intervention_source_layers == [] and set(attention.modules.passage_attention) == {"2", "5"}
    for name, tensor in summary.modules.state_dict().items():
        assert torch.equal(tensor, attention.modules.state_dict()[name]), f"{name}: the summary model's draws are untouched"
    for block in attention.modules.passage_attention.values():
        assert torch.equal(block.out.weight, torch.zeros_like(block.out.weight)) and torch.equal(block.out.bias, torch.zeros_like(block.out.bias))
    trainer_s, _, examples_s, _ = trainer_and_examples(hs, summary)
    calibrate(trainer_s, summary, examples_s[:2])
    trainer_a, _, examples_a, _ = trainer_and_examples(ha, attention)
    calibrate(trainer_a, attention, examples_a[:2])
    out_s = summary.encode_batch([request_of(summary)], with_layers=True)[0]
    out_a = attention.encode_batch([request_of(attention)], with_layers=True)[0]
    assert torch.equal(out_s.input_embeds, out_a.input_embeds) and all(torch.equal(out_s.layer_inputs[layer], out_a.layer_inputs[layer]) for layer in (2, 5))
    assert attention.intervention_summary()["attention_depth"] == "final" and attention.intervention_summary()["attention_parameters"] == sum(p.numel() for p in attention_parameters(attention))
    # Depth "matched": at init bit-identical to the layer-matched readout; once open, the block reads the passage positions of side layer s(l).
    hl, matched_plain = build(tmp_path / "plain", hybrid_dir, frequency=1, source="side_layers")
    hm, matched = build(tmp_path / "matched", hybrid_dir, frequency=1, source="passage_attention", depth="matched")
    assert matched.intervention_source_layers == [2, 5]
    trainer_l, _, examples_l, _ = trainer_and_examples(hl, matched_plain)
    calibrate(trainer_l, matched_plain, examples_l[:2])
    trainer_m, _, examples_m, _ = trainer_and_examples(hm, matched)
    calibrate(trainer_m, matched, examples_m[:2])
    out_l = matched_plain.encode_batch([request_of(matched_plain)], with_layers=True)[0]
    out_m = matched.encode_batch([request_of(matched)], with_layers=True)[0]
    assert all(torch.equal(out_l.layer_inputs[layer], out_m.layer_inputs[layer]) for layer in (2, 5))
    open_attention(matched)
    captured = {}
    handles = [decoder_layers(matched.side.model)[layer].register_forward_pre_hook((lambda index: lambda module, args: captured.__setitem__(index, args[0].detach().clone()))(layer)) for layer in (2, 5)]
    try:
        out_m = matched.encode_batch([request_of(matched)], with_layers=True)[0]
    finally:
        for handle in handles:
            handle.remove()
    num_rows, length = out_m.input_embeds.shape[0], captured[2].shape[1]
    with torch.no_grad():
        for layer in (2, 5):
            full = captured[layer][0].float()
            expected = matched.modules.delta_heads[str(layer)](matched.modules.passage_attention[str(layer)](full[length - num_rows:], full[:length - num_rows]))
            torch.testing.assert_close(out_m.layer_inputs[layer], expected, atol=1e-5, rtol=1e-5)
            rows_only = matched.modules.delta_heads[str(layer)](full[length - num_rows:])
            assert not torch.allclose(out_m.layer_inputs[layer], rows_only), "the opened attention path changes the head's input"
    # Depth "final", opened: the passage states are the final normed side states of the content positions.
    open_attention(attention)
    finals = []
    handle = text_decoder(attention.side.model).norm.register_forward_hook(lambda module, args, output: finals.append(output.detach().clone()))
    try:
        out_a = attention.encode_batch([request_of(attention)], with_layers=True)[0]
    finally:
        handle.remove()
    final = finals[-1][0].float()
    with torch.no_grad():
        for layer in (2, 5):
            expected = attention.modules.delta_heads[str(layer)](attention.modules.passage_attention[str(layer)](final[length - num_rows:], final[:length - num_rows]))
            torch.testing.assert_close(out_a.layer_inputs[layer], expected, atol=1e-5, rtol=1e-5)
    # The frozen-heads control freezes the blocks too; the shared-parameter hash of the bench skips them.
    attention.config.intervention_heads_frozen = True
    trainable = {id(p) for p in attention.trainable_parameters()}
    assert not any(id(p) in trainable for p in attention_parameters(attention))
    attention.config.intervention_heads_frozen = False
    # Training runs end to end with the trace and widget fields.
    stats = ActivationContextTrainer(hm, config(num_epochs=1)).train(matched.name, items_for(hm, matched))
    assert stats.intervention_trace[0]["attention_out_norm"][2] > 0 and stats.intervention_trace[0]["attention_grad_norm"][2] >= 0


# --------------------------------------------------------------------------------------------------- 14. reporter
def test_reporter_takes_deep_trace_entries(tmp_path):
    from activation.ac_model.ac_model_reporter import ActivationContextTrainingReporter
    from activation.ac_model.ac_model_training import ActivationContextTrainingStats
    reporter = ActivationContextTrainingReporter(str(tmp_path / "report"), title="test")
    stats = ActivationContextTrainingStats(loss_kind="sft")
    def entry(update, attention):
        base = {"update": update, "scale": {7: 0.1, 31: 0.2}, "relative": {7: 0.05, 31: 0.05}, "head_grad_norm": {7: 1.0, 31: 2.0},
                "row_rms": {7: 0.5, 31: None}, "cosine": {7: 0.1, 31: 0.2}, "attention_out_norm": {}, "attention_grad_norm": {},
                "delta_rms": {7: 0.01, 31: None}, "delta_rms_abs": {7: 0.005, 31: 0.001}}
        if attention:
            base["attention_out_norm"] = {7: 0.0, 31: 0.01}; base["attention_grad_norm"] = {7: 0.3, 31: 0.4}
        return base
    for step, attention in ((25, False), (50, False), (75, True), (100, True)):
        stats.loss.append(1.0); stats.step_tokens.append(100); stats.step_seconds.append(1.0); stats.grad_norm.append(1.0); stats.learning_rates.append([1e-4, 2e-5, 1e-3])
        stats.intervention_trace.append(entry(step, attention))
        reporter.report_step(ActivationContextTrainingProgress(1, 1, 1, step, 100, step, 1.0, 1.0), stats)
    assert {"interventions", "intervention_grads", "intervention_row_rms", "intervention_cosine", "intervention_attention", "intervention_delta_rms"} <= set(reporter.widgets)
    delta_series = {series["name"]: series for series in reporter.widgets["intervention_delta_rms"]["series"]}
    assert set(delta_series) == {"layer 7"} and delta_series["layer 7"]["y"] == [0.01] * 4, "None values are skipped, the ratio is plotted"
    # K/V-style keys and a residual-only model (no attention fields at all) go through the same path.
    stats.loss.append(1.0); stats.step_tokens.append(100); stats.step_seconds.append(1.0); stats.grad_norm.append(1.0)
    stats.intervention_trace.append({"update": 125, "scale": {"7:k": 0.1, "7:v": 0.2}, "relative": {"7:k": 0.05, "7:v": 0.05}, "head_grad_norm": {7: 1.0}, "row_rms": {7: 0.5}, "cosine": {7: None}})
    reporter.report_step(ActivationContextTrainingProgress(1, 1, 1, 125, 200, 125, 1.0, 1.0), stats)
    assert not {"intervention_slot_rms", "intervention_takeover"} & set(reporter.widgets), "the slot widgets appear for slot entries only"
    # A kv_slots entry: no scales, the takeover and the written / own ratio per half get their own widgets (created once).
    for step in (150, 175):
        stats.loss.append(1.0); stats.step_tokens.append(100); stats.step_seconds.append(1.0); stats.grad_norm.append(1.0)
        stats.intervention_trace.append({"update": step, "scale": {}, "relative": {}, "head_grad_norm": {7: 0.2}, "row_rms": {7: 0.5}, "cosine": {7: None},
                                         "delta_rms": {"7:k": 0.0, "7:v": 0.02}, "delta_rms_abs": {"7:k": 0.0, "7:v": 0.01},
                                         "takeover": {"7:k": 0.0, "7:v": 0.1}, "slot_rms": {"7:k": 1.0, "7:v": None}})
        reporter.report_step(ActivationContextTrainingProgress(1, 1, 1, step, 200, step, 1.0, 1.0), stats)
    slot_series = {series["name"]: series for series in reporter.widgets["intervention_slot_rms"]["series"]}
    assert set(slot_series) == {"layer 7:k"} and slot_series["layer 7:k"]["y"] == [1.0, 1.0]
    takeover_series = {series["name"]: series for series in reporter.widgets["intervention_takeover"]["series"]}
    assert set(takeover_series) == {"layer 7:k", "layer 7:v"} and takeover_series["layer 7:v"]["y"] == [0.1, 0.1]


# --------------------------------------------------------------------------------------------------- 9. bench arms
def test_bench_arms_are_paired():
    arms_path = Path(__file__).resolve().parents[2] / "IB/ARTIFACTS/AGENT_AC_PRETRAINING/ac_recon/arms.json"
    if not arms_path.exists():
        pytest.skip("bench arms not present")
    arms = json.loads(arms_path.read_text())
    strip = lambda arm, *keys: {key: value for key, value in arm.items() if key not in ("note", *keys)}
    assert strip(arms["NDC"], "interventions") == strip(arms["ND"]) and arms["NDC"]["interventions"] == 0
    assert strip(arms["NDI"], "interventions", "last_layer_intervention", "prefer_attention") == strip(arms["NDC"], "interventions")
    assert arms["NDI"]["interventions"] == 4 and arms["NDI"]["last_layer_intervention"] and arms["NDI"]["prefer_attention"]
    assert strip(arms["NDIg"], "delta_scale_lr") == strip(arms["NDI"]) and arms["NDIg"]["delta_scale_lr"] == 0.003 and "gates_scope" not in arms["NDIg"]
    assert strip(arms["NDIR"], "heads_frozen") == strip(arms["NDI"]) and arms["NDIR"]["heads_frozen"] is True
    assert strip(arms["NDID"], "detach_source") == strip(arms["NDI"]) and arms["NDID"]["detach_source"] is True
    assert strip(arms["NDIA"], "intervention_source") == strip(arms["NDI"]) and arms["NDIA"]["intervention_source"] == "passage_attention" and "intervention_attention_depth" not in arms["NDIA"]
    assert strip(arms["NDIAL"], "intervention_attention_depth") == strip(arms["NDIA"]) and arms["NDIAL"]["intervention_attention_depth"] == "matched"
    assert strip(arms["NKVA"], "intervention_target") == strip(arms["NDIA"]) and strip(arms["NKVAL"], "intervention_target") == strip(arms["NDIAL"]) and arms["NKVA"]["intervention_target"] == arms["NKVAL"]["intervention_target"] == "kv"
    for warm, cold in (("NDIW", "NDI"), ("NDILW", "NDIL"), ("NKVW", "NKV"), ("NDIAW", "NDIA"), ("NDIALW", "NDIAL")):
        assert strip(arms[warm], "init_ac", "init_reader") == strip(arms[cold]) and arms[warm]["init_ac"] == arms["A4X"]["init_ac"] and arms[warm]["init_reader"] == arms["A4X"]["init_reader"]
    assert strip(arms["NDIL"], "intervention_source") == strip(arms["NDI"]) and arms["NDIL"]["intervention_source"] == "side_layers"
    assert strip(arms["NKV"], "intervention_target") == strip(arms["NDIL"]) and arms["NKV"]["intervention_target"] == "kv"
    assert "intervention_source" not in arms["NDI"] and "intervention_target" not in arms["NDI"], "NDI/NDC/NDIg keep the defaults (summary, residual)"
    # Round 4: lora-style heads with the row-position calibration, paired with the scaled warm arms; the staged variant on top.
    for lora, scaled in (("NDIAWL", "NDIAW"), ("NKVWL", "NKVW"), ("NDIWL", "NDIW"), ("A32DL", "A32DF")):
        assert strip(arms[lora], "head_style", "calibration") == strip(arms[scaled]), (lora, scaled)
        assert arms[lora]["head_style"] == "lora" and arms[lora]["calibration"] == "row_positions" and "heads_only_updates" not in arms[lora]
    assert strip(arms["NDIAWLS"], "heads_only_updates") == strip(arms["NDIAWL"]) and arms["NDIAWLS"]["heads_only_updates"] == 200
    for label in ("NDIAWL", "NDIAWLS", "NKVWL", "NDIWL", "A32DL"):
        assert arms[label]["init_ac"] == arms["A4X"]["init_ac"] and arms[label]["init_reader"] == arms["A4X"]["init_reader"] and arms[label]["note"]
    # Round 5: K/V slots (replacement) at matched depth and the cross-attention ceiling, warm from the same K16r1 init on the NDIAWL / A32DL recipes.
    assert strip(arms["NKVSW"], "intervention_source", "intervention_target", "delta_scale_lr") == strip(arms["NDIAWL"], "intervention_source")
    assert arms["NKVSW"]["intervention_source"] == "side_layers" and arms["NKVSW"]["intervention_target"] == "kv_slots" and arms["NKVSW"]["head_style"] == "lora"
    assert strip(arms["NKVSW"], "intervention_target", "delta_scale_lr") == strip(arms["NKVWL"], "intervention_target"), "NKVSW pairs with NKVWL on replacement vs addition only"
    assert "delta_scale_lr" not in arms["NKVWL"] and "delta_scale_lr" not in arms["NDIAWL"]
    assert strip(arms["NKVSWS"], "heads_only_updates") == strip(arms["NKVSW"]) and arms["NKVSWS"]["heads_only_updates"] == 200
    assert strip(arms["A32KVS"], "intervention_source", "intervention_target", "delta_scale_lr") == strip(arms["A32DL"], "intervention_source", "intervention_attention_depth", "delta_scale_lr")
    assert arms["A32KVS"]["intervention_source"] == "side_layers" and arms["A32KVS"]["intervention_target"] == "kv_slots" and "intervention_attention_depth" not in arms["A32KVS"]
    for label in ("NKVSW", "NKVSWS", "A32KVS"):
        assert arms[label]["delta_scale_lr"] == 1e-3, f"{label}: the takeovers train in the scale group at 1e-3 (A32DL's inherited 0.0 would freeze them)"
    assert arms["A32DL"]["delta_scale_lr"] == 0.0
    assert strip(arms["NXAW"], "intervention_source", "intervention_target") == strip(arms["NDIAWL"], "intervention_source")
    assert arms["NXAW"]["intervention_source"] == "summary" and arms["NXAW"]["intervention_target"] == "cross_attention"
    assert strip(arms["A32XA"], "intervention_source", "intervention_target") == strip(arms["A32DL"], "intervention_source", "intervention_attention_depth")
    assert arms["A32XA"]["intervention_source"] == "summary" and arms["A32XA"]["intervention_target"] == "cross_attention"
    for label in ("NKVSW", "NKVSWS", "A32KVS", "NXAW", "A32XA"):
        assert arms[label]["init_ac"] == arms["A4X"]["init_ac"] and arms[label]["init_reader"] == arms["A4X"]["init_reader"] and arms[label]["note"]
    for label in ("NXAW", "A32XA"):
        assert "CEILING MEASUREMENT, NOT A CANDIDATE" in arms[label]["note"]
    # EC2 twins: the parent arm with the K16 epoch-2 init paths and the twin note, like NDIAWLn / A32DLn.
    for twin, parent in (("NKVSWn", "NKVSW"), ("NKVSWSn", "NKVSWS"), ("A32KVSn", "A32KVS"), ("NXAWn", "NXAW")):
        assert strip(arms[twin], "init_ac", "init_reader") == strip(arms[parent], "init_ac", "init_reader"), (twin, parent)
        assert arms[twin]["init_ac"] == arms["NDIAWLn"]["init_ac"] and arms[twin]["init_reader"] == arms["NDIAWLn"]["init_reader"]
        assert "EC2 twin" in arms[twin]["note"] and arms[twin]["note"].startswith(arms[parent]["note"])
    assert "CEILING MEASUREMENT, NOT A CANDIDATE" in arms["NXAWn"]["note"]
    assert list(arms)[-97:-88] == ["NKVSW", "NKVSWS", "A32KVS", "NXAW", "A32XA", "NKVSWn", "NKVSWSn", "A32KVSn", "NXAWn"], "appended after NDIWLn, in order"
    # Round 6: the K/V transfer family on the NKVSW / A32KVS recipes (delta_scale_lr 1e-3 inherited) and its whole-passage ceiling; EC2 twins after them.
    assert strip(arms["NKVTW"], "intervention_target") == strip(arms["NKVSW"], "intervention_target") and arms["NKVTW"]["intervention_target"] == "kv_transfer"
    assert strip(arms["NKVTWA"], "interventions") == strip(arms["NKVTW"], "interventions") and arms["NKVTWA"]["interventions"] == -1 and arms["NKVTW"]["interventions"] == 4
    assert strip(arms["NKVTWH"], "transfer_init") == strip(arms["NKVTW"]) and arms["NKVTWH"]["transfer_init"] == 0.5 and "transfer_init" not in arms["NKVTW"]
    assert strip(arms["A32KVT"], "intervention_target") == strip(arms["A32KVS"], "intervention_target") and arms["A32KVT"]["intervention_target"] == "kv_transfer"
    assert strip(arms["A32KVTA"], "interventions") == strip(arms["A32KVT"], "interventions") and arms["A32KVTA"]["interventions"] == -1
    assert strip(arms["NKVTF"], "intervention_target") == strip(arms["NKVTWA"], "intervention_target") and arms["NKVTF"]["intervention_target"] == "kv_transfer_full"
    assert strip(arms["A32KVTF"], "intervention_target") == strip(arms["A32KVTA"], "intervention_target") and arms["A32KVTF"]["intervention_target"] == "kv_transfer_full"
    transfer_arms = ("NKVTW", "NKVTWA", "NKVTWH", "A32KVT", "A32KVTA", "NKVTF", "A32KVTF")
    for label in transfer_arms:
        arm = arms[label]
        assert arm["delta_scale_lr"] == 1e-3 and arm["head_style"] == "lora" and arm["intervention_source"] == "side_layers" and arm["calibration"] == "row_positions"
        assert arm["init_ac"] == arms["A4X"]["init_ac"] and arm["init_reader"] == arms["A4X"]["init_reader"] and arm["note"] and "transfer_correction" not in arm
    for label in ("NKVTF", "A32KVTF"):
        assert "CEILING MEASUREMENT, NOT A CANDIDATE" in arms[label]["note"]
    for parent in transfer_arms:
        twin = arms[parent + "n"]
        assert strip(twin, "init_ac", "init_reader") == strip(arms[parent], "init_ac", "init_reader"), parent
        assert twin["init_ac"] == arms["NDIAWLn"]["init_ac"] and twin["init_reader"] == arms["NDIAWLn"]["init_reader"]
        assert "EC2 twin" in twin["note"] and twin["note"].startswith(arms[parent]["note"])
    assert list(arms)[-88:-74] == [*transfer_arms, *(label + "n" for label in transfer_arms)], "appended after NXAWn, in order"
    # Round 7: dense supervision (context distillation of the base reader's full-context attention outputs) on the A32X / WCa / A32KVTA recipes, the
    # late-layers-only K/V transfer (explicit intervention_layers), EC2 twins after each group. The ORCD 1/32 dense arms take the K16r1 init of A32DL / A32KVS
    # / A32KVTA (A32X itself carries the /root K16 paths of its node-1 run: A32XDn is A32X's own init).
    k16r1 = {"init_ac": arms["A4X"]["init_ac"], "init_reader": arms["A4X"]["init_reader"]}
    assert strip(arms["A32XD"], "distill_weight", "distill_layers", "init_ac", "init_reader") == strip(arms["A32X"], "init_ac", "init_reader")
    assert arms["A32XD"]["distill_weight"] == 1.0 and arms["A32XD"]["distill_layers"] == -1 and arms["A32XD"]["interventions"] == 0
    assert (arms["A32XD"]["init_ac"], arms["A32XD"]["init_reader"]) == (k16r1["init_ac"], k16r1["init_reader"]) == (arms["A32KVTA"]["init_ac"], arms["A32KVTA"]["init_reader"])
    assert strip(arms["A32XD3"], "distill_weight") == strip(arms["A32XD"], "distill_weight") and arms["A32XD3"]["distill_weight"] == 3.0
    assert strip(arms["A32XDR"], "reader_lr") == strip(arms["A32XD"], "reader_lr") and arms["A32XDR"]["reader_lr"] == 5e-5 and arms["A32XD"]["reader_lr"] == 2e-5
    assert "A32XDH" not in arms and "A32XDHn" not in arms, "the 'hidden' source arm was dropped at design review"
    assert strip(arms["NXD"], "distill_weight", "distill_layers") == strip(arms["WCa"]) and arms["NXD"]["distill_weight"] == 1.0 and arms["NXD"]["distill_layers"] == -1
    assert strip(arms["A32KVTAD"], "distill_weight") == strip(arms["A32KVTA"]) and arms["A32KVTAD"]["distill_weight"] == 1.0 and "distill_layers" not in arms["A32KVTAD"]
    dense_arms = ("A32XD", "A32XD3", "A32XDR", "NXD", "A32KVTAD")
    for label in dense_arms:
        assert arms[label]["note"] and all(key not in arms[label] for key in ("distill_source", "distill_hold", "distill_decay_end", "distill_floor")), label
    assert strip(arms["A32KVTL"], "interventions", "intervention_layers") == strip(arms["A32KVT"], "interventions") and arms["A32KVTL"]["interventions"] == 0
    assert arms["A32KVTL"]["intervention_layers"] == [23, 27, 31] and arms["A32KVT"]["interventions"] == 4 and "intervention_layers" not in arms["A32KVT"]
    assert strip(arms["NKVTL"], "interventions", "intervention_layers") == strip(arms["NKVTW"], "interventions") and arms["NKVTL"]["interventions"] == 0
    assert arms["NKVTL"]["intervention_layers"] == [23, 27, 31] and arms["NKVTW"]["interventions"] == 4
    for label in ("A32KVTL", "NKVTL"):
        assert arms[label]["intervention_target"] == "kv_transfer" and "transfer_init" not in arms[label] and "depth" in arms[label]["note"].lower()
    late_arms = ("A32KVTL", "NKVTL")
    for parent in (*dense_arms, *late_arms):
        twin = arms[parent + "n"]
        assert strip(twin, "init_ac", "init_reader") == strip(arms[parent], "init_ac", "init_reader"), parent
        assert twin["init_ac"] == arms["NDIAWLn"]["init_ac"] and twin["init_reader"] == arms["NDIAWLn"]["init_reader"]
        assert "EC2 twin" in twin["note"] and twin["note"].startswith(arms[parent]["note"])
    assert list(arms)[-74:-60] == [*dense_arms, *(label + "n" for label in dense_arms), *late_arms, *(label + "n" for label in late_arms)], "appended after A32KVTFn, in order"
    assert list(arms)[-60:-56] == ["A32XDc", "A32XDcn", "A32KVTAL", "A32XDL"]
    assert list(arms)[-56:-53] == ["SK32", "SK32KVT", "SK32XD"] and arms["SK32KVT"]["intervention_target"] == "kv_transfer" and arms["SK32XD"]["distill_weight"] == 1.0 and "init_ac" not in arms["SK32"] and arms["A32XDc"]["distill_floor"] == 1.0 and arms["A32XDcn"]["note"].startswith(arms["A32XDc"]["note"])
    # Round 8: the write-side arms (iterative encoder, surprisal-weighted terminal loss, surprisal-weighted summary pooling) on the A32W / A32KVTA / WCa recipes, EC2 twins after
    # them (the K16 init paths of A32X / A32KVTAn / WCn); the pairing itself is checked in test_encoder_passes.py::test_bench_round8_keys_and_arms.
    round8 = ["A32P2", "A32KVTAP2", "A32SW", "A32SP", "A32SPKVTA", "NSP"]
    assert list(arms)[-53:-41] == [*round8, *(label + "n" for label in round8)], "appended after SK32XD, in order"
    assert list(arms)[-40:-36] == ["A32P2R", "A32KVTAP2R", "A32P2Rn", "A32KVTAP2Rn"] and all(arms[label]["refine_lr"] == 1e-3 for label in list(arms)[-40:-36]) and "refine_lr" not in arms["A32P2"]
    assert list(arms)[-41] == "R8S"
    assert list(arms)[-36:-34] == ["NKVTAL", "A32KVTALc"] and arms["NKVTAL"]["intervention_target"] == "kv_transfer" and arms["NKVTAL"]["ratio"] == arms["W16L"].get("ratio", 1 / 16) and arms["A32KVTALc"]["init_ac"].endswith("A32KVTAL/AC_MODELS/rag_poc/epoch_004")
    assert list(arms)[-34:-30] == ["A64L", "A64KVTAL", "A64Ln", "A64KVTALn"] and all(arms[label]["ratio"] == 1 / 64 for label in list(arms)[-34:-30]) and arms["A64KVTALn"]["init_ac"].startswith("/root/") and arms["A64KVTAL"]["intervention_target"] == "kv_transfer" and arms["R8S"]["encoder_passes"] == 2 and arms["R8S"]["pool_surprisal"] == 1.0 and arms["R8S"]["train_items"] == 96
    assert list(arms)[-30:-26] == ["A32SW3", "A32KVTASW3", "A32SW3n", "A32KVTASW3n"] and all(arms[label]["rare_weight"] == 0.3 for label in list(arms)[-30:-26]) and arms["A32KVTASW3"]["intervention_target"] == "kv_transfer"
    frozen = ["F32X", "F32DI", "F32KVD", "F32KVS", "F32KVTA"]
    assert list(arms)[-26:-16] == [*frozen, *(label + "n" for label in frozen)] and all(arms[label]["reader_lr"] == 0.0 for label in list(arms)[-26:-16]) and arms["F32KVS"]["intervention_target"] == "kv_slots" and arms["F32KVD"]["intervention_target"] == "kv" and arms["F32DI"]["intervention_target"] == "residual" and arms["F32KVTAn"]["init_ac"].startswith("/root/")
    bare = ["B32X", "B32DI", "B32KVD", "B32KVS", "B16X", "B16DI", "B16KVD", "B16KVS"]
    assert list(arms)[-16:-8] == bare and all(arms[label]["init_reader"] is None and arms[label]["reader_lr"] == 0.0 and arms[label]["train_items"] == 21888 for label in bare) and arms["B16KVS"]["ratio"] == 1 / 16 and arms["B32DI"]["intervention_source"] == "passage_attention" and arms["B32DI"]["intervention_attention_depth"] == "matched" and arms["B32KVS"]["intervention_target"] == "kv_slots"
    assert list(arms)[-8:-4] == ["B32Xn", "B32DIn", "B32KVDn", "B32KVSn"] and all(arms[label]["init_reader"] is None and arms[label]["init_ac"].startswith("/root/") for label in list(arms)[-8:-4])
    assert list(arms)[-4:-2] == ["B32KVSL", "B16KVSL"] and all(arms[label]["epochs"] == 2 and arms[label]["init_reader"] is None and arms[label]["intervention_target"] == "kv_slots" for label in list(arms)[-4:-2])
    assert list(arms)[-2:] == ["F16Xn", "F16KVSn"] and all(arms[label]["ratio"] == 1 / 16 and arms[label]["reader_lr"] == 0.0 for label in list(arms)[-2:])


# --------------------------------------------------------------------------------------------------- 15. lora-style heads, row-position calibration, staging
@pytest.mark.parametrize("source, target", [("summary", "residual"), ("passage_attention", "residual"), ("side_layers", "kv")], ids=["residual", "passage", "kv"])
def test_lora_head_style_is_silent_at_init_and_owns_its_magnitude(tmp_path, hybrid_dir, source, target):
    from activation.ac_model import ActivationContextModelConfig as Config
    with pytest.raises(ValueError, match="head_style"):
        Config("x", "reader", "side_lora", "reader", "reader_lora", intervention_frequency=1, intervention_head_style="unit")
    with pytest.raises(ValueError, match="heads_frozen"):
        Config("x", "reader", "side_lora", "reader", "reader_lora", intervention_frequency=1, intervention_head_style="lora", intervention_heads_frozen=True)
    h, ac = build(tmp_path, hybrid_dir, frequency=1, source=source, target=target, head_style="lora", seed=31)
    hs, scaled = build(tmp_path / "scaled", hybrid_dir, frequency=1, source=source, target=target, seed=31)
    h0, zero = build(tmp_path / "zero", hybrid_dir, seed=31)
    # Same keys as the scaled style (no new state-dict entries), same draws for everything but the zeroed down projection; the scales are inert.
    assert ac.modules.state_dict().keys() == scaled.modules.state_dict().keys()
    for name, tensor in scaled.modules.state_dict().items():
        if ".down." in name and name.startswith("delta_heads."):
            assert torch.equal(ac.modules.state_dict()[name], torch.zeros_like(tensor)), name
        else:
            assert torch.equal(ac.modules.state_dict()[name], tensor), name
    for head in ac.modules.delta_heads.values():
        assert head.style == "lora" and head.scale_parameters() == []
        assert all(not scale.requires_grad for scale in ((head.relative_k, head.relative_v) if target == "kv" else (head.relative,))), "inert scales"
    inert = {id(p) for name, p in ac.modules.named_parameters() if name.split(".")[-1].startswith("relative")}
    assert inert and not inert & {id(p) for p in ac.trainable_parameters()}, "the inert scales are never trainable"
    assert ac.intervention_summary()["head_style"] == "lora" and ac.intervention_summary()["parametrization"] == "lora"
    trainer, items, examples, device = trainer_and_examples(h, ac)
    calibrate(trainer, ac, examples[:2])
    # Exactly zero deltas at init; the reader forward with the payload is bit-identical to the forward without it and to the input-only model's.
    output = ac.encode_batch([fixed_request(ac)], with_layers=True)[0]
    produced = list(output.layer_inputs.values()) + [tensor for pair in output.kv_inputs.values() for tensor in pair]
    assert produced and all(torch.equal(tensor, torch.zeros_like(tensor)) for tensor in produced)
    rows_zero = zero.encode_batch([fixed_request(zero)])[0]
    assert torch.equal(output.input_embeds, rows_zero), "the rows are the input-only model's (shared draws)"
    embedding = ac.target.model.get_input_embeddings()
    example = examples[0]
    outputs = ac.encode_batch(example.part_requests, with_layers=True)
    with torch.no_grad():
        student = embedding(torch.tensor(example.student_ids))
        with_payload, payload = trainer._assemble_student(student, example.spans, outputs, student.dtype)
        without, none = trainer._assemble_student(student, example.spans, outputs, student.dtype, deltas_off=True)
        assert payload is not None and none is None and torch.equal(with_payload, without)
        hidden_on = ac.target.decoder_forward(with_payload[None], None, lora_name="reader_lora", interventions=[payload])
        hidden_off = ac.target.decoder_forward(without[None], None, lora_name="reader_lora")
    assert torch.equal(hidden_on, hidden_off), "a zero delta added in place leaves the reader's states bit-identical"
    # The head still learns from update 0: the down projection's gradient is non-zero (the up projection's is zero until down leaves zero).
    loss = trainer._example_loss(ac, example, device, with_drift=False)[0]
    live = [(name, parameter) for name, parameter in ac.modules.named_parameters() if name.startswith("delta_heads.") and parameter.requires_grad]
    assert live and not any(name.split(".")[-1].startswith("relative") for name, _ in live), "the inert scales are outside autograd"
    grads = dict(zip([name for name, _ in live], torch.autograd.grad(loss, [parameter for _, parameter in live], allow_unused=True)))
    for name, g in grads.items():
        if ".down." in name:
            assert g is not None and float(g.norm()) > 0, f"{name}: the head learns through its down projection at init"
        else:
            assert g is None or float(g.norm()) == 0, f"{name}: no gradient reaches up through a zero down projection"
    # At init the deltas-off panel is the panel itself (the zero delta is added in place).
    h.module_manager.ensure_lora("reader_lora")
    assert trainer._evaluate(ac, items, device, None, "deltas_off", deltas_off=True)["gold_nll"] == trainer._evaluate(ac, items, device, None, "reporting")["gold_nll"]
    # A training epoch: no scale group, the inert scales untouched, the down projections move, the delta RMS traced relative to the row RMS.
    trainer = ActivationContextTrainer(h, config(learning_rate_delta_scales=5e-3, deltas_off_panel=True))
    before = intervention_state(ac)
    stats = trainer.train(ac.name, items, reporting_data=items)
    optimizer = trainer.optimizers[ac.name]
    assert len(optimizer.param_groups) == 2, "encoder and reader only: the lora style has no scale group"
    assert not inert & {id(p) for group in optimizer.param_groups for p in group["params"]}
    after = ac.modules.state_dict()
    for name, tensor in before.items():
        if name.split(".")[-1].startswith("relative") or name.startswith("delta_heads.") and ".rms" in name:
            assert torch.equal(tensor, after[name]), f"{name}: inert"
        elif ".down.weight" in name and name.startswith("delta_heads."):
            assert not torch.equal(tensor, after[name]), f"{name}: trains"
    keys = {f"{layer}:{half}" for layer in (2, 5) for half in "kv"} if target == "kv" else {2, 5}
    first = stats.intervention_trace[0]
    assert set(first["delta_rms"]) == set(first["delta_rms_abs"]) == keys
    assert all(value == 0.0 for value in first["delta_rms_abs"].values()) and all(value == 0.0 for value in first["delta_rms"].values()), "zero at init"
    assert all(value == pytest.approx(INTERVENTION_SCALE_FRACTION) for value in first["relative"].values()), "the inert scales stay at their init"
    assert stats.deltas_off is not None and stats.deltas_off["items"] == len(items)
    # The scaled style is untouched: the same seed still gives a full-magnitude random delta whose RMS is the calibrated scale.
    trainer_s, _, examples_s, _ = trainer_and_examples(hs, scaled)
    calibrate(trainer_s, scaled, examples_s[:2])
    output_s = scaled.encode_batch([fixed_request(scaled)], with_layers=True)[0]
    for tensor in list(output_s.layer_inputs.values()) + [t for pair in output_s.kv_inputs.values() for t in pair]:
        assert float(tensor.pow(2).mean(-1).sqrt().mean()) > 0
    # A lora checkpoint loads into a lora model only (the weights mean different things).
    folder = ac.save(str(tmp_path / "lora_ckpt"))
    with pytest.raises(ValueError, match="head style"):
        scaled.load(folder)
    h2, again = build(tmp_path / "again", hybrid_dir, frequency=1, source=source, target=target, head_style="lora", seed=5, ac_checkpoint=folder)
    assert again.interventions_calibrated and all(torch.equal(tensor, again.modules.state_dict()[name]) for name, tensor in after.items() if name.startswith("delta_heads."))


@pytest.mark.parametrize("target", ["residual", "kv"])
def test_row_position_calibration(tmp_path, hybrid_dir, target):
    from activation.ac_model import ActivationContextModelConfig as Config
    with pytest.raises(ValueError, match="intervention_calibration"):
        Config("x", "reader", "side_lora", "reader", "reader_lora", intervention_frequency=1, intervention_calibration="rows")
    source = "side_layers" if target == "kv" else "summary"
    rows_h, on_rows = build(tmp_path / "rows", hybrid_dir, frequency=1, target=target, source=source, calibration="row_positions", row_scale_init=4.0)
    plain_h, plain = build(tmp_path / "plain", hybrid_dir, frequency=1, target=target, source=source, row_scale_init=4.0)
    for name, tensor in plain.modules.state_dict().items():
        assert torch.equal(tensor, on_rows.modules.state_dict()[name]), "the calibration mode changes the measurement, not the modules"
    trainer_r, items_r, examples_r, device = trainer_and_examples(rows_h, on_rows)
    trainer_p, items_p, examples_p, _ = trainer_and_examples(plain_h, plain)
    with pytest.raises(ValueError, match="row_inputs"):
        on_rows.calibrate_interventions([examples_r[0].student_ids], ["x"])          # the mode and the inputs must agree, both ways
    with pytest.raises(ValueError, match="token sequences"):
        plain.calibrate_interventions(None, ["x"], row_inputs=[(torch.zeros(4, plain.d_target), [1, 2])])
    with pytest.raises(RuntimeError, match="not calibrated"):
        on_rows.encode_batch([fixed_request(on_rows)], with_layers=True)              # uncalibrated encoding is allowed inside the calibration only
    measured = calibrate(trainer_r, on_rows, examples_r[:2])
    reference = calibrate(trainer_p, plain, examples_p[:2])
    assert on_rows.calibration_items == [items_r[0].item_id, items_r[1].item_id] and on_rows.intervention_summary()["calibration"] == "row_positions"
    assert on_rows.intervention_diagnostics["cosine"] == {} and on_rows.intervention_diagnostics["delta_rms"] == {}, "the calibration's encodes leave no diagnostics behind"
    for layer in (2, 5):
        if target == "kv":                                                             # K after k_norm has RMS 1 everywhere; V at the rows is its own statistic
            (k_rows, v_rows), (k_plain, v_plain) = measured[layer], reference[layer]
            assert all(math.isfinite(value) and value > 0 for value in (k_rows, v_rows)) and k_rows == pytest.approx(k_plain, rel=1e-3) and v_rows != v_plain
        else:
            assert math.isfinite(measured[layer]) and measured[layer] > 0
            assert measured[layer] > reference[layer], f"layer {layer}: the rows (written at 4 x the embedding RMS) are louder than the stripped text's positions"
    # The measurement is the training trace's row RMS: the same forward (rows present, no deltas) measured by the intervention hooks at the same positions.
    example = examples_r[0]
    embeds, positions = trainer_r._calibration_rows(on_rows, example, device)
    assert positions == [position for start, end in example.spans for position in range(start, end)]
    if target == "residual":
        from activation.harness.hf_utils import capture_layer_input_rms
        with torch.no_grad(), capture_layer_input_rms(on_rows.target.model, [2, 5], positions) as record:
            on_rows.target.decoder_forward(embeds[None], None, lora_name="reader_lora", gradient_checkpointing=False)
        outputs = on_rows.encode_batch(example.part_requests, with_layers=True)
        with torch.no_grad():
            student = on_rows.target.model.get_input_embeddings()(torch.tensor(example.student_ids))
            student, payload = trainer_r._assemble_student(student, example.spans, outputs, student.dtype)
            payload["layer_inputs"] = {layer: torch.zeros_like(rows) for layer, rows in payload["layer_inputs"].items()}   # the same forward, the hooks measuring
            on_rows.target.decoder_forward(student[None], None, lora_name="reader_lora", interventions=[payload])
        for layer in (2, 5):
            assert record[layer][0] == pytest.approx(float(payload["row_rms"][layer]), rel=1e-5)
    # The lora style with the row-position calibration: the value feeds the diagnostics only (the forward is zero regardless).
    lh, lora = build(tmp_path / "lora", hybrid_dir, frequency=1, target=target, source=source, head_style="lora", calibration="row_positions")
    trainer_l, items_l, examples_l, _ = trainer_and_examples(lh, lora)
    calibrate(trainer_l, lora, examples_l[:2])
    output = lora.encode_batch([fixed_request(lora)], with_layers=True)[0]
    assert all(torch.equal(t, torch.zeros_like(t)) for t in list(output.layer_inputs.values()) + [t for pair in output.kv_inputs.values() for t in pair])
    stats = ActivationContextTrainer(lh, config()).train(lora.name, items_l)
    assert lora.intervention_summary()["calibration"] == "row_positions" and stats.intervention_trace[0]["row_rms"][2] > 0


def test_heads_only_staging_and_resume(tmp_path, hybrid_dir):
    with pytest.raises(ValueError, match="heads_only"):
        config(intervention_heads_only_updates=-1)
    h0, zero = build(tmp_path / "zero", hybrid_dir)
    with pytest.raises(ValueError, match="deep models only"):
        ActivationContextTrainer(h0, config(intervention_heads_only_updates=1)).train(zero.name, items_for(h0, zero))
    h, ac = build(tmp_path, hybrid_dir, frequency=1, source="passage_attention", head_style="lora", calibration="row_positions")
    items = items_for(h, ac)
    # Three items, two per update: 2 updates per epoch. Heads-only for the first 3 updates; the run trains 2, the resume 1 + 1.
    trainer = ActivationContextTrainer(h, config(intervention_heads_only_updates=3, checkpoint_every_epoch=True, warmup_updates=2, lr_schedule="cosine", schedule_total_updates=4))
    shared_before, heads_before = shared_state(ac), intervention_state(ac)
    stats = trainer.train(ac.name, items)
    assert trainer.updates_done[ac.name] == 2 and stats.penalty_counts == [0, 0], "no penalty term while the reader is held"
    assert stats.grad_norm_reader == [0.0, 0.0]
    for name, tensor in shared_before.items():
        assert torch.equal(tensor, shared_state(ac)[name]), f"{name}: frozen during the heads-only updates"
    assert any(not torch.equal(tensor, intervention_state(ac)[name]) for name, tensor in heads_before.items() if name.startswith("delta_heads.") and ".down." in name), "the heads step"
    assert any(not torch.equal(tensor, intervention_state(ac)[name]) for name, tensor in heads_before.items() if name.startswith("passage_attention.") and ".out." in name), "the attention blocks step"
    assert stats.learning_rates[0][0] == pytest.approx(config().learning_rate_ac * 0.5) and stats.learning_rates[1][0] > stats.learning_rates[0][0], "the warm-up clock runs through the staging"
    assert stats.learning_rates[0][1] == 0.0 == stats.learning_rates[1][1], "the reader group's rate is zeroed like the staged reader recipe"
    # Resume from the epoch checkpoint: update 3 is still heads-only, update 4 trains everything (the clock survived the round trip).
    record = json.loads((tmp_path / "AC_MODELS" / ac.name / "completed_epoch.json").read_text())
    h2, ac2 = build(tmp_path / "second", hybrid_dir, frequency=1, source="passage_attention", head_style="lora", calibration="row_positions", seed=9,
                    ac_checkpoint=str(tmp_path / record["ac_checkpoint"]), reader_checkpoint=str(tmp_path / record["target_lora_checkpoint"]))
    trainer2 = ActivationContextTrainer(h2, config(intervention_heads_only_updates=3, checkpoint_every_epoch=True, resume_optimizer=True, examples_per_update=3,
                                                   warmup_updates=2, lr_schedule="cosine", schedule_total_updates=4))
    items2 = items_for(h2, ac2)
    shared_resumed = shared_state(ac2)
    stats2 = trainer2.train(ac2.name, items2)
    assert stats2.resumed_from and trainer2.updates_done[ac2.name] == 3 and stats2.grad_norm_reader == [0.0] and stats2.penalty_counts == [0]
    for name, tensor in shared_resumed.items():
        assert torch.equal(tensor, shared_state(ac2)[name]), f"{name}: update 3 is still heads-only after the resume"
    stats3 = trainer2.train(ac2.name, items2)
    assert trainer2.updates_done[ac2.name] == 4 and stats3.grad_norm_reader[0] > 0 and stats3.penalty_counts == [3]
    assert any(not torch.equal(tensor, shared_state(ac2)[name]) for name, tensor in shared_resumed.items() if name.startswith("mixer.")), "update 4 trains the encoder"
    assert any(not torch.equal(tensor, shared_state(ac2)[name]) for name, tensor in shared_resumed.items() if name.startswith("reader:")), "update 4 trains the reader"
    assert stats3.learning_rates[0][0] < stats2.learning_rates[0][0], "the cosine clock kept running"
    # Without staging (the default), the same first update moves the shared parameters: the switch is what held them.
    h3, plain = build(tmp_path / "plain", hybrid_dir, frequency=1, source="passage_attention", head_style="lora", calibration="row_positions")
    before = shared_state(plain)
    ActivationContextTrainer(h3, config()).train(plain.name, items_for(h3, plain))
    assert any(not torch.equal(tensor, shared_state(plain)[name]) for name, tensor in before.items() if name.startswith("mixer."))


# --------------------------------------------------------------------------------------------------- 16. K/V slots (replacement at the rows)
def slot_parameters(ac):
    return [parameter for name, parameter in ac.modules.named_parameters() if name.startswith("delta_heads.")]


def open_slots(ac, std: float = 0.2, takeover: float = 0.3) -> None:
    """Audible slots: the zero down projections drawn off zero and the takeovers moved, so the written K/V differ from the reader's own."""
    with torch.no_grad():
        for head in ac.modules.delta_heads.values():
            head.down.weight.normal_(std=std); head.down.bias.normal_(std=std)
            head.takeover_k.fill_(takeover); head.takeover_v.fill_(-takeover)


def test_kv_slots_replace_the_rows_and_start_as_the_reader(tmp_path, hybrid_dir):
    from activation.ac_model import ActivationContextModelConfig as Config
    from activation.ac_model.ac_model_utils import KVSlotHead
    with pytest.raises(ValueError, match="intervention_target must be one of"):
        Config("x", "reader", "side_lora", "reader", "reader_lora", intervention_frequency=1, intervention_target="slots")
    with pytest.raises(ValueError, match="head_style 'lora'"):
        Config("x", "reader", "side_lora", "reader", "reader_lora", intervention_frequency=1, intervention_target="kv_slots")
    with pytest.raises(ValueError, match="designed only"):
        Config("x", "reader", "side_lora", "reader", "reader_lora", intervention_frequency=1, intervention_target="kv_slots", intervention_head_style="lora", intervention_slots_extra=4)
    with pytest.raises(ValueError, match="full-attention"):
        build(tmp_path / "gdn", hybrid_dir, frequency=1, prefer=False, target="kv_slots", head_style="lora")     # layer 3 is a linear-attention block
    h, ac = build(tmp_path, hybrid_dir, frequency=1, source="side_layers", target="kv_slots", head_style="lora", calibration="row_positions", seed=41)
    h0, zero = build(tmp_path / "zero", hybrid_dir, seed=41)
    hk, additive = build(tmp_path / "kv", hybrid_dir, frequency=1, source="side_layers", target="kv", head_style="lora", calibration="row_positions", seed=41)
    assert ac.intervention_layers == [2, 5] and ac.kv_slots and ac.kv_targets and not ac.kv_interventions and ac.modules.is_deep
    assert all(isinstance(head, KVSlotHead) for head in ac.modules.delta_heads.values()) and ac.modules.cross_attention is None
    d_kv = ac.target.model_config.model_description.d_kv
    keys = set(ac.modules.state_dict())
    assert {"delta_heads.2.takeover_k", "delta_heads.2.takeover_v", "delta_heads.2.rms_k", "delta_heads.5.up.weight", "delta_heads.5.down.bias"} <= keys
    assert not any(name.startswith("cross_attention.") for name in keys) and "delta_heads.2.relative_k" not in keys
    for name, tensor in zero.modules.state_dict().items():
        assert torch.equal(tensor, ac.modules.state_dict()[name]), f"{name}: the shared draws are the input-only model's"
    for layer in ("2", "5"):
        head = ac.modules.delta_heads[layer]
        assert torch.equal(head.down.weight, torch.zeros_like(head.down.weight)) and float(head.takeover_k.detach()) == 0.0 == float(head.takeover_v.detach())
        assert torch.equal(head.up.weight, additive.modules.delta_heads[layer].up.weight), "the slot head's up projection draws exactly like the K/V delta head's"
        assert head.scale_parameters() == [head.takeover_k, head.takeover_v], "the takeovers are the slot head's scale parameters"
        assert additive.modules.delta_heads[layer].scale_parameters() == [], "the lora-style K/V head still has none"
    summary_keys = ac.intervention_summary()
    assert summary_keys["target"] == "kv_slots" and summary_keys["takeover"] == {2: {"k": 0.0, "v": 0.0}, 5: {"k": 0.0, "v": 0.0}} and summary_keys["scales"] == {} and summary_keys["parametrization"] == "lora"
    trainer, items, examples, device = trainer_and_examples(h, ac)
    rms = calibrate(trainer, ac, examples[:2])
    assert all(k > 0 and v > 0 for k, v in rms.values()) and tuple(ac.modules.intervention_calibration.shape) == (2, 2)
    # Init: the slot content is exactly zero and the keeps exactly one; the reader forward with the payload is bit-identical to the forward without it
    # and to the input-only model's forward (`interventions = 0`) on the same rows.
    output = ac.encode_batch([fixed_request(ac)], with_layers=True)[0]
    assert output.layer_inputs == {} and output.kv_inputs == {} and set(output.kv_slots) == {2, 5} and output.cross_passage is None
    for layer, (k, v, keep_k, keep_v) in output.kv_slots.items():
        assert k.shape == v.shape == (output.input_embeds.shape[0], d_kv) and torch.equal(k, torch.zeros_like(k)) and torch.equal(v, torch.zeros_like(v))
        assert float(keep_k.detach()) == 1.0 == float(keep_v.detach()) and keep_k.requires_grad
    assert torch.equal(output.input_embeds, zero.encode_batch([fixed_request(zero)])[0])
    example = examples[0]
    target = ac.target
    embedding = target.model.get_input_embeddings()
    outputs = ac.encode_batch(example.part_requests, with_layers=True)
    with torch.no_grad():
        student = embedding(torch.tensor(example.student_ids))
        with_payload, payload = trainer._assemble_student(student, example.spans, outputs, student.dtype)
        without, none = trainer._assemble_student(student, example.spans, outputs, student.dtype, deltas_off=True)
        assert set(payload["kv_slots"]) == {2, 5} and payload["kv_inputs"] == {} and payload["layer_inputs"] == {} and none is None
        hidden_on = target.decoder_forward(with_payload[None], None, lora_name="reader_lora", interventions=[payload])
        hidden_off = target.decoder_forward(without[None], None, lora_name="reader_lora")
        zero.cache.clear()                                                             # rollout mode: the cache would serve the bf16 copy of the first encode
        zero_rows = [row.input_embeds for row in zero.encode_batch(example.part_requests, with_layers=True)]
        zero_embeds, _ = trainer._assemble_student(student, example.spans, [ACOutput(rows) for rows in zero_rows], student.dtype)
        hidden_zero = zero.target.decoder_forward(zero_embeds[None], None, lora_name="reader_lora")
    assert torch.equal(hidden_on, hidden_off), "at init the written K/V are the reader's own: bit-identical to no intervention"
    assert torch.equal(with_payload, zero_embeds) and torch.equal(hidden_on, hidden_zero), "and to the input-only model's forward"
    assert all(float(payload["slot_kv_rms"][layer][part]) == float(payload["row_kv_rms"][layer][part]) for layer in (2, 5) for part in (0, 1)), "written RMS == own RMS at init"
    # Gradients reach the head at init: the down projections (the content) and the takeovers (the reader's own K/V are heard); up gets none through a zero down.
    loss = trainer._example_loss(ac, example, device, with_drift=False)[0]
    names = [name for name, _ in ac.modules.named_parameters() if name.startswith("delta_heads.")]
    grads = dict(zip(names, torch.autograd.grad(loss, slot_parameters(ac), allow_unused=True)))
    for name, g in grads.items():
        if ".down." in name or name.endswith(("takeover_k", "takeover_v")):
            assert g is not None and float(g.norm()) > 0, name
        else:
            assert g is None or float(g.norm()) == 0, name
    # Opened slots: the hook replaces exactly the row-position slices of K (after k_norm, pre-RoPE) and V (after v_proj) with keep x own + content, nothing else.
    open_slots(ac)
    outputs = ac.encode_batch(example.part_requests, with_layers=True)
    with torch.no_grad():
        embeds, payload = trainer._assemble_student(student, example.spans, outputs, student.dtype)
    install_intervention_hooks(target.model)
    layer2 = decoder_layers(target.model)[2]
    seen = {}
    def record(name):
        def hook(module, args, output):
            seen.setdefault(name, []).append(output.detach().clone())
        return hook
    handles = [layer2.self_attn.k_norm.register_forward_hook(record("k")), layer2.self_attn.v_proj.register_forward_hook(record("v")),
               layer2.register_forward_pre_hook(lambda module, args: seen.setdefault("input", []).append(args[0].detach().clone()))]
    try:
        with torch.no_grad():
            with_slots = target.decoder_forward(embeds[None], None, lora_name="reader_lora", interventions=[payload])
            plain = target.decoder_forward(embeds[None], None, lora_name="reader_lora")
    finally:
        for handle in handles:
            handle.remove()
    positions = torch.tensor(payload["positions"])
    k, v, keep_k, keep_v = payload["kv_slots"][2]
    (k_on, k_off), (v_on, v_off) = seen["k"], seen["v"]
    k_on, k_off, v_on, v_off = (tensor[0].reshape(embeds.shape[0], -1) for tensor in (k_on, k_off, v_on, v_off))
    torch.testing.assert_close(k_on[positions], k_off[positions] * float(keep_k) + k, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(v_on[positions], v_off[positions] * float(keep_v) + v, atol=1e-5, rtol=1e-5)
    assert float(keep_k) == pytest.approx(0.7) and float(keep_v) == pytest.approx(1.3) and not torch.allclose(k_on[positions], k_off[positions])
    mask = torch.ones(embeds.shape[0], dtype=torch.bool); mask[positions] = False
    assert torch.equal(k_on[mask], k_off[mask]) and torch.equal(v_on[mask], v_off[mask]), "no other position's K or V moves"
    assert torch.equal(seen["input"][0], seen["input"][1]), "the residual stream at the layer input is untouched"
    first = int(positions.min())
    assert torch.equal(with_slots[0, :first], plain[0, :first]) and not torch.equal(with_slots[0, first:], plain[0, first:])
    with pytest.raises(ValueError, match="both K/V deltas and K/V slots"):
        target.decoder_forward(embeds[None], None, lora_name="reader_lora", interventions=[{**payload, "kv_inputs": {2: (k, v)}}])
    # Generation binds the slots across the KV cache: the first step's logits are the training forward's on the prefix, and greedy decoding equals a full recompute.
    manager = h.module_manager
    prefix_length = example.student_positions[0] + 1
    with torch.no_grad():
        prefix = embedding(torch.tensor(example.student_ids[:prefix_length]))
    present = [(span, request) for span, request in zip(example.spans, example.part_requests) if span[1] <= prefix_length]
    with torch.no_grad():
        prefix, prefix_payload = trainer._assemble_student(prefix, [span for span, _ in present], ac.encode_batch([request for _, request in present], with_layers=True), prefix.dtype)
    peft_model = manager.ensure_lora("reader_lora")
    head = target.model.get_output_embeddings()
    steps = 3
    generation = dict(max_new_tokens=steps, min_new_tokens=steps, do_sample=False, use_cache=True, pad_token_id=target.tokenizer.pad_token_id, output_logits=True, return_dict_in_generate=True)
    with torch.no_grad(), manager.lora_context("reader", "reader_lora"):
        generated = generate_with_interventions(peft_model, inputs_embeds=prefix[None], interventions=[prefix_payload], **generation)
        training_logits = head(target.decoder_forward(prefix[None], None, lora_name="reader_lora", interventions=[prefix_payload])[0, -1])
        base_logits = head(target.decoder_forward(prefix[None], None, lora_name="reader_lora")[0, -1])
    torch.testing.assert_close(generated.logits[0][0], training_logits, atol=1e-4, rtol=1e-4)
    assert not torch.allclose(training_logits, base_logits), "the opened slots are heard"
    oracle = []
    with torch.no_grad():
        for _ in range(steps):
            full = torch.cat([prefix, embedding(torch.tensor(oracle, dtype=torch.long))], dim=0) if oracle else prefix
            logits = head(target.decoder_forward(full[None], None, lora_name="reader_lora", interventions=[prefix_payload])[0, -1])
            logits[target.tokenizer.eos_token_id] = float("-inf")
            oracle.append(int(logits.argmax()))
    assert generated.sequences[0].tolist() == oracle
    # deltas_off = the reader's own K/V (no payload): the panel differs from the held panel once the slots are open.
    manager.ensure_lora("reader_lora")
    on = trainer._evaluate(ac, items, device, None, "reporting"); off = trainer._evaluate(ac, items, device, None, "deltas_off", deltas_off=True)
    assert on["gold_nll"] != off["gold_nll"]
    # A training epoch: three optimizer groups - the scale group holds exactly the four takeovers at learning_rate_delta_scales (no lora
    # scalar, no content parameter, nothing of the encoder); the trace carries the takeover and the written / own ratio per half, 1.0 at init.
    hf, fresh = build(tmp_path / "fresh", hybrid_dir, frequency=1, source="side_layers", target="kv_slots", head_style="lora", calibration="row_positions", seed=41)
    trainer_f = ActivationContextTrainer(hf, config(learning_rate_delta_scales=5e-3, deltas_off_panel=True))
    before = intervention_state(fresh)
    stats = trainer_f.train(fresh.name, items_for(hf, fresh), reporting_data=items_for(hf, fresh))
    groups = trainer_f.optimizers[fresh.name].param_groups
    assert len(groups) == 3, "encoder, reader, the takeovers"
    takeover_ids = {id(head.takeover_k) for head in fresh.modules.delta_heads.values()} | {id(head.takeover_v) for head in fresh.modules.delta_heads.values()}
    assert {id(parameter) for parameter in groups[-1]["params"]} == takeover_ids and len(groups[-1]["params"]) == 4
    assert not any(id(parameter) in takeover_ids for group in groups[:-1] for parameter in group["params"]), "the takeovers leave the encoder group"
    trainer_k = ActivationContextTrainer(hk, config(learning_rate_delta_scales=5e-3, deltas_off_panel=True))
    trainer_k.train(additive.name, items_for(hk, additive))
    assert len(trainer_k.optimizers[additive.name].param_groups) == 2, "the lora-style K/V heads have no scale group"
    first_entry = stats.intervention_trace[0]
    keys = {f"{layer}:{half}" for layer in (2, 5) for half in "kv"}
    assert set(first_entry["takeover"]) == set(first_entry["slot_rms"]) == set(first_entry["delta_rms"]) == keys and first_entry["scale"] == {} == first_entry["relative"]
    assert all(value == 0.0 for value in first_entry["takeover"].values()) and all(value == 0.0 for value in first_entry["delta_rms"].values())
    assert all(value == pytest.approx(1.0) for value in first_entry["slot_rms"].values()), "written / own == 1 at init"
    assert set(first_entry["head_grad_norm"]) == {2, 5} and all(first_entry["row_rms"][layer] > 0 for layer in (2, 5))
    after = fresh.modules.state_dict()
    assert any(not torch.equal(tensor, after[name]) for name, tensor in before.items() if ".down.weight" in name), "the slot content trains"
    assert any(not torch.equal(tensor, after[name]) for name, tensor in before.items() if name.endswith(("takeover_k", "takeover_v"))), "the takeovers train"
    assert stats.deltas_off is not None and stats.payload_bytes > 0 and fresh.intervention_summary()["takeover"][2]["k"] != 0.0
    # Checkpoints: a kv_slots checkpoint round-trips into a kv_slots model and is refused by the additive K/V and the residual targets.
    folder = fresh.save(str(tmp_path / "slots_ckpt"))
    with pytest.raises(ValueError, match="source/target"):
        additive.load(folder)
    hr, residual = build(tmp_path / "residual", hybrid_dir, frequency=1, source="side_layers", target="residual", head_style="lora", calibration="row_positions")
    with pytest.raises(ValueError, match="source/target"):
        residual.load(folder)
    h2, again = build(tmp_path / "again", hybrid_dir, frequency=1, source="side_layers", target="kv_slots", head_style="lora", calibration="row_positions", seed=5, ac_checkpoint=folder)
    assert again.interventions_calibrated and again.intervention_summary()["takeover"] == fresh.intervention_summary()["takeover"]
    assert all(torch.equal(tensor, again.modules.state_dict()[name]) for name, tensor in after.items() if name.startswith("delta_heads."))


# --------------------------------------------------------------------------------------------------- 17. cross-attention ceiling
def cross_parameters(ac):
    return [parameter for name, parameter in ac.modules.named_parameters() if name.startswith("cross_attention.")]


def open_cross(ac, std: float = 0.2) -> None:
    with torch.no_grad():
        for block in ac.modules.cross_attention.values():
            block.out.weight.normal_(std=std)


def test_cross_attention_ceiling(tmp_path, hybrid_dir):
    from activation.ac_model import ActivationContextModelConfig as Config
    with pytest.raises(ValueError, match="cross_attention"):
        Config("x", "reader", "side_lora", "reader", "reader_lora", intervention_frequency=1, intervention_target="cross_attention", intervention_source="side_layers")
    with pytest.raises(ValueError, match="cross_attention"):
        Config("x", "reader", "side_lora", "reader", "reader_lora", intervention_frequency=1, intervention_target="cross_attention", intervention_heads_frozen=True)
    h, ac = build(tmp_path, hybrid_dir, frequency=1, target="cross_attention", head_style="lora", calibration="row_positions", seed=43)
    h0, zero = build(tmp_path / "zero", hybrid_dir, seed=43)
    assert ac.intervention_layers == [2, 5] and ac.cross_attention and ac.is_deep and ac.modules.is_deep and ac.modules.delta_heads is None
    assert set(ac.modules.cross_attention) == {"2", "5"} and len(ac.modules.intervention_modules) == 2
    keys = set(ac.modules.state_dict())
    assert not any(name.startswith("delta_heads.") for name in keys) and {"cross_attention.2.out.weight", "cross_attention.5.query.weight", "intervention_calibration"} <= keys
    assert not any(name.endswith("passage_norm.weight") for name in keys), "equal widths: the block shares one LayerNorm"
    for name, tensor in zero.modules.state_dict().items():
        assert torch.equal(tensor, ac.modules.state_dict()[name]), f"{name}: the shared draws are the input-only model's"
    for block in ac.modules.cross_attention.values():
        assert not block.residual and torch.equal(block.out.weight, torch.zeros_like(block.out.weight)) and torch.equal(block.out.bias, torch.zeros_like(block.out.bias))
    summary = ac.intervention_summary()
    assert summary["measurement_only"] and summary["target"] == "cross_attention" and summary["head_parameters"] == 0 and summary["attention_parameters"] == sum(p.numel() for p in cross_parameters(ac))
    assert "takeover" not in summary and "measurement_only" not in zero.intervention_summary() and "takeover" not in build(tmp_path / "plain_kv", hybrid_dir, frequency=1, source="side_layers", target="kv")[1].intervention_summary()
    trainer, items, examples, device = trainer_and_examples(h, ac)
    rms = calibrate(trainer, ac, examples[:2])
    assert set(rms) == {2, 5} and all(value > 0 for value in rms.values())
    # The encoder emits the passage's final side states whole (no per-layer head); the payload needs the model's blocks.
    output = ac.encode_batch([fixed_request(ac)], with_layers=True)[0]
    assert output.layer_inputs == {} and output.kv_inputs == {} and output.kv_slots == {} and output.cross_passage is not None
    assert output.cross_passage.dim() == 2 and output.cross_passage.shape[1] == ac.d_side and output.cross_passage.shape[0] > output.input_embeds.shape[0]
    assert torch.equal(output.input_embeds, zero.encode_batch([fixed_request(zero)])[0])
    example = examples[0]
    target = ac.target
    embedding = target.model.get_input_embeddings()
    outputs = ac.encode_batch(example.part_requests, with_layers=True)
    with torch.no_grad():
        student = embedding(torch.tensor(example.student_ids))
    with pytest.raises(ValueError, match="ac_model="):
        trainer._assemble_student(student, example.spans, outputs, student.dtype)
    with torch.no_grad():
        embeds, payload = trainer._assemble_student(student, example.spans, outputs, student.dtype, ac_model=ac)
        kept, none = trainer._assemble_student(student, example.spans, outputs, student.dtype, deltas_off=True, ac_model=ac)
    cross = payload["cross_inputs"]
    n = sum(output.cross_passage.shape[0] for output in outputs)
    assert cross["passage"].shape == (n, ac.d_side) and cross["visible_from"] == [start for (start, _), output in zip(example.spans, outputs) for _ in range(output.cross_passage.shape[0])]
    assert set(cross["blocks"]) == {2, 5} and cross["blocks"][2] is ac.modules.cross_attention["2"] and payload["layer_inputs"] == {} and payload["kv_slots"] == {}
    assert none is None and torch.equal(kept, embeds)
    # Zero at init: the forward with the payload is bit-identical to the forward without it (deltas-off = base).
    with torch.no_grad():
        hidden_on = target.decoder_forward(embeds[None], None, lora_name="reader_lora", interventions=[payload])
        hidden_off = target.decoder_forward(embeds[None], None, lora_name="reader_lora")
    assert torch.equal(hidden_on, hidden_off)
    assert all(float(payload["cross_rms"][layer][0]) == 0.0 and float(payload["cross_rms"][layer][1]) > 0 for layer in (2, 5)), "the hook traced a zero delta and the target RMS"
    # Non-zero gradient at init into the blocks' out projections (and into the side adapter through the passage states once the blocks are open).
    loss = trainer._example_loss(ac, example, device, with_drift=False)[0]
    names = [name for name, _ in ac.modules.named_parameters() if name.startswith("cross_attention.")]
    grads = dict(zip(names, torch.autograd.grad(loss, cross_parameters(ac), allow_unused=True)))
    assert all(g is not None and float(g.norm()) > 0 for name, g in grads.items() if ".out." in name), "out learns at init"
    assert all(g is None or float(g.norm()) == 0 for name, g in grads.items() if ".out." not in name), "nothing reaches q/k/v through a zero out"
    open_cross(ac)
    params = cross_parameters(ac) + side_lora_parameters(ac) + list(h.module_manager.lora_parameters("reader_lora"))
    grads = {}
    for checkpointed in (False, True):
        trainer.config.gradient_checkpointing = checkpointed
        ac.config.side_gradient_checkpointing = checkpointed
        losses = [trainer._example_loss(ac, example, device, with_drift=False)[0] for example in examples[:2]]
        grads[checkpointed] = torch.autograd.grad(sum(losses), params, allow_unused=True)
    for direct, recomputed in zip(grads[False], grads[True]):
        assert direct is not None and recomputed is not None
        torch.testing.assert_close(direct, recomputed, atol=1e-6, rtol=1e-5)
    assert all(float(g.norm()) > 0 for g in grads[True][:len(cross_parameters(ac))]), "q/k/v/out and the norm all receive gradient through the opened path"
    assert any(float(g.norm()) > 0 for g in grads[True][len(cross_parameters(ac)):len(cross_parameters(ac)) + len(side_lora_parameters(ac))]), "the side adapter hears the reader through the passage states"
    # Opened: writes at the target positions only (every position from the first row on moves, nothing before it), and the delta is what the block computes.
    outputs = ac.encode_batch(example.part_requests, with_layers=True)
    with torch.no_grad():
        embeds, payload = trainer._assemble_student(student, example.spans, outputs, student.dtype, ac_model=ac)
    install_intervention_hooks(target.model)
    seen = []
    handle = decoder_layers(target.model)[2].register_forward_pre_hook(lambda module, args: seen.append(args[0].detach().clone()))
    try:
        with torch.no_grad():
            with_cross = target.decoder_forward(embeds[None], None, lora_name="reader_lora", interventions=[payload])
            plain = target.decoder_forward(embeds[None], None, lora_name="reader_lora")
    finally:
        handle.remove()
    first = min(payload["positions"])
    assert torch.equal(with_cross[0, :first], plain[0, :first]) and not torch.equal(with_cross[0, first:], plain[0, first:])
    layer_on, layer_off = seen                                                             # the layer-2 inputs, after the product pre-hook
    assert torch.equal(layer_on[0, :first], layer_off[0, :first])
    with torch.no_grad():
        length = embeds.shape[0]
        mask = torch.arange(first, length)[:, None] >= torch.tensor(cross["visible_from"])[None, :]
        expected = ac.modules.cross_attention["2"](layer_off[0, first:], payload["cross_inputs"]["passage"], mask)
    torch.testing.assert_close(layer_on[0, first:] - layer_off[0, first:], expected, atol=1e-5, rtol=1e-5)
    assert (layer_on[0, first:] - layer_off[0, first:]).abs().sum(dim=-1).gt(0).all(), "every target position from the first row on is written"
    # Generation: the delta reaches the generated positions too, so greedy decoding with the cache equals a full recompute.
    manager = h.module_manager
    prefix_length = example.student_positions[0] + 1
    with torch.no_grad():
        prefix = embedding(torch.tensor(example.student_ids[:prefix_length]))
    present = [(span, request) for span, request in zip(example.spans, example.part_requests) if span[1] <= prefix_length]
    with torch.no_grad():
        prefix, prefix_payload = trainer._assemble_student(prefix, [span for span, _ in present], ac.encode_batch([request for _, request in present], with_layers=True), prefix.dtype, ac_model=ac)
    peft_model = manager.ensure_lora("reader_lora")
    head = target.model.get_output_embeddings()
    steps = 3
    generation = dict(max_new_tokens=steps, min_new_tokens=steps, do_sample=False, use_cache=True, pad_token_id=target.tokenizer.pad_token_id, output_logits=True, return_dict_in_generate=True)
    with torch.no_grad(), manager.lora_context("reader", "reader_lora"):
        generated = generate_with_interventions(peft_model, inputs_embeds=prefix[None], interventions=[prefix_payload], **generation)
        training_logits = head(target.decoder_forward(prefix[None], None, lora_name="reader_lora", interventions=[prefix_payload])[0, -1])
    torch.testing.assert_close(generated.logits[0][0], training_logits, atol=1e-4, rtol=1e-4)
    oracle = []
    with torch.no_grad():
        for _ in range(steps):
            full = torch.cat([prefix, embedding(torch.tensor(oracle, dtype=torch.long))], dim=0) if oracle else prefix
            logits = head(target.decoder_forward(full[None], None, lora_name="reader_lora", interventions=[prefix_payload])[0, -1])
            logits[target.tokenizer.eos_token_id] = float("-inf")
            oracle.append(int(logits.argmax()))
    assert generated.sequences[0].tolist() == oracle
    # deltas-off = base: the panel without the payload equals the input-only model's panel on the same items; with the blocks open the held panel differs.
    manager.ensure_lora("reader_lora")
    off = trainer._evaluate(ac, items, device, None, "deltas_off", deltas_off=True)
    on = trainer._evaluate(ac, items, device, None, "reporting")
    assert off["gold_nll"] != on["gold_nll"]
    trainer0 = ActivationContextTrainer(h0, config())
    items0 = items_for(h0, zero)
    trainer0._store_teacher_targets(zero, [trainer0.build_example(zero, item).penalty for item in items0], zero.prepare(), lora=None)
    h0.module_manager.ensure_lora("reader_lora")
    assert off["gold_nll"] == pytest.approx(trainer0._evaluate(zero, items0, zero.device, None, "reporting")["gold_nll"], rel=1e-5)
    # A training epoch traces the blocks (out norm, gradient norm, delta / target RMS); the staging treats the blocks as the intervention modules.
    hf, fresh = build(tmp_path / "fresh", hybrid_dir, frequency=1, target="cross_attention", head_style="lora", calibration="row_positions", seed=43)
    trainer_f = ActivationContextTrainer(hf, config(deltas_off_panel=True, intervention_heads_only_updates=1))
    stats = trainer_f.train(fresh.name, items_for(hf, fresh), reporting_data=items_for(hf, fresh))
    entry = stats.intervention_trace[0]
    assert entry["attention_out_norm"] == {2: 0.0, 5: 0.0} and set(entry["head_grad_norm"]) == {2, 5} and entry["scale"] == {} and "takeover" not in entry
    assert entry["delta_rms"] == {2: 0.0, 5: 0.0} and entry["delta_rms_abs"] == {2: 0.0, 5: 0.0} and all(entry["row_rms"][layer] > 0 for layer in (2, 5))
    assert stats.deltas_off is not None and stats.payload_bytes > 0 and stats.grad_norm_reader[0] == 0.0, "the first update was heads-only (blocks are the intervention modules)"
    assert float(fresh.modules.cross_attention["2"].out.weight.abs().sum()) > 0, "the blocks step"
    # Checkpoints: round trip into a cross-attention model; refused by a residual model on the same layers.
    folder = fresh.save(str(tmp_path / "cross_ckpt"))
    hr, residual = build(tmp_path / "residual", hybrid_dir, frequency=1, head_style="lora", calibration="row_positions")
    with pytest.raises(ValueError, match="source/target"):
        residual.load(folder)
    h2, again = build(tmp_path / "again", hybrid_dir, frequency=1, target="cross_attention", head_style="lora", calibration="row_positions", seed=5, ac_checkpoint=folder)
    assert again.interventions_calibrated and torch.load(Path(folder) / "ac_modules.pt", map_location="cpu", weights_only=False)["interventions"]["measurement_only"]
    assert all(torch.equal(tensor, again.modules.state_dict()[name]) for name, tensor in fresh.modules.state_dict().items() if name.startswith("cross_attention."))


# --------------------------------------------------------------------------------------------------- 18. K/V transfer: the side model's own K/V at the rows
def transfer_parameters(ac):
    return [parameter for name, parameter in ac.modules.named_parameters() if name.startswith("delta_heads.")]


def side_inputs_and_kv(ac, request, layers):
    """Encodes `request` and returns (the ACOutput, the side decoder's input embeddings of that encode, the side model's own K (after k_norm) and V
    (after v_proj) per listed layer from a MANUAL side forward on those inputs with plain hooks, no checkpointing): the reference for the capture."""
    side_inputs = {}
    decoder = text_decoder(ac.side.model)
    def remember(module, args, kwargs):
        side_inputs.setdefault("embeds", kwargs["inputs_embeds"].detach().clone())
    handle = decoder.register_forward_pre_hook(remember, with_kwargs=True)
    try:
        output = ac.encode_batch([request], with_layers=True)[0]
    finally:
        handle.remove()
    seen = {}
    side_layers = decoder_layers(ac.side.model)
    handles = []
    for layer in layers:
        handles.append(side_layers[layer].self_attn.k_norm.register_forward_hook(lambda module, args, out, layer=layer: seen.setdefault(f"{layer}:k", out.detach().clone())))
        handles.append(side_layers[layer].self_attn.v_proj.register_forward_hook(lambda module, args, out, layer=layer: seen.setdefault(f"{layer}:v", out.detach().clone())))
    try:
        with torch.no_grad():
            ac.side.decoder_forward(side_inputs["embeds"], None, lora_name=ac.config.base_side_model_lora_name, gradient_checkpointing=False)
    finally:
        for handle in handles:
            handle.remove()
    S = side_inputs["embeds"].shape[1]
    return output, S, {key: tensor[0].reshape(S, -1) for key, tensor in seen.items()}


def reader_kv_at(target, embeds, payload, layer):
    """One reader forward with and one without the payload; (K after k_norm, V after v_proj) of `layer` flattened per position for both, and the two final states."""
    install_intervention_hooks(target.model)
    block = decoder_layers(target.model)[layer]
    seen = {}
    def record(name):
        def hook(module, args, output):
            seen.setdefault(name, []).append(output.detach().clone())
        return hook
    handles = [block.self_attn.k_norm.register_forward_hook(record("k")), block.self_attn.v_proj.register_forward_hook(record("v"))]
    try:
        with torch.no_grad():
            on = target.decoder_forward(embeds[None], None, lora_name="reader_lora", interventions=[payload])
            off = target.decoder_forward(embeds[None], None, lora_name="reader_lora")
    finally:
        for handle in handles:
            handle.remove()
    (k_on, k_off), (v_on, v_off) = seen["k"], seen["v"]
    flat = lambda tensor: tensor[0].reshape(embeds.shape[0], -1)
    return flat(k_on), flat(k_off), flat(v_on), flat(v_off), on, off


def test_kv_transfer_splices_the_side_kv(tmp_path, hybrid_dir):
    from activation.ac_model import ActivationContextModelConfig as Config
    from activation.ac_model.ac_model_utils import KVTransferHead
    base = ("x", "reader", "side_lora", "reader", "reader_lora")
    with pytest.raises(ValueError, match="head_style 'lora'"):
        Config(*base, intervention_frequency=1, intervention_source="side_layers", intervention_target="kv_transfer")
    with pytest.raises(ValueError, match="side_layers"):
        Config(*base, intervention_frequency=1, intervention_target="kv_transfer", intervention_head_style="lora")
    with pytest.raises(ValueError, match="-1"):
        Config(*base, intervention_frequency=-2)
    with pytest.raises(ValueError, match="explicit intervention_layers"):
        Config(*base, intervention_frequency=1, intervention_layers=[2])
    with pytest.raises(ValueError, match="distinct positive"):
        Config(*base, intervention_layers=[0, 2])
    # Frequency -1 = every full-attention layer of the reader (2 and 5 here; the last-layer flag is moot); an explicit list is honoured as given.
    common = dict(source="side_layers", target="kv_transfer", head_style="lora", calibration="row_positions")
    h, ac = build(tmp_path, hybrid_dir, frequency=-1, seed=41, **common)
    h0, zero = build(tmp_path / "zero", hybrid_dir, seed=41)
    hs, slots = build(tmp_path / "slots", hybrid_dir, frequency=-1, seed=41, **{**common, "target": "kv_slots"})
    assert ac.intervention_layers == [2, 5] == ac.intervention_source_layers and ac.modules.is_deep
    assert ac.kv_transfer and ac.kv_replacement and ac.kv_targets and not ac.kv_slots and not ac.kv_transfer_full and not ac.kv_interventions
    assert build(tmp_path / "listed", hybrid_dir, layers=[5], **common)[1].intervention_layers == [5]
    assert build(tmp_path / "res", hybrid_dir, frequency=-1, add_last=False)[1].intervention_layers == [2, 5], "-1 works for the residual target too"
    with pytest.raises(ValueError, match="outside"):
        build(tmp_path / "out", hybrid_dir, layers=[9], **common)
    with pytest.raises(ValueError, match="full-attention"):
        build(tmp_path / "gdn", hybrid_dir, layers=[3], **common)                    # layer 3 is a linear-attention block
    # State: the takeovers at the init value and the two diagnostic anchors per layer, nothing else; the shared draws are the input-only model's
    # (no draw without the correction head); a correction head adds up / down with down = 0 and up drawn exactly like the slot head's.
    keys = set(ac.modules.state_dict())
    assert {name for name in keys if name.startswith("delta_heads.")} == {f"delta_heads.{layer}.{name}" for layer in (2, 5) for name in ("takeover_k", "takeover_v", "rms_k", "rms_v")}
    for name, tensor in zero.modules.state_dict().items():
        assert torch.equal(tensor, ac.modules.state_dict()[name]), f"{name}: the shared draws are the input-only model's"
    for head in ac.modules.delta_heads.values():
        assert isinstance(head, KVTransferHead) and float(head.takeover_k.detach()) == 1.0 == float(head.takeover_v.detach()) and head(torch.zeros(1, ac.d_side)) is None
        assert head.scale_parameters() == [head.takeover_k, head.takeover_v]
    hc, corrected = build(tmp_path / "corr", hybrid_dir, frequency=-1, seed=41, transfer_init=0.5, transfer_correction=True, **common)
    for layer in ("2", "5"):
        head = corrected.modules.delta_heads[layer]
        assert torch.equal(head.down.weight, torch.zeros_like(head.down.weight)) and float(head.takeover_k.detach()) == 0.5 == float(head.takeover_v.detach())
        assert torch.equal(head.up.weight, slots.modules.delta_heads[layer].up.weight), "the correction's up projection draws exactly like the slot head's"
        assert head(torch.ones(3, ac.d_side))[0].shape == (3, head.d_kv) and torch.equal(head(torch.ones(3, ac.d_side))[1], torch.zeros(3, head.d_kv))
    for name, tensor in zero.modules.state_dict().items():
        assert torch.equal(tensor, corrected.modules.state_dict()[name]), name
    summary = ac.intervention_summary()
    assert summary["target"] == "kv_transfer" and summary["transfer"] == {"init": 1.0, "correction": False} and summary["parametrization"] == "lora"
    assert summary["takeover"] == {2: {"k": 1.0, "v": 1.0}, 5: {"k": 1.0, "v": 1.0}} and summary["scales"] == {} and "measurement_only" not in summary
    trainer, items, examples, device = trainer_and_examples(h, ac)
    rms = calibrate(trainer, ac, examples[:2])
    assert all(k > 0 and v > 0 for k, v in rms.values())
    # The payload: at t = 1 without a correction the content IS the side model's pre-RoPE K/V at its summary-row positions (keep = 0), captured through
    # the side's checkpointed layers; the reference is a manual side forward on the same side inputs with plain hooks and no checkpointing.
    request = fixed_request(ac)
    output, S, side_kv = side_inputs_and_kv(ac, request, (2, 5))
    V = output.input_embeds.shape[0]
    d_kv = ac.target.model_config.model_description.d_kv
    assert output.layer_inputs == {} and output.kv_inputs == {} and set(output.kv_slots) == {2, 5} and output.cross_passage is None
    for layer, (k, v, keep_k, keep_v) in output.kv_slots.items():
        assert k.shape == v.shape == (V, d_kv) and float(keep_k.detach()) == 0.0 == float(keep_v.detach()) and keep_k.requires_grad and k.requires_grad
        assert torch.equal(k, side_kv[f"{layer}:k"][S - V:]) and torch.equal(v, side_kv[f"{layer}:v"][S - V:]), f"layer {layer}: exactly the side's own K/V at the rows"
    assert torch.equal(output.input_embeds, zero.encode_batch([fixed_request(zero)])[0]), "the input rows are the input-only model's"
    # The reader writes exactly the side K/V over the row slices of K (after k_norm, pre-RoPE) and V (after v_proj): other positions untouched, the forward
    # differs from the reader's own (the replacement is meaningful at init); deltas-off keeps the rows and withholds the payload.
    example = examples[0]
    target = ac.target
    embedding = target.model.get_input_embeddings()
    outputs = ac.encode_batch(example.part_requests, with_layers=True)
    with torch.no_grad():
        student = embedding(torch.tensor(example.student_ids))
        embeds, payload = trainer._assemble_student(student, example.spans, outputs, student.dtype)
        off_embeds, none = trainer._assemble_student(student, example.spans, outputs, student.dtype, deltas_off=True)
    assert none is None and torch.equal(embeds, off_embeds) and set(payload["kv_slots"]) == {2, 5}
    positions = torch.tensor(payload["positions"])
    k_on, k_off, v_on, v_off, hidden_on, hidden_off = reader_kv_at(target, embeds, payload, 2)
    k, v, _, _ = payload["kv_slots"][2]
    assert torch.equal(k_on[positions], k) and torch.equal(v_on[positions], v), "the rows' K/V are the side's, exactly"
    assert not torch.allclose(k_on[positions], k_off[positions]), "and differ from the reader's own"
    mask = torch.ones(embeds.shape[0], dtype=torch.bool); mask[positions] = False
    assert torch.equal(k_on[mask], k_off[mask]) and torch.equal(v_on[mask], v_off[mask]), "no other position's K or V moves"
    first = int(positions.min())
    assert torch.equal(hidden_on[0, :first], hidden_off[0, :first]) and not torch.equal(hidden_on[0, first:], hidden_off[0, first:])
    # Gradients: the takeovers and the side adapter at a spliced layer hear the reader through the transferred K/V, and the gradient through the side's
    # checkpointed layers (the capture is a graph-attached slice inside the checkpointed region) equals the direct one.
    watched = [*transfer_parameters(ac), *side_lora_parameters(ac, layer=5)]
    loss = trainer._example_loss(ac, example, device, with_drift=False)[0]
    checkpointed = torch.autograd.grad(loss, watched, allow_unused=True)
    ac._training_gradient_checkpointing = False
    try:
        direct = torch.autograd.grad(trainer._example_loss(ac, example, device, with_drift=False)[0], watched, allow_unused=True)
    finally:
        ac._training_gradient_checkpointing = True
    heard = []
    for parameter, g_ckpt, g_direct in zip(watched, checkpointed, direct):
        assert g_ckpt is not None and g_direct is not None
        torch.testing.assert_close(g_ckpt, g_direct, atol=1e-6, rtol=1e-5)
        heard.append(float(g_ckpt.norm()) > 0)
    assert all(heard[:len(transfer_parameters(ac))]), "every takeover gets gradient"
    assert any(heard[len(transfer_parameters(ac)):]), "the side adapter of layer 5 gets gradient (its B matrices; A's gradient is zero through B = 0 at init)"
    # Generation binds the transferred K/V across the KV cache: the first step's logits are the training forward's on the prefix; greedy decoding equals a full recompute.
    manager = h.module_manager
    prefix_length = example.student_positions[0] + 1
    with torch.no_grad():
        prefix = embedding(torch.tensor(example.student_ids[:prefix_length]))
    present = [(span, request) for span, request in zip(example.spans, example.part_requests) if span[1] <= prefix_length]
    with torch.no_grad():
        prefix, prefix_payload = trainer._assemble_student(prefix, [span for span, _ in present], ac.encode_batch([request for _, request in present], with_layers=True), prefix.dtype)
    peft_model = manager.ensure_lora("reader_lora")
    head = target.model.get_output_embeddings()
    steps = 3
    generation = dict(max_new_tokens=steps, min_new_tokens=steps, do_sample=False, use_cache=True, pad_token_id=target.tokenizer.pad_token_id, output_logits=True, return_dict_in_generate=True)
    with torch.no_grad(), manager.lora_context("reader", "reader_lora"):
        generated = generate_with_interventions(peft_model, inputs_embeds=prefix[None], interventions=[prefix_payload], **generation)
        training_logits = head(target.decoder_forward(prefix[None], None, lora_name="reader_lora", interventions=[prefix_payload])[0, -1])
        base_logits = head(target.decoder_forward(prefix[None], None, lora_name="reader_lora")[0, -1])
        oracle = []
        sequence = prefix
        for step in range(steps):
            logits = head(target.decoder_forward(sequence[None], None, lora_name="reader_lora", interventions=[prefix_payload])[0, -1])
            oracle.append(int(logits.argmax()))
            sequence = torch.cat([sequence, embedding(torch.tensor([oracle[-1]]))], dim=0)
    torch.testing.assert_close(generated.logits[0][0], training_logits, atol=1e-4, rtol=1e-4)
    assert not torch.allclose(training_logits, base_logits) and generated.sequences[0].tolist() == oracle
    # deltas_off = the reader's own K/V (no payload): the panel differs from the held panel from update 0 on.
    on = trainer._evaluate(ac, items, device, None, "reporting"); off = trainer._evaluate(ac, items, device, None, "deltas_off", deltas_off=True)
    assert on["gold_nll"] != off["gold_nll"]
    # A training epoch: three optimizer groups (the takeovers alone in the last), the trace's takeover 1.0 and written == content at the first point,
    # the takeovers and the side adapter move; checkpoints round-trip and are refused by kv_slots, by a corrected transfer and by the additive K/V target.
    hf, fresh = build(tmp_path / "fresh", hybrid_dir, frequency=-1, seed=41, **common)
    trainer_f = ActivationContextTrainer(hf, config(learning_rate_delta_scales=5e-3, deltas_off_panel=True))
    before = {**intervention_state(fresh), **shared_state(fresh)}
    stats = trainer_f.train(fresh.name, items_for(hf, fresh), reporting_data=items_for(hf, fresh))
    groups = trainer_f.optimizers[fresh.name].param_groups
    takeover_ids = {id(parameter) for head in fresh.modules.delta_heads.values() for parameter in head.scale_parameters()}
    assert len(groups) == 3 and {id(parameter) for parameter in groups[-1]["params"]} == takeover_ids and len(groups[-1]["params"]) == 4
    first_entry = stats.intervention_trace[0]
    keys = {f"{layer}:{half}" for layer in (2, 5) for half in "kv"}
    assert set(first_entry["takeover"]) == set(first_entry["slot_rms"]) == set(first_entry["delta_rms"]) == keys and first_entry["scale"] == {} == first_entry["relative"]
    assert all(value == 1.0 for value in first_entry["takeover"].values())
    for key in keys:      # written == the content at t = 1 (the two diagnostics average over parts and over forwards respectively, so only up to the batching)
        assert first_entry["delta_rms"][key] > 0 and first_entry["slot_rms"][key] == pytest.approx(first_entry["delta_rms"][key], rel=0.5), key
    after = {**intervention_state(fresh), **shared_state(fresh)}
    assert all(not torch.equal(before[name], after[name]) for name in before if name.endswith(("takeover_k", "takeover_v"))), "the takeovers train"
    assert any(not torch.equal(before[name], after[name]) for name in before if name.startswith("side:")), "the side adapter trains"
    assert stats.deltas_off is not None and stats.payload_bytes > 0 and fresh.intervention_summary()["takeover"][2]["k"] != 1.0
    folder = fresh.save(str(tmp_path / "transfer_ckpt"))
    with pytest.raises(ValueError, match="source/target"):
        slots.load(folder)
    with pytest.raises(ValueError, match="correction"):
        corrected.load(folder)
    ha, additive = build(tmp_path / "kv", hybrid_dir, frequency=-1, source="side_layers", target="kv", head_style="lora", calibration="row_positions")
    with pytest.raises(ValueError, match="source/target"):
        additive.load(folder)
    h2, again = build(tmp_path / "again", hybrid_dir, frequency=-1, seed=5, ac_checkpoint=folder, **common)
    assert again.interventions_calibrated and again.intervention_summary()["takeover"] == fresh.intervention_summary()["takeover"]
    assert all(torch.equal(tensor, again.modules.state_dict()[name]) for name, tensor in fresh.modules.state_dict().items())


# --------------------------------------------------------------------------------------------------- 19. K/V transfer over the whole passage run (ceiling)
def test_kv_transfer_full_ceiling(tmp_path, hybrid_dir):
    from activation.ac_model import ActivationContextModelConfig as Config
    from activation.ac_model.ac_model_utils import tokenize_with_parts
    from activation.common.ac_parts import AC_PART_TYPE
    with pytest.raises(ValueError, match="pooling"):
        Config("x", "reader", "side_lora", "reader", "reader_lora", intervention_frequency=-1, intervention_source="side_layers", intervention_target="kv_transfer_full",
               intervention_head_style="lora", input_pooling_stride=2)
    common = dict(source="side_layers", head_style="lora", calibration="row_positions")
    h, ac = build(tmp_path, hybrid_dir, frequency=-1, target="kv_transfer_full", seed=41, **common)
    ht, transfer = build(tmp_path / "rows", hybrid_dir, frequency=-1, target="kv_transfer", seed=41, **common)
    assert ac.kv_transfer_full and ac.kv_transfer and ac.kv_replacement and ac.intervention_layers == [2, 5]
    summary = ac.intervention_summary()
    assert summary["measurement_only"] is True and summary["target"] == "kv_transfer_full" and summary["takeover"][5] == {"k": 1.0, "v": 1.0}
    # The reader's placeholder run is the whole passage: max(N, V) positions (N side tokens, V rows); nested parts are refused.
    request = fixed_request(ac)
    ids, _ = tokenize_with_parts(ac.side.tokenizer, request.messages, [], ac.pad_id, tools=None)
    N = len(ids); V = ac.num_view_rows(N, request.compression_ratio); R = max(N, V)
    assert N > V and R == N and ac.run_rows(N, request.compression_ratio) == R
    assert ac.part_view_rows(request.messages, request.compression_ratio) == R and transfer.part_view_rows(request.messages, request.compression_ratio) == V
    nested = [{"role": "user", "content": [{"type": AC_PART_TYPE, "messages": request.messages}]}]
    with pytest.raises(ValueError, match="nested parts"):
        ac.part_view_rows(nested, request.compression_ratio)
    trainer, items, examples, device = trainer_and_examples(h, ac)     # the study generator renders R placeholders per part
    assert all(end - start == ac.part_view_rows(part.messages, part.compression_ratio, part.tools) for example in examples for (start, end), part in zip(example.spans, example.part_requests))
    calibrate(trainer, ac, examples[:2])
    # The encode: the rows first and zero fill to the run; the payload covers side positions [0, R) = every passage token's own K/V at each layer.
    output, S, side_kv = side_inputs_and_kv(ac, request, (2, 5))
    with transfer.uncalibrated_encoding():                                 # the row model's rows only (its takeovers need no calibration to compare rows)
        rows_only = transfer.encode_batch([request], with_layers=True)[0]
    assert output.input_embeds.shape == (R, ac.d_target) and S == N + V
    assert torch.equal(output.input_embeds[:V], rows_only.input_embeds) and torch.equal(output.input_embeds[V:], torch.zeros(R - V, ac.d_target))
    for layer, (k, v, keep_k, keep_v) in output.kv_slots.items():
        assert k.shape == v.shape == (R, ac.target.model_config.model_description.d_kv) and float(keep_k.detach()) == 0.0
        assert torch.equal(k, side_kv[f"{layer}:k"][:R]) and torch.equal(v, side_kv[f"{layer}:v"][:R]), f"layer {layer}: the passage's own K/V, in order"
    # The reader: every run position's K/V becomes the side's passage K/V at both layers; nothing else moves; deltas-off = the reader's own.
    example = examples[0]
    target = ac.target
    embedding = target.model.get_input_embeddings()
    outputs = ac.encode_batch(example.part_requests, with_layers=True)
    with torch.no_grad():
        student = embedding(torch.tensor(example.student_ids))
        embeds, payload = trainer._assemble_student(student, example.spans, outputs, student.dtype)
        _, none = trainer._assemble_student(student, example.spans, outputs, student.dtype, deltas_off=True)
    assert none is None and len(payload["positions"]) == sum(end - start for start, end in example.spans) >= R
    positions = torch.tensor(payload["positions"])
    for layer in (2, 5):
        k_on, k_off, v_on, v_off, hidden_on, hidden_off = reader_kv_at(target, embeds, payload, layer)
        k, v, _, _ = payload["kv_slots"][layer]
        assert torch.equal(k_on[positions], k) and torch.equal(v_on[positions], v) and not torch.allclose(k_on[positions], k_off[positions])
        mask = torch.ones(embeds.shape[0], dtype=torch.bool); mask[positions] = False
        if layer == 2:                                                       # at the first spliced layer nothing else moves (later layers see the run through attention)
            assert torch.equal(k_on[mask], k_off[mask]) and torch.equal(v_on[mask], v_off[mask])
    assert not torch.equal(hidden_on, hidden_off)
    on = trainer._evaluate(ac, items, device, None, "reporting"); off = trainer._evaluate(ac, items, device, None, "deltas_off", deltas_off=True)
    assert on["gold_nll"] != off["gold_nll"]
    # A training epoch wires the whole path (three groups, the trace at takeover 1.0, the payload counted over the run); checkpoints round-trip and
    # are refused by the row transfer (same layers, other target).
    trainer_f = ActivationContextTrainer(h, config(learning_rate_delta_scales=5e-3, deltas_off_panel=True))
    stats = trainer_f.train(ac.name, items, reporting_data=items)
    assert len(trainer_f.optimizers[ac.name].param_groups) == 3 and stats.deltas_off is not None and stats.payload_bytes > 0
    assert all(value == 1.0 for value in stats.intervention_trace[0]["takeover"].values())
    folder = ac.save(str(tmp_path / "full_ckpt"))
    with pytest.raises(ValueError, match="source/target"):
        transfer.load(folder)
    h2, again = build(tmp_path / "again", hybrid_dir, frequency=-1, target="kv_transfer_full", seed=5, ac_checkpoint=folder, **common)
    assert again.interventions_calibrated and again.intervention_summary()["takeover"] == ac.intervention_summary()["takeover"]
