"""Dense supervision (round 7): the full-context base reader's per-layer attention outputs distilled into the row-fed reader on tiny CPU models.
Position alignment, the teacher pass (adapters off, no rows), zero loss when the student is the teacher, the λ schedule and its resume, gradients
from the term alone into the rows' writer and the reader LoRA under checkpointing, the batched path, the held-out panel (rows on / off, the rows'
attention share against eager attention weights), bit-identity of distill_weight 0, and the bench keys and arms.

  PYTHONPATH=. .venv/bin/python -m pytest activation/tests/test_distill.py -q -p no:cacheprovider
"""
import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from test_interventions import (ActivationContextTrainer, ActivationContextTrainingConfig, build, config, hybrid_dir, items_for,   # noqa: F401 (hybrid_dir is a fixture)
                                side_lora_parameters, trainer_and_examples, transfer_parameters)
from activation.ac_model.ac_model_training import DISTILL_PANEL_ITEMS
from activation.harness.hf_utils import attention_shares, capture_attention_qk, capture_layer_outputs, contiguous_runs, decoder_layers

torch.set_num_threads(2)
BENCH = Path(__file__).resolve().parents[2] / "IB/ARTIFACTS/AGENT_AC_PRETRAINING/ac_recon"
TRANSFER = dict(frequency=-1, source="side_layers", target="kv_transfer", head_style="lora", calibration="row_positions")


def reader_lora_parameters(h, ac, layer: int | None = None) -> list[torch.nn.Parameter]:
    """The reader adapter's parameters, optionally those of one reader layer."""
    manager = h.module_manager
    granted = {id(parameter) for parameter in manager.lora_parameters(ac.config.target_model_lora_name)}
    return [parameter for name, parameter in ac.target.model.named_parameters() if id(parameter) in granted and (layer is None or f".layers.{layer}." in name)]


def manual_attention_outputs(target, embeds: torch.Tensor, layers, lora_name, positions) -> dict[int, torch.Tensor]:
    """Plain hooks on `self_attn` (no checkpointing): the attention block's output (after o_proj) at `positions`, per layer."""
    seen = {}
    def remember(layer):
        def hook(module, args, out):
            seen.setdefault(layer, out[0].detach().clone())
        return hook
    handles = [decoder_layers(target.model)[layer].self_attn.register_forward_hook(remember(layer)) for layer in layers]
    try:
        with torch.no_grad():
            target.decoder_forward(embeds[None], None, lora_name=lora_name, gradient_checkpointing=False)
    finally:
        for handle in handles:
            handle.remove()
    return {layer: tensor[0, positions] for layer, tensor in seen.items()}


# --------------------------------------------------------------------------------------------------- 1. config, layers, alignment, the teacher, zero loss
def test_distill_config_layers_alignment_and_teacher(tmp_path, hybrid_dir):
    for wrong in (dict(distill_weight=-1.0), dict(distill_weight=float("nan")), dict(distill_source="hidden"), dict(distill_hold=0.8, distill_decay_end=0.5),
                  dict(distill_floor=2.0), dict(distill_layers=[]), dict(distill_layers=[2, 2]), dict(distill_layers=[-3]), dict(distill_layers="2")):
        with pytest.raises(ValueError):
            config(**wrong)
    assert config(distill_layers=[5, 2]).distill_layers == [2, 5] and config(distill_layers=-1).distill_layers == -1 and config().distill_layers is None
    assert config().distill_weight == 0.0 and config().distill_source == "attn_out" and (config().distill_hold, config().distill_decay_end, config().distill_floor) == (0.25, 0.75, 0.05)
    assert contiguous_runs([5, 6, 7, 12, 3]) == [(3, 4), (5, 8), (12, 13)] and contiguous_runs([]) == []
    # Layers: the input-only model's default is every full-attention layer (2 and 5 on the hybrid); -1 the same; a list as given; linear-attention and out-of-range refused;
    # a deep model's default is its intervention layers.
    h, ac = build(tmp_path, hybrid_dir, seed=21)
    resolve = lambda ac, **overrides: ActivationContextTrainer(ac.harness, config(distill_weight=1.0, **overrides))._resolve_distill_layers(ac)
    assert resolve(ac) == [2, 5] == resolve(ac, distill_layers=-1) and resolve(ac, distill_layers=[5]) == [5]
    with pytest.raises(ValueError, match="full-attention"):
        resolve(ac, distill_layers=[3])
    with pytest.raises(ValueError, match="outside"):
        resolve(ac, distill_layers=[9])
    hd, deep = build(tmp_path / "deep", hybrid_dir, seed=21, **TRANSFER)
    assert deep.intervention_layers == [2, 5] and resolve(deep) == [2, 5]
    assert resolve(build(tmp_path / "one", hybrid_dir, seed=21, layers=[5], **{k: v for k, v in TRANSFER.items() if k != "frequency"})[1]) == [5]
    # Alignment: with distill_weight > 0 the SFT examples carry the in-context (teacher) tokens; at every aligned target position the predicting token and the target agree.
    trainer, items, examples, device = trainer_and_examples(h, ac, distill_weight=1.0)
    assert all(example.teacher_ids and example.teacher_positions for example in examples)
    for example in examples:
        assert len(example.teacher_positions) == len(example.student_positions) == len(example.target_ids)
        assert [example.teacher_ids[p] for p in example.teacher_positions] == [example.student_ids[q] for q in example.student_positions]
        assert [example.teacher_ids[p + 1] for p in example.teacher_positions] == [example.student_ids[q + 1] for q in example.student_positions] == example.target_ids
    assert any(len(example.teacher_ids) != len(example.student_ids) for example in examples), "the passage tokens exist only in the teacher, the rows only in the student"
    with pytest.raises(ValueError, match="not aligned"):
        example = examples[0]
        trainer._check_target_alignment("x", example.teacher_ids, [p + 1 for p in example.teacher_positions], example.student_ids, example.student_positions)
    with pytest.raises(ValueError, match="target positions"):
        trainer._check_target_alignment("x", example.teacher_ids, example.teacher_positions[:-1], example.student_ids, example.student_positions)
    off = ActivationContextTrainer(h, config())
    assert all(not off.build_example(ac, item).teacher_ids for item in items), "distill_weight 0: no teacher tokens for SFT examples"
    # The teacher: the base with the adapters disabled and no rows on the full-context history, equal to a manual forward with plain hooks; unchanged by the reader adapter.
    trainer._distill_layers = trainer._resolve_distill_layers(ac)
    example = examples[0]
    target = ac.target
    embedding = target.model.get_input_embeddings()
    teacher_ids = torch.tensor(example.teacher_ids, device=device)
    teacher = trainer._distill_teacher(ac, [example], device)[0]
    manual = manual_attention_outputs(target, embedding(teacher_ids), (2, 5), None, example.teacher_positions)
    assert set(teacher) == {2, 5} and all(torch.equal(teacher[layer], manual[layer]) for layer in (2, 5))
    assert all(tensor.shape == (len(example.target_ids), ac.d_target) and not tensor.requires_grad for tensor in teacher.values())
    with torch.no_grad():
        for parameter in reader_lora_parameters(h, ac):
            parameter.add_(torch.randn_like(parameter) * 0.1)
    again = trainer._distill_teacher(ac, [example], device)[0]
    assert all(torch.equal(teacher[layer], again[layer]) for layer in (2, 5)), "the teacher never sees the reader adapter"
    adapted = manual_attention_outputs(target, embedding(teacher_ids), (2, 5), "reader_lora", example.teacher_positions)
    assert not torch.allclose(adapted[5], manual[5])
    # Batched teacher (right-padded rows) equals the single-example teacher.
    batched = trainer._distill_teacher(ac, examples, device)
    for row, other in enumerate(examples):
        single = trainer._distill_teacher(ac, [other], device)[0]
        for layer in (2, 5):
            torch.testing.assert_close(batched[row][layer], single[layer], atol=1e-5, rtol=1e-5)
    # Zero loss when the student is the teacher: the base with no rows on the full context, captured under the checkpointed reader forward; positive for the real student.
    trainer._distill_lambda = 1.0
    with capture_layer_outputs(target.model, [2, 5], [contiguous_runs(example.teacher_positions)]) as captured:
        target.decoder_forward(embedding(teacher_ids)[None], None, lora_name=None, gradient_checkpointing=True)
    term, values = trainer._distill_terms(captured, 0, teacher)
    assert values == {2: 0.0, 5: 0.0} and float(term) == 0.0          # (nothing requires grad with the adapters disabled: the base is frozen)
    trainer._example_loss(ac, example, device)
    real, real_values = trainer._distill_term
    assert all(value > 0 for value in real_values.values()) and float(real.detach()) == pytest.approx(sum(real_values.values()) / 2)
    zeros = {layer: [[torch.zeros_like(teacher[layer])]] for layer in (2, 5)}
    assert trainer._distill_terms(zeros, 0, teacher)[1] == {2: pytest.approx(1.0), 5: pytest.approx(1.0)}, "a student that writes nothing scores exactly 1 per layer"


# --------------------------------------------------------------------------------------------------- 2. the schedule and its resume
def test_distill_schedule_and_resume(tmp_path, hybrid_dir):
    h, ac = build(tmp_path, hybrid_dir, seed=8)
    trainer = ActivationContextTrainer(h, config(distill_weight=2.0))
    values = [trainer._distill_lambda_at(update, 8) for update in range(8)]
    assert values == pytest.approx([2.0, 2.0, 2.0, 1.525, 1.05, 0.575, 0.1, 0.1])       # hold through 0.25, linear to the floor 0.05 x 2 at 0.75, then the floor
    assert trainer._distill_lambda_at(100, 8) == pytest.approx(0.1) and trainer._distill_lambda_at(0, None) == 2.0
    step = ActivationContextTrainer(h, config(distill_weight=1.0, distill_hold=0.5, distill_decay_end=0.5, distill_floor=0.0))
    assert [step._distill_lambda_at(update, 4) for update in range(4)] == [1.0, 1.0, 1.0, 0.0]
    # A run of 2 updates on an 8-update clock holds λ; the resumed run (a new harness loading the epoch pair with resume_optimizer) continues the decay from update 2.
    items = items_for(h, ac)
    trainer = ActivationContextTrainer(h, config(distill_weight=2.0, schedule_total_updates=8, checkpoint_every_epoch=True))
    stats = trainer.train(ac.name, items)
    assert stats.distill_lambda == [2.0, 2.0] and stats.distill_layers == [2, 5] and len(stats.distill_loss) == 2 and set(stats.distill_loss[0]) == {2, 5}
    assert trainer.updates_done[ac.name] == 2 and (Path(stats.checkpoint_path) / "optimizer.pt").exists()
    h2, ac2 = build(tmp_path, hybrid_dir, seed=9, ac_checkpoint=stats.checkpoint_path, reader_checkpoint=stats.target_lora_checkpoint_path)
    trainer2 = ActivationContextTrainer(h2, config(distill_weight=2.0, schedule_total_updates=8, checkpoint_every_epoch=True, resume_optimizer=True, num_epochs=2))
    stats2 = trainer2.train(ac2.name, items_for(h2, ac2))
    assert stats2.resumed_from == str(Path(stats.checkpoint_path) / "optimizer.pt") and stats2.global_steps == [3, 4, 5, 6]
    assert stats2.distill_lambda == pytest.approx([2.0, 1.525, 1.05, 0.575]), "λ continues on the restored update clock"


# --------------------------------------------------------------------------------------------------- 3. gradients: the term alone reaches the writer and the reader LoRA; checkpointed == direct; batched == single
@pytest.mark.parametrize("kind", ["input_only", "kv_transfer"])
def test_distill_gradients_reach_writer_and_reader(tmp_path, hybrid_dir, kind):
    h, ac = build(tmp_path, hybrid_dir, seed=31, **(TRANSFER if kind == "kv_transfer" else {}))
    trainer, items, examples, device = trainer_and_examples(h, ac, distill_weight=1.0, learning_rate_target_lora=1e-2)
    if ac.is_deep:
        from test_interventions import calibrate
        calibrate(trainer, ac, examples[:2])
    trainer._distill_layers = trainer._resolve_distill_layers(ac)
    trainer._distill_lambda = 1.0
    example = examples[0]
    encoder = [parameter for parameter in ac.trainable_parameters()]
    reader_low = reader_lora_parameters(h, ac, layer=2)
    reader_high = reader_lora_parameters(h, ac, layer=5)
    watched = [*encoder, *reader_low, *reader_high]
    # The terminal loss plays no part: only the distillation term is differentiated (terminal weight 0 in this unit).
    trainer._example_loss(ac, example, device)
    term, _ = trainer._distill_term
    checkpointed = torch.autograd.grad(term, watched, allow_unused=True)
    ac._training_gradient_checkpointing = False
    try:
        trainer._example_loss(ac, example, device)
        direct = torch.autograd.grad(trainer._distill_term[0], watched, allow_unused=True)
    finally:
        ac._training_gradient_checkpointing = True
    norms = []
    for parameter, g_ckpt, g_direct in zip(watched, checkpointed, direct):
        assert (g_ckpt is None) == (g_direct is None)
        if g_ckpt is not None:
            torch.testing.assert_close(g_ckpt, g_direct, atol=1e-6, rtol=1e-5)
        norms.append(float(g_ckpt.norm()) if g_ckpt is not None else 0.0)
    encoder_norms, low_norms, high_norms = norms[:len(encoder)], norms[len(encoder):len(encoder) + len(reader_low)], norms[len(encoder) + len(reader_low):]
    assert any(norm > 0 for norm in encoder_norms), "the rows' writer (the encoder) hears the distillation term"
    assert any(norm > 0 for norm in low_norms) and any(norm > 0 for norm in high_norms), "the reader LoRA at layers <= the supervised layers hears it"
    if ac.is_deep:
        trainer._example_loss(ac, example, device)
        heads = torch.autograd.grad(trainer._distill_term[0], [*transfer_parameters(ac), *side_lora_parameters(ac, layer=5)], allow_unused=True)
        assert all(g is not None and float(g.norm()) > 0 for g in heads[:len(transfer_parameters(ac))]), "the K/V takeovers hear the term alone"
        assert any(g is not None and float(g.norm()) > 0 for g in heads[len(transfer_parameters(ac)):]), "the side adapter hears it through the transferred K/V"
    # The batched path (right-padded micro-batch) gives the per-example terms.
    single = {}
    for other in examples:
        trainer._example_loss(ac, other, device)
        single[other.item.item_id] = trainer._distill_term[1]
    batched = ActivationContextTrainer(h, config(distill_weight=1.0, micro_batch_examples=3))
    batched._distill_layers, batched._distill_lambda = trainer._distill_layers, 1.0
    batched.teacher_cache = trainer.teacher_cache
    for other, result in zip(examples, batched._batch_loss(ac, examples, device)):
        for layer, value in result["distill"][1].items():
            assert value == pytest.approx(single[other.item.item_id][layer], rel=1e-4, abs=1e-6), (other.item.item_id, layer)
        assert float(result["distill"][0].detach()) == pytest.approx(sum(result["distill"][1].values()) / len(result["distill"][1]))


# --------------------------------------------------------------------------------------------------- 4. training: the term changes the update; distill_weight 0 leaves everything as before
def test_distill_training_and_off_is_inert(tmp_path, hybrid_dir):
    h, ac = build(tmp_path, hybrid_dir, seed=12)
    h0, off = build(tmp_path / "off", hybrid_dir, seed=12)
    before = {name: tensor.clone() for name, tensor in ac.modules.state_dict().items()}
    trainer_on, trainer_off = ActivationContextTrainer(h, config(distill_weight=1.0, schedule_total_updates=8)), ActivationContextTrainer(h0, config(schedule_total_updates=8))
    torch.manual_seed(1); stats = trainer_on.train(ac.name, items_for(h, ac), reporting_data=items_for(h, ac))
    torch.manual_seed(1); stats0 = trainer_off.train(off.name, items_for(h0, off), reporting_data=items_for(h0, off))
    assert stats.distill_lambda == [1.0, 1.0] and all(set(entry) == {2, 5} and all(value > 0 for value in entry.values()) for entry in stats.distill_loss)
    assert stats0.distill_lambda == [] and stats0.distill_loss == [] and stats0.distill_panel == [] and stats0.distill_layers == []
    assert set(ac.modules.state_dict()) == set(off.modules.state_dict()) == set(before), "no new state-dict keys either way"
    layout = lambda trainer, model: [len(group["params"]) for group in trainer.optimizers[model.name].param_groups]
    assert layout(trainer_on, ac) == layout(trainer_off, off), "no new optimizer group"
    moved = [name for name in before if not torch.equal(before[name], ac.modules.state_dict()[name])]
    assert moved and any(not torch.equal(ac.modules.state_dict()[name], off.modules.state_dict()[name]) for name in moved), "the term changes the update"
    assert stats.teacher_tokens > 0 and stats0.teacher_tokens == 0, "the teacher view is tokenised only when the term is on (SFT)"
    # The held-out panel: baseline + one panel, rows on / off and the attention share per layer, all in range; the panel never fires with the term off.
    assert [entry["update"] for entry in stats.distill_panel] == [0, 2]
    for entry in stats.distill_panel:
        assert set(entry["rows_on"]) == set(entry["rows_off"]) == set(entry["rows_attention_share"]) == {2, 5}
        assert all(0 <= value <= 1 for value in entry["rows_attention_share"].values()) and all(value > 0 for value in entry["rows_on"].values())
    summary = stats.summarize()
    assert summary["distill_layers"] == [2, 5] and summary["distill_lambda_last"] == 1.0 and summary["distill_panel"] == stats.distill_panel and "distill_layers" not in stats0.summarize()
    # The reporter takes the widgets.
    from activation.ac_model import ActivationContextTrainingReporter
    reporter = ActivationContextTrainingReporter(str(tmp_path / "report"), "distill")
    hr, again = build(tmp_path / "rep", hybrid_dir, seed=12)
    ActivationContextTrainer(hr, config(distill_weight=1.0)).train(again.name, items_for(hr, again), reporting_data=items_for(hr, again), reporter=reporter)
    assert {"distill_loss", "distill_lambda", "distill_panel", "rows_attention_share"} <= set(reporter.widgets)
    assert {series["name"] for series in reporter.widgets["distill_panel"]["series"]} == {f"layer {layer} ({label})" for layer in (2, 5) for label in ("rows on", "rows off")}
    assert {series["name"] for series in reporter.widgets["distill_loss"]["series"]} == {"layer 2", "layer 5"}


# --------------------------------------------------------------------------------------------------- 5. the rows' attention share equals the model's own (eager) attention weights
def test_rows_attention_share_matches_eager_attention(tmp_path, hybrid_dir):
    h, ac = build(tmp_path, hybrid_dir, seed=5, **TRANSFER)          # a K/V payload: the share must read the K the payload wrote
    trainer, items, examples, device = trainer_and_examples(h, ac, distill_weight=1.0)
    from test_interventions import calibrate
    calibrate(trainer, ac, examples[:2])
    trainer._distill_layers = [2, 5]
    example = examples[0]
    target = ac.target
    embedding = target.model.get_input_embeddings()
    with torch.no_grad():
        outputs = ac.encode_batch(example.part_requests, with_layers=True)
        embeds, payload = trainer._assemble_student(embedding(torch.tensor(example.student_ids)), example.spans, outputs, torch.float32, ac_model=ac)
    rows = [position for start, end in example.spans for position in range(start, end)]
    attention = {layer: decoder_layers(target.model)[layer].self_attn for layer in (2, 5)}
    previous = {layer: module.config._attn_implementation for layer, module in attention.items()}
    weights = {}
    def remember(layer):
        def hook(module, args, out):
            weights.setdefault(layer, out[1].detach())
        return hook
    handles = [module.register_forward_hook(remember(layer)) for layer, module in attention.items()]
    try:
        for module in attention.values():
            module.config._attn_implementation = "eager"
        with torch.no_grad(), capture_attention_qk(target.model, [2, 5]) as captured:
            target.decoder_forward(embeds[None], None, lora_name="reader_lora", interventions=[payload])
    finally:
        for handle in handles:
            handle.remove()
        for layer, module in attention.items():
            module.config._attn_implementation = previous[layer]
    for layer in (2, 5):
        manual = attention_shares(target.model, captured, layer, example.student_positions, rows)              # [heads, targets]
        eager = weights[layer][0][:, example.student_positions][:, :, rows].sum(dim=-1)
        assert manual.shape == eager.shape == (4, len(example.student_positions))
        torch.testing.assert_close(manual, eager, atol=1e-5, rtol=1e-5)
        assert 0 < float(manual.mean()) < 1
    panel = trainer._distill_panel(ac, items * (DISTILL_PANEL_ITEMS // len(items) + 1), device)        # more items than the cap: the first DISTILL_PANEL_ITEMS with rows
    assert set(panel) == {"rows_on", "rows_off", "rows_attention_share"} and set(panel["rows_attention_share"]) == {2, 5}
    assert panel["rows_on"][2] > 0 and panel["rows_off"][2] > 0 and panel["rows_on"] != panel["rows_off"]


# --------------------------------------------------------------------------------------------------- 6. bench keys and arms
def test_bench_distill_keys_and_arms():
    runner = BENCH / "run_recon.py"
    if not runner.exists():
        pytest.skip("bench not present")
    spec = importlib.util.spec_from_file_location("run_recon", runner)
    run_recon = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_recon)
    default = run_recon.DEFAULT_ARM
    assert (default["distill_weight"], default["distill_layers"], default["distill_source"], default["distill_hold"], default["distill_decay_end"], default["distill_floor"]) == (0.0, None, "attn_out", 0.25, 0.75, 0.05)
    assert default["intervention_layers"] is None
    training = run_recon.config(dict(default, distill_weight=1.0, distill_layers=[23, 31], distill_hold=0.1))
    assert training.distill_weight == 1.0 and training.distill_layers == [23, 31] and training.distill_hold == 0.1 and training.distill_floor == 0.05
    assert run_recon.config(default).distill_weight == 0.0
    with pytest.raises(ValueError, match="interventions 0"):
        run_recon.harness(True, interventions=4, intervention_layers=[23, 27, 31])
    arms = run_recon.arms()
    strip = lambda arm, *keys: {key: value for key, value in arm.items() if key not in ("note", *keys)}
    k16r1 = {"init_ac": arms["A4X"]["init_ac"], "init_reader": arms["A4X"]["init_reader"]}
    k16 = {"init_ac": arms["NDIAWLn"]["init_ac"], "init_reader": arms["NDIAWLn"]["init_reader"]}
    assert "/home/ngom/" in k16r1["init_ac"] and "/root/" in k16["init_ac"]
    for label, arm in arms.items():                      # every arm (defaults merged) builds a training config with its distillation keys
        expected = run_recon.config(arm)
        assert expected.distill_weight == arm["distill_weight"] and expected.distill_layers in (None, -1, arm["distill_layers"]), label
    raw = json.loads((BENCH / "arms.json").read_text())
    for label in ("A32XD", "A32XD3", "A32XDR", "NXD", "A32KVTAD", "A32KVTL", "NKVTL"):
        for arm in (raw[label], raw[label + "n"]):
            for key in ("distill_source", "distill_hold", "distill_decay_end", "distill_floor"):
                assert key not in arm, f"{label}: the schedule keys keep their defaults"
    assert all(arms[label]["distill_weight"] == 1.0 for label in ("A32XD", "A32XDR", "NXD", "A32KVTAD", "A32XDn", "A32XDRn", "NXDn", "A32KVTADn")) and arms["A32XD3"]["distill_weight"] == 3.0
    assert all(arms[label]["distill_weight"] == 0.0 and arms[label]["intervention_layers"] == [23, 27, 31] for label in ("A32KVTL", "NKVTL", "A32KVTLn", "NKVTLn"))
