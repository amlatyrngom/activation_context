"""Round 8 write-side features on tiny CPU models: the iterative encoder (`encoder_passes`: a zero-initialised refine adapter, the rows and the
side K/V from the last pass, the passes_off panel and the refine / pass-cosine traces), the surprisal-weighted terminal loss (`rare_weight`: per-token
weights from the stored base no-context surprisal, the metric unweighted) and the surprisal-weighted summary pooling (`pool_surprisal`: mass bins and a
zero-initialised per-head logit bias), their bit-identity at the defaults, checkpoints, and the bench keys and arms.

  PYTHONPATH=. .venv/bin/python -m pytest activation/tests/test_encoder_passes.py -q -p no:cacheprovider
"""
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path

import pytest
import torch

from test_interventions import (ActivationContextTrainer, ActivationContextTrainingConfig, HarnessRuntime, HarnessRuntimeConfig, ModelConfig,   # noqa: F401
                                calibrate, config, hybrid_dir, items_for, side_lora_parameters, trainer_and_examples)
from activation.ac_model import ActivationContextModelConfig
from activation.ac_model.ac_model_utils import AdaptiveSummaryPooling, direct_parts, mass_bins, tokenize_with_parts
from activation.harness.hf_utils import decoder_layers

torch.set_num_threads(2)
BENCH = Path(__file__).resolve().parents[2] / "IB/ARTIFACTS/AGENT_AC_PRETRAINING/ac_recon"
TRANSFER = dict(frequency=-1, source="side_layers", target="kv_transfer", head_style="lora", calibration="row_positions")


def build8(tmp_path: Path, model_dir: Path, *, passes: int = 1, pool: float = 0.0, seed: int = 1234, ac_checkpoint: str | None = None, frequency: int = 0,
           source: str = "summary", target: str = "residual", head_style: str = "scaled", calibration: str = "no_rows"):
    """test_interventions.build with the round-8 model settings (the same construction order: the AC modules draw first, the adapters at injection)."""
    os.environ["ACTIVATION_SYNC_ROOT"] = str(tmp_path)
    torch.manual_seed(seed)
    h = HarnessRuntime(HarnessRuntimeConfig(model_configs={"reader": ModelConfig("reader", str(model_dir))}))
    h.module_manager.register_lora("reader_lora", "reader", rank=4, dropout=0.0)
    deep = {"intervention_frequency": frequency, "intervention_source": source, "intervention_target": target, "intervention_head_style": head_style,
            "intervention_calibration": calibration} if frequency else {}
    ac = h.module_manager.register_ac_model(ActivationContextModelConfig(
        "ac_private", "reader", "side_lora", "reader", "reader_lora", mixer_heads=4, mixer_window=4, side_lora_rank=4, max_view_rows=8,
        encode_batch_max_tokens=2048, checkpoint_path=ac_checkpoint, encoder_passes=passes, pool_surprisal=pool, **deep))
    h.module_manager.ensure_lora("reader_lora")
    ac.prepare()
    for loaded in h.loaded_models.values():
        loaded.model.float()
    return h, ac


def perturb_refine(ac, std: float = 0.3) -> None:
    """Moves the zero-initialised output projection so pass 2 differs from pass 1 (and gradient flows through the second pass)."""
    with torch.no_grad():
        torch.manual_seed(99)
        ac.modules.refine_adapter.up.weight.normal_(std=std)
        ac.modules.refine_adapter.up.bias.normal_(std=std)


def rows_of(ac, request) -> torch.Tensor:
    with torch.no_grad():
        return ac.encode_batch([request], with_layers=True)[0].input_embeds.clone()


# --------------------------------------------------------------------------------------------------- 1. config, modules, state dict, checkpoints
def test_encoder_passes_config_modules_and_checkpoints(tmp_path, hybrid_dir):
    Config = ActivationContextModelConfig
    for bad in ({"encoder_passes": 0}, {"encoder_passes": 1.5}, {"pool_surprisal": -1.0}, {"pool_surprisal": float("nan")},
                {"encoder_passes": 2, "intervention_frequency": -1, "intervention_source": "side_layers", "intervention_target": "kv_transfer_full", "intervention_head_style": "lora"},
                {"encoder_passes": 2, "intervention_frequency": 1, "intervention_target": "cross_attention"}):
        with pytest.raises(ValueError):
            Config("x", "reader", "side_lora", "reader", "reader_lora", **bad)
    Config("x", "reader", "side_lora", "reader", "reader_lora", encoder_passes=3, intervention_frequency=-1, intervention_source="side_layers",
           intervention_target="kv_transfer", intervention_head_style="lora")
    h1, one = build8(tmp_path / "one", hybrid_dir, passes=1, seed=3)
    h2, two = build8(tmp_path / "two", hybrid_dir, passes=2, seed=3)
    hp, pooled = build8(tmp_path / "pool", hybrid_dir, pool=1.0, seed=3)
    keys_one, keys_two, keys_pool = set(one.modules.state_dict()), set(two.modules.state_dict()), set(pooled.modules.state_dict())
    refine_keys = {"refine_adapter.down.weight", "refine_adapter.down.bias", "refine_adapter.up.weight", "refine_adapter.up.bias"}
    assert keys_two - keys_one == refine_keys and keys_one <= keys_two, "the refine adapter's keys exist only with encoder_passes >= 2"
    assert keys_pool - keys_one == {"summary_pooling.surprisal_bias"} and keys_one <= keys_pool, "the surprisal bias exists only with pool_surprisal > 0"
    assert one.modules.refine_adapter is None and one.modules.summary_pooling.surprisal_bias is None
    assert torch.equal(two.modules.refine_adapter.up.weight, torch.zeros_like(two.modules.refine_adapter.up.weight)) and torch.equal(two.modules.refine_adapter.up.bias, torch.zeros(32))
    assert torch.equal(pooled.modules.summary_pooling.surprisal_bias, torch.zeros(4))
    assert two.modules.refine_adapter.down.weight.shape == (32, 32) and two.modules.refine_adapter.down.weight.abs().sum() > 0, "rank min(256, d_side)"
    for name in keys_one:                                       # the forked stream: every shared module draws what it draws at encoder_passes 1 / pool_surprisal 0
        assert torch.equal(one.modules.state_dict()[name], two.modules.state_dict()[name]) and torch.equal(one.modules.state_dict()[name], pooled.modules.state_dict()[name]), name
    for a, b in zip(side_lora_parameters(one), side_lora_parameters(two)):
        assert torch.equal(a, b), "the adapters draw the same sequence"
    trainable = {id(parameter) for parameter in two.trainable_parameters()}
    assert all(id(parameter) in trainable for parameter in two.modules.refine_adapter.parameters()) and all(parameter.numel() > 1 for parameter in two.modules.refine_adapter.parameters()), \
        "trainable, and no scalar parameter (they would land in the gate group)"
    assert id(pooled.modules.summary_pooling.surprisal_bias) in {id(parameter) for parameter in pooled.trainable_parameters()} and pooled.modules.summary_pooling.surprisal_bias.numel() == 4
    # Checkpoints: a single-pass checkpoint loads into a two-pass model (the adapter stays at zero) and into a pooling model (the bias stays at zero); the reverse is refused.
    one.modules.summary_marker.data.add_(0.5)
    saved = one.save(str(tmp_path / "ckpt_one"))
    two.load(saved); pooled.load(saved)
    assert torch.equal(two.modules.summary_marker, one.modules.summary_marker) and torch.equal(two.modules.refine_adapter.up.weight, torch.zeros(32, 32))
    assert torch.equal(pooled.modules.summary_marker, one.modules.summary_marker) and torch.equal(pooled.modules.summary_pooling.surprisal_bias, torch.zeros(4))
    perturb_refine(two)
    saved_two = two.save(str(tmp_path / "ckpt_two"))
    with pytest.raises(ValueError, match="refine adapter"):
        one.load(saved_two)
    h3, three = build8(tmp_path / "three", hybrid_dir, passes=3, seed=3)
    three.load(saved_two)                                        # an adapter trained at 2 passes loads into 3 (the same mechanism, more passes)
    assert torch.equal(three.modules.refine_adapter.up.weight, two.modules.refine_adapter.up.weight)
    pooled.modules.summary_pooling.surprisal_bias.data.fill_(0.25)
    saved_pool = pooled.save(str(tmp_path / "ckpt_pool"))
    with pytest.raises(ValueError, match="surprisal pooling bias"):
        one.load(saved_pool)


# --------------------------------------------------------------------------------------------------- 2. passes 2 == passes 1 at init; gradients; checkpointed == direct
def test_two_passes_equal_one_at_init_then_move(tmp_path, hybrid_dir):
    h1, one = build8(tmp_path / "one", hybrid_dir, passes=1, seed=5)
    h2, two = build8(tmp_path / "two", hybrid_dir, passes=2, seed=5)
    part = direct_parts(items_for(h1, one)[0].ac_messages)[0]
    request = one._child_request(part, 0.25)._replace(is_recursive=False)
    assert torch.equal(rows_of(one, request), rows_of(two, request)), "an exact re-read at init"
    # Gradient: the terminal loss reaches the refine adapter's output projection at init (the LoRA-style bootstrap: `down` gets none until `up` moves).
    trainer, items, examples, device = trainer_and_examples(h2, two)
    two.trainable_parameters()                                   # the manager grants the side adapter's trainability on request
    loss, *_ = trainer._example_loss(two, examples[0], device)
    loss.backward()
    adapter = two.modules.refine_adapter
    assert adapter.up.weight.grad is not None and adapter.up.weight.grad.abs().sum() > 0 and adapter.up.bias.grad.abs().sum() > 0
    assert adapter.down.weight.grad is None or adapter.down.weight.grad.abs().sum() == 0
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in side_lora_parameters(two))
    for parameter in two.trainable_parameters():
        parameter.grad = None
    # One optimizer step each, the same seed: the two-pass rows leave the one-pass rows, the adapter moved; side_tokens count a part once whatever the passes.
    torch.manual_seed(1); stats_one = ActivationContextTrainer(h1, config(examples_per_update=3)).train(one.name, items_for(h1, one))
    torch.manual_seed(1); stats_two = ActivationContextTrainer(h2, config(examples_per_update=3)).train(two.name, items_for(h2, two))
    assert stats_one.steps == stats_two.steps == 1 and stats_one.side_tokens == stats_two.side_tokens > 0
    assert not torch.equal(adapter.up.weight, torch.zeros_like(adapter.up.weight))
    assert not torch.equal(rows_of(one, request), rows_of(two, request))
    assert stats_two.summarize()["encoder_passes"] == 2 and stats_one.summarize()["encoder_passes"] == 1
    # Checkpointed vs direct gradients under two passes (the adapter open): the side LoRA, the adapter and the target head agree.
    perturb_refine(two)
    trainer, items, examples, device = trainer_and_examples(h2, two)
    def gradients(checkpointed: bool) -> list[torch.Tensor]:
        two._training_gradient_checkpointing = checkpointed
        for parameter in two.trainable_parameters():
            parameter.grad = None
        loss, *_ = trainer._example_loss(two, examples[1], device)
        loss.backward()
        watched = [*two.modules.refine_adapter.parameters(), *two.modules.target_head.parameters(), *side_lora_parameters(two, layer=2)]
        return [parameter.grad.detach().clone() if parameter.grad is not None else torch.zeros_like(parameter) for parameter in watched]
    with_checkpoint, direct = gradients(True), gradients(False)
    assert any(gradient.abs().sum() > 0 for gradient in with_checkpoint[:4]), "the adapter takes gradient once open"
    for a, b in zip(with_checkpoint, direct):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
    # A padded batch of two parts (two passes each) gives every part its single-encode rows.
    requests = [example.part_requests[0] for example in examples[:2]]
    with torch.no_grad():
        pair = two.encode_batch(requests, with_layers=True)
        singles = [two.encode_batch([request], with_layers=True)[0] for request in requests]
    assert len({request.messages[0]["content"][0]["text"] if isinstance(request.messages[0]["content"], list) else str(request.messages) for request in requests}) == 2
    for batched, single in zip(pair, singles):
        torch.testing.assert_close(batched.input_embeds, single.input_embeds, rtol=1e-4, atol=1e-4)


# --------------------------------------------------------------------------------------------------- 3. the side K/V come from the last pass
def test_kv_transfer_capture_is_the_last_pass(tmp_path, hybrid_dir):
    h, ac = build8(tmp_path, hybrid_dir, passes=2, seed=7, **TRANSFER)
    trainer, items, examples, device = trainer_and_examples(h, ac)
    calibrate(trainer, ac, examples[:2])
    perturb_refine(ac)
    request = examples[0].part_requests[0]
    attention = decoder_layers(ac.side.model)[2].self_attn
    decoder = ac.side.model.get_decoder()
    seen, inputs = [], []
    handles = [attention.k_norm.register_forward_hook(lambda module, args, output: seen.append(output.detach().clone())),
               decoder.register_forward_pre_hook(lambda module, args, kwargs: inputs.append(kwargs["inputs_embeds"].detach().clone()), with_kwargs=True)]
    try:
        with torch.no_grad():
            output = ac.encode_batch([request], with_layers=True)[0]
    finally:
        for handle in handles:
            handle.remove()
    assert len(seen) == 2 and len(inputs) == 2, "two side passes, one k_norm call each"
    assert not torch.equal(inputs[0], inputs[1]) and torch.equal(inputs[0][0, :-output.input_embeds.shape[0]], inputs[1][0, :-output.input_embeds.shape[0]]), \
        "pass 2 changes the summary inputs only"
    num_rows = output.input_embeds.shape[0]
    content_k = output.kv_slots[2][0]                            # takeover 1 at init, no correction: the captured side K itself
    assert torch.equal(content_k, seen[1][0, -num_rows:].reshape(num_rows, -1)) and not torch.equal(content_k, seen[0][0, -num_rows:].reshape(num_rows, -1)), \
        "the transferred K is the LAST pass's k_norm output at the rows"
    # ... and equals a standalone side forward over the pass-2 inputs.
    again = []
    handle = attention.k_norm.register_forward_hook(lambda module, args, out: again.append(out.detach().clone()))
    try:
        with torch.no_grad():
            ac.side.decoder_forward(inputs[1], None, lora_name=ac.config.base_side_model_lora_name, gradient_checkpointing=False)
    finally:
        handle.remove()
    assert torch.equal(again[0][0, -num_rows:].reshape(num_rows, -1), content_k)
    # Training mode (checkpointed): the capture still fires once, on the last pass, and the loss reaches the adapter through the transferred K/V.
    for parameter in ac.trainable_parameters():
        parameter.grad = None
    loss, *_ = trainer._example_loss(ac, examples[0], device)
    loss.backward()
    assert ac.modules.refine_adapter.down.weight.grad.abs().sum() > 0


# --------------------------------------------------------------------------------------------------- 4. the passes_off panel and the traces
def test_passes_off_panel_and_encoder_traces(tmp_path, hybrid_dir):
    from activation.ac_model import ActivationContextTrainingReporter
    h, ac = build8(tmp_path, hybrid_dir, passes=2, seed=11)
    trainer = ActivationContextTrainer(h, config())
    items = items_for(h, ac)
    device = ac.prepare(); h.module_manager.ensure_lora("reader_lora")
    examples = [trainer.build_example(ac, item) for item in items]
    trainer._store_teacher_targets(ac, [example.penalty for example in examples], device, lora=None)
    on, off = trainer._evaluate(ac, items, device), trainer._evaluate(ac, items, device, passes_off=True)
    assert on["loss"] == off["loss"] and "_encoder_passes_override" not in ac.__dict__, "passes off equals passes on at init (an exact re-read); the override is cleared"
    reporter = ActivationContextTrainingReporter(str(tmp_path / "report"), "passes")
    stats = ActivationContextTrainer(h, config(examples_per_update=1, num_epochs=2)).train(ac.name, items, reporting_data=items, reporter=reporter)
    assert stats.steps == 6 and stats.passes_off is not None and stats.passes_off["items"] == len(items) and stats.epochs[-1].passes_off is stats.passes_off
    assert stats.passes_off["loss"] != stats.reporting["loss"], "after training the second pass does something"
    assert [entry["update"] for entry in stats.encoder_trace] == [0, 5] and all(set(entry) == {"update", "refine_rms", "pass_cosine"} for entry in stats.encoder_trace)
    assert stats.encoder_trace[0]["refine_rms"] == 0.0 and stats.encoder_trace[0]["pass_cosine"] == pytest.approx(1.0, abs=1e-6), "at init: no refinement, identical rows"
    assert stats.encoder_trace[-1]["refine_rms"] > 0 and stats.encoder_trace[-1]["pass_cosine"] < 1.0
    assert [entry["update"] for entry in stats.encoder_panel] == [0, 3, 6] and stats.encoder_panel[0]["refine_rms"] == 0.0 and stats.encoder_panel[-1]["refine_rms"] > 0
    summary = stats.summarize()
    assert summary["encoder_passes"] == 2 and summary["passes_off"] == stats.passes_off and summary["encoder_trace_last"] == stats.encoder_trace[-1] and summary["encoder_panel"] == stats.encoder_panel
    assert {"refine_rms", "pass_cosine", "refine_rms_panel", "pass_cosine_panel"} <= set(reporter.widgets) and "pool_bins" not in reporter.widgets
    assert "passes_off" in {series["name"] for series in reporter.widgets["reporting"]["series"]}
    # A single-pass model records nothing of this.
    h1, one = build8(tmp_path / "one", hybrid_dir, passes=1, seed=11)
    stats1 = ActivationContextTrainer(h1, config(examples_per_update=1)).train(one.name, items_for(h1, one), reporting_data=items_for(h1, one))
    assert stats1.passes_off is None and stats1.encoder_trace == [] and stats1.encoder_panel == [] and "passes_off" not in stats1.summarize() and one.intervention_diagnostics["encoder"] == {}


# --------------------------------------------------------------------------------------------------- 5. mass bins and the surprisal pooling module
def test_mass_bins_and_surprisal_pooling():
    for length, num_rows in ((10, 4), (7, 3), (6, 3), (5, 8), (1, 3), (64, 8), (13, 13)):
        ordinal = torch.arange(num_rows)
        starts, ends = mass_bins(torch.ones(length), num_rows)
        assert torch.equal(starts, ordinal * length // num_rows) and torch.equal(ends, ((ordinal + 1) * length + num_rows - 1) // num_rows), (length, num_rows)
    torch.manual_seed(0)
    for length, num_rows in ((40, 5), (100, 7), (3, 6), (200, 200)):
        mass = 1.0 + torch.rand(length) * 5
        starts, ends = mass_bins(mass, num_rows)
        assert bool((ends > starts).all()) and int(starts[0]) == 0 and int(ends[-1]) == length, "nonempty bins from the first token to the last"
        assert bool((starts[1:] >= starts[:-1]).all()) and bool((ends[1:] >= ends[:-1]).all()), "ordered"
        assert bool((starts[1:] <= ends[:-1]).all()) and bool((ends[:-1] - starts[1:] <= 1).all()), "no gap, at most one token of overlap"
        covered = torch.zeros(length, dtype=torch.long)
        for start, end in zip(starts.tolist(), ends.tolist()):
            covered[start:end] += 1
        assert bool((covered >= 1).all())
        # a bin's mass share: a heavy region gets more rows than a light one of the same length
    mass = torch.cat([torch.full((50,), 0.1), torch.full((50,), 3.0)])
    starts, ends = mass_bins(mass, 10)
    assert int((starts >= 50).sum()) >= 7, "the surprising half takes most of the rows"
    torch.manual_seed(1)
    plain, weighted = AdaptiveSummaryPooling(16, 4), AdaptiveSummaryPooling(16, 4, surprisal=1.0)
    weighted.load_state_dict({**plain.state_dict(), "surprisal_bias": torch.zeros(4)})
    rows = torch.randn(23, 16)
    uniform = torch.full((23,), 2.5)
    assert torch.allclose(plain(rows, 5), weighted(rows, 5, uniform), atol=1e-6), "uniform surprisal + zero bias = the token-count pooling"
    assert torch.equal(plain(rows, 5), weighted(rows, 5)), "no surprisal given: the token-count path"
    surprisal = torch.rand(23) * 4
    starts, ends, width, relative = weighted.bins(23, 5, surprisal)
    assert width == int((ends - starts).max()) and width <= 2 * ((23 + 4) // 5 + 1) + 2 and relative.mean() == pytest.approx(1.0, abs=1e-5)
    out = weighted(rows, 5, surprisal)
    assert out.shape == (5, 16) and not torch.allclose(out, plain(rows, 5))
    out.sum().backward()
    assert weighted.surprisal_bias.grad is not None and weighted.surprisal_bias.grad.abs().sum() > 0, "gradient reaches the pooling bias"


# --------------------------------------------------------------------------------------------------- 6. the surprisal pooling in the encoder
def test_pool_surprisal_encoder_and_training(tmp_path, hybrid_dir):
    from activation.ac_model import ActivationContextTrainingReporter
    h, ac = build8(tmp_path, hybrid_dir, pool=1.0, seed=13)
    rollout_part = direct_parts(items_for(h, ac)[0].ac_messages)[0]
    rollout_rows = rows_of(ac, ac._child_request(rollout_part, 0.25)._replace(is_recursive=False))      # rollout mode (no grad, cache) runs the surprisal path too
    assert rollout_rows.shape[0] >= 2 and len(ac.surprisal_cache) == 1
    ac.surprisal_cache.clear()
    trainer, items, examples, device = trainer_and_examples(h, ac)
    request = examples[0].part_requests[0]
    ids, spans = tokenize_with_parts(ac.side.tokenizer, request.messages, [], ac.pad_id, tools=request.tools)
    surprisal = ac._content_surprisal([(ids, spans, [], 2, False)], device)[0]
    other = examples[1].part_requests[0]
    ids2, spans2 = tokenize_with_parts(ac.side.tokenizer, other.messages, [], ac.pad_id, tools=other.tools)
    ac.surprisal_cache.clear()
    pair = ac._content_surprisal([(ids, spans, [], 2, False), (ids2, spans2, [], 2, False)], device)          # a right-padded batch: each part's own values
    torch.testing.assert_close(pair[0], surprisal, rtol=1e-4, atol=1e-4)
    assert pair[1].shape[0] == len(ids2) and len(ids) != len(ids2)
    ac.surprisal_cache.clear()
    torch.testing.assert_close(ac._content_surprisal([(ids2, spans2, [], 2, False)], device)[0], pair[1], rtol=1e-4, atol=1e-4)
    ac.surprisal_cache.clear()
    surprisal = ac._content_surprisal([(ids, spans, [], 2, False)], device)[0]
    embedding, head = ac.side.model.get_input_embeddings(), ac.side.model.get_output_embeddings()
    with torch.no_grad():
        hidden = ac.side.decoder_forward(embedding(torch.tensor(ids))[None], None, lora_name=None, gradient_checkpointing=False)[0]
        logp = torch.log_softmax(head(hidden).float(), dim=-1)
        expected = torch.cat([torch.zeros(1), -logp[:-1].gather(-1, torch.tensor(ids[1:])[:, None])[:, 0]])
    torch.testing.assert_close(surprisal, expected, rtol=1e-5, atol=1e-5)
    assert bool((surprisal[1:] > 0).all()) and len(ac.surprisal_cache) == 1
    assert torch.equal(ac._content_surprisal([(ids, spans, [], 2, False)], device)[0], surprisal), "cached by the token ids"
    with torch.no_grad():
        for parameter in side_lora_parameters(ac):
            parameter.add_(0.5)
    ac.surprisal_cache.clear()
    torch.testing.assert_close(ac._content_surprisal([(ids, spans, [], 2, False)], device)[0], expected, rtol=1e-5, atol=1e-5), "the BASE's surprisal: the adapter plays no part"
    # Training: the bias takes gradient and moves, the traces and the panel carry the bin statistics, the checkpoint round-trips.
    reporter = ActivationContextTrainingReporter(str(tmp_path / "report"), "pool")
    stats = ActivationContextTrainer(h, config(examples_per_update=1)).train(ac.name, items, reporting_data=items, reporter=reporter)
    assert not torch.equal(ac.modules.summary_pooling.surprisal_bias, torch.zeros(4)), "the pooling bias trains"
    keys = {"pool_row_mean", "pool_first", "pool_last", "pool_width_max", "pool_width_uniform"}
    assert stats.encoder_trace and all(keys <= set(entry) for entry in stats.encoder_trace) and stats.encoder_panel and keys <= set(stats.encoder_panel[-1])
    assert all(entry["pool_width_max"] <= 2 * entry["pool_width_uniform"] + 2 for entry in stats.encoder_trace)
    assert stats.passes_off is None and {"pool_bins", "pool_bins_panel"} <= set(reporter.widgets) and "refine_rms" not in reporter.widgets
    assert stats.reporting.get("rare_token_loss") is not None and 0 <= stats.reporting["rare_token_accuracy"] <= 1, "the rare-token panel comes with the surprisal features"
    saved = ac.save(str(tmp_path / "ckpt"))
    h2, again = build8(tmp_path / "again", hybrid_dir, pool=1.0, seed=13)
    again.load(saved)
    assert torch.equal(again.modules.summary_pooling.surprisal_bias, ac.modules.summary_pooling.surprisal_bias)


# --------------------------------------------------------------------------------------------------- 7. the surprisal-weighted terminal loss
def test_rare_weight_training_and_off_is_inert(tmp_path, hybrid_dir):
    from activation.ac_model import ActivationContextTrainingReporter
    with pytest.raises(ValueError, match="rare_weight"):
        config(rare_weight=-1.0)
    with pytest.raises(ValueError, match="rare_weight"):
        config(loss_kind="kl", rare_weight=1.0)
    h, ac = build8(tmp_path, hybrid_dir, seed=17)
    trainer, items, examples, device = trainer_and_examples(h, ac, rare_weight=1.0, no_context_penalty_weight=0.0)
    assert all(example.penalty is not None for example in examples), "the no-context reference is built for the surprisal even with the penalty off"
    weights = trainer._token_weights(examples[0], device)
    surprisal = trainer._target_surprisal(examples[0])
    assert weights.shape == (examples[0].num_completion,) and weights.mean().item() == pytest.approx(1.0, abs=1e-6) and bool((weights > 0).all())
    expected = 1.0 + surprisal / surprisal.mean()
    torch.testing.assert_close(weights, (expected / expected.mean()).to(device))
    assert bool((surprisal >= 0).all()) and surprisal.numel() == examples[0].num_completion
    # The objective is the weighted cross-entropy, the metric the unweighted one (= the rare_weight-0 trainer's loss on the same state).
    plain = ActivationContextTrainer(h, config(no_context_penalty_weight=0.0))
    plain.teacher_target_logp = trainer.teacher_target_logp
    torch.manual_seed(0); weighted_loss, *_ = trainer._example_loss(ac, examples[0], device)
    torch.manual_seed(0); plain_loss, *_ = plain._example_loss(ac, examples[0], device)
    assert trainer._loss_metric == pytest.approx(float(plain_loss.detach()), abs=1e-6) and plain._loss_metric is None
    cross_entropy, _, _ = trainer._chunked_sft_tokens(ac.target.decoder_forward(ac.target.model.get_input_embeddings()(torch.tensor(examples[0].student_ids))[None], None, lora_name="reader_lora",
                                                                                gradient_checkpointing=False)[0][examples[0].student_positions], ac.target.model.get_output_embeddings(),
                                                       torch.tensor(examples[0].target_ids))
    assert float(weighted_loss) != float(plain_loss) or bool((weights == 1).all())
    # The batched path gives the per-example values.
    batched = ActivationContextTrainer(h, config(rare_weight=1.0, no_context_penalty_weight=0.0, micro_batch_examples=2))
    batched.teacher_target_logp = trainer.teacher_target_logp
    results = batched._batch_loss(ac, examples[:2], device)
    single = [trainer._example_loss(ac, example, device)[0] for example in examples[:2]]
    for result, loss in zip(results, single):
        assert float(result["objective"]) == pytest.approx(float(loss), abs=1e-5) and result["metric"] is not None
    evaluated = batched._batch_loss(ac, examples[:2], device, with_penalty=False)
    assert all(result["metric"] is None for result in evaluated), "evaluation forwards are unweighted"
    # A training call: the recorded loss is the unweighted metric, the rare-token panel exists with sane values, the reporter takes it; off, nothing of it exists.
    reporter = ActivationContextTrainingReporter(str(tmp_path / "report"), "rare")
    stats = ActivationContextTrainer(h, config(rare_weight=1.0, examples_per_update=1)).train(ac.name, items, reporting_data=items, reporter=reporter)
    assert stats.rare_weight == 1.0 and all(math.isfinite(value) for value in stats.loss)
    for row in stats.reporting["per_item"]:
        assert row["rare_positions"] == max(1, math.ceil(0.2 * row["positions"])) and math.isfinite(row["rare_token_loss"]) and 0 <= row["rare_token_accuracy"] <= 1
    assert stats.reporting["rare_token_loss"] == pytest.approx(sum(row["rare_token_loss"] for row in stats.reporting["per_item"]) / len(stats.reporting["per_item"]))
    assert all("rare_token_loss" in kind for kind in stats.reporting["by_kind"].values()) and stats.epochs[-1].validation is None
    assert {"rare_token_loss", "rare_token_accuracy"} <= set(reporter.widgets)
    h0, off = build8(tmp_path / "off", hybrid_dir, seed=17)
    stats0 = ActivationContextTrainer(h0, config(examples_per_update=1)).train(off.name, items_for(h0, off), reporting_data=items_for(h0, off))
    assert stats0.rare_weight == 0.0 and "rare_token_loss" not in stats0.reporting and all("rare_token_loss" not in row for row in stats0.reporting["per_item"])
    assert all("rare_token_loss" not in kind for kind in stats0.reporting["by_kind"].values()) and stats0.encoder_trace == [] and "encoder_panel" not in stats0.summarize()
    assert set(off.modules.state_dict()) == set(ac.modules.state_dict()), "no module either way"


# --------------------------------------------------------------------------------------------------- 8. bench keys and arms
def test_bench_round8_keys_and_arms():
    runner = BENCH / "run_recon.py"
    if not runner.exists():
        pytest.skip("bench not present")
    spec = importlib.util.spec_from_file_location("run_recon", runner)
    run_recon = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_recon)
    default = run_recon.DEFAULT_ARM
    assert (default["encoder_passes"], default["rare_weight"], default["pool_surprisal"]) == (1, 0.0, 0.0)
    parameters = inspect.signature(run_recon.harness).parameters
    assert parameters["encoder_passes"].default == 1 and parameters["pool_surprisal"].default == 0.0
    assert run_recon.config(dict(default, rare_weight=1.0)).rare_weight == 1.0 and run_recon.config(default).rare_weight == 0.0
    arms = run_recon.arms()
    for label, arm in arms.items():                      # every arm (defaults merged) builds a training config with its rare_weight
        assert run_recon.config(arm).rare_weight == arm["rare_weight"] and arm["encoder_passes"] >= 1 and arm["pool_surprisal"] >= 0, label
    strip = lambda arm, *keys: {key: value for key, value in arm.items() if key not in ("note", *keys)}
    raw = json.loads((BENCH / "arms.json").read_text())
    k16 = {"init_ac": raw["A32X"]["init_ac"], "init_reader": raw["A32X"]["init_reader"]}
    assert "/root/" in k16["init_ac"] and "/home/ngom/" in raw["A32W"]["init_ac"]
    pairs = {"A32P2": ("A32W", {"encoder_passes": 2}), "A32KVTAP2": ("A32KVTA", {"encoder_passes": 2}), "A32SW": ("A32W", {"rare_weight": 1.0}),
             "A32SP": ("A32W", {"pool_surprisal": 1.0}), "A32SPKVTA": ("A32KVTA", {"pool_surprisal": 1.0}), "NSP": ("WCa", {"pool_surprisal": 1.0})}
    for label, (base, change) in pairs.items():
        arm = raw[label]
        assert strip(arm, *change) == strip(raw[base]) and all(arm[key] == value for key, value in change.items()), label
        assert arm["init_ac"] == raw[base]["init_ac"] and "round 8" in arm["note"] and "Read held" in arm["note"], label
        twin = raw[label + "n"]
        assert strip(twin, "init_ac", "init_reader") == strip(arm, "init_ac", "init_reader") and (twin["init_ac"], twin["init_reader"]) == (k16["init_ac"], k16["init_reader"]), label
        assert twin["note"].startswith(arm["note"]) and "EC2 twin" in twin["note"], label
    for label in ("A32P2n", "A32SWn", "A32SPn"):
        assert "A32X 0.322" in raw[label]["note"]
    assert "A32KVTAn 0.300" in raw["A32KVTAP2n"]["note"] and "A32KVTAn 0.300" in raw["A32SPKVTAn"]["note"] and "WCn 0.122" in raw["NSPn"]["note"]
    assert raw["NSP"]["seed"] == raw["WCa"]["seed"] == raw["NSPn"]["seed"], "the 1/16 arms keep WCa's seed (like NXD / NXDn)"
    assert list(raw)[-53:-41] == [*pairs, *(label + "n" for label in pairs)] and list(raw)[-54] == "SK32XD" and list(raw)[-41] == "R8S"
    for label in ("A32P3", "A32AN", "A32P2AN", "NP2", "A32P3n", "A32ANn", "A32P2ANn", "NP2n"):
        assert label not in raw, "dropped at design review"


# --------------------------------------------------------------------------------------------------- 9. the refine adapter's own optimizer group
def test_refine_adapter_own_learning_rate_group(tmp_path, hybrid_dir):
    """learning_rate_refine None keeps the adapter in the encoder group (bit-identical layout); a rate gives it its own last group at that rate,
    holding exactly the adapter's parameters, and the adapter moves faster than at the encoder rate."""
    h1, one = build8(tmp_path / "plain", hybrid_dir, passes=2, seed=3)
    torch.manual_seed(1); trainer1 = ActivationContextTrainer(h1, config(examples_per_update=3))
    stats1 = trainer1.train(one.name, items_for(h1, one))
    groups1 = trainer1.optimizers[one.name].param_groups
    h2, two = build8(tmp_path / "own", hybrid_dir, passes=2, seed=3)
    torch.manual_seed(1); trainer2 = ActivationContextTrainer(h2, config(examples_per_update=3, learning_rate_refine=4e-2))
    stats2 = trainer2.train(two.name, items_for(h2, two))
    groups2 = trainer2.optimizers[two.name].param_groups
    assert len(groups2) == len(groups1) + 1 and stats2.learning_rates[-1][-1] == pytest.approx(4e-2)
    refine = {id(parameter) for parameter in two.modules.refine_adapter.parameters()}
    assert {id(parameter) for parameter in groups2[-1]["params"]} == refine
    assert not any(id(parameter) in refine for group in groups2[:-1] for parameter in group["params"])
    assert any(id(parameter) in {id(q) for q in one.modules.refine_adapter.parameters()} for parameter in groups1[0]["params"]), "at None the adapter stays in the encoder group"
    assert two.modules.refine_adapter.up.weight.abs().sum() > one.modules.refine_adapter.up.weight.abs().sum()
    with pytest.raises(ValueError):
        config(learning_rate_refine=-1.0)
