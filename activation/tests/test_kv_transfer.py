"""Key/value transfer (harness/kv_transfer.py, ActivationContextModelConfig.kv_transfer) on the tiny CPU Qwen3:
the transferred run equals one masked full run, generation equals a full recompute, the AC model emits keys and values of
every layer, the untrained pipeline equals the plain model reading the part with markers, gradients agree with and without
gradient checkpointing, a short training moves the encoder, checkpoints round-trip.

  PYTHONPATH=. .venv/bin/python -m pytest activation/tests/test_kv_transfer.py -q -p no:cacheprovider
"""
import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.modules.setdefault("fla", None)
import pytest
import torch

from activation.ac_model import ActivationContextTrainer
from activation.ac_model.ac_model_utils import ACOutput, interleaved_positions
from activation.harness.hf_utils import generate_batch_with_interventions, generate_with_interventions, text_decoder
from activation.harness.kv_transfer import LayerInputCapture, bind_kv_prefix, check_kv_model
from activation.tests.test_interventions import TINY_QWEN3, build, config, fixed_request, side_lora_parameters, trainer_and_examples

torch.set_num_threads(2)


def kv_build(tmp_path, **overrides):
    return build(tmp_path, TINY_QWEN3, summary_layout="interleaved", kv_transfer=True, **overrides)


def masked_full_run(decoder, header, part, tail, markers, steps=0, head=None, embed=None):
    """The oracle: one sequence [header, part (tokens and markers), tail] in which the part attends only itself and the tail
    attends the header, the part's markers and itself. Returns the hidden states at the header and the tail (and, with `steps`,
    the greedy continuation recomputed in full at every step)."""
    h, p = header.shape[0], part.shape[0]
    tokens = []
    while True:
        sequence = torch.cat([header, part, tail])
        n = sequence.shape[0]
        mask = torch.ones(n, n, dtype=torch.bool).tril()
        mask[h:h + p, :h] = False                                   # the part does not see the header (it is encoded alone)
        hidden_columns = torch.ones(n, dtype=torch.bool)
        hidden_columns[h:h + p] = True
        hidden_columns[h + markers] = False
        hidden_columns[:h] = False
        hidden_columns[h + p:] = False
        mask[h + p:, hidden_columns] = False                        # the tail does not see the part's tokens, only its markers
        hidden = decoder(inputs_embeds=sequence[None], attention_mask={"full_attention": mask[None, None]}, use_cache=False).last_hidden_state[0]
        if len(tokens) == steps:
            return torch.cat([hidden[:h], hidden[h + p:h + p + tail.shape[0] - len(tokens)]]), tokens
        token = int(head(hidden[-1]).argmax())
        tokens.append(token)
        tail = torch.cat([tail, embed(torch.tensor([token]))])


def test_transfer_equals_a_masked_full_run_and_generation_a_full_recompute(tmp_path):
    h, ac = kv_build(tmp_path)
    model = ac.target.model
    decoder = text_decoder(model)
    layers, heads, head_dim = check_kv_model(model)
    embed = model.get_input_embeddings()
    torch.manual_seed(3)
    with torch.no_grad():
        header, tokens, tail = (embed(torch.randint(5, 60, (count,))) for count in (3, 12, 5))
        token_positions, markers = interleaved_positions(12, 3, torch.device("cpu"))
        part = torch.zeros(15, header.shape[1]).index_copy(0, token_positions, tokens).index_copy(0, markers, torch.randn(3, header.shape[1]) * 0.05)
        # Encoder: the part alone; keys and values of every layer at the markers.
        capture = LayerInputCapture([markers])
        ac.target.decoder_forward(part[None], None, capture=capture)
        assert sorted(capture.keys) == list(range(layers)) and capture.keys[0][0].shape == (3, heads, head_dim)
        payload = {"positions": [3, 4, 5], "layer_inputs": {}, "kv_values": {layer: (capture.keys[layer][0], capture.values[layer][0]) for layer in capture.keys},
                   "kv_positions": markers, "kv_runs": [(3, 6, 15)]}
        prompt = torch.cat([header, torch.zeros(3, header.shape[1]), tail])
        transferred = ac.target.decoder_forward(prompt[None], None, interventions=[payload])[0]
        oracle, _ = masked_full_run(decoder, header, part, tail, markers)
    torch.testing.assert_close(torch.cat([transferred[:3], transferred[6:]]), oracle, atol=2e-5, rtol=1e-4)
    # A different placeholder content changes nothing (nobody attends the placeholders), and a payload-free row is the plain forward.
    with torch.no_grad():
        noisy = prompt.clone()
        noisy[3:6] = torch.randn(3, header.shape[1])
        again = ac.target.decoder_forward(noisy[None], None, interventions=[payload])[0]
        plain = ac.target.decoder_forward(torch.stack([prompt, prompt]), None, interventions=[payload, None])
    torch.testing.assert_close(torch.cat([again[:3], again[6:]]), torch.cat([transferred[:3], transferred[6:]]), atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(plain[0], transferred, atol=1e-6, rtol=1e-5)
    # Generation: the greedy loop over the growing cache equals a full recompute at every step; a left-padded batch equals its rows.
    head = model.get_output_embeddings()
    with torch.no_grad():
        _, expected = masked_full_run(decoder, header, part, tail, markers, steps=5, head=head, embed=embed)
        single = generate_with_interventions(model, inputs_embeds=prompt[None], interventions=[payload], max_new_tokens=5, do_sample=False)[0].tolist()
        short = torch.cat([header, torch.zeros(3, header.shape[1]), tail[:3]])
        alone = generate_with_interventions(model, inputs_embeds=short[None], interventions=[payload], max_new_tokens=5, do_sample=False)[0].tolist()
        padded = torch.stack([prompt, torch.cat([torch.zeros(2, header.shape[1]), short])])
        mask = torch.ones(2, prompt.shape[0], dtype=torch.long)
        mask[1, :2] = 0
        shifted = {**payload, "positions": [5, 6, 7], "kv_runs": [(5, 8, 15)]}
        both = generate_batch_with_interventions(model, inputs_embeds=padded, attention_mask=mask, interventions=[payload, shifted], max_new_tokens=5, do_sample=False).tolist()
    assert single == expected and both == [single, alone]


def test_pipeline_untrained_equals_the_plain_model_with_markers(tmp_path):
    h, ac = kv_build(tmp_path)
    trainer, items, examples, device = trainer_and_examples(h, ac)
    layers, heads, head_dim = check_kv_model(ac.target.model)
    with pytest.raises(RuntimeError, match="with_layers=True"):
        ac.encode_batch([fixed_request(ac)])
    output = ac.encode_batch([fixed_request(ac)], with_layers=True)[0]
    rows = output.input_embeds.shape[0]
    assert isinstance(output, ACOutput) and not output.has_interventions and sorted(output.kv_values) == list(range(layers))
    assert all(keys.shape == values.shape == (rows, heads, head_dim) for keys, values in output.kv_values.values())
    assert torch.equal(output.input_embeds, torch.zeros_like(output.input_embeds)) and output.kv_positions.shape == (rows,) and output.kv_length > rows
    assert float(ac.modules.kv_marker.detach().norm()) > 0, "the marker starts as the mean input embedding"
    # A padded encode batch returns each part's own keys and values.
    example = examples[0]
    long = examples[1].part_requests[0]
    alone = ac.encode_batch(example.part_requests, with_layers=True)[0]
    padded = ac.encode_batch([example.part_requests[0], long], with_layers=True)[0]
    for layer in alone.kv_values:
        torch.testing.assert_close(alone.kv_values[layer][0], padded.kv_values[layer][0], atol=1e-5, rtol=1e-4)
    # The reader forward through the trainer's assembly: placeholders stay, the payload carries the entries; with the two adapters
    # at their zero init this is the plain model's masked full run over [prompt before the part, the part with markers, the rest].
    embedding = ac.target.model.get_input_embeddings()
    (start, end), = example.spans
    with torch.no_grad():
        student = embedding(torch.tensor(example.student_ids))
        embeds, payload = trainer._assemble_student(student, example.spans, [alone], student.dtype)
        assert payload["kv_runs"] == [(start, end, alone.kv_length)] and payload["positions"] == list(range(start, end)) and not payload["layer_inputs"]
        hidden = ac.target.decoder_forward(embeds[None], None, lora_name="reader_lora", interventions=[payload])[0]
        from activation.ac_model.ac_model_utils import tokenize_with_parts
        request = example.part_requests[0]
        ids, _ = tokenize_with_parts(ac.side.tokenizer, request.messages, [], ac.pad_id, tools=request.tools)
        token_positions, markers = interleaved_positions(len(ids), rows, torch.device("cpu"))
        part = torch.zeros(len(ids) + rows, ac.d_side).index_copy(0, token_positions, embedding(torch.tensor(ids))).index_copy(0, markers, ac.modules.kv_marker.detach().expand(rows, -1))
        oracle, _ = masked_full_run(text_decoder(ac.target.model), student[:start], part, student[end:], markers)
    torch.testing.assert_close(torch.cat([hidden[:start], hidden[end:]]), oracle, atol=5e-5, rtol=1e-4)
    # Withholding the payload (the deltas-off switch) leaves a prompt that carries nothing of the part.
    with torch.no_grad():
        _, none = trainer._assemble_student(student, example.spans, [alone], student.dtype, deltas_off=True)
    assert none is None


def test_gradients_and_training(tmp_path):
    h, ac = kv_build(tmp_path)
    trainer, items, examples, device = trainer_and_examples(h, ac)
    with torch.no_grad():                                                          # PEFT's zero lora_B would leave lora_A without gradient
        for name, parameter in ac.side.model.named_parameters():
            if "lora_B" in name:
                parameter.normal_(std=0.05)
    params = [ac.modules.kv_marker] + list(h.module_manager.lora_parameters("reader_lora")) + side_lora_parameters(ac)
    grads = {}
    for checkpointed in (False, True):
        trainer.config.gradient_checkpointing = checkpointed
        ac.config.side_gradient_checkpointing = checkpointed
        losses = [trainer._example_loss(ac, example, device, with_penalty=False)[0] for example in examples[:2]]
        grads[checkpointed] = torch.autograd.grad(sum(losses), params, allow_unused=True)
    unused = 0
    for direct, recomputed in zip(grads[False], grads[True]):
        assert (direct is None) == (recomputed is None)
        if direct is None:                     # the encoder's last layer past its key/value projections feeds nothing that is transferred
            unused += 1
            continue
        torch.testing.assert_close(direct, recomputed, atol=1e-6, rtol=1e-5)
    side_count = len(side_lora_parameters(ac))
    assert 0 < unused < side_count and all(g is not None for g in grads[True][:-side_count])
    assert float(grads[True][0].norm()) > 0, "the marker receives gradient through the encoder and the transferred entries"
    assert any(float(g.norm()) > 0 for g in grads[True][-side_count:] if g is not None), "the encoder's adapter receives gradient"
    assert any(float(g.norm()) > 0 for g in grads[True][1:-side_count]), "the reader's adapter receives gradient"
    # A training epoch runs end to end (two optimizer groups), the marker and the encoder's adapter move, samples decode.
    trainer = ActivationContextTrainer(h, config())
    before = ac.modules.kv_marker.detach().clone()
    side_before = [parameter.detach().clone() for parameter in side_lora_parameters(ac)]
    stats = trainer.train(ac.name, items, reporting_data=items)
    assert len(trainer.optimizers[ac.name].param_groups) == 2
    assert not torch.equal(before, ac.modules.kv_marker.detach()), "the marker trains"
    assert any(not torch.equal(a, b.detach()) for a, b in zip(side_before, side_lora_parameters(ac))), "the encoder's adapter trains"
    assert stats.intervention_trace == [] and stats.payload_bytes > 0
    # Checkpoint round trip: the same keys and values from a fresh model that loads the folder; a row-based model refuses it.
    ac.set_mode("rollout")
    folder = ac.save(str(tmp_path / "kv_checkpoint"))
    saved = ac.encode_batch([fixed_request(ac)], with_layers=True)[0]
    h2, twin = kv_build(tmp_path / "twin", ac_checkpoint=folder)
    loaded = twin.encode_batch([fixed_request(twin)], with_layers=True)[0]
    for layer in saved.kv_values:
        torch.testing.assert_close(saved.kv_values[layer][0], loaded.kv_values[layer][0], atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(saved.kv_values[layer][1], loaded.kv_values[layer][1], atol=1e-5, rtol=1e-4)
    with pytest.raises(ValueError, match="kv_transfer"):
        build(tmp_path / "rows", TINY_QWEN3, summary_layout="interleaved", ac_checkpoint=folder)


def test_shared_part_encodings_and_part_grouped_order(tmp_path):
    """One encoding per distinct part in a padded forward: the same losses and gradients as one encoding per example, fewer encodes."""
    from activation.ac_model.ac_model_utils import part_group_key, part_grouped_order
    h, ac = kv_build(tmp_path)
    trainer, items, examples, device = trainer_and_examples(h, ac, micro_batch_examples=3, share_part_encodings=True)
    with torch.no_grad():
        for name, parameter in ac.side.model.named_parameters():
            if "lora_B" in name:
                parameter.normal_(std=0.05)
    params = [ac.modules.kv_marker] + list(h.module_manager.lora_parameters("reader_lora")) + side_lora_parameters(ac)
    batch = [examples[0], examples[1], examples[0]]                                # two examples with the same part, one with another
    assert part_group_key(batch[0]) == part_group_key(batch[2]) != part_group_key(batch[1])
    losses, grads, encodes = {}, {}, {}
    for shared in (False, True):
        trainer.config.share_part_encodings = shared
        before = ac.stats.encodes
        results = trainer._batch_loss(ac, batch, device, with_penalty=False)
        encodes[shared] = ac.stats.encodes - before
        losses[shared] = torch.stack([result["objective"] for result in results])
        grads[shared] = torch.autograd.grad(losses[shared].sum(), params, allow_unused=True)
    assert encodes == {False: 3, True: 2}
    torch.testing.assert_close(losses[False], losses[True], atol=1e-5, rtol=1e-4)
    for separate, shared in zip(grads[False], grads[True]):
        assert (separate is None) == (shared is None)
        if separate is not None:
            torch.testing.assert_close(separate, shared, atol=1e-5, rtol=1e-3)
    keys = [index % 5 for index in range(43)]
    order = list(range(43))[::-1]
    grouped = part_grouped_order(order, keys, 4, seed=1)
    assert sorted(grouped) == list(range(43)) and part_grouped_order(order, keys, 0, seed=1) is order
    assert all(len({keys[index] for index in grouped[start:start + 4]}) == 1 for start in range(0, 40, 4)) and len(grouped[40:]) == 3
    with pytest.raises(ValueError, match="share_part_encodings"):
        config(share_part_encodings=True)
    with pytest.raises(ValueError, match="both reorder"):
        config(part_group_examples=4, cost_bucket_examples=8)


# --------------------------------------------------------------------------------------------------- overwrite mode (hybrid decoders)
@pytest.fixture(scope="module")
def hybrid_dir(tmp_path_factory):
    from activation.tests.test_interventions import tiny_hybrid
    return tiny_hybrid(tmp_path_factory.mktemp("models") / "tiny_hybrid")


def test_overwrite_mode_on_a_hybrid_decoder(tmp_path, hybrid_dir):
    from activation.harness.kv_transfer import kv_layers
    h, ac = build(tmp_path, hybrid_dir, summary_layout="interleaved", kv_transfer=True, kv_overwrite=True)
    model = ac.target.model
    assert kv_layers(model) == [2, 5], "the tiny hybrid has attention at layers 2 and 5 only"
    embed = model.get_input_embeddings()
    torch.manual_seed(5)
    with torch.no_grad():
        prompt = embed(torch.randint(5, 60, (14,)))
        positions = torch.tensor([4, 5, 6, 7])
        # Overwriting the rows with the reader's OWN keys and values (captured from a plain forward) changes nothing.
        capture = LayerInputCapture([positions])
        plain = ac.target.decoder_forward(prompt[None], None, capture=capture)[0]
        assert sorted(capture.keys) == [2, 5]
        own = {"positions": positions.tolist(), "layer_inputs": {}, "kv_overwrite": True, "kv_values": {layer: (capture.keys[layer][0], capture.values[layer][0]) for layer in capture.keys}}
        same = ac.target.decoder_forward(prompt[None], None, interventions=[own])[0]
        torch.testing.assert_close(same, plain, atol=1e-5, rtol=1e-4)
        # Other keys and values change what follows the rows, not what precedes them; the attention-entry mode refuses a hybrid.
        other = {**own, "kv_values": {layer: (keys + 0.5, values - 0.5) for layer, (keys, values) in own["kv_values"].items()}}
        changed = ac.target.decoder_forward(prompt[None], None, interventions=[other])[0]
        torch.testing.assert_close(changed[:4], plain[:4], atol=1e-5, rtol=1e-4)
        assert not torch.allclose(changed[8:], plain[8:], atol=1e-3)
        with pytest.raises(ValueError, match="full-attention decoder"):
            bind_kv_prefix(model, prompt[None], None, [{**own, "kv_positions": positions, "kv_runs": [(4, 8, 8)]}])
        # Generation (HF's loop, the overwrite bound at the prefill) equals a full recompute at every step; a left-padded batch equals its rows.
        head = model.get_output_embeddings()
        def recompute(sequence, payload, steps=4):
            tokens = []
            for _ in range(steps):
                hidden = ac.target.decoder_forward(sequence[None], None, interventions=[payload])[0]
                tokens.append(int(head(hidden[-1]).argmax()))
                sequence = torch.cat([sequence, embed(torch.tensor(tokens[-1:]))])
            return tokens
        generation = dict(max_new_tokens=4, min_new_tokens=4, do_sample=False, use_cache=True, pad_token_id=ac.target.tokenizer.pad_token_id)
        single = generate_with_interventions(model, inputs_embeds=prompt[None], interventions=[other], **generation)[0].tolist()
        assert single == recompute(prompt, other)
        short = prompt[:11]
        alone = generate_with_interventions(model, inputs_embeds=short[None], interventions=[other], **generation)[0].tolist()
        padded = torch.stack([prompt, torch.cat([torch.zeros(3, prompt.shape[1]), short])])
        mask = torch.ones(2, 14, dtype=torch.long)
        mask[1, :3] = 0
        shifted = {**other, "positions": [7, 8, 9, 10]}
        both = generate_batch_with_interventions(model, inputs_embeds=padded, attention_mask=mask, interventions=[other, shifted], **generation).tolist()
        assert both == [single, alone]
    # The pipeline: the encoder emits keys and values of its attention layers, the rows enter the reader as the marker embedding.
    trainer, items, examples, device = trainer_and_examples(h, ac)
    output = ac.encode_batch([fixed_request(ac)], with_layers=True)[0]
    assert sorted(output.kv_values) == [2, 5] and output.kv_overwrite
    assert torch.equal(output.input_embeds.detach(), ac.modules.kv_marker.detach().float().expand_as(output.input_embeds))
    with torch.no_grad():
        for name, parameter in ac.side.model.named_parameters():
            if "lora_B" in name:
                parameter.normal_(std=0.05)
    params = [ac.modules.kv_marker] + list(h.module_manager.lora_parameters("reader_lora")) + side_lora_parameters(ac)
    grads = {}
    for checkpointed in (False, True):
        trainer.config.gradient_checkpointing = checkpointed
        ac.config.side_gradient_checkpointing = checkpointed
        losses = [trainer._example_loss(ac, example, device, with_penalty=False)[0] for example in examples[:2]]
        grads[checkpointed] = torch.autograd.grad(sum(losses), params, allow_unused=True)
    for direct, recomputed in zip(grads[False], grads[True]):
        assert (direct is None) == (recomputed is None)
        if direct is not None:
            torch.testing.assert_close(direct, recomputed, atol=1e-6, rtol=1e-5)
    assert float(grads[True][0].norm()) > 0
    stats = ActivationContextTrainer(h, config()).train(ac.name, items, reporting_data=items)
    assert stats.payload_bytes > 0
