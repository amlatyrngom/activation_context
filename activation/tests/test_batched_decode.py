"""The checkpoint decode's speed options on tiny CPU models: the encoder-output cache returns the tensors of the uncached encode;
the batched decode (left-padded rows, row positions moved with the pads, a stop and a new-token budget per row) returns the
completions of the one-entry-per-call path token for token, with and without the cache; the review script keeps its progress file.

  PYTHONPATH=. .venv/bin/python -m pytest activation/tests/test_batched_decode.py -q -p no:cacheprovider

The reader is the tiny hybrid Qwen3.5 (linear-attention blocks with their convolution and recurrent state, full attention every
third block), the AC model is deep with audible deltas, and the prompts run past the 64-token chunk of the linear-attention rule,
so a padded row's chunks are cut elsewhere than its single-sequence run's.
"""
import importlib.util
import json
import os
import random
import sys
from contextlib import contextmanager
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.modules.setdefault("fla", None)          # the kernels are GPU-only; the models fall back to torch
import pytest
import torch

from activation.ac_model import ActivationContextStudyGenerator, ActivationContextTrainer
from activation.ac_model.ac_model_training import ActivationContextTrainingItem, ActivationContextTrainingProgress
from activation.common.ac_parts import ac_part
from activation.harness import hf_utils
from activation.harness.hf_utils import INTERVENTIONS_KWARG, decoder_layers, generate_batch_with_interventions, generate_with_interventions
from activation.tests.test_interventions import build, config, open_heads, tiny_hybrid

BENCH = Path(__file__).resolve().parents[2] / "IB/ARTIFACTS/FIRST_TASK_AWARE_FINETUNING/bench"
PROGRESS = ActivationContextTrainingProgress(0, 1, 0, 0, 1, 0, 0.0, 0.0)
BUDGET = 16                                  # new tokens per completion unless the reader's capacity leaves fewer
CAPACITY_ROOM = 5                            # ... which it does for the longest prompt: this many
torch.set_num_threads(2)


@contextmanager
def recorded(tokenizer):
    """The token lists decoded into completions (the `skip_special_tokens=True` calls), in call order."""
    seen, real = [], tokenizer.decode
    def decode(ids, *args, **kwargs):
        if kwargs.get("skip_special_tokens"):
            seen.append([int(token) for token in ids])
        return real(ids, *args, **kwargs)
    tokenizer.decode = decode
    try:
        yield seen
    finally:
        del tokenizer.decode


class Decode:
    """The models, a mixed set of items and the sequential path's completions (texts and token lists) for them."""

    def __init__(self, root: Path) -> None:
        self.model_dir = tiny_hybrid(root / "tiny_hybrid")
        self.h, self.ac = build(root, self.model_dir, frequency=1)
        open_heads(self.ac, std=0.5)
        self.target = self.ac.target
        self.tokenizer = self.target.tokenizer
        words = sorted(word for word in self.tokenizer.get_vocab() if not word.startswith("<"))
        rng = random.Random(3)
        text = lambda count: " ".join(rng.choice(words) for _ in range(count))
        self.articles = [text(60), text(45), text(30)]
        # (name, articles read through rows, question words): the questions of one article are apart in the list, the prompts run
        # from a few tokens to past two 64-token chunks, one item reads two articles and one reads none.
        shapes = [("a1", [0], 3), ("c1", [2], 70), ("d", [0, 1], 20), ("a2", [0], 150), ("e", [], 12), ("a3", [0], 1), ("c2", [2], 101), ("b1", [1], 40)]
        self.items = [self.item(name, parts, text(count)) for name, parts, count in shapes]
        self.trainer = ActivationContextTrainer(self.h, config(completion_max_new_tokens=BUDGET, no_context_penalty_weight=0.0))
        examples = [self.trainer.build_example(self.ac, item) for item in self.items]
        self.prefixes = [example.student_positions[0] + 1 for example in examples]
        self.part_counts = [len(example.part_requests) for example in examples]
        # The longest prompt has room for CAPACITY_ROOM new tokens only: its budget is its own.
        self.text_config = self.target.model.config.get_text_config() if hasattr(self.target.model.config, "get_text_config") else self.target.model.config
        self.capacity_before = self.text_config.max_position_embeddings
        self.text_config.max_position_embeddings = max(self.prefixes) + CAPACITY_ROOM
        self.budgets = [min(BUDGET, self.text_config.max_position_embeddings - prefix) for prefix in self.prefixes]
        # An end-of-turn token that ends some rows early and that other rows never write: taken from the free-running completions.
        _, free = self.sequential()
        stop = next(token for tokens in free for token in tokens[1:BUDGET - 3] if sum(token not in other for other in free) >= 2)
        self.stop = stop
        self.target.model_config.model_description.eot_token = self.tokenizer.convert_ids_to_tokens(stop)
        self.texts, self.tokens = self.sequential()

    def item(self, name: str, parts: list[int], question: str) -> ActivationContextTrainingItem:
        content = [{"type": "text", "text": "Passages:\n"}]
        for article in parts:
            content += [ac_part([{"role": "user", "content": self.articles[article]}], self.ac.name, 0.25), {"type": "text", "text": "\n\n"}]
        content.append({"type": "text", "text": question})
        messages = [{"role": "system", "content": "read"}, {"role": "user", "content": content}, {"role": "assistant", "content": "answer"}]
        return ActivationContextTrainingItem(name, "rag_qa", None, messages, 0, 2, [self.tokenizer.encode("answer first", add_special_tokens=False)],
                                             info={"question": question, "answer": "answer", "panel_group": "mc/tiny"})

    def sequential(self, items=None) -> tuple[list[str], list[list[int]]]:
        """One entry per call: the path of a decode with a progress file."""
        with recorded(self.tokenizer) as seen:
            texts = [self.trainer._sample_completions(self.ac, [item], PROGRESS)[0]["student"] for item in (self.items if items is None else items)]
        return texts, seen

    def batched(self, batch: int, cache: int, items=None, **kwargs) -> tuple[list[str], list[list[int]], list[int]]:
        """Texts and token lists in the items' order, and the order the entries were decoded in."""
        items, order = (self.items if items is None else items), []
        with recorded(self.tokenizer) as seen:
            samples = self.trainer._sample_completions_batched(self.ac, items, PROGRESS, batch_size=batch, row_cache=cache,
                                                               on_batch=lambda indexes, _: order.extend(indexes), **kwargs)
        assert [sample["item_id"] for sample in samples] == [item.item_id for item in items]
        tokens = dict(zip(order, seen, strict=True))
        return [sample["student"] for sample in samples], [tokens[index] for index in range(len(items))], order

    def assembled(self, item) -> tuple[torch.Tensor, dict | None]:
        """The item's prompt embeddings with its rows written in, and its payload: what both decode paths hand to generate."""
        example = self.trainer.build_example(self.ac, item)
        prefix = example.student_positions[0] + 1
        embeds = self.target.model.get_input_embeddings()(torch.tensor(example.student_ids[:prefix]))
        outputs = self.ac.encode_batch(example.part_requests, with_layers=True) if example.part_requests else []
        return self.trainer._assemble_student(embeds, example.spans, outputs, embeds.dtype)

    def close(self) -> None:
        self.text_config.max_position_embeddings = self.capacity_before
        self.target.model_config.model_description.eot_token = None


@pytest.fixture(scope="module")
def decode(tmp_path_factory):
    state = Decode(tmp_path_factory.mktemp("batched_decode"))
    yield state
    state.close()


def same_output(a, b) -> bool:
    return torch.equal(a.input_embeds, b.input_embeds) and a.layer_inputs.keys() == b.layer_inputs.keys() and all(torch.equal(a.layer_inputs[layer], b.layer_inputs[layer]) for layer in a.layer_inputs)


# --------------------------------------------------------------------------------------------------- the mixed set is mixed
def test_the_set_covers_the_cases(decode):
    assert len(set(decode.prefixes)) == len(decode.prefixes) and min(decode.prefixes) < 32 and max(decode.prefixes) > 128, "prompts of different lengths, past two chunks"
    assert set(decode.part_counts) == {0, 1, 2}
    assert sorted(set(decode.budgets)) == [CAPACITY_ROOM, BUDGET], "one row's budget is cut by the reader's capacity"
    lengths = [len(tokens) for tokens in decode.tokens]
    early = [index for index, tokens in enumerate(decode.tokens) if tokens[-1] == decode.stop and len(tokens) < decode.budgets[index]]
    full = [index for index, tokens in enumerate(decode.tokens) if len(tokens) == BUDGET and decode.stop not in tokens]
    assert early and full, f"some rows stop early and some run to the budget: lengths {lengths}"
    assert all(len(tokens) <= budget for tokens, budget in zip(decode.tokens, decode.budgets))
    assert len(decode.tokens[decode.budgets.index(CAPACITY_ROOM)]) <= CAPACITY_ROOM


# --------------------------------------------------------------------------------------------------- 1. the encoder-output cache
def test_cached_rows_are_the_uncached_rows(decode):
    ac, trainer = decode.ac, decode.trainer
    one = trainer.build_example(ac, decode.items[0]).part_requests            # article 0
    two = trainer.build_example(ac, decode.items[2]).part_requests            # articles 0 and 1 in one item
    other = trainer.build_example(ac, decode.items[1]).part_requests          # article 2
    assert len(one) == 1 and len(two) == 2 and two[0] == one[0]
    previous = ac.mode
    ac.set_mode("training")                                                   # as both decode paths encode
    try:
        with torch.no_grad():
            cache: dict = {}
            first, hit = trainer._encode_parts_cached(ac, one, cache, 2)
            assert not hit and len(cache) == 1
            fresh = ac.encode_batch(one, with_layers=True)
            assert first[0].layer_inputs and same_output(first[0], fresh[0]), "the stored rows and deltas are the uncached encode's, bit for bit"
            encodes = ac.stats.encodes
            again, hit = trainer._encode_parts_cached(ac, one, cache, 2)
            assert hit and again is first and ac.stats.encodes == encodes, "a hit runs no encoder"
            assert same_output(again[0], ac.encode_batch(one, with_layers=True)[0])
            # A list of two parts is its own entry (the parts share a padded encode batch): it returns that call's tensors.
            pair, hit = trainer._encode_parts_cached(ac, two, cache, 2)
            assert not hit and len(cache) == 2 and all(same_output(a, b) for a, b in zip(pair, ac.encode_batch(two, with_layers=True), strict=True))
            # Bounded: the least recently used list goes first.
            assert trainer._encode_parts_cached(ac, one, cache, 2)[1]         # `one` is now the most recent
            trainer._encode_parts_cached(ac, other, cache, 2)
            assert len(cache) == 2 and not trainer._encode_parts_cached(ac, two, cache, 2)[1], "the pair was evicted"
            assert trainer._encode_parts_cached(ac, other, cache, 2)[1]
            # Off: nothing is kept. A new encoder version: nothing is reused.
            off: dict = {}
            assert not trainer._encode_parts_cached(ac, one, off, 0)[1] and not trainer._encode_parts_cached(ac, one, off, 0)[1] and not off
            assert trainer._encode_parts_cached(ac, [], off, 2) == ([], False) and not off
            kept: dict = {}
            trainer._encode_parts_cached(ac, one, kept, 2)
            ac.version += 1
            try:
                assert not trainer._encode_parts_cached(ac, one, kept, 2)[1]
            finally:
                ac.version -= 1
    finally:
        ac.set_mode(previous)


# --------------------------------------------------------------------------------------------------- 2. batched == sequential
@pytest.mark.parametrize("batch,cache", [(1, 0), (1, 2), (2, 0), (3, 1), (4, 4), (8, 0), (8, 4), (64, 1)])
def test_batched_completions_are_the_sequential_ones(decode, batch, cache):
    stats: dict = {}
    texts, tokens, order = decode.batched(batch, cache, stats=stats)
    assert tokens == decode.tokens, "token for token"
    assert texts == decode.texts
    assert sorted(order) == list(range(len(decode.items)))
    with_parts = sum(count > 0 for count in decode.part_counts)
    distinct = 4                                                              # article 0, article 2, articles 0+1, article 1
    assert stats["items"] == len(decode.items) and stats["batches"] == -(-len(decode.items) // batch)
    assert stats["new_tokens"] == sum(len(row) for row in decode.tokens)
    assert (stats["encodes"], stats["cache_hits"]) == ((distinct, with_parts - distinct) if cache else (with_parts, 0)), "one encode per distinct part list, at any cache size"


def test_entries_of_one_article_decode_together(decode):
    _, _, order = decode.batched(3, 1)
    names = [decode.items[index].item_id for index in order]
    assert names == ["a1", "a2", "a3", "c1", "c2", "d", "e", "b1"], "groups in order of first appearance, the items' order inside a group"
    calls = []
    decode.trainer._sample_completions_batched(decode.ac, decode.items, PROGRESS, batch_size=3, on_batch=lambda indexes, samples: calls.append((list(indexes), [s["item_id"] for s in samples])))
    assert [len(indexes) for indexes, _ in calls] == [3, 3, 2] and all([decode.items[i].item_id for i in indexes] == ids for indexes, ids in calls)
    assert decode.trainer._sample_completions_batched(decode.ac, [], PROGRESS, batch_size=3) == []
    with pytest.raises(ValueError, match="at least 1"):
        decode.trainer._sample_completions_batched(decode.ac, decode.items, PROGRESS, batch_size=0)


def test_default_path_is_untouched_by_the_batched_one(decode):
    """The sequential path gives the same completions before and after batched decodes (no state is left behind), and the models
    come back in the modes they were in."""
    modes = (decode.ac.mode, decode.target.model.training)
    counted = []
    real = decode.ac.side.tokenizer.apply_chat_template
    def counting(*args, **kwargs):
        counted.append(1)
        return real(*args, **kwargs)
    decode.ac.side.tokenizer.apply_chat_template = counting
    try:
        decode.batched(8, 0)
        uncached, counted[:] = len(counted), []
        decode.batched(8, 4)
    finally:
        del decode.ac.side.tokenizer.apply_chat_template
    assert 0 < len(counted) < uncached, "with the cache, a part's side tokenization (its row count, its encode) happens once per distinct part"
    assert "part_view_rows" not in decode.ac.__dict__, "the row-count memo lives for the call only"
    assert (decode.ac.mode, decode.target.model.training) == modes
    texts, tokens = decode.sequential()
    assert texts == decode.texts and tokens == decode.tokens


def test_the_row_positions_follow_the_padding(decode, monkeypatch):
    """Negative controls: the completions depend on the deltas and on where they land, so the equality above checks both."""
    real = hf_utils.generate_batch_with_interventions
    def unshifted(peft_model, *, inputs_embeds, attention_mask, interventions, **kwargs):
        pads = (attention_mask == 0).sum(-1).tolist()
        back = [None if payload is None else {"positions": [position - pad for position in payload["positions"]], "layer_inputs": payload["layer_inputs"]}
                for payload, pad in zip(interventions, pads)]
        return real(peft_model, inputs_embeds=inputs_embeds, attention_mask=attention_mask, interventions=back, **kwargs)
    monkeypatch.setattr(hf_utils, "generate_batch_with_interventions", unshifted)
    wrong, _, _ = decode.batched(8, 0)
    padded_with_parts = [index for index, count in enumerate(decode.part_counts) if count and decode.prefixes[index] < max(decode.prefixes)]
    assert sum(wrong[index] != decode.texts[index] for index in padded_with_parts) >= 3, "rows at their unpadded positions land on the wrong tokens"
    silent = lambda peft_model, *, inputs_embeds, attention_mask, interventions, **kwargs: real(
        peft_model, inputs_embeds=inputs_embeds, attention_mask=attention_mask, interventions=[None] * len(interventions), **kwargs)
    monkeypatch.setattr(hf_utils, "generate_batch_with_interventions", silent)
    without, _, _ = decode.batched(8, 0)
    with_parts = [index for index, count in enumerate(decode.part_counts) if count]
    assert sum(without[index] != decode.texts[index] for index in with_parts) >= 3, "the deltas are heard"
    assert [without[index] for index, count in enumerate(decode.part_counts) if not count] == [decode.texts[index] for index, count in enumerate(decode.part_counts) if not count]


# --------------------------------------------------------------------------------------------------- 3. padding, numerically
def test_left_padded_prefill_equals_the_single_rows(decode):
    """The hidden state that predicts the first new token, each row alone against the left-padded batch: the pads in front leave
    the linear-attention state, the attention and the row deltas of every row as they are (float noise only)."""
    target = decode.target
    with torch.no_grad():
        previous = decode.ac.mode
        decode.ac.set_mode("training")
        try:
            rows = [decode.assembled(item) for item in decode.items]
        finally:
            decode.ac.set_mode(previous)
        alone = [target.decoder_forward(embeds[None], None, lora_name="reader_lora", interventions=[payload])[0, -1] for embeds, payload in rows]
        length = max(embeds.shape[0] for embeds, _ in rows)
        pads = [length - embeds.shape[0] for embeds, _ in rows]
        padded = torch.stack([torch.cat([embeds.new_zeros((pad, embeds.shape[1])), embeds]) for (embeds, _), pad in zip(rows, pads)])
        mask = torch.tensor([[0] * pad + [1] * (length - pad) for pad in pads])
        positions = (mask.cumsum(-1) - 1).clamp_min(0)                        # what generate derives from the mask
        shifted = [None if payload is None else {"positions": [position + pad for position in payload["positions"]], "layer_inputs": payload["layer_inputs"]}
                   for (_, payload), pad in zip(rows, pads)]
        together = target.decoder_forward(padded, mask, position_ids=positions, lora_name="reader_lora", interventions=shifted)[:, -1]
        scale = float(torch.stack(alone).abs().max())
        assert max(float((a - b).abs().max()) for a, b in zip(alone, together)) < 1e-4 * scale
        # Right padding reads the same last real token (causal), but a decode would continue from a state that ran over the pads:
        # the cached linear-attention state after the prefill differs from the unpadded row's.
        short = min(range(len(rows)), key=lambda index: rows[index][0].shape[0])
        embeds = rows[short][0]
        def state(inputs, attention_mask):
            model = decode.h.module_manager.ensure_lora("reader_lora")
            with decode.h.module_manager.lora_context("reader", "reader_lora"):
                cache = model(inputs_embeds=inputs, attention_mask=attention_mask, use_cache=True).past_key_values
            return cache.layers[0].recurrent_states[0]
        ones = torch.ones(1, embeds.shape[0], dtype=torch.long)
        zeros = torch.zeros(1, pads[short], dtype=torch.long)
        blank = embeds.new_zeros((1, pads[short], embeds.shape[1]))
        reference = state(embeds[None], ones)
        left = state(torch.cat([blank, embeds[None]], dim=1), torch.cat([zeros, ones], dim=1))
        right = state(torch.cat([embeds[None], blank], dim=1), torch.cat([ones, zeros], dim=1))
        assert torch.allclose(left, reference, atol=1e-5 * float(reference.abs().max()) + 1e-8), "left pads: the recurrent state the decode continues from is the row's own"
        assert not torch.allclose(right, reference, atol=1e-3 * float(reference.abs().max())), "right pads: the state decayed over the pads"


def test_the_payload_is_bound_for_the_prefill_only(decode):
    target, manager = decode.target, decode.h.module_manager
    peft_model = manager.ensure_lora("reader_lora")
    with torch.no_grad():
        previous = decode.ac.mode
        decode.ac.set_mode("training")
        try:
            rows = [decode.assembled(decode.items[index]) for index in (0, 2)]
        finally:
            decode.ac.set_mode(previous)
    layer = decode.ac.intervention_layers[0]
    seen = []
    def spy(module, args, kwargs):
        seen.append((INTERVENTIONS_KWARG in kwargs, args[0].shape[1]))
    hf_utils.install_intervention_hooks(peft_model.get_base_model())
    handle = decoder_layers(peft_model.get_base_model())[layer].register_forward_pre_hook(spy, with_kwargs=True, prepend=True)
    generation = dict(max_new_tokens=4, min_new_tokens=4, do_sample=False, use_cache=True, pad_token_id=decode.tokenizer.pad_token_id)
    try:
        with torch.no_grad(), manager.lora_context("reader", "reader_lora"):
            length = max(embeds.shape[0] for embeds, _ in rows)
            pads = [length - embeds.shape[0] for embeds, _ in rows]
            padded = torch.stack([torch.cat([embeds.new_zeros((pad, embeds.shape[1])), embeds]) for (embeds, _), pad in zip(rows, pads)])
            mask = torch.tensor([[0] * pad + [1] * (length - pad) for pad in pads])
            shifted = [{"positions": [position + pad for position in payload["positions"]], "layer_inputs": payload["layer_inputs"]} for (_, payload), pad in zip(rows, pads)]
            together = generate_batch_with_interventions(peft_model, inputs_embeds=padded, attention_mask=mask, interventions=shifted, **generation)
            batched_calls, seen[:] = list(seen), []
            alone = [generate_with_interventions(peft_model, inputs_embeds=embeds[None], interventions=[payload], **generation)[0] for embeds, payload in rows]
    finally:
        handle.remove()
    assert batched_calls == [(True, length)] + [(False, 1)] * 3, "the prefill carries the payload; the decode steps run without the hook work"
    assert all(carried for carried, _ in seen), "the single-sequence path binds it at every step (and adds nothing after the prefill)"
    assert together.tolist() == [row.tolist() for row in alone]
    assert "prepare_inputs_for_generation" not in peft_model.get_base_model().__dict__, "the binding is removed"
    with pytest.raises(ValueError, match="left-padded"):
        generate_batch_with_interventions(peft_model, inputs_embeds=padded, attention_mask=mask.flip(-1), interventions=shifted, **generation)
    with pytest.raises(ValueError, match="one payload per row"):
        generate_batch_with_interventions(peft_model, inputs_embeds=padded, attention_mask=mask, interventions=shifted[:1], **generation)
    with pytest.raises(ValueError, match="greedy"):
        generate_batch_with_interventions(peft_model, inputs_embeds=padded, attention_mask=mask, interventions=shifted, do_sample=True, max_new_tokens=2)


def test_multimodal_wrapper_batches_like_single_rows(decode):
    """A real Qwen3.5 reader may load as the multimodal wrapper, whose generate keeps `rope_deltas` of the last batch size:
    per-step logits of the padded batch equal the single rows', and the wrapper's state is put back."""
    try:
        from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration
        text = json.loads((decode.model_dir / "config.json").read_text())
        vision = {"depth": 1, "hidden_size": 32, "intermediate_size": 64, "num_heads": 4, "patch_size": 4, "temporal_patch_size": 1,
                  "spatial_merge_size": 1, "out_hidden_size": 32, "in_channels": 3, "num_position_embeddings": 16, "deepstack_visual_indexes": []}
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(5)
            model = Qwen3_5ForConditionalGeneration(Qwen3_5Config(text_config=text, vision_config=vision)).eval()
    except Exception as error:                                                     # noqa: BLE001 - the tiny vision tower is best effort
        pytest.skip(f"tiny multimodal Qwen3.5 not constructible here: {error!r}")
    generator = torch.Generator().manual_seed(1)
    lengths = [70, 9, 131, 40]
    rows = [torch.randn(length, 32, generator=generator) * 0.05 for length in lengths]
    payloads = [{"positions": list(range(length - 6, length - 1)), "layer_inputs": {layer: torch.randn(5, 32, generator=generator) * 0.05 for layer in (2, 5)}} for length in lengths]
    payloads[1] = None
    generation = dict(max_new_tokens=6, min_new_tokens=6, do_sample=False, use_cache=True, pad_token_id=1, output_logits=True, return_dict_in_generate=True)
    def single(row, payload):
        if payload is None:
            return model.generate(inputs_embeds=row[None], **generation)
        return generate_with_interventions(model, inputs_embeds=row[None], interventions=[payload], **generation)
    with torch.no_grad():
        alone = [single(row, payload) for row, payload in zip(rows, payloads)]
        plain = [single(row, None) for row in rows]
        length = max(lengths)
        pads = [length - row.shape[0] for row in rows]
        padded = torch.stack([torch.cat([row.new_zeros((pad, 32)), row]) for row, pad in zip(rows, pads)])
        mask = torch.tensor([[0] * pad + [1] * (length - pad) for pad in pads])
        shifted = [None if payload is None else {"positions": [position + pad for position in payload["positions"]], "layer_inputs": payload["layer_inputs"]}
                   for payload, pad in zip(payloads, pads)]
        marker = torch.full((3, 1), 7)
        model.model.rope_deltas = marker
        together = generate_batch_with_interventions(model, inputs_embeds=padded, attention_mask=mask, interventions=shifted, **generation)
        assert model.model.rope_deltas is marker, "the wrapper's cached position deltas are put back"
        unshifted = generate_batch_with_interventions(model, inputs_embeds=padded, attention_mask=mask, interventions=payloads, **generation)
    logits = lambda output, row=0: torch.stack([step[row] for step in output.logits])
    scale = float(logits(alone[0]).abs().max())
    for row in range(len(rows)):
        assert together.sequences[row].tolist() == alone[row].sequences[0].tolist()
        assert float((logits(together, row) - logits(alone[row])).abs().max()) < 1e-4 * scale
    heard = [float((logits(alone[row]) - logits(plain[row])).abs().max()) for row in (0, 2, 3)]
    assert min(heard) > 1e-2 * scale, "the deltas move the logits"
    moved = [float((logits(unshifted, row) - logits(alone[row])).abs().max()) for row in (0, 3)]                # the padded rows with a payload
    assert min(moved) > 1e-3 * scale, "deltas at the unpadded positions of a padded row give other logits"


# --------------------------------------------------------------------------------------------------- 4. the review script
@pytest.fixture(scope="module")
def review():
    sys.path.insert(0, str(BENCH))
    try:
        spec = importlib.util.spec_from_file_location("review_snapshot_under_test", BENCH / "review_snapshot.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(BENCH))
    return module


def test_progress_file_keeps_every_batch_and_a_restart_continues(decode, review, tmp_path, capsys):
    entries = [(item, item.info["panel_group"]) for item in decode.items]
    keys = [review.progress_key(item, group) for item, group in entries]
    expected = dict(zip(keys, decode.texts))
    path = tmp_path / "out" / "review.progress.jsonl"
    # A decode that stops after its second batch: both finished batches are on disk, one line per entry.
    calls = []
    real = decode.trainer._sample_completions_batched
    def stopping(ac, items, progress, *, on_batch, **kwargs):
        def keep(indexes, samples):
            on_batch(indexes, samples)
            calls.append(len(indexes))
            if len(calls) == 2:
                raise KeyboardInterrupt
        return real(ac, items, progress, on_batch=keep, **kwargs)
    decode.trainer._sample_completions_batched = stopping
    try:
        with pytest.raises(KeyboardInterrupt):
            review.decode_entries(decode.trainer, decode.ac, entries, PROGRESS, path, 3, 2)
    finally:
        del decode.trainer._sample_completions_batched
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(lines) == 6 and all(expected[line["key"]] == line["completion"] for line in lines)
    with path.open("a") as f:
        f.write('{"key": "mc/tiny\\tcut')                                      # a line cut off by the stop
    # The restart decodes the two entries that are left and nothing else.
    stats: dict = {}
    done = review.decode_entries(decode.trainer, decode.ac, entries, PROGRESS, path, 3, 2, stats)
    assert stats["items"] == 2 and done == expected
    assert "6 of 8 entries already there" in capsys.readouterr().out
    assert review.read_progress(path) == expected
    assert len(path.read_text().splitlines()) == 9, "one line per entry, plus the cut one"
    stats = {}
    assert review.decode_entries(decode.trainer, decode.ac, entries, PROGRESS, path, 3, 2, stats) == expected and not stats.get("items")
    # A shard decodes its own entries; no progress file: nothing is written.
    shard = review.take_shard(entries, (1, 3))
    done = review.decode_entries(decode.trainer, decode.ac, shard, PROGRESS, None, 2, 1)
    assert done == {review.progress_key(item, group): expected[review.progress_key(item, group)] for item, group in shard} and len(done) == 3
    assert sorted(p.name for p in path.parent.iterdir()) == [path.name]


def test_compare_reports_identical_decodes(decode, review):
    from activation.dataset.scoring import exact_match
    entries = [(item, item.info["panel_group"]) for item in decode.items]
    parse = ActivationContextStudyGenerator(decode.h, decode.ac.name).dialect.parse
    report = review.compare_decodes(decode.trainer, decode.ac, parse, entries[:6], PROGRESS, 4, 2, exact_match)
    summary = report["compare"]
    assert (summary["entries"], summary["identical_completions"], summary["differing_completions"], summary["differing_answers"]) == (6, 6, 0, 0)
    assert report["differences"] == [] and summary["seconds"]["sequential"] > 0 and summary["seconds"]["batched"] > 0 and summary["speedup"] > 0
    assert summary["batched_counts"]["items"] == 6 and summary["exact_match"]["sequential"] == summary["exact_match"]["batched"]
    # A path that decodes something else is counted, entry by entry.
    real = decode.trainer._sample_completions_batched
    def other(ac, items, progress, **kwargs):
        samples = real(ac, items, progress, **kwargs)
        return [dict(sample, student=sample["student"] + " x") if index == 1 else sample for index, sample in enumerate(samples)]
    decode.trainer._sample_completions_batched = other
    try:
        report = review.compare_decodes(decode.trainer, decode.ac, parse, entries[:6], PROGRESS, 4, 2, exact_match)
    finally:
        del decode.trainer._sample_completions_batched
    assert report["compare"]["differing_completions"] == 1 and report["compare"]["identical_completions"] == 5
    assert [row["key"] for row in report["differences"]] == [review.progress_key(*entries[1])]


def test_review_script_commands(decode, review, tmp_path, monkeypatch, capsys):
    """review_snapshot.main() on the tiny models: the command as it is run today, the same command with the batched decode
    (whole and as a shard) and the compare command."""
    import pickle
    from dataclasses import asdict
    from types import SimpleNamespace
    monkeypatch.setitem(sys.modules, "run_qa", SimpleNamespace(harness=lambda ac_path, reader_path: decode.h, AC=decode.ac.name))
    panel = tmp_path / "task_panel"
    panel.mkdir()
    with (panel / "panel.pkl").open("wb") as f:
        pickle.dump({"items": decode.items}, f)
    recipe = tmp_path / "recipe.json"
    recipe.write_text(json.dumps({"training_config": {**asdict(decode.trainer.config), "schedule_total_updates": 10}}, default=str))
    def run(name, *options):
        output = tmp_path / f"{name}.json"
        monkeypatch.setattr(sys, "argv", ["review_snapshot.py", "--mode", "trained", "--ac", str(tmp_path / "ac_step_3"), "--reader", str(tmp_path / "reader_step_3"),
                                          "--update", "3", "--task-panel", str(panel), "--recipe", str(recipe), "--decode-groups", "mc/tiny", "--answers", "1000",
                                          "--max-new-tokens", str(BUDGET), "--nll-items", "1", "--output", str(output), *options])
        review.main()
        return json.loads(output.read_text())
    today = run("today", "--progress-file", str(tmp_path / "today.progress.jsonl"))
    assert [row["completion"] for row in today["answers"]["mc/tiny"]] == decode.texts and "decode" not in today
    batched = run("batched", "--progress-file", str(tmp_path / "batched.progress.jsonl"), "--decode-batch", "3", "--row-cache", "2")
    assert all(batched[key] == today[key] for key in ("answers", "answer_scores", "nll", "snapshot", "mode"))
    assert batched["decode"] == {"batch": 3, "row_cache": 2, "items": 8, "batches": 3, "encodes": 4, "cache_hits": 3, "new_tokens": sum(len(row) for row in decode.tokens)}
    assert review.read_progress(tmp_path / "batched.progress.jsonl") == review.read_progress(tmp_path / "today.progress.jsonl")
    cached = run("cached", "--row-cache", "2")                                # the cache alone, no progress file
    assert cached["answers"] == today["answers"] and cached["decode"]["batches"] == 8
    shard = run("shard", "--decode-batch", "3", "--shard", "1/2")
    assert shard["answers"]["mc/tiny"] == today["answers"]["mc/tiny"][1::2]
    capsys.readouterr()
    compared = run("compared", "--progress-file", str(tmp_path / "unused.progress.jsonl"), "--decode-batch", "4", "--row-cache", "2", "--compare-sequential", "6")
    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert compared["compare"] == summary and (summary["entries"], summary["identical_completions"], summary["differing_answers"]) == (6, 6, 0)
    assert "answers" not in compared and not (tmp_path / "unused.progress.jsonl").exists(), "the compare run decodes nothing else and leaves the progress file alone"
    for options in (["--decode-batch", "0"], ["--row-cache", "-1"], ["--compare-sequential", "4"]):
        with pytest.raises(SystemExit):
            run("refused", *options)
