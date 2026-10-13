"""Nested (two-level) reconstruction material on tiny CPU models: a passage with `segments` renders into a user message whose
content mixes text with activation_context parts (one per child segment, at the child's ratio), the plain view is the
concatenation of the segments, the "continue" cue ends inside the child, `variant_only` restricts the variant, flat
passages keep their shape; and the encoder reads a rendered nested example with the child's rows counted as side tokens.

  PYTHONPATH=. .venv/bin/python -m pytest activation/tests/test_nested_material.py -q -p no:cacheprovider
"""
import math
import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.modules.setdefault("fla", None)          # the kernels are GPU-only; the models fall back to torch
import torch

from activation.tests.test_interventions import TINY_QWEN3, build, config as training_config    # hides fla, two torch threads
from activation.ac_model import ActivationContextStudyGenerator, ActivationContextTrainer
from activation.ac_model.ac_model_utils import AC_PART_TYPE, EncodeRequest, direct_parts, tokenize_with_parts

torch.set_num_threads(2)

LEAD = "Alpha beta gamma delta.\n"
CHILD = "Epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi.\n"
TAIL = "Rho sigma tau upsilon phi chi psi omega."
NESTED = {"segments": [{"kind": "text", "text": LEAD}, {"kind": "child", "text": CHILD, "ratio": 0.5}, {"kind": "text", "text": TAIL}],
          "ratio": 0.25, "doc_id": "n1", "source": "test", "tokens": 30}
CONTINUE_ONLY = dict(NESTED, doc_id="n2", variant_only="continue")
FLAT = {"text": "one two three four five six seven eight", "doc_id": "f1", "source": "test"}


def study_for(tmp_path):
    h, ac = build(tmp_path, TINY_QWEN3)
    return h, ac, ActivationContextStudyGenerator(h, ac.name)


def user_content(item):
    return item.ac_messages[item.ac_start - 1]["content"]              # the message before the first target


def test_nested_passage_renders_parts_in_order(tmp_path):
    h, ac, study = study_for(tmp_path)
    full = LEAD + CHILD + TAIL
    items = study.generate_reconstruction_samples([NESTED, FLAT], ratio=1 / 8, variant="reconstruct")
    assert [item.doc_ids for item in items] == [["n1"], ["f1"]]
    nested, flat = items
    content = user_content(nested)
    assert [piece.get("type") for piece in content] == ["text", AC_PART_TYPE, "text"] and content[0]["text"] == "Passage:\n"
    part = content[1]
    assert part["ac_name"] == ac.name and part["compression_target"] == 0.25, "the parent's ratio is the passage's, not the call's"
    (inner,) = part["messages"]
    assert inner["role"] == "user"
    assert [piece["type"] for piece in inner["content"]] == ["text", AC_PART_TYPE, "text"]
    assert inner["content"][0] == {"type": "text", "text": LEAD} and inner["content"][2] == {"type": "text", "text": TAIL}
    child = inner["content"][1]
    assert child["ac_name"] == ac.name and child["compression_target"] == 0.5 and child["messages"] == [{"role": "user", "content": CHILD}]
    assert len(direct_parts(part["messages"])) == 1, "one part per child segment"
    # The plain (teacher) view and the target are the concatenation of the segments.
    assert nested.in_context_messages[nested.in_context_start - 1]["content"] == f"Passage:\n{full}\n\n" + content[2]["text"].lstrip("\n")
    assert nested.ac_messages[-1]["content"] == full and nested.in_context_messages[-1]["content"] == full
    assert nested.info["ratio"] == 0.25 and nested.info["children"] == 1 and nested.info["child_ratios"] == [0.5] and nested.info["chars"] == len(full)
    # A flat passage keeps its shape: a plain string inside the part, the call's ratio, no nested keys.
    flat_part = user_content(flat)[1]
    assert flat_part["compression_target"] == 1 / 8 and flat_part["messages"] == [{"role": "user", "content": FLAT["text"]}]
    assert flat.info["ratio"] == 1 / 8 and "children" not in flat.info and flat.ac_messages[-1]["content"] == FLAT["text"]


def test_continue_cue_ends_inside_the_child(tmp_path):
    h, ac, study = study_for(tmp_path)
    full = LEAD + CHILD + TAIL
    child_start, child_end = len(LEAD), len(LEAD) + len(CHILD)
    (item,) = study.generate_reconstruction_samples([NESTED], ratio=1 / 8, variant="continue", cue_fraction=0.2)
    cut = item.info["cue_chars"]
    assert child_start < cut < child_end, "the cue ends inside the child segment"
    cue, target = full[:cut].rstrip(), full[cut:].lstrip()
    assert cue.startswith(LEAD) and target.endswith(TAIL) and target.startswith(CHILD[cut - child_start:].lstrip()), "the continuation crosses from the child into the parent's text"
    instruction = user_content(item)[2]["text"]
    assert f"begins with:\n{cue}\n\n" in instruction
    assert item.ac_messages[-1]["content"] == target
    assert item.in_context_messages[item.in_context_start - 1]["content"] == f"Passage:\n{full}\n\n" + instruction.lstrip("\n")
    assert user_content(item)[1]["messages"][0]["content"][1]["compression_target"] == 0.5


def test_variant_only_restricts_the_variant(tmp_path):
    h, ac, study = study_for(tmp_path)
    reconstruct = study.generate_reconstruction_samples([NESTED, CONTINUE_ONLY, FLAT], ratio=1 / 8, variant="reconstruct")
    assert [item.doc_ids[0] for item in reconstruct] == ["n1", "f1"]
    cont = study.generate_reconstruction_samples([NESTED, CONTINUE_ONLY, FLAT], ratio=1 / 8, variant="continue")
    assert [item.doc_ids[0] for item in cont] == ["n1", "n2", "f1"]


def test_encoder_counts_child_rows_as_side_tokens(tmp_path):
    h, ac, study = study_for(tmp_path)
    (item,) = study.generate_reconstruction_samples([NESTED], ratio=1 / 8, variant="reconstruct")
    (part,) = direct_parts(item.ac_messages)
    child = ac._child_request(part, ac.config.default_compression_ratio)
    request = EncodeRequest(child.messages, child.compression_ratio, False, child.tools)
    assert request.compression_ratio == 0.25
    # Expected rows: the child's V is a placeholder run in the parent's side tokenization, so the parent's V counts it.
    (child_part,) = direct_parts(request.messages)
    child_rows = ac.part_view_rows(child_part["messages"], child_part["compression_target"])
    assert child_rows == ac.num_view_rows(len(tokenize_with_parts(ac.side.tokenizer, child_part["messages"], [], ac.pad_id)[0]), 0.5)
    ids, spans = tokenize_with_parts(ac.side.tokenizer, request.messages, [child_rows], ac.pad_id)
    assert len(spans) == 1 and spans[0][1] - spans[0][0] == child_rows and all(token == ac.pad_id for token in ids[spans[0][0]:spans[0][1]])
    text_tokens = len(ids) - child_rows
    expected = ac.num_view_rows(len(ids), 0.25)
    assert ac.config.min_view_rows < expected < ac.config.max_view_rows, "the check must not be decided by the clamp"
    assert expected == math.ceil((text_tokens + child_rows) * 0.25) == ac.part_view_rows(request.messages, 0.25)
    before = ac.stats.encodes
    (output,) = ac.encode_batch([request], with_layers=True)
    assert output.input_embeds.shape == (expected, ac.d_target) and output.layer_inputs == {}
    assert ac.stats.encodes - before == 2, "the child and the parent both go through the network"
    # The training path builds the example with the same row count in the student's single span.
    trainer = ActivationContextTrainer(h, training_config(no_context_penalty_weight=0.0))
    example = trainer.build_example(ac, item)
    assert len(example.spans) == 1 and example.spans[0][1] - example.spans[0][0] == expected
    assert [r.compression_ratio for r in example.part_requests] == [0.25]
