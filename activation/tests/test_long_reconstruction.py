"""Long multi-passage reconstruction material on tiny CPU models: compose_reconstruction_inputs fills token budgets with BM25
neighbours or random passages; a multi-passage input renders as ONE part in the RAG evidence format; the "passage" variant
reproduces Passage k, the "lookup" variant continues a cue (unique in the input) from inside one passage; item ids hash the content.

  PYTHONPATH=. .venv/bin/python -m pytest activation/tests/test_long_reconstruction.py -q -p no:cacheprovider
"""
import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.modules.setdefault("fla", None)          # the kernels are GPU-only; the models fall back to torch
import pytest
import torch

from activation.tests.test_interventions import TINY_QWEN3, build
from activation.ac_model import ActivationContextStudyGenerator
from activation.ac_model.ac_model_study import LOOKUP_INSTRUCTION, PASSAGE_INSTRUCTION, SPAN_INSTRUCTION, join_passages
from activation.ac_model.ac_model_utils import AC_PART_TYPE, assistant_target_ids, without_stop_token

torch.set_num_threads(2)

WORDS = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon".split()
POOL = [{"text": " ".join(f"{WORDS[(i + j) % len(WORDS)]}{i}-{j}" for j in range(40 + 7 * (i % 5))), "doc_id": f"d{i}", "source": "test"} for i in range(30)]
REPEATED = {"texts": [" ".join(WORDS) * 3, " ".join(WORDS)], "doc_ids": ["r1", "r2"], "source": "composed"}   # every cue recurs


def study_for(tmp_path):
    h, ac = build(tmp_path, TINY_QWEN3)
    return ac, ActivationContextStudyGenerator(h, ac.name)


def test_compose_fills_budgets(tmp_path):
    _, study = study_for(tmp_path)
    inputs = study.compose_reconstruction_inputs(POOL, 12, token_range=(200, 600), bm25_fraction=.5, seed=3)
    assert len(inputs) == 12 and {x["composition"] for x in inputs} == {"bm25", "random"}
    for x in inputs:
        assert len(set(x["texts"])) == len(x["texts"]) >= 2, "distinct passages"
        assert x["input_tokens"] <= x["budget_tokens"] and 200 <= x["budget_tokens"] <= 600
        assert len(x["doc_ids"]) == len(x["texts"]) and x["source"] == "composed"
    assert inputs == study.compose_reconstruction_inputs(POOL, 12, token_range=(200, 600), bm25_fraction=.5, seed=3), "seeded"


def test_passage_and_lookup_items(tmp_path):
    ac, study = study_for(tmp_path)
    texts = [p["text"] for p in POOL[:4]]
    joined = join_passages(texts)
    source = {"texts": texts, "doc_ids": ["a", "b", "c", "d"], "source": "composed", "composition": "random", "target_index": 2}
    (item,) = study.generate_reconstruction_samples([source], ratio=.25, variant="passage")
    content = item.ac_messages[item.ac_start - 1]["content"]
    assert [piece.get("type") for piece in content] == ["text", AC_PART_TYPE, "text"] and content[0]["text"] == "Passages:\n"
    assert content[1]["messages"] == [{"role": "user", "content": joined}], "one part holds every passage"
    assert content[2]["text"] == "\n\n" + PASSAGE_INSTRUCTION.format(k=3) and item.ac_messages[-1]["content"] == texts[2]
    assert item.in_context_messages[item.in_context_start - 1]["content"] == f"Passages:\n{joined}\n\n" + PASSAGE_INSTRUCTION.format(k=3)
    assert joined[round(item.info["target_position"] * len(joined)):].startswith(texts[2][:20])
    assert item.info["passages"] == 4 and item.doc_ids == ["a", "b", "c", "d"] and item.kind == "passage"
    lookups = study.generate_reconstruction_samples([dict(source, target_index=None)] * 5, ratio=.25, variant="lookup",
                                                    cue_tokens=4, target_tokens=(6, 10), seed=1)
    assert len({i.item_id for i in lookups}) > 1, "draws differ, ids differ"
    for item in lookups:
        k = item.info["passage_index"]
        target = item.ac_messages[-1]["content"]
        cue = item.ac_messages[item.ac_start - 1]["content"][2]["text"].split("Text: ", 1)[1]
        assert item.ac_messages[item.ac_start - 1]["content"][2]["text"] == "\n\n" + LOOKUP_INSTRUCTION.format(cue=cue)
        assert cue + target in texts[k], "cue then continuation, verbatim, inside one passage"
    again = study.generate_reconstruction_samples([dict(source, target_index=None)] * 5, ratio=.25, variant="lookup",
                                                  cue_tokens=4, target_tokens=(6, 10), seed=1)
    assert [i.item_id for i in again] == [i.item_id for i in lookups], "content-hash ids are stable"
    assert study.generate_reconstruction_samples([REPEATED], ratio=.25, variant="lookup", cue_tokens=4, target_tokens=(6, 10)) == [], \
        "an ambiguous cue (occurring twice in the input) yields no item"


def test_span_items_and_stop_masking(tmp_path):
    import pytest
    ac, study = study_for(tmp_path)
    texts = [p["text"] for p in POOL[:4]]
    source = {"texts": texts, "doc_ids": ["a", "b", "c", "d"], "source": "composed", "composition": "random"}
    spans = study.generate_reconstruction_samples([source] * 6, ratio=.25, variant="span", cue_tokens=(2, 4), target_tokens=(5, 9), seed=2)
    assert spans and {i.kind for i in spans} == {"span"}
    tokenizer = study.tokenizer
    for item in spans:
        k = item.info["passage_index"]
        instruction = item.ac_messages[item.ac_start - 1]["content"][2]["text"]
        cue, end_cue = instruction.split("Start: ", 1)[1].split("\nEnd: ")
        assert instruction == "\n\n" + SPAN_INSTRUCTION.format(cue=cue, end_cue=end_cue)
        target = item.ac_messages[-1]["content"]
        assert cue + target + end_cue in texts[k], "start cue, the span, then the end cue, verbatim inside one passage"
        assert join_passages(texts).count(cue) == 1 and join_passages(texts).count(end_cue) == 1, "both cues unique in the input"
        assert 2 <= item.info["cue_tokens"] <= 4 and 2 <= item.info["end_cue_tokens"] <= 4
        assert 5 <= len(tokenizer.encode(target, add_special_tokens=False)) <= 9
        assert item.assistant_token_ids[0] == assistant_target_ids(tokenizer, item.ac_messages[-1]), "a span keeps its (predictable) stop"
    assert len({i.info["cue_tokens"] for i in spans} | {i.info["end_cue_tokens"] for i in spans}) > 1, "cue lengths vary within the range"
    lookups = study.generate_reconstruction_samples([source] * 4, ratio=.25, variant="lookup", cue_tokens=(2, 4), target_tokens=(5, 9), seed=3)
    masked = study.generate_reconstruction_samples([source] * 4, ratio=.25, variant="lookup", cue_tokens=(2, 4), target_tokens=(5, 9), seed=3, mask_stop=True)
    assert len(masked) == len(lookups) > 0
    for plain, item in zip(lookups, masked):
        full = assistant_target_ids(tokenizer, item.ac_messages[-1])
        assert plain.assistant_token_ids[0] == full and item.assistant_token_ids[0] == without_stop_token(tokenizer, full)
        assert len(item.assistant_token_ids[0]) < len(full) and item.info["stop_masked"] and "stop_masked" not in plain.info
        assert item.ac_messages == plain.ac_messages and item.item_id != plain.item_id, "same sample, its own id"
    with pytest.raises(ValueError):
        study.generate_reconstruction_samples([source], ratio=.25, variant="span", cue_tokens=(2, 4), target_tokens=(5, 9), mask_stop=True)
    with pytest.raises(ValueError):
        without_stop_token(tokenizer, without_stop_token(tokenizer, full))
    assert study.generate_reconstruction_samples([REPEATED], ratio=.25, variant="span", cue_tokens=3, target_tokens=(4, 6)) == [], \
        "repeated cues yield no span either"
    fixed = study.generate_reconstruction_samples([source] * 6, ratio=.25, variant="lookup", cue_tokens=4, target_tokens=(6, 10), seed=1)
    same = study.generate_reconstruction_samples([source] * 6, ratio=.25, variant="lookup", cue_tokens=(4, 4), target_tokens=(6, 10), seed=1)
    assert [i.item_id for i in fixed] == [i.item_id for i in same], "a degenerate range draws nothing: fixed-cue lookups keep their draws"



def test_multispan_items(tmp_path):
    """The "multispan" variant: one passage alone, several non-overlapping reads listed in random order and reproduced in that
    order joined by the separator; a start read begins the passage (its opening is never in the instruction), an end read ends
    it; every cue is unique; seeded draws reproduce; a passage holding the separator is skipped."""
    from activation.ac_model.ac_model_study import MULTISPAN_SEPARATOR
    _, study = study_for(tmp_path)
    rows = [dict(p, ratio=.25) for p in POOL]
    items = study.generate_reconstruction_samples(rows, ratio=.25, variant="multispan", seed=3, cue_tokens=(3, 5), target_tokens=(4, 9),
                                                  reads=(2, 5), boundary_share=.5)
    assert len(items) >= 15 and len({i.item_id for i in items}) == len(items)   # short synthetic passages: some draws find no unique cue
    kinds = set()
    for item in items:
        passage = next(p["text"] for p in POOL if p["doc_id"] == item.doc_ids[0])
        reads, order = item.info["reads"], item.info["read_order"]
        parts = item.ac_messages[-1]["content"].split(MULTISPAN_SEPARATOR)
        assert len(parts) == len(reads) == len(order) and 2 <= len(reads) <= 5 and sorted(order) == list(range(len(reads)))
        content = item.in_context_messages[item.in_context_start - 1]["content"]
        instruction = content[content.index("Reproduce the following parts"):]   # the instruction block after the passage
        assert item.kind == "multispan" and "1. From" in instruction and f"{len(reads)}. From" in instruction
        for j, part in zip(order, parts):
            kinds.add(reads[j]["kind"])
            assert part.strip() and passage.count(part) >= 1 and 4 <= reads[j]["tokens"] <= 9
            if reads[j]["kind"] == "start":
                assert passage.startswith(part) and reads[j]["cue_tokens"] == 0 and part not in instruction
            if reads[j]["kind"] == "end":
                assert passage.endswith(part) and reads[j]["end_cue_tokens"] == 0
        assert [r["start"] for r in reads] == sorted(r["start"] for r in reads), "reads recorded in passage order"
        assert all(reads[j]["start"] + reads[j]["cue_tokens"] + reads[j]["tokens"] + reads[j]["end_cue_tokens"] <= reads[j + 1]["start"] for j in range(len(reads) - 1)), "non-overlapping"
    assert kinds == {"start", "mid", "end"}
    again = study.generate_reconstruction_samples(rows, ratio=.25, variant="multispan", seed=3, cue_tokens=(3, 5), target_tokens=(4, 9), reads=(2, 5), boundary_share=.5)
    assert [i.item_id for i in again] == [i.item_id for i in items], "seeded"
    assert not study.generate_reconstruction_samples([dict(POOL[0], text=POOL[0]["text"] + MULTISPAN_SEPARATOR + "tail", ratio=.25)], ratio=.25, variant="multispan", seed=1,
                                                     cue_tokens=(3, 5), target_tokens=(4, 9)), "a passage holding the separator is skipped"
    with pytest.raises(ValueError, match="one passage alone"):
        study.generate_reconstruction_samples([{"texts": [POOL[0]["text"], POOL[1]["text"]], "doc_ids": ["a", "b"], "source": "composed"}], ratio=.25, variant="multispan")
    with pytest.raises(ValueError, match="reads"):
        study.generate_reconstruction_samples(rows[:1], ratio=.25, variant="multispan", reads=(0, 2))
    short = [{"text": " ".join(f"{w}{i}" for i, w in enumerate(WORDS[:12])), "doc_id": "s", "source": "test", "ratio": .25}] * 6   # ~24 tokens: one or two reads fit
    few = study.generate_reconstruction_samples(short, ratio=.25, variant="multispan", seed=5, cue_tokens=(2, 3), target_tokens=(4, 6), reads=(5, 5))
    assert few and all(1 <= len(i.info["reads"]) <= 2 for i in few), "a short passage keeps the reads that fit"
    single = study.generate_reconstruction_samples(rows, ratio=.25, variant="multispan", seed=7, cue_tokens=(3, 5), target_tokens=(4, 9), reads=(1, 1), boundary_share=1.)
    assert single and all(len(i.info["reads"]) == 1 for i in single), "K never exceeds the reads range"
