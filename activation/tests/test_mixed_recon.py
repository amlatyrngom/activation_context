"""Mixed span reconstruction arms of the reconstruction bench (run_recon.py, arms with `mix`) on tiny CPU models: the example plan
(shares, the one-span quarter, span lengths, no repeated targets, log-uniform ratios, determinism, the same columns at every
size), the held panel (the same items at every ratio, held material only), the schedule (constant, linear landing, no warm-up
replay in a continuation), a mid-run resume that sees the planned remaining examples (and, for the shuffled arms, the same
order as an uninterrupted run), the kept state and the continuation that starts from it, the frozen-reader arm, the CPU token
table, and the earlier arms' unchanged configuration.

  PYTHONPATH=. .venv/bin/python -m pytest activation/tests/test_mixed_recon.py -q -p no:cacheprovider
"""
import collections
import importlib.util
import json
import math
import os
import random
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.modules.setdefault("fla", None)          # the kernels are GPU-only; the models fall back to torch
import pytest
import torch
from transformers import AutoTokenizer

from activation.ac_model import ActivationContextTrainer, ActivationContextTrainingConfig, ActivationContextTrainingReporter
from activation.ac_model.ac_model_study import MULTISPAN_PASSAGES_INSTRUCTION, MULTISPAN_SEPARATOR, join_passages
from activation.agent_training.agent_training_utils import lr_factor
from activation.common.ac_parts import direct_parts
from activation.tests.test_interventions import TINY_QWEN3, build, tiny_hybrid

torch.set_num_threads(2)
BENCH = Path(__file__).resolve().parents[2] / "IB/ARTIFACTS/AGENT_AC_PRETRAINING/ac_recon/run_recon.py"
MIX = {"whole": .2, "short_spans": .4, "long_spans": .4}


class Interrupted(BaseException):
    """A kill right after a mid-run checkpoint was published."""


def load_bench(report_root: Path):
    previous = os.environ.get("AC_RECON_ROOT")
    os.environ["AC_RECON_ROOT"] = str(report_root)
    try:
        spec = importlib.util.spec_from_file_location(f"run_recon_{abs(hash(str(report_root)))}", BENCH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        os.environ.pop("AC_RECON_ROOT") if previous is None else os.environ.__setitem__("AC_RECON_ROOT", previous)
    return module


def material(train: int, held: int, seed: int = 5) -> dict:
    """Passages of 8-90 words of the tiny tokenizer's vocabulary (one token per word): `tokens` is exact for both tiny tokenizers."""
    words = [word for word in AutoTokenizer.from_pretrained(str(TINY_QWEN3)).get_vocab() if not word.startswith("<")]
    rng = random.Random(seed)
    def passage(index, split):
        count = rng.randint(8, 90)
        return {"text": " ".join(rng.choice(words) for _ in range(count)), "doc_id": f"{split}{index}", "source": rng.choice(["a", "b"]), "tokens": count}
    return {"train": [passage(i, "t") for i in range(train)], "held": [passage(i, "h") for i in range(held)], "wiki": []}


def tiny_arm(rr, **values) -> dict:
    arm = dict(rr.DEFAULT_ARM, mix=MIX, span_tokens=(4, 8), span_cue_tokens=(2, 3), long_tokens=(150, 400), updates=12, examples_per_update=4, warmup=2, landing=4,
               beat_every=4, ckpt_every=2, panel_items=2, epochs=1, material="material.json", intervention_layers=[1], deltas_off=False, completion_samples=1,
               completion_tokens=64, final_train_subset=4, encoder_lr=1e-2, reader_lr=1e-2, penalty_fraction=1.0, ckpt_min_tokens=0, side_model=str(TINY_QWEN3),
               reader_model=str(TINY_QWEN3))
    return dict(arm, **values)


def part_text(item) -> str:
    (part,) = direct_parts(item.ac_messages)
    return part["messages"][0]["content"]


# --------------------------------------------------------------------------------------------------- the bench on tiny models
@pytest.fixture(scope="module")
def bench(tmp_path_factory):
    """The bench module on a temporary report root with tiny models (a 2-layer Qwen3 reader, a narrower Qwen3.5-style encoder),
    a helper that runs an arm under its own checkpoint root (optionally killed right after the checkpoint of a step), and the
    uninterrupted mixed run 'T' (12 updates of 4, panel every 4, checkpoint every 2, landing over the last 4)."""
    tmp = tmp_path_factory.mktemp("mixed")
    patch = pytest.MonkeyPatch()
    patch.delenv("AC_RECON_KEEP_ROOT", raising=False)
    rr = load_bench(tmp / "SYNC" / "AC_RECON")
    data = material(400, 80)
    (rr.REPORT_ROOT / "preparation").mkdir(parents=True)
    (rr.REPORT_ROOT / "preparation" / "material.json").write_text(json.dumps(data))
    side_dir = tiny_hybrid(tmp / "side_model", hidden=16, seed=222)
    base = tiny_arm(rr)
    old = dict(rr.DEFAULT_ARM, train_items=32, epochs=1, reporting_interval=0.5, ckpt_every=1, held_per_source=2, material="material.json", completion_samples=1, completion_tokens=64,
               final_train_subset=4, encoder_lr=1e-2, reader_lr=1e-2, ckpt_min_tokens=0, intervention_layers=[1], deltas_off=False, ratio=1 / 4)
    warm = {"init_ac": str(tmp / "SYNC_RUNS" / "T" / "AC_MODELS" / "ac_private" / "epoch_001"), "init_reader": str(tmp / "SYNC_RUNS" / "T" / "LORAS" / "reader_lora" / "epoch_001")}
    table = {"T": base, "Tr": dict(base), "Tm": dict(base), "Tn": dict(base), "Tc": dict(base, continue_from="T", continue_at=8, updates=20), "Tx": dict(base, continue_from="Tn", continue_at=8, updates=20),
             "Tf": dict(base, reader_lr=0.0),
             "O": old, "Or": dict(old), "W": dict(old, **warm), "Wr": dict(old, **warm)}
    patch.setattr(rr, "arms", lambda: {label: dict(arm) for label, arm in table.items()})
    for name, value in (("AC", "ac_private"), ("READER", "reader_lora"), ("NAME", "reader")):
        patch.setattr(rr, name, value)
    def harness(with_ac=False, ac_path=None, reader_path=None, seed=rr.SEED, intervention_layers=None, **_):
        h, _ = build(Path(os.environ["ACTIVATION_SYNC_ROOT"]), TINY_QWEN3, seed=seed, side_dir=side_dir, layers=intervention_layers, min_view_rows=2,
                     ac_checkpoint=str(ac_path) if ac_path else None, reader_checkpoint=str(reader_path) if reader_path else None)
        return h
    patch.setattr(rr, "harness", harness)
    state = {"label": None, "stop": None}
    steps, stats = collections.defaultdict(list), {}
    report_step, train, save_partial = ActivationContextTrainingReporter.report_step, ActivationContextTrainer.train, ActivationContextTrainer._save_partial
    def recording_step(self, progress, run_stats):
        size = len(run_stats.example_records) // run_stats.steps
        steps[state["label"]].append((progress.global_step, tuple(sorted(record["item_id"] for record in run_stats.example_records[-size:])), tuple(run_stats.learning_rates[-1])))
        return report_step(self, progress, run_stats)
    def recording_train(self, *args, **kwargs):
        result = train(self, *args, **kwargs)
        stats[state["label"]] = result
        return result
    def interrupting_save(self, ac_model, epoch_number, step, total):
        result = save_partial(self, ac_model, epoch_number, step, total)
        if state["stop"] == step:
            raise Interrupted(step)
        return result
    patch.setattr(ActivationContextTrainingReporter, "report_step", recording_step)
    patch.setattr(ActivationContextTrainer, "train", recording_train)
    patch.setattr(ActivationContextTrainer, "_save_partial", interrupting_save)
    def run(label, stop=None) -> bool:
        patch.setenv("ACTIVATION_SYNC_ROOT", str(tmp / "SYNC_RUNS" / label))
        state.update(label=label, stop=stop)
        try:
            rr.run(label)
        except Interrupted:
            return False
        return True
    def beats(label):
        return [json.loads(line) for line in (rr.REPORT_ROOT / label / "beats.jsonl").read_text().splitlines()]
    assert run("T")
    yield SimpleNamespace(rr=rr, tmp=tmp, run=run, steps=steps, stats=stats, table=table, material=data, beats=beats,
                          study=rr.token_study(AutoTokenizer.from_pretrained(str(TINY_QWEN3))))
    patch.undo()


# --------------------------------------------------------------------------------------------------- (a) the example plan
def test_example_plan():
    rr = load_bench(Path("/nonexistent/AC_RECON"))
    tokenizer = AutoTokenizer.from_pretrained(str(TINY_QWEN3))
    study, data = rr.token_study(tokenizer), material(2600, 80)
    arm = tiny_arm(rr, updates=200, landing=40, beat_every=20, ckpt_every=10)
    child = dict(arm, continue_from="P", continue_at=160, updates=360)
    table = {"P": arm, "C": child}
    built = list(rr.mix_block(study, arm, data, 0, 200, 0))
    slots, items = [slot for slot, _ in built], [item for _, item in built]
    assert len(built) == 800 and {(slot["update"], slot["column"]) for slot in slots} == {(u, c) for u in range(200) for c in range(4)}
    # Shares, exact in every column: 20 % whole, 40 % short spans, 40 % long spans; one span in a quarter of the span examples.
    for column in range(4):
        kinds = collections.Counter(slot["kind"] for slot in slots if slot["column"] == column)
        assert kinds == {"whole": 40, "short_1": 20, "short_n": 60, "long_1": 20, "long_n": 60}
    for slot, item in built:
        assert item.kind == slot["kind"] and item.info["ratio"] == slot["ratio"] and 0.01 <= slot["ratio"] <= 0.1
        text, target = part_text(item), data["train"][slot["target"]]["text"]
        if slot["kind"] == "whole":
            assert text == target and item.ac_messages[-1]["content"] == target and not slot["context"]
            continue
        reads = item.info["reads"]
        assert data["train"][slot["target"]]["tokens"] >= 2 * 3 + 8, "span targets hold two cues and a full span"
        assert len(reads) == 1 if slot["kind"].endswith("_1") else 1 <= len(reads) <= 5
        assert all(4 <= read["tokens"] <= 8 and read["cue_tokens"] in (0, 2, 3) and read["end_cue_tokens"] in (0, 2, 3) for read in reads)
        parts = item.ac_messages[-1]["content"].split(MULTISPAN_SEPARATOR)
        assert len(parts) == len(reads) and all(part in target for part in parts), "every span is read from the target passage"
        if slot["kind"].startswith("long"):
            passages = [data["train"][index]["text"] for index in [slot["target"], *slot["context"]]]
            shown = [piece.split(":\n", 1)[1] for piece in ("\n\n" + text).split("\n\nPassage ")[1:]]
            assert item.info["passages"] == len(passages) >= 2 and join_passages(shown) == text and sorted(shown) == sorted(passages), "the target among other training passages"
            assert shown[item.info["passage_index"]] == target
            assert all(read["kind"] == "mid" for read in reads) and item.ac_messages[1]["content"][2]["text"].startswith("\n\n" + MULTISPAN_PASSAGES_INSTRUCTION[:40])
            assert item.info["budget_tokens"] <= 400 and len(tokenizer(text)["input_ids"]) <= 400 + 8, "the composed input stays within the long range"
        else:
            assert text == target and not slot["context"]
    counts = collections.Counter(len(item.info["reads"]) for slot, item in built if slot["kind"].endswith("_n"))
    assert set(counts) == {1, 2, 3, 4, 5} and counts[1] < 0.15 * sum(counts.values()), "2..5 spans, fewer only where the passage is short"
    roomy = [len(item.info["reads"]) for slot, item in built if slot["kind"] == "short_n" and data["train"][slot["target"]]["tokens"] >= 5 * 14]
    assert min(roomy) >= 2 and max(collections.Counter(roomy).values()) < 0.45 * len(roomy), "uniform 2..5 where five spans fit"
    # No passage is a target twice, within the block and across the continuation's block; contexts may repeat.
    targets = [slot["target"] for slot in slots]
    assert len(set(targets)) == 800
    chain = rr.mix_check("C", table)
    assert [(label, first, last) for label, _, first, last in chain] == [("P", 0, 200), ("C", 160, 360)] and rr.mix_offset(chain, 4) == 4 * 250
    later = [slot["target"] for slot, _ in rr.mix_block(study, child, data, 1, 200, rr.mix_offset(chain, 4))]
    assert len(set(later)) == 800 and not set(later) & set(targets), "the continuation's targets are fresh"
    assert collections.Counter(index for slot in slots for index in slot["context"]).most_common(1)[0][1] > 1, "context passages reappear"
    # Ratios: log-uniform in [1/100, 1/10] (Kolmogorov-Smirnov against the log-uniform law).
    logs = sorted((math.log(slot["ratio"]) - math.log(0.01)) / math.log(10) for slot in slots)
    assert max(abs((i + 1) / len(logs) - value) for i, value in enumerate(logs)) < 0.06
    # Deterministic from the seed; a column is the same whatever the number of columns; another seed gives another plan.
    again = list(rr.mix_block(study, arm, data, 0, 200, 0))
    assert [item.item_id for _, item in again] == [item.item_id for item in items]
    wide = {(slot["update"], slot["column"]): item.item_id for slot, item in rr.mix_block(study, arm, data, 0, 200, 0, 6)}
    assert len(wide) == 1200 and all(wide[slot["update"], slot["column"]] == item.item_id for slot, item in built), "the first columns of a wider plan"
    other = list(rr.mix_block(study, dict(arm, seed=7), data, 0, 200, 0))
    assert {slot["target"] for slot, _ in other} != set(targets)
    # The lanes must fit the material: a size that does not fails with the numbers.
    with pytest.raises(RuntimeError, match="2600 in material.json"):
        rr.mix_slots(arm, data, 0, 200, 0, 11)
    with pytest.raises(ValueError, match="kept state"):
        rr.mix_check("C", {"P": arm, "C": dict(child, continue_at=120)})
    with pytest.raises(ValueError, match="differ"):
        rr.mix_check("C", {"P": arm, "C": dict(child, examples_per_update=6)})


def test_held_panel():
    rr = load_bench(Path("/nonexistent/AC_RECON"))
    study, data = rr.token_study(AutoTokenizer.from_pretrained(str(TINY_QWEN3))), material(300, 80)
    arm = tiny_arm(rr, panel_items=3)
    panel, floor = rr.mix_panel(study, arm, data)
    names = ["10", "20", "50", "100"]
    assert collections.Counter(item.kind for item in panel) == {f"{kind}@{name}": 3 for kind in rr.MIX_KINDS for name in names} and len(floor) == 15
    assert [item.kind for item in panel[:3]] == ["whole@10"] * 3, "the completion samples are whole passages"
    held_texts, train_texts = [p["text"] for p in data["held"]], [p["text"] for p in data["train"]]
    used = collections.defaultdict(set)
    for kind in rr.MIX_KINDS:
        cells = [[item for item in panel if item.kind == f"{kind}@{name}"] for name in names]
        for column in zip(*cells):     # the same item at the four ratios: the input, the instruction and the target are equal, the ratio alone differs
            assert len({part_text(item) for item in column}) == 1 and len({item.ac_messages[1]["content"][2]["text"] for item in column}) == 1
            assert len({json.dumps(item.assistant_token_ids) for item in column}) == 1 and len({item.item_id for item in column}) == 4
            assert [item.info["ratio"] for item in column] == [1 / 10, 1 / 20, 1 / 50, 1 / 100]
            assert [direct_parts(item.ac_messages)[0]["compression_target"] for item in column] == [1 / 10, 1 / 20, 1 / 50, 1 / 100]
            text = part_text(column[0])
            assert not any(passage in text for passage in train_texts), "held contexts hold held passages only"
            inside = [passage for passage in held_texts if passage in text]
            assert len(inside) == (1 if kind in ("whole", "short_1", "short_n") else column[0].info["passages"])
            used[kind].add(column[0].ac_messages[-1]["content"])
    assert sum(len(values) for values in used.values()) == 15
    for item in floor:
        assert item.kind in rr.MIX_KINDS and not direct_parts(item.ac_messages) and item.item_id.endswith(":no_context")
    small, _ = rr.mix_panel(study, dict(arm, panel_items=2), data)
    assert {item.item_id for item in small} < {item.item_id for item in panel}, "a smaller panel is a prefix of the larger one"
    assert [item.item_id for item in rr.mix_panel(study, dict(arm, seed=9), data)[0]] == [item.item_id for item in panel], "the panel does not move with the arm's seed"
    with pytest.raises(RuntimeError, match="also in the training pool"):
        rr.mix_panel(study, arm, dict(data, train=data["train"] + data["held"][:1]))


# --------------------------------------------------------------------------------------------------- (b) the schedule
def test_schedule(bench):
    rr = load_bench(Path("/nonexistent/AC_RECON"))
    arms = rr.arms()
    factor = lambda config, update: lr_factor(update, config.warmup_updates, total_updates=config.schedule_total_updates, schedule=config.lr_schedule,
                                              final_fraction=config.lr_final_fraction, cooldown_updates=config.cooldown_updates, cooldown_shape=config.cooldown_shape)
    config = rr.config(arms["G4"])
    assert (config.lr_schedule, config.cooldown_updates, config.cooldown_shape, config.lr_final_fraction, config.schedule_total_updates) == ("cooldown", 500, "linear", .01, 5000)
    assert (config.warmup_updates, config.learning_rate_ac, config.learning_rate_target_lora, config.examples_per_update, config.num_epochs) == (50, 1e-4, 2e-5, 8, 1)
    assert config.shuffle_examples is False and config.checkpoint_every_updates == 250 and config.penalty_active and not config.resume_optimizer
    assert [factor(config, update) for update in (0, 24, 49)] == [1 / 50, 1 / 2, 1.0], "50 warm-up updates"
    assert all(factor(config, update) == 1.0 for update in range(49, 4501)), "constant to update 4,500"
    landing = [factor(config, update) for update in range(4500, 5001)]
    assert all(a > b for a, b in zip(landing, landing[1:])) and landing[250] == pytest.approx(0.505) and landing[-1] == pytest.approx(0.01), "linear to 1 % at 5,000"
    from activation.ac_model.ac_model_training import reporting_boundaries
    assert reporting_boundaries(5000, config.reporting_interval) == list(range(500, 5001, 500))
    assert reporting_boundaries(30, rr.config(arms["G4s"]).reporting_interval) == [10, 20, 30] and rr.config(arms["G4s"]).checkpoint_every_updates == 10
    for label, size in (("G4", 8), ("G4_e12", 12), ("G4_e16", 16)):
        assert rr.config(arms[label]).examples_per_update == size and {k: v for k, v in arms[label].items() if k != "examples_per_update"} == {k: v for k, v in arms["G4"].items() if k != "examples_per_update"}
    # The continuation: the absolute clock continues, so no warm-up replay; constant to 9,000, its own landing to 9,500; it restores the optimizer.
    continued = rr.config(arms["G4c"])
    assert continued.resume_optimizer and continued.schedule_total_updates == 9500 and continued.warmup_updates == 50
    assert all(factor(continued, update) == 1.0 for update in range(4500, 9001)) and factor(continued, 9250) == pytest.approx(0.505) and factor(continued, 9500) == pytest.approx(0.01)
    assert reporting_boundaries(5000, continued.reporting_interval) == list(range(500, 5001, 500)), "beats at 5,000, 5,500, ... 9,500"
    assert [(label, first, last) for label, _, first, last in rr.mix_check("G4c")] == [("G4", 0, 5000), ("G4c", 4500, 9500)]
    # The ablations differ from G4 in the reader rate and the layers only; a frozen reader carries no penalty.
    for label, layers, reader in (("G4", [3, 7, 11, 15, 19, 23, 27, 31, 35], 2e-5), ("G4_in", None, 2e-5), ("G4_in_frozen", None, 0.0), ("G4_frozen", [3, 7, 11, 15, 19, 23, 27, 31, 35], 0.0)):
        arm = arms[label]
        assert (arm["intervention_layers"], arm["reader_lr"], arm["interventions"]) == (layers, reader, 0) and rr.config(arm).penalty_active == (reader > 0)
        assert {k: v for k, v in arm.items() if k not in ("intervention_layers", "reader_lr", "note")} == {k: v for k, v in arms["G4"].items() if k not in ("intervention_layers", "reader_lr", "note")}
        rr.mix_check(label)
    # The trainer followed it on the tiny run: 2 warm-up updates, constant, linear landing over the last 4 of 12.
    rates = [rate[0] for _, _, rate in bench.steps["T"]]
    assert rates == pytest.approx([1e-2 * value for value in (.5, 1, 1, 1, 1, 1, 1, 1, 1, 1 - .99 / 4, 1 - .99 / 2, 1 - .99 * 3 / 4)])


# --------------------------------------------------------------------------------------------------- beats, snapshots, the kept state
def test_beats_and_states(bench):
    rr, tmp = bench.rr, bench.tmp
    records = bench.beats("T")
    assert [record["update"] for record in records] == [0, 4, 8, 12] and not any(record["restart"] for record in records)
    for record in records:
        assert set(record["loss"]) == set(rr.MIX_KINDS) and all(set(cells) == {"10", "20", "50", "100"} for cells in record["loss"].values())
        cells = [value for kind in rr.MIX_KINDS for value in record["loss"][kind].values()]
        assert record["loss_mean"] == pytest.approx(sum(cells) / 20) and record["loss_by_kind"]["whole"] == pytest.approx(sum(record["loss"]["whole"].values()) / 4)
        assert record["loss_by_ratio"]["50"] == pytest.approx(sum(record["loss"][kind]["50"] for kind in rr.MIX_KINDS) / 5)
        assert set(record["floor"]) == set(rr.MIX_KINDS) and record["floor_measured"] and 0 <= record["token_accuracy"] <= 1 and record["items_per_cell"] == 2
    plan = json.loads((rr.REPORT_ROOT / "T" / "plan.json").read_text())
    assert plan["examples"] == 48 and plan["by_kind"] == {"whole": 8, "short_1": 4, "short_n": 16, "long_1": 4, "long_n": 16}
    # The trained examples are the planned ones, update by update.
    planned = [item.item_id for item in rr.mix_training(bench.study, bench.table["T"], bench.material, "T")[0]]
    assert [ids for _, ids, _ in bench.steps["T"]] == [tuple(sorted(planned[4 * step:4 * step + 4])) for step in range(12)]
    assert [update for update, _, _ in bench.steps["T"]] == list(range(1, 13))
    # Weights-only snapshots at every beat, kept; the full state before the landing (update 8) kept with its optimizer.
    root = tmp / "SYNC_RUNS" / "T"
    assert sorted(path.name for path in (root / "BEATS").iterdir()) == ["PRESERVE", "update_00000", "update_00004", "update_00008", "update_00012"]
    assert not list((root / "BEATS").rglob("optimizer.pt")) and all((root / "BEATS" / f"update_{update:05d}" / "ac" / "ac_modules.pt").exists() for update in (0, 4, 8, 12))
    kept = tmp / "SYNC" / "KEEP_STATE" / "T" / "update_00008"
    assert rr.KEEP_ROOT == tmp / "SYNC" / "KEEP_STATE" and (kept / "PRESERVE").exists() and json.loads((kept / "kept.json").read_text())["update"] == 8
    assert torch.load(kept / "ac" / "optimizer.pt", map_location="cpu", weights_only=False)["updates_done"] == 8
    assert not (root / "PRESERVE").exists() and not list((root / "AC_MODELS" / "ac_private").glob("epoch_001_step_*")), "the run's own mid-run checkpoints are still pruned"
    modules = lambda folder: torch.load(folder / "ac_modules.pt", map_location="cpu", weights_only=False)["modules"]
    snapshot, state = modules(root / "BEATS" / "update_00008" / "ac"), modules(kept / "ac")
    assert all(torch.equal(tensor, state[name]) for name, tensor in snapshot.items()), "snapshot and kept state hold the same weights"
    assert any(not torch.equal(tensor, modules(root / "BEATS" / "update_00012" / "ac")[name]) for name, tensor in snapshot.items())
    rates = json.loads((rr.REPORT_ROOT / "T" / "rates.json").read_text())
    assert rates["updates"] == 12 and rates["examples_by_kind"] == plan["by_kind"] and rates["seconds_per_update"]["mean"] > 0
    budget = rr.mix_budget("T", "T", sizes=(4, 8))
    assert budget["sizes"]["4"]["measured_by"] == "T" and budget["sizes"]["4"]["seconds_per_update"] == pytest.approx(rates["seconds_per_update"]["mean"])
    assert budget["sizes"]["8"]["seconds_per_update"] == pytest.approx(8 * rates["seconds_per_example"]) and budget["sizes"]["8"]["hours"] > budget["sizes"]["4"]["hours"] > 0


# --------------------------------------------------------------------------------------------------- (c) mid-run resume
def test_resume_sees_the_planned_remaining_examples(bench):
    rr = bench.rr
    assert not bench.run("Tr", stop=6), "killed right after the checkpoint of update 6"
    assert [update for update, _, _ in bench.steps["Tr"]] == list(range(1, 7))
    assert bench.run("Tr")
    assert bench.steps["Tr"] == bench.steps["T"], "the same examples at the same updates and rates as the uninterrupted run"
    assert bench.stats["Tr"].resumed_from.endswith("epoch_001_step_00006/optimizer.pt") and bench.stats["Tr"].global_steps == list(range(7, 13))
    records, whole = bench.beats("Tr"), bench.beats("T")
    assert [(record["update"], record["restart"]) for record in records] == [(0, False), (4, False), (6, True), (8, False), (12, False)]
    for record, reference in zip([records[0], records[1], records[3], records[4]], whole):
        assert record["loss_mean"] == pytest.approx(reference["loss_mean"], abs=2e-4) and record["floor"]["whole"] == pytest.approx(reference["floor"]["whole"], abs=2e-4)
    assert (bench.tmp / "SYNC" / "KEEP_STATE" / "Tr" / "update_00008" / "ac" / "optimizer.pt").exists()
    assert sorted(path.name for path in (bench.tmp / "SYNC_RUNS" / "Tr" / "BEATS").iterdir()) == ["PRESERVE", "update_00000", "update_00004", "update_00008", "update_00012"]
    result = lambda label: json.loads((rr.REPORT_ROOT / label / "result.json").read_text())
    assert result("Tr")["updates"] == 12 and result("Tr")["held"]["gold_nll"] == pytest.approx(result("T")["held"]["gold_nll"], abs=2e-4)
    # A plan that no longer matches the checkpoints refuses to resume.
    assert not bench.run("Tm", stop=2)
    bench.table["Tm"] = dict(bench.table["Tm"], span_tokens=(4, 7))
    with pytest.raises(RuntimeError, match="rebuilt plan differs"):
        bench.run("Tm")


def test_resume_keeps_the_shuffled_order(bench):
    """The earlier (shuffled, epoch-based) arms: a run killed mid-epoch and resumed trains the same examples at the same updates as
    an uninterrupted one — from scratch ('O') and warm-started from another run's epoch checkpoint ('W': absolute epoch numbers)."""
    for whole, cut in (("O", "Or"), ("W", "Wr")):
        assert bench.run(whole) and not bench.run(cut, stop=2) and bench.run(cut)
        assert len(bench.steps[whole]) == 4 and [ids for _, ids, _ in bench.steps[cut]] == [ids for _, ids, _ in bench.steps[whole]], whole
        assert [update for update, _, _ in bench.steps[cut]] == [1, 2, 3, 4] and bench.stats[cut].global_steps == [3, 4]
    assert bench.stats["W"].start_epoch == 2 and bench.stats["Wr"].start_epoch == 2, "a warm start inherits the absolute epoch number, and so does its resume"
    assert bench.stats["Or"].resumed_from.endswith("epoch_001_step_00002/optimizer.pt") and bench.stats["Wr"].resumed_from.endswith("epoch_002_step_00002/optimizer.pt")


# --------------------------------------------------------------------------------------------------- the continuation
def test_continuation_starts_from_the_kept_state(bench):
    rr, tmp = bench.rr, bench.tmp
    before = sorted(str(path) for path in (tmp / "SYNC" / "KEEP_STATE" / "T").rglob("*"))
    assert not bench.run("Tc", stop=6), "the continuation itself is killed after its checkpoint at update 14"
    arm = json.loads((rr.REPORT_ROOT / "Tc" / "arm.json").read_text())
    assert arm["init_ac"] == str(tmp / "SYNC" / "KEEP_STATE" / "T" / "update_00008" / "ac") and arm["init_reader"] == str(tmp / "SYNC" / "KEEP_STATE" / "T" / "update_00008" / "reader")
    assert bench.run("Tc"), "and resumes from its own checkpoint"
    stats, steps = bench.stats["Tc"], bench.steps["Tc"]
    assert [update for update, _, _ in steps] == list(range(9, 21)), "the update clock came with the kept optimizer state"
    assert stats.resumed_from == str(tmp / "SYNC_RUNS" / "Tc" / "AC_MODELS" / "ac_private" / "epoch_001_step_00006" / "optimizer.pt") and stats.global_steps == list(range(15, 21))
    rates = [rate[0] for _, _, rate in steps]
    assert rates[:8] == pytest.approx([1e-2] * 8) and rates[8:] == pytest.approx([1e-2 * (1 - .99 * k / 4) for k in range(4)]), "no warm-up replay; constant; its own landing"
    records = bench.beats("Tc")
    assert [(record["update"], record["restart"]) for record in records] == [(8, False), (12, False), (14, True), (16, False), (20, False)]
    parent = next(record for record in bench.beats("T") if record["update"] == 8)
    assert records[0]["loss_mean"] == pytest.approx(parent["loss_mean"], abs=2e-4), "the kept weights, on the same panel"
    assert all(records[0]["loss"][kind] == pytest.approx(parent["loss"][kind], abs=2e-3) for kind in rr.MIX_KINDS) and records[0]["floor"] == pytest.approx(parent["floor"], abs=2e-3)
    first = {item.item_id for item in rr.mix_training(bench.study, bench.table["T"], bench.material, "T")[0]}
    trained = {item_id for _, ids, _ in steps for item_id in ids}
    assert len(trained) == 48 and not trained & first, "fresh examples"
    plan = json.loads((rr.REPORT_ROOT / "Tc" / "plan.json").read_text())
    assert (plan["block"], plan["first_update"], plan["examples"]) == (1, 8, 48)
    assert (tmp / "SYNC" / "KEEP_STATE" / "Tc" / "update_00016" / "ac" / "optimizer.pt").exists(), "its own kept state before its landing"
    assert torch.load(tmp / "SYNC" / "KEEP_STATE" / "Tc" / "update_00016" / "ac" / "optimizer.pt", map_location="cpu", weights_only=False)["updates_done"] == 16
    assert sorted(path.name for path in (tmp / "SYNC_RUNS" / "Tc" / "BEATS").iterdir()) == ["PRESERVE", "update_00008", "update_00012", "update_00016", "update_00020"]
    assert sorted(str(path) for path in (tmp / "SYNC" / "KEEP_STATE" / "T").rglob("*")) == before, "the parent's kept state is read, not touched"
    assert json.loads((rr.REPORT_ROOT / "Tc" / "result.json").read_text())["updates"] == 20
    with pytest.raises(RuntimeError, match="no kept state"):
        bench.run("Tx")      # its parent never ran


# --------------------------------------------------------------------------------------------------- (d) the frozen reader
def test_frozen_reader_arm(bench, monkeypatch):
    rr, tmp = bench.rr, bench.tmp
    assert not bench.run("Tf", stop=6) and bench.run("Tf")
    assert [ids for _, ids, _ in bench.steps["Tf"]] == [ids for _, ids, _ in bench.steps["T"]], "the same data as the trained-reader arm, across the resume"
    stats = bench.stats["Tf"]
    assert not stats.penalty and not stats.no_context_reference and all(count == 0 for count in stats.penalty_counts), "no penalty, no reference pass"
    assert not (tmp / "SYNC_RUNS" / "Tf" / "AC_MODELS" / "ac_private" / "no_context_reference.pt").exists()
    result = json.loads((rr.REPORT_ROOT / "Tf" / "result.json").read_text())
    assert result["penalty_last"] is None and result["configuration"]["learning_rate_target_lora"] == 0.0 and result["updates"] == 12
    from safetensors.torch import load_file
    reader = lambda update: load_file(next((tmp / "SYNC_RUNS" / "Tf" / "BEATS" / f"update_{update:05d}" / "reader").glob("*.safetensors")))
    start = reader(0)
    for update in (4, 8, 12):
        assert all(torch.equal(tensor, reader(update)[name]) for name, tensor in start.items()), "no reader parameter moved"
    assert all(not tensor.any() for name, tensor in start.items() if "lora_B" in name), "and the frozen adapter is the base reader"
    modules = lambda update: torch.load(tmp / "SYNC_RUNS" / "Tf" / "BEATS" / f"update_{update:05d}" / "ac" / "ac_modules.pt", map_location="cpu", weights_only=False)["modules"]
    assert any(not torch.equal(tensor, modules(12)[name]) for name, tensor in modules(0).items()), "the encoder trained"
    records = bench.beats("Tf")
    assert [(record["update"], record["restart"], record["floor_measured"]) for record in records] == [(0, False, True), (4, False, False), (6, True, True), (8, False, False), (12, False, False)]
    assert len({json.dumps(record["floor"]) for record in records}) == 1, "the floor of a frozen reader does not move"
    kept = tmp / "SYNC" / "KEEP_STATE" / "Tf" / "update_00008"
    state = torch.load(kept / "ac" / "optimizer.pt", map_location="cpu", weights_only=False)
    assert state["updates_done"] == 8 and (kept / "reader" / "adapter_config.json").exists()
    # Round trip: the kept pair loads into a fresh model and gives back the snapshot's weights.
    monkeypatch.setenv("ACTIVATION_SYNC_ROOT", str(tmp / "SYNC_RUNS" / "Tf_reload"))
    h = rr.harness(True, ac_path=kept / "ac", reader_path=kept / "reader", intervention_layers=[1])
    ac = h.module_manager.get_ac_model("ac_private")
    assert all(torch.equal(tensor, ac.modules.state_dict()[name]) for name, tensor in modules(8).items())


# --------------------------------------------------------------------------------------------------- the CPU token table
def test_plan_command_matches_the_run(bench, capsys, monkeypatch):
    rr = bench.rr
    record = rr.plan("T", sizes=(2, 4, 64), reader_tokenizer=str(TINY_QWEN3))
    printed = capsys.readouterr().out
    assert "T · 4 examples per update · 12 updates" in printed and "seen tokens" in printed and "target passages 48" in printed
    assert json.loads((rr.REPORT_ROOT / "preparation" / "plan_T.json").read_text())["sizes"].keys() == {"2", "4", "64"} and "error" in record["sizes"]["64"]
    four, two = record["sizes"]["4"], record["sizes"]["2"]
    assert four["digest"] == json.loads((rr.REPORT_ROOT / "T" / "plan.json").read_text())["digest"], "the plan built on the tokenizers alone is the run's plan"
    assert four["total"]["examples"] == 48 and two["total"]["examples"] == 24 and four["target_passages"] == 48
    assert {kind: row["examples"] for kind, row in four["by_kind"].items()} == {"whole": 8, "short_1": 4, "short_n": 16, "long_1": 4, "long_n": 16}
    narrow = bench.rr.mix_training(bench.study, dict(bench.table["T"], examples_per_update=2), bench.material, "T")[1]
    assert two["digest"] == narrow["digest"], "a narrower plan is the first columns of the wider one"
    assert four["context_uses"] >= four["context_passages"] > 0 and four["context_mean_uses"] >= 1 and four["long_input_tokens"]["max"] <= 400 + 16
    continued = rr.plan("Tc", sizes=(4,), reader_tokenizer=str(TINY_QWEN3))
    assert continued["block"] == 1 and continued["sizes"]["4"]["digest"] == rr.mix_training(bench.study, bench.table["Tc"], bench.material, "Tc")[1]["digest"]
    # The table counts what the trainer builds: target tokens of the run, and per example the encoder tokens, rows and reader input of the trainer's own example.
    assert four["total"]["reconstructed_tokens"] == bench.stats["T"].completion_tokens
    monkeypatch.setenv("ACTIVATION_SYNC_ROOT", str(bench.tmp / "SYNC_RUNS" / "table"))
    h = rr.harness(True, intervention_layers=[1])
    ac = h.module_manager.get_ac_model("ac_private")
    trainer = ActivationContextTrainer(h, rr.config(bench.table["T"]))
    tokenizer = AutoTokenizer.from_pretrained(str(TINY_QWEN3))
    checked = 0
    for item in rr.mix_training(bench.study, bench.table["T"], bench.material, "T")[0]:
        seen, targets, rows, reader_input = rr.mix_tokens(item, tokenizer, tokenizer)
        example = trainer.build_example(ac, item)
        assert targets == example.num_completion and seen == len(ac.side.tokenizer.apply_chat_template(example.part_requests[0].messages, tokenize=True, return_dict=True)["input_ids"])
        if rows <= ac.config.max_view_rows:      # the tiny model caps its rows at 8, the bench's at 4,096
            assert rows == ac.part_view_rows(example.part_requests[0].messages, example.part_requests[0].compression_ratio) and reader_input == len(example.student_ids)
            checked += 1
    assert checked >= 12


# --------------------------------------------------------------------------------------------------- (e) the earlier arms
def test_earlier_arms_keep_their_configuration():
    """Every arm without `mix` builds the trainer configuration the bench built before the mixed arms existed (the mapping restated
    here), keeps the values it had, and the new keys are neutral for it."""
    rr = load_bench(Path("/nonexistent/AC_RECON"))
    arms = rr.arms()
    extra = json.loads((BENCH.parent / "arms.json").read_text())
    new = ("mix", "ratio_range", "span_reads", "single_span_share", "span_tokens", "span_cue_tokens", "long_tokens", "bm25_fraction", "updates", "examples_per_update", "landing",
           "landing_fraction", "beat_every", "panel_ratios", "panel_items", "continue_from", "continue_at", "encode_batch_tokens")
    assert set(new) == set(rr.DEFAULT_ARM) - {"ratio", "encoder_lr", "reader_lr", "penalty_weight", "penalty_fraction", "continue_share", "train_items", "epochs", "reporting_interval",
        "completion_samples", "completion_tokens", "budget_seconds", "final_train_subset", "micro_batch", "ckpt_min_tokens", "material", "seed", "warmup", "ckpt_every", "mixer_layers",
        "interventions", "last_layer_intervention", "intervention_layers", "deltas_off", "side_model", "reader_model", "reader_rank", "held_per_source", "recursive_keys",
        "pooling_stride", "pooling_window", "init_ac", "init_reader", "note"}
    earlier = [label for label, arm in arms.items() if not arm["mix"]]
    assert {"S", "A16", "A4"} <= set(earlier) and set(extra) <= set(earlier) and len(earlier) == 3 + len(extra)
    for label in earlier:
        arm = arms[label]
        assert all(arm[key] == rr.DEFAULT_ARM[key] for key in new) and arm["examples_per_update"] == 8 and arm["encode_batch_tokens"] is None, label
        if label in extra:
            assert all(arm[key] == value for key, value in extra[label].items()), label
        expected = ActivationContextTrainingConfig(
            loss_kind="sft", learning_rate_ac=arm["encoder_lr"], learning_rate_target_lora=arm["reader_lr"], examples_per_update=8, num_epochs=arm["epochs"],
            warmup_updates=arm["warmup"], checkpoint_every_updates=arm["ckpt_every"], checkpoint_every_epoch=True, completion_samples=arm["completion_samples"],
            completion_max_new_tokens=arm["completion_tokens"], reporting_interval=arm["reporting_interval"], seed=rr.SEED if arm["seed"] is None else arm["seed"],
            teacher_cache_top_k=rr.TOP_K, gradient_checkpointing_min_tokens=arm["ckpt_min_tokens"], micro_batch_examples=arm["micro_batch"],
            no_context_penalty_weight=arm["penalty_weight"], no_context_penalty_fraction=arm["penalty_fraction"], deltas_off_panel=arm["deltas_off"])
        assert asdict(rr.config(arm)) == asdict(expected), label
        resumed = rr.config(arm, num_epochs=1, resume_optimizer=True, resume_skip_steps=3)
        assert (resumed.shuffle_examples, resumed.lr_schedule, resumed.schedule_total_updates, resumed.resume_skip_steps) == (True, "cosine", None, 3), label
