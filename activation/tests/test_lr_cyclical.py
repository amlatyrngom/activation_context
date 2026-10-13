"""The cyclical learning-rate schedule (lr_schedule="cyclical": warm-up, constant, decay, repeated every cycle_updates) and its optional
optimizer reset at cycle starts, on CPU: the factor at every update against hand-written tables, the older schedules unchanged to the
bit, the config's validation, and tiny training runs (resets exactly at cycle starts; a run killed mid-cycle or on a boundary and resumed).

  PYTHONPATH=. .venv/bin/python -m pytest activation/tests/test_lr_cyclical.py -q -p no:cacheprovider
"""
import math
import os
import sys
from types import SimpleNamespace

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.modules.setdefault("fla", None)          # the kernels are GPU-only; the models fall back to torch
import pytest
import torch
from torch.optim.optimizer import register_optimizer_step_pre_hook

from activation.ac_model import ActivationContextStudyGenerator, ActivationContextTrainer, ActivationContextTrainingConfig
from activation.ac_model.ac_model_utils import AdamMomentPlacement
from activation.agent_training.agent_training_utils import cycle_span, lr_factor, reset_optimizer_state, set_learning_rates
from activation.tests.test_interventions import PASSAGES, TINY_QWEN3, build, config

PAPER = dict(total_updates=2_446, schedule="cyclical", final_fraction=0.01, cooldown_updates=150, cycle_updates=489)   # 5 cycles: 4 of 489 updates, the last of 490


def _lr_factor_before(updates_done, warmup_updates, *, total_updates=None, schedule="constant", final_fraction=0.1, anchor_update=None, anchor_factor=None,
                      cooldown_updates=None, cooldown_shape="linear"):
    """lr_factor as it was before the cyclical schedule (copied on 2026-10-10): the reference for the older schedules."""
    factor = min(1.0, (updates_done + 1) / warmup_updates) if warmup_updates > 0 else 1.0
    if schedule == "cosine" and total_updates and anchor_update is not None and updates_done >= anchor_update:
        progress = min(1.0, (updates_done - anchor_update) / max(1, total_updates - anchor_update))
        factor *= final_fraction + (anchor_factor - final_fraction) * 0.5 * (1.0 + math.cos(math.pi * progress))
    elif schedule == "cosine" and total_updates:
        span = max(1, total_updates - warmup_updates)
        progress = min(1.0, max(0.0, (updates_done - warmup_updates) / span))
        factor *= final_fraction + (1.0 - final_fraction) * 0.5 * (1.0 + math.cos(math.pi * progress))
    elif schedule == "cooldown" and total_updates and cooldown_updates:
        start = max(warmup_updates, total_updates - cooldown_updates)
        if updates_done >= start:
            progress = min(1.0, (updates_done - start) / max(1, total_updates - start))
            if cooldown_shape == "cosine":
                factor *= final_fraction + (1.0 - final_fraction) * 0.5 * (1.0 + math.cos(math.pi * progress))
            elif cooldown_shape == "linear":
                factor *= 1.0 - (1.0 - final_fraction) * progress
            else:
                raise ValueError(f"unknown cooldown_shape {cooldown_shape!r}: 'linear' or 'cosine'")
    return factor


# --------------------------------------------------------------------------------------------------- the factor
def test_small_cycles_match_the_hand_written_factors():
    factors = lambda total, warmup, **kw: [lr_factor(update, warmup, total_updates=total, schedule="cyclical", **kw) for update in range(total)]
    # Cycles of 10 with a 2-update warm-up from the floor 0.1 and a 3-update decay back to it: the floor itself is never applied
    # (it is the value one update before a cycle and one update after it), the warm-up's last update and the decay's first are at 1.
    cycle = [0.55, 1, 1, 1, 1, 1, 1, 1, 0.7, 0.4]
    assert factors(20, 2, final_fraction=0.1, cooldown_updates=3, cycle_updates=10) == pytest.approx(cycle + cycle)
    # 23 updates: two cycles, and the second takes the 3 left over (its constant stretch is longer; the decay still ends the run).
    assert factors(23, 2, final_fraction=0.1, cooldown_updates=3, cycle_updates=10) == pytest.approx(cycle + [0.55] + [1] * 10 + [0.7, 0.4])
    assert factors(29, 2, final_fraction=0.1, cooldown_updates=3, cycle_updates=10) == pytest.approx(cycle + [0.55] + [1] * 16 + [0.7, 0.4]), "never a short third cycle"
    # No warm-up, floor 0: constant, then the decay.
    assert factors(8, 0, final_fraction=0.0, cooldown_updates=2, cycle_updates=4) == pytest.approx([1, 1, 1, 0.5] * 2)
    # Warm-up and decay fill the cycle: no constant stretch.
    assert factors(8, 2, final_fraction=0.0, cooldown_updates=2, cycle_updates=4) == pytest.approx([0.5, 1, 1, 0.5] * 2)
    # The cosine shape bends the decay only.
    cosine = factors(10, 2, final_fraction=0.1, cooldown_updates=4, cycle_updates=10, cooldown_shape="cosine")
    assert cosine[:6] == pytest.approx([0.55, 1, 1, 1, 1, 1]) and cosine[6:] == pytest.approx([0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * k / 4)) for k in range(4)])
    with pytest.raises(ValueError, match="cooldown_shape"):
        lr_factor(9, 2, total_updates=10, schedule="cyclical", cooldown_updates=4, cycle_updates=10, cooldown_shape="sqrt")
    # Past the plan the last cycle holds the floor, and without a cycle length the schedule is the plain warm-up (as a cooldown without its length).
    assert [lr_factor(update, 2, total_updates=20, schedule="cyclical", final_fraction=0.1, cooldown_updates=3, cycle_updates=10) for update in (20, 21, 500)] == pytest.approx([0.1] * 3)
    assert lr_factor(700, 50, total_updates=1_000, schedule="cyclical", cooldown_updates=200) == 1.0


def test_paper_shape_cycles():
    """2,446 updates in 5 cycles (489, the last 490), 50 warm-up and 150 decay updates, floor 1 %. `update` counts the updates done."""
    factor = lambda update: lr_factor(update, 50, **PAPER)
    assert [cycle_span(update, 489, 2_446) for update in (0, 488, 489, 1_955, 1_956, 2_445, 2_446, 9_999)] == [
        (0, 489), (0, 489), (489, 489), (1_467, 489), (1_956, 490), (1_956, 490), (1_956, 490), (1_956, 490)]
    assert cycle_span(1_956, 489) == (1_956, 489) and cycle_span(2_445, 489) == (2_445, 489), "without a plan every cycle has the same length"
    for start in (0, 489, 978, 1_467, 1_956):                                           # the first update of each cycle, the warm-up's last, the constant stretch
        assert factor(start) == pytest.approx(0.01 + 0.99 / 50) and factor(start + 1) == pytest.approx(0.01 + 0.99 * 2 / 50)
        assert factor(start + 48) == pytest.approx(0.9802) and factor(start + 49) == 1.0 and factor(start + 50) == 1.0 and factor(start + 200) == 1.0
    for end in (489, 978, 1_467, 1_956, 2_446):                                         # the decay: the 150 updates before the next cycle (or the run's end)
        assert factor(end - 151) == 1.0 and factor(end - 150) == 1.0 and factor(end - 149) == pytest.approx(1 - 0.99 / 150)
        assert factor(end - 75) == pytest.approx(0.505) and factor(end - 2) == pytest.approx(0.0232) and factor(end - 1) == pytest.approx(0.01 + 0.99 / 150)
    assert factor(1_956 + 339) == 1.0 and factor(1_956 + 340) == 1.0 and factor(1_956 + 341) == pytest.approx(1 - 0.99 / 150), "the last cycle's extra update is a constant one"
    assert factor(2_446) == pytest.approx(0.01) and factor(5_000) == pytest.approx(0.01)
    # Every update against the rule written out once more, independently of cycle_span.
    for update in range(2_446):
        start = min(update // 489, 4) * 489
        position, length = update - start, 490 if start == 1_956 else 489
        expected = 0.01 + 0.99 * (position + 1) / 50 if position < 50 else 1.0 if position < length - 150 else 1 - 0.99 * (position - (length - 150)) / 150
        assert factor(update) == pytest.approx(expected, abs=1e-12), update
    assert min(map(factor, range(2_446))) == pytest.approx(0.01 + 0.99 / 150) and max(map(factor, range(2_446))) == 1.0


def test_a_cycle_decays_exactly_like_a_cooldown_run_of_its_length():
    """After its warm-up a cycle is, to the bit, the 'cooldown' recipe of a run of the cycle's length (what a restart by hand trains with);
    the warm-up differs by starting from the floor instead of 0: at most final_fraction above the by-hand warm-up."""
    by_hand = lambda position, length: lr_factor(position, 50, total_updates=length, schedule="cooldown", final_fraction=0.01, cooldown_updates=150)
    for start, length in ((0, 489), (978, 489), (1_956, 490)):
        for position in range(length):
            cyclical = lr_factor(start + position, 50, **PAPER)
            if position >= 50:
                assert cyclical == by_hand(position, length), (start, position)
            else:
                assert 0 <= cyclical - by_hand(position, length) <= 0.01, (start, position)


def test_older_schedules_are_bit_identical():
    recipes = [
        (2_446, 50, dict(schedule="constant")),
        (300, 0, dict(schedule="constant")),
        (2_446, 50, dict(schedule="cooldown", final_fraction=0.01, cooldown_updates=500, cooldown_shape="linear")),      # the packet builder's default landing
        (2_446, 50, dict(schedule="cooldown", final_fraction=0.01, cooldown_updates=150, cooldown_shape="linear")),
        (5_000, 50, dict(schedule="cooldown", final_fraction=0.01, cooldown_updates=500, cooldown_shape="linear")),
        (1_000, 50, dict(schedule="cooldown", final_fraction=0.0, cooldown_updates=200, cooldown_shape="cosine")),
        (300, 50, dict(schedule="cooldown", final_fraction=0.1, cooldown_updates=1_000)),                                # a decay longer than the run
        (1_050, 50, dict(schedule="cosine", final_fraction=0.1)),
        (9_500, 50, dict(schedule="cosine", final_fraction=0.1, anchor_update=5_000, anchor_factor=0.4)),
    ]
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.AdamW([{"params": [parameter], "lr": 2e-5, "base_lr": 2e-5}])
    for total, warmup, kw in recipes:
        for update in range(total + 60):                                                                                # past the plan too
            before = _lr_factor_before(update, warmup, total_updates=total, **kw)
            assert lr_factor(update, warmup, total_updates=total, **kw) == before, (kw, update)                         # ==: the same float, not a close one
            assert lr_factor(update, warmup, total_updates=total, cycle_updates=None, **kw) == before, (kw, update)     # as the trainer calls it
        for update in (0, warmup, total // 2, total - 1, total):
            assert set_learning_rates(optimizer, update, warmup, total_updates=total, cycle_updates=None, **kw) == _lr_factor_before(update, warmup, total_updates=total, **kw)
            assert optimizer.param_groups[0]["lr"] == 2e-5 * _lr_factor_before(update, warmup, total_updates=total, **kw)
    assert all(lr_factor(update, 50) == _lr_factor_before(update, 50) for update in range(120)), "the agent trainer's call: warm-up only"
    # A stray cycle length does nothing to another schedule.
    assert lr_factor(1_300, 50, total_updates=2_446, schedule="cooldown", final_fraction=0.01, cooldown_updates=500, cycle_updates=489) == 1.0


def test_config_validation():
    cyclical = lambda **kw: ActivationContextTrainingConfig(**{**dict(lr_schedule="cyclical", cooldown_updates=150, cycle_updates=489, lr_final_fraction=0.01, schedule_total_updates=2_446), **kw})
    assert (cyclical().cycle_updates, cyclical().cycle_reset_optimizer, cyclical(cycle_reset_optimizer=True).cycle_reset_optimizer) == (489, False, True)
    cyclical(schedule_total_updates=None)                                               # the plan may come at call time
    cyclical(cycle_updates=200)                                                         # the warm-up and the decay exactly fill a cycle
    cyclical(cycle_updates=2_446)                                                       # one cycle
    cyclical(cooldown_shape="cosine")
    for wrong in (dict(cycle_updates=None), dict(cycle_updates=199), dict(cycle_updates=2_447), dict(cycle_updates=489.0), dict(cycle_updates=True), dict(cooldown_updates=None),
                  dict(cooldown_updates=0), dict(cooldown_shape="sqrt"), dict(cycle_reset_optimizer=1), dict(warmup_updates=400)):
        with pytest.raises(ValueError, match="cyclical"):
            cyclical(**wrong)
    # The cycle fields belong to the cyclical schedule; every other recipe is as it was.
    for other in (dict(lr_schedule="constant"), dict(lr_schedule="cosine"), dict(lr_schedule="cooldown", cooldown_updates=300)):
        for stray in (dict(cycle_updates=489), dict(cycle_reset_optimizer=True)):
            with pytest.raises(ValueError, match="belong to lr_schedule 'cyclical'"):
                ActivationContextTrainingConfig(**other, **stray)
        made = ActivationContextTrainingConfig(**other)
        assert made.cycle_updates is None and made.cycle_reset_optimizer is False
    assert ActivationContextTrainingConfig().lr_schedule == "cosine" and ActivationContextTrainingConfig().cycle_updates is None


# --------------------------------------------------------------------------------------------------- the reset
def test_reset_leaves_what_a_new_optimizer_would_have():
    torch.manual_seed(0)
    weights = [torch.randn(4, 3), torch.randn(5)]
    gradients = [[torch.randn_like(weight) for weight in weights] for _ in range(7)]
    make = lambda values: [torch.nn.Parameter(value.clone()) for value in values]
    adam = lambda parameters: torch.optim.AdamW([{"params": parameters[:1], "lr": 1e-2, "base_lr": 1e-2}, {"params": parameters[1:], "lr": 3e-3, "base_lr": 3e-3}],
                                                betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01)
    def steps(optimizer, parameters, batches):
        for batch in batches:
            for parameter, gradient in zip(parameters, batch):
                parameter.grad = gradient.clone()
            optimizer.step()
    kept = make(weights); optimizer = adam(kept)
    placement = AdamMomentPlacement(optimizer)                                           # its step hooks belong to the optimizer object: the reset keeps them
    steps(optimizer, kept, gradients[:4])
    assert all(int(state["step"]) == 4 for state in optimizer.state.values())
    reset_optimizer_state(optimizer)
    assert not optimizer.state_dict()["state"] and [group["lr"] for group in optimizer.param_groups] == [1e-2, 3e-3] and len(placement.handles) == 2
    restarted = make([parameter.detach() for parameter in kept]); fresh = adam(restarted)   # the restart by hand: a new optimizer on the pair as it stands
    steps(optimizer, kept, gradients[4:]); steps(fresh, restarted, gradients[4:])
    assert all(torch.equal(a, b) for a, b in zip(kept, restarted)), "the same weights to the bit"
    assert all(int(state["step"]) == 3 for state in optimizer.state.values())
    assert all(torch.equal(optimizer.state[a][key], fresh.state[b][key]) for a, b in zip(kept, restarted) for key in ("step", "exp_avg", "exp_avg_sq"))
    placement.close()


# --------------------------------------------------------------------------------------------------- tiny training runs
WORDS = " ".join(passage["text"] for passage in PASSAGES).split()
MATERIAL = [{"text": " ".join(WORDS[(5 * index + offset) % len(WORDS)] for offset in range(11 + index)), "doc_id": f"c{index}", "source": "test"} for index in range(8)]
CYCLES = dict(lr_schedule="cyclical", warmup_updates=1, cooldown_updates=2, cycle_updates=4, lr_final_fraction=0.1, schedule_total_updates=8)   # 8 updates: two cycles of 4
RATES = [1e-2 * factor for factor in (0.1 + 0.9 / 1, 1, 1, 0.55)] * 2               # per cycle: a 1-update warm-up (already at 1), constant, the 2-update decay (1, then 0.55)


class Killed(Exception):
    """The process dies right after a mid-epoch checkpoint."""


def run(root, *, stop=None, resume=None, **overrides):
    """One epoch of 8 updates (one example each) on the tiny reader, a checkpoint after every update. `stop`: die after that update's
    checkpoint; `resume`: (root of the killed run, its last update): a new process on that pair with its optimizer.pt, as a requeue."""
    checkpoints = {}
    if resume is not None:
        leaf = f"epoch_001_step_{resume[1]:05d}"
        checkpoints = dict(ac_checkpoint=str(resume[0] / "AC_MODELS" / "ac_private" / leaf), reader_checkpoint=str(resume[0] / "LORAS" / "reader_lora" / leaf))
        overrides = dict(overrides, resume_optimizer=True, resume_skip_steps=resume[1])
    h, ac = build(root, TINY_QWEN3, **checkpoints)
    items = ActivationContextStudyGenerator(h, ac.name).generate_reconstruction_samples(MATERIAL, ratio=0.25)
    assert len(items) == 8
    trainer = ActivationContextTrainer(h, config(examples_per_update=1, completion_samples=0, checkpoint_every_epoch=True, checkpoint_every_updates=1, shuffle_examples=False, **overrides))
    if stop is not None:
        save = trainer._save_partial
        def save_then_die(ac_model, epoch_number, step, steps):
            save(ac_model, epoch_number, step, steps)
            if step == stop:
                raise Killed
        trainer._save_partial = save_then_die
    adam_steps, rates = [], []          # before each update: the largest Adam step count among the parameters (0: no state, a fresh optimizer) and the encoder's rate
    def before_step(optimizer, args, kwargs):
        adam_steps.append(max((int(state["step"]) for state in optimizer.state.values() if state), default=0))
        rates.append(optimizer.param_groups[0]["lr"])
    handle = register_optimizer_step_pre_hook(before_step)
    try:
        if stop is None:
            stats = trainer.train(ac.name, items)
        else:
            with pytest.raises(Killed):
                trainer.train(ac.name, items)
            stats = None
    finally:
        handle.remove()
    parameters = ac.trainable_parameters() + h.module_manager.lora_parameters(ac.config.target_model_lora_name)
    return SimpleNamespace(stats=stats, adam_steps=adam_steps, rates=rates, updates=trainer.updates_done.get(ac.name, 0), root=root,
                           weights=[parameter.detach().clone() for parameter in parameters])


@pytest.fixture(scope="module")
def whole(tmp_path_factory):
    """The uninterrupted cyclical run with the reset."""
    return run(tmp_path_factory.mktemp("whole"), cycle_reset_optimizer=True, **CYCLES)


def test_reset_happens_at_cycle_starts_only(whole, tmp_path):
    assert whole.updates == 8 and whole.stats.global_steps == list(range(1, 9))
    assert whole.rates == pytest.approx(RATES) and [pair[0] for pair in whole.stats.learning_rates] == pytest.approx(RATES)
    assert [pair[1] / pair[0] for pair in whole.stats.learning_rates] == pytest.approx([1.0] * 8), "the reader's rate follows the same factor"
    assert whole.stats.optimizer_resets == [4] and whole.adam_steps == [0, 1, 2, 3, 0, 1, 2, 3], "fresh moments at update 5, the second cycle's first; never at the run's first update"
    # The same cycles without the option: the same rates, no reset; and so for a cooldown recipe (it cannot even carry the option).
    kept = run(tmp_path / "kept", **CYCLES)
    assert kept.rates == pytest.approx(RATES) and kept.stats.optimizer_resets == [] and kept.adam_steps == list(range(8))
    assert any(not torch.equal(a, b) for a, b in zip(kept.weights, whole.weights)), "the reset changed the second cycle's updates"
    cooldown = run(tmp_path / "cooldown", lr_schedule="cooldown", warmup_updates=1, cooldown_updates=2, lr_final_fraction=0.1, schedule_total_updates=8)
    assert cooldown.stats.optimizer_resets == [] and cooldown.adam_steps == list(range(8))
    assert cooldown.rates == pytest.approx([1e-2] * 7 + [1e-2 * 0.55])


def test_resume_mid_cycle_continues_without_a_reset(whole, tmp_path):
    killed = run(tmp_path / "killed", stop=6, cycle_reset_optimizer=True, **CYCLES)
    assert killed.updates == 6 and killed.adam_steps == [0, 1, 2, 3, 0, 1] and killed.rates == pytest.approx(RATES[:6])
    resumed = run(tmp_path / "resumed", resume=(killed.root, 6), cycle_reset_optimizer=True, **CYCLES)
    assert resumed.stats.resumed_from.endswith("epoch_001_step_00006/optimizer.pt") and resumed.stats.global_steps == [7, 8]
    assert resumed.rates == whole.rates[6:], "the same factor as the uninterrupted run, to the bit"
    assert resumed.stats.optimizer_resets == [] and resumed.adam_steps == [2, 3], "the moments of the two updates since the cycle's start came back; no second reset"
    assert all(torch.equal(a, b) for a, b in zip(resumed.weights, whole.weights)), "and the run ends where the uninterrupted one does"


def test_resume_on_a_cycle_boundary_resets_once(whole, tmp_path):
    killed = run(tmp_path / "killed", stop=4, cycle_reset_optimizer=True, **CYCLES)
    saved = torch.load(killed.root / "AC_MODELS" / "ac_private" / "epoch_001_step_00004" / "optimizer.pt", map_location="cpu", weights_only=False)
    assert saved["updates_done"] == 4 and saved["optimizer"]["state"] and all(int(state["step"]) == 4 for state in saved["optimizer"]["state"].values()), \
        "the pair saved on the boundary keeps the moments of the cycle that trained it: the reset belongs to the next update"
    assert killed.adam_steps == [0, 1, 2, 3]
    resumed = run(tmp_path / "resumed", resume=(killed.root, 4), cycle_reset_optimizer=True, **CYCLES)
    assert resumed.stats.global_steps == [5, 6, 7, 8] and resumed.rates == whole.rates[4:]
    assert resumed.stats.optimizer_resets == [4] and resumed.adam_steps == [0, 1, 2, 3], "the cycle start that had not been done yet is done once, by the resumed run"
    assert all(torch.equal(a, b) for a, b in zip(resumed.weights, whole.weights))
    # Without the option the same resume keeps the restored moments.
    carried = run(tmp_path / "carried", resume=(killed.root, 4), **CYCLES)
    assert carried.stats.optimizer_resets == [] and carried.adam_steps == [4, 5, 6, 7] and carried.rates == whole.rates[4:]
