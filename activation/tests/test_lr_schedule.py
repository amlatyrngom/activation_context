"""The learning-rate factor of the three schedules (constant, cosine, cooldown) and the training config's validation of them."""
import math

import pytest

from activation.ac_model.ac_model_training import ActivationContextTrainingConfig
from activation.agent_training.agent_training_utils import lr_factor


def test_warmup_then_constant():
    assert lr_factor(0, 50) == pytest.approx(1 / 50)
    assert lr_factor(24, 50) == pytest.approx(0.5)
    assert lr_factor(49, 50) == pytest.approx(1.0)
    assert lr_factor(5_000, 50, total_updates=1_000, schedule="constant") == 1.0
    assert lr_factor(7, 0) == 1.0


def test_cosine_endpoints():
    assert lr_factor(50, 50, total_updates=1_050, schedule="cosine", final_fraction=0.1) == pytest.approx(1.0)
    assert lr_factor(550, 50, total_updates=1_050, schedule="cosine", final_fraction=0.1) == pytest.approx(0.55)
    assert lr_factor(1_050, 50, total_updates=1_050, schedule="cosine", final_fraction=0.1) == pytest.approx(0.1)
    assert lr_factor(9_999, 50, total_updates=1_050, schedule="cosine", final_fraction=0.1) == pytest.approx(0.1)   # holds the floor past the plan


def test_cooldown_linear_stays_flat_then_decays_to_the_floor():
    kw = dict(total_updates=2_400, schedule="cooldown", final_fraction=0.01, cooldown_updates=400)
    assert lr_factor(0, 50, **kw) == pytest.approx(1 / 50)                   # the warm-up is unchanged
    assert lr_factor(49, 50, **kw) == 1.0
    assert lr_factor(1_999, 50, **kw) == 1.0                                  # flat until total - cooldown
    assert lr_factor(2_000, 50, **kw) == 1.0
    assert lr_factor(2_200, 50, **kw) == pytest.approx(1.0 - 0.99 * 0.5)     # halfway down, linear
    assert lr_factor(2_400, 50, **kw) == pytest.approx(0.01)
    assert lr_factor(3_000, 50, **kw) == pytest.approx(0.01)                 # holds the floor past the plan
    for a, b in zip(range(2_000, 2_400), range(2_001, 2_401)):               # monotone, no jump at the start of the decay
        assert lr_factor(a, 50, **kw) >= lr_factor(b, 50, **kw)


def test_cooldown_cosine_shape_and_short_plans():
    kw = dict(total_updates=1_000, schedule="cooldown", final_fraction=0.0, cooldown_updates=200, cooldown_shape="cosine")
    assert lr_factor(800, 50, **kw) == pytest.approx(1.0)
    assert lr_factor(900, 50, **kw) == pytest.approx(0.5)
    assert lr_factor(1_000, 50, **kw) == pytest.approx(0.0)
    assert lr_factor(850, 50, **kw) > lr_factor(850, 50, **dict(kw, cooldown_shape="linear"))   # the cosine stays higher early
    # a cooldown longer than the plan after the warm-up starts right after the warm-up and still ends at the floor
    assert lr_factor(49, 50, total_updates=300, schedule="cooldown", final_fraction=0.1, cooldown_updates=1_000) == 1.0
    assert lr_factor(300, 50, total_updates=300, schedule="cooldown", final_fraction=0.1, cooldown_updates=1_000) == pytest.approx(0.1)
    # without a plan or a length the schedule degrades to constant rather than failing
    assert lr_factor(700, 50, schedule="cooldown", cooldown_updates=200) == 1.0
    assert lr_factor(700, 50, total_updates=1_000, schedule="cooldown") == 1.0


def test_config_validates_the_cooldown():
    ActivationContextTrainingConfig(lr_schedule="cooldown", cooldown_updates=300, schedule_total_updates=2_400)
    ActivationContextTrainingConfig(lr_schedule="cooldown", cooldown_updates=300, cooldown_shape="cosine")      # the plan may come at call time
    with pytest.raises(ValueError):
        ActivationContextTrainingConfig(lr_schedule="cooldown")                                                   # needs a length
    with pytest.raises(ValueError):
        ActivationContextTrainingConfig(lr_schedule="cooldown", cooldown_updates=0)
    with pytest.raises(ValueError):
        ActivationContextTrainingConfig(lr_schedule="cooldown", cooldown_updates=300, cooldown_shape="sqrt")
    with pytest.raises(ValueError):
        ActivationContextTrainingConfig(lr_schedule="cooldown", cooldown_updates=2_400, schedule_total_updates=2_400, warmup_updates=50)   # no room after the warm-up
    with pytest.raises(ValueError):
        ActivationContextTrainingConfig(lr_schedule="wsd")
    assert ActivationContextTrainingConfig(lr_schedule="cosine").cooldown_updates is None                         # existing recipes are untouched


def test_cooldown_edges_from_review():
    # an over-long cooldown decays from the end of the warm-up: 1.0 at update 50, then linear over the 250 remaining updates
    kw = dict(total_updates=300, schedule="cooldown", final_fraction=0.1, cooldown_updates=1_000)
    assert lr_factor(50, 50, **kw) == 1.0
    assert lr_factor(175, 50, **kw) == pytest.approx(0.55)
    # no warm-up: the decay may start at update 0
    assert lr_factor(0, 0, total_updates=100, schedule="cooldown", final_fraction=0.0, cooldown_updates=100) == 1.0
    assert lr_factor(50, 0, total_updates=100, schedule="cooldown", final_fraction=0.0, cooldown_updates=100) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        lr_factor(90, 0, total_updates=100, schedule="cooldown", final_fraction=0.0, cooldown_updates=100, cooldown_shape="sqrt")
    # the config's exact-fit boundary and its refusal of the fields under the other schedules
    ActivationContextTrainingConfig(lr_schedule="cooldown", cooldown_updates=2_350, schedule_total_updates=2_400, warmup_updates=50)
    with pytest.raises(ValueError):
        ActivationContextTrainingConfig(lr_schedule="cooldown", cooldown_updates=2_351, schedule_total_updates=2_400, warmup_updates=50)
    with pytest.raises(ValueError):
        ActivationContextTrainingConfig(lr_schedule="cooldown", cooldown_updates="300")
    with pytest.raises(ValueError):
        ActivationContextTrainingConfig(lr_schedule="cosine", cooldown_updates=300)
    with pytest.raises(ValueError):
        ActivationContextTrainingConfig(lr_schedule="constant", cooldown_shape="cosine")
