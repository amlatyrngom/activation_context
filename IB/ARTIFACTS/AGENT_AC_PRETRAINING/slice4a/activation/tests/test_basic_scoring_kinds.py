"""Scoring kinds added for the campaign: math-verify grading and unscored tasks (score() is None, reporters blank)."""
import pytest

from activation.dataset import ANSWER_RULES, DatasetTask, DatasetTaskMetricsKind, bare_prompt
from activation.dataset.scoring import math_verify_match


@pytest.mark.parametrize("pred, gold, expected", [
    ("42", "42", 1.0), ("42.0", "42", 1.0), ("\\frac{1}{2}", "1/2", 1.0), ("0.5", "\\frac{1}{2}", 1.0),
    ("$\\sqrt{2}$", "\\sqrt{2}", 1.0), ("x^2 + 1", "x^{2}+1", 1.0), ("Yes", "yes", 1.0), ("No", "Yes", 0.0),
    ("43", "42", 0.0), ("", "42", 0.0), ("[0, 1]", "[0,1]", 1.0),
])
def test_math_verify_match(pred, gold, expected):
    assert math_verify_match(pred, [gold]) == expected


def test_unscored_task_scores_none_and_math_task_scores():
    unscored = DatasetTask(task_id="u", dataset_id="loong___x", task_datum={}, reference_metrics_kind=DatasetTaskMetricsKind.UNSCORED,
                           gold_answer="whatever", agent_prompt=bare_prompt("Q", ANSWER_RULES["free"]))
    assert unscored.score({"answer": "anything"}) is None and not unscored.is_scored
    math = DatasetTask(task_id="m", dataset_id="deepmath___x", task_datum={}, reference_metrics_kind=DatasetTaskMetricsKind.MATH_VERIFY,
                       gold_answer="\\frac{3}{4}", agent_prompt=bare_prompt("Q", ANSWER_RULES["math"]))
    assert math.score({"answer": "3/4"}) == 1.0 and math.score({"answer": "0.7"}) == 0.0 and math.is_scored
