"""
Shared exact, token, numeric, judge, and submit-answer scoring helpers.
"""

from __future__ import annotations

import math
import re
import string


def normalize_answer(text) -> str:
    """
    Apply SQuAD-style lowercasing, punctuation/article removal, and spacing.
    """
    value = str(text or "").lower()
    value = "".join(" " if ch in string.punctuation else ch for ch in value)
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    return " ".join(value.split())


def exact_match(pred, golds) -> float:
    """
    Return one when a normalized prediction equals any accepted gold.
    """
    normalized = normalize_answer(pred)
    return float(any(normalized == normalize_answer(gold) for gold in golds))


def token_f1(pred, golds) -> float:
    """
    Return the maximum bag-of-tokens F1 across the accepted gold strings.
    """
    from collections import Counter

    predicted = normalize_answer(pred).split()
    best = 0.0
    for gold in golds:
        expected = normalize_answer(gold).split()
        if not predicted or not expected:
            score = float(predicted == expected)
        else:
            common = sum((Counter(predicted) & Counter(expected)).values())
            if common == 0:
                score = 0.0
            else:
                precision = common / len(predicted)
                recall = common / len(expected)
                score = 2 * precision * recall / (precision + recall)
        best = max(best, score)
    return best


_NUMBER = re.compile(r"[-+]?\$?\s*(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?\s*%?")


def _number(text) -> tuple[float, bool] | None:
    """
    Parse the first currency/comma/percent number from text.
    """
    match = _NUMBER.search(str(text or ""))
    if match is None:
        return None
    token = match.group(0).replace("$", "").replace(",", "").strip()
    percent = token.endswith("%")
    if percent:
        token = token[:-1].strip()
    try:
        return float(token), percent
    except ValueError:
        return None


def numeric_approx(pred, golds, rel_tol: float = 0.01) -> float:
    """
    Match parsed numbers within one-percent relative tolerance.

    When exactly one side carries a percent sign, that side is divided by
    100, so `12%` and `0.12` compare in the same unit.
    """
    parsed_pred = _number(pred)
    if parsed_pred is None:
        return 0.0
    pred_value, pred_percent = parsed_pred
    for gold in golds:
        parsed_gold = _number(gold)
        if parsed_gold is None:
            continue
        gold_value, gold_percent = parsed_gold
        if pred_percent != gold_percent:
            if pred_percent:
                pred_cmp, gold_cmp = pred_value / 100.0, gold_value
            else:
                pred_cmp, gold_cmp = pred_value, gold_value / 100.0
        else:
            pred_cmp, gold_cmp = pred_value, gold_value
        if math.isclose(pred_cmp, gold_cmp, rel_tol=rel_tol, abs_tol=1e-9):
            return 1.0
    return 0.0

_EXACT_NUMBER = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:/\d+)?")


def _last_exact_number(text):
    """
    The last number in the text as an exact Fraction: \boxed{} unwrapped, commas, dollar signs and
    spaces removed; "a/b" fractions and decimals accepted. None when the text has no number.
    """
    from fractions import Fraction

    value = str(text or "")
    boxed = re.findall(r"\\boxed\{([^{}]*)\}", value)
    if boxed:
        value = boxed[-1]
    value = value.replace("$", "")
    matches = _EXACT_NUMBER.findall(value)
    if not matches:
        return None
    token = matches[-1].replace(",", "").replace(" ", "")
    try:
        return Fraction(token)
    except (ValueError, ZeroDivisionError):
        return None


def numeric_exact(pred, golds) -> float:
    """
    Exact numeric check: the last number of the prediction equals a gold exactly. Returns 0.0/1.0.
    """
    predicted = _last_exact_number(pred)
    if predicted is None:
        return 0.0
    for gold in golds:
        expected = _last_exact_number(gold)
        if expected is not None and expected == predicted:
            return 1.0
    return 0.0

def _finqa_boolean(text) -> bool | None:
    """
    Map FinQA's yes/no execution answers and natural true/false equivalents.
    """
    normalized = normalize_answer(text)
    if normalized in {"yes", "true"}:
        return True
    if normalized in {"no", "false"}:
        return False
    return None


def finqa_match(pred, golds, rel_tol: float = 0.01) -> float:
    """
    Score a FinQA execution answer: yes/no exactly, otherwise numeric tolerance.

    The official `qa.exe_ans` field mixes raw numeric execution results with
    `yes`/`no` results from the `greater` operation. True/false are accepted as
    natural equivalents when an LLM submits the terminal answer.
    """
    boolean_golds = [_finqa_boolean(gold) for gold in golds]
    if any(value is not None for value in boolean_golds):
        predicted = _finqa_boolean(pred)
        return float(predicted is not None and predicted in boolean_golds)
    return numeric_approx(pred, golds, rel_tol=rel_tol)



def wrap_submit_answer(gold) -> dict:
    """
    Return a native assistant submit_answer tool-call message.
    """
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "type": "function",
            "function": {
                "name": "submit_answer",
                "arguments": {"answer": gold},
            },
        }],
    }



def unwrap_submit_answer(run_result) -> str:
    """
    Extract the comparable answer from an AgentRunResult or compatible value.
    """
    answer = getattr(run_result, "answer", run_result)
    if isinstance(answer, dict) and "answer" in answer:
        answer = answer["answer"]
    return "" if answer is None else str(answer)
