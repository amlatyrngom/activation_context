"""
AIME 2024 / 2025 problem statements (`math-ai/aime24`, `math-ai/aime25`, split `test`, 30 rows each),
the benchmark statements the math loader decontaminates against. Not a dataset loader: a plain list.
"""
from __future__ import annotations

import re

from datasets import load_dataset

AIME_DATASETS = {2024: "math-ai/aime24", 2025: "math-ai/aime25"}
BOXED = re.compile(r"\\boxed\{([^}]*)\}")


def load_aime_problems(years: tuple[int, ...] = (2024, 2025)) -> list[dict]:
    """[{"year", "id", "problem", "answer"}] for the requested years (aime24 stores the answer as `solution` = \\boxed{...})."""
    problems = []
    for year in years:
        for row in load_dataset(AIME_DATASETS[year], split="test"):
            answer = row.get("answer")
            if answer is None and row.get("solution"):
                match = BOXED.search(str(row["solution"]))
                answer = match.group(1) if match else str(row["solution"])
            problems.append({"year": year, "id": str(row.get("id")), "problem": str(row["problem"]), "answer": None if answer is None else str(answer)})
    return problems


def aime_problem_statements(years: tuple[int, ...] = (2024, 2025)) -> list[str]:
    return [problem["problem"] for problem in load_aime_problems(years)]
