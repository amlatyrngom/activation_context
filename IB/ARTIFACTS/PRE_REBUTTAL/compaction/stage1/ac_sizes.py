"""
Sizes of the activation-context (AC) rows that replace a prefix, stated as the benchmark states them: a fraction r of the
PREFIX (1/5, 1/10, 1/20 of about 20,000 prefix tokens = 4,000 / 2,000 / 1,000 rows).

The encoder does not count that way. Its row count is

    rows = clamp(ceil(side_tokens x compression_target), min_view_rows, max_view_rows)

where side_tokens is the encoder-side length of the whole part it reads (system prompt, tool definitions, task and prefix:
about 10 % longer than the prefix). So a size is first turned into a row count from the prefix alone and then into the
compression target that gives exactly that many rows for this part. Training (prepare_agent_continuation.py) and the
rollout (resume_program.py) both go through these two functions. Standard library only.
"""
import math


def rows_wanted_for(prefix_tokens: int, r: float) -> int:
    """Rows for a prefix of `prefix_tokens` reader tokens at size r (a fraction of the prefix), at least one."""
    return max(1, round(prefix_tokens * r))


def compression_target_for(rows_wanted: int, side_tokens: int) -> float:
    """
    The part's compression target for which ceil(side_tokens x target) == rows_wanted. `side_tokens` is the length of the
    part as ActivationContextModel.part_view_rows tokenizes it (encoder tokenizer, tool definitions included). The half row
    of margin keeps the ceiling exact under floating point. The encoder still clamps to its own row limits: a row count
    past max_view_rows comes out as max_view_rows.
    """
    if rows_wanted < 1 or side_tokens < 1:
        raise ValueError(f"rows_wanted ({rows_wanted}) and side_tokens ({side_tokens}) must be positive")
    target = (rows_wanted - 0.5) / side_tokens
    if math.ceil(side_tokens * target) != rows_wanted:
        raise ValueError(f"no exact target for {rows_wanted} rows over {side_tokens} side tokens")
    return target
