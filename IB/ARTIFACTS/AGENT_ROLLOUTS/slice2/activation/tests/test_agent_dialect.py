"""
The token-level dialect (agent_utils): a conversation built from segments (templated first prompt,
sampled assistant tokens verbatim, the dialect's wrapper for tool results and the nudge) must carry
exactly what the chat template puts between turns. Tokenizers only, no GPU.

Templates re-render history in ways the model never saw at sampling time: Qwen3 drops the empty
`<think>` block from every finished assistant turn, Qwen3.5 only once a user message follows. The
segments keep the generation-prompt form the model actually read, so the comparison is strict token
equality where the template preserves it (Qwen3.5 tool loops) and equality of the rendering with those
blocks normalised away otherwise.
"""
import pytest

from activation.agent.agent_utils import ModelDialect

CASES = [
    ("Qwen/Qwen3.5-4B", {"enable_thinking": False}, True),    # XML function calls; history keeps the think block through tool loops
    ("Qwen/Qwen3-0.6B", {"enable_thinking": False}, False),   # Hermes JSON calls; history drops it
]
EMPTY_THINK = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
TOOLS = [{"type": "function", "function": {"name": "python", "description": "Run Python.",
          "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}}}]


def _normalised(text: str) -> str:
    return text.replace(EMPTY_THINK, "<|im_start|>assistant\n")


@pytest.mark.parametrize("model_id,template_kwargs,history_keeps_think", CASES)
def test_segments_match_template(model_id, template_kwargs, history_keeps_think):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    dialect = ModelDialect.for_tokenizer(tokenizer)
    messages = [{"role": "system", "content": "You solve problems."}, {"role": "user", "content": "What is 6 * 7?"}]
    assistant_1 = "Let me compute.\n\n<tool_call>\n<function=python>\n<parameter=code>\nprint(6 * 7)\n</parameter>\n</function>\n</tool_call>"
    tool_1 = {"role": "tool", "content": "42\n"}          # the tool role: both templates render it as a user turn of <tool_response> blocks
    assistant_2 = "The answer is 42."
    nudge = {"role": "user", "content": dialect.nudge_message}
    assistant_3 = "Done."

    prefix = dialect.prompt_token_ids(tokenizer, messages, TOOLS, template_kwargs)
    end_of_turn = tokenizer.encode(tokenizer.eos_token, add_special_tokens=False)
    history = list(messages)

    def expected_ids(conversation):
        ids = tokenizer.apply_chat_template(conversation, tools=TOOLS, add_generation_prompt=True, tokenize=True, **template_kwargs)
        return list(ids["input_ids"] if hasattr(ids, "keys") else ids)

    def expected_text(conversation):
        return tokenizer.apply_chat_template(conversation, tools=TOOLS, add_generation_prompt=True, tokenize=False, **template_kwargs)

    # Tool loops: the segment prefix is the template's own rendering, turn after turn (token for token where
    # the template keeps the generation-prompt form of finished turns; otherwise up to the empty think blocks).
    for text in (assistant_1, assistant_1.replace("6 * 7", "42 * 1")):
        sampled = tokenizer.encode(text, add_special_tokens=False) + end_of_turn          # what the engine would return
        wrapper = dialect.continuation_token_ids(tokenizer, history, [tool_1], TOOLS, template_kwargs)
        prefix += sampled + dialect.join_continuation(sampled, wrapper)
        history += [{"role": "assistant", "content": text}, tool_1]
        if history_keeps_think:
            assert prefix == expected_ids(history), f"{model_id}: segment prefix differs from the template after {len(history)} messages"
        assert _normalised(tokenizer.decode(prefix)) == _normalised(expected_text(history)), f"{model_id}: wrapper content differs after {len(history)} messages"

    # The nudge (a user message): the wrapper is the template's tail. The full prefix may differ from a fresh
    # rendering only where the template rewrites *earlier* assistant turns (Qwen3.5 drops their empty think
    # blocks once a user message follows); the segments keep what the model actually read.
    sampled = tokenizer.encode(assistant_2, add_special_tokens=False) + end_of_turn
    wrapper = dialect.continuation_token_ids(tokenizer, history, [nudge], TOOLS, template_kwargs)
    prefix += sampled + dialect.join_continuation(sampled, wrapper)
    history += [{"role": "assistant", "content": assistant_2}, nudge]
    rendered = expected_text(history)
    assert rendered.endswith(tokenizer.decode(wrapper)), f"{model_id}: nudge wrapper is not the template's tail"
    assert _normalised(tokenizer.decode(prefix)) == _normalised(rendered)

    # A turn cut off by max_tokens has no end-of-turn: the wrapper supplies it.
    sampled = tokenizer.encode(assistant_3, add_special_tokens=False)
    wrapper = dialect.continuation_token_ids(tokenizer, history, [tool_1], TOOLS, template_kwargs)
    joined = dialect.join_continuation(sampled, wrapper)
    assert joined == wrapper and joined[0] == end_of_turn[0]
