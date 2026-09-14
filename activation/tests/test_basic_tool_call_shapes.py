"""
Tool-call shapes the oracle produces under parallel delegation: nested calls with flattened arguments, a null `task`
next to a flattened one, a subagent call without a brief, and a top-level prompt of None (the task travels in
messages_input). None of them may crash an agent; the first two must reach the tool as the model meant them.
"""
from activation.agent.agent import Agent
from activation.agent.agent_tools import ParallelCallTool, SubagentTool


def test_nested_call_arguments_are_merged_from_flattened_keys():
    good = {"name": "subagent", "arguments": {"task": "brief"}}
    flat = {"name": "subagent", "task": "brief"}
    both = {"name": "subagent", "arguments": {"task": None}, "task": "brief"}
    other = {"tool": "shell", "args": {"script": "ls"}, "timeout": 5}
    assert ParallelCallTool._call_arguments(good) == {"task": "brief"}
    assert ParallelCallTool._call_arguments(flat) == {"task": "brief"}
    assert ParallelCallTool._call_arguments(both) == {"task": "brief"}
    assert ParallelCallTool._call_arguments(other) == {"script": "ls", "timeout": 5}
    assert ParallelCallTool._call_arguments({"name": "shell"}) == {}
    assert ParallelCallTool._call_arguments("shell") == {}


def test_subagent_without_a_brief_is_a_tool_error_not_a_crash():
    class Dummy:
        pass
    for task in (None, "", "   ", 3):
        result = SubagentTool.execute(Dummy(), task=task)          # the guard runs before any agent state is touched
        assert result.is_error and "brief" in result.output


def test_offloaded_prompt_passes_none_and_short_prompts_through():
    class Dummy:
        step_mode = False
        parent_agent = None
    assert Agent._offloaded_prompt(Dummy(), None) is None
    assert Agent._offloaded_prompt(Dummy(), "short") == "short"
