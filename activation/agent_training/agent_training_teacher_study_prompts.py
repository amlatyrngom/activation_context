"""
Prompts of the teacher study, written out in full so they are easy to edit.

`META_AGENT_PROMPT_TEMPLATES` maps (task kind, teacher kind) to the templates the study samples from, uniformly. Each
template is one plain text with one slot, `{task}`, which receives the loader's bare task prompt (task text plus the
scorer's submission rule); what the tools do is said in the tool descriptions, not here. Add a wording by appending
another triple-quoted string to the list of its kind; a template is identified in reports by its position
(`<task_kind>/<teacher_kind>/<index>`).
"""
from enum import StrEnum, auto

from activation.dataset import DatasetTaskKind


class AgentTeacherKind(StrEnum):
    BASE = auto()                      # the task as is; models already behave this way, so it gets the lowest weight
    SEQUENTIAL_MULTI_AGENT = auto()    # divide-and-delegate in sequence, or solve then review
    PARALLEL_MULTI_AGENT = auto()      # full solves or sub-solves in parallel, then synthesize


SYSTEM_PROMPT = """
You are a capable, careful agent solving a task in a Linux sandbox. Use the tools to look things up and compute
rather than guessing, delegate self-contained sub-tasks to subagents when that saves time, and give your final
answer exactly once through submit_answer.
""".strip()

# ============================================================================================== base: the task as is
BASE = [
    """
{task}
""".strip(),
]


# ============================================================================================== semantic search

# These two are not useful as far as I know.
# Have these be fully parallel solving and synthesizing the best answer. One alternative.
# Have stronger wording that its role is to delegate to 3 searchers and synthesize their outputs.
SEARCH_PARALLEL = [
    """
{task}

Your role is to coordinate, not to search yourself: delegate the whole question to three subagents in one
parallel_tool_call, each with the same self-contained brief (the full question, the required answer format, and a
request for the answer with the passage ids that support it). When they return, compare the three answers, settle a
disagreement by reading the passages they cite with semantic_search, and submit the best-supported answer.
""".strip(),
    """
{task}

Your role is to delegate to three searchers and synthesize their outputs. Split the question into three search
briefs (different sub-questions, or different angles on the same one) and run them as three subagent calls in one
parallel_tool_call; each brief is self-contained and asks for the relevant passages with their ids and a short
conclusion. Combine what they found into the answer; search yourself only to settle a conflict between them.
""".strip(),
]

SEARCH_SEQUENTIAL = [
    """
{task}

Suggested approach: search for the first entity, read what the passages say, search again for the entity that the
answer depends on, and cross-check the final fact in a second passage before answering. Use a subagent to check a
step you are unsure about.
""".strip(),
    """
{task}

Suggested approach: search, read the top passages, refine the query with the vocabulary they use, and iterate until
the passages actually answer the question; then delegate a subagent to review your answer against the passages
before submitting.
""".strip(),
]


# ============================================================================================== math
# Have stronger wording that its role is to delegate to 3 solvers and synthesize their outputs.
MATH_PARALLEL = [
    """
{task}

Your role is to delegate to three solvers and synthesize their outputs, not to solve the problem yourself first.
Send the full problem to three subagents in one parallel_tool_call, each brief self-contained (the complete problem
statement, the required answer format, and a request for the final value with the method used). Compare the three
results: if they agree, submit; if not, settle the disputed step with your own python computation, then submit.
""".strip(),
]

MATH_SEQUENTIAL = [
    """
{task}

Suggested approach: write a short plan, solve it with python rather than by hand, then delegate a validation brief
to a subagent (the problem, your answer, and a request to check it independently). Revise if the validation
disagrees.
""".strip(),
]


# ============================================================================================== code search
# Have stronger wording that its role is to delegate to 3 searchers and synthesize their outputs.
# One alternative says that the domains can be different or same to get a diversity of responses.
CODE_PARALLEL = [
    """
{task}

Your role is to delegate to three searchers and synthesize their outputs. Send the issue to three subagents in one
parallel_tool_call; each brief is self-contained (the issue text, the repository path, the required answer format)
and asks for the file most likely responsible together with the evidence (symbols, lines, call path). Compare their
findings, confirm the winner by reading the cited code with ripgrep, and submit.
""".strip(),
    """
{task}

Your role is to delegate to three searchers and synthesize their outputs. Skim the repository layout with ripgrep,
then write three self-contained briefs and run them in one parallel_tool_call. The briefs may cover different areas
of the repository (each searcher its own directories) or the same area (independent looks, for a diversity of
answers); each asks for the responsible file with evidence. Pick the answer the evidence supports best and submit.
""".strip(),
]

CODE_SEQUENTIAL = [
    """
{task}

Suggested approach: reproduce or trace the symptom from the issue text, follow the call path through the code with
ripgrep, and confirm the responsible file by reading it. Delegate a subagent to double-check your conclusion
against the issue before answering.
""".strip(),
]


# ============================================================================================== file search (long documents)
FILE_PARALLEL = [
    """
{task}

Your role is to delegate to three readers and synthesize their outputs. Count the document's lines, split it into
three line ranges, and send one subagent per range in one parallel_tool_call (each brief names the file, the line
range and the question, and asks for the relevant facts with quotes). Combine their findings, verify the decisive
passage yourself, and submit.
""".strip(),
]

FILE_SEQUENTIAL = [
    """
{task}

Suggested approach: read the document in order with the shell (sed -n or head/tail over line ranges), keep notes of
who, what and when as you go, and answer from your notes. Compact when asked and keep the notes in the summary.
""".strip(),
    """
{task}

Suggested approach: search the document for the entities in the question (grep and semantic_search), read the
surrounding passages, decide, then delegate a subagent to validate the decision against the document before
answering.
""".strip(),
]


# ============================================================================================== general (no kind)
GENERAL_PARALLEL = [
    """
{task}

Your role is to delegate and synthesize. Split the work into three self-contained parts, or three independent
attempts at the whole task if it does not split, and run them as subagent calls in one parallel_tool_call. Integrate
their results, verify the final answer yourself, and submit.
""".strip(),
]

GENERAL_SEQUENTIAL = [
    """
{task}

Suggested approach: plan, solve step by step with the tools, then delegate a validation brief to a subagent and
revise if it disagrees.
""".strip(),
]


META_AGENT_PROMPT_TEMPLATES: dict[tuple[DatasetTaskKind, AgentTeacherKind], list[str]] = {
    (DatasetTaskKind.SEMANTIC_SEARCH, AgentTeacherKind.BASE): BASE,
    (DatasetTaskKind.SEMANTIC_SEARCH, AgentTeacherKind.PARALLEL_MULTI_AGENT): SEARCH_PARALLEL,
    (DatasetTaskKind.SEMANTIC_SEARCH, AgentTeacherKind.SEQUENTIAL_MULTI_AGENT): SEARCH_SEQUENTIAL,
    (DatasetTaskKind.MATH, AgentTeacherKind.BASE): BASE,
    (DatasetTaskKind.MATH, AgentTeacherKind.PARALLEL_MULTI_AGENT): MATH_PARALLEL,
    (DatasetTaskKind.MATH, AgentTeacherKind.SEQUENTIAL_MULTI_AGENT): MATH_SEQUENTIAL,
    (DatasetTaskKind.CODE_SEARCH, AgentTeacherKind.BASE): BASE,
    (DatasetTaskKind.CODE_SEARCH, AgentTeacherKind.PARALLEL_MULTI_AGENT): CODE_PARALLEL,
    (DatasetTaskKind.CODE_SEARCH, AgentTeacherKind.SEQUENTIAL_MULTI_AGENT): CODE_SEQUENTIAL,
    (DatasetTaskKind.FILE_SEARCH, AgentTeacherKind.BASE): BASE,
    (DatasetTaskKind.FILE_SEARCH, AgentTeacherKind.PARALLEL_MULTI_AGENT): FILE_PARALLEL,
    (DatasetTaskKind.FILE_SEARCH, AgentTeacherKind.SEQUENTIAL_MULTI_AGENT): FILE_SEQUENTIAL,
    (DatasetTaskKind.GENERAL, AgentTeacherKind.BASE): BASE,
    (DatasetTaskKind.GENERAL, AgentTeacherKind.PARALLEL_MULTI_AGENT): GENERAL_PARALLEL,
    (DatasetTaskKind.GENERAL, AgentTeacherKind.SEQUENTIAL_MULTI_AGENT): GENERAL_SEQUENTIAL,
}


def render_template(template: str, task_prompt: str) -> str:
    return template.replace("{task}", task_prompt.strip()).strip()
