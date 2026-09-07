"""
Every prompt of the dataset study: the two study kinds (a hard question with its answer, or a
search-friendly description), the relevance-label prompt for either kind, their JSON schemas and
the chat message builders. `dataset_study.py` keeps only the loops.
"""
from .dataset import LabeledRetrievalQAExample

STUDY_QA_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "question": {"type": "string"},
        "answer": {"type": "string"},
    },
    "required": ["question", "answer"],
    "additionalProperties": False,
}

STUDY_DESCRIPTION_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "needs_abstraction_bridge": {"type": "boolean"},
        "retrieval_summarization": {"type": ["string", "null"]},
    },
    "required": ["needs_abstraction_bridge", "retrieval_summarization"],
    "additionalProperties": False,
}

LABEL_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "positives": {"type": "array", "items": {"type": "integer"}},
        "negatives": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["positives", "negatives"],
    "additionalProperties": False,
}


def _study_context_block(study_context: str | None, framing: str) -> str:
    if not study_context:
        return ""
    return f"""
# Exam Context
The following emphasizes the kind of questions students are expected to handle.
```
-----
{study_context}
-----
```
{framing}
"""


def make_study_qa_prompt(study_context: str | None) -> tuple[str, str, dict]:
    """
    Returns system prompt, instructions, json schema for study q/a tasks.
    """
    system_prompt = """
# Core Guidelines
- You are a study/exam-like questions formulator.
- You are given a snippet from the corpus students are expected to study and understand.
- From this, your goal is to generate a hard question (unanswerable with common knowledge or simple keyword look ups).
    - This is IMPORTANT: the question should be **hard**, or it won't test the student's learning.
    - There should be a reasonably high lexical distance to prevent simple keyword lookups.
    - The question should have an accompanying answer.
- Your priority is: a hard question along with its expected, correct answer.
- By default, keep your question short to medium-short unless the exam recommends longer questions.
- Other notes:
    - Avoid things like "according to the text". Ask the question as is.
"""
    system_prompt += _study_context_block(study_context, "Frame your questions in light of this.")
    system_prompt += """
# Response Format
Your answer must be formatted as a json object like:
```json
{
    "question": "...",
    "answer": "..."
}
```
"""
    instructions = """
# Instructions
Generate the exam-like question for student's study.
"""
    return system_prompt, instructions, STUDY_QA_JSON_SCHEMA


def make_study_description_prompt(study_context: str | None) -> tuple[str, str, dict]:
    """
    Returns system prompt, instructions, json schema for study description tasks.
    """
    system_prompt = """
# Core Guidelines
You perform a search-friendly summarization.
- You are given a snippet from a corpus.
- You make summary that makes it easy for students to perform various kinds of search for semantic information retrieval.

# What a good summarization does
- It bridges an abstraction gap:
    - The snippet contains the concrete information like:
        - The code for specific algorithm.
        - A specific biological process.
        - An agentic trace.
    - The students search more abstract terms like:
        - Efficient sorting algorithms with custom comparators.
        - Biological processes that transform molecule X.
        - Traces where an agent is stuck on formatting issues.
    - Notice how the properties the students are looking are clearly from the snippets:
        - They are just not not explicitly stated as a distinct property yet, making it hard to search for them.
        - There can also be many such properties. List them all.
- The description should covers the gap between the two:
    - It identifies key, specific properties of the document that make such implicit searches easier.
    - It avoids restating things that already searchable within the documents.
    - It avoids being so vague the property is likely shared by many other things within the corpus.
        - It's distinctive: a summary maps strongly to the given snippet.
- It's relatively concise.

# What a good description does NOT do
- It does not summarize or paraphrase the surface content. The raw document is already indexed; restating it adds nothing.
    - To that end, you are given a boolean for `needs_abstraction_bridge`. When set, you can mark the snippet as not needing an abstraction bridge.
- It does not invent properties the text cannot support. If a property is ambiguous, omit it rather than guess.
- It's neither too abstract nor too specific. In case of doubt though, prioritize specifity.
"""
    system_prompt += _study_context_block(
        study_context, "Favor the properties such students would search for.",
    )
    system_prompt += """
# Response Format
Your answer must be formatted as a json object like:
```json
{
    "needs_abstraction_bridge": true|false,
    "retrieval_summarization": "..."|null
}
```
Set `needs_abstraction_bridge` to false and `retrieval_summarization` to null when the snippet's searchable properties are already stated in it.
"""
    instructions = """
# Instructions
Write the search-friendly description of the snippet.
"""
    return system_prompt, instructions, STUDY_DESCRIPTION_JSON_SCHEMA


def make_label_prompt(for_qa: bool = True) -> tuple[str, str, dict]:
    """
    Returns system prompt, instructions and json schema for the snippet labeling task: a question
    with its reference answer (for_qa) or a description alone.
    """
    if for_qa:
        given = "the study question, its reference answer, and numbered corpus snippets"
        target = "answer the question"
        positive = "the snippet contains information that clearly helps answer the question"
        negative = "the snippet contains information that clearly does not help answer the question"
        related = "related to the question without necessarily helping"
        task = "for this question"
    else:
        given = "a description a student might search with, and numbered corpus snippets"
        target = "find documents matching a description"
        positive = "the description clearly fits the snippet (the snippet is what such a search should find)"
        negative = "the description clearly does not fit the snippet"
        related = "related to the description without necessarily matching it"
        task = "for this description"
    system_prompt = f"""
# Core Guidelines
- You are a relevance judge determining what snippets from a corpus students should pay attention to {target}.
- You are given {given}.
- Label each snippet index as:
    - **positive**: {positive}.
    - **negative**: {negative}.
    - Leave out anything where you are unsure.
- Be careful:
    - The snippets are intentionally chosen to be *{related}*.
    - Don't mark something as positive just because of shared vocabulary.
- Every listed index must come from the snippet numbering; never list the same index on both sides.
"""
    system_prompt += """
# Response Format
Your answer must be formatted as a json object of snippet indices like:
```json
{
    "positives": [0, 2],
    "negatives": [3, 5, 6]
}
```
Each is a list of indexes that should map to one of the given snippet indices.
"""
    instructions = f"""
Label the snippets as positives / negatives {task}. Leave out ambiguous ones.
"""
    return system_prompt, instructions, LABEL_JSON_SCHEMA


def study_messages(system_prompt: str, instructions: str, snippet: str) -> list[dict]:
    """The chat for one study snippet (question or description kind)."""
    return [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": [{"type": "text", "text": f"# Corpus Snippet\n```\n{snippet}\n```\n{instructions}"}]},
    ]


def label_messages(
    system_prompt: str, instructions: str, example: LabeledRetrievalQAExample, pool_snippets: list[str], for_qa: bool,
) -> list[dict]:
    """The chat for labeling one example's pool: question + reference answer, or the description alone."""
    snippets = "\n".join(f"[Index={index}]\n```\n{snippet}\n```" for index, snippet in enumerate(pool_snippets))
    if for_qa:
        header = f"# Question\n{example.query}\n\n# Reference Answer\n{example.gold_answers[0]}\n\n"
    else:
        header = f"# Description\n{example.query}\n\n"
    return [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": [{"type": "text", "text": f"{header}# Snippets\n{snippets}\n{instructions}"}]},
    ]
