"""Small study prompt presets; generation and caching stay in dataset_study."""
from .dataset import DataModality

STUDY_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "has_valuable_question": {"type": "boolean"},
        "question": {"type": ["string", "null"]},
        "answer": {"type": ["string", "null"]},
    },
    "required": ["has_valuable_question", "question", "answer"],
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



def make_study_prompt(study_context: str | None, modality: DataModality = DataModality.TEXT) -> tuple[str, str, dict]:
    """
    Returns system prompt, instructions, json schema. The trajectory variant asks about a fact inside
    an agent's turns; both let the model declare a snippet worthless (`has_valuable_question: false`).
    """
    system_prompt = """
# Core Guidelines
- You are a study/exam-like questions formulator.
- You are given a snippet from the corpus students are expected to study and understand.
- From this, your goal is to generate a hard question (unanswerable with common knowledge or simple keyword look ups).
    - This is IMPORTANT: the question should be **hard**, or it won't test the student's learning.
    - The question should have an accompanying answer.
- Your priority is: a hard question along with its expected, correct answer.
- By default, keep your question short to medium-short unless the exam recommends longer questions.
- When the snippet holds nothing worth asking about (boilerplate, navigation, an empty listing, pure formatting), set has_valuable_question to false and leave question and answer null.
"""
    if modality == DataModality.TRAJECTORY:
        system_prompt += """
# Trajectory Snippets
- The snippet is part of an agent's conversation: its reasoning, its tool calls and the tool results.
- Ask about a specific fact that appears in these turns (a value a tool returned, a decision the agent took, an error it saw, a file or quantity it found), phrased so someone who read the turns can answer it.
- Never ask something answerable from the task statement alone, and never ask about the formatting of the transcript.
"""

    if study_context:
        system_prompt += f"""
# Exam Context
The following emphasizes the kind of questions students are expected to handle.
```
-----
{study_context}
-----
```
Frame your questions in light of this.
"""
    system_prompt += """
# Response Format
Your answer must be formatted as a json object like:
```json
{
    "has_valuable_question": true|false,
    "question": "..."|null,
    "answer": "..."|null
}
```
"""
    instructions = f"""
# Instructions
Generate the exam-like question for student's study.
"""
    return system_prompt, instructions, STUDY_JSON_SCHEMA


def make_label_prompt() -> tuple[str, str, dict]:
    """
    Returns system prompt, instructions and json schema for the snippet labeling task.
    """
    system_prompt = """
# Core Guidelines
- You are a relevance judge determining what snippets from a corpus students should pay attention to answer a question.
- You are given the study question, its reference answer, and numbered corpus snippets.
- Label each snippet index as:
    - **positive**: the snippet contains information that clearly helps answer the question.
    - **negative**: the snippet contains information that clearly does not help answer the question.
    - Leave out anything where you are unsure.
- Be careful:
    - The snippets are intentionally chosen to be *related* to the question without necessarily helping.
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
    instructions = """
Label the snippets as positives / negatives for this question. Leave out ambiguous ones.
"""
    return system_prompt, instructions, LABEL_JSON_SCHEMA



def make_source_study_prompt(study_context, modality, prompt_kind="qa", max_questions=1, require_multiple_sources=False):
    presets = {
        "qa": "Ask a concrete question whose answer depends on the supplied evidence.",
        "code": "Ask about code behavior, a concrete input/output, an API condition, or a dependency. Do not invent missing implementation.",
        "numeric": "Ask for a short calculation or quantitative conclusion grounded in the supplied numbers; include units in the answer when applicable.",
        "constraints": "Ask for a decision that applies explicit rules or constraints in the evidence; make all needed conditions available.",
    }
    if prompt_kind not in presets:
        raise ValueError(f"Unknown prompt_kind: {prompt_kind}")
    system, _, _ = make_study_prompt(study_context, modality)
    system = system.split("# Response Format")[0]
    system += "\n" + presets[prompt_kind] + f"\nReturn 0 to {max_questions} distinct, self-contained questions with concise correct answers. Return none for unusable evidence. Do not mention 'the snippet' or its numbering in questions."
    if require_multiple_sources:
        system += " Every question must require combining at least two numbered source units, and cite those units in evidence_indices. Skip if this cannot be done naturally."
    else:
        system += " Cite the source units supporting each answer in evidence_indices."
    entry = {"type": "object", "properties": {"question": {"type": "string"}, "answer": {"type": "string"},
             "evidence_indices": {"type": "array", "items": {"type": "integer"}}},
             "required": ["question", "answer", "evidence_indices"], "additionalProperties": False}
    schema = {"type": "object", "properties": {"questions": {"type": "array", "items": entry, "maxItems": max_questions}},
              "required": ["questions"], "additionalProperties": False}
    return system, "Return JSON with a questions array; evidence_indices use the zero-based Source numbers.", schema
