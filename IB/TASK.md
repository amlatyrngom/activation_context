# Task

Use this user-owned scratchpad for an objective, constraints, notes, or acceptance criteria when useful.

Agents must not read or edit this file automatically. They may access it only when the user explicitly asks, and at the relevant point in the work.


# Ready to move on harness co-design.
```
study.py
prompts.py
workflow.py
training.py
scoring.py
```

Let's consider the problem of mathematical reasoning. The goal is to create a workflow that maximizes problem-solving performance on a mathematical dataset.

Initially, the harness designer (Human, Claude Code, Codex, Tressoir, etc), writes maybe a simple workflow like this:
```py
def workflow(harness: HarnessRuntime, task_input: dict) -> tuple[str, list[dict]]:
    global MODEL_NAME
    problem = task_input["problem"]
    environment = Environment(harness, python_version="3.12", packages=["numpy", "scipy"], readonly=True, subagents=True)
    solver = Agent(harness, MODEL_NAME, problem, env=environment, lora_name=None)
    run_results = solver.run()
    # Answer, and something serialized that can be used for scoring or training.
    return run_results.answer, [run_results.serialize()]
```

Now some performance can be gained simply by running RAFT or RAFT++ on the workflow above, with a basic lora.
However, there are many, many ways to improve this:
- Workflow improvements: fundamentally about improving the code above.
    - E.g., parallelism + synthesis. Retrieval of similar problems, etc.
- Model improvements: fundamentally about improving the model(s) used in the workflow (their loras and ACs).
    - Maybe a lora for retrieval, one for non-retrieval-generation, one for retrieval + generation, and one for synthesis.
    - The synthesizer can take full subagent trajectories from the AC to have better synthesis material.
    - Models can be improved under a fixed workflow, or the workflow can be improved under fixed models (model+loras+acs).
- Training/Study Dataset improvements: augmenting the dataset used at study time to improve performance.
    - Use an oracle model with <problem, answer> pairs to batch generate reasoning chains.
    - Do rounds of SFT on them, iteratively improving the model.
- Test-time dataset improvements: augmenting the dataset used at test-time to improve performance.
    - This is about adding reasoning chains to underlying examples and making them easy to retrieve and use in the workflow.

The key idea is to make all of the above programmatically accessible to the harness designer, so that it can iteratively improve them through code.
Let's think through the study interface, as that's what's freshest in our minds.
The ultimate goal of dataset study is to create training datasets that improve either the workflow or the model.
```py
def dataset_study(harness: HarnessRuntime) -> dict:
    global STUDY_MODEL_NAME
    # Load any relevant datasets for the study model.
    dataset = harness.dataset_manager[...]
    ... # Code calling the study model name with the dataset to create examples for training.
    ... # Example: an agent is given the problem with the answer, and is tasked with generating a decontaminated code+reasoning trajectory that solves the problem, have its own lora, etc for improvement, etc.
    ... # Or give it the problem without the answer, and have it generate a reasoning chain that leads to the answer, etc.
    ... # Or first solve the solvable ones, then for the unsolvable ones, retrieve similar solved problems, and iterate until convergence or a timeout, or something like that.
```