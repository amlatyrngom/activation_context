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

# Next Step: Activation Context Modeling and Training
## Modeling
Here is what we'll broadly aim for:
```py
class ActicationContextModel: # This is now the one canonical activation context model. Delete the retrieval-specifc path.
    """
    Accepts input of the form:
    messages = [
        {
            "role": "x",
            "content": {
                {"type": "text", "text": "Here is subcontent1",
                {"type": "activation_context", "messages": [...]},
                {"type": "text", "text": "Here is subcontent2"}, 
                {"type": "activation_context", "messages": [...]},
            }
        },
        ...
    ]
    and produces embedding tensor.
    The recursive AC messages are independently passed into this model to get their values, ran into a standard ffn-based adapter, then passes in as embeddings at their right full position in the input stream.

    It's important to have a very simple (cpu) cache of hash -> final tensor (NOT inner kvs). Just that given the same messages input (whether top-level or recursive), avoid recomputation.

    It's also important to support (mostly through the utils files), schedule and batching that makes parallel calls to forward be very efficient. I don't know whether there should be a different mode for training and rollouts. I also don't know if we should reduce engine occupancy by ~5-10% to give room for this (is vllm literraly hogging things in way that makes any other side cache impossible?)

    Tentative first architecture (give me an html visualization so I know we agree this makes sense).
    - Run the recursive steps, adapt their output, then tokenize with placeholders where these outputs would go.
        - Cache comes into play at recursive steps here or at the adapt step.
    - Run the tokens through the frozen embeddings layer of the base model. Replace the embeddings.
    - Do a self-attention based pooling to get the main major reduction in computation.
    - At the same time, on do a larger pooling pass to create the view tokens. This creates the target number of tokens + a learned view marker. Append them to the stream.
    - You now have <N/P_initial, N/P_target> tokens.
    - Don't forget to add any necessary positional embeddings.
    - Forward through the model as usual and read out the view tokens + some kind of head ffn (different from the recursive adapter; this targets the generative model, not the AC itself).
    - Attach a lora to the base model.
    - I am undecided on whether remove causality is worth it (embedding models don't seem to care either way, but this is slightly different). Let's defer if it introduces too many hacks to make it work. But let's do it if it's just a flag change.
    """
    def __init__(
        self,
        harness: HarnessRuntime,
        side_base_model_name: str,
        side_lora_name: str,
        target_model_name: str, # read the target d_model from here.
        pooling_factor: int = 4,
        pooling_window: int = 32,
        target_compression_ratio: float = 16.0, # Reduce the given input by this much.
    ):
        pass # Every other parameters should be as 


    def forward(
        messages: list[dict]|str, # support simple string input by converting it to a user message.
    ):
        pass


    def forward_batch(
        messages: list[dict]|str, # ...
        for_training: bool, # needed ???
    ):
        pass
```

