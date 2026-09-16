"""
(Not in the main code for now).
Currently, we can only elicit certain behaviors through the prompt, and trust the agent, even in a 2B-parameter model, to respect it.
I think that's unrealistic at this small scale of models.
Ultimately, if we are doing harness-model co-design, this is the kind of programmatic control harness designers should have anyway.

Agent Programs are the remedy. These are harness-written logic, easily extensible, to programmatically perform a given workflow.

Here is how they operate:
- Takes in the initialized but unstarted agent.
- Execute their program.
    - Can ultimately call agent.augment_context(messages={text, activation, etc}):
        - Also a convenient run_result = agent.run_subagent(...) which appends to run results to save the program from boilerplate.
        - Also a convenient agent.set_final_answer(...). If programmatically set, no need for a run.
- There can be many such programs.
- Those get appended to the initial message sent to the agent.

So pure parallel solving + synthesis becomes:
- Run N solvers programmatically.
- Augment context. Ask the main agent to just synthesize.

Now there is a problem (just like our tools I guess) around serializability. Pick any common sense/convenient approach.
But double-check the following:
- The harness designers can make adhoc tools and programs, and run result serde won't prevent harvesting.
- As in, the env/tool/program class can live in the test script that run the agents; not necessarily in the main codebase?
- If not, that's probably also fine: it just means the main code base has to have well-known extension points to avoid task-specific pollution.

(Unrelated issue: I think score should be float|None = None).
"""

# @AI: AgentConfig gains agent_programs: dict[str, list[tuple[AgentProgram, dict]]].
class AgentProgram:
    def __init__(self, agent: 'Agent', otherargs):
        pass

    def execute(self):
        pass