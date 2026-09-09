# Next: Agent Changes
The tricky part here, I think, won't be the agent changes themselves, the changes to engine input interface.

## Agent Changes
Let's just do the following:
Write these rules explicitly in an html diagram so I can review that we agree.
- Compaction: As described. Place the recursive compaction tree in the AC. The top-level trajectory sees <ac, text>.
    - When threshold is reached, agent is told to stop and call the compact tool. Refuse appending to the trajectory until it's done. Keep pestering with messages requiring the compaction tool and fail within ~5 approaches (magic constants).
    - Default to 1.0/10.0. Delta threshold 32k as is already the case I think, with some larger max cap that should almost never be reached by our natural code.
    - Let's do this: in run results, make compactions a different field from subtrajectories. It gets unconditionally unrolled by our trajectory selectors.
- Subagent: Also as described. Instead of returning {text: subagent answer, } return {activation: subagent trajectory text: subagent answer}.
    - Default to 1.0/20.0
    - In the subagent trajectory task say: {text: here is the parent that spawned you, activation: parent traj up to here + task, text: task}.
- Large tool call output: Again, the full text simply goes in the AC.
    - If something is too large, place it in the AC. Should still be truncated if unreasonably large.
    - Default to 1.0/40.0.
    - ac should have current parent traj as recursive input to condition its compression well.
- Semantic Search: fetch an extra top min(20, 2*top_k) as pass it in as an AC (the model sees top_k and this extra afterwards)
    - Default to 1.0/20.0.
    - ac should have current parent traj as recursive input to condition it well.
- One thing that probably has to change is the config not having a notion of output order between activations and text (activations seem to uniformly be before/after)?
    - I think the current ac_input should simply become messages_input. task is tacked after it if it's given.
    - Something has to cleanly change in the step object too. 
- Anything else I am forgetting?

## Engine Interface Changes
We need to use the vllm input interface.
This is low-level, so I can't spec it here. Spend time investigate (and reviewing) how exactly this should work and how we should implement it.

## Running/Testing
Write a test similar to our basic agent test.
However, for the first time, we have a step that depends on training preceeding steps well to be effective.

To avoid that, here is what I propose:
1. @staticmethod resume_from_run_result() Allows resuming from a run results.
2. In utils, have programmatic ways to construct synthethic trajectories up to a point.
3. Have a simulate_step(agent_config) method that just runs a single step.
    - Auto-executes the last tool call results if unpopulated.
        - Includes subagents/compactions/etc as needed.
        - Be careful about subagents now, as step could lead to them executing fully.
        - The calls should detect step mode and just create the objects and populate a dummy submit_answer ("Agent <agent_id> simulated run").
        - I think all other tools can be legit.
        - Makes a call down into the AC to get out its vectors, though they will likely be junk.
    - Makes the next expected message, but should not submit it I think (unless it does not somehow lead to junk getting produced).
    - Inspect the agent's state.

Unfortunately, this test will violate of our testing rules: it puts code that looks nothing like producting code in one of the core tests; let's ensure that it's a one time thing.
After this, we'll have pre-trained ACs (using the methods in slice 3a) to make regular runs/harvesting work.
    - The problem is that that's a full prod training pipeline, not an implementation test.
    - Still, try to make things as prod-like as possible.
    - This should not be an excuse to start introducing mocks, which will ruin e2e testability and completely pollute the testing interface.
        - I even prefer a call stats in the AC models if that allows minimal inspection over mocking.
    - Remember that you can always have detailed tests in the IB/ artifacts, just not the main testing code.
    - Give me the full test code in the plan for me to review.
    - You might have to lower some of the thresholds to make compaction testable without creating huge trajectories.
