# Harness/Model Co-Design
## Updated Teacher Campaign
Note that we in probing/dev mode now. Avoid any long-running experiments.

### Public Trajectory KL
At pre-planning, go through our code, and identify the changes to make in:
- The dataset loaders that load trajectories.
- The ac model study file.
To make these (1) trajectories and (2) the AC KL items as close to what our agents/agentic programs would naturally use.
Don't overcomplicate; just make it as reasonably close as possible so the training isn't fighting the online rollouts.

Then, fix the following problem if it exists:
- I believe the current KL is litterally just one step, which I think is an anti-pattern, unless I am wrong. Let's now do this.
- The meaning of `max_post_compaction_tokens` changes.
- It means do KL over the whole assistant spans up to that point. Is this doable and what's actually recommended.

### KL over Our Own Trajectories
The idea here is to take a trajectory that did not have AC turned on, then to format it to make an SFT when the student has AC turned on.
- In line with the idea of KL over whole assistent spans, this should support having activation inputs in many parts of the message.

It's possible the way to unify the two is to remove the notion of prefix/suffix, and just require that the two lengths/patterns be the same (by pattern I mean the precise alternation of system/user/assistent/tool/etc; the assistant spans must be the same, and are used for KL). Give this to me in a decision item if it's a possibility.

Also encode the three default selections (so we don't have to bother even configuration). Anything with a score of 1, 1+unscored, or all. Defaults to all. I am not even sure there is any other option for KL.

### Co-Training over Our Own Trajectories (On-Policy)
This might be the main complexity here.
- During training, whatever is selected, whether the trajectories come from a teacher or from ours, whether they have logprobs or not, can we make the gradients flow to the activation model and its lora.
- Confirm this: if I am using the same model for agents and for AC models, the model itself is loaded only once; it's just that the loras differ.
- The requirement is that the agent lora being trained, is the main model lora in the target AC.
- So I want select_agent_runs to return tuple[str, str, str|None] -> None.
    - Alternatively, make this an object like TrainingTarget or something like that.
- This means the user/tool span is not strictly masked anymore right? At the latent pieces.
    - And this must work for trajectories that did not hace AC turned on.

Let's begin with a decision-heavy intent phase, then interface, then reviewed details (a subagent has to confirm the details when we get here; notify me of interface changes.)



```py
    intervention_frequency: int = 4 # 0=current behavior. 1 means half-point. 2 means 1/3, 2/3, etc.
    add_last_layer_intervention: bool = True
    prefer_attention_interventions: bool = True
```



Ok fork vllm as a submodule in IB/REPOS/vllm
See if you can reuse my git creds to push the fork to my github. Also commit the local code.
Then, make plan for how we'll implement this MVP solidly. The human-facing projection should show focus on the non-vllm interface changes, whereas the agent-facing projection should probably focus on vllm so reviewers know what you are about to do.
Hopefully, the local interface-level changes are:
1. Add interventions to the places that call/decode/train models whether hf or vllm.
2. Add interventions to the steps/serialization/caches/etc.
3. Make the changes to the AC/training/agent/etc.