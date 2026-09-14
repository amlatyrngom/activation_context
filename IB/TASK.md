

DEFERED: custom static workflows. If the harness-designer wants that, they should put that in the task description of the main agent, and rely on it to follow it.

## Planning
Let's open a fresh new artifact folder and a fresh new phase: Agent AC Pre-Training
M0 of that slice 4a should be about the minor changes I've prototyped (see git diff):
- (agent env setups needed for many benchmarks, enabling the general_subagent tool by default, etc).
- This should be relatively quick.

M1 should be about looking into making training multi-gpu and sky, and also looking into MIT orcd. (I've used it for an old version of this project, and I've attached a skill we used to use for it: IB/TMP/ORCD_OLD_SKILL.md).
- We can use of 2xH200s from there.
- You'll have to make a script near identical to sky.py, like orcd.py (with all the syncing features and such).
- Regarless, we need to make sure that our rollouts can be on 2 gpus, which should be the case.
- Also make sure our hardware auto-tune works correctly there.
- We should also try to find a hassle-free way to make training multi-gpu. I've been avoiding because, from what I previously know, too much of the code has to architecture to shard training examples. MY HOPE: that all the training ddp logic can isolated inside the various utils files, so nothing else knows about it.


On key line of non-implementation work actually occurs at pre-planning (interactive discussion) to figure out what exactly our training schedule should be, especially the teacher trajectory thing. Once we have that cached, the rest, include the mass pretraining, is just programmatic transformations of trajectories to make AC/Agent sft items.
I especially want to settle the task choices/sizes, etc.


Musique: 0.2. # Search.
DAPO: 0.1.
OMNI-Math Hard: 0.15
LCA: 0.3
