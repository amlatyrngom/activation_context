"""
Sandbox images and task-specific setups for agent benchmarks. Images: `default_dockerfile()` (the richer default) and
domain variants built from `render_dockerfile`. Setups: the `AgentEnvSetup` subclasses in `activation.agent.agent_env`
(write files from the task datum, copy a cached checkout in), referenced by the loaders on `DatasetTask.env_setups`.
"""
from .default import APT_PACKAGES, DEFAULT_BASE, PIP_PACKAGES, default_dockerfile, render_dockerfile
