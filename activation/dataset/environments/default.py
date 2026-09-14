"""
The default sandbox image: python:3.12-slim plus the scientific stack and the shell tools the benchmark tasks lean on.
`default_dockerfile()` writes the recipe under the synced area and returns its path; `AgentEnv` builds and tags the image
by the file's content hash, so an unchanged recipe never rebuilds and a changed one gets a new tag. Domain images add
`extra_lines` (or another base); task-specific data goes through `AgentEnvSetup`, not the image.
"""
from __future__ import annotations

import hashlib

from activation.common.data_syncing import resolve_path

DEFAULT_BASE = "docker.io/library/python:3.12-slim"
APT_PACKAGES = ("git", "ripgrep", "jq", "build-essential", "curl", "less", "tree")
PIP_PACKAGES = ("numpy", "scipy", "pandas", "sympy", "matplotlib", "networkx", "scikit-learn", "pyyaml", "lxml",
                "beautifulsoup4", "tqdm", "regex", "requests", "pytest")
ENVIRONMENTS_FOLDER = "ENVIRONMENTS"


def render_dockerfile(base: str = DEFAULT_BASE, apt: tuple[str, ...] = APT_PACKAGES, pip: tuple[str, ...] = PIP_PACKAGES,
                      extra_lines: tuple[str, ...] = ()) -> str:
    lines = [
        f"FROM {base}",
        "ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 DEBIAN_FRONTEND=noninteractive",
    ]
    if apt:
        lines.append(f"RUN apt-get update && apt-get install -y --no-install-recommends {' '.join(apt)} && rm -rf /var/lib/apt/lists/*")
    if pip:
        lines.append(f"RUN pip install --no-cache-dir {' '.join(pip)}")
    lines += ["RUN mkdir -p /workspace /tmp/agent_outputs", "WORKDIR /workspace", *extra_lines]
    return "\n".join(lines) + "\n"


def default_dockerfile(base: str = DEFAULT_BASE, apt: tuple[str, ...] = APT_PACKAGES, pip: tuple[str, ...] = PIP_PACKAGES,
                       extra_lines: tuple[str, ...] = ()) -> str:
    """The Dockerfile path for `AgentConfig.env_dockerfile_path`: ENVIRONMENTS/<sha12>/Dockerfile under the synced area."""
    text = render_dockerfile(base, apt, pip, extra_lines)
    digest = hashlib.sha256(text.encode()).hexdigest()[:12]
    path = resolve_path(f"{ENVIRONMENTS_FOLDER}/{digest}") / "Dockerfile"   # resolve_path treats a suffix-less name as a folder
    if not path.exists() or path.read_text() != text:
        path.write_text(text)
    return str(path)
