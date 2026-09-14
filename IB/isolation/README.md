# Tressoir isolation starters

These are adaptable starting points. Ask an agent to inspect your project and tailor the chosen
configuration's image, features, extensions, ports, environment, and lifecycle commands to your
actual stack while preserving the boundary described below.

| What you want | Start from | Result |
|---|---|---|
| You and the agent share one Dev Container | `user.devcontainer.json` | The project stays mounted read-write from the host. |
| You stay on the host; the agent is isolated | `agent.devcontainer.json` | The agent gets scratch source plus read-write IB. |
| You and the agent use separate containers | both files | Open each configuration in a separate VS Code window. |

Keep `agent-setup.sh` at `IB/isolation/agent-setup.sh`. Both templates use this one readable helper
to prepare the development user, verify the image's baked toolchain, and sync project dependencies.
The agent template also runs the source-copy and isolation checks from the mounted `/workspace/IB`.

Both templates build from the same image, which bakes in Node.js 24 LTS and npm, uv, CPython 3.14,
and the selected Claude, Codex, and Pi CLIs. Codex and Claude selections add their official VS Code
extensions. Pi is installed as a CLI only; there is no official Pi VS Code extension, so this
starter does not substitute a third-party one.

For this project, `agent-setup.sh` runs `uv sync --all-groups` when `pyproject.toml` is present.
The resulting `.venv` is container-local in the agent workspace and is recreated from the committed
`uv.lock` whenever the Agent Container is rebuilt.

## Put a template where VS Code finds it

Fresh Tressoir setup creates these relative links when the paths are free:

```text
.devcontainer/user/devcontainer.json  -> ../../IB/isolation/user.devcontainer.json
.devcontainer/agent/devcontainer.json -> ../../IB/isolation/agent.devcontainer.json
```

Keep the links when adapting the starter: edit the canonical file under `IB/isolation/` so the
recognized VS Code configuration and the shared IB copy cannot drift apart. If the filesystem or
project cannot use symlinks, copy the selected template instead and treat the two copies as a
manual synchronization boundary. Setup never replaces an occupied `.devcontainer` path.

Then open the project in VS Code, run **Dev Containers: Reopen in Container** from the Command
Palette, and select the configuration. For separate user and agent containers, open a second VS
Code window on the same project and select the other configuration.

The user configuration uses the normal read-write host-backed workspace, so deleting its container
does not delete project work. The agent configuration mounts host source read-only at `/source`,
creates disposable writable source at `/workspace`, and mounts the host's `IB/` read-write at
`/workspace/IB`. Only work placed in IB automatically reaches the user from that container.

The baseline image supplies a host-UID/GID-aligned `node` user with passwordless `sudo`. Startup
repairs ownership of its container-local home and standard cache/config/local/state directories,
then verifies the relevant writable surfaces. Ordinary files therefore remain user-owned while
system administration stays unrestricted through `sudo`. If the project mounts anything into that
home, adapt the ownership step so it does not traverse the added mount.

If the project already has a Dev Container, adapt that configuration and its development user
rather than replacing working project setup. Do not weaken the agent mount boundary when adapting
it.
