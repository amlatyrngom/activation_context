---
name: tressoir-isolation
description: Plan, adapt, create, or verify a host, shared Dev Container, or dedicated Agent Container setup with explicit writable and persistent boundaries. Use for environment-isolation requests, not ordinary Docker application work.
---

# Tressoir Isolation

Choose and configure the isolation mode that fits the project. The current environment determines
where changes are allowed; requesting a different future mode does not widen current access.

## Modes

| Mode | User | Agent | Automatically durable |
|---|---|---|---|
| Pure Host | host | host | current project |
| Pure Dev Container | same read-write container | same container | host-backed project |
| Host + Agent Container | host | dedicated container | mounted `IB/` only |
| User + Agent Containers | read-write user container | separate agent container | user: project; agent: mounted `IB/` only |

A dedicated Agent Container has read-only `/source`, disposable `/workspace`, and read-write
`/workspace/IB`. Otherwise, work normally in the current writable project within the user's task.

## Adapt the starter

Inspect the project's `.devcontainer/`, Docker files, manifests, toolchain, root guidance, Git
state, and IB before editing. Adapt working project configuration instead of replacing it.

The installer-owned reference bases are:

- `IB/skills/tressoir-isolation/templates/user.devcontainer.template.json`;
- `IB/skills/tressoir-isolation/templates/agent.devcontainer.template.json`;
- `IB/skills/tressoir-isolation/templates/agent-setup.template.sh`; and
- `IB/skills/tressoir-isolation/templates/README.template.md`.

Treat those files as read-only upstream references. Adapt the corresponding create-once working
files under `IB/isolation/`; those belong to the project and setup never overwrites them, including
under `--override`.

Start from `IB/isolation/user.devcontainer.json` or `agent.devcontainer.json`. Prefer relative
symlinks at `.devcontainer/user/devcontainer.json` and
`.devcontainer/agent/devcontainer.json` so VS Code's recognized configuration stays synchronized
with IB. Edit the canonical IB file through that link. Copy only when the filesystem or project
cannot use symlinks, and then call out the manual synchronization boundary. Preserve any occupied
Dev Container path rather than replacing it. Tailor the image, features, extensions, ports,
environment, and project lifecycle commands to the project.

After a Tressoir override refreshes the reference templates, compare each reference with its
adapted counterpart and port only changes relevant to the current project. Preserve the adapted
image, feature choices, mounts, ports, lifecycle commands, selected harnesses, extensions, and
local comments unless the user asks to change them. Never replace the adapted tree wholesale and
never edit the references to represent project-specific choices; doing either destroys the useful
upstream-versus-local comparison.

- **Pure Host:** add nothing unless the user requests container setup.
- **Pure Dev Container:** keep the project on a read-write host bind. Volumes are appropriate for
  caches, not the only source copy.
- **Host + Agent Container:** preserve the dedicated boundary below.
- **User + Agent Containers:** open independent, host-rooted user and agent configurations in
  separate VS Code windows. The user configuration is read-write; the agent boundary is dedicated.

## Dedicated Agent Container contract

Project-specific changes may alter the image and toolchain. Preserve:

- host source mounted read-only at `/source`;
- writable container-local source at `/workspace`, used as the agent's working folder;
- host `IB/` mounted read-write at `/workspace/IB`;
- `IB/isolation/agent-setup.sh` invoked from the mounted IB;
- source copied with dotfiles and symlinks intact and `IB/` excluded;
- project harness/config files included in the copy;
- the isolation notice appended to container-local root `AGENTS.md` and `CLAUDE.md` only when each
  file exists;
- a named non-root development user with host UID/GID alignment;
- passwordless `sudo`, a user-owned container-local home and standard runtime directories, and
  writable home/cache/workspace surfaces without changing host source ownership; and
- startup failure when `/source` is not read-only or `/workspace/IB` is not writable.

The notice says that `/workspace` source edits are scratch-only and durable handoffs belong in IB.
Never add it to host guidance. Resolve permission problems by aligning the container user and
repairing only its container-local home; never use `chmod -R 777`, broad host ownership changes,
privileged mode, or a root agent. If the project adds mounts beneath the development user's home,
exclude them from recursive ownership repair. Do not mount the host home, Docker socket, or broad
credentials by default. This is filesystem and dependency
isolation, not a complete security sandbox.

## Start and verify

In VS Code, run **Dev Containers: Reopen in Container** and select the linked configuration. For
separate user and agent containers, open the same host project in a second VS Code window.

Verify:

1. the connected user is non-root;
2. `/source` rejects writes;
3. `/workspace` is writable and contains the expected source and harness files;
4. an IB test write appears on the host;
5. host source and guidance remain byte-identical; and
6. existing copied guidance contains the isolation notice.

Also verify passwordless `sudo`, writable `$HOME` and `${XDG_CACHE_HOME:-$HOME/.cache}`, the
expected Node/npm versions, each selected CLI, and each declared official VS Code extension. Pi has
no official editor extension; do not replace it with an unrelated marketplace package.

Rebuild the Agent Container to refresh its source snapshot. If runtime validation is unavailable,
report exactly which checks remain.

Use proportional validation. Exercise harness selections and rendered JSON with fast static or
mocked tests. When the shared image, lifecycle, permissions, or mount boundary changes, use one real
Agent Container as the representative end-to-end check; it covers the stricter common runtime.
Resolve the User Container configuration separately, but do not rebuild both containers merely to
repeat the same tool installation. Reuse a still-relevant runtime result, and reserve the complete
release matrix for the publication checkpoint.

## Work and hand off while isolated

The agent may freely modify `/workspace` scratch and `/workspace/IB` within the task. Before
handoff, place durable artifacts, patches, or apply instructions in `IB/ARTIFACTS/<WORKSTREAM>/` and
identify any changes that remain only in scratch.
