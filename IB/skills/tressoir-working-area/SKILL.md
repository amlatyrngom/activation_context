---
name: tressoir-working-area
description: Use and maintain the project-local IB (Interpretable Blueprint), including explicit-request TASK.md access, proactively consolidated STATE.md, task ARTIFACTS, version-controlled CANON_ARTIFACTS, and disposable TMP.
---

# Tressoir Working Area

**IB (Interpretable Blueprint)** is the project-local, human-inspectable workspace and persistent memory shared across harnesses and sessions.

```text
IB/
├── TRESSOIR.md
├── .gitignore
├── TASK.md
├── STATE.md
├── ARTIFACTS/
├── TMP/
├── CANON/
│   ├── ROOT_CANON.md
│   └── CANON_ARTIFACTS/
│       └── README.md
└── skills/
```

## `TASK.md`

`IB/TASK.md` is the user's task description and scratchpad. It is neither an automatic startup input nor an agent-maintained status file.

- Read it only when the user explicitly asks you to use, inspect, summarize, reorganize, or act from it.
- Edit it only when the user explicitly requests an edit.
- Access it at the relevant point in the task, not reflexively at the start of every session.
- When asked to reorganize it, preserve the user's meaning: summarize completed work without erasing it and keep current/incomplete items visible.
- Do not autonomously clear, replace, normalize, or declare it stale.

A plan folder may also contain a local `TASK.md`. Apply the same explicit-request rule. Do not confuse it with `interactions.json`, which is the expected place to read human answers from a rendered artifact.

## `STATE.md`

`IB/STATE.md` is the concise living project state maintained by the agent. Read it before substantial work when it exists.

Record:

- accepted decisions;
- current progress and milestone state;
- important constraints and preferences;
- blockers and residual risks;
- important validation commands/results;
- the next meaningful step;
- pointers to active artifacts and relevant canon.

Update it at decision and milestone boundaries.

### Proactive consolidation

Because `STATE.md` is frequently loaded, keep it lean. When it grows to roughly 16,000 characters, proactively condense it toward 8,000:

1. Replace completed chronological entries with grouped outcome summaries.
2. Remove superseded constraints and duplicate details.
3. Preserve active decisions, current architecture, validation, risks, and next steps.
4. Promote durable lessons into the relevant canon file before deleting their only useful explanation.
5. Point to authoritative canon, skills, docs, or artifacts rather than duplicating them.
6. Avoid machine-specific absolute paths unless they are necessary current evidence.

Consolidation is maintenance of agent-owned state and does not require rewriting `TASK.md`.

## `ARTIFACTS/`

Use `IB/ARTIFACTS/` for durable outputs likely to be reviewed or reused:

- plan and research pairs;
- explainers and interactive artifacts;
- reusable scripts or fixtures;
- architecture notes;
- handoff reports.

Use one sticky upper-case folder per broad task or coherent workstream, not per artifact or phase. Reuse that folder for the task's whole lifetime. It may hold the original `PLAN.md` + `PLAN.tressoir.md`, a user-requested `<STEP>_PLAN.md` + `<STEP>_PLAN.tressoir.md` subplan, an explainer, research, a raw HTML view, and later surfaces such as `NEXT_INTERACTIVE.tressoir.md`. Shared sibling assets and `interactions.json` also stay there. Create a new upper-case folder only for a genuinely separate workstream or when the user requests one.

Before creating a `.tressoir.md`, `.tressoir.html`, or plan, read the corresponding skill. Use the template folder and checker described by `tressoir-artifact-md`.

When an artifact is completed or superseded, keep it only while it remains a useful input. Move clearly old durable work to an `ARTIFACTS/ARCHIVES/` folder when that helps retrieval. Ask before moving recent, ambiguous, or canonical-looking artifacts.

Durable canon and skills should not treat a temporary plan or report as their permanent authority. Promote the timeless lesson into canon, a skill, documentation, or a referenced helper, then point future work to that authoritative source.

## `CANON/CANON_ARTIFACTS/`

Use `IB/CANON/CANON_ARTIFACTS/` for reviewed, validated files that are known-good reusable examples: configuration files, scripts, templates, fixtures, and small reference implementations. These are canonical project knowledge in executable or native form, not task-working artifacts.

Canon artifacts should be version-controlled. Use descriptive stable paths, exclude secrets and machine-specific state, and document purpose, usage, prerequisites, validation, and limits in the directory `README.md` or a nearby Markdown file. Keep candidates in `ARTIFACTS/` or `TMP/` until their correctness and reuse value are established. Revalidate affected examples when source behavior or prose canon changes.

## `TMP/`

Use `IB/TMP/` for disposable internal material:

- logs;
- screenshots;
- data dumps;
- exploratory scripts;
- temporary downloads;
- intermediate research notes.

Do not present a low-value debugging dump as a durable artifact. Promote material into `ARTIFACTS/` or `CANON/` only after it becomes reusable; promote native-file examples into `CANON/CANON_ARTIFACTS/` only after validation.

Setup may create a narrow default `IB/.gitignore` when none exists. It ignores `TMP/`, `vendor/` cache directories, `.env`, and `.DS_Store`; it deliberately keeps state, prose canon, canon artifacts, skills, durable task artifacts, and interaction records visible to version control. Projects decide their final version-control policy. Do not silently edit an existing `.gitignore`.

## Normal operation

- Work end-to-end unless the user requested a checkpoint or the plan forbids implementation.
- Treat user edits as normal collaboration, not an implicit stop signal.
- Do not infer a stop request from changes under `IB/`; stop only on a clear request or genuine design blocker.
- Preserve existing files. Report collisions instead of overwriting.
