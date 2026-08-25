---
name: tressoir-canon
description: Maintain, promote, split, realign, and merge-reconcile lightweight durable project canon and its known-good reusable CANON_ARTIFACTS.
---

# Tressoir Canon

Within the **IB (Interpretable Blueprint)**, use plain Markdown canon for knowledge that should influence future work beyond the current task. This is a small, human-readable analogue of a canon learner: it has no TOML schema, database, background service, or automatic log ingestion, but it preserves the useful curation behaviors.

## Layout and retrieval

```text
IB/CANON/
├── ROOT_CANON.md
├── EDITOR_CANON.md
├── RELEASE_CANON.md
├── <SUBSYSTEM>_CANON.md
└── CANON_ARTIFACTS/
    ├── README.md
    └── <known-good reusable examples>
```

`ROOT_CANON.md` stays small and frequently readable. It contains only cross-cutting durable guidance plus an index with one-line descriptions of subsystem files. Read only the subsystem canon relevant to current work.

Prefer a few coherent domain files over many tiny files. Do not put all knowledge in the root, and do not create a subsystem file for one debugging note.

## Canon artifacts

Use `IB/CANON/CANON_ARTIFACTS/` for reviewed, known-good examples that are more useful in their native file format than rewritten as prose. Suitable contents include reusable configuration files, scripts, templates, fixtures, and small reference implementations.

Canon artifacts are part of canon and **should be version-controlled**. Promote a file there only after its behavior and reuse value are understood and validated. Give it a descriptive path, remove secrets and machine-specific assumptions, and document its purpose, usage, prerequisites, validation, and important limits in `CANON_ARTIFACTS/README.md` or a nearby Markdown file.

Do not use `CANON_ARTIFACTS/` for generated output, dependencies, binaries, raw downloads, active task work, unreviewed experiments, or one-off debugging. Keep candidates in `ARTIFACTS/` or `TMP/` until they are proven. When prose canon relies on an example, link to its stable relative path instead of duplicating its full contents.

## Canon entity types

Use clear Markdown headings or labels:

- **Intent** — a durable product or architectural direction.
- **Decision** — an accepted choice, rationale, and consequences.
- **Invariant / rule** — behavior that must remain true.
- **Constraint** — a compatibility, security, platform, or user boundary.
- **Convention** — a consistent local pattern that reduces ambiguity.
- **Technique** — a repeatable validated method.
- **Insight / learning** — a non-obvious fact established by evidence.
- **Glossary** — a project term and its precise meaning.
- **Reference** — a stable source, path, command, or specification.
- **Open question** — a durable unresolved issue and its closing condition.
- **Superseded item** — useful history linked to the current replacement.

Choose the smallest category that makes the item's role clear.

## Quality and promotion

A canon item should be durable, behavior-changing, evidence-based, correctly scoped, concise, and versioned when behavior may change.

At natural decision, implementation, and review milestones, proactively inspect current `STATE.md`, accepted artifacts, source, tests, and user corrections for stable lessons. Promote a lesson when future work would otherwise have to rediscover it.

Promotion means:

1. Search root and relevant subsystem files first.
2. Update or merge an existing item instead of appending a duplicate.
3. Rewrite task-specific chronology into timeless current-system guidance.
4. Put it in the narrowest coherent subsystem.
5. Record a stable source or code anchor when useful.
6. Update the root index only when a subsystem file is added or its scope changes.
7. After promotion, condense duplicated detail from `STATE.md`.
8. Do not make temporary `ARTIFACTS/` or `TMP/` files the permanent authority. Move a durable rule into prose canon, a reusable known-good example into `CANON_ARTIFACTS/`, or a workflow into a skill or documentation.

Do not promote transient progress, failed experiments without a reusable lesson, raw logs, or “we just did X” announcements.

## Proactive size management

Use judgment rather than rigid quotas, with these review triggers:

- Keep `ROOT_CANON.md` roughly below 20,000 characters.
- Consider splitting a subsystem file around 40,000 characters or earlier when it mixes unrelated domains.
- Split by coherent subsystem or retrieval need, not by arbitrary size chunks.
- A single incoming lesson may be decomposed across several existing files when it spans domains.
- Use strong headings and summaries inside a domain file before creating micro-files.

When splitting, update root index links, preserve stable anchors where practical, and verify no active item was lost or duplicated.

## Canon realignment

Run a focused realignment when canon becomes duplicated, stale, contradictory, cluttered with chronology, poorly grouped, or visibly drifted from current code/docs/tests. Also realign after a broad merge, pull, migration, or refactor that may invalidate prior guidance.

### Scope spectrum

- **Small cleanup:** deduplicate, tighten wording, remove resolved low-value questions, and regroup misplaced items.
- **Medium realignment:** reconcile recent durable lessons with existing canon; remove transient status and old implementation chronology; split or merge domain files for retrieval.
- **Broad realignment:** audit current source, docs, tests, accepted decisions, and `STATE.md` when visible drift or a large integration warrants it.

### Procedure

1. Read `ROOT_CANON.md`, every referenced domain file in scope, and the current authoritative source needed to judge drift.
2. Establish the current accepted system and user intent before editing memory.
3. Preserve durable guidance; remove or mark stale guidance that no longer describes the system.
4. Merge duplicates and near-duplicates instead of retaining chronological repetitions.
5. Move root clutter into coherent domains. Split mixed or oversized files; merge tiny overlapping files.
6. Convert completed phases, migrations, and “for now” narration into timeless present-tense guidance. Keep recent context only when it still explains a constraint.
7. Decompose incoming mixed lessons across the correct domains.
8. Promote only very stable cross-cutting items to the root.
9. Revalidate affected `CANON_ARTIFACTS/` examples and update, replace, or remove examples that no longer match current canon or source behavior.
10. Reconcile related skills and docs when they now contradict canon; do not leave several active sources saying different things.
11. Validate links, headings, index descriptions, canon-artifact references, contradictions, and changed-file scope. Summarize what moved, merged, split, or became superseded.

Realignment is curation, not an append-only merge and not an excuse to scan or rewrite unrelated project files.

## Merge and rebase reconciliation

A source merge and a canon merge are different problems.

1. Inspect `git status`, diffs, conflicts, staged files, and both branches' relevant IB changes.
2. Separate source conflicts from `STATE.md`, skills/docs, prose-canon, and canon-artifact conflicts.
3. Resolve source according to accepted product intent first when possible.
4. Reconcile canon against the resulting system; do not choose a side solely because its lines won a textual merge.
5. Preserve complementary durable lessons from both branches, deduplicate equivalent entries, and mark useful superseded history.
6. Reconcile skills, docs, and `CANON_ARTIFACTS/` examples when merged behavior changed a durable workflow.
7. Never synthesize, read, or rewrite `TASK.md` during merge work unless the user explicitly requests it.
8. Ask the user when branches encode genuinely incompatible product, safety, or knowledge-retention intent.
9. Validate source tests plus canon links and consistency after reconciliation.

Prefer an atomic complete Markdown file replacement over leaving a half-edited canon file visible.

## Example

```md
## Invariant — Root guidance registration is explicit and narrow

Setup may create skill adapters at unoccupied paths. Only when the user opts
in, it may create or append an idempotent `# Tressoir Guidance` section, or
upgrade the exact section generated by an earlier release, in root `CLAUDE.md`,
root `AGENTS.md`, or Pi-only `.pi/APPEND_SYSTEM.md`. It preserves edited,
duplicated, unrelated, and unsafe guidance targets. For selected Claude setup,
the same opt-in may merge the two documented Tressoir defaults into a regular,
non-symlink `.claude/settings.json` while preserving other keys and restrictive
permissions. Setup never edits nested Claude instructions, overrides, rules,
global instructions, or Pi `SYSTEM.md` files.

Evidence: accepted Tressoir External MVP root-guidance decision.
```

Avoid vague entries such as “be careful with setup.” Canon is curated guidance, not an event log.
