---
name: tressoir-plan
description: Create, review, and maintain a paired PLAN.md and PLAN.tressoir.md for substantial work that needs human decisions or approval before implementation.
---

# Tressoir Planning

Use this workflow for substantial design or implementation that needs a human checkpoint before
code changes. Read the `tressoir-artifact-md` skill before authoring the projection.

A plan is a pair in one sticky upper-case folder under `IB/ARTIFACTS/`:

- `PLAN.md` — verbose agent-facing source of truth.
- `PLAN.tressoir.md` — concise, complete human-facing projection.
- `TASK.md` — optional human-only scratchpad; access only when explicitly requested.
- `<artifact-stem>.interactions.json` — default per-artifact answers written by the extension.

Reuse that broad-task folder for later subplans, explainers, and interactive artifacts. Create a new
folder only for a genuinely separate workstream or when the user requests it.

## Intent first

When the central approach is unresolved, present the intent, realistic alternatives, immediate
tradeoffs, and a recommendation before inventing a detailed milestone plan. Crystallize the full
plan only after the human selects the shape.

## Required projection structure

Start with an H1 and lede, then put these sections first and in order.

### 1. Executive Summary

Give the mental model in roughly one screen: the goal, the chosen approach, the architecture or
workflow, and the important boundaries. A reader should get a good-enough understanding of the
system from the Executive Summary alone, without needing to open the milestone cards.

Actively use compact tables for comparisons, mappings, ownership, or constraints, and architecture
or data-flow diagrams for components, boundaries, and sequences when they make the system easier to
grasp. Prefer comprehension over arbitrary compactness; every visual should explain the system, not
merely decorate the plan.

### 2. Requested Decisions

Use the exact one-decision HTML component documented by `tressoir-artifact-md`. Each question must
be self-contained, offer realistic checkbox answers with one clear recommendation when appropriate,
and include one Free Response field. Show only decisions relevant to the current pass. When a
decision is integrated, replace the component with a concise accepted-decision record.

### 3. Milestones

Represent each coherent milestone with the standard optional-section card:

````html
<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">M2 — Name</span>
    <span class="card-oneliner">What this chunk delivers.</span>
    <span class="card-badge">Planning</span>
  </summary>

#### Planning Overview

Name what changes and where, then explain rationale, boundaries, and edge cases.

#### Planned Changes

`path/to/file · symbol()`

```diff
@@ path/to/file — symbol() @@
- before
+ after
```

</details>
````

Lifecycle labels are `TBD`, `Planning`, `Implementing`, `Review`, and `Completed`.

- A `TBD` milestone has only its concise overview.
- `Planning` or later includes `#### Planned Changes`.
- Put a backticked path or symbol directly before every planned snippet.
- Use the artifact skill's **Moderate** diff level by default: include meaningful per-file hunks
  sufficient to understand the plan and mark every omission explicitly. Use Low only for a genuine
  overview and Detailed / Exact only when the exact patch is necessary for the decision.
- When changing a public interface or lifecycle, show its types, creator, owner, consumers, and
  end-to-end data flow. Skip that ceremony for internal edits without interface consequences.

## Approval and implementation loop

1. Research enough to avoid a naive plan; put disposable evidence in `IB/TMP/`.
2. Update `PLAN.md` and project every human-relevant fact into `PLAN.tressoir.md`.
3. Run the Markdown artifact checker and inspect source/projection agreement.
4. Hand back for decisions or approval.
5. Integrate answers from chat or the projection's scoped interaction file into both files.
6. Do not implement product code until the relevant milestone is agreed.
7. Mark approved work `Implementing`, implement it, and validate it.
8. Move it to `Review` with the actual completion report.

## Completion report

At `Review` or `Completed`, lead each milestone with:

- **What landed** — actual behavior and files.
- **Drifts, challenges, and unplanned steps** — honest differences from the forward plan.
- **Focused actual diffs** — Moderate, per-file excerpts by default; make omissions explicit.
- **Validation** — commands and important observed results.

Keep the original planned changes below as reference and update `IB/STATE.md` at milestone
boundaries.
