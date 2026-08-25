---
name: tressoir-structured-review
description: Perform a structured review of a plan, implementation, diff, or risky change using Tressoir verdicts and disciplined solver-side finding reclassification.
---

# Tressoir Structured Review

Use structured review for non-trivial plans, broad diffs, security-sensitive changes, compatibility boundaries, lifecycle code, or work nearing handoff.

Use a harness-native reviewer or isolated agent when readily available and proportionate. Otherwise perform the same review directly in the current context. A subagent is optional, not a requirement.

## Review inputs

Gather:

- the authoritative user request and accepted decisions;
- relevant `IB/STATE.md` and canon;
- the plan or artifact being reviewed;
- changed files or focused diffs;
- tests already run and tests still pending;
- explicit out-of-scope behavior;
- accepted functionality the review must not re-litigate.

Store disposable reports under `IB/TMP/`. Put a durable report beside the artifact only when the human will use it.

## Verdict vocabulary

Return exactly one overall verdict:

- `pass` — correct and appropriately scoped.
- `pedantic` — materially correct; only low-value polish remains.
- `incorrect-or-missing` — a requirement is absent or behavior is wrong.
- `oversimplified` — the solution omits necessary cases or invariants.
- `overcomplicated` — complexity exceeds what the task needs.
- `anti-pattern` — the approach introduces a known harmful pattern.
- `pattern-mismatch` — it conflicts with established neighboring design without justification.
- `duplication` — it recreates existing behavior or sources of truth unnecessarily.

## What to review

Check:

1. **Task fit** — every explicit requirement is addressed and no prohibited action occurred.
2. **Accepted intent** — do not override or re-litigate the user's chosen design.
3. **Scope** — the result is neither underbuilt nor inflated.
4. **Invariants** — security, permissions, concurrency, persistence, rollback, and compatibility remain correct.
5. **Integration** — names, paths, APIs, and neighboring patterns are coherent.
6. **Failure behavior** — partial failure and collisions are honest and safe.
7. **Tests** — focused coverage proves the risky behavior; missing broad tests are identified without pretending they ran.
8. **Documentation** — important trust and platform boundaries are visible to users.

Findings should name a concrete file/symbol or plan section, explain impact, and propose the smallest correction.

## Solver-side reclassification

The implementer must independently classify every reviewer finding as:

- `genuine` — correct and worth fixing or explicitly deferring.
- `cheap-nit` — low-risk polish worth doing if inexpensive.
- `reviewer-overkill` — theoretically defensible but disproportionate to the task.
- `bad-review` — conflicts with requirements, accepted intent, or evidence.

Do not let a reviewer expand the project by default. Reject overkill and bad review with a short reason.

## Loop

1. Review and record the verdict/findings.
2. Reclassify each finding.
3. Apply genuine fixes and worthwhile cheap nits.
4. Re-run focused tests.
5. Review the revision again when the change is substantial.
6. Stop at `pass` or `pedantic`, or when the human explicitly accepts a residual issue.
7. Report the final verdict, fixes, validation, and residual risks.

For a tiny change, one direct pass is enough. Do not manufacture a review ceremony.
