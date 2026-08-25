# Plan title

Explain in one concise paragraph what this changes, why it matters, and what the human should review.

## Executive Summary

### Goal

State what success looks like in one or two sentences.

### Approach

Name the main approach, boundaries, and immediate consequences. Keep important facts visible.

## Requested Decisions

<article class="decision" data-tressoir-decision data-decision-state="unresolved"
  aria-labelledby="plan-decision-question">
  <header class="decision-header">
    <div>
      <h3 class="decision-title" id="plan-decision-question">What immediate choice must the human make?</h3>
      <p class="decision-context">Check every applicable answer. Choosing at least one marks the decision resolved.</p>
    </div>
    <span class="decision-state" data-decision-indicator role="status" aria-live="polite">Unresolved</span>
  </header>
  <fieldset class="decision-options">
    <legend class="visually-hidden">Decision answers</legend>
    <label class="decision-option">
      <input type="checkbox" data-tressoir-input="area.decision.recommended">
      <span><strong>Proceed with the recommendation</strong><small>Explain its immediate effect.</small></span>
    </label>
    <label class="decision-option">
      <input type="checkbox" data-tressoir-input="area.decision.revise">
      <span><strong>Revise first</strong><small>Describe the necessary change below.</small></span>
    </label>
  </fieldset>
  <div class="field decision-feedback">
    <label for="area-decision-response">Free Response</label>
    <textarea id="area-decision-response" rows="2" data-tressoir-input="area.decision.feedback"
      data-tressoir-autogrow="2:6" placeholder="Add anything the choices miss…"></textarea>
  </div>
</article>

## Milestones

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">M1 — First coherent implementation chunk</span>
    <span class="card-oneliner">Describe the concrete result in plain language.</span>
    <span class="card-badge">Planning</span>
  </summary>

#### Planning Overview

Describe what changes and where, then explain rationale, boundaries, and important edge cases.

#### Planned Changes

`path/to/file · symbol()`

```diff
@@ path/to/file — symbol() @@
- before
+ after
```

</details>

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">M2 — Dependent chunk</span>
    <span class="card-oneliner">Detail this after its prerequisite is resolved.</span>
    <span class="card-badge">TBD</span>
  </summary>

#### Planning Overview

State the dependency that must be resolved before detailed changes are planned.

</details>
