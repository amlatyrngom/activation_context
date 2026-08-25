---
name: tressoir-artifact-md
description: Author or revise a human-facing .tressoir.md artifact, including projected plans, research, decisions, interactions, and attention-aware presentation.
---

# Tressoir Markdown Artifacts

Use this skill when a human should review, decide on, or navigate substantial work in a rendered
`.tressoir.md` surface stored in the **IB (Interpretable Blueprint)**.

Write ordinary Markdown by default. The extension supplies the accepted linear light/dark theme,
math, syntax-highlighted code and diffs, overflow containment, stored controls, and floating
**Feedback Form** automatically. Do not link KaTeX, Prism, CodeMirror, or the linear CSS/runtime.

## Choose and place the artifact

Use `.tressoir.md` for plans, research, reviews, and decision surfaces. Use plain Markdown for
lightweight notes. Use `.tressoir.html` only when arbitrary HTML/CSS/JavaScript is genuinely useful.

Keep related sources, projections, interaction files, and later explainers together in the existing
upper-case broad-task folder under `IB/ARTIFACTS/`. Create another upper-case folder only for a
genuinely separate workstream or when the user asks.

For non-interactive work, keep an agent-facing source and human projection together, such as
`PLAN.md` plus `PLAN.tressoir.md`. The projection is the complete human-facing surface: include every
material decision, caveat, change, and validation result there. Never read or edit a plan-local
`TASK.md` unless the user explicitly asks at the relevant point.

## Public document shape

Start with a single H1 and one concise lede paragraph. Those ordinary Markdown lines are the visible
heading and description.

```md
# Artifact title

One concise paragraph explaining what this is and why it matters.

## First section

Keep ordinary information visible in a simple linear flow.
```

Frontmatter is optional configuration only. Use it rarely, for example:

```yaml
---
feedback: disabled
links:
  - ./genuinely-extra-library.js
---
```

`links:` is only for a genuinely extra dependency. Interactions are stored by default in
`<artifact-stem>.interactions.json`; `interactions: shared` is the explicit compatibility override.

## Optional-section card

Use a card only when one complete section is genuinely optional. Keep concise detail visible. The
marked body accepts ordinary Markdown; do not add the runtime-owned body wrapper or collapse gutter.

```html
<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">One optional complete section</span>
    <span class="card-oneliner">Open the full section</span>
    <span class="card-badge">Optional</span>
  </summary>

### Ordinary Markdown inside the card

Paragraphs, lists, code, math, and tables remain Markdown here.

</details>
```

The renderer adds the accepted selectable content column and full-height outer-edge collapse gutter
with its bottom end hook. Do not nest marked cards.

## One decision

A decision represents one question. Give every checkbox and the single **Free Response** textarea a
stable, artifact-specific key. Checking at least one answer resolves it; Free Response alone does
not. Keep the textarea at `rows="2"` and `data-tressoir-autogrow="2:6"`.

```html
<article class="decision" data-tressoir-decision data-decision-state="unresolved"
  aria-labelledby="release-question">
  <header class="decision-header">
    <div>
      <h3 class="decision-title" id="release-question">Which parts should change before release?</h3>
      <p class="decision-context">Check every applicable answer. Choosing at least one marks the decision resolved.</p>
    </div>
    <span class="decision-state" data-decision-indicator role="status" aria-live="polite">Unresolved</span>
  </header>
  <fieldset class="decision-options">
    <legend class="visually-hidden">Decision answers</legend>
    <label class="decision-option">
      <input type="checkbox" data-tressoir-input="release.scope.accept">
      <span><strong>Nothing — accept it as shown</strong><small>The component is ready.</small></span>
    </label>
    <label class="decision-option">
      <input type="checkbox" data-tressoir-input="release.scope.adjust">
      <span><strong>Adjust it</strong><small>Describe the necessary change below.</small></span>
    </label>
  </fieldset>
  <div class="field decision-feedback">
    <label for="release-response">Free Response</label>
    <textarea id="release-response" rows="2" data-tressoir-input="release.scope.feedback"
      data-tressoir-autogrow="2:6" placeholder="Add anything the choices miss…"></textarea>
  </div>
</article>
```

Persistence is silent. Do not add saved/restored status text or reuse the decision indicator as a
decorative badge.

## Standard content

- Use `\(...\)` for inline math and `\[...\]` for display math. Dollar delimiters are disabled.
- Use fenced language code blocks and `diff` or language-aware `diff-LANGUAGE` fences.
- Put a backticked file path or symbol immediately before a planned code/diff snippet.
- Tables and SVG diagrams gain local horizontal scrolling when they are wider than the document.
- Use raw HTML/SVG only when it materially clarifies structure.
- Keep simple, concise language and modest heading depth. Do not turn ordinary rows into disclosures.

The automatic floating **Feedback Form** is enabled unless `feedback: disabled` is present. It uses
highlighted Markdown, persists silently, and stays non-modal so the page remains readable and
scrollable while the reader writes.

## Reading human input

Controls write their primitive values under each `data-tressoir-input` key. The automatic feedback
text uses `<artifact-stem>-free_form_feedback`. Read the artifact's scoped interaction file on the
next turn, integrate answers into both source and projection, then remove resolved controls and
their obsolete stored keys. Never repurpose a published key.

## Template and checker

Copy the relevant file from `user_artifact_md_template/` only when the destination does not exist.
For a projected subplan, also create its matching agent-facing source. Preserve any existing
interaction file and do not copy placeholder keys.

Run before handoff:

```bash
node IB/skills/tressoir-artifact-md/scripts/check_md.js path/to/file.tressoir.md
```

Then open the artifact in the custom editor and verify light/dark rendering, interactions,
keyboard use, local links, wide content, feedback, live morphs, and persisted reloads.

## Trust boundary

`.tressoir.md` is trusted executable content, not a sanitized preview. Raw HTML and scripts can run,
and extra declared resources can cause network access or local caching. Open only trusted artifacts
and workspaces.
