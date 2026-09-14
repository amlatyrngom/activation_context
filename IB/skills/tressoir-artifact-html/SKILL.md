---
name: tressoir-artifact-html
description: Author or revise a raw .tressoir.html artifact with custom HTML, CSS, JavaScript, contained interactions, HTTPS or local resources, and live morph behavior.
---

# Tressoir HTML Artifacts

Use `.tressoir.html` when you know the exact interface you need and the structured `.tressoir.md` projection is not flexible enough.

The Tressoir Artifacts VS Code extension renders the file directly in a webview. Your HTML, CSS, and JavaScript are the page. For a new artifact, copy `html_artifact_template/ARTIFACT.tressoir.html` into the broad-task artifact directory and rename it unless the user requests a genuinely different composition. Copy `SIBLING.md` only when keeping the starter's example local link.

Keep the starter's pinned HTTPS core CSS and JavaScript references, single linear flow, and floating feedback dock intact. Replace its example content and add only narrow, token-based artifact CSS, preferably inline when it is short and unique to the artifact. Do not copy a `vendor/` tree or duplicate the standard `tressoir-linear.css` and `tressoir-linear.js` into the artifact. This is an authoring default, not styling imposed by the extension on arbitrary HTML. The extension bundles the same runtime bytes internally for Markdown, so do not fork the standard component styles.

Prefer `.tressoir.md` for plans, research, standard decisions, reveal rows, and focused review surfaces. Custom reports and explainers are common HTML uses. Keep plans and decision surfaces in `.tressoir.md`; Markdown accepts inline raw HTML and SVG, so a single table or diagram never requires moving a plan to another format.

## Folder placement

Put the artifact in the existing sticky broad-task folder under `IB/ARTIFACTS/`. Reuse that folder for related plans, subplans, explainers, interactive surfaces, sibling assets, data, and interaction files. A later HTML explainer does not need a new folder. Create a new upper-case folder only for a genuinely separate workstream or when the user requests one.

## Trusted executable content

Opening `.tressoir.html` runs authored scripts in the artifact webview. Only open artifacts from trusted authors and workspaces. This is not a safe preview for arbitrary downloaded HTML.

The webview hides the raw VS Code API and exposes a narrow helper, but authored code remains powerful within the rendered artifact and may load resources allowed by the content security policy.

## Contained helper API

Authored scripts can use:

```js
const context = window.tressoirNotebook?.context?.()
await window.tressoirNotebook?.openFile?.('./sibling.md')
await window.tressoirNotebook?.storeInteraction?.('choice', { option: 'A' })
const saved = await window.tressoirNotebook?.getInteraction?.('choice')
```

- `openFile(relativePath)` opens a sibling inside the artifact folder. Absolute paths and `..` traversal are rejected.
- `context()` returns one frozen object with the current artifact's generated `feedbackKey` and `interactionsFile`; it exposes no filesystem path or additional authority.
- `storeInteraction(key, value, filename?)` persists a JSON-serializable value.
- `getInteraction(key, filename?)` reads it back.
- The default filename is `interactions.json`.
- Alternate filenames must match `interactions.<suffix>.json`.
- Keys and serialized files are bounded by the extension.

Guard helper calls if the HTML might also be opened in a normal browser.

## Live morph behavior

External edits update the view by morphing the DOM rather than recreating the entire realm. Scroll, focused controls, open disclosures, and suitable runtime state can survive.

Use:

```html
<div data-morph-keep-class="open selected"></div>
<div data-morph-skip><!-- runtime-owned subtree --></div>
```

Listen for `tressoir:render` to reapply idempotent upgrades after both initial render and later morphs:

```js
window.addEventListener('tressoir:render', async () => {
  // Restore persisted selections or re-run safe visual enhancement.
})
```

A one-time script does not necessarily rerun after every morph.

## Styling and resources

- Inline `<style>` works.
- Relative stylesheets, scripts, images, and data files resolve from the artifact folder.
- Direct HTTPS stylesheets and scripts are permitted. Insecure HTTP scripts remain blocked.
- Inline scripts work because the extension applies its nonce.
- Relative local scripts work.
- `eval` and `new Function` are not allowed.
- Theme variables such as `--nb-bg`, `--nb-surface`, `--nb-text`, and `--nb-accent` follow the editor theme.

For the standard compact, network-backed artifact:

- reference dependencies with ordinary `<link href="https://…">` and `<script src="https://…">` tags so the same HTML works in a browser and the VS Code renderer;
- pin every dependency to an explicit release or immutable commit; never use `latest`, a mutable branch such as `main`, or another moving URL;
- prefer a CDN that serves JavaScript with the correct MIME type over a GitHub raw-file URL;
- preserve dependency order and use `defer` where the local starter does;
- include `integrity` and `crossorigin="anonymous"` when the publisher supplies a Subresource Integrity hash; and
- keep important explanatory HTML present before scripts run. If a script cannot load, Tressoir shows a visible resource error, but static content should still be useful.

Remote code executes with the same trusted-artifact authority as local authored code. Only reference publishers and exact versions the project trusts. If the user explicitly requires an offline artifact, vendor the required files as a deliberate project-specific adaptation, keep them inspectable, record their provenance, and decide whether the project should commit or ignore them.

The standard starter references pinned KaTeX, Prism, CodeMirror, and Tressoir linear assets. Use its
documented math, code, diff, single-decision, card, interaction, link, table/SVG, and floating feedback forms instead of
adding another component system. Write in simple, concise language and keep ordinary information
visible. Use a card only when one complete section is genuinely optional; do not turn individual
paragraphs or list items into disclosures. There is deliberately no static card box, separate
reveal component, or default item row.

Keep `data-tressoir-markdown` on standard cards so their public opening tags match Markdown. In raw
HTML, author the included `.card-body`, `.card-gutter`, and `.card-content` wrapper exactly as the
starter does. The full-height outer-edge gutter collapses a long open card without covering its
selectable, inset content; its bottom hook and end rule delineate the optional section.

Keep the feedback dock non-modal: its small bottom-right trigger opens the form above the page
without a backdrop, focus trap, or scroll lock. The document must remain readable and scrollable
while the form is open. Preserve the close button, Escape handling, and silent persistence.

When an HTML artifact needs a decision, use one `.decision` for one clearly worded question. Its
checkbox answers use stable interaction keys, and it has one **Free Response** textarea. The
starter derives **Unresolved** versus **Resolved** from whether any checkbox is selected. Free
Response text alone never resolves it. Preserve the `rows="2"` and
`data-tressoir-autogrow="2:6"` contract so the textarea grows through six lines and then scrolls.
Persistence is silent; do not add saved/restored status text. Do not use the decision indicator as
a decorative status badge elsewhere.

## Diff views and defaults

Use the standard diff-view presentation for both `.tressoir.html` and `.tressoir.md`: one clean,
linear list of files, open by default but independently collapsible, with each relative file path
opening its target inside VS Code. Separate files with open seams rather than nested cards. Combine
language syntax highlighting with addition/deletion highlighting. Keep the hunk prelude nearly
seamless with the code, and use a soft full-row hover/focus wash to reveal the collapse target
without interfering with the file link or selectable diff text. Contain horizontal overflow inside
the individual diff.

Use **Low** for an intent-and-risk overview, **Moderate** for meaningful per-file hunks with
explicit elisions, and **Detailed / Exact** for the complete available delta with exact line
mapping. Moderate is the default for plans and ordinary review; identify generated, binary,
unavailable, or truncated content explicitly. Do not add a level switcher or canned narration when
the artifact has one known purpose.

INTERACTIVE code suggestions default to Detailed / Exact sequential slices. Slice 1 is relative to
the accepted starting point; each later coarse slice contains only the new delta on top of the
previous slice, regardless of whether Git is still uncommitted. Fold minor edits into a coherent
slice and regenerate affected later slices if an earlier slice changes.

## Human-facing quality

Even though the format is unconstrained:

- use one simple linear document flow by default; specialized compositions should be requested or clearly justified by the content;
- keep body text near browser defaults and use a modest heading scale rather than display-sized typography;
- do not wrap the whole authored page in a decorative card or rounded outer sheet;
- let the starter automatically contain document tables and SVG diagrams; keep other overflow-risk preformatted text, canvases, and wide widgets in local horizontal scroll regions so they never widen or clip the document;
- lead with the user's goal and current decision;
- use plain labels and expand uncommon acronyms;
- keep important state visible;
- use the standardized decision component only for one actual question, never as a collection of unrelated controls;
- prefer concise visible prose; reserve a card for an optional whole section;
- make controls keyboard-accessible;
- respect light and dark themes;
- avoid ornamental interaction that obscures the work.

## Verification

Open the artifact in the custom editor and exercise:

1. initial rendering;
2. light/dark theme changes;
3. external source edits and morph preservation;
4. each interaction read/write;
5. sibling opening and traversal rejection;
6. local resources and, when promised by the artifact, offline behavior;
7. any third-party library under the actual webview content security policy.
