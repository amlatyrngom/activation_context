# Tressoir linear HTML starter

Copy `ARTIFACT.tressoir.html` into the existing broad-task folder under `IB/ARTIFACTS/`, rename it,
and replace the example content. Copy `SIBLING.md` only if you keep the example local link.

The starter is one compact, browser-portable HTML file. It loads exact KaTeX, PrismJS, and
CodeMirror releases from public CDNs and loads the reviewed Tressoir CSS/runtime from release tag
`v0.1.7`. Direct HTTPS scripts work in Tressoir and a normal browser. Preserve their order and do
not replace them with `latest`, mutable branches, or GitHub raw-file URLs that are served with a
non-JavaScript MIME type.

## Authoring rules

- Replace the example content; keep one linear reading flow.
- Keep the pinned core CSS and JavaScript references unchanged. Prefer a short inline `<style>` for
  narrow artifact-specific rules; use a sibling stylesheet when those rules become substantial.
- Do not add a page-type selector, whole-page card, or one-off palette.
- Use simple, concise language and keep ordinary information visible. A card summary must make sense while closed.
- Tables and SVG diagrams are automatically placed in `.scroll-region`; mark only tiny inline SVG
  icons with `data-tressoir-inline`. Wrap other wide widgets manually. Code blocks contain their own overflow.
- Use stable, artifact-specific interaction keys such as `migration.rollback_note`.
- Use one `.decision` per question. Do not combine several decisions into one control group.
- Keep the feedback dock as the final authored component. Markdown emits the same dock
  automatically; raw HTML keeps its explicit markup for ordinary browser fallback.

## Small component vocabulary

| Need | Markup |
|---|---|
| Document | `.tressoir-document`, `.document-header`, `.section`, `.lede`, `.meta` |
| One decision | `.decision` with one question, checkbox answers, `.decision-state`, and one `textarea[data-tressoir-autogrow="2:6"]` |
| Full optional section | `<details class="card" data-tressoir-markdown>` with the standard summary, `.card-body`, outer-edge `.card-gutter`, and selectable `.card-content` |
| Wide content | `.scroll-region`, `.table`, `.code-block`, `.math-display` |
| Stored control | `data-tressoir-input="stable.key"` plus a matching label |
| Local VS Code link | Relative `<a href>` with `data-tressoir-open-file` |
| Feedback while reading | The bundled floating **Feedback Form** dock with its `data-tressoir-feedback` textarea |

The card is a clickable header with a right-aligned badge. Its revealed section returns to the
ordinary page background. The full-height left gutter remains available when a long card's header
has scrolled away; it aligns with the header's outer edge and ends in a short hook and rule that
marks the section boundary without covering the separately inset, selectable content. The card has
no disclosure arrow and is not a static white content box. There is deliberately no separate
reveal component, default `item` row, or exclusive accordion.

## Decisions

A decision represents one question, not a list of unrelated approvals. Give every checkbox and the
single **Free Response** textarea its own stable `data-tressoir-input` key. The indicator begins
**Unresolved** and changes to **Resolved** as soon as at least one checkbox is selected; feedback
alone does not resolve the decision. The feedback textarea starts at two lines, grows with its
content through six lines, then scrolls internally. Keep `data-tressoir-autogrow="2:6"` and
`rows="2"` on that textarea.

## Tables and SVG diagrams

The runtime idempotently wraps document `<table>` and `<svg>` elements in a keyboard-focusable
`.scroll-region`. The region scrolls only when its content is wider than the document. Tiny inline
SVG icons can opt out with `data-tressoir-inline`.

## Math

Use `\(...\)` inline and `\[...\]` for display math. Dollar delimiters are not enabled because
they conflict with ordinary currency. KaTeX uses `trust: false`, `strict: "warn"`, and
`throwOnError: false`; malformed math stays readable and cannot stop later enhancements.

## Code and diffs

Use `<pre class="code-block"><code class="language-LANGUAGE">...</code></pre>`.

The pinned Prism pack supports markup/HTML, CSS, JavaScript, TypeScript, JSX, TSX, JSON, YAML,
Markdown, Bash, Python, Rust, SQL, and Diff. For language-aware diffs use:

```html
<pre class="code-block diff-highlight"><code class="language-diff-typescript">-const n = 1
+const n: number = 2</code></pre>
```

Unknown languages remain readable without highlighting.

For a multi-file change, present one clean linear file list. Each file is open by default,
independently collapsible, and linked by its relative path to the VS Code target. Keep open seams
between files, use a soft collapse-target hover/focus wash, weaken the hunk prelude, combine
language and diff highlighting, and contain overflow within each diff.

Use Low for an overview, Moderate for meaningful per-file hunks with explicit elisions, and
Detailed / Exact for the complete available delta. Moderate is the default for plans and ordinary
reviews. Interactive code suggestions default to Detailed / Exact non-cumulative slices: each
coarse slice is a delta on top of the preceding slice even when Git remains uncommitted.

## Reader settings and browser fallback

The root uses `font-size: var(--tressoir-base-font-size, 16px)`. In Tressoir, the
`tressoir.html.baseFontSize` setting controls the full rem-based scale; in a browser the fallback is
16px. Light and dark browser palettes follow `prefers-color-scheme`; VS Code palette variables take
precedence inside the extension.

Interactions persist silently through the extension's contained bridge. In a normal browser the
controls still work and links remain ordinary relative links, but values do not persist.

The small bottom-right feedback trigger opens a non-modal **Feedback Form** above the page. The
document remains readable and scrollable while it is open; use its close button or Escape to close
it. The form uses the pinned CodeMirror Markdown mode and persists silently through the
same contained interaction bridge without changing the stored plain Markdown string.

## Pinned remote dependencies

- KaTeX 0.18.4 from jsDelivr.
- PrismJS 1.30.0 from jsDelivr.
- CodeMirror 5.65.16 from cdnjs.
- Tressoir linear CSS/runtime from release `v0.1.7` through jsDelivr.

The template makes network requests for those resources. It uses no runtime loader,
package-manager call, `eval`, or `new Function`. For an explicitly offline deliverable, vendor the
same pinned files as a project-specific adaptation and document their provenance.
