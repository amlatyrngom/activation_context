# Extension handoff: stable live training reports

The report currently jumps upward during updates. There is a confirmed report-side defect: `activation/common/reporting.py` purges Plotly plots and replaces all widget nodes whenever JSON changes or `tressoir:render` fires. We will fix that locally. HTML is also rewritten approximately every 10 seconds. These facts do not yet establish an extension bug.

The extension work is to verify and, where necessary, fix the following contract. No GPU is needed to reproduce it.

| Boundary | Required behavior |
| --- | --- |
| Sibling JSON | An HTML artifact can reliably read its adjacent `report_data.json`, including subsequent atomic replacements. Use the existing local-resource mechanism; document the URL resolution or data-delivery API the report should use. |
| Ordinary external HTML edits | Update the existing view without a full navigation/realm reset when compatible with the artifact's structure. Preserve the reading position and user control state. |
| Runtime-owned subtrees | Respect the existing `data-morph-skip`/state-preservation contract. A benign source edit must not wipe content created by the report's JavaScript. |
| Render notification | Document when `tressoir:render` fires and on which target. An update must not cause repeated script initialization or accumulate handlers/timers. The report will make its own handler idempotent. |
| Failures and reloads | Make unavailable local resources diagnosable. A deliberate full reload is distinct from a normal data refresh; it should not happen silently on each data change. |

Reproduce with a long artifact containing stable sections, a disclosure, a text input and a runtime-created plot or equivalent fixed-height widget. Scroll well below the top, open the disclosure, focus/type in the input and change the plot view. Then test these paths separately:

1. Atomically replace only sibling JSON several times. HTML bytes stay unchanged. Data should update without a navigation, scroll jump or duplicate polling loop.
2. Make a benign external HTML edit outside the runtime-owned region. Observe the morph event and verify scroll, focus, input value, disclosure state and runtime content survive.
3. Add or remove a section deliberately. Keep a visible content anchor stable where possible; distinguish expected layout movement from a jump to the top.
4. Deliver the render event repeatedly. Initialization/listener counts should remain stable.
5. Temporarily make sibling JSON unavailable, then restore it. The artifact should expose the failure and recover through the documented mechanism, without repeated full reloads.

Run the equivalent fixture in a normal browser as a control. If only the extension fails, inspect its resource/lifecycle behavior; if both fail, inspect the artifact renderer. The current production report is not a clean extension-only reproducer because of its destructive rebuild.

When returning with the reinstalled extension, include its version and the JSON access mechanism, morph-event timing/target, runtime-owned subtree behavior, and which checks passed. We will then verify the corrected actual training report in that installation before planning another long GPU run.

Our local work remains: stable keyed widget updates, Plotly updates that preserve interaction state, a single polling/render lifecycle, bounded explicit HTML fallback, clear phase counters, and real evaluation curves. The extension does not need to understand training epochs, KL references, or report-specific metrics.
