# Structured AC research

Open [RESEARCH.tressoir.html](RESEARCH.tressoir.html) for the brainstorming document.

It covers six research directions, programs composed exclusively of structured model calls, the distinction between scratch slots and recurrent computation, a proposed `structured_ac_model.py` interface grounded in current source, training and evaluation controls, and an interactive illustrative latency calculator. Primary research sources are linked beside the relevant claims.

This is a research proposal. The APIs and programs are pseudocode; no model implementation, training configuration, or experiment was changed. The recommendation is not an accepted project decision.

The HTML retains the project's pinned Tressoir, KaTeX, Prism, and CodeMirror resources. It needs network access for those resources; the explanatory HTML is present without JavaScript. Calculator controls work in an ordinary browser, with silent interaction persistence when opened through the Tressoir helper. The standard feedback dock is preserved.

Validation on 21 September 2026: HTML IDs and internal anchors checked; inline JavaScript syntax checked; Chromium rendered all external resources without request failures or script errors. Desktop light/dark and a 390-pixel mobile viewport had no document-level horizontal overflow. Exercised calculator presets and negative net head overhead, disclosure open/collapse, feedback open/Escape, and render-event input preservation. Interaction writes and the sibling link were checked against a mock Tressoir helper.

The actual VS Code custom editor, its content security policy, disk persistence, external-edit DOM morphing, and traversal rejection were not available for verification. Browser checks do not establish those integration properties. Disposable scripts, screenshots, and validation results are in `IB/TMP/STRUCTURED_AC_RESEARCH/`.
