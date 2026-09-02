# LoRA engine explainer

LORA_EXPLAINER.tressoir.html answers the inline questions from
activation/tests/test_basic_engine.py about model lifecycle, request-specific LoRA routing, and
concurrency.

The folder is based on the standard Tressoir linear HTML starter. tressoir-linear.css,
tressoir-linear.js, and vendor/ are copied without modification. Artifact-specific styling is
isolated in lora-explainer.css.

Open the HTML file with the Tressoir Artifacts custom editor. It also works as an offline browser
page; only the primary-reference links require network access.

## Version-control note

The existing IB/.gitignore ignores every vendor/ directory. The complete offline bundle is present
in the shared workspace, but Git will omit those dependencies unless the project adds a narrow
exception for this artifact. This task deliberately leaves the existing ignore policy unchanged.
