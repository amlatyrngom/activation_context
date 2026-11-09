"""Oct 30 simplification pass: make_diff.py with the current file names (no renames since a77931d), the new AI-disclosure file,
and a notes panel saying why each remaining change is there.

  UV_PYTHON_INSTALL_DIR=$HOME/.local/share/uv/python uv run --no-project --python 3.13 --with pymupdf \
      python make_diff_v2.py a77931d
Output: diff_a77931d_to_working.html next to this script.
"""
import html
import sys

import make_diff as M

M.FILES = [("00_abstract.tex", "00_abstract.tex", "Abstract"),
           ("01_introduction.tex", "01_introduction.tex", "1 Introduction"),
           ("02_approach.tex", "02_approach.tex", "2 Activation Context"),
           ("03_evaluation.tex", "03_evaluation.tex", "3 Evaluation"),
           ("04_related_works.tex", "04_related_works.tex", "4 Related Work"),
           ("05_conclusion.tex", "05_conclusion.tex", "5 Conclusion"),
           ("06_ai_disclosure.tex", None, "AI Usage Disclosure (new file, after the references)"),
           ("appendix.tex", "appendix.tex", "Appendix"),
           ("main.tex", "main.tex", "main.tex"),
           ("references.bib", "references.bib", "references.bib")]

NOTES = [
    ("Abstract", "Kept as you approved: 'matches' text accuracy for retrieval, the corpus and multi-agent wording, plus the "
                 "'Activation Contexts:' punctuation and 'their ability'.", "you approved"),
    ("Introduction", "Your text, except that Flamingo is removed from the 'attaches embeddings to the input stream' citation "
                     "list (Flamingo uses cross-attention, as your Related Work says).", "citation error"),
    ("Approach", "New sentence after 'Non-uniform KVs give a better representation...': they also require an additional "
                 "per-entry attention bias (not shown in ACWrite). Two typos. LoRA citation.", "you approved / typo / citation"),
    ("Evaluation, start", "'We now evaluate ACs.' replaces the longer opening sentence (page fit).", "you approved"),
    ("Evaluation, retrieval", "'Following [IRCoT], we collect...' is now 'We collect...'. IRCoT used HotpotQA's own Wikipedia "
                              "corpus.", "you approved"),
    ("Evaluation, corpus", "Baselines: rerunning AM's released code and reusing the Cartridges results reported by AM (the "
                           "appendix already said so).", "you approved"),
    ("Evaluation, multi-agent", "Synthesis gap 7% is now 6.6% (94.1 - 87.5).", "you approved"),
    ("Evaluation, all", "golden to gold, pareto to Pareto, Firstly to First, 'almost near' to 'nearly at', Qwen names as "
                        "Qwen3.5-9B etc. Citations added for BM25, Qwen3, Long Code Arena, DeepMath-103K and FineWeb-Edu.",
                        "small fix / citation"),
    ("Related Work", "LatentMAS moved from 'last-layer hidden states' to 'KV caches': it passes KV caches between agents.",
                     "citation error"),
    ("Conclusion", "Your text; no change.", "-"),
    ("AI Usage Disclosure", "New section after the references (MLSys 2027 rule; outside the page limit). Your rewrite "
                            "still to come.", "MLSys rule"),
    ("main.tex", "Author notes off (\\showauthornotesfalse); the disclosure is input after the bibliography.", "submission"),
    ("Appendix", "Title now Harness/Model. The softened 'structural changes' sentence. 'r = 0.1' becomes '0.1' in the "
                 "compression row (r is also the LoRA rank there). Everything else is your text.", "you approved / small fix"),
    ("references.bib", "Titles brace-protected (the style was lowercasing them). 'arXiv preprint' shown on preprints. Published "
                       "venues for 15 entries. Tressoir CAIS pages and DOI. Qwen3 report with its full author list. Six new "
                       "entries for the citations above.", "reference errors"),
    ("Figure 3", "The 'Select -> text' label no longer overlaps the k labels.", "you approved"),
]


def notes_html():
    rows = "".join(f"<tr><td>{html.escape(w)}</td><td>{html.escape(t)}</td><td class='why'>{html.escape(k)}</td></tr>"
                   for w, t, k in NOTES)
    return ("<section id='notes'><h2>What is left, and why</h2><p class='meta'>Everything else is back to your text at "
            "a77931d .</p>"
            "<table class='notes'><thead><tr><th>Where</th><th>Change</th><th>Reason</th></tr></thead><tbody>"
            + rows + "</tbody></table></section>").replace("{", "{{").replace("}", "}}")


STYLE = """table.notes {{ width:100%; border-collapse:collapse; font:14px/1.5 -apple-system, "Segoe UI", sans-serif; background:var(--panel); }}
table.notes th, table.notes td {{ text-align:left; vertical-align:top; padding:8px 10px; border-bottom:1px solid var(--line); }}
table.notes th {{ color:var(--muted); font-weight:600; }} table.notes td:first-child {{ white-space:nowrap; font-weight:600; }}
table.notes td.why {{ color:var(--accent); white-space:nowrap; }}
@media (max-width: 860px) {{ table.notes td:first-child, table.notes td.why {{ white-space:normal; }} }}
</style>"""

M.TEMPLATE = (M.TEMPLATE.replace("</style>", STYLE, 1)
              .replace("<main>{body}</main>", "<main>" + notes_html() + "{body}</main>")
              .replace("{toc}", '<a href="#notes"><span>What is left, and why</span><b>' + str(len(NOTES)) + "</b></a>{toc}")
              .replace(" Renamed files are matched to their old names.", ""))

if __name__ == "__main__":
    M.main()
