"""Readable HTML diff of the paper between a git revision and the working tree (word-level, per paragraph, rename-aware,
changed figures rendered side by side). Plain standalone HTML (no tressoir), opens in any browser.

  UV_PYTHON_INSTALL_DIR=$HOME/.local/share/uv/python uv run --no-project --python 3.13 --with pymupdf \
      python make_diff.py [BASE_REV]          (default bec20a8 = the user's last full editing pass)
Output: diff_<BASE>_to_working.html next to this script.
"""
import base64
import difflib
import html
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pymupdf

PAPER = Path("/workspace/paper")
HERE = Path(__file__).resolve().parent
BASE = sys.argv[1] if len(sys.argv) > 1 else "bec20a8"
# new name -> old name at BASE (None = new file)
FILES = [("00_abstract.tex", "00_abstract.tex", "Abstract"),
         ("01_introduction.tex", "01_introduction.tex", "1 Introduction"),
         ("02_approach.tex", "03_approach.tex", "2 Activation Context (was 03_approach.tex)"),
         ("03_evaluation.tex", "04_evaluation.tex", "3 Evaluation (was 04_evaluation.tex)"),
         ("04_related_works.tex", "05_related_works.tex", "4 Related Work (was 05_related_works.tex)"),
         ("05_conclusion.tex", "06_conclusion.tex", "5 Conclusion (was 06_conclusion.tex)"),
         ("appendix.tex", None, "Appendix (new file, separate upload)"),
         ("main.tex", "main.tex", "main.tex"),
         ("references.bib", "references.bib", "references.bib")]
FIGS = ["fig_retrieval_v3", "fig_corpus_compression", "fig_compaction", "fig_multiagent", "fig_sec45_v4"]


def git_show(rev, path):
    r = subprocess.run(["git", "-C", str(PAPER), "show", f"{rev}:{path}"], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def paragraphs(text):
    return [p.strip("\n") for p in re.split(r"\n\s*\n", text) if p.strip()]


def tokens(s):
    return re.findall(r"\s+|[^\s]+", s)


def word_diff(a, b):
    ta, tb = tokens(a), tokens(b)
    out = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, ta, tb, autojunk=False).get_opcodes():
        x, y = html.escape("".join(ta[i1:i2])), html.escape("".join(tb[j1:j2]))
        if op == "equal":
            out.append(x)
        else:
            if x.strip():
                out.append(f"<del>{x}</del>")
            elif x:
                out.append(x)
            if y.strip():
                out.append(f"<ins>{y}</ins>")
    return "".join(out)


def align(old, new):
    """Yield ('same', o, n) / ('changed', o, n) / ('deleted', o, None) / ('added', None, n)."""
    sm = difflib.SequenceMatcher(None, old, new, autojunk=False)
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            for k in range(i2 - i1):
                yield "same", old[i1 + k], new[j1 + k]
            continue
        a, b = old[i1:i2], new[j1:j2]
        used = set()
        pairs = {}
        for bi, pb in enumerate(b):
            best, score = None, 0.35
            for ai, pa in enumerate(a):
                if ai in used:
                    continue
                r = difflib.SequenceMatcher(None, tokens(pa), tokens(pb), autojunk=False).ratio()
                if r > score:
                    best, score = ai, r
            if best is not None:
                used.add(best); pairs[bi] = best
        ai_next = 0
        for bi, pb in enumerate(b):
            if bi in pairs:
                for ai in range(ai_next, pairs[bi]):
                    if ai not in used:
                        yield "deleted", a[ai], None
                ai_next = max(ai_next, pairs[bi] + 1)
                yield "changed", a[pairs[bi]], pb
            else:
                yield "added", None, pb
        for ai in range(ai_next, len(a)):
            if ai not in used:
                yield "deleted", a[ai], None


def render_pdf(data, dpi=110):
    doc = pymupdf.open(stream=data, filetype="pdf")
    png = doc[0].get_pixmap(dpi=dpi).tobytes("png")
    return "data:image/png;base64," + base64.b64encode(png).decode()


def main():
    sections, toc = [], []
    total = {"changed": 0, "added": 0, "deleted": 0}
    for idx, (new_name, old_name, title) in enumerate(FILES):
        new_text = (PAPER / new_name).read_text()
        old_raw = git_show(BASE, old_name) if old_name else None
        old_text = old_raw.decode() if old_raw else ""
        counts = {"changed": 0, "added": 0, "deleted": 0}
        blocks = []
        if old_name is None:
            for p in paragraphs(new_text):
                blocks.append(f'<div class="para new-file">{html.escape(p)}</div>')
            counts["added"] = len(paragraphs(new_text))
        else:
            run = []
            def flush():
                if run:
                    inner = "".join(f'<div class="para">{html.escape(p)}</div>' for p in run)
                    blocks.append(f'<details class="same"><summary>{len(run)} unchanged paragraph{"s" if len(run) > 1 else ""}</summary>{inner}</details>')
                    run.clear()
            for kind, o, n in align(paragraphs(old_text), paragraphs(new_text)):
                if kind == "same":
                    run.append(n); continue
                flush(); counts[kind] += 1
                if kind == "changed":
                    blocks.append(f'<div class="para changed"><span class="tag">edited</span>{word_diff(o, n)}</div>')
                elif kind == "added":
                    blocks.append(f'<div class="para added"><span class="tag">added</span>{html.escape(n)}</div>')
                else:
                    blocks.append(f'<div class="para deleted"><span class="tag">removed</span>{html.escape(o)}</div>')
            flush()
        for k in total:
            total[k] += counts[k]
        n_changes = sum(counts.values())
        badge = "new" if old_name is None else (f"{n_changes}" if n_changes else "no change")
        toc.append(f'<a href="#s{idx}"><span>{html.escape(title)}</span><b class="{"zero" if not n_changes else ""}">{badge}</b></a>')
        sections.append(f'<section id="s{idx}"><h2>{html.escape(title)}</h2>'
                        f'<p class="meta">{counts["changed"]} edited · {counts["added"]} added · {counts["deleted"]} removed</p>'
                        + "".join(blocks) + "</section>")
    # figures
    fig_html = []
    for f in FIGS:
        old = git_show(BASE, f"figures/{f}.pdf"); new = (PAPER / "figures" / f"{f}.pdf").read_bytes()
        if old == new:
            continue
        fig_html.append(f'<div class="fig"><h3>{f}.pdf</h3><div class="pair">'
                        f'<figure><figcaption>Before ({BASE})</figcaption><img src="{render_pdf(old)}"></figure>'
                        f'<figure><figcaption>Now</figcaption><img src="{render_pdf(new)}"></figure></div></div>')
    toc.append(f'<a href="#figs"><span>Figures</span><b>{len(fig_html)}</b></a>')
    sections.append('<section id="figs"><h2>Figures</h2><p class="meta">Changed figure files, before and after.</p>' + "".join(fig_html) + "</section>")
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    page = TEMPLATE.format(base=BASE, stamp=stamp, toc="".join(toc), body="".join(sections), **total)
    out = HERE / f"diff_{BASE}_to_working.html"
    out.write_text(page)
    print(out, f"{len(page) / 1e6:.1f} MB", total)


TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Paper Changes Since {base}</title>
<style>
:root {{ --bg:#fbfaf7; --panel:#ffffff; --ink:#1d1d1f; --muted:#6b6b70; --line:#e6e3dc; --del-bg:#fde8e6; --del-ink:#a1261b;
  --ins-bg:#e3f4e6; --ins-ink:#16602b; --accent:#b8441f; --tag:#f1ede4; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#161616; --panel:#1f1f1f; --ink:#e9e7e2; --muted:#a09d96;
  --line:#333; --del-bg:#4a1f1b; --del-ink:#ffb4aa; --ins-bg:#173a22; --ins-ink:#a6e3b4; --accent:#f08a5d; --tag:#2b2925; }} }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--ink); font:16px/1.65 Georgia, "Times New Roman", serif; }}
header {{ padding:28px 24px 18px; border-bottom:1px solid var(--line); background:var(--panel); }}
header h1 {{ margin:0 0 4px; font:600 22px/1.3 -apple-system, "Segoe UI", sans-serif; }}
header p {{ margin:0; color:var(--muted); font:14px/1.5 -apple-system, "Segoe UI", sans-serif; }}
.legend {{ margin-top:10px; display:flex; gap:14px; flex-wrap:wrap; font:13px -apple-system, "Segoe UI", sans-serif; color:var(--muted); }}
.wrap {{ display:grid; grid-template-columns:260px minmax(0,1fr); max-width:1280px; margin:0 auto; }}
nav {{ position:sticky; top:0; align-self:start; height:100vh; overflow:auto; padding:20px 12px; border-right:1px solid var(--line);
  font:14px -apple-system, "Segoe UI", sans-serif; }}
nav a {{ display:flex; justify-content:space-between; gap:8px; padding:7px 10px; border-radius:6px; color:var(--ink); text-decoration:none; }}
nav a:hover {{ background:var(--tag); }}
nav b {{ font-weight:600; color:var(--accent); }} nav b.zero {{ color:var(--muted); font-weight:400; }}
nav .tools {{ margin-top:16px; padding:0 10px; }}
nav button {{ font:13px -apple-system, "Segoe UI", sans-serif; padding:6px 10px; border:1px solid var(--line); background:var(--panel);
  color:var(--ink); border-radius:6px; cursor:pointer; margin:0 6px 6px 0; }}
main {{ padding:8px 32px 80px; min-width:0; }}
section {{ padding-top:18px; }}
h2 {{ font:600 19px/1.3 -apple-system, "Segoe UI", sans-serif; margin:22px 0 2px; }}
h3 {{ font:600 15px -apple-system, "Segoe UI", sans-serif; margin:18px 0 8px; }}
.meta {{ color:var(--muted); font:13px -apple-system, "Segoe UI", sans-serif; margin:0 0 12px; }}
.para {{ background:var(--panel); border:1px solid var(--line); border-left:4px solid var(--line); border-radius:6px;
  padding:12px 16px; margin:10px 0; white-space:pre-wrap; overflow-wrap:anywhere; }}
.para.changed {{ border-left-color:var(--accent); }}
.para.added, .para.new-file {{ border-left-color:var(--ins-ink); }}
.para.deleted {{ border-left-color:var(--del-ink); color:var(--del-ink); text-decoration:line-through; opacity:.8; }}
.tag {{ display:inline-block; font:600 11px -apple-system, "Segoe UI", sans-serif; text-transform:uppercase; letter-spacing:.04em;
  color:var(--muted); background:var(--tag); border-radius:4px; padding:1px 6px; margin-right:8px; text-decoration:none; }}
del {{ background:var(--del-bg); color:var(--del-ink); text-decoration:line-through; border-radius:3px; padding:0 1px; }}
ins {{ background:var(--ins-bg); color:var(--ins-ink); text-decoration:none; border-radius:3px; padding:0 1px; }}
body.only-new del {{ display:none; }}
body.only-new ins {{ background:none; color:inherit; }}
details.same > summary {{ cursor:pointer; color:var(--muted); font:13px -apple-system, "Segoe UI", sans-serif; padding:6px 2px; }}
details.same .para {{ color:var(--muted); }}
.pair {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; }}
figure {{ margin:0; background:#fff; border:1px solid var(--line); border-radius:6px; padding:10px; }}
figcaption {{ font:12px -apple-system, "Segoe UI", sans-serif; color:#666; margin-bottom:6px; }}
figure img {{ width:100%; height:auto; display:block; }}
@media (max-width: 860px) {{ .wrap {{ grid-template-columns:1fr; }} nav {{ position:static; height:auto; border-right:0; border-bottom:1px solid var(--line); }}
  main {{ padding:8px 16px 60px; }} .pair {{ grid-template-columns:1fr; }} }}
</style></head>
<body>
<header><h1>Paper Changes Since {base}</h1>
<p>From your last full editing pass ({base}) to the current working tree · generated {stamp} · {changed} paragraphs edited, {added} added, {deleted} removed. Renamed files are matched to their old names.</p>
<div class="legend"><span><del>removed text</del></span><span><ins>added text</ins></span><span>Unchanged paragraphs are folded; click to expand.</span></div>
</header>
<div class="wrap"><nav>{toc}<div class="tools">
<button onclick="document.querySelectorAll('details.same').forEach(d=>d.open=true)">Expand unchanged</button>
<button onclick="document.querySelectorAll('details.same').forEach(d=>d.open=false)">Fold unchanged</button>
<button onclick="document.body.classList.toggle('only-new')">Toggle clean view</button>
</div></nav>
<main>{body}</main></div>
</body></html>
"""

if __name__ == "__main__":
    main()
