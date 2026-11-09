#!/usr/bin/env bash
# Compile the paper (submodule paper/) with pdfLaTeX + BibTeX via latexmk, the same engine as Overleaf.
# Output: paper/main.pdf and paper/appendix.pdf (the separately submitted appendix, built from paper/appendix.tex). Intermediate files go to
# IB/ARTIFACTS/PAPER_WRITING/build/, so nothing else is written inside paper/.
# TeX comes from TinyTeX (user-level, ~/.TinyTeX); missing LaTeX packages are installed on demand.
# Do not use Tectonic/XeTeX here: the template's Times setup silently drops to regular-weight
# Latin Modern under XeTeX (no bold, no italics).
# Usage: IB/ARTIFACTS/PAPER_WRITING/build_pdf.sh            (from /workspace)
#        IB/ARTIFACTS/PAPER_WRITING/build_pdf.sh --pages    (also write page PNGs to build/pages/)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
OUT="$ROOT/IB/ARTIFACTS/PAPER_WRITING/build"
TEXBIN="$HOME/.TinyTeX/bin/x86_64-linux"
if [ ! -x "$TEXBIN/latexmk" ]; then
  curl -sSL https://yihui.org/tinytex/install-bin-unix.sh | sh
  "$TEXBIN/tlmgr" update --self
fi
export PATH="$TEXBIN:$PATH"
mkdir -p "$OUT"
cd "$ROOT/paper"
for _ in $(seq 1 20); do
  latexmk -g -pdf -interaction=nonstopmode -halt-on-error -outdir="$OUT" main.tex >/dev/null 2>&1 || true
  miss=$(grep -oE "File \`[^']+' not found" "$OUT/main.log" | head -1 | sed -E "s/File \`([^']+)' not found/\1/" || true)
  [ -z "$miss" ] && break
  pkg=$(tlmgr search --global --file "/$miss" 2>/dev/null | grep -E "^[a-z0-9.-]+:$" | head -1 | tr -d ':' || true)
  [ -z "$pkg" ] && { echo "missing $miss and no TeX Live package provides it"; break; }
  echo "installing $pkg (for $miss)"
  tlmgr install "$pkg" >/dev/null
done
grep -E "^! |LaTeX Warning: (Reference|Citation)" "$OUT/main.log" || true
grep -E "Output written" "$OUT/main.log"
cp "$OUT/main.pdf" "$ROOT/paper/main.pdf"
echo "PDF: $ROOT/paper/main.pdf"
if [ -f appendix.tex ]; then
  latexmk -g -pdf -interaction=nonstopmode -halt-on-error -outdir="$OUT" appendix.tex >/dev/null 2>&1 || true
  grep -E "^! |LaTeX Warning: (Reference|Citation)" "$OUT/appendix.log" || true
  grep -E "Output written" "$OUT/appendix.log"
  cp "$OUT/appendix.pdf" "$ROOT/paper/appendix.pdf"
  echo "PDF: $ROOT/paper/appendix.pdf"
fi
if [ "${1:-}" = "--pages" ]; then
  # the container's default (/opt/uv/python) is root-owned; fall back to a user directory
  if [ ! -w "${UV_PYTHON_INSTALL_DIR:-/nonexistent}" ]; then export UV_PYTHON_INSTALL_DIR="$HOME/.uv/python"; fi
  uv run --no-project --python 3.13 --with pymupdf python - "$OUT" <<'EOF'
import sys, pathlib, pymupdf
out = pathlib.Path(sys.argv[1]); (out / "pages").mkdir(exist_ok=True)
for old in (out / "pages").glob("page_*.png"):
    old.unlink()
doc = pymupdf.open(out / "main.pdf")
for i, page in enumerate(doc):
    page.get_pixmap(dpi=110).save(out / "pages" / f"page_{i + 1:02d}.png")
print(f"{len(doc)} pages -> {out / 'pages'}")
EOF
fi
