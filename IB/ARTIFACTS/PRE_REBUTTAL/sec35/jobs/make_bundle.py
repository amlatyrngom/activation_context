"""
The code bundle the section 4.5 jobs run: one tar.gz of exactly the files they import, taken from THIS working tree (the
`activation` package as it is now, including uncommitted changes; the kit; the two shared kits it calls into), plus a manifest
with each file's sha256 and the git HEAD it sits on. Uploaded next to the data so every cluster runs the same bytes.

  bundle layout          activation/ (no tests/)  pyproject.toml  uv.lock
                         kit/       IB/TMP/PAPER_SEC45/kit/*.py
                         rq_kit/    rq_common.py rq_train.py rq_teacher.py   (PAPER_RETRIEVAL_QA/kit; read only, unchanged)
                         bench/     run_qa.py run_qa_9b.py                   (PAPER_CORPUS_QA/kit/bench; read only, unchanged)
                         MANIFEST.json

  python make_bundle.py --out BUNDLE.tar.gz [--upload s3://.../code/]
"""
import argparse
import hashlib
import io
import json
import subprocess
import sys
import tarfile
import time
from pathlib import Path

SEC45 = Path(__file__).resolve().parents[1]           # IB/ARTIFACTS/PRE_REBUTTAL/sec35
REPO = SEC45.parents[3]
FILES = {"kit": sorted((SEC45 / "kit").glob("*.py")), "text_lora": sorted((SEC45 / "text_lora").glob("*.py")),
         "rq_kit": [REPO / "IB/TMP/PAPER_RETRIEVAL_QA/kit" / n for n in ("rq_common.py", "rq_train.py", "rq_teacher.py")],
         "bench": [REPO / "IB/TMP/PAPER_CORPUS_QA/kit/bench" / n for n in ("run_qa.py", "run_qa_9b.py")]}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True); p.add_argument("--upload")
    a = p.parse_args()
    if a.out.exists():
        raise SystemExit(f"{a.out} exists (never replaced)")
    members = [(path, str(path.relative_to(REPO))) for path in sorted((REPO / "activation").rglob("*"))
               if path.is_file() and "__pycache__" not in path.parts and "tests" not in path.relative_to(REPO / "activation").parts[:1] and path.suffix != ".pyc"]
    members += [(REPO / "pyproject.toml", "pyproject.toml"), (REPO / "uv.lock", "uv.lock")]
    members += [(path, f"{folder}/{path.name}") for folder, paths in FILES.items() for path in paths]
    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "activation"], capture_output=True, text=True).stdout.splitlines()
    manifest = {"created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "git_head": head, "activation_uncommitted": dirty,
                "files": {name: hashlib.sha256(path.read_bytes()).hexdigest() for path, name in members}}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(a.out, "w:gz") as tar:
        for path, name in members:
            tar.add(path, arcname=name)
        data = json.dumps(manifest, indent=1).encode()
        info = tarfile.TarInfo("MANIFEST.json"); info.size = len(data); info.mtime = int(time.time())
        tar.addfile(info, io.BytesIO(data))
    digest = hashlib.sha256(a.out.read_bytes()).hexdigest()
    a.out.with_name(a.out.name + ".sha256").write_text(f"{digest}  {a.out.name}\n")
    print(json.dumps({"bundle": str(a.out), "files": len(members), "bytes": a.out.stat().st_size, "sha256": digest, "git_head": head,
                      "activation_uncommitted_files": len(dirty)}))
    if a.upload:
        for path in (a.out, a.out.with_name(a.out.name + ".sha256")):
            subprocess.check_call([sys.executable, str(SEC45 / "kit/s3x.py"), "put", str(path), a.upload.rstrip("/") + "/" + path.name])


if __name__ == "__main__":
    main()
