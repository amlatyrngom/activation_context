"""
sha256 of a saved pair folder (DIR/ac/** and DIR/reader/**: relative paths and bytes, sorted; meta.json, DONE and README.md are
not part of it). Standard library only, so it runs with any python3 (job node, login node, this container).

  python pairhash.py DIR              prints the hash
  python pairhash.py DIR --check DONE exit 0 when DIR's hash equals DONE's "final_sha256", else exit 4 (prints both)
"""
import hashlib
import json
import sys
from pathlib import Path


def pair_hash(folder) -> str:
    folder = Path(folder)
    digest = hashlib.sha256()
    files = sorted(p for part in ("ac", "reader") for p in (folder / part).rglob("*") if p.is_file() and p.name != "README.md")
    if not any(p.name == "ac_modules.pt" for p in files) or not any(p.parent.name == "reader" and p.suffix == ".safetensors" for p in files):
        raise SystemExit(f"pairhash: {folder} is not a complete pair (ac/ac_modules.pt and reader/*.safetensors)")
    for path in files:
        digest.update(str(path.relative_to(folder)).encode() + b"\0")
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 24), b""):
                digest.update(block)
        digest.update(b"\0")
    return digest.hexdigest()


def main():
    if len(sys.argv) not in (2, 4) or (len(sys.argv) == 4 and sys.argv[2] != "--check"):
        raise SystemExit(__doc__)
    value = pair_hash(sys.argv[1])
    if len(sys.argv) == 2:
        print(value)
        return
    expected = json.loads(Path(sys.argv[3]).read_text()).get("final_sha256")
    print(f"pair {value}\nDONE {expected}")
    sys.exit(0 if value == expected else 4)


if __name__ == "__main__":
    main()
