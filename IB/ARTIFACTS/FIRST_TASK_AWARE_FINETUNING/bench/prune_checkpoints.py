"""Agent-operated checkpoint retention, never an unattended deletion daemon.

Inventory explicit SYNC_RUNS-like roots; inspect the JSON; execute that exact
manifest. Groups are whole run AC_MODELS + LORAS trees, so paired checkpoints
cannot be separated. Reports and other run files stay in place. PRESERVE at any
ancestor or anywhere in either member protects the group. Keep at least the
newest 250 GiB of *unflagged* checkpoint groups, in addition to flagged groups.
Run inventories and deletion on Slurm compute nodes, never MIT login nodes.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import time


def inspect_group(run):
    run = Path(run).resolve(strict=True)
    members = [run / name for name in ("AC_MODELS", "LORAS")
               if (run / name).is_dir() and not (run / name).is_symlink()]
    if not members:
        return None
    flags = [str(p / "PRESERVE") for p in (run, *run.parents)
             if (p / "PRESERVE").exists()]
    entries, unique, size, physical, newest = [], set(), 0, 0, 0
    for member in members:
        for root, dirs, files in os.walk(member, followlinks=False):
            # Symlinked directories are themselves removed, never traversed.
            for name in [".", *dirs, *files]:
                path = Path(root) if name == "." else Path(root) / name
                st = path.lstat()
                rel = str(path.relative_to(run))
                entries.append((rel, st.st_mode, st.st_size, st.st_mtime_ns,
                                st.st_ctime_ns, st.st_dev, st.st_ino))
                newest = max(newest, st.st_mtime_ns)
                if name == "PRESERVE":
                    flags.append(str(path))
                if stat.S_ISREG(st.st_mode) and (st.st_dev, st.st_ino) not in unique:
                    unique.add((st.st_dev, st.st_ino))
                    size += st.st_size
                    physical += st.st_blocks * 512
    digest = hashlib.sha256(json.dumps(sorted(entries)).encode()).hexdigest()
    return dict(run=str(run), members=[str(p) for p in members],
                bytes=size, allocated_bytes=physical, newest_ns=newest,
                preserve=sorted(set(flags)), fingerprint=digest)


def inventory(roots, keep_gib):
    roots = [Path(p).resolve(strict=True) for p in roots]
    if len({str(p) for p in roots}) != len(roots):
        raise ValueError("Duplicate roots")
    groups = []
    for root in roots:
        for child in sorted(root.iterdir()):
            if child.is_dir() and not child.is_symlink():
                group = inspect_group(child)
                if group:
                    groups.append(group)
    retained = 0
    for group in sorted(groups, key=lambda g: g["newest_ns"], reverse=True):
        if group["preserve"]:
            group["decision"] = "preserve_flag"
        elif retained < keep_gib * 2**30:
            group["decision"] = "keep_recent"
            retained += group["bytes"]
        else:
            group["decision"] = "candidate"
    return dict(version=1, created_unix=time.time(), roots=[str(p) for p in roots],
                keep_gib=keep_gib, recent_retained_bytes=retained, groups=groups)


def execute(manifest, ledger):
    # Recompute the entire protection boundary immediately before deletion.
    fresh = inventory(manifest["roots"], manifest["keep_gib"])
    by_run = {g["run"]: g for g in fresh["groups"]}
    candidates = [g for g in manifest["groups"] if g["decision"] == "candidate"]
    for old in candidates:
        now = by_run.get(old["run"])
        if (not now or now["decision"] != "candidate"
                or now["fingerprint"] != old["fingerprint"]
                or now["members"] != old["members"]):
            raise RuntimeError("Inventory changed; re-inventory before deletion: " + old["run"])
    with Path(ledger).open("a") as out:
        for old in candidates:
            # Recheck flags/fingerprint at the last possible moment. The caller
            # must separately confirm these historical runs are inactive.
            now = inspect_group(old["run"])
            if now["preserve"] or now["fingerprint"] != old["fingerprint"]:
                raise RuntimeError("Group changed before deletion: " + old["run"])
            event = dict(time_unix=time.time(), run=old["run"],
                         members=old["members"], bytes=old["bytes"],
                         allocated_bytes_estimate=old["allocated_bytes"])
            out.write(json.dumps(dict(event, action="begin")) + "\n")
            out.flush()
            os.fsync(out.fileno())
            for member in old["members"]:
                path = Path(member)
                if path.parent != Path(old["run"]) or path.name not in ("AC_MODELS", "LORAS"):
                    raise ValueError("Unsafe member path")
                shutil.rmtree(path)
            out.write(json.dumps(dict(event, action="deleted")) + "\n")
            out.flush()
            os.fsync(out.fileno())
            print(json.dumps(event), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", default=[])
    parser.add_argument("--keep-gib", type=float, default=250)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--ledger")
    args = parser.parse_args()
    if args.execute:
        if not args.ledger:
            parser.error("--execute requires --ledger")
        execute(json.loads(Path(args.manifest).read_text()), args.ledger)
    else:
        if not args.root:
            parser.error("Inventory requires explicit --root")
        result = inventory(args.root, args.keep_gib)
        Path(args.manifest).write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({"groups": len(result["groups"]),
                          "recent_retained_gib": result["recent_retained_bytes"] / 2**30,
                          "candidate_gib": sum(g["bytes"] for g in result["groups"]
                                               if g["decision"] == "candidate") / 2**30}))


if __name__ == "__main__":
    main()
