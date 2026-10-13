"""Merge raw corpora into nine explicit synthesis families without repetition.

Reads plain source data, never a study cache. Output count is raw source windows,
not an accepted-question guarantee. Office mixture stops at the limiting source
type instead of silently filling the rest with API documentation.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
import random


FAMILIES = ("code", "math_arxiv", "math_textbook", "financial", "web", "technical_web", "documents", "legal", "office")
SHARES = dict(zip(FAMILIES, (.30, .12, .08, .05, .20, .05, .08, .06, .06)))
SEED = 20260922


def sha_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for part in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    root = Path("IB/TMP/FIRST_TASK_AWARE_FINETUNING")
    parser.add_argument("--existing", type=Path, default=root / "bank_sources_100k.json")
    parser.add_argument("--extra", type=Path, default=root / "expanded_sources")
    parser.add_argument("--output", type=Path, default=root / "bank_sources_200k.json")
    args = parser.parse_args()
    pools = {family: [] for family in FAMILIES}
    seen, excluded = set(), collections.Counter()

    def add(family, row):
        ident = row["source_id"]
        if ident in seen:
            excluded["exact_repeated_text"] += 1
            return
        seen.add(ident)
        if "domain" in row and row["domain"] != family:
            row["source_domain"] = row["domain"]
        row["domain"] = family
        pools[family].append(row)

    existing_sha = sha_file(args.existing)
    with args.existing.open() as file:
        old = json.load(file)
    old_counts = {key: len(rows) for key, rows in old["pools"].items()}
    for domain, rows in old["pools"].items():
        for row in rows:
            kind = row["source_kind"]
            if domain == "code":
                family = "code"
            elif kind == "arxiv_paper_window":
                if sum(row["text"].count(char) for char in ("=", "\\", "^", "_")) < 20:
                    excluded["arxiv_without_mathematical_markup"] += 1
                    continue
                family = "math_arxiv"
            elif domain == "numeric":
                family = "financial"
            elif kind == "technical_post" or row["doc_id"].startswith("bright/"):
                family = "technical_web"
            elif domain == "web":
                family = "web"
            elif kind == "document_window":
                # Old document-window entries here are Loong research papers.
                # Scientific exposition must not inflate the manuals/rules pool.
                excluded["loong_scientific_paper_windows"] += 1
                continue
            else:
                family = "documents"
            row.setdefault("provenance", {"legacy_source_bank": args.existing.name,
                                           "source_bank_sha256": existing_sha})
            add(family, row)
    del old

    extra_manifest = json.loads((args.extra / "manifest.json").read_text())
    file_map = {"math_libretexts.jsonl": "math_textbook", "law.jsonl": "legal",
                "technical_peps.jsonl": "documents", "technical_repository_docs.jsonl": "documents"}
    for filename, family in file_map.items():
        with (args.extra / filename).open() as file:
            for line in file:
                add(family, json.loads(line))

    office = {}
    for kind, filename in {"automation": "office_automation.jsonl", "business": "office_business.jsonl",
                           "manuals": "office_libreoffice.jsonl"}.items():
        with (args.extra / filename).open() as file:
            office[kind] = [json.loads(line) for line in file]
        random.Random(f"{SEED}:office:{kind}").shuffle(office[kind])
    # Each complete ten-window unit has 6 automation, 3 business, 1 procedural.
    office_pattern = ("automation", "business", "automation", "manuals", "automation",
                      "business", "automation", "automation", "business", "automation")
    office_source_counts = {key: len(rows) for key, rows in office.items()}
    office_rounds = min(len(office["automation"]) // 6, len(office["business"]) // 3, len(office["manuals"]))
    office_used = collections.Counter()
    for _ in range(office_rounds):
        for kind in office_pattern:
            row = office[kind][office_used[kind]]
            office_used[kind] += 1
            add("office", row)

    # Shuffle each major family independently. Keep office's deterministic blend
    # order: consecutive 10-window selections should retain its internal mixture.
    for family, rows in pools.items():
        if family != "office":
            random.Random(f"{SEED}:{family}").shuffle(rows)
    counts = {family: len(rows) for family, rows in pools.items()}
    before_rejection = {family: int(counts[family] / SHARES[family]) for family in FAMILIES}
    summary = {"seed": SEED, "counts": counts, "source_shares_for_capacity_calculation": SHARES,
               "raw_one_question_total_capacity_by_family_before_rejection": before_rejection,
               "limiting_raw_one_question_family": min(before_rejection, key=before_rejection.get),
               "note": "Raw-window capacity only: acceptance rejects some questions; long windows may yield more than one. No source is repeated to reach the 200k request.",
               "existing_source_sha256": existing_sha, "existing_source_counts": old_counts,
               "excluded": dict(excluded), "office": {"raw_counts": office_source_counts, "used": dict(office_used),
               "unused": {key: office_source_counts[key] - office_used[key] for key in office},
               "limiting_pool": min(office_source_counts, key=lambda key: office_source_counts[key] / {"automation": .6, "business": .3, "manuals": .1}[key])},
               "extra_source_manifest": extra_manifest}
    payload = {"seed": SEED, "pools": pools,
               "note": "Raw corpora only; no native benchmark QA. Nine explicit source families. Office windows interleave 60% automation, 30% business/communication, 10% procedural help until the limiting pool is exhausted. Plain source files remain reusable across synthesis recipes.",
               "source_summary": {key: value for key, value in summary.items() if key != "extra_source_manifest"}}
    temporary = args.output.with_suffix(".json.partial")
    with temporary.open("w") as file:
        json.dump(payload, file, ensure_ascii=False, separators=(",", ":"))
    temporary.replace(args.output)
    summary["output"] = str(args.output)
    summary["bytes"] = args.output.stat().st_size
    summary["sha256"] = sha_file(args.output)
    args.output.with_suffix(".manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    args.output.with_suffix(".PRESERVE").write_text("Reusable raw source bank; do not remove with transient training checkpoints.\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "extra_source_manifest"}, indent=2))


if __name__ == "__main__":
    main()
