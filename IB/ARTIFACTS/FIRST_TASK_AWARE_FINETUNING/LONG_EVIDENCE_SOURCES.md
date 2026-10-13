# Long evidence candidates

Prepared for the overnight task-aware campaign: natural long documents and coherent office-reference dossiers, with exact shorter support sections for economical question synthesis. These files contain raw sources, not generated questions.

Data lives in `IB/TMP/FIRST_TASK_AWARE_FINETUNING/long_evidence_sources/`, under a `PRESERVE` flag. `candidates.jsonl` is the immutable publication; `manifest.json` records counts, hashes, tokenizer revision and length bands. The preparation script is `bench/prepare_long_evidence.py`.

Published **748 candidates** (47,443,858 bytes), with SHA-256 `b30e60b355edf8469cea8b8ab167182c13fed4543b4fdcaa8e4e2441d7b67d41`:

| Source family | 8k band | 16k band | 32k band | Total |
|---|---:|---:|---:|---:|
| Code | 85 | 85 | 85 | 255 |
| Manuals | 55 | 52 | 23 | 130 |
| Legal opinions | 50 | 50 | 50 | 150 |
| Mathematical/scientific papers | 50 | 50 | 50 | 150 |
| Office reference dossiers | 35 | 23 | 5 | 63 |
| **Total** | **275** | **260** | **213** | **748** |

There are 685 intact source parents and 63 explicitly partial Office dossiers. The smaller number of long Office and manual packets reflects available coherent material; no unrelated snippets were concatenated to force domain quotas.

The original snippet pools mostly cannot supply intact 32k-token documents. Inventory of whole cached code files found 4,185 files of at least 32k characters, 1,628 of at least 60k, and 696 of at least 100k. Legal source-window metadata identified 1,553 original parents of at least 32k characters, 475 of at least 60k, and 177 of at least 100k; the preparer fetches complete original opinions rather than concatenating their sparse cached windows. Office standalone pages are mostly short, so its long contexts are explicit related-page dossiers.

Source token counts use the cached **Qwen3.5-2B encoder tokenizer**, revision `15852e8c16360a2fea060d615a32b45270f8a8fc`. Natural lengths are retained in three bands: 6,000–8,100, 11,000–16,000, and 22,000–31,500 raw tokens, labeled 8k/16k/32k. Nothing is padded to reach a label. Counts exclude the encoder prompt wrapper; the runner must check its final rendered input against its configured limit.

Each JSONL row contains:

- `source_id`: SHA-256 of the exact evidence text; `text`, `domain`, `doc_id`, `title`, `source_kind`, and pinned `provenance`.
- `source_tokens`, `target_tokens`, and `complete_parent`.
- `split_group_id`, which groups the whole original document or Office application/object. `parent_members` identifies the component pages of a dossier.
- `sections`: ordered exact character spans with IDs, token counts and provenance.
- `suggested_support_section_ids`: three sections totaling at most 6,000 tokens. Their text is an exact substring of the parent evidence.

Cached source-code and manual files, official documentation, raw judicial opinions, and raw arXiv records are retained intact as provided by their pinned sources (`complete_parent=true`). The mathematical-paper category includes quantitative/scientific articles with mathematical exposition. Microsoft Office dossiers combine alphabetically ordered reference pages for one application/object and are explicitly marked `complete_parent=false`; they are selected reference packets, not whole manuals. No incomplete stitched SEC filing is presented as an intact document.

For synthesis, extract two or three exact support spans and pass them as distinct numbered units using a multi-source study mode with `require_multiple_sources=True`. A single `document` source mode supplies one numbered unit and is unsuitable for that requirement. The whole longer `text` remains the evidence for the resulting training example. Suggested sections provide starting points; generation may decline unproductive combinations. A later selector can choose different sections without changing or invalidating this source file.

Keep development/confirmation partitions at `split_group_id` granularity, and retain component document identities for dossiers. Different windows of the same parent must not become separate split groups. Raw source reuse from other tasks is allowed; no native benchmark question/answer or solution-supervision records were ingested.

Validation checked all row/file hashes, unique parent identities, contiguous exact section spans, two or three supports totaling at most 6k tokens, and exclusive dossier-member ownership by parent group. Fifteen representative domain/band combinations also passed exact tokenizer recounts; cached-file examples matched their complete originals. Evidence is `validation.json`. No GPU or generation job was launched, and no active bank, product or runner file was changed.
