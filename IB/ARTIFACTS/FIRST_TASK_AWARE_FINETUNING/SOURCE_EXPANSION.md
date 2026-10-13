# Additional raw sources for the reusable QA bank

Prepared 2026-09-21. These are **raw source windows, not generated training questions**. They are ready for synthesis and are independent of the prompt/code cache identity. The existing `bank_sources_100k.json` was not modified.

Local data: `IB/TMP/FIRST_TASK_AWARE_FINETUNING/expanded_sources/`. The directory has a `PRESERVE` flag, a combined `manifest.json`, and a manifest beside each plain JSONL pool.

| Pool | Windows | Distinct documents | Material |
|---|---:|---:|---|
| `law.jsonl` | 30,433 | 18,477 | 14,147 regulatory notices/rules/proposed rules; 16,286 judicial-opinion windows |
| `math_libretexts.jsonl` | 7,713 | 5,062 | Mathematical and statistical exposition, including worked explanations |
| `office_business.jsonl` | 3,333 | 2,549 | Business management, organizational communication, accounting and related exposition |
| `office_libreoffice.jsonl` | 1,972 | 1,887 | Writer, Calc, Impress and other office-application help |
| `office_automation.jsonl` | 24,959 | 24,925 | Official Microsoft Office application automation documentation |
| `technical_peps.jsonl` | 1,787 | 740 | Python language and process specifications |
| `technical_repository_docs.jsonl` | 6,530 | 4,894 | Manuals, READMEs and contribution guides from the existing pinned repository cache |

Law uses the raw [Common Pile Regulations.gov corpus](https://huggingface.co/datasets/common-pile/regulations) and [Caselaw Access Project corpus](https://huggingface.co/datasets/common-pile/caselaw_access_project), sampling multiple shards. Supporting scientific appendices and obvious fragmented OCR are omitted. Legal material is a dated corpus snapshot; sources retain document dates where supplied.

Math and business use [Common Pile LibreTexts](https://huggingface.co/datasets/common-pile/libretexts). Selection uses the original site/domain and topic URL. Exercise/answer-set titles and pages with explicit embedded exercise sections are omitted; worked explanatory examples remain. Known reader/navigation prefixes and serialized MindTouch navigation are stripped; placeholder `New_Page` and front/back-matter pages are omitted. This is a quick corpus-content filter, not a benchmark overlap audit.

Office procedures use [LibreOffice's authored English help](https://github.com/LibreOffice/help). Automation uses [Microsoft's VBA documentation](https://github.com/MicrosoftDocs/VBA-Docs), preserving visible text and examples while dropping repetitive support footers and Markdown metadata. It covers Excel (4,871 windows), Word (4,754), PowerPoint (2,184), Outlook (3,364), Access, Project, Visio and Publisher. These are office-domain operations, but remain text evidence rather than workbook/GUI execution tasks.

Technical specifications use actual root `peps/pep-*.rst` files from [Python PEPs](https://github.com/python/peps); the Sphinx test fixture is excluded. Repository manuals come from the existing revision-named checkouts; the preparer reads documentation files only. RFC and USGPO ingestion were investigated but are not part of these outputs.

For an overall 6% office share, a workable initial internal allocation is 60% application automation, 30% business/communication and 10% procedural help: 360/180/60 accepted questions per 10,000. Prefer Excel/Word/PowerPoint/Outlook within automation. Do not sample merely in proportion to pool size. Substantial business windows can support 2–3 distinct questions; short help/API pages should normally supply one. This internal allocation is a suggestion for orchestration, not an already-enforced quota.

Each JSONL row has `source_id`/`sha256` (the exact text hash), `text`, original `doc_id`, `source_kind`, `domain`, `subdomain`, `title`, and `provenance` containing source URL, pinned revision, and offsets within the normalized document. Keep original document grouping when selecting held-out data or related windows. Readability and reuse do not depend on loading the original study cache.

Preparation: `bench/prepare_extra_sources.py`. The script uses at most four CPU download/preparation workers and bounds contributions per document. `manifest.json` records pinned revisions, pool counts, hashes and representative samples. Validation: syntax compilation; all four new substantive law/math/business/automation pools parsed with exact row text-hash and provenance checks; representative samples inspected. No GPU was allocated, no product files were changed, and no generation or benchmark validation is claimed here.

## Combined nine-family bank

`bench/merge_bank_sources.py` produced `IB/TMP/FIRST_TASK_AWARE_FINETUNING/bank_sources_200k.json` (1,577,330,777 bytes), with a sibling manifest. SHA-256: `18340ba16c1d584d4f5c2354cac7d112f662dcdf98d266028bce7b51f31afe5c`. The existing source bank remains intact.

| Family | Raw windows |
|---|---:|
| code | 63,207 |
| math_arxiv | 82,047 |
| math_textbook | 7,713 |
| financial | 4,605 |
| web | 55,000 |
| technical_web | 6,178 |
| documents | 8,317 |
| legal | 30,433 |
| office | 11,108 |

The merge moves mathematically marked arXiv windows into `math_arxiv` and BRIGHT posts into `technical_web`; scientific-paper windows do not inflate the manuals/specifications family. Office uses complete 60/30/10 source units until business text runs out, retaining the surplus in its original standalone pool. Two exact repeated office texts were omitted. Other families use deterministic independent shuffles.

The filename expresses the requested campaign target, **not a claim of 200k accepted capacity**. At one question per source before rejection, financial supports roughly 92k total questions at its proposed 5% share, textbooks 96k at 8%, documents 104k at 8%, and technical web 124k at 5%. Multi-question long contexts improve capacity; additional raw sources or an explicit allocation change will still be needed for a reliable 200k campaign. Code also needs rejection headroom beyond its current 63k source windows for a 60k accepted target. Office's raw internal limit is business text; 18,293 automation windows were deliberately not used to fill that gap.

## Published source extensions

The subsequent bounded expansion added **89,175 distinct windows (495 MB)** in seven immutable JSONL files under `IB/TMP/FIRST_TASK_AWARE_FINETUNING/bank_source_extensions/`. These are appended to the shipped bank; its bytes and order remain unchanged. A filename is `<family>__<version>.jsonl`; writing uses a `.partial` file followed by an atomic rename. The directory contains `PRESERVE`, a manifest, individual pool manifests and validation evidence. The synthesis worker can consume completed files while other source preparation or generation continues.

| Family | Added windows | Total available | Accepted questions needed per source for 200k, assuming one question/source |
|---|---:|---:|---:|
| code | 14,339 | 77,546 | 77.4% |
| math_arxiv | — | 82,047 | 29.3% |
| math_textbook | 17,932 | 25,645 | 62.4% |
| financial | 18,294 | 22,899 | 43.7% |
| web | — | 55,000 | 72.7% |
| technical_web | 12,045 | 18,223 | 54.9% |
| documents | 18,000 | 26,317 | 60.8% |
| legal | — | 30,433 | 39.4% |
| office | 8,565 | 19,673 | 61.0% |

This removes the initial raw-count shortfalls. Actual accepted yield remains a measurement; code and ordinary web have the narrowest margins. Long contexts can contribute several distinct questions where useful. The fractions above use the proposed 30/12/8/5/20/5/8/6/6 allocation and do not claim those acceptance rates have been observed on these additions.

Financial additions use [raw SEC 10-K text and tables](https://huggingface.co/datasets/astr010/sec-10k-lsh-chunks), covering 650 annual filings across 68 companies. The corpus's original blocks are roughly 500 characters; contiguous original block IDs are joined into meaningful context windows, retaining filing identity and table indicators. The source includes annual-report narrative and governance material as well as financial statements, so synthesis must still choose supported, useful numerical questions.

The mathematical-exposition extension comprises 5,931 formula-rich LibreTexts physics/engineering/quantitative-chemistry windows and 12,001 academic/reference web windows from [OpenWebMath](https://huggingface.co/datasets/open-web-math/open-web-math), selecting university and named mathematical reference/tutorial sites. These have distinct subdomain tags rather than being described as pure mathematics textbooks. Exam/homework/problem-set URLs and explicit exercise headings are omitted. No benchmark question or reasoning fields are read.

Manual additions are official Python, Kubernetes and MDN documentation at pinned revisions. Technical-web additions use only BRIGHT's raw document files across robotics, Stack Overflow, biology, earth science, economics, psychology, sustainable living and Pony; AoPS, LeetCode and theorem-question files are not opened. Added code comes from 11,000 previously unused files in the existing pinned repository cache. Office additions blend unused automation/help pages with business/commercial-operations exposition, preserving approximately 60/30/10 source composition after exact duplicates are removed; this is not a claim of exact accepted-output proportions.

Preparation: `bench/prepare_source_extensions.py`. It indexes the base through streaming JSON, uses bounded document sampling, and does not reread an entire bank into memory. An initial full-JSON indexing attempt was killed before publishing data; it was replaced by the streaming path. Preparation retries left all already-published files untouched. Validation parsed every extension row, checked exact text hashes and file hashes/counts, verified pinned provenance, and confirmed no repeated exact source IDs against the base or other extensions. Added code document IDs were also confirmed absent from the original code selection. Source offsets refer to the normalized input block; financial provenance additionally identifies the original contiguous chunk IDs. No comprehensive overlap audit was performed.

An eighth immutable extension, `web__msmarco_headroom_20260921_v1.jsonl`, adds **40,000 raw MS MARCO passages** for future capacity/headroom. Its Parquet reader projects only `passages.passage_text`; queries, answers and relevance fields do not enter materialization. Exact source IDs exclude the base and all seven prior extensions. Ordinary web now has **95,000 source windows**, requiring about 42.1% acceptance for its 40k-question allocation in a 200k campaign. The short current v2 sample had 108 accepted from 128 committed attempts (84.4%); this addition addresses future headroom rather than treating that sample as a deficit. File size: 46,752,214 bytes; SHA-256 `4587781c6ccabc8a2d8e1fa7be2fdc465d020f856bdece539dd3976b7e3879ff`. Total published extensions are now 129,175 windows across eight files.
