# Slice 1a — final handoff: Round 1 results, the description learning curves, and the diffs to apply

This is the closing document of slice 1a. It carries the three probe measurements, the Round 1 runs (each fitting mode alone, one epoch, on its own RTX PRO 6000), the description learning curves on the test queries, the study cache you now hold, the overnight incidents and their fixes, and at the end the exact deltas of the workspace against `/source` (25 cards, staged copies under `activation/` in this folder). Every node is torn down.

**Headline.** Description fitting reaches the trained Qwen3-Embedding-4B reference on BRIGHT economics (in-batch rank-1 0.61 / MRR@10 0.67 / nDCG@10 0.63 against 0.61 / 0.67 / 0.64), and it gets within a few points of it after about a hundred descriptions. Questions plateau lower (0.58 / 0.65 / 0.59) and do not improve from 500 to 25,000 examples. Programmatic spans get worse with more data (0.44 → 0.39 rank-1). Your conclusion: far fewer examples than expected put the model within a few points of carefully trained embedders, so the next tasks matter more than optimizing a saturated benchmark.

## Round 1 (2026-09-06, one epoch each, evaluation on the 103 test queries as one batch against their 800 gold chunks)

| run | study set | batch, step, throughput | training time | rank-1 / MRR@10 / nDCG@10 | loss |
| --- | --- | --- | --- | --- | --- |
| description, corpus pass 0 | 18,587 of 18,966 descriptions (2 % held out) | 12, 8.9 s, 2.8k tok/s, peak 42 GB | 3 h 49 m | **0.61 / 0.67 / 0.63** | 3.02 |
| question, 25k chunks | 24,498 questions | 13, 7.9 s, 2.8k tok/s, 44.6 GB | 4 h 07 m | 0.58 / 0.65 / 0.59 | 3.39 |
| programmatic, 150k spans | 147,000 spans | 64, 7.3 s, 2.2k tok/s, 37 GB | 4 h 39 m | 0.39 / 0.45 / 0.39 | 5.07 |
| probe, 145 descriptions, 2 epochs | | | 4 min | 0.54 / 0.62 / 0.60 | 3.46 |
| probe, 502 questions, 2 epochs | | | 10 min | 0.59 / 0.65 / 0.60 | 3.43 |
| probe, 2,007 spans, 2 epochs | | | 7 min | 0.44 / 0.51 / 0.48 | — |
| **Qwen3-Embedding-4B (reference, no document instruction)** | | | | **0.61 / 0.67 / 0.64** | 3.90 |
| frozen Qwen3-4B (EOS readout) | | | | 0.01 / 0.03 / 0.02 | 7.46 |

Reports and summaries: `IB/TMP/SYNC/BRIGHT_ECON_ABLATION/REPORTS/{description,question,programmatic}_20260906_1520/` (`report.tressoir.html`, `report_data.json`, `run_summary.json`). All three came entirely from the cache on the study side; the batch heuristic picked smaller batches than in the probes because the full sets contain heavier examples, so the wall times matched the probe forecast to within 10 %.

### Description learning curves on the test set (`--report-on-test`, a point every 5 % of the epoch)

The reporting batch is the eval batch itself (103 queries, 800 gold chunks), scored by the model only; the reference lines are known. Three runs over nested prefixes of pass 0 (same seed, same cache), each with its own learning-rate schedule:

| examples seen | 2.7k chunks: 968 examples, 70 steps | 5k chunks: 1,832 examples, 141 steps | pass 0: 18,587 examples, 1,549 steps (stopped at 30 %) |
| --- | --- | --- | --- |
| 0 | 0.02 / 0.03 / 0.02 | 0.02 / 0.03 / 0.02 | 0.02 / 0.03 / 0.02 |
| about 60 to 90 | 0.15 / 0.17 / 0.11 | 0.44 / 0.51 / 0.45 | — |
| about 110 to 180 | 0.59 / 0.65 / 0.61 | 0.53 / 0.63 / 0.59 | — |
| about 900 (5 %) | 0.53 / 0.60 / 0.55 (end) | 0.55 / 0.63 / 0.58 | 0.56 / 0.62 / 0.58 |
| about 1,850 (10 %) | | 0.53 / 0.62 / 0.56 (end) | 0.54 / 0.63 / 0.60 |
| 2,800 (15 %) | | | 0.53 / 0.63 / 0.59 |
| 3,700 (20 %) | | | 0.64 / 0.69 / 0.63 |
| 4,600 (25 %) | | | 0.62 / 0.69 / 0.64 |
| 5,500 (30 %) | | | 0.63 / 0.69 / 0.62 |

Readings: the climb is over after 110 to 180 examples; below 2k examples the plateau is 0.53 to 0.57 rank-1 with ±0.04 of noise between adjacent points; the long run reaches the reference band (0.62 to 0.64) between 3k and 5k examples, a quarter of the way through its epoch, and its final 0.61 (Round 1) says the rest of the epoch adds nothing. Folders: `REPORTS/description_n2700_20260906_2018/`, `description_n5000_20260906_2018/`, `description_n50811_20260906_2101_be834e59/` (partial, stopped on your call).

## The study cache you hold

`IB/TMP/SYNC/BRIGHT_ECON_ABLATION/CACHE/`, caching id `econ`, generator and labeler RedHatAI/Qwen3.5-4B-FP8-dynamic, archived as `IB/TMP/SAVED_RESULTS/sep6_queries_and_labels_4B.tgz` (28 MB gzipped, 156 MB raw):

| file | rows | note |
| --- | --- | --- |
| questions, seed 0 and seed 1 | 50,811 each | every chunk, one question per row (5 and 3 parse failures) |
| descriptions, seed 0 and seed 1 | 50,811 each | 18,966 kept at seed 0 (37 %), 34,009 at seed 1 (67 %); null rows are the "no bridge" chunks |
| question labels, `labels-study` | 101,612 | both passes; two prompts over the context limit skipped |
| description labels, `labels-description` | 52,975 | both passes |
| A3B labels | 512 questions, 8,192 descriptions | partial, from the probe and the crashed labeling run |

Measured on the caching run: generation 45 to 55 chunks per second (about 19 minutes per corpus pass), labeling with the 4B 23 labels per second over the full set (73 minutes per 101k), both about half of what the 512-chunk probes extrapolated. Sampling per pass is seeded with `study_seed + pass`; because vLLM gives every request of a pass the same seed, all requests of a pass share one noise sequence, which is why the two description passes kept 37 % and 67 %. You accepted this (the second pass is for diversity); a per-request seed remains the fix if it ever matters.

## What changed in the code since the first cards

- `dataset_study.py`: `_generate_examples` loops over corpus-size passes, pass k at seed `study_seed + k` for the shuffle, the sampling and the cache key, example ids continuing across passes, one progress line per pass; `_sample_study_chunks` and `_make_engine_chat_kwargs` take the seed. A label pool over the engine context (`VLLMValidationError`) no longer fails the batch: prompts over `LABEL_PROMPT_TOKEN_LIMIT` (18k tokens) are skipped and the rest of the batch runs. The label cache key carries the study kind (`labels-study`, `labels-description`): questions and descriptions labeled on two nodes wrote one file name and their synced copies overwrote each other.
- `bench/bright_econ_ablation/_common.py`: `--study-only` (generate and label into the cache, then stop), `--report-on-test` (the test queries as the reporting batch, a learning curve on the eval metric), `--reporting-fraction`, and report folders named `<mode>_n<samples>_<stamp>_<uuid8>` so parallel runs of one mode never share a folder (two curve runs launched in the same minute did, and the page flickered between them).
- Everything from the first cards stands: origins and study kinds, prompts file, JSONL cache, `resolve_path` and `exec --sync`, prefix-plus-view AC model with the document instruction, `train` without validation plus `eval` on one batch with the references, the three benches with `--probe`, the tests.

## Incidents worth knowing

- **A3B NVFP4 labeler segfault** in a CUDA-graph replay after 8,192 description labels; the vLLM client then hung in cleanup for twelve hours with the node up. The 4B labels cover everything, so nothing depends on the A3B.
- **Label file collision and lost labels.** With one label file name for both kinds, the description node's pulls overwrote the question labels locally, and a manual `sky sync` pushed that file back over the question node's copy: 51.7k question labels were lost and relabeled (73 minutes). Fixed by the per-kind key; the originals of the split files are under `IB/TMP/BRIGHT_ECON_ABLATION/cache_backup/`.
- **Idle autostop plus capacity.** The question node stopped on its 30-minute autostop after the context-length crash, and us-east-1a had no RTX PRO 6000 capacity for a while; a raw `sky start` also reused a stale registry login, so a stopped node must be resumed through the wrapper's `setup`.
- **Programmatic overfits with volume.** 147k spans at batch 64 drove the training loss to 0.02 and the reporting batch to rank-1 1.00 while the test metric fell from 0.44 to 0.39. Round 2's programmatic pretraining, as planned at 50k spans, is unlikely to help and was not run.

## Nodes

Every node of this slice is torn down. `sky ls` shows only the stopped `ac-fp4-probe` from before; it and the untracked instance still need your two commands (`activation/cloud/sky.py` and the wrapper's AWS CLI):

```
uv run sky teardown ac-fp4-probe
uv tool run --no-config --python 3.13 --from awscli==1.46.1 aws ec2 terminate-instances --region us-east-1 --instance-ids i-0f794f54eae4b1c41
```

## Validation

| check | where | result |
| --- | --- | --- |
| CPU end-to-end training test (P=4, V=4, d_ac 256, 2 layers; `train` then `eval` with the frozen-base reference) | `IB/TMP/BRIGHT_ECON_ABLATION/cpu_e2e.log` | `1 passed in 963 s` (before the pass and label-key changes; not rerun) |
| batch-composition invariance with prefix rows | `IB/TMP/BRIGHT_ECON_ABLATION/invariant_pv.log` | INVARIANT HOLDS (max diff 1e-6) |
| pass loop (slicing, seeds, running example index) | stub check in this session | 120 over 50 chunks → passes of 50, 50, 20 at seeds 0, 1, 2 |
| three probes, three Round 1 runs, three curve runs, two caching runs | `IB/TMP/BRIGHT_ECON_ABLATION/{probe_*,r1_*,curve_*,cache_*}.log` | all EXIT 0 except the two documented crashes |
| cache round trip and reuse | Round 1 and the curve runs | every study stage "from the cache", no engine loaded |

## Applying the diffs

The cards below are the exact deltas of the workspace against `/source` for every file this slice touched, in reading order, `git diff --no-index` form. The staged copies under `activation/` next to this document are the same files, so `cp -r` from there is the alternative to reading the hunks. `runtime_config.py` shows only your own lines (unchanged by me). `pyproject.toml` and `uv.lock` are as slice 1 left them.

## Diffs to apply

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">activation/dataset/dataset.py</span>
    <span class="card-oneliner">`DataOrigin` split (your lines) plus one stats counter for descriptions that need no bridge.</span>
    <span class="card-badge">Diff</span>
  </summary>

Exact delta vs `/source/activation/dataset/dataset.py`:

````diff-python
diff --git asource/activation/dataset/dataset.py bworkspace/activation/dataset/dataset.py
index bfcdead..909a6e5 100644
--- asource/activation/dataset/dataset.py
+++ bworkspace/activation/dataset/dataset.py
@@ -153,6 +153,7 @@ class DatasetStats:
     study_output_tokens: list[int] = field(default_factory=list)
     study_batch_latencies: list[float] = field(default_factory=list)
     study_num_parse_failures: int = 0
+    study_num_description_no_bridge: int = 0 # Snippets the generator marked as needing no abstraction bridge.
     study_label_prompt_tokens: list[int] = field(default_factory=list)
     study_label_output_tokens: list[int] = field(default_factory=list)
     study_label_batch_latencies: list[float] = field(default_factory=list)
@@ -206,6 +207,7 @@ class DatasetStats:
                 if self.study_batch_latencies else 0.0
             ),
             "study_num_parse_failures": self.study_num_parse_failures,
+            "study_num_description_no_bridge": self.study_num_description_no_bridge,
             "num_study_label_requests": len(self.study_label_prompt_tokens),
             "avg_study_label_prompt_tokens": average(self.study_label_prompt_tokens),
             "avg_study_label_output_tokens": average(self.study_label_output_tokens),
````

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">activation/dataset/dataset_study_prompts.py</span>
    <span class="card-oneliner">Every study prompt, schema and message builder: questions, descriptions (your draft), labels for either kind.</span>
    <span class="card-badge">Diff</span>
  </summary>

Exact delta vs `/source/activation/dataset/dataset_study_prompts.py`:

````diff-python
diff --git asource/activation/dataset/dataset_study_prompts.py bworkspace/activation/dataset/dataset_study_prompts.py
index 1f4c821..16b11f7 100644
--- asource/activation/dataset/dataset_study_prompts.py
+++ bworkspace/activation/dataset/dataset_study_prompts.py
@@ -1,4 +1,215 @@
 """
-@AI: Move all the prompt functions/schemas here.
-As well the nested string constructions.
-"""
\ No newline at end of file
+Every prompt of the dataset study: the two study kinds (a hard question with its answer, or a
+search-friendly description), the relevance-label prompt for either kind, their JSON schemas and
+the chat message builders. `dataset_study.py` keeps only the loops.
+"""
+from .dataset import LabeledRetrievalQAExample
+
+STUDY_QA_JSON_SCHEMA = {
+    "type": "object",
+    "properties": {
+        "question": {"type": "string"},
+        "answer": {"type": "string"},
+    },
+    "required": ["question", "answer"],
+    "additionalProperties": False,
+}
+
+STUDY_DESCRIPTION_JSON_SCHEMA = {
+    "type": "object",
+    "properties": {
+        "needs_abstraction_bridge": {"type": "boolean"},
+        "retrieval_summarization": {"type": ["string", "null"]},
+    },
+    "required": ["needs_abstraction_bridge", "retrieval_summarization"],
+    "additionalProperties": False,
+}
+
+LABEL_JSON_SCHEMA = {
+    "type": "object",
+    "properties": {
+        "positives": {"type": "array", "items": {"type": "integer"}},
+        "negatives": {"type": "array", "items": {"type": "integer"}},
+    },
+    "required": ["positives", "negatives"],
+    "additionalProperties": False,
+}
+
+
+def _study_context_block(study_context: str | None, framing: str) -> str:
+    if not study_context:
+        return ""
+    return f"""
+# Exam Context
+The following emphasizes the kind of questions students are expected to handle.
+```
+-----
+{study_context}
+-----
+```
+{framing}
+"""
+
+
+def make_study_qa_prompt(study_context: str | None) -> tuple[str, str, dict]:
+    """
+    Returns system prompt, instructions, json schema for study q/a tasks.
+    """
+    system_prompt = """
+# Core Guidelines
+- You are a study/exam-like questions formulator.
+- You are given a snippet from the corpus students are expected to study and understand.
+- From this, your goal is to generate a hard question (unanswerable with common knowledge or simple keyword look ups).
+    - This is IMPORTANT: the question should be **hard**, or it won't test the student's learning.
+    - There should be a reasonably high lexical distance to prevent simple keyword lookups.
+    - The question should have an accompanying answer.
+- Your priority is: a hard question along with its expected, correct answer.
+- By default, keep your question short to medium-short unless the exam recommends longer questions.
+- Other notes:
+    - Avoid things like "according to the text". Ask the question as is.
+"""
+    system_prompt += _study_context_block(study_context, "Frame your questions in light of this.")
+    system_prompt += """
+# Response Format
+Your answer must be formatted as a json object like:
+```json
+{
+    "question": "...",
+    "answer": "..."
+}
+```
+"""
+    instructions = """
+# Instructions
+Generate the exam-like question for student's study.
+"""
+    return system_prompt, instructions, STUDY_QA_JSON_SCHEMA
+
+
+def make_study_description_prompt(study_context: str | None) -> tuple[str, str, dict]:
+    """
+    Returns system prompt, instructions, json schema for study description tasks.
+    """
+    system_prompt = """
+# Core Guidelines
+You perform a search-friendly summarization.
+- You are given a snippet from a corpus.
+- You make summary that makes it easy for students to perform various kinds of search for semantic information retrieval.
+
+# What a good summarization does
+- It bridges an abstraction gap:
+    - The snippet contains the concrete information like:
+        - The code for specific algorithm.
+        - A specific biological process.
+        - An agentic trace.
+    - The students search more abstract terms like:
+        - Efficient sorting algorithms with custom comparators.
+        - Biological processes that transform molecule X.
+        - Traces where an agent is stuck on formatting issues.
+    - Notice how the properties the students are looking are clearly from the snippets:
+        - They are just not not explicitly stated as a distinct property yet, making it hard to search for them.
+        - There can also be many such properties. List them all.
+- The description should covers the gap between the two:
+    - It identifies key, specific properties of the document that make such implicit searches easier.
+    - It avoids restating things that already searchable within the documents.
+    - It avoids being so vague the property is likely shared by many other things within the corpus.
+        - It's distinctive: a summary maps strongly to the given snippet.
+- It's relatively concise.
+
+# What a good description does NOT do
+- It does not summarize or paraphrase the surface content. The raw document is already indexed; restating it adds nothing.
+    - To that end, you are given a boolean for `needs_abstraction_bridge`. When set, you can mark the snippet as not needing an abstraction bridge.
+- It does not invent properties the text cannot support. If a property is ambiguous, omit it rather than guess.
+- It's neither too abstract nor too specific. In case of doubt though, prioritize specifity.
+"""
+    system_prompt += _study_context_block(
+        study_context, "Favor the properties such students would search for.",
+    )
+    system_prompt += """
+# Response Format
+Your answer must be formatted as a json object like:
+```json
+{
+    "needs_abstraction_bridge": true|false,
+    "retrieval_summarization": "..."|null
+}
+```
+Set `needs_abstraction_bridge` to false and `retrieval_summarization` to null when the snippet's searchable properties are already stated in it.
+"""
+    instructions = """
+# Instructions
+Write the search-friendly description of the snippet.
+"""
+    return system_prompt, instructions, STUDY_DESCRIPTION_JSON_SCHEMA
+
+
+def make_label_prompt(for_qa: bool = True) -> tuple[str, str, dict]:
+    """
+    Returns system prompt, instructions and json schema for the snippet labeling task: a question
+    with its reference answer (for_qa) or a description alone.
+    """
+    if for_qa:
+        given = "the study question, its reference answer, and numbered corpus snippets"
+        target = "answer the question"
+        positive = "the snippet contains information that clearly helps answer the question"
+        negative = "the snippet contains information that clearly does not help answer the question"
+        related = "related to the question without necessarily helping"
+        task = "for this question"
+    else:
+        given = "a description a student might search with, and numbered corpus snippets"
+        target = "find documents matching a description"
+        positive = "the description clearly fits the snippet (the snippet is what such a search should find)"
+        negative = "the description clearly does not fit the snippet"
+        related = "related to the description without necessarily matching it"
+        task = "for this description"
+    system_prompt = f"""
+# Core Guidelines
+- You are a relevance judge determining what snippets from a corpus students should pay attention to {target}.
+- You are given {given}.
+- Label each snippet index as:
+    - **positive**: {positive}.
+    - **negative**: {negative}.
+    - Leave out anything where you are unsure.
+- Be careful:
+    - The snippets are intentionally chosen to be *{related}*.
+    - Don't mark something as positive just because of shared vocabulary.
+- Every listed index must come from the snippet numbering; never list the same index on both sides.
+"""
+    system_prompt += """
+# Response Format
+Your answer must be formatted as a json object of snippet indices like:
+```json
+{
+    "positives": [0, 2],
+    "negatives": [3, 5, 6]
+}
+```
+Each is a list of indexes that should map to one of the given snippet indices.
+"""
+    instructions = f"""
+Label the snippets as positives / negatives {task}. Leave out ambiguous ones.
+"""
+    return system_prompt, instructions, LABEL_JSON_SCHEMA
+
+
+def study_messages(system_prompt: str, instructions: str, snippet: str) -> list[dict]:
+    """The chat for one study snippet (question or description kind)."""
+    return [
+        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
+        {"role": "user", "content": [{"type": "text", "text": f"# Corpus Snippet\n```\n{snippet}\n```\n{instructions}"}]},
+    ]
+
+
+def label_messages(
+    system_prompt: str, instructions: str, example: LabeledRetrievalQAExample, pool_snippets: list[str], for_qa: bool,
+) -> list[dict]:
+    """The chat for labeling one example's pool: question + reference answer, or the description alone."""
+    snippets = "\n".join(f"[Index={index}]\n```\n{snippet}\n```" for index, snippet in enumerate(pool_snippets))
+    if for_qa:
+        header = f"# Question\n{example.query}\n\n# Reference Answer\n{example.gold_answers[0]}\n\n"
+    else:
+        header = f"# Description\n{example.query}\n\n"
+    return [
+        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
+        {"role": "user", "content": [{"type": "text", "text": f"{header}# Snippets\n{snippets}\n{instructions}"}]},
+    ]
````

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">activation/dataset/dataset_study.py</span>
    <span class="card-oneliner">One generation loop for both study kinds, `origins` filters, description-mode labeling, programmatic examples, cache hooks, selection moved here, `select_testing_data`.</span>
    <span class="card-badge">Diff</span>
  </summary>

Exact delta vs `/source/activation/dataset/dataset_study.py`:

````diff-python
diff --git asource/activation/dataset/dataset_study.py bworkspace/activation/dataset/dataset_study.py
index f9aa30f..c8503f5 100644
--- asource/activation/dataset/dataset_study.py
+++ bworkspace/activation/dataset/dataset_study.py
@@ -5,16 +5,24 @@ Study generation is simple. Until study budget is reached:
 - Select random chunk.
 - Prompt.
 - Parse.
+Two study kinds share that loop: a hard question with its answer (origin SYNTHETIC_QA) and a
+search-friendly description that becomes the query (origin SYNTHETIC_DESCRIPTION).
 Retrieval label generation is also simple:
 - Ask the model to pick positives/hard-negative from the top bm25 pool.
 - Use native labels (treating the top-ranked chunk as a positive/negative).
 - Labeler always wins.
+Programmatic examples (origin PROGRAMMATIC) need no model: a span of a chunk is the query and the
+chunk its positive, with in-batch negatives only; they live in their own dict and are never labeled.
+Generation and labeling outputs can be cached by a caller-chosen caching id (dataset_caching.py).
+Prompts live in dataset_study_prompts.py.
 """
 import json
 import random
 import time
 import typing as t
+from dataclasses import dataclass
 
+from vllm.exceptions import VLLMValidationError
 from vllm.sampling_params import StructuredOutputsParams
 
 from .dataset import (
@@ -24,164 +32,71 @@ from .dataset import (
     LoadedDataset,
     DatasetDocumentChunk,
 )
+from .dataset_caching import StudyCache, StudyCacheKey
+from .dataset_study_prompts import (
+    label_messages,
+    make_label_prompt,
+    make_study_description_prompt,
+    make_study_qa_prompt,
+    study_messages,
+)
 from .dataset_utils import safe_truncate_embedding_chunk, shuffle_fill_truncate
 from ..harness.vllm_wrapper import RECOMMENDED_BATCH_SIZE
 
+LABEL_PROMPT_TOKEN_LIMIT = 18_000   # under the engine's 20k context, with room for the answer
+
 if t.TYPE_CHECKING:
     from activation.harness import HarnessRuntime
 
 
-STUDY_JSON_SCHEMA = {
-    "type": "object",
-    "properties": {
-        "question": {"type": "string"},
-        "answer": {"type": "string"},
-    },
-    "required": ["question", "answer"],
-    "additionalProperties": False,
-}
+@dataclass(frozen=True)
+class StudyKind:
+    """What differs between the two generated study kinds: the prompt, the parser and the origin."""
+    tag: str
+    origin: DataOrigin
+    make_prompt: t.Callable[[str | None], tuple[str, str, dict]]
+    parse: t.Callable[[str], tuple[str, list[str] | None] | None]
+    """Output text -> (query, gold answers) or None when the output yields no example."""
 
-LABEL_JSON_SCHEMA = {
-    "type": "object",
-    "properties": {
-        "positives": {"type": "array", "items": {"type": "integer"}},
-        "negatives": {"type": "array", "items": {"type": "integer"}},
-    },
-    "required": ["positives", "negatives"],
-    "additionalProperties": False,
-}
 
+def _parse_qa(text: str) -> tuple[str, list[str] | None] | None:
+    qa_pair = json.loads(text)
+    question = str(qa_pair["question"]).strip()
+    answer = str(qa_pair["answer"]).strip()
+    if not question or not answer:
+        return None
+    return question, [answer]
 
 
-def make_study_qa_prompt(study_context: str | None) -> tuple[str, str, dict]:
-    """
-    Returns system prompt, instructions, json schema for study q/a tasks.
-    """
-    system_prompt = """
-# Core Guidelines
-- You are a study/exam-like questions formulator.
-- You are given a snippet from the corpus students are expected to study and understand.
-- From this, your goal is to generate a hard question (unanswerable with common knowledge or simple keyword look ups).
-    - This is IMPORTANT: the question should be **hard**, or it won't test the student's learning.
-    - There should be a reasonably high lexical distance to prevent simple keyword lookups.
-    - The question should have an accompanying answer.
-- Your priority is: a hard question along with its expected, correct answer.
-- By default, keep your question short to medium-short unless the exam recommends longer questions.
-- Other notes:
-    - Avoid things like "according to the text". Ask the question as is.
-"""
+def _parse_description(text: str) -> tuple[str, list[str] | None] | None:
+    parsed = json.loads(text)
+    description = parsed.get("retrieval_summarization")
+    if not parsed["needs_abstraction_bridge"] or not description or not str(description).strip():
+        return None                                                    # the snippet needs no bridge: no example
+    return str(description).strip(), None
 
-    if study_context:
-        system_prompt += f"""
-# Exam Context
-The following emphasizes the kind of questions students are expected to handle.
-```
------
-{study_context}
------
-```
-Frame your questions in light of this.
-"""
-    system_prompt += """
-# Response Format
-Your answer must be formatted as a json object like:
-```json
-{
-    "question": "...",
-    "answer": "..."
-}
-```
-"""
-    instructions = f"""
-# Instructions
-Generate the exam-like question for student's study.
-"""
-    return system_prompt, instructions, STUDY_JSON_SCHEMA
 
+QA_KIND = StudyKind("study", DataOrigin.SYNTHETIC_QA, make_study_qa_prompt, _parse_qa)
+DESCRIPTION_KIND = StudyKind("description", DataOrigin.SYNTHETIC_DESCRIPTION, make_study_description_prompt, _parse_description)
+ORIGIN_KIND_TAGS = {QA_KIND.origin: QA_KIND.tag, DESCRIPTION_KIND.origin: DESCRIPTION_KIND.tag}   # label cache file per kind
 
-def make_study_description_prompt(study_context: str | None) -> tuple[str, str, dict]:
-    """
-    Returns system prompt, instructions, json schema for study description tasks.
-    """
-    system_prompt = """
-# Core Guidelines
-You perform a search-friendly summarization.
-- You are given a snippet from a corpus.
-- You make summary that makes it easy for students to perform various kinds of search for semantic information retrieval.
-
-# What a good summarization does
-- It bridges an abstraction gap:
-    - The snippet contains the concrete information like:
-        - The code for specific algorithm.
-        - A specific biological process.
-        - An agentic trace.
-    - The students search more abstract terms like:
-        - Efficient sorting algorithms with custom comparators.
-        - Biological processes that transform molecule X.
-        - Traces where an agent is stuck on formatting issues.
-    - Notice how the properties the students are looking are clearly from the snippets:
-        - They are just not not explicitly stated as a distinct property yet, making it hard to search for them.
-        - There can also be many such properties. List them all.
-- The description should covers the gap between the two:
-    - It identifies key, specific properties of the document that make such implicit searches easier.
-    - It avoids restating things that already searchable within the documents.
-    - It avoids being so vague the property is likely shared by many other things within the corpus.
-        - It's distinctive: a summary maps strongly to the given snippet.
-- It's relatively concise.
-    
-# What a good description does NOT do
-- It does not summarize or paraphrase the surface content. The raw document is already indexed; restating it adds nothing.
-    - To that end, you are given a boolean for `needs_abstraction_bridge`. When set, you can mark the snippet as not needing an abstraction bridge.
-- It does not invent properties the text cannot support. If a property is ambiguous, omit it rather than guess.
-- It's neither too abstract nor too specific. In case of doubt though, prioritize specifity.
- 
-# Response Format
-Your answer must be formatted as a json object like:
-```json
-{
-    "needs_abstraction_bridge": true|false,
-    "retrieval_summarization": "..."|null
-}
-```
-"""
-    if study_context:
-        # The following emphasizes what descriptions should aim to extractio.
-        pass
-    # @AI: Implement me.
 
-def make_label_prompt(for_qa: bool = True) -> tuple[str, str, dict]:
+def programmatic_query(text: str, rng: random.Random, min_chars: int, max_chars: int) -> str:
     """
-    Returns system prompt, instructions and json schema for the snippet labeling task.
+    A random contiguous span of the text, min_chars to max_chars long (the whole text when it is
+    shorter than min_chars), snapped outward to whitespace so it starts and ends on whole words.
     """
-    # @AI: adapt to the case with just description. Should be nearly identical with some tweaks.
-    system_prompt = """
-# Core Guidelines
-- You are a relevance judge determining what snippets from a corpus students should pay attention to answer a question.
-- You are given the study question, its reference answer, and numbered corpus snippets.
-- Label each snippet index as:
-    - **positive**: the snippet contains information that clearly helps answer the question.
-    - **negative**: the snippet contains information that clearly does not help answer the question.
-    - Leave out anything where you are unsure.
-- Be careful:
-    - The snippets are intentionally chosen to be *related* to the question without necessarily helping.
-    - Don't mark something as positive just because of shared vocabulary.
-- Every listed index must come from the snippet numbering; never list the same index on both sides.
-"""
-    system_prompt += """
-# Response Format
-Your answer must be formatted as a json object of snippet indices like:
-```json
-{
-    "positives": [0, 2],
-    "negatives": [3, 5, 6]
-}
-```
-Each is a list of indexes that should map to one of the given snippet indices.
-"""
-    instructions = """
-Label the snippets as positives / negatives for this question. Leave out ambiguous ones.
-"""
-    return system_prompt, instructions, LABEL_JSON_SCHEMA
+    text = text.strip()
+    if len(text) <= min_chars:
+        return text
+    length = rng.randint(min_chars, min(max_chars, len(text)))
+    start = rng.randint(0, len(text) - length)
+    end = start + length
+    while start > 0 and not text[start - 1].isspace():
+        start -= 1
+    while end < len(text) and not text[end].isspace():
+        end += 1
+    return text[start:end].strip()
 
 
 class DatasetStudyGenerator:
@@ -189,8 +104,8 @@ class DatasetStudyGenerator:
         self.harness = harness
         self.loaded_dataset = loaded_dataset
         self.dataset_id = loaded_dataset.dataset_id
-        self.dataset_index = harness.dataset_manager.dataset_indexes[self.dataset_id]
-        self.study_context = loaded_dataset.study_context
+        self.dataset_index = harness.dataset_manager._get_or_create_index(self.dataset_id)
+        self.study_context = None # @AI: Add me later
         self.qa_model_name = harness.harness_config.dataset_study_qa_model_name
         self.label_model_name = harness.harness_config.dataset_study_label_model_name
         self.qa_batch_size = harness.harness_config.dataset_study_qa_batch_size or RECOMMENDED_BATCH_SIZE
@@ -202,111 +117,194 @@ class DatasetStudyGenerator:
         self.label_pool_max_chars = harness.harness_config.dataset_study_label_pool_max_chars
         self.programmatic_min_chars = harness.harness_config.dataset_study_programmatic_min_chars
         self.programmatic_max_chars = harness.harness_config.dataset_study_programmatic_max_chars
+        self.cache = StudyCache(harness.harness_config.cache_storage_dir)
         assert self.dataset_index.bm25_index is not None, "Study requires built bm25 indexes."
 
 
-    def _sample_study_chunks(self, num_samples: int) -> list[DatasetDocumentChunk]:
-        rng = random.Random(self.study_seed) # Reproducible study sets.
+    def _sample_study_chunks(self, num_samples: int, seed: int | None = None) -> list[DatasetDocumentChunk]:
+        rng = random.Random(self.study_seed if seed is None else seed) # Reproducible study sets.
         chunk_ids = shuffle_fill_truncate(list(self.dataset_index.chunk_ids), num_samples, rng)  # every chunk before any repeat
         return [self.dataset_index.chunks[chunk_id] for chunk_id in chunk_ids]
 
-    def _make_engine_chat_kwargs(self, json_schema: dict) -> dict:
+    def _make_engine_chat_kwargs(self, json_schema: dict, seed: int | None = None) -> dict:
+        """vllm seeds its sampler per request (same prompt + same seed = same text), hence a seed per generation pass."""
         chat_kwargs = dict(self.chat_kwargs or {})
         sampling_params_kwargs = dict(chat_kwargs.pop("sampling_params", None) or {})
         sampling_params_kwargs.setdefault("max_tokens", 2048)
-        sampling_params_kwargs.setdefault("seed", self.study_seed)
+        sampling_params_kwargs.setdefault("seed", self.study_seed if seed is None else seed)
         sampling_params_kwargs["structured_outputs"] = StructuredOutputsParams(json=json_schema)
         chat_kwargs["sampling_params"] = sampling_params_kwargs
         return chat_kwargs
 
+    def _model_id(self, model_name: str) -> str:
+        return self.harness.loaded_models[model_name].model_config.model_id
+
+    # ----------------------------------------------------------------------------- generation
+    def generate_examples_qa(self, num_samples: int, caching_id: str | None = None) -> list[LabeledRetrievalQAExample]:
+        return self._generate_examples(num_samples, QA_KIND, caching_id)
+
+    def generate_examples_descriptions(self, num_samples: int, caching_id: str | None = None) -> list[LabeledRetrievalQAExample]:
+        return self._generate_examples(num_samples, DESCRIPTION_KIND, caching_id)
+
+    def _register_generated(self, kind: StudyKind, chunk: DatasetDocumentChunk, query: str, gold_answers: list[str] | None, index: int) -> LabeledRetrievalQAExample:
+        example = LabeledRetrievalQAExample(
+            example_id=f"{self.dataset_id}:{kind.tag}:{index}",
+            dataset_id=self.dataset_id,
+            query=query,
+            gold_answers=gold_answers,
+            origin=kind.origin,
+            split=DataSplit.TRAIN,
+            positive_doc_ids=[chunk.doc_id],
+            positive_chunk_ids=[chunk.chunk_id],
+        )
+        self.loaded_dataset.labeled_retrieval_examples[example.example_id] = example
+        return example
 
-    def generate_examples_descriptions(self, num_samples: int) -> list[LabeledRetrievalQAExample]:
-        pass # @AI: Refactor with generate_examples_qa. Only the prompt/parsing should differ. Uses the qa model.
+    def _generate_examples(self, num_samples: int, kind: StudyKind, caching_id: str | None) -> list[LabeledRetrievalQAExample]:
+        """
+        One pass per corpus-size slice of the request, pass k seeded with study_seed + k: its own
+        shuffle (every chunk once), its own sampling seed (a chunk revisited gets a new question or
+        description) and its own cache file. A request within the corpus size is one pass at the
+        study seed, so existing caches stay valid; pass 1 of seed s is the file of pass 0 of seed s + 1.
+        """
+        corpus_size = len(self.dataset_index.chunk_ids)
+        starts = list(range(0, num_samples, corpus_size))
+        examples: list[LabeledRetrievalQAExample] = []
+        for pass_index, start in enumerate(starts):
+            pass_samples = min(corpus_size, num_samples - start)
+            print(
+                f"{self.dataset_id} - Study ({kind.tag}): pass {pass_index + 1}/{len(starts)}, {pass_samples} chunks, "
+                f"seed {self.study_seed + pass_index} ({len(examples)} examples so far).", flush=True,
+            )
+            examples.extend(self._generate_examples_pass(
+                pass_samples, kind, caching_id, seed=self.study_seed + pass_index, first_index=len(examples),
+            ))
+        return examples
+
+    def _generate_examples_pass(
+        self, num_samples: int, kind: StudyKind, caching_id: str | None, seed: int, first_index: int,
+    ) -> list[LabeledRetrievalQAExample]:
+        """
+        One example per sampled chunk that the generator turns into a query (questions always,
+        descriptions when the snippet needs a bridge). With a caching id, cached rows serve the
+        sampled prefix they cover (matched by chunk id) and the rest is generated and appended.
+        """
+        study_samples = self._sample_study_chunks(num_samples, seed)
+        stats = self.loaded_dataset.stats
+        examples: list[LabeledRetrievalQAExample] = []
+        cache_key = None
+        if caching_id:
+            cache_key = StudyCacheKey(caching_id, self.dataset_id, self._model_id(self.qa_model_name), kind.tag, seed)
+        cached_rows = self.cache.read(cache_key) if cache_key else []
+        next_index = first_index                                           # example ids keep counting across passes
+        num_from_cache = 0
+        for row in cached_rows:                                            # cached prefix: same chunks in the same order
+            if num_from_cache >= len(study_samples) or row.get("chunk_id") != study_samples[num_from_cache].chunk_id:
+                break
+            chunk = study_samples[num_from_cache]
+            num_from_cache += 1
+            if row.get("query"):
+                examples.append(self._register_generated(kind, chunk, row["query"], row.get("gold_answers"), next_index))
+                next_index += 1
+        if num_from_cache:
+            print(f"{self.dataset_id} - Study ({kind.tag}): {num_from_cache} chunks from the cache ({len(examples)} examples).")
+        remaining = study_samples[num_from_cache:]
+        if not remaining:
+            return examples
 
-    def generate_examples_qa(self, num_samples: int) -> list[LabeledRetrievalQAExample]:
         loaded_model = self.harness.loaded_models[self.qa_model_name]
         # vllm batches a whole conversation list inside one chat() call, so a
         # simple single-threaded loop needs no locks.
         loaded_model.ensure_engine_loaded() # Keep the cold load out of the batch timings.
-        study_samples = self._sample_study_chunks(num_samples)
-        system_prompt, instructions, json_schema = make_study_qa_prompt(self.study_context)
-        chat_kwargs = self._make_engine_chat_kwargs(json_schema)
-        stats = self.loaded_dataset.stats
-        examples: list[LabeledRetrievalQAExample] = []
-        example_count = 0
+        system_prompt, instructions, json_schema = kind.make_prompt(self.study_context)
+        chat_kwargs = self._make_engine_chat_kwargs(json_schema, seed)
         total_generation_time = 0
         reporting_interval = 0
-        for batch_start in range(0, len(study_samples), self.qa_batch_size):
-            batch = study_samples[batch_start:batch_start + self.qa_batch_size]
+        for batch_start in range(0, len(remaining), self.qa_batch_size):
+            batch = remaining[batch_start:batch_start + self.qa_batch_size]
             conversations = []
             for chunk in batch:
                 snippet = self.dataset_index.get_chunk_section(chunk)
                 snippet = safe_truncate_embedding_chunk(snippet, self.chunk_input_limit)
-                # @AI: Move to prompt file.
-                conversations.append([
-                    {
-                        "role": "system", "content": [
-                            {"type": "text", "text": system_prompt},
-                        ]
-                    },
-                    {
-                        "role": "user", "content": [
-                            {"type": "text", "text": f"# Corpus Snippet\n```\n{snippet}\n```\n{instructions}"},
-                        ]
-                    },
-                ])
+                conversations.append(study_messages(system_prompt, instructions, snippet))
             batch_start_time = time.time()
             outputs = loaded_model.engine_chat_many(conversations, chat_kwargs=chat_kwargs)
             elapsed = time.time() - batch_start_time
             total_generation_time += elapsed
             stats.study_batch_latencies.append(elapsed)
+            cache_rows = []
             for chunk, output in zip(batch, outputs):
                 stats.study_prompt_tokens.append(output.prompt_token_count)
                 stats.study_output_tokens.append(output.output_token_count)
                 try:
-                    qa_pair = json.loads(output.text)
-                    question = str(qa_pair["question"]).strip()
-                    answer = str(qa_pair["answer"]).strip()
-                except (json.JSONDecodeError, KeyError, TypeError):
+                    parsed = kind.parse(output.text)
+                except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
                     # Structured outputs make this rare (e.g. max_tokens cutoff).
                     stats.study_num_parse_failures += 1
+                    parsed = None
+                    cache_rows.append({"chunk_id": chunk.chunk_id, "doc_id": chunk.doc_id, "query": None, "failed": True})
                     continue
-                if not question or not answer:
-                    stats.study_num_parse_failures += 1
+                if parsed is None:                                         # a valid answer that yields no example
+                    if kind is QA_KIND:
+                        stats.study_num_parse_failures += 1
+                    else:
+                        stats.study_num_description_no_bridge += 1
+                    cache_rows.append({"chunk_id": chunk.chunk_id, "doc_id": chunk.doc_id, "query": None})
                     continue
-                example = LabeledRetrievalQAExample(
-                    example_id=f"{self.dataset_id}:study:{example_count}",
-                    dataset_id=self.dataset_id,
-                    query=question,
-                    gold_answers=[answer],
-                    origin=DataOrigin.SYNTHETIC,
-                    split=DataSplit.TRAIN,
-                    positive_doc_ids=[chunk.doc_id],
-                    positive_chunk_ids=[chunk.chunk_id],
-                )
-                self.loaded_dataset.labeled_retrieval_examples[example.example_id] = example
-                examples.append(example)
-                example_count += 1
+                query, gold_answers = parsed
+                examples.append(self._register_generated(kind, chunk, query, gold_answers, next_index))
+                next_index += 1
+                cache_rows.append({"chunk_id": chunk.chunk_id, "doc_id": chunk.doc_id, "query": query, "gold_answers": gold_answers})
+            if cache_key:
+                self.cache.append(cache_key, cache_rows)
             if reporting_interval <= 0:
                 print(
-                    f"{self.dataset_id} - Study: {batch_start + len(batch)}/{len(study_samples)} chunks, "
-                    f"{example_count} questions. {total_generation_time:.2f}s."
+                    f"{self.dataset_id} - Study ({kind.tag}): {num_from_cache + batch_start + len(batch)}/{len(study_samples)} chunks, "
+                    f"{len(examples)} examples. {total_generation_time:.2f}s."
                 )
-                reporting_interval = max(1, len(study_samples) // 20)
+                reporting_interval = max(1, len(remaining) // 20)
             reporting_interval -= len(batch)
         return examples
 
+    def generate_programmatic_examples(self, num_samples: int) -> list[LabeledRetrievalQAExample]:
+        """
+        Programmatic examples for massive AC model training: a random span of a sampled chunk is
+        the query, the chunk is its own positive, negatives come from the batch at training time.
+        Chunks repeat past the corpus size with a different span each time. Not cached, never
+        labeled, stored apart from the labeled examples.
+        """
+        chunks = self._sample_study_chunks(num_samples)
+        rng = random.Random(self.study_seed + 1)
+        examples: list[LabeledRetrievalQAExample] = []
+        start_index = len(self.loaded_dataset.programmatic_retrieval_examples)
+        for offset, chunk in enumerate(chunks):
+            query = programmatic_query(chunk.chunk_text, rng, self.programmatic_min_chars, self.programmatic_max_chars)
+            example = LabeledRetrievalQAExample(
+                example_id=f"{self.dataset_id}:programmatic:{start_index + offset}",
+                dataset_id=self.dataset_id,
+                query=query,
+                origin=DataOrigin.PROGRAMMATIC,
+                split=DataSplit.TRAIN,
+                positive_doc_ids=[chunk.doc_id],
+                positive_chunk_ids=[chunk.chunk_id],
+            )
+            self.loaded_dataset.programmatic_retrieval_examples[example.example_id] = example
+            examples.append(example)
+        print(f"{self.dataset_id} - {len(examples)} programmatic examples ({len(self.dataset_index.chunk_ids)} chunks in the corpus).")
+        return examples
 
-    def _select_examples_to_label(self, num_samples: int, synthetic_only: bool) -> list[LabeledRetrievalQAExample]:
+    # ----------------------------------------------------------------------------- labeling
+    def _select_examples_to_label(self, num_samples: int, origins: list[DataOrigin] | None) -> list[LabeledRetrievalQAExample]:
         """
-        Select examples to label.
+        Select examples to label: unlabeled ones with something to judge against (a reference
+        answer, or a description), of the given origins (None: every origin).
         """
         candidates = [
             example
             for example in self.loaded_dataset.labeled_retrieval_examples.values()
             if not example.oracle_labeled
-            and example.gold_answers and example.gold_answers[0]
-            and (not synthetic_only or example.origin == DataOrigin.SYNTHETIC)
+            and ((example.gold_answers and example.gold_answers[0]) or example.origin == DataOrigin.SYNTHETIC_DESCRIPTION)
+            and (origins is None or example.origin in origins)
         ]
         rng = random.Random(self.study_seed)
         rng.shuffle(candidates)
@@ -411,28 +409,58 @@ class DatasetStudyGenerator:
                 setattr(stats, counter, getattr(stats, counter) + 1)
 
 
-    def generate_examples_labels(self, num_samples: int, synthetic_only: bool) -> list[LabeledRetrievalQAExample]:
+    def generate_examples_labels(
+        self, num_samples: int, origins: list[DataOrigin] | None = None, caching_id: str | None = None,
+        label_model_name: str | None = None,
+    ) -> list[LabeledRetrievalQAExample]:
         """
-        Oracle-label up to num_samples examples that were not labeled yet.
-        When synthetic_only is true, only synthetically generated questions are considered.
+        Oracle-label up to num_samples examples of the given origins (None: all) that were not
+        labeled yet; descriptions are judged alone, questions with their reference answer. With a
+        caching id, examples whose labels are cached take them without a model call; new labels
+        are appended. label_model_name overrides the configured labeler (a probe compares two).
         Returns the examples that ended up labeled in this pass.
         """
-        examples = self._select_examples_to_label(num_samples, synthetic_only)
+        examples = self._select_examples_to_label(num_samples, origins)
         if not examples:
             return []
-        loaded_model = self.harness.loaded_models[self.label_model_name]
-        loaded_model.ensure_engine_loaded() # Keep the cold load out of the batch timings.
-        system_prompt, instructions, json_schema = make_label_prompt()
-        chat_kwargs = self._make_engine_chat_kwargs(json_schema)
+        label_model_name = label_model_name or self.label_model_name
         stats = self.loaded_dataset.stats
+        cache_key = None
+        if caching_id:
+            # One file per study kind: questions and descriptions are labeled by different runs (and nodes),
+            # and a shared file name would make their synced copies overwrite each other.
+            kinds = sorted({ORIGIN_KIND_TAGS.get(example.origin, "native") for example in examples})
+            cache_key = StudyCacheKey(caching_id, self.dataset_id, self._model_id(self.qa_model_name), "labels-" + "+".join(kinds),
+                                      self.study_seed, label_model_id=self._model_id(label_model_name))
+        cached = self.cache.read_by_id(cache_key, "example_id") if cache_key else {}
+        labeled: list[LabeledRetrievalQAExample] = []
+        to_label = []
+        for example in examples:
+            row = cached.get(example.example_id)
+            if row is None:
+                to_label.append(example)
+                continue
+            example.positive_chunk_ids = list(row["positive_chunk_ids"])
+            example.hard_negative_chunk_ids = list(row["hard_negative_chunk_ids"]) or None
+            example.oracle_labeled = True
+            labeled.append(example)
+        if labeled:
+            print(f"{self.dataset_id} - Study labels: {len(labeled)} examples from the cache.")
+        if not to_label:
+            return labeled
+
+        loaded_model = self.harness.loaded_models[label_model_name]
+        loaded_model.ensure_engine_loaded() # Keep the cold load out of the batch timings.
+        prompts = {for_qa: make_label_prompt(for_qa) for for_qa in (True, False)}
+        chat_kwargs = self._make_engine_chat_kwargs(prompts[True][2])
         all_ranked = self.dataset_index.bm25_query_many_frozen(
-            [example.query for example in examples],
+            [example.query for example in to_label],
             top_k=self.label_top_k,
-            excluded_doc_ids=[example.excluded_doc_ids for example in examples],
+            excluded_doc_ids=[example.excluded_doc_ids for example in to_label],
         )
-        labeled: list[LabeledRetrievalQAExample] = []
         example_pool_pairs = []
-        for example, ranked in zip(examples, all_ranked):
+        cache_rows = []
+        for example, ranked in zip(to_label, all_ranked):
             pool = self._build_label_pool(example, ranked)
             if pool:
                 example_pool_pairs.append((example, pool))
@@ -441,32 +469,36 @@ class DatasetStudyGenerator:
                 self._inherit_labels(example, pool)
                 example.oracle_labeled = True
                 labeled.append(example)
+                cache_rows.append(self._label_row(example))
+        if cache_key:
+            self.cache.append(cache_key, cache_rows)
         total_label_time = 0
         reporting_interval = 0
         for batch_start in range(0, len(example_pool_pairs), self.label_batch_size):
             batch = example_pool_pairs[batch_start:batch_start + self.label_batch_size]
             conversations = []
             for example, pool in batch:
-                snippets = "\n".join(
-                    f"[Index={index}]\n```\n{snippet}\n```"
-                    for index, (_, snippet) in enumerate(pool)
-                )
-                # @AI: Support the case with query only: Those should say: Description.
-                # Also move into prompt file.
-                user_text = (
-                    f"# Question\n{example.query}\n\n"
-                    f"# Reference Answer\n{example.gold_answers[0]}\n\n"
-                    f"# Snippets\n{snippets}\n{instructions}"
-                )
-                conversations.append([
-                    {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
-                    {"role": "user", "content": [{"type": "text", "text": user_text}]},
-                ])
+                for_qa = example.origin != DataOrigin.SYNTHETIC_DESCRIPTION
+                system_prompt, instructions, _ = prompts[for_qa]
+                conversations.append(label_messages(system_prompt, instructions, example, [snippet for _, snippet in pool], for_qa))
             batch_start_time = time.time()
-            outputs = loaded_model.engine_chat_many(conversations, chat_kwargs=chat_kwargs)
+            try:
+                outputs = loaded_model.engine_chat_many(conversations, chat_kwargs=chat_kwargs)
+            except VLLMValidationError:
+                # One pool over the engine's context fails the whole batch (glyph-dense chunks tokenize
+                # near one token per character): skip those prompts, unlabeled, and run the rest.
+                lengths = [len(loaded_model.tokenizer("\n".join(part["text"] for message in conversation for part in message["content"])).input_ids)
+                           for conversation in conversations]
+                kept = [index for index, length in enumerate(lengths) if length <= LABEL_PROMPT_TOKEN_LIMIT]
+                skipped = len(conversations) - len(kept)
+                stats.study_num_label_parse_failures += skipped
+                print(f"{self.dataset_id} - Study labels: {skipped} prompts over {LABEL_PROMPT_TOKEN_LIMIT} tokens skipped (max {max(lengths)}).")
+                batch = [batch[index] for index in kept]
+                outputs = loaded_model.engine_chat_many([conversations[index] for index in kept], chat_kwargs=chat_kwargs) if kept else []
             elapsed = time.time() - batch_start_time
             total_label_time += elapsed
             stats.study_label_batch_latencies.append(elapsed)
+            cache_rows = []
             for (example, pool), output in zip(batch, outputs):
                 stats.study_label_prompt_tokens.append(output.prompt_token_count)
                 stats.study_label_output_tokens.append(output.output_token_count)
@@ -486,9 +518,12 @@ class DatasetStudyGenerator:
                 self._inherit_labels(example, pool)
                 example.oracle_labeled = True
                 labeled.append(example)
+                cache_rows.append(self._label_row(example))
+            if cache_key:
+                self.cache.append(cache_key, cache_rows)
             if reporting_interval <= 0:
                 print(
-                    f"{self.dataset_id} - Study labels: {batch_start + len(batch)}/{len(example_pool_pairs)} questions, "
+                    f"{self.dataset_id} - Study labels: {batch_start + len(batch)}/{len(example_pool_pairs)} examples, "
                     f"+{stats.study_num_label_positives} pos / {stats.study_num_label_negatives} neg / "
                     f"{stats.study_num_label_ambiguous} ambiguous. {total_label_time:.2f}s."
                 )
@@ -496,12 +531,108 @@ class DatasetStudyGenerator:
             reporting_interval -= len(batch)
         return labeled
 
+    @staticmethod
+    def _label_row(example: LabeledRetrievalQAExample) -> dict:
+        return {"example_id": example.example_id, "positive_chunk_ids": list(example.positive_chunk_ids or []),
+                "hard_negative_chunk_ids": list(example.hard_negative_chunk_ids or [])}
+
+    # ----------------------------------------------------------------------------- selection
+    def _training_candidates(self, origins: list[DataOrigin] | None, oracle_labeled_only: bool) -> list[LabeledRetrievalQAExample]:
+        """Examples of the given origins from both dicts, chunk labels inherited where missing; no positive chunk drops one."""
+        stats = self.loaded_dataset.stats
+        pools = [self.loaded_dataset.labeled_retrieval_examples.values()]
+        if origins is None or DataOrigin.PROGRAMMATIC in origins:
+            pools.append(self.loaded_dataset.programmatic_retrieval_examples.values())
+        selected: list[LabeledRetrievalQAExample] = []
+        for pool in pools:
+            for example in pool:
+                if origins is not None and example.origin not in origins:
+                    continue
+                if oracle_labeled_only and not example.oracle_labeled:
+                    continue
+                if not example.oracle_labeled and not example.positive_chunk_ids:
+                    self._inherit_labels(example, pool=[])
+                if example.positive_chunk_ids:
+                    selected.append(example)
+                else:
+                    stats.training_select_num_dropped_no_positive += 1
+        return selected
+
+    def select_training_data(
+        self,
+        num_samples: int,
+        origins: list[DataOrigin] | None = None,
+        oracle_labeled_only: bool = True,
+        val_ratio: float = 0.1,
+        max_reporting_size: int = 50,
+        force_partition: bool = True, # Most datasets only have training. This forces a val set.
+        seed: int = 0,
+    ) -> tuple[list[LabeledRetrievalQAExample], list[LabeledRetrievalQAExample], list[LabeledRetrievalQAExample]]:
+        """
+        Select training data. Returns tuples with the following:
+        - training data: actually used to update the gradients.
+        - validation: small amount of data for validation (~10% in general).
+        - reporting: trivial amount of data (subset of the validation set). Used for plotting.
+        There is a general min of 10 for training and validation, and 1 for reporting regardless of the fractions.
+
+        num_samples examples are selected before the split, of the given origins (None: every
+        origin, programmatic included); validation is carved from them by val_ratio unless
+        force_partition is False and the dataset carries native validation-split examples.
+        Oracle-labeled examples are used as they are. Without the oracle, an example inherits
+        chunk labels from its document-level labels: one chunk per positive document and, where
+        the dataset carries native hard negatives, one chunk per hard-negative document. Examples
+        without a positive chunk are dropped and counted in the dataset stats. Test-split examples
+        are never selected here; they belong to select_testing_data.
+        """
+        dataset_id = self.dataset_id
+        stats = self.loaded_dataset.stats
+        selected = [example for example in self._training_candidates(origins, oracle_labeled_only) if example.split != DataSplit.TEST]
+        rng = random.Random(seed)
+        native_validation = [example for example in selected if example.split == DataSplit.VAL]
+        if not force_partition and native_validation:
+            training_pool = [example for example in selected if example.split != DataSplit.VAL]
+            rng.shuffle(training_pool)
+            rng.shuffle(native_validation)
+            training_data = training_pool[:num_samples]
+            validation_data = native_validation[:max(10, round(num_samples * val_ratio))]
+        else:
+            rng.shuffle(selected)
+            chosen = selected[:num_samples]
+            num_validation = max(10, round(len(chosen) * val_ratio))
+            validation_data = chosen[:num_validation]
+            training_data = chosen[num_validation:]
+        assert len(training_data) >= 10, (
+            f"{dataset_id} - Only {len(training_data)} training examples after the split; need at least 10 "
+            f"({len(selected)} selectable, {stats.training_select_num_dropped_no_positive} dropped without a positive chunk)."
+        )
+        assert len(validation_data) >= 10, f"{dataset_id} - Only {len(validation_data)} validation examples; need at least 10."
+        reporting_data = validation_data[:max(1, min(max_reporting_size, len(validation_data)))]
+        print(
+            f"{dataset_id} - Selected {len(training_data)} training / {len(validation_data)} validation / "
+            f"{len(reporting_data)} reporting examples."
+        )
+        return training_data, validation_data, reporting_data
 
-    def generate_programmatic_examples(self, num_samples: int):
+    def select_testing_data(self, num_samples: int | None = None, seed: int = 0) -> list[LabeledRetrievalQAExample]:
         """
-        Generate programmatic examples for massive AC Model training.
+        The native test-split examples (BRIGHT: every labeled query) with chunk labels inherited
+        from their gold documents, shuffled by the seed and truncated to num_samples (None: all).
+        Examples without a positive chunk are dropped and counted.
         """
-        study_samples = self._sample_study_chunks(num_samples)
-        # @AI: simple programmatic formation.
-        # The chunk is its own positive.
-        # only use the in-batch negatives as negatives.
+        stats = self.loaded_dataset.stats
+        selected = []
+        for example in self.loaded_dataset.labeled_retrieval_examples.values():
+            if example.split != DataSplit.TEST or example.origin != DataOrigin.NATIVE:
+                continue
+            if not example.oracle_labeled and not example.positive_chunk_ids:
+                self._inherit_labels(example, pool=[])
+            if example.positive_chunk_ids:
+                selected.append(example)
+            else:
+                stats.training_select_num_dropped_no_positive += 1
+        rng = random.Random(seed)
+        rng.shuffle(selected)
+        if num_samples is not None:
+            selected = selected[:num_samples]
+        print(f"{self.dataset_id} - Selected {len(selected)} testing examples.")
+        return selected
````

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">activation/dataset/dataset_caching.py</span>
    <span class="card-oneliner">`StudyCache`: JSONL rows per (caching id, dataset, model, kind, seed[, labeler]); prefix reuse for generation, by-id reuse for labels.</span>
    <span class="card-badge">Diff</span>
  </summary>

Exact delta vs `/source/activation/dataset/dataset_caching.py`:

````diff-python
diff --git asource/activation/dataset/dataset_caching.py bworkspace/activation/dataset/dataset_caching.py
index 6a2763c..931e70d 100644
--- asource/activation/dataset/dataset_caching.py
+++ bworkspace/activation/dataset/dataset_caching.py
@@ -1,12 +1,73 @@
 """
-Very simple caching.
-For now caches oracle generation steps.
-As in the study generation steps can be bypassed by reading from a cache.
-@AI: Don't implement sophisticated checks: if the tester gives a stable id, return whatever is found.
-The goal is not hyperefficient caching. More like: don't repeat 1h processes if it can be avoided.
-In addition to caching_id, dataset_id, model_id and likely num_samples should be used to identify an item.
-If subsample-level caching is easy (e.g., seed + num_samples is enough) do it too.
-- As in, if num_samples increases from 50k to 100k, the first 50k labelings should be skipped.
-
-Do not cache cheap operations that take barely a few seconds/minute.
-"""
\ No newline at end of file
+Very simple caching of the oracle study steps (generation and labeling), so an hour-long pass is
+not repeated when a run is restarted or a later run asks for the same or fewer samples.
+
+One JSONL file per key (caching id, dataset, model, kind, seed; plus the label model for labels),
+rows in generation order. A request for n samples is served from the first n rows when the file
+holds at least n: the study sampler is a seeded shuffle, so the first n chunks of a larger request
+are the same chunks. A request beyond the cached rows reuses them and generates the tail, which is
+appended. Labels are keyed by example id. Nothing is checked beyond the key and, for generation,
+the chunk id of each row; a stable caching id from the caller is trusted. Cheap steps (programmatic
+examples, selection) are not cached.
+"""
+import json
+import re
+from dataclasses import dataclass
+from pathlib import Path
+
+
+@dataclass(frozen=True)
+class StudyCacheKey:
+    caching_id: str
+    dataset_id: str
+    model_id: str
+    kind: str                       # "qa", "description", "labels"
+    seed: int
+    label_model_id: str | None = None
+
+    def file_name(self) -> str:
+        parts = [self.caching_id, self.dataset_id, self.kind, self.model_id, f"seed{self.seed}"]
+        if self.label_model_id:
+            parts.append(self.label_model_id)
+        return "__".join(re.sub(r"[^0-9A-Za-z_.-]+", "_", part) for part in parts) + ".jsonl"
+
+
+class StudyCache:
+    """JSONL rows under a root folder; `None` root disables every method (reads return None, appends are no-ops)."""
+
+    def __init__(self, root: str | Path | None):
+        self.root = Path(root) if root else None
+        if self.root is not None:
+            self.root.mkdir(parents=True, exist_ok=True)
+
+    def path(self, key: StudyCacheKey) -> Path | None:
+        return None if self.root is None else self.root / key.file_name()
+
+    def read(self, key: StudyCacheKey) -> list[dict]:
+        path = self.path(key)
+        if path is None or not path.exists():
+            return []
+        rows = []
+        with open(path) as handle:
+            for line in handle:
+                line = line.strip()
+                if line:
+                    try:
+                        rows.append(json.loads(line))
+                    except json.JSONDecodeError:
+                        break                                                      # a torn last line from a killed run
+        return rows
+
+    def append(self, key: StudyCacheKey, rows: list[dict]) -> None:
+        """Flushed per call, so a killed run keeps the rows it produced."""
+        path = self.path(key)
+        if path is None or not rows:
+            return
+        with open(path, "a") as handle:
+            for row in rows:
+                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
+            handle.flush()
+
+    def read_by_id(self, key: StudyCacheKey, id_field: str) -> dict[str, dict]:
+        """Rows keyed by one field (labels by example id); a later row for the same id wins."""
+        return {row[id_field]: row for row in self.read(key) if id_field in row}
````

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">activation/dataset/dataset_manager.py</span>
    <span class="card-oneliner">Entry points: description and programmatic synthesis, `origins` and `caching_id` on labeling and selection, `select_testing_data`.</span>
    <span class="card-badge">Diff</span>
  </summary>

Exact delta vs `/source/activation/dataset/dataset_manager.py`:

````diff-python
diff --git asource/activation/dataset/dataset_manager.py bworkspace/activation/dataset/dataset_manager.py
index ff17f45..84a0066 100644
--- asource/activation/dataset/dataset_manager.py
+++ bworkspace/activation/dataset/dataset_manager.py
@@ -56,98 +56,54 @@ class DatasetManager:
 
     
     def synthesize_study_examples_qa(self, dataset_id: str, num_samples: int, caching_id: str|None=None):
-        """Synthesize study examples"""
+        """Synthesize study questions (one hard question with its answer per sampled chunk)."""
         study_generator = self._get_or_create_study_generator(dataset_id)
-        return study_generator.generate_examples_qa(num_samples)
+        return study_generator.generate_examples_qa(num_samples, caching_id)
 
-    def synthesize_study_examples_description(self, args): # @AI: Implement me. includes caching id.
-        pass
+    def synthesize_study_examples_description(self, dataset_id: str, num_samples: int, caching_id: str|None=None):
+        """Synthesize study descriptions (one search-friendly description per sampled chunk that needs one)."""
+        study_generator = self._get_or_create_study_generator(dataset_id)
+        return study_generator.generate_examples_descriptions(num_samples, caching_id)
+
+    def synthesize_programmatic_examples(self, dataset_id: str, num_samples: int):
+        """Programmatic examples: a span of a chunk as the query, the chunk as its positive; no model."""
+        study_generator = self._get_or_create_study_generator(dataset_id)
+        return study_generator.generate_programmatic_examples(num_samples)
 
     def label_study_examples(
         self,
         dataset_id: str,
         num_samples: int,
-        origins: list[str]|None, # None=all origins. @AI: Propagate this.
-        caching_id: str|None = None
+        origins: list[DataOrigin]|None = None, # None = all origins.
+        caching_id: str|None = None,
+        label_model_name: str|None = None,
     ):
         study_generator = self._get_or_create_study_generator(dataset_id)
-        return study_generator.generate_examples_labels(num_samples, synthetic_only)
-
+        return study_generator.generate_examples_labels(num_samples, origins, caching_id, label_model_name)
 
     def select_training_data(
         self,
         dataset_id: str,
         num_samples: int,
-        origins: list[str]|None, # @AI Also propagate.
+        origins: list[DataOrigin]|None = None, # None = all origins, programmatic included.
         oracle_labeled_only: bool = True,
         val_ratio: float = 0.1,
         max_reporting_size: int = 50,
         force_partition: bool = True, # Most datasets only have training. This forces a val set.
         seed: int = 0,
     ) -> tuple[list[LabeledRetrievalQAExample], list[LabeledRetrievalQAExample], list[LabeledRetrievalQAExample]]:
-        """
-        Select training data. Returns tuples with the following:
-        - training data: actually used to update the gradients.
-        - validation: small amount of data for validation (~10% in general).
-        - reporting: trivial amount of data (subset of the validation set). Used for plotting.
-        There is a general min of 10 for training and validation, and 1 for reporting regardless of the fractions.
-
-        num_samples examples are selected before the split; validation is carved from them by
-        val_ratio unless force_partition is False and the dataset carries native validation-split
-        examples. Oracle-labeled examples are used as they are. Without the oracle, an example
-        inherits chunk labels from its document-level labels: one chunk per positive document and,
-        where the dataset carries native hard negatives, one chunk per hard-negative document.
-        Examples without a positive chunk are dropped and counted in the dataset stats.
-        """
-        # @AI: Move into dataset_study.py.
-        # Double-check: either qa-gen alone, or description-gen alone is enough to re-use 
-        loaded_dataset = self.loaded_datasets[dataset_id]
-        stats = loaded_dataset.stats
-        candidates = [
-            example
-            for example in loaded_dataset.labeled_retrieval_examples.values()
-            if (not synthetic_only or example.origin == DataOrigin.SYNTHETIC)
-            and (not oracle_labeled_only or example.oracle_labeled)
-        ]
+        """Training / validation / reporting examples; see DatasetStudyGenerator.select_training_data."""
         study_generator = self._get_or_create_study_generator(dataset_id)
-        selected: list[LabeledRetrievalQAExample] = []
-        for example in candidates:
-            if not example.oracle_labeled and not example.positive_chunk_ids:
-                study_generator._inherit_labels(example, pool=[])
-            if example.positive_chunk_ids:
-                selected.append(example)
-            else:
-                stats.training_select_num_dropped_no_positive += 1
-        rng = random.Random(seed)
-        native_validation = [example for example in selected if example.split == DataSplit.VAL]
-        if not force_partition and native_validation:
-            training_pool = [example for example in selected if example.split != DataSplit.VAL]
-            rng.shuffle(training_pool)
-            rng.shuffle(native_validation)
-            training_data = training_pool[:num_samples]
-            validation_data = native_validation[:max(10, round(num_samples * val_ratio))]
-        else:
-            rng.shuffle(selected)
-            chosen = selected[:num_samples]
-            num_validation = max(10, round(len(chosen) * val_ratio))
-            validation_data = chosen[:num_validation]
-            training_data = chosen[num_validation:]
-        assert len(training_data) >= 10, (
-            f"{dataset_id} - Only {len(training_data)} training examples after the split; need at least 10 "
-            f"({len(selected)} selectable, {stats.training_select_num_dropped_no_positive} dropped without a positive chunk)."
+        return study_generator.select_training_data(
+            num_samples, origins, oracle_labeled_only, val_ratio, max_reporting_size, force_partition, seed,
         )
-        assert len(validation_data) >= 10, f"{dataset_id} - Only {len(validation_data)} validation examples; need at least 10."
-        reporting_data = validation_data[:max(1, min(max_reporting_size, len(validation_data)))]
-        print(
-            f"{dataset_id} - Selected {len(training_data)} training / {len(validation_data)} validation / "
-            f"{len(reporting_data)} reporting examples."
-        )
-        return training_data, validation_data, reporting_data
-
 
     def select_testing_data(
         self,
         dataset_id: str,
         num_samples: int|None = None, # None=all
-    ):
-        pass # @AI: Implement me.
\ No newline at end of file
+        seed: int = 0,
+    ) -> list[LabeledRetrievalQAExample]:
+        """The native test-split examples with inherited chunk labels; see DatasetStudyGenerator.select_testing_data."""
+        study_generator = self._get_or_create_study_generator(dataset_id)
+        return study_generator.select_testing_data(num_samples, seed)
````

</details>

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">activation/harness/runtime_config.py</span>
    <span class="card-oneliner">Your knobs (programmatic span bounds, `cache_storage_dir`); unchanged by me.</span>
    <span class="card-badge">Unchanged</span>
  </summary>

No change to `activation/harness/runtime_config.py`.

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">activation/harness/vllm_wrapper.py</span>
    <span class="card-oneliner">Engine and chat recommendations for the 4B FP8 generator (256 sequences, thinking off).</span>
    <span class="card-badge">Diff</span>
  </summary>

Exact delta vs `/source/activation/harness/vllm_wrapper.py`:

````diff-python
diff --git asource/activation/harness/vllm_wrapper.py bworkspace/activation/harness/vllm_wrapper.py
index d155dc7..65f22cc 100644
--- asource/activation/harness/vllm_wrapper.py
+++ bworkspace/activation/harness/vllm_wrapper.py
@@ -59,6 +59,8 @@ class VLLMWrapper:
             # A3B MoE labeler: MTP taxes prefill, and labeling is prefill-bound.
             "nvidia/Qwen3.6-35B-A3B-NVFP4": base,
             "Qwen/Qwen3.6-35B-A3B-FP8": base,
+            # Small dense generator for cheap synthetic passes (descriptions, questions): 256 sequences measured best.
+            "RedHatAI/Qwen3.5-4B-FP8-dynamic": base | {"max_num_seqs": 256},
         }
         if model_id not in known:
             print(f"{model_id} - No recommended engine kwargs; using the base formula without speculative decoding.")
@@ -87,6 +89,8 @@ class VLLMWrapper:
             "Qwen/Qwen3.8-27B-FP8": qwen_non_thinking,
             "nvidia/Qwen3.6-35B-A3B-NVFP4": qwen_non_thinking,
             "Qwen/Qwen3.6-35B-A3B-FP8": qwen_non_thinking,
+            # Qwen3.5 thinks by default; the study prompts want the direct JSON answer.
+            "RedHatAI/Qwen3.5-4B-FP8-dynamic": qwen_non_thinking | {"chat_template_kwargs": {"enable_thinking": False}},
         }
         if model_id not in known:
             print(f"{model_id} - No recommended chat kwargs; using engine defaults.")
````

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">activation/common/data_syncing.py</span>
    <span class="card-oneliner">`resolve_path`: the synced folder, local here and `~/activation_artifacts/SYNC` on a node.</span>
    <span class="card-badge">Diff</span>
  </summary>

Exact delta vs `/source/activation/common/data_syncing.py`:

````diff-python
diff --git asource/activation/common/data_syncing.py bworkspace/activation/common/data_syncing.py
index 611b5bd..877c41c 100644
--- asource/activation/common/data_syncing.py
+++ bworkspace/activation/common/data_syncing.py
@@ -1,16 +1,36 @@
 """
-Helper class for coarse-grained but effective data syncing. Uses env vars to detect where it's running.
-Simple:
-- Always on for exec_cmd --sync
-- bidirectionally rsyncs IB/TMP/SYNC/, even if it contains large jsonl files.
-    - The remote folder can be the existing ~/activation_artifacts, it does not matter much.
-    - Final rsync when command is done.
-- When, for example, the oracle caching requires:
-    - data_sync.resolve_path(BRIGHT_ECON_ABLATION/CACHE/), writes to it, and those are propagated here to my machine.
-    - a new exec_cmd in a new node gets it too. Not just setup.
-- Same with reports.
-- When a run is fully local, resolves to the same path as given, and has no rsync watcher.
-- Replace watch with this: a simple sync to potentially restart orphaned syncing.
+Coarse-grained data syncing between this machine and a Sky node.
+
+One folder, `IB/TMP/SYNC/`, is the synced working area: `uv run sky exec --sync` pushes it to the
+node before the job (as `~/activation_artifacts/SYNC/`), pulls it back every few seconds while the
+job runs and once more when the job ends, so caches and reports written on the node land here as
+they are written and a later job on a fresh node starts with everything a previous one wrote.
+Code never cares where it runs: `resolve_path("BRIGHT_ECON_ABLATION/CACHE")` is the local folder on
+this machine and the remote folder on a node (the Sky wrapper exports `ACTIVATION_SYNC_ROOT`).
+A fully local run resolves to the local folder and has no watcher.
+"""
+import os
+from pathlib import Path
+
+SYNC_ROOT_ENV = "ACTIVATION_SYNC_ROOT"
+"""Set by the Sky wrapper on the node; unset here."""
+
+PROJECT_ROOT = Path(__file__).resolve().parents[2]
+LOCAL_SYNC_ROOT = PROJECT_ROOT / "IB" / "TMP" / "SYNC"
+REMOTE_SYNC_ROOT = "/root/activation_artifacts/SYNC"
+
+
+def sync_root() -> Path:
+    """The synced folder on this machine: the local root here, the artifacts folder on a node."""
+    return Path(os.environ.get(SYNC_ROOT_ENV) or LOCAL_SYNC_ROOT)
 
 
-"""
\ No newline at end of file
+def resolve_path(relative: str, create: bool = True) -> Path:
+    """A folder (or file path) under the synced area, created on request; `relative` may not escape it."""
+    relative_path = Path(relative)
+    if relative_path.is_absolute() or ".." in relative_path.parts:
+        raise ValueError(f"sync paths are relative to the synced folder: {relative!r}")
+    resolved = sync_root() / relative_path
+    if create:
+        (resolved if not resolved.suffix else resolved.parent).mkdir(parents=True, exist_ok=True)
+    return resolved
````

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">activation/cloud/sky.py</span>
    <span class="card-oneliner">`exec --sync` (push, pull loop, final pull, `ACTIVATION_SYNC_ROOT`), `exec --watch`, `sync` command.</span>
    <span class="card-badge">Diff</span>
  </summary>

Exact delta vs `/source/activation/cloud/sky.py`:

````diff-python
diff --git asource/activation/cloud/sky.py bworkspace/activation/cloud/sky.py
index 342f489..921adfe 100644
--- asource/activation/cloud/sky.py
+++ bworkspace/activation/cloud/sky.py
@@ -14,6 +14,7 @@ import os
 from pathlib import Path
 import re
 import shlex
+import threading
 import shutil
 import socket
 import subprocess
@@ -68,6 +69,9 @@ REMOTE_WORKDIR = "/root/sky_workdir"
 REMOTE_ARTIFACTS = "/root/activation_artifacts"
 PUBLIC_REMOTE_ARTIFACTS = "~/activation_artifacts"
 LOCAL_DOWNLOAD_ROOT = PROJECT_ROOT / "IB" / "TMP"
+LOCAL_SYNC_ROOT = LOCAL_DOWNLOAD_ROOT / "SYNC"
+REMOTE_SYNC_ROOT = f"{REMOTE_ARTIFACTS}/SYNC"
+SYNC_ROOT_ENV = "ACTIVATION_SYNC_ROOT"
 ECR_REGISTRY = re.compile(
     r"^[0-9]+\.dkr\.ecr\.(?P<region>[a-z0-9-]+)\.amazonaws\.com$"
 )
@@ -614,7 +618,50 @@ def download(
     return _run_rsync(*args).returncode
 
 
-# @AI: Double-check this works with directories too.
+def _sync_rsync_args() -> list[str]:
+    return ["-azu", "--itemize-changes", "--protect-args", "--no-owner", "--no-group", "--chmod=D755,F644"]
+
+
+def _push_sync(record: dict) -> None:
+    """Local IB/TMP/SYNC -> the node's SYNC folder; update-only, so files the node wrote more recently survive."""
+    LOCAL_SYNC_ROOT.mkdir(parents=True, exist_ok=True)
+    result = subprocess.run(["rsync", *_sync_rsync_args(), "--rsync-path", f"mkdir -p {REMOTE_SYNC_ROOT} && rsync",
+                             f"{LOCAL_SYNC_ROOT}/", f"{record['name']}:{REMOTE_SYNC_ROOT}/"], text=True, capture_output=True)
+    if result.returncode != 0:
+        raise RuntimeError(f"sync push failed: {result.stderr.strip().splitlines()[-1] if result.stderr.strip() else result.returncode}")
+    print(f"Synced {LOCAL_SYNC_ROOT} -> {record['name']}:{REMOTE_SYNC_ROOT}", flush=True)
+
+
+def _pull_once(record: dict, pairs: list[tuple[str, Path]], quiet: bool = False) -> None:
+    """Each (remote folder, local folder) pair pulled once; changed files replaced, nothing deleted locally."""
+    for source, destination in pairs:
+        destination.mkdir(parents=True, exist_ok=True)
+        result = subprocess.run(["rsync", *_sync_rsync_args(), f"{record['name']}:{source}", str(destination)], text=True, capture_output=True)
+        stamp = time.strftime("%H:%M:%S")
+        changed = [line for line in result.stdout.splitlines() if line[:1] in ("<", ">", "c")]
+        if result.returncode != 0:
+            detail = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else ""
+            if "No such file" not in detail and not quiet:
+                print(f"[{stamp}] rsync exit {result.returncode}: {detail}", flush=True)
+        elif changed and not quiet:
+            names = ", ".join(line.split()[-1] for line in changed[:6]) + (" ..." if len(changed) > 6 else "")
+            print(f"[{stamp}] {len(changed)} file(s) updated: {names}", flush=True)
+
+
+def _pull_loop(record: dict, pairs: list[tuple[str, Path]], interval_seconds: float, stop: threading.Event) -> None:
+    while not stop.wait(interval_seconds):
+        _pull_once(record, pairs)
+
+
+def sync(name: str) -> int:
+    """One push and one pull of IB/TMP/SYNC, e.g. after a watcher died with its session."""
+    record, _, _ = _managed_record(name)
+    _start_if_stopped(record)
+    _push_sync(record)
+    _pull_once(record, [(f"{REMOTE_SYNC_ROOT}/", LOCAL_SYNC_ROOT)])
+    return 0
+
+
 def watch(
     name: str,
     remote_path: str,
@@ -668,8 +715,21 @@ def watch(
         return 0
 
 
-# @AI: make this more practical by allow --watch  etc.
-def exec_cmd(name: str, cmd: str | list[str]) -> int:
+def exec_cmd(
+    name: str,
+    cmd: str | list[str],
+    *,
+    sync: bool = False,
+    watch_paths: tuple[str, str] | None = None,
+    interval_seconds: float = 15.0,
+) -> int:
+    """
+    Upload the mirror and run the command. With sync, IB/TMP/SYNC is pushed to the node first
+    (update-only), pulled back every interval while the command runs and once more when it ends,
+    and the command sees ACTIVATION_SYNC_ROOT so `data_syncing.resolve_path` lands in that folder.
+    watch_paths (remote folder under ~/activation_artifacts, local folder under IB/TMP) adds one
+    more pulled pair, replacing a separate `watch` session.
+    """
     command = [cmd] if isinstance(cmd, str) else cmd.copy()
     if command[:1] == ["--"]:
         command = command[1:]
@@ -679,11 +739,21 @@ def exec_cmd(name: str, cmd: str | list[str]) -> int:
     record, gpu_type, gpu_count = _managed_record(name)
     _start_if_stopped(record)
     _upload_record(record)
+    pairs: list[tuple[str, Path]] = []
+    if sync:
+        _push_sync(record)
+        pairs.append((f"{REMOTE_SYNC_ROOT}/", LOCAL_SYNC_ROOT))
+    if watch_paths:
+        remote_folder = _remote_artifact_path(watch_paths[0])
+        if not remote_folder.endswith("/"):
+            raise ValueError("watch needs a remote folder (end the remote path with '/')")
+        pairs.append((remote_folder, _local_download_path(watch_paths[1])))
 
     remote_command = shlex.join(
         [
             "env",
             "VLLM_WORKER_MULTIPROC_METHOD=spawn",
+            f"{SYNC_ROOT_ENV}={REMOTE_SYNC_ROOT}",
             # The baked env is a warm cache: sync it to the uploaded lock
             # (old images set UV_NO_SYNC=1) without re-resolving on the node.
             "UV_NO_SYNC=0",
@@ -695,15 +765,27 @@ def exec_cmd(name: str, cmd: str | list[str]) -> int:
     secret_file_args = (
         ["--secret-file", str(ENV_FILE)] if ENV_FILE.is_file() else []
     )
-    return _run_skypilot(
-        "exec",
-        *secret_file_args,
-        "--gpus",
-        f"{GPU_TYPES[gpu_type]}:{gpu_count}",
-        name,
-        "--",
-        remote_command,
-    ).returncode
+    stop = threading.Event()
+    puller = None
+    if pairs:
+        puller = threading.Thread(target=_pull_loop, args=(record, pairs, interval_seconds, stop), daemon=True)
+        puller.start()
+        print(f"Pulling {', '.join(source for source, _ in pairs)} every {interval_seconds:g}s while the command runs.", flush=True)
+    try:
+        return _run_skypilot(
+            "exec",
+            *secret_file_args,
+            "--gpus",
+            f"{GPU_TYPES[gpu_type]}:{gpu_count}",
+            name,
+            "--",
+            remote_command,
+        ).returncode
+    finally:
+        if puller is not None:
+            stop.set()
+            puller.join()
+            _pull_once(record, pairs)
 
 
 def _positive_int(value: str) -> int:
@@ -755,9 +837,17 @@ def _build_parser() -> argparse.ArgumentParser:
         "exec",
         help="upload current code and run a command on a node",
     )
+    exec_parser.add_argument("--sync", action="store_true",
+                             help="mirror IB/TMP/SYNC to the node before, during and after the command (flags go before the name)")
+    exec_parser.add_argument("--watch", nargs=2, metavar=("REMOTE_FOLDER", "LOCAL_FOLDER"), default=None,
+                             help="also pull a ~/activation_artifacts folder to a local IB/TMP folder while the command runs")
+    exec_parser.add_argument("--interval", type=float, default=15.0, help="seconds between pulls (default 15)")
     exec_parser.add_argument("name", help="cluster name")
     exec_parser.add_argument("command", nargs=argparse.REMAINDER)
 
+    sync_parser = commands.add_parser("sync", help="push and pull IB/TMP/SYNC once (e.g. after a watcher died)")
+    sync_parser.add_argument("name", help="cluster name")
+
     upload_parser = commands.add_parser(
         "upload",
         help="exactly mirror local code into the remote workdir",
@@ -832,7 +922,10 @@ def main() -> int:
                 rebuild_image=not args.no_image_rebuild,
             )
         if args.action == "exec":
-            return exec_cmd(args.name, args.command)
+            return exec_cmd(args.name, args.command, sync=args.sync, watch_paths=tuple(args.watch) if args.watch else None,
+                            interval_seconds=args.interval)
+        if args.action == "sync":
+            return sync(args.name)
         if args.action == "upload":
             return upload(args.name, dry_run=args.dry_run)
         if args.action == "download":
````

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">activation/retrieval/retrieval_ac.py</span>
    <span class="card-oneliner">Prefix-plus-view rows: P + V learned queries, `num_rows`, defaults 1024 wide and 8 layers.</span>
    <span class="card-badge">Diff</span>
  </summary>

Exact delta vs `/source/activation/retrieval/retrieval_ac.py`:

````diff-python
diff --git asource/activation/retrieval/retrieval_ac.py bworkspace/activation/retrieval/retrieval_ac.py
index f7fa974..0b6808f 100644
--- asource/activation/retrieval/retrieval_ac.py
+++ bworkspace/activation/retrieval/retrieval_ac.py
@@ -103,19 +103,20 @@ class WindowedBytePooling(nn.Module):
 
 class StandardRetrievalACModel(nn.Module):
     """
-    Byte-level side encoder producing V view rows in the base model's input space.
+    Byte-level side encoder producing P prefix rows and V view rows in the base model's input space.
     Byte table 256 x d_ac -> positional encoding -> windowed attention (window 8, stride 4) pooled to
-    one vector per window -> num_ac_layers bidirectional pre-norm layers (FFN mult 4) -> V learned
-    queries cross-attend the sequence -> linear d_ac -> d_model.
+    one vector per window -> num_ac_layers bidirectional pre-norm layers (FFN mult 4) -> P + V learned
+    queries cross-attend the sequence -> linear d_ac -> d_model. The prefix rows go before the text
+    tokens (a soft prompt written from the bytes), the view rows after EOS (the readout).
     """
     def __init__(
         self,
         harness: "HarnessRuntime",
         base_model_name: str,
-        d_ac_model: int | None = None,      # None: d_model // 4
-        num_prefix_tokens: int | None = 16, # @AI: This now becomes the prompt prefix.
-        num_view_tokens: int = 8,
-        num_ac_layers: int | None = None,   # None: base layer count // 4
+        d_ac_model: int | None = None,      # None: 1024
+        num_prefix_tokens: int = 16,        # rows placed before the text tokens
+        num_view_tokens: int = 8,           # rows placed after EOS, the readout
+        num_ac_layers: int | None = None,   # None: 8
     ):
         super().__init__()
         description = harness.loaded_models[base_model_name].model_config.model_description
@@ -123,6 +124,7 @@ class StandardRetrievalACModel(nn.Module):
         self.base_model_name = base_model_name
         self.d_model = description.d_model
         self.d_ac_model = d_ac_model or 1024
+        self.num_prefix_tokens = num_prefix_tokens
         self.num_view_tokens = num_view_tokens
         self.num_ac_layers = num_ac_layers or 8
         self.input_limit_chars = harness.harness_config.doc_embedding_input_limit_chars
@@ -141,7 +143,7 @@ class StandardRetrievalACModel(nn.Module):
             for _ in range(self.num_ac_layers)
         ])
         self.final_norm = nn.LayerNorm(self.d_ac_model)
-        self.view_queries = nn.Parameter(torch.randn(num_view_tokens, self.d_ac_model) * 0.02) # @AI: P+V
+        self.row_queries = nn.Parameter(torch.randn(num_prefix_tokens + num_view_tokens, self.d_ac_model) * 0.02)  # P then V
         self.view_attention = nn.MultiheadAttention(self.d_ac_model, num_heads, dropout=dropout, batch_first=True)
         self.view_norm = nn.LayerNorm(self.d_ac_model)
         self.output_projection = nn.Linear(self.d_ac_model, self.d_model)
@@ -151,6 +153,11 @@ class StandardRetrievalACModel(nn.Module):
     def device(self) -> torch.device:
         return self.byte_embedding.weight.device
 
+    @property
+    def num_rows(self) -> int:
+        """Rows added to a sequence: P prefix rows plus V view rows."""
+        return self.num_prefix_tokens + self.num_view_tokens
+
     def encode_bytes(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
         """UTF-8 byte ids [B, L] (right-padded) and mask [B, L] of the char-limited texts."""
         encoded = [
@@ -166,10 +173,7 @@ class StandardRetrievalACModel(nn.Module):
         return ids.to(self.device), mask.to(self.device)
 
     def forward(self, texts: list[str]) -> torch.Tensor:
-        """
-        [B, V, d_model] view rows, one group per text.
-        @AI: This should now be P+V (prefix + view).
-        """
+        """[B, P + V, d_model]: the P prefix rows then the V view rows, one group per text."""
         ids, mask = self.encode_bytes(texts)
         x = self.byte_embedding(ids)
         x = x + sinusoidal_positions(x.shape[1], x.shape[2], x.device, x.dtype)[None]
@@ -182,9 +186,9 @@ class StandardRetrievalACModel(nn.Module):
             else:
                 x = layer(x, src_key_padding_mask=key_padding)
         x = self.final_norm(x)
-        queries = self.view_queries[None].expand(x.shape[0], -1, -1)
+        queries = self.row_queries[None].expand(x.shape[0], -1, -1)
         views, _ = self.view_attention(queries, x, x, key_padding_mask=key_padding, need_weights=False)
-        rows = self.output_projection(self.view_norm(queries + views))                       # [B, V, d_model]
+        rows = self.output_projection(self.view_norm(queries + views))                       # [B, P + V, d_model]
         rows = rows * torch.rsqrt(rows.float().pow(2).mean(dim=-1, keepdim=True) + 1e-6).to(rows.dtype)
         return rows * self.output_scale
````

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">activation/retrieval/retrieval_model.py</span>
    <span class="card-oneliner">Document instruction; sequence [prefix rows][instruction + text][EOS][view rows] with per-row prefix placement.</span>
    <span class="card-badge">Diff</span>
  </summary>

Exact delta vs `/source/activation/retrieval/retrieval_model.py`:

````diff-python
diff --git asource/activation/retrieval/retrieval_model.py bworkspace/activation/retrieval/retrieval_model.py
index 4e874e9..4ead28f 100644
--- asource/activation/retrieval/retrieval_model.py
+++ bworkspace/activation/retrieval/retrieval_model.py
@@ -2,10 +2,11 @@
 Retrieval model: a frozen base LLM with an optional LoRA and an optional activation-context (AC)
 model, plus a projection head, embedding queries and documents through the same path.
 
-Sequence layout: text tokens, the EOS token, then the V view rows from the AC model, all
-left-padded so the readout is always the last V positions (the last position alone without an AC
-model). Scores between a query and a candidate use MaxSim over their V vectors, which is the plain
-dot product when V = 1.
+Sequence layout: the P prefix rows from the AC model, the instruction and text tokens, the EOS
+token, then the V view rows from the AC model, all left-padded so the readout is always the last V
+positions (the last position alone without an AC model; P = 0 then too). Queries and documents each
+get their own instruction prefix. Scores between a query and a candidate use MaxSim over their V
+vectors, which is the plain dot product when V = 1.
 """
 import typing as t
 from contextlib import nullcontext
@@ -23,7 +24,7 @@ if t.TYPE_CHECKING:
 
 
 DEFAULT_QUERY_INSTRUCTION = "Instruct: Retrieve passages that answer the question\nQuery: "
-# @AI: I think there should be a DOCUMENT_INSTRUCTION like Summarize this document or something like that.
+DEFAULT_DOCUMENT_INSTRUCTION = "Instruct: Summarize this document for retrieval\nDocument: "
 
 
 class RetrievalModel(nn.Module):
@@ -37,6 +38,7 @@ class RetrievalModel(nn.Module):
         ac_name: str | None,                # None: no AC, V = 1 EOS readout
         lora_name: str | None,              # None: no LoRA (base frozen; only AC + head train)
         query_instruction: str = DEFAULT_QUERY_INSTRUCTION,
+        document_instruction: str = DEFAULT_DOCUMENT_INSTRUCTION,
     ):
         super().__init__()
         self.harness = harness
@@ -55,6 +57,7 @@ class RetrievalModel(nn.Module):
         self.d_embedding_result = d_embedding_result or self.d_model
         self.head = nn.Linear(self.d_model, d_embedding_result, bias=False) if d_embedding_result else None
         self.query_instruction = query_instruction
+        self.document_instruction = document_instruction
         self.input_limit_chars = harness.harness_config.doc_embedding_input_limit_chars
         self.eos_token_id = self.loaded_model.tokenizer.convert_tokens_to_ids(description.eos_token)
         self.gradient_checkpointing = False
@@ -106,45 +109,52 @@ class RetrievalModel(nn.Module):
 
     def embed_batch(self, texts: list[str], is_query: bool) -> torch.Tensor:
         """
-        [B, V, d_embedding_result], each vector L2-normalized. Queries get the instruction
-        prefix, documents are raw. Gradients flow in training mode.
+        [B, V, d_embedding_result], each vector L2-normalized. Queries get the query instruction,
+        documents the document instruction. With an AC model the sequence is
+        [pad][P prefix rows][instruction + text][EOS][V view rows] per row, left-padded, so the
+        readout is the last V positions. Gradients flow in training mode.
         """
         texts = [safe_truncate_embedding_chunk(text, self.input_limit_chars) for text in texts]
-        if is_query:
-            texts = [self.query_instruction + text for text in texts] # @AI: Distinguish query and doc?
+        instruction = self.query_instruction if is_query else self.document_instruction
+        texts = [instruction + text for text in texts]
         tokenizer = self.loaded_model.tokenizer
         token_rows = tokenizer(texts, add_special_tokens=False, padding=False)["input_ids"]
         token_rows = [row + [self.eos_token_id] for row in token_rows]
         device = self.device
         embedding_layer = self.loaded_model.embedding_layer
         base_dtype = embedding_layer.weight.dtype
-        num_views = self.num_vectors if self.ac_model is not None else 0
-        length = max(len(row) for row in token_rows) + num_views
+        num_prefix, num_views = (self.ac_model.num_prefix_tokens, self.num_vectors) if self.ac_model is not None else (0, 0)
+        length = num_prefix + max(len(row) for row in token_rows) + num_views
         batch_size = len(texts)
         if self.view_scale is None:                                                     # RMS of a token embedding row
             self.view_scale = float(embedding_layer.weight.detach().float().pow(2).mean().sqrt().item())
-        padded_ids = torch.full((batch_size, length - num_views), self.eos_token_id, dtype=torch.long)
+        padded_ids = torch.full((batch_size, length), self.eos_token_id, dtype=torch.long)  # prefix/view positions are replaced below
         attention_mask = torch.zeros(batch_size, length, dtype=torch.long)
+        prefix_positions = torch.zeros(batch_size, num_prefix, dtype=torch.long)
         for index, row in enumerate(token_rows):
-            padded_ids[index, length - num_views - len(row):] = torch.tensor(row, dtype=torch.long)
-            attention_mask[index, length - num_views - len(row):] = 1
+            start = length - num_views - len(row)
+            padded_ids[index, start:start + len(row)] = torch.tensor(row, dtype=torch.long)
+            attention_mask[index, start - num_prefix:] = 1
+            if num_prefix:
+                prefix_positions[index] = torch.arange(start - num_prefix, start)
         padded_ids, attention_mask = padded_ids.to(device), attention_mask.to(device)
 
         with self._autocast():
-            # @AI: Handle prefix and view here.
-            view_rows = None
-            if self.ac_model is not None:                                               # [B, V, d_model] at embedding scale
-                view_rows = (self.ac_model(texts) * self.view_scale).to(base_dtype)
-            inputs_embeds = embedding_layer(padded_ids)
-            if view_rows is not None:
+            inputs_embeds = embedding_layer(padded_ids)                                 # padding rows are masked out
+            if self.ac_model is not None:                                               # [B, P + V, d_model] at embedding scale
+                rows = (self.ac_model(texts) * self.view_scale).to(base_dtype)
+                prefix_rows, view_rows = rows[:, :num_prefix], rows[:, num_prefix:]
                 inputs_embeds = torch.cat([inputs_embeds[:, :length - num_views], view_rows], dim=1)
+                if num_prefix:                                                          # each row's prefix sits right before its first token
+                    index = prefix_positions.to(device)[:, :, None].expand(-1, -1, inputs_embeds.shape[-1])
+                    inputs_embeds = inputs_embeds.scatter(1, index, prefix_rows)
             position_ids = (attention_mask.cumsum(dim=1) - 1).clamp_min(0)
             hidden = self.loaded_model.decoder_forward(inputs_embeds, attention_mask, position_ids, self.lora_name or None)  # [B, S, d_model]
             readout = hidden[:, -self.num_vectors:, :]
         readout = readout.float()
         if self.head is not None:
             readout = self.head(readout)
-        self.real_tokens_embedded += sum(len(row) for row in token_rows) + batch_size * num_views   # = mask sum, no GPU sync
+        self.real_tokens_embedded += sum(len(row) for row in token_rows) + batch_size * (num_prefix + num_views)  # = mask sum, no GPU sync
         self.padded_tokens_embedded += batch_size * length
         self.forwards_embedded += 1
         return F.normalize(readout, p=2, dim=-1)
````

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">activation/retrieval/retrieval_batching.py</span>
    <span class="card-oneliner">Length estimates count P + V rows and the document instruction; no input limit tolerated.</span>
    <span class="card-badge">Diff</span>
  </summary>

Exact delta vs `/source/activation/retrieval/retrieval_batching.py`:

````diff-python
diff --git asource/activation/retrieval/retrieval_batching.py bworkspace/activation/retrieval/retrieval_batching.py
index cdf11d1..94871a2 100644
--- asource/activation/retrieval/retrieval_batching.py
+++ bworkspace/activation/retrieval/retrieval_batching.py
@@ -106,9 +106,14 @@ def flatten_candidates(batch: RetrievalBatch) -> tuple[list[DatasetDocumentChunk
     return chunks, per_example
 
 
-def estimated_tokens(text_length: int, num_views: int) -> int:
-    """Padded-length estimate shared by grouping and sizing: characters / 4, plus EOS and the view rows."""
-    return text_length // CHARS_PER_TOKEN + 1 + num_views
+def estimated_tokens(text_length: int, num_rows: int) -> int:
+    """Padded-length estimate shared by grouping and sizing: characters / 4, plus EOS and the AC rows (prefix + views)."""
+    return text_length // CHARS_PER_TOKEN + 1 + num_rows
+
+
+def ac_rows(retrieval_model) -> int:
+    """Rows the AC model adds to every sequence (P prefix + V views), 0 without one."""
+    return retrieval_model.ac_model.num_rows if retrieval_model.ac_model is not None else 0
 
 
 def token_budget_groups(lengths: list[int], budget_tokens: int, num_views: int) -> list[list[int]]:
@@ -131,9 +136,10 @@ def token_budget_groups(lengths: list[int], budget_tokens: int, num_views: int)
 
 
 def embedded_text_length(retrieval_model: "RetrievalModel", text: str, is_query: bool) -> int:
-    """Characters embed_batch will actually see: cut to the input limit, plus the query instruction."""
-    prefix = len(retrieval_model.query_instruction) if is_query else 0
-    return min(len(text), retrieval_model.input_limit_chars) + prefix
+    """Characters embed_batch will actually see: cut to the input limit, plus the query or document instruction."""
+    prefix = len(retrieval_model.query_instruction if is_query else retrieval_model.document_instruction)
+    limit = retrieval_model.input_limit_chars
+    return (len(text) if limit is None else min(len(text), limit)) + prefix
 
 
 def embed_in_length_groups(
@@ -148,7 +154,7 @@ def embed_in_length_groups(
     """
     if budget_tokens is None:
         budget_tokens = FORWARD_TOKEN_BUDGET if retrieval_model.device.type == "cuda" else CPU_FORWARD_TOKEN_BUDGET
-    num_views = retrieval_model.num_vectors if retrieval_model.ac_model is not None else 0
+    num_views = ac_rows(retrieval_model)
     lengths = [embedded_text_length(retrieval_model, text, is_query) for text in texts]
     groups = token_budget_groups(lengths, budget_tokens, num_views)
     embedded = [retrieval_model.embed_batch([texts[index] for index in group], is_query) for group in groups]
@@ -164,7 +170,7 @@ def example_token_counts(
     retrieval_model: "RetrievalModel",
 ) -> list[int]:
     """Per example: estimated padded tokens of its query plus all its candidates, as embed_batch will see them."""
-    num_views = retrieval_model.num_vectors if retrieval_model.ac_model is not None else 0
+    num_views = ac_rows(retrieval_model)
     counts = []
     for example in training_data:
         total = estimated_tokens(embedded_text_length(retrieval_model, example.query, True), num_views)
````

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">activation/retrieval/retrieval_baseline.py</span>
    <span class="card-oneliner">Round 4 file (new to `/source`): baseline embedder and frozen-base reference; documents get no instruction.</span>
    <span class="card-badge">New file</span>
  </summary>

Exact delta vs `/source/activation/retrieval/retrieval_baseline.py`:

````diff-python
diff --git aworkspace/activation/retrieval/retrieval_baseline.py bworkspace/activation/retrieval/retrieval_baseline.py
new file mode 100644
index 0000000..643fa47
--- /dev/null
+++ bworkspace/activation/retrieval/retrieval_baseline.py
@@ -0,0 +1,65 @@
+"""
+Reference embedders for the in-batch metrics: the same validation batches scored by a well-trained
+single-vector embedder (Qwen/Qwen3-Embedding-0.6B by default) and by the frozen base alone, so a
+training curve has a trained reference and a floor even at small scales. Nothing beyond the batch
+is embedded; corpus-level comparisons belong to the slice 2 index.
+"""
+import typing as t
+
+import torch
+
+from ..dataset.dataset_utils import safe_truncate_embedding_chunk
+from ..harness.hf_utils import FREE_DEVICE, TARGET_DEVICE
+from .retrieval_model import DEFAULT_QUERY_INSTRUCTION, RetrievalModel
+
+if t.TYPE_CHECKING:
+    from ..harness import HarnessRuntime
+
+
+class BaselineEmbedder:
+    """
+    A harness embedding model behind the scoring interface of RetrievalModel (embed_batch [B, 1, d],
+    similarity, num_vectors, device, query_instruction, document_instruction, input_limit_chars, ac_model), so the trainer
+    evaluates it with the same batches, loss and metrics as the model under training. Queries get
+    the same instruction prefix (the Qwen3-Embedding "Instruct: ...\\nQuery: " format).
+    """
+    num_vectors = 1
+    ac_model = None
+
+    def __init__(self, harness: "HarnessRuntime", model_name: str, query_instruction: str = DEFAULT_QUERY_INSTRUCTION):
+        self.harness = harness
+        self.model_name = model_name
+        self.loaded_model = harness.loaded_models[model_name]
+        assert self.loaded_model.model_config.model_description.is_embedding_model, f"{model_name} is not an embedding model"
+        self.query_instruction = query_instruction
+        self.document_instruction = ""                                                # Qwen3-Embedding: no instruction on documents
+        self.input_limit_chars = harness.harness_config.doc_embedding_input_limit_chars
+        self.device = torch.device(TARGET_DEVICE)
+        self.real_tokens_embedded = self.padded_tokens_embedded = self.forwards_embedded = 0
+
+    def embed_batch(self, texts: list[str], is_query: bool) -> torch.Tensor:
+        """[B, 1, d], L2-normalized, on the CPU (the harness embedding path returns there)."""
+        texts = [safe_truncate_embedding_chunk(text, self.input_limit_chars) for text in texts]
+        if is_query:
+            texts = [self.query_instruction + text for text in texts]
+        embeddings = self.loaded_model.simple_vector_embed_many(texts)                 # [B, d] normalized
+        self.forwards_embedded += 1
+        return embeddings.float().unsqueeze(1)
+
+    similarity = staticmethod(RetrievalModel.similarity)
+
+    def free(self) -> None:
+        self.loaded_model.model_to_device(FREE_DEVICE)
+
+
+def frozen_base_reference(harness: "HarnessRuntime", retrieval_model: RetrievalModel) -> RetrievalModel:
+    """
+    The base of the model under training with no LoRA, no AC model and no head: the EOS readout of
+    the frozen language model, L2-normalized. It shares the resident base, so it costs no memory.
+    """
+    reference = RetrievalModel(
+        harness, retrieval_model.base_model_name, d_embedding_result=None, ac_name=None, lora_name=None,
+        query_instruction=retrieval_model.query_instruction,
+    )
+    reference.set_training_mode(False)
+    return reference
````

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">activation/retrieval/retrieval_trainer.py</span>
    <span class="card-oneliner">`train` without validation; `eval(model, reporter, eval_data, config, label, progress)` scores a set as one batch with the references.</span>
    <span class="card-badge">Diff</span>
  </summary>

Exact delta vs `/source/activation/retrieval/retrieval_trainer.py`:

````diff-python
diff --git asource/activation/retrieval/retrieval_trainer.py bworkspace/activation/retrieval/retrieval_trainer.py
index 478edc2..1e23799 100644
--- asource/activation/retrieval/retrieval_trainer.py
+++ bworkspace/activation/retrieval/retrieval_trainer.py
@@ -39,6 +39,7 @@ from .retrieval_batching import (
     probe_batch_size,
     recommended_batch_size,
 )
+from .retrieval_baseline import BaselineEmbedder, frozen_base_reference
 from .retrieval_model import RetrievalModel
 from .retrieval_reporter import METRIC_NAMES, RetrievalReporter
 from .retrieval_training_config import RetrievalTrainingConfig, RetrievalTrainingStats
@@ -47,6 +48,11 @@ if t.TYPE_CHECKING:
     from ..harness import HarnessRuntime
 
 
+EVAL_SINGLE_BATCH_MAX = 128
+"""Eval sets up to this many queries are scored as one batch (the pool is every query's candidates); larger
+sets use batches of the reporting size."""
+
+
 class RetrievalTrainer:
     def __init__(self, harness: "HarnessRuntime"):
         self.harness = harness
@@ -124,7 +130,10 @@ class RetrievalTrainer:
 
     @torch.no_grad()
     def _evaluate(self, retrieval_model: RetrievalModel, batches: list[RetrievalBatch], temperature: float) -> tuple[float, dict[str, float]]:
-        """Example-weighted loss and in-batch metrics over the given batches, in eval mode."""
+        """
+        Example-weighted loss and in-batch metrics over the given batches, in eval mode. Any embedder
+        with the RetrievalModel scoring interface works here, including a BaselineEmbedder.
+        """
         total_loss, total_examples = 0.0, 0
         totals = {name: 0.0 for name in METRIC_NAMES}
         for batch in batches:
@@ -137,6 +146,30 @@ class RetrievalTrainer:
         divisor = max(1, total_examples)
         return total_loss / divisor, {name: value / divisor for name, value in totals.items()}
 
+    def evaluate_references(
+        self, config: RetrievalTrainingConfig, retrieval_model: RetrievalModel, batches: list[RetrievalBatch],
+    ) -> dict[str, tuple[float, dict[str, float]]]:
+        """
+        {reference name: (loss, in-batch metrics)} on the given batches for the references the config
+        asks for: the frozen base alone (the floor) and a well-trained baseline embedder (the target).
+        The baseline model is loaded for the pass and freed afterwards; the frozen base shares the
+        resident base. Same batches, pool, loss and metrics as the model under training. Leaves the
+        model in eval mode.
+        """
+        references: dict[str, tuple[float, dict[str, float]]] = {}
+        if not batches:
+            return references
+        if config.reference_frozen_base:
+            reference = frozen_base_reference(self.harness, retrieval_model)
+            references["frozen base"] = self._evaluate(reference, batches, config.temperature)
+        if config.baseline_model_name:
+            baseline = BaselineEmbedder(self.harness, config.baseline_model_name, retrieval_model.query_instruction)
+            try:
+                references[config.baseline_model_name] = self._evaluate(baseline, batches, config.temperature)
+            finally:
+                baseline.free()
+        return references
+
     # ----------------------------------------------------------------------------- training
     def _dataset_indexes(self, *example_lists: list[LabeledRetrievalQAExample]) -> DatasetIndexes:
         dataset_ids = {example.dataset_id for examples in example_lists for example in examples}
@@ -177,21 +210,21 @@ class RetrievalTrainer:
         retrieval_reporter: RetrievalReporter,
         training_data: list[LabeledRetrievalQAExample],
         reporting_data: list[LabeledRetrievalQAExample],
-        validation_data: list[LabeledRetrievalQAExample] | None = None, # @AI: Remove from here. Move to eval. Reporting is enough here.
     ) -> RetrievalTrainingStats:
         """
         Holds base residency for the whole run. Order: ensure_resident -> batch size (config or
-        recommended) -> probe -> epochs. Raises before the first real step if the probe cannot fit.
+        recommended) -> probe -> epoch-0 reporting point -> epochs. Raises before the first real
+        step if the probe cannot fit. Learning feedback is the reporting batch only; held-out sets
+        and the reference embedders are scored by eval().
         """
         config = training_config
         reporter = retrieval_reporter
         stats = RetrievalTrainingStats(num_examples=len(training_data), num_epochs=config.epochs)
-        validation_data = validation_data or []
         assert training_data, "No training data."
         torch.manual_seed(config.seed)
         retrieval_model.ensure_resident()
         device = retrieval_model.device
-        dataset_index = self._dataset_indexes(training_data, reporting_data, validation_data)
+        dataset_index = self._dataset_indexes(training_data, reporting_data)
         groups = retrieval_model.trainable_parameter_groups()
         assert groups, "Nothing to train: no LoRA, AC model or head."
         all_parameters = [parameter for params in groups.values() for parameter in params]
@@ -226,14 +259,12 @@ class RetrievalTrainer:
         optimizer = self._make_optimizer(config, groups)
         scheduler = self._make_scheduler(config, optimizer, total_steps)
         reporting_batches = fixed_batches(reporting_data, dataset_index, None, config.seed)
-        validation_batches = fixed_batches(validation_data, dataset_index, max(1, len(reporting_data)), config.seed)
         counts = {
-            "training_examples": len(training_data), "validation_examples": len(validation_data),
-            "reporting_examples": len(reporting_data), "steps_per_epoch": steps_per_epoch, "total_steps": total_steps,
-            "batch_size": batch_size,
+            "training_examples": len(training_data), "reporting_examples": len(reporting_data),
+            "steps_per_epoch": steps_per_epoch, "total_steps": total_steps, "batch_size": batch_size,
         }
         lora_config = self.harness.module_manager.get_lora_config(retrieval_model.lora_name) if retrieval_model.lora_name else None
-        reporter.initialize_run(config, retrieval_model, lora_config, counts, batch_sizing, groups)
+        reporter.initialize_run(config, retrieval_model, lora_config, counts, batch_sizing, groups, self._reference_names(config))
         if device.type == "cuda":
             torch.cuda.reset_peak_memory_stats(device)
 
@@ -255,6 +286,8 @@ class RetrievalTrainer:
         run_start = time.time()
         step = 0
         last_report_step = -1
+        report_progress(0, 0.0)                                                       # the untrained model: epoch 0 of every curve
+        reporter.render(force=True)
         for epoch in range(1, config.epochs + 1):
             rng = random.Random(config.seed + epoch)
             epoch_start = time.time()
@@ -299,20 +332,11 @@ class RetrievalTrainer:
                     reporter.render(force=True)
                 else:
                     reporter.render()
-            # Epoch end: reporting point, full validation, throughput row.
+            # Epoch end: reporting point, throughput row.
             epoch_time = time.time() - epoch_start
             if last_report_step != step:
                 report_progress(step, step / steps_per_epoch)
                 last_report_step = step
-            if validation_batches:
-                start = time.time()
-                retrieval_model.set_training_mode(False)
-                validation_loss, validation_metrics = self._evaluate(retrieval_model, validation_batches, config.temperature)
-                retrieval_model.set_training_mode(True, config.gradient_checkpointing)
-                stats.total_validation_time += time.time() - start
-                stats.validation_losses.append((epoch, validation_loss))
-                stats.validation_metrics.append((epoch, validation_metrics))
-                reporter.report_validation(epoch, validation_loss, validation_metrics)
             reporter.report_epoch(str(epoch), epoch_steps, epoch_shapes, epoch_time, peak_memory(), running=False)
             reporter.render(force=True)
 
@@ -322,11 +346,38 @@ class RetrievalTrainer:
         reporter.report_finished(step, time.time() - run_start)
         return stats
 
+    @staticmethod
+    def _reference_names(config: RetrievalTrainingConfig) -> list[str]:
+        return (["frozen base"] if config.reference_frozen_base else []) + ([config.baseline_model_name] if config.baseline_model_name else [])
 
     def eval(
         self,
         retrieval_model: RetrievalModel,
         retrieval_reporter: RetrievalReporter,
-        eval_data: list[LabeledRetrievalQAExample] | None = None, # @AI: can be validation data or test data.
-    ):
-        pass # @AI: Implement me.
\ No newline at end of file
+        eval_data: list[LabeledRetrievalQAExample],
+        training_config: RetrievalTrainingConfig,
+        label: str = "eval",
+        progress: float | None = None,
+    ) -> dict[str, tuple[float, dict[str, float]]]:
+        """
+        Score a validation or test set with the in-batch loss and metrics: the model under training
+        ("model") and the reference embedders the config asks for (the frozen base, a baseline
+        embedder), all on the same batches. Up to EVAL_SINGLE_BATCH_MAX queries form one batch, so
+        every query is ranked against every query's candidates and the numbers are deterministic;
+        larger sets use batches of the reporting size. Nothing beyond the batches is embedded. The
+        reporter gets one table row per embedder and, with a progress (epochs), a point on the
+        metrics plot; the reference values become flat lines. Returns {embedder: (loss, metrics)}.
+        """
+        config = training_config
+        assert eval_data, "No eval data."
+        retrieval_model.ensure_resident()
+        retrieval_model.set_training_mode(False)
+        dataset_index = self._dataset_indexes(eval_data)
+        batch_size = None if len(eval_data) <= EVAL_SINGLE_BATCH_MAX else config.reporting_size
+        batches = fixed_batches(eval_data, dataset_index, batch_size, config.seed)
+        pool_sizes = [len(flatten_candidates(batch)[0]) for batch in batches]
+        start = time.time()
+        results = {"model": self._evaluate(retrieval_model, batches, config.temperature)}
+        results.update(self.evaluate_references(config, retrieval_model, batches))
+        retrieval_reporter.report_eval(label, progress, results, len(eval_data), pool_sizes, time.time() - start)
+        return results
\ No newline at end of file
````

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">activation/retrieval/retrieval_training_config.py</span>
    <span class="card-oneliner">`reporting_size`; validation and reference stats removed (eval returns its own results).</span>
    <span class="card-badge">Diff</span>
  </summary>

Exact delta vs `/source/activation/retrieval/retrieval_training_config.py`:

````diff-python
diff --git asource/activation/retrieval/retrieval_training_config.py bworkspace/activation/retrieval/retrieval_training_config.py
index 746938b..c5f67e0 100644
--- asource/activation/retrieval/retrieval_training_config.py
+++ bworkspace/activation/retrieval/retrieval_training_config.py
@@ -5,7 +5,8 @@ Configuration and statistics of a retrieval training run.
 recipe: LoRA 1e-4, AC 5e-4, head 1e-3, weight decay 0.01 on matrices only, AdamW (0.9, 0.999) eps 1e-8,
 10 % warmup then cosine to zero, clip 1.0, temperature 0.02 on view-mean MaxSim, checkpointing on).
 `RetrievalTrainingStats` records what the run did and `summarize()` flattens it so a 1000-query run
-extrapolates to 50k by arithmetic.
+extrapolates to 50k by arithmetic. Evaluations (validation or test sets, with the reference embedders)
+are separate `RetrievalTrainer.eval` calls and return their own results.
 """
 import typing as t
 from dataclasses import dataclass, field
@@ -26,7 +27,10 @@ class RetrievalTrainingConfig:
     gradient_checkpointing: bool = True     # base and AC model
     memory_headroom_fraction: float = 0.1
     reporting_fraction: float = 0.1         # of an epoch
+    reporting_size: int = 50                # eval batch size for sets larger than EVAL_SINGLE_BATCH_MAX queries
     seed: int = 0
+    baseline_model_name: str | None = None  # a harness embedding model scored by eval() on the same batches as a trained reference
+    reference_frozen_base: bool = True      # eval() also scores the frozen base alone (no LoRA, AC or head) as the floor
 
 
 @dataclass
@@ -44,9 +48,6 @@ class RetrievalTrainingStats:
     step_times: list[float] = field(default_factory=list)
     reporting_losses: list[tuple[float, float]] = field(default_factory=list)           # (progress in epochs, loss)
     reporting_metrics: list[tuple[float, dict[str, float]]] = field(default_factory=list)  # (progress, {in-batch rank-1, mrr@10, ndcg@10})
-    validation_losses: list[tuple[int, float]] = field(default_factory=list)            # (epoch, loss)
-    validation_metrics: list[tuple[int, dict[str, float]]] = field(default_factory=list)  # (epoch, in-batch metrics)
-    total_validation_time: float = 0.0
     peak_memory_bytes: int = 0
     batch_sizing: str = ""                                                              # the printed arithmetic
 
@@ -76,7 +77,6 @@ class RetrievalTrainingStats:
             "train_time_per_epoch": train_time / self.num_epochs if self.num_epochs else 0.0,
             "avg_step_time": average(self.step_times),
             "total_reporting_time": self.total_reporting_time,
-            "total_validation_time": self.total_validation_time,
             "reporting_time_fraction": self.total_reporting_time / train_time if train_time else 0.0,
             # Throughput (training steps only; reporting and validation excluded).
             "examples_per_s": examples / train_time if train_time else 0.0,
@@ -97,6 +97,4 @@ class RetrievalTrainingStats:
             "last_reporting_loss": reporting[-1] if reporting else None,
             "min_reporting_loss": min(reporting) if reporting else None,
             "last_reporting_metrics": self.reporting_metrics[-1][1] if self.reporting_metrics else None,
-            "validation_losses": list(self.validation_losses),
-            "validation_metrics": list(self.validation_metrics),
         }
````

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">activation/retrieval/retrieval_reporter.py</span>
    <span class="card-oneliner">Reporting loss only; metrics plot shared by training and eval; evaluations table; reference lines.</span>
    <span class="card-badge">Diff</span>
  </summary>

Exact delta vs `/source/activation/retrieval/retrieval_reporter.py`:

````diff-python
diff --git asource/activation/retrieval/retrieval_reporter.py bworkspace/activation/retrieval/retrieval_reporter.py
index 91255b3..d0ffe13 100644
--- asource/activation/retrieval/retrieval_reporter.py
+++ bworkspace/activation/retrieval/retrieval_reporter.py
@@ -38,6 +38,8 @@ class RetrievalReporter(HtmlReporter):
         )
         self.total_steps = 0
         self.epochs = 0
+        self.reference_values: dict[str, dict[str, float]] = {}
+        self._reference_lines_drawn = False
 
     # ----------------------------------------------------------------------------- layout
     def initialize_run(
@@ -48,27 +50,18 @@ class RetrievalReporter(HtmlReporter):
         counts: dict,
         batch_sizing: str,
         groups: dict[str, list],
+        reference_names: list[str] = (),
     ) -> None:
         """Create every widget of the run and write the first page."""
         self.total_steps = counts["total_steps"]
         self.epochs = config.epochs
         reporting_examples = counts["reporting_examples"]
         self.initialize_line_plot(
-            "reporting", "Reporting and validation loss",
-            f"Reporting: the fixed {reporting_examples}-query batch every {config.reporting_fraction:.0%} of an epoch. "
-            f"Validation: the full validation split at each epoch end in batches of {reporting_examples} queries, "
-            "so both losses see the same number of in-batch candidates.",
-            "epochs", "loss", ["reporting", "validation"],
-        )
-        self.initialize_line_plot(
-            "metrics", "In-batch ranking metrics",
-            "Rank-1 (top candidate is an own positive), MRR@10 and nDCG@10 (binary relevance over own positives), all "
-            "in-batch: each query is ranked against the batch's distinct candidates (its own positives and hard negatives "
-            "plus the other queries' candidates) with same-document collisions masked, not against the corpus. "
-            "The reporting batch at every reporting point, the validation split at each epoch end.",
-            "epochs", "metric",
-            [f"reporting {metric}" for metric in METRIC_NAMES] + [f"validation {metric}" for metric in METRIC_NAMES],
+            "reporting", "Reporting loss",
+            f"The fixed {reporting_examples}-query reporting batch every {config.reporting_fraction:.0%} of an epoch (epoch 0: untrained).",
+            "epochs", "loss", ["reporting"],
         )
+        self._ensure_metrics_plot(list(reference_names))
         self.initialize_line_plot(
             "step_loss", "Training loss per step", "Raw per-step loss (faint) with a 25-step moving average.",
             "step", "loss", ["loss"], smoothing_window=25,
@@ -77,10 +70,6 @@ class RetrievalReporter(HtmlReporter):
             "learning_rate", "Learning rates", "Warmup then cosine, one group per module.",
             "step", "lr", list(groups.keys()), log_y=True,
         )
-        self.initialize_bar_plot(
-            "epoch_validation", "Validation per epoch", "Loss and the in-batch ranking metrics on the validation split (batches of the reporting size).",
-            "epoch", "value", ["loss", *METRIC_NAMES],
-        )
         self.initialize_table(
             "epochs", "Throughput per epoch",
             "Padding fraction is the share of padded positions in the embedded sequences after length grouping.",
@@ -102,8 +91,38 @@ class RetrievalReporter(HtmlReporter):
             "trainable_parameters": {name: sum(p.numel() for p in params) for name, params in groups.items()},
         }, indent=2))
         self.set_status(epoch=f"0 / {self.epochs}", step=f"0 / {self.total_steps:,}", batch=f"{counts['batch_size']} examples")
+        self._draw_reference_lines()
         self.render(force=True)
 
+    def _ensure_metrics_plot(self, reference_names: list[str]) -> None:
+        """The metrics plot is shared by training (reporting points) and eval (eval points, reference lines); whoever comes first creates it."""
+        if "metrics" in self.widgets:
+            return
+        reference_note = ""
+        if reference_names:
+            reference_note = (" Flat lines: " + ", ".join(reference_names) + " scored on the eval batches (the frozen base alone "
+                              "is the floor, a well-trained embedder the target).")
+        self.initialize_line_plot(
+            "metrics", "In-batch ranking metrics",
+            "Rank-1 (top candidate is an own positive), MRR@10 and nDCG@10 (binary relevance over own positives), all "
+            "in-batch: each query is ranked against the batch's distinct candidates (its own positives and hard negatives "
+            "plus the other queries' candidates) with same-document collisions masked, not against the corpus. "
+            "Reporting: the reporting batch at every reporting point (epoch 0: untrained). Eval: a held-out set scored by "
+            "eval(), as one batch when it has at most 128 queries." + reference_note,
+            "epochs", "metric",
+            [f"reporting {metric}" for metric in METRIC_NAMES] + [f"eval {metric}" for metric in METRIC_NAMES]
+            + [f"{name} {metric}" for name in reference_names for metric in METRIC_NAMES],
+        )
+
+    def _draw_reference_lines(self) -> None:
+        """Flat lines across the run once both the reference values and the run length are known."""
+        if self._reference_lines_drawn or not self.reference_values or not self.epochs or "metrics" not in self.widgets:
+            return
+        for name, metrics in self.reference_values.items():
+            for x in (0.0, float(self.epochs)):
+                self.add_data_point("metrics", {"x": x, **{f"{name} {metric}": value for metric, value in metrics.items()}})
+        self._reference_lines_drawn = True
+
     # ----------------------------------------------------------------------------- events
     def report_training_step(
         self, step: int, progress: float, loss: float, learning_rates: dict[str, float],
@@ -124,11 +143,37 @@ class RetrievalReporter(HtmlReporter):
         self.set_status(last_reporting_loss=f"{loss:.3f}", **{f"reporting_{name}": f"{value:.2f}" for name, value in metrics.items()})
         print(f"Step {step}/{self.total_steps} (epoch {progress:.2f}): reporting loss {loss:.4f}, {_format_metrics(metrics)}")
 
-    def report_validation(self, epoch: int, loss: float, metrics: dict[str, float]) -> None:
-        self.add_data_point("reporting", {"x": float(epoch), "validation": loss})
-        self.add_data_point("metrics", {"x": float(epoch), **{f"validation {name}": value for name, value in metrics.items()}})
-        self.add_data_point("epoch_validation", {"label": f"epoch {epoch}", "loss": loss, **metrics})
-        print(f"Epoch {epoch}/{self.epochs}: validation loss {loss:.4f}, {_format_metrics(metrics)}")
+    def report_eval(
+        self, label: str, progress: float | None, results: dict[str, tuple[float, dict[str, float]]],
+        num_queries: int, pool_sizes: list[int], seconds: float,
+    ) -> None:
+        """
+        One table row per embedder ("model" and the references); the model's metrics as an eval
+        point at `progress` epochs when given; reference values as flat lines across the run.
+        """
+        reference_names = [name for name in results if name != "model"]
+        self._ensure_metrics_plot(reference_names)
+        if "evaluations" not in self.widgets:
+            self.initialize_table(
+                "evaluations", "Evaluations",
+                "In-batch loss and metrics of the model and the reference embedders on a held-out set, same batches for all: "
+                "one batch for sets of at most 128 queries (pool = every query's candidates), else batches of the reporting size.",
+                ["evaluation", "embedder", "queries", "pool", "loss", *METRIC_NAMES, "time"],
+            )
+        pool = f"{pool_sizes[0]}" if len(pool_sizes) == 1 else f"{len(pool_sizes)} x ~{sum(pool_sizes) / len(pool_sizes):.0f}"
+        for name, (loss, metrics) in results.items():
+            self.add_data_point("evaluations", {
+                "evaluation": label, "embedder": name, "queries": num_queries, "pool": pool, "loss": f"{loss:.4f}",
+                **{metric: f"{value:.3f}" for metric, value in metrics.items()}, "time": format_seconds(seconds),
+            })
+            print(f"Eval {label} / {name}: loss {loss:.4f}, {_format_metrics(metrics)} ({num_queries} queries, pool {pool})")
+            if name == "model":
+                if progress is not None:
+                    self.add_data_point("metrics", {"x": progress, **{f"eval {metric}": value for metric, value in metrics.items()}})
+            else:
+                self.reference_values[name] = metrics
+        self._draw_reference_lines()
+        self.render(force=True)
 
     def report_epoch(
         self, label: str, steps: int, shapes: list[tuple[int, int, int, int, int]], epoch_time: float,
````

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">activation/retrieval/__init__.py</span>
    <span class="card-oneliner">Exports the baseline module.</span>
    <span class="card-badge">Diff</span>
  </summary>

Exact delta vs `/source/activation/retrieval/__init__.py`:

````diff-python
diff --git asource/activation/retrieval/__init__.py bworkspace/activation/retrieval/__init__.py
index e6e2e70..1e36961 100644
--- asource/activation/retrieval/__init__.py
+++ bworkspace/activation/retrieval/__init__.py
@@ -1,5 +1,6 @@
 from .retrieval_ac import StandardRetrievalACModel
 from .retrieval_model import RetrievalModel
+from .retrieval_baseline import BaselineEmbedder, frozen_base_reference
 from .retrieval_batching import (
     RetrievalBatch,
     make_batches,
````

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">activation/bench/bright_econ_ablation/_common.py</span>
    <span class="card-oneliner">Shared bench code: args, harness, corpus, model, stage timer, two-labeler comparison, eval, budget extrapolation.</span>
    <span class="card-badge">New file</span>
  </summary>

Exact delta vs `/source/activation/bench/bright_econ_ablation/_common.py`:

````diff-python
diff --git aworkspace/activation/bench/bright_econ_ablation/_common.py bworkspace/activation/bench/bright_econ_ablation/_common.py
new file mode 100644
index 0000000..c588398
--- /dev/null
+++ bworkspace/activation/bench/bright_econ_ablation/_common.py
@@ -0,0 +1,357 @@
+"""
+What the three BRIGHT economics fitting benches share: arguments, the harness with every model,
+the whole economics corpus, the training model, stage timing, the evaluation on the full test set
+(one batch, next to the frozen base and the trained reference embedder), the report folders under
+the synced area, and the timing file with the budget arithmetic that a probe prints.
+
+A bench runs, in order, on one node: generation (engine), labeling (engine, one or two labelers),
+engines freed, the base loaded, eval before, training, eval after. `--probe` shrinks every stage
+to a few minutes and writes `probe_timing.json` with the rates and the hours for 25k–200k samples.
+"""
+import argparse
+import datetime as dt
+import json
+import os
+import uuid
+import time
+import typing as t
+from contextlib import contextmanager
+
+import torch
+
+from activation.common.data_syncing import resolve_path
+from activation.dataset.dataset import DataOrigin, LabeledRetrievalQAExample, LoadedDataset
+from activation.dataset.loaders import BrightDataset
+from activation.harness import FREE_DEVICE, SUPPORTS_FP4, HarnessRuntime, HarnessRuntimeConfig, ModelConfig
+from activation.retrieval import (
+    RetrievalModel,
+    RetrievalReporter,
+    RetrievalTrainer,
+    RetrievalTrainingConfig,
+    StandardRetrievalACModel,
+)
+
+DOMAIN = "economics"
+SYNC_FOLDER = "BRIGHT_ECON_ABLATION"
+DEFAULT_LABELER = "nvidia/Qwen3.6-35B-A3B-NVFP4" if SUPPORTS_FP4 else "Qwen/Qwen3.6-35B-A3B-FP8"
+BUDGET_HOURS = 5.5                    # 6 h round minus 30 min of safety
+EXTRAPOLATION_SAMPLES = (25_000, 50_000, 100_000, 200_000)
+
+MODE_ORIGINS = {
+    "programmatic": DataOrigin.PROGRAMMATIC,
+    "description": DataOrigin.SYNTHETIC_DESCRIPTION,
+    "question": DataOrigin.SYNTHETIC_QA,
+}
+PROBE_SAMPLES = {"programmatic": 2048, "description": 512, "question": 512}
+FULL_SAMPLES = {"programmatic": 200_000, "description": 50_000, "question": 50_000}
+
+
+def parse_args(mode: str) -> argparse.Namespace:
+    parser = argparse.ArgumentParser(description=f"BRIGHT {DOMAIN}: {mode} fitting", formatter_class=argparse.RawDescriptionHelpFormatter)
+    parser.add_argument("--probe", action="store_true", help="short run of every stage; writes probe_timing.json with the budget arithmetic")
+    parser.add_argument("--num-samples", type=int, default=None, help=f"chunks to study (default: {PROBE_SAMPLES[mode]} probe, {FULL_SAMPLES[mode]} full)")
+    parser.add_argument("--epochs", type=int, default=None, help="default: 2 probe, 1 full")
+    parser.add_argument("--pretrain-programmatic", type=int, default=0, help="Round 2: this many programmatic examples trained first, in the same process")
+    parser.add_argument("--study-only", action="store_true", help="generate and label into the cache, then stop (no model, eval or training)")
+    parser.add_argument("--base-model-id", default="Qwen/Qwen3-4B")
+    parser.add_argument("--baseline-model-id", default="Qwen/Qwen3-Embedding-4B", help="trained reference embedder; 'none' to skip")
+    parser.add_argument("--generator-model-id", default="RedHatAI/Qwen3.5-4B-FP8-dynamic")
+    parser.add_argument("--label-model-ids", nargs="+", default=None,
+                        help=f"labelers; the first one's labels are kept, the others are compared (default: {DEFAULT_LABELER}, plus the generator in a probe)")
+    parser.add_argument("--caching-id", default="econ", help="cache key for generation and labels under the synced CACHE folder")
+    parser.add_argument("--batch-size", type=int, default=None, help="examples per step; default: sized for the GPU")
+    parser.add_argument("--lora-rank", type=int, default=128)
+    parser.add_argument("--num-prefix-tokens", type=int, default=16)
+    parser.add_argument("--num-view-tokens", type=int, default=8)
+    parser.add_argument("--d-ac-model", type=int, default=1024)
+    parser.add_argument("--num-ac-layers", type=int, default=8)
+    parser.add_argument("--chunk-size-chars", type=int, default=4096)
+    parser.add_argument("--val-ratio", type=float, default=0.02, help="share of the study examples held out as the reporting pool")
+    parser.add_argument("--reporting-size", type=int, default=50)
+    parser.add_argument("--reporting-fraction", type=float, default=0.1, help="a reporting point every this fraction of an epoch")
+    parser.add_argument("--report-on-test", action="store_true",
+                        help="report on the test queries (one batch, the eval's pool) instead of the study hold-out: a learning curve on the eval metric")
+    parser.add_argument("--report-root", default=None, help=f"default: the synced folder {SYNC_FOLDER}/")
+    parser.add_argument("--seed", type=int, default=0)
+    args = parser.parse_args()
+    args.mode = mode
+    args.num_samples = args.num_samples or (PROBE_SAMPLES[mode] if args.probe else FULL_SAMPLES[mode])
+    args.epochs = args.epochs or (2 if args.probe else 1)
+    if args.label_model_ids is None:
+        args.label_model_ids = [DEFAULT_LABELER] + ([args.generator_model_id] if args.probe else [])
+    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M")
+    root = args.report_root or str(resolve_path(SYNC_FOLDER))
+    # Sample count and a random suffix keep parallel runs of one mode (same minute) in separate folders.
+    args.report_folder = os.path.join(root, "PROBE_REPORTS" if args.probe else "REPORTS", f"{mode}_n{args.num_samples}_{stamp}_{uuid.uuid4().hex[:8]}")
+    os.makedirs(args.report_folder, exist_ok=True)
+    return args
+
+
+def model_name_of(model_id: str) -> str:
+    return model_id.split("/")[-1].lower()
+
+
+def build_harness(args: argparse.Namespace) -> HarnessRuntime:
+    """Base, baseline, generator and labelers registered; the study cache under the synced CACHE folder."""
+    ids = [args.base_model_id, args.generator_model_id, *args.label_model_ids]
+    if args.baseline_model_id.lower() != "none":
+        ids.append(args.baseline_model_id)
+    model_configs = {model_name_of(model_id): ModelConfig(model_name_of(model_id), model_id) for model_id in dict.fromkeys(ids)}
+    return HarnessRuntime(HarnessRuntimeConfig(
+        model_configs=model_configs,
+        doc_chunk_size_chars=args.chunk_size_chars,
+        doc_embedding_input_limit_chars=args.chunk_size_chars,
+        dataset_study_qa_model_name=model_name_of(args.generator_model_id),
+        dataset_study_label_model_name=model_name_of(args.label_model_ids[0]),
+        dataset_study_seed=args.seed,
+        cache_storage_dir=str(resolve_path(f"{SYNC_FOLDER}/CACHE")),
+    ))
+
+
+def load_economics(harness: HarnessRuntime) -> LoadedDataset:
+    """The whole domain: every labeled query and every document, BM25 built."""
+    dataset = BrightDataset.load(harness, max_examples=None, domain=DOMAIN, max_corpus_documents=None)
+    harness.dataset_manager.build_bm25_indexes()
+    return dataset
+
+
+def build_model(harness: HarnessRuntime, args: argparse.Namespace) -> RetrievalModel:
+    base = model_name_of(args.base_model_id)
+    lora_name, ac_name = f"{base}-econ-lora", f"{base}-econ-ac"
+    harness.module_manager.register_lora(lora_name, base, rank=args.lora_rank)
+    torch.manual_seed(args.seed)
+    ac_model = StandardRetrievalACModel(
+        harness, base, d_ac_model=args.d_ac_model, num_prefix_tokens=args.num_prefix_tokens,
+        num_view_tokens=args.num_view_tokens, num_ac_layers=args.num_ac_layers,
+    )
+    harness.module_manager.register_retrieval_ac(ac_name, base, ac_model)
+    return RetrievalModel(harness, base, d_embedding_result=None, ac_name=ac_name, lora_name=lora_name)
+
+
+def training_config(args: argparse.Namespace, harness: HarnessRuntime, epochs: int) -> RetrievalTrainingConfig:
+    baseline = None if args.baseline_model_id.lower() == "none" else model_name_of(args.baseline_model_id)
+    return RetrievalTrainingConfig(
+        epochs=epochs, batch_size=args.batch_size, seed=args.seed, reporting_size=args.reporting_size,
+        reporting_fraction=args.reporting_fraction, baseline_model_name=baseline, reference_frozen_base=True,
+    )
+
+
+class StageTimer:
+    """
+    Wall time and unit count per stage, in order; a stage name may repeat (labelers). Model and
+    engine loads that happen inside a stage are measured apart (`load_seconds`, from the harness
+    stats), so `net_seconds` is the per-unit work and the loads are a run's fixed cost.
+    """
+
+    def __init__(self, harness: HarnessRuntime):
+        self.harness = harness
+        self.stages: list[dict] = []
+
+    def _loads(self) -> float:
+        return sum(seconds for _, seconds in self.harness.harness_stats.model_loading_times)
+
+    @contextmanager
+    def stage(self, name: str, units: int = 0):
+        start, loads_before = time.time(), self._loads()
+        record = {"stage": name, "units": units, "seconds": 0.0, "load_seconds": 0.0, "net_seconds": 0.0}
+        self.stages.append(record)
+        print(f"=== {name} ({units} units) ...", flush=True)
+        try:
+            yield record
+        finally:
+            record["seconds"] = time.time() - start
+            record["load_seconds"] = self._loads() - loads_before
+            record["net_seconds"] = record["seconds"] - record["load_seconds"]
+            rate = record["units"] / record["net_seconds"] if record["net_seconds"] > 0 and record["units"] else 0.0
+            print(f"=== {name}: {record['seconds']:.1f} s ({record['load_seconds']:.1f} s of model loads)"
+                  + (f", {rate:.2f} units/s net" if rate else ""), flush=True)
+
+    def net_seconds(self, prefix: str) -> float:
+        return sum(record["net_seconds"] for record in self.stages if record["stage"].startswith(prefix))
+
+
+def free_engines(harness: HarnessRuntime) -> None:
+    for loaded in harness.loaded_models.values():
+        loaded.engine_to_device(FREE_DEVICE)
+
+
+def label_examples(
+    harness: HarnessRuntime, dataset: LoadedDataset, examples: list[LabeledRetrievalQAExample], origin: DataOrigin,
+    args: argparse.Namespace, timer: StageTimer,
+) -> dict:
+    """
+    Label the examples with every labeler in turn (each pass timed separately, each cached under
+    its own key); the first labeler's labels are the ones kept. Between labelers the examples are
+    reset to their source-chunk positive so each labeler judges the same pool. Returns per-labeler
+    counts and, for the extra labelers, the agreement with the first (mean Jaccard of the positive
+    and of the negative sets over the examples both labeled).
+    """
+    ids = {example.example_id for example in examples}
+    snapshots: dict[str, dict[str, tuple[set, set]]] = {}
+    summary: dict = {"labelers": {}}
+    for model_id in args.label_model_ids:
+        for example in examples:                                                         # same starting point for every labeler
+            example.positive_chunk_ids = list(example.positive_chunk_ids[:1])
+            example.hard_negative_chunk_ids = None
+            example.oracle_labeled = False
+        name = model_name_of(model_id)
+        with timer.stage(f"labeling:{name}", units=len(examples)) as record:
+            labeled = harness.dataset_manager.label_study_examples(
+                dataset.dataset_id, len(examples), origins=[origin], caching_id=args.caching_id, label_model_name=name,
+            )
+            record["units"] = len(labeled)
+        snapshots[name] = {e.example_id: (set(e.positive_chunk_ids or []), set(e.hard_negative_chunk_ids or [])) for e in labeled if e.example_id in ids}
+        summary["labelers"][name] = {
+            "labeled": len(labeled),
+            "positives_per_example": sum(len(p) for p, _ in snapshots[name].values()) / max(1, len(snapshots[name])),
+            "negatives_per_example": sum(len(n) for _, n in snapshots[name].values()) / max(1, len(snapshots[name])),
+        }
+    first = model_name_of(args.label_model_ids[0])
+    for name, snapshot in snapshots.items():
+        if name == first:
+            continue
+        shared = [example_id for example_id in snapshot if example_id in snapshots[first]]
+        def jaccard(a: set, b: set) -> float:
+            return 1.0 if not a and not b else len(a & b) / len(a | b)
+        summary["labelers"][name]["agreement_with_first"] = {
+            "examples": len(shared),
+            "positives_jaccard": sum(jaccard(snapshot[i][0], snapshots[first][i][0]) for i in shared) / max(1, len(shared)),
+            "negatives_jaccard": sum(jaccard(snapshot[i][1], snapshots[first][i][1]) for i in shared) / max(1, len(shared)),
+        }
+    for example in examples:                                                             # keep the first labeler's verdicts
+        if example.example_id in snapshots.get(first, {}):
+            positives, negatives = snapshots[first][example.example_id]
+            example.positive_chunk_ids = [example.positive_chunk_ids[0]] + sorted(positives - {example.positive_chunk_ids[0]})
+            example.hard_negative_chunk_ids = sorted(negatives) or None
+            example.oracle_labeled = True
+    return summary
+
+
+def evaluate(
+    trainer: RetrievalTrainer, model: RetrievalModel, reporter: RetrievalReporter, test_data: list[LabeledRetrievalQAExample],
+    config: RetrievalTrainingConfig, label: str, progress: float, timer: StageTimer,
+) -> dict:
+    """The full test set as one batch (103 queries against every query's gold chunks), the model next to the references."""
+    with timer.stage(f"eval:{label}", units=len(test_data)):
+        results = trainer.eval(model, reporter, test_data, config, label=label, progress=progress)
+    return {name: {"loss": loss, **metrics} for name, (loss, metrics) in results.items()}
+
+
+def load_seconds(harness: HarnessRuntime) -> dict[str, float]:
+    """Model and engine load time per model name, from the harness stats (fixed costs of a run)."""
+    totals: dict[str, float] = {}
+    for name, seconds in harness.harness_stats.model_loading_times:
+        totals[name] = totals.get(name, 0.0) + seconds
+    return totals
+
+
+def extrapolate(mode: str, timer: StageTimer, loads: dict[str, float], training_summary: dict, probe_epochs: int, num_samples: int) -> dict:
+    """
+    Hours for n study chunks from the probe's net rates: fixed costs (every model and engine load,
+    the corpus, both evals) plus n x (generation + first-labeler labeling seconds per chunk) plus
+    epochs x (training examples per chunk) x n x seconds per training example. Descriptions yield
+    fewer examples than chunks (needs_abstraction_bridge), which the examples-per-chunk ratio
+    carries. Given for one epoch (Round 1) and for the probe's epochs.
+    """
+    generation = next((r for r in timer.stages if r["stage"] == "generation"), None)
+    labeling = next((r for r in timer.stages if r["stage"].startswith("labeling:")), None)
+    study = next((r for r in timer.stages if r["stage"] == "study data"), None)
+    per_chunk = 0.0
+    if generation and generation["units"]:
+        per_chunk += generation["net_seconds"] / generation["units"]
+    if labeling and labeling["units"]:
+        per_chunk += labeling["net_seconds"] / labeling["units"] * (labeling["units"] / max(1, num_samples))
+    examples_trained = training_summary["num_examples"]
+    train_seconds_per_example_epoch = training_summary["total_train_time"] / max(1, examples_trained * probe_epochs)
+    examples_per_chunk = (study["units_out"] if study and study.get("units_out") else num_samples) / max(1, num_samples)
+    fixed = sum(loads.values()) + timer.net_seconds("eval:") + timer.net_seconds("corpus")
+
+    def table(epochs: int) -> dict:
+        per_sample = per_chunk + epochs * examples_per_chunk * train_seconds_per_example_epoch
+        hours = {str(n): round((fixed + n * per_sample) / 3600, 2) for n in EXTRAPOLATION_SAMPLES}
+        largest = int(max(0.0, BUDGET_HOURS * 3600 - fixed) / per_sample) if per_sample else None
+        return {"epochs": epochs, "seconds_per_chunk": per_sample, "hours_for_chunks": hours, "largest_chunks_under_budget": largest}
+
+    return {
+        "mode": mode, "probe_epochs": probe_epochs, "probe_chunks": num_samples,
+        "fixed_seconds": fixed, "model_load_seconds": loads,
+        "generation_seconds_per_chunk": generation["net_seconds"] / generation["units"] if generation and generation["units"] else None,
+        "labeling_seconds_per_example": labeling["net_seconds"] / labeling["units"] if labeling and labeling["units"] else None,
+        "examples_per_chunk": examples_per_chunk,
+        "training_seconds_per_example_epoch": train_seconds_per_example_epoch,
+        "training_tokens_per_example": training_summary["avg_tokens_per_example"],
+        "training_tokens_per_s": training_summary["tokens_per_s"],
+        "budget_hours": BUDGET_HOURS,
+        "round_1": table(1), "probe_shape": table(probe_epochs),
+    }
+
+
+def run(mode: str, make_training_data: t.Callable[[HarnessRuntime, LoadedDataset, argparse.Namespace, StageTimer], tuple[list[LabeledRetrievalQAExample], dict]]) -> None:
+    """The bench: corpus, study data, engines freed, base, eval, training, eval, timing file."""
+    args = parse_args(mode)
+    print(f"{mode} fitting -> {args.report_folder}\n{json.dumps(vars(args), indent=1, default=str)}", flush=True)
+    harness = build_harness(args)
+    timer = StageTimer(harness)
+    with timer.stage("corpus"):
+        dataset = load_economics(harness)
+    test_data = harness.dataset_manager.select_testing_data(dataset.dataset_id, seed=args.seed)
+    study_summary: dict = {}
+    with timer.stage("study data", units=args.num_samples) as study_record:
+        study_examples, study_summary = make_training_data(harness, dataset, args, timer)
+        study_record["units_out"] = len(study_examples)
+    if args.study_only:
+        output = {"args": vars(args), "stages": timer.stages, "model_load_seconds": load_seconds(harness), "study": study_summary,
+                  "dataset_stats": dataset.stats.summarize(), "harness": harness.harness_stats.summarize()}
+        with open(os.path.join(args.report_folder, "study_summary.json"), "w") as handle:
+            json.dump(output, handle, indent=1, default=str)
+        print(json.dumps({"stages": timer.stages, "study": study_summary}, indent=1, default=str), flush=True)
+        return
+    pretrain_examples = []
+    if args.pretrain_programmatic:
+        pretrain_examples = harness.dataset_manager.synthesize_programmatic_examples(dataset.dataset_id, args.pretrain_programmatic)
+    free_engines(harness)
+
+    model = build_model(harness, args)
+    trainer = RetrievalTrainer(harness)
+    reporter = RetrievalReporter(
+        args.report_folder, title=f"BRIGHT {DOMAIN}: {mode} fitting" + (" (probe)" if args.probe else ""),
+        description=f"{args.base_model_id} frozen + LoRA r{args.lora_rank} + AC (P={args.num_prefix_tokens}, V={args.num_view_tokens}, "
+                    f"d {args.d_ac_model}, {args.num_ac_layers} layers); {len(study_examples):,} {mode} examples, {args.epochs} epoch(s); "
+                    f"eval on the {len(test_data)} native queries as one batch.",
+    )
+    config = training_config(args, harness, args.epochs)
+    evaluations = {"before": evaluate(trainer, model, reporter, test_data, config, "before training", 0.0, timer)}
+    summaries = {}
+    if pretrain_examples:
+        train, _, report = harness.dataset_manager.select_training_data(
+            dataset.dataset_id, len(pretrain_examples), origins=[DataOrigin.PROGRAMMATIC], oracle_labeled_only=False,
+            val_ratio=args.val_ratio, max_reporting_size=args.reporting_size, seed=args.seed,
+        )
+        if args.report_on_test:
+            report = test_data
+        with timer.stage("training:programmatic", units=len(train)):
+            summaries["programmatic"] = trainer.train(training_config(args, harness, 1), model, reporter, train, report).summarize()
+        evaluations["after programmatic"] = evaluate(trainer, model, reporter, test_data, config, "after programmatic", 1.0, timer)
+    train, _, report = harness.dataset_manager.select_training_data(
+        dataset.dataset_id, len(study_examples), origins=[MODE_ORIGINS[mode]], oracle_labeled_only=(mode != "programmatic"),
+        val_ratio=args.val_ratio, max_reporting_size=args.reporting_size, seed=args.seed,
+    )
+    if args.report_on_test:
+        report = test_data                                                 # the eval batch as the learning curve
+    with timer.stage(f"training:{mode}", units=len(train)):
+        summaries[mode] = trainer.train(config, model, reporter, train, report).summarize()
+    evaluations["after"] = evaluate(trainer, model, reporter, test_data, config, "after training", float(args.epochs), timer)
+
+    loads = load_seconds(harness)
+    timing = extrapolate(mode, timer, loads, summaries[mode], args.epochs, args.num_samples)
+    output = {
+        "args": vars(args), "stages": timer.stages, "model_load_seconds": loads, "study": study_summary,
+        "training": summaries, "evaluations": evaluations, "extrapolation": timing,
+        "dataset_stats": dataset.stats.summarize(), "harness": harness.harness_stats.summarize(),
+    }
+    name = "probe_timing.json" if args.probe else "run_summary.json"
+    with open(os.path.join(args.report_folder, name), "w") as handle:
+        json.dump(output, handle, indent=1, default=str)
+    reporter.set_text("timing", "Stages and budget", json.dumps({"stages": timer.stages, "extrapolation": timing, "study": study_summary}, indent=2, default=str))
+    reporter.render(force=True)
+    print(json.dumps({"evaluations": evaluations, "extrapolation": timing, "study": study_summary}, indent=1, default=str), flush=True)
````

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">activation/bench/bright_econ_ablation/bench_programmatic_fitting.py</span>
    <span class="card-oneliner">Programmatic mode.</span>
    <span class="card-badge">Diff</span>
  </summary>

Exact delta vs `/source/activation/bench/bright_econ_ablation/bench_programmatic_fitting.py`:

````diff-python
diff --git asource/activation/bench/bright_econ_ablation/bench_programmatic_fitting.py bworkspace/activation/bench/bright_econ_ablation/bench_programmatic_fitting.py
index 34f0114..61f6a3c 100644
--- asource/activation/bench/bright_econ_ablation/bench_programmatic_fitting.py
+++ bworkspace/activation/bench/bright_econ_ablation/bench_programmatic_fitting.py
@@ -1,18 +1,24 @@
 """
-Simple benchmark to figure how much we can fit bright's economics split with dumb programmatic fitting.
-The final eval should produce the test-time results (in batch ndcg for now) vs baseline embedding model.
+How far dumb programmatic fitting takes BRIGHT economics: a random span of a chunk is the query,
+the chunk its positive, negatives from the batch only. No oracle stages; the cost is training.
+The eval is the 103 native queries in one batch, before and after, next to the frozen base and
+the trained reference embedder.
 
-num_samples should be ~4x the corpus size, but this will ultimately be decided by the timing of things.
+    uv run sky exec --sync <node> -- uv run python -m activation.bench.bright_econ_ablation.bench_programmatic_fitting --probe
+
+`--probe` trains 2,048 examples for 2 epochs and writes PROBE_REPORTS/programmatic_<stamp>/probe_timing.json
+with the hours per sample count; without it the run is the Round 1 shape (200k examples, 1 epoch;
+`--num-samples` overrides). `--pretrain-programmatic` is meaningless here. Reports go to the synced
+folder IB/TMP/SYNC/BRIGHT_ECON_ABLATION/(PROBE_)REPORTS/.
+"""
+from activation.bench.bright_econ_ablation._common import run
+
+
+def make_training_data(harness, dataset, args, timer):
+    with timer.stage("programmatic examples", units=args.num_samples):
+        examples = harness.dataset_manager.synthesize_programmatic_examples(dataset.dataset_id, args.num_samples)
+    return examples, {"examples": len(examples)}
 
-Should runnable with uv run sky exec_cmd ...
-Use datetime on the report file.
-Should accept a --probe flag that runs for ~100-1000 examples (enough to give a signal for expected timing).
-    - This signal well tell us what we need for 2 rounds of running overnight.
-    - Round 1: bare techniques only.
-    - Round 2: programmatic fit + (cached) training.
-        - (no need for model checkpointing yet though).
-Should run in parallel with the other 3 modes.
-Reports go to IB/TMP/SYNC/BRIGHT_ECON_ABLATION/(PROBE_)REPORTS/
 
-Use qwen 3 4b as base.
-"""
\ No newline at end of file
+if __name__ == "__main__":
+    run("programmatic", make_training_data)
````

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">activation/bench/bright_econ_ablation/bench_description_fitting.py</span>
    <span class="card-oneliner">Description mode.</span>
    <span class="card-badge">Diff</span>
  </summary>

Exact delta vs `/source/activation/bench/bright_econ_ablation/bench_description_fitting.py`:

````diff-python
diff --git asource/activation/bench/bright_econ_ablation/bench_description_fitting.py bworkspace/activation/bench/bright_econ_ablation/bench_description_fitting.py
index ddabe18..f7737b7 100644
--- asource/activation/bench/bright_econ_ablation/bench_description_fitting.py
+++ bworkspace/activation/bench/bright_econ_ablation/bench_description_fitting.py
@@ -1,5 +1,25 @@
 """
-Like programmatic, but the generated descriptions are used.
-Should also have a --probe mode: benchmarks study gen, labeling, and training for a small sample.
-Here, caching should be enabled to write examples/labels to a synced file in IB/TMP/SYNC/BRIGHT_ECON_ABLATION/CACHE/
-"""
\ No newline at end of file
+Like programmatic, but a generated search-friendly description of each chunk is the query and the
+labeler adds positives and hard negatives from the BM25 top-10. Generation and labels are cached
+under the synced folder IB/TMP/SYNC/BRIGHT_ECON_ABLATION/CACHE/ by `--caching-id`, so Round 2
+(`--pretrain-programmatic N`) pays training only.
+
+    uv run sky exec --sync <node> -- uv run python -m activation.bench.bright_econ_ablation.bench_description_fitting --probe
+
+`--probe`: 512 chunks through generation, both labelers (the configured one and the generator,
+agreement reported), 2 training epochs, the eval on the 103 native queries before and after.
+"""
+from activation.bench.bright_econ_ablation._common import MODE_ORIGINS, label_examples, run
+
+
+def make_training_data(harness, dataset, args, timer):
+    with timer.stage("generation", units=args.num_samples) as record:
+        examples = harness.dataset_manager.synthesize_study_examples_description(dataset.dataset_id, args.num_samples, caching_id=args.caching_id)
+        record["units"] = args.num_samples
+    summary = {"chunks": args.num_samples, "descriptions": len(examples), "samples": [(e.query, e.positive_chunk_ids[0]) for e in examples[:4]]}
+    summary |= label_examples(harness, dataset, examples, MODE_ORIGINS["description"], args, timer)
+    return examples, summary
+
+
+if __name__ == "__main__":
+    run("description", make_training_data)
````

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">activation/bench/bright_econ_ablation/bench_qa_fitting.py</span>
    <span class="card-oneliner">Question mode.</span>
    <span class="card-badge">Diff</span>
  </summary>

Exact delta vs `/source/activation/bench/bright_econ_ablation/bench_qa_fitting.py`:

````diff-python
diff --git asource/activation/bench/bright_econ_ablation/bench_qa_fitting.py bworkspace/activation/bench/bright_econ_ablation/bench_qa_fitting.py
index edf5f56..84d0983 100644
--- asource/activation/bench/bright_econ_ablation/bench_qa_fitting.py
+++ bworkspace/activation/bench/bright_econ_ablation/bench_qa_fitting.py
@@ -1,3 +1,21 @@
 """
-Like description but with qa generation.
-"""
\ No newline at end of file
+Like description but with question generation: a hard question with its answer per chunk (the
+existing study pipeline), labeled against the BM25 top-10. Same flags, cache and probe shape as
+the description bench.
+
+    uv run sky exec --sync <node> -- uv run python -m activation.bench.bright_econ_ablation.bench_qa_fitting --probe
+"""
+from activation.bench.bright_econ_ablation._common import MODE_ORIGINS, label_examples, run
+
+
+def make_training_data(harness, dataset, args, timer):
+    with timer.stage("generation", units=args.num_samples) as record:
+        examples = harness.dataset_manager.synthesize_study_examples_qa(dataset.dataset_id, args.num_samples, caching_id=args.caching_id)
+        record["units"] = args.num_samples
+    summary = {"chunks": args.num_samples, "questions": len(examples), "samples": [(e.query, e.gold_answers[0], e.positive_chunk_ids[0]) for e in examples[:4]]}
+    summary |= label_examples(harness, dataset, examples, MODE_ORIGINS["question"], args, timer)
+    return examples, summary
+
+
+if __name__ == "__main__":
+    run("question", make_training_data)
````

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">activation/bench/agent_probes/retrieval_training_bench.py</span>
    <span class="card-oneliner">Your moved copy plus the round-4 baseline flag and the train/eval split.</span>
    <span class="card-badge">Diff</span>
  </summary>

Exact delta vs `/source/activation/bench/agent_probes/retrieval_training_bench.py`:

````diff-python
diff --git asource/activation/bench/agent_probes/retrieval_training_bench.py bworkspace/activation/bench/agent_probes/retrieval_training_bench.py
index 9e72f43..ca17380 100644
--- asource/activation/bench/agent_probes/retrieval_training_bench.py
+++ bworkspace/activation/bench/agent_probes/retrieval_training_bench.py
@@ -27,6 +27,8 @@ from activation.retrieval import (
 def main() -> None:
     parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
     parser.add_argument("--base-model-id", default="Qwen/Qwen3-0.6B")
+    parser.add_argument("--baseline-model-id", default="Qwen/Qwen3-Embedding-0.6B",
+                        help="well-trained embedder scored on the validation set as a reference; 'none' to skip")
     parser.add_argument("--num-queries", type=int, default=1000, help="queries selected before the validation split")
     parser.add_argument("--val-ratio", type=float, default=0.05)
     parser.add_argument("--reporting-size", type=int, default=50)
@@ -43,8 +45,13 @@ def main() -> None:
 
     base_model_name = args.base_model_id.split("/")[-1].lower()
     lora_name, ac_name = f"{base_model_name}-retrieval-lora", f"{base_model_name}-retrieval-ac"
+    model_configs = {base_model_name: ModelConfig(base_model_name, args.base_model_id)}
+    baseline_model_name = None
+    if args.baseline_model_id.lower() != "none":
+        baseline_model_name = args.baseline_model_id.split("/")[-1].lower()
+        model_configs[baseline_model_name] = ModelConfig(baseline_model_name, args.baseline_model_id)
     harness = HarnessRuntime(HarnessRuntimeConfig(
-        model_configs={base_model_name: ModelConfig(base_model_name, args.base_model_id)},
+        model_configs=model_configs,
         doc_chunk_size_chars=args.chunk_size_chars,
         doc_embedding_input_limit_chars=args.chunk_size_chars,
     ))
@@ -65,15 +72,18 @@ def main() -> None:
     retrieval_model = RetrievalModel(
         harness, base_model_name, d_embedding_result=args.d_embedding_result, ac_name=ac_name, lora_name=lora_name,
     )
-    config = RetrievalTrainingConfig(epochs=args.epochs, batch_size=args.batch_size, seed=args.seed)
+    config = RetrievalTrainingConfig(epochs=args.epochs, batch_size=args.batch_size, seed=args.seed, baseline_model_name=baseline_model_name)
     reporter = RetrievalReporter(
         args.report_folder,
         title=f"Retrieval training — MS-MARCO train, {len(training_data):,} queries",
         description=f"{args.base_model_id} frozen + LoRA r{args.lora_rank} + AC (V={args.num_view_tokens}), "
                     f"{args.epochs} epochs, {len(validation_data)} validation / {len(reporting_data)} reporting queries.",
     )
-    stats = RetrievalTrainer(harness).train(config, retrieval_model, reporter, training_data, reporting_data, validation_data)
-    summary = stats.summarize()
+    trainer = RetrievalTrainer(harness)
+    evaluations = {"before": trainer.eval(retrieval_model, reporter, validation_data, config, label="validation before", progress=0.0)}
+    stats = trainer.train(config, retrieval_model, reporter, training_data, reporting_data)
+    evaluations["after"] = trainer.eval(retrieval_model, reporter, validation_data, config, label="validation after", progress=float(args.epochs))
+    summary = stats.summarize() | {"evaluations": evaluations}
     print(json.dumps(summary, indent=2))
     with open(os.path.join(args.report_folder, "training_stats.json"), "w") as handle:
         json.dump({"summary": summary, "harness": harness.harness_stats.summarize()}, handle, indent=1)
````

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">activation/bench/agent_probes/cheap_synthetic_bench.py</span>
    <span class="card-oneliner">Import of the moved prompt.</span>
    <span class="card-badge">Diff</span>
  </summary>

Exact delta vs `/source/activation/bench/agent_probes/cheap_synthetic_bench.py`:

````diff-python
diff --git asource/activation/bench/agent_probes/cheap_synthetic_bench.py bworkspace/activation/bench/agent_probes/cheap_synthetic_bench.py
index 0fde34c..070fbb3 100644
--- asource/activation/bench/agent_probes/cheap_synthetic_bench.py
+++ bworkspace/activation/bench/agent_probes/cheap_synthetic_bench.py
@@ -21,7 +21,7 @@ import torch
 from vllm.sampling_params import StructuredOutputsParams
 
 from activation.common.reporting import HtmlReporter
-from activation.dataset.dataset_study import STUDY_JSON_SCHEMA, make_study_prompt
+from activation.dataset.dataset_study_prompts import STUDY_QA_JSON_SCHEMA as STUDY_JSON_SCHEMA, make_study_qa_prompt as make_study_prompt
 from activation.dataset.dataset_utils import safe_truncate_embedding_chunk, shuffle_fill_truncate
 from activation.dataset.loaders import MsMarcoDataset
 from activation.harness import FREE_DEVICE, HarnessRuntime, HarnessRuntimeConfig, ModelConfig
````

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">activation/tests/test_basic_retrieval_training.py</span>
    <span class="card-oneliner">Small P+V AC, `train` then `eval` on the validation split.</span>
    <span class="card-badge">Diff</span>
  </summary>

Exact delta vs `/source/activation/tests/test_basic_retrieval_training.py`:

````diff-python
diff --git asource/activation/tests/test_basic_retrieval_training.py bworkspace/activation/tests/test_basic_retrieval_training.py
index b16dbe7..b7b91c0 100644
--- asource/activation/tests/test_basic_retrieval_training.py
+++ bworkspace/activation/tests/test_basic_retrieval_training.py
@@ -9,6 +9,7 @@ from activation.dataset.loaders import MsMarcoDataset
 from activation.harness import HarnessRuntime, HarnessRuntimeConfig, ModelConfig
 from activation.harness.hf_utils import FREE_DEVICE
 from activation.retrieval import (
+    METRIC_NAMES,
     RetrievalModel,
     RetrievalReporter,
     RetrievalTrainer,
@@ -47,7 +48,7 @@ def test_basic_retrieval_training():
 
     harness.module_manager.register_lora(LORA_NAME, BASE_MODEL_NAME, rank=32)
     torch.manual_seed(0)                                                               # seeded AC and head init
-    ac_model = StandardRetrievalACModel(harness, BASE_MODEL_NAME, num_view_tokens=4)
+    ac_model = StandardRetrievalACModel(harness, BASE_MODEL_NAME, d_ac_model=256, num_prefix_tokens=4, num_view_tokens=4, num_ac_layers=2)
     harness.module_manager.register_retrieval_ac(AC_NAME, BASE_MODEL_NAME, ac_model)
     retrieval_model = RetrievalModel(
         harness, BASE_MODEL_NAME, d_embedding_result=256, ac_name=AC_NAME, lora_name=LORA_NAME,
@@ -62,10 +63,11 @@ def test_basic_retrieval_training():
         description="Qwen3-0.6B + LoRA + AC on 20 MS-MARCO queries, CPU.",
     )
     trainer = RetrievalTrainer(harness)
-    stats = trainer.train(
-        training_config, retrieval_model, reporter, training_data, reporting_data, validation_data,
-    )
+    stats = trainer.train(training_config, retrieval_model, reporter, training_data, reporting_data)
     print(json.dumps(stats.summarize(), indent=2))
+    # Held-out evaluation is a separate call: the model next to the frozen base, one batch (10 queries).
+    results = trainer.eval(retrieval_model, reporter, validation_data, training_config, label="validation", progress=float(training_config.epochs))
+    assert set(results) == {"model", "frozen base"} and all(set(metrics) == set(METRIC_NAMES) for _, metrics in results.values())
 
     assert stats.num_epochs == 4 and stats.num_steps == 20
     assert all(math.isfinite(loss) for _step, loss in stats.step_losses)
@@ -78,7 +80,6 @@ def test_basic_retrieval_training():
     assert sum(step_losses[-5:]) < 0.5 * sum(step_losses[:5]), step_losses
     reporting_losses = [loss for _progress, loss in stats.reporting_losses]
     assert min(reporting_losses) < reporting_losses[0], reporting_losses
-    assert len(stats.validation_losses) == 4
     assert os.path.exists(os.path.join(REPORT_FOLDER, "report.tressoir.html"))
     assert os.path.exists(os.path.join(REPORT_FOLDER, "report_data.json"))
````

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">activation/tests/test_basic_dataset_study.py</span>
    <span class="card-oneliner">`origins=[SYNTHETIC_QA]` instead of `synthetic_only`.</span>
    <span class="card-badge">Diff</span>
  </summary>

Exact delta vs `/source/activation/tests/test_basic_dataset_study.py`:

````diff-python
diff --git asource/activation/tests/test_basic_dataset_study.py bworkspace/activation/tests/test_basic_dataset_study.py
index d5224ab..9df79a6 100644
--- asource/activation/tests/test_basic_dataset_study.py
+++ bworkspace/activation/tests/test_basic_dataset_study.py
@@ -58,7 +58,7 @@ def test_basic_dataset_study():
         synthetic = [
             example
             for example in dataset.labeled_retrieval_examples.values()
-            if example.origin == DataOrigin.SYNTHETIC
+            if example.origin == DataOrigin.SYNTHETIC_QA
         ]
         # Every sampled chunk yields exactly one question.
         assert len(synthetic) == STUDY_NUM_QUESTIONS
@@ -73,9 +73,9 @@ def test_basic_dataset_study():
 
         # Labeling pass 1: the synthetic questions.
         labeled = harness.dataset_manager.label_study_examples(
-            dataset.dataset_id, STUDY_NUM_QUESTIONS, synthetic_only=True,
+            dataset.dataset_id, STUDY_NUM_QUESTIONS, origins=[DataOrigin.SYNTHETIC_QA],
         )
-        assert labeled and all(example.origin == DataOrigin.SYNTHETIC for example in labeled)
+        assert labeled and all(example.origin == DataOrigin.SYNTHETIC_QA for example in labeled)
         for example in labeled:
             assert example.oracle_labeled
             assert not set(example.positive_chunk_ids) & set(example.hard_negative_chunk_ids or [])
@@ -90,7 +90,7 @@ def test_basic_dataset_study():
         # positives arrive from the oracle or by inheritance from the known positive documents.
         stats_before = dataset.stats.study_num_label_positives + dataset.stats.study_num_label_inherited_positives
         labeled_native = harness.dataset_manager.label_study_examples(
-            dataset.dataset_id, len(native), synthetic_only=False,
+            dataset.dataset_id, len(native), origins=None,
         )
         assert labeled_native and all(example.origin == DataOrigin.NATIVE for example in labeled_native)
         for example in labeled_native:
````

</details>
