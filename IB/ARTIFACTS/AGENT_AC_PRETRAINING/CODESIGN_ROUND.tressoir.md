# AC and agent training co-design

The accepted implementation is in the existing objects and utility files. AgentTrainer trains its reader adapter on exact recorded activation rows; ACTrainer re-encodes raw parts and trains the AC modules, side adapter and reader adapter with KL or SFT over every retained assistant output.

## What changed

| Area | Result |
| --- | --- |
| Trajectory rows | Captured CPU tensors stay in their existing span dictionaries until the JSONL writer saves `saved_tensors/<digest>.pt` beside the JSONL, flushes the record, then releases the retained buffers. |
| Portable records | Serialized paths are relative to the JSONL folder; deserialization resolves them lazily. Export copies tensors to the new destination, and resume uses their exact saved bytes. |
| Policy training | Thinking removal, padding and packing preserve row positions. The encoder is not consulted; only the reader adapter receives the clipped policy update. |
| AC training | Independent teacher/student offsets select the same complete assistant targets. SFT performs no teacher/cache/drift work; KL retains its current-reader and top-k behavior. |
| Histories | Public continuations retain complete turns within the all-role budget. Recorded roots, reset segments and children retain every recorded output, including incomplete tails, with stable root grouping and inherited selection/weight. |
| Shared base | Side and reader can share one physical model under distinct adapters. Each forward and checkpoint recomputation retains its own adapter and checkpoint policy. |
| Phase handoff | Optimizer reset is explicit; weights and progress clocks persist. Paired AC/reader saves and reader exchange remain supported. |

No new utility module or storage abstraction was introduced. The only mainline test changes are the two full files below and the approved `base_dir` migration in the existing basic AC test.

## Validation

| Check | Result |
| --- | --- |
| Full `test_basic_agent_ac_training.py` on Sky | 3 passed: public compaction, trajectory-QA, RAG-QA; two epochs, held comparisons, inspections and paired saves. |
| Full `test_basic_ac_self_distillation.py` on Sky | 2 passed: KL and SFT; four real easy HotpotQA tasks, eight root/child items, training, exact paired reload, reader exchange, AC rollout and saved-row/cache replay. |
| Private BF16 GPU proofs | 4 passed: independent FP32 head loss/gradient references, shared adapter recomputation and unequal 88/153-token padded/packed fixed-row policy updates across a recurrent chunk boundary. |
| Private CPU proofs | 47 passed: storage/failures/aliases/backpressure, loss/gradient/update oracles, exact histories, reasoning/source frames, selection/capacity, caches, reset and checkpoint/report behavior. |
| Focused existing regressions | 15 passed. Eight marked GPU/slow cases were skipped in that local run; the two requested files were then executed fully on Sky above. |

Both GPU files ran against snapshot `632c256275f3475206df1c0cc8338f512392498d` on `ac-4a-camp` (two 96 GB RTX PRO 6000 GPUs). The full-file pytest commands used `--gpu --slow -s`; the private probes used the same workspace source and a real BF16 Qwen3.5 hybrid decoder. JUnit records show zero failures, errors or skips in those nine GPU cases.

The private GPU checks compare actual parameter gradients and a real reader update, including frozen base/side/AC/row checks. Their independent review passed after strengthening the references and sequence lengths. Storage tests inject failed writes and blocked writers privately. These mechanics require no long learning curve.

| Same four HotpotQA training questions | Loss before → after (reload matches exactly) | Mean task score before → after |
| --- | --- | --- |
| KL | 0.8186 → 0.7728 | 0.85 → 0.85 |
| SFT | 0.8890 → 1.8999 | 0.85 → 0.35 |

The tiny SFT run worsened loss and answer quality. These tests establish the complete pipeline and numerical behavior, without asserting that two updates produce a useful training recipe. The HotpotQA scores use the training questions. Public/QA held comparisons retain matched reader adapters and are also reported as diagnostics.

| Mainline held comparison | Initial AC KL | Final AC KL | Final no-context KL |
| --- | --- | --- | --- |
| compaction | 0.2729 | 0.2555 | 0.2521 |
| traj_qa | 0.7638 | 0.5261 | 0.5214 |
| rag_qa | 0.6757 | 0.5346 | 0.5336 |

Matched no-context KL was slightly lower than AC KL in all three small public/QA runs; these measurements do not establish an AC quality advantage.

Product syntax and whitespace checks, artifact checks, source/staged hash parity and forward/reverse patch checks are part of handoff verification. The custom editor visual view was unavailable; no visual inspection is claimed.

## Review and practical limits

The first independent plan review and final implementation re-review passed. Seven routine final-review corrections covered stable root IDs, native missing raw parts, reporter mode reuse, nested source tool frames, example delegation, drift checkpoint thresholds and inline recorded reasoning. They required no new user decision.

Captured rows are released after successful cache publication, but text/token/raw-part histories still accumulate, active or unsaved runs retain their rows, and the bounded encoder row cache remains. Missing or malformed tensors and incompatible/oversize histories fail explicitly. Historical corpus conversion, DDP and a long quality-training campaign remain outside this change.

## Handoff files

[Product patch](codesign/changes.patch), [source and staged hashes](codesign/source_manifest.json), [validation record](codesign/validation_summary.json), and [application notes](codesign/README.md). The patch contains 21 product files and is relative to the current `/source`; it is already applied in this workspace. Private proof sources and GPU logs remain in `IB/TMP/AGENT_AC_PRETRAINING/codesign_implementation/`.

The tagged GPU snapshot leaves the user's `main` worktree and index in place. The Sky node is paused again, retaining its existing corpus/cache and the new checkpoint/tensor artifacts on disk. Logs, JUnit and the HTML/JSON reports were recovered locally before pausing. A stalled local SSH agent was bypassed without restarting the node; no GPU-driven product correction was needed.

## Full final test files

Both files below are the final product source, also available as normal file links.

<details class="card" data-tressoir-markdown open>
  <summary><span class="card-title"><a href="codesign/activation/tests/test_basic_agent_ac_training.py">test_basic_agent_ac_training.py</a></span><span class="card-oneliner">Complete public-compaction, trajectory-QA and RAG-QA training examples on real data.</span><span class="card-badge">Full file</span></summary>

````python
"""
Slice 3a1 key tests: two AC training epochs per item kind on real data, small on purpose (side
Qwen3.5-0.8B, target Qwen3.5-4B, a few dozen items). The asserts are about mechanics (items built,
finite KL, matching evaluation items, checkpoints written); AC versus empty-context quality is a reported diagnostic,
not a dominance guarantee from a few held examples. The KL values are
printed and reported to HTML under AC_TRAINING_TEST/. The study questions (traj_qa, rag_qa) come
from the dataset study engine (dataset_study_qa_model_name), cached by caching_id.

  uv run pytest activation/tests/test_basic_agent_ac_training.py --gpu --slow -s
"""
import json
import math
import os

import pytest

from activation.ac_model import (
    ActivationContextModelConfig,
    ActivationContextStudyGenerator,
    ActivationContextTrainer,
    ActivationContextTrainingConfig,
    ActivationContextTrainingReporter,
    ActivationContextTrainingItem,
)
from activation.ac_model.ac_model_study import split_items
from activation.common.data_syncing import resolve_path
from activation.dataset.loaders import BrightDataset, NemotronMathDataset, OpenSweTracesDataset, S1DeepResearchDataset, aime_problem_statements
from activation.harness import FREE_DEVICE, HarnessRuntime, HarnessRuntimeConfig, ModelConfig

SIDE_NAME, SIDE_ID = "qwen3.5-0.8b", "Qwen/Qwen3.5-0.8B"
TARGET_NAME, TARGET_ID = "qwen3.5-4b", "Qwen/Qwen3.5-4B"
AC_NAME, SIDE_LORA, TARGET_LORA = "ac_dev", "ac_side", "ac_target"
QA_MODEL_NAME = "RedHatAI/Qwen3.5-9B-FP8-dynamic"
ITEMS, HELD = 48, 8
CACHING_ID = "ac_training_test_codesign_v3_9b"


def _harness(with_study: bool) -> HarnessRuntime:
    model_configs = {SIDE_NAME: ModelConfig(SIDE_NAME, SIDE_ID), TARGET_NAME: ModelConfig(TARGET_NAME, TARGET_ID)}
    if with_study:
        model_configs[QA_MODEL_NAME] = ModelConfig(QA_MODEL_NAME, QA_MODEL_NAME)
    harness = HarnessRuntime(HarnessRuntimeConfig(model_configs=model_configs, dataset_study_qa_model_name=QA_MODEL_NAME if with_study else None))
    harness.module_manager.register_lora(TARGET_LORA, TARGET_NAME, rank=64)
    harness.module_manager.register_ac_model(ActivationContextModelConfig(AC_NAME, SIDE_NAME, SIDE_LORA, TARGET_NAME, TARGET_LORA))
    return harness


def _trainer(harness: HarnessRuntime) -> ActivationContextTrainer:
    return ActivationContextTrainer(harness, ActivationContextTrainingConfig(loss_kind="kl", examples_per_update=8, num_epochs=2))


def _print_item(item: ActivationContextTrainingItem, tokenizer) -> None:
    messages = item.in_context_messages or item.ac_messages
    question = next((message.get("content", "") for message in reversed(messages) if message.get("role") == "user"), "")
    if isinstance(question, list):
        question = "".join(piece.get("text", "") for piece in question if isinstance(piece, dict))
    answer = "\n".join(tokenizer.decode(ids, skip_special_tokens=False) for ids in item.assistant_token_ids)
    print(f"  [{item.item_id}] Q: {str(question)[-160:]!r} A: {answer[:80]!r}; assistant spans={len(item.assistant_token_ids)}")


def _run(harness: HarnessRuntime, items: list[ActivationContextTrainingItem], name: str) -> None:
    """Two epochs, exact held/reference comparisons with matched adapters, and paired saves."""
    assert len(items) > HELD + 8, f"only {len(items)} items"
    held, train, dropped_shared = split_items(items, HELD, len(items))
    assert train and held
    print(f"{len(train)} train, {len(held)} held, {dropped_shared} shared-origin items excluded")
    references = ActivationContextStudyGenerator(harness, AC_NAME).transform_for_eval_reference(held, "no_context")
    trainer = _trainer(harness)
    reporter = ActivationContextTrainingReporter(str(resolve_path(f"AC_TRAINING_TEST/{name}")), f"AC training test: {name}")
    try:
        before = trainer.eval(AC_NAME, held, release=False, reporter=reporter, reference_name="untrained_ac")
        no_context = trainer.eval(AC_NAME, references, release=False, reporter=reporter, reporting_name="initial_no_context", reference_name="no_context")
        stats = trainer.train(AC_NAME, train, reporting_data=held, reporter=reporter)
        after = stats.reporting
        matched_no_context = trainer.eval(AC_NAME, references, reporter=reporter, reporting_name="final_no_context")
        assert len(stats.epochs) == 2
        assert all(len(epoch.samples) == min(4, after["items"]) for epoch in stats.epochs)
        print(f"\n=== {name}: held KL untrained {before['kl']:.4f}, no-context {no_context['kl']:.4f}, after two epochs {after['kl']:.4f} "
              f"(agreement {before['agreement']:.3f} -> {after['agreement']:.3f}) ===")
        print(json.dumps(stats.summarize(), indent=2))
        assert stats.steps >= 2 and all(value == value and value < float("inf") for value in stats.kl)
        assert stats.checkpoint_path and os.path.exists(os.path.join(stats.checkpoint_path, "ac_modules.pt"))
        assert stats.target_lora_checkpoint_path
        assert after["items"] == before["items"] == no_context["items"] == matched_no_context["items"] > 0
        assert all(math.isfinite(result["kl"]) for result in (before, no_context, after, matched_no_context))
        comparison = {"initial_ac": before, "initial_no_context": no_context,
                      "final_ac": after, "final_no_context": matched_no_context,
                      "ac_kl_advantage": matched_no_context["kl"] - after["kl"]}
        resolve_path(f"AC_TRAINING_TEST/{name}/comparison.json").write_text(json.dumps(comparison, indent=2))
        print(f"Matched-adapter AC KL advantage (positive is better): {comparison['ac_kl_advantage']:+.6f}")
    finally:
        reporter.finish()
        ac_model = harness.module_manager.get_ac_model(AC_NAME)
        ac_model.release()
        harness.module_manager.free_lora(ac_model.config.target_model_name, ac_model.config.target_model_lora_name)
        harness.module_manager.free_lora(ac_model.config.base_side_model_name, ac_model.config.base_side_model_lora_name)   # adapters first: a base with adapters cannot be freed
        for loaded in harness.loaded_models.values():
            loaded.engine_to_device(FREE_DEVICE)
            loaded.model_to_device(FREE_DEVICE)


@pytest.mark.gpu
@pytest.mark.slow
def test_ac_compaction() -> None:
    harness = _harness(with_study=False)
    swe = OpenSweTracesDataset.load(harness, max_examples=ITEMS, min_chars=40_000)
    math = NemotronMathDataset.load(harness, max_examples=ITEMS, subset="tir", excluded_problems=aime_problem_statements(), min_chars=40_000, min_tool_calls=1)
    generator = ActivationContextStudyGenerator(harness, AC_NAME)
    items = generator.generate_compaction_samples(swe.dataset_id, ITEMS // 2, seed=0) + generator.generate_compaction_samples(math.dataset_id, ITEMS // 2, seed=0)
    print(f"\n{len(items)} compaction items; depths {sorted({item.info['depth'] for item in items})}; "
          f"assistant spans {sum(len(item.assistant_token_ids) for item in items)}; "
          f"assistant targets {sum(len(ids) for item in items for ids in item.assistant_token_ids)}")
    assert all(item.assistant_token_ids and all(item.assistant_token_ids) for item in items)
    _run(harness, items, "compaction")


@pytest.mark.gpu
@pytest.mark.slow
def test_ac_trajectory_qa() -> None:
    harness = _harness(with_study=True)
    s1 = S1DeepResearchDataset.load(harness, max_examples=ITEMS)
    swe = OpenSweTracesDataset.load(harness, max_examples=ITEMS)
    harness.dataset_manager.build_bm25_indexes()
    generator = ActivationContextStudyGenerator(harness, AC_NAME)
    items = (generator.generate_trajectory_qa_samples(s1.dataset_id, ITEMS // 2, seed=0, caching_id=CACHING_ID)
             + generator.generate_trajectory_qa_samples(swe.dataset_id, ITEMS // 2, seed=0, caching_id=CACHING_ID))
    harness.loaded_models[QA_MODEL_NAME].engine_to_device(FREE_DEVICE)
    print(f"\n{len(items)} traj_qa items; distractors {sorted({item.info['distractors'] for item in items})}")
    for item in items[:3]:
        _print_item(item, harness.loaded_models[TARGET_NAME].tokenizer)
    _run(harness, items, "traj_qa")


@pytest.mark.gpu
@pytest.mark.slow
def test_ac_rag_qa() -> None:
    harness = _harness(with_study=True)
    bright = BrightDataset.load(harness, max_examples=100, domain="biology", max_corpus_documents=2000)
    harness.dataset_manager.build_bm25_indexes()
    generator = ActivationContextStudyGenerator(harness, AC_NAME)
    items = generator.generate_rag_qa_samples(bright.dataset_id, ITEMS, seed=0, caching_id=CACHING_ID)
    harness.loaded_models[QA_MODEL_NAME].engine_to_device(FREE_DEVICE)
    print(f"\n{len(items)} rag_qa items; passages per item {sorted({item.info['passages'] for item in items})}")
    for item in items[:3]:
        _print_item(item, harness.loaded_models[TARGET_NAME].tokenizer)
    _run(harness, items, "rag_qa")
````

</details>

<details class="card" data-tressoir-markdown open>
  <summary><span class="card-title"><a href="codesign/activation/tests/test_basic_ac_self_distillation.py">test_basic_ac_self_distillation.py</a></span><span class="card-oneliner">Four easy HotpotQA tasks through the custom program, KL/SFT training, paired reload and AC rollout/cache pipeline.</span><span class="card-badge">Full file</span></summary>

````python
"""Four real HotpotQA problems through rollout, AC self-distillation and AC rollout.

The custom program retrieves evidence, asks one real solver subagent, then lets the
main agent answer. No mocks, synthetic trajectories or gold answers enter the program.
Each loss mode runs one epoch with two optimizer updates. Scores are printed as
same-problem diagnostics; this small test does not assert a quality improvement.

    uv run pytest activation/tests/test_basic_ac_self_distillation.py --gpu --slow -s

Both loss modes use the paired-history, loss_kind and folder-local row APIs.
"""
import json
import math
import uuid
from pathlib import Path

import pytest
import torch

from activation.ac_model import (
    ActivationContextModelConfig,
    ActivationContextStudyGenerator,
    ActivationContextTrainer,
    ActivationContextTrainingConfig,
    ActivationContextTrainingReporter,
)
from activation.agent import AgentConfig, AgenticProgram, AgentRunResult, RolloutReporter, SemanticSearchTool
from activation.agent.agent_utils import load_ac_rows
from activation.common.data_syncing import resolve_path
from activation.dataset.loaders import HotpotQADataset
from activation.harness import FREE_DEVICE, HarnessRuntime, HarnessRuntimeConfig, ModelConfig

SIDE_NAME, SIDE_ID = "self_distill_side", "Qwen/Qwen3.5-0.8B"
READER_NAME, READER_ID = "self_distill_reader", "Qwen/Qwen3.5-4B"
PROBLEMS = 4


class RetrieveThenSolveProgram(AgenticProgram):
    """Actual retrieval and a solver precede the main agent's own recorded answer."""
    name = "retrieve_then_solve"

    def execute(self) -> AgentRunResult:
        question = self.agent.agent_config.dataset_task.task_datum["question"]
        evidence = self.agent.run_tool("semantic_search", query=question, top_k=2)
        if evidence.is_error:
            raise RuntimeError(evidence.output)
        solver = self.agent.run_subagent(
            f"Find evidence for this question using semantic_search, then submit a short answer: {question}",
            max_duration=60.0,
        )
        self.agent.augment_context([
            evidence,
            solver,
            "Use the retrieved evidence and check the solver's suggestion. You may search again. "
            "Submit only the short answer to the original question.",
        ])
        return self.agent.run()


def _harness(ac_name: str, side_lora: str, reader_lora: str) -> HarnessRuntime:
    harness = HarnessRuntime(HarnessRuntimeConfig(
        model_configs={
            SIDE_NAME: ModelConfig(SIDE_NAME, SIDE_ID),
            READER_NAME: ModelConfig(READER_NAME, READER_ID),
        },
        agent_max_concurrent=2,
    ))
    harness.module_manager.register_lora(reader_lora, READER_NAME, rank=16)
    harness.module_manager.register_ac_model(ActivationContextModelConfig(
        ac_name, SIDE_NAME, side_lora, READER_NAME, reader_lora,
        side_lora_rank=16,
    ))
    return harness


def _tasks(harness: HarnessRuntime):
    # Select by source difficulty and question length, never by model success.
    # The bounded source prefix supplies ordinary distractor documents as well.
    dataset = HotpotQADataset.load(harness, max_examples=128, split="train", seed=0)
    harness.dataset_manager.register_dataset(dataset)
    harness.dataset_manager.build_bm25_indexes()
    easy = [task for task in dataset.scorable_tasks.values() if task.task_datum.get("level") == "easy"]
    assert len(easy) >= PROBLEMS, f"expected four easy questions in the source prefix, found {len(easy)}"
    tasks = sorted(easy, key=lambda task: (len(task.task_datum["question"]), task.task_id))[:PROBLEMS]
    for task in tasks:
        print(f"HotpotQA {task.task_id}: {task.task_datum['question']}")
    return tasks


def _configs(tasks, ac_name: str | None, reader_lora: str | None) -> list[AgentConfig]:
    return [AgentConfig(
        agent_name=f"hotpot_{index}",
        system_prompt=(
            "Answer the question from the Wikipedia corpus using semantic_search. "
            "Check the evidence and call submit_answer with the short answer only."
        ),
        user_prompt=task.agent_prompt,
        dataset_task=task,
        model_name=READER_NAME,
        lora_name=reader_lora,
        ac_model_name=ac_name,
        tools={"semantic_search": (SemanticSearchTool, {"dataset_id": task.dataset_id, "max_chars": 1200}), "subagent": None},
        agentic_program=(RetrieveThenSolveProgram, {}),
        metadata={"program": RetrieveThenSolveProgram.name, "benchmark": "hotpotqa"},
        call_kwargs={"sampling_params": {"max_tokens": 1024, "temperature": 0.0}},
        max_turns=4,
        max_duration=120.0,
        max_tool_errors=2,
        compaction_threshold_tokens=16_384,
        absolute_trajectory_cap=32_000,
    ) for index, task in enumerate(tasks)]


def _rollouts(harness, configs, caching_id: str, report_folder: Path, seed: int) -> list[AgentRunResult]:
    reporter = RolloutReporter(str(report_folder), title=f"HotpotQA: {caching_id}")
    try:
        runs = harness.rollout_manager.perform_single_rollouts(
            configs, seed=seed, caching_id=caching_id, perform_scoring=True, reporter=reporter,
        )
    finally:
        reporter.finish()
    assert len(runs) == PROBLEMS
    for run in runs:
        assert run.finish_reason != "error", run.score_feedback
        assert len(run.subagent_results) == 1
        assert run.subagent_results[0].finish_reason != "error", run.subagent_results[0].score_feedback
        assert run.score is not None and math.isfinite(run.score)
        print(f"{run.agent_config.agent_name}: score={run.score:.3f}, answer={run.answer!r}, "
              f"finish={run.finish_reason}, rows={run.num_ac_rows}")
    return runs


def _segments(run: AgentRunResult):
    for segment in [*run.compactions, run]:
        yield segment
        for child in segment.subagent_results:
            yield from _segments(child)


def _spans(runs: list[AgentRunResult]):
    for root in runs:
        for segment in _segments(root):
            yield from segment.prompt_ac_spans
            for step in segment.trajectory:
                yield from step.get("ac_spans", [])


def _assert_coverage(runs, items) -> None:
    recorded = sum(
        len(step.get("token_ids") or [])
        for root in runs for segment in _segments(root) for step in segment.trajectory
        if step["role"] == "assistant"
    )
    selected = sum(len(ids) for item in items for ids in item.assistant_token_ids)
    assert selected == recorded > 0, (selected, recorded)
    assert all(item.weight == 1.0 for item in items)


def _assert_eval(summary, count: int) -> None:
    assert summary["items"] == count > 0
    assert summary["dropped_too_long"] == 0
    assert summary["loss"] is not None and math.isfinite(summary["loss"])


def _check_saved_rows(runs, caching_id: str, width: int) -> None:
    folder = resolve_path(f"ROLLOUTS/{caching_id}", create=False)
    assert (folder / "rollouts.jsonl").is_file()
    assert (folder / "saved_tensors").is_dir()
    spans = list(_spans(runs))
    assert spans, "the AC rollouts must produce actual captured rows"
    for span in spans:
        assert "rows" not in span, "successfully saved results must release their tensor fields"
        path = Path(span["row_path"])
        assert path.is_absolute() and path.is_file()
        assert path.parent == (folder / "saved_tensors").resolve()
    rows = load_ac_rows(spans[0])
    assert rows.device.type == "cpu" and not rows.requires_grad
    assert rows.shape == (spans[0]["length"], width)
    assert torch.isfinite(rows).all()


def _release(harness, ac_name: str) -> None:
    for loaded in harness.loaded_models.values():
        loaded.engine_to_device(FREE_DEVICE)
    ac_model = harness.module_manager.get_ac_model(ac_name)
    ac_model.release()
    for model_name, lora_name in (
        (READER_NAME, ac_model.config.target_model_lora_name),
        (SIDE_NAME, ac_model.config.base_side_model_lora_name),
    ):
        harness.module_manager.free_lora(model_name, lora_name)
    for loaded in harness.loaded_models.values():
        loaded.model_to_device(FREE_DEVICE)


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("loss_kind", ["kl", "sft"])
def test_basic_ac_self_distillation(loss_kind: str) -> None:
    # A new namespace prevents cached rollouts/checkpoints from hiding this run.
    run_id = f"hotpot_ac_{loss_kind}_{uuid.uuid4().hex[:10]}"
    ac_name, side_lora, reader_lora = run_id, f"{run_id}_side", f"{run_id}_reader"
    folder = resolve_path(f"AC_SELF_DISTILLATION_TEST/{run_id}")
    harness = _harness(ac_name, side_lora, reader_lora)
    training_reporter = None
    try:
        tasks = _tasks(harness)
        before_configs = _configs(tasks, ac_name=None, reader_lora=None)
        before_runs = _rollouts(harness, before_configs, f"{run_id}_text", folder / "text_rollouts", seed=0)
        assert all(run.num_ac_rows == 0 for run in before_runs)
        assert all(len(run.injected_input) >= 2 for run in before_runs)

        config = ActivationContextTrainingConfig(
            loss_kind=loss_kind,
            num_epochs=1,
            updates_per_epoch=2,
            reporting_interval=1.0,
            completion_samples=1,
            completion_max_new_tokens=32,
            logits_chunk_tokens=128,
            seed=0,
        )
        study = ActivationContextStudyGenerator(harness, ac_name)
        # Explicitly include all four for this execution example in both modes.
        # The SFT default score_1 selection is tested privately; scores stay real.
        items = study.items_from_run_results(before_runs, loss_kind=loss_kind, selection="all")
        _assert_coverage(before_runs, items)
        assert len(items) >= PROBLEMS
        trainer = ActivationContextTrainer(harness, config)
        training_reporter = ActivationContextTrainingReporter(str(folder / "training"), f"HotpotQA self-distillation: {loss_kind}")
        before = trainer.eval(ac_name, items, reporter=training_reporter, reporting_name="before")
        _assert_eval(before, len(items))
        stats = trainer.train(ac_name, items, reporting_data=items, reporter=training_reporter)
        after = stats.reporting
        _assert_eval(after, len(items))
        assert stats.steps == 2 and len(stats.epochs) == 1
        assert stats.loss_kind == loss_kind and all(math.isfinite(value) for value in stats.loss)
        assert stats.checkpoint_path and (Path(stats.checkpoint_path) / "ac_modules.pt").is_file()
        assert stats.target_lora_checkpoint_path and Path(stats.target_lora_checkpoint_path).is_dir()
        if loss_kind == "sft":
            assert stats.teacher_tokens == 0 and all(row["teacher_tokens"] == 0 for row in after["per_item"])
            assert after["kl"] is None and after["agreement"] is None
        print(json.dumps(stats.summarize(), indent=2))

        ac_model = harness.module_manager.get_ac_model(ac_name)
        completed = ac_model.load_completed_epoch()
        assert resolve_path(completed["ac_checkpoint"], create=False).resolve() == Path(stats.checkpoint_path).resolve()
        reloaded = ActivationContextTrainer(harness, config).eval(ac_name, items)
        _assert_eval(reloaded, len(items))
        assert math.isclose(reloaded["loss"], after["loss"], rel_tol=5e-3, abs_tol=5e-3)
        harness.module_manager.exchange_lora(READER_NAME, reader_lora)

        ac_configs = _configs(tasks, ac_name=ac_name, reader_lora=reader_lora)
        ac_cache = f"{run_id}_ac"
        ac_runs = _rollouts(harness, ac_configs, ac_cache, folder / "ac_rollouts", seed=1)
        assert all(run.ac_model_name == ac_name and run.lora_name == reader_lora for run in ac_runs)
        assert all(len(run.prompt_ac_spans) >= 2 for run in ac_runs), "program should inject multiple actual AC parts"
        _check_saved_rows(ac_runs, ac_cache, ac_model.d_target)

        cached = harness.rollout_manager.perform_single_rollouts(ac_configs, seed=1, caching_id=ac_cache, perform_scoring=True)
        assert len(cached) == PROBLEMS
        assert [run.answer for run in cached] == [run.answer for run in ac_runs]
        _check_saved_rows(cached, ac_cache, ac_model.d_target)
        # Native AC histories also convert to differentiable training data.
        ac_items = study.items_from_run_results(cached, loss_kind=loss_kind, selection="all")
        _assert_coverage(cached, ac_items)

        summary = {
            "loss_kind": loss_kind,
            "task_ids": [task.task_id for task in tasks],
            "text_scores": [run.score for run in before_runs],
            "ac_scores": [run.score for run in ac_runs],
            "training_items": len(items),
            "ac_history_items": len(ac_items),
            "before_loss": before["loss"],
            "after_loss": after["loss"],
            "reloaded_loss": reloaded["loss"],
            "checkpoint": completed,
            "quality_scope": "same four training questions; diagnostic only",
        }
        (folder / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
    finally:
        if training_reporter is not None:
            training_reporter.finish()
        _release(harness, ac_name)
````

</details>

## Product changes against the source

<details class="card" data-tressoir-markdown open>
  <summary><span class="card-title"><a href="codesign/activation/agent/agent_utils.py">activation/agent/agent_utils.py</a></span><span class="card-oneliner">Capture, portable tensor persistence, lazy resolution, exact loading and retained-row release in existing utilities.</span><span class="card-badge">Diff</span></summary>

````diff
diff --git a/activation/agent/agent_utils.py b/activation/agent/agent_utils.py
--- a/activation/agent/agent_utils.py
+++ b/activation/agent/agent_utils.py
@@ -21,20 +21,175 @@
 from __future__ import annotations
 
 import json
+import hashlib
+import os
 import re
+import tempfile
 import typing as t
 import uuid
-from dataclasses import dataclass, field
+from dataclasses import dataclass, field, fields
+from pathlib import Path
+
+import torch
 
 from activation.common.ac_parts import encode_with_part_sentinels, flatten_with_sentinels
 
 if t.TYPE_CHECKING:
     from .agent import Agent
     from .agent_config import AgentConfig
+    from .agent_config import AgentRunResult, TrajectoryStep
     from .agent_tools import ToolCallResult
     from activation.harness import HarnessRuntime
 
 PARSE_ERROR_TOOL = "parse_error"
+
+
+def capture_ac_spans(spans: list[tuple[int, int]], rows: list[torch.Tensor]) -> list[dict]:
+    """Keep the exact final-cast observation; already contiguous CPU rows are not copied."""
+    if len(spans) != len(rows):
+        raise ValueError("AC spans and captured rows differ in count")
+    captured = []
+    for (start, end), tensor in zip(spans, rows):
+        span = {"start": start, "length": end - start, "rows": tensor.detach().cpu().contiguous()}
+        load_ac_rows(span)
+        captured.append(span)
+    return captured
+
+
+def trajectory_step_dict(step: "TrajectoryStep") -> dict:
+    """Convert the existing record without asdict's recursive tensor deepcopy."""
+    return {entry.name: getattr(step, entry.name) for entry in fields(step)}
+
+
+def _map_ac_records(data, convert) -> dict:
+    """Visit only owned row fields; raw parts and arbitrary tool payloads are untouched."""
+    record = dict(data if isinstance(data, dict) else vars(data))
+    config = record.get("agent_config")
+    if config is not None and not isinstance(config, dict):
+        record["agent_config"] = config.serialize()
+    if "prompt_ac_spans" in record:
+        record["prompt_ac_spans"] = [convert(span) for span in record["prompt_ac_spans"]]
+    if "trajectory" in record:
+        record["trajectory"] = [dict(step, ac_spans=[convert(span) for span in step["ac_spans"]])
+                                if "ac_spans" in step else dict(step) for step in record["trajectory"]]
+    for key in ("compactions", "subagent_results"):
+        if key in record:
+            record[key] = [_map_ac_records(child, convert) for child in record[key]]
+    return record
+
+
+def _ac_rows_digest(rows: torch.Tensor) -> str:
+    digest = hashlib.sha256(f"{rows.dtype}:{tuple(rows.shape)}:".encode())
+    digest.update(rows.contiguous().view(torch.uint8).numpy().tobytes())
+    return digest.hexdigest()
+
+
+def load_ac_rows(span: dict) -> torch.Tensor:
+    """Read captured rows or a resolved tensor file, preserving the recorded CPU dtype."""
+    rows = span.get("rows")
+    path = None
+    if rows is None:
+        if not span.get("row_path"):
+            raise ValueError("AC span has no captured rows; regenerate the rollout")
+        path = Path(span["row_path"])
+        if not path.is_absolute():
+            raise ValueError("AC row path is unresolved; deserialize with the JSONL parent as base_dir")
+        try:
+            rows = torch.load(path, map_location="cpu", weights_only=True)
+        except FileNotFoundError:
+            raise
+        except Exception as exc:
+            raise ValueError(f"Cannot read AC rows from {path}: {exc}") from exc
+    if not isinstance(rows, torch.Tensor) or rows.ndim != 2 or not rows.is_floating_point():
+        raise ValueError("AC rows must be a floating-point matrix")
+    if rows.device.type != "cpu" or rows.requires_grad or not rows.is_contiguous():
+        raise ValueError("Captured AC rows must be detached contiguous CPU tensors")
+    if rows.shape[0] != span.get("length") or rows.shape[0] < 1 or rows.shape[1] < 1:
+        raise ValueError("AC row shape does not match its span")
+    if path is not None and len(path.stem) == 64 and _ac_rows_digest(rows) != path.stem:
+        raise ValueError(f"AC row content hash mismatch: {path}")
+    return rows
+
+
+def serialize_ac_rows(data: dict, *, base_dir: Path | None = None) -> dict:
+    """Publish folder-local tensor files and return relative references; retain input buffers."""
+    destination = None if base_dir is None else Path(base_dir).resolve()
+
+    def save(span: dict) -> dict:
+        if destination is None:
+            raise ValueError("Serializing AC rows requires the JSONL parent as base_dir")
+        source = dict(span)
+        if source.get("row_path") and not Path(source["row_path"]).is_absolute():
+            source["row_path"] = str(destination / source["row_path"])
+        rows = load_ac_rows(source)
+        digest = _ac_rows_digest(rows)
+        folder = destination / "saved_tensors"
+        folder.mkdir(parents=True, exist_ok=True)
+        target = folder / f"{digest}.pt"
+        if target.exists():
+            load_ac_rows({"length": span["length"], "row_path": str(target)})
+        else:
+            temporary = None
+            try:
+                with tempfile.NamedTemporaryFile(dir=folder, prefix=f".{digest}.", suffix=".tmp", delete=False) as handle:
+                    temporary = Path(handle.name)
+                    torch.save(rows, handle)
+                    handle.flush()
+                    os.fsync(handle.fileno())
+                os.replace(temporary, target)
+            finally:
+                if temporary is not None:
+                    temporary.unlink(missing_ok=True)
+        return {key: value for key, value in span.items() if key not in ("rows", "row_path")} | {
+            "row_path": f"saved_tensors/{digest}.pt"}
+
+    return _map_ac_records(data, save)
+
+
+def deserialize_ac_rows(data: dict, *, base_dir: Path | None = None) -> dict:
+    """Resolve references once, without loading tensor payloads."""
+    def resolve(span: dict) -> dict:
+        span = dict(span)
+        if span.get("row_path"):
+            path = Path(span["row_path"])
+            if not path.is_absolute():
+                if base_dir is None:
+                    raise ValueError("Relative AC row references require the JSONL parent as base_dir")
+                path = Path(base_dir) / path
+            span["row_path"] = str(path.resolve())
+        return span
+    return _map_ac_records(data, resolve)
+
+
+def release_ac_rows(result: "AgentRunResult", serialized: dict, *, base_dir: Path) -> None:
+    """After JSONL publication, replace tensors in the actual retained span dictionaries."""
+    def release(record, saved):
+        data = record if isinstance(record, dict) else vars(record)
+        pairs = [(data.get("prompt_ac_spans", []), saved.get("prompt_ac_spans", []))]
+        pairs += [(step.get("ac_spans", []), saved_step.get("ac_spans", []))
+                  for step, saved_step in zip(data.get("trajectory", []), saved.get("trajectory", []))]
+        for spans, saved_spans in pairs:
+            for span, saved_span in zip(spans, saved_spans):
+                span["row_path"] = str((Path(base_dir) / saved_span["row_path"]).resolve())
+                span.pop("rows", None)
+        for key in ("compactions", "subagent_results"):
+            for child, saved_child in zip(data.get(key, []), saved.get(key, [])):
+                release(child, saved_child)
+    release(result, serialized)
+
+
+def ac_spans_for_report(spans: list[dict]) -> list[dict]:
+    """Inspection metadata only; reports are not another tensor dataset."""
+    result = []
+    for span in spans:
+        item = {key: value for key, value in span.items() if key not in ("rows", "row_path")}
+        rows = span.get("rows")
+        if rows is not None:
+            item.update(shape=list(rows.shape), dtype=str(rows.dtype))
+        if span.get("row_path"):
+            item["saved_tensor"] = Path(span["row_path"]).name
+        result.append(item)
+    return result
 ASSISTANT_SENTINEL = "\u241f TRESSOIR_ASSISTANT_TURN \u241f"   # never in real text; marks where the sampled tokens sit in a rendering
 TOOL_CALL_BLOCK = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)
 TOOL_CALL_OPEN = re.compile(r"<tool_call>(?!.*</tool_call>)(.*)$", re.S)   # an unterminated block (max_tokens hit)
````

</details>

<details class="card" data-tressoir-markdown open>
  <summary><span class="card-title"><a href="codesign/activation/agent/agent_config.py">activation/agent/agent_config.py</a></span><span class="card-oneliner">Folder-aware run serialization and captured span records.</span><span class="card-badge">Diff</span></summary>

````diff
diff --git a/activation/agent/agent_config.py b/activation/agent/agent_config.py
--- a/activation/agent/agent_config.py
+++ b/activation/agent/agent_config.py
@@ -7,6 +7,7 @@
 import json
 import typing as t
 from dataclasses import dataclass, field, replace
+from pathlib import Path
 
 from activation.common.utils import class_spec_name, resolve_class_name
 
@@ -129,7 +130,7 @@
                                                                # (recorded whether or not it was rendered), subagent_index. Empty on old records.
     messages: list[dict] = field(default_factory=list)         # the dialect messages this step appended, activation_context parts inline and in
                                                                # order: one assistant message, the tool messages, or the nudge
-    ac_spans: list[dict] = field(default_factory=list)         # [{"start", "length"}] placeholder runs inside token_ids, one per part of `messages`
+    ac_spans: list[dict] = field(default_factory=list)         # {start, length, rows (CPU) | row_path}, relative to this step's token_ids
     # Token segment (slice 2): the prompt the model read is AgentRunResult.prompt_token_ids + every step's token_ids in order.
     token_ids: list[int] = field(default_factory=list)         # assistant: the sampled tokens verbatim; tool/user: the template's wrapper up to the next generation prompt
     logprobs: list[float] = field(default_factory=list)        # assistant only: the engine's log-prob of each sampled token (pi_old); empty when not recorded
@@ -161,7 +162,7 @@
     seed: int = 0
     prompt_token_ids: list[int] = field(default_factory=list)  # the segment's first prompt (system + first messages + tools, templated once)
     prompt_messages: list[dict] = field(default_factory=list)  # the dialect messages of that prompt, parts inline (a compaction tree lives here)
-    prompt_ac_spans: list[dict] = field(default_factory=list)  # [{"start", "length"}] placeholder runs inside prompt_token_ids
+    prompt_ac_spans: list[dict] = field(default_factory=list)  # {start, length, rows (CPU) | row_path}, relative to prompt_token_ids
     ac_model_name: str | None = None                           # the encoder that produced the rows, and its version at run time
     ac_model_version: int | None = None
     source: str = "policy"                                     # "policy" | "hinted:<model>" | "oracle:<model>": who produced this run
@@ -201,20 +202,18 @@
                 total[key] = total.get(key, 0) + value
         return total
 
-    def serialize(self) -> dict:
-        data = {key: value for key, value in self.__dict__.items()
-                if key not in ("agent_config", "subagent_results", "compactions", "trajectory", "answer")}
+    def serialize(self, *, base_dir: Path | None = None) -> dict:
+        from .agent_utils import serialize_ac_rows
+        data = dict(self.__dict__)
         data["agent_config"] = self.agent_config.serialize()
-        data["answer"] = _jsonable(self.answer)
-        data["trajectory"] = _jsonable(self.trajectory)
-        data["subagent_results"] = [result.serialize() for result in self.subagent_results]
-        data["compactions"] = [result.serialize() for result in self.compactions]
-        return data
+        return _jsonable(serialize_ac_rows(data, base_dir=base_dir))
 
     @staticmethod
     def deserialize(data: dict, harness: "HarnessRuntime | None" = None, agent_config: AgentConfig | None = None,
-                    strict: bool = False) -> "AgentRunResult":
+                    strict: bool = False, *, base_dir: Path | None = None) -> "AgentRunResult":
         """With agent_config given (the caller's live config), the stored config is not rebuilt. `strict`: see AgentConfig.deserialize."""
+        from .agent_utils import deserialize_ac_rows
+        data = deserialize_ac_rows(data, base_dir=base_dir)
         fields = {field_.name for field_ in AgentRunResult.__dataclass_fields__.values()}
         data = {key: value for key, value in data.items() if key in fields}          # cache rows carry extra keys
         config_data = data.pop("agent_config")
````

</details>

<details class="card" data-tressoir-markdown open>
  <summary><span class="card-title"><a href="codesign/activation/agent/agent.py">activation/agent/agent.py</a></span><span class="card-oneliner">Capture final reader inputs and resume directly from their saved rows.</span><span class="card-badge">Diff</span></summary>

````diff
diff --git a/activation/agent/agent.py b/activation/agent/agent.py
--- a/activation/agent/agent.py
+++ b/activation/agent/agent.py
@@ -27,7 +27,6 @@
 import typing as t
 import uuid
 from copy import deepcopy
-from dataclasses import asdict
 
 import torch
 
@@ -37,7 +36,8 @@
 from .agent_env import AgentEnv
 from .agent_tools import (DEFAULT_TOOLS, AgentTool, CompactionTool, SubagentTool, ToolCallResult, injected_content, segment_messages_with_parts,
                           tool_content, truncate_output, subagent_config_of)
-from .agent_utils import PARSE_ERROR_TOOL, ModelDialect, parameter_types_of
+from .agent_utils import (PARSE_ERROR_TOOL, ModelDialect, parameter_types_of, capture_ac_spans,
+                          trajectory_step_dict, load_ac_rows)
 
 MAX_CONSECUTIVE_NO_TOOL_TURNS = 3   # nudged after each; the run ends with no_tool_call at the third in a row
 MAX_COMPACTION_REFUSALS = 5         # turns that ignore a due compaction before the run ends with compaction_refused
@@ -275,7 +275,7 @@
         self.segment_start = len(ids)
         self.last_sampled = []
         results.prompt_token_ids, results.prompt_messages = list(ids), deepcopy(messages)
-        results.prompt_ac_spans = [{"start": start, "length": end - start} for start, end in spans]
+        results.prompt_ac_spans = capture_ac_spans(spans, rows)
         self._after_append()
 
     def compact(self, summary: str) -> None:
@@ -312,7 +312,7 @@
         self.messages.append(message)
         self.prefix += list(token_ids)
         self.last_sampled = list(token_ids)
-        self.run_results.trajectory.append(asdict(step))
+        self.run_results.trajectory.append(trajectory_step_dict(step))
 
     def append_user_message(self, text: str) -> None:
         message = {"role": "user", "content": text}
@@ -342,12 +342,12 @@
     def _append_step(self, step: TrajectoryStep, messages: list[dict], spans: list[tuple[int, int]], rows: list[torch.Tensor]) -> None:
         offset = len(self.prefix)
         assert [int(r.shape[0]) for r in rows] == [end - start for start, end in spans], "placeholder runs differ from the encoded rows"
-        step.ac_spans = [{"start": start, "length": end - start} for start, end in spans]
+        step.ac_spans = capture_ac_spans(spans, rows)
         self.messages.extend(messages)
         self.prefix += step.token_ids
         self.spans += [(offset + start, offset + end) for start, end in spans]
         self.rows += rows
-        self.run_results.trajectory.append(asdict(step))
+        self.run_results.trajectory.append(trajectory_step_dict(step))
         self._after_append()
 
     def _after_append(self) -> None:
@@ -641,31 +641,27 @@
     def resume_from_run_result(harness: "HarnessRuntime", run_result: AgentRunResult, reporter: "RolloutReporter | None" = None) -> "Agent":
         """
         An agent positioned exactly after the recorded steps of `run_result`'s current segment: messages,
-        prefix and spans from the record, rows re-encoded from the recorded parts (the record never
-        stores rows). Counters continue from the record.
+        prefix, spans and exact captured rows from the record. Counters continue from the record.
         """
         agent = Agent(harness, run_result.agent_config, reporter=reporter, seed=run_result.seed)
         agent.run_results = run_result
         agent.dialect = ModelDialect.for_tokenizer(agent.loaded_model.tokenizer)
         agent._prepare_tools()
-        if agent.ac_model is not None and run_result.ac_model_version is not None and agent.ac_model.version != run_result.ac_model_version:
-            print(f"Agent {agent.agent_id[:8]} - resuming rows with AC version {agent.ac_model.version}, recorded {run_result.ac_model_version}", flush=True)
         agent.messages = deepcopy(run_result.prompt_messages)
         agent.prefix = list(run_result.prompt_token_ids)
         agent.segment_start = len(agent.prefix)
         spans = [(span["start"], span["start"] + span["length"]) for span in run_result.prompt_ac_spans]
-        parts_messages: list[dict] = list(run_result.prompt_messages)
+        rows = [load_ac_rows(span) for span in run_result.prompt_ac_spans]
         for step in run_result.trajectory:
             offset = len(agent.prefix)
             agent.messages.extend(deepcopy(step["messages"]))
             agent.prefix += list(step["token_ids"])
             spans += [(offset + span["start"], offset + span["start"] + span["length"]) for span in step["ac_spans"]]
-            parts_messages += step["messages"]
+            rows += [load_ac_rows(span) for span in step.get("ac_spans", [])]
             if step["role"] == "assistant":
                 agent.last_sampled = list(step["token_ids"])
         agent.spans = spans
-        agent.rows = agent._encode_parts(parts_messages)
-        assert [int(r.shape[0]) for r in agent.rows] == [end - start for start, end in spans], "recorded spans differ from the re-encoded rows"
+        agent.rows = rows
         agent.started = True
         agent._after_append()
         return agent
````

</details>

<details class="card" data-tressoir-markdown open>
  <summary><span class="card-title"><a href="codesign/activation/agent/rollout_caching.py">activation/agent/rollout_caching.py</a></span><span class="card-oneliner">Publish tensors and JSONL together; release rows only after append and flush succeed.</span><span class="card-badge">Diff</span></summary>

````diff
diff --git a/activation/agent/rollout_caching.py b/activation/agent/rollout_caching.py
--- a/activation/agent/rollout_caching.py
+++ b/activation/agent/rollout_caching.py
@@ -15,6 +15,7 @@
 from activation.common.data_syncing import resolve_path
 
 from .agent_config import AgentConfig, AgentRunResult, _class_spec
+from .agent_utils import release_ac_rows
 
 CACHE_FILENAME = "rollouts.jsonl"
 
@@ -100,9 +101,16 @@
     def append(self, result: AgentRunResult) -> None:
         if self.path is None:
             return
-        row = result.serialize() | {"config_key": config_key(result.agent_config)}
+        row = result.serialize(base_dir=self.path.parent) | {"config_key": config_key(result.agent_config)}
         with self.lock:
+            with open(self.path, "a") as handle:
+                offset = handle.tell()
+                try:
+                    handle.write(json.dumps(row, ensure_ascii=False, default=repr) + "\n")
+                    handle.flush()
+                except Exception:
+                    handle.seek(offset)
+                    handle.truncate()
+                    raise
             self.rows[(row["config_key"], int(result.seed))] = row
-            with open(self.path, "a") as handle:
-                handle.write(json.dumps(row, ensure_ascii=False, default=repr) + "\n")
-                handle.flush()
+            release_ac_rows(result, row, base_dir=self.path.parent)
````

</details>

<details class="card" data-tressoir-markdown open>
  <summary><span class="card-title"><a href="codesign/activation/agent/rollout_manager.py">activation/agent/rollout_manager.py</a></span><span class="card-oneliner">Hold the concurrency permit through cache publication and resolve cached paths against their folder.</span><span class="card-badge">Diff</span></summary>

````diff
diff --git a/activation/agent/rollout_manager.py b/activation/agent/rollout_manager.py
--- a/activation/agent/rollout_manager.py
+++ b/activation/agent/rollout_manager.py
@@ -90,7 +90,7 @@
         results: list[AgentRunResult | None] = []
         for config, seed in jobs:
             row = cache.get(config_key(config), seed)
-            results.append(None if row is None else AgentRunResult.deserialize(row, self.harness, agent_config=config))
+            results.append(None if row is None else AgentRunResult.deserialize(row, self.harness, agent_config=config, base_dir=cache.path.parent))
         pending = [index for index, result in enumerate(results) if result is None]
         skipped = 0
         if redo is not None and redo.only:                                       # the rows to redo alone; never-cached tasks stay None
@@ -177,7 +177,10 @@
             try:
                 config, seed = jobs[index]
                 if deadline is None:
-                    return self._run_one(config, seed, perform_scoring, reporter)
+                    result = self._run_one(config, seed, perform_scoring, reporter)
+                    if result.finish_reason != "deadline":
+                        cache.append(result)
+                    return result
                 now = time.time()
                 with counts_lock:
                     counts["skipped" if now >= deadline else "started"] += 1
@@ -192,6 +195,8 @@
                         result.finish_reason = "deadline"
                     if result.agent_config is run_config:
                         result.agent_config = config           # the clip is an execution detail: the result belongs to the job's own config
+                if result.finish_reason != "deadline":
+                    cache.append(result)
                 return result
             finally:
                 limiter.release()
@@ -207,8 +212,6 @@
                             continue
                         results[index] = result
                         finished[index] = (time.time(), result)
-                        if result.finish_reason != "deadline":        # a cut run is not a result: the next launch redoes it
-                            cache.append(result)
                         if probe and on_probe is not None and len(finished) >= len(probe):   # the first N to finish: a straggler must not delay the tuning by its whole duration
                             now = time.time()
                             phases.append(self._phase_metrics(label, started, now, loaded_model, sampler.samples, finished))
````

</details>

<details class="card" data-tressoir-markdown open>
  <summary><span class="card-title"><a href="codesign/activation/agent/rollout_reporter.py">activation/agent/rollout_reporter.py</a></span><span class="card-oneliner">Show row metadata without loading or dumping tensor contents.</span><span class="card-badge">Diff</span></summary>

````diff
diff --git a/activation/agent/rollout_reporter.py b/activation/agent/rollout_reporter.py
--- a/activation/agent/rollout_reporter.py
+++ b/activation/agent/rollout_reporter.py
@@ -16,6 +16,7 @@
 from pathlib import Path
 
 from activation.common.reporting import HtmlReporter, _write_atomic, format_seconds
+from .agent_utils import ac_spans_for_report
 
 if t.TYPE_CHECKING:
     from .agent import Agent
@@ -201,8 +202,9 @@
         _write_atomic(folder / file, json.dumps({
             "agent_id": agent.agent_id, "state": state, "system_prompt": agent.agent_config.system_prompt,
             "user_prompt": agent.agent_config.user_prompt,
-            "trajectory": [{key: value for key, value in step.items() if key not in ("token_ids", "logprobs", "messages")} for step in results.trajectory],
-            "compactions": len(results.compactions), "prompt_ac_spans": results.prompt_ac_spans,
+            "trajectory": [{key: ac_spans_for_report(value) if key == "ac_spans" else value
+                            for key, value in step.items() if key not in ("token_ids", "logprobs", "messages")} for step in results.trajectory],
+            "compactions": len(results.compactions), "prompt_ac_spans": ac_spans_for_report(results.prompt_ac_spans),
             "answer": results.answer, "finish_reason": results.finish_reason, "score": results.score,
         }, indent=1, default=str))
     steps = []
````

</details>

<details class="card" data-tressoir-markdown open>
  <summary><span class="card-title"><a href="codesign/activation/agent_training/agent_training_utils.py">activation/agent_training/agent_training_utils.py</a></span><span class="card-oneliner">Carry exact fixed rows through thinking removal, padding and packing into reader embeddings.</span><span class="card-badge">Diff</span></summary>

````diff
diff --git a/activation/agent_training/agent_training_utils.py b/activation/agent_training/agent_training_utils.py
--- a/activation/agent_training/agent_training_utils.py
+++ b/activation/agent_training/agent_training_utils.py
@@ -7,10 +7,11 @@
 from __future__ import annotations
 
 import typing as t
-from dataclasses import dataclass
+from dataclasses import dataclass, field
 
 import torch
 import torch.utils.checkpoint
+from ..agent.agent_utils import load_ac_rows
 
 if t.TYPE_CHECKING:
     from .agent_trainer import AgentTrainingItem
@@ -25,6 +26,7 @@
     weight: float                  # the advantage A of the whole trajectory
     item_index: int
     needs_old_logprobs: bool       # teacher / hinted items (or runs recorded without log-probs): pi_old comes from a no-grad pass
+    ac_spans: list[dict] = field(default_factory=list)  # absolute sequence offsets; captured rows or resolved file references
 
     @property
     def num_loss_tokens(self) -> int:
@@ -54,18 +56,23 @@
     replaced by `think_replacement` (no loss on it), so the student never conditions on reasoning it will not have.
     """
     run = item.run_results
-    if run.prompt_ac_spans or any(step.get("ac_spans") for step in run.trajectory):
-        # Placeholder ids at part positions would be embedded as pad tokens: training on AC-bearing sequences
-        # needs the rows rebuilt from the recorded parts (deferred to the training-recipe slices).
-        raise ValueError("Training on AC-bearing runs awaits the fixed-row trainer integration")
     token_ids = list(run.prompt_token_ids)
     if not token_ids:
         return None
     loss_mask = [False] * len(token_ids)
     old_logprobs = [0.0] * len(token_ids)
+    ac_spans = [dict(span) for span in run.prompt_ac_spans]
+    if any(span["start"] < 0 or span["start"] + span["length"] > len(token_ids) for span in ac_spans):
+        raise ValueError("AC span lies outside its recorded prompt")
     any_recorded = False
     for step in run.trajectory:
         segment = list(step.get("token_ids") or [])
+        if step.get("ac_spans") and step.get("role") == "assistant":
+            raise ValueError("Sampled assistant tokens cannot contain AC input rows")
+        for span in step.get("ac_spans", []):
+            if span["start"] < 0 or span["start"] + span["length"] > len(segment):
+                raise ValueError("AC span lies outside its recorded token segment")
+            ac_spans.append(dict(span, start=len(token_ids) + span["start"]))
         if step.get("role") == "assistant":
             logprobs = [float(value) for value in (step.get("logprobs") or [])]
             think_end = step.get("think_end")
@@ -94,9 +101,16 @@
     loss_mask[0] = False                                                   # the first token is never predicted
     if not any(loss_mask):
         return None
+    previous_end = 0
+    for span in ac_spans:
+        start, end = span["start"], span["start"] + span["length"]
+        if start < previous_end or end <= start or end > len(token_ids) or any(loss_mask[start:end]):
+            raise ValueError("AC spans must be ordered, nonoverlapping input-only positions within the trajectory")
+        previous_end = end
     return TrainingExample(
         token_ids=token_ids, loss_mask=loss_mask, old_logprobs=old_logprobs, weight=float(item.weight), item_index=item_index,
         needs_old_logprobs=item.ignore_logprobs or not any_recorded,
+        ac_spans=ac_spans,
     )
 
 
@@ -141,6 +155,7 @@
     segment_offsets: list[int]     # where each example starts in its row (0 unless packed)
     model_attention_mask: torch.Tensor | None = None   # what the decoder gets: None = causal only (right padding never reaches a real token)
     cu_seq_lens: torch.Tensor | None = None            # packed: int32 [E + 1] boundaries, for the linear-attention kernels (fla varlen)
+    ac_spans: list[tuple[int, dict]] = field(default_factory=list)  # batch row and span at its padded/packed offset
 
     @property
     def packed(self) -> bool:
@@ -191,6 +206,7 @@
         segment_ids=torch.arange(len(indices), device=device)[:, None].expand(len(indices), length).contiguous(),
         segment_offsets=[0] * len(indices),
         model_attention_mask=None if causal_only else attention_mask.to(device),
+        ac_spans=[(row, span) for row, index in enumerate(indices) for span in examples[index].ac_spans],
     )
 
 
@@ -236,7 +252,44 @@
         segment_offsets=offsets,
         model_attention_mask=None,
         cu_seq_lens=torch.tensor(boundaries, dtype=torch.int32, device=device),
+        ac_spans=[(0, dict(span, start=offset + span["start"]))
+                  for index, offset in zip(indices, offsets) for span in examples[index].ac_spans],
     )
+
+
+def fixed_rows_for_model(span: dict, embedding: torch.nn.Module) -> torch.Tensor:
+    rows = load_ac_rows(span)
+    if rows.shape[1] != embedding.weight.shape[1] or rows.dtype != embedding.weight.dtype:
+        raise ValueError(f"Captured AC rows {tuple(rows.shape)}/{rows.dtype} do not match reader embeddings "
+                         f"{embedding.weight.shape[1]}/{embedding.weight.dtype}")
+    return rows
+
+
+def validate_fixed_rows(items, examples: list[TrainingExample], loaded_model) -> None:
+    """Preflight payloads one at a time without retaining a dataset of loaded tensors."""
+    for example in examples:
+        if not example.ac_spans:
+            continue
+        source_name = items[example.item_index].run_results.agent_config.model_name
+        if source_name != loaded_model.model_config.model_name:
+            source = loaded_model.harness.loaded_models.get(source_name)
+            if source is None or source.model_config.model_id != loaded_model.model_config.model_id:
+                raise ValueError("Captured AC rows belong to a different reader model")
+        for span in example.ac_spans:
+            fixed_rows_for_model(span, loaded_model.model.get_input_embeddings())
+
+
+def training_inputs_embeds(model, batch: Collated) -> torch.Tensor:
+    """Replay the recorded observation for both reference and policy passes; no encoder."""
+    embedding = model.get_input_embeddings()
+    inputs = embedding(batch.input_ids)
+    if batch.ac_spans:
+        inputs = inputs.clone()
+        for row, span in batch.ac_spans:
+            rows = fixed_rows_for_model(span, embedding)
+            start = span["start"]
+            inputs[row, start:start + span["length"]] = rows.to(inputs.device)
+    return inputs
 
 
 class _HeadMatmul(torch.autograd.Function):
````

</details>

<details class="card" data-tressoir-markdown open>
  <summary><span class="card-title"><a href="codesign/activation/agent_training/agent_trainer.py">activation/agent_training/agent_trainer.py</a></span><span class="card-oneliner">Preflight fixed observations and support explicit optimizer reset between phases.</span><span class="card-badge">Diff</span></summary>

````diff
diff --git a/activation/agent_training/agent_trainer.py b/activation/agent_training/agent_trainer.py
--- a/activation/agent_training/agent_trainer.py
+++ b/activation/agent_training/agent_trainer.py
@@ -55,6 +55,8 @@
     set_checkpointing,
     set_learning_rates,
     sampled_logprobs,
+    training_inputs_embeds,
+    validate_fixed_rows,
 )
 
 if t.TYPE_CHECKING:
@@ -114,6 +116,8 @@
         training_data: list[AgentTrainingItem],
         reporting_data: t.Sequence[AgentTrainingItem] = (),
         reporter: AgentTrainingReporter | None = None,
+        *,
+        reset_optimizer: bool = False,
     ) -> AgentTrainingStats:
         """
         One round on `lora_name`: engine to sleep, base + adapter on the GPU, one clipped-surrogate pass
@@ -162,6 +166,8 @@
                             "gradient_checkpointing_min_tokens": variables["gradient_checkpointing_min_tokens"]}
         module_manager.ensure_lora(lora_name)
         base = loaded_model.model
+        validate_fixed_rows(training_data, examples, loaded_model)
+        validate_fixed_rows(reporting_data, reporting_examples, loaded_model)
         device = next(base.parameters()).device
         base.train()
         previous_attn = base.config._attn_implementation
@@ -175,6 +181,8 @@
         disable_dropout(base)
         parameters = module_manager.lora_parameters(lora_name)
         assert parameters, f"LoRA {lora_name!r} has no trainable parameters"
+        if reset_optimizer:
+            self.optimizers.pop(lora_name, None)
         optimizer = persistent_optimizer(self.optimizers, lora_name, [{"params": parameters, "lr": config.learning_rate}], device,
                                          betas=config.adam_betas, eps=config.adam_eps, weight_decay=config.weight_decay)
         pad_token_id = loaded_model.tokenizer.pad_token_id
@@ -278,7 +286,7 @@
     # ----------------------------------------------------------------------------- internals
     def _hidden_and_logprobs(self, loaded_model, lora_name: str, batch: Collated) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
         base = loaded_model.model
-        inputs_embeds = base.get_input_embeddings()(batch.input_ids)
+        inputs_embeds = training_inputs_embeds(base, batch)
         kwargs = dict(self.config.decoder_kwargs)
         attention_mask = batch.model_attention_mask
         if batch.packed:
````

</details>

<details class="card" data-tressoir-markdown open>
  <summary><span class="card-title"><a href="codesign/activation/ac_model/ac_model_utils.py">activation/ac_model/ac_model_utils.py</a></span><span class="card-oneliner">Build complete assistant target histories, preserve reasoning, validate paired views/capacity and normalize public continuations.</span><span class="card-badge">Diff</span></summary>

````diff
diff --git a/activation/ac_model/ac_model_utils.py b/activation/ac_model/ac_model_utils.py
--- a/activation/ac_model/ac_model_utils.py
+++ b/activation/ac_model/ac_model_utils.py
@@ -11,6 +11,7 @@
 import threading
 import time
 import typing as t
+from copy import deepcopy
 from collections import OrderedDict
 from concurrent.futures import Future
 
@@ -62,7 +63,8 @@
     renders one sentinel string per part; the text is split at the sentinels and the pieces tokenized
     (no special tokens, as the template's own tokenization). The caller writes rows over the spans.
     """
-    flat = flatten_with_sentinels(messages, parts="sentinel")
+    flat = [_training_template_message(tokenizer, message, reasoning_as_text=True)
+            for message in flatten_with_sentinels(messages, parts="sentinel")]
     index = sum(1 for _ in direct_parts(messages))
     assert index == len(part_lengths), f"{index} parts in the messages, {len(part_lengths)} lengths given"
     if tools and not template_takes_tools(tokenizer):
@@ -72,6 +74,226 @@
     text = tokenizer.apply_chat_template(flat, tools=tools, add_generation_prompt=add_generation_prompt, tokenize=False,
                                          **(chat_template_kwargs or {}))
     return encode_with_part_sentinels(tokenizer, text, list(part_lengths), pad_id)
+
+
+def _training_template_message(tokenizer, message: dict, *, reasoning_as_text: bool = False) -> dict:
+    """Keep separate reasoning and structured calls visible under the selected template."""
+    from ..agent.agent_utils import ModelDialect
+    dialect = ModelDialect.for_tokenizer(tokenizer)
+    message = dict(message)
+    if reasoning_as_text and message.get("role") == "assistant" and isinstance(message.get("content"), str):
+        # Agent records may carry inline reasoning. Escape its delimiters so templates cannot strip earlier thoughts.
+        message["content"] = message["content"].replace("<think>", "[Reasoning]").replace("</think>", "[/Reasoning]")
+    reasoning = message.pop("reasoning", None) or message.get("reasoning_content")
+    if reasoning:
+        if not reasoning_as_text and "reasoning" in (tokenizer.chat_template or ""):
+            message["reasoning_content"] = reasoning
+        else:
+            message.pop("reasoning_content", None)
+            message["content"] = str(reasoning) + "\n\n" + str(message.get("content") or "")
+    if not template_takes_tools(tokenizer) and message.get("tool_calls"):
+        calls = [dict(call.get("function", call), id=call.get("id", f"call_{index}"))
+                 for index, call in enumerate(message["tool_calls"])]
+        message = dialect.rendering.assistant_message(message.get("content") or "", calls)
+    return message
+
+
+def assistant_target_ids(tokenizer, message: dict, tools: list[dict] | None = None,
+                         chat_template_kwargs: dict | None = None) -> list[int]:
+    """Render a complete public assistant output once, excluding the supplied generation prompt."""
+    message = _training_template_message(tokenizer, message)
+    prefix = [{"role": "user", "content": ""}]
+    if tools and not template_takes_tools(tokenizer):
+        prefix, tools = tools_in_system_text(prefix, tools), None
+    kwargs = {"enable_thinking": bool(message.get("reasoning_content"))} if chat_template_kwargs is None else chat_template_kwargs
+    opened = tokenizer.apply_chat_template(prefix, tools=tools, tokenize=False, add_generation_prompt=True, **kwargs)
+    completed = tokenizer.apply_chat_template(prefix + [message], tools=tools, tokenize=False, add_generation_prompt=False, **kwargs)
+    if not completed.startswith(opened):
+        raise ValueError("Assistant output does not extend this tokenizer's generation prompt")
+    output = completed[len(opened):]
+    from ..harness.hf_utils import canonical_eot_token
+    eot = canonical_eot_token(tokenizer) or tokenizer.eos_token
+    if eot and eot in output and not output.rsplit(eot, 1)[1].strip():
+        output = output[:output.rfind(eot) + len(eot)]
+    ids = tokenizer.encode(output, add_special_tokens=False)
+    if not ids:
+        raise ValueError("Assistant output has no target tokens")
+    return list(ids)
+
+
+def prepare_training_history(tokenizer, messages: list[dict], start: int, assistant_ids: list[list[int]],
+                             part_lengths: list[int], *, tools: list[dict] | None = None,
+                             chat_template_kwargs: dict | None = None) -> tuple[list[int], list[tuple[int, int]], list[int]]:
+    """Incremental runtime-shaped tokens, AC spans and predicting positions for one history."""
+    from ..agent.agent_utils import ModelDialect
+    if not isinstance(start, int) or not 0 <= start < len(messages):
+        raise ValueError("Training history start lies outside its messages")
+    selected = [index for index in range(start, len(messages)) if messages[index].get("role") == "assistant"]
+    if len(selected) != len(assistant_ids) or not selected:
+        raise ValueError("Every retained assistant message needs exactly one target-token sequence")
+    vocabulary_size = len(tokenizer)
+    for ids in assistant_ids:
+        if not ids or any(type(token) is not int or not 0 <= token < vocabulary_size for token in ids):
+            raise ValueError("Assistant targets must be nonempty valid tokenizer IDs")
+    if any(direct_parts([message]) for message in messages if message.get("role") == "assistant"):
+        raise ValueError("Assistant outputs cannot contain AC input parts")
+    dialect = ModelDialect.for_tokenizer(tokenizer)
+    chat_template_kwargs = {"enable_thinking": False} if chat_template_kwargs is None else chat_template_kwargs
+    messages = list(messages)
+    if tools and not template_takes_tools(tokenizer):
+        previous_count = len(messages)
+        messages, tools = tools_in_system_text(messages, tools), None
+        start += len(messages) - previous_count
+    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (tokenizer.eos_token_id or 0)
+    cursor = 0
+    def lengths(block):
+        nonlocal cursor
+        count = len(direct_parts(block))
+        values = part_lengths[cursor:cursor + count]
+        cursor += count
+        if len(values) != count:
+            raise ValueError("AC part lengths do not match the message history")
+        return values
+    first = next(index for index in range(start, len(messages)) if messages[index].get("role") == "assistant")
+    if first == 0:
+        raise ValueError("An assistant target needs preceding prompt context")
+    prefix = [_training_template_message(tokenizer, message, reasoning_as_text=True) for message in messages[:first]]
+    ids, spans = dialect.prompt_tokens(tokenizer, prefix, tools, chat_template_kwargs, lengths(messages[:first]), pad)
+    if not ids:
+        raise ValueError("An assistant target needs a nonempty token prefix")
+    positions = []
+    selected_ids = iter(assistant_ids)
+    index = first
+    while index < len(messages):
+        message = messages[index]
+        if message.get("role") != "assistant":
+            raise ValueError("Expected an assistant output after a generation prompt")
+        output = list(next(selected_ids))
+        positions.extend(range(len(ids) - 1, len(ids) + len(output) - 1))
+        ids.extend(output)
+        end = index + 1
+        while end < len(messages) and messages[end].get("role") != "assistant":
+            end += 1
+        if end > index + 1 or end < len(messages):
+            following = messages[index + 1:end]
+            wrapper, local_spans = dialect.continuation_tokens(tokenizer, messages[:index + 1], following,
+                                tools, chat_template_kwargs, lengths(following), pad)
+            joined = dialect.join_continuation(output, wrapper)
+            shift = len(wrapper) - len(joined)
+            spans.extend((len(ids) + a - shift, len(ids) + b - shift) for a, b in local_spans)
+            ids.extend(joined)
+        index = end
+    if cursor != len(part_lengths):
+        raise ValueError("Unused AC part lengths in the message history")
+    return ids, spans, positions
+
+
+def validate_paired_histories(teacher: list[dict], teacher_start: int, student: list[dict], student_start: int) -> None:
+    """Suffix roles/calls agree; tool evidence can differ, assistant outputs cannot."""
+    if not 0 <= teacher_start < len(teacher) or not 0 <= student_start < len(student):
+        raise ValueError("Paired history starts lie outside their messages")
+    left, right = teacher[teacher_start:], student[student_start:]
+    if len(left) != len(right):
+        raise ValueError("Paired histories have different retained message counts")
+    for a, b in zip(left, right):
+        if a.get("role") != b.get("role"):
+            raise ValueError("Paired histories have different retained role order")
+        for key in ("tool_calls", "tool_call_id", "name"):
+            if a.get(key) != b.get(key):
+                raise ValueError("Paired histories have different retained call identity/order")
+        if a.get("role") == "assistant" and any(a.get(key) != b.get(key) for key in ("content", "reasoning", "reasoning_content")):
+            raise ValueError("Paired histories have different assistant outputs")
+
+
+def validate_part_capacity(ac_model, request, capacity: int) -> None:
+    """Check every recursive side forward, including pooled content and summary rows."""
+    children = [ac_model._child_request(part, request.compression_ratio) for part in direct_parts(request.messages)]
+    lengths = [ac_model.part_view_rows(child.messages, child.compression_ratio, child.tools) for child in children]
+    tokenizer = ac_model.side.tokenizer
+    ids, _ = tokenize_with_parts(tokenizer, request.messages, lengths, tokenizer.pad_token_id or 0, tools=request.tools)
+    content_length = len(ids)
+    if ac_model.config.input_pooling_stride:
+        window, stride = ac_model.config.input_pooling_window, ac_model.config.input_pooling_stride
+        content_length = 1 if len(ids) <= window else -(-(len(ids) - window) // stride) + 1
+    total = content_length + ac_model.part_view_rows(request.messages, request.compression_ratio, request.tools)
+    if total > capacity:
+        raise ValueError(f"AC side forward needs {total} positions, exceeding supported capacity {capacity}")
+    for child in children:
+        validate_part_capacity(ac_model, child, capacity)
+
+
+def normalize_public_assistants(tokenizer, messages: list[dict], tools: list[dict] | None,
+                                max_tokens: int) -> tuple[list[dict], list[dict] | None]:
+    """Split only public text/reasoning into complete, budgeted Python continuation turns."""
+    from ..dataset.loaders.trajectory_utils import map_definition, TOOL_MAP
+    if max_tokens < 1:
+        raise ValueError("max_assistant_tokens must be positive")
+    messages, tools = deepcopy(messages), deepcopy(tools or [])
+    used = {call.get("id") for message in messages for call in message.get("tool_calls") or []}
+    output, counter = [], 0
+    for message in messages:
+        if message.get("role") != "assistant":
+            output.append(message)
+            continue
+        while len(assistant_target_ids(tokenizer, message, tools, {"enable_thinking": True})) > max_tokens:
+            field = next((key for key in ("reasoning", "reasoning_content", "content")
+                          if isinstance(message.get(key), str) and message[key]), None)
+            if field is None:
+                raise ValueError("Public assistant has an indivisible call exceeding its token budget")
+            counter += 1
+            while f"ac_continue_{counter}" in used:
+                counter += 1
+            call_id = f"ac_continue_{counter}"
+            used.add(call_id)
+            if not any(tool.get("function", tool).get("name") == "python" for tool in tools):
+                tools.append(map_definition({"name": "python"}, TOOL_MAP))
+            call = {"id": call_id, "type": "function", "function": {
+                "name": "python", "arguments": {"code": f"print('continue {counter}')"}}}
+            def turn(text):
+                return {"role": "assistant", "content": "", field: text, "tool_calls": [call]}
+            text = message[field]
+            low, high, best = 1, len(text), 0
+            while low <= high:
+                middle = (low + high) // 2
+                if len(assistant_target_ids(tokenizer, turn(text[:middle]), tools, {"enable_thinking": True})) <= max_tokens:
+                    best, low = middle, middle + 1
+                else:
+                    high = middle - 1
+            if not best:
+                raise ValueError("Public continuation call cannot fit the assistant token budget")
+            output.extend([turn(text[:best]), {"role": "tool", "name": "python", "tool_call_id": call_id,
+                                               "content": f"continue {counter}\n"}])
+            message[field] = text[best:]
+        output.append(message)
+    return output, tools or None
+
+
+def complete_public_turns(messages: list[dict]) -> list[list[dict]]:
+    """Assistant outputs with their complete replies and intervening input, without dangling calls."""
+    groups = []
+    index = 0
+    while index < len(messages):
+        if messages[index].get("role") != "assistant":
+            raise ValueError("Public continuation must begin with an assistant")
+        end = index + 1
+        while end < len(messages) and messages[end].get("role") != "assistant":
+            end += 1
+        group = messages[index:end]
+        calls = group[0].get("tool_calls") or []
+        replies = [message for message in group[1:] if message.get("role") == "tool"]
+        terminal_answer = (end == len(messages) and len(group) == 1 and len(calls) == 1
+                           and calls[0].get("function", calls[0]).get("name") == "submit_answer")
+        if len(calls) != len(replies) and not terminal_answer:
+            raise ValueError("Public trajectory has unmatched tool calls or replies")
+        for call, reply in zip(calls, replies):
+            if reply.get("tool_call_id") and reply["tool_call_id"] != call.get("id"):
+                raise ValueError("Public tool reply does not match its call")
+        if end == len(messages):
+            while len(group) > 1 and group[-1].get("role") in ("user", "system"):
+                group = group[:-1]
+        groups.append(group)
+        index = end
+    return groups
 
 
 def sinusoidal_positions(length: int, d: int, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
````

</details>

<details class="card" data-tressoir-markdown open>
  <summary><span class="card-title"><a href="codesign/activation/ac_model/ac_model_study.py">activation/ac_model/ac_model_study.py</a></span><span class="card-oneliner">Whole-history public/QA/recorded items, stable root grouping, selection and full source frames.</span><span class="card-badge">Diff</span></summary>

````diff
diff --git a/activation/ac_model/ac_model_study.py b/activation/ac_model/ac_model_study.py
--- a/activation/ac_model/ac_model_study.py
+++ b/activation/ac_model/ac_model_study.py
@@ -1,22 +1,4 @@
-"""
-Training items for the AC model, three kinds (ac_model_training.ActivationContextTrainingItem):
-
-- compaction: a trajectory cut at depth d (1-2) into segments of `threshold` tokens; the in-context
-  prefix is the real messages up to the last cut (an assistant turn may be cut mid-text), the AC
-  prefix nests the segments (`AC_d{AC_{d-1}{...}, segment d}`) in one user message followed by the
-  compaction instructions; the completion is the rest of the cut turn or the next assistant turn.
-  Trains the model to "see" its own compactions.
-- traj_qa: a study question about a trajectory chunk (dataset_study, `traj_qa`); the in-context
-  prefix holds the transcript window around the chunk, the AC prefix the window and 0-2 distractor
-  trajectories (bm25 top documents for the question, the source excluded) as separate parts, each
-  with the task text first so the compressor knows what the reader is after. Trains "seeing" a
-  subagent or parent.
-- rag_qa: a study question about a text chunk; the in-context prefix holds the gold passage, the AC
-  prefix one part with the gold and bm25 distractor passages shuffled to a token budget.
-
-Every item's completion is real text (the trajectory's own continuation, the gold answer): no
-sampling is needed at generation time. Token budgets are counted with the target tokenizer.
-"""
+"""Paired whole-history compaction/QA items and exact recorded-run self-distillation."""
 from __future__ import annotations
 
 import json
@@ -30,9 +12,10 @@
 from ..dataset.dataset import DataModality, DatasetDocument, DatasetDocumentChunk, DatasetQAExample
 from ..dataset.loaders.trajectory_utils import submit_answer_definition
 from ..dataset.dataset_utils import message_text, render_messages
-from .ac_model_training import ActivationContextTrainingItem
-from .ac_model_utils import AC_PART_TYPE, direct_parts, is_ac_part
-from ..common.ac_parts import COMPACTION_INSTRUCTIONS, ac_part   # shared with the agent loop (agent_tools)
+from .ac_model_training import ActivationContextTrainingItem, ActivationContextTrainer, ActivationContextTrainingConfig
+from .ac_model_utils import (AC_PART_TYPE, direct_parts, is_ac_part, assistant_target_ids,
+                             prepare_training_history, normalize_public_assistants, complete_public_turns)
+from ..common.ac_parts import COMPACTION_INSTRUCTIONS, ac_part, tools_in_system_text
 
 if t.TYPE_CHECKING:
     from ..agent.agent_config import AgentRunResult
@@ -101,259 +84,212 @@
 
     # ------------------------------------------------------------------------------------------ compaction
     def generate_compaction_samples(
-        self,
-        dataset_id: str,
-        num_samples: int,
-        seed: int = 0,
+        self, dataset_id: str, num_samples: int, seed: int = 0,
         depth_range: tuple[int, int] = (1, 2),
         threshold_range_tokens: tuple[int, int] = (8192, 32768),
         ratios_range: tuple[float, float] = (1.0 / 8.0, 1.0 / 16.0),
         max_post_compaction_tokens: int = 8192,
-        immediate_continuation_ratio: float = 0.25, # Still bias towards many trajectories being immediate continuations, as that's most important.
-        completion_max_tokens: int = 512,
         max_total_tokens: int = 72_000,
         min_trajectory_tokens: int = MIN_COMPACTION_TOKENS,
+        *, max_assistant_tokens: int | None = None,
     ) -> list[ActivationContextTrainingItem]:
-        """
-        One item per sampled trajectory document (documents are drawn with replacement once every
-        document was used; a document shorter than `min_trajectory_tokens` before its completion is
-        skipped). Segment thresholds are scaled down to fit the trajectory and `max_total_tokens`.
-        """
-        if max_post_compaction_tokens < 0 or not 0 <= immediate_continuation_ratio <= 1 or completion_max_tokens < 1:
-            raise ValueError("Invalid continuation budget or immediate mixture")
+        """Complete public continuations; every assistant output is a target, within both exact budgets."""
+        if max_post_compaction_tokens < 1 or max_total_tokens < 1 or min(depth_range) < 1:
+            raise ValueError("Invalid compaction depth or token budgets")
+        if max_assistant_tokens is None:
+            from ..harness.vllm_wrapper import VLLMWrapper
+            defaults = VLLMWrapper.recommended_chat_kwargs(self.ac_model.target.model_config.model_id)
+            max_assistant_tokens = (defaults.get("sampling_params") or {}).get("max_tokens", 1024)
+        if max_assistant_tokens < 1:
+            raise ValueError("max_assistant_tokens must be positive")
         rng = random.Random(seed)
         documents = self._documents(dataset_id, DataModality.TRAJECTORY)
         assert documents, f"{dataset_id} holds no trajectory documents"
-        order = list(documents)
-        rng.shuffle(order)
-        items: list[ActivationContextTrainingItem] = []
-        attempts = 0
-        requested_immediate = rng.random() < immediate_continuation_ratio
-        while len(items) < num_samples and attempts < 4 * num_samples + len(order):
-            document = order[attempts % len(order)]
-            attempts += 1
-            item = self._compaction_item(document, dataset_id, rng, len(items), seed, depth_range, threshold_range_tokens,
-                                         ratios_range, completion_max_tokens, max_total_tokens, min_trajectory_tokens,
-                                         max_post_compaction_tokens, requested_immediate)
+        rng.shuffle(documents)
+        items = []
+        for attempt in range(4 * num_samples + len(documents)):
+            if len(items) >= num_samples:
+                break
+            item = self._compaction_item(documents[attempt % len(documents)], dataset_id, rng, len(items), seed,
+                        depth_range, threshold_range_tokens, ratios_range, max_total_tokens, min_trajectory_tokens,
+                        max_post_compaction_tokens, max_assistant_tokens)
             if item is not None:
                 items.append(item)
-                requested_immediate = rng.random() < immediate_continuation_ratio
         if len(items) < num_samples:
-            print(f"AC study - produced {len(items)}/{num_samples} compaction items; continuation candidates exhausted")
+            print(f"AC study - produced {len(items)}/{num_samples} compaction items; complete candidates exhausted")
         return items
 
-    def _compaction_item(self, document: DatasetDocument, dataset_id: str, rng: random.Random, index: int, seed: int,
-                         depth_range: tuple[int, int], threshold_range_tokens: tuple[int, int], ratios_range: tuple[float, float],
-                         completion_max_tokens: int, max_total_tokens: int, min_trajectory_tokens: int,
-                         max_post_compaction_tokens: int = 8192, immediate: bool = True) -> ActivationContextTrainingItem | None:
-        messages = document.trajectory or []
-        if len(messages) < 3 or messages[0]["role"] != "user":
+    def _compaction_item(self, document, dataset_id, rng, index, seed, depth_range, threshold_range_tokens,
+                         ratios_range, max_total_tokens, min_trajectory_tokens, max_post_compaction_tokens,
+                         max_assistant_tokens) -> ActivationContextTrainingItem | None:
+        source = document.trajectory or []
+        if len(source) < 2 or source[0].get("role") != "user":
             return None
-        tools = (document.trajectory_kwargs or {}).get("tools") or None
+        try:
+            messages, tools = normalize_public_assistants(self.tokenizer, source,
+                        (document.trajectory_kwargs or {}).get("tools"), max_assistant_tokens)
+            first = next((index for index, message in enumerate(messages) if message.get("role") == "assistant"), len(messages))
+            initial = messages[:first]
+            groups = complete_public_turns(messages[first:])
+        except ValueError:                    # an indivisible public call/reply cannot be a complete candidate
+            return None
+        if len(groups) < 2:
+            return None
         system = self._system_messages(document)
-        task = messages[0]
-        body = messages[1:]
-        tokens = [self.message_tokens(message) for message in body]
-        # The completion must be an assistant turn after the last cut: the body up to the last assistant turn is cuttable.
-        last_assistant = max((i for i, message in enumerate(body) if message["role"] == "assistant"), default=-1)
-        if last_assistant < 1:
+        depth = min(rng.randint(min(depth_range), max(depth_range)), len(groups) - 1)
+        if depth < min(depth_range):
             return None
-        available = sum(tokens[:last_assistant + 1])                            # the last assistant turn may be cut mid-text
-        if available < min_trajectory_tokens:
+        costs = [sum(self.message_tokens(message) for message in group) for group in groups]
+        budget = min(sum(costs[:-1]), max_total_tokens - max_post_compaction_tokens)
+        threshold = min(rng.randint(min(threshold_range_tokens), max(threshold_range_tokens)), max(1, budget // depth))
+        cuts, offset = [], 0
+        for level in range(depth):
+            total, end = 0, offset
+            limit = len(groups) - (depth - level)
+            while end < limit and (total < threshold or end == offset):
+                total += costs[end]
+                end += 1
+            cuts.append(end)
+            offset = end
+        before = [message for group in groups[:cuts[-1]] for message in group]
+        prefix = system + initial + before
+        chat_kwargs = {"enable_thinking": True}
+        continuation, targets, exact_tokens, post_tokens = [], [], 0, 0
+        for group in groups[cuts[-1]:]:
+            proposed = continuation + group
+            proposed_targets = targets + [assistant_target_ids(self.tokenizer, group[0], tools, chat_kwargs)]
+            ids, _, positions = prepare_training_history(self.tokenizer, prefix + proposed, len(prefix), proposed_targets, [], tools=tools,
+                                                        chat_template_kwargs=chat_kwargs)
+            if positions[0] + 1 < min_trajectory_tokens:
+                return None
+            cost = len(ids) - (positions[0] + 1)
+            if cost > max_post_compaction_tokens or len(ids) > max_total_tokens:
+                break
+            continuation, targets, exact_tokens, post_tokens = proposed, proposed_targets, len(ids), cost
+        if not targets:
             return None
-        depth = rng.randint(min(depth_range), max(depth_range))
-        thresholds = [rng.randint(min(threshold_range_tokens), max(threshold_range_tokens)) for _ in range(depth)]
-        budget = min(max_total_tokens - completion_max_tokens - self.message_tokens(task) - 256, int(available * 0.9))
-        if budget < 256:
-            return None
-        if sum(thresholds) > budget:
-            scale = budget / sum(thresholds)
-            thresholds = [max(256, int(threshold * scale)) for threshold in thresholds]
-        # Cuts: (message index, character offset in that assistant text or None for a boundary).
-        cuts: list[tuple[int, int | None]] = []
-        cumulative = 0
-        target_total = 0
-        position = 0
-        for threshold in thresholds:
-            target_total += threshold
-            while position < len(body) and cumulative + tokens[position] <= target_total:
-                cumulative += tokens[position]
-                position += 1
-            if position >= len(body):
-                return None
-            message = body[position]
-            text = message_text(message)
-            if message["role"] == "assistant" and not message.get("tool_calls") and len(text) > 200 and cumulative < target_total:
-                fraction = (target_total - cumulative) / max(1, tokens[position])
-                full_ids = self.tokenizer.encode(text, add_special_tokens=False)
-                cut_tokens = max(1, min(len(full_ids) - 1, int(len(full_ids) * min(0.9, max(0.1, fraction)))))
-                partial, remainder, cut_tokens = self._token_boundary(text, full_ids, cut_tokens)
-                offset = len(partial)
-                cuts.append((position, offset))
-                break                                                              # a mid-turn cut ends the chain (the rest of the turn is the completion)
-            cuts.append((position, None))
-        if not cuts:
-            return None
-        # Segments between cuts, the completion after the last one.
-        segments: list[list[dict]] = []
-        start = 0
-        partial_text = ""
-        for cut_index, (position, offset) in enumerate(cuts):
-            if offset is None:
-                segments.append(body[start:position])
-                start = position
-            else:
-                message = body[position]
-                text = message_text(message)
-                segments.append(body[start:position] + [{"role": "assistant", "content": text[:offset]}])
-                partial_text = text[:offset]
-                start = position
-        segments = [segment for segment in segments if segment]
-        if not segments:
-            return None
-        last_position, last_offset = cuts[-1]
-        visible: list[dict] = []
-        teacher_partial_ids: list[int] | None = None
-        if last_offset is not None:
-            turn_text = message_text(body[last_position])
-            full_ids = self.tokenizer.encode(turn_text, add_special_tokens=False)
-            # The cut was selected at an exact target-token boundary above.
-            teacher_partial_ids = full_ids[:cut_tokens]
-            completion_ids = full_ids[cut_tokens:]
-            immediate_target = last_position
-            visible_after_cut = [{**body[last_position], "content": turn_text[last_offset:]}]
-        else:
-            immediate_target = next((i for i in range(last_position, len(body)) if body[i]["role"] == "assistant"), None)
-            if immediate_target is None:
-                return None
-            visible_after_cut = []
-        if immediate:
-            target_position = immediate_target
-            if last_offset is None:
-                visible = body[last_position:target_position]
-                completion_ids = self.tokenizer.encode(self.assistant_text(body[target_position]), add_special_tokens=False)
-            in_context_tail = body[:target_position]
-        else:
-            candidates = []
-            for target_position in range(immediate_target + 1, len(body)):
-                if body[target_position]["role"] != "assistant":
-                    continue
-                suffix = (visible_after_cut + body[last_position + 1:target_position] if last_offset is not None
-                          else body[last_position:target_position])
-                suffix_tokens = sum(self.message_tokens(message) for message in suffix)
-                teacher_tokens = sum(tokens[:target_position]) + self.message_tokens(task)
-                if suffix_tokens <= max_post_compaction_tokens and teacher_tokens + completion_max_tokens + 256 <= max_total_tokens:
-                    candidates.append((target_position, suffix))
-            if not candidates:
-                return None
-            target_position, visible = rng.choice(candidates)
-            in_context_tail = body[:target_position]
-            partial_text, teacher_partial_ids = "", None
-            completion_ids = self.tokenizer.encode(self.assistant_text(body[target_position]), add_special_tokens=False)
-        if not completion_ids:
-            return None
-        complete = len(completion_ids) <= completion_max_tokens
-        completion_ids = completion_ids[:completion_max_tokens]
-        # Avoid a cap ending inside a multibyte character; preserve original IDs.
-        while completion_ids and self.tokenizer.decode(completion_ids, clean_up_tokenization_spaces=False).endswith("\ufffd"):
-            completion_ids = completion_ids[:-1]
-        completion_text = self.tokenizer.decode(completion_ids, clean_up_tokenization_spaces=False)
-        if not completion_text.strip():
-            return None
-        visible_tokens = self._visible_tokens(visible, tools)
-        if visible_tokens > max_post_compaction_tokens:
-            return None
-        if sum(tokens[:target_position]) + len(teacher_partial_ids or []) + len(completion_ids) + self.message_tokens(task) + 256 > max_total_tokens:
-            return None
-        ratio = self._sample_ratio(rng, ratios_range)
-        nested: dict | None = None
-        for segment in segments:
-            inner = [{"role": "user", "content": [nested]}] if nested is not None else [task]   # the innermost segment carries the task: what the reader is after
-            nested = ac_part(deepcopy(system) + inner + segment, self.ac_name, ratio)           # a segment part carries its system message first
+        ratio, nested, offset = self._sample_ratio(rng, ratios_range), None, 0
+        for end in cuts:
+            frame = [{"role": "user", "content": [nested]}] if nested is not None else initial
+            segment = [message for group in groups[offset:end] for message in group]
+            nested = ac_part(deepcopy(system + frame + segment), self.ac_name, ratio, kind="compaction")
             if tools:
                 nested["tools"] = deepcopy(tools)
-        ac_user = {"role": "user", "content": [nested, {"type": "text", "text": "\n\n" + COMPACTION_INSTRUCTIONS}]}
+            offset = end
+        student_prefix = system + initial + [{"role": "user", "content": [nested,
+                            {"type": "text", "text": "\n\n" + COMPACTION_INSTRUCTIONS}]}]
         item = ActivationContextTrainingItem(
-            item_id=f"compaction:{dataset_id}:{seed}:{index}",
-            kind="compaction",
-            in_context_prefix=system + [task] + in_context_tail,
-            ac_prefix=system + [task, ac_user] + deepcopy(visible),
-            completion_text=completion_text,
-            completion_complete=complete,
-            teacher_partial_text=partial_text,
-            teacher_partial_token_ids=teacher_partial_ids,
-            completion_token_ids=completion_ids,
-            tools=tools,
-            dataset_id=dataset_id,
-            doc_ids=[document.doc_id],
-            info={"depth": len(segments), "thresholds": thresholds, "ratio": ratio, "in_context_tokens": sum(tokens[:len(in_context_tail)]),
-                  "immediate": immediate, "visible_suffix_tokens": visible_tokens,
-                  "delay_turns": target_position - immediate_target,
-                  "target_position": target_position, "mid_turn_cut": last_offset is not None,   # the completion's body message; run harvesting maps it back
-                  "observed_terminal_answer": (document.trajectory_kwargs or {}).get("observed_terminal_answer")},
-        )
-        prefix = self.tokenizer.apply_chat_template(item.in_context_prefix, tools=tools, add_generation_prompt=True, tokenize=True)
-        prefix = prefix["input_ids"] if hasattr(prefix, "keys") else prefix
-        eot = self.ac_model.target.model_config.model_description.eot_token
-        end_tokens = len(self.tokenizer.encode(eot, add_special_tokens=False)) if complete and eot else 0
-        exact_tokens = len(prefix) + len(teacher_partial_ids or []) + len(completion_ids) + end_tokens
-        if exact_tokens > max_total_tokens:
-            return None
-        item.info["in_context_tokens"] = exact_tokens
+            item_id=f"compaction:{dataset_id}:{seed}:{index}", kind="compaction",
+            in_context_messages=prefix + continuation, ac_messages=deepcopy(student_prefix + continuation),
+            in_context_start=len(prefix), ac_start=len(student_prefix), assistant_token_ids=targets,
+            tools=tools, dataset_id=dataset_id, doc_ids=[document.doc_id],
+            info={"depth": depth, "thresholds": [threshold] * depth, "ratio": ratio,
+                  "chat_template_kwargs": chat_kwargs,
+                  "in_context_tokens": exact_tokens, "post_compaction_tokens": post_tokens,
+                  "max_assistant_tokens": max_assistant_tokens, "assistant_turns": len(targets),
+                  "observed_terminal_answer": (document.trajectory_kwargs or {}).get("observed_terminal_answer")})
+        # Student wrappers and row counts also count against the overall guard.
+        try:
+            ActivationContextTrainer(self.harness, ActivationContextTrainingConfig(max_example_tokens=max_total_tokens)).build_example(self.ac_model, item)
+        except ValueError as error:
+            if "max_example_tokens" in str(error):
+                return None
+            raise
         return item
-
-    def _visible_tokens(self, visible: list[dict], tools: list[dict] | None) -> int:
-        if not visible:
-            return 0
-        anchor = [{"role": "user", "content": ""}]
-        def count(messages: list[dict]) -> int:
-            ids = self.tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=True, tokenize=True)
-            return len(ids["input_ids"] if hasattr(ids, "keys") else ids)
-        return max(0, count(anchor + visible) - count(anchor))
 
     # ------------------------------------------------------------------------------------------ run-result harvesting
     def items_from_run_results(
-        self,
-        runs: list["AgentRunResult"],
-        *,
+        self, runs: list["AgentRunResult"], *, loss_kind: t.Literal["kl", "sft"] = "kl",
+        selection: t.Literal["score_1", "score_1_or_unscored", "all"] | None = None,
         weight_of: t.Callable[["AgentRunResult"], float] | None = None,
-        seed: int = 0,
-        completion_max_tokens: int | None = None,
-        text_run_kwargs: dict | None = None,
-        items_per_text_run: int = 1,
     ) -> list[ActivationContextTrainingItem]:
-        """
-        Self-distillation items from agent run results: the target reading a run as plain text (teacher)
-        against the same target reading the run's real activation-context prompt (student), on the turn
-        the agent actually sampled. Every run recorded with this AC model yields one item per assistant
-        turn whose prompt holds a part (compaction trees, subagent prompts and returns, tool-output and
-        search parts); the run's compacted segments and every subagent beneath it are visited. A run
-        without an AC model is converted to a trajectory document and cut by the 3a1 generator
-        (`text_run_kwargs` override its defaults), with the recorded sampled tokens as the completion
-        where the cut lands on a whole turn. Weights multiply a KL and must be finite and nonnegative;
-        `weight_of(run)` defaults to the run's score, zero-weight runs are skipped. Records are not
-        mutated; `harvest_report` holds candidate/accepted/skip counts.
-        """
-        rng = random.Random(seed)
-        report = self.harvest_report = {"roots": len(runs), "runs": 0, "candidates": 0, "accepted": 0, "skipped": {}}
-        items: list[ActivationContextTrainingItem] = []
+        """One whole-history item per nonempty segment, preserving every selected recorded output."""
+        from ..agent_training.agent_training_utils import activation_messages_of
+        from ..agent.rollout_caching import config_key
+        if loss_kind not in ("kl", "sft"):
+            raise ValueError(loss_kind)
+        selection = selection or ("all" if loss_kind == "kl" else "score_1")
+        if selection not in ("all", "score_1", "score_1_or_unscored"):
+            raise ValueError(selection)
+        report = self.harvest_report = {"roots": len(runs), "runs": 0, "candidates": 0, "accepted": 0,
+                                      "loss_kind": loss_kind, "selection": selection, "skipped": {}}
+        items = []
         for root in runs:
-            weight = float(root.score if weight_of is None else weight_of(root))
+            if selection != "all" and root.score != 1 and not (selection == "score_1_or_unscored" and root.score is None):
+                self._harvest_skip("selection")
+                continue
+            weight = 1.0 if weight_of is None else float(weight_of(root))
             if not math.isfinite(weight) or weight < 0:
-                raise ValueError("AC distillation item weights must be finite and nonnegative: the KL target has no sign to flip "
-                                 "(negative advantages belong to the policy-gradient path)")
-            if weight == 0.0:
+                raise ValueError("AC item weights must be finite and nonnegative")
+            if weight == 0:
                 self._harvest_skip("zero_weight")
                 continue
-            for source_key, run in self._rollout_sources(root):
+            root_key = f"{config_key(root.agent_config)}:{root.seed}"
+            for source_key, run in self._rollout_sources(root, root_key):
                 report["runs"] += 1
-                if run.ac_model_name is None:
-                    for index in range(items_per_text_run):
-                        items += self._items_from_text_run(run, f"{source_key}.{index}", weight, rng, completion_max_tokens, text_run_kwargs)
-                elif run.ac_model_name == self.ac_name:
-                    items += self._items_from_ac_run(run, source_key, weight, completion_max_tokens)
-                else:
-                    raise ValueError(f"run recorded with AC model {run.ac_model_name!r}; this generator is bound to {self.ac_name!r}")
+                source_model = self.harness.loaded_models.get(run.agent_config.model_name)
+                if source_model is None:
+                    raise ValueError(f"Unknown recorded tokenizer source {run.agent_config.model_name!r}")
+                source_tokenizer = source_model.tokenizer
+                if (source_tokenizer.get_vocab() != self.tokenizer.get_vocab()
+                        or source_tokenizer.special_tokens_map != self.tokenizer.special_tokens_map
+                        or source_tokenizer.chat_template != self.tokenizer.chat_template):
+                    raise ValueError("Recorded assistant tokens require a compatible tokenizer and chat template")
+                tools = self._run_tools(run)
+                for segment_index, segment in enumerate([*run.compactions, run]):
+                    report["candidates"] += 1
+                    targets = []
+                    for step in segment.trajectory:
+                        assistants = [message for message in step.get("messages") or [] if message.get("role") == "assistant"]
+                        if step.get("role") == "assistant":
+                            ids = list(step.get("token_ids") or [])
+                            if len(assistants) != 1 or not ids:
+                                raise ValueError("Recorded assistant step needs one message and its nonempty sampled token IDs; regenerate this run")
+                            targets.append(ids)
+                        elif assistants:
+                            raise ValueError("Recorded input step contains an assistant output")
+                    if not targets:
+                        self._harvest_skip("empty_segment")
+                        continue
+                    config = replace(segment.agent_config, ac_model_name=self.ac_name)
+                    student = activation_messages_of(segment, config)
+                    start = len(segment.prompt_messages)
+                    if segment.ac_model_name:
+                        recorded = [(student[:start], segment.prompt_ac_spans)]
+                        cursor = start
+                        for step in segment.trajectory:
+                            end = cursor + len(step.get("messages") or [])
+                            recorded.append((student[cursor:end], step.get("ac_spans") or []))
+                            cursor = end
+                        if any(len(direct_parts(messages)) != len(spans) for messages, spans in recorded):
+                            raise ValueError("Recorded AC spans need matching raw activation parts; regenerate this run")
+                    teacher_prefix = self._expand_parts(student[:start]) if loss_kind == "kl" else None
+                    if not segment.ac_model_name:
+                        # Text runs also compress their complete initial frame, retaining raw injected parts within it.
+                        frame, suffix = student[:start], student[start:]
+                        system = [message for message in frame if message.get("role") == "system"]
+                        part = ac_part(deepcopy(frame), self.ac_name, self.ac_model.config.default_compression_ratio)
+                        if tools:
+                            part["tools"] = deepcopy(tools)
+                        student = system + [{"role": "user", "content": [part]}] + suffix
+                        start = len(system) + 1
+                    teacher = None
+                    teacher_start = 0
+                    if loss_kind == "kl":
+                        teacher = teacher_prefix + self._expand_parts(student[start:])
+                        teacher_start = len(teacher_prefix)
+                    item = ActivationContextTrainingItem(
+                        item_id=f"rollout:{source_key}:{segment_index}", kind="compaction",
+                        in_context_messages=teacher, ac_messages=deepcopy(student),
+                        in_context_start=teacher_start, ac_start=start, assistant_token_ids=targets,
+                        tools=tools, weight=weight, dataset_id="agent_runs", doc_ids=[root_key],
+                        info={"source": "agent_run", "source_key": source_key, "root_key": root_key,
+                              "chat_template_kwargs": source_model.chat_template_kwargs(segment.agent_config.call_kwargs),
+                              "segment": segment_index, "score": root.score, "completion_source": "recorded",
+                              "channels": sorted({part.get("kind") or "unknown" for part in direct_parts(student)}),
+                              "ac_model_name": segment.ac_model_name, "ac_model_version": segment.ac_model_version})
+                    ActivationContextTrainer(self.harness, ActivationContextTrainingConfig(loss_kind=loss_kind)).build_example(self.ac_model, item)
+                    items.append(item)
         report["accepted"] = len(items)
         return items
 
@@ -361,70 +297,18 @@
         skipped = self.harvest_report["skipped"]
         skipped[reason] = skipped.get(reason, 0) + 1
 
-    def _rollout_sources(self, root: "AgentRunResult", key: str | None = None) -> t.Iterator[tuple[str, "AgentRunResult"]]:
-        """The run, then every subagent attached to any of its segments (recursively), each once with a path key."""
-        key = f"{root.agent_config.agent_name}:{root.seed}" if key is None else key
+    def _rollout_sources(self, root: "AgentRunResult", key: str) -> t.Iterator[tuple[str, "AgentRunResult"]]:
         yield key, root
-        for segment_index, segment in enumerate(list(root.compactions) + [root]):
+        for segment_index, segment in enumerate([*root.compactions, root]):
             for child_index, child in enumerate(segment.subagent_results):
                 yield from self._rollout_sources(child, f"{key}/s{segment_index}c{child_index}")
 
     def _run_tools(self, run: "AgentRunResult") -> list[dict] | None:
-        """The tool definitions the run's prompts were templated with (None under the inline rendering, whose listing is in the system prompt)."""
         from ..agent.agent import Agent
         agent = Agent(self.harness, run.agent_config)
         agent.dialect = self.dialect
         agent._prepare_tools()
         return agent.template_tools
-
-    def _rollout_completion(self, recorded_ids: list[int], max_tokens: int | None = None) -> tuple[list[int], bool]:
-        """
-        Recorded sampled ids as completion fields: exactly one terminal end-of-turn sequence is stripped
-        (the trainer appends it again when `completion_complete`), the rest is kept verbatim; a cap marks
-        the completion incomplete. No cap by default: a run's turn is bounded by its own sampling max_tokens.
-        """
-        eot = self.ac_model.target.model_config.model_description.eot_token or ""
-        eot_ids = self.tokenizer.encode(eot, add_special_tokens=False) if eot else []
-        ids = list(recorded_ids)
-        complete = bool(eot_ids) and len(ids) >= len(eot_ids) and ids[-len(eot_ids):] == eot_ids
-        if complete:
-            ids = ids[:-len(eot_ids)]
-        if max_tokens is not None and len(ids) > max_tokens:
-            return ids[:max_tokens], False
-        return ids, complete
-
-    def _items_from_ac_run(self, run: "AgentRunResult", source_key: str, weight: float, completion_max_tokens: int | None) -> list[ActivationContextTrainingItem]:
-        """One item per assistant turn whose student prefix (its segment's real prompt plus the steps before it) holds a part."""
-        tools = self._run_tools(run)
-        items: list[ActivationContextTrainingItem] = []
-        for k, segment in enumerate(list(run.compactions) + [run]):
-            system = [message for message in segment.prompt_messages if message.get("role") == "system"]
-            base = [message for message in segment.prompt_messages if message.get("role") != "system"]
-            history: list[dict] = []
-            for t_index, step in enumerate(segment.trajectory):
-                if step.get("role") == "assistant":
-                    student = base + history
-                    self.harvest_report["candidates"] += 1
-                    parts = direct_parts(student)
-                    if not parts:
-                        self._harvest_skip("no_parts")
-                    else:
-                        ids, complete = self._rollout_completion(step.get("token_ids") or [], completion_max_tokens)
-                        if not ids:
-                            self._harvest_skip("empty_completion")
-                        else:
-                            items.append(ActivationContextTrainingItem(
-                                item_id=f"rollout:{source_key}:{k}:{t_index}", kind="compaction",
-                                in_context_prefix=system + self._expand_parts(student), ac_prefix=deepcopy(system + student),
-                                completion_text=self.tokenizer.decode(ids, clean_up_tokenization_spaces=False),
-                                completion_token_ids=ids, completion_complete=complete, tools=tools, weight=weight,
-                                dataset_id="agent_runs", doc_ids=[source_key],
-                                info={"source": "agent_run", "source_key": source_key, "segment": k, "turn": t_index, "score": run.score,
-                                      "channels": sorted({part.get("kind") or "unknown" for part in parts}), "completion_source": "recorded",
-                                      "ac_model_name": run.ac_model_name, "ac_model_version": run.ac_model_version},
-                            ))
-                history += deepcopy(step.get("messages") or [])
-        return items
 
     def _expand_parts(self, messages: list[dict]) -> list[dict]:
         """
@@ -442,6 +326,15 @@
             tree = next((part for part in content if is_ac_part(part) and part.get("kind") == "compaction"), None)
             if tree is not None:
                 expanded = self._expand_tree(tree)
+                # A nested source frame must remain visible even when the reader has another system prompt.
+                framed = []
+                for entry in expanded:
+                    if entry.get("role") == "system" and out:
+                        if entry in out:
+                            continue
+                        entry = {"role": "user", "content": "Source system frame:\n" + message_text(entry)}
+                    framed.append(entry)
+                expanded = framed
                 if out and expanded and out[-1] == expanded[0]:
                     out.pop()                                                       # the tree starts with the task the prompt already shows
                 out.extend(expanded)
@@ -458,9 +351,9 @@
                     elif kind == "search":
                         pieces.append("\n\n" + next((m.get("content", "") for m in inner if m.get("role") == "tool"), ""))
                     elif kind == "subagent_return":
-                        pieces.append("Subagent transcript:\n" + render_messages(self._expand_parts(inner)) + "\n\n")
+                        pieces.append("Subagent transcript:\n" + render_messages(self._expand_parts(tools_in_system_text(inner, part.get("tools")))) + "\n\n")
                     else:                                                           # subagent_prompt, parent_context, untagged: a transcript
-                        pieces.append("Parent transcript:\n" + render_messages(self._expand_parts(inner)) + "\n\n")
+                        pieces.append("Parent transcript:\n" + render_messages(self._expand_parts(tools_in_system_text(inner, part.get("tools")))) + "\n\n")
                 elif isinstance(part, dict):
                     if drop_next_text and part.get("type") == "text":
                         drop_next_text = False
@@ -473,75 +366,11 @@
 
     def _expand_tree(self, tree: dict) -> list[dict]:
         """The history a compaction tree holds, as text: its first user message (recursively) then the segment through the compact call."""
-        inner = [message for message in tree.get("messages") or [] if message.get("role") != "system"]   # the frame is the reader's own
+        inner = tools_in_system_text(tree.get("messages") or [], tree.get("tools"))
         if not inner:
             return []
         expanded = self._expand_parts([inner[0]]) + self._expand_parts(inner[1:])
-        last = expanded[-1] if expanded else None
-        if last is not None and last.get("role") == "assistant" and last.get("tool_calls"):
-            expanded.append({"role": "tool", "content": "Context compacted."})        # the compact call's result, so the transcript stays well formed
         return expanded
-
-    def _rollout_document(self, run: "AgentRunResult", key: str, tools: list[dict] | None) -> tuple[DatasetDocument | None, list[tuple[int, int] | None]]:
-        """A text-only run as a trajectory document (task first, every segment's steps as text, compaction messages dropped) with each body message's (segment, step) origin."""
-        segments = list(run.compactions) + [run]
-        first = segments[0].prompt_messages
-        task = next((message for message in first if message.get("role") == "user"), None)
-        if task is None:
-            return None, []
-        system = next((message.get("content") for message in first if message.get("role") == "system"), None)
-        trajectory = self._expand_parts([task])
-        origins: list[tuple[int, int] | None] = []
-        for k, segment in enumerate(segments):
-            for t_index, step in enumerate(segment.trajectory):
-                expanded = self._expand_parts(step.get("messages") or [])
-                trajectory += expanded
-                origins += [(k, t_index)] * len(expanded)
-            if k < len(segments) - 1 and trajectory and trajectory[-1].get("role") == "assistant" and trajectory[-1].get("tool_calls"):
-                trajectory.append({"role": "tool", "content": "Context compacted."})
-                origins.append(None)
-        document = DatasetDocument(doc_id=key, dataset_id="agent_runs", trajectory=trajectory,
-                                   trajectory_kwargs={"tools": tools, "system_prompt": system, "answer": run.answer}, modality=DataModality.TRAJECTORY)
-        return document, origins
-
-    def _items_from_text_run(self, run: "AgentRunResult", source_key: str, weight: float, rng: random.Random,
-                             completion_max_tokens: int | None, text_run_kwargs: dict | None) -> list[ActivationContextTrainingItem]:
-        """A text-only run cut by the 3a1 generator; a whole-turn completion takes the recorded sampled tokens, a mid-turn cut keeps the re-encoded rest of the turn."""
-        tools = self._run_tools(run)
-        document, origins = self._rollout_document(run, source_key, tools)
-        self.harvest_report["candidates"] += 1
-        if document is None:
-            self._harvest_skip("no_task")
-            return []
-        kwargs = dict(depth_range=(1, 2), threshold_range_tokens=(8192, 32768), ratios_range=(1.0 / 8.0, 1.0 / 16.0), completion_max_tokens=512,
-                      max_total_tokens=72_000, min_trajectory_tokens=MIN_COMPACTION_TOKENS, max_post_compaction_tokens=8192,
-                      immediate_continuation_ratio=0.25) | dict(text_run_kwargs or {})
-        immediate = rng.random() < kwargs.pop("immediate_continuation_ratio")
-        item = self._compaction_item(document, "agent_runs", rng, 0, 0, kwargs["depth_range"], kwargs["threshold_range_tokens"],
-                                     kwargs["ratios_range"], kwargs["completion_max_tokens"], kwargs["max_total_tokens"],
-                                     kwargs["min_trajectory_tokens"], kwargs["max_post_compaction_tokens"], immediate)
-        if item is None:
-            self._harvest_skip("no_cut")
-            return []
-        item.item_id = f"rollout_text:{source_key}"
-        item.weight = weight
-        item.doc_ids = [source_key]
-        item.info.update({"source": "agent_run_text", "source_key": source_key, "score": run.score})
-        position = item.info.get("target_position")
-        origin = origins[position] if position is not None and position < len(origins) else None
-        if item.info.get("mid_turn_cut") or origin is None:
-            item.info["completion_source"] = "reencoded" if item.info.get("mid_turn_cut") else "reencoded_unmapped"
-            return [item]
-        segment = (list(run.compactions) + [run])[origin[0]]
-        step = segment.trajectory[origin[1]]
-        ids, complete = self._rollout_completion(step.get("token_ids") or [], completion_max_tokens)
-        if not ids:
-            self._harvest_skip("empty_completion")
-            return []
-        item.completion_token_ids, item.completion_complete = ids, complete
-        item.completion_text = self.tokenizer.decode(ids, clean_up_tokenization_spaces=False)
-        item.info["completion_source"] = "recorded"
-        return [item]
 
     # ------------------------------------------------------------------------------------------ trajectory QA
     def _study_examples(self, dataset_id: str, num_samples: int, seed: int, modality: DataModality, caching_id: str | None) -> list[DatasetQAExample]:
@@ -592,6 +421,7 @@
             if start != 0 or end != len(text):
                 message["content"] = text[start:end]
                 message.pop("reasoning", None)
+                message.pop("reasoning_content", None)
             if not calls:
                 message.pop("tool_calls", None)
             result.append(message)
@@ -648,6 +478,8 @@
                 kwargs = source.trajectory_kwargs or {}                              # the frame the trajectory ran with, when the loader kept it
                 frame = [{"role": "system", "content": kwargs["system_prompt"]}] if kwargs.get("system_prompt") else []
                 part = ac_part(frame + messages, self.ac_name, ratio)
+                if kwargs.get("tools"):
+                    part["tools"] = deepcopy(kwargs["tools"])
                 if depth >= 2:
                     part = ac_part(frame + [messages[0], {"role": "user", "content": [part]}], self.ac_name, ratio)
                 if kwargs.get("tools"):
@@ -661,16 +493,18 @@
             for number, (_, part) in enumerate(parts, start=1):
                 content.extend([{"type": "text", "text": f"Trajectory {number}:\n"}, part, {"type": "text", "text": "\n\n"}])
             content.append({"type": "text", "text": f"{TRAJ_QA_INSTRUCTIONS}\nQuestion: {question}"})
-            in_context_user = {"role": "user", "content": f"Trajectory:\n{render_messages(window)}\n\n{TRAJ_QA_INSTRUCTIONS}\nQuestion: {question}"}
+            source_frame = tools_in_system_text(self._system_messages(document) + window, (document.trajectory_kwargs or {}).get("tools"))
+            in_context_user = {"role": "user", "content": f"Trajectory:\n{render_messages(source_frame)}\n\n{TRAJ_QA_INSTRUCTIONS}\nQuestion: {question}"}
             system = [{"role": "system", "content": TRAJ_QA_SYSTEM_PROMPT}]
+            output = self.dialect.rendering.assistant_message("", [{"id": "answer", "name": "submit_answer", "arguments": {"answer": example.gold_answers[0].strip()}}])
             items.append(ActivationContextTrainingItem(
                 item_id=f"traj_qa:{dataset_id}:{seed}:{item_index}",
                 kind="traj_qa",
-                in_context_prefix=system + [in_context_user],
-                ac_prefix=system + [{"role": "user", "content": content}],
-                completion_text=self.dialect.preferred_format.render("submit_answer", {"answer": example.gold_answers[0].strip()}),
+                in_context_messages=system + [in_context_user, output],
+                ac_messages=system + [{"role": "user", "content": content}, deepcopy(output)],
+                in_context_start=len(system) + 1, ac_start=len(system) + 1,
+                assistant_token_ids=[assistant_target_ids(self.tokenizer, output, [submit_answer_definition()])],
                 tools=[submit_answer_definition()],
-                completion_complete=True,
                 dataset_id=dataset_id,
                 doc_ids=[document.doc_id] + [candidate.doc_id for candidate in candidates[:num_distractors]],
                 info={"depth": depth, "budget": budget, "distractors": num_distractors, "ratio": ratio,
@@ -727,14 +561,15 @@
                        {"type": "text", "text": f"\n\n{RAG_QA_INSTRUCTIONS}\nQuestion: {question}"}]
             in_context_user = {"role": "user", "content": f"Passage:\n{gold.chunk_text}\n\n{RAG_QA_INSTRUCTIONS}\nQuestion: {question}"}
             system = [{"role": "system", "content": RAG_QA_SYSTEM_PROMPT}]
+            output = self.dialect.rendering.assistant_message("", [{"id": "answer", "name": "submit_answer", "arguments": {"answer": example.gold_answers[0].strip()}}])
             items.append(ActivationContextTrainingItem(
                 item_id=f"rag_qa:{dataset_id}:{seed}:{item_index}",
                 kind="rag_qa",
-                in_context_prefix=system + [in_context_user],
-                ac_prefix=system + [{"role": "user", "content": content}],
-                completion_text=self.dialect.preferred_format.render("submit_answer", {"answer": example.gold_answers[0].strip()}),
+                in_context_messages=system + [in_context_user, output],
+                ac_messages=system + [{"role": "user", "content": content}, deepcopy(output)],
+                in_context_start=len(system) + 1, ac_start=len(system) + 1,
+                assistant_token_ids=[assistant_target_ids(self.tokenizer, output, [submit_answer_definition()])],
                 tools=[submit_answer_definition()],
-                completion_complete=True,
                 dataset_id=dataset_id,
                 doc_ids=sorted({passage.doc_id for passage in passages}),
                 info={"depth": depth, "budget": budget, "passages": len(passages), "ratio": ratio,
@@ -749,7 +584,7 @@
             raise ValueError(kind)
         output = []
         for item in items:
-            messages = deepcopy(item.ac_prefix)
+            messages = deepcopy(item.ac_messages)
             part_index = 0
             for message in messages:
                 if not isinstance(message.get("content"), list):
@@ -771,7 +606,7 @@
                         pieces.append(self.tokenizer.decode(ids))
                     part_index += 1
                 message["content"] = "".join(pieces)
-            output.append(replace(item, item_id=item.item_id + ":" + kind, ac_prefix=messages,
+            output.append(replace(item, item_id=item.item_id + ":" + kind, ac_messages=messages,
                                   info=dict(item.info, reference_label="oracle gold text" if kind == "recent_text" and item.kind != "compaction" else kind)))
         return output
 
````

</details>

<details class="card" data-tressoir-markdown open>
  <summary><span class="card-title"><a href="codesign/activation/ac_model/ac_model_training.py">activation/ac_model/ac_model_training.py</a></span><span class="card-oneliner">KL or literal SFT on every retained assistant output, per-forward checkpoint policies and phase reset.</span><span class="card-badge">Diff</span></summary>

````diff
diff --git a/activation/ac_model/ac_model_training.py b/activation/ac_model/ac_model_training.py
--- a/activation/ac_model/ac_model_training.py
+++ b/activation/ac_model/ac_model_training.py
@@ -1,24 +1,8 @@
-"""
-Training the AC model in isolation (self-distillation): the target model reading the full context
-(the teacher, the same target adapter, no gradient) is matched by the target model reading the AC
-prefix (the student) on the completion tokens, with a full-vocabulary KL(teacher || student). One
-round trains the AC modules and the side LoRA (`learning_rate_ac`) and the target LoRA
-(`learning_rate_target_lora`) together; the AdamW state and the warm-up clock persist across
-rounds (one optimizer per AC model, parked on the host between rounds).
-
-Teacher tokens: the in-context prefix under the target chat template with the generation prompt
-open, then (for a mid-turn cut) the text the turn had already produced, then the completion text
-and, when the completion is a whole turn, the end-of-turn token. Student tokens: the AC prefix
-through `tokenize_with_parts` (rows written over the placeholder runs), then the same completion
-tokens. The loss averages over the completion positions.
-
-`drift_term_weight` > 0 adds KL(frozen base || target-with-adapter) on the in-context tokens so
-the target adapter cannot drift from the base's in-context behaviour; off by default (the teacher
-already carries the adapter).
-"""
+"""Whole-history AC self-distillation: current-reader KL or teacher-free supervised loss."""
 from __future__ import annotations
 
 import random
+import math
 import hashlib
 import json
 import shutil
@@ -34,13 +18,14 @@
 from torch.utils.checkpoint import checkpoint
 
 from ..agent_training.agent_training_utils import (
-    disable_dropout, head_logits, optimizer_state_to, persistent_optimizer, set_checkpointing, set_learning_rates,
+    disable_dropout, head_logits, optimizer_state_to, persistent_optimizer, set_learning_rates,
 )
 from ..autotuners import get_ac_training_autotuning_variables, configure_fla_runtime
 from ..harness.hf_utils import SOURCE_DEVICE, TARGET_DEVICE
 from ..common.data_syncing import resolve_path
 from .ac_model import ROLLOUT, TRAINING, ActivationContextModel
-from .ac_model_utils import EncodeRequest, direct_parts, tokenize_with_parts
+from .ac_model_utils import (EncodeRequest, direct_parts, prepare_training_history,
+                             validate_paired_histories, validate_part_capacity)
 
 if t.TYPE_CHECKING:
     from ..harness import HarnessRuntime
@@ -64,13 +49,11 @@
 class ActivationContextTrainingItem:
     item_id: str
     kind: str                                  # compaction | traj_qa | rag_qa (or a reference variant's name)
-    in_context_prefix: list[dict]              # the teacher's messages (plain text)
-    ac_prefix: list[dict]                      # the student's messages, with activation_context parts (or plain text for references)
-    completion_text: str                       # what both must predict (a whole turn or the rest of a cut turn)
-    completion_complete: bool = True           # the completion ends the turn: the end-of-turn token is a target too
-    teacher_partial_token_ids: list[int] | None = None
-    completion_token_ids: list[int] | None = None
-    teacher_partial_text: str = ""             # mid-turn cut: the text already produced, raw tokens for the teacher (the AC side holds it compressed)
+    in_context_messages: list[dict] | None    # full teacher view; unused in SFT
+    ac_messages: list[dict]                   # full student view, including all later input parts
+    in_context_start: int                     # independent logical starts, in message indexes
+    ac_start: int
+    assistant_token_ids: list[list[int]]      # one authoritative output per retained assistant message
     tools: list[dict] | None = None            # tool definitions for the template (both sides)
     weight: float = 1.0
     dataset_id: str = ""
@@ -80,6 +63,7 @@
 
 @dataclass
 class ActivationContextTrainingConfig:
+    loss_kind: t.Literal["kl", "sft"] = "kl"
     learning_rate_ac: float = 5e-4             # AC modules and the side LoRA
     learning_rate_target_lora: float = 2e-5    # the target adapter
     weight_decay: float = 0.0
@@ -89,8 +73,7 @@
     warmup_updates: int = 0                    # linear warm-up of both learning rates over this many updates (counted across calls)
     updates_per_epoch: int = 4                 # maximum gradient steps per epoch when examples_per_update is unset
     examples_per_update: int | None = None     # when set, overrides updates_per_epoch: ceil(items / examples_per_update) steps
-    max_example_tokens: int = 72_000           # longer teacher sequences are dropped (counted), never truncated: depth 2 at the
-                                               # default 32k compaction threshold, depth 1 up to a 50k threshold (task + completion included)
+    max_example_tokens: int = 72_000           # explicit error; no dropping or truncation
     teacher_cache_top_k: int = 0               # > 0: the teacher's top-k log-probs (+ the remainder mass) per completion position are
                                                # kept per item after its first pass and stand in for the teacher afterwards (epoch 2
                                                # skips the teacher forward); training then minimizes the KL of the k+1-way coarsened
@@ -109,6 +92,10 @@
     completion_max_new_tokens: int = 64
 
     def __post_init__(self) -> None:
+        if self.loss_kind not in ("kl", "sft") or self.max_example_tokens < 1 or self.teacher_cache_top_k < 0:
+            raise ValueError("Invalid loss kind, example capacity, or teacher cache size")
+        if self.loss_kind == "sft" and (self.teacher_cache_top_k != 0 or self.drift_term_weight != 0):
+            raise ValueError("SFT is incompatible with teacher_cache_top_k or drift_term_weight")
         if self.num_epochs < 1 or self.updates_per_epoch < 1 or not 0 < self.reporting_interval <= 1:
             raise ValueError("Epochs/updates must be positive and reporting_interval must be in (0, 1]")
         if self.examples_per_update is not None and self.examples_per_update < 1:
@@ -132,24 +119,30 @@
 class EvalItemSummary(t.TypedDict):
     item_id: str
     kind: str
-    kl: float
-    agreement: float
+    loss_kind: str
+    loss: float | None
+    kl: float | None
+    agreement: float | None
     positions: int
     teacher_tokens: int
     student_tokens: int
 
 
 class EvalKindSummary(t.TypedDict):
-    kl: float
-    agreement: float
+    loss_kind: str
+    loss: float | None
+    kl: float | None
+    agreement: float | None
     items: int
 
 
 class ActivationContextEvalSummary(t.TypedDict):
     items: int
     dropped_too_long: int
-    kl: float
-    agreement: float
+    loss_kind: str
+    loss: float | None
+    kl: float | None
+    agreement: float | None
     by_kind: dict[str, EvalKindSummary]
     per_item: list[EvalItemSummary]
 
@@ -158,7 +151,7 @@
     item_id: str
     kind: str
     reference: str
-    teacher: str
+    teacher: str | None
     student: str
     epoch_number: int
     global_step: int
@@ -191,6 +184,9 @@
 
 @dataclass
 class ActivationContextTrainingStats:
+    loss_kind: str = "kl"
+    loss: list[float] = field(default_factory=list)
+    loss_by_kind: dict[str, list[float]] = field(default_factory=dict)
     start_epoch: int = 0
     epochs: list[ActivationContextEpochStats] = field(default_factory=list)
     baseline_samples: list[CompletionSample] = field(default_factory=list)
@@ -240,6 +236,9 @@
         tokens = sum(self.step_tokens)
         seconds = sum(self.step_seconds)
         return {
+            "loss_kind": self.loss_kind, "loss_first": self.loss[0] if self.loss else None,
+            "loss_last": self.loss[-1] if self.loss else None,
+            "loss_by_kind": {kind: sum(values) / len(values) for kind, values in self.loss_by_kind.items() if values},
             "autotuning": self.autotuning,
             "start_epoch": self.start_epoch, "epochs": [asdict(epoch) for epoch in self.epochs],
             "placement_seconds": self.placement_seconds, "cleanup_seconds": self.cleanup_seconds, "other_seconds": self.other_seconds,
@@ -261,10 +260,16 @@
     item: ActivationContextTrainingItem
     teacher_ids: list[int]
     student_ids: list[int]
+    teacher_positions: list[int]
+    student_positions: list[int]
+    target_ids: list[int]
     spans: list[tuple[int, int]]
     part_requests: list[EncodeRequest]
-    num_completion: int                        # completion tokens (+ eot) at the end of both sequences
-    teacher_cache_key: tuple[str, str, int, int]
+    teacher_cache_key: tuple[str, str, int]
+
+    @property
+    def num_completion(self) -> int:
+        return len(self.target_ids)
 
 
 class ActivationContextTrainer:
@@ -274,34 +279,50 @@
         self.epochs_done: dict[str, int] = {}
         self.updates_done: dict[str, int] = {}                    # optimizer steps so far per AC model (the warm-up clock)
         self.optimizers: dict[str, torch.optim.AdamW] = {}        # one per AC model; moments survive train() calls
-        self.teacher_cache: dict[tuple[str, str, int, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}   # prepared-token identity -> (ids [C, k], log-probs [C, k], log remainder [C]), CPU
+        self.teacher_cache: dict[tuple[str, str, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}   # prepared-token identity -> (ids [C, k], log-probs [C, k], log remainder [C]), CPU
 
     # ------------------------------------------------------------------------------------------ examples
     def build_example(self, ac_model: ActivationContextModel, item: ActivationContextTrainingItem) -> _Example:
+        if not math.isfinite(item.weight) or item.weight < 0:
+            raise ValueError(f"{item.item_id}: weight must be finite and nonnegative")
         tokenizer = ac_model.target.tokenizer
-        eot = ac_model.target.model_config.model_description.eot_token
-        completion_ids = (list(item.completion_token_ids) if item.completion_token_ids is not None
-                          else tokenizer.encode(item.completion_text, add_special_tokens=False))
-        if item.completion_complete and eot:
-            completion_ids = completion_ids + tokenizer.encode(eot, add_special_tokens=False)
-        teacher_prefix = tokenizer.apply_chat_template(_flatten(item.in_context_prefix), tools=item.tools, add_generation_prompt=True, tokenize=True)
-        teacher_prefix = list(teacher_prefix["input_ids"] if hasattr(teacher_prefix, "keys") else teacher_prefix)
-        if item.teacher_partial_token_ids is not None:
-            teacher_prefix += list(item.teacher_partial_token_ids)
-        elif item.teacher_partial_text:
-            teacher_prefix += tokenizer.encode(item.teacher_partial_text, add_special_tokens=False)
-        parts = direct_parts(item.ac_prefix)
-        requests = [ac_model._child_request(part, ac_model.config.default_compression_ratio) for part in parts]
-        requests = [EncodeRequest(request.messages, request.compression_ratio, False, request.tools) for request in requests]   # top-level parts: target rows
+        chat_kwargs = item.info.get("chat_template_kwargs")
+        requests = [ac_model._child_request(part, ac_model.config.default_compression_ratio)
+                    for part in direct_parts(item.ac_messages)]
+        requests = [EncodeRequest(request.messages, request.compression_ratio, False, request.tools) for request in requests]
         lengths = [ac_model.part_view_rows(request.messages, request.compression_ratio, request.tools) for request in requests]
-        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (tokenizer.eos_token_id or 0)
-        student_prefix, spans = tokenize_with_parts(tokenizer, item.ac_prefix, lengths, pad_id, tools=item.tools, add_generation_prompt=True)
-        if not completion_ids or not teacher_prefix or not student_prefix:
-            raise ValueError(f"{item.item_id}: training requires nonempty prefixes and completion")
-        teacher_ids = teacher_prefix + completion_ids
-        digest = hashlib.sha256(json.dumps(teacher_ids, separators=(",", ":")).encode()).hexdigest()
-        key = (ac_model.name, digest, len(completion_ids), self.config.teacher_cache_top_k)
-        return _Example(item, teacher_ids, student_prefix + completion_ids, spans, requests, len(completion_ids), key)
+        student_ids, spans, student_positions = prepare_training_history(tokenizer, item.ac_messages, item.ac_start,
+                            item.assistant_token_ids, lengths, tools=item.tools, chat_template_kwargs=chat_kwargs)
+        teacher_ids, teacher_positions = [], []
+        if self.config.loss_kind == "kl":
+            if item.in_context_messages is None:
+                raise ValueError(f"{item.item_id}: KL needs an in-context teacher history")
+            validate_paired_histories(item.in_context_messages, item.in_context_start, item.ac_messages, item.ac_start)
+            teacher_ids, _, teacher_positions = prepare_training_history(tokenizer, item.in_context_messages,
+                                item.in_context_start, item.assistant_token_ids, [], tools=item.tools, chat_template_kwargs=chat_kwargs)
+        for view, ids in (("teacher", teacher_ids), ("student", student_ids)):
+            if len(ids) > self.config.max_example_tokens:
+                raise ValueError(f"{item.item_id}: {view} history has {len(ids)} tokens, exceeding max_example_tokens={self.config.max_example_tokens}; regenerate or raise the guard")
+        digest = hashlib.sha256(json.dumps([teacher_ids, teacher_positions], separators=(",", ":")).encode()).hexdigest()
+        key = (ac_model.name, digest, self.config.teacher_cache_top_k)
+        return _Example(item, teacher_ids, student_ids, teacher_positions, student_positions,
+                        [token for output in item.assistant_token_ids for token in output], spans, requests, key)
+
+    def _check_capacity(self, ac_model: ActivationContextModel, examples: t.Iterable[_Example]) -> None:
+        def capacity(model):
+            config = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
+            return getattr(config, "max_position_embeddings", None)
+        target_limit, side_limit = capacity(ac_model.target.model), capacity(ac_model.side.model)
+        for example in examples:
+            for view, ids in (("teacher", example.teacher_ids), ("student", example.student_ids)):
+                if target_limit and len(ids) > target_limit:
+                    raise ValueError(f"{example.item.item_id}: {view} reader history has {len(ids)} tokens, exceeding supported capacity {target_limit}")
+            if side_limit:
+                for request in example.part_requests:
+                    try:
+                        validate_part_capacity(ac_model, request, side_limit)
+                    except ValueError as error:
+                        raise ValueError(f"{example.item.item_id}: {error}") from error
 
     # ------------------------------------------------------------------------------------------ training
     def train(
@@ -312,27 +333,24 @@
         reporter: "ActivationContextTrainingReporter | None" = None,
         *,
         validation_data: t.Sequence[ActivationContextTrainingItem] = (),
+        reset_optimizer: bool = False,
     ) -> ActivationContextTrainingStats:
         """Complete epochs in one residency scope; optimizer/global step persist across calls."""
         config = self.config
-        assert training_data, "No training data."
+        if not training_data:
+            raise ValueError("No eligible AC training data")
         self.teacher_cache.clear()                 # teacher targets belong to this invocation only
         manager = self.harness.module_manager
         ac_model = manager.get_ac_model(ac_model_name)
         target = ac_model.target
         target_lora = ac_model.config.target_model_lora_name
         start_epoch = self._next_epoch_number(ac_model)
-        stats = ActivationContextTrainingStats(start_epoch=start_epoch, items=len(training_data))
+        stats = ActivationContextTrainingStats(loss_kind=config.loss_kind, start_epoch=start_epoch, items=len(training_data))
         started = time.monotonic()
         torch.manual_seed(config.seed + start_epoch)
-        examples = []
-        for item in training_data:
-            example = self.build_example(ac_model, item)
-            if max(len(example.teacher_ids), len(example.student_ids)) > config.max_example_tokens:
-                stats.dropped_too_long += 1
-            else:
-                examples.append(example)
-        assert examples, f"every item was dropped ({stats.dropped_too_long} too long)"
+        examples = [self.build_example(ac_model, item) for item in training_data]
+        reporting_examples = [self.build_example(ac_model, item) for item in reporting_data]
+        validation_examples = [self.build_example(ac_model, item) for item in validation_data]
         stats.examples = len(examples)
         stats.completion_tokens = sum(example.num_completion for example in examples)
         stats.teacher_tokens = sum(len(example.teacher_ids) for example in examples)
@@ -340,14 +358,7 @@
         per_step = config.examples_per_update or -(-len(examples) // min(config.updates_per_epoch, len(examples)))
         steps = -(-len(examples) // per_step)
         boundaries = set(reporting_boundaries(steps, config.reporting_interval))
-        fixed_items = []
-        if config.completion_samples:
-            for item in reporting_data:
-                example = self.build_example(ac_model, item)
-                if max(len(example.teacher_ids), len(example.student_ids)) <= config.max_example_tokens:
-                    fixed_items.append(item)
-                if len(fixed_items) == config.completion_samples:
-                    break
+        fixed_items = list(reporting_data[:config.completion_samples])
         kernel_scope = ExitStack()
         optimizer = None
         device = None
@@ -366,19 +377,19 @@
             manager.ensure_lora(target_lora)
             device = ac_model.prepare()
             ac_model.set_mode(TRAINING)
-            bases = [target.model, ac_model.side.model]
+            bases = list(dict.fromkeys([target.model, ac_model.side.model]))
             for base in bases:
                 base.train()
                 disable_dropout(base)
-            min_tokens = variables["target_gradient_checkpointing_min_tokens"]
-            side_min_tokens = variables["side_gradient_checkpointing_min_tokens"]
-            side_config = _CheckpointConfig(config.gradient_checkpointing and ac_model.config.side_gradient_checkpointing,
-                                            side_min_tokens)
-            set_checkpointing(target.model, config, min_tokens, min_tokens)
-            set_checkpointing(ac_model.side.model, side_config, side_min_tokens, side_min_tokens)
+            self._target_gradient_checkpointing_min_tokens = variables["target_gradient_checkpointing_min_tokens"]
+            ac_model._side_gradient_checkpointing_min_tokens = variables["side_gradient_checkpointing_min_tokens"]
+            ac_model._training_gradient_checkpointing = config.gradient_checkpointing
+            self._check_capacity(ac_model, [*examples, *reporting_examples, *validation_examples])
             ac_parameters = ac_model.trainable_parameters()
             target_parameters = manager.lora_parameters(target_lora)
             assert ac_parameters and target_parameters
+            if reset_optimizer:
+                self.optimizers.pop(ac_model_name, None)
             optimizer = persistent_optimizer(self.optimizers, ac_model_name,
                     [{"params": ac_parameters, "lr": config.learning_rate_ac}, {"params": target_parameters, "lr": config.learning_rate_target_lora}],
                     device, betas=config.adam_betas, eps=config.adam_eps, weight_decay=config.weight_decay)
@@ -412,23 +423,24 @@
                     before_side_tokens = ac_model.stats.side_tokens
                     mini = order[(step - 1) * per_step:step * per_step]
                     optimizer.zero_grad(set_to_none=True)
-                    totals = {"kl": 0.0, "agreement": 0.0, "drift": 0.0, "tokens": 0}
+                    totals = {"loss": 0.0, "agreement": 0.0, "drift": 0.0, "tokens": 0}
                     for index in mini:
                         example = examples[index]
                         example_started = time.monotonic()
-                        set_checkpointing(target.model, config, len(example.teacher_ids), min_tokens)
-                        cached = config.teacher_cache_top_k > 0 and example.teacher_cache_key in self.teacher_cache
-                        kl, agreement, drift, _ = self._example_loss(ac_model, example, device)
-                        loss = (kl + config.drift_term_weight * drift) * (example.item.weight / len(mini))
+                        cached = config.loss_kind == "kl" and config.teacher_cache_top_k > 0 and example.teacher_cache_key in self.teacher_cache
+                        objective, agreement, drift, _ = self._example_loss(ac_model, example, device)
+                        loss = (objective + config.drift_term_weight * drift) * (example.item.weight / len(mini))
                         loss.backward()
-                        kl_value, agreement_value = float(kl.detach()), float(agreement)
+                        loss_value = float(objective.detach())
                         stats.example_records.append({"kind": example.item.kind, "teacher_tokens": len(example.teacher_ids),
                             "student_tokens": len(example.student_ids), "seconds": round(time.monotonic() - example_started, 3), "teacher_cached": cached})
                         stats.teacher_cache_hits += int(cached)
-                        stats.kl_by_kind.setdefault(example.item.kind, []).append(kl_value)
-                        stats.agreement_by_kind.setdefault(example.item.kind, []).append(agreement_value)
-                        totals["kl"] += kl_value * example.item.weight
-                        totals["agreement"] += agreement_value
+                        stats.loss_by_kind.setdefault(example.item.kind, []).append(loss_value)
+                        totals["loss"] += loss_value * example.item.weight
+                        if config.loss_kind == "kl":
+                            stats.kl_by_kind.setdefault(example.item.kind, []).append(loss_value)
+                            stats.agreement_by_kind.setdefault(example.item.kind, []).append(agreement)
+                            totals["agreement"] += agreement
                         totals["drift"] += float(drift.detach()) * example.item.weight
                         totals["tokens"] += len(example.teacher_ids) + len(example.student_ids)
                     grad_norm = float(torch.nn.utils.clip_grad_norm_(ac_parameters + target_parameters, config.max_grad_norm))
@@ -443,9 +455,11 @@
                     seconds = time.monotonic() - step_started
                     record.training_seconds += seconds
                     stats.steps += 1
-                    stats.kl.append(totals["kl"] / len(mini))
-                    stats.agreement.append(totals["agreement"] / len(mini))
-                    stats.drift.append(totals["drift"] / len(mini))
+                    stats.loss.append(totals["loss"] / len(mini))
+                    if config.loss_kind == "kl":
+                        stats.kl.append(stats.loss[-1])
+                        stats.agreement.append(totals["agreement"] / len(mini))
+                        stats.drift.append(totals["drift"] / len(mini))
                     stats.grad_norm.append(grad_norm)
                     stats.step_seconds.append(seconds)
                     stats.step_tokens.append(totals["tokens"])
@@ -453,7 +467,7 @@
                     stats.global_steps.append(self.updates_done[ac_model_name])
                     stats.learning_rates.append([float(group["lr"]) for group in optimizer.param_groups])
                     print(f"AC epoch {epoch}/{config.num_epochs} step {step}/{steps} global {stats.global_steps[-1]}: "
-                          f"kl {stats.kl[-1]:.4f}, agreement {stats.agreement[-1]:.3f}, {seconds:.1f}s", flush=True)
+                          f"{config.loss_kind} loss {stats.loss[-1]:.4f}, {seconds:.1f}s", flush=True)
                     if reporter is not None:
                         reporter.report_step(progress(step), stats)
                     if step in boundaries and reporting_data:
@@ -505,12 +519,15 @@
                 base.eval()
                 if base.is_gradient_checkpointing:
                     base.gradient_checkpointing_disable()
+            ac_model.__dict__.pop("_training_gradient_checkpointing", None)
+            ac_model.__dict__.pop("_side_gradient_checkpointing_min_tokens", None)
             ac_model.set_mode(ROLLOUT)
             if device is not None and device.type == "cuda":
                 stats.peak_memory_bytes = int(torch.cuda.max_memory_allocated(device))
             ac_model.release()
             target.model_to_device(SOURCE_DEVICE)
-            ac_model.side.model_to_device(SOURCE_DEVICE)
+            if ac_model.side is not target:
+                ac_model.side.model_to_device(SOURCE_DEVICE)
             if torch.cuda.is_available():
                 torch.cuda.empty_cache()
         stats.cleanup_seconds = time.monotonic() - cleanup_started
@@ -559,10 +576,12 @@
             return []
         target = ac_model.target
         manager = self.harness.module_manager
-        bases = [target.model, ac_model.side.model]
+        bases = list(dict.fromkeys([target.model, ac_model.side.model]))
         modes = [(base.training, base.is_gradient_checkpointing) for base in bases]
         previous_mode = ac_model.mode
         result = []
+        model_config = target.model.config.get_text_config() if hasattr(target.model.config, "get_text_config") else target.model.config
+        capacity = getattr(model_config, "max_position_embeddings", None)
         try:
             ac_model.set_mode(TRAINING)
             for base in bases:
@@ -571,30 +590,32 @@
                     base.gradient_checkpointing_disable()
             for item in items:
                 example = self.build_example(ac_model, item)
-                if max(len(example.teacher_ids), len(example.student_ids)) > self.config.max_example_tokens:
-                    continue
                 texts = {}
-                for side in ("teacher", "student"):
+                for side in (("teacher", "student") if self.config.loss_kind == "kl" else ("student",)):
                     sequence = example.teacher_ids if side == "teacher" else example.student_ids
-                    ids = torch.tensor(sequence[:-example.num_completion], device=ac_model.device)
+                    positions = example.teacher_positions if side == "teacher" else example.student_positions
+                    prefix_length = positions[0] + 1
+                    ids = torch.tensor(sequence[:prefix_length], device=ac_model.device)
                     embeds = target.model.get_input_embeddings()(ids)
                     if side == "student" and example.part_requests:
-                        rows = ac_model.encode_batch(example.part_requests)
+                        present = [(span, request) for span, request in zip(example.spans, example.part_requests) if span[1] <= prefix_length]
+                        rows = ac_model.encode_batch([request for _, request in present]) if present else []
                         pieces, cursor = [], 0
-                        for (start, end), part_rows in zip(example.spans, rows):
+                        for ((start, end), _), part_rows in zip(present, rows):
                             pieces.extend([embeds[cursor:start], part_rows.to(embeds.dtype)])
                             cursor = end
                         embeds = torch.cat(pieces + [embeds[cursor:]], dim=0)
                     lora = ac_model.config.target_model_lora_name
                     peft_model = manager.ensure_lora(lora)
+                    budget = min(self.config.completion_max_new_tokens, capacity - prefix_length) if capacity else self.config.completion_max_new_tokens
                     with manager.lora_context(target.model_config.model_name, lora):
-                        output = peft_model.generate(inputs_embeds=embeds[None], max_new_tokens=self.config.completion_max_new_tokens,
+                        output = peft_model.generate(inputs_embeds=embeds[None], max_new_tokens=budget,
                                         do_sample=False, use_cache=True,
                                         pad_token_id=target.tokenizer.pad_token_id if target.tokenizer.pad_token_id is not None else target.tokenizer.eos_token_id)
                     texts[side] = target.tokenizer.decode(output[0], skip_special_tokens=True)
                 result.append(CompletionSample(item_id=item.item_id, kind=item.kind,
-                    reference=str(item.info.get("answer") or item.info.get("observed_terminal_answer") or item.completion_text),
-                    teacher=texts["teacher"], student=texts["student"], epoch_number=progress.epoch_number,
+                    reference=target.tokenizer.decode(item.assistant_token_ids[0], skip_special_tokens=False),
+                    teacher=texts.get("teacher"), student=texts["student"], epoch_number=progress.epoch_number,
                     global_step=progress.global_step, ac_version=ac_model.version, target_lora=ac_model.config.target_model_lora_name))
         finally:
             for base, (training, checkpointed) in zip(bases, modes):
@@ -635,50 +656,52 @@
             summary = self._evaluate(ac_model, items, device)
             if reporter is not None:
                 reporter.report_eval(progress, reporting_name or reference_name or "evaluation", summary)
-                if reference_name is not None:
-                    reporter.report_reference(reference_name, summary["kl"], provenance=reference_provenance)
+                if reference_name is not None and summary["items"]:
+                    reporter.report_reference(reference_name, summary["loss"], provenance=reference_provenance, loss_kind=self.config.loss_kind)
             return summary
         finally:
             kernel_scope.close()
             if release:
                 ac_model.release()
                 target.model_to_device(SOURCE_DEVICE)
-                ac_model.side.model_to_device(SOURCE_DEVICE)
+                if ac_model.side is not target:
+                    ac_model.side.model_to_device(SOURCE_DEVICE)
 
     # ------------------------------------------------------------------------------------------ internals
     @torch.no_grad()
     def _evaluate(self, ac_model: ActivationContextModel, items: list[ActivationContextTrainingItem], device: torch.device) -> ActivationContextEvalSummary:
+        examples = [self.build_example(ac_model, item) for item in items]
+        self._check_capacity(ac_model, examples)
         previous_mode = ac_model.mode
-        ac_model.set_mode(TRAINING)                                          # no cache; the no_grad above makes it free of gradients
-        target_base = ac_model.target.model
-        training_modes = [target_base.training, ac_model.side.model.training]
-        target_base.eval()
-        ac_model.side.model.eval()
+        ac_model.set_mode(TRAINING)
+        bases = list(dict.fromkeys([ac_model.target.model, ac_model.side.model]))
+        training_modes = [base.training for base in bases]
+        for base in bases:
+            base.eval()
         per_item = []
-        dropped = 0
+        is_kl = self.config.loss_kind == "kl"
         try:
-            for item in items:
-                example = self.build_example(ac_model, item)
-                if len(example.teacher_ids) > self.config.max_example_tokens or len(example.student_ids) > self.config.max_example_tokens:
-                    dropped += 1
-                    continue
-                kl, agreement, _, positions = self._example_loss(ac_model, example, device, with_drift=False)
-                per_item.append({"item_id": item.item_id, "kind": item.kind, "kl": float(kl), "agreement": float(agreement), "positions": positions,
-                                 "teacher_tokens": len(example.teacher_ids), "student_tokens": len(example.student_ids)})
+            for example in examples:
+                loss, agreement, _, positions = self._example_loss(ac_model, example, device, with_drift=False)
+                per_item.append({"item_id": example.item.item_id, "kind": example.item.kind,
+                    "loss_kind": self.config.loss_kind, "loss": float(loss), "kl": float(loss) if is_kl else None,
+                    "agreement": agreement, "positions": positions,
+                    "teacher_tokens": len(example.teacher_ids), "student_tokens": len(example.student_ids)})
         finally:
-            for base, training in zip((target_base, ac_model.side.model), training_modes):
+            for base, training in zip(bases, training_modes):
                 base.train(training)
                 disable_dropout(base)
             ac_model.set_mode(previous_mode)
-        by_kind: dict[str, list[dict]] = {}
-        for row in per_item:
-            by_kind.setdefault(row["kind"], []).append(row)
-        summary = {kind: {"kl": sum(r["kl"] for r in rows) / len(rows), "agreement": sum(r["agreement"] for r in rows) / len(rows), "items": len(rows)}
-                   for kind, rows in by_kind.items()}
-        return {"items": len(per_item), "dropped_too_long": dropped, "kl": sum(r["kl"] for r in per_item) / max(1, len(per_item)),
-                "agreement": sum(r["agreement"] for r in per_item) / max(1, len(per_item)), "by_kind": summary, "per_item": per_item}
-
-    def _example_loss(self, ac_model: ActivationContextModel, example: _Example, device: torch.device, with_drift: bool = True) -> tuple[torch.Tensor, float, torch.Tensor, int]:
+        def summarize(rows):
+            loss = sum(row["loss"] for row in rows) / len(rows) if rows else None
+            return {"loss_kind": self.config.loss_kind, "loss": loss, "kl": loss if is_kl else None,
+                    "agreement": sum(row["agreement"] for row in rows) / len(rows) if is_kl and rows else None,
+                    "items": len(rows)}
+        by_kind = {kind: summarize([row for row in per_item if row["kind"] == kind])
+                   for kind in dict.fromkeys(row["kind"] for row in per_item)}
+        return {**summarize(per_item), "dropped_too_long": 0, "by_kind": by_kind, "per_item": per_item}
+
+    def _example_loss(self, ac_model: ActivationContextModel, example: _Example, device: torch.device, with_drift: bool = True) -> tuple[torch.Tensor, float | None, torch.Tensor, int]:
         """(kl, agreement, drift, positions) of one example: teacher no-grad, student with the rows; drift on the in-context tokens when configured."""
         config = self.config
         target = ac_model.target
@@ -688,20 +711,21 @@
         head = base.get_output_embeddings()
         lora = ac_model.config.target_model_lora_name
         num = example.num_completion
-        top_k = config.teacher_cache_top_k if torch.is_grad_enabled() else 0                  # eval is exact
+        is_kl = config.loss_kind == "kl"
+        top_k = config.teacher_cache_top_k if is_kl and torch.is_grad_enabled() else 0                  # eval is exact
         cached = self.teacher_cache.get(example.teacher_cache_key) if top_k else None
-        teacher_ids = torch.tensor(example.teacher_ids, device=device)
+        teacher_ids = torch.tensor(example.teacher_ids, device=device, dtype=torch.long) if is_kl else None
         teacher_states = None
-        if cached is None:
+        if is_kl and cached is None:
             # Teacher.
             with torch.no_grad():
-                teacher_hidden = target.decoder_forward(embedding(teacher_ids)[None], None, lora_name=lora)[0]
-                teacher_states = teacher_hidden[-num - 1:-1]
+                teacher_hidden = target.decoder_forward(embedding(teacher_ids)[None], None, lora_name=lora, gradient_checkpointing=False)[0]
+                teacher_states = teacher_hidden[example.teacher_positions]
             if top_k:
                 cached = self._teacher_top_k(teacher_states, head, top_k)
                 self.teacher_cache[example.teacher_cache_key] = tuple(tensor.cpu() for tensor in cached)
                 teacher_states = None
-        else:
+        elif cached is not None:
             cached = tuple(tensor.to(device, non_blocking=True) for tensor in cached)
         # Student: rows for the parts (grad), written over the placeholders.
         rows = ac_model.encode_batch(example.part_requests) if example.part_requests else []
@@ -716,19 +740,39 @@
                 cursor = end
             pieces.append(student_embeds[cursor:])
             student_embeds = torch.cat(pieces, dim=0)
-        student_hidden = target.decoder_forward(student_embeds[None], None, lora_name=lora)[0]
-        student_states = student_hidden[-num - 1:-1]
-        kl, agreement = self._chunked_kl(teacher_states, student_states, head, cached=cached)
+        checkpointed = (config.gradient_checkpointing and torch.is_grad_enabled()
+                        and len(example.student_ids) >= getattr(self, "_target_gradient_checkpointing_min_tokens", config.gradient_checkpointing_min_tokens or 0))
+        student_hidden = target.decoder_forward(student_embeds[None], None, lora_name=lora, gradient_checkpointing=checkpointed)[0]
+        student_states = student_hidden[example.student_positions]
+        if is_kl:
+            loss, agreement = self._chunked_kl(teacher_states, student_states, head, cached=cached)
+        else:
+            loss, agreement = self._chunked_sft(student_states, head, example.target_ids), None
         drift = torch.zeros((), device=device)
-        if with_drift and config.drift_term_weight > 0:
+        if is_kl and with_drift and config.drift_term_weight > 0:
             # The adapter's in-context behaviour on the last in-context positions, held to the frozen base's.
-            prefix = len(example.teacher_ids) - num
+            prefix = example.teacher_positions[0] + 1
             window = slice(max(0, prefix - 1024), prefix)
             with torch.no_grad():
-                frozen_states = target.decoder_forward(embedding(teacher_ids)[None], None, lora_name=None)[0][window]
-            adapted_states = target.decoder_forward(embedding(teacher_ids)[None], None, lora_name=lora)[0][window]
+                frozen_states = target.decoder_forward(embedding(teacher_ids)[None], None, lora_name=None, gradient_checkpointing=False)[0][window]
+            drift_checkpointed = (config.gradient_checkpointing and torch.is_grad_enabled()
+                                  and len(example.teacher_ids) >= getattr(self, "_target_gradient_checkpointing_min_tokens", config.gradient_checkpointing_min_tokens or 0))
+            adapted_states = target.decoder_forward(embedding(teacher_ids)[None], None, lora_name=lora,
+                                                    gradient_checkpointing=drift_checkpointed)[0][window]
             drift, _ = self._chunked_kl(frozen_states, adapted_states, head)
-        return kl, agreement, drift, num
+        return loss, agreement, drift, num
+
+    def _chunked_sft(self, states: torch.Tensor, head: torch.nn.Module, target_ids: list[int]) -> torch.Tensor:
+        chunk = getattr(self, "_logits_chunk_tokens", self.config.logits_chunk_tokens or 64)
+        targets = torch.tensor(target_ids, device=states.device)
+        def cross_entropy(hidden, labels):
+            return F.cross_entropy(head_logits(hidden, head, self.config.fp32_head_matmul), labels, reduction="sum")
+        total = states.new_zeros((), dtype=torch.float32)
+        for start in range(0, len(target_ids), chunk):
+            arguments = states[start:start + chunk], targets[start:start + chunk]
+            total = total + (checkpoint(cross_entropy, *arguments, use_reentrant=False)
+                             if torch.is_grad_enabled() and states.requires_grad else cross_entropy(*arguments))
+        return total / len(target_ids)
 
     @torch.no_grad()
     def _teacher_top_k(self, teacher_states: torch.Tensor, head: torch.nn.Module, k: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
@@ -792,21 +836,3 @@
                         self.config, ac_config=ac_model.config, device=TARGET_DEVICE)
         self._logits_chunk_tokens = variables["logits_chunk_tokens"]
         return variables
-
-
-
-@dataclass
-class _CheckpointConfig:
-    gradient_checkpointing: bool
-    gradient_checkpointing_min_tokens: int | None
-
-
-def _flatten(messages: list[dict]) -> list[dict]:
-    out = []
-    for message in messages:
-        message = dict(message)
-        content = message.get("content")
-        if isinstance(content, list):
-            message["content"] = "".join(part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text")
-        out.append(message)
-    return out
````

</details>

<details class="card" data-tressoir-markdown open>
  <summary><span class="card-title"><a href="codesign/activation/ac_model/ac_model_reporter.py">activation/ac_model/ac_model_reporter.py</a></span><span class="card-oneliner">Report the selected objective and reject mixed loss modes in one report.</span><span class="card-badge">Diff</span></summary>

````diff
diff --git a/activation/ac_model/ac_model_reporter.py b/activation/ac_model/ac_model_reporter.py
--- a/activation/ac_model/ac_model_reporter.py
+++ b/activation/ac_model/ac_model_reporter.py
@@ -17,26 +17,49 @@
 class ActivationContextTrainingReporter(HtmlReporter):
     def __init__(self, folder: str, title: str = "AC training", description: str = "", **kwargs: t.Any) -> None:
         super().__init__(folder, title, description, eyebrow="AC training", **kwargs)
+        self.loss_kind: str | None = None
         self.references: dict[str, float] = {}
         self.global_step = 0
         self.epoch_x = 0.0
         self.epoch_origin: int | None = None
-        self.initialize_line_plot("kl", "Training KL", "Weighted item mean; cached teachers use coarsened KL.",
-                                  "global optimizer step", "KL", ["train"], smoothing_window=5)
-        self.initialize_line_plot("reporting", "Exact held-out KL", "Current-adapter reporting/validation, with labeled initial-adapter references.",
-                                  "epochs in this report", "KL", ["reporting", "validation", *REFERENCE_SERIES])
-        self.initialize_line_plot("agreement", "Top-1 agreement", "Mean teacher/student agreement over items.", "global optimizer step", "fraction", ["train"])
-        self.initialize_line_plot("throughput", "Token-equivalent throughput", "Original teacher + student tokens per training second; cached teacher tokens are equivalent work, not executed tokens.",
+        self.initialize_line_plot("loss", "Training loss", "Weighted item mean.", "global optimizer step", "loss", ["train"], smoothing_window=5)
+        self.initialize_line_plot("reporting", "Held-out loss", "Current-reader reporting/validation and initial-reader references.",
+                                  "epochs in this report", "loss", ["reporting", "validation", *REFERENCE_SERIES])
+        self.initialize_line_plot("throughput", "Token throughput", "Teacher and student tokens per training second.",
                                   "global optimizer step", "tokens/s", ["tokens/s"])
         self.initialize_line_plot("grad", "Gradient norm", "Before clipping.", "global optimizer step", "norm", ["grad_norm"])
         self.initialize_table("epochs", "Epochs", "Phase times in seconds.",
                               ["epoch", "global step", "train s", "report s", "validation s", "samples s", "checkpoint s", "total s", "checkpoint"])
-        self.initialize_table("kinds", "Per kind", "Exact reporting/validation results.", ["epoch", "set", "kind", "items", "kl", "agreement"])
+        self.initialize_table("kinds", "Per kind", "Exact reporting/validation results.", ["epoch", "set", "kind", "items", "loss kind", "loss", "agreement"])
         self.initialize_table("evals", "Evaluations", "Exact current-adapter results; empty and dropped sets are counted.",
-                              ["epoch", "global step", "name", "items", "dropped", "kl", "agreement"])
-        self.initialize_table("references", "Reference measurements", "QA recent_text is oracle gold text; compaction uses the text tail.", ["name", "KL", "measured with"])
+                              ["epoch", "global step", "name", "items", "dropped", "loss kind", "loss", "agreement"])
+        self.initialize_table("references", "Reference measurements", "QA recent_text is oracle gold text; compaction uses the text tail.", ["name", "loss kind", "loss", "measured with"])
+
+    def _set_loss_kind(self, loss_kind: str) -> None:
+        if self.loss_kind == loss_kind:
+            return
+        if loss_kind not in ("kl", "sft") or self.loss_kind is not None:
+            raise ValueError("Use a separate AC reporter for each loss kind")
+        self.loss_kind = loss_kind
+        label = "KL" if loss_kind == "kl" else "Cross-entropy"
+        for name in ("loss", "reporting"):
+            self.widgets[name]["y_label"] = label
+            self.widgets[name]["title"] = ("Training " if name == "loss" else "Held-out ") + label
+        if loss_kind == "kl":
+            self.initialize_line_plot("agreement", "Top-1 agreement", "Mean teacher/student agreement over items.",
+                                      "global optimizer step", "fraction", ["train"])
+            self.widgets["loss"]["description"] = "Weighted item mean; cached teachers use coarsened KL."
+            self.widgets["throughput"]["description"] = "Original teacher + student tokens; cached teachers count as equivalent work."
+        else:
+            self.remove_widget("agreement")
+            self.remove_widget("rates")
+            for name in ("kinds", "evals"):
+                self.widgets[name]["columns"] = [column for column in self.widgets[name]["columns"] if column != "agreement"]
+            self.widgets["loss"]["description"] = "Weighted mean cross-entropy over retained assistant tokens."
+            self.widgets["throughput"]["description"] = "Student tokens per training second."
 
     def initialize_training(self, config: "ActivationContextTrainingConfig", stats: "ActivationContextTrainingStats") -> None:
+        self._set_loss_kind(config.loss_kind)
         if self.epoch_origin is None:
             self.epoch_origin = stats.start_epoch - 1
         self.epoch_x = float(stats.start_epoch - 1 - self.epoch_origin)
@@ -56,40 +79,49 @@
         self.render()
 
     def report_step(self, progress: "ActivationContextTrainingProgress", stats: "ActivationContextTrainingStats") -> None:
+        self._set_loss_kind(stats.loss_kind)
         self.report_phase(progress, "training")
-        self.add_data_point("kl", {"x": progress.global_step, "train": stats.kl[-1]})
-        self.add_data_point("agreement", {"x": progress.global_step, "train": stats.agreement[-1]})
+        self.add_data_point("loss", {"x": progress.global_step, "train": stats.loss[-1]})
+        if stats.loss_kind == "kl":
+            self.add_data_point("agreement", {"x": progress.global_step, "train": stats.agreement[-1]})
         self.add_data_point("throughput", {"x": progress.global_step, "tokens/s": stats.step_tokens[-1] / max(stats.step_seconds[-1], 1e-6)})
         self.add_data_point("grad", {"x": progress.global_step, "grad_norm": stats.grad_norm[-1]})
         if self.references:
             self.add_data_point("reporting", {"x": self.epoch_x, **self.references})
-        self.set_status(kl=f"{stats.kl[-1]:.4f}", agreement=f"{stats.agreement[-1]:.3f}", learning_rates=", ".join(f"{lr:.3g}" for lr in stats.learning_rates[-1]))
+        self.set_status(loss_kind=stats.loss_kind, loss=f"{stats.loss[-1]:.4f}", learning_rates=", ".join(f"{lr:.3g}" for lr in stats.learning_rates[-1]))
+        if stats.loss_kind == "kl":
+            self.set_status(agreement=f"{stats.agreement[-1]:.3f}")
         self.render()
 
-    def report_reference(self, name: "ReferenceName", kl: float, *, provenance: str = "initial adapter; held set") -> None:
+    def report_reference(self, name: "ReferenceName", loss: float, *, provenance: str = "initial adapter; held set", loss_kind: str = "kl") -> None:
         assert name in REFERENCE_SERIES, name
-        self.references[name] = kl
-        self.add_data_point("references", {"name": name, "KL": kl, "measured with": provenance})
-        self.add_data_point("reporting", {"x": 0, name: kl})
+        self._set_loss_kind(loss_kind)
+        self.references[name] = loss
+        self.add_data_point("references", {"name": name, "loss kind": loss_kind, "loss": loss, "measured with": provenance})
+        self.add_data_point("reporting", {"x": 0, name: loss})
         if self.epoch_x:
-            self.add_data_point("reporting", {"x": self.epoch_x, name: kl})
+            self.add_data_point("reporting", {"x": self.epoch_x, name: loss})
         self.render()
 
     def report_eval(self, progress: "ActivationContextTrainingProgress | None", name: str, summary: "ActivationContextEvalSummary") -> None:
+        self._set_loss_kind(summary["loss_kind"])
         if progress is not None:
             self.report_phase(progress, name)
         epoch = progress.epoch_number - (self.epoch_origin or 0) if progress else "external"
         self.add_data_point("evals", {"epoch": epoch, "global step": self.global_step, "name": name, "items": summary["items"],
-                                      "dropped": summary["dropped_too_long"], "kl": summary["kl"], "agreement": summary["agreement"]})
+                                      "dropped": summary["dropped_too_long"], "loss kind": summary["loss_kind"], "loss": summary["loss"],
+                                      **({"agreement": summary["agreement"]} if self.loss_kind == "kl" else {})})
         if name in ("reporting", "validation") and summary["items"]:
-            self.add_data_point("reporting", {"x": self.epoch_x, name: summary["kl"], **self.references})
+            self.add_data_point("reporting", {"x": self.epoch_x, name: summary["loss"], **self.references})
         for kind, values in summary["by_kind"].items():
-            self.add_data_point("kinds", {"epoch": epoch, "set": name, "kind": kind, **values})
+            self.add_data_point("kinds", {"epoch": epoch, "set": name, "kind": kind, "items": values["items"],
+                "loss kind": values["loss_kind"], "loss": values["loss"],
+                **({"agreement": values["agreement"]} if self.loss_kind == "kl" else {})})
         self.render()
 
     def report_completions(self, progress: "ActivationContextTrainingProgress", rows: list["CompletionSample"]) -> None:
         text = "\n\n".join(f"[{r['item_id']}] ({r['kind']}, global step {r['global_step']}, AC version {r['ac_version']})\n"
-                           f"reference: {r['reference']}\nteacher: {r['teacher']}\nstudent: {r['student']}" for r in rows)
+                           f"reference: {r['reference']}\n" + (f"teacher: {r['teacher']}\n" if r["teacher"] is not None else "") + f"student: {r['student']}" for r in rows)
         label = "Baseline" if progress.epoch == 0 else f"Epoch {progress.epoch_number - (self.epoch_origin or 0)}"
         key = f"completions_{'baseline' if progress.epoch == 0 else 'epoch'}_{progress.epoch_number:03d}"
         self.set_text(key, f"{label} inspection samples", text)
@@ -106,9 +138,10 @@
 
     def report_training(self, stats: "ActivationContextTrainingStats") -> None:
         self.set_status(phase="complete", peak_memory=f"{stats.peak_memory_bytes / 2**30:.2f} GB", elapsed=f"{stats.duration_s:.1f}s")
-        groups = {name: [r["seconds"] for r in stats.example_records if r["teacher_cached"] == cached]
-                  for name, cached in (("live", False), ("cached", True))}
-        self.set_text("rates", "Live and cached teacher examples", json.dumps({name: {"examples": len(values), "seconds": sum(values),
-            "examples_per_second": len(values) / sum(values) if sum(values) else None} for name, values in groups.items()}, indent=2))
+        if stats.loss_kind == "kl":
+            groups = {name: [r["seconds"] for r in stats.example_records if r["teacher_cached"] == cached]
+                      for name, cached in (("live", False), ("cached", True))}
+            self.set_text("rates", "Live and cached teacher examples", json.dumps({name: {"examples": len(values), "seconds": sum(values),
+                "examples_per_second": len(values) / sum(values) if sum(values) else None} for name, values in groups.items()}, indent=2))
         self.set_text("training_summary", "Training summary", json.dumps(stats.summarize(), indent=2))
         self.render(force=True)
````

</details>

<details class="card" data-tressoir-markdown open>
  <summary><span class="card-title"><a href="codesign/activation/ac_model/ac_model.py">activation/ac_model/ac_model.py</a></span><span class="card-oneliner">Allow one shared base with distinct adapters and choose side checkpointing per forward.</span><span class="card-badge">Diff</span></summary>

````diff
diff --git a/activation/ac_model/ac_model.py b/activation/ac_model/ac_model.py
--- a/activation/ac_model/ac_model.py
+++ b/activation/ac_model/ac_model.py
@@ -159,8 +159,8 @@
         self.name = config.ac_model_name
         self.side = harness.loaded_models[config.base_side_model_name]
         self.target = harness.loaded_models[config.target_model_name]
-        if self.side is self.target:
-            raise ValueError("AC side and target require separate LoadedModel objects (separate loads of one model ID are allowed)")
+        if config.base_side_model_lora_name == config.target_model_lora_name:
+            raise ValueError("AC side and reader require distinct LoRA adapters")
         self.d_side = self.side.model_config.model_description.d_model
         self.d_target = self.target.model_config.model_description.d_model
         self.modules = ActivationContextModules(config, self.d_side, self.d_target)
@@ -410,7 +410,11 @@
             attention_mask = torch.zeros(batch_size, max_seq, device=device, dtype=torch.long)
             for row, length in enumerate(seq_lengths):
                 attention_mask[row, :length] = 1
-        hidden = self.side.decoder_forward(inputs_embeds.to(base_dtype), attention_mask, lora_name=self.config.base_side_model_lora_name)
+        enabled = (self.mode == TRAINING and torch.is_grad_enabled() and self.config.side_gradient_checkpointing
+                   and getattr(self, "_training_gradient_checkpointing", True)
+                   and max_seq >= getattr(self, "_side_gradient_checkpointing_min_tokens", 0))
+        hidden = self.side.decoder_forward(inputs_embeds.to(base_dtype), attention_mask,
+                    lora_name=self.config.base_side_model_lora_name, gradient_checkpointing=enabled)
         phase_started = self._phase("side_decoder", phase_started, device)
         for row, (ids, _, _, num_rows, is_recursive) in enumerate(items):
             end = seq_lengths[row]
````

</details>

<details class="card" data-tressoir-markdown open>
  <summary><span class="card-title"><a href="codesign/activation/harness/module_manager.py">activation/harness/module_manager.py</a></span><span class="card-oneliner">Preserve adapter activation, disabling and gradient ownership through nested contexts.</span><span class="card-badge">Diff</span></summary>

````diff
diff --git a/activation/harness/module_manager.py b/activation/harness/module_manager.py
--- a/activation/harness/module_manager.py
+++ b/activation/harness/module_manager.py
@@ -194,28 +194,45 @@
         """
         if lora_name is None:
             peft_model = self.peft_models.get(model_name)
-            if peft_model is None:
-                yield
-            else:
-                with peft_model.disable_adapter():
-                    yield
+        else:
+            lora_config = self.get_lora_config(lora_name)
+            assert lora_config.model_name == model_name, f"LoRA {lora_name!r} belongs to {lora_config.model_name!r}, not {model_name!r}."
+            peft_model = self.ensure_lora(lora_name)
+        if peft_model is None:
+            yield
             return
-        lora_config = self.get_lora_config(lora_name)
-        assert lora_config.model_name == model_name, f"LoRA {lora_name!r} belongs to {lora_config.model_name!r}, not {model_name!r}."
-        peft_model = self.ensure_lora(lora_name)
-        peft_model.set_adapter(lora_config.adapter_name)
-        yield
+        previous_adapter = peft_model.active_adapter
+        trainability = [(parameter, parameter.requires_grad) for parameter in peft_model.parameters()]
+        from peft.tuners.tuners_utils import BaseTunerLayer
+        layers = [(module, module.disable_adapters) for module in peft_model.modules() if isinstance(module, BaseTunerLayer)]
+        try:
+            if lora_name is not None:
+                peft_model.set_adapter(lora_config.adapter_name)
+            for layer, _ in layers:
+                layer.enable_adapters(lora_name is not None)
+            # PEFT routing changes trainability. Optimizer ownership, not the active adapter, controls it here.
+            for parameter, trainable in trainability:
+                parameter.requires_grad_(trainable)
+            yield
+        finally:
+            peft_model.set_adapter(previous_adapter)
+            for layer, disabled in layers:
+                layer.enable_adapters(not disabled)
+            for parameter, trainable in trainability:
+                parameter.requires_grad_(trainable)
 
     def lora_parameters(self, lora_name: str) -> list[nn.Parameter]:
         """The trainable parameters of that adapter only (injecting it if needed)."""
         peft_model = self.ensure_lora(lora_name)
         adapter_name = self.get_lora_config(lora_name).adapter_name
-        peft_model.set_adapter(adapter_name)
-        return [
+        parameters = [
             parameter
             for name, parameter in peft_model.named_parameters()
-            if f".{adapter_name}." in name and parameter.requires_grad
+            if f".{adapter_name}." in name
         ]
+        for parameter in parameters:
+            parameter.requires_grad_(True)
+        return parameters
 
     def free_lora(self, model_name: str, lora_name: str, checkpoint_path: str | None = None) -> None:
         """
````

</details>

<details class="card" data-tressoir-markdown open>
  <summary><span class="card-title"><a href="codesign/activation/harness/loaded_model.py">activation/harness/loaded_model.py</a></span><span class="card-oneliner">Scope decoder forwards and recomputation to their original adapter.</span><span class="card-badge">Diff</span></summary>

````diff
diff --git a/activation/harness/loaded_model.py b/activation/harness/loaded_model.py
--- a/activation/harness/loaded_model.py
+++ b/activation/harness/loaded_model.py
@@ -487,6 +487,7 @@
         attention_mask: torch.Tensor | None,
         position_ids: torch.Tensor|None = None,
         lora_name: str|None = None,
+        gradient_checkpointing: bool | None = None,
         **kwargs,
     ) -> torch.Tensor:
         """
@@ -500,7 +501,9 @@
         assert self.model is not None, f"{self.model_config.model_name} - Model is not loaded."
         model = self.model
         decoder = model.get_decoder() if hasattr(model, "get_decoder") else getattr(model, model.base_model_prefix)
-        with self.harness.module_manager.lora_context(self.model_config.model_name, lora_name):
+        from .hf_utils import checkpoint_adapter_scope
+        adapter_context = lambda: self.harness.module_manager.lora_context(self.model_config.model_name, lora_name)
+        with adapter_context(), checkpoint_adapter_scope(model, adapter_context, gradient_checkpointing):
             outputs = decoder(
                 inputs_embeds=inputs_embeds,
                 attention_mask=attention_mask,
@@ -532,4 +535,4 @@
 
     def simple_vector_embed(self, text: str) -> torch.Tensor:
         """Simple function to test embeddings."""
-        return self.simple_vector_embed_many([text])[0]
\ No newline at end of file
+        return self.simple_vector_embed_many([text])[0]
````

</details>

<details class="card" data-tressoir-markdown open>
  <summary><span class="card-title"><a href="codesign/activation/harness/hf_utils.py">activation/harness/hf_utils.py</a></span><span class="card-oneliner">Keep checkpoint callback and adapter-context details in existing HF utilities.</span><span class="card-badge">Diff</span></summary>

````diff
diff --git a/activation/harness/hf_utils.py b/activation/harness/hf_utils.py
--- a/activation/harness/hf_utils.py
+++ b/activation/harness/hf_utils.py
@@ -3,6 +3,7 @@
 Stores overly detailed/specific logic to keep the rest of code cleaner.
 """
 from collections import Counter
+from contextlib import contextmanager
 import typing as t
 import torch
 import torch.nn.functional as F
@@ -41,6 +42,36 @@
     {"gated_delta_net", "linear_attention", "mamba", "recurrent", "ssm"}
 )
 _FFN_ONLY_NAMES = frozenset({"mlp", "moe"})
+
+
+@contextmanager
+def checkpoint_adapter_scope(model, adapter_context, enabled: bool | None = None):
+    """Bind each layer's recomputation to the adapter used for its original forward."""
+    previous = [(module, module.gradient_checkpointing, getattr(module, "_gradient_checkpointing_func", None))
+                for module in model.modules() if hasattr(module, "gradient_checkpointing")]
+    was_enabled = model.is_gradient_checkpointing
+    if enabled is not None and enabled != was_enabled:
+        if enabled:
+            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
+        else:
+            model.gradient_checkpointing_disable()
+    try:
+        for module in model.modules():
+            if not getattr(module, "gradient_checkpointing", False) or not hasattr(module, "_gradient_checkpointing_func"):
+                continue
+            original = module._gradient_checkpointing_func
+            def scoped(function, *args, _checkpoint=original, **kwargs):
+                return _checkpoint(function, *args, **(kwargs | {
+                    "use_reentrant": False, "context_fn": lambda: (adapter_context(), adapter_context())}))
+            module._gradient_checkpointing_func = scoped
+        yield
+    finally:
+        for module, checkpointed, original in previous:
+            module.gradient_checkpointing = checkpointed
+            if original is not None:
+                module._gradient_checkpointing_func = original
+            elif hasattr(module, "_gradient_checkpointing_func"):
+                del module._gradient_checkpointing_func
 
 
 
````

</details>

<details class="card" data-tressoir-markdown open>
  <summary><span class="card-title"><a href="codesign/activation/dataset/loaders/trajectory_utils.py">activation/dataset/loaders/trajectory_utils.py</a></span><span class="card-oneliner">Retain separate reasoning in normalized source trajectories.</span><span class="card-badge">Diff</span></summary>

````diff
diff --git a/activation/dataset/loaders/trajectory_utils.py b/activation/dataset/loaders/trajectory_utils.py
--- a/activation/dataset/loaders/trajectory_utils.py
+++ b/activation/dataset/loaders/trajectory_utils.py
@@ -14,7 +14,7 @@
 into our messages: the source system prompt is dropped (a short domain prompt rides in
 `trajectory_kwargs["system_prompt"]`), tool names and argument keys are mapped onto our tools
 (`shell(script)`, `python(code)`; unknown tools keep their names and definitions), reasoning stays
-as plain text ahead of the reply (`reasoning="keep"`, since the math `cot` rows are nothing but
+in its separate field (`reasoning="keep"`, since the math `cot` rows are nothing but
 reasoning) or is dropped, calls become structured `tool_calls` (the shape `MessageRendering.assistant_message`
 produces), and every tool result is one `tool` message.
 """
@@ -25,6 +25,7 @@
 import typing as t
 
 from ...agent.agent_utils import parse_tool_calls
+from ..dataset_utils import message_text
 
 TOOL_MAP: dict[str, tuple[str, dict[str, str]]] = {
     "bash": ("shell", {"command": "script"}),
@@ -206,9 +207,9 @@
             calls = [map_call(call, tool_map) for call in message.get("tool_calls") or []]
             text = str(message.get("content") or "").strip()
             reasoning_text = (message.get("reasoning") or "").strip()
+            out: dict = {"role": "assistant", "content": text}
             if reasoning == "keep" and reasoning_text:
-                text = f"{reasoning_text}\n\n{text}".strip() if text else reasoning_text
-            out: dict = {"role": "assistant", "content": text}
+                out["reasoning"] = reasoning_text
             if calls:
                 out["tool_calls"] = []
                 for call in calls:
@@ -253,7 +254,7 @@
     """Characters of content and arguments, the cheap size measure loaders filter on."""
     total = 0
     for message in messages:
-        total += len(str(message.get("content") or ""))
+        total += len(message_text(message))
         for call in message.get("tool_calls") or []:
             total += len(json.dumps(call.get("function", call).get("arguments", {}), ensure_ascii=False))
     return total
````

</details>

<details class="card" data-tressoir-markdown open>
  <summary><span class="card-title"><a href="codesign/activation/dataset/dataset_utils.py">activation/dataset/dataset_utils.py</a></span><span class="card-oneliner">Keep reasoning visible exactly once in source text, indexing and teacher transcripts.</span><span class="card-badge">Diff</span></summary>

````diff
diff --git a/activation/dataset/dataset_utils.py b/activation/dataset/dataset_utils.py
--- a/activation/dataset/dataset_utils.py
+++ b/activation/dataset/dataset_utils.py
@@ -69,19 +69,18 @@
 def message_text(message: dict) -> str:
     """The text of a message whose content is a string or a list of parts (activation_context parts render as a marker)."""
     content = message.get("content")
-    if content is None:
-        return ""
-    if isinstance(content, str):
-        return content
-    pieces = []
-    for part in content:
-        if not isinstance(part, dict):
-            pieces.append(str(part))
-        elif part.get("type") == "text":
-            pieces.append(part.get("text", ""))
-        elif part.get("type") == "activation_context":
-            pieces.append("[activation context]")
-    return "".join(pieces)
+    if isinstance(content, list):
+        pieces = []
+        for part in content:
+            if not isinstance(part, dict):
+                pieces.append(str(part))
+            elif part.get("type") == "text":
+                pieces.append(part.get("text", ""))
+            elif part.get("type") == "activation_context":
+                pieces.append("[activation context]")
+        content = "".join(pieces)
+    reasoning = message.get("reasoning") or message.get("reasoning_content") or ""
+    return "\n\n".join(piece for piece in (str(reasoning), str(content or "")) if piece)
 
 
 def render_message(message: dict) -> str:
````

</details>

<details class="card" data-tressoir-markdown open>
  <summary><span class="card-title"><a href="codesign/activation/tests/test_basic_agent_ac_training.py">activation/tests/test_basic_agent_ac_training.py</a></span><span class="card-oneliner">Complete public-compaction, trajectory-QA and RAG-QA training examples on real data.</span><span class="card-badge">Diff</span></summary>

````diff
diff --git a/activation/tests/test_basic_agent_ac_training.py b/activation/tests/test_basic_agent_ac_training.py
--- a/activation/tests/test_basic_agent_ac_training.py
+++ b/activation/tests/test_basic_agent_ac_training.py
@@ -32,7 +32,7 @@
 AC_NAME, SIDE_LORA, TARGET_LORA = "ac_dev", "ac_side", "ac_target"
 QA_MODEL_NAME = "RedHatAI/Qwen3.5-9B-FP8-dynamic"
 ITEMS, HELD = 48, 8
-CACHING_ID = "ac_training_test_3a1_9b"
+CACHING_ID = "ac_training_test_codesign_v3_9b"
 
 
 def _harness(with_study: bool) -> HarnessRuntime:
@@ -46,7 +46,16 @@
 
 
 def _trainer(harness: HarnessRuntime) -> ActivationContextTrainer:
-    return ActivationContextTrainer(harness, ActivationContextTrainingConfig(examples_per_update=8, num_epochs=2))
+    return ActivationContextTrainer(harness, ActivationContextTrainingConfig(loss_kind="kl", examples_per_update=8, num_epochs=2))
+
+
+def _print_item(item: ActivationContextTrainingItem, tokenizer) -> None:
+    messages = item.in_context_messages or item.ac_messages
+    question = next((message.get("content", "") for message in reversed(messages) if message.get("role") == "user"), "")
+    if isinstance(question, list):
+        question = "".join(piece.get("text", "") for piece in question if isinstance(piece, dict))
+    answer = "\n".join(tokenizer.decode(ids, skip_special_tokens=False) for ids in item.assistant_token_ids)
+    print(f"  [{item.item_id}] Q: {str(question)[-160:]!r} A: {answer[:80]!r}; assistant spans={len(item.assistant_token_ids)}")
 
 
 def _run(harness: HarnessRuntime, items: list[ActivationContextTrainingItem], name: str) -> None:
@@ -98,7 +107,10 @@
     math = NemotronMathDataset.load(harness, max_examples=ITEMS, subset="tir", excluded_problems=aime_problem_statements(), min_chars=40_000, min_tool_calls=1)
     generator = ActivationContextStudyGenerator(harness, AC_NAME)
     items = generator.generate_compaction_samples(swe.dataset_id, ITEMS // 2, seed=0) + generator.generate_compaction_samples(math.dataset_id, ITEMS // 2, seed=0)
-    print(f"\n{len(items)} compaction items; depths {sorted({item.info['depth'] for item in items})}; mid-turn cuts {sum(bool(item.teacher_partial_text) for item in items)}")
+    print(f"\n{len(items)} compaction items; depths {sorted({item.info['depth'] for item in items})}; "
+          f"assistant spans {sum(len(item.assistant_token_ids) for item in items)}; "
+          f"assistant targets {sum(len(ids) for item in items for ids in item.assistant_token_ids)}")
+    assert all(item.assistant_token_ids and all(item.assistant_token_ids) for item in items)
     _run(harness, items, "compaction")
 
 
@@ -115,7 +127,7 @@
     harness.loaded_models[QA_MODEL_NAME].engine_to_device(FREE_DEVICE)
     print(f"\n{len(items)} traj_qa items; distractors {sorted({item.info['distractors'] for item in items})}")
     for item in items[:3]:
-        print(f"  [{item.item_id}] Q: {item.ac_prefix[-1]['content'][-1]['text'][-160:]!r} A: {item.completion_text[:80]!r}")
+        _print_item(item, harness.loaded_models[TARGET_NAME].tokenizer)
     _run(harness, items, "traj_qa")
 
 
@@ -130,5 +142,5 @@
     harness.loaded_models[QA_MODEL_NAME].engine_to_device(FREE_DEVICE)
     print(f"\n{len(items)} rag_qa items; passages per item {sorted({item.info['passages'] for item in items})}")
     for item in items[:3]:
-        print(f"  [{item.item_id}] Q: {item.in_context_prefix[-1]['content'][-160:]!r} A: {item.completion_text[:80]!r}")
+        _print_item(item, harness.loaded_models[TARGET_NAME].tokenizer)
     _run(harness, items, "rag_qa")
````

</details>

<details class="card" data-tressoir-markdown open>
  <summary><span class="card-title"><a href="codesign/activation/tests/test_basic_ac_self_distillation.py">activation/tests/test_basic_ac_self_distillation.py</a></span><span class="card-oneliner">Four easy HotpotQA tasks through the custom program, KL/SFT training, paired reload and AC rollout/cache pipeline.</span><span class="card-badge">Diff</span></summary>

````diff
diff --git a/activation/tests/test_basic_ac_self_distillation.py b/activation/tests/test_basic_ac_self_distillation.py
new file mode 100644
--- /dev/null
+++ b/activation/tests/test_basic_ac_self_distillation.py
@@ -0,0 +1,286 @@
+"""Four real HotpotQA problems through rollout, AC self-distillation and AC rollout.
+
+The custom program retrieves evidence, asks one real solver subagent, then lets the
+main agent answer. No mocks, synthetic trajectories or gold answers enter the program.
+Each loss mode runs one epoch with two optimizer updates. Scores are printed as
+same-problem diagnostics; this small test does not assert a quality improvement.
+
+    uv run pytest activation/tests/test_basic_ac_self_distillation.py --gpu --slow -s
+
+Both loss modes use the paired-history, loss_kind and folder-local row APIs.
+"""
+import json
+import math
+import uuid
+from pathlib import Path
+
+import pytest
+import torch
+
+from activation.ac_model import (
+    ActivationContextModelConfig,
+    ActivationContextStudyGenerator,
+    ActivationContextTrainer,
+    ActivationContextTrainingConfig,
+    ActivationContextTrainingReporter,
+)
+from activation.agent import AgentConfig, AgenticProgram, AgentRunResult, RolloutReporter, SemanticSearchTool
+from activation.agent.agent_utils import load_ac_rows
+from activation.common.data_syncing import resolve_path
+from activation.dataset.loaders import HotpotQADataset
+from activation.harness import FREE_DEVICE, HarnessRuntime, HarnessRuntimeConfig, ModelConfig
+
+SIDE_NAME, SIDE_ID = "self_distill_side", "Qwen/Qwen3.5-0.8B"
+READER_NAME, READER_ID = "self_distill_reader", "Qwen/Qwen3.5-4B"
+PROBLEMS = 4
+
+
+class RetrieveThenSolveProgram(AgenticProgram):
+    """Actual retrieval and a solver precede the main agent's own recorded answer."""
+    name = "retrieve_then_solve"
+
+    def execute(self) -> AgentRunResult:
+        question = self.agent.agent_config.dataset_task.task_datum["question"]
+        evidence = self.agent.run_tool("semantic_search", query=question, top_k=2)
+        if evidence.is_error:
+            raise RuntimeError(evidence.output)
+        solver = self.agent.run_subagent(
+            f"Find evidence for this question using semantic_search, then submit a short answer: {question}",
+            max_duration=60.0,
+        )
+        self.agent.augment_context([
+            evidence,
+            solver,
+            "Use the retrieved evidence and check the solver's suggestion. You may search again. "
+            "Submit only the short answer to the original question.",
+        ])
+        return self.agent.run()
+
+
+def _harness(ac_name: str, side_lora: str, reader_lora: str) -> HarnessRuntime:
+    harness = HarnessRuntime(HarnessRuntimeConfig(
+        model_configs={
+            SIDE_NAME: ModelConfig(SIDE_NAME, SIDE_ID),
+            READER_NAME: ModelConfig(READER_NAME, READER_ID),
+        },
+        agent_max_concurrent=2,
+    ))
+    harness.module_manager.register_lora(reader_lora, READER_NAME, rank=16)
+    harness.module_manager.register_ac_model(ActivationContextModelConfig(
+        ac_name, SIDE_NAME, side_lora, READER_NAME, reader_lora,
+        side_lora_rank=16,
+    ))
+    return harness
+
+
+def _tasks(harness: HarnessRuntime):
+    # Select by source difficulty and question length, never by model success.
+    # The bounded source prefix supplies ordinary distractor documents as well.
+    dataset = HotpotQADataset.load(harness, max_examples=128, split="train", seed=0)
+    harness.dataset_manager.register_dataset(dataset)
+    harness.dataset_manager.build_bm25_indexes()
+    easy = [task for task in dataset.scorable_tasks.values() if task.task_datum.get("level") == "easy"]
+    assert len(easy) >= PROBLEMS, f"expected four easy questions in the source prefix, found {len(easy)}"
+    tasks = sorted(easy, key=lambda task: (len(task.task_datum["question"]), task.task_id))[:PROBLEMS]
+    for task in tasks:
+        print(f"HotpotQA {task.task_id}: {task.task_datum['question']}")
+    return tasks
+
+
+def _configs(tasks, ac_name: str | None, reader_lora: str | None) -> list[AgentConfig]:
+    return [AgentConfig(
+        agent_name=f"hotpot_{index}",
+        system_prompt=(
+            "Answer the question from the Wikipedia corpus using semantic_search. "
+            "Check the evidence and call submit_answer with the short answer only."
+        ),
+        user_prompt=task.agent_prompt,
+        dataset_task=task,
+        model_name=READER_NAME,
+        lora_name=reader_lora,
+        ac_model_name=ac_name,
+        tools={"semantic_search": (SemanticSearchTool, {"dataset_id": task.dataset_id, "max_chars": 1200}), "subagent": None},
+        agentic_program=(RetrieveThenSolveProgram, {}),
+        metadata={"program": RetrieveThenSolveProgram.name, "benchmark": "hotpotqa"},
+        call_kwargs={"sampling_params": {"max_tokens": 1024, "temperature": 0.0}},
+        max_turns=4,
+        max_duration=120.0,
+        max_tool_errors=2,
+        compaction_threshold_tokens=16_384,
+        absolute_trajectory_cap=32_000,
+    ) for index, task in enumerate(tasks)]
+
+
+def _rollouts(harness, configs, caching_id: str, report_folder: Path, seed: int) -> list[AgentRunResult]:
+    reporter = RolloutReporter(str(report_folder), title=f"HotpotQA: {caching_id}")
+    try:
+        runs = harness.rollout_manager.perform_single_rollouts(
+            configs, seed=seed, caching_id=caching_id, perform_scoring=True, reporter=reporter,
+        )
+    finally:
+        reporter.finish()
+    assert len(runs) == PROBLEMS
+    for run in runs:
+        assert run.finish_reason != "error", run.score_feedback
+        assert len(run.subagent_results) == 1
+        assert run.subagent_results[0].finish_reason != "error", run.subagent_results[0].score_feedback
+        assert run.score is not None and math.isfinite(run.score)
+        print(f"{run.agent_config.agent_name}: score={run.score:.3f}, answer={run.answer!r}, "
+              f"finish={run.finish_reason}, rows={run.num_ac_rows}")
+    return runs
+
+
+def _segments(run: AgentRunResult):
+    for segment in [*run.compactions, run]:
+        yield segment
+        for child in segment.subagent_results:
+            yield from _segments(child)
+
+
+def _spans(runs: list[AgentRunResult]):
+    for root in runs:
+        for segment in _segments(root):
+            yield from segment.prompt_ac_spans
+            for step in segment.trajectory:
+                yield from step.get("ac_spans", [])
+
+
+def _assert_coverage(runs, items) -> None:
+    recorded = sum(
+        len(step.get("token_ids") or [])
+        for root in runs for segment in _segments(root) for step in segment.trajectory
+        if step["role"] == "assistant"
+    )
+    selected = sum(len(ids) for item in items for ids in item.assistant_token_ids)
+    assert selected == recorded > 0, (selected, recorded)
+    assert all(item.weight == 1.0 for item in items)
+
+
+def _assert_eval(summary, count: int) -> None:
+    assert summary["items"] == count > 0
+    assert summary["dropped_too_long"] == 0
+    assert summary["loss"] is not None and math.isfinite(summary["loss"])
+
+
+def _check_saved_rows(runs, caching_id: str, width: int) -> None:
+    folder = resolve_path(f"ROLLOUTS/{caching_id}", create=False)
+    assert (folder / "rollouts.jsonl").is_file()
+    assert (folder / "saved_tensors").is_dir()
+    spans = list(_spans(runs))
+    assert spans, "the AC rollouts must produce actual captured rows"
+    for span in spans:
+        assert "rows" not in span, "successfully saved results must release their tensor fields"
+        path = Path(span["row_path"])
+        assert path.is_absolute() and path.is_file()
+        assert path.parent == (folder / "saved_tensors").resolve()
+    rows = load_ac_rows(spans[0])
+    assert rows.device.type == "cpu" and not rows.requires_grad
+    assert rows.shape == (spans[0]["length"], width)
+    assert torch.isfinite(rows).all()
+
+
+def _release(harness, ac_name: str) -> None:
+    for loaded in harness.loaded_models.values():
+        loaded.engine_to_device(FREE_DEVICE)
+    ac_model = harness.module_manager.get_ac_model(ac_name)
+    ac_model.release()
+    for model_name, lora_name in (
+        (READER_NAME, ac_model.config.target_model_lora_name),
+        (SIDE_NAME, ac_model.config.base_side_model_lora_name),
+    ):
+        harness.module_manager.free_lora(model_name, lora_name)
+    for loaded in harness.loaded_models.values():
+        loaded.model_to_device(FREE_DEVICE)
+
+
+@pytest.mark.gpu
+@pytest.mark.slow
+@pytest.mark.parametrize("loss_kind", ["kl", "sft"])
+def test_basic_ac_self_distillation(loss_kind: str) -> None:
+    # A new namespace prevents cached rollouts/checkpoints from hiding this run.
+    run_id = f"hotpot_ac_{loss_kind}_{uuid.uuid4().hex[:10]}"
+    ac_name, side_lora, reader_lora = run_id, f"{run_id}_side", f"{run_id}_reader"
+    folder = resolve_path(f"AC_SELF_DISTILLATION_TEST/{run_id}")
+    harness = _harness(ac_name, side_lora, reader_lora)
+    training_reporter = None
+    try:
+        tasks = _tasks(harness)
+        before_configs = _configs(tasks, ac_name=None, reader_lora=None)
+        before_runs = _rollouts(harness, before_configs, f"{run_id}_text", folder / "text_rollouts", seed=0)
+        assert all(run.num_ac_rows == 0 for run in before_runs)
+        assert all(len(run.injected_input) >= 2 for run in before_runs)
+
+        config = ActivationContextTrainingConfig(
+            loss_kind=loss_kind,
+            num_epochs=1,
+            updates_per_epoch=2,
+            reporting_interval=1.0,
+            completion_samples=1,
+            completion_max_new_tokens=32,
+            logits_chunk_tokens=128,
+            seed=0,
+        )
+        study = ActivationContextStudyGenerator(harness, ac_name)
+        # Explicitly include all four for this execution example in both modes.
+        # The SFT default score_1 selection is tested privately; scores stay real.
+        items = study.items_from_run_results(before_runs, loss_kind=loss_kind, selection="all")
+        _assert_coverage(before_runs, items)
+        assert len(items) >= PROBLEMS
+        trainer = ActivationContextTrainer(harness, config)
+        training_reporter = ActivationContextTrainingReporter(str(folder / "training"), f"HotpotQA self-distillation: {loss_kind}")
+        before = trainer.eval(ac_name, items, reporter=training_reporter, reporting_name="before")
+        _assert_eval(before, len(items))
+        stats = trainer.train(ac_name, items, reporting_data=items, reporter=training_reporter)
+        after = stats.reporting
+        _assert_eval(after, len(items))
+        assert stats.steps == 2 and len(stats.epochs) == 1
+        assert stats.loss_kind == loss_kind and all(math.isfinite(value) for value in stats.loss)
+        assert stats.checkpoint_path and (Path(stats.checkpoint_path) / "ac_modules.pt").is_file()
+        assert stats.target_lora_checkpoint_path and Path(stats.target_lora_checkpoint_path).is_dir()
+        if loss_kind == "sft":
+            assert stats.teacher_tokens == 0 and all(row["teacher_tokens"] == 0 for row in after["per_item"])
+            assert after["kl"] is None and after["agreement"] is None
+        print(json.dumps(stats.summarize(), indent=2))
+
+        ac_model = harness.module_manager.get_ac_model(ac_name)
+        completed = ac_model.load_completed_epoch()
+        assert resolve_path(completed["ac_checkpoint"], create=False).resolve() == Path(stats.checkpoint_path).resolve()
+        reloaded = ActivationContextTrainer(harness, config).eval(ac_name, items)
+        _assert_eval(reloaded, len(items))
+        assert math.isclose(reloaded["loss"], after["loss"], rel_tol=5e-3, abs_tol=5e-3)
+        harness.module_manager.exchange_lora(READER_NAME, reader_lora)
+
+        ac_configs = _configs(tasks, ac_name=ac_name, reader_lora=reader_lora)
+        ac_cache = f"{run_id}_ac"
+        ac_runs = _rollouts(harness, ac_configs, ac_cache, folder / "ac_rollouts", seed=1)
+        assert all(run.ac_model_name == ac_name and run.lora_name == reader_lora for run in ac_runs)
+        assert all(len(run.prompt_ac_spans) >= 2 for run in ac_runs), "program should inject multiple actual AC parts"
+        _check_saved_rows(ac_runs, ac_cache, ac_model.d_target)
+
+        cached = harness.rollout_manager.perform_single_rollouts(ac_configs, seed=1, caching_id=ac_cache, perform_scoring=True)
+        assert len(cached) == PROBLEMS
+        assert [run.answer for run in cached] == [run.answer for run in ac_runs]
+        _check_saved_rows(cached, ac_cache, ac_model.d_target)
+        # Native AC histories also convert to differentiable training data.
+        ac_items = study.items_from_run_results(cached, loss_kind=loss_kind, selection="all")
+        _assert_coverage(cached, ac_items)
+
+        summary = {
+            "loss_kind": loss_kind,
+            "task_ids": [task.task_id for task in tasks],
+            "text_scores": [run.score for run in before_runs],
+            "ac_scores": [run.score for run in ac_runs],
+            "training_items": len(items),
+            "ac_history_items": len(ac_items),
+            "before_loss": before["loss"],
+            "after_loss": after["loss"],
+            "reloaded_loss": reloaded["loss"],
+            "checkpoint": completed,
+            "quality_scope": "same four training questions; diagnostic only",
+        }
+        (folder / "summary.json").write_text(json.dumps(summary, indent=2))
+        print(json.dumps(summary, indent=2))
+    finally:
+        if training_reporter is not None:
+            training_reporter.finish()
+        _release(harness, ac_name)
````

</details>

<details class="card" data-tressoir-markdown open>
  <summary><span class="card-title"><a href="codesign/activation/tests/test_basic_agent_ac.py">activation/tests/test_basic_agent_ac.py</a></span><span class="card-oneliner">Approved minimal base_dir migration for the existing serialization round trip.</span><span class="card-badge">Diff</span></summary>

````diff
diff --git a/activation/tests/test_basic_agent_ac.py b/activation/tests/test_basic_agent_ac.py
--- a/activation/tests/test_basic_agent_ac.py
+++ b/activation/tests/test_basic_agent_ac.py
@@ -175,8 +175,9 @@
         _probe_engine(agent, "search")
 
         # --- resume: the serialized record rebuilds the same prefix, spans and rows.
-        data = json.loads(json.dumps(agent.run_results.serialize()))
-        resumed = Agent.resume_from_run_result(harness, AgentRunResult.deserialize(data, harness))
+        base_dir = resolve_path("AGENT_AC_TEST/roundtrip")
+        data = json.loads(json.dumps(agent.run_results.serialize(base_dir=base_dir)))
+        resumed = Agent.resume_from_run_result(harness, AgentRunResult.deserialize(data, harness, base_dir=base_dir))
         assert resumed.prefix == agent.prefix and resumed.spans == agent.spans
         assert [tuple(r.shape) for r in resumed.rows] == [tuple(r.shape) for r in agent.rows]
         assert all(torch.equal(a, b) for a, b in zip(resumed.rows, agent.rows))
````

</details>

