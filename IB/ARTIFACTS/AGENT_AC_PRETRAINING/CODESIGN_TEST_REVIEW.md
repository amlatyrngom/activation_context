# AC training test review — source

User-requested complete mainline test proposals, not applied product tests. The source/projection below embeds both native files verbatim; update all three representations together. Any private test code belongs in IB/TMP and is not part of this artifact. The behavior matrix defines the proof obligations for the detailed plan and its required subagent review. No new utilities module is permitted.

Review dependencies: the planned loss_kind/paired-history AC trainer, folder-local row serialization/release and load_ac_rows in existing agent_utils. Directly audited current APIs: HotpotQADataset.load, AgenticProgram.execute, Agent.run_tool/run_subagent/augment_context/run, module registration, load_completed_epoch and exchange_lora. Both loss modes exercise the same four source questions without outcome-based selection or a quality-improvement assertion. The self-distillation file demonstrates AC training; the real frozen-row AgentTrainer numerical/gradient proof is private.


## Interface acceptance and next-phase boundary — 2026-09-15

The user accepted the interfaces and both full test proposals in chat on 2026-09-15. Their next green light is intended to cover details → independent review → implementation → re-review → acknowledgement with handoff report in one continuous phase. Resolve ordinary technical findings within the agreed intent; return for input when a finding requires a user decision or scope/public-behavior change. The green light itself has not yet been given.

## Human-facing projection

# AC training — full proposed test files

The user accepted these complete proposed versions of the two mainline test files in chat on 2026-09-15. These are the accepted proposal snapshots. Product implementation and validation are tracked in [the implementation plan](CODESIGN_IMPLEMENTATION_PLAN.tressoir.md); [the final handoff](CODESIGN_ROUND.tressoir.md) carries both final full files and the actual CPU/GPU results. No private test code is shown.

## What the mainline files demonstrate

| File | Real-data behavior |
| --- | --- |
| [test_basic_agent_ac_training.py](./interface_tests/test_basic_agent_ac_training.py) | Keeps the existing public compaction, trajectory-QA and RAG-QA examples; adopts paired histories/all assistant targets and retains two-epoch training, held comparisons and paired saves. |
| [test_basic_ac_self_distillation.py](./interface_tests/test_basic_ac_self_distillation.py) | Four short HotpotQA questions labeled easy, actual corpus/search and a custom retrieval-plus-subagent program; text rollouts → complete items → one epoch/two updates of KL or SFT → paired reload → reader exchange → fresh AC rollouts → folder-local row/cache replay and native-AC harvesting. |

Only these two test files belong in this work's mainline test delta. Both files are complete, with imports, setup, helpers, assertions and cleanup. Mainline examples use actual models/tools and contain no mocks or synthetic trajectories. Storage helpers live in the existing `activation/agent/agent_utils.py`; no new utilities file is planned.

The self-distillation example runs both losses with an explicit `selection="all"` so it exercises training on all four problems even if the model answers all of them incorrectly. It never substitutes gold answers or fabricated scores. Default successful-only SFT selection is checked privately. The post-training scores use the same four questions and are labeled diagnostics. Neither test asserts that brief training improves answer quality.

## Private proof of behavior

We can establish the execution and gradient contracts with short deterministic checks. Synthetic data makes boundary cases controllable. Mocks are used for forbidden calls, external boundaries and failure injection; losses, backpropagation and parameter updates run through actual code with tiny real models and independent references.

| Behavior to establish | Private proof, without long training |
| --- | --- |
| Frozen-row AgentTrainer uses the actual observation | Multi-turn synthetic records with several distinct rows; make every encoder entry point fail if called, then run the real trainer on a tiny reader. Check actual input embeddings, unchanged encoder/row values and a reader-LoRA update. Perturb one row and verify the expected causal effect on later predictions. Repeat with saved/reloaded rows and with the encoder absent. |
| Existing agent objective and reference logprobs | Compute an independent per-example token-mean clipped surrogate for positive and negative weights, including clipping boundaries; compare loss, gradients and one optimizer update. Check recorded old logprobs and the no-grad recomputation path use the same rows. Do not mock the loss or optimizer. |
| Packing, padding and masks | Compare separate sequences against padded/packed execution on a tiny real model, including spacer tokens, multiple row spans, thinking-strip shifts and compaction resets. Require matching selected logits/loss/gradients and no cross-sequence or future-context influence. Tool results and AC input rows have zero target loss. |
| Joint AC KL/SFT learning | Tiny real AC encoder/reader and distinct LoRAs, with an asymmetric fixture. Compare KL/CE to independent logits-based references. Require gradients and updates in each intended parameter group, including recursive parts; base weights remain frozen. SFT has no teacher or drift forward even through baseline eval, samples and reporting. |
| Shared base and adapter switching | One registered tiny PEFT base with two adapters; compare outputs and gradients with equivalent separate-base weights, checkpointing off/on, nested parts and reversed call order. Verify adapter activation/trainability restoration, alias-aware lifecycle handling and same/effectively-colliding adapter-name rejection. |
| Whole-history items and public windows | Different-length prefixes with matching retained roles/calls/assistant IDs; offsets checked against an independent token ledger. Cover all assistant outputs exactly once across roots/children/resets, incomplete recorded tails, multiple raw channels and origin-disjoint splits. Public windows count all roles within 8192, exceed the old 512 cap when appropriate, and end complete turns; synthetic calls are scored and tool replies masked. No recorded-run truncation, fabricated ending or gold leakage. |
| Selection and explicit errors | All three score filters, KL/SFT defaults, root-to-child inheritance, unit/custom/zero weights, nonfinite/negative weights and empty selections. Malformed alignment, incompatible tokenizer/row shape/dtype, missing raw data and over-capacity examples must fail explicitly before updates. SFT does not validate or allocate an unused teacher view. |
| Folder-local persistence and memory ownership | Real tiny tensor files: exact dtype/byte round trips, deduplication, moving a folder, cross-folder exports and lazy loading. Inject write/append failures and delays; failed publication keeps captured data, success releases it through all retained result aliases, and a slow writer holds its rollout slot. Cover concurrent saves, missing/corrupt paths, no-copy capture, inspection reports, disabled caching and deliberately unsaved deadline results. |
| Optimizers, teacher cache, checkpoints and reports | Consecutive calls retain intended moments/clocks; explicit phase reset discards receiving moments while preserving weights/global clocks. KL cache identity includes target positions and expires per call; evaluation stays exact. Verify SFT's zero teacher work and honest CE labels, mixed-mode reporter rejection, matched paired reload, partial-checkpoint rejection and reader exchange ordering. |

Most checks run on CPU with tiny sequences and one or two updates. A bounded GPU serving/precision check covers the actual engine and relevant attention/recomputation kernels where CPU tests cannot. These checks establish mechanics and numerical contracts; they do not establish useful compression or a policy-quality gain. Long learning curves are separate experiments.

## Full proposed files

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">activation/tests/test_basic_agent_ac_training.py</span>
    <span class="card-oneliner">Complete proposed file, 146 lines.</span>
    <span class="card-badge">Review</span>
  </summary>

[Open the native file](./interface_tests/test_basic_agent_ac_training.py).

`activation/tests/test_basic_agent_ac_training.py`

```python
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
```

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">activation/tests/test_basic_ac_self_distillation.py</span>
    <span class="card-oneliner">Complete proposed file, 286 lines.</span>
    <span class="card-badge">Review</span>
  </summary>

[Open the native file](./interface_tests/test_basic_ac_self_distillation.py).

`activation/tests/test_basic_ac_self_distillation.py`

```python
"""Four real HotpotQA problems through rollout, AC self-distillation and AC rollout.

The custom program retrieves evidence, asks one real solver subagent, then lets the
main agent answer. No mocks, synthetic trajectories or gold answers enter the program.
Each loss mode runs one epoch with two optimizer updates. Scores are printed as
same-problem diagnostics; this small test does not assert a quality improvement.

    uv run pytest activation/tests/test_basic_ac_self_distillation.py --gpu --slow -s

This file targets the planned paired-history, loss_kind and folder-local row APIs.
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
        tools={"semantic_search": (SemanticSearchTool, {"dataset_id": task.dataset_id, "max_chars": 1200})},
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
```

</details>

## Validation status

Static validation passed: both complete files parse, native files match the embedded proposals, all three artifacts pass their document checker, and source/projection pairs, links, control keys, planned constructor arguments and the two-file/no-mock boundary agree. Current imports and call sites were audited against existing source plus declared interface additions; tracked product files and prior answers were preserved. Direct structured review: pass for these proposals. Model execution, GPU checks and the proposed private behavior suite have not run; they follow implementation. The full-file review checkpoint is accepted; the combined details/review/implementation/handoff phase awaits the user’s green light. Both final complete files will be shown again with the implementation handoff.
