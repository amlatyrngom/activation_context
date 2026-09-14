"""
Agent training teacher study: which benchmarks the oracle runs, under which meta-agent prompts, and how a rollout
campaign over them is generated and executed. Replaces the suite and framing modules.

- Benchmarks (`BENCHMARKS`): loader, task kind, whether semantic_search is offered, whether it is scored.
- Teacher kinds (`AgentTeacherKind`): base (the task as is), sequential multi-agent, parallel multi-agent. The prompt
  templates live in `agent_training_teacher_study_prompts.META_AGENT_PROMPT_TEMPLATES`, keyed by (task kind, teacher kind).
- `AgentTrainingStudy(harness, dataset_choice_weights, teacher_kind_weights)`: `setup_reusable_artifacts` loads the
  datasets in parallel, builds the sandbox image recipe and the search indexes once; `generate_rollout_campaign_tasks`
  samples (benchmark, task, teacher kind, template) into agent configs, every config labelled in `metadata`;
  `execute_rollout_campaign_tasks` runs them through the rollout manager with the live rollout report (which summarizes
  per benchmark and teacher kind) and returns the results. The canonical entry point is
  `activation.bench.canonical_training.generate_agent_teacher_trajectories`.
"""
from __future__ import annotations

import json
import random
import time
import typing as t
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from activation.agent import AgentConfig, SemanticSearchTool
from activation.agent.agent_config import AgentRunResult, _resolve_class_spec
from activation.agent.rollout_caching import RedoPolicy
from activation.agent.rollout_reporter import RolloutReporter
from activation.common.data_syncing import resolve_path
from activation.dataset import DatasetTask, DatasetTaskKind, DatasetTaskMetricsKind, LoadedDataset, ANSWER_RULES, bare_prompt
from activation.dataset.environments import default_dockerfile
from activation.dataset.loaders import (BrightDataset, DapoMathDataset, DeepMathDataset, HotpotQADataset, LcaBugLocalizationDataset, LoftDataset,
                                        LoongDataset, MuSiQueDataset, NarrativeQADataset, QualityDataset)

from .agent_training_teacher_study_prompts import META_AGENT_PROMPT_TEMPLATES, SYSTEM_PROMPT, AgentTeacherKind, render_template

if t.TYPE_CHECKING:
    from activation.harness import HarnessRuntime

ORACLE_MODEL_NAME = "oracle"
ORACLE_MODEL_ID = "unsloth/Qwen3.8-27B-NVFP4"
NON_THINKING_CALL_KWARGS = {"sampling_params": {"max_tokens": 4096}}   # the harness profile (temperature 0.7, top_p 0.8, presence 1.5)
REASONING_EFFORTS = ("low", "medium", "xhigh")                          # Qwen3.8 chat template values; "high" is an alias of xhigh there
THINKING_SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5, "max_tokens": 8192}


def oracle_call_kwargs(reasoning: str | None) -> dict:
    """Per-run chat kwargs: non-thinking, or thinking at the given reasoning effort (its own sampling and a longer turn)."""
    if reasoning is None:
        return NON_THINKING_CALL_KWARGS
    if reasoning not in REASONING_EFFORTS:
        raise ValueError(f"reasoning must be one of {REASONING_EFFORTS}, not {reasoning!r}")
    return {"sampling_params": dict(THINKING_SAMPLING), "chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": reasoning}}


class BrightTasks:
    """BRIGHT labeled queries as unscored research tasks (metric UNSCORED: score() is None)."""

    @classmethod
    def load(cls, harness: "HarnessRuntime", max_examples: int | None, seed: int = 0, domain: str = "biology",
             max_corpus_documents: int | None = 2000) -> LoadedDataset:
        loaded = BrightDataset.load(harness, max_examples, domain=domain, max_corpus_documents=max_corpus_documents)
        for example_id, example in loaded.labeled_qa_examples.items():
            task = "Research question (the corpus behind semantic_search holds the passages that answer it):\n\n" + example.query
            loaded.scorable_tasks[example_id] = DatasetTask(
                task_id=example_id, dataset_id=loaded.dataset_id,
                task_datum={"query": example.query, "positive_doc_ids": list(example.positive_doc_ids or [])},
                reference_metrics_kind=DatasetTaskMetricsKind.UNSCORED, gold_answer=(example.gold_answers or [""])[0],
                agent_prompt=bare_prompt(task, ANSWER_RULES["search"]), task_kind=DatasetTaskKind.SEMANTIC_SEARCH,
            )
        return loaded


@dataclass(frozen=True)
class StudyBenchmark:
    name: str
    loader: t.Any                 # a loader class with load(harness, max_examples, seed=..., ...)
    task_kind: DatasetTaskKind
    loader_kwargs: dict = field(default_factory=dict)
    scored: bool = True
    searchable: bool = True       # the semantic_search tool is offered over the dataset's documents


BENCHMARKS: dict[str, StudyBenchmark] = {b.name: b for b in (
    StudyBenchmark("musique", MuSiQueDataset, DatasetTaskKind.SEMANTIC_SEARCH),
    StudyBenchmark("hotpotqa", HotpotQADataset, DatasetTaskKind.SEMANTIC_SEARCH),                                 # kept loadable; weight 0 in the campaign
    StudyBenchmark("dapo_math", DapoMathDataset, DatasetTaskKind.MATH, searchable=False),
    StudyBenchmark("deepmath", DeepMathDataset, DatasetTaskKind.MATH, {"min_difficulty": 6}, searchable=False),
    StudyBenchmark("lca_bug_localization", LcaBugLocalizationDataset, DatasetTaskKind.CODE_SEARCH, {"index_files_per_repo": 0, "config": "py"}, searchable=False),  # ripgrep, not a passage index
    StudyBenchmark("lca_bug_localization_java", LcaBugLocalizationDataset, DatasetTaskKind.CODE_SEARCH, {"index_files_per_repo": 0, "config": "java"}, searchable=False),
    StudyBenchmark("lca_bug_localization_kt", LcaBugLocalizationDataset, DatasetTaskKind.CODE_SEARCH, {"index_files_per_repo": 0, "config": "kt"}, searchable=False),
    StudyBenchmark("quality", QualityDataset, DatasetTaskKind.FILE_SEARCH, {"questions_per_article": 50}),        # kept loadable; weight 0 in the campaign (5k-token passages)
    StudyBenchmark("narrativeqa", NarrativeQADataset, DatasetTaskKind.FILE_SEARCH, {"questions_per_document": 50}),  # every question: 3,461 over 115 documents
    StudyBenchmark("loft", LoftDataset, DatasetTaskKind.FILE_SEARCH),                                              # passage collections on disk, 128k and 1m haystacks
    StudyBenchmark("loong", LoongDataset, DatasetTaskKind.FILE_SEARCH, searchable=False),                          # English multi-document sets, unscored
    StudyBenchmark("bright", BrightTasks, DatasetTaskKind.SEMANTIC_SEARCH, {"domain": "biology"}, scored=False),
)}

# Campaign defaults: weights over benchmarks (task counts) and over teacher kinds per task kind.
DEFAULT_DATASET_WEIGHTS: dict[str, float] = {                      # 12,000 draws: each share stays under its pool except loong and bright, which exhaust and redistribute
    "musique": 0.20, "dapo_math": 0.08, "deepmath": 0.19,
    "lca_bug_localization": 0.14, "lca_bug_localization_java": 0.08, "lca_bug_localization_kt": 0.03,
    "narrativeqa": 0.12, "loft": 0.09, "loong": 0.06, "bright": 0.01,
}
DEFAULT_TEACHER_WEIGHTS: dict[tuple[DatasetTaskKind, AgentTeacherKind], float] = {
    (kind, teacher): weight
    for kind in DatasetTaskKind
    for teacher, weight in ((AgentTeacherKind.BASE, 0.2), (AgentTeacherKind.SEQUENTIAL_MULTI_AGENT, 0.4), (AgentTeacherKind.PARALLEL_MULTI_AGENT, 0.4))
}


def template_id(task_kind: DatasetTaskKind, teacher_kind: AgentTeacherKind, index: int) -> str:
    return f"{task_kind}/{teacher_kind}/{index}"


class AgentTrainingStudy:
    def __init__(
        self,
        harness: "HarnessRuntime",
        dataset_choice_weights: dict[str, float] | None = None,                                  # benchmark -> share of tasks
        teacher_kind_weights: dict[tuple[DatasetTaskKind, AgentTeacherKind], float] | None = None,  # (task kind, teacher kind) -> weight
        model_name: str = ORACLE_MODEL_NAME,
        reasoning: str | None = "medium",
        max_turns: int = 100,
        max_duration: float = 1800.0,
        compaction_threshold_tokens: int = 32_768,
        max_tool_errors: int = 8,
        env_dockerfile_path: str | None = None,                                                  # None: default_dockerfile()
        system_prompt: str = SYSTEM_PROMPT,
    ):
        self.harness = harness
        self.dataset_choice_weights = {name: float(w) for name, w in (dataset_choice_weights or DEFAULT_DATASET_WEIGHTS).items() if w > 0}
        unknown = set(self.dataset_choice_weights) - set(BENCHMARKS)
        if unknown:
            raise KeyError(f"unknown benchmarks {sorted(unknown)}; known: {sorted(BENCHMARKS)}")
        self.teacher_kind_weights = dict(teacher_kind_weights or DEFAULT_TEACHER_WEIGHTS)
        self.model_name, self.reasoning = model_name, reasoning
        self.max_turns, self.max_duration, self.compaction_threshold_tokens, self.max_tool_errors = max_turns, max_duration, compaction_threshold_tokens, max_tool_errors
        self.env_dockerfile_path = env_dockerfile_path
        self.system_prompt = system_prompt
        self.datasets: dict[str, LoadedDataset] = {}

    # ------------------------------------------------------------------------------------------------ artifacts
    def setup_reusable_artifacts(self, per_benchmark: int | dict[str, int], seed: int = 0, parallel: int = 4) -> dict[str, LoadedDataset]:
        """
        Everything a campaign reuses, built once: the sandbox image recipe, the datasets (loaded in parallel; the repository
        loader fetches its checkouts into the local cache), their registration, and the search index of every searchable
        benchmark (the index builder and the registry are locked, so this is safe alongside a running pool too).
        """
        if self.env_dockerfile_path is None:
            self.env_dockerfile_path = default_dockerfile()
        counts = {name: (per_benchmark if isinstance(per_benchmark, int) else per_benchmark.get(name, 0)) for name in self.dataset_choice_weights}
        started = time.time()

        def load(name: str) -> tuple[str, LoadedDataset]:
            benchmark = BENCHMARKS[name]
            return name, benchmark.loader.load(self.harness, counts[name], seed=seed, **benchmark.loader_kwargs)

        with ThreadPoolExecutor(max_workers=max(1, parallel)) as pool:
            loaded = dict(pool.map(load, [name for name, count in counts.items() if count > 0]))
        for name, dataset in loaded.items():
            if dataset.dataset_id not in self.harness.dataset_manager.loaded_datasets:
                self.harness.dataset_manager.register_dataset(dataset)
            if BENCHMARKS[name].searchable and dataset.documents:
                self.harness.dataset_manager._get_or_create_index(dataset.dataset_id).build_bm25_index()
        self.datasets.update(loaded)
        print(f"AgentTrainingStudy - artifacts ready in {time.time() - started:.0f}s: " + ", ".join(f"{name} {len(d.scorable_tasks)}" for name, d in loaded.items()), flush=True)
        return loaded

    # ------------------------------------------------------------------------------------------------ campaign tasks
    def generate_rollout_campaign_tasks(self, total: int, seed: int = 0, record: "Path | None" = None, head: "list[dict] | None" = None) -> list[AgentConfig]:
        """
        `total` agent configs: benchmarks by their weights (each benchmark's tasks in a seeded order, none repeated while
        it has unused tasks; a benchmark that runs out is dropped from the draw), teacher kind by the weights of the task's
        kind, template uniformly among the kind's templates. Labels in `config.metadata`: benchmark, task_kind, teacher_kind, template.

        `record` (a JSONL path next to the cache): the draw is written there the first time and replayed after, entry by
        entry, so later runs under the same caching id see the same tasks with the same teachers even when a loader's
        task set drifts (a GitHub checkout that fails one day and succeeds the next reorders a plain shuffle). An entry
        whose task is no longer loaded is skipped with a note. `head`: entries placed first when there is no record yet
        (the rows of an existing cache drawn before records existed); the fresh draw then excludes their tasks.
        """
        if record is not None and record.exists():
            entries = [json.loads(line) for line in record.read_text().splitlines() if line.strip()]
            configs, missing = self._configs_for_entries(entries)
            if missing:
                print(f"AgentTrainingStudy - draw record {record}: {missing} of {len(entries)} entries name tasks that are not loaded now; skipped.", flush=True)
            print(f"AgentTrainingStudy - draw replayed from {record}: {len(configs)} tasks.", flush=True)
            return configs
        rng = random.Random(seed)
        configs, missing = self._configs_for_entries(list(head or []))
        if head and missing:
            print(f"AgentTrainingStudy - {missing} of {len(head)} head entries name tasks that are not loaded now; skipped.", flush=True)
        taken = {(config.dataset_task.dataset_id, config.dataset_task.task_id) for config in configs}
        pools: dict[str, list[DatasetTask]] = {}
        for name, dataset in self.datasets.items():
            tasks = list(dataset.scorable_tasks.values())
            rng.shuffle(tasks)
            pools[name] = [task for task in tasks if (task.dataset_id, task.task_id) not in taken]
        weights = {name: w for name, w in self.dataset_choice_weights.items() if pools.get(name)}
        while len(configs) < total and weights:
            names = list(weights)
            name = rng.choices(names, weights=[weights[n] for n in names])[0]
            task = pools[name].pop()
            if not pools[name]:
                weights.pop(name)
            configs.append(self.config_for_task(name, task, rng))
        if len(configs) < total:
            print(f"AgentTrainingStudy - only {len(configs)} of {total} tasks: every weighted benchmark is exhausted", flush=True)
        if record is not None:
            record.parent.mkdir(parents=True, exist_ok=True)
            record.write_text("".join(json.dumps(self.draw_entry(config)) + "\n" for config in configs))
            print(f"AgentTrainingStudy - draw recorded in {record}: {len(configs)} tasks.", flush=True)
        return configs

    @staticmethod
    def draw_entry(config: AgentConfig) -> dict:
        """The record of one drawn config: benchmark, task, teacher kind and template label (enough to rebuild it)."""
        task = config.dataset_task
        return {"benchmark": config.metadata["benchmark"], "dataset_id": task.dataset_id, "task_id": task.task_id,
                "teacher_kind": config.metadata["teacher_kind"], "template": config.metadata["template"]}

    def _configs_for_entries(self, entries: list[dict]) -> tuple[list[AgentConfig], int]:
        configs: list[AgentConfig] = []
        missing = 0
        for entry in entries:
            dataset = self.datasets.get(entry["benchmark"])
            task = dataset.scorable_tasks.get(entry["task_id"]) if dataset is not None and dataset.dataset_id == entry["dataset_id"] else None
            if task is None:
                missing += 1
                continue
            index = int(str(entry["template"]).rsplit("/", 1)[-1])
            configs.append(self._config_for(entry["benchmark"], task, AgentTeacherKind(entry["teacher_kind"]), index))
        return configs, missing

    def config_for_task(self, benchmark_name: str, task: DatasetTask, rng: random.Random) -> AgentConfig:
        benchmark = BENCHMARKS[benchmark_name]
        task_kind = task.task_kind or benchmark.task_kind
        teacher_kind = self._sample_teacher_kind(task_kind, rng)
        index = rng.randrange(len(META_AGENT_PROMPT_TEMPLATES[(task_kind, teacher_kind)]))
        return self._config_for(benchmark_name, task, teacher_kind, index)

    def _config_for(self, benchmark_name: str, task: DatasetTask, teacher_kind: AgentTeacherKind, index: int) -> AgentConfig:
        benchmark = BENCHMARKS[benchmark_name]
        task_kind = task.task_kind or benchmark.task_kind
        templates = META_AGENT_PROMPT_TEMPLATES[(task_kind, teacher_kind)]
        tools: dict = {}
        if benchmark.searchable:
            tools["semantic_search"] = (SemanticSearchTool, {"dataset_id": task.dataset_id})
        return AgentConfig(
            system_prompt=self.system_prompt, user_prompt=render_template(templates[index], task.agent_prompt), dataset_task=task,
            model_name=self.model_name, call_kwargs=oracle_call_kwargs(self.reasoning), env_dockerfile_path=self.env_dockerfile_path, tools=tools,
            env_setups={name: _resolve_class_spec(spec) for name, spec in task.env_setups.items()},
            max_turns=self.max_turns, max_duration=self.max_duration, max_tool_errors=self.max_tool_errors,
            compaction_threshold_tokens=self.compaction_threshold_tokens,
            metadata={"benchmark": benchmark_name, "task_kind": str(task_kind), "teacher_kind": str(teacher_kind), "template": template_id(task_kind, teacher_kind, index)},
        )

    def _sample_teacher_kind(self, task_kind: DatasetTaskKind, rng: random.Random) -> AgentTeacherKind:
        kinds = [teacher for teacher in AgentTeacherKind if self.teacher_kind_weights.get((task_kind, teacher), 0.0) > 0 and (task_kind, teacher) in META_AGENT_PROMPT_TEMPLATES]
        if not kinds:
            return AgentTeacherKind.BASE
        return rng.choices(kinds, weights=[self.teacher_kind_weights[(task_kind, k)] for k in kinds])[0]

    # ------------------------------------------------------------------------------------------------ execution
    def execute_rollout_campaign_tasks(
        self, configs: list[AgentConfig], seed: int, caching_id: str, report_folder: str, title: str = "Teacher campaign",
        autotune_id: str | None = None, probe_tasks_per_gpu: int = 200, max_wall_seconds: float | None = None, perform_scoring: bool = True,
        redo: "RedoPolicy | None" = None, agents_per_gpu: int | None = None,
    ) -> list[AgentRunResult]:
        """One rollout-manager call over every config (all benchmarks interleaved); the live report goes under the synced report folder."""
        model_id = self.harness.loaded_models[self.model_name].model_config.model_id
        reporter = RolloutReporter(str(resolve_path(report_folder)), title=title,
                                   description=f"{model_id}; {len(configs)} tasks; seed {seed}; cache {caching_id}; reasoning {self.reasoning or 'off'}.")
        kwargs: dict = {"autotune_id": autotune_id, "probe_tasks_per_gpu": probe_tasks_per_gpu, "redo": redo, "agents_per_gpu": agents_per_gpu}
        if max_wall_seconds is not None:
            kwargs["max_wall_seconds"] = max_wall_seconds
        return self.harness.rollout_manager.perform_single_rollouts(configs, seed=seed, caching_id=caching_id, perform_scoring=perform_scoring,
                                                                    reporter=reporter, **kwargs)


def make_harness(model_id: str = ORACLE_MODEL_ID, model_name: str = ORACLE_MODEL_NAME, agent_max_concurrent: int | None = None,
                 agents_per_gpu: int = 40) -> "HarnessRuntime":
    from activation.harness import HarnessRuntime, HarnessRuntimeConfig, ModelConfig
    return HarnessRuntime(HarnessRuntimeConfig(model_configs={model_name: ModelConfig(model_name, model_id)},
                                               agent_max_concurrent=agent_max_concurrent, agent_max_concurrent_per_gpu=agents_per_gpu))
